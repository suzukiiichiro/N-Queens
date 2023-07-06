/**
 *
 * bash版キャリーチェーンのC言語版
 * Ｃ言語版キャリーチェーン並列処理版
 *
 * 簡単な実行
 * bash-3.2$ gcc 08GCC_pthread.c -pthread && ./a.out
 *  
 * 高速な実行 
 * $ gcc -Wall -W -O3 -g -ftrapv -std=c99 -mtune=native -march=native -pthread 08GCC_pthread.c && ./a.out
 *
 * さらにより最適で高速なコンパイルオプション
 * $ gcc -Wshift-negative-value -Wall -W -O3 -g -ftrapv -std=c99 -mtune=native -march=native -pthread 08GCC_pthread.c && ./a.out
 *
 *
 * 今回のテーマ
 * pthreadの実装 
 * THREADフラグを作成して スレッドのオン・オフで動作を確認しながら実装
 * 
 * 構造体初期化メソッドの実装
 *
 *
    誤
    for(unsigned int w=0;w<=(unsigned)(g.size/2)*(g.size-3);++w){
    正
    for(unsigned int w=0;w<(unsigned)(g.size/2)*(g.size-3)+1;++w){
 *
 * これにより以下の部分の末尾に１を加える必要がある

  Local l[(g.size/2)*(g.size-3)+1];
  pthread_t pt[(g.size/2)*(g.size-3)+1];
 *
 * pthreadはマルチプロセスで動くため、これまでの計測方法では、プロセスの合計で計測される。
 * 実行段階から実行終了までの計測は、 gettimeofday(&t1, NULL);を使う必要がある。
 *
 *
 困ったときには以下のＵＲＬがとても参考になります。

 C++ 値渡し、ポインタ渡し、参照渡しを使い分けよう
 https://qiita.com/agate-pris/items/05948b7d33f3e88b8967
 値渡しとポインタ渡し
 https://tmytokai.github.io/open-ed/activity/c-pointer/text06/page01.html
 C言語 値渡しとアドレス渡し
 https://skpme.com/199/
 アドレスとポインタ
 https://yu-nix.com/archives/c-struct-pointer/

  余談
  Google Collaboでは、セッションを最大で２４時間までしか継続できません。
  ときに数時間でセッションが途絶え、処理が終了してしまうことも多いです。
  以下のスクリプトを実行し、ソース部分にカーソルを点滅させた状態であれば、
  セッションを少しは長く継続できるようです。

from pynput.mouse import Controller, Button
import time
mouse = Controller()
while True:
    mouse.click(Button.left, 1)
    print('clicked')
    time.sleep(5)


普通の実行オプション
bash-3.2$ gcc 08GCC_pthread.c -pthread -o 08GCC && ./08GCC
７．キャリーチェーン
 N:        Total       Unique        dd:hh:mm:ss.ms
 4:            2            1        00:00:00:00.00
 5:           10            2        00:00:00:00.00
 6:            4            1        00:00:00:00.00
 7:           40            6        00:00:00:00.00
 8:           92           12        00:00:00:00.00
 9:          352           46        00:00:00:00.00
10:          724           92        00:00:00:00.00
11:         2680          341        00:00:00:00.00
12:        14200         1788        00:00:00:00.01
13:        73712         9237        00:00:00:00.03
14:       365596        45771        00:00:00:00.11
15:      2279184       285095        00:00:00:00.41
16:     14772512      1847425        00:00:00:02.29
17:     95815104     11979381        00:00:00:18.08
18:    666090624     83274576        00:00:03:40.04
19:   4968057848    621051686        00:00:27:37.22
20:  39029188884   4878995797        00:03:39:23.79
bash-3.2$


より最適で高速なコンパイルオプション
bash-3.2$ gcc -Wshift-negative-value -Wall -W -O3 -g -ftrapv -std=c99 -mtune=native -march=native -pthread 08GCC_pthread.c -o 08GCC && ./08GCC
７．キャリーチェーン
 N:        Total       Unique        dd:hh:mm:ss.ms
 4:            2            1        00:00:00:00.00
 5:           10            2        00:00:00:00.00
 6:            4            1        00:00:00:00.00
 7:           40            6        00:00:00:00.00
 8:           92           12        00:00:00:00.00
 9:          352           46        00:00:00:00.00
10:          724           92        00:00:00:00.00
11:         2680          341        00:00:00:00.00
12:        14200         1788        00:00:00:00.00
13:        73712         9237        00:00:00:00.01
14:       365596        45771        00:00:00:00.06
15:      2279184       285095        00:00:00:00.25
16:     14772512      1847425        00:00:00:01.42
17:     95815104     11979381        00:00:00:10.25
18:    666090624     83274576        00:00:01:17.58
19:   4968057848    621051686        00:00:09:19.01
20:  39029188884   4878995797        00:01:12:06.58
bash-3.2$
*/
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#define MAX 27
// システムによって以下のマクロが必要であればコメントを外してください。
//#define UINT64_C(c) c ## ULL
//
// グローバル変数
typedef unsigned long long uint64_t;
uint64_t TOTAL=0; 
uint64_t UNIQUE=0;
// 構造体
typedef struct{
  unsigned int size;
  unsigned int pres_a[930]; 
  unsigned int pres_b[930];
}Global; Global g;
// 構造体
typedef struct{
  uint64_t row;
  uint64_t down;
  uint64_t left;
  uint64_t right;
  uint64_t x[MAX];
}Board ;
typedef struct{
  Board B;
  Board nB;
  Board eB;
  Board sB;
  Board wB;
  unsigned n;
  unsigned e;
  unsigned s;
  unsigned w;
  uint64_t dimx;
  uint64_t dimy;
  uint64_t COUNTER[3];      
  uint64_t COUNT2;
  uint64_t COUNT4;
  uint64_t COUNT8;
}Local;
//
/**
 * pthreadの実行
 */
// bool THREAD=0; // スレッドしない
bool THREAD=1;  // スレッドする
//
//hh:mm:ss.ms形式に処理時間を出力
void TimeFormat(clock_t utime,char* form)
{
  int dd,hh,mm;
  float ftime,ss;
  ftime=(float)utime/CLOCKS_PER_SEC;
  mm=(int)ftime/60;
  ss=ftime-(int)(mm*60);
  dd=mm/(24*60);
  mm=mm%(24*60);
  hh=mm/60;
  mm=mm%60;
  if(dd)
    sprintf(form,"%4d %02d:%02d:%05.2f",dd,hh,mm,ss);
  else if(hh)
    sprintf(form,"     %2d:%02d:%05.2f",hh,mm,ss);
  else if(mm)
    sprintf(form,"        %2d:%05.2f",mm,ss);
  else
    sprintf(form,"           %5.2f",ss);
}
// ボード外側２列を除く内側のクイーン配置処理
uint64_t solve(uint64_t row,uint64_t left,uint64_t down,uint64_t right)
{
  if(down+1==0){ return  1; }
  while((row&1)!=0) { 
    row>>=1;
    left<<=1;
    right>>=1;
  }
  row>>=1;
  uint64_t total=0;
  for(uint64_t bitmap=~(left|down|right);bitmap!=0;){
    uint64_t const bit=bitmap&-bitmap;
    total+=solve(row,(left|bit)<<1,down|bit,(right|bit)>>1);
    bitmap^=bit;
  }
  return total;
} 
// クイーンの効きをチェック
bool placement(void* args)
{
  Local *l=(Local *)args;
  if(l->B.x[l->dimx]==l->dimy){ return true;  }  
  if (l->B.x[0]==0){
    if (l->B.x[1]!=(uint64_t)-1){
      if((l->B.x[1]>=l->dimx)&&(l->dimy==1)){ return false; }
    }
  }else{
    if( (l->B.x[0]!=(uint64_t)-1) ){
      if(( (l->dimx<l->B.x[0]||l->dimx>=g.size-l->B.x[0])
        && (l->dimy==0 || l->dimy==g.size-1)
      )){ return 0; } 
      if ((  (l->dimx==g.size-1)&&((l->dimy<=l->B.x[0])||
          l->dimy>=g.size-l->B.x[0]))){
        return 0;
      } 
    }
  }
  l->B.x[l->dimx]=l->dimy;                    //xは行 yは列
  uint64_t row=UINT64_C(1)<<l->dimx;
  uint64_t down=UINT64_C(1)<<l->dimy;
  uint64_t left=UINT64_C(1)<<(g.size-1-l->dimx+l->dimy); //右上から左下
  uint64_t right=UINT64_C(1)<<(l->dimx+l->dimy);       // 左上から右下
  if((l->B.row&row)||(l->B.down&down)||(l->B.left&left)||(l->B.right&right)){ return false; }     
  l->B.row|=row; l->B.down|=down; l->B.left|=left; l->B.right|=right;
  return true;
}
//対称解除法
void carryChain_symmetry(void* args)
{
  Local *l=(Local *)args;
  // 対称解除法 
  unsigned const int ww=(g.size-2)*(g.size-1)-1-l->w;
  unsigned const int w2=(g.size-2)*(g.size-1)-1;
  // # 対角線上の反転が小さいかどうか確認する
  if((l->s==ww)&&(l->n<(w2-l->e))){ return ; }
  // # 垂直方向の中心に対する反転が小さいかを確認
  if((l->e==ww)&&(l->n>(w2-l->n))){ return; }
  // # 斜め下方向への反転が小さいかをチェックする
  if((l->n==ww)&&(l->e>(w2-l->s))){ return; }
  // 枝刈り １行目が角の場合回転対称チェックせずCOUNT8にする
  if(l->B.x[0]==0){ 
    l->COUNTER[l->COUNT8]+=solve(l->B.row>>2,
    l->B.left>>4,((((l->B.down>>2)|(~0<<(g.size-4)))+1)<<(g.size-5))-1,(l->B.right>>4)<<(g.size-5));
    return ;
  }
  // n,e,s==w の場合は最小値を確認する。右回転で同じ場合は、
  // w=n=e=sでなければ値が小さいのでskip  w=n=e=sであれば90度回転で同じ可能性
  if(l->s==l->w){ if((l->n!=l->w)||(l->e!=l->w)){ return; } 
    l->COUNTER[l->COUNT2]+=solve(l->B.row>>2,
    l->B.left>>4,((((l->B.down>>2)|(~0<<(g.size-4)))+1)<<(g.size-5))-1,(l->B.right>>4)<<(g.size-5));
    return;
  }
  // e==wは180度回転して同じ 180度回転して同じ時n>=sの時はsmaller?
  if((l->e==l->w)&&(l->n>=l->s)){ if(l->n>l->s){ return; } 
    l->COUNTER[l->COUNT4]+=solve(l->B.row>>2,
    l->B.left>>4,((((l->B.down>>2)|(~0<<(g.size-4)))+1)<<(g.size-5))-1,(l->B.right>>4)<<(g.size-5));
    return;
  }
  l->COUNTER[l->COUNT8]+=solve(l->B.row>>2,
  l->B.left>>4,((((l->B.down>>2)|(~0<<(g.size-4)))+1)<<(g.size-5))-1,(l->B.right>>4)<<(g.size-5));
  return;
}
// pthread run()
void* thread_run(void* args)
{
  Local *l=(Local *)args;

  // memcpy(&l->B,&l->wB,sizeof(Board));       // B=wB;
  l->B=l->wB;
  l->dimx=0; l->dimy=g.pres_a[l->w]; 
  //if(!placement(l)){ continue; } 
  // if(!placement(l)){ return; } 
  if(!placement(l)){ return 0; } 
  l->dimx=1; l->dimy=g.pres_b[l->w]; 
  // if(!placement(l)){ continue; } 
  // if(!placement(l)){ return; } 
  if(!placement(l)){ return 0; } 
  //２ 左２行に置く
  // memcpy(&l->nB,&l->B,sizeof(Board));       // nB=B;
  l->nB=l->B;
  for(l->n=l->w;l->n<(g.size-2)*(g.size-1)-l->w;++l->n){
    // memcpy(&l->B,&l->nB,sizeof(Board));     // B=nB;
    l->B=l->nB;
    l->dimx=g.pres_a[l->n]; l->dimy=g.size-1; 
    if(!placement(l)){ continue; } 
    l->dimx=g.pres_b[l->n]; l->dimy=g.size-2; 
    if(!placement(l)){ continue; } 
    // ３ 下２行に置く
    // memcpy(&l->eB,&l->B,sizeof(Board));     // eB=B;
    l->eB=l->B;
    for(l->e=l->w;l->e<(g.size-2)*(g.size-1)-l->w;++l->e){
      // memcpy(&l->B,&l->eB,sizeof(Board));   // B=eB;
      l->B=l->eB;
      l->dimx=g.size-1; l->dimy=g.size-1-g.pres_a[l->e]; 
      if(!placement(l)){ continue; } 
      l->dimx=g.size-2; l->dimy=g.size-1-g.pres_b[l->e]; 
      if(!placement(l)){ continue; } 
      // ４ 右２列に置く
      // memcpy(&l->sB,&l->B,sizeof(Board));   // sB=B;
      l->sB=l->B;
      for(l->s=l->w;l->s<(g.size-2)*(g.size-1)-l->w;++l->s){
        // memcpy(&l->B,&l->sB,sizeof(Board)); // B=sB;
        l->B=l->sB;
        l->dimx=g.size-1-g.pres_a[l->s]; l->dimy=0; 
        if(!placement(l)){ continue; } 
        l->dimx=g.size-1-g.pres_b[l->s]; l->dimy=1; 
        if(!placement(l)){ continue; } 
        // 対称解除法
        carryChain_symmetry(l);
      } //w
    } //e
  } //n
  return 0;
}
// 構造体の初期化
void initLocal(void* args)
{
  Local *l=(Local *)args;
  l->n=l->e=l->s=l->w=0;
  l->dimx=l->dimy=0;
  l->COUNT2=0; 
  l->COUNT4=1; 
  l->COUNT8=2;
  l->COUNTER[l->COUNT2]=
  l->COUNTER[l->COUNT4]=
  l->COUNTER[l->COUNT8]=0;
  l->B.row=
  l->B.down=
  l->B.left=
  l->B.right=0;
  for(unsigned int i=0;i<g.size;++i){ l->B.x[i]=-1; }
}
// チェーンのビルド
void buildChain()
{
  Local l[(g.size/2)*(g.size-3)+1];
  pthread_t pt[(g.size/2)*(g.size-3)+1];
  if(THREAD){
    for(unsigned int w=0;w<(unsigned)(g.size/2)*(g.size-3)+1;++w){
      initLocal(&l[w]); //初期化
      l[w].wB=l[w].B; // memcpy(&l->wB,&l->B,sizeof(Board));  // wB=B;
    }
  }else{
    initLocal(&l); //初期化
    l->wB=l->B; // memcpy(&l->wB,&l->B,sizeof(Board));         // wB=B;
  }
  //１ 上２行に置く
  for(unsigned int w=0;w<(unsigned)(g.size/2)*(g.size-3)+1;++w){
    if(THREAD){ // pthreadのスレッドの生成
      l[w].w=w;
      int iFbRet;
      iFbRet=pthread_create(&pt[w],NULL,&thread_run,&l[w]);
      if(iFbRet>0){
        printf("[mainThread] pthread_create #%d: %d\n", l[w].w, iFbRet);
      }
    }else{ 
      l->w=w;
      thread_run(&l);
    }
  } 
  /**
   * スレッド版 joinする
   */
  if(THREAD){
    for(unsigned int w=0;w<(unsigned)(g.size/2)*(g.size-3)+1;++w){
      pthread_join(pt[w],NULL);
    } 
  }
  /**
   * 集計
   */
  if(THREAD){
    for(unsigned int w=0;w<(unsigned)(g.size/2)*(g.size-3)+1;++w){
      UNIQUE+=l[w].COUNTER[l[w].COUNT2]+
              l[w].COUNTER[l[w].COUNT4]+
              l[w].COUNTER[l[w].COUNT8];
      TOTAL+= l[w].COUNTER[l[w].COUNT2]*2+
              l[w].COUNTER[l[w].COUNT4]*4+
              l[w].COUNTER[l[w].COUNT8]*8;
    } 
  }else{
    UNIQUE= l->COUNTER[l->COUNT2]+
            l->COUNTER[l->COUNT4]+
            l->COUNTER[l->COUNT8];
    TOTAL=  l->COUNTER[l->COUNT2]*2+
            l->COUNTER[l->COUNT4]*4+
            l->COUNTER[l->COUNT8]*8;
  }
}
// チェーンのリストを作成
void listChain()
{
  unsigned int idx=0;
  for(unsigned int a=0;a<(unsigned)g.size;++a){
    for(unsigned int b=0;b<(unsigned)g.size;++b){
      if(((a>=b)&&(a-b)<=1)||((b>a)&&(b-a)<=1)){ continue; }
      g.pres_a[idx]=a;
      g.pres_b[idx]=b;
      ++idx;
    }
  }
}
// キャリーチェーン
void carryChain()
{
  listChain();  //チェーンのリストを作成
  buildChain(); // チェーンのビルド
  // calcChain(&l);  // 集計
}
//メインメソッド
int main(int argc,char** argv)
{
  bool cpu=false,cpur=false;
  int argstart=2;
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else{ cpur=true;}
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g]\n",argv[0]);
    printf("  -c: CPU Without recursion\n");
    printf("  -r: CPUR Recursion\n");
  }
  printf("\n\n８．並列処理 pthread\n");
  printf("%s\n"," N:        Total       Unique        dd:hh:mm:ss.ms");
  // clock_t st;           //速度計測用
  // char t[20];           //hh:mm:ss.msを格納
  unsigned int min=4;
  unsigned int targetN=21;
  // pthread用計測
  struct timeval t0;
  struct timeval t1;
  // sizeはグローバル
  for(unsigned int size=min;size<=targetN;++size){
    TOTAL=UNIQUE=0; 
    g.size=size;
    // pthread用計測
    // st=clock();
    gettimeofday(&t0, NULL);
    if(cpu){ carryChain(); }
    else{ carryChain(); }
    // pthread用計測
    // TimeFormat(clock()-st,t);
    gettimeofday(&t1, NULL);
    // printf("%2d:%13lld%16lld%s\n",size,TOTAL,UNIQUE,t);
    int ss;int ms;int dd;
    if(t1.tv_usec<t0.tv_usec) {
      dd=(t1.tv_sec-t0.tv_sec-1)/86400;
      ss=(t1.tv_sec-t0.tv_sec-1)%86400;
      ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
    }else {
      dd=(t1.tv_sec-t0.tv_sec)/86400;
      ss=(t1.tv_sec-t0.tv_sec)%86400;
      ms=(t1.tv_usec-t0.tv_usec+500)/10000;
    }
    int hh=ss/3600;
    int mm=(ss-hh*3600)/60;
    ss%=60;
    printf("%2d:%13lld%13lld%10.2d:%02d:%02d:%02d.%02d\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
  }
  return 0;
}
