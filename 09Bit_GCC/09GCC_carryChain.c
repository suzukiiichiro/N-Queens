/**
 *
 * bash版キャリーチェーンのC言語版
 * 最終的に 08Bash_carryChain_parallel.sh のように
 * 並列処理 pthread版の作成が目的
 *
 * 今回のテーマ
 * carryChain_symmetry()に n,e,s,w,Bをわたしていたが、Local* l を渡すようにする
    
 構造体のコピー
 https://yu-nix.com/archives/c-struct-copy/
 memcpy
 https://yu-nix.com/archives/c-struct-copy/
 アドレスとポインタ
 https://yu-nix.com/archives/c-struct-pointer/


最適化オプション含め以下を参考に
bash$ gcc -Wall -W -O3 -mtune=native -march=native 07GCC_carryChain.c -o nq27 && ./nq27 -r
７．キャリーチェーン
７．キャリーチェーン
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.00
12:        14200            1788            0.01
13:        73712            9237            0.05
14:       365596           45771            0.19
15:      2279184          285095            1.01
16:     14772512         1847425            6.10
17:     95815104        11979381           40.53


 bash-3.2$ gcc -Wall -W -O3 GCC12.c && ./a.out -r
１２．CPUR 再帰 対称解除法の最適化
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.00
12:        14200            1787            0.00
13:        73712            9233            0.01
14:       365596           45752            0.07
15:      2279184          285053            0.41
16:     14772512         1846955            2.66
17:     95815104        11977939           18.41
18:    666090624        83263591         2:14.44
19:   4968057848       621012754        17:06.46
*/
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include <pthread.h>
#define MAX 27
// グローバル変数
uint64_t TOTAL=0; 
uint64_t UNIQUE=0;
// 構造体
typedef struct{
  unsigned int size;
  unsigned int pres_a[930]; 
  unsigned int pres_b[930];
  uint64_t COUNTER[3];      
  //カウンター配列
  unsigned int COUNT2;
  unsigned int COUNT4;
  unsigned int COUNT8;
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
}Local;
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
// 集計
void calcChain()
{
  UNIQUE= g.COUNTER[g.COUNT2]+
          g.COUNTER[g.COUNT4]+
          g.COUNTER[g.COUNT8];
  TOTAL=  g.COUNTER[g.COUNT2]*2+
          g.COUNTER[g.COUNT4]*4+
          g.COUNTER[g.COUNT8]*8;
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
// solve()を呼び出して再帰を開始する
void process(unsigned const int sym,Board* B)
{
  g.COUNTER[sym]+=solve(B->row>>2,
  B->left>>4,((((B->down>>2)|(~0<<(g.size-4)))+1)<<(g.size-5))-1,(B->right>>4)<<(g.size-5));
}
// クイーンの効きをチェック
bool placement(uint64_t dimx,uint64_t dimy,Board* B)
{
  if(B->x[dimx]==dimy){ return true;  }  
  if (B->x[0]==0){
    if (B->x[1]!=(uint64_t)-1){
      if((B->x[1]>=dimx)&&(dimy==1)){ return false; }
    }
  }else{
    if( (B->x[0]!=(uint64_t)-1) ){
      if(( (dimx<B->x[0]||dimx>=g.size-B->x[0])
        && (dimy==0 || dimy==g.size-1)
      )){ return 0; } 
      if ((  (dimx==g.size-1)&&((dimy<=B->x[0])||
          dimy>=g.size-B->x[0]))){
        return 0;
      } 
    }
  }
  B->x[dimx]=dimy;                    //xは行 yは列
  uint64_t row=UINT64_C(1)<<dimx;
  uint64_t down=UINT64_C(1)<<dimy;
  uint64_t left=UINT64_C(1)<<(g.size-1-dimx+dimy); //右上から左下
  uint64_t right=UINT64_C(1)<<(dimx+dimy);       // 左上から右下
  if((B->row&row)||(B->down&down)||(B->left&left)||(B->right&right)){ return false; }     
  B->row|=row; B->down|=down; B->left|=left; B->right|=right;
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
    process(g.COUNT8,&l->B); return ;
  }
  // n,e,s==w の場合は最小値を確認する。右回転で同じ場合は、
  // w=n=e=sでなければ値が小さいのでskip  w=n=e=sであれば90度回転で同じ可能性
  if(l->s==l->w){ if((l->n!=l->w)||(l->e!=l->w)){ return; } 
    process(g.COUNT2,&l->B); return;
  }
  // e==wは180度回転して同じ 180度回転して同じ時n>=sの時はsmaller?
  if((l->e==l->w)&&(l->n>=l->s)){ if(l->n>l->s){ return; } 
    process(g.COUNT4,&l->B); return;
  }
  process(g.COUNT8,&l->B); return;
}
// チェーンのビルド
void buildChain()
{
  Local l[(g.size/2)*(g.size-3)];

  // Board B;
  // カウンターの初期化
  g.COUNTER[g.COUNT2]=g.COUNTER[g.COUNT4]=g.COUNTER[g.COUNT8]=0;
  g.COUNT2=0; g.COUNT4=1; g.COUNT8=2;
  // Board の初期化 nB,eB,sB,wB;
  l->B.row=l->B.down=l->B.left=l->B.right=0;
  // Board x[]の初期化
  for(unsigned int i=0;i<g.size;++i){ l->B.x[i]=-1; }
  // Board wB;
  memcpy(&l->wB,&l->B,sizeof(Board));
  // wB=B;//１ 上２行に置く
  for(l->w=0;l->w<=(unsigned)(g.size/2)*(g.size-3);++l->w){
    // B=wB;
    memcpy(&l->B,&l->wB,sizeof(Board));
    if(!placement(0,g.pres_a[l->w],&l->B)){ continue; } 
    if(!placement(1,g.pres_b[l->w],&l->B)){ continue; }
    // nB=B;//２ 左２行に置く
    // Board nB;
    memcpy(&l->nB,&l->B,sizeof(Board));
    for(l->n=l->w;l->n<(g.size-2)*(g.size-1)-l->w;++l->n){
      // B=nB;
      memcpy(&l->B,&l->nB,sizeof(Board));
      if(!placement(g.pres_a[l->n],g.size-1,&l->B)){ continue; }
      if(!placement(g.pres_b[l->n],g.size-2,&l->B)){ continue; }
      // eB=B;// ３ 下２行に置く
      // Board eB;
      memcpy(&l->eB,&l->B,sizeof(Board));
      for(l->e=l->w;l->e<(g.size-2)*(g.size-1)-l->w;++l->e){
        // B=eB;
        memcpy(&l->B,&l->eB,sizeof(Board));
        if(!placement(g.size-1,g.size-1-g.pres_a[l->e],&l->B)){ continue; }
        if(!placement(g.size-2,g.size-1-g.pres_b[l->e],&l->B)){ continue; }
        // sB=B;// ４ 右２列に置く
        // Board sB;
        memcpy(&l->sB,&l->B,sizeof(Board));
        for(l->s=l->w;l->s<(g.size-2)*(g.size-1)-l->w;++l->s){
          // B=sB;
          memcpy(&l->B,&l->sB,sizeof(Board));
          if(!placement(g.size-1-g.pres_a[l->s],0,&l->B)){ continue; }
          if(!placement(g.size-1-g.pres_b[l->s],1,&l->B)){ continue; }
          // 対称解除法
          // carryChain_symmetry(l->n,l->e,l->s,l->w,&l->B);
          carryChain_symmetry(&l);
        } //w
      } //e
    } //n
  } //w
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
  calcChain();  // 集計
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
  printf("\n\n７．キャリーチェーン\n");
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  clock_t st;           //速度計測用
  char t[20];           //hh:mm:ss.msを格納
  unsigned int min=4;
  unsigned int targetN=21;
  // sizeはグローバル
  for(unsigned int size=min;size<=targetN;++size){
    TOTAL=UNIQUE=0; 
    g.size=size;
    st=clock();
    if(cpu){
      carryChain();
    }else{
      carryChain();
    }
    TimeFormat(clock()-st,t);
    printf("%2d:%13lld%16lld%s\n",size,TOTAL,UNIQUE,t);
  }
  return 0;
}