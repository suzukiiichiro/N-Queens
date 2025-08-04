/**
bash-5.1$ gcc -W -Wall -O3 00GCC_CarryChain.c && ./a.out
キャリーチェーン
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2           1      00:00:00:00.00
 5:           10           2      00:00:00:00.00
 6:            4           1      00:00:00:00.00
 7:           40           6      00:00:00:00.00
 8:           92          12      00:00:00:00.00
 9:          352          46      00:00:00:00.00
10:          724          92      00:00:00:00.00
11:         2680         341      00:00:00:00.00
12:        14200        1788      00:00:00:00.02
13:        73712        9237      00:00:00:00.06
14:       365596       45771      00:00:00:00.21
15:      2279184      285095      00:00:00:00.99
16:     14772512     1847425      00:00:00:05.68
17:     95815104    11979381      00:00:00:38.60
18:    666090624    83274576      00:00:04:34.42
bash-5.1$ g++ -W -Wall -O3 00GCC_NodeLayer.c && ./a.out
ノードレイヤー
 N:        Total      Unique      dd:hh:mm:ss.ms
15:      2279184           0      00:00:00:00.70
16:     14772512           0      00:00:00:04.69
17:     95815104           0      00:00:00:32.70
18:    666090624           0      00:00:03:59.95

bash-5.1$ gcc -W -Wall -O3 01CUDA_Bit_Symmetry.c && ./a.out
ビット
 N:        Total      Unique      dd:hh:mm:ss.ms
15:      2279184      285053      00:00:00:00.33
16:     14772512     1846955      00:00:00:02.16
17:     95815104    11977939      00:00:00:14.89
18:    666090624    83263591      00:00:01:45.44
 *
 */
#include <stdio.h>
#include <stdbool.h>
#include <sys/time.h>
#define MAX 27
#define UINT64_C(c) c ## ULL
typedef unsigned long long uint64_t;
uint64_t TOTAL=0;
uint64_t UNIQUE=0;
typedef struct{
  unsigned int size;
  unsigned int pres_a[930];
  unsigned int pres_b[930];
}Global;
Global g;
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
  unsigned int COUNT2;
  unsigned int COUNT4;
  unsigned int COUNT8;
}Local;
/**
 * ボード外側２列を除く内側のクイーン配置処理
 */
uint64_t solve(uint64_t row,uint64_t left,uint64_t down,uint64_t right)
{
  if(down+1==0){ return 1; }
  while((row&1)!=0) { row>>=1; left<<=1; right>>=1; }
  row>>=1;
  uint64_t total=0;
  for(uint64_t bitmap=~(left|down|right);bitmap!=0;){
    uint64_t const bit=bitmap&-bitmap;
    total+=solve(row,(left|bit)<<1,down|bit,(right|bit)>>1);
    bitmap^=bit;
  }
  return total;
}
/**
 * クイーンの効きをチェック
 */
bool placement(void* args)
{
  Local *l=(Local *)args;
  if(l->B.x[l->dimx]==l->dimy){ return true;  }
  if (l->B.x[0]==0){
    if (l->B.x[1]!=(uint64_t)-1){ if((l->B.x[1]>=l->dimx)&&(l->dimy==1)){ return false; } }
  }else{
    if( (l->B.x[0]!=(uint64_t)-1) ){
      if(( (l->dimx<l->B.x[0]||l->dimx>=g.size-l->B.x[0]) && (l->dimy==0 || l->dimy==g.size-1))){ return 0; }
      if ((  (l->dimx==g.size-1)&&((l->dimy<=l->B.x[0])||l->dimy>=g.size-l->B.x[0]))){ return 0; }
    }
  }
  l->B.x[l->dimx]=l->dimy;
  uint64_t row=UINT64_C(1)<<l->dimx;
  uint64_t down=UINT64_C(1)<<l->dimy;
  uint64_t left=UINT64_C(1)<<(g.size-1-l->dimx+l->dimy);
  uint64_t right=UINT64_C(1)<<(l->dimx+l->dimy);
  if((l->B.row&row)||(l->B.down&down)||(l->B.left&left)||(l->B.right&right)){ return false; }
  l->B.row|=row; l->B.down|=down; l->B.left|=left; l->B.right|=right;
  return true;
}
/**
 *
 */
void process(void* args,int sym)
{
  Local *l=(Local *)args;
  l->COUNTER[sym]+=solve(l->B.row>>2,l->B.left>>4,((((l->B.down>>2)|~0<<(g.size-4))+1)<<(g.size-5))-1,(l->B.right>>4)<<(g.size-5));
}
/**
 * 対称解除法
 */
void carryChain_symmetry(void* args)
{
  Local *l=(Local *)args;
  unsigned const int ww=(g.size-2)*(g.size-1)-1-l->w;
  unsigned const int w2=(g.size-2)*(g.size-1)-1;
  // 対角線上の反転が小さいかどうか確認する
  if((l->s==ww)&&(l->n<(w2-l->e))){ return ; }
  // 垂直方向の中心に対する反転が小さいかを確認
  if((l->e==ww)&&(l->n>(w2-l->n))){ return; }
  // 斜め下方向への反転が小さいかをチェックする
  if((l->n==ww)&&(l->e>(w2-l->s))){ return; }
  // 枝刈り １行目が角の場合回転対称チェックせずCOUNT8にする
  if(l->B.x[0]==0){ process(l,2); return ; } //COUNT8
  // n,e,s==w の場合は最小値を確認する。右回転で同じ場合は、w=n=e=sでなければ値が小さいのでskip  w=n=e=sであれば90度回転で同じ可能性 COUNT2
  if(l->s==l->w){ if((l->n!=l->w)||(l->e!=l->w)){ return; } process(l,0); return; } //COUNT2
  // e==wは180度回転して同じn>=sの時はsmaller? COUNT4
  if((l->e==l->w)&&(l->n>=l->s)){ if(l->n>l->s){ return; } process(l,1); return; } //COUNT4
  process(l,2); return;//COUNT8
}
/**
 * pthreadにも対応できるように
 */
void thread_run(void* args)
{
  Local *l=(Local *)args;
  l->B=l->wB;
  l->dimx=0; l->dimy=g.pres_a[l->w];
  if(!placement(l)){ return; }
  l->dimx=1; l->dimy=g.pres_b[l->w];
  if(!placement(l)){ return; }
  l->nB=l->B; //２ 左２行に置く
  for(l->n=l->w;l->n<(g.size-2)*(g.size-1)-l->w;++l->n){
    l->B=l->nB;
    l->dimx=g.pres_a[l->n]; l->dimy=g.size-1;
    if(!placement(l)){ continue; }
    l->dimx=g.pres_b[l->n]; l->dimy=g.size-2;
    if(!placement(l)){ continue; }
    l->eB=l->B; // ３ 下２行に置く
    for(l->e=l->w;l->e<(g.size-2)*(g.size-1)-l->w;++l->e){
      l->B=l->eB;
      l->dimx=g.size-1; l->dimy=g.size-1-g.pres_a[l->e];
      if(!placement(l)){ continue; }
      l->dimx=g.size-2; l->dimy=g.size-1-g.pres_b[l->e];
      if(!placement(l)){ continue; }
      l->sB=l->B; // ４ 右２列に置く
      for(l->s=l->w;l->s<(g.size-2)*(g.size-1)-l->w;++l->s){
        l->B=l->sB;
        l->dimx=g.size-1-g.pres_a[l->s]; l->dimy=0;
        if(!placement(l)){ continue; }
        l->dimx=g.size-1-g.pres_b[l->s]; l->dimy=1;
        if(!placement(l)){ continue; }
        carryChain_symmetry(l);// 対称解除法
      }
    }
  }
}
// チェーンのビルド
void buildChain()
{
  Local l[(g.size/2)*(g.size-3)];
  l->COUNT2=0; l->COUNT4=1; l->COUNT8=2;
  l->COUNTER[l->COUNT2]=l->COUNTER[l->COUNT4]=l->COUNTER[l->COUNT8]=0;
  l->B.row=l->B.down=l->B.left=l->B.right=0;
  for(unsigned int i=0;i<g.size;++i){ l->B.x[i]=-1; }
  l->wB=l->B; //１ 上２行に置く
  for(l->w=0;l->w<=(unsigned)(g.size/2)*(g.size-3);++l->w){ thread_run(&l); }
  UNIQUE= l->COUNTER[l->COUNT2]+
          l->COUNTER[l->COUNT4]+
          l->COUNTER[l->COUNT8];
  TOTAL=  l->COUNTER[l->COUNT2]*2+
          l->COUNTER[l->COUNT4]*4+
          l->COUNTER[l->COUNT8]*8;
}
/**
 * チェーンのリストを作成
 */
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
/**
 * キャリーチェーン
 */
void carryChain()
{
  listChain();  //チェーンのリストを作成
  buildChain(); // チェーンのビルド
}
/**
 * メイン
 */
int main()
{
  printf("キャリーチェーン\n");
  printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
  unsigned int min=4;
  unsigned int targetN=18;
  struct timeval t0;
  struct timeval t1;
  int ss,ms,dd,hh,mm;
  for(unsigned int size=min;size<=targetN;++size){
    TOTAL=UNIQUE=0;
    gettimeofday(&t0,NULL);
    g.size=size;
    carryChain();
    gettimeofday(&t1,NULL);
    if(t1.tv_usec<t0.tv_usec){
      dd=(t1.tv_sec-t0.tv_sec-1)/86400;
      ss=(t1.tv_sec-t0.tv_sec-1)%86400;
      ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
    }else{
      dd=(t1.tv_sec-t0.tv_sec)/86400;
      ss=(t1.tv_sec-t0.tv_sec)%86400;
      ms=(t1.tv_usec-t0.tv_usec+500)/10000;
    }
    hh=ss/3600;
    mm=(ss-hh*3600)/60;
    ss%=60;
    printf("%2d:%13llu%12llu%8.2d:%02d:%02d:%02d.%02d\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
  }
  return 0;
}


