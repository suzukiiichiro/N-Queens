/**
bash-5.1$ g++ -W -Wall -O3 00GCC_CarryChain.c && ./a.out
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
12:        14200        1788      00:00:00:00.01
13:        73712        9237      00:00:00:00.05
14:       365596       45771      00:00:00:00.19
15:      2279184      285095      00:00:00:00.93
16:     14772512     1847425      00:00:00:05.46
17:     95815104    11979381      00:00:00:36.85
18:    657378384    82181924      00:00:04:21.49

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
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#define MAX 27
//
typedef long long uint64_t;
typedef struct{
  uint64_t bv;
  uint64_t down;
  uint64_t left;
  uint64_t right;
  uint64_t x[MAX];
}Board ;
Board B;
uint64_t COUNT8=2;
uint64_t COUNT4=1;
uint64_t COUNT2=0;
uint64_t cnt[3];
uint64_t pre[3];
uint64_t TOTAL=0;
uint64_t UNIQUE=0;
/**
 * solve
 */
uint64_t solve(uint64_t bv,uint64_t left,uint64_t down,uint64_t right)
{
  uint64_t s=0;
  uint64_t bit;
  if(down+1==0){ return  1;}
  while((bv&1)!=0) {
    bv>>=1;
    left<<=1;
    right>>=1;
  }
  bv>>=1;
  for(uint64_t bitmap=~(left|down|right);bitmap!=0;bitmap^=bit){
    bit=bitmap&-bitmap;
    s+=solve(bv,(left|bit)<<1,down|bit,(right|bit)>>1);
  }
  return s;
}
/**
 * process
 */
void process(int size,Board B,int sym)
{
  pre[sym]++;
  cnt[sym]+=solve(B.bv>>2,B.left>>4,((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,(B.right>>4)<<(size-5));
}
/**
 * placement
 */
bool placement(int size,int x,int y)
{
  if(B.x[x]==y){ return true;  }
  B.x[x]=y;
  uint64_t bv=1<<x;
  uint64_t down=1<<y;
  uint64_t left=1<<(size-1-x+y);
  uint64_t right=1<<(x+y);
  if((B.bv&bv)||(B.down&down)||(B.left&left)||(B.right&right)){ return false; }
  B.bv |=bv;
  B.down |=down;
  B.left |=left;
  B.right |=right;
  return true;
}
/**
 * nqueens
 */
void nqueens(int size)
{
  uint64_t pres_a[930];
  uint64_t pres_b[930];
  uint64_t idx=0;
  for(int a=0;a<size;a++){
    for(int b=0;b<size;b++){
      if((a>=b&&(a-b)<=1)||(b>a&&(b-a)<=1)){ continue; }
      pres_a[idx]=a;
      pres_b[idx]=b;
      idx++;
    }
  }
  //プログレス
  // printf("\t\t  First side bound: (%d,%d)/(%d,%d)",(unsigned)pres_a[(size/2)*(size-3)  ],(unsigned)pres_b[(size/2)*(size-3)  ],(unsigned)pres_a[(size/2)*(size-3)+1],(unsigned)pres_b[(size/2)*(size-3)+1]);
  //プログレス
  Board wB=B;
  for(int w=0;w<=(size/2)*(size-3);w++){
    B=wB;
    B.bv=B.down=B.left=B.right=0;
    for(int i=0;i<size;i++){ B.x[i]=-1; }
    //プログレス
    // printf("\r(%d/%d)",w,((size/2)*(size-3)));
    // printf("\r");
    // fflush(stdout);
    //プログレス
    placement(size,0,pres_a[w]);
    placement(size,1,pres_b[w]);
    // Board nB=B;
    Board nB;
    memcpy(&nB,&B,sizeof(Board));
    int lsize=(size-2)*(size-1)-w;
    for(int n=w;n<lsize;n++){
      // B=nB;
      memcpy(&B,&nB,sizeof(Board));
      if(placement(size,pres_a[n],size-1)==false){ continue; }
      if(placement(size,pres_b[n],size-2)==false){ continue; }
      // Board eB=B;
      Board eB;
      memcpy(&eB,&B,sizeof(Board));
      for(int e=w;e<lsize;e++){
        // B=eB;
        memcpy(&B,&eB,sizeof(Board));
        if(placement(size,size-1,size-1-pres_a[e])==false){ continue; }
        if(placement(size,size-2,size-1-pres_b[e])==false){ continue; }
        // Board sB=B;
        Board sB;
        memcpy(&sB,&B,sizeof(Board));
        for(int s=w;s<lsize;s++){
          // B=sB;
          memcpy(&B,&sB,sizeof(Board));
          if(placement(size,size-1-pres_a[s],0)==false){ continue; }
          if(placement(size,size-1-pres_b[s],1)==false){ continue; }
          int ww=(size-2)*(size-1)-1-w;
          int w2=(size-2)*(size-1)-1;
          if((s==ww)&&(n<(w2-e))){ continue; }
          if((e==ww)&&(n>(w2-n))){ continue;}
          if((n==ww)&&(e>(w2-s))){ continue; }
          if(s==w){ if((n!=w)||(e!=w)){ continue; } process(size,B,COUNT2);continue; }
          if((e==w)&&(n>=s)){ if(n>s){ continue; } process(size,B,COUNT4);continue; }
          process(size,B,COUNT8);continue;
        }
      }
    }
  }
  UNIQUE=cnt[COUNT2]+cnt[COUNT4]+cnt[COUNT8];
  TOTAL=cnt[COUNT2]*2+cnt[COUNT4]*4+cnt[COUNT8]*8;
}
/**
 * メイン
*/
int main(int argc,char** argv)
{
  printf("%s\n","キャリーチェーン");
  printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
  int min=4;
  int targetN=18;
  struct timeval t0;
  struct timeval t1;
  uint64_t ss,ms,dd,hh,mm;
  for(int size=min;size<=targetN;size++){
    TOTAL=0; UNIQUE=0;
    for(int j=0;j<=2;j++){
      pre[j]=0;
      cnt[j]=0;
    }
    gettimeofday(&t0,NULL);
    nqueens(size);
    gettimeofday(&t1,NULL);
    if(t1.tv_usec<t0.tv_usec) {
      dd=(t1.tv_sec-t0.tv_sec-1)/86400;
      ss=(t1.tv_sec-t0.tv_sec-1)%86400;
      ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
    }else {
      dd=(t1.tv_sec-t0.tv_sec)/86400;
      ss=(t1.tv_sec-t0.tv_sec)%86400;
      ms=(t1.tv_usec-t0.tv_usec+500)/10000;
    }
    hh=ss/3600;
    mm=(ss-hh*3600)/60;
    ss%=60;
    printf("%2d:%13lld%12lld%8.2lld:%02lld:%02lld:%02lld.%02lld\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
  }
  return 0;
}
