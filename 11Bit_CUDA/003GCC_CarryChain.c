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
13:        73712        9237      00:00:00:00.04
14:       365596       45771      00:00:00:00.17
15:      2279184      285095      00:00:00:00.90
16:     16314044     2040171      00:00:00:05.39
17:    167611052    20954665      00:00:00:38.43
18:   2368560040   296093363      00:00:05:27.09

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
#include <sys/time.h>
#define MAX 27
//
//typedef unsigned long long uint;
typedef uint uint;
typedef struct{
  uint bv;
  uint down;
  uint left;
  uint right;
  int x[MAX];
}Board ;
Board B;
uint COUNT8=2;
uint COUNT4=1;
uint COUNT2=0;
long cnt[3];
long pre[3];
long TOTAL=0;
long UNIQUE=0;
/**
 * solve
 */
long solve(uint bv,uint left,uint down,uint right)
{
  uint s=0;
  uint bit;
  if(down+1==0){ return  1;}
  while((bv&1)!=0) {
    bv>>=1;
    left<<=1;
    right>>=1;
  }
  bv>>=1;
  for(uint bitmap=~(left|down|right);bitmap!=0;bitmap^=bit){
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
  uint bv=1<<x;
  uint down=1<<y;
  uint left=1<<(size-1-x+y);
  uint right=1<<(x+y);
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
  int pres_a[930];
  int pres_b[930];
  int idx=0;
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
    Board nB=B;
    int lsize=(size-2)*(size-1)-w;
    for(int n=w;n<lsize;n++){
      B=nB;
      if(placement(size,pres_a[n],size-1)==false){ continue; }
      if(placement(size,pres_b[n],size-2)==false){ continue; }
      Board eB=B;
      for(int e=w;e<lsize;e++){
        B=eB;
        if(placement(size,size-1,size-1-pres_a[e])==false){ continue; }
        if(placement(size,size-2,size-1-pres_b[e])==false){ continue; }
        Board sB=B;
        for(int s=w;s<lsize;s++){
          B=sB;
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
//メインメソッド
int main(int argc,char** argv)
{
  printf("%s\n","キャリーチェーン");
  printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
  // clock_t st;
  // char t[20];
  int min=4;
  int targetN=18;
  // int mask;
  struct timeval t0;
  struct timeval t1;
  uint ss,ms,dd,hh,mm;
  for(int size=min;size<=targetN;size++){
    TOTAL=0; UNIQUE=0;
    for(int j=0;j<=2;j++){
      pre[j]=0;
      cnt[j]=0;
    }
    // mask=(1<<size)-1;
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
    printf("%2d:%13ld%12ld%8.2d:%02d:%02d:%02d.%02d\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
  }
  return 0;
}
