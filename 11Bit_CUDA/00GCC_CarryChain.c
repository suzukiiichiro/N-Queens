#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#define THREAD_NUM		96
#define MAX 27
//
typedef unsigned long long uint64;
typedef struct{
  uint64 bv;
  uint64 down;
  uint64 left;
  uint64 right;
  int x[MAX];
}Board ;
Board B;
unsigned int COUNT8=2;
unsigned int COUNT4=1;
unsigned int COUNT2=0;
long cnt[3];
long pre[3];
long TOTAL=0;
long UNIQUE=0;
/**
 * TimeFormat
 */
void TimeFormat(clock_t utime,char *form)
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
  if(dd){ sprintf(form,"%4d %02d:%02d:%05.2f",dd,hh,mm,ss); }
  else if(hh){ sprintf(form,"     %2d:%02d:%05.2f",hh,mm,ss); }
  else if(mm){ sprintf(form,"        %2d:%05.2f",mm,ss); }
  else{ sprintf(form,"           %5.2f",ss); }
}
/**
 *
 */
long solve_nqueenr(uint64 bv,uint64 left,uint64 down,uint64 right)
{
  if(down+1==0){ return  1;}
  while((bv&1)!=0) { 
    bv>>=1;
    left<<=1;
    right>>=1;
  }
  bv>>=1;
  long s=0;
  uint64 bit;
  for(uint64 bitmap=~(left|down|right);bitmap!=0;bitmap^=bit){
    bit=bitmap&-bitmap;
    s+=solve_nqueenr(bv,(left|bit)<<1,down|bit,(right|bit)>>1);
  }
  return s;
}
/**
 *
 */
void process(int si,Board B,int sym)
{
  pre[sym]++;
  cnt[sym] += solve_nqueenr(B.bv >> 2,
      B.left>>4,
      ((((B.down>>2)|(~0<<(si-4)))+1)<<(si-5))-1,
      (B.right>>4)<<(si-5));
}
/**
 *
 */
bool board_placement(int si,int x,int y)
{
  if(B.x[x]==y){ return true;  }
  B.x[x]=y;
  uint64 bv=1<<x;
  uint64 down=1<<y;
  uint64 left=1<<(si-1-x+y);
  uint64 right=1<<(x+y);
  if((B.bv&bv)||(B.down&down)||(B.left&left)||(B.right&right)){ return false; }
  B.bv |=bv;
  B.down |=down;
  B.left |=left;
  B.right |=right;
  return true;
}
/**
 *
 */
void NQueenR(int size)
{
  int pres_a[930];
  int pres_b[930];
  int idx=0;
  for(int a=0;a<size;a++){
    for(int b=0;b<size;b++){
      if((a>=b&&(a-b)<=1)||(b>a&&(b-a)<=1)){
        continue;
      }     
      pres_a[idx]=a;
      pres_b[idx]=b;
      idx++;
    }
  }
  //プログレス
  printf("\t\t  First side bound: (%d,%d)/(%d,%d)",(unsigned)pres_a[(size/2)*(size-3)  ],(unsigned)pres_b[(size/2)*(size-3)  ],(unsigned)pres_a[(size/2)*(size-3)+1],(unsigned)pres_b[(size/2)*(size-3)+1]);
  //プログレス
  Board wB=B;
  for(int w=0;w<=(size/2)*(size-3);w++){
    B=wB;
    B.bv=B.down=B.left=B.right=0;
    for(int i=0;i<size;i++){ B.x[i]=-1; }
    //プログレス
    printf("\r(%d/%d)",w,((size/2)*(size-3)));
    printf("\r");
    fflush(stdout);
    //プログレス
    board_placement(size,0,pres_a[w]);
    board_placement(size,1,pres_b[w]);
    Board nB=B;
    int lsize=(size-2)*(size-1)-w;
    for(int n=w;n<lsize;n++){
      B=nB;
      if(board_placement(size,pres_a[n],size-1)==false){ continue; }
      if(board_placement(size,pres_b[n],size-2)==false){ continue; }
      Board eB=B;
      for(int e=w;e<lsize;e++){
        B=eB;
        if(board_placement(size,size-1,size-1-pres_a[e])==false){ continue; }
        if(board_placement(size,size-2,size-1-pres_b[e])==false){ continue; }
        Board sB=B;
        for(int s=w;s<lsize;s++){
          B=sB;
          if(board_placement(size,size-1-pres_a[s],0)==false){ continue; }
          if(board_placement(size,size-1-pres_b[s],1)==false){ continue; }
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
  printf("\n\n７．CPU 非再帰 バックトラック＋ビットマップ＋対称解除法\n");
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  clock_t st;           //速度計測用
  char t[20];           //hh:mm:ss.msを格納
  int min=4;int targetN=17;
  int mask;
  for(int i=min;i<=targetN;i++){
    TOTAL=0; UNIQUE=0;
    for(int j=0;j<=2;j++){
      pre[j]=0;
      cnt[j]=0;
    }
    mask=(1<<i)-1;
    st=clock();
    NQueenR(i); 
    TimeFormat(clock()-st,t); 
    printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
  }
  return 0;
}
