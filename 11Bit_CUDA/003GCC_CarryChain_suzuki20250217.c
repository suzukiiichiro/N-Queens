//日本語
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include <string.h>
#define UINT64_C(c) c ## ULL
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long long xlong;
/**
 *
 */
ulong solve(uint row,uint left,uint down,uint right){
  if(down+1==0) return 1;
  while((row&1)!=0){ row>>=1; left<<=1; right>>=1; }
  row>>=1;
  ulong total=0;
  ulong bit;
  for(ulong bitmap=~(left|down|right);bitmap!=0;bitmap^=bit){
    bit=-bitmap&bitmap;
    total+=solve(row,(left|bit)<<1,down|bit,(right|bit)>>1);
  }
  return total;
}
/**
 *
 */
ulong process(int size,int sym,int B[]){
  return sym*solve(B[0]>>2,B[1]>>4,((((B[2]>>2)|~(0U)<<(size-4))+1)<<(size-5))-1,(unsigned)(B[3]>>4)<<(size-5));
}
/**
 *
 */
ulong Symmetry(int size,int n,int w,int s,int e,int B[],int B4[]){
  int ww=(size-2)*(size-1)-1-w;
  int w2=(size-2)*(size-1)-1;
  if(s==ww&&n<(w2-e)) return 0;
  if(e==ww&&n>(w2-n)) return 0;
  if(n==ww&&e>(w2-s)) return 0;
  if(!B4[0]) return process(size,8,B);
  if(s==w){ if(n!=w||e!=w) return 0; return process(size,2,B); }
  if(e==w&&n>=s){ if(n>s) return 0; return process(size,4,B); }
  return process(size,8,B);
}
/**
 *
 */
bool placement(int size,int dimx,int dimy,int B[],int B4[]){
  if(B4[dimx]==dimy) return true;
  if(B4[0]){
    if((B4[0]!=-1&&((dimx<B4[0]||dimx>=size-B4[0]) &&
      (dimy==0||dimy==size-1)))||
      ((dimx==size-1)&&
      (dimy<=B4[0]||dimy>=size-B4[0]))){ return false;}
  } else if((B4[1]!=-1)&&(B4[1]>=dimx&&dimy==1)){ return false;}
  if((B[0]&(1<<dimx))||
    (B[1]&(1<<(size-1-dimx+dimy)))||
    (B[2]&(1<<dimy))||
    (B[3]&(1<<(dimx+dimy)))){ return false;}
  xlong row=UINT64_C(1)<<dimx;
  xlong left=UINT64_C(1)<<(size-1-dimx+dimy);
  xlong down=UINT64_C(1)<<dimy;
  xlong right=UINT64_C(1)<<(dimx+dimy);
  B4[dimx]=dimy;
  if((B[0]&row)||(B[1]&left)||(B[2]&down)||(B[3]&right)){ return false; }
  B[0]|=row; B[1]|=left; B[2]|=down; B[3]|=right;
  return true;
}
/**
 *
 */
ulong buildChain(int size,int pres_a[],int pres_b[]){
  //プログレス
  printf("\t\t  First side bound: (%d,%d)/(%d,%d)",(unsigned)pres_a[(size/2)*(size-3)  ],(unsigned)pres_b[(size/2)*(size-3)  ],(unsigned)pres_a[(size/2)*(size-3)+1],(unsigned)pres_b[(size/2)*(size-3)+1]);

  ulong total=0;
  int B[4]={0,0,0,0};//row/left/down/right
  int B4[size];
  for(int i=0;i<size;i++) B4[i]=-1;
  int sizeE=size-1;
  int sizeEE=size-2;
  int range_size=(size/2)*(size-3)+1;
  int wB[4];
  int wB4[size];
  for(int w=0;w<range_size;w++){
    //プログレス
    printf("\r(%d/%d)",w,((size/2)*(size-3)));// << std::flush;
    printf("\r");
    fflush(stdout);
    memcpy(wB,B,4*sizeof(int));
    memcpy(wB4,B4,size*sizeof(int));
    if(!placement(size,0,pres_a[w],wB,wB4)||!placement(size,1,pres_b[w],wB,wB4)) continue;
    int nB[4];
    int nB4[size];
    for(int n=w;n<(sizeEE)*(sizeE)-w;n++){
      memcpy(nB,wB,4*sizeof(int));
      memcpy(nB4,wB4,size*sizeof(int));
      if(!placement(size,pres_a[n],sizeE,nB,nB4)||!placement(size,pres_b[n],sizeEE,nB,nB4)) continue;
      int eB[4];
      int eB4[size];
      for(int e=w;e<(sizeEE)*(sizeE)-w;e++){
        memcpy(eB,nB,4*sizeof(int));
        memcpy(eB4,nB4,size*sizeof(int));
        if(!placement(size,sizeE,sizeE-pres_a[e],eB,eB4)||!placement(size,sizeEE,sizeE-pres_b[e],eB,eB4)) continue;
        int sB[4];
        int sB4[size];
        for(int s=w;s<(sizeEE)*(sizeE)-w;s++){
          memcpy(sB,eB,4*sizeof(int));
          memcpy(sB4,eB4,size*sizeof(int));
          if(!placement(size,sizeE-pres_a[s],0,sB,sB4)||!placement(size,sizeE-pres_b[s],1,sB,sB4)) continue;
          total+=Symmetry(size,n,w,s,e,sB,sB4);
        }
      }
    }
  }
  return total;
}
/**
 *
 */
void initChain(int size,int pres_a[],int pres_b[]){
  int idx=0;
  for(int a=0;a<size;a++){
    for(int b=0;b<size;b++){
      if(abs(a-b)<=1) continue;
      pres_a[idx]=a;
      pres_b[idx]=b;
      idx++;
    }
  }
}
/*
 *
 */
ulong carryChain(int size){
  int pres_a[930];
  int pres_b[930];
  initChain(size,pres_a,pres_b);
  return buildChain(size,pres_a,pres_b);
}
/**
 *
 */
void mainNQueens17(){
  int nmin=4;
  int nmax=18;
  struct timeval t0;
  struct timeval t1;
  int ss,ms,dd,hh,mm;
  printf("%s\n","キャリーチェーン");
  printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
  for(int size=nmin;size<=nmax;size++){
    gettimeofday(&t0,NULL);
    ulong TOTAL=carryChain(size);
    ulong UNIQUE=0;
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
    printf("%2d:%13lu%12lu%8.2d:%02d:%02d:%02d.%02d\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
  }
}
/**
 *
 */
int main(){
  mainNQueens17();
  return 0;
}

