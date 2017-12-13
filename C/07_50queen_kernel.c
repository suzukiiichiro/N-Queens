﻿//  単体で動かすときは以下のコメントを外す
//#define GCC_STYLE
#ifndef OPENCL_STYLE
#include "stdio.h"
#include "stdint.h"
typedef int64_t qint;
int get_global_id(int dimension){ return 0;}
#define CL_KERNEL_KEYWORD
#define CL_GLOBAL_KEYWORD
#define CL_CONSTANT_KEYWORD
#define CL_PACKED_KEYWORD
#define SIZE 17
#else
//typedef long qint;
//typedef long int64_t;
//typedef ulong uint64_t;
typedef ushort uint16_t;
#define CL_KERNEL_KEYWORD __kernel
#define CL_GLOBAL_KEYWORD __global
#define CL_CONSTANT_KEYWORD __constant
#define CL_PACKED_KEYWORD  __attribute__ ((packed))
#endif
#define MAX 27  
CL_PACKED_KEYWORD struct HIKISU{
  int Y;
  int I;
  int M;
  int L;
  int D;
  int R;
  int B;
};
CL_PACKED_KEYWORD struct STACK {
  struct HIKISU param[MAX];
  int current;
};
struct CL_PACKED_KEYWORD queenState {
  int BOUND1;
  int si;
  int aB[MAX];
  long lTotal; // Number of solutinos found so far.
  int step;
  int y;
  int startCol; // First column this individual computation was tasked with filling.
  int bm;
  int BOUND2;
  int TOPBIT;
  int ENDBIT;
  int SIDEMASK;
  int LASTMASK;
  long lUnique; // Number of solutinos found so far.
  int bend;
  int rflg;
  struct STACK stParam;
  int msk;
  int l;
  int d;
  int r;
  int B1;
};
CL_KERNEL_KEYWORD void place(CL_GLOBAL_KEYWORD struct queenState *state){
  int index = get_global_id(0);
  int BOUND1=state[index].BOUND1;
  int si= state[index].si;
  int aB[MAX];
  for (int i = 0; i < si; i++)
    aB[i]=state[index].aB[i];
  long lTotal = state[index].lTotal;
  int step      = state[index].step;
  int y       = state[index].y;
  int startCol  = state[index].startCol;
  int bm     = state[index].bm;
  int BOUND2    =state[index].BOUND2;
  int TOPBIT    =state[index].TOPBIT;
  int ENDBIT    =state[index].ENDBIT;
  int SIDEMASK    =state[index].SIDEMASK;
  int LASTMASK  =state[index].LASTMASK;
  long lUnique  =state[index].lUnique;
  int bend  =state[index].bend;
  int rflg  =state[index].rflg;
  struct STACK sp=state[index].stParam;
  int msk= state[index].msk;
  int l= state[index].l;
  int d= state[index].d;
  int r= state[index].r;
  int B1= state[index].B1;
  printf("BOUND1:%d\n",BOUND1);
  printf("si:%d\n",si);
  printf("lTotal:%ld\n",lTotal);
  printf("step:%d\n",step);
  printf("y:%d\n",y);
  printf("startCol:%d\n",startCol);
  printf("bm:%d\n",bm);
  printf("BOUND2:%d\n",BOUND2);
  printf("TOPBIT:%d\n",TOPBIT);
  printf("ENDBIT:%d\n",ENDBIT);
  printf("SIDEMASK:%d\n",SIDEMASK);
  printf("LASTMASK:%d\n",LASTMASK);
  printf("lUnique:%ld\n",lUnique);
  printf("bend:%d\n",bend);
  printf("rflg:%d\n",rflg);
  printf("msk:%d\n",msk);
  printf("l:%d\n",l);
  printf("d:%d\n",d);
  printf("r:%d\n",r);
  printf("B1:%d\n",B1);
state[index].BOUND1=BOUND1;
state[index].si=si;
for(int i=0;i<si;i++){state[index].aB[i]=aB[i];}
state[index].lTotal=lTotal;
state[index].step=step;
state[index].y=y;
state[index].startCol=startCol;
state[index].bm=bm;
state[index].BOUND2=BOUND2;
state[index].TOPBIT=TOPBIT;
state[index].ENDBIT=ENDBIT;
state[index].SIDEMASK=SIDEMASK;
state[index].LASTMASK=LASTMASK;
state[index].lUnique=lUnique;
state[index].bend=bend;
state[index].rflg=rflg;
state[index].stParam=sp;
state[index].msk=msk;
state[index].l=l;
state[index].d=d;
state[index].r=r;
state[index].B1=B1;
}
#ifdef GCC_STYLE
int main(){
  int target=12;
  /**********/
  struct queenState l[MAX];
  /**********/
  printf("%s\n"," N:          Total        Unique\n");
  /**********/
  for(int si=4;si<=target;si++){
    long gTotal=0;
    for (int i=0;i<si;i++){
      l[i].BOUND1=i;
      l[i].si=si;
      l[i].id=i;
      for (int j=0;j<si;j++){ l[i].aB[j]=j;}
      l[i].lTotal=0;
      l[i].step=0;
      l[i].y=0;
      l[i].startCol=1;
      l[i].bm=(1<<si)-1;
      l[i].down=0;
      l[i].right=0;
      l[i].left=0;
      place(&l[i]);
    }
  /**********/
    for(int i=0;i<si;i++){
      gTotal+=l[i].lTotal;
    }
  /**********/
    printf("%2d:%18ld\n", si,gTotal);
  }
  return 0;
}
#endif
