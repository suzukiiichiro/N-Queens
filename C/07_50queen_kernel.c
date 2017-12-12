//  単体で動かすときは以下のコメントを外す
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
  int id;
  int aB[MAX];
  long lTotal; // Number of solutinos found so far.
  int step;
  int y;
  int startCol; // First column this individual computation was tasked with filling.
  int bm;
  long down;
  long right;
  long left;
  int BOUND2;
  int TOPBIT;
  int ENDBIT;
  int SIDEMASK;
  int LASTMASK;
  long lUnique; // Number of solutinos found so far.
  int bend;
  int rflg;
  struct STACK stParam;
};
CL_KERNEL_KEYWORD void place(CL_GLOBAL_KEYWORD struct queenState *state){
  int index = get_global_id(0);
  int BOUND1=state[index].BOUND1;
  int si= state[index].si;
  int id= state[index].id;
  int aB[MAX];
  for (int i = 0; i < si; i++)
    aB[i]=state[index].aB[i];
  long lTotal = state[index].lTotal;
  int step      = state[index].step;
  int y       = state[index].y;
  int startCol  = state[index].startCol;
  int bm     = state[index].bm;
  long down     = state[index].down;
  long right      = state[index].right;
  long left      = state[index].left;
  int BOUND2    =state[index].BOUND2;
  int TOPBIT    =state[index].TOPBIT;
  int ENDBIT    =state[index].ENDBIT;
  int SIDEMASK    =state[index].SIDEMASK;
  int LASTMASK  =state[index].LASTMASK;
  long lUnique  =state[index].lUnique;
  int bend  =state[index].bend;
  int rflg  =state[index].rflg;
  struct STACK sp=state[index].stParam;
  printf("BOUND1:%d\n",BOUND1);
  printf("si:%d\n",si);
  printf("id:%d\n",id);
  printf("lTotal:%ld\n",lTotal);
  printf("step:%d\n",step);
  printf("y:%d\n",y);
  printf("startCol:%d\n",startCol);
  printf("bm:%d\n",bm);
  printf("down:%ld\n",down);
  printf("right:%ld\n",right);
  printf("left:%ld\n",left);
  printf("BOUND2:%d\n",BOUND2);
  printf("TOPBIT:%d\n",TOPBIT);
  printf("ENDBIT:%d\n",ENDBIT);
  printf("SIDEMASK:%d\n",SIDEMASK);
  printf("LASTMASK:%d\n",LASTMASK);
  printf("lUnique:%ld\n",lUnique);
  printf("bend:%d\n",bend);
  printf("rflg:%d\n",rflg);
state[index].BOUND1=BOUND1;
state[index].si=si;
state[index].id=id;
for(int i=0;i<si;i++){state[index].aB[i]=aB[i];}
state[index].lTotal=lTotal;
state[index].step=step;
state[index].y=y;
state[index].startCol=startCol;
state[index].bm=bm;
state[index].down=down;
state[index].right=right;
state[index].left=left;
state[index].BOUND2=BOUND2;
state[index].TOPBIT=TOPBIT;
state[index].ENDBIT=ENDBIT;
state[index].SIDEMASK=SIDEMASK;
state[index].LASTMASK=LASTMASK;
state[index].lUnique=lUnique;
state[index].bend=bend;
state[index].rflg=rflg;
state[index].stParam=sp;
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
