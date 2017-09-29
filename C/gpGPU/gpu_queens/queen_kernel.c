#ifndef OPENCL_STYLE
#include "stdio.h"
#include "stdint.h"
int get_global_id(int dimension){ return 0;}
#define CL_KERNEL_KEYWORD
#define CL_GLOBAL_KEYWORD
#define CL_CONSTANT_KEYWORD
#define CL_PACKED_KEYWORD
#define SIZE 8
#else
#define CL_KERNEL_KEYWORD __kernel
#define CL_GLOBAL_KEYWORD __global
#define CL_CONSTANT_KEYWORD __constant
#define CL_PACKED_KEYWORD  __attribute__ ((packed))
#endif
#define MAX 27  
struct CL_PACKED_KEYWORD queenState {
  int si;
  int BOUND1;
  int id;
  int aB[MAX];
  int step;
  int y;
  int startCol;
  int msk;
  int bm;
  int down;
  int right;
  int left;
  long lTotal;
};
CL_KERNEL_KEYWORD void place(CL_GLOBAL_KEYWORD struct queenState *state){
  int index=get_global_id(0);
  int si        =state[index].si;
  int BOUND1    =state[index].BOUND1;
  int id    =state[index].id;
  int aB[MAX];
  for (int i=0;i<si;i++)
    aB[i]=state[index].aB[i];
  int step      =state[index].step;
  int y         =state[index].y;
  int startCol  =state[index].startCol;
  int msk       =state[index].msk;
  int bm        =state[index].bm;
  int down      =state[index].down;
  int right     =state[index].right;
  int left      =state[index].left;
  long lTotal   =state[index].lTotal;
  int bit;
  while(1){
    if(step==1){
      if(y==startCol){
        step=2;
        break;
      }
      bm=aB[--y];
    }
    if(y==0){
      if(bm & (1<<BOUND1)){
        bit=1<<BOUND1;
        aB[y]=bit;
      }else{
        step=2;
        break;
      }
    }else{
      bit=bm&-bm;
      aB[y]=bit;
    }
    down  ^= bit;
    right ^= bit<<y;
    left  ^= bit<<(si-1-y);
    if(step==0){
      aB[y++]=bm;
      if(y==si){
        lTotal += 1;
        step=1;
      }else{
        bm=msk&~(down|(right>>y)|(left>>((si-1)-y)));
        if(bm==0)
          step=1;
      }
    }else{
      bm ^= bit;
      if(bm==0)
        step=1;
      else
        step=0;
    }
  }
  state[index].si       =si;
  state[index].BOUND1   =BOUND1;
  state[index].id   =id;
  for(int i=0;i<si;i++)
    state[index].aB[i]=aB[i];
  state[index].step     =step;
  state[index].y        =y;
  state[index].startCol =startCol;
  state[index].msk      =msk;
  state[index].bm       =bm;
  state[index].down     =down;
  state[index].right    =right;
  state[index].left     =left;
  state[index].lTotal   =lTotal;
}
#ifdef GCC_STYLE
int main(){
  int si=8; 
  struct queenState l[SIZE];
  long gTotal=0;
  for (int i=0;i<SIZE;i++){
    l[i].si=si;
    l[i].BOUND1=i;
    l[i].aB[i]=i;
    l[i].step=0;
    l[i].y=0;
    l[i].startCol=1;
    l[i].msk=(1<<SIZE)-1;
    l[i].bm=(1<<SIZE)-1;
    l[i].down=0;
    l[i].right=0;
    l[i].left=0;
    l[i].lTotal=0;
    place(&l[i]);
    gTotal+=l[i].lTotal;
    printf("%ld\n", l[i].lTotal);
  }
  printf("%ld\n", gTotal);
  return 0;
}
#endif
