//  単体で動かすときは以下のコメントを外す
//#define GCC_STYLE

#ifndef OPENCL_STYLE
  // Declarations appropriate to this program being compiled with gcc.
  #include "stdio.h"
  #include "stdint.h"
  typedef int64_t qint;
  // A stub for OpenCL's get_global_id function.
  int get_global_id(int dimension){ return 0;}
#define CL_KERNEL_KEYWORD
#define CL_GLOBAL_KEYWORD
#define CL_CONSTANT_KEYWORD
#define CL_PACKED_KEYWORD
#define SIZE 10
#define BOUND1 0
#else
  // Declarations appropriate to this program being compiled as an OpenCL
  // kernel. OpenCL has a 64 bit long and requires special keywords to designate
  // where and how different objects are stored in memory.
//typedef long qint;
//typedef long int64_t;
//typedef ulong uint64_t;
  typedef long qint;
  typedef long int64_t;
  typedef ulong uint64_t;
  typedef ushort uint16_t;
#define CL_KERNEL_KEYWORD __kernel
#define CL_GLOBAL_KEYWORD __global
#define CL_CONSTANT_KEYWORD __constant
#define CL_PACKED_KEYWORD  __attribute__ ((packed))
#endif


#define MAX 27  

struct CL_PACKED_KEYWORD queenState {
  //int BOUND1;
//  qint BOUND1;
  //int BOUND2;
  qint BOUND2;
  //int BOUND3;
  qint BOUND3;
//  int si;
  int id;
  //int aB[MAX];
  //int aB[SIZE];
  //qint aB[SIZE];
  qint aB[MAX];
  //long lTotal; // Number of solutinos found so far.
  uint64_t lTotal; // Number of solutinos found so far.
  //int step;
  char step;
  //int y;
  char y;
  //int startCol; // First column this individual computation was tasked with filling.
  char startCol; // First column this individual computation was tasked with filling.
  //int bm;
  qint bm;
  //long down;
  qint down;
  //long right;
  qint right;
  //long left;
  qint left;
};

CL_CONSTANT_KEYWORD const qint msk=(1<<SIZE)-1;
CL_KERNEL_KEYWORD void place(CL_GLOBAL_KEYWORD struct queenState *state){
  int index = get_global_id(0);
  //int BOUND1=state[index].BOUND1;
//  qint BOUND1=state[index].BOUND1;
  //int BOUND2=state[index].BOUND2;
  qint BOUND2=state[index].BOUND2;
  //int BOUND3=state[index].BOUND3;
  qint BOUND3=state[index].BOUND3;
//  int si= state[index].si;
  int id= state[index].id;
  //int aB[MAX];
  qint aB[SIZE];
  //for (int i = 0; i < si; i++)
  for (int i = 0; i < SIZE; i++)
    aB[i] = state[index].aB[i];
  //long lTotal = state[index].lTotal;
  uint64_t lTotal = state[index].lTotal;
  //int step      = state[index].step;
  char step      = state[index].step;
  //int y       = state[index].y;
  char y       = state[index].y;
  //int startCol  = state[index].startCol;
  char startCol  = state[index].startCol;
  //int bm     = state[index].bm;
  qint bm     = state[index].bm;
  //long down     = state[index].down;
  qint down     = state[index].down;
  //long right      = state[index].right;
  qint right      = state[index].right;
  //long left      = state[index].left;
  qint left      = state[index].left;
  //long bit;
  qint bit;
//  int msk = (1 << si) - 1;
//  printf("bound:%d:startCol:%d:ltotal:%ld:step:%d:y:%d:bm:%d:down:%d:right:%d:left:%d\n", BOUND1,startCol,lTotal,step,y,bm,down,right,left);
  uint16_t i = 1;
  //long i=0;
  //while (i <300000)
//  printf("#######BOUND1:%d\n",BOUND1);
  while (i != 0) {
  	i++;
    if(step==1){
      if(y<=startCol){
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
    }else if(y==1){
      if(bm & (1<<BOUND2)){
        bit=1<<BOUND2;
      }else{
        step=2;
        break;
      }
    }else if(y==2){
      if(bm & (1<<BOUND3)){
        bit=1<<BOUND3;
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
    //left  ^= bit<<(si-1-y);
    left  ^= bit<<(SIZE-1-y);
    if(step==0){
      aB[y++]=bm;
      //if(y==si){
      if(y==SIZE){
        lTotal += 1;
        step=1;
      }else{
        //bm=msk&~(down|(right>>y)|(left>>((si-1)-y)));
        bm=msk&~(down|(right>>y)|(left>>((SIZE-1)-y)));
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
//  state[index].BOUND1   =BOUND1;
  state[index].BOUND2   =BOUND2;
  state[index].BOUND3   =BOUND3;
  //state[index].si      = si;
  state[index].id      = id;
  //for (int i = 0; i < si; i++)
  for (int i = 0; i < SIZE; i++)
    state[index].aB[i] = aB[i];
  state[index].lTotal = lTotal;
  state[index].step      = step;
  state[index].y       = y;
  state[index].startCol  = startCol;
  state[index].bm      = bm;
  state[index].down      = down;
  state[index].right       = right;
  state[index].left       = left;
}
#ifdef GCC_STYLE
int main(){
//  int si=10; 
  struct queenState l[SIZE*SIZE*SIZE];
  //long gTotal=0;
  uint64_t gTotal=0;
  for (int i=0;i<SIZE;i++){
    for(int j=0;j<si;j++){
      for(int k=0;k<si;k++){
//    l[i*SIZE*SIZE+j*SIZE+k].BOUND1=i;
    l[i*SIZE*SIZE+j*SIZE+k].BOUND2=j;
    l[i*SIZE*SIZE+j*SIZE+k].BOUND3=k;
    l[i*SIZE*SIZE+j*SIZE+k].SIZE=SIZE;
    for (int m=0;m< SIZE;m++){
      l[i*SIZE*SIZE+j*SIZE+k].aB[m]=m;
    }
    l[i*SIZE*SIZE+j*SIZE+k].step=0;
    l[i*SIZE*SIZE+j*SIZE+k].y=0;
    l[i*SIZE*SIZE+j*SIZE+k].startCol=3;
//    l[i].msk=(1<<SIZEZE)-1;
    l[i*SIZE*SIZE+j*SIZE+k].bm=(1<<SIZEZE)-1;
    l[i*SIZE*SIZE+j*SIZE+k].down=0;
    l[i*SIZE*SIZE+j*SIZE+k].right=0;
    l[i*SIZE*SIZE+j*SIZE+k].left=0;
    l[i*SIZE*SIZE+j*SIZE+k].lTotal=0;
    place(&l[i*SIZE*SIZE+j*SIZE+k]);
    gTotal+=l[i*SIZE*SIZE+j*SIZE+k].lTotal;
    //printf("%ld\n", l[i*SIZE*SIZE+j*SIZE+k].lTotal);
    printf("%llu\n", l[i*SIZE*SIZE+j*SIZE+k].lTotal);
      }
    }
  }
  printf("%ld\n", gTotal);
  return 0;
}
#endif
