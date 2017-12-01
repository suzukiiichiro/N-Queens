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
//#define USE_DEBUG 0 
//
struct HIKISU{
  int R;
  int I;
};
//
struct CL_PACKED_KEYWORD STACK {
  struct HIKISU param[MAX];
  int current;
};
//
struct CL_PACKED_KEYWORD queenState {
  int si;
  int id;
  qint aB[MAX];
  uint64_t lTotal;
  char step;
  char y;
  int bend;
  int rflg;
  int r;
  int fA[MAX];
  int fB[MAX];
  int fC[MAX];
  struct STACK stParam;
};
void push(struct STACK *pStack,int I,int R){
  if(pStack->current<MAX){
    pStack->param[pStack->current].I=I;
    pStack->param[pStack->current].R=R;
    (pStack->current)++;
  }
}
//
void pop(struct STACK *pStack){
  if(pStack->current>0){
    pStack->current--;
  }
}
//
CL_KERNEL_KEYWORD void place(CL_GLOBAL_KEYWORD struct queenState *state){
  int index=get_global_id(0);
  struct queenState s ;
  s.si=state[index].si;
  s.id=state[index].id;
  for (int j=0;j<s.si;j++){
    s.aB[j]=state[index].aB[j];
  }
  s.lTotal=state[index].lTotal;
  s.step=state[index].step;
  s.y=state[index].y;
  s.bend=state[index].bend;
  s.rflg=state[index].rflg;
  s.r=state[index].y;
  for (int j=0;j<s.si;j++){
    s.fA[j]=state[index].fA[j];
    s.fB[j]=state[index].fB[j];
    s.fC[j]=state[index].fC[j];
  }
  s.stParam=state[index].stParam;

  //uint16_t j=1;

  //while (j!=0) {
  while (1) {
    if(s.r==s.si && s.rflg==0){
      s.lTotal++;
    }else{
      for(int i=0;i<s.si;i++){
        if(s.rflg==0){
          s.aB[s.r]=i ;
        }
        if((s.fA[i]==0&&s.fB[s.r-i+(s.si-1)]==0&&s.fC[s.r+i]==0) || s.rflg==1){
          if(s.rflg==0){
            s.fA[i]=s.fB[s.r-s.aB[s.r]+s.si-1]=s.fC[s.r+s.aB[s.r]]=1; 
            push(&s.stParam,i,s.r); 
            s.r=s.r+1;
            s.bend=1;
            break;
          }
          if(s.rflg==1){
            pop(&s.stParam);
            s.r=s.stParam.param[s.stParam.current].R;
            i=s.stParam.param[s.stParam.current].I;
            s.fA[i]=s.fB[s.r-s.aB[s.r]+s.si-1]=s.fC[s.r+s.aB[s.r]]=0; 
            s.rflg=0;
          }
        }else{
          s.bend=0;
        }
      }
      if(s.bend==1 && s.rflg==0){
        s.bend=0;
        continue;
      }
    }
    if(s.r==0){
      s.step=2;
      break;
    }else{
      s.rflg=1;
    }
    //j++;
  }
  state[index].si=s.si;
  state[index].id=s.id;
  for (int j=0;j<s.si;j++){
    state[index].aB[j] = s.aB[j];
  }
  state[index].lTotal=s.lTotal;
  state[index].step=s.step;
  state[index].y=s.y;
  state[index].bend=s.bend;
  state[index].rflg=s.rflg;
  state[index].y=s.y;
  for (int j=0;j<s.si;j++){
    state[index].fA[j]=s.fA[j];
    state[index].fB[j]=s.fB[j];
    state[index].fC[j]=s.fC[j];
  }
  state[index].stParam=s.stParam;
}
#ifdef GCC_STYLE
int main(){
  int si=10; 
  struct queenState l[1];
  long gTotal=0;
    l[0].BOUND1=i;
    l[0].si=si;
    for (int m=0;m< si;m++){
      l[0].aB[m]=m;
    }
    l[0].step=0;
    l[0].y=0;
    l[0].bm=(1<<SIZE)-1;
    l[0].lTotal=0;
    place(&l[0]);
    gTotal+=l[0].lTotal;
      }
    }
  }
  return 0;
}
#endif
