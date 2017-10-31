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
#define SIZE 10
#else
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
typedef struct{
    int array[MAX];
    int current;
}STACK;
void init(STACK*);
int push(STACK*,int);
int pop(STACK*,int*);
void init(STACK* pStack)
{
    int i;
    for(i = 0; i < MAX; i++){
        pStack->array[i] = 0;
    }
    pStack->current = 0;
}
int push(STACK* pStack,int value)
{
    if(pStack->current < MAX){
        pStack->array[pStack->current] = value;
        pStack->current++;
        return 1;
    }
    return 0;
}
int pop(STACK* pStack,int* pValue)
{
    if(pStack->current > 0){
        pStack->current--;
        *pValue = pStack->array[pStack->current];
        return *pValue;
    }
    return 0;
}
int leng(STACK* pStack)
{
    if(pStack->current > 0){
     return 1;
    }
    return 0;
}
struct CL_PACKED_KEYWORD queenState {
  int si;
  int id;
  qint aB[MAX];
  uint64_t lTotal;
  char step;
  char y;
};
CL_KERNEL_KEYWORD void place(CL_GLOBAL_KEYWORD struct queenState *state){
  int index = get_global_id(0);
  struct queenState s ;
  s.si= state[index].si;
  s.id= state[index].id;
  for (int i = 0; i < s.si; i++)
    s.aB[i] = state[index].aB[i];
  s.lTotal = state[index].lTotal;
  s.step      = state[index].step;
  s.y       = state[index].y;
  if(s.step !=2){
    STACK R;
    STACK I;
    init(&R);
    init(&I);
    int bend=0;
    int rflg=0;
    int si=s.si;
    int r=s.y;
    int fA[MAX];
    int fB[MAX];
    int fC[MAX];
    uint16_t i = 1;
  while (i != 0) {
  	i++;
      if(r==si && rflg==0){
        s.lTotal++;
      }else{
        for(int i=0;i<si;i++){
          if(rflg==0){
            s.aB[r]=i ;
          }
          if(fA[i]==0&&fB[r-i+(si-1)]==0&&fC[r+i]==0){
            if(rflg==0){
              fA[i]=fB[r-s.aB[r]+si-1]=fC[r+s.aB[r]]=1; 
              push(&R,r); 
              push(&I,i); 
              r=r+1;
              bend=1;
              break;
            }
            if(rflg==1){
              r=pop(&R,&r);
              i=pop(&I,&i);
              fA[i]=fB[r-s.aB[r]+si-1]=fC[r+s.aB[r]]=0; 
              rflg=0;
            }
          }
          if(bend==1 && rflg==0){
            bend=0;
            continue;
          }
        }
      }
      if(r==0){
        s.step=2;
        break;
      }else{
        rflg=1;
      }
  }
  }
  state[index].si      = s.si;
  state[index].id      = s.id;
  for (int i = 0; i < s.si; i++)
    state[index].aB[i] = s.aB[i];
  state[index].lTotal = s.lTotal;
  state[index].step= s.step;
  state[index].y= s.y;
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
