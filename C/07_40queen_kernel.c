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
#define USE_DEBUG 1
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
//
//int push(struct STACK *pStack,int I,int R){
void push(struct STACK *pStack,int I,int R){
  if(USE_DEBUG>0) printf("####################I: %d R: %d\n", I,R);
  if(pStack->current<MAX){
    pStack->param[pStack->current].I=I;
    pStack->param[pStack->current].R=R;
    if(USE_DEBUG>0) printf("pStack->current:%d\n",pStack->current);
    if(USE_DEBUG>0) printf("#########END###########I: %d R: %d\n",pStack->param[pStack->current].I,pStack->param[pStack->current].R);
    (pStack->current)++;
  }
}
//
int pop(struct STACK *pStack){
  int v;
  if(pStack->current>0){
//    pStack->current--;
//    *pValue = pStack->param[pStack->current];
    v=(pStack->current)--;
    return v;
    //return --(pStack->current);
  }
  return 0;
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
  if(USE_DEBUG>0) printf("s.si: %d\n",s.si);
  s.lTotal=state[index].lTotal;
  s.step=state[index].step;
  s.y=state[index].y;

//  init(&s.R);
//  init(&s.I);

//  struct STACK R;
//  struct STACK I;


  s.bend=state[index].bend;
  s.rflg=state[index].rflg;
  s.r=state[index].y;
  for (int j=0;j<s.si;j++){
    s.fA[j]=state[index].fA[j];
    s.fB[j]=state[index].fB[j];
    s.fC[j]=state[index].fC[j];
  }
  s.stParam=state[index].stParam;

  uint16_t j=1;
  int current ;

  while (j!=0) {
    if(USE_DEBUG>0) printf("methodstart\n");
    if(USE_DEBUG>0) printf("###y:%d\n",s.r);
    if(USE_DEBUG>0) printf("###si:%d\n",s.si);
    for(int k=0;k<s.si;k++){
      if(USE_DEBUG>0) printf("###i:%d\n",k);
      if(USE_DEBUG>0) printf("###fa[k]:%d\n",s.fA[k]);
      if(USE_DEBUG>0) printf("###fB[k]:%d\n",s.fB[k]);
      if(USE_DEBUG>0) printf("###fC[k]:%d\n",s.fC[k]);
    }
    if(s.r==s.si && s.rflg==0){
      if(USE_DEBUG>0) printf("if(r==si){\n");
      s.lTotal++;
      if(USE_DEBUG>0) printf("Total++;\n");
    }else{
      if(USE_DEBUG>0) printf("}else{\n");
      for(int i=0;i<s.si;i++){
      if(USE_DEBUG>0) printf("for(int i=0;i<si;i++){\n");
        if(s.rflg==0){
          s.aB[s.r]=i ;
        }
         if(USE_DEBUG>0) printf("aB[r]=i ;\n");
          if(USE_DEBUG>0) printf("###i:%d\n",i);
          if(USE_DEBUG>0) printf("###r:%d\n",s.r);
          for(int k=0;k<s.si;k++){
            if(USE_DEBUG>0) printf("###i:%d\n",k);
            if(USE_DEBUG>0) printf("###fa[k]:%d\n",s.fA[k]);
            if(USE_DEBUG>0) printf("###fB[k]:%d\n",s.fB[k]);
            if(USE_DEBUG>0) printf("###fC[k]:%d\n",s.fC[k]);
          }
        if((s.fA[i]==0&&s.fB[s.r-i+(s.si-1)]==0&&s.fC[s.r+i]==0) || s.rflg==1){
            if(USE_DEBUG>0) printf("if(fA[i]==0&&fB[r-i+(si-1)]==0&&fC[r+i]==0){\n");
          if(s.rflg==0){
            if(USE_DEBUG>0) printf(": %d\n",current);
            s.fA[i]=s.fB[s.r-s.aB[s.r]+s.si-1]=s.fC[s.r+s.aB[s.r]]=1; 
              if(USE_DEBUG>0) printf("fA[i]=fB[r-aB[r]+si-1]=fC[r+aB[r]]=1;\n");
              if(USE_DEBUG>0) printf("###before_nqueen\n");
              if(USE_DEBUG>0) printf("###i:%d\n",i);
              if(USE_DEBUG>0) printf("###r:%d\n",s.r);
              for(int k=0;k<s.si;k++){
                if(USE_DEBUG>0) printf("###i:%d\n",k);
                if(USE_DEBUG>0) printf("###fa[k]:%d\n",s.fA[k]);
                if(USE_DEBUG>0) printf("###fB[k]:%d\n",s.fB[k]);
                if(USE_DEBUG>0) printf("###fC[k]:%d\n",s.fC[k]);
              }
            push(&s.stParam,i,s.r); 
            s.r=s.r+1;
            s.bend=1;
            break;
          }
          if(s.rflg==1){
            current =pop(&s.stParam);
            if(USE_DEBUG>0) printf("###############current : %d\n",current);
            s.r=s.stParam.param[current].R;
            i=s.stParam.param[current].I;
              if(USE_DEBUG>0) printf("###after_nqueen\n");
              if(USE_DEBUG>0) printf("###i:%d\n",i);
              if(USE_DEBUG>0) printf("###r:%d\n",s.r);
              for(int k=0;k<s.si;k++){
                if(USE_DEBUG>0) printf("###i:%d\n",k);
                if(USE_DEBUG>0) printf("###fa[k]:%d\n",s.fA[k]);
                if(USE_DEBUG>0) printf("###fB[k]:%d\n",s.fB[k]);
                if(USE_DEBUG>0) printf("###fC[k]:%d\n",s.fC[k]);
              }
            s.fA[i]=s.fB[s.r-s.aB[s.r]+s.si-1]=s.fC[s.r+s.aB[s.r]]=0; 
            s.rflg=0;
              if(USE_DEBUG>0) printf("fA[i]=fB[r-aB[r]+si-1]=fC[r+aB[r]]=0;\n");
          }
        }else{
          s.bend=0;
        }
          if(USE_DEBUG>0) printf("}#after:if(fA[i]==0&&fB[r-i+(si-1)]==0&&fC[r+i]==0){\n");
          if(USE_DEBUG>0) printf("###bend:%d\n",s.bend);
      }
        if(USE_DEBUG>0) printf("after:for\n");
      if(s.bend==1 && s.rflg==0){
        if(USE_DEBUG>0) printf(": %d\n",current);
        s.bend=0;
        continue;
      }
    }
      if(USE_DEBUG>0) printf("after:else\n");
    if(s.r==0){
      s.step=2;
      break;
    }else{
      s.rflg=1;
    }
    j++;
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
