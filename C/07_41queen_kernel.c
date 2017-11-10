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
//
struct HIKISU{
  int Y;
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
  long lTotal;
  long lUnique; // Number of solutinos found so far.
  char step;
  char y;
  int bend;
  int rflg;
  int fA[MAX];
  int fB[MAX];
  int fC[MAX];
  qint aT[MAX];        //aT:aTrial[]
  qint aS[MAX];        //aS:aScrath[]
  struct STACK stParam;
};
//void nrotate(qint chk[],qint scr[],int n,int neg){
//void nrotate(qint *chk,qint *scr,int n,int neg){
void nrotate(struct queenState *s,int neg){
  int k=neg?0:s->si-1;
  int incr=(neg?+1:-1);
  //for(int i=0;i<s->si;i++){ 
  //  printf("chk[%d]=%lld\n",i,s->aT[i]);
  //  printf("scr[%d]=%lld\n",i,s->aS[i]);
  //}
  //for(int j=0;j<s.si;k+=incr){ scr[j++]=chk[k];}
  for(int j=0;j<s->si;k+=incr){ s->aS[j++]=s->aT[k];}
  k=neg?s->si-1:0;
  //for(int j=0;j<s.si;k-=incr){ chk[scr[j++]]=k;}
  for(int j=0;j<s->si;k-=incr){ s->aT[s->aS[j++]]=k;}
}

void vMirror(struct queenState *s){
  //for(int j=0;j<s.si;j++){ chk[j]=(n-1)- chk[j];}
  for(int j=0;j<s->si;j++){ s->aT[j]=(s->si-1)-s->aT[j];}
}
int intncmp(struct queenState *s){
  int rtn=0;
  for(int k=0;k<s->si;k++){
    //rtn=lt[k]-rt[k];
    rtn=s->aB[k]-s->aT[k];
    if(rtn!=0){ break;}
  }
  return rtn;
}
//int symmetryOps(int si,qint aB[],qint aT,qint aS[]){
int symmetryOps(struct queenState *s){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<s->si;i++){ s->aT[i]=s->aB[i];
  //  printf("aT[%d]=%lld\n",i,s->aT[i]);
  }
  nrotate(s,0);       //時計回りに90度回転
  //int k=intncmp(aB,aT,si);
  int k=intncmp(s);
  if(k>0)return 0;
  if(k==0){ nEquiv=1; }else{
    nrotate(s,0);     //時計回りに180度回転
    k=intncmp(s);
    if(k>0)return 0;
    if(k==0){ nEquiv=2; }else{
      nrotate(s,0);   //時計回りに270度回転
      k=intncmp(s);
      if(k>0){ return 0; }
      nEquiv=4;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<s->si;i++){ s->aT[i]=s->aB[i];}
  vMirror(s);           //垂直反転
  k=intncmp(s);
  if(k>0){ return 0; }
  if(nEquiv>1){             //-90度回転 対角鏡と同等       
    nrotate(s,1);
    k=intncmp(s);
    if(k>0){return 0; }
    if(nEquiv>2){           //-180度回転 水平鏡像と同等
      nrotate(s,1);
      k=intncmp(s);
      if(k>0){ return 0; }  //-270度回転 反対角鏡と同等
      nrotate(s,1);
      k=intncmp(s);
      if(k>0){ return 0; }
    }
  }
  return nEquiv*2;
}

void push(struct STACK *pStack,int I,int Y){
  if(pStack->current<MAX){
    pStack->param[pStack->current].I=I;
    pStack->param[pStack->current].Y=Y;
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
  s.lUnique=state[index].lUnique;
  s.step=state[index].step;
  s.y=state[index].y;
  s.bend=state[index].bend;
  s.rflg=state[index].rflg;
  for (int j=0;j<s.si;j++){
    s.fA[j]=state[index].fA[j];
    s.fB[j]=state[index].fB[j];
    s.fC[j]=state[index].fC[j];
    s.aT[j]=state[index].aT[j];
    s.aS[j]=state[index].aS[j];
  }
  s.stParam=state[index].stParam;

  //uint16_t j=1;
  //unsigned long j=1;
  unsigned long j=1;
  //int sum;
  //while (j!=0) {
  //while (1) {
  //printf("while:%d\n",sum);
  while (j<200000) {
    if(s.y==s.si && s.rflg==0){
   //   s.lTotal++;
      //sum=symmetryOps(s.si,s.aB,s.aT,s.aS);//対称解除法
   //   printf("return:%d\n",sum);
      int sum=symmetryOps(&s);//対称解除法
   //   printf("sum:%d\n",sum);
      if(sum!=0){ s.lUnique++; s.lTotal+=sum; } //解を発見
    }else{
      for(int i=0;i<s.si;i++){
        if(s.rflg==0){
          s.aB[s.y]=i ;
        }
        if((s.fA[i]==0&&s.fB[s.y-i+(s.si-1)]==0&&s.fC[s.y+i]==0) || s.rflg==1){
          if(s.rflg==0){
            s.fA[i]=s.fB[s.y-s.aB[s.y]+s.si-1]=s.fC[s.y+s.aB[s.y]]=1; 
            push(&s.stParam,i,s.y); 
            s.y=s.y+1;
            s.bend=1;
            break;
          }
          if(s.rflg==1){
            pop(&s.stParam);
            s.y=s.stParam.param[s.stParam.current].Y;
            i=s.stParam.param[s.stParam.current].I;
            s.fA[i]=s.fB[s.y-s.aB[s.y]+s.si-1]=s.fC[s.y+s.aB[s.y]]=0; 
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
    if(s.y==0){
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
  state[index].lUnique=s.lUnique;
  state[index].step=s.step;
  state[index].y=s.y;
  state[index].bend=s.bend;
  state[index].rflg=s.rflg;
  for (int j=0;j<s.si;j++){
    state[index].fA[j]=s.fA[j];
    state[index].fB[j]=s.fB[j];
    state[index].fC[j]=s.fC[j];
    state[index].aT[j]=s.aT[j];
    state[index].aS[j]=s.aS[j];
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
