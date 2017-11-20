//  単体で動かすときは以下のコメントを外す
//#define GCC_STYLE

#ifndef OPENCL_STYLE
  // Declarations appropriate to this program being compiled with gcc.
  #include "stdio.h"
  #include "stdint.h"
  #include <math.h>
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
CL_PACKED_KEYWORD struct HIKISU{
  int Y;
  int I;
  int M;
  int L;
  int D;
  int R;
  int B;
};
//
CL_PACKED_KEYWORD struct STACK {
  struct HIKISU param[MAX];
  int current;
};
//
CL_PACKED_KEYWORD struct queenState {
  int si;
  int id;
  qint aB[MAX];
  long lTotal;
  long lUnique; // Number of solutinos found so far.
  char step;
  char y;
  int bend;
  int rflg;
  qint aT[MAX];        //aT:aTrial[]
  qint aS[MAX];        //aS:aScrath[]
  struct STACK stParam;
  int msk;
  int l;
  int d;
  int r;
  int bm;
  int bit;
  int BOUND1;
};
int rh(int a,int sz){
  int tmp=0;
  for(int i=0;i<=sz;i++){
    if(a&(1<<i)){ return tmp|=(1<<(sz-i)); }
  }
  return tmp;
}
//void nrotate(qint chk[],qint scr[],int n,int neg){
//void nrotate(qint *chk,qint *scr,int n,int neg){
int intncmp(qint lt[],qint rt[],int si){
  // printf("lt[]:%lld\n",lt[2]);
  int rtn=0;
  for(int k=0;k<si;k++){
    rtn=lt[k]-rt[k];
    if(rtn!=0){ break;}
  }
  return rtn;
}
//void symmetryOps_bm(int si){
int symmetryOps_bm(struct queenState *s){
  int nEquiv;
  // printf("s->si:%d\n",s->si);
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<s->si;i++){ s->aT[i]=s->aB[i];}
  //rotate_bitmap(s);    //時計回りに90度回転
  //rotate_bitmap(aT,aS,si);    //時計回りに90度回転

  for(int i=0;i<s->si;i++){
    int t=0;
    for(int j=0;j<s->si;j++){
      t|=((s->aT[j]>>i)&1)<<(s->si-j-1); // x[j] の i ビット目を
    }
    s->aS[i]=t;                        // y[i] の j ビット目にする
    // printf("2####\n");
  }
  // printf("s->aB:%lld\n",s->aB[0]);
  int k=intncmp(s->aB,s->aS,s->si);
  // printf("intcmp:%d\n",k);
  if(k>0)return 0;
  if(k==0){ nEquiv=2;}else{
    //rotate_bitmap(s);  //時計回りに180度回転
    //rotate_bitmap(aS,aT,si);  //時計回りに180度回転
    for(int i=0;i<s->si;i++){
      int t=0;
      for(int j=0;j<s->si;j++){
        t|=((s->aS[j]>>i)&1)<<(s->si-j-1); // x[j] の i ビット目を
      }
      s->aT[i]=t;                        // y[i] の j ビット目にする
    }
    //k=intncmp(s);
    k=intncmp(s->aB,s->aT,s->si);
    if(k>0)return 0;
    if(k==0){ nEquiv=4;}else{
      //rotate_bitmap(s);//時計回りに270度回転
      //rotate_bitmap(aT,aS,si);//時計回りに270度回転
      for(int i=0;i<s->si;i++){
        int t=0;
        for(int j=0;j<s->si;j++){
          t|=((s->aT[j]>>i)&1)<<(s->si-j-1); // x[j] の i ビット目を
        }
        s->aS[i]=t;                        // y[i] の j ビット目にする
      }
      //k=intncmp(s);
      k=intncmp(s->aB,s->aS,s->si);
      if(k>0){ return 0;}
      nEquiv=8;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<s->si;i++){ s->aS[i]=s->aB[i];}
  //vMirror_bitmap(s);   //垂直反転
  //vMirror_bitmap(aS,aT,si);   //垂直反転
  int score ;
  for(int i=0;i<s->si;i++) {
    score=s->aS[i];
    s->aT[i]=rh(score,s->si-1);
  }
  //k=intncmp(s);
  k=intncmp(s->aB,s->aT,s->si);
  // printf("#k:%d\n",k);
  if(k>0){ return 0; }
  if(nEquiv>2){               //-90度回転 対角鏡と同等       
    //rotate_bitmap(s);
    //rotate_bitmap(aT,aS,si);
    for(int i=0;i<s->si;i++){
      int t=0;
      for(int j=0;j<s->si;j++){
        t|=((s->aT[j]>>i)&1)<<(s->si-j-1); // x[j] の i ビット目を
      }
      s->aS[i]=t;                        // y[i] の j ビット目にする
    }
    //k=intncmp(s);
    k=intncmp(s->aB,s->aS,s->si);
    // printf("#k:%d\n",k);
    if(k>0){return 0;}


    if(nEquiv>4){             //-180度回転 水平鏡像と同等
      //rotate_bitmap(s);
      //rotate_bitmap(aS,aT,si);
      for(int i=0;i<s->si;i++){
        int t=0;
        for(int j=0;j<s->si;j++){
          t|=((s->aS[j]>>i)&1)<<(s->si-j-1); // x[j] の i ビット目を
        }
        s->aT[i]=t;                        // y[i] の j ビット目にする
      }
      //k=intncmp(s);
      k=intncmp(s->aB,s->aT,s->si);
      //printf("#k:%d\n",k);
      if(k>0){ return 0;}       //-270度回転 反対角鏡と同等
      //rotate_bitmap(s);
      //rotate_bitmap(aT,aS,si);
      for(int i=0;i<s->si;i++){
        int t=0;
        for(int j=0;j<s->si;j++){
          t|=((s->aT[j]>>i)&1)<<(s->si-j-1); // x[j] の i ビット目を
        }
        s->aS[i]=t;                        // y[i] の j ビット目にする
      }
      //k=intncmp(s);
      k=intncmp(s->aB,s->aS,s->si);
      if(k>0){ return 0;}
    }
  }
  //if(nEquiv==2){ C2++; }
  //if(nEquiv==4){ C4++; }
  //if(nEquiv==8){ C8++; }
  return nEquiv;
}
/**
//void push(struct STACK *pStack,int I,int Y){
void push(struct queenState *s,int I){
  //if(pStack->current<MAX){
  if(s->stParam.current<MAX){
    //pStack->param[pStack->current].I=I;
    s->stParam.param[s->stParam.current].I=I;
    //pStack->param[pStack->current].Y=Y;
    s->stParam.param[s->stParam.current].Y=s->y;
    //(pStack->current)++;
    (s->stParam.current)++;
  }
}
//
//void pop(struct STACK *pStack){
void pop(struct queenState *s){
  //if(pStack->current>0){
  if(s->stParam.current>0){
    //pStack->current--;
    s->stParam.current--;
  }
}
*/
//
CL_KERNEL_KEYWORD void place(CL_GLOBAL_KEYWORD struct queenState *state){
  int index=get_global_id(0);
  struct queenState s ;
  s.si=state[index].si;
  // printf("#state[index].si:%d\n",state[index].si);
  // printf("#s.si:%d\n",s.si);
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
    s.aT[j]=state[index].aT[j];
    s.aS[j]=state[index].aS[j];
  }
  s.stParam=state[index].stParam;
  // s.msk=state[index].msk;
  s.msk=(1<<s.si)-1;
  // printf("kernel:s.msk:%lld\n",s.msk);
  s.l=state[index].l;
  s.d=state[index].d;
  s.r=state[index].r;
  s.bm=state[index].bm;
  s.BOUND1=state[index].BOUND1;

  if (s.BOUND1==0){
    //backtrack1;
    unsigned long j=1;
    // int sum;
    int bit;

    while (j<200000) {
      //  start:
      if(s.rflg==0){
        s.bm=s.msk&~(s.l|s.d|s.r); /* 配置可能フィールド */
        // printf("s.bm:%d s.msk:%lld s.l:%d s.d:%d\n",s.bm,s.msk,s.l,s.d);
      }
      // printf("s.si:%d\n",s.si);
      if (s.y==s.si && !s.bm && s.rflg==0) {
        // printf("s.si:%d\n",s.si);
        // if(!s.bm){
        s.aB[s.y]=s.bm;
        // printf("s.si:%d\n",s.si);
        int sum=symmetryOps_bm(&s);//対称解除法
        // printf("sum:%d\n",sum);
        if(sum!=0){ s.lUnique++; s.lTotal+=sum; } //解を発見
        //        printf("lTotal:%ld\n",s.lTotal);
        // printf("lUnique:%ld\n",s.lUnique);
        // }
      }else{
        if(s.y==0){
          s.aB[0]=bit=(1<<s.BOUND1);
          if(s.stParam.current<MAX){
            s.stParam.param[s.stParam.current].Y=s.y;
            s.stParam.param[s.stParam.current].I=s.si;
            s.stParam.param[s.stParam.current].M=s.msk;
            s.stParam.param[s.stParam.current].L=s.l;
            s.stParam.param[s.stParam.current].D=s.d;
            s.stParam.param[s.stParam.current].R=s.r;
            s.stParam.param[s.stParam.current].B=s.bm;
            (s.stParam.current)++;
          }
          s.y=s.y+1;
          s.l=bit<<1;
          s.d=bit;
          s.r=bit>>1;
          s.bend=1;
          // printf("s.bm:%d\n",s.bm);
        }else{
        while (s.bm || s.rflg==1) {
          if(s.rflg==0){
            s.bm^=s.aB[s.y]=bit=(-s.bm&s.bm); //最も下位の１ビットを抽出
            if(s.stParam.current<MAX){
              s.stParam.param[s.stParam.current].Y=s.y;
              s.stParam.param[s.stParam.current].I=s.si;
              s.stParam.param[s.stParam.current].M=s.msk;
              s.stParam.param[s.stParam.current].L=s.l;
              s.stParam.param[s.stParam.current].D=s.d;
              s.stParam.param[s.stParam.current].R=s.r;
              s.stParam.param[s.stParam.current].B=s.bm;
              (s.stParam.current)++;
            }
            s.y=s.y+1;
            s.l=(s.l|bit)<<1;
            s.d=(s.d|bit);
            s.r=(s.r|bit)>>1;
            s.bend=1;
            break;
          }
          //goto start;
          //NQueen(si,msk,y+1,(l|bit)<<1,d|bit,(r|bit)>>1);
          //ret:
          if(s.rflg==1){ 
            if(s.stParam.current>0){
              s.stParam.current--;
            }
            s.si=s.stParam.param[s.stParam.current].I;
            s.y=s.stParam.param[s.stParam.current].Y;
            s.msk=s.stParam.param[s.stParam.current].M;
            s.l=s.stParam.param[s.stParam.current].L;
            s.d=s.stParam.param[s.stParam.current].D;
            s.r=s.stParam.param[s.stParam.current].R;
            s.bm=s.stParam.param[s.stParam.current].B;
            s.rflg=0;
          }
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
    }else{
      //backtrack2;
      unsigned long j=1;
      // int sum;
      int bit;

      while (j<200000) {
        //  start:
        if(s.rflg==0){
          s.bm=s.msk&~(s.l|s.d|s.r); /* 配置可能フィールド */
          // printf("s.bm:%d s.msk:%lld s.l:%d s.d:%d\n",s.bm,s.msk,s.l,s.d);
        }
        // printf("s.si:%d\n",s.si);
        if (s.y==s.si && !s.bm && s.rflg==0) {
          // printf("s.si:%d\n",s.si);
          // if(!s.bm){
          s.aB[s.y]=s.bm;
          // printf("s.si:%d\n",s.si);
          int sum=symmetryOps_bm(&s);//対称解除法
          // printf("sum:%d\n",sum);
          if(sum!=0){ s.lUnique++; s.lTotal+=sum; } //解を発見
          //        printf("lTotal:%ld\n",s.lTotal);
          // printf("lUnique:%ld\n",s.lUnique);
          // }
        }else{
        if(s.y==0){
          s.aB[0]=bit=(1<<s.BOUND1);
          if(s.stParam.current<MAX){
            s.stParam.param[s.stParam.current].Y=s.y;
            s.stParam.param[s.stParam.current].I=s.si;
            s.stParam.param[s.stParam.current].M=s.msk;
            s.stParam.param[s.stParam.current].L=s.l;
            s.stParam.param[s.stParam.current].D=s.d;
            s.stParam.param[s.stParam.current].R=s.r;
            s.stParam.param[s.stParam.current].B=s.bm;
            (s.stParam.current)++;
          }
          s.y=s.y+1;
          s.l=bit<<1;
          s.d=bit;
          s.r=bit>>1;
          s.bend=1;
          // printf("s.bm:%d\n",s.bm);
        }else{
          // printf("s.bm:%d\n",s.bm);
          while (s.bm || s.rflg==1) {
            if(s.rflg==0){
              s.bm^=s.aB[s.y]=bit=(-s.bm&s.bm); //最も下位の１ビットを抽出
              if(s.stParam.current<MAX){
                s.stParam.param[s.stParam.current].Y=s.y;
                s.stParam.param[s.stParam.current].I=s.si;
                s.stParam.param[s.stParam.current].M=s.msk;
                s.stParam.param[s.stParam.current].L=s.l;
                s.stParam.param[s.stParam.current].D=s.d;
                s.stParam.param[s.stParam.current].R=s.r;
                s.stParam.param[s.stParam.current].B=s.bm;
                (s.stParam.current)++;
              }
              s.y=s.y+1;
              s.l=(s.l|bit)<<1;
              s.d=(s.d|bit);
              s.r=(s.r|bit)>>1;
              s.bend=1;
              break;
            }
            //goto start;
            //NQueen(si,msk,y+1,(l|bit)<<1,d|bit,(r|bit)>>1);
            //ret:
            if(s.rflg==1){ 
              if(s.stParam.current>0){
                s.stParam.current--;
              }
              s.si=s.stParam.param[s.stParam.current].I;
              s.y=s.stParam.param[s.stParam.current].Y;
              s.msk=s.stParam.param[s.stParam.current].M;
              s.l=s.stParam.param[s.stParam.current].L;
              s.d=s.stParam.param[s.stParam.current].D;
              s.r=s.stParam.param[s.stParam.current].R;
              s.bm=s.stParam.param[s.stParam.current].B;
              s.rflg=0;
            }
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
    }
    //uint16_t j=1;
    //unsigned long j=1;

    state[index].si=s.si;
    state[index].id=s.id;
    for (int j=0;j<s.si;j++){
      state[index].aB[j] = s.aB[j];
    }
    state[index].lTotal=s.lTotal;
    // printf("lTotal:%ld\n",s.lTotal);
    state[index].lUnique=s.lUnique;
    state[index].step=s.step;
    state[index].y=s.y;
    state[index].bend=s.bend;
    state[index].rflg=s.rflg;
    for (int j=0;j<s.si;j++){
      state[index].aT[j]=s.aT[j];
      state[index].aS[j]=s.aS[j];
    }
    state[index].stParam=s.stParam;
    state[index].msk=s.msk;
    state[index].l=s.l;
    state[index].d=s.d;
    state[index].r=s.r;
    state[index].bm=s.bm;
    state[index].BOUND1=s.BOUND1;
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
