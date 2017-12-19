//  単体で動かすときは以下のコメントを外す
// #define GCC_STYLE
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
  /***********/
  int j;
  long lt;
  /***********/
};
void symmetryOps_bm(struct queenState *s){
  int nEquiv;
  int own,ptn,you,bit;
  //90度回転
  if(s->aB[s->BOUND2]==1){ 
    own=1;
    ptn=2;
    while(own<=(s->si-1)){
      bit=1; 
      you=s->si-1;
      while((s->aB[you]!=ptn)&&(s->aB[own]>=bit)){ 
        bit<<=1; 
        you--; 
      }//end while
      if(s->aB[own]>bit){ 
        return; 
      }//end if 
      if(s->aB[own]<bit){
        printf("");
        break; 
      }//end if
      own++; 
      ptn<<=1;
    }//end while
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>s->si-1){ 
      s->lTotal+=2;
      s->lt+=2;
      s->lUnique++; 
      return;
    }//end if
  }//end if
  //180度回転
  if(s->aB[s->si-1]==s->ENDBIT){ 
    own=1; 
    you=s->si-2;
    while(own<=s->si-1){ 
      bit=1; 
      ptn=s->TOPBIT;
      while((s->aB[you]!=ptn)&&(s->aB[own]>=bit)){ 
        bit<<=1; 
        ptn>>=1; 
      }
      if(s->aB[own]>bit){ 
        return ; 
      } 
      if(s->aB[own]<bit){ 
        break; 
      }
      own++; 
      you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>s->si-1){ 
      s->lTotal+=4;
      s->lt+=4;
      s->lUnique++;
      return ;
    }
  }
  //270度回転
  if(s->aB[s->BOUND1]==s->TOPBIT){ 
    own=1; 
    ptn=s->TOPBIT>>1;
    while(own<=s->si-1){ 
      bit=1; 
      you=0;
      while((s->aB[you]!=ptn)&&(s->aB[own]>=bit)){ 
        bit<<=1; 
        you++; 
      }
      if(s->aB[own]>bit){ 
        return ; 
      } 
      if(s->aB[own]<bit){ 
        break; 
      }
      own++; 
      ptn>>=1;
    }
  }
  s->lTotal+=8;
  s->lt+=8;
  s->lUnique++;
}
void backTrack1(struct queenState *s){
  int bit;
  if(s->step!=1){
    s->y=1;s->l=(1)<<1;s->d=(1);s->r=(1>>1);
  }
  unsigned long j=1;
  while(1){
#ifdef GCC_STYLE
#else
    if(j==100000){
      s->step=1;
      return;
    }
#endif
    if(s->rflg==0){
      s->bm=s->msk&~(s->l|s->d|s->r); 
      if (s->y==s->si-1){ 
        if(s->bm){
          s->aB[s->y]=s->bm;
          s->lTotal+=8;
          s->lt+=8;
          s->lUnique++;
        }
      }
    /******************************/
      if(s->y==1 && s->j>=0){
        if(s->bm & (1<<s->j)){ 
          s->aB[s->y]=bit=1<<s->j; 
          s->B1=bit;
        }else{ 
          return;
        }
        if(s->stParam.current<MAX){
          s->stParam.param[s->stParam.current].Y=s->y;
          s->stParam.param[s->stParam.current].I=s->si;
          s->stParam.param[s->stParam.current].M=s->msk;
          s->stParam.param[s->stParam.current].L=s->l;
          s->stParam.param[s->stParam.current].D=s->d;
          s->stParam.param[s->stParam.current].R=s->r;
          s->stParam.param[s->stParam.current].B=s->bm;
          (s->stParam.current)++;
        }
        s->l=(s->l|bit)<<1; s->d=(s->d|bit); s->r=(s->r|bit)>>1;
        s->y++; 
        continue;
      }
    }
    /******************************/
    if(s->y>1&&(1<<s->y)<s->B1){   
      s->bm&=~2;
    }
    while(s->bm || s->rflg==1){
      if(s->rflg==0){
        s->bm^=s->aB[s->y]=bit=(-s->bm&s->bm);
        if(s->y==1){
          s->B1=bit;
        }
        if(s->stParam.current<MAX){
          s->stParam.param[s->stParam.current].Y=s->y;
          s->stParam.param[s->stParam.current].I=s->si;
          s->stParam.param[s->stParam.current].M=s->msk;
          s->stParam.param[s->stParam.current].L=s->l;
          s->stParam.param[s->stParam.current].D=s->d;
          s->stParam.param[s->stParam.current].R=s->r;
          s->stParam.param[s->stParam.current].B=s->bm;
          (s->stParam.current)++;
        }
        s->l=(s->l|bit)<<1; s->d=(s->d|bit); s->r=(s->r|bit)>>1;
        s->y++; s->bend=1;
        break;
      }else{ // s->rflg==1
        if(s->stParam.current>0){
          s->stParam.current--;
        }
        s->si=s->stParam.param[s->stParam.current].I;
        s->y=s->stParam.param[s->stParam.current].Y;
        s->msk=s->stParam.param[s->stParam.current].M;
        s->l=s->stParam.param[s->stParam.current].L;
        s->d=s->stParam.param[s->stParam.current].D;
        s->r=s->stParam.param[s->stParam.current].R;
        s->bm=s->stParam.param[s->stParam.current].B;
        s->rflg=0;
      }
    }
    if(s->bend==1){
      s->bend=0;
      continue;
    }
    /***************/
    //if(s->y==1){
    if(s->y==2){
    /***************/
      s->step=2;
      return;
    }else{
      s->rflg=1;
    }
    j++;
  }
}
void backTrack2(struct queenState *s){
  int bit;
  unsigned long j=1;
  while (1){
#ifdef GCC_STYLE
#else
    if(j==100000){
      s->step=1;
      return;
    }
#endif
    if(s->rflg==0){
      s->bm=s->msk&~(s->l|s->d|s->r); 
      if(s->y==s->si-1&&s->bm){
        if((s->bm&s->LASTMASK)==0){
          s->aB[s->y]=s->bm;
          symmetryOps_bm(s);
        }
      }
      if(s->y<s->BOUND1){
        s->bm&=~s->SIDEMASK; 
      }else if(s->y==s->BOUND2){
        if((s->d&s->SIDEMASK)==0){ 
          s->rflg=1;
        }
        if((s->d&s->SIDEMASK)!=s->SIDEMASK){ 
          s->bm&=s->SIDEMASK; 
        }
      }
  /***************************/
      if(s->y==1){
        if(s->j>=0){
          if(s->bm & (1<<s->j)){ 
            s->aB[s->y]=bit=1<<s->j; 
          }else{ 
            return;
          }
          if(s->stParam.current<MAX){
            s->stParam.param[s->stParam.current].Y=s->y;
            s->stParam.param[s->stParam.current].I=s->si;
            s->stParam.param[s->stParam.current].M=s->msk;
            s->stParam.param[s->stParam.current].L=s->l;
            s->stParam.param[s->stParam.current].D=s->d;
            s->stParam.param[s->stParam.current].R=s->r;
            s->stParam.param[s->stParam.current].B=s->bm;
            (s->stParam.current)++;
          }
          s->l=(s->l|bit)<<1; s->d=(s->d|bit); s->r=(s->r|bit)>>1;
          s->y++; 
          continue;
        }
      }
  /***************************/
    }
    while(s->bm || s->rflg==1){
      if(s->rflg==0){
        s->bm^=s->aB[s->y]=bit=(-s->bm&s->bm); 
        if(s->stParam.current<MAX){
          s->stParam.param[s->stParam.current].Y=s->y;
          s->stParam.param[s->stParam.current].I=s->si;
          s->stParam.param[s->stParam.current].M=s->msk;
          s->stParam.param[s->stParam.current].L=s->l;
          s->stParam.param[s->stParam.current].D=s->d;
          s->stParam.param[s->stParam.current].R=s->r;
          s->stParam.param[s->stParam.current].B=s->bm;
          (s->stParam.current)++;
        }
        s->l=(s->l|bit)<<1; s->d=(s->d|bit); s->r=(s->r|bit)>>1;
        s->y++; s->bend=1;
        break;
      }else{
        if(s->stParam.current>0){
          s->stParam.current--;
        }
        s->si=s->stParam.param[s->stParam.current].I;
        s->y=s->stParam.param[s->stParam.current].Y;
        s->msk=s->stParam.param[s->stParam.current].M;
        s->l=s->stParam.param[s->stParam.current].L;
        s->d=s->stParam.param[s->stParam.current].D;
        s->r=s->stParam.param[s->stParam.current].R;
        s->bm=s->stParam.param[s->stParam.current].B;
        s->rflg=0;
      }
    }
    if(s->bend==1){
      s->bend=0;
      continue;
    }
    // }
    /**************************/
    if(s->y==2){
    /**************************/
      s->step=2;
      return;
    }else{
      s->rflg=1;
    }
    j++;
  } 
}

CL_KERNEL_KEYWORD void place(CL_GLOBAL_KEYWORD struct queenState *state){
  int index = get_global_id(0);
  struct queenState s;
  s.BOUND1=state[index].BOUND1;
  s.si= state[index].si;
  for (int i = 0; i < s.si; i++)
    s.aB[i]=state[index].aB[i];
  s.lTotal = state[index].lTotal;
  s.step      = state[index].step;
  s.y       = state[index].y;
  s.startCol  = state[index].startCol;
  s.bm     = state[index].bm;
  s.BOUND2    =state[index].BOUND2;
  s.TOPBIT    =state[index].TOPBIT;
  s.ENDBIT    =state[index].ENDBIT;
  s.SIDEMASK    =state[index].SIDEMASK;
  s.LASTMASK  =state[index].LASTMASK;
  s.lUnique  =state[index].lUnique;
  s.bend  =state[index].bend;
  s.rflg  =state[index].rflg;
  s.stParam=state[index].stParam;
  //s.msk= state[index].msk;
  s.msk=(1<<s.si)-1;
  s.l= state[index].l;
  s.d= state[index].d;
  s.r= state[index].r;
  s.B1= state[index].B1;
  s.j= state[index].j;
  s.lt=state[index].lt;
    int bit;
    if(s.BOUND1==0 && s.step !=2){ 
      if(s.step!=1){
        s.aB[0]=1;
        s.TOPBIT=1<<(s.si-1);
      }
backTrack1(&s);
    }else if(s.BOUND1 !=0 && s.step !=2){ 
      if(s.step!=1){
      s.TOPBIT=1<<(s.si-1);
      s.ENDBIT=s.TOPBIT>>s.BOUND1;
      s.SIDEMASK=s.LASTMASK=(s.TOPBIT|1);
      }
      if(s.BOUND1>0&&s.BOUND2<s.si-1&&s.BOUND1<s.BOUND2){
        if(s.step!=1){
          for(int i=1;i<s.BOUND1;i++){
            s.LASTMASK=s.LASTMASK|s.LASTMASK>>1|s.LASTMASK<<1;
          }
          s.aB[0]=bit=(1<<s.BOUND1);
          s.y=1;s.l=bit<<1;s.d=bit;s.r=bit>>1;
        }
backTrack2(&s);
        if(s.step!=1){
            s.ENDBIT>>=s.si;
        }
      }
    }
state[index].BOUND1=s.BOUND1;
state[index].si=s.si;
for(int i=0;i<s.si;i++){state[index].aB[i]=s.aB[i];}
state[index].lTotal=s.lTotal;
if(s.step==1){
  state[index].step=1;
  state[index].msk=1;
}else{
  state[index].step=2;
  state[index].msk=2;
}
state[index].y=s.y;
state[index].startCol=s.startCol;
state[index].bm=s.bm;
state[index].BOUND2=s.BOUND2;
state[index].TOPBIT=s.TOPBIT;
state[index].ENDBIT=s.ENDBIT;
state[index].SIDEMASK=s.SIDEMASK;
state[index].LASTMASK=s.LASTMASK;
state[index].lUnique=s.lUnique;
state[index].bend=s.bend;
state[index].rflg=s.rflg;
state[index].stParam=s.stParam;
// state[index].msk=s.msk;
state[index].l=s.l;
state[index].d=s.d;
state[index].r=s.r;
state[index].B1=s.B1;
/******************/
state[index].j=s.j;
state[index].lt=s.lt;
/******************/
}
#ifdef GCC_STYLE
int main(){
  int target=17;
  /**********/
  struct queenState inProgress[MAX*MAX];
  /**********/
  printf("%s\n"," N:          Total        Unique\n");
  for(int si=4;si<=target;si++){
    long gTotal=0;
    long gUnique=0;
    for(int i=0,B2=si-1;i<si;i++,B2--){ // N
      for(int j=0;j<si;j++){ // N
      inProgress[i*si+j].si=si;
      inProgress[i*si+j].B1=-1;
      inProgress[i*si+j].BOUND1=i;
      inProgress[i*si+j].BOUND2=B2;
      inProgress[i*si+j].j=j;
      inProgress[i*si+j].ENDBIT=0;
      inProgress[i*si+j].TOPBIT=1<<(si-1);
      inProgress[i*si+j].SIDEMASK=0;
      inProgress[i*si+j].LASTMASK=0;
      for (int m=0;m< si;m++){ inProgress[i*si+j].aB[m]=m;}
      inProgress[i*si+j].lTotal=0;
      inProgress[i*si+j].lUnique=0;
      inProgress[i*si+j].step=0;
      inProgress[i*si+j].y=0;
      inProgress[i*si+j].bend=0;
      inProgress[i*si+j].rflg=0;
      for (int m=0;m<si;m++){
        inProgress[i*si+j].stParam.param[m].Y=0;
        inProgress[i*si+j].stParam.param[m].I=si;
        inProgress[i*si+j].stParam.param[m].M=0;
        inProgress[i*si+j].stParam.param[m].L=0;
        inProgress[i*si+j].stParam.param[m].D=0;
        inProgress[i*si+j].stParam.param[m].R=0;
        inProgress[i*si+j].stParam.param[m].B=0;
      }
      inProgress[i*si+j].stParam.current=0;
      inProgress[i*si+j].msk=(1<<si)-1;
      inProgress[i*si+j].l=0;
      inProgress[i*si+j].d=0;
      inProgress[i*si+j].r=0;
      inProgress[i*si+j].bm=0;
      place(&inProgress[i*si+j]);
      }
    }
    for(int i=0;i<si;i++){
      for(int j=0;j<si;j++){ // N
        gTotal+=inProgress[i*si+j].lTotal;
        gUnique+=inProgress[i*si+j].lUnique;
      }
    }
  /**********/
      printf("%2d:%18lu%18lu\n", si,gTotal,gUnique);
  }
  return 0;
}
#endif
