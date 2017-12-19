//  単体で動かすときは以下のコメントを外す
#define GCC_STYLE
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
  // int startCol; // First column this individual computation was tasked with filling.
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
  int j;
  int k;
  long lt;
  // long C2;
  // long C4;
  // long C8;
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
      // s->C2++;
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
      // s->C4++;
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
  // s->C8++;
}
void backTrack1(struct queenState *s){
  //printf("backtrack1:start\n");
  int bit;
  if(s->step!=1){
    s->y=1;s->l=(1)<<1;s->d=(1);s->r=(1>>1);
    //printf("s->y=1;s->l=(1)<<1;s->d=(1);s->r=(1>>1);\n");
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
      //printf("s->bm=s->msk&~(s->l|s->d|s->r);\n");
    }
    if (s->y==s->si-1&&s->rflg==0){ 
      //printf("if (s->y==s->si-1&&s->rflg==0){\n");
      if(s->bm>0){
        //printf("if(s->bm>0){\n");
        s->aB[s->y]=s->bm;
        s->lTotal+=8;
        s->lt+=8;
        s->lUnique++;
        // s->C8++;
      }
    }else{
      //printf("}else{#if (s->y==s->si-1&&s->rflg==0){\n");
      if(s->y>1&&(1<<s->y)<s->B1 && s->rflg==0){   
        s->bm&=~2;
        //printf("s->bm&=~2;\n");
      }
      if(s->y==1 && s->j>=0 && s->rflg==0){
        //printf("if(s->y==1 && s->j>=0 && s->rflg==0){\n");
        if(s->bm & (1<<s->j)){ 
          //printf("if(s->bm & (1<<s->j)){\n");
          s->aB[s->y]=bit=1<<s->j; 
          s->B1=bit;
        }else{ 
          //printf("}else{ #if(s->bm & (1<<s->j)){\n");
          return;
        }
          //printf("if(s->rflg==0){#inParam\n");
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
          // inParam(s);
          s->l=(s->l|bit)<<1; s->d=(s->d|bit); s->r=(s->r|bit)>>1;
          s->y++; 
          continue;
      }else if(s->y==2 && s->k>=0 && s->rflg==0){
        //printf("if(s->y==1 && s->j>=0 && s->rflg==0){\n");
        if(s->bm & (1<<s->k)){ 
          //printf("if(s->bm & (1<<s->j)){\n");
          s->aB[s->y]=bit=1<<s->k; 
          // s->B1=bit;
        }else{ 
          //printf("}else{ #if(s->bm & (1<<s->j)){\n");
          return;
        }
          //printf("if(s->rflg==0){#inParam\n");
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
          // inParam(s);
          s->l=(s->l|bit)<<1; s->d=(s->d|bit); s->r=(s->r|bit)>>1;
          s->y++; 
          continue;
      }else {
        //printf("}else{ #if(s->y>1&&(1<<s->y)<s->B1){\n");
        while(s->bm || s->rflg==1){
          //printf("while(s->bm || s->rflg==1){\n");
          if(s->rflg==0){
            //printf("inparam\n");
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
            // inParam(s);
            s->l=(s->l|bit)<<1; s->d=(s->d|bit); s->r=(s->r|bit)>>1;
            s->y++; s->bend=1;
            break;
          }else{ // s->rflg==1
            //printf("outparam\n");
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
            // outParam(s);
            s->rflg=0;
          }
        }
      }
      if(s->bend==1 && s->rflg==0){
        //printf("if(s->bend==1){\n");
        s->bend=0;
        continue;
      }
    }
    if(s->y==3){
      //printf("if(s->y==1){\n");
      s->step=2;
      return;
    }else{
      //printf("}else{#if(s->y==1){");
      s->rflg=1;
    }
    j++;
  }
}
void backTrack2(struct queenState *s){
  // printf("backtrack2:start\n");
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
      // printf("s->bm=s->msk&~(s->l|s->d|s->r);\n");
    }
    if (s->y==s->si-1 && s->rflg==0){
      // printf("if (s->y==s->si-1&& s->bm){\n");
      if(s->bm>0 && (s->bm&s->LASTMASK)==0){
        // printf("if((s->bm&s->LASTMASK)==0){#symmetryOps\n");
        s->aB[s->y]=s->bm;
        symmetryOps_bm(s);
      }
    }else{
      if(s->y<s->BOUND1&&s->rflg==0){
        // printf("if(s->y<s->BOUND1){\n");
        s->bm&=~s->SIDEMASK; 
      }else if(s->y==s->BOUND2&&s->rflg==0){
        // printf("}else if(s->y==s->BOUND2){\n");
        if((s->d&s->SIDEMASK)==0&&s->rflg==0){ 
          // printf("if((s->d&s->SIDEMASK)==0){\n");
          s->rflg=1;
        }
        if((s->d&s->SIDEMASK)!=s->SIDEMASK&&s->rflg==0){ 
          // printf("if((s->d&s->SIDEMASK)!=s->SIDEMASK){\n");
          s->bm&=s->SIDEMASK; 
        }
      }

      if(s->y==1 && s->j>=0 && s->rflg==0){
        // printf("if(s->y==1 && s->j>=0 && s->rflg==0){\n");
        if(s->bm & (1<<s->j)){ 
          // printf("if(s->bm & (1<<s->j)){\n");
          s->aB[s->y]=bit=1<<s->j; 

        } else{ 
          // printf("} else{#if(s->bm & (1<<s->j)){");
          return;
        }
          // printf("inparam\n");
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
          // inParam(s);
          s->l=(s->l|bit)<<1; s->d=(s->d|bit); s->r=(s->r|bit)>>1;
          s->y++; 
          continue;
      }else if(s->y==2 && s->k>=0 && s->rflg==0){
        // printf("if(s->y==1 && s->j>=0 && s->rflg==0){\n");
        if(s->bm & (1<<s->k)){ 
          // printf("if(s->bm & (1<<s->j)){\n");
          s->aB[s->y]=bit=1<<s->k; 

        } else{ 
          // printf("} else{#if(s->bm & (1<<s->j)){");
          return;
        }
          // printf("inparam\n");
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
          // inParam(s);
          s->l=(s->l|bit)<<1; s->d=(s->d|bit); s->r=(s->r|bit)>>1;
          s->y++; 
          continue;
      }else{
        while(s->bm || s->rflg==1){
          // printf("while(s->bm || s->rflg==1){\n");
          if(s->rflg==0){
            // printf("inparam\n");
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
            // inParam(s);
            s->l=(s->l|bit)<<1; s->d=(s->d|bit); s->r=(s->r|bit)>>1;
            s->y++; s->bend=1;
            break;
          }else{
            // printf("outparam\n");
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
            // outParam(s);
            s->rflg=0;
          }
        }//end while
      }
      if(s->bend==1 && s->rflg==0){
        // printf("if(s->bend==1){\n");
        s->bend=0;
        continue;
      }
    }
    if(s->y==3){
      // printf("if(s->y==1){\n");
      s->step=2;
      return;
    }else{
      // printf("}else{#if(s->y==1){");
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
  s.bm     = state[index].bm;
  s.BOUND2    =state[index].BOUND2;
  s.ENDBIT    =state[index].ENDBIT;
  s.TOPBIT    =state[index].TOPBIT;
  s.SIDEMASK    =state[index].SIDEMASK;
  s.LASTMASK  =state[index].LASTMASK;
  s.lUnique  =state[index].lUnique;
  s.bend  =state[index].bend;
  s.rflg  =state[index].rflg;
  s.stParam=state[index].stParam;
  //s.msk= state[index].msk;
  s.msk= (1<<s.si)-1;
  s.l= state[index].l;
  s.d= state[index].d;
  s.r= state[index].r;
  s.B1= state[index].B1;
  s.j= state[index].j;
  s.k= state[index].k;
  s.lt= state[index].lt;
  // s.C2=state[index].C2;
  // s.C4=state[index].C4;
  // s.C8=state[index].C8;
   // printf("BOUND1:%d\n",s.BOUND1);
   // printf("j:%d\n",s.j);
  // printf("si:%d\n",s.si);
  // printf("b:step:%d\n",s.step);
  // printf("y:%d\n",s.y);
  // printf("startCol:%d\n",s.startCol);
  // printf("bm:%d\n",s.bm);
  // printf("BOUND2:%d\n",s.BOUND2);
  // printf("TOPBIT:%d\n",s.TOPBIT);
  // printf("ENDBIT:%d\n",s.ENDBIT);
  // printf("SIDEMASK:%d\n",s.SIDEMASK);
  // printf("LASTMASK:%d\n",s.LASTMASK);
  // printf("lUnique:%ld\n",s.lUnique);
  // printf("bend:%d\n",s.bend);
  // printf("rflg:%d\n",s.rflg);
  // printf("msk:%d\n",s.msk);
  // printf("l:%d\n",s.l);
  // printf("d:%d\n",s.d);
  // printf("r:%d\n",s.r);
  // printf("B1:%d\n",s.B1);
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
 // printf("lTotal:%ld\n",s.lTotal);
state[index].BOUND1=s.BOUND1;
state[index].si=s.si;
for(int i=0;i<s.si;i++){state[index].aB[i]=s.aB[i];}
state[index].lTotal=s.lTotal;
if(s.step==1){
  state[index].step=1;
// state[index].msk=1;
}else{
  state[index].step=2;
// state[index].msk=2;
}
state[index].y=s.y;
// state[index].startCol=0;
state[index].bm=s.bm;
state[index].BOUND2=s.BOUND2;
state[index].ENDBIT=s.ENDBIT;
state[index].TOPBIT=s.TOPBIT;
state[index].SIDEMASK=s.SIDEMASK;
state[index].LASTMASK=s.LASTMASK;
state[index].lUnique=s.lUnique;
state[index].bend=s.bend;
state[index].rflg=s.rflg;
state[index].stParam=s.stParam;
 // printf("m:step:%d\n",state[index].msk);
state[index].l=s.l;
state[index].d=s.d;
state[index].r=s.r;
state[index].B1=s.B1;
state[index].j=s.j;
state[index].k=s.k;
state[index].lt=s.lt;
// state[index].C2=s.C2;
// state[index].C4=s.C4;
// state[index].C8=s.C8;
}
#ifdef GCC_STYLE
int main(){
  int target=17;
  /**********/
  struct queenState inProgress[MAX*MAX*MAX];
  /**********/
  printf("%s\n"," N:          Total        Unique\n");
  for(int si=4;si<=target;si++){
    long gTotal=0;
    long gUnique=0;
    for(int i=0,B2=si-1;i<si;i++,B2--){ // N
      for(int j=0;j<si;j++){ // N
        for(int k=0;k<si;k++){
          inProgress[i*si*si+j*si+k].si=si;
          inProgress[i*si*si+j*si+k].B1=-1;
          inProgress[i*si*si+j*si+k].BOUND1=i;
          inProgress[i*si*si+j*si+k].BOUND2=B2;
          inProgress[i*si*si+j*si+k].j=j;
          inProgress[i*si*si+j*si+k].k=k;
          inProgress[i*si*si+j*si+k].ENDBIT=0;
          inProgress[i*si*si+j*si+k].TOPBIT=1<<(si-1);
          inProgress[i*si*si+j*si+k].SIDEMASK=0;
          inProgress[i*si*si+j*si+k].LASTMASK=0;
          for (int m=0;m< si;m++){ inProgress[i*si*si+j*si+k].aB[m]=m;}
          inProgress[i*si*si+j*si+k].lTotal=0;
          inProgress[i*si*si+j*si+k].lUnique=0;
          inProgress[i*si*si+j*si+k].step=0;
          inProgress[i*si*si+j*si+k].y=0;
          inProgress[i*si*si+j*si+k].bend=0;
          inProgress[i*si*si+j*si+k].rflg=0;
          for (int m=0;m<si;m++){
            inProgress[i*si*si+j*si+k].stParam.param[m].Y=0;
            inProgress[i*si*si+j*si+k].stParam.param[m].I=si;
            inProgress[i*si*si+j*si+k].stParam.param[m].M=0;
            inProgress[i*si*si+j*si+k].stParam.param[m].L=0;
            inProgress[i*si*si+j*si+k].stParam.param[m].D=0;
            inProgress[i*si*si+j*si+k].stParam.param[m].R=0;
            inProgress[i*si*si+j*si+k].stParam.param[m].B=0;
          }
          inProgress[i*si*si+j*si+k].stParam.current=0;
          inProgress[i*si*si+j*si+k].msk=(1<<si)-1;
          inProgress[i*si*si+j*si+k].l=0;
          inProgress[i*si*si+j*si+k].d=0;
          inProgress[i*si*si+j*si+k].r=0;
          inProgress[i*si*si+j*si+k].bm=0;
          // inProgress[i*si*si+j*si+k].C2=0;
          // inProgress[i*si*si+j*si+k].C4=0;
          // inProgress[i*si*si+j*si+k].C8=0;
          place(&inProgress[i*si*si+j*si+k]);
        }
      }
    }
    for(int i=0;i<si;i++){
      for(int j=0;j<si;j++){ // N
        for(int k=0;k<si;k++){
          // gTotal+=inProgress[i*si*si+j*si+k].C2*2+inProgress[i*si*si+j*si+k].C4*4+inProgress[i*si*si+j*si+k].C8*8;
          gTotal+=inProgress[i*si*si+j*si+k].lTotal;
          gUnique+=inProgress[i*si*si+j*si+k].lUnique;
          // gUnique+=inProgress[i*si*si+j*si+k].C2+inProgress[i*si*si+j*si+k].C4+inProgress[i*si*si+j*si+k].C8;
        }
      }
    }
  /**********/
      printf("%2d:%18lu%18lu\n", si,gTotal,gUnique);
  }
  return 0;
}
#endif
