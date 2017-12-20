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
struct CL_PACKED_KEYWORD globalState {
  long lTotal; // Number of solutinos found so far.
  long lUnique; // Number of solutinos found so far.
};
struct CL_PACKED_KEYWORD queenState {
  int BOUND1;
  int si;
  int aB[MAX];
  // long lTotal; // Number of solutinos found so far.
  int step;
  int y;
  // int startCol; // First column this individual computation was tasked with filling.
  int bm;
  int BOUND2;
  int TOPBIT;
  int ENDBIT;
  int SIDEMASK;
  int LASTMASK;
  // long lUnique; // Number of solutinos found so far.
  int bend;
  int rflg;
  struct STACK stParam;
  int msk;
  int l;
  int d;
  int r;
  int B1;
  int j;
};
void symmetryOps_bm(struct queenState *s,struct globalState *g){
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
      g->lTotal+=2;
      g->lUnique++; 
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
      g->lTotal+=4;
      g->lUnique++;
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
  g->lTotal+=8;
  g->lUnique++;
}
void backTrack1(struct queenState *s,struct globalState *g){
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
    }
    if (s->y==s->si-1&&s->rflg==0){ 
      if(s->bm>0){
        s->aB[s->y]=s->bm;
        g->lTotal+=8; 
        g->lUnique++;
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
    if(s->y==2){
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
void backTrack2(struct queenState *s,struct globalState *g){
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
        symmetryOps_bm(s,g);
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
    if(s->y==2){
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

CL_KERNEL_KEYWORD void place(
    CL_GLOBAL_KEYWORD struct queenState *l,
    CL_GLOBAL_KEYWORD struct globalState *g){
  int index = get_global_id(0);
  struct queenState _l;
  struct globalState _g;
  _l.BOUND1=l[index].BOUND1;
  _l.si= l[index].si;
  for (int i = 0; i < _l.si; i++)
    _l.aB[i]=l[index].aB[i];
  _g.lTotal = g[index].lTotal;
  _l.step      = l[index].step;
  _l.y       = l[index].y;
  _l.bm     = l[index].bm;
  _l.BOUND2    =l[index].BOUND2;
  _l.ENDBIT    =l[index].ENDBIT;
  _l.TOPBIT    =l[index].TOPBIT;
  _l.SIDEMASK    =l[index].SIDEMASK;
  _l.LASTMASK  =l[index].LASTMASK;
  _g.lUnique  =g[index].lUnique;
  _l.bend  =l[index].bend;
  _l.rflg  =l[index].rflg;
  _l.stParam=l[index].stParam;
  //_l.msk= l[index].msk;
  _l.msk= (1<<_l.si)-1;
  _l.l= l[index].l;
  _l.d= l[index].d;
  _l.r= l[index].r;
  _l.B1= l[index].B1;
  _l.j= l[index].j;
  // _l.C2=l[index].C2;
  // _l.C4=l[index].C4;
  // _l.C8=l[index].C8;
   // printf("BOUND1:%d\n",_l.BOUND1);
   // printf("j:%d\n",_l.j);
  // printf("si:%d\n",_l.si);
  // printf("b:step:%d\n",_l.step);
  // printf("y:%d\n",_l.y);
  // printf("startCol:%d\n",_l.startCol);
  // printf("bm:%d\n",_l.bm);
  // printf("BOUND2:%d\n",_l.BOUND2);
  // printf("TOPBIT:%d\n",_l.TOPBIT);
  // printf("ENDBIT:%d\n",_l.ENDBIT);
  // printf("SIDEMASK:%d\n",_l.SIDEMASK);
  // printf("LASTMASK:%d\n",_l.LASTMASK);
  // printf("lUnique:%ld\n",_l.lUnique);
  // printf("bend:%d\n",_l.bend);
  // printf("rflg:%d\n",_l.rflg);
  // printf("msk:%d\n",_l.msk);
  // printf("l:%d\n",_l.l);
  // printf("d:%d\n",_l.d);
  // printf("r:%d\n",_l.r);
  // printf("B1:%d\n",_l.B1);
    int bit;
    if(_l.BOUND1==0 && _l.step !=2){ 
      if(_l.step!=1){
        _l.aB[0]=1;
        _l.TOPBIT=1<<(_l.si-1);
      }
backTrack1(&_l,&_g);
    }else if(_l.BOUND1 !=0 && _l.step !=2){ 
      if(_l.step!=1){
      _l.TOPBIT=1<<(_l.si-1);
      _l.ENDBIT=_l.TOPBIT>>_l.BOUND1;
      _l.SIDEMASK=_l.LASTMASK=(_l.TOPBIT|1);
      }
      if(_l.BOUND1>0&&_l.BOUND2<_l.si-1&&_l.BOUND1<_l.BOUND2){
        if(_l.step!=1){
          for(int i=1;i<_l.BOUND1;i++){
            _l.LASTMASK=_l.LASTMASK|_l.LASTMASK>>1|_l.LASTMASK<<1;
          }
          _l.aB[0]=bit=(1<<_l.BOUND1);
          _l.y=1;_l.l=bit<<1;_l.d=bit;_l.r=bit>>1;
        }
backTrack2(&_l,&_g);
        if(_l.step!=1){
            _l.ENDBIT>>=_l.si;
        }
      }
    }
 // printf("lTotal:%ld\n",_l.lTotal);
l[index].BOUND1=_l.BOUND1;
l[index].si=_l.si;
for(int i=0;i<_l.si;i++){l[index].aB[i]=_l.aB[i];}
g[index].lTotal=_g.lTotal;
if(_l.step==1){
  l[index].step=1;
// l[index].msk=1;
}else{
  l[index].step=2;
// l[index].msk=2;
}
l[index].y=_l.y;
// l[index].startCol=0;
l[index].bm=_l.bm;
l[index].BOUND2=_l.BOUND2;
l[index].ENDBIT=_l.ENDBIT;
l[index].TOPBIT=_l.TOPBIT;
l[index].SIDEMASK=_l.SIDEMASK;
l[index].LASTMASK=_l.LASTMASK;
g[index].lUnique=_g.lUnique;
l[index].bend=_l.bend;
l[index].rflg=_l.rflg;
l[index].stParam=_l.stParam;
 // printf("m:step:%d\n",l[index].msk);
l[index].l=_l.l;
l[index].d=_l.d;
l[index].r=_l.r;
l[index].B1=_l.B1;
l[index].j=_l.j;
// l[index].C2=_l.C2;
// l[index].C4=_l.C4;
// l[index].C8=_l.C8;
}
#ifdef GCC_STYLE
int main(){
  int target=17;
  /**********/
  struct queenState inProgress[MAX*MAX];
  struct globalState gProgress[MAX*MAX];
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
      gProgress[i*si+j].lTotal=0;
      gProgress[i*si+j].lUnique=0;
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
      place(&inProgress[i*si+j],&gProgress[i*si+j]);
      }
    }
    for(int i=0;i<si;i++){
      for(int j=0;j<si;j++){ // N
        gTotal+=gProgress[i*si+j].lTotal;
        gUnique+=gProgress[i*si+j].lUnique;
      }
    }
  /**********/
      printf("%2d:%18lu%18lu\n", si,gTotal,gUnique);
  }
  return 0;
}
#endif
