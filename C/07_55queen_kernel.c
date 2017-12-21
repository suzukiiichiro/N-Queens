//  単体で動かすときは以下のコメントを外す
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
#define MAX 17  
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
  int k;
  int j;
  int BOUND1;
  int BOUND2;
  int si;
  int B1;
  int step;
};
struct CL_PACKED_KEYWORD queenState {
  int aB[MAX];
  int y;
  // int startCol; // First column this individual computation was tasked with filling.
  int bm;
  int TOPBIT;
  int ENDBIT;
  int SIDEMASK;
  int LASTMASK;
  int bend;
  int rflg;
  struct STACK stParam;
  int msk;
  int l;
  int d;
  int r;
};
void symmetryOps_bm(struct queenState *s,struct globalState *g){
  int nEquiv;
  int own,ptn,you,bit;
  //90度回転
  if(s->aB[g->BOUND2]==1){ 
    own=1;
    ptn=2;
    while(own<=(g->si-1)){
      bit=1; 
      you=g->si-1;
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
    if(own>g->si-1){ 
      g->lTotal+=2;
      g->lUnique++; 
      return;
    }//end if
  }//end if
  //180度回転
  if(s->aB[g->si-1]==s->ENDBIT){ 
    own=1; 
    you=g->si-2;
    while(own<=g->si-1){ 
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
    if(own>g->si-1){ 
      g->lTotal+=4;
      g->lUnique++;
      return ;
    }
  }
  //270度回転
  if(s->aB[g->BOUND1]==s->TOPBIT){ 
    own=1; 
    ptn=s->TOPBIT>>1;
    while(own<=g->si-1){ 
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
  if(g->step!=1){
    s->y=1;s->l=(1)<<1;s->d=(1);s->r=(1>>1);
  }
  g->step=0;
  unsigned long j=1;
  while(j>0){
#ifdef GCC_STYLE
#else
    if(j==100000){
      printf("b1_over\n");
      g->step=1;
      return;
    }
#endif
    if(s->rflg==0){
      s->bm=s->msk&~(s->l|s->d|s->r); 
    }
    if (s->y==g->si-1&&s->rflg==0){ 
      if(s->bm>0){
        s->aB[s->y]=s->bm;
        g->lTotal+=8; 
        g->lUnique++;
      }
    }else{
      // printf("}else{#if (s->y==g->si-1&&s->rflg==0){\n");
      if(s->y>1&&(1<<s->y)<g->B1 && s->rflg==0){   
        s->bm&=~2;
        // printf("s->bm&=~2;\n");
      }
      if(s->y==1 && g->j>=0){
        if(s->rflg==0){
        // printf("if(s->y==1 && s->j>=0 && s->rflg==0){\n");
        if(s->bm & (1<<g->j)){ 
          // printf("if(s->bm & (1<<s->j)){\n");
          s->aB[s->y]=bit=1<<g->j; 
          g->B1=bit;
        }else{ 
          // printf("}else{ #if(s->bm & (1<<s->j)){\n");
          g->step=2;
          return;
        }
          // printf("if(s->rflg==0){#inParam\n");
          if(s->stParam.current<MAX){
            s->stParam.param[s->stParam.current].Y=s->y;
            s->stParam.param[s->stParam.current].I=g->si;
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
            // printf("b1_outparam\n");
            if(s->stParam.current>0){
              s->stParam.current--;
            }
            g->si=s->stParam.param[s->stParam.current].I;
            s->y=s->stParam.param[s->stParam.current].Y;
            s->msk=s->stParam.param[s->stParam.current].M;
            s->l=s->stParam.param[s->stParam.current].L;
            s->d=s->stParam.param[s->stParam.current].D;
            s->r=s->stParam.param[s->stParam.current].R;
            s->bm=s->stParam.param[s->stParam.current].B;
            // outParam(s);
            s->rflg=0;
        
        }
      }else if(s->y==2 && g->k>=0){
        if(s->rflg==0){
        // printf("if(s->y==1 && s->j>=0 && s->rflg==0){\n");
        if(s->bm & (1<<g->k)){ 
          // printf("if(s->bm & (1<<s->j)){\n");
          s->aB[s->y]=bit=1<<g->k; 
          // s->B1=bit;
        }else{ 
          // printf("}else{ #if(s->bm & (1<<s->j)){\n");
          g->step=2;
          return;
        }
          // printf("if(s->rflg==0){#inParam\n");
          if(s->stParam.current<MAX){
            s->stParam.param[s->stParam.current].Y=s->y;
            s->stParam.param[s->stParam.current].I=g->si;
            // s->stParam.param[s->stParam.current].M=s->msk;
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
            // printf("b1_y2outparam\n");
            if(s->stParam.current>0){
              s->stParam.current--;
            }
            g->si=s->stParam.param[s->stParam.current].I;
            s->y=s->stParam.param[s->stParam.current].Y;
            // s->msk=s->stParam.param[s->stParam.current].M;
            s->l=s->stParam.param[s->stParam.current].L;
            s->d=s->stParam.param[s->stParam.current].D;
            s->r=s->stParam.param[s->stParam.current].R;
            s->bm=s->stParam.param[s->stParam.current].B;
            // outParam(s);
            s->rflg=0;
        }
      }else {
        // printf("}else{ #if(s->y>1&&(1<<s->y)<s->B1){\n");
        while(s->bm || s->rflg==1){
          // printf("while(s->bm || s->rflg==1){\n");
          if(s->rflg==0){
            // printf("inparam\n");
            s->bm^=s->aB[s->y]=bit=(-s->bm&s->bm);
            if(s->stParam.current<MAX){
              s->stParam.param[s->stParam.current].Y=s->y;
              s->stParam.param[s->stParam.current].I=g->si;
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
            // printf("outparam\n");
            if(s->stParam.current>0){
              s->stParam.current--;
            }
            g->si=s->stParam.param[s->stParam.current].I;
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
        // printf("if(s->bend==1){\n");
        s->bend=0;
        continue;
      }
    }
    if(s->y<=3){
      // printf("if(s->y==1){\n");
      g->step=2;
      return;
    }else{
      // printf("}else{#if(s->y==1){");
      s->rflg=1;
    }
    j++;
  }   
  g->step=2;
}
void backTrack2(struct queenState *s,struct globalState *g){
  int bit;
  unsigned long j=1;
  g->step=0;
  while (j>0){
#ifdef GCC_STYLE
#else
    if(j==100000){
      g->step=1;
      printf("b2_over\n");
      return;
    }
#endif
    if(s->rflg==0){
      s->bm=s->msk&~(s->l|s->d|s->r); 
      // printf("s->bm=s->msk&~(s->l|s->d|s->r);\n");
    }
    if (s->y==g->si-1 && s->rflg==0){
      // printf("if (s->y==g->si-1&& s->bm){\n");
      if(s->bm>0 && (s->bm&s->LASTMASK)==0){
        // printf("if((s->bm&s->LASTMASK)==0){#symmetryOps\n");
        s->aB[s->y]=s->bm;
        symmetryOps_bm(s,g);
      }
    }else{
      if(s->y<g->BOUND1&&s->rflg==0){
        // printf("if(s->y<g->BOUND1){\n");
        s->bm&=~s->SIDEMASK; 
      }else if(s->y==g->BOUND2&&s->rflg==0){
        // printf("}else if(s->y==g->BOUND2){\n");
        if((s->d&s->SIDEMASK)==0&&s->rflg==0){ 
          // printf("if((s->d&s->SIDEMASK)==0){\n");
          s->rflg=1;
        }
        if((s->d&s->SIDEMASK)!=s->SIDEMASK&&s->rflg==0){ 
          // printf("if((s->d&s->SIDEMASK)!=s->SIDEMASK){\n");
          s->bm&=s->SIDEMASK; 
        }
      }

      if(s->y==1 && g->j>=0){
        if(s->rflg==0){
        // printf("if(s->y==1 && s->j>=0 && s->rflg==0){\n");
        if(s->bm & (1<<g->j)){ 
          // printf("if(s->bm & (1<<s->j)){\n");
          s->aB[s->y]=bit=1<<g->j; 
        } else{ 
          // printf("} else{#if(s->bm & (1<<s->j)){");
          g->step=2;
          return;
        }
          // printf("inparam\n");
          if(s->stParam.current<MAX){
            s->stParam.param[s->stParam.current].Y=s->y;
            s->stParam.param[s->stParam.current].I=g->si;
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
            // printf("b2_y1outparam\n");
            if(s->stParam.current>0){
              s->stParam.current--;
            }
            g->si=s->stParam.param[s->stParam.current].I;
            s->y=s->stParam.param[s->stParam.current].Y;
            s->msk=s->stParam.param[s->stParam.current].M;
            s->l=s->stParam.param[s->stParam.current].L;
            s->d=s->stParam.param[s->stParam.current].D;
            s->r=s->stParam.param[s->stParam.current].R;
            s->bm=s->stParam.param[s->stParam.current].B;
            // outParam(s);
            s->rflg=0;
        
        }
      }else if(s->y==2 && g->k>=0){
        if(s->rflg==0){
         // printf("if(s->y==2 && s->j>=0 && s->rflg==0){\n");
        if(s->bm & (1<<g->k)){ 
           // printf("2if(s->bm & (1<<s->j)){\n");
          s->aB[s->y]=bit=1<<g->k; 

        } else{ 
           // printf("2} else{#if(s->bm & (1<<s->j)){");
          g->step=2;
          return;
        }
           // printf("2inparam\n");
          if(s->stParam.current<MAX){
            s->stParam.param[s->stParam.current].Y=s->y;
            s->stParam.param[s->stParam.current].I=g->si;
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
            // printf("2outparam\n");
            if(s->stParam.current>0){
              s->stParam.current--;
            }
            g->si=s->stParam.param[s->stParam.current].I;
            s->y=s->stParam.param[s->stParam.current].Y;
            s->msk=s->stParam.param[s->stParam.current].M;
            s->l=s->stParam.param[s->stParam.current].L;
            s->d=s->stParam.param[s->stParam.current].D;
            s->r=s->stParam.param[s->stParam.current].R;
            s->bm=s->stParam.param[s->stParam.current].B;
            // outParam(s);
            s->rflg=0;
        }
      }else{
        while((s->bm || s->rflg==1)&&s->y>2){
          // printf("while(s->bm || s->rflg==1){\n");
          if(s->rflg==0){
            // printf("inparam\n");
            s->bm^=s->aB[s->y]=bit=(-s->bm&s->bm); 
            if(s->stParam.current<MAX){
              s->stParam.param[s->stParam.current].Y=s->y;
              s->stParam.param[s->stParam.current].I=g->si;
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
            g->si=s->stParam.param[s->stParam.current].I;
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
    if(s->y<=3){
      // printf("if(s->y==1){\n");
      g->step=2;
      return;
    }else{
      s->rflg=1;
    }
    j++;
  } 
  g->step=2;
}

CL_KERNEL_KEYWORD void place(
    CL_GLOBAL_KEYWORD struct queenState *l,
    CL_GLOBAL_KEYWORD struct globalState *g){
  int index = get_global_id(0);
  struct queenState _l;
  struct globalState _g;

  _g.BOUND1=g[index].BOUND1;
  _g.si= g[index].si;
  for (int i = 0; i < _g.si; i++)
    _l.aB[i]=l[index].aB[i];
  _g.lTotal = g[index].lTotal;
  _g.step      = g[index].step;
  _l.y       = l[index].y;
  _l.bm     = l[index].bm;
  _g.BOUND2    =g[index].BOUND2;
  _l.ENDBIT    =l[index].ENDBIT;
  _l.TOPBIT    =l[index].TOPBIT;
  _l.SIDEMASK    =l[index].SIDEMASK;
  _l.LASTMASK  =l[index].LASTMASK;
  _g.lUnique  =g[index].lUnique;
  _l.bend  =l[index].bend;
  _l.rflg  =l[index].rflg;
  _l.stParam=l[index].stParam;
  //_l.msk= l[index].msk;
  _l.msk= (1<<_g.si)-1;
  _l.l= l[index].l;
  _l.d= l[index].d;
  _l.r= l[index].r;
  _g.B1= g[index].B1;
  _g.j= g[index].j;
  _g.k= g[index].k;
  // _l.C2=l[index].C2;
  // _l.C4=l[index].C4;
  // _l.C8=l[index].C8;
//    printf("BOUND1:%d\n",_g.BOUND1);
//    printf("j:%d\n",_g.j);
//   printf("k:%d\n",_g.k);
//  printf("si:%d\n",_g.si);
  printf("b:step:%d\n",_g.step);
  printf("y:%d\n",_l.y);
  printf("bm:%d\n",_l.bm);
//  printf("BOUND2:%d\n",_g.BOUND2);
  printf("TOPBIT:%d\n",_l.TOPBIT);
  printf("ENDBIT:%d\n",_l.ENDBIT);
  printf("SIDEMASK:%d\n",_l.SIDEMASK);
  printf("LASTMASK:%d\n",_l.LASTMASK);
  printf("lUnique:%ld\n",_g.lUnique);
  printf("bend:%d\n",_l.bend);
  printf("rflg:%d\n",_l.rflg);
  printf("msk:%d\n",_l.msk);
  printf("l:%d\n",_l.l);
  printf("d:%d\n",_l.d);
  printf("r:%d\n",_l.r);
//  printf("B1:%d\n",_g.B1);
    int bit;
    if(_g.BOUND1==0 && _g.step !=2){ 
      if(_g.step!=1){
        _l.aB[0]=1;
        _l.TOPBIT=1<<(_g.si-1);
      }
backTrack1(&_l,&_g);
    }else if(_g.BOUND1 !=0 && _g.step !=2){ 
      if(_g.step!=1){
      _l.TOPBIT=1<<(_g.si-1);
      _l.ENDBIT=_l.TOPBIT>>_g.BOUND1;
      _l.SIDEMASK=_l.LASTMASK=(_l.TOPBIT|1);
      }
      if(_g.BOUND1>0&&_g.BOUND2<_g.si-1&&_g.BOUND1<_g.BOUND2){
        if(_g.step!=1){
          for(int i=1;i<_g.BOUND1;i++){
            _l.LASTMASK=_l.LASTMASK|_l.LASTMASK>>1|_l.LASTMASK<<1;
          }
          _l.aB[0]=bit=(1<<_g.BOUND1);
          _l.y=1;_l.l=bit<<1;_l.d=bit;_l.r=bit>>1;
        }
backTrack2(&_l,&_g);
        if(_g.step!=1){
            _l.ENDBIT>>=_g.si;
        }
      }
    }
  printf("lTotal:%ld\n",_g.lTotal);
g[index].BOUND1=_g.BOUND1;
g[index].si=_g.si;
for(int i=0;i<_g.si;i++){l[index].aB[i]=_l.aB[i];}
g[index].lTotal=_g.lTotal;
if(_g.step==1){
  g[index].step=1;
// l[index].msk=1;
}else{
  g[index].step=2;
// l[index].msk=2;
}
 printf("m:step:%d:BOUND1:%d:k:%d:j:%d\n",g[index].step,g[index].BOUND1,g[index].k,g[index].j);
l[index].y=_l.y;
// l[index].startCol=0;
l[index].bm=_l.bm;
g[index].BOUND2=_g.BOUND2;
l[index].ENDBIT=_l.ENDBIT;
l[index].TOPBIT=_l.TOPBIT;
l[index].SIDEMASK=_l.SIDEMASK;
l[index].LASTMASK=_l.LASTMASK;
g[index].lUnique=_g.lUnique;
l[index].bend=_l.bend;
l[index].rflg=_l.rflg;
l[index].stParam=_l.stParam;
l[index].l=_l.l;
l[index].d=_l.d;
l[index].r=_l.r;
g[index].B1=_g.B1;
g[index].j=_g.j;
g[index].k=_g.k;
// l[index].C2=_l.C2;
// l[index].C4=_l.C4;
// l[index].C8=_l.C8;
}
#ifdef GCC_STYLE
int main(){
  int target=17;
  /**********/
  struct queenState inProgress[MAX*MAX*MAX];
  struct globalState gProgress[MAX*MAX*MAX];
  /**********/
  printf("%s\n"," N:          Total        Unique\n");
  for(int si=4;si<=target;si++){
    long gTotal=0;
    long gUnique=0;
    for(int i=0,B2=si-1;i<si;i++,B2--){ // N
      for(int j=0;j<si;j++){ // N
        for(int k=0;k<si;k++){ // N
          inProgress[i*si*si+j*si+k].si=si;
          inProgress[i*si*si+j*si+k].B1=-1;
          inProgress[i*si*si+j*si+k].BOUND1=i;
          inProgress[i*si*si+j*si+k].BOUND2=B2;
          inProgress[i*si*si+j*si+k].j=j;
          gProgress[i*si*si+j*si+k].k=k;
          inProgress[i*si*si+j*si+k].ENDBIT=0;
          inProgress[i*si*si+j*si+k].TOPBIT=1<<(si-1);
          inProgress[i*si*si+j*si+k].SIDEMASK=0;
          inProgress[i*si*si+j*si+k].LASTMASK=0;
          for (int m=0;m< si;m++){ inProgress[i*si*si+j*si+k].aB[m]=m;}
          gProgress[i*si*si+j*si+k].lTotal=0;
          gProgress[i*si*si+j*si+k].lUnique=0;
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
          place(&inProgress[i*si*si+j*si+k],&gProgress[i*si*si+j*si+k]);
        }
      }
    }
    for(int i=0;i<si;i++){
      for(int j=0;j<si;j++){ // N
        for(int k=0;k<si;k++){ // N
          gTotal+=gProgress[i*si*si+j*si+k].lTotal;
          gUnique+=gProgress[i*si*si+j*si+k].lUnique;
        }
      }
    }
  /**********/
      printf("%2d:%18lu%18lu\n", si,gTotal,gUnique);
  }
  return 0;
}
#endif
