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
#define SIZE 27 
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
struct CL_PACKED_KEYWORD gtState {
  long lTotal; // Number of solutinos found so far.
  long lUnique; // Number of solutinos found so far.
};
struct CL_PACKED_KEYWORD globalState {
  int k;
  int j;
  int BOUND1;
  int BOUND2;
  int si;
  int B1;
  int step;
  int bend;
  int y;
  int bm;
  int rflg;
  int msk;
  int TOPBIT;
  int ENDBIT;
  int SIDEMASK;
  int LASTMASK;
  int l;
  int d;
  int r;
};
struct CL_PACKED_KEYWORD queenState {
  int aB[MAX];
  // int startCol; // First column this individual computation was tasked with filling.
  struct STACK stParam;
};
void symmetryOps_bm(struct queenState *s,struct globalState *g,struct gtState *gt){
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
      gt->lTotal+=2;
      gt->lUnique++; 
      return;
    }//end if
  }//end if
  //180度回転
  if(s->aB[g->si-1]==g->ENDBIT){ 
    own=1; 
    you=g->si-2;
    while(own<=g->si-1){ 
      bit=1; 
      ptn=g->TOPBIT;
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
      gt->lTotal+=4;
      gt->lUnique++;
      return ;
    }
  }
  //270度回転
  if(s->aB[g->BOUND1]==g->TOPBIT){ 
    own=1; 
    ptn=g->TOPBIT>>1;
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
  gt->lTotal+=8;
  gt->lUnique++;
}
void backTrack1(struct queenState *s,struct globalState *g,struct gtState *gt){
  int bit;
  if(g->step!=1){
    g->y=1;g->l=(1)<<1;g->d=(1);g->r=(1>>1);
  }
  g->step=0;
  unsigned long COUNT=1;
  while(COUNT>0){
#ifdef GCC_STYLE
#else
    if(COUNT==100000){
      // printf("b1_over\n");
      g->step=1;
      return;
    }
#endif
    if(g->rflg==0){
      g->bm=g->msk&~(g->l|g->d|g->r); 
    }
    if (g->y==g->si-1&&g->rflg==0){ 
      if(g->bm>0){
        s->aB[g->y]=g->bm;
        gt->lTotal+=8; 
        gt->lUnique++;
      }
    }else{
      // printf("}else{#if (s->y==g->si-1&&g->rflg==0){\n");
      if(g->y>1&&(1<<g->y)<g->B1 && g->rflg==0){   
        g->bm&=~2;
        // printf("g->bm&=~2;\n");
      }
      if(g->y==1 && g->j>=0){
        if(g->rflg==0){
        // printf("if(s->y==1 && s->j>=0 && g->rflg==0){\n");
        if(g->bm & (1<<g->j)){ 
          // printf("if(g->bm & (1<<s->j)){\n");
          s->aB[g->y]=bit=1<<g->j; 
          g->B1=bit;
        }else{ 
          // printf("}else{ #if(g->bm & (1<<s->j)){\n");
          g->step=2;
          printf("");
 // printf("return:%lu m:step:%d:BOUND1:%d:k:%d:j:%d\n",gt->lTotal,g->step,g->BOUND1,g->k,g->j);
          return;
        }
          // printf("if(g->rflg==0){#inParam\n");
          if(s->stParam.current<MAX){
            s->stParam.param[s->stParam.current].Y=g->y;
            s->stParam.param[s->stParam.current].I=g->si;
            s->stParam.param[s->stParam.current].M=g->msk;
            s->stParam.param[s->stParam.current].L=g->l;
            s->stParam.param[s->stParam.current].D=g->d;
            s->stParam.param[s->stParam.current].R=g->r;
            s->stParam.param[s->stParam.current].B=g->bm;
            (s->stParam.current)++;
          }
          // inParam(s);
          g->l=(g->l|bit)<<1; g->d=(g->d|bit); g->r=(g->r|bit)>>1;
          g->y++; 
          continue;
        }else{
            // printf("b1_outparam\n");
            if(s->stParam.current>0){
              s->stParam.current--;
            }
            g->si=s->stParam.param[s->stParam.current].I;
            g->y=s->stParam.param[s->stParam.current].Y;
            g->msk=s->stParam.param[s->stParam.current].M;
            g->l=s->stParam.param[s->stParam.current].L;
            g->d=s->stParam.param[s->stParam.current].D;
            g->r=s->stParam.param[s->stParam.current].R;
            g->bm=s->stParam.param[s->stParam.current].B;
            // outParam(s);
            g->rflg=0;
        
        }
      }else if(g->y==2 && g->k>=0){
        if(g->rflg==0){
        // printf("if(s->y==1 && s->j>=0 && g->rflg==0){\n");
        if(g->bm & (1<<g->k)){ 
          // printf("if(g->bm & (1<<s->j)){\n");
          s->aB[g->y]=bit=1<<g->k; 
          // s->B1=bit;
        }else{ 
          // printf("}else{ #if(g->bm & (1<<s->j)){\n");
          g->step=2;
 // printf("return_2:%lu m:step:%d:BOUND1:%d:k:%d:j:%d\n",gt->lTotal,g->step,g->BOUND1,g->k,g->j);
          printf("");
          return;
        }
          // printf("if(g->rflg==0){#inParam\n");
          if(s->stParam.current<MAX){
            s->stParam.param[s->stParam.current].Y=g->y;
            s->stParam.param[s->stParam.current].I=g->si;
            // s->stParam.param[s->stParam.current].M=g->msk;
            s->stParam.param[s->stParam.current].L=g->l;
            s->stParam.param[s->stParam.current].D=g->d;
            s->stParam.param[s->stParam.current].R=g->r;
            s->stParam.param[s->stParam.current].B=g->bm;
            (s->stParam.current)++;
          }
          // inParam(s);
          g->l=(g->l|bit)<<1; g->d=(g->d|bit); g->r=(g->r|bit)>>1;
          g->y++; 
          continue;
        }else{
            // printf("b1_y2outparam\n");
            if(s->stParam.current>0){
              s->stParam.current--;
            }
            g->si=s->stParam.param[s->stParam.current].I;
            g->y=s->stParam.param[s->stParam.current].Y;
            // g->msk=s->stParam.param[s->stParam.current].M;
            g->l=s->stParam.param[s->stParam.current].L;
            g->d=s->stParam.param[s->stParam.current].D;
            g->r=s->stParam.param[s->stParam.current].R;
            g->bm=s->stParam.param[s->stParam.current].B;
            // outParam(s);
            g->rflg=0;
        }
      }else {
        // printf("}else{ #if(s->y>1&&(1<<s->y)<s->B1){\n");
        while(g->bm || g->rflg==1){
          // printf("while(g->bm || g->rflg==1){\n");
          if(g->rflg==0){
            // printf("inparam\n");
            g->bm^=s->aB[g->y]=bit=(-g->bm&g->bm);
            if(s->stParam.current<MAX){
              s->stParam.param[s->stParam.current].Y=g->y;
              s->stParam.param[s->stParam.current].I=g->si;
              s->stParam.param[s->stParam.current].M=g->msk;
              s->stParam.param[s->stParam.current].L=g->l;
              s->stParam.param[s->stParam.current].D=g->d;
              s->stParam.param[s->stParam.current].R=g->r;
              s->stParam.param[s->stParam.current].B=g->bm;
              (s->stParam.current)++;
            }
            // inParam(s);
            g->l=(g->l|bit)<<1; g->d=(g->d|bit); g->r=(g->r|bit)>>1;
            g->y++; g->bend=1;
            break;
          }else{ // g->rflg==1
            // printf("outparam\n");
            if(s->stParam.current>0){
              s->stParam.current--;
            }
            g->si=s->stParam.param[s->stParam.current].I;
            g->y=s->stParam.param[s->stParam.current].Y;
            g->msk=s->stParam.param[s->stParam.current].M;
            g->l=s->stParam.param[s->stParam.current].L;
            g->d=s->stParam.param[s->stParam.current].D;
            g->r=s->stParam.param[s->stParam.current].R;
            g->bm=s->stParam.param[s->stParam.current].B;
            // outParam(s);
            g->rflg=0;
          }
        }
      }
      if(g->bend==1 && g->rflg==0){
        // printf("if(s->bend==1){\n");
        g->bend=0;
        continue;
      }
    }
    if(g->y<=3){
      // printf("if(s->y==1){\n");
      g->step=2;
      return;
    }else{
      // printf("}else{#if(s->y==1){");
      g->rflg=1;
    }
    COUNT++;
  }   
  g->step=2;
  return;
}
void backTrack2(struct queenState *s,struct globalState *g,struct gtState *gt){
  int bit;
  unsigned long COUNT=1;
  g->step=0;
  while (COUNT>0){
#ifdef GCC_STYLE
#else
    if(COUNT==100000){
      g->step=1;
      // printf("b2_over\n");
      return;
    }
#endif
    if(g->rflg==0){
      g->bm=g->msk&~(g->l|g->d|g->r); 
      // printf("g->bm=g->msk&~(g->l|g->d|g->r);\n");
    }
    if (g->y==g->si-1 && g->rflg==0){
      // printf("if (s->y==g->si-1&& g->bm){\n");
      if(g->bm>0 && (g->bm&g->LASTMASK)==0){
        // printf("if((g->bm&g->lASTMASK)==0){#symmetryOps\n");
        s->aB[g->y]=g->bm;
        symmetryOps_bm(s,g,gt);
      }
    }else{
      if(g->y<g->BOUND1&&g->rflg==0){
        // printf("if(s->y<g->BOUND1){\n");
        g->bm&=~g->SIDEMASK; 
      }else if(g->y==g->BOUND2&&g->rflg==0){
        // printf("}else if(s->y==g->BOUND2){\n");
        if((g->d&g->SIDEMASK)==0&&g->rflg==0){ 
          // printf("if((g->d&g->SIDEMASK)==0){\n");
          g->rflg=1;
        }
        if((g->d&g->SIDEMASK)!=g->SIDEMASK&&g->rflg==0){ 
          // printf("if((g->d&g->SIDEMASK)!=g->SIDEMASK){\n");
          g->bm&=g->SIDEMASK; 
        }
      }

      if(g->y==1 && g->j>=0){
        if(g->rflg==0){
        // printf("if(s->y==1 && s->j>=0 && g->rflg==0){\n");
        if(g->bm & (1<<g->j)){ 
          // printf("if(g->bm & (1<<s->j)){\n");
          s->aB[g->y]=bit=1<<g->j; 
        } else{ 
          // printf("} else{#if(g->bm & (1<<s->j)){");
          g->step=2;
// printf("return:%lu m:step:%d:BOUND1:%d:k:%d:j:%d\n",gt->lTotal,g->step,g->BOUND1,g->k,g->j);
          return;
        }
          // printf("inparam\n");
          if(s->stParam.current<MAX){
            s->stParam.param[s->stParam.current].Y=g->y;
            s->stParam.param[s->stParam.current].I=g->si;
            s->stParam.param[s->stParam.current].M=g->msk;
            s->stParam.param[s->stParam.current].L=g->l;
            s->stParam.param[s->stParam.current].D=g->d;
            s->stParam.param[s->stParam.current].R=g->r;
            s->stParam.param[s->stParam.current].B=g->bm;
            (s->stParam.current)++;
          }
          // inParam(s);
          g->l=(g->l|bit)<<1; g->d=(g->d|bit); g->r=(g->r|bit)>>1;
          g->y++; 
          continue;
        }else{
            // printf("b2_y1outparam\n");
            if(s->stParam.current>0){
              s->stParam.current--;
            }
            g->si=s->stParam.param[s->stParam.current].I;
            g->y=s->stParam.param[s->stParam.current].Y;
            g->msk=s->stParam.param[s->stParam.current].M;
            g->l=s->stParam.param[s->stParam.current].L;
            g->d=s->stParam.param[s->stParam.current].D;
            g->r=s->stParam.param[s->stParam.current].R;
            g->bm=s->stParam.param[s->stParam.current].B;
            // outParam(s);
            g->rflg=0;
        
        }
      }else if(g->y==2 && g->k>=0){
        if(g->rflg==0){
         // printf("if(s->y==2 && s->j>=0 && g->rflg==0){\n");
        if(g->bm & (1<<g->k)){ 
           // printf("2if(g->bm & (1<<s->j)){\n");
          s->aB[g->y]=bit=1<<g->k; 

        } else{ 
           // printf("2} else{#if(g->bm & (1<<s->j)){");
          g->step=2;
// printf("return:%lu m:step:%d:BOUND1:%d:k:%d:j:%d\n",gt->lTotal,g->step,g->BOUND1,g->k,g->j);
          return;
        }
           // printf("2inparam\n");
          if(s->stParam.current<MAX){
            s->stParam.param[s->stParam.current].Y=g->y;
            s->stParam.param[s->stParam.current].I=g->si;
            s->stParam.param[s->stParam.current].M=g->msk;
            s->stParam.param[s->stParam.current].L=g->l;
            s->stParam.param[s->stParam.current].D=g->d;
            s->stParam.param[s->stParam.current].R=g->r;
            s->stParam.param[s->stParam.current].B=g->bm;
            (s->stParam.current)++;
          }
          // inParam(s);
          g->l=(g->l|bit)<<1; g->d=(g->d|bit); g->r=(g->r|bit)>>1;
          g->y++; 
          continue;
        }else{
            // printf("2outparam\n");
            if(s->stParam.current>0){
              s->stParam.current--;
            }
            g->si=s->stParam.param[s->stParam.current].I;
            g->y=s->stParam.param[s->stParam.current].Y;
            g->msk=s->stParam.param[s->stParam.current].M;
            g->l=s->stParam.param[s->stParam.current].L;
            g->d=s->stParam.param[s->stParam.current].D;
            g->r=s->stParam.param[s->stParam.current].R;
            g->bm=s->stParam.param[s->stParam.current].B;
            // outParam(s);
            g->rflg=0;
        }
      }else{
        while((g->bm || g->rflg==1)&&g->y>2){
          // printf("while(g->bm || g->rflg==1){\n");
          if(g->rflg==0){
            // printf("inparam\n");
            g->bm^=s->aB[g->y]=bit=(-g->bm&g->bm); 
            if(s->stParam.current<MAX){
              s->stParam.param[s->stParam.current].Y=g->y;
              s->stParam.param[s->stParam.current].I=g->si;
              s->stParam.param[s->stParam.current].M=g->msk;
              s->stParam.param[s->stParam.current].L=g->l;
              s->stParam.param[s->stParam.current].D=g->d;
              s->stParam.param[s->stParam.current].R=g->r;
              s->stParam.param[s->stParam.current].B=g->bm;
              (s->stParam.current)++;
            }
            // inParam(s);
            g->l=(g->l|bit)<<1; g->d=(g->d|bit); g->r=(g->r|bit)>>1;
            g->y++; g->bend=1;
            break;
          }else{
            // printf("outparam\n");
            if(s->stParam.current>0){
              s->stParam.current--;
            }
            g->si=s->stParam.param[s->stParam.current].I;
            g->y=s->stParam.param[s->stParam.current].Y;
            g->msk=s->stParam.param[s->stParam.current].M;
            g->l=s->stParam.param[s->stParam.current].L;
            g->d=s->stParam.param[s->stParam.current].D;
            g->r=s->stParam.param[s->stParam.current].R;
            g->bm=s->stParam.param[s->stParam.current].B;
            // outParam(s);
            g->rflg=0;
          }
        }//end while
      }
      if(g->bend==1 && g->rflg==0){
        // printf("if(s->bend==1){\n");
        g->bend=0;
        continue;
      }
    }
    if(g->y<=3){
      // printf("if(s->y==1){\n");
      g->step=2;
      return;
    }else{
      g->rflg=1;
    }
    COUNT++;
  } 
  g->step=2;
}

CL_KERNEL_KEYWORD void place(
    CL_GLOBAL_KEYWORD struct queenState *l,
    CL_GLOBAL_KEYWORD struct globalState *g,
    CL_GLOBAL_KEYWORD struct gtState *gt){
  int index = get_global_id(0);
  struct queenState _l;
  struct globalState _g;
  struct gtState _gt;

  _g.BOUND1=g[index].BOUND1;
  _g.si= g[index].si;
  for (int i = 0; i < _g.si; i++)
    _l.aB[i]=l[index].aB[i];
  _gt.lTotal = gt[index].lTotal;
  _g.step      = g[index].step;
  _g.y       = g[index].y;
  _g.bm     = g[index].bm;
  _g.BOUND2    =g[index].BOUND2;
  _g.ENDBIT    =g[index].ENDBIT;
  _g.TOPBIT    =g[index].TOPBIT;
  _g.SIDEMASK    =g[index].SIDEMASK;
  _g.LASTMASK  =g[index].LASTMASK;
  _gt.lUnique  =gt[index].lUnique;
  _g.bend  =g[index].bend;
  _g.rflg  =g[index].rflg;
  _l.stParam=l[index].stParam;
  //_l.msk= l[index].msk;
  _g.msk= (1<<_g.si)-1;
  _g.l= g[index].l;
  _g.d= g[index].d;
  _g.r= g[index].r;
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
//   printf("b:step:%d\n",_g.step);
//   printf("y:%d\n",_g.y);
//   printf("bm:%d\n",_g.bm);
// //  printf("BOUND2:%d\n",_g.BOUND2);
//   printf("TOPBIT:%d\n",_g.TOPBIT);
//   printf("ENDBIT:%d\n",_g.ENDBIT);
//   printf("SIDEMASK:%d\n",_g.SIDEMASK);
//   printf("LASTMASK:%d\n",_g.LASTMASK);
//   printf("lUnique:%ld\n",_gt.lUnique);
//   printf("bend:%d\n",_g.bend);
//   printf("rflg:%d\n",_g.rflg);
//   printf("msk:%d\n",_g.msk);
//   printf("l:%d\n",_g.l);
//   printf("d:%d\n",_g.d);
//   printf("r:%d\n",_g.r);
//  printf("B1:%d\n",_g.B1);
    int bit;
    if(_g.BOUND1==0 && _g.step !=2){ 
      if(_g.step!=1){
        _l.aB[0]=1;
        _g.TOPBIT=1<<(_g.si-1);
      }
      int rtn;
      backTrack1(&_l,&_g,&_gt);
      // if(rtn==1){
        // printf("ltotal:%ld:lUnique:%ld:BOUND1:%d:k:%d:j:%d\n",_gt.lTotal,_gt.lUnique,_g.BOUND1,_g.k,_g.j);
        // _gt.lTotal=0;
        // _gt.lUnique=0;
        // printf("ltotal:%ld:lUnique:%ld:BOUND1:%d:k:%d:j:%d\n",_gt.lTotal,_gt.lUnique,_g.BOUND1,_g.k,_g.j);
      // }
    }else if(_g.BOUND1 !=0 && _g.step !=2){ 
      if(_g.step!=1){
      _g.TOPBIT=1<<(_g.si-1);
      _g.ENDBIT=_g.TOPBIT>>_g.BOUND1;
      _g.SIDEMASK=_g.LASTMASK=(_g.TOPBIT|1);
      }
      if(_g.BOUND1>0&&_g.BOUND2<_g.si-1&&_g.BOUND1<_g.BOUND2){
        if(_g.step!=1){
          for(int i=1;i<_g.BOUND1;i++){
            _g.LASTMASK=_g.LASTMASK|_g.LASTMASK>>1|_g.LASTMASK<<1;
          }
          _l.aB[0]=bit=(1<<_g.BOUND1);
          _g.y=1;_g.l=bit<<1;_g.d=bit;_g.r=bit>>1;
        }
backTrack2(&_l,&_g,&_gt);
        if(_g.step!=1){
            _g.ENDBIT>>=_g.si;
        }
      }
    }
  // printf("lTotal:%ld\n",_gt.lTotal);
g[index].BOUND1=_g.BOUND1;
g[index].si=_g.si;
for(int i=0;i<_g.si;i++){l[index].aB[i]=_l.aB[i];}
if(_g.step==1){
  g[index].step=1;
// l[index].msk=1;
}else{
  g[index].step=2;
// l[index].msk=2;
}
 // printf("m:step:%d:BOUND1:%d:k:%d:j:%d\n",g[index].step,g[index].BOUND1,g[index].k,g[index].j);
gt[index].lTotal=_gt.lTotal;
gt[index].lUnique=_gt.lUnique;
g[index].y=_g.y;
// l[index].startCol=0;
g[index].bm=_g.bm;
g[index].BOUND2=_g.BOUND2;
g[index].ENDBIT=_g.ENDBIT;
g[index].TOPBIT=_g.TOPBIT;
g[index].SIDEMASK=_g.SIDEMASK;
g[index].LASTMASK=_g.LASTMASK;
g[index].bend=_g.bend;
g[index].rflg=_g.rflg;
l[index].stParam=_l.stParam;
g[index].l=_g.l;
g[index].d=_g.d;
g[index].r=_g.r;
g[index].B1=_g.B1;
g[index].j=_g.j;
g[index].k=_g.k;
// l[index].C2=_l.C2;
// l[index].C4=_l.C4;
// l[index].C8=_l.C8;
//printf("########### _gt.lTotal %lu gt[index].lTotal %lu index :%d:\n", _gt.lTotal,gt[index].lTotal,index);
// printf("###############lTotal         %lu m:step:%d:BOUND1:%d:k:%d:j:%d\n",_gt.lTotal,g[index].step,g[index].BOUND1,g[index].k,g[index].j);
// printf("###############lTotal         %lu m:step:%d:BOUND1:%d:k:%d:j:%d\n",gt[index].lTotal,g[index].step,g[index].BOUND1,g[index].k,g[index].j);
}
#ifdef GCC_STYLE
int main(){
  int target=14;
  /**********/
  struct queenState inProgress[MAX*MAX*MAX];
  struct globalState gProgress[MAX*MAX*MAX];
  struct gtState gtProgress[MAX*MAX*MAX];
  /**********/
  printf("%s\n"," N:          Total        Unique\n");
  for(int si=4;si<=target;si++){
    long gTotal=0;
    long gUnique=0;
    for(int i=0,B2=si-1;i<si;i++,B2--){ // N
      for(int j=0;j<si;j++){ // N
        for(int k=0;k<si;k++){ // N
          gProgress[i*si*si+j*si+k].si=si;
          gProgress[i*si*si+j*si+k].B1=-1;
          gProgress[i*si*si+j*si+k].BOUND1=i;
          gProgress[i*si*si+j*si+k].BOUND2=B2;
          gProgress[i*si*si+j*si+k].j=j;
          gProgress[i*si*si+j*si+k].k=k;
          gProgress[i*si*si+j*si+k].ENDBIT=0;
          gProgress[i*si*si+j*si+k].TOPBIT=1<<(si-1);
          gProgress[i*si*si+j*si+k].SIDEMASK=0;
          gProgress[i*si*si+j*si+k].LASTMASK=0;
          for (int m=0;m< si;m++){ inProgress[i*si*si+j*si+k].aB[m]=m;}
          gtProgress[i*si*si+j*si+k].lTotal=0;
          gtProgress[i*si*si+j*si+k].lUnique=0;
          gProgress[i*si*si+j*si+k].step=0;
          gProgress[i*si*si+j*si+k].y=0;
          gProgress[i*si*si+j*si+k].bend=0;
          gProgress[i*si*si+j*si+k].rflg=0;
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
          gProgress[i*si*si+j*si+k].msk=(1<<si)-1;
          gProgress[i*si*si+j*si+k].l=0;
          gProgress[i*si*si+j*si+k].d=0;
          gProgress[i*si*si+j*si+k].r=0;
          gProgress[i*si*si+j*si+k].bm=0;
          place(&inProgress[i*si*si+j*si+k],&gProgress[i*si*si+j*si+k],
              &gtProgress[i*si*si+j*si+k]);
        }
      }
    }
    for(int i=0;i<si;i++){
      for(int j=0;j<si;j++){ // N
        for(int k=0;k<si;k++){ // N
          gTotal+=gtProgress[i*si*si+j*si+k].lTotal;
          gUnique+=gtProgress[i*si*si+j*si+k].lUnique;
        }
      }
    }
  /**********/
      printf("%2d:%18lu%18lu\n", si,gTotal,gUnique);
  }
  return 0;
}
#endif
