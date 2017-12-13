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
  int si;
  int B1;
  int BOUND1;
  int BOUND2;
  int TOPBIT;
  int ENDBIT;
  int SIDEMASK;
  int LASTMASK;
  qint aB[MAX];
  long lTotal;
  long lUnique; // Number of solutinos found so far.
  char step;
  char y;
  int bend;
  int rflg;
  qint aT[MAX];
  qint aS[MAX];
  struct STACK stParam;
  int msk;
  int l;
  int d;
  int r;
  int bm;
};
struct CL_PACKED_KEYWORD localState {
  int BOUND1;
  int si;
  int aB[MAX];
  long lTotal; // Number of solutinos found so far.
  int step;
  int y;
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
};
int symmetryOps_2(struct localState *s){
	int own,ptn,you,bit;
  //90度回転
		own=1;ptn=2;
		while(own<=(s->si-1)){
			bit=1;
			you=s->si-1;
			while((s->aB[you]!=ptn)&&(s->aB[own]>=bit)){
				bit<<=1;you--;
			}
			if(s->aB[own]>bit){
				return 0;
			}
			else if(s->aB[own]<bit){
				printf("");
				// break;
				goto go;
			}
			own++;ptn<<=1;
		}
go:
		if(own>s->si-1){
			s->lTotal+=2;
			s->lUnique++;
			return 0;
		}
	return 1;
}
int symmetryOps_4(struct localState *s){
  int own,ptn,you,bit;
  //180度回転
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
        return 0; 
      } 
      else if(s->aB[own]<bit){ 
        break; 
      }
      own++; 
      you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>s->si-1){ 
      s->lTotal+=4;
      s->lUnique++;
      return 0;
    }
  return 1;
}
//int symmetryOps_8(struct queenState *s,struct symmetry *sym){
int symmetryOps_8(struct localState *s){
  int own,ptn,you,bit;
  //270度回転
  // if(s->aB[s->BOUND1]==s->TOPBIT){ 
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
        return 0; 
      } 
      else if(s->aB[own]<bit){ 
        break; 
      }
      own++; 
      ptn>>=1;
    }
  // }
  return 1;
}
void symmetryOps_bm(struct localState *s){
  //90度回転
	if(s->aB[s->BOUND2]==1){
		if(symmetryOps_2(s)==0){ return; }
	}
  //180度回転
  if(s->aB[s->si-1]==s->ENDBIT){ 
		if(symmetryOps_4(s)==0){ return; }
	}
  //270度回転
  if(s->aB[s->BOUND1]==s->TOPBIT){ 
		if(symmetryOps_8(s)==0){ return; }
	}
  s->lTotal+=8;
  s->lUnique++;
}
void inParam(struct localState *s){
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
}
void outParam(struct localState *s){
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
}
void inStruct(struct localState *s,CL_GLOBAL_KEYWORD struct queenState *state,int index){
  s->si=state[index].si;
  s->B1=state[index].B1;
  s->BOUND1=state[index].BOUND1;
  s->BOUND2=state[index].BOUND2;
  s->ENDBIT=state[index].ENDBIT;
  s->TOPBIT=state[index].TOPBIT;
  s->SIDEMASK=state[index].SIDEMASK;
  s->LASTMASK=state[index].LASTMASK;
  //printf("BOUND1:%d\n",s->BOUND1);
  //printf("BOUND2:%d\n",s->BOUND2);
  //printf("B1:%d\n",s->B1);
  for (int j=0;j<s->si;j++){
    s->aB[j]=state[index].aB[j];
  }
  s->lTotal=state[index].lTotal;
  s->lUnique=state[index].lUnique;
  //s->step=state[index].step;
  s->step=0;
  s->y=state[index].y;
  s->bend=state[index].bend;
  s->rflg=state[index].rflg;
  // for (int j=0;j<s->si;j++){
  //   s->aT[j]=state[index].aT[j];
  //   s->aS[j]=state[index].aS[j];
  // }
  s->stParam=state[index].stParam;
  s->msk=(1<<s->si)-1;
  s->l=state[index].l;
  s->d=state[index].d;
  s->r=state[index].r;
  s->bm=state[index].bm;

}
void outStruct(CL_GLOBAL_KEYWORD struct queenState *state,struct localState *s,int index){
  state[index].si=s->si;
  //state[index].id=s->id;
  state[index].B1=s->B1;
  state[index].BOUND1=s->BOUND1;
  state[index].BOUND2=s->BOUND2;
  state[index].ENDBIT=s->ENDBIT;
  state[index].TOPBIT=s->TOPBIT;
  state[index].SIDEMASK=s->SIDEMASK;
  state[index].LASTMASK=s->LASTMASK;
  for (int j=0;j<s->si;j++){
    state[index].aB[j] = s->aB[j];
  }//end for
  state[index].lTotal=s->lTotal;
  state[index].lUnique=s->lUnique;
  state[index].step=s->step;
  state[index].y=s->y;
  state[index].bend=s->bend;
  state[index].rflg=s->rflg;
  // for (int j=0;j<s->si;j++){
  //   state[index].aT[j]=s->aT[j];
  //   state[index].aS[j]=s->aS[j];
  // }//end for
  state[index].stParam=s->stParam;
  state[index].msk=s->msk;
  state[index].l=s->l;
  state[index].d=s->d;
  state[index].r=s->r;
  state[index].bm=s->bm;
}

void backTrack1(struct localState *s){
  int bit;
        s->aB[1]=bit=(1<<s->B1);
        s->y=2;s->l=(2|bit)<<1;s->d=(1|bit);s->r=(bit>>1);
        unsigned long j=1;
        while(1){
#ifdef GCC_STYLE
#else
          if(j==500000){
            break;
          }
#endif
          if(s->rflg==0){
            s->bm=s->msk&~(s->l|s->d|s->r); 
          }
          if (s->y==s->si-1&&s->rflg==0){ 
            if(s->bm>0){
              s->aB[s->y]=s->bm;
              s->lTotal+=8;
              s->lUnique++;
            }
          }else{
            if(s->y<s->B1&&s->rflg==0){   
              s->bm&=~2;
            }
            while(s->bm>0|| s->rflg==1){
              if(s->rflg==0){
                s->bm^=s->aB[s->y]=bit=(-s->bm&s->bm);
inParam(s);
                s->y++;
                s->l=(s->l|bit)<<1;
                s->d=(s->d|bit);
                s->r=(s->r|bit)>>1;
                s->bend=1;
                break;
              }
              if(s->rflg==1){ 
outParam(s);
                s->rflg=0;
              }
            }
            if(s->bend==1 && s->rflg==0){
              s->bend=0;
              continue;
            }
          }
          if(s->y==2){
            s->step=2;
            break;
          }else{
            s->rflg=1;
          }
          j++;
        }
}
void backTrack2(struct localState *s){
  int bit;
        unsigned long j=1;
        while (j>0){
#ifdef GCC_STYLE
#else
    if(j==100){
      break;
    }
#endif
          if(s->rflg==0){
            s->bm=s->msk&~(s->l|s->d|s->r); 
          }
          if (s->y==s->si-1&&s->rflg==0) {
            if(s->bm>0 && (s->bm&s->LASTMASK)==0){
              s->aB[s->y]=s->bm;
              symmetryOps_bm(s);
            }
          }else{
            if(s->y<s->BOUND1&&s->rflg==0){
              s->bm&=~s->SIDEMASK; 
            }else if(s->y==s->BOUND2&&s->rflg==0){
              if((s->d&s->SIDEMASK)==0&&s->rflg==0){ 
                s->rflg=1;
              }
              if((s->d&s->SIDEMASK)!=s->SIDEMASK&&s->rflg==0){ 
                s->bm&=s->SIDEMASK; 
              }
            }
            while(s->bm>0|| s->rflg==1){
              if(s->rflg==0){
                s->bm^=s->aB[s->y]=bit=(-s->bm&s->bm); 
inParam(s);
                s->y++;
                s->l=(s->l|bit)<<1;
                s->d=(s->d|bit);
                s->r=(s->r|bit)>>1;
                s->bend=1;
                break;
              }
              if(s->rflg==1){ 
outParam(s);
                s->rflg=0;
              }
            }
            if(s->bend==1 && s->rflg==0){
              s->bend=0;
              continue;
            }
          }
          if(s->y==1){
            s->step=2;
            break;
          }else{
            s->rflg=1;
          }
          j++;
        } 
}
CL_KERNEL_KEYWORD void place(CL_GLOBAL_KEYWORD struct queenState *state){
  int index = get_global_id(0);
  struct localState s;
  s.BOUND1=state[index].BOUND1;
	inStruct(&s,state,index);
  // int bflg=0;
    // if(bflg==1){
    //   s.BOUND1--;
    //   s.BOUND2++;
    //   s.step=0;
    //   break;
    // }
    int bit;
    if(s.BOUND1==0){ 
      s.aB[0]=1;
      // if(bflg==0){
        s.TOPBIT=1<<(s.si-1);
      // }
      while(1){
        // if(bflg==1){
        //   s.B1--;
        //   break;
        // }
        if(s.B1==s.si-1){
          break;
        }
backTrack1(&s);
        s.B1=s.B1+1;
      }
    }else{ 
        // if(bflg==0){
        s.TOPBIT=1<<(s.si-1);
        s.ENDBIT=s.TOPBIT>>s.BOUND1;
        s.SIDEMASK=s.LASTMASK=(s.TOPBIT|1);
        // }
        if(s.BOUND1>0&&s.BOUND2<s.si-1&&s.BOUND1<s.BOUND2){
          // if(bflg==0){
            for(int i=1;i<s.BOUND1;i++){
              s.LASTMASK=s.LASTMASK|s.LASTMASK>>1|s.LASTMASK<<1;
            }
          // }
          s.aB[0]=bit=(1<<s.BOUND1);
          s.y=1;s.l=bit<<1;s.d=bit;s.r=bit>>1;
backTrack2(&s);
          // if(bflg==0){
            s.ENDBIT>>=s.si;
          // }
        }
  }
outStruct(state,&s,index);
}
#ifdef GCC_STYLE
int main(){
  int target=12;
  /**********/
  struct queenState inProgress[MAX];
  /**********/
  printf("%s\n"," N:          Total        Unique\n");
  for(int si=4;si<=target;si++){
    long gTotal=0;
    long gUnique=0;
    // int B2=si-1;
    for(int i=0,B2=si-1;i<si;i++,B2--){ // N
      inProgress[i].si=si;
      //inProgress[i].id=i;
      inProgress[i].B1=2;
      inProgress[i].BOUND1=i;
      inProgress[i].BOUND2=B2;
      // B2--;
      inProgress[i].ENDBIT=0;
      inProgress[i].TOPBIT=1<<(si-1);
      inProgress[i].SIDEMASK=0;
      inProgress[i].LASTMASK=0;
      for (int m=0;m< si;m++){ inProgress[i].aB[m]=m;}
      inProgress[i].lTotal=0;
      inProgress[i].lUnique=0;
      inProgress[i].step=0;
      inProgress[i].y=0;
      inProgress[i].bend=0;
      inProgress[i].rflg=0;
      for (int m=0;m<si;m++){ 
        inProgress[i].stParam.param[m].Y=0;
        inProgress[i].stParam.param[m].I=si;
        inProgress[i].stParam.param[m].M=0;
        inProgress[i].stParam.param[m].L=0;
        inProgress[i].stParam.param[m].D=0;
        inProgress[i].stParam.param[m].R=0;
        inProgress[i].stParam.param[m].B=0;
      }
      inProgress[i].stParam.current=0;
      inProgress[i].msk=(1<<si)-1;
      inProgress[i].l=0;
      inProgress[i].d=0;
      inProgress[i].r=0;
      inProgress[i].bm=0;

      //
      place(&inProgress[i]);
    }
    for(int i=0;i<si;i++){
      gTotal+=inProgress[i].lTotal;
      gUnique+=inProgress[i].lUnique;
    }
  /**********/
      printf("%2d:%18lu%18lu\n", si,gTotal,gUnique);
  }
  return 0;
}
#endif
