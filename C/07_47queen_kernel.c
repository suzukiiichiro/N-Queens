//  単体で動かすときは以下のコメントを外す
// #define GCC_STYLE
#ifndef OPENCL_STYLE
#include "stdio.h"
#include "stdint.h"
#include <math.h>
typedef int64_t qint;
int get_global_id(int dimension){ return 0;}
#define CL_KERNEL_KEYWORD
#define CL_GLOBAL_KEYWORD
#define CL_CONSTANT_KEYWORD
#define CL_PACKED_KEYWORD
#define SIZE 24
#else
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
CL_PACKED_KEYWORD struct queenState {
  int BOUND1;
  int BOUND2;
  int si;
  long lTotal;
  long lUnique; // Number of solutinos found so far.
  char step;
};
CL_PACKED_KEYWORD struct localState {
  char y;
  int B1;
  int TOPBIT;
  int ENDBIT;
  int SIDEMASK;
  int LASTMASK;
  int bend;
  int rflg;
  int l;
  int d;
  int r;
  int bm;
  int msk;
  qint aB[MAX];
  struct STACK stParam;
};
int symmetryOps_2(struct queenState *s,struct localState *lo){
	int own,ptn,you,bit;
  //90度回転
		own=1;ptn=2;
		while(own<=(s->si-1)){
			bit=1;
			you=s->si-1;
			while((lo->aB[you]!=ptn)&&(lo->aB[own]>=bit)){
				bit<<=1;you--;
			}
			if(lo->aB[own]>bit){
				return 0;
			}
			else if(lo->aB[own]<bit){
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
int symmetryOps_4(struct queenState *s,struct localState *lo){
  int own,ptn,you,bit;
  //180度回転
    own=1; 
    you=s->si-2;
    while(own<=s->si-1){ 
      bit=1; 
      ptn=lo->TOPBIT;
      while((lo->aB[you]!=ptn)&&(lo->aB[own]>=bit)){ 
        bit<<=1; 
        ptn>>=1; 
      }
      if(lo->aB[own]>bit){ 
        return 0; 
      } 
      else if(lo->aB[own]<bit){ 
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
int symmetryOps_8(struct queenState *s,struct localState *lo){
  int own,ptn,you,bit;
  //270度回転
  // if(s->aB[s->BOUND1]==s->TOPBIT){ 
    own=1; 
    ptn=lo->TOPBIT>>1;
    while(own<=s->si-1){ 
      bit=1; 
      you=0;
      while((lo->aB[you]!=ptn)&&(lo->aB[own]>=bit)){ 
        bit<<=1; 
        you++; 
      }
      if(lo->aB[own]>bit){ 
        return 0; 
      } 
      else if(lo->aB[own]<bit){ 
        break; 
      }
      own++; 
      ptn>>=1;
    }
  // }
  return 1;
}
void symmetryOps_bm(struct queenState *s,struct localState *lo){
  //90度回転
	if(lo->aB[s->BOUND2]==1){
		if(symmetryOps_2(s,lo)==0){ return; }
	}
  //180度回転
  if(lo->aB[s->si-1]==lo->ENDBIT){ 
		if(symmetryOps_4(s,lo)==0){ return; }
	}
  //270度回転
  if(lo->aB[s->BOUND1]==lo->TOPBIT){ 
		if(symmetryOps_8(s,lo)==0){ return; }
	}
  s->lTotal+=8;
  s->lUnique++;
}
void inStruct(struct queenState *s,CL_GLOBAL_KEYWORD struct queenState *state,int index,struct localState *lo){
  s->BOUND1=state[index].BOUND1;
  s->BOUND2=state[index].BOUND2;
  s->si=state[index].si;
  s->lTotal=state[index].lTotal;
  s->lUnique=state[index].lUnique;
  //s->step=state[index].step;
  s->step=0;
  lo->y=0;
  lo->B1=2;
  lo->ENDBIT=0;
  lo->TOPBIT=1<<(s->si-1);
  lo->SIDEMASK=0;
  lo->LASTMASK=0;
  //printf("BOUND1:%d\n",s->BOUND1);
  //printf("BOUND2:%d\n",s->BOUND2);
  //printf("B1:%d\n",s->B1);
  for (int j=0;j<s->si;j++){
    lo->aB[j]=j;
  }
  //s->bend=state[index].bend;
  lo->bend=0;
  //s->rflg=state[index].rflg;
  lo->rflg=0;
  for (int m=0;m<s->si;m++){ 
    lo->stParam.param[m].Y=0;
    lo->stParam.param[m].I=s->si;
    lo->stParam.param[m].M=0;
    lo->stParam.param[m].L=0;
    lo->stParam.param[m].D=0;
    lo->stParam.param[m].R=0;
    lo->stParam.param[m].B=0;
  }
  lo->stParam.current=0;
  lo->msk=(1<<s->si)-1;
  lo->l=0;
  lo->d=0;
  lo->r=0;
  lo->bm=0;
  printf("si:%d\n",s->si);
  printf("B1:%d\n",lo->B1);
  printf("BOUND1:%d\n",s->BOUND1);
  printf("BOUND2:%d\n",s->BOUND2);
  printf("ENDBIT:%d\n",lo->ENDBIT);
  printf("TOPBIT:%d\n",lo->TOPBIT);
  printf("SIDEMASK:%d\n",lo->SIDEMASK);
  printf("LASTMASK:%d\n",lo->LASTMASK);
  printf("msk:%d\n",lo->msk);

}
void outStruct(CL_GLOBAL_KEYWORD struct queenState *state,struct queenState *s,int index){
  state[index].si=s->si;
  //state[index].id=s->id;
  state[index].BOUND1=s->BOUND1;
  state[index].BOUND2=s->BOUND2;
  state[index].lTotal=s->lTotal;
  state[index].lUnique=s->lUnique;
  state[index].step=s->step;
  //state[index].bend=s->bend;
  //state[index].rflg=s->rflg;
}
void inParam(struct queenState *s,struct localState *lo){
                if(lo->stParam.current<MAX){
                  lo->stParam.param[lo->stParam.current].Y=lo->y;
                  lo->stParam.param[lo->stParam.current].I=s->si;
                  lo->stParam.param[lo->stParam.current].M=lo->msk;
                  lo->stParam.param[lo->stParam.current].L=lo->l;
                  lo->stParam.param[lo->stParam.current].D=lo->d;
                  lo->stParam.param[lo->stParam.current].R=lo->r;
                  lo->stParam.param[lo->stParam.current].B=lo->bm;
                  (lo->stParam.current)++;
                }
}
void outParam(struct queenState *s,struct localState *lo){
                if(lo->stParam.current>0){
                  lo->stParam.current--;
                }
                s->si=lo->stParam.param[lo->stParam.current].I;
                lo->y=lo->stParam.param[lo->stParam.current].Y;
                lo->msk=lo->stParam.param[lo->stParam.current].M;
                lo->l=lo->stParam.param[lo->stParam.current].L;
                lo->d=lo->stParam.param[lo->stParam.current].D;
                lo->r=lo->stParam.param[lo->stParam.current].R;
                lo->bm=lo->stParam.param[lo->stParam.current].B;
}

void backTrack1(struct queenState *s,struct localState *lo){
  int bit;
        lo->aB[1]=bit=(1<<lo->B1);
        lo->y=2;lo->l=(2|bit)<<1;lo->d=(1|bit);lo->r=(bit>>1);
        unsigned long j=1;
        while(1){
          if(lo->rflg==0){
            lo->bm=lo->msk&~(lo->l|lo->d|lo->r); 
          }
          if (lo->y==s->si-1&&lo->rflg==0){ 
            if(lo->bm>0){
              lo->aB[lo->y]=lo->bm;
              s->lTotal+=8;
              s->lUnique++;
            }
          }else{
            if(lo->y<lo->B1&&lo->rflg==0){   
              lo->bm&=~2;
            }
            while(lo->bm>0|| lo->rflg==1){
              if(lo->rflg==0){
                lo->bm^=lo->aB[lo->y]=bit=(-lo->bm&lo->bm);
inParam(s,lo);
                lo->y++;
                lo->l=(lo->l|bit)<<1;
                lo->d=(lo->d|bit);
                lo->r=(lo->r|bit)>>1;
                lo->bend=1;
                break;
              }
              if(lo->rflg==1){ 
outParam(s,lo);
                lo->rflg=0;
              }
            }
            if(lo->bend==1 && lo->rflg==0){
              lo->bend=0;
              continue;
            }
          }
          if(lo->y==2){
            s->step=2;
            break;
          }else{
            lo->rflg=1;
          }
          j++;
        }
}
void backTrack2(struct queenState *s,struct localState *lo){
  int bit;
        unsigned long j=1;
        while (1){
          if(lo->rflg==0){
            lo->bm=lo->msk&~(lo->l|lo->d|lo->r); 
          }
          if (lo->y==s->si-1&&lo->rflg==0) {
            if(lo->bm>0 && (lo->bm&lo->LASTMASK)==0){
              lo->aB[lo->y]=lo->bm;
              symmetryOps_bm(s,lo);
            }
          }else{
            if(lo->y<s->BOUND1&&lo->rflg==0){
              lo->bm&=~lo->SIDEMASK; 
            }else if(lo->y==s->BOUND2&&lo->rflg==0){
              if((lo->d&lo->SIDEMASK)==0&&lo->rflg==0){ 
                lo->rflg=1;
              }
              if((lo->d&lo->SIDEMASK)!=lo->SIDEMASK&&lo->rflg==0){ 
                lo->bm&=lo->SIDEMASK; 
              }
            }
            while(lo->bm>0|| lo->rflg==1){
              if(lo->rflg==0){
                lo->bm^=lo->aB[lo->y]=bit=(-lo->bm&lo->bm); 
inParam(s,lo);
                lo->y++;
                lo->l=(lo->l|bit)<<1;
                lo->d=(lo->d|bit);
                lo->r=(lo->r|bit)>>1;
                lo->bend=1;
                break;
              }
              if(lo->rflg==1){ 
outParam(s,lo);
                lo->rflg=0;
              }
            }
            if(lo->bend==1 && lo->rflg==0){
              lo->bend=0;
              continue;
            }
          }
          if(lo->y==1){
            s->step=2;
            break;
          }else{
            lo->rflg=1;
          }
          j++;
        } 

}
CL_KERNEL_KEYWORD void place(CL_GLOBAL_KEYWORD struct queenState *state){
  int index=get_global_id(0);
  struct queenState s ;
  struct localState lo ;
inStruct(&s,state,index,&lo);
    int bit;
    if(s.BOUND1==0){ 
      lo.aB[0]=1;
        lo.TOPBIT=1<<(s.si-1);
      while(1){
        if(lo.B1>=s.si-1){
          break;
        }
backTrack1(&s,&lo);
        lo.B1=lo.B1+1;
      }
    }else{ 
        lo.TOPBIT=1<<(s.si-1);
        lo.ENDBIT=lo.TOPBIT>>s.BOUND1;
        lo.SIDEMASK=lo.LASTMASK=(lo.TOPBIT|1);
        if(s.BOUND1>0&&s.BOUND2<s.si-1&&s.BOUND1<s.BOUND2){
            for(int i=1;i<s.BOUND1;i++){
              lo.LASTMASK=lo.LASTMASK|lo.LASTMASK>>1|lo.LASTMASK<<1;
            }
          lo.aB[0]=bit=(1<<s.BOUND1);
          lo.y=1;lo.l=bit<<1;lo.d=bit;lo.r=bit>>1;
backTrack2(&s,&lo);
            lo.ENDBIT>>=s.si;
        }
    }
outStruct(state,&s,index);
}

#ifdef GCC_STYLE
int main(){
  struct queenState inProgress[MAX];
  printf("%s\n"," N:          Total        Unique\n");
  for(int si=4;si<17;si++){
  long gTotal=0;
  long gUnique=0;
    int B2=si-1;
    for(int i=0;i<si;i++){ //single
      inProgress[i].si=si;
      //inProgress[i].id=i;
      inProgress[i].B1=2;
      inProgress[i].BOUND1=i;
      inProgress[i].BOUND2=B2;
      B2--;
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
        inProgress[i].aT[m]=0;
        inProgress[i].aS[m]=0;
      }
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
      gTotal+=inProgress[i].lTotal;
      gUnique+=inProgress[i].lUnique;
    }
      printf("%2d:%18lu%18lu\n", si,gTotal,gUnique);
  }
  return 0;
}
#endif
