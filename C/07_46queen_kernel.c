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
int symmetryOps_2(struct queenState *s){
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
int symmetryOps_4(struct queenState *s){
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
int symmetryOps_8(struct queenState *s){
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
void symmetryOps_bm(struct queenState *s){
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
void inStruct(struct queenState *s,CL_GLOBAL_KEYWORD struct queenState *state,int index){
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
  for (int j=0;j<s->si;j++){
    s->aT[j]=state[index].aT[j];
    s->aS[j]=state[index].aS[j];
  }
  s->stParam=state[index].stParam;
  s->msk=(1<<s->si)-1;
  s->l=state[index].l;
  s->d=state[index].d;
  s->r=state[index].r;
  s->bm=state[index].bm;

}
void outStruct(CL_GLOBAL_KEYWORD struct queenState *state,struct queenState *s,int index){
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
  for (int j=0;j<s->si;j++){
    state[index].aT[j]=s->aT[j];
    state[index].aS[j]=s->aS[j];
  }//end for
  state[index].stParam=s->stParam;
  state[index].msk=s->msk;
  state[index].l=s->l;
  state[index].d=s->d;
  state[index].r=s->r;
  state[index].bm=s->bm;
}
void inParam(struct queenState *s){
                  s->stParam.param[s->stParam.current].Y=s->y;
                  s->stParam.param[s->stParam.current].I=s->si;
                  s->stParam.param[s->stParam.current].M=s->msk;
                  s->stParam.param[s->stParam.current].L=s->l;
                  s->stParam.param[s->stParam.current].D=s->d;
                  s->stParam.param[s->stParam.current].R=s->r;
                  s->stParam.param[s->stParam.current].B=s->bm;
}
void outParam(struct queenState *s){
                s->si=s->stParam.param[s->stParam.current].I;
                s->y=s->stParam.param[s->stParam.current].Y;
                s->msk=s->stParam.param[s->stParam.current].M;
                s->l=s->stParam.param[s->stParam.current].L;
                s->d=s->stParam.param[s->stParam.current].D;
                s->r=s->stParam.param[s->stParam.current].R;
                s->bm=s->stParam.param[s->stParam.current].B;
}

CL_KERNEL_KEYWORD void place(CL_GLOBAL_KEYWORD struct queenState *state){
  int index=get_global_id(0);
  struct queenState s ;
/*
  s.si=state[index].si;
  s.B1=state[index].B1;
  s.BOUND1=state[index].BOUND1;
  s.BOUND2=state[index].BOUND2;
  s.ENDBIT=state[index].ENDBIT;
  s.TOPBIT=state[index].TOPBIT;
  s.SIDEMASK=state[index].SIDEMASK;
  s.LASTMASK=state[index].LASTMASK;
  //printf("BOUND1:%d\n",s.BOUND1);
  //printf("BOUND2:%d\n",s.BOUND2);
  //printf("B1:%d\n",s.B1);
  for (int j=0;j<s.si;j++){
    s.aB[j]=state[index].aB[j];
  }
  s.lTotal=state[index].lTotal;
  s.lUnique=state[index].lUnique;
  //s.step=state[index].step;
  s.step=0;
  s.y=state[index].y;
  s.bend=state[index].bend;
  s.rflg=state[index].rflg;
  for (int j=0;j<s.si;j++){
    s.aT[j]=state[index].aT[j];
    s.aS[j]=state[index].aS[j];
  }
  s.stParam=state[index].stParam;
  s.msk=(1<<s.si)-1;
  s.l=state[index].l;
  s.d=state[index].d;
  s.r=state[index].r;
  s.bm=state[index].bm;
  // s.TOPBIT=1<<(s.si-1);
  // int LASTMASK;
  // int SIDEMASK;
  // int ENDBIT;
  */
  inStruct(&s,state,index);
  //----
  // barrier(CLK_LOCAL_MEM_FENCE);
  //for(int BOUND1=0,BOUND2=s.si-2;BOUND1<s.si;BOUND1++,BOUND2--){
  int bflg=0;
  while(1){
    if(bflg==1){
      //printf("docomo\n");
      s.BOUND1--;
      s.BOUND2++;
      s.step=0;
      break;
    }
    if(s.BOUND1==s.si){
      break;
    }
    int bit;
    if(s.BOUND1==0){ // BackTrack1
      s.aB[0]=1;
      while(1){
        //printf("B1:%d\n",s.B1);
        if(bflg==1){
          s.B1--;
          break;
        }
        if(s.B1==s.si-1){
          break;
        }
        s.aB[1]=bit=(1<<s.B1);
        s.y=2;s.l=(2|bit)<<1;s.d=(1|bit);s.r=(bit>>1);
        // backTrack1(&s,s.bm);
        // backTrack1
        unsigned long j=1;
        while(1){
#ifdef GCC_STYLE
#else
          if(j==500000){
            bflg=1;
            break;
          }
#endif
          //    printf("methodstart:backtrack1\n");
          //    printf("###y:%d\n",s.y);
          //    printf("###l:%d\n",s.l);
          //    printf("###d:%d\n",s.d);
          //    printf("###r:%d\n",s.r);
          //    for(int k=0;k<s.si;k++){
          //      printf("###i:%d\n",k);
          //     printf("###aB[k]:%d\n",s.aB[k]);
          //    }
          if(s.rflg==0){
            s.bm=s.msk&~(s.l|s.d|s.r); /* 配置可能フィールド */
          }
          if (s.y==s.si-1&&s.rflg==0){ 
            // printf("if(y==si-1){\n");
            if(s.bm>0){
              // printf("if(bm>0){\n");
              s.aB[s.y]=s.bm;
              //int sum=symmetryOps_bm(&s);
              //sum=symmetryOps_bm(&s);
              //if(sum!=0){ s.lUnique++; s.lTotal+=sum; } //解を発見
              s.lTotal+=8;
              s.lUnique++;
            }
          }else{
            // printf("}else{#y==si-1\n");
            // printf("y:%d:BOUND1:%d:rflg:%d\n",s.y,s.B1,s.rflg);
            if(s.y<s.B1&&s.rflg==0){   
              // printf("if(y<BOUND1){\n");
              //printf("if(y<BOUND1){\n");
              //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
              // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
              s.bm&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
            }//end if
            // printf("}#if(y<BOUND1){\n");
            while(s.bm>0|| s.rflg==1){
              // printf("while(bm>0){\n");
              if(s.rflg==0){
                s.bm^=s.aB[s.y]=bit=(-s.bm&s.bm); //最も下位の１ビットを抽出
                // printf("beforebitmap\n");
                // printf("###y:%d\n",s.y);
                // printf("###l:%d\n",s.l);
                // printf("###d:%d\n",s.d);
                // printf("###r:%d\n",s.r);
                // printf("###bm:%d\n",s.bm);
                //       for(int k=0;k<s.si;k++){
                //         printf("###i:%d\n",k);
                //         printf("###aB[k]:%d\n",s.aB[k]);
                //       }
                if(s.stParam.current<MAX){
                  /*
                  s.stParam.param[s.stParam.current].Y=s.y;
                  s.stParam.param[s.stParam.current].I=s.si;
                  s.stParam.param[s.stParam.current].M=s.msk;
                  s.stParam.param[s.stParam.current].L=s.l;
                  s.stParam.param[s.stParam.current].D=s.d;
                  s.stParam.param[s.stParam.current].R=s.r;
                  s.stParam.param[s.stParam.current].B=s.bm;
                  */
  inParam(&s);
                  
                  (s.stParam.current)++;
                }//end if
                s.y++;
                s.l=(s.l|bit)<<1;
                s.d=(s.d|bit);
                s.r=(s.r|bit)>>1;
                s.bend=1;
                break;
              }//end if
              if(s.rflg==1){ 
                if(s.stParam.current>0){
                  s.stParam.current--;
                }
                /*
                s.si=s.stParam.param[s.stParam.current].I;
                s.y=s.stParam.param[s.stParam.current].Y;
                s.msk=s.stParam.param[s.stParam.current].M;
                s.l=s.stParam.param[s.stParam.current].L;
                s.d=s.stParam.param[s.stParam.current].D;
                s.r=s.stParam.param[s.stParam.current].R;
                s.bm=s.stParam.param[s.stParam.current].B;
                */
  outParam(&s);
                // printf("afterbitmap\n");
                // printf("###y:%d\n",s.y);
                // printf("###l:%d\n",s.l);
                // printf("###d:%d\n",s.d);
                // printf("###r:%d\n",s.r);
                // printf("###bm:%d\n",s.bm);
                //       for(int k=0;k<s.si;k++){
                //         printf("###i:%d\n",k);
                //         printf("###aB[k]:%d\n",s.aB[k]);
                //       }
                s.rflg=0;
              }//end if
            }//end while
            // printf("}:end while(bm){\n");
            if(s.bend==1 && s.rflg==0){
              s.bend=0;
              continue;
            }
          }//end else 
          // printf("}:end else\n");
          if(s.y==2){
            s.step=2;
            break;
          }else{
            s.rflg=1;
          } //end if
          j++;
        }
        s.B1=s.B1+1;
      } // end while
      if(bflg==0){
        s.SIDEMASK=s.LASTMASK=(s.TOPBIT|1);
        s.ENDBIT=(s.TOPBIT>>1);
        // printf("EB:SIDEMASK:%d\n",s.SIDEMASK);
        // printf("EB:LASTMASK:%d\n",s.LASTMASK);
      }//end if
    }else{ // BackTrack2
      if(s.BOUND1<s.BOUND2){
        s.aB[0]=bit=(1<<s.BOUND1);
        s.y=1;s.l=bit<<1;s.d=bit;s.r=bit>>1;
        // backTrack2(&s,s.bm);
        // Backtrack2
        unsigned long j=1;
        // int sum;
        while (1){
#ifdef GCC_STYLE
#else
          if(j==100){
            bflg=1;
            break;
          }
#endif
          if(s.rflg==0){
            s.bm=s.msk&~(s.l|s.d|s.r); /* 配置可能フィールド */
          }
          if (s.y==s.si-1&&s.rflg==0) {
            if(s.bm>0 && (s.bm&s.LASTMASK)==0){
              s.aB[s.y]=s.bm;
              symmetryOps_bm(&s);
            }
          }else{
            if(s.y<s.BOUND1&&s.rflg==0){             //【枝刈り】上部サイド枝刈り
              //printf("y<BOUND1\n");
              s.bm&=~s.SIDEMASK; 
            }else if(s.y==s.BOUND2&&s.rflg==0){     //【枝刈り】下部サイド枝刈り
              //printf("else if(y==BOUND2)\n");
              if((s.d&s.SIDEMASK)==0&&s.rflg==0){ 
                //printf("if((d&SIDEMASK)==0){\n");
                //goto ret2; 
                s.rflg=1;
              }
              if((s.d&s.SIDEMASK)!=s.SIDEMASK&&s.rflg==0){ 
                //printf("if((d&SIDEMASK)!=SIDEMASK){\n");
                s.bm&=s.SIDEMASK; 
              }
            } // end if
            while(s.bm>0|| s.rflg==1){
              if(s.rflg==0){
                s.bm^=s.aB[s.y]=bit=(-s.bm&s.bm); //最も下位の１ビットを抽出
                if(s.stParam.current<MAX){
  inParam(&s);
                  /*
                  s.stParam.param[s.stParam.current].Y=s.y; 
                  s.stParam.param[s.stParam.current].I=s.si;
                  s.stParam.param[s.stParam.current].M=s.msk;
                  s.stParam.param[s.stParam.current].L=s.l;
                  s.stParam.param[s.stParam.current].D=s.d;
                  s.stParam.param[s.stParam.current].R=s.r;
                  s.stParam.param[s.stParam.current].B=s.bm;
                  */
                  (s.stParam.current)++;
                }
                s.y++;
                s.l=(s.l|bit)<<1;
                s.d=(s.d|bit);
                s.r=(s.r|bit)>>1;
                s.bend=1;
                break;
              }
              if(s.rflg==1){ 
                if(s.stParam.current>0){
                  s.stParam.current--;
                }
  outParam(&s);
                /*
                s.si=s.stParam.param[s.stParam.current].I;
                s.y=s.stParam.param[s.stParam.current].Y;
                s.msk=s.stParam.param[s.stParam.current].M;
                s.l=s.stParam.param[s.stParam.current].L;
                s.d=s.stParam.param[s.stParam.current].D;
                s.r=s.stParam.param[s.stParam.current].R;
                s.bm=s.stParam.param[s.stParam.current].B;
                */
                s.rflg=0;
              }
            } // end while
            if(s.bend==1 && s.rflg==0){
              s.bend=0;
              continue;
            }
          } // end if
          if(s.y==1){
            s.step=2;
            break;
          }else{
            s.rflg=1;
          }
          j++;
        } // end while
        if(bflg==0){
          s.LASTMASK|=s.LASTMASK>>1|s.LASTMASK<<1;
          s.ENDBIT>>=1;
        }
      }// end if
    }//end if
    s.BOUND1=s.BOUND1+1;
    s.BOUND2=s.BOUND2-1;
  }//end while
  outStruct(state,&s,index);
  //----
  //    printf("for分脱出\n");
  /*
  state[index].si=s.si;
  //state[index].id=s.id;
  state[index].B1=s.B1;
  state[index].BOUND1=s.BOUND1;
  state[index].BOUND2=s.BOUND2;
  state[index].ENDBIT=s.ENDBIT;
  state[index].TOPBIT=s.TOPBIT;
  state[index].SIDEMASK=s.SIDEMASK;
  state[index].LASTMASK=s.LASTMASK;
  for (int j=0;j<s.si;j++){
    state[index].aB[j] = s.aB[j];
  }//end for
  state[index].lTotal=s.lTotal;
  state[index].lUnique=s.lUnique;
  state[index].step=s.step;
  state[index].y=s.y;
  state[index].bend=s.bend;
  state[index].rflg=s.rflg;
  for (int j=0;j<s.si;j++){
    state[index].aT[j]=s.aT[j];
    state[index].aS[j]=s.aS[j];
  }//end for
  state[index].stParam=s.stParam;
  state[index].msk=s.msk;
  state[index].l=s.l;
  state[index].d=s.d;
  state[index].r=s.r;
  state[index].bm=s.bm;
  */
}

#ifdef GCC_STYLE
int main(){
  struct queenState inProgress[MAX];
  long gTotal=0;
  printf("%s\n"," N:          Total        Unique\n");
  for(int si=4;si<17;si++){
    for(int i=0;i<1;i++){ //single
      inProgress[i].si=si;
      //inProgress[i].id=i;
      inProgress[i].B1=2;
      inProgress[i].BOUND1=0;
      inProgress[i].BOUND2=si-1;
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
      printf("%2d:%18lu%18lu\n", si,inProgress[i].lTotal,inProgress[i].lUnique);
    }
  }
  return 0;
}
#endif
