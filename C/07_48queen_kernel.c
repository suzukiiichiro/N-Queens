﻿//  単体で動かすときは以下のコメントを外す
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
struct CL_PACKED_KEYWORD queenState {
  int BOUND1;
  int BOUND2;
  int si;
  int id;
  int aB[MAX];
  long lTotal; // Number of solutinos found so far.
  int step;
  int y;
  int startCol; // First column this individual computation was tasked with filling.
  int bm;
  long down;
  long right;
  long left;
};
CL_KERNEL_KEYWORD void place(CL_GLOBAL_KEYWORD struct queenState *state){
  int index = get_global_id(0);
  int BOUND1=state[index].BOUND1;
  int BOUND2=state[index].BOUND2;
  int si= state[index].si;
  int id= state[index].id;
  int aB[MAX];
  for (int i = 0; i < si; i++)
    aB[i]=state[index].aB[i];
  long lTotal = state[index].lTotal;
  int step      = state[index].step;
  //int y       = state[index].y;
  int startCol  = state[index].startCol;
  int bm     = state[index].bm;
  long down     = state[index].down;
  long right      = state[index].right;
  long left      = state[index].left;
  int bit;
  int msk=(1<<si)-1;
  unsigned long j=1;
  // uint16_t i = 1;
  //long i=0;
  //while (i <300000)
  while (j>0) {
#ifdef GCC_STYLE
#else
    if(j==100){
      break;
    }
#endif
    if(step==1){
      if(y<=startCol){
        step=2;
        break;
      }
      bm=aB[--y];
    }
    if(y==0){
      if(bm & (1<<BOUND1)){
        bit=1<<BOUND1;
        aB[y]=bit;
      }else{
        step=2;
        break;
      }
    }else if(y==1){
      if(bm & (1<<BOUND2)){
        bit=1<<BOUND2;
      }else{
        step=2;
        break;
      }
    }else if(y==2){
      if(bm & (1<<BOUND3)){
        bit=1<<BOUND3;
      }else{
        step=2;
        break;
      }
    }else{
      bit=bm&-bm;
      aB[y]=bit;
    }
    down^=bit;
    right^=bit<<y;
    left^=bit<<(si-1-y);
    if(step==0){
      aB[y++]=bm;
      if(y==si){
        lTotal+=1;
        step=1;
      }else{
        bm=msk&~(down|(right>>y)|(left>>((si-1)-y)));
        //bm=msk&~(down|(right>>y)|(left>>((SIZE-1)-y)));
        if(bm==0)
          step=1;
      }
    }else{
      bm^=bit;
      if(bm==0)
        step=1;
      else
        step=0;
    }
  	j++;
  }
  state[index].BOUND1=BOUND1;
  state[index].BOUND2=BOUND2;
  state[index].BOUND3=BOUND3;
  state[index].si=si;
  state[index].id=id;
  for (int i=0;i<si;i++)
  //for (int i = 0; i < SIZE; i++)
    state[index].aB[i]=aB[i];
  state[index].lTotal=lTotal;
  state[index].step=step;
  state[index].y=y;
  state[index].startCol=startCol;
  state[index].bm=bm;
  state[index].down=down;
  state[index].right=right;
  state[index].left=left;
}
#ifdef GCC_STYLE
int main(){
/*
  struct queenState l[MAX];
  printf("%s\n"," N:          Total        Unique\n");
  for(int si=4;si<MAX;si++){
    long gTotal=0;
    for (int i=0;i<si;i++){
      l[i].BOUND1=i;
      l[i].si=si;
      l[i].id=i;
      for (int j=0;j<si;j++){ l[i].aB[j]=j;}
      l[i].lTotal=0;
      l[i].step=0;
      l[i].y=0;
      l[i].startCol=1;
      l[i].bm=(1<<si)-1;
      l[i].down=0;
      l[i].right=0;
      l[i].left=0;
      place(&l[i]);
    }
    for(int i=0;i<si;i++){
      gTotal+=l[i].lTotal;
    }
*/ 
/*

  int si=10; 
  struct queenState l[SIZE*SIZE*SIZE];
  //long gTotal=0;
  uint64_t gTotal=0;
  for (int i=0;i<SIZE;i++){
    for(int j=0;j<si;j++){
      for(int k=0;k<si;k++){
    l[i*SIZE*SIZE+j*SIZE+k].BOUND1=i;
    l[i*SIZE*SIZE+j*SIZE+k].BOUND2=j;
    l[i*SIZE*SIZE+j*SIZE+k].BOUND3=k;
    l[i*SIZE*SIZE+j*SIZE+k].SIZE=SIZE;
    for (int m=0;m< SIZE;m++){
      l[i*SIZE*SIZE+j*SIZE+k].aB[m]=m;
    }
    l[i*SIZE*SIZE+j*SIZE+k].step=0;
    l[i*SIZE*SIZE+j*SIZE+k].y=0;
    l[i*SIZE*SIZE+j*SIZE+k].startCol=3;
    // l[i].msk=(1<<SIZE)-1;
    l[i*SIZE*SIZE+j*SIZE+k].bm=(1<<SIZE)-1;
    l[i*SIZE*SIZE+j*SIZE+k].down=0;
    l[i*SIZE*SIZE+j*SIZE+k].right=0;
    l[i*SIZE*SIZE+j*SIZE+k].left=0;
    l[i*SIZE*SIZE+j*SIZE+k].lTotal=0;
    place(&l[i*SIZE*SIZE+j*SIZE+k]);
    gTotal+=l[i*SIZE*SIZE+j*SIZE+k].lTotal;
    //printf("%ld\n", l[i*SIZE*SIZE+j*SIZE+k].lTotal);
    printf("%llu\n", l[i*SIZE*SIZE+j*SIZE+k].lTotal);
      }
    }
  }
    printf("%2d:%18llu\n", si,gTotal);
  // printf("%ld\n", gTotal);
  */
  return 0;
}
#endif
