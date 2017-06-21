/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
  Java版 N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_N-Queen
  Bash版 N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_Bash
  Lua版  N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_Lua
 
  ステップバイステップでＮ−クイーン問題を最適化
   １．ブルートフォース（力まかせ探索） NQueen01()
   ２．配置フラグ（制約テスト高速化）   NQueen02()
   ３．バックトラック                   NQueen03() N16: 1:07
   ４．対称解除法(回転と斜軸）          NQueen04() N16: 1:09
   ５．枝刈りと最適化                   NQueen05() N16: 0:18
   ６．ビットマップ                     NQueen06() N16: 0:13
   ７．ビットマップ+対称解除法          NQueen07() N16: 0:21
   ８．ビットマップ+クイーンの場所で分岐NQueen08() N16: 0:13
   ９．ビットマップ+枝刈りと最適化      NQueen09() N16: 0:02
 <>10．もっとビットマップ(takaken版)    NQueen10()
   11．マルチスレッド(構造体)           NQueen11() 
   12．マルチスレッド(pthread)          NQueen12()
   13．マルチスレッド(join)             NQueen13() N16: 0:02
   14．マルチスレッド(mutex)            NQueen14() N16: 0:04
   15．マルチスレッド(アトミック対応)   NQueen15() N16: 0:00

  １０．もっとビットマップ(takaken版)

  実行結果
   N:           Total          Uniquei  days hh:mm:ss.--
   5:              10               2               0.00
   6:               4               1               0.00
   7:              40               6               0.00
   8:              92              12               0.00
   9:             352              46               0.00
  10:             724              92               0.00
  11:            2680             341               0.00
  12:           14200            1787               0.00
  13:           73712            9233               0.01
  14:          365596           45752               0.04
  15:         2279184          285053               0.26
  16:        14772512         1846955               1.70
  17:        95815104        11977939              11.69
  18:       666090624        83263591            1:24.96
*/
#include <stdio.h>
#include <time.h>
// OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif
#define  MAXSIZE   30
#define  MINSIZE    5
//#define  i64  __int64
#include <stdint.h>
#define  i64  uint64_t
//**********************************************
// Display the aBoard Image
//**********************************************
void Display(int n,int *aBoard) {
  int  y,bit,topb=1<<(n-1);
  printf("N= %d\n",n);
  for(y=0; y<n; y++){
    for(bit=topb; bit; bit>>=1)
      printf("%s ",(aBoard[y] & bit)? "Q": "-");
    printf("\n");
  }
  printf("\n");
}
//**********************************************
// Check Unique Solutions
//**********************************************
void symmetryOps_bitmap(int *aBoard,int iSize,int SIZEE,int TOPBIT,int BOUND2,int ENDBIT,int BOUND1,i64 *COUNT8,i64 *COUNT4,i64 *COUNT2) {
  int  own,you,bit,ptn;
  // 90-degree rotation
  if (aBoard[BOUND2]==1){
    for(own=1,ptn=2; own<iSize; own++,ptn<<=1){
      for(you=SIZEE,bit=1; aBoard[own]!=bit && aBoard[you]!=ptn; you--,bit<<=1);
      if (aBoard[own] != bit)return;
      if (aBoard[you] != ptn)break;
    }
    if (own==iSize){
      (*COUNT2)++;
      //Display(iSize,aBoard);
      return;
    }
  }
  // 180-degree rotation
  if (aBoard[SIZEE]==ENDBIT){
    for(own=1,you=iSize-2; own<iSize; own++,you--){
      for(ptn=TOPBIT,bit=1; aBoard[own]!=bit && aBoard[you]!=ptn; ptn>>=1,bit<<=1);
      if (aBoard[own] != bit)return;
      if (aBoard[you] != ptn)break;
    }
    if (own==iSize){
      (*COUNT4)++;
      //Display(iSize,aBoard);
      return;
    }
  }
  // 270-degree rotation
  if (aBoard[BOUND1]==TOPBIT){
    for(own=1,ptn=TOPBIT>>1; own<iSize; own++,ptn>>=1){
      for(you=0,bit=1; aBoard[own]!=bit && aBoard[you]!=ptn; you++,bit<<=1);
      if (aBoard[own] != bit)return;
      if (aBoard[you] != ptn)break;
    }
  }
  (*COUNT8)++;
  //Display(iSize,aBoard);
}
//**********************************************
// First queen is inside
//**********************************************
void backTrack2(int iSize_v,int y,int BOUND1,i64 *uniq,i64 *allc) {
  int  iSize,SIZEE,i;
  int  bitmap,bit,MASK,left,rigt;
  int  BOUND2,ENDBIT,TOPBIT,SIDEMASK,gate;
  int  aBoard[MAXSIZE];
  int  s_mask[MAXSIZE];
  int  s_left[MAXSIZE];
  int  s_rigt[MAXSIZE];
  int  s_bits[MAXSIZE];
  i64  COUNT8,COUNT4,COUNT2;
  // Initialize
  iSize=iSize_v;
  SIZEE=iSize_v-1;
  MASK=(1<<iSize_v)-1;
  COUNT8=COUNT4=COUNT2=0;
  // ControlValue
  TOPBIT=1<<SIZEE;
  SIDEMASK=TOPBIT|1;
  gate=(MASK>>y)& (MASK<<y);
  BOUND2=SIZEE-y;
  ENDBIT=TOPBIT>>y;
  // y=0: 000001110 (select)
  // y=1: 111111111 (select)
  aBoard[0]=1<<y;
  aBoard[1]=bit=1<<BOUND1;
  MASK=MASK ^ (aBoard[0]|bit);
  left=aBoard[0]<<2|bit<<1;
  rigt=aBoard[0]>>2|bit>>1;
  iSize_v=i=2;
  // y -> posc
  if (y==1)goto NEXT2;
  MASK=MASK ^ SIDEMASK;
NEXT1:
  if (i==y){
    MASK |= SIDEMASK;
    goto NEXT2;
  }
  bitmap=MASK & ~(left|rigt);
  if (bitmap){
    s_mask[i]=MASK;
    s_left[i]=left;
    s_rigt[i]=rigt;
PROC1:
    bitmap^=bit=-bitmap & bitmap;
    aBoard[i]=bit;
    s_bits[i++]=bitmap;
    MASK=MASK ^ bit;
    left=(left|bit)<<1;
    rigt=(rigt|bit)>>1;
    goto NEXT1;
BACK1:
    bitmap=s_bits[--i];
    if (bitmap){
      MASK=s_mask[i];
      left=s_left[i];
      rigt=s_rigt[i];
      goto PROC1;
    }
  }
  if (i==iSize_v)goto FINISH;
  goto BACK1;
  // posc -> BOUND2
NEXT2:
  bitmap=MASK & ~(left|rigt);
  if (bitmap){
    s_mask[i]=MASK;
    s_left[i]=left;
    s_rigt[i]=rigt;
PROC2:
    bitmap^=bit=-bitmap & bitmap;
    aBoard[i]=bit;
    s_bits[i++]=bitmap;
    MASK=MASK ^ bit;
    left=(left|bit)<<1;
    rigt=(rigt|bit)>>1;
    if (i==BOUND2){
      if (MASK & TOPBIT)goto BACK2;
      if (MASK & 1){
        if ((left|rigt)& 1)goto BACK2;
        bitmap=1;
      } else {
        bitmap=MASK & ~(left|rigt);
        if (!bitmap)goto BACK2;
      }
      goto NEXT3;
    } else {
      goto NEXT2;
    }
BACK2:
    bitmap=s_bits[--i];
    if (bitmap){
      MASK=s_mask[i];
      left=s_left[i];
      rigt=s_rigt[i];
      goto PROC2;
    }
  }
  if (i==iSize_v)goto FINISH;
  if (i>y)goto BACK2;
  goto BACK1;
  // BOUND2 -> last
NEXT3:
  if (i==SIZEE){
    if (bitmap & gate){
      aBoard[i]=bitmap;
      symmetryOps_bitmap(aBoard,iSize,SIZEE,TOPBIT,BOUND2,ENDBIT,y,&COUNT8,&COUNT4,&COUNT2);
    }
    goto BACK3;
  }
  s_mask[i]=MASK;
  s_left[i]=left;
  s_rigt[i]=rigt;
PROC3:
  bitmap^=bit=-bitmap & bitmap;
  aBoard[i]=bit;
  s_bits[i++]=bitmap;
  MASK=MASK ^ bit;
  left=(left|bit)<<1;
  rigt=(rigt|bit)>>1;
  bitmap=MASK & ~(left|rigt);
  if (bitmap)goto NEXT3;
BACK3:
  bitmap=s_bits[--i];
  if (bitmap){
    MASK=s_mask[i];
    left=s_left[i];
    rigt=s_rigt[i];
    goto PROC3;
  }
  if (i>BOUND2)goto BACK3;
  goto BACK2;
FINISH:
  *uniq=COUNT8     + COUNT4     + COUNT2;
  *allc=COUNT8*8 + COUNT4*4 + COUNT2*2;
}
//**********************************************
// First queen is in the corner
//**********************************************
void backTrack1(int y,int BOUND1,i64 *uniq,i64 *allc) {
  int  size,last,i;
  int  bitmap,bit,MASK,left,rigt;
  int  aBoard[MAXSIZE];
  int  s_mask[MAXSIZE];
  int  s_left[MAXSIZE];
  int  s_rigt[MAXSIZE];
  int  s_bitmap[MAXSIZE];
  i64  COUNT8;
  // Initialize
  size=y;
  last=y-1;
  MASK=(1<<y)-1;
  COUNT8=0;
  // y=0: 000000001 (static)
  // y=1: 011111100 (select)
  aBoard[0]=1;
  aBoard[1]=bit=1<<BOUND1;
  MASK=MASK ^ (1|bit);
  left=1<<2|bit<<1;
  rigt=1>>2|bit>>1;
  y=i=2;
  // y -> BOUND2
  MASK=MASK ^ 2;
NEXT1:
  if (i==BOUND1){
    MASK |= 2;
    goto NEXT2;
  }
  bitmap=MASK & ~(left|rigt);
  if (bitmap){
    s_mask[i]=MASK;
    s_left[i]=left;
    s_rigt[i]=rigt;
PROC1:
    bitmap^=bit=-bitmap & bitmap;
    aBoard[i]=bit;
    s_bitmap[i++]=bitmap;
    MASK=MASK ^ bit;
    left=(left|bit)<<1;
    rigt=(rigt|bit)>>1;
    goto NEXT1;
BACK1:
    bitmap=s_bitmap[--i];
    if (bitmap){
      MASK=s_mask[i];
      left=s_left[i];
      rigt=s_rigt[i];
      goto PROC1;
    }
  }
  if (i==y)goto FINISH;
  goto BACK1;
  // BOUND2 -> last
NEXT2:
  bitmap=MASK & ~(left|rigt);
  if (bitmap){
    if (i==last){
      aBoard[i]=bitmap;
      COUNT8++;
      //Display(size,aBoard);
      goto BACK2;
    }
    s_mask[i]=MASK;
    s_left[i]=left;
    s_rigt[i]=rigt;
PROC2:
    bitmap^=bit=-bitmap & bitmap;
    aBoard[i]=bit;
    s_bitmap[i++]=bitmap;
    MASK=MASK ^ bit;
    left=(left|bit)<<1;
    rigt=(rigt|bit)>>1;
    goto NEXT2;
BACK2:
    bitmap=s_bitmap[--i];
    if (bitmap){
      MASK=s_mask[i];
      left=s_left[i];
      rigt=s_rigt[i];
      goto PROC2;
    }
  }
  if (i==y)goto FINISH;
  if (i>BOUND1)goto BACK2;
  goto BACK1;
FINISH:
  *uniq=COUNT8;
  *allc=COUNT8*8;
}
//**********************************************
// Search of N-Queens
//**********************************************
void NQueen(int iSize,i64 *unique,i64 *allcnt) {
  int  y,BOUND1;
  i64  uniq,allc;
  *unique=*allcnt=0;
  for(y=0; y<iSize/2; y++){
// OpenMP
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(BOUND1=0; BOUND1<iSize; BOUND1++){
      if (y==0){
        // y=0: 000000001 (static)
        // y=1: 011111100 (select)
        if (BOUND1<2 || BOUND1==iSize-1)continue;
        backTrack1(iSize,BOUND1,&uniq,&allc);
      } else {
        // y=0: 000001110 (select)
        // y=1: 111111111 (select)
        if (BOUND1>=y-1 && BOUND1<=y+1)continue;
        if (y>1 && (BOUND1==0 || BOUND1==iSize-1))continue;
        backTrack2(iSize,y,BOUND1,&uniq,&allc);
      }
      *unique += uniq;
      *allcnt += allc;
    }
  }
}
//**********************************************
// Format of Used Time
//**********************************************
void TimeFormat(clock_t utime,char *form) {
  int  dd,hh,mm;
  double ftime,ss;
  ftime=(double)utime/CLOCKS_PER_SEC;
  mm=(int)ftime/60;
  ss=ftime-(double)(mm*60);
  dd=mm/(24*60);
  mm=mm % (24*60);
  hh=mm/60;
  mm=mm % 60;
  if (dd)sprintf(form,"%4d %02d:%02d:%05.2f",dd,hh,mm,ss);
  else if (hh)sprintf(form,"     %2d:%02d:%05.2f",hh,mm,ss);
  else if (mm)sprintf(form,"        %2d:%05.2f",mm,ss);
  else sprintf(form,"           %5.2f",ss);
}
//**********************************************
// N-Queens Solutions MAIN
//**********************************************
int main(void) {
  i64  unique,allcnt;
  clock_t starttime;
  char form[20],line[100];
  printf("<------  N-Queens Solutions  -----> <---- time ---->\n");
  printf(" N:           Total          Unique days hh:mm:ss.--\n");
  for(int i=MINSIZE; i<=MAXSIZE; i++){
    starttime=clock();
    NQueen(i,&unique,&allcnt);
    TimeFormat(clock()-starttime,form);
    // sprintf(line,"%2d:%16I64d%16I64d %s\n",n,allcnt,unique,form);
    sprintf(line,"%2d:%16llu%16llu%s\n",i,allcnt,unique,form);
    printf("%s",line);
  }
  return 0;
}
