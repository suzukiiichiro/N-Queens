/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイル
 $ gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm -pthread 07_12N-Queen.c -o 12N-Queen

 実行
 $ ./11N-Queen

 １２．並列処理：pthreadと構造体
 *
 *  実行結果
 N:           Total           Unique          dd:hh:mm:ss.ms
 4:               2                1          00:00:00:00.00
 5:              10                2          00:00:00:00.00
 6:               4                1          00:00:00:00.00
 7:              40                6          00:00:00:00.00
 8:              92               12          00:00:00:00.00
 9:             352               46          00:00:00:00.00
10:             724               92          00:00:00:00.00
11:            2680              341          00:00:00:00.00
12:           14200             1787          00:00:00:00.00
13:           73712             9233          00:00:00:00.00
14:          365596            45752          00:00:00:00.02
15:         2279184           285053          00:00:00:00.12
16:        14772512          1846955          00:00:00:00.66
17:        95815104         11977939          00:00:00:04.30
*/

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#define MAX 24

// pthreadはパラメータを１つしか渡せないので構造体に格納
//グローバル構造体
typedef struct {
  int size;
  int sizeE;
  long lTOTAL,lUNIQUE;
}GCLASS, *GClass;
GCLASS G;

//ローカル構造体
typedef struct{
  int BOUND1,BOUND2,TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
  int mask;
  int aBoard[MAX];
  long COUNT2[MAX],COUNT4[MAX],COUNT8[MAX];
}local ;

void symmetryOps(local *l);
void backTrack2(int y,int left,int down,int right,local *l);
void backTrack1(int y,int left,int down,int right,local *l);
void *run(void *args);
void *NQueenThread();
void NQueen();

void symmetryOps(local *l){
  int own,ptn,you,bit;
  //90度回転
  if(l->aBoard[l->BOUND2]==1){ own=1; ptn=2;
    while(own<=G.sizeE){ bit=1; you=G.sizeE;
      while((l->aBoard[you]!=ptn)&&(l->aBoard[own]>=bit)){ bit<<=1; you--; }
      if(l->aBoard[own]>bit){ return; } if(l->aBoard[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>G.sizeE){
      l->COUNT2[l->BOUND1]++;
      return; }
  }
  //180度回転
  if(l->aBoard[G.sizeE]==l->ENDBIT){ own=1; you=G.sizeE-1;
    while(own<=G.sizeE){ bit=1; ptn=l->TOPBIT;
      while((l->aBoard[you]!=ptn)&&(l->aBoard[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(l->aBoard[own]>bit){ return; } if(l->aBoard[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>G.sizeE){
       l->COUNT4[l->BOUND1]++;
      return; }
  }
  //270度回転
  if(l->aBoard[l->BOUND1]==l->TOPBIT){ own=1; ptn=l->TOPBIT>>1;
    while(own<=G.sizeE){ bit=1; you=0;
      while((l->aBoard[you]!=ptn)&&(l->aBoard[own]>=bit)){ bit<<=1; you++; }
      if(l->aBoard[own]>bit){ return; } if(l->aBoard[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  l->COUNT8[l->BOUND1]++;
}
/**********************************************/
/* 最上段行のクイーンが角以外にある場合の探索 */
/**********************************************/
/**
  １行目角にクイーンが無い場合、クイーン位置より右位置の８対称位置にクイーンを
  置くことはできない
  １行目位置が確定した時点で、配置可能位置を計算しておく（☓の位置）
  lt, dn, lt 位置は効きチェックで配置不可能となる
  回転対称チェックが必要となるのは、クイーンがａ, ｂ, ｃにある場合だけなので、
  90度、180度、270度回転した状態のユニーク判定値との比較を行うだけで済む

  【枝刈り図】
  x x - - - Q x x
  x - - - / | ＼x
  c - - / - | -rt
  - - / - - | - -
  - / - - - | - -
  lt- - - - | - a
  x - - - - | - x
  x x b - - dnx x
  */
void backTrack2(int row,int left,int down,int right,local *l){
	int bit;
	int bitmap=l->mask&~(left|down|right);
	if(row==G.sizeE){ 								// 【枝刈り】
		if(bitmap){
			if((bitmap&l->LASTMASK)==0){ 	//【枝刈り】 最下段枝刈り
				l->aBoard[row]=bitmap;
				symmetryOps(l);
			}
		}
	}else{
    if(row<l->BOUND1){             	//【枝刈り】上部サイド枝刈り
      bitmap&=~l->SIDEMASK;
    }else if(row==l->BOUND2) {     	//【枝刈り】下部サイド枝刈り
      if((down&l->SIDEMASK)==0){ return; }
      if((down&l->SIDEMASK)!=l->SIDEMASK){ bitmap&=l->SIDEMASK; }
    }
		while(bitmap){
			bitmap^=l->aBoard[row]=bit=(-bitmap&bitmap);
			backTrack2(row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
		}
	}
}
/**********************************************/
/* 最上段行のクイーンが角にある場合の探索     */
/**********************************************/
/*
   １行目角にクイーンがある場合、回転対称形チェックを省略することが出来る
   １行目角にクイーンがある場合、他の角にクイーンを配置することは不可
   鏡像についても、主対角線鏡像のみを判定すればよい
   ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
*/
void backTrack1(int row,int left,int down,int right,local *l){
	int bit;
	int bitmap=l->mask&~(left|down|right);
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
  if(row==G.sizeE) {
    if(bitmap){
      l->aBoard[row]=bitmap;
      l->COUNT8[l->BOUND1]++;
    }
  }else{
		//【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
		// ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
    if(row<l->BOUND1) {
      bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
		while(bitmap){
			bitmap^=l->aBoard[row]=bit=(-bitmap&bitmap);
			backTrack1(row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
		}
	}
}
void *run(void *args){
	local *l=(local *)args;
  int bit ;
  l->aBoard[0]=1;
  l->TOPBIT=1<<G.sizeE;
  l->mask=(1<<G.size)-1;
  // 最上段のクイーンが角にある場合の探索
  if(l->BOUND1>1 && l->BOUND1<G.sizeE) {
    if(l->BOUND1<G.sizeE) {
      // 角にクイーンを配置
      l->aBoard[1]=bit=(1<<l->BOUND1);
      //２行目から探索
      backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1),l);
    }
  }
  l->ENDBIT=(l->TOPBIT>>l->BOUND1);
  l->SIDEMASK=l->LASTMASK=(l->TOPBIT|1);
  /* 最上段行のクイーンが角以外にある場合の探索
     ユニーク解に対する左右対称解を予め削除するには、
     左半分だけにクイーンを配置するようにすればよい */
  if(l->BOUND1>0&&l->BOUND2<G.sizeE&&l->BOUND1<l->BOUND2){
    for(int i=1; i<l->BOUND1; i++){
      l->LASTMASK=l->LASTMASK|l->LASTMASK>>1|l->LASTMASK<<1;
    }
    if(l->BOUND1<l->BOUND2) {
      l->aBoard[0]=bit=(1<<l->BOUND1);
      backTrack2(1,bit<<1,bit,bit>>1,l);
    }
    l->ENDBIT>>=G.size;
  }
  return 0;   //*run()の場合はreturn 0;が必要
}
void *NQueenThread(){
  local l[MAX];                //構造体 local型
  pthread_t pt[G.size];                 //スレッド childThread
  for(int BOUND1=G.sizeE,BOUND2=0;BOUND2<G.sizeE;BOUND1--,BOUND2++){
    l[BOUND1].BOUND1=BOUND1; l[BOUND1].BOUND2=BOUND2;         //B1 と B2を初期化
    for(int j=0;j<G.size;j++){ l[l->BOUND1].aBoard[j]=j; } // aB[]の初期化
    l[BOUND1].COUNT2[BOUND1]=l[BOUND1].COUNT4[BOUND1]=l[BOUND1].COUNT8[BOUND1]=0;//カウンターの初期化
    // チルドスレッドの生成
    int iFbRet=pthread_create(&pt[BOUND1],NULL,&run,&l[BOUND1]);
    if(iFbRet>0){
      printf("[mainThread] pthread_create #%d: %d\n", l[BOUND1].BOUND1, iFbRet);
    }
  }
  for(int BOUND1=G.sizeE,BOUND2=0;BOUND2<G.sizeE;BOUND1--,BOUND2++){
    pthread_join(pt[BOUND1],NULL);
  }
  //スレッド毎のカウンターを合計
  for(int BOUND1=G.sizeE,BOUND2=0;BOUND2<G.sizeE;BOUND1--,BOUND2++){
    G.lTOTAL+=l[BOUND1].COUNT2[BOUND1]*2+l[BOUND1].COUNT4[BOUND1]*4+l[BOUND1].COUNT8[BOUND1]*8;
    G.lUNIQUE+=l[BOUND1].COUNT2[BOUND1]+l[BOUND1].COUNT4[BOUND1]+l[BOUND1].COUNT8[BOUND1];
  }
  return 0;
}
void NQueen(){
  pthread_t pth;  //スレッド変数
  // メインスレッドの生成
  int iFbRet = pthread_create(&pth, NULL, &NQueenThread, NULL);
  if(iFbRet>0){
    printf("[main] pthread_create: %d\n", iFbRet); //エラー出力デバッグ用
  }
  pthread_join(pth,NULL); /* いちいちjoinをする */
}
int main(void){
  struct timeval t0;
  struct timeval t1;
  int min=4;
  printf("%s\n"," N:           Total           Unique          dd:hh:mm:ss.ms");
  for(int i=min;i<=MAX;i++){
    G.size=i; G.sizeE=i-1; //初期化
    G.lTOTAL=G.lUNIQUE=0;
    gettimeofday(&t0, NULL);
    NQueen();
    gettimeofday(&t1, NULL);
    int ss;int ms;int dd;
    if (t1.tv_usec<t0.tv_usec) {
      dd=(t1.tv_sec-t0.tv_sec-1)/86400;
      ss=(t1.tv_sec-t0.tv_sec-1)%86400;
      ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
    } else {
      dd=(t1.tv_sec-t0.tv_sec)/86400;
      ss=(t1.tv_sec-t0.tv_sec)%86400;
      ms=(t1.tv_usec-t0.tv_usec+500)/10000;
    }
    int hh=ss/3600;
    int mm=(ss-hh*3600)/60;
    ss%=60;
    printf("%2d:%16ld%17ld%12.2d:%02d:%02d:%02d.%02d\n", i,G.lTOTAL,G.lUNIQUE,dd,hh,mm,ss,ms);
  }
}
