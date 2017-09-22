/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
  11.マルチスレッド（構造体）

		コンパイルと実行
		$ make nq11 && ./07_11NQueen


 * N=2 から順を追って 実行関数 NQueen()を呼び出します。
 * 最大値は 先頭行でMAXをdefineしています。
 * G は グローバル構造体で宣言しています。

//グローバル構造体
typedef struct {
int nThread;
int si;
int siE;
long C2;
long C4;
long C8;
}GCLASS, *GClass;
GCLASS G; //グローバル構造体

グローバル構造体を使う場合は
G.si=i ; 
のようにドットを使ってアクセスします。

NQueen()実行関数は forの中の値iがインクリメントする度に
Nのサイズが大きくなりクイーンの数を解法します。 
  実行結果
 N:        Total       Unique        hh:mm:ss.ms
 2:               0                0          00:00:00:00.00
 3:               0                0          00:00:00:00.00
 4:               2                1          00:00:00:00.00
 5:              10                2          00:00:00:00.00
 6:               4                1          00:00:00:00.00
 7:              40                6          00:00:00:00.00
 8:              92               12          00:00:00:00.00
 9:             352               46          00:00:00:00.00
10:             724               92          00:00:00:00.00
11:            2680              341          00:00:00:00.00
12:           14200             1787          00:00:00:00.00
13:           73712             9233          00:00:00:00.01
14:          365596            45752          00:00:00:00.05
15:         2279184           285053          00:00:00:00.32
16:        14772512          1846955          00:00:00:02.08
17:        95815104         11977939          00:00:00:14.42

*/

#include<stdio.h>
#include<time.h>
#include<sys/time.h>
#define MAX 27

long getUnique();
long getTotal();
void TimeFormat(clock_t utime,char *form);
void backTrack2(int y,int l,int d,int r);
void backTrack1(int y,int left,int down,int right);
void run();
void NQueenThread();
//
// pthreadはパラメータを１つしか渡せないので構造体に格納
typedef struct {
  int si; //size
  int siE;//size-1
  int msk;//mask
  int B1; //BOUND1
  int B2; //BOUND2
  int TB; //TOPBIT
  int EB; //ENDBIT
  int SM; //SIDEMASK
  int LM; //LASTMASK
  long C2;//COUNT2
  long C4;//COUNT4
  long C8;//COUNT8
  int aB[MAX];//aBoard[]
}CLASS, *Class;
CLASS C; //構造体

void symmetryOps_bm(){
  int own,ptn,you,bit;
  //90度回転
  if(C.aB[C.B2]==1){ own=1; ptn=2;
    while(own<=C.siE){ bit=1; you=C.siE;
      while((C.aB[you]!=ptn)&&(C.aB[own]>=bit)){ bit<<=1; you--; }
      if(C.aB[own]>bit){ return; } if(C.aB[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>C.siE){ C.C2++; return; }
  }
  //180度回転
  if(C.aB[C.siE]==C.EB){ own=1; you=C.siE-1;
    while(own<=C.siE){ bit=1; ptn=C.TB;
      while((C.aB[you]!=ptn)&&(C.aB[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(C.aB[own]>bit){ return; } if(C.aB[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>C.siE){ C.C4++; return; }
  }
  //270度回転
  if(C.aB[C.B1]==C.TB){ own=1; ptn=C.TB>>1;
    while(own<=C.siE){ bit=1; you=0;
      while((C.aB[you]!=ptn)&&(C.aB[own]>=bit)){ bit<<=1; you++; }
      if(C.aB[own]>bit){ return; } if(C.aB[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  C.C8++;
}
void backTrack2(int y,int l,int d,int r){
  int bit=0;
  int bm=C.msk&~(l|d|r); 
  if(y==C.siE){
    if(bm>0&&(bm&C.LM)==0){ //【枝刈り】最下段枝刈り
      C.aB[y]=bm;
      symmetryOps_bm(); //  takakenの移植版の移植版
    }
  }else{
    if(y<C.B1){             //【枝刈り】上部サイド枝刈り
      bm&=~C.SM; 
      // bm|=SM; 
      // bm^=SM;(bm&=~SMと同等)
    }else if(y==C.B2) {     //【枝刈り】下部サイド枝刈り
      if((d&C.SM)==0){ return; }
      if((d&C.SM)!=C.SM){ bm&=C.SM; }
    }
    while(bm>0) {
      bm^=C.aB[y]=bit=-bm&bm;
      //backTrack2(y+1,(l|bit)<<1,d|bit,(r|bit)>>1);
      backTrack2(y+1,(l|bit)<<1,d|bit,(r|bit)>>1);
    }
  }
}
void backTrack1(int y,int l,int d,int r){
  int bit=0;
  int bm=C.msk&~(l|d|r); 
  if(y==C.siE) {
    if(bm>0){
      C.aB[y]=bm;
      //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      C.C8++;
    }
  }else{
    if(y<C.B1) {   
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      bm&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
    while(bm>0) {
      bm^=C.aB[y]=bit=-bm&bm;
      backTrack1(y+1,(l|bit)<<1,d|bit,(r|bit)>>1);
    }
  } 
}
void run(){
  int bit=0;
  C.aB[0]=1;
  C.TB=1<<(C.siE);
  // 最上段のクイーンが角にある場合の探索
  if(C.B1>1&&C.B1<(C.siE)) { 
    if(C.B1<(C.siE)) {
      C.aB[1]=bit=(1<<C.B1);
      backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1));
    }
  }
  C.EB=(C.TB>>C.B1);
  C.SM=C.LM=(C.TB|1);
  // 最上段のクイーンが角以外にある場合の探索
  if(C.B1>0&&C.B2<C.siE&&C.B1<C.B2){ 
    for(int i=1;i<C.B1;i++){
      C.LM=C.LM|C.LM>>1|C.LM<<1;
    }
    if(C.B1<C.B2) {
      C.aB[0]=bit=(1<<C.B1);
      backTrack2(1,bit<<1,bit,bit>>1);
    }
    C.EB>>=C.si;
  }
}
void NQueenThread(){
  for(C.B1=C.siE,C.B2=0;C.B2<C.siE;C.B1--,C.B2++){
    run();
  }
}
int main(void){
  struct timeval t0;
  struct timeval t1;
  int min=2;
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  for(int i=min;i<=MAX;i++){
    C.C2=C.C4=C.C8=0; C.si=i; C.siE=i-1; C.msk=(1<<i)-1; // 初期化
    for(int j=0;j<i;j++){ C.aB[j]=j; }
    gettimeofday(&t0, NULL);
    NQueenThread();
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
    printf("%2d:%16ld%17ld%12.2d:%02d:%02d:%02d.%02d\n", i,getTotal(),getUnique(),dd,hh,mm,ss,ms); 
  } 
}
long getUnique(){ 
  return C.C2+C.C4+C.C8;
}
long getTotal(){ 
  return C.C2*2+C.C4*4+C.C8*8;
}
