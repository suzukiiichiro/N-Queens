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
 <>９．ビットマップ+枝刈りと最適化      NQueen09() N16: 0:02
   10．もっとビットマップ               NQueen10()
   11．マルチスレッド                   NQueen11()

  実行結果
   N:        Total       Unique        dd:hh:mm:ss
   2:            0               0      0 00:00:00
   3:            0               0      0 00:00:00
   4:            2               1      0 00:00:00
   5:           10               2      0 00:00:00
   6:            4               1      0 00:00:00
   7:           40               6      0 00:00:00
   8:           92              12      0 00:00:00
   9:          352              46      0 00:00:00
  10:          724              92      0 00:00:00
  11:         2680             341      0 00:00:00
  12:        14200            1787      0 00:00:00
  13:        73664            9227      0 00:00:00
  14:       365492           45739      0 00:00:00
  15:      2278664          284988      0 00:00:00
  16:     14768296         1846428      0 00:00:02
  17:     95795792        11975525      0 00:00:14
  18:    665976024        83249266      0 00:01:44
  19:   4967407864       620931506      0 00:13:20
  20:  39024327300      4878059110      0 01:47:04
  21: 314633809992     39329273383      0 15:07:13
*/

#include<stdio.h>
#include<time.h>
#include <math.h>
#include "pthread.h"
#define MAXSIZE 27

long lTotal=1 ; //合計解
long lUnique=0; //ユニーク解
long COUNT2=0; long COUNT4=0; long COUNT8=0;
int aBoard[MAXSIZE];  //チェス盤の横一列
pthread_mutex_t mutex;
int iSize;
//int aTrial[MAXSIZE];
//int aScratch[MAXSIZE];
//int MASK;
//int bit;
//int BOUND1;
//int BOUND2;
//int TOPBIT;
//int SIZEE;
//int SIDEMASK;
//int LASTMASK;
//int ENDBIT;

/** 時刻のフォーマット変換 */
void TimeFormat(clock_t utime,char *form){
    int dd,hh,mm;
    float ftime,ss;
    ftime=(float)utime/CLOCKS_PER_SEC;
    mm=(int)ftime/60;
    ss=ftime-(int)(mm*60);
    dd=mm/(24*60);
    mm=mm%(24*60);
    hh=mm/60;
    mm=mm%60;
    sprintf(form,"%7d %02d:%02d:%02.0f",dd,hh,mm,ss);
}
/** ユニーク解のget */
long getUnique(){ 
  return COUNT2+COUNT4+COUNT8;
}
/** 総合計のget */
long getTotal(){ 
  return COUNT2*2+COUNT4*4+COUNT8*8;
}
/**********************************************/
/** 対称解除法                               **/
/** ユニーク解から全解への展開               **/
/**********************************************/
/**
ひとつの解には、盤面を90度・180度・270度回転、及びそれらの鏡像の合計8個の対称解が存在する

    １２ ４１ ３４ ２３
    ４３ ３２ ２１ １４

    ２１ １４ ４３ ３２
    ３４ ２３ １２ ４１
    
上図左上がユニーク解。
1行目はユニーク解を90度、180度、270度回転したもの
2行目は1行目のそれぞれを左右反転したもの。
2行目はユニーク解を左右反転、対角反転、上下反転、逆対角反転したものとも解釈可 
ただし、 回転・線対称な解もある
**/
/**
クイーンが右上角にあるユニーク解を考えます。
斜軸で反転したパターンがオリジナルと同型になることは有り得ないことと(×２)、
右上角のクイーンを他の３つの角に写像させることができるので(×４)、
このユニーク解が属するグループの要素数は必ず８個(＝２×４)になります。

(1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルから180度回転)
　　させても、さらに90度回転(オリジナルから270度回転)させてもオリジナルと同型になる。 
(2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナルとは異なる。
　　ただし、180度回転させた場合はオリジナルと同型になることも有り得る。

　(1)に該当するユニーク解が属するグループの要素数は、左右反転させたパターンを加えて
２個しかありません。(2)に該当するユニーク解が属するグループの要素数は、180度回転させ
て同型になる場合は４個(左右反転×縦横回転)、そして180度回転させてもオリジナルと異なる
場合は８個になります。(左右反転×縦横回転×上下反転)
*/
void symmetryOps_bitmap(int BOUND1,int BOUND2,int TOPBIT,int ENDBIT,int SIZEE,int bit){
  int own,ptn,you;
  //90度回転
  if(aBoard[BOUND2]==1){ own=1; ptn=2;
    while(own<=SIZEE){ bit=1; you=SIZEE;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you--; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>SIZEE){ COUNT2++; return; }
  }
  //180度回転
  if(aBoard[SIZEE]==ENDBIT){ own=1; you=SIZEE-1;
    while(own<=SIZEE){ bit=1; ptn=TOPBIT;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>SIZEE){ COUNT4++; return; }
  }
  //270度回転
  if(aBoard[BOUND1]==TOPBIT){ own=1; ptn=TOPBIT>>1;
    while(own<=SIZEE){ bit=1; you=0;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you++; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  COUNT8++;
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
//void backTrack2(int y,int left,int down,int right){
void backTrack2(int y,int left,int down,int right,int BOUND1,int BOUND2,int MASK,int SIDEMASK,int LASTMASK,int TOPBIT,int ENDBIT,int SIZEE){
  int bitmap=MASK&~(left|down|right); 
  int bit=0;
  if(y==SIZEE){
    if(bitmap>0 && (bitmap&LASTMASK)==0){ //【枝刈り】最下段枝刈り
      aBoard[y]=bitmap;
      //symmetryOps_bitmap(); //  takakenの移植版の移植版
      symmetryOps_bitmap(BOUND1,BOUND2,TOPBIT,ENDBIT,SIZEE,bit); //  takakenの移植版の移植版
      //symmetryOps_bitmap_old();// 兄が作成した労作
      //symmetryOps_bitmap_old(BOUND1,BOUND2,TOPBIT,ENDBIT);// 兄が作成した労作
    }
  }else{
    if(y<BOUND1){             //【枝刈り】上部サイド枝刈り
      bitmap&=~SIDEMASK; 
      // bitmap|=SIDEMASK; 
      // bitmap^=SIDEMASK;(bitmap&=~SIDEMASKと同等)
    }else if(y==BOUND2) {     //【枝刈り】下部サイド枝刈り
      if((down&SIDEMASK)==0){ return; }
      if((down&SIDEMASK)!=SIDEMASK){ bitmap&=SIDEMASK; }
    }
    while(bitmap>0) {
      bitmap^=aBoard[y]=bit=-bitmap&bitmap;
      //backTrack2(y+1,(left|bit)<<1,down|bit,(right|bit)>>1);
      backTrack2(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,BOUND1,BOUND2,MASK,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,SIZEE);
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
void backTrack1(int y,int left,int down,int right,int BOUND1,int MASK,int SIZEE){
  int bitmap=MASK&~(left|down|right); 
  int bit;
  if(y==SIZEE) {
    if(bitmap>0){
      aBoard[y]=bitmap;
      //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      COUNT8++;
    }
  }else{
    if(y<BOUND1) {   
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      bitmap&=~2; // bitmap|=2; bitmap^=2; (bitmap&=~2と同等)
    }
    while(bitmap>0) {
      bitmap^=aBoard[y]=bit=-bitmap&bitmap;
      backTrack1(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,BOUND1,MASK,SIZEE);
    }
  } 
}
// pthreadはパラメータを１つしか渡せないので構造体に格納
typedef struct {
  int BOUND1;
  int BOUND2;
  pthread_t PTID;
  int SIZEE;
}CLASS, *Class;

void *run(void *args){
//  pthread_mutex_init(&mutex, NULL);
//  pthread_mutex_lock(&mutex);
//  pthread_mutex_unlock(&mutex);

  Class C=(Class)args;
  int BOUND1=C->BOUND1;
  int BOUND2=C->BOUND2;
  pthread_t PTID=C->PTID; 
  int SIZEE=C->SIZEE;
  
  // 最上段のクイーンが角以外にある場合の探索
  aBoard[0]=1;
  int MASK=(1<<iSize)-1;
  int TOPBIT=1<<SIZEE;
  int bit ;
  if(BOUND1>1&&BOUND1<SIZEE) { 
    if(BOUND1<SIZEE) {
      aBoard[1]=bit=(1<<BOUND1);
      backTrack1(2, (2|bit)<<1, (1|bit), (bit>>1),BOUND1,MASK,SIZEE);
    }
  }
  int ENDBIT=(TOPBIT>>BOUND1);
  int SIDEMASK=(TOPBIT|1);
  int LASTMASK=(TOPBIT|1);
  // 最上段のクイーンが角にある場合の探索
  if(BOUND1>0 && BOUND2<iSize-1 && BOUND1<BOUND2){ 
    for(int i=1; i<BOUND1; i++){
      LASTMASK=LASTMASK|LASTMASK>>1|LASTMASK<<1;
    }
    if(BOUND1<BOUND2) {
      aBoard[0]=bit=(1<<BOUND1);
      backTrack2(1, bit<<1, bit, bit>>1,BOUND1,BOUND2,MASK,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,SIZEE);
    }
    ENDBIT>>=iSize;
  }
  return 0;
}
pthread_t pt;
void NQueenThread(){
  CLASS C; //構造体
  for(int BOUND1=iSize-1,BOUND2=0;BOUND2<iSize-1;BOUND1--,BOUND2++){
    C.BOUND1=BOUND1;//構造体に値を格納
    C.BOUND2=BOUND2;//構造体に値を格納
    //C.PTID=pt;
    C.SIZEE=iSize-1;
    pthread_create(&pt,NULL,&run,&C);
    pthread_join(pt, NULL);
  }
}
int main(void){
  clock_t st; char t[20];
  printf("%s\n"," N:        Total       Unique        dd:hh:mm:ss");
  for(int i=2;i<=MAXSIZE;i++){
    iSize=i;lTotal=0; lUnique=0;
	  COUNT2=COUNT4=COUNT8=0;
    for(int j=0;j<i;j++){ aBoard[j]=j; }
    st=clock();
    //NQueen(0,0,0,0);
    NQueenThread();
    TimeFormat(clock()-st,t);
    printf("%2d:%13ld%16ld%s\n",i,getTotal(),getUnique(),t);
  } 
}

