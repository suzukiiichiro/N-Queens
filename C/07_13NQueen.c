/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
   １．ブルートフォース（力まかせ探索） NQueen01()
   ２．配置フラグ（制約テスト高速化）   NQueen02()
   ３．バックトラック                   NQueen03() N17: 8:05
   ４．対称解除法(回転と斜軸）          NQueen04() N17: 7:54
   ５．枝刈りと最適化                   NQueen05() N17: 2:14
   ６．ビットマップ                     NQueen06() N17: 1:30
   ７．ビットマップ+対称解除法          NQueen07() N17: 2:24
   ８．ビットマップ+クイーンの場所で分岐NQueen08() N17: 1:26
   ９．ビットマップ+枝刈りと最適化      NQueen09() N17: 0:16
   10．もっとビットマップ(takaken版)    NQueen10() N17: 0:10
   11．マルチスレッド(構造体)           NQueen11() N17: 0:14
   12．マルチスレッド(pthread)          NQueen12() N17: 0:13
 <>13．マルチスレッド(join)             NQueen13() N17: 0:17
   14．マルチスレッド(mutex)            NQueen14() N17: 0:27
   15．マルチスレッド(アトミック対応)   NQueen15() N17: 0:05
   16．アドレスとポインタ               NQueen16() N17: 0:04
   17．アドレスとポインタ(脱構造体)     NQueen17() N17: 

 # Java/C/Lua/Bash版
 # https://github.com/suzukiiichiro/N-Queen
 
  13.マルチスレッド(join）
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
  13:        73712            9233      0 00:00:00
  14:       365596           45752      0 00:00:00
  15:      2279184          285053      0 00:00:00
  16:     14772512         1846955      0 00:00:02
  17:     95815104        11977939      0 00:00:17
 *
*/

#include<stdio.h>
#include<time.h>
#include <math.h>
#include "pthread.h"
#define MAXSIZE 27

pthread_mutex_t mutex;   //マルチスレッド排他処理mutexの宣言
pthread_cond_t cond;     // mutex varable

//
// pthreadはパラメータを１つしか渡せないので構造体に格納

/** スレッドローカル構造体 */
struct local{
  int bit;
  int BOUND1;
  int BOUND2;
  int TOPBIT;
  int ENDBIT;
  int MASK;
  int SIDEMASK;
  int LASTMASK;
  int aBoard[MAXSIZE];
};

//グローバル構造体
typedef struct {
  int nThread;
  int SIZE;
  int SIZEE;
  long COUNT2;
  long COUNT4;
  long COUNT8;
}GCLASS, *GClass;
GCLASS G; //グローバル構造体
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
void setCount(long C2,long C4,long C8){
  G.COUNT2+=C2;
  G.COUNT4+=C4;
  G.COUNT8+=C8;
}
long getUnique(){ 
  return G.COUNT2+G.COUNT4+G.COUNT8;
}
/** 総合計のget */
long getTotal(){ 
  return G.COUNT2*2+G.COUNT4*4+G.COUNT8*8;
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
void symmetryOps_bitmap(int BOUND1,int BOUND2,int MASK,int SIDEMASK,int LASTMASK,int TOPBIT,int ENDBIT,int aBoard[]){
  //Class C=(Class)args;
  //struct local *l=(struct local *)args;
  int own,ptn,you,bit;
  //90度回転
  if(aBoard[BOUND2]==1){ own=1; ptn=2;
    while(own<=G.SIZEE){ bit=1; you=G.SIZEE;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you--; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>G.SIZEE){ 
      //G.COUNT2++; 
      pthread_mutex_lock(&mutex);
      // メソッド化
      setCount(1,0,0);
      pthread_mutex_unlock(&mutex);
      return; }
  }
  //180度回転
  if(aBoard[G.SIZEE]==ENDBIT){ own=1; you=G.SIZEE-1;
    while(own<=G.SIZEE){ bit=1; ptn=TOPBIT;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>G.SIZEE){ 
      //G.COUNT4++; 
      pthread_mutex_lock(&mutex);
      // メソッド化
      setCount(0,1,0);
      pthread_mutex_unlock(&mutex);
      return; }
  }
  //270度回転
  if(aBoard[BOUND1]==TOPBIT){ own=1; ptn=TOPBIT>>1;
    while(own<=G.SIZEE){ bit=1; you=0;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you++; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  //G.COUNT8++;
  pthread_mutex_lock(&mutex);
  // メソッド化
  setCount(0,0,1);
  pthread_mutex_unlock(&mutex);
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
void backTrack2(int y,int left,int down,int right,int BOUND1,int BOUND2,int MASK,int SIDEMASK,int LASTMASK,int TOPBIT,int ENDBIT,int aBoard[]){
  //配置可能フィールド
  int bitmap=MASK&~(left|down|right); 
  int bit=0;
  if(y==G.SIZEE){
    if(bitmap>0 && (bitmap&LASTMASK)==0){ //【枝刈り】最下段枝刈り
      aBoard[y]=bitmap;
      //対称解除法
      symmetryOps_bitmap(BOUND1,BOUND2,MASK,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,aBoard); //  takakenの移植版の移植版
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
      //最も下位の１ビットを抽出
      bitmap^=aBoard[y]=bit=-bitmap&bitmap;
      //backTrack2(y+1,(left|bit)<<1,down|bit,(right|bit)>>1);
      backTrack2(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,BOUND1,BOUND2,MASK,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,aBoard);
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
//void backTrack1(int y,int left,int down,int right,void *args){
void backTrack1(int y,int left,int down,int right,int BOUND1,int BOUND2,int MASK,int SIDEMASK,int LASTMASK,int TOPBIT,int ENDBIT,int aBoard[]){
//  printf("bt1. BOUND1:%d BOUND2:%d MASK:%d SIDEMASK:%d LASTMAK:%d TOPBIT:%d ENDBIT:%d\n",BOUND1,BOUND2,MASK,SIDEMASK,LASTMASK,TOPBIT,ENDBIT);
  //配置可能フィールド
  int bitmap=MASK&~(left|down|right); 
  int bit;
  if(y==G.SIZEE) {
    if(bitmap>0){
      aBoard[y]=bitmap;
      //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      //G.COUNT8++;
      pthread_mutex_lock(&mutex);
      // メソッド化
      setCount(0,0,1);
      pthread_mutex_unlock(&mutex);
    }
  }else{
    if(y<BOUND1) {   
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      bitmap&=~2; 
      // bitmap|=2; 
      // bitmap^=2; (bitmap&=~2と同等)
    }
    while(bitmap>0) {
      //最も下位の１ビットを抽出
      bitmap^=aBoard[y]=bit=-bitmap&bitmap;
      //backTrack1(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,&l);
      backTrack1(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,BOUND1,BOUND2,MASK,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,aBoard);
    }
  } 
}
void *run(void *args){
  struct local *l=(struct local *)args;
  l->aBoard[0]=1;
  l->MASK=(1<<G.SIZE)-1;
  l->TOPBIT=1<<G.SIZEE;
  int bit ;
  // 最上段のクイーンが角にある場合の探索
  if(l->BOUND1>1 && l->BOUND1<G.SIZEE) { 
    if(l->BOUND1<G.SIZEE) {
      // 角にクイーンを配置 
      l->aBoard[1]=bit=(1<<l->BOUND1);
//      printf("  1. l->BOUND1:%d l->BOUND2:%d l->MASK:%d l->SIDEMASK:%d l->LASTMAK:%d l->TOPBIT:%d l->ENDBIT:%d\n",l->BOUND1,l->BOUND2,l->MASK,l->SIDEMASK,l->LASTMASK,l->TOPBIT,l->ENDBIT);
      /**
       *
       */
      //２行目から探索
      backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1),l->BOUND1,l->BOUND2,l->MASK,l->SIDEMASK,l->LASTMASK,l->TOPBIT,l->ENDBIT,l->aBoard);
    }
  }
  l->ENDBIT=(l->TOPBIT>>l->BOUND1);
  l->SIDEMASK=l->LASTMASK=(l->TOPBIT|1);
  /* 最上段行のクイーンが角以外にある場合の探索 
     ユニーク解に対する左右対称解を予め削除するには、
     左半分だけにクイーンを配置するようにすればよい */
  if(l->BOUND1>0&&l->BOUND2<G.SIZE-1&&l->BOUND1<l->BOUND2){ 
    for(int i=1; i<l->BOUND1; i++){
      l->LASTMASK=l->LASTMASK|l->LASTMASK>>1|l->LASTMASK<<1;
    }
    if(l->BOUND1<l->BOUND2) {
      l->aBoard[0]=bit=(1<<l->BOUND1);
      //backTrack2(1,bit<<1,bit,bit>>1,&l);
      backTrack2(1,bit<<1,bit,bit>>1,l->BOUND1,l->BOUND2,l->MASK,l->SIDEMASK,l->LASTMASK,l->TOPBIT,l->ENDBIT,l->aBoard);
    }
    l->ENDBIT>>=G.SIZE;
  }
  return 0;
}
/**********************************************/
/* マルチスレッド　排他処理                   */
/**********************************************/
/**
 * マルチスレッド pthreadには排他処理をします。
   まずmutexの宣言は以下の通りです。

      pthread_mutex_t mutex;   // mutexの宣言

 * さらにmutexは以下の方法のいずれかで初期化します。
    pthread_mutex_t m=PTHREAD_MUTEX_INITIALIZER;//mutexの初期化
    pthread_mutex_init(&mutex, NULL);     //pthread 排他処理
 
 * 実行部分は以下のようにロックとロック解除で処理を挟みます。
      pthread_mutex_lock(&mutex);     //ロックの開始
      setCount(0,0,1);                //保護されている処理
      pthread_mutex_unlock(&mutex);   //ロックの終了
 *
  使い終わったら破棄します。
    pthread_mutex_destroy(&mutex);        //nutexの破棄
 *
 */
void *NQueenThread(){
  pthread_t cth[G.SIZE];                //スレッド childThread
  pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;//mutexの初期化
  pthread_mutex_init(&mutex, NULL);     //pthread 排他処理
  /**
   *
   * N=8の場合は8つのスレッドがおのおののrowを担当し処理を行います。

        メインスレッド  N=8
            +--BOUND1=7----- run()
            +--BOUND1=6----- run()
            +--BOUND1=5----- run()
            +--BOUND1=4----- run()
            +--BOUND1=3----- run()
            +--BOUND1=2----- run()
            +--BOUND1=1----- run()
            +--BOUND1=0----- run()
    
   * そこで、それぞれのスレッド毎にスレッドローカルな構造体を持ちます。
   *
        // スレッドローカル構造体 
        struct local{
          int bit;
          int BOUND1;
          int BOUND2;
          int TOPBIT;
          int ENDBIT;
          int MASK;
          int SIDEMASK;
          int LASTMASK;
          int aBoard[MAXSIZE];
        };
   * 
   * スレッドローカルな構造体の宣言は以下の通りです。
   *
   *    //スレッドローカルな構造体
   *    struct local l[MAXSIZE];
   *
   * アクセスはグローバル構造体同様 . ドットでアクセスします。
        l[BOUND1].BOUND1=BOUND1;
        l[BOUND1].BOUND2=BOUND2;
   *
   */
  // スレッドローカルな構造体
  struct local l[MAXSIZE];              //構造体 local型 
  // BOUND1から順にスレッドを生成しながら処理を分担する 
  for(int BOUND1=G.SIZE-1,BOUND2=0;BOUND2<G.SIZE-1;BOUND1--,BOUND2++){
    //BOUND1 と BOUND2を初期化
    l[BOUND1].BOUND1=BOUND1; 
    l[BOUND1].BOUND2=BOUND2;
    // aBoard[]の初期化
    //なくてもよいみたい
    //for(int j=0;j<G.SIZE;j++){ l[l->BOUND1].aBoard[j]=j; } 
    // チルドスレッドの生成
    int iFbRet=pthread_create(&cth[BOUND1],NULL,&run,&l[BOUND1]);
    if(iFbRet>0){
      printf("[mainThread] pthread_create #%d: %d\n", l[BOUND1].BOUND1, iFbRet);
    }
    //処理を待って次の処理へ
    pthread_join(cth[BOUND1],NULL);  
  }
  //nutexの破棄
  pthread_mutex_destroy(&mutex);        
  return 0;
}
/**********************************************/
/*  マルチスレッド pthread                    */
/**********************************************/
/**
 *  マルチスレッドには pthreadを使います。
 *  pthread を宣言するには pthread_t 型の変数を宣言します。
 *
      pthread_t tId;  //スレッド変数
    
    スレッドを生成するには pthread_create()を呼び出します。
    戻り値iFbRetにはスレッドの状態が格納されます。正常作成は0になります。
    pthread_join()はスレッドの終了を待ちます。
 */
void NQueen(){
  pthread_t pth;  //スレッド変数
  // メインスレッドの生成
  int iFbRet = pthread_create(&pth, NULL, &NQueenThread, NULL);
  if(iFbRet>0){
    printf("[main] pthread_create: %d\n", iFbRet); //エラー出力デバッグ用
  }
  pthread_join(pth, NULL); //スレッドの終了を待つ
}
/**********************************************/
/*  メイン関数                                */
/**********************************************/
/**
 * N=2 から順を追って 実行関数 NQueen()を呼び出します。
 * 最大値は 先頭行でMAXSIZEをdefineしています。
 * G は グローバル構造体で宣言しています。

    //グローバル構造体
    typedef struct {
      int nThread;
      int SIZE;
      int SIZEE;
      long COUNT2;
      long COUNT4;
      long COUNT8;
    }GCLASS, *GClass;
    GCLASS G; //グローバル構造体

グローバル構造体を使う場合は
  G.SIZE=i ; 
  のようにドットを使ってアクセスします。
 
  NQueen()実行関数は forの中の値iがインクリメントする度に
  Nのサイズが大きくなりクイーンの数を解法します。 
 */
int main(void){
  clock_t st;  // 計測開始時刻
  char t[20];  // 計測結果出力
  printf("%s\n"," N:        Total       Unique        dd:hh:mm:ss");
  for(int i=2;i<=MAXSIZE;i++){
  //for(int i=2;i<=8;i++){
    G.SIZE=i;     // サイズ N
    G.SIZEE=i-1;  // サイズ N から -1　を差し引いたサイズ
    G.COUNT2=G.COUNT4=G.COUNT8=0;   // ユニークカウント格納用変数 long 型
    st=clock();   // 計測開始
    NQueen();     // 実行関数
    TimeFormat(clock()-st,t); //計測時間のフォーマット変換
    printf("%2d:%13ld%16ld%s\n",i,getTotal(),getUnique(),t); // 出力
  } 
}

