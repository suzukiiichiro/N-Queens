/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

   １．ブルートフォース（力まかせ探索） NQueen01()
   ２．配置フラグ（制約テスト高速化）   NQueen02()
   ３．バックトラック                   NQueen03() 
   ４．対称解除法(回転と斜軸）          NQueen04() 
   ５．枝刈りと最適化                   NQueen05() 
   ６．ビットマップ                     NQueen06() 
   ７．ビットマップ+対称解除法          NQueen07() 
   ８．ビットマップ+クイーンの場所で分岐NQueen08() 
   ９．ビットマップ+枝刈りと最適化      NQueen09() 
   10．もっとビットマップ(takaken版)    NQueen10() 
   11．マルチスレッド(構造体)           NQueen11() 
   12．マルチスレッド(pthread)          NQueen12() 
   13．マルチスレッド(mutex)            NQueen13() 
 <>14．マルチスレッド(mutexattr)        NQueen14() 
   15．マルチスレッド(脱mutex COUNT強化)NQueen15() 
   15t.もっとマルチスレッド(takaken版) NQueen15_t() 
   16．アドレスとポインタ(考察１)       NQueen16() 
   17．アドレスとポインタ(考察２)       NQueen17() 
   18．アドレスとポインタ(考察３)       NQueen18()
   19．アドレスとポインタ(考察４)       NQueen19()
   20．アドレスとポインタ(考察５)       NQueen20()
   21．アドレスとポインタ(考察６)       NQueen21() 
   22．アドレスとポインタ(考察７)       NQueen22() 
   23．アドレスとポインタ(考察８)       NQueen23() 
   24．アドレスとポインタ(完結)         NQueen24() 
  
 # Java/C/Lua/Bash版
 # https://github.com/suzukiiichiro/N-Queen
 
  14．C版マルチスレッド(mutex) 
  mutexによるロックとロック解除がボトルネックになり、
　並行処理のメリットが出ません。

 N:        Total       Unique        hh:mm:ss.ms
 2:            0               0   0000:00:00.00
 3:            0               0   0000:00:00.00
 4:            2               1   0000:00:00.00
 5:           10               2   0000:00:00.00
 6:            4               1   0000:00:00.00
 7:           40               6   0000:00:00.00
 8:           92              12   0000:00:00.00
 9:          352              46   0000:00:00.00
10:          724              92   0000:00:00.00
11:         2680             341   0000:00:00.00
12:        14200            1787   0000:00:00.00
13:        73712            9233   0000:00:00.00
14:       365596           45752   0000:00:00.02
15:      2279184          285053   0000:00:00.16
16:     14772512         1846955   0000:00:01.07
17:     95815104        11977939   0000:00:07.07
*/

#include<stdio.h>
#include<time.h>
#include <sys/time.h>
#include "pthread.h"

#define MAX 27

pthread_mutex_t mutex;   //マルチスレッド排他処理mutexの宣言

//
// pthreadはパラメータを１つしか渡せないので構造体に格納

/** スレッドローカル構造体 */
struct local{
  int B1;
  int B2;
  int TB;
  int EB;
  int msk;
  int SM;
  int LM;
  int aB[MAX];
};

//グローバル構造体
typedef struct {
  int si;
  int siE;
  long C2;
  long C4;
  long C8;
}GCLASS, *GClass;
GCLASS G; 

void symmetryOps_bm(void *args);
void backTrack2(int y,int left,int down,int right,void *args);
void backTrack1(int y,int left,int down,int right,void *args);
void *run(void *args);
void *NQueenThread();
void NQueen();
long getUnique();
long getTotal();

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
void symmetryOps_bm(void *args){
  struct local *l=(struct local *)args;
  int own,ptn,you,bit;
  //90度回転
  if(l->aB[l->B2]==1){ own=1; ptn=2;
    while(own<=G.siE){ bit=1; you=G.siE;
      while((l->aB[you]!=ptn)&&(l->aB[own]>=bit)){ bit<<=1; you--; }
      if(l->aB[own]>bit){ return; } if(l->aB[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>G.siE){ 
      pthread_mutex_lock(&mutex);
      G.C2++; 
      pthread_mutex_unlock(&mutex);
      return; }
  }
  //180度回転
  if(l->aB[G.siE]==l->EB){ own=1; you=G.siE-1;
    while(own<=G.siE){ bit=1; ptn=l->TB;
      while((l->aB[you]!=ptn)&&(l->aB[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(l->aB[own]>bit){ return; } if(l->aB[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>G.siE){ 
      pthread_mutex_lock(&mutex);
      G.C4++; 
      pthread_mutex_unlock(&mutex);
      return; }
  }
  //270度回転
  if(l->aB[l->B1]==l->TB){ own=1; ptn=l->TB>>1;
    while(own<=G.siE){ bit=1; you=0;
      while((l->aB[you]!=ptn)&&(l->aB[own]>=bit)){ bit<<=1; you++; }
      if(l->aB[own]>bit){ return; } if(l->aB[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  pthread_mutex_lock(&mutex);
  G.C8++;
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
void backTrack2(int y,int left,int down,int right,void *args){
  struct local *l=(struct local *)args;
  //配置可能フィールド
  int bm=l->msk&~(left|down|right); 
  int bit=0;
  if(y==G.siE){
    if(bm>0 && (bm&l->LM)==0){ //【枝刈り】最下段枝刈り
      l->aB[y]=bm;
      symmetryOps_bm(&*l);//対称解除法
    }
  }else{
    if(y<l->B1){             //【枝刈り】上部サイド枝刈り
      bm&=~l->SM; 
    }else if(y==l->B2) {     //【枝刈り】下部サイド枝刈り
      if((down&l->SM)==0){ return; }
      if((down&l->SM)!=l->SM){ bm&=l->SM; }
    }
    while(bm>0) {
      //最も下位の１ビットを抽出
      bm^=l->aB[y]=bit=-bm&bm;
      backTrack2(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,&*l);
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
void backTrack1(int y,int left,int down,int right,void *args){
  struct local *l=(struct local *)args;
  int bit;
  int bm=l->msk&~(left|down|right); 
  if(y==G.siE) {
    if(bm>0){
      l->aB[y]=bm;
      //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      pthread_mutex_lock(&mutex);
      G.C8++;
      pthread_mutex_unlock(&mutex);
    }
  }else{
    if(y<l->B1) {   
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      bm&=~2; 
    }
    while(bm>0) {
      //最も下位の１ビットを抽出
      bm^=l->aB[y]=bit=-bm&bm;
      backTrack1(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,&*l);
    }
  } 
}
void *run(void *args){
  struct local *l=(struct local *)args;
  int bit ;
  l->aB[0]=1;
  l->msk=(1<<G.si)-1;
  l->TB=1<<G.siE;
  // 最上段のクイーンが角にある場合の探索
  if(l->B1>1 && l->B1<G.siE) { 
    if(l->B1<G.siE) {
      // 角にクイーンを配置 
      l->aB[1]=bit=(1<<l->B1);
      //２行目から探索
      backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1),&*l);
    }
  }
  l->EB=(l->TB>>l->B1);
  l->SM=l->LM=(l->TB|1);
  /* 最上段行のクイーンが角以外にある場合の探索 
     ユニーク解に対する左右対称解を予め削除するには、
     左半分だけにクイーンを配置するようにすればよい */
  if(l->B1>0&&l->B2<G.siE&&l->B1<l->B2){ 
    for(int i=1; i<l->B1; i++){
      l->LM=l->LM|l->LM>>1|l->LM<<1;
    }
    if(l->B1<l->B2) {
      l->aB[0]=bit=(1<<l->B1);
      backTrack2(1,bit<<1,bit,bit>>1,&*l);
    }
    l->EB>>=G.si;
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
 COUNT4++;
 pthread_mutex_unlock(&mutex);   //ロックの終了
 *
 使い終わったら破棄します。
 pthread_mutex_destroy(&mutex);        //nutexの破棄
 *
 */
/**
 *
 * N=8の場合は8つのスレッドがおのおののrowを担当し処理を行います。

 メインスレッド  N=8
 +--B1=7----- run()
 +--B1=6----- run()
 +--B1=5----- run()
 +--B1=4----- run()
 +--B1=3----- run()
 +--B1=2----- run()
 +--B1=1----- run()
 +--B1=0----- run()

 * そこで、それぞれのスレッド毎にスレッドローカルな構造体を持ちます。
 *
// スレッドローカル構造体 
struct local{
int bit;
int B1;
int B2;
int TB;
int EB;
int msk;
int SM;
int LM;
int aB[MAX];
};
 * 
 * スレッドローカルな構造体の宣言は以下の通りです。
 *
 *    //スレッドローカルな構造体
 *    struct local l[MAX];
 *
 * アクセスはグローバル構造体同様 . ドットでアクセスします。
 l[B1].B1=B1;
 l[B1].B2=B2;
 *
 */
void *NQueenThread(){
  pthread_t pt[G.si];                //スレッド childThread
  pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;//mutexの初期化
  pthread_mutex_init(&mutex, NULL);     //pthread 排他処理
  // スレッドローカルな構造体
  struct local l[MAX];              //構造体 local型 
  // B1から順にスレッドを生成しながら処理を分担する 
  //-- mutex 追記
  // pthread_mutexattr_t 変数を用意します。
  pthread_mutexattr_t mutexattr;
  // pthread_mutexattr_t 変数にロック方式を設定します。
  pthread_mutexattr_init(&mutexattr);
  //pthread_mutexattr_settype(&mutexattr, PTHREAD_MUTEX_RECURSIVE);
  //pthread_mutexattr_settype(&mutexattr, PTHREAD_MUTEX_NORMAL);
  // ミューテックスを初期化します。
  pthread_mutex_init(&mutex, &mutexattr);
  //pthread_mutex_init(&mutex, NULL); // 通常はこう書きますが遅いです
  //--
  for(int B1=G.siE,B2=0;B2<G.siE;B1--,B2++){
    //B1 と B2を初期化
    l[B1].B1=B1; 
    l[B1].B2=B2;
    // aB[]の初期化
    for(int j=0;j<G.si;j++){ l[l->B1].aB[j]=j; } 
    // チルドスレッドの生成
    int iFbRet=pthread_create(&pt[B1],NULL,&run,&l[B1]);
    if(iFbRet>0){
      printf("[mainThread] pthread_create #%d: %d\n", l[B1].B1, iFbRet);
    }
    //処理を待って次の処理へ
    //pthread_join(pt[B1],NULL);  
  }
  for(int i=G.siE;i>0;i--){
    pthread_join(pt[i],NULL);//処理が終わったら 全ての処理をjoinする
  }
  pthread_mutexattr_destroy(&mutexattr);//不要になった変数の破棄
  pthread_mutex_destroy(&mutex); //nutexの破棄       
  //nutexの破棄
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
*/
int main(void){
  int min=2;
  struct timeval t0;
  struct timeval t1;
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  for(int i=min;i<=MAX;i++){
    G.si=i; G.siE=i-1; G.C2=G.C4=G.C8=0;
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
    printf("%2d:%16ld%17ld%12.2d:%02d:%02d:%02d.%02d\n", i,getTotal(),getUnique(),dd,hh,mm,ss,ms); 
  } 
}
long getUnique(){ 
  return G.C2+G.C4+G.C8;
}
long getTotal(){ 
  return G.C2*2+G.C4*4+G.C8*8;
}
