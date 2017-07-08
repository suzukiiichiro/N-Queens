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
   13．マルチスレッド(join)             NQueen13() 
   14．マルチスレッド(mutex)            NQueen14() 
 <>15．マルチスレッド(アトミック対応)   NQueen15() 
   16．アドレスとポインタ               NQueen16() 
   17．アドレスとポインタ(脱構造体)     NQueen17() 
   18．アドレスとポインタ(脱配列)       NQueen18()

 # Java/C/Lua/Bash版
 # https://github.com/suzukiiichiro/N-Queen
 

 15．マルチスレッド(アトミック対応) 
 mutex１つをロック・ロック解除で使い回すことでボトルネックが発生しました。
 また、mutexをスレッドの数だけ生成し、スレッド毎にロック/ロック解除を
 繰り返すことでオーバーヘッドは少なくなったものの、依然としてシングルスレッ
 ドよりも速度は遅くなることとなりました。
 高速化を実現するならばmutexで排他処理を行うよりも、アトミックに
 メモリアクセスする方が良さそうです。
 排他処理に必要な箇所はCOUNT++する箇所となります。
 具体的にはカウントする変数をスレッド毎の配列に格納し、
 COUNT2[BOUND1] COUNT4[BOUND1] COUNT8[BOUND1]で実装します。
 実行結果 
 
 N:           Total           Unique          hh:mm:ss.ms
 2:               0                0        0000:00:00.00
 3:               0                0        0000:00:00.00
 4:               2                1        0000:00:00.00
 5:              10                2        0000:00:00.00
 6:               4                1        0000:00:00.00
 7:              40                6        0000:00:00.00
 8:              92               12        0000:00:00.00
 9:             352               46        0000:00:00.00
10:             724               92        0000:00:00.00
11:            2680              341        0000:00:00.00
12:           14200             1787        0000:00:00.00
13:           73712             9233        0000:00:00.00
14:          365596            45752        0000:00:00.02
15:         2279184           285053        0000:00:00.12
16:        14772512          1846955        0000:00:00.78
17:        95815104         11977939        0000:00:05.05
*/

#include<stdio.h>
#include<time.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>
#define MAXSIZE 27

// pthreadはパラメータを１つしか渡せないので構造体に格納
/** スレッドローカル構造体 */
struct local{
  int BOUND1;
  int BOUND2;
  int TOPBIT;
  int ENDBIT;
  int MASK;
  int SIDEMASK;
  int LASTMASK;
  int aBoard[MAXSIZE];
  int SIZE;
  int SIZEE;
};
// mutexの配列
// pthread_mutex_t mutex[MAXSIZE];

//グローバル構造体
typedef struct {
  int SIZE; //SIZEはスレッドローカルにコピーします。
  int SIZEE;//SIZEEはスレッドローカルにコピーします。
  //カウントをスレッド毎に管理する配列 アトミック対応
  long COUNT2[MAXSIZE];
  long COUNT4[MAXSIZE];
  long COUNT8[MAXSIZE];
  long lTotal;
  long lUnique;
}GCLASS, *GClass;

GCLASS G; //グローバル構造体

/** 時刻のフォーマット変換 */
void TimeFormat(clock_t utime,char *form){
  int dd,hh,mm;
  float ftime,ss;
  //ftime=(float)utime/CLOCKS_PER_SEC;
  ftime=(float)utime/CLOCKS_PER_SEC;
  mm=(int)ftime/60;
  ss=ftime-(int)(mm*60);
  dd=mm/(24*60);
  mm=mm%(24*60);
  hh=mm/60;
  mm=mm%60;
  sprintf(form,"%7d %02d:%02d:%02.0f",dd,hh,mm,ss);
}
// /** ユニーク解のset */
void setCount2(int BOUND1){
  // pthread_mutex_lock(&mutex[BOUND1]);//ロックします
  //こういったことが必要なくなったのです ↑
  G.COUNT2[BOUND1]++;
  //こういったことが必要なくなったのです ↓
  // pthread_mutex_unlock(&mutex[BOUND1]);//ロック解除します
}
void setCount4(int BOUND1){
  // pthread_mutex_lock(&mutex[BOUND1]);//ロックします
  //こういったことが必要なくなったのです ↑
  G.COUNT4[BOUND1]++;
  //こういったことが必要なくなったのです ↓
  // pthread_mutex_unlock(&mutex[BOUND1]);//ロック解除します
}
void setCount8(int BOUND1){
  // pthread_mutex_lock(&mutex[BOUND1]);//ロックします
  //こういったことが必要なくなったのです ↑
  G.COUNT8[BOUND1]++;
  //こういったことが必要なくなったのです ↓
  // pthread_mutex_unlock(&mutex[BOUND1]);//ロック解除します
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
void symmetryOps_bitmap(int BOUND1,int BOUND2,int TOPBIT,int ENDBIT,int aBoard[],int SIZEE){
  int own,ptn,you,bit;
  //90度回転
  if(aBoard[BOUND2]==1){ own=1; ptn=2;
    while(own<=SIZEE){ bit=1; you=SIZEE;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you--; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; } own++; ptn<<=1; }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>SIZEE){ setCount2(BOUND1); return; } }
  //180度回転
  if(aBoard[SIZEE]==ENDBIT){ own=1; you=SIZEE-1;
    while(own<=SIZEE){ bit=1; ptn=TOPBIT;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; } own++; you--; }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>SIZEE){ setCount4(BOUND1); return; } }
  //270度回転
  if(aBoard[BOUND1]==TOPBIT){ own=1; ptn=TOPBIT>>1;
    while(own<=SIZEE){ bit=1; you=0;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you++; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; } own++; ptn>>=1; } } setCount8(BOUND1);
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
void backTrack2(int y,int left,int down,int right,
    int BOUND1,int BOUND2,int MASK,int SIDEMASK,int LASTMASK,
    int TOPBIT,int ENDBIT,int aBoard[],int SIZE,int SIZEE){
  //配置可能フィールド
  int bitmap=MASK&~(left|down|right); 
  int bit=0;
  if(y==SIZEE){
    if(bitmap!=0){
      //【枝刈り】最下段枝刈り
      if( (bitmap&LASTMASK)==0){ 
        aBoard[y]=bitmap;
        //対称解除法
        symmetryOps_bitmap(BOUND1,BOUND2,TOPBIT,ENDBIT,aBoard,SIZEE); } }
  }else{
    //【枝刈り】上部サイド枝刈り
    if(y<BOUND1){             
      bitmap&=~SIDEMASK; 
      // bitmap|=SIDEMASK; bitmap^=SIDEMASK;(bitmap&=~SIDEMASKと同等)
    }else if(y==BOUND2) {     
      //【枝刈り】下部サイド枝刈り
      if((down&SIDEMASK)==0){ return; }
      if((down&SIDEMASK)!=SIDEMASK){ bitmap&=SIDEMASK; } }
    while(bitmap!=0) {
      //最も下位の１ビットを抽出
      bitmap^=aBoard[y]=bit=-bitmap&bitmap;
      backTrack2(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,
          BOUND1,BOUND2,MASK,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,aBoard,SIZE,SIZEE); } }
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
void backTrack1(int y,int left,int down,int right,
    int BOUND1,int BOUND2,int MASK,int SIDEMASK,int LASTMASK,
    int TOPBIT,int ENDBIT,int aBoard[],int SIZE,int SIZEE){
  //配置可能フィールド
  int bit;
  int bitmap=MASK&~(left|down|right); 
  if(y==SIZEE) {
    if(bitmap!=0){
      aBoard[y]=bitmap;
      //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      setCount8(BOUND1); }
  }else{
    if(y<BOUND1) {   
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      // bitmap|=2; // bitmap^=2; //(bitmap&=~2と同等)
      bitmap&=~2; }
    while(bitmap!=0) {
      //最も下位の１ビットを抽出
      bitmap^=aBoard[y]=bit=(-bitmap&bitmap);
      backTrack1(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,
          BOUND1,BOUND2,MASK,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,aBoard,SIZE,SIZEE); } } 
}
void *run(void *args){
  struct local *l=(struct local *)args;
  int bit ; int SIZEE=l->SIZEE; int SIZE=l->SIZE; int aBoard[MAXSIZE]; aBoard[0]=1;
  int BOUND1=l->BOUND1; int BOUND2=l->BOUND2;
  int MASK=(1<<l->SIZE)-1; int SIDEMASK=l->SIDEMASK; int LASTMASK=l->LASTMASK;
  int TOPBIT=1<<l->SIZEE; int ENDBIT=l->ENDBIT;
  // 最上段のクイーンが角にある場合の探索
  if(BOUND1>1 && BOUND1<SIZEE) { 
    // 角にクイーンを配置 
    aBoard[1]=(1<<BOUND1);
    bit=(1<<BOUND1);
    //２行目から探索
    backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1),
        BOUND1,BOUND2,MASK,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,aBoard,SIZE,SIZEE); }
  ENDBIT=(TOPBIT>>BOUND1);
  SIDEMASK=(TOPBIT|1);
  LASTMASK=(TOPBIT|1);
  /* 最上段行のクイーンが角以外にある場合の探索 
     ユニーク解に対する左右対称解を予め削除するには、
     左半分だけにクイーンを配置するようにすればよい */
  if(BOUND1>0 && BOUND2<SIZE-1 && BOUND1<BOUND2){ 
    for(int i=1; i<BOUND1; i++){
      LASTMASK=LASTMASK|LASTMASK>>1|LASTMASK<<1; }
    if(BOUND1<BOUND2) {
      aBoard[0]=bit=(1<<BOUND1);
      backTrack2(1,bit<<1,bit,bit>>1,
          BOUND1,BOUND2,MASK,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,aBoard,SIZE,SIZEE); }
    ENDBIT>>=SIZE; }
  return 0;
}
/**********************************************/
/* マルチスレッド　排他処理  mutex            */
/**********************************************/
/**
 * マルチスレッド pthreadには排他処理 mutexがあります。
   まずmutexの宣言は以下の通りです。

      // mutexの宣言
      pthread_mutex_t mutex;   
      //pthread_mutexattr_t 変数を用意します。
      pthread_mutexattr_t mutexattr;
      // pthread_mutexattr_t 変数にロック方式を設定します。
      pthread_mutexattr_init(&mutexattr);
      //以下の第二パラメータでロック方式を指定できます。（これはとても重要です）
        PTHREAD_MUTEX_NORMAL  PTHREAD_MUTEX_FAST_NP 
        誰かがロックしているときに、それが解放されるまで永遠に待ちます。
        （同一スレッド内でのロックもブロック、その代り動作が速い）

        PTHREAD_MUTEX_RECURSIVE PTHREAD_MUTEX_RECURSIVE_NP  
        誰かがロックしているときに、それが解放されるまで永遠に待ちます。
        （同一スレッド内での２度目以降のロックは素通り）

        PTHREAD_MUTEX_ERRORCHECK  PTHREAD_MUTEX_ERRORCHECK_NP 
        誰かがロックしているときに、直ちに EDEADLK (11) を戻り値に返します。
        （同一スレッド内で 2 度目のロックがあったことを検出できる）      

        <>第 2 引数で NULL を指定した場合は、PTHREAD_MUTEX_NORMAL が指定されたのと同じになります。

      pthread_mutexattr_settype(&mutexattr, PTHREAD_MUTEX_RECURSIVE);
      // ミューテックスを初期化します。
      pthread_mutex_init(&mutex, &mutexattr);
      //pthread_mutex_init(&mutex, NULL); // 通常はこう書きますが遅いです

      実際にロックする場合はできるだけ局所的に以下の構文を挟み込むようにします。
      //pthread_mutex_lock(&mutex);
      //pthread_mutex_unlock(&mutex);
 
 * 実行部分は以下のようにロックとロック解除で処理を挟みます。
      pthread_mutex_lock(&mutex);     //ロックの開始
        COUNT2+=C2;                //保護されている処理
        COUNT4+=C4;                //保護されている処理
        COUNT8+=C8;                //保護されている処理
      pthread_mutex_unlock(&mutex);   //ロックの終了
 *
  使い終わったら破棄します。
    pthread_mutexattr_destroy(&mutexattr);//不要になった変数の破棄
    pthread_mutex_destroy(&mutex);        //nutexの破棄
 *
 */
void *NQueenThread( void *args){
  struct local l[MAXSIZE];              //構造体 local型 
  int SIZE=*(int *)args;
  int SIZEE=SIZE-1;
  pthread_t cth[SIZE];                //スレッド childThread
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
  /**
  // mutexを廃止したことで以下の宣言が不要となりました。
     //pthread_mutexattr_t 変数を用意します。
     pthread_mutexattr_t mutexattr;
     //pthread_mutexattr_t 変数にロック方式を設定します。
     pthread_mutexattr_init(&mutexattr);
     pthread_mutexattr_settype(&mutexattr, PTHREAD_MUTEX_NORMAL);
  */
  // BOUND1から順にスレッドを生成しながら処理を分担する 
  for(int BOUND1=SIZEE,BOUND2=0;BOUND2<SIZEE;BOUND1--,BOUND2++){
    //スレッド毎の変数の初期化
    l[BOUND1].BOUND1=BOUND1; l[BOUND1].BOUND2=BOUND2;
    l[BOUND1].SIZE=SIZE; l[BOUND1].SIZEE=SIZEE;
    for(int j=0;j<SIZE;j++){ l[BOUND1].aBoard[j]=j; } 
    //カウンターの初期化
    G.COUNT2[BOUND1]=G.COUNT4[BOUND1]=G.COUNT8[BOUND1]=0;
    /**
     * mutexの廃止により以下の初期化は不要となりました
      //mutex配列の初期化します。
      // pthread_mutex_init(&mutex[BOUND1], &mutexattr);
    */ 
    // マルチスレッドの生成 
    int iFbRet=pthread_create(&cth[BOUND1],NULL,run, (void *) &l[BOUND1]);
    //エラー出力デバッグ用
    if(iFbRet>0){ printf("[mainThread] pthread_create #%d: %d\n", l[BOUND1].BOUND1, iFbRet); }
    //処理を待って次の処理へ
    //以下をコメントアウトすることによってBOUND1の順次処理の度にjoinせずに並行処理する
    //コメントを外すとシングルスレッドになります。マルチスレッドではコメントアウトしておきます
    //pthread_join(cth[BOUND1],NULL);  
  }
  //処理が終わったら 全ての処理をjoinする
  for(int BOUND1=SIZEE,BOUND2=0;BOUND2<SIZEE;BOUND1--,BOUND2++){ pthread_join(cth[BOUND1],NULL); }
  /**
  //mutexの廃止によりmutexとmutexattrを破棄する処理が不要となりました。
  //不要なmutexとmutexattrを破棄
  // for(int BOUND1=SIZEE,BOUND2=0;BOUND2<SIZEE;BOUND1--,BOUND2++){
    // pthread_mutexattr_destroy(&mutexattr);//不要になった変数の破棄
    // pthread_mutex_destroy(&mutex[BOUND1]); //mutexの破棄       
  // }
  */
  //スレッド毎のカウンターを合計
  for(int BOUND1=SIZEE,BOUND2=0;BOUND2<SIZEE;BOUND1--,BOUND2++){
    G.lTotal+=G.COUNT2[BOUND1]*2+G.COUNT4[BOUND1]*4+G.COUNT8[BOUND1]*8;
    G.lUnique+=G.COUNT2[BOUND1]+G.COUNT4[BOUND1]+G.COUNT8[BOUND1];
  }
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
void NQueen(int SIZE){
  //スレッド変数
  pthread_t pth;  
  // メインスレッドの生成
  int iFbRet = pthread_create(&pth, NULL, NQueenThread,(void *) &SIZE);
  //エラー出力デバッグ用
  if(iFbRet>0){ printf("[main] pthread_create: %d\n", iFbRet); }
  //スレッドの終了を待つ
  pthread_join(pth, NULL); 
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
  printf("%s\n"," N:           Total           Unique          hh:mm:ss.ms");
  struct timeval t0;
  struct timeval t1;
  for(int i=2;i<=MAXSIZE;i++){
    //マルチスレッドの場合、これまでの計測方法ではマルチコアで処理される
    //全てのスレッドの処理時間の合計となるため、gettimeofday()で計測する
    G.lTotal=G.lUnique=0;
    gettimeofday(&t0, NULL);
    NQueen(i);     // 実行関数
    gettimeofday(&t1, NULL);
    int ss;
    int ms;
    if (t1.tv_usec < t0.tv_usec) {
      ss=(t1.tv_sec-t0.tv_sec-1)%86400;
      ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
    }
    else {
      ss=(t1.tv_sec-t0.tv_sec)%86400;
      ms=(t1.tv_usec - t0.tv_usec+500)/10000;
    }
    int hh=ss/3600;
    int mm=(ss-hh*3600)/60;
    ss%=60;
    printf("%2d:%16ld%17ld%12.4d:%02d:%02d.%02d\n", i,G.lTotal,G.lUnique,hh,mm,ss,ms); 
  } 
}

