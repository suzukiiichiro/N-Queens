/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

  16．マルチスレッド(アドレスとポインタ) 

		コンパイルと実行
		$ make nq16 && ./07_16NQueen


  実行結果 
 N:           Total           Unique          hh:mm:ss.ms
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
13:           73712             9233          00:00:00:00.00
14:          365596            45752          00:00:00:00.02
15:         2279184           285053          00:00:00:00.14
16:        14772512          1846955          00:00:00:00.84
17:        95815104         11977939          00:00:00:05.94
*/

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#define MAX 27

// pthreadはパラメータを１つしか渡せないので構造体に格納
//グローバル構造体
typedef struct {
  int si; //siはスレッドローカルにコピーします。
  int siE;//siEはスレッドローカルにコピーします。
  long lTotal;
  long lUnique;
}GCLASS, *GClass;

//ローカル構造体
struct local{
  int B1;
  int B2;
  int TB;
  int EB;
  int msk;
  int SM;
  int LM;
  int aB[MAX];
  int si;
  int siE;
  long COUNT2;
  long COUNT4;
  long COUNT8;
};

// mutexの配列
// pthread_mutex_t mutex[MAX];
GCLASS G; //グローバル構造体

void backTrack2(int y,int left,int down,int right,void *args);
void backTrack1(int y,int left,int down,int right,void *args);
void *run(void *args);
void *NQueenThread(void *args);
void NQueen(int si);
void symmetryOps_bm(void *args);



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
  int bit=0;
  int bm=l->msk&~(left|down|right); /* 配置可能フィールド */
  if(y==l->siE){                    //【枝刈り】
    if(bm){
      if((bm&l->LM)==0){           //【枝刈り】最下段枝刈り
        l->aB[y]=bm;
        symmetryOps_bm(l);
      }
    }
  }else{
    if(y<l->B1){             //【枝刈り】上部サイド枝刈り
      bm&=~l->SM; 
    }else if(y==l->B2) {     //【枝刈り】下部サイド枝刈り
      if((down&l->SM)==0){ return; }
      if((down&l->SM)!=l->SM){ bm&=l->SM; }
    }
    while(bm) { //最も下位の１ビットを抽出
      bm^=l->aB[y]=bit=-bm&bm;
      backTrack2(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
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
  int bit=0;                  
  int bm=l->msk&~(left|down|right); /* 配置可能フィールド */
  if(y==l->siE) {
    if(bm){//【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      l->aB[y]=bm;
      l->COUNT8++;
    }
  }else{
    if(y<l->B1) { 
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      bm&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
    while(bm) {
      bm^=l->aB[y]=bit=-bm&bm;//最も下位の１ビットを抽出
      backTrack1(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  } 
}

void *run(void *args){
  struct local *l=(struct local *)args;
  int bit ;
  l->aB[0]=1;
  l->TB=1<<l->siE;
  l->msk=(1<<l->si)-1;
  l->COUNT2=l->COUNT4=l->COUNT8=0;//初期化
  
  // 最上段のクイーンが角にある場合の探索
  if(l->B1>1&&l->B1<l->siE) {
    if(l->B1<l->siE) {
      // 角にクイーンを配置 
      l->aB[1]=bit=(1<<l->B1);
      //２行目から探索
      backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1),l);
    }
  }
  l->EB=(l->TB>>l->B1);
  l->SM=l->LM=(l->TB|1);
  /* 最上段行のクイーンが角以外にある場合の探索 
     ユニーク解に対する左右対称解を予め削除するには、
     左半分だけにクイーンを配置するようにすればよい */
  if(l->B1>0&&l->B2<l->si-1&&l->B1<l->B2){ 
    for(int i=1;i<l->B1;i++){
      l->LM=l->LM|l->LM>>1|l->LM<<1;
    }
    if(l->B1<l->B2) {
      l->aB[0]=bit=(1<<l->B1);
      backTrack2(1,bit<<1,bit,bit>>1,l);
    }
    l->EB>>=G.si;
  }
  return 0;   //*run()の場合はreturn 0;が必要
}
/**********************************************/
/*構造体*/
/**********************************************/
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


  /**********************************************/
  /* マルチスレッド　排他処理  mutex            */
  /**********************************************/
  /**
  // mutexを廃止したことで以下の宣言が不要となりました。
     //pthread_mutexattr_t 変数を用意します。
     pthread_mutexattr_t mutexattr;
     //pthread_mutexattr_t 変数にロック方式を設定します。
     pthread_mutexattr_init(&mutexattr);
     pthread_mutexattr_settype(&mutexattr, PTHREAD_MUTEX_NORMAL);
   *
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

void *NQueenThread(void *args){
  struct local l[MAX];                //構造体 local型 
  int si=*(int *)args;
  int siE=si-1;
  pthread_t cth[si];                //スレッド childThread
  // B1から順にスレッドを生成しながら処理を分担する 
  for(int B1=siE,B2=0;B2<siE;B1--,B2++){
    //スレッド毎の変数の初期化
    l[B1].B1=B1; l[B1].B2=B2;
    l[B1].si=si; l[B1].siE=siE;
    for(int j=0;j<si;j++){ l[B1].aB[j]=j; } 
    // マルチスレッドの生成 
    int iFbRet=pthread_create(&cth[B1],NULL,run, (void *) &l[B1]);
    //エラー出力デバッグ用
    if(iFbRet>0){ printf("[mainThread] pthread_create #%d: %d\n", l[B1].B1, iFbRet); }
  }
  //処理が終わったら 全ての処理をjoinする
  for(int B1=siE,B2=0;B2<siE;B1--,B2++){ pthread_join(cth[B1],NULL); }
  //スレッド毎のカウンターを合計
  for(int B1=siE,B2=0;B2<siE;B1--,B2++){
    G.lTotal+=l[B1].COUNT2*2+l[B1].COUNT4*4+l[B1].COUNT8*8;
    G.lUnique+=l[B1].COUNT2+l[B1].COUNT4+l[B1].COUNT8;
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
void NQueen(int si){
  //スレッド変数
  pthread_t pth;  
  // メインスレッドの生成
  int iFbRet=pthread_create(&pth, NULL, NQueenThread,(void *)&si);
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
 * 最大値は 先頭行でMAXをdefineしています。
 * G は グローバル構造体で宣言しています。

    //グローバル構造体
    typedef struct {
      int nThread;
      int si;
      int siE;
      long COUNT2;
      long COUNT4;
      long COUNT8;
    }GCLASS, *GClass;
    GCLASS G; //グローバル構造体

グローバル構造体を使う場合は
  G.si=i ; 
  のようにドットを使ってアクセスします。
 
  NQueen()実行関数は forの中の値iがインクリメントする度に
  Nのサイズが大きくなりクイーンの数を解法します。 
 */
int main(void){
  struct timeval t0;
  struct timeval t1;
  int min=2;
  printf("%s\n"," N:           Total           Unique          hh:mm:ss.ms");
  for(int i=min;i<=MAX;i++){
    G.lTotal=G.lUnique=0;
    gettimeofday(&t0, NULL);
    NQueen(i);     // 実行関数
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
    printf("%2d:%16ld%17ld%12.2d:%02d:%02d:%02d.%02d\n", i,G.lTotal,G.lUnique,dd,hh,mm,ss,ms); 
  } 
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

void symmetryOps_bm(void *args){
  struct local *l=(struct local *)args;
  int own,ptn,you,bit;
  //90度回転
  if(l->aB[l->B2]==1){ own=1; ptn=2;
    while(own<=l->siE){ bit=1; you=l->siE;
      while((l->aB[you]!=ptn)&&(l->aB[own]>=bit)){ bit<<=1; you--; }
      if(l->aB[own]>bit){ return; } if(l->aB[own]<bit){ break; } own++; ptn<<=1; }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>l->siE){ l->COUNT2++; return; } }
  //180度回転
  if(l->aB[l->siE]==l->EB){ own=1; you=l->siE-1;
    while(own<=l->siE){ bit=1; ptn=l->TB;
      while((l->aB[you]!=ptn)&&(l->aB[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(l->aB[own]>bit){ return; } if(l->aB[own]<bit){ break; } own++; you--; }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>l->siE){ l->COUNT4++; return; } }
  //270度回転
  if(l->aB[l->B1]==l->TB){ own=1; ptn=l->TB>>1;
    while(own<=l->siE){ bit=1; you=0;
      while((l->aB[you]!=ptn)&&(l->aB[own]>=bit)){ bit<<=1; you++; }
      if(l->aB[own]>bit){ return; } if(l->aB[own]<bit){ break; } own++; ptn>>=1; } }
  l->COUNT8++;
}

