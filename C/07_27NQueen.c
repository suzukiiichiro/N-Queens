/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 <>27．もっとマルチスレッド(NxN) 2段目	        NQueen27() N17=00.78

		コンパイルと実行
		$ make nq27 && ./07_27NQueen

  BackTrack2について修正。
  BackTrack2の処理を最上段のクイーンの場所のスレッドに加えて、
  上から２段目のクイーンも固定しスレッド化。((N-1)xN)

  pthread_t pt1[si];//スレッド childThread
  pthread_t pt2[si][si];//スレッド childThread


          backTrack1                   backTrack2
  最上段の角にクイーンがある   最上段の済みにクイーンがない
          1   2   3   4                1   2   3   4
        +   +   +   +   *            +   +   +   +   *
  1                   Q        1       Q   Q   Q          
        +   +   +   +   *            +   +   +   +   *
  2       Q   Q   Q   Q        2       Q   Q   Q   Q    ←ここを新設した。
        +   +   +   +   *            +   +   +   +   *
  3                            3       
        +   +   +   +   *            +   +   +   +   *
  4                            4
        +   +   +   +   *            +   +   +   +   *



  26．マルチスレッド再びでは、run()を、 最上段行のクイーンが角にある場合のrun()と、
      最上段行のクイーンが角にない場合のrun2に分解し、 それぞれをスレッド化(兄の労作）したが、
      今回は、backtrack2の処理を最上段から２行目の列数分さらに分割しスレッド化
      ロジック的にはN数倍高速化されるはず。（今回も兄の労作）
			以下の通り、マウントされているCPUが全てフルに稼働していることがわかる。

       87654321 backtrack1        
 1行目□□□□□□□□               backtrack1は1行目が右端の場合の処理  
 2行目□□□□□□□□            □  1行目は右端固定  
 3行目□□□□□□□□    □□□□□□□□□  2行目はB1の値に配置
 4行目□□□□□□□□               スレッド数は 1xN=N個        
 5行目□□□□□□□□   
 6行目□□□□□□□□  backtrack2
 7行目□□□□□□□□               backtrack2は1行目のクイーン位置が右端の場合以外 
 8行目□□□□□□□□  □□□□□□□□     1行目が右端以外 B1の値に配置
                             スレッド数は N-1個
 
  

top - 00:43:00 up 3 days, 21:28,  2 users,  load average: 161.01, 161.03, 161.00
Tasks: 296 total,   1 running, 295 sleeping,   0 stopped,   0 zombie
%Cpu0  :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu1  :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu2  :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu3  :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu4  :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu5  :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu6  :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu7  :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu8  :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu9  :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu10 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu11 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu12 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu13 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu14 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu15 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu16 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu17 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu18 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu19 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu20 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu21 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu22 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu23 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu24 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu25 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu26 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu27 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu28 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu29 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu30 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu31 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu32 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu33 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu34 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu35 :100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
KiB Mem : 61403004 total, 60445832 free,   485444 used,   471728 buff/cache
KiB Swap:        0 total,        0 free,        0 used. 60359736 avail Mem 


    実行結果
 N:        Total       Unique                 dd:hh:mm:ss.ms
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
14:          365596            45752          00:00:00:00.01
15:         2279184           285053          00:00:00:00.07
16:        14772512          1846955          00:00:00:00.45
17:        95815104         11977939          00:00:00:03.23
18:       666090624         83263591          00:00:00:27.21

  参考（Bash版 07_8NQueen.lua）
  13:           73712             9233                99
  14:          365596            45752               573
  15:         2279184           285053              3511

  参考（Lua版 07_8NQueen.lua）
  14:          365596            45752          00:00:00
  15:         2279184           285053          00:00:03
  16:        14772512          1846955          00:00:20

  参考（Java版 NQueen8.java マルチスレット）
  16:        14772512          1846955          00:00:00
  17:        95815104         11977939          00:00:04
  18:       666090624         83263591          00:00:34
  19:      4968057848        621012754          00:04:18
  20:     39029188884       4878666808          00:35:07
  21:    314666222712      39333324973          04:41:36
  22:   2691008701644     336376244042          39:14:59
 *
*/

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#define MAX 27

#ifdef _GNU_SOURCE
/** cpu affinityを有効にするときは以下の１行（#define _GNU_SOURCE)を、
 * #ifdef _GNU_SOURCE の上に移動 
 * CPU Affinity はLinuxのみ動作します。　Macでは動きません*/
#define _GNU_SOURCE   
#include <sched.h> 
#include <unistd.h>
#include <sys/syscall.h>
#include <errno.h>
#define handle_error_en(en, msg) do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)
#endif

//ローカル構造体
typedef struct{
	int bit;
  int own;
	int ptn;
	int you;
  int k;
  int B1;
  int B2;
  int TB;
  int EB;
  int msk;
  int SM;
  int LM;
  int aB[MAX];
  long C2[MAX][2];
  long C4[MAX][2];
  long C8[MAX][2];
  int BK;
}local ;

int si;  
int siE;
long lTotal;
long lUnique;

void backTrack3(int y,int left,int down,int right,int bm,local *l);
void backTrack2(int y,int left,int down,int right,int bm,local *l2);
void backTrack1(int y,int left,int down,int right,int bm,local *l);
void *run(void *args);
void *run2(void *args);
void *NQueenThread();
void NQueen();
void symmetryOps_bm(local *l);

//backtrack2の2行目の列数を固定して場合分けすることによりスレッドを分割する
void backTrack3(int y,int left,int down,int right,int bm,local *l){
  l->bit=0;
  bm=l->msk&~(left|down|right); //配置可能フィールド
  if(y==siE){
    if(bm){
      if((bm&l->LM)==0){ //【枝刈り】最下段枝刈り
        l->aB[y]=bm;
        symmetryOps_bm(l);//対称解除法
      }
    }
  }else{
    if(y<l->B1){ //【枝刈り】上部サイド枝刈り            
      bm&=~l->SM; 
    }else if(y==l->B2) { //【枝刈り】下部サイド枝刈り    
      if((down&l->SM)==0){ 
        return; 
      }
      if((down&l->SM)!=l->SM){ 
        bm&=l->SM; 
      }
    }
    if(bm & (1<<l->k)){
      //スレッドの引数として指定した2行目のクイーンの位置kを固定で指定する
      l->aB[y]=l->bit=1<<l->k;
    }else{
      //left,down,rightなどkの値がクイーンの位置として指定できない場合はスレッド終了させる
      return;
    }
    //3行目以降は通常のbacktrack2の処理に渡す
    backTrack2(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
  }
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
void backTrack2(int y,int left,int down,int right,int bm,local *l){
  bm=l->msk&~(left|down|right); //配置可能フィールド
  l->bit=0;
  if(y==siE){
    if(bm>0 && (bm&l->LM)==0){ //【枝刈り】最下段枝刈り
      l->aB[y]=bm;
      symmetryOps_bm(l);
    }
  }else{
    if(y<l->B1){ //【枝刈り】上部サイド枝刈り            
      bm&=~l->SM; 
    }else if(y==l->B2) { //【枝刈り】下部サイド枝刈り    
      if((down&l->SM)==0){ 
        return; 
      }
      if((down&l->SM)!=l->SM){ 
        bm&=l->SM; 
      }
    }
    while(bm>0) {
      //最も下位の１ビットを抽出
      bm^=l->aB[y]=l->bit=-bm&bm;
      backTrack2(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
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
void backTrack1(int y,int left,int down,int right,int bm,local *l){
  l->bit=0;
  bm=l->msk&~(left|down|right);  //配置可能フィールド
  if(y==siE) {
    if(bm){//【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      l->aB[y]=bm;
      l->C8[l->B1][l->BK]++;
    }
  }else{
    if(y<l->B1){
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      bm&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
    while(bm){
      bm^=l->aB[y]=l->bit=-bm&bm;//最も下位の１ビットを抽出
      backTrack1(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }
  } 
}
void *run(void *args){
  local *l=(local *)args;
  l->bit=0 ; 
  l->aB[0]=1; 
  l->msk=(1<<si)-1; 
  l->TB=1<<siE; l->BK=0;
  if(l->B1>1 && l->B1<siE) { // 最上段のクイーンが角にある場合の探索
    l->aB[1]=l->bit=(1<<l->B1);// 角にクイーンを配置 
    backTrack1(2,(2|l->bit)<<1,(1|l->bit),(l->bit>>1),0,l);//２行目から探索
  }
  return 0;
}
//backtrack2のマルチスレッド処理
//今までは１行目にクイーンを置ける可能性のある数分スレッドが立っていたが
//１行目のクイーンが配置可能数*２行目のクイーンが配置可能数分スレッドがたつ
void *run2(void *args){
  local *l=(local *)args;
  l->msk=(1<<si)-1; l->TB=1<<siE;
  l->BK=1;
  l->EB=(l->TB>>l->B1);
  l->SM=l->LM=(l->TB|1);
  if(l->B1>0 && l->B2<siE && l->B1<l->B2){ // 最上段行のクイーンが角以外にある場合の探索 
    for(int i=1; i<l->B1; i++){
      l->LM=l->LM|l->LM>>1|l->LM<<1;
    }
    //１行目のクイーンの位置はB1の値によって決まる
    l->aB[0]=l->bit=(1<<l->B1);
    //２行目のクイーンの位置を固定することによってN分スレッドを分割する
    backTrack3(1,l->bit<<1,l->bit,l->bit>>1,0,l);
    l->EB>>=si;
  }
  return 0;
}
/**********************************************/
/* マルチスレッド */
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
void *NQueenThread(){
  pthread_t pt1[si];//スレッド childThread
  pthread_t pt2[si][si];//スレッド childThread
  local l[si];//構造体 local型 
  //backtrack2の処理に必要な構造体の数を算定する
  //1行目のクイーンのパタン*2行目のクイーンのパタン
  //1行目 最上段の行のクイーンの位置は中央を除く右側の領域に限定。
  //2行目 N個
/*
  int th=0;
  for(int B1=1,B2=siE-1;B1<siE;B1++,B2--){
    for(int k=1;k<=si;k++){
      th++;
    }
  }
  local l2[th];//構造体 local型 
*/
  local l2[si][si];
  for(int B1=1,B2=siE-1;B1<siE;B1++,B2--){// B1から順にスレッドを生成しながら処理を分担する 
    l[B1].B1=B1; l[B1].B2=B2; //B1 と B2を初期化
    for(int k=1;k<=si;k++){
      //l2[si*(B1-1)+k].B1=B1; l2[si*(B1-1)+k].B2=B2; //B1 と B2を初期化
      l2[B1][k].B1=B1; l2[B1][k].B2=B2; //B1 と B2を初期化
    }
    for(int j=0;j<siE;j++){ 
      l[B1].aB[j]=j; // aB[]の初期化
      for(int k=1;k<=si;k++){
        //l2[si*(B1-1)+k].aB[j]=j; // aB[]の初期化
        l2[B1][k].aB[j]=j; // aB[]の初期化
      }
    } 
    l[B1].C2[B1][0]=
      l[B1].C4[B1][0]=
      l[B1].C8[B1][0]=0;	//カウンターの初期化
    pthread_create(&pt1[B1],NULL,&run,(void*)&l[B1]);// チルドスレッドの生成
    for(int k=1;k<=si;k++){
    //l2[si*(B1-1)+k].C2[B1][1]=l2[si*(B1-1)+k].C4[B1][1]=l2[si*(B1-1)+k].C8[B1][1]=0;	//カウンターの初期化
    l2[B1][k].C2[B1][1]=
      //l2[si*(B1-1)+k].C4[B1][1]=l2[si*(B1-1)+k].C8[B1][1]=0;	//カウンターの初期化
      l2[B1][k].C4[B1][1]=
      l2[B1][k].C8[B1][1]=0;	//カウンターの初期化
    }
    for(int k=1;k<=si;k++){
      //l2[si*(B1-1)+k].k=k;
      l2[B1][k].k=k;
      //pthread_create(&pt[si+si*(B1-1)+k],NULL,&run2,(void*)&l2[si*(B1-1)+k]);// チルドスレッドの生成
      pthread_create(&pt2[B1][k],NULL,&run2,(void*)&l2[B1][k]);// チルドスレッドの生成
    }
  }
  for(int B1=1;B1<siE;B1++){ 
    pthread_join(pt1[B1],NULL); 
  //}
  //for(int B1=1;B1<siE;B1++){ 
    for(int k=1;k<=si;k++){
      //pthread_join(pt[si+si*(B1-1)+k],NULL); 
      pthread_join(pt2[B1][k],NULL); 
    }
  }
/**
  for(int B1=1;B1<siE;B1++){ 
     pthread_detach(pt[B1]);
  }
  for(int B1=1;B1<siE;B1++){ 
      for(int k=1;k<=si;k++){
        pthread_detach(pt[si+si*(B1-1)+k]);
      }
  }
*/
  for(int B1=1;B1<siE;B1++){//スレッド毎のカウンターを合計
    lTotal+=l[B1].C2[B1][0]*2+
      l[B1].C4[B1][0]*4+
      l[B1].C8[B1][0]*8;
    lUnique+=l[B1].C2[B1][0]+
      l[B1].C4[B1][0]+
      l[B1].C8[B1][0]; 
    for(int k=1;k<=si;k++){
      //lTotal+=l2[si*(B1-1)+k].C2[B1][1]*2+l2[si*(B1-1)+k].C4[B1][1]*4+l2[si*(B1-1)+k].C8[B1][1]*8;
      lTotal+=l2[B1][k].C2[B1][1]*2+
        l2[B1][k].C4[B1][1]*4+
        l2[B1][k].C8[B1][1]*8;
      //lUnique+=l2[si*(B1-1)+k].C2[B1][1]+l2[si*(B1-1)+k].C4[B1][1]+l2[si*(B1-1)+k].C8[B1][1]; 
      lUnique+=l2[B1][k].C2[B1][1]+
        l2[B1][k].C4[B1][1]+
        l2[B1][k].C8[B1][1]; 
    }
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
void NQueen(){
  pthread_t pth;//スレッド変数
  pthread_create(&pth, NULL, &NQueenThread, NULL);// メインスレッドの生成
  pthread_join(pth, NULL); //スレッドの終了を待つ
  pthread_detach(pth);
}
int main(void){
  struct timeval t0;
  struct timeval t1;
  int min=2;
  printf("%s\n"," N:        Total       Unique                 dd:hh:mm:ss.ms");
  for(int i=min;i<=MAX;i++){
    si=i; siE=i-1; 
    lTotal=lUnique=0;
    gettimeofday(&t0, NULL);
    NQueen();     // 実行関数
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
    printf("%2d:%16ld%17ld%12.2d:%02d:%02d:%02d.%02d\n", i,lTotal,lUnique,dd,hh,mm,ss,ms); 
  } 
  return 0;
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
void symmetryOps_bm(local *l){
  l->own=l->ptn=l->you=l->bit=0;
  l->C8[l->B1][l->BK]++;
  //90度回転
  if(l->aB[l->B2]==1){ 
    for(l->own=1,l->ptn=2;l->own<=siE;l->own++,l->ptn<<=1){ 
      for(l->bit=1,l->you=siE;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->you--){}
      if(l->aB[l->own]>l->bit){ l->C8[l->B1][l->BK]--; return; 
      }else if(l->aB[l->own]<l->bit){ break; } }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(l->own>siE){ l->C2[l->B1][l->BK]++; l->C8[l->B1][l->BK]--; return ; } 
  }
  //180度回転
  if(l->aB[siE]==l->EB){ 
    for(l->own=1,l->you=siE-1;l->own<=siE;l->own++,l->you--){ 
      for(l->bit=1,l->ptn=l->TB;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->ptn>>=1){}
      if(l->aB[l->own]>l->bit){ l->C8[l->B1][l->BK]--; return; } 
      else if(l->aB[l->own]<l->bit){ break; } }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(l->own>siE){ l->C4[l->B1][l->BK]++; l->C8[l->B1][l->BK]--; return; } 
  }
  //270度回転
  if(l->aB[l->B1]==l->TB){ 
    for(l->own=1,l->ptn=l->TB>>1;l->own<=siE;l->own++,l->ptn>>=1){ 
      for(l->bit=1,l->you=0;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->you++){}
      if(l->aB[l->own]>l->bit){ l->C8[l->B1][l->BK]--; return; } 
      else if(l->aB[l->own]<l->bit){ break; } } }
}
