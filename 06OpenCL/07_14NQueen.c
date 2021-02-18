/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

  14．C版マルチスレッド(mutex) 　排他処理                   

		コンパイルと実行
		$ make nq14 && ./07_14NQueen

  mutexによるロックとロック解除がボトルネックになり、
　並行処理のメリットが出ません。

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
13:           73712             9233          00:00:00:00.00
14:          365596            45752          00:00:00:00.01
15:         2279184           285053          00:00:00:00.10
16:        14772512          1846955          00:00:00:00.71
17:        95815104         11977939          00:00:00:05.71
*/

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#define MAX 27

// pthreadはパラメータを１つしか渡せないので構造体に格納
//グローバル構造体
typedef struct {
  int si;
  int siE;
  long C2;
  long C4;
  long C8;
}GCLASS, *GClass;
GCLASS G; 

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
};

pthread_mutex_t mutex;   //マルチスレッド排他処理mutexの宣言

void backTrack2(int y,int left,int down,int right,void *args);
void backTrack1(int y,int left,int down,int right,void *args);
void *run(void *args);
void *NQueenThread();
void NQueen();
void symmetryOps_bm(void *args);
long getUnique();
long getTotal();

void backTrack2(int y,int left,int down,int right,void *args){
  struct local *l=(struct local *)args;
  int bit=0;
  int bm=l->msk&~(left|down|right); /* 配置可能フィールド */
  if(y==G.siE){                     //【枝刈り】
    if(bm){
      if((bm&l->LM)==0){            //【枝刈り】最下段枝刈り
        l->aB[y]=bm;
        symmetryOps_bm(&*l);
      }
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
void backTrack1(int y,int left,int down,int right,void *args){
  struct local *l=(struct local *)args;
  int bit=0;                  /* 配置可能フィールド */
  int bm=l->msk&~(left|down|right); 
  if(y==G.siE) {
    if(bm){
      l->aB[y]=bm;
      //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      pthread_mutex_lock(&mutex);   //ロックして
      G.C8++;
      pthread_mutex_unlock(&mutex); //アンロックする
    }
  }else{
    if(y<l->B1) {   
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      bm&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
    while(bm) {
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
  l->TB=1<<G.siE;
  l->msk=(1<<G.si)-1;
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
  return 0;   //*run()の場合はreturn 0;が必要
}

void *NQueenThread(){
  pthread_t pt[G.si];                 //スレッド childThread
  pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;//mutexの初期化
  pthread_mutex_init(&mutex, NULL);   //pthread 排他処理
  struct local l[MAX];                //構造体 local型 
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
    l[B1].B1=B1; l[B1].B2=B2;         //B1 と B2を初期化
    for(int j=0;j<G.si;j++){ l[l->B1].aB[j]=j; } // aB[]の初期化
    // チルドスレッドの生成
    int iFbRet=pthread_create(&pt[B1],NULL,&run,&l[B1]);
    if(iFbRet>0){
      printf("[mainThread] pthread_create #%d: %d\n", l[B1].B1, iFbRet);
    }
    pthread_join(pt[B1],NULL);        //処理を待って次の処理へ
  }
  for(int i=G.siE;i>0;i--){
    pthread_join(pt[i],NULL);//処理が終わったら 全ての処理をjoinする
  }
  pthread_mutexattr_destroy(&mutexattr);//不要になった変数の破棄
  pthread_mutex_destroy(&mutex); //nutexの破棄       
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
  int min=2;
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  for(int i=min;i<=MAX;i++){
    G.si=i; G.siE=i-1; G.C2=G.C4=G.C8=0; //初期化
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
      pthread_mutex_lock(&mutex); //ロックする
      G.C2++; 
      pthread_mutex_unlock(&mutex); //アンロックする
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
      pthread_mutex_lock(&mutex); //ロックする
      G.C4++; 
      pthread_mutex_unlock(&mutex); //アンロックする
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
  pthread_mutex_lock(&mutex); //ロックする
  G.C8++;
  pthread_mutex_unlock(&mutex); //アンロックする
}
long getUnique(){ 
  return G.C2+G.C4+G.C8;
}
long getTotal(){ 
  return G.C2*2+G.C4*4+G.C8*8;
}
