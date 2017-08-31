/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 <>33. マルチスレッド 脱malloc 

 N18までは07_27で実装した２段階 k
 N19は07_28で実装した３段階     j
 N20は07_29で実装した４段階     kj4
 N21は07_31で実装した５段階で処理するようにした。 kj5
 さらに N22,N23,N24,N25とNが大きくなるにつれて、スレッドの階層も増える。
 これにより、Nが大きくなればなるほど不均衡で偏りが出て遅くなる傾向にあった処理速度の遅延は改善された。 の、malloc卒業版
常にスレッド数はＮに制限され、最大で9段階にシフトしながら解析処理を進めていく。


 N:          Total        Unique                 dd:hh:mm:ss.ms
 2:                 0                 0          00:00:00:00.00
 3:                 0                 0          00:00:00:00.00
 4:                 2                 1          00:00:00:00.00
 5:                10                 2          00:00:00:00.00
 6:                 4                 1          00:00:00:00.00
 7:                40                 6          00:00:00:00.00
 8:                92                12          00:00:00:00.00
 9:               352                46          00:00:00:00.00
10:               724                92          00:00:00:00.00
11:              2680               341          00:00:00:00.00
12:             14200              1787          00:00:00:00.00
13:             73712              9233          00:00:00:00.00
14:            365596             45752          00:00:00:00.01
15:           2279184            285053          00:00:00:00.07
16:          14772512           1846955          00:00:00:00.52
17:          95815104          11977939          00:00:00:03.86
18:         666090624          83263591          00:00:00:30.94
19:        4968057848         621012754          00:00:05:08.98
20:       39029188884        4878666808          00:00:40:31.91
21:      314666222712       39333324973          00:05:38:49.56
22:     2691008701644      336376244042          02:02:03:49.27
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include "unistd.h"

// OpenCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX 27      //求めるNの最大値
#define DEBUG 0     //TRUE:1 FALSE:0
#define THREAD 1    //TRUE:1 FALSE:0


/** グローバル変数 */
int si;             //size
int siE;            //size-1
long lTotal;        //解の総合計
long lUnique;       //解のユニーク解数

/** スレッドローカル構造体 */
typedef struct{
	int bit;
  int own;
	int ptn;
	int you;
  int k;            //上から2行目のスレッドに使う
  int j;            //上から3行目のスレッドに使う
  int kj4;          //上から4行目のスレッドに使う
  int kj5;          //上から5行目のスレッドに使う
  int kj6;          //上から6行目のスレッドに使う
  int kj7;          //上から7行目のスレッドに使う
  int kj8;          //上から8行目のスレッドに使う
  int B1;           //BOUND1
  int B2;           //BOUND2
  int TB;           //TOPBIT
  int EB;           //ENDBIT
  int msk;          //mask
  int SM;           //SIDEMASK
  int LM;           //LASTMASK
  int aB[MAX];      //Board配列
  long C2[MAX][2];  //COUNT2　カウンター
  long C4[MAX][2];  //COUNT4　カウンター
  long C8[MAX][2];  //COUNT8　カウンター
  int BK;
}local ;

/**
 * 関数定義
 */
void thMonitor(local *l,int i);
void symmetryOps_bm(local *l);
void backTrack1(int y,int left,int down,int right,int bm,local *l);
void backTrack2(int y,int left,int down,int right,int bm,local *l);
void *run(void *args);
void *run3(void *args);
void *NQueenThread();
void NQueen();
void NQueenThread_18();
void NQueenThread_19();
void NQueenThread_20();
void NQueenThread_21();
void NQueenThread_22();
void NQueenThread_23();
void NQueenThread_24();
void NQueenThread_25();

#ifdef DEBUG
const int spc[]={'/', '-', '\\', '|'};
const int spl=sizeof(spc)/sizeof(spc[0]);
void thMonitor(local *l,int i);
void hoge();
void hoge(){
  clock_t t;
  t = clock() + CLOCKS_PER_SEC/10;
  while(t>clock());
}
//FILE *f;
#endif

/**
 * チェスボードのクイーンの場所を確認
 */
int db=0;
void thMonitor(local *l,int i){
  if(THREAD>0){ }else{
    db++;
    printf("N%d =%d C%d\n",si,db,i);
    for (int y=0;y<si;y++) {
      for (l->bit=l->TB; l->bit; l->bit>>=1){
        if(l->aB[y]==l->bit){
          printf("Q ");
        }else{
          printf(". ");
        }
      }
      printf("\n");
    }
    printf("\n");
  } 
}
/**
 * 回転・反転の解析処理
 */
void symmetryOps_bm(local *l){
  l->own=l->ptn=l->you=l->bit=0;
  if(l->aB[l->B2]==1){ //90度回転
    for(l->own=1,l->ptn=2;l->own<=siE;l->own++,l->ptn<<=1){ 
      for(l->bit=1,l->you=siE;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->you--){}
      if(l->aB[l->own]>l->bit){ return; }else if(l->aB[l->own]<l->bit){ break; } }
    //90度回転して同型なら180度/270度回転も同型である
    if(l->own>siE){ l->C2[l->B1][l->BK]++; if(DEBUG>0) thMonitor(l,2); return ; } }
  if(l->aB[siE]==l->EB){ //180度回転
    for(l->own=1,l->you=siE-1;l->own<=siE;l->own++,l->you--){ 
      for(l->bit=1,l->ptn=l->TB;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->ptn>>=1){}
      if(l->aB[l->own]>l->bit){ return; } 
      else if(l->aB[l->own]<l->bit){ break; } }
    //90度回転が同型でなくても180度回転が同型である事もある
    if(l->own>siE){ l->C4[l->B1][l->BK]++; if(DEBUG>0) thMonitor(l,4); return; } }
  if(l->aB[l->B1]==l->TB){ //270度回転
    for(l->own=1,l->ptn=l->TB>>1;l->own<=siE;l->own++,l->ptn>>=1){ 
      for(l->bit=1,l->you=0;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->you++){}
      if(l->aB[l->own]>l->bit){ return; } else if(l->aB[l->own]<l->bit){ break; } } }
  l->C8[l->B1][l->BK]++;
  if(DEBUG>0) thMonitor(l,8); 
}
void backTrack2(int y,int left,int down,int right,int bm,local *l){
  //配置可能フィールド
  bm=l->msk&~(left|down|right); 
  l->bit=0;
  if(y==siE){
    //【枝刈り】最下段枝刈り
    if(bm>0 && (bm&l->LM)==0){ l->aB[y]=bm; symmetryOps_bm(l); }
  }else{
    //【枝刈り】上部サイド枝刈り
    if(y<l->B1){ bm&=~l->SM; }
    //【枝刈り】下部サイド枝刈り 
    else if(y==l->B2) { if((down&l->SM)==0){ return; } if((down&l->SM)!=l->SM){ bm&=l->SM; } }
    if(y==1 && l->k>=0){
      if(bm & (1<<l->k)){ l->aB[y]=l->bit=1<<l->k; } else{ return; }
      backTrack2(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }else if(y==2 && l->j>=0){
      if(bm & (1<<l->j)){ l->aB[y]=l->bit=1<<l->j; } else{ return; }
      backTrack2(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }else if(y==3 && l->kj4>=0){
      if(bm & (1<<l->kj4)){ l->aB[y]=l->bit=1<<l->kj4; } else{ return; }
      backTrack2(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }else if(y==4 && l->kj5>=0){
      if(bm & (1<<l->kj5)){ l->aB[y]=l->bit=1<<l->kj5; } else{ return; }
      backTrack2(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }else if(y==5 && l->kj6>=0){
      if(bm & (1<<l->kj6)){ l->aB[y]=l->bit=1<<l->kj6; } else{ return; }
      backTrack2(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }else if(y==6 && l->kj7>=0){
      if(bm & (1<<l->kj7)){ l->aB[y]=l->bit=1<<l->kj7; } else{ return; }
      backTrack2(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }else if(y==7 && l->kj8>=0){
      if(bm & (1<<l->kj8)){ l->aB[y]=l->bit=1<<l->kj8; } else{ return; }
      backTrack2(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }else{
      while(bm>0) {
        bm^=l->aB[y]=l->bit=-bm&bm;
        backTrack2(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
      }
    }
  }
}
void backTrack1(int y,int left,int down,int right,int bm,local *l){
  bm=l->msk&~(left|down|right); l->bit=0;
  if(y==siE) {
    //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
    if(bm>0){ l->aB[y]=bm; l->C8[l->B1][l->BK]++; if(DEBUG>0) thMonitor(l,82); } }
    //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
    else{ if(y<l->B1) { bm&=~2; }
    if(y==2 && l->k>=0){
      if(bm & (1<<l->k)){ l->aB[y]=l->bit=1<<l->k; } else{ return; }
      backTrack1(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }else if(y==3 && l->j>=0){
      if(bm & (1<<l->j)){ l->aB[y]=l->bit=1<<l->j; } else{ return; }
      backTrack1(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }else if(y==4 && l->kj4>=0){
      if(bm & (1<<l->kj4)){ l->aB[y]=l->bit=1<<l->kj4; } else{ return; }
      backTrack1(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }else if(y==5 && l->kj5>=0){
      if(bm & (1<<l->kj5)){ l->aB[y]=l->bit=1<<l->kj5; } else{ return; }
      backTrack1(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }else if(y==6 && l->kj6>=0){
      if(bm & (1<<l->kj6)){ l->aB[y]=l->bit=1<<l->kj6; } else{ return; }
      backTrack1(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }else if(y==7 && l->kj7>=0){
      if(bm & (1<<l->kj7)){ l->aB[y]=l->bit=1<<l->kj7; } else{ return; }
      backTrack1(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }else{
      while(bm>0) {
        bm^=l->aB[y]=l->bit=-bm&bm;
        backTrack1(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
      }
    }
  } 
}
/**
 * backTrack2のマルチスレッド処理
 * ３行目のクイーンの位置まで固定して別スレッドで走らせる
 * NXNXNスレッドが立っている
*/
void *run3(void *args){
  //l->makはクイーンを置ける場所 si分1が並ぶ
  local *l=(local *)args; l->msk=(1<<si)-1; 
  //si=8 なら 1が8個並ぶ
  l->TB=1<<siE; l->BK=1; l->EB=(l->TB>>l->B1); l->SM=l->LM=(l->TB|1);
  // 最上段行のクイーンが角以外にある場合の探索 
  if(l->B1>0 && l->B2<siE && l->B1<l->B2){ 
    for(int i=1; i<l->B1; i++){ l->LM=l->LM|l->LM>>1|l->LM<<1; }
    //１行目のクイーンの位置はB1の値によって決まる
    l->aB[0]=l->bit=(1<<l->B1);
    //2行目のクイーンの位置を固定することによってN分スレッドを分割する
    backTrack2(1,l->bit<<1,l->bit,l->bit>>1,0,l);
    l->EB>>=si;
  }
  return 0;
}
/**
 * backTrack1のマルチスレッド処理
*/
void *run(void *args){
  local *l=(local *)args; l->bit=0 ; 
  //backtrack1は1行目のクイーンの位置を右端に固定
  l->aB[0]=1; l->msk=(1<<si)-1; l->TB=1<<siE; l->BK=0;
  if(l->B1>1 && l->B1<siE) { // 最上段のクイーンが角にある場合の探索
    //backtrack1は2行目のクイーンの位置はl->B1
    l->aB[1]=l->bit=(1<<l->B1);// 角にクイーンを配置 
    //3行目のクイーンの位置を固定することによってN分スレッドを分割する
    backTrack1(2,(2|l->bit)<<1,(1|l->bit),(l->bit>>1),0,l);
  }
  return 0;
}
/**
 * Nの数だけスレッドをもたせて同時並列処理をする
*/
void *NQueenThread(){
  if(si<=18){ NQueenThread_18();  } //07_27NQueen.c
  else if(si==19){ NQueenThread_19(); }
  else if(si==20){ NQueenThread_20(); }
  else if(si==21){ NQueenThread_21(); }
  else if(si==22){ NQueenThread_22(); }
  else if(si==23){ NQueenThread_23(); }
  else if(si==24){ NQueenThread_24(); }
  else if(si>=25){ NQueenThread_25(); }
  return 0;
}
/**
 * メインスレッドの生成
 */
void NQueen(){
  pthread_t pth;//スレッド変数
  pthread_create(&pth, NULL, &NQueenThread, NULL);// メインスレッドの生成
  pthread_join(pth, NULL); //スレッドの終了を待つ
  pthread_detach(pth);
}
/**
 * メイン
 */
int main(void){
  int min=2;
  struct timeval t0;
  struct timeval t1;
  printf("%s\n"," N:          Total        Unique                 dd:hh:mm:ss.ms");
  for(int i=min;i<=MAX;i++){
		db=0;
    si=i; siE=i-1; 
    lTotal=lUnique=0;
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
    printf("%2d:%18ld%18ld%12.2d:%02d:%02d:%02d.%02d\n", i,lTotal,lUnique,dd,hh,mm,ss,ms); 
  } 
  return 0;
}
/**
 *
 */
void NQueenThread_18(){
  //printf("si==18");
  pthread_t pt1[si];//スレッド childThread
  pthread_t pt2[si][si];//スレッド childThread
  local l[si];//構造体 local型 
  local l2[si][si];
  for(int B1=1,B2=siE-1;B1<siE;B1++,B2--){// B1から順にスレッドを生成しながら処理を分担する 
    l[B1].B1=B1; l[B1].B2=B2; //B1 と B2を初期化
    for(int k=0;k<si;k++){
      l2[B1][k].B1=B1; l2[B1][k].B2=B2; //B1 と B2を初期化
    }
    for(int j=0;j<siE;j++){ 
      l[B1].aB[j]=j; // aB[]の初期化
      for(int k=0;k<si;k++){
        l2[B1][k].aB[j]=j; // aB[]の初期化
      }
    } 
    l[B1].k=-1;	l[B1].j=-1;	l[B1].kj4=-1;	l[B1].kj5=-1;	l[B1].kj6=-1; l[B1].kj7=-1; l[B1].kj8=-1;
    l[B1].C2[B1][0]=0; l[B1].C4[B1][0]=0; l[B1].C8[B1][0]=0;	//カウンターの初期化 
    for(int k=0;k<si;k++){
      l2[B1][k].k=-1;	l2[B1][k].j=-1;	l2[B1][k].kj4=-1;	l2[B1][k].kj5=-1;	l2[B1][k].kj6=-1; l2[B1][k].kj7=-1; l2[B1][k].kj8=-1;
      l2[B1][k].C2[B1][1]=0; l2[B1][k].C4[B1][1]=0; l2[B1][k].C8[B1][1]=0;	//カウンターの初期化 
    }
    if(B1<si/2){
      for(int k=0;k<si;k++){
        l2[B1][k].k=k;
        pthread_create(&pt2[B1][k],NULL,&run3,(void*)&l2[B1][k]);// チルドスレッドの生成
      }
    }
    pthread_create(&pt1[B1],NULL,&run,(void*)&l[B1]);// チルドスレッドの生成

  }
  for(int B1=1;B1<siE;B1++){//スレッド毎のカウンターを合計
    pthread_join(pt1[B1],NULL); 
    for(int k=0;k<si;k++){
      pthread_join(pt2[B1][k],NULL); 
    }
  }
  for(int B1=1;B1<siE;B1++){//スレッド毎のカウンターを合計
    lTotal+=l[B1].C2[B1][0]*2+ l[B1].C4[B1][0]*4+ l[B1].C8[B1][0]*8;
    lUnique+=l[B1].C2[B1][0]+ l[B1].C4[B1][0]+ l[B1].C8[B1][0]; 
    for(int k=0;k<si;k++){
      lTotal+=l2[B1][k].C2[B1][1]*2+ l2[B1][k].C4[B1][1]*4+ l2[B1][k].C8[B1][1]*8;
      lUnique+=l2[B1][k].C2[B1][1]+ l2[B1][k].C4[B1][1]+ l2[B1][k].C8[B1][1]; 
    }
  }
}
void NQueenThread_19(){
  //printf("si==19");
  for(int B1=1,B2=siE-1;B1<siE;B1++,B2--){// B1から順にスレッドを生成しながら処理を分担する 
    pthread_t pt1;//スレッド childThread
    pthread_t pt3[si];//スレッド childThread
    local l;//構造体 local型 
    local l3[si];
    l.B1=B1; 
		l.B2=B2;     
    for(int k=0;k<si;k++){//backtrack1のB1
      l3[k].B1=B1; 
			l3[k].B2=B2;
    }
    for(int i=0;i<siE;i++){ //aB[]の初期化
      l.aB[i]=i;
      for(int k=0;k<si;k++){
        l3[k].aB[i]=i;  // 上から３行目のスレッドに使う構造体aB[]の初期化
      }
    }
    l.k=-1;	
    l.j=-1;	
    l.kj4=-1;	
    l.kj5=-1;	
    l.kj6=-1;	
    l.kj7=-1;	
    l.kj8=-1;	
    l.C2[B1][0]=0;
		l.C4[B1][0]=0;
		l.C8[B1][0]=0;	
    for(int k=0;k<si;k++){
      l3[k].k=-1;
      l3[k].j=-1;
      l3[k].kj4=-1;
      l3[k].kj5=-1;
      l3[k].kj6=-1;
      l3[k].kj7=-1;
      l3[k].kj8=-1;
      l3[k].C2[B1][1]=0;
			l3[k].C4[B1][1]=0;
			l3[k].C8[B1][1]=0;	
    }
    if(B1<si/2){
      for(int k=0;k<si;k++){
        l3[k].k=k;
        pthread_create(&pt3[k],NULL,&run3,(void*)&l3[k]);// チルドスレッドの生成
        if(THREAD<1){ // not Thread
          pthread_join(pt3[k],NULL); 
          pthread_detach(pt3[k]);
        }
      }
      for(int k=0;k<si;k++){
        pthread_join(pt3[k],NULL); 
        pthread_detach(pt3[k]);
      }
    }
    pthread_create(&pt1,NULL,&run,(void*)&l);// チルドスレッドの生成
    if(THREAD<1){ // not Thread
      pthread_join(pt1,NULL); 
      pthread_detach(pt1);
    }
    pthread_join(pt1,NULL); 
    pthread_detach(pt1);
    lTotal+= l.C2[B1][0]*2+ l.C4[B1][0]*4+ l.C8[B1][0]*8;
    lUnique+= l.C2[B1][0]+ l.C4[B1][0]+ l.C8[B1][0]; 
    for(int k=0;k<si;k++){//backtrack1の集計
      lTotal+= l3[k].C2[B1][1]*2+ l3[k].C4[B1][1]*4+ l3[k].C8[B1][1]*8;
      lUnique+= l3[k].C2[B1][1]+ l3[k].C4[B1][1]+ l3[k].C8[B1][1]; 
    }
  }
}
void NQueenThread_20(){
  //printf("si==20");
  for(int B1=1,B2=siE-1;B1<siE;B1++,B2--){// B1から順にスレッドを生成しながら処理を分担する 
    for(int k=0;k<si;k++){
			pthread_t pt1;//スレッド childThread
			pthread_t pt3[si];//スレッド childThread
			local l;//構造体 local型 
			local l3[si];
			l.B1=B1; 
			l.B2=B2;     
      for(int j=0;j<si;j++){//backtrack1のB1
        l3[j].B1=B1; 
				l3[j].B2=B2;
      }
      for(int i=0;i<siE;i++){ //aB[]の初期化
        l.aB[i]=i;
        for(int j=0;j<si;j++){
          l3[j].aB[i]=i;  // 上から３行目のスレッドに使う構造体aB[]の初期化
        }
      }
      l.k=-1;	
      l.j=-1;	
      l.kj4=-1;	
      l.kj5=-1;	
      l.kj6=-1;	
      l.kj7=-1;	
      l.kj8=-1;	
      l.C2[B1][0]=0;
			l.C4[B1][0]=0;
			l.C8[B1][0]=0;	
      for(int j=0;j<si;j++){
        l3[j].k=-1;
        l3[j].j=-1;
        l3[j].kj4=-1;
        l3[j].kj5=-1;
        l3[j].kj6=-1;
        l3[j].kj7=-1;
        l3[j].kj8=-1;
        l3[j].C2[B1][1]=0;
				l3[j].C4[B1][1]=0;
				l3[j].C8[B1][1]=0;	
      }
      if(B1<si/2){
        for(int j=0;j<si;j++){
          l3[j].k=k; 
					l3[j].j=j;
          pthread_create(&pt3[j],NULL,&run3,(void*)&l3[j]);// チルドスレッドの生成
          if(THREAD<1){ // not Thread
            pthread_join(pt3[j],NULL); 
            pthread_detach(pt3[j]);
          }
        }
        for(int j=0;j<si;j++){
          pthread_join(pt3[j],NULL); 
          pthread_detach(pt3[j]);
        }
      }
      l.k=k; 
      pthread_create(&pt1,NULL,&run,(void*)&l);// チルドスレッドの生成
      if(THREAD<1){ // not Thread
        pthread_join(pt1,NULL); 
        pthread_detach(pt1);
      }
      pthread_join(pt1,NULL); 
      pthread_detach(pt1);
      lTotal+= l.C2[B1][0]*2+ l.C4[B1][0]*4+ l.C8[B1][0]*8;
      lUnique+= l.C2[B1][0]+ l.C4[B1][0]+ l.C8[B1][0]; 
      for(int j=0;j<si;j++){//backtrack1の集計
        lTotal+= l3[j].C2[B1][1]*2+ l3[j].C4[B1][1]*4+ l3[j].C8[B1][1]*8;
        lUnique+= l3[j].C2[B1][1]+ l3[j].C4[B1][1]+ l3[j].C8[B1][1]; 
      }
    }
  }
}
void NQueenThread_21(){
  //printf("si==21");
  for(int B1=1,B2=siE-1;B1<siE;B1++,B2--){// B1から順にスレッドを生成しながら処理を分担する 
    for(int k=0;k<si;k++){
      for(int j=0;j<si;j++){//backtrack1のB1
        //この中で回す
        pthread_t pt1;//スレッド childThread
        pthread_t pt3[si];//スレッド childThread
        local l;//構造体 local型 
        local l3[si];
        l.B1=B1; l.B2=B2;     
        for(int kj4=0;kj4<si;kj4++){//backtrack1のB1
          l3[kj4].B1=B1; l3[kj4].B2=B2;
        }
        for(int i=0;i<siE;i++){ //aB[]の初期化
          l.aB[i]=i;
          for(int kj4=0;kj4<si;kj4++){
            l3[kj4].aB[i]=i;  // 上から３行目のスレッドに使う構造体aB[]の初期化
          }
        }
        l.k=-1;	
        l.j=-1;	
        l.kj4=-1;	
        l.kj5=-1;	
        l.kj6=-1;	
        l.kj7=-1;	
        l.kj8=-1;	
        l.C2[B1][0]= l.C4[B1][0]= l.C8[B1][0]=0;	
        for(int kj4=0;kj4<si;kj4++){
          l3[kj4].k=-1;
          l3[kj4].j=-1;
          l3[kj4].kj4=-1;
          l3[kj4].kj5=-1;
          l3[kj4].kj6=-1;
          l3[kj4].kj7=-1;
          l3[kj4].kj8=-1;
          l3[kj4].C2[B1][1]= l3[kj4].C4[B1][1]= l3[kj4].C8[B1][1]=0;	
        }
        if(B1<si/2){
          for(int kj4=0;kj4<si;kj4++){
            l3[kj4].k=k; 
						l3[kj4].j=j;
            l3[kj4].kj4=kj4;
            pthread_create(&pt3[kj4],NULL,&run3,(void*)&l3[kj4]);// チルドスレッドの生成
            if(THREAD<1){ // not Thread
              pthread_join(pt3[kj4],NULL); 
              pthread_detach(pt3[kj4]);
            }
          }
          for(int kj4=0;kj4<si;kj4++){
            pthread_join(pt3[kj4],NULL); 
            pthread_detach(pt3[kj4]);
          }
        }
        l.k=k; 
        l.j=j;
        pthread_create(&pt1,NULL,&run,(void*)&l);// チルドスレッドの生成
        if(THREAD<1){ // not Thread
          pthread_join(pt1,NULL); 
          pthread_detach(pt1);
        }
        pthread_join(pt1,NULL); 
        pthread_detach(pt1);
        lTotal+= l.C2[B1][0]*2+ l.C4[B1][0]*4+ l.C8[B1][0]*8;
        lUnique+= l.C2[B1][0]+ l.C4[B1][0]+ l.C8[B1][0]; 
        for(int kj4=0;kj4<si;kj4++){//backtrack1の集計
          lTotal+= l3[kj4].C2[B1][1]*2+ l3[kj4].C4[B1][1]*4+ l3[kj4].C8[B1][1]*8;
          lUnique+= l3[kj4].C2[B1][1]+ l3[kj4].C4[B1][1]+ l3[kj4].C8[B1][1]; 
        }
      }
    }
  }
}
void NQueenThread_22(){
  //printf("si==22");
  for(int B1=1,B2=siE-1;B1<siE;B1++,B2--){// B1から順にスレッドを生成しながら処理を分担する 
    for(int k=0;k<si;k++){
      for(int j=0;j<si;j++){//backtrack1のB1
        for(int kj4=0;kj4<si;kj4++){//backtrack1のB1
          //この中で回す
          pthread_t pt1;//スレッド childThread
          pthread_t pt3[si];//スレッド childThread
          local l;//構造体 local型 
          local l3[si];
          l.B1=B1; l.B2=B2;     
          for(int kj5=0;kj5<si;kj5++){//backtrack1のB1
            l3[kj5].B1=B1; l3[kj5].B2=B2;
          }
          for(int i=0;i<siE;i++){ //aB[]の初期化
            l.aB[i]=i;
            for(int kj5=0;kj5<si;kj5++){
              l3[kj5].aB[i]=i;  // 上から３行目のスレッドに使う構造体aB[]の初期化
            }
          }
          l.k=-1;	
          l.j=-1;	
          l.kj4=-1;	
          l.kj5=-1;	
          l.kj6=-1;	
          l.kj7=-1;	
          l.kj8=-1;	
          l.C2[B1][0]= l.C4[B1][0]= l.C8[B1][0]=0;	
          for(int kj5=0;kj5<si;kj5++){
            l3[kj5].k=-1;
            l3[kj5].j=-1;
            l3[kj5].kj4=-1;
            l3[kj5].kj5=-1;
            l3[kj5].kj6=-1;
            l3[kj5].kj7=-1;
            l3[kj5].kj8=-1;
            l3[kj5].C2[B1][1]= l3[kj5].C4[B1][1]= l3[kj5].C8[B1][1]=0;	
          }
          if(B1<si/2){

            for(int kj5=0;kj5<si;kj5++){
              l3[kj5].k=k; l3[kj5].j=j;
              l3[kj5].kj4=kj4;
              l3[kj5].kj5=kj5;
              pthread_create(&pt3[kj5],NULL,&run3,(void*)&l3[kj5]);// チルドスレッドの生成
              if(THREAD<1){ // not Thread
                pthread_join(pt3[kj5],NULL); 
                pthread_detach(pt3[kj5]);
              }
            }
            for(int kj5=0;kj5<si;kj5++){
              pthread_join(pt3[kj5],NULL); 
              pthread_detach(pt3[kj5]);
            }
          }
          l.k=k; 
          l.j=j;
          l.kj4=kj4;
          pthread_create(&pt1,NULL,&run,(void*)&l);// チルドスレッドの生成
          if(THREAD<1){ // not Thread
            pthread_join(pt1,NULL); 
            pthread_detach(pt1);
          }
          pthread_join(pt1,NULL); 
          pthread_detach(pt1);
          lTotal+= l.C2[B1][0]*2+ l.C4[B1][0]*4+ l.C8[B1][0]*8;
          lUnique+= l.C2[B1][0]+ l.C4[B1][0]+ l.C8[B1][0]; 
          for(int kj5=0;kj5<si;kj5++){//backtrack1の集計
            lTotal+= l3[kj5].C2[B1][1]*2+ l3[kj5].C4[B1][1]*4+ l3[kj5].C8[B1][1]*8;
            lUnique+= l3[kj5].C2[B1][1]+ l3[kj5].C4[B1][1]+ l3[kj5].C8[B1][1]; 
          }
        }
      }
    }
  }
}
void NQueenThread_23(){
  //printf("si==23");
  for(int B1=1,B2=siE-1;B1<siE;B1++,B2--){// B1から順にスレッドを生成しながら処理を分担する 
    for(int k=0;k<si;k++){
      for(int j=0;j<si;j++){//backtrack1のB1
        for(int kj4=0;kj4<si;kj4++){//backtrack1のB1
          for(int kj5=0;kj5<si;kj5++){//backtrack1のB1
            //この中で回す
            pthread_t pt1;//スレッド childThread
            pthread_t pt3[si];//スレッド childThread
            local l;//構造体 local型 
            local l3[si];
            l.B1=B1; l.B2=B2;     
            for(int kj6=0;kj6<si;kj6++){//backtrack1のB1
              l3[kj6].B1=B1; l3[kj6].B2=B2;
            }
            for(int i=0;i<siE;i++){ //aB[]の初期化
              l.aB[i]=i;
              for(int kj6=0;kj6<si;kj6++){
                l3[kj6].aB[i]=i;  // 上から３行目のスレッドに使う構造体aB[]の初期化
              }
            }
            l.k=-1;	
            l.j=-1;	
            l.kj4=-1;	
            l.kj5=-1;	
            l.kj6=-1;	
            l.kj7=-1;	
            l.kj8=-1;	
            l.C2[B1][0]= l.C4[B1][0]= l.C8[B1][0]=0;	
            for(int kj6=0;kj6<si;kj6++){
              l3[kj6].k=-1;
              l3[kj6].j=-1;
              l3[kj6].kj4=-1;
              l3[kj6].kj5=-1;
              l3[kj6].kj6=-1;
              l3[kj6].kj7=-1;
              l3[kj6].kj8=-1;
              l3[kj6].C2[B1][1]= l3[kj6].C4[B1][1]= l3[kj6].C8[B1][1]=0;	
            }
            if(B1<si/2){
              for(int kj6=0;kj6<si;kj6++){
                l3[kj6].k=k; l3[kj6].j=j;
                l3[kj6].kj4=kj4;
                l3[kj6].kj5=kj5;
                l3[kj6].kj6=kj6;
                pthread_create(&pt3[kj6],NULL,&run3,(void*)&l3[kj6]);// チルドスレッドの生成
                if(THREAD<1){ // not Thread
                  pthread_join(pt3[kj6],NULL); 
                  pthread_detach(pt3[kj6]);
                }
              }
              for(int kj6=0;kj6<si;kj6++){
                pthread_join(pt3[kj6],NULL); 
                pthread_detach(pt3[kj6]);
              }
            }
            l.k=k; 
            l.j=j;
            l.kj4=kj4;
            l.kj5=kj5;
            pthread_create(&pt1,NULL,&run,(void*)&l);// チルドスレッドの生成
            if(THREAD<1){ // not Thread
              pthread_join(pt1,NULL); 
              pthread_detach(pt1);
            }
            pthread_join(pt1,NULL); 
            pthread_detach(pt1);
            lTotal+= l.C2[B1][0]*2+ l.C4[B1][0]*4+ l.C8[B1][0]*8;
            lUnique+= l.C2[B1][0]+ l.C4[B1][0]+ l.C8[B1][0]; 
            for(int kj6=0;kj6<si;kj6++){//backtrack1の集計
              lTotal+= l3[kj6].C2[B1][1]*2+ l3[kj6].C4[B1][1]*4+ l3[kj6].C8[B1][1]*8;
              lUnique+= l3[kj6].C2[B1][1]+ l3[kj6].C4[B1][1]+ l3[kj6].C8[B1][1]; 
            }
          }
        }
      }
    }
  }
}
void NQueenThread_24(){
  //printf("si==24");
  for(int B1=1,B2=siE-1;B1<siE;B1++,B2--){// B1から順にスレッドを生成しながら処理を分担する 
    for(int k=0;k<si;k++){
      for(int j=0;j<si;j++){//backtrack1のB1
        for(int kj4=0;kj4<si;kj4++){//backtrack1のB1
          for(int kj5=0;kj5<si;kj5++){//backtrack1のB1
            for(int kj6=0;kj6<si;kj6++){//backtrack1のB1
              //この中で回す
              pthread_t pt1;//スレッド childThread
              pthread_t pt3[si];//スレッド childThread
              local l;//構造体 local型 
              local l3[si];
              l.B1=B1; l.B2=B2;     
              for(int kj7=0;kj7<si;kj7++){//backtrack1のB1
                l3[kj7].B1=B1; l3[kj7].B2=B2;
              }
              for(int i=0;i<siE;i++){ //aB[]の初期化
                l.aB[i]=i;
                for(int kj7=0;kj7<si;kj7++){
                  l3[kj7].aB[i]=i;  // 上から３行目のスレッドに使う構造体aB[]の初期化
                }
              }
              l.k=-1;	
              l.j=-1;	
              l.kj4=-1;	
              l.kj5=-1;	
              l.kj6=-1;	
              l.kj7=-1;	
              l.kj8=-1;	
              l.C2[B1][0]= l.C4[B1][0]= l.C8[B1][0]=0;	
              for(int kj7=0;kj7<si;kj7++){
                l3[kj7].k=-1;
                l3[kj7].j=-1;
                l3[kj7].kj4=-1;
                l3[kj7].kj5=-1;
                l3[kj7].kj6=-1;
                l3[kj7].kj7=-1;
                l3[kj7].kj8=-1;
                l3[kj7].C2[B1][1]= l3[kj7].C4[B1][1]= l3[kj7].C8[B1][1]=0;	
              }
              if(B1<si/2){

                for(int kj7=0;kj7<si;kj7++){
                  l3[kj7].k=k; l3[kj7].j=j;
                  l3[kj7].kj4=kj4;
                  l3[kj7].kj5=kj5;
                  l3[kj7].kj6=kj6;
                  l3[kj7].kj7=kj7;
                  pthread_create(&pt3[kj7],NULL,&run3,(void*)&l3[kj7]);// チルドスレッドの生成
                  if(THREAD<1){ // not Thread
                    pthread_join(pt3[kj7],NULL); 
                    pthread_detach(pt3[kj7]);
                  }
                }
                for(int kj7=0;kj7<si;kj7++){
                  pthread_join(pt3[kj7],NULL); 
                  pthread_detach(pt3[kj7]);
                }
              }
              l.k=k; 
              l.j=j;
              l.kj4=kj4;
              l.kj5=kj5;
              l.kj6=kj6;
              pthread_create(&pt1,NULL,&run,(void*)&l);// チルドスレッドの生成
              if(THREAD<1){ // not Thread
                pthread_join(pt1,NULL); 
                pthread_detach(pt1);
              }
              pthread_join(pt1,NULL); 
              pthread_detach(pt1);
              lTotal+= l.C2[B1][0]*2+ l.C4[B1][0]*4+ l.C8[B1][0]*8;
              lUnique+= l.C2[B1][0]+ l.C4[B1][0]+ l.C8[B1][0]; 
              for(int kj7=0;kj7<si;kj7++){//backtrack1の集計
                lTotal+= l3[kj7].C2[B1][1]*2+ l3[kj7].C4[B1][1]*4+ l3[kj7].C8[B1][1]*8;
                lUnique+= l3[kj7].C2[B1][1]+ l3[kj7].C4[B1][1]+ l3[kj7].C8[B1][1]; 
              }
            }
          }
        }
      }
    }
  }
}
void NQueenThread_25(){
  //printf("si>=25");
  for(int B1=1,B2=siE-1;B1<siE;B1++,B2--){// B1から順にスレッドを生成しながら処理を分担する 
    for(int k=0;k<si;k++){
      for(int j=0;j<si;j++){//backtrack1のB1
        for(int kj4=0;kj4<si;kj4++){//backtrack1のB1
          for(int kj5=0;kj5<si;kj5++){//backtrack1のB1
            for(int kj6=0;kj6<si;kj6++){//backtrack1のB1
              for(int kj7=0;kj7<si;kj7++){//backtrack1のB1
                //この中で回す
                pthread_t pt1;//スレッド childThread
                pthread_t pt3[si];//スレッド childThread
                local l;//構造体 local型 
                local l3[si];
                l.B1=B1; l.B2=B2;     
                for(int kj8=0;kj8<si;kj8++){//backtrack1のB1
                  l3[kj8].B1=B1; l3[kj8].B2=B2;
                }
                for(int i=0;i<siE;i++){ //aB[]の初期化
                  l.aB[i]=i;
                  for(int kj8=0;kj8<si;kj8++){
                    l3[kj8].aB[i]=i;  // 上から３行目のスレッドに使う構造体aB[]の初期化
                  }
                }
                l.k=-1;	
                l.j=-1;	
                l.kj4=-1;	
                l.kj5=-1;	
                l.kj6=-1;	
                l.kj7=-1;	
                l.kj8=-1;	
                l.C2[B1][0]= l.C4[B1][0]= l.C8[B1][0]=0;	
                for(int kj8=0;kj8<si;kj8++){
                  l3[kj8].k=-1;
                  l3[kj8].j=-1;
                  l3[kj8].kj4=-1;
                  l3[kj8].kj5=-1;
                  l3[kj8].kj6=-1;
                  l3[kj8].kj7=-1;
                  l3[kj8].kj8=-1;
                  l3[kj8].C2[B1][1]= l3[kj8].C4[B1][1]= l3[kj8].C8[B1][1]=0;	
                }
                if(B1<si/2){

                  for(int kj8=0;kj8<si;kj8++){
                    l3[kj8].k=k; l3[kj8].j=j;
                    l3[kj8].kj4=kj4;
                    l3[kj8].kj5=kj5;
                    l3[kj8].kj6=kj6;
                    l3[kj8].kj7=kj7;
                    l3[kj8].kj8=kj8;
                    pthread_create(&pt3[kj8],NULL,&run3,(void*)&l3[kj8]);// チルドスレッドの生成
                    if(THREAD<1){ // not Thread
                      pthread_join(pt3[kj8],NULL); 
                      pthread_detach(pt3[kj8]);
                    }
                  }
                  for(int kj8=0;kj8<si;kj8++){
                    pthread_join(pt3[kj8],NULL); 
                    pthread_detach(pt3[kj8]);
                  }
                }
                l.k=k; 
                l.j=j;
                l.kj4=kj4;
                l.kj5=kj5;
                l.kj6=kj6;
                l.kj7=kj7;
                pthread_create(&pt1,NULL,&run,(void*)&l);// チルドスレッドの生成
                if(THREAD<1){ // not Thread
                  pthread_join(pt1,NULL); 
                  pthread_detach(pt1);
                }
                pthread_join(pt1,NULL); 
                pthread_detach(pt1);
                lTotal+= l.C2[B1][0]*2+ l.C4[B1][0]*4+ l.C8[B1][0]*8;
                lUnique+= l.C2[B1][0]+ l.C4[B1][0]+ l.C8[B1][0]; 
                for(int kj8=0;kj8<si;kj8++){//backtrack1の集計
                  lTotal+= l3[kj8].C2[B1][1]*2+ l3[kj8].C4[B1][1]*4+ l3[kj8].C8[B1][1]*8;
                  lUnique+= l3[kj8].C2[B1][1]+ l3[kj8].C4[B1][1]+ l3[kj8].C8[B1][1]; 
                }
              }
            }
          }
        }
      }
    }
  }
}
