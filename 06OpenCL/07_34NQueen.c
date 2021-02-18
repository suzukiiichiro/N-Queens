/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 <>34. チェスボードのレイアウト表示機能

		コンパイルと実行
		$ make nq34 && ./07_34NQueen

  	#define BOARDLAYOUTVISIBLE 1 //ボードレイアウト表示
		をひとたび　1 をセットすると、08-QueenSolというテキストファイル群が
		カレントディレクトリに展開される。
		ソースは07_33NQueen.cを継承し、ロジックは07_27NQueens部分を生かした。

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
12:             14200              1787          00:00:00:00.01
13:             73712              9233          00:00:00:00.09
14:            365596             45752          00:00:00:00.50
15:           2279184            285053          00:00:00:03.45
16:          14772512           1846955          00:00:00:25.28
 
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include "unistd.h"

#define MAX 27      //求めるNの最大値
#define BOARDLAYOUTVISIBLE 1 //ボードレイアウト表示

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


/**
 * チェスボードのクイーンの場所を確認
 */
int db=0; //解のユニーク数のカウンタ
FILE *f ;
void thMonitor(local *l,int i){
  if(BOARDLAYOUTVISIBLE>0){
    db++;
    fprintf(f,"#N%d =%d C%d\n",si,db,i);
    for (int y=0;y<si;y++) {
      for (l->bit=l->TB; l->bit; l->bit>>=1){
        if(l->aB[y]==l->bit){
          fprintf(f,"Q ");
        }else{
          fprintf(f,". ");
        }
      }
      fprintf(f,"\n");
    }
    fprintf(f,"\n");
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
    if(l->own>siE){ l->C2[l->B1][l->BK]++; if(BOARDLAYOUTVISIBLE>0) thMonitor(l,2); return ; } }
  if(l->aB[siE]==l->EB){ //180度回転
    for(l->own=1,l->you=siE-1;l->own<=siE;l->own++,l->you--){ 
      for(l->bit=1,l->ptn=l->TB;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->ptn>>=1){}
      if(l->aB[l->own]>l->bit){ return; } 
      else if(l->aB[l->own]<l->bit){ break; } }
    //90度回転が同型でなくても180度回転が同型である事もある
    if(l->own>siE){ l->C4[l->B1][l->BK]++; if(BOARDLAYOUTVISIBLE>0) thMonitor(l,4); return; } }
  if(l->aB[l->B1]==l->TB){ //270度回転
    for(l->own=1,l->ptn=l->TB>>1;l->own<=siE;l->own++,l->ptn>>=1){ 
      for(l->bit=1,l->you=0;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->you++){}
      if(l->aB[l->own]>l->bit){ return; } else if(l->aB[l->own]<l->bit){ break; } } }
  l->C8[l->B1][l->BK]++;
  if(BOARDLAYOUTVISIBLE>0) thMonitor(l,8); 
}
void backTrack2(int y,int left,int down,int right,int bm,local *l){
  bm=l->msk&~(left|down|right); //配置可能フィールド
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
    }
    else{
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
    if(bm>0){ l->aB[y]=bm; l->C8[l->B1][l->BK]++; if(BOARDLAYOUTVISIBLE>0) thMonitor(l,82); } }
    //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
    else{ if(y<l->B1) { bm&=~2; }
    if(y==2 && l->k>=0){
      if(bm & (1<<l->k)){ l->aB[y]=l->bit=1<<l->k; } else{ return; }
      backTrack1(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }else if(y==3 && l->j>=0){
      if(bm & (1<<l->j)){ l->aB[y]=l->bit=1<<l->j; } else{ return; }
      backTrack1(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }
    else{
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
    l[B1].k=-1;	l[B1].j=-1;	
    l[B1].C2[B1][0]=0; l[B1].C4[B1][0]=0; l[B1].C8[B1][0]=0;	//カウンターの初期化 
    for(int k=0;k<si;k++){
      l2[B1][k].k=-1;	l2[B1][k].j=-1;	
      l2[B1][k].C2[B1][1]=0; l2[B1][k].C4[B1][1]=0; l2[B1][k].C8[B1][1]=0;	//カウンターの初期化 
    }
    if(B1<si/2){
      for(int k=0;k<si;k++){
        l2[B1][k].k=k;
        pthread_create(&pt2[B1][k],NULL,&run3,(void*)&l2[B1][k]);// チルドスレッドの生成
        if(BOARDLAYOUTVISIBLE>0){
          pthread_join(pt2[B1][k],NULL);
        }
      }
    }
    pthread_create(&pt1[B1],NULL,&run,(void*)&l[B1]);// チルドスレッドの生成
    if(BOARDLAYOUTVISIBLE>0){
      pthread_join(pt1[B1],NULL);
    }

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
  //for(int i=8;i<=8;i++){
    //Enableならボードレイアウトを出力
    if(BOARDLAYOUTVISIBLE>0){ 
			char filename[100];
			sprintf(filename, "%.2d-QueenSol", i);
      f=fopen(filename,"w"); db=0; 
    }
    si=i; siE=i-1; lTotal=lUnique=0;
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
    fclose(f);
  } 
  return 0;
}
