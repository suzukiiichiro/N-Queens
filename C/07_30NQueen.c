/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 <>30. マルチスレッドもっと最適化４段目    			        

		コンパイルと実行
		$ make nq30 && ./07_30NQueen

  現行の処理ではすでに最上段と上から3段目のクイーンまでは固定化してスレッド化している。
  今回は、上記に加えて上から4段目のクイーンまでを固定化してスレッド化（1XNxNxN)


	07_28 07_29は暴れん坊過ぎてハングアップするので修正、かつ強化。

          1   2   3   4
        +   +   +   +   *
  1       Q   Q   Q   Q     
        +   +   +   +   *
  2       Q   Q   Q   Q    
        +   +   +   +   *
  3       Q   Q   Q   Q    
        +   +   +   +   *
  4       Q   Q   Q   Q
        +   +   +   +   *                    

  補助機能として、THREAD フラグのトグルで、シングルスレッドモード、スレッドモードへの
  切り替えを可能とした。

  シングルスレッドモードで、Debugフラグを( 1=TRUE ) にすると、チェスボードのクイーンＱ
  配置を確認できる機能を実装した。

 N:        Total       Unique                 dd:hh:mm:ss.ms
 2:               0                0          00:00:00:00.00
 3:               0                0          00:00:00:00.00
 4:               8                4          00:00:00:00.00
 5:              10                2          00:00:00:00.00
 6:               4                1          00:00:00:00.00
 7:              40                6          00:00:00:00.01
 8:              92               12          00:00:00:00.02
 9:             352               46          00:00:00:00.03
10:             724               92          00:00:00:00.06
11:            2680              341          00:00:00:00.08
12:           14200             1787          00:00:00:00.13
13:           73712             9233          00:00:00:00.17
14:          365596            45752          00:00:00:00.25
15:         2279184           285053          00:00:00:00.36
16:        14772512          1846955          00:00:00:00.95
17:        95815104         11977939          00:00:00:04.53
18:       666090624         83263591          00:00:00:31.91
19:      4968057848        621012754          00:00:04:13.62
20:     39029188884       4878666808          00:00:31:20.65
21:    314666222712      39333324973          00:04:15:55.95 

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include "unistd.h"

#define MAX 27
#define DEBUG 1   // TRUE:1 FALSE:0
#define THREAD 0  // TRUE:1 FALSE:0

int si;  
int siE;
long lTotal;
long lUnique;

/** スレッドローカル構造体 */
typedef struct{
	int bit;
  int own;
	int ptn;
	int you;
  int k;  //上から２行目のスレッドに使う
  int j;  //上から３行目のスレッドに使う
  int kj4;
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

int db=0;
void thMonitor(local *l,int i){
  if(THREAD>0){
    //
  }else{
    db++;
  //	fprintf(f,"N%d =%d C%d\n",si,db,i);
    printf("N%d =%d C%d\n",si,db,i);
    for (int y=0;y<si;y++) {
      for (l->bit=l->TB; l->bit; l->bit>>=1){
        if(l->aB[y]==l->bit){
  //				fprintf(f, "Q ");
          printf("Q ");
        }else{
  //				fprintf(f, ". ");
          printf(". ");
        }
      }
  //		fprintf(f,"\n");
      printf("\n");
    }
  //	fprintf(f,"\n");
    printf("\n");
  // sleep(1);
  } 
}


void symmetryOps_bm(local *l);
void backTrack1stLine(int y,int left,int down,int right,int bm,local *l);
void backTrack1stLine2(int y,int left,int down,int right,int bm,local *l);
void backTrack2ndLine(int y,int left,int down,int right,int bm,local *l);
void backTrack3rdLine(int y,int left,int down,int right,int bm,local *l);
void backTrack3rdLine2(int y,int left,int down,int right,int bm,local *l);
void NoCornerQ(int y,int left,int down,int right,int bm,local *l2);
void cornerQ(int y,int left,int down,int right,int bm,local *l);
void *run(void *args);
void *run2(void *args);
void *run3(void *args);
void *NQueenThread();
void NQueen();

void symmetryOps_bm(local *l){
  l->own=l->ptn=l->you=l->bit=0;
  //l->C8[l->B1][l->BK]++;
  //if(DEBUG>0) thMonitor(l,8); 
  //90度回転
  if(l->aB[l->B2]==1){ 
    for(l->own=1,l->ptn=2;l->own<=siE;l->own++,l->ptn<<=1){ 
      for(l->bit=1,l->you=siE;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->you--){}
      if(l->aB[l->own]>l->bit){ 
        //l->C8[l->B1][l->BK]--; 
        return; 
      }else if(l->aB[l->own]<l->bit){ 
        break; 
      }
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(l->own>siE){ 
      l->C2[l->B1][l->BK]++;
      if(DEBUG>0) thMonitor(l,2); 
      //l->C8[l->B1][l->BK]--;
      return ; 
    } 
  }
  //180度回転
  if(l->aB[siE]==l->EB){ 
    for(l->own=1,l->you=siE-1;l->own<=siE;l->own++,l->you--){ 
      for(l->bit=1,l->ptn=l->TB;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->ptn>>=1){}
      if(l->aB[l->own]>l->bit){ 
        //l->C8[l->B1][l->BK]--; 
        return; 
      } 
      else if(l->aB[l->own]<l->bit){ 
        break; 
      }
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(l->own>siE){ 
      l->C4[l->B1][l->BK]++;
      if(DEBUG>0) thMonitor(l,4); 
      //l->C8[l->B1][l->BK]--;
      return; 
    } 
  }
  //270度回転
  if(l->aB[l->B1]==l->TB){ 
    for(l->own=1,l->ptn=l->TB>>1;l->own<=siE;l->own++,l->ptn>>=1){ 
      for(l->bit=1,l->you=0;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->you++){}
      if(l->aB[l->own]>l->bit){ 
        //l->C8[l->B1][l->BK]--; 
        return; 
      } 
      else if(l->aB[l->own]<l->bit){ 
        break; 
      }
    }
  }
  l->C8[l->B1][l->BK]++;
  if(DEBUG>0) thMonitor(l,8); 
}
//backtrack2の3行目の列数を固定して場合分けすることによりスレッドを分割する
//void backTrack3(int y,int left,int down,int right,int bm,local *l){
void backTrack3rdLine(int y,int left,int down,int right,int bm,local *l){
  bm=l->msk&~(left|down|right); //配置可能フィールド
  l->bit=0;
  if(y==siE){
    if(bm>0 && (bm&l->LM)==0){ //【枝刈り】最下段枝刈り
      l->aB[y]=bm;
      symmetryOps_bm(l);//対称解除法
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
    if(bm & (1<<l->j)){
      //スレッドの引数として指定した3行目のクイーンの位置jを固定で指定する
      l->aB[y]=l->bit=1<<l->j;
    }else{
      //left,down,rightなどkの値がクイーンの位置として指定できない場合はスレッド終了させる
      return;
    }
    //4行目以降は通常のbacktrack2の処理に渡す
    //NoCornerQ(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    backTrack3rdLine2(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
  }
}
void backTrack3rdLine2(int y,int left,int down,int right,int bm,local *l){
  bm=l->msk&~(left|down|right); //配置可能フィールド
  l->bit=0;
  if(y==siE){
    if(bm>0 && (bm&l->LM)==0){ //【枝刈り】最下段枝刈り
      l->aB[y]=bm;
      symmetryOps_bm(l);//対称解除法
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
    //if(bm & (1<<l->j)){
    if(bm & (1<<l->kj4)){
      //スレッドの引数として指定した3行目のクイーンの位置jを固定で指定する
      //l->aB[y]=l->bit=1<<l->j;
      l->aB[y]=l->bit=1<<l->kj4;
    }else{
      //left,down,rightなどkの値がクイーンの位置として指定できない場合はスレッド終了させる
      return;
    }
    //4行目以降は通常のbacktrack2の処理に渡す
    NoCornerQ(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
  }
}
//backtrack2の2行目の列数を固定して場合分けすることによりスレッドを分割する
//void backTrack3(int y,int left,int down,int right,int bm,local *l){
void backTrack2ndLine(int y,int left,int down,int right,int bm,local *l){
  bm=l->msk&~(left|down|right); //配置可能フィールド
  l->bit=0;
  if(y==siE){
    if(bm>0 && (bm&l->LM)==0){ //【枝刈り】最下段枝刈り
      l->aB[y]=bm;
      symmetryOps_bm(l);//対称解除法
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
    //backtrack2に行かず、backtrack3rdlineに行き3行目のクイーンの位置も固定する
    backTrack3rdLine(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
  }
}
//backtrack1の3行目のクイーンの値を固定
void backTrack1stLine(int y,int left,int down,int right,int bm,local *l){
  bm=l->msk&~(left|down|right); 
  l->bit=0;
  if(y==siE) {
    if(bm>0){//【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      l->aB[y]=bm;
      l->C8[l->B1][l->BK]++;
      if(DEBUG>0) thMonitor(l,8); 
    }
  }else{
    if(y<l->B1) { //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい  
      bm&=~2; 
    }
    if(bm & (1<<l->k)){
      //スレッドの引数として指定した3行目のクイーンの位置kを固定で指定する
      l->aB[y]=l->bit=1<<l->k;
    }else{
      //left,down,rightなどkの値がクイーンの位置として指定できない場合はスレッド終了させる
      return;
    }
    //4行目以降はbacktrack1の処理
      //cornerQ(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
      backTrack1stLine2(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
  } 
}
//backtrack1の3行目のクイーンの値を固定
void backTrack1stLine2(int y,int left,int down,int right,int bm,local *l){
  bm=l->msk&~(left|down|right); 
  l->bit=0;
  if(y==siE) {
    if(bm>0){//【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      l->aB[y]=bm;
      l->C8[l->B1][l->BK]++;
      if(DEBUG>0) thMonitor(l,8); 
    }
  }else{
    if(y<l->B1) { //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい  
      bm&=~2; 
    }
    //if(bm & (1<<l->k)){
    if(bm & (1<<l->j)){
      //スレッドの引数として指定した3行目のクイーンの位置kを固定で指定する
      //l->aB[y]=l->bit=1<<l->k;
      l->aB[y]=l->bit=1<<l->j;
    }else{
      //left,down,rightなどkの値がクイーンの位置として指定できない場合はスレッド終了させる
      return;
    }
    //4行目以降はbacktrack1の処理
      cornerQ(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
  } 
}
//上から１行目角にクイーンがない場合の処理
//void backTrack2(int y,int left,int down,int right,int bm,local *l){
void NoCornerQ(int y,int left,int down,int right,int bm,local *l){
  bm=l->msk&~(left|down|right); //配置可能フィールド
  //bmはクイーンが置ける場所
  //l->msk はsi分1が並んでいる
  //そこから引数に渡されてきたleft,right,downを取り除く。
  //msk
  //11111111
  //left
  //00011000
  //down
  //00001010
  //right
  //00000100
  //bmp
  //11100001
  l->bit=0;
  if(y==siE){
  //yが1番下に来たら
    if(bm>0 && (bm&l->LM)==0){ //【枝刈り】最下段枝刈り
      //1番下の行にクイーンを置けるか判定する
      //bm>0について
      //bmは残り1個しか残っていないので bm>0かどうかだけ判定し
      //0だったら配置する場所がないので抜ける
      //0より大きければ最下位のビットを抽出するまでものくその値がaB[y]になる
      //bm:   00001000
      //l->aB:00001000
      //(bm&l->LM)==0について
      //最下段でLMにひっかかるものはここで刈り取られる
      //最下段はLMに当たる場所にクイーンはおけない
      //両端どちらかが1
      //この場合はOK
      //bm      :00100000
      //LM      :11000011
      //bm&l->LM:00000000
      //この場合は刈り取られる
      //bm      :00000010
      //LM      :11000011
      //bm&l->LM:00000010
      l->aB[y]=bm;
      symmetryOps_bm(l);//対称解除法
    }
  }else{
    if(y<l->B1){ //【枝刈り】上部サイド枝刈り            
      bm&=~l->SM; 
      //SMは左右両端が1 10000001
      //左右両端を刈り込む
      //bm:11110001
      //SM:10000001
      //bm:01110000
    }else if(y==l->B2) { //【枝刈り】下部サイド枝刈り    
      if((down&l->SM)==0){ 
      //downの両端が0の場合にdown&SM=0になる
      //down   :10011110
      //SM     :10000001
      //down&SM:10000000
      //down   :01011110
      //SM     :10000001
      //down&SM:00000000
        return; 
      }
      if((down&l->SM)!=l->SM){ 
      //(down&l->SM)!=l->SM
      //両端どちらも1の場合は(down&l->SM)==l->SM
        bm&=l->SM; 
        //両端の1だけ残す
        //bm:00000001
        //SM:10000001
        //bm:00000001
      }
    }
    while(bm>0) {
      //bmが0になると抜ける
      //最も下位の1をとってaB[y],l->bitに設定する
      //bmが0になるとクイーンを置ける可能性がある場所がなくなるので抜ける
      //yが最後まで行っていなくてもbmが0になれば抜ける
      bm^=l->aB[y]=l->bit=-bm&bm;
      //最も下位の１ビットを抽出
      //bmの中で1番桁数が少ない1を0にする
      //aB[y],l->bitにその値を設定する
      //11100001
      //この場合1番桁数の低い右端の1が選択される
      //aB[y]
      //00000001
      //bm
      //11100000
      //次のbacktrackに渡すleft,down,rightを設定する
      //left,down,rightは、y1から蓄積されていく
      //left はleft(今までのleftライン)とl->bit(今回選択されたクイーンの位置)を左に1ビットシフト
      //left        00110010
      //l->bit      00000100
      //left|l->bit 00110110
      //1bit左シフト01101100
      //downはdown(今までのdownライン) と l->bit(今回選択されたクイーンの位置)
      //down        00001011
      //l->bit      00000100
      //down|l->bit 00001111
      //rightはright(今までのrightライン)とl->bit(今回選択されたクイーンの位置)を右に1ビットシフト
      //right       00000010
      //l->bit      00000100
      //right|l->bit00000110
      //1bit右シフト00000011
      NoCornerQ(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }
  }
}
//上から１行目角にクイーンがある場合の処理
//void backTrack1(int y,int left,int down,int right,int bm,local *l){
void cornerQ(int y,int left,int down,int right,int bm,local *l){
  bm=l->msk&~(left|down|right); 
  //bmはクイーンが置ける場所
  //l->msk はsi分1が並んでいる
  //そこから引数に渡されてきたleft,right,downを取り除く。
  //msk
  //11111111
  //left
  //00011000
  //down
  //00001010
  //right
  //00000100
  //bmp
  //11100001
  l->bit=0;
  if(y==siE) {
  //yが1番下に来たら
    if(bm>0){
      //1番下の行にクイーンを置けるか判定する
      //bmは残り1個しか残っていないので bm>0かどうかだけ判定し
      //0だったら配置する場所がないので抜ける
      //0より大きければ最下位のビットを抽出するまでものくその値がaB[y]になる
      //bm:   00001000
      //l->aB:00001000
      l->aB[y]=bm;
      //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      //y<B1の時に右から2列目を刈り込んでおけばいい
      l->C8[l->B1][l->BK]++;
      if(DEBUG>0) thMonitor(l,8); 
    }
  }else{
    if(y<l->B1) { //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい  
      //backtrack1ではy<B1の間は右から2個目にクイーンを配置しない。
      //これでユニーク解であることが保証される
      //bm:10001010
      // 2:00000010
      //bm:10001000 
      bm&=~2; 
    }
    while(bm>0) {
      //bmが0になると抜ける
      //最も下位の1をとってaB[y],l->bitに設定する
      //bmが0になるとクイーンを置ける可能性がある場所がなくなるので抜ける
      //yが最後まで行っていなくてもbmが0になれば抜ける
      bm^=l->aB[y]=l->bit=-bm&bm;
      //最も下位の１ビットを抽出
      //bmの中で1番桁数が少ない1を0にする
      //aB[y],l->bitにその値を設定する
      //11100001
      //この場合1番桁数の低い右端の1が選択される
      //aB[y]
      //00000001
      //bm
      //11100000
      cornerQ(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
      //次のbacktrackに渡すleft,down,rightを設定する
      //left,down,rightは、y1から蓄積されていく
      //left はleft(今までのleftライン)とl->bit(今回選択されたクイーンの位置)を左に1ビットシフト
      //left        00110010
      //l->bit      00000100
      //left|l->bit 00110110
      //1bit左シフト01101100
      //downはdown(今までのdownライン) と l->bit(今回選択されたクイーンの位置)
      //down        00001011
      //l->bit      00000100
      //down|l->bit 00001111
      //rightはright(今までのrightライン)とl->bit(今回選択されたクイーンの位置)を右に1ビットシフト
      //right       00000010
      //l->bit      00000100
      //right|l->bit00000110
      //1bit右シフト00000011
    }
  } 
}
//backtrack2のマルチスレッド処理
//３行目のクイーンの位置まで固定して別スレッドで走らせる
//NXNXNスレッドが立っている
void *run3(void *args){
  local *l=(local *)args;
  l->msk=(1<<si)-1; //l->makはクイーンを置ける場所 si分1が並ぶ
  //si=8 なら 1が8個並ぶ
  l->TB=1<<siE;
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
    //backTrack3(1,l->bit<<1,l->bit,l->bit>>1,0,l);
    backTrack2ndLine(1,l->bit<<1,l->bit,l->bit>>1,0,l);
    l->EB>>=si;
  }
  return 0;
}
//backtrack1のマルチスレッド処理
void *run(void *args){
  local *l=(local *)args;
  l->bit=0 ; 
  //backtrack1は1行目のクイーンの位置を右端に固定
  l->aB[0]=1; 
  l->msk=(1<<si)-1; l->TB=1<<siE; l->BK=0;
  if(l->B1>1 && l->B1<siE) { // 最上段のクイーンが角にある場合の探索
    //backtrack1は2行目のクイーンの位置はl->B1
    l->aB[1]=l->bit=(1<<l->B1);// 角にクイーンを配置 
    //3行目をkの値に固定する
    backTrack1stLine(2,(2|l->bit)<<1,(1|l->bit),(l->bit>>1),0,l);//２行目から探索
  }
  return 0;
}
void *NQueenThread(){
  //Nの数だけスレッドをもたせて同時並列処理をする
  //backtrack1
  //スレッド数は1xNxN
  //1行目 クイーンは右端に固定
  //2行目 B1の値によってスレッドを分割する
  //3行目 kの値によってスレッドを分割する
  //N=4の場合のスレッドの例
  //aB[0]は１行目のクイーンの位置、aB[1]は２行目のクイーンの位置、aB[2]は３行目のクイーンの位置
  //backtrack1の場合は１行目のクイーンの位置は右端1に固定
  //aB[0]=1,aB[1]=1,aB[2]=1
  //aB[0]=1,aB[1]=1,aB[2]=2
  //aB[0]=1,aB[1]=1,aB[2]=3
  //aB[0]=1,aB[1]=1,aB[2]=4
  //aB[0]=1,aB[1]=2,aB[2]=1
  //aB[0]=1,aB[1]=2,aB[2]=2
  //aB[0]=1,aB[1]=2,aB[2]=3
  //aB[0]=1,aB[1]=2,aB[2]=4
  //aB[0]=1,aB[1]=3,aB[2]=1
  //aB[0]=1,aB[1]=3,aB[2]=2
  //aB[0]=1,aB[1]=3,aB[2]=3
  //aB[0]=1,aB[1]=3,aB[2]=4
  //aB[0]=1,aB[1]=4,aB[2]=1
  //aB[0]=1,aB[1]=4,aB[2]=2
  //aB[0]=1,aB[1]=4,aB[2]=3
  //aB[0]=1,aB[1]=4,aB[2]=4
  //pthread_t pt1[si][si];    //上から2段目のスレッド childThread
  //pthread_t pt1[si][si][si];    //上から3段目のスレッド childThread
  //backtrack2
  //スレッド数はNxNxN
  //1行目 B1の値によってスレッドを分割する
  //2行目 kの値によってスレッドを分割する
  //3行目 jの値によってスレッドを分割する
  //N=4の場合のスレッドの例
  //aB[0]は１行目のクイーンの位置、aB[1]は２行目のクイーンの位置
  //backtrack2の場合は１行目のクイーンの位置は右端以外2~4
  //２行目、３行目は1~4
  //aB[0]=2,aB[1]=1,aB[2]=1
  //aB[0]=2,aB[1]=1,aB[2]=2
  //aB[0]=2,aB[1]=1,aB[2]=3
  //aB[0]=2,aB[1]=1,aB[2]=4
  //aB[0]=2,aB[1]=2,aB[2]=1
  //aB[0]=2,aB[1]=2,aB[2]=2
  //aB[0]=2,aB[1]=2,aB[2]=3
  //aB[0]=2,aB[1]=2,aB[2]=4
  //aB[0]=2,aB[1]=3,aB[2]=1
  //aB[0]=2,aB[1]=3,aB[2]=2
  //aB[0]=2,aB[1]=3,aB[2]=3
  //aB[0]=2,aB[1]=3,aB[2]=4
  //aB[0]=2,aB[1]=4,aB[2]=1
  //aB[0]=2,aB[1]=4,aB[2]=2
  //aB[0]=2,aB[1]=4,aB[2]=3
  //aB[0]=2,aB[1]=4,aB[2]=4
  //aB[0]=3,aB[1]=1,aB[2]=1
  //aB[0]=3,aB[1]=1,aB[2]=2
  //aB[0]=3,aB[1]=1,aB[2]=3
  //aB[0]=3,aB[1]=1,aB[2]=4
  //aB[0]=3,aB[1]=2,aB[2]=1
  //aB[0]=3,aB[1]=2,aB[2]=2
  //aB[0]=3,aB[1]=2,aB[2]=3
  //aB[0]=3,aB[1]=2,aB[2]=4
  //aB[0]=3,aB[1]=3,aB[2]=1
  //aB[0]=3,aB[1]=3,aB[2]=2
  //aB[0]=3,aB[1]=3,aB[2]=3
  //aB[0]=3,aB[1]=3,aB[2]=4
  //aB[0]=3,aB[1]=4,aB[2]=1
  //aB[0]=3,aB[1]=4,aB[2]=2
  //aB[0]=3,aB[1]=4,aB[2]=3
  //aB[0]=3,aB[1]=4,aB[2]=4
  //aB[0]=4,aB[1]=1,aB[2]=1
  //aB[0]=4,aB[1]=1,aB[2]=2
  //aB[0]=4,aB[1]=1,aB[2]=3
  //aB[0]=4,aB[1]=1,aB[2]=4
  //aB[0]=4,aB[1]=2,aB[2]=1
  //aB[0]=4,aB[1]=2,aB[2]=2
  //aB[0]=4,aB[1]=2,aB[2]=3
  //aB[0]=4,aB[1]=2,aB[2]=4
  //aB[0]=4,aB[1]=3,aB[2]=1
  //aB[0]=4,aB[1]=3,aB[2]=2
  //aB[0]=4,aB[1]=3,aB[2]=3
  //aB[0]=4,aB[1]=3,aB[2]=4
  //aB[0]=4,aB[1]=4,aB[2]=1
  //aB[0]=4,aB[1]=4,aB[2]=2
  //aB[0]=4,aB[1]=4,aB[2]=3
  //aB[0]=4,aB[1]=4,aB[2]=4
  //pthread_t pt3[si][si][si][si];//上から4段目のスレッド childThread

  pthread_t ***pt1=(pthread_t***)malloc(sizeof(pthread_t*)*si*si*si); //B1xk
  for(int B1=1;B1<=si;B1++){
      pt1=(pthread_t***)malloc(sizeof(pthread_t)*si);
      if( pt1 == NULL ) { printf( "memory cannot alloc!\n" ); }
    for(int j=0;j<si;j++){
      pt1[j]=(pthread_t**)malloc(sizeof(pthread_t)*si);
      if( pt1[j] == NULL ) { printf( "memory cannot alloc!\n" ); }
      for(int k=0;k<si;k++){
        pt1[j][k]=(pthread_t*)malloc(sizeof(pthread_t)*si);
        if( pt1[j][k] == NULL ) { printf( "memory cannot alloc!\n" ); }
      }
    }
  } 
  pthread_t ****pt3=(pthread_t****)malloc(sizeof(pthread_t*)*si*si*si*si); //1xkxj
  for(int B1=1;B1<=si;B1++){
      pt3=(pthread_t****)malloc(sizeof(pthread_t)*si);
    for(int j=0;j<si;j++){
        pt3[j]=(pthread_t***)malloc(sizeof(pthread_t)*si);
        if( pt3[j] == NULL ) { printf( "memory cannot alloc!\n" ); }
      for(int k=0;k<si;k++){
        pt3[j][k]=(pthread_t**)malloc(sizeof(pthread_t)*si);
        if( pt3[j][k] == NULL ) { printf( "memory cannot alloc!\n" ); }
        for(int kj4=0;kj4<si;kj4++){
          pt3[j][k][kj4]=(pthread_t*)malloc(sizeof(pthread_t)*si);
          if( pt3[j][k][kj4] == NULL ) { printf( "memory cannot alloc!\n" ); }
        }
      }
    }
  }
  //local l[si][si];   //構造体 local型  backtrack1
  local ***l=(local***)malloc(sizeof(local*)*si*si*si); //B1xk
  for(int B1=1;B1<si;B1++){
      l=(local***)malloc(sizeof(local)*si);
      if( l == NULL ) { printf( "memory cannot alloc!\n" ); }
    for(int j=0;j<si;j++){
      l[j]=(local**)malloc(sizeof(local)*si);
      if( l[j] == NULL ) { printf( "memory cannot alloc!\n" ); }
      for(int k=0;k<si;k++){
        l[j][k]=(local*)malloc(sizeof(local)*si);
        if( l[j][k] == NULL ) { printf( "memory cannot alloc!\n" ); }
      }
    }
  } 
  //local l3[si][si][si];   //構造体 local型  backtrack2
  local ****l3=(local****)malloc(sizeof(local*)*si*si*si*si); //1xkxj
  for(int B1=1;B1<=si;B1++){
      l3=(local****)malloc(sizeof(local)*si);
    for(int j=0;j<si;j++){
        l3[j]=(local***)malloc(sizeof(local)*si);
        if( l3[j] == NULL ) { printf( "memory cannot alloc!\n" ); }
      for(int k=0;k<si;k++){
        l3[j][k]=(local**)malloc(sizeof(local)*si);
        if( l3[j][k] == NULL ) { printf( "memory cannot alloc!\n" ); }
        for(int kj4=0;kj4<si;kj4++){
          l3[j][k][kj4]=(local*)malloc(sizeof(local)*si);
          if( l3[j][k][kj4] == NULL ) { printf( "memory cannot alloc!\n" ); }
        }
      }
    }
  }
  
  for(int B1=1,B2=siE-1;B1<siE;B1++,B2--){// B1から順にスレッドを生成しながら処理を分担する 
  //1行目のクイーンのパタン*2行目のクイーンのパタン
  //1行目 最上段の行のクイーンの位置は中央を除く右側の領域に限定。
  //B1 と B2を初期化
    for(int k=0;k<si;k++){
      //backtrack1のB1
      for(int j=0;j<si;j++){
        l[B1][k][j].B1=B1; 
        l[B1][k][j].B2=B2;     
        for(int kj4=0;kj4<si&&B1<si/2;kj4++){
          l3[B1][k][j][kj4].B1=B1;
          l3[B1][k][j][kj4].B2=B2;
        }
      }
    }
    //aB[]の初期化
    for(int i=0;i<siE;i++){ 
      for(int k=0;k<si;k++){
        for(int j=0;j<si;j++){
          l[B1][k][j].aB[i]=i;
          for(int kj4=0;kj4<si&&B1<si/2;kj4++){
            l3[B1][k][j][kj4].aB[i]=i;  // 上から３行目のスレッドに使う構造体aB[]の初期化
          }
        }
      }
    } 
  }
  for(int B1=1,B2=siE-1;B1<siE;B1++,B2--){// B1から順にスレッドを生成しながら処理を分担する 
   //カウンターの初期化
    for(int k=0;k<si;k++){
      for(int j=0;j<si;j++){
        l[B1][k][j].C2[B1][0]=
        l[B1][k][j].C4[B1][0]= 
        l[B1][k][j].C8[B1][0]=0;	
        for(int kj4=0;kj4<si&&B1<si/2;kj4++){
          l3[B1][k][j][kj4].C2[B1][1]= 
          l3[B1][k][j][kj4].C4[B1][1]= 
          l3[B1][k][j][kj4].C8[B1][1]=0;	
        }
      }
    }
  }
  for(int B1=1,B2=siE-1;B1<siE;B1++,B2--){// B1から順にスレッドを生成しながら処理を分担する 
    //if(B1>0&&B1<siE&&B1<B2){// B1から順にスレッドを生成しながら処理を分担する 
    //backtrack1のチルドスレッドの生成
    //B,kのfor文の中で回っているのでスレッド数はNxN
      if(B1<si/2){
        for(int k=0;k<si;k++){
          for(int j=0;j<si;j++){
            for(int kj4=0;kj4<si;kj4++){
              l3[B1][k][j][kj4].k=k;
              l3[B1][k][j][kj4].j=j;
              l3[B1][k][j][kj4].kj4=kj4;
              pthread_create(&pt3[B1][k][j][kj4],NULL,&run3,(void*)&l3[B1][k][j][kj4]);// チルドスレッドの生成
              if(THREAD<1){ // not Thread
                pthread_join(pt3[B1][k][j][kj4],NULL); 
                pthread_detach(pt3[B1][k][j][kj4]);
              }
            }
            for(int kj4=0;kj4<si;kj4++){
              pthread_join(pt3[B1][k][j][kj4],NULL); 
              pthread_detach(pt3[B1][k][j][kj4]);
            }
          }
        }
      }
      for(int k=0;k<si;k++){
        for(int j=0;j<si;j++){
          l[B1][k][j].k=k;
          l[B1][k][j].j=j;
          pthread_create(&pt1[B1][k][j],NULL,&run,(void*)&l[B1][k][j]);// チルドスレッドの生成
          if(THREAD<1){ // not Thread
            pthread_join(pt1[B1][k][j],NULL); 
            pthread_detach(pt1[B1][k][j]);
          }
        }
        for(int j=0;j<si;j++){
          pthread_join(pt1[B1][k][j],NULL); 
          pthread_detach(pt1[B1][k][j]);
        }
      }
  }
  /**
  //スレッドのjoin
  for(int B1=1;B1<siE;B1++){ 
    for(int k=0;k<si;k++){
      for(int j=0;j<si;j++){
        pthread_join(pt1[B1][k][j],NULL); 
        for(int kj4=0;kj4<si;kj4++){
          pthread_join(pt3[B1][k][j][kj4],NULL); 
        }
      }
    }
  }
  */
  //スレッド毎のカウンターを合計
  for(int B1=1;B1<siE;B1++){
    for(int k=0;k<si;k++){
      //backtrack1の集計
      for(int j=0;j<si;j++){
        lTotal+=
          l[B1][k][j].C2[B1][0]*2+
          l[B1][k][j].C4[B1][0]*4+
          l[B1][k][j].C8[B1][0]*8;
        lUnique+=
          l[B1][k][j].C2[B1][0]+
          l[B1][k][j].C4[B1][0]+
          l[B1][k][j].C8[B1][0]; 
        for(int kj4=0;kj4<si&&B1<si/2;kj4++){
          //backtrack2の集計
          lTotal+=
            l3[B1][k][j][kj4].C2[B1][1]*2+
            l3[B1][k][j][kj4].C4[B1][1]*4+
            l3[B1][k][j][kj4].C8[B1][1]*8;
          lUnique+=
            l3[B1][k][j][kj4].C2[B1][1]+
            l3[B1][k][j][kj4].C4[B1][1]+
            l3[B1][k][j][kj4].C8[B1][1]; 
          }
      }
    }
  }
  return 0;
}
void NQueen(){
  pthread_t pth;//スレッド変数
  pthread_create(&pth, NULL, &NQueenThread, NULL);// メインスレッドの生成
  pthread_join(pth, NULL); //スレッドの終了を待つ
  pthread_detach(pth);
}
int main(void){
  int min=8;
  struct timeval t0;
  struct timeval t1;
//	f=fopen("out","w"); 
  printf("%s\n"," N:        Total       Unique                 dd:hh:mm:ss.ms");
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
    printf("%2d:%16ld%17ld%12.2d:%02d:%02d:%02d.%02d\n", i,lTotal,lUnique,dd,hh,mm,ss,ms); 
  } 
//	fclose(f);
  return 0;
}
