/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
   １．ブルートフォース（力まかせ探索） NQueen01() N8=16,777,216通り
   ２．配置フラグ（制約テスト高速化）   NQueen02() N8=    40,320通り
   ３．バックトラック                   NQueen03() N17= 12:29.59
   ４．対称解除法(回転と斜軸）          NQueen04() N17=  8:11.18
   ５．枝刈りと最適化                   NQueen05() N17=  2:40.86
   ６．ビットマップ                     NQueen06() N17=  1:23.70
   ７．ビットマップ+対称解除法          NQueen07() N17=  2:41.67
   ８．ビットマップ+クイーンの場所で分岐NQueen08() N17=  1:36.28
   ９．ビットマップ+枝刈りと最適化      NQueen09() N17=    15.77
   10．もっとビットマップ(takaken版)    NQueen10() N17=    19.16
   11．マルチスレッド(構造体)           NQueen11() N17=    15.57
   12．マルチスレッド(pthread)          NQueen12() N17=    15.47
   13．マルチスレッド(mutex)            NQueen13() N17=    15.65
   14．マルチスレッド(mutexattr)        NQueen14() N17=    07.07
   15．マルチスレッド(脱mutex COUNT強化)NQueen15() N17=    04.15
   15t.もっとマルチスレッド(takaken版) NQueen15_t()N17=    12.59 
   16．アドレスとポインタ(考察１)       NQueen16() N17=    04.11
   17．アドレスとポインタ(考察２)       NQueen17() N17=    05.48
   18．アドレスとポインタ(考察３)       NQueen18() N17=    05.71
   19．アドレスとポインタ(考察４)       NQueen19() N17=    03.66
   20．アドレスとポインタ(考察５)       NQueen20() N17=    03.82
   21．アドレスとポインタ(考察６)       NQueen21() N17=    03.80
   22．アドレスとポインタ(考察７)       NQueen22() N17=    03.85
   23．アドレスとポインタ(考察８)       NQueen23() N17=    03.73
   24．アドレスとポインタ(完結)         NQueen24() N17=    03.73
 <>25．最適化 									        NQueen25() N17=    03.70

 # Java/C/Lua/Bash版
 # https://github.com/suzukiiichiro/N-Queen
 

 <>25．最適化 									        NQueen25() N17=03:70
=== 1 ===
 G構造体に格納していた int si int siE int lTotal int lUniqueを
 グローバル変数に置き換えました。ちょっと速くなりました。

=== 2 ===
 L構造体に格納していたC2/C4/C8カウンターの置き場所を変えて比較

1.
// long C2[MAX]; //グローバル環境に置くと N=17: 08.04
// long C4[MAX];
// long C8[MAX];

2.
//  long C2; // 構造体の中の配列をなくすとN=17: 05.24
//  long C4;
//  long C8;

3. 構造体の中でポインタアクセスにしてみる // N=17 : 05.87
   さらにcallocにより、宣言時に適切なメモリサイズを割り当てる
// int *ab; 
//  l[B1].aB=calloc(G.si,sizeof(int));

4.
  long C2[MAX];//構造体の中の配列を活かすと   N=17: 04.33
  long C4[MAX];
  long C8[MAX];

 よって、カウンターはL構造体の中に配置し、スレッド毎にカウンター
を管理する配列で構築しました。
同様に、カウントする箇所は以下のように書き換えました。

			l->C4[l->B1]++;

これによりちょっと速くなりました。

=== 3 ===
　symmetryOps_bm()/trackBack1()/trackBack2()のメソッドないで宣言されている
ローカル変数を撲滅しました。
　symmetryOps_bm()の中では以下の通りです。

  int own,ptn,you,bit;

こちらは全てL構造体でもち、
　l->own などでアクセスするようにしました。構造体に配置すると遅くなる
　という本をよく見ますが、激しく呼び出されるメソッドで変数が都度生成される
　コストと比べると計測から見れば、構造体で持った方が速いと言うことがわかりました。
これによりちょっと速くなりました。

=== 4 ===
 backTrack1()/backTrack2()のbm以外の変数はbitだけです。こちらは簡単に構造体に
　格納して実装することができました。問題はbm(bitmap)です。
　こちらは、再帰で変化する変数で、スレッド毎に値も変わることから値渡しである
　必要があります。よって関数の引数の中に格納することとしました。

void backTrack2(int y,int left,int down,int right,int bm,local *l){
void backTrack1(int y,int left,int down,int right,int bm,local *l){
これによりちょっと速くなりました。

=== 5 ===
pthreadや構造体 lは　#defineで宣言されるMAX=27を使って初期化していました。
si siEをグローバル変数としたことで、これらもNの値で初期化することとしました。

void *NQueenThread(){
  //pthread_t pt[G.si];//スレッド childThread
  pthread_t pt[si];//スレッド childThread
  //local l[MAX];//構造体 local型 
  local l[si];//構造体 local型 


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
15:         2279184           285053          00:00:00:00.10
16:        14772512          1846955          00:00:00:00.58
17:        95815104         11977939          00:00:00:03.69


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
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#define MAX 27

int si;  //si siE lTotal lUnique をグローバルに置くと N=17: 04.26
int siE;
long lTotal;
long lUnique;

/** スレッドローカル構造体 */
typedef struct{
	int bit;
  int own;
	int ptn;
	int you;
  int B1;
  int B2;
  int TB;
  int EB;
  int msk;
  int SM;
  int LM;
  int aB[MAX]; 
	// int *ab; // N=17 : 05.87
  //  l[B1].aB=calloc(G.si,sizeof(int));
  long C2[MAX];//構造体の中の配列を活かすと   N=17: 04.33
  long C4[MAX];
  long C8[MAX];
//  long C2; // 構造体の中の配列をなくすとN=17: 05.24
//  long C4;
//  long C8;
}local ;

// long C2[MAX]; //グローバル環境に置くと N=17: 08.04
// long C4[MAX];
// long C8[MAX];

//グローバル構造体
//typedef struct {
//  int si;
//  int siE;
//  long lTotal;
//  long lUnique;
//}GCLASS, *GClass;
//GCLASS G; 


void symmetryOps_bm(local *l);
void backTrack2(int y,int left,int down,int right,int bm,local *l);
void backTrack1(int y,int left,int down,int right,int bm,local *l);
void *run(void *args);
void *NQueenThread();
void NQueen();

void symmetryOps_bm(local *l){
  //int own,ptn,you,bit;
  l->own=l->ptn=l->you=l->bit=0;
  //90度回転
  if(l->aB[l->B2]==1){ l->own=1; l->ptn=2;
    //while(l->own<=G.siE){ l->bit=1; l->you=G.siE;
    while(l->own<=siE){ l->bit=1; l->you=siE;
      while((l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit)){ l->bit<<=1; l->you--; }
      if(l->aB[l->own]>l->bit){ return; } if(l->aB[l->own]<l->bit){ break; }
      l->own++; l->ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    //if(l->own>G.siE){ 
    if(l->own>siE){ 
			//C2[l->B1]++;
			l->C2[l->B1]++;
      return; }
  }
  //180度回転
  //if(l->aB[G.siE]==l->EB){ l->own=1; l->you=G.siE-1;
  if(l->aB[siE]==l->EB){ l->own=1; l->you=siE-1;
    //while(l->own<=G.siE){ l->bit=1; l->ptn=l->TB;
    while(l->own<=siE){ l->bit=1; l->ptn=l->TB;
      while((l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit)){ l->bit<<=1; l->ptn>>=1; }
      if(l->aB[l->own]>l->bit){ return; } if(l->aB[l->own]<l->bit){ break; }
      l->own++; l->you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    //if(l->own>G.siE){ 
    if(l->own>siE){ 
			//C4[l->B1]++;
			l->C4[l->B1]++;
      return; }
  }
  //270度回転
  if(l->aB[l->B1]==l->TB){ l->own=1; l->ptn=l->TB>>1;
    //while(l->own<=G.siE){ l->bit=1; l->you=0;
    while(l->own<=siE){ l->bit=1; l->you=0;
      while((l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit)){ l->bit<<=1; l->you++; }
      if(l->aB[l->own]>l->bit){ return; } if(l->aB[l->own]<l->bit){ break; }
      l->own++; l->ptn>>=1;
    }
  }
  // G.C8[l->B1]++;
  //(*C8)++;
	//C8[l->B1]++;
	l->C8[l->B1]++;
}

void backTrack2(int y,int left,int down,int right,int bm,local *l){
  //配置可能フィールド
  //int bit=0;
  //int bm=l->msk&~(left|down|right); 
  bm=l->msk&~(left|down|right); 
  l->bit=0;
  //if(y==G.siE){
  if(y==siE){
    if(bm>0 && (bm&l->LM)==0){ //【枝刈り】最下段枝刈り
      l->aB[y]=bm;
      //対称解除法
      //symmetryOps_bm(l,C2,C4,C8);//対称解除法
      symmetryOps_bm(l);
    }
  }else{
    //【枝刈り】上部サイド枝刈り
    if(y<l->B1){             
      bm&=~l->SM; 
    //【枝刈り】下部サイド枝刈り
    }else if(y==l->B2) {     
      if((down&l->SM)==0){ return; }
      if((down&l->SM)!=l->SM){ bm&=l->SM; }
    }
    while(bm>0) {
      //最も下位の１ビットを抽出
      //bm^=l->aB[y]=bit=-bm&bm;
      bm^=l->aB[y]=l->bit=-bm&bm;
      //backTrack2(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
      backTrack2(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }
  }
}
void backTrack1(int y,int left,int down,int right,int bm,local *l){
  //int bit;
  //int bm=l->msk&~(left|down|right); 
  bm=l->msk&~(left|down|right); 
  l->bit=0;
  //if(y==G.siE) {
  if(y==siE) {
    if(bm>0){
      l->aB[y]=bm;
      //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
			//C8[l->B1]++;
			l->C8[l->B1]++;
    }
  }else{
    if(y<l->B1) {   
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      //bm|=2; 
      //bm^=2;
      bm&=~2; 
    }
    while(bm>0) {
      //最も下位の１ビットを抽出
      //bm^=l->aB[y]=bit=-bm&bm;
      bm^=l->aB[y]=l->bit=-bm&bm;
      //backTrack1(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
      backTrack1(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }
  } 
}
void *run(void *args){
  local *l=(local *)args;
  //int bit ;
  //l->bit=0 ; l->aB[0]=1; l->msk=(1<<G.si)-1; l->TB=1<<G.siE;
  l->bit=0 ; l->aB[0]=1; l->msk=(1<<si)-1; l->TB=1<<siE;
  //if(l->B1>1 && l->B1<G.siE) { // 最上段のクイーンが角にある場合の探索
  if(l->B1>1 && l->B1<siE) { // 最上段のクイーンが角にある場合の探索
    l->aB[1]=l->bit=(1<<l->B1);// 角にクイーンを配置 
    backTrack1(2,(2|l->bit)<<1,(1|l->bit),(l->bit>>1),0,l);//２行目から探索
  }
  l->EB=(l->TB>>l->B1);
  l->SM=l->LM=(l->TB|1);
  //if(l->B1>0&&l->B2<G.siE&&l->B1<l->B2){ // 最上段行のクイーンが角以外にある場合の探索 
  if(l->B1>0&&l->B2<siE&&l->B1<l->B2){ // 最上段行のクイーンが角以外にある場合の探索 
    for(int i=1; i<l->B1; i++){
      l->LM=l->LM|l->LM>>1|l->LM<<1;
    }
    l->aB[0]=l->bit=(1<<l->B1);
    backTrack2(1,l->bit<<1,l->bit,l->bit>>1,0,l);
    //l->EB>>=G.si;
    l->EB>>=si;
  }
  return 0;
}
void *NQueenThread(){
  //pthread_t pt[G.si];//スレッド childThread
  pthread_t pt[si];//スレッド childThread
  //local l[MAX];//構造体 local型 
  local l[si];//構造体 local型 
  //for(int B1=G.siE,B2=0;B2<G.siE;B1--,B2++){// B1から順にスレッドを生成しながら処理を分担する 
  for(int B1=siE,B2=0;B2<siE;B1--,B2++){// B1から順にスレッドを生成しながら処理を分担する 
    l[B1].B1=B1; l[B1].B2=B2; //B1 と B2を初期化
    //for(int j=0;j<G.siE;j++){ l[l->B1].aB[j]=j; } // aB[]の初期化
    for(int j=0;j<siE;j++){ l[l->B1].aB[j]=j; } // aB[]の初期化
	  l[B1].C2[B1]=l[B1].C4[B1]=l[B1].C8[B1]=0;	//カウンターの初期化
    //int iFbRet=pthread_create(&pt[B1],NULL,&run,&l[B1]);// チルドスレッドの生成
    int iFbRet=pthread_create(&pt[B1],NULL,&run,(void*)&l[B1]);// チルドスレッドの生成
    if(iFbRet>0){
      printf("[mainThread] pthread_create #%d: %d\n", l[B1].B1, iFbRet);
    }
  }
  //for(int B1=G.siE,B2=0;B2<G.siE;B1--,B2++){ 
  for(int B1=siE,B2=0;B2<siE;B1--,B2++){ 
    pthread_join(pt[B1],NULL); 
  }
  //for(int B1=G.siE,B2=0;B2<G.siE;B1--,B2++){//スレッド毎のカウンターを合計
  for(int B1=siE,B2=0;B2<siE;B1--,B2++){//スレッド毎のカウンターを合計
    //G.lTotal+=l[B1].C2[B1]*2+l[B1].C4[B1]*4+l[B1].C8[B1]*8;
    lTotal+=l[B1].C2[B1]*2+l[B1].C4[B1]*4+l[B1].C8[B1]*8;
    //G.lUnique+=l[B1].C2[B1]+l[B1].C4[B1]+l[B1].C8[B1]; 
    lUnique+=l[B1].C2[B1]+l[B1].C4[B1]+l[B1].C8[B1]; 
  }
  return 0;
}
void NQueen(){
  pthread_t pth;//スレッド変数
  int iFbRet = pthread_create(&pth, NULL, &NQueenThread, NULL);// メインスレッドの生成
  if(iFbRet>0){
    printf("[main] pthread_create: %d\n", iFbRet); //エラー出力デバッグ用
  }
  pthread_join(pth, NULL); //スレッドの終了を待つ
}
int main(void){
  int min=2;
  struct timeval t0;
  struct timeval t1;
  printf("%s\n"," N:        Total       Unique                 dd:hh:mm:ss.ms");
  for(int i=min;i<=MAX;i++){
    //G.si=i; G.siE=i-1; 
    si=i; siE=i-1; 
    //G.lTotal=G.lUnique=0;
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
    //printf("%2d:%16ld%17ld%12.4d:%02d:%02d.%02d\n", i,G.lTotal,G.lUnique,hh,mm,ss,ms); 
    printf("%2d:%16ld%17ld%12.2d:%02d:%02d:%02d.%02d\n", i,lTotal,lUnique,dd,hh,mm,ss,ms); 
  } 
	return 0;
}
