/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 <>26．マルチスレッド再び 							        NQueen25() N17=03.58

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
15:         2279184           285053          00:00:00:00.09
16:        14772512          1846955          00:00:00:00.55
17:        95815104         11977939          00:00:00:03.58
18:       666090624         83263591          00:00:00:24.52


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
#define DEBUG 0

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
  long C2[MAX][2];//構造体の中の配列を活かすと   N=17: 04.33
  long C4[MAX][2];
  long C8[MAX][2];
//  long C2; // 構造体の中の配列をなくすとN=17: 05.24
//  long C4;
//  long C8;
  int BK;
}local ;

// long C2[MAX]; //グローバル環境に置くと N=17: 08.04
// long C4[MAX];
// long C8[MAX];

void symmetryOps_bm(local *l);
void backTrack2(int y,int left,int down,int right,int bm,local *l2);
void backTrack1(int y,int left,int down,int right,int bm,local *l);
void *run(void *args);
void *run2(void *args);
void *NQueenThread();
void NQueen();

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
#endif

void thMonitor(local *l,int i){
  printf("\033[G");
  if(i==2){
    printf("\rN:%2d C2[%c] C4[ ] C8[ ] C8BT[ ] B1[%2d] B2[%2d]",si,spc[l->C2[l->B1][l->BK]%spl],l->B1,l->B2);
  }
  else if(i==4){
    printf("\rN:%2d C2[ ] C4[%c] C8[ ] C8BT[ ] B1[%2d] B2[%2d]",si,spc[l->C4[l->B1][l->BK]%spl],l->B1,l->B2);
  }
  else if(i==8){
    printf("\rN:%2d C2[ ] C4[ ] C8[%c] C8BT[ ] B1[%2d] B2[%2d]",si,spc[l->C8[l->B1][l->BK]%spl],l->B1,l->B2);
  }
  else if(i==82){ 
    printf("\rN:%2d C2[ ] C4[ ] C8[ ] C8BT[%c] B1[%2d] B2[%2d]",si,spc[l->C8[l->B1][l->BK]%spl],l->B1,l->B2);
  }
  printf("\033[G");
/*
  printf("\n");
  for (int y=0;y<si;y++) {
    for (l->bit=l->TB; l->bit; l->bit>>=1){
        char c;
        if(l->aB[y]==l->bit){
          c='Q';
        }else{
          c='-';
        }
        putchar(c);
    }
    printf("|\n");
  }
  printf("\n\n");
*/
}
void symmetryOps_bm(local *l){
  l->own=l->ptn=l->you=l->bit=0;
  l->C8[l->B1][l->BK]++;
  if(DEBUG>0) thMonitor(l,8); 
  //90度回転
  if(l->aB[l->B2]==1){ 
    //l->own=1; l->ptn=2;
    //while(l->own<=siE){ 
    for(l->own=1,l->ptn=2;l->own<=siE;l->own++,l->ptn<<=1){ 
      //l->bit=1; l->you=siE;
      //while((l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit)){
      for(l->bit=1,l->you=siE;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->you--){}
      //{
      //   l->bit<<=1; l->you--; 
      //}
      if(l->aB[l->own]>l->bit){ 
        l->C8[l->B1][l->BK]--; 
        return; 
      }else if(l->aB[l->own]<l->bit){ 
        break; 
      }
      //l->own++; l->ptn<<=1; 
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(l->own>siE){ 
      l->C2[l->B1][l->BK]++;
      l->C8[l->B1][l->BK]--;
      if(DEBUG>0) thMonitor(l,2);
      return ; 
    } 
  }
  //180度回転
  if(l->aB[siE]==l->EB){ 
    //l->own=1; l->you=siE-1;
    //while(l->own<=siE){ 
    for(l->own=1,l->you=siE-1;l->own<=siE;l->own++,l->you--){ 
      //l->bit=1; l->ptn=l->TB;
      //while((l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit)){ 
      for(l->bit=1,l->ptn=l->TB;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->ptn>>=1){}
      //l->bit<<=1; l->ptn>>=1; 
      //}
      if(l->aB[l->own]>l->bit){ 
        l->C8[l->B1][l->BK]--; 
        return; 
      } 
      //if(l->aB[l->own]<l->bit){ break; }
      else if(l->aB[l->own]<l->bit){ 
        break; 
      }
      //l->own++; l->you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(l->own>siE){ 
      l->C4[l->B1][l->BK]++;
      l->C8[l->B1][l->BK]--;
      if(DEBUG>0) thMonitor(l,4); 
      return; 
    } 
  }
  //270度回転
  if(l->aB[l->B1]==l->TB){ 
    //l->own=1; l->ptn=l->TB>>1;
    //while(l->own<=siE){ 
    for(l->own=1,l->ptn=l->TB>>1;l->own<=siE;l->own++,l->ptn>>=1){ 
      //l->bit=1; l->you=0;
      //while((l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit)){ 
      for(l->bit=1,l->you=0;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->you++){}
      // { 
      //   l->bit<<=1; l->you++; 
      // }
      if(l->aB[l->own]>l->bit){ 
        l->C8[l->B1][l->BK]--; 
        return; 
      } 
      //if(l->aB[l->own]<l->bit){ break; }
      else if(l->aB[l->own]<l->bit){ 
        break; 
      }
      //l->own++; l->ptn>>=1;
    }
  }
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
      if((down&l->SM)==0){ 
        return; 
      }
      if((down&l->SM)!=l->SM){ 
        bm&=l->SM; 
      }
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
      l->C8[l->B1][l->BK]++;
      if(DEBUG>0) thMonitor(l,82);
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
  l->BK=0;
  //if(l->B1>1 && l->B1<G.siE) { // 最上段のクイーンが角にある場合の探索
  if(l->B1>1 && l->B1<siE) { // 最上段のクイーンが角にある場合の探索
    l->aB[1]=l->bit=(1<<l->B1);// 角にクイーンを配置 
    backTrack1(2,(2|l->bit)<<1,(1|l->bit),(l->bit>>1),0,l);//２行目から探索
  }
  return 0;
}
void *run2(void *args){
  local *l=(local *)args;
  //int bit ;
  //l->bit=0 ; l->aB[0]=1; l->msk=(1<<G.si)-1; l->TB=1<<G.siE;
  l->bit=0 ; l->aB[0]=1; l->msk=(1<<si)-1; l->TB=1<<siE;
  l->BK=1;
  l->EB=(l->TB>>l->B1);
  l->SM=l->LM=(l->TB|1);
  //if(l->B1>0&&l->B2<G.siE&&l->B1<l->B2){ // 最上段行のクイーンが角以外にある場合の探索 
  if(l->B1>0 && l->B2<siE && l->B1<l->B2){ // 最上段行のクイーンが角以外にある場合の探索 
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
  pthread_t pt[si*2];//スレッド childThread
  //local l[MAX];//構造体 local型 
  local l[si];//構造体 local型 
  local l2[si];//構造体 local型 
  //for(int B1=G.siE,B2=0;B2<G.siE;B1--,B2++){// B1から順にスレッドを生成しながら処理を分担する 
  for(int B1=1,B2=siE-1;B1<siE;B1++,B2--){
    l[B1].B1=l2[B1].B1=B1; l[B1].B2=l2[B1].B2=B2; //B1 と B2を初期化
    //l[B1].B1=B1; l[B1].B2=B2; //B1 と B2を初期化
    //l2[B1].B1=B1; l2[B1].B2=B2; //B1 と B2を初期化
    for(int j=0;j<siE;j++){ 
      l[B1].aB[j]=l2[B1].aB[j]=j; // aB[]の初期化
      //l[B1].aB[j]=j; // aB[]の初期化
      //l2[B1].aB[j]=j; // aB[]の初期化
    } 
    l[B1].C2[B1][0]=l[B1].C4[B1][0]=l[B1].C8[B1][0]=0;	//カウンターの初期化
    l2[B1].C2[B1][1]=l2[B1].C4[B1][1]=l2[B1].C8[B1][1]=0;	//カウンターの初期化
    pthread_create(&pt[B1],NULL,&run,(void*)&l[B1]);// チルドスレッドの生成
    //int iFbRet=pthread_create(&pt[B1],NULL,&run,(void*)&l[B1]);// チルドスレッドの生成
    //if(iFbRet>0){
    //  printf("[Thread] pthread_create #%d: %d\n", l[B1].B1, iFbRet);
    //}
    pthread_create(&pt[si+B1],NULL,&run2,(void*)&l2[B1]);// チルドスレッドの生成
    //int iFbRet2=pthread_create(&pt[si+B1],NULL,&run2,(void*)&l2[B1]);// チルドスレッドの生成
    //if(iFbRet2>0){
    //  printf("[Thread] pthread_create #%d: %d\n", l2[B1].B1, iFbRet2);
    //}
  }
  for(int B1=1;B1<siE;B1++){ 
    pthread_join(pt[B1],NULL); 
  //}
  //for(int B1=1;B1<siE;B1++){ 
    pthread_join(pt[si+B1],NULL); 
  //}
  //for(int B1=1;B1<siE;B1++){ 
    pthread_detach(pt[B1]);
  //}
  //for(int B1=1;B1<siE;B1++){ 
    pthread_detach(pt[si+B1]);
  //}
  //for(int B1=1;B1<siE;B1++){//スレッド毎のカウンターを合計
    lTotal+=l[B1].C2[B1][0]*2+l[B1].C4[B1][0]*4+l[B1].C8[B1][0]*8;
    lUnique+=l[B1].C2[B1][0]+l[B1].C4[B1][0]+l[B1].C8[B1][0]; 
    lTotal+=l2[B1].C2[B1][1]*2+l2[B1].C4[B1][1]*4+l2[B1].C8[B1][1]*8;
    lUnique+=l2[B1].C2[B1][1]+l2[B1].C4[B1][1]+l2[B1].C8[B1][1]; 
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
