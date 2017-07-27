/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
<>27. デバッグ

*/

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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <unistd.h>

#define MAX 27
#define DEBUG 1

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
  int B1;
  int B2;
  int TB;
  int EB;
  int msk;
  int SM;
  int LM;
  int aB[MAX]; 
  long C2[MAX];
  long C4[MAX];
  long C8[MAX];
}local ;

void symmetryOps_bm(local *l);
void backTrack2(int y,int left,int down,int right,int bm,local *l);
void backTrack1(int y,int left,int down,int right,int bm,local *l);
void *run(void *args);
void *NQueenThread();
void NQueen();
//Debug


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
    printf("\rN:%2d C2[%c] C4[ ] C8[ ] C8BT[ ] B1[%2d] B2[%2d]",si,spc[l->C2[l->B1]%spl],l->B1,l->B2);
  }
  else if(i==4){
    printf("\rN:%2d C2[ ] C4[%c] C8[ ] C8BT[ ] B1[%2d] B2[%2d]",si,spc[l->C4[l->B1]%spl],l->B1,l->B2);
  }
  else if(i==8){
    printf("\rN:%2d C2[ ] C4[ ] C8[%c] C8BT[ ] B1[%2d] B2[%2d]",si,spc[l->C8[l->B1]%spl],l->B1,l->B2);
  }
  else if(i==82){ 
    printf("\rN:%2d C2[ ] C4[ ] C8[ ] C8BT[%c] B1[%2d] B2[%2d]",si,spc[l->C8[l->B1]%spl],l->B1,l->B2);
  }
  printf("\033[G");
}
void symmetryOps_bm(local *l){
  l->own=l->ptn=l->you=l->bit=0;
  //90度回転
  if(l->aB[l->B2]==1){ l->own=1; l->ptn=2;
    while(l->own<=siE){ l->bit=1; l->you=siE;
      while((l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit)){ l->bit<<=1; l->you--; }
      if(l->aB[l->own]>l->bit){ return; } if(l->aB[l->own]<l->bit){ break; }
      l->own++; l->ptn<<=1; }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(l->own>siE){ 
			l->C2[l->B1]++;
      if(DEBUG>0) thMonitor(l,2);
      return ; } }
  //180度回転
  if(l->aB[siE]==l->EB){ l->own=1; l->you=siE-1;
    while(l->own<=siE){ l->bit=1; l->ptn=l->TB;
      while((l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit)){ l->bit<<=1; l->ptn>>=1; }
      if(l->aB[l->own]>l->bit){ return; } if(l->aB[l->own]<l->bit){ break; }
      l->own++; l->you--; }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(l->own>siE){ 
			l->C4[l->B1]++;
      if(DEBUG>0) thMonitor(l,4); 
      return; } }
  //270度回転
  if(l->aB[l->B1]==l->TB){ l->own=1; l->ptn=l->TB>>1;
    while(l->own<=siE){ l->bit=1; l->you=0;
      while((l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit)){ l->bit<<=1; l->you++; }
      if(l->aB[l->own]>l->bit){ return; } if(l->aB[l->own]<l->bit){ break; }
      l->own++; l->ptn>>=1;
    }
  }
	l->C8[l->B1]++;
  if(DEBUG>0) thMonitor(l,8); 
}

void backTrack2(int y,int left,int down,int right,int bm,local *l){
  //配置可能フィールド
  bm=l->msk&~(left|down|right); 
  l->bit=0;
  if(y==siE){
    if(bm>0 && (bm&l->LM)==0){ //【枝刈り】最下段枝刈り
      l->aB[y]=bm;
      //対称解除法
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
      bm^=l->aB[y]=l->bit=-bm&bm;
      backTrack2(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }
  }
}
void backTrack1(int y,int left,int down,int right,int bm,local *l){
  bm=l->msk&~(left|down|right); 
  l->bit=0;
  if(y==siE) {
    if(bm>0){
      l->aB[y]=bm;
      //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
			l->C8[l->B1]++;
      if(DEBUG>0) thMonitor(l,82);
    }
  }else{
    if(y<l->B1) {   
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      bm&=~2; 
    }
    while(bm>0) {
      //最も下位の１ビットを抽出
      bm^=l->aB[y]=l->bit=-bm&bm;
      backTrack1(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }
  } 
}
void *run(void *args){
  local *l=(local *)args;
#ifdef _GNU_SOURCE
  pthread_t thread = pthread_self();
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(l->B1, &cpuset);
  int s=pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (s != 0){ handle_error_en(s, "pthread_setaffinity_np"); }
  s=pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (s != 0){ handle_error_en(s, "pthread_getaffinity_np"); }
  printf("pid:%10ld#l->B1:%2d#cpuset:%p\n",thread,l->B1,&cpuset);
#endif

  l->bit=0 ; l->aB[0]=1; l->msk=(1<<si)-1; l->TB=1<<siE;
  if(l->B1>1 && l->B1<siE) { // 最上段のクイーンが角にある場合の探索
    l->aB[1]=l->bit=(1<<l->B1);// 角にクイーンを配置 
    backTrack1(2,(2|l->bit)<<1,(1|l->bit),(l->bit>>1),0,l);//２行目から探索
  }
  l->EB=(l->TB>>l->B1);
  l->SM=l->LM=(l->TB|1);
  if(l->B1>0&&l->B2<siE&&l->B1<l->B2){ // 最上段行のクイーンが角以外にある場合の探索 
    int i;
    for(i=1;i<l->B1;i++){
      l->LM=l->LM|l->LM>>1|l->LM<<1;
    }
    l->aB[0]=l->bit=(1<<l->B1);
    backTrack2(1,l->bit<<1,l->bit,l->bit>>1,0,l);
    l->EB>>=si;
  }
  return 0;
}
void *NQueenThread(){
  pthread_t pt[si];//スレッド childThread
  local l[si];//構造体 local型 
  //for(int B1=siE,B2=0;B2<siE;B1--,B2++){// B1から順にスレッドを生成しながら処理を分担する 
  int B1; int B2=siE; int j; int iFbRet;
  for(B1=0;B1<si;B1++){
    l[B1].B1=B1; l[B1].B2=B2; //B1 と B2を初期化
    for(j=0;j<siE;j++){ l[l->B1].aB[j]=j; } // aB[]の初期化
	  l[B1].C2[B1]=l[B1].C4[B1]=l[B1].C8[B1]=0;	//カウンターの初期化
    iFbRet=pthread_create(&pt[B1],NULL,&run,(void*)&l[B1]);// チルドスレッドの生成
    if(DEBUG>0){
      printf("\r\033[2K[Thread] pthread_create #%d: %d\n", l[B1].B1, iFbRet);
    }
    B2--;
  }
  //for(int B1=siE,B2=0;B2<siE;B1--,B2++){ 
  for(B1=0;B1<si;B1++){ 
    pthread_join(pt[B1],NULL); 
  }
  for(B1=0;B1<si;B1++){ 
    pthread_detach(pt[B1]);
  }
  //for(int B1=siE,B2=0;B2<siE;B1--,B2++){//スレッド毎のカウンターを合計
  for(B1=0;B1<si;B1++){//スレッド毎のカウンターを合計
    lTotal+=l[B1].C2[B1]*2+l[B1].C4[B1]*4+l[B1].C8[B1]*8;
    lUnique+=l[B1].C2[B1]+l[B1].C4[B1]+l[B1].C8[B1]; 
  }
  return 0;
}
void NQueen(){
  pthread_t pth;//スレッド変数
  int iFbRet=pthread_create(&pth, NULL, &NQueenThread, NULL);// メインスレッドの生成
  if(DEBUG>0){
    printf("\r\033[2K[main] pthread_create: %d\n", iFbRet); //エラー出力デバッグ用
  }
  pthread_join(pth, NULL); //スレッドの終了を待つ
  pthread_detach(pth);
}
int main(void){
  int min=2;
  struct timeval t0;
  struct timeval t1;
  printf("%s\n"," N:        Total       Unique                 dd:hh:mm:ss.ms");
  int i;
  for(i=min;i<=MAX;i++){
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
	return 0;
}
