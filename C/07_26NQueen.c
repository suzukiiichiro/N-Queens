/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
<>26. CPU affinity
	論理CPUにスレッドを割り当てる

#if _GNU_SOURCE
#define _GNU_SOURCE
#include <sched.h> 
#include <unistd.h>
#include <sys/syscall.h>
#include <errno.h>
#define handle_error_en(en, msg) do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)
#endif


# run ソース部分：
#ifdef _GNU_SOURCE
  pthread_t thread = pthread_self();
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(l->B1, &cpuset);
  int s=pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (s != 0){ handle_error_en(s, "pthread_setaffinity_np"); }
  s=pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (s != 0){ handle_error_en(s, "pthread_getaffinity_np"); }
  //printf("pid:%10d#l->B1:%2d#cpuset:%d\n",thread,l->B1,&cpuset);
#endif

  実行結果  スレッドが立ってる
pid:139811194939136#l->B1: 7#cpuset:1419517552
pid:139811100808960#l->B1: 4#cpuset:1325387376
pid:139811084023552#l->B1: 3#cpuset:1308601968
pid:139811109201664#l->B1: 5#cpuset:1333780080
pid:139811203331840#l->B1: 8#cpuset:1427910256
pid:139811117594368#l->B1: 6#cpuset:1342172784
pid:139811067238144#l->B1: 2#cpuset:1291816560
pid:139811050452736#l->B1: 1#cpuset:1275031152
16:        14772512          1846955          00:00:00:00.56
pid:139811228509952#l->B1:12#cpuset:1453088368
pid:139811084023552#l->B1:14#cpuset:1308601968
pid:139811211724544#l->B1:10#cpuset:1436302960
pid:139811067238144#l->B1:15#cpuset:1291816560
pid:139811220117248#l->B1:11#cpuset:1444695664
pid:139811050452736#l->B1:16#cpuset:1275031152
pid:139811236902656#l->B1:13#cpuset:1461481072
pid:139811203331840#l->B1: 9#cpuset:1427910256
pid:139811109201664#l->B1: 6#cpuset:1333780080
pid:139811058845440#l->B1: 2#cpuset:1283423856
pid:139811117594368#l->B1: 7#cpuset:1342172784
pid:139811042060032#l->B1: 1#cpuset:1266638448
pid:139811092416256#l->B1: 4#cpuset:1316994672
pid:139811100808960#l->B1: 5#cpuset:1325387376
pid:139811075630848#l->B1: 3#cpuset:1300209264
pid:139811194939136#l->B1: 8#cpuset:1419517552
17:        95815104         11977939          00:00:00:03.63

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
16:        14772512          1846955          00:00:00:00.56
17:        95815104         11977939          00:00:00:03.63
18:       666090624         83263591          00:00:00:24.75
19:      4968057848        621012754          00:00:02:58.81
20:     39029188884       4878666808          00:00:22:23.29
21:    314666222712      39333324973          00:02:55:11.88
22:   2691008701644     336376244042          01:00:59:21.78



僕のMacBookProでも凄く速い！

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
15:         2279184           285053          00:00:00:00.11
16:        14772512          1846955          00:00:00:00.67
17:        95815104         11977939          00:00:00:04.41
18:       666090624         83263591          00:00:00:33.88
19:      4968057848        621012754          00:00:04:22.25
20:     39029188884       4878666808          00:00:34:26.79
21:    314666222712      39333324973          00:04:44:55.30
22:   2691008701644     336376244042          01:17:12:40.88

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

#define MAX 27

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

void symmetryOps_bm(local *l){
  l->own=l->ptn=l->you=l->bit=0;
  //90度回転
  if(l->aB[l->B2]==1){ l->own=1; l->ptn=2;
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
  if(l->aB[siE]==l->EB){ l->own=1; l->you=siE-1;
    while(l->own<=siE){ l->bit=1; l->ptn=l->TB;
      while((l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit)){ l->bit<<=1; l->ptn>>=1; }
      if(l->aB[l->own]>l->bit){ return; } if(l->aB[l->own]<l->bit){ break; }
      l->own++; l->you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(l->own>siE){ 
			l->C4[l->B1]++;
      return; }
  }
  //270度回転
  if(l->aB[l->B1]==l->TB){ l->own=1; l->ptn=l->TB>>1;
    while(l->own<=siE){ l->bit=1; l->you=0;
      while((l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit)){ l->bit<<=1; l->you++; }
      if(l->aB[l->own]>l->bit){ return; } if(l->aB[l->own]<l->bit){ break; }
      l->own++; l->ptn>>=1;
    }
  }
	l->C8[l->B1]++;
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
    for(int i=1; i<l->B1; i++){
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
  for(int B1=siE,B2=0;B2<siE;B1--,B2++){// B1から順にスレッドを生成しながら処理を分担する 
    l[B1].B1=B1; l[B1].B2=B2; //B1 と B2を初期化
    for(int j=0;j<siE;j++){ l[l->B1].aB[j]=j; } // aB[]の初期化
	  l[B1].C2[B1]=l[B1].C4[B1]=l[B1].C8[B1]=0;	//カウンターの初期化
    int iFbRet=pthread_create(&pt[B1],NULL,&run,(void*)&l[B1]);// チルドスレッドの生成
    if(iFbRet>0){
      printf("[mainThread] pthread_create #%d: %d\n", l[B1].B1, iFbRet);
    }
  }
  for(int B1=siE,B2=0;B2<siE;B1--,B2++){ 
    pthread_join(pt[B1],NULL); 
  }
  for(int B1=siE,B2=0;B2<siE;B1--,B2++){//スレッド毎のカウンターを合計
    lTotal+=l[B1].C2[B1]*2+l[B1].C4[B1]*4+l[B1].C8[B1]*8;
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
