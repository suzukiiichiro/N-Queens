
// $ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC13NR.c && ./a.out [-c|-r]
//
// 【注意】
// -pthread オプションが必要です
// Eclipse では -O3を追加しないと計算結果が合いません
// Eclipse では、threadを追加しないと高速処理になりません。
//
/**
bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC13NR.c && ./a.out -r
１３．CPUR 再帰 並列処理 pthread
 N:           Total           Unique          dd:hh:mm:ss.ms
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
16:        14772512          1846955          00:00:00:00.65
17:        95815104         11977939          00:00:00:04.33 


bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC13NR.c && ./a.out -c
１３．CPU 非再帰 並列処理 pthread
 N:           Total           Unique          dd:hh:mm:ss.ms
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
16:        14772512          1846955          00:00:00:00.62
17:        95815104         11977939          00:00:00:04.15
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <stdbool.h>
#include <pthread.h>
//
#define MAX 27
//

/** 非再帰 再帰で実行 */
int CPU,CPUR=0;

//
//変数宣言
// pthreadはパラメータを１つしか渡せないので構造体に格納
//グローバル構造体
typedef struct {
  int size;
  int sizeE;
  long lTOTAL,lUNIQUE;
}GCLASS, *GClass;
GCLASS G;
//ローカル構造体
typedef struct{
  int BOUND1,BOUND2,TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
  int mask;
  int aBoard[MAX];
  long COUNT2[MAX],COUNT4[MAX],COUNT8[MAX];
}local ;
//関数宣言
void symmetryOps(local *l);
void backTrack2_NR(int y,int left,int down,int right,local *l);
void backTrack1_NR(int y,int left,int down,int right,local *l);
void backTrack2(int y,int left,int down,int right,local *l);
void backTrack1(int y,int left,int down,int right,local *l);
void *run(void *args);
void *NQueenThread();
void NQueen();
//
void symmetryOps(local *l){
  int own,ptn,you,bit;
  //90度回転
  if(l->aBoard[l->BOUND2]==1){ own=1; ptn=2;
    while(own<=G.sizeE){ bit=1; you=G.sizeE;
      while((l->aBoard[you]!=ptn)&&(l->aBoard[own]>=bit)){ bit<<=1; you--; }
      if(l->aBoard[own]>bit){ return; } if(l->aBoard[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>G.sizeE){ l->COUNT2[l->BOUND1]++; return; }
  }
  //180度回転
  if(l->aBoard[G.sizeE]==l->ENDBIT){ own=1; you=G.sizeE-1;
    while(own<=G.sizeE){ bit=1; ptn=l->TOPBIT;
      while((l->aBoard[you]!=ptn)&&(l->aBoard[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(l->aBoard[own]>bit){ return; } if(l->aBoard[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>G.sizeE){ l->COUNT4[l->BOUND1]++; return; }
  }
  //270度回転
  if(l->aBoard[l->BOUND1]==l->TOPBIT){ own=1; ptn=l->TOPBIT>>1;
    while(own<=G.sizeE){ bit=1; you=0;
      while((l->aBoard[you]!=ptn)&&(l->aBoard[own]>=bit)){ bit<<=1; you++; }
      if(l->aBoard[own]>bit){ return; } if(l->aBoard[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  l->COUNT8[l->BOUND1]++;
}
//CPU 非再帰版 backTrack2
void backTrack2_NR(int row,int left,int down,int right,local *l){
  int bitmap,bit;
  int b[100], *p=b;
mais1:bitmap=l->mask&~(left|down|right);
  // 【枝刈り】
  if(row==G.sizeE){
    if(bitmap){
      //【枝刈り】 最下段枝刈り
      if((bitmap&l->LASTMASK)==0){
        l->aBoard[row]=bitmap;
        symmetryOps(l);
      }
    }
  }else{
    //【枝刈り】上部サイド枝刈り
    if(row<l->BOUND1){
      bitmap&=~l->SIDEMASK;
      //【枝刈り】下部サイド枝刈り
    }else if(row==l->BOUND2){
      if(!(down&l->SIDEMASK))
        goto volta;
      if((down&l->SIDEMASK)!=l->SIDEMASK)
        bitmap&=l->SIDEMASK;
    }
    if(bitmap){
outro:bitmap^=l->aBoard[row]=bit=-bitmap&bitmap;
  if(bitmap){
    *p++=left;
    *p++=down;
    *p++=right;
  }
  *p++=bitmap;
  row++;
  left=(left|bit)<<1;
  down=down|bit;
  right=(right|bit)>>1;
  goto mais1;
  //Backtrack2(y+1, (left | bit)<<1, down | bit, (right | bit)>>1);
volta:if(p<=b)
    return;
  row--;
  bitmap=*--p;
  if(bitmap){
    right=*--p;
    down=*--p;
    left=*--p;
    goto outro;
  }else{
    goto volta;
  }
    }
  }
  goto volta;
}
//CPU 非再帰版 backTrack
void backTrack1_NR(int row,int left,int down,int right,local *l){
  int bitmap,bit;
  int b[100], *p=b;
b1mais1:bitmap=l->mask&~(left|down|right);
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
  if(row==G.sizeE){
    if(bitmap){
      l->aBoard[row]=bitmap;
      l->COUNT8[l->BOUND1]++;
    }
  }else{
    //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
    // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
    if(row<l->BOUND1) {
      bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
    if(bitmap){
b1outro:bitmap^=l->aBoard[row]=bit=-bitmap&bitmap;
  if(bitmap){
    *p++=left;
    *p++=down;
    *p++=right;
  }
  *p++=bitmap;
  row++;
  left=(left|bit)<<1;
  down=down|bit;
  right=(right|bit)>>1;
  goto b1mais1;
  //Backtrack1(y+1, (left | bit)<<1, down | bit, (right | bit)>>1);
b1volta:if(p<=b)
    return;
  row--;
  bitmap=*--p;
  if(bitmap){
    right=*--p;
    down=*--p;
    left=*--p;
    goto b1outro;
  }else{
    goto b1volta;
  }
    }
  }
  goto b1volta;
}
//
void backTrack2(int row,int left,int down,int right,local *l){
  int bit;
  int bitmap=l->mask&~(left|down|right);
  if(row==G.sizeE){ 								// 【枝刈り】
    if(bitmap){
      if((bitmap&l->LASTMASK)==0){ 	//【枝刈り】 最下段枝刈り
        l->aBoard[row]=bitmap;
        symmetryOps(l);
      }
    }
  }else{
    if(row<l->BOUND1){             	//【枝刈り】上部サイド枝刈り
      bitmap&=~l->SIDEMASK;
    }else if(row==l->BOUND2) {     	//【枝刈り】下部サイド枝刈り
      if((down&l->SIDEMASK)==0){ return; }
      if((down&l->SIDEMASK)!=l->SIDEMASK){ bitmap&=l->SIDEMASK; }
    }
    while(bitmap){
      bitmap^=l->aBoard[row]=bit=(-bitmap&bitmap);
      backTrack2(row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
}
//
void backTrack1(int row,int left,int down,int right,local *l){
  int bit;
  int bitmap=l->mask&~(left|down|right);
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
  if(row==G.sizeE) {
    if(bitmap){
      l->aBoard[row]=bitmap;
      l->COUNT8[l->BOUND1]++;
    }
  }else{
    //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
    // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
    if(row<l->BOUND1) {
      bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
    while(bitmap){
      bitmap^=l->aBoard[row]=bit=(-bitmap&bitmap);
      backTrack1(row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
}
//
void *run(void *args){
  local *l=(local *)args;
  int bit;
  l->aBoard[0]=1;
  l->TOPBIT=1<<(G.sizeE);
  l->mask=(1<<G.size)-1;
  // 最上段のクイーンが角にある場合の探索
  if(l->BOUND1>1 && l->BOUND1<G.sizeE) {
    if(l->BOUND1<G.sizeE) {
      // 角にクイーンを配置
      l->aBoard[1]=bit=(1<<l->BOUND1);
      //２行目から探索
      //  backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1),l);
      if(CPU==1){
        //非再帰
        backTrack1_NR(2,(2|bit)<<1,(1|bit),(bit>>1),l);
      }else{
        //再帰
        backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1),l);
      }
    }
  }
  l->ENDBIT=(l->TOPBIT>>l->BOUND1);
  l->SIDEMASK=l->LASTMASK=(l->TOPBIT|1);
  /* 最上段行のクイーンが角以外にある場合の探索
     ユニーク解に対する左右対称解を予め削除するには、
     左半分だけにクイーンを配置するようにすればよい */
  if(l->BOUND1>0&&l->BOUND2<G.sizeE&&l->BOUND1<l->BOUND2){
    for(int i=1; i<l->BOUND1; i++){
      l->LASTMASK=l->LASTMASK|l->LASTMASK>>1|l->LASTMASK<<1;
    }
    if(l->BOUND1<l->BOUND2) {
      l->aBoard[0]=bit=(1<<l->BOUND1);
      //backTrack2(1,bit<<1,bit,bit>>1,l);
      if(CPU==1){
        //非再帰
        backTrack2_NR(1,bit<<1,bit,bit>>1,l);
      }else{
        //再帰
        backTrack2(1,bit<<1,bit,bit>>1,l);
      }
    }
    l->ENDBIT>>=G.size;
  }
  return 0;   //*run()の場合はreturn 0;が必要
}
//
void *NQueenThread(){
  local l[MAX];                //構造体 local型
  pthread_t pt[G.size];                 //スレッド childThread
  for(int BOUND1=G.sizeE,BOUND2=0;BOUND2<G.sizeE;BOUND1--,BOUND2++){
    l[BOUND1].BOUND1=BOUND1; l[BOUND1].BOUND2=BOUND2;         //B1 と B2を初期化
    //初期化は不要です
    // for(int j=0;j<G.size;j++){ l[l->BOUND1].aBoard[j]=j; } // aB[]の初期化
    l[BOUND1].COUNT2[BOUND1]=l[BOUND1].COUNT4[BOUND1]=l[BOUND1].COUNT8[BOUND1]=0;//カウンターの初期化
    // チルドスレッドの生成
    int iFbRet=pthread_create(&pt[BOUND1],NULL,&run,&l[BOUND1]);
    if(iFbRet>0){
      printf("[mainThread] pthread_create #%d: %d\n", l[BOUND1].BOUND1, iFbRet);
    }
  }
  for(int BOUND1=G.sizeE,BOUND2=0;BOUND2<G.sizeE;BOUND1--,BOUND2++){
    pthread_join(pt[BOUND1],NULL);
  }
  //スレッド毎のカウンターを合計
  for(int BOUND1=G.sizeE,BOUND2=0;BOUND2<G.sizeE;BOUND1--,BOUND2++){
    G.lTOTAL+=l[BOUND1].COUNT2[BOUND1]*2+l[BOUND1].COUNT4[BOUND1]*4+l[BOUND1].COUNT8[BOUND1]*8;
    G.lUNIQUE+=l[BOUND1].COUNT2[BOUND1]+l[BOUND1].COUNT4[BOUND1]+l[BOUND1].COUNT8[BOUND1];
  }
  return 0;
}
//
void NQueen(){
  pthread_t pth;  //スレッド変数
  // メインスレッドの生成
  int iFbRet = pthread_create(&pth, NULL, &NQueenThread, NULL);
  if(iFbRet>0){
    printf("[main] pthread_create: %d\n", iFbRet); //エラー出力デバッグ用
  }
  pthread_join(pth,NULL); /* いちいちjoinをする */
}
//メインメソッド
int main(int argc,char** argv) {
  /** 出力と実行 */
  // 不要となります
  // bool cpu=false,cpur=false;
  int argstart=2;
  /** 起動パラメータの処理 */
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){CPU=1;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){CPUR=1;}
    else{ CPUR=1;}
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g]\n",argv[0]);
    printf("  -c: CPU Without recursion\n");
    printf("  -r: CPUR Recursion\n");
  }
  if(CPU){
    printf("\n\n１３．CPU 非再帰 並列処理 pthread\n");
  }else if(CPUR){
    printf("\n\n１３．CPUR 再帰 並列処理 pthread\n");
  }
  printf("%s\n"," N:           Total           Unique          dd:hh:mm:ss.ms");
  struct timeval t0;
  struct timeval t1;
  int min=4; int targetN=17;
  for(int i=min;i<=targetN;i++){
    G.size=i; G.sizeE=i-1; //初期化
    G.lTOTAL=G.lUNIQUE=0;
    gettimeofday(&t0, NULL);
    /**
      aBoard配列の初期化は
      void *NQueenThread()で行います。
      が、実際不要です。(※)
      */
    NQueen();
    gettimeofday(&t1, NULL);
    int ss;int ms;int dd;
    if(t1.tv_usec<t0.tv_usec) {
      dd=(t1.tv_sec-t0.tv_sec-1)/86400;
      ss=(t1.tv_sec-t0.tv_sec-1)%86400;
      ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
    }else {
      dd=(t1.tv_sec-t0.tv_sec)/86400;
      ss=(t1.tv_sec-t0.tv_sec)%86400;
      ms=(t1.tv_usec-t0.tv_usec+500)/10000;
    }
    int hh=ss/3600;
    int mm=(ss-hh*3600)/60;
    ss%=60;
    printf("%2d:%16ld%17ld%12.2d:%02d:%02d:%02d.%02d\n", i,G.lTOTAL,G.lUNIQUE,dd,hh,mm,ss,ms);
  }
}
