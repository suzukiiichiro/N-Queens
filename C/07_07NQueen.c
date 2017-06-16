/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
  Java版 N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_N-Queen
  Bash版 N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_Bash
  Lua版  N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_Lua
 
  ステップバイステップでＮ−クイーン問題を最適化
   １．ブルートフォース（力まかせ探索） NQueen01()
   ２．配置フラグ（制約テスト高速化）   NQueen02()
   ３．バックトラック                   NQueen03() N16: 1:07
   ４．対称解除法(回転と斜軸）          NQueen04() N16: 1:09
   ５．枝刈りと最適化                   NQueen05() N16: 0:18
   ６．ビットマップ                     NQueen06() N16: 0:13
 <>７．ビットマップ+対称解除法          NQueen07() N16: 0:21
   ８．ビットマップ+クイーンの場所で分岐NQueen08() 
   ９．ビットマップ+枝刈りと最適化      NQueen09() 
   10．もっとビットマップ               NQueen10()
   11．マルチスレッド(構造体)           NQueen11() 
   12．マルチスレッド(pthread)          NQueen12()
   13．マルチスレッド(排他処理)         NQueen13()

  実行結果
 N:        Total       Unique        dd:hh:mm:ss
 2:            0               0      0 00:00:00
 3:            0               0      0 00:00:00
 4:            2               1      0 00:00:00
 5:           10               2      0 00:00:00
 6:            4               1      0 00:00:00
 7:           40               6      0 00:00:00
 8:           92              12      0 00:00:00
 9:          352              46      0 00:00:00
10:          724              92      0 00:00:00
11:         2680             341      0 00:00:00
12:        14200            1787      0 00:00:00
13:        73712            9233      0 00:00:00
14:       365596           45752      0 00:00:00
15:      2279184          285053      0 00:00:03
16:     14772512         1846955      0 00:00:20
 */

#include<stdio.h>
#include<time.h>
#include <math.h>

#define MAXSIZE 27

int lTotal=1 ; //合計解
int lUnique=0; //ユニーク解
int iSize;     //Ｎ
int aBoard[MAXSIZE];  //チェス盤の横一列
int aTrial[MAXSIZE];
int aScratch[MAXSIZE];
int iMask;
int bit;
int COUNT2=0; int COUNT4=0; int COUNT8=0;


void dtob(int score,int size) {
  //int bit=1,i;
	int bit=1;
  char c[size];
  //for (i=0;i<size;i++) {
  for (int i=0;i<size;i++) {
    if (score&bit){ c[i]='1'; }else{ c[i]='0'; }
    bit<<=1;
  }
  // 計算結果の表示
  for (int i=size-1;i>=0;i--){ putchar(c[i]); }
  printf("\n");
}
void TimeFormat(clock_t utime,char *form){
    int dd,hh,mm;
    float ftime,ss;
    ftime=(float)utime/CLOCKS_PER_SEC;
    mm=(int)ftime/60;
    ss=ftime-(int)(mm*60);
    dd=mm/(24*60);
    mm=mm%(24*60);
    hh=mm/60;
    mm=mm%60;
    sprintf(form,"%7d %02d:%02d:%02.0f",dd,hh,mm,ss);
}
long getUnique(){ 
  return COUNT2+COUNT4+COUNT8;
}
long getTotal(){ 
  return COUNT2*2+COUNT4*4+COUNT8*8;
}
void rotate_bitmap(int abefore[],int aafter[]){
  for(int i=0;i<iSize;i++){
    int t=0;
    for(int j=0;j<iSize;j++){
			t|=((abefore[j]>>i)&1)<<(iSize-j-1); // x[j] の i ビット目を
		}
    aafter[i]=t;                        // y[i] の j ビット目にする
  }
}
int rh(int a,int sz){
	int tmp=0;
	for(int i=0;i<=sz;i++){
		if(a&(1<<i)){ return tmp|=(1<<(sz-i)); }
	}
	return tmp;
}
void vMirror_bitmap(int abefore[],int aafter[]){
  for(int i=0;i< iSize;i++) {
    int score=abefore[i];
    aafter[i]=rh(score,iSize-1);
  }
}
int intncmp(int lt[],int rt[]){
  int rtn=0;
  for(int k=0;k<iSize;k++){
    rtn=lt[k]-rt[k];
    if(rtn!=0){ break;}
  }
  return rtn;
}
void symmetryOps_bitmap(){
  int nEquiv;
  int aTrial[iSize];
  int aScratch[iSize];
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<iSize;i++){ aTrial[i]=aBoard[i];}
  rotate_bitmap(aTrial,aScratch);  //時計回りに90度回転
  int k=intncmp(aBoard,aScratch);
  if(k>0)return;
  if(k==0){ nEquiv=2;}else{
    rotate_bitmap(aScratch,aTrial);//時計回りに180度回転
    k=intncmp(aBoard,aTrial);
    if(k>0)return;
    if(k==0){ nEquiv=4;}else{
      rotate_bitmap(aTrial,aScratch);//時計回りに270度回転
      k=intncmp(aBoard,aScratch);
      if(k>0){ return;}
      nEquiv=8;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<iSize;i++){ aScratch[i]=aBoard[i];}
  vMirror_bitmap(aScratch,aTrial);    //垂直反転
  k=intncmp(aBoard,aTrial);
  if(k>0){ return; }
  if(nEquiv>2){             //-90度回転 対角鏡と同等       
    rotate_bitmap(aTrial,aScratch);
    k=intncmp(aBoard,aScratch);
    if(k>0){return;}
    if(nEquiv>4){           //-180度回転 水平鏡像と同等
      rotate_bitmap(aScratch,aTrial);
      k=intncmp(aBoard,aTrial);
      if(k>0){ return;}  //-270度回転 反対角鏡と同等
      rotate_bitmap(aTrial,aScratch);
      k=intncmp(aBoard,aScratch);
      if(k>0){ return;}
    }
  }
  if(nEquiv==2){ COUNT2++; }
  if(nEquiv==4){ COUNT4++; }
  if(nEquiv==8){ COUNT8++; }
}
void NQueen(int y, int left, int down, int right){
  int bitmap=iMask&~(left|down|right); /* 配置可能フィールド */
  if (y==iSize) {
    if(!bitmap){
	    aBoard[y]=bitmap;
			symmetryOps_bitmap();
    }
  }else{
    while (bitmap) {
      bitmap^=aBoard[y]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
      NQueen(y+1,(left|bit)<<1,down|bit,(right|bit)>>1);
     }
  } 
}
int main(void){
  clock_t st; char t[20];
  printf("%s\n"," N:        Total       Unique        dd:hh:mm:ss");
  for(int i=2;i<=MAXSIZE;i++){
    iSize=i; lTotal=0; lUnique=0;
		COUNT2=0;COUNT4=0;COUNT8=0;
    for(int j=0;j<iSize;j++){ aBoard[j]=j; }
    st=clock();
    iMask=(1<<iSize)-1; // 初期化
    NQueen(0,0,0,0);
    TimeFormat(clock()-st,t);
    printf("%2d:%13ld%16ld%s\n",iSize,getTotal(),getUnique(),t);
  } 
}

