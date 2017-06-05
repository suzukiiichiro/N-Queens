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
   １．ブルートフォース（力まかせ探索） NQueen1()
   ２．配置フラグ（制約テスト高速化）   NQueen2()
   ３．バックトラック                   NQueen3() N16: 1:07
   ４．対称解除法(回転と斜軸）          NQueen4() N16: 1:09
   ５．枝刈りと最適化                   NQueen5() N16: 0:18
   ６．ビットマップ                     NQueen6() N16: 0:13
 <>７．                                 NQueen7()
   ８．                                 NQueen8()
   ９．完成型                           NQueen9() N16: 0:02
   10．マルチスレッド                   NQueen10()

  実行結果


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

void dtob(int score,int size) {
  int bit = 1, i;
  char c[size];
 
  for (i = 0; i < size; i++) {
    if (score & bit)
      c[i] = '1';
    else
      c[i] = '0';
    bit <<= 1;
  }
  // 計算結果の表示
  for ( i = size - 1; i >= 0; i-- ) {
      putchar(c[i]);
  }
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
  return lUnique;
}
long getTotal(){ 
  return lTotal;
}
void rotate(int check[],int scr[],int n,int neg){
  int k=neg?0:n-1;
  int incr=(neg?+1:-1);
  for(int j=0;j<n;k+=incr){ scr[j++]=check[k];}
  k=neg?n-1:0;
  for(int j=0;j<n;k-=incr){ check[scr[j++]]=k;}
}
void vMirror(int check[],int n){
  for(int j=0;j<n;j++){ check[j]=(n-1)- check[j];}
}
int intncmp(int lt[],int rt[],int n){
  int rtn=0;
  for(int k=0;k<n;k++){
    rtn=lt[k]-rt[k];
    if(rtn!=0){ break;}
  }
  return rtn;
}
int symmetryOps(int bitmap){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<iSize;i++){ aTrial[i]=aBoard[i];}
  rotate(aTrial,aScratch,iSize,0);  //時計回りに90度回転
  int k=intncmp(aBoard,aTrial,iSize);
  if(k>0)return 0;
  if(k==0){ nEquiv=1; }else{
    rotate(aTrial,aScratch,iSize,0);//時計回りに180度回転
    k=intncmp(aBoard,aTrial,iSize);
    if(k>0)return 0;
    if(k==0){ nEquiv=2; }else{
      rotate(aTrial,aScratch,iSize,0);//時計回りに270度回転
      k=intncmp(aBoard,aTrial,iSize);
      if(k>0){ return 0; }
      nEquiv=4;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<iSize;i++){ aTrial[i]=aBoard[i];}
  vMirror(aTrial,iSize);    //垂直反転
  k=intncmp(aBoard,aTrial,iSize);
  if(k>0){ return 0; }
  if(nEquiv>1){             //-90度回転 対角鏡と同等       
    rotate(aTrial,aScratch,iSize,1);
    k=intncmp(aBoard,aTrial,iSize);
    if(k>0){return 0; }
    if(nEquiv>2){           //-180度回転 水平鏡像と同等
      rotate(aTrial,aScratch,iSize,1);
      k=intncmp(aBoard,aTrial,iSize);
      if(k>0){ return 0; }  //-270度回転 反対角鏡と同等
      rotate(aTrial,aScratch,iSize,1);
      k=intncmp(aBoard,aTrial,iSize);
      if(k>0){ return 0; }
    }
  }
  return nEquiv * 2;
}
int rh(int a,int sz){
  int tmp;
  int i;
  sz=sz-1;
  /* ビット入替 */
  tmp = 0;
  for( i = 0; i <= sz; i++ )
  {
    if( a & ( 1 << i ) )
    {
      tmp |= ( 1 << ( sz - i ) );
    }
  }
  a = tmp;
  return tmp;
}
void revHorzBitmap(int abefore[],int aafter[]){
  for(int i=0;i< iSize;i++) {
    int score=abefore[i];
    aafter[i]=rh(score,iSize);
  }
}

void rotateBitmap90(int abefore[],int aafter[]){
  for(int i=0;i<iSize;i++) {
    int t = 0;
    for (int j = 0; j < iSize; j++)
        t |= ((abefore[j] >> i) & 1) << j; // x[j] の i ビット目を
    aafter[i] = t;                        // y[i] の j ビット目にする
  }
  int ta[iSize];
  for(int i=0;i<iSize;i++) {
    ta[i]=aafter[i];
  }
  revHorzBitmap(ta,aafter);
}
int less(int cgd[],int org[]){
  for(int i=0;i<iSize;i++) {
    if(cgd[i] > org[i]){
     return 0;
    }else if(cgd[i] < org[i]){
     return 1;
    } 
  }
  return 1;
}
int checkSymmetryBitmap(int aorg[]){
  int t[iSize];
  int t2[iSize];
  revHorzBitmap(aorg,t2);
/*
printf("元の配列\n");
for(int i=0;i< iSize;i++) {
      dtob(aorg[i],iSize);
}
printf("左右対称\n");
for(int i=0;i< iSize;i++) {
      dtob(t2[i],iSize);
}
*/
  if(less(t2,aorg)==0) return 0;
	rotateBitmap90(aorg,t);
/*
printf("元の配列\n");
for(int i=0;i< iSize;i++) {
      dtob(aorg[i],iSize);
}
printf("90度回転\n");
for(int i=0;i< iSize;i++) {
      dtob(t[i],iSize);
}
*/
	if(less(t,aorg)==0) return 0;
	revHorzBitmap(t,t2);
/*
printf("元の配列\n");
for(int i=0;i< iSize;i++) {
      dtob(t[i],iSize);
}
printf("左右対称\n");
for(int i=0;i< iSize;i++) {
      dtob(t2[i],iSize);
}
*/
	if(less(t2,aorg)==0) return 0;
	rotateBitmap90(t,t2);
/*
printf("元の配列\n");
for(int i=0;i< iSize;i++) {
      dtob(t[i],iSize);
}
printf("９０度回転\n");
for(int i=0;i< iSize;i++) {
      dtob(t2[i],iSize);
}
*/
	if(less(t2,aorg)==0) return 0;
	revHorzBitmap(t2,t);
/*
printf("元の配列\n");
for(int i=0;i< iSize;i++) {
      dtob(t2[i],iSize);
}
printf("左右対称\n");
for(int i=0;i< iSize;i++) {
      dtob(t[i],iSize);
}
*/
	if(less(t,aorg)==0) return 0;
	rotateBitmap90(t2,t);
/*
printf("元の配列\n");
for(int i=0;i< iSize;i++) {
      dtob(t2[i],iSize);
}
printf("９０度回転\n");
for(int i=0;i< iSize;i++) {
      dtob(t[i],iSize);
}
*/
	if(less(t,aorg)==0) return 0;
	revHorzBitmap(t,t2);
/*
printf("元の配列\n");
for(int i=0;i< iSize;i++) {
      dtob(t[i],iSize);
}
printf("左右対称\n");
for(int i=0;i< iSize;i++) {
      dtob(t2[i],iSize);
}
*/
	if(less(t2,aorg)==0) return 0;
	return 1;
}
void NQueen6(int y, int left, int down, int right){
  int bitmap=iMask&~(left|down|right); /* 配置可能フィールド */
  if (y==iSize) {
    if(!bitmap){
	    aBoard[y]=bitmap;
      lTotal++;
    //int k=symmetryOps(bitmap);
      if(checkSymmetryBitmap(aBoard) == 1) {
        lUnique++;
      }
//      printf("check終了\n");
    }
  }else{
    while (bitmap) {
      //aBoard[y]=bit=-bitmap&bitmap;       /* 最も下位の１ビットを抽出 */
      //bitmap^=bit;
      bitmap^=aBoard[y]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
      NQueen6(y+1,(left|bit)<<1,down|bit,(right|bit)>>1);
     }
  } 
}
int main(void){
  clock_t st; char t[20];
  printf("%s\n"," N:        Total       Unique        dd:hh:mm:ss");
  //for(int i=2;i<=MAXSIZE;i++){
  for(int i=2;i<=17;i++){
    iSize=i; lTotal=0; lUnique=0;
    for(int j=0;j<iSize;j++){ aBoard[j]=j; }
    st=clock();
    iMask=(1<<iSize)-1; // 初期化
    NQueen6(0,0,0,0);
    TimeFormat(clock()-st,t);
    printf("%2d:%13ld%16ld%s\n",iSize,getTotal(),getUnique(),t);
  } 
}

