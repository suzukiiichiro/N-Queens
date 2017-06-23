/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
   １．ブルートフォース（力まかせ探索） NQueen01()
   ２．配置フラグ（制約テスト高速化）   NQueen02()
   ３．バックトラック                   NQueen03() N17: 8:05
   ４．対称解除法(回転と斜軸）          NQueen04() N17: 7:54
   ５．枝刈りと最適化                   NQueen05() N17: 2:14
   ６．ビットマップ                     NQueen06() N17: 1:30
   ７．ビットマップ+対称解除法          NQueen07() N17: 2:24
 <>８．ビットマップ+クイーンの場所で分岐NQueen08() N17: 1:26
   ９．ビットマップ+枝刈りと最適化      NQueen09() N17: 0:16
   10．もっとビットマップ(takaken版)    NQueen10() N17: 0:10
   11．マルチスレッド(構造体)           NQueen11() N17: 0:14
   12．マルチスレッド(pthread)          NQueen12() N17: 0:13
   13．マルチスレッド(join)             NQueen13() N17: 0:17
   14．マルチスレッド(mutex)            NQueen14() N17: 0:27
   15．マルチスレッド(アトミック対応)   NQueen15() N17: 0:05
   16．アドレスとポインタ               NQueen16() N17: 0:04
   17．アドレスとポインタ(多重配列)     NQueen17() N17: 

  Java版 N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_N-Queen
  Bash版 N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_Bash
  Lua版  N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_Lua
 

  ８．ビットマップ＋クイーンの場所で分岐
 *
 * ■ユニーク解の個数を求める
 *   先ず最上段の行のクイーンの位置に着目します。その位置が左半分の領域にあればユ
 * ニーク解には成り得ません。何故なら左右反転によって得られるパターンのユニーク判
 * 定値の方が確実に小さくなるからです。また、Ｎが奇数の場合に中央にあった場合はど
 * うでしょう。これもユニーク解には成り得ません。何故なら仮に中央にあった場合、そ
 * れがユニーク解であるためには少なくとも他の外側の３辺におけるクイーンの位置も中
 * 央になければならず、それは互いの効き筋にあたるので有り得ません。
 *
 *
 * ***********************************************************************
 * 最上段の行のクイーンの位置は中央を除く右側の領域に限定されます。(ただし、N ≧ 2)
 * ***********************************************************************
 * 
 *   次にその中でも一番右端(右上の角)にクイーンがある場合を考えてみます。他の３つ
 * の角にクイーンを置くことはできないので(効き筋だから）、ユニーク解であるかどうか
 * を判定するには、右上角から左下角を通る斜軸で反転させたパターンとの比較だけになり
 * ます。突き詰めれば、
 * 
 * [上から２行目のクイーンの位置が右から何番目にあるか]
 * [右から２列目のクイーンの位置が上から何番目にあるか]
 * 
 *
 * を比較するだけで判定することができます。この２つの値が同じになることはないからです。
 * 
 *       3 0
 *       ↓↓
 * - - - - Q ←0
 * - Q - - - ←3
 * - - - - -         上から２行目のクイーンの位置が右から４番目にある。
 * - - - Q -         右から２列目のクイーンの位置が上から４番目にある。
 * - - - - -         しかし、互いの効き筋にあたるのでこれは有り得ない。
 * 
 *   結局、再帰探索中において下図の X への配置を禁止する枝刈りを入れておけば、得
 * られる解は総てユニーク解であることが保証されます。
 * 
 * - - - - X Q
 * - Q - - X -
 * - - - - X -
 * - - - - X -
 * - - - - - -
 * - - - - - -
 * 
 *   次に右端以外にクイーンがある場合を考えてみます。オリジナルがユニーク解である
 * ためには先ず下図の X への配置は禁止されます。よって、その枝刈りを先ず入れておき
 * ます。
 * 
 * X X - - - Q X X
 * X - - - - - - X
 * - - - - - - - -
 * - - - - - - - -
 * - - - - - - - -
 * - - - - - - - -
 * X - - - - - - X
 * X X - - - - X X
 * 
 *   次にクイーンの利き筋を辿っていくと、結局、オリジナルがユニーク解ではない可能
 * 性があるのは、下図の A,B,C の位置のどこかにクイーンがある場合に限られます。従っ
 * て、90度回転、180度回転、270度回転の３通りの変換パターンだけを調べれはよいこと
 * になります。
 * 
 * X X x x x Q X X
 * X - - - x x x X
 * C - - x - x - x
 * - - x - - x - -
 * - x - - - x - -
 * x - - - - x - A
 * X - - - - x - X
 * X X B - - x X X
 *
 *
 * ■ユニーク解から全解への展開
 *   これまでの考察はユニーク解の個数を求めるためのものでした。全解数を求めるには
 * ユニーク解を求めるための枝刈りを取り除いて全探索する必要があります。したがって
 * 探索時間を犠牲にしてしまうことになります。そこで「ユニーク解の個数から全解数を
 * 導いてしまおう」という試みが考えられます。これは、左右反転によるパターンの探索
 * を省略して最後に結果を２倍するというアイデアの拡張版といえるものです。そしてそ
 * れを実現させるには「あるユニーク解が属するグループの要素数はいくつあるのか」と
 * いう考察が必要になってきます。
 * 
 *   最初に、クイーンが右上角にあるユニーク解を考えます。斜軸で反転したパターンが
 * オリジナルと同型になることは有り得ないことと(×２)、右上角のクイーンを他の３つの
 * 角に写像させることができるので(×４)、このユニーク解が属するグループの要素数は必
 * ず８個(＝２×４)になります。
 * 
 *   次に、クイーンが右上角以外にある場合は少し複雑になりますが、考察を簡潔にする
 * ために次の事柄を確認します。
 *
 * TOTAL = (COUNT8 * 8) + (COUNT4 * 4) + (COUNT2 * 2);
 *   (1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルか
 *    ら180度回転)させても、さらに90度回転(オリジナルから270度回転)させてもオリ
 *    ジナルと同型になる。  
 *
 *    COUNT2 * 2
 * 
 *   (2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナ
 *    ルとは異なる。ただし、180度回転させた場合はオリジナルと同型になることも有
 *    り得る。 
 *
 *    COUNT4 * 4
 * 
 *   (3) (1) に該当するユニーク解が属するグループの要素数は、左右反転させたパターンを
 *       加えて２個しかありません。(2)に該当するユニーク解が属するグループの要素数は、
 *       180度回転させて同型になる場合は４個(左右反転×縦横回転)、そして180度回転させても
 *       オリジナルと異なる場合は８個になります。(左右反転×縦横回転×上下反転)
 * 
 *    COUNT8 * 8 
 *
 *   以上のことから、ひとつひとつのユニーク解が上のどの種類に該当するのかを調べる
 * ことにより全解数を計算で導き出すことができます。探索時間を短縮させてくれる枝刈
 * りを外す必要がなくなったというわけです。 
 * 
 *   UNIQUE  COUNT2      +  COUNT4      +  COUNT8
 *   TOTAL  (COUNT2 * 2) + (COUNT4 * 4) + (COUNT8 * 8)
 *
 * 　これらを実現すると、前回のNQueen3()よりも実行速度が遅くなります。
 * 　なぜなら、対称・反転・斜軸を反転するための処理が加わっているからです。
 * ですが、今回の処理を行うことによって、さらにNQueen5()では、処理スピードが飛
 * 躍的に高速化されます。そのためにも今回のアルゴリズム実装は必要なのです。
 *

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
  15:      2279184          285053      0 00:00:02
  16:     14772512         1846955      0 00:00:13
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
void backTrack2(int y, int left, int down, int right){
  int bitmap=iMask&~(left|down|right); /* 配置可能フィールド */
  if (y==iSize) {
    if(!bitmap){
	    aBoard[y]=bitmap;
			symmetryOps_bitmap();
    }
  }else{
    while(bitmap) {
      bitmap^=aBoard[y]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
      backTrack2(y+1,(left|bit)<<1,down|bit,(right|bit)>>1);
     }
  } 
}
void backTrack1(int y, int left, int down, int right){
  int bitmap=iMask&~(left|down|right); /* 配置可能フィールド */
  if (y==iSize) {
    if(!bitmap){
	    aBoard[y]=bitmap;
			symmetryOps_bitmap();
    }
  }else{
    while(bitmap) {
      bitmap^=aBoard[y]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
      backTrack1(y+1,(left|bit)<<1,down|bit,(right|bit)>>1);
     }
  } 
}
int BOUND1;
int BOUND2;
int TOPBIT;
int SIZEE;
int SIDEMASK;
int LASTMASK;
int ENDBIT;
void NQueen(int y, int left, int down, int right){
  SIZEE=iSize-1;
	TOPBIT=1<<SIZEE;

  aBoard[0]=1;
  for(BOUND1=2;BOUND1<iSize-1;BOUND1++){
    aBoard[1]=bit=(1<<BOUND1);
    backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1));
  }

  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  for(BOUND1=1,BOUND2=iSize-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    backTrack2(1,bit<<1,bit,bit>>1);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  }
  // 07_7
  //backTrack1(0,0,0,0);
}
int main(void){
  clock_t st; char t[20];
  printf("%s\n"," N:        Total       Unique        dd:hh:mm:ss");
  for(int i=2;i<=MAXSIZE;i++){
    iSize=i; lTotal=0; lUnique=0;
	  COUNT2=COUNT4=COUNT8=0;
    iMask=(1<<iSize)-1; // 初期化
    for(int j=0;j<iSize;j++){ aBoard[j]=j; }
    st=clock();
    NQueen(0,0,0,0);
    TimeFormat(clock()-st,t);
    printf("%2d:%13ld%16ld%s\n",iSize,getTotal(),getUnique(),t);
  } 
}

