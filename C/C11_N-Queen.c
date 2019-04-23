/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイル
 $ gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm C11_N-Queen.c -o C11_N-Queen

 実行
 $ ./C11_N-Queen

 １１．枝刈り

  前章のコードは全ての解を求めた後に、ユニーク解以外の対称解を除去していた
  ある意味、「生成検査法（generate ＆ test）」と同じである
  問題の性質を分析し、バックトラッキング/前方検査法と同じように、無駄な探索を省略することを考える
  ユニーク解に対する左右対称解を予め削除するには、1行目のループのところで、
  右半分だけにクイーンを配置するようにすればよい
  Nが奇数の場合、クイーンを1行目中央に配置する解は無い。
  他の3辺のクィーンが中央に無い場合、その辺が上辺に来るよう回転し、場合により左右反転することで、
  最小値解とすることが可能だから、中央に配置したものしかユニーク解には成り得ない
  しかし、上辺とその他の辺の中央にクィーンは互いの効きになるので、配置することが出来ない


  1. １行目角にクイーンがある場合、とそうでない場合で処理を分ける
    １行目かどうかの条件判断はループ外に出してもよい
    処理時間的に有意な差はないので、分かりやすいコードを示した
  2.１行目角にクイーンがある場合、回転対称形チェックを省略することが出来る
    １行目角にクイーンがある場合、他の角にクイーンを配置することは不可
    鏡像についても、主対角線鏡像のみを判定すればよい
    ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい

  １行目角にクイーンが無い場合、クイーン位置より右位置の８対称位置にクイーンを置くことはできない
  置いた場合、回転・鏡像変換により得られる状態のユニーク判定値が明らかに大きくなる
    ☓☓・・・Ｑ☓☓
    ☓・・・／｜＼☓
    ｃ・・／・｜・rt
    ・・／・・｜・・
    ・／・・・｜・・
    lt・・・・｜・ａ
    ☓・・・・｜・☓
    ☓☓ｂ・・dn☓☓
    
  １行目位置が確定した時点で、配置可能位置を計算しておく（☓の位置）
  lt, dn, lt 位置は効きチェックで配置不可能となる
  回転対称チェックが必要となるのは、クイーンがａ, ｂ, ｃにある場合だけなので、
  90度、180度、270度回転した状態のユニーク判定値との比較を行うだけで済む

 *
 *  実行結果
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.00
12:        14200            1787            0.00
13:        73712            9233            0.02
14:       365596           45752            0.14
15:      2279184          285053            0.91
16:     14772512         1846955            6.40
17:     95815104        11977939           45.94
*/

#include <stdio.h>
#include <time.h>
#define MAX 24
//
int aBoard[MAX];
int aT[MAX];
int aS[MAX];
int bit;
int COUNT2,COUNT4,COUNT8;
int BOUND1,BOUND2,TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
//
void TimeFormat(clock_t utime,char *form);
void rotate_bitmap(int bf[],int af[],int si);
void vMirror_bitmap(int bf[],int af[],int si);
int intncmp(int lt[],int rt[],int n);
void dtob(int score,int si);
int rh(int a,int sz);
void symmetryOps_bitmap(int si);
long getUnique();
long getTotal();
void backTrack2(int si,int mask,int y,int l,int d,int r);
void backTrack1(int si,int mask,int y,int l,int d,int r);
void NQueen(int size,int mask);
//
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
	if(dd)
	sprintf(form,"%4d %02d:%02d:%05.2f",dd,hh,mm,ss);
	else if(hh)
	sprintf(form,"     %2d:%02d:%05.2f",hh,mm,ss);
	else if(mm)
	sprintf(form,"        %2d:%05.2f",mm,ss);
	else
	sprintf(form,"           %5.2f",ss);
}
//
void dtob(int score,int size){
	int bit=1;
	char c[size];
	for(int i=0;i<size;i++){
		if(score&bit){
			c[i]='1';
		}else{
			c[i]='0';
		}
		bit<<=1;
	}
	for(int i=size-1;i>=0;i--){
		putchar(c[i]);
	}
	printf("\n");
}
//
int rh(int a,int size){
	int tmp=0;
	for(int i=0;i<=size;i++){
		if(a&(1<<i)){
			return tmp|=(1<<(size-i));
		}
	}
	return tmp;
}
//
void vMirror_bitmap(int bf[],int af[],int size){
	int score;
	for(int i=0;i<size;i++){
		score=bf[i];
		af[i]=rh(score,size-1);
	}
}
//
void rotate_bitmap(int bf[],int af[],int size){
	int t;
	for(int i=0;i<size;i++){
		t=0;
		for(int j=0;j<size;j++){
			t|=((bf[j]>>i)&1)<<(size-j-1); // x[j] の i ビット目を
		}
		af[i]=t;                        // y[i] の j ビット目にする
	}
}
//
int intncmp(int lt[],int rt[],int n){
	int rtn=0;
	for(int k=0;k<n;k++){
		rtn=lt[k]-rt[k];
		if(rtn!=0){
			break;
		}
	}
	return rtn;
}
//
long getUnique(){
	return COUNT2+COUNT4+COUNT8;
}
//
long getTotal(){
	return COUNT2*2+COUNT4*4+COUNT8*8;
}
//
void symmetryOps_bitmap(int size){
	int nEquiv;
	// 回転・反転・対称チェックのためにboard配列をコピー
	for(int i=0;i<size;i++){
		aT[i]=aBoard[i];
	}
	rotate_bitmap(aT,aS,size);    //時計回りに90度回転
	int k=intncmp(aBoard,aS,size);
	if(k>0) return;
	if(k==0){
		nEquiv=2;
	}else{
		rotate_bitmap(aS,aT,size);  //時計回りに180度回転
		k=intncmp(aBoard,aT,size);
		if(k>0) return;
		if(k==0){
			nEquiv=4;
		}else{
			rotate_bitmap(aT,aS,size);  //時計回りに270度回転
			k=intncmp(aBoard,aS,size);
			if(k>0){
				return;
			}
			nEquiv=8;
		}
	}
	// 回転・反転・対称チェックのためにboard配列をコピー
	for(int i=0;i<size;i++){
		aS[i]=aBoard[i];
	}
	vMirror_bitmap(aS,aT,size);   //垂直反転
	k=intncmp(aBoard,aT,size);
	if(k>0){
		return;
	}
	if(nEquiv>2){             //-90度回転 対角鏡と同等
		rotate_bitmap(aT,aS,size);
		k=intncmp(aBoard,aS,size);
		if(k>0){
			return;
		}
		if(nEquiv>4){           //-180度回転 水平鏡像と同等
			rotate_bitmap(aS,aT,size);
			k=intncmp(aBoard,aT,size);
			if(k>0){
				return;
			}       //-270度回転 反対角鏡と同等
			rotate_bitmap(aT,aS,size);
			k=intncmp(aBoard,aS,size);
			if(k>0){
				return;
			}
		}
	}
	if(nEquiv==2){
		COUNT2++;
	}
	if(nEquiv==4){
		COUNT4++;
	}
	if(nEquiv==8){
		COUNT8++;
	}
}
//
void backTrack2(int size,int mask,int row,int left,int down,int right){
	int bit;
	int bitmap=mask&~(left|down|right);
  // 【枝刈り】
	if(row==size-1){ 								
		if(bitmap){
      //【枝刈り】 最下段枝刈り
			if((bitmap&LASTMASK)==0){ 	
				aBoard[row]=bitmap;
				symmetryOps_bitmap(size);
			}
		}
	}else{
    //【枝刈り】上部サイド枝刈り
    if(row<BOUND1){             	
      bitmap&=~SIDEMASK;
    //【枝刈り】下部サイド枝刈り
    }else if(row==BOUND2) {     	
      if((down&SIDEMASK)==0){ return; }
      if((down&SIDEMASK)!=SIDEMASK){ bitmap&=SIDEMASK; }
    }
		while(bitmap){
			bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
			backTrack2(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
		}
	}
}
//
void backTrack1(int size,int mask,int row,int left,int down,int right){
	int bit;
	int bitmap=mask&~(left|down|right);
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
  if(row==size-1) {
    if(bitmap){
      aBoard[row]=bitmap;
      COUNT8++;
    }
  }else{
		//【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
		// ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
    if(row<BOUND1) {
      bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
		while(bitmap){
			bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
			backTrack1(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
		}
	}
}
//
void NQueen(int size,int mask){
	int bit;
	TOPBIT=1<<(size-1);
	aBoard[0]=1;
	for(BOUND1=2;BOUND1<size-1;BOUND1++){
		aBoard[1]=bit=(1<<BOUND1);
		backTrack1(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
	}
	SIDEMASK=LASTMASK=(TOPBIT|1);
	ENDBIT=(TOPBIT>>1);
	for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
		aBoard[0]=bit=(1<<BOUND1);
		backTrack2(size,mask,1,bit<<1,bit,bit>>1);
		LASTMASK|=LASTMASK>>1|LASTMASK<<1;
		ENDBIT>>=1;
	}
}
//
int main(void){
	clock_t st;
	char t[20];
	int min=4;
	int mask=0;
	printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
	for(int i=min;i<=MAX;i++){
		COUNT2=COUNT4=COUNT8=0;
		mask=(1<<i)-1;
		for(int j=0;j<i;j++){
			aBoard[j]=j;
		}
		st=clock();
		NQueen(i,mask);
		TimeFormat(clock()-st,t);
		printf("%2d:%13ld%16ld%s\n",i,getTotal(),getUnique(),t);
	}
	return 0;
}
