/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイル
 $ gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm C12_N-Queen.c -o C12_N-Queen

 実行
 $ ./C12_N-Queen

 １２．対称解除法の最適化
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
13:        73712            9233            0.01
14:       365596           45752            0.06
15:      2279184          285053            0.37
16:     14772512         1846955            2.41
17:     95815104        11977939           16.79
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
long getUnique();
long getTotal();
void symmetryOps(int si);
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
long getUnique(){
	return COUNT2+COUNT4+COUNT8;
}
//
long getTotal(){
	return COUNT2*2+COUNT4*4+COUNT8*8;
}
//
void symmetryOps(int si){
  int own,ptn,you,bit;
  //90度回転
  if(aBoard[BOUND2]==1){ own=1; ptn=2;
    while(own<=si-1){ bit=1; you=si-1;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you--; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>si-1){ COUNT2++; return; }
  }
  //180度回転
  if(aBoard[si-1]==ENDBIT){ own=1; you=si-1-1;
    while(own<=si-1){ bit=1; ptn=TOPBIT;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>si-1){ COUNT4++; return; }
  }
  //270度回転
  if(aBoard[BOUND1]==TOPBIT){ own=1; ptn=TOPBIT>>1;
    while(own<=si-1){ bit=1; you=0;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you++; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  COUNT8++;
}
//
void backTrack2(int size,int mask,int row,int left,int down,int right){
	int bit;
	int bitmap=mask&~(left|down|right);
	if(row==size-1){ 								// 【枝刈り】
		if(bitmap){
			if((bitmap&LASTMASK)==0){ 	//【枝刈り】 最下段枝刈り
				aBoard[row]=bitmap;
				symmetryOps(size);
			}
		}
	}else{
    if(row<BOUND1){             	//【枝刈り】上部サイド枝刈り
      bitmap&=~SIDEMASK;
    }else if(row==BOUND2) {     	//【枝刈り】下部サイド枝刈り
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
