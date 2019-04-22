/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイル
 $ gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm C06_N-Queen.c -o C06_N-Queen

 実行
 $ ./06N-Queen

 * ６．バックトラック＋ビットマップ
 *

 実行結果
  N:        Total       Unique        hh:mm:ss.ms
 */
//
#include <stdio.h>
#include <time.h>
#define MAX 24
//
long TOTAL=0;
long UNIQUE=0;
int size;
int aBoard[MAX];
int MASK ;
// int fA[2*MAX-1];	//縦列にクイーンを一つだけ配置
// int fB[2*MAX-1];	//斜め列にクイーンを一つだけ配置
// int fC[2*MAX-1];	//斜め列にクイーンを一つだけ配置
// int aT[MAX];       //aT:aTrial[]
// int aS[MAX];       //aS:aScrath[]
//
void TimeFormat(clock_t utime,char *form);
void NQueen(int row,int left,int down,int right);
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
void NQueen(int row,int left,int down,int right){
  int bitmap=0;
  int bit=0;
  if(row==size){
    TOTAL++;
  }else{
    bitmap=(MASK&~(left|down|right));
    while(bitmap){
      bit=(-bitmap&bitmap);
      bitmap=(bitmap^bit);
      NQueen(row+1,(left|bit)<<1, down|bit, (right|bit)>>1);
    }
  }
}
//
int main(void){
	clock_t st;
	char t[20];
	int min=4;
  int max=17;
	printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
	for(size=min;size<=max;size++){
		TOTAL=0;
		UNIQUE=0;
    MASK=((1<<size)-1);
		for(int j=0;j<size;j++){
			aBoard[j]=j;
		}
		st=clock();
		NQueen(0,0,0,0);
		TimeFormat(clock()-st,t);
		printf("%2d:%13ld%16ld%s\n",size,TOTAL,UNIQUE,t);
	}
	return 0;
}
