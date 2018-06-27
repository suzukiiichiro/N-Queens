/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイル
 $ gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm 07_04N-Queen.c -o 04N-Queen

 実行
 $ ./04N-Queen

 ４．出力結果の表示処理
      Nが順次大きくなりつつ処理を継続できるように処理を修正

 実行結果
  N:        Total       Unique        hh:mm:ss.ms
  2:            0               0            0.00
  3:            0               0            0.00
  4:            2               0            0.00
  5:           10               0            0.00
  6:            4               0            0.00
  7:           40               0            0.00
  8:           92               0            0.00
  9:          352               0            0.00
 10:          724               0            0.00
 11:         2680               0            0.01
 12:        14200               0            0.05
 13:        73712               0            0.30
 14:       365596               0            1.93
 15:      2279184               0           13.50
 16:     14772512               0         1:39.30
 17:     95815104               0        12:29.59
 */

#include <stdio.h>
#include <time.h>
#define MAX 24

long TOTAL=0;
long UNIQUE=0;
int aBoard[MAX];
int fA[2*MAX-1];	//縦列にクイーンを一つだけ配置
int fB[2*MAX-1];	//斜め列にクイーンを一つだけ配置
int fC[2*MAX-1];	//斜め列にクイーンを一つだけ配置

void NQueen(int row,int size);
void TimeFormat(clock_t utime,char *form);

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
void NQueen(int row,int size){
	if(row==size){
		TOTAL++;
	}else{
		for(int i=0;i<size;i++){
			aBoard[row]=i;
			if(fA[i]==0&&fB[row-i+(size-1)]==0&&fC[row+i]==0){
				fA[i]=fB[row-i+(size-1)]=fC[row+i]=1;
				NQueen(row+1,size);
				fA[i]=fB[row-i+(size-1)]=fC[row+i]=0;
			}
		}
	}
}
int main(void){
	clock_t st;
	char t[20];
	int min=4;
	printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
	for(int i=min;i<=MAX;i++){
		TOTAL=0;
		UNIQUE=0;
		for(int j=0;j<i;j++){
			aBoard[j]=j;
		}
		st=clock();
		NQueen(0,i);
		TimeFormat(clock()-st,t);
		printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
	}
	return 0;
}
