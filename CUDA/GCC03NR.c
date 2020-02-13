
// $ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC03NR.c && ./a.out
/**
３．CPUR 再帰 バックトラック
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               0            0.00
 5:           10               0            0.00
 6:            4               0            0.00
 7:           40               0            0.00
 8:           92               0            0.00
 9:          352               0            0.00
10:          724               0            0.00
11:         2680               0            0.01
12:        14200               0            0.05
13:        73712               0            0.26
14:       365596               0            1.56
15:      2279184               0           10.00
16:     14772512               0         1:08.35
17:     95815104               0         8:09.30

３．CPU 非再帰 バックトラック
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               0            0.00
 5:           10               0            0.00
 6:            4               0            0.00
 7:           40               0            0.00
 8:           92               0            0.00
 9:          352               0            0.00
10:          724               0            0.00
11:         2680               0            0.01
12:        14200               0            0.05
13:        73712               0            0.29
14:       365596               0            1.68
15:      2279184               0           10.63
16:     14772512               0         1:12.37
17:     95815104               0         8:39.61 

*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
//
#define MAX 27
//変数宣言
int aBoard[MAX];
int down[2*MAX-1]; 	//down:flagA 縦 配置フラグ
int right[2*MAX-1];  //right:flagB 斜め配置フラグ
int left[2*MAX-1]; //left:flagC 斜め配置フラグ
long TOTAL=0;
long UNIQUE=0;
//関数宣言
void NQueen(int row,int size);
void TimeFormat(clock_t utime,char *form);
void NQueenR(int row,int size);
void print(int size);
//出力
void print(int size){
	printf("%ld: ",TOTAL);
	for(int j=0;j<size;j++){
		printf("%d ",aBoard[j]);
	}
	printf("\n");
}
//CPU 非再帰版 ロジックメソッド
void NQueen(int row,int size){
	int sizeE=size-1;
	bool matched;
	while(row>=0){
		matched=false;
		// １回目はaBoard[row]が-1なのでcolを0で初期化
		// ２回目以降はcolを<sizeまで右へシフト
		for(int col=aBoard[row]+1;col<size;col++){
			if(down[col]==0
					&& right[col-row+sizeE]==0
					&&left[col+row]==0){ 	//まだ効き筋がない
				if(aBoard[row]!=-1){		//Qを配置済み
					//colがaBoard[row]におきかわる
					down[aBoard[row]]
						=right[aBoard[row]-row+sizeE]
						=left[aBoard[row]+row]=0;
				}
				aBoard[row]=col;				//Qを配置
				down[col]
				  =right[col-row+sizeE]
					=left[col+row]=1;			//効き筋とする
				matched=true;						//配置した
				break;
			}
		}
		if(matched){								//配置済みなら
			row++;										//次のrowへ
			if(row==size){
				//print(size); //print()でTOTALを++しない
				TOTAL++;
				row--;
			}
		}else{
			if(aBoard[row]!=-1){
				int col=aBoard[row]; /** col の代用 */
				aBoard[row]=-1;
				down[col]
				  =right[col-row+sizeE]
				  =left[col+row]=0;
			}
			row--;										//バックトラック
		}
	}
}
// CPUR 再帰版 ロジックメソッド
void NQueenR(int row,int size){
	int sizeE=size-1;
	if(row==size){
		TOTAL++;
	}else{
		for(int col=0;col<size;col++){
			aBoard[row]=col;
			if(down[col]==0
					&& right[row-col+sizeE]==0
					&& left[row+col]==0){
				down[col]
				  =right[row-col+sizeE]
				  =left[row+col]=1;
				NQueenR(row+1,size);
				down[col]
					=right[row-col+sizeE]
					=left[row+col]=0;
			}
		}
	}
}
//hh:mm:ss.ms形式に処理時間を出力
void TimeFormat(clock_t utime,char* form){
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
//メインメソッド
int main(int argc,char** argv){
	/** CPUで実行 */
	bool cpu=true,cpur=false;
	/** CPURで実行 */
	//bool cpu=false,cpur=true;
	//
	if(cpu){
		printf("\n\n３．CPU 非再帰 バックトラック\n");
	}else if(cpur){
		printf("\n\n３．CPUR 再帰 バックトラック\n");
	}
	printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
	clock_t st;           //速度計測用
	char t[20];           //hh:mm:ss.msを格納
	int min=4;
	int targetN=17;
	//int targetN=4;
	for(int i=min;i<=targetN;i++){
		TOTAL=0;
		UNIQUE=0;
		st=clock();
		if(cpu){
			/** 非再帰は-1で初期化 */
			for(int j=0;j<=targetN;j++){
				aBoard[j]=-1;
			}
			NQueen(0,i);
		}
		if(cpur){
			/** 再帰は0で初期化 */
			for(int j=0;j<=targetN;j++){
				aBoard[j]=0;
			}
			NQueenR(0,i);
		}
		TimeFormat(clock()-st,t);
		printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
	}
	return 0;
}
