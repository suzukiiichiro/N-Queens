
// $ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC02NR.c && ./a.out

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
//
#define MAX 8
//変数宣言
int down[2*MAX-1]; 	//down: flagA 	縦 配置フラグ
int aBoard[MAX]; 		//版の配列
int COUNT=0;     		//カウント用
//関数宣言
void print(int size);
void NQueen(int row,int size);
void NQueenR(int row,int size);
//出力
void print(int size){
	printf("%d: ",++COUNT);
	for(int j=0;j<size;j++){
		printf("%d ",aBoard[j]);
	}
	printf("\n");
}
//CPU 非再帰版ロジックメソッド
void NQueen(int row,int size){
	bool matched;
	while(row>=0){
		matched=false;
		for(int col=aBoard[row]+1;col<size;col++){
			if(down[col]==0){			//downは効き筋ではない
				if(aBoard[row]!=-1){//Qは配置済み
					down[aBoard[row]]=0;//downの効き筋を外す
				}
				aBoard[row]=col;		//Qを配置
				down[col]=1;				//downは効き筋である
				matched=true;
				break;
			}
		}
		if(matched){
			row++;
			if(row==size){
				print(size);
				row--;
			}
		}else{									//置けるところがない
			if(aBoard[row]!=-1){
				int col=aBoard[row]; /** colの代用 */
				aBoard[row]=-1;			//空き地に戻す
				down[col]=0;				//downの効き筋を解除
			}
			row--;
		}
	}
}
//CPUR 再帰 ロジックメソッド
void NQueenR(int row,int size){
	if(row==size){
		print(size);
	}else{
		for(int col=0;col<size;col++){
			aBoard[row]=col;
			if(down[col]==0){
				down[col]=1;
				NQueenR(row+1,size);
				down[col]=0;
			}
		}
	}
}
//メインメソッド
int main(){
	int size=8;

	/** 非再帰 */
	for(int i=0;i<size;i++){ aBoard[i]=-1; }
	NQueen(0,size);

	/** 再帰 */
	//for(int i=0;i<size;i++){ aBoard[i]=0; }
	//NQueenR(0,size);
	return 0;
}
