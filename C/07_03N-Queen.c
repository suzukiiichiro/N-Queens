/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイル
 $ gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm 07_03N-Queen.c -o 03N-Queen

 実行
 $ ./03N-Queen

 ３．バックトラック

   　各列、対角線上にクイーンがあるかどうかのフラグを用意し、途中で制約を満た
   さない事が明らかな場合は、それ以降のパターン生成を行わない。
   　各列、対角線上にクイーンがあるかどうかのフラグを用意することで高速化を図る。
   　これまでは行方向と列方向に重複しない組み合わせを列挙するものですが、王妃
   は斜め方向のコマをとることができるので、どの斜めライン上にも王妃をひとつだ
   けしか配置できない制限を加える事により、深さ優先探索で全ての葉を訪問せず木
   を降りても解がないと判明した時点で木を引き返すということができます。

 実行結果
 	 :
 	 :
	83: 6 1 5 2 0 3 7 4
	84: 6 2 0 5 7 4 1 3
	85: 6 2 7 1 4 0 5 3
	86: 6 3 1 4 7 0 2 5
	87: 6 3 1 7 5 0 2 4
	88: 6 4 2 0 5 7 1 3
	89: 7 1 3 0 6 4 2 5
	90: 7 1 4 2 0 6 3 5
	91: 7 2 0 5 1 4 6 3
	92: 7 3 0 2 5 1 6 4
 */

#include <stdio.h>
#include <time.h>
#define MAX 24

int SIZE=8;
int COUNT=0;
int aBoard[MAX];
int fA[2*MAX-1];	//縦列にクイーンを一つだけ配置
int fB[2*MAX-1];	//斜め列にクイーンを一つだけ配置
int fC[2*MAX-1];	//斜め列にクイーンを一つだけ配置

void print();
void NQueen(int row);

void print(){
	printf("%d: ",++COUNT);
	for(int j=0;j<SIZE;j++){
		printf("%d ",aBoard[j]);
	}
	printf("\n");
}
void NQueen(int row){
	if(row==SIZE){
		print();
	}else{
		for(int i=0;i<SIZE;i++){
			aBoard[row]=i;
			if(fA[i]==0&&fB[row-i+(SIZE-1)]==0&&fC[row+i]==0){
				fA[i]=fB[row-i+(SIZE-1)]=fC[row+i]=1;
				NQueen(row+1);
				fA[i]=fB[row-i+(SIZE-1)]=fC[row+i]=0;
			}
		}
	}
}
int main(void){
	NQueen(0);
	return 0;
}
