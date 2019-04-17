/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイル
 $ gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm C02_N-Queen.c -o C02_N-Queen

 実行
 $ ./C02_N-Queen


 ２．配置フラグ（制約テスト高速化）

   パターンを生成し終わってからチェックを行うのではなく、途中で制約を満たさな
   い事が明らかな場合は、それ以降のパターン生成を行わない。
  「手を進められるだけ進めて、それ以上は無理（それ以上進めても解はない）という
  事がわかると一手だけ戻ってやり直す」という考え方で全ての手を調べる方法。
  (※)各行列に一個の王妃配置する組み合わせを再帰的に列挙分枝走査を行っても、組
  み合わせを列挙するだけであって、8王妃問題を解いているわけではありません。

 実行結果
 :
 :
 40312: 7 6 5 4 2 1 3 0
 40313: 7 6 5 4 2 3 0 1
 40314: 7 6 5 4 2 3 1 0
 40315: 7 6 5 4 3 0 1 2
 40316: 7 6 5 4 3 0 2 1
 40317: 7 6 5 4 3 1 0 2
 40318: 7 6 5 4 3 1 2 0
 40319: 7 6 5 4 3 2 0 1
 40320: 7 6 5 4 3 2 1 0
 */
#include <stdio.h>
#include <time.h>
#define MAX 17
//
int SIZE=8;       //Nを8に固定
int COUNT=0;
int aBoard[MAX];
int fA[2*MAX-1];	//縦列にクイーンを一つだけ配置
//
void print(){
	printf("%d: ",++COUNT);
	for(int j=0;j<SIZE;j++){
		printf("%d ",aBoard[j]);
	}
	printf("\n");
}
//
void NQueen(int row){
	if(row==SIZE-1){      //0から始まるのでN=8から1を引きます
		print();            //出力
	}else{
		for(int i=0;i<SIZE;i++){
			aBoard[row]=i;
			if(fA[i]==0){     //縦列にクイーンがない場合
				fA[i]=1;
				NQueen(row+1);  //1を足して再帰
				fA[i]=0;
			}
		}
	}
}
//
int main(void){
	NQueen(0);
	return 0;
}
