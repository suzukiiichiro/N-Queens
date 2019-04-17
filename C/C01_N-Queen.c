
/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイル
 $ gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm C01_N-Queen.c -o C01_N-Queen

 実行
 $ ./C01_N-Queen


 1. ブルートフォース　力任せ探索

 　全ての可能性のある解の候補を体系的に数え上げ、それぞれの解候補が問題の解とな
 るかをチェックする方法

   (※)各行に１個の王妃を配置する組み合わせを再帰的に列挙組み合わせを生成するだ
   けであって8王妃問題を解いているわけではありません

 実行結果
 :
 :
 16777209: 7 7 7 7 7 7 7 0
 16777210: 7 7 7 7 7 7 7 1
 16777211: 7 7 7 7 7 7 7 2
 16777212: 7 7 7 7 7 7 7 3
 16777213: 7 7 7 7 7 7 7 4
 16777214: 7 7 7 7 7 7 7 5
 16777215: 7 7 7 7 7 7 7 6
 16777216: 7 7 7 7 7 7 7 7
 **/
//
#include <stdio.h>
#include <time.h>
#define MAX 17
//
int SIZE=8;      //Nは8で固定
int COUNT=0;     //カウント用
int aBoard[MAX]; //版の配列
//出力用のメソッド
void print(){
	printf("%d: ",++COUNT);
	for(int j=0;j<SIZE;j++){
		printf("%d ",aBoard[j]);
	}
	printf("\n");
}
//ロジックメソッド
void NQueen(int row){
	if(row==SIZE){  //SIZEは8で固定
		print();      //rowが8になったら出力
	}else{
		for(int i=0;i<SIZE;i++){
			aBoard[row]=i;
			NQueen(row+1);  // インクリメントしながら再帰
		}
	}
}
//メインメソッド
int main(void){
	NQueen(0);//ロジックメソッドを0を渡して呼び出し
	return 0;
}

