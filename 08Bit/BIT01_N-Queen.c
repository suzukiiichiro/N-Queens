/**
 BITで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)


１．ブルートフォース


 コンパイルと実行
 $ gcc -O3 BIT01_N-Queen.c && ./a.out [-c|-r] 
                    -c:cpu 
                    -r cpu再帰 

**/


#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
//
#define MAX 5
//変数宣言
int aBoard[MAX]; 	//版の配列
int COUNT=0;		//カウント用
//巻数宣言
void output(int size);
void NQueen(int size,int row);
void NQueenR(int size,int row);
//出力
void output(int size){
	printf("pattern %d\n", ++COUNT);
	for(int i=0;i<size;i++){
		for(int j=0;j<size;j++){
			putchar(aBoard[i]&1<<j?'Q':'*');
		}
		putchar('\n');
	}
}
//CPU
void NQueen(int size,int row){
  bool matched;
  while(row>=0){
    matched=false;
    for(int col=aBoard[row]+1;col<size;col++){
      aBoard[row]=col;      //Qを配置
      matched=true;
      break;
    }
    if(matched){
      row++;
      if(row==size){
        output(size);
        row--;
      }
    }else{
      if(aBoard[row]!=-1){
        aBoard[row]=-1;
      }
      row--;
    }
  }
}
//CPUR 
void NQueenR(int size,int row){
	if(row==size){
		output(size);
	}else{
		for(int col=aBoard[row]+1;col<size;col++){
			aBoard[row]=col;	//Ｑを配置
			NQueenR(size,row+1);
			aBoard[row]=-1;		//空き地に戻す
		}
	}
}
//メイン
int main(int argc,char** argv){
  int size=5;
  bool cpu=false,cpur=false;
  int argstart=2;
  /** 起動パラメータの処理 */
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else{ cpur=true;}
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-r]\n",argv[0]);
    printf("  -c: CPU Without recursion\n");
    printf("  -r: CPUR Recursion\n");
  }
  // aBoard配列の初期化
  for(int i=0;i<size;i++){ aBoard[i]=-1; }
  /**  非再帰 */
  if(cpu){ NQueen(size,0); }
  /**  再帰 */
  if(cpur){ NQueenR(size,0); }
  return 0;
}