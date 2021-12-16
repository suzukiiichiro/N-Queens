/**
 BITで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)


２．配置フラグ


 コンパイルと実行
 $ gcc -O3 BIT02_N-Queen.c && ./a.out [-c|-r] 
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
//関数宣言
typedef struct {
	int left, down, right, bitmap, bit;
} rec;
//
void output(int size,rec *d);
void outputR(int size);
void NQueen(int size,int row,int down);
void NQueenR(int size,int row,int down);
//非再帰用出力
void output(int size,rec *d){
	printf("pattern %d\n", ++COUNT);
	for(int i=0;i<size;i++){
		for(int j=0;j<size;j++){
			putchar(d[i].bit&1<<j?'Q':'*');
		}
		putchar('\n');
	}
}
//再帰用出力
void outputR(int size){
	printf("pattern %d\n", ++COUNT);
	for(int i=0;i<size;i++){
		for(int j=0;j<size;j++){
			putchar(aBoard[i]&1<<j?'Q':'*');
		}
		putchar('\n');
	}
}
//CPU
void NQueen(int size,int row,int down){
	int MASK=((1<<size)-1);
	rec d[size];
	rec *p=d;
    p->down=0;
    p->bit=0;
	p->bitmap=MASK;
	while(1){
		if(p->bitmap){
			p->bit=-p->bitmap&p->bitmap;
			p->bitmap&=~p->bit;
			if (p-d<size-1) {
				rec*p0=p++;
				p->down=p0->down|p0->bit;
				p->bitmap=~(p->down)&MASK;
			}
			else output(size,d);
		}
		else if (--p<d) return;
	}
}
//CPUR
void NQueenR(int size,int row,int down) {
	int bit,bitmap;
	int MASK=((1<<size)-1);
	int sizeE=size-1;
	for(bitmap=~(down)&MASK;bitmap;bitmap&=~bit){
		aBoard[row]=bit=-bitmap&bitmap;
		if(row<sizeE){
			NQueenR(size,row+1,bit|down);
		}
		else outputR(size);
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
  if(cpu){ NQueen(size,0,0); }
  /**  再帰 */
  if(cpur){ NQueenR(size,0,0); }
  return 0;
}