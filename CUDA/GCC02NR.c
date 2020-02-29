
// $ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC02NR.c && ./a.out [-c|-r]

/**
 ２．CPU 非再帰 配置フラグ（制約テスト高速化）
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

 ２．CPUR 再帰 配置フラグ（制約テスト高速化）
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
#include <stdlib.h>
#include <stdbool.h>
//
#define MAX 8
//変数宣言
int down[2*MAX-1];   //down: flagA   縦 配置フラグ
int aBoard[MAX];     //版の配列
int COUNT=0;         //カウント用
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
      if(down[col]==0){      //downは効き筋ではない
        if(aBoard[row]!=-1){ //Qは配置済み
          down[aBoard[row]]=0;//downの効き筋を外す
        }
        aBoard[row]=col;     //Qを配置
        down[col]=1;         //downは効き筋である
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
    }else{                   //置けるところがない
      if(aBoard[row]!=-1){
        int col=aBoard[row]; /** colの代用 */
        down[col]=0;         //downの効き筋を解除
        aBoard[row]=-1;      //空き地に戻す
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
    for(int col=aBoard[row]+1;col<size;col++){
      aBoard[row]=col;
      if(down[col]==0){
        down[col]=1;
        NQueenR(row+1,size);
        down[col]=0;
      }
      aBoard[row]=-1;
    }
  }
}
//メインメソッド
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
    printf("Usage: %s [-c|-g]\n",argv[0]);
    printf("  -c: CPU Without recursion\n");
    printf("  -r: CPUR Recursion\n");
  }
  // aBoard配列の初期化
  for(int i=0;i<size;i++){ aBoard[i]=-1; }
  /**  非再帰 */
  if(cpu){ NQueen(0,size); }
  /**  再帰 */
  if(cpur){ NQueenR(0,size); }
  return 0;
}
