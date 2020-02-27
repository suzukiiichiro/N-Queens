
// $ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC01NR.c && ./a.out

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
//
#define MAX 8
//変数宣言
int aBoard[MAX]; //版の配列
int COUNT=0;     //カウント用
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
      aBoard[row]=col;//Qを配置
      matched=true;
      break;
    }
    if(matched){
      row++;
      if(row==size){
        print(size);
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
//CPUR 再帰版ロジックメソッド
void NQueenR(int row,int size){
  if(row==size){            //SIZEは8で固定
    print(size);            //rowが8になったら出力
  }else{
    for(int col=aBoard[row]+1;col<size;col++){
      aBoard[row]=col;
      NQueenR(row+1,size);  // インクリメントしながら再帰
      aBoard[row]=-1;
    }
  }
}
//メインメソッド
int main(int argc,char** argv){
  int size=5;
  bool cpu=false,cpur=false;
  int argstart=2;

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
  for(int i=0;i<size;i++){ aBoard[i]=-1; }
  /**  非再帰 */
  if(cpu){ NQueen(0,size); }
  /**  再帰 */
  if(cpur){ NQueenR(0,size); }
  return 0;
}
