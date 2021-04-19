//日本語
#include <stdio.h>

#define MAX 4

int aBoard[MAX]; //版の配列
int COUNT=0;     //カウント用
//関数宣言
void print(int size);
void NQueenR(int row,int size);
//
//出力
void print(int size){
  printf("%d: ",++COUNT);
  for(int j=0;j<size;j++){
    printf("%d ",aBoard[j]);
  }
  printf("\n");
}
int down[2*MAX-1];   //down: flagA   縦 配置フラグ
int right[2*MAX-1];  //right:flagB 斜め配置フラグ
int left[2*MAX-1]; //left:flagC 斜め配置フラグ
//
void NQueenR_03(int row,int size){
  int sizeE=size-1;
  if(row==size){
    print(size);
  }else{
    for(int col=aBoard[row]+1;col<size;col++){
      aBoard[row]=col;    //Qを配置
      if(down[col]==0
          && right[row-col+sizeE]==0
          && left[row+col]==0){
        down[col]
          =right[row-col+sizeE]
          =left[row+col]=1;
        NQueenR_03(row+1,size);
        down[col]
          =right[row-col+sizeE]
          =left[row+col]=0;
      }
      aBoard[row]=-1;     //空き地に戻す
    }
  }
}

//CPUR 再帰 ロジックメソッド
void NQueenR_02(int row,int size){
  if(row==size){
    print(size);
  }else{
    for(int col=aBoard[row]+1;col<size;col++){
      aBoard[row]=col;  //Qを配置
      if(down[col]==0){
        down[col]=1;
        NQueenR_02(row+1,size);
        down[col]=0;
      }
      aBoard[row]=-1;   //空き地に戻す
    }
  }
}
void NQueenR_01(int row,int size){
  if(row==size){
    print(size);
  }else{
    for(int col=aBoard[row]+1;col<size;col++){
      aBoard[row]=col;
      NQueenR_01(row+1,size);
      aBoard[row]=-1;
    }
  }
}

int main(){
  int N=4;
  //NQueenR_01(0,N);
  //NQueenR_02(0,N);
  NQueenR_03(0,N);
}

