#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#define THREAD_NUM		96
#define MAX 27
//変数宣言
int down[2*MAX-1];  //CPU down:flagA 縦 配置フラグ　
int left[2*MAX-1];  //CPU left:flagB 斜め配置フラグ　
int right[2*MAX-1]; //CPU right:flagC 斜め配置フラグ　
int board[MAX];
int TOTAL;
int COUNT2;
//非再帰版ミラーロジック
void solve_nqueen(int size,int mask, int row,int h_left,int h_down,int h_right)
{
  int bitmap[size];
	left[row]=h_left;
	down[row]=h_down;
	right[row]=h_right;
  int bit;
  //固定していれた行より上はいかない
	bitmap[row]=mask&~(left[row]|down[row]|right[row]);
  while(row>0){//row=1 row>=1, row=2 row>=2
    if(bitmap[row]==0){
      --row;
    }else{
      //bitmap[row]^=board[row]=bit=(-bitmap[row]&bitmap[row]); 
      bit=(-bitmap[row]&bitmap[row]); 
      bitmap[row]=bitmap[row]^bit;
      board[row]=bit;
      if((bit&mask)!=0){
        if(row==(size-1)){
          COUNT2++;
          --row;
        }else{
          int n=row++;
          left[row]=(left[n]|bit)<<1;
          down[row]=down[n]|bit;
          right[row]=(right[n]|bit)>>1;
          bitmap[row]=mask&~(left[row]|down[row]|right[row]);
        }
      }else{
         --row;
      }
    }  
  }
}
//非再帰版ミラー
void mirror_NR(int size,int mask)
{
  int bit=0;
  /*
   偶数、奇数ともに右半分にクイーンを置く 
   00001111
  */
  /*
   奇数の場合
   奇数は中央にもクイーンを置く
   00100
   １行目の左側半分にクイーンを置けないようにする
   11100

   1行目にクイーンが中央に置かれた場合は
   00100
   2行目の左側半分にクイーンを置けない
   00100
   11100
   さらに1行目のdown,rightもクイーンを置けないので(size/2)-1となる
   11100

   偶数の場合
   １行目の左側半分にクイーンを置けないようにする
   1100
  */
  int limit=size%2 ? size/2-1 : size/2;
  for(int i=0;i<size/2;i++){
    bit=(1<<i);         
    board[0]=bit;       //1行目にクイーンを置く
    solve_nqueen(size,mask,1,bit<<1,bit,bit>>1);
  }
  if(size%2){
    //1行目はクイーンを中央に置く
    bit=(1<<( (size-1)/2));
    board[0]=(1<<((size-1)/2) );
    int left=bit<<1;
    int down=bit;
    int right=bit>>1;

    for(int i=0;i<limit;i++){
      bit=(1<<i);         
      board[1]=bit;       //2行目にクイーンを置く
      solve_nqueen(size,mask,2,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
  TOTAL=COUNT2<<1;        // 倍にする
}

//再帰版ミラー ロジック
void solve_nqueenr(int size,int mask, int row,int left,int down,int right)
{
 int bit=0;
 int bitmap=(mask&~(left|down|right)); //クイーンが配置可能な位置を表す
 if(row==(size-1)){
   if(bitmap){
     COUNT2++;
   }
  }else{
    while(bitmap){
      bit=(-bitmap&bitmap); // 一番右のビットを取り出す
      bitmap=bitmap^bit;    //配置可能なパターンが一つずつ取り出される
      board[row]=bit;       //Qを配置
      solve_nqueenr(size,mask,row+1,(left|bit)<<1, down|bit,(right|bit)>>1);
    }
  }
}
//再帰版ミラー
void mirror_R(int size,int mask)
{
  int bit=0;
  /*
   偶数、奇数ともに右半分にクイーンを置く 
   00001111
  */
  /*
   奇数の場合
   奇数は中央にもクイーンを置く
   00100
   １行目の左側半分にクイーンを置けないようにする
   11100

   1行目にクイーンが中央に置かれた場合は
   00100
   2行目の左側半分にクイーンを置けない
   00100
   11100
   さらに1行目のdown,rightもクイーンを置けないので(size/2)-1となる
   11100

   偶数の場合
   １行目の左側半分にクイーンを置けないようにする
   1100
  */
  int limit=size%2 ? size/2-1 : size/2;
  for(int i=0;i<size/2;i++){
    bit=(1<<i);         
    board[0]=bit;       //1行目にクイーンを置く
    solve_nqueenr(size,mask,1,bit<<1,bit,bit>>1);
  }
  if(size%2){
    //1行目はクイーンを中央に置く
    bit=(1<<( (size-1)/2));
    board[0]=(1<<((size-1)/2) );
    int left=bit<<1;
    int down=bit;
    int right=bit>>1;

    for(int i=0;i<limit;i++){
      bit=(1<<i);         
      board[1]=bit;       //2行目にクイーンを置く
      solve_nqueenr(size,mask,2,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
  TOTAL=COUNT2<<1;        // 倍にする
}
//
//メインメソッド
int main(int argc,char** argv)
{
  COUNT2=0;                 //グローバル
  int size=5;
  int mask=(1<<size)-1;
//  mirror_R(size,mask);      //再帰版ミラー
   mirror_NR(size,mask);  //非再帰版ミラー
  printf("%2d:%13d\n",size,TOTAL);
  return 0;
}

