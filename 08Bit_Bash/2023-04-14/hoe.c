#include <stdio.h>

int main(void){
  int size=5;
  int board[size];
  int* tBoard;
  for(int i=0;i<size;i++){
    //board[i]=-1;
    board[i]=i;
  }
  printf("--------------\n");
  printf("board+1\n--------------\n");
  tBoard=board+1;

  for(int j=0;j<size;j++){
    printf("board[%d]:%d tBoard[%d]:%d\n",j,board[j],j,tBoard[j]);
  }


  printf("--------------\n");
  printf("board[3]=10\n--------------\n");
  board[3]=10;
  for(int j=0;j<size;j++){
    printf("board[%d]:%d tBoard[%d]:%d\n",j,board[j],j,tBoard[j]);
  }

  printf("--------------\n");
  printf("tBoard--\n--------------\n");
  tBoard--;
  for(int j=0;j<size;j++){
    printf("board[%d]:%d tBoard[%d]:%d\n",j,board[j],j,tBoard[j]);
  }

  printf("--------------\n");
  printf("tBoard++\n--------------\n");
  tBoard++;
  for(int j=0;j<size;j++){
    printf("board[%d]:%d tBoard[%d]:%d\n",j,board[j],j,tBoard[j]);
  }
  printf("--------------\n");
  printf("board:%ls\n",board);
  return 0;
}
