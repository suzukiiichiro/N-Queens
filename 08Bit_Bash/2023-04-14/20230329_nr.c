#include <stdio.h>
#include <stdlib.h>

static int TOTAL=0;

/* 
 * 特定の位置を指定して、以前の位置と衝突したかどうかを検出します。
 * 入力行は (1<<行) でエンコードする必要があります。
 * 衝突した場合は 1 を返し、そうでない場合は 0 を返します
 *
 */
int collide(int col,int row,int *board){
  int mask=0;
  int down=0;
  int left=0;
  int right=0;
  for(int i=0;i<col;i++){
    down=board[i];
    left=board[i]>>(col-i);
    right=board[i]<<(col-i);
    mask|=(down|left|right);
  }
  return mask & row;
}

void nonrecursive_solver(int size){
  /* マップは行配列で、それぞれに (1<<row) があります */
  int *board;
  board=malloc(sizeof(int) * size);
  for (int i=0;i<size;i++){
    board[i]=0;
  }
  /* 行のみが 2 のべき乗形式です */
  unsigned int row=1;
  unsigned int col=0;
  while (col<size){
    while (row<(1<<size)){
      /* もし衝突していなければ */
      if (!collide(col,row,board)){
        /* boardをリセット */
        board[col] |= row;
        row=1;
        break;
      }else{
        /* 衝突した場合は別のものを見つけます */
        row=row<<1;
      }
    }
    /* 利用可能なポジションが残っていない場合 */
    if (!board[col]){
      /* exit if in the first col */
      if (col==0){
        break;
      }else{
        /* 前のcolに戻り別の候補を見つける */
        col--;
        row=board[col]<<1;
        board[col]=0;
        continue;
      }
    }
    /* 利用可能なポジションが見つかった場合 */
    /* 最後の列の場合 */
    if (col==size-1){
      /* 解を増やす */
      TOTAL++;
      /* rowを次のrowに設定 */
      row=board[col]<<1;
      /* boardをリセット */
      board[col]=0;
      continue;
    }else{
      /* 最後の列でない場合は、次に進みます */
      col++;
    }
  }
  free(board);
}

void solve_queens(int size){
  nonrecursive_solver(size);
  printf("solutions=%d\n",TOTAL);
}
int main(){
  solve_queens(8);
  return 0;
}
