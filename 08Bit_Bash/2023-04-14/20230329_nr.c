#include <stdio.h>
#include <stdlib.h>

static int TOTAL=0;

/* 
 * $BFCDj$N0LCV$r;XDj$7$F!"0JA0$N0LCV$H>WFM$7$?$+$I$&$+$r8!=P$7$^$9!#(B
 * $BF~NO9T$O(B (1<<$B9T(B) $B$G%(%s%3!<%I$9$kI,MW$,$"$j$^$9!#(B
 * $B>WFM$7$?>l9g$O(B 1 $B$rJV$7!"$=$&$G$J$$>l9g$O(B 0 $B$rJV$7$^$9(B
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
  /* $B%^%C%W$O9TG[Ns$G!"$=$l$>$l$K(B (1<<row) $B$,$"$j$^$9(B */
  int *board;
  board=malloc(sizeof(int) * size);
  for (int i=0;i<size;i++){
    board[i]=0;
  }
  /* $B9T$N$_$,(B 2 $B$N$Y$->h7A<0$G$9(B */
  unsigned int row=1;
  unsigned int col=0;
  while (col<size){
    while (row<(1<<size)){
      /* $B$b$7>WFM$7$F$$$J$1$l$P(B */
      if (!collide(col,row,board)){
        /* board$B$r%j%;%C%H(B */
        board[col] |= row;
        row=1;
        break;
      }else{
        /* $B>WFM$7$?>l9g$OJL$N$b$N$r8+$D$1$^$9(B */
        row=row<<1;
      }
    }
    /* $BMxMQ2DG=$J%]%8%7%g%s$,;D$C$F$$$J$$>l9g(B */
    if (!board[col]){
      /* exit if in the first col */
      if (col==0){
        break;
      }else{
        /* $BA0$N(Bcol$B$KLa$jJL$N8uJd$r8+$D$1$k(B */
        col--;
        row=board[col]<<1;
        board[col]=0;
        continue;
      }
    }
    /* $BMxMQ2DG=$J%]%8%7%g%s$,8+$D$+$C$?>l9g(B */
    /* $B:G8e$NNs$N>l9g(B */
    if (col==size-1){
      /* $B2r$rA}$d$9(B */
      TOTAL++;
      /* row$B$r<!$N(Brow$B$K@_Dj(B */
      row=board[col]<<1;
      /* board$B$r%j%;%C%H(B */
      board[col]=0;
      continue;
    }else{
      /* $B:G8e$NNs$G$J$$>l9g$O!"<!$K?J$_$^$9(B */
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
