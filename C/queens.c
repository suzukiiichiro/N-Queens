#define N 8
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>

typedef enum { false,
               true } bool;
void continuar() {
  printf("\n Pressione [Enter] para outra solucao.");
  while(getchar() != '\n')
    ;
}
void _wait(unsigned int secs) {
  unsigned int retTime= time(0) + secs;
  while(time(0) < retTime) {
    ;  // Loop until it arrives.
  }
}
int color(int i, int j) {
  return (i + j) % 2;
}
int countSLT() {
  static int k= 0;
  return k++;
}
#define COLOR_BGW "\x1b[47m"
#define COLOR_BGB "\x1b[100m"
#define COLOR_BLK "\x1b[30m"
#define COLOR_RESET "\x1b[0m"
void printBoard(int board[N][N], int delay) {
  system("clear");
  int i, j;
  printf("\n N-QUEENS\n\n");
  for(i= 0; i < N; i++) {
    printf(" %d ", i + 1);
    for(j= 0; j < N; j++)
      if(!color(i, j)) {
        if(board[i][j])
          printf(COLOR_BGW COLOR_BLK " \u265B " COLOR_RESET);
        else
          printf(COLOR_BGW "   " COLOR_RESET);
      } else {
        if(board[i][j])
          printf(COLOR_BGB COLOR_BLK " \u265B " COLOR_RESET);
        else
          printf(COLOR_BGB "   " COLOR_RESET);
      }
    printf("\n");
  }
  printf("   ");
  for(i= 0; i < N; i++) {
    printf(" %c ", i + 97);
  }
  printf("\n");
}
bool valida(int board[N][N], int row, int col) {
  int i, j;
  for(i= 0; i < col; i++)
    if(board[row][i])
      return false;
  for(i= row, j= col; i >= 0 && j >= 0; i--, j--)
    if(board[i][j])
      return false;
  for(i= row, j= col; j >= 0 && i < N; i++, j--)
    if(board[i][j])
      return false;
  return true;
}

bool solve2(int board[N][N], int col) {
  int i;
  if(col >= N) return true;
  for(i= 0; i < N; i++) {
    if(valida(board, i, col)) {
      board[i][col]= 1;
      if(solve2(board, col + 1))
        return true;
      board[i][col]= 0;
    }
  }
  return false;
}

bool solve(int board[N][N], int col, int delay) {
  int  i;
  bool solucao= false;
  if(col == N) {
    printBoard(board, delay);
    printf("\n Solucao: %d", countSLT());
    continuar();
    return true;
  }
  for(i= 0; i < N; i++) {
    if(valida(board, i, col)) {
      board[i][col]= 1;
      printBoard(board, delay);
      if(delay) _wait(1);
      solucao      = solve(board, col + 1, delay) || solucao;
      board[i][col]= 0;
      printBoard(board, delay);
      if(delay) _wait(1);
    }
  }
  return solucao;
}
void callNQ() {
  system("clear");
  int  board[N][N];
  bool delay= false;
  char ask;
  memset(board, 0, sizeof(board));
  printf("\n Usar o delay para visualizar o backtracking? [s/n]");
  ask= getchar();
  if(ask == 's') delay= true;
  if(!solve(board, 0, delay)) {
    printf("Nao existe uma solucao.\n");
    return;
  }
  return;
}
int main(void) {
  callNQ();
  return 0;
}
