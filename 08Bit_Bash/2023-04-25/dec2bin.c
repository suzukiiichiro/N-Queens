#include <stdio.h>
 
int main(void)
{
  /* 変数の宣言 */
  int i, decimal;
  int binary[32];
 
  /* 10進数の入力 */
  printf("10進数 = ");
  scanf("%d", &decimal);
 
  /* 10進数→2進数の変換 */
  for(i=0;decimal>0;i++){
    binary[i] = decimal % 2;
    decimal = decimal / 2;
  }
 
  /* 2進数の出力 */
  printf(" 2進数 = ");
  while( i>0 ){
    printf("%d", binary[--i]);
  }
  printf("\n");
 
  return 0;
}

