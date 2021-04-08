#include <stdio.h>
#define MAX_BIT 32 
//
//2進数を10進数に変換
int bin2dec(int binary)
{
	int decimal=0;
	int base=1;
  while(binary>0){
    decimal = decimal + ( binary % 10 ) * base;
    binary = binary / 10;
    base = base * 2;
  }
	return decimal;
}
//10進数を2進数に変換
int dec2bin(int decimal,int*aBoard)
{
	int n=0;
	int a;
	while(n<MAX_BIT){
		/* 第n桁のみ1の値を算出 */
		a=(unsigned int)1<<n;
		/* &演算で第n桁の値取得 */
		if((decimal&a)==0){
			aBoard[n]=0;
		}else{
			aBoard[n]=1;
		}
		/* 次の桁へ */
		n++;
	}
  return n;
}
int main(void)
{
  /* 
   * 10進数の入力 
   **/
	int binary;

  printf("2進数 = ");
  scanf("%d", &binary);
  printf("10進数 = %d\n", bin2dec(binary));

  /* 
   * 2進数の入力 
   **/
	int decimal;

  int aBoard[MAX_BIT];
  printf("10進数 = ");
  scanf("%d", &decimal);
  //aBoardの配列の数だけ返却
  int n=dec2bin(decimal,aBoard);
  /* 求めた２進数を表示 */
  printf("2進数 = \n");
  /* 第i桁を表示 */
  for(int i=0;i<n;i++){
    printf("%u",aBoard[n-1-i]);
  }
  printf("\n");
  return 0;
}

