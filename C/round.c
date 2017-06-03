#include <stdio.h>
#include <time.h>
#include <math.h> 
#include <stdlib.h>

int iSize=10;
int rh(int a,int sz){
  int tmp;
  int i;
  sz=sz-1;
  /* ビット入替 */
  tmp = 0;
  for( i = 0; i <= sz; i++ )
  {
    if( a & ( 1 << i ) )
    {
      tmp |= ( 1 << ( sz - i ) );
    }
  }
  a = tmp;
  return tmp;
}
void revHorzBitmap(int abefore[],int aafter[]){
for(int i=0;i< iSize;i++) {
 int score=abefore[i];
 aafter[i]=rh(score,iSize);
}

}

void dtob(int score,int size) {
  int bit = 1, i;
  char c[size];
 
  for (i = 0; i < size; i++) {
    if (score & bit)
      c[i] = '1';
    else
      c[i] = '0';
    bit <<= 1;
  }
  // 計算結果の表示
  for ( i = size - 1; i >= 0; i-- ) {
      putchar(c[i]);
  }
  printf("\n");
}
void rotateBitmap90(int abefore[],int aafter[]){
  for(int i=0;i<iSize;i++) {
    int t = 0;
    for (int j = 0; j < iSize; j++)
        t |= ((abefore[j] >> i) & 1) << j; // x[j] の i ビット目を
    aafter[i] = t;                        // y[i] の j ビット目にする
  }
}
int main(void) {
int x[iSize];
int y[iSize]; // x が元のデータ、y が回転後のデータ.
int z[iSize]; // x が元のデータ、y が回転後のデータ.
//int z[iSize];
printf("元の配列\n");
for(int i=0;i<iSize;i++) {
      x[i] = pow(1,i)+rand()%2; // load, and, store, 各32回
      dtob(x[i],iSize);
}
printf("90度回転したもの\n");
rotateBitmap90(x,y);
for(int i=0;i< iSize;i++) {
      dtob(y[i],iSize);
}
printf("左右反転\n");
  /* ビット数を取得 */
revHorzBitmap(y,z);
for(int i=0;i< iSize;i++) {
      dtob(z[i],iSize);
}
}
