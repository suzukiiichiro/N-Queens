#include <stdio.h>
#include <time.h>
#include <math.h> 
int iSize=3;

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
int main(void) {
int x[iSize];
int y[iSize]; // x が元のデータ、y が回転後のデータ.
printf("元の配列\n");
for(int i=0;i<iSize;i++) {
      x[i] = pow(1,i); // load, and, store, 各32回
      dtob(x[i],iSize);
}
printf("90度回転したもの\n");
for(int i=0;i<iSize;i++) {
    int t = 0;
    for (int j = 0; j < iSize; j++)
        t |= ((x[j] >> i) & 1) << j; // x[j] の i ビット目を
    y[i] = t;                        // y[i] の j ビット目にする
}
for(int i=0;i< iSize;i++) {
      dtob(y[i],iSize);
}
}
