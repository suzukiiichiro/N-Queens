#include <stdio.h>
#include <math.h>

/**
bash-5.1$ gcc sqrt.c && ./a.out
OUT:0
OUT:1
OUT:2
OUT:3
OUT:4
OUT:5
OUT:6
**/

int exec(int a){
  return log2(a);
}
int main(void){
  int a;
  a=1; printf("OUT:%d\n",exec(a));
  a=2; printf("OUT:%d\n",exec(a));
  a=4; printf("OUT:%d\n",exec(a));
  a=8; printf("OUT:%d\n",exec(a));
  a=16; printf("OUT:%d\n",exec(a));
  a=32; printf("OUT:%d\n",exec(a));
  a=64; printf("OUT:%d\n",exec(a));

  return 0;
}

