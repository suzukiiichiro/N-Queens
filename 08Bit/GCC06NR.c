
//$ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC06NR.c && ./a.out [-c|-r]

/**
bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC06NR.c && ./a.out -r
６．CPUR 再帰 バックトラック＋ビットマップ
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               0            0.00
 5:           10               0            0.00
 6:            4               0            0.00
 7:           40               0            0.00
 8:           92               0            0.00
 9:          352               0            0.00
10:          724               0            0.00
11:         2680               0            0.00
12:        14200               0            0.01
13:        73712               0            0.04
14:       365596               0            0.23
15:      2279184               0            1.40
16:     14772512               0            9.37
17:     95815104               0         1:05.71


bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC06NR.c && ./a.out -c
６．CPU 非再帰 バックトラック＋ビットマップ
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               0            0.00
 5:           10               0            0.00
 6:            4               0            0.00
 7:           40               0            0.00
 8:           92               0            0.00
 9:          352               0            0.00
10:          724               0            0.00
11:         2680               0            0.00
12:        14200               0            0.01
13:        73712               0            0.04
14:       365596               0            0.24
15:      2279184               0            1.47
16:     14772512               0            9.75
17:     95815104               0         1:08.46
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
//
#define MAX 27
//
long TOTAL=0;
long UNIQUE=0;
// int aBoard[MAX];
// int aT[MAX];       	//aT:aTrial[]
// int aS[MAX];       	//aS:aScrath[]
//関数宣言
void TimeFormat(clock_t utime,char *form);
void NQueenR(int size,int mask,int row,int left,int down,int right);
void NQueen(int size,int mask,int row);
//hh:mm:ss.ms形式に処理時間を出力
void TimeFormat(clock_t utime,char* form){
  int dd,hh,mm;
  float ftime,ss;
  ftime=(float)utime/CLOCKS_PER_SEC;
  mm=(int)ftime/60;
  ss=ftime-(int)(mm*60);
  dd=mm/(24*60);
  mm=mm%(24*60);
  hh=mm/60;
  mm=mm%60;
  if(dd)
    sprintf(form,"%4d %02d:%02d:%05.2f",dd,hh,mm,ss);
  else if(hh)
    sprintf(form,"     %2d:%02d:%05.2f",hh,mm,ss);
  else if(mm)
    sprintf(form,"        %2d:%05.2f",mm,ss);
  else
    sprintf(form,"           %5.2f",ss);
}
//CPU 非再帰版 ロジックメソッド
void NQueen(int size,int mask,int row){
  register int aStack[size];
  register int* pnStack;
  register int bit;
  register int bitmap;
  register int sizeE=size-1;
  register int down[size],right[size],left[size];
  aStack[0]=-1; 
  pnStack=aStack+1;
  bit=0;
  bitmap=mask;
  down[0]=left[0]=right[0]=0;
  while(true){
    if(bitmap){
      bitmap^=bit=(-bitmap&bitmap); 
      if(row==sizeE){
        TOTAL++;
        bitmap=*--pnStack;
        --row;
        continue;
      }else{
        int n=row++;
        left[row]=(left[n]|bit)<<1;
        down[row]=down[n]|bit;
        right[row]=(right[n]|bit)>>1;
        *pnStack++=bitmap;
        bitmap=mask&~(left[row]|down[row]|right[row]);
        continue;
      }
    }else{ 
      bitmap=*--pnStack;
      if(pnStack==aStack){ break ; }
      --row;
      continue;
    }
  }
}
//CPUR 再帰版　ロジックメソッド
void NQueenR(int size,int mask,int row,int left,int down,int right){
  int bitmap=0;
  int bit=0;
  if(row==size){
    TOTAL++;
  }else{
    bitmap=(mask&~(left|down|right));
    while(bitmap){
      bitmap^=bit=(-bitmap&bitmap);
      NQueenR(size,mask,row+1,(left|bit)<<1, down|bit,(right|bit)>>1);
    }
  }
}
//メインメソッド
int main(int argc,char** argv){
  bool cpu=false,cpur=false;
  int argstart=2;
  /** 起動パラメータの処理 */
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else{ cpur=true;}
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g]\n",argv[0]);
    printf("  -c: CPU Without recursion\n");
    printf("  -r: CPUR Recursion\n");
  }
  if(cpu){
    printf("\n\n６．CPU 非再帰 バックトラック＋ビットマップ\n");
  }else if(cpur){
    printf("\n\n６．CPUR 再帰 バックトラック＋ビットマップ\n");
  }
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  clock_t st;           //速度計測用
  char t[20];           //hh:mm:ss.msを格納
  int min=4;
  int targetN=17;
  int mask=0;
  for(int i=min;i<=targetN;i++){
    TOTAL=0;
    UNIQUE=0;
    mask=((1<<i)-1);
    st=clock();
    if(cpu){
      // aBaordを使用しないので
      // 初期化の必要はありません
      // for(int j=0;j<=targetN;j++){
      // 	aBoard[j]=-1;
      // }
      NQueen(i,mask,0);
    }
    if(cpur){
      // aBaordを使用しないので
      // 初期化の必要はありません
      // for(int j=0;j<=targetN;j++){
      // 	aBoard[j]=j;
      // }
      NQueenR(i,mask,0,0,0,0);
    }
    TimeFormat(clock()-st,t);
    printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
  }
  return 0;
}
