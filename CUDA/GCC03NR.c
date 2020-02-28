
// $ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC03NR.c && ./a.out [-c|-r]

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
//
#define MAX 27
//変数宣言
int aBoard[MAX];
int down[2*MAX-1];   //down:flagA 縦 配置フラグ
int right[2*MAX-1];  //right:flagB 斜め配置フラグ
int left[2*MAX-1]; //left:flagC 斜め配置フラグ
long TOTAL=0;
long UNIQUE=0;
//関数宣言
void TimeFormat(clock_t utime,char *form);
void NQueen(int row,int size);
void NQueenR(int row,int size);
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
void NQueen(int row,int size){
  int sizeE=size-1;
  bool matched;
  while(row>=0){
    matched=false;
    // １回目はaBoard[row]が-1なのでcolを0で初期化
    // ２回目以降はcolを<sizeまで右へシフト
    for(int col=aBoard[row]+1;col<size;col++){
      if(down[col]==0
          &&right[col-row+sizeE]==0
          &&left[col+row]==0){   //まだ効き筋がない
        if(aBoard[row]!=-1){    //Qを配置済み
          //colがaBoard[row]におきかわる
          down[aBoard[row]]
            =right[aBoard[row]-row+sizeE]
            =left[aBoard[row]+row]=0;
        }
        aBoard[row]=col;        //Qを配置
        down[col]
          =right[col-row+sizeE]
          =left[col+row]=1;      //効き筋とする
        matched=true;            //配置した
        break;
      }
    }
    if(matched){                //配置済みなら
      row++;                    //次のrowへ
      if(row==size){
        //print(size); //print()でTOTALを++しない
        TOTAL++;
        row--;
      }
    }else{
      if(aBoard[row]!=-1){
        int col=aBoard[row]; /** col の代用 */
        down[col]
          =right[col-row+sizeE]
          =left[col+row]=0;
        aBoard[row]=-1;
      }
      row--;                    //バックトラック
    }
  }
}
// CPUR 再帰版 ロジックメソッド
void NQueenR(int row,int size){
  int sizeE=size-1;
  if(row==size){
    TOTAL++;
  }else{
    //for(int col=0;col<size;col++){
    for(int col=aBoard[row]+1;col<size;col++){
      aBoard[row]=col;
      if(down[col]==0
          && right[row-col+sizeE]==0
          && left[row+col]==0){
        down[col]
          =right[row-col+sizeE]
          =left[row+col]=1;
        NQueenR(row+1,size);
        down[col]
          =right[row-col+sizeE]
          =left[row+col]=0;
      }
      aBoard[row]=-1;
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
    printf("\n\n３．CPU 非再帰 バックトラック\n");
  }else if(cpur){
    printf("\n\n３．CPUR 再帰 バックトラック\n");
  }
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  clock_t st;           //速度計測用
  char t[20];           //hh:mm:ss.msを格納
  int min=4;
  int targetN=17;
  for(int i=min;i<=targetN;i++){
    TOTAL=0;
    UNIQUE=0;
    st=clock();
    // aBoard配列の初期化
    for(int j=0;j<=targetN;j++){ aBoard[j]=-1; }
    /**  非再帰 */
    if(cpu){ NQueen(0,i); }
    /**  再帰 */
    if(cpur){ NQueenR(0,i); }
    TimeFormat(clock()-st,t);
    printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
  }
  return 0;
}
