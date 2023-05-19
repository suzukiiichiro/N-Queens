
// $ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC04NR.c && ./a.out [-c|-r]


/**
bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC04NR.c && ./a.out -r
４．CPUR 再帰 バックトラック＋対称解除法
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.01
12:        14200            1787            0.05
13:        73712            9233            0.28
14:       365596           45752            1.65
15:      2279184          285053           10.61
16:     14772512         1846955         1:12.29
17:     95815104        11977939         8:42.16


bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC04NR.c && ./a.out -c
４．CPU 非再帰 バックトラック＋対称解除法
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.01
12:        14200            1787            0.05
13:        73712            9233            0.28
14:       365596           45752            1.71
15:      2279184          285053           10.92
16:     14772512         1846955         1:13.64
17:     95815104        11977939         8:46.42
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
//
#define MAX 27
//
int aBoard[MAX];
int down[2*MAX-1];   //down:flagA 縦 配置フラグ
int right[2*MAX-1]; //right:flagB 斜め配置フラグ
int left[2*MAX-1];   //left:flagC 斜め配置フラグ
long TOTAL=0;
long UNIQUE=0;
int aT[MAX];         //aT:aTrial[]
int aS[MAX];         //aS:aScrath[]
//関数宣言
void TimeFormat(clock_t utime,char *form);
void rotate(int chk[],int scr[],int n,int neg);
void vMirror(int chk[],int n);
int intncmp(int lt[],int rt[],int n);
int symmetryOps(int size);
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
//回転
void rotate(int chk[],int scr[],int n,int neg){
  int k=neg ? 0 : n-1;
  int incr=(neg ? +1 : -1);
  for(int j=0;j<n;k+=incr){
    scr[j++]=chk[k];
  }
  k=neg ? n-1 : 0;
  for(int j=0;j<n;k-=incr){
    chk[scr[j++]]=k;
  }
}
//反転
void vMirror(int chk[],int n){
  for(int j=0;j<n;j++){
    chk[j]=(n-1)-chk[j];
  }
}
//
int intncmp(int lt[],int rt[],int n){
  int rtn=0;
  for(int k=0;k<n;k++){
    rtn=lt[k]-rt[k];
    if(rtn!=0){
      break;
    }
  }
  return rtn;
}
//対称解除法
int symmetryOps(int size){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<size;i++){
    aT[i]=aBoard[i];
  }
  //時計回りに90度回転
  rotate(aT,aS,size,0);
  int k=intncmp(aBoard,aT,size);
  if(k>0) return 0;
  if(k==0){
    nEquiv=1;
  }else{
    //時計回りに180度回転
    rotate(aT,aS,size,0);
    k=intncmp(aBoard,aT,size);
    if(k>0) return 0;
    if(k==0){
      nEquiv=2;
    }else{
      //時計回りに270度回転
      rotate(aT,aS,size,0);
      k=intncmp(aBoard,aT,size);
      if(k>0){
        return 0;
      }
      nEquiv=4;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<size;i++){
    aT[i]=aBoard[i];
  }
  //垂直反転
  vMirror(aT,size);
  k=intncmp(aBoard,aT,size);
  if(k>0){
    return 0;
  }
  //-90度回転 対角鏡と同等
  if(nEquiv>1){
    rotate(aT,aS,size,1);
    k=intncmp(aBoard,aT,size);
    if(k>0){
      return 0;
    }
    //-180度回転 水平鏡像と同等
    if(nEquiv>2){
      rotate(aT,aS,size,1);
      k=intncmp(aBoard,aT,size);
      if(k>0){
        return 0;
      }
      //-270度回転 反対角鏡と同等
      rotate(aT,aS,size,1);
      k=intncmp(aBoard,aT,size);
      if(k>0){
        return 0;
      }
    }
  }
  return nEquiv*2;
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
          && right[col-row+sizeE]==0
          && left[col+row]==0){ //まだ効き筋がない
        if(aBoard[row]!=-1){    //Qを配置済み
          //colがaBoard[row]におきかわる
          down[aBoard[row]]
            =right[aBoard[row]-row+sizeE]
            =left[aBoard[row]+row]=0;
        }
        aBoard[row]=col;        //Qを配置
        down[col]
          =right[col-row+sizeE]
          =left[col+row]=1;     //効き筋とする
        matched=true;           //配置した
        break;
      }
    }
    if(matched){                //配置済みなら
      row++;                    //次のrowへ
      if(row==size){
        //print(size); //print()でTOTALを++しない
        /** 対称解除法の導入 */
        int s=symmetryOps(size);
        if(s!=0){
          UNIQUE++;             //ユニーク解を加算
          TOTAL+=s;   //対称解除で得られた解数を加算
        }
        // TOTAL++;
        /** 対称解除法の導入 */
        row--;
      }
    }else{
      if(aBoard[row]!=-1){
        int col=aBoard[row];    /** col の代用 */
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
    /** 対称解除法の導入 */
    int s=symmetryOps(size);
    if(s!=0){
      UNIQUE++;       //ユニーク解を加算
      TOTAL+=s;       //対称解除で得られた解数を加算
    }
    // TOTAL++;
    /** 対称解除法の導入 */
  }else{
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
    printf("\n\n４．CPU 非再帰 バックトラック＋対称解除法\n");
  }else if(cpur){
    printf("\n\n４．CPUR 再帰 バックトラック＋対称解除法\n");
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
