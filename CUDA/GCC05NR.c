
// $ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC05NR.c && ./a.out [-c|-r]

/**
bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC05NR.c && ./a.out -r
５．CPUR 再帰 バックトラック＋対称解除法＋枝刈りと最適化
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.00
12:        14200            1787            0.01
13:        73712            9233            0.09
14:       365596           45752            0.48
15:      2279184          285053            3.22
16:     14772512         1846955           20.18
17:     95815104        11977939         2:32.19


bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC05NR.c && ./a.out -c
５．CPU 非再帰 バックトラック＋対称解除法＋枝刈りと最適化
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.01
12:        14200            1787            0.03
13:        73712            9233            0.16
14:       365596           45752            0.90
15:      2279184          285053            6.19
16:     14772512         1846955           38.90
17:     95815104        11977939         5:00.78
*/
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
  bool matched;
  int sizeE=size-1;
  while(row>=0){
    matched=false;
    int lim=(row!=0)?size:(size+1)/2;
    for(int col=aBoard[row]+1;col<lim;col++){
      if(down[col]==0
          && right[col-row+sizeE]==0
          && left[col+row]==0){
        if(aBoard[row]!=-1){
          down[aBoard[row]]
            =right[aBoard[row]-row+sizeE]
            =left[aBoard[row]+row]=0;
        }
        aBoard[row]=col;
        down[col]
          =right[col-row+sizeE]
          =left[col+row]=1;
        matched=true;
        break;
      }
    }
    if(matched){
      row++;
      if(row==size){
        if(aBoard[row]!=-1){
          if(down[aBoard[row]]==1
              || right[aBoard[row]-row+sizeE]==1
              || left[aBoard[row]+row]==1){
            return;
          }
        }
        int s=symmetryOps(size);
        if(s!=0){
          UNIQUE++;
          TOTAL+=s;
        }
        row--;
      }
    }else{
      if(aBoard[row]>=0){
        int col=aBoard[row];
        down[col]
          =right[col-row+sizeE]
          =left[col+row]=0;
        aBoard[row]=-1;
      }
      row--;
    }
  }
}
//CPUR 再帰版　ロジックメソッド
void NQueenR(int row,int size){
  /**
   * main()をチェック
   * 【注意】初期化が前のステップと異なります
   * 前のステップでは 0　で初期化しいました
   * for(int j=0;j<=targetN;j++){ aBoard[j]=j; }
   */
  int tmp;
  int sizeE=size-1;
  /** 枝刈り */
  //if(row==size){
  if(row==sizeE){
    /** 枝刈り */
    if(right[row-aBoard[row]+sizeE]==1
        || left[row+aBoard[row]]==1){
      return;
    }
    int s=symmetryOps(size);  //対称解除法の導入
    if(s!=0){
      UNIQUE++;
      TOTAL+=s;
    }
  }else{
    /** 枝刈り */
    //for(int col=0;col<size;col++){
    int lim=(row!=0) ? size : (size+1)/2;
    // col のシフトがなくなりrowとなります
    for(int col=row;col<lim;col++){
      /** コメントアウト */
      //aBoard[row]=col;
      /** 交換 */
      tmp=aBoard[col];
      aBoard[col]=aBoard[row];
      aBoard[row]=tmp;
      /** 交換 */
      // col は aBoard[row]に置き換わります
      if(right[row-aBoard[row]+sizeE]==0
          && left[row+aBoard[row]]==0){
        right[row-aBoard[row]+sizeE]
          =left[row+aBoard[row]]=1;
        NQueenR(row+1,size);
        right[row-aBoard[row]+sizeE]
          =left[row+aBoard[row]]=0;
      }
    }
    /** 交換 */
    tmp=aBoard[row];
    for(int col=row;col<lim;col++){
      aBoard[col]=aBoard[col+1];
    }
    aBoard[size-1]=tmp;
    /** 交換 */
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
      printf("\n\n５．CPU 非再帰 バックトラック＋対称解除法＋枝刈りと最適化\n");
    }else if(cpur){
      printf("\n\n５．CPUR 再帰 バックトラック＋対称解除法＋枝刈りと最適化\n");
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
      if(cpu){
        //aBoardを-1で初期化
        for(int j=0;j<=targetN;j++){ aBoard[j]=-1; }
        NQueen(0,i);
      }
      if(cpur){
        /** 再帰は0で初期化 */
        for(int j=0;j<=targetN;j++){
          /** 【注意】初期化が前のステップと異なります */
          //aBoard[j]=0;
          aBoard[j]=j;
        }
        NQueenR(0,i);
      }
      TimeFormat(clock()-st,t);
      printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
    }
    return 0;
  }
