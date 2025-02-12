
//$ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC07NR.c && ./a.out [-c|-r]

/**
bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC07NR.c && ./a.out -r

７．CPUR 再帰 バックトラック＋ビットマップ＋対称解除法
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
13:        73712            9233            0.08
14:       365596           45752            0.48
15:      2279184          285053            3.20
16:     14772512         1846955           22.49
17:     95815104        11977939         2:41.93


bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC07NR.c && ./a.out -c
７．CPU 非再帰 バックトラック＋ビットマップ＋対称解除法
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
14:       365596           45752            0.49
15:      2279184          285053            3.25
16:     14772512         1846955           22.96
17:     95815104        11977939         2:43.94
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
//
#define MAX 27
//
int down[2*MAX-1]; 	//down:flagA 縦 配置フラグ
int right[2*MAX-1]; //right:flagB 斜め配置フラグ
int left[2*MAX-1]; 	//left:flagC 斜め配置フラグ
long TOTAL=0;
long UNIQUE=0;
int aBoard[MAX];
int aT[MAX];       	//aT:aTrial[]
int aS[MAX];       	//aS:aScrath[]
int COUNT2,COUNT4,COUNT8;
//関数宣言
void TimeFormat(clock_t utime,char *form);
int rh(int a,int sz);
void vMirror_bitmap(int chk[],int af[],int si);
void rotate_bitmap(int chk[],int af[],int si);
int intncmp(int lt[],int rt[],int n);
long getUnique();
long getTotal();
void symmetryOps_bitmap(int si);
void NQueen(int size,int mask,int row);
void NQueenR(int size,int mask,int row,int left,int down,int right);
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
//
int rh(int a,int sz){
  int tmp=0;
  for(int i=0;i<=sz;i++){
    if(a&(1<<i)){ return tmp|=(1<<(sz-i)); }
  }
  return tmp;
}
//
void vMirror_bitmap(int bf[],int af[],int si){
  int score ;
  for(int i=0;i<si;i++) {
    score=bf[i];
    af[i]=rh(score,si-1);
  }
}
//
void rotate_bitmap(int bf[],int af[],int si){
  for(int i=0;i<si;i++){
    int t=0;
    for(int j=0;j<si;j++){
      t|=((bf[j]>>i)&1)<<(si-j-1); // x[j] の i ビット目を
    }
    af[i]=t;                        // y[i] の j ビット目にする
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
//
long getUnique(){
  return COUNT2+COUNT4+COUNT8;
}
//
long getTotal(){
  return COUNT2*2+COUNT4*4+COUNT8*8;
}
//
void symmetryOps_bitmap(int si){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aT[i]=aBoard[i];}
  rotate_bitmap(aT,aS,si);    //時計回りに90度回転
  int k=intncmp(aBoard,aS,si);
  if(k>0)return;
  if(k==0){ nEquiv=2;}else{
    rotate_bitmap(aS,aT,si);  //時計回りに180度回転
    k=intncmp(aBoard,aT,si);
    if(k>0)return;
    if(k==0){ nEquiv=4;}else{
      rotate_bitmap(aT,aS,si);//時計回りに270度回転
      k=intncmp(aBoard,aS,si);
      if(k>0){ return;}
      nEquiv=8;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aS[i]=aBoard[i];}
  vMirror_bitmap(aS,aT,si);   //垂直反転
  k=intncmp(aBoard,aT,si);
  if(k>0){ return; }
  if(nEquiv>2){             //-90度回転 対角鏡と同等
    rotate_bitmap(aT,aS,si);
    k=intncmp(aBoard,aS,si);
    if(k>0){return;}
    if(nEquiv>4){           //-180度回転 水平鏡像と同等
      rotate_bitmap(aS,aT,si);
      k=intncmp(aBoard,aT,si);
      if(k>0){ return;}       //-270度回転 反対角鏡と同等
      rotate_bitmap(aT,aS,si);
      k=intncmp(aBoard,aS,si);
      if(k>0){ return;}
    }
  }
  if(nEquiv==2){COUNT2++;}
  if(nEquiv==4){COUNT4++;}
  if(nEquiv==8){COUNT8++;}
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
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap); 
      if(row==sizeE){
        /* 対称解除法の追加 */
        //TOTAL++;
        symmetryOps_bitmap(size); 
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
  int bit;
  int bitmap=mask&~(left|down|right);
  if(row==size){
    /* 対称解除法の追加 */
    //TOTAL++;
    symmetryOps_bitmap(size);
  }else{
    while(bitmap){
      //bitmap^=bit=(-bitmap&bitmap);
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
      NQueenR(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
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
    printf("\n\n７．CPU 非再帰 バックトラック＋ビットマップ＋対称解除法\n");
  }else if(cpur){
    printf("\n\n７．CPUR 再帰 バックトラック＋ビットマップ＋対称解除法\n");
  }
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  clock_t st;           //速度計測用
  char t[20];           //hh:mm:ss.msを格納
  int min=4;
  int targetN=17;
  int mask;
  for(int i=min;i<=targetN;i++){
    //TOTAL=0; UNIQUE=0;
    COUNT2=COUNT4=COUNT8=0;
    mask=(1<<i)-1;
    st=clock();
    if(cpu){
      //初期化は不要です
      // 再帰ではaBaordを使用しないので
      // 初期化の必要はありません
      /** 非再帰は-1で初期化 */
      // for(int j=0;j<=targetN;j++){
      // 	aBoard[j]=-1;
      // }
      NQueen(i,mask,0);
    }
    if(cpur){
      //初期化は不要です
      /** 再帰は0で初期化 */
      // for(int j=0;j<=targetN;j++){
      /** 【注意】初期化が前のステップと異なります */
      //aBoard[j]=0;
      //aBoard[j]=j;
      // }
      NQueenR(i,mask,0,0,0,0);
    }
    TimeFormat(clock()-st,t);
    printf("%2d:%13ld%16ld%s\n",i,getTotal(),getUnique(),t);
  }
  return 0;
}
