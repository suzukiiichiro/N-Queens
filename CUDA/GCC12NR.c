
// $ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC12NR.c && ./a.out [-c|-r]

/**
bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC12NR.c && ./a.out -r
１２．CPUR 再帰 対称解除法の最適化
１２．CPUR 再帰 対称解除法の最適化
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00 1 0 0
 5:           10               2            0.00 1 0 1
 6:            4               1            0.00 0 1 0
 7:           40               6            0.00 0 2 4
 8:           92              12            0.00 0 1 11
 9:          352              46            0.00 0 4 42
10:          724              92            0.00 0 3 89
11:         2680             341            0.00 0 12 329
12:        14200            1787            0.00 4 18 1765
13:        73712            9233            0.01 4 32 9197
14:       365596           45752            0.07 0 105 45647
15:      2279184          285053            0.40 0 310 284743
16:     14772512         1846955            2.61 32 734 1846189
17:     95815104        11977939           18.05 64 2006 11975869


bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC12NR.c && ./a.out -c
１２．CPU 非再帰 対称解除法の最適化
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00 1 0 0
 5:           10               2            0.00 1 0 1
 6:            4               1            0.00 0 1 0
 7:           40               6            0.00 0 2 4
 8:           92              12            0.00 0 1 11
 9:          352              46            0.00 0 4 42
10:          724              92            0.00 0 3 89
11:         2680             341            0.00 0 12 329
12:        14200            1787            0.00 4 18 1765
13:        73712            9233            0.01 4 32 9197
14:       365596           45752            0.06 0 105 45647
15:      2279184          285053            0.34 0 310 284743
16:     14772512         1846955            2.24 32 734 1846189
17:     95815104        11977939           15.72 64 2006 11975869
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <stdbool.h>
//
#define MAX 27
//変数宣言
long Total=0 ;      //合計解
long Unique=0;
int aBoard[MAX];
int COUNT2,COUNT4,COUNT8;
int BOUND1,BOUND2,TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
//関数宣言
void TimeFormat(clock_t utime,char *form);
long getUnique();
long getTotal();
void symmetryOps(int si);
void backTrack2_NR(int si,int mask,int y,int l,int d,int r);
void backTrack1_NR(int si,int mask,int y,int l,int d,int r);
void NQueen(int size,int mask);
// void backTrack2(int si,int mask,int y,int l,int d,int r);
void backTrack2(int size,int mask,int row,int left,int down,int right,int ex1,int ex2);
// void backTrack1(int si,int mask,int y,int l,int d,int r);
void backTrack1(int size,int mask,int row,int left,int down,int right,int ex1,int ex2);
// void NQueenR(int size,int mask);
void NQueenR(int size,int mask,int ex1,int ex2);
//
//hh:mm:ss.ms形式に処理時間を出力
void TimeFormat(clock_t utime,char *form){
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
long getUnique(){
  return COUNT2+COUNT4+COUNT8;
}
//
long getTotal(){
  return COUNT2*2+COUNT4*4+COUNT8*8;
}
//
void symmetryOps(int si){
  int own,ptn,you,bit;
  //90度回転
  if(aBoard[BOUND2]==1){ own=1; ptn=2;
    while(own<=si-1){ bit=1; you=si-1;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you--; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>si-1){ COUNT2++; return; }
  }
  //180度回転
  if(aBoard[si-1]==ENDBIT){ own=1; you=si-1-1;
    while(own<=si-1){ bit=1; ptn=TOPBIT;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>si-1){ COUNT4++; return; }
  }
  //270度回転
  if(aBoard[BOUND1]==TOPBIT){ own=1; ptn=TOPBIT>>1;
    while(own<=si-1){ bit=1; you=0;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you++; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  COUNT8++;
}
//CPU 非再帰版 backTrack2
void backTrack2_NR(int size,int mask,int row,int left,int down,int right){
	int bitmap,bit;
	int b[100], *p=b;
  int sizeE=size-1;
  int odd=size&1; //奇数:1 偶数:0
  for(int i=0;i<(1+odd);++i){
    bitmap=0;
    if(0==i){
      int half=size>>1; // size/2
      bitmap=(1<<half)-1;
    }else{
      bitmap=1<<(size>>1);
      // down[1]=bitmap;
      // right[1]=(bitmap>>1);
      // left[1]=(bitmap<<1);
      // pnStack=aStack+1;
      // *pnStack++=0;
    }
    mais1:bitmap=mask&~(left|down|right);
    // 【枝刈り】
    //if(row==size){
    if(row==sizeE){
      //if(!bitmap){
      if(bitmap){
        //【枝刈り】 最下段枝刈り
        if((bitmap&LASTMASK)==0){
          aBoard[row]=bitmap; //symmetryOpsの時は代入します。
          symmetryOps(size);
        }
      }
    }else{
      //【枝刈り】上部サイド枝刈り
      if(row<BOUND1){
        bitmap&=~SIDEMASK;
        //【枝刈り】下部サイド枝刈り
      }else if(row==BOUND2){
        if(!(down&SIDEMASK))
          goto volta;
        if((down&SIDEMASK)!=SIDEMASK)
          bitmap&=SIDEMASK;
      }
      if(bitmap){
  outro:bitmap^=aBoard[row]=bit=-bitmap&bitmap;
        if(bitmap){
          *p++=left;
          *p++=down;
          *p++=right;
        }
        *p++=bitmap;
        row++;
        left=(left|bit)<<1;
        down=down|bit;
        right=(right|bit)>>1;
        goto mais1;
        //Backtrack2(y+1, (left | bit)<<1, down | bit, (right | bit)>>1);
  volta:if(p<=b)
          return;
        row--;
        bitmap=*--p;
        if(bitmap){
          right=*--p;
          down=*--p;
          left=*--p;
          goto outro;
        }else{
          goto volta;
        }
      }
    }
    goto volta;
  }
}
//CPU 非再帰版 backTrack
void backTrack1_NR(int size,int mask,int row,int left,int down,int right){
	int bitmap,bit;
	int b[100], *p=b;
  int sizeE=size-1;
  int odd=size&1; //奇数:1 偶数:0
  for(int i=0;i<(1+odd);++i){
    bitmap=0;
    if(0==i){
      int half=size>>1; // size/2
      bitmap=(1<<half)-1;
    }else{
      bitmap=1<<(size>>1);
      // down[1]=bitmap;
      // right[1]=(bitmap>>1);
      // left[1]=(bitmap<<1);
      // pnStack=aStack+1;
      // *pnStack++=0;
    }
    b1mais1:bitmap=mask&~(left|down|right);
    //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
    //if(row==size){
    if(row==sizeE){
      //if(!bitmap){
      if(bitmap){
        // aBoard[row]=bitmap;
        //symmetryOps_bitmap(size);
        COUNT8++;
      }
    }else{
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      if(row<BOUND1) {
        bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
      }
      if(bitmap){
  b1outro:bitmap^=aBoard[row]=bit=-bitmap&bitmap;
        if(bitmap){
          *p++=left;
          *p++=down;
          *p++=right;
        }
        *p++=bitmap;
        row++;
        left=(left|bit)<<1;
        down=down|bit;
        right=(right|bit)>>1;
        goto b1mais1;
        //Backtrack1(y+1, (left | bit)<<1, down | bit, (right | bit)>>1);
  b1volta:if(p<=b)
          return;
        row--;
        bitmap=*--p;
        if(bitmap){
          right=*--p;
          down=*--p;
          left=*--p;
          goto b1outro;
        }else{
          goto b1volta;
        }
      }
    }
    goto b1volta;
  }
}
//CPU 非再帰版 ロジックメソッド
void NQueen(int size,int mask){
  int bit;
  TOPBIT=1<<(size-1);
  aBoard[0]=1;
  for(BOUND1=2;BOUND1<size-1;BOUND1++){
    aBoard[1]=bit=(1<<BOUND1);
    //backTrack1(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
    backTrack1_NR(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    //backTrack1(size,mask,1,bit<<1,bit,bit>>1);
    backTrack2_NR(size,mask,1,bit<<1,bit,bit>>1);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  }
}
//
// void backTrack2(int size,int mask,int row,int left,int down,int right){
void backTrack2(int size,int mask,int row,int left,int down,int right,int ex1,int ex2){
  int bit;
  int bitmap=mask&~(left|down|right);
  // 省略したソースは下を参考
  // if(size%2){ //奇数
  //   if(ex2!=0){ //１回目の再帰
  //     bitmap=mask&~(left|down|right);   //BOUNDで対応済み
  //   }else{      //２回目以降の再帰
  //     bitmap=mask&~(left|down|right);
  //   }
  // }else{  //偶数
  //   if(ex1!=0){ //１回目の再帰
  //     bitmap=mask&~(left|down|right);   //BOUNDで対応済み
  //   }else{      //２回目以降の再帰
  //     bitmap=mask&~(left|down|right|ex1);
  //   }
  // }
  // 上を省略すると以下の通りとなります。
  bitmap=mask&~(left|down|right);
  if(size%2==0 && ex1==0){ //偶数の２回目以降の再帰のみ対応
      bitmap=mask&~(left|down|right|ex1);
  }
  if(row==size-1){ 								// 【枝刈り】
    if(bitmap){
      if((bitmap&LASTMASK)==0){ 	//【枝刈り】 最下段枝刈り
        aBoard[row]=bitmap; //symmetryOpsの時は代入します。
        symmetryOps(size);
      }
    }
  }else{
    if(row<BOUND1){             	//【枝刈り】上部サイド枝刈り
      bitmap&=~SIDEMASK;
    }else if(row==BOUND2) {     	//【枝刈り】下部サイド枝刈り
      if((down&SIDEMASK)==0){ return; }
      if((down&SIDEMASK)!=SIDEMASK){ bitmap&=SIDEMASK; }
    }
    while(bitmap){
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
      // backTrack2(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
      backTrack2(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1,ex2,0);
      ex2=0;
    }
  }
}
//
// void backTrack1(int size,int mask,int row,int left,int down,int right){
void backTrack1(int size,int mask,int row,int left,int down,int right,int ex1,int ex2){
	int bit;
	int bitmap=mask&~(left|down|right);
  // 省略したソースは下を参考
  // NQueenR(i,mask,0,0,0,0,excl,i%2 ? excl : 0);
  // if(size%2){ //奇数
  //   if(ex2!=0){ //１回目の再帰
  //     bitmap=mask&~(left|down|right);   //BOUNDで対応済み
  //   }else{      //２回目以降の再帰
  //     bitmap=mask&~(left|down|right);
  //   }
  // }else{  //偶数
  //   if(ex1!=0){ //１回目の再帰
  //     bitmap=mask&~(left|down|right);   //BOUNDで対応済み
  //   }else{      //２回目以降の再帰
  //     bitmap=mask&~(left|down|right|ex1);
  //   }
  // }
  // 上を省略すると以下の通りとなります。
  bitmap=mask&~(left|down|right);
  if(size%2==0 && ex1==0){ //偶数の２回目以降の再帰のみ対応
      bitmap=mask&~(left|down|right|ex1);
  }
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
  if(row==size-1) {
    if(bitmap){
      // aBoard[row]=bitmap;
      COUNT8++;
    }
  }else{
    //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
    // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
    if(row<BOUND1) {
      bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
    while(bitmap){
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
      // backTrack1(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
      backTrack1(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1,ex2,0);
      ex2=0;
    }
  }
}
//CPUR 再帰版 ロジックメソッド
// void NQueenR(int size,int mask){
void NQueenR(int size,int mask,int ex1,int ex2){
  int bit;
  TOPBIT=1<<(size-1);
  aBoard[0]=1;
  for(BOUND1=2;BOUND1<size-1;BOUND1++){
    aBoard[1]=bit=(1<<BOUND1);
    // backTrack1(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
    backTrack1(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1),ex1,ex2);
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    // backTrack2(size,mask,1,bit<<1,bit,bit>>1);
    backTrack2(size,mask,1,bit<<1,bit,bit>>1,ex1,ex2);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  }
}
//メインメソッド
int main(int argc,char** argv) {
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
    printf("\n\n１２．CPU 非再帰 対称解除法の最適化\n");
  }else if(cpur){
    printf("\n\n１２．CPUR 再帰 対称解除法の最適化\n");
  }
  if(cpu||cpur){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    clock_t st;           //速度計測用
    char t[20];           //hh:mm:ss.msを格納
    int min=4; int targetN=17;
    int mask;
    int excl;
    for(int i=min;i<=targetN;i++){
      //TOTAL=0; UNIQUE=0;
      COUNT2=COUNT4=COUNT8=0;
      mask=(1<<i)-1;
      excl=(1<<((i/2)^0))-1;
      st=clock();
      if(cpu){
        //初期化は不要です
        //非再帰は-1で初期化
        // for(int j=0;j<=targetN;j++){ aBoard[j]=-1; }
        NQueen(i,mask);
      }
      if(cpur){
        //初期化は不要です
        //再帰は0で初期化
        //for(int j=0;j<=targetN;j++){ aBoard[j]=0; }
        // for(int j=0;j<=targetN;j++){ aBoard[j]=j; }
        // NQueenR(i,mask);
        NQueenR(i,mask,excl,i%2 ? excl : 0);
      }
      TimeFormat(clock()-st,t);
      // printf("%2d:%13ld%16ld%s\n",i,getTotal(),getUnique(),t);
      printf("%2d:%13ld%16ld%s %d %d %d\n",i,getTotal(),getUnique(),t,COUNT2,COUNT4,COUNT8);
    }
  }
  return 0;
}
