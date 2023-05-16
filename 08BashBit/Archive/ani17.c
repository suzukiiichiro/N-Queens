/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 実行
 $ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC06.c && ./a.out [-c|-r]


６．CPUR 再帰 バックトラック＋ビットマップ
 

bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC06.c && ./a.out -r
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


bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC06.c && ./a.out -c
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
#define THREAD_NUM		96
#define MAX 27
typedef unsigned long long uint64;
typedef struct{
  uint64 bv;
  uint64 down;
  uint64 left;
  uint64 right;
  int x[MAX];
}Board ;
//
Board B;
unsigned int COUNT8=2;
unsigned int COUNT4=1;
unsigned int COUNT2=0;
long cnt[3];
long pre[3];
//変数宣言
long TOTAL=0;         //CPU,CPUR
long UNIQUE=0;        //CPU,CPUR

//デバッグ用
long STEPCOUNT=0;
long KOHO2=0;
long KOHO4=0;
long KOHO8=0;


//関数宣言 CPU
void TimeFormat(clock_t utime,char *form);
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
//出力
void dec2bin(int size, int dec)
{
  int i, b[32];
  for (i = 0; i < size; i++)
  {
    b[i] = dec % 2;
    dec = dec / 2;
  }
  while (i > 0)
    printf("%1d", b[--i]);
}


/**
 * デバッグ用
 * クイーンの配置を表示する
 *
 * N9:316807425<--各行のクイーンの列番号
 * 000001000   <--クイーンの置かれている場所が1、それ以外は0
 * 000000010
 * 001000000
 * 100000000
 * 000000001
 * 010000000
 * 000010000
 * 000000100
 * 000100000
 */
void breakpoint(int size,char* string,int* board,int row,int bit)
{
  printf("%s\n",string);
  printf("<>N=%d STEP:",size);
  for(int i=0;i<size;i++){
    if(board[i]==1){ printf("0"); }
    else if(board[i]==2){ printf("1"); }
    else if(board[i]==4){ printf("2"); }
    else if(board[i]==8){ printf("3"); }
    else if(board[i]==16){ printf("4"); }
    else if(board[i]==32){ printf("5"); }
    else if(board[i]==64){ printf("6"); }
    else if(board[i]==128){ printf("7"); }
    else if(board[i]==256){ printf("8"); }
    else if(board[i]==512){ printf("9"); }
    else if(board[i]==1024){ printf("10"); }
    else if(board[i]==2048){ printf("11"); }
    else if(board[i]==4096){ printf("12"); }
    else if(board[i]==8192){ printf("13"); }
    else if(board[i]==-1){ printf("-"); }
  }
  printf("  ");
  printf("row:%d  bit:%d\n",row,bit);
  printf("\n");
  //colの座標表示
  printf("   ");
  for (int j=size-1;j>=0;j--){
    printf(" %2d",j);
  }
  printf("\n");
  printf(" =============");
  for (int j=0;j<size;j++){
    printf("==");
  }
  printf("\n");
  //row
  for (int j=0;j<size;j++){
    printf("%2d| ",j);
    if(board[j]==-1){ dec2bin(size,0); }
    else{ 
      dec2bin(size,board[j]); 
    }
    printf("\n");
  }
  printf("\n");
  /**
   *
   *
   */
  //int moji;
	//while ((moji = getchar()) != EOF){
//		switch (moji){
//		case '\n':
//		  return;
//		default:
//			break;
//		}
//	}
}
//
void breakpoint_nq27(int size,char* string,int* x,int row)
{
  printf("%s\n",string);
  printf("<>N=%d STEP:",size);
  for(int i=0;i<size;i++){
    if(x[i]==-1){ 
      printf("-"); 
    }else{
      printf("%d",x[i]);
    }
  }
  printf("  ");
  printf("row:%d\n",row);
  printf("\n");
  //colの座標表示
  printf("   ");
  for (int j=size-1;j>=0;j--){
    printf(" %2d",j);
  }
  printf("\n");
  printf(" =============");
  for (int j=0;j<size;j++){
    printf("==");
  }
  printf("\n");
  //row
  for (int j=0;j<size;j++){
    printf("%2d| ",j);
    if(x[j]==-1){ dec2bin(size,0); }
    else{ 
      dec2bin(size,1<<x[j]); 
    }
    printf("\n");
  }
  printf("\n");
  /**
   *
   *
   */
  //int moji;
	//while ((moji = getchar()) != EOF){
//		switch (moji){
//		case '\n':
//		  return;
//		default:
//			break;
//		}
//	}
}
//countCompletionsは変更していない
long bit93_countCompletions(int size,int row,int aBoard[],long left,long down,long right,int sym,int bBoard[],long bmask)
{
  //printf("0:countCompletions start\n");
  long bitmap=0;
  long bit=0;
  long cnt = 0;
  //既にクイーンを置いている行はスキップする
  //while((bv&1)!=0) {
  while(aBoard[row]!=-1&&row<size) {
    //printf("1:already placed\n");
    //bv>>=1;   //右に１ビットシフト
    left<<=1; //left 左に１ビットシフト
    right>>=1;//right 右に１ビットシフト  
    row++; 
  }
  //bv>>=1;
  if(row==size){ 
    //printf("2:all placed\n");
    //breakpoint(size,"F",bBoard,row,bit);
    return 1; 
  }
  else{
    //printf("3:else\n");
    long amask=0;
    if(row>1&&row<size-2){
     amask=bmask;
    }
    //printf("row:%d:down:%d:left:%d:right:%d\n",row,down,left,right);
    bitmap=~(amask|left|down|right);   
    while(bitmap>0){
      //printf("4:place:\n");
      //aBoard[row]=bit=(-bitmap&bitmap);
      bit=(-bitmap&bitmap);
      //int abit;
      //if(size==5){
      //  bBoard[row]=(bit<<2);
      //}else if(size==6){
      //  bBoard[row]=(bit<<1);
      //}else{
      //  bBoard[row]=(bit>>(size-7));
      //}
      //print(size,"クイーンを配置",bBoard,"N");
      bitmap=(bitmap^bit);
      //printf("5:start recursion\n");
      cnt+=bit93_countCompletions(size,row+1,aBoard,(left|bit)<<1,down|bit,(right|bit)>>1,sym,bBoard,bmask);
      //printf("6:finish recursion\n");
    }
    //printf("7:finish while");
  }
  return cnt;
}
//通常版 CPUR 再帰版　ロジックメソッド
//まず、backtrackを使って上下左右２行２列にクイーンを配置する
//上下２行以外は両端２列にだけクイーンを置くようにしながら最後の行までbacktrackする
//最後の行までbacktrackしたらcountCompletionsを呼びだす。
//lleft,ldown,lrightはboardplacementで設定していたのと同じ値（全行共通のleft,down,right)
int symmetryOps(int size,int aBoard[],int bBoard[],int BOUND1,int BOUND2,long TOPBIT,long ENDBIT){
     //aBoardからtop,left,bottom,rightを割り当てる
     int topSide_0=-1;
     int topSide_1=-1;
     int leftSide_0=-1;
     int leftSide_1=-1;
     int bottomSide_0=-1;
     int bottomSide_1=-1;
     int rightSide_0=-1;
     int rightSide_1=-1;
     for(int row=0;row<size;row++){
       int x=aBoard[row]; 
       //symmetry
       if (row==0){
         topSide_0=x;
       }
       if (row==1){ 
         topSide_1=x;
       }
       if (x==size-1){
         leftSide_0=row;
       }
       if (x==size-2){
         leftSide_1=row;
       }
       if (row==size-1){
         bottomSide_0=size-1-x;
       }
       if (row==size-2){
         bottomSide_1=size-1-x;
       }
       if (x==0){
         rightSide_0=size-1-row;  
       }
       if (x==1){
         rightSide_1=size-1-row;
       }
     }
    //割り当てられないものは枝刈り
    if((topSide_0==-1)||(topSide_1==-1)||(leftSide_0==-1)||(leftSide_1==-1)||(bottomSide_0==-1)||(bottomSide_1==-1)||(rightSide_0==-1)||(rightSide_1==-1)){
     return 3;
    }
    //printf("start:t0:%d,t1:%d,l0:%d,l1:%d,b0:%d,b1:%d,r0:%d,r1:%d\n",topSide_0,topSide_1,leftSide_0,leftSide_1,bottomSide_0,bottomSide_1,rightSide_0,rightSide_1);
    //右90度回転して同じだったら大小比較。全部同じならCOUNT2
    //
   if(bBoard[BOUND2]==1){
     if((topSide_0 > rightSide_0)||((topSide_0==rightSide_0)&&(topSide_1 > rightSide_1))){
       //printf("1\n");
       return 3;
     }
     if((topSide_0==rightSide_0&&topSide_1==rightSide_1)&&((leftSide_0>topSide_0)||((leftSide_0==topSide_0)&&(leftSide_1>topSide_1)))){ 
       //printf("2\n");
       return 3;
     }
     if((topSide_0==rightSide_0&&topSide_1==rightSide_1&&leftSide_0==topSide_0&&leftSide_1==topSide_1)&&((bottomSide_0>leftSide_0)||((bottomSide_0==leftSide_0)&&(bottomSide_1>leftSide_1)))){ 
       //printf("3\n");
       return 3;
     }
     if((topSide_0==rightSide_0&&topSide_1==rightSide_1&&leftSide_0==topSide_0&&leftSide_1==topSide_1&&bottomSide_0==leftSide_0&&bottomSide_1==leftSide_1)&&((rightSide_0>bottomSide_0)||((rightSide_0==bottomSide_0)&&(rightSide_1>bottomSide_1)))){ 
       //printf("4\n");
       return 3;
     }
     if((topSide_0==rightSide_0)&&(topSide_1==rightSide_1)&&(leftSide_0==topSide_0)&&(leftSide_1==topSide_1)&&(bottomSide_0==leftSide_0)&&(bottomSide_1==leftSide_1)&&(rightSide_0==bottomSide_0)&&(rightSide_1==bottomSide_1)){
      //printf("sym:0:t0:%d,t1:%d,l0:%d,l1:%d,b0:%d,b1:%d,r0:%d,r1:%d\n",topSide_0,topSide_1,leftSide_0,leftSide_1,bottomSide_0,bottomSide_1,rightSide_0,rightSide_1);
       //printf("5\n");
       KOHO2++;
       return 0;
     }
   }
   //右180度回転して同じだったら大小比較。全部同じならCOUNT4
   //
   if(bBoard[size-1]==ENDBIT){
     if((topSide_0 > bottomSide_0)||((topSide_0==bottomSide_0)&&(topSide_1 > bottomSide_1))){
       //printf("6\n");
       return 3;
     }
     if((topSide_0==bottomSide_0&&topSide_1==bottomSide_1)&&((leftSide_0>rightSide_0)||((leftSide_0==rightSide_0)&&(leftSide_1>rightSide_1)))){ 
       //printf("7\n");
       return 3;
     }
     if((topSide_0==bottomSide_0&&topSide_1==bottomSide_1&&leftSide_0==rightSide_0&&leftSide_1==rightSide_1)&&((bottomSide_0>topSide_0)||((bottomSide_0==topSide_0)&&(bottomSide_1>topSide_1)))){ 
       //printf("8\n");
       return 3;
     }
     if((topSide_0==bottomSide_0&&topSide_1==bottomSide_1&&leftSide_0==rightSide_0&&leftSide_1==rightSide_1&&bottomSide_0==topSide_0&&bottomSide_1==topSide_1)&&((rightSide_0>leftSide_0)||((rightSide_0==leftSide_0)&&(rightSide_1>leftSide_1)))){ 
       //printf("9\n");
       return 3;
     }
     if((topSide_0==bottomSide_0)&&(topSide_1==bottomSide_1)&&(leftSide_0==rightSide_0)&&(leftSide_1==rightSide_1)&&(bottomSide_0==topSide_0)&&(bottomSide_1==topSide_1)&&(rightSide_0==leftSide_0)&&(rightSide_1==leftSide_1)){
       //printf("sym:1:t0:%d,t1:%d,l0:%d,l1:%d,b0:%d,b1:%d,r0:%d,r1:%d\n",topSide_0,topSide_1,leftSide_0,leftSide_1,bottomSide_0,bottomSide_1,rightSide_0,rightSide_1);
       //printf("10\n");
       KOHO4++;
       return 1;
     }
   }
   //右270度回転して同じだったら大小比較。
   //
   if(bBoard[BOUND1]==TOPBIT){
     if((topSide_0 > leftSide_0)||((topSide_0==leftSide_0)&&(topSide_1 > leftSide_1))){
       //printf("11\n");
       return 3;
     }
     if((topSide_0==leftSide_0&&topSide_1==leftSide_1)&&((leftSide_0>bottomSide_0)||((leftSide_0==bottomSide_0)&&(leftSide_1>bottomSide_1)))){ 
       //printf("12\n");
       return 3;
     }
     if((topSide_0==leftSide_0&&topSide_1==leftSide_1&&leftSide_0==bottomSide_0&&leftSide_1==bottomSide_1)&&((bottomSide_0>rightSide_0)||((bottomSide_0==rightSide_0)&&(bottomSide_1>rightSide_1)))){ 
       //printf("13\n");
       return 3;
     }
     if((topSide_0==leftSide_0&&topSide_1==leftSide_1&&leftSide_0==bottomSide_0&&leftSide_1==bottomSide_1&&bottomSide_0==rightSide_0&&bottomSide_1==rightSide_1)&&((rightSide_0>topSide_0)||((rightSide_0==topSide_0)&&(rightSide_1>topSide_1)))){ 
       //printf("14\n");
       return 3;
     }

   }
   //それ以外はCOUNT8
   //printf("sym:2:t0:%d,t1:%d,l0:%d,l1:%d,b0:%d,b1:%d,r0:%d,r1:%d\n",topSide_0,topSide_1,leftSide_0,leftSide_1,bottomSide_0,bottomSide_1,rightSide_0,rightSide_1);
   //printf("15\n");
   KOHO8++;
   return 2;
}
void backTrackR1(int size,long mask,int row,long left,long down,long right,int aBoard[],long lleft,long ldown,long lright,int bBoard[],long bmask,int BOUND1){
  //printf("a:nqueen start\n");
  int bitmap=0;
  int bit=0;
  if(row==size){
     int topSide_0=-1;
     int topSide_1=-1;
     int leftSide_0=-1;
     int leftSide_1=-1;
     int bottomSide_0=-1;
     int bottomSide_1=-1;
     int rightSide_0=-1;
     int rightSide_1=-1;
     for(int row=0;row<size;row++){
       int x=aBoard[row]; 
       //symmetry
       if (row==0){
         topSide_0=x;
       }
       if (row==1){ 
         topSide_1=x;
       }
       if (x==size-1){
         leftSide_0=row;
       }
       if (x==size-2){
         leftSide_1=row;
       }
       if (row==size-1){
         bottomSide_0=size-1-x;
       }
       if (row==size-2){
         bottomSide_1=size-1-x;
       }
       if (x==0){
         rightSide_0=size-1-row;  
       }
       if (x==1){
         rightSide_1=size-1-row;
       }
     }
    //割り当てられないものは枝刈り
    if((topSide_0==-1)||(topSide_1==-1)||(leftSide_0==-1)||(leftSide_1==-1)||(bottomSide_0==-1)||(bottomSide_1==-1)||(rightSide_0==-1)||(rightSide_1==-1)){
     return;
    }
    //printf("sym:2:t0:%d,t1:%d,l0:%d,l1:%d,b0:%d,b1:%d,r0:%d,r1:%d\n",topSide_0,topSide_1,leftSide_0,leftSide_1,bottomSide_0,bottomSide_1,rightSide_0,rightSide_1);
    //printf("b:all placed\n");
    //breakpoint(size,"上下左右２行２列配置完了",bBoard,row,bit);
    //int sym=symmetryOps(size,aBoard);
    int sym=2;
    KOHO8++;
    int q=bit93_countCompletions(size,2,aBoard,lleft>>4,((((ldown>>2)|(~0<<(size-4)))+1)<<(size-5))-1,(lright>>4)<<(size-5),sym,bBoard,bmask);
    UNIQUE+=q; 
    TOTAL+=q*8;
  }else{
    long amask=mask;
    //上２行、下２行以外は左端２列、右端２列にだけクイーンを置くようにマスクを設置する
    if(row>1&&row<size-2){
      amask=(((1<<(size-4))-1)<<2);
      //printf("row:%d amask:%d,lll:%d,dddd:%d,rrr:%d\n",row,amask,left,down,right);
      if(amask&~(left|down|right)){
        //printf("f:amask\n");
        aBoard[row]=-1;
        bBoard[row]=-1;
        STEPCOUNT++;
        //上２行、下２行以外で左端２列、右端２列以外にクイーンを置ける余地がある場合はその行を空白にして次の行へ進むルートも作る
        backTrackR1(size,mask,row+1,(left)<<1, down,(right)>>1,aBoard,lleft,ldown,lright,bBoard,bmask,BOUND1);
      }
      //左端２列、右端２列にだけクイーンを置くようにマスクを設定する
      amask=amask^mask;
    }
    //printf("row:%d bbbbamask:%d left:%d down:%d right:%d\n",row,amask,left,down,right);
    bitmap=(amask&~(left|down|right));
    if(row<BOUND1) {
      bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
    //printf("bitmap:%d\n",bitmap);
    while(bitmap){
      //printf("c:place\n");
      //クイーンを置く
      bitmap^=bBoard[row]=bit=(-bitmap&bitmap);
      STEPCOUNT++;
      //printf("row:%d bit:%d\n",row,bit);
      //board_placementに合わせた値を設定するためにbitだけでなく何列目にクイーンを置いたかを算出
      int x=0;
      if(bit==1){
        x=0;
      }else if(bit==2){
        x=1;
      }else if(bit==4){
        x=2;
      }else if(bit==8){
        x=3;
      }else if(bit==16){
        x=4;
      }else if(bit==32){
        x=5;
      }else if(bit==64){
        x=6;
      }else if(bit==128){
        x=7;
      }else if(bit==256){
        x=8;
      }else if(bit==512){
        x=9;
      }else if(bit==1024){
        x=10;
      }else if(bit==2048){
        x=11;
      }else if(bit==4096){
        x=12;
      }else if(bit==8192){
        x=13;
      }else if(bit==16384){
        x=14;
      }else if(bit==32768){
        x=15;
      }else if(bit==65536){
        x=16;
      }else if(bit==131072){
        x=17;
      }
     aBoard[row]=x;
     //left,down,rightだけでなく、全行でleft,down,rightを表現できる値も設定している(board_placementで設定していた値)
     //printf("row:%d,lleft:%d,ldown:%d,lright:%d\n",row,lleft,ldown,lright);
      //printf("d:recursivele start\n");
      backTrackR1(size,mask,row+1,(left|bit)<<1,(down|bit),(right|bit)>>1,aBoard,lleft|1<<(size-1-row+x),ldown|bit,lright|1<<(row+x),bBoard,bmask,BOUND1);
      //printf("e:recursivele start\n");
    }
  }
}
void backTrackR2(int size,long mask,int row,long left,long down,long right,int aBoard[],long lleft,long ldown,long lright,int bBoard[],long bmask,int BOUND1,int BOUND2,long TOPBIT,long ENDBIT,long LASTMASK,long SIDEMASK){
  //printf("a:nqueen start\n");
  int bitmap=0;
  int bit=0;
  if(row==size){
      int sym=symmetryOps(size,aBoard,bBoard,BOUND1,BOUND2,TOPBIT,ENDBIT);
      if(sym!=3){
         //breakpoint(size,"上下左右２行２列配置完了",bBoard,row,bit);
         int q=bit93_countCompletions(size,2,aBoard,lleft>>4,((((ldown>>2)|(~0<<(size-4)))+1)<<(size-5))-1,lright=(lright>>4)<<(size-5),sym,bBoard,bmask);
         UNIQUE+=q; 
         if(sym==0){
           TOTAL+=q*2;
         }else if(sym==1){
           TOTAL+=q*4;
         }else if(sym==2){
           TOTAL+=q*8;
         }
      }
  }else{
    long amask=mask;
    //上２行、下２行以外は左端２列、右端２列にだけクイーンを置くようにマスクを設置する
    if(row>1&&row<size-2){
      amask=(((1<<(size-4))-1)<<2);
      //printf("row:%d amask:%d,lll:%d,dddd:%d,rrr:%d\n",row,amask,left,down,right);
      if(amask&~(left|down|right)){
        //printf("f:amask\n");
        aBoard[row]=-1;
        bBoard[row]=-1;
        STEPCOUNT++;
        //上２行、下２行以外で左端２列、右端２列以外にクイーンを置ける余地がある場合はその行を空白にして次の行へ進むルートも作る
        backTrackR2(size,mask,row+1,(left)<<1, down,(right)>>1,aBoard,lleft,ldown,lright,bBoard,bmask,BOUND1,BOUND2,TOPBIT,ENDBIT,LASTMASK,SIDEMASK);
      }
      //左端２列、右端２列にだけクイーンを置くようにマスクを設定する
      amask=amask^mask;
    }
    //printf("row:%d bbbbamask:%d left:%d down:%d right:%d\n",row,amask,left,down,right);
    bitmap=(amask&~(left|down|right));
    //【枝刈り】上部サイド枝刈り
    if(row<BOUND1){    
      //printf("eda_0\n");         	
      bitmap&=~SIDEMASK;
    //【枝刈り】下部サイド枝刈り
    }else if(row==BOUND2) {     	
      //printf("sidemask:row:%d BOUND2:%d SIDEMASK:%ld down:%ld\n",row,BOUND2,SIDEMASK,down);         	
      if((down&SIDEMASK)==0){ 
        //printf("eda_1\n");         	
        return; 
      }
      if((down&SIDEMASK)!=SIDEMASK){ 
        //printf("eda_2\n");         	
        //printf("before:%ld",bitmap);
        bitmap&=SIDEMASK; 
        //printf("after:%ld",bitmap);
      }
    }else if(row == size-1){
      //printf("eda_3\n");         	
      bitmap&=~LASTMASK;
    }else if(row > BOUND2){
      bitmap&=~SIDEMASK;
    }
   
    //printf("bitmap:%d\n",bitmap);
    while(bitmap){
      //printf("c:place\n");
      //クイーンを置く
      bitmap^=bBoard[row]=bit=(-bitmap&bitmap);
      STEPCOUNT++;
      //printf("row:%d bit:%d\n",row,bit);
      //board_placementに合わせた値を設定するためにbitだけでなく何列目にクイーンを置いたかを算出
      int x=0;
      if(bit==1){
        x=0;
      }else if(bit==2){
        x=1;
      }else if(bit==4){
        x=2;
      }else if(bit==8){
        x=3;
      }else if(bit==16){
        x=4;
      }else if(bit==32){
        x=5;
      }else if(bit==64){
        x=6;
      }else if(bit==128){
        x=7;
      }else if(bit==256){
        x=8;
      }else if(bit==512){
        x=9;
      }else if(bit==1024){
        x=10;
      }else if(bit==2048){
        x=11;
      }else if(bit==4096){
        x=12;
      }else if(bit==8192){
        x=13;
      }else if(bit==16384){
        x=14;
      }else if(bit==32768){
        x=15;
      }else if(bit==65536){
        x=16;
      }else if(bit==131072){
        x=17;
      }
     aBoard[row]=x;
     //left,down,rightだけでなく、全行でleft,down,rightを表現できる値も設定している(board_placementで設定していた値)
     //printf("row:%d,lleft:%d,ldown:%d,lright:%d\n",row,lleft,ldown,lright);
      //printf("d:recursivele start\n");
      backTrackR2(size,mask,row+1,(left|bit)<<1, (down|bit),(right|bit)>>1,aBoard,lleft|1<<(size-1-row+x),ldown|bit,lright|1<<(row+x),bBoard,bmask,BOUND1,BOUND2,TOPBIT,ENDBIT,LASTMASK,SIDEMASK);
      //printf("e:recursivele start\n");
    }
  }
}
//
void NQueenR(int size,long mask,int aBoard[],int bBoard[],long bmask){
  int bit;
  //枝刈りはまだしないのでTOPBIT,SIDEMASK,LASTMASK,ENDBITは使用しない
  //backtrack1
  //1行め右端 0
  int BOUND1;
  int BOUND2; 
  long TOPBIT=1<<(size-1);
  long SIDEMASK;
  long LASTMASK;
  long ENDBIT;
  int col=0;
  bBoard[0]=bit=(1<<col);
  aBoard[0]=0;
  STEPCOUNT++;
  long left=bit<<1;
  long down=bit;
  long right=bit>>1;
  long lleft=1<<(size-1);
  long ldown=bit;
  long lright=1<<(0);
  //2行目は右から3列目から左端から2列目まで
  for(int col_j=2;col_j<size-1;col_j++){
      bBoard[1]=bit=(1<<col_j);
      STEPCOUNT++;
      BOUND1=col_j;
      int x=0;
      if(bit==1){
        x=0;
      }else if(bit==2){
        x=1;
      }else if(bit==4){
        x=2;
      }else if(bit==8){
        x=3;
      }else if(bit==16){
        x=4;
      }else if(bit==32){
        x=5;
      }else if(bit==64){
        x=6;
      }else if(bit==128){
        x=7;
      }else if(bit==256){
        x=8;
      }else if(bit==512){
        x=9;
      }else if(bit==1024){
        x=10;
      }else if(bit==2048){
        x=11;
      }else if(bit==4096){
        x=12;
      }else if(bit==8192){
        x=13;
      }else if(bit==16384){
        x=14;
      }else if(bit==32768){
        x=15;
      }else if(bit==65536){
        x=16;
      }else if(bit==131072){
        x=17;
      }
      aBoard[1]=x;
      backTrackR1(size,mask,2,(left|bit)<<1,(down|bit),(right|bit)>>1,aBoard,lleft|1<<(size-1-1+x),ldown|bit,lright|1<<(1+x),bBoard,bmask,BOUND1);
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  //backtrack2
  //1行目右から2列目から
  //偶数個は1/2 n=8 なら 1,2,3 奇数個は1/2+1 n=9 なら 1,2,3,4
  for(int col=1,col2=size-2;col<col2;col++,col2--){
      bBoard[0]=bit=(1<<col);
      STEPCOUNT++;
      BOUND1=col;
      BOUND2=col2;
      int x=0;
      if(bit==1){
        x=0;
      }else if(bit==2){
        x=1;
      }else if(bit==4){
        x=2;
      }else if(bit==8){
        x=3;
      }else if(bit==16){
        x=4;
      }else if(bit==32){
        x=5;
      }else if(bit==64){
        x=6;
      }else if(bit==128){
        x=7;
      }else if(bit==256){
        x=8;
      }else if(bit==512){
        x=9;
      }else if(bit==1024){
        x=10;
      }else if(bit==2048){
        x=11;
      }else if(bit==4096){
        x=12;
      }else if(bit==8192){
        x=13;
      }else if(bit==16384){
        x=14;
      }else if(bit==32768){
        x=15;
      }else if(bit==65536){
        x=16;
      }else if(bit==131072){
        x=17;
      }
      aBoard[0]=x;
      backTrackR2(size,mask,1,bit<<1,bit,bit>>1,aBoard,1<<(size-1+x),bit,1<<(x),bBoard,bmask,BOUND1,BOUND2,TOPBIT,ENDBIT,LASTMASK,SIDEMASK);
      LASTMASK|=LASTMASK>>1|LASTMASK<<1;
      ENDBIT>>=1;
  }

}
//nq27
long solve_nqueenr(uint64 bv,uint64 left,uint64 down,uint64 right)
{
  // Placement Complete?
  //printf("countCompletions_start\n");
  //printf("bv:%d\n",bv);
  //printf("bh:%d\n",bh);
  //printf("bu:%d\n",bu);
  //printf("bd:%d\n",bd);
  //bh=-1 1111111111 すべての列にクイーンを置けると-1になる
  if(down+1==0){
    //printf("return_bh+1==0:%d\n",bh);  
    return  1;
  }
  // -> at least one more queen to place
  while((bv&1)!=0) { // Column is covered by pre-placement
    //bv 右端にクイーンがすでに置かれていたら。クイーンを置かずに１行下に移動する
    //bvを右端から１ビットずつ削っていく。ここではbvはすでにクイーンが置かれているかどうかだけで使う
    bv>>=1;//右に１ビットシフト
    left<<=1;//left 左に１ビットシフト
    right>>=1;//right 右に１ビットシフト
    //printf("while:bv:%d\n",bv);
    //printf("while:bu:%d\n",bu);
    //printf("while:bd:%d\n",bd);
    //printf("while:bv&1:%d\n",bv&1);
  }
  //１行下に移動する
  bv>>=1;
  //printf("onemore_bv:%d\n",bv);
  //printf("onemore_bh:%d\n",bh);
  //printf("onemore_bu:%d\n",bu);
  //printf("onemore_bd:%d\n",bd);
  //
  // Column needs to be placed
  long  s=0;
  uint64 bit;
  //bh:down bu:left bd:right
  //クイーンを置いていく
  //slotsはクイーンの置ける場所
  for(uint64 bitmap=~(left|down|right);bitmap!=0;bitmap^=bit){
    //printf("colunm needs to be placed\n");
    //printf("slots:%d\n",slots);
    bit=bitmap&-bitmap;
    //printf("slot:%d\n",slot);
    //printf("bv:%d:bh|slot:%d:(bu|slot)<<1:%d:(bd|slot)>>1:%d\n",bv,bh|slot,(bu|slot)<<1,(bd|slot)>>1);
    s+=solve_nqueenr(bv,(left|bit)<<1,down|bit,(right|bit)>>1);
    //slots^=slot;
    //printf("slots:%d\n",slots);
  }
  //途中でクイーンを置くところがなくなるとここに来る
  //printf("return_cnt:%d\n",cnt);
  return s;
} // countCompletions()
//
void process(int si,Board B,int sym)
{
  //printf("process\n");
  pre[sym]++;
  //printf("N:%d\n",si);
  //BVは行 x 
  //printf("getBV:%d\n",B.bv);
  //BHはdown y
  //printf("getBH:%d\n",B.bh);
  //BU left N-1-x+y 右上から左下
  //printf("getBU:%d\n",B.bu);
  //BD right x+y 左上から右下
  //printf("getBD:%d\n",B.bd);
  //printf("before_cnt_sym:%d\n",cnt[sym]);
  cnt[sym] += solve_nqueenr(B.bv >> 2,
      B.left>>4,
      ((((B.down>>2)|(~0<<(si-4)))+1)<<(si-5))-1,
      (B.right>>4)<<(si-5));

  //行 brd.getBV()>>2 右2ビット削除 すでに上２行はクイーンを置いているので進める BVは右端を１ビットずつ削っていく
  //列 down ((((brd.getBH()>>2)|(~0<<(N-4)))+1)<<(brd.N-5))-1 8だと左に1シフト 9:2 10:3 
  //brd.getBU()>>4 left  右４ビット削除
  //(brd.getBD()>>4)<<(N-5)) right 右４ビット削除後N-5個分左にシフト
  //printf("cnt_sym:%d\n",cnt[sym]);
}
bool board_placement(int si,int x,int y)
{
  STEPCOUNT++;
  //同じ場所に置くかチェック
  //printf("i:%d:x:%d:y:%d\n",i,B.x[i],B.y[i]);
  if(B.x[x]==y){
    //printf("Duplicate x:%d:y:%d\n",x,y);
    ////同じ場所に置くのはOK
    return true;  
  }
  B.x[x]=y;
  //xは行 yは列 p.N-1-x+yは右上から左下 x+yは左上から右下
  uint64 bv=1<<x;
  uint64 down=1<<y;
  uint64 left=1<<(si-1-x+y);
  uint64 right=1<<(x+y);
  //printf("check valid x:%d:y:%d:p.N-1-x+y:%d;x+y:%d\n",x,y,si-1-x+y,x+y);
  //printf("check valid pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
  //printf("bvcheck:%d:bhcheck:%d:bucheck:%d:bdcheck:%d\n",B.bv&bv,B.bh&bh,B.bu&bu,B.bd&bd);
  if((B.bv&bv)||(B.down&down)||(B.left&left)||(B.right&right)){
    //printf("valid_false\n");
    return false;
  }     
  //printf("before pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
  B.bv |=bv;
  B.down |=down;
  B.left |=left;
  B.right |=right;
  //printf("after pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
  //printf("valid_true\n");
  return true;
}
//CPUR 再帰版 ロジックメソッド
void NQueen_nq27(int size)
{
  int pres_a[930];
  int pres_b[930];
  int idx=0;
  for(int a=0;a<size;a++){
    for(int b=0;b<size;b++){
      if((a>=b&&(a-b)<=1)||(b>a&&(b-a)<=1)){
        continue;
      }     
      pres_a[idx]=a;
      pres_b[idx]=b;
      idx++;
    }
  }
  //プログレス
  //printf("\t\t  First side bound: (%d,%d)/(%d,%d)",(unsigned)pres_a[(size/2)*(size-3)  ],(unsigned)pres_b[(size/2)*(size-3)  ],(unsigned)pres_a[(size/2)*(size-3)+1],(unsigned)pres_b[(size/2)*(size-3)+1]);

  Board wB=B;
  for(int w=0;w<=(size/2)*(size-3);w++){
  //上２行にクイーンを置く
  //上１行は２分の１だけ実行
  //q=7なら (7/2)*(7-4)=12
  //1行目は0,1,2で,2行目0,1,2,3,4,5,6 で利き筋を置かないと13パターンになる
    B=wB;
    //
    // B.bv=0;
    // B.bh=0;
    // B.bu=0;
    // B.bd=0;
    B.bv=B.down=B.left=B.right=0;
    //
    for(int i=0;i<size;i++){
      B.x[i]=-1;
    }
    // 不要
    // int wa=pres_a[w];
    // int wb=pres_b[w];
    //
    //プログレス
    //printf("\r(%d/%d)",w,((size/2)*(size-3)));// << std::flush;
    //printf("\r");
    //fflush(stdout);
  
    //上２行　0行目,1行目にクイーンを置く
    //
    // 謎１
    // 
    // 置き換え
    // board_placement(size,0,wa);
    //0行目にクイーンを置く
    board_placement(size,0,pres_a[w]);
    //printf("placement_pwa:x:0:y:%d\n",pres_a[w]);
    //
    //
    // 謎２
    // 
    //置き換え
    //board_placement(size,1,wb);
    //1行目にクイーンを置く
    board_placement(size,1,pres_b[w]);
    //printf("placement_pwb:x:1:y:%d\n",pres_b[w]);

    Board nB=B;
    //追加
    int lsize=(size-2)*(size-1)-w;
    //for(int n=w;n<(size-2)*(size-1)-w;n++){
    for(int n=w;n<lsize;n++){
    //左２列にクイーンを置く
      B=nB;
      //printf("nloop:n:%d\n",n);
      //
      // 不要
      // int na=pres_a[n];
      // int nb=pres_b[n];   
      //
      //置き換え
      //bool pna=board_placement(size,na,size-1);
      //bool pna=board_placement(size,pres_a[n],size-1);
      //インライン
      //if(pna==false){
      if(board_placement(size,pres_a[n],size-1)==false){
        //printf("pnaskip:na:%d:N-1:%d\n",na,size-1);
        continue;
      }
      //printf("placement_pna:x:%d:yk(N-1):%d\n",pres_a[n],size-1);
      //置き換え
      //bool pnb=board_placement(size,nb,size-2);
      //bool pnb=board_placement(size,pres_b[n],size-2);
      //インライン
      //if(pnb==false){
      if(board_placement(size,pres_b[n],size-2)==false){
        //printf("pnbskip:nb:%d:N-2:%d\n",nb,size-2);
        continue;
      }
      //printf("placement_pnb:x:%d:yk(N-2):%d\n",pres_b[n],size-2);
      Board eB=B;
      //for(int e=w;e<(size-2)*(size-1)-w;e++){
      for(int e=w;e<lsize;e++){
      //下２行に置く
        B=eB;
        //printf("eloop:e:%d\n",e);
        //不要
        //int ea=pres_a[e];
        //int eb=pres_b[e];
        //置き換え
        //bool pea=board_placement(size,size-1,size-1-ea);
        //インライン
        //if(pea==false){
        if(board_placement(size,size-1,size-1-pres_a[e])==false){
          //printf("peaskip:N-1:%d:N-1-ea:%d\n",size-1,size-1-ea);
          continue;
        }
        //printf("placement_pea:xk(N-1):%d:y:%d\n",size-1,size-1-pres_a[e]);
        //置き換え
        //bool peb=board_placement(size,size-2,size-1-eb);
        //インライン
        //if(peb==false){
        if(board_placement(size,size-2,size-1-pres_b[e])==false){
          //printf("pebskip:N-2:%d:N-1-eb:%d\n",size-2,size-1-eb);
          continue;
        }
        //printf("placement_peb:xk(N-2):%d:y:%d\n",size-2,size-1-pres_b[e]);
        Board sB=B;
        //for(int s=w;s<(size-2)*(size-1)-w;s++){
        for(int s=w;s<lsize;s++){
        ////右２列に置く
          B=sB;
          //printf("sloop:s:%d\n",s);
          //
          //不要
          //int sa =pres_a[s];
          //int sb =pres_b[s];
          //
          //置き換え
          //bool psa=board_placement(size,size-1-sa,0);
          //インライン
          //if(psa==false){
          if(board_placement(size,size-1-pres_a[s],0)==false){
            //printf("psaskip:N-1-sa:%d:0\n",size-1-sa);
            continue;
          }
          //printf("psa:x:%d:yk(0):0\n",size-1-pres_a[s]);
          //bool psb=board_placement(size,size-1-sb,1);
          //if(psb==false){
          if(board_placement(size,size-1-pres_b[s],1)==false){
            //printf("psbskip:N-1-sb:%d:1\n",size-1-sb);
            continue;
          }
          //printf("psb:x:%d:yk(1):1\n",size-1-pres_b[s]);
          //printf("noskip\n");
          //printf("pwa:xk(0):0:y:%d\n",pres_a[w]);
          //printf("pwb:xk(1):1:y:%d\n",pres_b[w]);
          //printf("pna:x:%d:yk(N-1):%d\n",pres_a[n],size-1);
          //printf("pnb:x:%d:yk(N-2):%d\n",pres_b[n],size-2);
          //printf("pea:xk(N-1):%d:y:%d\n",size-1,size-1-pres_a[e]);
          //printf("peb:xk(N-2):%d:y:%d\n",size-2,size-1-pres_b[e]);
          //printf("psa:x:%d:yk(0):0\n",size-1-pres_a[s]);
          //printf("psb:x:%d:yk(1):1\n",size-1-pres_b[s]);
          //
          //// Check for minimum if n, e, s = (N-2)*(N-1)-1-w
          int ww=(size-2)*(size-1)-1-w;
          //新設
          int w2=(size-2)*(size-1)-1;
          //if(s==ww){
          if((s==ww)&&(n<(w2-e))){
          //check if flip about the up diagonal is smaller
            //if(n<(size-2)*(size-1)-1-e){
            //if(n<(w2-e)){
              continue;
            //}
          }
          //if(e==ww){
          if((e==ww)&&(n>(w2-n))){
            //check if flip about the vertical center is smaller
            //if(n>(size-2)*(size-1)-1-n){
            //if(n>(w2-n)){
              continue;       
            //}
          }
          //if(n==ww){
          if((n==ww)&&(e>(w2-s))){
            //// check if flip about the down diagonal is smaller
            //if(e>(size-2)*(size-1)-1-s){
            //if(e>(w2-s)){
              continue;
            //}
          }
          //// Check for minimum if n, e, s = w
          if(s==w){
            if((n!=w)||(e!=w)){
            // right rotation is smaller unless  w = n = e = s
            //右回転で同じ場合w=n=e=sでなければ値が小さいのでskip
              continue;
            }
            //printf("t0:%d,t1:%d,l0:%d,l1:%d,b0:%d,b1:%d,r0:%d,r1:%d\n",pres_a[w],pres_b[w],pres_a[n],pres_b[n],pres_a[e],pres_b[e],pres_a[s],pres_b[s]);
            //breakpoint(size,"上下左右２行２列配置完了",B.x,size-1);
            //w=n=e=sであれば90度回転で同じ可能性
            //この場合はミラーの2
            KOHO2++;
            process(size,B,COUNT2);
            //(*act)(board, Symmetry::ROTATE);
            continue;
          }
          if((e==w)&&(n>=s)){
            //if(n>=s){
            //e==wは180度回転して同じ
              if(n>s){
              //180度回転して同じ時n>=sの時はsmaller?
                continue;
              }
              //この場合は4
              //printf("t0:%d,t1:%d,l0:%d,l1:%d,b0:%d,b1:%d,r0:%d,r1:%d\n",pres_a[w],pres_b[w],pres_a[n],pres_b[n],pres_a[e],pres_b[e],pres_a[s],pres_b[s]);
            //breakpoint_nq27(size,"上下左右２行２列配置完了",B.x,size-1);
              KOHO4++;
              process(size,B,COUNT4);
              //(*act)(board, Symmetry::POINT);   
              continue;
            //}
          }
            printf("sym:2:t0:%d,t1:%d,l0:%d,l1:%d,b0:%d,b1:%d,r0:%d,r1:%d\n",pres_a[w],pres_b[w],pres_a[n],pres_b[n],pres_a[e],pres_b[e],pres_a[s],pres_b[s]);
            //breakpoint(size,"上下左右２行２列配置完了",B.x,size-1);
          KOHO8++;
          process(size,B,COUNT8);
          //(*act)(board, Symmetry::NONE);
          //この場合は8
          continue;
        }
      }    
    }
  }
  //printf("ROTATE_0:%d\n",cnt[ROTATE]);
  //printf("POINT_1:%d\n",cnt[POINT]);
  //printf("NONE_2:%d\n",cnt[NONE]);
  UNIQUE=cnt[COUNT2]+cnt[COUNT4]+cnt[COUNT8];
  TOTAL=cnt[COUNT2]*2+cnt[COUNT4]*4+cnt[COUNT8]*8;
}
//メインメソッド
int main(int argc,char** argv) {
  bool cpu=false,cpur=false,gpu=false,sgpu=false,nq=false;
  int argstart=1;
  /** パラメータの処理 */
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else if(argv[1][1]=='q'||argv[1][1]=='Q'){nq=true;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
    else if(argv[1][1]=='s'||argv[1][1]=='S'){sgpu=true;}
    else
      cpur=true;
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g|-r|-s|-q]\n",argv[0]);
    printf("  -c: CPU only\n");
    printf("  -r: CPUR only\n");
    printf("  -g: GPU only\n");
    printf("  -s: SGPU only\n");
    printf("  -q: nq27 only\n");
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  if(cpu){
    printf("\n\n６．CPU 非再帰 バックトラック＋ビットマップ\n");
  }else if(cpur){
    printf("\n\n６．CPUR 再帰 バックトラック＋ビットマップ\n");
  }else if(nq){
    printf("\n\n６．nq27 再帰 バックトラック＋ビットマップ\n");
  }else if(gpu){
    printf("\n\n６．GPU 非再帰 バックトラック＋ビットマップ\n");
  }else if(sgpu){
    printf("\n\n６．SGPU 非再帰 バックトラック＋ビットマップ\n");
  }
  if(cpu||cpur||nq){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    clock_t st;          //速度計測用
    char t[20];          //hh:mm:ss.msを格納
    int min=4;
    int targetN=18;
    //min=7;
    //targetN=7;
    int mask;
    long bmask;
    int aBoard[MAX];
    int bBoard[MAX];
    for(int i=min;i<=targetN;i++){
      TOTAL=0;
      UNIQUE=0;
      STEPCOUNT=0;
      KOHO2=0;
      KOHO4=0;
      KOHO8=0;
      for(int j=0;j<=2;j++){
        pre[j]=0;
        cnt[j]=0;
      }
      mask=((1<<i)-1);
      bmask=(1<<(i-1)|1<<(i-2)|2|1);
      bmask=((((bmask>>2)|(~0<<(i-4)))+1)<<(i-5))-1;
      st=clock();
      for(int j=0;j<i;j++){
        aBoard[j]=-1;
      }
      for(int j=0;j<i;j++){
        bBoard[j]=-1;
      }
      //
      //再帰
      if(cpur){ 
        //NQueenR(i,mask,0,0,0,0,aBoard,0,0,0,bBoard,bmask);//通常版
        NQueenR(i,mask,aBoard,bBoard,bmask);
      }
      if(cpu){ 
        //NQueenR(i,mask,0,0,0,0,aBoard,0,0,0,bBoard,bmask);//通常版
        NQueenR(i,mask,aBoard,bBoard,bmask);
      }
      //nq27
      if(nq){ 
        NQueen_nq27(i);
      }
      //
      TimeFormat(clock()-st,t);
      printf("%2d:%13ld%16ld%s:STEP:%ld:KOHO2:%ld:KOHO4:%ld:KOHO8:%ld\n",i,TOTAL,UNIQUE,t,STEPCOUNT,KOHO2,KOHO4,KOHO8);
    }
  }
  return 0;
}
