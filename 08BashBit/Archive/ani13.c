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
//変数宣言
long TOTAL=0;         //CPU,CPUR
long UNIQUE=0;        //CPU,CPUR
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
int symmetryOps(int size,int aBoard[]){
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
    if((topSide_0==-1)||(topSide_1==-1)||(leftSide_0==-1)||(leftSide_1==-1)||(bottomSide_0==-1)||(bottomSide_1==-1)||(rightSide_0==-1)||(rightSide_1==-1)){
     return 3;
    }
    int sizeE=size-1;
    int mtopSide_0=sizeE-topSide_0;
    int mtopSide_1=sizeE-topSide_1;
    int mleftSide_0=sizeE-leftSide_0;
    int mleftSide_1=sizeE-leftSide_1;
    int mbottomSide_0=sizeE-bottomSide_0;
    int mbottomSide_1=sizeE-bottomSide_1;
    int mrightSide_0=sizeE-rightSide_0;
    int mrightSide_1=sizeE-rightSide_1;
    //small判定
    if(((topSide_0 > leftSide_0)||((topSide_0==leftSide_0)&&(topSide_1 > leftSide_1)))||((topSide_0 > bottomSide_0)||((topSide_0==bottomSide_0)&&(topSide_1 > bottomSide_1)))||((topSide_0 > rightSide_0)||((topSide_0==rightSide_0)&&(topSide_1 > rightSide_1)))||((topSide_0 > mtopSide_0)||((topSide_0==mtopSide_0)&&(topSide_1 > mtopSide_1)))||((topSide_0 > mleftSide_0)||((topSide_0==mleftSide_0)&&(topSide_1 > mleftSide_1)))||((topSide_0 > mbottomSide_0)||((topSide_0==mbottomSide_0)&&(topSide_1 > mbottomSide_1)))||((topSide_0 > mrightSide_0)||((topSide_0==mrightSide_0)&&(topSide_1 > mrightSide_1)))){
      return 3;
    }
    //同じ時は反時計回りに９０度回転したものが同じかチェックする
    //top ==  right   left > top
    if((topSide_0==rightSide_0)&&(topSide_1==rightSide_1)){
      if((leftSide_0>topSide_0)||((leftSide_0==topSide_0)&&(leftSide_1>topSide_1))){
        return 3;
      }
    }
    //２辺が同じ場合は３辺目を比較する
    //top == right left==top  bottom>left
    if(((topSide_0==rightSide_0)&&(topSide_1==rightSide_1))&&((leftSide_0==topSide_0)&&(leftSide_1==topSide_1))){
      if((bottomSide_0>leftSide_0)||((bottomSide_0==leftSide_0)&&(bottomSide_1>leftSide_1))){
        return 3;
      }
    }
    //top ==  bottom  left > right
    if((topSide_0==bottomSide_0)&&(topSide_1==bottomSide_1)){
      if((leftSide_0>rightSide_0)||((leftSide_0==rightSide_0)&&(leftSide_1>rightSide_1))){
        return 3;
      }
    }
    //top ==  left    left > bottom
    if((topSide_0==leftSide_0)&&(topSide_1==leftSide_1)){
      if((leftSide_0>bottomSide_0)||((leftSide_0==bottomSide_0)&&(leftSide_1>bottomSide_1))){
        return 3;
      }
    }
    //top ==  left  left == bottom bottom > right
    if(((topSide_0==leftSide_0)&&(topSide_1==leftSide_1))&&((leftSide_0==bottomSide_0)&&(leftSide_1==bottomSide_1))){
      if((bottomSide_0>rightSide_0)||((bottomSide_0==rightSide_0)&&(bottomSide_1>rightSide_1))){
        return 3;
      }
    }

    //top == mtop     left > mleft
    if((topSide_0==mtopSide_0)&&(topSide_1==mtopSide_1)){
      if((leftSide_0>mleftSide_0)||((leftSide_0==mleftSide_0)&&(leftSide_1>mleftSide_1))){
        return 3;
      }
    }
    //top ==  mtop  left == mleft bottom > mbottom 
    if(((topSide_0==mtopSide_0)&&(topSide_1==mtopSide_1))&&((leftSide_0==mleftSide_0)&&(leftSide_1==mleftSide_1))){
      if((bottomSide_0>mbottomSide_0)||((bottomSide_0==mbottomSide_0)&&(bottomSide_1>mbottomSide_1))){
        return 3;
      }
    }
    //top == mright   left > mtop
    if((topSide_0==mrightSide_0)&&(topSide_1==mrightSide_1)){
      if((leftSide_0>mtopSide_0)||((leftSide_0==mtopSide_0)&&(leftSide_1>mtopSide_1))){
        return 3;
      }
    }
    //top ==  mright  left == mtop bottom > mleft 
    if(((topSide_0==mrightSide_0)&&(topSide_1==mrightSide_1))&&((leftSide_0==mtopSide_0)&&(leftSide_1==mtopSide_1))){
      if((bottomSide_0>mleftSide_0)||((bottomSide_0==mleftSide_0)&&(bottomSide_1>mleftSide_1))){
        return 3;
      }
    }
    //top == mbottom  left > mright
    if((topSide_0==mbottomSide_0)&&(topSide_1==mbottomSide_1)){
      if((leftSide_0>mrightSide_0)||((leftSide_0==mrightSide_0)&&(leftSide_1>mrightSide_1))){
        return 3;
      }
    }
    //top ==  mbottom  left == mright bottom > mtop 
    if(((topSide_0==mbottomSide_0)&&(topSide_1==mbottomSide_1))&&((leftSide_0==mrightSide_0)&&(leftSide_1==mrightSide_1))){
      if((bottomSide_0>mtopSide_0)||((bottomSide_0==mtopSide_0)&&(bottomSide_1>mtopSide_1))){
        return 3;
      }
    }
    //top == mleft    left > mbottom  
    if((topSide_0==mleftSide_0)&&(topSide_1==mleftSide_1)){
      if((leftSide_0>mbottomSide_0)||((leftSide_0==mbottomSide_0)&&(leftSide_1>mbottomSide_1))){
        return 3;
      }
    }
    //top ==  mleft  left == mbottom  bottom > mright 
    if(((topSide_0==mleftSide_0)&&(topSide_1==mleftSide_1))&&((leftSide_0==mbottomSide_0)&&(leftSide_1==mbottomSide_1))){
      if((bottomSide_0>mrightSide_0)||((bottomSide_0==mrightSide_0)&&(bottomSide_1>mrightSide_1))){
        return 3;
      }
    }
    //同じ時は反時計回りに９０度回転したものが同じかチェックする
    //printf("t0:%d,t1:%d,l0:%d,l1:%d,b0:%d,b1:%d,r0:%d,r1:%d\n",topSide_0,topSide_1,leftSide_0,leftSide_1,bottomSide_0,bottomSide_1,rightSide_0,rightSide_1);
    if((rightSide_0==topSide_0 && rightSide_1==topSide_1)&&(leftSide_0==topSide_0 && leftSide_1==topSide_1)&&(bottomSide_0==topSide_0 && bottomSide_1==topSide_1)){
      //printf("sym:0:t0:%d,t1:%d,l0:%d,l1:%d,b0:%d,b1:%d,r0:%d,r1:%d\n",topSide_0,topSide_1,leftSide_0,leftSide_1,bottomSide_0,bottomSide_1,rightSide_0,rightSide_1);
      return 0;
    }else if(bottomSide_0==topSide_0 && bottomSide_1==topSide_1){
      if(leftSide_0==rightSide_0&&leftSide_1==rightSide_1){
        //printf("sym:1:t0:%d,t1:%d,l0:%d,l1:%d,b0:%d,b1:%d,r0:%d,r1:%d\n",topSide_0,topSide_1,leftSide_0,leftSide_1,bottomSide_0,bottomSide_1,rightSide_0,rightSide_1);

        return 1;
      }
      //printf("sym:2:t0:%d,t1:%d,l0:%d,l1:%d,b0:%d,b1:%d,r0:%d,r1:%d\n",topSide_0,topSide_1,leftSide_0,leftSide_1,bottomSide_0,bottomSide_1,rightSide_0,rightSide_1);
      return 2;
    }else{
      //printf("sym:2:t0:%d,t1:%d,l0:%d,l1:%d,b0:%d,b1:%d,r0:%d,r1:%d\n",topSide_0,topSide_1,leftSide_0,leftSide_1,bottomSide_0,bottomSide_1,rightSide_0,rightSide_1);
      return 2;
    }
    
    //return 1;
    //symmetryOpsを実行する
    if(rightSide_0==topSide_0 && rightSide_1==topSide_1){
      if(((leftSide_0 !=topSide_0)||(leftSide_1!=topSide_1))||((bottomSide_0!=topSide_0)||(bottomSide_1!=topSide_1))){

        return 3;
      }
      //printf("t0:%d,t1:%d,l0:%d,l1:%d,b0:%d,b1:%d,r0:%d,r1:%d\n",topSide_0,topSide_1,leftSide_0,leftSide_1,bottomSide_0,bottomSide_1,rightSide_0,rightSide_1);
      return 0;
    }
    if((bottomSide_0==topSide_0)&&(bottomSide_1==topSide_1)&&((leftSide_0>=rightSide_0)||(leftSide_1>=rightSide_1))){
      if((leftSide_0>rightSide_0)||(leftSide_1>rightSide_1)){
        return 3;
      }
      //printf("t0:%d,t1:%d,l0:%d,l1:%d,b0:%d,b1:%d,r0:%d,r1:%d\n",topSide_0,topSide_1,leftSide_0,leftSide_1,bottomSide_0,bottomSide_1,rightSide_0,rightSide_1);
      return 1;
    }



    return 2;
}
void backTrackR1(int size,long mask,int row,long left,long down,long right,int aBoard[],long lleft,long ldown,long lright,int bBoard[],long bmask){
  //printf("a:nqueen start\n");
  int bitmap=0;
  int bit=0;
  if(row==size){
    //printf("t0:%d,t1:%d,l0:%d,l1:%d,b0:%d,b1:%d,r0:%d,r1:%d\n",topSide_0,topSide_1,leftSide_0,leftSide_1,bottomSide_0,bottomSide_1,rightSide_0,rightSide_1);
    //printf("b:all placed\n");
    //breakpoint(size,"上下左右２行２列配置完了",bBoard,row,bit);
    int sym=symmetryOps(size,aBoard);
    if(sym!=3){
      //breakpoint(size,"上下左右２行２列配置完了",bBoard,row,bit);
       int q=bit93_countCompletions(size,2,aBoard,lleft>>4,((((ldown>>2)|(~0<<(size-4)))+1)<<(size-5))-1,(lright>>4)<<(size-5),sym,bBoard,bmask);
       UNIQUE+=q; 
       if(sym==0){
         TOTAL+=q*2;
       }else if(sym==1){
         TOTAL+=q*4;
       }else if(sym==2){
         TOTAL+=q*8;
       }

    }
    //symmetryOpsを実行する
    //aBoardの内容から
    //上下左右２行２列に配置が完了したら3行目からcountCompletionsを呼び出して上からやる。
    //if(sym !=3){
    //  int rtn=bit93_countCompletions(size,2,aBoard,lleft>>4,((((ldown>>2)|(~0<<(size-4)))+1)<<(size-5))-1,(lright>>4)<<(size-5),sym,bBoard,bmask);
    //  if(sym==0){
    //    TOTAL+=rtn*2;
    //    UNIQUE+=rtn; 
    //  }else if(sym==1){
    //    TOTAL+=rtn*4;
    //    UNIQUE+=rtn; 
    //  }else if(sym==2){
    //    TOTAL+=rtn*8;
    //    UNIQUE+=rtn; 
    //  }
    //}
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
        //上２行、下２行以外で左端２列、右端２列以外にクイーンを置ける余地がある場合はその行を空白にして次の行へ進むルートも作る
        backTrackR1(size,mask,row+1,(left)<<1, down,(right)>>1,aBoard,lleft,ldown,lright,bBoard,bmask);
      }
      //左端２列、右端２列にだけクイーンを置くようにマスクを設定する
      amask=amask^mask;
    }
    //printf("row:%d bbbbamask:%d left:%d down:%d right:%d\n",row,amask,left,down,right);
    bitmap=(amask&~(left|down|right));
    //printf("bitmap:%d\n",bitmap);
    while(bitmap){
      //printf("c:place\n");
      //クイーンを置く
      bitmap^=bBoard[row]=bit=(-bitmap&bitmap);
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
     //printf("row:%d,lleft:%d,ldown:%d,lright:%d\n",row,lleft,ldown,right);
      //printf("d:recursivele start\n");
      backTrackR1(size,mask,row+1,(left|bit)<<1, (down|bit),(right|bit)>>1,aBoard,lleft|1<<(size-1-row+x),ldown|bit,lright|1<<(row+x),bBoard,bmask);
      //printf("e:recursivele start\n");
    }
  }
}
void backTrackR2(int size,long mask,int row,long left,long down,long right,int aBoard[],long lleft,long ldown,long lright,int bBoard[],long bmask){
  //printf("a:nqueen start\n");
  int bitmap=0;
  int bit=0;
  if(row==size){
    //printf("t0:%d,t1:%d,l0:%d,l1:%d,b0:%d,b1:%d,r0:%d,r1:%d\n",topSide_0,topSide_1,leftSide_0,leftSide_1,bottomSide_0,bottomSide_1,rightSide_0,rightSide_1);
    //printf("b:all placed\n");
    //breakpoint(size,"上下左右２行２列配置完了",bBoard,row,bit);
    int sym=symmetryOps(size,aBoard);
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
    //symmetryOpsを実行する
    //aBoardの内容から
    //上下左右２行２列に配置が完了したら3行目からcountCompletionsを呼び出して上からやる。
    //if(sym !=3){
    //  int rtn=bit93_countCompletions(size,2,aBoard,lleft>>4,((((ldown>>2)|(~0<<(size-4)))+1)<<(size-5))-1,lright=(lright>>4)<<(size-5),sym,bBoard,bmask);
    //  if(sym==0){
    //    TOTAL+=rtn*2;
    //    UNIQUE+=rtn; 
    //  }else if(sym==1){
    //    TOTAL+=rtn*4;
    //    UNIQUE+=rtn; 
    //  }else if(sym==2){
    //    TOTAL+=rtn*8;
    //    UNIQUE+=rtn; 
    //  }
    //}
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
        //上２行、下２行以外で左端２列、右端２列以外にクイーンを置ける余地がある場合はその行を空白にして次の行へ進むルートも作る
        backTrackR2(size,mask,row+1,(left)<<1, down,(right)>>1,aBoard,lleft,ldown,lright,bBoard,bmask);
      }
      //左端２列、右端２列にだけクイーンを置くようにマスクを設定する
      amask=amask^mask;
    }
    //printf("row:%d bbbbamask:%d left:%d down:%d right:%d\n",row,amask,left,down,right);
    bitmap=(amask&~(left|down|right));
    //printf("bitmap:%d\n",bitmap);
    while(bitmap){
      //printf("c:place\n");
      //クイーンを置く
      bitmap^=bBoard[row]=bit=(-bitmap&bitmap);
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
      backTrackR2(size,mask,row+1,(left|bit)<<1, (down|bit),(right|bit)>>1,aBoard,lleft|1<<(size-1-row+x),ldown|bit,lright|1<<(row+x),bBoard,bmask);
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
  int col=0;
  bBoard[0]=bit=(1<<col);
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
  long left=bit<<1;
  long down=bit;
  long right=bit>>1;
  long lleft=1<<(size-1+x);
  long ldown=bit;
  long lright=1<<(x);
  //2行目は右から3列目から左端から2列目まで
  for(int col_j=2;col_j<size-1;col_j++){
      bBoard[1]=bit=(1<<col_j);
      x=0;
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
      backTrackR1(size,mask,2,(left|bit)<<1,(down|bit),(right|bit)>>1,aBoard,lleft|1<<(size-1-1+x),ldown|bit,lright|1<<(1+x),bBoard,bmask);
  }

  //backtrack2
  //1行目右から2列目から
  //偶数個は1/2 n=8 なら 1,2,3 奇数個は1/2+1 n=9 なら 1,2,3,4
  for(int col=1,col2=size-2;col<col2;col++,col2--){
      bBoard[0]=bit=(1<<col);
      x=0;
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
      backTrackR2(size,mask,1,bit<<1,bit,bit>>1,aBoard,1<<(size-1+x),bit,1<<(x),bBoard,bmask);
  }

}
//メインメソッド
int main(int argc,char** argv) {
  bool cpu=false,cpur=false,gpu=false,sgpu=false;
  int argstart=1;
  /** パラメータの処理 */
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
    else if(argv[1][1]=='s'||argv[1][1]=='S'){sgpu=true;}
    else
      cpur=true;
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g|-r|-s]\n",argv[0]);
    printf("  -c: CPU only\n");
    printf("  -r: CPUR only\n");
    printf("  -g: GPU only\n");
    printf("  -s: SGPU only\n");
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  if(cpu){
    printf("\n\n６．CPU 非再帰 バックトラック＋ビットマップ\n");
  }else if(cpur){
    printf("\n\n６．CPUR 再帰 バックトラック＋ビットマップ\n");
  }else if(gpu){
    printf("\n\n６．GPU 非再帰 バックトラック＋ビットマップ\n");
  }else if(sgpu){
    printf("\n\n６．SGPU 非再帰 バックトラック＋ビットマップ\n");
  }
  if(cpu||cpur){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    clock_t st;          //速度計測用
    char t[20];          //hh:mm:ss.msを格納
    int min=4;
    int targetN=18;
    //min=8;
    //targetN=8;
    int mask;
    long bmask;
    int aBoard[MAX];
    int bBoard[MAX];
    for(int i=min;i<=targetN;i++){
      TOTAL=0;
      UNIQUE=0;
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
      //非再帰
      if(cpu){ 
        //NQueenR(i,mask,0,0,0,0,aBoard,0,0,0,bBoard,bmask);//通常版
        NQueenR(i,mask,aBoard,bBoard,bmask);
      }
      //
      TimeFormat(clock()-st,t);
      printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
    }
  }
  return 0;
}
