/*  
Jeff Somers Copyright(c) 2002 jsomers@alumni.williams.edu or allagash98@yahoo.com April,2002

Program:  nq

build and execute.
$ gcc -Wall nq.c -o nq && ./nq 

Program to find size of solutions to the N queens problem.
Nクイーン問題の解の大きさを求めるプログラム。
This program assumes a twos complement architecture.
このプログラムは、2つの補数アーキテクチャを仮定しています。

For example,you can arrange 4 queens on 4 x 4 chess so that none of the queens can attack each other:
例えば、4×4のチェス版に4つのクイーンを並べて、どのクイーンも互いに攻撃できないようにすることができます：

Two solutions:
２つの解決策:
_ Q _ _        _ _ Q _
_ _ _ Q        Q _ _ _
Q _ _ _        _ _ _ Q
_ _ Q _    and _ Q _ _

Note that these are separate solutions,even though they are mirror images of each other.
なお、これらは鏡像であっても別解である。

Likewise,a 8 x 8 chess board has 92 solutions to the 8 queens problem.
同様に、8×8のチェス盤には、8つのクイーンの問題に対して92の解があります。


Command Line Usage:
コマンドラインの使い方:
$ gcc -Wall nq.c -o nq && ./nq

where N is the size of the N x N board.  For example,nq 4 will find the 4 queen solution for the 4 x 4 chess board.
ここで、NはN×N盤の大きさである。 例えば、nq 4は4×4のチェス盤に対して4クイーンの解を求めます。

By default,this program will only print the size of solutions,not board arrangements which are the solutions.  To print the boards,uncomment the call to printtable in the Nqueen function.  Note that printing the board arrangements slows down the program quite a bit,unless you pipe the output to a text file:
デフォルトでは、このプログラムは解の大きさのみを印刷し、解となるボードの配置は印刷しません。 盤面を印刷するには、Nqueen関数のprinttableの呼び出しをアンコメントします。 ボードアレンジメントを印刷すると、出力をテキストファイルにパイプしない限り、プログラムの動作がかなり遅くなることに注意してください：

Command Line Usage:
コマンドラインの使い方:
nq > output.txt


The size of solutions for the N queens problems are known for boards up to 23 x 23.  With this program,I've calculated the results for boards up to 21 x 21,and that took over a week on an 800 MHz PC.  The algorithm is approximated O(n!)(i.e. slow),and calculating the results for a 22 x 22 board will take about 8.5 times the amount of time for the 21 x 21 board,or over 8 1/2 weeks.  Even with a 10 GHz machine,calculating the results for a 23 x 23 board would take over a month.  Of course,setting up a cluster of machines(or a distributed client) would do the work in less time.
N個の女王の問題の解の大きさは、23×23までのボードについて知られています。 このプログラムでは、21×21までのボードについて計算したことがあるが、800MHzのPCで1週間以上かかった。 このアルゴリズムは近似的にO(n!)（つまり遅い）であり、22×22のボードの結果を計算するには、21×21のボードの約8.5倍の時間、つまり8週間半以上かかる。 10GHzのマシンでも、23×23のボードの結果を計算すると、1ヶ月以上かかる。 もちろん、マシンのクラスタ（または分散型クライアント）をセットアップすれば、より短時間で作業を行うことができます。


(from Sloane's On-Line Encyclopedia of Integer Sequences,Sequence A000170 http://www.research.att.com/cgi-bin/access.cgi/as/njas/sequences/eisA.cgi?Anum=000170)

Board Size:       size of Solutions to          Time to calculate 
(length of one        N queens problem:              on 800MHz PC
side of N x N                                   (Hours:Mins:Secs)
chessboard)

1                                  1                    n/a
2                                  0                  <0 seconds
3                                  0                  <0 seconds
4                                  2                  <0 seconds
5                                 10                  <0 seconds
6                                  4                  <0 seconds
7                                 40                  <0 seconds
8                                 92                  <0 seconds
9                                352                  <0 seconds 
10                                724                  <0 seconds
11                               2680                  <0 seconds
12                              14200                  <0 seconds
13                              73712                  <0 seconds
14                             365596                  00:00:01
15                            2279184                  00:00:04
16                           14772512                  00:00:23
17                           95815104                  00:02:38
18                          666090624                  00:19:26
19                         4968057848                  02:31:24
20                        39029188884                  20:35:06
21                       314666222712                 174:53:45
22                      2691008701644                     ?
23                     24233937684440                     ?
24                                  ?                     ?
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* 
  Notes on MAX:
  MAXに関する注意事項：
  A 32 bit unsigned long is sufficient to hold the results for an 18 x 18 board(666090624 solutions) but not for a 19 x 19 board(4968057848 solutions).
  18×18ボード(666090624解)の結果を保持するには32bit unsigned longで十分ですが、19×19ボード(4968057848解)の結果を保持するには不十分です。

  In Win32,I use a 64 bit variable to hold the results,and merely set the MAX to 21 because that's the largest board for which I've calculated a result.
  Win32では、64bitの変数で結果を保持し、MAXを21にしているのは、結果を算出した最大のボードだからです。

  Note: a 20x20 board will take over 20 hours to run on a Pentium III 800MHz,while a 21x21 board will take over a week to run on the same PC. 
  注：20x20ボードはPentium III 800MHzで動作させると20時間以上、21x21ボードは同じPCで動作させると1週間以上かかると言われています。

  On Unix,you could probably change the type of g_numsolutions from unsigned long to unsigned long long,or change the code to use two 32 bit ints to store the results for board sizes 19 x 19 and up.
  Unixでは、g_numsolutionsの型をunsigned longからunsigned long longに変更するか、ボードサイズ19×19以上の結果を格納するために2つの32ビットintを使用するようにコードを変更すればよいでしょう。
*/

#ifdef WIN32

#define MAX 21
typedef unsigned __int64 SOLUTIONTYPE;

#else

#define MAX 18
typedef unsigned long SOLUTIONTYPE;

#endif

#define MIN 2

SOLUTIONTYPE g_numsolutions=0;


/* Print a chess table with queens positioned for a solution */
/* 解決のためにクイーンが配置されたチェステーブルを印刷する */

/* This is not a critical path function & I didn't try to optimize it. */
/* これはクリティカルパス関数ではないので、最適化はしていません。*/
void printtable(int size,int* board,SOLUTIONTYPE numSolution)
{
  int i,j,k,row;
  /*
  We only calculated half the solutions,because we can derive the other half by reflecting the solution across the "Y axis". 
  解を半分だけ計算したのは、解を「Y軸」を挟んで反転させることで残りの半分を導き出すことができるからです。
  */
  for(k=0;k<2;++k){
#ifdef WIN32
    printf("*** Solution #: %I64d ***\n",2 * numSolution+k-1);
#else
    printf("*** Solution #: %ld ***\n",2 * numSolution+k-1);
#endif
    for(i=0;i<size;i++){
      unsigned int bitf;
      /*
        Get the column that was set(i.e. find the first,least significant,bit set).
        If board[i]=011010b,then bitf=000010b
        設定された列を取得する（つまり、設定された最初の、最下位の、ビットを見つける）。
        board[i]=011010bの場合、bitf=000010b
      */
      bitf=board[i];
      /* get least significant bit */
      /* 最下位ビットを取得する */
      row=bitf^(bitf &(bitf-1));
      for(j=0;j<size;j++){
        /* 
         keep shifting row over to the right until we find the one '1' in the binary representation.  There will only be one '1'. 
         1を見つけるまで、行を右にずらし続けます。1は1つしかありません。
        */
        if(0==k &&((row>>j) & 1)){
          printf("Q ");
        }
        /* this is the board reflected across the "Y axis" */
        /* Y軸方向に映し出されたボードです。*/
        else if(1==k&&(row&(1<<(size-j-1)))){
          printf("Q ");
        }else{
          printf(". ");
        }
      }
      printf("\n");
    }
    printf("\n");
  }
}
/* 
The function which calculates the N queen solutions.We calculate one-half the solutions,then flip the results over the "Y axis" of the board.  Every solution can be reflected that way to generate another unique solution(assuming the board size isn't 1 x 1).  That's because a solution cannot be symmetrical across the Y-axis(because you can't have two queens in the same horizontal row).  A solution also cannot consist of queens down the middle column of a board with an odd size of columns,since you can't have two queens in the same vertical row.
N個のクイーン解を計算する関数です。解の2分の1を計算し、その結果を盤面の「Y軸」の上に反転させます。 すべての解は、そのように反射して別のユニークな解を生成することができます（ボードのサイズが1×1でないと仮定します）。 それは、解がY軸を挟んで対称にならないからです（同じ横列に2つのクイーンを置くことはできないからです）。 また、同じ縦列に2つのクイーンを置くことはできないので、奇数列のボードの中央列をクイーンで構成することもできない。

This is a backtracking algorithm.  We place a queen in the top row,then note the column and diagonals it occupies.  We then place a queen in the next row down,taking care not to place it in the same column or diagonal.  We then update the occupied columns & diagonals & move on to the next row.  If no position is open in the next row,we back track to the previous row & move the queen over to the next available spot in its row & the process starts over again.
これはバックトラックのアルゴリズムである。 一番上の行にクイーンを置き、そのクイーンが占める列と対角線に注意する。 次に、同じ列や対角線にクイーンを配置しないように注意しながら、次の行にクイーンを配置します。 その後、占有している列と対角線を更新し、次の行に進む。 次の行に空きがない場合は、前の行に戻り、クイーンをその行の次の空いた場所に移動させ、また同じことを繰り返します。
*/
void Nqueen(int size)
{
  int board[MAX];/* results */
  int down[MAX];/* marks colummns which already have queens */
  int left[MAX];/* marks "positive diagonals" which already have queens */
  int right[MAX];/* marks "negative diagonals" which already have queens */
  int aStack[MAX+2];/* we use a stack instead of recursion */
  register int* pnStack;
  register int row=0;/* row redundant-could use stack */
  register unsigned int bit;/* least significant bit */
  register unsigned int bitmap;/* bits which are set mark possible positions for a queen */
  int odd=size&1;/* 0 if size even,1 if odd */
  int sizeE=size-1;/* board size-1 */
  int mask=(1<<size)-1;/* if board size is N,mask consists of N 1's */
  /* Initialize stack */
  /* スタックの初期化 */
  aStack[0]=-1;/* set sentinel -- signifies end of stack */
  /* NOTE:(size & 1) is true iff size is odd */
  /* 注意：(size & 1)は、sizeが奇数であれば真となる */
  /* We need to loop through 2x if size is odd */
  /* sizeが奇数の場合、2xをループする必要がある */
  for(int i=0;i <(1+odd);++i) {
    /* We don't have to optimize this part;it ain't the critical loop */
    /* この部分は最適化する必要はありません;重要なループではありません */
    bitmap=0;
    if(0==i){
      /* 
      Handle half of the board,except the middle column. So if the board is 5 x 5,the first row will be: 00011,since we're not worrying about placing a queen in the center column(yet).
      ボードの真ん中の列を除く半分を処理します。つまり、盤面が5×5の場合、1列目はこうなります： 中央の列にクイーンを置く心配は（まだ）ないので、00011となります。
      */
      int half=size>>1;/* divide by two */
      /* 
      fill in rightmost 1's in bitmap for half of size If size is 7,half of that is 3(we're discarding the remainder) and bitmap will be set to 111 in binary. 
      sizeの半分のbitmapの右端の1を埋める sizeが7の場合、その半分は3（残りは捨てる）、bitmapは2進数で111に設定されます。
      */
      bitmap=(1<<half)-1;
      pnStack=aStack+1;/* stack pointer */
      board[0]=0;
      down[0]=left[0]=right[0]=0;
    }else{
      /* 
      Handle the middle column(of a odd-sized board).  Set middle column bit to 1,then set half of next row.  So we're processing first row(one element) & half of next.  So if the board is 5 x 5,the first row will be: 00100,and the next row will be 00011.
      奇数サイズのボードの中列を処理する。 中列のビットを1にセットし、次の行の半分をセットする。 つまり、最初の行（1つの要素）と次の行の半分を処理することになります。 つまり、盤面が5×5の場合、最初の行はこうなります： 00100となり、次の行は00011となります。
      */
      bitmap=1 <<(size>>1);
      row=1;/* prob. already 0 */
      /* The first row just has one queen(in the middle column).*/
      /* 最初の行にはクイーンが1人（真ん中の列）だけです*/
      board[0]=bitmap;
      down[0]=left[0]=right[0]=0;
      down[1]=bitmap;
      /* 
      Now do the next row.  Only set bits in half of it,because we'll flip the results over the "Y-axis".
      次の行を実行します。 Y軸の結果を反転させるので、半分だけビットを設定します。 
      */
      right[1]=(bitmap>>1);
      left[1]=(bitmap<<1);
      pnStack=aStack+1;/* stack pointer */
      *pnStack++=0;/* we're done w/ this row -- only 1 element & we've done it */
      bitmap=(bitmap-1)>>1;/* bitmap -1 is all 1's to the left of the single 1 */
    }
    /* this is the critical loop */
    /* これがクリティカルループです */
    for(;;){
      /* 
      could use bit=bitmap ^(bitmap &(bitmap -1));
      to get first(least sig) "1" bit,but that's slower. 
      bit=bitmap ^(bitmap &(bitmap -1))を使うことができます。
      最初の(最小信号の)「1」ビットを得ることができますが、これはより遅いです。
      */
      bit=-((signed)bitmap) & bitmap;/* this assumes a 2's complement architecture */
      if(0==bitmap){
        bitmap=*--pnStack;/* get prev. bitmap from stack */
        if(pnStack==aStack){ /* if sentinel hit.... */
          break ;
        }
        --row;
        continue;
      }
      /* toggle off this bit so we don't try it again */
      /* 再試行しないように、このビットをトグルオフにする */
      bitmap&=~bit;
      board[row]=bit;/* save the result */
      /* we still have more rows to process? */
      /* まだ処理する行があるのでは？*/
      if(row<sizeE){
        row++;
        down[row]=down[row-1]|bit;
        right[row]=(right[row-1]|bit)>>1;
        left[row]=(left[row-1]|bit)<<1;
        *pnStack++=bitmap;
        /* 
        We can't consider positions for the queen which are in the same column,same positive diagonal,or same negative diagonal as another queen already on the board. 
        すでにボード上にある他のクイーンと同じ列、同じ正対角、同じ負対角にあるクイーンのポジションは考慮しない。
        */
        bitmap=mask&~(down[row]|right[row]|left[row]);
        continue;
      }else{
        /* 
        We have no more rows to process;we found a solution. Comment out the call to printtable in order to print the solutions as board position printtable(size,board,g_numsolutions+1); 
        もう処理する行はない。解決策を見つけた。解を盤面位置として表示するために、printtableの呼び出しをコメントアウトする printtable(size,board,g_numsolutions+1)； 
        */
        ++g_numsolutions;
        bitmap=*--pnStack;
        --row;
        continue;
      }
    }
  }
  /* multiply solutions by two,to count mirror images */
  /* 鏡像の数を数えるために、解を2倍する */
  g_numsolutions *= 2;
}
/* Print the results at the end of the run */
/* 実行終了時に結果を表示します */
void TimeFormat(float ftime,char *form)
{
  int  itime,dd,hh,mm;
  float ss;
  itime=(int)ftime;
  mm=itime / 60;
  ss=ftime -(float)(mm * 60);
  dd=mm /(24 * 60);
  mm=mm %(24 * 60);
  hh=mm / 60;
  mm=mm % 60;
  if(dd)
    sprintf(form,"%4d %02d:%02d:%05.2f",dd,hh,mm,ss);
  else if(hh)
    sprintf(form,"     %2d:%02d:%05.2f",hh,mm,ss);
  else if(mm)
    sprintf(form,"        %2d:%05.2f",mm,ss);
  else
    sprintf(form,"           %5.2f",ss);
}
/* main routine for N Queens program.*/
/* N Queensプログラムのメインルーチン */
int main(int argc,char** argv)
{
  int size;
  clock_t starttime;
  char form[20];
  printf("<--- N-Queens ----> <---- time ---->\n");
  printf(" N: Total Solutions days hh:mm:ss.--\n");
  for(size=MIN;size<=MAX;size++) {
    starttime=clock();
    g_numsolutions=0;
    Nqueen(size);/* find solutions */
    TimeFormat((float)(clock()-starttime) / CLOCKS_PER_SEC,form);
#ifdef WIN32
    printf("%2d:%16I64d %s\n",size,g_numsolutions,form);
#else
    printf("%2d:%16ld %s\n",size,g_numsolutions,form);
#endif
  }
  return 0;
}
