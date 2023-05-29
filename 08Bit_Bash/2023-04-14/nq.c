/*  
Jeff Somers Copyright(c) 2002 jsomers@alumni.williams.edu or allagash98@yahoo.com April,2002

Program:  nq

build and execute.
$ gcc -Wall nq.c -o nq && ./nq 

Program to find size of solutions to the N queens problem.
N$B%/%$!<%sLdBj$N2r$NBg$-$5$r5a$a$k%W%m%0%i%`!#(B
This program assumes a twos complement architecture.
$B$3$N%W%m%0%i%`$O!"(B2$B$D$NJd?t%"!<%-%F%/%A%c$r2>Dj$7$F$$$^$9!#(B

For example,you can arrange 4 queens on 4 x 4 chess so that none of the queens can attack each other:
$BNc$($P!"(B4$B!_(B4$B$N%A%'%9HG$K(B4$B$D$N%/%$!<%s$rJB$Y$F!"$I$N%/%$!<%s$b8_$$$K967b$G$-$J$$$h$&$K$9$k$3$H$,$G$-$^$9!'(B

Two solutions:
$B#2$D$N2r7h:v(B:
_ Q _ _        _ _ Q _
_ _ _ Q        Q _ _ _
Q _ _ _        _ _ _ Q
_ _ Q _    and _ Q _ _

Note that these are separate solutions,even though they are mirror images of each other.
$B$J$*!"$3$l$i$O6@A|$G$"$C$F$bJL2r$G$"$k!#(B

Likewise,a 8 x 8 chess board has 92 solutions to the 8 queens problem.
$BF1MM$K!"(B8$B!_(B8$B$N%A%'%9HW$K$O!"(B8$B$D$N%/%$!<%s$NLdBj$KBP$7$F(B92$B$N2r$,$"$j$^$9!#(B


Command Line Usage:
$B%3%^%s%I%i%$%s$N;H$$J}(B:
$ gcc -Wall nq.c -o nq && ./nq

where N is the size of the N x N board.  For example,nq 4 will find the 4 queen solution for the 4 x 4 chess board.
$B$3$3$G!"(BN$B$O(BN$B!_(BN$BHW$NBg$-$5$G$"$k!#(B $BNc$($P!"(Bnq 4$B$O(B4$B!_(B4$B$N%A%'%9HW$KBP$7$F(B4$B%/%$!<%s$N2r$r5a$a$^$9!#(B

By default,this program will only print the size of solutions,not board arrangements which are the solutions.  To print the boards,uncomment the call to printtable in the Nqueen function.  Note that printing the board arrangements slows down the program quite a bit,unless you pipe the output to a text file:
$B%G%U%)%k%H$G$O!"$3$N%W%m%0%i%`$O2r$NBg$-$5$N$_$r0u:~$7!"2r$H$J$k%\!<%I$NG[CV$O0u:~$7$^$;$s!#(B $BHWLL$r0u:~$9$k$K$O!"(BNqueen$B4X?t$N(Bprinttable$B$N8F$S=P$7$r%"%s%3%a%s%H$7$^$9!#(B $B%\!<%I%"%l%s%8%a%s%H$r0u:~$9$k$H!"=PNO$r%F%-%9%H%U%!%$%k$K%Q%$%W$7$J$$8B$j!"%W%m%0%i%`$NF0:n$,$+$J$jCY$/$J$k$3$H$KCm0U$7$F$/$@$5$$!'(B

Command Line Usage:
$B%3%^%s%I%i%$%s$N;H$$J}(B:
nq > output.txt


The size of solutions for the N queens problems are known for boards up to 23 x 23.  With this program,I've calculated the results for boards up to 21 x 21,and that took over a week on an 800 MHz PC.  The algorithm is approximated O(n!)(i.e. slow),and calculating the results for a 22 x 22 board will take about 8.5 times the amount of time for the 21 x 21 board,or over 8 1/2 weeks.  Even with a 10 GHz machine,calculating the results for a 23 x 23 board would take over a month.  Of course,setting up a cluster of machines(or a distributed client) would do the work in less time.
N$B8D$N=w2&$NLdBj$N2r$NBg$-$5$O!"(B23$B!_(B23$B$^$G$N%\!<%I$K$D$$$FCN$i$l$F$$$^$9!#(B $B$3$N%W%m%0%i%`$G$O!"(B21$B!_(B21$B$^$G$N%\!<%I$K$D$$$F7W;;$7$?$3$H$,$"$k$,!"(B800MHz$B$N(BPC$B$G(B1$B=54V0J>e$+$+$C$?!#(B $B$3$N%"%k%4%j%:%`$O6a;wE*$K(BO(n!)$B!J$D$^$jCY$$!K$G$"$j!"(B22$B!_(B22$B$N%\!<%I$N7k2L$r7W;;$9$k$K$O!"(B21$B!_(B21$B$N%\!<%I$NLs(B8.5$BG\$N;~4V!"$D$^$j(B8$B=54VH>0J>e$+$+$k!#(B 10GHz$B$N%^%7%s$G$b!"(B23$B!_(B23$B$N%\!<%I$N7k2L$r7W;;$9$k$H!"(B1$B%v7n0J>e$+$+$k!#(B $B$b$A$m$s!"%^%7%s$N%/%i%9%?!J$^$?$OJ,;67?%/%i%$%"%s%H!K$r%;%C%H%"%C%W$9$l$P!"$h$jC;;~4V$G:n6H$r9T$&$3$H$,$G$-$^$9!#(B


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
  MAX$B$K4X$9$kCm0U;v9`!'(B
  A 32 bit unsigned long is sufficient to hold the results for an 18 x 18 board(666090624 solutions) but not for a 19 x 19 board(4968057848 solutions).
  18$B!_(B18$B%\!<%I(B(666090624$B2r(B)$B$N7k2L$rJ];}$9$k$K$O(B32bit unsigned long$B$G==J,$G$9$,!"(B19$B!_(B19$B%\!<%I(B(4968057848$B2r(B)$B$N7k2L$rJ];}$9$k$K$OIT==J,$G$9!#(B

  In Win32,I use a 64 bit variable to hold the results,and merely set the MAX to 21 because that's the largest board for which I've calculated a result.
  Win32$B$G$O!"(B64bit$B$NJQ?t$G7k2L$rJ];}$7!"(BMAX$B$r(B21$B$K$7$F$$$k$N$O!"7k2L$r;;=P$7$?:GBg$N%\!<%I$@$+$i$G$9!#(B

  Note: a 20x20 board will take over 20 hours to run on a Pentium III 800MHz,while a 21x21 board will take over a week to run on the same PC. 
  $BCm!'(B20x20$B%\!<%I$O(BPentium III 800MHz$B$GF0:n$5$;$k$H(B20$B;~4V0J>e!"(B21x21$B%\!<%I$OF1$8(BPC$B$GF0:n$5$;$k$H(B1$B=54V0J>e$+$+$k$H8@$o$l$F$$$^$9!#(B

  On Unix,you could probably change the type of g_numsolutions from unsigned long to unsigned long long,or change the code to use two 32 bit ints to store the results for board sizes 19 x 19 and up.
  Unix$B$G$O!"(Bg_numsolutions$B$N7?$r(Bunsigned long$B$+$i(Bunsigned long long$B$KJQ99$9$k$+!"%\!<%I%5%$%:(B19$B!_(B19$B0J>e$N7k2L$r3JG<$9$k$?$a$K(B2$B$D$N(B32$B%S%C%H(Bint$B$r;HMQ$9$k$h$&$K%3!<%I$rJQ99$9$l$P$h$$$G$7$g$&!#(B
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
/* $B2r7h$N$?$a$K%/%$!<%s$,G[CV$5$l$?%A%'%9%F!<%V%k$r0u:~$9$k(B */

/* This is not a critical path function & I didn't try to optimize it. */
/* $B$3$l$O%/%j%F%#%+%k%Q%94X?t$G$O$J$$$N$G!":GE,2=$O$7$F$$$^$;$s!#(B*/
void printtable(int size,int* board,SOLUTIONTYPE numSolution)
{
  int i,j,k,row;
  /*
  We only calculated half the solutions,because we can derive the other half by reflecting the solution across the "Y axis". 
  $B2r$rH>J,$@$17W;;$7$?$N$O!"2r$r!V(BY$B<4!W$r64$s$GH?E>$5$;$k$3$H$G;D$j$NH>J,$rF3$-=P$9$3$H$,$G$-$k$+$i$G$9!#(B
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
        $B@_Dj$5$l$?Ns$r<hF@$9$k!J$D$^$j!"@_Dj$5$l$?:G=i$N!":G2<0L$N!"%S%C%H$r8+$D$1$k!K!#(B
        board[i]=011010b$B$N>l9g!"(Bbitf=000010b
      */
      bitf=board[i];
      /* get least significant bit */
      /* $B:G2<0L%S%C%H$r<hF@$9$k(B */
      row=bitf^(bitf &(bitf-1));
      for(j=0;j<size;j++){
        /* 
         keep shifting row over to the right until we find the one '1' in the binary representation.  There will only be one '1'. 
         1$B$r8+$D$1$k$^$G!"9T$r1&$K$:$i$7B3$1$^$9!#(B1$B$O(B1$B$D$7$+$"$j$^$;$s!#(B
        */
        if(0==k &&((row>>j) & 1)){
          printf("Q ");
        }
        /* this is the board reflected across the "Y axis" */
        /* Y$B<4J}8~$K1G$7=P$5$l$?%\!<%I$G$9!#(B*/
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
N$B8D$N%/%$!<%s2r$r7W;;$9$k4X?t$G$9!#2r$N(B2$BJ,$N(B1$B$r7W;;$7!"$=$N7k2L$rHWLL$N!V(BY$B<4!W$N>e$KH?E>$5$;$^$9!#(B $B$9$Y$F$N2r$O!"$=$N$h$&$KH?<M$7$FJL$N%f%K!<%/$J2r$r@8@.$9$k$3$H$,$G$-$^$9!J%\!<%I$N%5%$%:$,(B1$B!_(B1$B$G$J$$$H2>Dj$7$^$9!K!#(B $B$=$l$O!"2r$,(BY$B<4$r64$s$GBP>N$K$J$i$J$$$+$i$G$9!JF1$82#Ns$K(B2$B$D$N%/%$!<%s$rCV$/$3$H$O$G$-$J$$$+$i$G$9!K!#(B $B$^$?!"F1$8=DNs$K(B2$B$D$N%/%$!<%s$rCV$/$3$H$O$G$-$J$$$N$G!"4q?tNs$N%\!<%I$NCf1{Ns$r%/%$!<%s$G9=@.$9$k$3$H$b$G$-$J$$!#(B

This is a backtracking algorithm.  We place a queen in the top row,then note the column and diagonals it occupies.  We then place a queen in the next row down,taking care not to place it in the same column or diagonal.  We then update the occupied columns & diagonals & move on to the next row.  If no position is open in the next row,we back track to the previous row & move the queen over to the next available spot in its row & the process starts over again.
$B$3$l$O%P%C%/%H%i%C%/$N%"%k%4%j%:%`$G$"$k!#(B $B0lHV>e$N9T$K%/%$!<%s$rCV$-!"$=$N%/%$!<%s$,@j$a$kNs$HBP3Q@~$KCm0U$9$k!#(B $B<!$K!"F1$8Ns$dBP3Q@~$K%/%$!<%s$rG[CV$7$J$$$h$&$KCm0U$7$J$,$i!"<!$N9T$K%/%$!<%s$rG[CV$7$^$9!#(B $B$=$N8e!"@jM-$7$F$$$kNs$HBP3Q@~$r99?7$7!"<!$N9T$K?J$`!#(B $B<!$N9T$K6u$-$,$J$$>l9g$O!"A0$N9T$KLa$j!"%/%$!<%s$r$=$N9T$N<!$N6u$$$?>l=j$K0\F0$5$;!"$^$?F1$8$3$H$r7+$jJV$7$^$9!#(B
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
  /* $B%9%?%C%/$N=i4|2=(B */
  aStack[0]=-1;/* set sentinel -- signifies end of stack */
  /* NOTE:(size & 1) is true iff size is odd */
  /* $BCm0U!'(B(size & 1)$B$O!"(Bsize$B$,4q?t$G$"$l$P??$H$J$k(B */
  /* We need to loop through 2x if size is odd */
  /* size$B$,4q?t$N>l9g!"(B2x$B$r%k!<%W$9$kI,MW$,$"$k(B */
  for(int i=0;i <(1+odd);++i) {
    /* We don't have to optimize this part;it ain't the critical loop */
    /* $B$3$NItJ,$O:GE,2=$9$kI,MW$O$"$j$^$;$s(B;$B=EMW$J%k!<%W$G$O$"$j$^$;$s(B */
    bitmap=0;
    if(0==i){
      /* 
      Handle half of the board,except the middle column. So if the board is 5 x 5,the first row will be: 00011,since we're not worrying about placing a queen in the center column(yet).
      $B%\!<%I$N??$sCf$NNs$r=|$/H>J,$r=hM}$7$^$9!#$D$^$j!"HWLL$,(B5$B!_(B5$B$N>l9g!"(B1$BNsL\$O$3$&$J$j$^$9!'(B $BCf1{$NNs$K%/%$!<%s$rCV$/?4G[$O!J$^$@!K$J$$$N$G!"(B00011$B$H$J$j$^$9!#(B
      */
      int half=size>>1;/* divide by two */
      /* 
      fill in rightmost 1's in bitmap for half of size If size is 7,half of that is 3(we're discarding the remainder) and bitmap will be set to 111 in binary. 
      size$B$NH>J,$N(Bbitmap$B$N1&C<$N(B1$B$rKd$a$k(B size$B$,(B7$B$N>l9g!"$=$NH>J,$O(B3$B!J;D$j$O<N$F$k!K!"(Bbitmap$B$O(B2$B?J?t$G(B111$B$K@_Dj$5$l$^$9!#(B
      */
      bitmap=(1<<half)-1;
      pnStack=aStack+1;/* stack pointer */
      board[0]=0;
      down[0]=left[0]=right[0]=0;
    }else{
      /* 
      Handle the middle column(of a odd-sized board).  Set middle column bit to 1,then set half of next row.  So we're processing first row(one element) & half of next.  So if the board is 5 x 5,the first row will be: 00100,and the next row will be 00011.
      $B4q?t%5%$%:$N%\!<%I$NCfNs$r=hM}$9$k!#(B $BCfNs$N%S%C%H$r(B1$B$K%;%C%H$7!"<!$N9T$NH>J,$r%;%C%H$9$k!#(B $B$D$^$j!":G=i$N9T!J(B1$B$D$NMWAG!K$H<!$N9T$NH>J,$r=hM}$9$k$3$H$K$J$j$^$9!#(B $B$D$^$j!"HWLL$,(B5$B!_(B5$B$N>l9g!":G=i$N9T$O$3$&$J$j$^$9!'(B 00100$B$H$J$j!"<!$N9T$O(B00011$B$H$J$j$^$9!#(B
      */
      bitmap=1 <<(size>>1);
      row=1;/* prob. already 0 */
      /* The first row just has one queen(in the middle column).*/
      /* $B:G=i$N9T$K$O%/%$!<%s$,(B1$B?M!J??$sCf$NNs!K$@$1$G$9(B*/
      board[0]=bitmap;
      down[0]=left[0]=right[0]=0;
      down[1]=bitmap;
      /* 
      Now do the next row.  Only set bits in half of it,because we'll flip the results over the "Y-axis".
      $B<!$N9T$r<B9T$7$^$9!#(B Y$B<4$N7k2L$rH?E>$5$;$k$N$G!"H>J,$@$1%S%C%H$r@_Dj$7$^$9!#(B 
      */
      right[1]=(bitmap>>1);
      left[1]=(bitmap<<1);
      pnStack=aStack+1;/* stack pointer */
      *pnStack++=0;/* we're done w/ this row -- only 1 element & we've done it */
      bitmap=(bitmap-1)>>1;/* bitmap -1 is all 1's to the left of the single 1 */
    }
    /* this is the critical loop */
    /* $B$3$l$,%/%j%F%#%+%k%k!<%W$G$9(B */
    for(;;){
      /* 
      could use bit=bitmap ^(bitmap &(bitmap -1));
      to get first(least sig) "1" bit,but that's slower. 
      bit=bitmap ^(bitmap &(bitmap -1))$B$r;H$&$3$H$,$G$-$^$9!#(B
      $B:G=i$N(B($B:G>.?.9f$N(B)$B!V(B1$B!W%S%C%H$rF@$k$3$H$,$G$-$^$9$,!"$3$l$O$h$jCY$$$G$9!#(B
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
      /* $B:F;n9T$7$J$$$h$&$K!"$3$N%S%C%H$r%H%0%k%*%U$K$9$k(B */
      bitmap&=~bit;
      board[row]=bit;/* save the result */
      /* we still have more rows to process? */
      /* $B$^$@=hM}$9$k9T$,$"$k$N$G$O!)(B*/
      if(row<sizeE){
        row++;
        down[row]=down[row-1]|bit;
        right[row]=(right[row-1]|bit)>>1;
        left[row]=(left[row-1]|bit)<<1;
        *pnStack++=bitmap;
        /* 
        We can't consider positions for the queen which are in the same column,same positive diagonal,or same negative diagonal as another queen already on the board. 
        $B$9$G$K%\!<%I>e$K$"$kB>$N%/%$!<%s$HF1$8Ns!"F1$8@5BP3Q!"F1$8IiBP3Q$K$"$k%/%$!<%s$N%]%8%7%g%s$O9MN8$7$J$$!#(B
        */
        bitmap=mask&~(down[row]|right[row]|left[row]);
        continue;
      }else{
        /* 
        We have no more rows to process;we found a solution. Comment out the call to printtable in order to print the solutions as board position printtable(size,board,g_numsolutions+1); 
        $B$b$&=hM}$9$k9T$O$J$$!#2r7h:v$r8+$D$1$?!#2r$rHWLL0LCV$H$7$FI=<($9$k$?$a$K!"(Bprinttable$B$N8F$S=P$7$r%3%a%s%H%"%&%H$9$k(B printtable(size,board,g_numsolutions+1)$B!((B 
        */
        ++g_numsolutions;
        bitmap=*--pnStack;
        --row;
        continue;
      }
    }
  }
  /* multiply solutions by two,to count mirror images */
  /* $B6@A|$N?t$r?t$($k$?$a$K!"2r$r(B2$BG\$9$k(B */
  g_numsolutions *= 2;
}
/* Print the results at the end of the run */
/* $B<B9T=*N;;~$K7k2L$rI=<($7$^$9(B */
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
/* N Queens$B%W%m%0%i%`$N%a%$%s%k!<%A%s(B */
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
