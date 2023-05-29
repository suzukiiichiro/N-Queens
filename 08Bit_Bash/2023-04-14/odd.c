#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#define THREAD_NUM		96
#define MAX 27
//$BJQ?t@k8@(B
int down[2*MAX-1];  //CPU down:flagA $B=D(B $BG[CV%U%i%0!!(B
int left[2*MAX-1];  //CPU left:flagB $B<P$aG[CV%U%i%0!!(B
int right[2*MAX-1]; //CPU right:flagC $B<P$aG[CV%U%i%0!!(B
int board[MAX];
int TOTAL;
int COUNT2;
//$BHs:F5"HG%_%i!<%m%8%C%/(B
void solve_nqueen(int size,int mask, int row,int h_left,int h_down,int h_right)
{
  int bitmap[size];
	left[row]=h_left;
	down[row]=h_down;
	right[row]=h_right;
  int bit;
  //$B8GDj$7$F$$$l$?9T$h$j>e$O$$$+$J$$(B
	bitmap[row]=mask&~(left[row]|down[row]|right[row]);
  while(row>0){//row=1 row>=1, row=2 row>=2
    if(bitmap[row]==0){
      --row;
    }else{
      //bitmap[row]^=board[row]=bit=(-bitmap[row]&bitmap[row]); 
      bit=(-bitmap[row]&bitmap[row]); 
      bitmap[row]=bitmap[row]^bit;
      board[row]=bit;
      if((bit&mask)!=0){
        if(row==(size-1)){
          COUNT2++;
          --row;
        }else{
          int n=row++;
          left[row]=(left[n]|bit)<<1;
          down[row]=down[n]|bit;
          right[row]=(right[n]|bit)>>1;
          bitmap[row]=mask&~(left[row]|down[row]|right[row]);
        }
      }else{
         --row;
      }
    }  
  }
}
//$BHs:F5"HG%_%i!<(B
void mirror_NR(int size,int mask)
{
  int bit=0;
  /*
   $B6v?t!"4q?t$H$b$K1&H>J,$K%/%$!<%s$rCV$/(B 
   00001111
  */
  /*
   $B4q?t$N>l9g(B
   $B4q?t$OCf1{$K$b%/%$!<%s$rCV$/(B
   00100
   $B#19TL\$N:8B&H>J,$K%/%$!<%s$rCV$1$J$$$h$&$K$9$k(B
   11100

   1$B9TL\$K%/%$!<%s$,Cf1{$KCV$+$l$?>l9g$O(B
   00100
   2$B9TL\$N:8B&H>J,$K%/%$!<%s$rCV$1$J$$(B
   00100
   11100
   $B$5$i$K(B1$B9TL\$N(Bdown,right$B$b%/%$!<%s$rCV$1$J$$$N$G(B(size/2)-1$B$H$J$k(B
   11100

   $B6v?t$N>l9g(B
   $B#19TL\$N:8B&H>J,$K%/%$!<%s$rCV$1$J$$$h$&$K$9$k(B
   1100
  */
  int limit=size%2 ? size/2-1 : size/2;
  for(int i=0;i<size/2;i++){
    bit=(1<<i);         
    board[0]=bit;       //1$B9TL\$K%/%$!<%s$rCV$/(B
    solve_nqueen(size,mask,1,bit<<1,bit,bit>>1);
  }
  if(size%2){
    //1$B9TL\$O%/%$!<%s$rCf1{$KCV$/(B
    bit=(1<<( (size-1)/2));
    board[0]=(1<<((size-1)/2) );
    int left=bit<<1;
    int down=bit;
    int right=bit>>1;

    for(int i=0;i<limit;i++){
      bit=(1<<i);         
      board[1]=bit;       //2$B9TL\$K%/%$!<%s$rCV$/(B
      solve_nqueen(size,mask,2,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
  TOTAL=COUNT2<<1;        // $BG\$K$9$k(B
}

//$B:F5"HG%_%i!<(B $B%m%8%C%/(B
void solve_nqueenr(int size,int mask, int row,int left,int down,int right)
{
 int bit=0;
 int bitmap=(mask&~(left|down|right)); //$B%/%$!<%s$,G[CV2DG=$J0LCV$rI=$9(B
 if(row==(size-1)){
   if(bitmap){
     COUNT2++;
   }
  }else{
    while(bitmap){
      bit=(-bitmap&bitmap); // $B0lHV1&$N%S%C%H$r<h$j=P$9(B
      bitmap=bitmap^bit;    //$BG[CV2DG=$J%Q%?!<%s$,0l$D$:$D<h$j=P$5$l$k(B
      board[row]=bit;       //Q$B$rG[CV(B
      solve_nqueenr(size,mask,row+1,(left|bit)<<1, down|bit,(right|bit)>>1);
    }
  }
}
//$B:F5"HG%_%i!<(B
void mirror_R(int size,int mask)
{
  int bit=0;
  /*
   $B6v?t!"4q?t$H$b$K1&H>J,$K%/%$!<%s$rCV$/(B 
   00001111
  */
  /*
   $B4q?t$N>l9g(B
   $B4q?t$OCf1{$K$b%/%$!<%s$rCV$/(B
   00100
   $B#19TL\$N:8B&H>J,$K%/%$!<%s$rCV$1$J$$$h$&$K$9$k(B
   11100

   1$B9TL\$K%/%$!<%s$,Cf1{$KCV$+$l$?>l9g$O(B
   00100
   2$B9TL\$N:8B&H>J,$K%/%$!<%s$rCV$1$J$$(B
   00100
   11100
   $B$5$i$K(B1$B9TL\$N(Bdown,right$B$b%/%$!<%s$rCV$1$J$$$N$G(B(size/2)-1$B$H$J$k(B
   11100

   $B6v?t$N>l9g(B
   $B#19TL\$N:8B&H>J,$K%/%$!<%s$rCV$1$J$$$h$&$K$9$k(B
   1100
  */
  int limit=size%2 ? size/2-1 : size/2;
  for(int i=0;i<size/2;i++){
    bit=(1<<i);         
    board[0]=bit;       //1$B9TL\$K%/%$!<%s$rCV$/(B
    solve_nqueenr(size,mask,1,bit<<1,bit,bit>>1);
  }
  if(size%2){
    //1$B9TL\$O%/%$!<%s$rCf1{$KCV$/(B
    bit=(1<<( (size-1)/2));
    board[0]=(1<<((size-1)/2) );
    int left=bit<<1;
    int down=bit;
    int right=bit>>1;

    for(int i=0;i<limit;i++){
      bit=(1<<i);         
      board[1]=bit;       //2$B9TL\$K%/%$!<%s$rCV$/(B
      solve_nqueenr(size,mask,2,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
  TOTAL=COUNT2<<1;        // $BG\$K$9$k(B
}
//
//$B%a%$%s%a%=%C%I(B
int main(int argc,char** argv)
{
  COUNT2=0;                 //$B%0%m!<%P%k(B
  int size=5;
  int mask=(1<<size)-1;
//  mirror_R(size,mask);      //$B:F5"HG%_%i!<(B
   mirror_NR(size,mask);  //$BHs:F5"HG%_%i!<(B
  printf("%2d:%13d\n",size,TOTAL);
  return 0;
}

