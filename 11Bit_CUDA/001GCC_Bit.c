/**
bash-5.1$ gcc -W -Wall -O3 01CUDA_Bit_Symmetry.c && ./a.out
ビット
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2           1      00:00:00:00.00
 5:           10           2      00:00:00:00.00
 6:            4           1      00:00:00:00.00
 7:           40           6      00:00:00:00.00
 8:           92          12      00:00:00:00.00
 9:          352          46      00:00:00:00.00
10:          724          92      00:00:00:00.00
11:         2680         341      00:00:00:00.00
12:        14200        1787      00:00:00:00.00
13:        73712        9233      00:00:00:00.00
14:       365596       45752      00:00:00:00.05
15:      2279184      285053      00:00:00:00.33
16:     14772512     1846955      00:00:00:02.16
17:     95815104    11977939      00:00:00:14.89
18:    666090624    83263591      00:00:01:45.44
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define MAX 27
#define THREAD_NUM 96
// システムによって以下のマクロが必要であればコメントを外してください
//#define UINT64_C(c) c ## ULL
ulong TOTAL=0;
ulong UNIQUE=0;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef struct local
{
  uint BOUND1,BOUND2;
  uint TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
  ulong board[MAX];
  ulong COUNT2,COUNT4,COUNT8,TOTAL,UNIQUE;
}local;
/**
  CPU 再帰/非再帰共通 対称解除法
*/
void symmetryOps(uint size,struct local* l)
{
  uint ptn,own,bit,you;
  //９０度回転
  if(l->board[l->BOUND2]==1){
    for(ptn=2,own=1;own<size;++own,ptn<<=1){
      for(bit=1,you=size-1;(l->board[you]!=ptn)&&l->board[own]>=bit;--you){ bit<<=1; }
      if(l->board[own]>bit){ return ; }
      if(l->board[own]<bit){ break; }
    }
    if(own>size-1){ l->COUNT2++; return ; }
  }
  //１８０度回転
  if(l->board[size-1]==l->ENDBIT){
    for(you=size-1-1,own=1;own<=size-1;++own,--you){
      for(bit=1,ptn=l->TOPBIT;(ptn!=l->board[you])&&(l->board[own]>=bit);ptn>>=1){ bit<<=1; }
      if(l->board[own]>bit){ return ; }
      if(l->board[own]<bit){ break; }
    }
    //９０度回転が同型でなくても１８０度回転が同型であることもある
    if(own>size-1){ l->COUNT4++; return ; }
  }
  //２７０度回転
  if(l->board[l->BOUND1]==l->TOPBIT){
    for(ptn=l->TOPBIT>>1,own=1;own<=size-1;++own,ptn>>=1){
      for(bit=1,you=0;(l->board[you]!=ptn)&&(l->board[own]>=bit);++you){ bit<<=1; }
      if(l->board[own]>bit){ return ; }
      if(l->board[own]<bit){ break; }
    }
  }
  l->COUNT8++;
}
/**
  再帰 角にQがないときのバックトラック
*/
void backTrack(uint size,uint row,uint left,uint down,uint right,struct local* l)
{
  uint bit;
  uint mask=(1<<size)-1;
  uint bitmap=mask&~(left|down|right);
  if(row==(size-1)){
    if(bitmap){
      if( (bitmap&l->LASTMASK)==0){
        l->board[row]=bitmap;
        symmetryOps(size,l);
      }
    }
  }else{
    if(row<l->BOUND1){
      bitmap=bitmap|l->SIDEMASK;
      bitmap=bitmap^l->SIDEMASK;
    }else{
      if(row==l->BOUND2){
        if((down&l->SIDEMASK)==0){ return; }
        if( (down&l->SIDEMASK)!=l->SIDEMASK){
          bitmap=bitmap&l->SIDEMASK;
        }
      }
    }
    while(bitmap){
      bit=-bitmap&bitmap;
      bitmap=bitmap^bit;
      l->board[row]=bit;
      backTrack(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
}
/**
  再帰 角にQがあるときのバックトラック
*/
void backTrack_corner(uint size,uint row,uint left,uint down,uint right,struct local* l)
{
  uint mask=(1<<size)-1;
  uint bitmap=mask&~(left|down|right);
  uint bit=0;
  if(row==(size-1)){
    if(bitmap){
      l->board[row]=bitmap;
      l->COUNT8++;
    }
  }else{
    if(row<l->BOUND1){   //枝刈り
      bitmap=bitmap|2;
      bitmap=bitmap^2;
    }
    while(bitmap){
      bit=-bitmap&bitmap;
      bitmap=bitmap^bit;
      l->board[row]=bit;   //Qを配置
      backTrack_corner(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
}
/**
  再帰 対称解除法
*/
void nqueens(uint size,struct local* l)
{
  uint bit=0;
  l->TOTAL=l->UNIQUE=l->COUNT2=l->COUNT4=l->COUNT8=0;
  l->TOPBIT=1<<(size-1);
  l->ENDBIT=l->LASTMASK=l->SIDEMASK=0;
  l->BOUND1=2;
  l->BOUND2=0;
  l->board[0]=1;
  while(l->BOUND1>1 && l->BOUND1<size-1){
    if(l->BOUND1<size-1){
      bit=1<<l->BOUND1;
      l->board[1]=bit;
      backTrack_corner(size,2,(2|bit)<<1,1|bit,(2|bit)>>1,l);
    }
    l->BOUND1++;
  }
  l->TOPBIT=1<<(size-1);
  l->ENDBIT=l->TOPBIT>>1;
  l->SIDEMASK=l->LASTMASK=l->TOPBIT|1;
  l->BOUND1=1;
  l->BOUND2=size-2;
  while(l->BOUND1>0 && l->BOUND2<size-1 && l->BOUND1<l->BOUND2){
    if(l->BOUND1<l->BOUND2){
      bit=1<<l->BOUND1;
      l->board[0]=bit;
      backTrack(size,1,bit<<1,bit,bit>>1,l);
    }
    l->BOUND1++;
    l->BOUND2--;
    l->ENDBIT=l->ENDBIT>>1;
    l->LASTMASK=l->LASTMASK<<1|l->LASTMASK|l->LASTMASK>>1;
  }
  UNIQUE=l->COUNT2+l->COUNT4+l->COUNT8;
  TOTAL=l->COUNT2*2+l->COUNT4*4+l->COUNT8*8;
}
/**
 * メイン
 */
int main(int argc,char** argv)
{
  uint min=4;
  uint targetN=18;
  struct timeval t0;
  struct timeval t1;
  printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
  uint ss,ms,dd,hh,mm;
	printf("%s\n","ビット");
  for(uint size=min;size<=targetN;size++){
    local l;
    gettimeofday(&t0,NULL);
    nqueens(size,&l);
    gettimeofday(&t1,NULL);
    if(t1.tv_usec<t0.tv_usec) {
      dd=(t1.tv_sec-t0.tv_sec-1)/86400;
      ss=(t1.tv_sec-t0.tv_sec-1)%86400;
      ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
    }else {
      dd=(t1.tv_sec-t0.tv_sec)/86400;
      ss=(t1.tv_sec-t0.tv_sec)%86400;
      ms=(t1.tv_usec-t0.tv_usec+500)/10000;
    }
    hh=ss/3600;
    mm=(ss-hh*3600)/60;
    ss%=60;
    printf("%2d:%13ld%12ld%8.2d:%02d:%02d:%02d.%02d\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
  }
  return 0;
}
