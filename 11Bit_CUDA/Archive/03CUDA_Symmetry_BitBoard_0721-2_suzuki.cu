/**
 *
 * bash版対称解除法のC言語版のGPU/CUDA移植版
 *
 詳しい説明はこちらをどうぞ
 https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

非再帰でのコンパイルと実行
$ nvcc -O3 -arch=sm_61 03CUDA_Symmetry_BitBoard.cu && ./a.out -c

再帰でのコンパイルと実行
$ nvcc -O3 -arch=sm_61 03CUDA_Symmetry_BitBoard.cu && ./a.out -r

GPU で並列処理せずに実行
$ nvcc -O3 -arch=sm_61 03CUDA_Symmetry_BitBoard.cu && ./a.out -n

GPU で並列処理で実行（ビットボード）
$ nvcc -O3 -arch=sm_61 03CUDA_Symmetry_BitBoard.cu && ./a.out -n



 *
*/
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define MAX 27
#define THREAD_NUM 96
using std::cout; using std::endl;
using std::vector; using std::string;
// システムによって以下のマクロが必要であればコメントを外してください。
//#define UINT64_C(c) c ## ULL
//
// グローバル変数
unsigned long TOTAL=0;
unsigned long UNIQUE=0;
//GPU で使うローカル構造体
typedef struct local
{
  unsigned int BOUND1,BOUND2;
  unsigned int TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
  unsigned long board[MAX];
  unsigned long COUNT2,COUNT4,COUNT8,TOTAL,UNIQUE;
  unsigned long TYPE;
}local;
// CPU 再帰/非再帰共通 対称解除法
void symmetryOps(unsigned int size,struct local* l)
{
  /**
  ２．クイーンが右上角以外にある場合、
  (1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルか
  ら180度回転)させても、さらに90度回転(オリジナルから270度回転)させてもオリ
  ジナルと同型になる。
  こちらに該当するユニーク解が属するグループの要素数は、左右反転させたパター
  ンを加えて２個しかありません。
  */
  if(l->board[l->BOUND2]==1){
    unsigned int ptn;
    unsigned int own;
    for(ptn=2,own=1;own<size;++own,ptn<<=1){
      unsigned int bit;
      unsigned int you;
      for(bit=1,you=size-1;(l->board[you]!=ptn)&&l->board[own]>=bit;--you){
        bit<<=1;
      }
      if(l->board[own]>bit){
        return ;
      }
      if(l->board[own]<bit){
        break;
      }
    }//end for
    // ９０度回転して同型なら１８０度回転しても２７０度回転しても同型である
    if(own>size-1){
      l->COUNT2++;
      return ;
    }//end if
  }//end if
  /**
  ２．クイーンが右上角以外にある場合、
    (2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナル
    とは異なる。ただし、180度回転させた場合はオリジナルと同型になることも有り得
    る。こちらに該当するユニーク解が属するグループの要素数は、180度回転させて同
    型になる場合は４個(左右反転×縦横回転)
   */
  //１８０度回転
  if(l->board[size-1]==l->ENDBIT){
    unsigned int you;
    unsigned int own;
    for(you=size-1-1,own=1;own<=size-1;++own,--you){
      unsigned int bit;
      unsigned int ptn;
      for(bit=1,ptn=l->TOPBIT;(ptn!=l->board[you])&&(l->board[own]>=bit);ptn>>=1){
        bit<<=1;
      }
      if(l->board[own]>bit){
        return ;
      }
      if(l->board[own]<bit){
        break;
      }
    }//end for
    //９０度回転が同型でなくても１８０度回転が同型であることもある
    if(own>size-1){
      l->COUNT4++;
      return ;
    }
  }//end if
  /**
  ２．クイーンが右上角以外にある場合、
    (3)180度回転させてもオリジナルと異なる場合は、８個(左右反転×縦横回転×上下反転)
  */
  //２７０度回転
  if(l->board[l->BOUND1]==l->TOPBIT){
    unsigned int ptn;
    unsigned int own;
    unsigned int you;
    unsigned int bit;
    for(ptn=l->TOPBIT>>1,own=1;own<=size-1;++own,ptn>>=1){
      for(bit=1,you=0;(l->board[you]!=ptn)&&(l->board[own]>=bit);++you){
        bit<<=1;
      }
      if(l->board[own]>bit){
        return ;
      }
      if(l->board[own]<bit){
        break;
      }
    }//end for
  }//end if
  l->COUNT8++;
}
/**
  CPU -c
  */
// 非再帰 角にQがないときのバックトラック
void symmetry_backTrack_NR(unsigned int size,unsigned int row,unsigned int _left,unsigned int _down,unsigned int _right,struct local *l)
{
  unsigned int mask=(1<<size)-1;
  unsigned int down[size];
  unsigned int left[size];
  unsigned int right[size];
  unsigned int bitmap[size];
  left[row]=_left;
  down[row]=_down;
  right[row]=_right;
  bitmap[row]=mask&~(left[row]|down[row]|right[row]);
  while(row>0){
    if(bitmap[row]>0){
      if(row<l->BOUND1){ //上部サイド枝刈り
        bitmap[row]|=l->SIDEMASK;
        bitmap[row]^=l->SIDEMASK;
      }else if(row==l->BOUND2){ //下部サイド枝刈り
        if((down[row]&l->SIDEMASK)==0){
          row--; 
        }
        if((down[row]&l->SIDEMASK)!=l->SIDEMASK){
          bitmap[row]&=l->SIDEMASK;
        }
      }
      unsigned int save_bitmap=bitmap[row];
      unsigned int bit=-bitmap[row]&bitmap[row];
      bitmap[row]^=bit;
      l->board[row]=bit; //Qを配置
      if((bit&mask)!=0){
        if(row==(size-1)){
          if( (save_bitmap&l->LASTMASK)==0){
            symmetryOps(size,l);  //対称解除法
          }
          row--;
        }else{
          unsigned int n=row++;
          left[row]=(left[n]|bit)<<1;
          down[row]=(down[n]|bit);
          right[row]=(right[n]|bit)>>1;
          bitmap[row]=mask&~(left[row]|down[row]|right[row]);
        }
      }else{
        row--;
      }
    }else{
      row--;
    }
  }//end while
}
// 非再帰 角にQがあるときのバックトラック
void symmetry_backTrack_corner_NR(unsigned int size,unsigned int row,unsigned int _left,unsigned int _down,unsigned int _right,struct local *l)
{
  unsigned int mask=(1<<size)-1;
  unsigned int bit=0;
  unsigned int down[size];
  unsigned int left[size];
  unsigned int right[size];
  unsigned int bitmap[size];
  left[row]=_left;
  down[row]=_down;
  right[row]=_right;
  bitmap[row]=mask&~(left[row]|down[row]|right[row]);
  while(row>=2){
    if(row<l->BOUND1){
      // bitmap[row]=bitmap[row]|2;
      // bitmap[row]=bitmap[row]^2;
      bitmap[row]&=~2;
    }
    if(bitmap[row]>0){
      bit=-bitmap[row]&bitmap[row];
      bitmap[row]^=bit;
      if(row==(size-1)){
        l->COUNT8++;
        row--;
      }else{
        unsigned int n=row++;
        left[row]=(left[n]|bit)<<1;
        down[row]=(down[n]|bit);
        right[row]=(right[n]|bit)>>1;
        l->board[row]=bit; //Qを配置
        //クイーンが配置可能な位置を表す
        bitmap[row]=mask&~(left[row]|down[row]|right[row]);
      }
    }else{
      row--;
    }
  }//end while
}
// 非再帰 対称解除法
void symmetry_NR(unsigned int size,struct local* l)
{
  l->TOTAL=l->UNIQUE=l->COUNT2=l->COUNT4=l->COUNT8=0;
  unsigned int bit=0;
  l->TOPBIT=1<<(size-1);
  l->ENDBIT=l->SIDEMASK=l->LASTMASK=0;
  l->BOUND1=2;
  l->BOUND2=0;
  l->board[0]=1;
  while(l->BOUND1>1&&l->BOUND1<size-1){
    if(l->BOUND1<size-1){
      bit=1<<l->BOUND1;
      l->board[1]=bit;   //２行目にQを配置
      //角にQがあるときのバックトラック
      symmetry_backTrack_corner_NR(size,2,(2|bit)<<1,1|bit,(2|bit)>>1,l);
    }
    l->BOUND1++;
  }
  l->TOPBIT=1<<(size-1);
  l->ENDBIT=l->TOPBIT>>1;
  l->SIDEMASK=l->TOPBIT|1;
  l->LASTMASK=l->TOPBIT|1;
  l->BOUND1=1;
  l->BOUND2=size-2;
  while(l->BOUND1>0 && l->BOUND2<size-1 && l->BOUND1<l->BOUND2){
    if(l->BOUND1<l->BOUND2){
      bit=1<<l->BOUND1;
      l->board[0]=bit;   //Qを配置
      //角にQがないときのバックトラック
      symmetry_backTrack_NR(size,1,bit<<1,bit,bit>>1,l);
    }
    l->BOUND1++;
    l->BOUND2--;
    l->ENDBIT=l->ENDBIT>>1;
    l->LASTMASK=l->LASTMASK<<1|l->LASTMASK|l->LASTMASK>>1;
  }//ene while
  UNIQUE=l->COUNT2+l->COUNT4+l->COUNT8;
  TOTAL=l->COUNT2*2+l->COUNT4*4+l->COUNT8*8;
}
/**
  CPU -r
  */
// 再帰 角にQがないときのバックトラック
void symmetry_backTrack(unsigned int size,unsigned int row,unsigned int left,unsigned int down,unsigned int right,struct local* l)
{
  unsigned int mask=(1<<size)-1;
  unsigned int bitmap=mask&~(left|down|right);
  if(row==(size-1)){
    if(bitmap){
      if( (bitmap&l->LASTMASK)==0){
        l->board[row]=bitmap;  //Qを配置
        symmetryOps(size,l);    //対称解除
      }
    }
  }else{
    if(row<l->BOUND1){
      bitmap=bitmap|l->SIDEMASK;
      bitmap=bitmap^l->SIDEMASK;
    }else{
      if(row==l->BOUND2){
        if((down&l->SIDEMASK)==0){
          return;
        }
        if( (down&l->SIDEMASK)!=l->SIDEMASK){
          bitmap=bitmap&l->SIDEMASK;
        }
      }
    }
    while(bitmap){
      unsigned int bit=-bitmap&bitmap;
      bitmap=bitmap^bit;
      l->board[row]=bit;
      symmetry_backTrack(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
}
// 再帰 角にQがあるときのバックトラック
void symmetry_backTrack_corner(unsigned int size,unsigned int row,unsigned int left,unsigned int down,unsigned int right,struct local* l)
{
  unsigned int mask=(1<<size)-1;
  unsigned int bitmap=mask&~(left|down|right);
  unsigned int bit=0;
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
      symmetry_backTrack_corner(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
}
// 再帰 対称解除法
void symmetry_R(unsigned int size,struct local* l)
{
  l->TOTAL=l->UNIQUE=l->COUNT2=l->COUNT4=l->COUNT8=0;
  unsigned int bit=0;
  l->TOPBIT=1<<(size-1);
  l->ENDBIT=l->LASTMASK=l->SIDEMASK=0;
  l->BOUND1=2;
  l->BOUND2=0;
  l->board[0]=1;
  while(l->BOUND1>1 && l->BOUND1<size-1){
    if(l->BOUND1<size-1){
      bit=1<<l->BOUND1;
      l->board[1]=bit;   //２行目にQを配置
      //角にQがあるときのバックトラック
      symmetry_backTrack_corner(size,2,(2|bit)<<1,1|bit,(2|bit)>>1,l);
    }
    l->BOUND1++;
  }//end while
  l->TOPBIT=1<<(size-1);
  l->ENDBIT=l->TOPBIT>>1;
  l->SIDEMASK=l->TOPBIT|1;
  l->LASTMASK=l->TOPBIT|1;
  l->BOUND1=1;
  l->BOUND2=size-2;
  while(l->BOUND1>0 && l->BOUND2<size-1 && l->BOUND1<l->BOUND2){
    if(l->BOUND1<l->BOUND2){
      bit=1<<l->BOUND1;
      l->board[0]=bit;   //Qを配置
      //角にQがないときのバックトラック
      symmetry_backTrack(size,1,bit<<1,bit,bit>>1,l);
    }
    l->BOUND1++;
    l->BOUND2--;
    l->ENDBIT=l->ENDBIT>>1;
    l->LASTMASK=l->LASTMASK<<1|l->LASTMASK|l->LASTMASK>>1;
  }//ene while
  UNIQUE=l->COUNT2+l->COUNT4+l->COUNT8;
  TOTAL=l->COUNT2*2+l->COUNT4*4+l->COUNT8*8;
}
/**
  GPU -g
  */
__device__
struct dlocal
{
  unsigned int BOUND1,BOUND2;
  unsigned int TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
  unsigned long board[MAX];
  unsigned long COUNT2,COUNT4,COUNT8,TOTAL,UNIQUE;
  unsigned long TYPE;
}dlocal;
__device__ struct dlocal gdl[9999];
// GPU 対称解除法
__host__ __device__
long GPU_symmetryOps(unsigned int size,struct dlocal* l)
{
  /**
  ２．クイーンが右上角以外にある場合、
  (1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルか
  ら180度回転)させても、さらに90度回転(オリジナルから270度回転)させてもオリ
  ジナルと同型になる。
  こちらに該当するユニーク解が属するグループの要素数は、左右反転させたパター
  ンを加えて２個しかありません。
  */
  if(l->board[l->BOUND2]==1){
    unsigned int ptn;
    unsigned int own;
    for(ptn=2,own=1;own<size;++own,ptn<<=1){
      unsigned int bit;
      unsigned int you;
      for(bit=1,you=size-1;(l->board[you]!=ptn)&& l->board[own]>=bit;--you){
        bit<<=1;
      }
      if(l->board[own]>bit){
        return 0;
      }
      if(l->board[own]<bit){
        break;
      }
    }//end for
    // ９０度回転して同型なら１８０度回転しても２７０度回転しても同型である
    if(own>size-1){
      l->COUNT2++;
      return 2;
    }//end if
  }//end if
  /**
  ２．クイーンが右上角以外にある場合、
    (2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナル
    とは異なる。ただし、180度回転させた場合はオリジナルと同型になることも有り得
    る。こちらに該当するユニーク解が属するグループの要素数は、180度回転させて同
    型になる場合は４個(左右反転×縦横回転)
   */
  //１８０度回転
  if(l->board[size-1]==l->ENDBIT){
    unsigned int you;
    unsigned int own;
    for(you=size-1-1,own=1;own<=size-1;++own,--you){
      unsigned int bit;
      unsigned int ptn;
      for(bit=1,ptn=l->TOPBIT;(ptn!=l->board[you])&&(l->board[own]>=bit);ptn>>=1){
        bit<<=1;
      }
      if(l->board[own]>bit){
        return 0;
      }
      if(l->board[own]<bit){
        break;
      }
    }//end for
    //９０度回転が同型でなくても１８０度回転が同型であることもある
    if(own>size-1){
      l->COUNT4++;
      return 4;
    }
  }//end if
  /**
  ２．クイーンが右上角以外にある場合、
    (3)180度回転させてもオリジナルと異なる場合は、８個(左右反転×縦横回転×上下反転)
  */
  //２７０度回転
  if(l->board[l->BOUND1]==l->TOPBIT){
    unsigned int ptn;
    unsigned int own;
    unsigned int you;
    unsigned int bit;
    for(ptn=l->TOPBIT>>1,own=1;own<=size-1;++own,ptn>>=1){
      for(bit=1,you=0;(l->board[you]!=ptn)&&(l->board[own]>=bit);++you){
        bit<<=1;
      }
      if(l->board[own]>bit){
        return 0;
      }
      if(l->board[own]<bit){
        break;
      }
    }//end for
  }//end if
  l->COUNT8++;
  return 8;
}
// GPU 角にQがないときのバックトラック
__host__ __device__
long GPU_symmetry_backTrack(unsigned int size,unsigned int row,unsigned int left,unsigned int down,unsigned int right,struct dlocal* l)
{
  unsigned long counter=0;
  unsigned int mask=(1<<size)-1;
  unsigned int bitmap=mask&~(left|down|right);
  if(row==(size-1)){
    if(bitmap){
      if( (bitmap& l->LASTMASK)==0){
        l->board[row]=bitmap;  //Qを配置
        counter+=GPU_symmetryOps(size,l);    //対称解除
      }
    }
  }else{
    if(row<l->BOUND1){
      bitmap=bitmap|l->SIDEMASK;
      bitmap=bitmap^l->SIDEMASK;
    }else{
      if(row==l->BOUND2){
        if((down&l->SIDEMASK)==0){
          return 0;
        }
        if( (down&l->SIDEMASK)!=l->SIDEMASK){
          bitmap=bitmap&l->SIDEMASK;
        }
      }
    }
    while(bitmap){
      unsigned int bit=-bitmap&bitmap;
      bitmap=bitmap^bit;
      l->board[row]=bit;
      counter+=GPU_symmetry_backTrack(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
  return counter;
}
// GPU 角にQがあるときのバックトラック
__host__ __device__
long GPU_symmetry_backTrack_corner(unsigned int size,unsigned int row,unsigned int left,unsigned int down,unsigned int right,struct dlocal* l)
{
  unsigned long counter=0;
  unsigned int mask=(1<<size)-1;
  unsigned int bitmap=mask&~(left|down|right);
  unsigned int bit=0;
  if(row==(size-1)){
    if(bitmap){
      l->board[row]=bitmap;
      l->COUNT8++;
      counter+=8;
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
      counter+=GPU_symmetry_backTrack_corner(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
  return counter;
}
// GPU 対称解除法 -g の実行時のみ呼び出されます
__host__ __device__
void GPU_symmetry_R(unsigned int size,struct local* hostLocal)
{
  // GPU内部で使うための dlocal構造体
  struct dlocal l;
  l.TOTAL=l.UNIQUE=l.COUNT2=l.COUNT4=l.COUNT8=0;
  unsigned int bit=0;
  l.TOPBIT=1<<(size-1);
  l.ENDBIT=l.LASTMASK=l.SIDEMASK=0;
  l.BOUND1=2;
  l.BOUND2=0;
  l.board[0]=1;
  while(l.BOUND1>1 && l.BOUND1<size-1){
    if(l.BOUND1<size-1){
      bit=1<<l.BOUND1;
      l.board[1]=bit;   //２行目にQを配置
      //角にQがあるときのバックトラック
      GPU_symmetry_backTrack_corner(size,2,(2|bit)<<1,1|bit,(2|bit)>>1,&l);
    }
    l.BOUND1++;
  }//end while
  l.TOPBIT=1<<(size-1);
  l.ENDBIT=l.TOPBIT>>1;
  l.SIDEMASK=l.TOPBIT|1;
  l.LASTMASK=l.TOPBIT|1;
  l.BOUND1=1;
  l.BOUND2=size-2;
  while(l.BOUND1>0 && l.BOUND2<size-1 && l.BOUND1<l.BOUND2){
    if(l.BOUND1<l.BOUND2){
      bit=1<<l.BOUND1;
      l.board[0]=bit;   //Qを配置
      //角にQがないときのバックトラック
      GPU_symmetry_backTrack(size,1,bit<<1,bit,bit>>1,&l);
    }
    l.BOUND1++;
    l.BOUND2--;
    l.ENDBIT=l.ENDBIT>>1;
    l.LASTMASK=l.LASTMASK<<1|l.LASTMASK|l.LASTMASK>>1;
  }//ene while
  // 集計値は hostLocalへ代入
  hostLocal->UNIQUE=l.COUNT2+l.COUNT4+l.COUNT8;
  hostLocal->TOTAL=l.COUNT2*2+l.COUNT4*4+l.COUNT8*8;
}
/**
  CUDA13
  */
// GPU -n 対称解除法
__device__ 
int BitBoard_symmetryOps(int si,unsigned int *d_aBoard,int BOUND1,int BOUND2,int TOPBIT,int ENDBIT)
{
  int own,ptn,you,bit;
  //90度回転
  if(d_aBoard[BOUND2]==1){ own=1; ptn=2;
    while(own<=si-1){ bit=1; you=si-1;
      while((d_aBoard[you]!=ptn)&&(d_aBoard[own]>=bit)){ bit<<=1; you--; }
      if(d_aBoard[own]>bit){ return 0; } else if(d_aBoard[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>si-1){ return 2; }
  }
  //180度回転
  if(d_aBoard[si-1]==ENDBIT){ own=1; you=si-1-1;
    while(own<=si-1){ bit=1; ptn=TOPBIT;
      while((d_aBoard[you]!=ptn)&&(d_aBoard[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(d_aBoard[own]>bit){ return 0; } else if(d_aBoard[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>si-1){ return 4; }
  }
  //270度回転
  if(d_aBoard[BOUND1]==TOPBIT){ own=1; ptn=TOPBIT>>1;
    while(own<=si-1){ bit=1; you=0;
      while((d_aBoard[you]!=ptn)&&(d_aBoard[own]>=bit)){ bit<<=1; you++; }
      if(d_aBoard[own]>bit){ return 0; } else if(d_aBoard[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  return 8; 
}
// GPU -n Ｑが角にある場合のバックトラック内の再帰処理をカーネルで行う
__global__
void BitBoard_cuda_kernel_b1(int size,int mark,unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,unsigned int* d_results,unsigned int* d_uniq,long totalCond,int h_row,int B1)
{
  const unsigned int mask=(1<<size)-1;
  unsigned long total=0;
  unsigned int unique=0;
  int row=0;
  unsigned int bit;
  //
  //スレッド
  //
  //ブロック内のスレッドID
  unsigned const int tid=threadIdx.x;
  //グリッド内のブロックID
  unsigned const int bid=blockIdx.x;
  //全体通してのID
  unsigned const int idx=bid*blockDim.x+tid;
  //
  //シェアードメモリ
  //
  //sharedメモリを使う ブロック内スレッドで共有
  //10固定なのは現在のmask設定で
  //GPUで実行するのは最大10だから
  //THREAD_NUMはブロックあたりのスレッド数
  __shared__ unsigned int down[THREAD_NUM][10];
  down[tid][row]=totalDown[idx];
  __shared__ unsigned int left[THREAD_NUM][10];
  left[tid][row]=totalLeft[idx];
  __shared__ unsigned int right[THREAD_NUM][10];
  right[tid][row]=totalRight[idx];
  __shared__ unsigned int bitmap[THREAD_NUM][10];
  //down,left,rightからbitmapを出す
  bitmap[tid][row]
    =mask&~(
        down[tid][row]
        |left[tid][row]
        |right[tid][row]);
  __shared__ unsigned int sum[THREAD_NUM];
  /***11 backTrack1ではaBoard不要 *********************/
  //unsigned int c_aBoard[MAX];
  __shared__ unsigned int usum[THREAD_NUM];
  //余分なスレッドは動かさない 
  //GPUはsteps数起動するがtotalCond以上は空回しする
  if(idx<totalCond){
    //totalDown,totalLeft,totalRightの情報を
    //down,left,rightに詰め直す 
    //CPU で詰め込んだ t_はsteps個あるが
    //ブロック内ではブロックあたりのスレッド数に限定
    //されるので idxでよい
    //
    /***11 backTrack1ではaBoard不要*********************/
    //for(int i=0;i<h_row;i++){
    //  c_aBoard[i]=t_aBoard[idx*h_row+i]; //２次元配列だが1次元的に利用  
    //}
    unsigned int bitmap_tid_row;
    unsigned int down_tid_row;
    unsigned int left_tid_row;
    unsigned int right_tid_row;
    while(row>=0){
      bitmap_tid_row=bitmap[tid][row];
      down_tid_row=down[tid][row];
      left_tid_row=left[tid][row];
      right_tid_row=right[tid][row];
      if(bitmap_tid_row==0){
        row--;
      }else{
        /**11 枝刈り**********/
        if(row+h_row<B1) {
          bitmap_tid_row=bitmap[tid][row]&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
        }  
        //クイーンを置く
        //置く場所があるかどうか
        bitmap[tid][row]
          /***11 backTrack1ではaBoard不要のためコメント*********************/
          //^=c_aBoard[row+h_row]
          //=bit
          ^=bit
          =(-bitmap_tid_row&bitmap_tid_row);       
        if((bit&mask)!=0){
          //最終行?最終行から１個前の行まで
          //無事到達したら 加算する
          if(row+1==mark){
            /**11 backTradk1ではsymmetryOps不要のためコメント*********************/
            //int s=symmetryOps(size,c_aBoard); 
            //if(s!=0){
            //print(size); //print()でTOTALを++しない
            //ホストに戻す配列にTOTALを入れる
            //スレッドが１つの場合は配列は１個
            unique++; 
            total+=8;   //対称解除で得られた解数を加算
            //}
            row--;
          }else{
            int rowP=row+1;
            down[tid][rowP]=down_tid_row|bit;
            left[tid][rowP]=(left_tid_row|bit)<<1;
            right[tid][rowP]=(right_tid_row|bit)>>1;
            bitmap[tid][rowP]
              =mask&~(
                  down[tid][rowP]
                  |left[tid][rowP]
                  |right[tid][rowP]);
            row++;
          }
        }else{
          //置く場所がなければ１個上に
          row--;
        }
      }
    }
    //最後sum[tid]に加算する
    sum[tid]=total;
    usum[tid]=unique;
  }else{
    //totalCond未満は空回しするのでtotalは加算しない
    sum[tid]=0;
    usum[tid]=0;
  } 
  //__syncthreads()でブロック内のスレッド間の同期
  //全てのスレッドが__syncthreads()に辿り着くのを待つ
  __syncthreads();if(tid<64&&tid+64<THREAD_NUM){
    sum[tid]+=sum[tid+64];
    usum[tid]+=usum[tid+64];
  }
  __syncwarp();if(tid<32){
    sum[tid]+=sum[tid+32];
    usum[tid]+=usum[tid+32];
  } 
  __syncwarp();if(tid<16){
    sum[tid]+=sum[tid+16];
    usum[tid]+=usum[tid+16];
  } 
  __syncwarp();if(tid<8){
    sum[tid]+=sum[tid+8];
    usum[tid]+=usum[tid+8];
  } 
  __syncwarp();if(tid<4){
    sum[tid]+=sum[tid+4];
    usum[tid]+=usum[tid+4];
  } 
  __syncwarp();if(tid<2){
    sum[tid]+=sum[tid+2];
    usum[tid]+=usum[tid+2];
  } 
  __syncwarp();if(tid<1){
    sum[tid]+=sum[tid+1];
    usum[tid]+=usum[tid+1];
  } 
  __syncwarp();if(tid==0){
    d_results[bid]=sum[0];
    d_uniq[bid]=usum[0];
  }
}
// GPU -n Ｑが角にない場合のバックトラック内の再帰処理をカーネルで行う
__global__
void BitBoard_cuda_kernel_b2(int size,int mark,unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,unsigned int* d_results,unsigned int* d_uniq,long totalCond,unsigned int* t_aBoard,int h_row,int B1,int B2,int SM,int LM,int TB,int EB)
{
  const unsigned int mask=(1<<size)-1;
  unsigned long total=0;
  unsigned int unique=0;
  int row=0;
  unsigned int bit;
  //
  //スレッド
  //
  //ブロック内のスレッドID
  unsigned const int tid=threadIdx.x;
  //グリッド内のブロックID
  unsigned const int bid=blockIdx.x;
  //全体通してのID
  unsigned const int idx=bid*blockDim.x+tid;
  //
  //シェアードメモリ
  //
  //sharedメモリを使う ブロック内スレッドで共有
  //10固定なのは現在のmask設定で
  //GPUで実行するのは最大10だから
  //THREAD_NUMはブロックあたりのスレッド数
  __shared__ unsigned int down[THREAD_NUM][10];
  down[tid][row]=totalDown[idx];
  __shared__ unsigned int left[THREAD_NUM][10];
  left[tid][row]=totalLeft[idx];
  __shared__ unsigned int right[THREAD_NUM][10];
  right[tid][row]=totalRight[idx];
  __shared__ unsigned int bitmap[THREAD_NUM][10];
  //down,left,rightからbitmapを出す
  bitmap[tid][row]
    =mask&~(
        down[tid][row]
        |left[tid][row]
        |right[tid][row]);
  __shared__ unsigned int sum[THREAD_NUM];
  unsigned int c_aBoard[MAX];
  __shared__ unsigned int usum[THREAD_NUM];
  //余分なスレッドは動かさない 
  //GPUはsteps数起動するがtotalCond以上は空回しする
  if(idx<totalCond){
    //totalDown,totalLeft,totalRightの情報を
    //down,left,rightに詰め直す 
    //CPU で詰め込んだ t_はsteps個あるが
    //ブロック内ではブロックあたりのスレッド数に限定
    //されるので idxでよい
    //
    for(int i=0;i<h_row;i++){
      c_aBoard[i]=t_aBoard[idx*h_row+i]; //２次元配列だが1次元的に利用  
    }
    unsigned int bitmap_tid_row;
    unsigned int down_tid_row;
    unsigned int left_tid_row;
    unsigned int right_tid_row;
    while(row>=0){
      bitmap_tid_row=bitmap[tid][row];
      down_tid_row=down[tid][row];
      left_tid_row=left[tid][row];
      right_tid_row=right[tid][row];
      //
      //bitmap[tid][row]=00000000 クイーンを
      //どこにも置けないので1行上に戻る
      if(bitmap_tid_row==0){
        row--;
      }else{
        /**11 枝刈り追加**********/
        //【枝刈り】上部サイド枝刈り
        if(row+h_row<B1){             	
          //printf("BOUND1_row:%d:h_row:%d:row+hrow:%d:bit:%d\n",row,h_row,row+h_row,bitmap[tid][row]);
          bitmap_tid_row=bitmap[tid][row]&=~SM;
          //【枝刈り】下部サイド枝刈り
        }else if(row+h_row==B2) {     	
          //printf("BOUND2_row:%d:h_row:%d:row+hrow:%d:bit:%d\n",row,h_row,row+h_row,bitmap[tid][row]);
          if((down_tid_row&SM)==0){ 
            row--; 
            continue;
            //printf("BOUND2_row\n");
          }
          if((down_tid_row&SM)!=SM){ 
            bitmap_tid_row=bitmap[tid][row]&=SM; 
            //printf("BOUND2_SIDEMASK\n");            
          }
        }
        int save_bitmap=bitmap[tid][row];
        //クイーンを置く
        //置く場所があるかどうか
        bitmap[tid][row]
          ^=c_aBoard[row+h_row]
          =bit
          =(-bitmap_tid_row&bitmap_tid_row);       
        if((bit&mask)!=0){
          //最終行?最終行から１個前の行まで
          //無事到達したら 加算する
          if(row+1==mark){
            /***11 LASTMASK枝刈り*********************/ 
            if((save_bitmap&LM)==0){ 
              /***12 symmetryOps 省力化のためBOUND1,BOUND2,TOPBIT,ENDBITを渡す*****/ 
              int s=BitBoard_symmetryOps(size,c_aBoard,B1,B2,TB,EB); 
              if(s!=0){
                //print(size); //print()でTOTALを++しない
                //ホストに戻す配列にTOTALを入れる
                //スレッドが１つの場合は配列は１個
                unique++; 
                total+=s;   //対称解除で得られた解数を加算
              }
              row--;
            }
          }else{
            int rowP=row+1;
            down[tid][rowP]=down_tid_row|bit;
            left[tid][rowP]=(left_tid_row|bit)<<1;
            right[tid][rowP]=(right_tid_row|bit)>>1;
            bitmap[tid][rowP]
              =mask&~(
                  down[tid][rowP]
                  |left[tid][rowP]
                  |right[tid][rowP]);
            row++;
          }
        }else{
          //置く場所がなければ１個上に
          row--;
        }
      }
    }
    //最後sum[tid]に加算する
    sum[tid]=total;
    usum[tid]=unique;
  }else{
    //totalCond未満は空回しするのでtotalは加算しない
    sum[tid]=0;
    usum[tid]=0;
  } 
  //__syncthreads()でブロック内のスレッド間の同期
  //全てのスレッドが__syncthreads()に辿り着くのを待つ
  __syncthreads();if(tid<64&&tid+64<THREAD_NUM){
    sum[tid]+=sum[tid+64];
    usum[tid]+=usum[tid+64];
  }
  __syncwarp();if(tid<32){
    sum[tid]+=sum[tid+32];
    usum[tid]+=usum[tid+32];
  } 
  __syncwarp();if(tid<16){
    sum[tid]+=sum[tid+16];
    usum[tid]+=usum[tid+16];
  } 
  __syncwarp();if(tid<8){
    sum[tid]+=sum[tid+8];
    usum[tid]+=usum[tid+8];
  } 
  __syncwarp();if(tid<4){
    sum[tid]+=sum[tid+4];
    usum[tid]+=usum[tid+4];
  } 
  __syncwarp();if(tid<2){
    sum[tid]+=sum[tid+2];
    usum[tid]+=usum[tid+2];
  } 
  __syncwarp();if(tid<1){
    sum[tid]+=sum[tid+1];
    usum[tid]+=usum[tid+1];
  } 
  __syncwarp();if(tid==0){
    d_results[bid]=sum[0];
    d_uniq[bid]=usum[0];
  }
}
// GPU -n Ｑが角にない
long BitBoard_backTrack2G(int size,int mask,int row,int n_left,int n_down,int n_right,int steps,int BOUND1,int BOUND2,int SIDEMASK,int LASTMASK,int TOPBIT,int ENDBIT,unsigned int* aBoard)
{
  //何行目からGPUで行くか。ここの設定は変更可能、設定値を多くするほどGPUで並行して動く
  /***11 size<8の時はmarkが2*********************/
  unsigned int mark=size>12?size-10:3;
  //unsigned int mark=size>11?size-9:3;
  if(size<8){ mark=2; }
  const unsigned int h_mark=row;
  long total=0;
  long totalCond=0;
  bool matched=false;
  //host
  unsigned int down[32];  down[row]=n_down;
  unsigned int right[32]; right[row]=n_right;
  unsigned int left[32];  left[row]=n_left;
  //bitmapを配列で持つことにより
  //stackを使わないで1行前に戻れる
  unsigned int bitmap[32];
  bitmap[row]=mask&~(left[row]|down[row]|right[row]);
  unsigned int bit;
  unsigned int* totalDown;
  cudaMallocHost((void**) &totalDown,sizeof(int)*steps);
  unsigned int* totalLeft;
  cudaMallocHost((void**) &totalLeft,sizeof(int)*steps);
  unsigned int* totalRight;
  cudaMallocHost((void**) &totalRight,sizeof(int)*steps);
  unsigned int* h_results;
  cudaMallocHost((void**) &h_results,sizeof(int)*steps);
  unsigned int* h_uniq;
  cudaMallocHost((void**) &h_uniq,sizeof(int)*steps);
  unsigned int* t_aBoard;
  cudaMallocHost((void**) &t_aBoard,sizeof(int)*steps*mark);
  //device
  unsigned int* downCuda;
  cudaMalloc((void**) &downCuda,sizeof(int)*steps);
  unsigned int* leftCuda;
  cudaMalloc((void**) &leftCuda,sizeof(int)*steps);
  unsigned int* rightCuda;
  cudaMalloc((void**) &rightCuda,sizeof(int)*steps);
  unsigned int* resultsCuda;
  cudaMalloc((void**) &resultsCuda,sizeof(int)*steps/THREAD_NUM);
  unsigned int* d_uniq;
  cudaMalloc((void**) &d_uniq,sizeof(int)*steps/THREAD_NUM);
  unsigned int* d_aBoard;
  cudaMalloc((void**) &d_aBoard,sizeof(int)*steps*mark);
  //12行目までは3行目までCPU->row==mark以下で 3行目までの
  //down,left,right情報を totalDown,totalLeft,totalRight
  //に格納
  //する->3行目以降をGPUマルチスレッドで実行し結果を取得
  //13行目以降はCPUで実行する行数が１個ずつ増えて行く
  //例えばn15だとrow=5までCPUで実行し、
  //それ以降はGPU(現在の設定だとGPUでは最大10行実行する
  //ようになっている)
  int rowP=0;
  while(row>=h_mark) {
    //bitmap[row]=00000000 クイーンを
    //どこにも置けないので1行上に戻る
    //06GPU こっちのほうが優秀
    if(bitmap[row]==0){ row--; }
    else{//おける場所があれば進む
      /***11 枝刈り追加*********************/
      //【枝刈り】上部サイド枝刈り
      if(row<BOUND1){             	
        bitmap[row]&=~SIDEMASK;
        //【枝刈り】下部サイド枝刈り
      }else if(row==BOUND2) {     	
        if((down[row]&SIDEMASK)==0){ row--; }
        if((down[row]&SIDEMASK)!=SIDEMASK){ bitmap[row]&=SIDEMASK; }
      }
      //06SGPU
      bitmap[row]^=aBoard[row]=bit=(-bitmap[row]&bitmap[row]);
      if((bit&mask)!=0){//置く場所があれば先に進む
        rowP=row+1;
        down[rowP]=down[row]|bit;
        left[rowP]=(left[row]|bit)<<1;
        right[rowP]=(right[row]|bit)>>1;
        bitmap[rowP]=mask&~(down[rowP]|left[rowP]|right[rowP]);
        row++;
        if(row==mark){
          //3行目(mark)にクイーンを１個ずつ置いていって、
          //down,left,right情報を格納、
          //その次の行へは進まない。その行で可能な場所にクイー
          //ン置き終わったらGPU並列実行
          //totalCond がthreadIdになる 各スレッドに down,left,right情報を渡す
          //row=2(13行目以降は増えていく。例えばn15だとrow=5)の情報を
          //totalDown,totalLeft,totalRightに格納する
          totalDown[totalCond]=down[row];
          totalLeft[totalCond]=left[row];
          totalRight[totalCond]=right[row];
          for(int i=0;i<mark;i++){
            t_aBoard[totalCond*mark+i]=aBoard[i];
          }
          //スレッド数をインクリメントする
          totalCond++;
          //最大GPU数に達してしまったら一旦ここでGPUを実行する。stepsはGPUの同
          //時並行稼働数を制御
          //nの数が少ないうちはtotalCondがstepsを超えることはないがnの数が増え
          //て行くと超えるようになる。
          //ここではtotalCond==stepsの場合だけこの中へ         
          if(totalCond==steps){
            //matched=trueの時にCOUNT追加 //GPU内でカウントしているので、GPUか
            //ら出たらmatched=trueになってる
            if(matched){
              cudaMemcpy(h_results,resultsCuda,
                  sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
              cudaMemcpy(h_uniq,d_uniq,
                  sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
              for(int col=0;col<steps/THREAD_NUM;col++){
                total+=h_results[col];
                UNIQUE+=h_uniq[col];
              }
              matched=false;
            }
            cudaMemcpy(downCuda,totalDown,
                sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            cudaMemcpy(leftCuda,totalLeft,
                sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            cudaMemcpy(rightCuda,totalRight,
                sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            cudaMemcpy(d_aBoard,t_aBoard,
                sizeof(int)*totalCond*mark,cudaMemcpyHostToDevice);
            /***12 TOPBIT,ENDBIT追加*********************/
            //cuda_kernel_b2<<<steps/THREAD_NUM,THREAD_NUM
            //  >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,row,BOUND1,BOUND2,SIDEMASK,LASTMASK);
            BitBoard_cuda_kernel_b2<<<steps/THREAD_NUM,THREAD_NUM
              >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,row,BOUND1,BOUND2,SIDEMASK,LASTMASK,TOPBIT,ENDBIT);
            //steps数の数だけマルチスレッドで起動するのだが、実際に計算が行われ
            //るのはtotalCondの数だけでそれ以外は空回しになる
            //GPU内でカウントしているので、GPUから出たらmatched=trueになってる
            matched=true;
            //totalCond==stepsルートでGPUを実行したらスレッドをまた0から開始す
            //る(これによりなんどもsteps数分だけGPUを起動できる)
            totalCond=0;           
          }
          //totalDown,totalLeft,totalRightに情報を格納したら1行上に上がる
          //これを繰り返すことにより row=2で可能な場所全てにクイーンを置いて
          //totalDown,totalLeft,totalRightに情報を格納する
          row--;
        }
      }else{
        //置く場所がなければ上に上がる。row==mark行に達するまではCPU側で普通に
        //nqueenをやる
        row--;
      }
    }
  }
  //matched=trueの時にCOUNT追加 //GPU内でカウントしているので、GPUから出たら
  //matched=trueになってる
  if(matched){
    cudaMemcpy(h_results,resultsCuda,
        sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uniq,d_uniq,
        sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    for(int col=0;col<steps/THREAD_NUM;col++){
      total+=h_results[col];
      UNIQUE+=h_uniq[col];
    }
    matched=false;
  }
  cudaMemcpy(downCuda,totalDown,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(leftCuda,totalLeft,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(rightCuda,totalRight,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(d_aBoard,t_aBoard,
      sizeof(int)*totalCond*mark,cudaMemcpyHostToDevice);
  //size-mark は何行GPUを実行するか totalCondはスレッド数
  //steps数の数だけマルチスレッドで起動するのだが、実際に計算が行われるのは
  //totalCondの数だけでそれ以外は空回しになる
  /***12 TOPBIT,ENDBIT追加*********************/
  //cuda_kernel_b2<<<steps/THREAD_NUM,THREAD_NUM
  //  >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,mark,BOUND1,BOUND2,SIDEMASK,LASTMASK);
  BitBoard_cuda_kernel_b2<<<steps/THREAD_NUM,THREAD_NUM
    >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,mark,BOUND1,BOUND2,SIDEMASK,LASTMASK,TOPBIT,ENDBIT);
  cudaMemcpy(h_results,resultsCuda,
      sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_uniq,d_uniq,
      sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  for(int col=0;col<steps/THREAD_NUM;col++){
    total+=h_results[col];
    UNIQUE+=h_uniq[col];
  }
  //
  cudaFree(downCuda);
  cudaFree(leftCuda);
  cudaFree(rightCuda);
  cudaFree(resultsCuda);
  cudaFree(d_uniq);
  cudaFree(d_aBoard);
  cudaFreeHost(totalDown);
  cudaFreeHost(totalLeft);
  cudaFreeHost(totalRight);
  cudaFreeHost(h_results);
  cudaFreeHost(h_uniq);
  cudaFreeHost(t_aBoard);
  return total;
}
// GPU -n Ｑが角にある
long BitBoard_backTrack1G(int size,int mask,int row,int n_left,int n_down,int n_right,int steps,int BOUND1)
{
  //何行目からGPUで行くか。ここの設定は変更可能、設定値を多くするほどGPUで並行して動く
  /***08 クイーンを２行目まで固定で置くためmarkが3以上必要*********************/
  const unsigned int mark=size>12?size-10:3;
  const unsigned int h_mark=row;
  long total=0;
  long totalCond=0;
  bool matched=false;
  //host
  unsigned int down[32];  down[row]=n_down;
  unsigned int right[32]; right[row]=n_right;
  unsigned int left[32];  left[row]=n_left;
  //bitmapを配列で持つことにより
  //stackを使わないで1行前に戻れる
  unsigned int bitmap[32];
  bitmap[row]=mask&~(left[row]|down[row]|right[row]);
  unsigned int bit;
  unsigned int* totalDown;
  cudaMallocHost((void**) &totalDown,sizeof(int)*steps);
  unsigned int* totalLeft;
  cudaMallocHost((void**) &totalLeft,sizeof(int)*steps);
  unsigned int* totalRight;
  cudaMallocHost((void**) &totalRight,sizeof(int)*steps);
  unsigned int* h_results;
  cudaMallocHost((void**) &h_results,sizeof(int)*steps);
  unsigned int* h_uniq;
  cudaMallocHost((void**) &h_uniq,sizeof(int)*steps);
  /***11 backTrack1ではaBoard不要のためコメント*********************/
  //unsigned int* t_aBoard;
  //cudaMallocHost((void**) &t_aBoard,sizeof(int)*steps*mark);
  //device
  unsigned int* downCuda;
  cudaMalloc((void**) &downCuda,sizeof(int)*steps);
  unsigned int* leftCuda;
  cudaMalloc((void**) &leftCuda,sizeof(int)*steps);
  unsigned int* rightCuda;
  cudaMalloc((void**) &rightCuda,sizeof(int)*steps);
  unsigned int* resultsCuda;
  cudaMalloc((void**) &resultsCuda,sizeof(int)*steps/THREAD_NUM);
  unsigned int* d_uniq;
  cudaMalloc((void**) &d_uniq,sizeof(int)*steps/THREAD_NUM);
  /***11 backTrack1ではaBoard不要のためコメント*********************/
  //unsigned int* d_aBoard;
  //cudaMalloc((void**) &d_aBoard,sizeof(int)*steps*mark);
  //12行目までは3行目までCPU->row==mark以下で 3行目までの
  //down,left,right情報を totalDown,totalLeft,totalRight
  //に格納
  //する->3行目以降をGPUマルチスレッドで実行し結果を取得
  //13行目以降はCPUで実行する行数が１個ずつ増えて行く
  //例えばn15だとrow=5までCPUで実行し、
  //それ以降はGPU(現在の設定だとGPUでは最大10行実行する
  //ようになっている)
  //while(row>=0) {
  int rowP=0;
  while(row>=h_mark) {
    //bitmap[row]=00000000 クイーンを
    //どこにも置けないので1行上に戻る
    //06GPU こっちのほうが優秀
    if(bitmap[row]==0){ row--; }
    else{//おける場所があれば進む
      /***11 枝刈り*********************/
      if(row<BOUND1) {
        bitmap[row]&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
      }
      //06SGPU
      /***11 aBoard不要*********************/
      //bitmap[row]^=aBoard[row]=bit=(-bitmap[row]&bitmap[row]);
      bitmap[row]^=bit=(-bitmap[row]&bitmap[row]);
      if((bit&mask)!=0){//置く場所があれば先に進む
        rowP=row+1;
        down[rowP]=down[row]|bit;
        left[rowP]=(left[row]|bit)<<1;
        right[rowP]=(right[row]|bit)>>1;
        bitmap[rowP]=mask&~(down[rowP]|left[rowP]|right[rowP]);
        row++;
        if(row==mark){
          //3行目(mark)にクイーンを１個ずつ置いていって、
          //down,left,right情報を格納、
          //その次の行へは進まない。その行で可能な場所にクイー
          //ン置き終わったらGPU並列実行
          //totalCond がthreadIdになる 各スレッドに down,left,right情報を渡す
          //row=2(13行目以降は増えていく。例えばn15だとrow=5)の情報を
          //totalDown,totalLeft,totalRightに格納する         
          totalDown[totalCond]=down[row];
          totalLeft[totalCond]=left[row];
          totalRight[totalCond]=right[row];
          /***11 aBoardコメント*********************/
          //for(int i=0;i<mark;i++){
          //  t_aBoard[totalCond*mark+i]=aBoard[i];
          //}
          //スレッド数をインクリメントする
          totalCond++;
          //最大GPU数に達してしまったら一旦ここでGPUを実行する。stepsはGPUの同
          //時並行稼働数を制御
          //nの数が少ないうちはtotalCondがstepsを超えることはないがnの数が増え
          //て行くと超えるようになる。
          //ここではtotalCond==stepsの場合だけこの中へ         
          if(totalCond==steps){
            //matched=trueの時にCOUNT追加 //GPU内でカウントしているので、GPUか
            //ら出たらmatched=trueになってる
            if(matched){
              cudaMemcpy(h_results,resultsCuda,
                  sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
              cudaMemcpy(h_uniq,d_uniq,
                  sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
              for(int col=0;col<steps/THREAD_NUM;col++){
                total+=h_results[col];
                UNIQUE+=h_uniq[col];
              }
              matched=false;
            }
            cudaMemcpy(downCuda,totalDown,
                sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            cudaMemcpy(leftCuda,totalLeft,
                sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            cudaMemcpy(rightCuda,totalRight,
                sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            /***11 aBoard不要のためコメント*********************/
            //cudaMemcpy(d_aBoard,t_aBoard,
            //    sizeof(int)*totalCond*mark,cudaMemcpyHostToDevice);
            /***11 BOUND1追加*********************/
            //cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
            //  >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,row);
            BitBoard_cuda_kernel_b1<<<steps/THREAD_NUM,THREAD_NUM
              >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,row,BOUND1);

            //steps数の数だけマルチスレッドで起動するのだが、実際に計算が行われ
            //るのはtotalCondの数だけでそれ以外は空回しになる
            //GPU内でカウントしているので、GPUから出たらmatched=trueになってる
            matched=true;
            //totalCond==stepsルートでGPUを実行したらスレッドをまた0から開始す
            //る(これによりなんどもsteps数分だけGPUを起動できる)
            totalCond=0;           
          }
          //totalDown,totalLeft,totalRightに情報を格納したら1行上に上がる
          //これを繰り返すことにより row=2で可能な場所全てにクイーンを置いて
          //totalDown,totalLeft,totalRightに情報を格納する
          row--;
        }
      }else{
        //置く場所がなければ上に上がる。row==mark行に達するまではCPU側で普通に
        //nqueenをやる
        row--;
      }
    }
  }
  //matched=trueの時にCOUNT追加 //GPU内でカウントしているので、GPUから出たら
  //matched=trueになってる
  if(matched){
    cudaMemcpy(h_results,resultsCuda,
        sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uniq,d_uniq,
        sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    for(int col=0;col<steps/THREAD_NUM;col++){
      total+=h_results[col];
      UNIQUE+=h_uniq[col];
    }
    matched=false;
  }
  cudaMemcpy(downCuda,totalDown,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(leftCuda,totalLeft,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(rightCuda,totalRight,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  /***11 aBoard不要のためコメント*********************/
  //cudaMemcpy(d_aBoard,t_aBoard,
  //    sizeof(int)*totalCond*mark,cudaMemcpyHostToDevice);
  //size-mark は何行GPUを実行するか totalCondはスレッド数
  //steps数の数だけマルチスレッドで起動するのだが、実際に計算が行われるのは
  //totalCondの数だけでそれ以外は空回しになる
  /***11 BOUND1追加*********************/
  //cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
  //  >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,mark);
  BitBoard_cuda_kernel_b1<<<steps/THREAD_NUM,THREAD_NUM
    >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,mark,BOUND1);
  cudaMemcpy(h_results,resultsCuda,
      sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_uniq,d_uniq,
      sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  for(int col=0;col<steps/THREAD_NUM;col++){
    total+=h_results[col];
    UNIQUE+=h_uniq[col];
  }
  //
  cudaFree(downCuda);
  cudaFree(leftCuda);
  cudaFree(rightCuda);
  cudaFree(resultsCuda);
  cudaFree(d_uniq);
  /***11 aBoardコメント**/
  //cudaFree(d_aBoard);
  cudaFreeHost(totalDown);
  cudaFreeHost(totalLeft);
  cudaFreeHost(totalRight);
  cudaFreeHost(h_results);
  cudaFreeHost(h_uniq);
  /***11 aBoardコメント**/
  //cudaFreeHost(t_aBoard);
  return total;
}
// GPU -n ビットボードの実行 角にＱがある・ないの分岐を行う
void symmetry_build_bitBoard(unsigned int size,int steps)
{
  if(size<=0||size>32){return;}
  /**
    パラメータは渡す変数はregisterとする
    int型は unsigned とする
    total: グローバル変数TOTALへのアクセスを極小化する
    sizeE:size-1といった計算を変数に格納しフラット化する 
    */
  unsigned long total=0;
  unsigned int sizeE=size-1;
  unsigned int aBoard[MAX];
  int bit=0;
  int mask=((1<<size)-1);
  int col=0;//1行め右端 0
  aBoard[0]=bit=(1<<col);
  int left=bit<<1,down=bit,right=bit>>1;
  /**
    2行目は右から3列目から左端から2列目まで
  */
  for(int BOUND1=2;BOUND1<sizeE;BOUND1++){
    aBoard[1]=bit=(1<<BOUND1);
    total+=BitBoard_backTrack1G(size,mask,2,
        (left|bit)<<1,(down|bit),(right|bit)>>1,
        steps,BOUND1);
  }
  int LASTMASK,SIDEMASK;
  int TOPBIT=1<<(sizeE);
  SIDEMASK=LASTMASK=(TOPBIT|1);
  int ENDBIT=(TOPBIT>>1);
  /**
    1行目右から2列目から
    偶数個は1/2 n=8 なら 1,2,3 奇数個は1/2+1 n=9 なら 1,2,3,4
  */
  for(int BOUND1=1,BOUND2=sizeE-1;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    total+=BitBoard_backTrack2G(size,mask,1,
        bit<<1,bit,bit>>1,
        steps,BOUND1,BOUND2,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,aBoard);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  }
  /**
    グローバル変数へのアクセスを極小化する
    */
  TOTAL=total;
}
// CUDA 初期化
bool InitCUDA()
{
  int count;
  cudaGetDeviceCount(&count);
  if(count==0){fprintf(stderr,"There is no device.\n");return false;}
  unsigned int i;
  for(i=0;i<count;++i){
    struct cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop,i)==cudaSuccess){if(prop.major>=1){break;} }
  }
  if(i==count){fprintf(stderr,"There is no device supporting CUDA 1.x.\n");return false;}
  cudaSetDevice(i);
  return true;
}
//メイン
int main(int argc,char** argv)
{
  bool cpu=false,cpur=false,gpu=false,gpuBitBoard=false;
  unsigned int argstart=2;
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
    else if(argv[1][1]=='n'||argv[1][1]=='N'){gpuBitBoard=true;}
    else{ gpuBitBoard=true; } //デフォルトをgpuとする
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g|-r|-s] n steps\n",argv[0]);
    printf("  -r: CPU 再帰\n");
    printf("  -c: CPU 非再帰\n");
    printf("  -g: GPU 再帰\n");
    printf("  -n: GPU ビットボード\n");
  }
  if(cpur){ printf("\n\n対称解除法 再帰 \n"); }
  else if(cpu){ printf("\n\n対称解除法 非再帰 \n"); }
  else if(gpu){ printf("\n\n対称解除法 GPU\n"); }
  else if(gpuBitBoard){ printf("\n\n対称解除法 GPUビットボード \n"); }
  if(cpu||cpur)
  {
    unsigned int min=4; 
    unsigned int targetN=17;
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    for(unsigned int size=min;size<=targetN;size++){
      local l;
      gettimeofday(&t0,NULL);//計測開始
      if(cpur){ //再帰
        symmetry_R(size,&l);
      }
      if(cpu){ //非再帰
        symmetry_NR(size,&l);
      }
      //
      gettimeofday(&t1,NULL);//計測終了
      unsigned int ss;
      unsigned int ms;
      unsigned int dd;
      if(t1.tv_usec<t0.tv_usec) {
        dd=(t1.tv_sec-t0.tv_sec-1)/86400;
        ss=(t1.tv_sec-t0.tv_sec-1)%86400;
        ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
      }else {
        dd=(t1.tv_sec-t0.tv_sec)/86400;
        ss=(t1.tv_sec-t0.tv_sec)%86400;
        ms=(t1.tv_usec-t0.tv_usec+500)/10000;
      }//end if
      unsigned int hh=ss/3600;
      unsigned int mm=(ss-hh*3600)/60;
      ss%=60;
      printf("%2d:%13ld%12ld%8.2d:%02d:%02d:%02d.%02d\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
    } //end for
  }//end if
  if(gpu||gpuBitBoard)
  {
    int steps=24576;
    if(!InitCUDA()){return 0;}
    unsigned int min=4;
    unsigned int targetN=21;
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    for(unsigned int size=min;size<=targetN;size++){
      gettimeofday(&t0,NULL);
      if(gpu){
        TOTAL=UNIQUE=0;
        local l[MAX];
        GPU_symmetry_R(size,&l[0]);
        TOTAL=l->TOTAL;
        UNIQUE=l->UNIQUE;
      }else if(gpuBitBoard){
        TOTAL=UNIQUE=0;
        symmetry_build_bitBoard(size,steps);
      }
      gettimeofday(&t1,NULL);
      unsigned int ss;
      unsigned int ms;
      unsigned int dd;
      if (t1.tv_usec<t0.tv_usec) {
        dd=(int)(t1.tv_sec-t0.tv_sec-1)/86400;
        ss=(t1.tv_sec-t0.tv_sec-1)%86400;
        ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
      } else {
        dd=(int)(t1.tv_sec-t0.tv_sec)/86400;
        ss=(t1.tv_sec-t0.tv_sec)%86400;
        ms=(t1.tv_usec-t0.tv_usec+500)/10000;
      }//end if
      unsigned int hh=ss/3600;
      unsigned int mm=(ss-hh*3600)/60;
      ss%=60;
      printf("%2d:%13ld%12ld%8.2d:%02d:%02d:%02d.%02d\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
    }
  }
  return 0;
}
