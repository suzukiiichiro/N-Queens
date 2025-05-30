/**
 *
 * bash版対称解除法のC言語版のGPU/CUDA移植版
 *
 詳しい説明はこちらをどうぞ
 https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

非再帰でのコンパイルと実行
$ nvcc -O3 -arch=sm_61 03CUDA_Symmetry_NodeLayer.cu && ./a.out -c

再帰でのコンパイルと実行
$ nvcc -O3 -arch=sm_61 03CUDA_Symmetry_NodeLayer.cu && ./a.out -r

GPU で並列処理せずに実行
$ nvcc -O3 -arch=sm_61 03CUDA_Symmetry_NodeLayer.cu && ./a.out -n

GPU で並列処理で実行（ノードレイヤー）
$ nvcc -O3 -arch=sm_61 03CUDA_Symmetry_NodeLayer.cu && ./a.out -n

対称解除法 GPUノードレイヤー
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            0           0      00:00:00:00.13
 5:            0           0      00:00:00:00.00
 6:            0           0      00:00:00:00.00
 7:           40           0      00:00:00:00.00
 8:           92           0      00:00:00:00.00
 9:          352           0      00:00:00:00.00
10:          724           0      00:00:00:00.00
11:         2680           0      00:00:00:00.00
12:        14200           0      00:00:00:00.00
13:        73712           0      00:00:00:00.03
14:       365596           0      00:00:00:00.15
15:      2279184           0      00:00:00:00.86
16:     14772512           0      00:00:00:05.50
17:     95815104           0      00:00:00:38.87
18:    666090624           0      00:00:04:46.16

以降はバーストします。

コメント
・std::vector<long> kLayer_nodeLayer(unsigned int size,unsigned int k,std::vector<local>& L)
 5行分backtrack1,2を実行し、実行結果をnodes,Lに格納する
 Lは、vector 構造体 l を格納する
 lはBOUND1,BOUND2,TOPBIT,ENDBIT,SIDEMASK,LASTMASK,board[MAX]を格納する
 (COUNT2,4,8 TOTAL,UNIQUEは今回は不要)

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
#define THREAD_NUM (1<<MAX)-1
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
struct dlocal{
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
  GPU -n
  */
// GPU -n の 対称解除法
__host__ __device__
long GPUN_symmetryOps(unsigned int size,struct local* l)
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
// 0以外のbitをカウント
__host__ __device__
unsigned int countBits_nodeLayer(unsigned long n)
{
  unsigned int counter=0;
  while(n){ n&=(n-1); counter++; }
  return counter;
}
// GPU -n ノードレイヤーによる対称解除法 -n の実行時に呼び出される
__host__ __device__ 
long GPU_symmetry_solve_nodeLayer_corner(unsigned int size,unsigned long left,unsigned long down,unsigned long right,struct local* l)
{
  unsigned long counter = 0;
  unsigned long mask=(1<<size)-1;
  unsigned long bitmap=mask&~(left|down|right);
  unsigned long bit=0;
  unsigned int row=countBits_nodeLayer(down);
  if(row==(size-1)){
    if(bitmap){
      l->board[row]=bitmap;
      return 8;
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
      counter += GPU_symmetry_solve_nodeLayer_corner(size,(left|bit)<<1,(down|bit),(right|bit)>>1,l);
    }
  }
  return counter;
}
// GPU -n ノードレイヤーによる対称解除法 -n の実行時に呼び出される
__host__ __device__ 
long GPU_symmetry_solve_nodeLayer(unsigned int size,unsigned long left,unsigned long down,unsigned long right,struct local* l)
{
  unsigned long counter = 0;
  unsigned long mask=(1<<size)-1;
  unsigned long bitmap=mask&~(left|down|right);
  unsigned int row=countBits_nodeLayer(down);
  if(row==(size-1)){
    if(bitmap){
      if( (bitmap& l->LASTMASK)==0){
        l->board[row]=bitmap;  //Qを配置
        return GPUN_symmetryOps(size,l);    //対称解除;
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
      unsigned long bit=-bitmap&bitmap;
      bitmap=bitmap^bit;
      l->board[row]=bit;
      counter+=GPU_symmetry_solve_nodeLayer(size,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
  return counter;
}
// GPU -n ノードレイヤー i 番目のメンバを i 番目の部分木の解で埋める
__global__ 
void dim_nodeLayer(unsigned int size,long* nodes,long* solutions,unsigned int numElements,struct local* l)
{
  unsigned int i=blockDim.x*blockIdx.x+threadIdx.x;
  if(i<numElements){
    if(l[i].TYPE==0){
      solutions[i]=GPU_symmetry_solve_nodeLayer_corner(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2],&l[i]);
    }else{
      solutions[i]=GPU_symmetry_solve_nodeLayer(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2],&l[i]);
    }
  }
}
// GPU -n Ｋレイヤー 再帰 角にQがないときのバックトラック
unsigned long kLayer_nodeLayer_backTrack(int size,std::vector<long>& nodes,unsigned int k,unsigned long left,unsigned long down,unsigned long right,std::vector<local>& L,struct local* l)
{
  unsigned long counter=0;
  unsigned long mask=(1<<size)-1;
  unsigned long bitmap=mask&~(left|down|right);
  unsigned int row=countBits_nodeLayer(down);
  if(row==k) {
      nodes.push_back(left);
      nodes.push_back(down);
      nodes.push_back(right);
      L.push_back(*l);
      return 1;
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
      unsigned long bit=-bitmap&bitmap;
      bitmap=bitmap^bit;
      l->board[row]=bit;
      counter+=kLayer_nodeLayer_backTrack(size,nodes,k,(left|bit)<<1,(down|bit),(right|bit)>>1,L,l); 
    }
  }
  return counter;
}
// GPU -n Ｋレイヤー 角にQがあるときのバックトラック
unsigned long kLayer_nodeLayer_backTrack_corner(unsigned int size,std::vector<long>& nodes,unsigned int k,unsigned long left,unsigned long down,unsigned long right,std::vector<local>& L,struct local* l)
{
  unsigned long counter=0;
  unsigned long mask=(1<<size)-1;
  unsigned long bitmap=mask&~(left|down|right);
  unsigned long bit=0;
  int row=countBits_nodeLayer(down);
    if(row==k) {
      nodes.push_back(left);
      nodes.push_back(down);
      nodes.push_back(right);
      L.push_back(*l);
    }
    if(row<l->BOUND1){   //枝刈り
      bitmap=bitmap|2;
      bitmap=bitmap^2;
    }
    while(bitmap){
      bit=-bitmap&bitmap;
      bitmap=bitmap^bit;
      l->board[row]=bit;   //Qを配置
      counter+=kLayer_nodeLayer_backTrack_corner(size,nodes,k,(left|bit)<<1,(down|bit),(right|bit)>>1,L,l); 
    }
  return counter;
}
// GPU -n Ｋレイヤー k 番目のレイヤのすべてのノードを含むベクトルを返す
std::vector<long> kLayer_nodeLayer(unsigned int size,unsigned int k,std::vector<local>& L)
{
  std::vector<long> nodes{};
  unsigned int bit=0;
  struct local l;
  l.TOTAL=l.UNIQUE=l.COUNT2=l.COUNT4=l.COUNT8=0;
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
      l.TYPE=0;
      kLayer_nodeLayer_backTrack_corner(size,nodes,k,(2|bit)<<1,1|bit,(2|bit)>>1,L,&l);
    }
    l.BOUND1++;
  }
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
      l.TYPE=1;
      kLayer_nodeLayer_backTrack(size,nodes,k,bit<<1,bit,bit>>1,L,&l);
    }
    l.BOUND1++;
    l.BOUND2--;
    l.ENDBIT=l.ENDBIT>>1;
    l.LASTMASK=l.LASTMASK<<1|l.LASTMASK|l.LASTMASK>>1;
  }
  return nodes;
}
// GPU -n ノードレイヤーの作成
void symmetry_build_nodeLayer(unsigned int size)
{
  // ツリーの3番目のレイヤーにあるノード
  //（それぞれ連続する3つの数字でエンコードされる）のベクトル。
  // レイヤー2以降はノードの数が均等なので対称性を利用できる。
  // レイヤ4には十分なノードがある（N16の場合、9844）。
  // ここではレイヤーを５に設定、Ｎに併せて増やしていく
  std::vector<local>L;
  // NodeLayerは、N18でabortします。
  std::vector<long> nodes=kLayer_nodeLayer(size,5,L); 

  // デバイスにはクラスがないので、
  // 最初の要素を指定してからデバイスにコピーする。
  size_t nodeSize=nodes.size() * sizeof(long);
  long* hostNodes=(long*)malloc(nodeSize);
  hostNodes=&nodes[0];
  long* deviceNodes=NULL;
  cudaMalloc((void**)&deviceNodes,nodeSize);
  cudaMemcpy(deviceNodes,hostNodes,nodeSize,cudaMemcpyHostToDevice);

  // host/device Local
  //size_t localSize=numSolutions * sizeof(struct local);
  size_t localSize=L.size() * sizeof(struct local);
  struct local* hostLocal=(local*)malloc(localSize);
  hostLocal=&L[0];
  local* deviceLocal=NULL;
  cudaMalloc((void**)&deviceLocal,localSize);
  cudaMemcpy(deviceLocal, hostLocal, localSize, cudaMemcpyHostToDevice);

  // デバイス出力の割り当て
  // 必要なのはノードの半分だけで、
  // 各ノードは3つの整数で符号化される。
  long* deviceSolutions=NULL;
  int numSolutions=nodes.size() / 3; 
  size_t solutionSize=numSolutions * sizeof(long);
  cudaMalloc((void**)&deviceSolutions,solutionSize);

  // CUDAカーネルを起動する。
  int threadsPerBlock=256;
  int blocksPerGrid=(numSolutions+threadsPerBlock-1)/threadsPerBlock;
  dim_nodeLayer <<<blocksPerGrid,threadsPerBlock>>>(size,deviceNodes,deviceSolutions,numSolutions,deviceLocal);

  // 結果をホストにコピー
  long* hostSolutions=(long*)malloc(solutionSize);
  cudaMemcpy(hostSolutions,deviceSolutions,solutionSize,cudaMemcpyDeviceToHost);
  
  // 部分解を加算し、結果を表示する。
  unsigned long solutions=0;
  for(unsigned long i=0;i<numSolutions;i++){
       solutions += hostSolutions[i]; // Symmetry
  }
  // 出力
  TOTAL=solutions;
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
  bool cpu=false,cpur=false,gpu=false,gpuNodeLayer=false;
  unsigned int argstart=2;
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
    else if(argv[1][1]=='n'||argv[1][1]=='N'){gpuNodeLayer=true;}
    else{ gpuNodeLayer=true; } //デフォルトをgpuとする
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g|-r|-s] n steps\n",argv[0]);
    printf("  -r: CPU 再帰\n");
    printf("  -c: CPU 非再帰\n");
    printf("  -g: GPU 再帰\n");
    printf("  -n: GPU ノードレイヤー\n");
  }
  if(cpur){ printf("\n\n対称解除法 再帰 \n"); }
  else if(cpu){ printf("\n\n対称解除法 非再帰 \n"); }
  else if(gpu){ printf("\n\n対称解除法 GPU\n"); }
  else if(gpuNodeLayer){ printf("\n\n対称解除法 GPUノードレイヤー \n"); }
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
  if(gpu||gpuNodeLayer)
  {
    if(!InitCUDA()){return 0;}
    /* int steps=24576; */
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
      }else if(gpuNodeLayer){
        TOTAL=UNIQUE=0;
        symmetry_build_nodeLayer(size);
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
