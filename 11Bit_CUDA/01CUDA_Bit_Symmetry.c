/**
 *
 * bash版対称解除法のC言語版のGPU/CUDA移植版
 *
 詳しい説明はこちらをどうぞ
 https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

 非再帰でのコンパイルと実行
 $ nvcc -O3 -arch=sm_61 04CUDA_Symmetry_BitBoard.cu && ./a.out -c

 再帰でのコンパイルと実行
 $ nvcc -O3 -arch=sm_61 04CUDA_Symmetry_BitBoard.cu && ./a.out -r

 GPU で並列処理せずに実行
 $ nvcc -O3 -arch=sm_61 04CUDA_Symmetry_BitBoard.cu && ./a.out -g

 GPU で並列処理で実行（ビットボード）
 $ nvcc -O3 -arch=sm_61 -m64 -ptx -prec-div=false 04CUDA_Symmetry_BitBoard.cu && POCL_DEBUG=all ./a.out -n ;

 CUDAのリアルタイム監視
 $ watch nvidia-smi

 # 実行
 $ nvcc -O3 -arch=sm_61 -m64 -ptx -prec-div=false 04CUDA_Symmetry_BitBoard.cu && POCL_DEBUG=all ./a.out -n ;


 # 実行結果

対称解除法 GPUビットボード
 N:            Total           Unique      dd:hh:mm:ss.ms
 4:                2                1     000:00:00:00.00
 5:               10                2     000:00:00:00.00
 6:                4                1     000:00:00:00.00
 7:               40                6     000:00:00:00.00
 8:               92               12     000:00:00:00.01
 9:              352               46     000:00:00:00.01
10:              724               92     000:00:00:00.01
11:             2680              341     000:00:00:00.01
12:            14200             1787     000:00:00:00.02
13:            73712             9233     000:00:00:00.04
14:           365596            45752     000:00:00:00.04
15:          2279184           285053     000:00:00:00.04
16:         14772512          1846955     000:00:00:00.07
17:         95815104         11977939     000:00:00:00.26
18:        666090624         83263591     000:00:00:01.65
19:       4968057848        621012754     000:00:00:13.80
20:      39029188884       4878666808     000:00:02:02.52
21:     314666222712      39333324973     000:00:18:46.52
22:    2691008701644     336376244042     000:03:00:22.54
23:   24233937684440    3029242658210     001:06:03:49.29
24:  227514171973736   28439272956934     012:23:38:21.02
25: 2207893435808352  275986683743434     140:07:39:29.96
*
  for(l.BOUND1=2;l.BOUND1<size-1;l.BOUND1++){
  for(l.BOUND1=1,l.BOUND2=size-1-1;l.BOUND1<l.BOUND2;l.BOUND1++,l.BOUND2--){
  のfor文の単位で複数回GPUが実行される
  そのため、BOUND1,BOUND2,TOPBIT,ENDBIT,LASTMASK,SIDEMASKはGPU内では同じ値になる

  STEPS数がGPU 1回でできる最大数なので、STEPSまでノード（GPUに渡す、LEFT,DOWN,RIGHT,board)が溜まったら、）が溜まったら息継ぎできに1回GPUを実行する
 

*/
// #include <iostream>
// #include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
typedef unsigned int uint;
typedef unsigned long ulong;
#define MAX 27
#define THREAD_NUM 96
// システムによって以下のマクロが必要であればコメントを外してください。
//#define UINT64_C(c) c ## ULL
//
// グローバル変数
ulong TOTAL=0;
ulong UNIQUE=0;
//GPU で使うローカル構造体
typedef struct local
{
  uint BOUND1,BOUND2;
  uint TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
  ulong board[MAX];
  ulong COUNT2,COUNT4,COUNT8,TOTAL,UNIQUE;
  uint STEPS;
}local;
//
//
/**
  GPU -g
  */
// __device__
struct dlocal
{
  uint BOUND1,BOUND2;
  uint TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
  ulong board[MAX];
  ulong COUNT2,COUNT4,COUNT8,TOTAL,UNIQUE;
}dlocal;
// __device__ struct dlocal gdl[9999];
struct dlocal gdl[9999];
// GPU 対称解除法
/**
__host__ __device__
long GPU_symmetryOps(uint size,struct dlocal* l)
{
  // ２．クイーンが右上角以外にある場合、
  // (1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルか
  // ら180度回転)させても、さらに90度回転(オリジナルから270度回転)させてもオリ
  // ジナルと同型になる。
  // こちらに該当するユニーク解が属するグループの要素数は、左右反転させたパター
  // ンを加えて２個しかありません。
  if(l->board[l->BOUND2]==1){
    uint ptn;
    uint own;
    for(ptn=2,own=1;own<size;++own,ptn<<=1){
      uint bit;
      uint you;
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
  // ２．クイーンが右上角以外にある場合、
  //   (2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナル
  //   とは異なる。ただし、180度回転させた場合はオリジナルと同型になることも有り得
  //   る。こちらに該当するユニーク解が属するグループの要素数は、180度回転させて同
  //   型になる場合は４個(左右反転×縦横回転)
  //１８０度回転
  if(l->board[size-1]==l->ENDBIT){
    uint you;
    uint own;
    for(you=size-1-1,own=1;own<=size-1;++own,--you){
      uint bit;
      uint ptn;
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
  // ２．クイーンが右上角以外にある場合、
  //   (3)180度回転させてもオリジナルと異なる場合は、８個(左右反転×縦横回転×上下反転)
  //２７０度回転
  if(l->board[l->BOUND1]==l->TOPBIT){
    uint ptn;
    uint own;
    uint you;
    uint bit;
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
*/
// GPU 角にQがないときのバックトラック
/**
__host__ __device__
long GPU_symmetry_backTrack(uint size,uint row,uint left,uint down,uint right,struct dlocal* l)
{
  ulong counter=0;
  uint mask=(1<<size)-1;
  uint bitmap=mask&~(left|down|right);
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
      uint bit=-bitmap&bitmap;
      bitmap=bitmap^bit;
      l->board[row]=bit;
      counter+=GPU_symmetry_backTrack(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
  return counter;
}
*/
// GPU 角にQがあるときのバックトラック
/**
__host__ __device__
long GPU_symmetry_backTrack_corner(uint size,uint row,uint left,uint down,uint right,struct dlocal* l)
{
  ulong counter=0;
  uint mask=(1<<size)-1;
  uint bitmap=mask&~(left|down|right);
  uint bit=0;
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
*/
// GPU 対称解除法 -g の実行時のみ呼び出されます
/**
__host__ __device__
void GPU_symmetry_R(uint size,struct local* hostLocal)
{
  // GPU内部で使うための dlocal構造体
  struct dlocal l;
  l.TOTAL=l.UNIQUE=l.COUNT2=l.COUNT4=l.COUNT8=0;
  uint bit=0;
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
*/
/**
  CUDA13
  */
// GPU -n 対称解除法
/**
__device__
int BitBoard_symmetryOps(uint size,uint* board,uint BOUND1,uint BOUND2,uint TOPBIT,uint ENDBIT)
{
  uint own,ptn,you,bit;
  //90度回転
  if(board[BOUND2]==1){ own=1; ptn=2;
    while(own<=size-1){ bit=1; you=size-1;
      while((board[you]!=ptn)&&(board[own]>=bit)){ bit<<=1; you--; }
      if(board[own]>bit){ return 0; } else if(board[own]<bit){ break; }
      own++; ptn<<=1;
    }
    // 90度回転して同型なら180度/270度回転も同型である
    if(own>size-1){ return 2; }
  }
  //180度回転
  if(board[size-1]==ENDBIT){ own=1; you=size-1-1;
    while(own<=size-1){ bit=1; ptn=TOPBIT;
      while((board[you]!=ptn)&&(board[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(board[own]>bit){ return 0; } else if(board[own]<bit){ break; }
      own++; you--;
    }
    // 90度回転が同型でなくても180度回転が同型である事もある
    if(own>size-1){ return 4; }
  }
  // 270度回転
  if(board[BOUND1]==TOPBIT){ own=1; ptn=TOPBIT>>1;
    while(own<=size-1){ bit=1; you=0;
      while((board[you]!=ptn)&&(board[own]>=bit)){ bit<<=1; you++; }
      if(board[own]>bit){ return 0; } else if(board[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  return 8;
}
*/
// GPU -n Ｑが角にある場合のバックトラック内の再帰処理をカーネルで行う
/**
__global__
void BitBoard_cuda_kernel_b1(const uint size,uint mark,uint* _down,uint* _left,uint* _right,uint* _total,uint* _unique,ulong _cond,uint _row,uint BOUND1)
{
  const uint mask=(1<<size)-1;
  ulong total=0;
  uint unique=0;
  int row=0;
  uint bit;
  //
  //スレッド
  //
  //ブロック内のスレッドID
  const uint tid=threadIdx.x;
  //グリッド内のブロックID
  const uint bid=blockIdx.x;
  //全体通してのID
  const uint idx=bid*blockDim.x+tid;
  //
  //シェアードメモリ
  //
  //sharedメモリを使う ブロック内スレッドで共有
  //10固定なのは現在のmask設定で
  //GPUで実行するのは最大10だから
  //THREAD_NUMはブロックあたりのスレッド数
  __shared__ uint down[THREAD_NUM][10];
  down[tid][row]=_down[idx];
  __shared__ uint left[THREAD_NUM][10];
  left[tid][row]=_left[idx];
  __shared__ uint right[THREAD_NUM][10];
  right[tid][row]=_right[idx];
  __shared__ uint bitmap[THREAD_NUM][10];
  bitmap[tid][row] =mask&~(down[tid][row]|left[tid][row]|right[tid][row]);
  __shared__ uint sum[THREAD_NUM];
  __shared__ uint usum[THREAD_NUM];
  //余分なスレッドは動かさない
  //GPUはSTEPS数起動するが_cond以上は空回しする
  if(idx<_cond){
    //_down,_left,_rightの情報を
    //down,left,rightに詰め直す
    //CPU で詰め込んだ t_はSTEPS個あるが
    //ブロック内ではブロックあたりのスレッド数に限定
    //されるので idxでよい
    //
    uint bitmap_tid_row;
    uint down_tid_row;
    uint left_tid_row;
    uint right_tid_row;
    while(row>=0){
      bitmap_tid_row=bitmap[tid][row];
      down_tid_row=down[tid][row];
      left_tid_row=left[tid][row];
      right_tid_row=right[tid][row];
      if(bitmap_tid_row==0){
        row--;
      }else{
        // 11 枝刈り
        if(row+_row<BOUND1) {
          bitmap_tid_row=bitmap[tid][row]&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
        }
        //クイーンを置く
        //置く場所があるかどうか
        bitmap[tid][row]
          ^=bit
          =(-bitmap_tid_row&bitmap_tid_row);
        if((bit&mask)!=0){
          //最終行?最終行から１個前の行まで
          //無事到達したら 加算する
          if(row+1==mark){
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
            bitmap[tid][rowP]=mask&~(down[tid][rowP]|left[tid][rowP]|right[tid][rowP]);
            row++;
          }
        }else{
          //置く場所がなければ１個上に
          row--;
        }
      }
    }
    // 最後sum[tid]に加算する
    sum[tid]=total;
    usum[tid]=unique;
  }else{
    //_cond未満は空回しするのでtotalは加算しない
    sum[tid]=0;
    usum[tid]=0;
  }
  //__syncthreads()でブロック内のスレッド間の同期
  //全てのスレッドが__syncthreads()に辿り着くのを待つ
  __syncthreads();
  if(tid<64&&tid+64<THREAD_NUM){
    sum[tid]+=sum[tid+64];
    usum[tid]+=usum[tid+64];
  }
  __syncwarp();
  if(tid<32){
    sum[tid]+=sum[tid+32];
    usum[tid]+=usum[tid+32];
  }
  __syncwarp();
  if(tid<16){
    sum[tid]+=sum[tid+16];
    usum[tid]+=usum[tid+16];
  }
  __syncwarp();
  if(tid<8){
    sum[tid]+=sum[tid+8];
    usum[tid]+=usum[tid+8];
  }
  __syncwarp();
  if(tid<4){
    sum[tid]+=sum[tid+4];
    usum[tid]+=usum[tid+4];
  }
  __syncwarp();
  if(tid<2){
    sum[tid]+=sum[tid+2];
    usum[tid]+=usum[tid+2];
  }
  __syncwarp();
  if(tid<1){
    sum[tid]+=sum[tid+1];
    usum[tid]+=usum[tid+1];
  }
  __syncwarp();
  if(tid==0){
    _total[bid]=sum[0];
    _unique[bid]=usum[0];
  }
}
*/
// GPU -n Ｑが角にない場合のバックトラック内の再帰処理をカーネルで行う
/**
__global__
void BitBoard_cuda_kernel_b2(uint size,uint mark,uint* _down,uint* _left,uint* _right,uint* _total,uint* _unique,ulong _cond,uint* board,uint _row,uint BOUND1,uint BOUND2,uint SIDEMASK,uint LASTMASK,uint TOPBIT,uint ENDBIT)
{
  const uint mask=(1<<size)-1;
  ulong total=0;
  uint unique=0;
  int row=0;
  uint bit;
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
  __shared__ uint down[THREAD_NUM][10];
  down[tid][row]=_down[idx];
  __shared__ uint left[THREAD_NUM][10];
  left[tid][row]=_left[idx];
  __shared__ uint right[THREAD_NUM][10];
  right[tid][row]=_right[idx];
  __shared__ uint bitmap[THREAD_NUM][10];
  //down,left,rightからbitmapを出す
  bitmap[tid][row]=mask&~(down[tid][row]|left[tid][row]|right[tid][row]);
  __shared__ uint sum[THREAD_NUM];
  uint c_aBoard[MAX];
  __shared__ uint usum[THREAD_NUM];
  //余分なスレッドは動かさない
  //GPUはSTEPS数起動するが_cond以上は空回しする
  if(idx<_cond){
    //_down,_left,_rightの情報を
    //down,left,rightに詰め直す
    //CPU で詰め込んだ t_はSTEPS個あるが
    //ブロック内ではブロックあたりのスレッド数に限定
    //されるので idxでよい
    //
    for(int i=0;i<_row;i++){
      c_aBoard[i]=board[idx*_row+i]; //２次元配列だが1次元的に利用
    }
    uint bitmap_tid_row;
    uint down_tid_row;
    uint left_tid_row;
    uint right_tid_row;
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
        // 11 枝刈り追加
        //【枝刈り】上部サイド枝刈り
        if(row+_row<BOUND1){
          bitmap_tid_row=bitmap[tid][row]&=~SIDEMASK;
          //【枝刈り】下部サイド枝刈り
        }else if(row+_row==BOUND2) {
          if((down_tid_row&SIDEMASK)==0){
            row--;
            continue;
          }
          if((down_tid_row&SIDEMASK)!=SIDEMASK){
            bitmap_tid_row=bitmap[tid][row]&=SIDEMASK;
          }
        }
        int save_bitmap=bitmap[tid][row];
        //クイーンを置く
        //置く場所があるかどうか
        bitmap[tid][row]^=c_aBoard[row+_row]=bit=(-bitmap_tid_row&bitmap_tid_row);
        if((bit&mask)!=0){
          //最終行?最終行から１個前の行まで
          //無事到達したら 加算する
          if(row+1==mark){
            // 11 LASTMASK枝刈り
            if((save_bitmap&LASTMASK)==0){
              // 12 symmetryOps 省力化のためBOUND1,BOUND2,TOPBIT,ENDBITを渡す
              //int s=BitBoard_symmetryOps(size,c_aBoard,l);
              int s=BitBoard_symmetryOps(size,c_aBoard,BOUND1,BOUND2,TOPBIT,ENDBIT);
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
    //_cond未満は空回しするのでtotalは加算しない
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
    _total[bid]=sum[0];
    _unique[bid]=usum[0];
  }
}
*/
// GPU -n Ｑが角にない
/**
void BitBoard_backTrack2G(const uint size,uint row,uint _left,uint _down,uint _right,struct local* l)
{
  //何行目からGPUで行くか。ここの設定は変更可能、設定値を多くするほどGPUで並行して動く
  // 11 size<8の時はmarkが2
  uint mark=size>12?size-10:3;
  //uint mark=size>11?size-9:3;
  if(size<8){ mark=2; }
  const uint h_mark=row;
  ulong totalCond=0;
  uint mask=(1<<size)-1;
  bool matched=false;
  //host
  uint down[32];  down[row]=_down;
  uint right[32]; right[row]=_right;
  uint left[32];  left[row]=_left;
  //bitmapを配列で持つことにより
  //stackを使わないで1行前に戻れる
  uint bitmap[32];
  bitmap[row]=mask&~(left[row]|down[row]|right[row]);
  uint bit;

  uint* hostDown;
  cudaMallocHost((void**) &hostDown,sizeof(int)*l->STEPS);
  uint* hostLeft;
  cudaMallocHost((void**) &hostLeft,sizeof(int)*l->STEPS);
  uint* hostRight;
  cudaMallocHost((void**) &hostRight,sizeof(int)*l->STEPS);
  uint* deviceDown;
  cudaMalloc((void**) &deviceDown,sizeof(int)*l->STEPS);
  uint* deviceLeft;
  cudaMalloc((void**) &deviceLeft,sizeof(int)*l->STEPS);
  uint* deviceRight;
  cudaMalloc((void**) &deviceRight,sizeof(int)*l->STEPS);
  uint* hostTotal;
  cudaMallocHost((void**) &hostTotal,sizeof(int)*l->STEPS/THREAD_NUM);
  uint* hostUnique;
  cudaMallocHost((void**) &hostUnique,sizeof(int)*l->STEPS/THREAD_NUM);
  uint* deviceTotal;
  cudaMalloc((void**) &deviceTotal,sizeof(int)*l->STEPS/THREAD_NUM);
  uint* deviceUnique;
  cudaMalloc((void**) &deviceUnique,sizeof(int)*l->STEPS/THREAD_NUM);

  uint* hostBoard;
  cudaMallocHost((void**) &hostBoard,sizeof(int)*l->STEPS*mark);
  uint* deviceBoard;
  cudaMalloc((void**) &deviceBoard,sizeof(int)*l->STEPS*mark);

  //12行目までは3行目までCPU->row==mark以下で 3行目までの
  //down,left,right情報をhostDown ,hostLeft,hostRight
  //に格納
  //する->3行目以降をGPUマルチスレッドで実行し結果を取得
  //13行目以降はCPUで実行する行数が１個ずつ増えて行く
  //例えばn15だとrow=5までCPUで実行し、
  //それ以降はGPU(現在の設定だとGPUでは最大10行実行する
  //ようになっている)
  uint rowP=0;
  ulong total=0;
  ulong unique=0;
  while(row>=h_mark) {
    //bitmap[row]=00000000 クイーンを
    //どこにも置けないので1行上に戻る
    //06GPU こっちのほうが優秀
    if(bitmap[row]==0){ row--; }
    else{//おける場所があれば進む
      // 11 枝刈り追加
      //【枝刈り】上部サイド枝刈り
      if(row<l->BOUND1){
        bitmap[row]&=~l->SIDEMASK;
        //【枝刈り】下部サイド枝刈り
      }else if(row==l->BOUND2) {
        if((down[row]&l->SIDEMASK)==0){ row--; }
        if((down[row]&l->SIDEMASK)!=l->SIDEMASK){ bitmap[row]&=l->SIDEMASK; }
      }
      //06SGPU
      bitmap[row]^=l->board[row]=bit=(-bitmap[row]&bitmap[row]);
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
          //hostDown,hostLeft,hostRightに格納する
          hostDown[totalCond]=down[row];
          hostLeft[totalCond]=left[row];
          hostRight[totalCond]=right[row];
          for(int i=0;i<mark;i++){
            hostBoard[totalCond*mark+i]=l->board[i];
          }
          //スレッド数をインクリメントする
          totalCond++;
          //最大GPU数に達してしまったら一旦ここでGPUを実行する。STEPSはGPUの同
          //時並行稼働数を制御
          //nの数が少ないうちはtotalCondがSTEPSを超えることはないがnの数が増え
          //て行くと超えるようになる。
          //ここではtotalCond==STEPSの場合だけこの中へ
          if(totalCond==l->STEPS){
            //matched=trueの時にCOUNT追加 //GPU内でカウントしているので、GPUか
            //ら出たらmatched=trueになってる
            if(matched){
              // デバイスからホストへ転送
              cudaMemcpy(hostTotal, deviceTotal, sizeof(int)*l->STEPS/THREAD_NUM,cudaMemcpyDeviceToHost);
              cudaMemcpy(hostUnique,deviceUnique,sizeof(int)*l->STEPS/THREAD_NUM,cudaMemcpyDeviceToHost);
              for(int col=0;col<l->STEPS/THREAD_NUM;col++){
                total+=hostTotal[col];
                unique+=hostUnique[col];
              }
              matched=false;
            }
            // ホストからデバイスへ転送
            cudaMemcpy(deviceDown, hostDown,sizeof(int)*totalCond, cudaMemcpyHostToDevice);
            cudaMemcpy(deviceLeft, hostLeft,sizeof(int)*totalCond, cudaMemcpyHostToDevice);
            cudaMemcpy(deviceRight,hostRight,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            cudaMemcpy(deviceBoard,hostBoard,sizeof(int)*totalCond*mark,cudaMemcpyHostToDevice);
            // CUDA起動
            BitBoard_cuda_kernel_b2<<<l->STEPS/THREAD_NUM,THREAD_NUM >>>(size,size-mark,deviceDown,deviceLeft,deviceRight,deviceTotal,deviceUnique,totalCond,deviceBoard,row,l->BOUND1,l->BOUND2,l->SIDEMASK,l->LASTMASK,l->TOPBIT,l->ENDBIT);
            //STEPS数の数だけマルチスレッドで起動するのだが、実際に計算が行われ
            //るのはtotalCondの数だけでそれ以外は空回しになる
            //GPU内でカウントしているので、GPUから出たらmatched=trueになってる
            matched=true;
            //totalCond==STEPSルートでGPUを実行したらスレッドをまた0から開始す
            //る(これによりなんどもSTEPS数分だけGPUを起動できる)
            totalCond=0;
          }
          //hostDown,hostLeft,hostRightに情報を格納したら1行上に上がる
          //これを繰り返すことにより row=2で可能な場所全てにクイーンを置いて
          //hostDown,hostLeft,hostRightに情報を格納する
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
    // デバイスからホストへ転送
    cudaMemcpy(hostTotal, deviceTotal, sizeof(int)*l->STEPS/THREAD_NUM,cudaMemcpyDeviceToHost);
    cudaMemcpy(hostUnique,deviceUnique,sizeof(int)*l->STEPS/THREAD_NUM,cudaMemcpyDeviceToHost);
    // 集計
    for(int col=0;col<l->STEPS/THREAD_NUM;col++){
      total+=hostTotal[col];
      unique+=hostUnique[col];
    }
    matched=false;
  }
  // ホストからデバイスへ転送
  cudaMemcpy(deviceDown, hostDown,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceLeft, hostLeft,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceRight,hostRight,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceBoard,hostBoard,sizeof(int)*totalCond*mark,cudaMemcpyHostToDevice);
  //size-mark は何行GPUを実行するか totalCondはスレッド数
  //STEPS数の数だけマルチスレッドで起動するのだが、実際に計算が行われるのは
  //totalCondの数だけでそれ以外は空回しになる
  // CUDA起動
  BitBoard_cuda_kernel_b2<<<l->STEPS/THREAD_NUM,THREAD_NUM >>>(size,size-mark,deviceDown,deviceLeft,deviceRight,deviceTotal,deviceUnique,totalCond,deviceBoard,mark,l->BOUND1,l->BOUND2,l->SIDEMASK,l->LASTMASK,l->TOPBIT,l->ENDBIT);
  // デバイスからホストへ転送
  cudaMemcpy(hostTotal, deviceTotal, sizeof(int)*l->STEPS/THREAD_NUM,cudaMemcpyDeviceToHost);
  cudaMemcpy(hostUnique,deviceUnique,sizeof(int)*l->STEPS/THREAD_NUM,cudaMemcpyDeviceToHost);
  // 集計
  for(int col=0;col<l->STEPS/THREAD_NUM;col++){
    total+=hostTotal[col];
    unique+=hostUnique[col];
  }
  TOTAL+=total;
  UNIQUE+=unique;
  //
  cudaFree(deviceDown);
  cudaFree(deviceLeft);
  cudaFree(deviceRight);
  cudaFree(deviceTotal);
  cudaFree(deviceUnique);
  cudaFree(deviceBoard);
  cudaFreeHost(hostDown);
  cudaFreeHost(hostLeft);
  cudaFreeHost(hostRight);
  cudaFreeHost(hostTotal);
  cudaFreeHost(hostUnique);
  cudaFreeHost(hostBoard);
}
*/
// GPU -n Ｑが角にある
/**
void BitBoard_backTrack1G(const uint size,uint row,uint _left,uint _down,uint _right,struct local* l)
{
  //何行目からGPUで行くか。ここの設定は変更可能、設定値を多くするほどGPUで並行して動く
  // 08 クイーンを２行目まで固定で置くためmarkが3以上必要
  const uint mark=size>12?size-10:3;
  //mark 行までCPU mark行以降はGPU
  const uint h_mark=row;
  const uint mask=(1<<size)-1;
  ulong totalCond=0;
  bool matched=false;
  //host
  uint down[32];  down[row]=_down;
  uint right[32]; right[row]=_right;
  uint left[32];  left[row]=_left;
  //bitmapを配列で持つことにより
  //stackを使わないで1行前に戻れる
  uint bitmap[32];
  bitmap[row]=mask&~(left[row]|down[row]|right[row]);
  uint bit;

  uint* hostDown;
  cudaMallocHost((void**) &hostDown,sizeof(int)*l->STEPS);
  uint* hostLeft;
  cudaMallocHost((void**) &hostLeft,sizeof(int)*l->STEPS);
  uint* hostRight;
  cudaMallocHost((void**) &hostRight,sizeof(int)*l->STEPS);
  uint* deviceDown;
  cudaMalloc((void**) &deviceDown,sizeof(int)*l->STEPS);
  uint* deviceLeft;
  cudaMalloc((void**) &deviceLeft,sizeof(int)*l->STEPS);
  uint* deviceRight;
  cudaMalloc((void**) &deviceRight,sizeof(int)*l->STEPS);

  uint* hostTotal;
  cudaMallocHost((void**) &hostTotal,sizeof(int)*l->STEPS/THREAD_NUM);
  uint* hostUnique;
  cudaMallocHost((void**) &hostUnique,sizeof(int)*l->STEPS/THREAD_NUM);
  uint* deviceTotal;
  cudaMalloc((void**) &deviceTotal,sizeof(int)*l->STEPS/THREAD_NUM);
  uint* deviceUnique;
  cudaMalloc((void**) &deviceUnique,sizeof(int)*l->STEPS/THREAD_NUM);

  //12行目までは3行目までCPU->row==mark以下で 3行目までの
  //down,left,right情報を hostDown,hostLeft,hostRight
  //に格納
  //する->3行目以降をGPUマルチスレッドで実行し結果を取得
  //13行目以降はCPUで実行する行数が１個ずつ増えて行く
  //例えばn15だとrow=5までCPUで実行し、
  //それ以降はGPU(現在の設定だとGPUでは最大10行実行する
  //ようになっている)
  //while(row>=0) {
  int rowP=0;
  ulong total=0;
  ulong unique=0;
  while(row>=h_mark) {
    //bitmap[row]=00000000 クイーンを
    //どこにも置けないので1行上に戻る
    //06GPU こっちのほうが優秀
    if(bitmap[row]==0){ row--; }
    else{//おける場所があれば進む
      if(row<l->BOUND1) { // 11 枝刈り
        bitmap[row]&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
      }
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
          //hostDown,hostLeft,hostRightに格納する
          hostDown[totalCond]=down[row];
          hostLeft[totalCond]=left[row];
          hostRight[totalCond]=right[row];
          //スレッド数をインクリメントする
          totalCond++;
          //最大GPU数に達してしまったら一旦ここでGPUを実行する。STEPSはGPUの同
          //時並行稼働数を制御
          //nの数が少ないうちはtotalCondがSTEPSを超えることはないがnの数が増え
          //て行くと超えるようになる。
          //ここではtotalCond==STEPSの場合だけこの中へ
          if(totalCond==l->STEPS){
            //matched=trueの時にCOUNT追加 //GPU内でカウントしているので、GPUか
            //ら出たらmatched=trueになってる
            if(matched){
              // デバイスからホストへ転送
              cudaMemcpy(hostTotal, deviceTotal, sizeof(int)*l->STEPS/THREAD_NUM,cudaMemcpyDeviceToHost);
              cudaMemcpy(hostUnique,deviceUnique,sizeof(int)*l->STEPS/THREAD_NUM,cudaMemcpyDeviceToHost);
              // 集計
              for(int col=0;col<l->STEPS/THREAD_NUM;col++){
                total+=hostTotal[col];
                unique+=hostUnique[col];
              }
              matched=false;
            }
            // ホストからデバイスへ転送
            cudaMemcpy(deviceDown, hostDown, sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            cudaMemcpy(deviceLeft, hostLeft, sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            cudaMemcpy(deviceRight,hostRight,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            // CUDA起動
            BitBoard_cuda_kernel_b1<<<l->STEPS/THREAD_NUM,THREAD_NUM >>>(size,size-mark,deviceDown,deviceLeft,deviceRight,deviceTotal,deviceUnique,totalCond,row,l->BOUND1);
            //STEPS数の数だけマルチスレッドで起動するのだが、実際に計算が行われ
            //るのはtotalCondの数だけでそれ以外は空回しになる
            //GPU内でカウントしているので、GPUから出たらmatched=trueになってる
            matched=true;
            //totalCond==STEPSルートでGPUを実行したらスレッドをまた0から開始す
            //る(これによりなんどもSTEPS数分だけGPUを起動できる)
            totalCond=0;
          }
          //hostDown,hostLeft,hostRightに情報を格納したら1行上に上がる
          //これを繰り返すことにより row=2で可能な場所全てにクイーンを置いて
          //hostDown,hostLeft,hostRightに情報を格納する
          row--;
        }
      }else{
        //置く場所がなければ上に上がる。row==mark行に達するまではCPU側で普通に
        //nqueenをやる
        row--;
      }
    }
  }
  //if(totalCond==l->STEPS){で処理されなかった残りがここで実行される
  //matched=trueの時にCOUNT追加 //GPU内でカウントしているので、GPUから出たら
  //matched=trueになってる
  //
  if(matched){
    // デバイスからホストへ転送
    cudaMemcpy(hostTotal, deviceTotal, sizeof(int)*l->STEPS/THREAD_NUM,cudaMemcpyDeviceToHost);
    cudaMemcpy(hostUnique,deviceUnique,sizeof(int)*l->STEPS/THREAD_NUM,cudaMemcpyDeviceToHost);
    // 集計
    for(int col=0;col<l->STEPS/THREAD_NUM;col++){
      total+=hostTotal[col];
      unique+=hostUnique[col];
    }
    matched=false;
  }
  // ホストからデバイスへ転送
  cudaMemcpy(deviceDown, hostDown, sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceLeft, hostLeft, sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceRight,hostRight,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  // CUDA起動
  BitBoard_cuda_kernel_b1<<<l->STEPS/THREAD_NUM,THREAD_NUM >>>(size,size-mark,deviceDown,deviceLeft,deviceRight,deviceTotal,deviceUnique,totalCond,mark,l->BOUND1);
  // デバイスからホストへ転送
  cudaMemcpy(hostTotal, deviceTotal, sizeof(int)*l->STEPS/THREAD_NUM,cudaMemcpyDeviceToHost);
  cudaMemcpy(hostUnique,deviceUnique,sizeof(int)*l->STEPS/THREAD_NUM,cudaMemcpyDeviceToHost);
  // 集計
  for(int col=0;col<l->STEPS/THREAD_NUM;col++){
    total+=hostTotal[col];
    unique+=hostUnique[col];
  }
  TOTAL+=total;
  UNIQUE+=unique;
  //開放
  cudaFree(deviceDown);
  cudaFree(deviceLeft);
  cudaFree(deviceRight);
  cudaFree(deviceTotal);
  cudaFree(deviceUnique);
  cudaFreeHost(hostDown);
  cudaFreeHost(hostLeft);
  cudaFreeHost(hostRight);
  cudaFreeHost(hostTotal);
  cudaFreeHost(hostUnique);
}
*/
// GPU -n ビットボードの実行 角にＱがある・ないの分岐を行う
/**
void BitBoard_build(const uint size,int STEPS)
{
  if(size<=0||size>32){return;}
    // int型は unsigned とする
    // total: グローバル変数TOTALへのアクセスを極小化する
  struct local l; //GPU で扱う構造体
  l.STEPS=STEPS;
  uint bit=1;
  l.board[0]=1;
  uint left=bit<<1,down=bit,right=bit>>1;
    // 2行目は右から3列目から左端から2列目まで
  for(l.BOUND1=2;l.BOUND1<size-1;l.BOUND1++){
    l.board[1]=bit=(1<<l.BOUND1);
    printf("\rBOUND1(%d/%d)",l.BOUND1,size-1);// << std::flush;
    printf("\r");
    fflush(stdout);
    BitBoard_backTrack1G(size,2,(left|bit)<<1,(down|bit),(right|bit)>>1,&l);
  }
  l.TOPBIT=1<<(size-1);
  l.SIDEMASK=l.LASTMASK=(l.TOPBIT|1);
  l.ENDBIT=(l.TOPBIT>>1);
  // 1行目右から2列目から
  // 偶数個は1/2 n=8 なら 1,2,3 奇数個は1/2+1 n=9 なら 1,2,3,4
  for(l.BOUND1=1,l.BOUND2=size-1-1;l.BOUND1<l.BOUND2;l.BOUND1++,l.BOUND2--){
    printf("\r  BOUND2(%d/%d)",l.BOUND1,size/2-1);// << std::flush;
    printf("\r");
    fflush(stdout);
    l.board[0]=bit=(1<<l.BOUND1);
    BitBoard_backTrack2G(size,1,bit<<1,bit,bit>>1,&l);
    l.LASTMASK|=l.LASTMASK>>1|l.LASTMASK<<1;
    l.ENDBIT>>=1;
  }
}
*/
// CUDA 初期化
/**
bool InitCUDA()
{
  int count;
  cudaGetDeviceCount(&count);
  if(count==0){fprintf(stderr,"There is no device.\n");return false;}
  uint i;
  for(i=0;i<count;++i){
    struct cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop,i)==cudaSuccess){if(prop.major>=1){break;} }
  }
  if(i==count){fprintf(stderr,"There is no device supporting CUDA 1.x.\n");return false;}
  cudaSetDevice(i);
  return true;
}
*/
//
// CPU 再帰/非再帰共通 対称解除法
void symmetryOps(uint size,struct local* l)
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
    uint ptn;
    uint own;
    for(ptn=2,own=1;own<size;++own,ptn<<=1){
      uint bit;
      uint you;
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
    uint you;
    uint own;
    for(you=size-1-1,own=1;own<=size-1;++own,--you){
      uint bit;
      uint ptn;
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
    uint ptn;
    uint own;
    uint you;
    uint bit;
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
void symmetry_backTrack_NR(uint size,uint row,uint _left,uint _down,uint _right,struct local *l)
{
  uint mask=(1<<size)-1;
  uint down[size];
  uint left[size];
  uint right[size];
  uint bitmap[size];
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
      uint save_bitmap=bitmap[row];
      uint bit=-bitmap[row]&bitmap[row];
      bitmap[row]^=bit;
      l->board[row]=bit; //Qを配置
      if((bit&mask)!=0){
        if(row==(size-1)){
          if( (save_bitmap&l->LASTMASK)==0){
            symmetryOps(size,l);  //対称解除法
          }
          row--;
        }else{
          uint n=row++;
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
void symmetry_backTrack_corner_NR(uint size,uint row,uint _left,uint _down,uint _right,struct local *l)
{
  uint mask=(1<<size)-1;
  uint bit=0;
  uint down[size];
  uint left[size];
  uint right[size];
  uint bitmap[size];
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
        uint n=row++;
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
void symmetry_NR(uint size,struct local* l)
{
  l->TOTAL=l->UNIQUE=l->COUNT2=l->COUNT4=l->COUNT8=0;
  uint bit=0;
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
void symmetry_backTrack(uint size,uint row,uint left,uint down,uint right,struct local* l)
{
  uint mask=(1<<size)-1;
  uint bitmap=mask&~(left|down|right);
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
      uint bit=-bitmap&bitmap;
      bitmap=bitmap^bit;
      l->board[row]=bit;
      symmetry_backTrack(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
}
// 再帰 角にQがあるときのバックトラック
void symmetry_backTrack_corner(uint size,uint row,uint left,uint down,uint right,struct local* l)
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
      symmetry_backTrack_corner(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
}
// 再帰 対称解除法
void symmetry_R(uint size,struct local* l)
{
  l->TOTAL=l->UNIQUE=l->COUNT2=l->COUNT4=l->COUNT8=0;
  uint bit=0;
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
//メイン
int main(int argc,char** argv)
{
  bool cpu=false,cpur=false,gpu=false,gpuBitBoard=false;
  uint argstart=2;
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
    printf("Usage: %s [-c|-g|-r|-s] n STEPS\n",argv[0]);
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
    uint min=4; 
    uint targetN=17;
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    for(uint size=min;size<=targetN;size++){
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
      uint ss;
      uint ms;
      uint dd;
      if(t1.tv_usec<t0.tv_usec) {
        dd=(t1.tv_sec-t0.tv_sec-1)/86400;
        ss=(t1.tv_sec-t0.tv_sec-1)%86400;
        ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
      }else {
        dd=(t1.tv_sec-t0.tv_sec)/86400;
        ss=(t1.tv_sec-t0.tv_sec)%86400;
        ms=(t1.tv_usec-t0.tv_usec+500)/10000;
      }//end if
      uint hh=ss/3600;
      uint mm=(ss-hh*3600)/60;
      ss%=60;
      printf("%2d:%13ld%12ld%8.2d:%02d:%02d:%02d.%02d\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
    } //end for
  }//end if
  /**
  if(gpu||gpuBitBoard)
  {
    int STEPS=24576;
    if(!InitCUDA()){return 0;}
    uint min=4;
    uint targetN=25;
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:            Total          Unique      dd:hh:mm:ss.ms");
    for(uint size=min;size<=targetN;size++){
      gettimeofday(&t0,NULL);
      if(gpu){
        TOTAL=UNIQUE=0;
        local l[MAX];
        GPU_symmetry_R(size,&l[0]);
        TOTAL=l->TOTAL;
        UNIQUE=l->UNIQUE;
      }else if(gpuBitBoard){
        TOTAL=UNIQUE=0;
        BitBoard_build(size,STEPS);
      }
      gettimeofday(&t1,NULL);
      uint ss;
      uint ms;
      uint dd;
      if (t1.tv_usec<t0.tv_usec) {
        dd=(int)(t1.tv_sec-t0.tv_sec-1)/86400;
        ss=(t1.tv_sec-t0.tv_sec-1)%86400;
        ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
      } else {
        dd=(int)(t1.tv_sec-t0.tv_sec)/86400;
        ss=(t1.tv_sec-t0.tv_sec)%86400;
        ms=(t1.tv_usec-t0.tv_usec+500)/10000;
      }//end if
      uint hh=ss/3600;
      uint mm=(ss-hh*3600)/60;
      ss%=60;
      printf("%2d:%17ld%16ld%8.3d:%02d:%02d:%02d.%02d\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
    }
  }
  */
  return 0;
}
