/**
 *
 * bash版対称解除法のC言語版のGPU/CUDA移植版
 *
 詳しい説明はこちらをどうぞ
 https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題
 *

対称解除法 GPUノードレイヤー
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            0           0      00:00:00:00.17
 5:            0           0      00:00:00:00.00
 6:            4           0      00:00:00:00.00
 7:           40           0      00:00:00:00.00
 8:           92           0      00:00:00:00.00
 9:          352           0      00:00:00:00.00
10:          724           0      00:00:00:00.00
11:         2680           0      00:00:00:00.00
12:        14200           0      00:00:00:00.00
13:        73712           0      00:00:00:00.00
14:       365596           0      00:00:00:00.06
15:      2279184           0      00:00:00:00.53
16:     14772512           0      00:00:00:04.40
17:     95815104           0      00:00:00:33.00
18:  20056878547           0      00:00:00:17.85

18以降はバースト

コメント
・std::vector<long> kLayer(unsigned int size,unsigned int k,std::vector<local>& L)
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
typedef unsigned int uint;
typedef unsigned long ulong;

using std::cout; using std::endl;
using std::vector; using std::string;
/**
  * システムによって以下のマクロが必要であればコメントを外してください。
  */
//#define UINT64_C(c) c ## ULL
/**
  *
  */
ulong TOTAL=0;
ulong UNIQUE=0;
/**
  *
  */
typedef struct local
{
  uint BOUND1,BOUND2;
  uint TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
  ulong board[MAX];
  ulong COUNT2,COUNT4,COUNT8,TOTAL,UNIQUE;
  ulong TYPE;
}local;
/**
  * 対称解除法
  */
__device__ long symmetry(uint size,struct local* l)
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
    uint bit;
    uint you;
    for(ptn=2,own=1;own<size;++own,ptn<<=1){
      for(bit=1,you=size-1;(l->board[you]!=ptn)&& l->board[own]>=bit;--you){
        bit<<=1;
      }
      if(l->board[own]>bit){ return 0; }
      if(l->board[own]<bit){ break; }
    }
    // ９０度回転して同型なら１８０度回転しても２７０度回転しても同型である
    if(own>size-1){ l->COUNT2++; return 2; }
  }
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
    uint bit;
    uint ptn;
    for(you=size-1-1,own=1;own<=size-1;++own,--you){
      for(bit=1,ptn=l->TOPBIT;(ptn!=l->board[you])&&(l->board[own]>=bit);ptn>>=1){
        bit<<=1;
      }
      if(l->board[own]>bit){ return 0; }
      if(l->board[own]<bit){ break; }
    }
    //９０度回転が同型でなくても１８０度回転が同型であることもある
    if(own>size-1){ l->COUNT4++; return 4; }
  }
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
      if(l->board[own]>bit){ return 0; }
      if(l->board[own]<bit){ break; }
    }
  }
  l->COUNT8++;
  return 8;
}
/**
  * 0以外のbitをカウント
  */
__host__ __device__ uint countBits(ulong n)
{
  uint counter=0;
  while(n){ n&=(n-1); counter++; }
  return counter;
}
/**
  * ノードレイヤーによる対称解除法
  */
__device__ long nodeLayer_backTrackCorner(uint size,ulong left,ulong down,ulong right,struct local* l)
{
  ulong counter = 0;
  ulong mask=(1<<size)-1;
  ulong bitmap=mask&~(left|down|right);
  ulong bit=0;
  uint row=countBits(down);
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
      counter+=nodeLayer_backTrackCorner(size,(left|bit)<<1,(down|bit),(right|bit)>>1,l);
    }
  }
  return counter;
}
/**
  * ノードレイヤーによる対称解除法 -n の実行時に呼び出される
  */
__device__ long nodeLayer_backTrack(uint size,ulong left,ulong down,ulong right,struct local* l)
{
  ulong counter = 0;
  ulong mask=(1<<size)-1;
  ulong bitmap=mask&~(left|down|right);
  uint row=countBits(down);
  if(row==(size-1)){
    if(bitmap){
      if( (bitmap& l->LASTMASK)==0){
        l->board[row]=bitmap;  //Qを配置
        return symmetry(size,l);    //対称解除;
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
      ulong bit=-bitmap&bitmap;
      bitmap=bitmap^bit;
      l->board[row]=bit;
      counter+=nodeLayer_backTrack(size,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
  return counter;
}
/** 
  * ノードレイヤー i 番目のメンバを i 番目の部分木の解で埋める
  */
__global__ void nodeLayer(uint size,long* nodes,long* solutions,uint numElements,struct local* l)
{
  uint i=blockDim.x*blockIdx.x+threadIdx.x;
  if(i<numElements){
    if(l[i].TYPE==0){
      solutions[i]=nodeLayer_backTrackCorner(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2],&l[i]);
    }else{
      solutions[i]=nodeLayer_backTrack(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2],&l[i]);
    }
  }
}
/**
  * Ｋレイヤー 再帰 角にQがないときのバックトラック
  */
ulong kLayer_backTrack(int size,std::vector<long>& nodes,uint k,ulong left,ulong down,ulong right,std::vector<local>& L,struct local* l)
{
  ulong counter=0;
  ulong mask=(1<<size)-1;
  ulong bitmap=mask&~(left|down|right);
  uint row=countBits(down);
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
      ulong bit=-bitmap&bitmap;
      bitmap=bitmap^bit;
      l->board[row]=bit;
      counter+=kLayer_backTrack(size,nodes,k,(left|bit)<<1,(down|bit),(right|bit)>>1,L,l); 
    }
  }
  return counter;
}
/** Ｋレイヤー 角にQがあるときのバックトラック
  *
  */
ulong kLayer_backTrackCorner(uint size,std::vector<long>& nodes,uint k,ulong left,ulong down,ulong right,std::vector<local>& L,struct local* l)
{
  ulong counter=0;
  ulong mask=(1<<size)-1;
  ulong bitmap=mask&~(left|down|right);
  ulong bit=0;
  int row=countBits(down);
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
      counter+=kLayer_backTrackCorner(size,nodes,k,(left|bit)<<1,(down|bit),(right|bit)>>1,L,l); 
    }
  return counter;
}
/**
  * Ｋレイヤー k 番目のレイヤのすべてのノードを含むベクトルを返す
  */
std::vector<long> kLayer(uint size,uint k,std::vector<local>& L)
{
  std::vector<long> nodes{};
  uint bit=0;
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
      kLayer_backTrackCorner(size,nodes,k,(2|bit)<<1,1|bit,(2|bit)>>1,L,&l);
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
      kLayer_backTrack(size,nodes,k,bit<<1,bit,bit>>1,L,&l);
    }
    l.BOUND1++;
    l.BOUND2--;
    l.ENDBIT=l.ENDBIT>>1;
    l.LASTMASK=l.LASTMASK<<1|l.LASTMASK|l.LASTMASK>>1;
  }
  return nodes;
}
/**
  * ノードレイヤーの作成
  */
uint build_nodelayer(uint size)
{
  // ツリーの3番目のレイヤーにあるノード
  //（それぞれ連続する3つの数字でエンコードされる）のベクトル。
  // レイヤー2以降はノードの数が均等なので対称性を利用できる。
  // レイヤ4には十分なノードがある（N16の場合、9844）。
  // ここではレイヤーを５に設定、Ｎに併せて増やしていく
  std::vector<local>L;
  // NodeLayerは、N18でabortします。
  std::vector<long> nodes=kLayer(size,5,L); 
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
  nodeLayer <<<blocksPerGrid,threadsPerBlock>>>(size,deviceNodes,deviceSolutions,numSolutions,deviceLocal);
  // 結果をホストにコピー
  long* hostSolutions=(long*)malloc(solutionSize);
  cudaMemcpy(hostSolutions,deviceSolutions,solutionSize,cudaMemcpyDeviceToHost);
  // 部分解を加算し、結果を表示する。
  ulong solutions=0;
  for(ulong i=0;i<numSolutions;i++){ solutions += hostSolutions[i]; }
  // 出力
  return solutions;
}
/**
  * CUDA 初期化
  */
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
/**
  * メイン
  */
int main(int argc,char** argv)
{
  if(!InitCUDA()){return 0;}
  /* int steps=24576; */
  uint min=4;
  uint targetN=18;
  struct timeval t0;
  struct timeval t1;
  printf("%s\n"," N:            Total          Unique      dd:hh:mm:ss.ms");
  for(uint size=min;size<=targetN;size++){
    gettimeofday(&t0,NULL);
    TOTAL=UNIQUE=0;
    TOTAL=build_nodelayer(size);
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
  return 0;
}
