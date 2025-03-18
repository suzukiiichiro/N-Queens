/**
 *
 * bash版ミラーのC言語版のGPU/CUDA移植版
 *
 詳しい説明はこちらをどうぞ
 https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題
 *
NQueens_suzuki$ nvcc -O3 -arch=sm_61 03CUDA_Nodelayer_Mirror.cu && ./a.out -g
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2               0  00:00:00:00.17
 5:           10               0  00:00:00:00.00
 6:            4               0  00:00:00:00.00
 7:           40               0  00:00:00:00.00
 8:           92               0  00:00:00:00.00
 9:          352               0  00:00:00:00.00
10:          724               0  00:00:00:00.00
11:         2680               0  00:00:00:00.00
12:        14200               0  00:00:00:00.00
13:        73712               0  00:00:00:00.00
14:       365596               0  00:00:00:00.03
15:      2279184               0  00:00:00:00.18
16:     14772512               0  00:00:00:01.59
17:     95815104               0  00:00:00:15.02
18:    666090624               0  00:00:02:39.07


・std::vector<long> kLayer_nodeLayer(int size,int k)
ここでミラー処理をしている。

  for(unsigned int i=0;i<size/2;++i){
  は偶数、奇数かかわらず半分だけ実行する
  n4なら0,1 n5なら 0,1  n6なら 0,1,2 n7なら 0,1,2

  if(size%2){                 
  は奇数の場合だけ実行する
  n5なら 2  n7 なら 3
  この場合2行目は、半分だけ実行する

・void mirror_nodeLayer(int size)
  int numSolutions=nodes.size() / 3;
  left,down,rightで1セットなので/3 kLayer_nodeLayerで半分だけ実行しているのでここは/3のまま

  for (long i=0;i < numSolutions;i++) {
      solutions += 2*hostSolutions[i];// Symmetry
  }
  で最後にGPUの結果を2倍にする
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
#define THREAD_NUM		96
#define MAX 27
/**
  * システムによって以下のマクロが必要であればコメントを外してください。
  */
//#define UINT64_C(c) c ## ULL
typedef unsigned int uint;
typedef unsigned long ulong;
/**
  * ミラー処理部分
  */
__device__ ulong mirror_solve_nodeLayer(uint size,ulong left,ulong down,ulong right)
{
  uint mask=(1<<size)-1;
  uint bit=0;
  ulong counter=0;
  if(down==mask){
    return 1;
  }else{
    for(uint bitmap=mask&~(left|down|right);bitmap;bitmap=bitmap&~bit){
      bit=-bitmap&bitmap;
      counter+=mirror_solve_nodeLayer(size,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
  return counter;
}
/**
  * クイーンの効きを判定して解を返す
  */
__device__ ulong mirror_nodeLayer(uint size,ulong left,ulong down,ulong right)
{
  ulong counter=0;
  uint bit=0;
  uint limit=size%2 ? size/2-1 : size/2;
  for(uint i=0;i<size/2;++i){
    bit=1<<i;
    counter+=mirror_solve_nodeLayer(size,bit<<1,bit,bit>>1);
  }
 if(size%2){   //奇数で通過
    bit=1<<(size-1)/2;
    uint left=bit<<1;
    uint down=bit;
    uint right=bit>>1;
    for(uint i=0;i<limit;++i){
      bit=1<<i;
      counter+=mirror_solve_nodeLayer(size,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
  return counter<<1;// 倍にする
}
/** 
  * i番目のメンバを i 番目の部分木の解で埋める
  */
__global__ void dim_nodeLayer(uint size,ulong* nodes,ulong* solutions,uint numElements)
{
  int i=blockDim.x * blockIdx.x+threadIdx.x;
  if(i<numElements){
    // ミラーのGPUスレッド(-n)の場合は、予めCPU側で奇数と偶数で分岐
    // させるので、奇数と偶数を条件分岐するmirror_nodeLayer()を通過
    // させる必要がない
    solutions[i]=mirror_solve_nodeLayer(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2]);
  }
}
/**
  * 0以外のbitをカウント
  */
int countBits(long n)
{
  int count=0;
  while (n){
    n&=(n-1);// 右端のゼロ以外の数字を削除
    count++;
  }
  return count;
}
/**
  * ノードをk番目のレイヤーのノードで埋める
  */
void kLayer_backTrack(uint size,std::vector<ulong> &nodes,uint layer,uint left,uint down,uint right)
{
  uint mask=(1<<size)-1;
  // すべてのdownが埋まったら、解決策を見つけたことになる。
  if (countBits(down)==layer){
    nodes.push_back(left);
    nodes.push_back(down);
    nodes.push_back(right);
    return;
  }
  uint bit=0;
  // 解を加えて対角線をずらす
  for(uint bitmap=mask&~(left|down|right);bitmap;bitmap^=bit){
    bit=-bitmap&bitmap;
    kLayer_backTrack(size,nodes,layer,(left|bit)<<1,down|bit,(right|bit)>>1);
  }
}
/**
  * k番目のレイヤのすべてのノードを含むベクトルを返す。
  */
void kLayer_nodeLayer(uint size,std::vector<ulong> &nodes,uint layer)
{
  uint bit=0;
  uint limit=size%2 ? size/2-1 : size/2;
  for(uint i=0;i<size/2;++i){
    bit=1<<i;
    kLayer_backTrack(size,nodes,layer,bit<<1,bit,bit>>1);
  }
  if(size%2){               //奇数で通過
    bit=1<<(size-1)/2;
    uint left=bit<<1;
    uint down=bit;
    uint right=bit>>1;
    for(uint i=0;i<limit;++i){
      bit=1<<i;
      kLayer_backTrack(size,nodes,layer,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
}
/**
  * ノードレイヤーの作成
  */
ulong mirror_nodeLayer(uint size)
{
  // ツリーの3番目のレイヤーにあるノード
  //（それぞれ連続する3つの数字でエンコードされる）のベクトル。
  // レイヤー2以降はノードの数が均等なので、対称性を利用できる。
  // レイヤ4には十分なノードがある（N16の場合、9844）。
  int layer=4;
  std::vector<ulong> nodes;
  kLayer_nodeLayer(size,nodes,layer);
  // デバイスにはクラスがないので、
  // 最初の要素を指定してからデバイスにコピーする。
  size_t nodeSize=nodes.size() * sizeof(ulong);
  ulong* hostNodes=(ulong*)malloc(nodeSize);
  hostNodes=&nodes[0];
  ulong* deviceNodes=NULL;
  cudaMalloc((void**)&deviceNodes,nodeSize);
  cudaMemcpy(deviceNodes,hostNodes,nodeSize,cudaMemcpyHostToDevice);
  // デバイス出力の割り当て
  ulong* deviceSolutions=NULL;
  /** ミラーでは/6 を /3に変更する */
  // 必要なのはノードの半分だけで
  // 各ノードは3つの整数で符号化される。
  //int numSolutions=nodes.size() / 6;
  int numSolutions=nodes.size() / 3;
  size_t solutionSize=numSolutions * sizeof(ulong);
  cudaMalloc((void**)&deviceSolutions,solutionSize);
  // CUDAカーネルを起動する。
  int threadsPerBlock=256;
  int blocksPerGrid=(numSolutions+threadsPerBlock-1)/threadsPerBlock;
  dim_nodeLayer<<<blocksPerGrid,threadsPerBlock>>>(size,deviceNodes,deviceSolutions,numSolutions);
  // 結果をホストにコピー
  ulong* hostSolutions=(ulong*)malloc(solutionSize);
  cudaMemcpy(hostSolutions,deviceSolutions,solutionSize,cudaMemcpyDeviceToHost);
  // 部分解を加算し、結果を表示する。
  ulong solutions=0;
  for(ulong i=0;i<numSolutions;i++){solutions+=2*hostSolutions[i]; }
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
  int i;
  for(i=0;i<count;i++){
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
  /* int steps=24576;*/
  int min=4;
  int targetN=21;
  struct timeval t0;
  struct timeval t1;
  printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
  for(int size=min;size<=targetN;size++){
    gettimeofday(&t0,NULL);  // 計測開始
    ulong TOTAL=0;
    ulong UNIQUE=0;
    TOTAL=mirror_nodeLayer(size);// ミラー
    gettimeofday(&t1,NULL);  // 計測終了
    int ss;int ms;int dd;
    if (t1.tv_usec<t0.tv_usec) {
      dd=(int)(t1.tv_sec-t0.tv_sec-1)/86400;
      ss=(t1.tv_sec-t0.tv_sec-1)%86400;
      ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
    } else {
      dd=(int)(t1.tv_sec-t0.tv_sec)/86400;
      ss=(t1.tv_sec-t0.tv_sec)%86400;
      ms=(t1.tv_usec-t0.tv_usec+500)/10000;
    }
    int hh=ss/3600;
    int mm=(ss-hh*3600)/60;
    ss%=60;
    printf("%2d:%17ld%16ld%8.3d:%02d:%02d:%02d.%02d\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
  }
  return 0;
}
