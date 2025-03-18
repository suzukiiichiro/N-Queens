/**
 *
 * bash版ビットマップのC言語版のGPU/CUDA移植版
 *
 詳しい説明はこちらをどうぞ
 https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題
 *
NQueens_suzuki$ nvcc -O3 -arch=sm_61  02CUDA_Nodelayer_bitmap.cu && POCL_DEBUG=all ./a.out
 N:            Total          Unique      dd:hh:mm:ss.ms
 4:                2               0     000:00:00:00.13
 5:               10               0     000:00:00:00.00
 6:                4               0     000:00:00:00.00
 7:               40               0     000:00:00:00.00
 8:               92               0     000:00:00:00.00
 9:              352               0     000:00:00:00.00
10:              724               0     000:00:00:00.00
11:             2680               0     000:00:00:00.00
12:            14200               0     000:00:00:00.00
13:            73712               0     000:00:00:00.00
14:           365596               0     000:00:00:00.03
15:          2279184               0     000:00:00:00.14
16:         14772512               0     000:00:00:00.91
17:         95815104               0     000:00:00:08.36
18:        666090624               0     000:00:01:25.29
19:       4968057848               0     000:00:15:31.61


$ nvcc -O3 -arch=sm_61 -m64 -ptx -prec-div=false 01CUDA_Symmetry.cu && POCL_DEBUG=all ./a.out ;
 N:        Total      Unique      dd:hh:mm:ss.ms
16:         14772512          1846955     000:00:00:00.07
17:         95815104         11977939     000:00:00:00.26
18:        666090624         83263591     000:00:00:01.65



nodeLayerはNが増えるとどんどん遅くなる。
結論として01CUDA_bitmap.cu のほうがメモリ効率が高く高速

・kLayer_nodeLayer 
GPUで並列実行するためのleft,right,downを作成する
kLayer_nodeLayer(size,4)
第2引数の4は4行目までnqueenを実行し、それまでのleft,down,rightをnodes配列に格納する

nodesはベクター配列で構造体でもなんでも格納できる
push_backで追加。
nodes配列は3個で１セットleft,dwon,rightの情報を同じ配列に格納する
[0]left[1]down[2]right

・bitmap_build_nodeLayer
  int numSolutions=nodes.size() / 6; 
  3個で1セットなので/3 さらにnodesの2分の1だけ実行すればミラーになるので/6

  
  solutions += 2*hostSolutions[i]; // Symmetry
  GPUごとのTOTALを集計している。ミラー分最後に2倍する

・dim_nodeLayer 
GPU並列処理
bitmap_solve_nodeLayerを再帰呼び出しし、counter(最終行までクイーンを置けると+1)をsolutionsに格納する
solutionsは配列でGPUのステップ数分ある

・bitmap_solve_ndoeLayer
down==maskが最終行までクイーンを置けた状態
ビットだとクイーンを置けない場所に1が立つ
downだとクイーンを置いた場所に1が立つ

maskは、size分1が立っているもの
n8だと11111111

downはクイーンが配置されるたびに配置された列に1が立って行くので最終行までクイーンを置くと全列に1が立った状態になりmaskと同じ内容になる
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
  * グローバル変数
  */
ulong TOTAL=0; 
ulong UNIQUE=0;
/**
  * クイーンの効きを判定して解を返す
  */
__device__ uint solve_nodeLayer(const uint size,uint left,uint down,uint right)
{
  uint mask=(1<<size)-1;
  uint count=0;
  // downがすべて専有され解が見つかる
  if (down==mask) { return 1; }
  uint bit=0;
  for(uint bitmap=mask&~(left|down|right);bitmap;bitmap^=bit){
    bit=-bitmap&bitmap;
    count+=solve_nodeLayer(size,(left|bit)>>1,down|bit,(right|bit)<<1); 
  }
  return count;
}
/**
  * i 番目のメンバを i 番目の部分木の解で埋める
  */
__global__ void dim_nodeLayer(const uint size,ulong* nodes,ulong* solutions,uint numElements)
{
  uint i=blockDim.x * blockIdx.x + threadIdx.x;
  if(i<numElements){ solutions[i]=solve_nodeLayer(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2]); }
}
/** 
  * 0以外のbitをカウント
  */
uint countBits_nodeLayer(uint n)
{
  uint count=0;
  while(n){
    n&=(n-1); // 右端のゼロ以外の数字を削除
    count++;
  }
  return count;
}
/**
  * ノードをk番目のレイヤーのノードで埋める
  */
void kLayer_nodeLayer(const uint size,std::vector<ulong> &nodes,uint k,uint left,uint down,uint right)
{
  uint mask=(1<<size)-1;
  // すべてのdownが埋まったら、解決策を見つけたことになる。
  if (countBits_nodeLayer(down)==k){
    nodes.push_back(left);
    nodes.push_back(down);
    nodes.push_back(right);
    return ;
  }
  uint bit=0;
  for(uint bitmap=mask&~(left|down|right);bitmap;bitmap^=bit){
    bit=-bitmap&bitmap;
    // 解を加えて対角線をずらす
    kLayer_nodeLayer(size,nodes,k,(left|bit)>>1,down|bit,(right|bit)<<1); 
  }
}
/**
  * ノードレイヤーの作成
  */
ulong bitmap_build_nodeLayer(const uint size)
{
  // ツリーの3番目のレイヤーにあるノード
  //（それぞれ連続する3つの数字でエンコードされる）のベクトル。
  // レイヤー2以降はノードの数が均等なので、対称性を利用できる。
  // レイヤ4には十分なノードがある（N16の場合、9844）。
  int layer=4;
  std::vector<ulong> nodes;
  kLayer_nodeLayer(size,nodes,layer,0,0,0); 
  // デバイスにはクラスがないので、
  // 最初の要素を指定してからデバイスにコピーする。
  size_t nodeSize=nodes.size() * sizeof(ulong);
  ulong* hostNodes=(ulong*)malloc(nodeSize);
  hostNodes=&nodes[0];
  ulong* deviceNodes=NULL;
  cudaMalloc((void**)&deviceNodes, nodeSize);
  cudaMemcpy(deviceNodes, hostNodes, nodeSize, cudaMemcpyHostToDevice);
  // デバイス出力の割り当て
  ulong* deviceSolutions=NULL;
  // 必要なのはノードの半分だけで、各ノードは3つの整数で符号化される。
  uint numSolutions=nodes.size() / 6; 
  size_t solutionSize=numSolutions * sizeof(ulong);
  cudaMalloc((void**)&deviceSolutions, solutionSize);
  // CUDAカーネルを起動する。
  uint threadsPerBlock=256;
  uint blocksPerGrid=(numSolutions + threadsPerBlock - 1) / threadsPerBlock;
  dim_nodeLayer <<<blocksPerGrid, threadsPerBlock >>> (size,deviceNodes, deviceSolutions, numSolutions);
  // 結果をホストにコピー
  ulong* hostSolutions=(ulong*)malloc(solutionSize);
  cudaMemcpy(hostSolutions, deviceSolutions, solutionSize, cudaMemcpyDeviceToHost);
  // 部分解を加算し、結果を表示する。
  ulong solutions=0;
  for (uint i=0;i<numSolutions;i++) { solutions+=2*hostSolutions[i]; }
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
  /* int steps=24576; */
  int min=4;
  int targetN=19;
  struct timeval t0;
  struct timeval t1;
  printf("%s\n"," N:            Total          Unique      dd:hh:mm:ss.ms");
  for(uint size=min;size<=targetN;size++){
    gettimeofday(&t0,NULL);   // 計測開始
    TOTAL=UNIQUE=0;
    TOTAL=bitmap_build_nodeLayer(size); // ビットマップ
    gettimeofday(&t1,NULL);   // 計測終了
    int ss;int ms;int dd;
    if (t1.tv_usec<t0.tv_usec) {
      dd=(int)(t1.tv_sec-t0.tv_sec-1)/86400;
      ss=(t1.tv_sec-t0.tv_sec-1)%86400;
      ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
    } else {
      dd=(int)(t1.tv_sec-t0.tv_sec)/86400;
      ss=(t1.tv_sec-t0.tv_sec)%86400;
      ms=(t1.tv_usec-t0.tv_usec+500)/10000;
    }//end if
    int hh=ss/3600;
    int mm=(ss-hh*3600)/60;
    ss%=60;
    printf("%2d:%17ld%16ld%8.3d:%02d:%02d:%02d.%02d\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
  }//end for
  return 0;
}
