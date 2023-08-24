/**
 *
 * bash版ミラーのC言語版のGPU/CUDA移植版
 *
 詳しい説明はこちらをどうぞ
 https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題
 *
アーキテクチャの指定（なくても問題なし、あれば高速）
-arch=sm_13 or -arch=sm_61

CPUの再帰での実行
$ nvcc -O3 -arch=sm_61 02CUDA_Mirror.cu && ./a.out -r

CPUの非再帰での実行
$ nvcc -O3 -arch=sm_61 02CUDA_Mirror.cu && ./a.out -c

GPUのシングルスレッド
$ nvcc -O3 -arch=sm_61 02CUDA_Mirror.cu && ./a.out -g

GPUのマルチスレッド
ミラー GPUノードレイヤー
$ nvcc -O3 -arch=sm_61 02CUDA_Mirror.cu && ./a.out -n
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2               0  00:00:00:00.13
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
15:      2279184               0  00:00:00:00.19
16:     14772512               0  00:00:00:01.60
17:     95815104               0  00:00:00:15.10
18:    666090624               0  00:00:02:40.78


・std::vector<long> kLayer_nodeLayer(int size,int k)
ここでミラー処理をしている。

  for(unsigned int i=0;i<size/2;++i){
  は偶数、奇数かかわらず半分だけ実行する
  n4なら0,1 n5なら 0,1  n6なら 0,1,2 n7なら 0,1,2

  if(size%2){                 
  は奇数の場合だけ実行する
  n5なら 2  n7 なら 3
  この場合2行目は、半分だけ実行する

・void mirror_build_nodeLayer(int size)
  int numSolutions = nodes.size() / 3; 
  left,down,rightで1セットなので/3 kLayer_nodeLayerで半分だけ実行しているのでここは/3のまま
 

  for (long i = 0; i < numSolutions; i++) {
      solutions += 2*hostSolutions[i]; // Symmetry
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
// システムによって以下のマクロが必要であればコメントを外してください。
//#define UINT64_C(c) c ## ULL
//
// グローバル変数
unsigned long TOTAL=0; 
unsigned long UNIQUE=0;
//ミラー処理部分 非再帰版
void mirror_solve_NR(unsigned int size,unsigned int row,unsigned int _left,unsigned int _down, unsigned int _right)
{
  unsigned int mask=(1<<size)-1;
  unsigned int bit=0;
  unsigned int down[MAX];   //ポストフラグ/ビットマップ/ミラー
  unsigned int left[MAX];   //ポストフラグ/ビットマップ/ミラー
  unsigned int right[MAX];  //ポストフラグ/ビットマップ/ミラー
  unsigned int bitmap[MAX]; //ミラー
  left[row]=_left;
  down[row]=_down;
  right[row]=_right;
  bitmap[row]=mask&~(left[row]|down[row]|right[row]);
  while(row>0){
    if(bitmap[row]>0){
      bit=-bitmap[row]&bitmap[row];
      bitmap[row]=bitmap[row]^bit;
      if(row==(size-1)){
        TOTAL++;
        row--;
      }else{
        unsigned int n=row++;
        left[row]=(left[n]|bit)<<1;
        down[row]=(down[n]|bit);
        right[row]=(right[n]|bit)>>1;
        //クイーンが配置可能な位置を表す
        bitmap[row]=mask&~(left[row]|down[row]|right[row]);
      }
    }else{
      row--;
    }
  }
}
// ミラー 非再帰版
void mirror_NR(unsigned int size)
{
  unsigned int bit=0;
  unsigned int limit=size%2 ? size/2-1 : size/2;
  for(unsigned int i=0;i<size/2;++i){ //奇数でも偶数でも通過
    bit=1<<i;
    mirror_solve_NR(size,1,bit<<1,bit,bit>>1);
  }
  if(size%2){                 //奇数で通過
    bit=1<<(size-1)/2;
    unsigned int left=bit<<1;
    unsigned int down=bit;
    unsigned int right=bit>>1;
    for(unsigned int i=0;i<limit;++i){
      bit=1<<i;
      mirror_solve_NR(size,2,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
  TOTAL=TOTAL<<1;    //倍にする
}
// ミラー処理部分 再帰版
void mirror_solve_R(unsigned int size,unsigned int row,unsigned int left,unsigned int down,unsigned int right)
{
  unsigned int mask=(1<<size)-1;
  unsigned int bit=0;
  if(row==size){
    TOTAL++;
  }else{
    for(unsigned int bitmap=mask&~(left|down|right);bitmap;bitmap=bitmap&~bit){
      bit=-bitmap&bitmap;
      mirror_solve_R(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
}
// ミラー 再帰版
void mirror_R(unsigned int size)
{
  unsigned int bit=0;
  unsigned int limit=size%2 ? size/2-1 : size/2;
  for(unsigned int i=0;i<size/2;++i){
    bit=1<<i;
    mirror_solve_R(size,1,bit<<1,bit,bit>>1);
  }
  if(size%2){               //奇数で通過
    bit=1<<(size-1)/2;
    unsigned int left=bit<<1;
    unsigned int down=bit;
    unsigned int right=bit>>1;
    for(unsigned int i=0;i<limit;++i){
      bit=1<<i;
      mirror_solve_R(size,2,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
  TOTAL=TOTAL<<1;    //倍にする
}
// ミラー 処理部分
__host__ __device__ 
long mirror_solve_nodeLayer(int size,long left,long down,long right)
{
  unsigned int mask=(1<<size)-1;
  unsigned int bit=0;
  unsigned long counter=0;
  if(down==mask){
    return 1;
  }else{
    for(unsigned int bitmap=mask&~(left|down|right);bitmap;bitmap=bitmap&~bit){
      bit=-bitmap&bitmap;
      counter+=mirror_solve_nodeLayer(size,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
  return counter;
}
// ミラー クイーンの効きを判定して解を返す
__host__ __device__ 
long mirror_logic_nodeLayer(int size,long left,long down,long right)
{
  unsigned long counter = 0;
  unsigned int bit=0;
  unsigned int limit=size%2 ? size/2-1 : size/2;
  for(unsigned int i=0;i<size/2;++i){
    bit=1<<i;
    counter+=mirror_solve_nodeLayer(size,bit<<1,bit,bit>>1);
  }
  if(size%2){               //奇数で通過
    bit=1<<(size-1)/2;
    unsigned int left=bit<<1;
    unsigned int down=bit;
    unsigned int right=bit>>1;
    for(unsigned int i=0;i<limit;++i){
      bit=1<<i;
      counter+=mirror_solve_nodeLayer(size,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
  return counter<<1; // 倍にする
}
// i 番目のメンバを i 番目の部分木の解で埋める
__global__ 
void dim_nodeLayer(int size,long* nodes, long* solutions, int numElements)
{
  int i=blockDim.x * blockIdx.x + threadIdx.x;
  if(i<numElements){
    /**
      ミラーのGPUスレッド(-n)の場合は、予めCPU側で奇数と偶数で分岐させるので、
      奇数と偶数を条件分岐するmirror_logic_nodeLayer()を通過させる必要がない
      */
    solutions[i]=mirror_solve_nodeLayer(size,nodes[3 * i],nodes[3 * i + 1],nodes[3 * i + 2]);
  }
}
// 0以外のbitをカウント
int countBits_nodeLayer(long n)
{
  int counter = 0;
  while (n){
    n &= (n - 1); // 右端のゼロ以外の数字を削除
    counter++;
  }
  return counter;
}
// ノードをk番目のレイヤーのノードで埋める
long kLayer_nodeLayer(int size,std::vector<long>& nodes, int k, long left, long down, long right)
{
  long counter=0;
  long mask=(1<<size)-1;
  // すべてのdownが埋まったら、解決策を見つけたことになる。
  if (countBits_nodeLayer(down) == k) {
    nodes.push_back(left);
    nodes.push_back(down);
    nodes.push_back(right);
    return 1;
  }
  long bit=0;
  for(long bitmap=mask&~(left|down|right);bitmap;bitmap^=bit){
    bit=-bitmap&bitmap;
    // 解を加えて対角線をずらす
    //counter+=kLayer_nodeLayer(size,nodes,k,(left|bit)>>1,(down|bit),(right|bit)<<1); 
    counter+=kLayer_nodeLayer(size,nodes,k,(left|bit)<<1,(down|bit),(right|bit)>>1); 
  }
  return counter;
}
// k 番目のレイヤのすべてのノードを含むベクトルを返す。
std::vector<long> kLayer_nodeLayer(int size,int k)
{
  std::vector<long> nodes{};
  /**
    CPU側で奇数と偶数で条件分岐する
    */
  //kLayer_nodeLayer(size,nodes, k, 0, 0, 0);
  unsigned int bit=0;
  unsigned int limit=size%2 ? size/2-1 : size/2;
  for(unsigned int i=0;i<size/2;++i){
    bit=1<<i;
    kLayer_nodeLayer(size,nodes,k,bit<<1,bit,bit>>1);
  }
  if(size%2){               //奇数で通過
    bit=1<<(size-1)/2;
    unsigned int left=bit<<1;
    unsigned int down=bit;
    unsigned int right=bit>>1;
    for(unsigned int i=0;i<limit;++i){
      bit=1<<i;
      kLayer_nodeLayer(size,nodes,k,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
  return nodes;
}
// 【GPU ミラー】ノードレイヤーの作成
void mirror_build_nodeLayer(int size)
{
  // ツリーの3番目のレイヤーにあるノード
  //（それぞれ連続する3つの数字でエンコードされる）のベクトル。
  // レイヤー2以降はノードの数が均等なので、対称性を利用できる。
  // レイヤ4には十分なノードがある（N16の場合、9844）。
  std::vector<long> nodes = kLayer_nodeLayer(size,4); 

  // デバイスにはクラスがないので、
  // 最初の要素を指定してからデバイスにコピーする。
  size_t nodeSize = nodes.size() * sizeof(long);
  long* hostNodes = (long*)malloc(nodeSize);
  hostNodes = &nodes[0];
  long* deviceNodes = NULL;
  cudaMalloc((void**)&deviceNodes, nodeSize);
  cudaMemcpy(deviceNodes, hostNodes, nodeSize, cudaMemcpyHostToDevice);

  // デバイス出力の割り当て
  long* deviceSolutions = NULL;
  /** ミラーでは/6 を /3に変更する */
  // 必要なのはノードの半分だけで
  // 各ノードは3つの整数で符号化される。
  //int numSolutions = nodes.size() / 6; 
  int numSolutions = nodes.size() / 3; 
  size_t solutionSize = numSolutions * sizeof(long);
  cudaMalloc((void**)&deviceSolutions, solutionSize);

  // CUDAカーネルを起動する。
  int threadsPerBlock = 256;
  int blocksPerGrid = (numSolutions + threadsPerBlock - 1) / threadsPerBlock;
  dim_nodeLayer <<<blocksPerGrid, threadsPerBlock >>> (size,deviceNodes, deviceSolutions, numSolutions);

  // 結果をホストにコピー
  long* hostSolutions = (long*)malloc(solutionSize);
  cudaMemcpy(hostSolutions, deviceSolutions, solutionSize, cudaMemcpyDeviceToHost);

  // 部分解を加算し、結果を表示する。
  long solutions = 0;
  for (long i = 0; i < numSolutions; i++) {
      solutions += 2*hostSolutions[i]; // Symmetry
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
  int i;
  for(i=0;i<count;i++){
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
  int argstart=2;
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
  if(cpur){ printf("\n\nミラー 再帰 \n"); }
  else if(cpu){ printf("\n\nミラー 非再帰 \n"); }
  else if(gpu){ printf("\n\nミラー GPU\n"); }
  else if(gpuNodeLayer){ printf("\n\nミラー GPUノードレイヤー \n"); }
  if(cpu||cpur)
  {
    int min=4; 
    int targetN=17;
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:           Total           Unique          dd:hh:mm:ss.ms");
    for(int size=min;size<=targetN;size++){
      TOTAL=UNIQUE=0;
      gettimeofday(&t0, NULL);//計測開始
      if(cpur){ //再帰
        mirror_R(size);
      }
      if(cpu){ //非再帰
        mirror_NR(size);
      }
      //
      gettimeofday(&t1, NULL);//計測終了
      int ss;int ms;int dd;
      if(t1.tv_usec<t0.tv_usec) {
        dd=(t1.tv_sec-t0.tv_sec-1)/86400;
        ss=(t1.tv_sec-t0.tv_sec-1)%86400;
        ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
      }else {
        dd=(t1.tv_sec-t0.tv_sec)/86400;
        ss=(t1.tv_sec-t0.tv_sec)%86400;
        ms=(t1.tv_usec-t0.tv_usec+500)/10000;
      }//end if
      int hh=ss/3600;
      int mm=(ss-hh*3600)/60;
      ss%=60;
      printf("%2d:%16ld%17ld%12.2d:%02d:%02d:%02d.%02d\n",
          size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
    } //end for
  }//end if
  if(gpu||gpuNodeLayer)
  {
    if(!InitCUDA()){return 0;}
    /* int steps=24576; */
    int min=4;
    int targetN=21;
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    for(int size=min;size<=targetN;size++){
      gettimeofday(&t0,NULL);   // 計測開始
      if(gpu){
        TOTAL=UNIQUE=0;
        // GPUは起動するがノードレイヤーは行わない
        TOTAL=mirror_logic_nodeLayer(size,0,0,0); //ミラー
      }else if(gpuNodeLayer){
        TOTAL=UNIQUE=0;
        // GPUを起動し、ノードレイヤーも行う
        mirror_build_nodeLayer(size); // ミラー
      }
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
      printf("%2d:%13ld%16ld%4.2d:%02d:%02d:%02d.%02d\n",
          size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
    }//end for
  }//end if
  return 0;
}
