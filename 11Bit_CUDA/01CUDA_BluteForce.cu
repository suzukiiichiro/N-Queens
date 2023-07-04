/**
 *
 * bash版ブルートフォースのC言語版のGPU/CUDA移植版
 *
 詳しい説明はこちらをどうぞ
 https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題
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
#define THREAD_NUM		96
#define MAX 27
// システムによって以下のマクロが必要であればコメントを外してください。
//#define UINT64_C(c) c ## ULL
//
// グローバル変数
unsigned long TOTAL=0; 
unsigned long UNIQUE=0;
int board[MAX];  //ボード配列
// ブルートフォース 効き筋をチェック
int check_bluteForce(unsigned int size)
{
  for(unsigned int r=1;r<size;++r){
    unsigned int val=0;
    for(unsigned int i=0;i<r;++i){
      if(board[i]>=board[r]){
        val=board[i]-board[r];
      }else{
        val=board[r]-board[i];
      }
      if(board[i]==board[r]||val==(r-i)){
        return 0;
      }
    }
  }
  return 1;
}
//ブルートフォース 非再帰版
void bluteForce_NR(unsigned int size,int row)
{
  // １．非再帰は初期化が必要
  for(unsigned int i=0;i<size;++i){
    board[i]=-1;
  }
  // ２．再帰で呼び出される関数内を回す処理
  while(row>-1){
    unsigned int matched=0;   //クイーンを配置したか
    // ３．再帰処理のループ部分
    // 非再帰では過去の譜石を記憶するためにboard配列を使う
    for(unsigned int col=board[row]+1;col<size;++col){
      board[row]=col;
      matched=1;
      break;
    }
    // ４．配置したら実行したい処理
    if(matched){
      row++;
      // ５．最下部まで到達したときの処理';
      if(row==size){
        row--;
        // 効きをチェック
        if(check_bluteForce(size)==1){
          TOTAL++;
        }
      }
      // ６．配置できなくてバックトラックしたい処理
    }else{
      if(board[row]!=-1){
        board[row]=-1;
      }
      row--;
    } // end if
  }//end while
}
//ブルートフォース 再帰版
void bluteForce_R(unsigned int size,int row)
{
  if(row==size){
    if(check_bluteForce(size)==1){
      TOTAL++; // グローバル変数
    }
  }else{
    for(int col=0;col<size;++col){
      board[row]=col;
      bluteForce_R(size,row+1);
    }
  }
}
// クイーンの効きを判定して解を返す
__host__ __device__ 
long bluteForce_nQueens(int size,long left,long down,long right)
{
  long mask=(1<<size)-1;
  long counter = 0;
  if (down==mask) { // downがすべて専有され解が見つかる
    return 1;
  }
  long bit=0;
  for(long bitmap=mask&~(left|down|right);bitmap;bitmap^=bit){
    bit=-bitmap&bitmap;
    counter += bluteForce_nQueens(size,(left|bit)>>1,(down|bit),(right|bit)<< 1); 
  }
  return counter;
}
// i 番目のメンバを i 番目の部分木の解で埋める
__global__ 
void calculateSolutions(int size,long* nodes, long* solutions, int numElements)
{
  int i=blockDim.x * blockIdx.x + threadIdx.x;
  if(i<numElements){
    solutions[i]=bluteForce_nQueens(size,nodes[3 * i],nodes[3 * i + 1],nodes[3 * i + 2]);
  }
}
// 0以外のbitをカウント
int countBits(long n)
{
  int counter = 0;
  while (n){
    n &= (n - 1); // 右端のゼロ以外の数字を削除
    counter++;
  }
  return counter;
}
// ノードをk番目のレイヤーのノードで埋める
long kLayer(int size,std::vector<long>& nodes, int k, long left, long down, long right)
{
  long counter=0;
  long mask=(1<<size)-1;
  // すべてのdownが埋まったら、解決策を見つけたことになる。
  if (countBits(down) == k) {
    nodes.push_back(left);
    nodes.push_back(down);
    nodes.push_back(right);
    return 1;
  }
  long bit=0;
  for(long bitmap=mask&~(left|down|right);bitmap;bitmap^=bit){
    bit=-bitmap&bitmap;
    // 解を加えて対角線をずらす
    counter+=kLayer(size,nodes,k,(left|bit)>>1,(down|bit),(right|bit)<<1); 
  }
  return counter;
}
// k 番目のレイヤのすべてのノードを含むベクトルを返す。
std::vector<long> kLayer(int size,int k)
{
  std::vector<long> nodes{};
  kLayer(size,nodes, k, 0, 0, 0);
  return nodes;
}
// ノードレイヤーの作成
void bluteForce_nodeLayer(int size)
{
  //int size=16;
  // ツリーの3番目のレイヤーにあるノード
  //（それぞれ連続する3つの数字でエンコードされる）のベクトル。
  // レイヤー2以降はノードの数が均等なので、対称性を利用できる。
  // レイヤ4には十分なノードがある（N16の場合、9844）。
  std::vector<long> nodes = kLayer(size,4); 

  // デバイスにはクラスがないので、
  // 最初の要素を指定してからデバイスにコピーする。
  size_t nodeSize = nodes.size() * sizeof(long);
  long* hostNodes = (long*)malloc(nodeSize);
  hostNodes = &nodes[0];
  long* deviceNodes = NULL;
  cudaMalloc((void**)&deviceNodes, nodeSize);
  cudaMemcpy(deviceNodes, hostNodes, nodeSize, cudaMemcpyHostToDevice);

  // board配列
  size_t boardSize = size*sizeof(int);
  int *board=(int*)malloc(boardSize);
  int *deviceBoard;
  cudaMalloc((void**)&deviceBoard,boardSize);
  cudaMemcpy(deviceBoard,board,boardSize,cudaMemcpyHostToDevice);


  // デバイス出力の割り当て
  long* deviceSolutions = NULL;
  int numSolutions = nodes.size() / 6; // We only need half of the nodes, and each node is encoded by 3 integers.
  size_t solutionSize = numSolutions * sizeof(long);
  cudaMalloc((void**)&deviceSolutions, solutionSize);

  // CUDAカーネルを起動する。
  int threadsPerBlock = 256;
  int blocksPerGrid = (numSolutions + threadsPerBlock - 1) / threadsPerBlock;
  calculateSolutions <<<blocksPerGrid, threadsPerBlock >>> (size,deviceNodes, deviceSolutions, numSolutions);

  // 結果をホストにコピー
  long* hostSolutions = (long*)malloc(solutionSize);
  cudaMemcpy(hostSolutions, deviceSolutions, solutionSize, cudaMemcpyDeviceToHost);

  // 部分解を加算し、結果を表示する。
  long solutions = 0;
  for (long i = 0; i < numSolutions; i++) {
      solutions += 2*hostSolutions[i]; // Symmetry
  }

  // 出力
  //std::cout << "We have " << solutions << " solutions on a " << size << " by " << size << " board." << std::endl;
  TOTAL=solutions;
  //return 0;
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
    else if(argv[1][1]=='n'||argv[1][1]=='N'){gpu=true;}
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
  if(cpur){ printf("\n\nブルートフォース 再帰 \n"); }
  else if(cpu){ printf("\n\nブルートフォース 非再帰 \n"); }
  else if(gpu){ printf("\n\nブルートフォース GPU\n"); }
  else if(gpuNodeLayer){ printf("\n\nブルートフォース GPUノードレイヤー \n"); }
  if(cpu||cpur){
    int min=4; 
    int targetN=17;
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:           Total           Unique          dd:hh:mm:ss.ms");
    for(int size=min;size<=targetN;size++){
      TOTAL=UNIQUE=0;
      gettimeofday(&t0, NULL);//計測開始
      if(cpur){ //再帰
        bluteForce_R(size,0);
      }
      if(cpu){ //非再帰
        bluteForce_NR(size,0);
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
  if(gpu||gpuNodeLayer){
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
        TOTAL=bluteForce_nQueens(size,0,0,0);
      }else if(gpuNodeLayer){
        TOTAL=UNIQUE=0;
        bluteForce_nodeLayer(size);
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
