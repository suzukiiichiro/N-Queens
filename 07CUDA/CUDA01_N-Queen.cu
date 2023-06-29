/**
 CUDAで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイルと実行
 $ nvcc -O3 CUDA**_N-Queen.cu && ./a.out (-c|-r|-g)
                    -c:cpu 
                    -r cpu再帰 
                    -g GPU 
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define THREAD_NUM		96
#define MAX 27
//変数宣言
long Total=0 ;    //GPU     
long Unique=0;    //GPU
int aBoard[MAX];  //版の配列
int COUNT=0;      //カウント用
//関数宣言CUDA
__global__ void nqueen_cuda(int *d_aBoard,int *d_results,int *d_count, int COUNT,int row,int size);
void solve_nqueen_cuda(int si,int steps);
bool InitCUDA();
//関数宣言CPU
void print(int size);
void NQueen(int row,int size);
void NQueenR(int row,int size);
//
// CUDA GPU
__global__
void nqueen_cuda(int *d_aBoard,int *d_results,int *d_count,int COUNT,int row,int size){
  bool matched;
  while(row>=0){
    matched=false;
    for(int col=d_aBoard[row]+1;col<size;col++){
      d_aBoard[row]=col;      //Qを配置
      matched=true;
      break;
    }
    if(matched){
      row++;
      if(row==size){
        //cudaの中で　printせず配列に格納して　hostに返却する
        //ex 0,1,1,3 だったら　3110
        int sum=0;
        for(int j=0;j<size;j++){
          sum+=d_aBoard[j]*pow(10,j);   
        }
        d_results[COUNT++]=sum;
        row--;
      }
    }else{
      if(d_aBoard[row]!=-1){
        d_aBoard[row]=-1;
      }
      row--;
    }
  }
	d_count[0]=COUNT;//カウントを代入
}
// CUDA CPU
void solve_nqueen_cuda(int si,int steps){
    //メモリ登録
    int *h_aBoard;
    int *h_count;
    int *h_results;
    cudaMallocHost((void**)&h_aBoard,sizeof(int)*MAX);
    cudaMallocHost((void**)&h_results,sizeof(int)*steps);
    cudaMallocHost((void**)&h_count,sizeof(int));
    int *d_aBoard;
    int *d_results;
    int *d_count;
    cudaMalloc((void**)&d_aBoard,sizeof(int)*MAX);
    cudaMalloc((void**)&d_results,sizeof(int)*steps);
    cudaMalloc((void**)&d_count,sizeof(int));
    //初期化
    for(int i=0;i<si;i++){
        h_aBoard[i]=-1;
    }
    //カウンターを初期化
    h_count[0]=0;
    //host to device
    cudaMemcpy(d_aBoard,h_aBoard,
      sizeof(int)*MAX,cudaMemcpyHostToDevice);
    cudaMemcpy(d_count,h_count,
      sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_results,h_results,
      sizeof(int)*steps,cudaMemcpyHostToDevice);
    nqueen_cuda<<<1,1>>>(d_aBoard,d_results,d_count,0,0,si);
    //device to host
    cudaMemcpy(h_count,d_count,
      sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_results,d_results,
      sizeof(int)*steps,cudaMemcpyDeviceToHost);
    //出力
    for(int i=0;i<h_count[0];i++){
      printf("%d:%08d\n",i+1,h_results[i]);  
    }
    //開放
    cudaFreeHost(h_aBoard);
    cudaFreeHost(h_results);
    cudaFreeHost(h_count);
    cudaFree(d_aBoard);
    cudaFree(d_results);
    cudaFree(d_count);
}
// CUDA 初期化
bool InitCUDA(){
  int count;
  cudaGetDeviceCount(&count);
  if(count==0){fprintf(stderr,"There is no device.\n");return false;}
  int i;
  for(i=0;i<count;i++){
    cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop,i)==cudaSuccess){if(prop.major>=1){break;} }
  }
  if(i==count){fprintf(stderr,"There is no device supporting CUDA 1.x.\n");return false;}
  cudaSetDevice(i);
  return true;
}
//出力用のメソッド
void print(int size){
  printf("%d: ",++COUNT);
  for(int j=0;j<size;j++){
    printf("%d ",aBoard[j]);
  }
  printf("\n");
}
//非再帰版ロジックメソッド
void NQueen(int row,int size){
  bool matched;
  while(row>=0){
    matched=false;
    for(int col=aBoard[row]+1;col<size;col++){
      aBoard[row]=col;      //Qを配置
      matched=true;
      break;
    }
    if(matched){
      row++;
      if(row==size){
        print(size);
        row--;
      }
    }else{
      if(aBoard[row]!=-1){
        aBoard[row]=-1;
      }
      row--;
    }
  }
}
//再帰版ロジックメソッド
void NQueenR(int row,int size){
  if(row==size){         //SIZEは5で固定
    print(size);         //rowが5になったら出力
  }else{
    for(int col=aBoard[row]+1;col<size;col++){
      aBoard[row]=col;  //Qを配置
      NQueenR(row+1,size);
      aBoard[row]=-1;   //空き地に戻す
    }
  }
}
//メインメソッド
int main(int argc,char** argv) {
  int size=5;
  bool cpu=false,cpur=false,gpu=false;
  int argstart=1,steps=24576;
  /** パラメータの処理 */
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
    else{ cpur=true; }
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g|-r]\n",argv[0]);
    printf("  -c: CPU only\n");
    printf("  -r: CPUR only\n");
    printf("  -g: GPU only\n");
    printf("Default CPUR to CPU 8 queen\n");
  }
  /** 出力と実行 */
  //aBoard配列を-1 で初期化
  for(int i=0;i<size;i++){ aBoard[i]=-1; }
  if(cpu){ 
    printf("\n\n1. CPU 非再帰 ブルートフォース　力任せ探索\n");
    NQueen(0,size); 
	}
  if(cpur){ 
    printf("\n\n1. CPU 再帰 ブルートフォース　力任せ探索\n");
    NQueenR(0,size); 
	}
  if(gpu){
    printf("\n\n1. GPU 非再帰 ブルートフォース　力任せ探索\n");
    if(!InitCUDA()){return 0;}
    solve_nqueen_cuda(size,steps);
  }
  return 0;
}
