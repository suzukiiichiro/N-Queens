/**
 CUDAで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイルと実行
 $ nvcc CUDA**_N-Queen.cu && ./a.out (-c|-r|-g|-s)
                    -c:cpu 
                    -r cpu再帰 
                    -g GPU 
                    -s SGPU(サマーズ版と思われる)

 ３．バックトラック

 　各列、対角線上にクイーンがあるかどうかのフラグを用意し、途中で制約を満た
 さない事が明らかな場合は、それ以降のパターン生成を行わない。
 　各列、対角線上にクイーンがあるかどうかのフラグを用意することで高速化を図る。
 　これまでは行方向と列方向に重複しない組み合わせを列挙するものですが、王妃
 は斜め方向のコマをとることができるので、どの斜めライン上にも王妃をひとつだ
 けしか配置できない制限を加える事により、深さ優先探索で全ての葉を訪問せず木
 を降りても解がないと判明した時点で木を引き返すということができます。


 実行結果
$ nvcc CUDA03_N-Queen.cu  && ./a.out -r
３．CPUR 再帰 バックトラック
 4:            2               0            0.00
 5:           10               0            0.00
 6:            4               0            0.00
 7:           40               0            0.00
 8:           92               0            0.00
 9:          352               0            0.00
10:          724               0            0.00
11:         2680               0            0.01
12:        14200               0            0.05
13:        73712               0            0.27
14:       365596               0            1.66
15:      2279184               0           10.63
16:     14772512               0         1:12.95
17:     95815104               0         8:40.60

$ nvcc CUDA03_N-Queen.cu  && ./a.out -c
３．CPU 非再帰 バックトラック
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               0            0.00
 5:           10               0            0.00
 6:            4               0            0.00
 7:           40               0            0.00
 8:           92               0            0.00
 9:          352               0            0.00
10:          724               0            0.00
11:         2680               0            0.01
12:        14200               0            0.05
13:        73712               0            0.29
14:       365596               0            1.73
15:      2279184               0           11.38
16:     14772512               0         1:14.90
17:     95815104               0         9:00.09

$ nvcc CUDA03_N-Queen.cu  && ./a.out -g
３．GPU 非再帰 バックトラック
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2               0  00:00:00:00.02
 5:           10               0  00:00:00:00.00
 6:            4               0  00:00:00:00.00
 7:           40               0  00:00:00:00.00
 8:           92               0  00:00:00:00.01
 9:          352               0  00:00:00:00.06
10:          724               0  00:00:00:00.27
11:         2680               0  00:00:00:01.09
12:        14200               0  00:00:00:05.15
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
long Unique=0;      //GPU
int down[2*MAX-1];  //down:flagA 縦 配置フラグ　
int left[2*MAX-1];  //left:flagB 斜め配置フラグ　
int right[2*MAX-1]; //right:flagC 斜め配置フラグ　
long TOTAL=0;       //CPU,CPUR
long UNIQUE=0;      //CPU,CPUR
int aBoard[MAX];
//関数宣言 SGPU
__global__ void sgpu_cuda_kernel(int size,int mark,unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,unsigned int* results,int totalCond);
long long sgpu_solve_nqueen_cuda(int size,int steps); 
//関数宣言GPU
__global__ void nqueen_cuda(int *d_aBoard,int *d_down,int *d_right,int *d_left,long *d_results,long TOTAL,int row,int size);
void solve_nqueen_cuda(int si,long results[2],int steps);
bool InitCUDA();
//関数宣言CPU
void TimeFormat(clock_t utime,char *form);
void NQueen(int row,int size);
void NQueenR(int row,int size);
//
__global__ void sgpu_cuda_kernel(
    int size,int mark,
    unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,
    unsigned int* results,int totalCond){
  const int tid=threadIdx.x;
  const int bid=blockIdx.x;
  const int idx=bid*blockDim.x+tid;
  __shared__ unsigned int down[THREAD_NUM][10];
  __shared__ unsigned int left[THREAD_NUM][10];
  __shared__ unsigned int right[THREAD_NUM][10];
  __shared__ unsigned int bitmap[THREAD_NUM][10];
  __shared__ unsigned int sum[THREAD_NUM];
  const unsigned int mask=(1<<size)-1;
  int total=0;
  int row=0;
  unsigned int bit;
  if(idx<totalCond){
    down[tid][row]=totalDown[idx];
    left[tid][row]=totalLeft[idx];
    right[tid][row]=totalRight[idx];
    bitmap[tid][row]=down[tid][row]|left[tid][row]|right[tid][row];
    while(row>=0){
      if((bitmap[tid][row]&mask)==mask){row--;}
      else{
        bit=(bitmap[tid][row]+1)&~bitmap[tid][row];
        bitmap[tid][row]|=bit;
        if((bit&mask)!=0){
          if(row+1==mark){total++;row--;}
          else{
            down[tid][row+1]=down[tid][row]|bit;
            left[tid][row+1]=(left[tid][row]|bit)<<1;
            right[tid][row+1]=(right[tid][row]|bit)>>1;
            bitmap[tid][row+1]=(down[tid][row+1]|left[tid][row+1]|right[tid][row+1]);
            row++;
          }
        }else{row--;}
      }
    }
    sum[tid]=total;
  }else{sum[tid]=0;} 
  __syncthreads();if(tid<64&&tid+64<THREAD_NUM){sum[tid]+=sum[tid+64];} 
  __syncthreads();if(tid<32){sum[tid]+=sum[tid+32];} 
  __syncthreads();if(tid<16){sum[tid]+=sum[tid+16];} 
  __syncthreads();if(tid<8){sum[tid]+=sum[tid+8];} 
  __syncthreads();if(tid<4){sum[tid]+=sum[tid+4];} 
  __syncthreads();if(tid<2){sum[tid]+=sum[tid+2];} 
  __syncthreads();if(tid<1){sum[tid]+=sum[tid+1];} 
  __syncthreads();if(tid==0){results[bid]=sum[0];}
}
//
long long sgpu_solve_nqueen_cuda(int size,int steps) {
  unsigned int down[32];
  unsigned int left[32];
  unsigned int right[32];
  unsigned int bitmap[32];
  unsigned int bit;
  if(size<=0||size>32){return 0;}
  unsigned int* totalDown=new unsigned int[steps];
  unsigned int* totalLeft=new unsigned int[steps];
  unsigned int* totalRight=new unsigned int[steps];
  unsigned int* results=new unsigned int[steps];
  unsigned int* downCuda;
  unsigned int* leftCuda;
  unsigned int* rightCuda;
  unsigned int* resultsCuda;
  cudaMalloc((void**) &downCuda,sizeof(int)*steps);
  cudaMalloc((void**) &leftCuda,sizeof(int)*steps);
  cudaMalloc((void**) &rightCuda,sizeof(int)*steps);
  cudaMalloc((void**) &resultsCuda,sizeof(int)*steps/THREAD_NUM);
  const unsigned int mask=(1<<size)-1;
  const unsigned int mark=size>11?size-10:2;
  long long total=0;
  int totalCond=0;
  int row=0;
  down[0]=0;
  left[0]=0;
  right[0]=0;
  bitmap[0]=0;
  bool matched=false;
  for(int col=0;col<size/2;col++){
    bit=(1<<col);
    bitmap[0]|=bit;
    down[1]=bit;
    left[1]=bit<<1;
    right[1]=bit>>1;
    bitmap[1]=(down[1]|left[1]|right[1]);
    row=1;
    while(row>0){
      if((bitmap[row]&mask)==mask){row--;}
      else{
        bit=(bitmap[row]+1)&~bitmap[row];
        bitmap[row]|=bit;
        if((bit&mask)!=0){
          down[row+1]=down[row]|bit;
          left[row+1]=(left[row]|bit)<<1;
          right[row+1]=(right[row]|bit)>>1;
          bitmap[row+1]=(down[row+1]|left[row+1]|right[row+1]);
          row++;
          if(row==mark){
            totalDown[totalCond]=down[row];
            totalLeft[totalCond]=left[row];
            totalRight[totalCond]=right[row];
            totalCond++;
            if(totalCond==steps){
              if(matched){
                cudaMemcpy(results,resultsCuda,
                    sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
                for(int col=0;col<steps/THREAD_NUM;col++){total+=results[col];}
                matched=false;
              }
              cudaMemcpy(downCuda,totalDown,
                  sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              cudaMemcpy(leftCuda,totalLeft,
                  sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              cudaMemcpy(rightCuda,totalRight,
                  sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              /** backTrack+bitmap*/
              sgpu_cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
                >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
              matched=true;
              totalCond=0;
            }
            row--;
          }
        }else{row--;}
      }
    }
  }
  if(matched){
    cudaMemcpy(results,resultsCuda,
        sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    for(int col=0;col<steps/THREAD_NUM;col++){total+=results[col];}
    matched=false;
  }
  cudaMemcpy(downCuda,totalDown,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(leftCuda,totalLeft,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(rightCuda,totalRight,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  /** backTrack+bitmap*/
  sgpu_cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
    >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
  cudaMemcpy(results,resultsCuda,
      sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  for(int col=0;col<steps/THREAD_NUM;col++){total+=results[col];}	
  total*=2;


  if(size%2==1){
    matched=false;
    totalCond=0;
    bit=(1<<(size-1)/2);
    bitmap[0]|=bit;
    down[1]=bit;
    left[1]=bit<<1;
    right[1]=bit>>1;
    bitmap[1]=(down[1]|left[1]|right[1]);
    row=1;
    while(row>0){
      if((bitmap[row]&mask)==mask){row--;}
      else{
        bit=(bitmap[row]+1)&~bitmap[row];
        bitmap[row]|=bit;
        if((bit&mask)!=0){
          down[row+1]=down[row]|bit;
          left[row+1]=(left[row]|bit)<<1;
          right[row+1]=(right[row]|bit)>>1;
          bitmap[row+1]=(down[row+1]|left[row+1]|right[row+1]);
          row++;
          if(row==mark){
            totalDown[totalCond]=down[row];
            totalLeft[totalCond]=left[row];
            totalRight[totalCond]=right[row];
            totalCond++;
            if(totalCond==steps){
              if(matched){
                cudaMemcpy(results,resultsCuda,
                    sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
                for(int col=0;col<steps/THREAD_NUM;col++){total+=results[col];}
                matched=false;
              }
              cudaMemcpy(downCuda,totalDown,
                  sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              cudaMemcpy(leftCuda,totalLeft,
                  sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              cudaMemcpy(rightCuda,totalRight,
                  sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              /** backTrack+bitmap*/
              sgpu_cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
                >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
              matched=true;
              totalCond=0;
            }
            row--;
          }
        }else{row--;}
      }
    }
    if(matched){
      cudaMemcpy(results,resultsCuda,
          sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
      for(int col=0;col<steps/THREAD_NUM;col++){total+=results[col];}
      matched=false;
    }
    cudaMemcpy(downCuda,totalDown,
        sizeof(int)*totalCond,cudaMemcpyHostToDevice);
    cudaMemcpy(leftCuda,totalLeft,
        sizeof(int)*totalCond,cudaMemcpyHostToDevice);
    cudaMemcpy(rightCuda,totalRight,
        sizeof(int)*totalCond,cudaMemcpyHostToDevice);
    /** backTrack+bitmap*/
    sgpu_cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
      >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
    cudaMemcpy(results,resultsCuda,
        sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    for(int col=0;col<steps/THREAD_NUM;col++){total+=results[col];}
  }
  cudaFree(downCuda);
  cudaFree(leftCuda);
  cudaFree(rightCuda);
  cudaFree(resultsCuda);
  delete[] totalDown;
  delete[] totalLeft;
  delete[] totalRight;
  delete[] results;
  return total;
}
//hh:mm:ss.ms形式に処理時間を出力
void TimeFormat(clock_t utime,char *form){
  int dd,hh,mm;
  float ftime,ss;
  ftime=(float)utime/CLOCKS_PER_SEC;
  mm=(int)ftime/60;
  ss=ftime-(int)(mm*60);
  dd=mm/(24*60);
  mm=mm%(24*60);
  hh=mm/60;
  mm=mm%60;
  if(dd)
    sprintf(form,"%4d %02d:%02d:%05.2f",dd,hh,mm,ss);
  else if(hh)
    sprintf(form,"     %2d:%02d:%05.2f",hh,mm,ss);
  else if(mm)
    sprintf(form,"        %2d:%05.2f",mm,ss);
  else
    sprintf(form,"           %5.2f",ss);
}
//GPUカーネル
__global__
void nqueen_cuda(int *d_aBoard,int *d_down,int *d_right,int *d_left,long *d_results,long TOTAL,int row,int size){
  int sizeE=size-1;
  bool matched;
  while(row>=0){
    matched=false;
    // １回目はaBoard[row]が-1なのでcolを0で初期化
    // ２回目以降はcolを<sizeまで右へシフト
    for(int col=d_aBoard[row]+1;col<size;col++){
      if(d_down[col]==0 && d_right[col-row+sizeE]==0 &&d_left[col+row]==0){ 	//まだ効き筋がない
        if(d_aBoard[row]!=-1){		//Qを配置済み
          //colがaBoard[row]におきかわる
          d_down[d_aBoard[row]] =d_right[d_aBoard[row]-row+sizeE] =d_left[d_aBoard[row]+row]=0;
        }
        d_aBoard[row]=col;				//Qを配置
        d_down[col] =d_right[col-row+sizeE] =d_left[col+row]=1;			//効き筋とする
        matched=true;						//配置した
        break;
      }
    }
    if(matched){								//配置済みなら
      row++;										//次のrowへ
      if(row==size){
        //print(size); //print()でTOTALを++しない
        TOTAL++;
        row--;
      }
    }else{
      if(d_aBoard[row]!=-1){
        int col=d_aBoard[row]; /** col の代用 */
        d_aBoard[row]=-1;
        d_down[col] =d_right[col-row+sizeE] =d_left[col+row]=0;
      }
      row--;										//バックトラック
    }
  }
  d_results[0]=TOTAL;
}
//CUDA実行関数
void solve_nqueen_cuda(int si,long results[2],int steps){
    //メモリ登録
    int *h_aBoard;
    int *h_down;
    int *h_right;
    int *h_left;
    long *h_results;
    cudaMallocHost((void**)&h_aBoard,sizeof(int)*MAX);
    cudaMallocHost((void**)&h_down,sizeof(int)*2*MAX-1);
    cudaMallocHost((void**)&h_right,sizeof(int)*2*MAX-1);
    cudaMallocHost((void**)&h_left,sizeof(int)*2*MAX-1);
    cudaMallocHost((void**)&h_results,sizeof(long)*steps);
    int *d_aBoard;
    int *d_down;
    int *d_right;
    int *d_left;
    long *d_results;
    cudaMalloc((void**)&d_aBoard,sizeof(int)*MAX);
    cudaMalloc((void**)&d_down,sizeof(int)*2*MAX-1);
    cudaMalloc((void**)&d_right,sizeof(int)*2*MAX-1);
    cudaMalloc((void**)&d_left,sizeof(int)*2*MAX-1);
    cudaMalloc((void**)&d_results,sizeof(long)*steps);
    //初期化
    for(int i=0;i<si;i++){
        h_aBoard[i]=-1;
    }
    //host to device
    cudaMemcpy(d_aBoard,h_aBoard,
      sizeof(int)*MAX,cudaMemcpyHostToDevice);
    cudaMemcpy(d_down,h_down,
      sizeof(int)*2*MAX-1,cudaMemcpyHostToDevice);
    cudaMemcpy(d_right,h_right,
      sizeof(int)*2*MAX-1,cudaMemcpyHostToDevice);
    cudaMemcpy(d_left,h_left,
      sizeof(int)*2*MAX-1,cudaMemcpyHostToDevice);
    cudaMemcpy(d_results,h_results,
      sizeof(long)*steps,cudaMemcpyHostToDevice);
    //実行
    nqueen_cuda<<<1,1>>>(d_aBoard,d_down,d_right,d_left,d_results,0,0,si);
    //device to host
    cudaMemcpy(h_results,d_results,
      sizeof(long)*steps,cudaMemcpyDeviceToHost);
    //return用
    results[0]=h_results[0];
    //開放
    cudaFreeHost(h_aBoard);
    cudaFreeHost(h_down);
    cudaFreeHost(h_right);
    cudaFreeHost(h_left);
    cudaFreeHost(h_results);
    cudaFree(d_aBoard);
    cudaFree(d_down);
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_results);
}
/** CUDA 初期化 **/
bool InitCUDA(){
  int count;
  cudaGetDeviceCount(&count);
  if(count==0){fprintf(stderr,"There is no device.\n");return false;}
  int i;
  for(i=0;i<count;i++){
    cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop,i)==cudaSuccess){if(prop.major>=1){break;} }
  }
  if(i==count){
    fprintf(stderr,"There is no device supporting CUDA 1.x.\n");return false;}
  cudaSetDevice(i);
  return true;
}
// CPU 非再帰版 ロジックメソッド
void NQueen(int row,int size){
  int sizeE=size-1;
  bool matched;
  while(row>=0){
    matched=false;
    // １回目はaBoard[row]が-1なのでcolを0で初期化
    // ２回目以降はcolを<sizeまで右へシフト
    for(int col=aBoard[row]+1;col<size;col++){
      if(down[col]==0 && right[col-row+sizeE]==0 &&left[col+row]==0){ 	//まだ効き筋がない
        if(aBoard[row]!=-1){		//Qを配置済み
          //colがaBoard[row]におきかわる
          down[aBoard[row]] =right[aBoard[row]-row+sizeE] =left[aBoard[row]+row]=0;
        }
        aBoard[row]=col;				//Qを配置
        down[col] =right[col-row+sizeE] =left[col+row]=1;			//効き筋とする
        matched=true;						//配置した
        break;
      }
    }
    if(matched){								//配置済みなら
      row++;										//次のrowへ
      if(row==size){
        //print(size); //print()でTOTALを++しない
        TOTAL++;
        row--;
      }
    }else{
      if(aBoard[row]!=-1){
        int col=aBoard[row]; /** col の代用 */
        aBoard[row]=-1;
        down[col] =right[col-row+sizeE] =left[col+row]=0;
      }
      row--;										//バックトラック
    }
  }
}
// CPUR 再帰版 ロジックメソッド
void NQueenR(int row,int size){
  int sizeE=size-1;
  if(row==size){
    TOTAL++;
  }else{
    for(int col=0;col<size;col++){
      aBoard[row]=col;
      if(down[col]==0
          && right[row-col+sizeE]==0
          && left[row+col]==0){
        down[col]
          =right[row-col+sizeE]
          =left[row+col]=1;
        NQueenR(row+1,size);
        down[col]
          =right[row-col+sizeE]
          =left[row+col]=0;
      }
    }
  }
}
//メインメソッド
int main(int argc,char** argv) {
  bool cpu=false,cpur=false,gpu=false,sgpu=false;
  int argstart=1,steps=24576;
  /** パラメータの処理 */
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
    else if(argv[1][1]=='s'||argv[1][1]=='S'){sgpu=true;}
    else
      cpur=true;
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g|-r|-s]\n",argv[0]);
    printf("  -c: CPU only\n");
    printf("  -r: CPUR only\n");
    printf("  -g: GPU only\n");
    printf("  -s: SGPU only\n");
    printf("Default CPUR to 8 queen\n");
  }
  /** 出力と実行 */
  if(cpu){
    printf("\n\n３．CPU 非再帰 バックトラック\n");
  }else if(cpur){
    printf("\n\n３．CPUR 再帰 バックトラック\n");
  }else if(gpu){
    printf("\n\n３．GPU 非再帰 バックトラック\n");
  }else if(sgpu){
    printf("\n\n３．SGPU 非再帰 バックトラック\n");
  }
  if(cpu||cpur){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    clock_t st;           //速度計測用
    char t[20];           //hh:mm:ss.msを格納
    int min=4; 
    int targetN=17;
    //aBaord配列を-1で初期化
    for(int i=min;i<=targetN;i++){
      TOTAL=0; UNIQUE=0;
      for(int j=0;j<=targetN;j++){ aBoard[j]=-1; }
      st=clock();
      if(cpu){ NQueen(0,i); }
      if(cpur){ NQueenR(0,i); }
      TimeFormat(clock()-st,t); 
      printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
    }
  }
  /** GPU */
  if(gpu||sgpu){
    if(!InitCUDA()){return 0;}
    int min=4;int targetN=18;
    struct timeval t0;struct timeval t1;
    int ss;int ms;int dd;
    long TOTAL;
    long results[2];//結果格納用
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    for(int i=min;i<=targetN;i++){
      gettimeofday(&t0,NULL);   // 計測開始
      if(gpu){
        solve_nqueen_cuda(i,results,steps);
        TOTAL=results[0];
        UNIQUE=results[1];
      }else if(sgpu){
        TOTAL=sgpu_solve_nqueen_cuda(i,steps);
      }
      gettimeofday(&t1,NULL);   // 計測終了
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
      printf("%2d:%13ld%16ld%4.2d:%02d:%02d:%02d.%02d\n", i,TOTAL,Unique,dd,hh,mm,ss,ms);
    }
  }
  return 0;
}
