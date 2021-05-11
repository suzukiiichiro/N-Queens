
/**
 CUDAで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイルと実行
 $ nvcc -O3 CUDA**_N-Queen.cu && ./a.out (-c|-r|-g|-s)
                    -c:cpu 
                    -r cpu再帰 
                    -g GPU 
                    -s SGPU(サマーズ版と思われる)



 ８．ビットマップ＋対称解除法＋枝刈り

 実行結果

$ nvcc -O3 CUDA08_N-Queen.cu  && ./a.out -r
８．CPUR 再帰 ビットマップ＋対称解除法＋枝刈り
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.00
12:        14200            1787            0.01
13:        73712            9233            0.07
14:       365596           45752            0.31
15:      2279184          285053            2.60
16:     14772512         1846955           14.94
17:     95815104        11977939         2:08.89

$ nvcc -O3 CUDA08_N-Queen.cu  && ./a.out -c
８．CPU 非再帰 ビットマップ＋対称解除法＋枝刈り
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.00
12:        14200            1787            0.01
13:        73712            9233            0.06
14:       365596           45752            0.30
15:      2279184          285053            2.16
16:     14772512         1846955           14.41
17:     95815104        11977939         1:48.61

$ nvcc -O3 CUDA08_N-Queen.cu  && ./a.out -g
８．GPU 非再帰 ビットマップ＋対称解除法＋枝刈り
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2               1  00:00:00:00.02
 5:           10               2  00:00:00:00.00
 6:            4               1  00:00:00:00.01
 7:           40               6  00:00:00:00.01
 8:           92              12  00:00:00:00.00
 9:          352              46  00:00:00:00.02
10:          724              92  00:00:00:00.02
11:         2680             341  00:00:00:00.07
12:        14200            1787  00:00:00:00.18
13:        73712            9233  00:00:00:00.47
14:       365596           45752  00:00:00:01.84
15:      2279184          285053  00:00:00:11.86
16:     14772512         1846955  00:00:01:15.51
17:     95815104        11977939  00:00:10:06.50
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
int down[2*MAX-1];  //CPU down:flagA 縦 配置フラグ　
int left[2*MAX-1];  //CPU left:flagB 斜め配置フラグ　
int right[2*MAX-1]; //CPU right:flagC 斜め配置フラグ　
/***07 aBoard*************************************/
unsigned int aBoard[MAX];//CPU,GPUで使用
/****************************************/
int aT[MAX];//CPUで使用
int aS[MAX];//CPUで使用
long TOTAL=0;//GPU,CPUで使用
/***07 uniq*************************************/
long UNIQUE=0;//GPU,CPUで使用
/****************************************/
int COUNT2,COUNT4,COUNT8;//CPUで使用
//関数宣言 GPU
/***07 d_uniq,t_aBoard,h_row追加に伴いコメント*************************************/
//void cuda_kernel(
//    int size,int mark,
//    unsigned int* t_down,unsigned int* t_left,unsigned int* t_right,
//    unsigned int* d_results,int totalCond,unsigned);
/****************************************/
/***07 d_uniq,t_aBoard,h_row追加*************************************/
__global__
void cuda_kernel(
    register int size,register int mark,
    unsigned int* t_down,unsigned int* t_left,unsigned int* t_right,
    unsigned int* d_results,unsigned int* d_uniq,int totalCond,unsigned int* t_aBoard,int h_row,int* aT,int* aS);
/****************************************/
long long solve_nqueen_cuda(int size,int steps);
void NQueenG(int size,int mask,int row,int steps);
__device__ int symmetryOps_bitmap_gpu(int si,unsigned int *d_aBoard,int *d_aT,int *d_aS);
//関数宣言 SGPU
__global__ void sgpu_cuda_kernel(int size,int mark,unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,unsigned int* results,int totalCond);
long long sgpu_solve_nqueen_cuda(int size,int steps);
//関数宣言 GPU
bool InitCUDA();
//関数宣言 CPU/GPU
__device__ __host__ void rotate_bitmap(int bf[],int af[],int si);
__device__ __host__ void vMirror_bitmap(int bf[],int af[],int si);
__device__ __host__ int intncmp(int lt[],int rt[],int n);
__device__ __host__ int rh(int a,int size);
//関数宣言
void TimeFormat(clock_t utime,char *form);
long getUnique();
long getTotal();
void symmetryOps_bitmap(int si);
//関数宣言 CPU
void solve_nqueen(int size,int mask, int row,int* left,int* down,int* right,int* bitmap);
void NQueen(int size,int mask);
//関数宣言 CPUR
void solve_nqueenr(int size,int mask, int row,int left,int down,int right);
void NQueenR(int size,int mask);
//関数宣言 通常版
void NQueenD(int size,int mask);
void NQueenDR(int size,int mask,int row,int left,int down,int right,int ex1,int ex2);
//
//
__global__ void sgpu_cuda_kernel(int size,int mark,unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,unsigned int* results,int totalCond){
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
//
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
  if(i==count){fprintf(stderr,"There is no device supporting CUDA 1.x.\n");return false;}
  cudaSetDevice(i);
  return true;
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
//
__device__ __host__
int rh(int a,int sz){
  int tmp=0;
  for(int i=0;i<=sz;i++){
    if(a&(1<<i)){ return tmp|=(1<<(sz-i)); }
  }
  return tmp;
}
//
__device__ __host__
void vMirror_bitmap(int bf[],int af[],int si){
  int score ;
  for(int i=0;i<si;i++) {
    score=bf[i];
    af[i]=rh(score,si-1);
  }
}
//
__device__ __host__
void rotate_bitmap(int bf[],int af[],int si){
  for(int i=0;i<si;i++){
    int t=0;
    for(int j=0;j<si;j++){
      t|=((bf[j]>>i)&1)<<(si-j-1); // x[j] の i ビット目を
    }
    af[i]=t;                        // y[i] の j ビット目にする
  }
}
//
__device__ __host__
int intncmp(unsigned int lt[],int rt[],int n){
  int rtn=0;
  for(int k=0;k<n;k++){
    rtn=lt[k]-rt[k];
    if(rtn!=0){
      break;
    }
  }
  return rtn;
}
//
long getUnique(){
  return COUNT2+COUNT4+COUNT8;
}
//
long getTotal(){
  return COUNT2*2+COUNT4*4+COUNT8*8;
}
//
void symmetryOps_bitmap(int si){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aT[i]=aBoard[i];}
  rotate_bitmap(aT,aS,si);    //時計回りに90度回転
  int k=intncmp(aBoard,aS,si);
  if(k>0)return;
  if(k==0){ nEquiv=2;}else{
    rotate_bitmap(aS,aT,si);  //時計回りに180度回転
    k=intncmp(aBoard,aT,si);
    if(k>0)return;
    if(k==0){ nEquiv=4;}else{
      rotate_bitmap(aT,aS,si);//時計回りに270度回転
      k=intncmp(aBoard,aS,si);
      if(k>0){ return;}
      nEquiv=8;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aS[i]=aBoard[i];}
  vMirror_bitmap(aS,aT,si);   //垂直反転
  k=intncmp(aBoard,aT,si);
  if(k>0){ return; }
  if(nEquiv>2){             //-90度回転 対角鏡と同等
    rotate_bitmap(aT,aS,si);
    k=intncmp(aBoard,aS,si);
    if(k>0){return;}
    if(nEquiv>4){           //-180度回転 水平鏡像と同等
      rotate_bitmap(aS,aT,si);
      k=intncmp(aBoard,aT,si);
      if(k>0){ return;}       //-270度回転 反対角鏡と同等
      rotate_bitmap(aT,aS,si);
      k=intncmp(aBoard,aS,si);
      if(k>0){ return;}
    }
  }
  if(nEquiv==2){COUNT2++;}
  if(nEquiv==4){COUNT4++;}
  if(nEquiv==8){COUNT8++;}
}
//

//
__device__
int symmetryOps_bitmap_gpu(int si,unsigned int *d_aBoard,int *d_aT,int *d_aS){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ d_aT[i]=d_aBoard[i];}
  rotate_bitmap(d_aT,d_aS,si);    //時計回りに90度回転
  int k=intncmp(d_aBoard,d_aS,si);
  if(k>0)return 0;
  if(k==0){ nEquiv=2;}else{
    rotate_bitmap(d_aS,d_aT,si);  //時計回りに180度回転
    k=intncmp(d_aBoard,d_aT,si);
    if(k>0)return 0;
    if(k==0){ nEquiv=4;}else{
      rotate_bitmap(d_aT,d_aS,si);//時計回りに270度回転
      k=intncmp(d_aBoard,d_aS,si);
      if(k>0){ return 0;}
      nEquiv=8;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ d_aS[i]=d_aBoard[i];}
  vMirror_bitmap(d_aS,d_aT,si);   //垂直反転
  k=intncmp(d_aBoard,d_aT,si);
  if(k>0){ return 0; }
  if(nEquiv>2){             //-90度回転 対角鏡と同等
    rotate_bitmap(d_aT,d_aS,si);
    k=intncmp(d_aBoard,d_aS,si);
    if(k>0){return 0;}
    if(nEquiv>4){           //-180度回転 水平鏡像と同等
      rotate_bitmap(d_aS,d_aT,si);
      k=intncmp(d_aBoard,d_aT,si);
      if(k>0){ return 0;}       //-270度回転 反対角鏡と同等
      rotate_bitmap(d_aT,d_aS,si);
      k=intncmp(d_aBoard,d_aS,si);
      if(k>0){ return 0;}
    }
  }
  return nEquiv;
  
}

//GPU
/***07 引数 追加に伴いコメント*********************/
//__global__ 
//void cuda_kernel(int size,int mark,unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,unsigned int* d_results,int totalCond)
/************************/
/***07 引数 d_uniq,t_aBoard,h_row追加 uniq,aBoardのため*********************/
__global__
void cuda_kernel(
    int size,
    int mark,
    unsigned int* totalDown,
    unsigned int* totalLeft,
    unsigned int* totalRight,
    unsigned int* d_results,
    unsigned int* d_uniq,
    register int totalCond,
    unsigned int* t_aBoard,
    int h_row,
    int* aT,
    int* aS)
{
  /************************/
  register const unsigned int mask=(1<<size)-1;
  register unsigned int total=0;
  /***07 uniq,aBoard追加*********************/
  register unsigned int unique=0;
  //int aT[MAX];
  //int aS[MAX];
  /************************/
  //row=0となってるが1行目からやっているわけではなく
  //mask行目以降からスタート 
  //n=8 なら mask==2 なので そこからスタート
  register int row=0;
  register unsigned int bit;
  //
  //スレッド
  //
  //ブロック内のスレッドID
  register unsigned const int tid=threadIdx.x;
  //グリッド内のブロックID
  register unsigned const int bid=blockIdx.x;
  //全体通してのID
  register unsigned const int idx=bid*blockDim.x+tid;
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
  /***07 aBoard,uniq追加*********************/
  /***07 shared に変更 **********************/
  __shared__ unsigned int usum[THREAD_NUM];
  /***07 registerに変更 *********************/
  register int c_aT[MAX];
  register int c_aS[MAX];
  register unsigned int c_aBoard[MAX];
  /************************/
  //
  //余分なスレッドは動かさない 
  //GPUはsteps数起動するがtotalCond以上は空回しする
  if(idx<totalCond){
    //totalDown,totalLeft,totalRightの情報を
    //down,left,rightに詰め直す 
    //CPU で詰め込んだ t_はsteps個あるが
    //ブロック内ではブロックあたりのスレッド数に限定
    //されるので idxでよい
    //
    /***07 aBoard追加*********************/
    for(int i=0;i<h_row;i++){
      //c_aBoard[tid][i]=t_aBoard[idx][i];   
      c_aBoard[i]=t_aBoard[idx*h_row+i]; //２次元配列だが1次元的に利用  
    }
    /************************/
    /**07 スカラー変数に置き換えた**********/
    register unsigned int bitmap_tid_row;
    register unsigned int down_tid_row;
    register unsigned int left_tid_row;
    register unsigned int right_tid_row;
    while(row>=0){
      //bitmap[tid][row]をスカラー変数に置き換え
      bitmap_tid_row=bitmap[tid][row];
      down_tid_row=down[tid][row];
      left_tid_row=left[tid][row];
      right_tid_row=right[tid][row];
    /***************************************/
      //
      //bitmap[tid][row]=00000000 クイーンを
      //どこにも置けないので1行上に戻る
      /**07 スカラー変数に置き換えた**********/
      //if(bitmap[tid][row]==0){
      if(bitmap_tid_row==0){
      /***************************************/
        row--;
      }else{
        //クイーンを置く
        //bitmap[tid][row]
        //  ^=bit
        //  =(-bitmap[tid][row]&bitmap[tid][row]);
        //置く場所があるかどうか
        /***07 aBoard追加*********************/
        bitmap[tid][row]
          ^=c_aBoard[row+h_row]
          =bit
          /**07 スカラー変数に置き換えた**********/
          //=(-bitmap[tid][row]&bitmap[tid][row]);       
          =(-bitmap_tid_row&bitmap_tid_row);       
          /***************************************/
        /************************/
        if((bit&mask)!=0){
          //最終行?最終行から１個前の行まで
          //無事到達したら 加算する
          if(row+1==mark){
           /***07 symmetryOpsの処理を追加*********************/
           int s=symmetryOps_bitmap_gpu(size,c_aBoard,c_aT,c_aS); 
           //int s=0;//=symmetryOps_bitmap_gpu(size,c_aBoard[tid],aT,aS); 
           if(s!=0){
           //print(size); //print()でTOTALを++しない
           //ホストに戻す配列にTOTALを入れる
           //スレッドが１つの場合は配列は１個
              unique++; 
              total+=s;   //対称解除で得られた解数を加算
           }
           /************************/
           /***07 symmetryOpsの処理追加に伴いコメント*********************/
           //total++;
           /************************/
            row--;
          }else{
            int rowP=row+1;
            /**07スカラー変数に置き換えてregister対応 ****/
            //down[tid][rowP]=down[tid][row]|bit;
            down[tid][rowP]=down_tid_row|bit;
            //left[tid][rowP]=(left[tid][row]|bit)<<1;
            left[tid][rowP]=(left_tid_row|bit)<<1;
            //right[tid][rowP]=(right[tid][row]|bit)>>1;
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
    /***07 uniq追加*********************/
    usum[tid]=unique;
    /************************/
  }else{
    //totalCond未満は空回しするのでtotalは加算しない
    sum[tid]=0;
    /***07 uniq追加*********************/
    usum[tid]=0;
    /************************/
  } 
  //__syncthreads()でブロック内のスレッド間の同期
  //全てのスレッドが__syncthreads()に辿り着くのを待つ
  __syncthreads();if(tid<64&&tid+64<THREAD_NUM){
    sum[tid]+=sum[tid+64];
    /***07 uniq追加*********************/
    usum[tid]+=usum[tid+64];
    /************************/
  }
  __syncthreads();if(tid<32){
    sum[tid]+=sum[tid+32];
    /***07 uniq追加*********************/
    usum[tid]+=usum[tid+32];
    /************************/
  } 
  __syncthreads();if(tid<16){
    sum[tid]+=sum[tid+16];
    /***07 uniq追加*********************/
    usum[tid]+=usum[tid+16];
    /************************/  
  } 
  __syncthreads();if(tid<8){
    sum[tid]+=sum[tid+8];
    /***07 uniq追加*********************/
    usum[tid]+=usum[tid+8];
    /************************/
  } 
  __syncthreads();if(tid<4){
    sum[tid]+=sum[tid+4];
    /***07 uniq追加*********************/
    usum[tid]+=usum[tid+4];
    /************************/  
  } 
  __syncthreads();if(tid<2){
    sum[tid]+=sum[tid+2];
    /***07 uniq追加*********************/
    usum[tid]+=usum[tid+2];
    /************************/  
  } 
  __syncthreads();if(tid<1){
    sum[tid]+=sum[tid+1];
    /***07 uniq追加*********************/
    usum[tid]+=usum[tid+1];
    /************************/  
  } 
  __syncthreads();if(tid==0){
    d_results[bid]=sum[0];
    /****07 uniq追加********************/
    d_uniq[bid]=usum[0];
    /************************/
  }
}
//
// GPU
long solve_nqueen_cuda(int size,int mask,int row,int n_left,int n_down,int n_right,int steps)
{
  //何行目からGPUで行くか。ここの設定は変更可能、設定値を多くするほどGPUで並行して動く
  /***08 クイーンを２行目まで固定で置くためmarkが3以上必要のためコメント*********************/
  //const unsigned int mark=size>11?size-10:2;
  /************************/
  /***08 クイーンを２行目まで固定で置くためmarkが3以上必要*********************/
  const unsigned int mark=size>12?size-10:3;
  /************************/  
  const unsigned int h_mark=row;
  long total=0;
  int totalCond=0;
  bool matched=false;
  //host
  unsigned int down[32];  down[row]=n_down;
  unsigned int right[32]; right[row]=n_right;
  unsigned int left[32];  left[row]=n_left;
  //bitmapを配列で持つことにより
  //stackを使わないで1行前に戻れる
  unsigned int bitmap[32];
  //bitmap[row]=(left[row]|down[row]|right[row]);
  /***07 aBoard追加に伴いbit処理をGPU*********************/
  bitmap[row]=mask&~(left[row]|down[row]|right[row]);
  /************************/
  unsigned int bit;

  //unsigned int* totalDown=new unsigned int[steps];
  unsigned int* totalDown;
  cudaMallocHost((void**) &totalDown,sizeof(int)*steps);

  //unsigned int* totalLeft=new unsigned int[steps];
  unsigned int* totalLeft;
  cudaMallocHost((void**) &totalLeft,sizeof(int)*steps);

  //unsigned int* totalRight=new unsigned int[steps];
  unsigned int* totalRight;
  cudaMallocHost((void**) &totalRight,sizeof(int)*steps);

  //unsigned int* h_results=new unsigned int[steps];
  unsigned int* h_results;
  cudaMallocHost((void**) &h_results,sizeof(int)*steps);

  /***07 uniq,aBoard追加*********************/
  //unsigned int* h_uniq=new unsigned int[steps];
  unsigned int* h_uniq;
  cudaMallocHost((void**) &h_uniq,sizeof(int)*steps);

  //unsigned int* t_aBoard=new unsigned int[steps*mark];
  unsigned int* t_aBoard;
  cudaMallocHost((void**) &t_aBoard,sizeof(int)*steps*mark);
  /************************/
  //device
  unsigned int* downCuda;
  cudaMalloc((void**) &downCuda,sizeof(int)*steps);
  unsigned int* leftCuda;
  cudaMalloc((void**) &leftCuda,sizeof(int)*steps);
  unsigned int* rightCuda;
  cudaMalloc((void**) &rightCuda,sizeof(int)*steps);
  unsigned int* resultsCuda;
  cudaMalloc((void**) &resultsCuda,sizeof(int)*steps/THREAD_NUM);
  /***07 uniq,aBoard追加*********************/
  unsigned int* d_aT;
  cudaMalloc((void**) &d_aT,sizeof(int)*steps*MAX);
  unsigned int* d_aS;
  cudaMalloc((void**) &d_aS,sizeof(int)*steps*MAX);

  unsigned int* d_uniq;
  cudaMalloc((void**) &d_uniq,sizeof(int)*steps/THREAD_NUM);
  unsigned int* d_aBoard;
  cudaMalloc((void**) &d_aBoard,sizeof(int)*steps*mark);
  /************************/
  //12行目までは3行目までCPU->row==mark以下で 3行目までの
  //down,left,right情報を totalDown,totalLeft,totalRight
  //に格納
  //する->3行目以降をGPUマルチスレッドで実行し結果を取得
  //13行目以降はCPUで実行する行数が１個ずつ増えて行く
  //例えばn15だとrow=5までCPUで実行し、
  //それ以降はGPU(現在の設定だとGPUでは最大10行実行する
  //ようになっている)
  //while(row>=0) {
  register int rowP=0;
  while(row>=h_mark) {
    //bitmap[row]=00000000 クイーンを
    //どこにも置けないので1行上に戻る
    /***07 aBoard追加に伴いbit操作変更*********************/
    //06GPU こっちのほうが優秀
    if(bitmap[row]==0){ row--; }
    /************************/
    /***07 aBoard追加に伴いbit操作変更でコメント*********************/
    //06SGPU
    //if((bitmap[row]&mask)==mask){row--;}
    /************************/
    else{//おける場所があれば進む
      //06SGPU
      /***07 aBoard追加に伴いbit操作変更でコメント*********************/
      //bit=(bitmap[row]+1)&~bitmap[row];
      //bitmap[row]|=bit;
      /************************/
      //06GPU こっちのほうが優秀
      //bitmap[row]^=bit=(-bitmap[row]&bitmap[row]); //クイーンを置く
      /***07 aBoard追加*********************/
      bitmap[row]^=aBoard[row]=bit=(-bitmap[row]&bitmap[row]);
      /************************/ 
      if((bit&mask)!=0){//置く場所があれば先に進む
        rowP=row+1;
        down[rowP]=down[row]|bit;
        left[rowP]=(left[row]|bit)<<1;
        right[rowP]=(right[row]|bit)>>1;
        /***07 aBoard追加に伴いbit操作変更でコメント*********************/
        //bitmap[rowP]=(down[rowP]|left[rowP]|right[rowP]);
        /************************/
        /***07 aBoard追加に伴いbit操作変更*********************/
        bitmap[rowP]=mask&~(down[rowP]|left[rowP]|right[rowP]);
        /************************/
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
          /***07 aBoard追加*********************/
          for(int i=0;i<mark;i++){
            //t_aBoard[totalCond][i]=aBoard[i];
            t_aBoard[totalCond*mark+i]=aBoard[i];
          }
          /************************/
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
              /***07 uniq追加*********************/
              cudaMemcpy(h_uniq,d_uniq,
                  sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
              /************************/
              for(int col=0;col<steps/THREAD_NUM;col++){
                total+=h_results[col];
                /****07 uniq追加********************/
                UNIQUE+=h_uniq[col];
                /************************/                                        
              }
              matched=false;
            }
            cudaMemcpy(downCuda,totalDown,
                sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            cudaMemcpy(leftCuda,totalLeft,
                sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            cudaMemcpy(rightCuda,totalRight,
                sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            /***07 aBoard追加*********************/
            cudaMemcpy(d_aBoard,t_aBoard,
                sizeof(int)*totalCond*mark,cudaMemcpyHostToDevice);
            /************************/
            /** backTrack+bitmap*/
            //size-mark は何行GPUを実行するか totalCondはスレッド数
            /***07 d_uniq,d_aBoard,row追加に伴いコメント*********************/
            //cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
            //  >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
            /************************/
            /***07 d_uniq,d_aBoard,row追加*********************/
            cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
              >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,row,aT,aS);
            /************************/          
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
    /***07 uniq追加*********************/
    cudaMemcpy(h_uniq,d_uniq,
        sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    /************************/
   
    for(int col=0;col<steps/THREAD_NUM;col++){
      total+=h_results[col];
      /***07 uniq追加*********************/
      UNIQUE+=h_uniq[col];
      /************************/    
    }
    matched=false;
  }
  cudaMemcpy(downCuda,totalDown,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(leftCuda,totalLeft,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(rightCuda,totalRight,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  /***07 aBoard追加*********************/
  cudaMemcpy(d_aBoard,t_aBoard,
      sizeof(int)*totalCond*mark,cudaMemcpyHostToDevice);
  /************************/ 
  /** backTrack+bitmap*/
  //size-mark は何行GPUを実行するか totalCondはスレッド数
  //steps数の数だけマルチスレッドで起動するのだが、実際に計算が行われるのは
  //totalCondの数だけでそれ以外は空回しになる
  /***07 d_uniq,d_aBoard,mark追加に伴いコメント*********************/   
  //cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
  //  >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
  /***07 d_uniq,d_aBoard,mark追加*********************/  
  cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
    >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,mark,aT,aS);
  /************************/
  cudaMemcpy(h_results,resultsCuda,
      sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  /***07 uniq追加*********************/
  cudaMemcpy(h_uniq,d_uniq,
      sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  /************************/   
  for(int col=0;col<steps/THREAD_NUM;col++){
    total+=h_results[col];
    /***07 uniq追加*********************/
    UNIQUE+=h_uniq[col];
    /************************/    
  }
  //
  cudaFree(downCuda);
  cudaFree(leftCuda);
  cudaFree(rightCuda);
  cudaFree(resultsCuda);
  /***07 uniq,aBoard追加 cudaFreeHostへ変更**/
  cudaFree(d_uniq);
  cudaFree(d_aBoard);
  //delete[] totalDown;
  cudaFreeHost(totalDown);
  //delete[] totalLeft;
  cudaFreeHost(totalLeft);
  //delete[] totalRight;
  cudaFreeHost(totalRight);
  //delete[] h_results;
  cudaFreeHost(h_results);
  //delete[] h_uniq;
  cudaFreeHost(h_uniq);
  //delete[] t_aBoard;
  cudaFreeHost(t_aBoard);
  /************************/
  return total;
}
//
void NQueenG(int size,int steps){
  int bit=0;
  int mask=(1<<size)-1;
  int sizeE=size-1;
  //偶数、奇数共通 右側半分だけクイーンを置く
  int lim=(size%2==0)?size/2:sizeE/2;
  for(int col=0;col<lim;col++){
    bit=aBoard[0]=(1<<col);
    TOTAL+=solve_nqueen_cuda(size,mask,1,bit<<1,bit,bit>>1,steps);
  }
  //ミラーなのでTOTALを２倍する
  //TOTAL=TOTAL*2;
  //奇数の場合はさらに中央にクイーンを置く
  if(size%2==1){
    /***08 奇数の場合の処理、２行目左側半分にクイーンを置けない処理追加のためコメント*********************/
    //bit=(1<<(sizeE)/2);
    //TOTAL+=solve_nqueen_cuda(size,mask,1,bit<<1,bit,bit>>1,steps);
    /************************/
     /***08 奇数の場合の処理、２行目左側半分にクイーンを置けない処理追加*********************/
    int col=(sizeE)/2;
    //1行目はクイーンを中央に置く
    bit=aBoard[0]=(1<<col);
    int left=bit<<1;
    int down=bit;
    int right=bit>>1;
    for(int col_j=0;col_j<(size/2)-1;col_j++){
    //1行目にクイーンが中央に置かれた場合は2行目の左側半分にクイーンを置けない
    //0001000
    //xxxdroo  左側半分にクイーンを置けないがさらに1行目のdown,rightもクイーンを置けないので (size/2)-1となる
      //2行目にクイーンを置く
      aBoard[1]=bit=(1<<col_j);
      TOTAL+=solve_nqueen_cuda(size,mask,2,(left|bit)<<1,(down|bit),(right|bit)>>1,steps);
    }
    /************************/
  }
}
//
//
//CPU 非再帰版 ロジックメソッド
void solve_nqueen(int size,int mask, int row,int* left,int* down,int* right,int* bitmap){
    unsigned int bit;
    unsigned int sizeE=size-1;
    int mark=row;
    //固定していれた行より上はいかない
    while(row>=mark){//row=1 row>=1, row=2 row>=2
      if(bitmap[row]==0){
        --row;
      }else{
        bitmap[row]^=aBoard[row]=bit=(-bitmap[row]&bitmap[row]); 
        if((bit&mask)!=0){
          if(row==sizeE){
            symmetryOps_bitmap(size);
            --row;
            continue;
          }else{
            int n=row++;
            left[row]=(left[n]|bit)<<1;
            down[row]=down[n]|bit;
            right[row]=(right[n]|bit)>>1;
            bitmap[row]=mask&~(left[row]|down[row]|right[row]);
            continue;
          }
        }else{
           --row;
           continue;
        }
      }  
    }
}
void NQueen(int size,int mask){
  register int bitmap[size];
  register int down[size],right[size],left[size];
  register int bit;
  if(size<=0||size>32){return;}
  int sizeE=size-1;
  bit=0;
  bitmap[0]=mask;
  down[0]=left[0]=right[0]=0;
  //偶数、奇数ともに右半分にクイーンを置く
  for(int col=0;col<size/2;col++){
    //ex n=6 xxxooo n=7 xxxxooo 
    aBoard[0]=bit=(1<<col);
    down[1]=bit;//再帰の場合は down,left,right,bitmapは現在の行だけで良いが
    left[1]=bit<<1;//非再帰の場合は全行情報を配列に入れて行の上がり下がりをする
    right[1]=bit>>1;
    bitmap[1]=mask&~(left[1]|down[1]|right[1]);
    solve_nqueen(size,mask,1,left,down,right,bitmap);
  }
  //奇数については中央にもクイーンを置く
  if(size%2==1){
    int col=(sizeE)/2;
    //1行目はクイーンを中央に置く
    bit=aBoard[0]=(1<<col);
    down[1]=bit;//再帰の場合は down,left,right,bitmapは現在の行だけで良いが
    left[1]=bit<<1;//非再帰の場合は全行情報を配列に入れて行の上がり下がりをする
    right[1]=bit>>1;
    bitmap[1]=mask&~(left[1]|down[1]|right[1]);
    for(int col_j=0;col_j<(size/2)-1;col_j++){
    //1行目にクイーンが中央に置かれた場合は2行目の左側半分にクイーンを置けない
    //0001000
    //xxxdroo  左側半分にクイーンを置けないがさらに1行目のdown,rightもクイーンを置けないので (size/2)-1となる
      //2行目にクイーンを置く
      aBoard[1]=bit=(1<<col_j);
      down[2]=bit;//再帰の場合は down,left,right,bitmapは現在の行だけで良いが
      left[2]=bit<<1;//非再帰の場合は全行情報を配列に入れて行の上がり下がりをする
      right[2]=bit>>1;
      bitmap[2]=mask&~(left[2]|down[2]|right[2]);
      solve_nqueen(size,mask,2,left,down,right,bitmap);
    }
  }
}
//
//CPUR 再帰版 ロジックメソッド
void solve_nqueenr(int size,int mask, int row,int left,int down,int right){
 int bitmap=0;
 int bit=0;
 int sizeE=size-1;
 bitmap=(mask&~(left|down|right));
 if(row==sizeE){
   if(bitmap){
     aBoard[row]=(-bitmap&bitmap);
     symmetryOps_bitmap(size);
   }
  }else{
    while(bitmap){
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
      solve_nqueenr(size,mask,row+1,(left|bit)<<1, down|bit,(right|bit)>>1);
    }
  }
}
//
//CPUR 再帰版 ロジックメソッド
void NQueenR(int size,int mask){
  int bit=0;
  int sizeE=size-1;
  //偶数、奇数ともに右半分にクイーンを置く
  for(int col=0;col<size/2;col++){
    //ex n=6 xxxooo n=7 xxxxooo 
    bit=aBoard[0]=(1<<col);
    solve_nqueenr(size,mask,1,bit<<1,bit,bit>>1);
  }
  //奇数については中央にもクイーンを置く
  if(size%2==1){
    int col=(sizeE)/2;
    //1行目はクイーンを中央に置く
    bit=aBoard[0]=(1<<col);
    int left=bit<<1;
    int down=bit;
    int right=bit>>1;
    for(int col_j=0;col_j<(size/2)-1;col_j++){
    //1行目にクイーンが中央に置かれた場合は2行目の左側半分にクイーンを置けない
    //0001000
    //xxxdroo  左側半分にクイーンを置けないがさらに1行目のdown,rightもクイーンを置けないので (size/2)-1となる
      //2行目にクイーンを置く
      aBoard[1]=bit=(1<<col_j);
      solve_nqueenr(size,mask,2,(left|bit)<<1,(down|bit),(right|bit)>>1);
    }
  }
}
//
//CPU 非再帰版 ロジックメソッド
void NQueenD(int size,int mask){
  int aStack[size];
  register int* pnStack;
  register int row=0;
  register int bit;
  register int bitmap;
  int odd=size&1; //奇数:1 偶数:0
  int sizeE=size-1;
  /* センチネルを設定-スタックの終わりを示します*/
  aStack[0]=-1;
  for(int i=0;i<(1+odd);++i){
    bitmap=0;
    if(0==i){
      int half=size>>1; // size/2
      bitmap=(1<<half)-1;
      pnStack=aStack+1;
    }else{
      bitmap=1<<(size>>1);
      down[1]=bitmap;
      right[1]=(bitmap>>1);
      left[1]=(bitmap<<1);
      pnStack=aStack+1;
      *pnStack++=0;
    }
    while(true){
      if(bitmap){
        bitmap^=aBoard[row]=bit=(-bitmap&bitmap); 
        if(row==sizeE){
          /* 対称解除法の追加 */
          //TOTAL++;
          symmetryOps_bitmap(size); 
          bitmap=*--pnStack;
          --row;
          continue;
        }else{
          int n=row++;
          left[row]=(left[n]|bit)<<1;
          down[row]=down[n]|bit;
          right[row]=(right[n]|bit)>>1;
          *pnStack++=bitmap;
          bitmap=mask&~(left[row]|down[row]|right[row]);
          continue;
        }
      }else{ 
        bitmap=*--pnStack;
        if(pnStack==aStack){ break ; }
        --row;
        continue;
      }
    }
  }
}
//CPUR 再帰版　ロジックメソッド
void NQueenDR(int size,int mask,int row,int left,int down,int right,int ex1,int ex2){
  int bit;
  int bitmap=(mask&~(left|down|right|ex1));
  if(row==size){
    // TOTAL++;
    symmetryOps_bitmap(size);
  }else{
    while(bitmap){
      if(ex2!=0){
      	//奇数個の１回目は真ん中にクイーンを置く
        bitmap^=aBoard[row]=bit=(1<<(size/2+1));
      }else{
        bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
      }
     	//ここは２行目の処理。ex2を前にずらし除外するようにする
      NQueenDR(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1,ex2,0);
      //ex2の除外は一度適用したら（１行目の真ん中にクイーンが来る場合）もう適用
      //しないので0にする
      ex2=0;
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
    printf("Usage: %s [-c|-g|-r]\n",argv[0]);
    printf("  -c: CPU only\n");
    printf("  -r: CPUR only\n");
    printf("  -g: GPU only\n");
    printf("  -s: SGPU only\n");
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  if(cpu){
    printf("\n\n８．CPU 非再帰 ビットマップ＋対称解除法＋枝刈り\n");
  }else if(cpur){
    printf("\n\n８．CPUR 再帰 ビットマップ＋対称解除法＋枝刈り\n");
  }else if(gpu){
    printf("\n\n８．GPU 非再帰 ビットマップ＋対称解除法＋枝刈り\n");
  }else if(sgpu){
    printf("\n\n８．SGPU 非再帰 バックトラック＋ビットマップ\n");
  }
  if(cpu||cpur){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    clock_t st;           //速度計測用
    char t[20];           //hh:mm:ss.msを格納
    int min=4; int targetN=17;
    int mask;
    for(int i=min;i<=targetN;i++){
      //TOTAL=0; UNIQUE=0;
      COUNT2=COUNT4=COUNT8=0;
      mask=(1<<i)-1;
      //除外デフォルト ex 00001111  000001111
      //これだと１行目の右側半分にクイーンが置けない
      int excl=(1<<((i/2)^0))-1;
      //対象解除は右側にクイーンが置かれた場合のみ判定するので
      //除外を反転させ１行目の左側半分にクイーンを置けなくする
      //ex 11110000 111100000 
      if(i%2){
        excl=excl<<(i/2+1);
      }else{
        excl=excl<<(i/2);
      }
      //偶数の場合
      //１行目の左側半分にクイーンを置けないようにする
      //奇数の場合
      //１行目の左側半分にクイーンを置けないようにする
      //１行目にクイーンが中央に置かれた場合は２行目の左側半分にクイーンを置けない
      //ようにする
      //最終的に個数を倍にするのは対象解除のミラー判定に委ねる
      st=clock();
      //初期化は不要です
      /** 非再帰は-1で初期化 */
      // for(int j=0;j<=targetN;j++){
      //   aBoard[j]=-1;
      // }
      //
      //再帰
      if(cpur){ 
        NQueenR(i,mask);
        //printf("通常版\n");
        //NQueenDR(i,mask,0,0,0,0,excl,i%2 ? excl : 0);//通常版
      }
      //非再帰
      if(cpu){ 
        NQueen(i,mask); 
        //printf("通常版\n");
        //NQueenD(i,mask);//通常版
      }
      //
      TimeFormat(clock()-st,t); 
      printf("%2d:%13ld%16ld%s\n",i,getTotal(),getUnique(),t);
    }
  }
  if(gpu||sgpu){
    if(!InitCUDA()){return 0;}
    int min=4;int targetN=17;
    struct timeval t0;struct timeval t1;int ss;int ms;int dd;
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    for(int i=min;i<=targetN;i++){
      TOTAL=0;
      UNIQUE=0;
      gettimeofday(&t0,NULL);   // 計測開始
      if(gpu){
        TOTAL=0;
        UNIQUE=0;
        NQueenG(i,steps);
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
      printf("%2d:%13ld%16ld%4.2d:%02d:%02d:%02d.%02d\n", i,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
    }
  }
  return 0;
}

