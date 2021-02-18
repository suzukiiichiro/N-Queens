/**
 CUDAで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイル
 $ nvcc CUDA12_N-Queen.cu -o CUDA12_N-Queen

 実行
 $ ./CUDA12_N-Queen (-c|-r|-g)
                    -c:cpu -r cpu再帰 -g GPU

 １２．対称解除法の最適化

 実行結果

$ nvcc CUDA12_N-Queen.cu  && ./a.out -r
１２．CPUR 再帰 対称解除法の最適化
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.00
12:        14200            1787            0.00
13:        73712            9233            0.01
14:       365596           45752            0.07
15:      2279184          285053            0.40
16:     14772512         1846955            2.61
17:     95815104        11977939           18.05


$ nvcc CUDA12_N-Queen.cu  && ./a.out -c
１２．CPU 非再帰 対称解除法の最適化
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.00
12:        14200            1787            0.00
13:        73712            9233            0.01
14:       365596           45752            0.06
15:      2279184          285053            0.34
16:     14772512         1846955            2.24
17:     95815104        11977939           15.72


$ nvcc CUDA12_N-Queen.cu  && ./a.out -c
１２．GPU 非再帰 対称解除法の最適化
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2               1  00:00:00:00.02
 5:           10               2  00:00:00:00.00
 6:            4               1  00:00:00:00.00
 7:           40               6  00:00:00:00.00
 8:           92              12  00:00:00:00.00
 9:          352              46  00:00:00:00.00
10:          724              92  00:00:00:00.01
11:         2680             341  00:00:00:00.03
12:        14200            1787  00:00:00:00.14
13:        73712            9233  00:00:00:00.62
14:       365596           45752  00:00:00:02.96
15:      2279184          285053  00:00:00:18.15
16:      1744912          218114  00:00:00:24.20
Segmentation fault: 11
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define THREAD_NUM		96
#define MAX 27
//変数宣言
long Total=0 ;      //合計解
long Unique=0;
int aBoard[MAX];
int COUNT2,COUNT4,COUNT8;
int BOUND1,BOUND2,TOPBIT,ENDBIT,SIDEMASK,LASTMASK;


//関数宣言 GPU
__device__ int symmetryOps(int si,int *d_aBoard,int BOUND1,int BOUND2,int TOPBIT,int ENDBIT);
__global__ void nqueen_cuda_backTrack2(long *d_results,int *d_aBoard,int size,int mask,int row,int left,int down,int right,int BOUND1,int BOUND2,int SIDEMASK,int LASTMASK,int TOPBIT,int ENDBIT);
__global__ void nqueen_cuda_backTrack1(long *d_results,int *d_aBoard,int size,int mask,int row,int left,int down,int right,int BOUND1);
void solve_nqueen_cuda(int si,int mask,long result[2],int steps);
//関数宣言 SGPU
__global__ void sgpu_cuda_kernel(int size,int mark,unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,unsigned int* results,int totalCond);
long long sgpu_solve_nqueen_cuda(int size,int steps);
//関数宣言 CPU
void TimeFormat(clock_t utime,char *form);
long getUnique();
long getTotal();
void symmetryOps(int si);
void backTrack2_NR(int si,int mask,int y,int l,int d,int r);
void backTrack1_NR(int si,int mask,int y,int l,int d,int r);
void NQueen(int size,int mask);
void backTrack2(int si,int mask,int y,int l,int d,int r);
void backTrack1(int si,int mask,int y,int l,int d,int r);
void NQueenR(int size,int mask);
//
__device__
int symmetryOps(int si,int *d_aBoard,int BOUND1,int BOUND2,int TOPBIT,int ENDBIT){
      int own,ptn,you,bit;
  //90度回転
  if(d_aBoard[BOUND2]==1){ own=1; ptn=2;
    while(own<=si-1){ bit=1; you=si-1;
      while((d_aBoard[you]!=ptn)&&(d_aBoard[own]>=bit)){ bit<<=1; you--; }
      if(d_aBoard[own]>bit){ return 0; } if(d_aBoard[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>si-1){ return 2; }
  }
  //180度回転
  if(d_aBoard[si-1]==ENDBIT){ own=1; you=si-1-1;
    while(own<=si-1){ bit=1; ptn=TOPBIT;
      while((d_aBoard[you]!=ptn)&&(d_aBoard[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(d_aBoard[own]>bit){ return 0; } if(d_aBoard[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>si-1){ return 4; }
  }
  //270度回転
  if(d_aBoard[BOUND1]==TOPBIT){ own=1; ptn=TOPBIT>>1;
    while(own<=si-1){ bit=1; you=0;
      while((d_aBoard[you]!=ptn)&&(d_aBoard[own]>=bit)){ bit<<=1; you++; }
      if(d_aBoard[own]>bit){ return 0; } if(d_aBoard[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  return 8; 

}
__global__
void nqueen_cuda_backTrack2(long *d_results,int *d_aBoard,int size,int mask,int row,int left,int down,int right,int BOUND1,int BOUND2,int SIDEMASK,int LASTMASK,int TOPBIT,int ENDBIT){
  int bitmap,bit;
  int b[100], *p=b;
  int sizeE=size-1;
  int odd=size&1; //奇数:1 偶数:0
  for(int i=0;i<(1+odd);++i){
    bitmap=0;
    if(0==i){
      int half=size>>1; // size/2
      bitmap=(1<<half)-1;
    }else{
      bitmap=1<<(size>>1);
    }
      mais1:bitmap=mask&~(left|down|right);
      // 【枝刈り】
      if(row==sizeE){
        if(bitmap){
          //【枝刈り】 最下段枝刈り
          if((bitmap&LASTMASK)==0){
            d_aBoard[row]=bitmap; //symmetryOpsの時は代入します。
            int s=symmetryOps(size,d_aBoard,BOUND1,BOUND2,TOPBIT,ENDBIT);
            if(s!=0){
              //print(size); //print()でTOTALを++しない
              //ホストに戻す配列にTOTALを入れる
              //スレッドが１つの場合は配列は１個
              d_results[1]++; 
              d_results[0]+=s;   //対称解除で得られた解数を加算
            }
           
          }
        }
      }else{
        //【枝刈り】上部サイド枝刈り
        if(row<BOUND1){
          bitmap&=~SIDEMASK;
          //【枝刈り】下部サイド枝刈り
        }else if(row==BOUND2){
          if(!(down&SIDEMASK)){
            goto volta;
          }
          if((down&SIDEMASK)!=SIDEMASK){
            bitmap&=SIDEMASK;
          }
        }
        if(bitmap){
          outro:bitmap^=d_aBoard[row]=bit=-bitmap&bitmap;
          if(bitmap){
            *p++=left;
            *p++=down;
            *p++=right;
          }
          *p++=bitmap;
          row++;
          left=(left|bit)<<1;
          down=down|bit;
          right=(right|bit)>>1;
          goto mais1;
          //Backtrack2(y+1, (left | bit)<<1, down | bit, (right | bit)>>1);
          volta:if(p<=b)
          return;
          row--;
          bitmap=*--p;
          if(bitmap){
            right=*--p;
            down=*--p;
            left=*--p;
            goto outro;
          }else{
            goto volta;
          }
        }
      }
      goto volta;
      }


}
//
__global__
void nqueen_cuda_backTrack1(long *d_results,int *d_aBoard,int size,int mask,int row,int left,int down,int right,int BOUND1){
    int bitmap,bit;
    int b[100], *p=b;
    int sizeE=size-1;
    int odd=size&1; //奇数:1 偶数:0
    for(int i=0;i<(1+odd);++i){
      bitmap=0;
      if(0==i){
        int half=size>>1; // size/2
        bitmap=(1<<half)-1;
      }else{
        bitmap=1<<(size>>1);
      }
      b1mais1:bitmap=mask&~(left|down|right);
      //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      if(row==sizeE){
        if(bitmap){
            d_results[1]++; 
            d_results[0]+=8; 
        }
      }else{
        //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
        // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
        if(row<BOUND1) {
          bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
        }
        if(bitmap){
          b1outro:bitmap^=d_aBoard[row]=bit=-bitmap&bitmap;
          if(bitmap){
            *p++=left;
            *p++=down;
            *p++=right;
          }
          *p++=bitmap;
          row++;
          left=(left|bit)<<1;
          down=down|bit;
          right=(right|bit)>>1;
          goto b1mais1;
          //Backtrack1(y+1, (left | bit)<<1, down | bit, (right | bit)>>1);
          b1volta:if(p<=b)
          return;
          row--;
          bitmap=*--p;
          if(bitmap){
            right=*--p;
            down=*--p;
            left=*--p;
            goto b1outro;
          }else{
            goto b1volta;
          }
        }
      }
      goto b1volta;
    }

}
//
void solve_nqueen_cuda(int si,int mask,long results[2],int steps){
    int BOUND1,BOUND2,TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
    //メモリ登録
    long *h_results;
    int *h_aBoard;
    cudaMallocHost((void**)&h_results,sizeof(long)*steps);
    cudaMallocHost((void**)&h_aBoard,sizeof(int)*MAX);
    long *d_results;
    int *d_aBoard;
    cudaMalloc((void**)&d_results,sizeof(long)*steps);
    cudaMalloc((void**)&d_aBoard,sizeof(int)*MAX);
    //ロジック 
    int bit;
    TOPBIT=1<<(si-1);
    h_aBoard[0]=1;
    for(BOUND1=2;BOUND1<si-1;BOUND1++){
      h_aBoard[1]=bit=(1<<BOUND1);
      //host to device
      cudaMemcpy(d_results,h_results,
          sizeof(long)*steps,cudaMemcpyHostToDevice);
      cudaMemcpy(d_aBoard,h_aBoard,
          sizeof(int)*MAX,cudaMemcpyHostToDevice);
      nqueen_cuda_backTrack1<<<1,1>>>(d_results,d_aBoard,si,mask,2,(2|bit)<<1,(1|bit),(bit>>1),BOUND1);
      cudaMemcpy(h_results,d_results,
          sizeof(long)*steps,cudaMemcpyDeviceToHost);
   
    }
    SIDEMASK=LASTMASK=(TOPBIT|1);
    ENDBIT=(TOPBIT>>1);
    for(BOUND1=1,BOUND2=si-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
      h_aBoard[0]=bit=(1<<BOUND1);
      //host to device
      cudaMemcpy(d_aBoard,h_aBoard,
          sizeof(int)*MAX,cudaMemcpyHostToDevice);
      cudaMemcpy(d_results,h_results,
          sizeof(long)*steps,cudaMemcpyHostToDevice);
      //実行
      nqueen_cuda_backTrack2<<<1,1>>>(d_results,d_aBoard,si,mask,1,bit<<1,bit,bit>>1,BOUND1,BOUND2,SIDEMASK,LASTMASK,TOPBIT,ENDBIT);
      //device to host
      cudaMemcpy(h_results,d_results,
          sizeof(long)*steps,cudaMemcpyDeviceToHost);
      LASTMASK|=LASTMASK>>1|LASTMASK<<1;
      ENDBIT>>=1;
    }
    //解を代入
    results[0]=h_results[0];
    results[1]=h_results[1];
    //メモリ解放
    cudaFreeHost(h_results);
    cudaFreeHost(h_aBoard);
    cudaFree(d_aBoard);
    cudaFree(d_results);
}
//
__global__ 
void sgpu_cuda_kernel(int size,int mark,unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,unsigned int* results,int totalCond){
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
long getUnique(){
  return COUNT2+COUNT4+COUNT8;
}
//
long getTotal(){
  return COUNT2*2+COUNT4*4+COUNT8*8;
}
//
void symmetryOps(int si){
  int own,ptn,you,bit;
  //90度回転
  if(aBoard[BOUND2]==1){ own=1; ptn=2;
    while(own<=si-1){ bit=1; you=si-1;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you--; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>si-1){ COUNT2++; return; }
  }
  //180度回転
  if(aBoard[si-1]==ENDBIT){ own=1; you=si-1-1;
    while(own<=si-1){ bit=1; ptn=TOPBIT;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>si-1){ COUNT4++; return; }
  }
  //270度回転
  if(aBoard[BOUND1]==TOPBIT){ own=1; ptn=TOPBIT>>1;
    while(own<=si-1){ bit=1; you=0;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you++; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  COUNT8++;
}
//CPU 非再帰版 backTrack2
void backTrack2_NR(int size,int mask,int row,int left,int down,int right){
  int bitmap,bit;
  int b[100], *p=b;
  int sizeE=size-1;
  int odd=size&1; //奇数:1 偶数:0
  for(int i=0;i<(1+odd);++i){
    bitmap=0;
    if(0==i){
      int half=size>>1; // size/2
      bitmap=(1<<half)-1;
    }else{
      bitmap=1<<(size>>1);
      // down[1]=bitmap;
      // right[1]=(bitmap>>1);
      // left[1]=(bitmap<<1);
      // pnStack=aStack+1;
      // *pnStack++=0;
    }
mais1:bitmap=mask&~(left|down|right);
      // 【枝刈り】
      //if(row==size){
      if(row==sizeE){
        //if(!bitmap){
        if(bitmap){
          //【枝刈り】 最下段枝刈り
          if((bitmap&LASTMASK)==0){
            aBoard[row]=bitmap; //symmetryOpsの時は代入します。
            symmetryOps(size);
          }
        }
      }else{
        //【枝刈り】上部サイド枝刈り
        if(row<BOUND1){
          bitmap&=~SIDEMASK;
          //【枝刈り】下部サイド枝刈り
        }else if(row==BOUND2){
          if(!(down&SIDEMASK))
            goto volta;
          if((down&SIDEMASK)!=SIDEMASK)
            bitmap&=SIDEMASK;
        }
        if(bitmap){
outro:bitmap^=aBoard[row]=bit=-bitmap&bitmap;
      if(bitmap){
        *p++=left;
        *p++=down;
        *p++=right;
      }
      *p++=bitmap;
      row++;
      left=(left|bit)<<1;
      down=down|bit;
      right=(right|bit)>>1;
      goto mais1;
      //Backtrack2(y+1, (left | bit)<<1, down | bit, (right | bit)>>1);
volta:if(p<=b)
        return;
      row--;
      bitmap=*--p;
      if(bitmap){
        right=*--p;
        down=*--p;
        left=*--p;
        goto outro;
      }else{
        goto volta;
      }
      }
    }
    goto volta;
  }
}
//CPU 非再帰版 backTrack
void backTrack1_NR(int size,int mask,int row,int left,int down,int right){
  int bitmap,bit;
  int b[100], *p=b;
  int sizeE=size-1;
  int odd=size&1; //奇数:1 偶数:0
  for(int i=0;i<(1+odd);++i){
    bitmap=0;
    if(0==i){
      int half=size>>1; // size/2
      bitmap=(1<<half)-1;
    }else{
      bitmap=1<<(size>>1);
      // down[1]=bitmap;
      // right[1]=(bitmap>>1);
      // left[1]=(bitmap<<1);
      // pnStack=aStack+1;
      // *pnStack++=0;
    }
b1mais1:bitmap=mask&~(left|down|right);
    //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
    //if(row==size){
    if(row==sizeE){
      //if(!bitmap){
      if(bitmap){
        // aBoard[row]=bitmap;
        //symmetryOps_bitmap(size);
        COUNT8++;
      }
    }else{
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      if(row<BOUND1) {
        bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
      }
      if(bitmap){
b1outro:bitmap^=aBoard[row]=bit=-bitmap&bitmap;
        if(bitmap){
          *p++=left;
          *p++=down;
          *p++=right;
        }
        *p++=bitmap;
        row++;
        left=(left|bit)<<1;
        down=down|bit;
        right=(right|bit)>>1;
        goto b1mais1;
    //Backtrack1(y+1, (left | bit)<<1, down | bit, (right | bit)>>1);
b1volta:if(p<=b)
      return;
    row--;
    bitmap=*--p;
        if(bitmap){
          right=*--p;
          down=*--p;
          left=*--p;
          goto b1outro;
        }else{
          goto b1volta;
        }
      }
    }
    goto b1volta;
  }
}
//CPU 非再帰版 ロジックメソッド
void NQueen(int size,int mask){
  int bit;
  TOPBIT=1<<(size-1);
  aBoard[0]=1;
  for(BOUND1=2;BOUND1<size-1;BOUND1++){
    aBoard[1]=bit=(1<<BOUND1);
    //backTrack1(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
    backTrack1_NR(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    //backTrack1(size,mask,1,bit<<1,bit,bit>>1);
    backTrack2_NR(size,mask,1,bit<<1,bit,bit>>1);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  }
}
//
void backTrack2(int size,int mask,int row,int left,int down,int right){
  int bit;
  int bitmap=mask&~(left|down|right);
  if(row==size-1){ 								// 【枝刈り】
    if(bitmap){
      if((bitmap&LASTMASK)==0){ 	//【枝刈り】 最下段枝刈り
        aBoard[row]=bitmap; //symmetryOpsの時は代入します。
        symmetryOps(size);
      }
    }
  }else{
    if(row<BOUND1){             	//【枝刈り】上部サイド枝刈り
      bitmap&=~SIDEMASK;
    }else if(row==BOUND2) {     	//【枝刈り】下部サイド枝刈り
      if((down&SIDEMASK)==0){ return; }
      if((down&SIDEMASK)!=SIDEMASK){ bitmap&=SIDEMASK; }
    }
    while(bitmap){
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
      backTrack2(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
}
//
void backTrack1(int size,int mask,int row,int left,int down,int right){
  int bit;
  int bitmap=mask&~(left|down|right);
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
  if(row==size-1) {
    if(bitmap){
      // aBoard[row]=bitmap;
      COUNT8++;
    }
  }else{
    //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
    // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
    if(row<BOUND1) {
      bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
    while(bitmap){
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
      backTrack1(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
}
//
//CPUR 再帰版 ロジックメソッド
void NQueenR(int size,int mask){
  int bit;
  TOPBIT=1<<(size-1);
  aBoard[0]=1;
  for(BOUND1=2;BOUND1<size-1;BOUND1++){
    aBoard[1]=bit=(1<<BOUND1);
    backTrack1(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    backTrack2(size,mask,1,bit<<1,bit,bit>>1);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
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
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  if(cpu){
    printf("\n\n１２．CPU 非再帰 対称解除法の最適化\n");
  }else if(cpur){
    printf("\n\n１２．CPUR 再帰 対称解除法の最適化\n");
  }else if(gpu){
    printf("\n\n１２．GPU 非再帰 対称解除法の最適化\n");
  }else if(sgpu){
    printf("\n\n１２．SGPU 非再帰 対称解除法の最適化\n");
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
      st=clock();
      //初期化は不要です
      //非再帰は-1で初期化
      // for(int j=0;j<=targetN;j++){ aBoard[j]=-1; }
      if(cpu){ NQueen(i,mask); }
      if(cpur){ NQueenR(i,mask); }
      TimeFormat(clock()-st,t); 
      printf("%2d:%13ld%16ld%s\n",i,getTotal(),getUnique(),t);
    }
  }
  if(gpu||sgpu){
    if(!InitCUDA()){return 0;}
    int min=4;int targetN=18;int mask;
    struct timeval t0;struct timeval t1;int ss;int ms;int dd;
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    long TOTAL,UNIQUE;
    long results[2];//結果格納用
    for(int i=min;i<=targetN;i++){
      gettimeofday(&t0,NULL);   // 計測開始
      if(gpu){
        mask=((1<<i)-1);
        solve_nqueen_cuda(i,mask,results,steps);
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
      printf("%2d:%13ld%16ld%4.2d:%02d:%02d:%02d.%02d\n", i,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
    }
  }
  return 0;
}
