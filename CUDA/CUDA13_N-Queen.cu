/**
 CUDAで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイル
 $ nvcc CUDA13_N-Queen.cu -o CUDA13_N-Queen

 実行
 $ ./CUDA13_N-Queen (-c|-r|-g)
                    -c:cpu -r cpu再帰 -g GPU

 １３．並列処理 pthread


 【注意】
 cu(nvcc)のソースに参考のために 再帰・非再帰のpthread版を記載しましたが、
 cudaはpthreadをサポートしていないので、コンパイルは通りません。

 +590 行目のコメントアウトはそのためです。
 //iFbRet = pthread_create(&pth, NULL,&NQueenThread,NULL);

 pthreadの具体的なソースは、Cディレクトリの C13_N-Queen.cを見て下さい。
 こちらはきちんと動作します。
 【注意】



 実行結果

bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC13NR.c && ./a.out -r
１３．CPUR 再帰 並列処理 pthread
 N:           Total           Unique          dd:hh:mm:ss.ms
 4:               2                1          00:00:00:00.00
 5:              10                2          00:00:00:00.00
 6:               4                1          00:00:00:00.00
 7:              40                6          00:00:00:00.00
 8:              92               12          00:00:00:00.00
 9:             352               46          00:00:00:00.00
10:             724               92          00:00:00:00.00
11:            2680              341          00:00:00:00.00
12:           14200             1787          00:00:00:00.00
13:           73712             9233          00:00:00:00.00
14:          365596            45752          00:00:00:00.01
15:         2279184           285053          00:00:00:00.10
16:        14772512          1846955          00:00:00:00.65
17:        95815104         11977939          00:00:00:04.33 

bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC13NR.c && ./a.out -c
１３．CPU 非再帰 並列処理 pthread
 N:           Total           Unique          dd:hh:mm:ss.ms
 4:               2                1          00:00:00:00.00
 5:              10                2          00:00:00:00.00
 6:               4                1          00:00:00:00.00
 7:              40                6          00:00:00:00.00
 8:              92               12          00:00:00:00.00
 9:             352               46          00:00:00:00.00
10:             724               92          00:00:00:00.00
11:            2680              341          00:00:00:00.00
12:           14200             1787          00:00:00:00.00
13:           73712             9233          00:00:00:00.00
14:          365596            45752          00:00:00:00.01
15:         2279184           285053          00:00:00:00.10
16:        14772512          1846955          00:00:00:00.62
17:        95815104         11977939          00:00:00:03.15


$ nvcc CUDA13_N-Queen.cu  && ./a.out -g
１３．GPU 非再帰 並列処理 pthread

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <pthread.h>
//
#define THREAD_NUM		96
#define MAX 27
//

int NR;
// GPUで使います
long Total=0 ;      //合計解
long Unique=0;

//
//変数宣言
// pthreadはパラメータを１つしか渡せないので構造体に格納
//グローバル構造体
typedef struct {
  int size;
  int sizeE;
  long lTOTAL,lUNIQUE;
}GCLASS, *GClass;
GCLASS G;
//ローカル構造体
typedef struct{
  int BOUND1,BOUND2,TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
  int mask;
  int aBoard[MAX];
  long COUNT2[MAX],COUNT4[MAX],COUNT8[MAX];
}local ;
//関数宣言
void symmetryOps(local *l);
void backTrack2_NR(int y,int left,int down,int right,local *l);
void backTrack1_NR(int y,int left,int down,int right,local *l);
void backTrack2(int y,int left,int down,int right,local *l);
void backTrack1(int y,int left,int down,int right,local *l);
void *run(void *args);
void *NQueenThread();
void NQueen();
//
__global__ void solve_nqueen_cuda_kernel_bt_bm(
    int n,int mark,
    unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,
    unsigned int* results,int totalCond){
  const int tid=threadIdx.x,bid=blockIdx.x,idx=bid*blockDim.x+tid;
  __shared__ unsigned int down[THREAD_NUM][10],left[THREAD_NUM][10],right[THREAD_NUM][10],
  bitmap[THREAD_NUM][10],sum[THREAD_NUM];
  const unsigned int mask=(1<<n)-1;int total=0,i=0;unsigned int bit;
  if(idx<totalCond){
    down[tid][i]=totalDown[idx];
    left[tid][i]=totalLeft[idx];
    right[tid][i]=totalRight[idx];
    bitmap[tid][i]=down[tid][i]|left[tid][i]|right[tid][i];
    while(i>=0){
      if((bitmap[tid][i]&mask)==mask){i--;}
      else{
        bit=(bitmap[tid][i]+1)&~bitmap[tid][i];
        bitmap[tid][i]|=bit;
        if((bit&mask)!=0){
          if(i+1==mark){total++;i--;}
          else{
            down[tid][i+1]=down[tid][i]|bit;
            left[tid][i+1]=(left[tid][i]|bit)<<1;
            right[tid][i+1]=(right[tid][i]|bit)>>1;
            bitmap[tid][i+1]=(down[tid][i+1]|left[tid][i+1]|right[tid][i+1]);
            i++;
          }
        }else{i--;}
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
long long solve_nqueen_cuda(int n,int steps) {
  unsigned int down[32];unsigned int left[32];unsigned int right[32];
  unsigned int m[32];unsigned int bit;
  if(n<=0||n>32){return 0;}
  unsigned int* totalDown=new unsigned int[steps];
  unsigned int* totalLeft=new unsigned int[steps];
  unsigned int* totalRight=new unsigned int[steps];
  unsigned int* results=new unsigned int[steps];
  unsigned int* downCuda;unsigned int* leftCuda;unsigned int* rightCuda;
  unsigned int* resultsCuda;
  cudaMalloc((void**) &downCuda,sizeof(int)*steps);
  cudaMalloc((void**) &leftCuda,sizeof(int)*steps);
  cudaMalloc((void**) &rightCuda,sizeof(int)*steps);
  cudaMalloc((void**) &resultsCuda,sizeof(int)*steps/THREAD_NUM);
  const unsigned int mask=(1<<n)-1;
  const unsigned int mark=n>11?n-10:2;
  long long total=0;int totalCond=0;
  int i=0,j;down[0]=0;left[0]=0;right[0]=0;m[0]=0;bool computed=false;
  for(j=0;j<n/2;j++){
    bit=(1<<j);m[0]|=bit;
    down[1]=bit;left[1]=bit<<1;right[1]=bit>>1;
    m[1]=(down[1]|left[1]|right[1]);
    i=1;
    while(i>0){
      if((m[i]&mask)==mask){i--;}
      else{
        bit=(m[i]+1)&~m[i];m[i]|=bit;
        if((bit&mask)!=0){
          down[i+1]=down[i]|bit;left[i+1]=(left[i]|bit)<<1;right[i+1]=(right[i]|bit)>>1;
          m[i+1]=(down[i+1]|left[i+1]|right[i+1]);
          i++;
          if(i==mark){
            totalDown[totalCond]=down[i];totalLeft[totalCond]=left[i];totalRight[totalCond]=right[i];
            totalCond++;
            if(totalCond==steps){
              if(computed){
                cudaMemcpy(results,resultsCuda,sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
                for(int j=0;j<steps/THREAD_NUM;j++){total+=results[j];}
                computed=false;
              }
              cudaMemcpy(downCuda,totalDown,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              cudaMemcpy(leftCuda,totalLeft,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              cudaMemcpy(rightCuda,totalRight,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              /** backTrack+bitmap*/
              solve_nqueen_cuda_kernel_bt_bm<<<steps/THREAD_NUM,THREAD_NUM>>>(n,n-mark,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
              computed=true;totalCond=0;
            }
            i--;
          }
        }else{i --;}
      }
    }
  }
  if(computed){
    cudaMemcpy(results,resultsCuda,sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    for(int j=0;j<steps/THREAD_NUM;j++){total+=results[j];}
    computed=false;
  }
  cudaMemcpy(downCuda,totalDown,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(leftCuda,totalLeft,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(rightCuda,totalRight,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  /** backTrack+bitmap*/
  solve_nqueen_cuda_kernel_bt_bm<<<steps/THREAD_NUM,THREAD_NUM>>>(n,n-mark,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
  cudaMemcpy(results,resultsCuda,sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  for(int j=0;j<steps/THREAD_NUM;j++){total+=results[j];}	
  total*=2;
  if(n%2==1){
    computed=false;totalCond=0;bit=(1<<(n-1)/2);m[0]|=bit;
    down[1]=bit;left[1]=bit<<1;right[1]=bit>>1;
    m[1]=(down[1]|left[1]|right[1]);
    i=1;
    while(i>0){
      if((m[i]&mask)==mask){i--;}
      else{
        bit=(m[i]+1)&~m[i];m[i]|=bit;
        if((bit&mask)!=0){
          down[i+1]=down[i]|bit;left[i+1]=(left[i]|bit)<<1;right[i+1]=(right[i]|bit)>>1;
          m[i+1]=(down[i+1]|left[i+1]|right[i+1]);
          i++;
          if(i==mark){
            totalDown[totalCond]=down[i];totalLeft[totalCond]=left[i];totalRight[totalCond]=right[i];
            totalCond++;
            if(totalCond==steps){
              if(computed){
                cudaMemcpy(results,resultsCuda,sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
                for(int j=0;j<steps/THREAD_NUM;j++){total+=results[j];}
                computed=false;
              }
              cudaMemcpy(downCuda,totalDown,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              cudaMemcpy(leftCuda,totalLeft,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              cudaMemcpy(rightCuda,totalRight,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              /** backTrack+bitmap*/
              solve_nqueen_cuda_kernel_bt_bm<<<steps/THREAD_NUM,THREAD_NUM>>>(n,n-mark,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
              computed=true;totalCond=0;
            }
            i--;
          }
        }else{i --;}
      }
    }
    if(computed){
      cudaMemcpy(results,resultsCuda,sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
      for(int j=0;j<steps/THREAD_NUM;j++){total+=results[j];}
      computed=false;
    }
    cudaMemcpy(downCuda,totalDown,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
    cudaMemcpy(leftCuda,totalLeft,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
    cudaMemcpy(rightCuda,totalRight,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
    /** backTrack+bitmap*/
    solve_nqueen_cuda_kernel_bt_bm<<<steps/THREAD_NUM,THREAD_NUM>>>(n,n-mark,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
    cudaMemcpy(results,resultsCuda,sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    for(int j=0;j<steps/THREAD_NUM;j++){total+=results[j];}
  }
  cudaFree(downCuda);cudaFree(leftCuda);cudaFree(rightCuda);cudaFree(resultsCuda);
  delete[] totalDown;delete[] totalLeft;delete[] totalRight;delete[] results;
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
//
void symmetryOps(local *l){
  int own,ptn,you,bit;
  //90度回転
  if(l->aBoard[l->BOUND2]==1){ own=1; ptn=2;
    while(own<=G.sizeE){ bit=1; you=G.sizeE;
      while((l->aBoard[you]!=ptn)&&(l->aBoard[own]>=bit)){ bit<<=1; you--; }
      if(l->aBoard[own]>bit){ return; } if(l->aBoard[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>G.sizeE){ l->COUNT2[l->BOUND1]++; return; }
  }
  //180度回転
  if(l->aBoard[G.sizeE]==l->ENDBIT){ own=1; you=G.sizeE-1;
    while(own<=G.sizeE){ bit=1; ptn=l->TOPBIT;
      while((l->aBoard[you]!=ptn)&&(l->aBoard[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(l->aBoard[own]>bit){ return; } if(l->aBoard[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>G.sizeE){ l->COUNT4[l->BOUND1]++; return; }
  }
  //270度回転
  if(l->aBoard[l->BOUND1]==l->TOPBIT){ own=1; ptn=l->TOPBIT>>1;
    while(own<=G.sizeE){ bit=1; you=0;
      while((l->aBoard[you]!=ptn)&&(l->aBoard[own]>=bit)){ bit<<=1; you++; }
      if(l->aBoard[own]>bit){ return; } if(l->aBoard[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  l->COUNT8[l->BOUND1]++;
}
//CPU 非再帰版 backTrack2
void backTrack2_NR(int row,int left,int down,int right,local *l){
  int bitmap,bit;
  int b[100], *p=b;
  int odd=G.size&1; //奇数:1 偶数:0
  for(int i=0;i<(1+odd);++i){
    bitmap=0;
    if(0==i){
      int half=G.size>>1; // size/2
      bitmap=(1<<half)-1;
    }else{
      bitmap=1<<(G.size>>1);
      // down[1]=bitmap;
      // right[1]=(bitmap>>1);
      // left[1]=(bitmap<<1);
      // pnStack=aStack+1;
      // *pnStack++=0;
    }
mais1:bitmap=l->mask&~(left|down|right);
      // 【枝刈り】
      if(row==G.sizeE){
        if(bitmap){
          //【枝刈り】 最下段枝刈り
          if((bitmap&l->LASTMASK)==0){
            l->aBoard[row]=bitmap;
            symmetryOps(l);
          }
        }
      }else{
        //【枝刈り】上部サイド枝刈り
        if(row<l->BOUND1){
          bitmap&=~l->SIDEMASK;
          //【枝刈り】下部サイド枝刈り
        }else if(row==l->BOUND2){
          if(!(down&l->SIDEMASK))
            goto volta;
          if((down&l->SIDEMASK)!=l->SIDEMASK)
            bitmap&=l->SIDEMASK;
        }
        if(bitmap){
outro:bitmap^=l->aBoard[row]=bit=-bitmap&bitmap;
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
void backTrack1_NR(int row,int left,int down,int right,local *l){
  int bitmap,bit;
  int b[100], *p=b;
  int odd=G.size&1; //奇数:1 偶数:0
  for(int i=0;i<(1+odd);++i){
    bitmap=0;
    if(0==i){
      int half=G.size>>1; // size/2
      bitmap=(1<<half)-1;
    }else{
      bitmap=1<<(G.size>>1);
      // down[1]=bitmap;
      // right[1]=(bitmap>>1);
      // left[1]=(bitmap<<1);
      // pnStack=aStack+1;
      // *pnStack++=0;
    }
b1mais1:bitmap=l->mask&~(left|down|right);
        //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
        if(row==G.sizeE){
          if(bitmap){
            // l->aBoard[row]=bitmap;
            l->COUNT8[l->BOUND1]++;
          }
        }else{
          //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
          // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
          if(row<l->BOUND1) {
            bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
          }
          if(bitmap){
b1outro:bitmap^=l->aBoard[row]=bit=-bitmap&bitmap;
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
void backTrack2(int row,int left,int down,int right,local *l){
  int bit;
  int bitmap=l->mask&~(left|down|right);
  if(row==G.sizeE){ 								// 【枝刈り】
    if(bitmap){
      if((bitmap&l->LASTMASK)==0){ 	//【枝刈り】 最下段枝刈り
        l->aBoard[row]=bitmap;
        symmetryOps(l);
      }
    }
  }else{
    if(row<l->BOUND1){             	//【枝刈り】上部サイド枝刈り
      bitmap&=~l->SIDEMASK;
    }else if(row==l->BOUND2) {     	//【枝刈り】下部サイド枝刈り
      if((down&l->SIDEMASK)==0){ return; }
      if((down&l->SIDEMASK)!=l->SIDEMASK){ bitmap&=l->SIDEMASK; }
    }
    while(bitmap){
      bitmap^=l->aBoard[row]=bit=(-bitmap&bitmap);
      backTrack2(row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
}
//
void backTrack1(int row,int left,int down,int right,local *l){
  int bit;
  int bitmap=l->mask&~(left|down|right);
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
  if(row==G.sizeE) {
    if(bitmap){
      /* l->aBoard[row]=bitmap; */
      l->COUNT8[l->BOUND1]++;
    }
  }else{
    //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
    // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
    if(row<l->BOUND1) {
      bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
    while(bitmap){
      bitmap^=l->aBoard[row]=bit=(-bitmap&bitmap);
      backTrack1(row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
}
//
void *run(void *args){
  local *l=(local *)args;
  int bit=0;
  l->aBoard[0]=1;
  l->TOPBIT=1<<(G.sizeE);
  l->mask=(1<<G.size)-1;

  // 最上段のクイーンが角にある場合の探索
  if(l->BOUND1>1 && l->BOUND1<G.sizeE) {
    if(l->BOUND1<G.sizeE) {
      // 角にクイーンを配置
      l->aBoard[1]=bit=(1<<l->BOUND1);
      //２行目から探索
      //  backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1),l);
      if(NR==1){
        //非再帰
        backTrack1_NR(2,(2|bit)<<1,(1|bit),(bit>>1),l);
      }else{
        //再帰
        backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1),l);
      }
    }
  }
  l->ENDBIT=(l->TOPBIT>>l->BOUND1);
  l->SIDEMASK=l->LASTMASK=(l->TOPBIT|1);
  /* 最上段行のクイーンが角以外にある場合の探索
     ユニーク解に対する左右対称解を予め削除するには、
     左半分だけにクイーンを配置するようにすればよい */
  if(l->BOUND1>0&&l->BOUND2<G.sizeE&&l->BOUND1<l->BOUND2){
    for(int i=1; i<l->BOUND1; i++){
      l->LASTMASK=l->LASTMASK|l->LASTMASK>>1|l->LASTMASK<<1;
    }
    if(l->BOUND1<l->BOUND2) {
      l->aBoard[0]=bit=(1<<l->BOUND1);
      //backTrack2(1,bit<<1,bit,bit>>1,l);
      if(NR==1){
        //非再帰
        backTrack2_NR(1,bit<<1,bit,bit>>1,l);
      }else{
        //再帰
        backTrack2(1,bit<<1,bit,bit>>1,l);
      }
    }
    l->ENDBIT>>=G.size;
  }

  return 0;   //*run()の場合はreturn 0;が必要
}
//
void *NQueenThread(){
  local l[MAX];                //構造体 local型
  pthread_t pt[G.size];                 //スレッド childThread
  for(int BOUND1=G.sizeE,BOUND2=0;BOUND2<G.sizeE;BOUND1--,BOUND2++){
    l[BOUND1].BOUND1=BOUND1; l[BOUND1].BOUND2=BOUND2;         //B1 と B2を初期化
    for(int j=0;j<G.size;j++){ l[l->BOUND1].aBoard[j]=j; } // aB[]の初期化
    l[BOUND1].COUNT2[BOUND1]=l[BOUND1].COUNT4[BOUND1]=l[BOUND1].COUNT8[BOUND1]=0;//カウンターの初期化
    // チルドスレッドの生成
    int iFbRet=pthread_create(&pt[BOUND1],NULL,&run,&l[BOUND1]);
    if(iFbRet>0){
      printf("[mainThread] pthread_create #%d: %d\n", l[BOUND1].BOUND1, iFbRet);
    }
  }
  for(int BOUND1=G.sizeE,BOUND2=0;BOUND2<G.sizeE;BOUND1--,BOUND2++){
    pthread_join(pt[BOUND1],NULL);
  }
  //スレッド毎のカウンターを合計
  for(int BOUND1=G.sizeE,BOUND2=0;BOUND2<G.sizeE;BOUND1--,BOUND2++){
    G.lTOTAL+=l[BOUND1].COUNT2[BOUND1]*2+l[BOUND1].COUNT4[BOUND1]*4+l[BOUND1].COUNT8[BOUND1]*8;
    G.lUNIQUE+=l[BOUND1].COUNT2[BOUND1]+l[BOUND1].COUNT4[BOUND1]+l[BOUND1].COUNT8[BOUND1];
  }
  return 0;
}
//
void NQueen(){
  pthread_t pth;  //スレッド変数
  int iFbRet;
  // メインスレッドの生成
  // 拡張子 CUDA はpthreadをサポートしていませんので実行できません
  // コンパイルが通らないので 以下をコメントアウトします
  // Cディレクトリの 並列処理はC13_N-Queen.c を参考にして下さい。
  // iFbRet = pthread_create(&pth, NULL,&NQueenThread,NULL);
  if(iFbRet>0){
    printf("[main] pthread_create: %d\n", iFbRet); //エラー出力デバッグ用
  }
  pthread_join(pth,NULL); /* いちいちjoinをする */
}
//メインメソッド
int main(int argc,char** argv) {
  bool cpu=false,cpur=false,gpu=false;
  int argstart=1,steps=24576;
  /** パラメータの処理 */
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
    else
      cpur=true;
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g|-r] n steps\n",argv[0]);
    printf("  -c: CPU only\n");
    printf("  -r: CPUR only\n");
    printf("  -g: GPU only\n");
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  if(cpu){
    printf("\n\n１３．CPU 非再帰 並列処理 pthread\n");
  }else if(cpur){
    printf("\n\n１３．CPUR 再帰 並列処理 pthread\n");
  }else if(gpu){
    printf("\n\n１３．GPU 非再帰 並列処理 pthread\n");
  }
  if(cpu||cpur){
    printf("%s\n"," N:           Total           Unique          dd:hh:mm:ss.ms");
    struct timeval t0;
    struct timeval t1;
    int min=4; int targetN=18;
    for(int i=min;i<=targetN;i++){
      //TOTAL=0; UNIQUE=0;
      G.size=i; G.sizeE=i-1; //初期化
      G.lTOTAL=G.lUNIQUE=0;
      gettimeofday(&t0, NULL);
      NQueen();
      gettimeofday(&t1, NULL);
      int ss;int ms;int dd;
      if(t1.tv_usec<t0.tv_usec) {
        dd=(t1.tv_sec-t0.tv_sec-1)/86400;
        ss=(t1.tv_sec-t0.tv_sec-1)%86400;
        ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
      }else {
        dd=(t1.tv_sec-t0.tv_sec)/86400;
        ss=(t1.tv_sec-t0.tv_sec)%86400;
        ms=(t1.tv_usec-t0.tv_usec+500)/10000;
      }
      int hh=ss/3600;
      int mm=(ss-hh*3600)/60;
      ss%=60;
      printf("%2d:%16ld%17ld%12.2d:%02d:%02d:%02d.%02d\n", i,G.lTOTAL,G.lUNIQUE,dd,hh,mm,ss,ms);
    }
  }
  if(gpu){
    if(!InitCUDA()){return 0;}
    int min=4;int targetN=18;
    struct timeval t0;struct timeval t1;int ss;int ms;int dd;
    printf("%s\n"," N:          Total        Unique                 dd:hh:mm:ss.ms");
    for(int i=min;i<=targetN;i++){
      gettimeofday(&t0,NULL);   // 計測開始
      Total=solve_nqueen_cuda(i,steps);
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
      printf("%2d:%18ld%18ld%12.2d:%02d:%02d:%02d.%02d\n", i,Total,Unique,dd,hh,mm,ss,ms);
    }
  }
  return 0;
}
