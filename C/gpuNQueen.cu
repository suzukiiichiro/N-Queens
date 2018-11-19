/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 必要なこと
　１．$ lspci | grep -i  nvidia 
      で、nVidiaが存在しなければ絶対動かない。
　２．$ nvidia-smi
　　　で、以下の通りに出力され、ＧＰＵの存在が確認できなければ絶対に動かない。

bash-4.2$ nvidia-smi
Wed Jun 27 02:36:34 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.111                Driver Version: 384.111                   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           On   | 00000000:00:1E.0 Off |                    0 |
| N/A   38C    P8    30W / 149W |      1MiB / 11439MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+


　３．nVidia CUDAのインストールは思った以上に難しい。
　　　特にMacはOSのバージョン、xcode、command line toolsとの相性もある。
     
  手軽なのは、有料だが Amazon AWSのEC2でCUDA nVidia GPU 対応のサーバーを使うこと。
　（こちらが絶対的におすすめ）


  ４．ここまでが了解であれば、以下のコマンドで実行可能だ。


 コンパイルと実行

 # CPUだけの実行
 $ nvcc gpuNQueen.cu -o gpuNQueen && ./gpuNQueen -cpu 

 # GPUだけの実行
 $ nvcc gpuNQueen.cu -o gpuNQueen && ./gpuNQueen -gpu

 # CPUとGPUの実行
 $ nvcc gpuNQueen.cu -o gpuNQueen && ./gpuNQueen



 １３．ＧＰＵ nVidia-CUDA               N17=    1.67
 *
 *  実行結果

 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 0          00:00:00:00.00
 5:                10                 0          00:00:00:00.00
 6:                 4                 0          00:00:00:00.00
 7:                40                 0          00:00:00:00.00
 8:                92                 0          00:00:00:00.00
 9:               352                 0          00:00:00:00.00
10:               724                 0          00:00:00:00.00
11:              2680                 0          00:00:00:00.00
12:             14200                 0          00:00:00:00.00
13:             73712                 0          00:00:00:00.05
14:            365596                 0          00:00:00:00.29
15:           2279184                 0          00:00:00:01.94

 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 0          00:00:00:00.06
 5:                10                 0          00:00:00:00.00
 6:                 4                 0          00:00:00:00.00
 7:                40                 0          00:00:00:00.00
 8:                92                 0          00:00:00:00.00
 9:               352                 0          00:00:00:00.00
10:               724                 0          00:00:00:00.00
11:              2680                 0          00:00:00:00.00
12:             14200                 0          00:00:00:00.01
13:             73712                 0          00:00:00:00.02
14:            365596                 0          00:00:00:00.01
15:           2279184                 0          00:00:00:00.05
16:          14772512                 0          00:00:00:00.27
17:          95815104                 0          00:00:00:01.65
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define THREAD_NUM		96

/**
  case 1 : 非再帰　非CUDA
  1. バックトラック
**/
#define MAX 15
long Total=0 ;        //合計解
int fA[2*MAX-1]; //fA:flagA 縦 配置フラグ　
int fB[2*MAX-1];  //fB:flagB 斜め配置フラグ　
int fC[2*MAX-1];  //fC:flagC 斜め配置フラグ　
int aB[2*MAX-1];      //aB:aBoard[] チェス盤の横一列
typedef struct{
    //  データを格納数る配列
    int array[MAX];
    //  現在の位置
    int current;
}STACK;
 
//  スタックの初期化
void init(STACK*);
//  値のプッシュ
int push(STACK*,int);
//  値のポップ
int pop(STACK*,int*);
//  スタックの初期化
void init(STACK* pStack)
{
    int i;
    for(i = 0; i < MAX; i++){
        pStack->array[i] = 0;
    }
    //  カレントの値を0に。
    pStack->current = 0;
}
//  値のプッシュ
int push(STACK* pStack,int value)
{
    if(pStack->current < MAX){
        //  まだデータが格納できるのなら、データを格納し、一つずらす。
        pStack->array[pStack->current] = value;
        pStack->current++;
        return 1;
    }
    //  データを格納しきれなかった
    return 0;
}
//  値のポップ
int pop(STACK* pStack,int* pValue)
{
    if(pStack->current > 0){
        //  まだデータが格納できるのなら、データを格納し、一つずらす。
        pStack->current--;
        *pValue = pStack->array[pStack->current];
        return *pValue;
    }
    return 0;
}
int leng(STACK* pStack)
{
    if(pStack->current > 0){
     return 1;
    }
    return 0;
}
void solve_nqueen_nonRecursive_BT(int r,int n){
  STACK R;
  STACK I;
  init(&R);
  init(&I);
  int bend=0;
  int rflg=0;
  while(1){
  //start:
  //printf("methodstart\n");
  //printf("###r:%d\n",r);
  //for(int k=0;k<n;k++){
  //  printf("###i:%d\n",k);
  //  printf("###fa[k]:%d\n",fA[k]);
  //  printf("###fB[k]:%d\n",fB[k]);
  //  printf("###fC[k]:%d\n",fC[k]);
  //}
  if(r==n && rflg==0){
  //printf("if(r==n){\n");
    Total++; //解を発見
   // printf("Total++;\n");
  }else{
    //printf("}else{\n");
    for(int i=0;i<n;i++){
      //printf("for(int i=0;i<n;i++){\n");
      if(rflg==0){
        aB[r]=i ;
      }
      //printf("aB[r]=i ;\n");
      //printf("###i:%d\n",i);
      //printf("###r:%d\n",r);
     // for(int k=0;k<n;k++){
      //  printf("###i:%d\n",k);
      //  printf("###fa[k]:%d\n",fA[k]);
      //  printf("###fB[k]:%d\n",fB[k]);
      //  printf("###fC[k]:%d\n",fC[k]);
     // }
      //バックトラック 制約を満たしているときだけ進む
      if((fA[i]==0&&fB[r-i+(n-1)]==0&&fC[r+i]==0)  || rflg==1){
      //  printf("if(fA[i]==0&&fB[r-i+(n-1)]==0&&fC[r+i]==0){\n");
        if(rflg==0){
          fA[i]=fB[r-aB[r]+n-1]=fC[r+aB[r]]=1; 
       //   printf("fA[i]=fB[r-aB[r]+n-1]=fC[r+aB[r]]=1;\n");
       //   printf("###before_nqueen\n");
       //   printf("###i:%d\n",i);
       //   printf("###r:%d\n",r);
       //   for(int k=0;k<n;k++){
       //     printf("###i:%d\n",k);
       //     printf("###fa[k]:%d\n",fA[k]);
       //     printf("###fB[k]:%d\n",fB[k]);
       //     printf("###fC[k]:%d\n",fC[k]);
       //   }
          push(&R,r); 
          push(&I,i); 
          r=r+1;
          bend=1;
          break;
          //  goto start;
        }
        //NQueen(r+1,n);//再帰
        // ret:
        if(rflg==1){
          r=pop(&R,&r);
          i=pop(&I,&i);
        //  printf("###after_nqueen\n");
        //  printf("###i:%d\n",i);
        //  printf("###r:%d\n",r);
        //  for(int k=0;k<n;k++){
        //    printf("###i:%d\n",k);
        //    printf("###fa[k]:%d\n",fA[k]);
        //    printf("###fB[k]:%d\n",fB[k]);
        //    printf("###fC[k]:%d\n",fC[k]);
        //  }
          fA[i]=fB[r-aB[r]+n-1]=fC[r+aB[r]]=0; 
          rflg=0;
        }
        //printf("fA[i]=fB[r-aB[r]+n-1]=fC[r+aB[r]]=0;\n");
      }else{
        bend=0;
      }
      //printf("}#after:if(fA[i]==0&&fB[r-i+(n-1)]==0&&fC[r+i]==0){\n");
    }  
    //printf("after:for\n");
    if(bend==1 && rflg==0){
      bend=0;
      continue;
    }
  }
  //printf("after:else\n");
    if(r==0){
      break;
    }else{
      //goto ret;
      rflg=1;
    }
  }
}

/**
  case 2 : 再帰　非CUDA
  1. バックトラック
**/
void solve_nqueen_Recursive_BT(int r,int n){
  if(r==n){
    Total++; //解を発見
  }else{
    for(int i=0;i<n;i++){
      aB[r]=i ;
      //バックトラック 制約を満たしているときだけ進む
      if(fA[i]==0&&fB[r-i+(n-1)]==0&&fC[r+i]==0){
        fA[i]=fB[r-aB[r]+n-1]=fC[r+aB[r]]=1; 
        solve_nqueen_Recursive_BT(r+1,n);//再帰
        fA[i]=fB[r-aB[r]+n-1]=fC[r+aB[r]]=0; 
      }
    }  
  }
}

/** 
  case 3 : 非再帰 非CUDA
  1. バックトラック
  2. ビットマップ

14:            365596                 0          00:00:00:00.22
15:           2279184                 0          00:00:00:01.50
*/
long long solve_nqueen_nonRecursive_BT_BM(int n){
  unsigned int down[32];unsigned int left[32];unsigned int right[32];unsigned int bm[32];
  if(n<=0||n>32){return 0;}
  const unsigned int msk=(1<<n)-1;long long total=0;long long uTotal=0;
  int i=0;int j=0;unsigned int bit;
  down[0]=0;left[0]=0;right[0]=0;bm[0]=0;
  for(j=0;j<(n+1)/2;j++){
    bit=(1<<j);
    bm[0]|=bit;down[1]=bit;left[1]=bit<<1;right[1]=bit>>1;
    bm[1]=(down[1]|left[1]|right[1]);
    i=1;
    if(n%2==1&&j==(n+1)/2-1){uTotal=total;total=0;}
    while(i>0){
      if((bm[i]&msk)==msk){i--;}
      else{
        bit=((bm[i]+1)^bm[i])&~bm[i];
        bm[i]|=bit;
        if((bit&msk)!=0){
          if(i+1==n){total++;i--;}
          else{
            down[i+1]=down[i]|bit;left[i+1]=(left[i]|bit)<<1;right[i+1]=(right[i]|bit)>>1;
            bm[i+1]=(down[i+1]|left[i+1]|right[i+1]);
            i++;
          }
        }else{i--;}
      }
    }
  }
  if(n%2==0){return total*2;}
  else{return uTotal*2+total;}
}

/** 
  case 4 : 再帰 非CUDA
  1. バックトラック
  2. ビットマップ

14:            365596                 0          00:00:00:00.44
15:           2279184                 0          00:00:00:02.81
*/
long long nqInternal_BT_BM(int n,unsigned int left,unsigned int down,unsigned int right) {
  unsigned int msk=(1<<n)-1;
  if(down==msk){return 1;}
	unsigned int bm=(left|down|right);
	if((bm&msk)==msk){return 0;}
	long long total=0;
	unsigned int bit=(bm+1)&~bm;
	while((bit&msk)!=0){
		total+=nqInternal_BT_BM(n,(left|bit)<<1,down|bit,(right|bit)>>1);
		bm|=bit;
		bit=(bm+1)&~bm;
	}
	return total;
}
long long solve_nqueen_Recursive_BT_BM(int n){
	return nqInternal_BT_BM(n,0,0,0);
}

/**
  case 5 : 非再帰 非CUDA
  1. バックトラック BT
  2. ビットマップ   BM
  3. 対象解除法     SO
*/
long long solve_nqueen_nonRecursive_BT_BM_SO(int n){
  return true;
}

/**
  case 6 : 再帰 非CUDA
  1. バックトラック BT
  2. ビットマップ   BM
  3. 対象解除法     SO
*/
long long nqInternal_BT_BM_SO(int n,unsigned int left,unsigned int down,unsigned int right) {
  return true;
}
long long solve_nqueen_Recursive_BT_BM_SO(int n){
	return nqInternal_BT_BM(n,0,0,0);
}



/** #################################################################

  nVidia CUDA ブロック

#####################################################################*/
/** 
  CUDA 非再帰 CPUイテレータから複数の初期条件を受け取り、カウント
  1. バックトラック backTrack
  2. ビットマップ   bitmap
14:            365596                 0          00:00:00:00.08
15:           2279184                 0          00:00:00:00.49
*/
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
void execCPU(int procNo){
  long long solution;
  int min=4;int targetN=15;
  struct timeval t0;struct timeval t1;int ss;int ms;int dd;
  printf("\n%s\n"," N:          Total        Unique                 dd:hh:mm:ss.ms");
  for(int i=min;i<=targetN;i++){
    gettimeofday(&t0,NULL);   // 計測開始
    switch (procNo){
      case 1:
        //solution=solve_nqueen_nonRecursive_BT(i);
        Total=0 ;        //合計解
        solve_nqueen_nonRecursive_BT(0,i);
        solution=Total;
        break;
      case 2:
        //solution=solve_nqueen_Recursive_BT(0,i);
        for(int j=0;j<i;j++){ aB[j]=j; } //aBを初期化
        Total=0 ;        //合計解
        solve_nqueen_Recursive_BT(0,i);
        solution=Total;
        break;
      case 3:
        solution=solve_nqueen_nonRecursive_BT_BM(i);
        break;
      case 4:
        solution=solve_nqueen_Recursive_BT_BM(i);
        break;
      case 5:
        solution=solve_nqueen_nonRecursive_BT_BM_SO(i);
        break;
      case 6:
        solution=solve_nqueen_Recursive_BT_BM_SO(i);
        break;
      default:
        break;
    } 
    /** 再帰 */
    /** 非再帰 */
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
    long lGUnique=0;
    printf("%2d:%18llu%18llu%12.2d:%02d:%02d:%02d.%02d\n", i,(unsigned long long)solution,(unsigned long long)lGUnique,dd,hh,mm,ss,ms);
  }
}
int main(int argc,char** argv) {
  bool cpu=true,gpu=true;
  int argstart=1,steps=24576;
  /** パラメータの処理 */
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){gpu=false;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){cpu=false;}
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g] n steps\n",argv[0]);
    printf("  -c: CPU only\n");
    printf("  -g: GPU only\n");
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  /** CPU */
  if(cpu){
    printf("\n\n1. 非再帰＋バックトラック(BT)");
    execCPU(1); /* solve_nqueen_nonRecursive_BT     */
    printf("\n\n2. 再帰＋バックトラック(BT)");
    execCPU(2); /* solve_nqueen_Recursive_BT     */
    printf("\n\n3. 非再帰＋バックトラック(BT)＋ビットマップ(BM)");
    execCPU(3); /* solve_nqueen_nonRecursive_BT_BM  */
    printf("\n\n4. 再帰＋バックトラック(BT)＋ビットマップ(BM)");
    execCPU(4); /* 07_05 solve_nqueen_Recursive_BT_BM  */
    printf("\n\n5. 非再帰＋バックトラック(BT)＋ビットマップ(BM)＋対象解除法(SO)");
    execCPU(5); /* solve_nqueen_nonRecursive_BT_BM_SO     */
    printf("\n\n6. 再帰＋バックトラック(BT)＋ビットマップ(BM)＋対象解除法(SO)");
    execCPU(6); /* solve_nqueen_Recursive_BT_BM_SO     */
  }
  /** GPU */
  if(gpu){
    long long solution;
    if(!InitCUDA()){return 0;}
    int min=4;int targetN=17;
    struct timeval t0;struct timeval t1;int ss;int ms;int dd;
    printf("%s\n"," N:          Total        Unique                 dd:hh:mm:ss.ms");
    for(int i=min;i<=targetN;i++){
      gettimeofday(&t0,NULL);   // 計測開始
      solution=solve_nqueen_cuda(i,steps);
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
      long lGUnique=0;
      printf("%2d:%18llu%18llu%12.2d:%02d:%02d:%02d.%02d\n", i,(unsigned long long)solution,(unsigned long long)lGUnique,dd,hh,mm,ss,ms);
    }
  }
  return 0;
}
