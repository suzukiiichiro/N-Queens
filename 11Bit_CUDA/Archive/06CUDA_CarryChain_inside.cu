/**
 *
 * bash版キャリーチェーンのC言語版のGPU/CUDA移植版
 * CUDAの実行をfor文の一番内側、クイーンを上下左右2行2列置いたあと
 
 詳しい説明はこちらをどうぞ
 https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題
 *
・carryChain GPU inside backTrack部分でGPUを起動 stepsに達するまで貯めた

NQueens_suzuki$ nvcc -O3 -arch=sm_61  -Xcompiler -mcmodel=medium  06CUDA_CarryChain_inside.cu && ./a.out -n
 N:            Total          Unique      dd:hh:mm:ss.ms
 4:            2           0      00:00:00:00.13
 5:           10           0      00:00:00:00.00
 6:            4           0      00:00:00:00.00
 7:           40           0      00:00:00:00.00
 8:           92           0      00:00:00:00.00
 9:          352           0      00:00:00:00.00
10:          724           0      00:00:00:00.00
11:         2680           0      00:00:00:00.00
12:        14200           0      00:00:00:00.01
13:        73712           0      00:00:00:00.04
14:       365596           0      00:00:00:00.12
15:      2279184           0      00:00:00:00.43
16:     14772512           0      00:00:00:02.10
17:     95815104           0      00:00:00:13.29
18:    666090624           0      00:00:01:36.21
19:   4968057848           0      00:00:12:16.30
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
#define steps 24576
/**
  * システムによって以下のマクロが必要であればコメントを外してください。
  */
//#define UINT64_C(c) c ## ULL

typedef unsigned int uint;
typedef unsigned long ulong;

ulong TOTAL=0; 
ulong UNIQUE=0;
ulong totalCond=0;
typedef struct
{
  uint size;
  uint pres_a[930]; 
  uint pres_b[930];
}Global; Global g;
typedef struct
{
  ulong row;
  ulong down;
  ulong left;
  ulong right;
  long long x[MAX];
}Board ;
typedef struct
{
  Board B;
  Board nB;
  Board eB;
  Board sB;
  Board wB;
  uint n;
  uint e;
  uint s;
  uint w;
  ulong dimx;
  ulong dimy;
  ulong COUNTER[3];      
  uint COUNT2;
  uint COUNT4;
  uint COUNT8;
  uint type;
}Local;


ulong* totalDown=new ulong[steps];
ulong* totalLeft=new ulong[steps];
ulong* totalRight=new ulong[steps];
ulong* totalRow=new ulong[steps];
uint* totalType=new uint[steps];
ulong* results =new ulong[steps];
bool matched=false;
ulong* rowCuda;
ulong* downCuda;
ulong* leftCuda;
ulong* rightCuda;
ulong* resultsCuda;
uint* typeCuda;

/**
  *
  */
__global__ void solve(uint size,int current,uint* totalType,ulong* totalRow,ulong* totalDown,ulong* totalLeft,ulong* totalRight,ulong* results,uint totalCond)
{
  const uint tid=threadIdx.x;
  const uint bid=blockIdx.x;
  const uint idx=bid*blockDim.x+tid;
  ulong  row_a[MAX];
  ulong  down_a[MAX];
  ulong  left_a[MAX];
  ulong  right_a[MAX];
  ulong  bitmap_a[MAX];
  __shared__ int  sum[THREAD_NUM];
  ulong row=row_a[current]=totalRow[idx];
  ulong left=left_a[current]=totalLeft[idx];
  ulong down=down_a[current]=totalDown[idx];
  ulong right=right_a[current]=totalRight[idx];
  ulong bitmap=bitmap_a[current]=~(left_a[current]|down_a[current]|right_a[current]);
  uint total=0;
  ulong bit;
  uint ttype=totalType[idx];
  if(idx<totalCond){
  while(current>-1){
    if((bitmap!=0||row&1)&&current<size){
      if(!(down+1)){

        total+=ttype;
        current--;
        row=row_a[current];
        left=left_a[current];
        right=right_a[current];
        down=down_a[current];
        bitmap=bitmap_a[current];
        continue;
      }else if(row&1){
        while( row&1 ){
          row>>=1;
          left<<=1;
          right>>=1;
        }
        bitmap=~(left|down|right);  //再帰に必要な変数は必ず定義する必要があります。
        continue;
      }else{
        bit=-bitmap&bitmap;
        bitmap=bitmap^bit;
        if(current<size){
          row_a[current]=row;
          left_a[current]=left;
          down_a[current]=down;
          right_a[current]=right;
          bitmap_a[current]=bitmap;
          current++;
        }
        row>>=1;      //１行下に移動する
        left=(left|bit)<<1;
        down=down|bit;
        right=(right|bit)>>1;
        bitmap=~(left|down|right);  //再帰に必要な変数は必ず定義する必要があります。
      }
    }else{
      current--;
      row=row_a[current];
      left=left_a[current];
      right=right_a[current];
      down=down_a[current];
      bitmap=bitmap_a[current];
    }
  }
  sum[tid]=total;
  }else{
    sum[tid]=0;
  }
  __syncthreads();if(tid<64&&tid+64<THREAD_NUM){sum[tid]+=sum[tid+64];} 
  __syncthreads();if(tid<32){sum[tid]+=sum[tid+32];} 
  __syncthreads();if(tid<16){sum[tid]+=sum[tid+16];} 
  __syncthreads();if(tid<8){sum[tid]+=sum[tid+8];} 
  __syncthreads();if(tid<4){sum[tid]+=sum[tid+4];} 
  __syncthreads();if(tid<2){sum[tid]+=sum[tid+2];} 
  __syncthreads();if(tid<1){sum[tid]+=sum[tid+1];} 
  __syncthreads();if(tid==0){results[bid]=sum[0];}
}
/**
  *
  */
void append(void* args)
{
  Local *l=(Local *)args;
  totalRow[totalCond]=l->B.row>>2;
  totalDown[totalCond]=((((l->B.down>>2)|(~0<<(g.size-4)))+1)<<(g.size-5))-1;
  totalLeft[totalCond]=l->B.left>>4;
  totalRight[totalCond]=(l->B.right>>4)<<(g.size-5);
  totalType[totalCond]=l->type;
  totalCond++;
  if(totalCond==steps){
    if(matched){
      cudaMemcpy(results,resultsCuda,sizeof(long)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
      for(uint col=0;col<steps/THREAD_NUM;col++){TOTAL+=results[col];}
        matched=false;
      }
      cudaMemcpy(rowCuda,totalRow,sizeof(ulong)*totalCond,cudaMemcpyHostToDevice);
      cudaMemcpy(downCuda,totalDown,sizeof(ulong)*totalCond,cudaMemcpyHostToDevice);
      cudaMemcpy(leftCuda,totalLeft,sizeof(ulong)*totalCond,cudaMemcpyHostToDevice);
      cudaMemcpy(rightCuda,totalRight,sizeof(ulong)*totalCond,cudaMemcpyHostToDevice);
      cudaMemcpy(typeCuda,totalType,sizeof(uint)*totalCond,cudaMemcpyHostToDevice);
      solve<<<steps/THREAD_NUM,THREAD_NUM>>>(g.size,0,typeCuda,rowCuda,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
      cudaMemcpy(results,resultsCuda,
      sizeof(uint)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
      matched=true;
      totalCond=0;
  }
}
/**
  * 非再帰 対称解除法
  */
void carryChain_symmetry(void* args)
{
  Local *l=(Local *)args;
  // 対称解除法
  const uint ww=(g.size-2)*(g.size-1)-1-l->w;
  const uint w2=(g.size-2)*(g.size-1)-1;
  // # 対角線上の反転が小さいかどうか確認する
  if((l->s==ww)&&(l->n<(w2-l->e))){ return ; }
  // # 垂直方向の中心に対する反転が小さいかを確認
  if((l->e==ww)&&(l->n>(w2-l->n))){ return; }
  // # 斜め下方向への反転が小さいかをチェックする
  if((l->n==ww)&&(l->e>(w2-l->s))){ return; }
  // 枝刈り １行目が角の場合回転対称チェックせずCOUNT8にする
  if(l->B.x[0]==0){
    l->type=8;
    append(l);
    //l->COUNTER[l->COUNT8]+=solve(g.size,0,l->B.row>>2,
    //l->B.left>>4,((((l->B.down>>2)|(~0<<(g.size-4)))+1)<<(g.size-5))-1,(l->B.right>>4)<<(g.size-5));
    return ;
  }
  // n,e,s==w の場合は最小値を確認する。右回転で同じ場合は、
  // w=n=e=sでなければ値が小さいのでskip  w=n=e=sであれば90度回転で同じ可能性
  if(l->s==l->w){ if((l->n!=l->w)||(l->e!=l->w)){ return; }
    l->type=2;
    append(l);
    //l->COUNTER[l->COUNT2]+=solve(g.size,0,l->B.row>>2,
    //l->B.left>>4,((((l->B.down>>2)|(~0<<(g.size-4)))+1)<<(g.size-5))-1,(l->B.right>>4)<<(g.size-5));
    return;
  }
  // e==wは180度回転して同じ 180度回転して同じ時n>=sの時はsmaller?
  if((l->e==l->w)&&(l->n>=l->s)){ if(l->n>l->s){ return; }
    l->type=4;
    append(l);
    //l->COUNTER[l->COUNT4]+=solve(g.size,0,l->B.row>>2,
    //l->B.left>>4,((((l->B.down>>2)|(~0<<(g.size-4)))+1)<<(g.size-5))-1,(l->B.right>>4)<<(g.size-5));
    return;
  }
  l->type=8;
  append(l);
  //l->COUNTER[l->COUNT8]+=solve(g.size,0,l->B.row>>2,
  //l->B.left>>4,((((l->B.down>>2)|(~0<<(g.size-4)))+1)<<(g.size-5))-1,(l->B.right>>4)<<(g.size-5));
  return;
}
/**
  * クイーンの効きをチェック
  */
bool placement(void* args)
{
  Local *l=(Local *)args;
  if(l->B.x[l->dimx]==l->dimy){ return true;  }  
  if (l->B.x[0]==0){
    if (l->B.x[1]!=(ulong)-1){
      if((l->B.x[1]>=l->dimx)&&(l->dimy==1)){ return false; }
    }
  }else{
    if( (l->B.x[0]!=(ulong)-1) ){
      if(( (l->dimx<l->B.x[0]||l->dimx>=g.size-l->B.x[0])
        && (l->dimy==0 || l->dimy==g.size-1)
      )){ return 0; } 
      if ((  (l->dimx==g.size-1)&&((l->dimy<=l->B.x[0])||
          l->dimy>=g.size-l->B.x[0]))){
        return 0;
      } 
    }
  }
  l->B.x[l->dimx]=l->dimy;                    //xは行 yは列
  ulong row=UINT64_C(1)<<l->dimx;
  ulong down=UINT64_C(1)<<l->dimy;
  ulong left=UINT64_C(1)<<(g.size-1-l->dimx+l->dimy); //右上から左下
  ulong right=UINT64_C(1)<<(l->dimx+l->dimy);       // 左上から右下
  if((l->B.row&row)||(l->B.down&down)||(l->B.left&left)||(l->B.right&right)){ return false; }     
  l->B.row|=row; l->B.down|=down; l->B.left|=left; l->B.right|=right;
  return true;
}
/**
  * 
  */
void thread_run(void* args)
{
  Local *l=(Local *)args;
  // memcpy(&l->B,&l->wB,sizeof(Board));       // B=wB;
  l->B=l->wB;
  l->dimx=0; l->dimy=g.pres_a[l->w];
  //if(!placement(l)){ continue; }
  if(!placement(l)){ return; }
  l->dimx=1; l->dimy=g.pres_b[l->w];
  // if(!placement(l)){ continue; }
  if(!placement(l)){ return; }
  //２ 左２行に置く
  // memcpy(&l->nB,&l->B,sizeof(Board));       // nB=B;
  l->nB=l->B;
  for(l->n=l->w;l->n<(g.size-2)*(g.size-1)-l->w;++l->n){
    // memcpy(&l->B,&l->nB,sizeof(Board));     // B=nB;
    l->B=l->nB;
    l->dimx=g.pres_a[l->n]; l->dimy=g.size-1;
    if(!placement(l)){ continue; }
    l->dimx=g.pres_b[l->n]; l->dimy=g.size-2;
    if(!placement(l)){ continue; }
    // ３ 下２行に置く
    // memcpy(&l->eB,&l->B,sizeof(Board));     // eB=B;
    l->eB=l->B;
    for(l->e=l->w;l->e<(g.size-2)*(g.size-1)-l->w;++l->e){
      // memcpy(&l->B,&l->eB,sizeof(Board));   // B=eB;
      l->B=l->eB;
      l->dimx=g.size-1; l->dimy=g.size-1-g.pres_a[l->e];
      if(!placement(l)){ continue; }
      l->dimx=g.size-2; l->dimy=g.size-1-g.pres_b[l->e];
      if(!placement(l)){ continue; }
      // ４ 右２列に置く
      // memcpy(&l->sB,&l->B,sizeof(Board));   // sB=B;
      l->sB=l->B;
      for(l->s=l->w;l->s<(g.size-2)*(g.size-1)-l->w;++l->s){
        // memcpy(&l->B,&l->sB,sizeof(Board)); // B=sB;
        l->B=l->sB;
        l->dimx=g.size-1-g.pres_a[l->s]; l->dimy=0;
        if(!placement(l)){ continue; }
        l->dimx=g.size-1-g.pres_b[l->s]; l->dimy=1;
        if(!placement(l)){ continue; }
        // 対称解除法
        carryChain_symmetry(l);
      } //w
    } //e
  } //n
}
/**
  * 非再帰  チェーンのビルド
  */
void buildChain()
{
  Local l[(g.size/2)*(g.size-3)];
  cudaMalloc((void**) &rowCuda,sizeof(ulong)*steps);
  cudaMalloc((void**) &downCuda,sizeof(ulong)*steps);
  cudaMalloc((void**) &leftCuda,sizeof(ulong)*steps);
  cudaMalloc((void**) &rightCuda,sizeof(ulong)*steps);
  cudaMalloc((void**) &typeCuda,sizeof(uint)*steps);
  cudaMalloc((void**) &resultsCuda,sizeof(ulong)*steps/THREAD_NUM);
  // カウンターの初期化
  l->COUNT2=0; l->COUNT4=1; l->COUNT8=2;
  l->COUNTER[l->COUNT2]=l->COUNTER[l->COUNT4]=l->COUNTER[l->COUNT8]=0;
  // Board の初期化 nB,eB,sB,wB;
  l->B.row=l->B.down=l->B.left=l->B.right=0;
  // Board x[]の初期化
  for(uint i=0;i<g.size;++i){ l->B.x[i]=-1; }
  //１ 上２行に置く
  // memcpy(&l->wB,&l->B,sizeof(Board));         // wB=B;
  l->wB=l->B;
  for(l->w=0;l->w<=(unsigned)(g.size/2)*(g.size-3);++l->w){
    thread_run(&l);
  }
  if(matched){
    cudaMemcpy(results,resultsCuda,sizeof(long)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    for(uint col=0;col<steps/THREAD_NUM;col++){TOTAL+=results[col];}
    matched=false;
  }
  cudaMemcpy(rowCuda,totalRow,sizeof(ulong)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(downCuda,totalDown,sizeof(ulong)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(leftCuda,totalLeft,sizeof(ulong)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(rightCuda,totalRight,sizeof(ulong)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(typeCuda,totalType,sizeof(uint)*totalCond,cudaMemcpyHostToDevice);
  solve<<<steps/THREAD_NUM,THREAD_NUM>>>(g.size,0,typeCuda,rowCuda,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
  cudaMemcpy(results,resultsCuda,sizeof(long)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  for(uint col=0;col<steps/THREAD_NUM;col++){TOTAL+=results[col];}	
}
/**
  * チェーンのリストを作成
  */
void listChain()
{
  uint idx=0;
  for(uint a=0;a<(unsigned)g.size;++a){
    for(uint b=0;b<(unsigned)g.size;++b){
      if(((a>=b)&&(a-b)<=1)||((b>a)&&(b-a)<=1)){ continue; }
      g.pres_a[idx]=a;
      g.pres_b[idx]=b;
      ++idx;
    }
  }
}
/**
  * キャリーチェーン
  */
void carryChain()
{
  listChain();  //チェーンのリストを作成
  buildChain(); // チェーンのビルド
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
  for(int size=min;size<=targetN;size++){
    gettimeofday(&t0,NULL);   // 計測開始
    totalCond=0;
    TOTAL=UNIQUE=0;
    g.size=size;
    carryChain();
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
  }
  cudaFree(rowCuda);
  cudaFree(downCuda);
  cudaFree(leftCuda);
  cudaFree(rightCuda);
  cudaFree(typeCuda);
  cudaFree(resultsCuda);
  delete[] totalRow;
  delete[] totalDown;
  delete[] totalLeft;
  delete[] totalRight;
  delete[] totalType;
  delete[] results;
  return 0;
}
