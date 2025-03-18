/**
 *
 * bash版キャリーチェーンのC言語版のGPU/CUDA移植版
 * キャリーチェーンを全件完了した後に、CUDAを実行する
 キャリーチェーンの結果はメモリーに保存する
 詳しい説明はこちらをどうぞ
 https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題
 *

・carryChain  CPU デフォルト
 7:           40           0      00:00:00:00.00
 8:           92           0      00:00:00:00.00
 9:          352           0      00:00:00:00.00
10:          724           0      00:00:00:00.00
11:         2680           0      00:00:00:00.02
12:        14200           0      00:00:00:00.07
13:        73712           0      00:00:00:00.24
14:       365596           0      00:00:00:00.78

・carryChain GPU outside
一番外側の forでGPUを起動
 7:           40           0      00:00:00:00.41
 8:           92           0      00:00:00:00.96
 9:          352           0      00:00:00:04.15
10:          724           0      00:00:00:14.54
11:         2680           0      00:00:00:41.31
12:        14200           0      00:00:01:46.98
13:        73712           0      00:00:04:35.11
14:       365596           0      00:00:13:57.61

・carryChain GPU inside
backTrack部分でGPUを起動
stepsに達するまで貯めた

GPU
        Total      Unique      dd:hh:mm:ss.ms
 7:           40           0      00:00:00:00.40
 8:           92           0      00:00:00:00.00
 9:          352           0      00:00:00:00.00
10:          724           0      00:00:00:00.01
11:         2680           0      00:00:00:00.03
12:        14200           0      00:00:00:00.11
13:        73712           0      00:00:00:01.59
14:       365596           0      00:00:00:12.11

・carryChain GPU 先にcarryChainだけ全部やる

        Total      Unique      dd:hh:mm:ss.ms
 7:           40           0      00:00:00:00.34
 8:           92           0      00:00:00:00.00
 9:          352           0      00:00:00:00.00
10:          724           0      00:00:00:00.00
11:         2680           0      00:00:00:00.03
12:        14200           0      00:00:00:00.09
13:        73712           0      00:00:00:00.28
14:       365596           0      00:00:00:00.77

 
アーキテクチャの指定（なくても問題なし、あれば高速）
-arch=sm_13 or -arch=sm_61

CPUの再帰での実行
$ nvcc -O3 -arch=sm_61  -Xcompiler -mcmodel=medium  07CUDA_CarryChain_que_memory.cu && ./a.out -r

CPUの非再帰での実行
$ nvcc -O3 -arch=sm_61  -Xcompiler -mcmodel=medium  07CUDA_CarryChain_que_memory.cu && ./a.out -c

GPUのシングルスレッド
$ nvcc -O3 -arch=sm_61  -Xcompiler -mcmodel=medium  07CUDA_CarryChain_que_memory.cu && ./a.out -g

GPUのマルチスレッド
$ nvcc -O3 -arch=sm_61  -Xcompiler -mcmodel=medium  07CUDA_CarryChain_que_memory.cu && ./a.out -n
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
// システムによって以下のマクロが必要であればコメントを外してください。
//#define UINT64_C(c) c ## ULL
//
// グローバル変数
unsigned long TOTAL=0; 
unsigned long UNIQUE=0;
unsigned long totalCond=0;
long totalcnt=0;
// キャリーチェーン 非再帰版
// 構造体
typedef struct
{
  unsigned int size;
  unsigned int pres_a[400]; 
  unsigned int pres_b[400];
  // uint64_t COUNTER[3];      
  // //カウンター配列
  // unsigned int COUNT2;
  // unsigned int COUNT4;
  // unsigned int COUNT8;
}Global; Global g;
// 構造体
typedef struct
{
  uint64_t row;
  uint64_t down;
  uint64_t left;
  uint64_t right;
  long long x[MAX];
}Board ;
typedef struct
{
  Board B;
  Board nB;
  Board eB;
  Board sB;
  Board wB;
  unsigned n;
  unsigned e;
  unsigned s;
  unsigned w;
  uint64_t dimx;
  uint64_t dimy;
  uint64_t COUNTER[3];      
  //カウンター配列
  unsigned int COUNT2;
  unsigned int COUNT4;
  unsigned int COUNT8;
  unsigned int type;

}Local;


uint64_t* totalDown=new uint64_t[steps];
uint64_t* totalLeft=new uint64_t[steps];
uint64_t* totalRight=new uint64_t[steps];
uint64_t* totalRow=new uint64_t[steps];
int* totalType=new int[steps];
long* results =new long[steps];
bool matched=false;
uint64_t* rowCuda;
uint64_t* downCuda;
uint64_t* leftCuda;
uint64_t* rightCuda;
long* resultsCuda;
int* typeCuda;
Local totalL[steps*420];
/**
  CPU/CPUR 再帰・非再帰共通
  */
// チェーンのリストを作成
void listChain()
{
  unsigned int idx=0;
  for(unsigned int a=0;a<(unsigned)g.size;++a){
    for(unsigned int b=0;b<(unsigned)g.size;++b){
      if(((a>=b)&&(a-b)<=1)||((b>a)&&(b-a)<=1)){ continue; }
      g.pres_a[idx]=a;
      g.pres_b[idx]=b;
      ++idx;
    }
  }
  //printf("%d¥n",idx);
}
/**
  CPU 非再帰
*/
// クイーンの効きをチェック
bool placement(void* args)
{
  Local *l=(Local *)args;
  if(l->B.x[l->dimx]==l->dimy){ return true;  }  
  if (l->B.x[0]==0){
    if (l->B.x[1]!=(uint64_t)-1){
      if((l->B.x[1]>=l->dimx)&&(l->dimy==1)){ return false; }
    }
  }else{
    if( (l->B.x[0]!=(uint64_t)-1) ){
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
  uint64_t row=UINT64_C(1)<<l->dimx;
  uint64_t down=UINT64_C(1)<<l->dimy;
  uint64_t left=UINT64_C(1)<<(g.size-1-l->dimx+l->dimy); //右上から左下
  uint64_t right=UINT64_C(1)<<(l->dimx+l->dimy);       // 左上から右下
  if((l->B.row&row)||(l->B.down&down)||(l->B.left&left)||(l->B.right&right)){ return false; }     
  l->B.row|=row; l->B.down|=down; l->B.left|=left; l->B.right|=right;
  return true;
}
//非再帰
__global__ void solve(int size,int current,int* totalType,uint64_t* totalRow,uint64_t* totalDown,uint64_t* totalLeft,uint64_t* totalRight,
  long* results,int totalCond){
  const int tid=threadIdx.x;
  const int bid=blockIdx.x;
  const int idx=bid*blockDim.x+tid;
  uint64_t  row_a[MAX];
  uint64_t  down_a[MAX];
  uint64_t  left_a[MAX];
  uint64_t  right_a[MAX];
  uint64_t  bitmap_a[MAX];
  __shared__ int  sum[THREAD_NUM];
  
  uint64_t row=row_a[current]=totalRow[idx];
  uint64_t left=left_a[current]=totalLeft[idx];
  uint64_t down=down_a[current]=totalDown[idx];
  uint64_t right=right_a[current]=totalRight[idx];
  uint64_t bitmap=bitmap_a[current]=~(left_a[current]|down_a[current]|right_a[current]);
  int total=0;
  uint64_t bit;
  int ttype=totalType[idx];
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
void append(void* args){
  Local *l=(Local *)args;
  
  totalRow[totalCond]=l->B.row>>2;
  totalDown[totalCond]=((((l->B.down>>2)|(~0<<(g.size-4)))+1)<<(g.size-5))-1;
  totalLeft[totalCond]=l->B.left>>4;
  totalRight[totalCond]=(l->B.right>>4)<<(g.size-5);
  totalType[totalCond]=l->type;
  totalCond++;
 
  if(totalCond==steps){
    if(matched){
      cudaMemcpy(results,resultsCuda,
      sizeof(long)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
      for(int col=0;col<steps/THREAD_NUM;col++){TOTAL+=results[col];}
        matched=false;
      }
      cudaMemcpy(rowCuda,totalRow,
      sizeof(uint64_t)*totalCond,cudaMemcpyHostToDevice);
      cudaMemcpy(downCuda,totalDown,
      sizeof(uint64_t)*totalCond,cudaMemcpyHostToDevice);
      cudaMemcpy(leftCuda,totalLeft,
      sizeof(uint64_t)*totalCond,cudaMemcpyHostToDevice);
      cudaMemcpy(rightCuda,totalRight,
      sizeof(uint64_t)*totalCond,cudaMemcpyHostToDevice);
      
      cudaMemcpy(typeCuda,totalType,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);

      //cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
      //>>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
      solve<<<steps/THREAD_NUM,THREAD_NUM>>>(g.size,0,typeCuda,rowCuda,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
      
      cudaMemcpy(results,resultsCuda,
      sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);

      matched=true;
      totalCond=0;
  }
  

}
//非再帰 対称解除法
void carryChain_symmetry(void* args)
{
  Local *l=(Local *)args;
  // 対称解除法
  unsigned const int ww=(g.size-2)*(g.size-1)-1-l->w;
  unsigned const int w2=(g.size-2)*(g.size-1)-1;
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
//非再帰  pthread run()
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
        //キャリーチェーンの結果を格納する
        //lの詰め替え
        totalL[totalcnt].n=l->n;
        totalL[totalcnt].e=l->e;
        totalL[totalcnt].s=l->s;
        totalL[totalcnt].w=l->w;
        totalL[totalcnt].dimx=l->dimx;
        totalL[totalcnt].dimy=l->dimy;
        totalL[totalcnt].B.row=l->B.row;
        totalL[totalcnt].B.down=l->B.down;
        totalL[totalcnt].B.left=l->B.left;
        totalL[totalcnt].B.right=l->B.right;
        for(unsigned int i=0;i<g.size;++i){ totalL[totalcnt].B.x[i]=l->B.x[i]; }
        totalcnt++;
        
        
      } //w
    } //e
  } //n



}
void execgpu(){
  cudaMalloc((void**) &rowCuda,sizeof(uint64_t)*steps);
  cudaMalloc((void**) &downCuda,sizeof(uint64_t)*steps);
  cudaMalloc((void**) &leftCuda,sizeof(uint64_t)*steps);
  /* cudaMalloc((void**) &rightCuda,sizeof(uint64_t)*steps); */
  cudaMalloc((void**) &typeCuda,sizeof(int)*steps);
  cudaMalloc((void**) &resultsCuda,sizeof(long)*steps/THREAD_NUM);
  for(int i=0;i<totalcnt;i++){

    carryChain_symmetry(&totalL[i]);
  }
  
  if(matched){
    cudaMemcpy(results,resultsCuda,
        sizeof(long)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    for(int col=0;col<steps/THREAD_NUM;col++){TOTAL+=results[col];}
    matched=false;
  }
  cudaMemcpy(rowCuda,totalRow,
      sizeof(uint64_t)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(downCuda,totalDown,
      sizeof(uint64_t)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(leftCuda,totalLeft,
      sizeof(uint64_t)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(rightCuda,totalRight,
      sizeof(uint64_t)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(typeCuda,totalType,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  solve<<<steps/THREAD_NUM,THREAD_NUM
      >>>(g.size,0,typeCuda,rowCuda,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
  cudaMemcpy(results,resultsCuda,
      sizeof(long)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  for(int col=0;col<steps/THREAD_NUM;col++){TOTAL+=results[col];}	

}

//非再帰  チェーンのビルド
void buildChain()
{
  Local l[(g.size/2)*(g.size-3)];
  // カウンターの初期化
  l->COUNT2=0; l->COUNT4=1; l->COUNT8=2;
  l->COUNTER[l->COUNT2]=l->COUNTER[l->COUNT4]=l->COUNTER[l->COUNT8]=0;
  // Board の初期化 nB,eB,sB,wB;
  l->B.row=l->B.down=l->B.left=l->B.right=0;
  // Board x[]の初期化
  for(unsigned int i=0;i<g.size;++i){ l->B.x[i]=-1; }
  //１ 上２行に置く
  // memcpy(&l->wB,&l->B,sizeof(Board));         // wB=B;
  l->wB=l->B;
  for(l->w=0;l->w<=(unsigned)(g.size/2)*(g.size-3);++l->w){
    thread_run(&l);

  } //w
  /*  
  if(matched){
    cudaMemcpy(results,resultsCuda,
        sizeof(long)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    for(int col=0;col<steps/THREAD_NUM;col++){TOTAL+=results[col];}
    matched=false;
  }
  cudaMemcpy(rowCuda,totalRow,
      sizeof(uint64_t)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(downCuda,totalDown,
      sizeof(uint64_t)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(leftCuda,totalLeft,
      sizeof(uint64_t)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(rightCuda,totalRight,
      sizeof(uint64_t)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(typeCuda,totalType,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  solve<<<steps/THREAD_NUM,THREAD_NUM
      >>>(g.size,0,typeCuda,rowCuda,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
  cudaMemcpy(results,resultsCuda,
      sizeof(long)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  for(int col=0;col<steps/THREAD_NUM;col++){TOTAL+=results[col];}	
  */
  /**
   * 集計
   */
  /*
  UNIQUE= l->COUNTER[l->COUNT2]+
          l->COUNTER[l->COUNT4]+
          l->COUNTER[l->COUNT8];
  TOTAL=  l->COUNTER[l->COUNT2]*2+
          l->COUNTER[l->COUNT4]*4+
          l->COUNTER[l->COUNT8]*8;
  */        
}
//非再帰  キャリーチェーン
void carryChain()
{
  listChain();  //チェーンのリストを作成
  buildChain(); // チェーンのビルド<--ここでチェーンのすべてを算出する
  execgpu();
  // calcChain(&l);  // 集計
}

  /**
   * 集計
   */
  /*
  UNIQUE= l->COUNTER[l->COUNT2]+
          l->COUNTER[l->COUNT4]+
          l->COUNTER[l->COUNT8];
  TOTAL=  l->COUNTER[l->COUNT2]*2+
          l->COUNTER[l->COUNT4]*4+
          l->COUNTER[l->COUNT8]*8;
  */        
/**
  GPU 
 */
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
    printf("  -n: GPU キャリーチェーン\n");
  }
  if(cpur){ printf("\n\nCPU キャリーチェーン 再帰 \n"); }
  else if(cpu){ printf("\n\nCPU キャリーチェーン 非再帰 \n"); }
  else if(gpu){ printf("\n\nGPU キャリーチェーン シングルスレッド\n"); }
  else if(gpuNodeLayer){ printf("\n\nGPU キャリーチェーン マルチスレッド\n"); }
  if(cpu||cpur)
  {
    int min=7; 
    int targetN=18;
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    for(int size=min;size<=targetN;size++){
      TOTAL=UNIQUE=0;
      //for(int i=0;i<steps;i++){
      //  results[i]=0;
      //}  
      totalCond=0;
      totalcnt=0;
      gettimeofday(&t0, NULL);//計測開始
      if(cpur){ //再帰
        g.size=size;
        //carryChainR();
        carryChain();
      }
      if(cpu){ //非再帰
        g.size=size;
        carryChain();
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
      printf("%2d:%13ld%12ld%8.2d:%02d:%02d:%02d.%02d\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
    } //end for
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
  }//end if
  if(gpu||gpuNodeLayer)
  {
    if(!InitCUDA()){return 0;}
    /* int steps=24576; */
    int min=7;
    int targetN=18;
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:            Total          Unique      dd:hh:mm:ss.ms");
    for(int size=min;size<=targetN;size++){
      gettimeofday(&t0,NULL);   // 計測開始
      totalCond=0;
      totalcnt=0;
      if(gpu){
        TOTAL=UNIQUE=0;
        g.size=size;
        carryChain();
        //TOTAL=carryChain_solve_nodeLayer(size,0,0,0); //キャリーチェーン
      }else if(gpuNodeLayer){
        TOTAL=UNIQUE=0;
        g.size=size;
        carryChain();
        //carryChain_build_nodeLayer(size); // キャリーチェーン
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
      printf("%2d:%13ld%12ld%8.2d:%02d:%02d:%02d.%02d\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
    }
  }//end if
  return 0;
}
