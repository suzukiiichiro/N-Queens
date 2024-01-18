/**
 *
 * bash版キャリーチェーンのC言語版のGPU/CUDA移植版
 * CUDAの実行をfor文の一番外側（上2行にクイーンを置いた後）
 * 
 * ・処理の流れ
 * 1,carryChain_build_nodeLayer()
 *   listChain() CPU,GPU共通  
 *     2行2列に配置するクイーンのパターンを作成する
 *   cuda_kernel()
 *     listChainで作成した、2行分クイーンを置くパターン数 size/2*(size-3)でCUDAを呼び出す
 * 2,cuda_kernel() GPUのみ  
 *   左2列、下2行、右2列にクイーンを置く  
 *   placement() CPU,GPU共通
 *     下左右2行2列でクイーンを置く処理、置けない場合は抜ける
 *   carryChain_symmetry()　CPU,GPU共通
 *     上下左右2行2列にクイーンを置けたものを対象解除してCOUNT2,COUT4,COUNT8,スキップを判定する
 * 3,carryChain_symmetry()　CPU,GPU共通
 *   対象解除判定
 *   solve() CPU,GPU共通
 *     バックトラックでクイーンを置いていく
 * 4,solve()
 *   バックトラック

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

アーキテクチャの指定（なくても問題なし、あれば高速）
-arch=sm_13 or -arch=sm_61

CPUの再帰での実行
$ nvcc -O3 -arch=sm_61 05CUDA_CarryChain_outside.cu && ./a.out -r

CPUの非再帰での実行
$ nvcc -O3 -arch=sm_61 05CUDA_CarryChain_outside.cu && ./a.out -c

GPUのシングルスレッド
$ nvcc -O3 -arch=sm_61 05CUDA_CarryChain_outside.cu && ./a.out -g

  GPUのマルチスレッド
$ nvcc -O3 -arch=sm_61 05CUDA_CarryChain_outside.cu && ./a.out -n
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
#define steps 960
// システムによって以下のマクロが必要であればコメントを外してください。
//#define UINT64_C(c) c ## ULL
//
// グローバル変数
unsigned long TOTAL=0; 
unsigned long UNIQUE=0;
// キャリーチェーン 非再帰版
// 構造体
typedef  struct
{
  unsigned int size;
  unsigned int pres_a[steps]; 
  unsigned int pres_b[steps];
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
}Local;
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
}
// クイーンの効きをチェック
//CPU GPU共通
__device__ __host__
bool placement(void* args,int size)
{
  Local *l=(Local *)args;
  if(l->B.x[l->dimx]==l->dimy){ return true;  }  
  if (l->B.x[0]==0){
    if (l->B.x[1]!=(uint64_t)-1){
      if((l->B.x[1]>=l->dimx)&&(l->dimy==1)){ return false; }
    }
  }else{
    if( (l->B.x[0]!=(uint64_t)-1) ){
      if(( (l->dimx<l->B.x[0]||l->dimx>=size-l->B.x[0])
        && (l->dimy==0 || l->dimy==size-1)
      )){ return 0; } 
      if ((  (l->dimx==size-1)&&((l->dimy<=l->B.x[0])||
          l->dimy>=size-l->B.x[0]))){
        return 0;
      } 
    }
  }
  l->B.x[l->dimx]=l->dimy;                    //xは行 yは列
  uint64_t row=UINT64_C(1)<<l->dimx;
  uint64_t down=UINT64_C(1)<<l->dimy;
  uint64_t left=UINT64_C(1)<<(size-1-l->dimx+l->dimy); //右上から左下
  uint64_t right=UINT64_C(1)<<(l->dimx+l->dimy);       // 左上から右下
  if((l->B.row&row)||(l->B.down&down)||(l->B.left&left)||(l->B.right&right)){ return false; }     
  l->B.row|=row; l->B.down|=down; l->B.left|=left; l->B.right|=right;
  return true;
}
//非再帰
__device__ __host__
uint64_t solve(int size,int current,uint64_t row,uint64_t left,uint64_t down,uint64_t right)
{
  //printf("solve\n");
  printf("current:%d\n",current);
  uint64_t row_a[MAX];
  uint64_t right_a[MAX];
  uint64_t left_a[MAX];
  uint64_t down_a[MAX];
  uint64_t bitmap_a[MAX];
  for (int i=0;i<size;i++){
    row_a[i]=0;
    left_a[i]=0;
    down_a[i]=0;
    right_a[i]=0;
    bitmap_a[i]=0;
  }
  row_a[current]=row;
  left_a[current]=left;
  down_a[current]=down;
  right_a[current]=right;
  uint64_t bitmap=bitmap_a[current]=~(left_a[current]|down_a[current]|right_a[current]);
  uint64_t total=0;
  uint64_t bit;
  while(current>-1){
    if((bitmap!=0||row&1)&&current<size){
      if(!(down+1)){
        total++;
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
  return total;
}
//非再帰 対称解除法
__device__ __host__
void carryChain_symmetry(void* args,int size)
{
  //printf("symmetry\n");
  Local *l=(Local *)args;
  // 対称解除法
  unsigned const int ww=(size-2)*(size-1)-1-l->w;
  unsigned const int w2=(size-2)*(size-1)-1;
  // # 対角線上の反転が小さいかどうか確認する
  if((l->s==ww)&&(l->n<(w2-l->e))){ return ; }
  // # 垂直方向の中心に対する反転が小さいかを確認
  if((l->e==ww)&&(l->n>(w2-l->n))){ return; }
  // # 斜め下方向への反転が小さいかをチェックする
  if((l->n==ww)&&(l->e>(w2-l->s))){ return; }
  // 枝刈り １行目が角の場合回転対称チェックせずCOUNT8にする
  if(l->B.x[0]==0){
    l->COUNTER[l->COUNT8]+=solve(size,0,l->B.row>>2,
    l->B.left>>4,((((l->B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,(l->B.right>>4)<<(size-5));
    return ;
  }
  // n,e,s==w の場合は最小値を確認する。右回転で同じ場合は、
  // w=n=e=sでなければ値が小さいのでskip  w=n=e=sであれば90度回転で同じ可能性
  if(l->s==l->w){ if((l->n!=l->w)||(l->e!=l->w)){ return; }
    l->COUNTER[l->COUNT2]+=solve(size,0,l->B.row>>2,
    l->B.left>>4,((((l->B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,(l->B.right>>4)<<(size-5));
    return;
  }
  // e==wは180度回転して同じ 180度回転して同じ時n>=sの時はsmaller?
  if((l->e==l->w)&&(l->n>=l->s)){ if(l->n>l->s){ return; }
    l->COUNTER[l->COUNT4]+=solve(size,0,l->B.row>>2,
    l->B.left>>4,((((l->B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,(l->B.right>>4)<<(size-5));
    return;
  }
  l->COUNTER[l->COUNT8]+=solve(size,0,l->B.row>>2,
  l->B.left>>4,((((l->B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,(l->B.right>>4)<<(size-5));
  return;
}
//CPU 非再帰
void cpu_kernel(
  unsigned int* pres_a,unsigned int* pres_b,unsigned int* results,int totalCond,int idx,int size){
  if(idx<totalCond){
    //printf("test\n"); 
    Local l[1];
    l->w=idx;
    // カウンターの初期化
    l->COUNT2=0; l->COUNT4=1; l->COUNT8=2;
    l->COUNTER[l->COUNT2]=l->COUNTER[l->COUNT4]=l->COUNTER[l->COUNT8]=0;
    // Board の初期化 nB,eB,sB,wB;
    l->B.row=l->B.down=l->B.left=l->B.right=0;
    // Board x[]の初期化
    for(unsigned int i=0;i<size;++i){ l->B.x[i]=-1; }
    //１ 上２行に置く
    // memcpy(&l.wB,&l.B,sizeof(Board));         // wB=B;
    l->wB=l->B;
    // memcpy(&l.B,&l.wB,sizeof(Board));       // B=wB;
    //l.B=l.wB;
    l->dimx=0; 
    l->dimy=pres_a[l->w];
    //if(!placement(l)){ continue; }
    if(!placement(l,size)){ 
      //printf("p1\n");
      results[idx]=0;
      goto end; 
    }
    l->dimx=1; l->dimy=pres_b[l->w];
    // if(!placement(l)){ continue; }
    if(!placement(l,size)){ 
      //printf("p2\n");
      results[idx]=0;
      goto end; 
    }
    //２ 左２行に置く
    // memcpy(&l.nB,&l.B,sizeof(Board));       // nB=B;
    l->nB=l->B;
    for(l->n=l->w;l->n<(size-2)*(size-1)-l->w;++l->n){
      // memcpy(&l.B,&l.nB,sizeof(Board));     // B=nB;
      l->B=l->nB;
      l->dimx=pres_a[l->n]; l->dimy=size-1;
      if(!placement(l,size)){ 
        continue; 
      }
      l->dimx=pres_b[l->n]; l->dimy=size-2;
      if(!placement(l,size)){ 
        continue; 
      }
      // ３ 下２行に置く
      // memcpy(&l.eB,&l.B,sizeof(Board));     // eB=B;
      l->eB=l->B;
      for(l->e=l->w;l->e<(size-2)*(size-1)-l->w;++l->e){
        // memcpy(&l.B,&l.eB,sizeof(Board));   // B=eB;
        l->B=l->eB;
        l->dimx=size-1; l->dimy=size-1-pres_a[l->e];
        if(!placement(l,size)){ 
          continue; 
        }
        l->dimx=size-2; l->dimy=size-1-pres_b[l->e];
        if(!placement(l,size)){ 
          continue; 
        }
        // ４ 右２列に置く
        // memcpy(&l.sB,&l.B,sizeof(Board));   // sB=B;
        l->sB=l->B;
        for(l->s=l->w;l->s<(size-2)*(size-1)-l->w;++l->s){
          // memcpy(&l->B,&l->sB,sizeof(Board)); // B=sB;
          l->B=l->sB;
          l->dimx=size-1-pres_a[l->s]; l->dimy=0;
          if(!placement(l,size)){ 
            continue; 
          }
          l->dimx=size-1-pres_b[l->s]; l->dimy=1;
          if(!placement(l,size)){ 
            continue; 
          }
          // 対称解除法
          carryChain_symmetry(l,size);
        } //w
      } //e
    } //n
    results[idx]=l->COUNTER[l->COUNT2]*2+l->COUNTER[l->COUNT4]*4+l->COUNTER[l->COUNT8]*8;
  }else{
    results[idx]=0;
  }
  end:
  
} 

void carryChain(int){
  
  listChain();  //チェーンのリストを作成
  int totalCond=g.size/2*(g.size-3);
  unsigned int* results=new unsigned int[totalCond];
  for(int i=0;i<totalCond;i++){
    cpu_kernel(g.pres_a,g.pres_b,results,totalCond,i,g.size);
  }
  for(int col=0;col<totalCond;col++){
    TOTAL+=results[col];
  }	
}
/**
  */
//再帰 ボード外側２列を除く内側のクイーン配置処理
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
__global__ void cuda_kernel(
  unsigned int* pres_a,unsigned int* pres_b,unsigned int* results,int totalCond,int size){
  const int tid=threadIdx.x;
  const int bid=blockIdx.x;
  const int idx=bid*blockDim.x+tid;
  __shared__ unsigned int sum[THREAD_NUM];
  if(idx<totalCond){
    //printf("test\n"); 
    Local l[1];
    l->w=idx;
    // カウンターの初期化
    l->COUNT2=0; l->COUNT4=1; l->COUNT8=2;
    l->COUNTER[l->COUNT2]=l->COUNTER[l->COUNT4]=l->COUNTER[l->COUNT8]=0;
    // Board の初期化 nB,eB,sB,wB;
    l->B.row=l->B.down=l->B.left=l->B.right=0;
    // Board x[]の初期化
    for(unsigned int i=0;i<size;++i){ l->B.x[i]=-1; }
    //１ 上２行に置く
    // memcpy(&l.wB,&l.B,sizeof(Board));         // wB=B;
    l->wB=l->B;
    // memcpy(&l.B,&l.wB,sizeof(Board));       // B=wB;
    //l.B=l.wB;
    l->dimx=0; 
    l->dimy=pres_a[l->w];
    //if(!placement(l)){ continue; }
    if(!placement(l,size)){ 
      //printf("p1\n");
      sum[tid]=0;
      goto end; 
    }
    l->dimx=1; l->dimy=pres_b[l->w];
    // if(!placement(l)){ continue; }
    if(!placement(l,size)){ 
      //printf("p2\n");
      sum[tid]=0;
      goto end; 
    }
    //２ 左２行に置く
    // memcpy(&l.nB,&l.B,sizeof(Board));       // nB=B;
    l->nB=l->B;
    for(l->n=l->w;l->n<(size-2)*(size-1)-l->w;++l->n){
      // memcpy(&l.B,&l.nB,sizeof(Board));     // B=nB;
      l->B=l->nB;
      l->dimx=pres_a[l->n]; l->dimy=size-1;
      if(!placement(l,size)){ 
        continue; 
      }
      l->dimx=pres_b[l->n]; l->dimy=size-2;
      if(!placement(l,size)){ 
        continue; 
      }
      // ３ 下２行に置く
      // memcpy(&l.eB,&l.B,sizeof(Board));     // eB=B;
      l->eB=l->B;
      for(l->e=l->w;l->e<(size-2)*(size-1)-l->w;++l->e){
        // memcpy(&l.B,&l.eB,sizeof(Board));   // B=eB;
        l->B=l->eB;
        l->dimx=size-1; l->dimy=size-1-pres_a[l->e];
        if(!placement(l,size)){ 
          continue; 
        }
        l->dimx=size-2; l->dimy=size-1-pres_b[l->e];
        if(!placement(l,size)){ 
          continue; 
        }
        // ４ 右２列に置く
        // memcpy(&l.sB,&l.B,sizeof(Board));   // sB=B;
        l->sB=l->B;
        for(l->s=l->w;l->s<(size-2)*(size-1)-l->w;++l->s){
          // memcpy(&l->B,&l->sB,sizeof(Board)); // B=sB;
          l->B=l->sB;
          l->dimx=size-1-pres_a[l->s]; l->dimy=0;
          if(!placement(l,size)){ 
            continue; 
          }
          l->dimx=size-1-pres_b[l->s]; l->dimy=1;
          if(!placement(l,size)){ 
            continue; 
          }
          // 対称解除法
          carryChain_symmetry(l,size);
        } //w
      } //e
    } //n
    sum[tid]=l->COUNTER[l->COUNT2]*2+l->COUNTER[l->COUNT4]*4+l->COUNTER[l->COUNT8]*8;
  }else{
    sum[tid]=0;
  }
  end:
  __syncthreads();if(tid<64&&tid+64<THREAD_NUM){sum[tid]+=sum[tid+64];} 
  __syncthreads();if(tid<32){sum[tid]+=sum[tid+32];} 
  __syncthreads();if(tid<16){sum[tid]+=sum[tid+16];} 
  __syncthreads();if(tid<8){sum[tid]+=sum[tid+8];} 
  __syncthreads();if(tid<4){sum[tid]+=sum[tid+4];} 
  __syncthreads();if(tid<2){sum[tid]+=sum[tid+2];} 
  __syncthreads();if(tid<1){sum[tid]+=sum[tid+1];} 
  __syncthreads();if(tid==0){results[bid]=sum[0];}
} 
void carryChain_build_nodeLayer(int){
  unsigned int* pres_a_Cuda;
  unsigned int* pres_b_Cuda;
  unsigned int* resultsCuda;  
  listChain();  //チェーンのリストを作成
  cudaMalloc((void**) &pres_a_Cuda,sizeof(int)*steps);
  cudaMalloc((void**) &pres_b_Cuda,sizeof(int)*steps);
  cudaMalloc((void**) &resultsCuda,sizeof(int)*steps/THREAD_NUM);
  int totalCond=g.size/2*(g.size-3);
  printf("totalCond:%d\n",totalCond);
  cudaMemcpy(pres_a_Cuda,g.pres_a,
      sizeof(int)*steps,cudaMemcpyHostToDevice);
  cudaMemcpy(pres_b_Cuda,g.pres_b,
      sizeof(int)*steps,cudaMemcpyHostToDevice);
  unsigned int* results=new unsigned int[steps];
  cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
    >>>(pres_a_Cuda,pres_b_Cuda,resultsCuda,totalCond,g.size);
  cudaMemcpy(results,resultsCuda,
      sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  for(int col=0;col<steps/THREAD_NUM;col++){TOTAL+=results[col];}	
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
    int min=4; 
    int targetN=15;
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    for(int size=min;size<=targetN;size++){
      TOTAL=UNIQUE=0;
      gettimeofday(&t0, NULL);//計測開始
      if(cpur){ //再帰
        g.size=size;
        carryChain(size);
       // carryChainR();
      }
      if(cpu){ //非再帰
        g.size=size;
        carryChain(size);
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
  }//end if
  if(gpu||gpuNodeLayer)
  {
    if(!InitCUDA()){return 0;}
    /* int steps=24576; */
    int min=7;
    int targetN=14;
    min=7;
    targetN=14;
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    for(int size=min;size<=targetN;size++){
      gettimeofday(&t0,NULL);   // 計測開始
      if(gpu){
        TOTAL=UNIQUE=0;
        g.size=size;
        carryChain_build_nodeLayer(size); // キャリーチェーン
        //TOTAL=carryChain_solve_nodeLayer(size,0,0,0); //キャリーチェーン
      }else if(gpuNodeLayer){
        TOTAL=UNIQUE=0;
        g.size=size;
        carryChain_build_nodeLayer(size); // キャリーチェーン
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
    }//end for
  }//end if
  return 0;
}
