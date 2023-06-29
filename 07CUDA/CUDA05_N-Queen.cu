/**
 CUDAで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイルと実行
 $ nvcc -O3 CUDA**_N-Queen.cu && ./a.out (-c|-r|-g)
                    -c:cpu 
                    -r cpu再帰 
                    -g GPU 

実行結果
$ nvcc -O3 CUDA05_N-Queen.cu  && ./a.out -g
５．GPU 再帰 バックトラック＋対称解除法＋枝刈りと最適化
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2               1  00:00:00:00.02
 5:           10               2  00:00:00:00.00
 6:            4               1  00:00:00:00.00
 7:           40               6  00:00:00:00.00
 8:           92              12  00:00:00:00.01
 9:          352              46  00:00:00:00.04
10:          724              92  00:00:00:00.15
11:         2680             341  00:00:00:00.70
12:        14200            1787  00:00:00:02.71
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//
#define THREAD_NUM		96
#define MAX 27
//変数宣言
int aBoard[MAX];
int down[2*MAX-1];  //down:flagA 縦 配置フラグ　
int left[2*MAX-1];  //left:flagB 斜め配置フラグ　
int right[2*MAX-1]; //right:flagC 斜め配置フラグ　
long TOTAL=0;       //CPU,CPUR
long UNIQUE=0;      //CPU,CPUR
int aT[MAX];        //aT:aTrial[]
int aS[MAX];        //aS:aScrath[]
//関数宣言 GPU
__global__
void nqueen_cuda(int *d_aBoard,int *d_aT,int *d_aS,int *d_down,int *d_right,int *d_left,long *d_results,long TOTAL,long UNIQUE,int row,int size);
void solve_nqueen_cuda(int si,long results[2],int steps);
__device__
int symmetryOps(int size,int *d_aBoard,int *d_aT,int *d_aS);
//関数宣言 GPU/CPU
__device__ __host__
void rotate(int chk[],int scr[],int n,int neg);
__device__ __host__
void vMirror(int chk[],int n);
__device__ __host__
int intncmp(int lt[],int rt[],int n);
//関数宣言 CPU
void TimeFormat(clock_t utime,char *form);
int symmetryOps(int si);
void NQueen(int row,int size);
void NQueenR(int row,int size);
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
//回転
__device__ __host__
void rotate(int chk[],int scr[],int n,int neg){
  int k=neg ? 0 : n-1;
  int incr=(neg ? +1 : -1);
  for(int j=0;j<n;k+=incr){
    scr[j++]=chk[k];
  }
  k=neg ? n-1 : 0;
  for(int j=0;j<n;k-=incr){
    chk[scr[j++]]=k;
  }
}
//反転
__device__ __host__
void vMirror(int chk[],int n){
  for(int j=0;j<n;j++){
    chk[j]=(n-1)-chk[j];
  }
}
//
__device__ __host__
int intncmp(int lt[],int rt[],int n){
  int rtn=0;
  for(int k=0;k<n;k++){
    rtn=lt[k]-rt[k];
    if(rtn!=0){
      break;
    }
  }
  return rtn;
}
//対称解除法
int symmetryOps(int size){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<size;i++){
    aT[i]=aBoard[i];
  }
  //時計回りに90度回転
  rotate(aT,aS,size,0);       
  int k=intncmp(aBoard,aT,size);
  if(k>0) return 0;
  if(k==0){
    nEquiv=1;
  }else{
    //時計回りに180度回転
    rotate(aT,aS,size,0);     
    k=intncmp(aBoard,aT,size);
    if(k>0) return 0;
    if(k==0){
      nEquiv=2;
    }else{
      //時計回りに270度回転
      rotate(aT,aS,size,0);   
      k=intncmp(aBoard,aT,size);
      if(k>0){
        return 0;
      }
      nEquiv=4;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<size;i++){
    aT[i]=aBoard[i];
  }
  //垂直反転
  vMirror(aT,size);           
  k=intncmp(aBoard,aT,size);
  if(k>0){
    return 0;
  }
  //-90度回転 対角鏡と同等
  if(nEquiv>1){             
    rotate(aT,aS,size,1);
    k=intncmp(aBoard,aT,size);
    if(k>0){
      return 0;
    }
    //-180度回転 水平鏡像と同等
    if(nEquiv>2){           
      rotate(aT,aS,size,1);
      k=intncmp(aBoard,aT,size);
      if(k>0){
        return 0;
      }  //-270度回転 反対角鏡と同等
      rotate(aT,aS,size,1);
      k=intncmp(aBoard,aT,size);
      if(k>0){
        return 0;
      }
    }
  }
  return nEquiv*2;
}
__device__
int symmetryOps(int size,int *d_aBoard,int *d_aT,int *d_aS){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<size;i++){
    d_aT[i]=d_aBoard[i];
  }
  //時計回りに90度回転
  rotate(d_aT,d_aS,size,0);       
  int k=intncmp(d_aBoard,d_aT,size);
  if(k>0) return 0;
  if(k==0){
    nEquiv=1;
  }else{
    //時計回りに180度回転
    rotate(d_aT,d_aS,size,0);     
    k=intncmp(d_aBoard,d_aT,size);
    if(k>0) return 0;
    if(k==0){
      nEquiv=2;
    }else{
      //時計回りに270度回転
      rotate(d_aT,d_aS,size,0);   
      k=intncmp(d_aBoard,d_aT,size);
      if(k>0){
        return 0;
      }
      nEquiv=4;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<size;i++){
    d_aT[i]=d_aBoard[i];
  }
  //垂直反転
  vMirror(d_aT,size);           
  k=intncmp(d_aBoard,d_aT,size);
  if(k>0){
    return 0;
  }
  //-90度回転 対角鏡と同等
  if(nEquiv>1){             
    rotate(d_aT,d_aS,size,1);
    k=intncmp(d_aBoard,d_aT,size);
    if(k>0){
      return 0;
    }
    //-180度回転 水平鏡像と同等
    if(nEquiv>2){           
      rotate(d_aT,d_aS,size,1);
      k=intncmp(d_aBoard,d_aT,size);
      if(k>0){
        return 0;
      }  
      //-270度回転 反対角鏡と同等
      rotate(d_aT,d_aS,size,1);
      k=intncmp(d_aBoard,d_aT,size);
      if(k>0){
        return 0;
      }
    }
  }
  return nEquiv*2;
}
//
__global__
void nqueen_cuda(int *d_aBoard,int *d_aT,int *d_aS,int *d_down,int *d_right,int *d_left,long *d_results,long TOTAL,long UNIQUE,int row,int size){
  bool matched;
  int sizeE=size-1;
  while(row>=0){
    matched=false;
    /** 枝刈り */
    int lim=(row!=0)?size:(size+1)/2;
    for(int col=d_aBoard[row]+1;col<lim;col++){
      if(d_down[col]==0
          && d_right[col-row+sizeE]==0
          && d_left[col+row]==0){
        if(d_aBoard[row]!=-1){
          d_down[d_aBoard[row]]
            =d_right[d_aBoard[row]-row+sizeE]
            =d_left[d_aBoard[row]+row]=0;
        }
        d_aBoard[row]=col;
        d_down[col]
          =d_right[col-row+sizeE]
          =d_left[col+row]=1;
        matched=true;
        break;
      }
    }
    if(matched){
      row++;
      if(row==size){
        int s=symmetryOps(size,d_aBoard,d_aT,d_aS);
        if(s!=0){
          //print(size); //print()でTOTALを++しない
          //ホストに戻す配列にTOTALを入れる
          //スレッドが１つの場合は配列は１個
          d_results[1]=++UNIQUE; 
          d_results[0]+=s;   //対称解除で得られた解数を加算
        }
        row--;
      }
    }else{
      if(d_aBoard[row]!=-1){
        int col=d_aBoard[row];
        d_down[col]
          =d_right[col-row+sizeE]
          =d_left[col+row]=0;
        d_aBoard[row]=-1;
      }
      row--;
    }
  }
}
//
void solve_nqueen_cuda(int si,long results[2],int steps){
  //メモリ管理 
  int *h_aBoard;
  int *h_aT;
  int *h_aS;
  int *h_down;
  int *h_right;
  int *h_left;
  long *h_results;
  cudaMallocHost((void**)&h_aBoard,sizeof(int)*MAX);
  cudaMallocHost((void**)&h_aT,sizeof(int)*MAX);
  cudaMallocHost((void**)&h_aS,sizeof(int)*MAX);
  cudaMallocHost((void**)&h_down,sizeof(int)*2*MAX-1);
  cudaMallocHost((void**)&h_right,sizeof(int)*2*MAX-1);
  cudaMallocHost((void**)&h_left,sizeof(int)*2*MAX-1);
  cudaMallocHost((void**)&h_results,sizeof(long)*steps);
  int *d_aBoard;
  int *d_aT;
  int *d_aS;
  int *d_down;
  int *d_right;
  int *d_left;
  long *d_results;
  cudaMalloc((void**)&d_aBoard,sizeof(int)*MAX);
  cudaMalloc((void**)&d_aT,sizeof(int)*MAX);
  cudaMalloc((void**)&d_aS,sizeof(int)*MAX);
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
  cudaMemcpy(d_aT,h_aT,
    sizeof(int)*MAX,cudaMemcpyHostToDevice);
  cudaMemcpy(d_aS,h_aS,
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
  nqueen_cuda<<<1,1>>>(d_aBoard,d_aT,d_aS,d_down,d_right,d_left,d_results,0,0,0,si);
  //device to host
  cudaMemcpy(h_results,d_results,
    sizeof(long)*steps,cudaMemcpyDeviceToHost);
  //結果の代入
  results[0]=h_results[0];
  results[1]=h_results[1];
  //メモリ解放
  cudaFreeHost(h_aBoard);
  cudaFreeHost(h_aT);
  cudaFreeHost(h_aS);
  cudaFreeHost(h_down);
  cudaFreeHost(h_right);
  cudaFreeHost(h_left);
  cudaFreeHost(h_results);
  cudaFree(d_aBoard);
  cudaFree(d_aT);
  cudaFree(d_aS);
  cudaFree(d_down);
  cudaFree(d_left);
  cudaFree(d_right);
  cudaFree(d_results);
}
//
//CPU 非再帰版 ロジックメソッド
void NQueen(int row,int size){
  bool matched;
  int sizeE=size-1;
  while(row>=0){
    matched=false;
    /** 枝刈り */
    int lim=(row!=0)?size:(size+1)/2;
    for(int col=aBoard[row]+1;col<lim;col++){
      if(down[col]==0
          && right[col-row+sizeE]==0
          && left[col+row]==0){
        if(aBoard[row]!=-1){
          down[aBoard[row]]
            =right[aBoard[row]-row+sizeE]
            =left[aBoard[row]+row]=0;
        }
        aBoard[row]=col;
        down[col]
          =right[col-row+sizeE]
          =left[col+row]=1;
        matched=true;
        break;
      }
    }
    if(matched){
      row++;
      if(row==size){
        int s=symmetryOps(size);
        if(s!=0){
          UNIQUE++;
          TOTAL+=s;
        }
        row--;
      }
    }else{
      if(aBoard[row]!=-1){
        int col=aBoard[row];
        down[col]
          =right[col-row+sizeE]
          =left[col+row]=0;
        aBoard[row]=-1;
      }
      row--;
    }
  }
}
//CPUR 再帰版 ロジックメソッド
void NQueenR(int row,int size){
  int sizeE=size-1;
  if(row==size){
    int s=symmetryOps(size);  //対称解除法の導入
    if(s!=0){
      UNIQUE++;
      TOTAL+=s;
    }
  }else{
    /** 枝刈り */
    int lim=(row!=0) ? size : (size+1)/2;
    for(int col=aBoard[row]+1;col<lim;col++){
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
      aBoard[row]=-1;
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
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  if(cpu){
    printf("\n\n５．CPU 非再帰 バックトラック＋対称解除法＋枝刈りと最適化\n");
  }else if(cpur){
    printf("\n\n５．CPUR 再帰 バックトラック＋対称解除法＋枝刈りと最適化\n");
  }else if(gpu){
    printf("\n\n５．GPU 非再帰 バックトラック＋対象解除法＋枝刈りと最適化\n");
  }else if(sgpu){
    printf("\n\n５．SGPU 非再帰 バックトラック＋対象解除法＋枝刈りと最適化\n");
  }
  if(cpu||cpur){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    clock_t st;           //速度計測用
    char t[20];           //hh:mm:ss.msを格納
    int min=4; int 
      targetN=18;
    for(int i=min;i<=targetN;i++){
      //aBoard配列を-1で初期化
      for(int j=0;j<=targetN;j++){ aBoard[j]=-1; }
      TOTAL=0; 
      UNIQUE=0;
      st=clock();
      if(cpu){ NQueen(0,i); }
      if(cpur){ NQueenR(0,i); }
      TimeFormat(clock()-st,t); 
      printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
    }
  }
  if(gpu||sgpu){
    if(!InitCUDA()){return 0;}
    int min=4;int targetN=18;
    struct timeval t0;
    struct timeval t1;
    int ss;int ms;int dd;
    long TOTAL,UNIQUE;
    long results[2];//結果格納用
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    for(int i=min;i<=targetN;i++){
      gettimeofday(&t0,NULL);   // 計測開始
      if(gpu){
        solve_nqueen_cuda(i,results,steps);
        TOTAL=results[0];
        UNIQUE=results[1];
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
