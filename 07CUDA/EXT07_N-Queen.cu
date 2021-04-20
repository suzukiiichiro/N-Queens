
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


 ７．バックトラック＋ビットマップ＋対称解除法
 *     一つの解には、盤面を９０度、１８０度、２７０度回転、及びそれらの鏡像の合計
 *     ８個の対称解が存在する。対照的な解を除去し、ユニーク解から解を求める手法。
 * 
 * ■ユニーク解の判定方法
 *   全探索によって得られたある１つの解が、回転・反転などによる本質的に変わること
 * のない変換によって他の解と同型となるものが存在する場合、それを別の解とはしない
 * とする解の数え方で得られる解を「ユニーク解」といいます。つまり、ユニーク解とは、
 * 全解の中から回転・反転などによる変換によって同型になるもの同士をグループ化する
 * ことを意味しています。
 * 
 *   従って、ユニーク解はその「個数のみ」に着目され、この解はユニーク解であり、こ
 * の解はユニーク解ではないという定まった判定方法はありません。ユニーク解であるか
 * どうかの判断はユニーク解の個数を数える目的の為だけに各個人が自由に定義すること
 * になります。もちろん、どのような定義をしたとしてもユニーク解の個数それ自体は変
 * わりません。
 * 
 *   さて、Ｎクイーン問題は正方形のボードで形成されるので回転・反転による変換パター
 * ンはぜんぶで８通りあります。だからといって「全解数＝ユニーク解数×８」と単純には
 * いきません。ひとつのグループの要素数が必ず８個あるとは限らないのです。Ｎ＝５の
 * 下の例では要素数が２個のものと８個のものがあります。
 *
 *
 * Ｎ＝５の全解は１０、ユニーク解は２なのです。
 * 
 * グループ１: ユニーク解１つ目
 * - - - Q -   - Q - - -
 * Q - - - -   - - - - Q
 * - - Q - -   - - Q - -
 * - - - - Q   Q - - - -
 * - Q - - -   - - - Q -
 * 
 * グループ２: ユニーク解２つ目
 * - - - - Q   Q - - - -   - - Q - -   - - Q - -   - - - Q -   - Q - - -   Q - - - -   - - - - Q
 * - - Q - -   - - Q - -   Q - - - -   - - - - Q   - Q - - -   - - - Q -   - - - Q -   - Q - - -
 * Q - - - -   - - - - Q   - - - Q -   - Q - - -   - - - - Q   Q - - - -   - Q - - -   - - - Q -
 * - - - Q -   - Q - - -   - Q - - -   - - - Q -   - - Q - -   - - Q - -   - - - - Q   Q - - - -
 * - Q - - -   - - - Q -   - - - - Q   Q - - - -   Q - - - -   - - - - Q   - - Q - -   - - Q - -
 *
 * 
 *   それでは、ユニーク解を判定するための定義付けを行いますが、次のように定義する
 * ことにします。各行のクイーンが右から何番目にあるかを調べて、最上段の行から下
 * の行へ順番に列挙します。そしてそれをＮ桁の数値として見た場合に最小値になるもの
 * をユニーク解として数えることにします。尚、このＮ桁の数を以後は「ユニーク判定値」
 * と呼ぶことにします。
 * 
 * - - - - Q   0
 * - - Q - -   2
 * Q - - - -   4   --->  0 2 4 1 3  (ユニーク判定値)
 * - - - Q -   1
 * - Q - - -   3
 * 
 * 
 *   探索によって得られたある１つの解(オリジナル)がユニーク解であるかどうかを判定
 * するには「８通りの変換を試み、その中でオリジナルのユニーク判定値が最小であるか
 * を調べる」ことになります。しかし結論から先にいえば、ユニーク解とは成り得ないこ
 * とが明確なパターンを探索中に切り捨てるある枝刈りを組み込むことにより、３通りの
 * 変換を試みるだけでユニーク解の判定が可能になります。
 *  
 実行結果

$ nvcc CUDA07_N-Queen.cu  && ./a.out -r
７．CPUR 再帰 バックトラック＋ビットマップ＋対称解除法
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
13:        73712            9233            0.08
14:       365596           45752            0.48
15:      2279184          285053            3.20
16:     14772512         1846955           22.49
17:     95815104        11977939         2:41.93

$ nvcc CUDA07_N-Queen.cu  && ./a.out -c
７．CPU 非再帰 バックトラック＋ビットマップ＋対称解除法
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
13:        73712            9233            0.09
14:       365596           45752            0.49
15:      2279184          285053            3.25
16:     14772512         1846955           22.96
17:     95815104        11977939         2:43.94

bash-3.2$ nvcc CUDA06_N-Queen.cu && ./a.out -s
６．SGPU 非再帰 バックトラック＋ビットマップ
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2               0  00:00:00:00.02
 5:           10               0  00:00:00:00.00
 6:            4               0  00:00:00:00.00
 7:           40               0  00:00:00:00.00
 8:           92               0  00:00:00:00.00
 9:          352               0  00:00:00:00.00
10:          724               0  00:00:00:00.00
11:         2680               0  00:00:00:00.00
12:        14200               0  00:00:00:00.02
13:        73712               0  00:00:00:00.03
14:       365596               0  00:00:00:00.07
15:      2279184               0  00:00:00:00.48
16:     14772512               0  00:00:00:02.40
17:     95815104               0  00:00:00:18.30

$ nvcc CUDA07_N-Queen.cu  && ./a.out -g
７．GPU 非再帰 バックトラック＋ビットマップ＋対称解除法
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2               1  00:00:00:00.03
 5:           10               5  00:00:00:00.00
 6:            4               2  00:00:00:00.00
 7:           40              20  00:00:00:00.01
 8:           92              46  00:00:00:00.00
 9:          352             176  00:00:00:00.01
10:          724             362  00:00:00:00.04
11:         2680            1340  00:00:00:00.12
12:        14200            7100  00:00:00:00.50
13:        73712           36856  00:00:00:00.90
14:       365596          182798  00:00:00:00.94
15:      2279184         1139592  00:00:00:05.50
16:     14772512         7386256  00:00:00:37.28
17:     95815104        47907552  00:00:04:44.32
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
//#define THREAD_NUM		1
#define MAX 27
//変数宣言
long TOTAL=0; //GPU,CPUで使用
/***07 uniq*************************************/
long UNIQUE=0;//GPU,CPUで使用
/****************************************/
/***07 グローバルで使用していないためコメント*************************************/
//int down[2*MAX-1]; //down:flagA 縦 配置フラグ　//CPUで使用
//int left[2*MAX-1];  //left:flagB 斜め配置フラグ　//CPUで使用
//int right[2*MAX-1];  //right:flagC 斜め配置フラグ　//CPUで使用
/****************************************/
/***07 aBoardローカル化のためコメント*************************************/
//unsigned int aBoard[MAX];//CPU,GPUで使用
/****************************************/
/***07 aT,aSローカル化のためコメント.CPU,GPU同一関数化のためコメント*************************************/
//int aT[MAX];//CPUで使用
//int aS[MAX];//CPUで使用
//int COUNT2,COUNT4,COUNT8;//CPUで使用
/****************************************/
//関数宣言 GPU
//関数宣言 GPU/CPU
/***07 rh,vMirror同一化のためコメント*************************************/
//__device__ __host__ int rh(int a,int sz);
/****************************************
// **07 配列のポインタを戻り値で返却するように変更*************************************/
//__device__ __host__ void vMirror_bitmap(int bf[],int af[],int si);
//__device__ __host__ void rotate_bitmap(int bf[],int af[],int si);
__device__ __host__ int* vMirror_bitmap(int bf[],int af[],int si);
__device__ __host__ int* rotate_bitmap(int bf[],int af[],int si);
/****************************************
__device__ __host__ int intncmp(unsigned int lt[],int rt[],int n);
// 07 aT,aSロカール化,CPU,GPU同一関数化*********************************** **/
//__device__ int symmetryOps_bitmap_gpu(int si,int *d_aBoard,int *d_aT,int *d_aS);
/****************************************/
__device__  __host__ int symmetryOps_bitmap(int si,int *d_aBoard);
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
    unsigned int* d_results,unsigned int* d_uniq,int totalCond,unsigned int* t_aBoard,int h_row);
/****************************************/
/***07 aBoardローカル化*************************************/
//long solve_nqueen_cuda(int size,int mask,int row,int n_left,int n_down,int n_right,int steps);
long solve_nqueen_cuda(int size,int mask,int row,int n_left,int n_down,int n_right,int steps,unsigned int* aBoard);
/****************************************/
void NQueenG(int size,int steps);
//関数宣言 SGPU
__global__ 
void sgpu_cuda_kernel(int size,int mark,unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,unsigned int* results,int totalCond);
long long sgpu_solve_nqueen_cuda(int size,int steps);
bool InitCUDA();
//関数宣言 CPU
void TimeFormat(clock_t utime,char *form);
/***07 symmetryOpsCPU,GPU同一関数化のためコメント*************************************/
//long getUnique();
//long getTotal();
//void symmetryOps_bitmap(int si);
//関数宣言 非再帰版
/***07 aBoardロカール化*************************************/
//void solve_nqueen(int size,int mask, int row,int* left,int* down,int* right,int* bitmap);
void solve_nqueen(int size,int mask, int row,int* left,int* down,int* right,int* bitmap,unsigned int* aBoard);
/****************************************/
void NQueen(int size,int mask);
//関数宣言 GPUへの移行再帰版
/***07 aBoardロカール化*************************************/
void solve_nqueenr(int size,int mask, int row,int left,int down,int right,unsigned int* aBoard);
//void solve_nqueenr(int size,int mask, int row,int left,int down,int right);
/****************************************/
void NQueenR(int size,int mask);
//関数宣言 通常版
//  再帰
void NQueenDR(int size,int mask,int row,int left,int down,int right);
//  非再帰
void NQueenD(int size,int mask,int row);
//
//GPU マルチスレッド
//
/***07 symmetryOps*************************************/
/***07 vMirror,rh同一化のためコメント*************************************/
/**
__device__ __host__
int rh(int a,int sz)
{
  int tmp=0;
  for(int i=0;i<=sz;i++){
    if(a&(1<<i)){ return tmp|=(1<<(sz-i)); }
  }
  return tmp;
}
**/
/****************************************/
//
/***07 symmetryOps*************************************/
__device__ __host__
//void vMirror_bitmap(int bf[],int af[],int si)
int* vMirror_bitmap(int bf[],int af[],int si)
{
  int score ;
  for(int i=0;i<si;i++) {
    score=bf[i];
    //af[i]=rh(score,si-1);
    int t=0;
    for(int j=0;j<=si-1;j++){
      if(score&(1<<j)){ 
      //if(bf[i]&(1<<j)){ 
        t|=(1<<(si-1-j)); 
        break;                 
      }
    }
    af[i]=t;
  }
  return af;
}
/***07 vMirror,rh同一化のためコメント*************************************/
/**
__device__ __host__
void vMirror_bitmap_old(int bf[],int af[],int si)
{
  int score ;
  for(int i=0;i<si;i++) {
    score=bf[i];
    af[i]=rh(score,si-1);
  }
 **/

//
/***07 symmetryOps*************************************/
__device__ __host__
//void rotate_bitmap(int bf[],int af[],int si)
int* rotate_bitmap(int bf[],int af[],int si)
{
  for(int i=0;i<si;i++){
    int t=0;
    for(int j=0;j<si;j++){
      t|=((bf[j]>>i)&1)<<(si-j-1);
    }
    af[i]=t;
  }
  return af;
}
/****************************************/
//
/***07 symmetryOps*************************************/
__device__ __host__
int ncmp(unsigned int lt[],int rt[],int n,int icmp)
{
  for(int k=0;k<n;k++){
    icmp=lt[k]-rt[k];
    if(icmp!=0){
      break;
    }
  }
  return icmp;
}
__device__ __host__
int intncmp(unsigned int lt[],int rt[],int n)
{
  int rtn=0;
  for(int k=0;k<n;k++){
    rtn=lt[k]-rt[k];
    if(rtn!=0){
      break;
    }
  }
  return rtn;
}
/****************************************/
//
/***07 symmetryOps*************************************/
__device__ __host__
int symmetryOps_bitmap(int si,unsigned int *aBoard)
{
  int nEquiv;
  int aT[MAX];
  int aS[MAX];
  int icmp=0;
  // 回転・反転・対称チェックのためにboard配列をコピー
  //for(int i=0;i<si;i++){ aS[i]=aT[i]=aBoard[i];}
  memcpy(aT,aBoard,sizeof(int)*si);
  //時計回りに90度回転
  rotate_bitmap(aT,aS,si);
  //icmp=intncmp(aBoard,aS,si);
  ncmp(aBoard,aS,si,icmp);
  //if(intncmp(aBoard,aS,si)>0){ return 0; }
  if(icmp>0){ return 0; }
  if(icmp==0){ nEquiv=2; }
  else{
    //時計回りに180度回転
    rotate_bitmap(aS,aT,si);
    //icmp=intncmp(aBoard,aT,si);
    ncmp(aBoard,aT,si,icmp);
    if(icmp>0){ return 0;}
    if(icmp==0){ nEquiv=4;}
    else{
      //時計回りに270度回転
      rotate_bitmap(aT,aS,si);
      //icmp=intncmp(aBoard,aS,si);
      ncmp(aBoard,aS,si,icmp);
      if(icmp>0){ return 0;}
      nEquiv=8;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  //for(int i=0;i<si;i++){ aS[i]=aBoard[i];}
  memcpy(aS,aBoard,sizeof(int)*si);
  //垂直反転
  vMirror_bitmap(aS,aT,si);   
  icmp=ncmp(aBoard,aT,si,icmp);
  if(icmp>0){ return 0; }
  //-90度回転 対角鏡と同等
  if(nEquiv>2){
    rotate_bitmap(aT,aS,si);
    icmp=ncmp(aBoard,aS,si,icmp);
    if(icmp>0){return 0;}
    //-180度回転 水平鏡像と同等
    if(nEquiv>4){
      rotate_bitmap(aS,aT,si);
      icmp=ncmp(aBoard,aT,si,icmp);
      //-270度回転 反対角鏡と同等
      if(icmp>0){ return 0;}
      rotate_bitmap(aT,aS,si);
      icmp=ncmp(aBoard,aS,si,icmp);
      if(icmp>0){ return 0;}
    }
  }
  return nEquiv;  
}
/****************************************/
/**
__device__
int symmetryOps_bitmap_gpu_old(int si,unsigned int *d_aBoard,unsigned int *d_aT,unsigned int *d_aS)
{
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ d_aT[i]=d_aBoard[i];}
  rotate_bitmap(d_aT,d_aS,si);    //時計回りに90度回転
  int k=intncmp(d_aBoard,d_aS,si);
  //printf("1_k:%d\n",k);
  if(k>0)return 0;
  if(k==0){ nEquiv=2;}else{
    rotate_bitmap(d_aS,d_aT,si);  //時計回りに180度回転
    k=intncmp(d_aBoard,d_aT,si);
    //printf("2_k:%d\n",k);
    if(k>0)return 0;
    if(k==0){ nEquiv=4;}else{
      rotate_bitmap(d_aT,d_aS,si);//時計回りに270度回転
      k=intncmp(d_aBoard,d_aS,si);
      //printf("3_k:%d\n",k);
      if(k>0){ return 0;}
      nEquiv=8;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ d_aS[i]=d_aBoard[i];}
  vMirror_bitmap(d_aS,d_aT,si);   //垂直反転
  k=intncmp(d_aBoard,d_aT,si);
  //printf("4_k:%d\n",k);
  if(k>0){ return 0; }
  if(nEquiv>2){             //-90度回転 対角鏡と同等
    rotate_bitmap(d_aT,d_aS,si);
    k=intncmp(d_aBoard,d_aS,si);
    //printf("5_k:%d\n",k);
    if(k>0){return 0;}
    if(nEquiv>4){           //-180度回転 水平鏡像と同等
      rotate_bitmap(d_aS,d_aT,si);
      k=intncmp(d_aBoard,d_aT,si);
      //printf("6_k:%d\n",k);
      if(k>0){ return 0;}       //-270度回転 反対角鏡と同等
      rotate_bitmap(d_aT,d_aS,si);
      k=intncmp(d_aBoard,d_aS,si);
      //printf("7_k:%d\n",k);
      if(k>0){ return 0;}
    }
  }
  //printf("eq:%d\n",nEquiv);
  return nEquiv;  
}
**/
//
//GPU
/***07 引数 追加に伴いコメント*********************/
//__global__ 
//void cuda_kernel(int size,int mark,unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,unsigned int* d_results,int totalCond)
/************************/
/***07 引数 d_uniq,t_aBoard,h_row追加 uniq,aBoardのため*********************/
__global__
void cuda_kernel(
    register int size,
    register int mark,
    unsigned int* totalDown,
    unsigned int* totalLeft,
    unsigned int* totalRight,
    unsigned int* d_results,
    unsigned int* d_uniq,
    register int totalCond,
    unsigned int* t_aBoard,
    register int h_row)
    /***07 aT,aS ローカル化*********************/
    //int* aT,
    //int* aS
    /************************/
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
  unsigned int c_aBoard[MAX];
  /***07 aT,aSローカル化*********************/
  //unsigned int c_aT[MAX];
  //unsigned int c_aS[MAX];
  /************************/
  __shared__ unsigned int usum[THREAD_NUM];
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
           /***07 aT,aSローカル化*********************/
           int s=symmetryOps_bitmap(size,c_aBoard); 
           //int s=symmetryOps_bitmap_gpu(size,c_aBoard,c_aT,c_aS);
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
  __syncwarp();if(tid<32){
    sum[tid]+=sum[tid+32];
    /***07 uniq追加*********************/
    usum[tid]+=usum[tid+32];
    /************************/
  } 
  __syncwarp();if(tid<16){
    sum[tid]+=sum[tid+16];
    /***07 uniq追加*********************/
    usum[tid]+=usum[tid+16];
    /************************/  
  } 
  __syncwarp();if(tid<8){
    sum[tid]+=sum[tid+8];
    /***07 uniq追加*********************/
    usum[tid]+=usum[tid+8];
    /************************/
  } 
  __syncwarp();if(tid<4){
    sum[tid]+=sum[tid+4];
    /***07 uniq追加*********************/
    usum[tid]+=usum[tid+4];
    /************************/  
  } 
  __syncwarp();if(tid<2){
    sum[tid]+=sum[tid+2];
    /***07 uniq追加*********************/
    usum[tid]+=usum[tid+2];
    /************************/  
  } 
  __syncwarp();if(tid<1){
    sum[tid]+=sum[tid+1];
    /***07 uniq追加*********************/
    usum[tid]+=usum[tid+1];
    /************************/  
  } 
  __syncwarp();if(tid==0){
    d_results[bid]=sum[0];
    /****07 uniq追加********************/
    d_uniq[bid]=usum[0];
    /************************/
  }
}
//
// GPU
 /****07 aBoardローカル化********************/
//long solve_nqueen_cuda(int size,int mask,int row,int n_left,int n_down,int n_right,int steps)
long solve_nqueen_cuda(int size,int mask,int row,int n_left,int n_down,int n_right,int steps,unsigned int* aBoard)
/************************/
{
  //何行目からGPUで行くか。ここの設定は変更可能、設定値を多くするほどGPUで並行して動く
  const unsigned int mark=size>11?size-10:2;
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
  /***07 aT,aSローカル化*********************/
  //unsigned int* d_aT;
  //cudaMalloc((void**) &d_aT,sizeof(int)*steps*MAX);
  //unsigned int* d_aS;
  //cudaMalloc((void**) &d_aS,sizeof(int)*steps*MAX);
  /************************/
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
              >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,row);
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
    >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,mark);
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
//GPU
void NQueenG(int size,int steps)
{
  /***07 aBoardローカル化*********************/
  unsigned int aBoard[MAX];
  /************************/
  //register int sizeE=size-1;
  register int bit=0;
  register int mask=((1<<size)-1);
  if(size<=0||size>32){return;}
  /***07 ミラーリングしない*********************/
  //偶数、奇数共通
  for(int col=0;col<size;col++){
    aBoard[0]=bit=(1<<col);
    /***07 aBoardローカル化*********************/
    TOTAL+=solve_nqueen_cuda(size,mask,1,bit<<1,bit,bit>>1,steps,aBoard);
    //TOTAL+=solve_nqueen_cuda(size,mask,1,bit<<1,bit,bit>>1,steps);
    /************************/
  }
  /************************/
  /***07 ミラーリングしないためコメント*********************/
  //偶数、奇数共通 右側半分だけクイーンを置く
	//int lim=(size%2==0)?size/2:sizeE/2;
  //for(int col=0;col<lim;col++){
  //  bit=(1<<col);
  //  TOTAL+=solve_nqueen_cuda(size,mask,1,bit<<1,bit,bit>>1,steps);
  //}
  //ミラーなのでTOTALを２倍する
  //TOTAL=TOTAL*2;
  //奇数の場合はさらに中央にクイーンを置く
  //if(size%2==1){
  //  bit=(1<<(sizeE)/2);
  //  TOTAL+=solve_nqueen_cuda(size,mask,1,bit<<1,bit,bit>>1,steps);
  //}
  /************************/
}
//SGPU
__global__ 
void sgpu_cuda_kernel(int size,int mark,unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,unsigned int* results,int totalCond)
{
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
//SGPU
long long sgpu_solve_nqueen_cuda(int size,int steps)
{
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
/** GPU/SGPU CUDA 初期化 **/
bool InitCUDA()
{
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
//CPU/GPU
//hh:mm:ss.ms形式に処理時間を出力
void TimeFormat(clock_t utime,char *form)
{
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
/***07 CPU,GPU同一関数化のためコメント*********************/
/**
long getUnique()
{
  return COUNT2+COUNT4+COUNT8;
}
//
long getTotal()
{
  return COUNT2*2+COUNT4*4+COUNT8*8;
}
**/
//CPU
/***07 symmetryOps_titmap CPU,GPU同関数化のためコメント*********************/
/**
void symmetryOps_bitmap(int si)
{
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
**/
//
//CPU 非再帰版 ロジックメソッド
/***07 aBoardローカル化*********************/
//void solve_nqueen(int size,int mask, int row,int* left,int* down,int* right,int* bitmap)
void solve_nqueen(int size,int mask, int row,int* left,int* down,int* right,int* bitmap,unsigned int* aBoard)
/************************/
{
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
            /***07 symmetryOps CPU,GPU同一化*********************/
            int s=symmetryOps_bitmap(size,aBoard);
            if(s!=0){
              UNIQUE++;
              TOTAL+=s;
            }
            /************************/
            --row;
          }else{
            int n=row++;
            left[row]=(left[n]|bit)<<1;
            down[row]=down[n]|bit;
            right[row]=(right[n]|bit)>>1;
            bitmap[row]=mask&~(left[row]|down[row]|right[row]);
          }
        }else{
           --row;
        }
      }  
    }
}
//
//非再帰版
void NQueen(int size,int mask)
{
  register int bitmap[size];
  register int down[size],right[size],left[size];
  register int bit;
  /***07 aBoardローカル化*********************/
  unsigned int aBoard[MAX];
  /************************/  
  if(size<=0||size>32){return;}
  bit=0;
  bitmap[0]=mask;
  down[0]=left[0]=right[0]=0;
  //偶数、奇数共通
  for(int col=0;col<size;col++){
    aBoard[0]=bit=(1<<col);
    down[1]=bit;//再帰の場合は down,left,right,bitmapは現在の行だけで良いが
    left[1]=bit<<1;//非再帰の場合は全行情報を配列に入れて行の上がり下がりをする
    right[1]=bit>>1;
    bitmap[1]=mask&~(left[1]|down[1]|right[1]);
    /***07 aBoardローカル化*********************/
    solve_nqueen(size,mask,1,left,down,right,bitmap,aBoard);
    //solve_nqueen(size,mask,1,left,down,right,bitmap);
    /************************/  
  }
}
//CPUR 再帰版 ロジックメソッド
/***07 aBoardローカル化*********************/
void solve_nqueenr(int size,int mask, int row,int left,int down,int right,unsigned int* aBoard)
//void solve_nqueenr(int size,int mask, int row,int left,int down,int right)
/************************/
{
 int bitmap=0;
 int bit=0;
 int sizeE=size-1;
 bitmap=(mask&~(left|down|right));
 if(row==sizeE){
    if(bitmap){
      aBoard[row]=(-bitmap&bitmap);
      /***07 symmetryOps CPU,GPU同一化*********************/
      int s=symmetryOps_bitmap(size,aBoard);
      if(s!=0){
        UNIQUE++;
        TOTAL+=s;
      }
      /************************/

    }
  }else{
    while(bitmap){
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
      /***07 aBoardローカル化*********************/
      //solve_nqueenr(size,mask,row+1,(left|bit)<<1, down|bit,(right|bit)>>1);
      solve_nqueenr(size,mask,row+1,(left|bit)<<1, down|bit,(right|bit)>>1,aBoard);
      /************************/
    }
  }
}
//CPUR 再帰版 ロジックメソッド
void NQueenR(int size,int mask)
{
  int bit=0;
  /***07 aBoardローカル化*********************/
  unsigned int aBoard[MAX];
  /************************/  
  //1行目全てにクイーンを置く
  for(int col=0;col<size;col++){
    aBoard[0]=bit=(1<<col);
    /***07 aBoardローカル化*********************/
    //solve_nqueenr(size,mask,1,bit<<1,bit,bit>>1);
    solve_nqueenr(size,mask,1,bit<<1,bit,bit>>1,aBoard);
    /************************/  
  }
}
//
//通常版 CPU 非再帰版 ロジックメソッド
void NQueenD(int size,int mask,int row)
{
  int aStack[size];
  int* pnStack;
  int bit;
  int bitmap;
  int sizeE=size-1;
  int down[size],right[size],left[size];
  /***07 aBoardローカル化*********************/
  unsigned int aBoard[MAX];
  /************************/  
  aStack[0]=-1; 
  pnStack=aStack+1;
  bit=0;
  bitmap=mask;
  down[0]=left[0]=right[0]=0;
  while(true){
    if(bitmap){
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap); 
      if(row==sizeE){
        /* 対称解除法の追加 */
        //TOTAL++;
        /***07 symmetryOps CPU,GPU同一化*********************/
        int s=symmetryOps_bitmap(size,aBoard);
        if(s!=0){
          UNIQUE++;
          TOTAL+=s;
        }
        /************************/
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
//
//通常版 CPUR 再帰版　ロジックメソッド
void NQueenDR(int size,int mask,int row,int left,int down,int right)
{
  int bit;
  int bitmap=mask&~(left|down|right);
  /***07 aBoardローカル化*********************/
  unsigned int aBoard[MAX];
  /************************/  
  if(row==size){
    /* 対称解除法の追加 */
    //TOTAL++;
    /***07 symmetryOps CPU,GPU同一化*********************/
    int s=symmetryOps_bitmap(size,aBoard);
    if(s!=0){
      UNIQUE++;
      TOTAL+=s;
    }
    /************************/
  }else{
    while(bitmap){
      //bitmap^=bit=(-bitmap&bitmap);
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
      NQueenDR(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
}
//メインメソッド
int main(int argc,char** argv)
{
  bool cpu=false,cpur=false,gpu=false,sgpu=false;
  int argstart=1,steps=24576;
  //int argstart=1,steps=1;
  
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
    printf("\n\n７．CPU 非再帰 バックトラック＋ビットマップ＋対称解除法\n");
  }else if(cpur){
    printf("\n\n７．CPUR 再帰 バックトラック＋ビットマップ＋対称解除法\n");
  }else if(gpu){
    printf("\n\n７．GPU 非再帰 バックトラック＋ビットマップ＋対称解除法\n");
  }else if(sgpu){
    printf("\n\n７．SGPU 非再帰 バックトラック＋ビットマップ\n");
  }
  if(cpu||cpur){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    clock_t st;           //速度計測用
    char t[20];           //hh:mm:ss.msを格納
    int min=4; int targetN=17;
    int mask;
    for(int i=min;i<=targetN;i++){
      /***07 symmetryOps CPU,GPU同一化*********************/
      TOTAL=0; UNIQUE=0;
      //COUNT2=COUNT4=COUNT8=0;
      /************************/
      mask=(1<<i)-1;
      st=clock();
      //
      //【通常版】
      //if(cpur){ _NQueenR(i,mask,0,0,0,0); }
      //CPUR
      if(cpur){ 
        NQueenR(i,mask); 
        //printf("通常版\n");
        //NQueenDR(i,mask,0,0,0,0); //通常版
      }
      //CPU
      if(cpu){ 
        NQueen(i,mask); 
        //printf("通常版\n");
        //NQueenD(i,mask,0); //通常版
      }
      //
      TimeFormat(clock()-st,t); 
      /***07 symmetryOps CPU,GPU同一化*********************/
      //printf("%2d:%13ld%16ld%s\n",i,getTotal(),getUnique(),t);
      printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
      /************************/

    }
  }
  if(gpu||sgpu){
    if(!InitCUDA()){return 0;}
    int min=4;int targetN=17;
    //int min=7;int targetN=7;
   
    struct timeval t0;struct timeval t1;
    int ss;int ms;int dd;
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    for(int i=min;i<=targetN;i++){
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
