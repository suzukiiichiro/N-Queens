/**
 CUDAで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイル
 $ nvcc CUDA02_N-Queen.cu -o CUDA02_N-Queen

 実行
 $ ./CUDA02_N-Queen (-c|-r|-g)
                    -c:cpu -r cpu再帰 -g GPU

 ２．配置フラグ（制約テスト高速化）

   パターンを生成し終わってからチェックを行うのではなく、途中で制約を満たさな
   い事が明らかな場合は、それ以降のパターン生成を行わない。
  「手を進められるだけ進めて、それ以上は無理（それ以上進めても解はない）という
  事がわかると一手だけ戻ってやり直す」という考え方で全ての手を調べる方法。
  (※)各行列に一個の王妃配置する組み合わせを再帰的に列挙分枝走査を行っても、組
  み合わせを列挙するだけであって、8王妃問題を解いているわけではありません。


  N-Queen の データ配列について
  =============================

  総当たり
  結局全部のケースをやってみる（完全解）

  バックトラック
  とりあえずやってみる。ダメなら戻って別の道を探る


  N-Queen: クイーンの効き筋
  =========================
  クイーンの位置から、縦、横、斜めが効き筋となります。

  　　       column(列)
  row(行)_0___1___2___3___4_
       0|-*-|---|---|-*-|---|
        +-------------------+
       1|---|-*-|---|-*-|---|
        +-------------------+ 
       2|---|---|-*-|-*-|-*-| 
        +-------------------+ 
       3|-*-|-*-|-*-|-Q-|-*-|
        +-------------------+
       4|---|---|-*-|-*-|-*-|
        +-------------------+


  N-Queen: 盤面上で互いのクイーンが効き筋にならないように配置
  ===========================================================

        完成図は以下の通りです。

  　　       column(列)
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---|
        +-------------------+
       1|---|---|---|-Q-|---|
        +-------------------+ 
       2|---|-Q-|---|---|---| 
        +-------------------+ 
       3|---|---|---|---|-Q-|
        +-------------------+
       4|---|---|-Q-|---|---|
        +-------------------+


  効き筋の表現
  ============

  クイーンの位置から下側を走査対象とします。

  　すでに効き筋：FALSE(盤面ではF）
  　配置可能    ：TRUE

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|---|---|---|---|---| 
        +-------------------+
       1|---|---|---|---|---|
        +-------------------+ 
       2|---|-Q-|---|---|---| 
        +-------------------+ 
       3|-F-|-F-|-F-|---|---|
        +-------------------+
       4|---|-F-|---|-F-|---|
        +-------------------+
                      

  効き筋を三つの配列で表現
  ========================

  ■ 基本：aBoard[row]=col
           aBoard[2  ]=1

  　　       column(列)
  row(行)_0___1___2___3___4_
       0|---|---|---|---|---|
        +-------------------+
       1|---|---|---|---|---|
        +-------------------+ 
       2|---|-Q-|---|---|---| aBoard[2]=1 に配置
        +-------------------+
       3|---|---|---|---|---|
        +-------------------+
       4|---|---|---|---|---|
        +-------------------+


  ■配列1：down[row]

  そのrow(行)にQueenがいる場合はFALSE
                      いない場合はTRUE

  　　       column(列)
  row(行)_0___1___2___3___4_
       0|---|---|---|---|---|
        +-------------------+
       1|---|---|---|---|---|
        +-------------------+ 
       2|---|-Q-|---|---|---| 
        +-------------------+
       3|---|-F-|---|---|---|
        +-------------------+
       4|---|-F-|---|---|---|
        +-------------------+
             down[col(1)]==false (すでに効き筋）


  ■配列２：right[col-row+N-1]
                    right[col-row+N-1]==F
                        Qの場所：col(1)-row(2)+(4-1)=2なので
                        col-row+N-1が２のところがＦとなる 
  　　       column(列)
  row(行)_0___1___2___3___4_
       0|---|---|---|---|---|
        +-------------------+
       1|---|---|---|---|---|
        +-------------------+ 
       2|---|-Q-|---|---|---| 
        +-------------------+ 
       3|---|---|-F-|---|---|
        +-------------------+
       4|---|---|---|-F-|---|
        +-------------------+
                      right[col-row+(N-1)]==false(すでに効き筋）


  ■配列3：left[col+row]
                      left[col+row]==F 
                          Qの場所：col(1)+row(2)=3なので
                          col+rowが3になるところがFとなる。

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|---|---|---|---|---|
        +-------------------+
       1|---|---|---|---|---|
        +-------------------+ 
       2|---|-Q-|---|---|---| 
        +-------------------+ 
       3|-F-|---|---|---|---|
        +-------------------+
       4|---|---|---|---|---|
        +-------------------+
      left[col+row]


  ステップ１
  ==========
  row=0, col=0 にクイーンを配置してみます。

  aBoard[row]=col
     ↓
  aBoard[0]=0;

  　　       column(列) 
  row(行)_0___1___2___3___4_
   ->  0|-Q-|---|---|---|---| aBoard[row]=col
        +-------------------+ aBoard[0  ]=0  
       1|---|---|---|---|---|
        +-------------------+ 
       2|---|---|---|---|---| 
        +-------------------+ 
       3|---|---|---|---|---|
        +-------------------+
       4|---|---|---|---|---|
        +-------------------+


  考え方：２
  ==========
  効き筋を埋めます

  　　       column(列) 
  row(行)_0___1___2___3___4_
   ->  0|-Q-|---|---|---|---| 
        +-------------------+ 
       1|-F-|-F-|---|---|---|
        +-------------------+ left はありません
       2|-F-|---|-F-|---|---| 
        +-------------------+ 
       3|-F-|---|---|-F-|---|
        +-------------------+
       4|-F-|---|---|---|-F-|
        +-------------------+
        down[col]      right[col-row+(N-1)]


  考え方：３
  ==========
  rowが一つ下に降りて０から１となります。
  次の候補は以下のＡ，Ｂ，Ｃとなります

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---| 
        +-------------------+ 
   ->  1|-F-|-F-|-A-|-B-|-C-|
        +-------------------+ 
       2|-F-|---|-F-|---|---| 
        +-------------------+ 
       3|-F-|---|---|-F-|---|
        +-------------------+
       4|-F-|---|---|---|-F-|
        +-------------------+

  考え方：４
  ==========
  Ａにおいてみます。
  効き筋は以下の通りです。

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---| 
        +-------------------+ 
   ->  1|-F-|-F-|-Q-|---|---|
        +-------------------+ 
       2|-F-|-F-|-F-|-F-|---| 
        +-------------------+ 
       3|-F-|---|-F-|-F-|-F-| right[col-row+(N-q)]
        +-------------------+
       4|-F-|---|-F-|---|-F-|
        +-------------------+
  left[col+row]  down[col]


  考え方：５
  ==========
  rowが一つ下に降りて１から２となります。
  次の候補はＡとなります

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---| 
        +-------------------+ 
       1|-F-|-F-|-Q-|---|---|
        +-------------------+ 
   ->  2|-F-|-F-|-F-|-F-|-A-| 
        +-------------------+ 
       3|-F-|---|-F-|-F-|-F-| 
        +-------------------+
       4|-F-|---|-F-|---|-F-|
        +-------------------+

  考え方：６
  ==========
  効き筋は以下の通りです。
  特に加わるところはありません。

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---| 
        +-------------------+ 
       1|-F-|-F-|-Q-|---|---|
        +-------------------+ 
   ->  2|-F-|-F-|-F-|-F-|-Q-| 
        +-------------------+ 
       3|-F-|---|-F-|-F-|-F-| 
        +-------------------+
       4|-F-|---|-F-|---|-F-|
        +-------------------+

  考え方：７
  ==========
  rowが一つ下に降りて２から３となります。
  次の候補はＡとなります

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---| 
        +-------------------+ 
       1|-F-|-F-|-Q-|---|---|
        +-------------------+ 
       2|-F-|-F-|-F-|-F-|-Q-| 
        +-------------------+ 
   ->  3|-F-|-A-|-F-|-F-|-F-| 
        +-------------------+
       4|-F-|---|-F-|---|-F-|
        +-------------------+


  考え方：８
  ==========
  効き筋は以下の通りです。

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---| 
        +-------------------+ 
       1|-F-|-F-|-Q-|---|---|
        +-------------------+ 
       2|-F-|-F-|-F-|-F-|-Q-| 
        +-------------------+ 
   ->  3|-F-|-Q-|-F-|-F-|-F-| 
        +-------------------+
       4|-F-|-F-|-F-|---|-F-|
        +-------------------+


  考え方：９
  ==========
  今回は、うまくいっていますが、
  次の候補がなければ、キャンセルして、
  前のコマを次の候補にコマを移動し、
  処理を継続します。


  考え方：１０
  =========-=

  rowが一つ下に降りて３から４となります。
  候補はのこり１箇所しかありません。

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---| 
        +-------------------+ 
       1|-F-|-F-|-Q-|---|---|
        +-------------------+ 
       2|-F-|-F-|-F-|-F-|-Q-| 
        +-------------------+ 
       3|-F-|-Q-|-F-|-F-|-F-| 
        +-------------------+
   ->  4|-F-|-F-|-F-|-A-|-F-|
        +-------------------+



  考え方：１１
  ==========
  最後のクイーンをおきます
  columnの最終列は効き筋を確認する必要はありませんね。

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---| 
        +-------------------+ 
       1|-F-|-F-|-Q-|---|---|
        +-------------------+ 
       2|-F-|-F-|-F-|-F-|-Q-| 
        +-------------------+ 
       3|-F-|-Q-|-F-|-F-|-F-| 
        +-------------------+
   ->  4|-F-|-F-|-F-|-Q-|-F-|
        +-------------------+

  考え方：１２
  ==========
  rowの脇にcolの位置を示します。

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---|  [0]
        +-------------------+ 
       1|-F-|-F-|-Q-|---|---|  [2]
        +-------------------+ 
       2|-F-|-F-|-F-|-F-|-Q-|  [4]
        +-------------------+ 
       3|-F-|-Q-|-F-|-F-|-F-|  [1]
        +-------------------+
   ->  4|-F-|-F-|-F-|-Q-|-F-|  [3]
        +-------------------+


  考え方：１３
  ==========

  ボード配列は以下のように表します。
  aBoard[]={0,2,4,1,3]

  出力：
    1: 0 0 0 1
    2: 0 0 0 2
    3: 0 0 0 4
    :
    :



 実行結果
$ nvcc CUDA02_N-Queen.cu  && ./a.out -r
 ２．CPUR 再帰 配置フラグ（制約テスト高速化）
 :
 :
111: 4 2 1 0 3 
112: 4 2 1 3 0 
113: 4 2 3 0 1 
114: 4 2 3 1 0 
115: 4 3 0 1 2 
116: 4 3 0 2 1 
117: 4 3 1 0 2 
118: 4 3 1 2 0 
119: 4 3 2 0 1 
120: 4 3 2 1 0 

$ nvcc CUDA02_N-Queen.cu  && ./a.out -c
 ２．CPU 非再帰 配置フラグ（制約テスト高速化）
 :
 :
111: 4 2 1 0 3 
112: 4 2 1 3 0 
113: 4 2 3 0 1 
114: 4 2 3 1 0 
115: 4 3 0 1 2 
116: 4 3 0 2 1 
117: 4 3 1 0 2 
118: 4 3 1 2 0 
119: 4 3 2 0 1 
120: 4 3 2 1 0 

$ nvcc CUDA02_N-Queen.cu  && ./a.out -g
２. GPU 非再帰 配置フラグ（制約テスト高速化）


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
long Total=0 ;      //GPU
long Unique=0;			//GPU
int COUNT=0;     		//カウント用
int aBoard[MAX]; 		//版の配列
int down[2*MAX-1]; 	//down:flagA 縦 配置フラグ　
//関数宣言
void print(int size);
void NQueen(int row,int size);
void NQueenR(int row,int size);
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
//出力用のメソッド
void print(int size){
	printf("%d: ",++COUNT);
	for(int j=0;j<size;j++){
		printf("%d ",aBoard[j]);
	}
	printf("\n");
}
//CPU 非再帰 ロジックメソッド
void NQueen(int row,int size){
  bool matched;
  while(row>=0){
    matched=false;
    for(int col=aBoard[row]+1;col<size;col++){
      if(down[col]==0){      //downは効き筋ではない
        if(aBoard[row]!=-1){ //Qは配置済み
          down[aBoard[row]]=0;//downの効き筋を外す
        }
        aBoard[row]=col;     //Qを配置
        down[col]=1;         //downは効き筋である
        matched=true;
        break;
      }
    }
    if(matched){
      row++;
      if(row==size){
        print(size);
        row--;
      }
    }else{                   //置けるところがない
      if(aBoard[row]!=-1){
        int col=aBoard[row]; /** colの代用 */
        down[col]=0;         //downの効き筋を解除
        aBoard[row]=-1;      //空き地に戻す
      }
      row--;
    }
  }
}
//CPUR 再帰 ロジックメソッド
void NQueenR(int row,int size){
  if(row==size){
    print(size);
  }else{
    for(int col=aBoard[row]+1;col<size;col++){
      aBoard[row]=col;  //Qを配置
      if(down[col]==0){
        down[col]=1;
        NQueenR(row+1,size);
        down[col]=0;
      }
      aBoard[row]=-1;   //空き地に戻す
    }
  }
}
//メインメソッド
int main(int argc,char** argv) {
	int size=5;
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
    printf("Default CPUR to 8 queen\n");
  }
  /** 出力と実行 */
	//aBoard配列を-1で初期化
  for(int i=0;i<size;i++){ aBoard[i]=-1; }
  /** CPU */
  if(cpu){
    printf("\n\n２．CPU 非再帰 配置フラグ（制約テスト高速化）\n");
    NQueen(0,size);
  }
  /** CPUR */
  if(cpur){
    printf("\n\n２．CPUR 再帰 配置フラグ（制約テスト高速化）\n");
    NQueenR(0,size);//ロジックメソッドを0を渡して呼び出し
  }
  /** GPU */
  if(gpu){
    if(!InitCUDA()){return 0;}
    int min=4;int targetN=18;
    struct timeval t0;struct timeval t1;int ss;int ms;int dd;
    printf("\n\n2. GPU 非再帰 配置フラグ（制約テスト高速化）\n");
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
