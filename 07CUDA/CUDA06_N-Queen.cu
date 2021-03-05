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


６．バックトラック＋ビットマップ

   ビット演算を使って高速化 状態をビットマップにパックし、処理する
   単純なバックトラックよりも２０〜３０倍高速
 
 　ビットマップであれば、シフトにより高速にデータを移動できる。
  フラグ配列ではデータの移動にO(N)の時間がかかるが、ビットマップであればO(1)
  フラグ配列のように、斜め方向に 2*N-1の要素を用意するのではなく、Nビットで充
  分。

 　配置可能なビット列を flags に入れ、-flags & flags で順にビットを取り出し処理。
 　バックトラックよりも２０−３０倍高速。
 
 ===================
 考え方 1
 ===================

 　Ｎ×ＮのチェスボードをＮ個のビットフィールドで表し、ひとつの横列の状態をひと
 つのビットフィールドに対応させます。(クイーンが置いてある位置のビットをONに
 する)
 　そしてバックトラッキングは0番目のビットフィールドから「下に向かって」順にい
 ずれかのビット位置をひとつだけONにして進めていきます。

 
 -----Q--    00000100 0番目のビットフィールド
 ---Q----    00010000 1番目のビットフィールド
 ------ Q-   00000010 2番目のビットフィールド
  Q-------   10000000 3番目のビットフィールド
 -------Q    00000001 4番目のビットフィールド
 -Q------    01000000 5番目のビットフィールド
 ---- Q---   00001000 6番目のビットフィールド
 -- Q-----   00100000 7番目のビットフィールド


 ===================
 考え方 2
 ===================

 次に、効き筋をチェックするためにさらに３つのビットフィールドを用意します。

 1. 左下に効き筋が進むもの: left 
 2. 真下に効き筋が進むもの: down
 3. 右下に効き筋が進むもの: right

次に、斜めの利き筋を考えます。
 上図の場合、
 1列目の右斜め上の利き筋は 3 番目(0x08)
 2列目の右斜め上の利き筋は 2 番目(0x04) になります。
 この値は 0 列目のクイーンの位置 0x10 を 1 ビットずつ「右シフト」すれば求める
 ことができます。
 また、左斜め上の利き筋の場合、1 列目では 5 番目(0x20) で 2 列目では 6 番目(0x40)
になるので、今度は 1 ビットずつ「左シフト」すれば求めることができます。

つまり、右シフトの利き筋を right、左シフトの利き筋を left で表すことで、クイー
ンの効き筋はrightとleftを1 ビットシフトするだけで求めることができるわけです。

  *-------------
 |. . . . . .
 |. . . -3. .  0x02 -|
 |. . -2. . .  0x04  |(1 bit 右シフト right)
 |. -1. . . .  0x08 -|
 |Q . . . . .  0x10 ←(Q の位置は 4   down)
 |. +1. . . .  0x20 -| 
 |. . +2. . .  0x40  |(1 bit 左シフト left)  
 |. . . +3. .  0x80 -|
  *-------------
  図：斜めの利き筋のチェック

 n番目のビットフィールドからn+1番目のビットフィールドに探索を進めるときに、そ
 の３つのビットフィールドとn番目のビットフィールド(bit)とのOR演算をそれぞれ行
 います。leftは左にひとつシフトし、downはそのまま、rightは右にひとつシフトして
 n+1番目のビットフィールド探索に渡してやります。

 left :(left |bit)<<1
 right:(right|bit)>>1
 down :   down|bit


 ===================
 考え方 3
 ===================

   n+1番目のビットフィールドの探索では、この３つのビットフィールドをOR演算した
 ビットフィールドを作り、それがONになっている位置は効き筋に当たるので置くことが
 できない位置ということになります。次にその３つのビットフィールドをORしたビッ
 トフィールドをビット反転させます。つまり「配置可能なビットがONになったビットフィー
 ルド」に変換します。そしてこの配置可能なビットフィールドを bitmap と呼ぶとして、
 次の演算を行なってみます。
 
 bit=-bitmap & bitmap;//一番右のビットを取り出す
 
   この演算式の意味を理解するには負の値がコンピュータにおける２進法ではどのよう
 に表現されているのかを知る必要があります。負の値を２進法で具体的に表わしてみる
 と次のようになります。
 
  00000011   3
  00000010   2
  00000001   1
  00000000   0
  11111111  -1
  11111110  -2
  11111101  -3
 
   正の値nを負の値-nにするときは、nをビット反転してから+1されています。そして、
 例えばn=22としてnと-nをAND演算すると下のようになります。nを２進法で表したときの
 一番下位のONビットがひとつだけ抽出される結果が得られるのです。極めて簡単な演算
 によって1ビット抽出を実現させていることが重要です。
 
      00010110   22
  AND 11101010  -22
 ------------------
      00000010
 
   さて、そこで下のようなwhile文を書けば、このループは bitmap のONビットの数の
 回数だけループすることになります。配置可能なパターンをひとつずつ全く無駄がなく
 生成されることになります。
 
 while(bitmap) {
     bit=-bitmap & bitmap;
     bitmap ^= bit;
     //ここでは配置可能なパターンがひとつずつ生成される(bit) 
 }

 実行結果
$ nvcc CUDA06_N-Queen.cu  && ./a.out -r
６．CPUR 再帰 バックトラック＋ビットマップ
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               0            0.00
 5:           10               0            0.00
 6:            4               0            0.00
 7:           40               0            0.00
 8:           92               0            0.00
 9:          352               0            0.00
10:          724               0            0.00
11:         2680               0            0.00
12:        14200               0            0.01
13:        73712               0            0.04
14:       365596               0            0.19
15:      2279184               0            1.24
16:     14772512               0            7.79
17:     95815104               0           57.57

$ nvcc CUDA06_N-Queen.cu  && ./a.out -c
６．CPU 非再帰 バックトラック＋ビットマップ
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               0            0.00
 5:           10               0            0.00
 6:            4               0            0.00
 7:           40               0            0.00
 8:           92               0            0.00
 9:          352               0            0.00
10:          724               0            0.00
11:         2680               0            0.00
12:        14200               0            0.01
13:        73712               0            0.04
14:       365596               0            0.21
15:      2279184               0            1.40
16:     14772512               0            8.78
17:     95815104               0         1:05.00

$ nvcc CUDA06_N-Queen.cu  && ./a.out -g
６．GPU 非再帰 バックトラック＋ビットマップ
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2               0  00:00:00:00.03
 5:           10               0  00:00:00:00.00
 6:            4               0  00:00:00:00.00
 7:           40               0  00:00:00:00.00
 8:           92               0  00:00:00:00.00
 9:          352               0  00:00:00:00.00
10:          724               0  00:00:00:00.00
11:         2680               0  00:00:00:00.02
12:        14200               0  00:00:00:00.03
13:        73712               0  00:00:00:00.05
14:       365596               0  00:00:00:00.09
15:      2279184               0  00:00:00:00.50
16:     14772512               0  00:00:00:02.41
17:     95815104               0  00:00:00:18.30

bash-3.2$ nvcc CUDA06_N-Queen.cu && ./a.out -s
６．SGPU 非再帰 バックトラック＋ビットマップ
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2               0  00:00:00:00.05
 5:           10               0  00:00:00:00.00
 6:            4               0  00:00:00:00.00
 7:           40               0  00:00:00:00.00
 8:           92               0  00:00:00:00.00
 9:          352               0  00:00:00:00.02
10:          724               0  00:00:00:00.00
11:         2680               0  00:00:00:00.01
12:        14200               0  00:00:00:00.02
13:        73712               0  00:00:00:00.05
14:       365596               0  00:00:00:00.09
15:      2279184               0  00:00:00:00.49
16:     14772512               0  00:00:00:02.43
17:     95815104               0  00:00:00:18.44
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
long TOTAL=0;         //CPU,CPUR
long UNIQUE=0;        //CPU,CPUR
//関数宣言 SGPU
__global__ 
void sgpu_cuda_kernel(int size,int mark,
		unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,
		unsigned int* results,int totalCond);
long long sgpu_solve_nqueen_cuda(int size,int steps); 
//関数宣言 GPU
__global__
void cuda_kernel(
    int size,int mark,
    unsigned int* t_down,unsigned int* t_left,unsigned int* t_right,
    unsigned int* d_results,int totalCond);
long long solve_nqueen_cuda(int size,int steps);
//関数宣言 CPU
void TimeFormat(clock_t utime,char *form);
//関数宣言 CPUR
void solve_nqueen(int size,int mask, int row,int* left,int* down,int* right,int* bitmap);
void NQueen(int size,int mask);
void solve_nqueenr(int size,int mask, int row,int left,int down,int right);
void NQueenR(int size,int mask,int row,int left,int down,int right);
//
//SGPU
__global__ 
void sgpu_cuda_kernel(
    int size,int mark,
    unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,
    unsigned int* results,int totalCond){
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
//SGPU
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
// GPU
__global__ 
void cuda_kernel(
    int size,int mark,
    unsigned int* t_down,unsigned int* t_left,unsigned int* t_right,
    unsigned int* d_results,int totalCond){
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
  if(idx<totalCond){//余分なスレッドは動かさない
    down[tid][row]=t_down[idx];
    left[tid][row]=t_left[idx];
    right[tid][row]=t_right[idx];
    bitmap[tid][row]=mask&~(down[tid][row]|left[tid][row]|right[tid][row]);
    while(row>=0){
      if(bitmap[tid][row]==0){
        --row;
      }else{
        bitmap[tid][row]^=bit=(-bitmap[tid][row]&bitmap[tid][row]); 
        if((bit&mask)!=0){//置く場所があるかどうか
          if(row+1==mark){
            total++;
            --row;
          }else{
            int n=row++;
            down[tid][row]=down[tid][n]|bit;
            left[tid][row]=(left[tid][n]|bit)<<1;
            right[tid][row]=(right[tid][n]|bit)>>1;
            bitmap[tid][row]=mask&~(down[tid][row]|left[tid][row]|right[tid][row]);
          }
        }else{
            --row;
        }//置く場所がなければ１個上に
      }
    }
    sum[tid]=total;//最後sum[tid]に加算する
  }else{//totalCond未満は回らないのでtotal加算しない
      sum[tid]=0;
      } 
  __syncthreads();if(tid<64&&tid+64<THREAD_NUM){sum[tid]+=sum[tid+64];} 
  __syncthreads();if(tid<32){sum[tid]+=sum[tid+32];} 
  __syncthreads();if(tid<16){sum[tid]+=sum[tid+16];} 
  __syncthreads();if(tid<8){sum[tid]+=sum[tid+8];} 
  __syncthreads();if(tid<4){sum[tid]+=sum[tid+4];} 
  __syncthreads();if(tid<2){sum[tid]+=sum[tid+2];} 
  __syncthreads();if(tid<1){sum[tid]+=sum[tid+1];} 
  __syncthreads();if(tid==0){d_results[bid]=sum[0];}
}
//
// GPU
long long solve_nqueen_cuda(int size,int steps) {
  register int bitmap[32];
  register int bit;
  register int h_down[size],h_right[size],h_left[size];
  if(size<=0||size>32){return 0;}
  unsigned int* t_down=new unsigned int[steps];
  unsigned int* t_left=new unsigned int[steps];
  unsigned int* t_right=new unsigned int[steps];
  unsigned int* h_results=new unsigned int[steps];
  unsigned int* d_down;
  unsigned int* d_left;
  unsigned int* d_right;
  unsigned int* d_results;
  cudaMalloc((void**) &d_down,sizeof(int)*steps);
  cudaMalloc((void**) &d_left,sizeof(int)*steps);
  cudaMalloc((void**) &d_right,sizeof(int)*steps);
  cudaMalloc((void**) &d_results,sizeof(int)*steps/THREAD_NUM);
  const unsigned int mask=(1<<size)-1;
  const unsigned int mark=size>11?size-10:2;//何行目からGPUで行くか。ここの設定は変更可能、設定値を多くするほどGPUで並行して動く
  long long total=0;
  int totalCond=0;
  int row=0;
  bit=0;
  h_down[0]=h_left[0]=h_right[0]=0;
  bool matched=false;
  for(int col=0;col<size/2;col++){//右側半分だけやる
    bit=(1<<col);
    bitmap[0]=mask;
    h_down[1]=bit;
    h_left[1]=bit<<1;
    h_right[1]=bit>>1;
    bitmap[1]=mask&~(h_down[1]|h_left[1]|h_right[1]);
    row=1;
    while(row>0){
      if(bitmap[row]==0){
          row--;
      }else{//おける場所があれば進む
        bitmap[row]^=bit=(-bitmap[row]&bitmap[row]); 
        if((bit&mask)!=0){//置く場所があれば先に進む
          int n=row++;
          h_down[row]=h_down[n]|bit;
          h_left[row]=(h_left[n]|bit)<<1;
          h_right[row]=(h_right[n]|bit)>>1;
          bitmap[row]=mask&~(h_down[row]|h_left[row]|h_right[row]);
          if(row==mark){
            //2行目(mark)にクイーンを１個ずつ置いていって、down,left,right情報を格納、
            //その次の行へは進まない。その行で可能な場所にクイーン置き終わったらGPU並列実行
            t_down[totalCond]=h_down[row];
            t_left[totalCond]=h_left[row];
            t_right[totalCond]=h_right[row];
            totalCond++;
            //最大GPU数に達してしまったらここでGPUを実行する。stepsはGPUの同時並行稼働数を制御
            if(totalCond==steps){//ここではtotalCond==stepsの場合だけこの中へ
              if(matched){//matched=trueの時にCOUNT追加 //GPU内でカウントしているので、GPUから出たらmatched=trueになってる
                cudaMemcpy(h_results,d_results,
                    sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
                for(int col=0;col<steps/THREAD_NUM;col++){total+=h_results[col];}
                matched=false;
              }
              cudaMemcpy(d_down,t_down,
                  sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              cudaMemcpy(d_left,t_left,
                  sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              cudaMemcpy(d_right,t_right,
                  sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              /** backTrack+bitmap*/
              cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
                >>>(size,size-mark,d_down,d_left,d_right,d_results,totalCond);
              matched=true;//GPU内でカウントしているので、GPUから出たらmatched=trueになってる
              totalCond=0;
              
            }
            //GPUに渡す行数についてはクイーンを１個ずつ置く
            //GPU呼び出してなくても上に上がる
            --row;
          }
        }else{
          --row;
        }//置く場所がなければ上に上がる
      }
    }
  }
  if(matched){//matche=trueの時COUNT追加
    cudaMemcpy(h_results,d_results,
        sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    for(int col=0;col<steps/THREAD_NUM;col++){total+=h_results[col];}
    matched=false;
  }
  cudaMemcpy(d_down,t_down,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(d_left,t_left,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(d_right,t_right,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  /** backTrack+bitmap*/
  cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
    >>>(size,size-mark,d_down,d_left,d_right,d_results,totalCond);
  cudaMemcpy(h_results,d_results,
      sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  for(int col=0;col<steps/THREAD_NUM;col++){total+=h_results[col];}	
  total*=2;//ミラーなので２倍する
  //
  if(size%2==1){//奇数の場合はミラーがないので1倍で同じ処理をする
    matched=false;
    totalCond=0;
    bit=(1<<(size-1)/2);
    bitmap[0]=mask;
    h_down[1]=bit;
    h_left[1]=bit<<1;
    h_right[1]=bit>>1;
    bitmap[1]=mask&~(h_down[1]|h_left[1]|h_right[1]);
    row=1;
    while(row>0){
      if(bitmap[row]==0){
        row--;
      }else{
        bitmap[row]^=bit=(-bitmap[row]&bitmap[row]);
        if((bit&mask)!=0){
          int n=row++;
          h_down[row]=h_down[n]|bit;
          h_left[row]=(h_left[n]|bit)<<1;
          h_right[row]=(h_right[n]|bit)>>1;
          bitmap[row]=mask&~(h_down[row]|h_left[row]|h_right[row]);
          if(row==mark){
            t_down[totalCond]=h_down[row];
            t_left[totalCond]=h_left[row];
            t_right[totalCond]=h_right[row];
            totalCond++;
            if(totalCond==steps){
              if(matched){
                cudaMemcpy(h_results,d_results,
                    sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
                for(int col=0;col<steps/THREAD_NUM;col++){total+=h_results[col];}
                matched=false;
              }
              cudaMemcpy(d_down,t_down,
                  sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              cudaMemcpy(d_left,t_left,
                  sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              cudaMemcpy(d_right,t_right,
                  sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              /** backTrack+bitmap*/
              cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
                >>>(size,size-mark,d_down,d_left,d_right,d_results,totalCond);
              matched=true;
              totalCond=0;
            }
            --row;
          }
        }else{--row;}
      }
    }
    if(matched){
      cudaMemcpy(h_results,d_results,
          sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
      for(int col=0;col<steps/THREAD_NUM;col++){total+=h_results[col];}
      matched=false;
    }
    cudaMemcpy(d_down,t_down,
        sizeof(int)*totalCond,cudaMemcpyHostToDevice);
    cudaMemcpy(d_left,t_left,
        sizeof(int)*totalCond,cudaMemcpyHostToDevice);
    cudaMemcpy(d_right,t_right,
        sizeof(int)*totalCond,cudaMemcpyHostToDevice);
    /** backTrack+bitmap*/
    cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
      >>>(size,size-mark,d_down,d_left,d_right,d_results,totalCond);
    cudaMemcpy(h_results,d_results,
        sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    for(int col=0;col<steps/THREAD_NUM;col++){total+=h_results[col];}
  }
  cudaFree(d_down);
  cudaFree(d_left);
  cudaFree(d_right);
  cudaFree(d_results);
  cudaFreeHost(t_down);
  cudaFreeHost(t_left);
  cudaFreeHost(t_right);
  cudaFreeHost(h_down);
  cudaFreeHost(h_left);
  cudaFreeHost(h_right);
  cudaFreeHost(h_results);
  return total;
}
//
//CUDA 初期化
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
        bitmap[row]^=bit=(-bitmap[row]&bitmap[row]); 
        if((bit&mask)!=0){
          if(row==sizeE){
            TOTAL++;
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
void NQueen(int size,int mask){
  register int sizeE=size-1;
  register int bitmap[size];
  register int down[size],right[size],left[size];
  register int bit;
  if(size<=0||size>32){return;}
  bit=0;
  bitmap[0]=mask;
  down[0]=left[0]=right[0]=0;
  //偶数、奇数共通
  for(int col=0;col<size/2;col++){//右側半分だけクイーンを置く
    bit=(1<<col);//
    down[1]=bit;//再帰の場合は down,left,right,bitmapは現在の行だけで良いが
    left[1]=bit<<1;//非再帰の場合は全行情報を配列に入れて行の上がり下がりをする
    right[1]=bit>>1;
    bitmap[1]=mask&~(left[1]|down[1]|right[1]);
    solve_nqueen(size,mask,1,left,down,right,bitmap);
  }
  TOTAL*=2;//ミラーなのでTOTALを２倍する
  //奇数の場合はさらに中央にクイーンを置く
  if(size%2==1){
    bit=(1<<(sizeE)/2);
    down[1]=bit;
    left[1]=bit<<1;
    right[1]=bit>>1;
    bitmap[1]=mask&~(left[1]|down[1]|right[1]);
    solve_nqueen(size,mask,1,left,down,right,bitmap);
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
      TOTAL++;
    }
  }else{
    while(bitmap){
      bitmap^=bit=(-bitmap&bitmap);
      solve_nqueenr(size,mask,row+1,(left|bit)<<1, down|bit,(right|bit)>>1);
    }
  }
}
//
//CPUR 再帰版 ロジックメソッド
void NQueenR(int size,int mask, int row,int left,int down,int right){
  int bit=0;
  int sizeE=size-1;
  for(int col=0;col<size/2;col++){
    bit=(1<<col);
    solve_nqueenr(size,mask,1,bit<<1,bit,bit>>1);
  }
  TOTAL*=2;
  if(size%2==1){
    bit=(1<<(sizeE)/2);
    solve_nqueenr(size,mask,1,bit<<1,bit,bit>>1);
  }
}
//
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
    printf("\n\n６．CPU 非再帰 バックトラック＋ビットマップ\n");
  }else if(cpur){
    printf("\n\n６．CPUR 再帰 バックトラック＋ビットマップ\n");
  }else if(gpu){
    printf("\n\n６．GPU 非再帰 バックトラック＋ビットマップ\n");
  }else if(sgpu){
    printf("\n\n６．SGPU 非再帰 バックトラック＋ビットマップ\n");
  }
  if(cpu||cpur){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    clock_t st;          //速度計測用
    char t[20];          //hh:mm:ss.msを格納
    int min=4;
    int targetN=17;
    int mask;
    for(int i=min;i<=targetN;i++){
      TOTAL=0;
      UNIQUE=0;
      mask=((1<<i)-1);
      st=clock();
      if(cpu){ NQueen(i,mask); }
      if(cpur){ NQueenR(i,mask,0,0,0,0); }
      TimeFormat(clock()-st,t);
      printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
    }
  }
  if(gpu||sgpu){
    if(!InitCUDA()){return 0;}
    int min=4;int targetN=17;
    struct timeval t0;struct timeval t1;
    int ss;int ms;int dd;
    long TOTAL,UNIQUE;
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    for(int i=min;i<=targetN;i++){
      gettimeofday(&t0,NULL);  // 計測開始
      if(gpu){
      	TOTAL=solve_nqueen_cuda(i,steps);
      	UNIQUE=0;
      }else if(sgpu){
        TOTAL=sgpu_solve_nqueen_cuda(i,steps);
      	UNIQUE=0;
      }
      gettimeofday(&t1,NULL);  // 計測終了
      if(t1.tv_usec<t0.tv_usec) {
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
