
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

９．クイーンの位置による分岐BOUND1

  前章のコードは全ての解を求めた後に、ユニーク解以外の対称解を除去していた
  ある意味、「生成検査法（generate ＆ test）」と同じである
  問題の性質を分析し、バックトラッキング/前方検査法と同じように、無駄な探索を省略することを考える
  ユニーク解に対する左右対称解を予め削除するには、1行目のループのところで、
  右半分だけにクイーンを配置するようにすればよい
  Nが奇数の場合、クイーンを1行目中央に配置する解は無い。
  他の3辺のクィーンが中央に無い場合、その辺が上辺に来るよう回転し、場合により左右反転することで、
  最小値解とすることが可能だから、中央に配置したものしかユニーク解には成り得ない
  しかし、上辺とその他の辺の中央にクィーンは互いの効きになるので、配置することが出来ない


  1. １行目角にクイーンがある場合、とそうでない場合で処理を分ける
    １行目かどうかの条件判断はループ外に出してもよい
    処理時間的に有意な差はないので、分かりやすいコードを示した
  2.１行目角にクイーンがある場合、回転対称形チェックを省略することが出来る
    １行目角にクイーンがある場合、他の角にクイーンを配置することは不可
    鏡像についても、主対角線鏡像のみを判定すればよい
    ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい

  １行目角にクイーンが無い場合、クイーン位置より右位置の８対称位置にクイーンを置くことはできない
  置いた場合、回転・鏡像変換により得られる状態のユニーク判定値が明らかに大きくなる
    ☓☓・・・Ｑ☓☓
    ☓・・・／｜＼☓
    ｃ・・／・｜・rt
    ・・／・・｜・・
    ・／・・・｜・・
    lt・・・・｜・ａ
    ☓・・・・｜・☓
    ☓☓ｂ・・dn☓☓
    
  １行目位置が確定した時点で、配置可能位置を計算しておく（☓の位置）
  lt, dn, lt 位置は効きチェックで配置不可能となる
  回転対称チェックが必要となるのは、クイーンがａ, ｂ, ｃにある場合だけなので、
  90度、180度、270度回転した状態のユニーク判定値との比較を行うだけで済む
 *
 *

 実行結果

$ nvcc CUDA09_N-Queen.cu  && ./a.out -r
９．CPUR 再帰 クイーンの位置による分岐BOUND1
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
13:        73712            9233            0.05
14:       365596           45752            0.33
15:      2279184          285053            1.98
16:     14772512         1846955           15.74
17:     95815104        11977939         1:41.91

$ nvcc CUDA09_N-Queen.cu  && ./a.out -c
９．CPU 非再帰 クイーンの位置による分岐BOUND1
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
13:        73712            9233            0.05
14:       365596           45752            0.29
15:      2279184          285053            1.88
16:     14772512         1846955           14.23
17:     95815104        11977939         1:36.77

$ nvcc CUDA09_N-Queen.cu  && ./a.out -g
９．GPU 非再帰 クイーンの位置による分岐BOUND1
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2               1  00:00:00:00.02
 5:           10               2  00:00:00:00.00
 6:            4               1  00:00:00:00.00
 7:           40               6  00:00:00:00.00
 8:           92              12  00:00:00:00.01
 9:          352              46  00:00:00:00.01
10:          724              92  00:00:00:00.03
11:         2680             341  00:00:00:00.06
12:        14200            1787  00:00:00:00.23
13:        73712            9233  00:00:00:01.02
14:       365596           45752  00:00:00:01.34
15:      2279184          285053  00:00:00:05.00
16:     14772512         1846955  00:00:00:32.69
17:     95815104        11977939  00:00:04:09.32

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
long TOTAL=0; //GPU,CPUで使用
long UNIQUE=0;//GPU,CPUで使用
//
__device__ __host__
int* vMirror(int* bf,int* af,int si)
{
  int bf_i;
  int tmp;
  for(int i=0;i<si;i++) {
    bf_i=bf[i];
    tmp=0;
    for(int j=0;j<=si-1;j++){
      if(bf_i&(1<<j)){ 
        tmp|=(1<<(si-1-j)); 
        break;                 
      }
    }
    af[i]=tmp;
  }
  return af;
}
//
__device__ __host__
int* rotate(int* bf,int* af,int si)
{
  int t;
  for(int i=0;i<si;i++){
    t=0;
    for(int j=0;j<si;j++){
      t|=((bf[j]>>i)&1)<<(si-j-1);
    }
    af[i]=t;
  }
  return af;
}
//
__device__ __host__
int intncmp(unsigned int* lt,int* rt,int n)
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
//
__device__ __host__
int symmetryOps(int si,unsigned int *aBoard)
{
  int nEquiv=0;
  int aT[MAX];
  int aS[MAX];
  // 回転・反転・対称チェックのためにboard配列をコピー
  memcpy(aT,aBoard,sizeof(int)*si);
  //時計回りに90度回転
  rotate(aT,aS,si);
  int icmp=intncmp(aBoard,aS,si);
  if(icmp>0){ return 0; }
  else if(icmp==0){ nEquiv=2; }
  else{//時計回りに180度回転
    rotate(aS,aT,si);
    icmp=intncmp(aBoard,aT,si);
    if(icmp>0){ return 0;}
    else if(icmp==0){ nEquiv=4;}
    else{//時計回りに270度回転
      rotate(aT,aS,si);
      icmp=intncmp(aBoard,aS,si);
      if(icmp>0){ return 0;}
      nEquiv=8;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  //for(int i=0;i<si;i++){ aS[i]=aBoard[i];}
  memcpy(aS,aBoard,sizeof(int)*si);
  //垂直反転
  vMirror(aS,aT,si);   
  icmp=intncmp(aBoard,aT,si);
  if(icmp>0){ return 0; }
  //-90度回転 対角鏡と同等
  if(nEquiv>2){
    rotate(aT,aS,si);
    icmp=intncmp(aBoard,aS,si);
    if(icmp>0){return 0;}
    //-180度回転 水平鏡像と同等
    else if(nEquiv>4){
      rotate(aS,aT,si);
      icmp=intncmp(aBoard,aT,si);
      //-270度回転 反対角鏡と同等
      if(icmp>0){ return 0;}
      rotate(aT,aS,si);
      icmp=intncmp(aBoard,aS,si);
      if(icmp>0){ return 0;}
    }
  }
  return nEquiv;  
}
//
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
{
  register const unsigned int mask=(1<<size)-1;
  register unsigned int total=0;
  register unsigned int unique=0;
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
  unsigned int c_aBoard[MAX];
  __shared__ unsigned int usum[THREAD_NUM];
  //余分なスレッドは動かさない 
  //GPUはsteps数起動するがtotalCond以上は空回しする
  if(idx<totalCond){
    //totalDown,totalLeft,totalRightの情報を
    //down,left,rightに詰め直す 
    //CPU で詰め込んだ t_はsteps個あるが
    //ブロック内ではブロックあたりのスレッド数に限定
    //されるので idxでよい
    //
    for(int i=0;i<h_row;i++){
      c_aBoard[i]=t_aBoard[idx*h_row+i]; //２次元配列だが1次元的に利用  
    }
    register unsigned int bitmap_tid_row;
    register unsigned int down_tid_row;
    register unsigned int left_tid_row;
    register unsigned int right_tid_row;
    while(row>=0){
      bitmap_tid_row=bitmap[tid][row];
      down_tid_row=down[tid][row];
      left_tid_row=left[tid][row];
      right_tid_row=right[tid][row];
      if(bitmap_tid_row==0){
        row--;
      }else{
        //クイーンを置く
        //置く場所があるかどうか
        bitmap[tid][row]
          ^=c_aBoard[row+h_row]
          =bit
          =(-bitmap_tid_row&bitmap_tid_row);       
        if((bit&mask)!=0){
          //最終行?最終行から１個前の行まで
          //無事到達したら 加算する
          if(row+1==mark){
           int s=symmetryOps(size,c_aBoard); 
           if(s!=0){
           //print(size); //print()でTOTALを++しない
           //ホストに戻す配列にTOTALを入れる
           //スレッドが１つの場合は配列は１個
              unique++; 
              total+=s;   //対称解除で得られた解数を加算
           }
            row--;
          }else{
            int rowP=row+1;
            down[tid][rowP]=down_tid_row|bit;
            left[tid][rowP]=(left_tid_row|bit)<<1;
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
    usum[tid]=unique;
  }else{
    //totalCond未満は空回しするのでtotalは加算しない
    sum[tid]=0;
    usum[tid]=0;
  } 
  //__syncthreads()でブロック内のスレッド間の同期
  //全てのスレッドが__syncthreads()に辿り着くのを待つ
  __syncthreads();if(tid<64&&tid+64<THREAD_NUM){
    sum[tid]+=sum[tid+64];
    usum[tid]+=usum[tid+64];
  }
  __syncwarp();if(tid<32){
    sum[tid]+=sum[tid+32];
    usum[tid]+=usum[tid+32];
  } 
  __syncwarp();if(tid<16){
    sum[tid]+=sum[tid+16];
    usum[tid]+=usum[tid+16];
  } 
  __syncwarp();if(tid<8){
    sum[tid]+=sum[tid+8];
    usum[tid]+=usum[tid+8];
  } 
  __syncwarp();if(tid<4){
    sum[tid]+=sum[tid+4];
    usum[tid]+=usum[tid+4];
  } 
  __syncwarp();if(tid<2){
    sum[tid]+=sum[tid+2];
    usum[tid]+=usum[tid+2];
  } 
  __syncwarp();if(tid<1){
    sum[tid]+=sum[tid+1];
    usum[tid]+=usum[tid+1];
  } 
  __syncwarp();if(tid==0){
    d_results[bid]=sum[0];
    d_uniq[bid]=usum[0];
  }
}
//
long solve_nqueen_cuda(int size,int mask,int row,int n_left,int n_down,int n_right,int steps,unsigned int* aBoard)
{
  //何行目からGPUで行くか。ここの設定は変更可能、設定値を多くするほどGPUで並行して動く
  /***08 クイーンを２行目まで固定で置くためmarkが3以上必要*********************/
  const unsigned int mark=size>12?size-10:3;
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
  bitmap[row]=mask&~(left[row]|down[row]|right[row]);
  unsigned int bit;
  unsigned int* totalDown;
  cudaMallocHost((void**) &totalDown,sizeof(int)*steps);
  unsigned int* totalLeft;
  cudaMallocHost((void**) &totalLeft,sizeof(int)*steps);
  unsigned int* totalRight;
  cudaMallocHost((void**) &totalRight,sizeof(int)*steps);
  unsigned int* h_results;
  cudaMallocHost((void**) &h_results,sizeof(int)*steps);
  unsigned int* h_uniq;
  cudaMallocHost((void**) &h_uniq,sizeof(int)*steps);
  unsigned int* t_aBoard;
  cudaMallocHost((void**) &t_aBoard,sizeof(int)*steps*mark);
  //device
  unsigned int* downCuda;
  cudaMalloc((void**) &downCuda,sizeof(int)*steps);
  unsigned int* leftCuda;
  cudaMalloc((void**) &leftCuda,sizeof(int)*steps);
  unsigned int* rightCuda;
  cudaMalloc((void**) &rightCuda,sizeof(int)*steps);
  unsigned int* resultsCuda;
  cudaMalloc((void**) &resultsCuda,sizeof(int)*steps/THREAD_NUM);
  unsigned int* d_uniq;
  cudaMalloc((void**) &d_uniq,sizeof(int)*steps/THREAD_NUM);
  unsigned int* d_aBoard;
  cudaMalloc((void**) &d_aBoard,sizeof(int)*steps*mark);
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
    //06GPU こっちのほうが優秀
    if(bitmap[row]==0){ row--; }
    else{//おける場所があれば進む
      //06SGPU
      bitmap[row]^=aBoard[row]=bit=(-bitmap[row]&bitmap[row]);
      if((bit&mask)!=0){//置く場所があれば先に進む
        rowP=row+1;
        down[rowP]=down[row]|bit;
        left[rowP]=(left[row]|bit)<<1;
        right[rowP]=(right[row]|bit)>>1;
        bitmap[rowP]=mask&~(down[rowP]|left[rowP]|right[rowP]);
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
          for(int i=0;i<mark;i++){
            t_aBoard[totalCond*mark+i]=aBoard[i];
          }
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
              cudaMemcpy(h_uniq,d_uniq,
                  sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
              for(int col=0;col<steps/THREAD_NUM;col++){
                total+=h_results[col];
                UNIQUE+=h_uniq[col];
              }
              matched=false;
            }
            cudaMemcpy(downCuda,totalDown,
                sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            cudaMemcpy(leftCuda,totalLeft,
                sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            cudaMemcpy(rightCuda,totalRight,
                sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            cudaMemcpy(d_aBoard,t_aBoard,
                sizeof(int)*totalCond*mark,cudaMemcpyHostToDevice);
            cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
              >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,row);
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
    cudaMemcpy(h_uniq,d_uniq,
        sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    for(int col=0;col<steps/THREAD_NUM;col++){
      total+=h_results[col];
      UNIQUE+=h_uniq[col];
    }
    matched=false;
  }
  cudaMemcpy(downCuda,totalDown,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(leftCuda,totalLeft,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(rightCuda,totalRight,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(d_aBoard,t_aBoard,
      sizeof(int)*totalCond*mark,cudaMemcpyHostToDevice);
  //size-mark は何行GPUを実行するか totalCondはスレッド数
  //steps数の数だけマルチスレッドで起動するのだが、実際に計算が行われるのは
  //totalCondの数だけでそれ以外は空回しになる
  cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
    >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,mark);
  cudaMemcpy(h_results,resultsCuda,
      sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_uniq,d_uniq,
      sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  for(int col=0;col<steps/THREAD_NUM;col++){
    total+=h_results[col];
    UNIQUE+=h_uniq[col];
  }
  //
  cudaFree(downCuda);
  cudaFree(leftCuda);
  cudaFree(rightCuda);
  cudaFree(resultsCuda);
  cudaFree(d_uniq);
  cudaFree(d_aBoard);
  cudaFreeHost(totalDown);
  cudaFreeHost(totalLeft);
  cudaFreeHost(totalRight);
  cudaFreeHost(h_results);
  cudaFreeHost(h_uniq);
  cudaFreeHost(t_aBoard);
  return total;
}
//GPU
void NQueenG(int size,int steps)
{
  unsigned int aBoard[MAX];
  /***09 sizeEここでは使用しないのでコメント*********************/
  //register int sizeE=size-1;
  /************************/
  register int bit=0;
  register int mask=((1<<size)-1);
  if(size<=0||size>32){return;}
  /***09 backtrack1,2を意識し１行目右端とそれ以外で処理を分ける*********************/
  //09では枝借りはまだしないのでTOPBIT,SIDEMASK,LASTMASK,ENDBITは使用しない
  //backtrack1
  //1行め右端 0
  int col=0;
  aBoard[0]=bit=(1<<col);
  int left=bit<<1;
  int down=bit;
  int right=bit>>1;
  //2行目は右から3列目から左端から2列目まで
  for(int col_j=2;col_j<size-1;col_j++){
      aBoard[1]=bit=(1<<col_j);
      TOTAL+=solve_nqueen_cuda(size,mask,2,(left|bit)<<1,(down|bit),(right|bit)>>1,steps,aBoard);
  }
  /***09 backtrack2*********************/
  //1行目右から2列目から
  //偶数個は1/2 n=8 なら 1,2,3 奇数個は1/2+1 n=9 なら 1,2,3,4
  for(int col=1,col2=size-2;col<col2;col++,col2--){
      aBoard[0]=bit=(1<<col);
      TOTAL+=solve_nqueen_cuda(size,mask,1,bit<<1,bit,bit>>1,steps,aBoard);
  }
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
void solve_nqueen(int size,int mask, int row,int h_left,int h_down,int h_right,unsigned int* aBoard)
{
    unsigned int left[size];
    unsigned int down[size];
	  unsigned int right[size];
    unsigned int bitmap[size];
	  left[row]=h_left;
	  down[row]=h_down;
	  right[row]=h_right;
	  bitmap[row]=mask&~(left[row]|down[row]|right[row]);
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
            int s=symmetryOps(size,aBoard);
            if(s!=0){
              UNIQUE++;
              TOTAL+=s;
            }
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
  register int bit;
  unsigned int aBoard[MAX];
  bit=0;
  if(size<=0||size>32){return;}
  /***09 枝借りはまだしないのでTOPBIT,SIDEMASK,LASTMASK,ENDBITは使用しない***/
  //backtrack1
  //1行め右端 0
  int col=0;
  aBoard[0]=bit=(1<<col);
  int left=bit<<1;
  int down=bit;
  int right=bit>>1;
  /***09 2行目は右から3列目から左端から2列目まで***/
  for(int col_j=2;col_j<size-1;col_j++){
      aBoard[1]=bit=(1<<col_j);
      solve_nqueen(size,mask,2,(left|bit)<<1,(down|bit),(right|bit)>>1,aBoard);
  }
  /***09 backtrack2***/
  //1行目右から2列目から
  //偶数個は1/2 n=8 なら 1,2,3 奇数個は1/2+1 n=9 なら 1,2,3,4
  for(int col=1,col2=size-2;col<col2;col++,col2--){
      aBoard[0]=bit=(1<<col);
      solve_nqueen(size,mask,1,bit<<1,bit,bit>>1,aBoard);
  }
}
//CPUR 再帰版 ロジックメソッド
void solve_nqueenr(int size,int mask, int row,int left,int down,int right,unsigned int* aBoard)
{
 int bitmap=0;
 int bit=0;
 int sizeE=size-1;
 bitmap=(mask&~(left|down|right));
 if(row==sizeE){
    if(bitmap){
      aBoard[row]=(-bitmap&bitmap);
      int s=symmetryOps(size,aBoard);
      if(s!=0){
        UNIQUE++;
        TOTAL+=s;
      }
    }
  }else{
    while(bitmap){
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
      solve_nqueenr(size,mask,row+1,(left|bit)<<1, down|bit,(right|bit)>>1,aBoard);
    }
  }
}
//CPUR 再帰版 ロジックメソッド
void NQueenR(int size,int mask)
{
  int bit=0;
  unsigned int aBoard[MAX];
  /***09では枝借りはまだしないのでTOPBIT,SIDEMASK,LASTMASK,ENDBITは使用しない*********************/
  /***09　backtrack1*********************/
  //1行め右端 0
  int col=0;
  aBoard[0]=bit=(1<<col);
  int left=bit<<1;
  int down=bit;
  int right=bit>>1;
  //2行目は右から3列目から左端から2列目まで
  for(int col_j=2;col_j<size-1;col_j++){
    aBoard[1]=bit=(1<<col_j);
    solve_nqueenr(size,mask,2,(left|bit)<<1,(down|bit),(right|bit)>>1,aBoard);
  }
  /***09 backtrack2*********************/
  //1行目右から2列目から
  //偶数個は1/2 n=8 なら 1,2,3 奇数個は1/2+1 n=9 なら 1,2,3,4
  for(int col=1,col2=size-2;col<col2;col++,col2--){
      aBoard[0]=bit=(1<<col);
      solve_nqueenr(size,mask,1,bit<<1,bit,bit>>1,aBoard);
  }
}
//
//通常版
/***09 backTrack登場メソッド名だけ枝刈りはまだしない*****/ 
void backTrack1D_NR(int size,int mask,int row,int left,int down,int right,unsigned int* aBoard){
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
  if(row==sizeE){
    if(bitmap){
      aBoard[row]=bitmap;
      int s=symmetryOps(size,aBoard);
      if(s!=0){
        UNIQUE++;
        TOTAL+=s;
      }
    }
  }else{
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
//通常版 CPU 非再帰版 ロジックメソッド
/***09 backTrack登場メソッド名だけ枝刈りはまだしない*****/  
void NQueenD(int size,int mask){
  int TOPBIT;
  int ENDBIT;
  int LASTMASK;
  //int SIDEMASK;
  int BOUND1;
  int BOUND2;
  int bit;
  unsigned int aBoard[MAX];
  TOPBIT=1<<(size-1);
  aBoard[0]=1;
  for(BOUND1=2;BOUND1<size-1;BOUND1++){
    aBoard[1]=bit=(1<<BOUND1);
    //backTrack1(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
    backTrack1D_NR(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1),aBoard);
  }
  //SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    //backTrack1(size,mask,1,bit<<1,bit,bit>>1);
    backTrack1D_NR(size,mask,1,bit<<1,bit,bit>>1,aBoard);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  }
}
//
// 通常版
/***09 backTrack登場メソッド名だけ枝刈りはまだしない*****/  
void backTrack1D(int size,int mask,int row,int left,int down,int right,unsigned int* aBoard){
  int bit;
  int bitmap=(mask&~(left|down|right));
  if(row==size){
    // TOTAL++;
    aBoard[row]=bitmap; //symmetryOpsの時は代入します。
    int s=symmetryOps(size,aBoard);
    if(s!=0){
      UNIQUE++;
      TOTAL+=s;
    }
  }else{
    while(bitmap){
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap); //ロジック用
      backTrack1D(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1,aBoard);
    }
  }
}
//
//通常版 CPUR 再帰版　ロジックメソッド
/***09 ミラーリング処理しない*****/ 
void NQueenDR(int size,int mask)
//void NQueenDR(int size,int mask,int row,int left,int down,int right,int ex1,int ex2)
{
  int TOPBIT;
  int ENDBIT;
  int LASTMASK;
  //int SIDEMASK;
  int BOUND1;
  int BOUND2;
  int bit;
  unsigned int aBoard[MAX];
 /***09 backTrack登場メソッド名だけ枝刈りはまだしない*****/  
  TOPBIT=1<<(size-1);
  aBoard[0]=1;
  for(BOUND1=2;BOUND1<size-1;BOUND1++){
    aBoard[1]=bit=(1<<BOUND1);
    backTrack1D(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1),aBoard);
  }
  //SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    backTrack1D(size,mask,1,bit<<1,bit,bit>>1,aBoard);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  }
}
//メインメソッド
int main(int argc,char** argv)
{
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
    printf("\n\n９．CPU 非再帰 クイーンの位置による分岐BOUND1\n");
  }else if(cpur){
    printf("\n\n９．CPUR 再帰 クイーンの位置による分岐BOUND1\n");
  }else if(gpu){
    printf("\n\n９．GPU 非再帰 クイーンの位置による分岐BOUND1\n");
  }else if(sgpu){
    printf("\n\n９．SGPU 非再帰 クイーンの位置による分岐BOUND1\n");
  }
  if(cpu||cpur){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    clock_t st;           //速度計測用
    char t[20];           //hh:mm:ss.msを格納
    int min=4; int targetN=17;
    int mask;
    for(int i=min;i<=targetN;i++){
      TOTAL=0; UNIQUE=0;
      //COUNT2=COUNT4=COUNT8=0;
      mask=(1<<i)-1;
      st=clock();
      //
      //【通常版】
      //if(cpur){ _NQueenR(i,mask,0,0,0,0); }
      //CPUR
      if(cpur){ 
        NQueenR(i,mask); 
        //printf("通常版\n");
        /***09 ミラーリング処理しない*****/ 
        //NQueenDR(i,mask,0,0,0,0,excl,i%2 ? excl : 0);//通常版
        //NQueenDR(i,mask);//通常版
      }
      //CPU
      if(cpu){ 
        NQueen(i,mask); 
        //printf("通常版\n");
        //NQueenD(i,mask,0); //通常版
      }
      //
      TimeFormat(clock()-st,t); 
      printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
    }
  }
  if(gpu||sgpu){
    if(!InitCUDA()){return 0;}
    int min=4;int targetN=17;
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

