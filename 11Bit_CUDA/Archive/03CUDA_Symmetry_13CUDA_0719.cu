/**
 CUDAで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイルと実行
 $ nvcc -O3 CUDA**_N-Queen.cu && ./a.out (-c|-r|-g)
                    -c:cpu 
                    -r cpu再帰 
                    -g GPU 

$ nvcc -O3 CUDA13_N-Queen.cu  && ./a.out -g
１３．GPU 非再帰 並列処理 CUDA
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2               1  00:00:00:00.37
 5:           10               2  00:00:00:00.00
 6:            4               1  00:00:00:00.00
 7:           40               6  00:00:00:00.00
 8:           92              12  00:00:00:00.01
 9:          352              46  00:00:00:00.01
10:          724              92  00:00:00:00.01
11:         2680             341  00:00:00:00.01
12:        14200            1787  00:00:00:00.02
13:        73712            9233  00:00:00:00.03
14:       365596           45752  00:00:00:00.03
15:      2279184          285053  00:00:00:00.04
16:     14772512         1846955  00:00:00:00.08
17:     95815104        11977939  00:00:00:00.35
18:    666090624        83263591  00:00:00:02.60
19:   4968057848       621012754  00:00:00:22.23
20:  39029188884      4878666808  00:00:03:26.80
21: 314666222712     39333324973  00:00:33:09.52
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <pthread.h>
#define THREAD_NUM		96
#define MAX 27

//変数宣言
/**
  CPU/CPURの場合において、
  バックトラックをnon-recursive/recursiveのいずれかを選択
  再帰：0
  非再帰：1
  */
int NR; 
/**
  グローバル変数として合計解とユニーク解を格納する変数
  */
long TOTAL=0;
long UNIQUE=0;
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
//
//ローカル構造体
typedef struct{
  int BOUND1,BOUND2,TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
  int mask;
  int aBoard[MAX];
  long COUNT2[MAX],COUNT4[MAX],COUNT8[MAX];
}local ;
//
__device__ __host__
int symmetryOps(int si,unsigned int *d_aBoard,int BOUND1,int BOUND2,int TOPBIT,int ENDBIT)
{
  int own,ptn,you,bit;
  //90度回転
  if(d_aBoard[BOUND2]==1){ own=1; ptn=2;
    while(own<=si-1){ bit=1; you=si-1;
      while((d_aBoard[you]!=ptn)&&(d_aBoard[own]>=bit)){ bit<<=1; you--; }
      if(d_aBoard[own]>bit){ return 0; } else if(d_aBoard[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>si-1){ return 2; }
  }
  //180度回転
  if(d_aBoard[si-1]==ENDBIT){ own=1; you=si-1-1;
    while(own<=si-1){ bit=1; ptn=TOPBIT;
      while((d_aBoard[you]!=ptn)&&(d_aBoard[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(d_aBoard[own]>bit){ return 0; } else if(d_aBoard[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>si-1){ return 4; }
  }
  //270度回転
  if(d_aBoard[BOUND1]==TOPBIT){ own=1; ptn=TOPBIT>>1;
    while(own<=si-1){ bit=1; you=0;
      while((d_aBoard[you]!=ptn)&&(d_aBoard[own]>=bit)){ bit<<=1; you++; }
      if(d_aBoard[own]>bit){ return 0; } else if(d_aBoard[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  return 8; 
}
//
__global__
void cuda_kernel_b1(
    register int size,
    register int mark,
    unsigned int* totalDown,
    unsigned int* totalLeft,
    unsigned int* totalRight,
    unsigned int* d_results,
    unsigned int* d_uniq,
    register int totalCond,
    /**11 backTrack1ではaBoard不要のためコメント*********************/
    //unsigned int* t_aBoard,
    register int h_row,
    /**11 BOUND1追加*********************/
    int B1
    )
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
  /***11 backTrack1ではaBoard不要 *********************/
  //unsigned int c_aBoard[MAX];
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
    /***11 backTrack1ではaBoard不要*********************/
    //for(int i=0;i<h_row;i++){
    //  c_aBoard[i]=t_aBoard[idx*h_row+i]; //２次元配列だが1次元的に利用  
    //}
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
        /**11 枝刈り**********/
        if(row+h_row<B1) {
          bitmap_tid_row=bitmap[tid][row]&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
        }  
        //クイーンを置く
        //置く場所があるかどうか
        bitmap[tid][row]
          /***11 backTrack1ではaBoard不要のためコメント*********************/
          //^=c_aBoard[row+h_row]
          //=bit
          ^=bit
          =(-bitmap_tid_row&bitmap_tid_row);       
        if((bit&mask)!=0){
          //最終行?最終行から１個前の行まで
          //無事到達したら 加算する
          if(row+1==mark){
            /**11 backTradk1ではsymmetryOps不要のためコメント*********************/
            //int s=symmetryOps(size,c_aBoard); 
            //if(s!=0){
            //print(size); //print()でTOTALを++しない
            //ホストに戻す配列にTOTALを入れる
            //スレッドが１つの場合は配列は１個
            unique++; 
            total+=8;   //対称解除で得られた解数を加算
            //}
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
//
/***11 cuda_kernel_b2新設*********************/
__global__
void cuda_kernel_b2(
    register int size,
    register int mark,
    unsigned int* totalDown,
    unsigned int* totalLeft,
    unsigned int* totalRight,
    unsigned int* d_results,
    unsigned int* d_uniq,
    register int totalCond,
    unsigned int* t_aBoard,
    register int h_row,
    register int B1,
    register int B2,
    register int SM,
    register int LM,
    /***12 symmetryOps 省力化のためTOPBIT,ENDBITを渡す*****/ 
    register int TB,
    register int EB
    )
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
      //
      //bitmap[tid][row]=00000000 クイーンを
      //どこにも置けないので1行上に戻る
      if(bitmap_tid_row==0){
        row--;
      }else{
        /**11 枝刈り追加**********/
        //【枝刈り】上部サイド枝刈り
        if(row+h_row<B1){             	
          //printf("BOUND1_row:%d:h_row:%d:row+hrow:%d:bit:%d\n",row,h_row,row+h_row,bitmap[tid][row]);
          bitmap_tid_row=bitmap[tid][row]&=~SM;
          //【枝刈り】下部サイド枝刈り
        }else if(row+h_row==B2) {     	
          //printf("BOUND2_row:%d:h_row:%d:row+hrow:%d:bit:%d\n",row,h_row,row+h_row,bitmap[tid][row]);
          if((down_tid_row&SM)==0){ 
            row--; 
            continue;
            //printf("BOUND2_row\n");
          }
          if((down_tid_row&SM)!=SM){ 
            bitmap_tid_row=bitmap[tid][row]&=SM; 
            //printf("BOUND2_SIDEMASK\n");            
          }
        }
        int save_bitmap=bitmap[tid][row];
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
            /***11 LASTMASK枝刈り*********************/ 
            if((save_bitmap&LM)==0){ 
              /***12 symmetryOps 省力化のためBOUND1,BOUND2,TOPBIT,ENDBITを渡す*****/ 
              int s=symmetryOps(size,c_aBoard,B1,B2,TB,EB); 
              if(s!=0){
                //print(size); //print()でTOTALを++しない
                //ホストに戻す配列にTOTALを入れる
                //スレッドが１つの場合は配列は１個
                unique++; 
                total+=s;   //対称解除で得られた解数を加算
              }
              row--;
            }
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
long backTrack2G(int size,int mask,int row,int n_left,int n_down,int n_right,int steps,int BOUND1,int BOUND2,int SIDEMASK,int LASTMASK,int TOPBIT,int ENDBIT,unsigned int* aBoard)
{
  //何行目からGPUで行くか。ここの設定は変更可能、設定値を多くするほどGPUで並行して動く
  /***11 size<8の時はmarkが2*********************/
  unsigned int mark=size>12?size-10:3;
  //unsigned int mark=size>11?size-9:3;
  if(size<8){ mark=2; }
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
  register int rowP=0;
  while(row>=h_mark) {
    //bitmap[row]=00000000 クイーンを
    //どこにも置けないので1行上に戻る
    //06GPU こっちのほうが優秀
    if(bitmap[row]==0){ row--; }
    else{//おける場所があれば進む
      /***11 枝刈り追加*********************/
      //【枝刈り】上部サイド枝刈り
      if(row<BOUND1){             	
        bitmap[row]&=~SIDEMASK;
        //【枝刈り】下部サイド枝刈り
      }else if(row==BOUND2) {     	
        if((down[row]&SIDEMASK)==0){ row--; }
        if((down[row]&SIDEMASK)!=SIDEMASK){ bitmap[row]&=SIDEMASK; }
      }
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
            /***12 TOPBIT,ENDBIT追加*********************/
            //cuda_kernel_b2<<<steps/THREAD_NUM,THREAD_NUM
            //  >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,row,BOUND1,BOUND2,SIDEMASK,LASTMASK);
            cuda_kernel_b2<<<steps/THREAD_NUM,THREAD_NUM
              >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,row,BOUND1,BOUND2,SIDEMASK,LASTMASK,TOPBIT,ENDBIT);
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
  /***12 TOPBIT,ENDBIT追加*********************/
  //cuda_kernel_b2<<<steps/THREAD_NUM,THREAD_NUM
  //  >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,mark,BOUND1,BOUND2,SIDEMASK,LASTMASK);
  cuda_kernel_b2<<<steps/THREAD_NUM,THREAD_NUM
    >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,mark,BOUND1,BOUND2,SIDEMASK,LASTMASK,TOPBIT,ENDBIT);
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
//
long backTrack1G(int size,int mask,int row,int n_left,int n_down,int n_right,int steps,int BOUND1)
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
  /***11 backTrack1ではaBoard不要のためコメント*********************/
  //unsigned int* t_aBoard;
  //cudaMallocHost((void**) &t_aBoard,sizeof(int)*steps*mark);
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
  /***11 backTrack1ではaBoard不要のためコメント*********************/
  //unsigned int* d_aBoard;
  //cudaMalloc((void**) &d_aBoard,sizeof(int)*steps*mark);
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
      /***11 枝刈り*********************/
      if(row<BOUND1) {
        bitmap[row]&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
      }
      //06SGPU
      /***11 aBoard不要*********************/
      //bitmap[row]^=aBoard[row]=bit=(-bitmap[row]&bitmap[row]);
      bitmap[row]^=bit=(-bitmap[row]&bitmap[row]);
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
          /***11 aBoardコメント*********************/
          //for(int i=0;i<mark;i++){
          //  t_aBoard[totalCond*mark+i]=aBoard[i];
          //}
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
            /***11 aBoard不要のためコメント*********************/
            //cudaMemcpy(d_aBoard,t_aBoard,
            //    sizeof(int)*totalCond*mark,cudaMemcpyHostToDevice);
            /***11 BOUND1追加*********************/
            //cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
            //  >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,row);
            cuda_kernel_b1<<<steps/THREAD_NUM,THREAD_NUM
              >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,row,BOUND1);

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
  /***11 aBoard不要のためコメント*********************/
  //cudaMemcpy(d_aBoard,t_aBoard,
  //    sizeof(int)*totalCond*mark,cudaMemcpyHostToDevice);
  //size-mark は何行GPUを実行するか totalCondはスレッド数
  //steps数の数だけマルチスレッドで起動するのだが、実際に計算が行われるのは
  //totalCondの数だけでそれ以外は空回しになる
  /***11 BOUND1追加*********************/
  //cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
  //  >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,mark);
  cuda_kernel_b1<<<steps/THREAD_NUM,THREAD_NUM
    >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,mark,BOUND1);
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
  /***11 aBoardコメント**/
  //cudaFree(d_aBoard);
  cudaFreeHost(totalDown);
  cudaFreeHost(totalLeft);
  cudaFreeHost(totalRight);
  cudaFreeHost(h_results);
  cudaFreeHost(h_uniq);
  /***11 aBoardコメント**/
  //cudaFreeHost(t_aBoard);
  return total;
}
//
//GPU
void NQueenG(register int size,register int steps)
{
  if(size<=0||size>32){return;}
  /**
    パラメータは渡す変数はregisterとする
    int型は unsigned とする
    total: グローバル変数TOTALへのアクセスを極小化する
    sizeE:size-1といった計算を変数に格納しフラット化する 
    */
  unsigned int total=0;
  unsigned int sizeE=size-1;
  register unsigned int aBoard[MAX];
  register int bit=0;
  register int mask=((1<<size)-1);
  int col=0;//1行め右端 0
  aBoard[0]=bit=(1<<col);
  register int left=bit<<1,down=bit,right=bit>>1;
  /**
    2行目は右から3列目から左端から2列目まで
  */
  for(register int BOUND1=2;BOUND1<sizeE;BOUND1++){
    aBoard[1]=bit=(1<<BOUND1);
    total+=backTrack1G(size,mask,2,
        (left|bit)<<1,(down|bit),(right|bit)>>1,
        steps,BOUND1);
  }
  register int LASTMASK,SIDEMASK;
  register int TOPBIT=1<<(sizeE);
  SIDEMASK=LASTMASK=(TOPBIT|1);
  register int ENDBIT=(TOPBIT>>1);
  /**
    1行目右から2列目から
    偶数個は1/2 n=8 なら 1,2,3 奇数個は1/2+1 n=9 なら 1,2,3,4
  */
  for(register int BOUND1=1,BOUND2=sizeE-1;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    total+=backTrack2G(size,mask,1,
        bit<<1,bit,bit>>1,
        steps,BOUND1,BOUND2,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,aBoard);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  }
  /**
    グローバル変数へのアクセスを極小化する
    */
  TOTAL=total;
}
/** CUDA 初期化 **/
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
//
void symmetryOps(local *l)
{
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
//
//CPU 非再帰版 backTrack2//新しく記述
void backTrack2_NR(int row,int h_left,int h_down,int h_right,local *l)
{
    unsigned int left[G.size];
    unsigned int down[G.size];
    unsigned int right[G.size];
    unsigned int bitmap[G.size];
    left[row]=h_left;
    down[row]=h_down;
    right[row]=h_right;
    bitmap[row]=l->mask&~(left[row]|down[row]|right[row]);
    unsigned int bit;
    int mark=row;
    //固定していれた行より上はいかない
    while(row>=mark){//row=1 row>=1, row=2 row>=2
      if(bitmap[row]==0){
        --row;
      }else{
	//【枝刈り】上部サイド枝刈り
	if(row<l->BOUND1){             	
	  bitmap[row]&=~l->SIDEMASK;
        //【枝刈り】下部サイド枝刈り
        }else if(row==l->BOUND2) {     	
          if((down[row]&l->SIDEMASK)==0){ row--; }
          if((down[row]&l->SIDEMASK)!=l->SIDEMASK){ bitmap[row]&=l->SIDEMASK; }
        }
        int save_bitmap=bitmap[row];
        bitmap[row]^=l->aBoard[row]=bit=(-bitmap[row]&bitmap[row]); 
        if((bit&l->mask)!=0){
          if(row==G.sizeE){
            if((save_bitmap&l->LASTMASK)==0){ 	
              symmetryOps(l);
              --row;
		    }
          }else{
            int n=row++;
            left[row]=(left[n]|bit)<<1;
            down[row]=down[n]|bit;
            right[row]=(right[n]|bit)>>1;
            bitmap[row]=l->mask&~(left[row]|down[row]|right[row]);
          }
        }else{
           --row;
        }
      }  
    }
}
//
//通常版 CPU 非再帰版 backTrack2
void backTrack2D_NR(int row,int left,int down,int right,local *l)
{
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
void backTrack1_NR(int row,int h_left,int h_down,int h_right,local *l)
{
  unsigned int left[G.size];
  unsigned int down[G.size];
  unsigned int right[G.size];
  unsigned int bitmap[G.size];
  left[row]=h_left;
  down[row]=h_down;
  right[row]=h_right;
  bitmap[row]=l->mask&~(left[row]|down[row]|right[row]);
  unsigned int bit;
  int mark=row;
  //固定していれた行より上はいかない
  while(row>=mark){//row=1 row>=1, row=2 row>=2
    if(bitmap[row]==0){
      --row;
    }else{
      if(row<l->BOUND1) {
        bitmap[row]&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
      }
      bitmap[row]^=l->aBoard[row]=bit=(-bitmap[row]&bitmap[row]); 
      if((bit&l->mask)!=0){
        if(row==G.sizeE){
          l->COUNT8[l->BOUND1]++;
          --row;
        }else{
          int n=row++;
          left[row]=(left[n]|bit)<<1;
          down[row]=down[n]|bit;
          right[row]=(right[n]|bit)>>1;
          bitmap[row]=l->mask&~(left[row]|down[row]|right[row]);
        }
      }else{
        --row;
      }
    }  
  }
}
//通常版 CPU 非再帰版 backTrack
void backTrack1D_NR(int row,int left,int down,int right,local *l)
{
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
//CPU 再帰版 backTrack
void backTrack2(int row,int left,int down,int right,local *l)
{
  int bitmap=0;
  int bit=0;
  bitmap=(l->mask&~(left|down|right));
  if(row==G.sizeE){
    if(bitmap){
      //【枝刈り】 最下段枝刈り
      if((bitmap&l->LASTMASK)==0){ 	
        l->aBoard[row]=(-bitmap&bitmap);
        symmetryOps(l);
      }
    }
  }else{
    //【枝刈り】上部サイド枝刈り
    if(row<l->BOUND1){             	
      bitmap&=~l->SIDEMASK;
      //【枝刈り】下部サイド枝刈り
    }else if(row==l->BOUND2) {     	
      if((down&l->SIDEMASK)==0){ return; }
      if((down&l->SIDEMASK)!=l->SIDEMASK){ bitmap&=l->SIDEMASK; }
    }
    while(bitmap){
      bitmap^=l->aBoard[row]=bit=(-bitmap&bitmap);
      backTrack2(row+1,(left|bit)<<1, down|bit,(right|bit)>>1,l);
    }
  }
}
//通常版 CPU 再帰版 backTrack
void backTrack2D(int row,int left,int down,int right,local *l)
{
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
      backTrack2D(row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
}
//
//CPU 再帰版 backTrack
void backTrack1(int row,int left,int down,int right,local *l)
{
  int bitmap=0;
  int bit=0;
  bitmap=(l->mask&~(left|down|right));
  if(row==G.sizeE){
    if(bitmap){
      l->COUNT8[l->BOUND1]++;
    }
  }else{
    if(row<l->BOUND1) {
      bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
    while(bitmap){
      bitmap^=l->aBoard[row]=bit=(-bitmap&bitmap);
      backTrack1(row+1,(left|bit)<<1, down|bit,(right|bit)>>1,l);
    }
  }
}
//通常版 CPU 再帰版 backTrack
void backTrack1D(int row,int left,int down,int right,local *l)
{
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
      backTrack1D(row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
}
//チルドスレッドの実行処理
void *run(void *args)
{
  /**
    //グローバル構造体
    typedef struct {
      int size;
      int sizeE;
      long lTOTAL,lUNIQUE;
    }GCLASS, *GClass;
    GCLASS G;
  */
  local *l=(local *)args;
  /**
    最上段のクイーンが「角」にある場合の探索
  */
  int bit=0;
  int col=0;
  if(l->BOUND1>1 && l->BOUND1<G.sizeE) {
    l->aBoard[0]=bit=(1<<col);
    int left=bit<<1;int down=bit;int right=bit>>1;
    if(l->BOUND1<G.sizeE) {
      col=l->BOUND1;// 角にクイーンを配置
      l->aBoard[1]=bit=(1<<col);
      if(NR==1){//２行目から非再帰
        backTrack1_NR(2,(left|bit)<<1,(down|bit),(right|bit)>>1,l);//GPU適用版
        //backTrack1D_NR(2,(left|bit)<<1,(down|bit),(right|bit)>>1,l);
      }else{//２行目から再帰
        backTrack1(2,(left|bit)<<1,(down|bit),(right|bit)>>1,l);//GPU適用版
        //backTrack1D(2,(left|bit)<<1,(down|bit),(right|bit)>>1,l);//通常版
      }
    }
  }
  l->TOPBIT=1<<(G.sizeE);
  l->ENDBIT=(l->TOPBIT>>l->BOUND1);
  l->SIDEMASK=l->LASTMASK=(l->TOPBIT|1);
  /**
    最上段行のクイーンが「角以外」にある場合の探索
    ユニーク解に対する左右対称解を予め削除するには、
    左半分だけにクイーンを配置するようにすればよい 
   */
  if(l->BOUND1>0&&l->BOUND2<G.sizeE&&l->BOUND1<l->BOUND2){
    for(int i=1; i<l->BOUND1; i++){
      l->LASTMASK=l->LASTMASK|l->LASTMASK>>1|l->LASTMASK<<1;
    }
    if(l->BOUND1<l->BOUND2){
      int col=l->BOUND1;
      l->aBoard[0]=bit=(1<<col);
      if(NR==1){//２行目から非再帰
        backTrack2_NR(1,bit<<1,bit,bit>>1,l); //GPU適用版
        //backTrack2D_NR(1,bit<<1,bit,bit>>1,l);//通常版
      }else{//２行目から再帰
        backTrack2(1,bit<<1,bit,bit>>1,l); //GPU適用版
        //backTrack2D(1,bit<<1,bit,bit>>1,l);//通常版
      }
    }
    l->ENDBIT>>=G.size;
  }
  return 0;//*run()の場合はreturn 0;が必要
}
//pthreadによるスレッド生成
void *NQueenThread()
{
  /**
    //ローカル構造体
    typedef struct{
      int BOUND1,BOUND2,TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
      int mask;
      int aBoard[MAX];
      long COUNT2[MAX],COUNT4[MAX],COUNT8[MAX];
    }local ;
  */
  local l[MAX];//構造体 local型
  /**
    pthreadのチルドスレッド
    */
  pthread_t pt[G.size];
  /**
    初期化とチルドスレッドの生成 
    */
  for(int BOUND1=G.sizeE,BOUND2=0;BOUND2<G.sizeE;BOUND1--,BOUND2++){
    /**
        aBoardとカウンターの初期化
      */
    l[BOUND1].mask=(1<<G.size)-1;
    l[BOUND1].BOUND1=BOUND1;l[BOUND1].BOUND2=BOUND2;//B1 と B2を初期化
    for(int j=0;j<G.size;j++){ l[l->BOUND1].aBoard[j]=j; }// aB[]の初期化
    l[BOUND1].COUNT2[BOUND1]=l[BOUND1].COUNT4[BOUND1]=
      l[BOUND1].COUNT8[BOUND1]=0;//カウンターの初期化
    /**
      チルドスレッドの生成
      pthread_createでチルドスレッドを生成します。
      BOUND1がインクリメントされ、Nの数だけチルドスレッドが生成されます。
      run()がチルドスレッドの実行関数となります。
    */
    int iFbRet=pthread_create(&pt[BOUND1],NULL,&run,&l[BOUND1]);
    if(iFbRet>0){
      printf("[mainThread] pthread_create #%d: %d\n", l[BOUND1].BOUND1, iFbRet);
    }
  }
  /**
    チルドスレッドが実行され、全ての処理が完了するまでjoin()により待機します。
    */
  for(int BOUND1=G.sizeE,BOUND2=0;BOUND2<G.sizeE;BOUND1--,BOUND2++){
    pthread_join(pt[BOUND1],NULL);
  }
  //スレッド毎のカウンターを合計
  for(int BOUND1=G.sizeE,BOUND2=0;BOUND2<G.sizeE;BOUND1--,BOUND2++){
    G.lTOTAL+=l[BOUND1].COUNT2[BOUND1]*2+
      l[BOUND1].COUNT4[BOUND1]*4+l[BOUND1].COUNT8[BOUND1]*8;
    G.lUNIQUE+=l[BOUND1].COUNT2[BOUND1]+
      l[BOUND1].COUNT4[BOUND1]+l[BOUND1].COUNT8[BOUND1];
  }
  return 0;
}
// CPU/CPURの並列処理(pthread)
void NQueen()
{
  /**
   メインスレッドの生成
   拡張子 CUDA はpthreadをサポートしていませんので実行できません
   コンパイルが通らないので 以下をコメントアウトします
   Cディレクトリの 並列処理はC13_N-Queen.c を参考にして下さい。
  
   pthreadを使いたいときはここのコメントアウトを外します
   //iFbRet = pthread_create(&pth, NULL,&NQueenThread,NULL);
  
  */
  pthread_t pth;  //スレッド変数
  int iFbRet;
  //pthreadを使いたいときはここのコメントアウトを外します
  //iFbRet = pthread_create(&pth, NULL,&NQueenThread,NULL);
  //
  if(iFbRet>0){
    printf("[main] pthread_create: %d\n", iFbRet); //エラー出力デバッグ用
  }
  pthread_join(pth,NULL); /* いちいちjoinをする */
}
//
int main(int argc,char** argv)
{
  /**
    実行パラメータの処理
    $ nvcc -O3 CUDA13_N-Queen.cu && ./a.out (-c|-r|-g|-s)
                       -c:cpu 
                       -r cpu再帰 
                       -g GPU 
                       -s SGPU(サマーズ版と思われる)
    */
  bool cpu=false,cpur=false,gpu=false,sgpu=false;
  int argstart=1;
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
    else if(argv[1][1]=='s'||argv[1][1]=='S'){sgpu=true;}
    else{ gpu=true; } //デフォルトをgpuとする
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g|-r|-s] n steps\n",argv[0]);
    printf("  -r: CPUR only\n");
    printf("  -c: CPU only\n");
    printf("  -g: GPU only\n");
    printf("  -s: SGPU only\n");
    printf("Default to 8 queen\n");
  }
  /** 
    出力
      $ nvcc コマンドではpthreadは動きません
      cpu/cpurを実行したい場合はファイル名の拡張子をcに変更した上で、
      以下の行頭のコメントを外してください。
      #
      //iFbRet = pthread_create(&pth, NULL,&NQueenThread,NULL);
      #
      １．ソースファイルを複写し、
      複写したソースファイルの拡張子を .cu から .cにリネームする
      CUDA13_N-Queen.cu -> CUDA13_N-Queen.c
      ２．ソースファイルから以下の行を探し、行頭のコメントを外してください。
      //iFbRet = pthread_create(&pth, NULL,&NQueenThread,NULL);
      ３．以下のコマンドで実行する
      $ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread CUDA13_N-Queen.c && ./a.out [-c|-r]
   */
  if(cpu){
    printf("\n\n１３．CPU 非再帰 並列処理\n");
    printf("pthread\n※nvccではpthreadは動きません！\n");
  }else if(cpur){
    printf("\n\n１３．CPUR 再帰 並列処理\n");
    printf("pthread\n※nvccではpthreadは動きません！\n");
  }else if(gpu){
    printf("\n\n１３．GPU 非再帰 並列処理 CUDA\n");
  }else if(sgpu){
    printf("\n\n１３．SGPU 非再帰 並列処理 CUDA\n");
  }
  /**
    CPU（CPUによる非再起処理）とCPUR（CPUによる再帰処理）の実行
    */
  if(cpu||cpur)
  {
    int min=4; int targetN=17;//実行の開始Nと終了Nの指定
    /**
      処理時刻計測のための変数
      */
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:           Total           Unique          dd:hh:mm:ss.ms");
    for(int i=min;i<=targetN;i++){
      /**
        size/sizeE/lTOTAL/lUNIQUEといった変数は構造体に格納されています。

        typedef struct {
          int size;
          int sizeE;
          long lTOTAL,lUNIQUE;
        }GCLASS, *GClass;
        GCLASS G;

        */
      G.size=i;G.sizeE=i-1;//size sizeEの初期化
      G.lTOTAL=G.lUNIQUE=0;//TOTAL UNIQUEの初期化
      //
      gettimeofday(&t0, NULL);//計測開始
      /**
        CPU/CPURの場合において、
        バックトラックをnon-recursive/recursiveのいずれかを選択
        再帰：0
        非再帰：1
        */
      if(cpur){ //再帰
        //NR=0;NQueenD();
        NR=0;NQueen();
      }
      if(cpu){ //非再帰
        //NR=1;NQueenD();
        NR=1;NQueen();
      }
      //
      gettimeofday(&t1, NULL);//計測終了
      /**
        時刻表記の処理
          Total           Unique          dd:hh:mm:ss.ms
        15:      2279184          285053  00:00:00:00.33
        16:     14772512         1846955  00:00:00:01.59
        17:     95815104        11977939  00:00:00:10.92
        */
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
      /**
        出力
        */
      printf("%2d:%16ld%17ld%12.2d:%02d:%02d:%02d.%02d\n",
          i,G.lTOTAL,G.lUNIQUE,dd,hh,mm,ss,ms);
    } //end for
  }//end if
  /**
    GPU（過去の記録樹立者サマーズGPU版SGPU処理）と
    GPUR（GPUによる再帰処理）の実行
    */
  if(gpu||sgpu)
  {
    /**
      実行時にデバイスがCUDAがサポートされているかを確認
    */
    if(!InitCUDA()){return 0;}
    int steps=24576;
    int min=4;int targetN=21;//実行の開始Nと終了Nの指定
    /**
      処理時刻計測のための変数
      */
    struct timeval t0;
    struct timeval t1;
    /**
      出力
      */
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    /**
      実行処理
      */
    for(int i=min;i<=targetN;i++){
      gettimeofday(&t0,NULL);   // 計測開始
      if(gpu){//本ソースのメイン処理
        TOTAL=0;
        UNIQUE=0;
        NQueenG(i,steps);
      }
      gettimeofday(&t1,NULL);   // 計測終了
      /**
        時刻表記の処理
          Total           Unique          dd:hh:mm:ss.ms
        15:      2279184          285053  00:00:00:00.33
        16:     14772512         1846955  00:00:00:01.59
        17:     95815104        11977939  00:00:00:10.92
        */
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
      printf("%2d:%13ld%16ld%4.2d:%02d:%02d:%02d.%02d\n",
          i,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
    }//end for
  }//end if
  return 0;
}
