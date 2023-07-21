
/**
 CUDAで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイルと実行
 $ nvcc CUDA**_N-Queen.cu && ./a.out (-c|-r|-g)
                    -c:cpu 
                    -r cpu再帰 
                    -g GPU 

１２．対称解除法の最適化

bash-3.2$ nvcc CUDA12_N-Queen.cu && ./a.out -g
１２．GPU 非再帰 枝刈り
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2               1  00:00:00:00.04
 5:           10               2  00:00:00:00.00
 6:            4               1  00:00:00:00.00
 7:           40               6  00:00:00:00.01
 8:           92              12  00:00:00:00.02
 9:          352              46  00:00:00:00.02
10:          724              92  00:00:00:00.02
11:         2680             341  00:00:00:00.03
12:        14200            1787  00:00:00:00.07
13:        73712            9233  00:00:00:00.16
14:       365596           45752  00:00:00:00.13
15:      2279184          285053  00:00:00:00.34
16:     14772512         1846955  00:00:00:01.62
17:     95815104        11977939  00:00:00:10.87
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
__device__  __host__
int symmetryOps(int si,unsigned int *d_aBoard,int BOUND1,int BOUND2,int TOPBIT,int ENDBIT){
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
    register long totalCond,
    /**11 backTrack1ではaBoard不要のためコメント*********************/
    //unsigned int* t_aBoard,
    register int h_row,
    /**11 BOUND1追加*********************/
    int B1
    )
{
  register const unsigned int mask=(1<<size)-1;
  register unsigned long total=0;
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
    register long totalCond,
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
  register unsigned long total=0;
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
/***12 symmetryOps 省力化のためTOPBIT,ENDBITを渡す*****/ 
//long backTrack2G(int size,int mask,int row,int n_left,int n_down,int n_right,int steps,int BOUND1,int BOUND2,int SIDEMASK,int LASTMASK,unsigned int* aBoard)
long backTrack2G(int size,int mask,int row,int n_left,int n_down,int n_right,int steps,int BOUND1,int BOUND2,int SIDEMASK,int LASTMASK,int TOPBIT,int ENDBIT,unsigned int* aBoard)
{
  //何行目からGPUで行くか。ここの設定は変更可能、設定値を多くするほどGPUで並行して動く
  /***11 size<8の時はmarkが2*********************/
  unsigned int mark=size>12?size-10:3;
  //unsigned int mark=size>11?size-9:3;
  if(size<8){ mark=2; }
  const unsigned int h_mark=row;
  long total=0;
  long totalCond=0;
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
/***11 枝刈りをするので引数を追加,aBoardは不要*********************/
//long backTrack1G(int size,int mask,int row,int n_left,int n_down,int n_right,int steps,unsigned int* aBoard)
long backTrack1G(int size,int mask,int row,int n_left,int n_down,int n_right,int steps,int BOUND1)
{
  //何行目からGPUで行くか。ここの設定は変更可能、設定値を多くするほどGPUで並行して動く
  /***08 クイーンを２行目まで固定で置くためmarkが3以上必要*********************/
  const unsigned int mark=size>12?size-10:3;
  const unsigned int h_mark=row;
  long total=0;
  long totalCond=0;
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
void NQueenG(int size,int steps)
{
  int TOPBIT;
  int ENDBIT;
  int LASTMASK;
  int SIDEMASK;
  int BOUND1;
  int BOUND2;
  unsigned int aBoard[MAX];
  register int bit=0;
  register int mask=((1<<size)-1);
  TOPBIT=1<<(size-1);
  if(size<=0||size>32){return;}
  /***09 backtrack1*********************/
  //1行め右端 0
  int col=0;
  aBoard[0]=bit=(1<<col);
  int left=bit<<1;
  int down=bit;
  int right=bit>>1;
  //2行目は右から3列目から左端から2列目まで
  for(int col_j=2;col_j<size-1;col_j++){
      aBoard[1]=bit=(1<<col_j);
      /***11 BOUND1*********************/
      BOUND1=col_j;
      /***11 枝刈りするので引数を渡す*********************/
      //TOTAL+=backTrack1G(size,mask,2,(left|bit)<<1,(down|bit),(right|bit)>>1,steps,aBoard);
      TOTAL+=backTrack1G(size,mask,2,(left|bit)<<1,(down|bit),(right|bit)>>1,steps,BOUND1);
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  /***09 backtrack2*********************/
  //1行目右から2列目から
  //偶数個は1/2 n=8 なら 1,2,3 奇数個は1/2+1 n=9 なら 1,2,3,4
  for(int col=1,col2=size-2;col<col2;col++,col2--){
      aBoard[0]=bit=(1<<col);
      BOUND1=col;
      BOUND2=col2;
      /***12 symmetryOps 省力化のためTOPBIT,ENDBITを渡す*****/ 
      //TOTAL+=backTrack2G(size,mask,1,bit<<1,bit,bit>>1,steps,BOUND1,BOUND2,SIDEMASK,LASTMASK,aBoard);
      TOTAL+=backTrack2G(size,mask,1,bit<<1,bit,bit>>1,steps,BOUND1,BOUND2,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,aBoard);
      LASTMASK|=LASTMASK>>1|LASTMASK<<1;
      ENDBIT>>=1;
  }
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

/***12 symmetryOps 省力化のためTOPBIT,ENDBITを渡す*****/ 
//void backTrack2(int size,int mask, int row,int h_left,int h_down,int h_right,int BOUND1,int BOUND2,int SIDEMASK,int LASTMASK,unsigned int* aBoard)
void backTrack2(int size,int mask, int row,int h_left,int h_down,int h_right,int BOUND1,int BOUND2,int SIDEMASK,int LASTMASK,int TOPBIT,int ENDBIT,unsigned int* aBoard)
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
        /***11 【枝刈り】上部サイド枝刈り*********************/
        if(row<BOUND1){             	
          bitmap[row]&=~SIDEMASK;
        /***11 【枝刈り】下部サイド枝刈り*********************/
        }else if(row==BOUND2) {     	
          if((down[row]&SIDEMASK)==0){ 
              row--; 
          }
          if((down[row]&SIDEMASK)!=SIDEMASK){ 
              bitmap[row]&=SIDEMASK;
              }
        }
        int save_bitmap=bitmap[row];
        bitmap[row]^=aBoard[row]=bit=(-bitmap[row]&bitmap[row]); 
        if((bit&mask)!=0){
          if(row==sizeE){
            /***11 【枝刈り】 最下段枝刈り*********************/
            if((save_bitmap&LASTMASK)==0){ 
              /***12 symmetryOps 省力化のためBOUND1,BOUND2,TOPBIT,ENDBITを渡す*****/ 
              int s=symmetryOps(size,aBoard,BOUND1,BOUND2,TOPBIT,ENDBIT);
              if(s!=0){
                UNIQUE++;
                TOTAL+=s;
              }
              --row;
            }
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
/***11 枝刈りをするので引数を追加*********************/
 //void backTrack1(int size,int mask, int row,int h_left,int h_down,int h_right,unsigned int* aBoard)
void backTrack1(int size,int mask, int row,int h_left,int h_down,int h_right,int BOUND1,unsigned int* aBoard)
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
        /***11　【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい　*****/
        // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
        if(row<BOUND1) {
          bitmap[row]&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
        }
        bitmap[row]^=aBoard[row]=bit=(-bitmap[row]&bitmap[row]); 
        if((bit&mask)!=0){
          if(row==sizeE){
            /***11　【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略　*****/   
            //int s=symmetryOps(size,aBoard);
            //if(s!=0){
            UNIQUE++;
            TOTAL+=8;
            //}
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
  int TOPBIT;
  int ENDBIT;
  int LASTMASK;
  int SIDEMASK;
  int BOUND1;
  int BOUND2;
  register int bit;
  TOPBIT=1<<(size-1);
  unsigned int aBoard[MAX];
  bit=0;
  if(size<=0||size>32){return;}
  /***09 backtrack1***/
  //1行め右端 0
  int col=0;
  aBoard[0]=bit=(1<<col);
  int left=bit<<1;
  int down=bit;
  int right=bit>>1;
  /***09 2行目は右から3列目から左端から2列目まで***/
  for(int col_j=2;col_j<size-1;col_j++){
      aBoard[1]=bit=(1<<col_j);
      BOUND1=col_j;
      /***11 枝刈りするので引数を渡す*********************/
      //backTrack1(size,mask,2,(left|bit)<<1,(down|bit),(right|bit)>>1,aBoard);
      backTrack1(size,mask,2,(left|bit)<<1,(down|bit),(right|bit)>>1,BOUND1,aBoard);
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  /***09 backtrack2***/
  //1行目右から2列目から
  //偶数個は1/2 n=8 なら 1,2,3 奇数個は1/2+1 n=9 なら 1,2,3,4
  for(int col=1,col2=size-2;col<col2;col++,col2--){
      aBoard[0]=bit=(1<<col);
      BOUND1=col;
      BOUND2=col2;
      /***12 symmetryOps 省力化のためTOPBIT,ENDBITを渡す*****/ 
      //backTrack2(size,mask,1,bit<<1,bit,bit>>1,BOUND1,BOUND2,SIDEMASK,LASTMASK,aBoard);
      backTrack2(size,mask,1,bit<<1,bit,bit>>1,BOUND1,BOUND2,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,aBoard);
      LASTMASK|=LASTMASK>>1|LASTMASK<<1;
      ENDBIT>>=1;
  }
}
//CPUR 再帰版 ロジックメソッド
/***12 symmetryOps 省力化のためTOPBIT,ENDBITを渡す*****/ 
//void backTrackR2(int size,int mask, int row,int left,int down,int right,int BOUND1,int BOUND2,int SIDEMASK,int LASTMASK,unsigned int* aBoard)
void backTrackR2(int size,int mask, int row,int left,int down,int right,int BOUND1,int BOUND2,int SIDEMASK,int LASTMASK,int TOPBIT,int ENDBIT,unsigned int* aBoard)
{
 int bitmap=0;
 int bit=0;
 int sizeE=size-1;
 bitmap=(mask&~(left|down|right));
 if(row==sizeE){
    if(bitmap){
      /***11 【枝刈り】 最下段枝刈り*********************/
      if((bitmap&LASTMASK)==0){ 
        aBoard[row]=(-bitmap&bitmap);
        /***12 symmetryOps 省力化のためBOUND1,BOUND2,TOPBIT,ENDBITを渡す*****/ 
        int s=symmetryOps(size,aBoard,BOUND1,BOUND2,TOPBIT,ENDBIT);
        if(s!=0){
          UNIQUE++;
          TOTAL+=s;
        }
      }
    }
  }else{
    /***11 【枝刈り】上部サイド枝刈*********************/
    if(row<BOUND1){             	
      bitmap&=~SIDEMASK;
    /***11 【枝刈り】下部サイド枝刈り*********************/
    }else if(row==BOUND2) {     	
      if((down&SIDEMASK)==0){ return; }
      if((down&SIDEMASK)!=SIDEMASK){ bitmap&=SIDEMASK; }
    }
    while(bitmap){
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
      backTrackR2(size,mask,row+1,(left|bit)<<1, down|bit,(right|bit)>>1,BOUND1,BOUND2,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,aBoard);
    }
  }
}
//
/***11 枝刈りをするので引数を追加する*********************/
//void backTrackR1(int size,int mask, int row,int left,int down,int right,unsigned int* aBoard)
void backTrackR1(int size,int mask, int row,int left,int down,int right,int BOUND1,unsigned int* aBoard)
{
 int bitmap=0;
 int bit=0;
 int sizeE=size-1;
 bitmap=(mask&~(left|down|right));
 if(row==sizeE){
    if(bitmap){
      /***11　【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略　*****/   
      //aBoard[row]=(-bitmap&bitmap);
      //int s=symmetryOps(size,aBoard);
      //if(s!=0){
      UNIQUE++;
      TOTAL+=8;
      //}
    }
  }else{
    /***11　【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい　*****/
    // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
    if(row<BOUND1) {
      bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
    while(bitmap){
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
      backTrackR1(size,mask,row+1,(left|bit)<<1, down|bit,(right|bit)>>1,BOUND1,aBoard);
    }
  }
}  
//CPUR 再帰版 ロジックメソッド
void NQueenR(int size,int mask)
{
  int TOPBIT;
  int ENDBIT;
  int LASTMASK;
  int SIDEMASK;
  int BOUND1;
  int BOUND2;
  int bit=0;
  TOPBIT=1<<(size-1);
  unsigned int aBoard[MAX];
  /***09 backtrack1*********************/
  //1行め右端 0
  int col=0;
  aBoard[0]=bit=(1<<col);
  int left=bit<<1;
  int down=bit;
  int right=bit>>1;
  //2行目は右から3列目から左端から2列目まで
  for(int col_j=2;col_j<size-1;col_j++){
    aBoard[1]=bit=(1<<col_j);
    BOUND1=col_j;
    //backTrackR1(size,mask,2,(left|bit)<<1,(down|bit),(right|bit)>>1,aBoard);
    backTrackR1(size,mask,2,(left|bit)<<1,(down|bit),(right|bit)>>1,BOUND1,aBoard);
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  /***09 backtrack2*********************/
  //1行目右から2列目から
  //偶数個は1/2 n=8 なら 1,2,3 奇数個は1/2+1 n=9 なら 1,2,3,4
  for(int col=1,col2=size-2;col<col2;col++,col2--){
      aBoard[0]=bit=(1<<col);
      BOUND1=col;
      BOUND2=col2;
      /***11 枝刈りするので引数を渡す*********************/
      //backTrackR2(size,mask,1,bit<<1,bit,bit>>1,aBoard);
      /***12 symmetryOps 省力化のためTOPBIT,ENDBITを渡す*****/ 
      backTrackR2(size,mask,1,bit<<1,bit,bit>>1,BOUND1,BOUND2,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,aBoard);
      LASTMASK|=LASTMASK>>1|LASTMASK<<1;
      ENDBIT>>=1;
  }
}
//
//通常版
/***12 symmetryOps 省力化のためTOPBIT,ENDBITを渡す*****/ 
void backTrack2D_NR(int size,int mask,int row,int left,int down,int right,int BOUND1,int BOUND2,int SIDEMASK,int LASTMASK,int TOPBIT,int ENDBIT,unsigned int* aBoard){
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
      /***11 【枝刈り】 最下段枝刈り*****/ 
      if((bitmap&LASTMASK)==0){
        aBoard[row]=bitmap;
        /***12 symmetryOps 省力化のためTOPBIT,ENDBITを渡す*****/ 
        int s=symmetryOps(size,aBoard,BOUND1,BOUND2,TOPBIT,ENDBIT);
        if(s!=0){
          UNIQUE++;
          TOTAL+=s;
        }
      }
    }
  }else{
    /***11 【枝刈り】 上部サイド枝刈り*****/ 
    if(row<BOUND1){
      bitmap&=~SIDEMASK;
    /***11 【枝刈り】 下部サイド枝刈り*****/ 
    }else if(row==BOUND2){
      if(!(down&SIDEMASK))
        goto b1volta;
      if((down&SIDEMASK)!=SIDEMASK)
        bitmap&=SIDEMASK;
    }
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
      //Backtrack2(y+1, (left | bit)<<1, down | bit, (right | bit)>>1);
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
/***11 枝刈りする*****/ 
void backTrack1D_NR(int size,int mask,int row,int left,int down,int right,int BOUND1,unsigned int* aBoard){
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
      /***11　【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略　*****/ 
      //aBoard[row]=bitmap;
      //int s=symmetryOps(size,aBoard);
      //if(s!=0){
      UNIQUE++;
      TOTAL+=8;
      //}
    }
  }else{
    /***11　【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい　*****/
    // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
    if(row<BOUND1) {
      bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
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
// 
//通常版 CPU 非再帰版 ロジックメソッド
/***09 backTrack登場メソッド名だけ枝刈りはまだしない*****/  
void NQueenD(int size,int mask){
  int TOPBIT;
  int ENDBIT;
  int LASTMASK;
  int SIDEMASK;
  int BOUND1;
  int BOUND2;
  int bit;
  unsigned int aBoard[MAX];
  TOPBIT=1<<(size-1);
  aBoard[0]=1;
  for(BOUND1=2;BOUND1<size-1;BOUND1++){
    aBoard[1]=bit=(1<<BOUND1);
    //backTrack1(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
    backTrack1D_NR(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1),BOUND1,aBoard);
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    //backTrack1(size,mask,1,bit<<1,bit,bit>>1);
    /***12 symmetryOps 省力化のためTOPBIT,ENDBITを渡す*****/ 
    backTrack2D_NR(size,mask,1,bit<<1,bit,bit>>1,BOUND1,BOUND2,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,aBoard);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  }
}
//
// 通常版
/***12 symmetryOps 省力化のためTOPBIT,ENDBITを渡す*****/ 
void backTrack2D(int size,int mask,int row,int left,int down,int right,int BOUND1,int BOUND2,int SIDEMASK,int LASTMASK,int TOPBIT,int ENDBIT,unsigned int* aBoard){
  int bit;
  int bitmap=(mask&~(left|down|right));
  /***11 枝刈り*****/
  //if(row==size){
  if(row==size-1){
    if(bitmap){
    /***11 【枝刈り】 最下段枝刈り*****/
      if((bitmap&LASTMASK)==0){ 	
        aBoard[row]=bitmap; //symmetryOpsの時は代入します。
        /***12 symmetryOps 省力化のためTOPBIT,ENDBITを渡す*****/ 
        int s=symmetryOps(size,aBoard,BOUND1,BOUND2,TOPBIT,ENDBIT);
        if(s!=0){
          UNIQUE++;
          TOTAL+=s;
        }
      }
    }else{
      /***11 【枝刈り】上部サイド枝刈り*****/
      if(row<BOUND1){             	
        bitmap&=~SIDEMASK;
      /***11 【枝刈り】下部サイド枝刈り*****/
      }else if(row==BOUND2) {     	
        if((down&SIDEMASK)==0){ return; }
        if((down&SIDEMASK)!=SIDEMASK){ bitmap&=SIDEMASK; }
      }
      while(bitmap){
        bitmap^=aBoard[row]=bit=(-bitmap&bitmap); //ロジック用
        backTrack2D(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1,BOUND1,BOUND2,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,aBoard);
      }
    }
  }
}
//
void backTrack1D(int size,int mask,int row,int left,int down,int right,int BOUND1,unsigned int* aBoard){
  int bit;
  int bitmap=(mask&~(left|down|right));
  /***11 【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略*****/
  //if(row==size){
  if(row==size-1){    
    //aBoard[row]=bitmap; //symmetryOpsの時は代入します。
    //int s=symmetryOps(size,aBoard);
    //if(s!=0){
    if(bitmap){
      UNIQUE++;
      TOTAL+=8;
    }
  }else{
    /***11 【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい*****/
    // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
    if(row<BOUND1) {
      bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
    while(bitmap){
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap); //ロジック用
      backTrack1D(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1,BOUND1,aBoard);
    }
  }
}
// 
//通常版 CPUR 再帰版　ロジックメソッド
void NQueenDR(int size,int mask)
{
  int TOPBIT;
  int ENDBIT;
  int LASTMASK;
  int SIDEMASK;
  int BOUND1;
  int BOUND2;
  int bit;
  unsigned int aBoard[MAX]; 
  TOPBIT=1<<(size-1);
  aBoard[0]=1;
  for(BOUND1=2;BOUND1<size-1;BOUND1++){
    aBoard[1]=bit=(1<<BOUND1);
    backTrack1D(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1),BOUND1,aBoard);
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    /***12 symmetryOps 省力化のためTOPBIT,ENDBITを渡す*****/
    backTrack2D(size,mask,1,bit<<1,bit,bit>>1,BOUND1,BOUND2,SIDEMASK,LASTMASK,TOPBIT,ENDBIT,aBoard);
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
    printf("\n\n１２．CPU 非再帰 枝刈り\n");
  }else if(cpur){
    printf("\n\n１２．CPUR 再帰 枝刈り\n");
  }else if(gpu){
    printf("\n\n１２．GPU 非再帰 枝刈り\n");
  }else if(sgpu){
    printf("\n\n１２．SGPU 非再帰 バックトラック＋ビットマップ\n");
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
