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

１２．対称解除法の最適化


bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC12.c && ./a.out -r
１２．CPUR 再帰 対称解除法の最適化
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.00
12:        14200            1787            0.00
13:        73712            9233            0.01
14:       365596           45752            0.07
15:      2279184          285053            0.40
16:     14772512         1846955            2.61
17:     95815104        11977939           18.05


bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC12.c && ./a.out -c
１２．CPU 非再帰 対称解除法の最適化
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.00
12:        14200            1787            0.00
13:        73712            9233            0.01
14:       365596           45752            0.06
15:      2279184          285053            0.34
16:     14772512         1846955            2.24
17:     95815104        11977939           15.72

bash-3.2$ nvcc -O3 CUDA12_4_N-Queen.cu && ./a.out -g
１２．GPU 非再帰 枝刈り
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
22:2691008701644    336376244042  00:05:36:20.53
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define THREAD_NUM	90
#define MAX 27
//変数宣言
long TOTAL=0; //GPU,CPUで使用
long UNIQUE=0;//GPU,CPUで使用

__device__  __host__
int symmetryOps(int si,unsigned int *d_aBoard,int BOUND1,int BOUND2,int TOPBIT,int ENDBIT){
      int own,ptn,you,bit;
  //90度回転
  if(d_aBoard[BOUND2]==1){ own=1; ptn=2;
    while(own<=si-1){ bit=1; you=si-1;
      while((d_aBoard[you]!=ptn)&&(d_aBoard[own]>=bit)){ bit<<=1; you--; }
      if(d_aBoard[own]>bit){ return 0; } if(d_aBoard[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>si-1){ return 2; }
  }
  //180度回転
  if(d_aBoard[si-1]==ENDBIT){ own=1; you=si-1-1;
    while(own<=si-1){ bit=1; ptn=TOPBIT;
      while((d_aBoard[you]!=ptn)&&(d_aBoard[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(d_aBoard[own]>bit){ return 0; } if(d_aBoard[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>si-1){ return 4; }
  }
  //270度回転
  if(d_aBoard[BOUND1]==TOPBIT){ own=1; ptn=TOPBIT>>1;
    while(own<=si-1){ bit=1; you=0;
      while((d_aBoard[you]!=ptn)&&(d_aBoard[own]>=bit)){ bit<<=1; you++; }
      if(d_aBoard[own]>bit){ return 0; } if(d_aBoard[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  return 8; 
}
//
//
__device__
void cuda_kernel_b1(unsigned int down[][19],unsigned int left[][19],unsigned int right[][19],unsigned int bitmap[][19],register unsigned int tid,register int size,register unsigned int mask,register int row,unsigned int B1,unsigned int* sum,unsigned int* usum,register int N){
    register unsigned int bit;
    register unsigned int bitmap_tid_row;
    register unsigned int down_tid_row;
    register unsigned int left_tid_row;
    register unsigned int right_tid_row;
    register unsigned int total=0;
    register unsigned int unique=0;
    while(row>=N){
      bitmap_tid_row=bitmap[tid][row];
      down_tid_row=down[tid][row];
      left_tid_row=left[tid][row];
      right_tid_row=right[tid][row];
      if(bitmap_tid_row==0){
        row--;
      }else{
        /**11 枝刈り**********/
        if(row<B1) {
          bitmap_tid_row=bitmap[tid][row]&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
        }  
        //クイーンを置く
        //置く場所があるかどうか
        bitmap[tid][row]
          ^=bit
          =(-bitmap_tid_row&bitmap_tid_row);       
        if((bit&mask)!=0){
          //最終行?最終行から１個前の行まで
          //無事到達したら 加算する
          if(row==size-1){
           /**11 backTradk1ではsymmetryOps不要のためコメント*********************/
           //int s=symmetryOps(size,c_aBoard); 
           //if(s!=0){
           //print(size); //print()でTOTALを++しない
           //ホストに戻す配列にTOTALを入れる
           //スレッドが１つの場合は配列は１個
            unique++; 
            total+=8;   //対称解除で得られた解数を加算
            //printf("total:%d\n",total);
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
    sum[tid]=total;
    usum[tid]=unique;
}
__device__
void cuda_kernel_b2(unsigned int down[][19],unsigned int left[][19],unsigned int right[][19],unsigned int bitmap[][19],unsigned int* c_aBoard,register unsigned int tid,register int size,register unsigned int mask,register int row,unsigned int B1,unsigned int B2,unsigned int SM,unsigned  int LM,register int TB,unsigned  int EB,unsigned int* sum,unsigned int* usum,register int N){
    register unsigned int bit;
    register unsigned int bitmap_tid_row;
    register unsigned int down_tid_row;
    register unsigned int left_tid_row;
    register unsigned int right_tid_row;
    register unsigned int total=0;
    register unsigned int unique=0;
    while(row>=N){
      bitmap_tid_row=bitmap[tid][row];
      down_tid_row=down[tid][row];
      left_tid_row=left[tid][row];
      right_tid_row=right[tid][row];
      if(bitmap_tid_row==0){
        row--;
      }else{
        /**11 枝刈り追加**********/
        //【枝刈り】上部サイド枝刈り
	      if(row<B1){             	
        //printf("BOUND1_row:%d:row:%d bit:%d\n",row,bitmap[tid][row]);
          bitmap_tid_row=bitmap[tid][row]&=~SM;
        //【枝刈り】下部サイド枝刈り
        }else if(row==B2) {     	
        //printf("BOUND2_row:%d:bit:%d\n",row,bitmap[tid][row]);
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
          ^=c_aBoard[row]
          =bit
          =(-bitmap_tid_row&bitmap_tid_row);       
        if((bit&mask)!=0){
          //最終行?最終行から１個前の行まで
          //無事到達したら 加算する
          if(row==size-1){
            
            /***11 LASTMASK枝刈り*********************/ 
            if((save_bitmap&LM)==0){ 
              /***12 symmetryOps 省力化のためBOUND1,BOUND2,TOPBIT,ENDBITを渡す*****/ 
              //for(int i=0;i<size;i++){
              //  printf("i:%d aBoard;%d\n",i,c_aBoard[i]);
              //}
              int s=symmetryOps(size,c_aBoard,B1,B2,TB,EB); 
              //printf("total:%d\n",s);
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
    sum[tid]=total;
    usum[tid]=unique;
}
//
__global__
void cuda_kernel_b(
    register int size,
    unsigned int* totalDown,
    unsigned int* totalLeft,
    unsigned int* totalRight,
    unsigned int* d_results,
    unsigned int* d_uniq,
    register int totalCond,
    unsigned int* t_aBoard,
    unsigned  int* BOUND1,
    unsigned  int* BOUND2,
    unsigned  int* SIDEMASK,
    unsigned  int* LASTMASK,
    /***12 symmetryOps 省力化のためTOPBIT,ENDBITを渡す*****/ 
    register int TB,
    unsigned  int* ENDBIT,
    register int N
    )
{
  register const unsigned int mask=(1<<size)-1;
  
  
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
  register int row=N;
  //シェアードメモリ
  __shared__ unsigned int down[THREAD_NUM][19];
  __shared__ unsigned int left[THREAD_NUM][19];
  __shared__ unsigned int right[THREAD_NUM][19];
 __shared__ unsigned int bitmap[THREAD_NUM][19];
  //
  //sharedメモリを使う ブロック内スレッドで共有
  //10固定なのは現在のmask設定で
  //GPUで実行するのは最大10だから
  
  unsigned int B1=BOUND1[idx];
  unsigned int B2=BOUND2[idx];
  unsigned int SM=SIDEMASK[idx];
  unsigned int LM=LASTMASK[idx];
  unsigned int EB=ENDBIT[idx];
  
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
    for(int i=0;i<N;i++){
      c_aBoard[i]=t_aBoard[idx*N+i]; //２次元配列だが1次元的に利用  
    }
    down[tid][row]=totalDown[idx];
    left[tid][row]=totalLeft[idx];
    right[tid][row]=totalRight[idx];
    //down,left,rightからbitmapを出す
    bitmap[tid][row]
        =mask&~(
         down[tid][row]
        |left[tid][row]
        |right[tid][row]);
    if(c_aBoard[0]==1){
      //THREAD_NUMはブロックあたりのスレッド数  
      cuda_kernel_b1(down,left,right,bitmap,tid,size,mask,row,B1,sum,usum,N);
    }else{
      //THREAD_NUMはブロックあたりのスレッド数
      cuda_kernel_b2(down,left,right,bitmap,c_aBoard,tid,size,mask,row,B1,B2,SM,LM,TB,EB,sum,usum,N);
    }

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
//CPU側の処理 N階層分backtrack1,backtrack2を実行してtotal...配列にそれまでの実行結果を格納し,最後に1回CUDAを呼び出す
void NQueenG(int size,int steps)
{
  //Nで階層数を設定(少なくとも1階層はGPUで実行するようにプログラムを組んでる)
  int N=size>4?4:3;
  //CPU側で使用する変数
  int row=0;
  int TOPBIT;
  int ENDBIT;
  int LASTMASK;
  int SIDEMASK;
  int BOUND1;
  int BOUND2;
  //CPU側で使用する変数だがCPUで複数階層実行する場合、階層の上り下りがあるので配列で持つ必要がある
  unsigned int down[MAX];
  unsigned int right[MAX];
  unsigned int left[MAX];
  //bitmapを配列で持つことにより
  //stackを使わないで1行前に戻れる
  unsigned int bitmap[MAX];
  unsigned int aBoard[MAX];
  
  int totalCond=0;//GPU側へ渡す数
  //totalDown,totalLeft,totalRight,totalBOUND1,totalBOUND2,totalSIDEMASK,totalLASTMASK,totalENDBIT
  //CPU側でN階層まで実行した結果を配列に詰め込む
  unsigned int* totalDown;
  cudaMallocHost((void**) &totalDown,sizeof(int)*steps);
  unsigned int* totalLeft;
  cudaMallocHost((void**) &totalLeft,sizeof(int)*steps);
  unsigned int* totalRight;
  cudaMallocHost((void**) &totalRight,sizeof(int)*steps);
  unsigned int* totalBOUND1;
  cudaMallocHost((void**) &totalBOUND1,sizeof(int)*steps);
  unsigned int* totalBOUND2;
  cudaMallocHost((void**) &totalBOUND2,sizeof(int)*steps);
  unsigned int* totalSIDEMASK;
  cudaMallocHost((void**) &totalSIDEMASK,sizeof(int)*steps);
  unsigned int* totalLASTMASK;
  cudaMallocHost((void**) &totalLASTMASK,sizeof(int)*steps);
  unsigned int* totalENDBIT;
  cudaMallocHost((void**) &totalENDBIT,sizeof(int)*steps);
  unsigned int* h_results;
  cudaMallocHost((void**) &h_results,sizeof(int)*steps);
  unsigned int* h_uniq;
  cudaMallocHost((void**) &h_uniq,sizeof(int)*steps);
  
  //device
  unsigned int* downCuda;
  cudaMalloc((void**) &downCuda,sizeof(int)*steps);
  unsigned int* leftCuda;
  cudaMalloc((void**) &leftCuda,sizeof(int)*steps);
  unsigned int* rightCuda;
  cudaMalloc((void**) &rightCuda,sizeof(int)*steps);
  unsigned int* BOUND1Cuda;
  cudaMalloc((void**) &BOUND1Cuda,sizeof(int)*steps);
  unsigned int* BOUND2Cuda;
  cudaMalloc((void**) &BOUND2Cuda,sizeof(int)*steps);
  unsigned int* SIDEMASKCuda;
  cudaMalloc((void**) &SIDEMASKCuda,sizeof(int)*steps);
  unsigned int* LASTMASKCuda;
  cudaMalloc((void**) &LASTMASKCuda,sizeof(int)*steps);
  unsigned int* ENDBITCuda;
  cudaMalloc((void**) &ENDBITCuda,sizeof(int)*steps);
  unsigned int* resultsCuda;
  cudaMalloc((void**) &resultsCuda,sizeof(int)*steps/THREAD_NUM);
  unsigned int* d_uniq;
  cudaMalloc((void**) &d_uniq,sizeof(int)*steps/THREAD_NUM);
  unsigned int* t_aBoard;
  cudaMallocHost((void**) &t_aBoard,sizeof(int)*steps*N);
  unsigned int* d_aBoard;
  cudaMalloc((void**) &d_aBoard,sizeof(int)*steps*N);
  
  register int bit=0;
  register int mask=((1<<size)-1);
  TOPBIT=1<<(size-1);
  if(size<=0||size>MAX){return;}
  /***backtrack1の処理*********************/
  //1行め右端 0
  int col=0;
  aBoard[0]=bit=(1<<col);
  left[1]=bit<<1;
  down[1]=bit;
  right[1]=bit>>1;
 
  //2行目は右から3列目から左端から2列目まで
  for(int col_j=2;col_j<size-1;col_j++){
      aBoard[1]=bit=(1<<col_j);
      /***11 BOUND1*********************/
      BOUND1=col_j;
      left[2]=(left[1]|bit)<<1;
      down[2]=(down[1]|bit);
      right[2]=(right[1]|bit)>>1;
      bitmap[2]=mask&~(left[2]|down[2]|right[2]);
      register int rowP=0;
      row=2;
      while(row>=2) {
        if(bitmap[row]==0){ row--; }
        else{//おける場所があれば進む
          if(row<BOUND1) {
            bitmap[row]&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
          }
          bitmap[row]^=aBoard[row]=bit=(-bitmap[row]&bitmap[row]);
          if((bit&mask)!=0){//置く場所があれば先に進む
            rowP=row+1;
            down[rowP]=down[row]|bit;
            left[rowP]=(left[row]|bit)<<1;
            right[rowP]=(right[row]|bit)>>1;
            bitmap[rowP]=mask&~(down[rowP]|left[rowP]|right[rowP]);
            row++;
            if(row==N){
              totalDown[totalCond]=down[row];
              totalLeft[totalCond]=left[row];
              totalRight[totalCond]=right[row];
              for(int i=0;i<N;i++){
                t_aBoard[totalCond*N+i]=aBoard[i];
              }
              totalBOUND1[totalCond]=BOUND1;        
              totalCond++;
              row--;
            }
        }else{  
          row--;
        }  
      }
    }   
  } 
  /***09 backtrack2*********************/
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  for(int col=1,col2=size-2;col<col2;col++,col2--){
      aBoard[0]=bit=(1<<col);
      BOUND1=col;
      BOUND2=col2;
      left[1]=bit<<1;
      down[1]=bit;
      right[1]=bit>>1;
      bitmap[1]=mask&~(left[1]|down[1]|right[1]);
      register int rowP=0;
      row=1;
      while(row>=1) {
        if(bitmap[row]==0){ row--; }
        else{
          if(row<BOUND1){             	
	          bitmap[row]&=~SIDEMASK;
          }else if(row==BOUND2) {     	
            if((down[row]&SIDEMASK)==0){ row--; }
            if((down[row]&SIDEMASK)!=SIDEMASK){ bitmap[row]&=SIDEMASK; }
          }
          bitmap[row]^=aBoard[row]=bit=(-bitmap[row]&bitmap[row]);
          if((bit&mask)!=0){//置く場所があれば先に進む
            rowP=row+1;
            down[rowP]=down[row]|bit;
            left[rowP]=(left[row]|bit)<<1;
            right[rowP]=(right[row]|bit)>>1;
            bitmap[rowP]=mask&~(down[rowP]|left[rowP]|right[rowP]);
            row++;
            if(row==N){
              totalDown[totalCond]=down[row];
              totalLeft[totalCond]=left[row];
              totalRight[totalCond]=right[row];
              for(int i=0;i<N;i++){
                t_aBoard[totalCond*N+i]=aBoard[i];
              }
              totalBOUND1[totalCond]=BOUND1;
              totalBOUND2[totalCond]=BOUND2;
              totalSIDEMASK[totalCond]=SIDEMASK;
              totalLASTMASK[totalCond]=LASTMASK;
              totalENDBIT[totalCond]=ENDBIT;
              totalCond++;
              row--;
            }
          }else{
              row--;
          }
        }
  
      }
      LASTMASK|=LASTMASK>>1|LASTMASK<<1;
      ENDBIT>>=1;
  }
  cudaMemcpy(downCuda,totalDown,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(leftCuda,totalLeft,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(rightCuda,totalRight,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(d_aBoard,t_aBoard,
      sizeof(int)*totalCond*N,cudaMemcpyHostToDevice);
  cudaMemcpy(BOUND1Cuda,totalBOUND1,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(BOUND2Cuda,totalBOUND2,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(SIDEMASKCuda,totalSIDEMASK,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(LASTMASKCuda,totalLASTMASK,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(ENDBITCuda,totalENDBIT,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  //cudaを呼び出すのは最後の１回だけ
  cuda_kernel_b<<<steps/THREAD_NUM,THREAD_NUM
              >>>(size,downCuda,leftCuda,rightCuda,resultsCuda,d_uniq,totalCond,d_aBoard,BOUND1Cuda,BOUND2Cuda,SIDEMASKCuda,LASTMASKCuda,TOPBIT,ENDBITCuda,N);
  //printf("totalCond;%d\n",totalCond);  
  cudaMemcpy(h_results,resultsCuda,
      sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_uniq,d_uniq,
      sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  for(int col=0;col<steps/THREAD_NUM;col++){
    TOTAL+=h_results[col];
    UNIQUE+=h_uniq[col];
  }
  cudaFree(downCuda);
  cudaFree(leftCuda);
  cudaFree(rightCuda);
  cudaFree(BOUND1Cuda);
  cudaFree(BOUND2Cuda);
  cudaFree(SIDEMASKCuda);
  cudaFree(LASTMASKCuda);
  cudaFree(resultsCuda);
  cudaFree(d_uniq);
  /***11 aBoardコメント**/
  //cudaFree(d_aBoard);
  cudaFreeHost(totalDown);
  cudaFreeHost(totalLeft);
  cudaFreeHost(totalRight);
  cudaFreeHost(totalBOUND1);
  cudaFreeHost(totalBOUND2);
  cudaFreeHost(totalSIDEMASK);
  cudaFreeHost(totalLASTMASK);
  cudaFreeHost(h_results);
  cudaFreeHost(h_uniq);
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
              //for(int i=0;i<size;i++){
              //  printf("i:%d aBoard;%d",i,aBoard[i]);
              //}  
              //printf("\n");
              /***12 symmetryOps 省力化のためBOUND1,BOUND2,TOPBIT,ENDBITを渡す*****/ 
              int s=symmetryOps(size,aBoard,BOUND1,BOUND2,TOPBIT,ENDBIT);
              printf("s:%d\n",s);
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
            printf("i:%d\n",row);
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
      printf("B1:%d:B2:%d:TOP:%d:END:%d:SIDE:%d:LAST:%d\n",BOUND1,BOUND2,TOPBIT,ENDBIT,SIDEMASK,LASTMASK);
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
    //int min=4; int targetN=15;
    int min=4; int targetN=15;

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
    //steps=8192;
    //steps=24576;
    steps=12288;
    //steps=196608;
    if(!InitCUDA()){return 0;}
    //int min=4;int targetN=24;
    //int min=4;int targetN=17;
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

