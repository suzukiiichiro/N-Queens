/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイル
 $ nvcc CUDA04_N-Queen.cu -o CUDA04_N-Queen

 実行
 $ ./CUDA04_N-Queen (-c|-r|-g)
                    -c:cpu -r cpu再帰 -g GPU

 ４．バックトラック＋対称解除法

 　一つの解には、盤面を９０度、１８０度、２７０度回転、及びそれらの鏡像の合計
 　８個の対称解が存在する。対照的な解を除去し、ユニーク解から解を求める手法。
 
 ■ユニーク解の判定方法
   全探索によって得られたある１つの解が、回転・反転などによる本質的に変わること
 のない変換によって他の解と同型となるものが存在する場合、それを別の解とはしない
 とする解の数え方で得られる解を「ユニーク解」といいます。つまり、ユニーク解とは、
 全解の中から回転・反転などによる変換によって同型になるもの同士をグループ化する
 ことを意味しています。
 
   従って、ユニーク解はその「個数のみ」に着目され、この解はユニーク解であり、こ
 の解はユニーク解ではないという定まった判定方法はありません。ユニーク解であるか
 どうかの判断はユニーク解の個数を数える目的の為だけに各個人が自由に定義すること
 になります。もちろん、どのような定義をしたとしてもユニーク解の個数それ自体は変
 わりません。
 
   さて、Ｎクイーン問題は正方形のボードで形成されるので回転・反転による変換パター
 ンはぜんぶで８通りあります。だからといって「全解数＝ユニーク解数×８」と単純には
 いきません。ひとつのグループの要素数が必ず８個あるとは限らないのです。Ｎ＝５の
 下の例では要素数が２個のものと８個のものがあります。


 Ｎ＝５の全解は１０、ユニーク解は２なのです。
 
 グループ１: ユニーク解１つ目
 - - - Q -   - Q - - -
 Q - - - -   - - - - Q
 - - Q - -   - - Q - -
 - - - - Q   Q - - - -
 - Q - - -   - - - Q -
 
 グループ２: ユニーク解２つ目
 - - - - Q   Q - - - -   - - Q - -   - - Q - -   - - - Q -   - Q - - -   Q - - - -   - - - - Q
 - - Q - -   - - Q - -   Q - - - -   - - - - Q   - Q - - -   - - - Q -   - - - Q -   - Q - - -
 Q - - - -   - - - - Q   - - - Q -   - Q - - -   - - - - Q   Q - - - -   - Q - - -   - - - Q -
 - - - Q -   - Q - - -   - Q - - -   - - - Q -   - - Q - -   - - Q - -   - - - - Q   Q - - - -
 - Q - - -   - - - Q -   - - - - Q   Q - - - -   Q - - - -   - - - - Q   - - Q - -   - - Q - -

 
   それでは、ユニーク解を判定するための定義付けを行いますが、次のように定義する
 ことにします。各行のクイーンが右から何番目にあるかを調べて、最上段の行から下
 の行へ順番に列挙します。そしてそれをＮ桁の数値として見た場合に最小値になるもの
 をユニーク解として数えることにします。尚、このＮ桁の数を以後は「ユニーク判定値」
 と呼ぶことにします。
 
 - - - - Q   0
 - - Q - -   2
 Q - - - -   4   --->  0 2 4 1 3  (ユニーク判定値)
 - - - Q -   1
 - Q - - -   3
 
 
   探索によって得られたある１つの解(オリジナル)がユニーク解であるかどうかを判定
 するには「８通りの変換を試み、その中でオリジナルのユニーク判定値が最小であるか
 を調べる」ことになります。しかし結論から先にいえば、ユニーク解とは成り得ないこ
 とが明確なパターンを探索中に切り捨てるある枝刈りを組み込むことにより、３通りの
 変換を試みるだけでユニーク解の判定が可能になります。
  
 
 ■ユニーク解の個数を求める
   先ず最上段の行のクイーンの位置に着目します。その位置が左半分の領域にあればユ
 ニーク解には成り得ません。何故なら左右反転によって得られるパターンのユニーク判
 定値の方が確実に小さくなるからです。また、Ｎが奇数の場合に中央にあった場合はど
 うでしょう。これもユニーク解には成り得ません。何故なら仮に中央にあった場合、そ
 れがユニーク解であるためには少なくとも他の外側の３辺におけるクイーンの位置も中
 央になければならず、それは互いの効き筋にあたるので有り得ません。

  ***********************************************************************
  最上段の行のクイーンの位置は中央を除く右側の領域に限定されます。(ただし、N ≧ 2)
  ***********************************************************************
  
    次にその中でも一番右端(右上の角)にクイーンがある場合を考えてみます。他の３つ
  の角にクイーンを置くことはできないので(効き筋だから）、ユニーク解であるかどうか
  を判定するには、右上角から左下角を通る斜軸で反転させたパターンとの比較だけになり
  ます。突き詰めれば、
  
  [上から２行目のクイーンの位置が右から何番目にあるか]
  [右から２列目のクイーンの位置が上から何番目にあるか]
  
 
  を比較するだけで判定することができます。この２つの値が同じになることはないからです。
  
        3 0
        ↓↓
  - - - - Q ←0
  - Q - - - ←3
  - - - - -         上から２行目のクイーンの位置が右から４番目にある。
  - - - Q -         右から２列目のクイーンの位置が上から４番目にある。
  - - - - -         しかし、互いの効き筋にあたるのでこれは有り得ない。
  
    結局、再帰探索中において下図の X への配置を禁止する枝刈りを入れておけば、得
  られる解は総てユニーク解であることが保証されます。
  
  - - - - X Q
  - Q - - X -
  - - - - X -
  - - - - X -
  - - - - - -
  - - - - - -
  
    次に右端以外にクイーンがある場合を考えてみます。オリジナルがユニーク解である
  ためには先ず下図の X への配置は禁止されます。よって、その枝刈りを先ず入れておき
  ます。
  
  X X - - - Q X X
  X - - - - - - X
  - - - - - - - -
  - - - - - - - -
  - - - - - - - -
  - - - - - - - -
  X - - - - - - X
  X X - - - - X X
  
    次にクイーンの利き筋を辿っていくと、結局、オリジナルがユニーク解ではない可能
  性があるのは、下図の A,B,C の位置のどこかにクイーンがある場合に限られます。従っ
  て、90度回転、180度回転、270度回転の３通りの変換パターンだけを調べれはよいこと
  になります。
  
  X X x x x Q X X
  X - - - x x x X
  C - - x - x - x
  - - x - - x - -
  - x - - - x - -
  x - - - - x - A
  X - - - - x - X
  X X B - - x X X
 
 
  ■ユニーク解から全解への展開
    これまでの考察はユニーク解の個数を求めるためのものでした。全解数を求めるには
  ユニーク解を求めるための枝刈りを取り除いて全探索する必要があります。したがって
  探索時間を犠牲にしてしまうことになります。そこで「ユニーク解の個数から全解数を
  導いてしまおう」という試みが考えられます。これは、左右反転によるパターンの探索
  を省略して最後に結果を２倍するというアイデアの拡張版といえるものです。そしてそ
  れを実現させるには「あるユニーク解が属するグループの要素数はいくつあるのか」と
  いう考察が必要になってきます。
  
    最初に、クイーンが右上角にあるユニーク解を考えます。斜軸で反転したパターンが
  オリジナルと同型になることは有り得ないことと(×２)、右上角のクイーンを他の３つの
  角に写像させることができるので(×４)、このユニーク解が属するグループの要素数は必
  ず８個(＝２×４)になります。
  
    次に、クイーンが右上角以外にある場合は少し複雑になりますが、考察を簡潔にする
  ために次の事柄を確認します。
 
  TOTAL = (COUNT8 * 8) + (COUNT4 * 4) + (COUNT2 * 2);
    (1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルか
     ら180度回転)させても、さらに90度回転(オリジナルから270度回転)させてもオリ
     ジナルと同型になる。  
 
     COUNT2 * 2
  
    (2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナ
     ルとは異なる。ただし、180度回転させた場合はオリジナルと同型になることも有
     り得る。 
 
     COUNT4 * 4
  
    (3) (1) に該当するユニーク解が属するグループの要素数は、左右反転させたパターンを
        加えて２個しかありません。(2)に該当するユニーク解が属するグループの要素数は、
        180度回転させて同型になる場合は４個(左右反転×縦横回転)、そして180度回転させても
        オリジナルと異なる場合は８個になります。(左右反転×縦横回転×上下反転)
  
     COUNT8 * 8 
 
    以上のことから、ひとつひとつのユニーク解が上のどの種類に該当するのかを調べる
  ことにより全解数を計算で導き出すことができます。探索時間を短縮させてくれる枝刈
  りを外す必要がなくなったというわけです。 
  
    UNIQUE  COUNT2      +  COUNT4      +  COUNT8
    TOTAL  (COUNT2 * 2) + (COUNT4 * 4) + (COUNT8 * 8)
 
  　これらを実現すると、前回のNQueen3()よりも実行速度が遅くなります。
  　なぜなら、対称・反転・斜軸を反転するための処理が加わっているからです。
  ですが、今回の処理を行うことによって、さらにNQueen5()では、処理スピードが飛
  躍的に高速化されます。そのためにも今回のアルゴリズム実装は必要なのです。




実行結果

４．CPUR 再帰 バックトラック＋対称解除法
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.02
12:        14200            1787            0.09
13:        73712            9233            0.49
14:       365596           45752            2.98
15:      2279184          285053           19.17
16:     14772512         1846955         2:11.46
17:     95815104        11977939        15:46.88

４．CPU 非再帰 バックトラック＋対称解除法
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.02
12:        14200            1787            0.09
13:        73712            9233            0.50
14:       365596           45752            2.99
15:      2279184          285053           19.30
16:     14772512         1846955         2:12.39
17:     95815104        11977939        15:51.69

４．GPU 非再帰 バックトラック＋対称解除法

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
long Total=0 ;        //合計解
long Unique=0;
int down[2*MAX-1]; //down:flagA 縦 配置フラグ　
int left[2*MAX-1];  //left:flagB 斜め配置フラグ　
int right[2*MAX-1];  //right:flagC 斜め配置フラグ　
long TOTAL=0;
long UNIQUE=0;
int aBoard[MAX];
int aT[MAX];       //aT:aTrial[]
int aS[MAX];       //aS:aScrath[]
//関数宣言
void NQueen(int row,int size);
void NQueenR(int row,int size);
int symmetryOps(int si);
void rotate(int chk[],int scr[],int n,int neg);
void vMirror(int chk[],int n);
int intncmp(int lt[],int rt[],int n);
void TimeFormat(clock_t utime,char *form);
void print(int size);
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
//出力
void print(int size){
	printf("%ld: ",TOTAL);
	for(int j=0;j<size;j++){
		printf("%d ",aBoard[j]);
	}
	printf("\n");
}
//回転
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
void vMirror(int chk[],int n){
	for(int j=0;j<n;j++){
		chk[j]=(n-1)-chk[j];
	}
}
//
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
			}  
      //-270度回転 反対角鏡と同等
			rotate(aT,aS,size,1);
			k=intncmp(aBoard,aT,size);
			if(k>0){
				return 0;
			}
		}
	}
	return nEquiv*2;
}
// CPU 非再帰版 ロジックメソッド
void NQueen(int row,int size){
	int sizeE=size-1;
	bool matched;
	while(row>=0){
		matched=false;
		// １回目はaBoard[row]が-1なのでcolを0で初期化
		// ２回目以降はcolを<sizeまで右へシフト
		for(int col=aBoard[row]+1;col<size;col++){
			if(down[col]==0
					&& right[col-row+sizeE]==0
					&& left[col+row]==0){ 	//まだ効き筋がない
				if(aBoard[row]!=-1){		//Qを配置済み
					//colがaBoard[row]におきかわる
					down[aBoard[row]]
						=right[aBoard[row]-row+sizeE]
						=left[aBoard[row]+row]=0;
				}
				aBoard[row]=col;				//Qを配置
				down[col]
				  =right[col-row+sizeE]
					=left[col+row]=1;			//効き筋とする
				matched=true;						//配置した
				break;
			}
		}
		if(matched){								//配置済みなら
			row++;										//次のrowへ
			if(row==size){
				//print(size); //print()でTOTALを++しない
				/** 対称解除法の導入 */
        int s=symmetryOps(size);
        if(s!=0){
          UNIQUE++;   //ユニーク解を加算
          TOTAL+=s;   //対称解除で得られた解数を加算
        }
				// TOTAL++;
				row--;
			}
		}else{
			if(aBoard[row]!=-1){
				int col=aBoard[row]; /** col の代用 */
				aBoard[row]=-1;
				down[col]
				  =right[col-row+sizeE]
				  =left[col+row]=0;
			}
			row--;										//バックトラック
		}
	}
}
// CPUR 再帰版 ロジックメソッド
void NQueenR(int row,int size){
	int sizeE=size-1;
	if(row==size){
		/** 対称解除法の導入 */
		int s=symmetryOps(size);
		if(s!=0){
			UNIQUE++;       //ユニーク解を加算
			TOTAL+=s;       //対称解除で得られた解数を加算
		}
		// TOTAL++;
	}else{
		for(int col=0;col<size;col++){
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
		}
	}
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
//メインメソッド
int main(int argc,char** argv) {
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
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  if(cpu){
    printf("\n\n４．CPU 非再帰 バックトラック＋対称解除法\n");
  }else if(cpur){
    printf("\n\n４．CPUR 再帰 バックトラック＋対称解除法\n");
  }else if(gpu){
    printf("\n\n４．GPU 非再帰 バックトラック\n");
  }
  if(cpu||cpur){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
		clock_t st;           //速度計測用
		char t[20];           //hh:mm:ss.msを格納
    int min=4; int targetN=18;
    /* int targetN=MAX; */
    for(int i=min;i<=targetN;i++){
      TOTAL=0; UNIQUE=0;
      st=clock();
      if(cpu){
        //非再帰は-1で初期化
        for(int j=0;j<=targetN;j++){ aBoard[j]=-1; }
        NQueen(0,i);
      }
      if(cpur){
        //再帰は0で初期化
        for(int j=0;j<=targetN;j++){ aBoard[j]=0; } 
        NQueenR(0,i);
      }
      TimeFormat(clock()-st,t); 
      printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
    }
  }
  if(gpu){
    if(!InitCUDA()){return 0;}
    int min=4;int targetN=18;
    struct timeval t0;struct timeval t1;int ss;int ms;int dd;
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
