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
//関数宣言CUDA
__global__ void nqueen_cuda(int *d_aBoard,int *d_results,int *d_count, int COUNT,int row,int size);
void solve_nqueen_cuda(int si,int steps);
bool InitCUDA();
//関数宣言CPU
void print(int size);
void NQueen(int row,int size);
void NQueenR(int row,int size);
//
__global__
void nqueen_cuda(int *d_aBoard,int *d_down,int *d_results,int *d_count,int COUNT,int row,int size){
    bool matched;
  while(row>=0){
    matched=false;
    for(int col=d_aBoard[row]+1;col<size;col++){
      if(d_down[col]==0){      //downは効き筋ではない
        if(d_aBoard[row]!=-1){ //Qは配置済み
          d_down[d_aBoard[row]]=0;//downの効き筋を外す
        }
        d_aBoard[row]=col;     //Qを配置
        d_down[col]=1;         //downは効き筋である
        matched=true;
        break;
      }
    }
    if(matched){
      row++;
      if(row==size){
        //cudaの中で　printせず配列に格納して　hostに返却する
        //ex 0,1,1,3 だったら　3110
        int sum=0;
        for(int j=0;j<size;j++){
          sum+=d_aBoard[j]*pow(10,j);   
        }
        d_results[COUNT++]=sum;
        row--;
      }
    }else{                   //置けるところがない
      if(d_aBoard[row]!=-1){
        int col=d_aBoard[row]; /** colの代用 */
        d_down[col]=0;         //downの効き筋を解除
        d_aBoard[row]=-1;      //空き地に戻す
      }
      row--;
    }
  }
	d_count[0]=COUNT;//カウントを代入
}
//
void solve_nqueen_cuda(int si,int steps){
    //メモリ登録
    int *h_aBoard;
    int *h_down;
    int *h_results;
    int *h_count;
    cudaMallocHost((void**)&h_aBoard,sizeof(int)*MAX);
    cudaMallocHost((void**)&h_down,sizeof(int)*2*MAX-1);
    cudaMallocHost((void**)&h_results,sizeof(int)*steps);
    cudaMallocHost((void**)&h_count,sizeof(int));
    int *d_aBoard;
    int *d_down;
    int *d_results;
    int *d_count;
    cudaMalloc((void**)&d_aBoard,sizeof(int)*MAX);
    cudaMalloc((void**)&d_down,sizeof(int)*2*MAX-1);
    cudaMalloc((void**)&d_results,sizeof(int)*steps);
    cudaMalloc((void**)&d_count,sizeof(int));
    //初期化
    for(int i=0;i<si;i++){
        h_aBoard[i]=-1;
    }
    //カウンターを初期化
    h_count[0]=0;
    //host to device
    cudaMemcpy(d_aBoard,h_aBoard,
      sizeof(int)*MAX,cudaMemcpyHostToDevice);
    cudaMemcpy(d_down,h_down,
      sizeof(int)*2*MAX-1,cudaMemcpyHostToDevice);
    cudaMemcpy(d_results,h_results,
      sizeof(int)*steps,cudaMemcpyHostToDevice);
    cudaMemcpy(d_count,h_count,
      sizeof(int),cudaMemcpyHostToDevice);
    //実行
    nqueen_cuda<<<1,1>>>(d_aBoard,d_down,d_results,d_count,0,0,si);
    //device to host
    cudaMemcpy(h_results,d_results,
      sizeof(int)*steps,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_count,d_count,
      sizeof(int),cudaMemcpyDeviceToHost);
    //出力
    for(int i=0;i<h_count[0];i++){
      printf("%d:%08d\n",i+1,h_results[i]);  
    }
    //開放
    cudaFreeHost(h_aBoard);
    cudaFreeHost(h_down);
    cudaFreeHost(h_results);
    cudaFreeHost(h_count);
    cudaFree(d_aBoard);
    cudaFree(d_down);
    cudaFree(d_results);
    cudaFree(d_count);
}
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
    else{
      cpur=true;
		}
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g|-r]\n",argv[0]);
    printf("  -c: CPU only\n");
    printf("  -r: CPUR only\n");
    printf("  -g: GPU only\n");
    printf("Default CPUR to 8 queen\n");
  }
  /** 出力と実行 */
  //aBoard配列を-1で初期化
  for(int i=0;i<size;i++){ aBoard[i]=-1; }
  if(cpu){ 
    printf("\n\n２．CPU 非再帰 配置フラグ（制約テスト高速化）\n");
    NQueen(0,size); 
  }
  if(cpur){ 
    printf("\n\n２．CPU 再帰 配置フラグ（制約テスト高速化）\n");
    NQueenR(0,size); 
  }
  if(gpu){
    printf("\n\n２．GPU 非再帰 配置フラグ（制約テスト高速化）\n");
    if(!InitCUDA()){return 0;}
    solve_nqueen_cuda(size,steps);
  }
  return 0;
}
