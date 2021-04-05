
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

$ nvcc CUDA06_N-Queen.cu  && ./a.out -g
６．GPU 非再帰 バックトラック＋ビットマップ
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2               0  00:00:00:00.02
 5:           10               0  00:00:00:00.00
 6:            4               0  00:00:00:00.00
 7:           40               0  00:00:00:00.00
 8:           92               0  00:00:00:00.00
 9:          352               0  00:00:00:00.00
10:          724               0  00:00:00:00.00
11:         2680               0  00:00:00:00.01
12:        14200               0  00:00:00:00.05
13:        73712               0  00:00:00:00.07
14:       365596               0  00:00:00:00.07
15:      2279184               0  00:00:00:00.37
16:     14772512               0  00:00:00:02.30
17:     95815104               0  00:00:00:18.07
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
//関数宣言 GPU
__global__ void cuda_kernel(int size,int mark,unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,unsigned int* d_results,int totalCond);
long long solve_nqueen_cuda(int size,int steps);
void NQueenG(int size,int mask,int row,int steps);
//関数宣言 SGPU
__global__ void sgpu_cuda_kernel(int size,int mark,unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,unsigned int* results,int totalCond);
long long sgpu_solve_nqueen_cuda(int size,int steps); 
//関数宣言 CPU
void TimeFormat(clock_t utime,char *form);
//関数宣言 CPU
void solve_nqueen(int size,int mask, int row,int* left,int* down,int* right,int* bitmap);
void NQueen(int size,int mask);
//関数宣言 CPUR
void solve_nqueenr(int size,int mask, int row,int left,int down,int right);
void NQueenR(int size,int mask,int row,int left,int down,int right);
//関数宣言 通常版
//  非再帰
void NQueenD(int size,int mask,int row);
//  再帰
void NQueenDR(int size,int mask,int row,int left,int down,int right);
//
//GPU
__global__ 
void cuda_kernel(
    int size,
    int mark,
    unsigned int* totalDown,
    unsigned int* totalLeft,
    unsigned int* totalRight,
    unsigned int* d_results,
    int totalCond)
{
  register const unsigned int mask=(1<<size)-1;
  register unsigned int total=0;
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
    /**06 スカラー変数に置き換えた**********/
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
      /**06 スカラー変数に置き換えた**********/
      //if(bitmap[tid][row]==0){
      if(bitmap_tid_row==0){
      /***************************************/
        row--;
      }else{
        //クイーンを置く
        bitmap[tid][row]
          ^=bit
          /**06 スカラー変数に置き換えた**********/
          //=(-bitmap[tid][row]&bitmap[tid][row]);
          =(-bitmap_tid_row&bitmap_tid_row);       
          /***************************************/
        //置く場所があるかどうか
        if((bit&mask)!=0){
          //最終行?最終行から１個前の行まで
          //無事到達したら 加算する
          if(row+1==mark){
           total++;
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
  }else{
    //totalCond未満は空回しするのでtotalは加算しない
    sum[tid]=0;
  } 
  //__syncthreads()でブロック内のスレッド間の同期
  //全てのスレッドが__syncthreads()に辿り着くのを待つ
  __syncthreads();if(tid<64&&tid+64<THREAD_NUM){
    sum[tid]+=sum[tid+64];
  }
  __syncthreads();if(tid<32){
    sum[tid]+=sum[tid+32];
  } 
  __syncthreads();if(tid<16){
    sum[tid]+=sum[tid+16];
  } 
  __syncthreads();if(tid<8){
    sum[tid]+=sum[tid+8];
  } 
  __syncthreads();if(tid<4){
    sum[tid]+=sum[tid+4];
  } 
  __syncthreads();if(tid<2){
    sum[tid]+=sum[tid+2];
  } 
  __syncthreads();if(tid<1){
    sum[tid]+=sum[tid+1];
  } 
  __syncthreads();if(tid==0){
    d_results[bid]=sum[0];
  }
}
//
// GPU
long solve_nqueen_cuda(int size,int mask,int row,int n_left,int n_down,int n_right,int steps)
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
  /***06 bit処理をGPU*********************/
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
  //device
  unsigned int* downCuda;
  cudaMalloc((void**) &downCuda,sizeof(int)*steps);
  unsigned int* leftCuda;
  cudaMalloc((void**) &leftCuda,sizeof(int)*steps);
  unsigned int* rightCuda;
  cudaMalloc((void**) &rightCuda,sizeof(int)*steps);
  unsigned int* resultsCuda;
  cudaMalloc((void**) &resultsCuda,sizeof(int)*steps/THREAD_NUM);
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
    /***06 bit操作変更*********************/
    //06GPU こっちのほうが優秀
    if(bitmap[row]==0){ row--; }
    /************************/
    /***06 bit操作変更でコメント*********************/
    //06SGPU
    //if((bitmap[row]&mask)==mask){row--;}
    /************************/
    else{//おける場所があれば進む
      //06SGPU
      /***06 bit操作変更でコメント*********************/
      //bit=(bitmap[row]+1)&~bitmap[row];
      //bitmap[row]|=bit;
      /************************/
      //06GPU こっちのほうが優秀
      bitmap[row]^=bit=(-bitmap[row]&bitmap[row]); //クイーンを置く
      if((bit&mask)!=0){//置く場所があれば先に進む
        rowP=row+1;
        down[rowP]=down[row]|bit;
        left[rowP]=(left[row]|bit)<<1;
        right[rowP]=(right[row]|bit)>>1;
        /***06 bit操作変更でコメント*********************/
        //bitmap[rowP]=(down[rowP]|left[rowP]|right[rowP]);
        /************************/
        /***06 bit操作変更*********************/
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
              for(int col=0;col<steps/THREAD_NUM;col++){
                total+=h_results[col];
              }
              matched=false;
            }
            cudaMemcpy(downCuda,totalDown,
                sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            cudaMemcpy(leftCuda,totalLeft,
                sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            cudaMemcpy(rightCuda,totalRight,
                sizeof(int)*totalCond,cudaMemcpyHostToDevice);
            /** backTrack+bitmap*/
            //size-mark は何行GPUを実行するか totalCondはスレッド数
            cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
              >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
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
    for(int col=0;col<steps/THREAD_NUM;col++){
      total+=h_results[col];
    }
    matched=false;
  }
  cudaMemcpy(downCuda,totalDown,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(leftCuda,totalLeft,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(rightCuda,totalRight,
      sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  /** backTrack+bitmap*/
  //size-mark は何行GPUを実行するか totalCondはスレッド数
  //steps数の数だけマルチスレッドで起動するのだが、実際に計算が行われるのは
  //totalCondの数だけでそれ以外は空回しになる
  cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM
    >>>(size,size-mark,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
  cudaMemcpy(h_results,resultsCuda,
      sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  for(int col=0;col<steps/THREAD_NUM;col++){
    total+=h_results[col];
  }
  //
  cudaFree(downCuda);
  cudaFree(leftCuda);
  cudaFree(rightCuda);
  cudaFree(resultsCuda);
  /***06 cudaFreeHostへ変更**/
  //delete[] totalDown;
  cudaFreeHost(totalDown);
  //delete[] totalLeft;
  cudaFreeHost(totalLeft);
  //delete[] totalRight;
  cudaFreeHost(totalRight);
  //delete[] h_results;
  cudaFreeHost(h_results);
  /************************/
  return total;
}
//GPU
void NQueenG(int size,int steps)
{
  register int sizeE=size-1;
  register int bit=0;
  register int mask=((1<<size)-1);
  if(size<=0||size>32){return;}
  //偶数、奇数共通 右側半分だけクイーンを置く
	int lim=(size%2==0)?size/2:sizeE/2;
  for(int col=0;col<lim;col++){
    bit=(1<<col);
    TOTAL+=solve_nqueen_cuda(size,mask,1,bit<<1,bit,bit>>1,steps);
  }
  //ミラーなのでTOTALを２倍する
  TOTAL=TOTAL*2;
  //奇数の場合はさらに中央にクイーンを置く
  if(size%2==1){
    bit=(1<<(sizeE)/2);
    TOTAL+=solve_nqueen_cuda(size,mask,1,bit<<1,bit,bit>>1,steps);
  }
}
//
//SGPU
__global__ 
void sgpu_cuda_kernel(int size,int mark,unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,unsigned int* d_results,int totalCond)
{
  //スレッド
  const int tid=threadIdx.x;//ブロック内のスレッドID
  const int bid=blockIdx.x;//グリッド内のブロックID
  const int idx=bid*blockDim.x+tid;//全体通してのID
  //シェアードメモリ
  __shared__ unsigned int down[THREAD_NUM][10];//sharedメモリを使う ブロック内スレッドで共有
  __shared__ unsigned int left[THREAD_NUM][10];//THREAD_NUMはブロックあたりのスレッド数
  __shared__ unsigned int right[THREAD_NUM][10];//10で固定なのは現在のmaskの設定でGPUで実行するのは最大10だから
  __shared__ unsigned int bitmap[THREAD_NUM][10];
  __shared__ unsigned int sum[THREAD_NUM];
  //
  const unsigned int mask=(1<<size)-1;
  int total=0;
  int row=0;//row=0となってるが1行目からやっているわけではなくmask行目以降からスタート n=8 なら mask==2 なので そこからスタート
  unsigned int bit;
  if(idx<totalCond){//余分なスレッドは動かさない GPUはsteps数起動するがtotalCond以上は空回しする
    down[tid][row]=totalDown[idx];//totalDown,totalLeft,totalRightの情報をdown,left,rightに詰め直す 
    left[tid][row]=totalLeft[idx];//CPU で詰め込んだ t_はsteps個あるがブロック内ではブロックあたりのスレッドすうに限定されるので idxでよい
    right[tid][row]=totalRight[idx];
    bitmap[tid][row]=down[tid][row]|left[tid][row]|right[tid][row];//down,left,rightからbitmapを出す
    while(row>=0){
      //
      //06のGPU
      //if(bitmap[tid][row]==0){//bitmap[tid][row]=00000000 クイーンをどこにも置けないので1行上に戻る
      //06のSGPU
      if((bitmap[tid][row]&mask)==mask){//bitmap[tid][row]=00000000 クイーンをどこにも置けないので1行上に戻る
      //
        row--;
      }else{
        //
        //06GPU
        //bitmap[tid][row]^=bit=(-bitmap[tid][row]&bitmap[tid][row]); //クイーンを置く
        //06SGPU
        bit=(bitmap[tid][row]+1)&~bitmap[tid][row];
        bitmap[tid][row]|=bit;
        //
        if((bit&mask)!=0){//置く場所があるかどうか
          if(row+1==mark){//最終行?最終行から１個前の行まで無事到達したら 加算する
            total++;
            row--;
          }
          else{
            down[tid][row+1]=down[tid][row]|bit;
            left[tid][row+1]=(left[tid][row]|bit)<<1;
            right[tid][row+1]=(right[tid][row]|bit)>>1;
            bitmap[tid][row+1]=(down[tid][row+1]|left[tid][row+1]|right[tid][row+1]);
            row++;
          }
        }else{//置く場所がなければ１個上に
          row--;
        }
      }
    }
    sum[tid]=total;//最後sum[tid]に加算する
  }else{//totalCond未満は空回しするので当然 totalは加算しない
    sum[tid]=0;
  } 
  //__syncthreads()で、ブロック内のスレッド間の同期をとれます。
  //同期を取るということは、全てのスレッドが__syncthreads()に辿り着くのを待つ
  __syncthreads();if(tid<64&&tid+64<THREAD_NUM){sum[tid]+=sum[tid+64];}//__syncthreads();は複数個必要1個だけ記述したら数が違った
  __syncthreads();if(tid<32){sum[tid]+=sum[tid+32];} 
  __syncthreads();if(tid<16){sum[tid]+=sum[tid+16];} 
  __syncthreads();if(tid<8){sum[tid]+=sum[tid+8];} 
  __syncthreads();if(tid<4){sum[tid]+=sum[tid+4];} 
  __syncthreads();if(tid<2){sum[tid]+=sum[tid+2];} 
  __syncthreads();if(tid<1){sum[tid]+=sum[tid+1];} 
  __syncthreads();if(tid==0){d_results[bid]=sum[0];}
}
//
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
  unsigned int* h_results=new unsigned int[steps];

  //device
  unsigned int* downCuda;
  cudaMalloc((void**) &downCuda,sizeof(int)*steps);
  unsigned int* leftCuda;
  cudaMalloc((void**) &leftCuda,sizeof(int)*steps);
  unsigned int* rightCuda;
  cudaMalloc((void**) &rightCuda,sizeof(int)*steps);
  unsigned int* resultsCuda;
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
                cudaMemcpy(h_results,resultsCuda,
                    sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
                for(int col=0;col<steps/THREAD_NUM;col++){total+=h_results[col];}
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
    cudaMemcpy(h_results,resultsCuda,
        sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    for(int col=0;col<steps/THREAD_NUM;col++){total+=h_results[col];}
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
  cudaMemcpy(h_results,resultsCuda,
      sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  for(int col=0;col<steps/THREAD_NUM;col++){total+=h_results[col];}	
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
                cudaMemcpy(h_results,resultsCuda,
                    sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
                for(int col=0;col<steps/THREAD_NUM;col++){total+=h_results[col];}
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
      cudaMemcpy(h_results,resultsCuda,
          sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
      for(int col=0;col<steps/THREAD_NUM;col++){total+=h_results[col];}
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
    cudaMemcpy(h_results,resultsCuda,
        sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    for(int col=0;col<steps/THREAD_NUM;col++){total+=h_results[col];}
  }
  cudaFree(downCuda);
  cudaFree(leftCuda);
  cudaFree(rightCuda);
  cudaFree(resultsCuda);
  delete[] totalDown;
  delete[] totalLeft;
  delete[] totalRight;
  delete[] h_results;
  return total;
}
//
//CUDA 初期化
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
//通常版 CPU 非再帰版 ロジックメソッド
void NQueenD(int size,int mask,int row){
  int aStack[size];
  int* pnStack;
  int bit;
  int bitmap;
  int sizeE=size-1;
  int down[size],right[size],left[size];
  aStack[0]=-1; 
  pnStack=aStack+1;
  bit=0;
  bitmap=mask;
  down[0]=left[0]=right[0]=0;
  while(true){
    if(bitmap){
      bitmap^=bit=(-bitmap&bitmap); 
      if(row==sizeE){
        TOTAL++;
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
void NQueenDR(int size,int mask,int row,int left,int down,int right){
  int bitmap=0;
  int bit=0;
  if(row==size){
    TOTAL++;
  }else{
    bitmap=(mask&~(left|down|right));
    while(bitmap){
      bitmap^=bit=(-bitmap&bitmap);
      NQueenDR(size,mask,row+1,(left|bit)<<1, down|bit,(right|bit)>>1);
    }
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
      //
      //CPUR
      if(cpur){ 
        NQueenR(i,mask,0,0,0,0); 
        //NQueenDR(i,mask,0,0,0,0);//通常版
      }
      //CPU
      if(cpu){ 
        NQueen(i,mask); 
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
      gettimeofday(&t0,NULL);  // 計測開始
      if(gpu){
        TOTAL=0;
        UNIQUE=0;
        NQueenG(i,steps);
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

