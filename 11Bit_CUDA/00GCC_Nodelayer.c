/**
bash-5.1$ gcc -W -Wall -O3 02CUDA_Nodelayer_bitmap.c && ./a.out
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2           0      00:00:00:00.00
 5:           10           0      00:00:00:00.00
 6:            4           0      00:00:00:00.00
 7:           40           0      00:00:00:00.00
 8:           92           0      00:00:00:00.00
 9:          352           0      00:00:00:00.00
10:          724           0      00:00:00:00.00
11:         2680           0      00:00:00:00.00
12:        14200           0      00:00:00:00.00
13:        73712           0      00:00:00:00.03
14:       365596           0      00:00:00:00.22
15:      2279184           0      00:00:00:01.36
16:     14772512           0      00:00:00:10.12
17:     95815104           0      00:00:01:04.76
18:    666090624           0      00:00:08:18.07
 */
 #include <iostream>
 #include <vector>
 #include <stdio.h>
 #include <stdlib.h>
 #include <stdbool.h>
 #include <math.h>
 #include <string.h>
 #include <time.h>
 #include <sys/time.h>
 #define MAX 27
 #define THREAD_NUM 96
 // システムによって以下のマクロが必要であればコメントを外してください
 //#define UINT64_C(c) c ## ULL
 ulong TOTAL=0;
 ulong UNIQUE=0;
 typedef unsigned int uint;
 typedef unsigned long ulong;
  long bitmap_solve_nodeLayer(int size,long left,long down,long right)
 {
   long mask=(1<<size)-1;
   long counter = 0;
   if (down==mask) { // downがすべて専有され解が見つかる
     return 1;
   }
   long bit=0;
   for(long bitmap=mask&~(left|down|right);bitmap;bitmap^=bit){
     bit=-bitmap&bitmap;
     counter += bitmap_solve_nodeLayer(size,(left|bit)>>1,(down|bit),(right|bit)<< 1); 
   }
   return counter;
 }
 // 0以外のbitをカウント
 int countBits_nodeLayer(long n)
 {
   int counter = 0;
   while (n){
     n &= (n - 1); // 右端のゼロ以外の数字を削除
     counter++;
   }
   return counter;
 }
 // ノードをk番目のレイヤーのノードで埋める
 long kLayer_nodeLayer(int size,std::vector<long>& nodes, int k, long left, long down, long right)
 {
   long counter=0;
   long mask=(1<<size)-1;
   // すべてのdownが埋まったら、解決策を見つけたことになる。
   if (countBits_nodeLayer(down) == k) {
     nodes.push_back(left);
     nodes.push_back(down);
     nodes.push_back(right);
     return 1;
   }
   long bit=0;
   for(long bitmap=mask&~(left|down|right);bitmap;bitmap^=bit){
     bit=-bitmap&bitmap;
     // 解を加えて対角線をずらす
     counter+=kLayer_nodeLayer(size,nodes,k,(left|bit)>>1,(down|bit),(right|bit)<<1); 
   }
   return counter;
 }
 // k 番目のレイヤのすべてのノードを含むベクトルを返す。
 std::vector<long> kLayer_nodeLayer(int size,int k)
 {
   std::vector<long> nodes{};
   kLayer_nodeLayer(size,nodes, k, 0, 0, 0);
   return nodes;
 }
 //ノードレイヤーの作成
 void nqueens(int size)
 {
   // ツリーの3番目のレイヤーにあるノード
   //（それぞれ連続する3つの数字でエンコードされる）のベクトル。
   // レイヤー2以降はノードの数が均等なので、対称性を利用できる。
   // レイヤ4には十分なノードがある（N16の場合、9844）。
   std::vector<long> nodes = kLayer_nodeLayer(size,4); 
   // 必要なのはノードの半分だけで、各ノードは3つの整数で符号化される。
   int numSolutions = nodes.size() / 6; 
   //backtrackでクイーンを置く
   long solutions[numSolutions];
   for (long i = 0; i < numSolutions; i++) {
     solutions[i]=bitmap_solve_nodeLayer(size,nodes[3 * i],nodes[3 * i + 1],nodes[3 * i + 2]);	
   }
   // 部分解を加算し、結果を表示する。
   //ミラーなので２倍する
   for (long i = 0; i < numSolutions; i++) {
     TOTAL+=solutions[i]*2;
   }
 }
 /**
  * メイン
  */
 int main(int argc,char** argv)
 {
   uint min=4;
   uint targetN=18;
   struct timeval t0;
   struct timeval t1;
   printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
   uint ss,ms,dd,hh,mm;
   for(uint size=min;size<=targetN;size++){
     TOTAL=UNIQUE=0;
     gettimeofday(&t0,NULL);
     nqueens(size);
     gettimeofday(&t1,NULL);
     if(t1.tv_usec<t0.tv_usec) {
       dd=(t1.tv_sec-t0.tv_sec-1)/86400;
       ss=(t1.tv_sec-t0.tv_sec-1)%86400;
       ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
     }else {
       dd=(t1.tv_sec-t0.tv_sec)/86400;
       ss=(t1.tv_sec-t0.tv_sec)%86400;
       ms=(t1.tv_usec-t0.tv_usec+500)/10000;
     }
     hh=ss/3600;
     mm=(ss-hh*3600)/60;
     ss%=60;
     printf("%2d:%13ld%12ld%8.2d:%02d:%02d:%02d.%02d\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
   }
   return 0;
 }
 