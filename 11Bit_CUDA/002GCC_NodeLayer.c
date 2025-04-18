/**
bash-5.1$ g++ -W -Wall -O3 00GCC_NodeLayer.c && ./a.out
ノードレイヤー
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
13:        73712           0      00:00:00:00.02
14:       365596           0      00:00:00:00.11
15:      2279184           0      00:00:00:00.70
16:     14772512           0      00:00:00:04.69
17:     95815104           0      00:00:00:32.70
18:    666090624           0      00:00:03:59.95

bash-5.1$ gcc -W -Wall -O3 01CUDA_Bit_Symmetry.c && ./a.out
 N:        Total      Unique      dd:hh:mm:ss.ms
15:      2279184      285053      00:00:00:00.33
16:     14772512     1846955      00:00:00:02.16
17:     95815104    11977939      00:00:00:14.89
18:    666090624    83263591      00:00:01:45.44
*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define MAX 27
#define THREAD_NUM 96
// システムによって以下のマクロが必要であればコメントを外してください
//#define UINT64_C(c) c ## ULL
typedef unsigned int uint;
typedef unsigned long ulong;
ulong TOTAL=0;
ulong UNIQUE=0;
/**
 * solve
 */
ulong solve(uint size,uint left,uint down,uint right)
{
  uint bit=0;
  uint mask=(1<<size)-1;
  uint counter = 0;
  if (down==mask) { // downがすべて専有され解が見つかる
    return 1;
  }
  for(ulong bitmap=mask&~(left|down|right);bitmap;bitmap^=bit){
    bit=-bitmap&bitmap;
    counter+=solve(size,(left|bit)>>1,(down|bit),(right|bit)<< 1);
  }
  return counter;
}
/**
 * 0以外のbitをカウント
 */
uint countBits(ulong n)
{
  uint counter=0;
  while (n){ n&=(n-1);counter++; }
  return counter;
}
/**
 * ノードをk番目のレイヤーのノードで埋める
 */
ulong nodeLayer(uint size,ulong **nodes,uint k,uint left,uint down,uint right, uint *index)
{
  uint bit=0;
  uint counter=0;
  uint mask=(1<<size)-1;
  if (countBits(down)==k) {
    (*nodes)[(*index)++]=left;
    (*nodes)[(*index)++]=down;
    (*nodes)[(*index)++]=right;
    return 1;
  }
  for(ulong bitmap=mask&~(left|down|right);bitmap;bitmap^=bit){
    bit=-bitmap&bitmap;
    counter+=nodeLayer(size,nodes,k,(left|bit)>>1,(down|bit),(right|bit)<<1,index);
  }
  return counter;
}
/**
 * k 番目のレイヤのすべてのノードを含むベクトルを返す。
*/
ulong *kLayer(uint size,uint k,uint *node_count)
{
  uint max_nodes = 1000000;
  ulong *nodes = (ulong *)malloc(max_nodes * sizeof(ulong));
  if (!nodes) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  *node_count=0;
  nodeLayer(size,&nodes,k,0,0,0,node_count);
  if (*node_count > max_nodes) {
    fprintf(stderr, "Buffer overflow detected!\n");
    free(nodes);
    exit(1);
  }
  return nodes;
}
/**
 * nqueens
 */
void nqueens(uint size)
{
  // ツリーの3番目のレイヤーにあるノード
  //（それぞれ連続する3つの数字でエンコードされる）のベクトル。
  // レイヤー2以降はノードの数が均等なので、対称性を利用できる。
  // レイヤ4には十分なノードがある（N16の場合、9844）。
  uint node_count = 0;
  ulong *nodes = kLayer(size,4,&node_count);
  // 必要なのはノードの半分だけで、各ノードは3つの整数で符号化される。
  uint numSolutions = node_count / 6;
  //backtrackでクイーンを置く
  ulong *solutions = (ulong *)malloc(numSolutions * sizeof(ulong));
  if (!solutions) {
    fprintf(stderr, "Memory allocation failed\n");
    free(nodes);
    exit(1);
  }
  for (ulong i=0;i<numSolutions;i++) {
    solutions[i]=solve(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2]);
  }
  // 部分解を加算し、結果を表示する。
  //ミラーなので２倍する
  for (ulong i=0;i<numSolutions;i++) { TOTAL+=solutions[i]*2; }
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
  printf("%s\n","ノードレイヤー");
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
