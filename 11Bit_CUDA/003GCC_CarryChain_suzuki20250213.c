/**
bash-5.1$ g++ -W -Wall -O3 00GCC_CarryChain.c && ./a.out
キャリーチェーン
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:            2           1      00:00:00:00.00
 5:           10           2      00:00:00:00.00
 6:            4           1      00:00:00:00.00
 7:           40           6      00:00:00:00.00
 8:           92          12      00:00:00:00.00
 9:          352          46      00:00:00:00.00
10:          724          92      00:00:00:00.00
11:         2680         341      00:00:00:00.00
12:        14200        1788      00:00:00:00.01
13:        73712        9237      00:00:00:00.05
14:       365596       45771      00:00:00:00.19
15:      2279184      285095      00:00:00:00.93
16:     14772512     1847425      00:00:00:05.46
17:     95815104    11979381      00:00:00:36.85
18:    657378384    82181924      00:00:04:18.49

bash-5.1$ g++ -W -Wall -O3 00GCC_NodeLayer.c && ./a.out
ノードレイヤー
 N:        Total      Unique      dd:hh:mm:ss.ms
15:      2279184           0      00:00:00:00.70
16:     14772512           0      00:00:00:04.69
17:     95815104           0      00:00:00:32.70
18:    666090624           0      00:00:03:59.95

bash-5.1$ gcc -W -Wall -O3 01CUDA_Bit_Symmetry.c && ./a.out
ビット
 N:        Total      Unique      dd:hh:mm:ss.ms
15:      2279184      285053      00:00:00:00.33
16:     14772512     1846955      00:00:00:02.16
17:     95815104    11977939      00:00:00:14.89
18:    666090624    83263591      00:00:01:45.44
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#define MAX 18
typedef unsigned long int;
// NQueens17 構造体
typedef struct{
  long total;
} NQueens17;
/*
 *
 */
long solve(int row,int left,int down,int right){
  long total=0;
  if((down+1)==0){
    return 1;
  }
  while(row&1){
    row>>=1;
    left<<=1;
    right>>=1;
  }
  row>>=1;
  int bitmap=~(left|down|right);
  while (bitmap!=0){
    int bit=-bitmap&bitmap;
    total+=solve(row,(left|bit)<<1,down|bit,(right|bit)>>1);
    bitmap^=bit;
  }
  return total;
}
/**
 *
 */
long process(int size,int sym,int B[]){
  return sym*solve(B[0]>>2,B[1]>>4,(((B[2]>>2|~0<<(size-4))+1)<<(size-5))-1,B[3]>>4<<(size-5));
}
/*
 *
 */
long Symmetry(int size,int n,int w,int s,int e,int B[],int B4[]){
  // 前計算
  int ww=(size-2)*(size-1)-1-w;
  int w2=(size-2)*(size-1)-1;
  // 対角線上の反転が小さいかどうか確認する
  if (s==ww && n<(w2-e)) return 0;
  // 垂直方向の中心に対する反転が小さいかを確認
  if (e==ww && n>(w2-n)) return 0;
  // 斜め下方向への反転が小さいかをチェックする
  if (n==ww && e>(w2-s)) return 0;
  // 【枝刈り】1行目が角の場合
  if (!B4[0]) return process(size,8,B); // COUNT8
  // n,e,s==w の場合は最小値を確認
  if (s==w){
    if (n!=w||e!=w) return 0;
    return process(size,2,B); // COUNT2
  }
  // e==w は180度回転して同じ
  if (e==w && n >= s){
    if (n>s) return 0;
    return process(size,4,B); // COUNT4
  }
  // その他の場合
  return process(size,8,B); // COUNT8
}
/*
 *
 */
int placement(int size,int dimx,int dimy,int B[],int B4[]){
  if (B4[dimx]==dimy) return 1;
  if (B4[0]){
    if ((B4[0]!=-1 && ((dimx<B4[0]||dimx >= size-B4[0]) && (dimy==0||dimy==size-1))) ||
        ((dimx==size-1) && (dimy<=B4[0]||dimy >= size-B4[0]))){
      return 0;
    }
  } else if ((B4[1]!=-1) && (B4[1] >= dimx && dimy==1)){
    return 0;
  }
  if ((B[0]&(1<<dimx))||(B[1]&(1<<(size-1-dimx+dimy)))||
      (B[2]&(1<<dimy))||(B[3]&(1<<(dimx+dimy)))){
    return 0;
  }
  B[0]|=1<<dimx;
  B[1]|=1<<(size-1-dimx+dimy);
  B[2]|=1<<dimy;
  B[3]|=1<<(dimx+dimy);
  B4[dimx]=dimy;
  return 1;
}
/*
 *
 */
void deepcopy(int *src,int *dest,int size){
  memcpy(dest,src,size*sizeof(int));
}
/*
 *
 */
int buildChain(int size,int pres_a[],int pres_b[]){
  long total=0;
  int B[4]={0,0,0,0};
  int B4[MAX];
  for (int i=0;i<size;i++){
    B4[i]=-1;
  }
  int sizeE=size-1;
  int sizeEE=size-2;
  int range_size=(size/2)*(size-3)+1;
  for (int w=0;w<range_size;w++){
    int wB[4],wB4[MAX];
    deepcopy(B,wB,4);
    deepcopy(B4,wB4,size);
    if (!placement(size,0,pres_a[w],wB,wB4)||!placement(size,1,pres_b[w],wB,wB4)) continue;
    for (int n=w;n<(sizeEE)*(sizeE)-w;n++){
      int nB[4],nB4[MAX];
      deepcopy(wB,nB,4);
      deepcopy(wB4,nB4,size);
      if (!placement(size,pres_a[n],sizeE,nB,nB4)||!placement(size,pres_b[n],sizeEE,nB,nB4)) continue;
      for (int e=w;e<(sizeEE)*(sizeE)-w;e++){
        int eB[4],eB4[MAX];
        deepcopy(nB,eB,4);
        deepcopy(nB4,eB4,size);
        if (!placement(size,sizeE,sizeE-pres_a[e],eB,eB4)||!placement(size,sizeEE,sizeE-pres_b[e],eB,eB4)) continue;
        for (int s=w;s<(sizeEE)*(sizeE)-w;s++){
          int sB[4],sB4[MAX];
          deepcopy(eB,sB,4);
          deepcopy(eB4,sB4,size);
          if (!placement(size,sizeE-pres_a[s],0,sB,sB4)||!placement(size,sizeE-pres_b[s],1,sB,sB4)) continue;
          total += Symmetry(size,n,w,s,e,sB,sB4);
        }
      }
    }
  }
  return total;
}
/**
 *
 */
void initChain(int size,int pres_a[],int pres_b[]){
  int idx=0;
  for (int a=0;a<size;a++){
    for (int b=0;b<size;b++){
      if (abs(a-b)<=1) continue;
      pres_a[idx]=a;
      pres_b[idx]=b;
      idx++;
    }
  }
}
/*
 *
 */
long carryChain(int size){
  int pres_a[930]={0};
  int pres_b[930]={0};
  initChain(size,pres_a,pres_b);
  return buildChain(size,pres_a,pres_b);
}
/**
 *
 */
void mainNQueens17(){
  int nmin=4;
  int nmax=18;
  struct timeval t0;
  struct timeval t1;
  int ss,ms,dd,hh,mm;
  printf("%s\n","キャリーチェーン");
  printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
  for (int size=nmin;size<=nmax;size++){
    gettimeofday(&t0,NULL);
    long TOTAL=carryChain(size);
    long UNIQUE=0;
    gettimeofday(&t1,NULL);
    if(t1.tv_usec<t0.tv_usec){
      dd=(t1.tv_sec-t0.tv_sec-1)/86400;
      ss=(t1.tv_sec-t0.tv_sec-1)%86400;
      ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
    }else{
      dd=(t1.tv_sec-t0.tv_sec)/86400;
      ss=(t1.tv_sec-t0.tv_sec)%86400;
      ms=(t1.tv_usec-t0.tv_usec+500)/10000;
    }
    hh=ss/3600;
    mm=(ss-hh*3600)/60;
    ss%=60;
    printf("%2d:%13ld%12ld%8.2d:%02d:%02d:%02d.%02d\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
  }
}
/**
 *
 */
int main(){
  mainNQueens17();
  return 0;
}

