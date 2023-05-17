#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#define THREAD_NUM		96
#define MAX 27
int size;
long TOTAL=0; 
long UNIQUE=0;
int COUNT8=0;
int COUNT4=0;
int COUNT2=0;
typedef struct{
  int row;
  int down;
  int left;
  int right;
  int x[MAX];
}Board ;
Board B;
long cnt[3];
//
// void process(int sym)
// {
//   cnt[sym] += solve(B.row >> 2,
//       B.left>>4,
//       ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
//       (B.right>>4)<<(size-5));
// }
//
long solve(int row,int left,int down,int right)
{
  // Placement Complete?
  //bh=-1 1111111111 すべての列にクイーンを置けると-1になる
  if(down+1==0){ 
    return  1; 
  }
  // -> at least one more queen to place
  // Column is covered by pre-placement
  //row 右端にクイーンがすでに置かれていたら。クイーンを置かずに１行下に移動する
  //rowを右端から１ビットずつ削っていく。ここではrowはすでにクイーンが置かれているかどうかだけで使う
  while((row&1)!=0) { 
    row>>=1;  //右に１ビットシフト
    left<<=1; //left 左に１ビットシフト
    right>>=1;//right 右に１ビットシフト
  }
  row>>=1; //１行下に移動する
  // Column needs to be placed
  long  s=0;
  int bitmap=~(left|down|right);
  int bit;
  while(bitmap!=0){
    bit=bitmap&-bitmap;
    bitmap^=bit;
    s+=solve(row,(left|bit)<<1,down|bit,(right|bit)>>1);
  }
  return s; //途中でクイーンを置くところがなくなるとここに来る
} 
void process(int size,Board B,int sym)
{
  cnt[sym] += solve(B.row >> 2,
      B.left>>4,
      ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
      (B.right>>4)<<(size-5));
}
//
bool placement(int dimx,int dimy)
{
  if(B.x[dimx]==dimy){ 
    return true;  //同じ場所に置くのはOK
  }  
  B.x[dimx]=dimy;                       //xは行 yは列
  int row=1<<dimx;
  int down=1<<dimy;
  int left=1<<(size-1-dimx+dimy);    //右上から左下
  int right=1<<(dimx+dimy);          // 左上から右下
  if((B.row&row)||(B.down&down)||(B.left&left)||(B.right&right)){
    return false;
  }     
  B.row|=row; B.down|=down; B.left|=left; B.right|=right;
  return true;
}
//
void NQueenR()
{
  int pres_a[930];
  int pres_b[930];
  int idx=0;
  for(int a=0;a<size;a++){
    for(int b=0;b<size;b++){
      if( ((a>=b)&&(a-b)<=1)||((b>a)&&(b-a)<=1)){ continue; }     
      pres_a[idx]=a;
      pres_b[idx]=b;
      idx++;
    }
  }
  //
  //
  //
  //
  //
  //１ 上２行にクイーンを置く
  //上１行は２分の１だけ実行
  //q=7なら (7/2)*(7-4)=12
  //1行目は0,1,2で,2行目0,1,2,3,4,5,6 で利き筋を置かないと13パターンになる
  Board wB=B;
  for(int w=0;w<=(size/2)*(size-3);w++){
    B=wB;
    B.row=B.down=B.left=B.right=0;
    for(int i=0;i<size;i++){ B.x[i]=-1; }
    int pna;
    pna=placement(0,pres_a[w]); //0行目にクイーンを置く
    // printf("pna:%d\n",pna);
    pna=placement(1,pres_b[w]); //1行目にクイーンを置く
    // printf("pna:%d\n",pna);
    //
    //
    //
    //
    // ２ 左２列にクイーンを置く
    Board nB=B;
    for(int n=w;n<(size-2)*(size-1)-w;n++){
      B=nB;
      // printf("pres_a:%d\n",pres_a[n]);
      pna=placement(pres_a[n],size-1);
      // printf("%d",pna);
      if(pna==false){ continue; }
      // printf("pres_a:%d\n",pres_a[n]);
      pna=placement(pres_b[n],size-2);
      if(pna==false){ continue; }
      // printf("pres_b:%d\n",pres_b[n]);
      //
      //
      //
      //
      // ３ 下２行に置く
      Board eB=B;
      for(int e=w;e<(size-2)*(size-1)-w;e++){
        B=eB;
        if(placement(size-1,size-1-pres_a[e])==false){ continue; }
        if(placement(size-2,size-1-pres_b[e])==false){ continue; }
        //
        //
        //
        //
        // ４ 右２列に置く
        Board sB=B;
        for(int s=w;s<(size-2)*(size-1)-w;s++){
          B=sB;
          if(placement(size-1-pres_a[s],0)==false){ continue; }
          if(placement(size-1-pres_b[s],1)==false){ continue; }
          //
          //
          //対象解除法
          //
          //// Check for minimum if n, e, s = (N-2)*(N-1)-1-w
          int ww=(size-2)*(size-1)-1-w;
          int w2=(size-2)*(size-1)-1;
          //check if flip about the up diagonal is smaller
          if((s==ww)&&(n<(w2-e))){ continue; }
          //if(e==ww){
          //check if flip about the vertical center is smaller
          if((e==ww)&&(n>(w2-n))){ continue; }
          // check if flip about the down diagonal is smaller
          if((n==ww)&&(e>(w2-s))){ continue; }
          //// Check for minimum if n, e, s = w
          // right rotation is smaller unless  w = n = e = s
          //右回転で同じ場合w=n=e=sでなければ値が小さいのでskip
          //w=n=e=sであれば90度回転で同じ可能性
          //この場合はミラーの2
          if(s==w){ if((n!=w)||(e!=w)){ continue; } 
            // process(COUNT2); 
          // cnt[COUNT2] += solve(B.row >> 2,
          COUNT2 += solve(B.row >> 2,
              B.left>>4,
              ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
              (B.right>>4)<<(size-5));
            continue; 
          }
          //if(n>=s){
          //e==wは180度回転して同じ
          //180度回転して同じ時n>=sの時はsmaller?
          //この場合は4
          if((e==w)&&(n>=s)){ if(n>s){ continue; } 
            // process(COUNT4); 
            // cnt[COUNT4] += solve(B.row >> 2,
            COUNT4 += solve(B.row >> 2,
                B.left>>4,
                ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
                (B.right>>4)<<(size-5));
            continue; 
          }
          //この場合は8
          // process(COUNT8);
          //cnt[COUNT8] += solve(B.row >> 2,
          COUNT8 += solve(B.row >> 2,
              B.left>>4,
              ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
              (B.right>>4)<<(size-5));
          continue;
        }
      }    
    }
  }
  UNIQUE=COUNT2+COUNT4+COUNT8;
  TOTAL=COUNT2*2+COUNT4*4+COUNT8*8;
}

int main(void){
  // size=5; 
  // TOTAL=0; UNIQUE=0; 
  // COUNT2=COUNT4=COUNT8=0;
  // NQueenR(); 
  // printf("%2d:%13ld%16ld\n", size,TOTAL,UNIQUE);

  size=12; 
  TOTAL=0;UNIQUE=0;
  COUNT2=COUNT4=COUNT8=0;
  NQueenR(); 
  printf("%2d:%13ld%16ld\n", size,TOTAL,UNIQUE);
}
