
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
typedef unsigned int uint;
typedef unsigned long ulong;
#define MAX 27
#define THREAD_NUM 96
// システムによって以下のマクロが必要であればコメントを外してください
//#define UINT64_C(c) c ## ULL
ulong TOTAL=0;
ulong UNIQUE=0;
typedef struct local
{
  uint BOUND1,BOUND2;
  uint TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
  ulong board[MAX];
  ulong COUNT2,COUNT4,COUNT8,TOTAL,UNIQUE;
  uint STEPS;
}local;
//
struct dlocal
{
  uint BOUND1,BOUND2;
  uint TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
  ulong board[MAX];
  ulong COUNT2,COUNT4,COUNT8,TOTAL,UNIQUE;
}dlocal;
struct dlocal gdl[9999];
/**
  CPU 再帰/非再帰共通 対称解除法
*/
void symmetryOps(uint size,struct local* l)
{
  if(l->board[l->BOUND2]==1){
    uint ptn;
    uint own;
    for(ptn=2,own=1;own<size;++own,ptn<<=1){
      uint bit;
      uint you;
      for(bit=1,you=size-1;(l->board[you]!=ptn)&&l->board[own]>=bit;--you){
        bit<<=1;
      }
      if(l->board[own]>bit){
        return ;
      }
      if(l->board[own]<bit){
        break;
      }
    }
    if(own>size-1){
      l->COUNT2++;
      return ;
    }
  }
  //１８０度回転
  if(l->board[size-1]==l->ENDBIT){
    uint you;
    uint own;
    for(you=size-1-1,own=1;own<=size-1;++own,--you){
      uint bit;
      uint ptn;
      for(bit=1,ptn=l->TOPBIT;(ptn!=l->board[you])&&(l->board[own]>=bit);ptn>>=1){
        bit<<=1;
      }
      if(l->board[own]>bit){
        return ;
      }
      if(l->board[own]<bit){
        break;
      }
    }
    //９０度回転が同型でなくても１８０度回転が同型であることもある
    if(own>size-1){
      l->COUNT4++;
      return ;
    }
  }
  //２７０度回転
  if(l->board[l->BOUND1]==l->TOPBIT){
    uint ptn;
    uint own;
    uint you;
    uint bit;
    for(ptn=l->TOPBIT>>1,own=1;own<=size-1;++own,ptn>>=1){
      for(bit=1,you=0;(l->board[you]!=ptn)&&(l->board[own]>=bit);++you){
        bit<<=1;
      }
      if(l->board[own]>bit){
        return ;
      }
      if(l->board[own]<bit){
        break;
      }
    }
  }
  l->COUNT8++;
}
/**
  非再帰 角にQがないときのバックトラック
*/
void symmetry_backTrack_NR(uint size,uint row,uint _left,uint _down,uint _right,struct local *l)
{
  uint mask=(1<<size)-1;
  uint down[size];
  uint left[size];
  uint right[size];
  uint bitmap[size];
  left[row]=_left;
  down[row]=_down;
  right[row]=_right;
  bitmap[row]=mask&~(left[row]|down[row]|right[row]);
  while(row>0){
    if(bitmap[row]>0){
      if(row<l->BOUND1){ //上部サイド枝刈り
        bitmap[row]|=l->SIDEMASK;
        bitmap[row]^=l->SIDEMASK;
      }else if(row==l->BOUND2){ //下部サイド枝刈り
        if((down[row]&l->SIDEMASK)==0){
          row--; 
        }
        if((down[row]&l->SIDEMASK)!=l->SIDEMASK){
          bitmap[row]&=l->SIDEMASK;
        }
      }
      uint save_bitmap=bitmap[row];
      uint bit=-bitmap[row]&bitmap[row];
      bitmap[row]^=bit;
      l->board[row]=bit; //Qを配置
      if((bit&mask)!=0){
        if(row==(size-1)){
          if( (save_bitmap&l->LASTMASK)==0){
            symmetryOps(size,l);  //対称解除法
          }
          row--;
        }else{
          uint n=row++;
          left[row]=(left[n]|bit)<<1;
          down[row]=(down[n]|bit);
          right[row]=(right[n]|bit)>>1;
          bitmap[row]=mask&~(left[row]|down[row]|right[row]);
        }
      }else{
        row--;
      }
    }else{
      row--;
    }
  }
}
/**
  非再帰 角にQがあるときのバックトラック
*/
void symmetry_backTrack_corner_NR(uint size,uint row,uint _left,uint _down,uint _right,struct local *l)
{
  uint mask=(1<<size)-1;
  uint bit=0;
  uint down[size];
  uint left[size];
  uint right[size];
  uint bitmap[size];
  left[row]=_left;
  down[row]=_down;
  right[row]=_right;
  bitmap[row]=mask&~(left[row]|down[row]|right[row]);
  while(row>=2){
    if(row<l->BOUND1){
      bitmap[row]&=~2;
    }
    if(bitmap[row]>0){
      bit=-bitmap[row]&bitmap[row];
      bitmap[row]^=bit;
      if(row==(size-1)){
        l->COUNT8++;
        row--;
      }else{
        uint n=row++;
        left[row]=(left[n]|bit)<<1;
        down[row]=(down[n]|bit);
        right[row]=(right[n]|bit)>>1;
        l->board[row]=bit; //Qを配置
        bitmap[row]=mask&~(left[row]|down[row]|right[row]);
      }
    }else{
      row--;
    }
  }
}
/**
  非再帰 対称解除法
*/
void symmetry_NR(uint size,struct local* l)
{
  l->TOTAL=l->UNIQUE=l->COUNT2=l->COUNT4=l->COUNT8=0;
  uint bit=0;
  l->TOPBIT=1<<(size-1);
  l->ENDBIT=l->SIDEMASK=l->LASTMASK=0;
  l->BOUND1=2;
  l->BOUND2=0;
  l->board[0]=1;
  while(l->BOUND1>1&&l->BOUND1<size-1){
    if(l->BOUND1<size-1){
      bit=1<<l->BOUND1;
      l->board[1]=bit;   //２行目にQを配置
      symmetry_backTrack_corner_NR(size,2,(2|bit)<<1,1|bit,(2|bit)>>1,l);
    }
    l->BOUND1++;
  }
  l->TOPBIT=1<<(size-1);
  l->ENDBIT=l->TOPBIT>>1;
  l->SIDEMASK=l->TOPBIT|1;
  l->LASTMASK=l->TOPBIT|1;
  l->BOUND1=1;
  l->BOUND2=size-2;
  while(l->BOUND1>0 && l->BOUND2<size-1 && l->BOUND1<l->BOUND2){
    if(l->BOUND1<l->BOUND2){
      bit=1<<l->BOUND1;
      l->board[0]=bit;   //Qを配置
      symmetry_backTrack_NR(size,1,bit<<1,bit,bit>>1,l);
    }
    l->BOUND1++;
    l->BOUND2--;
    l->ENDBIT=l->ENDBIT>>1;
    l->LASTMASK=l->LASTMASK<<1|l->LASTMASK|l->LASTMASK>>1;
  }
  UNIQUE=l->COUNT2+l->COUNT4+l->COUNT8;
  TOTAL=l->COUNT2*2+l->COUNT4*4+l->COUNT8*8;
}
/**
  再帰 角にQがないときのバックトラック
*/
void symmetry_backTrack(uint size,uint row,uint left,uint down,uint right,struct local* l)
{
  uint mask=(1<<size)-1;
  uint bitmap=mask&~(left|down|right);
  if(row==(size-1)){
    if(bitmap){
      if( (bitmap&l->LASTMASK)==0){
        l->board[row]=bitmap;  //Qを配置
        symmetryOps(size,l);    //対称解除
      }
    }
  }else{
    if(row<l->BOUND1){
      bitmap=bitmap|l->SIDEMASK;
      bitmap=bitmap^l->SIDEMASK;
    }else{
      if(row==l->BOUND2){
        if((down&l->SIDEMASK)==0){
          return;
        }
        if( (down&l->SIDEMASK)!=l->SIDEMASK){
          bitmap=bitmap&l->SIDEMASK;
        }
      }
    }
    while(bitmap){
      uint bit=-bitmap&bitmap;
      bitmap=bitmap^bit;
      l->board[row]=bit;
      symmetry_backTrack(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
}
/**
  再帰 角にQがあるときのバックトラック
*/
void symmetry_backTrack_corner(uint size,uint row,uint left,uint down,uint right,struct local* l)
{
  uint mask=(1<<size)-1;
  uint bitmap=mask&~(left|down|right);
  uint bit=0;
  if(row==(size-1)){
    if(bitmap){
      l->board[row]=bitmap;
      l->COUNT8++;
    }
  }else{
    if(row<l->BOUND1){   //枝刈り
      bitmap=bitmap|2;
      bitmap=bitmap^2;
    }
    while(bitmap){
      bit=-bitmap&bitmap;
      bitmap=bitmap^bit;
      l->board[row]=bit;   //Qを配置
      symmetry_backTrack_corner(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1,l);
    }
  }
}
/**
  再帰 対称解除法
*/
void symmetry_R(uint size,struct local* l)
{
  l->TOTAL=l->UNIQUE=l->COUNT2=l->COUNT4=l->COUNT8=0;
  uint bit=0;
  l->TOPBIT=1<<(size-1);
  l->ENDBIT=l->LASTMASK=l->SIDEMASK=0;
  l->BOUND1=2;
  l->BOUND2=0;
  l->board[0]=1;
  while(l->BOUND1>1 && l->BOUND1<size-1){
    if(l->BOUND1<size-1){
      bit=1<<l->BOUND1;
      l->board[1]=bit;   //２行目にQを配置
      symmetry_backTrack_corner(size,2,(2|bit)<<1,1|bit,(2|bit)>>1,l);
    }
    l->BOUND1++;
  }//end while
  l->TOPBIT=1<<(size-1);
  l->ENDBIT=l->TOPBIT>>1;
  l->SIDEMASK=l->TOPBIT|1;
  l->LASTMASK=l->TOPBIT|1;
  l->BOUND1=1;
  l->BOUND2=size-2;
  while(l->BOUND1>0 && l->BOUND2<size-1 && l->BOUND1<l->BOUND2){
    if(l->BOUND1<l->BOUND2){
      bit=1<<l->BOUND1;
      l->board[0]=bit;   //Qを配置
      symmetry_backTrack(size,1,bit<<1,bit,bit>>1,l);
    }
    l->BOUND1++;
    l->BOUND2--;
    l->ENDBIT=l->ENDBIT>>1;
    l->LASTMASK=l->LASTMASK<<1|l->LASTMASK|l->LASTMASK>>1;
  }
  UNIQUE=l->COUNT2+l->COUNT4+l->COUNT8;
  TOTAL=l->COUNT2*2+l->COUNT4*4+l->COUNT8*8;
}
/**
 * メイン
 */
int main(int argc,char** argv)
{
  bool cpu=false,cpur=false,gpu=false,gpuBitBoard=false;
  uint argstart=2;
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
    else if(argv[1][1]=='n'||argv[1][1]=='N'){gpuBitBoard=true;}
    else{ gpuBitBoard=true; } //デフォルトをgpuとする
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g|-r|-s] n STEPS\n",argv[0]);
    printf("  -r: CPU 再帰\n");
    printf("  -c: CPU 非再帰\n");
    printf("  -g: GPU 再帰\n");
    printf("  -n: GPU ビットボード\n");
  }
  if(cpur){ printf("\n\n対称解除法 再帰 \n"); }
  else if(cpu){ printf("\n\n対称解除法 非再帰 \n"); }
  else if(gpu){ printf("\n\n対称解除法 GPU\n"); }
  else if(gpuBitBoard){ printf("\n\n対称解除法 GPUビットボード \n"); }
  if(cpu||cpur)
  {
    uint min=4; 
    uint targetN=17;
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    for(uint size=min;size<=targetN;size++){
      local l;
      gettimeofday(&t0,NULL);//計測開始
      if(cpur){ //再帰
        symmetry_R(size,&l);
      }
      if(cpu){ //非再帰
        symmetry_NR(size,&l);
      }
      //
      gettimeofday(&t1,NULL);//計測終了
      uint ss;
      uint ms;
      uint dd;
      if(t1.tv_usec<t0.tv_usec) {
        dd=(t1.tv_sec-t0.tv_sec-1)/86400;
        ss=(t1.tv_sec-t0.tv_sec-1)%86400;
        ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
      }else {
        dd=(t1.tv_sec-t0.tv_sec)/86400;
        ss=(t1.tv_sec-t0.tv_sec)%86400;
        ms=(t1.tv_usec-t0.tv_usec+500)/10000;
      }//end if
      uint hh=ss/3600;
      uint mm=(ss-hh*3600)/60;
      ss%=60;
      printf("%2d:%13ld%12ld%8.2d:%02d:%02d:%02d.%02d\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
    }
  }
  return 0;
}
