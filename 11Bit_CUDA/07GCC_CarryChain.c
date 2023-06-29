/**
 *
 * bash版キャリーチェーンのC言語版
 *
 詳しい説明はこちらをどうぞ
 https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題
 *
 *
 * キャリーチェーンは、対象解除法よりも確かに処理速度は遅いのですが、配列などを極力排除し、高速演算処理ができるビットで処理しています。また、処理の冒頭で対象解除法とミラーを行い、通過した解の候補だけクイーンの配置処理を行うなどロジック的にとても優れていると同時に、省メモリで長時間の実行を可能としています。
 *
 * ちなみに対象解除法はＮ２４でメモリ領域が枯渇してバーストしましたが、キャリーチェーンは行けそうな「気が」します。
 *
bash-3.2$ gcc 07GCC_CarryChain.c && ./a.out
Usage: ./a.out [-c|-r|-g]
  -c: CPU Without recursion
  -r: CPUR Recursion
  -g: GPU Without Recursion
７．キャリーチェーン
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.01
12:        14200            1788            0.04
13:        73712            9237            0.14
14:       365596           45771            0.48
15:      2279184          285095            2.06
16:     14772512         1847425           11.57
17:     95815104        11979381         1:15.20
18:    666090624        83274576         8:34.84
bash-3.2$
 *
 */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#define MAX 27
// システムによって以下のマクロが必要であればコメントを外してください。
//#define UINT64_C(c) c ## ULL
//
// グローバル変数
typedef unsigned long long uint64_t;
uint64_t TOTAL=0; 
uint64_t UNIQUE=0;
// 構造体
typedef struct{
  unsigned int size;
  unsigned int pres_a[930]; 
  unsigned int pres_b[930];
  // uint64_t COUNTER[3];      
  // //カウンター配列
  // unsigned int COUNT2;
  // unsigned int COUNT4;
  // unsigned int COUNT8;
}Global; Global g;
// 構造体
typedef struct{
  uint64_t row;
  uint64_t down;
  uint64_t left;
  uint64_t right;
  uint64_t x[MAX];
}Board ;
typedef struct{
  Board B;
  Board nB;
  Board eB;
  Board sB;
  Board wB;
  unsigned n;
  unsigned e;
  unsigned s;
  unsigned w;
  uint64_t dimx;
  uint64_t dimy;
  uint64_t COUNTER[3];      
  //カウンター配列
  unsigned int COUNT2;
  unsigned int COUNT4;
  unsigned int COUNT8;
}Local;
//hh:mm:ss.ms形式に処理時間を出力
void TimeFormat(clock_t utime,char* form)
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
// ボード外側２列を除く内側のクイーン配置処理
uint64_t solve(uint64_t row,uint64_t left,uint64_t down,uint64_t right)
{
  if(down+1==0){ return  1; }
  while((row&1)!=0) { 
    row>>=1;
    left<<=1;
    right>>=1;
  }
  row>>=1;
  uint64_t total=0;
  for(uint64_t bitmap=~(left|down|right);bitmap!=0;){
    uint64_t const bit=bitmap&-bitmap;
    total+=solve(row,(left|bit)<<1,down|bit,(right|bit)>>1);
    bitmap^=bit;
  }
  return total;
} 
// クイーンの効きをチェック
bool placement(void* args)
{
  Local *l=(Local *)args;
  if(l->B.x[l->dimx]==l->dimy){ return true;  }  
  if (l->B.x[0]==0){
    if (l->B.x[1]!=(uint64_t)-1){
      if((l->B.x[1]>=l->dimx)&&(l->dimy==1)){ return false; }
    }
  }else{
    if( (l->B.x[0]!=(uint64_t)-1) ){
      if(( (l->dimx<l->B.x[0]||l->dimx>=g.size-l->B.x[0])
        && (l->dimy==0 || l->dimy==g.size-1)
      )){ return 0; } 
      if ((  (l->dimx==g.size-1)&&((l->dimy<=l->B.x[0])||
          l->dimy>=g.size-l->B.x[0]))){
        return 0;
      } 
    }
  }
  l->B.x[l->dimx]=l->dimy;                    //xは行 yは列
  uint64_t row=UINT64_C(1)<<l->dimx;
  uint64_t down=UINT64_C(1)<<l->dimy;
  uint64_t left=UINT64_C(1)<<(g.size-1-l->dimx+l->dimy); //右上から左下
  uint64_t right=UINT64_C(1)<<(l->dimx+l->dimy);       // 左上から右下
  if((l->B.row&row)||(l->B.down&down)||(l->B.left&left)||(l->B.right&right)){ return false; }     
  l->B.row|=row; l->B.down|=down; l->B.left|=left; l->B.right|=right;
  return true;
}
//対称解除法
void carryChain_symmetry(void* args)
{
  Local *l=(Local *)args;
  // 対称解除法 
  unsigned const int ww=(g.size-2)*(g.size-1)-1-l->w;
  unsigned const int w2=(g.size-2)*(g.size-1)-1;
  // # 対角線上の反転が小さいかどうか確認する
  if((l->s==ww)&&(l->n<(w2-l->e))){ return ; }
  // # 垂直方向の中心に対する反転が小さいかを確認
  if((l->e==ww)&&(l->n>(w2-l->n))){ return; }
  // # 斜め下方向への反転が小さいかをチェックする
  if((l->n==ww)&&(l->e>(w2-l->s))){ return; }
  // 枝刈り １行目が角の場合回転対称チェックせずCOUNT8にする
  if(l->B.x[0]==0){ 
    l->COUNTER[l->COUNT8]+=solve(l->B.row>>2,
    l->B.left>>4,((((l->B.down>>2)|(~0<<(g.size-4)))+1)<<(g.size-5))-1,(l->B.right>>4)<<(g.size-5));
    return ;
  }
  // n,e,s==w の場合は最小値を確認する。右回転で同じ場合は、
  // w=n=e=sでなければ値が小さいのでskip  w=n=e=sであれば90度回転で同じ可能性
  if(l->s==l->w){ if((l->n!=l->w)||(l->e!=l->w)){ return; } 
    l->COUNTER[l->COUNT2]+=solve(l->B.row>>2,
    l->B.left>>4,((((l->B.down>>2)|(~0<<(g.size-4)))+1)<<(g.size-5))-1,(l->B.right>>4)<<(g.size-5));
    return;
  }
  // e==wは180度回転して同じ 180度回転して同じ時n>=sの時はsmaller?
  if((l->e==l->w)&&(l->n>=l->s)){ if(l->n>l->s){ return; } 
    l->COUNTER[l->COUNT4]+=solve(l->B.row>>2,
    l->B.left>>4,((((l->B.down>>2)|(~0<<(g.size-4)))+1)<<(g.size-5))-1,(l->B.right>>4)<<(g.size-5));
    return;
  }
  l->COUNTER[l->COUNT8]+=solve(l->B.row>>2,
  l->B.left>>4,((((l->B.down>>2)|(~0<<(g.size-4)))+1)<<(g.size-5))-1,(l->B.right>>4)<<(g.size-5));
  return;
}
// pthread run()
void thread_run(void* args)
{
  Local *l=(Local *)args;

  // memcpy(&l->B,&l->wB,sizeof(Board));       // B=wB;
  l->B=l->wB;
  l->dimx=0; l->dimy=g.pres_a[l->w]; 
  //if(!placement(l)){ continue; } 
  if(!placement(l)){ return; } 
  l->dimx=1; l->dimy=g.pres_b[l->w]; 
  // if(!placement(l)){ continue; } 
  if(!placement(l)){ return; } 
  //２ 左２行に置く
  // memcpy(&l->nB,&l->B,sizeof(Board));       // nB=B;
  l->nB=l->B;
  for(l->n=l->w;l->n<(g.size-2)*(g.size-1)-l->w;++l->n){
    // memcpy(&l->B,&l->nB,sizeof(Board));     // B=nB;
    l->B=l->nB;
    l->dimx=g.pres_a[l->n]; l->dimy=g.size-1; 
    if(!placement(l)){ continue; } 
    l->dimx=g.pres_b[l->n]; l->dimy=g.size-2; 
    if(!placement(l)){ continue; } 
    // ３ 下２行に置く
    // memcpy(&l->eB,&l->B,sizeof(Board));     // eB=B;
    l->eB=l->B;
    for(l->e=l->w;l->e<(g.size-2)*(g.size-1)-l->w;++l->e){
      // memcpy(&l->B,&l->eB,sizeof(Board));   // B=eB;
      l->B=l->eB;
      l->dimx=g.size-1; l->dimy=g.size-1-g.pres_a[l->e]; 
      if(!placement(l)){ continue; } 
      l->dimx=g.size-2; l->dimy=g.size-1-g.pres_b[l->e]; 
      if(!placement(l)){ continue; } 
      // ４ 右２列に置く
      // memcpy(&l->sB,&l->B,sizeof(Board));   // sB=B;
      l->sB=l->B;
      for(l->s=l->w;l->s<(g.size-2)*(g.size-1)-l->w;++l->s){
        // memcpy(&l->B,&l->sB,sizeof(Board)); // B=sB;
        l->B=l->sB;
        l->dimx=g.size-1-g.pres_a[l->s]; l->dimy=0; 
        if(!placement(l)){ continue; } 
        l->dimx=g.size-1-g.pres_b[l->s]; l->dimy=1; 
        if(!placement(l)){ continue; } 
        // 対称解除法
        carryChain_symmetry(l);
      } //w
    } //e
  } //n
}
// チェーンのビルド
void buildChain()
{
  Local l[(g.size/2)*(g.size-3)];

  // カウンターの初期化
  l->COUNT2=0; l->COUNT4=1; l->COUNT8=2;
  l->COUNTER[l->COUNT2]=l->COUNTER[l->COUNT4]=l->COUNTER[l->COUNT8]=0;
  // Board の初期化 nB,eB,sB,wB;
  l->B.row=l->B.down=l->B.left=l->B.right=0;
  // Board x[]の初期化
  for(unsigned int i=0;i<g.size;++i){ l->B.x[i]=-1; }
  //１ 上２行に置く
  // memcpy(&l->wB,&l->B,sizeof(Board));         // wB=B;
  l->wB=l->B;
  for(l->w=0;l->w<=(unsigned)(g.size/2)*(g.size-3);++l->w){
    thread_run(&l);
  } //w
  /**
   * 集計
   */
  UNIQUE= l->COUNTER[l->COUNT2]+
          l->COUNTER[l->COUNT4]+
          l->COUNTER[l->COUNT8];
  TOTAL=  l->COUNTER[l->COUNT2]*2+
          l->COUNTER[l->COUNT4]*4+
          l->COUNTER[l->COUNT8]*8;
}
// チェーンのリストを作成
void listChain()
{
  unsigned int idx=0;
  for(unsigned int a=0;a<(unsigned)g.size;++a){
    for(unsigned int b=0;b<(unsigned)g.size;++b){
      if(((a>=b)&&(a-b)<=1)||((b>a)&&(b-a)<=1)){ continue; }
      g.pres_a[idx]=a;
      g.pres_b[idx]=b;
      ++idx;
    }
  }
}
// キャリーチェーン
void carryChain()
{
  listChain();  //チェーンのリストを作成
  buildChain(); // チェーンのビルド
  // calcChain(&l);  // 集計
}
unsigned int board[MAX];  //ボード配列
unsigned int down[MAX];   //ポストフラグ/ビットマップ/ミラー
unsigned int left[MAX];   //ポストフラグ/ビットマップ/ミラー
unsigned int right[MAX];  //ポストフラグ/ビットマップ/ミラー
unsigned int bitmap[MAX]; //ミラー
unsigned long COUNT2=0;   //ミラー/対象解除法
unsigned long COUNT4=0;   //対象解除法
unsigned long COUNT8=0;   //対象解除法
unsigned int BOUND1=0;    //対象解除法
unsigned int BOUND2=0;    //対象解除法
unsigned int SIDEMASK=0;  //対象解除法
unsigned int LASTMASK=0;  //対象解除法
unsigned int TOPBIT=0;    //対象解除法
unsigned int ENDBIT=0;    //対象解除法
// 対象解除法
void symmetryOps(unsigned int size)
{
  /**
  ２．クイーンが右上角以外にある場合、
  (1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルか
  ら180度回転)させても、さらに90度回転(オリジナルから270度回転)させてもオリ
  ジナルと同型になる。
  こちらに該当するユニーク解が属するグループの要素数は、左右反転させたパター
  ンを加えて２個しかありません。
  */
  if(board[BOUND2]==1){
    unsigned int ptn;
    unsigned int own;
    for(ptn=2,own=1;own<size;++own,ptn<<=1){
      unsigned int bit;
      unsigned int you;
      for(bit=1,you=size-1;(board[you]!=ptn)&&board[own]>=bit;--you){
        bit<<=1;
      }
      if(board[own]>bit){
        return ;
      }
      if(board[own]<bit){
        break;
      }
    }//end for
    // ９０度回転して同型なら１８０度回転しても２７０度回転しても同型である
    if(own>size-1){
      COUNT2++;
      return ;
    }//end if
  }//end if
  /**
  ２．クイーンが右上角以外にある場合、
    (2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナル
    とは異なる。ただし、180度回転させた場合はオリジナルと同型になることも有り得
    る。こちらに該当するユニーク解が属するグループの要素数は、180度回転させて同
    型になる場合は４個(左右反転×縦横回転)
   */
  //１８０度回転
  if(board[size-1]==ENDBIT){
    unsigned int you;
    unsigned int own;
    for(you=size-1-1,own=1;own<=size-1;++own,--you){
      unsigned int bit;
      unsigned int ptn;
      for(bit=1,ptn=TOPBIT;(ptn!=board[you])&&(board[own]>=bit);ptn>>=1){
        bit<<=1;
      }
      if(board[own]>bit){
        return ;
      }
      if(board[own]<bit){
        break;
      }
    }//end for
    //９０度回転が同型でなくても１８０度回転が同型であることもある
    if(own>size-1){
      COUNT4++;
      return ;
    }
  }//end if
  /**
  ２．クイーンが右上角以外にある場合、
    (3)180度回転させてもオリジナルと異なる場合は、８個(左右反転×縦横回転×上下反転)
  */
  //２７０度回転
  if(board[BOUND1]==TOPBIT){
    unsigned int ptn;
    unsigned int own;
    unsigned int you;
    unsigned int bit;
    for(ptn=TOPBIT>>1,own=1;own<=size-1;++own,ptn>>=1){
      for(bit=1,you=0;(board[you]!=ptn)&&(board[own]>=bit);++you){
        bit<<=1;
      }
      if(board[own]>bit){
        return ;
      }
      if(board[own]<bit){
        break;
      }
    }//end for
  }//end if
  COUNT8++;
}
//
void symmetry_backTrack_NR(unsigned int size,unsigned int row,unsigned int _left,unsigned int _down,unsigned int _right)
{
  unsigned int mask=(1<<size)-1;
  left[row]=_left;
  down[row]=_down;
  right[row]=_right;
  bitmap[row]=mask&~(left[row]|down[row]|right[row]);
  while(row>0){
    if(bitmap[row]>0){
      if(row<BOUND1){ //上部サイド枝刈り
        bitmap[row]|=SIDEMASK;
        bitmap[row]^=SIDEMASK;
      }else if(row==BOUND2){ //下部サイド枝刈り
        if((down[row]&SIDEMASK)==0){
          row--; 
        }
        if((down[row]&SIDEMASK)!=SIDEMASK){
          bitmap[row]&=SIDEMASK;
        }
      }
      unsigned int save_bitmap=bitmap[row];
      unsigned int bit=-bitmap[row]&bitmap[row];
      bitmap[row]^=bit;
      board[row]=bit; //Qを配置
      if((bit&mask)!=0){
        if(row==(size-1)){
          if( (save_bitmap&LASTMASK)==0){
            symmetryOps(size);  //対象解除法
          }
          row--;
        }else{
          unsigned int n=row++;
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
  }//end while
}
void symmetry_backTrack_corner_NR(unsigned int size,unsigned int row,unsigned int _left,unsigned int _down, unsigned int _right)
{
  unsigned int mask=(1<<size)-1;
  unsigned int bit=0;
  left[row]=_left;
  down[row]=_down;
  right[row]=_right;
  bitmap[row]=mask&~(left[row]|down[row]|right[row]);
  while(row>=2){
    if(row<BOUND1){
      // bitmap[row]=bitmap[row]|2;
      // bitmap[row]=bitmap[row]^2;
      bitmap[row]&=~2;
    }
    if(bitmap[row]>0){
      bit=-bitmap[row]&bitmap[row];
      bitmap[row]^=bit;
      if(row==(size-1)){
        COUNT8++;
        row--;
      }else{
        unsigned int n=row++;
        left[row]=(left[n]|bit)<<1;
        down[row]=(down[n]|bit);
        right[row]=(right[n]|bit)>>1;
        board[row]=bit; //Qを配置
        //クイーンが配置可能な位置を表す
        bitmap[row]=mask&~(left[row]|down[row]|right[row]);
      }
    }else{
      row--;
    }
  }//end while
}
// 対象解除法 非再帰版
void symmetry_NR(unsigned int size)
{
  TOTAL=UNIQUE=COUNT2=COUNT4=COUNT8=0;
  unsigned int mask=(1<<size)-1;
  unsigned int bit=0;
  TOPBIT=1<<(size-1);
  ENDBIT=SIDEMASK=LASTMASK=0;
  BOUND1=2;
  BOUND2=0;
  board[0]=1;
  while(BOUND1>1&&BOUND1<size-1){
    if(BOUND1<size-1){
      bit=1<<BOUND1;
      board[1]=bit;   //２行目にQを配置
      //角にQがあるときのバックトラック
      symmetry_backTrack_corner_NR(size,2,(2|bit)<<1,1|bit,(2|bit)>>1);
    }
    BOUND1++;
  }
  TOPBIT=1<<(size-1);
  ENDBIT=TOPBIT>>1;
  SIDEMASK=TOPBIT|1;
  LASTMASK=TOPBIT|1;
  BOUND1=1;
  BOUND2=size-2;
  while(BOUND1>0 && BOUND2<size-1 && BOUND1<BOUND2){
    if(BOUND1<BOUND2){
      bit=1<<BOUND1;
      board[0]=bit;   //Qを配置
      //角にQがないときのバックトラック
      symmetry_backTrack_NR(size,1,bit<<1,bit,bit>>1);
    }
    BOUND1++;
    BOUND2--;
    ENDBIT=ENDBIT>>1;
    LASTMASK=LASTMASK<<1|LASTMASK|LASTMASK>>1;
  }//ene while
  UNIQUE=COUNT2+COUNT4+COUNT8;
  TOTAL=COUNT2*2+COUNT4*4+COUNT8*8;
}
// 再帰 角にQがないときのバックトラック
void symmetry_backTrack(unsigned int size,unsigned int row,unsigned int left,unsigned int down,unsigned int right)
{
  unsigned int mask=(1<<size)-1;
  unsigned int bitmap=mask&~(left|down|right);
  if(row==(size-1)){
    if(bitmap){
      if( (bitmap&LASTMASK)==0){
        board[row]=bitmap;  //Qを配置
        symmetryOps(size);    //対象解除
      }
    }
  }else{
    if(row<BOUND1){
      bitmap=bitmap|SIDEMASK;
      bitmap=bitmap^SIDEMASK;
    }else{
      if(row==BOUND2){
        if((down&SIDEMASK)==0){
          return;
        }
        if( (down&SIDEMASK)!=SIDEMASK){
          bitmap=bitmap&SIDEMASK;
        }
      }
    }
    while(bitmap){
      unsigned int bit=-bitmap&bitmap;
      bitmap=bitmap^bit;
      board[row]=bit;
      symmetry_backTrack(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }//end while
  }//end if
}
// 再帰 角にQがあるときのバックトラック
void symmetry_backTrack_corner(unsigned int size,unsigned int row,unsigned int left,unsigned int down,unsigned int right)
{
  unsigned int mask=(1<<size)-1;
  unsigned int bitmap=mask&~(left|down|right);
  unsigned int bit=0;
  if(row==(size-1)){
    if(bitmap){
      board[row]=bitmap;
      COUNT8++;
    }
  }else{
    if(row<BOUND1){   //枝刈り
      bitmap=bitmap|2;
      bitmap=bitmap^2;
    }
    while(bitmap){
      bit=-bitmap&bitmap;
      bitmap=bitmap^bit;
      board[row]=bit;   //Qを配置
      symmetry_backTrack_corner(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
}
// 対象解除法 再帰版
void symmetry_R(unsigned int size)
{
  TOTAL=UNIQUE=COUNT2=COUNT4=COUNT8=0;
  unsigned int mask=(1<<size)-1;
  unsigned int bit=0;
  TOPBIT=1<<(size-1);
  ENDBIT=LASTMASK=SIDEMASK=0;
  BOUND1=2;
  BOUND2=0;
  board[0]=1;
  while(BOUND1>1 && BOUND1<size-1){
    if(BOUND1<size-1){
      bit=1<<BOUND1;
      board[1]=bit;   //２行目にQを配置
      //角にQがあるときのバックトラック
      symmetry_backTrack_corner(size,2,(2|bit)<<1,1|bit,(2|bit)>>1);
    }
    BOUND1++;
  }//end while
  TOPBIT=1<<(size-1);
  ENDBIT=TOPBIT>>1;
  SIDEMASK=TOPBIT|1;
  LASTMASK=TOPBIT|1;
  BOUND1=1;
  BOUND2=size-2;
  while(BOUND1>0 && BOUND2<size-1 && BOUND1<BOUND2){
    if(BOUND1<BOUND2){
      bit=1<<BOUND1;
      board[0]=bit;   //Qを配置
      //角にQがないときのバックトラック
      symmetry_backTrack(size,1,bit<<1,bit,bit>>1);
    }
    BOUND1++;
    BOUND2--;
    ENDBIT=ENDBIT>>1;
    LASTMASK=LASTMASK<<1|LASTMASK|LASTMASK>>1;
  }//ene while
  UNIQUE=COUNT2+COUNT4+COUNT8;
  TOTAL=COUNT2*2+COUNT4*4+COUNT8*8;
}
//ミラー処理部分 非再帰版
void mirror_solve_NR(unsigned int size,unsigned int row,unsigned int _left,unsigned int _down, unsigned int _right)
{
  unsigned int mask=(1<<size)-1;
  unsigned int bit=0;
  left[row]=_left;
  down[row]=_down;
  right[row]=_right;
  bitmap[row]=mask&~(left[row]|down[row]|right[row]);
  while(row>0){
    if(bitmap[row]>0){
      bit=-bitmap[row]&bitmap[row];
      bitmap[row]=bitmap[row]^bit;
      board[row]=bit;
      if(row==(size-1)){
        COUNT2++;
        row--;
      }else{
        unsigned int n=row++;
        left[row]=(left[n]|bit)<<1;
        down[row]=(down[n]|bit);
        right[row]=(right[n]|bit)>>1;
        board[row]=bit;   //Qを配置
        //クイーンが配置可能な位置を表す
        bitmap[row]=mask&~(left[row]|down[row]|right[row]);
      }
    }else{
      row--;
    }
  }
}
// ミラー 非再帰版
void mirror_NR(unsigned int size)
{
  COUNT2=0;
  unsigned int mask=(1<<size)-1;
  unsigned int bit=0;
  unsigned int limit=size%2 ? size/2-1 : size/2;
  for(unsigned int i=0;i<size/2;++i){ //奇数でも偶数でも通過
    bit=1<<i;
    board[0]=bit;             //１行目にQを置く
    mirror_solve_NR(size,1,bit<<1,bit,bit>>1);
  }
  if(size%2){                 //奇数で通過
    bit=1<<(size-1)/2;
    board[0]=1<<(size-1)/2;   //１行目の中央にQを配置
    unsigned int left=bit<<1;
    unsigned int down=bit;
    unsigned int right=bit>>1;
    for(unsigned int i=0;i<limit;++i){
      bit=1<<i;
      mirror_solve_NR(size,2,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
  TOTAL=COUNT2<<1;    //倍にする
}
//ミラーロジック 再帰版
void mirror_solve_R(unsigned int size,unsigned int row,unsigned int left,unsigned int down,unsigned int right)
{
  unsigned int mask=(1<<size)-1;
  unsigned int bit=0;
  if(row==size){
    COUNT2++;
  }else{
    for(unsigned int bitmap=mask&~(left|down|right);bitmap;bitmap=bitmap&~bit){
      bit=-bitmap&bitmap;
      board[row]=bit;
      mirror_solve_R(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
}
// ミラー 再帰版
void mirror_R(unsigned int size)
{
  COUNT2=0;
  unsigned int mask=(1<<size)-1;
  unsigned int bit=0;
  unsigned int limit=size%2 ? size/2-1 : size/2;
  for(unsigned int i=0;i<size/2;++i){
    bit=1<<i;
    board[0]=bit;           //１行目にQを置く
    mirror_solve_R(size,1,bit<<1,bit,bit>>1);
  }
  if(size%2){               //奇数で通過
    bit=1<<(size-1)/2;
    board[0]=1<<(size-1)/2; //１行目の中央にQを配置
    unsigned int left=bit<<1;
    unsigned int down=bit;
    unsigned int right=bit>>1;
    for(unsigned int i=0;i<limit;++i){
      bit=1<<i;
      mirror_solve_R(size,2,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
  TOTAL=COUNT2<<1;    //倍にする
}
// ビットマップ 非再帰版
void bitmap_NR(unsigned int size,int row)
{
  unsigned int mask=(1<<size)-1;
  unsigned int bitmap[size];
  unsigned int bit=0;
  bitmap[row]=mask;
  while(row>-1){
    if(bitmap[row]>0){
      bit=-bitmap[row]&bitmap[row];//一番右のビットを取り出す
      bitmap[row]=bitmap[row]^bit;//配置可能なパターンが一つずつ取り出される
      board[row]=bit;
      if(row==(size-1)){
        TOTAL++;
        row--;
      }else{
        unsigned int n=row++;
        left[row]=(left[n]|bit)<<1;
        down[row]=down[n]|bit;
        right[row]=(right[n]|bit)>>1;
        board[row]=bit;
        //クイーンが配置可能な位置を表す
        bitmap[row]=mask&~(left[row]|down[row]|right[row]);
      }
    }else{
      row--;
    }
  }//end while
}
// ビットマップ 再帰版
void bitmap_R(unsigned int size,unsigned int row,unsigned int left,unsigned int down, unsigned int right)
{
  unsigned int mask=(1<<size)-1;
  unsigned int bit=0;
  if(row==size){
    TOTAL++;
  }else{
    for(unsigned int bitmap=mask&~(left|down|right);bitmap;bitmap=bitmap&~bit){
      bit=-bitmap&bitmap;
      board[row]=bit;
      bitmap_R(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
}
// ポストフラグ 非再帰版
void postFlag_NR(unsigned int size,int row)
{
  // １．非再帰は初期化が必要
  for(unsigned int i=0;i<size;++i){
    board[i]=-1;
  }
  // ２．再帰で呼び出される関数内を回す処理
  while(row>-1){
    unsigned int matched=0; //クイーンを配置したか
    // ３．再帰処理のループ部分
    // 非再帰では過去の譜石を記憶するためにboard配列を使う
    for(unsigned int col=board[row]+1;col<size;++col){
      if(!down[col]
          && !right[col-row+size-1]
          && !left[col+row]){
        // unsigned int dix=col;
        // unsigned int rix=row-col+(size-1);
        // unsigned int lix=row+col;
        /** バックトラックではここで効きをチェックしていた
        check_backTracking "$row";  # 効きをチェック
        */
        // 効きとしてフラグをfalseにする
        if(board[row]!=-1){
          down[board[row]]=0;
          right[board[row]-row+(size-1)]=0;
          left[board[row]+row]=0;
        }
        board[row]=col; //クイーンを配置
        // 効きを開放（trueに）する
        down[col]=1;
        right[col-row+(size-1)]=1;
        left[col+row]=1;
        matched=1;
        break;
      } //end if
    }//end for
    // ４．配置したら実行したい処理
    if(matched){
      row++;
      // ５．最下部まで到達したときの処理
      if(row==size){
        row--;
        /** ブルートフォースではここで効きをチェックしていた
        // check_bluteForce "$size";   # 効きをチェック
        */
        TOTAL++;
      }
    // ６．配置できなくてバックトラックしたい時の処理
    }else{
      if(board[row]!=-1){
        down[board[row]]=0;
        right[board[row]-row+(size-1)]=0;
        left[board[row]+row]=0;
        board[row]=-1;
      }
      row--;
    }
  }//end while
}
// ポストフラグ 再帰版
void postFlag_R(unsigned int size,unsigned int row)
{
  if(row==size){
    TOTAL++;
  }else{
    for(unsigned int col=0;col<size;++col){
      board[row]=col;
      if(down[col]==0
          && right[row-col+size-1]==0
          && left[row+col]==0){
        down[col]=1;
        right[row-col+(size-1)]=1;
        left[row+col]=1;
        postFlag_R(size,row+1);
        down[col]=0;
        right[row-col+(size-1)]=0;
        left[row+col]=0;
      }//end if
    }//end for
  }//end if
}
// バックトラック　効き筋をチェック
int check_backTracking(unsigned int row)
{
  for(unsigned int i=0;i<row;++i){
    unsigned int val=0;
    if(board[i]>=board[row]){
      val=board[i]-board[row];
    }else{
      val=board[row]-board[i];
    }
    if(board[i]==board[row]||val==(row-i)){
      return 0;
    }
  }
  return 1;
}
// バックトラック 非再帰版
void backTracking_NR(unsigned int size,int row)
{
  // １．非再帰は初期化が必要
  for(unsigned int i=0;i<size;++i){
    board[i]=-1;
  }
  // ２．再帰で呼び出される関数内を回す処理
  while(row>-1){
    unsigned int matched=0;   //クイーンを配置したか
    // ３．再帰処理のループ部分
    for(unsigned int col=board[row]+1;col<size;++col){
      board[row]=col;   // クイーンを配置
      // 効きをチェック
      if(check_backTracking(row)==1){
        matched=1;
        break;
      } // end if
    } // end for
    // ４．配置したら実行したい処理
    if(matched){
      row++;
      // ５．最下部まで到達したときの処理
      if(row==size){
        row--;
        TOTAL++;
      }
    // ６．配置できなくてバックトラックしたい時の処理
    }else{
      if(board[row]!=-1){
        board[row]=-1;
      }
      row--;
    }
  } //end while
}
// バックトラック 再帰版
void backTracking_R(unsigned int size,unsigned int row)
{
  if(row==size){
    TOTAL++;
  }else{
    for(unsigned int col=0;col<size;++col){
      board[row]=col;
      if(check_backTracking(row)==1){
        backTracking_R(size,row+1);
      }
    }// end for
  }//end if
}
// ブルートフォース 効き筋をチェック
int check_bluteForce()
{
  unsigned int size=g.size; 
  for(unsigned int r=1;r<size;++r){
    unsigned int val=0;
    for(unsigned int i=0;i<r;++i){
      if(board[i]>=board[r]){
        val=board[i]-board[r];
      }else{
        val=board[r]-board[i];
      }
      if(board[i]==board[r]||val==(r-i)){
        return 0;
      }
    }
  }
  return 1;
}
//ブルートフォース 非再帰版
void bluteForce_NR(unsigned int size,int row)
{
  // １．非再帰は初期化が必要
  for(unsigned int i=0;i<size;++i){
    board[i]=-1;
  }
  // ２．再帰で呼び出される関数内を回す処理
  while(row>-1){
    unsigned int matched=0;   //クイーンを配置したか
    // ３．再帰処理のループ部分
    // 非再帰では過去の譜石を記憶するためにboard配列を使う
    for(unsigned int col=board[row]+1;col<size;++col){
      board[row]=col;
      matched=1;
      break;
    }
    // ４．配置したら実行したい処理
    if(matched){
      row++;
      // ５．最下部まで到達したときの処理';
      if(row==size){
        row--;
        // 効きをチェック
        if(check_bluteForce()==1){
          TOTAL++;
        }
      }
      // ６．配置できなくてバックトラックしたい処理
    }else{
      if(board[row]!=-1){
        board[row]=-1;
      }
      row--;
    } // end if
  }//end while
}
//ブルートフォース 再帰版
void bluteForce_R(unsigned int size,int row)
{
  if(row==size){
    if(check_bluteForce()==1){
      TOTAL++; // グローバル変数
    }
  }else{
    for(int col=0;col<size;++col){
      board[row]=col;
      bluteForce_R(size,row+1);
    }
  }
}
//メインメソッド
int main(int argc,char** argv)
{
  bool cpu=false,cpur=false;
  int argstart=2;
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else{ cpur=true;}
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-r|-g]\n",argv[0]);
    printf("  -c: CPU Without recursion\n");
    printf("  -r: CPUR Recursion\n");
    printf("  -g: GPU Without Recursion\n");
  }
  printf("７．キャリーチェーン\n");
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  clock_t st;           //速度計測用
  char t[20];           //hh:mm:ss.msを格納
  unsigned int min=4;
  unsigned int targetN=21;
  // sizeはグローバル
  for(unsigned int size=min;size<=targetN;++size){
    TOTAL=UNIQUE=0; 
    st=clock();
    g.size=size;
    if(cpu){  // 非再帰
      carryChain();           //７．キャリーチェーン
      // symmetry_NR(size);      //６．対象解除法
      // mirror_NR(size);        //５．ミラー
      // bitmap_NR(size,0);      //４．ビットマップ
      // postFlag_NR(size,0);    //３．ポストフラグ
      // backTracking_NR(size,0);//２．バックトラック
      // bluteForce_NR(size,0);  //１．ブルートフォース
      // carryChain();
    }else{    // 再帰
      carryChain();           //７．キャリーチェーン
      // symmetry_R(size);       //６．対象解除法
      // mirror_R(size);         //５．ミラー
      // bitmap_R(size,0,0,0,0); //４．ビットマップ
      // postFlag_R(size,0);     //３．ポストフラグ
      // backTracking_R(size,0); //２．バックトラック
      // bluteForce_R(size,0);   //１．ブルートフォース
    }
    TimeFormat(clock()-st,t);
    printf("%2d:%13lld%16lld%s\n",size,TOTAL,UNIQUE,t);
  }
  return 0;
}
