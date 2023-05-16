/**
 *
 * 兄とやること
 * １．ユニーク界が違う
 * ２．Cとの速度比較
 *
 *
 *
 * キャリーチェーンC言語版

 
 $ gcc -Wall -W -O3 07GCC_carryChain.c -o 07GCC && ./07GCC
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
12:        14200            1788            0.03
13:        73712            9237            0.08
14:       365596           45771            0.29
15:      2279184          285095            1.22
16:     14772512         1847425            6.68


 bash-3.2$ gcc -Wall -W -O3 GCC12.c && ./a.out -r
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
15:      2279184          285053            0.41
16:     14772512         1846955            2.66
17:     95815104        11977939           18.41
18:    666090624        83263591         2:14.44
19:   4968057848       621012754        17:06.46
*/



#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>

#define MAX 27
typedef unsigned long long uint64;
typedef struct{
  uint64 row;
  uint64 down;
  uint64 left;
  uint64 right;
  uint64 x[MAX];
}Board ;
Board B;
uint64 TOTAL=0; 
uint64 UNIQUE=0;
uint64 COUNTER[3];  //カウンター配列
unsigned int COUNT2=0; //配列用
unsigned int COUNT4=1; //配列用
unsigned int COUNT8=2; //配列用
// ボード外側２列を除く内側のクイーン配置処理
uint64 solve(uint64 row,uint64 left,uint64 down,uint64 right)
{
  if(down+1==0){ return  1; }
  while((row&1)!=0) { 
    row>>=1;
    left<<=1;
    right>>=1;
  }
  row>>=1;
  uint64 total=0;
  uint64 bit;
  for(uint64 bitmap=~(left|down|right);bitmap!=0;bitmap^=bit){
    bit=bitmap&-bitmap;
    total+=solve(row,(left|bit)<<1,down|bit,(right|bit)>>1);
  }
  return total;
} 
//
void process(int size,Board B,int sym)
{
  COUNTER[sym]+=solve(B.row>>2,
      B.left>>4,
      ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
      (B.right>>4)<<(size-5));
}
// クイーンの効きをチェック
bool placement(int size,int dimx,int dimy)
{
  if(B.x[dimx]==dimy){ return true;  }  
  /** 
  #
  #
  # 【枝刈り】Qが角にある場合の枝刈り
  #  ２．２列めにクイーンは置かない
  #  （１はcarryChainSymmetry()内にあります）
  #
  #  Qが角にある場合は、
  #  2行目のクイーンの位置 t_x[1]が BOUND1
  #  BOUND1行目までは2列目にクイーンを置けない
  */ 
  if (B.x[0]==0){
    if (B.x[1]!=-1){
      // bitmap=$(( bitmap|2 ));
      // bitmap=$(( bitmap^2 ));
      // 上と下は同じ趣旨
      if((B.x[1]>=dimx)&&(dimy==1)){ return false; }
    }
  }else{
    /**
    # 【枝刈り】Qが角にない場合
    #   １．上部サイド枝刈り
    #  if ((row<BOUND1));then        
    #    bitmap=$(( bitmap|SIDEMASK ));
    #    bitmap=$(( bitmap^=SIDEMASK ));
    #
    #  BOUND1はt_x[0]
    #
    #  ２．下部サイド枝刈り
    #  if ((row==BOUND2));then     
    #    if (( !(down&SIDEMASK) ));then
    #      return ;
    #    fi
    #    if (( (down&SIDEMASK)!=SIDEMASK ));then
    #      bitmap=$(( bitmap&SIDEMASK ));
    #    fi
    #  fi
    #
    #  ２．最下段枝刈り
    #  LSATMASKの意味は最終行でBOUND1以下または
    #  BOUND2以上にクイーンは置けないということ
    #  BOUND2はsize-t_x[0]
    #  if(row==sizeE){
    #    //if(!bitmap){
    #    if(bitmap){
    #      if((bitmap&LASTMASK)==0){
    **/
    if( (B.x[0]!=-1) ){
      if(( (dimx<B.x[0]||dimx>=size-B.x[0])
        && (dimy==0 || dimy==size-1)
      )){ return 0; } 
      if ((  (dimx==size-1)&&((dimy<=B.x[0])||
          dimy>=size-B.x[0]))){
        return 0;
      } 
    }
  }
  B.x[dimx]=dimy;                    //xは行 yは列
  uint64 row=1<<dimx;
  uint64 down=1<<dimy;
  uint64 left=1<<(size-1-dimx+dimy);    //右上から左下
  uint64 right=1<<(dimx+dimy);          // 左上から右下
  if((B.row&row)||(B.down&down)||(B.left&left)||(B.right&right)){ return false; }     
  B.row|=row; B.down|=down; B.left|=left; B.right|=right;
  return true;
}
// キャリーチェーン対象解除法
void carryChainSymmetry(int size,int w,int s,int e,int n)
{
  // # n,e,s=(N-2)*(N-1)-1-w の場合は最小値を確認する。
  int ww=(size-2)*(size-1)-1-w;
  int w2=(size-2)*(size-1)-1;
  // # 対角線上の反転が小さいかどうか確認する
  if((s==ww)&&(n<(w2-e))){ return; }
  // # 垂直方向の中心に対する反転が小さいかを確認
  if((e==ww)&&(n>(w2-n))){ return; }
  // # 斜め下方向への反転が小さいかをチェックする
  if((n==ww)&&(e>(w2-s))){ return; }
  /*
   枝刈り １行目が角の場合回転対称チェックせずCOUNT8にする
  **/
  if(B.x[0]==0){
    // COUNT8 += solve(B.row >> 2,
    //     B.left>>4,
    //     ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
    //     (B.right>>4)<<(size-5));
    process(size,B,COUNT8);
    return ;
  }
  /**
  # n,e,s==w の場合は最小値を確認する。
  # : '右回転で同じ場合は、
  # w=n=e=sでなければ値が小さいのでskip
  # w=n=e=sであれば90度回転で同じ可能性 ';
   */
  if(s==w){ 
    if((n!=w)||(e!=w)){ return; } 
    // COUNT2 += solve(B.row >> 2,
    //     B.left>>4,
    //     ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
    //     (B.right>>4)<<(size-5));
    process(size,B,COUNT2);
    return ;
  }
  // # : 'e==wは180度回転して同じ
  // # 180度回転して同じ時n>=sの時はsmaller?  ';
  if((e==w)&&(n>=s)){ 
    if(n>s){ return; } 
    // COUNT4 += solve(B.row >> 2,
    //     B.left>>4,
    //     ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
    //     (B.right>>4)<<(size-5));
    process(size,B,COUNT4);
    return;
  }
  // COUNT8 += solve(B.row >> 2,
  //     B.left>>4,
  //     ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
  //     (B.right>>4)<<(size-5));
  process(size,B,COUNT8);
  return ;
}
// キャリーチェーン
void carryChain(int size)
{
  int pres_a[930];
  int pres_b[930];
  int idx=0;
  for(int a=0;a<size;a++){
    for(int b=0;b<size;b++){
      if(((a>=b)&&(a-b)<=1)||((b>a)&&(b-a)<=1)){ continue; }
      pres_a[idx]=a;
      pres_b[idx]=b;
      idx++;
    }
  }
  //
  int w,s,e,n;
  Board wB=B;
  for(w=0;w<=(size/2)*(size-3);w++){
    B=wB;
    B.row=B.down=B.left=B.right=0;
    for(int i=0;i<size;i++){ B.x[i]=-1; }
    if(!placement(size,0,pres_a[w])){ continue; } 
    if(!placement(size,1,pres_b[w])){ continue; }
    int lSize=(size-2)*(size-1)-w;
    Board nB=B;//２ 左２行に置く
    for(n=w;n<lSize;n++){
      B=nB;
      if(!placement(size,pres_a[n],size-1)){ continue; }
      if(!placement(size,pres_b[n],size-2)){ continue; }
      Board eB=B;// ３ 下２行に置く
      for(e=w;e<lSize;e++){
        B=eB;
        if(!placement(size,size-1,size-1-pres_a[e])){ continue; }
        if(!placement(size,size-2,size-1-pres_b[e])){ continue; }
        Board sB=B;// ４ 右２列に置く
        for(s=w;s<lSize;s++){
          B=sB;
          if(!placement(size,size-1-pres_a[s],0)){ continue; }
          if(!placement(size,size-1-pres_b[s],1)){ continue; }
          carryChainSymmetry(size,w,s,e,n);//対象解除法
          continue;
        }
      }    
    }
  }
  UNIQUE=COUNTER[COUNT2]+COUNTER[COUNT4]+COUNTER[COUNT8];
  TOTAL=COUNTER[COUNT2]*2+COUNTER[COUNT4]*4+COUNTER[COUNT8]*8;
}
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
    printf("Usage: %s [-c|-g]\n",argv[0]);
    printf("  -c: CPU Without recursion\n");
    printf("  -r: CPUR Recursion\n");
  }
  printf("\n\n７．キャリーチェーン\n");
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  clock_t st;           //速度計測用
  char t[20];           //hh:mm:ss.msを格納
  int min=4;
  int targetN=21;
  // sizeはグローバル
  for(int size=min;size<=targetN;size++){
    TOTAL=UNIQUE=COUNTER[COUNT2]=COUNTER[COUNT4]=COUNTER[COUNT8]=0;
    st=clock();
    if(cpu){
      carryChain(size);
    }else{
      carryChain(size);
    }
    TimeFormat(clock()-st,t);
    printf("%2d:%13lld%16lld%s\n",size,TOTAL,UNIQUE,t);
  }
  return 0;
}
