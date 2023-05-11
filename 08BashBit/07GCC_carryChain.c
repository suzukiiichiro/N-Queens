
// $ gcc -Wall -W -O3 07GCC_carryChain.c -o 07GCC && ./07GCC


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>

#define MAX 27
int size;
long TOTAL=0; 
long UNIQUE=0;
int COUNT8=0;
int COUNT4=0;
int COUNT2=0;
int pres_a[930];
int pres_b[930];
int w,s,e,n;
typedef struct{
  int row;
  int down;
  int left;
  int right;
  int x[MAX];
}Board ;
Board B;
//
long solve(int row,int left,int down,int right)
{
  if(down+1==0){ return  1; }
  while((row&1)!=0) { 
    row>>=1;
    left<<=1;
    right>>=1;
  }
  row>>=1;
  int total=0;
  int bit;
  for(int bitmap=~(left|down|right);bitmap!=0;bitmap^=bit){
    bit=bitmap&-bitmap;
    total+=solve(row,(left|bit)<<1,down|bit,(right|bit)>>1);
  }
  return total;
} 
//
bool placement(int dimx,int dimy)
{
  if(B.x[dimx]==dimy){ return true;  }  
  /** 
    Qが角にある場合の枝刈り
    Qが角にある場合は2行目のクイーンの位置t_x[1]がBOUND1
    BOUND1行目までは2列目にクイーンを置くことはできない
  */ 
  if ((B.x[1]!=-1)&&(B.x[0]==0)){
    // bitmap=$(( bitmap|2 ));
    // bitmap=$(( bitmap^2 ));
    // 上と下は同じ趣旨
    if((B.x[1]>=dimx)&&(dimy==1)){ return false; }
  }
  /**
    Qが角にない場合の上部サイド枝刈り
    if ((row<BOUND1));then        
      bitmap=$(( bitmap|SIDEMASK ));
      bitmap=$(( bitmap^=SIDEMASK ));
    BOUND1はt_x[0]
  **/
  if(( (B.x[0]!=-1) && (B.x[0]!=0) )){
    if (((dimx<B.x[0])&&(dimy==0||dimy==size-1))){
      return 0;
    } 
  }
  B.x[dimx]=dimy;                    //xは行 yは列
  int row=1<<dimx;
  int down=1<<dimy;
  int left=1<<(size-1-dimx+dimy);    //右上から左下
  int right=1<<(dimx+dimy);          // 左上から右下
  if((B.row&row)||(B.down&down)||(B.left&left)||(B.right&right)){ return false; }     
  B.row|=row; B.down|=down; B.left|=left; B.right|=right;
  return true;
}
//
void carryChainSymmetry()
{
  int ww=(size-2)*(size-1)-1-w;
  int w2=(size-2)*(size-1)-1;
  if((s==ww)&&(n<(w2-e))){ return; }
  if((e==ww)&&(n>(w2-n))){ return; }
  if((n==ww)&&(e>(w2-s))){ return; }
  if(s==w){ 
    if((n!=w)||(e!=w)){ return; } 
    COUNT2 += solve(B.row >> 2,
        B.left>>4,
        ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
        (B.right>>4)<<(size-5));
    return ;
  }
  if((e==w)&&(n>=s)){ 
    if(n>s){ return; } 
    COUNT4 += solve(B.row >> 2,
        B.left>>4,
        ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
        (B.right>>4)<<(size-5));
    return;
  }
  COUNT8 += solve(B.row >> 2,
      B.left>>4,
      ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
      (B.right>>4)<<(size-5));
}
//
void buildChain()
{
  Board wB=B;
  for(w=0;w<=(size/2)*(size-3);w++){
    B=wB;
    B.row=B.down=B.left=B.right=0;
    for(int i=0;i<size;i++){ B.x[i]=-1; }
    if(!placement(0,pres_a[w])){ continue; } 
    if(!placement(1,pres_b[w])){ continue; }
    int lSize=(size-2)*(size-1)-w;
    Board nB=B;//２ 左２行に置く
    for(n=w;n<lSize;n++){
      B=nB;
      if(!placement(pres_a[n],size-1)){ continue; }
      if(!placement(pres_b[n],size-2)){ continue; }
      Board eB=B;// ３ 下２行に置く
      for(e=w;e<lSize;e++){
        B=eB;
        if(!placement(size-1,size-1-pres_a[e])){ continue; }
        if(!placement(size-2,size-1-pres_b[e])){ continue; }
        Board sB=B;// ４ 右２列に置く
        for(s=w;s<lSize;s++){
          B=sB;
          if(!placement(size-1-pres_a[s],0)){ continue; }
          if(!placement(size-1-pres_b[s],1)){ continue; }
          carryChainSymmetry();//対象解除法
          continue;
        }
      }    
    }
  }
}
//
void initChain()
{
  int idx=0;
  for(int a=0;a<size;a++){
    for(int b=0;b<size;b++){
      if(((a>=b)&&(a-b)<=1)||((b>a)&&(b-a)<=1)){ continue; }     
      pres_a[idx]=a;
      pres_b[idx]=b;
      idx++;
    }
  }
}
//
void carryChain()
{
  initChain() ;     // チェーンの初期化
  buildChain() ;    // チェーンの構築 
  UNIQUE=COUNT2+COUNT4+COUNT8;// 集計
  TOTAL=COUNT2*2+COUNT4*4+COUNT8*8;
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
  int targetN=20;
  // sizeはグローバル
  for(size=min;size<=targetN;size++){
    TOTAL=UNIQUE=COUNT2=COUNT4=COUNT8=0;
    st=clock();
    if(cpu){
      carryChain();
    }else{
      carryChain();
    }
    TimeFormat(clock()-st,t);
    printf("%2d:%13ld%16ld%s\n",size,TOTAL,UNIQUE,t);
  }
  return 0;
}
