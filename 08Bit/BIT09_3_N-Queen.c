/**
 CUDAで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

９−３．CPUR 再帰 ビットマップ＋対象解除＋q２７枝刈＋BackTrack1＋BackTrack2
 N:        Total       Unique        hh:mm:ss.ms
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.00
12:        14200            1788            0.01
13:        73712            9237            0.04
14:       365596           45771            0.23
15:      2279184          285095            1.40

 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#define THREAD_NUM		96
#define MAX 27
//変数宣言
long TOTAL=0;         //CPU,CPUR
long UNIQUE=0;        //CPU,CPUR
typedef unsigned long long uint64;
//typedef unsigned long uint64;
typedef struct{
  int bv;
  long down;
  long left;
  long right;
  int cnt;
  int x[MAX];
  int y[MAX];
}Board ;
//
Board *GBoard;
long bcnt=0;
unsigned int NONE=2;
unsigned int POINT=1;
unsigned int ROTATE=0;
long cnt[3];
long pre[3];

//hh:mm:ss.ms形式に処理時間を出力
void TimeFormat(clock_t utime,char *form)
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
//q27,bit93で内容が同じなので共通化する
bool board_placement(int si,int x,int y)
{
  //同じ場所に置くかチェック
  if(GBoard[bcnt].x[x]==y){
    //printf("Duplicate x:%d:y:%d\n",x,y);
    ////同じ場所に置くのはOK
    return true;  
  }
  GBoard[bcnt].x[x]=y;
  //xは行 yは列 p.N-1-x+yは右上から左下 x+yは左上から右下
  int bv=1<<x;
  long down=1<<y;
  GBoard[bcnt].y[x]=GBoard[bcnt].y[x]+down;
  long left=1<<(si-1-x+y);
  long right=1<<(x+y);
  if((GBoard[bcnt].bv&bv)||(GBoard[bcnt].down&down)||(GBoard[bcnt].left&left)||(GBoard[bcnt].right&right)){
    //printf("valid_false\n");
    return false;
  }     
  GBoard[bcnt].bv|=bv;
  GBoard[bcnt].down|=down;
  GBoard[bcnt].left|=left;
  GBoard[bcnt].right|=right;
  return true;
}
//
//q27,bit93で内容が同じなので共通化する
int symmetryOps_n27(int w,int e,int n,int s,int size)
{
  //// Check for minimum if n, e, s = (N-2)*(N-1)-1-w
  int ww=(size-2)*(size-1)-1-w;
  //新設
  int w2=(size-2)*(size-1)-1;
  //if(s==ww){
  if((s==ww)&&(n<(w2-e))){
    //check if flip about the up diagonal is smaller
    //if(n<(size-2)*(size-1)-1-e){
    //if(n<(w2-e)){
    return 3;
    //}
  }
  //if(e==ww){
  if((e==ww)&&(n>(w2-n))){
    //check if flip about the vertical center is smaller
    //if(n>(size-2)*(size-1)-1-n){
    //if(n>(w2-n)){
    return 3;       
    //}
  }
  //if(n==ww){
  if((n==ww)&&(e>(w2-s))){
    //// check if flip about the down diagonal is smaller
    //if(e>(size-2)*(size-1)-1-s){
    //if(e>(w2-s)){
    return 3;
    //}
  }
  if(s==w){
    if((n!=w)||(e!=w)){
      // right rotation is smaller unless  w = n = e = s
      //右回転で同じ場合w=n=e=sでなければ値が小さいのでskip
      return 3;
    }
    //w=n=e=sであれば90度回転で同じ可能性
    //この場合はミラーの2
    return 0;
  }
  if((e==w)&&(n>=s)){
    //e==wは180度回転して同じ
    if(n>s){
      //180度回転して同じ時n>=sの時はsmaller?
      return 3;
    }
    //この場合は4
    return 1;
  }
  return 2;   
}
//
long q27_countCompletions(int bv,long down,long left,long right)
{
  if(down+1 == 0){
    return  1;
  }
  while((bv&1) != 0) { // Column is covered by pre-placement
    bv >>= 1;//右に１ビットシフト
    left <<= 1;//left 左に１ビットシフト
    right >>= 1;//right 右に１ビットシフト
  }
  bv >>= 1;//１行下に移動する
  long  cnt = 0;// Column needs to be placed
  //bh:down bu:left bd:right
  //クイーンを置いていく
  //slotsはクイーンの置ける場所
  for(uint64  bitmap = ~(down|left|right); bitmap != 0;) {
    uint64 const  bit = bitmap & -bitmap;
    cnt   += q27_countCompletions(bv, down|bit, (left|bit) << 1, (right|bit) >> 1);
    bitmap ^= bit;
  }
  //途中でクイーンを置くところがなくなるとここに来る
  //printf("return_cnt:%d\n",cnt);
  return  cnt;
}
//CPUR 再帰版 ロジックメソッド
long bit93_countCompletions(int size, int row,int bv,long left,long down,long right){
  long cnt=0;
  uint64 bitmap=0;
  uint64 bit=0;
  //既にクイーンを置いている行はスキップする
  while((bv&1)!=0) {
    bv>>=1;//右に１ビットシフト
    left<<=1;//left 左に１ビットシフト
    right>>=1;//right 右に１ビットシフト  
    row++; 
  }
  bv>>=1;
  if(row==size){
    //TOTAL++;
      //UNIQUE++;       //ユニーク解を加算
      //TOTAL+=cnt;       //対称解除で得られた解数を加算
      return 1;
  }else{
      //bitmap=mask&~(left|down|right);//maskつけると10桁目以降数が出なくなるので外した
      bitmap=~(left|down|right);   
      while(bitmap>0){
          bit=(-bitmap&bitmap);
          bitmap=(bitmap^bit);
          cnt +=bit93_countCompletions(size,row+1,bv,(left|bit)<<1,down|bit,(right|bit)>>1);
      }

  }
  return cnt;
}

void bit93_process(int si,Board lb,int sym)
{
  pre[sym]++;
  cnt[sym]+=bit93_countCompletions(si,2,lb.bv >> 2,
      lb.left>>4,
      ((((lb.down>>2)|(~0<<(si-4)))+1)<<(si-5))-1,
      (lb.right>>4)<<(si-5));

}
//
void bit93_NQueen(int size)
{
  for (long bc=0;bc<=bcnt;bc++){
    bit93_process(size,GBoard[bc],GBoard[bc].cnt);
  }
} 
//
void q27_process(int si,Board lb,int sym)
{
  pre[sym]++;
  cnt[sym] += q27_countCompletions(lb.bv >> 2,
    ((((lb.down>>2)|(~0<<(si-4)))+1)<<(si-5))-1,
    lb.left>>4,(lb.right>>4)<<(si-5));
}
//
void q27_NQueen(int size)
{
  for (long bc=0;bc<bcnt;bc++){
    q27_process(size,GBoard[bc],GBoard[bc].cnt);
    //bit93_process(size,GBoard[bc],GBoard[bc].cnt);
  }
}
//
//q27,bit93で内容が同じなので共通化する
void prepare(int size)
{
  //CPUR
  int pres_a[930];
  int pres_b[930];
  int idx=0;
  for(int a=0;a<size;a++){
    for(int b=0;b<size;b++){
      if((a>=b&&(a-b)<=1)||(b>a&&(b-a)<=1)){ continue; }     
      pres_a[idx]=a;
      pres_b[idx]=b;
      idx++;
    }
  }
  //プログレス
  printf("\t\t  First side bound: (%d,%d)/(%d,%d)",(unsigned)pres_a[(size/2)*(size-3)  ],(unsigned)pres_b[(size/2)*(size-3)  ],(unsigned)pres_a[(size/2)*(size-3)+1],(unsigned)pres_b[(size/2)*(size-3)+1]);
  //プログレス
  
  Board wB=GBoard[bcnt];
  //for(int w=0;w<idx;w++){
  for (int w = 0; w <= (size / 2) * (size - 3); w++){
    GBoard[bcnt]=wB;
    GBoard[bcnt].bv=GBoard[bcnt].down=GBoard[bcnt].left=GBoard[bcnt].right=0;
    for(int j=0;j<size;j++){ GBoard[bcnt].x[j]=-1; }
      //プログレス
      printf("\r(%d/%d)",w,((size/2)*(size-3)));// << std::flush;
      printf("\r");
      fflush(stdout);
      //プログレス
    board_placement(size,0,pres_a[w]);
    board_placement(size,1,pres_b[w]);
    Board nB=GBoard[bcnt];
    int lsize=(size-2)*(size-1)-w;
    for(int n=w;n<lsize;n++){
      //for(int n=0;n<idx;n++){
      GBoard[bcnt]=nB;
      if(board_placement(size,pres_a[n],size-1)==false){ continue; }
      if(board_placement(size,pres_b[n],size-2)==false){ continue; }
      Board eB=GBoard[bcnt];
      for(int e=w;e<lsize;e++){
        //for(int e=0;e<idx;e++){
        GBoard[bcnt]=eB;  
        if(board_placement(size,size-1,size-1-pres_a[e])==false){ continue; }
        if(board_placement(size,size-2,size-1-pres_b[e])==false){ continue; }
        Board sB=GBoard[bcnt];
        for(int s=w;s<lsize;s++){
          //for(int s=0;s<idx;s++){
          GBoard[bcnt]=sB;
          if(board_placement(size,size-1-pres_a[s],0)==false){ continue; }
          if(board_placement(size,size-1-pres_b[s],1)==false){ continue; }
          int scnt=symmetryOps_n27(w,e,n,s,size);
          if(scnt !=3){
            GBoard[bcnt].cnt=scnt;
            bcnt++;                
          }
        }
      } 
    }
  }
}
//メインメソッド
int main(int argc,char** argv)
{
  bool cpu=false,cpur=false,gpu=false,sgpu=false,q27=false;
  int argstart=1;
  GBoard= (Board*)calloc(24576000, sizeof(Board));
  //,steps=24576;
  /** パラメータの処理 */
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
    else if(argv[1][1]=='s'||argv[1][1]=='S'){sgpu=true;}
    else if(argv[1][1]=='q'||argv[1][1]=='Q'){q27=true;}
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
    printf("  -q: q27 Version\n");
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  if(cpu){
    printf("\n\n９−３．CPU 非再帰 ビットマップ＋対象解除＋q２７枝刈＋BackTrack1＋BackTrack2\n");
  }else if(cpur){
    printf("\n\n９−３．CPUR 再帰 ビットマップ＋対象解除＋q２７枝刈＋BackTrack1＋BackTrack2\n");
  }else if(gpu){
    printf("\n\n９−３．GPU 非再帰 ビットマップ＋対象解除＋q２７枝刈＋BackTrack1＋BackTrack2\n");
  }else if(sgpu){
    printf("\n\n９−３．SGPU 非再帰 ビットマップ＋対象解除＋q２７枝刈＋BackTrack1＋BackTrack2\n");
  }
  if(cpu||cpur){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    clock_t st;          //速度計測用
    char t[20];          //hh:mm:ss.msを格納
    int min=5;
    int targetN=17;
    //uint64 mask;
    for(int i=min;i<=targetN;i++){
      TOTAL=0;
      UNIQUE=0;
      bcnt=0;
      for(int j=0;j<=2;j++){
        pre[j]=0;
        cnt[j]=0;
      }
      //mask=((1<<i)-1);
      int size=i;
      //事前準備 上下左右2行2列にクイーンを配置する
      prepare(size);
      //事前準備が終わってから時間を計測する
      st=clock();
    
      if(cpur){
	      bit93_NQueen(size);
        UNIQUE=cnt[ROTATE]+cnt[POINT]+cnt[NONE];
        TOTAL=cnt[ROTATE]*2+cnt[POINT]*4+cnt[NONE]*8;
      }else if(cpu){
        //
        //開発中
      }                
      //
      TimeFormat(clock()-st,t);
      printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
    }
  }
  if(q27){
    printf("\n\nＱ２７．SGPU 非再帰 ビットマップ＋対象解除＋q２７枝刈＋BackTrack1＋BackTrack2\n");
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    clock_t st;          //速度計測用
    char t[20];          //hh:mm:ss.msを格納
    int min=5;
    int targetN=17;
    //uint64 mask;
    for(int i=min;i<=targetN;i++){
      TOTAL=0;
      UNIQUE=0;
      bcnt=0;
      for(int j=0;j<=2;j++){
        pre[j]=0;
        cnt[j]=0;
      }
      int size=i;
      //事前準備 上下左右2行2列にクイーンを配置する
      prepare(size);
      //事前準備が終わってから時間を計測する
      st=clock();
      
      q27_NQueen(size);
      UNIQUE=cnt[ROTATE]+cnt[POINT]+cnt[NONE];
      TOTAL=cnt[ROTATE]*2+cnt[POINT]*4+cnt[NONE]*8;
      //
      TimeFormat(clock()-st,t);
      printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);             
    }
  }
  return 0;
}

