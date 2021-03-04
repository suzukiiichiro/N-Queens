
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#define THREAD_NUM		96
#define MAX 27
//変数宣言
int down[2*MAX-1];  //CPU down:flagA 縦 配置フラグ　
int left[2*MAX-1];  //CPU left:flagB 斜め配置フラグ　
int right[2*MAX-1]; //CPU right:flagC 斜め配置フラグ　
int aBoard[MAX];
int aT[MAX];
int aS[MAX];
int COUNT2,COUNT4,COUNT8;
//関数宣言 CPU/GPU
void rotate_bitmap(int bf[],int af[],int si);
void vMirror_bitmap(int bf[],int af[],int si);
int intncmp(int lt[],int rt[],int n);
int rh(int a,int size);
//関数宣言
void TimeFormat(clock_t utime,char *form);
long getUnique();
long getTotal();
void symmetryOps_bitmap(int si);
void NQueen(int size,int mask);
void NQueenR(int size,int mask);
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
int rh(int a,int sz){
  int tmp=0;
  for(int i=0;i<=sz;i++){
    if(a&(1<<i)){ return tmp|=(1<<(sz-i)); }
  }
  return tmp;
}
//
void vMirror_bitmap(int bf[],int af[],int si){
  int score ;
  for(int i=0;i<si;i++) {
    score=bf[i];
    af[i]=rh(score,si-1);
  }
}
//
void rotate_bitmap(int bf[],int af[],int si){
  for(int i=0;i<si;i++){
    int t=0;
    for(int j=0;j<si;j++){
      t|=((bf[j]>>i)&1)<<(si-j-1); // x[j] の i ビット目を
    }
    af[i]=t;                        // y[i] の j ビット目にする
  }
}
//
int intncmp(int lt[],int rt[],int n){
  int rtn=0;
  for(int k=0;k<n;k++){
    rtn=lt[k]-rt[k];
    if(rtn!=0){
      break;
    }
  }
  return rtn;
}
//
long getUnique(){
  return COUNT2+COUNT4+COUNT8;
}
//
long getTotal(){
  return COUNT2*2+COUNT4*4+COUNT8*8;
}
//
void symmetryOps_bitmap(int si){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aT[i]=aBoard[i];}
  rotate_bitmap(aT,aS,si);    //時計回りに90度回転
  int k=intncmp(aBoard,aS,si);
  if(k>0)return;
  if(k==0){ nEquiv=2;}else{
    rotate_bitmap(aS,aT,si);  //時計回りに180度回転
    k=intncmp(aBoard,aT,si);
    if(k>0)return;
    if(k==0){ nEquiv=4;}else{
      rotate_bitmap(aT,aS,si);//時計回りに270度回転
      k=intncmp(aBoard,aS,si);
      if(k>0){ return;}
      nEquiv=8;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aS[i]=aBoard[i];}
  vMirror_bitmap(aS,aT,si);   //垂直反転
  k=intncmp(aBoard,aT,si);
  if(k>0){ return; }
  if(nEquiv>2){             //-90度回転 対角鏡と同等
    rotate_bitmap(aT,aS,si);
    k=intncmp(aBoard,aS,si);
    if(k>0){return;}
    if(nEquiv>4){           //-180度回転 水平鏡像と同等
      rotate_bitmap(aS,aT,si);
      k=intncmp(aBoard,aT,si);
      if(k>0){ return;}       //-270度回転 反対角鏡と同等
      rotate_bitmap(aT,aS,si);
      k=intncmp(aBoard,aS,si);
      if(k>0){ return;}
    }
  }
  if(nEquiv==2){COUNT2++;}
  if(nEquiv==4){COUNT4++;}
  if(nEquiv==8){COUNT8++;}
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
        bitmap[row]^=aBoard[row]=bit=(-bitmap[row]&bitmap[row]); 
        if((bit&mask)!=0){
          if(row==sizeE){
            symmetryOps_bitmap(size);
            --row;
            continue;
          }else{
            int n=row++;
            left[row]=(left[n]|bit)<<1;
            down[row]=down[n]|bit;
            right[row]=(right[n]|bit)>>1;
            bitmap[row]=mask&~(left[row]|down[row]|right[row]);
            continue;
          }
        }else{
           --row;
           continue;
        }
      }  
    }
}
void NQueen(int size,int mask){
  register int bitmap[size];
  register int down[size],right[size],left[size];
  register int bit;
  if(size<=0||size>32){return;}
  int sizeE=size-1;
  int row=0;
  bit=0;
  bitmap[0]=mask;
  down[0]=left[0]=right[0]=0;
  //偶数、奇数ともに右半分にクイーンを置く
  for(int col=0;col<size/2;col++){
    //ex n=6 xxxooo n=7 xxxxooo 
    aBoard[0]=bit=(1<<col);
    down[1]=bit;//再帰の場合は down,left,right,bitmapは現在の行だけで良いが
    left[1]=bit<<1;//非再帰の場合は全行情報を配列に入れて行の上がり下がりをする
    right[1]=bit>>1;
    bitmap[1]=mask&~(left[1]|down[1]|right[1]);
    solve_nqueen(size,mask,1,left,down,right,bitmap);
  }
  //奇数については中央にもクイーンを置く
  if(size%2==1){
    int col=(sizeE)/2;
    //1行目はクイーンを中央に置く
    bit=aBoard[0]=(1<<col);
    down[1]=bit;//再帰の場合は down,left,right,bitmapは現在の行だけで良いが
    left[1]=bit<<1;//非再帰の場合は全行情報を配列に入れて行の上がり下がりをする
    right[1]=bit>>1;
    bitmap[1]=mask&~(left[1]|down[1]|right[1]);
    for(int col_j=0;col_j<(size/2)-1;col_j++){
    //1行目にクイーンが中央に置かれた場合は2行目の左側半分にクイーンを置けない
    //0001000
    //xxxdroo  左側半分にクイーンを置けないがさらに1行目のdown,rightもクイーンを置けないので (size/2)-1となる
      //2行目にクイーンを置く
      aBoard[1]=bit=(1<<col_j);
      down[2]=bit;//再帰の場合は down,left,right,bitmapは現在の行だけで良いが
      left[2]=bit<<1;//非再帰の場合は全行情報を配列に入れて行の上がり下がりをする
      right[2]=bit>>1;
      bitmap[2]=mask&~(left[2]|down[2]|right[2]);
      solve_nqueen(size,mask,2,left,down,right,bitmap);
    }
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
     aBoard[row]=(-bitmap&bitmap);
     symmetryOps_bitmap(size);
   }
  }else{
    while(bitmap){
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
      solve_nqueenr(size,mask,row+1,(left|bit)<<1, down|bit,(right|bit)>>1);
    }
  }
}
//
//CPUR 再帰版 ロジックメソッド
void NQueenR(int size,int mask){
  int bit=0;
  int sizeE=size-1;
  //偶数、奇数ともに右半分にクイーンを置く
  for(int col=0;col<size/2;col++){
    //ex n=6 xxxooo n=7 xxxxooo 
    bit=aBoard[0]=(1<<col);
    solve_nqueenr(size,mask,1,bit<<1,bit,bit>>1);
  }
  //奇数については中央にもクイーンを置く
  if(size%2==1){
    int col=(sizeE)/2;
    //1行目はクイーンを中央に置く
    bit=aBoard[0]=(1<<col);
    int left=bit<<1;
    int down=bit;
    int right=bit>>1;
    for(int col_j=0;col_j<(size/2)-1;col_j++){
    //1行目にクイーンが中央に置かれた場合は2行目の左側半分にクイーンを置けない
    //0001000
    //xxxdroo  左側半分にクイーンを置けないがさらに1行目のdown,rightもクイーンを置けないので (size/2)-1となる
      //2行目にクイーンを置く
      aBoard[1]=bit=(1<<col_j);
      solve_nqueenr(size,mask,2,(left|bit)<<1,(down|bit),(right|bit)>>1);
    }
  }
}
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
    printf("Usage: %s [-c|-g|-r]\n",argv[0]);
    printf("  -c: CPU only\n");
    printf("  -r: CPUR only\n");
    printf("  -g: GPU only\n");
    printf("  -s: SGPU only\n");
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  if(cpu){
    printf("\n\n８．CPU 非再帰 ビットマップ＋対称解除法＋枝刈り\n");
  }else if(cpur){
    printf("\n\n８．CPUR 再帰 ビットマップ＋対称解除法＋枝刈り\n");
  }else if(gpu){
    printf("\n\n８．GPU 非再帰 ビットマップ＋対称解除法＋枝刈り\n");
  }else if(sgpu){
    printf("\n\n８．SGPU 非再帰 ビットマップ＋対称解除法＋枝刈り\n");
  }
  if(cpu||cpur){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    clock_t st;           //速度計測用
    char t[20];           //hh:mm:ss.msを格納
    int min=4; int targetN=17;
    int mask;
    for(int i=min;i<=targetN;i++){
      //TOTAL=0; UNIQUE=0;
      COUNT2=COUNT4=COUNT8=0;
      mask=(1<<i)-1;
      st=clock();
      //初期化は不要です
      /** 非再帰は-1で初期化 */
      // for(int j=0;j<=targetN;j++){
      //   aBoard[j]=-1;
      // }
      if(cpu){ NQueen(i,mask); }
      if(cpur){ 
        /* NQueenR(i,mask,0,0,0,0);  */
        NQueenR(i,mask);
      }
      TimeFormat(clock()-st,t); 
      printf("%2d:%13ld%16ld%s\n",i,getTotal(),getUnique(),t);
    }
  }
  return 0;
}
