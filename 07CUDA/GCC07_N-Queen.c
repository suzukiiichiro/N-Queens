
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
int aBoard[MAX];
int aT[MAX];
int aS[MAX];
int COUNT2,COUNT4,COUNT8;
//関数宣言 CPU
void TimeFormat(clock_t utime,char *form);
void symmetryOps_bitmap(int si);
void NQueen(int size,int mask,int row);
//関数宣言 CPUR
void solve_nqueenr(int size,int mask, int row,int left,int down,int right);
void NQueenR(int size,int mask,int row,int left,int down,int right);
//
//
//hh:mm:ss.ms形式に処理時間を出力
//
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
//CPU
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
//
//CPU 非再帰版 ロジックメソッド
void NQueen(int size,int mask,int row){
  register int bit;
  register int bitmap[32];
  register int sizeE=size-1;
  register int down[size],right[size],left[size];
  if(size<=0||size>32){return;}
  bit=0;
  down[0]=left[0]=right[0]=0;
  for(int col=0;col<size/2;col++){//右側半分だけやる
    bit=(1<<col);
    bitmap[0]=mask;
    down[1]=bit;
    left[1]=bit<<1;
    right[1]=bit>>1;
    bitmap[1]=mask&~(left[1]|down[1]|right[1]);
    row=1;
    while(row>0){
      if(bitmap[row]==0){
        row--;
      }else{
        bitmap[row]^=bit=(-bitmap[row]&bitmap[row]); 
        if((bit&mask)!=0){
          if(row==sizeE){
            TOTAL++;
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
  TOTAL*=2;
  if(size%2==1){
    bit=(1<<(sizeE)/2);
    bitmap[0]=mask;
    down[1]=bit;
    left[1]=bit<<1;
    right[1]=bit>>1;
    bitmap[1]=mask&~(left[1]|down[1]|right[1]);
    row=1;
    while(row>0){
      if(bitmap[row]==0){
        row--;
      }else{
        bitmap[row]^=bit=(-bitmap[row]&bitmap[row]); 
        if((bit&mask)!=0){
          if(row==sizeE){
            TOTAL++;
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
}
//
//CPUR 再帰版 ロジックメソッド
void NQueenR(int size,int mask, int row,int left,int down,int right){
  int bit;
  int bitmap=mask&~(left|down|right);
  int sizeE=size-1;
  if(row==sizeE){
    if(bitmap){
      aBoard[row]=(-bitmap&bitmap);
      symmetryOps_bitmap(size);
    }
   }else{
    while(bitmap){
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
      NQueenR(size,mask,row+1,(left|bit)<<1, down|bit,(right|bit)>>1);
    }
   } 
}
//
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
    printf("Usage: %s [-c|-g|-r|-s]\n",argv[0]);
    printf("  -c: CPU only\n");
    printf("  -r: CPUR only\n");
    printf("  -g: GPU only\n");
    printf("  -s: SGPU only\n");
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  if(cpu){
   printf("\n\n７．CPU 非再帰 バックトラック＋ビットマップ＋対称解除法\n");
  }else if(cpur){
    printf("\n\n７．CPUR 再帰 バックトラック＋ビットマップ＋対称解除法\n");
  }else if(gpu){
    printf("\n\n７．GPU 非再帰 バックトラック＋ビットマップ＋対称解除法\n");
  }else if(sgpu){
    printf("\n\n７．SGPU 非再帰 バックトラック＋ビットマップ＋対称解除法\n");
  }
  if(cpu||cpur){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    clock_t st;          //速度計測用
    char t[20];          //hh:mm:ss.msを格納
    int min=4;
    int targetN=17;
    int mask;
    for(int i=min;i<=targetN;i++){
      TOTAL=0;
      UNIQUE=0;
      COUNT2=COUNT4=COUNT8=0;
      mask=((1<<i)-1);
      st=clock();
      if(cpu){ NQueen(i,mask,0); }
      if(cpur){ NQueenR(i,mask,0,0,0,0); }
      TimeFormat(clock()-st,t);
      printf("%2d:%13ld%16ld%s\n",i,getTotal(),getUnique(),t);
    }
  }
  return 0;
}
