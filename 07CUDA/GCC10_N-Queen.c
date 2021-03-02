
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#define THREAD_NUM		96
#define MAX 27
//変数宣言
long Total=0 ;      //GPU
long Unique=0;      //GPU
int aBoard[MAX];
int aT[MAX];
int aS[MAX];
int COUNT2,COUNT4,COUNT8;
int BOUND1,BOUND2,TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
//
//関数宣言 CPU/GPU
void rotate_bitmap(int bf[],int af[],int si);
void vMirror_bitmap(int bf[],int af[],int si);
int intncmp(int lt[],int rt[],int n);
int rh(int a,int sz);
//関数宣言 CPU
void TimeFormat(clock_t utime,char *form);
long getUnique();
long getTotal();
void symmetryOps_bitmap(int si);
void backTrack2_NR(int si,int mask,int y,int l,int d,int r);
void backTrack1_NR(int si,int mask,int y,int l,int d,int r);
void NQueen(int size,int mask);
//【通常版】
void backTrack2(int si,int mask,int y,int l,int d,int r);
void backTrack1(int si,int mask,int y,int l,int d,int r);
void NQueenR(int size,int mask);
//【GPU移行版】

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
int rh(int a,int size){
  int tmp=0;
  for(int i=0;i<=size;i++){
    if(a&(1<<i)){
      return tmp|=(1<<(size-i));
    }
  }
  return tmp;
}
//
void vMirror_bitmap(int bf[],int af[],int size){
  int score;
  for(int i=0;i<size;i++){
    score=bf[i];
    af[i]=rh(score,size-1);
  }
}
//
void rotate_bitmap(int bf[],int af[],int size){
  int t;
  for(int i=0;i<size;i++){
    t=0;
    for(int j=0;j<size;j++){
      t|=((bf[j]>>i)&1)<<(size-j-1); // x[j] の i ビット目を
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
void symmetryOps_bitmap(int size){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<size;i++){
    aT[i]=aBoard[i];
  }
  rotate_bitmap(aT,aS,size);    //時計回りに90度回転
  int k=intncmp(aBoard,aS,size);
  if(k>0) return;
  if(k==0){
    nEquiv=2;
  }else{
    rotate_bitmap(aS,aT,size);  //時計回りに180度回転
    k=intncmp(aBoard,aT,size);
    if(k>0) return;
    if(k==0){
      nEquiv=4;
    }else{
      rotate_bitmap(aT,aS,size);  //時計回りに270度回転
      k=intncmp(aBoard,aS,size);
      if(k>0){
        return;
      }
      nEquiv=8;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<size;i++){
    aS[i]=aBoard[i];
  }
  vMirror_bitmap(aS,aT,size);   //垂直反転
  k=intncmp(aBoard,aT,size);
  if(k>0){
    return;
  }
  if(nEquiv>2){             //-90度回転 対角鏡と同等
    rotate_bitmap(aT,aS,size);
    k=intncmp(aBoard,aS,size);
    if(k>0){
      return;
    }
    if(nEquiv>4){           //-180度回転 水平鏡像と同等
      rotate_bitmap(aS,aT,size);
      k=intncmp(aBoard,aT,size);
      if(k>0){
        return;
      }       //-270度回転 反対角鏡と同等
      rotate_bitmap(aT,aS,size);
      k=intncmp(aBoard,aS,size);
      if(k>0){
        return;
      }
    }
  }
  if(nEquiv==2){ COUNT2++; }
  if(nEquiv==4){ COUNT4++; }
  if(nEquiv==8){ COUNT8++; }
}
//CPU 非再帰版 backTrack2
void backTrack2_NR(int size,int mask,int row,int left,int down,int right){
  int bitmap,bit;
  int b[100], *p=b;
  int odd=size&1; //奇数:1 偶数:0
  for(int i=0;i<(1+odd);++i){
    bitmap=0;
    if(0==i){
      int half=size>>1; // size/2
      bitmap=(1<<half)-1;
    }else{
      bitmap=1<<(size>>1);
      // down[1]=bitmap;
      // right[1]=(bitmap>>1);
      // left[1]=(bitmap<<1);
      // pnStack=aStack+1;
      // *pnStack++=0;
    }
mais1:bitmap=mask&~(left|down|right);
      if(row==size){
        if(!bitmap){
          aBoard[row]=bitmap;
          symmetryOps_bitmap(size);
        }
      }else{
        if(bitmap){
outro:bitmap^=aBoard[row]=bit=-bitmap&bitmap;
      if(bitmap){
        *p++=left;
        *p++=down;
        *p++=right;
      }
      *p++=bitmap;
      row++;
      left=(left|bit)<<1;
      down=down|bit;
      right=(right|bit)>>1;
      goto mais1;
      //Backtrack2(y+1, (left | bit)<<1, down | bit, (right | bit)>>1);
volta:if(p<=b)
        return;
      row--;
      bitmap=*--p;
      if(bitmap){
        right=*--p;
        down=*--p;
        left=*--p;
        goto outro;
      }else{
        goto volta;
      }
        }
      }
      goto volta;
  }
}
//CPU 非再帰版 backTrack
void backTrack1_NR(int size,int mask,int row,int left,int down,int right){
  int bitmap,bit;
  int b[100], *p=b;
  int sizeE=size-1;
  int odd=size&1; //奇数:1 偶数:0
  for(int i=0;i<(1+odd);++i){
    bitmap=0;
    if(0==i){
      int half=size>>1; // size/2
      bitmap=(1<<half)-1;
    }else{
      bitmap=1<<(size>>1);
      // down[1]=bitmap;
      // right[1]=(bitmap>>1);
      // left[1]=(bitmap<<1);
      // pnStack=aStack+1;
      // *pnStack++=0;
    }
b1mais1:bitmap=mask&~(left|down|right);
        if(row==sizeE){
          if(bitmap){
            aBoard[row]=bitmap;
            symmetryOps_bitmap(size);
          }
        }else{
          if(bitmap){
b1outro:bitmap^=aBoard[row]=bit=-bitmap&bitmap;
        if(bitmap){
          *p++=left;
          *p++=down;
          *p++=right;
        }
        *p++=bitmap;
        row++;
        left=(left|bit)<<1;
        down=down|bit;
        right=(right|bit)>>1;
        goto b1mais1;
        //Backtrack1(y+1, (left | bit)<<1, down | bit, (right | bit)>>1);
b1volta:if(p<=b)
          return;
        row--;
        bitmap=*--p;
        if(bitmap){
          right=*--p;
          down=*--p;
          left=*--p;
          goto b1outro;
        }else{
          goto b1volta;
        }
          }
        }
        goto b1volta;
  }
}
//CPU 非再帰版 ロジックメソッド
void NQueen(int size,int mask){
  int bit;
  TOPBIT=1<<(size-1);
  aBoard[0]=1;
  for(BOUND1=2;BOUND1<size-1;BOUND1++){
    aBoard[1]=bit=(1<<BOUND1);
    //backTrack1(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
    backTrack1_NR(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    //backTrack1(size,mask,1,bit<<1,bit,bit>>1);
    backTrack2_NR(size,mask,1,bit<<1,bit,bit>>1);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  }
}
//
void backTrack2(int size,int mask,int row,int left,int down,int right){
  int bit;
  int bitmap=mask&~(left|down|right); /* 配置可能フィールド */
  if(row==size){
    aBoard[row]=bitmap; //symmetryOpsの時は代入します。
    symmetryOps_bitmap(size);
  }else{
    while(bitmap){
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
      backTrack2(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
}
//
void backTrack1(int size,int mask,int row,int left,int down,int right){
  int bit;
  int bitmap=mask&~(left|down|right);   //BOUNDで対応済み
  if(row==size){
    aBoard[row]=bitmap; //symmetryOpsの時は代入します。
    symmetryOps_bitmap(size);
  }else{
    while(bitmap){
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
      backTrack1(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
}
//CPUR 再帰版 ロジックメソッド
void NQueenR(int size,int mask){
  int bit;
  TOPBIT=1<<(size-1);
  aBoard[0]=1;
  for(BOUND1=2;BOUND1<size-1;BOUND1++){
    aBoard[1]=bit=(1<<BOUND1);
    backTrack1(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    backTrack2(size,mask,1,bit<<1,bit,bit>>1);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
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
    printf("Usage: %s [-c|-g|-r|-s]\n",argv[0]);
    printf("  -c: CPU only\n");
    printf("  -r: CPUR only\n");
    printf("  -g: GPU only\n");
    printf("  -s: SGPU only\n");
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  if(cpu){
    printf("\n\n１０．CPU 非再帰 クイーンの位置による分岐BOUND1,2\n");
  }else if(cpur){
    printf("\n\n１０．CPUR 再帰 クイーンの位置による分岐BOUND1,2\n");
  }else if(gpu){
    printf("\n\n１０．GPU 非再帰 クイーンの位置による分岐BOUND1,2\n");
  }else if(sgpu){
    printf("\n\n１０．SGPU 非再帰 クイーンの位置による分岐BOUND1,2\n");
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
      //非再帰は-1で初期化
      // for(int j=0;j<=targetN;j++){ aBoard[j]=-1; }
      if(cpu){ NQueen(i,mask); }
      //【通常版】
      if(cpur){ NQueenR(i,mask); }
      //【GPU移行版】
      //if(cpur){ NQueenR(i,mask); }

      TimeFormat(clock()-st,t); 
      printf("%2d:%13ld%16ld%s\n",i,getTotal(),getUnique(),t);
    }
  }
  return 0;
}
