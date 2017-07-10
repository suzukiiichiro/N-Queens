/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
   １．ブルートフォース（力まかせ探索） NQueen01()
   ２．配置フラグ（制約テスト高速化）   NQueen02()
   ３．バックトラック                   NQueen03() 
   ４．対称解除法(回転と斜軸）          NQueen04() 
 <>５．枝刈りと最適化                   NQueen05() 
   ６．ビットマップ                     NQueen06() 
   ７．ビットマップ+対称解除法          NQueen07() 
   ８．ビットマップ+クイーンの場所で分岐NQueen08() 
   ９．ビットマップ+枝刈りと最適化      NQueen09() 
   10．もっとビットマップ(takaken版)    NQueen10() 
   11．マルチスレッド(構造体)           NQueen11() 
   12．マルチスレッド(pthread)          NQueen12() 
   13．マルチスレッド(join)             NQueen13() 
   14．マルチスレッド(mutex)            NQueen14() 
   15．マルチスレッド(アトミック対応)   NQueen15() 
   16．アドレスとポインタ               NQueen16() 
   17．アドレスとポインタ(脱構造体)     NQueen17() 
   18．アドレスとポインタ(脱配列)       NQueen18()

 # Java/C/Lua/Bash版
 # https://github.com/suzukiiichiro/N-Queen
 

 * ５．枝刈りと最適化
 * 　単純ですのでソースのコメントを見比べて下さい。
 *   単純ではありますが、枝刈りの効果は絶大です。

   実行結果
 N:        Total       Unique        dd:hh:mm:ss
 2:            0               0      0 00:00:00
 3:            0               0      0 00:00:00
 4:            2               1      0 00:00:00
 5:           10               2      0 00:00:00
 6:            4               1      0 00:00:00
 7:           40               6      0 00:00:00
 8:           92              12      0 00:00:00
 9:          352              46      0 00:00:00
10:          724              92      0 00:00:00
11:         2680             341      0 00:00:00
12:        14200            1787      0 00:00:00
13:        73712            9233      0 00:00:00
14:       365596           45752      0 00:00:00
15:      2279184          285053      0 00:00:03
16:     14772512         1846955      0 00:00:20
17:     95815104        11977939      0 00:02:31
 */
#include<stdio.h>
#include<time.h>

#define MAX 27

long Total=1;      //合計解
long Unique=0;      //ユニーク解
int fA [2*MAX-1];   //fA:flagA[] 縦 配置フラグ　
int fB[2*MAX-1];    //fB:flagB[] 斜め配置フラグ　
int fC[2*MAX-1];    //fC:flagC[] 斜め配置フラグ　
int aB[MAX];        //aB:aBoard[] チェス盤の横一列
int aT[MAX];        //aT:aTrial[]
int aS[MAX];        //aS:aScrath[]

void NQueen(int si,int row);
void TimeFormat(clock_t utime,char *form);
int symmetryOps(int si);
void rotate(int chk[],int scr[],int n,int neg);
void vMirror(int chk[],int n);
int intncmp(int lt[],int rt[],int n);

// i:col si:size r:row fA:縦 fB:斜め fC:斜め
void NQueen(int si,int r){
  int t; //t:temp
  if(r==si-1){
    // 枝刈り 
    if ((fB[r-aB[r]+si-1]||fC[r+aB[r]])){ return; }
    int s=symmetryOps(si);//対称解除法
    if(s!=0){ Unique++; Total+=s; } //解を発見
  }else{
    // 枝刈り 半分だけ捜査
    int lim=(r!=0)?si:(si+1)/2; 
    // i:col
    for(int i=r;i<lim;i++){
      t=aB[i]; aB[i]=aB[r]; aB[r]=t; // swap
      // 枝刈り バックトラック 制約を満たしているときだけ進む
      if(!(fB[r-aB[r]+si-1]||fC[r+aB[r]])){
        fB[r-aB[r]+si-1]=fC[r+aB[r]]=1;
        NQueen(si,r+1); //再帰
        fB[r-aB[r]+si-1]=fC[r+aB[r]]=0;
      }
    }
    t=aB[r];
    for(int i=r+1;i<si;i++){ aB[i-1]=aB[i]; }
    aB[si-1]=t;
  }
}
int main(void){
  clock_t st; char t[20];
  int min=2;
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  for(int i=min;i<=MAX;i++){
    Total=0; Unique=0;
    for(int j=0;j<i;j++){ aB[j]=j; } //aBを初期化
    st=clock();
    NQueen(i,0);
    TimeFormat(clock()-st,t);
    printf("%2d:%13ld%16ld%s\n",i,Total,Unique,t);
  } 
  return 0;
}
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
  if (dd) sprintf(form,"%4d %02d:%02d:%05.2f",dd,hh,mm,ss);
  else if (hh) sprintf(form, "     %2d:%02d:%05.2f",hh,mm,ss);
  else if (mm) sprintf(form, "        %2d:%05.2f",mm,ss);
  else sprintf(form, "           %5.2f",ss);
}
//si:size
int symmetryOps(int si){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aT[i]=aB[i];}
  rotate(aT,aS,si,0);     //時計回りに90度回転
  int k=intncmp(aB,aT,si);
  if(k>0)return 0;
  if(k==0){ nEquiv=1; }else{
    rotate(aT,aS,si,0);   //時計回りに180度回転
    k=intncmp(aB,aT,si);
    if(k>0)return 0;
    if(k==0){ nEquiv=2; }else{
      rotate(aT,aS,si,0); //時計回りに270度回転
      k=intncmp(aB,aT,si);
      if(k>0){ return 0; }
      nEquiv=4;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aT[i]=aB[i];}
  vMirror(aT,si);         //垂直反転
  k=intncmp(aB,aT,si);
  if(k>0){ return 0; }
  if(nEquiv>1){           //-90度回転 対角鏡と同等       
    rotate(aT,aS,si,1);
    k=intncmp(aB,aT,si);
    if(k>0){return 0; }
    if(nEquiv>2){         //-180度回転 水平鏡像と同等
      rotate(aT,aS,si,1);
      k=intncmp(aB,aT,si);
      if(k>0){ return 0; }//-270度回転 反対角鏡と同等
      rotate(aT,aS,si,1);
      k=intncmp(aB,aT,si);
      if(k>0){ return 0; }
    }
  }
  return nEquiv*2;
}
void rotate(int chk[],int scr[],int n,int neg){
  int k=neg?0:n-1;
  int incr=(neg?+1:-1);
  for(int j=0;j<n;k+=incr){ scr[j++]=chk[k];}
  k=neg?n-1:0;
  for(int j=0;j<n;k-=incr){ chk[scr[j++]]=k;}
}
void vMirror(int chk[],int n){
  for(int j=0;j<n;j++){ chk[j]=(n-1)- chk[j];}
}
int intncmp(int lt[],int rt[],int n){
  int rtn=0;
  for(int k=0;k<n;k++){
    rtn=lt[k]-rt[k];
    if(rtn!=0){ break;}
  }
  return rtn;
}
