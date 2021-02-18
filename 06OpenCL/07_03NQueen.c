/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
  ３．バックトラック

		コンパイルと実行
		$ make nq3 && ./07_03NQueen

   　各列、対角線上にクイーンがあるかどうかのフラグを用意し、途中で制約を満た
   さない事が明らかな場合は、それ以降のパターン生成を行わない。
   　各列、対角線上にクイーンがあるかどうかのフラグを用意することで高速化を図る。
   　これまでは行方向と列方向に重複しない組み合わせを列挙するものですが、王妃
   は斜め方向のコマをとることができるので、どの斜めライン上にも王妃をひとつだ
   けしか配置できない制限を加える事により、深さ優先探索で全ての葉を訪問せず木
   を降りても解がないと判明した時点で木を引き返すということができます。
 
 
  実行結果
 N:        Total       Unique        hh:mm:ss.ms
 2:            0               0            0.00
 3:            0               0            0.00
 4:            2               0            0.00
 5:           10               0            0.00
 6:            4               0            0.00
 7:           40               0            0.00
 8:           92               0            0.00
 9:          352               0            0.00
10:          724               0            0.00
11:         2680               0            0.01
12:        14200               0            0.05
13:        73712               0            0.30
14:       365596               0            1.93
15:      2279184               0           13.50
16:     14772512               0         1:39.30
17:     95815104               0        12:29.59
 */

#include <stdio.h>
#include <time.h>

#define MAX 27

long Total=1;      //合計解
long Unique=0;     //ユニーク解
int aB[MAX];       //aB:aBoard[] チェス盤の横一列
int fA[2*MAX-1];   //fA:flagA[] 縦 配置フラグ
int fB[2*MAX-1];   //fB:flagB[] 斜め配置フラグ
int fC[2*MAX-1];   //fC:flagC[] 斜め配置フラグ

void NQueen(int r,int si);
void TimeFormat(clock_t utime,char *form);

// i:col si:size r:row fA:縦 fB:斜め fC:斜め
void NQueen(int r,int si){
  if(r==si){
    Total++; //解を発見
  }else{
    for(int i=0;i<si;i++){
      aB[r]=i;
      //バックトラック 制約を満たしているときだけ進む
      if(fA[i]==0&&fB[r-i+(si-1)]==0&&fC[r+i]==0){
        fA[i]=fB[r-aB[r]+si-1]=fC[r+aB[r]]=1;
        NQueen(r+1,si);//再帰
        fA[i]=fB[r-aB[r]+si-1]=fC[r+aB[r]]=0;
      }
    }
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
    NQueen(0,i);
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
