/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
  Java版 N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_N-Queen
  Bash版 N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_Bash
  Lua版  N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_Lua
 
  ステップバイステップでＮ−クイーン問題を最適化
   １．ブルートフォース（力まかせ探索） NQueen1()
   ２．配置フラグ（制約テスト高速化）   NQueen2()
 <>３．バックトラック                   NQueen3() N16: 1:07
   ４．対称解除法(回転と斜軸）          NQueen4() 
   ５．枝刈りと最適化                   NQueen5() 
   ６．ビットマップ                     NQueen6() 
   ７．ビットマップ+対称解除法          NQueen7() 
   ８．ビットマップ+クイーンの場所で分岐NQueen8() 
   ９．ビットマップ+枝刈りと最適化      NQueen8() 
   10．完成型                           NQueen9() N16: 0:02
   11．マルチスレッド                   NQueen10()

  ３．バックトラック
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
 13:        73712               0            0.26
 14:       365596               0            1.54
 15:      2279184               0            9.85
 16:     14772512               0         1:07.42
 */

#include <stdio.h>
#include <time.h>

#define MAXSIZE 27

int iTotal=1 ; //合計解
int iUnique=0; //ユニーク解
int iSize;     //Ｎ
int colChk [2*MAXSIZE-1]; //縦 配置フラグ　
int diagChk[2*MAXSIZE-1]; //斜め配置フラグ　
int antiChk[2*MAXSIZE-1]; //斜め配置フラグ　
int aBoard[MAXSIZE];  //チェス盤の横一列

void TimeFormat(clock_t utime, char *form) {
    int dd,hh,mm;
    float ftime,ss;
    ftime=(float)utime/CLOCKS_PER_SEC;
    mm=(int)ftime/60;
    ss=ftime-(float)(mm * 60);
    dd=mm/(24*60);
    mm=mm%(24*60);
    hh=mm/60;
    mm=mm%60;
    if (dd) sprintf(form,"%4d %02d:%02d:%05.2f",dd,hh,mm,ss);
    else if (hh) sprintf(form, "     %2d:%02d:%05.2f",hh,mm,ss);
    else if (mm) sprintf(form, "        %2d:%05.2f",mm,ss);
    else sprintf(form, "           %5.2f",ss);
}
void NQueen3(int row){
  if(row==iSize){
    iTotal++; //解を発見
  }else{
    for(int col=0;col<iSize;col++){
      aBoard[row]=col ;
      //バックトラック 制約を満たしているときだけ進む
      if(colChk[col]==0 && diagChk[row-col+(iSize-1)]==0 && antiChk[row+col]==0){
        colChk[col]=diagChk[row-aBoard[row]+iSize-1]=antiChk[row+aBoard[row]]=1; 
        NQueen3(row+1);//再帰
        colChk[col]=diagChk[row-aBoard[row]+iSize-1]=antiChk[row+aBoard[row]]=0; 
      }
    }  
  }
}
int main(void) {
  clock_t st; char t[20];
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  for(int i=2;i<=MAXSIZE;i++){
    iSize=i; iTotal=0; iUnique=0; 
    for(int j=0;j<iSize;j++){ aBoard[j]=j; } //aBoardを初期化
    st=clock();
    NQueen3(0);
    TimeFormat(clock()-st,t);
    printf("%2d:%13d%16d%s\n",iSize,iTotal,iUnique,t) ;
  } 
}

