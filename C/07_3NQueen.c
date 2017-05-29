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
 
 
 
  N-Queens問題とは
  https://ja.wikipedia.org/wiki/エイト・クイーン
     Nクイーン問題とは、「8列×8行のチェスボードに8個のクイーンを、互いに効きが
     当たらないように並べよ」という８クイーン問題のクイーン(N)を、どこまで大き
     なNまで解を求めることができるかという問題。
     クイーンとは、チェスで使われているクイーンを指し、チェス盤の中で、縦、横、
     斜めにどこまでも進むことができる駒で、日本の将棋でいう「飛車と角」を合わ
     せた動きとなる。８列×８行で構成される一般的なチェスボードにおける8-Queens
     問題の解は、解の総数は92個である。比較的単純な問題なので、学部レベルの演
     習問題として取り上げられることが多い。
     8-Queens問題程度であれば、人力またはプログラムによる「力まかせ探索」でも
     解を求めることができるが、Nが大きくなると解が一気に爆発し、実用的な時間で
     は解けなくなる。
     現在すべての解が判明しているものは、2004年に電気通信大学で264CPU×20日をか
     けてn=24を解決し世界一に、その後2005 年にニッツァ大学でn=25、2016年にドレ
     スデン工科大学でn=27の解を求めることに成功している。
 
 	ステップバイステップでＮ−クイーン問題を最適化
   １．ブルートフォース（力まかせ探索） NQueen1()
   ２．バックトラック                   NQueen2()
 <>３．配置フラグ（制約テスト高速化）   NQueen3()
   ４．対称解除法(回転と斜軸）          NQueen4()
   ５．ビットマップ                     NQueen7()
   ６．マルチスレッド                   NQueen8()

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
 *
 */
#include <stdio.h>
#include <time.h>

#define MAXSIZE 27

int iTotal=1 ;
int iUnique=0;
int iSize;
int colChk [2*MAXSIZE-1];
int diagChk[2*MAXSIZE-1];
int antiChk[2*MAXSIZE-1];
int aBoard[MAXSIZE];

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
    iTotal++;
  }else{
    for(int col=0;col<iSize;col++){
      aBoard[row]=col ;
      if(colChk[col]==0 && diagChk[row-col+(iSize-1)]==0 && antiChk[row+col]==0){
        colChk[col]=diagChk[row-aBoard[row]+iSize-1]=antiChk[row+aBoard[row]]=1; 
        NQueen3(row+1); 
        colChk[col]=diagChk[row-aBoard[row]+iSize-1]=antiChk[row+aBoard[row]]=0; 
      }
    }  
  }
}
int main(void) {
  clock_t st; 
  char t[20];
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  for(int i=2;i<=MAXSIZE;i++){
    iSize=i;
    iTotal=0; 
    iUnique=0; 
    for(int j=0;j<iSize;j++){
      aBoard[j]=j;
    }
    st=clock();
    NQueen3(0);
    TimeFormat(clock()-st,t);
    printf("%2d:%13d%16d%s\n",iSize,iTotal,iUnique,t) ;
  } 
}

