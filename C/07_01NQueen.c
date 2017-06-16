/**

  Cで学ぶアルゴリズムとデータ構造  
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
 
  Java版はこちらを
  https://github.com/suzukiiichiro/AI_Algorithm_N-Queen/Java
  Lua版
  https://github.com/suzukiiichiro/AI_Algorithm_N-Queen/Lua
  Bash版
  https://github.com/suzukiiichiro/AI_Algorithm_N-Queen/Bash
 

  再帰  Nクイーン問題
  https://ja.wikipedia.org/wiki/エイト・クイーン
 
  N-Queens問題とは
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
 <>１．ブルートフォース（力まかせ探索） NQueen01()
   ２．配置フラグ（制約テスト高速化）   NQueen02()
   ３．バックトラック                   NQueen03() 
   ４．対称解除法(回転と斜軸）          NQueen04() 
   ５．枝刈りと最適化                   NQueen05() 
   ６．ビットマップ                     NQueen06() 
   ７．ビットマップ+対称解除法          NQueen07() 
   ８．ビットマップ+クイーンの場所で分岐NQueen08() 
   ９．ビットマップ+枝刈りと最適化      NQueen09() 
   10．もっとビットマップ               NQueen10()
   11．マルチスレッド(構造体)           NQueen11() N16: 0:02
   12．マルチスレッド(pthread)          NQueen12()

 1. ブルートフォース　力任せ探索
 　全ての可能性のある解の候補を体系的に数え上げ、それぞれの解候補が問題の解と
   なるかをチェックする方法
   (※)各行に１個の王妃を配置する組み合わせを再帰的に列挙組み合わせを生成するだ
   けであって8王妃問題を解いているわけではありません

  実行結果
  :
  :
  16777207: 7 7 7 7 7 7 6 6
  16777208: 7 7 7 7 7 7 6 7
  16777209: 7 7 7 7 7 7 7 0
  16777210: 7 7 7 7 7 7 7 1
  16777211: 7 7 7 7 7 7 7 2
  16777212: 7 7 7 7 7 7 7 3
  16777213: 7 7 7 7 7 7 7 4
  16777214: 7 7 7 7 7 7 7 5
  16777215: 7 7 7 7 7 7 7 6
  16777216: 7 7 7 7 7 7 7 7
**/

#include <stdio.h>
#include <time.h>

#define MINSIZE 0
#define MAXSIZE 8

int iCount=1 ; //# c:count
int aBoard[MAXSIZE];

void NQueen(int iMin,int iMax) {
  for(int i=0;i<iMax;i++){
      aBoard[iMin]=i ;
      if (iMin==iMax-1){ 
        printf("%d: ",iCount++);
        for(int j=0;j<iMax;j++){
          printf("%d ",aBoard[j]);
        }
        printf("\n");
      }else{
        NQueen(iMin+1,iMax);
      }
  }  
}
int main(void) {
  NQueen(MINSIZE,MAXSIZE);
}

