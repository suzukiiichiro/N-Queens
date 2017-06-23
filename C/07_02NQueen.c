/**
  Cで学ぶアルゴリズムとデータ構造  
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
 
   １．ブルートフォース（力まかせ探索） NQueen01()
 <>２．配置フラグ（制約テスト高速化）   NQueen02()
   ３．バックトラック                   NQueen03() N17: 8:05
   ４．対称解除法(回転と斜軸）          NQueen04() N17: 7:54
   ５．枝刈りと最適化                   NQueen05() N17: 2:14
   ６．ビットマップ                     NQueen06() N17: 1:30
   ７．ビットマップ+対称解除法          NQueen07() N17: 2:24
   ８．ビットマップ+クイーンの場所で分岐NQueen08() N17: 1:26
   ９．ビットマップ+枝刈りと最適化      NQueen09() N17: 0:16
   10．もっとビットマップ(takaken版)    NQueen10() N17: 0:10
   11．マルチスレッド(構造体)           NQueen11() N17: 0:14
   12．マルチスレッド(pthread)          NQueen12() N17: 0:13
   13．マルチスレッド(join)             NQueen13() N17: 0:17
   14．マルチスレッド(mutex)            NQueen14() N17: 0:27
   15．マルチスレッド(アトミック対応)   NQueen15() N17: 0:05
   16．アドレスとポインタ               NQueen16() N17: 0:04
   17．アドレスとポインタ(脱構造体)     NQueen17() N17: 

  Java版はこちらを
  https://github.com/suzukiiichiro/AI_Algorithm_N-Queen/Java
  Lua版
  https://github.com/suzukiiichiro/AI_Algorithm_N-Queen/Lua
  Bash版
  https://github.com/suzukiiichiro/AI_Algorithm_N-Queen/Bash
 

  ２．配置フラグ（制約テスト高速化）
   パターンを生成し終わってからチェックを行うのではなく、途中で制約を満たさな
   い事が明らかな場合は、それ以降のパターン生成を行わない。
  「手を進められるだけ進めて、それ以上は無理（それ以上進めても解はない）という
  事がわかると一手だけ戻ってやり直す」という考え方で全ての手を調べる方法。
  (※)各行列に一個の王妃配置する組み合わせを再帰的に列挙分枝走査を行っても、組
  み合わせを列挙するだけであって、8王妃問題を解いているわけではありません。
 
  実行結果
  :
  :
  40312: 7 6 5 4 2 1 3 0
  40313: 7 6 5 4 2 3 0 1
  40314: 7 6 5 4 2 3 1 0
  40315: 7 6 5 4 3 0 1 2
  40316: 7 6 5 4 3 0 2 1
  40317: 7 6 5 4 3 1 0 2
  40318: 7 6 5 4 3 1 2 0
  40319: 7 6 5 4 3 2 0 1
  40320: 7 6 5 4 3 2 1 0
 */

#include <stdio.h>
#include <time.h>

#define MINSIZE 0
#define MAXSIZE 8

int iCount=1 ; //# c:count
int aBoard[MAXSIZE];
int fA[MAXSIZE]; //配置フラグ

void NQueen(int iMin,int iMax) {
  for(int i=0;i<iMax;i++){
    if(! fA[i]){
      aBoard[iMin]=i ;
      if(iMin==iMax-1){
        printf("%d: ",iCount++);
        for(int j=0;j<iMax;j++){
          printf("%d ",aBoard[j]);
        }
        printf("\n");
      }else{
        fA[i]=1;         
        NQueen(iMin+1,iMax);
        fA[i]=0; 
      }
    }
  }
}
int main(void) {
  NQueen(MINSIZE,MAXSIZE);
}
