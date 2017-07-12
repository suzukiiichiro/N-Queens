/**
  Cで学ぶアルゴリズムとデータ構造  
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
 
   １．ブルートフォース（力まかせ探索） NQueen01()
 <>２．配置フラグ（制約テスト高速化）   NQueen02()
   ３．バックトラック                   NQueen03() 
   ４．対称解除法(回転と斜軸）          NQueen04() 
   ５．枝刈りと最適化                   NQueen05() 
   ６．ビットマップ                     NQueen06() 
   ７．ビットマップ+対称解除法          NQueen07() 
   ８．ビットマップ+クイーンの場所で分岐NQueen08() 
   ９．ビットマップ+枝刈りと最適化      NQueen09() 
   10．もっとビットマップ(takaken版)    NQueen10() 
   11．マルチスレッド(構造体)           NQueen11() 
   12．マルチスレッド(pthread)          NQueen12() 
   13．マルチスレッド(mutex)            NQueen13() 
   14．マルチスレッド(mutexattr)        NQueen14() 
   15．マルチスレッド(脱mutex COUNT強化)NQueen15() 
   15t.もっとマルチスレッド(takaken版) NQueen15_t() 
   16．アドレスとポインタ(考察１)       NQueen16() 
   17．アドレスとポインタ(考察２)       NQueen17() 
   18．アドレスとポインタ(考察３)       NQueen18()
   19．アドレスとポインタ(考察４)       NQueen19()
   20．アドレスとポインタ(考察５)       NQueen20()

 # Java/C/Lua/Bash版
 # https://github.com/suzukiiichiro/N-Queen
 

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

int c=1 ;   // c:count
int aB[8];  // aB:aBoard[]
int fA[8];  // fA:flagArray[] 配置フラグ

// mi:min ma:max
void NQueen(int mi,int ma) {
  for(int i=0;i<ma;i++){
    if(fA[i]==0){
      aB[mi]=i ;
      if(mi==ma-1){
        printf("%d: ",c++);
        for(int j=0;j<ma;j++){
          printf("%d ",aB[j]);
        }
        printf("\n");
      }else{
        fA[i]=1;         
        NQueen(mi+1,ma);
        fA[i]=0; 
      }
    }
  }
}
int main(void) {
  NQueen(0,8);
  return 0;
}
