/**
  Cで学ぶアルゴリズムとデータ構造  
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
 
 <>１．ブルートフォース（力まかせ探索） NQueen01()
   ２．配置フラグ（制約テスト高速化）   NQueen02()
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
   13．マルチスレッド(join)             NQueen13() 
   14．マルチスレッド(mutex)            NQueen14() 
   15．マルチスレッド(アトミック対応)   NQueen15() 
   16．アドレスとポインタ               NQueen16() 
   17．アドレスとポインタ(脱構造体)     NQueen17()
   18．アドレスとポインタ(脱配列)       NQueen18()

 # Java/C/Lua/Bash版
 # https://github.com/suzukiiichiro/N-Queen
 
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

int c=1 ;   // c:count
int aB[8];  // aB:aBboard[] 

// mi:min ma:max
void NQueen(int mi,int ma) {
  for(int i=0;i<ma;i++){
    aB[mi]=i ;
    if (mi==ma-1){ 
      printf("%d: ",c++);
      for(int j=0;j<ma;j++){
        printf("%d ",aB[j]);
      }
      printf("\n");
    }else{
      NQueen(mi+1,ma);
    }
  }  
}
int main(void) {
  NQueen(0,8);
  return 0;
}

