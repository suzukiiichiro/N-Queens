#!/bin/bash
#
#
# Bash（シェルスクリプト）で学ぶ「アルゴリズムとデータ構造」
# 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#
# ---------------------------------------------------------------------------------
##
# ２．配置フラグ（制約テスト高速化）
#  パターンを生成し終わってからチェックを行うのではなく、途中で制約を満たさな
#  い事が明らかな場合は、それ以降のパターン生成を行わない。
# 「手を進められるだけ進めて、それ以上は無理（それ以上進めても解はない）という
# 事がわかると一手だけ戻ってやり直す」という考え方で全ての手を調べる方法。
# (※)各行列に一個の王妃配置する組み合わせを再帰的に列挙分枝走査を行っても、組
# み合わせを列挙するだけであって、8王妃問題を解いているわけではありません。
#
# ---------------------------------------------------------------------------------
#
# 実行結果
#  :
#  :
#  40313: 7 6 5 4 2 3 0 1
#  40314: 7 6 5 4 2 3 1 0
#  40315: 7 6 5 4 3 0 1 2
#  40316: 7 6 5 4 3 0 2 1
#  40317: 7 6 5 4 3 1 0 2
#  40318: 7 6 5 4 3 1 2 0
#  40319: 7 6 5 4 3 2 0 1
#  40320: 7 6 5 4 3 2 1 0
#
COUNT=1; # グローバル変数は大文字
N-Queen2(){
  # ローカル変数は明示的に local をつけ、代入する場合は ""ダブルクォートが必要です。
  # -i は 変数の型が整数であることを示しています
  local -i min=$1;        # ひとつ目のパラメータ $1をminに代入
  local -i size=$2;       # ふたつ目のパラメータ $2をsizeに代入
  local flag_a="";
  local -i i=0;           # 再帰するために forで使う変数も宣言が必要
  local -i j=0;
  # forはこういうＣ的な書き方のほうが見やすい
  for((i=0;i<size;i++)){        # (()) の中の変数に $ は不要です 
    [ "${flag_a[i]}" != "true" ]&&{   #わかりづらいですが、この文はif文 文字列比較の場合は [ ] を使います
      pos[$min]="$i" ;          # 代入する場合、posの前には$ は不要ですが、添え字には$が必要
      ((min==(size-1)))&&{      # (()) の中の変数に $ は不要です
        echo -n "$((COUNT++)): ";     # $((COUNT++))はCOUNTのインクリメント
        for((j=0;j<size;j++)){
          echo -n "${pos[j]} " ;      # 配列変数を呼び出す場合は ${}をつけます
        }
        echo "" ;               # 改行を入れる
      }||{                      # elseのはじまり
        flag_a[$i]="true" ;     # 配列の中の添え字には $ をつけます 
        N-Queen2 "$((min+1))" "$size" ; # 再帰する場合は $((min++))ではなく $((min+1))
        flag_a[$i]="" ; 
      }
    }
  }
}
#
# 実行はコメントアウトを外して、 $ ./BASH_N-Queen.sh 
#
  echo "<>２．配置フラグ（制約テスト高速化） N-Queen2()";
  N-Queen2 0 8;
#
#
