#!/bin/bash

#日本語
##
 # アルゴリズムとデータ構造  
 # 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
 #
 # Bash/Lua/Java/C/GPU版
 # https://github.com/suzukiiichiro/N-Queen
 #
 # ステップバイステップでＮ−クイーン問題を最適化
 #  １．ブルートフォース（力まかせ探索） NQueen1()
 #  ２．配置フラグ（制約テスト高速化）   NQueen2()
 #  ３．バックトラック                   NQueen3()

 #  ４．対称解除法
 #　５．枝刈り
 #  ６．ビットマップ
 #　７．クイーンの位置による振り分け
#
############################################
# N-Queen
############################################
#
##
# 再帰  Nクイーン問題
#
# https://ja.wikipedia.org/wiki/エイト・クイーン
#
# N-Queens問題とは
#    Nクイーン問題とは、「8列×8行のチェスボードに8個のクイーンを、互いに効きが
#    当たらないように並べよ」という８クイーン問題のクイーン(N)を、どこまで大き
#    なNまで解を求めることができるかという問題。
#    クイーンとは、チェスで使われているクイーンを指し、チェス盤の中で、縦、横、
#    斜めにどこまでも進むことができる駒で、日本の将棋でいう「飛車と角」を合わ
#    せた動きとなる。８列×８行で構成される一般的なチェスボードにおける8-Queens
#    問題の解は、解の総数は92個である。比較的単純な問題なので、学部レベルの演
#    習問題として取り上げられることが多い。
#    8-Queens問題程度であれば、人力またはプログラムによる「力まかせ探索」でも
#    解を求めることができるが、Nが大きくなると解が一気に爆発し、実用的な時間で
#    は解けなくなる。
#    現在すべての解が判明しているものは、2004年に電気通信大学で264CPU×20日をか
#    けてn=24を解決し世界一に、その後2005 年にニッツァ大学でn=25、2016年にドレ
#    スデン工科大学でn=27の解を求めることに成功している。
## 
#
# ---------------------------------------------------------------------------------
## 1. ブルートフォース　力任せ探索
#　全ての可能性のある解の候補を体系的に数え上げ、それぞれの解候補が問題の解と
#  なるかをチェックする方法
#  (※)各行に１個の王妃を配置する組み合わせを再帰的に列挙組み合わせを生成するだ
#  けであって8王妃問題を解いているわけではありません
# ---------------------------------------------------------------------------------
#
#  実行結果
#  :
#  :
#  16777209: 7 7 7 7 7 7 7 0
#  16777210: 7 7 7 7 7 7 7 1
#  16777211: 7 7 7 7 7 7 7 2
#  16777212: 7 7 7 7 7 7 7 3
#  16777213: 7 7 7 7 7 7 7 4
#  16777214: 7 7 7 7 7 7 7 5
#  16777215: 7 7 7 7 7 7 7 6
#  16777216: 7 7 7 7 7 7 7 7
#
COUNT=1 ; # グローバル変数は大文字
N-Queen1(){
  # ローカル変数は明示的に local をつけ、代入する場合は ""ダブルクォートが必要です。
  # -i は 変数の型が整数であることを示しています
  local -i min="$1";      # ひとつ目のパラメータ $1をminに代入
  local -i size="$2" ;    # ふたつ目のパラメータ $2をsizeに代入
  local -i i=0;           # 再帰するために forで使う変数も宣言が必要
  local -i j=0;
  # forはこういうＣ的な書き方のほうが見やすい
  for((i=0;i<size;i++)) { # (()) の中の変数に $ は不要です
    pos[$min]="$i" ;      # 代入する場合、posの前には$ は不要ですが、添え字には$が必要
    ((min==(size-1)))&&{  # わかりづらいですが、この文は if 文
      # echo -n は　行末で改行をしないオプション
      echo -n "$((COUNT++)): ";      # $((COUNT++))はCOUNTのインクリメント
      for((j=0;j<size;j++)){
        echo -n "${pos[j]} ";        # 配列変数を呼び出す場合は ${}をつけます
      }
      echo "" ;           # 改行を入れる
    # || パイプで繋いで処理を継続
    }||N-Queen1 "$((min+1))" "$size" ; # 再帰する場合は $((min++))ではなく $((min+1))
  }  
}
#
# 実行はコメントアウトを外して、 $ ./BASH_N-Queen.sh 
#
 # echo "N-Queen1 : ブルートフォース" ;
 # N-Queen1 0 8;
#
#
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
 # echo "N-Queen2 : 配置フラグ" ;
 # N-Queen2 0 8;
#
#
#
# ---------------------------------------------------------------------------------
##
# ３．バックトラック
# 　各列、対角線上にクイーンがあるかどうかのフラグを用意し、途中で制約を満た
# さない事が明らかな場合は、それ以降のパターン生成を行わない。
# 　各列、対角線上にクイーンがあるかどうかのフラグを用意することで高速化を図る。
# 　これまでは行方向と列方向に重複しない組み合わせを列挙するものですが、王妃
# は斜め方向のコマをとることができるので、どの斜めライン上にも王妃をひとつだ
# けしか配置できない制限を加える事により、深さ優先探索で全ての葉を訪問せず木
# を降りても解がないと判明した時点で木を引き返すということができます。
#
# ---------------------------------------------------------------------------------
#
# 実行結果
#
# N:        Total       Unique        hh:mm:ss
# 2:            0            0         0:00:00
# 3:            0            0         0:00:00
# 4:            2            0         0:00:00
# 5:           10            0         0:00:00
# 6:            4            0         0:00:00
# 7:           40            0         0:00:00
# 8:           92            0         0:00:01
# 9:          352            0         0:00:03
#10:          724            0         0:00:17
#11:         2680            0         0:01:23
#12:        14200            0         0:07:30
#
##
#
# グローバル変数は大文字
TOTAL=0;
UNIQUE=0;
typeset -a flag_a="";     # -a は配列の型を宣言します
typeset -a flag_b="";
typeset -a flag_c="";
N-Queen3_rec(){
  # ローカル変数は明示的に local をつけ、代入する場合は ""ダブルクォートが必要です。
  # -i は 変数の型が整数であることを示しています
  local -i min="$1";      # ひとつ目のパラメータ $1をminに代入
  local -i size=$2;       # ふたつ目のパラメータ $2をsizeに代入
  local -i i=0;           # 再帰するために forで使う変数も宣言が必要
  # forはこういうＣ的な書き方のほうが見やすい
  for((i=0;i<size;i++)){        # (()) の中の変数に $ は不要です 
    #わかりづらいですが、この文はif文 文字列比較の場合は [ ] を使います
    # 長い文章は \ （スペースバックスラッシュ）で改行することができます
    [ "${flag_a[$i]}" != "true"  ]&& \
    [ "${flag_b[$min+$i]}" != "true" ]&& \
    [ "${flag_c[$min-$i+$size-1]}" != "true" ]&&{ #この文はif文 文字列比較の場合は [ ] を使います
      pos[$min]=$i ;            # 代入する場合、posの前には$ は不要ですが、添え字には$が必要
      ((min==(size-1)))&&{      # (()) の中の変数に $ は不要です
        ((TOTAL++));            # ((TOTAL++))はTOTALのインクリメント (()) の中の変数に $ は不要です
      }||{                      # elseのはじまり                     
        flag_a[$i]="true" ;     # 配列の中の添え字には $ をつけます 
        flag_b[$min+$i]="true" ; 
        flag_c[$min-$i+$size-1]="true" ; 
        N-Queen3_rec "$((min+1))" "$size" ; # 再帰する場合は $((min++))ではなく $((min+1))
        flag_a[$i]="" ;           
        flag_b[$min+$i]="" ;   
        flag_c[$min-$i+$size-1]="" ; 
      }          
    }
  }  
}
#
N-Queen3(){
  local -i max=15;
  local -i min=2;
  local -i N="$min";
  local startTime=0;
	local endTime=0;
	local hh=mm=ss=0; 		# いっぺんにに初期化することもできます
  echo " N:        Total       Unique        hh:mm:ss" ;
  for((N=min;N<=max;N++)){
    TOTAL=0;      # Nが更新される度に TOTALとUNIQUEを初期化
    UNIQUE=0;
    startTime=`date +%s` ;      # 計測開始時間
    N-Queen3_rec 0 "$N";
		endTime=`date +%s`;					# 計測終了時間
		ss=`expr ${endTime} - ${startTime}` # hh:mm:ss 形式に変換
		hh=`expr ${ss} / 3600`
		ss=`expr ${ss} % 3600`
		mm=`expr ${ss} / 60`
		ss=`expr ${ss} % 60`
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $N $TOTAL $UNIQUE $hh $mm $ss ;
  } 
}
#
# 実行はコメントアウトを外して、 $ ./BASH_N-Queen.sh 
#
  # echo "N-Queen3 : バックトラック";
  # N-Queen3;
#
#
#
# ---------------------------------------------------------------------------------
# ４．バックトラック＋対象解除法
# 
# 　一つの解には、盤面を９０度、１８０度、２７０度回転、及びそれらの鏡像の合計
# 　８個の対称解が存在する。対照的な解を除去し、ユニーク解から解を求める手法。
# 
# ■ユニーク解の判定方法
#   全探索によって得られたある１つの解が、回転・反転などによる本質的に変わること
# のない変換によって他の解と同型となるものが存在する場合、それを別の解とはしない
# とする解の数え方で得られる解を「ユニーク解」といいます。つまり、ユニーク解とは、
# 全解の中から回転・反転などによる変換によって同型になるもの同士をグループ化する
# ことを意味しています。
# 
#   従って、ユニーク解はその「個数のみ」に着目され、この解はユニーク解であり、こ
# の解はユニーク解ではないという定まった判定方法はありません。ユニーク解であるか
# どうかの判断はユニーク解の個数を数える目的の為だけに各個人が自由に定義すること
# になります。もちろん、どのような定義をしたとしてもユニーク解の個数それ自体は変
# わりません。
# 
#   さて、Ｎクイーン問題は正方形のボードで形成されるので回転・反転による変換パター
# ンはぜんぶで８通りあります。だからといって「全解数＝ユニーク解数×８」と単純には
# いきません。ひとつのグループの要素数が必ず８個あるとは限らないのです。Ｎ＝５の
# 下の例では要素数が２個のものと８個のものがあります。
#
#
# Ｎ＝５の全解は１０、ユニーク解は２なのです。
# 
# グループ１: ユニーク解１つ目
# - - - Q -   - Q - - -
# Q - - - -   - - - - Q
# - - Q - -   - - Q - -
# - - - - Q   Q - - - -
# - Q - - -   - - - Q -
# 
# グループ２: ユニーク解２つ目
# - - - - Q   Q - - - -   - - Q - -   - - Q - -   - - - Q -   - Q - - -   Q - - - -   - - - - Q
# - - Q - -   - - Q - -   Q - - - -   - - - - Q   - Q - - -   - - - Q -   - - - Q -   - Q - - -
# Q - - - -   - - - - Q   - - - Q -   - Q - - -   - - - - Q   Q - - - -   - Q - - -   - - - Q -
# - - - Q -   - Q - - -   - Q - - -   - - - Q -   - - Q - -   - - Q - -   - - - - Q   Q - - - -
# - Q - - -   - - - Q -   - - - - Q   Q - - - -   Q - - - -   - - - - Q   - - Q - -   - - Q - -
#
# 
#   それでは、ユニーク解を判定するための定義付けを行いますが、次のように定義する
# ことにします。各行のクイーンが右から何番目にあるかを調べて、最上段の行から下
# の行へ順番に列挙します。そしてそれをＮ桁の数値として見た場合に最小値になるもの
# をユニーク解として数えることにします。尚、このＮ桁の数を以後は「ユニーク判定値」
# と呼ぶことにします。
# 
# - - - - Q   0
# - - Q - -   2
# Q - - - -   4   --->  0 2 4 1 3  (ユニーク判定値)
# - - - Q -   1
# - Q - - -   3
# 
# 
#   探索によって得られたある１つの解(オリジナル)がユニーク解であるかどうかを判定
# するには「８通りの変換を試み、その中でオリジナルのユニーク判定値が最小であるか
# を調べる」ことになります。しかし結論から先にいえば、ユニーク解とは成り得ないこ
# とが明確なパターンを探索中に切り捨てるある枝刈りを組み込むことにより、３通りの
# 変換を試みるだけでユニーク解の判定が可能になります。
#  
# 
# ■ユニーク解の個数を求める
#   先ず最上段の行のクイーンの位置に着目します。その位置が左半分の領域にあればユ
# ニーク解には成り得ません。何故なら左右反転によって得られるパターンのユニーク判
# 定値の方が確実に小さくなるからです。また、Ｎが奇数の場合に中央にあった場合はど
# うでしょう。これもユニーク解には成り得ません。何故なら仮に中央にあった場合、そ
# れがユニーク解であるためには少なくとも他の外側の３辺におけるクイーンの位置も中
# 央になければならず、それは互いの効き筋にあたるので有り得ません。
#
#   TOTAL = (COUNT8  8) + (COUNT4  4) + (COUNT2  2);
#     (1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルか
#      ら180度回転)させても、さらに90度回転(オリジナルから270度回転)させてもオリ
#      ジナルと同型になる。  
#  
#      COUNT2  2
#   
#     (2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナ
#      ルとは異なる。ただし、180度回転させた場合はオリジナルと同型になることも有
#      り得る。 
#  
#      COUNT4  4
#   
#     (3) (1) に該当するユニーク解が属するグループの要素数は、左右反転させたパターンを
#         加えて２個しかありません。(2)に該当するユニーク解が属するグループの要素数は、
#         180度回転させて同型になる場合は４個(左右反転×縦横回転)、そして180度回転させても
#         オリジナルと異なる場合は８個になります。(左右反転×縦横回転×上下反転)
#   
#      COUNT8  8 
#  
#     以上のことから、ひとつひとつのユニーク解が上のどの種類に該当するのかを調べる
#   ことにより全解数を計算で導き出すことができます。探索時間を短縮させてくれる枝刈
#   りを外す必要がなくなったというわけです。 
#   
#     UNIQUE  COUNT2      +  COUNT4      +  COUNT8
#     TOTAL  (COUNT2  2) + (COUNT4  4) + (COUNT8  8)
#  
#   　これらを実現すると、前回のNQueen3()よりも実行速度が遅くなります。
#   　なぜなら、対称・反転・斜軸を反転するための処理が加わっているからです。
#   ですが、今回の処理を行うことによって、さらにNQueen5では、処理スピードが飛
#   躍的に高速化されます。そのためにも今回のアルゴリズム実装は必要なのです。
#
# 実行結果
# N-Queen4 : バックトラック＋対称解除法
#  N:        Total       Unique        hh:mm:ss
#  2:            0            0         0:00:00
#  3:            0            0         0:00:00
#  4:            2            1         0:00:00
#  5:           10            2         0:00:01
#  6:            4            1         0:00:00
#  7:           40            6         0:00:00
#  8:           92           12         0:00:01
#  9:          352           46         0:00:05
# 10:          724           92         0:00:16
# 11:         2680          341         0:01:14
# 12:        14200         1787         0:06:48
#
#
# グローバル変数は大文字
typeset -i TOTAL=0;
typeset -i UNIQUE=0;
typeset -a flag_a="";     # -a は配列の型を宣言します
typeset -a flag_b="";
typeset -a flag_c="";
typeset -a trial="";
typeset -a board="";
typeset -a scratch="";
#
function intncmp(){
  local -i k; 
  local -i rtn=0;
  local -i n=$1;
  for((k=0;k<n;k++)){
    rtn=$((board[k]-trial[k]));
    ((rtn!=0))&&{ break; }
  }
  echo "$rtn";
}
#
function rotate() {
  local -i j;
  local -i k;
  local -i n=$1;
  local -i incr;
  local neg=$2;
  if [ "$neg" = "true" ];then
    k=0;
  else
    k=$((n-1)); 
  fi 
  if [ "$neg" = "true" ];then
    incr=$((incr+1));
  else
    incr=$((incr-1));
  fi 
  for((j=0;j<n;k+=incr)){ 
    j=$((j+1))
    scratch[$j]=${trial[$k]};
  }
  if [ "$neg" = "true" ];then
    k=$((n-1));
  else
    k=0;
  fi 
  for((j=0;j<n;k-=incr)){ 
    j=$((j+1))
    trial[${scratch[$j]}]=$k;
  }
}
#
function vMirror(){
  local -i j;
  local -i n=$1;
  for((j=0;j<n;j++)){
    local -i n1=$((n-1));
    trial[$j]=$((n1-trial[j]));
  }
}
#
function symmetryOps() {
  local -i k;
  local -i nEquiv;
  local -i size=$1;
  
  #// 回転・反転・対称チェックのためにboard配列をコピー
  for((k=0;k<size;k++)){
    trial[$k]=${board[$k]};
  }
  #//時計回りに90度回転
  rotate "$size" "false";
  k=$(intncmp "$size");
  ((k>0))&&{
    echo 0; 
    return;
  }  
  ((k==0))&&{
     nEquiv=1;
  }||{
   #//時計回りに180度回転
     rotate "$size" "false";
     k=$(intncmp "$size");
     ((k>0))&&{
       echo 0; 
       return;
     }
     ((k==0))&&{
        nEquiv=2;
   }||{
      #//時計回りに270度回転
      rotate "$size" "false";
      k=$(intncmp "$size");
      ((k>0))&&{
        echo 0; 
        return;
      }  
      nEquiv=4;
     }
  }
  #// 回転・反転・対称チェックのためにboard配列をコピー
  for((k=0;k<size;k++)){ 
    trial[$k]=${board[$k]};
  }
  #//垂直反転
  vMirror "$size";
  k=$(intncmp "$size");
  ((k>0))&&{ 
    echo 0; 
    return;
  }
  #// 4回転とは異なる場合
  ((nEquiv>1))&&{
   #// -90度回転 対角鏡と同等
     rotate "$size" "true";
     k=$(intncmp "$size");
     ((k>0))&&{
       echo 0;
       return;
     }
     ((nEquiv>2))&&{     #// 2回転とは異なる場合
      #// -180度回転 水平鏡像と同等
        rotate "$size" "true";
        k=$(intncmp "$size");
        ((k>0))&&{ 
          echo 0;
          return;
        }
        #// -270度回転 反対角鏡と同等
        rotate "$size" "true";
        k=$(intncmp "$size");
        ((k>0))&&{
          echo 0;
          return;
        }
     }
  }
  rtn=$((nEquiv * 2));
  echo "$rtn";
  return;
 }
N-Queen4_rec(){
  # ローカル変数は明示的に local をつけ、代入する場合は ""ダブルクォートが必要です。
  # -i は 変数の型が整数であることを示しています
  local -i min="$1";                # ひとつ目のパラメータ $1をminに代入
  local -i size=$2;                 # ふたつ目のパラメータ $2をsizeに代入
  local -i i=0;                     # 再帰するために forで使う変数も宣言が必要
  # forはこういうＣ的な書き方のほうが見やすい
  for((i=0;i<size;i++)){        # (()) の中の変数に $ は不要です 
    #わかりづらいですが、この文はif文 文字列比較の場合は [ ] を使います
    # 長い文章は \ （スペースバックスラッシュ）で改行することができます
    [ "${flag_a[$i]}" != "true"  ]&& \
    [ "${flag_b[$min+$i]}" != "true" ]&& \
    [ "${flag_c[$min-$i+$size-1]}" != "true" ]&&{   #この文はif文 文字列比較の場合は [ ] を使います
      board[$min]=$i ;              # 代入する場合、boardの前には$ は不要ですが、添え字には$が必要
      ((min==(size-1)))&&{          # (()) の中の変数に $ は不要です
        tst=$(symmetryOps "$size");
        ((tst!=0))&&{
          ((UNIQUE++));             # ((TOTAL++))はTOTALのインクリメント (()) の中の変数に $ は不要です
          TOTAL=$((TOTAL+tst));     # ((TOTAL++))はTOTALのインクリメント (()) の中の変数に $ は不要です
        }
      }||{                          # elseのはじまり                     
        flag_a[$i]="true";          # 配列の中の添え字には $ をつけます 
        flag_b[$min+$i]="true"; 
        flag_c[$min-$i+$size-1]="true"; 
        N-Queen4_rec "$((min+1))" "$size"; # 再帰する場合は $((min++))ではなく $((min+1))
        flag_a[$i]="";           
        flag_b[$min+$i]="";   
        flag_c[$min-$i+$size-1]=""; 
      }          
    }
  }  
}
#
N-Queen4(){
  local -i max=15;
  local -i min=2;
  local -i N="$min";
  local startTime=0;
	local endTime=0;
	local hh=mm=ss=0; 		# いっぺんにに初期化することもできます
  echo " N:        Total       Unique        hh:mm:ss";
  for((N=min;N<=max;N++)){
    TOTAL=0;      # Nが更新される度に TOTALとUNIQUEを初期化
    UNIQUE=0;
    startTime=`date +%s`;      # 計測開始時間
    for((k=0;k<N;k++)){ board[k]=k;}
    N-Queen4_rec 0 "$N";
		endTime=`date +%s`;					# 計測終了時間
		ss=`expr ${endTime} - ${startTime}`; # hh:mm:ss 形式に変換
		hh=`expr ${ss} / 3600`;
		ss=`expr ${ss} % 3600`;
		mm=`expr ${ss} / 60`;
		ss=`expr ${ss} % 60`;
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $N $TOTAL $UNIQUE $hh $mm $ss ;
  } 
}
#
# 実行はコメントアウトを外して、 $ ./BASH_N-Queen.sh 
  # echo "N-Queen4 : バックトラック＋対称解除法";
  # N-Queen4;
#
#
#
# ---------------------------------------------------------------------------------
# ５．枝刈りと最適化
#
# N-Queen5 : バックトラック＋対称解除法＋枝刈りと最適化
#  N:        Total       Unique        hh:mm:ss
#  2:            0            0         0:00:00
#  3:            0            0         0:00:00
#  4:            2            1         0:00:00
#  5:           10            2         0:00:00
#  6:            4            1         0:00:00
#  7:           40            6         0:00:00
#  8:           92           12         0:00:01
#  9:          352           46         0:00:03
# 10:          724           92         0:00:08
# 11:         2680          341         0:00:45
# 12:        14200         1787         0:03:36
#
#
# グローバル変数は大文字
typeset -i TOTAL=0;
typeset -i UNIQUE=0;
typeset -a flag_a="";     # -a は配列の型を宣言します
typeset -a flag_b="";
typeset -a flag_c="";
typeset -a trial="";
typeset -a board="";
typeset -a scratch="";
#
function intncmp(){
  local -i k; 
  local -i rtn=0;
  local -i n=$1;
  for((k=0;k<n;k++)){
    rtn=$((board[k]-trial[k]));
    ((rtn!=0))&&{ break; }
  }
  echo "$rtn";
}
#
function rotate() {
  local -i j;
  local -i k;
  local -i n=$1;
  local -i incr;
  local neg=$2;
  if [ "$neg" = "true" ];then
    k=0;
  else
    k=$((n-1)); 
  fi 
  if [ "$neg" = "true" ];then
    incr=$((incr+1));
  else
    incr=$((incr-1));
  fi 
  for((j=0;j<n;k+=incr)){ 
    j=$((j+1))
    scratch[$j]=${trial[$k]};
  }
  if [ "$neg" = "true" ];then
    k=$((n-1));
  else
    k=0;
  fi 
  for((j=0;j<n;k-=incr)){ 
    j=$((j+1))
    trial[${scratch[$j]}]=$k;
  }
}
#
function vMirror(){
  local -i j;
  local -i n=$1;
  for((j=0;j<n;j++)){
    local -i n1=$((n-1));
    trial[$j]=$((n1-trial[j]));
  }
}
#
function symmetryOps() {
  local -i k;
  local -i nEquiv;
  local -i size=$1;
  
  #// 回転・反転・対称チェックのためにboard配列をコピー
  for((k=0;k<size;k++)){
    trial[$k]=${board[$k]};
  }
  #//時計回りに90度回転
  rotate "$size" "false";
  k=$(intncmp "$size");
  ((k>0))&&{
    echo 0; 
    return;
  }  
  ((k==0))&&{
     nEquiv=1;
  }||{
   #//時計回りに180度回転
     rotate "$size" "false";
     k=$(intncmp "$size");
     ((k>0))&&{
       echo 0; 
       return;
     }
     ((k==0))&&{
        nEquiv=2;
   }||{
      #//時計回りに270度回転
      rotate "$size" "false";
      k=$(intncmp "$size");
      ((k>0))&&{
        echo 0; 
        return;
      }  
      nEquiv=4;
     }
  }
  #// 回転・反転・対称チェックのためにboard配列をコピー
  for((k=0;k<size;k++)){ 
    trial[$k]=${board[$k]};
  }
  #//垂直反転
  vMirror "$size";
  k=$(intncmp "$size");
  ((k>0))&&{ 
    echo 0; 
    return;
  }
  #// 4回転とは異なる場合
  ((nEquiv>1))&&{
   #// -90度回転 対角鏡と同等
     rotate "$size" "true";
     k=$(intncmp "$size");
     ((k>0))&&{
       echo 0;
       return;
     }
     ((nEquiv>2))&&{     #// 2回転とは異なる場合
      #// -180度回転 水平鏡像と同等
        rotate "$size" "true";
        k=$(intncmp "$size");
        ((k>0))&&{ 
          echo 0;
          return;
        }
        #// -270度回転 反対角鏡と同等
        rotate "$size" "true";
        k=$(intncmp "$size");
        ((k>0))&&{
          echo 0;
          return;
        }
     }
  }
  rtn=$((nEquiv * 2));
  echo "$rtn";
  return;
 }
#
N-Queen5_rec(){
  # ローカル変数は明示的に local をつけ、代入する場合は ""ダブルクォートが必要です。
  # -i は 変数の型が整数であることを示しています
  local -i min="$1";      # ひとつ目のパラメータ $1をminに代入
  local -i size=$2;       # ふたつ目のパラメータ $2をsizeに代入
  local -i i=0;           # 再帰するために forで使う変数も宣言が必要
  local -i s;
  local -i lim;
  local -i vTemp;

  ((min<size-1))&&{
		# flag_aを枝刈りによって使う必要がなくなった
    [ "${flag_c[$min-${board[$min]}+$size-1]}" != "true" ]&& \
    [ "${flag_b[$min+${board[$min]}]}" != "true" ]&&{ 
	    flag_c[$min-${board[$min]}+$size-1]="true";
      flag_b[$min+${board[$min]}]="true";
      N-Queen5_rec "$((min+1))" "$size";
	    flag_c[$min-${board[$min]}+$size-1]=""; 
      flag_b[$min+${board[$min]}]="";
    }
		# 枝刈り
		((min != 0))&&{
			lim=$size;
		}||{
			lim=$(((size+1)/2)); 
		}
		for((s=min+1;s<lim;s++)){
			vTemp=${board[$s]};
			board[$s]=${board[$min]};
			board[$min]=${vTemp};
			# flag_aを枝刈りによって使う必要がなくなった
			[ "${flag_c[$min-${board[$min]}+$size-1]}" != "true" ]&& \
			[ "${flag_b[$min+${board[$min]}]}" != "true" ]&& {
				flag_c[$min-${board[$min]}+$size-1]="true"; 
				flag_b[$min+${board[$min]}]="true";
				N-Queen5_rec "$((min+1))" "$size";
				flag_c[$min-${board[$min]}+$size-1]=""; 
				flag_b[$min+${board[$min]}]="";
			}
		}
		vTemp=${board[$min]};
		for((s=min+1;s<size;s++)){
			board[$s-1]=${board[$s]};
		}
		board[$s-1]=${vTemp};
	}||{ 
		if [ "${flag_c[$min-${board[$min]}+$size-1]}" = "true" -o "${flag_b[$min+${board[$min]}]}" == "true" ];then
			return;
		fi	
		tst=$(symmetryOps "$size");
		((tst!=0))&&{
			((UNIQUE++));
			TOTAL=$((TOTAL+tst));
		}
	}
	return;
}
#
N-Queen5(){
  local -i max=15;
  local -i min=2;
  local -i N="$min";
  local startTime=0;
	local endTime=0;
	local hh=mm=ss=0; 		# いっぺんにに初期化することもできます
  echo " N:        Total       Unique        hh:mm:ss";
  for((N=min;N<=max;N++)){
    TOTAL=0;      # Nが更新される度に TOTALとUNIQUEを初期化
    UNIQUE=0;
    startTime=`date +%s`;      # 計測開始時間
    for((k=0;k<N;k++)){ board[$k]=$k;}
    N-Queen5_rec 0 "$N";
		endTime=`date +%s`;					# 計測終了時間
		ss=`expr ${endTime} - ${startTime}`; # hh:mm:ss 形式に変換
		hh=`expr ${ss} / 3600`;
		ss=`expr ${ss} % 3600`;
		mm=`expr ${ss} / 60`;
		ss=`expr ${ss} % 60`;
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $N $TOTAL $UNIQUE $hh $mm $ss ;
  } 
}
#
# 実行はコメントアウトを外して、 $ ./BASH_N-Queen.sh 
  # echo "N-Queen5 : バックトラック＋対称解除法＋枝刈りと最適化";
  # N-Queen5;
#
#
#
# ---------------------------------------------------------------------------------
##
# ６．バックトラック＋ビットマップ
#
#   ビット演算を使って高速化 状態をビットマップにパックし、処理する
#   単純なバックトラックよりも２０〜３０倍高速
# 
# 　ビットマップであれば、シフトにより高速にデータを移動できる。
#  フラグ配列ではデータの移動にO(N)の時間がかかるが、ビットマップであればO(1)
#  フラグ配列のように、斜め方向に 2*N-1の要素を用意するのではなく、Nビットで充
#  分。
#
# 　配置可能なビット列を flags に入れ、-flags & flags で順にビットを取り出し処理。
# 　バックトラックよりも２０−３０倍高速。
# 
# ===================
# 考え方 1
# ===================
#
# 　Ｎ×ＮのチェスボードをＮ個のビットフィールドで表し、ひとつの横列の状態をひと
# つのビットフィールドに対応させます。(クイーンが置いてある位置のビットをONに
# する)
# 　そしてバックトラッキングは0番目のビットフィールドから「下に向かって」順にい
# ずれかのビット位置をひとつだけONにして進めていきます。
#
# 
#- - - - - Q - -    00000100 0番目のビットフィールド
#- - - Q - - - -    00010000 1番目のビットフィールド
#- - - - - - Q -    00000010 2番目のビットフィールド
#Q - - - - - - -    10000000 3番目のビットフィールド
#- - - - - - - Q    00000001 4番目のビットフィールド
#- Q - - - - - -    01000000 5番目のビットフィールド
#- - - - Q - - -    00001000 6番目のビットフィールド
#- - Q - - - - -    00100000 7番目のビットフィールド
#
#
# ===================
# 考え方 2
# ===================
#
# 次に、効き筋をチェックするためにさらに３つのビットフィールドを用意します。
#
# 1. 左下に効き筋が進むもの: left 
# 2. 真下に効き筋が進むもの: down
# 3. 右下に効き筋が進むもの: right
#
#次に、斜めの利き筋を考えます。
# 上図の場合、
# 1列目の右斜め上の利き筋は 3 番目 (0x08)
# 2列目の右斜め上の利き筋は 2 番目 (0x04) になります。
# この値は 0 列目のクイーンの位置 0x10 を 1 ビットずつ「右シフト」すれば求める
# ことができます。
# また、左斜め上の利き筋の場合、1 列目では 5 番目 (0x20) で 2 列目では 6 番目 (0x40)
#になるので、今度は 1 ビットずつ「左シフト」すれば求めることができます。
#
#つまり、右シフトの利き筋を right、左シフトの利き筋を left で表すことで、クイー
#ンの効き筋はrightとleftを1 ビットシフトするだけで求めることができるわけです。
#
#  *-------------
#  | . . . . . .
#  | . . . -3. .  0x02 -|
#  | . . -2. . .  0x04  |(1 bit 右シフト right)
#  | . -1. . . .  0x08 -|
#  | Q . . . . .  0x10 ←(Q の位置は 4   down)
#  | . +1. . . .  0x20 -| 
#  | . . +2. . .  0x40  |(1 bit 左シフト left)  
#  | . . . +3. .  0x80 -|
#  *-------------
#  図：斜めの利き筋のチェック
#
# n番目のビットフィールドからn+1番目のビットフィールドに探索を進めるときに、そ
# の３つのビットフィールドとn番目のビットフィールド(bit)とのOR演算をそれぞれ行
# います。leftは左にひとつシフトし、downはそのまま、rightは右にひとつシフトして
# n+1番目のビットフィールド探索に渡してやります。
#
# left : (left |bit)<<1
# right: (right|bit)>>1
# down :   down|bit
#
#
# ===================
# 考え方 3
# ===================
#
#   n+1番目のビットフィールドの探索では、この３つのビットフィールドをOR演算した
# ビットフィールドを作り、それがONになっている位置は効き筋に当たるので置くことが
# できない位置ということになります。次にその３つのビットフィールドをORしたビッ
# トフィールドをビット反転させます。つまり「配置可能なビットがONになったビットフィー
# ルド」に変換します。そしてこの配置可能なビットフィールドを bitmap と呼ぶとして、
# 次の演算を行なってみます。
# 
# bit = -bitmap & bitmap; //一番右のビットを取り出す
# 
#   この演算式の意味を理解するには負の値がコンピュータにおける２進法ではどのよう
# に表現されているのかを知る必要があります。負の値を２進法で具体的に表わしてみる
# と次のようになります。
# 
#  00000011   3
#  00000010   2
#  00000001   1
#  00000000   0
#  11111111  -1
#  11111110  -2
#  11111101  -3
# 
#   正の値nを負の値-nにするときは、nをビット反転してから+1されています。そして、
# 例えばn=22としてnと-nをAND演算すると下のようになります。nを２進法で表したときの
# 一番下位のONビットがひとつだけ抽出される結果が得られるのです。極めて簡単な演算
# によって1ビット抽出を実現させていることが重要です。
# 
#      00010110   22
#  AND 11101010  -22
# ------------------
#      00000010
# 
#   さて、そこで下のようなwhile文を書けば、このループは bitmap のONビットの数の
# 回数だけループすることになります。配置可能なパターンをひとつずつ全く無駄がなく
# 生成されることになります。
# 
# while (bitmap) {
#     bit = -bitmap & bitmap;
#     bitmap ^= bit;
#     //ここでは配置可能なパターンがひとつずつ生成される(bit) 
# }
#
#	実行結果
# N-Queen6 : バックトラック＋ビットマップ
#  N:        Total       Unique        hh:mm:ss
#  2:            0            0         0:00:00
#  3:            0            0         0:00:00
#  4:            2            0         0:00:00
#  5:           10            0         0:00:00
#  6:            4            0         0:00:00
#  7:           40            0         0:00:00
#  8:           92            0         0:00:00
#  9:          352            0         0:00:01
# 10:          724            0         0:00:05
# 11:         2680            0         0:00:21
# 12:        14200            0         0:01:52
#
#
#
typeset -i TOTAL=0;
typeset -i UNIQUE=0;
typeset -i size=0;
typeset -i MASK=0;
N-Queen6_rec(){
	#y: l:left d:down r:right b:bit bm:bitmap
  local -i min="$1";
	local -i left="$2";
	local -i down="$3";
	local -i right="$4";
	local -i bitmap=;
	local -i bit=;
  ((min==size))&&((TOTAL++))||{
    bitmap=$((MASK&~(left|down|right)));
    while ((bitmap)); do
      bit=$((-bitmap&bitmap)) ;
      bitmap=$((bitmap^bit)) ;
      N-Queen6_rec "$((min+1))" "$(((left|bit)<<1))" "$((down|bit))" "$(((right|bit)>>1))"  ;
    done
  }
}
N-Queen6(){
  local -i max=15;
	local -i min=2;
	local startTime=;
	local endTime= ;
	local hh=mm=ss=0; 		# いっぺんにに初期化することもできます
  echo " N:        Total       Unique        hh:mm:ss" ;
  for ((size=min;size<=max;size++)) {
    TOTAL=0;
		UNIQUE=0;
		MASK=$(((1<<size)-1));
		startTime=`date +%s` ;
    N-Queen6_rec 0 0 0 0 ;
    endTime=$((`date +%s` - st)) ;
		ss=`expr ${endTime} - ${startTime}`; # hh:mm:ss 形式に変換
		hh=`expr ${ss} / 3600`;
		ss=`expr ${ss} % 3600`;
		mm=`expr ${ss} / 60`;
		ss=`expr ${ss} % 60`;
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $size $TOTAL $UNIQUE $hh $mm $ss ;
  } 
}

# 実行はコメントアウトを外して、 $ ./BASH_N-Queen.sh 
   # echo "N-Queen6 : バックトラック＋ビットマップ";
   # N-Queen6;
#
#
# ---------------------------------------------------------------------------------
##
# ７．バックトラック＋ビットマップ＋対称解除法
#
#
typeset -i TOTAL=0;
typeset -i UNIQUE=0;
typeset -i size=0;
typeset -i MASK=0;
typeset -a board="";
typeset -a trial="";
typeset -a scratch="";
typeset -i COUNT2=0;
typeset -i COUNT4=0;
typeset -i COUNT8=0;
#
function getUnique(){ 
  echo $((COUNT2+COUNT4+COUNT8));
}
#
function getTotal(){ 
  echo $(( COUNT2*2 + COUNT4*4 + COUNT8*8));
}
#
function rotate_bitmap_ts(){
  local -i t=0;
  for((i=0;i<size;i++)){
    #local -i t=0;
    t=0;
    for((j=0;j<size;j++)){
      ((t|=((trial[j]>>i)&1)<<(size-j-1))); 
    }
    scratch[$i]=$t; 
  }
}
#
function rotate_bitmap_st(){
  local -i t=0;
  for((i=0;i<size;i++)){
    #local -i t=0;
    t=0;
    for((j=0;j<size;j++)){
      ((t|=((scratch[j]>>i)&1)<<(size-j-1))); 
    }
    trial[$i]=$t; 
  }
}
#
function rh(){
  local -i a=$1;
  local -i sz=$2;
  local -i tmp=0;
  for((i=0;i<=sz;i++)){
    ((a&(1<<i)))&&{ 
     #echo $((tmp|=(1<<(sz-i)))); 
     let tmp="tmp|=(1<<(sz-i))"; 
    }
  }
  echo $tmp;
}
#
function vMirror_bitmap(){
#  local -i j;
#  local -i n=$1;
#  local -i n1=$((size-1));
#  for((i=0;i<size;i++)){
#    #n1=$((size-1));
#    trial[$i]=$((n1-trial[i]));
#  }
  local -i score=0;
  local -i sizeE=$((size-1));
  for((i=0;i<size;i++)){
    score=${scratch[$i]};
    trial[$i]=$(rh "$score" $sizeE);
    #trial[$i]=$(rh "$score");
  }
}
function intncmp(){
#  local -i k; 
#  local -i rtn=0;
#  local -i n=$1;
  #for((k=0;k<n;k++)){
  #for((k=0;k<size;k++)){
  for((i=0;i<size;i++)){
    rtn=$((board[i]-trial[i]));
    ((rtn!=0))&&{ break; }
  }
  echo "$rtn";
#  local -a lt=$1; 
#  local -a rt=$2;
##  local -i si=$3;
#  local -i rtn=0;
#  local -i ltk=0;
#  local -i rtk=0;
#  for((k=0;k<size;k++)){
#    ltk=${lt[$k]};
#    rtk=${rt[$k]};
#    rtn=$((ltk-rtk));
#    ((rtn!=0))&&{ 
#     break;
#    }
#  }
}
function intncmp_bs(){
  local -i rtn=0;
  for((i=0;i<size;i++)){
    #rtn=$((board[i]-scratch[i]));
    rtn=$(echo "${board[$i]}-${scratch[$i]}"|bc);
    ((rtn!=0))&&{ break; }
  }
  echo "$rtn";
}
function intncmp_bt(){
  local -i rtn=0;
  for((i=0;i<size;i++)){
    #rtn=$((board[i]-trial[i]));
    rtn=$(echo "${board[$i]}-${trial[$i]}"|bc);
    ((rtn!=0))&&{ break; }
  }
  echo "$rtn";
}
function symmetryOps_bm(){
#  local -i si=$1;
  local -i nEquiv=0;
  #回転・反転・対称チェックのためにboard配列をコピー
  for((i=0;i<size;i++)){ 
    #trial[$i]=$board{[$i]};
    trial[$i]=${board[$i]};
  }
  #rotate_bitmap_ts "$size";
  rotate_bitmap_ts; 
  #    //時計回りに90度回転
  #k=$(intncmp "${board}" "${scratch}" "$size");
  k=$(intncmp_bs);
  ((k>0))&&{ 
#  echo "1:$k";
   return; 
  }
  ((k==0))&&{ 
    nEquiv=2;
#    echo "2:$k";
  }||{
    #rotate_bitmap_st "$size";
    rotate_bitmap_st;
    #  //時計回りに180度回転
    #k=$(intncmp "${board}" "${trial}" "$size");
    k=$(intncmp_bt);
    ((k>0))&&{ 
#     echo "3:$k";
     return; 
    }
    ((k==0))&&{ 
#      echo "4:$k";
      nEquiv=4;
    }||{
      #rotate_bitmap_ts "$size";
      rotate_bitmap_ts;
      #//時計回りに270度回転
      #k=$(intncmp "${board}" "${scratch}" "$size");
      k=$(intncmp_bs);
      ((k>0))&&{ 
#        echo "5:$k";
        return;
      }
#      echo "6:$k";
      nEquiv=8;
    }
  }
  #// 回転・反転・対称チェックのためにboard配列をコピー
  for((i=0;i<size;i++)){ 
    scratch[$i]=${board[$i]};
  }
  #vMirror_bitmap "$size";
  vMirror_bitmap;
  #//垂直反転
  #k=$(intncmp "${board}" "${trial}" "$size");
  k=$(intncmp_bt);
  ((k>0))&&{ 
#   echo "7:$k";
   return; 
  }
  ((nEquiv>2))&&{
  #               //-90度回転 対角鏡と同等       
    #rotate_bitmap_ts "$size";
    rotate_bitmap_ts;
    #k=$(intncmp "${board}" "${scratch}" "$size");
    k=$(intncmp_bs);
#    echo "8:$k";
    ((k>0))&&{
#      echo "9:$k";
      return;
    }
    ((nEquiv>4))&&{
#      echo "10:$k";
    #             //-180度回転 水平鏡像と同等
      #rotate_bitmap_st "$size";
      rotate_bitmap_st;
      #k=$(intncmp "${board}" "${trial}" "$size");
      k=$(intncmp_bt);
      ((k>0))&&{ 
#        echo "11:$k";
        return;
      } 
      #      //-270度回転 反対角鏡と同等
      #rotate_bitmap_ts "$size";
      rotate_bitmap_ts;
      #k=$(intncmp "${board}" "${scratch}" "$size");
      k=$(intncmp_bs);
      ((k>0))&&{ 
#        echo "12:$k";
        return;
      }
    }
  }
#  echo "13:$k";
  ((nEquiv==2))&&{
    ((COUNT2++));
  }
  ((nEquiv==4))&&{
    ((COUNT4++));
  }
  ((nEquiv==8))&&{
    ((COUNT8++));
  }
#  if [ $nEquiv -eq 2 ];then
#   ((COUNT2++));
#  fi
#  if [ $nEquiv -eq 4 ];then
#   ((COUNT4++));
#  fi
#  if [ $nEquiv -eq 8 ];then
#   ((COUNT8++));
#  fi
}
#
function N-Queen7_rec(){
	#y: l:left d:down r:right b:bit bm:bitmap
  local -i min="$1";
	local -i left="$2";
	local -i down="$3";
	local -i right="$4";
	local -i bitmap=0;
#	local -i bit=;
  bitmap=$((MASK&~(left|down|right)));
  ((min==size&&!bitmap))&&{
    board[$min]=$bitmap;
    #symmetryOps_bm "$size";
    symmetryOps_bm;
}||{
    while ((bitmap)); do
      bit=$((-bitmap&bitmap)) ;
      board[$min]=$bit;
      bitmap=$((bitmap^bit)) ;
      N-Queen7_rec "$((min+1))" "$(((left|bit)<<1))" "$((down|bit))" "$(((right|bit)>>1))"  ;
    done
  }
}
N-Queen7(){
  local -i max=15;
	local -i min=2;
	local startTime=;
	local endTime= ;
	local hh=mm=ss=0; 		# いっぺんにに初期化することもできます
  echo " N:        Total       Unique        hh:mm:ss" ;
  for ((size=min;size<=max;size++)) {
    TOTAL=0;
		UNIQUE=0;
    COUNT2=COUNT4=COUNT8=0;
    for((j=0;j<$size;j++)){
     board[$j]=$j; 
    }
		MASK=$(((1<<size)-1));
		startTime=`date +%s` ;
    N-Queen7_rec 0 0 0 0 ;
    endTime=$((`date +%s` - st)) ;
		ss=`expr ${endTime} - ${startTime}`; # hh:mm:ss 形式に変換
		hh=`expr ${ss} / 3600`;
		ss=`expr ${ss} % 3600`;
		mm=`expr ${ss} / 60`;
		ss=`expr ${ss} % 60`;
    TOTAL=$(getTotal);
    UNIQUE=$(getUnique);
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $size $TOTAL $UNIQUE $hh $mm $ss ;
  } 
}

# 実行はコメントアウトを外して、 $ ./BASH_N-Queen.sh 
   echo "N-Queen7 : バックトラック＋ビットマップ＋対称解除法";
   N-Queen7;
#
#
#
# ---------------------------------------------------------------------------------
##
# ８．バックトラック＋ビットマップ＋対称解除法＋枝刈りと最適化
#
#
# 実行はコメントアウトを外して、 $ ./BASH_N-Queen.sh 
  # echo "N-Queen8 : バックトラック＋ビットマップ＋対称解除法＋枝刈りと最適化";
  # N-Queen8;
#
#
# ---------------------------------------------------------------------------------
##
# ９．バックトラック＋ビットマップ＋対称解除法＋枝刈りと最適化＋クイーンの位置による振り分け
#
#
# 最上段の行のクイーンの位置は中央を除く右側の領域に限定されます。(ただし、N ≧ 2)
# 
#   次にその中でも一番右端(右上の角)にクイーンがある場合を考えてみます。他の３つ
# の角にクイーンを置くことはできないので(効き筋だから）、ユニーク解であるかどうか
# を判定するには、右上角から左下角を通る斜軸で反転させたパターンとの比較だけになり
# ます。突き詰めれば、
# 
# [上から２行目のクイーンの位置が右から何番目にあるか]
# [右から２列目のクイーンの位置が上から何番目にあるか]
# 
#
# を比較するだけで判定することができます。この２つの値が同じになることはないからです。
# 
#       3 0
#       ↓↓
# - - - - Q ←0
# - Q - - - ←3
# - - - - -         上から２行目のクイーンの位置が右から４番目にある。
# - - - Q -         右から２列目のクイーンの位置が上から４番目にある。
# - - - - -         しかし、互いの効き筋にあたるのでこれは有り得ない。
# 
#   結局、再帰探索中において下図の X への配置を禁止する枝刈りを入れておけば、得
# られる解は総てユニーク解であることが保証されます。
# 
# - - - - X Q
# - Q - - X -
# - - - - X -
# - - - - X -
# - - - - - -
# - - - - - -
# 
#   次に右端以外にクイーンがある場合を考えてみます。オリジナルがユニーク解である
# ためには先ず下図の X への配置は禁止されます。よって、その枝刈りを先ず入れておき
# ます。
# 
# X X - - - Q X X
# X - - - - - - X
# - - - - - - - -
# - - - - - - - -
# - - - - - - - -
# - - - - - - - -
# X - - - - - - X
# X X - - - - X X
# 
#   次にクイーンの利き筋を辿っていくと、結局、オリジナルがユニーク解ではない可能
# 性があるのは、下図の A,B,C の位置のどこかにクイーンがある場合に限られます。従っ
# て、90度回転、180度回転、270度回転の３通りの変換パターンだけを調べれはよいこと
# になります。
# 
# X X x x x Q X X
# X - - - x x x X
# C - - x - x - x
# - - x - - x - -
# - x - - - x - -
# x - - - - x - A
# X - - - - x - X
# X X B - - x X X
#
#
# 実行結果
#
#
#
##
# グローバル変数
typeset -i TOTAL=0;
typeset -i UNIQUE=0;
typeset -i COUNT2=COUNT4=COUNT8=0;
typeset -i N=;
typeset -i sizeE=; 			# sizeE = ((N-1))
typeset -i MASK=SIDEMASK=LASTMASK=0;
typeset -i BIT=TOPBIT=ENDBIT=0;
typeset -i BOUNT1=BOUND2=0;
typeset -a aBoard;
#
function symmetryOps(){
	((aBoard[BOUND2]==1))&&{
		for((p=2,o=1;o<=sizeE;o++,p<<=1)){
			for((BIT=1,y=sizeE;(aBoard[y]!=p)&&(aBoard[o]>=BIT);y--)){
				((BIT<<=1));
			}
			((aBoard[o]>BIT))&& return ;
			((aBoard[o]<BIT))&& break ;
		}
		#90度回転して同型なら180度回転も270度回転も同型である
		((o>sizeE))&&{ 
			((COUNT2++));
			return;
		}
	}
	#180度回転
	((aBoard[sizeE]==ENDBIT))&&{ 
		for ((y=sizeE-1,o=1;o<=sizeE;o++,y--)){
			for ((BIT=1,p=TOPBIT;(p!=aBoard[y])&&(aBoard[o]>=BIT);p>>=1)){
					((BIT<<=1)) ;
			}
			((aBoard[o]>BIT))&& return ;
			((aBoard[o]<BIT))&& break ;
		}
		#90度回転が同型でなくても180度回転が同型であることもある
		((o>sizeE))&&{ 
			((COUNT4++));
			return;
		}
	}
	#270度回転
	((aBoard[BOUND1]==TOPBIT))&&{ 
		for((p=TOPBIT>>1,o=1;o<=sizeE;o++,p>>=1)){
			for((BIT=1,y=0;(aBoard[y]!=p)&&(aBoard[o]>=BIT);y++)){
					((BIT<<=1)) ;
			}
			((aBoard[o]>BIT))&& return ;
			((aBoard[o]<BIT))&& break ;
		}
	}
	((COUNT8++));
}
#
# 最上段行のクイーンが角以外にある場合の探索 */
function Backtrack2(){
	local v=$1;		# v:virtical l:left d:down r:right
	local l=$2;
	local d=$3;
	local r=$4; 
	local bitmap=$((MASK & ~(l|d|r)));
	((v==sizeE))&&{ 
		((bitmap))&&{
			((!(bitmap&LASTMASK)))&&{
					aBoard[v]=$bitmap;
					symmetryOps ;
			}
		}
	}||{
		((v<BOUND1))&&{  #上部サイド枝刈り
			((bitmap|=SIDEMASK));
			((bitmap^=SIDEMASK));
		} 
		((v==BOUND2))&&{ #下部サイド枝刈り
				((!(d&SIDEMASK)))&& return ;
				(((d&SIDEMASK)!=SIDEMASK))&&((bitmap&=SIDEMASK));
		}
		while((bitmap));do
			((bitmap^=aBoard[v]=BIT=-bitmap&bitmap)); 
			Backtrack2 $((v+1)) $(((l|BIT)<<1)) $(((d|BIT)))  $(((r|BIT)>>1)) ;
		done
	}
}
#
# 最上段行のクイーンが角にある場合の探索
function Backtrack1(){
	local y=$1;		#y: l:left d:down r:right bm:bitmap
	local l=$2;
	local d=$3;
	local r=$4; 
	local bitmap=$((MASK & ~(l|d|r)));
	((y==sizeE))&&{
		 ((bitmap))&&{
			 	aBoard[y]=$bm;
				((COUNT8++)) ;
		 }
	}||{
		 ((y<BOUND1))&&{
			 	((bitmap|=2));
			 	((bitmap^=2));
		 }
		 while((bitmap));do
			((bitmap^=aBoard[y]=BIT=(-bitmap&bitmap))) ;
			Backtrack1 $((y+1)) $(((l|BIT)<<1))  $((d|BIT)) $(((r|BIT)>>1)) ;
		 done
	}
}
function func_BOUND1(){
	(($1<sizeE))&&{
		((aBoard[1]=BIT=1<<BOUND1));
		Backtrack1 2 $(((2|BIT)<<1)) $((1|BIT)) $((BIT>>1));
	}
}
function func_BOUND2(){
	(($1<$2))&&{
		((aBoard[0]=BIT=1<<BOUND1));
		Backtrack2 1 $((BIT<<1)) $BIT $((BIT>>1)) ;
	}
}
#
function N-QueenLogic_Q9(){
	aBoard[0]=1;
	((sizeE=(N-1))); 	
	((MASK=(1<<N)-1));
	((TOPBIT=1<<sizeE));
	BOUND1=2;
	while((BOUND1>1&&BOUND1<sizeE));do
		func_BOUND1 BOUND1;
		((BOUND1++));
	done
	((SIDEMASK=LASTMASK=(TOPBIT|1)));
	((ENDBIT=TOPBIT>>1));
	BOUND1=1;
	((BOUND2=N-2));
	while((BOUND1>0&&BOUND2<sizeE&&BOUND1<BOUND2));do
		func_BOUND2 BOUND1 BOUND2;
		((BOUND1++,BOUND2--));
		((ENDBIT>>=1));
		((LASTMASK|=LASTMASK>>1|LASTMASK<<1)) ;
	done
	((UNIQUE=COUNT8+COUNT4+COUNT2)) ;
	((TOTAL=COUNT8*8+COUNT4*4+COUNT2*2));
}
#
N-Queen9(){
  local -i max=17;
	local -i min=2;
	local startTime=0;
	local endTime=0;
	local hh=mm=ss=0; 		# いっぺんにに初期化することもできます
  echo " N:        Total       Unique        hh:mm:ss" ;
  for ((N=min;N<=max;N++));do
		COUNT2=COUNT4=COUNT8=0;
		startTime=`date +%s` ;
		N-QueenLogic_Q7 ;
		endTime=`date +%s`;					# 計測終了時間
		ss=`expr ${endTime} - ${startTime}` # hh:mm:ss 形式に変換
		hh=`expr ${ss} / 3600`
		ss=`expr ${ss} % 3600`
		mm=`expr ${ss} / 60`
		ss=`expr ${ss} % 60`
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $N $TOTAL $UNIQUE $hh $mm $ss ;
  done
}
#
# 実行はコメントアウトを外して、 $ ./bash_n-queen.sh 
	# echo "N-Queen9 : バックトラック＋ビットマップ＋対称解除法＋枝刈り＋最適化クイーンの位置による振り分け";
	# N-Queen9  ;
#
#
#
exit ;


