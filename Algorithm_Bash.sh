#!/bin/bash

##
 # Bash(シェルスクリプト)で学ぶアルゴリズムとデータ構造  
 # ステップバイステップでＮ−クイーン問題を最適化
 # 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
 # 
 # 目次
 # 1. ソートアルゴリズム
 #   バブルソート
 #   選択ソート
 #   挿入ソート
 #   マージソート
 #   シェルソート
 #   クイックソート
 # 
 # 2. 再帰
 #   三角数
 #   階乗
 #   ユークリッドの互除法
 #   ハノイの塔
 # 
 # 3.  Nクイーン問題
 #  １．ブルートフォース（力まかせ探索） NQueen1() * N 8: 00:04:15
 #  ２．バックトラック                   NQueen2() * N 8: 00:00:01
 #  ３．配置フラグ（制約テスト高速化）   NQueen3() * N16: 00:01:35
 #  ４．対称解除法(回転と斜軸）          NQueen4() * N16: 00:01:50
 #  ５．枝刈りと最適化                   NQueen5() * N16: 00:00:24
 #  ６．マルチスレッド1                  NQueen6() * N16: 00:00:05
 #  ７．ビットマップ                     NQueen7() * N16: 00:00:02
 #  ８．マルチスレッド2                  NQueen9() * N16: 00:00:00
# #

##
# 共通部分
function display(){
  for((i=0;i<nElems;i++)){
    echo "$i" "${array[i]}";
  }
  echo "-----";
}
function insert(){
  array[nElems++]="$1";
}
function setArray(){
  nElems=0;
  for((i=0;i<$1;i++)){
    insert $(echo "$RANDOM");
  }
}

##
 # 1. バブルソート 13404mm
 # https://ja.wikipedia.org/wiki/バブルソート
 # https://www.youtube.com/watch?v=8Kp-8OGwphY
 #   平均計算時間が O(N^2)
 #   安定ソート
 #   比較回数は「  n(n-1)/2  」
 #   交換回数は「  n^2/2  」
 #   派生系としてシェーカーソートやコムソート
##
function bubbleSort(){
  local i j t ; # t:temp
  for((i=nElems;i>0;i--)){
    for((j=0;j<i-1;j++)){
      ((array[j]>array[j+1]))&&{
        t="${array[j]}" ;
        array[j]="${array[j+1]}" ;
        array[j+1]="$t" ;
      }
    }
  }
}

##
 # 選択ソート 3294mm
 # https://ja.wikipedia.org/wiki/選択ソート
 # https://www.youtube.com/watch?v=f8hXR_Hvybo
 #   平均計算時間が O(N^2)
 #   安定ソートではない
 #   比較回数は「  n(n-1)/2  」
 #   交換回数は「  n-1  」
##
function selectionSort(){
  local i j t m ; # t:temp m:min
  for((i=0;i<nElems;i++)){
    m="$i" ;
    for((j=i+1;j<nElems;j++)){
      ((array[m]>array[j]))&& m="$j"; 
    }
    ((m==i))&& continue;
    t="${array[m]}" ;
    array[m]="${array[i]}" ;
    array[i]="$t" ;
  }
}

##
 # 挿入ソート 3511mm
 # https://ja.wikipedia.org/wiki/挿入ソート
 # https://www.youtube.com/watch?v=DFG-XuyPYUQ
 #   平均計算時間が O(N^2)
 #   安定ソート
 #   比較回数は「  n(n-1)/2以下  」
 #   交換回数は「  約n^2/2以下  」
## 
function insertionSort(){
  local o i t ; # o:out i:in t:temp
  for((o=1;o<nElems;o++)){
    t="${array[o]}" ;
    for((i=o;i>0&&array[i-1]>t;i--)){
      array[i]="${array[i-1]}" ;
    }
    array[i]="$t" ;
  }
}

##
 # マージソート 1085mm
 # https://ja.wikipedia.org/wiki/マージソート
 # https://www.youtube.com/watch?v=EeQ8pwjQxTM
 #   平均計算時間が O(N(Log N))
 #   安定ソート
 #   50以下は挿入ソート、5万以下はマージソート、あとはクイックソートがおすすめ。
 #   バブルソート、挿入ソート、選択ソートがO(N^2)の時間を要するのに対し、マージ
 #   ソートはO(N*logN)です。
 #   例えば、N(ソートする項目の数）が10,000ですと、N^2は100,000,000ですが、
 #   n*logNは40,000です。別の言い方をすると、マージソートで４０秒を要するソート
 #   は、挿入ソートでは約２８時間かかります。
 #   マージソートの欠点は、ソートする配列と同サイズの配列をもう一つ必要とする事
 #   です。
 #   元の配列がかろうじてメモリに治まるという大きさだったら、マージソートは使え
 #   ません。
##
function mergeSortLogic(){
  local f=$1 m=$2 l=$3 ; # f:first m:mid l:last w:workArray
  local n i j n1 ;
  ((n=l-f+1)) ;
  for((i=f,j=0;i<=l;)){
    w[j++]="${array[i++]}" ;
  }
  ((m>l))&&((m=(f+l)/2)) ;
  ((n1=m-f+1)) ;
  for((i=f,j=0,k=n1;i<=l;i++)){
    {
      ((j<n1))&&{
        ((k==n))||{ 
          ((${w[j]}<${w[k]}))
        }
      }
    }&&{ 
      array[i]="${w[j++]}" ;
    }||{
      array[i]="${w[k++]}" ;
    }
  }
}
function mergeSort(){
    local f="$1" l="$2" m= ; # f:first l:last m:mid
    ((l>f))||return 0;
    m=$(((f+l)/2));
    mergeSort "$f" "$m" ;
    mergeSort "$((m+1))" "$l"
    mergeSortLogic "$f" "$m" "$l" ;
}

##
 # シェルソート 1052mm
 # https://ja.wikipedia.org/wiki/シェルソート
 # https://www.youtube.com/watch?v=M9YCh-ZeC7Y
 #   平均計算時間が O(N((log N)/(log log N))^2)
 #   安定ソートではない
 #   挿入ソート改造版
 #   ３倍して１を足すという処理を要素を越えるまで行う
##
function shellSort(){
  local s=1 in t ; #s:shell in:inner t:temp
  while((s<nElems/3)); do
      s=$((s*3+1)) ;
  done
  while((s>0)); do
    for((i=s;i<nElems;i++)){
      t="${array[i]}" ;
      in="$i" ;
      while((in>s-1&&array[in-s]>=t)); do
        array[in]="${array[in-s]}" ;
        in=$((in-s)) ;
      done
      array[in]="$t" ;
    }
    s=$(((s-1)/3)) ;
  done
}

##
 # クイックソート 1131mm
 # https://ja.wikipedia.org/wiki/クイックソート
 # https://www.youtube.com/watch?v=aQiWF4E8flQ
 #   平均計算時間が O(n Log n)
 #   安定ソートではない
 #   最大計算時間が O(n^2)
 # データ数が 50 以下なら挿入ソート (Insertion Sort)
 # データ数が 5 万以下ならマージソート (Merge Sort)
 # データ数がそれより多いならクイックソート (Quick Sort)
##
function quickSort() {
  local -i l r m p t i j k; #r:right l:left m:middle p:part t:temp 
  ((l=i=$1,r=j=$2,m=(l+r)/2));
  p="${array[m]}" ;
  while((j>i)); do
    while [[ 1 ]]; do
      ((array[i]<p))&&((i++))||break ;
    done
    while [[ 1 ]]; do
      ((array[j]>p))&&((j--))||break ;
    done
    ((i<=j))&&{
      t="${array[i]}";
      array[i]="${array[j]}";
      array[j]="$t" ;
      ((i++,j--)) ;
    }
  done
  ((l<j)) && quickSort $l $j ;
  ((r>i)) && quickSort $i $r ;
}
##
# 実行メソッド
##
function SortCase(){
  setArray $1 ;
#  display ;
  case "$2" in
    bubbleSort) 
      echo "bubbleSort" ;
      bubbleSort ;;
    selectionSort) 
      echo "selectionSort" ;
      selectionSort ;;
    insertionSort) 
      echo "insertionSort" ;
      insertionSort ;;
    mergeSort) 
      echo "mergeSort" ;
      mergeSort 0 $((nElems-1));;
    shellSort) 
      echo "shellSort" ;
      shellSort ;;
    quickSort) 
      echo "quickSort" ;
      quickSort 0 $((nElems-1)) ;;
  esac
#  display ;
}
#
function Sort(){
  time SortCase 1000 "bubbleSort";
  time SortCase 1000 "selectionSort";
  time SortCase 1000 "insertionSort";
  time SortCase 1000 "mergeSort";
  time SortCase 1000 "shellSort" ;
  time SortCase 1000 "quickSort" ;
}

##
# 実行は以下のコメントを外して実行
#Sort ;
#exit ;
#
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

##
 #   ステップ
 # 
 #  １．ブルートフォース（力まかせ探索） NQueen1()
 #  ２．バックトラック                   NQueen2()
 #  ３．配置フラグ（制約テスト高速化）   NQueen3()
 #  ４．ビットマップ                     NQueen4()
 #  ５．対称解除法(回転と斜軸）          NQueen5()
 #  ６．マルチスレッド                   NQueen6()
##
##

## 1. ブルートフォース　力任せ探索
 #　全ての可能性のある解の候補を体系的に数え上げ、それぞれの解候補が問題の解と
 #  なるかをチェックする方法
 #  (※)各行に１個の王妃を配置する組み合わせを再帰的に列挙組み合わせを生成するだ
 #  けであって8王妃問題を解いているわけではありません
#  :
#  :
#  7 7 7 7 7 7 6 7 : 16777208
#  7 7 7 7 7 7 7 0 : 16777209
#  7 7 7 7 7 7 7 1 : 16777210
#  7 7 7 7 7 7 7 2 : 16777211
#  7 7 7 7 7 7 7 3 : 16777212
#  7 7 7 7 7 7 7 4 : 16777213
#  7 7 7 7 7 7 7 5 : 16777214
#  7 7 7 7 7 7 7 6 : 16777215
#  7 7 7 7 7 7 7 7 : 16777216

c=1 ; # c:count
N-Queen1(){
  local -i i="$1" j= s="$2" ; # s:size
  for((j=0;j<s;j++)) {
      pos[i]="$j" ;
      ((i==s-1))&&{ 
        echo -n "$((c++)): " ;
        for((x=0;x<s;x++)){
          echo -n "${pos[x]}" ;
        }
        echo "" ;
      }||N-Queen1 "$((i+1))" "$s" ;
  }  
}

##
 # ２．配置フラグ（制約テスト高速化）
 #  パターンを生成し終わってからチェックを行うのではなく、途中で制約を満たさな
 #  い事が明らかな場合は、それ以降のパターン生成を行わない。
 # 「手を進められるだけ進めて、それ以上は無理（それ以上進めても解はない）という
 # 事がわかると一手だけ戻ってやり直す」という考え方で全ての手を調べる方法。
 # (※)各行列に一個の王妃配置する組み合わせを再帰的に列挙分枝走査を行っても、組
 # み合わせを列挙するだけであって、8王妃問題を解いているわけではありません。
 #
#  :
#  :
#  7 6 5 4 2 0 3 1 : 40310
#  7 6 5 4 2 1 0 3 : 40311
#  7 6 5 4 2 1 3 0 : 40312
#  7 6 5 4 2 3 0 1 : 40313
#  7 6 5 4 2 3 1 0 : 40314
#  7 6 5 4 3 0 1 2 : 40315
#  7 6 5 4 3 0 2 1 : 40316
#  7 6 5 4 3 1 0 2 : 40317
#  7 6 5 4 3 1 2 0 : 40318
#  7 6 5 4 3 2 0 1 : 40319
#  7 6 5 4 3 2 1 0 : 40320

c=1 ;
N-Queen2(){
  local -i i=$1 j= size=$2 ;
  for ((j=0;j<size;j++)) {
    [[ -z "${flag_a[j]}" ]] && {
      pos[i]="$j" ; 
      ((i==(size-1)))&&{
        echo -n "$((c++)): " ;
        for((i=0;i<size;i++)){
          echo -n "${pos[i]}" ;
        }
        echo "" ;
      }||{
        flag_a[j]="true" ;         
        N-Queen2 "$((i+1))" "$size" ;
        flag_a[j]="" ; 
      }
    }
  }
}

##
 # ３．バックトラック
 #  　各列、対角線上にクイーンがあるかどうかのフラグを用意し、途中で制約を満た
 #  さない事が明らかな場合は、それ以降のパターン生成を行わない。
 #  　各列、対角線上にクイーンがあるかどうかのフラグを用意することで高速化を図る。
 #  　これまでは行方向と列方向に重複しない組み合わせを列挙するものですが、王妃
 #  は斜め方向のコマをとることができるので、どの斜めライン上にも王妃をひとつだ
 #  けしか配置できない制限を加える事により、深さ優先探索で全ての葉を訪問せず木
 #  を降りても解がないと判明した時点で木を引き返すということができます。
##
# N:        Total       Unique        hh:mm
# 2:            0            0            0
# 3:            0            0            0
# 4:            2            0            0
# 5:           10            0            0
# 6:            4            0            0
# 7:           40            0            0
# 8:           92            0            1
# 9:          352            0            1
#10:          724            0            7
#11:         2680            0           33
#12:        14200            0          183
##

T=1 ;
N-Queen3_rec(){
  local -i i="$1" j s=$2 ; # s:size
  for((j=0;j<s;j++)){
    [[ -z "${fa[j]}" ]] && [[ -z "${fb[i+j]}" ]] && [[ -z "${fc[i-j+s-1]}" ]] && {
      pos[$i]=$j ;
      ((i==(s-1)))&&((T++))||{
        fa[j]="true" ;
        fb[i+j]="true" ; 
        fc[i-j+s-1]="true" ; 
        N-Queen3_rec "$((i+1))" "$s" ; 
        fa[j]="" ;           
        fb[i+j]="" ;   
        fc[i-j+s-1]="" ; 
      }          
    }
  }  
}
N-Queen3(){
  # m: max mi:min s:size st:starttime t:time
  local -i m=12 mi=2 s=$mi st= t= ;
  echo " N:        Total       Unique        hh:mm" ;
  for((s=mi;s<=m;s++)){
    T=0 U=0 st=`date +%s` ;
    N-Queen3_rec 0 "$s";
    t=$((`date +%s` - st)) ;
    printf "%2d:%13d%13d%13d\n" $s $T $U $t ;
  } 
}

##
# ビットマップ
#
# N:        Total       Unique        hh:mm
# 2:            0            0            0
# 3:            0            0            0
# 4:            2            0            0
# 5:           10            0            0
# 6:            4            0            0
# 7:           40            0            0
# 8:           92            0            0
# 9:          352            0            1
#10:          724            0            3
#11:         2680            0           14
#12:        14200            0           71
#13:        73712            0          392

T=0; # T:total
U=0  # U:unique
S=0; # S:size
M=;  # M:mask
N-Queen4_rec(){
	#y: l:left d:down r:right b:bit bm:bitmap
  local y="$1" l="$2" d="$3" r="$4" bm= b=;
  ((y==S))&&((T++))||{
    bm=$((M&~(l|d|r)));
    while ((bm)); do
      b=$((-bm&bm)) ;
      bm=$((bm^b)) ;
      N-Queen4_rec "$((y+1))" "$(((l|b)<<1))" "$((d|b))" "$(((r|b)>>1))"  ;
    done
  }
}
N-Queen4(){
  local ma=13 mi=2 st= t= ; # ma:maxsize mi:minsize st:starttime t:time
  echo " N:        Total       Unique        hh:mm" ;
  for ((S=mi;S<=ma;S++)) {
    T=0 U=0 M=$(((1<<S)-1)) st=`date +%s` ;
    N-Queen4_rec 0 0 0 0 ;
    t=$((`date +%s` - st)) ;
    printf "%2d:%13d%13d%13d\n" $S $T $U $t ;
  } 
}

##
# 5. ビット演算に加えてユニーク解(回転・反転）を使って高速化
# ユニーク解の判定とユニーク解の種類の判定 
##
#
# N:        Total       Unique        hh:mm
# 2:            0            0            0
# 3:            0            0            0
# 4:            2            1            0
# 5:           10            2            0
# 6:            4            1            0
# 7:           40            6            0
# 8:           92           12            0
# 9:          352           46            1
#10:          724           92            0
#11:         2680          341            3
#12:        14200         1787           18
#13:        73712         9233           99
#14:       365596        45752          573
#15:      2279184       285053         3511
#
##
#---------------------ここから共通部分 ------------------------
# グローバル
U= T= ;				# U:unique T:total
C2= C4= C8= ; # C2:count2 C4:count4 C8:count8
S= SE=; 			# S:size SE:sizee(size-1)
M= SM= LM= ;  # M:mask SM:sidemask LM:lastmask
B= TB= EB=;  	# B:bit TB:topbit EB:endbit
B1= B2=; 			# B1:bound1 B2:bound2
function Check_Qset(){
	((aB[B2]==1))&&{
		for ((p=2,o=1;o<=SE;o++,p<<=1)){
			for ((B=1,y=SE;(aB[y]!=p)&&(aB[o]>=B);y--)){
				 ((B<<=1)) ;
			}
			((aB[o]>B))&& return ;
			((aB[o]<B))&& break ;
		}
		((o>SE))&&{ #90度回転して同型なら180度回転も270度回転も同型である
			((C2++));
			return;
		}
	}
	((aB[SE]==EB))&&{ #180度回転
		for ((y=SE-1,o=1;o<=SE;o++,y--)){
			for ((B=1,p=TB;(p!=aB[y])&&(aB[o]>=B);p>>=1)){
					((B<<=1)) ;
			}
			((aB[o]>B))&& return ;
			((aB[o]<B))&& break ;
		}
		((o>SE))&&{ #90度回転が同型でなくても180度回転が同型であることもある
			((C4++));
			return;
		}
	}
	((aB[B1]==TB))&&{ #270度回転
		for((p=TB>>1,o=1;o<=SE;o++,p>>=1)){
			for((B=1,y=0;(aB[y]!=p)&&(aB[o]>=B);y++)){
					((B<<=1)) ;
			}
			((aB[o]>B))&& return ;
			((aB[o]<B))&& break ;
		}
	}
	((C8++));
}
# 最上段行のクイーンが角以外にある場合の探索 */
function Backtrack2(){
	local v=$1 l=$2 d=$3 r=$4; # v:virtical l:left d:down r:right
	local bm=$((M & ~(l|d|r)));
	((v==SE))&&{ 
		((bm))&&{
			((!(bm&LM)))&&{
					aB[v]=$bm;
					Check_Qset ;
			}
		}
	}||{
		((v<B1))&&{  #上部サイド枝刈り
			((bm|=SM));
			((bm^=SM));
		} 
	 ((v==B2))&&{ #下部サイド枝刈り
			((!(d&SM)))&& return ;
			(((d&SM)!=SM))&&((bm&=SM));
		}
		while ((bm)); do
			((bm^=aB[v]=B=-bm&bm)); 
			Backtrack2 $((v+1)) $(((l|B)<<1)) $(((d|B)))  $(((r|B)>>1)) ;
		done
	}
}
# 最上段行のクイーンが角にある場合の探索
function Backtrack1(){
	local y=$1 l=$2 d=$3 r=$4; #y: l:left d:down r:right bm:bitmap
	local bm=$((M & ~(l|d|r)));
	((y==SE))&&{
		 ((bm))&&{
			 	aB[y]=$bm;
				((C8++)) ;
		 }
	}||{
		 ((y<B1))&&{
			 	((bm|=2));
			 	((bm^=2));
		 }
		 while ((bm)) ;do
			(( bm^=aB[y]=B=(-bm&bm) )) ;
			Backtrack1 $((y+1)) $(((l|B)<<1))  $((d|B)) $(((r|B)>>1)) ;
		 done
	}
}
function BOUND1(){
	(($1<SE))&&{
		(( aB[1]=B=1<<B1 ));
		Backtrack1 2 $(((2|B)<<1)) $((1|B)) $((B>>1));
	}
}
function BOUND2(){
	(($1<$2))&&{
		(( aB[0]=B=1<<B1 ));
		Backtrack2 1 $((B<<1)) $B $((B>>1)) ;
	}
}
#---------------------ここまで共通部分 ------------------------
function BOUND1_Q5(){
	(($1<SE))&&{
		(( aB[1]=B=1<<B1 ));
		Backtrack1 2 $(((2|B)<<1)) $((1|B)) $((B>>1));
	}
}
function BOUND2_Q5(){
	(( $1<$2 ))&&{
		(( aB[0]=B=1<<B1 ));
		Backtrack2 1 $((B<<1)) $B $((B>>1)) ;
	}
}
function N-QueenLogic_Q5(){
	aB[0]=1; 					# aB:BOARD[]
	((SE=(S-1))); 	# SE:sizee
	((M=(1<<S)-1)); # m:mask
	((TB=1<<SE)); 	# TB:topbit
	B1=2;
	while((B1>1&&B1<SE));do
		BOUND1 B1;
		((B1++));
	done
	((SM=LM=(TB|1)));
	((EB=TB>>1));
	B1=1;
	B2=S-2;
	while((B1>0&&B2<B2<SE&&B1<B2));do
		BOUND2 B1 B2;
		((B1++,B2--));
		((EB>>=1));
		((LM|=LM>>1|LM<<1)) ;
	done
	((U=C8+C4+C2)) ;
	((T=C8*8+C4*4+C2*2));
}
N-Queen5(){
	# m:max mi:min st:starttime t:time s:S
  local m=15 mi=2 st= t= s=; 
  echo " N:        Total       Unique        hh:mm" ;
  for ((S=mi;S<=m;S++));do
    C2=0; C4=0; C8=0 st=`date +%s` ;
    N-QueenLogic_Q5 ;
    t=$((`date +%s` - st)) ;
    printf "%2d:%13d%13d%13d\n" $S $T $U $t ;
  done
}

##   6．マルチスレッド
 # 
 # ここまでの処理は、一つのスレッドが順番にＡ行の１列目から順を追って処理判定をし
 # てきました。この節では、Ａ行の列それぞれに別々のスレッドを割り当て、全てのス
 # レッドを同時に処理判定させます。Ａ行それぞれの列の処理判定結果はBoardクラスで
 # 管理し、処理完了とともに結果を出力します。スレッドはWorkEngineクラスがNの数だ
 # け生成されます。WorkEngineクラスは自身の持ち場のＡ行＊列の処理だけを担当しま
 # す。これらはマルチスレッド処理と言い、並列処理のための同期、排他、ロック、集計
 # など複雑な処理を理解する知識が必要です。そして処理の最後に合計値を算出する方法
 # をマルチスレッド処理と言います。
 # １Ｘ１，２Ｘ２，３Ｘ３，４Ｘ４，５Ｘ５，６Ｘ６，７ｘ７、８Ｘ８のボートごとの計
 # 算をスレッドに割り当てる手法がちまたでは多く見受けられます。これらの手法は、
 # 実装は簡単ですが、Ｎが７の計算をしながら別スレッドでＮが８の計算を並列処理する
 # といった矛盾が原因で、Ｎが大きくなるとむしろ処理時間がかかります。
 #   ここでは理想的なアルゴリズムとして前者の手法でプログラミングします。
##
 # 共通メソッドは 5.ビット演算に加えてユニーク解(回転・反転）を使って高速化 を参照
##

function N-QueenLogic_Q6(){
	aB[0]=1; 				# aB:BOARD[]
	((SE=(S-1))); 	# SE:sizee
	((M=(1<<S)-1)); # m:mask
	((TB=1<<SE)); 	# TB:topbit
#	B1=2;
#	while((B1>1&&B1<SE));do
#		BOUND1_Q6 B1;
#		((B1++));
#	done
	((B1>1&&B1<SE))&&{
		BOUND1 B1;
	}

	((SM=LM=(TB|1)));
	((EB=TB>>1));
#	B1=1;
#	B2=S-2;
#	while((B1>0&&B2<B2<SE&&B1<B2));do
#		BOUND2_Q6 B1 B2;
#		((B1++,B2--));
#		((EB>>=1));
#		((LM|=LM>>1|LM<<1)) ;
#	done
	((B1>0&&B2<SE&&B1<B2))&&{
		for((i=1;i<B1;i++)){
			((LM|=LM>>1|LM<<1)) ;
		}
		BOUND2 B1 B2 ;
		((EB>>=nT));
	}
	((U=C8+C4+C2)) ;
	((T=C8*8+C4*4+C2*2));
}
function N-Queen6_thread(){
	nT=$1 B1=$2 B2=$3; 
	while((nT>0)); do
		N-QueenLogic_Q6;
		((nT--));
		((B1--));
		((B2++));
	done
}
function N-Queen6(){
	# m:max mi:min st:starttime t:time s:S
  local m=15 mi=2 st= t= s=; 
  echo " N:        Total       Unique        hh:mm" ;
  for ((S=mi;S<=m;S++));do
    C2=0; C4=0; C8=0 st=`date +%s` ;
		N-Queen6_thread S $((S-1)) 0; # size B1 B2
    t=$((`date +%s` - st)) ;
    printf "%2d:%13d%13d%13d\n" $S $T $U $t ;
  done
}

function N-Queen(){
#  N-Queen1 0 8;      # ブルートフォース
#  N-Queen2 0 8;      # 配置フラグ
#  N-Queen3 		      # バックトラック
#  N-Queen4  ;				# ビットマップ
  N-Queen5 ; 				# ユニーク解
  # 作業中です。すみません
#  N-Queen6 ; 				# マルチスレッド
}

N-Queen ;
exit ;



