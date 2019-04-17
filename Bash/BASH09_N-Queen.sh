#!/bin/bash
#
#
# Bash（シェルスクリプト）で学ぶ「アルゴリズムとデータ構造」
# 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#
#
# ---------------------------------------------------------------------------------
##
# ９．バックトラック＋ビットマップ＋対称解除法＋枝刈りと最適化＋対称解除法のビットマップ化＋クイーンの位置による振り分け（BOUND1)
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
# 実行結果
# <>９．BT＋Bit＋対称解除Bit＋クイーンの位置による振り分け(BOUND1) N-Queen9()
#  N:        Total       Unique        hh:mm:ss
#  2:            0            0         0:00:00
#  3:            0            0         0:00:00
#  4:            0            0         0:00:00
#  5:           10            2         0:00:00
#  6:            4            1         0:00:00
#  7:           40            6         0:00:00
#  8:           92           12         0:00:01
#  9:          352           46         0:00:03
# 10:          724           92         0:00:09
# 11:         2680          341         0:00:38
# 12:        14200         1787         0:03:19
#
typeset -i TOTAL=0;
typeset -i UNIQUE=0;
typeset -i size=0;
typeset -i MASK=0;
typeset -i BOUND1=0;
typeset -i BOUND2=0;
typeset -i TOPBIT=0;
typeset -i ENDBIT=0;
typeset -i SIDEMASK=0;
typeset -i LASTMASK=0;
typeset -a board="";
typeset -a trial="";
typeset -a scratch="";
typeset -a flag_a="";     # -a は配列の型を宣言します
typeset -a flag_b="";
typeset -a flag_c="";
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
     #let tmp="tmp|=(1<<(sz-i))"; 
     ((tmp|=(1<<(sz-i)))); 
    }
  }
  echo $tmp;
}
#
function vMirror_bitmap(){
  local -i score=0;
  local -i sizeE=$((size-1));
  for((i=0;i<size;i++)){
    score=${scratch[$i]};
    trial[$i]=$(rh "$score" $sizeE);
  }
}
function intncmp_bs(){
  local -i rtn=0;
  for((i=0;i<size;i++)){
    #rtn=$(echo "${board[$i]}-${scratch[$i]}"|bc);
    rtn=$(echo "${board[$i]}-${scratch[$i]}"+10);
    #((rtn!=0))&&{ break; }
    ((rtn!=10))&&{ break; }
  }
  echo "$rtn";
}
function intncmp_bt(){
  local -i rtn=0;
  for((i=0;i<size;i++)){
    #rtn=$(echo "${board[$i]}-${trial[$i]}"|bc);
    rtn=$(echo "${board[$i]}-${trial[$i]}"+10);
    #((rtn!=0))&&{ break; }
    ((rtn!=10))&&{ break; }
  }
  echo "$rtn";
}
function symmetryOps_bm(){
#  local -i si=$1;
  local -i nEquiv=0;
  #回転・反転・対称チェックのためにboard配列をコピー
  for((i=0;i<size;i++)){ 
    trial[$i]=${board[$i]};
  }
  #rotate_bitmap_ts "$size";
  rotate_bitmap_ts; 
  #    //時計回りに90度回転
  k=$(intncmp_bs);
  #((k>0))&&{ 
  ((k>10))&&{ 
   return; 
  }
  #((k==0))&&{ 
  ((k==10))&&{ 
    nEquiv=2;
  }||{
    rotate_bitmap_st;
    #  //時計回りに180度回転
    k=$(intncmp_bt);
    #((k>0))&&{ 
    ((k>10))&&{ 
     return; 
    }
    #((k==0))&&{ 
    ((k==10))&&{ 
      nEquiv=4;
    }||{
      rotate_bitmap_ts;
      #//時計回りに270度回転
      k=$(intncmp_bs);
      #((k>0))&&{ 
      ((k>10))&&{ 
        return;
      }
      nEquiv=8;
    }
  }
  #// 回転・反転・対称チェックのためにboard配列をコピー
  for((i=0;i<size;i++)){ 
    scratch[$i]=${board[$i]};
  }
  vMirror_bitmap;
  #//垂直反転
  k=$(intncmp_bt);
  #((k>0))&&{ 
  ((k>10))&&{ 
   return; 
  }
  ((nEquiv>2))&&{
  #               //-90度回転 対角鏡と同等       
    rotate_bitmap_ts;
    k=$(intncmp_bs);
    #((k>0))&&{
    ((k>10))&&{
      return;
    }
    ((nEquiv>4))&&{
    #             //-180度回転 水平鏡像と同等
      #rotate_bitmap_st "$size";
      rotate_bitmap_st;
      k=$(intncmp_bt);
      #((k>0))&&{ 
      ((k>10))&&{ 
        return;
      } 
      #      //-270度回転 反対角鏡と同等
      rotate_bitmap_ts;
      k=$(intncmp_bs);
      #((k>0))&&{ 
      ((k>10))&&{ 
        return;
      }
    }
  }
  ((nEquiv==2))&&{
    ((COUNT2++));
  }
  ((nEquiv==4))&&{
    ((COUNT4++));
  }
  ((nEquiv==8))&&{
    ((COUNT8++));
  }
}
backTrack(){
	local -i bit;
  local -i min="$1";
	local -i left="$2";
	local -i down="$3";
	local -i right="$4";
	local -i bitmap=0;
  bitmap=$((MASK&~(left|down|right)));
	((min==size&&!bitmap))&&{
	  board[$min]=bitmap;
    symmetryOps_bm;
	}||{
		while((bitmap));do
      bit=$((-bitmap&bitmap)) ;
      board[$min]=$bit;
      bitmap=$((bitmap^bit)) ;
      backTrack "$((min+1))" "$(((left|bit)<<1))" "$((down|bit))" "$(((right|bit)>>1))"  ;
		done
	}
}
#
function N-Queen9_rec(){
 	local -i min="$1";
	((TOPBIT=1<<(size-1)));
  LASTMASK=$((TOPBIT|1));
	SIDEMASK=$LASTMASK;
	ENDBIT=$((TOPBIT>>1));
	BOUND2=$((size-2));
	for ((BOUND1=0;BOUND1<BOUND2;BOUND1++)){
	  bit=$((1<<BOUND1));
	  board[0]=$bit;
    backTrack "1" "$((bit<<1))" "$((bit))" "$((bit>>1))";
	  ((LASTMASK|=LASTMASK>>1|LASTMASK<<1));
	  ((ENDBIT>>=1));
	  ((BOUND2--));
	}
}
N-Queen9(){
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
    N-Queen9_rec 0;
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
#
#
#
# 実行はコメントアウトを外して、 $ ./BASH_N-Queen.sh 
  echo "<>９．BT＋Bit＋対称解除Bit＋クイーンの位置による振り分け(BOUND1) N-Queen9()";
  N-Queen9;
#

