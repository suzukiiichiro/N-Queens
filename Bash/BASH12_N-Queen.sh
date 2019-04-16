#!/bin/bash
#
#
# アルゴリズムとデータ構造  
# 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#
# １２．BT＋Bit＋対称解除Bit＋クイーンの位置による振り分け(BOUND1+BOUND2)＋枝刈り＋最適化
#
# 実行結果
# <>１２．BT＋Bit＋対称解除Bit＋クイーンの位置による振り分け(BOUND1+BOUND2)＋枝刈り＋最適化 N-Queen12()

#  N:        Total       Unique        hh:mm:ss
#  2:            0            0         0:00:00
#  3:            0            0         0:00:00
#  4:            2            1         0:00:00
#  5:           10            2         0:00:00
#  6:            4            1         0:00:00
#  7:           40            6         0:00:00
#  8:           92           12         0:00:00
#  9:          352           46         0:00:00
# 10:          724           92         0:00:01
# 11:         2680          341         0:00:05
# 12:        14200         1787         0:00:25
#
##
# グローバル変数
typeset -i TOTAL=0;
typeset -i UNIQUE=0;
typeset -i COUNT2=COUNT4=COUNT8=0;
typeset -i size=;
typeset -i sizeE=;
typeset -i MASK=SIDEMASK=LASTMASK=0;
typeset -i bit=TOPBIT=ENDBIT=0;
typeset -i BOUND1=BOUND2=0;
typeset -a board;
#
function symmetryOps(){
	((board[BOUND2]==1))&&{
		for((ptn=2,own=1;own<=sizeE;own++,ptn<<=1)){
			for((bit=1,you=sizeE;(board[you]!=ptn)&&(board[own]>=bit);you--)){
				((bit<<=1));
			}
			((board[own]>bit))&& return ;
			((board[own]<bit))&& break ;
		}
		#90度回転して同型なら180度回転も270度回転も同型である
		((own>sizeE))&&{ 
			((COUNT2++));
			return;
		}
	}
	#180度回転
	((board[sizeE]==ENDBIT))&&{ 
		for ((you=sizeE-1,own=1;own<=sizeE;own++,you--)){
			for ((bit=1,ptn=TOPBIT;(ptn!=board[you])&&(board[own]>=bit);ptn>>=1)){
					((bit<<=1)) ;
			}
			((board[own]>bit))&& return ;
			((board[own]<bit))&& break ;
		}
		#90度回転が同型でなくても180度回転が同型であることもある
		((own>sizeE))&&{ 
			((COUNT4++));
			return;
		}
	}
	#270度回転
	((board[BOUND1]==TOPBIT))&&{ 
		for((ptn=TOPBIT>>1,own=1;own<=sizeE;own++,ptn>>=1)){
			for((bit=1,you=0;(board[you]!=ptn)&&(board[own]>=bit);you++)){
					((bit<<=1)) ;
			}
			((board[own]>bit))&& return ;
			((board[own]<bit))&& break ;
		}
	}
	((COUNT8++));
}
#
# 最上段行のクイーンが角以外にある場合の探索 */
function Backtrack2(){
	local min=$1;
	local left=$2;
	local down=$3;
	local right=$4; 
	local bitmap=$((MASK&~(left|down|right)));
	((min==sizeE))&&{ 
		((bitmap))&&{
			((!(bitmap&LASTMASK)))&&{
					board[min]=$bitmap;
					symmetryOps ;
			}
		}
	}||{
    #枝刈り
		((min<BOUND1))&&{  #上部サイド枝刈り
			((bitmap|=SIDEMASK));
			((bitmap^=SIDEMASK));
		} 
    #枝刈り
		((min==BOUND2))&&{ #下部サイド枝刈り
				((!(down&SIDEMASK)))&& return ;
				(((down&SIDEMASK)!=SIDEMASK))&&((bitmap&=SIDEMASK));
		}
		while((bitmap));do
			((bitmap^=board[min]=bit=-bitmap&bitmap)); 
			Backtrack2 $((min+1)) $(((left|bit)<<1)) $(((down|bit)))  $(((right|bit)>>1)) ;
		done
	}
}
#
# 最上段行のクイーンが角にある場合の探索
function Backtrack1(){
	local min=$1;
	local left=$2;
	local down=$3;
	local right=$4; 
	local bitmap=$((MASK&~(left|down|right)));
	((min==sizeE))&&{
		 ((bitmap))&&{
			 	board[min]=$bm;
        #枝刈りによりsymmetryOpsは不要
				#symmetryOps ;
				((COUNT8++)) ;
		 }
	}||{
     #枝刈り
		 ((min<BOUND1))&&{
			 	((bitmap|=2));
			 	((bitmap^=2));
		 }
		 while((bitmap));do
			((bitmap^=board[min]=bit=(-bitmap&bitmap))) ;
			Backtrack1 $((min+1)) $(((left|bit)<<1))  $((down|bit)) $(((right|bit)>>1)) ;
		 done
	}
}
function func_BOUND1(){
	(($1<sizeE))&&{
		((board[1]=bit=1<<BOUND1));
		Backtrack1 2 $(((2|bit)<<1)) $((1|bit)) $((bit>>1));
	}
}
function func_BOUND2(){
	(($1<$2))&&{
		((board[0]=bit=1<<BOUND1));
		Backtrack2 1 $((bit<<1)) $bit $((bit>>1)) ;
	}
}
#
function N-QueenLogic_Q12(){
	board[0]=1;
	((sizeE=(size-1))); 	
	((MASK=(1<<size)-1));
	((TOPBIT=1<<sizeE));
	BOUND1=2;
	while((BOUND1>1&&BOUND1<sizeE));do
		func_BOUND1 "$BOUND1";
		((BOUND1++));
	done
	((SIDEMASK=LASTMASK=(TOPBIT|1)));
	((ENDBIT=TOPBIT>>1));
	BOUND1=1;
	((BOUND2=size-2));
	while((BOUND1>0&&BOUND2<sizeE&&BOUND1<BOUND2));do
		func_BOUND2 "$BOUND1" "$BOUND2";
		((BOUND1++,BOUND2--));
		((ENDBIT>>=1));
		((LASTMASK|=LASTMASK>>1|LASTMASK<<1)) ;
	done
	((UNIQUE=COUNT8+COUNT4+COUNT2)) ;
	((TOTAL=COUNT8*8+COUNT4*4+COUNT2*2));
}
#
N-Queen12(){
  local -i max=17;
	local -i min=2;
	local startTime=0;
	local endTime=0;
	local hh=mm=ss=0; 		# いっぺんにに初期化することもできます
  echo " N:        Total       Unique        hh:mm:ss" ;
  for ((size=min;size<=max;size++));do
		COUNT2=COUNT4=COUNT8=0;
		startTime=`date +%s` ;
		N-QueenLogic_Q12 ;
		endTime=`date +%s`;					# 計測終了時間
		ss=`expr ${endTime} - ${startTime}` # hh:mm:ss 形式に変換
		hh=`expr ${ss} / 3600`
		ss=`expr ${ss} % 3600`
		mm=`expr ${ss} / 60`
		ss=`expr ${ss} % 60`
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $size $TOTAL $UNIQUE $hh $mm $ss ;
  done
}
#
# 実行はコメントアウトを外して、 $ ./BASH_N-Queen.sh 
  echo "<>１２．BT＋Bit＋対称解除Bit＋クイーンの位置による振り分け(BOUND1+BOUND2)＋枝刈り＋最適化 N-Queen12()";
  N-Queen12;
#
#
