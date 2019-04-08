#!/bin/bash
#
#
# アルゴリズムとデータ構造  
# 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#
# １１．BT＋Bit＋対称解除Bit＋クイーンの位置による振り分け(BOUND1+BOUND2)＋枝刈り
#
# 実行結果
# <>１１．BT＋Bit＋対称解除Bit＋クイーンの位置による振り分け(BOUND1+BOUND2)＋枝刈り N-Queen11()
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
function N-QueenLogic_Q11(){
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
N-Queen11(){
  local -i max=17;
	local -i min=2;
	local startTime=0;
	local endTime=0;
	local hh=mm=ss=0; 		# いっぺんにに初期化することもできます
  echo " N:        Total       Unique        hh:mm:ss" ;
  for ((N=min;N<=max;N++));do
		COUNT2=COUNT4=COUNT8=0;
		startTime=`date +%s` ;
		N-QueenLogic_Q11 ;
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
# 実行はコメントアウトを外して、 $ ./BASH_N-Queen.sh 
  echo "<>１１．BT＋Bit＋対称解除Bit＋クイーンの位置による振り分け(BOUND1+BOUND2)＋枝刈り N-Queen11()";
  N-Queen11;
#
#
