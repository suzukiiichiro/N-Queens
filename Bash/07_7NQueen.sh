#!/bin/bash

# ５．枝刈りと最適化
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
    ((rtn!=0))&&{
      break;
    }
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
    trial[${scratch[j]}]=$k;
  }
}
#
function vMirror(){
  local -i j;
  local -i n=$1;
  local -i n1;
  for((j=0;j<n;j++)){
    n1=$((n-1));
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
  ((nEquiv>1))&&{        #// 4回転とは異なる場合
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
N-Queen5_rec(){
  # ローカル変数は明示的に local をつけ、代入する場合は ""ダブルクォートが必要です。
  # -i は 変数の型が整数であることを示しています
  local -i s;
  local -i lim;
  local -i vTemp;

  local -i min="$1";      # ひとつ目のパラメータ $1をminに代入
  local -i size=$2;       # ふたつ目のパラメータ $2をsizeに代入
  local -i i=0;           # 再帰するために forで使う変数も宣言が必要

  ((min<size-1))&&{
    [ "${flag_c[$min-${board[$min]}+$size-1]}" != "true" ]&& \
    [ "${flag_b[$min+${board[$min]}]}" != "true" ]&&{ 
	    flag_c[$min-${board[$min]}+$size-1]="true";
      flag_b[$min+${board[$min]}]="true";
      N-Queen5_rec "$((min+1))" "$size";
	    flag_c[$min-${board[$min]}+$size-1]=""; 
      flag_b[$min+${board[$min]}]="";
    }
      ((min != 0))&&{
			  lim=$size;
      }||{
        lim=$(((size+1)/2));
      }
      for((s=min+1;s<lim;s++)){
				vTemp=${board[$s]};
				board[$s]=${board[$min]};
				board[$min]=${vTemp};
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
  echo " N:        Total       Unique        hh:mm:ss" ;
  for((N=min;N<=max;N++)){
    TOTAL=0;      # Nが更新される度に TOTALとUNIQUEを初期化
    UNIQUE=0;
    startTime=`date +%s` ;      # 計測開始時間
    for((k=0;k<N;k++)){ board[$k]=$k;}
    N-Queen5_rec 0 "$N";
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
N-Queen5 		      # バックトラック
