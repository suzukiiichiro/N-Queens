#!/bin/bash

function display(){
  for((i=0;i<"$nElems";i++)){
    echo "$i" "${array["$i"]}";
  }
  echo "-----";
}
function insert(){
  array[((nElems++))]="$1";
}
function setArray(){
  nElems=0;
  for((i=0;i<"$1";i++)){
    insert $(echo "$RANDOM");
  }
}

##
# qsort():  (Quick sort)
# Average:
# Stability:
# Uses:
function quickSort() {
  local -i l r m i j k part temp ;
  (( l=i=$1, r=j=$2, m=(l+r)/2 ));
  part="${array[m]}" ;
  while ((j > i)); do
    while [[ 1 ]]; do
      (( "${array[i]}"<"$part"))&&((i++))||break ;
    done
    while [[ 1 ]]; do
      (( "${array[j]}">"$part"))&&((j--))||break ;
    done
    if (( i <= j )); then
      temp="${array[i]}";
      array[i]="${array[j]}";
      array[j]="$temp" ;
      (( i++, j-- )) ;
    fi
  done
  (( l<j )) && quickSort $l $j ;
  (( r>i )) && quickSort $i $r ;
}

##
# shellSort(): (Shell Sort)
# Worst-case:  (N((log N)/(log log N))^2)
#              (can be fast, but sensitive to input data)
# Stability:   stable
# Sensitivity: sensitive
function shellSort(){
    shell=1 ;
    while (( shell < nElems/3 )); do
        shell=$(( shell*3+1 )) ;
    done
    while (( shell>0 )); do
      for(( i=shell; i<nElems; i++ )); do
        t="${array[i]}" ;
        inner="$i" ;
        while (( inner>shell-1 && "${array[inner-shell]}">=t )); do
          array[inner]="${array[inner-shell]}" ;
          inner=$(( inner-shell )) ;
        done
        array[inner]="$t" ;
      done
      shell=$(( (shell-1)/3 )) ;
    done
}

##
# msort():     (Merge sort)
# Average:     (N(log N)) (pretty fast)
# Stability:   stable
# Sensitivity: insensitive to the key distribution of the input
# Drawbacks:   Requires temporary space equal in size to the input array.
# msort will use recursion by default -- msort_iter is additionally provided.
function mergeSortLogic(){
  local first=$1 ;
  local middle=$2 ;
  local last=$3 ;
  local n i j n1 ;
  (( n=last-first+1 )) ;

  for (( i=first, j=0; i<=last; )); do
    msortWrkArr[j++]="${array[i++]}" ;
  done

  (( middle>last )) && (( middle=(first+last)/2 )) ;
  (( n1=middle-first+1 )) ;
  for (( i=first, j=0, k=n1; i<=last; i++ )); do
    if {
      (( j < n1 )) && {
        (( k == n )) || { 
          (( ${msortWrkArr[j]} < ${msortWrkArr[k]} ))
        }
      }
    }; then
      array[i]="${msortWrkArr[j++]}" ;
    else
      array[i]="${msortWrkArr[k++]}" ;
    fi
  done
}
function mergeSort(){
    local first="$1" ;
    local last="$2" ;
    (( last > first )) || return 0;
    local mid=$(( (first+last) / 2 ));
    mergeSort "$first" "$mid" ;
    mergeSort "$((mid+1))" "$last"
    mergeSortLogic "$first" "$mid" "$last" ;
}

##
# insertionSort():    (Insertion Sort)
# Worst-case:  (N^2) (slow)
# Stability:   stable
# Sensitivity: insensitive
# Use: Fastest type of sort for nearly-sorted data.
function insertionSort(){
  for(( out=1; out<"$nElems"; out++ ));do
    t="${array[out]}" ;
    for(( in=out; in>0 && "${array[in-1]}">t; in-- ));do
      array[in]="${array[in-1]}" ;
    done
    array[in]="$t" ;
  done
}

##
# selectionSort():   (Selection Sort)
# Worst-case:  (N^2) (slow)
# Stability:   stable
# Sensitivity: insensitive
# Use:         Quickly find extrema for unordered data
function selectionSort(){
  for(( i=0; i<nElems; i++ ));do
    min="$i" ;
    for(( j=i+1; j<nElems; j++ ));do
      if [ "${array[min]}" -gt "${array[j]}" ] ; then
        min="$j";
      fi
    done
    (( min == i )) && continue;
    t="${array[min]}" ;
    array[min]="${array[i]}" ;
    array[i]="$t" ;
  done
}

##
# bubbleSort():     (Bubble Sort)
# Worst-case:  (N^2) (slow)
# Stability:   stable
# Sensitivity: insensitive
# Use: Fastest type of sort for nearly-sorted data.
function bubbleSort(){
  for(( i=nElems; i>0; i-- )) ;do
    for(( j=0; j<i-1; j++ )) ;do
      if [ "${array[j]}" -gt "${array[j+1]}" ];then
        t="${array[j]}" ; 
        array[j]="${array[j+1]}" ;
        array[j+1]="$t" ;
      fi 
    done 
  done
}
function Sort(){
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
  time Sort 1000 "bubbleSort";
  time Sort 1000 "selectionSort";
  time Sort 1000 "insertionSort";
  time Sort 1000 "mergeSort";
  time Sort 1000 "shellSort" ;
  time Sort 1000 "quickSort" ;
}
#
############################################
# N-Queen
############################################
#
##
# ビット演算に加えてユニーク解(回転・反転）を使って高速化
# 
# ユニーク解の判定とユニーク解の種類の判定   */
COUNT0=0;
Check_Qset5(){
	_BOARD=0 ;
	_BOARD1=$BOUND1;
	_BOARD2=$BOUND2;
	_BOARDE=$SIZEE ;
	 ((BOARD[_BOARD2]==1)) && {
		for ((ptn=2, own=_BOARD+1; own<=_BOARDE; own++, ptn<<=1)) {
			bit=1;
			for ((you=_BOARDE; (BOARD[you]!=ptn) && (BOARD[own]>=bit); you--)) {
				 ((bit<<=1)) ;
			}
			((BOARD[own]>bit)) && return ;
			((BOARD[own]<bit)) && break ;
		}
		#90度回転して同型なら180度回転も270度回転も同型である
		((own>_BOARDE)) && {
				((COUNT2++));
				return;
		}
	}
	#180度回転
	((BOARD[_BOARDE]==ENDBIT)) && {
		for ((you=_BOARDE-1,own=_BOARD+1; own<=_BOARDE; own++,you--)) {
			bit=1;
			for ((ptn=TOPBIT; (ptn!=BOARD[you]) && (BOARD[own]>=bit); ptn>>=1)) {
					((bit<<=1)) ;
			}
			((BOARD[own]>bit)) && return ;
			((BOARD[own]<bit)) && break ;
		}
		#90度回転が同型でなくても180度回転が同型であることもある
		((own>_BOARDE)) && {
		  ((COUNT4++));
			return;
		}
	}
	#270度回転
	((BOARD[_BOARD1]==TOPBIT)) && {
		for ((ptn=TOPBIT>>1,own=_BOARD+1; own<=_BOARDE; own++,ptn>>=1)) {
			bit=1;
			for ((you=_BOARD; (BOARD[you]!=ptn)&&(BOARD[own]>=bit); you++)) {
					 ((bit<<=1)) ;
			}
			((BOARD[own]>bit)) && return;
			((BOARD[own]<bit)) && break;
		}
	}
	((COUNT8++));
}

# 最上段行のクイーンが角以外にある場合の探索 */
Backtrack2_Qset5(){
	local y=$1 left=$2 down=$3 right=$4;
	local bitmap=$((MASK & ~(left|down|right)));
	((y==SIZEE)) && {
		((bitmap)) && {
			((! (bitmap&LASTMASK))) && {
					BOARD[y]=$bitmap;
					Check_Qset5 ;
			}
		}
	} || {
		((y<BOUND1)) && {  #上部サイド枝刈り
			((bitmap|=SIDEMASK));
			((bitmap^=SIDEMASK));
		} 
	 ((y==BOUND2)) && { #下部サイド枝刈り
			((! (down&SIDEMASK))) && return ;
			(((down&SIDEMASK)!=SIDEMASK)) && ((bitmap&=SIDEMASK));
		}
		while ((bitmap)); do
			((bitmap^=BOARD[$y]=bit=-bitmap&bitmap)); 
			Backtrack2_Qset5 $((y+1)) $(((left|bit)<<1)) $(((down|bit)))  $(((right|bit)>>1)) ;
		done
	}
}

# 最上段行のクイーンが角にある場合の探索
Backtrack1_Qset5(){
	local y=$1 left=$2 down=$3 right=$4;
	local bitmap=$((MASK & ~(left|down|right)));
	((y==SIZEE)) && {
		 ((bitmap)) && {
			 	BOARD[y]=$bitmap;
				((COUNT8++)) ;
		 }
	} || {
		 ((y<BOUND1)) && {
			 	((bitmap|=2));
			 	((bitmap^=2));
		 }
		 while ((bitmap)) ;do
			((bitmap^=BOARD[y]=bit=(-bitmap&bitmap))) ;
			Backtrack1_Qset5 $((y+1)) $(((left|bit)<<1))  $((down|bit)) $(((right|bit)>>1)) ;
		 done
	}
}

N-Queen5_logic(){
	local SIZEE=$((SIZE-1));
	local TOPBIT=$((1<<SIZEE));
	local MASK=$(( (1<<SIZE)-1 ));
	BOARD[0]=1;
	for ((BOUND1=2; BOUND1<SIZEE; BOUND1++)) {
			((BOARD[1]=bit=1<<BOUND1));
			Backtrack1_Qset5 2 $(((2|bit)<<1)) $((1|bit)) $((bit>>1));
	}
	((SIDEMASK=LASTMASK=(TOPBIT|1)));
	local ENDBIT=$((TOPBIT>>1));
	for ((BOUND1=1,BOUND2=SIZE-2; BOUND1<BOUND2; BOUND1++,BOUND2--)) {
			((BOARD[0]=bit=1<<BOUND1));
			Backtrack2_Qset5 1 $((bit<<1)) $bit $((bit>>1)) ;
			((LASTMASK|=LASTMASK>>1|LASTMASK<<1)) ;
			((ENDBIT>>=1));
	}
	UNIQUE=$((COUNT8 + COUNT4 + COUNT2)) ;
	TOTAL=$(( COUNT8*8 + COUNT4*4 + COUNT2*2));
}

#
N-Queen5(){
  MAXSIZE=15 ;
  MINSIZE=2 ;
  SIZE=$MINSIZE ;
  echo " N:        Total       Unique        hh:mm" ;
  for (( SIZE=MINSIZE; SIZE<=MAXSIZE; SIZE++ ));do
    COUNT2=0; COUNT4=0; COUNT8=0; 
    local starttime=`date +%s` ;
    N-Queen5_logic ;
    local time=$((`date +%s` - starttime)) ;
    printf "%2d:%13d%13d%13d\n" $SIZE $TOTAL $UNIQUE $time ;
  done
}

# N:        Total       Unique        hh:mm
# 2:            0            0            0
# 3:            0            0            0
# 4:            2            1            0
# 5:           10            2            0
# 6:            4            1            0
# 7:           40            6            0
# 8:           92           12            0
# 9:          352           46            1
#10:          724           92            1
#11:         2680          341            4
#12:        14200         1787           20
#13:        73712         9233          111
#14:       365596        45752          638

##
# ビットマップ
#
#
TOTAL=1;
N-Queen4_rec(){
  local y="$1" left="$2" down="$3" right="$4" bitmap=;
  ((y==SIZE)) && ((TOTAL++))||{
    bitmap=$(( MASK & ~(left|down|right) ));
    while (( bitmap )); do
      bit=$(( -bitmap&bitmap )) ;
      bitmap=$(( bitmap^bit )) ;
      N-Queen4_rec "$((y+1))" "$(((left|bit)<<1))" "$((down|bit))" "$(((right|bit)>>1))"  ;
    done
  }
}
N-Queen4(){
  local MAXSIZE=13;
  local MINSIZE=2 ;
  SIZE=$MINSIZE ;
  echo " N:        Total       Unique        hh:mm" ;
  for (( SIZE=MINSIZE; SIZE<=MAXSIZE; SIZE++ )) {
    TOTAL=0;
    UNIQUE=0;
  	MASK=$(( (1<<SIZE)-1 )) ;
    local starttime=`date +%s` ;
    N-Queen4_rec 0 0 0 0 ;
    local time=$((`date +%s` - starttime)) ;
    printf "%2d:%13d%13d%13d\n" $SIZE $TOTAL $UNIQUE $time ;
  } 
}

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

##
# バックトラック
#
#
TOTAL=1 ;
N-Queen3_rec(){
  local i="$1" j ;
  size=$2 ;
  #flag_a: flag_b: flag_c 
  for((j=0; j<size; j++)){
    [[ -z "${flag_a[j]}" ]] && [[ -z "${flag_b[i+j]}" ]] && [[ -z "${flag_c[i-j+size-1]}" ]] && {
      pos[$i]=$j ;
      (( i == (size-1) )) && ((TOTAL++)) || {
        flag_a[j]="true" ;
        flag_b[i+j]="true" ; 
        flag_c[i-j+size-1]="true" ; 
        N-Queen3_rec "$((i+1))" "$size" ; 
        flag_a[j]="" ;           
        flag_b[i+j]="" ;   
        flag_c[i-j+size-1]="" ; 
      }          
    }
  }  
}

# N:        Total       Unique        hh:mm
# 2:            0            0            0
# 3:            0            0            0
# 4:            2            0            0
# 5:           10            0            0
# 6:            4            0            0
# 7:           40            0            1
# 8:           92            0            0
# 9:          352            0            1
#10:          724            0            7
#11:         2680            0           35
#12:        14200            0          189

N-Queen3(){
  MAXSIZE=12;
  MINSIZE=2 ;
  SIZE=$MINSIZE ;
  echo " N:        Total       Unique        hh:mm" ;
  for (( SIZE=MINSIZE; SIZE<=MAXSIZE; SIZE++ )) {
    TOTAL=0;
    UNIQUE=0;
    local starttime=`date +%s` ;
    N-Queen3_rec 0 "$SIZE";
    local time=$((`date +%s` - starttime)) ;
    printf "%2d:%13d%13d%13d\n" $SIZE $TOTAL $UNIQUE $time ;
  } 
}

##
# 配置フラグ
#
#
COUNT=1 ;
N-Queen2(){
  local i=$1 j size=$2 ;
  for (( j=0; j<size; j++ )) {
    [[ -z "${flag_a[j]}" ]] && {
      pos[i]="$j" ; 
      (( i == (size-1) )) && {
        echo -n "$((COUNT++)): " ;
        for (( i=0; i<size; i++)) {
          echo -n "${pos[i]}" ;
        }
        echo "" ;
      } || {
        flag_a[j]="true" ;         
        N-Queen2 "$((i+1))" "$size" ;
        flag_a[j]="" ; 
      }
    }
  }
}

##
# ブルートフォース　力任せ探索
#
#
COUNT=1 ;
N-Queen1(){
  local i="$1" j size="$2" ;
  for(( j=0; j<size; j++ )) {
      pos[i]="$j" ;
      (( i == size-1 )) && { 
        echo -n "$((COUNT++)): " ;
        for (( x=0; x<size; x++)) {
          echo -n "${pos[x]}" ;
        }
        echo "" ;
      } || N-Queen1 "$((i+1))" "$size" ;
  }  
}

function N-Queen(){
#  N-Queen1 0 8;      # ブルートフォース
#  N-Queen2 0 8;      # 配置フラグ
#  N-Queen3 		      # バックトラック
#  N-Queen4  ;				# ビットマップ
N-Queen5 ; 						# ユニーク解

}


#Sort ;
N-Queen ;
exit ;
