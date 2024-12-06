#!/usr/bin/bash
#
# ソースの下部分に各ステップの実行コマンドがコメントアウトされています。
# 実行したいステップのコメントを解除して以下のコマンドで実行します。
# 
# 実行方法
# $ bash 01Bash.sh 
#
# bash-5.1$ bash 01Bash.sh
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
# 11:         2680          341         0:00:02
# 12:        14200         1787         0:00:09
# 
#                                Bash
#<> 12. BOUND1,2の枝刈りと最適化 N12=00:08
#<> 11.BOUND1,2の枝刈り          N12=02:00
#<> 10.BOUND1, BOUND2            N12=03:50
#<>  8. 枝刈り                   N12=05:54
#<>  7. ビットマップと対象解除法 N12=05:47
#<>  6. ビットマップ             N12=01:52
#<>  5. 枝刈り                   N12=03:36
#<>  4. 対象解除法               N12=06:48
#<>  3. バックトラック           N12=07:30
#<>  2. 配置フラグ  
#<>  1. ブルートフォース

typeset -i TOTAL=0;
typeset -i UNIQUE=0;
typeset -a flag_a="";
typeset -a flag_b="";
typeset -a flag_c="";
typeset -a board="";
typeset -a scratch="";
typeset -i size=0;
typeset -i sizeE=size-1;
typeset -i MASK=SIDEMASK=LASTMASK=0;
typeset -i COUNT2=0;
typeset -i COUNT4=0;
typeset -i COUNT8=0;
typeset -i BOUND1=BOUND2=0;
typeset -i bit=TOPBIT=ENDBIT=0;
#
function symmetryOps_pruning()
{
  ((board[BOUND2]==1))&&{
    for((ptn=2,own=1;own<=sizeE;own++,ptn<<=1)){
      for((bit=1,you=sizeE;(board[you]!=ptn)&&(board[own]>=bit);you--)){
        ((bit<<=1));
      }
      ((board[own]>bit))&& return ;
      ((board[own]<bit))&& break ;
    }
    # 90
    ((own>sizeE))&&{
      ((COUNT2++));
      return ;
    }
  }
  # 180
  ((board[sizeE]==ENDBIT))&&{
    for((you=sizeE-1,own=1;own<=sizeE;own++,you--)){
      for((bit=1,ptn=TOPBIT;(ptn!=board[you])&&(board[own]>=bit);ptn>>=1)){
        ((bit<<=1));
      }
      ((board[own]>bit))&& return ;
      ((board[own]<bit))&& break ;
    }
    ((own>sizeE))&&{
      ((COUNT4++));
      return ;
    }
  }
  # 270
  ((board[BOUND1]==TOPBIT))&&{
    for((ptn=TOPBIT>>1,own=1;own<=sizeE;own++,ptn>>=1)){
      for((bit=1,you=0;(board[you]!=ptn)&&(board[own]>=bit);you++)){
        ((bit<<=1));
      }
      ((board[own]>bit))&& return ;
      ((board[own]<bit))&& break ;
    }
  }
  ((COUNT8++));
}
function BackTrack2()
{
  local -i min="$1";
  local -i left="$2";
  local -i down="$3";
  local -i right="$4";
  local bitmap=$((MASK&~(left|down|right)));
  ((min==sizeE))&&{
    ((bitmap))&&{
      ((!(bitmap&LASTMASK)))&&{
        board[$min]=$bitmap;
        symmetryOps_pruning;
      }
    }
  }||{
    ((min<BOUND1))&&{
      ((bitmap|=SIDEMASK));
      ((bitmap^=SIDEMASK));
    }
    ((min==BOUND2))&&{
      ((!(down&SIDEMASK)))&& return ;
      (((down&SIDEMASK)!=SIDEMASK))&&((bitmap&=SIDEMASK));
    }
    while((bitmap));do
    ((bitmap^=board[min]=bit=-bitmap&bitmap));
    BackTrack2 "$((min+1))" "$(((left|bit)<<1))" "$((down|bit))" "$(((right|bit)>>1))";
    done
  }
}
function BackTrack1()
{
  local -i min="$1";
  local -i left="$2";
  local -i down="$3";
  local -i right="$4";
  local bitmap=$((MASK&~(left|down|right)));
  ((min==sizeE))&&{
    ((bitmap))&&{
      board[$min]=$bitmap;
      ((COUNT8++));
    }
  }||{
    ((min<BOUND1))&&{
      ((bitmap|=2));
      ((bitmap^=2));
    }
    while((bitmap));do
      ((bitmap^=board[min]=bit=(-bitmap&bitmap)));
      BackTrack1 "$((min+1))" "$(((left|bit)<<1))" "$((down|bit))" "$(((right|bit)>>1))";
    done
  }
}
function N-Queens12_rec()
{
  board[0]=1;
  ((sizeE=(size-1)));
  ((MASK=(1<<size)-1));
  ((TOPBIT=1<<sizeE));
  BOUND1=2;
  while((BOUND1>1&&BOUND1<sizeE));do
    ((board[1]=bit=1<<BOUND1)) ;
    BackTrack1 2 $(((2|bit)<<1)) $((1|bit)) $((bit>>1)) ;
    ((BOUND1++));
  done
  ((SIDEMASK=LASTMASK=(TOPBIT|1)));
  ((ENDBIT=TOPBIT>>1));
  BOUND1=1;
  ((BOUND2=size-2));
  while((BOUND1>0&&BOUND2<sizeE&&BOUND1<BOUND2));do
    ((board[0]=bit=1<<BOUND1));
    BackTrack2 1 $((bit<<1)) $bit $((bit>>1)) ;
    ((BOUND1++,BOUND2--));
    ((ENDBIT>>=1));
    ((LASTMASK|=LASTMASK>>1|LASTMASK<<1));
  done
  ((UNIQUE=COUNT2+COUNT4+COUNT8));
  ((TOTAL=COUNT2*2 + COUNT4*4 + COUNT8*8));
}
# 12.BOUND1,2と枝刈り、最適化
function N-Queens12()
{
  local -i max=17;
  local -i min=2;
  local startTime=0;
  local endTime=0;
  local hh=mm=ss=0;
  echo " N:        Total       Unique        hh:mm:ss" ;
  for((size=min;size<=max;size++)){
    COUNT2=COUNT4=COUNT8=0;
    startTime=$(date +%s);
    N-Queens12_rec ;
    endTime=$(date +%s);
    ss=$((endTime-startTime));
    hh=$((ss/3600));
    ss=$((ss%3600));
    mm=$((ss/60));
    ss=$((ss%60));
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $size $TOTAL $UNIQUE $hh $mm $ss ;
  }
}
function getUnique()
{
  echo $((COUNT2+COUNT4+COUNT8));
}
function getTotal()
{
  echo $(( COUNT2*2+COUNT4*4+COUNT8*8));
}
function rotate_bitmap_ts()
{
  local -i t=0;
  for((i=0;i<size;i++)){
    t=0;
    for((j=0;j<size;j++)){
      ((t|=((trial[j]>>i)&1)<<(size-j-1)));
    }
    scratch[$i]=$t;
  }
}
function rotate_bitmap_st()
{
  local -i t=0;
  for((i=0;i<size;i++)){
    t=0;
    for((j=0;j<size;j++)){
      ((t|=((scratch[j]>>i)&1)<<(size-j-1)));
    }
    trial[$i]=$t;
  }
}
function rh()
{
  local -i a=$1;
  local -i sz=$2;
  local -i tmp=0;
  for((i=0;i<=sz;i++)){
    ((a&(1<<i)))&&{
      ((tmp|=(1<<(sz-i)) ));
    }
  }
  echo $tmp;
}
function vMirror_bitmap()
{
  local -i score=0;
  local -i sizeE=$((size-1));
  for((i=0;i<size;i++)){
    score=${scratch[$i]};
    trial[$i]=$(rh "$score" $sizeE);
  }
}
function intncmp_bs()
{
  local -i rtn=0;
  for((i=0;i<size;i++)){
    rtn=$(echo "${board[$i]}-${scratch[$i]}"+10);
    ((rtn!=10))&&{ break ; }
  }
  echo "$rtn";
}
function intncmp_bt()
{
  local -i rtn=0;
  for((i=0;i<size;i++)){
    rtn=$(echo "${board[$i]}-${trial[$i]}"+10);
    ((rtn!=10))&&{ break; }
  } 
  echo "$rtn";
}
function symmetryOps_bitmap()
{
  local -i nEquiv=0;
  for((i=0;i<size;i++)){ 
    trial[$i]=${board[$i]}; 
  }
  rotate_bitmap_ts;
  # 90
  k=$(intncmp_bs);
  ((k>10))&&{
    return ;
  }
  ((k==10))&&{
    nEquiv=2;
  }||{
    rotate_bitmap_st;
    # 180
    k=$(intncmp_bt);
    ((k>10))&&{
      return ;
    }
    ((k==10))&&{
      nEquiv=4;
    }||{
      rotate_bitmap_ts;
      # 270
      k=$(intncmp_bs);
      ((k>10))&&{
        return ;
      }
      nEquiv=8;
    }
  }
  for((i=0;i<size;i++)){ 
    scratch[$i]=${board[$i]}; 
  }
  vMirror_bitmap;
  k=$(intncmp_bt);
  ((k>10))&&{
    return ;
  }
  ((nEquiv>2))&&{
    # 90
    rotate_bitmap_ts;
    k=$(intncmp_bs);
    ((k>10))&&{
      return ;
    }
    ((nEquiv>4))&&{
      # 180
      rotate_bitmap_st;
      k=$(intncmp_bt);
      ((k>10))&&{
        return ;
      }
      # 270
      rotate_bitmap_ts;
      k=$(intncmp_bs);
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
function N-Queens11_rec()
{
  local -i min="$1";
  ((TOPBIT=1<<(size-1)));
  board[0]=1;
  for((BOUND1=2;BOUND1<size-1;BOUND1++)){
    bit=$((1<<BOUND1));
    board[1]=$bit;
    backTrack1 "2" "$(((2|bit)<<1))" "$(((1|bit)))" "$((bit>>1))";
  }
  LASTMASK=$((TOPBIT|1));
  SIDEMASK=$LASTMASK;
  ENDBIT=$((TOPBIT>>1));
  BOUND2=$((size-2));
  for((BOUND1=1;BOUND1<BOUND2;BOUND1++)){
    bit=$((1<<BOUND1));
    board[0]=$bit;
    backTrack2 "1" "$((bit<<1))" "$((bit))" "$((bit>>1))";
    ((LASTMASK|=LASTMASK>>1|LASTMASK<<1));
    ((ENDBIT>>=1));
    ((BOUND2--));
  }
}
# 11.BOUND1,2の枝刈り
function N-Queens11()
{
  local -i max=15
  local -i min=2;
  local startTime=;
  local endTime=;
  local hh=mm=ss=0;
  echo " N:        Total       Unique        hh:mm:ss" ;
  for((size=min;size<=max;size++)){
    TOTAL=0;
    UNIQUE=0;
    COUNT2=COUNT4=COUNT8=0;
    for((j=0;j<$size;j++)){ board[$j]=$j; }
    MASK=$(((1<<size)-1));
    startTime=$(date +%s);
    N-Queens11_rec 0;
    endTime=$(date +%s);
    ss=$((endTime-startTime));
    hh=$((ss/3600));
    ss=$((ss%3600));
    mm=$((ss/60));
    ss=$((ss%60));
    TOTAL=$(getTotal);
    UNIQUE=$(getUnique);
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $size $TOTAL $UNIQUE $hh $mm $ss ;
  }
}
function backTrack2()
{
  local -i bit;
  local -i min="$1";
  local -i left="$2";
  local -i down="$3";
  local -i right="$4";
  local -i bitmap=0;
  bitmap=$((MASK&~(left|down|right)));
  ((min==size&&!bitmap))&&{
    board[$min]=bitmap;
    symmetryOps_bitmap;
  }||{
    while((bitmap));do
      bit=$((-bitmap&bitmap));
      board[$min]=$bit
      bitmap=$((bitmap^bit));
      backTrack2 "$((min+1))" "$(((left|bit)<<1))" "$((down|bit))" "$(((right|bit)>>1))" ;
    done
  }
}
function backTrack1()
{
  local -i bit;
  local -i min="$1";
  local -i left="$2";
  local -i down="$3";
  local -i right="$4";
  local -i bitmap=0;
  bitmap=$((MASK&~(left|down|right)));
  ((min==size&&!bitmap))&&{
    board[$min]=bitmap;
    symmetryOps_bitmap;
  }||{
    while((bitmap));do
      bit=$((-bitmap&bitmap));
      board[$min]=$bit;
      bitmap=$((bitmap^bit));
      backTrack1 "$((min+1))" "$(((left|bit)<<1))" "$((down|bit))" "$(((right|bit)>>1))" ;
    done
  }
}
function N-Queens10_rec()
{
  local -i min="$1";
  ((TOPBIT=1<<(size-1)));
  board[0]=1;
  for((BOUND1=2;BOUND1<size-1;BOUND1++)){
    bit=$((1<<BOUND1));
    board[1]=$bit;
    backTrack1 "2" "$(((2|bit)<<1))" "$(((1|bit)))" "$((bit>>1))" ;
  }
  LASTMASK=$((TOPBIT|1));
  SIDEMASK=$LASTMASK;
  ENDBIT=$((TOPBIT>>1));
  BOUND2=$((size-2));
  for((BOUND1=1;BOUND1<BOUND2;BOUND1++)){
    bit=$((1<<BOUND1)) ;
    board[0]=$bit;
    backTrack2 "1" "$((bit<<1))" "$((bit))" "$((bit>>1))" ;
    ((LASTMASK|=LASTMASK>>1|LASTMASK<<1));
    ((ENDBIT>>=1)) ;
    ((BOUND2--));
  }
}
# 10. BOUND1, BOUND2
function N-Queens10()
{
  local -i max=15;
  local -i min=2;
  local startTime=;
  local endTime=;
  local hh=mm=ss=0;
  echo " N:        Total       Unique        hh:mm:ss" ;
  for((size=min;size<=max;size++)){
    TOTAL=0;
    UNIQUE=0;
    COUNT2=COUNT4=COUNT8=0;
    for((j=0;j<$size;j++)){
      board[$j]=$j;
    }
    MASK=$(((1<<size)-1));
    startTime=$(date +%s);
    N-Queens10_rec 0;
    endTime=$(date +%s);
    ss=$((endTime-startTime));
    hh=$((ss/3600));
    ss=$((ss%3600));
    mm=$((ss/60));
    ss=$((ss%60));
    TOTAL=$(getTotal);
    UNIQUE=$(getUnique);
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $size $TOTAL $UNIQUE $hh $mm $ss ;

  }
}
function backTrack()
{
  local -i bit;
  local -i min="$1";
  local -i left="$2";
  local -i down="$3";
  local -i right="$4";
  local -i bitmap=0;
  bitmap=$((MASK&~(left|down|right)));
  ((min==size&&!bitmap))&&{
    board[$min]=bitmap;
    symmetryOps_bitmap;
  }||{
    while((bitmap));do
      bit=$((-bitmap&bitmap));
      board[$min]=$bit;
      bitmap=$((bitmap^bit));
      backTrack "$((min+1))" "$(((left|bit)<<1))" "$((down|bit))" "$(((right|bit)>>1))" ;
    done
  }
}
function N-Queens9_rec()
{
  local -i min="$1";
  ((TOPBIT=1<<(size-1)));
  LASTMASK=$((TOPBIT|1));
  SIDEMASK=$LASTMASK;
  ENDBIT=$((TOPBIT>>1));
  BOUND2=$((size-2));
  for((BOUND1=0;BOUND1<BOUND2;BOUND1++)){
    bit=$((1<<BOUND1));
    board[0]=$bit;
    backTrack "1" "$((bit<<1))" "$((bit))" "$((bit>>1))";
    ((LASTMASK|=LASTMASK>>1|LASTMASK<<1));
    ((ENDBIT>>=1));
    ((BOUND2--));
  }
}
# 9.BOUND1
function N-Queens9()
{
  local -i max=15;
  local -i min=2;
  local startTime=;
  local endTime=;
  local hh=mm=ss=0;
  echo " N:        Total       Unique        hh:mm:ss" ;
  for((size=min;size<=max;size++)){
    TOTAL=0;
    UNIQUE=0;
    COUNT2=COUNT4=COUNT8=0;
    for((j=0;j<$size;j++)){
      board[$j]=$j;
    }
    MASK=$(((1<<size)-1));
    startTime=$(date +%s);
    N-Queens9_rec 0;
    endTime=$(date +%s);
    ss=$((endTime-startTime));
    hh=$((ss/3600));
    ss=$((ss%3600));
    mm=$((ss/60));
    ss=$((ss%60));
    TOTAL=$(getTotal);
    UNIQUE=$(getUnique);
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $size $TOTAL $UNIQUE $hh $mm $ss ;
  }
}
function N-Queens8_rec()
{
  local -i min="$1";
  local -i left="$2";
  local -i down="$3";
  local -i right="$4";
  local -i bitmap=0;
  bitmap=$((MASK&~(left|down|right)));
  ((min==size&&!bitmap))&&{
    board[$min]=$bitmap;
    symmetryOps_bitmap;
  }||{
    ((min!=0))&&{
      lim=$size;
    }||{
      lim=$(((size+1)/2));
    }
    for((s=min+1;s<lim,bitmap;s++)){
      bit=$((-bitmap&bitmap));
      board[$min]=$bit;
      bitmap=$((bitmap^bit));
      N-Queens8_rec "$((min+1))" "$(((left|bit)<<1))" "$((down|bit))" "$(((right|bit)>>1))" ;
    }
  }
}
# 8. Bitmapと対象解除法の枝刈り
function N-Queens8()
{
  local -i max=15;
  local -i min=2;
  local startTime;
  local endTime=;
  local hh=mm=ss=0;
  echo " N:        Total       Unique        hh:mm:ss" ;
  for((size=min;size<=max;size++)){
    TOTAL=0;
    UNIQUE=0;
    COUNT2=COUNT4=COUNT8=0;
    for((j=0;j<$size;j++)){
      board[$j]=$j;
    }
    MASK=$(((1<<size)-1));
    startTime=$(date +%s);
    N-Queens8_rec 0 0 0 0;
    endTime=$(date +%s);
    ss=$((endTime-startTime));
    hh=$((ss/3600));
    ss=$((ss%3600));
    mm=$((ss/60));
    ss=$((ss%60));
    TOTAL=$(getTotal);
    UNIQUE=$(getUnique);
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $size $TOTAL $UNIQUE $hh $mm $ss ;
  }
}
function N-Queens7_rec()
{
  local -i min="$1";
  local -i left="$2";
  local -i down="$3";
  local -i right="$4";
  local -i bitmap=0;
  bitmap=$((MASK&~(left|down|right)));
  ((min==size&&!bitmap))&&{
    board[$min]=$bitmap;
    symmetryOps_bitmap;
  }||{
    while((bitmap));do
      bit=$((-bitmap&bitmap));
      board[$min]=$bit;
      bitmap=$((bitmap^bit));
      N-Queens7_rec "$((min+1))" "$(((left|bit)<<1))" "$((down|bit))" "$(((right|bit)>>1))" ;
    done
  }
}
# 7. Bitmapと対象解除法
function N-Queens7()
{
  local -i max=15;
  local -i min=2;
  local startTime=;
  local endTime=;
  local hh=mm=ss=0;
  echo " N:        Total       Unique        hh:mm:ss" ;
  for((size=min;size<=max;size++)){
    TOTAL=0;
    UNIQUE=0;
    COUNT2=COUNT4=COUNT8=0;
    for((j=0;j<size;j++)){ board[$j]=$j; }
    MASK=$(((1<<size)-1));
    startTime=$(date +%s);
    N-Queens7_rec 0 0 0 0;
    endTime=$(date +%s);
    ss=$((endTime-startTime));
    hh=$((ss/3600));
    ss=$((ss%3600));
    mm=$((ss/60));
    ss=$((ss%60));
    TOTAL=$(getTotal);
    UNIQUE=$(getUnique);
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $size $TOTAL $UNIQUE $hh $mm $ss ;
  }
}
function intncmp()
{
  local -i k;
  local -i rtn=0;
  local -i n=$1;
  for((k=0;k<n;k++)){
    rtn=$((board[k]-trial[k]));
    ((rtn!=0))&&{ break ;}
  }
  echo "$rtn";
}
function rotate()
{
  local -i j;
  local -i k;
  local -i n=$1;
  local -i incr;
  local neg=$2;
  if [ "$neg" = "true" ]; then
    k=0;
  else
    k=$((n-1));
  fi
  if [ "$neg" = "true" ]; then
    incr=$((incr+1));
  else
    incr=$((incr-1));
  fi
  for((j=0;j<n;k+=incr)){
    j=$((j+1))
    scratch[$j]=${trial[$k]};
  }
  if [ "$neg" = "true" ]; then
    k=$((n-1));
  else
    k=0;
  fi
  for((j=0;j<n;k-=incr)){
    j=$((j+1));
    trial[${scratch[$j]}]=$k;
  }
}
function vMirror()
{
  local -i j;
  local -i n=$1;
  for((j=0;j<n;j++)){
    local -i n1=$((n-1));
    trial[$j]=$((n1-trial[j]));
  }
}
function symmetryOps()
{
  local -i k;
  local -i nEquiv;
  local -i size=$1;
  for((k=0;k<size;k++)){
    trial[$k]=${board[$k]};
  }
  # 90
  rotate "$size" "false";
  k=$(intncmp "$size");
  ((k>0))&&{
    echo 0;
    return ;
  }
  ((k==0))&&{
    nEquiv=1;
  }||{
    # 180
    rotate "$size" "false";
    k=$(intncmp "$size");
    ((k>0))&&{
      echo 0;
      return ;
    }
    ((k==0))&&{
      nEquiv=2;
    }||{
      # 270
      rotate "$size" "false";
      k=$(intncmp "$size");
      ((k>0))&&{
        echo 0;
        return ;
      }
      nEquiv=4;
    }
  }
  for((k=0;k<size;k++)){
    trial[$k]=${board[$k]};
  }
  vMirror "$size";
  k=$(intncmp "$size" );
  ((k>0))&&{
    echo 0;
    return ;
  }
  # 4回転とはことなる場合
  ((nEquiv>1))&&{
    # 90
    rotate "$size" "true";
    k=$(intncmp "$size");
    ((k>0))&&{
      echo 0;
      return ;
    }
    # 180
    ((nEquiv>2))&&{
      rotate "$size" "true" ;
      k=$(intncmp "$size");
      ((k>0))&&{
        echo 0;
        return ;
      }
      # 270
      rotate "$size" "true";
      k=$(intncmp "$size");
      ((k>0))&&{
        echo 0;
        return ;
      }
    }
  }
  rtn=$((nEquiv * 2 ));
  echo "$rtn";
  return ;
}
function N-Queens6_rec()
{
  local -i min="$1";
  local -i left="$2";
  local -i down="$3";
  local -i right="$4";
  local -i bitmap=0;
  local -i bit=0;
  ((min==size))&&((TOTAL++))||{
    bitmap=$((MASK&~(left|down|right)));
    while ((bitmap));do
      bit=$((-bitmap&bitmap));
      bitmap=$((bitmap^bit));
      N-Queens6_rec "$((min+1))" "$(((left|bit)<<1))" "$((down|bit))" "$(((right|bit)>>1))" ;
    done
  }
}
# 6. Bitmap
function N-Queens6()
{
  local -i max=15;
  local -i min=2;
  local startTime=;
  local endTime=;
  local hh=mm=ss=0;
  echo " N:        Total       Unique        hh:mm:ss" ;
  for((size=min;size<=max;size++)){
    TOTAL=0;
    UNIQUE=0;
    MASK=$(((1<<size)-1));
    startTime=$(date +%s);
    N-Queens6_rec 0 0 0 0;
    endTime=$(date +%s);
    ss=$((endTime-startTime));
    hh=$((ss/3600));
    ss=$((ss%3600));
    mm=$((ss/60));
    ss=$((ss%60));
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $size $TOTAL $UNIQUE $hh $mm $ss ;
  }
}
# 5. 枝刈り
function N-Queens5_rec()
{
  local -i min="$1";
  local -i size="$2";
  local -i i=0;
  ((min != 0))&&{
    lim=$size;
  }||{
    lim=$(((size+1)/2));
  }
  for((i=0;i<lim;i++)){
    [ "${flag_a[$i]}" != "true" ]&& \
    [ "${flag_b[$min+$i]}" != "true" ] && \
    [ "${flag_c[$min-$i+$size-1]}" != "true" ]&& {
      board[$min]=$i;
      ((min==(size-1)))&&{
        tst=$(symmetryOps "$size");
        ((tst!=0))&&{
          ((UNIQUE++));
          TOTAL=$((TOTAL+tst));
        }
      }||{
        flag_a[$i]="true";
        flag_b[$min+$i]="true";
        flag_c[$min-$i+$size-1]="true";
        N-Queens5_rec "$((min+1))" "$size";
        flag_a[$i]="";
        flag_b[$min+$i]="";
        flag_c[$min-$i+$size-1]="";
      }
    }
  }
}
function N-Queens5()
{
  local -i max=15;
  local -i min=2;
  local -i N="$min";
  local startTime=0;
  local endTime=0;
  local hh=mm=ss=0;
  echo " N:        Total       Unique        hh:mm:ss" ;
  for((N=min;N<=max;N++)){
    TOTAL=0;
    UNIQUE=0;
    startTime=$(date +%s);
    for((k=0;k<N;k++)){ board[$k]=$k;}
    N-Queens5_rec 0 "$N";
    endTime=$(date +%s);
    ss=$((endTime-startTime));
    hh=$((ss/3600));
    ss=$((ss%3600));
    mm=$((ss/60));
    ss=$((ss%60));
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $N $TOTAL $UNIQUE $hh $mm $ss ;
  }
}
# 4.Symmetry Ops
function N-Queens4_rec()
{
  local -i min="$1";
  local -i size="$2";
  local -i i=0;
  for((i=0;i<size;i++)){
    [ "${flag_a[$i]}" != "true" ]&& \
    [ "${flag_b[$min+$i]}" != "true" ]&& \
    [ "${flag_c[$min-$i+$size-1]}" != "true" ]&&{
      board[$min]="$i";
      ((min==(size-1)))&&{
        tst=$(symmetryOps "$size");
        ((tst!=0))&&{
          ((UNIQUE++));
          TOTAL=$((TOTAL+tst));
        }
      }||{
        flag_a[$i]="true";
        flag_b[$min+$i]="true";
        flag_c[$min-$i+$size-1]="true";
        N-Queens4_red "$((min+1))" "$size";
        flag_a[$i]="";
        flag_b[$min+$i]="";
        flag_c[$min-$i+$size-1]="";
      }
    }
  }
}
# 4.SymmetryOps
function N-Queens4()
{
  local -i max=15;
  local -i min=2;
  local -i N="$min";
  local startTime=0;
  local endTime=0;
  local hh=mm=ss=0;
  echo " N:        Total       Unique        hh:mm:ss" ;
  for((N=min;N<=max;N++)){
    TOTAL=0;
    UNIQUE=0;
    startTime=$(date +%s);
    for((k=0;k<N;k++)){ board[k]=k;}
    N-Queens4_rec 0 "$N";
    endTime=$(date +%s);
    ss=$((endTime-startTime));
    hh=$((ss/3600));
    ss=$((ss%3600));
    mm=$((ss/60));
    ss=$((ss%60));
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $N $TOTAL $UNIQUE $hh $mm $ss ;
  }
}
function N-Queens3_rec()
{
  local -i min="$1";
  local -i size="$2";
  local -i i=0;
  for((i=0;i<size;i++)){
    [ "${flag_a[$i]}" != "true" ]&& \
    [ "${flag_b[$min+$i]}" != "true" ]&& \
    [ "${flag_c[$min-$i+$size-1]}" != "true" ]&& {
      pos[$min]=$i;
      ((min==(size-1)))&&{
        ((TOTAL++));
      }||{
        flag_a[$i]="true";
        flag_b[$min+$i]="true";
        flag_c[$min-$i+$size-1]="true";
        N-Queens3_rec "$((min+1))" "$size"; 
        flag_a[$i]="";
        flag_b[$min+$i]="";
        flag_c[$min-$i+$size-1]="";
      }
    }
  }
}
# 3.BackTrack
function N-Queens3()
{
  local -i max=15;
  local -i min=2;
  local -i N="$min";
  local startTime=0;
  local endTime=0;
  local hh=mm=ss=0;
  echo " N:        Total       Unique        hh:mm:ss" ;
  for((N=min;N<=max;N++)){
    TOTAL=0;
    UNIQUE=0;
    startTime=$(date +%s);
    N-Queens3_rec 0 "$N";
    endTime=$(date +%s);
    ss=$((endTime-startTime));
    hh=$((ss/3600));
    ss=$((ss/3600));
    mm=$((ss/60));
    ss=$((ss%60));
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $N $TOTAL $UNIQUE $hh $mm $ss ;
  }
}
# 2.flag
COUNT=1;
function N-Queens2()
{
  local -i min="$1";
  local -i size="$2";
  local flag_a="";
  local -i i=0;
  local -i j=0;
  for((i=0;i<size;i++)){
    [ "${flag_a[i]}" != "true" ]&&{
      pos[$min]="$i";
      ((min==(size-1)))&&{
        echo -n "$((COUNT++)): ";
        for((j=0;j<size;j++)){
          echo -n "${pos[j]} ";
        }
        echo "";
      }||{ 
        flag_a[$i]="true" ;
        N-Queens2 "$((min+1))" "$size";
        flag_a[$i]="";
      }
    }
  }
}
# 1.BluteForce
function N-Queens1()
{
  local -i min="$1";
  local -i size="$2";
  local -i i=0;
  local -i j=0;
  for((i=0;i<size;i++)){
    pos[$min]="$i";
    ((min==(size-1)))&&{
      echo -n "$((COUNT++)): ";
      for((j=0;j<size;j++)){
        echo -n "${pos[j]} ";
      }
      echo "";
    }||N-Queens1 "$((min+1))" "$size" ;
  }
}
#
#
# 実行は以下のコメント部分を解除してください
#
echo "<> 12. BOUND1,2の枝刈りと最適化 N12=0:00:08
N-Queens12 ;
# 
#echo "<> 11.BOUND1,2の枝刈り          N12=0:02:00
#N-Queens11 ;
#
#echo "<> 10.BOUND1, BOUND2            N12=0:03:50
#N-Queens10 ;
#
#echo "<> 9. BOUND1                    N12=0:03:19
#N-Queens9 ;
#
#echo "<> 8. 枝刈り                    N12=0:05:54
#N-Queens8 ;
#
#echo "<> 7. ビットマップと対象解除法  N12=0:05:47
# N-Queens7 ;
#
#echo "<> 6. ビットマップ              N12=0:01:52
# N-Queens6;
#
#echo "<> 5. 枝刈り                    N12=0:03:36
# N-Queens5 ;
#
#echo "<> 4. 対象解除法";              N12=0:06:48
# N-Queens4 ;
#
#echo "<> 3. バックトラック";          N12=0:07:30
# N-Queens3;
#
#echo "<> 2. 配置フラグ";
# N-Queens2 0 8;
#
#echo "<> 1. ブルートフォース";
# N-Queens1 0 8 ;

exit;
