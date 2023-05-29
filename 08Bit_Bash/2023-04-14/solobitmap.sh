#!/usr/bin/bash

declare -i TOTAL=0;     # $B%+%&%s%?!<(B
#
: '$B%\!<%I%l%$%"%&%H$r=PNO(B';
function printRecord()
{
  size="$1";
  flag="$2"; # bitmap$BHG$O(B1 $B$=$l0J30$O(B 0
  echo "$TOTAL";
  sEcho=" ";  
  : '$B%S%C%H%^%C%WHG(B
     $B%S%C%H%^%C%WHG$+$i$O!":8$+$i?t$($^$9(B
     $B>e2<H?E>:81&BP>N$J$N$G!"$3$l$^$G$N>e$+$i?t$($k<jK!$H(B
     row$B$r2<$K$?$I$C$F:8$+$i?t$($kJ}K!$H2r$N?t$KJQ$o$j$O$"$j$^$;$s!#(B
     0 2 4 1 3 
    +-+-+-+-+-+
    |O| | | | | 0
    +-+-+-+-+-+
    | | |O| | | 2
    +-+-+-+-+-+
    | | | | |O| 4
    +-+-+-+-+-+
    | |O| | | | 1
    +-+-+-+-+-+
    | | | |O| | 3
    +-+-+-+-+-+
  ';
  if ((flag));then
    for((i=0;i<size;i++)){
      for((j=0;j<size;j++)){
        if (( board[i]&1<<j ));then
          sEcho="${sEcho}$((j)) ";
        fi 
      }
    }
  else 
  : '$B%S%C%H%^%C%WHG0J30(B
     ($B%V%k!<%H%U%)!<%9!"%P%C%/%H%i%C%/!"G[CV%U%i%0(B)
     $B>e$+$i?t$($^$9(B
     0 2 4 1 3 
    +-+-+-+-+-+
    |O| | | | |
    +-+-+-+-+-+
    | | | |O| |
    +-+-+-+-+-+
    | |O| | | |
    +-+-+-+-+-+
    | | | | |O|
    +-+-+-+-+-+
    | | |O| | |
    +-+-+-+-+-+

     ';
    for((i=0;i<size;i++)){
      sEcho="${sEcho}${board[i]} ";
    }
  fi
  echo "$sEcho";

  echo -n "+";
  for((i=0;i<size;i++)){
    echo -n "-";
    if((i<(size-1)));then
      echo -n "+";
    fi
  }
  echo "+";
  for((i=0;i<size;i++)){
    echo -n "|";
    for((j=0;j<size;j++)){
      if ((flag));then
        if (( board[i]&1<<j ));then
          echo -n "O";
        else
          echo -n " ";
        fi
      else
        if((i==board[j]));then
          echo -n "O";
        else
          echo -n " ";
        fi
      fi
      if((j<(size-1)));then
        echo -n "|";
      fi
    }
  echo "|";
  if((i<(size-1)));then
    echo -n "+";
    for((j=0;j<size;j++)){
      echo -n "-";
      if((j<(size-1)));then
        echo -n "+";
      fi
    }
  echo "+";
  fi
  }
  echo -n "+";
  for((i=0;i<size;i++)){
    echo -n "-";
    if((i<(size-1)));then
      echo -n "+";
    fi
  }  
  echo "+";
  echo "";
}
#
: '$BHs:F5"HG%S%C%H%^%C%W(B';
function bitmap_NR()
{
  local -i size="$1";
  local -i row="$2";
  local -i mask=$(( (1<<size)-1 ));
  local -a left[$size];
  local -a down[$size];
  local -a right[$size];
  local -a bitmap[$size]
  local -i bitmap[$row]=mask;
  local -i bit=0;
  while ((row>-1));do
    if (( bitmap[row]>0 ));then
      bit=$(( -bitmap[row]&bitmap[row] ));  # $B0lHV1&$N%S%C%H$r<h$j=P$9(B
      bitmap[$row]=$(( bitmap[row]^bit ));  # $BG[CV2DG=$J%Q%?!<%s$,0l$D$:$D<h$j=P$5$l$k(B
      board[$row]="$bit";                   # Q$B$rG[CV(B
      if (( row==(size-1) ));then
        ((TOTAL++));
        printRecord "$size" "1";            # $B=PNO(B 1:bitmap$BHG(B 0:$B$=$l0J30(B
        ((row--));
      else
        local -i n=$((row++));
        left[$row]=$(((left[n]|bit)<<1));
        down[$row]=$(((down[n]|bit)));
        right[$row]=$(((right[n]|bit)>>1));
        board[$row]="$bit";                 # Q$B$rG[CV(B
        # $B%/%$!<%s$,G[CV2DG=$J0LCV$rI=$9(B
        bitmap[$row]=$(( mask&~(left[row]|down[row]|right[row]) ));
      fi
    else
      ((row--));
    fi
  done 
}
#
: '$B:F5"HG%S%C%H%^%C%W(B';
function bitmap_R()
{
  local -i size="$1"; local -i row="$2";
  local -i left="$3"; local -i down="$4"; local -i right="$5";
  local -i bitmap=;
  local -i bit=;
  local -i col=0;                     # $B:F5"$KI,MW(B
  local -i mask=$(( (1<<size)-1 ));
  if (( row==size ));then
     ((TOTAL++));
     printRecord "$size" "1";         # $B=PNO(B 1:bitmap$BHG(B 0:$B$=$l0J30(B
  else
    bitmap=$(( mask&~(left|down|right) )); # $B%/%$!<%s$,G[CV2DG=$J0LCV$rI=$9(B
    while (( bitmap ));do
      bit=$((-bitmap&bitmap)) ;       # $B0lHV1&$N%S%C%H$r<h$j=P$9(B
      bitmap=$((bitmap&~bit)) ;       # $BG[CV2DG=$J%Q%?!<%s$,0l$D$:$D<h$j=P$5$l$k(B
      board[$row]="$bit";             # Q$B$rG[CV(B
      bitmap_R "$size" "$((row+1))" "$(( (left|bit)<<1 ))" "$((down|bit))" "$(( (right|bit)>>1 ))";
    done
  fi
}
# $BHs:F5"HG%S%C%H%^%C%W(B
# time bitmap_NR 5 0;
#
# $B:F5"HG%S%C%H%^%C%W(B
 time bitmap_R 5 0 0 0 0;    
 echo "$TOTAL";
#
exit;

