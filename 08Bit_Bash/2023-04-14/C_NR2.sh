declare -i TOTAL=0;
declare -i UNIQUE=0;

function srch(){  
  local -i size="$1";
  local -i row="$2";
  local -i bitmap;
  local -i bit;
  local -a board[$size];
  local -a left[$size];
  local -a down[$size];
  local -a right[$size];
  for ((i=0;i<size;i++));do
    board[$i]=-1; 
    left[$size]=0;
    down[$size]=0;
    right[$size]=0;
  done 
  # Cではこう書かれている
  # tmpBoard=board+1;
  # tmpBoard=board[1];
  tRow=1;
  #board[$(( 0 +1))]  = board[$(( 0 +tRow))]
  #board[$(( 1 +1))]  = board[$(( 1 +tRow))]
  #board[$(( 2 +1))]  = board[$(( 2 +tRow))]
  while true ;do
    echo "tRow:$tRow";
    if((bitmap));then
      bit=$((-bitmap&bitmap)); 
      bitmap=$(( bitmap^bit ));
      if (( row==(size-1) ));then
        ((TOTAL++));
        #bitmap=*--tRow;      # ☆
        ((tRow--)); 
        bitmap=$(( board[row+tRow] ));
        ##-----
        ((row--));
        continue;
      else
        local -i n=$((row++));
        left[$row]=$(((  left[n]|bit)<<1 ));
        down[$row]=$((   down[n]|bit ));
        right[$row]=$(((right[n]|bit)>>1 ));
        #*tRow++=bitmap;       # ☆
        board[$row+$tRow]=$bitmap;
        ((tRow++));
        ##-----
        bitmap=$(( mask&~(left[row]|down[row]|right[row]) ));
        continue;
      fi
    else
      if (( board[row]!=-1 ));then
        board[$row]=-1;
      fi
      #bitmap=*--pnStack;         # ☆
      ((--tRow));
      bitmap=$(( board[row+tRow] ));
      if (( board[row+tRow]==board[row]));then
        break ;
      fi
      ##-----
      (( row-- ));
      continue
    fi
  done 
}
#
mask=$(( (1<<size)-1 ));
srch 5 0;
echo "$TOTAL";
