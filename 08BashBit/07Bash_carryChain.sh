#!/usr/bin/bash

declare -i size;
declare -i TOTAL=0;
declare -i UNIQUE=0;
declare -i COUNT2=0;
declare -i COUNT4=1;
declare -i COUNT8=2;
declare -A cnt;
declare -A board=([row]=0 [bh]=0 [bu]=0 [bd]=0 [left]=0 [down]=0 [right]=0);
declare -a pres_a;
declare -a pres_b;
declare -a B_x;
#
: '';
function process()
{
  #cnt[$sym]+=$( solve_nqueenr "board[row]>>2" "board[left]>>4" "((((board[down]>>2)|(~0<<(size-4)))+1)<<(size-5))-1" "(board[right]>>4)<<(size-5)" ) ;
  #cnt[$sym];
  # `solve_nqueenr "board[row]>>2" "board[left]>>4" "((((board[down]>>2)|(~0<<(size-4)))+1)<<(size-5))-1" "(board[right]>>4)<<(size-5)"`
  :
}
#
: '';
function solve_nqueenr()
{
  local -i row="$1";
  local -i left="$2";
  local -i down="$3";
  local -i right="$4";
  if (( down+1==0 ));then 
    return 1; 
  fi
  while (( (row&1)!=0 ));do
    ((row>>=1))
    ((left<<=1));
    ((right>>=1));
  done
  ((row>>=1));
  local -i total;
  local -i bit=0;
  local -i bitmap=0;
  for (( bitmap=~(left|down|right);bitmap!=0;bitmap^=bit ));do
    (( bit=bitmap&-bitmap ));
    #(( total+=solve_nqueenr "$row" "(left|bit)<<1" "(down|bit)" "(right|bit)>>1" )); 
  #
  ## ここが無限ループしている！
  #
  solve_nqueenr "$row" "(left|bit)<<1" "(down|bit)" "(right|bit)>>1"  ; 
  echo $?
  done
  echo "$total";
  #return "$total";
}
#
: 'クイーンの効きをチェック';
function board_placement()
{
  local -i dimx="$1";
  local -i dimy="$2";
  if (( B_x[$dimx]==$dimy ));then 
    return 1;
  fi
  B_x[$dimx]="$dimy";
  ((row=1<<dimx));
  ((down=1<<dimy));
  ((left=1<<(size-1-dimx+dimy)));
  ((right=1<<(dimx+dimy)));
  if (( (B[row]&row)||
        (B[down]&down)||
        (B[left]&left)||
        (B[right]&right) ));then
    return 0;
  fi 
  ((B[row]|=row));
  ((B[down]|=down));
  ((B[left]|=left));
  ((B[right]|=right));
  return 1;
}
: 'キャリーチェーン';
function carryChain()
{
  local -i idx=0;
  local -i a=0;
  local -i b=0;
  for ((a=0;a<size;a++));do
    for ((b=0;b<size;b++));do
      if (( ((a>=b)&&(a-b)<=1)||((b>a)&&(b-a)<=1) ));then
        continue;
      fi
      pres_a[$idx]="$a";
      pres_b[$idx]="$b";
      (( idx++ ));
    done
  done
  #
  # 90度回転
  local -A wB=( 
    [row]=0 
    [left]=0 
    [down]=0 
    [right]=0
  );
  # B=( $wB[@] );
  declare -A B;
  for i in ${!wB[@]};do B[$i]=${wB[$i]} ; done
  # 1
  for ((w=0;w<=(size/2)*(size-3);w++));do
    # B=( $wB[@] );
    for i in ${!wB[@]};do B[$i]=${wB[$i]} ; done
    #
    # B構造体の初期化
    B[row]=0;
    B[down]=0;
    B[left]=0;
    B[right]=0;
    # board配列の初期化
    for ((i=0;i<size;i++));do B_x[$row]=-1; done
    #
    # Qを配置
    board_placement "0" "$((pres_a[w]))" ; #１行目
    echo "$?";
    board_placement "1" "$((pres_b[w]))" ; #２行目
    #
    # 90度回転
    local -A nB;
    #nB=( ${B[@]} );
    for i in ${!B[@]};do nB[$i]=${B[$i]}; done
    # 1
    for ((n=w;n<(size-2)*(size-1)-w;n++));do 
      #B=( $nB[@] );
      for i in ${!nB[@]};do B[$i]=${nB[$i]}; done
      #
      # Qを配置
      board_placement "$((pres_a[n]))" "$((size-1))"; 
      if (( $?==0 ));then continue; fi
      board_placement "$((pres_b[n]))" "$((size-2))";
      if (( $?==0 ));then continue; fi
      #
      # 90度回転
      local -A eB;
      #eB=( ${B[@]} );
      for i in ${!B[@]};do eB[$i]=${B[$i]}; done
      # 3
      for ((e=w;e<(size-2)*(size-1)-w;e++));do 
        #B=( ${eB[@]} );
        for i in ${!eB[@]};do B[$i]=${eB[$i]}; done
        board_placement "$((size-1))" "$((size-1-pres_a[e]))"; 
        if (( $?==0 ));then continue; fi
        board_placement "$((size-2))" "$((size-1-pres_b[e]))"; 
        if (( $?==0 ));then continue; fi
        #
        # 90度回転
        local -A sB;
        #sB=( ${B[@]} );
        for i in ${!B[@]};do sB[$i]=${B[$i]}; done
        # 4
        for ((s=w;s<(size-2)*(size-1)-w;s++));do
          # B=( ${sB[@]} );
          for i in ${!sB[@]};do B[$i]=${sB[$i]}; done
          board_placement "$((size-1-pres_a[s]))" "0";
          if (( $?==0 ));then continue; fi
          board_placement "$((size-1-pres_b[s]))" "1"; 
          if (( $?==1));then continue; fi
          #
          # 対象解除法
          #
          local -i ww=$(( (size-2)*(size-1)-1-w ));
          local -i w2=$(( (size-2)*(size-1)-1 ));
          if (( (s==ww)&&(n<(w2-e)) ));then continue; fi
          if (( (e==ww)&&(n>(w2-n)) ));then continue; fi
          if (( (n==ww)&&(e>(w2-s)) ));then continue; fi
          if ((s==w));then
            if(( (n!=w)||(e!=w) ));then continue; fi
            cnt[$COUNT2]+=$( solve_nqueenr "board[row]>>2" "board[left]>>4" "((((board[down]>>2)|(~0<<(size-4)))+1)<<(size-5))-1" "(board[right]>>4)<<(size-5)" );
            continue;
          fi
          if (( (e==w)&&(n>=s) ));then
            if(( n>s));then continue; fi
            cnt[$COUNT4]+=$( solve_nqueenr "board[row]>>2" "board[left]>>4" "((((board[down]>>2)|(~0<<(size-4)))+1)<<(size-5))-1" "(board[right]>>4)<<(size-5)" );
            continue;
          fi
          cnt[$COUNT8]=+=$( solve_nqueenr "board[row]>>2" "board[left]>>4" "((((board[down]>>2)|(~0<<(size-4)))+1)<<(size-5))-1" "(board[right]>>4)<<(size-5)" );
          continue;
        done
      done
    done
  done
  UNIQUE=$((cnt[$COUNT2]+cnt[COUNT4]+cnt[COUNT8]));
  TOTAL=$((cnt[$COUNT2]*2+cnt[COUNT4]*4+cnt[COUNT8]*8));
}
#
size=5;
carryChain "$size";
echo "size:$size TOTAL:$TOTAL UNIQUE:$UNIQUE";
echo "COUNT2:$COUNT2 COUNT4:$COUNT4 COUNT8:$COUNT8";
exit;

