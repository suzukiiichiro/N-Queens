#!/usr/bin/bash
declare -i size;
declare -i TOTAL=0;
declare -i UNIQUE=0;
declare -i COUNT2=0;
declare -i COUNT4=1;
declare -i COUNT8=2;
declare -A cnt;
declare -A pre;
declare -A board=([x]=0 [y]=0 [row]=0 [bh]=0 [bu]=0 [bd]=0 [left]=0 [down]=0 [right]=0);
declare -A B=( ${board[@]} );
#
: '';
function solve_nqueenr()
{
  local -i row="$1";
  local -i left="$2";
  local -i down="$3";
  local -i right="$4";
  
  if (( down+1==0 ));then return 1; fi
  while (( row&1!=0 ));do
    row=$((row>>1));
    left=$(( left<<1 ));
    right=$(( right>>1 ));
  done
  row=$((row>>1));
  local -i s=0;
  local -i bit=0;
  for (( bitmap=~(left|down|right);bitmap!=0;bitmap^=bit ));do
    bit=$(( bitmap&-bitmap ));
    s=$(( s+solve_nqueenr row (left|bit)<<1 (down|bit) (right|bit)>>1 )); 
  done
  return $s;
}
#
# : 'クイーンの効きをチェック';
function board_placement()
{
  size="$1";
  x="$2";
  y="$3";
  local -i flag=1;
  if (( board[x]==y ));then 
    flag=1;
  fi
  board[$x]="$y";

  local -i row=$(( 1<<x ));
  local -i down=$(( 1<<y ));
  local -i left=$(( 1<<(size-1-x+y) ));
  local -i right=$(( 1<<(x+y) ));
  if (( (board[row]&row)||
        (board[down]&down)||
        (board[left]&left)||
        (board[right]&right) ));then
    flag=0; 
  fi 
  board["row"]=$(( board[row]|row   ));
  board["down"]=$(( board[down]|down ));
  board["left"]=$(( board[left]|left ));
  board["right"]=$(( board[right]|right ));

  [[ $flag -eq 0 ]]
  return $?;
}

declare -a pres_a;
declare -a pres_b;
# チェーンの初期化
function makeChain()
{
  size="$1";
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
}
# : 'キャリーチェーン';
function carryChain()
{
  size="$1";
  makeChain "$size"
  read -p "pres_a: ${pres_a[@]}";
  exit; 

  # 90度回転
  local -A wB=( $B[@] ); 
  for ((w=0;w<=(size/2)*(size-3);w++ ));do
    B=( $wB[@] );
    B[row]=B[down]=B[left]=B[right]=0;
    # board配列の初期化
    for ((row=0;row<size;row++));do board[$row]=-1; done
    #
    board_placement "$size" 0 "$((pres_a[w]))" ; #１行目
    board_placement "$size" 1 "$((pres_b[w]))" ; #２行目
    #
    # 90度回転
    local -A nB=( ${B[@]} );
    for ((n=w;n<(size-2)*(size-1)-w;n++));do 
      B=( ${nB[@]} );
      board_placement "$size" $((pres_a[n])) "$((size-1))"; #１行目
      if (( $?==0 ));then continue; fi
      board_placement "$size" $((pres_b[n]))" "$((size-2))"; #２行目
      if (( $?==0 ));then continue; fi
      # 90度回転
      local -A eB=( ${B[@]} );
      for ((e=w;e<(size-2)*(size-1)-n;e++));do 
        B=( ${eB[@]} );
        board_placement "$size" "$((size-1))" "$((size-1-pres_a[e]))"; 
        if (( $?==0 ));then continue; fi
        board_placement "$size" "$((size-2))" "$((size-1-pres_b[e]))"; 
        if (( $?==0 ));then continue; fi
        # 90度回転
        local -A sB=( ${B[@]} );
        for ((s=w;s<(size-2)*(size-1)-e;s++));do
          B=( ${sB[@]} );
          board_placement "$size" "$((size-1-pres_a[s]))" 0;
          if (( $?==0 ));then continue; fi
          board_placement "$size" "$((size-1-pres_b[s]))" 1; 
          if (( $?==1));then continue; fi
          # 対象解除法
          local -i ww=$(( (size-2)*(size-1)-1-2 ));
          local -i w2=$(( (size-2)*(size-1)-1 ));
          if (( (s==ww)&&(n<(w2-e)) ));then continue; fi
          if (( (e==ww)&&(n>(w2-n)) ));then continue; fi
          if (( (n==ww)&&(e>(w2-s)) ));then continue; fi
          if ((s==w));then
            if(( n!=w || e!=w ));then continue; fi
            #process "$size" COUNT2 ; continue;
            (( pre[COUNT2]++ ));
            cnt[COUNT2]=$(( cnt[COUNT2] + solve_nqueenr "baord[row]>>2" "board[left]>>4" "(((board[down]>>2|(~0<<size-4))+1)<<size-5)-1" "(baord[right]>>4)<<size-5" )) ;
          fi
          if (( (e==w)&&(n>=w) ));then
            if(( n>s));then continue; fi
            #process "$size" COUNT4 ; continue;
            (( pre[COUNT4]++ ));
            cnt[COUNT4]=$(( cnt[COUNT4]+ solve_nqueenr "baord[row]>>2" "board[left]>>4" "(((board[down]>>2|(~0<<size-4))+1)<<size-5)-1" "(baord[right]>>4)<<size-5" )) ;
          fi
          #process "$size" COUNT8 ;
          (( pre[COUNT8]++ ));
          cnt[COUNT8]=$(( cnt[COUNT8]+ solve_nqueenr "baord[row]>>2" "board[left]>>4" "(((board[down]>>2|(~0<<size-4))+1)<<size-5)-1" "(baord[right]>>4)<<size-5" )) ;
          continue;
        done
      done
    done
  done
  UNIQUE=$((cnt[COUNT2]+cnt[COUNT4]+cnt[COUNT8]));
  TOTAL=$((cnt[COUNT2]*2+cnt[COUNT4]*4+cnt[COUNT8]*8));
}
#
size=5;
carryChain "$size";
exit;

