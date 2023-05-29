#!/usr/bin/bash

declare -i COUNT2=0;
declare -i TOTAL=0;
declare -a board;
declare -i DISPLAY=0;   # ボード出力するか
#
: 'ボードレイアウトを出力 ビットマップ対応版';
function printRecord()
{
  ((TOTAL++));
  size="$1";
  flag="$2"; # bitmap版は1 それ以外は 0
  echo "$TOTAL";
  sEcho=" ";  
  : 'ビットマップ版
     ビットマップ版からは、左から数えます
     上下反転左右対称なので、これまでの上から数える手法と
     rowを下にたどって左から数える方法と解の数に変わりはありません。
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
    local -i i=0;
    local -i j=0;
    for ((i=0;i<size;i++));do
      for ((j=0;j<size;j++));do
       if (( board[i]&1<<j ));then
          sEcho="${sEcho}$((j)) ";
       fi 
      done
    done
  else 
  : 'ビットマップ版以外
     (ブルートフォース、バックトラック、配置フラグ)
     上から数えます
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
    local -i i=0;
    for((i=0;i<size;i++)){
      sEcho="${sEcho}${board[i]} ";
    }
  fi
  echo "$sEcho";
  echo -n "+";
  local -i i=0;
  for((i=0;i<size;i++)){
    echo -n "-";
    if((i<(size-1)));then
      echo -n "+";
    fi
  }
  echo "+";
  local -i i=0;
  local -i j=0;
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
    local -i j=0;
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
  local -i i=0;
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
#
: '非再帰版ミラーロジック';
function solve_NR()
{
  local -i size="$1";
  local -i row="$2";
  local -i mask="$(( (1<<size)-1 ))";
  local -a bitmap[$size];
  local -a left[$size];
  local -a down[$size];
  local -a right[$size];
  local -i bit=0;
  left[$row]="$3";
  down[$row]="$4";
  right[$row]="$5";
  bitmap[$row]=$(( mask&~(left[row]|down[row]|right[row]) ));
  while ((row>0));do
    if (( bitmap[row]>0 ));then
      bit=$(( -bitmap[row]&bitmap[row] ));  # 一番右のビットを取り出す
      bitmap[$row]=$(( bitmap[row]^bit ));  # 配置可能なパターンが一つずつ取り出される
      board[$row]="$bit";                   # Qを配置
      if (( row==(size-1) ));then
        ((COUNT2++));
        printRecord "$size" "1";            # 出力 1:bitmap版 0:それ以外
        ((row--));
      else
        local -i n=$((row++));
        left[$row]=$(((left[n]|bit)<<1));
        down[$row]=$(((down[n]|bit)));
        right[$row]=$(((right[n]|bit)>>1));
        board[$row]="$bit";                 # Qを配置
        # クイーンが配置可能な位置を表す
        bitmap[$row]=$(( mask&~(left[row]|down[row]|right[row]) ));
      fi
    else
      ((row--));
    fi
  done
}
#
: '
 偶数、奇数ともに右半分にクイーンを置く 
 00001111

 奇数の場合
 奇数は中央にもクイーンを置く
 00100
 １行目の左側半分にクイーンを置けないようにする
 11100

 1行目にクイーンが中央に置かれた場合は
 00100
 2行目の左側半分にクイーンを置けない
 00100
 11100
 さらに1行目のdown,rightもクイーンを置けないので(size/2)-1となる
 11100

 偶数の場合
 １行目の左側半分にクイーンを置けないようにする
 1100
';
: '非再帰版ミラー';
function mirror_NR()
{
  local -i size="$1";
  local -i mask="$(( (1<<size)-1 ))";
  local -i bit=0; 
  for ((i=0;i<size/2;i++));do         # 奇数でも偶数でも通過
    bit="$(( 1<<i ))";
    board[0]="$bit";                  # １行目にQを置く
    solve_NR "$size" "1" "$((bit<<1))" "$bit" "$((bit>>1))";
  done
  if ((size%2));then                  # 奇数で通過
    bit=$(( 1<<(size-1)/2 ));
    board[0]=$(( 1<<((size-1)/2) ));  # １行目の中央にQを配置
    local -i left=$(( bit<<1 ));
    local -i down=$(( bit ));
    local -i right=$(( bit>>1 ));
    local -i limit="$(( size%2 ? size/2-1 : size/2 ))";
    for ((i=0;i<limit;i++));do
      bit="$(( 1<<i ))";
      solve_NR "$size" "2" "$(( (left|bit)<<1 ))" "$(( down|bit ))" "$(( (right|bit)>>1))";
    done
  fi
  TOTAL="$(( COUNT2<<1 ))";     # 倍にする
}
#
: '再帰版ミラーロジック';
function solve_R()
{
  local -i size="$1";
  local -i row="$2";
  local -i left="$3";
  local -i down="$4";
  local -i right="$5";
  local -i mask="$(( (1<<size)-1 ))";
  local -i bit;
  local -i bitmap;
  if (( row==size ));then
    ((COUNT2++));
    printRecord "$size" "1";       # 出力 1:bitmap版 0:それ以外
  else
    # Qが配置可能な位置を表す
    bitmap="$(( mask&~(left|down|right) ))";
    while ((bitmap));do
      bit="$(( -bitmap&bitmap ))"; # 一番右のビットを取り出す
      bitmap="$(( bitmap^bit ))";  # 配置可能なパターンが一つずつ取り出される
      board["$row"]="$bit";        # Qを配置
      solve_R "$size" "$((row+1))" "$(( (left|bit)<<1 ))" "$((down|bit))" "$(( (right|bit)>>1 ))";
    done
  fi
}
: '再帰版ミラー';
function mirror_R()
{
  local -i size="$1";
  local -i mask="$(( (1<<size)-1 ))";
  local -i bit=0; 
  for ((i=0;i<size/2;i++));do         # 奇数でも偶数でも通過
    bit="$(( 1<<i ))";
    board[0]="$bit";                  # １行目にQを置く
    solve_R "$size" "1" "$((bit<<1))" "$bit" "$((bit>>1))";
  done
  if ((size%2));then                  # 奇数で通過
    bit=$(( 1<<(size-1)/2 ));
    board[0]=$(( 1<<((size-1)/2) ));  # １行目の中央にQを配置
    local -i left=$(( bit<<1 ));
    local -i down=$(( bit ));
    local -i right=$(( bit>>1 ));
    local -i limit="$(( size%2 ? size/2-1 : size/2 ))";
    for ((i=0;i<limit;i++));do
      bit="$(( 1<<i ))";
      solve_R "$size" "2" "$(( (left|bit)<<1 ))" "$(( down|bit ))" "$(( (right|bit)>>1))";
    done
  fi
  TOTAL="$(( COUNT2<<1 ))";     # 倍にする
}
#
declare -i size=5;
#
# 再帰版ミラー  
mirror_R "$size";
#
# 非再帰版ミラー  
# mirror_NR "$size";
# 出力
echo "size: $size" "TOTAL:$TOTAL COUNT2:$COUNT2";

