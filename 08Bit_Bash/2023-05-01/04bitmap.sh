#!/usr/bin/bash

declare -i COUNT2=0;
declare -i TOTAL=0;     # カウンター
declare -i UNIQUE=0;    # ユニークユーザー
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
: '非再帰版ミラーロジック';
function mirror_solve_NR()
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
  while ((row>-1));do
    if (( bitmap[row]>0 ));then
      bit=$(( -bitmap[row]&bitmap[row] ));  # 一番右のビットを取り出す
      bitmap[$row]=$(( bitmap[row]^bit ));  # 配置可能なパターンが一つずつ取り出される
      board[$row]="$bit";                   # Qを配置
      if (( row==(size-1) ));then
        ((COUNT2++));
        if ((DISPLAY==1));then
          printRecord "$size" "1";            # 出力 1:bitmap版 0:それ以外
        fi
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
: '非再帰版ミラー';
function mirror_NR()
{
  local -i size="$1";
  local -i mask="$(( (1<<size)-1 ))";
  local -i bit=0; 
  : '
    if ((size%2));then                #  以下のif文と等価です
      limit="$((size/2-1))";
    else
      limit="$((size/2))";
    fi
  ';
  local -i limit="$(( size%2 ? size/2-1 : size/2 ))";
  for ((i=0;i<size/2;i++));do         # 奇数でも偶数でも通過
    bit="$(( 1<<i ))";
    board[0]="$bit";                  # １行目にQを置く
    mirror_solve_NR "$size" "1" "$((bit<<1))" "$bit" "$((bit>>1))";
  done
  if ((size%2));then                  # 奇数で通過
    bit=$(( 1<<(size-1)/2 ));
    board[0]=$(( 1<<((size-1)/2) ));  # １行目の中央にQを配置
    local -i left=$(( bit<<1 ));
    local -i down=$(( bit ));
    local -i right=$(( bit>>1 ));
    for ((i=0;i<limit;i++));do
      bit="$(( 1<<i ))";
      mirror_solve_NR "$size" "2" "$(( (left|bit)<<1 ))" "$(( down|bit ))" "$(( (right|bit)>>1))";
    done
  fi
  TOTAL="$(( COUNT2<<1 ))";     # 倍にする

}
#
: '再帰版ミラーロジック';
function mirror_solve_R()
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
    if ((DISPLAY));then
      printRecord "$size" "1";       # 出力 1:bitmap版 0:それ以外
    fi
  else
    # Qが配置可能な位置を表す
    bitmap="$(( mask&~(left|down|right) ))";
    while ((bitmap));do
      bit="$(( -bitmap&bitmap ))"; # 一番右のビットを取り出す
      bitmap="$(( bitmap^bit ))";  # 配置可能なパターンが一つずつ取り出される
      board["$row"]="$bit";        # Qを配置
      mirror_solve_R "$size" "$((row+1))" "$(( (left|bit)<<1 ))" "$((down|bit))" "$(( (right|bit)>>1 ))";
    done
  fi
}
#
: '再帰版ミラー';
function mirror_R()
{
  local -i size="$1";
  local -i mask="$(( (1<<size)-1 ))";
  local -i bit=0; 
  : '
    if ((size%2));then                #  以下のif文と等価です
      limit="$((size/2-1))";
    else
      limit="$((size/2))";
    fi
  ';
  local -i limit="$(( size%2 ? size/2-1 : size/2 ))";
  for ((i=0;i<size/2;i++));do         # 奇数でも偶数でも通過
    bit="$(( 1<<i ))";
    board[0]="$bit";                  # １行目にQを置く
    mirror_solve_R "$size" "1" "$((bit<<1))" "$bit" "$((bit>>1))";
  done
  if ((size%2));then                  # 奇数で通過
    bit=$(( 1<<(size-1)/2 ));
    board[0]=$(( 1<<((size-1)/2) ));  # １行目の中央にQを配置
    local -i left=$(( bit<<1 ));
    local -i down=$(( bit ));
    local -i right=$(( bit>>1 ));
    for ((i=0;i<limit;i++));do
      bit="$(( 1<<i ))";
      mirror_solve_R "$size" "2" "$(( (left|bit)<<1 ))" "$(( down|bit ))" "$(( (right|bit)>>1))";
    done
  fi
  TOTAL="$(( COUNT2<<1 ))";     # 倍にする
}
#
: '非再帰版ビットマップ';
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
  bitmap[$row]=$(( mask&~(left[row]|down[row]|right[row]) ));
  while ((row>-1));do
    if (( bitmap[row]>0 ));then
      bit=$(( -bitmap[row]&bitmap[row] ));  # 一番右のビットを取り出す
      bitmap[$row]=$(( bitmap[row]^bit ));  # 配置可能なパターンが一つずつ取り出される
      board[$row]="$bit";                   # Qを配置
      if (( row==(size-1) ));then
        ((TOTAL++));
        if ((DISPLAY==1));then
          printRecord "$size" "1";            # 出力 1:bitmap版 0:それ以外
        fi
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
: '再帰版ビットマップ';
function bitmap_R()
{
  local -i size="$1"; 
  local -i row="$2";
  local -i mask="$3";
  local -i left="$4"; 
  local -i down="$5"; 
  local -i right="$6";
  local -i bitmap=;
  local -i bit=;
  local -i col=0;                     # 再帰に必要
  if (( row==size ));then
    ((TOTAL++));
    if ((DISPLAY==1));then
      printRecord "$size" "1";         # 出力 1:bitmap版 0:それ以外
    fi
  else
    bitmap=$(( mask&~(left|down|right) )); # クイーンが配置可能な位置を表す
    while (( bitmap ));do
      bit=$((-bitmap&bitmap)) ;       # 一番右のビットを取り出す
      bitmap=$((bitmap&~bit)) ;       # 配置可能なパターンが一つずつ取り出される
      board[$row]="$bit";             # Qを配置
      bitmap_R "$size" "$((row+1))" "$mask" "$(( (left|bit)<<1 ))" "$((down|bit))" "$(( (right|bit)>>1 ))";
    done
  fi
}
#
: '非再帰版配置フラグ(right/down/left flag)';
function postFlag_NR()
{
  local -i size="$1";
  local -i row="$2"
  local -i matched=0;
  for ((i=0;i<size;i++)){ board[$i]=-1; }
  while ((row>-1));do
    matched=0;
    for ((col=board[row]+1;col<size;col++)){
      if (( !down[col]
        &&  !right[col-row+size-1]
        &&  !left[col+row] ));then
        dix=$col;
        rix=$((row-col+(size-1)));
        lix=$((row+col));
        if ((board[row]!=-1));then
          down[${board[$row]}]=0;
          right[${board[$row]}-$row+($size-1)]=0;
          left[${board[$row]}+$row]=0;
        fi       
        board[$row]=$col;   # Qを配置
        down[$col]=1;
        right[$col-$row+($size-1)]=1;
        left[$col+$row]=1;  # 効き筋とする
        matched=1;          # 配置した
        break;
      fi
    }
    if ((matched));then     # 配置済み
      ((row++));            #次のrowへ
      if ((row==size));then
        ((TOTAL++));
        if ((DISPLAY==1));then
          printRecord "$size";# 出力
        fi
        ((row--));
      fi
    else
      if ((board[row]!=-1));then
        down[${board[$row]}]=0;
        right[${board[$row]}-$row+($size-1)]=0;
        left[${board[$row]}+$row]=0;
        board[$row]=-1;
      fi
      ((row--));            # バックトラック
    fi
  done
}
#
: '再帰版配置フラグ';
function postFlag_R()
{
  local -i size="$1";
  local -i row="$2";
  local -i col=0;       # 再帰に必要
  if (( row==size ));then
     ((TOTAL++));
    if (( DISPLAY==1 ));then
      printRecord "$size";# 出力
    fi
  else
    for(( col=0;col<size;col++ )){
      board[$row]="$col";
      if (( down[col]==0 
        && right[row-col+size-1]==0
        && left[row+col]==0));then
        down[$col]=1;
        right[$row-$col+($size-1)]=1;
        left[$row+$col]=1;
        postFlag_R "$size" "$((row+1))";
        down[$col]=0;
        right[$row-$col+($size-1)]=0;
        left[$row+$col]=0;
      fi
    }
  fi
}
#
: 'バックトラック版効き筋をチェック';
function check_backTracking()
{
  local -i row="$1";
  local -i flag=0;
  for ((i=0;i<row;++i)){
    if (( board[i]>=board[row] ));then
      val=$(( board[i]-board[row] ));
    else
      val=$(( board[row]-board[i] ));
    fi
    if (( board[i]==board[row] || val==(row-i) ));then
      flag=0;
      return ;
    fi
  }
  flag=1;
  [[ $flag -eq 0 ]]
  return $?;
}
#
: '非再帰版バックトラック';
function backTracking_NR()
{
  local -i size="$1";
  local -i row="$2";
  for ((i=0;i<size;i++)){ board[$i]=-1; }
  while ((row>-1));do
    local -i matched=0;
    local -i col=0;  
    for((col=board[row]+1;col<size;col++)){
      board[$row]=$col;
      check_backTracking "$row";  # 効きをチェック
      if (($?==1));then # 直前のreturnを利用
        matched=1;
        break;
      fi
    }
    if ((matched));then
      ((row++));
      if ((row==size));then  # 最下部まで到達
        ((row--));
        ((TOTAL++));
        if (( DISPLAY==1 ));then
          printRecord "$size";# 出力
        fi
      fi
    else
      if ((board[row]!=-1));then
        board[$row]=-1;
      fi
      ((row--));
    fi
 done  
}
#
: '再帰版バックトラック';
function backTracking_R()
{
  local -i size="$1";
  local -i row="$2";
  local -i col=0;
  if ((row==size));then
    ((TOTAL++));
    if (( DISPLAY==1 ));then
      printRecord "$size";# 出力
    fi
  else
    for(( col=0;col<size;col++ )){
      board["$row"]="$col";
      check_backTracking "$row";
      if (($?==1));then 
        backTracking_R  $size $((row+1));
      fi
    }
  fi
}
#
: 'ブルートフォース版効き筋をチェック';
function check_bluteForce()
{
  local -i size="$1";
  local -i flag=1;
  for ((r=1;r<size;++r)){
    for ((i=0;i<r;++i)){
      #echo `$(($1-$2)) | sed -e "s/^-//g"`;
      if (( board[i]>=board[r] ));then
        val=$(( board[i]-board[r] ));
      else
        val=$(( board[r]-board[i] ));
      fi

      if (( board[i]==board[r] || val==(r-i) ));then
        flag=0; 
        return ;
      fi
    }
  }
  flag=1;
  [[ $flag -eq 0 ]]
  return $?;
}
#
: '非再帰版ブルートフォース';
function bluteForce_NR()
{
  local -i size="$1";
  local -i row="$2";
  for ((i=0;i<size;i++)){ board[$i]=-1; }
  while ((row>-1));do
    local -i matched=0;
    local -i col=0;  
    for((col=board[row]+1;col<size;col++)){
      board[$row]=$col;
      matched=1;
      break;
    }
    if ((matched));then
      ((row++));
      if ((row==size));then  # 最下部まで到達
        ((row--));
        check_bluteForce "$size";  # 効きをチェック
        if (($?==1));then # 直前のreturnを利用
          ((TOTAL++));
          if (( DISPLAY==1 ));then
            printRecord "$size";# 出力
          fi
        fi
      fi
    else
      if ((board[row]!=-1));then
        board[$row]=-1;
      fi
      ((row--));
    fi
 done  
}
#
: '再帰版ブルートフォース';
function bluteForce_R()
{
  local -i size="$1";
  local -i row="$2";
  local -i col=;
  if ((row==size));then
    check_bluteForce "$size";
    if (( $?==1 ));then 
      ((TOTAL++));
      if (( DISPLAY==1 ));then
        printRecord "$size";# 出力
      fi
    fi
  else
    #for(( col=0;col<(size-row);col++ )){
    for(( col=0;col<size;col++ )){
      board["$row"]="$col";
      bluteForce_R  $size $((row+1));
    }
  fi
}
#
function NQ()
{
  local selectName="$1";
  local -i max=15;
  local -i min=4;
  local -i N="$min";
  local -i mask=0;
  local -i bit=0
  local -i row=0;
  local startTime=0;
	local endTime=0;
	local hh=mm=ss=0; 
  echo " N:        Total       Unique        hh:mm:ss" ;
  local -i N;
  for((N=min;N<=max;N++)){
    TOTAL=0; UNIQUE=0; COUNT2=0; row=0;
    mask=$(( (1<<N)-1 ));
    startTime=$(date +%s);# 計測開始時間

    "$selectName" "$N" "$row" "$mask" 0 0 0;

    endTime=$(date +%s); 	# 計測終了時間
    ss=$((endTime-startTime));# hh:mm:ss 形式に変換
    hh=$((ss/3600));
    ss=$((ss%3600));
    mm=$((ss/60));
    ss=$((ss%60));
    printf "%2d:%13d%13d%10d:%.2d:%.2d\n" $N $TOTAL $UNIQUE $hh $mm $ss ;
  } 
}

while :
do
read -n1 -p "
エイト・クイーン メニュー
実行したい番号を選択
5) ミラー
4) ビットマップ
3) 配置フラグ 
2) バックトラック 
1) ブルートフォース 

echo "行頭の番号を入力してください";

" selectNo;
echo 
case "$selectNo" in
  5)
    while :
    do 
      read -n1 -p "
      y|Y) ボード画面表示をする
      n|N) ボード画面表示をしない
      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    while :
    do 
      read -n1 -p "
      y|Y) 再帰
      n|N) 非再帰
      " select;
      echo; 
      case "$select" in
        y|Y) NQ mirror_R; break; ;;
        n|N) NQ mirror_NR; break; ;;
      esac
    done
    ;;
  4)
    while :
    do 
      read -n1 -p "
      y|Y) ボード画面表示をする
      n|N) ボード画面表示をしない
      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    while :
    do 
      read -n1 -p "
      y|Y) 再帰
      n|N) 非再帰
      " select;
      echo; 
      case "$select" in
        y|Y) NQ bitmap_R; break; ;;
        n|N) NQ bitmap_NR; break; ;;
      esac
    done
    ;;
  3)
    while :
    do 
      read -n1 -p "
      y|Y) ボード画面表示をする
      n|N) ボード画面表示をしない
      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    while :
    do 
      read -n1 -p "
      y|Y) 再帰
      n|N) 非再帰
      " select;
      echo; 
      case "$select" in
        y|Y) NQ postFlag_R; break; ;;
        n|N) NQ postFlag_NR; break; ;;
      esac
    done
    ;;
  2)
    while :
    do 
      read -n1 -p "
      y|Y) ボード画面表示をする
      n|N) ボード画面表示をしない

      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    while :
    do 
      read -n1 -p "
      y|Y) 再帰
      n|N) 非再帰

      " select;
      echo; 
      case "$select" in
    
        y|Y) NQ backTracking_R; break; ;;
        n|N) NQ backTracking_NR; break; ;;
      esac
    done
    ;;
  1)
    while :
    do 
      read -n1 -p "
      y|Y) ボード画面表示をする
      n|N) ボード画面表示をしない

      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    while :
    do 
      read -n1 -p "
      y|Y) 再帰
      n|N) 非再帰

      " select;
      echo; 
      case "$select" in
        y|Y) NQ bluteForce_R; break; ;;
        n|N) NQ bluteForce_NR;break; ;;
      esac
    done
    ;;
  *)
    ;; 
esac
done
exit;
