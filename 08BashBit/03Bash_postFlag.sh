#!/usr/bin/bash

declare -i TOTAL=0;     # カウンター
#
: 'ボードレイアウトを出力';
function printRecord(){
  size="$1";
  echo "$TOTAL";
  sEcho=" ";  
  for((i=0;i<size;i++)){
    sEcho="${sEcho}${board[i]} ";
  }
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
      if((i==board[j]));then
        echo -n "O";
      else
        echo -n " ";
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
: '非再帰版配置フラグ(right/down/left flag)';
function postFlag_NR(){
  local -i row="$1"
  local -i size="$2";
  local -i col=0;
  ## 
  : '１．非再帰は初期化が必要';
  for ((i=0;i<size;i++)){ board[$i]=-1; }
  ##
  : '２．再帰で呼び出される関数内を回す処理';
  ##
  while (( row>-1 ));do
    local -i matched=0;     # クイーンを配置したか
    ##
    : '３．再帰処理のループ部分';
    ##
    # 非再帰では過去の譜石を記憶するためにboard配列を使う
    for ((col=board[row]+1;col<size;col++));do
      if (( !down[col]
        &&  !right[col-row+size-1]
        &&  !left[col+row] ));then
        dix=$col;
        rix=$((row-col+(size-1)));
        lix=$((row+col));
        ## バックトラックではここで効きをチェックしていた
        # check_backTracking "$row";  # 効きをチェック
        ## 
        # 効きとしてフラグをfalseにする
        if ((board[row]!=-1));then
          down[${board[$row]}]=0;
          right[${board[$row]}-$row+($size-1)]=0;
          left[${board[$row]}+$row]=0;
        fi       
        board[$row]=$col;     # クイーンを配置
        # 効きを開放（trueに）する
        down[$col]=1;
        right[$col-$row+($size-1)]=1;
        left[$col+$row]=1;  # 効き筋とする
        matched=1;          # 配置した
        break;              # 配置したらクイーンを抜ける
      fi
    done
    ##
    : '４．配置したら実行したい処理';
    ##
    if ((matched));then     # 配置済み
      ((row++));            #次のrowへ
      ##
      : '５．最下部まで到達したときの処理';
      ##
      if ((row==size));then
        ((row--));
        ## ブルートフォースではここで効きをチェックしていた
        # check_bluteForce "$size";   # 効きをチェック
        ##
        ((TOTAL++));
        printRecord "$size";# 出力
      fi
    ## 
    : '６．配置できなくてバックトラックしたい時の処理';
    ## 
    else
      if ((board[row]!=-1));then
        down[${board[$row]}]=0;
        right[${board[$row]}-$row+($size-1)]=0;
        left[${board[$row]}+$row]=0;
        board[$row]=-1;     # クイーンの配置を開放
      fi
      ((row--));            # バックトラック
    fi
  done
}
#
#
: '再帰版配置フラグ';
function postFlag_R(){
  local -i row="$1";
  local -i size="$2";
  local -i col=0;       # 再帰に必要
  if (( row==size ));then
     ((TOTAL++));
     printRecord "$size";# 出力
  else
    for(( col=0;col<size;col++ )){
      board[$row]="$col";
      if (( down[col]==0 
        && right[row-col+size-1]==0
        && left[row+col]==0));then
        down[$col]=1;
        right[$row-$col+($size-1)]=1;
        left[$row+$col]=1;
        postFlag_R "$((row+1))" "$size" ;
        down[$col]=0;
        right[$row-$col+($size-1)]=0;
        left[$row+$col]=0;
      fi
    }
  fi
}
#
# 非再帰版配置フラグ
time postFlag_NR 0 5;    
#
# 再帰版配置フラグ
# time postFlag_R 0 5;    
#
exit;