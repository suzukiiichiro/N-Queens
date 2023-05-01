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
: 'バックトラック版効き筋をチェック';
function check_backTracking(){
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
function backTracking_NR(){
  local -i row="$1";
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
      board[$row]=$col;     # クイーンを配置
      ##
      : '効きをチェック' ;
      ##
      check_backTracking "$row";  # 効きをチェック
      if (($?==1));then     # 直前の関数のreturnを利用
        matched=1;          # 配置した
        break;              # 配置したらクイーンを抜ける
      fi
    done
    ##
    : '４．配置したら実行したい処理';
    ##
    if ((matched));then
      ((row++));
      ##
      : '５．最下部まで到達したときの処理';
      ##
      if ((row==size));then  # 最下部まで到達
        ((row--));
        ## ブルートフォースではここで効きをチェックしていた
        # check_bluteForce "$size";   # 効きをチェック
        ##
        ((TOTAL++));         # 解をカウント
        printRecord "$size"; # 出力
      fi
    ## 
    : '６．配置できなくてバックトラックしたい時の処理';
    ## 
    else
      if ((board[row]!=-1));then
        board[$row]=-1;      # クイーンの配置を開放
      fi
      ((row--));             # バックトラック
    fi
 done  
}
#
: '再帰版バックトラック';
function backTracking_R(){
  local -i row="$1";
  local -i size="$2";
  local -i col=0;
  if ((row==size));then
    ((TOTAL++));
    printRecord "$size";   # 出力
  else
    for(( col=0;col<size;col++ )){
      board["$row"]="$col";
      check_backTracking "$row";
      if (($?==1));then 
        backTracking_R $((row+1)) $size ;
      fi
    }
  fi
}
#
# 非再帰版バックトラック
# time backTracking_NR 0 5;    
#
# 再帰版バックトラック
 time backTracking_R 0 5;    
#
exit;
