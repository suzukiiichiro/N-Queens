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
: 'ブルートフォース版効き筋をチェック';
function check_bluteForce(){
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
: '再帰版ブルートフォース';
function bluteForce_R(){
  local -i row="$1";
  local -i size="$2";
  local -i col=;
  if ((row==size));then
    check_bluteForce "$size";
    if (( $?==1 ));then 
      ((TOTAL++));
      printRecord "$size";   # 出力しないならコメント
    fi
  else
    for(( col=0;col<size;col++ )){
      board["$row"]="$col";
      bluteForce_R $((row+1)) $size ;
    }
  fi
}

# 非再帰版ブルートフォース
# time bluteForce_NR 0 5;
#
# 再帰版ブルートフォース
time bluteForce_R 0 5;
#
exit;

