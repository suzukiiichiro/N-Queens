#!/usr/bin/bash

declare -i TOTAL=0;     # カウンター
#
#
: 'ボードレイアウトを出力 ビットマップ対応版';
function printRecord(){
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
    for((i=0;i<size;i++)){
      for((j=0;j<size;j++)){
        if (( board[i]&1<<j ));then
          sEcho="${sEcho}$((j)) ";
        fi 
      }
    }
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
: '
特定の位置を指定して、以前の位置と衝突したかどうかを検出します。入力行は (1<<行) でエンコードする必要があります。衝突した場合は 1 を返し、そうでない場合は 0 を返します
';
#
: 'ビットマップ版効き筋をチェック';
function check_bitmap(){
  local -i col="$1";
  local -i row="$2";
  local -i mask=down=left=right=flag=0;
  for((i=0;i<col;i++));do
    down=$(( board[i] ));
    left=$(( board[i]>>(col-i)  ));
    right=$((board[i]<<(col-i)  ));
    mask=$(( mask|(down|left|right) ));
  done
  flag=$(( mask&row )); 
  [[ $flag -eq 0 ]]
  return $?;
}
#
: '非再帰ビットマップ版';
function bitmap_NR(){
  local -i size="$1";       # size=Ｎ
  for ((i=0;i<size;i++));do board[$i]=0; done
  local -i col=0;           # 列
  local -i row=1;           # 行
  while (( col<size ));do
    while (( row<(1<<size) ));do
      check_bitmap "$col" "$row";# 効きのチェック
      if (( $?==0 ));then   # 衝突していなければ
        (( board[col] |= row )); # boardをリセット
        row=1;
        break;
      else                  # 衝突した場合
        row=$(( row<<1 ));  
      fi
    done
    : '利用可能なポジションが残っていない場合' ;
    if (( board[col]==0 ));then
      # 最初の列にある場合は終了
      if (( col==0 ));then  
        break;
      else
        (( col-- ));        # バックトラック
        row=$(( board[col]<<1 )); # rowを次のrowに設定
        board[$col]=0;      # boardをリセット
        continue;
      fi
    fi
    : '利用可能なポジションが見つかった場合' ;
    if (( col==(size-1) ));then # 最後の列の場合
      (( TOTAL++ ));            # 解を増やす
      printRecord "$size" "1"; 
      row=$(( board[col]<<1 )); # rowを次のrowに設定
      board[$col]=0;            # boardをリセット
      continue;
    else
      ((col++)); # 最後の列でない場合は次に進みます
    fi
  done
}
#
: '再帰版ビットマップ';
function bitmap_R(){
  local -i size="$1";
  local -i row="$2";
  local -i mask="$3";
	local -i left="$4";
	local -i down="$5";
	local -i right="$6";
	local -i bitmap=;
	local -i bit=0;
  if (( row==size ));then
    ((TOTAL++));
    printRecord "$size" "1";
  else
    bitmap=$((mask&~(left|down|right)));
    while ((bitmap)); do
      bit=$((-bitmap&bitmap)) ;
      bitmap=$((bitmap^bit)) ;
      bitmap_R "$size" "$((row+1))" "$mask" "$(((left|bit)<<1))" "$((down|bit))" "$(((right|bit)>>1))"  ;
    done
  fi
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
        break;              # 配置したら抜ける
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
        board[$row]=-1;
      fi
      ((row--));
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
: '非再帰版ブルートフォース';
function bluteForce_NR(){
	local -i row="$1";
	local -i size="$2";
  local -i col=;
  ##
  : '１．非再帰は初期化が必要';
	for ((i=0;i<size;i++)){ board[$i]=-1; }
  ##
  : '２．再帰で呼び出される関数内を回す処理';
  ##
  while (( row>-1 ));do
    local -i matched=0;   # クイーンを配置したか
    ##
    : '３．再帰処理のループ部分';
    ##
    # 非再帰では過去の譜石を記憶するためにboard配列を使う
    for ((col=board[row]+1;col<size;col++));do
      board[$row]=$col;   # クイーンを配置
      matched=1;          # 配置した
      break;              # 配置し終わったらループを抜ける
    done
    ##
    : '４．配置したら実行したい処理';
    ##
    if ((matched));then        # 配置できている
      ((row++));               # 次の配置処理へ
      ##
      : '５．最下部まで到達したときの処理';
      ##
      if ((row==size));then	   # 最下部まで到達
        ((row--));             # 到達したらバックトラック
        ##
        : '効きをチェック';
        ##
        check_bluteForce "$size";   # 効きをチェック
        if (($?==1));then      # 直前のreturnを利用
          ((TOTAL++));         # 解をカウント
          printRecord "$size"; # 出力
        fi
      fi
    ## 
    : '６．配置できなくてバックトラックしたい処理';
    ## 
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
function bluteForce_R(){
  local -i row="$1";
  local -i size="$2";
  local -i col=;
  if ((row==size));then
    check_bluteForce "$size";
    if (( $?==1 ));then 
      ((TOTAL++));
      # 出力しないならコメント
      printRecord "$size";   
    fi
  else
    for(( col=0;col<size;col++ )){
      board["$row"]="$col";
      bluteForce_R $((row+1)) $size ;
    }
  fi
}
#
size=5;
mask=$(( (1<<size)-1 ));
# 非再帰版ビットマップ
time bitmap_NR 5 $mask ;
echo "$TOTAL";

# 再帰版ビットマップ
#time bitmap_R 5 0 $mask 0 0 0;
#echo "$TOTAL";

# 非再帰版配置フラグ
# time postFlag_NR 0 5;
# 再帰版配置フラグ
#time postFlag_R 0 5;
# 非再帰版バックトラッキング
# time backTracking_NR 0 5;
# 再帰版バックトラッキング
#time backTracking_R 0 5;
# 非再帰版ブルートフォース
# time bluteForce_NR 0 5;
# 再帰版ブルートフォース
# time bluteForce_R 0 5;
#
exit;

