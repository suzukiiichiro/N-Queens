#!/usr/bin/bash

: '
エイト・クイーンの解説をしています。ご興味のある方はどうぞ。

Ｎクイーン問題（１８）第四章　偉人のソースを読む「エイト・クイーンノスタルジー」
https://suzukiiichiro.github.io/posts/2023-04-18-01-n-queens-suzuki/
Ｎクイーン問題（１７）第四章　偉人のソースを読む「Ｎ２４を発見 Ｊｅｆｆ Ｓｏｍｅｒｓ」
https://suzukiiichiro.github.io/posts/2023-04-21-01-n-queens-suzuki/
Ｎクイーン問題（１６）第三章　対象解除法 ソース解説
https://suzukiiichiro.github.io/posts/2023-04-18-01-n-queens-suzuki/
Ｎクイーン問題（１５）第三章　対象解除法 ロジック解説
https://suzukiiichiro.github.io/posts/2023-04-13-02-nqueens-suzuki/
Ｎクイーン問題（１４）第三章　ミラー
https://suzukiiichiro.github.io/posts/2023-04-13-01-nqueens-suzuki/
Ｎクイーン問題（１３）第三章　ビットマップ
https://suzukiiichiro.github.io/posts/2023-04-05-01-nqueens-suzuki/
Ｎクイーン問題（１２）第二章　まとめ
https://suzukiiichiro.github.io/posts/2023-03-17-02-n-queens-suzuki/
Ｎクイーン問題（１１）第二章　配置フラグの再帰・非再帰
https://suzukiiichiro.github.io/posts/2023-03-17-01-n-queens-suzuki/
Ｎクイーン問題（１０）第二章　バックトラックの再帰・非再帰
https://suzukiiichiro.github.io/posts/2023-03-16-01-n-queens-suzuki/
Ｎクイーン問題（９）第二章　ブルートフォースの再帰・非再帰
https://suzukiiichiro.github.io/posts/2023-03-14-01-n-queens-suzuki/
Ｎクイーン問題（８）第一章　まとめ
https://suzukiiichiro.github.io/posts/2023-03-09-01-n-queens-suzuki/
Ｎクイーン問題（７）第一章　ブルートフォース再び
https://suzukiiichiro.github.io/posts/2023-03-08-01-n-queens-suzuki/
Ｎクイーン問題（６）第一章　配置フラグ
https://suzukiiichiro.github.io/posts/2023-03-07-01-n-queens-suzuki/
Ｎクイーン問題（５）第一章　進捗表示テーブルの作成
https://suzukiiichiro.github.io/posts/2023-03-06-01-n-queens-suzuki/
Ｎクイーン問題（４）第一章　バックトラック
https://suzukiiichiro.github.io/posts/2023-02-21-01-n-queens-suzuki/
Ｎクイーン問題（３）第一章　バックトラック準備編
https://suzukiiichiro.github.io/posts/2023-02-14-03-n-queens-suzuki/
Ｎクイーン問題（２）第一章　ブルートフォース
https://suzukiiichiro.github.io/posts/2023-02-14-02-n-queens-suzuki/
Ｎクイーン問題（１）第一章　エイトクイーンについて
https://suzukiiichiro.github.io/posts/2023-02-14-01-n-queens-suzuki/

エイト・クイーンのソース置き場 BashもJavaもPythonも！
https://github.com/suzukiiichiro/N-Queens
';



declare -i COUNT=0;
declare -i TOTAL=0;     # カウンター
declare -i UNIQUE=0;    # ユニークユーザー
declare -i DISPLAY=0;   # ボード出力するか
#
: 'ボードレイアウトを出力 ビットマップ対応版';
function printRecord()
{
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
: '非再帰ビットマップ版';
function bitmap_NR()
{
  local -i size="$1";
  local -i row="$2";
  local -i mask=$(( (1<<size)-1 ));
  local -i bitmap;
  local -i bit;
  local -a left[$size];
  local -a down[$size];
  local -a right[$size];
  local -a tmpBoard[$size];       #各rowのleft,right,downを足した配列
  local -a board[$size];          #クイーンを配置した場所を格納する配列
  bitmap=mask;
  while true ;do
    if (( bitmap ));then
      bit=$(( -bitmap&bitmap ));  # 一番右のビットを取り出す 
      bitmap=$(( bitmap^bit ));   # 配置可能なパターンが一つずつ取り出される
      board[$row]="$bit";         # Qを配置
      if (( row==(size-1) ));then
        ((TOTAL++));
        if ((DISPLAY==1));then
          printRecord "$size" "1";# 出力 1:bitmap版 0:それ以外
        fi
        bitmap=tmpBoard[row];
        ((--row));
        continue;
      else
        local -i n=$((row++));
        left[$row]=$(((  left[n]|bit)<<1 ));
        down[$row]=$((   down[n]|bit ));
        right[$row]=$(((right[n]|bit)>>1 ));
        tmpBoard[$row]=$bitmap;
        board[$row]="$bit";       # Qを配置
        bitmap=$(( mask&~(left[row]|down[row]|right[row]) ));
        continue;
      fi
    else
      bitmap=tmpBoard[row];
      if (( row==0));then
        break ;
      fi
      (( row-- ));
      continue
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
    bitmap=$(( mask&~(left|down|right) ));
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
function rdlFlag_NR()
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
  local startTime=0;
	local endTime=0;
	local hh=mm=ss=0; 
  echo " N:        Total       Unique        hh:mm:ss" ;
  for((N=min;N<=max;N++)){
    TOTAL=0;
    UNIQUE=0;
    startTime=$(date +%s);# 計測開始時間
    row=0;
    mask=$(( (1<<N)-1 ));
    "$selectName" "$N" 0 "$mask" 0 0 0;

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
8) 非再帰ビットマップ
7) 再帰ビットマップ
6) 非再帰　配置フラグ
5) 再帰　　配置フラグ 
4) 非再帰　バックトラック 
3) 再帰　　バックトラック
2) 非再帰　ブルートフォース 
1) 再帰　　ブルートフォース 

echo "行頭の番号を入力してください";

" selectNo;
echo 
case "$selectNo" in
  8)
    while :
    do 
      read -n1 -p "
      ボード画面の表示は？ 
      y) する（ブルートフォースはおすすめしない）
      n) しない

      echo "行頭のアルファベットを入力して下さい";

      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    NQ bitmap_NR
    break;
    ;;
  7)
    while :
    do 
      read -n1 -p "
      ボード画面の表示は？ 
      y) する（ブルートフォースはおすすめしない）
      n) しない

      echo "行頭のアルファベットを入力して下さい";

      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    NQ bitmap_R 
    break;
    ;;
  6)
    while :
    do 
      read -n1 -p "
      ボード画面の表示は？ 
      y) する（ブルートフォースはおすすめしない）
      n) しない

      echo "行頭のアルファベットを入力して下さい";

      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    NQ rdlFlag_NR
    break;
    ;;
  5)
    while :
    do 
      read -n1 -p "
      ボード画面の表示は？ 
      y) する（ブルートフォースはおすすめしない）
      n) しない

      echo "行頭のアルファベットを入力して下さい";

      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    NQ postFlag_R;
    break;
    ;;
  4)
    while :
    do 
      read -n1 -p "
      ボード画面の表示は？ 
      y) する（ブルートフォースはおすすめしない）
      n) しない

      echo "行頭のアルファベットを入力して下さい";

      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    NQ backTracking_NR;
    break;
    ;;
  3)
    while :
    do 
      read -n1 -p "
      ボード画面の表示は？ 
      y) する（ブルートフォースはおすすめしない）
      n) しない

      echo "行頭のアルファベットを入力して下さい";

      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    NQ backTracking_R;
    break;
    ;;
  2)
    while :
    do 
      read -n1 -p "
      ボード画面の表示は？ 
      y) する（ブルートフォースはおすすめしない）
      n) しない

      echo "行頭のアルファベットを入力して下さい";

      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    NQ bluteForce_NR;
    break;
    ;;
  1)
    while :
    do 
      read -n1 -p "
      ボード画面の表示は？ 
      y) する（ブルートフォースはおすすめしない）
      n) しない

      echo "行頭のアルファベットを入力して下さい";

      " select;
      echo; 
      case "$select" in
        y|Y) DISPLAY=1; break; ;;
        n|N) DISPLAY=0; break; ;;
      esac
    done
    NQ bluteForce_R;
    break;
    ;;
  *)
    ;; 
esac
done
exit;

