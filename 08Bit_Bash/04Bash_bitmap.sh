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



declare -i TOTAL=0;     # カウンター
#
: 'ボードレイアウトを出力';
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
  while ((row>-1));do
    if (( bitmap[row]>0 ));then
      bit=$(( -bitmap[row]&bitmap[row] ));  # 一番右のビットを取り出す
      bitmap[$row]=$(( bitmap[row]^bit ));  # 配置可能なパターンが一つずつ取り出される
      board[$row]="$bit";                   # Qを配置
      if (( row==(size-1) ));then
        ((TOTAL++));
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
: '再帰版ビットマップ';
function bitmap_R()
{
  local -i size="$1"; 
  local -i row="$2";
  local -i left="$3"; 
  local -i down="$4"; 
  local -i right="$5";
  local -i bitmap=;
  local -i bit=;
  local -i col=0;                     # 再帰に必要
  local -i mask=$(( (1<<size)-1 ));
  if (( row==size ));then
     ((TOTAL++));
     printRecord "$size" "1";         # 出力 1:bitmap版 0:それ以外
  else
    bitmap=$(( mask&~(left|down|right) )); # クイーンが配置可能な位置を表す
    while (( bitmap ));do
      bit=$((-bitmap&bitmap)) ;       # 一番右のビットを取り出す
      bitmap=$((bitmap&~bit)) ;       # 配置可能なパターンが一つずつ取り出される
      board[$row]="$bit";             # Qを配置
      bitmap_R "$size" "$((row+1))" "$(( (left|bit)<<1 ))" "$((down|bit))" "$(( (right|bit)>>1 ))";
    done
  fi
}
# 非再帰版ビットマップ
 time bitmap_NR 5 0;
#
# 再帰版ビットマップ
# time bitmap_R 5 0 0 0 0;    
 echo "$TOTAL";
#
exit;
