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
