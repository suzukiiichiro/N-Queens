#!/usr/bin/bash

declare -i TOTAL=0;         # 解の合計数
#

: '
特定の位置を指定して、以前の位置と衝突したかどうかを検出します。入力行は (1<<行) でエンコードする必要があります。衝突した場合は 1 を返し、そうでない場合は 0 を返します
';
: '効き筋をチェック';
function collide(){
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
      collide "$col" "$row";# 効きのチェック
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
      row=$(( board[col]<<1 )); # rowを次のrowに設定
      board[$col]=0;            # boardをリセット
      continue;
    else
      ((col++)); # 最後の列でない場合は次に進みます
    fi
  done
}
#
bitmap_NR 5;
echo "$TOTAL";
exit;
