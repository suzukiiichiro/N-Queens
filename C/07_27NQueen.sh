#!/bin/bash
#
#/**
#  Cで学ぶアルゴリズムとデータ構造  
#  ステップバイステップでＮ−クイーン問題を最適化
#  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
SIZE=$1;
if [ -z "$1" ];then
  echo "SIZEを入力してください";
  echo "./07_27NQueen.sh 16";
  exit;
fi
SIZEE=$((SIZE - 1));
B1=1;
B2=$((SIZEE - 1));
lTotal=0;
lUnique=0;
echo "サイズ:$SIZE";
for i in `seq 1 $SIZEE`;do
  RTN=$(./07_27NQueen_child "$SIZE" "$B1" "$B2");
  echo "$RTN";
  #si:16:B1:12:B2:3:C2:0:C4:0:C8:4256
  C2=$(echo "$RTN"|awk -F: '{print $8;}');
  C4=$(echo "$RTN"|awk -F: '{print $10;}');
  C8=$(echo "$RTN"|awk -F: '{print $12;}');
  lTotal=$((lTotal+ C2 * 2 + C4 * 4 + C8 * 8));
  lUnique=$((lUnique+ C2 + C4 + C8));
  if [ "$B1" -eq "$SIZEE" ];then
    echo "lTotal:$lTotal";
    echo "lUnique:$lUnique";
  fi
  B1=$((B1 + 1));
  B2=$((B2 - 1));
done
