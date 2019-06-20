#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  /**
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#  
#   実行
#   $ python Py06_N-Queen.py
#  
#   * ６．バックトラック＋ビットマップ
#   *
#  
#   実行結果
# N:        Total       Unique        hh:mm:ss.ms
# 4:            2            0         0:00:00.000
# 5:           10            0         0:00:00.000
# 6:            4            0         0:00:00.000
# 7:           40            0         0:00:00.000
# 8:           92            0         0:00:00.001
# 9:          352            0         0:00:00.005
#10:          724            0         0:00:00.024
#11:         2680            0         0:00:00.119
#12:        14200            0         0:00:00.606
#13:        73712            0         0:00:03.348
#14:       365596            0         0:00:19.538
#15:      2279184            0         0:02:05.652
#
#
# グローバル変数
MAX=16; #N=15
TOTAL=0;
UNIQUE=0;
SIZE=0;
MASK=0 ;
aBoard=[0 for i in range(MAX)];
#
# ロジックメソッド
def NQueen(row,left,down,right):
  global aBoard;
  global TOTAL;
  global SIZE;
  global UNIQUE;
  global MASK;
  if row==SIZE:
    TOTAL+=1;
  else:
    bitmap=(MASK&~(left|down|right));
    while bitmap:
      bit=(-bitmap&bitmap);
      bitmap=(bitmap^bit);
      NQueen(row+1,(left|bit)<<1, down|bit, (right|bit)>>1);
#
# メインメソッド
from datetime import datetime 
def main():
  global TOTAL;
  global UNIQUE;
  global aBoard;
  global MASK;
  global MAX;
  global SIZE;
  min=4;                          # Nの最小値（スタートの値）を格納
  print(" N:        Total       Unique        hh:mm:ss.ms");
  for i in range(min,MAX):
    SIZE=i;
    TOTAL=0;
    UNIQUE=0;                   # 初期化
    for j in range(i):
      aBoard[j]=j;              # 盤を初期化
    # 追加
    MASK=((1<<SIZE)-1);
    start_time = datetime.now() 
    # NQueen(0,i)
    NQueen(0,0,0,0);
    time_elapsed=datetime.now()-start_time 
    _text='{}'.format(time_elapsed)
    text=_text[:-3]
    print("%2d:%13d%13d%20s" % (SIZE,TOTAL,UNIQUE,text)); # 出力
#
main();
#
