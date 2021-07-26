#!/usr/bin/env python

# -*- coding: utf-8 -*-
""" py06_nqueen.py """

from datetime import datetime

#  /**
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#
#   実行
#   $ python py06_nqueen.py
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
MAX = 16 #N=15
TOTAL = 0
UNIQUE = 0
SIZE = 0
MASK = 0
ABOARD = [0 for i in range(MAX)]
#
def nqueen(row, left, down, right):
  """ nqueen() """
  global TOTAL    # pylint: disable=W0603
  if row == SIZE:
    TOTAL += 1
  else:
    bitmap = (MASK&~(left|down|right))
    while bitmap:
      bit = (-bitmap&bitmap)
      bitmap = (bitmap^bit)
      nqueen(row+1, (left|bit)<<1, down|bit, (right|bit)>>1)
#
def main():
  """ main() """
  global TOTAL    # pylint: disable=W0603
  global UNIQUE   # pylint: disable=W0603
  global ABOARD   # pylint: disable=W0603
  global MASK     # pylint: disable=W0603
  global SIZE     # pylint: disable=W0603
  nmin = 4                         # Nの最小値（スタートの値）を格納
  print(" N:        Total       Unique        hh:mm:ss.ms")
  for i in range(nmin, MAX):
    SIZE = i
    TOTAL = 0
    UNIQUE = 0                   # 初期化
    for j in range(i):
      ABOARD[j] = j            # 盤を初期化
    MASK = ((1<<SIZE)-1)
    start_time = datetime.now()
    nqueen(0, 0, 0, 0)
    time_elapsed = datetime.now()-start_time
    _text = '{}'.format(time_elapsed)
    text = _text[:-3]
    print("%2d:%13d%13d%20s" % (SIZE, TOTAL, UNIQUE, text)) # 出力
#
main()
#
