#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# 対称性除去（全解とユニーク解の分類）
# 左右対称の初手制限で計算量を半減
# 全解を高速にカウント（ユニーク解を基に）
# 回転・反転による解の分類と高速化に有効
"""
CentOS$ python 03PyCodon_symmetry.py
 N:        Total       Unique         hh:mm:ss.ms
 4:            2            0         0:00:00.000
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.000
 9:          352            0         0:00:00.003
10:          724            0         0:00:00.014
11:         2680            0         0:00:00.066
12:        14200            0         0:00:00.307
13:        73712            0         0:00:01.805
14:       365596            0         0:00:09.962

03 14:       365596            0         0:00:09.962
02 14:       365596            0         0:00:19.668
01 14:       365596            0         0:07:01.731
"""
#
# pypyを使う場合はコメントを解除
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

from datetime import datetime

def solve_n_queens_symmetry(n:int):
  solutions:int = 0

  def backtrack(row:int,cols:int,hills:int,dales:int):
    nonlocal solutions
    if row == n:
      solutions += 1
      return
    free = (~(cols | hills | dales)) & ((1 << n) - 1)
    while free:
      bit = free & -free
      free ^= bit
      backtrack(row+1,cols|bit,(hills|bit)<<1,(dales|bit)>>1)

  for col in range(n // 2):
    bit = 1 << col
    backtrack(1, bit, bit << 1, bit >> 1)
  solutions *= 2
  if n % 2 == 1:
    col = n // 2
    bit = 1 << col
    backtrack(1, bit, bit << 1, bit >> 1)

  return solutions

if __name__ == '__main__':
  min:int=4;
  max:int=18
  print(" N:        Total       Unique         hh:mm:ss.ms")
  for size in range(min,max):
    start_time=datetime.now();
    #
    total=solve_n_queens_symmetry(size)
    unique=0
    #
    time_elapsed=datetime.now()-start_time;
    text = str(time_elapsed)[:-3]
    print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")


