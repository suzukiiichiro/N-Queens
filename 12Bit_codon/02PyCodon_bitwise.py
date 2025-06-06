#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# ビット演算による高速化（上級者向け） 
# 非常に高速（ビット演算）
# 解の個数のみカウント（盤面出力なし）
# 大きな n に適している（例：n=15程度までOK）
#
# codon
# 02
# 17:     95815104            0         0:01:08.155
# 01
# abort
#
# pypyを使う場合はコメントを解除
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

from datetime import datetime

def solve_n_queens(n:int)->int:
  count:int = 0
  def backtrack(row:int, cols:int, hills:int, dales:int):
    nonlocal count
    if row == n:
      count += 1
      return
    free:int = (~(cols | hills | dales)) & ((1 << n) - 1)
    while free:
      bit:int = free & -free
      free ^= bit
      backtrack(row + 1, cols | bit, (hills | bit) << 1, (dales | bit) >> 1)
  backtrack(0, 0, 0, 0)
  return count

if __name__ == '__main__':
  min:int=4;
  max:int=18
  print(" N:        Total       Unique         hh:mm:ss.ms")
  for size in range(min,max):
    start_time=datetime.now();
    #
    total=solve_n_queens(size)
    unique=0
    #
    time_elapsed=datetime.now()-start_time;
    text = str(time_elapsed)[:-3]
    print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")


