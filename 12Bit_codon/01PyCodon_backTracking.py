#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# バックトラッキング（基本的な実装）
# 初学者向け
# O(n!)程度の時間計算量
# 解のリストが得られる（各行のクイーンの列位置）
#
# python
# 13:        73712            0         0:01:02.597 
# pypy
# 13:        73712            0         0:00:04.306
#
# pypyを使う場合はコメントを解除
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')

from datetime import datetime

def solve_n_queens(n:int) -> list[list[int]]:
  def is_safe(queens:list[int], row:int, col:int)->bool:
    for r, c in enumerate(queens):
      if c == col or abs(c - col) == abs(r - row):
        return False
    return True
  def backtrack(row:int, queens:list[int]):
    if row == n:
      solutions.append(queens[:])
      return
    for col in range(n):
      if is_safe(queens, row, col):
        queens.append(col)
        backtrack(row + 1, queens)
        queens.pop()
  solutions = []
  backtrack(0, [])
  return solutions

if __name__ == '__main__':
  min:int=4;
  max:int=18
  print(" N:        Total       Unique         hh:mm:ss.ms")
  for size in range(min,max):
    start_time=datetime.now();
    #
    total=len(solve_n_queens(size))
    unique=0
    #
    time_elapsed=datetime.now()-start_time;
    text = str(time_elapsed)[:-3]
    print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")

