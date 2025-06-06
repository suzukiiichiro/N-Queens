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
#import pypyjit
#pypyjit.set_param('max_unroll_recursion=-1')

import time

def solve_n_queens(n:int)->list[list[int]]:
  def is_safe(queens:list[int],row:int,col:int)->bool:
    for r,c in enumerate(queens):
      if c==col or abs(c-col)==abs(r-row):
        return False
    return True

  def backtrack(row:int,queens:list[int],solutions:list[list[int]]):
    if row==n:
      solutions.append(queens[:])
      return
    for col in range(n):
      if is_safe(queens,row,col):
        queens.append(col)
        backtrack(row+1,queens,solutions)
        queens.pop()

  solutions=[]
  backtrack(0,[],solutions)
  return solutions

def main():
  min_n:int=4
  max_n:int=18
  print(" N:        Total       Unique         hh:mm:ss.ms")
  for size in range(min_n,max_n):
    start_time=time.time()
    total=len(solve_n_queens(size))
    unique=0
    time_elapsed=time.time()-start_time
    msec=int((time_elapsed-int(time_elapsed))*1000)
    sec=int(time_elapsed)%60
    minutes=(int(time_elapsed)//60)%60
    hours=int(time_elapsed)//3600
    text=f"{hours}:{minutes:02d}:{sec:02d}.{msec:03d}"
    print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")

if __name__=="__main__":
  main()
