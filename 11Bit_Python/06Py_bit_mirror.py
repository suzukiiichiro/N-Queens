#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
bit バックトラッキング版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

fedora$ python 06Py_bit_mirror.py
 N:        Total       Unique        hh:mm:ss.ms
 4:            2            0         0:00:00.000
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.000
 9:          352            0         0:00:00.002
10:          724            0         0:00:00.008
11:         2680            0         0:00:00.047
12:        14200            0         0:00:00.210
13:        73712            0         0:00:01.252
14:       365596            0         0:00:06.819
15:      2279184            0         0:00:45.747
"""
from datetime import datetime 

# pypyを使う場合はコメントを解除
# import pypyjit
# pypyで再帰が高速化できる
# pypyjit.set_param('max_unroll_recursion=-1')

class NQueens06():
  total:int=0
  unique:int=0
  size:int=0
  def backtrack(self,size:int,row:int,down:int,left:int,right:int)->None:
    self.size=size
    mask:int=(1<<size)-1
    bit:int=0
    if row==size:
      self.total+=1
      return
    bitmap:int=mask&~(down|left|right)
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      self.backtrack(size,row+1,down|bit,(left|bit)<<1,(right|bit)>>1)
  def solve(self, size: int) -> None:
    self.size = size
    for col in range(size // 2):
      bit = 1 << col
      self.backtrack(size, 1, bit, bit << 1, bit >> 1)
    self.total *= 2
    if size % 2 == 1:
      col = size // 2
      bit = 1 << col
      self.backtrack(size, 1, bit, bit << 1, bit >> 1)
  def gettotal(self)->int:
    return self.total
  def main(self)->None:
    nmin:int=4
    nmax:int=18
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin, nmax):
      self.total=0;
      self.unique=0;
      start_time = datetime.now()
      self.solve(size)
      total=self.gettotal()
      time_elapsed = datetime.now()-start_time
      text = str(time_elapsed)[:-3]
      print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")
if __name__=='__main__':
  NQueens06().main();

