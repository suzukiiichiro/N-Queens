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

CentOS$ codon build -release 06Python_bit_mirror.py
CentOS$ ./06Python_bit_mirror
 N:        Total       Unique        hh:mm:ss.ms
 4:            2            0         0:00:00.000
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.000
 9:          352            0         0:00:00.000
10:          724            0         0:00:00.000
11:         2680            0         0:00:00.000
12:        14200            0         0:00:00.004
13:        73712            0         0:00:00.023
14:       365596            0         0:00:00.123
15:      2279184            0         0:00:00.738
16:     14772512            0         0:00:04.622

05Python_bit_backTraking.py
16:     14772512            0         0:00:09.082
04Python_symmetry.py
16:     14772512      1846955         0:00:36.163
03Python_backTracking.py
16:     14772512            0         0:01:50.603
"""

from datetime import datetime 
# pypyを使う場合はコメントを解除
# pypyで再帰が高速化できる
# import pypyjit
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
# 6.バックトラックとビットマップ
# $ python <filename>
# $ pypy <fileName>
# $ codon build -release <filename>
# 15:      2279184            0         0:00:01.422
if __name__=='__main__':
  NQueens06().main();

