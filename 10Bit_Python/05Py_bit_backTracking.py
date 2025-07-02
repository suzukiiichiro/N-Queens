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

CentOS$ codon build -release 05Python_bit_backTracking.py
CentOS$ ./05Python_bit_backTracking
 N:        Total       Unique        hh:mm:ss.ms
 4:            2            0         0:00:00.000
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.000
 9:          352            0         0:00:00.000
10:          724            0         0:00:00.000
11:         2680            0         0:00:00.001
12:        14200            0         0:00:00.007
13:        73712            0         0:00:00.042
14:       365596            0         0:00:00.218
15:      2279184            0         0:00:01.360
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

class NQueens05():
  total:int
  unique:int

  def __init__(self):
    pass

  def init(self):
    self.total=0
    self.unique=0

  def NQueens(self,size:int,row:int,left:int,down:int,right:int):
    if row==size:
      self.total+=1
    else:
      bit:int=0
      mask:int=(1<<size)-1
      bitmap:int=mask&~(left|down|right)
      while bitmap:
        bit=-bitmap&bitmap
        bitmap=bitmap&~bit
        self.NQueens(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1)

  def main(self):
    nmin:int=4
    nmax:int=18
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin, nmax):
      self.init()
      start_time = datetime.now()
      self.NQueens(size,0,0,0,0)
      time_elapsed = datetime.now()-start_time
      text = str(time_elapsed)[:-3]
      print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")
# 6.バックトラックとビットマップ
# $ python <filename>
# $ pypy <fileName>
# $ codon build -release <filename>
# 15:      2279184            0         0:00:01.422
if __name__=='__main__':
  NQueens05().main();
