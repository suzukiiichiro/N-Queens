#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
バックトラッキング版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

fedora$ pypy 03Py_backTracking_pypy.py
 N:        Total       Unique         hh:mm:ss.ms
 4:            2            0         0:00:00.000
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.003
 7:           40            0         0:00:00.008
 8:           92            0         0:00:00.002
 9:          352            0         0:00:00.003
10:          724            0         0:00:00.008
11:         2680            0         0:00:00.039
12:        14200            0         0:00:00.213
13:        73712            0         0:00:01.190
14:       365596            0         0:00:07.369
15:      2279184            0         0:00:47.718
"""

from datetime import datetime

# pypyを使う場合はコメントを解除
import pypyjit
# pypyで再帰が高速化できる
pypyjit.set_param('max_unroll_recursion=-1')

class NQueens03:
  total:int
  unique:int
  aboard:list[int]
  fa:list[int]
  fb:list[int]
  fc:list[int]
  def __init__(self)->None:
    pass
  def init(self,size:int)->None:
    self.total=0;
    self.unique=0;
    self.aboard=[0 for i in range(size)];
    self.fa=[0 for i in range(2*size-1)];
    self.fb=[0 for i in range(2*size-1)];
    self.fc=[0 for i in range(2*size-1)];
  def nqueens(self,row:int,size:int)->None:
    if row==size:
      self.total+=1;
    else:
      for i in range(size):
        self.aboard[row]=i;
        if self.fa[i]==0 and self.fb[row-i+(size-1)]==0 and self.fc[row+i]==0:
          self.fa[i]=self.fb[row-i+(size-1)]=self.fc[row+i]=1;
          self.nqueens(row+1,size);
          self.fa[i]=self.fb[row-i+(size-1)]=self.fc[row+i]=0;
  def main(self)->None:
    min:int=4;
    max:int=18
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(min,max):
      self.init(size)
      start_time=datetime.now();
      self.nqueens(0,size);
      time_elapsed=datetime.now()-start_time;
      text = str(time_elapsed)[:-3]  
      print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")
if __name__ == '__main__':
  NQueens03().main();

