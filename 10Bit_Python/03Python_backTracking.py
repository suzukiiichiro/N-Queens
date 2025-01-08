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

CentOS-5.1$ pypy 03Python_backTracking.py
 N:        Total       Unique         hh:mm:ss.ms
 4:            2            0         0:00:00.000
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.002
 7:           40            0         0:00:00.008
 8:           92            0         0:00:00.002
 9:          352            0         0:00:00.002
10:          724            0         0:00:00.007
11:         2680            0         0:00:00.037
12:        14200            0         0:00:00.196
13:        73712            0         0:00:01.129
14:       365596            0         0:00:06.816
15:      2279184            0         0:00:44.993
CentOS-5.1$

"""

from datetime import datetime

# pypyを使う場合はコメントを解除
# pypyで再帰が高速化できる
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')

class NQueens03:
  max:int
  total:int
  unique:int
  aboard:list[int]
  fa:list[int]
  fb:list[int]
  fc:list[int]
  def __init__(self):
    self.max=16;
    self.total=0;
    self.unique=0;
    self.aboard=[0 for i in range(self.max)];
    self.fa=[0 for i in range(2*self.max-1)];
    self.fb=[0 for i in range(2*self.max-1)];
    self.fc=[0 for i in range(2*self.max-1)];
  def nqueens(self,row:int,size:int):
    if row==size:
      self.total+=1;
    else:
      for i in range(size):
        self.aboard[row]=i;
        if self.fa[i]==0 and self.fb[row-i+(size-1)]==0 and self.fc[row+i]==0:
          self.fa[i]=self.fb[row-i+(size-1)]=self.fc[row+i]=1;
          self.nqueens(row+1,size);
          self.fa[i]=self.fb[row-i+(size-1)]=self.fc[row+i]=0;
  def main(self):
    min:int=4;
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(min,self.max):
      self.total=0;
      self.unique=0;
      for j in range(size):
        self.aboard[j]=j;
      start_time=datetime.now();
      self.nqueens(0,size);
      time_elapsed=datetime.now()-start_time;
      # _text='{}'.format(time_elapsed);
      # text=_text[:-3]
      # print("%2d:%13d%13d%20s" % (size,self.total,self.unique,text)); 
      # `.format` の代わりに文字列として直接処理
      text = str(time_elapsed)[:-3]  
      print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")

# 3.バックトラック
# $ python <filename>
# $ pypy <fileName>
# $ codon build -release <filename>
# 15:      2279184            0         0:00:16:449
if __name__ == '__main__':
  NQueens03().main();

