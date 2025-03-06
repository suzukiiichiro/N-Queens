#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
ビットマップ版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens
"""

"""
CentOS-5.1$ pypy 04Python_symmetry.py
 N:        Total       Unique         hh:mm:ss.ms
 4:            2            1         0:00:00.000
 5:           10            2         0:00:00.000
 6:            4            1         0:00:00.002
 7:           40            6         0:00:00.009
 8:           92           12         0:00:00.005
 9:          352           46         0:00:00.009
10:          724           92         0:00:00.018
11:         2680          341         0:00:00.046
12:        14200         1787         0:00:00.214
13:        73712         9233         0:00:01.206
14:       365596        45752         0:00:07.167
15:      2279184       285053         0:00:46.629

CentOS-5.1$ pypy 03Python_backTracking.py
 N:        Total       Unique         hh:mm:ss.ms
15:      2279184            0         0:00:44.993


"""

from datetime import datetime

# pypyを使う場合はコメントを解除
# pypyで再帰が高速化できる
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

class NQueens04:
  total:int
  unique:int
  aboard:list[int]
  fa:list[int]
  fb:list[int]
  fc:list[int]
  trial:list[int]
  scratch:list[int]
  def __init__(self)->None:
    pass
  def init(self,size:int)->None:
    self.total=0
    self.unique=0
    self.aboard=[0 for i in range(size)]
    self.fa=[0 for i in range(2*size-1)]
    self.fb=[0 for i in range(2*size-1)]
    self.fc=[0 for i in range(2*size-1)]
    self.trial=[0 for i in range(size)]
    self.scratch=[0 for i in range(size)]
  def rotate(self,chk:list[int],scr:list[int],_n:int,neg:int)->None:
    incr:int=0
    k:int=0
    if neg:
      k=0
    else:
      k=_n-1
    if neg:
      incr=1
    else:
      incr-=1
    for i in range(_n):
      scr[i]=chk[k]
      k+=incr
    k=_n-1 if neg else 0
    for i in range(_n):
      chk[scr[i]]=k
      k-=incr
  def vmirror(self,chk:list[int],neg:int)->None:
    for i in range(neg):
      chk[i]=(neg-1)-chk[i]
  def intncmp(self,_lt:list[int],_rt:list[int],neg:int)->int:
    rtn:int=0
    for i in range(neg):
      rtn=_lt[i]-_rt[i]
      if rtn!=0:
        break
    return rtn
  def symmetryops(self,size:int)->int:
    neqvuiv:int=0
    for i in range(size):
      self.trial[i]=self.aboard[i]
    # 90
    self.rotate(self.trial,self.scratch,size,0)
    k=self.intncmp(self.aboard,self.trial,size)
    if k>0:
      return 0
    if k==0:
      nequiv=1
    else:
      # 180
      self.rotate(self.trial,self.scratch,size,0)
      k=self.intncmp(self.aboard,self.trial,size)
      if k>0:
        return 0
      if k==0:
        nequiv=2
      else:
        # 270
        self.rotate(self.trial,self.scratch,size,0)
        k=self.intncmp(self.aboard,self.trial,size)
        if k>0:
          return 0
        nequiv=4
    for i in range(size):
      self.trial[i]=self.aboard[i]
    # 垂直反転
    self.vmirror(self.trial,size)
    k=self.intncmp(self.aboard,self.trial,size)
    if k>0:
      return 0
    if nequiv>1:
      # 90
      self.rotate(self.trial,self.scratch,size,1)
      k=self.intncmp(self.aboard,self.trial,size)
      if k>0:
        return 0
      if nequiv>2:
        # 180
        self.rotate(self.trial,self.scratch,size,1)
        k=self.intncmp(self.aboard,self.trial,size)
        if k>0:
          return 0
        # 270
        self.rotate(self.trial,self.scratch,size,1)
        k=self.intncmp(self.aboard,self.trial,size)
        if k>0:
          return 0
    return nequiv*2
  def nqueens_rec(self,row:int,size:int)->None:
    stotal:int
    if row==size:
      stotal=self.symmetryops(size)
      if stotal!=0:
        self.unique+=1
        self.total+=stotal
    else:
      for i in range(size):
        self.aboard[row]=i
        if self.fa[i]==0 and self.fb[row-i+(size-1)]==0 and self.fc[row+i]==0:
          self.fa[i]=self.fb[row-i+(size-1)]=self.fc[row+i]=1
          self.nqueens_rec(row+1,size)
          self.fa[i]=self.fb[row-i+(size-1)]=self.fc[row+i]=0
  def main(self)->None:
    min:int=4
    max:int=18
    size:int
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(min,max):
      self.init(size)
      start_time=datetime.now()
      self.nqueens_rec(0,size)
      time_elapsed=datetime.now()-start_time
      text = str(time_elapsed)[:-3]  
      print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")

# 4.対象解除法
# $ python <filename>
# $ pypy <fileName>
# $ codon build -release <filename>
# 15:      2279184       285053         0:00:49.855
if __name__ == '__main__':
  NQueens04().main();

