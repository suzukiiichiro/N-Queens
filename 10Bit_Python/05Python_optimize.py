#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
バックトラッキング 最適化版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens
"""

"""
CentOS-5.1$ pypy 05Python_optimize.py
 N:        Total       Unique         hh:mm:ss.ms
 4:            2            1         0:00:00.000
 5:           10            2         0:00:00.000
 6:            4            1         0:00:00.001
 7:           40            6         0:00:00.001
 8:           92           12         0:00:00.015
 9:          352           46         0:00:00.009
10:          724           92         0:00:00.010
11:         2680          341         0:00:00.023
12:        14200         1787         0:00:00.074
13:        73712         9233         0:00:00.396
14:       365596        45752         0:00:02.141
15:      2279184       285053         0:00:14.413

CentOS-5.1$ pypy 04Python_symmetry.py
 N:        Total       Unique         hh:mm:ss.ms
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

class NQueens05:
  total:int
  unique:int
  aboard:list[int]
  fa:list[int]
  fb:list[int]
  fc:list[int]
  trial:list[int]
  scratch:list[int]
  def __init__(self):
    pass
  def init(self,size:int)->None:
    self.total=0
    self.unique=0
    self.aboard=[i for i in range(size)]
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
  def intncmp(self,_lt:list[int],_rt:list[int],neg)->int:
    rtn:int=0
    for i in range(neg):
      rtn=_lt[i]-_rt[i]
      if rtn!=0:
        break
    return rtn
  def symmetryops(self,size:int)->int:
    nequiv:int=0
    for i in range(size):
      self.trial[i]=self.aboard[i]
    # 90
    self.rotate(self.trial,self.scratch,size,0)
    k:int=self.intncmp(self.aboard,self.trial,size)
    if k>0:
      return 0
    if k==0:
      nequiv=1
    else:
      #180
      self.rotate(self.trial,self.scratch,size,0)
      k=self.intncmp(self.aboard,self.trial,size)
      if k>0:
        return 0
      if k==0:
        nequiv=2
      else:
        #270
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
    # 90
    if nequiv > 1:
      self.rotate(self.trial,self.scratch,size,1)
      k=self.intncmp(self.aboard,self.trial,size)
      if k>0:
        return 0
      # 180
      if nequiv>2:
        self.rotate(self.trial,self.scratch,size,1)
        k=self.intncmp(self.aboard,self.trial,size)
        if k>0:
          return 0
        #270
        self.rotate(self.trial,self.scratch,size,1)
        k=self.intncmp(self.aboard,self.trial,size)
        if k>0:
          return 0
    return nequiv*2
  def nqueens(self,row:int,size:int)->None:
    if row==size-1:
      if self.fb[row-self.aboard[row]+size-1] or self.fc[row+self.aboard[row]]:
        return
      stotal:int=self.symmetryops(size)
      if stotal!=0:
        self.unique+=1
        self.total+=stotal
    else:
      lim:int=size if row!=0 else (size+1) //2
      tmp:int
      for i in range(row,lim):
        tmp=self.aboard[i]
        self.aboard[i]=self.aboard[row]
        self.aboard[row]=tmp
        if self.fb[row-self.aboard[row]+size-1]==0 and self.fc[row+self.aboard[row]]==0:
          self.fb[row-self.aboard[row]+size-1]=self.fc[row+self.aboard[row]]=1
          self.nqueens(row+1,size)
          self.fb[row-self.aboard[row]+size-1]=self.fc[row+self.aboard[row]]=0
      tmp=self.aboard[row]
      for i in range(row+1,size):
        self.aboard[i-1]=self.aboard[i]
      self.aboard[size-1]=tmp
  def main(self)->None:
    min:int=4
    max:int=18
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(min,max):
      self.init(size)
      start_time=datetime.now()
      self.nqueens(0,size)
      time_elapsed=datetime.now()-start_time
      text = str(time_elapsed)[:-3]
      print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")
# 5.枝刈りと最適化
# $ python <filename>
# $ pypy <fileName>
# $ codon build -release <filename>
# 15:      2279184       285053         0:00:15.677
if __name__=='__main__':
  NQueens05().main();

