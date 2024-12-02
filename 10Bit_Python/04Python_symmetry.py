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

# 実行 
$ python <filename.py>

# 実行結果
1
 0 2 4 1 3
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+

2
 0 3 1 4 2
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+

3
 1 3 0 2 4
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+

4
 1 4 2 0 3
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+

5
 2 0 3 1 4
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+

6
 2 4 1 3 0
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+

7
 3 0 2 4 1
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+

8
 3 1 4 2 0
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+

9
 4 1 3 0 2
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+

10
 4 2 0 3 1
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+


bash-3.2$ python 07Python_carryChain.py
size: 5 TOTAL: 10 UNIQUE: 2
bash-3.2$

"""

from datetime import datetime

# pypyを使う場合はコメントを解除
# import pypyjit
# pypyで再帰が高速化できる
# pypyjit.set_param('max_unroll_recursion=-1')

class NQueens04:
  max:int
  total:int
  unique:int
  aboard:list[int]
  fa:list[int]
  fb:list[int]
  fc:list[int]
  trial:list[int]
  scratch:list[int]
  def __init__(self):
    self.max=16
    self.total=0
    self.unique=0
    self.aboard=[0 for i in range(self.max)]
    self.fa=[0 for i in range(2*self.max-1)]
    self.fb=[0 for i in range(2*self.max-1)]
    self.fc=[0 for i in range(2*self.max-1)]
    self.trial=[0 for i in range(self.max)]
    self.scratch=[0 for i in range(self.max)]
  def rotate(self,chk:list[int],scr:list[int],_n:int,neg:int):
    incr=0
    k:int=0
    # k=0 if neg else _n-1
    if neg:
      k=0
    else:
      k=_n-1
    # incr=1 if neg else -1
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
  def vmirror(self,chk:list[int],neg:int):
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
  def nqueens_rec(self,row:int,size:int):
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
  def main(self):
    nmin:int=4
    size:int
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(nmin,self.max):
      self.total=0
      self.unique=0
      for j in range(size):
        self.aboard[j]=j
      start_time=datetime.now()
      self.nqueens_rec(0,size)
      time_elapsed=datetime.now()-start_time
      # _text='{}'.format(time_elapsed)
      # text=_text[:-3]
      # print("%2d:%13d%13d%20s" % (size,self.total,self.unique,text)); 
      text = str(time_elapsed)[:-3]  
      print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")

# 4.対象解除法
# $ python <filename>
# $ pypy <fileName>
# $ codon build -release <filename>
# 15:      2279184       285053         0:00:49.855
if __name__ == '__main__':
  NQueens04().main();

