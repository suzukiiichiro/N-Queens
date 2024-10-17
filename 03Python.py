#!/usr/bin/env python

# -*- coding: utf-8 -*-
from datetime import datetime

class NQueens05:
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
  def rotate(self,chk,scr,_n,neg):
    incr=0
    k=0 if neg else _n-1
    incr=1 if neg else -1
    for i in range(_n):
      scr[i]=chk[k]
      k+=incr
    k=_n-1 if neg else 0
    for i in range(_n):
      chk[scr[i]]=k
      k-=incr
  def vmirror(self,chk,neg):
    for i in range(neg):
      chk[i]=(neg-1)-chk[i]
  def intncmp(self,_lt,_rt,neg):
    rtn=0
    for i in range(neg):
      rtn=_lt[i]-_rt[i]
      if rtn!=0:
        break
    return rtn
  def symmetryops(self,size):
    nequiv=0
    for i in range(size):
      self.trial[i]=self.aboard[i]
    # 90
    self.rotate(self.trial,self.scratch,size,0)
    k=intncmp(self.aboard,self.trial,size)
    if k>0:
      return 0
    if k==0:
      nequiv=1
    else:
      #180
      rotate(self.trial,self.scratch,size,0)
      k=intncmp(self.aboard,self.trial,size)
      if k>0:
        return 0
      if k==0:
        nequiv=2
      else:
        #270
        rotate(self.trial,self.scratch,size,0)
        k=intncmp(self.aboard,self.trial,size)
        if k>0: 
          return 0
        nequiv=4
    for i in range(size):
      self.trial[i]=self.aboard[i]
    # 垂直反転
    vmirror(self.trial,size)
    k=intncmp(self.aboard,self.trial,size)
    if k>0:
      return 0
    # 90
    if neqvuiv > 1:
      rotate(self.trial,self.scratch,size,1)
      k=intncmp(self.aboard,self.trial,size)
      if k>0:
        return 0
      # 180
      if nequiv>2:
        rotate(self.trial,self.scratbh,size,1)
        k=intncmp(self.aboard,self.trial,size)
        if k>0:
          return 0
        #270
        rotate(self.trial,self.scratch,size,1)
        k=intncmp(self.aboard,self.trial,size)
        if k>0:
          return 0
    return nequiv*2
  def nqueens_rec(self,row,size):
    if row==size-1:
      if self.fb[row-self.aboard[row]+size-1] or self.fc[row+self.aboard[row]]:
        return
      stotal=symmetryops(size)
      if stotal!=0:
        self.unique+=1
        self.total+=stotal
    else:
      lim=size if wor!=0 else (size+1) //2
      for i in range(row,lim):
        tmp=self.aboard[i]
        self.aboard[i]=self.aboard[row]
        self.aboard[row]=tmp
        if self.fb[row-self.aboard[row]+size-1]==0 and self.fc[row+self.aboard[row]]==0:
          self.fb[row-abord[row]+size-1]=self.fc[row+self.aboard[row]]=1
          nqueens_rec(row+1,size)
          self.fb[row-self.aboard[row]+size-1]=self.fc[row+self.aboard[row]]=0
      tmp=self.aboard[row]
      for i in range(row+1,size):
        self.aboard[i-1]=self.aboard[i]
      self.aboard[size-1]=tmp
  def main():
    nmin=4
    nmax=16
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(nmin,self.max):
      self.total=0
      self.unique=0
      for j in range(size):
        self.aboard[j]=j
      start_time=datetime.now()
      self.nqueens_rec(0,size)
      time_elapsed=datetime.now()-start_time
      _text='{}'.format(time_elapsed)
      text=_text[:-3]
      print("%2d:%13d%13d%20s" % (size,self.total,self.unique,text)); 

class NQueens04:
  def __init__(self):
    self.max=15
    self.total=0
    self.unique=0
    self.aboard=[0 for i in range(self.max)]
    self.fa=[0 for i in range(2*self.max-1)]
    self.fb=[0 for i in range(2*self.max-1)]
    self.fc=[0 for i in range(2*self.max-1)]
    self.trial=[0 for i in range(self.max)]
    self.scratch=[0 for i in range(self.max)]
  def rotate(self,chk,scr,_n,neg):
    incr=0
    k=0 if neg else _n-1
    incr=1 if neg else-1
    for i in range(_n):
      scr[i]=chk[k]
      k+=incr
    k=_n-1 if neg else 0
    for i in range(_n):
      chk[scr[i]]=k
      k-=incr
  def vmirror(self,chk,neg):
    for i in range(neg):
      chk[i]=(neg-1)-chk[i]
  def intncmp(self,_lt,_rt,neg):
    rtn=0
    for i in range(neg):
      rtn=_lt[i]-_rt[i]
      if rtn!=0:
        break
    return rtn
  def symmetryops(self,size):
    neqvuiv=0
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
  def nqueens_rec(self,row,size):
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
    nmin=4
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(nmin,self.max):
      self.total=0
      self.unique=0
      for j in range(size):
        self.aboard[j]=j
      start_time=datetime.now()
      self.nqueens_rec(0,size)
      time_elapsed=datetime.now()-start_time
      _text='{}'.format(time_elapsed)
      text=_text[:-3]
      print("%2d:%13d%13d%20s" % (size,self.total,self.unique,text)); 

class NQueens03:
  def __init__(self):
    self.max=15;
    self.total=0;
    self.unique=0;
    self.aboard=[0 for i in range(self.max)];
    self.fa=[0 for i in range(2*self.max-1)];
    self.fb=[0 for i in range(2*self.max-1)];
    self.fc=[0 for i in range(2*self.max-1)];
  def nqueens(self,row,size):
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
    min=4;
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(min,self.max):
      self.total=0;
      self.unique=0;
      for j in range(size):
        self.aboard[j]=j;
      start_time=datetime.now();
      self.nqueens(0,size);
      time_elapsed=datetime.now()-start_time;
      _text='{}'.format(time_elapsed);
      text=_text[:-3]
      print("%2d:%13d%13d%20s" % (size,self.total,self.unique,text)); 

class NQueens02:
  def __init__(self):
    self.size=8;
    self.count=0;
    self.aboard=[0 for i in range(self.size)];
    self.fa=[0 for i in range(self.size)];
  def printout(self):
    self.count+=1;
    print(self.count,end=": ");
    for i in range(self.size):
      print(self.aboard[i],end="");
    print("");
  def nqueens(self,row):
    if row==self.size-1:
      self.printout();
    else:
      for i in range(self.size):
        self.aboard[row]=i;
        if self.fa[i]==0:
          self.fa[i]=1;
          self.nqueens(row+1);
          self.fa[i]=0;

class NQueens01:
  def __init__(self):
    self.size=8
    self.aboard=[0 for i in range(self.size)]
    self.count=0 
  def printout(self):
    self.count+=1;
    print(self.count, end=": ");
    for i in range(self.size):
      print(self.aboard[i], end="");
    print("");
  def nqueens(self,row):
    if row is self.size:
      self.printout();
    else:
      for i in range(self.size):
        self.aboard[row]=i;
        self.nqueens(row+1);
#
# 枝刈りと最適化
NQueens05().main();
#
# 対象解除法
#NQueens04().main();
#
# バックトラック
# NQueens03().main();
#
# 対象解除法
# NQueens04().main();
#
# 配置フラグ
# NQueens02().nqueens(0);
#
# 対象解除法
# NQueens04().main();
#
# ブルートフォース　ちからまかせ探索
# NQueens01().nqueen(0);
#
