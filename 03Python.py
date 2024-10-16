#!/usr/bin/env python

# -*- coding: utf-8 -*-
from datetime import datetime

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
    for i in range(min,self.max):
      self.total=0;
      self.unique=0;
      for j in range(i):
        self.aboard[j]=j;
      start_time=datetime.now();
      self.nqueens(0,i);
      time_elapsed=datetime.now()-start_time;
      _text='{}'.format(time_elapsed);
      text=_text[:-3]
      print("%2d:%13d%13d%20s" % (i,self.total,self.unique,text)); 

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

# バックトラック
NQueens03().main();

# 配置フラグ
# NQueens02().nqueens(0);

# ブルートフォース　ちからまかせ探索
# NQueens01().nqueen(0);
#
