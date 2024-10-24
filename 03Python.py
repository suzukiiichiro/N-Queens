#!/usr/bin/env python

# -*- coding: utf-8 -*-
import logging
import threading
from threading import Thread
from multiprocessing import Pool as ThreadPool
from datetime import datetime

class NQueens():
  def __init__(self,size):
    self.size=size
    self.sizeE=size-1
    self.total=0
    self.unique=0
    self.gttotal=[0]*self.size
    self.gtunique=[0]*self.size
    self.aboard=[[i for i in range(2*size-1)]for j in range(self.size)]
    self.mask=(1<<size)-1
    self.count2=0
    self.count4=0
    self.count8=0
    self.bound1=0
    self.bound2=0
    self.sidemask=0
    self.lastmask=0
    self.topbit=0
    self.endbit=0
  def getunique(self):
    return self.count2+self.count4+self.count8
  def gettotal(self):
    return self.count2*2+self.count4*4+self.count8*8
  def symmetryops(self):
    own=0
    ptn=0
    you=0
    bit=0
    #90
    if self.aboard[self.bound2]==1:
      own=1
      ptn=2
      while own<=self.size-1:
        bit=1
        you=self.size-1
        while(self.aboard[you]!=ptn)and(self.aboard[own]>=bit):
          bit<<=1
          you-=1
        if self.aboard[own]>bit:
          return
        if self.aboard[own]<bit:
          break
        own+=1
        ptn<<=1
      #90 180/270
      if own>self.size-1:
        self.count2+=1
        return
    #180
    if self.aboard[self.size-1]==self.endbit:
      own=1
      you=self.size-1-1
      while own<=self.size-1:
        bit=1
        ptn=self.topbit
        while(self.aboard[you]!=ptn)and(self.aboard[own]>=bit):
          bit<<=1
          ptn>>=1
        if self.aboard[own]>bit:
          return
        if self.aboard[own]<bit:
          break
        own+=1
        you-=1
      #90 
      if own>self.size-1:
        self.count4+=1
        return
    #270
    if self.aboard[self.bound1]==self.topbit:
      own=1
      ptn=self.topbit>>1
      while own<=self.size-1:
        bit=1
        you=0
        while(self.aboard[you]!=ptn)and(self.aboard[own]>=bit):
          bit<<=1
          you+=1
        if self.aboard[own]>bit:
          return
        if self.aboard[own]<bit:
          break
        own+=1
        ptn>>=1
    self.count8+=1
  def backTrack2(self,row,left,down,right):
    bit=0
    bitmap=self.mask&~(left|down|right)
    if row==self.size-1:
      if bitmap:
        if(bitmap&self.lastmask)==0:
          self.aboard[row]=bitmap
          self.symmetryops()
    else:
      if row<self.bound1:
        bitmap&=~self.sidemask
      elif row==self.bound2:
        if down&self.sidemask==0:
          return
        if down&self.sidemask!=self.sidemask:
          bitmap&=self.sidemask
      if row!=0:
        lim=self.size
      else:
        lim=(self.size+1)//2
      for i in range(row,lim):
        while bitmap:
          bit=(-bitmap&bitmap)
          self.aboard[row]=bit
          bitmap^=self.aboard[row]
          self.backTrack2(row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def backTrack1(self,row,left,down,right):
    bit=0
    bitmap=self.mask&~(left|down|right)
    if row==self.size-1:
      if bitmap:
        self.aboard[row]=bitmap
        self.count8+=1
    else:
      if row<self.bound1:
        bitmap&=~2
      if row!=0:
        lim=self.size
      else:
        lim=(self.size+1)//2
      for i in range(row,lim):
        while bitmap:
          bit=(-bitmap&bitmap)
          self.aboard[row]=bit
          bitmap^=self.aboard[row]
          self.backTrack1(row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def nqueen_single(self,thr_index):
    self.bit=0
    self.aboard[0]=1
    self.sizeE=self.size-1
    self.mask=(1<<self.size)-1
    self.topbit=1<<self.sizeE
    self.bound1=1
    for self.bound1 in range(2,self.sizeE):
      self.aboard[1]=bit=(1<<self.bound1)
      self.backTrack1(2,(2|bit)<<1,1|bit,bit>>1)
      self.bound1+=1
    self.sidemask=self.lastmask=(self.topbit|1)
    self.endbit=(self.topbit>>1)
    self.bound1=1
    self.bound2=self.sizeE-1
    for self.bound1 in range(1,self.bound2):
      self.aboard[0]=bit=(1<<self.bound1)
      self.backTrack2(1,bit<<1,bit,bit>>1)
      self.lastmask|=self.lastmask>>1|self.lastmask<<1
      self.endbit>>=1
      self.bound1+=1
      self.bound2-=1
    return self.gettotal(),self.getunique()
  def nqueen_multi(self,thr_index):
    self.aboard[0]=1
    self.sizeE=self.size-1
    self.topbit=1<<self.sizeE
    self.bound1=self.size-thr_index-1
    if self.bound1>1 and self.bound1<self.sizeE:
      self.aboard[1]=bit=(1<<self.bound1)
      self.backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1))
    self.endbit=(self.topbit>>1)
    self.sidemask=self.lastmask=(self.topbit|1)
    self.bound2=thr_index
    if self.bound1>0 and self.bound2<self.sizeE and self.bound1<self.bound2:
      self.aboard[0]=bit=(1<<self.bound1)
      for i in range(1,self.bound1):
        self.lastmask|=self.lastmask>>1|self.lastmask<<1
        self.endbit>>=1
      self.backTrack2(1,bit<<1,bit,bit>>1)
    return self.gettotal(),self.getunique()
  def solve(self):
    pool=ThreadPool(self.size)

    # シングル版 Nで割ると解が合う
    # gttotal:[(92, 12), (92, 12), (92, 12), (92, 12), (92, 12), (92, 12), (92, 12), (92, 12)]
    # 8:          736           96         0:00:00.119
    #self.gttotal=list(pool.map(self.nqueen_single,range(self.size)))
    # マルチ版
    self.gttotal=list(pool.map(self.nqueen_multi,range(self.size)))
    total=0
    unique=0
    for t,u in self.gttotal:
      total+=t
      unique+=u
    pool.close()
    pool.join()
    return total,unique
class NQueens13_3multiProcess:
  def main():
    nmin = 4
    nmax = 16
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for i in range(nmin, nmax):
      start_time = datetime.now()
      nqueen_obj = NQueens(i)
      total, unique = nqueen_obj.solve()
      time_elapsed = datetime.now()-start_time
      _text = '{}'.format(time_elapsed)
      text = _text[:-3]
      print("%2d:%13d%13d%20s" % (i,total,unique, text))  # 出力


class Board:
  def __init__(self,lock):
    self.count2=0
    self.count4=0
    self.count8=0
    self.lock=lock
  def setcount(self,count8,count4,count2):
    with self.lock:
      self.count8+=count8
      self.count4+=count4
      self.count2+=count2
  def getunique(self):
    return self.count2+self.count4+self.count8
  def gettotal(self):
    return self.count2*2+self.count4*4+self.count8*8
class multiThreadWorkingEngine(Thread):
  logging.basicConfig(level=logging.DEBUG,
      format='[%(levelname)s](%(threadName)-10s) %(message)s', )
  def __init__(self,size,nmore,info,B1,B2,bthread):
    super(multiThreadWorkingEngine,self).__init__()
    self.bthread=bthread
    self.size=size
    self.sizee=size-1
    self.aboard=[0 for i in range(size)]
    self.mask=(1<<size)-1
    self.info=info
    self.nmore=nmore
    self.child=None
    self.bound1=B1
    self.bound2=B2
    self.topbit=0
    self.endbit=0
    self.sidemask=0
    self.lastmask=0
    for i in range(size):
      self.aboard[i]=i
    if nmore>0:
      if bthread: # マルチスレッド
        self.child=multiThreadWorkingEngine(size,nmore-1,info,B1-1,B2+1,bthread)
        self.bound1=B1
        self.bound2=B2
        self.child.start()
        self.child.join()
      else: # シングルスレッド
        self.child=None
  def run(self):
    if self.child is None: # シングルスレッド
      if self.nmore>0:
        self.aboard[0]=1
        self.sizee=self.size-1
        self.topbit=1<<(self.size-1)
        self.bound1=2
        while(self.bound1>1)and(self.bound1<self.sizee):
          self.rec_bound1(self.bound1)
          self.bound1+=1
        self.sidemask=self.lastmask=(self.topbit|1)
        self.endbit=(self.topbit>>1)
        self.bound1=1
        self.bound2=self.size-2
        while(self.bound1>0)and(self.bound2<self.size-1)and(self.bound1<self.bound2):
          self.rec_bound2(self.bound1,self.bound2)
          self.bound1+=1
          self.bound2-=1
    else: # マルチスレッド
      self.aboard[0]=1
      self.sizee=self.size-1
      self.mask=(1<<self.size)-1
      self.topbit=(1<<self.sizee)
      if(self.bound1>1)and(self.bound1<self.sizee):
        self.rec_bound1(self.bound1)
      self.endbit=(self.topbit>>self.bound1)
      self.sidemask=self.lastmask=(self.topbit|1)
      if(self.bound1>0)and(self.bound2<self.size-1)and(self.bound1<self.bound2):
        for i in range(1,self.bound1):
          self.lastmask=self.lastmask|self.lastmask>>1|self.lastmask<<1
        self.rec_bound2(self.bound1,self.bound2)
        self.endbit>>=self.nmore
  def symmetryops(self):
    # 90
    if self.aboard[self.bound2]==1:
      own=1
      ptn=2
      while own<=self.size-1:
        bit=1
        you=self.size-1
        while(self.aboard[you]!=ptn)and(self.aboard[own]>=bit):
          bit<<=1
          you-=1
        if self.aboard[own]>bit:
          return
        if self.aboard[own]<bit:
          break
        own+=1
        ptn<<=1
      # not 90 / 180/270
      if own>self.size-1:
        self.info.setcount(0,0,1)
        return 
    # 180
    if self.aboard[self.size-1]==self.endbit:
      own=1
      you=self.size-1-1
      while own<=self.size-1:
        bit=1
        ptn=self.topbit
        while(self.aboard[you]!=ptn)and(self.aboard[own]>=bit):
          bit<<=1
          ptn>>=1
        if self.aboard[own]>bit:
          return
        if self.aboard[own]<bit:
          break
        own+=1
        you-=1
      #not 90 180
      if own>self.size-1:
        self.info.setcount(0,1,0)
        return
    #270
    if self.aboard[self.bound1]==self.topbit:
      own=1
      ptn=self.topbit>>1
      while own<=self.size-1:
        bit=1
        you=0
        while(self.aboard[you]!=ptn)and(self.aboard[own]>=bit):
          bit<<=1
          you+=1
        if self.aboard[own]>bit:
          return
        if self.aboard[own]<bit:
          break
        own+=1
        ptn>>=1
    self.info.setcount(1,0,0)
  def backTrack2(self,row,left,down,right):
    bitmap=self.mask&~(left|down|right)
    if row==self.size-1:
      if bitmap:
        if(bitmap&self.lastmask)==0:
          self.aboard[row]=bitmap
          self.symmetryops()
    else:
      if row<self.bound1:
        bitmap&=~self.sidemask
      elif row==self.bound2:
        if down&self.sidemask==0:
          return
        if down&self.sidemask!=self.sidemask:
          bitmap&=self.sidemask
      if row!=0:
        lim=self.size
      else:
        lim=(self.size+1)//2
      for i in range(row,lim):
        while bitmap:
          bit=(-bitmap&bitmap)
          self.aboard[row]=bit
          bitmap^=self.aboard[row]
          self.backTrack2(row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def backTrack1(self,row,left,down,right):
    bit=0
    bitmap=self.mask&~(left|down|right)
    if row==self.size-1:
      if bitmap:
        self.aboard[row]=bitmap
        self.info.setcount(1,0,0)
    else:
      if row<self.bound1:
        bitmap&=~2 # bm|=2 bm^=2 と同等
      if row!=0:
        lim=self.size
      else:
        lim=(self.size+1)//2
      for i in range(row,lim):
        while bitmap:
          bit=(-bitmap&bitmap)
          self.aboard[row]=bit
          bitmap^=self.aboard[row]
          self.backTrack1(row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def rec_bound2(self,bound1,bound2):
    self.bound1=bound1
    self.bound2=bound2
    self.aboard[0]=bit=(1<<bound1)
    self.backTrack2(1,bit<<1,bit,bit>>1)
    self.lastmask|=self.lastmask>>1|self.lastmask<<1
    self.endbit>>=1
  def rec_bound1(self,bound1):
    self.bound1=bound1
    self.aboard[1]=bit=(1<<bound1)
    self.backTrack1(2,(2|bit)<<1,1|bit,bit>>1)
  def nqueens(self):
    if self.child is None:
      if self.nmore>0:
        self.topbit=1<<(self.size-1)
        self.aboard[0]=1
        self.sizee=self.size-1
        self.bound1=2
        while(self.bound1>1)and(self.bound1<self.sizee):
          self.rec_bound1(self.bound1)
          self.bound1+=1
        self.sidemask=self.lastmask=(self.topbit|1)
        self.endbit=(self.topbit>>1)
        self.bound2=self.size-2
        self.bound1=1
        while(self.bound1>0)and(self.bound2<self.size-1)and(self.bound1<self.bound2):
          self.rec_bound2(self.bound1,self.bound2)
          self.bound1+=1
          self.bound2-=1
class NQueens13_2_multiThread:
  def main():
    nmax = 16
    nmin = 4  
    if BTHREAD:
      print("マルチスレッド")
    else:
      print("シングルスレッド")
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for i in range(nmin, nmax):
      lock = threading.Lock()
      info = Board(lock)
      start_time = datetime.now()
      child = multiThreadWorkingEngine(i, i, info, i - 1, 0,BTHREAD)
      child.start()
      child.join()
      time_elapsed = datetime.now() - start_time
      _text = '{}'.format(time_elapsed)
      text = _text[:-3]
      print("%2d:%13d%13d%20s" % (i, info.gettotal(), info.getunique(), text))  # 出力

class singleThreadWorkingEngine(Thread):
  logging.basicConfig(level=logging.DEBUG,
      format='[%(levelname)s](%(threadName)-10s) %(message)s', )
  def __init__(self,size,nmore,info,B1,B2):
    super(singleThreadWorkingEngine,self).__init__()
    self.size=size
    self.sizee=size-1
    self.aboard=[0 for i in range(size)]
    self.mask=(1<<size)-1
    self.info=info
    self.nmore=nmore
    self.child=None
    self.bound1=B1
    self.bound2=B2
    self.topbit=0
    self.endbit=0
    self.sidemask=0
    selflastmask=0
    for i in range(size):
      self.aboard[i]=i
  def run(self):
    if self.child is None:
      if self.nmore>0:
        self.aboard[0]=1
        self.sizee=self.size-1
        self.topbit=1<<(self.size-1)
        self.bound1=2
        while(self.bound1>1)and(self.bound1<self.sizee):
          self.rec_bound1(self.bound1)
          self.bound1+=1
        self.sidemask=self.lastmask=(self.topbit|1)
        self.endbit=(self.topbit>>1)
        self.bound1=1
        self.bound2=self.size-2
        while(self.bound1>0)and(self.bound2<self.size-1)and(self.bound1<self.bound2):
          self.rec_bound2(self.bound1,self.bound2)
          self.bound1+=1
          self.bound2-=1
  def symmetryops(self):
    #90
    if self.aboard[self.bound2]==1:
      own=1
      ptn=2
      while own<=self.size-1:
        bit=1
        you=self.size-1
        while(self.aboard[you]!=ptn)and(self.aboard[own]>=bit):
          bit<<=1
          you-=1
        if self.aboard[own]>bit:
          return 
        if self.aboard[own]<bit:
          break
        own+=1
        ptn<<=1
      #90
      if own>self.size-1:
        self.info.setcount(0,0,1)
        return 
    #180
    if self.aboard[self.size-1]==self.endbit:
      own=1
      you=self.size-1-1
      while own<=self.size-1:
        bit=1
        ptn=self.topbit
        while(self.aboard[you]!=ptn)and(self.aboard[own]>=bit):
          bit<<=1
          ptn>>=1
        if self.aboard[own]>bit:
          return 
        if self.aboard[own]<bit:
          break
        own+=1
        you-=1
      #not 90 ok 180
      if own>self.size-1:
        self.info.setcount(0,1,0)
        return
    #270
    if self.aboard[self.bound1]==self.topbit:
      own=1
      ptn=self.topbit>>1
      while own<=self.size-1:
        bit=1
        you=0
        while(self.aboard[you]!=ptn)and(self.aboard[own]>=bit):
          bit<<=1
          you+=1
        if self.aboard[own]>bit:
          return
        if self.aboard[own]<bit:
          break
        own+=1
        ptn>>=1
    self.info.setcount(1,0,0)
  def backTrack2(self,row,left,down,right):
    bitmap=self.mask&~(left|down|right)
    if row==self.size-1:
      if bitmap:
        if(bitmap&self.lastmask)==0:
          self.aboard[row]=bitmap
          self.symmetryops()
    else:
      if row<self.bound1:
        bitmap&=~self.sidemask
      elif row==self.bound2:
        if down&self.sidemask==0:
          return
        if down&self.sidemask!=self.sidemask:
          bitmap&=self.sidemask
      if row!=0:
        lim=self.size
      else:
        lim=(self.size+1)//2
      for i in range(row,lim):
        while bitmap:
          bit=(-bitmap&bitmap)
          self.aboard[row]=bit
          bitmap^=self.aboard[row]
          self.backTrack2(row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def backTrack1(self,row,left,down,right):
    bitmap=self.mask&~(left|down|right)
    if row==self.size-1:
      if bitmap:
        self.aboard[row]=bitmap
        self.info.setcount(1,0,0)
    else:
      if row<self.bound1:
        bitmap&=~2
      if row!=0:
        lim=self.size
      else:
        lim=(self.size+1)//2
      for i in range(row,lim):
        while bitmap:
          bit=(-bitmap&bitmap)
          self.aboard[row]=bit
          bitmap^=self.aboard[row]
          self.backTrack1(row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def rec_bound2(self,bound1,bound2):
    self.bound1=bound1
    self.bound2=bound2
    self.aboard[0]=bit=(1<<bound1)
    self.backTrack2(1,bit<<1,bit,bit>>1)
    self.lastmask|=self.lastmask>>1|self.lastmask<<1
    self.endbit>>=1
  def rec_bound1(self,bound1):
    self.bound1=bound1
    self.aboard[1]=bit=(1<<bound1)
    self.backTrack1(2,(2|bit)<<1,1|bit,bit>>1)
  def nqueens(self):
    if self.child is None:
      if self.nmore>0:
        self.topbit=1<<(self.size-1)
        self.aboard[0]=1
        self.sizee=self.size-1
        self.bound1=2
        while(self.bound1>1)and(self.bound1<self.sizee):
          self.rec_bound1(self.bound1)
          self.bound1+=1
        self.sidemask=self.lastmask=(self.topbit|1)
        self.endbit=(self.topbit>>1)
        self.bound2=self.size-2
        self.bound1=1
        while(self.bound1>0)and(self.bound2<self.size-1)and(self.bound1<self.bound2):
          self.rec_bound2(self.bound1,self.bound2)
          self.bound1+=1
          self.bound2-=1
class NQueens13_1_singleThread:
  def main():
    nmax = 16
    nmin = 4  # Nの最小値（スタートの値）を格納
    print("シングルスレッド")
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for i in range(nmin, nmax):
      lock = threading.Lock()
      info = Board(lock)
      start_time = datetime.now()
      child = singleThreadWorkingEngine(i, i, info, i - 1, 0)
      child.start()
      child.join()
      time_elapsed = datetime.now() - start_time
      _text = '{}'.format(time_elapsed)
      text = _text[:-3]
      print("%2d:%13d%13d%20s" % (i, info.gettotal(), info.getunique(), text))  # 出力


class NQueens12:
  def __init__(self):
    self.max=16
    self.aboard=[0 for i in range(self.max)]
    self.count2=0
    self.count4=0
    self.count8=0
    self.topbit=0
    self.endbit=0
    self.sidemask=0
    self.lastmask=0
    self.bound1=0
    self.bound2=0
  def symmetryops(self,size):
    own=0
    ptn=0
    you=0
    bit=0
    #90
    if self.aboard[self.bound2]==1:
      own=1
      ptn=2
      while own<=size-1:
        bit=1
        you=size-1
        while(self.aboard[you]!=ptn)and(self.aboard[own]>=bit):
          bit<<=1
          you-=1
        if self.aboard[own]>bit:
          return 
        if self.aboard[own]<bit:
          break
        own+=1
        ptn<<=1
      if own>size-1:
        self.count2+=1
        return 
    #180
    if self.aboard[size-1]==self.endbit:
      own=1
      you=size-1-1
      while own<=size-1:
        bit=1
        ptn=self.topbit
        while(self.aboard[you]!=ptn)and(self.aboard[own]>=bit):
          bit<<=1
          ptn>>=1
        if self.aboard[own]>bit:
          return 
        if self.aboard[own]<bit:
          break
        own+=1
        you-=1
      if own>size-1:
        self.count4+=1
        return 
    #270
    if self.aboard[self.bound1]==self.topbit:
      own=1
      ptn=self.topbit>>1
      while own<=size-1:
        bit=1
        you=0
        while(self.aboard[you]!=ptn)and(self.aboard[own]>=bit):
          bit<<=1
          you+=1
        if self.aboard[own]>bit:
          return
        if self.aboard[own]<bit:
          break
        own+=1
        ptn>>=1
    self.count8+=1
  def backTrack2(self,size,row,left,down,right):
    bit=0
    mask=(1<<size)-1
    bitmap=mask&~(left|down|right)
    if row==size-1:
      if bitmap:
        if (bitmap&self.lastmask)==0:
          self.aboard[row]=bitmap
          self.symmetryops(size)
    else:
      if row<self.bound1:
        bitmap&=~self.sidemask
      elif row==self.bound2:
        if down&self.sidemask==0:
          return
        if down&self.sidemask!=self.sidemask:
          bitmap&=self.sidemask
      if row!=0:
        lim=size
      else:
        lim=(size+1)//2
      for i in range(row,lim):
        while bitmap:
          bit=(-bitmap&bitmap)
          self.aboard[row]=bit
          bitmap^=self.aboard[row]
          self.backTrack2(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def backTrack1(self,size,row,left,down,right):
    bit=0
    mask=(1<<size)-1
    bitmap=mask&~(left|down|right)
    if row==size-1:
      if bitmap:
        self.aboard[row]=bitmap
        self.count8+=1
    else:
      if row<self.bound1:
        bitmap&=~2 # bitmap|=2
                   # bitmap^=2
                   #  ↓
                   # bitmap &=~2
      if row!=0:
        lim=size
      else:
        lim=(size+1)//2
      for i in range(row,lim):
        while bitmap:
          bit=(-bitmap&bitmap)
          self.aboard[row]=bit
          bitmap^=self.aboard[row]
          self.backTrack1(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def nqueens(self,size):
    bit=0
    self.topbit=1<<(size-1)
    self.aboard[0]=1
    for self.bound1 in range(2,size-1):
      self.aboard[1]=bit=(1<<self.bound1)
      self.backTrack1(size,2,(2|bit)<<1,(1|bit),(bit>>1))
    self.sidemask=self.lastmask=(self.topbit|1)
    self.endbit=(self.topbit>>1)
    self.bound2=size-2
    for self.bound1 in range(1,self.bound2):
      self.aboard[0]=bit=(1<<self.bound1)
      self.backTrack2(size,1,bit<<1,bit,bit>>1)
      self.lastmask|=self.lastmask>>1|self.lastmask<<1
      self.endbit>>=1
      self.bound2-=1
  def main(self):
    nmin=4
    nmax=16
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(nmin,nmax):
      self.total=0
      self.unique=0
      self.count2=0
      self.count4=0
      self.count8=0
      for j in range(size):
        self.aboard[j]=j
      start_time=datetime.now()
      self.nqueens(size)
      time_elapsed=datetime.now()-start_time
      _text='{}'.format(time_elapsed)
      text=_text[:-3]
      self.total=self.count2*2+self.count4*4+self.count8*8
      self.unique=self.count2+self.count4+self.count8
      print("%2d:%13d%13d%20s" % (size,self.total,self.unique,text)); 
      
class NQueens11:
  def __init__(self):
    self.max=16
    self.aboard=[0 for i in range(self.max)]
    self.trial=[0 for i in range(self.max)]
    self.scratch=[0 for i in range(self.max)]
    self.count2=0
    self.count4=0
    self.count8=0
    self.topbit=0
    self.endbit=0
    self.sidemask=0
    self.lastmask=0
    self.bound1=0
    self.bound2=0
  def rha(self,ah,size):
    tmp=0
    for i in range(size+1):
      if ah&(1<<i):
        tmp|=(1<<size-i)
    return tmp
  def vmirror_bitmap(self,bf,af,size):
    score=0
    for i in range(size):
      score=bf[i]
      af[i]=self.rha(score,size-1)
  def rotate_bitmap(self,bf,af,size):
    for i in range(size):
      tmp=0
      for j in range(size):
        tmp|=((bf[j]>>i)&1)<<(size-j-1)
      af[i]=tmp
  def intncmp(self,lt,rt,neg):
    rtn=0
    for i in range(neg):
      rtn=lt[i]-rt[i]
      if rtn!=0:
        break
    return rtn
  def symmetryops_bitmap(self,size):
    nequiv=0
    for i in range(size):
      self.trial[i]=self.aboard[i]
    self.rotate_bitmap(self.trial,self.scratch,size)
    k=self.intncmp(self.aboard,self.scratch,size)
    if k>0:
      return 
    if k==0:
      nequiv=2
    else:
      self.rotate_bitmap(self.scratch,self.trial,size)
      k=self.intncmp(self.aboard,self.trial,size)
      if k>0:
        return
      if k==0:
        nequiv=4
      else:
        self.rotate_bitmap(self.trial,self.scratch,size)
        k=self.intncmp(self.aboard,self.scratch,size)
        if k>0:
          return 
        nequiv=8
    for i in range(size):
      self.scratch[i]=self.aboard[i]
    self.vmirror_bitmap(self.scratch,self.trial,size)
    k=self.intncmp(self.aboard,self.trial,size)
    if k>0:
      return 
    if nequiv>2:
      self.rotate_bitmap(self.trial,self.scratch,size)
      k=self.intncmp(self.aboard,self.scratch,size)
      if k>0:
        return 
      if nequiv>4:
        self.rotate_bitmap(self.scratch,self.trial,size)
        k=self.intncmp(self.aboard,self.trial,size)
        if k>0:
          return 
        self.rotate_bitmap(self.trial,self.scratch,size)
        k=self.intncmp(self.aboard,self.scratch,size)
        if k>0:
          return
    if nequiv==2:
      self.count2+=1
    if nequiv==4:
      self.count4+=1
    if nequiv==8:
      self.count8+=1
  def backTrack2(self,size,row,left,down,right):
    bit=0
    mask=(1<<size)-1
    bitmap=mask&~(left|down|right)
    if row==size-1:
      if bitmap:
        if (bitmap&self.lastmask)==0:
          self.aboard[row]=bitmap
          self.symmetryops_bitmap(size)
    else:
      if row<self.bound1:
        bitmap&=~self.sidemask
      elif row==self.bound2:
        if down&self.sidemask==0:
          return 
        if down&self.sidemask!=self.sidemask:
          bitmap&=self.sidemask
      if row!=0:
        lim=size
      else:
        lim=(size+1)//2
      for i in range(row,lim):
        while bitmap:
          bit=(-bitmap&bitmap)
          self.aboard[row]=bit
          bitmap^=self.aboard[row]
          self.backTrack2(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def backTrack1(self,size,row,left,down,right):
    bit=0
    mask=(1<<size)-1
    bitmap=mask&~(left|down|right)
    if row==size-1:
      if bitmap:
        self.aboard[row]=bitmap
        self.count8+=1
    else:
      if row<self.bound1:
        bitmap&=~2
      if row!=0:
        lim=size
      else:
        lim=(size+1)//2
      for i in range(row,lim):
        while bitmap:
          bit=(-bitmap&bitmap)
          self.aboard[row]=bit
          bitmap^=self.aboard[row]
          self.backTrack1(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def nqueens(self,size):
    bit=0
    self.topbit=1<<(size-1)
    self.aboard[0]=1
    for self.bound1 in range(2,size-1):
      self.aboard[1]=bit=(1<<self.bound1)
      self.backTrack1(size,2,(2|bit)<<1,1|bit,bit>>1)
    self.sidemask=self.lastmask=(self.topbit|1)
    self.endbit=(self.topbit>>1)
    self.bound2=size-2
    for self.bound1 in range(1,self.bound2):
      self.aboard[0]=bit=(1<<self.bound1)
      self.backTrack2(size,1,bit<<1,bit,bit>>1)
      self.lastmask|=self.lastmask>>1|self.lastmask<<1
      self.endbit>>=1
      self.bound2-=1
  def main(self):
    nmin=4
    nmax=16
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(nmin,nmax):
      self.total=0
      self.unique=0
      self.count2=0
      self.count4=0
      self.count8=0
      for j in range(size):
        self.aboard[j]=j
      start_time=datetime.now()
      self.nqueens(size)
      time_elapsed=datetime.now()-start_time
      _text='{}'.format(time_elapsed)
      text=_text[:-3]
      self.total=self.count2*2+self.count4*4+self.count8*8
      self.unique=self.count2+self.count4+self.count8
      print("%2d:%13d%13d%20s" % (size,self.total,self.unique,text)); 
          

class NQueens10:
  def __init__(self):
    self.max=16
    self.aboard=[0 for i in range(self.max)]
    self.trial=[0 for i in range(self.max)]
    self.scratch=[0 for i in range(self.max)]
    self.count2=0
    self.count4=0
    self.count8=0
    self.topbit=0
    self.endbit=0
    self.sidemask=0
    self.lastmask=0
    self.bound1=0
    self.bound2=0
  def rha(self,ah,size):
    tmp=0
    for i in range(size+1):
      if ah&(1<<i):
        tmp|=(1<<size-i)
    return tmp
  def vmirror_bitmap(self,bf,af,size):
    socre=0
    for i in range(size):
      score=bf[i]
      af[i]=self.rha(score,size-1)
  def rotate_bitmap(self,bf,af,size):
    for i in range(size):
      tmp=0
      for j in range(size):
        tmp|=((bf[j]>>i)&1)<<(size-j-1)
      af[i]=tmp
  def intncmp(self,lt,rt,neg):
    rtn=0
    for i in range(neg):
      rtn=lt[i]-rt[i]
      if rtn!=0:
        break
    return rtn
  def symmetryops_bitmap(self,size):
    nequiv=0
    for i in range(size):
      self.trial[i]=self.aboard[i]
    self.rotate_bitmap(self.trial,self.scratch,size)
    k=self.intncmp(self.aboard,self.scratch,size)
    if k>0:
      return 
    if k==0:
      nequiv=2
    else:
      self.rotate_bitmap(self.scratch,self.trial,size)
      k=self.intncmp(self.aboard,self.trial,size)
      if k>0:
        return
      if k==0:
        nequiv=4
      else:
       self.rotate_bitmap(self.trial,self.scratch,size)
       k=self.intncmp(self.aboard,self.scratch,size)
       if k>0:
         return
       nequiv=8
    for i in range(size):
      self.scratch[i]=self.aboard[i]
    self.vmirror_bitmap(self.scratch,self.trial,size)
    k=self.intncmp(self.aboard,self.trial,size)
    if k>0:
      return 
    if nequiv>2:
      self.rotate_bitmap(self.trial,self.scratch,size)
      k=self.intncmp(self.aboard,self.scratch,size)
      if k>0:
        return 
      if nequiv>4:
        self.rotate_bitmap(self.scratch,self.trial,size)
        k=self.intncmp(self.aboard,self.trial,size)
        if k>0:
          return 
        self.rotate_bitmap(self.trial,self.scratch,size)
        k=self.intncmp(self.aboard,self.scratch,size)
        if k>0:
          return 
    if nequiv==2:
      self.count2+=1
    if nequiv==4:
      self.count4+=1
    if nequiv==8:
      self.count8+=1
  def backTrack2(self,size,row,left,down,right):
    bit=0
    mask=(1<<size)-1
    bitmap=mask&~(left|down|right)
    if row == size:
      if bitmap:
        pass
      else:
        self.aboard[row]=bitmap
        self.symmetryops_bitmap(size)
    else:
      if row!=0:
        lim=size
      else:
        lim=(size+1)//2
      for i in range(row,lim):
        while bitmap>0:
          bit=(-bitmap&bitmap)
          self.aboard[row]=bit
          bitmap^=self.aboard[row]
          self.backTrack2(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def backTrack1(self,size,row,left,down,right):
    bit=0
    mask=(1<<size)-1
    bitmap=mask&~(left|down|right)
    if row==size:
      if bitmap:
        pass
      else:
        self.aboard[row]=bitmap
        self.symmetryops_bitmap(size)
    else:
      if row!=0:
        lim=size
      else:
        lim=(size+1)//2
      for i in range(row,lim):
        while bitmap:
          bit=(-bitmap&bitmap)
          self.aboard[row]=bit
          bitmap^=self.aboard[row]
          self.backTrack1(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def nqueens(self,size):
    bit=0
    self.topbit=1<<(size-1)
    self.aboard[0]=1
    for self.bound1 in range(2,size-1):
      self.aboard[1]=bit=(1<<self.bound1)
      self.backTrack1(size,2,(2|bit)<<1,(1|bit),(bit>>1))
    self.sidemask=self.lastmask=(self.topbit|1)
    self.endbit=(self.topbit>>1)
    self.bound2=size-2
    for self.bound1 in range(1,self.bound2):
      self.aboard[0]=bit=(1<<self.bound1)
      self.backTrack2(size,1,bit<<1,bit,bit>>1)
      self.lastmask|=self.lastmask>>1|self.lastmask<<1
      self.endbit>>=1
      self.bound2-=1
  def main(self):
    nmin=4
    nmax=16
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(nmin,nmax):
      self.total=0
      self.unique=0
      self.count2=0
      self.count4=0
      self.count8=0
      for j in range(size):
        self.aboard[j]=j
      start_time=datetime.now()
      self.nqueens(size)
      time_elapsed=datetime.now()-start_time
      _text='{}'.format(time_elapsed)
      text=_text[:-3]
      self.total=self.count2*2+self.count4*4+self.count8*8
      self.unique=self.count2+self.count4+self.count8
      print("%2d:%13d%13d%20s" % (size,self.total,self.unique,text)); 





      






class NQueens09:
  def __init__(self):
    self.max=16
    self.aboard=[0 for i in range(self.max)]
    self.trial=[0 for i in range(self.max)]
    self.scratch=[0 for i in range(self.max)]
    self.count2=0
    self.count4=0
    self.count8=0
    self.topbit=0
    self.endbit=0
    self.sidemask=0
    self.lastmask=0
  def rha(self,ah,size):
    tmp=0
    for i in range(size+1):
      if ah&(1<<i):
        tmp|=(1<<size-i)
    return tmp
  def vmirror_bitmap(self,bf,af,size):
    score=0
    for i in range(size):
      score=bf[i]
      af[i]=self.rha(score,size-1)
  def rotate_bitmap(self,bf,af,size):
    for i in range(size):
      tmp=0
      for j in range(size):
        tmp|=((bf[j]>>i)&1)<<(size-j-1)
      af[i]=tmp
  def intncmp(self,lt,rt,neg):
    rtn=0
    for i in range(neg):
      rtn=lt[i]-rt[i]
      if rtn!=0:
        break
    return rtn
  def symmetryops_bitmap(self,size):
    nequiv=0
    for i in range(size):
      self.trial[i]=self.aboard[i]
    # 90
    self.rotate_bitmap(self.trial,self.scratch,size)
    k=self.intncmp(self.aboard,self.scratch,size)
    if k>0:
      return
    if k==0:
      nequiv=2
    else:
      #180
      self.rotate_bitmap(self.scratch,self.trial,size)
      k=self.intncmp(self.aboard,self.trial,size)
      if k>0:
        return 
      if k==0:
        nequiv=4
      else:
        #270
        self.rotate_bitmap(self.trial,self.scratch,size)
        k=self.intncmp(self.aboard,self.scratch,size)
        if k>0:
          return 
        nequiv=8
    for i in range(size):
      self.scratch[i]=self.aboard[i]
    #垂直反転
    self.vmirror_bitmap(self.scratch,self.trial,size)
    k=self.intncmp(self.aboard,self.trial,size)
    if k>0:
      return 
    if nequiv>2:
      #90
      self.rotate_bitmap(self.trial,self.scratch,size)
      k=self.intncmp(self.aboard,self.scratch,size)
      if k>0:
        return 
      if nequiv>4:
        #180
        self.rotate_bitmap(self.scratch,self.trial,size)
        k=self.intncmp(self.aboard,self.trial,size)
        if k>0:
          return
        #270
        self.rotate_bitmap(self.trial,self.scratch,size)
        k=self.intncmp(self.aboard,self.scratch,size)
        if k>0:
          return 
    if nequiv==2:
      self.count2+=1
    if nequiv==4:
      self.count4+=1
    if nequiv==8:
      self.count8+=1
  def backTrack1(self,size,row,left,down,right):
    bit=0
    mask=(1<<size)-1
    bitmap=mask&~(left|down|right)
    if row==size:
      if bitmap:
        pass
      else:
        self.aboard[row]=bitmap
        self.symmetryops_bitmap(size)
    else:
      if row!=0:
        lim=size
      else:
        lim=(size+1)//2
      for i in range(row,lim):
        while bitmap:
          bit=(-bitmap&bitmap)
          self.aboard[row]=bit
          bitmap^=self.aboard[row]
          self.backTrack1(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def nqueens(self,size):
    bit=0
    self.topbit=1<<(size-1)
    self.aboard[0]=1
    for bound1 in range(2,size-1):
      self.aboard[1]=bit=(1<<bound1)
      self.backTrack1(size,2,(2|bit)<<1,(1|bit),(bit>>1))
    self.sidemask=self.lastmask=(self.topbit|1)
    self.endbit=(self.topbit>>1)
    bound2=size-2
    for bound1 in range(1,bound2):
      self.aboard[0]=bit=(1<<bound1)
      self.backTrack1(size,1,bit<<1,bit,bit>>1)
      self.lastmask|=self.lastmask>>1|self.lastmask<<1
      self.endbit>>=1
      bound2-=1
  def main(self):
    nmin=4
    nmax=16
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(nmin,nmax):
      self.total=0
      self.unique=0
      self.count2=0
      self.count4=0
      self.count8=0
      for j in range(size):
        self.aboard[j]=j
      start_time=datetime.now()
      self.nqueens(size)
      time_elapsed=datetime.now()-start_time
      _text='{}'.format(time_elapsed)
      text=_text[:-3]
      self.total=self.count2*2+self.count4*4+self.count8*8
      self.unique=self.count2+self.count4+self.count8
      print("%2d:%13d%13d%20s" % (size,self.total,self.unique,text)); 

class NQueens08:
  def __init__(self):
    self.max=16
    self.aboard=[0 for i in range(self.max)]
    self.trial=[0 for i in range(self.max)]
    self.scratch=[0 for i in range(self.max)]
  def rha(self,ah,size):
    tmp=0
    for i in range(size+1):
      if ah&(1<<i):
        tmp|=(1<<size-i)
    return tmp
  def vmirror_bitmap(self,bf,af,size):
    score=0
    for i in range(size):
      score=bf[i]
      af[i]=self.rha(score,size-1)
  def rotate_bitmap(self,bf,af,size):
    for i in range(size):
      tmp=0
      for j in range(size):
        tmp|=((bf[j]>>i)&1)<<(size-j-1)
      af[i]=tmp
  def intncmp(self,lt,rt,neg):
    rtn=0
    for i in range(neg):
      rtn=lt[i]-rt[i]
      if rtn!=0:
        break
    return rtn
  def symmetryops_bitmap(self,size):
    nequiv=0
    for i in range(size):
      self.trial[i]=self.aboard[i]
    #90
    self.rotate_bitmap(self.trial,self.scratch,size)
    k=self.intncmp(self.aboard,self.scratch,size)
    if k>0:
      return 
    if k==0:
      nequiv=2
    else:
      #180
      self.rotate_bitmap(self.scratch,self.trial,size)
      k=self.intncmp(self.aboard,self.trial,size)
      if k>0:
        return 
      if k==0:
        nequiv=4
      else:
        #270
        self.rotate_bitmap(self.trial,self.scratch,size)
        k=self.intncmp(self.aboard,self.scratch,size)
        if k>0:
          return
        nequiv=8
    for i in range(size):
      self.scratch[i]=self.aboard[i]
    # 垂直反転
    self.vmirror_bitmap(self.scratch,self.trial,size)
    k=self.intncmp(self.aboard,self.trial,size)
    if k>0:
      return 
    if nequiv>2:
      #90
      self.rotate_bitmap(self.trial,self.scratch,size)
      k=self.intncmp(self.aboard,self.scratch,size)
      if k>0:
        return 
      if nequiv>4:
        #180
        self.rotate_bitmap(self.scratch,self.trial,size)
        k=self.intncmp(self.aboard,self.trial,size)
        if k>0:
          return 
        #270
        self.rotate_bitmap(self.trial,self.scratch,size)
        k=self.intncmp(self.aboard,self.scratch,size)
        if k>0:
          return 
    if nequiv==2:
      self.count2+=1
    if nequiv==4:
      self.count4+=1
    if nequiv==8:
      self.count8+=1
  def nqueens(self,size,row,left,down,right):
    mask=(1<<size)-1
    bitmap=mask&~(left|down|right)
    if row==size:
      if bitmap:
        pass
      else:
        self.aboard[row]=bitmap
        self.symmetryops_bitmap(size)
    else:
      if row!=0:
        lim=size
      else:
        lim=(size+1)//2
      for i in range(row,lim):
        while bitmap:
          bit=(-bitmap&bitmap)
          self.aboard[row]=bit
          bitmap^=self.aboard[row]
          self.nqueens(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def main(self):
    nmin=4
    nmax=16
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(nmin,nmax):
      self.total=0
      self.unique=0
      self.count2=0
      self.count4=0
      self.count8=0
      for j in range(size):
        self.aboard[j]=j
      start_time=datetime.now()
      self.nqueens(size,0,0,0,0)
      time_elapsed=datetime.now()-start_time
      _text='{}'.format(time_elapsed)
      text=_text[:-3]
      self.total=self.count2*2+self.count4*4+self.count8*8
      self.unique=self.count2+self.count4+self.count8
      print("%2d:%13d%13d%20s" % (size,self.total,self.unique,text)); 


class NQueens07:
  def __init__(self):
    self.max=16
    self.aboard=[0 for i in range(self.max)]
    self.trial=[0 for i in range(self.max)]
    self.scratch=[0 for i in range(self.max)]
    self.count2=0
    self.count4=0
    self.count8=0
  def rha(self,ah,size):
    tmp=0
    for i in range(size+1):
      if ah&(1<<i):
        tmp|=(1<<size-i)
    return tmp
  def vmirror_bitmap(self,bf,af,size):
    score=0
    for i in range(size):
      score=bf[i]
      af[i]=self.rha(score,size-1)
  def rotate_bitmap(self,bf,af,size):
    for i in range(size):
      tmp=0
      for j in range(size):
        tmp|=((bf[j]>>i)&1)<<(size-j-1)
      af[i]=tmp
  def intncmp(self,lt,rt,neg):
    rtn=0
    for i in range(neg):
      rtn=lt[i]-rt[i]
      if rtn!=0:
        break
    return rtn
  def symmetryops_bitmap(self,size):
    nequiv=0
    for i in range(size):
      self.trial[i]=self.aboard[i]
    #90
    self.rotate_bitmap(self.trial,self.scratch,size)
    k=self.intncmp(self.aboard,self.scratch,size)
    if k>0:
      return
    if k==0:
      nequiv=2
    else:
      #180
      self.rotate_bitmap(self.scratch,self.trial,size)
      k=self.intncmp(self.aboard,self.trial,size)
      if k>0:
        return 
      if k==0:
        nequiv=4
      else:
        #270
        self.rotate_bitmap(self.trial,self.scratch,size)
        k=self.intncmp(self.aboard,self.scratch,size)
        if k>0:
          return 
        nequiv=8
    for i in range(size):
      self.scratch[i]=self.aboard[i]
    #垂直反転
    self.vmirror_bitmap(self.scratch,self.trial,size)
    k=self.intncmp(self.aboard,self.trial,size)
    if k>0:
      return
    if nequiv>2:
      #90
      self.rotate_bitmap(self.trial,self.scratch,size)
      k=self.intncmp(self.aboard,self.scratch,size)
      if k>0:
        return 
      if nequiv>4:
        #180
        self.rotate_bitmap(self.scratch,self.trial,size)
        k=self.intncmp(self.aboard,self.trial,size)
        if k>0:
          return 
        #270
        self.rotate_bitmap(self.trial,self.scratch,size)
        k=self.intncmp(self.aboard,self.scratch,size)
        if k>0:
          return 
    if nequiv==2:
      self.count2+=1
    if nequiv==4:
      self.count4+=1
    if nequiv==8:
      self.count8+=1

  def nqueens(self,size,row,left,down,right):
    mask=(1<<size)-1
    bitmap=mask&~(left|down|right)
    if row==size:
      if bitmap:
        pass
      else:
        self.aboard[row]=bitmap
        self.symmetryops_bitmap(size)
    else:
      while bitmap:
        bit=(-bitmap&bitmap)
        self.aboard[row]=bit
        bitmap^=self.aboard[row]
        self.nqueens(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def main(self):
    nmin=4
    nmax=16
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(nmin,nmax):
      self.total=0
      self.unique=0
      self.count2=0
      self.count4=0
      self.count8=0
      for j in range(size):
        self.aboard[j]=j
      start_time=datetime.now()
      self.nqueens(size,0,0,0,0)
      time_elapsed=datetime.now()-start_time
      _text='{}'.format(time_elapsed)
      text=_text[:-3]
      self.total=self.count2*2+self.count4*4+self.count8*8
      self.unique=self.count2+self.count4+self.count8
      print("%2d:%13d%13d%20s" % (size,self.total,self.unique,text)); 

class NQueens06:
  def __init__(self):
    self.max=16
    self.total=0
    self.unique=0
    self.aboard=[0 for i in range(self.max)]
  def nqueens(self,size,row,left,down,right):
    mask=(1<<size)-1
    if row==size:
      self.total+=1
    else:
      bitmap=(mask&~(left|down|right))
      while bitmap:
        bit=(-bitmap&bitmap)
        bitmap=(bitmap^bit)
        self.nqueens(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
  def main(self):
    nmin=4
    nmax=16
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(nmin,nmax):
      self.total=0
      self.unique=0
      for j in range(size):
        self.aboard[j]=j
      start_time=datetime.now()
      self.nqueens(size,0,0,0,0)
      time_elapsed=datetime.now()-start_time
      _text='{}'.format(time_elapsed)
      text=_text[:-3]
      print("%2d:%13d%13d%20s" % (size,self.total,self.unique,text)); 
    
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
    k=self.intncmp(self.aboard,self.trial,size)
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

  def nqueens_rec(self,row,size):
    if row==size-1:
      if self.fb[row-self.aboard[row]+size-1] or self.fc[row+self.aboard[row]]:
        return
      stotal=self.symmetryops(size)
      if stotal!=0:
        self.unique+=1
        self.total+=stotal
    else:
      lim=size if row!=0 else (size+1) //2
      for i in range(row,lim):
        tmp=self.aboard[i]
        self.aboard[i]=self.aboard[row]
        self.aboard[row]=tmp
        if self.fb[row-self.aboard[row]+size-1]==0 and self.fc[row+self.aboard[row]]==0:
          self.fb[row-self.aboard[row]+size-1]=self.fc[row+self.aboard[row]]=1
          self.nqueens_rec(row+1,size)
          self.fb[row-self.aboard[row]+size-1]=self.fc[row+self.aboard[row]]=0
      tmp=self.aboard[row]
      for i in range(row+1,size):
        self.aboard[i-1]=self.aboard[i]
      self.aboard[size-1]=tmp

  def main(self):
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
    self.max=16;
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
# マルチプロセス
# 15:      2279184       285053         0:00:06.216
NQueens13_3multiProcess.main();
#
# マルチスレッド True/マルチスレッド False シングルスレッド
# 15:      2279184       285053         0:00:07.983
BTHREAD = True 
#NQueens13_2_multiThread.main();
#
# シングルスレッド
# 15:      2279184       285053         0:00:07.958
#NQueens13_1_singleThread.main();
#
# 最適化
# 15:      2279184       285053         0:00:10.607
#NQueens12().main();
#
# 枝刈り
# 15:      2279184       285053         0:00:10.927
#NQueens11().main();
#
# BOUND1,2
# 15:      2279184       285053         0:00:26.210
#NQueens10().main();
#
# BOUND1
# 15:      2279184       285053         0:00:25.239
#NQueens09().main();
#
# 枝刈り
# 15:      2279184       285053         0:00:28.357
#NQueens08().main();
#
# 対象解除法とビットマップ
# 15:      2279184       285053         0:00:19.711
#NQueens07().main();
#
# バックトラックとビットマップ
# 15:      2279184            0         0:00:11.417
#NQueens06().main();
#
# 枝刈りと最適化
# 15:      2279184       285053         0:00:15.677
#NQueens05().main();
#
# 対象解除法
# 15:      2279184       285053         0:00:49.855
#NQueens04().main();
#
# バックトラック
# 15:      2279184            0         0:00:44.558
# NQueens03().main();
#
# 配置フラグ
# NQueens02().nqueens(0);
#
# ブルートフォース　ちからまかせ探索
# NQueens01().nqueen(0);
#
