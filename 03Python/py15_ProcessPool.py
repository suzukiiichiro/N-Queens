# -*- coding: utf-8 -*-
import logging
import threading
from threading import Thread
from multiprocessing import Pool as ThreadPool
from datetime import datetime
# pypyで再帰が高速化できる

# pypyを使う場合はコメントを解除
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

# ThreadPoolとProcessPool
import os
import concurrent.futures
class NQueens20():
  def __init__(self):
    pass
  def getunique(self,counts):
    count2,count4,count8=counts
    return count2+count4+count8
  def gettotal(self,counts):
    count2,count4,count8=counts
    return count2*2+count4*4+count8*8
  def symmetryops(self,size,aboard,topbit,endbit,sidemask,lastmask,bound1,bound2):
    count2=count4=count8=0
    if aboard[bound2]==1:
      own,ptn=1,2
      for own in range(1,size):
        bit=1
        you=size-1
        while aboard[you]!=ptn and aboard[own]>=bit:
          bit<<=1
          you-=1
        if aboard[own]>bit:
          return count2,count4,count8
        if aboard[own]<bit:
          break
        ptn<<=1
      else:
        count2+=1
        return count2,count4,count8
    if aboard[size-1]==endbit:
      own,you=1,size-2
      for own in range(1,size):
        bit,ptn=1,topbit
        while aboard[you]!=ptn and aboard[own]>=bit:
          bit<<=1
          ptn>>=1
        if aboard[own]>bit:
          return count2,count4,count8
        if aboard[own]<bit:
          break
        you-=1
      else:
        count4+=1
        return count2,count4,count8
    if aboard[bound1]==topbit:
      ptn=topbit>>1
      for own in range(1,size):
        bit=1
        you=0
        while aboard[you]!=ptn and aboard[own]>=bit:
          bit<<=1
          you+=1
        if aboard[own]>bit:
          return count2,count4,count8
        if aboard[own]<bit:
          break
        ptn>>=1
    count8+=1
    return count2,count4,count8
  def backTrack2(self,size,row,left,down,right,aboard,topbit,endbit,sidemask,lastmask,bound1,bound2):
    count2=count4=count8=0
    mask=(1<<size)-1
    bitmap=mask&~(left|down|right)
    # 最下行の場合、最適化のための条件チェック
    if row==size-1:
      if bitmap and (bitmap&lastmask)==0:
        aboard[row]=bitmap
        count2,count4,count8=self.symmetryops(size,aboard,topbit,endbit,sidemask,lastmask,bound1,bound2)
      return count2,count4,count8
    # 上部の行であればサイドマスク適用
    if row<bound1:
      bitmap&=~sidemask
    elif row==bound2:
      # `bound2` 行の場合、
      # サイドマスクとの一致を確認し不要な分岐を排除
      if (down&sidemask)==0:
        return count2,count4,count8
      elif (down&sidemask)!=sidemask:
        bitmap&=sidemask
    while bitmap:
      bit=bitmap&-bitmap  # 最右ビットを抽出
      bitmap^=bit         # 最右ビットを消去
      aboard[row]=bit
      c2,c4,c8=self.backTrack2(size,row+1,(left|bit)<<1,down|bit,(right|bit) >> 1,aboard,topbit,endbit,sidemask,lastmask,bound1,bound2)
      count2+=c2
      count4+=c4
      count8+=c8
    return count2, count4, count8  
  def backTrack1(self,size,row,left,down,right,aboard,topbit,endbit,sidemask,lastmask,bound1,bound2):
    count2=0
    count4=0
    count8=0
    mask=(1<<size)-1
    bitmap=mask & ~(left|down|right)
    
    if row==size-1: # 最下行に達した場合の処理
      if bitmap:
        aboard[row]=bitmap
        count8+=1
      return count2,count4,count8
    if row<bound1:  # 上部の行であればマスク適用
      bitmap &= ~2
    while bitmap:
      bit=bitmap&-bitmap    # 最右ビットを抽出
      bitmap^=bit           # 最右ビットを消去
      aboard[row]=bit
      c2,c4,c8=self.backTrack1(size,row+1,(left|bit)<<1,down|bit,(right|bit) >> 1,aboard,topbit,endbit,sidemask,lastmask,bound1,bound2)
      count2+=c2
      count4+=c4
      count8+=c8
    return count2,count4,count8  
  def nqueen_multiProcess(self,value):
    thr_index,size=value
    sizeE=size-1
    aboard=[[i for i in range(2*size-1)]for j in range(size)]
    bit=topbit=endbit=sidemask=lastmask=bound1=bound2=count2=count4=count8=0
    aboard[0]=1
    topbit=1<<sizeE
    bound1=size-thr_index-1
    if 1<bound1<sizeE: 
      aboard[1]=bit=1<<bound1
      c2,c4,c8=self.backTrack1(size,2,(2|bit)<<1,(1|bit),(bit>>1),aboard,topbit,endbit,sidemask,lastmask,bound1,bound2)
      count2+=c2
      count4+=c4
      count8+=c8
    endbit=topbit>>1
    sidemask=lastmask=topbit|1
    bound2=thr_index
    if 0<bound1<bound2<sizeE:
      aboard[0]=bit=(1<<bound1)
      for i in range(1,bound1):
        lastmask|=lastmask>>1|lastmask<<1
        endbit>>=1
      c2,c4,c8=self.backTrack2(size,1,bit<<1,bit,bit>>1,aboard,topbit,endbit,sidemask,lastmask,bound1,bound2)
      count2+=c2
      count4+=c4
      count8+=c8
    return count2,count4,count8
  def nqueen_multiThread(self,value):
    thr_index,size=value
    sizeE=size-1
    aboard=[[i for i in range(2*size-1)]for j in range(size)]
    bit=topbit=endbit=sidemask=lastmask=bound1=bound2=count2=count4=count8=0
    aboard[0]=1
    topbit=1<<sizeE
    bound1=size-thr_index-1
    if 1<bound1<sizeE: 
      aboard[1]=bit=1<<bound1
      c2,c4,c8=self.backTrack1(size,2,(2|bit)<<1,(1|bit),(bit>>1),aboard,topbit,endbit,sidemask,lastmask,bound1,bound2)
      count2+=c2
      count4+=c4
      count8+=c8
    endbit=topbit>>1
    sidemask=lastmask=topbit|1
    bound2=thr_index
    if 0<bound1<bound2<sizeE:
      aboard[0]=bit=(1<<bound1)
      for i in range(1,bound1):
        lastmask|=lastmask>>1|lastmask<<1
        endbit>>=1
      c2,c4,c8=self.backTrack2(size,1,bit<<1,bit,bit>>1,aboard,topbit,endbit,sidemask,lastmask,bound1,bound2)
      count2+=c2
      count4+=c4
      count8+=c8
    return count2,count4,count8
  def solve(self,size):
    # マルチプロセス
    # 15:      2279184       285053         0:00:01.528
    with concurrent.futures.ProcessPoolExecutor() as executor:
      value=[(thr_index,size) for thr_index in range(size) ]
      results=list(executor.map(self.nqueen_multiProcess,value))

    # マルチスレッド
    # 15:      2279184       285053         0:00:04.684
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #   value=[(thr_index,size) for thr_index in range(size) ]
    #   results=list(executor.map(self.nqueen_multiThread,value))

    # スレッドごとの結果を集計
    total_counts=[sum(x) for x in zip(*results)]
    total=self.gettotal(total_counts)
    unique=self.getunique(total_counts)
    return total,unique
class NQueens20_multiProcess:
  def main(self):
    nmin = 4
    nmax = 18
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for i in range(nmin, nmax):
      start_time=datetime.now()
      NQ=NQueens20()
      total,unique=NQ.solve(i)
      time_elapsed=datetime.now()-start_time
      _text='{}'.format(time_elapsed)
      text=_text[:-3]
      print("%2d:%13d%13d%20s"%(i,total,unique, text))  
#
# マルチプロセス
# 15:      2279184       285053         0:00:01.528
if __name__ == '__main__':
  NQueens20_multiProcess().main()



