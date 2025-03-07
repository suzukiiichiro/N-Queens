#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
bit スレッドプール版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens
"""

"""
CentOS-5.1$ pypy 09Python_bit_symmetry_ThreadPool.py
 N:        Total       Unique        hh:mm:ss.ms
 4:            2            1         0:00:00.018
 5:           10            2         0:00:00.019
 6:            4            1         0:00:00.016
 7:           40            6         0:00:00.028
 8:           92           12         0:00:00.034
 9:          352           46         0:00:00.084
10:          724           92         0:00:00.150
11:         2680          341         0:00:00.250
12:        14200         1787         0:00:00.295
13:        73712         9233         0:00:00.634
14:       365596        45752         0:00:02.656
15:      2279184       285053         0:00:03.704
16:     14772512      1846955         0:00:17.872
17:     95815104     11977939         0:01:43.887

CentOS-5.1$ pypy 08Python_bit_symmetry.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184       285053         0:00:03.026

CentOS-5.1$ pypy 07Python_bit_mirror.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184            0         0:00:06.274

CentOS-5.1$ pypy 06Python_bit_backTrack.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184            0         0:00:12.610

CentOS-5.1$ pypy 05Python_optimize.py
 N:        Total       Unique         hh:mm:ss.ms
15:      2279184       285053         0:00:14.413

CentOS-5.1$ pypy 04Python_symmetry.py
 N:        Total       Unique         hh:mm:ss.ms
15:      2279184       285053         0:00:46.629

CentOS-5.1$ pypy 03Python_backTracking.py
 N:        Total       Unique         hh:mm:ss.ms
15:      2279184            0         0:00:44.993
"""

"""
pyenvでpypyをインストール
$ curl https://pyenv.run | bash

codonのインストール
/bin/bash -c "$(curl -fsSL https://exaloop.io/install.sh)"
echo 'export PATH="$HOME/.codon/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

codon の実行
JIT
$ codon run -release file.py

BUILD exe
$ codon build -release file.py

外部ライブラリの使い方

libpython*.so を探します
$ find / -name "libpython*.so"

だいたい以下にあります
$ locate libpython3
/home/suzuki/.pyenv/versions/3.13.0/lib/libpython3.so

CODON_PYTHONの環境変数を~/.bash_profileに追加します
echo "export CODON_PYTHON=$PYENV_ROOT/versions/3.13.0/lib/libpython3.13.so" >> ~/.bash_profile

"""

# -*- coding: utf-8 -*-
import subprocess
from datetime import datetime
#
# pypyを使うときは以下を活かしてcodon部分をコメントアウト
# pypy では ThreadPool/ProcessPoolが動きます 
#
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')
from threading import Thread
from multiprocessing import Pool as ThreadPool
import concurrent
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
#
#
# codonを使うときは以下を活かして上記をコメントアウト
# ThreadPool/ProcessPoolはcodonでは動きません
#
# from python import Pool as ThreadPool
# from python import Thread 
# from python import threading 
# from python import multiprocessing
# from python import concurrent
# from python import ThreadPoolExecutor
# from python import ProcessPoolExecutor

class NQueens09():
  def __init__(self):
    pass

  def getunique(self,counts:list)->int:
    count2:int
    count4:int
    count8:int
    count2,count4,count8=counts
    return count2+count4+count8

  def gettotal(self,counts:list)->int:
    count2:int
    count4:int
    count8:int
    count2,count4,count8=counts
    return count2*2+count4*4+count8*8

  def symmetryops(self,size:int,aboard:list,topbit:int,endbit:int,sidemask:int,lastmask:int,bound1:int,bound2:int)->list:
    count2:int
    count4:int
    count8:int
    count2=count4=count8=0
    own:int
    ptn:int
    you:int
    bit:int
    if aboard[bound2]==1:
      own,ptn=1,2
      for own in range(1,size):
        bit=1
        you=size-1
        while aboard[you]!=ptn and aboard[own]>=bit:
          bit<<=1
          you-=1
        if aboard[own]>bit:
          return [count2,count4,count8]
        if aboard[own]<bit:
          break
        ptn<<=1
      else:
        count2+=1
        return [count2,count4,count8]
    if aboard[size-1]==endbit:
      own,you=1,size-2
      for own in range(1,size):
        bit,ptn=1,topbit
        while aboard[you]!=ptn and aboard[own]>=bit:
          bit<<=1
          ptn>>=1
        if aboard[own]>bit:
          return [count2,count4,count8]
        if aboard[own]<bit:
          break
        you-=1
      else:
        count4+=1
        return [count2,count4,count8]
    if aboard[bound1]==topbit:
      ptn=topbit>>1
      for own in range(1,size):
        bit=1
        you=0
        while aboard[you]!=ptn and aboard[own]>=bit:
          bit<<=1
          you+=1
        if aboard[own]>bit:
          return [count2,count4,count8]
        if aboard[own]<bit:
          break
        ptn>>=1
    count8+=1
    return [count2,count4,count8]

  def backTrack2(self,size:int,row:int,left:int,down:int,right:int,aboard:list,topbit:int,endbit:int,sidemask:int,lastmask:int,bound1:int,bound2:int)->list:
    count2:int
    count4:int
    count8:int
    count2=count4=count8=0
    bit:int
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    # 最下行の場合、最適化のための条件チェック
    if row==size-1:
      if bitmap and (bitmap&lastmask)==0:
        aboard[row]=bitmap
        count2,count4,count8=self.symmetryops(size,aboard,topbit,endbit,sidemask,lastmask,bound1,bound2)
      return [count2,count4,count8]
    # 上部の行であればサイドマスク適用
    if row<bound1:
      bitmap&=~sidemask
    elif row==bound2:
      # `bound2` 行の場合、
      # サイドマスクとの一致を確認し不要な分岐を排除
      if (down&sidemask)==0:
        return [count2,count4,count8]
      elif (down&sidemask)!=sidemask:
        bitmap&=sidemask
    c2:int
    c4:int
    c8:int
    while bitmap:
      bit=bitmap&-bitmap  # 最右ビットを抽出
      bitmap^=bit         # 最右ビットを消去
      aboard[row]=bit
      c2,c4,c8=self.backTrack2(size,row+1,(left|bit)<<1,down|bit,(right|bit) >> 1,aboard,topbit,endbit,sidemask,lastmask,bound1,bound2)
      count2+=c2
      count4+=c4
      count8+=c8
    return [count2,count4,count8]

  def backTrack1(self,size:int,row:int,left:int,down:int,right:int,aboard:list,topbit:int,endbit:int,sidemask:int,lastmask:int,bound1:int,bound2:int)->list:
    count2:int=0
    count4:int=0
    count8:int=0
    c2:int
    c4:int
    c8:int
    bit:int
    mask:int=(1<<size)-1
    bitmap:int=mask & ~(left|down|right)
    if row==size-1: # 最下行に達した場合の処理
      if bitmap:
        aboard[row]=bitmap
        count8+=1
      return [count2,count4,count8]
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
    return [count2,count4,count8]

  def nqueen_processPool(self,value:list)->list:
    thr_index,size=value
    sizeE=size-1
    aboard:list[int]=[0 for i in range(size)]
    # aboard:list[int]
    # for i in range(size):
    #   aboard[i]=0
    # aboard=[[0]*size*2]*size
    # aboard=[[i for i in range(2*size-1)]for j in range(size)]
    bit:int
    topbit:int
    endbit:int
    sidemask:int
    lastmask:int
    bound1:int
    bound2:int
    count2:int
    count4:int
    count8:int
    c2:int
    c4:int
    c8:int
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

  def nqueen_threadPool(self,value:list)->list:
    thr_index,size=value
    sizeE:int=size-1
    aboard:list[int]=[0 for i in range(size)]
    # aboard:list[int]
    # aboard=[[0]*size*2]*size
    # aboard:list[int]=[[0]*size*2]*size
    # for i in range(size):
    #   aboard[i]=0
    # aboard=[[i for i in range(2*size-1)]for j in range(size)]
    # aboard:list[int]
    # for i in range(size):
    #   aboard.insert(i,0)
    bit:int
    topbit:int
    endbit:int
    sidemask:int
    lastmask:int
    bound1:int
    bound2:int
    count2:int
    count4:int
    count8:int
    c2:int
    c4:int
    c8:int
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
    return [count2,count4,count8]
  def solve(self,size:int)->list:
    with concurrent.futures.ThreadPoolExecutor() as executor:
      params=[(thr_index,size) for thr_index in range(size) ]
    #
    # concurrent.futuresマルチスレッド版
    # 15:      2279184       285053         0:00:05.440
      # results=executor.map(self.nqueen_threadPool,params)
    #
    # concurrent.futuresマルチプロセス版
    # 15:      2279184       285053         0:00:05.526
      # results=executor.map(self.nqueen_processPool,params)
    #
    #
    pool = ThreadPool(size)
    params=[(thr_index,size) for thr_index in range(size) ]
    # マルチスレッド版
    # 15:      2279184       285053         0:00:03.553
    results:list[int]=list(pool.map(self.nqueen_threadPool,params))
    #
    # マルチプロセス版
    # 15:      2279184       285053         0:00:02.378
    # results:list[int]=list(pool.map(self.nqueen_processPool,params))
    #
    #
    # スレッドごとの結果を集計
    total_counts:int=[sum(x) for x in zip(*results)]
    total:int=self.gettotal(total_counts)
    unique:int=self.getunique(total_counts)
    return [total,unique]

class NQueens09_threadPool:
  def finalize(self)->None:
    cmd="killall pypy"  # python or pypy
    p = subprocess.Popen("exec " + cmd, shell=True)
    p.kill()
  def main(self):
    nmin:int=4
    nmax:int=18
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin, nmax):
      start_time=datetime.now()
      NQ=NQueens09()
      total,unique=NQ.solve(size)
      time_elapsed=datetime.now()-start_time
      text = str(time_elapsed)[:-3]  
      print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")
      self.finalize()
#
# $ python <filename>
# $ pypy <fileName>
# $ codon build -release <filename>
# codon ではスレッドプールが動かなかった
# スレッドプール
# 15:      2279184       285053         0:00:04.684
if __name__ == '__main__':
  NQueens09_threadPool().main()

