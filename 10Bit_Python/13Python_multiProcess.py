#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import copy
from datetime import datetime
from multiprocessing import Pool as ThreadPool

"""
マルチプロセス対応版 Ｎクイーン



詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

# 実行 
$ python <filename.py>

# 実行結果
bash-3.2$ python 12Python_multiThread.py
キャリーチェーン シングルスレッド
 N:        Total       Unique        hh:mm:ss.ms
 5:           10            2         0:00:00.002
 6:            4            1         0:00:00.010
 7:           40            6         0:00:00.046
 8:           92           12         0:00:00.205
 9:          352           46         0:00:00.910
10:          724           92         0:00:03.542
11:         2680          341         0:00:12.184
12:        14200         1788         0:00:36.986


bash-3.2$ python 12Python_multiThread.py
キャリーチェーン マルチスレッド
 N:        Total       Unique        hh:mm:ss.ms
 5:           10            2         0:00:00.003
 6:            4            1         0:00:00.013
 7:           40            6         0:00:00.049
 8:           92           12         0:00:00.215
 9:          352           46         0:00:00.934
10:          724           92         0:00:03.690
11:         2680          341         0:00:12.708
12:        14200         1788         0:00:40.294

bash-3.2$ python 13Python_multiProcess.py
キャリーチェーン マルチプロセス
 N:        Total       Unique        hh:mm:ss.ms
 5:           10            2         0:00:00.061
 6:            4            1         0:00:00.126
 7:           40            6         0:00:00.135
 8:           92           12         0:00:00.327
 9:          352           46         0:00:00.766
10:          724           92         0:00:02.327
11:         2680          341         0:00:09.042
12:        14200         1788         0:00:19.871
13:        73712         9237         0:00:53.388
14:       365596        45771         0:02:19.974
15:      2279184       285095         0:05:51.522
16:     14772512      1847425         0:17:01.656
17:     95815104     11979381         1:07:04.713

 ## bash版
 <> 08Bash_carryChain_parallel.sh 並列処理
 N:        Total       Unique        hh:mm:ss
 4:            0            0         0:00:00
 5:            8            1         0:00:00
 6:            4            1         0:00:00
 7:           40            6         0:00:00
 8:           92           12         0:00:01
 9:          352           46         0:00:03
10:          724           92         0:00:15
11:         2680          341         0:00:52
12:        14200         1788         0:02:49
13:        73712         9237         0:09:18
14:       365596        45771         0:28:48
15:      2279184       285095         1:49:12 １時間４９分１２秒

## C pthread版
bash-3.2$ gcc 17GCC_carryChain.c -o 17GCC && ./17GCC
７．キャリーチェーン
 N:        Total       Unique        dd:hh:mm:ss.ms
 4:            2            1        00:00:00:00.00
 5:           10            2        00:00:00:00.00
 6:            4            1        00:00:00:00.00
 7:           40            6        00:00:00:00.00
 8:           92           12        00:00:00:00.00
 9:          352           46        00:00:00:00.00
10:          724           92        00:00:00:00.00
11:         2680          341        00:00:00:00.00
12:        14200         1788        00:00:00:00.01
13:        73712         9237        00:00:00:00.03
14:       365596        45771        00:00:00:00.11
15:      2279184       285095        00:00:00:00.41
16:     14772512      1847425        00:00:00:02.29
17:     95815104     11979381        00:00:00:18.08
18:    666090624     83274576        00:00:03:40.04
19:   4968057848    621051686        00:00:27:37.22
20:  39029188884   4878995797        00:03:39:23.79
bash-3.2$

"""
#
# Board ボードクラス
class Board:
  def __init__(self,size):
    self.size=size
    self.row=0
    self.left=0
    self.down=0
    self.right=0
    self.X=[-1 for i in range(size)]
#
# nQueens メインスレッドクラス
class nQueens(): # pylint:disable=RO902
  #
  # ユニーク数の集計
  def getUnique(self):
    return self.COUNTER[0]+self.COUNTER[1]+self.COUNTER[2]
  #
  # 合計解の集計
  def getTotal(self):
    return self.COUNTER[0]*2+self.COUNTER[1]*4+self.COUNTER[2]*8
  #
  # ボード外側２列を除く内側のクイーン配置処理
  def solve(self,row,left,down,right):
    total=0
    if not down+1:
      return 1
    while row&1:
      row>>=1
      left<<=1
      right>>=1
    row>>=1           # １行下に移動する
    bitmap=~(left|down|right)
    while bitmap!=0:
      bit=-bitmap&bitmap
      total+=self.solve(row,(left|bit)<<1,down|bit,(right|bit)>>1)
      bitmap^=bit
    return total
  #
  # キャリーチェーン　solve()を呼び出して再起を開始する
  def process(self,sym):
    self.COUNTER[sym]+=self.solve(self.B.row>>2,self.B.left>>4,((((self.B.down>>2)|(~0<<(self.size-4)))+1)<<(self.size-5))-1,(self.B.right>>4)<<(self.size-5))
  #
  # キャリーチェーン　対象解除
  def carryChainSymmetry(self,n,w,s,e):
    # n,e,s=(N-2)*(N-1)-1-w の場合は最小値を確認する。
    ww=(self.size-2)*(self.size-1)-1-w
    w2=(self.size-2)*(self.size-1)-1
    # 対角線上の反転が小さいかどうか確認する
    if s==ww and n<(w2-e): return 
    # 垂直方向の中心に対する反転が小さいかを確認
    if e==ww and n>(w2-n): return
    # 斜め下方向への反転が小さいかをチェックする
    if n==ww and e>(w2-s): return
    # 【枝刈り】１行目が角の場合
    # １．回転対称チェックせずにCOUNT8にする
    if not self.B.X[0]:
      self.process(2) # COUNT8
      return
    # n,e,s==w の場合は最小値を確認する。
    # : '右回転で同じ場合は、
    # w=n=e=sでなければ値が小さいのでskip
    # w=n=e=sであれば90度回転で同じ可能性 ';
    if s==w:
      if n!=w or e!=w: return
      self.process(0) # COUNT2
      return
    # : 'e==wは180度回転して同じ
    # 180度回転して同じ時n>=sの時はsmaller?  ';
    if e==w and n>=s:
      if n>s: return
      self.process(1) # COUNT4
      return
    self.process(2)   # COUNT8
    return
  #
  # キャリーチェーン 効きのチェック dimxは行 dimyは列
  def placement(self,dimx,dimy):
    if self.B.X[dimx]==dimy:
      return 1
    if self.B.X[0]:
      if self.B.X[0]!=-1:
        if (dimx<self.B.X[0] or dimx>=self.size-self.B.X[0]) and (dimy==0 or dimy==self.size-1): 
          return 0
        if (dimx==self.size-1) and (dimy<=self.B.X[0] or dimy>=self.size-self.B.X[0]):
          return 0
    else:
      if self.B.X[1]!=-1:
        if self.B.X[1]>=dimx and dimy==1: return 0
    if (self.B.row & 1<<dimx) or (self.B.left & 1<<(self.size-1-dimx+dimy)) or (self.B.down & 1<<dimy) or (self.B.right & 1<<(dimx+dimy)): 
      return 0
    self.B.row|=1<<dimx
    self.B.left|=1<<(self.size-1-dimx+dimy)
    self.B.down|=1<<dimy
    self.B.right|=1<<(dimx+dimy)
    self.B.X[dimx]=dimy
    return 1
  #
  # シングルプロセス版 チェーンのビルド
  def buildChain_singleThread(self,w):
    wB=copy.deepcopy(self.B)
    for w in range( (self.size//2)*(self.size-3) +1):
      self.B=copy.deepcopy(wB)
      # １．０行目と１行目にクイーンを配置
      if self.placement(0,self.pres_a[w])==0:
        continue
      if self.placement(1,self.pres_b[w])==0:
        continue
      # ２．９０度回転
      nB=copy.deepcopy(self.B)
      mirror=(self.size-2)*(self.size-1)-w
      for n in range(w,mirror,1):
        self.B=copy.deepcopy(nB)
        if self.placement(self.pres_a[n],self.size-1)==0:
          continue
        if self.placement(self.pres_b[n],self.size-2)==0:
          continue
        # ３．９０度回転
        eB=copy.deepcopy(self.B)
        for e in range(w,mirror,1):
          self.B=copy.deepcopy(eB)
          if self.placement(self.size-1,self.size-1-self.pres_a[e])==0:
            continue
          if self.placement(self.size-2,self.size-1-self.pres_b[e])==0:
            continue
          # ４．９０度回転
          sB=copy.deepcopy(self.B)
          for s in range(w,mirror,1):
            self.B=copy.deepcopy(sB)
            if self.placement(self.size-1-self.pres_a[s],0)==0:
              continue
            if self.placement(self.size-1-self.pres_b[s],1)==0:
              continue
            # 対象解除法
            self.carryChainSymmetry(n,w,s,e)
  #
  # マルチプロセス版 チェーンのビルド
  def buildChain_multiThread(self,w):
    wB=copy.deepcopy(self.B)
    # for w in range( (self.size//2)*(self.size-3) +1):
    self.B=copy.deepcopy(wB)
    # １．０行目と１行目にクイーンを配置
    if self.placement(0,self.pres_a[w])==0:
      # continue
      return 
    if self.placement(1,self.pres_b[w])==0:
      # continue
      return 
    # ２．９０度回転
    nB=copy.deepcopy(self.B)
    mirror=(self.size-2)*(self.size-1)-w
    for n in range(w,mirror,1):
      self.B=copy.deepcopy(nB)
      if self.placement(self.pres_a[n],self.size-1)==0:
        continue
      if self.placement(self.pres_b[n],self.size-2)==0:
        continue
      # ３．９０度回転
      eB=copy.deepcopy(self.B)
      for e in range(w,mirror,1):
        self.B=copy.deepcopy(eB)
        if self.placement(self.size-1,self.size-1-self.pres_a[e])==0:
          continue
        if self.placement(self.size-2,self.size-1-self.pres_b[e])==0:
          continue
        # ４．９０度回転
        sB=copy.deepcopy(self.B)
        for s in range(w,mirror,1):
          self.B=copy.deepcopy(sB)
          if self.placement(self.size-1-self.pres_a[s],0)==0:
            continue
          if self.placement(self.size-1-self.pres_b[s],1)==0:
            continue
          # 対象解除法
          self.carryChainSymmetry(n,w,s,e)
  #
  # チェーンの初期化
  def initChain(self):
    idx=0
    for a in range(self.size):
      for b in range(self.size):
        if (a>=b and (a-b)<=1) or (b>a and (b-a<=1)):
          continue
        self.pres_a[idx]=a
        self.pres_b[idx]=b
        idx+=1
  #
  # シングルプロセス
  def carryChain_single(self,w):
    self.initChain()     # チェーンの初期化
    self.buildChain_singleThread(w)
    return self.getTotal(),self.getUnique()
  #
  # マルチプロセス
  def carryChain_multi(self,w):
    self.initChain()     # チェーンの初期化
    self.buildChain_multiThread(w)
    return self.getTotal(),self.getUnique()
  #
  # マルチプロセス
  def execProcess(self):
    #pool=ThreadPool(self.size)
    pool=ThreadPool((self.size//2)*(self.size-3) +1)
    #
    # シングルプロセスで実行
    #gttotal=list(pool.map(self.carryChain_single,range(1)))
    #
    # マルチプロセスで実行
    gttotal=list(pool.map(self.carryChain_multi,range( (self.size//2)*(self.size-3) +1)))
    #
    #
    # 集計処理
    total = 0 # ローカル変数
    unique = 0
    for _t, _u in gttotal:
      total+=_t
      unique+=_u
    pool.close()
    pool.join()
    return total,unique
  #
  # 初期化
  def __init__(self,size): # pylint:disable=R0913
    self.size=size
    self.COUNTER=[0]*3
    self.pres_a=[0]*930
    self.pres_b=[0]*930
    self.B=Board(size)
#
# メイン
if __name__ == '__main__':
  nmin = 5
  nmax = 21
  print("キャリーチェーン マルチプロセス")
  print(" N:        Total       Unique        hh:mm:ss.ms")
  for size in range(nmin, nmax,1):
    start_time = datetime.now()
    nq=nQueens(size)
    TOTAL,UNIQUE=nq.execProcess()
    time_elapsed = datetime.now() - start_time
    _text = '{}'.format(time_elapsed)
    text = _text[:-3]
    print("%2d:%13d%13d%20s" % (size,TOTAL,UNIQUE,text))  # 出力
