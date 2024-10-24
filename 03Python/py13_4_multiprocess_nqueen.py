#!/usr/bin/env python

# -*- coding: utf-8 -*-
""" py13_4_multiprocess_nqueen.py """

from datetime import datetime
from multiprocessing import Pool as ThreadPool

# ThreadPoolのインストール
# $ pip install Pool 
#
# /**
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#
#
#  実行
#  $ python py13_4_multiprocess_nqueen.py
#
# １３−４．マルチプロセス
#  単一のCPUを複数のプロセスで使い回すマルチスレッドと異なり、
#  複数のCPUを複数のプロセスで使い回すマルチプロセスのほうが、
#  圧倒的に高速です
#  py13_1, 2, 3, では処理中に、「全ての」CPUの負荷が１００％になることはありませんでした。
#  処理中は全てのCPUの負荷が１００％になっていることを確認して下さい。
#
#
# py12_nqueen.py
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.000
#  5:           10            2         0:00:00.000
#  6:            4            1         0:00:00.000
#  7:           40            6         0:00:00.000
#  8:           92           12         0:00:00.000
#  9:          352           46         0:00:00.002
# 10:          724           92         0:00:00.012
# 11:         2680          341         0:00:00.057
# 12:        14200         1787         0:00:00.291
# 13:        73712         9233         0:00:01.512
# 14:       365596        45752         0:00:08.967
# 15:      2279184       285053         0:00:54.645
#
#
# シングルスレッドにて実行 py13_1_singlethread_nqueen.py
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.000
#  5:           10            2         0:00:00.000
#  6:            4            1         0:00:00.000
#  7:           40            6         0:00:00.000
#  8:           92           12         0:00:00.000
#  9:          352           46         0:00:00.003
# 10:          724           92         0:00:00.013
# 11:         2680          341         0:00:00.062
# 12:        14200         1787         0:00:00.309
# 13:        73712         9233         0:00:01.672
# 14:       365596        45752         0:00:09.599
# 15:      2279184       285053         0:01:00.039
#
#
# マルチスレッドにて実行 py13_2_multithread_nqueen.py
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.000
#  5:           10            2         0:00:00.000
#  6:            4            1         0:00:00.000
#  7:           40            6         0:00:00.000
#  8:           92           12         0:00:00.001
#  9:          352           46         0:00:00.004
# 10:          724           92         0:00:00.014
# 11:         2680          341         0:00:00.065
# 12:        14200         1787         0:00:00.315
# 13:        73712         9233         0:00:01.684
# 14:       365596        45752         0:00:09.602
# 15:      2279184       285053         0:00:59.849
#
#
# マルチスレッドにて py13_3_multithread_join_nqueen.py
# コア一つを使い回しているにすぎない。遅い
# joinを処理末で実行。本来のマルチスレッド
# Nの数分スレッドが起動しそれぞれ並列処理
#
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.001
#  5:           10            2         0:00:00.000
#  6:            4            1         0:00:00.000
#  7:           40            6         0:00:00.000
#  8:           92           12         0:00:00.001
#  9:          352           46         0:00:00.004
# 10:          724           92         0:00:00.015
# 11:         2680          341         0:00:00.069
# 12:        14200         1787         0:00:00.328
# 13:        73712         9233         0:00:01.919
# 14:       365596        45752         0:00:10.975
# 15:      2279184       285053         0:01:08.997
#
#
# マルチプロセス版 （このソース）
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.124
#  5:           10            2         0:00:00.110
#  6:            4            1         0:00:00.116
#  7:           40            6         0:00:00.115
#  8:           92           12         0:00:00.119
#  9:          352           46         0:00:00.118
# 10:          724           92         0:00:00.121
# 11:         2680          341         0:00:00.122
# 12:        14200         1787         0:00:00.228
# 13:        73712         9233         0:00:00.641
# 14:       365596        45752         0:00:03.227
# 15:      2279184       285053         0:00:19.973
#
#
# グローバル変数
#
class Nqueen(): # pylint: disable=R0902
  #
  # 初期化
  def __init__(self, size):
    self.size = size                    # N
    self.sizeE = size -1
    self._nthreads = self.size
    self.total = 0                      # スレッド毎の合計
    self.unique = 0
    self.gttotal = [0] * self.size      #総合計
    self.gtunique = [0] * self.size     #総合計
    self.aboard = [[i for i in range(2*size-1)] for j in range(self.size)]
    self.mask = (1<<size)-1
    self.count2 = 0
    self.count4 = 0
    self.count8 = 0
    self.bound1 = 0
    self.bound2 = 0
    self.sidemask = 0
    self.lastmask = 0
    self.topbit = 0
    self.endbit = 0
  #
  # ユニーク値を出力
  def getunique(self):
    """ getunique() """
    return self.count2+self.count4+self.count8
  #
  # 合計を出力
  def gettotal(self):
    """ gettotal() """
    return self.count2*2+self.count4*4+self.count8*8
  #
  # 対称解除法
  def symmetryops(self):  # pylint: disable=R0912,R0911,R0915
    """ symmetryops() """
    own = 0
    ptn = 0
    you = 0
    bit = 0
    # 90度回転
    if self.aboard[self.bound2] == 1:
      own = 1
      ptn = 2
      while own <= self.size-1:
        bit = 1
        you = self.size-1
        while (self.aboard[you] != ptn) and (self.aboard[own] >= bit):
          bit <<= 1
          you -= 1
        if self.aboard[own] > bit:
          return
        if self.aboard[own] < bit:
          break
        own += 1
        ptn <<= 1
      #90度回転して同型なら180度/270度回転も同型である */
      if own > self.size-1:
        self.count2 += 1
        return
    #180度回転
    if self.aboard[self.size-1] == self.endbit:
      own = 1
      you = self.size-1-1
      while own <= self.size-1:
        bit = 1
        ptn = self.topbit
        while (self.aboard[you] != ptn) and (self.aboard[own] >= bit):
          bit <<= 1
          ptn >>= 1
        if self.aboard[own] > bit:
          return
        if self.aboard[own] < bit:
          break
        own += 1
        you -= 1
      # 90度回転が同型でなくても180度回転が同型である事もある */
      if own > self.size-1:
        self.count4 += 1
        return
    #270度回転
    if self.aboard[self.bound1] == self.topbit:
      own = 1
      ptn = self.topbit>>1
      while own <= self.size-1:
        bit = 1
        you = 0
        while (self.aboard[you] != ptn) and (self.aboard[own] >= bit):
          bit <<= 1
          you += 1
        if self.aboard[own] > bit:
          return
        if self.aboard[own] < bit:
          break
        own += 1
        ptn >>= 1
    self.count8 += 1
  #
  # BackTrack2
  def backtrack2(self, row, left, down, right): # pylint: disable=R0913
    """ backtrack2() """
    bit = 0
    bitmap = self.mask&~(left|down|right)
    #枝刈り
    #if row == size:
    if row == self.size-1:
      if bitmap:
        #枝刈り
        if (bitmap&self.lastmask) == 0:
          self.aboard[row] = bitmap
          #symmetryOps_bitmap(size)
          self.symmetryops()
      #else:
      #  self.aboard[row] = bitmap
      #  symmetryOps_bitmap(size)
    else:
      #枝刈り 上部サイド枝刈り
      if row < self.bound1:
        bitmap &= ~self.sidemask
      #枝刈り 下部サイド枝刈り
      elif row == self.bound2:
        if down&self.sidemask == 0:
          return
        if down&self.sidemask != self.sidemask:
          bitmap &= self.sidemask
      # 枝刈り
      if row != 0:
        lim = self.size
      else:
        lim = (self.size+1)//2 # 割り算の結果を整数にするには //
      # 枝刈り
      for i in range(row, lim): # pylint: disable=W0612
        while bitmap:
          bit = (-bitmap&bitmap)
          self.aboard[row] = bit
          bitmap ^= self.aboard[row]
          self.backtrack2(row+1, (left|bit)<<1, down|bit, (right|bit)>>1)
  #
  # BackTrack1
  def backtrack1(self, row, left, down, right):  # pylint: disable=R0913
    """ backtrack1() """
    bit = 0
    bitmap = self.mask&~(left|down|right)
    #枝刈り
    #if row =  = size:
    if row == self.size-1:
      if bitmap:
        self.aboard[row] = bitmap
        #枝刈りにてcount8に加算
        self.count8 += 1
      #else:
      #  self.aboard[row] = bitmap
      #  symmetryOps_bitmap(size)
    else:
      #枝刈り 鏡像についても主対角線鏡像のみを判定すればよい
      # ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      if row < self.bound1:
        bitmap &= ~2 # bm| = 2 bm^ = 2 (bm& = ~2と同等)
      # 枝刈り
      if row != 0:
        lim = self.size
      else:
        lim = (self.size+1)//2 # 割り算の結果を整数にするには //
      # 枝刈り
      for i in range(row, lim):   # pylint: disable=W0612
        while bitmap:
          bit = (-bitmap&bitmap)
          self.aboard[row] = bit
          bitmap ^= self.aboard[row]
          self.backtrack1(row+1, (left|bit)<<1, down|bit, (right|bit)>>1)
  #
  def BOUND1_single(self, B1):
    bit = 0
    self.bound1 = B1 
    self.aboard[1] = bit = (1<<self.bound1)
    self.backtrack1(2, (2|bit)<<1, (1|bit), (bit>>1))
  #
  def BOUND2_single(self, B1, B2):
    bit = 0
    self.bound1 = B1
    self.bound2 = B2
    self.aboard[0] = bit = (1<<self.bound1)
    self.backtrack2(1, bit<<1, bit, bit>>1)
    self.lastmask|=self.lastmask>>1|self.lastmask<<1
    self.endbit >>= 1
    self.bound1 += 1
    self.bound2 -= 1
  #
  def BOUND1_multi(self, B1):
    bit = 0
    self.bound1 = B1 
    self.aboard[1] = bit = (1<<self.bound1)
    self.backtrack1(2, (2|bit)<<1, (1|bit), (bit>>1))
  #
  def BOUND2_multi(self, B1, B2):
    bit = 0
    self.bound1 = B1
    self.bound2 = B2
    self.aboard[0] = bit = (1<<self.bound1)
    for i in range(1, self.bound1):
      self.lastmask |= self.lastmask>>1|self.lastmask<<1
      self.endbit >>= 1
    self.backtrack2(1, bit<<1, bit, bit>>1)
  #
  #メインメソッド シングル版
  def nqueen_single(self, thr_index):
    """ nqueen_single() """
    self.aboard[0] = 1
    self.sizeE = self.size - 1
    self.mask = (1<<self.size)-1
    self.topbit = 1<<self.sizeE
    self.bound1 = 1
    for self.bound1 in range(2, self.sizeE):
      self.BOUND1_single(self.bound1)
      self.bound1 += 1
    self.sidemask = self.lastmask = (self.topbit|1)
    self.endbit = (self.topbit>>1)
    self.bound1 = 1
    self.bound2 = self.sizeE -1
    for self.bound1 in range(1, self.bound2):
      self.BOUND2_single(self.bound1, self.bound2)
    return self.gettotal(), self.getunique()
  #
  # メインメソッド マルチプロセス版
  def nqueen_multi(self, thr_index):
    self.aboard[0] = 1
    self.sizeE = self.size -1
    self.topbit = 1<<self.sizeE
    self.bound1 = (self.size)-thr_index-1
    if self.bound1 > 1 and self.bound1 < self.sizeE:
      self.BOUND1_multi(self.bound1)
    self.endbit = (self.topbit>>1)
    self.sidemask = self.lastmask = (self.topbit|1)
    self.bound2 = thr_index
    if self.bound1 > 0 and self.bound2<self.sizeE and self.bound1 < self.bound2:
      self.BOUND2_multi(self.bound1, self.bound2)
    return self.gettotal(), self.getunique()
  #
  # 解法
  def solve(self):
    pool = ThreadPool(self.size)
    #
    ## ロジック確認用
    ## シングル版 Nで割ると解が合う
    ## gttotal:[(92, 12), (92, 12), (92, 12), (92, 12), (92, 12), (92, 12), (92, 12), (92, 12)]
    ##  8:          736           96         0:00:00.119
    #
    # self.gttotal = list(pool.map(self.nqueen_single, range(self.size)))
    ##
    ## マルチプロセス版
    self.gttotal = list(pool.map(self.nqueen_multi, range(self.size)))
    #
    total = 0
    unique = 0
    for _t, _u in self.gttotal:
      total += _t
      unique += _u
    pool.close()
    pool.join()
    #
    return total, unique
#end class
#
# メインメソッド
if __name__ == '__main__':
  """ main() """
  nmin = 4                          # Nの最小値（スタートの値）を格納
  nmax = 16
  print(" N:        Total       Unique        hh:mm:ss.ms")
  for i in range(nmin, nmax):
    start_time = datetime.now()
    nqueen_obj = Nqueen(i)
    total, unique = nqueen_obj.solve()
    time_elapsed = datetime.now()-start_time
    _text = '{}'.format(time_elapsed)
    text = _text[:-3]
    print("%2d:%13d%13d%20s" % (i, total, unique, text)) # 出力
#
