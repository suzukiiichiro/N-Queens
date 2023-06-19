#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import numpy as np
import copy
from datetime import datetime
"""
シングルスレッド対応版 Ｎクイーン



詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

# 実行 
$ python <filename.py>

# 実行結果
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
  # 初期化
  def __init__(self,size): # pylint:disable=R0913
    super(nQueens,self).__init__()
    self.size=size
    self.COUNTER=[0]*3
    self.carryChain()
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
  def process(self,B,sym):
    self.COUNTER[sym]+=self.solve(B.row>>2,B.left>>4,((((B.down>>2)|(~0<<(self.size-4)))+1)<<(self.size-5))-1,(B.right>>4)<<(self.size-5))
  #
  # キャリーチェーン　対象解除
  def carryChainSymmetry(self,B,n,w,s,e):
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
    if not B.X[0]:
      self.process(B,2) # COUNT8
      return
    # n,e,s==w の場合は最小値を確認する。
    # : '右回転で同じ場合は、
    # w=n=e=sでなければ値が小さいのでskip
    # w=n=e=sであれば90度回転で同じ可能性 ';
    if s==w:
      if n!=w or e!=w: return
      self.process(B,0) # COUNT2
      return
    # : 'e==wは180度回転して同じ
    # 180度回転して同じ時n>=sの時はsmaller?  ';
    if e==w and n>=s:
      if n>s: return
      self.process(B,1) # COUNT4
      return
    self.process(B,2)   # COUNT8
    return
  #
  # キャリーチェーン 効きのチェック dimxは行 dimyは列
  def placement(self,B,dimx,dimy):
    if B.X[dimx]==dimy:
      return 1
    if B.X[0]:
      if B.X[0]!=-1:
        if((dimx<B.X[0] or dimx>=self.size-B.X[0]) and 
          (dimy==0 or dimy==self.size-1)): return 0
        if((dimx==self.size-1) and 
          (dimy<=B.X[0] or dimy>=self.size-B.X[0])):return 0
    else:
      if B.X[1]!=-1:
        if B.X[1]>=dimx and dimy==1: return 0
    if( (B.row & 1<<dimx) or 
        (B.left & 1<<(self.size-1-dimx+dimy)) or
        (B.down & 1<<dimy) or
        (B.right & 1<<(dimx+dimy))): return 0
    B.row|=1<<dimx
    B.left|=1<<(self.size-1-dimx+dimy)
    B.down|=1<<dimy
    B.right|=1<<(dimx+dimy)
    B.X[dimx]=dimy
    return 1
  #
  # チェーンのビルド
  def buildChain(self,pres_a,pres_b):
    #
    # Boardクラスのインスタンス
    B=Board(self.size)
    # 
    wB=copy.deepcopy(B)
    for w in range( (self.size//2)*(self.size-3) +1):
      B=copy.deepcopy(wB)
      # １．０行目と１行目にクイーンを配置
      if self.placement(B,0,pres_a[w])==0:
        continue
      if self.placement(B,1,pres_b[w])==0:
        continue
      # ２．９０度回転
      nB=copy.deepcopy(B)
      mirror=(self.size-2)*(self.size-1)-w
      for n in range(w,mirror,1):
        B=copy.deepcopy(nB)
        if self.placement(B,pres_a[n],self.size-1)==0:
          continue
        if self.placement(B,pres_b[n],self.size-2)==0:
          continue
        # ３．９０度回転
        eB=copy.deepcopy(B)
        for e in range(w,mirror,1):
          B=copy.deepcopy(eB)
          if self.placement(B,self.size-1,self.size-1-pres_a[e])==0:
            continue
          if self.placement(B,self.size-2,self.size-1-pres_b[e])==0:
            continue
          # ４．９０度回転
          sB=copy.deepcopy(B)
          for s in range(w,mirror,1):
            B=copy.deepcopy(sB)
            if self.placement(B,self.size-1-pres_a[s],0)==0:
              continue
            if self.placement(B,self.size-1-pres_b[s],1)==0:
              continue
            # 対象解除法
            self.carryChainSymmetry(B,n,w,s,e)
  #
  # チェーンの初期化
  def initChain(self,pres_a,pres_b):
    idx=0
    for a in range(self.size):
      for b in range(self.size):
        if (a>=b and (a-b)<=1) or (b>a and (b-a<=1)):
          continue
        pres_a[idx]=a
        pres_b[idx]=b
        idx+=1
  #
  # キャリーチェーン
  def carryChain(self):
    pres_a=[0]*930
    pres_b=[0]*930
    self.initChain(pres_a,pres_b)     # チェーンの初期化
    self.buildChain(pres_a,pres_b)    # チェーンのビルド
  #
  # ユニーク数の集計
  def getUnique(self):
    return self.COUNTER[0]+self.COUNTER[1]+self.COUNTER[2]
  #
  # 合計解の集計
  def getTotal(self):
    return self.COUNTER[0]*2+self.COUNTER[1]*4+self.COUNTER[2]*8
#
# メイン
def main():
  nmin = 5
  nmax = 21
  print("キャリーチェーン クラスの導入")
  print(" N:        Total       Unique        hh:mm:ss.ms")
  for size in range(nmin, nmax,1):
    start_time = datetime.now()
    nq=nQueens(size)
    time_elapsed = datetime.now() - start_time
    _text = '{}'.format(time_elapsed)
    text = _text[:-3]
    print("%2d:%13d%13d%20s" % (size, nq.getTotal(),nq.getUnique(),text))  # 出力
#
main()
#
