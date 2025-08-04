#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
キャリーチェーン マルチプロセス版Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

fedora$ pypy 16Py_carryChain_ProcessPool_pypy.py
 N:        Total       Unique        hh:mm:ss.ms
 5:           10            0         0:00:00.071
 6:            4            0         0:00:00.062
 7:           40            0         0:00:00.087
 8:           92            0         0:00:00.126
 9:          352            0         0:00:00.192
10:          724            0         0:00:00.302
11:         2680            0         0:00:00.499
12:        14200            0         0:00:00.730
13:        73712            0         0:00:01.230
14:       365596            0         0:00:02.252
15:      2279184            0         0:00:07.212
16:     14772512            0         0:00:26.773
"""
from datetime import datetime
import copy
# のこったプロセスをkillallするために必要
import subprocess

# pypyを使うときは以下を活かしてcodon部分をコメントアウト
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

# pypy では ThreadPool/ProcessPoolが動きます 
from threading import Thread
from multiprocessing import Pool as ThreadPool
import concurrent
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from functools import partial
class NQueens16:
  def __init__(self):
    pass
  def solve(self,row:int,left:int,down:int,right:int)->int:
    total:int=0
    if not down+1:
      return 1
    while row&1:
      row,left,right=row>>1,left<<1,right>>1
    row>>=1           # １行下に移動する
    bitmap:int=~(left|down|right)
    while bitmap!=0:
      bit=-bitmap&bitmap
      total+=self.solve(row,(left|bit)<<1,down|bit,(right|bit)>>1)
      bitmap^=bit
    return total
  def process(self,size:int,sym:int,B:list[int])->int:
    return sym*self.solve(B[0]>>2,B[1]>>4,(((B[2]>>2|~0<<size-4)+1)<<size-5)-1,B[3]>>4<<size-5) # sym 0:COUNT2 1:COUNT4 2:COUNT8
  def Symmetry(self,size:int,n:int,w:int,s:int,e:int,B:list[int],B4:list[int])->int:
    # 前計算
    ww=(size-2) * (size-1)-1-w
    w2=(size-2) * (size-1)-1
    # 対角線上の反転が小さいかどうか確認する
    if s==ww and n<(w2-e): return 0
    # 垂直方向の中心に対する反転が小さいかを確認
    if e==ww and n>(w2-n): return 0
    # 斜め下方向への反転が小さいかをチェックする
    if n==ww and e>(w2-s): return 0
    # 【枝刈り】1行目が角の場合
    if not B4[0]: return self.process(size,8,B)  # COUNT8
    # n,e,s==w の場合は最小値を確認
    if s==w:
      if n!=w or e!=w: return 0
      return self.process(size,2,B)  # COUNT2
    # e==w は180度回転して同じ
    if e==w and n>=s:
      if n>s: return 0
      return self.process(size,4,B)  # COUNT4
    # その他の場合
    return self.process(size,8,B)    # COUNT8
  def placement(self,size:int,dimx:int,dimy:int,B:list[int],B4:list[int])->int:
    if B4[dimx]==dimy: return 1
    if B4[0]:
      if ( B4[0]!=-1 and ((dimx<B4[0] or dimx>=size-B4[0]) and (dimy==0 or dimy==size-1)) ) or ((dimx==size-1) and (dimy<=B4[0] or dimy>=size-B4[0])): return 0
    elif (B4[1]!=-1) and (B4[1]>=dimx and dimy==1): return 0
    if ((B[0]&(1<<dimx)) or B[1]&(1<<(size-1-dimx+dimy))) or (B[2]&(1<<dimy)) or (B[3]&(1<<(dimx+dimy))): return 0
    B[0]|=1<<dimx
    B[1]|=1<<(size-1-dimx+dimy)
    B[2]|=1<<dimy
    B[3]|=1<<(dimx+dimy)
    B4[dimx]=dimy
    return 1
  def thread_run(self,params:list)->int:
    size,pres_a,pres_b,B,B4,w=params
    total:int=0
    sizeE:int=size-1
    sizeEE:int=size-2
    wB,wB4=B[:],B4[:]
    # １．０行目と１行目にクイーンを配置
    if not self.placement(size,0,pres_a[w],wB,wB4) or not self.placement(size,1,pres_b[w],wB,wB4): return total
    # ２．９０度回転
    wMirror=set(range(w,(sizeEE)*(sizeE)-w,1))
    for n in wMirror:
      nB,nB4=wB[:],wB4[:]
      if not self.placement(size,pres_a[n],sizeE,nB,nB4) or not self.placement(size,pres_b[n],sizeEE,nB,nB4): continue
      # ３．９０度回転
      for e in wMirror:
        eB,eB4=nB[:],nB4[:]
        if not self.placement(size,sizeE,sizeE-pres_a[e],eB,eB4) or not self.placement(size,sizeEE,sizeE-pres_b[e],eB,eB4): continue
        # ４．９０度回転
        for s in wMirror:
          sB,sB4=eB[:],eB4[:]
          if not self.placement(size,sizeE-pres_a[s],0,sB,sB4) or not self.placement(size,sizeE-pres_b[s],1,sB,sB4): continue
          # 対象解除法
          total+=self.Symmetry(size,n,w,s,e,sB,sB4)
    return total
  def buildChain(self,size:int,pres_a:list[int],pres_b:list[int])->int:
    total:int=0
    B:list[int]=[0,0,0,0]
    B4:list[int]=[-1]*size  # Bの初期化
    range_size:int=(size//2)*(size-3)+1
    pool=ThreadPool(size)
    params=[(size,pres_a,pres_b,B,B4,w) for w in range(range_size)]
    return sum(list(pool.map(self.thread_run,params)))
  def initChain(self,size:int,pres_a:list[int],pres_b:list[int])->None:
    idx:int=0
    for a in range(size):
      for b in range(size):
        if abs(a-b)<=1: continue
        pres_a[idx],pres_b[idx]=a,b
        idx+=1
  def carryChain(self,size:int)->int:
    pres_a:list[int]=[0]*930
    pres_b:list[int]=[0]*930
    self.initChain(size,pres_a,pres_b)
    return self.buildChain(size,pres_a,pres_b)
class NQueens16_carryChain():
  def finalize(self)->None:
    cmd="killall pypy"  # python or pypy
    p = subprocess.Popen("exec " + cmd, shell=True)
    p.kill()
  def main(self)->None:
    nmin:int=5
    nmax:int=19
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      NQ=NQueens16()
      total:int=NQ.carryChain(size)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
      self.finalize()
if __name__=="__main__":
  NQueens16_carryChain().main()

