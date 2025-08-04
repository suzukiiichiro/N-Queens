#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
キャリーチェーン版Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens
"""

"""
CentOS-5.1$ pypy 17Python_carryChain.py
 N:        Total       Unique        hh:mm:ss.ms
 5:           10            0         0:00:00.001
 6:            4            0         0:00:00.006
 7:           40            0         0:00:00.020
 8:           92            0         0:00:00.066
 9:          352            0         0:00:00.085
10:          724            0         0:00:00.173
11:         2680            0         0:00:00.488
12:        14200            0         0:00:01.305
13:        73712            0         0:00:01.962
14:       365596            0         0:00:03.803
15:      2279184            0         0:00:12.722
16:     14772512            0         0:01:00.409
17:     95815104            0         0:06:10.889
"""

"""
CentOS-5.1$ codon run 17Python_carryChain.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184            0         0:00:11.844

CentOS-5.1$ pypy 16Python_NodeLayer_symmetoryOps_ProcessPool.py
 N:        Total        Unique        hh:mm:ss.ms
15:      2279184            0         0:00:02.911

CentOS-5.1$ pypy 15Python_NodeLayer_symmetoryOps_class.py
 N:        Total        Unique        hh:mm:ss.ms
15:      2279184            0         0:00:05.425

CentOS-5.1$ pypy 14Python_NodeLayer_symmetoryOps_param.py
 N:        Total        Unique        hh:mm:ss.ms
15:      2279184            0         0:00:06.345

CentOS-5.1$ pypy 13Python_NodeLayer_mirror_ProcessPool.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184            0         0:00:02.926

CentOS-5.1$ pypy 11Python_NodeLayer.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184            0         0:00:06.160

CentOS-5.1$ pypy 10Python_bit_symmetry_ProcessPool.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184       285053         0:00:01.998

CentOS-5.1$ pypy 09Python_bit_symmetry_ThreadPool.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184       285053         0:00:02.111

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
from datetime import datetime
import copy

# pypyを使うときは以下を活かしてcodon部分をコメントアウト
import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
# pypy では ThreadPool/ProcessPoolが動きます 
#

# from threading import Thread
# from multiprocessing import Pool as ThreadPool
# import concurrent
# from concurrent.futures import ThreadPoolExecutor
# from concurrent.futures import ProcessPoolExecutor
#

class NQueens17:
  def __init__(self):
    pass
  def carryChain(self,size:int)->int:
    def process(size:int,sym:int,B:list[int])->int:
      def solve(row:int,left:int,down:int,right:int)->int:
        total:int=0
        if not down+1:
          return 1
        while row&1:
          row>>=1
          left<<=1
          right>>=1
        row>>=1           # １行下に移動する
        bitmap:int=~(left|down|right)
        while bitmap!=0:
          bit=-bitmap&bitmap
          total+=solve(row,(left|bit)<<1,down|bit,(right|bit)>>1)
          bitmap^=bit
        return total
      return sym*solve(B[0]>>2,B[1]>>4,(((B[2]>>2|~0<<size-4)+1)<<size-5)-1,B[3]>>4<<size-5) # sym 0:COUNT2 1:COUNT4 2:COUNT8
    def Symmetry(size:int,n:int,w:int,s:int,e:int,B:list[int],B4:list[int])->int:
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
      if not B4[0]: return process(size,8,B)  # COUNT8
      # n,e,s==w の場合は最小値を確認
      if s==w:
        if n!=w or e!=w: return 0
        return process(size,2,B)  # COUNT2
      # e==w は180度回転して同じ
      if e==w and n>=s:
        if n>s: return 0
        return process(size,4,B)  # COUNT4
      # その他の場合
      return process(size,8,B)    # COUNT8
    def placement(size:int,dimx:int,dimy:int,B:list[int],B4:list[int])->int:
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
    def buildChain(size:int,pres_a:list[int],pres_b:list[int])->int:
      def deepcopy(lst:list[int])->list:
        return [deepcopy(item) if isinstance(item,list) else item for item in lst]
      total:int=0
      B:list[int]=[0,0,0,0]
      B4:list[int]=[-1]*size  # Bの初期化
      sizeE:int=size-1
      sizeEE:int=size-2
      range_size:int=(size//2)*(size-3)+1
      for w in range(range_size):
        wB,wB4=deepcopy(B),deepcopy(B4)
        # １．０行目と１行目にクイーンを配置
        if not placement(size,0,pres_a[w],wB,wB4) or not placement(size,1,pres_b[w],wB,wB4): continue
        # ２．９０度回転
        wMirror=set(range(w,(sizeEE)*(sizeE)-w,1))
        for n in wMirror:
          nB,nB4=deepcopy(wB),deepcopy(wB4)
          if not placement(size,pres_a[n],sizeE,nB,nB4) or not placement(size,pres_b[n],sizeEE,nB,nB4): continue
          # ３．９０度回転
          for e in wMirror:
            eB,eB4=deepcopy(nB),deepcopy(nB4)
            if not placement(size,sizeE,sizeE-pres_a[e],eB,eB4) or not placement(size,sizeEE,sizeE-pres_b[e],eB,eB4): continue
            # ４．９０度回転
            for s in wMirror:
              sB,sB4=deepcopy(eB),deepcopy(eB4)
              if not placement(size,sizeE-pres_a[s],0,sB,sB4) or not placement(size,sizeE-pres_b[s],1,sB,sB4): continue
              # 対象解除法
              total+=Symmetry(size,n,w,s,e,sB,sB4)
      return total
    def initChain(size:int,pres_a:list[int],pres_b:list[int])->None:
      idx:int=0
      for a in range(size):
        for b in range(size):
          # if (a>=b and (a-b)<=1) or (b>a and (b-a<=1)):
          if abs(a-b)<=1: continue
          # pres_a[idx]=a
          # pres_b[idx]=b
          pres_a[idx],pres_b[idx]=a,b
          idx+=1
    pres_a:list[int]=[0]*930
    pres_b:list[int]=[0]*930
    initChain(size,pres_a,pres_b)
    return buildChain(size,pres_a,pres_b)
class NQueens17_carryChain():
  def main(self)->None:
    nmin:int=5
    nmax:int=18
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      NQ=NQueens17()
      total:int=NQ.carryChain(size)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
""" メイン実行部分 """
if __name__=="__main__":
    NQueens17_carryChain().main()
