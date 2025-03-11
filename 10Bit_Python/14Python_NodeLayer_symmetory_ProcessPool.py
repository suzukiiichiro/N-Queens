#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
ノードレイヤー 対象解除 マルチプロセス版Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens
"""

"""
CentOS-5.1$ pypy 16Python_NodeLayer_symmetoryOps_ProcessPool.py
 N:        Total        Unique        hh:mm:ss.ms
 4:            0            0         0:00:00.015
 5:           10            0         0:00:00.025
 6:            4            0         0:00:00.036
 7:           40            0         0:00:00.067
 8:           92            0         0:00:00.096
 9:          352            0         0:00:00.115
10:          724            0         0:00:00.123
11:         2680            0         0:00:00.224
12:        14200            0         0:00:00.250
13:        73712            0         0:00:00.344
14:       365596            0         0:00:00.780
15:      2279184            0         0:00:03.064
16:     14772512            0         0:00:17.305
17:     95815104            0         0:01:59.358
18:    666090624            0         0:14:48.210

CentOS-5.1$ pypy 10Python_bit_symmetry_ProcessPool.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184       285053         0:00:03.215
16:     14772512      1846955         0:00:16.017
17:     95815104     11977939         0:01:39.372
18:    666090624     83263591         0:11:29.141
"""

"""
CentOS-5.1$ pypy 16Python_NodeLayer_symmetoryOps_ProcessPool.py
 N:        Total        Unique        hh:mm:ss.ms
15:      2279184            0         0:00:03.064

CentOS-5.1$ pypy 15Python_NodeLayer_symmetoryOps_class.py
 N:        Total        Unique        hh:mm:ss.ms
15:      2279184            0         0:00:05.425

CentOS-5.1$ pypy 14Python_NodeLayer_symmetoryOps_param.py
 N:        Total        Unique        hh:mm:ss.ms
15:      2279184            0         0:00:06.345

CentOS-5.1$ pypy 13Python_NodeLayer_mirror_ProcessPool.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184            0         0:00:02.926

CentOS-5.1$ pypy 12Python_NodeLayer_mirror.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184            0         0:00:06.241

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
import subprocess
from datetime import datetime

# pypyを使うときは以下を活かしてcodon部分をコメントアウト
# pypy では ThreadPool/ProcessPoolが動きます 
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')

from threading import Thread
from multiprocessing import Pool as ThreadPool
import concurrent
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

#codonの修正点は2点です
#・board:list[int]　にする
#・TOTAL,UNIQUEをなくす
class Local:
  TOPBIT:int
  ENDBIT:int
  LASTMASK:int
  SIDEMASK:int
  BOUND1:int
  BOUND2:int
  board:list[int]
  def __init__(self,TOPBIT:int,ENDBIT:int,LASTMASK:int,SIDEMASK:int,BOUND1:int,BOUND2:int,board:list[int]):
    self.TOPBIT,self.ENDBIT,self.LASTMASK,self.SIDEMASK,self.BOUND1,self.BOUND2,self.board=TOPBIT,ENDBIT,LASTMASK,SIDEMASK,BOUND1,BOUND2,board
class NQueens21:
  def __init__(self):
    pass
  """ビットが1である数をカウント"""
  def count_bits_nodeLayer(self,n:int)->int:
    counter:int=0
    while n:
      n&=n-1
      counter+=1
    return counter
  """対称解除操作"""
  def symmetryOps(self,size:int,local:Local)->int:
    ptn:int=0
    own:int=0
    bit:int=0
    you:int=0
    # 90度回転
    if local.board[local.BOUND2]==1:
      ptn=2
      own=1
      while own<size:
        bit=1
        you=size-1
        while you>=0 and local.board[you]!=ptn and local.board[own] >= bit:
          bit<<=1
          you-=1
        if local.board[own]>bit:
          return 0
        if local.board[own]<bit:
          break
        ptn<<=1
        own+=1
      # 90度回転が同型
      if own>size-1:
        return 2
    # 180度回転
    if local.board[size-1]==local.ENDBIT:
      you=size-2
      own=1
      while own<=size-1:
        bit=1
        ptn=local.TOPBIT
        while ptn!=local.board[you] and local.board[own]>=bit:
          ptn>>=1
          bit<<=1
        if local.board[own]>bit:
          return 0
        if local.board[own]<bit:
          break
        you-=1
        own+=1
      # 180度回転が同型
      if own>size-1:
        return 4
    # 270度回転
    if local.board[local.BOUND1]==local.TOPBIT:
      ptn=local.TOPBIT>>1
      own=1
      while own<=size-1:
        bit=1
        you=0
        while you<size and local.board[you]!=ptn and local.board[own] >= bit:
          bit<<=1
          you+=1
        if local.board[own]>bit:
          return 0
        if local.board[own]<bit:
          break
        ptn>>=1
        own+= 1
    # すべての回転が異なる
    return 8

  """ 角にQがある場合のバックトラック """
  def symmetry_solve_nodeLayer_corner(self,size:int,left:int,down:int,right:int,local:Local)->int:
    counter:int=0
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    row:int=self.count_bits_nodeLayer(down)
    bit:int=0
    if row==(size-1):
      if bitmap:
        return 8
    else:
      if row<local.BOUND1:
        bitmap|=2
        bitmap^=2
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      counter+=self.symmetry_solve_nodeLayer_corner(size,(left|bit)<<1,down|bit,(right|bit)>>1,local)
    return counter


  """ 角にQがない場合のバックトラック """
  def symmetry_solve_nodeLayer(self,size:int,left:int,down:int,right:int,local:Local)->int:
    counter:int=0
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    row:int=self.count_bits_nodeLayer(down)
    bit:int=0
    if row==(size-1):
      if bitmap:
        if (bitmap&local.LASTMASK)==0:
          local.board[row]=bitmap # Qを配置
          return self.symmetryOps(size,local)
    else:
      if row<local.BOUND1:
        bitmap|=local.SIDEMASK
        bitmap^=local.SIDEMASK
      elif row==local.BOUND2:
        if (down&local.SIDEMASK)==0:
          return 0
        if (down&local.SIDEMASK)!=local.SIDEMASK:
          bitmap&=local.SIDEMASK
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      local.board[row]=bit
      counter+=self.symmetry_solve_nodeLayer(size,(left|bit)<<1,down|bit,(right|bit)>>1,local)
    return counter

  """ """
  # def symmetry_solve(self,size:int,left:int,down:int,right:int,local:Local)->int:
  def symmetry_solve(self,value:list)->int:
    size,left,down,right,local=value
    if local.board[0]==1:
      return self.symmetry_solve_nodeLayer_corner(size,left,down,right,local)
    else:
      return self.symmetry_solve_nodeLayer(size,left,down,right,local)

  """ 角にQがある場合のバックトラック """
  def kLayer_nodeLayer_backtrack_corner(self,size:int,nodes:list,k:int,left:int,down:int,right:int,local:Local,local_list:list):
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    bit:int=0
    row:int= self.count_bits_nodeLayer(down)
    if row==k:
      nodes.append(left)
      nodes.append(down)
      nodes.append(right)
      local_list.append(Local(TOPBIT=local.TOPBIT,ENDBIT=local.ENDBIT,LASTMASK=local.LASTMASK,SIDEMASK=local.SIDEMASK,BOUND1=local.BOUND1,BOUND2=local.BOUND2,board=local.board.copy()))
    else:
      if row<local.BOUND1:
        bitmap|=2
        bitmap^=2
      while bitmap:
        bit=-bitmap&bitmap
        bitmap^=bit
        self.kLayer_nodeLayer_backtrack_corner(size,nodes,k,(left|bit)<<1,down|bit,(right|bit)>>1,local,local_list)

  """ 角にQがない場合のバックトラック """
  def kLayer_nodeLayer_backtrack(self,size:int,nodes:list,k:int,left:int,down:int,right:int,local:Local,local_list:list)->None:
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    row:int= self.count_bits_nodeLayer(down)
    bit:int=0
    if row==k:
      nodes.append(left)
      nodes.append(down)
      nodes.append(right)
      local_list.append(Local(TOPBIT=local.TOPBIT,ENDBIT=local.ENDBIT,LASTMASK=local.LASTMASK,SIDEMASK=local.SIDEMASK,BOUND1=local.BOUND1,BOUND2=local.BOUND2,board=local.board.copy()))
      return
    else:
      if row<local.BOUND1:
        bitmap|=local.SIDEMASK
        bitmap^=local.SIDEMASK
      elif row==local.BOUND2:
        if (down&local.SIDEMASK)==0:
          return
        if (down&local.SIDEMASK)!=local.SIDEMASK:
          bitmap&=local.SIDEMASK
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      local.board[row]=bit
      self.kLayer_nodeLayer_backtrack(size,nodes,k,(left|bit)<<1,down|bit,(right|bit)>>1,local,local_list)



  """ kレイヤーのすべてのノードを含むベクトルを返す """
  def kLayer_nodeLayer(self,size:int,nodes:list,k:int,local_list:list):
    local=Local(TOPBIT=1<<(size-1),ENDBIT=0,LASTMASK=0,SIDEMASK=0,BOUND1=2,BOUND2=0,board=[0]*size)






    local.board[0]=1
    bit:int=0
    # 角にQがある場合のバックトラック
    while local.BOUND1>1 and local.BOUND1<size-1:
      if local.BOUND1<size-1:
        bit=1<<local.BOUND1
        local.board[1]=bit
        self.kLayer_nodeLayer_backtrack_corner(size,nodes,k,(2|bit)<<1,1|bit,(2|bit)>>1,local,local_list)
      local.BOUND1+= 1
    local.TOPBIT=1<<(size-1)
    local.ENDBIT=local.TOPBIT>>1
    local.SIDEMASK=local.TOPBIT|1
    local.LASTMASK=local.TOPBIT|1
    local.BOUND1=1
    local.BOUND2=size-2
    # 角にQがない場合のバックトラック
    while local.BOUND1>0 and local.BOUND2<size-1 and local.BOUND1<local.BOUND2:
      if local.BOUND1<local.BOUND2:
        bit=1<<local.BOUND1
        local.board[0]=bit
        self.kLayer_nodeLayer_backtrack(size,nodes,k,bit<<1,bit,bit>>1,local,local_list)
      local.BOUND1+=1
      local.BOUND2-=1
      local.ENDBIT=local.ENDBIT>>1
      local.LASTMASK=(local.LASTMASK<<1)|local.LASTMASK|(local.LASTMASK>>1)

  """ """
  def symmetry_build_nodeLayer(self,size:int)->int:
    # ツリーの3番目のレイヤーにあるノードを生成
    nodes:list[int]=[]
    local_list:list[Local]=[] # Localの配列を用意
    k:int=4 # 3番目のレイヤーを対象
    self.kLayer_nodeLayer(size,nodes,k,local_list)
    # 必要なのはノードの半分だけで、各ノードは3つの整数で符号化
    # ミラーでは/6 を /3に変更する
    num_solutions=len(nodes)//3
    pool=ThreadPool(size)
    params=[(
      size,
      nodes[3*i],
      nodes[3*i+1],
      nodes[3*i+2],
      local_list[i]
    ) for i in range(num_solutions)]
    return sum(list(pool.map(self.symmetry_solve,params)))
""" """
class NQueens21_NodeLayer:
  def finalize(self)->None:
    cmd="killall pypy"  # python or pypy
    p = subprocess.Popen("exec " + cmd, shell=True)
    p.kill()
  def main(self)->None:
    nmin:int=4
    nmax:int=19
    print(" N:        Total        Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      NQ=NQueens21()
      total:int=NQ.symmetry_build_nodeLayer(size)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3] 
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
      self.finalize()
# メイン実行部分
if __name__=="__main__":
  NQueens21_NodeLayer().main()
