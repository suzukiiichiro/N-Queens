#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
ノードレイヤー 対象解除版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens
"""

"""
CentOS-5.1$ pypy 14Python_NodeLayer_symmetoryOps_param.py
 N:        Total        Unique        hh:mm:ss.ms
 4:            0            0         0:00:00.000
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.003
 9:          352            0         0:00:00.009
10:          724            0         0:00:00.014
11:         2680            0         0:00:00.028
12:        14200            0         0:00:00.049
13:        73712            0         0:00:00.224
14:       365596            0         0:00:01.063
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
from datetime import datetime

# pypyを使うときは以下を活かしてcodon部分をコメントアウト
# pypy では ThreadPool/ProcessPoolが動きます 
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')

#codonの修正点は2点です
#・board:list[int]　にする
#・local_list:list　にする
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
  def symmetryOps(self,size:int,TOPBIT:int,ENDBIT:int,BOUND1:int,BOUND2:int,board:list[int])->int:
    ptn:int=0
    own:int=0
    bit:int=0
    you:int=0
    # 90度回転
    if board[BOUND2]==1:
      ptn=2
      own=1
      while own<size:
        bit=1
        you=size-1
        while you>=0 and board[you]!=ptn and board[own] >= bit:
          bit<<=1
          you-=1
        if board[own]>bit:
          return 0
        if board[own]<bit:
          break
        ptn<<=1
        own+=1
      # 90度回転が同型
      if own>size-1:
        return 2
    # 180度回転
    if board[size-1]==ENDBIT:
      you=size-2
      own=1
      while own<=size-1:
        bit=1
        ptn=TOPBIT
        while ptn!=board[you] and board[own]>=bit:
          ptn>>=1
          bit<<=1
        if board[own]>bit:
          return 0
        if board[own]<bit:
          break
        you-=1
        own+=1
      # 180度回転が同型
      if own>size-1:
        return 4
    # 270度回転
    if board[BOUND1]==TOPBIT:
      ptn=TOPBIT>>1
      own=1
      while own<=size-1:
        bit=1
        you=0
        while you<size and board[you]!=ptn and board[own] >= bit:
          bit<<=1
          you+=1
        if board[own]>bit:
          return 0
        if board[own]<bit:
          break
        ptn>>=1
        own+= 1
    # すべての回転が異なる
    return 8

  """ 角にQがある場合のバックトラック """
  def symmetry_solve_nodeLayer_corner(self,size:int,left:int,down:int,right:int,BOUND1:int,board:list[int])->int:
    counter:int=0
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    row:int=self.count_bits_nodeLayer(down)
    bit:int=0
    if row==(size-1):
      if bitmap:
        return 8
    else:
      if row<BOUND1:
        bitmap|=2
        bitmap^=2
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      counter+=self.symmetry_solve_nodeLayer_corner(size,(left|bit)<<1,down|bit,(right|bit)>>1,BOUND1,board)
    return counter


  """ 角にQがない場合のバックトラック """
  def symmetry_solve_nodeLayer(self,size:int,left:int,down:int,right:int,TOPBIT:int,ENDBIT:int,LASTMASK:int,SIDEMASK:int,BOUND1:int,BOUND2:int,board:list[int])->int:
    counter:int=0
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    row:int=self.count_bits_nodeLayer(down)
    bit:int=0
    if row==(size-1):
      if bitmap:
        if bitmap&LASTMASK==0:
          board[row]=bitmap # Qを配置
          return self.symmetryOps(size,TOPBIT,ENDBIT,BOUND1,BOUND2,board)
    else:
      if row<BOUND1:
        bitmap|=SIDEMASK
        bitmap^=SIDEMASK
      elif row==BOUND2:
        if (down&SIDEMASK)==0:
          return 0
        if (down&SIDEMASK)!=SIDEMASK:
          bitmap&=SIDEMASK
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      board[row]=bit
      counter+=self.symmetry_solve_nodeLayer(size,(left|bit)<<1,down|bit,(right|bit)>>1,TOPBIT,ENDBIT,LASTMASK,SIDEMASK,BOUND1,BOUND2,board)
    return counter

  """ """
  def symmetry_solve(self,size:int,left:int,down:int,right:int,local:Local)->int:
    if local.board[0]==1:
      return self.symmetry_solve_nodeLayer_corner(size,left,down,right,local.BOUND1,local.board)
    else:
      return self.symmetry_solve_nodeLayer(size,left,down,right,local.TOPBIT,local.ENDBIT,local.LASTMASK,local.SIDEMASK,local.BOUND1,local.BOUND2,local.board)

  """ 角にQがある場合のバックトラック """
  def kLayer_nodeLayer_backtrack_corner(self,size:int,nodes:list,k:int,left:int,down:int,right:int,TOPBIT:int,ENDBIT:int,LASTMASK:int,SIDEMASK:int,BOUND1:int,BOUND2:int,board:list[int],local_list:list)->None:
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    bit:int=0
    row:int= self.count_bits_nodeLayer(down)
    if row==k:
      nodes.append(left)
      nodes.append(down)
      nodes.append(right)
      local_list.append(Local(TOPBIT,ENDBIT,LASTMASK,SIDEMASK,BOUND1,BOUND2,board.copy()))
    else:
      if row<BOUND1:
        bitmap|=2
        bitmap^=2
      while bitmap:
        bit=-bitmap&bitmap
        bitmap^=bit
        self.kLayer_nodeLayer_backtrack_corner(size,nodes,k,(left|bit)<<1,down|bit,(right|bit)>>1,TOPBIT,ENDBIT,LASTMASK,SIDEMASK,BOUND1,BOUND2,board,local_list)

  """ 角にQがない場合のバックトラック """
  def kLayer_nodeLayer_backtrack(self,size:int,nodes:list,k:int,left:int,down:int,right:int,TOPBIT:int,ENDBIT:int,LASTMASK:int,SIDEMASK:int,BOUND1:int,BOUND2:int,board:list[int],local_list:list)->None:
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    row:int= self.count_bits_nodeLayer(down)
    bit:int=0
    if row==k:
      nodes.append(left)
      nodes.append(down)
      nodes.append(right)
      local_list.append(Local(TOPBIT,ENDBIT,LASTMASK,SIDEMASK,BOUND1,BOUND2,board.copy()))
      return
    else:
      if row<BOUND1:
        bitmap|=SIDEMASK
        bitmap^=SIDEMASK
      elif row==BOUND2:
        if (down&SIDEMASK)==0:
          return
        if (down&SIDEMASK)!=SIDEMASK:
          bitmap&=SIDEMASK
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      board[row]=bit
      self.kLayer_nodeLayer_backtrack(size,nodes,k,(left|bit)<<1,down|bit,(right|bit)>>1,TOPBIT,ENDBIT,LASTMASK,SIDEMASK,BOUND1,BOUND2,board,local_list)



  """ kレイヤーのすべてのノードを含むベクトルを返す """
  def kLayer_nodeLayer(self,size:int,nodes:list,k:int,local_list:list):
    TOPBIT=1<<(size-1)
    ENDBIT=0
    LASTMASK=0
    SIDEMASK=0
    BOUND1=2
    BOUND2=0
    board=[0]*size
    board[0]=1
    bit:int=0
    # 角にQがある場合のバックトラック
    while BOUND1>1 and BOUND1<size-1:
      if BOUND1<size-1:
        bit=1<<BOUND1
        board[1]=bit
        self.kLayer_nodeLayer_backtrack_corner(size,nodes,k,(2|bit)<<1,1|bit,(2|bit)>>1,TOPBIT,ENDBIT,LASTMASK,SIDEMASK,BOUND1,BOUND2,board,local_list)
      BOUND1+= 1
    TOPBIT=1<<(size-1)
    ENDBIT=TOPBIT>>1
    SIDEMASK=TOPBIT|1
    LASTMASK=TOPBIT|1
    BOUND1=1
    BOUND2=size-2
    # 角にQがない場合のバックトラック
    while BOUND1>0 and BOUND2<size-1 and BOUND1<BOUND2:
      if BOUND1<BOUND2:
        bit=1<<BOUND1
        board[0]=bit
        self.kLayer_nodeLayer_backtrack(size,nodes,k,bit<<1,bit,bit>>1,TOPBIT,ENDBIT,LASTMASK,SIDEMASK,BOUND1,BOUND2,board,local_list)
      BOUND1+=1
      BOUND2-=1
      ENDBIT=ENDBIT>>1
      LASTMASK=(LASTMASK<<1)|LASTMASK|(LASTMASK>>1)

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
    return sum( self.symmetry_solve(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2],local_list[i]) for i in range(num_solutions) )

""" """
class NQueens21_NodeLayer:
  def main(self)->None:
    nmin:int=4
    nmax:int=16
    print(" N:        Total        Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      NQ=NQueens21()
      total:int=NQ.symmetry_build_nodeLayer(size)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3] 
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
# メイン実行部分
if __name__=="__main__":
  NQueens21_NodeLayer().main()
