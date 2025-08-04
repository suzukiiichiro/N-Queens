#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
ノードレイヤー 対象解除版 クラス Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

fedora$ python 13Py_NodeLayer_symmetry.py
 N:        Total        Unique        hh:mm:ss.ms
 4:            0            0         0:00:00.000
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.000
 9:          352            0         0:00:00.002
10:          724            0         0:00:00.009
11:         2680            0         0:00:00.048
12:        14200            0         0:00:00.242
13:        73712            0         0:00:01.362
14:       365596            0         0:00:08.417
15:      2279184            0         0:00:53.576
16:     14772512            0         0:06:08.755
"""
from datetime import datetime

# pypyを使うときは以下を活かしてcodon部分をコメントアウト
# pypy では ThreadPool/ProcessPoolが動きます 
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

class Local:
  TOPBIT:int
  ENDBIT:int
  LASTMASK:int
  SIDEMASK:int
  BOUND1:int
  BOUND2:int
  board:list[int]
  def __init__(self,TOPBIT:int,ENDBIT:int,LASTMASK:int,SIDEMASK:int,BOUND1:int,BOUND2:int,board:list[int])->None:
    self.TOPBIT,self.ENDBIT,self.LASTMASK,self.SIDEMASK,self.BOUND1,self.BOUND2,self.board=TOPBIT,ENDBIT,LASTMASK,SIDEMASK,BOUND1,BOUND2,board
class NQueens13:
  def __init__(self)->None:
    pass
  def count_bits_nodeLayer(self,n:int)->int:
    """ビットが1である数をカウント"""
    counter:int=0
    while n:
      n&=n-1
      counter+=1
    return counter
  def symmetryOps(self,size:int,local:Local)->int:
    """対称解除操作"""
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
  def symmetry_solve_nodeLayer_corner(self,size:int,left:int,down:int,right:int,local:Local)->int:
    """ 角にQがある場合のバックトラック """
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
  def symmetry_solve_nodeLayer(self,size:int,left:int,down:int,right:int,local:Local)->int:
    """ 角にQがない場合のバックトラック """
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
  def symmetry_solve(self,size:int,left:int,down:int,right:int,local:Local)->int:
    if local.board[0]==1:
      return self.symmetry_solve_nodeLayer_corner(size,left,down,right,local)
    else:
      return self.symmetry_solve_nodeLayer(size,left,down,right,local)
  def kLayer_nodeLayer_backtrack_corner(self,size:int,nodes:list,k:int,left:int,down:int,right:int,local:Local,local_list:list)->None:
    """ 角にQがある場合のバックトラック """
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
  def kLayer_nodeLayer_backtrack(self,size:int,nodes:list,k:int,left:int,down:int,right:int,local:Local,local_list:list)->None:
    """ 角にQがない場合のバックトラック """
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
  def kLayer_nodeLayer(self,size:int,nodes:list,k:int,local_list:list)->None:
    """ kレイヤーのすべてのノードを含むベクトルを返す """
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
  def symmetry_build_nodeLayer(self,size:int)->int:
    """ ツリーの4番目のレイヤーにあるノードを生成 """
    nodes:list[int]=[]
    local_list:list[Local]=[] # Localの配列を用意
    k:int=4 # 4番目のレイヤーを対象
    self.kLayer_nodeLayer(size,nodes,k,local_list)
    # 必要なのはノードの半分だけで、各ノードは3つの整数で符号化
    # ミラーでは/6 を /3に変更する
    num_solutions=len(nodes)//3
    return sum( self.symmetry_solve(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2],local_list[i]) for i in range(num_solutions) )
class NQueens13_NodeLayer:
  def main(self)->None:
    nmin:int=4
    nmax:int=18
    print(" N:        Total        Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      NQ=NQueens13()
      total:int=NQ.symmetry_build_nodeLayer(size)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3] 
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
if __name__=="__main__":
  NQueens13_NodeLayer().main()
