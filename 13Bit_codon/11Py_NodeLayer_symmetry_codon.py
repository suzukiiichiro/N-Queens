#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Python/codon Ｎクイーン ノードレイヤー 対象解除版

   ,     #_
   ~\_  ####_        N-Queens
  ~~  \_#####\       https://suzukiiichiro.github.io/
  ~~     \###|       N-Queens for github
  ~~       \#/ ___   https://github.com/suzukiiichiro/N-Queens
   ~~       V~' '->
    ~~~         /
      ~~._.   _/
         _/ _/
       _/m/'

結論から言えば codon for python 17Py_ は GPU/CUDA 10Bit_CUDA/01CUDA_Bit_Symmetry.cu と同等の速度で動作します。

 $ nvcc -O3 -arch=sm_61 -m64 -ptx -prec-div=false 04CUDA_Symmetry_BitBoard.cu && POCL_DEBUG=all ./a.out -n ;
対称解除法 GPUビットボード
20:      39029188884       4878666808     000:00:02:02.52
21:     314666222712      39333324973     000:00:18:46.52
22:    2691008701644     336376244042     000:03:00:22.54
23:   24233937684440    3029242658210     001:06:03:49.29

amazon AWS m4.16xlarge x 1
$ codon build -release 15Py_constellations_optimize_codon.py && ./15Py_constellations_optimize_codon
20:      39029188884                0          0:02:52.430
21:     314666222712                0          0:24:25.554
22:    2691008701644                0          3:29:33.971
23:   24233937684440                0   1 day, 8:12:58.977

python 15py_ 以降の並列処理を除けば python でも動作します
$ python <filename.py>

codon for python ビルドしない実行方法
$ codon run <filename.py>

codon build for python ビルドすればC/C++ネイティブに変換し高速に実行します
$ codon build -release < filename.py> && ./<filename>


詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens
"""


"""
11Py_NodeLayer_symmetry_codon.py（レビュー＆注釈つき）

目的:
- ノードレイヤー法（k行ぶんのfrontier構築） + 対称性分類（COUNT2/4/8）で Total を得る。
- 角あり/角なしの分岐と、BOUND1/BOUND2/TOPBIT/ENDBIT/SIDEMASK/LASTMASK の制約を適用。

主な修正/指摘:
- 【バグ修正】角あり側の k層バックトラック `kLayer_nodeLayer_backtrack_corner()` にて、
  再帰に入る前に `local.board[row] = bit` を**設定していなかった**ため、
  収集した Local（盤面）状態が欠落 → 後段の `symmetryOps()` 判定が誤る可能性。
  → 行を追加して修正。
- Codon 安定性のため、クラス/ローカルのフィールドを型付きで明示。
- コメントを整理し、各関数の目的と重要行の意図を行レベルで記載。

メモ:
- 角ありの葉（row==size-1）では回転が全て異なる前提で 8 倍（count8相当）を返す仕様を維持。
- `bitmap` の更新は `^=`（= `&~bit`）で一貫。


fedora$ codon build -release 11Py_NodeLayer_symmetry_codon.py && ./11Py_NodeLayer_symmetry_codon
 N:        Total        Unique        hh:mm:ss.ms
 4:            0            0         0:00:00.000
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.000
 9:          352            0         0:00:00.000
10:          724            0         0:00:00.000
11:         2680            0         0:00:00.000
12:        14200            0         0:00:00.002
13:        73712            0         0:00:00.015
14:       365596            0         0:00:00.080
15:      2279184            0         0:00:00.423
16:     14772512            0         0:00:02.761
fedora$

"""
from datetime import datetime
from typing import List

class Local:
  TOPBIT:int
  ENDBIT:int
  LASTMASK:int
  SIDEMASK:int
  BOUND1:int
  BOUND2:int
  board:List[int]

  def __init__(self,TOPBIT:int,ENDBIT:int,LASTMASK:int,SIDEMASK:int,BOUND1:int,BOUND2:int,board:List[int])->None:
    self.TOPBIT=TOPBIT
    self.ENDBIT=ENDBIT
    self.LASTMASK=LASTMASK
    self.SIDEMASK=SIDEMASK
    self.BOUND1=BOUND1
    self.BOUND2=BOUND2
    self.board=board

class NQueens11:
  def __init__(self)->None:
    pass

  # ------------------------------------------------------------
  # popcount（Brian Kernighan）
  # ------------------------------------------------------------
  def count_bits_nodeLayer(self,n:int)->int:
    cnt=0
    while n:
      n&=n-1
      cnt+=1
    return cnt

  # ------------------------------------------------------------
  # 対称性分類: 90/180/270 を比較し、2/4/8 を返す（0 は代表でない）
  # board は row→bit のビット列
  # ------------------------------------------------------------
  def symmetryOps(self,size:int,local:Local)->int:
    # 90°
    if local.board[local.BOUND2]==1:
      ptn:int=2
      own:int=1
      while own<size:
        bit:int=1
        you:int=size-1
        while you>=0 and local.board[you]!=ptn and local.board[own]>=bit:
          bit<<=1
          you-=1
        if local.board[own]>bit:
          return 0
        if local.board[own]<bit:
          break
        ptn<<=1
        own+=1
      if own>size-1:
        return 2
    # 180°
    if local.board[size-1]==local.ENDBIT:
      you:int=size-2
      own:int=1
      while own<=size-1:
        bit:int=1
        ptn:int=local.TOPBIT
        while ptn!=local.board[you] and local.board[own]>=bit:
          ptn>>=1
          bit<<=1
        if local.board[own]>bit:
          return 0
        if local.board[own]<bit:
          break
        you-=1
        own+=1
      if own>size-1:
        return 4
    # 270°
    if local.board[local.BOUND1]==local.TOPBIT:
      ptn:int=local.TOPBIT>>1
      own:int=1
      while own<=size-1:
        bit:int=1
        you:int=0
        while you<size and local.board[you]!=ptn and local.board[own]>=bit:
          bit<<=1
          you+=1
        if local.board[own]>bit:
          return 0
        if local.board[own]<bit:
          break
        ptn>>=1
        own+=1
    return 8

  # ------------------------------------------------------------
  # 角に Q がある場合のバックトラック（葉で 8 を返す）
  # ------------------------------------------------------------
  def symmetry_solve_nodeLayer_corner(self,size:int,left:int,down:int,right:int,local:Local)->int:
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    row:int=self.count_bits_nodeLayer(down)
    if row==(size-1):
      if bitmap:
        return 8
      return 0
    if row<local.BOUND1:
      bitmap|=2
      bitmap^=2
    total=0
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      total+=self.symmetry_solve_nodeLayer_corner(size,(left|bit)<<1,(down|bit),(right|bit)>>1,local)
    return total

  # ------------------------------------------------------------
  # 角に Q がない場合のバックトラック（代表のみ symmetryOps へ）
  # ------------------------------------------------------------
  def symmetry_solve_nodeLayer(self,size:int,left:int,down:int,right:int,local:Local)->int:
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    row:int=self.count_bits_nodeLayer(down)
    if row==(size-1):
      if bitmap:
        if (bitmap&local.LASTMASK)==0:
          local.board[row]=bitmap
          return self.symmetryOps(size,local)
      return 0
    if row<local.BOUND1:
      bitmap|=local.SIDEMASK
      bitmap^=local.SIDEMASK
    elif row==local.BOUND2:
      if (down&local.SIDEMASK)==0:
        return 0
      if (down&local.SIDEMASK)!=local.SIDEMASK:
        bitmap&=local.SIDEMASK
    total=0
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      local.board[row]=bit
      total+=self.symmetry_solve_nodeLayer(size,(left|bit)<<1,(down|bit),(right|bit)>>1,local)
    return total

  def symmetry_solve(self,size:int,left:int,down:int,right:int,local:Local)->int:
    if local.board[0]==1:
      return self.symmetry_solve_nodeLayer_corner(size,left,down,right,local)
    else:
      return self.symmetry_solve_nodeLayer(size,left,down,right,local)

  # ------------------------------------------------------------
  # k層の frontiers 収集（各ノードに Local のスナップショットを紐付け）
  # ------------------------------------------------------------
  def kLayer_nodeLayer_backtrack_corner(self,size:int,nodes:List[int],k:int,left:int,down:int,right:int,local:Local,local_list:List[Local])->None:
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    row:int=self.count_bits_nodeLayer(down)
    if row==k:
      nodes.extend((left,down,right))
      local_list.append(Local(TOPBIT=local.TOPBIT,ENDBIT=local.ENDBIT,LASTMASK=local.LASTMASK,SIDEMASK=local.SIDEMASK,BOUND1=local.BOUND1,BOUND2=local.BOUND2,board=local.board.copy()))
      return
    if row<local.BOUND1:
      bitmap|=2
      bitmap^=2
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      # ★ 修正: 角あり側でも board に現在のビットを記録
      local.board[row]=bit
      self.kLayer_nodeLayer_backtrack_corner(size,nodes,k,(left|bit)<<1,(down|bit),(right|bit)>>1,local,local_list)

  def kLayer_nodeLayer_backtrack(self,size:int,nodes:List[int],k:int,left:int,down:int,right:int,local:Local,local_list:List[Local])->None:
    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    row:int=self.count_bits_nodeLayer(down)
    if row==k:
      nodes.extend((left,down,right))
      local_list.append(Local(TOPBIT=local.TOPBIT,ENDBIT=local.ENDBIT,LASTMASK=local.LASTMASK,SIDEMASK=local.SIDEMASK,BOUND1=local.BOUND1,BOUND2=local.BOUND2,board=local.board.copy()))
      return
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
      self.kLayer_nodeLayer_backtrack(size,nodes,k,(left|bit)<<1,(down|bit),(right|bit)>>1,local,local_list)

  def kLayer_nodeLayer(self,size:int,nodes:List[int],k:int,local_list:List[Local])->None:
    local=Local(TOPBIT=1<<(size-1),ENDBIT=0,LASTMASK=0,SIDEMASK=0,BOUND1=2,BOUND2=0,board=[0]*size)
    local.board[0]=1
    # 角あり
    while local.BOUND1>1 and local.BOUND1<size-1:
      if local.BOUND1<size-1:
        bit=1<<local.BOUND1
        local.board[1]=bit
        self.kLayer_nodeLayer_backtrack_corner(size,nodes,k,(2|bit)<<1,(1|bit),(2|bit)>>1,local,local_list)
      local.BOUND1+=1
    # 角なし
    local.TOPBIT=1<<(size-1)
    local.ENDBIT=local.TOPBIT>>1
    local.SIDEMASK=local.TOPBIT|1
    local.LASTMASK=local.TOPBIT|1
    local.BOUND1=1
    local.BOUND2=size-2
    while local.BOUND1>0 and local.BOUND2<size-1 and local.BOUND1<local.BOUND2:
      if local.BOUND1<local.BOUND2:
        bit=1<<local.BOUND1
        local.board[0]=bit
        self.kLayer_nodeLayer_backtrack(size,nodes,k,bit<<1,bit,bit>>1,local,local_list)
      local.BOUND1+=1
      local.BOUND2-=1
      local.ENDBIT>>=1
      local.LASTMASK=(local.LASTMASK<<1)|local.LASTMASK|(local.LASTMASK>>1)

  # ------------------------------------------------------------
  # 外側 API：frontier を作り、各ノードを対称つきで完全探索
  # ------------------------------------------------------------
  def symmetry_build_nodeLayer(self,size:int)->int:
    nodes:List[int]=[]
    local_list:List[Local]=[]
    k:int=4
    self.kLayer_nodeLayer(size,nodes,k,local_list)
    num_nodes=len(nodes)//3
    total=0
    for i in range(num_nodes):
      total+=self.symmetry_solve(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2],local_list[i])
    return total

class NQueens11_NodeLayer:
  def main(self)->None:
    nmin:int=4
    nmax:int=18
    print(" N:        Total        Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      nq=NQueens11()
      total=nq.symmetry_build_nodeLayer(size)
      dt=datetime.now()-start_time
      text=str(dt)[:-3]
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")

if __name__=="__main__":
  NQueens11_NodeLayer().main()
