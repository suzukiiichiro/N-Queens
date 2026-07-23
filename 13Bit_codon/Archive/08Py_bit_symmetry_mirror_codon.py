#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Python/codon Ｎクイーン bit 対象解除/ミラー版

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
N-Queens：ビットボード + 対称性分類 + 境界制約（最終安定版）
===========================================================
ファイル: 08Py_bitboard_symmetry_final.py
作成日: 2025-10-23

概要:
  - ビット演算によるバックトラック探索を基礎とし、左右対称・回転対称（90°/180°/270°）を考慮。
  - 対称性分類（COUNT2 / COUNT4 / COUNT8）により代表解だけを数え、
    Unique（代表解数）と Total（全解数＝代表×係数）を算出。
  - 境界制約（sidemask, lastmask, bound1/bound2）で枝刈りを行い、冗長探索を排除。

アルゴリズム要点（実ソース引用）:
  - 可置ビット集合: `bitmap = ((1 << size) - 1) & ~(left | down | right)`
  - LSB抽出:         `bit = -bitmap & bitmap`
  - 再帰呼出し:       `self.backTrack*(size, row+1, (left|bit)<<1, (down|bit), (right|bit)>>1)`
  - 対称性分類:
      90° / 180° / 270° 回転および垂直反転の比較により
      count2・count4・count8 のどれに属するかを決定。
  - 枝刈り:
      sidemask（左右端禁止）・lastmask（最終行制約）・bound1/2 により探索領域を限定。

検証の目安:
  N=13 → Unique=9233, Total=73712
  （COUNT2=4, COUNT4=32, COUNT8=9197）

構造:
  - backTrack1(): 角にQがある場合
  - backTrack2(): 角にQがない場合
  - symmetryops(): 回転・反転による同型分類
  - NQueens(): 探索全体のオーケストレーション

著者: suzuki / nqdev
ライセンス: MIT（必要に応じて変更）
"""


"""

fedora$ codon build -release 08Py_bit_symmetry_mirror_codon.py && ./08Py_bit_symmetry_mirror_codon
 N:        Total       Unique        hh:mm:ss.ms
 4:            2            1         0:00:00.000
 5:           10            2         0:00:00.000
 6:            4            1         0:00:00.000
 7:           40            6         0:00:00.000
 8:           92           12         0:00:00.000
 9:          352           46         0:00:00.000
10:          724           92         0:00:00.000
11:         2680          341         0:00:00.000
12:        14200         1787         0:00:00.002
13:        73712         9233         0:00:00.015
14:       365596        45752         0:00:00.080
15:      2279184       285053         0:00:00.418
16:     14772512      1846955         0:00:02.671
fedora$

"""
from datetime import datetime
from typing import List

# pypy を使う場合はコメントを解除（Codon では無効）
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')


class NQueens08:
  """
  ビットボード・対称性・境界制約を統合した N-Queens 完全版。
  特徴:
    - COUNT2/4/8 による同型分類で Unique/Total を算出。
    - sidemask/lastmask による左右・最下行制約で枝刈り。
    - backTrack1: 角にQがあるケース
    - backTrack2: 角にQがないケース
    - symmetryops: 同型判定（回転90/180/270 + 垂直ミラー）
  """

  # --- 結果/状態（Codon向けに事前宣言） ---
  total:int
  unique:int
  board:List[int]
  size:int
  bound1:int
  bound2:int
  topbit:int
  endbit:int
  sidemask:int
  lastmask:int
  count2:int
  count4:int
  count8:int

  def __init__(self)->None:
    # 実体は init(size) で与える
    pass

  def init(self,size:int)->None:
    """
    役割:
      サイズ N に応じて各種作業配列とカウンタを初期化。
    内容（引用）:
      - `board = [0 for _ in range(size)]`  # row→bit 配置
      - `count2 = count4 = count8 = 0`
      - `bound1/bound2 = 0`, `sidemask/lastmask = 0`
    """

    self.total=0
    self.unique=0
    self.board=[0 for _ in range(size)]
    self.size=size
    self.bound1=0
    self.bound2=0
    self.topbit=0
    self.endbit=0
    self.sidemask=0
    self.lastmask=0
    self.count2=0
    self.count4=0
    self.count8=0

  def symmetryops(self,size:int)->None:
    """
    役割:
      現在の配置 `board` に対して、回転・反転の同型をチェックし、
      COUNT2, COUNT4, COUNT8 のいずれかに分類して対応カウントを増加。
    判定ロジック（引用）:
      - 90°:  `if self.board[self.bound2] == 1: ...`
      - 180°: `if self.board[size-1] == self.endbit: ...`
      - 270°: `if self.board[self.bound1] == self.topbit: ...`
    処理概要:
      1. 各角度回転の比較ループで辞書順最小性を判定。
      2. 一致した場合はそのCOUNTに加算して return。
      3. いずれも一致しなければ count8 に加算。
    注意:
      - board[row] は 1 ビットのみ立った列ビット。
      - early return により不一致枝を高速スキップ。
    """

    # 90°
    if self.board[self.bound2]==1:
      own:int=1
      ptn:int=2
      while own<=size-1:
        bit:int=1
        you:int=size-1
        while self.board[you]!=ptn and self.board[own]>=bit:
          bit<<=1
          you-=1
        if self.board[own]>bit:
          return
        if self.board[own]<bit:
          break
        own+=1
        ptn<<=1
      if own>size-1:
        self.count2+=1
        return
    # 180°
    if self.board[size-1]==self.endbit:
      own=1
      you=size-2
      while own<=size-1:
        bit=1
        ptn=self.topbit
        while self.board[you]!=ptn and self.board[own]>=bit:
          bit<<=1
          ptn>>=1
        if self.board[own]>bit:
          return
        if self.board[own]<bit:
          break
        own+=1
        you-=1
      if own>size-1:
        self.count4+=1
        return
    # 270°
    if self.board[self.bound1]==self.topbit:
      own=1
      ptn=self.topbit>>1
      while own<=size-1:
        bit=1
        you=0
        while self.board[you]!=ptn and self.board[own]>=bit:
          bit<<=1
          you+=1
        if self.board[own]>bit:
          return
        if self.board[own]<bit:
          break
        own+=1
        ptn>>=1
    self.count8+=1

  def backTrack2(self,size:int,row:int,left:int,down:int,right:int)->None:
    """
    役割:
      角にQが「ない」ケースの探索。
      上辺/下辺の制約（bound1/bound2）と sidemask/lastmask を活用して枝刈り。

    主な処理（引用）:
      - 可置集合: `bitmap = ((1<<size)-1) & ~(left | down | right)`
      - 末行チェック:
          `if row == size-1 and bitmap and (bitmap & self.lastmask) == 0: ... symmetryops(size)`
      - 上辺制約: `bitmap = (bitmap | self.sidemask) ^ self.sidemask`
      - 下辺制約:
          `if (down & self.sidemask) == 0: return`
          `if (down & self.sidemask) != self.sidemask: bitmap &= self.sidemask`
      - LSB抽出: `bit = -bitmap & bitmap; bitmap ^= bit`
    意義:
      - 左右端・下辺の冗長対称探索を防止。
      - 対称性分類の代表解だけを残す。
    """

    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    if row==(size-1):
      if bitmap:
        if (bitmap&self.lastmask)==0:
          self.board[row]=bitmap
          self.symmetryops(size)
      return
    if row<self.bound1:
      # bitmap &= ~sidemask  を  (bitmap|sidemask) ^ sidemask で実装（分岐なしテク）
      bitmap=(bitmap|self.sidemask)^self.sidemask
    else:
      if row==self.bound2:
        if (down&self.sidemask)==0:
          return
        if (down&self.sidemask)!=self.sidemask:
          bitmap&=self.sidemask
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      # board[row] は「列ビット」を1つだけ立てた整数。例: Qが3列目→0b00001000。
      self.board[row]=bit
      self.backTrack2(size,row+1,(left|bit)<<1,(down|bit),(right|bit)>>1)

  def backTrack1(self,size:int,row:int,left:int,down:int,right:int)->None:
    """
    役割:
      角にQが「ある」ケースの探索。
      左上角を固定した構成から探索を開始し、COUNT8へ直接加算する。

    コア処理（引用）:
      - 可置集合: `bitmap = ((1<<size)-1) & ~(left | down | right)`
      - 上辺制約: `bitmap = (bitmap | 2) ^ 2`   # (= bitmap & ~2)
      - LSB抽出:  `bit = -bitmap & bitmap; bitmap ^= bit`
      - 末行:      `if row == size-1 and bitmap: board[row] = bitmap; count8 += 1`
    意義:
      - 初手固定により探索領域を半分に削減。
      - 角付きパターンは8回対称で自動的に全生成可能。
    """

    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    if row==(size-1):
      if bitmap:
        self.board[row]=bitmap
        self.count8+=1
      return
    if row<self.bound1:
      # bitmap &= ~2 の分岐なし実装
      bitmap=(bitmap|2)^2
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      self.board[row]=bit
      self.backTrack1(size,row+1,(left|bit)<<1,(down|bit),(right|bit)>>1)

  def NQueens(self,size:int)->None:
    """
    役割:
      角あり／角なし探索を切り替えて、COUNT2/4/8 の分類を行うオーケストレーション。

    処理構成（引用）:
      1. リセット:
         `self.count2 = self.count4 = self.count8 = 0`
      2. 角あり探索:
         `self.bound1 = 2; self.board[0] = 1`
         while ループで `bit = 1 << bound1; board[1] = bit; backTrack1(...)`
      3. 角なし探索:
         `self.topbit = 1 << (size-1); self.endbit = topbit >> 1`
         `self.sidemask = topbit | 1; self.lastmask = sidemask`
         while `bound1 < bound2`: `bit = 1 << bound1; board[0] = bit; backTrack2(...)`
         更新:
           `bound1 += 1; bound2 -= 1; endbit >>= 1`
           `lastmask = (lastmask << 1) | lastmask | (lastmask >> 1)`
      4. 合成:
         `unique = count2 + count4 + count8`
         `total  = count2*2 + count4*4 + count8*8`
    """

    self.total=0
    self.unique=0
    self.count2=self.count4=self.count8=0
    # --- 角に Q があるケース ---
    self.topbit=1<<(size-1)
    self.endbit=0
    self.lastmask=0
    self.sidemask=0
    self.bound1=2
    self.bound2=0
    self.board[0]=1
    while self.bound1>1 and self.bound1<size-1:
      if self.bound1<(size-1):
        bit=1<<self.bound1
        self.board[1]=bit
        self.backTrack1(size,2,(2|bit)<<1,(1|bit),(2|bit)>>1)
      self.bound1+=1
    # --- 角に Q がないケース ---
    self.topbit=1<<(size-1)
    self.endbit=self.topbit>>1
    self.sidemask=self.topbit|1
    self.lastmask=self.sidemask
    self.bound1=1
    self.bound2=size-2
    # 原典: while (bound1>0 && bound2<size-1 && bound1<bound2)
    while self.bound1>0 and self.bound2<size-1 and self.bound1<self.bound2:
      if self.bound1<self.bound2:
        bit=1<<self.bound1
        self.board[0]=bit
        self.backTrack2(size,1,bit<<1,bit,bit>>1)
      self.bound1+=1
      self.bound2-=1
      self.endbit>>=1
      # lastmask の漸進更新は左右端の「解空間」を段階的に広げるための境界調整。
      self.lastmask=(self.lastmask<<1)|self.lastmask|(self.lastmask>>1)
    # 合成
    self.unique=self.count2+self.count4+self.count8
    self.total=self.count2*2+self.count4*4+self.count8*8

  def main(self)->None:
    """
    役割:
      N=4..18 までを実行し、Total/Unique と経過時間を表形式で出力。
    出力形式（引用）:
      `print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")`
    注意:
      - range(nmin, nmax) → nmax=18 の直前まで。19を含めるなら +1。
      - 出力I/Oは性能測定の際に抑制を推奨。
    """

    nmin:int=4
    nmax:int=18
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      self.init(size)
      start_time=datetime.now()
      self.NQueens(size)
      dt=datetime.now()-start_time
      text=str(dt)[:-3]
      print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")

if __name__=='__main__':
  NQueens08().main()
