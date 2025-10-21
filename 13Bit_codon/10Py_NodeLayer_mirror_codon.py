#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
ノードレイヤー ミラー版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

fedora$ codon build -release 10Py_NodeLayer_mirror_codon.py && ./10Py_NodeLayer_mirror_codon
 N:        Total       Unique        hh:mm:ss.ms
 4:            2            0         0:00:00.000
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.000
 9:          352            0         0:00:00.000
10:          724            0         0:00:00.000
11:         2680            0         0:00:00.000
12:        14200            0         0:00:00.006
13:        73712            0         0:00:00.025
14:       365596            0         0:00:00.135
15:      2279184            0         0:00:00.767
16:     14772512            0         0:00:04.682

10Py_NodeLayer_mirror_codon.py（レビュー＆注釈つき）

目的:
- ノードレイヤー法 + 左右ミラー対称を用いて探索量を半減。
- 偶数Nは「左半分のみ列挙→×2」、奇数Nは中央列を別途1回だけ列挙（×2しない）。
- 収集した frontier ノード（left,down,right）から完全探索して合計。

主な修正/指摘:
- **Codon 向けに全フィールドをクラス先頭で型宣言**。
- `bitmap_solve_nodeLayer()` と `kLayer_nodeLayer_recursive()` の**シフト規約を統一**
  （次行へ `left<<=1`, `right>>=1` の伝播）。
- **奇数Nの中央列**はミラーと同一になるため **倍加しない**。左半分と中央を分離集計。
- `mask` は毎回 `(1<<size)-1` を作る代わりに、呼び出しごとに渡す/初期化で保持。
- `bitmap` 更新は `^=`（= `&~bit`）で簡潔に。

用語:
- `left`  : ↙（次行で `<<1`）
- `down`  : 縦（列）
- `right` : ↘（次行で `>>1`）
- `mask`  : 下位 N ビットが 1

既知の落とし穴（元コードのバグポイント）:
- **odd N で中央列も×2してしまう**と重複カウントになります（本修正版は分離集計）。
"""
from datetime import datetime
from typing import List


class NQueens10:
  # Codon向け: フィールドを事前宣言
  size:int
  mask:int

  def __init__(self)->None:
    self.size=0
    self.mask=0

  # ------------------------------------------------------------
  # 部分状態からの完全探索（down==mask で 1 解）
  # ------------------------------------------------------------
  def _solve_from_node(self,left:int,down:int,right:int)->int:
    if down==self.mask:
      return 1
    total=0
    bitmap:int=self.mask&~(left|down|right)
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      total+=self._solve_from_node((left|bit)<<1,(down|bit),(right|bit)>>1)
    return total

  # ------------------------------------------------------------
  # 深さ k まで frontier（ノード: left,down,right）を構築
  # `countBits(down)==k` で停止し、三つ組を nodes に push
  # ------------------------------------------------------------
  @staticmethod
  def _popcount(n:int)->int:
    c=0
    while n:
      n&=n-1
      c+=1
    return c

  def _collect_nodes(self,k:int,nodes:List[int],left:int,down:int,right:int)->int:
    if self._popcount(down)==k:
      nodes.extend((left,down,right))
      return 1
    total=0
    bitmap:int=self.mask&~(left|down|right)
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      total+=self._collect_nodes(k,nodes,(left|bit)<<1,(down|bit),(right|bit)>>1)
    return total

  # ------------------------------------------------------------
  # ミラー対称を使った frontier 構築：
  #  - 偶数N: 左半分の初手のみ（列 0..N/2-1）→ 後で×2
  #  - 奇数N: 左半分（0..N//2-1）を left_half、中央列（N//2）を center として分離集計
  # ------------------------------------------------------------
  def _collect_nodes_mirror(self,k:int)->tuple[List[int],List[int]]:
    nodes_left:List[int]=[]   # 左半分初手のノード
    nodes_center:List[int]=[] # 中央列初手（奇数のみ）
    half=self.size//2
    # 左半分の初手を seeds にして k 層まで掘る
    for col in range(half):
      bit=1<<col
      self._collect_nodes(k,nodes_left,bit<<1,bit,bit>>1)
    # 奇数Nの中央列
    if (self.size&1)==1:
      col=half
      bit=1<<col
      self._collect_nodes(k,nodes_center,bit<<1,bit,bit>>1)
    return nodes_left,nodes_center

  # ------------------------------------------------------------
  # 外側 API：ノードレイヤー + ミラー
  # k は 4 を標準（環境に応じて調整）
  # ------------------------------------------------------------
  def solve_with_mirror_layer(self,size:int,k:int=4)->int:
    self.size=size
    self.mask=(1<<size)-1
    # frontier を構築
    nodes_left,nodes_center=self._collect_nodes_mirror(k)
    total_left=0
    total_center=0
    # 3 要素で 1 ノード
    for i in range(0,len(nodes_left),3):
      total_left+=self._solve_from_node(nodes_left[i],nodes_left[i+1],nodes_left[i+2])
    for i in range(0,len(nodes_center),3):
      total_center+=self._solve_from_node(nodes_center[i],nodes_center[i+1],nodes_center[i+2])
    # 偶数: 全部×2、奇数: 左半分×2 + 中央そのまま
    return (total_left<<1)+total_center

# ------------------------------------------------------------
# CLI（元コード互換。Unique は 0 のまま）
# ------------------------------------------------------------
class NQueens10_NodeLayer:
  def main(self)->None:
    nmin:int=4
    nmax:int=18
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start=datetime.now()
      nq=NQueens10()
      total=nq.solve_with_mirror_layer(size,k=4)
      dt=datetime.now()-start
      text=str(dt)[:-3]
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")

if __name__=="__main__":
  NQueens10_NodeLayer().main()
