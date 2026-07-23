#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Python/codon Ｎクイーン ノードレイヤー ミラー版

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
N-Queens：ノードレイヤー法 + ミラー対称削減（Totalのみ計数）
==========================================================
ファイル: 10Py_node_layer_mirror_total.py
作成日: 2025-10-23

概要:
  - ビット演算DFSを frontier（深さk）で分割し、部分状態(left,down,right)をノードとして蓄積。
  - 各ノードから完全探索（_solve_from_node）を行うことで Total（全解数）を算出。
  - 左右対称の重複を除くため、初手を左半分に制限し、奇数Nでは中央列を別途処理。
  - Unique（代表解数）は未算出（常に0）。

設計のポイント（実ソース引用）:
  - 可置集合: `bitmap = mask & ~(left | down | right)`
  - LSB抽出:  `bit = -bitmap & bitmap`
  - frontier条件: `_popcount(down) == k`
  - 対称削減:
      - 偶数N → 左半分の初手のみを探索し、結果×2
      - 奇数N → 左半分×2 + 中央列（左右対称で唯一）

利点:
  - ノードレイヤー分割により、frontier以降は完全独立 → 並列化が容易。
  - 左右対称削減により探索量をおよそ半減。
  - Codon環境では @par 付与で大規模並列化に直結。

用途:
  - CPU/GPU 並列化テスト
  - Codon コンパイル・ビット演算最適化実験
  - Total 確認（Unique が不要な性能検証）

著者: suzuki / nqdev
ライセンス: MIT（必要に応じて変更）
"""


"""

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

"""
from datetime import datetime
from typing import List


class NQueens10:
  """
  ノードレイヤー + ミラー対称削減による Total 計数クラス。
  構成:
    - `_collect_nodes`: 深さkのfrontierを再帰的に収集
    - `_collect_nodes_mirror`: 左右対称を考慮したfrontier構築
    - `_solve_from_node`: frontierノードから葉まで完全探索
    - `solve_with_mirror_layer`: 外部API（k指定でTotalを返す）
  注意:
    - Uniqueは算出しない。
    - kを大きくするとfrontierが増える → 並列粒度が細かくなる。
  """

  # Codon向け: フィールドを事前宣言
  size:int
  mask:int

  def __init__(self)->None:
    self.size=0
    self.mask=0

  def _solve_from_node(self,left:int,down:int,right:int)->int:
    """
    役割:
      部分状態 (left, down, right) から完全探索を行い、葉まで到達した解をカウント。
    停止条件:
      `if down == self.mask: return 1`
    コア（引用）:
      - 可置集合: `bitmap = self.mask & ~(left | down | right)`
      - LSB抽出:  `bit = -bitmap & bitmap`
      - 伝播:      `self._solve_from_node((left|bit)<<1, (down|bit), (right|bit)>>1)`
    備考:
      - 各呼び出しは状態をコピーせずビット演算のみで更新するため高速。
      - frontier単位で完全独立 → 並列化ポイントに適する。
    """

    if down==self.mask:
      return 1
    total=0
    bitmap:int=self.mask&~(left|down|right)
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      total+=self._solve_from_node((left|bit)<<1,(down|bit),(right|bit)>>1)
    return total

  @staticmethod
  def _popcount(n:int)->int:
    """
    役割:
      整数 n の set bit 数（1 の数）を返す。
    アルゴリズム:
      Brian Kernighan 法:
        while n:
          n &= n - 1
          count += 1
    """

    c=0
    while n:
      n&=n-1
      c+=1
    return c

  def _collect_nodes(self,k:int,nodes:List[int],left:int,down:int,right:int)->int:
    """
    役割:
      深さ k の frontier ノードを再帰的に収集。
    条件:
      `if self._popcount(down) == k:` → 現在の部分状態を frontier とみなし nodes に push。
    格納形式:
      nodes = [l0, d0, r0, l1, d1, r1, ...]
    コア（引用）:
      - 可置集合: `bitmap = self.mask & ~(left | down | right)`
      - LSB抽出:  `bit = -bitmap & bitmap`
      - 伝播:      `self._collect_nodes(k, nodes, (left|bit)<<1, (down|bit), (right|bit)>>1)`
    戻り値:
      収集したノード数（int）
    """

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

  def _collect_nodes_mirror(self,k:int)->tuple[List[int],List[int]]:
    """
    役割:
      左右対称を考慮した frontier 構築。
    手順:
      - 偶数N: 左半分（0..N//2-1）の初手だけ探索し、結果×2。
      - 奇数N: 左半分に加えて中央列（N//2）を別途処理。
    実装（引用）:
      for col in range(half):
          bit = 1 << col
          self._collect_nodes(k, nodes_left, bit<<1, bit, bit>>1)
      if N が奇数:
          bit = 1 << (N//2)
          self._collect_nodes(k, nodes_center, bit<<1, bit, bit>>1)
    戻り値:
      (nodes_left, nodes_center)
    """

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

  def solve_with_mirror_layer(self,size:int,k:int=4)->int:
    """
    役割:
      ノードレイヤー＋ミラー対称による Total 計数の外部API。
    手順:
      1. `mask = (1 << size) - 1`
      2. `_collect_nodes_mirror(k)` で frontier を構築。
      3. 各ノードについて `_solve_from_node()` で完全探索。
      4. 集計:
         - 偶数N → 左半分結果×2
         - 奇数N → 左半分×2 + 中央列そのまま
    出力:
      Total（全解数）
    チューニング:
      - k を増やすほど並列化粒度が細かくなるが、frontier数が急増。
      - Codon 環境では @par デコレータでこのループを並列化可能。
    """

    self.size=size
    self.mask=(1<<size)-1
    # frontierノード3要素(left,down,right)は完全独立のため、
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
