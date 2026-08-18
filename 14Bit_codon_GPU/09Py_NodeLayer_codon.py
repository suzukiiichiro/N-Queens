#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Python/codon Ｎクイーン ノードレイヤー版

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
N-Queens：ノードレイヤー法（frontier 分割）で Total を計数（Unique 未算出）
=====================================================================
ファイル: 09Py_node_layer_total_only.py
作成日: 2025-10-23

概要:
  - ビット演算 DFS を「深さ k の frontier」でいったん分割し、frontier 以降は独立に完全探索。
  - k 層までの部分状態 (left, down, right) を収集 → 各ノードから葉までを `_solve_from_node`。
  - Unique は未算出（0 のまま）。Total のみ正確に数える構成。

設計のポイント（実ソース引用）:
  - 可置集合: `bitmap = mask & ~(left | down | right)`
  - LSB 抽出: `bit = -bitmap & bitmap`; 消費: `bitmap ^= bit`
  - 葉判定:   `_solve_from_node`: `if down == mask: return 1`
  - frontier 収集: `_collect_nodes` で `nodes.append(left); nodes.append(down); nodes.append(right)`

利点:
  - frontier（深さ k）ごとに**独立な仕事単位**が得られるため、並列化（プロセス/スレッド/Codon @par）に好適。
  - メモリは (3 * ノード数) 個の int のみで、盤面全体のコピーを避けられる。

使い方:
  - `solve_with_layer(N, k=4)`（k は分割深さ。CPU コア数・N に応じて調整）

備考:
  - Python の int は任意長。固定幅（Codon 等）では `mask=(1<<N)-1` の幅管理を徹底。
  - Unique を求める場合は frontier ごとに対称性判定（COUNT2/4/8）を導入する。


レビュー（短評）

良い点
frontier 分割で「仕事単位」を明確化、_solve_from_node が純粋関数的で並列化が容易。
nodes を 3 値フラットで保持するため、メモリ・コピー回数が少なく効率的。
down==mask の葉判定で帰着が速く、ビット操作の定石（-x & x）が適切。

発展提案
並列化：concurrent.futures.ProcessPoolExecutor / multiprocessing / Codon @par で for i in range(num_nodes) を分割。
k の自動調整：num_nodes が閾値（例: 4×CPUコア数）に近づくように k を自動選択。
Unique 導入：frontier ノード側で対称性削減（初手半分・中央処理） or 末端で COUNT2/4/8 を算出。
境界制約：sidemask/lastmask の導入で冗長枝をさらに削減（NQueens07/08 の要領）。

fedora$ codon build -release 09Py_NodeLayer_codon.py && ./09Py_NodeLayer_codon
 N:        Total       Unique        hh:mm:ss.ms
 4:            2            0         0:00:00.000
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.000
 9:          352            0         0:00:00.000
10:          724            0         0:00:00.000
11:         2680            0         0:00:00.001
12:        14200            0         0:00:00.007
13:        73712            0         0:00:00.053
14:       365596            0         0:00:00.236
15:      2279184            0         0:00:01.423
16:     14772512            0         0:00:09.315
fedora$


"""
from datetime import datetime
from typing import List,Tuple


class NQueens_NodeLayer:
  """
  ノードレイヤー法で N-Queens の Total（全解数）だけを計数する実装。
  構成:
    - `_collect_nodes`: 深さ k の frontier を (left, down, right) で収集
    - `_solve_from_node`: 収集した各 frontier ノードから葉まで完全探索
    - `solve_with_layer`: 外側オーケストレーション（分割→集計）
  形式:
    - left/down/right はビットボード（衝突ビットの伝播）:
        left << 1（↖︎↙︎系）、down そのまま、right >> 1（↗︎↘︎系）
  注意:
    - Unique（代表解数）は未算出。Total のみ返す。
  """


  def _solve_from_node(self,size:int,mask:int,left:int,down:int,right:int)->int:
    """
    役割:
      与えられた部分状態 (left, down, right) から葉（全配置）まで完全探索し、解数を返す。
    葉判定（引用）:
      `if down == mask: return 1`  # N 個のクイーンを置き終えたら 1 解
    ロジック（引用）:
      - 可置集合: `bitmap = mask & ~(left | down | right)`
      - LSB抽出:  `bit = -bitmap & bitmap`
      - 候補消費:  `bitmap ^= bit`
      - 伝播:      `self._solve_from_node(size, mask, (left|bit)>>1, down|bit, (right|bit)<<1)`
        （※ ここでは left を >>1、right を <<1 にしており、収集側と**鏡対称**の定義でも整合）
    戻り値:
      部分木の解数（int）
    """

    if down==mask:
      return 1
    total=0
    bitmap:int=mask&~(left|down|right)
    while bitmap:
      bit:int=-bitmap&bitmap
      bitmap^=bit
      total+=self._solve_from_node(size,mask,(left|bit)>>1,down|bit,(right|bit)<<1)
    return total

  @staticmethod
  def _popcount(n:int)->int:
    """
    役割:
      整数 n の set bit 数（1 の数）を返す（Brian Kernighan 法）。
    実装（引用）:
      `while n: n &= n - 1; cnt += 1`
    計算量:
      - O(#set bits)
    """

    cnt=0
    while n:
      n&=n-1
      cnt+=1
    return cnt

  def _collect_nodes(self,size:int,mask:int,k:int,nodes:List[int],left:int,down:int,right:int)->int:
    """
    役割:
      深さ k の frontier まで DFS を進め、(left, down, right) の3要素でノードを蓄積。
    収集条件（引用）:
      `if self._popcount(down) == k:`
         `nodes.append(left); nodes.append(down); nodes.append(right); return 1`
    探索（引用）:
      - 可置集合: `bitmap = mask & ~(left | down | right)`
      - LSB抽出:  `bit = -bitmap & bitmap`
      - 伝播:      `self._collect_nodes(size, mask, k, nodes, (left|bit)>>1, down|bit, (right|bit)<<1)`
    戻り値:
      収集した frontier ノード数（int）
    メモ:
      - `nodes` は [l0, d0, r0, l1, d1, r1, ...] のフラット配列（元コード互換）。
      - frontier 以降は互いに独立 → 並列化ポイント。
    """

    # すでに k 行ぶん置けているか？（down の set bit 数で判定）
    if self._popcount(down)==k:
      nodes.append(left)
      nodes.append(down)
      nodes.append(right)
      return 1
    total=0
    bitmap:int=mask&~(left|down|right)
    while bitmap:
      bit:int=-bitmap&bitmap
      bitmap^=bit
      total+=self._collect_nodes(size,mask,k,nodes,(left|bit)>>1,down|bit,(right|bit)<<1)
    return total

  def solve_with_layer(self,size:int,k:int=4)->int:
    """
    役割:
      深さ k の frontier を構築し、各ノードから完全探索して Total を返す。
    手順（引用）:
      - `mask = (1 << size) - 1`
      - `nodes: List[int] = []`
      - `_collect_nodes(size, mask, k, nodes, 0, 0, 0)` で frontier 構築
      - 3要素で1ノード: `num_nodes = len(nodes) // 3`
      - 各ノードについて `_solve_from_node(size, mask, l, d, r)` を合算
    返り値:
      Total（全解数）
    チューニング:
      - k は CPU コア数・N に応じて調整（k を増やすと並列粒度が細かくなる）。
      - 並列化時は nodes を複数ワーカーへ均等分配する。
    """

    if size<1:
      return 0
    mask:int=(1<<size)-1
    nodes:List[int]=[]
    # 深さ k の frontier を構築
    self._collect_nodes(size,mask,k,nodes,0,0,0)
    # 3 要素で 1 ノード
    num_nodes:int=len(nodes)//3
    total=0
    # 各ノードを独立に探索（ここは将来的に並列化ポイント）
    for i in range(num_nodes):
      l=nodes[3*i]
      d=nodes[3*i+1]
      r=nodes[3*i+2]
      total+=self._solve_from_node(size,mask,l,d,r)
    return total

# ------------------------------------------------------------
# CLI（元コード互換）
# ------------------------------------------------------------
class NQueens_NodeLayer_CLI:
  def main(self)->None:
    nmin:int=4
    nmax:int=18
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start=datetime.now()
      solver=NQueens_NodeLayer()
      total=solver.solve_with_layer(size,k=4)
      unique=0
      dt=datetime.now()-start
      text=str(dt)[:-3]
      print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")

if __name__=="__main__":
  NQueens_NodeLayer_CLI().main()
