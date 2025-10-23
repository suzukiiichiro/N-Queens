#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Python/codon Ｎクイーン bit バックトラッキング版

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

# -*- coding: utf-8 -*-
"""
N-Queens：ビット演算による高速バックトラック（対称性未実装）
==============================================================
ファイル: 05Py_bitboard_basic.py
作成日: 2025-10-23

概要:
  - 列・対角衝突を整数ビットで管理する最小構成のビットボード版。
  - 対称性（COUNT2/4/8）は未実装だが、全解 Total は正確に算出可能。
  - Codon / PyPy 両対応。Codon 実行時は型宣言で最適化。

設計ポイント（実ソース引用）:
  - 下位 N ビットを 1 にしたマスク:
      `self.mask = (1 << size) - 1`
  - 置ける位置の集合:
      `bitmap = self.mask & ~(left | down | right)`
  - 最下位ビットを抽出:
      `bit = -bitmap & bitmap`
  - 衝突伝播:
      - left  : (左下↙︎衝突)  → <<1
      - down  : (縦衝突)      → そのまま
      - right : (右下↘︎衝突)  → >>1

探索構造:
  再帰呼出し: `dfs(row+1, (left|bit)<<1, (down|bit), (right|bit)>>1)`
  停止条件:   `if row == size: self.total += 1; return`

メリット:
  - 1 行あたり O(1) 判定で再帰を進められる。
  - fb/fc 配列が不要になりメモリ定数化。
  - N=13 までの速度は配列版に比べて数十倍高速。

出力:
  - N, Total（全解数）, Unique（未実装のため常に 0）, 経過時間(ms)
  - 表形式で N=4..18 を一括計測。

拡張予定:
  - 対称性（COUNT2/4/8）導入 → Unique 算出
  - 左右ミラー除去（初手制限）による枝刈り
  - 並列化（Codon @par / multiprocessing）
  - CUDA 版への移行

著者: suzuki/nqdev
ライセンス: MIT（必要に応じて変更）



ビット意味：
- `left`  : 左下↙︎方向の衝突（次行で左シフト）
- `down`  : 縦（列）の衝突（そのまま）
- `right` : 右下↘︎方向の衝突（次行で右シフト）
- `mask`  : 盤のビット幅（下位 N ビットが 1）
- `bitmap`: 現在行で置ける安全マス集合 = `mask & ~(left | down | right)`

Codon 互換メモ：クラスの全フィールドは先頭で宣言しておく。


fedora$ codon build -release 05Py_bit_backTracking_codon.py && ./05Py_bit_backTracking_codon
 N:        Total       Unique        hh:mm:ss.ms
 4:            2            0         0:00:00.000
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.000
 9:          352            0         0:00:00.000
10:          724            0         0:00:00.000
11:         2680            0         0:00:00.001
12:        14200            0         0:00:00.008
13:        73712            0         0:00:00.058
14:       365596            0         0:00:00.267
15:      2279184            0         0:00:01.625
16:     14772512            0         0:00:10.842
fedora$

"""
from datetime import datetime
from typing import Optional


class NQueens05:
  """
  ビット演算による N-Queens バックトラックの基本版。
  目的:
    - 配列を使わず、1つの整数で列・対角衝突を管理。
    - 最下位ビット抽出 (-x & x) により候補列を順に展開。
  メンバー:
    total : 全解数（対称性未考慮）
    unique: 代表解数（現段では未使用=0）
    mask  : 下位 N ビットが 1 の定数（例: N=8→0b11111111）
    size  : 盤の大きさ（N）
  特徴:
    - 3ビット系列 (left, down, right) をシフト更新することで
      配列を使わずに対角の衝突情報を継承できる。
  """

  # --- 結果/設定（Codon 向けに先頭で型宣言） ---
  total:int
  unique:int
  mask:int          # 下位 N ビットを 1 にした定数（例: N=8 → 0b11111111）
  size:int          # 参照用

  def __init__(self)->None:
    # 実体は run(size) の中で都度 init() する
    pass

  def init(self,size:int)->None:
    """
    役割:
      サイズ N に合わせて定数を初期化する。
    引数:
      size: 盤の大きさ N（4以上推奨）
    実装（引用）:
      `self.total = 0`
      `self.unique = 0`
      `self.mask = (1 << size) - 1`
    解説:
      - mask は下位 N ビットを 1 にした定数。
        N=8 → 0b11111111
      - これにより `(x & self.mask)` で N 桁以上のビットを除外できる。
    """

    self.total=0
    self.unique=0  # 対称性未実装のため 0 のまま
    self.size=size
    # Codon 実装時には mask を compile-time const にすると LLVM 最適化が有効になる。
    self.mask=(1<<size)-1  # 再帰ごとに作らず 1 回だけ算出

  def dfs(self,row:int,left:int,down:int,right:int)->None:
    """
    役割:
      再帰的に N-Queens の配置を探索し、全解数を加算する。
    引数:
      row   : 現在の行（0-based）
      left  : 左下↙︎方向から伝播する衝突ビット列（次行で <<1）
      down  : すでに使用済みの列ビット集合
      right : 右下↘︎方向から伝播する衝突ビット列（次行で >>1）
    コアロジック（引用）:
      - 置ける位置の集合:
          `bitmap = self.mask & ~(left | down | right)`
      - 最下位ビット抽出:
          `bit = -bitmap & bitmap`
      - 候補を順に消費:
          `bitmap ^= bit`
      - 次行へ伝播:
          `self.dfs(row+1, (left|bit)<<1, (down|bit), (right|bit)>>1)`
    停止条件:
      `if row == self.size: self.total += 1; return`
    計算量:
      - 実効 O(N!) よりはるかに小さく、N=13 程度まで実用的。
    注意:
      - Python の int は無限長だが Codon では 64bit 上限を考慮すること。
      - (left|down|right) のビット長が mask を超えない前提で動作。
    """

    if row==self.size:
      self.total+=1
      return
    # 置ける位置の集合
    bitmap:int=self.mask&~(left|down|right)
    while bitmap:
      # 最下位 1bit（LSB）を取り出して配置
      # 最下位ビット抽出 (-x & x) は “1 ビットだけ立てた整数” を得る定石。
      bit:int=-bitmap&bitmap
      # 残りの候補から当該ビットを落とす
      # bitmap ^= bit で “そのビットを除外”して次の候補へ進む。
      bitmap^=bit  # (= bitmap & ~bit)
      # 次行へ。left は <<1、right は >>1 にシフトして伝播させる
      self.dfs(row+1,(left|bit)<<1,(down|bit),(right|bit)>>1)

  def run(self,size:int)->None:
    """
    役割:
      指定サイズ N の盤面を初期化し、ビット演算バックトラックを実行。
    流れ（引用）:
      `self.init(size)`
      `self.dfs(0, 0, 0, 0)`
    注意:
      - 対称性削減は未実装のため、Total は全解数。
      - Unique は 0 のまま。
    """

    self.init(size)
    self.dfs(0,0,0,0)

  def main(self)->None:
    """
    役割:
      N=4..18 を一括して走査し、Total/Unique/経過時間を表形式で出力する。
    出力（引用）:
      `print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")`
    注意:
      - 経過時間の整形（[:-3]）でミリ秒精度まで表示。
      - Codon 実行時は整数演算の型最適化により桁違いの高速化が得られる。
    """

    nmin:int=4
    nmax:int=18
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax+1):# 18 を含む
      start_time=datetime.now()
      self.run(size)
      dt=datetime.now()-start_time
      text=str(dt)[:-3]
      print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")

if __name__=='__main__':
    NQueens05().main()
