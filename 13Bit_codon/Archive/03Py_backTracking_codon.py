#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Python/codon Ｎクイーン バックトラッキング版

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
N-Queens 基本バックトラック（列・対角フラグ）/ Unique 未算出
===========================================================
ファイル: 03Py_flags_basic.py
作成日: 2025-10-23

概要:
  配列フラグ（列・2方向の対角）を用いて N-Queens の全解 Total を列挙する基本版。
  対称性分類（COUNT2/4/8）による Unique は本段では未算出（0のまま出力）。

データ構造（実ソース引用）:
  - 行→列の配置: `self.aboard[row] = col`
  - 列フラグ:     `self.fa[col]`
  - 左下/右上(ld): `ld = row - col + self._off`（`self._off = size - 1` → 0..2N-2 に平行移動）
  - 右下/左上(rd): `rd = row + col`（0..2N-2）
  - 使用有無の判定:
      `if (self.fa[col] | self.fb[ld] | self.fc[rd]) == 0:`

設計のポイント:
  - 停止条件（末行まで置けたら1解）:
      `if row == size: self.total += 1; return`
  - バックトラックの対称性（置く→再帰→戻す）:
      `self.fa[col]=1; ...; self.fa[col]=0`（ld/rd も同様）
  - 対角配列の長さは `2*size-1` を確保して境界外アクセスを回避。

計測・出力:
  - `main()` で N を 4..18 に振り、`Total`/`Unique(=0)`/経過時間を表示。
  - I/O は遅いので、大きな N は `printout()` を抑止した別モード推奨。

次段の発展:
  - ビットボード化（cols/ld/rd を int のビットで管理、候補抽出 `bit = free & -free`）
  - 対称性削減（COUNT2/4/8）で Unique を算出し、Total を復元
  - PyPy/Codon 最適化（JIT 展開、整数幅の固定、マスク併用）

著者: suzuki/nqdev
ライセンス: MIT（必要に応じて変更）


仕上げのメモ（レビュー要旨）

良い点
ld=row-col+self._off, rd=row+col と 境界を 0..2N-2 に統一しており安全。
置く→再帰→戻す の対称性がきれいで、バグになりにくい。
計測出力の整形（[:-3]）で読みやすい。

次の一手（必要ならコメントも用意できます）
ビットボード化：fa/fb/fc を int に統合し、free から bit=free&-free で候補を高速列挙
対称性削減と Unique 計算：初手の左右半分化＋COUNT2/4/8


Codon/Python 共通で動作。


fedora$ codon build -release 03Py_backTracking_codon.py && ./03Py_backTracking_codon
 N:        Total       Unique        hh:mm:ss.ms
 4:            2            0         0:00:00.000
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.000
 9:          352            0         0:00:00.000
10:          724            0         0:00:00.003
11:         2680            0         0:00:00.011
12:        14200            0         0:00:00.058
13:        73712            0         0:00:00.281
14:       365596            0         0:00:01.665
15:      2279184            0         0:00:11.015
"""
from datetime import datetime
from typing import List

# pypy を使う場合はコメントを解除（Codon では使わないこと）
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')


class NQueens03:
  """
  列・対角フラグによる基本バックトラック版（Unique は未算出）。
  目的:
    - 配列フラグ（列/ld/rd）で衝突を O(1) 判定し、全解 Total を列挙。
    - 対称性分類は未導入のため `unique=0` のまま表示する。
  メンバー:
    total  : 見つかった全解数（対称を含む）
    unique : 本段では未使用（将来 COUNT2/4/8 導入時に算出）
    aboard : row -> col の部分解/完成解格納（デバッグ/出力用）
    fa     : 列使用フラグ（長さ N）
    fb     : 左下↙︎/右上↗︎ 対角(ld) の使用フラグ（長さ 2N-1）
    fc     : 右下↘︎/左上↖︎ 対角(rd) の使用フラグ（長さ 2N-1）
    _off   : ld の負値補正オフセット = N-1（`ld=row-col+_off` を 0..2N-2 へ写像）
    _size  : 参照用にサイズを保持
  注意:
    - 配列境界: fb/fc は 0..(2N-2) を必ずカバー（長さ 2N-1）。
    - I/O（printout）は遅いので、ベンチ時は count 集計のみ推奨。
  """

  """列・対角フラグによる基本バックトラック版（Unique は未算出）。"""
  total:int
  unique:int        # 未使用（0）。対称性分類導入時に算出。
  aboard:List[int]  # row -> col 配置（部分解/完成解の保持）
  fa:List[int]      # 列の使用フラグ（サイズ: size）
  fb:List[int]      # 左下↙︎/右上↗︎ 対角（ld）の使用フラグ
  fc:List[int]      # 右下↘︎/左上↖︎ 対角（rd）の使用フラグ
  _off:int          # (row-col) の負値補正オフセット = size-1
  _size:int         # 参照用に保持（任意） 右下↘︎/左上↖︎ 対角（rd）

  def __init__(self)->None:
    """
    役割:
      インスタンス生成のみを行うプレースホルダ。
      実際の配列確保・カウンタ初期化は `init(size)` で行う。
    注意:
      - `init(size)` を呼ぶ前に `nqueens()` を実行しないこと。
    """

    # 実体は init(size) で与える。ここでは型レベルだけ確保。
    pass

  def init(self,size:int)->None:
    """
    役割:
      盤サイズ `size` に合わせて配列フラグ・配置配列・カウンタを初期化する。
    引数:
      size: 盤の大きさ N（N>=1）
    実装（引用）:
      - `self.aboard = [0 for _ in range(size)]`        # row->col
      - `self.fa = [0 for _ in range(size)]`            # 列フラグ
      - `self.fb = [0 for _ in range(2*size - 1)]`      # ld: 0..2N-2
      - `self.fc = [0 for _ in range(2*size - 1)]`      # rd: 0..2N-2
      - `self._off = size - 1`                          # ld の負値補正
    不変条件:
      - fb, fc の添字は常に 0..2N-2 に収まる（境界外アクセスしない）。
    """

    """サイズに応じて作業配列とカウンタを初期化。"""
    self.total=0
    self.unique=0  # 現段では未算出
    self.aboard=[0 for _ in range(size)]
    # 列フラグは size で十分（0..size-1 の列だけを指す）
    self.fa=[0 for _ in range(size)]
    # 対角は 0..(2*size-2) を張る
    self.fb=[0 for _ in range(2*size-1)]
    self.fc=[0 for _ in range(2*size-1)]
    # ld の添字は (row - col) を (size-1) だけオフセットして 0 始まりにする
    self._off=size-1
    self._size=size

  def nqueens(self,row:int,size:int)->None:
    """
    役割:
      行 `row` に衝突しない列 `col` を選んで配置し、`row+1` 行へ再帰する。
      末行まで配置できたら `self.total += 1`。
    核心ロジック（引用）:
      - 停止条件:
          `if row == size:
               self.total += 1
               return`
      - ld/rd のインデックス:
          `ld = row - col + self._off  # 0..(2*size-2)`
          `rd = row + col              # 0..(2*size-2)`
      - 衝突判定:
          `if (self.fa[col] | self.fb[ld] | self.fc[rd]) == 0:`
               配置 → フラグON → 再帰 → フラグOFF
    計算量:
      - バックトラック依存（平均・最悪とも指数的だが、
        列ユニークのみの順列列挙 Θ(N!) よりは大幅に小さい）。
    注意:
      - 戻しの順序は配置と逆順で対応（fa/fb/fc の対称性を維持）。
      - 将来のビットボード化では `free = ~(ld|rd|col) & mask` や
        `bit = free & -free` を用いる。
    """

    """row 行目に安全に置ける列を試し、再帰で全解を数える。"""
    if row==size:
      # 最後の行まで置けたら 1 解
      self.total+=1
      return
    # この行の全列を試す
    for col in range(size):
      # ld は (row-col) を _off (=N-1) で 0..2N-2 にシフト、rd は (row+col)
      ld=row-col+self._off  # 0..(2*size-2)
      rd=row+col              # 0..(2*size-2)
      # 列/ld/rd のいずれも未使用なら安全（0=未使用）
      if (self.fa[col]|self.fb[ld]|self.fc[rd])==0:
        # 置く
        self.aboard[row]=col
        self.fa[col]=1
        self.fb[ld]=1
        self.fc[rd]=1
        # 次の行へ
        self.nqueens(row+1,size)
        # 戻す（バックトラック）
        self.fa[col]=0
        self.fb[ld]=0
        self.fc[rd]=0

  def main(self)->None:
    """
    役割:
      N の範囲を走査し、各 N で Total/Unique(=0)/経過時間を表形式で出力する。
    出力（引用）:
      `print(" N:        Total       Unique        hh:mm:ss.ms")`
      `print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")`
    実装メモ:
      - `minN=4`, `maxN=18` を inclusive に回す（`range(minN, maxN+1)`）。
      - 経過時間の整形はミリ秒精度まで（`str(dt)[:-3]`）。
    注意:
      - 実測比較の際は端末 I/O の影響に留意（大量出力は計測を歪める）。
      - `Unique` は 0 のまま。次段で COUNT2/4/8 を導入予定。
    """

    # 範囲指定（組み込み関数の min/max と衝突しない名前に変更）
    minN:int=4
    maxN:int=18  # 18 を含めたいので、range は maxN+1 まで回す

    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(minN,maxN+1):
      self.init(size)
      start_time=datetime.now()
      self.nqueens(0,size)
      time_elapsed=datetime.now()-start_time
      # "0:00:00.123456" → ミリ秒精度までに整形（末尾3桁を落とす）
      text=str(time_elapsed)[:-3]
      print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")

if __name__=='__main__':
  NQueens03().main()
