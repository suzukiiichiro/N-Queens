#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Python/codon Ｎクイーン bit ミラー版

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
N-Queens：ビットボードDFS + 左右対称活用（初手半分＋中央列）
===========================================================
ファイル: 06Py_bitboard_sym_half.py
作成日: 2025-10-23

概要:
  - ビット演算DFSで列・対角衝突を O(1) 判定。
  - 対称活用: 1行目は「左半分のみ」を探索し、探索後に ×2。
    奇数 N は中央列を別途探索（中央は左右対称に含まれないため ×2 しない）。
  - 本段階では Unique は未算出（0のまま）。Total は正確。

実装の要点（実ソース引用）:
  - 可置ビット集合: `bitmap = mask & ~(left | down | right)`
  - LSB 抽出:       `bit = -bitmap & bitmap`
  - 次行への伝播:    `self._dfs(row+1, (left|bit)<<1, (down|bit), (right|bit)>>1)`
  - 初手半分:       `for col in range(size//2): ...; self.total *= 2`
  - 中央列（奇数N）: `if (size & 1) == 1: col = size//2; ...`（×2なし）

注意/メモ:
  - Python int は無限長だが、Codon 等の固定幅では `mask` による幅制約が重要。
  - (left|down|right) は毎手 `mask` と AND を取るので、シフト外ビットは自然に落ちる。
  - Unique（COUNT2/4/8）は未実装。次段で symmetryops / 分類導入。

出力:
  N, Total, Unique(=0), 経過時間（ms相当）を表形式で N=4..18 で出力。

著者: suzuki/nqdev
ライセンス: MIT（必要に応じて変更）


仕上げのレビュー（要点）
良い点
初手半分＋中央列のハンドリングが教科書どおりで、self.total *= 2 の位置も正しい。
DFS核が最小・明快（bitmap / bit / ^= / <<1 >>1 の定石を踏襲）。
mask を都度計算せず初期化で1回だけ作るのは◎。

注意点 / 次の一手
Unique の導入：COUNT2/4/8 を返す symmetryops()（回転・反転の辞書順最小チェック）をビットボード版にも追加予定。
半分探索との整合を保つなら、初手半分＋中央列は継続し、末端で unique += 1; total += coeff に変更。
Codon 固定幅：Codon で 64bit 想定なら size <= 32(～64) で使い分け、mask を型に合わせて明示。
並列化：初手列ごとの DFS を分割して @par or multiprocessing で並列実行しやすい構造。

fedora$ codon build -release 06Py_bit_mirror_codon.py && ./06Py_bit_mirror_codon
 N:        Total       Unique        hh:mm:ss.ms
 4:            2            0         0:00:00.000
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.000
 9:          352            0         0:00:00.000
10:          724            0         0:00:00.000
11:         2680            0         0:00:00.001
12:        14200            0         0:00:00.003
13:        73712            0         0:00:00.029
14:       365596            0         0:00:00.138
15:      2279184            0         0:00:00.848
16:     14772512            0         0:00:05.327
fedora$

"""
from datetime import datetime
from typing import Optional


class NQueens06:
  """
  ビットボードDFSに初手左右対称の削減（半分探索＋奇数の中央列）を組み合わせた Total 計数器。
  構成:
    - size, mask: 盤サイズと下位 N ビットが 1 のマスク（例: N=8→0b11111111）
    - total     : 全解数（対称含む）
    - unique    : 本段では未算出（0）
  特徴（引用）:
    - DFS核: `bitmap = mask & ~(left | down | right)` → `bit = -bitmap & bitmap`
    - 伝播:   `left<<1`, `down`そのまま, `right>>1`
    - 初手半分: `for col in range(size//2): ...; self.total *= 2`
    - 奇数中央: `if (size & 1) == 1: col = size//2; ...`（倍化しない）
  """

  # --- 結果/設定（Codon 向けに先頭で宣言） ---
  total:int
  unique:int
  size:int
  mask:int

  def __init__(self)->None:
    """
    役割:
      インスタンス生成のみ。実際の初期化は solve(size) 内で行う。
    注意:
      - solve(size) 呼び出し前に内部DFSを直接使わないこと。
    """

    # 実体は solve() 呼び出し時に設定
    pass

  def _dfs(self,row:int,left:int,down:int,right:int)->None:
    """
    役割:
      ビット演算によるバックトラックの中核。行 row にクイーンを1つ置き、再帰で次行へ。
    停止条件（引用）:
      `if row == self.size: self.total += 1; return`
    コア（引用）:
      - 可置集合: `bitmap = self.mask & ~(left | down | right)`
      - LSB抽出:  `bit = -bitmap & bitmap`
      - 候補消費:  `bitmap ^= bit`
      - 伝播:      `self._dfs(row+1, (left|bit)<<1, (down|bit), (right|bit)>>1)`
    注意:
      - `mask` によって盤外ビットは自然に落ちる（幅管理）。
      - Python ではオーバーフローはないが、Codon 等ではビット幅前提に一致させること。
    """

    if row==self.size:
      self.total+=1
      return
    bitmap:int=self.mask&~(left|down|right)
    while bitmap:
      bit:int=-bitmap&bitmap   # LSB を抽出
      bitmap^=bit                 # (= bitmap & ~bit)
      self._dfs(row+1,
                (left|bit)<<1,
                (down|bit),
                (right|bit)>>1)

  def solve(self,size:int)->None:
    """
    役割:
      盤サイズ size を設定し、初手左右対称を用いた半分探索＋中央列特別処理で Total を計数。
    初期化（引用）:
      `self.size = size; self.mask = (1 << size) - 1; self.total = 0; self.unique = 0`
    手順（引用）:
      - 左半分のみ探索（列 0..size//2-1）:
          `for col in range(half): bit = 1 << col; self._dfs(1, bit<<1, bit, bit>>1)`
        → 探索完了後に `self.total *= 2`
      - 奇数 N の中央列:
          `if (size & 1) == 1: col = half; bit = 1 << col; self._dfs(1, bit<<1, bit, bit>>1)`
        （中央は左右対称に含まれないため倍化しない）
    正当性メモ:
      - 左右反転で 1手目の対称解が対応付くため、左半分のみの探索で網羅できる。
      - 中央列（奇数N）は反転しても同一配置になるため、独立に1回だけ追加。
    """

    self.size=size
    self.mask=(1<<size)-1
    self.total=0
    self.unique=0  # 本段では未算出
    # 左半分のみ（0..size//2-1）
    half:int=size//2
    for col in range(half):
      bit=1<<col
      self._dfs(1,bit<<1,bit,bit>>1)
    self.total*=2
    # 奇数 N の中央列（左右対称と同一にはならない）
    if (size&1)==1:
      col=half
      bit=1<<col
      self._dfs(1,bit<<1,bit,bit>>1)

  def main(self)->None:
    """
    役割:
      N=4..18 を連続実行し、Total/Unique/経過時間を表形式で出力。
    出力（引用）:
      `print(" N:        Total       Unique        hh:mm:ss.ms")`
      `print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")`
    注意:
      - Unique は未算出（0）。次段で COUNT2/4/8 を導入する。
      - ベンチ用途では標準出力の行数に注意（I/Oは相対的に高コスト）。
    """

    nmin:int=4
    nmax:int=18
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax+1):# 18 を含む
      start=datetime.now()
      self.solve(size)
      dt=datetime.now()-start
      text=str(dt)[:-3]
      print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")

if __name__=='__main__':
    NQueens06().main()
