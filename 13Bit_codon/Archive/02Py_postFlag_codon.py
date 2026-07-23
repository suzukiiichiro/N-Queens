#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Python/codon Ｎクイーン ポストフラグ版

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
N-Queens（列ユニーク版 → 対角付き正式版）
=====================================
ファイル: 02Py_postFlag_codon_reviewed.py
作成日: 2025-10-23

概要:
  本ファイルは「段階的な理解」を目的に 2 つの実装を収録。
  1) NQueens02_MinimalFix:
     ・列ユニークの順列列挙のみ（斜め判定なし）
     ・学習用の基礎（バックトラックの形）として有用
     ・終了条件と列使用フラグの扱いを正しく修正
  2) NQueens02_WithDiagonals:
     ・列＋2 方向の対角衝突をフラグで排除する正式版
     ・`ld = row - col + (N-1)`, `rd = row + col` をインデックス化

設計のポイント（実ソース引用）:
  - 列ユニーク制約:
      `if self.used_col[col]==0:`
      `self.used_col[col]=1; ...; self.used_col[col]=0`
  - 対角フラグの算出:
      `ld = row - col + self.offset    # offset = N-1 で 0..(2N-2)`
      `rd = row + col                  # 0..(2N-2)`
    → 使用配列の長さは `2*N-1` を確保:
      `self.used_ld=[0 for _ in range(2*self.size-1)]`
      `self.used_rd=[0 for _ in range(2*self.size-1)]`
  - 停止条件（どちらも同じ）:
      `if row==self.size: self.printout(); return`

使い方:
  $ python3 02Py_postFlag_codon_reviewed.py N [raw]
     - 第2引数が "raw" のとき MinimalFix（列ユニーク順列）を実行
     - 省略時は WithDiagonals（正式版）を実行

計測/運用上の注意:
  - `printout()` の I/O はボトルネック。ベンチ時は表示を切る/カウントのみ推奨。
  - 大きな N ではフラグ配列よりもビットボード（int のビット）に移行すると高速化。
    例: cols/ld/rd をビットで持ち、候補抽出は `bit = free & -free; free &= free - 1`

検証/拡張のヒント:
  - N=8 の正式版の解数は 92（比較用）。回転/鏡の対称性削減（COUNT2/4/8）や
    奇数 N の中央列特別処理などは次段（03 以降）で導入予定。

ライセンス: MIT（必要に応じて変更）
著者: suzuki/nqdev



短評（レビュー）と発展の提案

良い点
列ユニーク→対角追加の二段階構成で学習曲線が明快。
対角のインデックス化（row-col+offset, row+col）が教科書的で分かりやすい。
停止条件とフラグのオン/オフ対称性が明確で、バグになりにくい。

改善の余地（次の段）
・ビットボード化
  used_col/used_ld/used_rd を整数ビットに置換し、
  free = ~(ld|rd|col) & mask → bit = free & -free; free &= free - 1 で高速化。
・対称性削減（COUNT2/4/8）
・初手を左右半分に制限・回転/鏡でユニーク復元 → 枝数をさらに圧縮。
・I/O 抑止/分離
・ベンチ時には printout() を無効化し、count 集計のみのモードを追加。
・入力バリデーション
・N<1 の扱いを明示（0、1 の境界）。N==1 は Total=1 が期待値。


fedora$ codon build -release 02Py_postFlag_codon.py && ./02Py_postFlag_codon 5 raw
:
:
114: 42310
115: 43012
116: 43021
117: 43102
118: 43120
119: 43201
120: 43210

Mode: raw
N: 5
Total: 120
Elapsed: 0.001s


fedora$ codon build -release 02Py_postFlag_codon.py && ./02Py_postFlag_codon 5 proper
:
1: 02413
2: 03142
3: 13024
4: 14203
5: 20314
6: 24130
7: 30241
8: 31420
9: 41302
10: 42031

Mode: proper
N: 5
Total: 10
Elapsed: 0.000s
bash-3.2$

"""

from typing import List
import sys
import time

class NQueens02_MinimalFix:
  """
  学習用の最小バックトラック（列ユニークの順列列挙）。
  機能:
    - 行ごとに未使用の列を 1 つ選んで配置し、N! 通りの順列を列挙。
    - 斜め衝突は「判定しない」（元仕様を保持）。
  特徴（引用）:
    - 列使用フラグ: `self.used_col[col]==0` を満たす列のみ採用
    - 停止条件: `if row==self.size: self.printout(); return`
  注意:
    - N-Queens の正しい解のみを列挙するものではない（斜め無視）。
    - I/O が重いため、大きな N では print を抑止するのが望ましい。
  """
  size:int
  count:int
  aboard:List[int]    # row -> col
  used_col:List[int]  # 列使用フラグ（0/1）

  def __init__(self,size:int)->None:
    """
    役割:
      盤サイズと、行→列の配置配列 `aboard`、列使用フラグ `used_col` を初期化。
    引数:
      size: 盤の大きさ N
    実装（引用）:
      `self.aboard=[0 for _ in range(self.size)]`
      `self.used_col=[0 for _ in range(self.size)]`
    """
    self.size=size
    self.count=0
    self.aboard=[0 for _ in range(self.size)]
    self.used_col=[0 for _ in range(self.size)]

  def printout(self)->None:
    """
    役割:
      現在の配置 `aboard` を 1 行の数字列で出力し、`count` をインクリメント。
    出力例（引用）:
      `print(self.count, end=": "); ...; print(self.aboard[i], end="")`
    注意:
      - ベンチ用途では表示を止めて集計のみ行うと高速。
    """
    self.count+=1
    print(self.count,end=": ")
    for i in range(self.size):
      print(self.aboard[i],end="")
    print("")

  def nqueens(self,row:int)->None:
    """
    役割:
      行 `row` に未使用の列 `col` を割り当て、次行に再帰（列ユニークのみ保証）。
    流れ（引用）:
      - 停止: `if row==self.size: self.printout(); return`
      - 反復: `for col in range(self.size):`
      - 採用: `if self.used_col[col]==0: ...`
      - 更新: `self.used_col[col]=1; ...; self.used_col[col]=0`
    計算量:
      - Θ(N!)（順列列挙）
    """

    # 正しい終了条件: row==size（最後の行も既に配置済みの状態）
    if row==self.size:
      self.printout()
      return
    # 各列を試す（列ユニーク制約のみ）
    for col in range(self.size):
      if self.used_col[col]==0:
        self.aboard[row]=col
        self.used_col[col]=1
        self.nqueens(row+1)
        self.used_col[col]=0

class NQueens02_WithDiagonals:
  """
  列＋2 方向の対角フラグで N-Queens を解く正式版。
  機能:
    - 列使用・左下/右上対角（ld）・右下/左上対角（rd）の 3 種の衝突をフラグで排除。
  特徴（引用）:
    - ld インデックス: `row - col + self.offset`（`offset = N-1`）
    - rd インデックス: `row + col`
    - 使用配列長: `2*N-1`（`0..2N-2` をカバー）
  注意:
    - 配列境界を越えないよう `offset` と配列長に一貫性を持たせる。
    - I/O はベンチ時に抑止推奨。
  """

  size:int
  count:int
  aboard:List[int]
  used_col:List[int]
  used_ld:List[int]
  used_rd:List[int]
  offset:int

  def __init__(self,size:int)->None:
    """
    役割:
      列・対角フラグ配列を確保し、配置配列とカウントを初期化。
    引数:
      size: 盤の大きさ N
    実装（引用）:
      `self.used_ld=[0 for _ in range(2*self.size-1)]`
      `self.used_rd=[0 for _ in range(2*self.size-1)]`
      `self.offset=self.size-1`
    """
    self.size=size
    self.count=0
    self.aboard=[0 for _ in range(self.size)]
    self.used_col=[0 for _ in range(self.size)]
    self.used_ld=[0 for _ in range(2*self.size-1)]
    self.used_rd=[0 for _ in range(2*self.size-1)]
    self.offset=self.size-1  # (row-col) の負値を 0 始まりにずらす

  def printout(self)->None:
    """
    役割:
      正しい N-Queens 解を 1 行の数字列で出力し、`count` をインクリメント。
    注意:
      - 出力が多い場合は計測値に I/O が影響するため注意。
    """

    self.count+=1
    print(self.count,end=": ")
    for i in range(self.size):
      print(self.aboard[i],end="")
    print("")

  def nqueens(self,row:int)->None:
    """
    役割:
      行 `row` において、列/対角のいずれにも衝突しない `col` のみを配置して再帰。
    主要ロジック（引用）:
      `ld = row - col + self.offset   # 0..2N-2`
      `rd = row + col                 # 0..2N-2`
      `if (self.used_col[col] | self.used_ld[ld] | self.used_rd[rd]) == 0:`
         配置 → フラグON → 再帰 → フラグOFF
    停止条件:
      `if row==self.size: self.printout(); return`
    計算量:
      - バックトラック依存（実効は MinimalFix より大幅に小さい）
    """

    if row==self.size:
      self.printout()
      return
    for col in range(self.size):
      ld=row-col+self.offset  # 0..2N-2
      rd=row+col                # 0..2N-2
      if (self.used_col[col]|self.used_ld[ld]|self.used_rd[rd])==0:
        self.aboard[row]=col
        self.used_col[col]=1
        self.used_ld[ld]=1
        self.used_rd[rd]=1
        self.nqueens(row+1)
        self.used_col[col]=0
        self.used_ld[ld]=0
        self.used_rd[rd]=0

def main()->None:
  """
  役割:
    コマンドライン引数を解釈し、列ユニーク版（raw）または正式版（既定）を実行。
  使い方（引用）:
    `python3 02Py_postFlag_codon_reviewed.py N [raw]`
  振る舞い:
    - N の検証: `int(sys.argv[1])` を試み、失敗時にメッセージを表示。
    - 実行: mode が "raw" なら `NQueens02_MinimalFix`、それ以外は `NQueens02_WithDiagonals`。
    - 計測: `time.perf_counter()` で経過秒を計測・表示。
  注意:
    - 出力件数が多い場合は端末 I/O が支配的になり、経過時間が増大。
  """

  # 使い方:
  #   python3 02Py_postFlag_codon_reviewed.py N [raw]
  #   raw を指定すると MinimalFix（列ユニークの順列）を実行。
  #   省略時は WithDiagonals（N-Queens 正式版）を実行。
  n=8
  mode="proper"
  if len(sys.argv)>=2:
    try:
      n=int(sys.argv[1])
    except ValueError:
      print("第1引数 N は整数で指定してください。例: 8")
      return
  if len(sys.argv)>=3 and sys.argv[2].lower()=="raw":
    mode="raw"

  t0=time.perf_counter()

  # MinimalFix（列ユニークの順列）を実行。
  if mode=="raw":
    solver=NQueens02_MinimalFix(n)
    solver.nqueens(0)
    total=solver.count
  # WithDiagonals（N-Queens 正式版）を実行。
  else:
    solver=NQueens02_WithDiagonals(n)
    solver.nqueens(0)
    total=solver.count

  t1=time.perf_counter()

  print(f"\nMode: {mode}")
  print(f"N: {n}")
  print(f"Total: {total}")
  print(f"Elapsed: {t1 - t0:.3f}s")

if __name__=="__main__":
  main()
