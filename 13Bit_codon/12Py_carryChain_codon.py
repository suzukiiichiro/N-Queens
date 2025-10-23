#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Python/codon Ｎクイーン キャリーチェーン版

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
N-Queens：キャリーチェーン法（carry-chain）による高速全解計数（Unique 未算出）
======================================================================
ファイル: 12Py_carry_chain_total_only.py
作成日: 2025-10-23

概要:
  - 盤外周の 4 辺（W/N/E/S）の配置を「鎖（chain）」として組み上げ、内部はビットDFSで充填。
  - ビットDFSは「キャリーチェーン表現」を使用：`down+1 == 0` を葉判定（= 全行が埋まった）。
  - 周辺4点の整合性を `placement()` で逐次検証し、対称クラス係数（2/4/8）の選択は `Symmetry()` で行う。
  - 本段では Unique は未算出（0 のまま）。Total（全解数）を返す。

キーポイント（実ソース引用）:
  - 葉判定: `if not down+1: return 1`  （`down` が -1 なら全列使用）
  - 候補抽出: `bitmap = ~(left | down | right)`（キャリーチェーン空間）
  - LSB抽出:  `bit = -bitmap & bitmap`
  - 境界スキップ: 先行の空行をまとめて圧縮（`while row & 1: ...`）
  - 外周配置: `placement()` が行/列/対角のビット占有を B[0..3] にエンコードして検証
  - 対称倍率: `Symmetry()` が状況に応じて 2/4/8 を返す（0 は代表でない）

用途:
  - 大きめ N の Total を高速算出
  - 外周の組合せ（pres_a/pres_b）を束ねる「鎖」で探索を分割し、並列化しやすい構造

注意:
  - Python の int は任意長だが、shift 幅は size に応じて妥当性が必要。
  - `placement()` は盤外の不変条件（四辺や対角制約）を厳密にチェックするため、変更時は要回帰テスト。



仕上げのレビュー（要点）

良い点
solve() のキャリーチェーン実装（down+1==0、空行圧縮、~(left|down|right)）が簡潔かつ高速。
外周の組立てと内部充填の分離により、境界条件とDFSが綺麗にデカップル。
Symmetry() で 2/4/8 を切り分け、冗長解を抑制する設計。

改善の余地
size <= 5 付近のシフト境界（size-4, size-5）はガード推奨（テーブルで分岐/早期 return）。
deepcopy 多用は重いので、再利用バッファや「差分適用/巻き戻し」方式で高速化可。
並列化は buildChain の 4 重ループ（w/n/e/s）または pres_* の分割が自然。
検証セット：N=8→Total=92、N=13→Total=73712 の回帰確認を推奨。

fedora$ codon build -release 12Py_carryChain_codon.py && ./12Py_carryChain_codon
 N:        Total       Unique        hh:mm:ss.ms
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.002
 9:          352            0         0:00:00.009
10:          724            0         0:00:00.044
11:         2680            0         0:00:00.109
12:        14200            0         0:00:00.302
13:        73712            0         0:00:00.903
14:       365596            0         0:00:01.998
15:      2279184            0         0:00:05.111
16:     14772512            0         0:00:15.222
fedora$
"""
from datetime import datetime
from typing import List


class NQueens12:
  """
  キャリーチェーン法で N-Queens の Total を計数する実装（Unique 未算出）。
  構成:
    - 外周(W/N/E/S)の候補列を pres_a/pres_b に前計算（initChain）
    - 外周の4点を鎖状に接続しながら配置検証（buildChain/placement）
    - 対称クラス係数（2/4/8）は Symmetry で決定、内部は solve() でビットDFS
  注意:
    - 本実装は Total のみ。Unique は 0 として出力。
  """

  size:int

  def __init__(self)->None:
    pass

  # ------------------------------------------------------------
  # 部分状態からの再帰（キャリーチェーン）
  #  down+1 == 0（≒ down が -1）で葉（1 解）
  # ------------------------------------------------------------
  def solve(self,row:int,left:int,down:int,right:int)->int:
    """
    役割:
      キャリーチェーン空間でのビットDFS。`down+1 == 0` を葉判定として全解数を返す。
    コア（引用）:
      - 葉判定: `if not down+1: return 1`  # down が -1（全列埋まった）なら 1 解
      - 空行圧縮: `while row & 1: row >>= 1; left <<= 1; right >>= 1`
      - 候補集合: `bitmap = ~(left | down | right)`
      - LSB抽出:  `bit = -bitmap & bitmap`
      - 再帰:      `self.solve(row, (left|bit)<<1, (down|bit), (right|bit)>>1)`
    注意:
      - ここでの `row/left/down/right` は通常の N ビット盤ではなく、
        「キャリーチェーン」用の圧縮・シフトを伴う内部表現。
    """

    total:int=0
    if not down+1:
      return 1
    while row&1:
      row>>=1
      left<<=1
      right>>=1
    row>>=1
    bitmap:int=~(left|down|right)
    while bitmap!=0:
      bit=-bitmap&bitmap
      total+=self.solve(row,(left|bit)<<1,(down|bit),(right|bit)>>1)
      bitmap^=bit
    return total

  def process(self,size:int,sym:int,B:List[int])->int:
    """
    役割:
      `B[0..3]`（列/斜線の占有ビット列）からキャリーチェーン初期状態を生成し、
      `sym * solve(...)` を返す。
    生成式（引用）:
      - `start_row  = B[0] >> 2`
      - `start_left = B[1] >> 4`
      - `start_down = (((B[2] >> 2) | (~0 << (size-4))) + 1) << (size-5); start_down -= 1`
      - `start_right= (B[3] >> 4) << (size-5)`
    注意:
      - シフト量（`size-4`, `size-5`）は size に依存。小さな N では境界に注意。
    """

    start_row=B[0]>>2
    start_left=B[1]>>4
    start_down=(((B[2]>>2)|(~0<<(size-4)))+1)<<(size-5)
    start_down-=1
    start_right=(B[3]>>4)<<(size-5)
    return sym*self.solve(start_row,start_left,start_down,start_right)

  def Symmetry(self,size:int,n:int,w:int,s:int,e:int,B:List[int],B4:List[int])->int:
    """
    役割:
      外周 4 点（W/N/E/S）の相対関係から代表性を判定し、対称クラス係数（2/4/8）を返す。
      代表でない場合は 0 を返却。
    ロジック（引用）:
      - 早期棄却（辞書順/境界条件）:
          `ww = (size-2)*(size-1)-1-w`
          `if s == ww and n < (w2 - e): return 0`  など
      - 係数決定:
          `if not B4[0]: return process(size, 8, B)`
          `if s == w: ... return process(size, 2, B)`
          `if e == w and n >= s: ... return process(size, 4, B)`
          それ以外は `process(size, 8, B)`
    注意:
      - B4 は「各列の配置（行インデックス）」の一時表現で、0/±1 判定を含む。
      - 条件は carry-chain の既知則に基づくため、並び替え時はテスト必須。
    """

    ww=(size-2)*(size-1)-1-w
    w2=(size-2)*(size-1)-1
    if s==ww and n<(w2-e):
      return 0
    if e==ww and n>(w2-n):
      return 0
    if n==ww and e>(w2-s):
      return 0
    if not B4[0]:
      return self.process(size,8,B)
    if s==w:
      if n!=w or e!=w:
        return 0
      return self.process(size,2,B)
    if e==w and n>=s:
      if n>s:
        return 0
      return self.process(size,4,B)
    return self.process(size,8,B)

  def placement(self,size:int,dimx:int,dimy:int,B:List[int],B4:List[int])->int:
    """
    役割:
      外周座標 (dimx, dimy) にクイーンを仮配置し、矛盾がなければ B/B4 を更新して 1 を返す。
      矛盾があれば 0 を返す（= その鎖は不成立）。
    判定（引用）:
      - 列/対角の占有（ビット）と衝突チェック:
          `(B[0] & (1<<dimx))`（列）, `B[1]`（↗︎↙︎）, `B[2]`（行）, `B[3]`（↖︎↘︎）
      - B の更新:
          `B[0] |= 1<<dimx; B[1] |= 1<<(size-1-dimx+dimy); ...`
      - B4 の更新:
          `B4[dimx] = dimy`
    追加の境界チェック（引用）:
      - 角付近・辺沿いの配置制限（`B4[0]` などの sentinel を参照）
    """

    if B4[dimx]==dimy:
      return 1
    if B4[0]:
      if ((B4[0]!=-1) and ((dimx<B4[0] or dimx>=size-B4[0]) and (dimy==0 or dimy==size-1))) or ((dimx==size-1) and (dimy<=B4[0] or dimy>=size-B4[0])
      ):
        return 0
    elif (B4[1] != -1) and (B4[1] >= dimx and dimy == 1):
      return 0
    if ((B[0] & (1 << dimx)) or (B[1] & (1 << (size - 1 - dimx + dimy))) or (B[2] & (1 << dimy)) or (B[3] & (1 << (dimx + dimy)))):
      return 0
    B[0] |= 1 << dimx
    B[1] |= 1 << (size - 1 - dimx + dimy)
    B[2] |= 1 << dimy
    B[3] |= 1 << (dimx + dimy)
    B4[dimx] = dimy
    return 1

  def buildChain(self, size: int, pres_a: List[int], pres_b: List[int], valid_count: int) -> int:
    """
    役割:
      あらかじめ生成した外周候補（pres_a/pres_b）から鎖を構築し、
      4 つの辺 W/N/E/S を順に確定しながら、代表性チェック→内部 solve を呼び出して合算。
    流れ（引用）:
      - B/B4 を都度 deepcopy（`board` 状態を枝ごとに独立化）
      - W: `placement(size, 0, pres_a[w])` と `placement(size, 1, pres_b[w])`
      - N: `placement(size, pres_a[n], size-1)` と `... size-2`
      - E: `placement(size, size-1, size-1-pres_a[e])` と `... size-2-pres_b[e]`
      - S: `placement(size, size-1-pres_a[s], 0)` と `... , 1`
      - 最後に `total += Symmetry(size, n, w, s, e, sB, sB4)`
    注意:
      - deepcopy は安全だがコストあり。Codon 等なら構造体のコピー最適化が有効。
    """

    def deepcopy(lst: List[int]) -> List[int]:
      return [deepcopy(item) if isinstance(item, list) else item for item in lst]

    total: int = 0
    B: List[int] = [0, 0, 0, 0]
    B4: List[int] = [-1] * size
    sizeE: int = size - 1
    sizeEE: int = size - 2

    for w in range(valid_count):
      wB, wB4 = deepcopy(B), deepcopy(B4)
      if not self.placement(size, 0, pres_a[w], wB, wB4):
        continue
      if not self.placement(size, 1, pres_b[w], wB, wB4):
        continue
      # ここからの n/e/s は pres_* に実際に入っているインデックスだけを使う
      for n in range(w, valid_count):
        nB, nB4 = deepcopy(wB), deepcopy(wB4)
        if not self.placement(size, pres_a[n], sizeE, nB, nB4):
          continue
        if not self.placement(size, pres_b[n], sizeEE, nB, nB4):
          continue
        for e in range(w, valid_count):
          eB, eB4 = deepcopy(nB), deepcopy(nB4)
          if not self.placement(size, sizeE, sizeE - pres_a[e], eB, eB4):
            continue
          if not self.placement(size, sizeEE, sizeE - pres_b[e], eB, eB4):
            continue
          for s in range(w, valid_count):
            sB, sB4 = deepcopy(eB), deepcopy(eB4)
            if not self.placement(size, sizeE - pres_a[s], 0, sB, sB4):
              continue
            if not self.placement(size, sizeE - pres_b[s], 1, sB, sB4):
              continue
            total += self.Symmetry(size, n, w, s, e, sB, sB4)
    return total

  def initChain(self, size: int, pres_a: List[int], pres_b: List[int]) -> int:
    """
    役割:
      外周に置く候補 (a,b) のペアを列挙。隣接（|a-b|<=1）を除外して pres_* に格納。
    実装（引用）:
      `for a in range(size):`
        `for b in range(size):`
          `if abs(a-b) <= 1: continue`
          `pres_a[idx], pres_b[idx] = a, b; idx += 1`
    戻り値:
      実際に埋めたエントリ数（valid_count）
    """

    idx: int = 0
    for a in range(size):
      for b in range(size):
        if abs(a - b) <= 1:
          continue
        pres_a[idx], pres_b[idx] = a, b
        idx += 1
    return idx  # 実際に埋めた有効エントリ数を返す

  def carryChain(self, size: int) -> int:
    """
    役割:
      pres_a/pres_b を用意して valid_count を受け取り、buildChain() を呼ぶ高位API。
    実装（引用）:
      `pres_a = [0] * 930; pres_b = [0] * 930`
      `valid = self.initChain(size, pres_a, pres_b)`
      `return self.buildChain(size, pres_a, pres_b, valid)`
    注意:
      - バッファ長 930 は上限想定。サイズ拡張時は念のため assert/境界チェック推奨。
    """

    pres_a: List[int] = [0] * 930
    pres_b: List[int] = [0] * 930
    valid = self.initChain(size, pres_a, pres_b)
    return self.buildChain(size, pres_a, pres_b, valid)

class NQueens12_carryChain:
  def main(self) -> None:
    """
    役割:
      N=5..17 を走査し、Total/Unique(=0)/経過時間を表形式で出力。
    出力（引用）:
      `print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")`
    注意:
      - range の上限は含まれない（18 を含めたい場合は nmax=19）。
      - I/O は計測に影響大。必要に応じて出力を抑制。
    """

    nmin: int = 5
    nmax: int = 18
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin, nmax):
      start_time = datetime.now()
      nq = NQueens12()
      total = nq.carryChain(size)
      dt = datetime.now() - start_time
      text = str(dt)[:-3]
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")

if __name__ == "__main__":
  NQueens12_carryChain().main()
