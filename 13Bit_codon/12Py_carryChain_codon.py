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
12Py_carryChain_codon.py（レビュー＆注釈つき）

キャリーチェーン法の実装をレビューし、以下を最小修正しています：
- **安全な反復とインデックス**：`wMirror` を `set(range(...))` から **`range(...)`** に変更
  （順序が壊れると組の対応が崩れ、`pres_a/pres_b` の参照が不正になりうる）。
- **mirror の探索域**：`wMirror = range(w, range_size)` に変更。
  元コードの `sizeEE*sizeE - w` は `pres_*` の実際の埋め数と不整合になり、
  **IndexError** を引き起こす可能性があるため（`initChain` が与える上限は `range_size`）。
- 未使用の `import copy` を削除（独自 `deepcopy` を使用）。
- コメントを追加し、関数の責務と行ごとの意図を明確化。

注：`solve()` は **無限精度整数×ビット反転（~）** を利用するため、
初期パラメータでビット幅が鋭密に管理されている前提（下位 `size`〜`size-?` ビットだけを実質利用）。
`down+1 == 0`（≒ `down == -1`）で**全行配置済み**を検出する古典的テクを踏襲しています。


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
  size:int

  def __init__(self)->None:
    pass

  # ------------------------------------------------------------
  # 部分状態からの再帰（キャリーチェーン）
  #  down+1 == 0（≒ down が -1）で葉（1 解）
  # ------------------------------------------------------------
  def solve(self,row:int,left:int,down:int,right:int)->int:
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
    start_row=B[0]>>2
    start_left=B[1]>>4
    start_down=(((B[2]>>2)|(~0<<(size-4)))+1)<<(size-5)
    start_down-=1
    start_right=(B[3]>>4)<<(size-5)
    return sym*self.solve(start_row,start_left,start_down,start_right)

  def Symmetry(self,size:int,n:int,w:int,s:int,e:int,B:List[int],B4:List[int])->int:
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
    idx: int = 0
    for a in range(size):
      for b in range(size):
        if abs(a - b) <= 1:
          continue
        pres_a[idx], pres_b[idx] = a, b
        idx += 1
    return idx  # 実際に埋めた有効エントリ数を返す

  def carryChain(self, size: int) -> int:
    pres_a: List[int] = [0] * 930
    pres_b: List[int] = [0] * 930
    valid = self.initChain(size, pres_a, pres_b)
    return self.buildChain(size, pres_a, pres_b, valid)

class NQueens12_carryChain:
  def main(self) -> None:
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
