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


"""
06Py_bit_mirror_codon.py（レビュー＆注釈つき）

ユーザー提供の "左右対称（ミラー）活用" ビットバックトラック版を、
Codon 互換と読みやすさを意識して最小修正＆詳細コメントを付与したものです。

要点:
- 1 行目のクイーン配置を左右対称で半分に制限し、探索後に **×2**。奇数 N は中央列を**別途1回**だけ探索。
- `unique` は未算出（常に 0）。本コードの `total` は **対称を含む全解数**（N=8 なら 92）。
- `mask` は毎回計算せず、サイズ確定時に 1 回だけ算出。
- Codon のため、クラス先頭で全フィールドを型付き宣言。

ビット意味:
- `left`  : ↙ 衝突（次行で <<1）
- `down`  : 縦（列）衝突
- `right` : ↘ 衝突（次行で >>1）
- `mask`  : 下位 N ビットが 1（盤面幅）

検算の目安（Total）: N=4→2, N=5→10, N=6→4, N=7→40, N=8→92 …


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
  # --- 結果/設定（Codon 向けに先頭で宣言） ---
  total:int
  unique:int
  size:int
  mask:int

  def __init__(self)->None:
    # 実体は solve() 呼び出し時に設定
    pass

  # ------------------------------------------------------------
  # 内部 DFS: ビット演算バックトラック本体
  # ------------------------------------------------------------
  def _dfs(self,row:int,left:int,down:int,right:int)->None:
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

  # ------------------------------------------------------------
  # 対称活用：1 行目の配置を半分に制限し、探索後に×2。奇数 N は中央列を追加探索。
  # ------------------------------------------------------------
  def solve(self,size:int)->None:
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

  # ------------------------------------------------------------
  # CLI 入口
  # ------------------------------------------------------------
  def main(self)->None:
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
