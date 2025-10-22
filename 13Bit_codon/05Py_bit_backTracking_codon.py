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


"""
05Py_bit_backTracking_codon.py（レビュー＆注釈つき）

ユーザー提供のビット演算バックトラック版をベースに、
- 行レベルの詳細コメントを付与
- Codon での静的フィールド宣言（クラス先頭）を追加
- 速度面の小改善（mask を毎フレーム再計算しない）
- ループ範囲を max を含む形に変更

本段は **ビットボード（列/対角をビットで表現）** による素朴バックトラック。
対称性削減は未実装なので `unique` は 0 のままです（次段で導入）。

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
  # --- 結果/設定（Codon 向けに先頭で型宣言） ---
  total:int
  unique:int
  mask:int          # 下位 N ビットを 1 にした定数（例: N=8 → 0b11111111）
  size:int          # 参照用

  def __init__(self)->None:
    # 実体は run(size) の中で都度 init() する
    pass

  # ------------------------------------------------------------
  # 初期化（サイズに応じた定数の算出）
  # ------------------------------------------------------------
  def init(self,size:int)->None:
    self.total=0
    self.unique=0  # 対称性未実装のため 0 のまま
    self.size=size
    self.mask=(1<<size)-1  # 再帰ごとに作らず 1 回だけ算出

  # ------------------------------------------------------------
  # ビット演算バックトラック本体
  #   row   : 現在の行 index
  #   left  : 1 行前の配置から伝播した ↙︎ 衝突ビット（次行で <<1）
  #   down  : 1 行前までに使用した列の集合
  #   right : 1 行前の配置から伝播した ↘︎ 衝突ビット（次行で >>1）
  # ------------------------------------------------------------
  def dfs(self,row:int,left:int,down:int,right:int)->None:
    if row==self.size:
      self.total+=1
      return
    # 置ける位置の集合
    bitmap:int=self.mask&~(left|down|right)
    while bitmap:
      # 最下位 1bit（LSB）を取り出して配置
      bit:int=-bitmap&bitmap
      # 残りの候補から当該ビットを落とす
      bitmap^=bit  # (= bitmap & ~bit)
      # 次行へ。left は <<1、right は >>1 にシフトして伝播させる
      self.dfs(row+1,(left|bit)<<1,(down|bit),(right|bit)>>1)

  # ------------------------------------------------------------
  # 1 サイズ分を実行
  # ------------------------------------------------------------
  def run(self,size:int)->None:
    self.init(size)
    self.dfs(0,0,0,0)

  # ------------------------------------------------------------
  # CLI 入口
  # ------------------------------------------------------------
  def main(self)->None:
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
