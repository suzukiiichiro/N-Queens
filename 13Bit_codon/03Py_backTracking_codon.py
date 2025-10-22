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
03Py_backTracking_codon.py（レビュー＆注釈つき）
ユーザー提供の NQueens03 をベースに、以下を実施：
 1) 行レベルを含む詳細コメントの付与（各関数の目的を明示）
 2) 目に見える不具合/紛らわしさの最小修正
    - `fa`（列フラグ）の配列長は `size` で十分（元は 2*size-1 と過剰）
    - `min`/`max` の変数名は組み込み関数と衝突するためリネーム
    - ループの上限は `range(minN, maxN + 1)`（元は 18 を含まず 17 で止まる）
 3) 将来の拡張ポイント（ユニーク解の計算）の注意書きを追加

注意：
- 本段は「列＋2方向対角フラグ」を使った**素朴バックトラック**の完成形です。
- `unique` は本実装では未計算（0 のまま）。対称性（COUNT2/4/8）の分類を導入した段で算出します。

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
    # 実体は init(size) で与える。ここでは型レベルだけ確保。
    pass

  def init(self,size:int)->None:
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
    """row 行目に安全に置ける列を試し、再帰で全解を数える。"""
    if row==size:
      # 最後の行まで置けたら 1 解
      self.total+=1
      return
    # この行の全列を試す
    for col in range(size):
      ld=row-col+self._off  # 0..(2*size-2)
      rd=row+col              # 0..(2*size-2)
      # 列/ld/rd すべて未使用なら安全
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
