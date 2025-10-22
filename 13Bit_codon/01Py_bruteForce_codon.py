#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python/codon Ｎクイーン ブルートフォース版

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
ブルートフォース版 Ｎクイーン（レビュー＆注釈つき）

このファイルは、ユーザー提供の 01Py_bluteForce_codon.py をベースに、
1) 問題点の指摘（行レベルコメント）
2) 最小修正（== と is の取り違えなどのバグ修正）
3) 本来の N-Queens（衝突チェックあり）版の併記
を行った“学習用”コードです。

ポイント要約：
- 【重大】`if row is self.size:` は **同一性比較** であり、整数の値比較ではありません。
  ここは `==` を使うべきです。Codon でも Python でもバグになります。
- 元のロジックは「各行に 0..size-1 を無条件に置く」総当り（size^size 通り）の列挙で、
  N-Queens の「利き（列／斜め）を避ける」チェックがありません。
  そのため N=5 のとき 5^5=3,125 行が出力され、末尾が 44444 で終わります（提示ログと一致）。
- ベンチマーク目的なら、`print()` は大きなオーバーヘッドです。計数のみで計るのが良いです。
- 言語的な注意：セミコロン `;` は Python では不要です（読みやすさのため削除推奨）。
- 命名：`aboard` は `queens` などにすると役割が明確です（今回は互換のため維持）。

下に 2 つのクラスを用意します：
- NQueens01_MinimalFix : オリジナルをほぼ保持しつつ「== に修正」だけ適用（学習用）
- NQueens01_ProperBF  : 本来の N-Queens（安全判定 is_safe を導入した素朴バックトラック）

CLI 実行例：
  $ codon build -release 01Py_bluteForce_codon_reviewed.py -o nqueen01
  $ ./nqueen01 5    # ProperBF を呼ぶ（デフォルト）
  $ ./nqueen01 5 raw  # MinimalFix を呼ぶ（“生”の総当り）


fedora$ codon build -release 01Py_bluteForce_codon.py
fedora$ ./01Py_bluteForce_codon 5 raw
:
:
3115: 44424
3116: 44430
3117: 44431
3118: 44432
3119: 44433
3120: 44434
3121: 44440
3122: 44441
3123: 44442
3124: 44443
3125: 44444

Mode: raw
N: 5
Total: 3125
Elapsed: 0.026s
bash-3.2$ exit
exit

fedora$ codon build -release 01Py_bluteForce_codon.py
fedora$ ./01Py_bluteForce_codon 5
:
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
fedora$
"""
from typing import List,Tuple
import sys
import time

# ------------------------------------------------------------
# 1) 最小修正版：元の挙動（size^size 通り）を保ちつつ致命的バグを修正
#    ※ N-Queens の解にはなりません。学習用参考。
# ------------------------------------------------------------
class NQueens01_MinimalFix:
  size:int
  aboard:List[int]
  count:int

  def __init__(self,size:int)->None:
    # 目的：サイズを保持し、各行の暫定列を保持する配列 aboard を初期化
    self.size=size
    self.aboard=[0 for _ in range(self.size)]
    self.count=0

  def printout(self)->None:
    # 目的：現在の aboard を 1 行の数字列で表示（ベンチマークには非推奨）
    self.count+=1
    # 例："3115: 44424" のように出力
    print(self.count,end=": ")
    for i in range(self.size):
      print(self.aboard[i],end="")
    print("")

  def nqueens(self,row:int)->None:
    # 目的：各行に 0..size-1 を“無条件”に置いていく総当り
    # 注意：N-Queens の安全判定なし／膨大な出力
    # BUG FIX: `is` → `==`（値比較に修正）
    if row==self.size:
      self.printout()
      return
    for i in range(self.size):
      self.aboard[row]=i
      self.nqueens(row+1)

# ------------------------------------------------------------
# 2) 正しい（素朴）N-Queens：列＆斜めの衝突を避けるチェックを追加
# ------------------------------------------------------------
class NQueens01_ProperBF:
  size:int
  aboard:List[int]
  count:int

  def __init__(self,size:int)->None:
    # 目的：サイズと作業配列の初期化
    self.size=size
    self.aboard=[0 for _ in range(self.size)]  # row→col を格納
    self.count=0

  def is_safe(self,row:int,col:int)->bool:
    """目的：既に置いた 0..row-1 のクイーンと衝突しないかを判定。
    - 列衝突：同じ列に既にある
    - 斜め衝突：|row - r| == |col - c|
    """
    for r in range(row):
      c=self.aboard[r]
      if c==col:
        return False
      if abs(row-r)==abs(col-c):
        return False
    return True

  def printout(self)->None:
    # 目的：見つけた“正しい”解を表示
    self.count+=1
    print(self.count,end=": ")
    for i in range(self.size):
      print(self.aboard[i],end="")
    print("")

  def solve(self,row:int)->None:
    # 目的：行ごとのバックトラックで安全な列のみを再帰的に探索
    if row==self.size:
      self.printout()
      return
    for col in range(self.size):
      if self.is_safe(row,col):
        self.aboard[row]=col
        self.solve(row+1)
        # 戻しは不要（次の col で上書き）だが、明示するなら 0 を入れてもよい


# ------------------------------------------------------------
# 3) CLI 入口：引数で raw / proper を切り替え可能
# ------------------------------------------------------------

def main()->None:
  # 使い方：
  #   python3 this.py N [raw]
  #   raw を付けると MinimalFix（総当り）を実行。
  #   省略時は ProperBF（本来の N-Queens）を実行。
  n=5
  mode="proper"  # default

  if len(sys.argv)>=2:
    try:
      n=int(sys.argv[1])
    except ValueError:
      print("第1引数 N は整数で指定してください。例: 8")
      return
  if len(sys.argv)>=3:
    if sys.argv[2].lower() in ("raw","proper"):
      mode=sys.argv[2].lower()

  t0=time.perf_counter()

  #  MinimalFix（総当り）を実行。
  if mode=="raw":
    solver=NQueens01_MinimalFix(n)
    solver.nqueens(0)
    total=solver.count
  #  ProperBF（本来の N-Queens）を実行。
  else:
    solver=NQueens01_ProperBF(n)
    solver.solve(0)
    total=solver.count

  t1=time.perf_counter()

  print(f"\nMode: {mode}")
  print(f"N: {n}")
  print(f"Total: {total}")
  print(f"Elapsed: {t1 - t0:.3f}s")

if __name__=="__main__":
  main()
