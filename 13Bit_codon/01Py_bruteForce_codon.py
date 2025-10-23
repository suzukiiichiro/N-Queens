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
N-Queens 学習用ミニマム実装（総当り/正解版の二段構え）
====================================================
ファイル: nqueens_minimal_and_proper.py
作成日: 2025-10-23

概要:
  - クラス1: NQueens01_MinimalFix … 学習用の総当り（N^N）版。
    N-Queens の「解」にはなりませんが、再帰・全列挙の流れを掴むための最小例。
    ※ 重要なバグ（`is` 比較）を `==` に修正済み。
  - クラス2: NQueens01_ProperBF … 本来の N-Queens（列・斜めの衝突を回避）を
    素朴なバックトラックで列挙。

設計のポイント:
  - MinimalFix: 「停止条件 → 出力 → 戻る」の基本構造を学ぶ目的。
    典型パターン:
      if row == self.size:
          self.printout(); return
      for i in range(self.size):
          self.aboard[row] = i
          self.nqueens(row + 1)
  - ProperBF: 行ごとに安全な列だけを試すことで大幅に枝刈り。
    典型パターン:
      for col in range(self.size):
          if self.is_safe(row, col):
              self.aboard[row] = col
              self.solve(row + 1)

使い方:
  $ python3 this.py N [raw]
    - 第2引数が "raw" のとき MinimalFix（総当り）を実行
    - 省略時は ProperBF を実行

注意/計測上のヒント:
  - printout() は I/O 負荷が高く、ベンチマークには不向き。
    大きな N ではカウントのみ集計する実装に差し替えるとよい。
  - 次の発展: ビットボード化・対称性（左右ミラー/回転）・中央列特別処理など。

ライセンス: MIT（必要に応じて変更）
著者: suzuki/nqdev



レビュー／ポイント／特徴（ソース引用つき）

学習用の二段構え
総当りのミニマム版 NQueens01_MinimalFix は「N^N 通り」列挙（N-Queensの正解にはならない）。
正しいバックトラック版 NQueens01_ProperBF は is_safe() で列・斜めをチェックして解を列挙。
→ main() で raw / proper を切替（python this.py N [raw]）。

バグ修正点が明確
# BUG FIX: is → ==（値比較に修正） の通り、

if row==self.size:
    self.printout()
の値比較で停止条件を正しく判定（is だと偶発バグ）。

素朴なバックトラックの定石
solve() は

for col in range(self.size):
    if self.is_safe(row, col):
        self.aboard[row] = col
        self.solve(row+1)

の典型形。is_safe() は「同列」と「斜め（行差＝列差）」のみで十分。

ベンチ時の注意
printout() が毎解呼ばれるため、Nが大きくなると I/O で劇的に遅くなります（計測時はカウントのみを推奨）。
発展余地（次ステップ）
列挙をビットボード化すれば O(1) 判定、奇数Nの中央列特別処理や左右ミラーで探索半減など、次の高性能版にスムーズに接続できます。

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

class NQueens01_MinimalFix:
  """
  学習用の最小総当り版（N^N 列挙）。
  目的:
    - 再帰の構造（停止条件・ループ・再帰呼び出し）の理解を優先。
    - N-Queens の「解」判定は行わない（安全性チェックなし）。
  特徴（引用）:
    停止条件: `if row==self.size: self.printout(); return`
    列挙本体: `for i in range(self.size): self.aboard[row]=i; self.nqueens(row+1)`
  注意:
    - 出力が膨大になるため、大きな N の実行は非推奨。
    - `is` → `==` の修正により、停止条件が値比較で正しく働く。
  """
  size:int
  aboard:List[int]
  count:int

  def __init__(self,size:int)->None:
    """
    役割:
      盤サイズと作業配列 `aboard`（row→col の一時配置）を初期化し、カウントを0に設定。
    引数:
      size: 盤の大きさ N
    効能:
      - `self.aboard = [0 for _ in range(self.size)]` で固定長の列格納配列を確保。
    """
    # 目的：サイズを保持し、各行の暫定列を保持する配列 aboard を初期化
    self.size=size
    self.aboard=[0 for _ in range(self.size)]
    self.count=0

  def printout(self)->None:
    """
    役割:
      現在の `aboard` を 1 行の数字列として表示し、解カウント `count` をインクリメント。
    出力形式（引用）:
      `print(self.count, end=": "); for i in range(self.size): print(self.aboard[i], end="")`
    注意:
      - ベンチ/大きな N では I/O ボトルネックになるため計測には不向き。
    """
    # 目的：現在の aboard を 1 行の数字列で表示（ベンチマークには非推奨）
    self.count+=1
    # 例："3115: 44424" のように出力
    print(self.count,end=": ")
    for i in range(self.size):
      print(self.aboard[i],end="")
    print("")

  def nqueens(self,row:int)->None:
    """
    役割:
      行 `row` に 0..N-1 のいずれかの列を**無条件**で割り当て、全探索を行う。
    処理の流れ（引用）:
      - 停止条件: `if row==self.size: self.printout(); return`
      - 列挙: `for i in range(self.size): self.aboard[row]=i; self.nqueens(row+1)`
    注意:
      - 安全性チェックは一切しないため、N-Queens の意味での「解」ではない。
    計算量:
      - Θ(N^N)（出力自体も N^N 件）
    """
    # 目的：各行に 0..size-1 を“無条件”に置いていく総当り
    # 注意：N-Queens の安全判定なし／膨大な出力
    # BUG FIX: `is` → `==`（値比較に修正）
    if row==self.size:
      # NOTE: 実運用の計測では I/O を止めて count のみ更新にすると高速になります。
      self.printout()
      return
    for i in range(self.size):
      self.aboard[row]=i
      self.nqueens(row+1)

class NQueens01_ProperBF:
  """
  本来の N-Queens を素朴なバックトラックで解く実装。
  目的:
    - 行ごとに「安全な列」だけを再帰的に試し、全解数を列挙。
  特徴（引用）:
    - 配置: `self.aboard[row] = col`
    - 停止: `if row==self.size: self.printout(); return`
    - 判定: `abs(row-r) == abs(col-c)` で斜め衝突を検出
  注意:
    - I/O を伴う `printout()` は大きな N の計測ではオフ（集計のみ）推奨。
  """
  size:int
  aboard:List[int]
  count:int

  def __init__(self,size:int)->None:
    """
    役割:
      盤サイズ `size` と、行→列の配置を格納する `aboard` を初期化し、カウントを0に。
    引数:
      size: 盤の大きさ N
    実装（引用）:
      `self.aboard=[0 for _ in range(self.size)]`
    """
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
    """
    役割:
      見つけた「正しい解」を 1 行の数字列で表示し、`count` をインクリメント。
    出力形式（引用）:
      `print(self.count, end=": "); ... print(self.aboard[i], end="")`
    注意:
      - ベンチ用途では非表示にし、`count` のみ集計するのが望ましい。
    """
    # 目的：見つけた“正しい”解を表示
    self.count+=1
    print(self.count,end=": ")
    for i in range(self.size):
      print(self.aboard[i],end="")
    print("")

  def solve(self,row:int)->None:
    """
    役割:
      行 `row` において安全な列 `col` だけを選んで配置し、次行へ再帰。
    処理の流れ（引用）:
      - 停止条件: `if row==self.size: self.printout(); return`
      - 反復: `for col in range(self.size):`
      - 判定: `if self.is_safe(row, col): self.aboard[row]=col; self.solve(row+1)`
    実装メモ:
      - 戻し代入は不要（次の col で上書き）だが、可読性のため 0 を入れても良い。
    """
    # 目的：行ごとのバックトラックで安全な列のみを再帰的に探索
    if row==self.size:
      self.printout()
      return
    for col in range(self.size):
      # TODO: ビットボード化（cols/ld/rd の3ビット列）に置き換えると O(1) 判定になります。
      if self.is_safe(row,col):
        self.aboard[row]=col
        self.solve(row+1)
        # 戻しは不要（次の col で上書き）だが、明示するなら 0 を入れてもよい

def main()->None:
  """
  役割:
    コマンドライン引数を解釈し、MinimalFix（総当り）/ ProperBF（正解版）を実行。
  使い方（引用）:
    - `python3 this.py N [raw]`
    - `raw` を付けると MinimalFix を実行、既定は ProperBF。
  フロー:
    - 引数の検証: `int(sys.argv[1])` を試み、失敗時はメッセージを出して終了。
    - 実行: mode に応じて `nqueens(0)` または `solve(0)` を起動。
    - 計測: `time.perf_counter()` で所要時間を測定し、`Total` を表示。
  注意:
    - 出力件数が多い場合、端末 I/O が支配的になるため計測差が出やすい。
  """
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
