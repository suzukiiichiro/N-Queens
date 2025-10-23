#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Python/codon Ｎクイーン 対象解除版

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
N-Queens：順列生成 + 対角フラグ + 対称性（COUNT2/4/8）で Unique/Total 算出
=====================================================================
ファイル: 04Py_perm_diag_symmetry.py
作成日: 2025-10-23

概要:
  - 行→列の順列を生成（列ユニークを配列スワップで保証）
  - 対角衝突は配列フラグ fb(ld)/fc(rd) で O(1) 判定
  - 代表解のみを数えるため symmetryops() で回転(90/180/270)・垂直反転を比較
    → 代表解なら倍率 2/4/8 を返し、Total = sum(倍率)、Unique = 代表解数

設計のポイント（実ソース引用）:
  - 対角添字:
      `ld = row - aboard[row] + _off`  # 0..(2N-2), _off = N-1
      `rd = row + aboard[row]`         # 0..(2N-2)
  - 衝突判定:
      `if (fb[ld] | fc[rd]) == 0:`
  - 1行目の半分制限（左右対称除去の定石）:
      `lim = size if row != 0 else (size + 1) // 2`
  - 順列生成（スワップ + 末尾回しの定型）:
      `aboard[i], aboard[row] = aboard[row], aboard[i]`
      最後に `aboard[row]` を末尾へ回す回転処理で次ループへ

検証の目安:
  - N=8 → Unique=92, Total=92（この実装は COUNT2/4/8 で一致するはず）

注意:
  - I/O は遅いので測定時は出力を抑止するか、代表解のみの記録に留める。
  - rotate()/vmirror() は row->col 形式（長さ N の配列）を保つこと。

著者: suzuki/nqdev
ライセンス: MIT（必要に応じて変更）


レビューと注意点（短評）

良い点
列ユニークを「順列スワップ」で保証し、衝突判定を対角2種だけに縮退→分岐が明快。
symmetryops() が辞書順最小性で代表解を判定し、2/4/8 の倍率に落とす正攻法。
1行目の半分制限と COUNT2/4/8 の設計が噛み合っている。

要注意
rotate() は row->col 形式を維持するための2段処理（コピー→再配置）。ここを崩すと symmetryops() が破綻します。
最終行（row == size-1）の直前に対角判定してから symmetryops() を呼ぶ順序は厳守。
aboard の末尾回し（回転）で順列の総当たりを漏れなく巡回すること（現在の実装は定型どおり OK）。

次の一手（必要なら別テンプレ提供できます）
ビットボード化（cols/ld/rd を int に統合、free=mask&~(ld|rd|col), bit=free&-free）
中心列の特別処理（奇数 N）：COUNT の整合とさらに効く枝刈り
Codon/PyPy 最適化：整数幅とマスク明示、再帰のアンローリングパラメータなど


Codon/Python 共通で動作。

fedora$ codon build -release 04Py_symmetry_codon.py && ./04Py_symmetry_codon
 N:        Total       Unique        hh:mm:ss.ms
 4:            2            1         0:00:00.000
 5:           10            2         0:00:00.000
 6:            4            1         0:00:00.000
 7:           40            6         0:00:00.000
 8:           92           12         0:00:00.000
 9:          352           46         0:00:00.000
10:          724           92         0:00:00.000
11:         2680          341         0:00:00.005
12:        14200         1787         0:00:00.055
13:        73712         9233         0:00:00.137
14:       365596        45752         0:00:00.769
15:      2279184       285053         0:00:04.810
"""

from datetime import datetime
from typing import List

# pypy を使う場合はコメントを解除（Codon では使わないこと）
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')


class NQueens04:
  """
  順列生成 + 対角フラグ（ld/rd）で探索し、symmetryops() で対称性分類して
  Unique/Total を同時計数する実装。

  構造:
    - aboard  : row->col の順列。初期は [0,1,...,N-1]（列ユニークをスワップで維持）
    - fb/fc   : 対角フラグ（長さ 2N-1）
    - trial   : 対称変換時の作業配列
    - scratch : 回転中間バッファ
    - _off    : ld の負値補正 = N-1（ld=row-col+_off を 0..2N-2 に写像）

  特徴（引用）:
    - 1行目は左右対称を避けるため半分のみ:
        `lim = size if row != 0 else (size + 1) // 2`
    - 最終行で symmetryops(size) → {0,2,4,8} を受け取り、
      0 以外なら Unique++, Total += 倍率。
  """

  # 結果カウンタ
  total:int
  unique:int
  # 作業配列
  aboard:List[int]   # row -> col（順列ベース：0..size-1 の並べ替え）
  fb:List[int]       # ld（左下↙︎/右上↗︎）対角フラグ: 0..(2*size-2)
  fc:List[int]       # rd（右下↘︎/左上↖︎）対角フラグ: 0..(2*size-2)
  trial:List[int]    # 対称操作時の作業バッファ
  scratch:List[int]  # 回転の中間バッファ
  _off:int           # (row-col) の負値補正 = size-1
  _size:int          # 参照用に保持（任意）

  def __init__(self)->None:
    """
    役割:
      インスタンス生成のみ（実配列は init(size) で確保）。
    注意:
      - init(size) を呼ぶ前に nqueens() を実行しないこと。
    """

    # 実体は init(size) で都度作る
    pass

  def init(self,size:int)->None:
    """
    役割:
      盤サイズに応じて作業領域を全て初期化する。
    実装（引用）:
      - `aboard = [i for i in range(size)]`  # 順列ベース
      - `fb = [0 for _ in range(2*size - 1)]`  # ld, 0..2N-2
      - `fc = [0 for _ in range(2*size - 1)]`  # rd, 0..2N-2
      - `trial/scratch = [0]*size`
      - `_off = size - 1`
    メモ:
      列ユニークは順列操作で保証するため、列フラグは不要。
    """

    self.total=0
    self.unique=0
    # 順列ベース：初期は [0,1,2,...,size-1]
    self.aboard=[i for i in range(size)]
    # 対角フラグ（列ユニークは順列で保証されるため不要）
    self.fb=[0 for _ in range(2*size-1)]
    self.fc=[0 for _ in range(2*size-1)]
    # 対称操作用バッファ
    self.trial=[0 for _ in range(size)]
    self.scratch=[0 for _ in range(size)]
    # (row - col) の負値補正用
    self._off=size-1
    self._size=size

  def rotate(self,chk:List[int],scr:List[int],n:int,neg:int)->None:
    """
    役割:
      row->col 配列 `chk` を 90° 回転した配置に書き換える。
      neg=1: 右回り（時計回り）、neg=0: 左回り（反時計）。
    手順（引用）:
      # 第1段: 順/逆で `chk` → `scr`
      `incr = 1 if neg else -1`
      `k = 0 if neg else n-1`
      `scr[i] = chk[k]; k += incr`
      # 第2段: scr の「値」を添字として新しい列を復元
      `k = n-1 if neg else 0`
      `chk[scr[i]] = k; k -= incr`
    注意:
      - 入出力とも row->col 形式（長さ N の 0..N-1 値）を維持する。
    """

    # 第1段：scr に chk を順方向/逆方向でコピー
    incr=1 if neg else-1
    k=0 if neg else n-1
    for i in range(n):
      scr[i]=chk[k]
      k+=incr
    # 第2段：scr の値（= 列）を添字として使い、chk に新しい列を埋め戻す
    k=n-1 if neg else 0
    for i in range(n):
      chk[scr[i]]=k
      k-=incr

  def vmirror(self,chk:List[int],n:int)->None:
    """
    役割:
      垂直反転（左右反転）。col 値を `col -> (n-1)-col` に変換。
    実装（引用）:
      `chk[i] = (n - 1) - chk[i]`
    """
    for i in range(n):
      chk[i]=(n-1)-chk[i]

  def intncmp(self,lt:List[int],rt:List[int],n:int)->int:
    """
    役割:
      配列 lt と rt の辞書順比較。lt < rt: 負、==: 0、>: 正 を返す。
    実装（引用）:
      `d = lt[i] - rt[i]; d != 0 なら d を返す。最後まで同じなら 0。`
    """
    for i in range(n):
      d=lt[i]-rt[i]
      if d!=0:
        return d
    return 0

  def symmetryops(self,size:int)->int:
    """
    役割:
      現在の `aboard` が対称群（90/180/270 の回転 + 垂直反転とその回転）における
      最小表現（辞書順最小）かどうかを判定し、等価解の倍率を返す。
    戻り値:
      0（代表でない）/ 2 / 4 / 8
    判定手順（引用の要点）:
      - `trial <- aboard`
      - 90°: `rotate(trial, scratch, size, 0)` → `intncmp(aboard, trial)`
        * k > 0 なら代表ではない → 0
        * k == 0 なら `nequiv = 1`
        * それ以外なら 180°, 270° を続けて比較（同値なら nequiv=2/4）
      - 垂直反転: `trial <- aboard` → `vmirror(trial, size)` → 比較
        * k > 0 なら 0
        * `nequiv > 1` のとき、反転後の 90/180/270 も比較
      - 返値: `nequiv * 2`（1→2倍, 2→4倍, 4→8倍）
    注意:
      - aboard 自体を書き換えずに `trial/scratch` だけを操作する。
      - 比較は常に `aboard` を「基準」に行い、辞書順最小性を確認。
    """

    nequiv=0
    # trial に原盤をコピー
    for i in range(size):
      self.trial[i]=self.aboard[i]
    # 90°
    self.rotate(self.trial,self.scratch,size,0)
    k=self.intncmp(self.aboard,self.trial,size)
    if k>0:
      return 0
    if k==0:
      nequiv=1
    else:
      # 180°
      self.rotate(self.trial,self.scratch,size,0)
      k=self.intncmp(self.aboard,self.trial,size)
      if k>0:
        return 0
      if k==0:
        nequiv=2
      else:
        # 270°
        self.rotate(self.trial,self.scratch,size,0)
        k=self.intncmp(self.aboard,self.trial,size)
        if k>0:
          return 0
        nequiv=4
    # 垂直反転
    for i in range(size):
      self.trial[i]=self.aboard[i]
    self.vmirror(self.trial,size)
    k=self.intncmp(self.aboard,self.trial,size)
    if k>0:
      return 0
    # 垂直反転後の回転
    if nequiv>1:
      # 90
      self.rotate(self.trial,self.scratch,size,1)
      k=self.intncmp(self.aboard,self.trial,size)
      if k>0:
        return 0
      if nequiv>2:
        # 180
        self.rotate(self.trial,self.scratch,size,1)
        k=self.intncmp(self.aboard,self.trial,size)
        if k>0:
          return 0
        # 270
        self.rotate(self.trial,self.scratch,size,1)
        k=self.intncmp(self.aboard,self.trial,size)
        if k>0:
          return 0
    return nequiv*2  # 1→2倍, 2→4倍, 4→8倍

  def nqueens(self,row:int,size:int)->None:
    """
    役割:
      行 `row` の列を順列スワップで決め、対角フラグで枝刈りしつつ再帰。
      最終行に到達したら symmetryops(size) で代表性を確認し、Unique/Total を更新。
    コアロジック（引用）:
      - 最終行:
          `if fb[row - aboard[row] + _off] or fc[row + aboard[row]]: return`
          `stotal = symmetryops(size)`
          `if stotal != 0: unique += 1; total += stotal`
      - 1行目の半分制限:
          `lim = size if row != 0 else (size + 1) // 2`
      - 順列生成（スワップ）と対角チェック:
          `aboard[i], aboard[row] = aboard[row], aboard[i]`
          `ld = row - aboard[row] + _off; rd = row + aboard[row]`
          `if (fb[ld] | fc[rd]) == 0: fb[ld]=fc[rd]=1; nqueens(row+1); fb[ld]=fc[rd]=0`
      - 末尾回し（次の外側ループへ）:
          `tmp = aboard[row]; aboard[row: size-1] = aboard[row+1: size]; aboard[size-1] = tmp`
    注意:
      - スワップと「戻し」の対応は対角フラグ側で行い、aboard は最後に回転で整合。
      - 1行目半分制限は左右対称分を排除するため（COUNT2/4/8 と整合）。
    """

    if row==size-1:
      # 最終行の候補が現在の aboard[row]。対角衝突だけ確認（列は順列で一意）
      if self.fb[row-self.aboard[row]+self._off] or self.fc[row+self.aboard[row]]:
        return
      stotal=self.symmetryops(size)
      if stotal!=0:
        self.unique+=1
        self.total+=stotal
      return

    # 1 行目は左右対称を避けるため半分だけ試す
    lim=size if row!=0 else (size+1)//2
    for i in range(row,lim):
      # row と i をスワップして次の行へ（順列生成）
      tmp=self.aboard[i]
      self.aboard[i]=self.aboard[row]
      self.aboard[row]=tmp
      # 対角フラグ（列は不要）
      ld=row-self.aboard[row]+self._off
      rd=row+self.aboard[row]
      if (self.fb[ld]|self.fc[rd])==0:
        self.fb[ld]=1
        self.fc[rd]=1
        self.nqueens(row+1,size)
        self.fb[ld]=0
        self.fc[rd]=0
    # row を末尾に回して、次の外側ループへ（順列生成の定型テク）
    tmp=self.aboard[row]
    for i in range(row+1,size):
      self.aboard[i-1]=self.aboard[i]
    self.aboard[size-1]=tmp

  def main(self)->None:
    """
    役割:
      N=4..18 を走査し、Total/Unique/経過時間を表形式で出力する。
    実装（引用）:
      `print(" N:        Total       Unique        hh:mm:ss.ms")`
      `print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")`
    注意:
      - 出力件数が多い場合は端末 I/O が支配的になる。
      - 実験比較のときは印字を減らし、計測を安定化させる。
    """

    minN:int=4
    maxN:int=18
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(minN,maxN+1):
      self.init(size)
      start_time=datetime.now()
      self.nqueens(0,size)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")

if __name__=='__main__':
    NQueens04().main()
