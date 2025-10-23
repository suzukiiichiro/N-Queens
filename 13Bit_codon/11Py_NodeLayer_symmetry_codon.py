#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Python/codon Ｎクイーン ノードレイヤー 対象解除版

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
N-Queens：frontier分割（ノードレイヤー）× 対称性分類（COUNT2/4/8）
==================================================================
ファイル: 11Py_node_layer_with_symmetry.py
作成日: 2025-10-23

概要:
  - 深さ k の frontier（部分状態: left/down/right）を収集し、各ノードから完全探索。
  - 同型判定（回転 90/180/270 + 垂直ミラー）で代表形のみを数え、2/4/8 倍で Total を復元。
  - Local 構造体に TOPBIT/ENDBIT/SIDEMASK/LASTMASK/BOUND1/BOUND2/board をスナップショット保存。
    → frontier 以降の探索は「完全独立」なので並列化に素直。

設計のポイント（実ソース引用）:
  - 可置集合: `bitmap = ((1<<size)-1) & ~(left | down | right)`
  - LSB抽出:  `bit = -bitmap & bitmap` / 消費: `bitmap ^= bit`
  - 葉判定:   `down == mask`
  - 対称判定: `symmetryOps()` → {0,2,4,8}
  - frontier:  `_popcount(down) == k` 到達で `(left,down,right)` と Local を push

並列化の勘所:
  - nodes は 3 要素（l,d,r）単位、Local はコピー済みの board を含むスナップショット。
  - 以後の完全探索は副作用分離 → 各ノードをプロセス/スレッド/Codon @par で分配可能。

検証の目安:
  - 例: N=13 → Unique=9233, Total=73712（count2/4/8 の和と倍数合成で一致）。

備考:
  - Python は任意長 int。固定幅（Codon等）では `mask=(1<<size)-1` で幅制約を維持。
  - I/O は計測を歪めるため、ベンチ時は印字最小化がおすすめ。


レビュー（短評）

良い点
frontier ごとに Local(board.copy()) を保存 → 後段の対称判定の 参照一貫性 が保てて安全。
角あり/角なしで 入口条件（TOPBIT/ENDBIT/BOUND/SIDEMASK/LASTMASK）を明確に切替。
「★ 修正: 角ありでも local.board[row]=bit 記録」の追記で、対称判定に必要な盤情報が欠落しないよう担保。

注意点 / 改善提案
k の自動調整
len(nodes)//3 が「コア数×数十」程度になるように k を自動化すると並列効率↑。

合算の可視化
count2/count4/count8 も最終合成して表示すると、検証が容易（現状は total のみ）。

型最適化（Codon）
int 幅や mask を compile-time const に寄せると LLVM 最適化の恩恵が大きい。

安全ガード
size < 1 早期 return、k > size のときは k = size に丸める等のバリデーションを追加しておくと堅牢。

検算の指針
小 n（5〜10）で 07/08 系の実装と Total 一致 を確認。
代表的値（N=8 → Total=92、N=13 → Total=73712）に一致すれば、frontier→対称の筋が通っています。


fedora$ codon build -release 11Py_NodeLayer_symmetry_codon.py && ./11Py_NodeLayer_symmetry_codon
 N:        Total        Unique        hh:mm:ss.ms
 4:            0            0         0:00:00.000
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.000
 9:          352            0         0:00:00.000
10:          724            0         0:00:00.000
11:         2680            0         0:00:00.000
12:        14200            0         0:00:00.002
13:        73712            0         0:00:00.015
14:       365596            0         0:00:00.080
15:      2279184            0         0:00:00.423
16:     14772512            0         0:00:02.761
fedora$

"""
from datetime import datetime
from typing import List

class Local:
  """
  frontier ノードに紐づく対称判定・境界制御のスナップショット。
  フィールド:
    TOPBIT  : 最上位ビット (1<<(N-1))
    ENDBIT  : 下辺側の終端ビット（角なし 180° 判定の入口）
    LASTMASK: 最終行の左右端制約
    SIDEMASK: 左右端の統合制約 (TOPBIT | 1)
    BOUND1  : 上辺側の境界行
    BOUND2  : 下辺側の境界行
    board   : row→bit の列配置（各行で 1 ビットだけ立つ）
  目的:
    - k 層までで確定した「文脈」を frontier ノードに持たせ、以後の探索を完全独立化。
  """

  TOPBIT:int
  ENDBIT:int
  LASTMASK:int
  SIDEMASK:int
  BOUND1:int
  BOUND2:int
  board:List[int]

  def __init__(self,TOPBIT:int,ENDBIT:int,LASTMASK:int,SIDEMASK:int,BOUND1:int,BOUND2:int,board:List[int])->None:
    self.TOPBIT=TOPBIT
    self.ENDBIT=ENDBIT
    self.LASTMASK=LASTMASK
    self.SIDEMASK=SIDEMASK
    self.BOUND1=BOUND1
    self.BOUND2=BOUND2
    self.board=board

class NQueens11:
  """
  ノードレイヤー（frontier分割）× 対称性分類（COUNT2/4/8）により
  Unique/Total を（ノード単位で）正確に積み上げ可能な探索器。

  モジュール構成:
    - count_bits_nodeLayer     : popcount（Brian Kernighan 法）
    - symmetryOps              : 同型（90/180/270 + 垂直）判定 → {0,2,4,8}
    - symmetry_solve_*         : 角あり/角なしの完全探索（代表のみをカウント）
    - kLayer_nodeLayer_*       : 深さ k の frontier 収集（Local スナップショット付き）
    - symmetry_build_nodeLayer : frontier から対称つきで合算（並列化ポイント）
  """

  def __init__(self)->None:
    pass

  def count_bits_nodeLayer(self,n:int)->int:
    """
    役割:
      set bit 数（1 の個数）を返す。frontier 到達判定に使用。
    アルゴリズム（引用）:
      `while n: n &= n - 1; cnt += 1`
    計算量:
      O(#set bits)
    """

    cnt=0
    while n:
      n&=n-1
      cnt+=1
    return cnt

  def symmetryOps(self,size:int,local:Local)->int:
    """
    役割:
      board（row→bit）に対し、回転(90/180/270) + 垂直ミラー同型を比較し、
      代表形なら 2/4/8 を返す。代表でない場合は 0 を返す。

    ロジック（引用の要点）:
      - 90°入口:  `if local.board[local.BOUND2] == 1:`
      - 180°入口: `if local.board[size-1] == local.ENDBIT:`
      - 270°入口: `if local.board[local.BOUND1] == local.TOPBIT:`
      比較ループでは `bit<<=1`/`ptn<<=1 or >>1` で辞書順最小性を判定。
      途中で `local.board[own] > bit` になったら代表性を否定（0）。

    戻り値:
      0（非代表）/ 2 / 4 / 8
    注意:
      - board は「各行 1ビットだけ立つ」ことが前提（探索側で保証）。
    """

    # 90°
    if local.board[local.BOUND2]==1:
      ptn:int=2
      own:int=1
      while own<size:
        bit:int=1
        you:int=size-1
        while you>=0 and local.board[you]!=ptn and local.board[own]>=bit:
          bit<<=1
          you-=1
        if local.board[own]>bit:
          return 0
        if local.board[own]<bit:
          break
        ptn<<=1
        own+=1
      if own>size-1:
        return 2
    # 180°
    if local.board[size-1]==local.ENDBIT:
      you:int=size-2
      own:int=1
      while own<=size-1:
        bit:int=1
        ptn:int=local.TOPBIT
        while ptn!=local.board[you] and local.board[own]>=bit:
          ptn>>=1
          bit<<=1
        if local.board[own]>bit:
          return 0
        if local.board[own]<bit:
          break
        you-=1
        own+=1
      if own>size-1:
        return 4
    # 270°
    if local.board[local.BOUND1]==local.TOPBIT:
      ptn:int=local.TOPBIT>>1
      own:int=1
      while own<=size-1:
        bit:int=1
        you:int=0
        while you<size and local.board[you]!=ptn and local.board[own]>=bit:
          bit<<=1
          you+=1
        if local.board[own]>bit:
          return 0
        if local.board[own]<bit:
          break
        ptn>>=1
        own+=1
    return 8

  def symmetry_solve_nodeLayer_corner(self,size:int,left:int,down:int,right:int,local:Local)->int:
    """
    役割:
      角に Q が「ある」系の完全探索。代表性チェックを省略し、葉で 8 を返す（8倍クラス）。

    コア（引用）:
      - 行 row: `row = self.count_bits_nodeLayer(down)`
      - 可置:   `bitmap = ((1<<size)-1) & ~(left | down | right)`
      - 上辺制約（row < BOUND1）:
          `bitmap |= 2; bitmap ^= 2`  # (= bitmap & ~2) 分岐なしテク
      - 末行: `if row == size-1 and bitmap: return 8`

    戻り値:
      この部分木の total（8 の倍数）
    """

    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    row:int=self.count_bits_nodeLayer(down)
    if row==(size-1):
      if bitmap:
        return 8
      return 0
    if row<local.BOUND1:
      bitmap|=2
      bitmap^=2
    total=0
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      total+=self.symmetry_solve_nodeLayer_corner(size,(left|bit)<<1,(down|bit),(right|bit)>>1,local)
    return total

  def symmetry_solve_nodeLayer(self,size:int,left:int,down:int,right:int,local:Local)->int:
    """
    役割:
      角に Q が「ない」系の完全探索。末端で代表性チェック（symmetryOps）を行う。

    コア（引用）:
      - 行 row: `row = self.count_bits_nodeLayer(down)`
      - 末行:
          `if bitmap and (bitmap & local.LASTMASK) == 0: local.board[row] = bitmap; return symmetryOps(...)`
      - 上辺/下辺制約:
          `bitmap = (bitmap | SIDEMASK) ^ SIDEMASK`  # (= bitmap & ~SIDEMASK)
          `if row == BOUND2: ... bitmap &= SIDEMASK`
      - LSB抽出: `bit = -bitmap & bitmap; bitmap ^= bit; local.board[row] = bit; ...`

    戻り値:
      この部分木の {0,2,4,8} の合計
    """

    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    row:int=self.count_bits_nodeLayer(down)
    if row==(size-1):
      if bitmap:
        if (bitmap&local.LASTMASK)==0:
          local.board[row]=bitmap
          return self.symmetryOps(size,local)
      return 0
    if row<local.BOUND1:
      bitmap|=local.SIDEMASK
      bitmap^=local.SIDEMASK
    elif row==local.BOUND2:
      if (down&local.SIDEMASK)==0:
        return 0
      if (down&local.SIDEMASK)!=local.SIDEMASK:
        bitmap&=local.SIDEMASK
    total=0
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      local.board[row]=bit
      total+=self.symmetry_solve_nodeLayer(size,(left|bit)<<1,(down|bit),(right|bit)>>1,local)
    return total

  def symmetry_solve(self,size:int,left:int,down:int,right:int,local:Local)->int:
    """
    役割:
      Local.board[0] が 1（= 角あり）かどうかで探索器を振り分け。
    戻り値:
      total（2/4/8 倍の合算）
    """

    if local.board[0]==1:
      return self.symmetry_solve_nodeLayer_corner(size,left,down,right,local)
    else:
      return self.symmetry_solve_nodeLayer(size,left,down,right,local)

  def kLayer_nodeLayer_backtrack_corner(self,size:int,nodes:List[int],k:int,left:int,down:int,right:int,local:Local,local_list:List[Local])->None:
    """
    役割:
      角あり系で深さ k の frontier を収集。Local のスナップショットも併記保存。

    コア（引用）:
      - 到達条件: `if self.count_bits_nodeLayer(down) == k:`
          `nodes.extend((left, down, right))`
          `local_list.append(Local(..., board=local.board.copy()))`
      - 上辺制約: `bitmap |= 2; bitmap ^= 2`
      - LSB抽出:  `bit = -bitmap & bitmap; bitmap ^= bit;`
      - ★修正: `local.board[row] = bit` を記録してから再帰（対称判定のために必要）

    備考:
      - board.copy() により frontier 間の独立性を保証。
    """

    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    row:int=self.count_bits_nodeLayer(down)
    if row==k:
      nodes.extend((left,down,right))
      local_list.append(Local(TOPBIT=local.TOPBIT,ENDBIT=local.ENDBIT,LASTMASK=local.LASTMASK,SIDEMASK=local.SIDEMASK,BOUND1=local.BOUND1,BOUND2=local.BOUND2,board=local.board.copy()))
      return
    if row<local.BOUND1:
      bitmap|=2
      bitmap^=2
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      # ★ 修正: 角あり側でも board に現在のビットを記録
      local.board[row]=bit
      self.kLayer_nodeLayer_backtrack_corner(size,nodes,k,(left|bit)<<1,(down|bit),(right|bit)>>1,local,local_list)

  def kLayer_nodeLayer_backtrack(self,size:int,nodes:List[int],k:int,left:int,down:int,right:int,local:Local,local_list:List[Local])->None:
    """
    役割:
      角なし系で深さ k の frontier を収集。Local スナップショットも保存。

    コア（引用）:
      - row == k 到達で (l,d,r) を push、Local(board.copy()) を local_list に push
      - 上辺制約: `bitmap = (bitmap | SIDEMASK) ^ SIDEMASK`
      - 下辺制約: `if row == BOUND2: ... bitmap &= SIDEMASK`
      - LSB抽出:  `bit = -bitmap & bitmap; bitmap ^= bit; local.board[row] = bit; ...`
    """

    mask:int=(1<<size)-1
    bitmap:int=mask&~(left|down|right)
    row:int=self.count_bits_nodeLayer(down)
    if row==k:
      nodes.extend((left,down,right))
      local_list.append(Local(TOPBIT=local.TOPBIT,ENDBIT=local.ENDBIT,LASTMASK=local.LASTMASK,SIDEMASK=local.SIDEMASK,BOUND1=local.BOUND1,BOUND2=local.BOUND2,board=local.board.copy()))
      return
    if row<local.BOUND1:
      bitmap|=local.SIDEMASK
      bitmap^=local.SIDEMASK
    elif row==local.BOUND2:
      if (down&local.SIDEMASK)==0:
        return
      if (down&local.SIDEMASK)!=local.SIDEMASK:
        bitmap&=local.SIDEMASK
    while bitmap:
      bit=-bitmap&bitmap
      bitmap^=bit
      local.board[row]=bit
      self.kLayer_nodeLayer_backtrack(size,nodes,k,(left|bit)<<1,(down|bit),(right|bit)>>1,local,local_list)

  def kLayer_nodeLayer(self,size:int,nodes:List[int],k:int,local_list:List[Local])->None:
    """
    役割:
      角あり → 角なしの順に、frontier（深さ k）を網羅的に構築し、(l,d,r) と Local を蓄積。

    手順（引用）:
      - 初期 Local（角あり）:
         `Local(TOPBIT=1<<(N-1), ENDBIT=0, LASTMASK=0, SIDEMASK=0, BOUND1=2, BOUND2=0, board=[0]*N)`
         `board[0] = 1`
         while `BOUND1 < N-1`: `board[1] = 1<<BOUND1; kLayer_nodeLayer_backtrack_corner(...)`
      - 角なしに切替:
         `TOPBIT=1<<(N-1); ENDBIT=TOPBIT>>1; SIDEMASK=TOPBIT|1; LASTMASK=SIDEMASK; BOUND1=1; BOUND2=N-2`
         while `BOUND1 < BOUND2`: `board[0] = 1<<BOUND1; kLayer_nodeLayer_backtrack(...)`
         ループ毎の更新:
           `BOUND1+=1; BOUND2-=1; ENDBIT>>=1; LASTMASK=(LASTMASK<<1)|LASTMASK|(LASTMASK>>1)`
    """

    local=Local(TOPBIT=1<<(size-1),ENDBIT=0,LASTMASK=0,SIDEMASK=0,BOUND1=2,BOUND2=0,board=[0]*size)
    local.board[0]=1
    # 角あり
    while local.BOUND1>1 and local.BOUND1<size-1:
      if local.BOUND1<size-1:
        bit=1<<local.BOUND1
        local.board[1]=bit
        self.kLayer_nodeLayer_backtrack_corner(size,nodes,k,(2|bit)<<1,(1|bit),(2|bit)>>1,local,local_list)
      local.BOUND1+=1
    # 角なし
    local.TOPBIT=1<<(size-1)
    local.ENDBIT=local.TOPBIT>>1
    local.SIDEMASK=local.TOPBIT|1
    local.LASTMASK=local.TOPBIT|1
    local.BOUND1=1
    local.BOUND2=size-2
    while local.BOUND1>0 and local.BOUND2<size-1 and local.BOUND1<local.BOUND2:
      if local.BOUND1<local.BOUND2:
        bit=1<<local.BOUND1
        local.board[0]=bit
        self.kLayer_nodeLayer_backtrack(size,nodes,k,bit<<1,bit,bit>>1,local,local_list)
      local.BOUND1+=1
      local.BOUND2-=1
      local.ENDBIT>>=1
      local.LASTMASK=(local.LASTMASK<<1)|local.LASTMASK|(local.LASTMASK>>1)

  def symmetry_build_nodeLayer(self,size:int)->int:
    """
    役割:
      深さ k（本実装は k=4）の frontier を作成し、各ノードを対称つきで完全探索して合算。
    手順（引用）:
      - `nodes: List[int] = []`, `local_list: List[Local] = []`, `k = 4`
      - `self.kLayer_nodeLayer(size, nodes, k, local_list)`
      - 各ノードについて `self.symmetry_solve(size, l, d, r, local)` を合算
    並列化:
      - for ループ（ノード列挙）をワーカーに分配可能（プロセス/スレッド/＠par）。
    戻り値:
      Total（全解数）
    """

    nodes:List[int]=[]
    local_list:List[Local]=[]
    k:int=4
    self.kLayer_nodeLayer(size,nodes,k,local_list)
    num_nodes=len(nodes)//3
    total=0
    for i in range(num_nodes):
      total+=self.symmetry_solve(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2],local_list[i])
    return total

class NQueens11_NodeLayer:
  def main(self)->None:
    """
    役割:
      N=4..17 を実行し、Total/Unique(0)/経過時間を表形式で出力（range の上限は含まれない点に注意）。
    出力（引用）:
      `print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")`
    メモ:
      - nmax を含めたい場合は `range(nmin, nmax+1)`。
      - ベンチ時は出力を抑えると計測が安定。
    """

    nmin:int=4
    nmax:int=18
    print(" N:        Total        Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      nq=NQueens11()
      total=nq.symmetry_build_nodeLayer(size)
      dt=datetime.now()-start_time
      text=str(dt)[:-3]
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")

if __name__=="__main__":
  NQueens11_NodeLayer().main()
