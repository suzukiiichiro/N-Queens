#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Python/codon Ｎクイーン コンステレーション版 CUDA 高速ソルバ

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


実行履歴
2026年  1月 22日 木曜日 10:26:29 UTC
Python/Codon amazon AWS m4.16xlarge x 1
suzuki@cudacodon$ codon build -release 19Py_constellations_cuda_codon_k.20260122.py
suzuki@cudacodon$ ./19Py_constellations_cuda_codon_k.20260122 -c
CPU mode selected
 N:             Total         Unique        hh:mm:ss.ms
 5:                10              0         0:00:00.000
 6:                 4              0         0:00:00.020    ok
 7:                40              0         0:00:00.007    ok
 8:                92              0         0:00:00.025    ok
 9:               352              0         0:00:00.005    ok
10:               724              0         0:00:00.002    ok
11:              2680              0         0:00:00.003    ok
12:             14200              0         0:00:00.010    ok
13:             73712              0         0:00:00.011    ok
14:            365596              0         0:00:00.018    ok
15:           2279184              0         0:00:00.027    ok
16:          14772512              0         0:00:00.055    ok
17:          95815104              0         0:00:00.326    ok
18:         666090624              0         0:00:02.122    ok
19:        4968057848              0         0:00:15.068    ok
20:       39029188884              0         0:01:58.882    ok

2026年  1月 22日 木曜日 10:26:29 UTC
Python/Codon amazon AWS m4.16xlarge x 1
suzuki@cudacodon$ ./19Py_constellations_cuda_codon_k.20260122 -g
GPU mode selected
 N:             Total         Unique        hh:mm:ss.ms
 5:                10              0         0:00:00.000
 6:                 4              0         0:00:00.052    ok
 7:                40              0         0:00:00.001    ok
 8:                92              0         0:00:00.001    ok
 9:               352              0         0:00:00.001    ok
10:               724              0         0:00:00.001    ok
11:              2680              0         0:00:00.001    ok
12:             14200              0         0:00:00.071    ok
13:             73712              0         0:00:00.005    ok
14:            365596              0         0:00:00.018    ok
15:           2279184              0         0:00:00.078    ok
16:          14772512              0         0:00:00.330    ok
17:          95815104              0         0:00:01.885    ok
18:         666090624              0         0:00:28.268    ok
19:        4968057848              0         0:03:46.083    ok
20:       39027754460              0         0:30:32.772    ng(39027754460!=3902
9188884)

2025 15Py_constellations_optimize_codon.py
Python/Codon amazon AWS m4.16xlarge x 1
$ codon build -release 15Py_constellations_optimize_codon.py 
$ ./15Py_constellations_optimize_codon 
 N:             Total         Unique         hh:mm:ss.ms
18:         666090624              0          0:00:02.961
19:        4968057848              0          0:00:22.049
20:       39029188884              0          0:02:52.430
21:      314666222712              0          0:24:25.554
22:     2691008701644              0          3:29:33.971
23:    24233937684440              0   1 day, 8:12:58.977

2023/11/22 現在の最高速実装（CUDA GPU 使用、Codon コンパイラ最適化版）
C/CUDA NVIDIA(GPU)
$ nvcc -O3 -arch=sm_61 -m64 -ptx -prec-div=false 04CUDA_Symmetry_BitBoard.cu && POCL_DEBUG=all ./a.out -n ;
対称解除法 GPUビットボード
18:         666090624        83263591    000:00:00:01.65
19:        4968057848       621012754    000:00:00:13.80
20:       39029188884      4878666808    000:00:02:02.52
21:      314666222712     39333324973    000:00:18:46.52
22:     2691008701644    336376244042    000:03:00:22.54
23:    24233937684440   3029242658210    001:06:03:49.29
24:   227514171973736  28439272956934    012:23:38:21.02
25:  2207893435808352 275986683743434    140:07:39:29.96
"""
# 18Py_constellations_cuda_codon.py
import gpu
import sys
from typing import List,Set,Dict,Tuple
from datetime import datetime

""" サブコンステレーション生成状態の最大深さ定義 """
MAXF:Static[int] = 64  # 28より大きければOK（固定で）
MAXD:Static[int] = 32
FCONST:Static[int]=28
""" 64bit マスク（Zobrist用途） """
# MASK64:int=(1<<64)-1
# MASK64:int=u64(0)-u64(1)
""" サブコンステレーション生成状態のキー型定義 """
StateKey=Tuple[int,int,int,int,int,int,int,int,int,int,int]

""" CUDA GPU 用 DFS カーネル関数  """
# 2026年 1月 23日 
# 17:          95815104              0         0:00:01.875    ok
@gpu.kernel
def kernel_dfs_iter_gpu(
    ld_arr, rd_arr, col_arr, row_arr, free_arr,
    jmark_arr, end_arr, mark1_arr, mark2_arr,
    funcid_arr, w_arr,
    meta_next,meta_avail,
    blockK_tbl, blockL_tbl,
    is_base, is_jmark, is_mark,
    mark_sel, mark_step, mark_add1,
    results,
    m: int, board_mask: int,
    # F:int
):
    # # ---- ローカル固定長スタック（型は環境に合わせて固定長配列に）----
    # fid    = [0] * MAXD
    # ld     = [0] * MAXD
    # rd     = [0] * MAXD
    # col    = [0] * MAXD
    # row    = [0] * MAXD
    # avail  = [0] * MAXD
    # inited = [0] * MAXD

    # step   = [0] * MAXD
    # add1   = [0] * MAXD
    # rowst  = [0] * MAXD
    # bK     = [0] * MAXD
    # bL     = [0] * MAXD
    # nextf  = [0] * MAXD
    # ub     = [0] * MAXD
    # fc     = [0] * MAXD
    # 
    # ↑ ★ list をやめて静的配列へ
    fid    = __array__[int](MAXD)
    ld     = __array__[int](MAXD)
    rd     = __array__[int](MAXD)
    col    = __array__[int](MAXD)
    row    = __array__[int](MAXD)
    avail  = __array__[int](MAXD)
    step   = __array__[int](MAXD)
    add1   = __array__[int](MAXD)
    rowst  = __array__[int](MAXD)
    bK     = __array__[int](MAXD)
    bL     = __array__[int](MAXD)
    nextf  = __array__[int](MAXD)
    inited = __array__[u8](MAXD)
    ub     = __array__[u8](MAXD)
    fc     = __array__[u8](MAXD)
    for k in range(MAXD):
      inited[k] = u8(0)
      ub[k] = u8(0)
      fc[k] = u8(0)

    # # シェアードメモリ対応
    # # funcidテーブルをローカルへ（スレッドごとに1回だけglobal→localコピー）
    # m_next  = __array__[int](MAXF)
    # m_avail = __array__[int](MAXF)
    # t_bK    = __array__[int](MAXF)
    # t_bL    = __array__[int](MAXF)
    # t_isb   = __array__[int](MAXF)
    # t_isj   = __array__[int](MAXF)
    # t_ism   = __array__[int](MAXF)
    # t_sel   = __array__[int](MAXF)
    # t_step  = __array__[int](MAXF)
    # t_add1  = __array__[int](MAXF)

    # # F個だけ埋める
    # # for f0 in range(F):
    # for f0 in range(FCONST):
    #     m_next[f0]  = meta_next[f0]
    #     m_avail[f0] = meta_avail[f0]
    #     t_bK[f0]    = blockK_tbl[f0]
    #     t_bL[f0]    = blockL_tbl[f0]
    #     t_isb[f0]   = is_base[f0]
    #     t_isj[f0]   = is_jmark[f0]
    #     t_ism[f0]   = is_mark[f0]
    #     t_sel[f0]   = mark_sel[f0]
    #     t_step[f0]  = mark_step[f0]
    #     t_add1[f0]  = mark_add1[f0]


    i = (gpu.block.x * gpu.block.dim.x) + gpu.thread.x
    if i >= m:
      return
    free0 = free_arr[i] & board_mask
    if free0 == 0:
      results[i] = 0
      return
    jmark = jmark_arr[i]
    endm  = end_arr[i]
    mark1 = mark1_arr[i]
    mark2 = mark2_arr[i]
    sp = 0
    fid[0]    = funcid_arr[i]
    ld[0]     = ld_arr[i]
    rd[0]     = rd_arr[i]
    col[0]    = col_arr[i]
    row[0]    = row_arr[i]
    avail[0]  = free0
    # inited[0] = u8(0)
    total = 0
    while sp >= 0:
      a = avail[sp]
      if a == 0:
        sp -= 1
        continue
      if inited[sp] == u8(0):
        inited[sp] = u8(1)
        f = fid[sp]
        # デバッグ用：負の値で原因を返す（落とさない）
        # if f < 0 or f >= F:
        # for f0 in range(FCONST) を条件付きに（mが小さい時に効く）
        # use_local_tbl = 1 if m >= 2048 else 0
        # if use_local_tbl:
        #   if f < 0 or f >= int(FCONST):
        #     results[i] = -1000000 - f
        #     return
        # 
        nfid  = meta_next[f]
        # nfid  = m_next[f]
        aflag = meta_avail[f]
        # aflag = m_avail[f]
        # ---- 基底 ----
        if is_base[f] == 1 and row[sp] == endm:
        # if t_isb[f] == 1 and row[sp] == endm:
          if f == 14: # SQd2B 特例
            total += 1 if (a >> 1) else 0
          else:
            total += 1
          sp -= 1
          continue
        # ---- 通常状態設定 ----
        step[sp]  = 1
        add1[sp]  = 0
        rowst[sp] = row[sp] + 1
        use_blocks = 0
        use_future = 1 if (aflag == 1) else 0
        bK[sp]    = 0
        bL[sp]    = 0
        nextf[sp] = f
        # ---- mark 行: step=2/3 + block ----
        if is_mark[f] == 1:
        # if t_ism[f] == 1:
          sel = mark_sel[f]  # 1:mark1 2:mark2
          # sel = t_sel[f]  # 1:mark1 2:mark2
          at_mark = 0
          if sel == 1:
            if row[sp] == mark1:
              at_mark = 1
          if sel == 2:
            if row[sp] == mark2:
              at_mark = 1
          if at_mark and a:
            use_blocks = 1
            use_future = 0
            step[sp]  = mark_step[f]
            # step[sp]  = t_step[f]
            add1[sp]  = mark_add1[f]
            # add1[sp]  = t_add1[f]
            rowst[sp] = row[sp] + step[sp]
            bK[sp]    = blockK_tbl[f]
            # bK[sp]    = t_bK[f]
            bL[sp]    = blockL_tbl[f]
            # bL[sp]    = t_bL[f]
            nextf[sp] = nfid
        # ---- jmark ----
        if is_jmark[f] == 1:
        # if t_isj[f] == 1:
          if row[sp] == jmark:
            a &= ~1
            avail[sp] = a
            if a == 0:
              sp -= 1
              continue
            ld[sp] |= 1
            nextf[sp] = nfid
        ub[sp] = u8(1) if use_blocks else u8(0)
        fc[sp] = u8(0)
        if use_future == 1 and rowst[sp] < endm:
          fc[sp] = u8(1)
      # ---- 1bit 展開 ----
      a = avail[sp]
      bit = a & -a
      avail[sp] = a ^ bit
      # ---- 次状態計算（2値選択はそのまま）----
      if ub[sp] == u8(1):
        nld = ((ld[sp] | bit) << step[sp]) | add1[sp] | bL[sp]
        nrd = ((rd[sp] | bit) >> step[sp]) | bK[sp]
      else:
        nld = (ld[sp] | bit) << 1
        nrd = (rd[sp] | bit) >> 1
      ncol = col[sp] | bit
      nf = board_mask & ~(nld | nrd | ncol)
      if nf == 0:
        continue
      if fc[sp] == u8(1):
        if (board_mask & ~((nld << 1) | (nrd >> 1) | ncol)) == 0:
          continue
      # ---- push ----
      sp += 1
      if sp >= MAXD:
        results[i] = total * w_arr[i]
        return
      inited[sp] = u8(0)
      fid[sp]    = nextf[sp-1]
      ld[sp]     = nld
      rd[sp]     = nrd
      col[sp]    = ncol
      row[sp]    = rowst[sp-1]
      avail[sp]  = nf
    results[i] = total * w_arr[i]

"""  構造体配列 (SoA) タスク管理クラス """
class TaskSoA:
  """ コンストラクタ """
  def __init__(self, m: int):
    self.ld_arr: List[int] = [0]*m
    self.rd_arr: List[int] = [0]*m
    self.col_arr: List[int] = [0]*m
    self.row_arr: List[int] = [0]*m
    self.free_arr: List[int] = [0]*m
    self.jmark_arr: List[int] = [0]*m
    self.end_arr: List[int] = [0]*m
    self.mark1_arr: List[int] = [0]*m
    self.mark2_arr: List[int] = [0]*m
    self.funcid_arr: List[int] = [0]*m
    self.ijkl_arr: List[int] = [0]*m

""" Ｎクイーン ソルバ本体クラス（Codon/Python 共通部分） """
class NQueens17:

  """ コンストラクタ """
  def __init__(self,size:int)->None:
    # dynamicKのときFalse
    self.use_visited_prune = False  
    self.zobrist_tables:Dict[int,Dict[str,List[int]]]={}
    self.jasmin_cache:Dict[Tuple[int,int],int]={}
    self.subconst_cache:Set[StateKey]=set()
    self._N=size
    self._N1=size-1
    self._N2=size-2
    self._NK=1<<(size-3)
    self._NJ=1<<(size-1)
    self._board_mask=(1<<size)-1
    self._small_mask=(1<<self._N2)-1
    self._blockK:list[int]=[]
    self._blockL:list[int]=[]
    self._meta:list[Tuple[int,int,int]]=[]

  """ splitmix64 ミキサ最終段 """
  def mix64(self,x:int)->int:
    # MASK64:int=(1<<64)-1 # 64bit マスク（Zobrist用途）
    # x&=MASK64
    # x=(x^(x>>30))*0xBF58476D1CE4E5B9&MASK64
    # x=(x^(x>>27))*0x94D049BB133111EB&MASK64
    x=(x^(x>>30))*0xBF58476D1CE4E5B9
    x=(x^(x>>27))*0x94D049BB133111EB
    x^=(x>>31)
    # return x&MASK64
    return x

  """ Zobrist テーブル用乱数リスト生成 """
  def gen_list(self,cnt:int,seed:int)->List[int]:
    # MASK64:int=(1<<64)-1 # 64bit マスク（Zobrist用途）
    out:List[int]=[]
    # s:int=seed&MASK64
    s:int=seed
    _mix64=self.mix64
    for _ in range(cnt):
      # s=(s+0x9E3779B97F4A7C15)&MASK64   # splitmix64 のインクリメント
      s=(s+0x9E3779B97F4A7C15)
      # out.append(self._mix64(s))
      out.append(_mix64(s))
    return out

  """ Zobrist テーブル初期化 """
  def init_zobrist(self,N:int)->None:
    """Zobrist テーブルを N ごとに初期化する。キーは 'ld','rd','col','LD','RD','row','queens','k','l'。 ※ キャッシュは self.zobrist_tables[N] に格納して再利用する。"""
    # MASK64:int=(1<<64)-1 # 64bit マスク（Zobrist用途）
    if N in self.zobrist_tables:
      return
    # base_seed:int=(0xC0D0_0000_0000_0000^(N<<32))&MASK64
    base_seed:int=(0xC0D0_0000_0000_0000^(N<<32))
    gen_list=self.gen_list
    tbl:Dict[str,List[int]]={
      'ld':gen_list(N,base_seed^0x01),
      'rd':gen_list(N,base_seed^0x02),
      'col':gen_list(N,base_seed^0x03),
      'LD':gen_list(N,base_seed^0x04),
      'RD':gen_list(N,base_seed^0x05),
      'row':gen_list(N,base_seed^0x06),
      'queens':gen_list(N,base_seed^0x07),
      'k':gen_list(N,base_seed^0x08),
      'l':gen_list(N,base_seed^0x09),
    }
    self.zobrist_tables[N]=tbl

  """ Zobrist Hash を用いた盤面の 64bit ハッシュ値生成  """
  def zobrist_hash(self,
                  ld: int, rd: int, col: int,
                  row: int, queens: int, k: int, l: int,
                  LD: int, RD: int, N: int) -> u64:
    self.init_zobrist(N)
    tbl = self.zobrist_tables[N]

    # ここでテーブルが u64 で作られている前提（init_zobrist側も u64 に）
    ld_tbl  = tbl["ld"]    # List[u64]
    rd_tbl  = tbl["rd"]    # List[u64]
    col_tbl = tbl["col"]   # List[u64]
    LD_tbl  = tbl["LD"]    # List[u64]
    RD_tbl  = tbl["RD"]    # List[u64]
    row_tbl = tbl["row"]   # List[u64]
    q_tbl   = tbl["queens"]# List[u64]
    k_tbl   = tbl["k"]     # List[u64]
    l_tbl   = tbl["l"]     # List[u64]

    # mask は u64 で作る（1<<N が int のままだと型が揺れやすい）
    mask: u64 = (u64(1) << u64(N)) - u64(1)

    # 入力ビット集合を u64 に揃えてマスク
    ld64: u64  = u64(ld)  & mask
    rd64: u64  = u64(rd)  & mask
    col64: u64 = u64(col) & mask
    LD64: u64  = u64(LD)  & mask
    RD64: u64  = u64(RD)  & mask

    h: u64 = u64(0)

    m: u64 = ld64
    i: int = 0
    while i < N:
      if (m & u64(1)) != u64(0):
        h ^= u64(ld_tbl[i])
      m >>= u64(1)
      i += 1

    m = rd64; i = 0
    while i < N:
      if (m & u64(1)) != u64(0):
        h ^= u64(rd_tbl[i])
      m >>= u64(1)
      i += 1

    m = col64; i = 0
    while i < N:
      if (m & u64(1)) != u64(0):
        h ^= u64(col_tbl[i])
      m >>= u64(1)
      i += 1

    m = LD64; i = 0
    while i < N:
      if (m & u64(1)) != u64(0):
        h ^= u64(LD_tbl[i])
      m >>= u64(1)
      i += 1

    m = RD64; i = 0
    while i < N:
      if (m & u64(1)) != u64(0):
        h ^= u64(RD_tbl[i])
      m >>= u64(1)
      i += 1

    if 0 <= row < N:
      h ^= u64(row_tbl[row])
    if 0 <= queens < N:
      h ^= u64(q_tbl[queens])
    if 0 <= k < N:
      h ^= u64(k_tbl[k])
    if 0 <= l < N:
      h ^= u64(l_tbl[l])

    return h

  # def _zobrist_hash(self,ld:int,rd:int,col:int,row:int,queens:int,k:int,l:int,LD:int,RD:int,N:int)->int:

  #   self.init_zobrist(N)

  #   tbl = self.zobrist_tables[N]
  #   ld_tbl, rd_tbl, col_tbl = tbl['ld'], tbl['rd'], tbl['col']
  #   LD_tbl, RD_tbl = tbl['LD'], tbl['RD']
  #   row_tbl, q_tbl, k_tbl, l_tbl = tbl['row'], tbl['queens'], tbl['k'], tbl['l']
  #   mask = self._board_mask
  #   ld &= mask; rd &= mask; col &= mask; LD &= mask; RD &= mask
  #   h = 0
  #   m = ld; i = 0
  #   while i < N:
  #       if m & 1: h ^= ld_tbl[i]
  #       m >>= 1; i += 1
  #   m = rd; i = 0
  #   while i < N:
  #       if m & 1: h ^= rd_tbl[i]
  #       m >>= 1; i += 1
  #   m = col; i = 0
  #   while i < N:
  #       if m & 1: h ^= col_tbl[i]
  #       m >>= 1; i += 1
  #   m = LD; i = 0
  #   while i < N:
  #       if m & 1: h ^= LD_tbl[i]
  #       m >>= 1; i += 1
  #   m = RD; i = 0
  #   while i < N:
  #       if m & 1: h ^= RD_tbl[i]
  #       m >>= 1; i += 1
  #   if 0 <= row < N:    h ^= row_tbl[row]
  #   if 0 <= queens < N: h ^= q_tbl[queens]
  #   if 0 <= k < N:      h ^= k_tbl[k]
  #   if 0 <= l < N:      h ^= l_tbl[l]
  #   return h

  """(i,j,k,l) を 5bit×4=20bit にパック/アンパックするユーティリティ。 mirvert は上下ミラー（行: N-1-?）＋ (k,l) の入れ替えで左右ミラー相当を実現。"""
  def to_ijkl(self,i:int,j:int,k:int,l:int)->int:return (i<<15)+(j<<10)+(k<<5)+l
  def mirvert(self,ijkl:int,N:int)->int:return self.to_ijkl(N-1-self.geti(ijkl),N-1-self.getj(ijkl),self.getl(ijkl),self.getk(ijkl))
  def ffmin(self,a:int,b:int)->int:return min(a,b)
  def geti(self,ijkl:int)->int:return (ijkl>>15)&0x1F
  def getj(self,ijkl:int)->int:return (ijkl>>10)&0x1F
  def getk(self,ijkl:int)->int:return (ijkl>>5)&0x1F
  def getl(self,ijkl:int)->int:return ijkl&0x1F

  """(i,j,k,l) パック値に対して盤面 90°/180° 回転を適用した新しいパック値を返す。 回転の定義: (r,c) -> (c, N-1-r)。対称性チェック・正規化に利用。"""
  def rot90(self,ijkl:int,N:int)->int:return ((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl)
  def rot180(self,ijkl:int,N:int)->int:return ((N-1-self.getj(ijkl))<<15)+((N-1-self.geti(ijkl))<<10)+((N-1-self.getl(ijkl))<<5)+(N-1-self.getk(ijkl))
  def symmetry(self,ijkl:int,N:int)->int:return 2 if self.symmetry90(ijkl,N) else 4 if self.geti(ijkl)==N-1-self.getj(ijkl) and self.getk(ijkl)==N-1-self.getl(ijkl) else 8
  def symmetry90(self,ijkl:int,N:int)->bool:return ((self.geti(ijkl)<<15)+(self.getj(ijkl)<<10)+(self.getk(ijkl)<<5)+self.getl(ijkl))==(((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl))

  """与えた (i,j,k,l) の 90/180/270° 回転形が既出集合 ijkl_list に含まれるかを判定する。"""
  def check_rotations(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:return any(rot in ijkl_list for rot in [((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i,((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k),(l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)])

  """ キャッシュ付き Jasmin 正規化ラッパー """
  def get_jasmin(self,c:int,N:int)->int:
    """ Jasmin 正規化のキャッシュ付ラッパ。盤面パック値 c を回転/ミラーで規約化した代表値を返す。
    ※ キャッシュは self.jasmin_cache[(c,N)] に保持。
    [Opt-08] キャッシュ付き jasmin() のラッパー """
    # jasmin_cache:Dict[Tuple[int,int],int]={}
    key=(c,N)
    if key in self.jasmin_cache:
      return self.jasmin_cache[key]
    result=self.jasmin(c,N)
    self.jasmin_cache[key]=result
    return result

  """ Jasmin 正規化。盤面パック値 ijkl を回転/ミラーで規約化した代表値を返す。"""
  def jasmin(self,ijkl:int,N:int)->int:
    # 最初の最小値と引数を設定
    arg=0
    min_val=self.ffmin(self.getj(ijkl),N-1-self.getj(ijkl))
    # i: 最初の行（上端） 90度回転2回
    if self.ffmin(self.geti(ijkl),N-1-self.geti(ijkl))<min_val:
      arg=2
      min_val=self.ffmin(self.geti(ijkl),N-1-self.geti(ijkl))
    # k: 最初の列（左端） 90度回転3回
    if self.ffmin(self.getk(ijkl),N-1-self.getk(ijkl))<min_val:
      arg=3
      min_val=self.ffmin(self.getk(ijkl),N-1-self.getk(ijkl))
    # l: 最後の列（右端） 90度回転1回
    if self.ffmin(self.getl(ijkl),N-1-self.getl(ijkl))<min_val:
      arg=1
      min_val=self.ffmin(self.getl(ijkl),N-1-self.getl(ijkl))
    # 90度回転を arg 回繰り返す
    _rot90=self.rot90
    for _ in range(arg):
      # ijkl=self.rot90(ijkl,N)
      ijkl=_rot90(ijkl,N)
    # 必要に応じて垂直方向のミラーリングを実行
    if self.getj(ijkl)<N-1-self.getj(ijkl):
      ijkl=self.mirvert(ijkl,N)
    return ijkl

  """dfs()の非再帰版"""
  def dfs_iter(self,functionid:int,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int)->int:
    board_mask:int=self._board_mask
    total:int=0
    # スタック要素: (functionid, ld, rd, col, row, free)
    stack: List[Tuple[int,int,int,int,int,int]] = [(functionid,ld,rd,col,row,free)]
    while stack:
      functionid,ld,rd,col,row,free = stack.pop()
      if not free:
        continue
      next_funcid,funcptn,avail_flag = self._meta[functionid]
      avail:int = free
      # ---- 基底 ----
      if funcptn==5 and row==endmark:
        if functionid==14:  # SQd2B 特例
          total += 1 if (avail>>1) else 0
        else:
          total += 1
        continueas
      # 既定（+1）
      step:int=1
      add1:int=0
      row_step:int=row+1
      use_blocks:bool=False
      use_future:bool=(avail_flag==1)
      blockK:int=0
      blockL:int=0
      local_next_funcid:int=functionid
      # ---- mark 行: step=2/3 + block ----
      if funcptn in (0,1,2):
        at_mark:bool=(row==mark1) if funcptn in (0,2) else (row==mark2)
        if at_mark and avail:
          step=2 if funcptn in (0,1) else 3
          add1=1 if (funcptn==1 and functionid==20) else 0
          row_step=row+step
          blockK=self._blockK[functionid]
          blockL=self._blockL[functionid]
          use_blocks=True
          use_future=False
          local_next_funcid=next_funcid
      # ---- jmark 特殊 ----
      elif funcptn==3 and row==jmark:
        avail &= ~1
        ld |= 1
        local_next_funcid = next_funcid
        if not avail:
          continue
      # ==== ループ１：block 適用 ====
      if use_blocks:
        while avail:
          bit:int = avail & -avail
          avail &= avail-1
          nld:int = ((ld|bit)<<step) | add1 | blockL
          nrd:int = ((rd|bit)>>step) | blockK
          ncol:int = col | bit
          nf:int = board_mask&~(nld|nrd|ncol)
          if nf:
            stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))
        continue
      # ==== ループ２：+1 素朴（先読みなし）====
      if not use_future:
        while avail:
          bit:int = avail & -avail
          avail &= avail-1
          nld:int = (ld|bit)<<1
          nrd:int = (rd|bit)>>1
          ncol:int = col|bit
          nf:int = board_mask&~(nld|nrd|ncol)
          if nf:
            stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))
        continue
      # ==== ループ３：+1 先読み（row_step >= endmark は短絡）====
      if row_step>=endmark:
        while avail:
          bit:int = avail & -avail
          avail &= avail-1
          nld:int = (ld|bit)<<1
          nrd:int = (rd|bit)>>1
          ncol:int = col|bit
          nf:int = board_mask&~(nld|nrd|ncol)
          if nf:
            stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))
        continue
      # ==== ループ３B：先読み本体（1手先が0なら枝刈り）====
      while avail:
        bit:int = avail & -avail
        avail &= avail-1
        nld:int = (ld|bit)<<1
        nrd:int = (rd|bit)>>1
        ncol:int = col|bit
        nf:int = board_mask&~(nld|nrd|ncol)
        if not nf:
          continue
        if board_mask&~((nld<<1)|(nrd>>1)|ncol):
          stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))
    return total

  """汎用 DFS カーネル。古い SQ???? 関数群を 1 本化し、func_meta の記述に従って(1) mark 行での step=2/3 + 追加ブロック、(2) jmark 特殊、(3) ゴール判定、(4) +1 先読み を切り替える。"""
  def dfs(self,functionid:int,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int)->int:
    """汎用 DFS カーネル。古い SQ???? 関数群を 1 本化し、func_meta の記述に従って
    (1) mark 行での step=2/3 + 追加ブロック、(2) jmark 特殊、(3) ゴール判定、(4) +1 先読み
    を切り替える。引数:
    functionid: 現在の分岐モード ID（次の ID, パターン, 先読み有無は func_meta で決定）
    ld/rd/col:   斜線/列の占有
    row/free:    現在行と空きビット
    jmark/endmark/mark1/mark2: 特殊行/探索終端
    board_mask:  盤面全域のビットマスク
    blockK_by_funcid/blockL_by_funcid: 関数 ID に紐づく追加ブロック
    func_meta:   (next_id, funcptn, availptn) のメタ情報配列
    """
    # ---- ローカル束縛（属性アクセス最小化）----
    # _dfs = self.dfs
    # meta = self._meta
    # blockK_tbl = self._blockK
    # blockL_tbl = self._blockL
    board_mask:int=self._board_mask
    N1:int=self._N1
    NJ:int=self._NJ
    next_funcid,funcptn,avail_flag=self._meta[functionid]
    avail:int=free
    if not avail:return 0
    total=0
    # ---- N10:47 P6: 早期終了（基底）----
    if funcptn==5 and row==endmark:
      #print("pt5")
      if functionid==14:# SQd2B 特例（列0以外が残っていれば1）
        return 1 if (avail>>1) else 0
      return 1
    # ---- P5: N1-jmark 行の入口（据え置き分岐）----
    # if funcptn == 4 and row == (N1 - jmark):
    #   rd |= NJ
    #   nf = bm & ~( (ld << 1) | (rd >> 1) | col )
    #   if nf:
    #     return _dfs(next_funcid, ld << 1, rd >> 1, col, row, nf,
    #                     jmark, endmark, mark1, mark2)
    #   return 0
    # 既定（+1）
    step:int=1
    add1:int=0
    row_step:int=row+1
    use_blocks:bool=False  # blockK/blockL を噛ませるか
    use_future:bool=(avail_flag==1)  # _should_go_plus1 を使うか
    blockK:int=0
    blockL:int=0
    local_next_funcid:int=functionid
    # N10:538 P1/P2/P3: mark 行での step=2/3 ＋ block 適用を共通ループへ設定
    if funcptn in (0,1,2):
      #print("pt0,1,2")
      at_mark:bool=(row==mark1) if funcptn in (0,2) else (row==mark2)
      if at_mark and avail:
        step=2 if funcptn in (0,1) else 3
        add1=1 if (funcptn==1 and functionid==20) else 0  # SQd1BlB のときだけ +1
        row_step=row+step
        # blockK = blockK_tbl[functionid]
        blockK=self._blockK[functionid]
        # blockL = blockL_tbl[functionid]
        blockL=self._blockL[functionid]
        use_blocks=True
        use_future=False
        local_next_funcid=next_funcid
    # ---- N10:3 P4: jmark 特殊（入口一回だけ）----
    elif funcptn==3 and row==jmark:
      #print("pt3")
      avail&=~1     # 列0禁止
      ld|=1         # 左斜線LSBを立てる
      local_next_funcid=next_funcid
      if not avail:return 0
    # ==== N10:267 ループ１：block 適用（step=2/3 系のホットパス）====
    if use_blocks:
      #print("a_use_blocks")
      # s = step
      # a1 = add1
      # bK = blockK
      # bL = blockL
      while avail:
        bit:int=avail&-avail
        avail&=avail-1
        # nld:int = ((ld | bit) << s) | a1 | bL
        # nld:int = ((ld | bit) << s) | a1 | blockL
        # nld:int = ((ld | bit) << s) | add1 | blockL
        nld:int=((ld|bit)<<step)|add1|blockL
        # nrd:int = ((rd | bit) >> s) | bK
        # nrd:int = ((rd | bit) >> s) | blockK
        nrd:int=((rd|bit)>>step)|blockK
        ncol:int=col|bit
        nf:int=board_mask&~(nld|nrd|ncol)
        if nf:
          # total += _dfs(local_next_funcid, nld, nrd, ncol, row_step, nf, jmark, endmark, mark1, mark2)
          total+=self.dfs(local_next_funcid,nld,nrd,ncol,row_step,nf,jmark,endmark,mark1,mark2)
      return total
    # ==== N10:271 ループ２：+1 素朴（先読みなし）====
    elif not use_future:
      # if step == 1:
      #print("a_not_use_future")
      #if step == 1:
        #print("a_not_use_future_step1")
      while avail:
        bit:int=avail&-avail
        avail&=avail-1
        nld:int=(ld|bit)<<1
        nrd:int=(rd|bit)>>1
        ncol:int=col|bit
        nf:int=board_mask&~(nld|nrd|ncol)
        if nf:
          # total += _dfs(local_next_funcid, nld, nrd, ncol, row_step, nf, jmark, endmark, mark1, mark2)
          total+=self.dfs(local_next_funcid,nld,nrd,ncol,row_step,nf,jmark,endmark,mark1,mark2)
      return total
    # else:
    #   s = step
    #   a1 = add1
    #   while avail:
    #     bit = avail & -avail
    #     avail &= avail - 1
    #     nld = ((ld | bit) << s) | a1
    #     nrd = (rd | bit) >> s
    #     ncol = col | bit
    #     nf = bm & ~(nld | nrd | ncol)
    #     if nf:
    #       total += _dfs(local_next_funcid, nld, nrd, ncol, row_step, nf,
    #                             jmark, endmark, mark1, mark2)
    #   return total
    # ==== N10:92 ループ３：+1 先読み（row_step >= endmark は基底で十分）====
    elif row_step>=endmark:
      #print("a_endmark")
      # もう1手置いたらゴール層に達する → 普通の分岐で十分
      while avail:
        bit:int=avail&-avail
        avail&=avail-1
        nld:int=((ld|bit)<<1)
        nrd:int=((rd|bit)>>1)
        ncol:int=col|bit
        nf:int=board_mask&~(nld|nrd|ncol)
        if nf:
          # total += _dfs(local_next_funcid, nld, nrd, ncol, row_step, nf, jmark, endmark, mark1, mark2)
          total+=self.dfs(local_next_funcid,nld,nrd,ncol,row_step,nf,jmark,endmark,mark1,mark2)
      return total
    # ==== N10:402 ループ３B：+1 先読み本体（1手先の空きがゼロなら枝刈り）====
    while avail:
      #print("a_common")
      bit:int=avail&-avail
      avail&=avail-1
      nld:int=(ld|bit)<<1
      nrd:int=(rd|bit)>>1
      ncol:int=col|bit
      nf:int=board_mask&~(nld|nrd|ncol)
      if not nf:
        continue
      # 1手先の空きをその場で素早くチェック（余計な再帰を抑止）
      #   next_free_next = bm & ~(((nld << 1) | (nrd >> 1) | ncol))
      #   if next_free_next == 0: continue
      if board_mask&~((nld<<1)|(nrd>>1)|ncol):
        # total += _dfs(local_next_funcid, nld, nrd, ncol, row_step, nf, jmark, endmark, mark1, mark2)
        total+=self.dfs(local_next_funcid,nld,nrd,ncol,row_step,nf,jmark,endmark,mark1,mark2)
    return total

  """SoA タスク群を board_mask に基づいて分割する。元タスクは残りの分岐を担当し、子タスクは分割したビットだけ担当する。 max_new > 0 なら追加分の上限を設ける。"""
  def split_tasks_by_free(soa, board_mask:int, max_new:int=0) -> None:
    # max_new=0 なら全て分割。上限を設けたいなら使う。
    # 追加分を一旦ローカルに溜める（appendで再確保が頻繁だと遅いので）
    add_ld: List[int] = []
    add_rd: List[int] = []
    add_col: List[int] = []
    add_row: List[int] = []
    add_free: List[int] = []
    add_jmark: List[int] = []
    add_end: List[int] = []
    add_mark1: List[int] = []
    add_mark2: List[int] = []
    add_funcid: List[int] = []

    new_count = 0
    m = len(soa.free_arr)

    for i in range(m):
      free = soa.free_arr[i] & board_mask
      if free == 0:
        continue
      bit = free & -free
      rest = free ^ bit  # free - bit と同じ（bitがLSBなのでOK）

      # 元タスクは残りの分岐だけ担当
      soa.free_arr[i] = rest

      # 子タスクは bit だけ担当
      add_ld.append(soa.ld_arr[i])
      add_rd.append(soa.rd_arr[i])
      add_col.append(soa.col_arr[i])
      add_row.append(soa.row_arr[i])
      add_free.append(bit)  # ←ここがキモ

      add_jmark.append(soa.jmark_arr[i])
      add_end.append(soa.end_arr[i])
      add_mark1.append(soa.mark1_arr[i])
      add_mark2.append(soa.mark2_arr[i])
      add_funcid.append(soa.funcid_arr[i])
      new_count += 1
      if max_new > 0 and new_count >= max_new:
        break

    # 追加分を末尾に連結
    if new_count > 0:
      soa.ld_arr.extend(add_ld)
      soa.rd_arr.extend(add_rd)
      soa.col_arr.extend(add_col)
      soa.row_arr.extend(add_row)
      soa.free_arr.extend(add_free)
      soa.jmark_arr.extend(add_jmark)
      soa.end_arr.extend(add_end)
      soa.mark1_arr.extend(add_mark1)
      soa.mark2_arr.extend(add_mark2)
      soa.funcid_arr.extend(add_funcid)

  """各 Constellation（部分盤面）ごとに最適分岐（functionid）を選び、`dfs()` で解数を取得。 結果は `solutions` に書き込み、最後に `symmetry()` の重みで補正する。前段で SoA 展開し 並列化区間のループ体を軽量化。"""
  def exec_solutions(self,constellations:List[Dict[str,int]],N:int,use_gpu:bool)->None:
    N1:int = self._N1
    N2:int = self._N2
    small_mask:int=self._small_mask

    board_mask:int=self._board_mask
    dfs=self.dfs
    symmetry=self.symmetry
    getj,getk,getl=self.getj,self.getk,self.getl
    FUNC_CATEGORY={
      # N-3
      "SQBkBlBjrB":3,"SQBlkBjrB":3,"SQBkBjrB":3,
      "SQd2BkBlB":3,"SQd2BkB":3,"SQd2BlkB":3,
      "SQd1BkBlB":3,"SQd1BlkB":3,"SQd1BkB":3,"SQd0BkB":3,
      # N-4
      "SQBklBjrB":4,"SQd2BklB":4,"SQd1BklB":4,
      # 0（上記以外）
      "SQBlBjrB":0,"SQBjrB":0,"SQB":0,"SQBlBkBjrB":0,
      "SQBjlBkBlBjrB":0,"SQBjlBklBjrB":0,"SQBjlBlBkBjrB":0,"SQBjlBlkBjrB":0,
      "SQd2BlB":0,"SQd2B":0,"SQd2BlBkB":0,
      "SQd1BlB":0,"SQd1B":0,"SQd1BlBkB":0,"SQd0B":0
    }
    FID={
      "SQBkBlBjrB":0,"SQBlBjrB":1,"SQBjrB":2,"SQB":3,
      "SQBklBjrB":4,"SQBlBkBjrB":5,"SQBkBjrB":6,"SQBlkBjrB":7,
      "SQBjlBkBlBjrB":8,"SQBjlBklBjrB":9,"SQBjlBlBkBjrB":10,"SQBjlBlkBjrB":11,
      "SQd2BkBlB":12,"SQd2BlB":13,"SQd2B":14,"SQd2BklB":15,"SQd2BlBkB":16,
      "SQd2BkB":17,"SQd2BlkB":18,"SQd1BkBlB":19,"SQd1BlB":20,"SQd1B":21,
      "SQd1BklB":22,"SQd1BlBkB":23,"SQd1BlkB":24,"SQd1BkB":25,"SQd0B":26,"SQd0BkB":27
    }

    # next_funcid, funcptn, availptn の3つだけ持つ
    func_meta=[
      (1,0,0),#  0 SQBkBlBjrB   -> P1, 先読みなし
      (2,1,0),#  1 SQBlBjrB     -> P2, 先読みなし
      (3,3,1),#  2 SQBjrB       -> P4, 先読みあり
      (3,5,1),#  3 SQB          -> P6, 先読みあり
      (2,2,0),#  4 SQBklBjrB    -> P3, 先読みなし
      (6,0,0),#  5 SQBlBkBjrB   -> P1, 先読みなし
      (2,1,0),#  6 SQBkBjrB     -> P2, 先読みなし
      (2,2,0),#  7 SQBlkBjrB    -> P3, 先読みなし
      (0,4,1),#  8 SQBjlBkBlBjrB-> P5, 先読みあり
      (4,4,1),#  9 SQBjlBklBjrB -> P5, 先読みあり
      (5,4,1),# 10 SQBjlBlBkBjrB-> P5, 先読みあり
      (7,4,1),# 11 SQBjlBlkBjrB -> P5, 先読みあり
      (13,0,0),# 12 SQd2BkBlB    -> P1, 先読みなし
      (14,1,0),# 13 SQd2BlB      -> P2, 先読みなし
      (14,5,1),# 14 SQd2B        -> P6, 先読みあり（avail 特例）
      (14,2,0),# 15 SQd2BklB     -> P3, 先読みなし
      (17,0,0),# 16 SQd2BlBkB    -> P1, 先読みなし
      (14,1,0),# 17 SQd2BkB      -> P2, 先読みなし
      (14,2,0),# 18 SQd2BlkB     -> P3, 先読みなし
      (20,0,0),# 19 SQd1BkBlB    -> P1, 先読みなし
      (21,1,0),# 20 SQd1BlB      -> P2, 先読みなし（add1=1 は dfs 内で特別扱い）
      (21,5,1),# 21 SQd1B        -> P6, 先読みあり
      (21,2,0),# 22 SQd1BklB     -> P3, 先読みなし
      (25,0,0),# 23 SQd1BlBkB    -> P1, 先読みなし
      (21,2,0),# 24 SQd1BlkB     -> P3, 先読みなし
      (21,1,0),# 25 SQd1BkB      -> P2, 先読みなし
      (26,5,1),# 26 SQd0B        -> P6, 先読みあり
      (26,0,0),# 27 SQd0BkB      -> P1, 先読みなし
    ]
    F = len(func_meta)
    is_base  = [0]*F   # ptn==5
    is_jmark = [0]*F   # ptn==3
    is_mark  = [0]*F   # ptn in {0,1,2}

    mark_sel  = [0]*F  # 0:none 1:mark1 2:mark2
    mark_step = [1]*F  # 1 or 2 or 3
    mark_add1 = [0]*F  # 0/1
    # is_p6   = [0] * len(func_meta)
    # is_p4   = [0] * len(func_meta)
    # is_mark = [0] * len(func_meta)

    # mark_sel  = [0] * len(func_meta)   # 0:なし, 1:mark1, 2:mark2
    # mark_step = [0] * len(func_meta)   # 0:なし, 2 or 3
    # mark_add1 = [0] * len(func_meta)   # 0/1

    # for f,(nxt,ptn,avail) in enumerate(func_meta):
    #     if ptn == 5:
    #         is_p6[f] = 1
    #     elif ptn == 3:
    #         is_p4[f] = 1
    #     elif ptn == 0 or ptn == 1 or ptn == 2:
    #         is_mark[f] = 1
    #         if ptn == 1:
    #             mark_sel[f] = 2
    #             mark_step[f] = 2
    #             if f == 20:
    #                 mark_add1[f] = 1
    #         else:
    #             mark_sel[f] = 1
    #             mark_step[f] = 2 if ptn == 0 else 3
    for f,(nxt,ptn,aflag) in enumerate(func_meta):
        if ptn == 5:
            is_base[f] = 1
        elif ptn == 3:
            is_jmark[f] = 1
        elif ptn == 0 or ptn == 1 or ptn == 2:
            is_mark[f] = 1
            if ptn == 1:
                mark_sel[f]  = 2
                mark_step[f] = 2
                if f == 20:
                    mark_add1[f] = 1
            else:
                mark_sel[f]  = 1
                mark_step[f] = 2 if ptn == 0 else 3

    n3=1<<max(0,N-3)   # 念のため負シフト防止
    n4=1<<max(0,N-4)   # N3,N4とは違います
    size=max(FID.values())+1
    blockK_by_funcid=[0]*size
    blockL_by_funcid=[0,1,0,0,1,1,0,2,0,0,0,0,0,1,0,1,1,0,2,0,0,0,1,1,2,0,0,0]
    for fn,cat in FUNC_CATEGORY.items():# FUNC_CATEGORY: {関数名: 3 or 4 or 0}
      fid=FID[fn]
      blockK_by_funcid[fid]=n3 if cat==3 else (n4 if cat==4 else 0)
    # ===== 前処理ステージ（単一スレッド） =====
    m=len(constellations)

    # self._N=N
    # self._N1=N-1
    # self._NK=1<<(N-3)
    # self._NJ=1<<(N-1)
    # self._board_mask=(1<<N)-1
    self._blockK=blockK_by_funcid
    self._blockL=blockL_by_funcid
    self._meta=func_meta

    # bm = board_mask   # 以降は bm を使うと安全
    # N1 = self._N1
    # N2 = self._N2
    # NK = 1<<(N-3)
    # NJ = 1<<N1

    # SoA（Structure of Arrays）に展開：並列本体が軽くなる
    # ld_arr=[0]*m;rd_arr=[0]*m;col_arr=[0]*m
    # row_arr=[0]*m;free_arr=[0]*m
    # jmark_arr=[0]*m;end_arr=[0]*m
    # mark1_arr=[0]*m;mark2_arr=[0]*m
    # funcid_arr=[0]*m
    # ijkl_arr=[0]*m   # symmetry 計算用

    soa = TaskSoA(m)

    results=[0]*m
    target:int=0
    for i,constellation in enumerate(constellations):
      jmark=mark1=mark2=0
      start_ijkl=constellation["startijkl"]
      start=start_ijkl>>20
      ijkl=start_ijkl&((1<<20)-1)
      j,k,l=getj(ijkl),getk(ijkl),getl(ijkl)
      ld,rd,col=constellation["ld"]>>1,constellation["rd"]>>1,(constellation["col"]>>1)|(~small_mask)
      # まず col も盤面幅に正規化（保険。なくても動くが入れると事故が減る）
      col &= board_mask

      LD=(1<<(N1-j))|(1<<(N1-l))
      ld|=LD>>(N-start)
      if start>k:rd|=(1<<(N1-(start-k+1)))
      if j>=2*N-33-start:rd|=(1<<(N1-j))<<(N2-start)
      # free=~(ld|rd|col)
      free=board_mask&~(ld|rd|col)

      # 事前に固定値
      N1,N2=N-1,N-2

      # よく使う比較はフラグ化（1回だけ計算）
      j_lt_N3=(j<N-3)
      j_eq_N3=(j==N-3)
      j_eq_N2=(j==N-2)
      k_lt_l=(k<l)
      start_lt_k=(start<k)
      start_lt_l=(start<l)
      l_eq_kp1=(l==k+1)
      k_eq_lp1=(k==l+1)
      j_gate=(j>2*N-34-start)   # 既存の「ゲート」条件

      # ここから分岐（大分類 → 小分類）。同じ式の再評価をなくし、ネストを浅く。
      if j_lt_N3:
        jmark=j+1
        endmark=N2
        if j_gate:
          if k_lt_l:
            mark1,mark2=k-1,l-1
            if start_lt_l:
              if start_lt_k:target=0 if (not l_eq_kp1) else 4   # SQBkBlBjrB / SQBklBjrB
              else:target=1  # SQBlBjrB
            else:target=2   # SQBjrB
          else:
            mark1,mark2=l-1,k-1
            if start_lt_k:
              if start_lt_l:target=5 if (not k_eq_lp1) else 7   # SQBlBkBjrB / SQBlkBjrB
              else:target=6  # SQBkBjrB
            else:target=2  # SQBjrB
        else:
          if k_lt_l:
            mark1,mark2=k-1,l-1
            target=8 if (not l_eq_kp1) else 9   # SQBjlBkBlBjrB / SQBjlBklBjrB
          else:
            mark1,mark2=l-1,k-1
            target=10 if (not k_eq_lp1) else 11  # SQBjlBlBkBjrB / SQBjlBlkBjrB
      elif j_eq_N3:
        endmark=N2
        if k_lt_l:
          mark1,mark2=k-1,l-1
          if start_lt_l:
            if start_lt_k:target=12 if (not l_eq_kp1) else 15     # SQd2BkBlB / SQd2BklB
            else:
              mark2=l-1
              target=13 # SQd2BlB
          else:target=14 # SQd2B
        else:
          mark1,mark2=l-1,k-1
          if start_lt_k:
            if start_lt_l:target=16 if (not k_eq_lp1) else 18     # SQd2BlBkB / SQd2BlkB
            else:
              mark2=k-1
              target=17  # SQd2BkB
          else:target=14  # SQd2B
      elif j_eq_N2:# j がコーナーから1列内側
        if k_lt_l:
          endmark=N2
          if start_lt_l:
            if start_lt_k:
              mark1=k-1
              if not l_eq_kp1:
                mark2=l-1
                target=19 # SQd1BkBlB
              else:target=22 # SQd1BklB
            else:
              mark2=l-1
              target=20 # SQd1BlB
          else:target=21 # SQd1B
        else:# l < k
          if start_lt_k:
            if start_lt_l:
              if k<N2:
                mark1,endmark=l-1,N2
                if not k_eq_lp1:
                  mark2=k-1
                  target=23 # SQd1BlBkB
                else:target=24 # SQd1BlkB
              else:
                if l!=(N-3):
                  mark2,endmark=l-1,N-3
                  target=20  # SQd1BlB
                else:
                  endmark=N-4
                  target=21 # SQd1B
            else:
              if k!=N2:
                mark2,endmark=k-1,N2
                target=25  # SQd1BkB
              else:
                endmark=N-3
                target=21 # SQd1B
          else:
            endmark=N2
            target=21 # SQd1B
      # j がコーナー
      else:
        endmark=N2
        if start>k:target=26 # SQd0B
        else:
          mark1=k-1
          target=27 # SQd0BkB
      # 配列へ格納
      # ld_arr[i],rd_arr[i],col_arr[i],row_arr[i],free_arr[i],jmark_arr[i],end_arr[i],mark1_arr[i],mark2_arr[i],funcid_arr[i],ijkl_arr[i]=ld,rd,col,start,free,jmark,endmark,mark1,mark2,target,ijkl
      soa.ld_arr[i] = ld
      soa.rd_arr[i] = rd
      soa.col_arr[i] = col
      soa.row_arr[i] = start
      soa.free_arr[i] = free
      soa.jmark_arr[i] = jmark
      soa.end_arr[i] = endmark
      soa.mark1_arr[i] = mark1
      soa.mark2_arr[i] = mark2
      soa.funcid_arr[i] = target
      soa.ijkl_arr[i] = ijkl

    # ===== 並列ステージ：計算だけ =====
    w_arr = [0]*m
    @par
    for i in range(m):
      w_arr[i] = symmetry(soa.ijkl_arr[i], N)

    m = len(constellations)
    if m == 0: return
    # BLOCK= 64
    # BLOCK= 96
    # BLOCK= 128
    # BLOCK= 160
    # BLOCK= 192
    BLOCK = 256
    GRID = (m + BLOCK - 1) // BLOCK
    if GRID == 0:
      return

    if use_gpu:
      # meta_next  = [t[0] for t in func_meta]
      m_next  = [t[0] for t in func_meta]
      # meta_avail = [t[2] for t in func_meta]
      m_avail = [t[2] for t in func_meta]

      # 正常な例
      # max(funcid_arr) = 23
      # len(meta_next) = 24
      #
      # 危険な例（即 illegal access）
      # max(funcid_arr) = 24
      # len(meta_next) = 24   ← 24 は範囲外！
      # → GPU 実行前にチェックしておく
      # ダメな場合 : max(funcid_arr) >= len(meta_next)
      # print("max(funcid_arr) =", max(soa.funcid_arr),">=len(meta_next) =", len(meta_next))
      # 
      #
      # すべてmと同じになっているか
      # print("m =", m)
      # print("len(soa.ld_arr)   =", len(soa.ld_arr))
      # print("len(soa.rd_arr)   =", len(soa.rd_arr))
      # print("len(soa.col_arr)  =", len(soa.col_arr))
      # print("len(soa.row_arr)  =", len(soa.row_arr))
      # print("len(soa.free_arr) =", len(soa.free_arr))
      # print("len(soa.jmark_arr)=", len(soa.jmark_arr))
      # print("len(soa.end_arr)  =", len(soa.end_arr))
      # print("len(soa.mark1_arr)=", len(soa.mark1_arr))
      # print("len(soa.mark2_arr)=", len(soa.mark2_arr))
      # print("len(soa.funcid_arr)=", len(soa.funcid_arr))
      # print("len(w_arr)        =", len(w_arr))
      # print("len(results)      =", len(results))
      #
      results = [0] * m
      kernel_dfs_iter_gpu(
        gpu.raw(soa.ld_arr), gpu.raw(soa.rd_arr), gpu.raw(soa.col_arr),
        gpu.raw(soa.row_arr), gpu.raw(soa.free_arr),
        gpu.raw(soa.jmark_arr), gpu.raw(soa.end_arr),
        gpu.raw(soa.mark1_arr), gpu.raw(soa.mark2_arr),
        gpu.raw(soa.funcid_arr), gpu.raw(w_arr),
        # gpu.raw(meta_next), gpu.raw(meta_avail),  # ★ここを2本に
        gpu.raw(m_next), gpu.raw(m_avail),  # ★ここを2本に
        gpu.raw(blockK_by_funcid), gpu.raw(blockL_by_funcid),
        gpu.raw(is_base), gpu.raw(is_jmark), gpu.raw(is_mark),
        gpu.raw(mark_sel), gpu.raw(mark_step), gpu.raw(mark_add1),
        gpu.raw(results),
        m, board_mask,
        # len(m_next),
        grid=GRID, block=BLOCK
      )
    else:
      @par
      for i in range(m):
        cnt = self.dfs_iter(soa.funcid_arr[i],
              soa.ld_arr[i], soa.rd_arr[i], soa.col_arr[i], soa.row_arr[i],
              soa.free_arr[i], soa.jmark_arr[i], soa.end_arr[i],
              soa.mark1_arr[i], soa.mark2_arr[i])
        results[i]=cnt*w_arr[i]

    for i,constellation in enumerate(constellations):
      constellation["solutions"]=results[i]

  """サブコンステレーション生成のキャッシュ付ラッパ。StateKey で一意化し、 同一状態での重複再帰を回避して生成量を抑制する。"""
  def set_pre_queens_cached(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:List[int],constellations:List[Dict[str,int]],N:int,preset_queens:int,visited:Set[int],constellation_signatures:Set[Tuple[int,int,int,int,int,int]])->None:
    key:StateKey=(ld,rd,col,k,l,row,queens,LD,RD,N,preset_queens)
    # key: StateKey = (ld, rd, col, row, queens, k, l, LD, RD, N)

    # subconst_cache:Set[StateKey]=set()
    # インスタンス共有キャッシュを使う（ローカル初期化しない！）
    sc=self.subconst_cache
    if key in sc:
      return
    # ここで登録してから本体を呼ぶと、並行再入の重複も抑止できる
    sc.add(key)

    # if key in subconst_cache:
    #   # 以前に同じ状態で生成済み → 何もしない（または再利用）
    #   return

    # 新規実行（従来通りset_pre_queensの本体処理へ）
    self.set_pre_queens(ld,rd,col,k,l,row,queens,LD,RD,counter,constellations,N,preset_queens,visited,constellation_signatures)
    # subconst_cache[key] = True  # マークだけでOK
    # subconst_cache.add(key)

  """事前に置く行 (k,l) を強制しつつ、queens==preset_queens に到達するまで再帰列挙。 `visited` には軽量な `state_hash` を入れて枝刈り。到達時は {ld,rd,col,startijkl} を constellation に追加。"""
  def set_pre_queens(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:list,constellations:List[Dict[str,int]],N:int,preset_queens:int,visited:Set[int],constellation_signatures:Set[Tuple[int,int,int,int,int,int]])->None:
    mask = self._board_mask
    # ---------------------------------------------------------------------
    # 状態ハッシュによる探索枝の枝刈り バックトラック系の冒頭に追加　やりすぎると解が合わない
    #
    # <>zobrist_hash
    # 各ビットを見てテーブルから XOR するため O(N)（ld/rd/col/LD/RDそれぞれで最大 N 回）。
    # とはいえ N≤17 なのでコストは小さめ。衝突耐性は高い。
    # マスク漏れや負数の扱いを誤ると不一致が起きる点に注意（先ほどの & ((1<<N)-1) 修正で解決）。
    #
    # h: int = self.zobrist_hash(ld & mask, rd & mask, col & mask, row, queens, k, l, LD & mask, RD & mask, N)
    h: int = int(self.zobrist_hash(ld & mask, rd & mask, col & mask, row, queens, k, l, LD & mask, RD & mask, N))

    #
    # <>state_hash
    # その場で数個の ^ と << を混ぜるだけの O(1) 計算。
    # 生成されるキーも 単一の int なので、set/dict の操作が最速＆省メモリ。
    # ただし理論上は衝突し得ます（実際はN≤17の範囲なら実害が出にくい設計にしていればOK）。
    # [Opt-09] Zobrist Hash（Opt-09）の導入とその用途
    # ビットボード設計でも、「盤面のハッシュ」→「探索済みフラグ」で枝刈りは可能です。
    # 1意になるように領域を分けて詰める（衝突ゼロ）
    # 5*N ビット分で ld/rd/col/LD/RD を入れ、以降に小さい値を詰める
    #
    # ldm = ld & mask
    # rdm = rd & mask
    # colm = col & mask
    # LDm = LD & mask
    # RDm = RD & mask
    # base = 5 * N
    # h:int = (
    #     ldm
    #   | (rdm  << (1 * N))
    #   | (colm << (2 * N))
    #   | (LDm  << (3 * N))
    #   | (RDm  << (4 * N))
    #   | (row  << (base + 0))
    #   | (queens << (base + 6))
    #   | (k     << (base + 12))
    #   | (l     << (base + 18))
    #   | (N     << (base + 24))
    # )
    #
    # <>StateKey（タプル）
    # 11個の整数オブジェクトを束ねるため、オブジェクト生成・GC負荷・ハッシュ合成が最も重い。
    # set の比較・保持も重く、メモリも一番食います。
    # 衝突はほぼ心配ないものの、速度とメモリ効率は最下位。
    #
    # h: StateKey = (ld, rd, col, row, queens, k, l, LD, RD, N)
    #
    if self.use_visited_prune:
      if h in visited:
        return
      visited.add(h)

    #
    # ---------------------------------------------------------------------
    # k行とl行はスキップ
    if row==k or row==l:
      self.set_pre_queens_cached(ld<<1,rd>>1,col,k,l,row+1,queens,LD,RD,counter,constellations,N,preset_queens,visited,constellation_signatures)
      return
    # クイーンの数がpreset_queensに達した場合、現在の状態を保存
    # クイーンの数がpreset_queensに達した場合、現在の状態を保存
    if queens == preset_queens:
      if preset_queens <= 5:
          sig = (ld, rd, col, k, l, row)    # これが signature (tuple)
          if sig in constellation_signatures:
              return
          constellation_signatures.add(sig)
      # signatures=constellation_signatures
      # if signature not in signatures:
      constellation={"ld":ld,"rd":rd,"col":col,"startijkl":row<<20,"solutions":0}
      constellations.append(constellation) #星座データ追加
      # signatures.add(signature)
      counter[0]+=1
      return
    # 現在の行にクイーンを配置できる位置を計算
    free=~(ld|rd|col|(LD>>(N-1-row))|(RD<<(N-1-row)))&mask
    _set_pre_queens_cached=self.set_pre_queens_cached
    while free:
      bit:int=free&-free
      free&=free-1
      _set_pre_queens_cached((ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,counter,constellations,N,preset_queens,visited,constellation_signatures)

  """開始コンステレーション（代表部分盤面）の列挙。中央列（奇数 N）特例、回転重複排除 （`check_rotations`）、Jasmin 正規化（`get_jasmin`）を経て、各 sc から `set_pre_queens_cached` でサブ構成を作る。"""
  def gen_constellations(self,ijkl_list:Set[int],constellations:List[Dict[str,int]],N:int,preset_queens:int)->None:
    # 実行ごとにメモ化をリセット（N や preset_queens が変わるとキーも変わるが、
    # 長時間プロセスなら念のためクリアしておくのが安全）
    self.subconst_cache.clear()

    halfN=(N+1)//2  # Nの半分を切り上げ
    N1:int=N-1
    constellation_signatures: Set[ Tuple[int, int, int, int, int, int] ] = set()
    # --- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---
    if N%2==1:
      center=N//2
      ijkl_list.update(
        self.to_ijkl(i,j,center,l)
        for l in range(center+1,N1)
        for i in range(center+1,N1)
        if i!=(N1)-l
        for j in range(N-center-2,0,-1)
        if j!=i and j!=l
        if not self.check_rotations(ijkl_list,i,j,center,l,N)
      )
    # --- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---
    # コーナーにクイーンがいない場合の開始コンステレーションを計算する
    ijkl_list.update(self.to_ijkl(i,j,k,l) for k in range(1,halfN) for l in range(k+1,N1) for i in range(k+1,N-1) if i!=(N-1)-l for j in range(N-k-2,0,-1) if j!=i and j!=l if not self.check_rotations(ijkl_list,i,j,k,l,N))
    # コーナーにクイーンがある場合の開始コンステレーションを計算する
    ijkl_list.update({self.to_ijkl(0,j,0,l) for j in range(1,N-2) for l in range(j+1,N1)})
    # Jasmin変換
    ijkl_list={self.get_jasmin(c,N) for c in ijkl_list}
    L=1<<(N1)  # Lは左端に1を立てる
    # ローカルアクセスに変更
    geti,getj,getk,getl=self.geti,self.getj,self.getk,self.getl
    to_ijkl=self.to_ijkl
    _set_pre_queens_cached=self.set_pre_queens_cached
    for sc in ijkl_list:
      # ここで毎回クリア（＝この sc だけの重複抑止に限定）
      constellation_signatures=set()
      i,j,k,l=geti(sc),getj(sc),getk(sc),getl(sc)
      Lj=L>>j;Li=L>>i;Ll=L>>l
      ld=((L>>(i-1)) if i>0 else 0)|(1<<(N-k))
      rd=(L>>(i+1))|(1<<(l-1))
      col=1|L|Li|Lj
      LD=Lj|Ll
      RD=Lj|(1<<k)
      counter:List[int]=[0] # サブコンステレーションを生成
      visited:Set[int]=set()
      # visited:Set[Tuple[int,int,int,int,int,int,int,int,int,int]] = set()
      # visited:Set[StateKey] = set()
      _set_pre_queens_cached(ld,rd,col,k,l,1,3 if j==N1 else 4,LD,RD,counter,constellations,N,preset_queens,visited,constellation_signatures)
      current_size=len(constellations)
      base=to_ijkl(i,j,k,l)
      for a in range(counter[0]):
        constellations[-1-a]["startijkl"]|=base

  """constellations の各要素が {ld, rd, col, startijkl} を全て持つかを検証する。"""
  def validate_constellation_list(self,constellations:List[Dict[str,int]])->bool: return all(all(k in c for k in ("ld","rd","col","startijkl")) for c in constellations)

  """32bit little-endian の相互変換ヘルパ。Codon/CPython の差異に注意。"""
  def read_uint32_le(self,b:str)->int: return (ord(b[0])&0xFF)|((ord(b[1])&0xFF)<<8)|((ord(b[2])&0xFF)<<16)|((ord(b[3])&0xFF)<<24)
  def int_to_le_bytes(self,x:int)->List[int]: return [(x>>(8*i))&0xFF for i in range(4)]

  """ファイル存在チェック（読み取り open の可否で判定）。"""
  def file_exists(self,fname:str)->bool:
    try:
      with open(fname,"rb"):
        return True
    except:
      return False

  """bin キャッシュのサイズ妥当性確認（1 レコード 16 バイトの整数倍か）。"""
  def validate_bin_file(self,fname:str)->bool:
    try:
      with open(fname,"rb") as f:
        f.seek(0,2)  # ファイル末尾に移動
        size=f.tell()
      return size%16==0
    except:
      return False

  """テキスト形式で constellations を保存/復元する（1 行 5 数値: ld rd col startijkl solutions）。"""
  def save_constellations_txt(self,path:str,constellations:List[Dict[str,int]])->None:
    with open(path,"w") as f:
      for c in constellations:
        ld=c["ld"]
        rd=c["rd"]
        col=c["col"]
        startijkl=c["startijkl"]
        solutions=c.get("solutions",0)
        f.write(f"{ld} {rd} {col} {startijkl} {solutions}\n")

  """テキスト形式で constellations を保存/復元する（1 行 5 数値: ld rd col startijkl solutions）。"""
  def load_constellations_txt(self,path:str)->List[Dict[str,int]]:
    out:List[Dict[str,int]]=[]
    with open(path,"r") as f:
      for line in f:
        parts=line.strip().split()
        if len(parts)!=5:
          continue
        ld=int(parts[0]);rd=int(parts[1]);col=int(parts[2])
        startijkl=int(parts[3]);solutions=int(parts[4])
        out.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions": solutions})
    return out

  """テキストキャッシュを読み込み。壊れていれば `gen_constellations()` で再生成して保存する。"""
  def load_or_build_constellations_txt(self,ijkl_list:Set[int],constellations,N:int,preset_queens:int)->List[Dict[str,int]]:
    # N と preset_queens に基づいて一意のファイル名を構成
    fname=f"constellations_N{N}_{preset_queens}.txt"
    # ファイルが存在すれば読み込むが、破損チェックも行う
    if self.file_exists(fname):
      try:
        constellations=self.load_constellations_txt(fname)
        if self.validate_constellation_list(constellations):
          return constellations
        else:
          print(f"[警告] 不正なキャッシュ形式: {fname} を再生成します")
      except Exception as e:
        print(f"[警告] キャッシュ読み込み失敗: {fname}, 理由: {e}")
    # ファイルがなければ生成・保存
    constellations:List[Dict[str,int]]=[]
    self.gen_constellations(ijkl_list,constellations,N,preset_queens)
    self.save_constellations_txt(fname,constellations)
    return constellations

  """bin 形式で constellations を保存/復元。Codon では str をバイト列として扱う前提のため、CPython では bytes で書き込むよう分岐/注意が必要。"""
  def save_constellations_bin(self,fname:str,constellations:List[Dict[str,int]])->None:
    _int_to_le_bytes=self.int_to_le_bytes
    with open(fname,"wb") as f:
      for d in constellations:
        for key in ["ld","rd","col","startijkl"]:
          b=_int_to_le_bytes(d[key])
          _int_to_le_bytes(d[key])
          f.write("".join(chr(c) for c in b))  # Codonでは str がバイト文字列扱い

  """bin 形式で constellations を保存/復元。Codon では str をバイト列として扱う前提のため、CPython では bytes で書き込むよう分岐/注意が必要。"""
  def load_constellations_bin(self,fname:str)->List[Dict[str,int]]:
    constellations:List[Dict[str,int]]=[]
    _read_uint32_le=self.read_uint32_le
    with open(fname,"rb") as f:
      while True:
        raw=f.read(16)
        if len(raw)<16:
          break
        ld=self.read_uint32_le(raw[0:4])
        rd=self.read_uint32_le(raw[4:8])
        col=self.read_uint32_le(raw[8:12])
        startijkl=_read_uint32_le(raw[12:16])
        constellations.append({ "ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0 })
    return constellations

  """bin キャッシュを読み込み。検証に失敗した場合は再生成して保存し、その結果を返す。"""
  def load_or_build_constellations_bin(self,ijkl_list:Set[int],constellations,N:int,preset_queens:int)->List[Dict[str,int]]:
    fname=f"constellations_N{N}_{preset_queens}.bin"
    if self.file_exists(fname):
      # ファイルが存在すれば読み込むが、破損チェックも行う
      try:
        constellations=self.load_constellations_bin(fname)
        if self.validate_bin_file(fname) and self.validate_constellation_list(constellations):
          return constellations
        else:
          print(f"[警告] 不正なキャッシュ形式: {fname} を再生成します")
      except Exception as e:
        print(f"[警告] キャッシュ読み込み失敗: {fname}, 理由: {e}")
    # ファイルがなければ生成・保存
    constellations:List[Dict[str,int]]=[]
    self.gen_constellations(ijkl_list,constellations,N,preset_queens)
    self.save_constellations_bin(fname,constellations)
    return constellations

  """プリセットクイーン数の選択。N と use_gpu に基づいて適切な値を返す。"""
  def choose_preset_queens(self, N: int, use_gpu: bool) -> int:
    # まずは分かりやすい段階式（後で調整しやすい）
    if not use_gpu:
      # CPU: これまでの値をほぼ踏襲
      if N <= 17: return 4
      if N <= 20: return 5
      if N <= 23: return 6
      return 7
    else:
      # GPU: タスク数不足・偏り対策で1段深め
      if N <= 17: return 4
      if N <= 19: return 5
      if N <= 22: return 6
      return 7

""" NQueens17_constellations クラス：小さな N 用の素朴な全列挙（対称重みなし）。ビットボードで列/斜線の占有を管理して再帰的に合計を返す。検算/フォールバック用。"""
class NQueens17_constellations():

  """プリセットクイーン数の選択。N と use_gpu に基づいて適切な値を返す。"""
  def choose_preset_queens(self, size:int, use_gpu: bool) -> int:
    if not use_gpu:
      # return 5  # まず固定でOK（後で詰める）
      # CPUはオーバーヘッドが軽いので控えめでも回る
      if size <= 17: return 5
      if size <= 19: return 6
      if size <= 22: return 7
      return 8
    # GPU: Nが上がるほど深めに
    if size <= 17: return 9
    if size <= 19: return 10
    if size <= 22: return 11
    return 12

  """プリセットクイーン数を動的に調整しつつ星座リストを生成/読み込み。GPU 時は目標タスク数 m_target を満たすまで深くする。最大3回試行。"""
  def build_constellations_dynamicK(self, NQ, size: int, use_gpu: bool) -> Tuple[int, List[Dict[str,int]]]:
    # 一時的に枝刈りを無効化
    had_flag = hasattr(NQ, "use_visited_prune")
    old = NQ.use_visited_prune if had_flag else False
    if had_flag:
      NQ.use_visited_prune = False
    try:
      # m_target = 200_000 if use_gpu else 30_000
      # GPU 時はサイズに応じて目標タスク数を変える
      m_target = 200_000 if use_gpu and size >= 15 else 0
      K = NQ.choose_preset_queens(size, use_gpu)
      # 動的対応が課題：最大2回試行（合計2回）できればなおオッケー
      # for _ in range(2):  # N18でＮＧになります
      for _ in range(1):  
        ijkl_list:Set[int] = set()
        constellations:List[Dict[str,int]] = []
        constellations = NQ.load_or_build_constellations_bin(ijkl_list, constellations, size, K)
        m = len(constellations)
        if m == 0:
          # 解は0ではないので、ここは「Kが深すぎてタスクが潰れた」扱い
          # → Kを下げて作り直す or CPUにフォールバック
          print(f"m==0 (preset_queens={K}) fallback")
          # 例：Kを1下げて作り直す
          K -= 1
          ijkl_list=set(); constellations=[]
          constellations = NQ.load_or_build_constellations_bin(ijkl_list,constellations,size,K)
          m = len(constellations)
        # 
        # print("K:", K, "m:", len(constellations))
        # 
        # 目標に届けば採用
        if m >= m_target:
          return K, constellations
        # 足りなければ K を増やして作り直し
        K += 1
      #
      # print(f"size={size} K={K} m={len(constellations)}")
      #
      return K, constellations
    finally:
      NQ.use_visited_prune = old

  """小さな N 用の素朴な全列挙（対称重みなし）。ビットボードで列/斜線の占有を管理して再帰的に合計を返す。検算/フォールバック用。"""
  def _bit_total(self,size:int)->int:
    mask=(1<<size)-1
    total=0

    """ 小さなNは正攻法で数える（対称重みなし・全列挙） """
    def bt(row:int,left:int,down:int,right:int):
      nonlocal total
      if row==size:
        total+=1
        return
      bitmap=mask&~(left|down|right)
      while bitmap:
        bit=-bitmap&bitmap
        bitmap^=bit
        bt(row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
    bt(0,0,0,0)
    return total

  """ 戻り値: use_gpu: bool """
  def parse_args(self,argv: list[str]) -> tuple[bool]:
    use_gpu = False
    i=0
    while i < len(argv):
      arg = argv[i]
      if arg == "-g":
          use_gpu = True
      elif arg == "-c":
          use_gpu = False
      else:
          raise ValueError(f"Unknown argument: {arg}")
      i += 1
    return use_gpu

  """N=5..17 の合計解を計測。N<=5 は `_bit_total()` のフォールバック、それ以外は星座キャッシュ（.bin/.txt）→ `exec_solutions()` → 合計→既知解 `expected` と照合。"""
  def main(self)->None:
    expected:List[int]=[0,0,0,0,0,10,4,40,92,352,724,2680,14200,73712,365596,2279184,14772512,95815104,666090624,4968057848,39029188884,314666222712,2691008701644,24233937684440,227514171973736,2207893435808352,22317699616364044,234907967154122528]     
    nmin:int=5
    nmax:int=len(expected)
    use_gpu=False
    argc=len(sys.argv)
    if argc == 1:
      print("CPU mode selected")
      pass
    elif argc == 2:
      arg = sys.argv[1]
      if arg == "-c":
        use_gpu = False
        print("CPU mode selected")
      elif arg == "-g":
        use_gpu = True
        print("GPU mode selected")
      else:
        print(f"Unknown option: {arg}")
        print("Usage: nqueens [-c | -g]")
        return
    else:
      print("Too many arguments")
      print("Usage: nqueens [-c | -g]")
      return
    print(" N:             Total         Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      if size<=5:
        total=self._bit_total(size)
        dt=datetime.now()-start_time
        text=str(dt)[:-3]
        print(f"{size:2d}:{total:18d}{0:15d}{text:>20s}")
        continue
      #
      ijkl_list:Set[int]=set()
      constellations:List[Dict[str,int]]=[]
      NQ=NQueens17(size)
      # 事前クイーン数の選択
      preset_queens=NQ.choose_preset_queens(size,use_gpu)
      #---------------------------------
      # 星座リストそのものをキャッシュ
      #---------------------------------
      #
      # キャッシュを使わない
      # print(f"size={size} chosen_K={preset_queens}")
      # NQ.gen_constellations(ijkl_list,constellations,size,preset_queens)
      # print(f"size={size} K={preset_queens} m={len(constellations)}")
      #
      # キャッシュを使う
      # t0 = datetime.now()
      preset_queens, constellations = self.build_constellations_dynamicK(NQ, size, use_gpu)
      # t1 = datetime.now()
      NQ.exec_solutions(constellations, size, use_gpu)
      # t2 = datetime.now()
      # print(f"build={(t1-t0)} exec={(t2-t1)} m={len(constellations)} K={preset_queens}")
      #
      # 合計
      total:int=sum(c['solutions'] for c in constellations if c['solutions']>0)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      status:str="ok" if expected[size]==total else f"ng({total}!={expected[size]})"
      print(f"{size:2d}:{total:18d}{0:15d}{text:>20s}    {status}")
""" エントリポイント """
if __name__=="__main__":
  NQueens17_constellations().main()
