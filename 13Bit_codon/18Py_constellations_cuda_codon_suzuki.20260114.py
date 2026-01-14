#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Python/codon Ｎクイーン コンステレーション版 インテグレート

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
 N-Queens 高速ソルバ（NQueens17）
   - ビットボード + “星座（Constellation）”分割探索
   - 回転・鏡像対称の正規化（Jasmin）と重複補正（symmetry）
   - 事前配置（preset_queens）で探索を分割し、部分盤面ごとにDFS
   - メモ化・キャッシュ（sub-constellation/state/Zobrist/星座I/O）

 ■ 目的
   大きめの N でも高速に全解数を数える。探索の“入口”を「星座」で細分化し、
   各星座を最適な分岐（SQ* 系）で解く。

 ■ この実装のキモ
   - 64bit マスク定義:  MASK64 = (1<<64) - 1
   - Zobrist の永続キャッシュ:  self.zobrist_tables[N] を使い回し
       -> zobrist_hash() は必ず self.init_zobrist(N) を呼んでから参照
          例: self.init_zobrist(N); tbl = self.zobrist_tables[N]
   - N ビット正規化の徹底（上位ビット落とし/負数対策）
          ld &= (1<<N)-1; rd &= (1<<N)-1; col &= (1<<N)-1; ...
   - 盤面の正規化: jasmin() で (i,j,k,l) を回転/鏡像して代表形へ
   - 対称重複補正: symmetry(ijkl,N) が 2/4/8 を返し、解数を補正
   - 探索の枝刈り:
       * state_hash (軽量O(1)) で visited を管理（衝突許容）
       * Zobrist（低衝突）も使用可（N<=17 では O(1)版で十分）
   - DFS の一本化（dfs()）:
       SQ* の分岐を function id + メタ情報（func_meta）に集約し、分岐・先読み・ブロックを
       同じ配置ループで処理して高速化。

 ■ パフォーマンスの要点
   - ループ内辞書アクセスの削減（ローカル束縛）:
       ld_tbl, rd_tbl = tbl['ld'], tbl['rd'] など
   - “立っているビットのみ”の走査（必要なら差し替え可）
   - SoA 配列化 → exec_solutions() の前処理で配列へ詰め、並列セクションを軽量化

 ■ 注意/設計上のポイント
   - Zobrist 初期化は必ず init_zobrist() 経由で。zobrist_hash() 内で new dict を作らない。
   - N ビット正規化を忘れるとハッシュ不一致・誤カウントの原因に。
   - set_pre_queens_* の visited はキー衝突に留意（N<=17 では実害少）。
   - I/O キャッシュ（.txt / .bin）読込時は validate_* でフォーマット検証。

 ■ 代表的な引用
   - “64bit マスク”： MASK64 = (1<<64) - 1
   - “Zobrist テーブル使用”：
        self.init_zobrist(N); tbl = self.zobrist_tables[N]
   - “N ビット正規化”：
        mask = (1<<N)-1; ld &= mask; rd &= mask; col &= mask; LD &= mask; RD &= mask
   - “回転/鏡像の正規化（Jasmin）”： jasmin(ijkl, N)
   - “対称重複補正”： symmetry(ijkl, N) -> {2,4,8}
   - “一本化 DFS”： dfs(functionid, ld, rd, col, row, free, ...)

"""


"""
17Py_constellations_integrate_codon.py（レビュー＆注釈つき）

amazon AWS m4.16xlarge x 1
$ codon build -release 15Py_constellations_optimize_codon.py && ./15Py_constellations_optimize_codon
 N:            Total       Unique        hh:mm:ss.ms
 5:               10            0         0:00:00.000
 6:                4            0         0:00:00.079
 7:               40            0         0:00:00.001
 8:               92            0         0:00:00.001
 9:              352            0         0:00:00.001
10:              724            0         0:00:00.002
11:             2680            0         0:00:00.102
12:            14200            0         0:00:00.002
13:            73712            0         0:00:00.005
14:           365596            0         0:00:00.011
15:          2279184            0         0:00:00.035
16:         14772512            0         0:00:00.078
17:         95815104            0         0:00:00.436
18:        666090624            0         0:00:02.961
19:       4968057848            0         0:00:22.049
20:      39029188884            0         0:02:52.430
21:     314666222712            0         0:24:25.554
22:    2691008701644            0         3:29:33.971
23:   24233937684440            0  1 day, 8:12:58.977


workspace#suzuki$ bash MAIN.SH 15Py_constellations_optimize_codon.py
 N:        Total       Unique        hh:mm:ss.ms
17:     95815104            0         0:00:02.987
18:    666090624            0         0:00:21.549
19:   4968057848            0         0:02:43.514

workspace#suzuki$ bash MAIN.SH 17Py_constellations_integrate_codon.py
 N:        Total       Unique        hh:mm:ss.ms
17:     95815104            0         0:00:03.445    ok
18:    666090624            0         0:00:29.301    ok
19:   4968057848            0         0:03:41.853    ok
"""

from typing import List,Set,Dict,Tuple
from datetime import datetime
import gpu

StateKey=Tuple[int,int,int,int,int,int,int,int,int,int,int]
MASK64=(1<<64)-1
MAXS=64  # N=18なら十分余裕（必要なら 96/128 に）
FIELDS=18
PACK_FIELDS  = 10   # pack は ld..fid の 10 要素
FRAME_FIELDS = 20
FRAME_MAXS   = 128  # 必要なら増やす
# ---- dfs_iter_buf 用：フレーム定義（20フィールド）----
F_FID        = 0
F_LD         = 1
F_RD         = 2
F_COL        = 3
F_ROW        = 4
F_AVAIL      = 5
F_JMARK      = 6
F_END        = 7
F_M1         = 8
F_M2         = 9
F_STATE      = 10   # 0=init, 1=loop
F_ACC        = 11

# 追加で保持する「dfs() の分岐結果」
F_STEP       = 12
F_ADD1       = 13
F_ROWSTEP    = 14
F_USE_BLOCKS = 15   # 0/1
F_USE_FUTURE = 16   # 0/1
F_BLOCKK     = 17
F_BLOCKL     = 18
F_LOCAL_NEXT = 19
# -----------------------------
# 非再帰DFS（GPU/CPU共通で使える）
# -----------------------------
def dfs_iter(functionid:int,ld0:int,rd0:int,col0:int,row0:int,free0:int,
             jmark0:int,endmark0:int,mark10:int,mark20:int,
             bm:int,N1:int,
             next_funcid:List[int],funcptn:List[int],avail_flag:List[int],
             blockK_tbl:List[int],blockL_tbl:List[int])->int:
  """
  17Py の dfs() を “忠実に” 非再帰化したもの。
  - 再帰呼び出しを stack push に置換
  - 戻り値は acc の集計で親へ伝播
  """
  MAXS=64  # N=18なら十分余裕（必要なら 96/128 に）
  FIELDS=18
  # dfs_iter の冒頭あたりで（jmark0 は引数で来てる）
  NJ = 1 << (N1 - jmark0)

  # FRAME_FIELDS = 20
  # FRAME_MAXS   = 128  # 必要なら増やす
  # PACK_FIELDS  = 10   # pack は ld..fid の 10 要素
  META_FIELDS  = 3    # (next_id, funcptn, avail_flag)
  BLOCK_FIELDS = 2    # (blockK, blockL)

  # ---- dfs_iter_buf 用：フレーム定義（20フィールド）----
  F_FID        = 0
  F_LD         = 1
  F_RD         = 2
  F_COL        = 3
  F_ROW        = 4
  F_AVAIL      = 5
  F_JMARK      = 6
  F_END        = 7
  F_M1         = 8
  F_M2         = 9
  F_STATE      = 10   # 0=init, 1=loop
  F_ACC        = 11

  # 追加で保持する「dfs() の分岐結果」
  F_STEP       = 12
  F_ADD1       = 13
  F_ROWSTEP    = 14
  F_USE_BLOCKS = 15   # 0/1
  F_USE_FUTURE = 16   # 0/1
  F_BLOCKK     = 17
  F_BLOCKL     = 18
  F_LOCAL_NEXT = 19

  # stack arrays
  s_func  = [0]*MAXS
  s_ld    = [0]*MAXS
  s_rd    = [0]*MAXS
  s_col   = [0]*MAXS
  s_row   = [0]*MAXS
  s_avail = [0]*MAXS
  s_jmark = [0]*MAXS
  s_end   = [0]*MAXS
  s_m1    = [0]*MAXS
  s_m2    = [0]*MAXS

  # per-frame initialized state
  s_inited = [0]*MAXS
  s_acc    = [0]*MAXS

  s_step   = [0]*MAXS
  s_add1   = [0]*MAXS
  s_rowst  = [0]*MAXS
  s_useblk = [0]*MAXS
  s_usefut = [0]*MAXS
  s_lnext  = [0]*MAXS
  s_bK     = [0]*MAXS
  s_bL     = [0]*MAXS
  s_ptn    = [0]*MAXS

  sp = 0
  s_func[0]  = functionid
  s_ld[0]    = ld0
  s_rd[0]    = rd0
  s_col[0]   = col0
  s_row[0]   = row0
  s_avail[0] = free0
  s_jmark[0] = jmark0
  s_end[0]   = endmark0
  s_m1[0]    = mark10
  s_m2[0]    = mark20

  s_inited[0] = 0
  s_acc[0] = 0

  while True:
    fid = s_func[sp]
    ld  = s_ld[sp]
    rd  = s_rd[sp]
    col = s_col[sp]
    row = s_row[sp]
    avail = s_avail[sp]
    jmark = s_jmark[sp]
    endmark = s_end[sp]
    mark1 = s_m1[sp]
    mark2 = s_m2[sp]

    if avail == 0:
      # このフレームは何もできない
      res = s_acc[sp]
      if sp == 0:
        return res
      sp -= 1
      s_acc[sp] += res
      continue

    if s_inited[sp] == 0:
      # ---- 初期化（dfs() 冒頭の処理をそのまま）----
      nf_id = next_funcid[fid]
      ptn   = funcptn[fid]
      aflag = avail_flag[fid]
      s_ptn[sp] = ptn

      # base (P6)
      if ptn == 5 and row == endmark:
        if fid == 14:
          # SQd2B 特例
          res = 1 if (avail >> 1) else 0
        else:
          res = 1
        if sp == 0:
          return res
        sp -= 1
        s_acc[sp] += res
        continue

      step = 1
      add1 = 0
      row_step = row + 1
      use_blocks = 0
      use_future = 1 if (aflag == 1) else 0
      blockK = 0
      blockL = 0
      local_next = fid

      # P1/P2/P3: mark 行の step=2/3 + blocks
      if ptn == 0 or ptn == 1 or ptn == 2:
        at_mark = (row == mark1) if (ptn == 0 or ptn == 2) else (row == mark2)
        if at_mark and avail:
          step = 2 if (ptn == 0 or ptn == 1) else 3
          add1 = 1 if (ptn == 1 and fid == 20) else 0  # SQd1BlB の特例
          row_step = row + step
          blockK = blockK_tbl[fid]
          blockL = blockL_tbl[fid]
          use_blocks = 1
          use_future = 0
          local_next = nf_id

      # P4: jmark 特殊（入口一回）
      elif ptn == 3 and row == jmark:
        avail = avail & (~1)
        ld = ld | 1
        local_next = nf_id
        if avail == 0:
          res = 0
          if sp == 0:
            return res
          sp -= 1
          s_acc[sp] += res
          continue

      # 保存（初期化完了）
      s_ld[sp]    = ld
      s_avail[sp] = avail
      s_step[sp]  = step
      s_add1[sp]  = add1
      s_rowst[sp] = row_step
      s_useblk[sp]= use_blocks
      s_usefut[sp]= use_future
      s_lnext[sp] = local_next
      s_bK[sp]    = blockK
      s_bL[sp]    = blockL
      s_inited[sp]= 1
      # 以降、同フレームのループ処理へ

    # ---- ループ本体（avail から 1bit ずつ）----
    avail = s_avail[sp]
    if avail == 0:
      res = s_acc[sp]
      if sp == 0:
        return res
      sp -= 1
      s_acc[sp] += res
      continue

    bit = avail & -avail
    avail = avail & (avail - 1)
    s_avail[sp] = avail

    step = s_step[sp]
    add1 = s_add1[sp]
    row_step = s_rowst[sp]
    use_blocks = s_useblk[sp]
    use_future = s_usefut[sp]
    local_next = s_lnext[sp]
    blockK = s_bK[sp]
    blockL = s_bL[sp]
    ld = s_ld[sp]
    rd = s_rd[sp]
    col = s_col[sp]

    # 子状態を計算
    if use_blocks == 1:
      nld = ((ld | bit) << step) | add1 | blockL
      nrd = ((rd | bit) >> step) | blockK
      ncol = col | bit
      nf = bm & ~(nld | nrd | ncol)
      if nf == 0:
        continue
      # push child
      sp += 1
      s_func[sp]  = local_next
      s_ld[sp]    = nld
      s_rd[sp]    = nrd
      s_col[sp]   = ncol
      s_row[sp]   = row_step
      s_avail[sp] = nf
      s_jmark[sp] = s_jmark[sp-1]
      s_end[sp]   = s_end[sp-1]
      s_m1[sp]    = s_m1[sp-1]
      s_m2[sp]    = s_m2[sp-1]
      s_inited[sp]= 0
      s_acc[sp]   = 0
      continue

    # not use_future（素朴 +1）
    if use_future == 0:
      nld = (ld | bit) << 1
      nrd = (rd | bit) >> 1
      ncol = col | bit
      nf = bm & ~(nld | nrd | ncol)
      if nf == 0:
        continue
      sp += 1
      s_func[sp]  = local_next
      s_ld[sp]    = nld
      s_rd[sp]    = nrd
      s_col[sp]   = ncol
      s_row[sp]   = row_step
      s_avail[sp] = nf
      s_jmark[sp] = s_jmark[sp-1]
      s_end[sp]   = s_end[sp-1]
      s_m1[sp]    = s_m1[sp-1]
      s_m2[sp]    = s_m2[sp-1]
      s_inited[sp]= 0
      s_acc[sp]   = 0
      continue

    # use_future かつ row_step >= endmark（先読み無しで十分）
    if row_step >= s_end[sp]:
      nld = (ld | bit) << 1
      nrd = (rd | bit) >> 1
      ncol = col | bit
      nf = bm & ~(nld | nrd | ncol)
      if nf == 0:
        continue
      sp += 1
      s_func[sp]  = local_next
      s_ld[sp]    = nld
      s_rd[sp]    = nrd
      s_col[sp]   = ncol
      s_row[sp]   = row_step
      s_avail[sp] = nf
      s_jmark[sp] = s_jmark[sp-1]
      s_end[sp]   = s_end[sp-1]
      s_m1[sp]    = s_m1[sp-1]
      s_m2[sp]    = s_m2[sp-1]
      s_inited[sp]= 0
      s_acc[sp]   = 0
      continue

    # use_future 本体（1手先空きがゼロなら枝刈り）
    nld = (ld | bit) << 1
    nrd = (rd | bit) >> 1
    ncol = col | bit
    nf = bm & ~(nld | nrd | ncol)
    if nf == 0:
      continue
    if (bm & ~((nld << 1) | (nrd >> 1) | ncol)) == 0:
      continue

    sp += 1
    s_func[sp]  = local_next
    s_ld[sp]    = nld
    s_rd[sp]    = nrd
    s_col[sp]   = ncol
    s_row[sp]   = row_step
    s_avail[sp] = nf
    s_jmark[sp] = s_jmark[sp-1]
    s_end[sp]   = s_end[sp-1]
    s_m1[sp]    = s_m1[sp-1]
    s_m2[sp]    = s_m2[sp-1]
    s_inited[sp]= 0
    s_acc[sp]   = 0
    continue
# ---- dfs_iter_buf stack frame layout (GLOBAL) ----
# 12 fields / frame
F_FID   = 0
F_LD    = 1
F_RD    = 2
F_COL   = 3
F_ROW   = 4
F_AVAIL = 5
F_JMARK = 6
F_END   = 7
F_M1    = 8
F_M2    = 9
F_STATE = 10
F_ACC   = 11

# must match host allocation
FRAME_FIELDS = 12
# FRAME_MAXS はあなたの MAXS に合わせてどこかで定義済みのはず。
# 無いならひとまず固定で：
# FRAME_MAXS = 64

@inline
def IDX(tid: int, sp: int, f: int, MAXS: int, FIELDS: int) -> int:
  # stackbuf layout: [tid][sp][field]
  return tid * (MAXS * FIELDS) + sp * FIELDS + f

@inline
def META(meta: List[int], fid: int, k: int) -> int:
  # meta: fid*3 + {0:next, 1:ptn, 2:avail_flag}
  return meta[fid * 3 + k]

@inline
def BLK_K(blocks: List[int], fid: int) -> int:
  return blocks[fid * 2 + 0]

@inline
def BLK_L(blocks: List[int], fid: int) -> int:
  return blocks[fid * 2 + 1]


  # partial = dfs_iter_buf(
  #     tid,
  #     funcid_arr,
  #     ld_arr, rd_arr, col_arr,
  #     row_arr, free_arr,
  #     jmark_arr, end_arr, mark1_arr, mark2_arr,
  #     bm,
  #     next_funcid, funcptn, avail_flag,
  #     blockK_tbl, blockL_tbl,
  #     stackbuf
  # )
def dfs_iter_buf(
  tid:int,
  func0:int, ld0:int, rd0:int, col0:int, row0:int, free0:int,
  jmark0:int, endmark0:int, mark10:int, mark20:int,
  bm:int,
  meta: List[int],        # [fid*3+0]=next, [fid*3+1]=ptn, [fid*3+2]=avail_flag
  blocks: List[int],      # [fid*2+0]=blockK, [fid*2+1]=blockL
  stackbuf: List[int]
) -> int:

  # ★必ず host 側の FRAME_MAXS / FRAME_FIELDS と一致させる
  MAXS   = FRAME_MAXS
  FIELDS = FRAME_FIELDS

  sp = 0

  # 初期フレーム
  stackbuf[IDX(tid,0,F_FID,MAXS,FIELDS)]   = func0
  stackbuf[IDX(tid,0,F_LD,MAXS,FIELDS)]    = ld0
  stackbuf[IDX(tid,0,F_RD,MAXS,FIELDS)]    = rd0
  stackbuf[IDX(tid,0,F_COL,MAXS,FIELDS)]   = col0
  stackbuf[IDX(tid,0,F_ROW,MAXS,FIELDS)]   = row0
  stackbuf[IDX(tid,0,F_AVAIL,MAXS,FIELDS)] = free0
  stackbuf[IDX(tid,0,F_JMARK,MAXS,FIELDS)] = jmark0
  stackbuf[IDX(tid,0,F_END,MAXS,FIELDS)]   = endmark0
  stackbuf[IDX(tid,0,F_M1,MAXS,FIELDS)]    = mark10
  stackbuf[IDX(tid,0,F_M2,MAXS,FIELDS)]    = mark20
  stackbuf[IDX(tid,0,F_STATE,MAXS,FIELDS)] = 0
  stackbuf[IDX(tid,0,F_ACC,MAXS,FIELDS)]   = 0

  # ガード
  steps = 0
  STEP_LIMIT = 4000000

  while sp >= 0:
    steps += 1
    if steps > STEP_LIMIT:
      return -999999  # timeout

    fid   = stackbuf[IDX(tid,sp,F_FID,MAXS,FIELDS)]
    ld    = stackbuf[IDX(tid,sp,F_LD,MAXS,FIELDS)]
    rd    = stackbuf[IDX(tid,sp,F_RD,MAXS,FIELDS)]
    col   = stackbuf[IDX(tid,sp,F_COL,MAXS,FIELDS)]
    row   = stackbuf[IDX(tid,sp,F_ROW,MAXS,FIELDS)]
    avail = stackbuf[IDX(tid,sp,F_AVAIL,MAXS,FIELDS)]
    jmark = stackbuf[IDX(tid,sp,F_JMARK,MAXS,FIELDS)]
    endm  = stackbuf[IDX(tid,sp,F_END,MAXS,FIELDS)]
    m1    = stackbuf[IDX(tid,sp,F_M1,MAXS,FIELDS)]
    m2    = stackbuf[IDX(tid,sp,F_M2,MAXS,FIELDS)]
    state = stackbuf[IDX(tid,sp,F_STATE,MAXS,FIELDS)]
    acc   = stackbuf[IDX(tid,sp,F_ACC,MAXS,FIELDS)]

    # --------------------
    # init（dfs()入口処理）
    # --------------------
    if state == 0:
      avail=bm&avail
      if avail == 0:
        # pop(0)
        sp -= 1
        if sp >= 0:
          stackbuf[IDX(tid,sp,F_ACC,MAXS,FIELDS)] += 0
        continue

      base_m = fid * 3
      nextid = meta[base_m + 0]
      ptn    = meta[base_m + 1]
      aflag  = meta[base_m + 2]

      # ---- P6: 早期終了（dfs() と同じ）----
      if ptn == 5 and row == endm:
        base = 1
        if fid == 14:  # SQd2B 特例（列0以外が残っていれば1）
          base = 1 if (avail >> 1) != 0 else 0
        # pop(base)
        sp -= 1
        if sp >= 0:
          stackbuf[IDX(tid,sp,F_ACC,MAXS,FIELDS)] += base
        else:
          return base
        continue

      # 既定（+1）
      step       = 1
      add1       = 0
      row_step   = row + 1
      use_blocks = 0
      use_future = 1 if aflag == 1 else 0
      blockK     = 0
      blockL     = 0
      local_next = fid

      # P1/P2/P3: mark 行で step=2/3 + block
      if ptn == 0 or ptn == 1 or ptn == 2:
        at_mark = (row == m1) if (ptn == 0 or ptn == 2) else (row == m2)
        if at_mark and avail != 0:
          step = 2 if (ptn == 0 or ptn == 1) else 3
          add1 = 1 if (ptn == 1 and fid == 20) else 0
          row_step = row + step
          bb = fid * 2
          blockK = blocks[bb + 0]
          blockL = blocks[bb + 1]
          use_blocks = 1
          use_future = 0
          local_next = nextid

      # P4: jmark 特殊（入口一回だけ）
      elif ptn == 3 and row == jmark:
        avail &= ~1    # 列0禁止
        ld |= 1        # 左斜線LSBを立てる
        local_next = nextid
        if avail == 0:
          sp -= 1
          if sp >= 0:
            stackbuf[IDX(tid,sp,F_ACC,MAXS,FIELDS)] += 0
          continue

      # フレームへ書き戻し
      stackbuf[IDX(tid,sp,F_LD,MAXS,FIELDS)]         = ld
      stackbuf[IDX(tid,sp,F_AVAIL,MAXS,FIELDS)]      = avail
      stackbuf[IDX(tid,sp,F_STATE,MAXS,FIELDS)]      = 1
      stackbuf[IDX(tid,sp,F_STEP,MAXS,FIELDS)]       = step
      stackbuf[IDX(tid,sp,F_ADD1,MAXS,FIELDS)]       = add1
      stackbuf[IDX(tid,sp,F_ROWSTEP,MAXS,FIELDS)]    = row_step
      stackbuf[IDX(tid,sp,F_USE_BLOCKS,MAXS,FIELDS)] = use_blocks
      stackbuf[IDX(tid,sp,F_USE_FUTURE,MAXS,FIELDS)] = use_future
      stackbuf[IDX(tid,sp,F_BLOCKK,MAXS,FIELDS)]     = blockK
      stackbuf[IDX(tid,sp,F_BLOCKL,MAXS,FIELDS)]     = blockL
      stackbuf[IDX(tid,sp,F_LOCAL_NEXT,MAXS,FIELDS)] = local_next

      continue

    # --------------------
    # loop（dfs()の while avail）
    # --------------------
    avail = stackbuf[IDX(tid,sp,F_AVAIL,MAXS,FIELDS)]
    if avail == 0:
      # pop(acc)
      sp -= 1
      if sp >= 0:
        stackbuf[IDX(tid,sp,F_ACC,MAXS,FIELDS)] += acc
      else:
        return acc
      continue

    # 1bit 取り出し
    bit = avail & -avail
    avail = avail & (avail - 1)
    stackbuf[IDX(tid,sp,F_AVAIL,MAXS,FIELDS)] = avail

    step       = stackbuf[IDX(tid,sp,F_STEP,MAXS,FIELDS)]
    add1       = stackbuf[IDX(tid,sp,F_ADD1,MAXS,FIELDS)]
    row_step   = stackbuf[IDX(tid,sp,F_ROWSTEP,MAXS,FIELDS)]
    use_blocks = stackbuf[IDX(tid,sp,F_USE_BLOCKS,MAXS,FIELDS)]
    use_future = stackbuf[IDX(tid,sp,F_USE_FUTURE,MAXS,FIELDS)]
    blockK     = stackbuf[IDX(tid,sp,F_BLOCKK,MAXS,FIELDS)]
    blockL     = stackbuf[IDX(tid,sp,F_BLOCKL,MAXS,FIELDS)]
    local_next = stackbuf[IDX(tid,sp,F_LOCAL_NEXT,MAXS,FIELDS)]

    # 子状態
    if use_blocks != 0:
      nld  = ((ld | bit) << step) | add1 | blockL
      nrd  = ((rd | bit) >> step) | blockK
      ncol = (col | bit)
      nf   = bm & ~(nld | nrd | ncol)
      if nf == 0:
        continue

    elif use_future == 0:
      nld  = (ld | bit) << 1
      nrd  = (rd | bit) >> 1
      ncol = (col | bit)
      nf   = bm & ~(nld | nrd | ncol)
      if nf == 0:
        continue

    elif row_step >= endm:
      # dfs() の “row_step>=endmark は普通分岐で十分” と同じ
      nld  = (ld | bit) << 1
      nrd  = (rd | bit) >> 1
      ncol = (col | bit)
      nf   = bm & ~(nld | nrd | ncol)
      if nf == 0:
        continue

    else:
      # 先読み（ループ3B）
      nld  = (ld | bit) << 1
      nrd  = (rd | bit) >> 1
      ncol = (col | bit)
      nf   = bm & ~(nld | nrd | ncol)
      if nf == 0:
        continue
      if (bm & ~((nld << 1) | (nrd >> 1) | ncol)) == 0:
        continue

    # push child
    if sp + 1 >= MAXS:
      return -777777  # stack overflow

    sp += 1
    stackbuf[IDX(tid,sp,F_FID,MAXS,FIELDS)]   = local_next
    stackbuf[IDX(tid,sp,F_LD,MAXS,FIELDS)]    = nld
    stackbuf[IDX(tid,sp,F_RD,MAXS,FIELDS)]    = nrd
    stackbuf[IDX(tid,sp,F_COL,MAXS,FIELDS)]   = ncol
    stackbuf[IDX(tid,sp,F_ROW,MAXS,FIELDS)]   = row_step
    stackbuf[IDX(tid,sp,F_AVAIL,MAXS,FIELDS)] = nf
    stackbuf[IDX(tid,sp,F_JMARK,MAXS,FIELDS)] = jmark
    stackbuf[IDX(tid,sp,F_END,MAXS,FIELDS)]   = endm
    stackbuf[IDX(tid,sp,F_M1,MAXS,FIELDS)]    = m1
    stackbuf[IDX(tid,sp,F_M2,MAXS,FIELDS)]    = m2
    stackbuf[IDX(tid,sp,F_STATE,MAXS,FIELDS)] = 0
    stackbuf[IDX(tid,sp,F_ACC,MAXS,FIELDS)]   = 0

  return 0

# @gpu.kernel
# def solve_kernel(pack: List[int],
#                  weight_arr: List[int],
#                  out_arr: List[int],
#                  bm: int,
#                  meta: List[int],
#                  blocks: List[int],
#                  off: int,
#                  cur: int,
#                  stackbuf: List[int]) -> None:

#   gid = gpu.block.x * gpu.block.dim.x + gpu.thread.x
#   if gid >= cur:
#     return

#   i = off + gid
#   tid = gid

#   # base = i * PACK_FIELDS
#   base = gid * PACK_FIELDS
#   ld    = pack[base + 0]
#   rd    = pack[base + 1]
#   col   = pack[base + 2]
#   row   = pack[base + 3]
#   free  = pack[base + 4]
#   jmark = pack[base + 5]
#   endm  = pack[base + 6]
#   mk1   = pack[base + 7]
#   mk2   = pack[base + 8]
#   fid   = pack[base + 9]

#   partial = dfs_iter_buf(
#     tid,
#     fid,
#     ld, rd, col, row, free,
#     jmark, endm, mk1, mk2,
#     bm,
#     meta, blocks,
#     stackbuf
#   )

#   if gid == 0:
#     out_arr[i] = partial


#   # out_arr[i] = partial * weight_arr[i]
#   out_arr[i] = partial   # * weight_arr[i] を外す
#   # out_arr[i]=free
# @gpu.kernel
# def solve_kernel(pack: List[int],
#                  weight_arr: List[int],
#                  out_arr: List[int],
#                  bm: int,
#                  N1: int,
#                  NJ: int,
#                  next_funcid: List[int],
#                  funcptn: List[int],
#                  avail_flag: List[int],
#                  blockK_tbl: List[int],
#                  blockL_tbl: List[int],
#                  off: int,
#                  cur: int) -> None:

#   gid = gpu.block.x * gpu.block.dim.x + gpu.thread.x
#   if gid >= cur:
#     return

#   i = off + gid

#   # ★バッチpack前提
#   base = gid * PACK_FIELDS
#   ld    = pack[base + 0]
#   rd    = pack[base + 1]
#   col   = pack[base + 2]
#   row   = pack[base + 3]
#   free  = pack[base + 4]
#   jmark = pack[base + 5]
#   endm  = pack[base + 6]
#   mk1   = pack[base + 7]
#   mk2   = pack[base + 8]
#   fid   = pack[base + 9]

#   partial = dfs_iter(
#     fid, ld, rd, col, row, free,
#     jmark, endm, mk1, mk2,
#     bm, N1, NJ,
#     next_funcid, funcptn, avail_flag,
#     blockK_tbl, blockL_tbl
#   )

#   out_arr[i] = partial  # まず weight は外してOK
@gpu.kernel
def solve_kernel(pack: List[int],
                 weight_arr: List[int],
                 out_arr: List[int],
                 bm: int,
                 meta:List[int],
                 blocks:List[int],
                 off: int,
                 cur: int,
                 stackbuf: List[int]) -> None:

  gid = gpu.block.x * gpu.block.dim.x + gpu.thread.x
  if gid >= cur: return
  i = off + gid
  tid=gid

  # ★バッチ pack 前提
  base = gid * PACK_FIELDS
  ld    = pack[base + 0]
  rd    = pack[base + 1]
  col   = pack[base + 2]
  row   = pack[base + 3]
  free  = pack[base + 4]
  jmark = pack[base + 5]
  endm  = pack[base + 6]
  mk1   = pack[base + 7]
  mk2   = pack[base + 8]
  fid   = pack[base + 9]

  # dfs_iter 側で NJ = 1<<(N-1-jmark) を計算する実装にしておく
  # gid = gpu.block.x * gpu.block.dim.x + gpu.thread.x
  # if gid >= cur: return
  # i = off + gid
  # tid=gid

  # partial = dfs_iter(
  #   fid, ld, rd, col, row, free,
  #   jmark, endm, mk1, mk2,
  #   bm, N,
  #   next_funcid, funcptn, avail_flag,
  #   blockK_tbl, blockL_tbl
  # )

  # out_arr[i] = partial   # まず weight を外して一致確認
  # デバッグマーク：ここがホストで見えれば「launchされてる」

  partial = dfs_iter_buf(
      tid,
      fid,
      ld, rd, col,
      row, free,
      jmark, endm, mk1, mk2,
      bm,
      meta,blocks,
      stackbuf
  )
  # out_arr[i]=fid
  # out_arr[i]=free
  out_arr[i]=(fid<<16)|(free&0xFFFF)
  # out_arr[i] = partial 
  # out_arr[i] = partial * weight_arr[i]   # ← weight も i
  return 



class NQueens17:
  def __init__(self)->None:
    self.zobrist_tables:Dict[int,Dict[str,List[int]]]={}
    self.jasmin_cache:Dict[Tuple[int,int],int]={}
    # サブコンステレーション生成状態のメモ化（実行中の重複再帰を抑制）
    self.subconst_cache:Set[StateKey]=set()
    # …既存初期化…
    self._N=0
    self._N1=0
    self._NK=0
    self._NJ=0
    self._board_mask=0
    # 配列系（空で作っておく）
    self._blockK:list[int]=[]
    self._blockL:list[int]=[]
    # (next_id, funcptn, avail_flag)
    self._meta:list[Tuple[int,int,int]]=[]

  def mix64(self,x:int)->int:
    """splitmix64 のミキサ最終段。64bit 値 x を (>>/XOR/乗算) の 3 段で拡散して返す。 Zobrist テーブルの乱数品質を担保するために使用。"""
    # MASK64:int=(1<<64)-1 # 64bit マスク（Zobrist用途）
    x&=MASK64
    x=(x^(x>>30))*0xBF58476D1CE4E5B9&MASK64
    x=(x^(x>>27))*0x94D049BB133111EB&MASK64
    x^=(x>>31)
    return x&MASK64

  def gen_list(self,cnt:int,seed:int)->List[int]:
    """Zobrist テーブル用の 64bit 乱数を cnt 個生成してリストで返す。 seed は splitmix64 のインクリメント規約 (0x9E3779B97F4A7C15) に従って更新。"""
    # MASK64:int=(1<<64)-1 # 64bit マスク（Zobrist用途）
    out:List[int]=[]
    s:int=seed&MASK64
    _mix64=self.mix64
    for _ in range(cnt):
      s=(s+0x9E3779B97F4A7C15)&MASK64   # splitmix64 のインクリメント
      # out.append(self._mix64(s))
      out.append(_mix64(s))
    return out

  def init_zobrist(self,N:int)->None:
    """Zobrist テーブルを N ごとに初期化する。キーは 'ld','rd','col','LD','RD','row','queens','k','l'。 ※ キャッシュは self.zobrist_tables[N] に格納して再利用する。"""
    # MASK64:int=(1<<64)-1 # 64bit マスク（Zobrist用途）
    if N in self.zobrist_tables:
      return
    base_seed:int=(0xC0D0_0000_0000_0000^(N<<32))&MASK64
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

  def zobrist_hash(self,ld:int,rd:int,col:int,row:int,queens:int,k:int,l:int,LD:int,RD:int,N:int)->int:
    """(ld, rd, col, LD, RD, row, queens, k, l) を Zobrist Hash によって 64bit にまとめる。 各マスクは N ビットに正規化してから XOR 混合する。衝突確率の低い探索済み判定に利用可能。"""
    # MASK64:int=(1<<64)-1

    # 1) テーブルを（必要なら）構築
    self.init_zobrist(N)
    # 2) 構築済みキャッシュを参照
    tbl=self.zobrist_tables[N]

    # zobrist_tables:Dict[int,Dict[str,List[int]]]={}
    # tbl=self.init_zobrist[N]

    # 3) ループ内で tbl['ld'][i] などの辞書アクセスを都度行うと遅いので、先にローカル束縛にすると微速化します：
    ld_tbl,rd_tbl,col_tbl=tbl['ld'],tbl['rd'],tbl['col']
    LD_tbl,RD_tbl=tbl['LD'],tbl['RD']
    row_tbl,q_tbl,k_tbl,l_tbl=tbl['row'],tbl['queens'],tbl['k'],tbl['l']

    # 4) N ビットへ正規化（上位ビットや負数を落とす）
    mask=(1<<N)-1
    ld&=mask
    rd&=mask
    col&=mask
    LD&=mask
    RD&=mask

    # 5) 各ビットを見て XOR
    h=0
    m=ld;i=0
    while i<N:
      if (m&1)!=0:
        # h^=tbl['ld'][i]
        h^=ld_tbl[i]
      m>>=1;i+=1
    m=rd;i=0
    while i<N:
      if (m&1)!=0:
        # h^=tbl['rd'][i]
        h^=rd_tbl[i]
      m>>=1;i+=1
    m=col;i=0
    while i<N:
      if (m&1)!=0:
        # h^=tbl['col'][i]
        h^=col_tbl[i]
      m>>=1;i+=1
    m=LD;i=0
    while i<N:
      if (m&1)!=0:
        # h^=tbl['LD'][i]
        h^=LD_tbl[i]
      m>>=1;i+=1
    m=RD;i=0
    while i<N:
      if (m&1)!=0:
        # h^=tbl['RD'][i]
        h^=RD_tbl[i]
      m>>=1;i+=1

    # 行数・個数・強制行などスカラー要素
    if 0<=row<N:h^=row_tbl[row]
    if 0<=queens<N:h^=q_tbl[queens]
    if 0<=k<N:h^=k_tbl[k]
    if 0<=l<N:h^=l_tbl[l]

    return h&MASK64

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

  def dfs(self,functionid:int,ld:int,rd:int,col:int,row:int,free:int,
          jmark:int,endmark:int,mark1:int,mark2:int)->int:
    """汎用 DFS カーネル。古い SQ???? 関数群を 1 本化し、func_meta の記述に従って
    (1) mark 行での step=2/3 + 追加ブロック、(2) jmark 特殊、(3) ゴール判定、(4) +1 先読み
    を切り替える。引数:
    functionid: 現在の分岐モード ID（次の ID, パターン, 先読み有無は func_meta で決定）
    ld/rd/col:   斜線/列の占有
    row/free:    現在行と空きビット
    jmark/endmark/mark1/mark2: 特殊行/探索終端
    board_mask:  盤面全域のビットマスク
    blockK_by_funcid/blockl_by_funcid: 関数 ID に紐づく追加ブロック
    func_meta:   (next_id, funcptn, availptn) のメタ情報配列
    """    
    # ---- ローカル束縛（属性アクセス最小化）----
    # _dfs = self.dfs
    # meta = self._meta
    # blockK_tbl = self._blockK
    # blockL_tbl = self._blockL
    bm:int=self._board_mask
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
    use_blocks:bool=False  # blockK/blockl を噛ませるか
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
        nf:int=bm&~(nld|nrd|ncol)
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
        nf:int=bm&~(nld|nrd|ncol)
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
        nf:int=bm&~(nld|nrd|ncol)
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
      nf:int=bm&~(nld|nrd|ncol)
      if not nf:
        continue
      # 1手先の空きをその場で素早くチェック（余計な再帰を抑止）
      #   next_free_next = bm & ~(((nld << 1) | (nrd >> 1) | ncol))
      #   if next_free_next == 0: continue
      if bm&~((nld<<1)|(nrd>>1)|ncol):
        # total += _dfs(local_next_funcid, nld, nrd, ncol, row_step, nf, jmark, endmark, mark1, mark2)
        total+=self.dfs(local_next_funcid,nld,nrd,ncol,row_step,nf,jmark,endmark,mark1,mark2)
    return total
 

  def exec_solutions_gpu(self,constellations:List[Dict[str,int]],N:int)->None:
      """
      既存 exec_solutions の「前処理(SoA)」をそのまま使い、
      GPUで dfs_iter を回して out[i] を作り、solutionsへ書き戻す（atomic不要）。
      """
      N2:int=N-2
      small_mask:int=(1<<N2)-1
      board_mask:int=(1<<N)-1
      symmetry=self.symmetry
      getj,getk,getl=self.getj,self.getk,self.getl

      FUNC_CATEGORY={
        "SQBkBlBjrB":3,"SQBlkBjrB":3,"SQBkBjrB":3,
        "SQd2BkBlB":3,"SQd2BkB":3,"SQd2BlkB":3,
        "SQd1BkBlB":3,"SQd1BlkB":3,"SQd1BkB":3,"SQd0BkB":3,
        "SQBklBjrB":4,"SQd2BklB":4,"SQd1BklB":4,
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

      func_meta=[
        (1,0,0),(2,1,0),(3,3,1),(3,5,1),
        (2,2,0),(6,0,0),(2,1,0),(2,2,0),
        (0,4,1),(4,4,1),(5,4,1),(7,4,1),
        (13,0,0),(14,1,0),(14,5,1),(14,2,0),
        (17,0,0),(14,1,0),(14,2,0),
        (20,0,0),(21,1,0),(21,5,1),(21,2,0),
        (25,0,0),(21,2,0),(21,1,0),
        (26,5,1),(26,0,0),
      ]

      n3=1<<max(0,N-3)
      n4=1<<max(0,N-4)
      size=max(FID.values())+1
      blockK_by_funcid=[0]*size
      blockL_by_funcid=[0,1,0,0,1,1,0,2,0,0,0,0,0,1,0,1,1,0,2,0,0,0,1,1,2,0,0,0]
      for fn,cat in FUNC_CATEGORY.items():
        fid=FID[fn]
        blockK_by_funcid[fid]=n3 if cat==3 else (n4 if cat==4 else 0)

      # meta を3本に分解（GPUに渡しやすく）
      next_funcid=[0]*size
      funcptn=[0]*size
      avail_flag=[0]*size
      for i,(nx,pt,af) in enumerate(func_meta):
        next_funcid[i]=nx
        funcptn[i]=pt
        avail_flag[i]=af

      # ---- SoA 前処理（あなたの exec_solutions のまま）----
      m=len(constellations)
      ld_arr=[0]*m;rd_arr=[0]*m;col_arr=[0]*m
      row_arr=[0]*m;free_arr=[0]*m
      jmark_arr=[0]*m;end_arr=[0]*m
      mark1_arr=[0]*m;mark2_arr=[0]*m
      funcid_arr=[0]*m
      weight_arr=[0]*m
      out_arr=[0]*m

      N1=N-1
      for i,constellation in enumerate(constellations):
        jmark=mark1=mark2=0
        start_ijkl=constellation["startijkl"]
        start=start_ijkl>>20
        ijkl=start_ijkl&((1<<20)-1)
        weight_arr[i]=symmetry(ijkl,N)

        j,k,l=getj(ijkl),getk(ijkl),getl(ijkl)
        ld,rd,col=constellation["ld"]>>1,constellation["rd"]>>1,(constellation["col"]>>1)|(~small_mask)
        LD=(1<<(N1-j))|(1<<(N1-l))
        ld|=LD>>(N-start)
        if start>k:rd|=(1<<(N1-(start-k+1)))
        if j>=2*N-33-start:rd|=(1<<(N1-j))<<(N2-start)
        # free=~(ld|rd|col)
        free = board_mask & ~(ld | rd | col)


        # ---- target 決定（あなたの元ロジックをそのまま）----
        target:int=0
        N1_,N2_=N-1,N-2
        j_lt_N3=(j<N-3)
        j_eq_N3=(j==N-3)
        j_eq_N2=(j==N-2)
        k_lt_l=(k<l)
        start_lt_k=(start<k)
        start_lt_l=(start<l)
        l_eq_kp1=(l==k+1)
        k_eq_lp1=(k==l+1)
        j_gate=(j>2*N-34-start)

        if j_lt_N3:
          jmark=j+1
          endmark=N2_
          if j_gate:
            if k_lt_l:
              mark1,mark2=k-1,l-1
              if start_lt_l:
                if start_lt_k:target=0 if (not l_eq_kp1) else 4
                else:target=1
              else:target=2
            else:
              mark1,mark2=l-1,k-1
              if start_lt_k:
                if start_lt_l:target=5 if (not k_eq_lp1) else 7
                else:target=6
              else:target=2
          else:
            if k_lt_l:
              mark1,mark2=k-1,l-1
              target=8 if (not l_eq_kp1) else 9
            else:
              mark1,mark2=l-1,k-1
              target=10 if (not k_eq_lp1) else 11

        elif j_eq_N3:
          endmark=N2_
          if k_lt_l:
            mark1,mark2=k-1,l-1
            if start_lt_l:
              if start_lt_k:target=12 if (not l_eq_kp1) else 15
              else:
                mark2=l-1
                target=13
            else:target=14
          else:
            mark1,mark2=l-1,k-1
            if start_lt_k:
              if start_lt_l:target=16 if (not k_eq_lp1) else 18
              else:
                mark2=k-1
                target=17
            else:target=14

        elif j_eq_N2:
          if k_lt_l:
            endmark=N2_
            if start_lt_l:
              if start_lt_k:
                mark1=k-1
                if not l_eq_kp1:
                  mark2=l-1
                  target=19
                else:target=22
              else:
                mark2=l-1
                target=20
            else:target=21
          else:
            if start_lt_k:
              if start_lt_l:
                if k<N2_:
                  mark1,endmark=l-1,N2_
                  if not k_eq_lp1:
                    mark2=k-1
                    target=23
                  else:target=24
                else:
                  if l!=(N-3):
                    mark2,endmark=l-1,N-3
                    target=20
                  else:
                    endmark=N-4
                    target=21
              else:
                if k!=N2_:
                  mark2,endmark=k-1,N2_
                  target=25
                else:
                  endmark=N-3
                  target=21
            else:
              endmark=N2_
              target=21
        else:
          endmark=N2_
          if start>k:target=26
          else:
            mark1=k-1
            target=27

        ld_arr[i]=ld;rd_arr[i]=rd;col_arr[i]=col
        row_arr[i]=start;free_arr[i]=free
        jmark_arr[i]=jmark;end_arr[i]=endmark
        mark1_arr[i]=mark1;mark2_arr[i]=mark2
        funcid_arr[i]=target

      mask20 = (1 << 20) - 1
      print("[dbg] startijkl head:", [c["startijkl"] for c in constellations[:8]])
      print("[dbg] ijkl head:", [(c["startijkl"] & mask20) for c in constellations[:8]])
      print("[dbg] row head:",  [(c["startijkl"] >> 20) for c in constellations[:8]])

      # ---- GPU 実行（atomicなし：out_arr[i] に書く）----
      # 事前に一度だけ作る（while の外）
      # FIELDS = 10         # ld,rd,col,row,free,jmark,end,mark1,mark2,funcid
      # META_FIELDS = 3     # next, ptn, avail
      # BLOCK_FIELDS = 2    # blockK, blockL
      PACK_FIELDS  = 10        # ld,rd,col,row,free,jmark,end,mark1,mark2,funcid
      META_FIELDS  = 3         # next, ptn, avail
      BLOCK_FIELDS = 2         # blockK, blockL
      # FRAME_FIELDS = 12        # dfs_iter_buf のスタックフレーム
      # FRAME_MAXS   = MAXS      # 既に引数 MAXS を使ってるならそれでOK

      # --- pack を作る（m件） ---
      # pack = [0] * (m * PACK_FIELDS)
      # for i in range(m):
      #   b = i * PACK_FIELDS
      #   pack[b+0] = ld_arr[i]
      #   pack[b+1] = rd_arr[i]
      #   pack[b+2] = col_arr[i]
      #   pack[b+3] = row_arr[i]
      #   pack[b+4] = free_arr[i]
      #   pack[b+5] = jmark_arr[i]
      #   pack[b+6] = end_arr[i]
      #   pack[b+7] = mark1_arr[i]
      #   pack[b+8] = mark2_arr[i]
      #   pack[b+9] = funcid_arr[i]

      # --- meta を作る（fid_max件） ---
      fid_max = len(next_funcid)
      meta = [0] * (fid_max * META_FIELDS)
      for fid in range(fid_max):
        meta[fid*3+0] = next_funcid[fid]
        meta[fid*3+1] = funcptn[fid]
        meta[fid*3+2] = avail_flag[fid]

      # --- blocks を作る（fid_max件） ---
      blocks = [0] * (fid_max * BLOCK_FIELDS)
      for fid in range(fid_max):
        blocks[fid*2+0] = blockK_by_funcid[fid]
        blocks[fid*2+1] = blockL_by_funcid[fid]

      # # --- バッチ実行 ---
      # BATCH = 1024
      # out_arr = [0] * m
      # BLOCK=32
      # # ★超重要：stackbuf は「BATCHスレッド分」なので tid は 0..BATCH-1 にする
      # # stackbuf = [0] * (BATCH * MAXS * FIELDS)
      # stackbuf = [0] * (BATCH * FRAME_MAXS * FRAME_FIELDS)


      # off = 0
      # while off < m:
      #   cur = BATCH if (m - off) > BATCH else (m - off)

      #   # BLOCK は 32/64/128 などでOK、ただし GRID*BLOCK が cur を覆う必要あり
      #   GRID = (cur + BLOCK - 1) // BLOCK
      #   print(f"[dbg] launch batch: off={off} cur={cur} GRID={GRID} BLOCK={BLOCK}")

      #   # ここは不要なら消してOK（デバッグ向け）
      #   for t in range(off, off + cur):
      #     out_arr[t] = 0

      #   solve_kernel(
      #     pack,
      #     weight_arr,
      #     out_arr,
      #     board_mask,
      #     meta,
      #     blocks,
      #     off, 
      #     cur,
      #     stackbuf,
      #     grid=GRID, 
      #     block=BLOCK
      #   )

      #   print("[dbg] out_arr head:", out_arr[off:off+cur])
      #   off += cur
      # print("[dbg] gpu batches done")
      # --- バッチ実行 ---
      BLOCK = 32
      BATCH = 1024
      out_arr = [0] * m
      stackbuf = [0] * (BATCH * FRAME_MAXS * FRAME_FIELDS)

      off = 0
      while off < m:
        cur = BATCH if (m - off) > BATCH else (m - off)
        GRID = (cur + BLOCK - 1) // BLOCK
        print(f"[dbg] launch batch: off={off} cur={cur} GRID={GRID} BLOCK={BLOCK}")

        # --- pack を cur 件だけ作る（★重要） ---
        pack = [0] * (cur * PACK_FIELDS)
        for j in range(cur):
          i = off + j
          b = j * PACK_FIELDS
          pack[b+0] = ld_arr[i]
          pack[b+1] = rd_arr[i]
          pack[b+2] = col_arr[i]
          pack[b+3] = row_arr[i]
          pack[b+4] = free_arr[i]
          pack[b+5] = jmark_arr[i]
          pack[b+6] = end_arr[i]
          pack[b+7] = mark1_arr[i]
          pack[b+8] = mark2_arr[i]
          pack[b+9] = funcid_arr[i]

        # デバッグ：この範囲だけクリア
        for t in range(off, off + cur):
          out_arr[t] = 0

        print("[dbg] pack fid head:", [pack[j*PACK_FIELDS + 9] for j in range(min(cur,8))])
        print("[dbg] pack free head:", [pack[j*PACK_FIELDS + 4] for j in range(min(cur,8))])

        solve_kernel(
          pack,
          weight_arr,
          out_arr,
          board_mask,
          meta,
          blocks,
          off,
          cur,
          stackbuf,
          grid=GRID,
          block=BLOCK
        )
        print("[dbg] out_arr head:", [hex(x) for x in out_arr[off:off+8]])
        print("[dbg] free head:", free_arr[off:off+min(cur,8)])
        print("[dbg] fid head:", funcid_arr[off:off+min(cur,8)])
        print("[dbg] w head:", weight_arr[off:off+min(cur,8)])

        print("[dbg] out_arr head:", out_arr[off:off+cur])
        off += cur
        print("[dbg] gpu batches done")

      # ---- 書き戻し ----
      for i,c in enumerate(constellations):
        c["solutions"]=out_arr[i]

  def exec_solutions(self,constellations:List[Dict[str,int]],N:int)->None:
    """各 Constellation（部分盤面）ごとに最適分岐（functionid）を選び、`dfs()` で解数を取得。 結果は `solutions` に書き込み、最後に `symmetry()` の重みで補正する。前段で SoA 展開し 並列化区間のループ体を軽量化。"""
    N2:int=N-2
    small_mask:int=(1<<N2)-1
    board_mask:int=(1<<N)-1
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
    n3=1<<max(0,N-3)   # 念のため負シフト防止
    n4=1<<max(0,N-4)
    size=max(FID.values())+1
    blockK_by_funcid=[0]*size
    blockl_by_funcid=[0,1,0,0,1,1,0,2,0,0,0,0,0,1,0,1,1,0,2,0,0,0,1,1,2,0,0,0]
    for fn,cat in FUNC_CATEGORY.items():# FUNC_CATEGORY: {関数名: 3 or 4 or 0}
      fid=FID[fn]
      blockK_by_funcid[fid]=n3 if cat==3 else (n4 if cat==4 else 0)
    # ===== 前処理ステージ（単一スレッド） =====
    m=len(constellations)
    # SoA（Structure of Arrays）に展開：並列本体が軽くなる
    ld_arr=[0]*m;rd_arr=[0]*m;col_arr=[0]*m
    row_arr=[0]*m;free_arr=[0]*m
    jmark_arr=[0]*m;end_arr=[0]*m
    mark1_arr=[0]*m;mark2_arr=[0]*m
    funcid_arr=[0]*m
    ijkl_arr=[0]*m   # symmetry 計算用
    N1=N-1
    NK=1<<(N-3)
    NJ=1<<N1
    results=[0]*m
    target:int=0
    for i,constellation in enumerate(constellations):
      jmark=mark1=mark2=0
      start_ijkl=constellation["startijkl"]
      start=start_ijkl>>20
      ijkl=start_ijkl&((1<<20)-1)
      j,k,l=getj(ijkl),getk(ijkl),getl(ijkl)
      ld,rd,col=constellation["ld"]>>1,constellation["rd"]>>1,(constellation["col"]>>1)|(~small_mask)
      LD=(1<<(N1-j))|(1<<(N1-l))
      ld|=LD>>(N-start)
      if start>k:rd|=(1<<(N1-(start-k+1)))
      if j>=2*N-33-start:rd|=(1<<(N1-j))<<(N2-start)
      # free=~(ld|rd|col)
      free = board_mask & ~(ld | rd | col)


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
      ld_arr[i],rd_arr[i],col_arr[i],row_arr[i],free_arr[i],jmark_arr[i],end_arr[i],mark1_arr[i],mark2_arr[i],funcid_arr[i],ijkl_arr[i]=ld,rd,col,start,free,jmark,endmark,mark1,mark2,target,ijkl

    # ===== 並列ステージ：計算だけ =====
    self._N=N
    self._N1=N-1
    self._NK=1<<(N-3)
    self._NJ=1<<(N-1)
    self._blockK=blockK_by_funcid
    self._blockL=blockl_by_funcid
    self._meta=func_meta
    self._board_mask=(1<<N)-1
    @par
    for i in range(m):
      #print("exec_start")
      cnt=dfs(funcid_arr[i],ld_arr[i],rd_arr[i],col_arr[i],row_arr[i],free_arr[i],jmark_arr[i],end_arr[i],mark1_arr[i],mark2_arr[i])
      results[i]=cnt*symmetry(ijkl_arr[i],N)

    # ===== 書き戻し（単一スレッド） =====
    # constellation["solutions"]=cnt*symmetry(ijkl,N)
    for i,constellation in enumerate(constellations):
      constellation["solutions"]=results[i]

  # def set_pre_queens_cached(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:List[int],constellations:List[Dict[str,int]],N:int,preset_queens:int,visited:Set[int],constellation_signatures:Set[Tuple[int,int,int,int,int,int]])->None:
  def set_pre_queens_cached(
    self,
    ld:int, rd:int, col:int,
    k:int, l:int,
    row:int, queens:int,
    LD:int, RD:int,
    counter:List[int],
    constellations:List[Dict[str,int]],
    N:int, preset_queens:int,
    visited:Set[int],
    constellation_signatures:Set[Tuple[int,int,int,int,int,int,int]],  # ←後述：型も7要素に
    ijkl:int                                     # ★追加
  ) -> None:

    """サブコンステレーション生成のキャッシュ付ラッパ。StateKey で一意化し、 同一状態での重複再帰を回避して生成量を抑制する。"""
    key:StateKey=(ld,rd,col,k,l,row,queens,LD,RD,N,preset_queens)

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
    self.set_pre_queens(ld,rd,col,k,l,row,queens,LD,RD,counter,constellations,N,preset_queens,visited,constellation_signatures,ijkl)
    # subconst_cache[key] = True  # マークだけでOK
    # subconst_cache.add(key)

  def set_pre_queens(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:list,constellations:List[Dict[str,int]],N:int,preset_queens:int,visited:Set[int],constellation_signatures:Set[Tuple[int,int,int,int,int,int,int]],ijkl:int)->None:
    """事前に置く行 (k,l) を強制しつつ、queens==preset_queens に到達するまで再帰列挙。 `visited` には軽量な `state_hash` を入れて枝刈り。到達時は {ld,rd,col,startijkl} を constellation に追加。"""
    mask=(1<<N)-1  # setPreQueensで使用
    # ---------------------------------------------------------------------
    # 状態ハッシュによる探索枝の枝刈り バックトラック系の冒頭に追加　やりすぎると解が合わない
    #
    # <>zobrist_hash
    # 各ビットを見てテーブルから XOR するため O(N)（ld/rd/col/LD/RDそれぞれで最大 N 回）。
    # とはいえ N≤17 なのでコストは小さめ。衝突耐性は高い。
    # マスク漏れや負数の扱いを誤ると不一致が起きる点に注意（先ほどの & ((1<<N)-1) 修正で解決）。
    # h: int = self.zobrist_hash(ld, rd, col, row, queens, k, l, LD, RD, N)
    #
    # <>state_hash
    # その場で数個の ^ と << を混ぜるだけの O(1) 計算。
    # 生成されるキーも 単一の int なので、set/dict の操作が最速＆省メモリ。
    # ただし理論上は衝突し得ます（実際はN≤17の範囲なら実害が出にくい設計にしていればOK）。
    # [Opt-09] Zobrist Hash（Opt-09）の導入とその用途
    # ビットボード設計でも、「盤面のハッシュ」→「探索済みフラグ」で枝刈りは可能です。
    h:int=(ld<<3)^(rd<<2)^(col<<1)^row^(queens<<7)^(k<<12)^(l<<17)^(LD<<22)^(RD<<27)^(N<<1)
    #
    # <>StateKey（タプル）
    # 11個の整数オブジェクトを束ねるため、オブジェクト生成・GC負荷・ハッシュ合成が最も重い。
    # set の比較・保持も重く、メモリも一番食います。
    # 衝突はほぼ心配ないものの、速度とメモリ効率は最下位。
    # h:StateKey = (ld, rd, col, row, queens, k, l, LD, RD)
    if h in visited:
      return
    visited.add(h)
    #
    # ---------------------------------------------------------------------
    # k行とl行はスキップ
    if row==k or row==l:
      self.set_pre_queens_cached(ld<<1,rd>>1,col,k,l,row+1,queens,LD,RD,counter,constellations,N,preset_queens,visited,constellation_signatures,ijkl)
      return
    # クイーンの数がpreset_queensに達した場合、現在の状態を保存
    if queens==preset_queens:
      # signatureの生成
      signature=(ld,rd,col,k,l,row,ijkl)  # 必要な変数でOK
      if not hasattr(self,"constellation_signatures"):
        constellation_signatures=set()
      signatures=constellation_signatures
      if signature not in signatures:
        constellation={"ld":ld,"rd":rd,"col":col,"startijkl":row<<20|ijkl,"solutions":0}
        constellations.append(constellation) #星座データ追加
        signatures.add(signature)
        counter[0]+=1
      return
    # 現在の行にクイーンを配置できる位置を計算
    free=~(ld|rd|col|(LD>>(N-1-row))|(RD<<(N-1-row)))&mask
    _set_pre_queens_cached=self.set_pre_queens_cached
    while free:
      bit:int=free&-free
      free&=free-1
      _set_pre_queens_cached((ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,counter,constellations,N,preset_queens,visited,constellation_signatures,ijkl)


  def gen_constellations(self,ijkl_list:Set[int],constellations:List[Dict[str,int]],N:int,preset_queens:int)->None:
    """開始コンステレーション（代表部分盤面）の列挙。中央列（奇数 N）特例、回転重複排除 （`check_rotations`）、Jasmin 正規化（`get_jasmin`）を経て、各 sc から `set_pre_queens_cached` でサブ構成を作る。"""
    # 実行ごとにメモ化をリセット（N や preset_queens が変わるとキーも変わるが、
    # 長時間プロセスなら念のためクリアしておくのが安全）
    self.subconst_cache.clear()

    halfN=(N+1)//2  # Nの半分を切り上げ
    N1:int=N-1
    constellation_signatures: Set[ Tuple[int, int, int, int, int, int,int] ] = set()
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
      # _set_pre_queens_cached(ld,rd,col,k,l,1,3 if j==N1 else 4,LD,RD,counter,constellations,N,preset_queens,visited,constellation_signatures)
        # target などを計算しているはず
      _set_pre_queens_cached(
        ld, rd, col, k, l, 1, 3 if j==N1 else 4, LD, RD,
        counter, constellations, N, preset_queens, visited, constellation_signatures,
        sc          # ★追加
      )
      current_size=len(constellations)
      base=to_ijkl(i,j,k,l)
      for a in range(counter[0]):
        constellations[-1-a]["startijkl"]|=base

  """constellations の各要素が {ld, rd, col, startijkl} を全て持つかを検証する。"""
  def validate_constellation_list(self,constellations:List[Dict[str,int]])->bool: return all(all(k in c for k in ("ld","rd","col","startijkl")) for c in constellations)

  """32bit little-endian の相互変換ヘルパ。Codon/CPython の差異に注意。"""
  def read_uint32_le(self,b:str)->int: return (ord(b[0])&0xFF)|((ord(b[1])&0xFF)<<8)|((ord(b[2])&0xFF)<<16)|((ord(b[3])&0xFF)<<24)

  def int_to_le_bytes(self,x:int)->List[int]: return [(x>>(8*i))&0xFF for i in range(4)]

  def file_exists(self,fname:str)->bool:
    """ファイル存在チェック（読み取り open の可否で判定）。"""
    try:
      with open(fname,"rb"):
        return True
    except:
      return False

  def validate_bin_file(self,fname:str)->bool:
    """bin キャッシュのサイズ妥当性確認（1 レコード 16 バイトの整数倍か）。"""
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

  def load_or_build_constellations_bin(self,ijkl_list:Set[int],constellations,N:int,preset_queens:int)->List[Dict[str,int]]:
    """bin キャッシュを読み込み。検証に失敗した場合は再生成して保存し、その結果を返す。"""
    fname=f"constellations_N{N}_{preset_queens}.bin"
    if self.file_exists(fname):
      constellations=self.load_constellations_bin(fname)
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

class NQueens17_constellations():
  # 小さなNは正攻法で数える（対称重みなし・全列挙）
  def _bit_total(self,size:int)->int:
    """小さな N 用の素朴な全列挙（対称重みなし）。ビットボードで列/斜線の占有を管理して再帰的に合計を返す。検算/フォールバック用。"""
    mask=(1<<size)-1
    total=0

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

  def main(self)->None:
    """N=5..17 の合計解を計測。N<=5 は `_bit_total()` のフォールバック、それ以外は星座キャッシュ（.bin/.txt）→ `exec_solutions()` → 合計→既知解 `expected` と照合。"""
    nmin:int=5
    nmax:int=9
    preset_queens:int=4  # 必要に応じて変更
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      if size<=5:
        # ← フォールバック：N=5はここで正しい10を得る
        total=self._bit_total(size)
        dt=datetime.now()-start_time
        text=str(dt)[:-3]
        print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
        continue
      ijkl_list:Set[int]=set()
      constellations:List[Dict[str,int]]=[]
      # NQ=NQueens17(size)
      NQ=NQueens17()
      #---------------------------------
      # 星座リストそのものをキャッシュ
      #---------------------------------
      # キャッシュを使わない
      # NQ.gen_constellations(ijkl_list,constellations,size,preset_queens)
      # キャッシュを使う、キャッシュの整合性もチェック
      # -- txt
      # constellations = NQ.load_or_build_constellations_txt(ijkl_list,constellations, size, preset_queens)
      # -- bin
      constellations = NQ.load_or_build_constellations_bin(ijkl_list,constellations, size, preset_queens)
      #---------------------------------
      # CPU
      # NQ.exec_solutions(constellations,size)
      
      # GPU
      NQ.gen_constellations(ijkl_list,constellations,size,preset_queens)
      print(f"[dbg] size={size} gen_constellations done: ijkl={len(ijkl_list)} consts={len(constellations)}")
      print(f"[dbg] size={size} start exec_solutions_gpu")
      NQ.exec_solutions_gpu(constellations,size)
      print(f"[dbg] size={size} exec_solutions_gpu done")

      total:int=sum(c['solutions'] for c in constellations if c['solutions']>0)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      # print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
      expected:List[int]=[0,0,0,0,0,10,4,40,92,352,724,2680,14200,73712,365596,2279184,14772512,95815104,666090624,4968057848]
      status:str="ok" if expected[size]==total else f"ng({total}!={expected[size]})"
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}    {status}")

if __name__=="__main__":
  NQueens17_constellations().main()
