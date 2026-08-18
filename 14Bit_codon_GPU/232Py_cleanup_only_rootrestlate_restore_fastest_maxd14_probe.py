#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# NQ_UPDATE_MEMO
# 232: cleanup-only。231Pyで217Py相当のrootrestlate最速系へ復帰し、N=21 full once正当性OKを確認したため、kernel探索ロジックは変更せず、巨大な過去ログ・長いdocstring・コメント履歴をREADMEへ退避する。runtime tagはsplit232へ更新し、futuremask、no-sibling、root one/two preroll、host task order、MAXD14 208 bytes/threadを維持する。
# Full update history: see README.md
# =============================================================================

"""
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

Python/Codon CUDA N-Queens constellation solver.

231Py note:
  230Py was correct but slower, so this restore branch returns to the validated
  217Py rootrestlate fastest baseline.  It removes the 230 root-preroll
  placement-common scalars and backs out the 225 generic ncol-early ordering.
  The futuremask guard, no-sibling stack behavior, root one/two-candidate
  preroll, host task order, and MAXD14 208-byte/thread shape are unchanged.

Full update history is maintained in README.md.
"""

"""
Python/codon Ｎクイーン コンステレーション版 CUDA 高速ソルバ

# ビルド
codon build -release 115Py_range_default_clean_cg_v2.py

# CPU実行
stdbuf -oL -eL ./115Py_range_default_clean_cg_v2 -c 2>&1 | tee 115Py_cpu_range_$(date +%Y%m%d_%H%M%S).log

# GPU実行
stdbuf -oL -eL ./115Py_range_default_clean_cg_v2 -c 2>&1 | tee 115Py_cpu_range_$(date +%Y%m%d_%H%M%S).log

# CPU実行結果
workspace#suzuki$ date
2026年  6月  9日 火曜日 14:23:02 JST
workspace#suzuki$ stdbuf -oL -eL ./115Py_range_default_clean_cg_v2 -c 2>&1 | tee 115Py_cpu_range.log
CPU mode selected
 N:             Total           Unique         hh:mm:ss.ms
 5:                10                0          0:00:00.000
 6:                 4                0          0:00:00.105    ok
 7:                40                0          0:00:00.000    ok
 8:                92                0          0:00:00.000    ok
 9:               352                0          0:00:00.015    ok
10:               724                0          0:00:00.011    ok
11:              2680                0          0:00:00.009    ok
12:             14200                0          0:00:00.020    ok
13:             73712                0          0:00:00.042    ok
14:            365596                0          0:00:00.091    ok
15:           2279184                0          0:00:00.173    ok
16:          14772512                0          0:00:00.270    ok
17:          95815104                0          0:00:00.412    ok
18:         666090624                0          0:00:04.433    ok
19:        4968057848                0          0:00:17.103    ok
20:       39029188884                0          0:02:11.042    ok
21:      314666222712                0          0:18:07.042    ok
22:     2691008701644                0          2:38:58.023    ok

# GPU実行結果
suzuki@cudacodon$ date
2026年  6月  9日 火曜日 05:55:00 UTC
suzuki@cudacodon$ stdbuf -oL -eL ./115Py_range_default_clean_cg_v2 -g 2>&1 | tee 115Py_cpu_range_$(date +%Y%m%d_%H%M%S).log
GPU mode selected
 N:             Total           Unique         hh:mm:ss.ms
 5:                10                0          0:00:00.000
 6:                 4                0          0:00:00.004    ok
 7:                40                0          0:00:00.004    ok
 8:                92                0          0:00:00.002    ok
 9:               352                0          0:00:00.002    ok
10:               724                0          0:00:00.003    ok
11:              2680                0          0:00:00.005    ok
12:             14200                0          0:00:00.007    ok
13:             73712                0          0:00:00.011    ok
14:            365596                0          0:00:00.018    ok
15:           2279184                0          0:00:00.036    ok
16:          14772512                0          0:00:00.104    ok
17:          95815104                0          0:00:00.465    ok
18:         666090624                0          0:00:03.475    ok
19:        4968057848                0          0:00:22.443    ok
20:       39029188884                0          0:03:04.146    ok
21:      314666222712                0          0:23:34.869    ok
22:     2691008701644                0          3:37:51.255    ok
23:    24233937684440                0  1 day, 11:20:40.926    ok

# CPU実行結果
workspace#suzuki$ date
2026年  5月 15日 金曜日 20:50:42 JST
workspace#suzuki$ uname -a
Linux ip-172-31-14-193.us-west-2.compute.internal 6.1.115-126.197.amzn2023.x86_64 #1 SMP PREEMPT_DYNAMIC Tue Nov  5 17:36:57 UTC 2024 x86_64 x86_64 x86_64 GNU/Linux
workspace#suzuki$ codon build -release 84Py_constellations_GPU_cuda_codon_dynamic_p8_stream.py
workspace#suzuki$ ./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream -c
CPU mode selected
 N:             Total           Unique         hh:mm:ss.ms
 5:                10                0          0:00:00.000
 6:                 4                0          0:00:00.088    ok
 7:                40                0          0:00:00.024    ok
 8:                92                0          0:00:00.001    ok
 9:               352                0          0:00:00.005    ok
10:               724                0          0:00:00.002    ok
11:              2680                0          0:00:00.010    ok
12:             14200                0          0:00:00.020    ok
13:             73712                0          0:00:00.041    ok
14:            365596                0          0:00:00.091    ok
15:           2279184                0          0:00:00.171    ok
16:          14772512                0          0:00:00.275    ok
17:          95815104                0          0:00:00.409    ok
18:         666090624                0          0:00:04.455    ok
19:        4968057848                0          0:00:17.064    ok
20:       39029188884                0          0:02:10.450    ok
21:      314666222712                0          0:18:05.956    ok
22:     2691008701644                0          2:38:08.664    ok
23:    24233937684440                0   1 day, 0:43:10.509    ok


# GPU実行結果
suzuki@cudacodon$ date
2026年  5月 15日 金曜日 09:34:47 UTC
suzuki@cudacodon$ codon build -release 84Py_constellations_GPU_cuda_codon_dynamic_p8_stream.py
suzuki@cudacodon$ ./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream -g
  or
suzuki@cudacodon$ ./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream -g 5 22 32 484 1 0 7

GPU mode selected
version        : 84 stream bin GPU runner from 82 dynamic preset P8
cross_stripe_safe: 0
 N:             Total           Unique         hh:mm:ss.ms
 5:                10                0          0:00:00.000
 6:                 4                0          0:00:00.004    ok
 7:                40                0          0:00:00.003    ok
 8:                92                0          0:00:00.002    ok
 9:               352                0          0:00:00.002    ok
10:               724                0          0:00:00.003    ok
11:              2680                0          0:00:00.004    ok
12:             14200                0          0:00:00.007    ok
13:             73712                0          0:00:00.011    ok
14:            365596                0          0:00:00.018    ok
15:           2279184                0          0:00:00.037    ok
16:          14772512                0          0:00:00.107    ok
17:          95815104                0          0:00:00.466    ok
18:         666090624                0          0:00:03.505    ok
19:        4968057848                0          0:00:22.592    ok
20:       39029188884                0          0:02:24.917    ok
21:      314666222712                0          0:25:38.459    ok
22:     2691008701644                0          3:18:42.963    ok
23:    24233937684440                0   1 day, 6:08:25.451    ok

g5.16xlarge は NVIDIA A10G GPU を搭載しており、CUDA 13.0 対応のドライバが入っています。
g5.xlarge は NVIDIA A10G GPU を搭載しており、CUDA 13.0 対応のドライバが入っています。

速度が上がらない理由
-----------------------
g5.xlarge  → A10G 1枚
g5.16xlarge → A10G 1枚
------------------------

2023/11/22 これまでの最高速実装（CUDA GPU 使用/C）
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


"""
231ログ確認しました。結果は問題ありません。

```text
final total              314666222712 OK
progress rows            131 OK
duplicate/missing        0 / 0 OK
dispatch task sum        2025282 OK
required_maxd            全chunk 14
selected_MAXD            全chunk 14
schedule_words           0
stack_bytes_per_thread   208
warning/error            0 / 0
N=21 full                0:07:07.818
```

217Py基準 `0:07:07.709` には `0.109秒` 届いていませんが、230Py `0:07:08.848` の退行は解消できています。231Pyは「217相当への復帰OK」と見てよいです。

その上で、次の **232Py cleanup-only** を作成しました。kernel探索ロジックは変えていません。

```text
231Py source : 11078行 / 465624 bytes
232Py source :  9432行 / 365682 bytes

削減:
  1646行
  99942 bytes
```

今回削ったものは、主にソース冒頭の巨大な過去ログ、長いdocstring、コメント履歴です。旧bench/profile関数そのものの削除はまだしていません。まずは安全側で「kernel no-changeの可読性改善」だけに留めています。

確認した同一性は以下です。

```text
MAXD14 kernel normalized code      identical
MAXD16-21 fallback normalized code identical
build_soa_for_range normalized     identical
```

こちらでは `STATIC_ONLY=1` を通しています。

```text
source_version_tag              OK
source_future_check_mask_guard  OK
source_nosibling_parent         OK
source_217_231_restore_shape    OK
source_split_tag                split145 split232 OK
```

cudacodon側では、まずこちらを確認してください。

```bash
STATIC_ONLY=1 bash 232Py_cleanup_only_rootrestlate_restore_fastest_maxd14_validate_full_once.sh
```

その後、full実行です。

```bash
bash 232Py_cleanup_only_rootrestlate_restore_fastest_maxd14_validate_full_once.sh
```

作成物はこちらです。

* [232Py_cleanup_only_rootrestlate_restore_fastest_maxd14_probe.py](sandbox:/mnt/data/232Py_cleanup_only_rootrestlate_restore_fastest_maxd14_probe.py)
* [232Py_cleanup_only_rootrestlate_restore_fastest_maxd14_validate_full_once.sh](sandbox:/mnt/data/232Py_cleanup_only_rootrestlate_restore_fastest_maxd14_validate_full_once.sh)
* [README.md](sandbox:/mnt/data/README.md)
* [232Py cleanup-only まとめtar](sandbox:/mnt/data/232Py_cleanup_only_rootrestlate_restore_fastest_maxd14_bundle_20260706.tar)
* [sha256一覧](sandbox:/mnt/data/232_sha256.txt)

sha256は以下です。

```text
9a2609bb5ac80c465a0269303fb2fe513f3920e99dbee7de35f9f93457b4d617  232Py_cleanup_only_rootrestlate_restore_fastest_maxd14_probe.py
ba6e40c35b8ec4d6c91a7473e4d44505682519a5b671238292412dce6b7719f4  232Py_cleanup_only_rootrestlate_restore_fastest_maxd14_validate_full_once.sh
8c61290f5d54d374b83f55dc967e8a8059173a821ad5f5545c878a0c4b63365b  README.md
fd88e8b1acd4e7af5dde1160eafcae584504ad8844758e4e01684f0b481a625e  232Py_cleanup_only_rootrestlate_restore_fastest_maxd14_bundle_20260706.tar
```

"""

import gpu
import sys
from typing import List,Set,Dict,Tuple
from datetime import datetime

MAXD14:Static[int]=14
MAXD14_ANCESTOR:Static[int]=13
MAXD16:Static[int]=16
MAXD18:Static[int]=18
MAXD20:Static[int]=20
MAXD21:Static[int]=21
SCHED_WORDS14:Static[int]=0  # 210 MAXD14 keeps scalar u32 nibble schedule fields, not local u32 schedule words
SCHED_WORDS16:Static[int]=4
SCHED_WORDS18:Static[int]=5
SCHED_WORDS20:Static[int]=5
SCHED_WORDS21:Static[int]=6

VERSION_TAG:str="232 cleanup-only: 231/217 rootrestlate fastest baseline; comments/history and validation parameters pruned; kernel unchanged"
CROSS_STRIPE_SAFE_DEFAULT:bool=False

A10G_FINAL_DEFAULT_N:int=22
A10G_FINAL_DEFAULT_BLOCK:int=32
A10G_FINAL_DEFAULT_MAX_BLOCKS:int=484
A10G_FINAL_DEFAULT_LOG_LEVEL:int=0
A10G_FINAL_DEFAULT_SORT_MODE:int=0
A10G_FINAL_DEFAULT_PRESET:int=7
A10G_FINAL_DEFAULT_BENCH_MODE:int=29
A10G_FINAL_DEFAULT_REORDER_WINDOW_MULT:int=8
A10G_FINAL_DEFAULT_REORDER_PHASE_JUMP:int=7
A10G_FINAL_DEFAULT_CROSS_STRIPE_SAFE:bool=False
A10G_FINAL_DEFAULT_WORKER_ID:int=0
A10G_FINAL_DEFAULT_WORKER_COUNT:int=1
A10G_FINAL_DEFAULT_BROADMARK_VARIANT:int=2
CPU_FINAL_DEFAULT_N:int=22
DEFAULT_RANGE_NMIN:int=5
DEFAULT_RANGE_NMAX_EXCLUSIVE:int=24  # range() upper bound; outputs N=5..23

DISABLE_CONSTELLATION_SIGNATURE_PRUNE:bool=False

class TaskSoA:
  def __init__(self,m:int)->None:
    self.ld_arr:List[u32]=[u32(0)]*m
    self.rd_arr:List[u32]=[u32(0)]*m
    self.col_arr:List[u32]=[u32(0)]*m
    self.row_arr:List[int]=[0]*m
    self.ctrl0_arr:List[u32]=[u32(0)]*m
    self.free_arr:List[u32]=[u32(0)]*m
    self.markctrl_arr:List[u32]=[u32(0)]*m
    self.jmark_arr:List[int]=[0]*m
    self.end_arr:List[int]=[0]*m
    self.mark1_arr:List[int]=[0]*m
    self.mark2_arr:List[int]=[0]*m
    self.funcid_arr:List[int]=[0]*m
    self.ijkl_arr:List[int]=[0]*m

def schedule_depth_for_task(ctrl0:u32,markctrl:u32,meta_next:List[u8])->int:
  IS_BASE_MASK_I:int=69222408
  IS_JMARK_MASK_I:int=4
  IS_MARK_MASK_I:int=199209203
  IS_P5_MASK_I:int=3840
  SEL2_MASK_I:int=34742338
  STP3_MASK_I:int=21266576

  raw:int=int(ctrl0)
  marks:int=int(markctrl)
  jmark:int=marks&31
  endm:int=(marks>>5)&31
  mark1:int=(marks>>10)&31
  mark2:int=(marks>>15)&31
  depth:int=0

  while True:
    fu:int=raw&31
    rowv:int=(raw>>5)&31

    if ((IS_P5_MASK_I>>fu)&1)!=0 and rowv==mark1:
      fu=int(meta_next[fu])

    if ((IS_BASE_MASK_I>>fu)&1)!=0 and rowv==endm:
      return depth

    stepv:int=1
    nextfid:int=fu
    if ((IS_MARK_MASK_I>>fu)&1)!=0:
      markv:int=mark2 if ((SEL2_MASK_I>>fu)&1)!=0 else mark1
      if rowv==markv:
        stepv=3 if ((STP3_MASK_I>>fu)&1)!=0 else 2
        nextfid=int(meta_next[fu])

    if ((IS_JMARK_MASK_I>>fu)&1)!=0 and rowv==jmark:
      nextfid=int(meta_next[fu])

    child_row:int=rowv+stepv
    depth+=1
    if depth>21 or child_row>31:
      return 22
    raw=nextfid|(child_row<<5)

def max_schedule_depth_of_tasks(soa:TaskSoA,m:int,meta_next:List[u8])->int:
  required_maxd:int=0
  i:int=0
  while i<m:
    d:int=schedule_depth_for_task(soa.ctrl0_arr[i],soa.markctrl_arr[i],meta_next)
    if d>required_maxd:
      required_maxd=d
    i+=1
  return required_maxd

def select_static_maxd(required_maxd:int)->int:
  if required_maxd<=14:
    return 14
  if required_maxd<=16:
    return 16
  if required_maxd<=18:
    return 18
  if required_maxd<=20:
    return 20
  if required_maxd<=21:
    return 21
  return 0

def packed_schedule_words_for_maxd(selected_maxd:int)->int:
  if selected_maxd==14:
    return 0
  if selected_maxd==16:
    return 4
  if selected_maxd==18:
    return 5
  if selected_maxd==20:
    return 5
  if selected_maxd==21:
    return 6
  return 0

def packed_stack_bytes_per_thread(selected_maxd:int)->int:
  if selected_maxd==14:
    return 208
  words:int=packed_schedule_words_for_maxd(selected_maxd)
  if words==0:
    return 0
  return selected_maxd*16+words*4

@gpu.kernel
def kernel_dfs_iter_gpu_maxd14(
    ld_arr:Ptr[u32],rd_arr:Ptr[u32],col_arr:Ptr[u32],ctrl0_arr:Ptr[u32],free_arr:Ptr[u32],
    markctrl_arr:Ptr[u32],w_arr:Ptr[u64],
    meta_next:Ptr[u8],
    results:Ptr[u64],
    m:int,board_mask:u32,
    n3:u32,n4:u32,
)->None:
    IS_BASE_MASK:u32=u32(69222408)
    IS_JMARK_MASK:u32=u32(4)
    IS_MARK_MASK:u32=u32(199209203)
    IS_P5_MASK:u32=u32(3840)
    SEL2_MASK:u32=u32(34742338)

    BLOCK_CODE_B0_MASK:u32=u32(173707345)
    BLOCK_CODE_B1_MASK:u32=u32(12689458)
    BLOCK_CODE_B2_MASK:u32=u32(18088064)

    OP_STEP3_MASK:u32=u32(24)  # codes 3,4
    OP_ADD1_MASK:u32=u32(32)   # code 5
    OP_BL1_MASK:u32=u32(12)    # codes 2,3
    OP_BL2_MASK:u32=u32(16)    # code 4
    OP_KN3_MASK:u32=u32(18)    # codes 1,4
    OP_KN4_MASK:u32=u32(8)     # code 3

    ld=__array__[u32](MAXD14_ANCESTOR)
    rd=__array__[u32](MAXD14_ANCESTOR)
    col=__array__[u32](MAXD14_ANCESTOR)
    avail=__array__[u32](MAXD14_ANCESTOR)
    bm:u32=board_mask
    i:int=(gpu.block.x*gpu.block.dim.x)+gpu.thread.x
    if i>=m:return

    markctrl:u32=markctrl_arr[i]
    jmark:u32=markctrl&u32(31)
    endm:u32=(markctrl>>u32(5))&u32(31)
    mark1:u32=(markctrl>>u32(10))&u32(31)
    mark2:u32=(markctrl>>u32(15))&u32(31)
    total:u64=u64(0)

    root_ld:u32=ld_arr[i]
    root_rd:u32=rd_arr[i]
    root_col:u32=col_arr[i]
    root_a:u32=free_arr[i]&bm
    if root_a==u32(0):
      results[i]=u64(0)
      return

    schedule_raw:u32=ctrl0_arr[i]
    schedule_depth:int=0
    schedule_lo:u32=u32(0)
    schedule_hi:u32=u32(0)
    child_jmark_mask:u32=u32(0)
    future_check_mask:u32=u32(0)
    terminal_parent_depth:int=0
    terminal_is_base14:u32=u32(0)
    root_action:u32=u32(0)
    while True:
      schedule_fu:u32=schedule_raw&u32(31)
      schedule_rowv:u32=(schedule_raw>>u32(5))&u32(31)

      if ((IS_P5_MASK>>schedule_fu)&u32(1))!=u32(0):
        if schedule_rowv==mark1:
          schedule_fu=u32(meta_next[int(schedule_fu)])

      frame_action:u32=u32(0)
      frame_nibble:u32=u32(0)
      frame_raw:u32=u32(0)
      schedule_isbu:u32=(IS_BASE_MASK>>schedule_fu)&u32(1)
      if schedule_isbu!=u32(0) and schedule_rowv==endm:
        frame_action=u32(3) if schedule_fu==u32(14) else u32(2)
      else:
        schedule_ismu:u32=(IS_MARK_MASK>>schedule_fu)&u32(1)
        schedule_block_code:u32=u32(0)
        schedule_stepv:u32=u32(1)
        schedule_use_futureu:u32=u32(1)-schedule_ismu
        schedule_nextfidu:u32=schedule_fu

        if schedule_ismu!=u32(0):
          schedule_markv:u32=mark2 if ((SEL2_MASK>>schedule_fu)&u32(1))!=u32(0) else mark1
          if schedule_rowv==schedule_markv:
            schedule_block_code=(
              ((BLOCK_CODE_B0_MASK>>schedule_fu)&u32(1))
              |(((BLOCK_CODE_B1_MASK>>schedule_fu)&u32(1))<<u32(1))
              |(((BLOCK_CODE_B2_MASK>>schedule_fu)&u32(1))<<u32(2))
            )
            schedule_stepv=u32(2)+((OP_STEP3_MASK>>schedule_block_code)&u32(1))
            schedule_use_futureu=u32(0)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_isju:u32=(IS_JMARK_MASK>>schedule_fu)&u32(1)
        if schedule_isju!=u32(0):
          if schedule_rowv==jmark:
            frame_action=u32(1)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_child_rowu:u32=schedule_rowv+schedule_stepv
        schedule_fcvu:u32=u32(0)
        if schedule_use_futureu!=u32(0) and schedule_child_rowu<endm:
          schedule_fcvu=u32(1)
        frame_nibble=schedule_block_code|(schedule_fcvu<<u32(3))
        frame_raw=schedule_nextfidu|(schedule_child_rowu<<u32(5))

      if schedule_depth==0:
        root_action=frame_action
      else:
        parent_depth:int=schedule_depth-1
        if frame_action==u32(1):
          child_jmark_mask|=u32(1)<<u32(parent_depth)
        elif frame_action>=u32(2):
          terminal_parent_depth=parent_depth
          terminal_is_base14=u32(1) if frame_action==u32(3) else u32(0)

      if frame_action>=u32(2):
        break

      if schedule_fcvu!=u32(0):
        future_check_mask|=u32(1)<<u32(schedule_depth)

      if schedule_depth<8:
        schedule_lo|=frame_nibble<<u32(schedule_depth*4)
      else:
        schedule_hi|=frame_nibble<<u32((schedule_depth-8)*4)
      schedule_raw=frame_raw
      schedule_depth+=1

    if root_action==u32(2):
      results[i]=w_arr[i]
      return
    if root_action==u32(3):
      total+=u64(1) if ((root_a&~u32(1))!=u32(0)) else u64(0)
      results[i]=total*w_arr[i]
      return
    if root_action==u32(1):
      root_a&=~u32(1)
      if root_a==u32(0):
        results[i]=u64(0)
        return
      root_ld|=u32(1)

    terminal_depth:int=terminal_parent_depth
    terminal_base14:u32=terminal_is_base14

    save_sp:int=0
    cur_depth:int=0
    cur_ld:u32=root_ld
    cur_rd:u32=root_rd
    cur_col:u32=root_col
    cur_avail:u32=root_a

    root_rest:u32=cur_avail&(cur_avail-u32(1))
    root_second:u32=root_rest&(u32(0)-root_rest)
    root_after_second:u32=root_rest^root_second

    if root_after_second==u32(0):
      root_first:u32=cur_avail&(u32(0)-cur_avail)
      pr_nibble_op:u32=schedule_lo&u32(15)
      pr_block_code:u32=pr_nibble_op&u32(7)
      pr_bit:u32=root_first

      pr_nld:u32=u32(0)
      pr_nrd:u32=u32(0)
      if pr_block_code!=u32(0):
        pr_stepu:u32=u32(2)+((OP_STEP3_MASK>>pr_block_code)&u32(1))
        pr_addvu:u32=(OP_ADD1_MASK>>pr_block_code)&u32(1)
        pr_bLiu:u32=(
          ((OP_BL1_MASK>>pr_block_code)&u32(1))
          |(((OP_BL2_MASK>>pr_block_code)&u32(1))<<u32(1))
        )
        pr_ktu:u32=(
          ((OP_KN3_MASK>>pr_block_code)&u32(1))
          |(((OP_KN4_MASK>>pr_block_code)&u32(1))<<u32(1))
        )
        pr_bKu:u32=(n3&(u32(0)-(pr_ktu&u32(1))))|(n4&(u32(0)-(pr_ktu>>u32(1))))
        pr_nld=((cur_ld|pr_bit)<<pr_stepu)|pr_addvu|pr_bLiu
        pr_nrd=((cur_rd|pr_bit)>>pr_stepu)|pr_bKu
      else:
        pr_nld=(cur_ld|pr_bit)<<u32(1)
        pr_nrd=(cur_rd|pr_bit)>>u32(1)
      pr_ncol:u32=cur_col|pr_bit
      pr_nf:u32=bm&~(pr_nld|pr_nrd|pr_ncol)
      pr_descend:u32=u32(1)
      if pr_nf==u32(0):
        pr_descend=u32(0)
      if pr_descend!=u32(0):
        if future_check_mask!=u32(0):
          if (pr_nibble_op&u32(8))!=u32(0):
            if (bm&~((pr_nld<<u32(1))|(pr_nrd>>u32(1))|pr_ncol))==u32(0):
              pr_descend=u32(0)

      if pr_descend!=u32(0):
        if terminal_depth==0:
          if terminal_base14==u32(0):
            total+=u64(1)
          else:
            total+=u64(1) if ((pr_nf&~u32(1))!=u32(0)) else u64(0)
          pr_descend=u32(0)

      if pr_descend!=u32(0):
        pr_child_jmark:u32=child_jmark_mask&u32(1)
        if pr_child_jmark!=u32(0):
          pr_nf&=~u32(1)
          if pr_nf==u32(0):
            pr_descend=u32(0)
          else:
            pr_nld|=u32(1)

      cur_avail=root_rest
      if pr_descend!=u32(0):
        if cur_avail!=u32(0):
          ld[save_sp]=cur_ld
          rd[save_sp]=cur_rd
          col[save_sp]=cur_col
          avail[save_sp]=cur_avail|(u32(cur_depth)<<u32(27))
          save_sp+=1
        cur_ld=pr_nld
        cur_rd=pr_nrd
        cur_col=pr_ncol
        cur_avail=pr_nf
        cur_depth=1

    while True:
      if cur_avail==u32(0):
        if save_sp==0:
          break
        save_sp-=1
        cur_ld=ld[save_sp]
        cur_rd=rd[save_sp]
        cur_col=col[save_sp]
        saved_avail:u32=avail[save_sp]
        cur_avail=saved_avail&bm
        cur_depth=int(saved_avail>>u32(27))
        continue

      nibble_op:u32=u32(0)
      if cur_depth<8:
        nibble_op=(schedule_lo>>u32(cur_depth*4))&u32(15)
      else:
        nibble_op=(schedule_hi>>u32((cur_depth-8)*4))&u32(15)
      block_code:u32=nibble_op&u32(7)

      bit:u32=cur_avail&(u32(0)-cur_avail)
      cur_avail=cur_avail^bit

      nld:u32=u32(0)
      nrd:u32=u32(0)
      if block_code!=u32(0):
        stepu:u32=u32(2)+((OP_STEP3_MASK>>block_code)&u32(1))
        addvu:u32=(OP_ADD1_MASK>>block_code)&u32(1)
        bLiu:u32=(
          ((OP_BL1_MASK>>block_code)&u32(1))
          |(((OP_BL2_MASK>>block_code)&u32(1))<<u32(1))
        )
        ktu:u32=(
          ((OP_KN3_MASK>>block_code)&u32(1))
          |(((OP_KN4_MASK>>block_code)&u32(1))<<u32(1))
        )
        bKu:u32=(n3&(u32(0)-(ktu&u32(1))))|(n4&(u32(0)-(ktu>>u32(1))))
        nld=((cur_ld|bit)<<stepu)|addvu|bLiu
        nrd=((cur_rd|bit)>>stepu)|bKu
      else:
        nld=(cur_ld|bit)<<u32(1)
        nrd=(cur_rd|bit)>>u32(1)
      ncol:u32=cur_col|bit
      nf:u32=bm&~(nld|nrd|ncol)
      if nf==u32(0):
        continue
      if future_check_mask!=u32(0):
        if (nibble_op&u32(8))!=u32(0):
          if (bm&~((nld<<u32(1))|(nrd>>u32(1))|ncol))==u32(0):
            continue

      if cur_depth==terminal_depth:
        if terminal_base14==u32(0):
          total+=u64(1)
        else:
          total+=u64(1) if ((nf&~u32(1))!=u32(0)) else u64(0)
        continue

      child_jmark:u32=(child_jmark_mask>>u32(cur_depth))&u32(1)
      if child_jmark!=u32(0):
        nf&=~u32(1)
        if nf==u32(0):
          continue
        nld|=u32(1)

      next_depth:int=cur_depth+1
      if cur_avail!=u32(0):
        ld[save_sp]=cur_ld
        rd[save_sp]=cur_rd
        col[save_sp]=cur_col
        avail[save_sp]=cur_avail|(u32(cur_depth)<<u32(27))
        save_sp+=1
      cur_ld=nld
      cur_rd=nrd
      cur_col=ncol
      cur_avail=nf
      cur_depth=next_depth
    results[i]=total*w_arr[i]

@gpu.kernel
def kernel_dfs_iter_gpu_maxd16(
    ld_arr:Ptr[u32],rd_arr:Ptr[u32],col_arr:Ptr[u32],ctrl0_arr:Ptr[u32],free_arr:Ptr[u32],
    markctrl_arr:Ptr[u32],w_arr:Ptr[u64],
    meta_next:Ptr[u8],
    results:Ptr[u64],
    m:int,board_mask:u32,
    n3:u32,n4:u32,
)->None:
    IS_BASE_MASK:u32=u32(69222408)
    IS_JMARK_MASK:u32=u32(4)
    IS_MARK_MASK:u32=u32(199209203)
    IS_P5_MASK:u32=u32(3840)
    SEL2_MASK:u32=u32(34742338)

    BLOCK_CODE_B0_MASK:u32=u32(173707345)
    BLOCK_CODE_B1_MASK:u32=u32(12689458)
    BLOCK_CODE_B2_MASK:u32=u32(18088064)

    OP_STEP3_MASK:u32=u32(24)  # codes 3,4
    OP_ADD1_MASK:u32=u32(32)   # code 5
    OP_BL1_MASK:u32=u32(12)    # codes 2,3
    OP_BL2_MASK:u32=u32(16)    # code 4
    OP_KN3_MASK:u32=u32(18)    # codes 1,4
    OP_KN4_MASK:u32=u32(8)     # code 3

    ld=__array__[u32](MAXD16)
    rd=__array__[u32](MAXD16)
    col=__array__[u32](MAXD16)
    avail=__array__[u32](MAXD16)
    packed_schedule=__array__[u32](SCHED_WORDS16)
    bm:u32=board_mask
    i:int=(gpu.block.x*gpu.block.dim.x)+gpu.thread.x
    if i>=m:return

    markctrl:u32=markctrl_arr[i]
    jmark:u32=markctrl&u32(31)
    endm:u32=(markctrl>>u32(5))&u32(31)
    mark1:u32=(markctrl>>u32(10))&u32(31)
    mark2:u32=(markctrl>>u32(15))&u32(31)
    total:u64=u64(0)

    root_ld:u32=ld_arr[i]
    root_rd:u32=rd_arr[i]
    root_col:u32=col_arr[i]
    root_a:u32=free_arr[i]&bm
    if root_a==u32(0):
      results[i]=u64(0)
      return

    schedule_raw:u32=ctrl0_arr[i]
    schedule_depth:int=0
    pack_word:u32=u32(0)
    pack_word_index:int=0
    root_action:u32=u32(0)
    while True:
      schedule_fu:u32=schedule_raw&u32(31)
      schedule_rowv:u32=(schedule_raw>>u32(5))&u32(31)

      if ((IS_P5_MASK>>schedule_fu)&u32(1))!=u32(0):
        if schedule_rowv==mark1:
          schedule_fu=u32(meta_next[int(schedule_fu)])

      frame_action:u32=u32(0)
      frame_opcode:u32=u32(0)
      frame_raw:u32=u32(0)
      schedule_isbu:u32=(IS_BASE_MASK>>schedule_fu)&u32(1)
      if schedule_isbu!=u32(0) and schedule_rowv==endm:
        frame_action=u32(3) if schedule_fu==u32(14) else u32(2)
      else:
        schedule_ismu:u32=(IS_MARK_MASK>>schedule_fu)&u32(1)
        schedule_block_code:u32=u32(0)
        schedule_stepv:u32=u32(1)
        schedule_use_futureu:u32=u32(1)-schedule_ismu
        schedule_nextfidu:u32=schedule_fu

        if schedule_ismu!=u32(0):
          schedule_markv:u32=mark2 if ((SEL2_MASK>>schedule_fu)&u32(1))!=u32(0) else mark1
          if schedule_rowv==schedule_markv:
            schedule_block_code=(
              ((BLOCK_CODE_B0_MASK>>schedule_fu)&u32(1))
              |(((BLOCK_CODE_B1_MASK>>schedule_fu)&u32(1))<<u32(1))
              |(((BLOCK_CODE_B2_MASK>>schedule_fu)&u32(1))<<u32(2))
            )
            schedule_stepv=u32(2)+((OP_STEP3_MASK>>schedule_block_code)&u32(1))
            schedule_use_futureu=u32(0)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_isju:u32=(IS_JMARK_MASK>>schedule_fu)&u32(1)
        if schedule_isju!=u32(0):
          if schedule_rowv==jmark:
            frame_action=u32(1)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_child_rowu:u32=schedule_rowv+schedule_stepv
        schedule_fcvu:u32=u32(0)
        if schedule_use_futureu!=u32(0) and schedule_child_rowu<endm:
          schedule_fcvu=u32(1)
        frame_opcode=schedule_block_code|(schedule_fcvu<<u32(3))
        frame_raw=schedule_nextfidu|(schedule_child_rowu<<u32(5))

      if schedule_depth==0:
        root_action=frame_action
      else:
        parent_lane:int=(schedule_depth-1)&3
        parent_action_shift:u32=u32(parent_lane*8+4)
        pack_word|=frame_action<<parent_action_shift
        if (schedule_depth&3)==0:
          packed_schedule[pack_word_index]=pack_word
          pack_word_index+=1
          pack_word=u32(0)

      if frame_action>=u32(2):
        if schedule_depth>0 and (schedule_depth&3)!=0:
          packed_schedule[pack_word_index]=pack_word
        break

      opcode_shift:u32=u32((schedule_depth&3)*8)
      pack_word|=frame_opcode<<opcode_shift
      schedule_raw=frame_raw
      schedule_depth+=1

    if root_action==u32(2):
      results[i]=w_arr[i]
      return
    if root_action==u32(3):
      total+=u64(1) if ((root_a&~u32(1))!=u32(0)) else u64(0)
      results[i]=total*w_arr[i]
      return
    if root_action==u32(1):
      root_a&=~u32(1)
      if root_a==u32(0):
        results[i]=u64(0)
        return
      root_ld|=u32(1)

    sp:int=0
    ld[0]=root_ld
    rd[0]=root_rd
    col[0]=root_col
    avail[0]=root_a

    while True:
      a:u32=avail[sp]
      if a==u32(0):
        if sp==0:
          break
        sp-=1
        continue

      opcode_word:u32=packed_schedule[sp>>2]
      opcode:u32=(opcode_word>>u32((sp&3)*8))&u32(255)
      block_code:u32=opcode&u32(7)
      bit:u32=a&(u32(0)-a)
      avail[sp]=a^bit

      nld:u32=u32(0)
      nrd:u32=u32(0)
      if block_code!=u32(0):
        stepu:u32=u32(2)+((OP_STEP3_MASK>>block_code)&u32(1))
        addvu:u32=(OP_ADD1_MASK>>block_code)&u32(1)
        bLiu:u32=(
          ((OP_BL1_MASK>>block_code)&u32(1))
          |(((OP_BL2_MASK>>block_code)&u32(1))<<u32(1))
        )
        ktu:u32=(
          ((OP_KN3_MASK>>block_code)&u32(1))
          |(((OP_KN4_MASK>>block_code)&u32(1))<<u32(1))
        )
        bKu:u32=(n3&(u32(0)-(ktu&u32(1))))|(n4&(u32(0)-(ktu>>u32(1))))
        nld=((ld[sp]|bit)<<stepu)|addvu|bLiu
        nrd=((rd[sp]|bit)>>stepu)|bKu
      else:
        nld=(ld[sp]|bit)<<u32(1)
        nrd=(rd[sp]|bit)>>u32(1)
      ncol:u32=col[sp]|bit
      nf:u32=bm&~(nld|nrd|ncol)
      if nf==u32(0):
        continue
      if (opcode&u32(8))!=u32(0):
        if (bm&~((nld<<u32(1))|(nrd>>u32(1))|ncol))==u32(0):
          continue

      child_action:u32=(opcode>>u32(4))&u32(3)
      if child_action>=u32(2):
        if child_action==u32(2):
          total+=u64(1)
        else:
          total+=u64(1) if ((nf&~u32(1))!=u32(0)) else u64(0)
        continue
      if child_action==u32(1):
        nf&=~u32(1)
        if nf==u32(0):
          continue
        nld|=u32(1)

      sp+=1
      ld[sp]=nld
      rd[sp]=nrd
      col[sp]=ncol
      avail[sp]=nf
    results[i]=total*w_arr[i]

@gpu.kernel
def kernel_dfs_iter_gpu_maxd18(
    ld_arr:Ptr[u32],rd_arr:Ptr[u32],col_arr:Ptr[u32],ctrl0_arr:Ptr[u32],free_arr:Ptr[u32],
    markctrl_arr:Ptr[u32],w_arr:Ptr[u64],
    meta_next:Ptr[u8],
    results:Ptr[u64],
    m:int,board_mask:u32,
    n3:u32,n4:u32,
)->None:
    IS_BASE_MASK:u32=u32(69222408)
    IS_JMARK_MASK:u32=u32(4)
    IS_MARK_MASK:u32=u32(199209203)
    IS_P5_MASK:u32=u32(3840)
    SEL2_MASK:u32=u32(34742338)

    BLOCK_CODE_B0_MASK:u32=u32(173707345)
    BLOCK_CODE_B1_MASK:u32=u32(12689458)
    BLOCK_CODE_B2_MASK:u32=u32(18088064)

    OP_STEP3_MASK:u32=u32(24)  # codes 3,4
    OP_ADD1_MASK:u32=u32(32)   # code 5
    OP_BL1_MASK:u32=u32(12)    # codes 2,3
    OP_BL2_MASK:u32=u32(16)    # code 4
    OP_KN3_MASK:u32=u32(18)    # codes 1,4
    OP_KN4_MASK:u32=u32(8)     # code 3

    ld=__array__[u32](MAXD18)
    rd=__array__[u32](MAXD18)
    col=__array__[u32](MAXD18)
    avail=__array__[u32](MAXD18)
    packed_schedule=__array__[u32](SCHED_WORDS18)
    bm:u32=board_mask
    i:int=(gpu.block.x*gpu.block.dim.x)+gpu.thread.x
    if i>=m:return

    markctrl:u32=markctrl_arr[i]
    jmark:u32=markctrl&u32(31)
    endm:u32=(markctrl>>u32(5))&u32(31)
    mark1:u32=(markctrl>>u32(10))&u32(31)
    mark2:u32=(markctrl>>u32(15))&u32(31)
    total:u64=u64(0)

    root_ld:u32=ld_arr[i]
    root_rd:u32=rd_arr[i]
    root_col:u32=col_arr[i]
    root_a:u32=free_arr[i]&bm
    if root_a==u32(0):
      results[i]=u64(0)
      return

    schedule_raw:u32=ctrl0_arr[i]
    schedule_depth:int=0
    pack_word:u32=u32(0)
    pack_word_index:int=0
    root_action:u32=u32(0)
    while True:
      schedule_fu:u32=schedule_raw&u32(31)
      schedule_rowv:u32=(schedule_raw>>u32(5))&u32(31)

      if ((IS_P5_MASK>>schedule_fu)&u32(1))!=u32(0):
        if schedule_rowv==mark1:
          schedule_fu=u32(meta_next[int(schedule_fu)])

      frame_action:u32=u32(0)
      frame_opcode:u32=u32(0)
      frame_raw:u32=u32(0)
      schedule_isbu:u32=(IS_BASE_MASK>>schedule_fu)&u32(1)
      if schedule_isbu!=u32(0) and schedule_rowv==endm:
        frame_action=u32(3) if schedule_fu==u32(14) else u32(2)
      else:
        schedule_ismu:u32=(IS_MARK_MASK>>schedule_fu)&u32(1)
        schedule_block_code:u32=u32(0)
        schedule_stepv:u32=u32(1)
        schedule_use_futureu:u32=u32(1)-schedule_ismu
        schedule_nextfidu:u32=schedule_fu

        if schedule_ismu!=u32(0):
          schedule_markv:u32=mark2 if ((SEL2_MASK>>schedule_fu)&u32(1))!=u32(0) else mark1
          if schedule_rowv==schedule_markv:
            schedule_block_code=(
              ((BLOCK_CODE_B0_MASK>>schedule_fu)&u32(1))
              |(((BLOCK_CODE_B1_MASK>>schedule_fu)&u32(1))<<u32(1))
              |(((BLOCK_CODE_B2_MASK>>schedule_fu)&u32(1))<<u32(2))
            )
            schedule_stepv=u32(2)+((OP_STEP3_MASK>>schedule_block_code)&u32(1))
            schedule_use_futureu=u32(0)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_isju:u32=(IS_JMARK_MASK>>schedule_fu)&u32(1)
        if schedule_isju!=u32(0):
          if schedule_rowv==jmark:
            frame_action=u32(1)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_child_rowu:u32=schedule_rowv+schedule_stepv
        schedule_fcvu:u32=u32(0)
        if schedule_use_futureu!=u32(0) and schedule_child_rowu<endm:
          schedule_fcvu=u32(1)
        frame_opcode=schedule_block_code|(schedule_fcvu<<u32(3))
        frame_raw=schedule_nextfidu|(schedule_child_rowu<<u32(5))

      if schedule_depth==0:
        root_action=frame_action
      else:
        parent_lane:int=(schedule_depth-1)&3
        parent_action_shift:u32=u32(parent_lane*8+4)
        pack_word|=frame_action<<parent_action_shift
        if (schedule_depth&3)==0:
          packed_schedule[pack_word_index]=pack_word
          pack_word_index+=1
          pack_word=u32(0)

      if frame_action>=u32(2):
        if schedule_depth>0 and (schedule_depth&3)!=0:
          packed_schedule[pack_word_index]=pack_word
        break

      opcode_shift:u32=u32((schedule_depth&3)*8)
      pack_word|=frame_opcode<<opcode_shift
      schedule_raw=frame_raw
      schedule_depth+=1

    if root_action==u32(2):
      results[i]=w_arr[i]
      return
    if root_action==u32(3):
      total+=u64(1) if ((root_a&~u32(1))!=u32(0)) else u64(0)
      results[i]=total*w_arr[i]
      return
    if root_action==u32(1):
      root_a&=~u32(1)
      if root_a==u32(0):
        results[i]=u64(0)
        return
      root_ld|=u32(1)

    sp:int=0
    ld[0]=root_ld
    rd[0]=root_rd
    col[0]=root_col
    avail[0]=root_a

    while True:
      a:u32=avail[sp]
      if a==u32(0):
        if sp==0:
          break
        sp-=1
        continue

      opcode_word:u32=packed_schedule[sp>>2]
      opcode:u32=(opcode_word>>u32((sp&3)*8))&u32(255)
      block_code:u32=opcode&u32(7)
      bit:u32=a&(u32(0)-a)
      avail[sp]=a^bit

      nld:u32=u32(0)
      nrd:u32=u32(0)
      if block_code!=u32(0):
        stepu:u32=u32(2)+((OP_STEP3_MASK>>block_code)&u32(1))
        addvu:u32=(OP_ADD1_MASK>>block_code)&u32(1)
        bLiu:u32=(
          ((OP_BL1_MASK>>block_code)&u32(1))
          |(((OP_BL2_MASK>>block_code)&u32(1))<<u32(1))
        )
        ktu:u32=(
          ((OP_KN3_MASK>>block_code)&u32(1))
          |(((OP_KN4_MASK>>block_code)&u32(1))<<u32(1))
        )
        bKu:u32=(n3&(u32(0)-(ktu&u32(1))))|(n4&(u32(0)-(ktu>>u32(1))))
        nld=((ld[sp]|bit)<<stepu)|addvu|bLiu
        nrd=((rd[sp]|bit)>>stepu)|bKu
      else:
        nld=(ld[sp]|bit)<<u32(1)
        nrd=(rd[sp]|bit)>>u32(1)
      ncol:u32=col[sp]|bit
      nf:u32=bm&~(nld|nrd|ncol)
      if nf==u32(0):
        continue
      if (opcode&u32(8))!=u32(0):
        if (bm&~((nld<<u32(1))|(nrd>>u32(1))|ncol))==u32(0):
          continue

      child_action:u32=(opcode>>u32(4))&u32(3)
      if child_action>=u32(2):
        if child_action==u32(2):
          total+=u64(1)
        else:
          total+=u64(1) if ((nf&~u32(1))!=u32(0)) else u64(0)
        continue
      if child_action==u32(1):
        nf&=~u32(1)
        if nf==u32(0):
          continue
        nld|=u32(1)

      sp+=1
      ld[sp]=nld
      rd[sp]=nrd
      col[sp]=ncol
      avail[sp]=nf
    results[i]=total*w_arr[i]

@gpu.kernel
def kernel_dfs_iter_gpu_maxd20(
    ld_arr:Ptr[u32],rd_arr:Ptr[u32],col_arr:Ptr[u32],ctrl0_arr:Ptr[u32],free_arr:Ptr[u32],
    markctrl_arr:Ptr[u32],w_arr:Ptr[u64],
    meta_next:Ptr[u8],
    results:Ptr[u64],
    m:int,board_mask:u32,
    n3:u32,n4:u32,
)->None:
    IS_BASE_MASK:u32=u32(69222408)
    IS_JMARK_MASK:u32=u32(4)
    IS_MARK_MASK:u32=u32(199209203)
    IS_P5_MASK:u32=u32(3840)
    SEL2_MASK:u32=u32(34742338)

    BLOCK_CODE_B0_MASK:u32=u32(173707345)
    BLOCK_CODE_B1_MASK:u32=u32(12689458)
    BLOCK_CODE_B2_MASK:u32=u32(18088064)

    OP_STEP3_MASK:u32=u32(24)  # codes 3,4
    OP_ADD1_MASK:u32=u32(32)   # code 5
    OP_BL1_MASK:u32=u32(12)    # codes 2,3
    OP_BL2_MASK:u32=u32(16)    # code 4
    OP_KN3_MASK:u32=u32(18)    # codes 1,4
    OP_KN4_MASK:u32=u32(8)     # code 3

    ld=__array__[u32](MAXD20)
    rd=__array__[u32](MAXD20)
    col=__array__[u32](MAXD20)
    avail=__array__[u32](MAXD20)
    packed_schedule=__array__[u32](SCHED_WORDS20)
    bm:u32=board_mask
    i:int=(gpu.block.x*gpu.block.dim.x)+gpu.thread.x
    if i>=m:return

    markctrl:u32=markctrl_arr[i]
    jmark:u32=markctrl&u32(31)
    endm:u32=(markctrl>>u32(5))&u32(31)
    mark1:u32=(markctrl>>u32(10))&u32(31)
    mark2:u32=(markctrl>>u32(15))&u32(31)
    total:u64=u64(0)

    root_ld:u32=ld_arr[i]
    root_rd:u32=rd_arr[i]
    root_col:u32=col_arr[i]
    root_a:u32=free_arr[i]&bm
    if root_a==u32(0):
      results[i]=u64(0)
      return

    schedule_raw:u32=ctrl0_arr[i]
    schedule_depth:int=0
    pack_word:u32=u32(0)
    pack_word_index:int=0
    root_action:u32=u32(0)
    while True:
      schedule_fu:u32=schedule_raw&u32(31)
      schedule_rowv:u32=(schedule_raw>>u32(5))&u32(31)

      if ((IS_P5_MASK>>schedule_fu)&u32(1))!=u32(0):
        if schedule_rowv==mark1:
          schedule_fu=u32(meta_next[int(schedule_fu)])

      frame_action:u32=u32(0)
      frame_opcode:u32=u32(0)
      frame_raw:u32=u32(0)
      schedule_isbu:u32=(IS_BASE_MASK>>schedule_fu)&u32(1)
      if schedule_isbu!=u32(0) and schedule_rowv==endm:
        frame_action=u32(3) if schedule_fu==u32(14) else u32(2)
      else:
        schedule_ismu:u32=(IS_MARK_MASK>>schedule_fu)&u32(1)
        schedule_block_code:u32=u32(0)
        schedule_stepv:u32=u32(1)
        schedule_use_futureu:u32=u32(1)-schedule_ismu
        schedule_nextfidu:u32=schedule_fu

        if schedule_ismu!=u32(0):
          schedule_markv:u32=mark2 if ((SEL2_MASK>>schedule_fu)&u32(1))!=u32(0) else mark1
          if schedule_rowv==schedule_markv:
            schedule_block_code=(
              ((BLOCK_CODE_B0_MASK>>schedule_fu)&u32(1))
              |(((BLOCK_CODE_B1_MASK>>schedule_fu)&u32(1))<<u32(1))
              |(((BLOCK_CODE_B2_MASK>>schedule_fu)&u32(1))<<u32(2))
            )
            schedule_stepv=u32(2)+((OP_STEP3_MASK>>schedule_block_code)&u32(1))
            schedule_use_futureu=u32(0)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_isju:u32=(IS_JMARK_MASK>>schedule_fu)&u32(1)
        if schedule_isju!=u32(0):
          if schedule_rowv==jmark:
            frame_action=u32(1)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_child_rowu:u32=schedule_rowv+schedule_stepv
        schedule_fcvu:u32=u32(0)
        if schedule_use_futureu!=u32(0) and schedule_child_rowu<endm:
          schedule_fcvu=u32(1)
        frame_opcode=schedule_block_code|(schedule_fcvu<<u32(3))
        frame_raw=schedule_nextfidu|(schedule_child_rowu<<u32(5))

      if schedule_depth==0:
        root_action=frame_action
      else:
        parent_lane:int=(schedule_depth-1)&3
        parent_action_shift:u32=u32(parent_lane*8+4)
        pack_word|=frame_action<<parent_action_shift
        if (schedule_depth&3)==0:
          packed_schedule[pack_word_index]=pack_word
          pack_word_index+=1
          pack_word=u32(0)

      if frame_action>=u32(2):
        if schedule_depth>0 and (schedule_depth&3)!=0:
          packed_schedule[pack_word_index]=pack_word
        break

      opcode_shift:u32=u32((schedule_depth&3)*8)
      pack_word|=frame_opcode<<opcode_shift
      schedule_raw=frame_raw
      schedule_depth+=1

    if root_action==u32(2):
      results[i]=w_arr[i]
      return
    if root_action==u32(3):
      total+=u64(1) if ((root_a&~u32(1))!=u32(0)) else u64(0)
      results[i]=total*w_arr[i]
      return
    if root_action==u32(1):
      root_a&=~u32(1)
      if root_a==u32(0):
        results[i]=u64(0)
        return
      root_ld|=u32(1)

    sp:int=0
    ld[0]=root_ld
    rd[0]=root_rd
    col[0]=root_col
    avail[0]=root_a

    while True:
      a:u32=avail[sp]
      if a==u32(0):
        if sp==0:
          break
        sp-=1
        continue

      opcode_word:u32=packed_schedule[sp>>2]
      opcode:u32=(opcode_word>>u32((sp&3)*8))&u32(255)
      block_code:u32=opcode&u32(7)
      bit:u32=a&(u32(0)-a)
      avail[sp]=a^bit

      nld:u32=u32(0)
      nrd:u32=u32(0)
      if block_code!=u32(0):
        stepu:u32=u32(2)+((OP_STEP3_MASK>>block_code)&u32(1))
        addvu:u32=(OP_ADD1_MASK>>block_code)&u32(1)
        bLiu:u32=(
          ((OP_BL1_MASK>>block_code)&u32(1))
          |(((OP_BL2_MASK>>block_code)&u32(1))<<u32(1))
        )
        ktu:u32=(
          ((OP_KN3_MASK>>block_code)&u32(1))
          |(((OP_KN4_MASK>>block_code)&u32(1))<<u32(1))
        )
        bKu:u32=(n3&(u32(0)-(ktu&u32(1))))|(n4&(u32(0)-(ktu>>u32(1))))
        nld=((ld[sp]|bit)<<stepu)|addvu|bLiu
        nrd=((rd[sp]|bit)>>stepu)|bKu
      else:
        nld=(ld[sp]|bit)<<u32(1)
        nrd=(rd[sp]|bit)>>u32(1)
      ncol:u32=col[sp]|bit
      nf:u32=bm&~(nld|nrd|ncol)
      if nf==u32(0):
        continue
      if (opcode&u32(8))!=u32(0):
        if (bm&~((nld<<u32(1))|(nrd>>u32(1))|ncol))==u32(0):
          continue

      child_action:u32=(opcode>>u32(4))&u32(3)
      if child_action>=u32(2):
        if child_action==u32(2):
          total+=u64(1)
        else:
          total+=u64(1) if ((nf&~u32(1))!=u32(0)) else u64(0)
        continue
      if child_action==u32(1):
        nf&=~u32(1)
        if nf==u32(0):
          continue
        nld|=u32(1)

      sp+=1
      ld[sp]=nld
      rd[sp]=nrd
      col[sp]=ncol
      avail[sp]=nf
    results[i]=total*w_arr[i]

@gpu.kernel
def kernel_dfs_iter_gpu_maxd21(
    ld_arr:Ptr[u32],rd_arr:Ptr[u32],col_arr:Ptr[u32],ctrl0_arr:Ptr[u32],free_arr:Ptr[u32],
    markctrl_arr:Ptr[u32],w_arr:Ptr[u64],
    meta_next:Ptr[u8],
    results:Ptr[u64],
    m:int,board_mask:u32,
    n3:u32,n4:u32,
)->None:
    IS_BASE_MASK:u32=u32(69222408)
    IS_JMARK_MASK:u32=u32(4)
    IS_MARK_MASK:u32=u32(199209203)
    IS_P5_MASK:u32=u32(3840)
    SEL2_MASK:u32=u32(34742338)

    BLOCK_CODE_B0_MASK:u32=u32(173707345)
    BLOCK_CODE_B1_MASK:u32=u32(12689458)
    BLOCK_CODE_B2_MASK:u32=u32(18088064)

    OP_STEP3_MASK:u32=u32(24)  # codes 3,4
    OP_ADD1_MASK:u32=u32(32)   # code 5
    OP_BL1_MASK:u32=u32(12)    # codes 2,3
    OP_BL2_MASK:u32=u32(16)    # code 4
    OP_KN3_MASK:u32=u32(18)    # codes 1,4
    OP_KN4_MASK:u32=u32(8)     # code 3

    ld=__array__[u32](MAXD21)
    rd=__array__[u32](MAXD21)
    col=__array__[u32](MAXD21)
    avail=__array__[u32](MAXD21)
    packed_schedule=__array__[u32](SCHED_WORDS21)
    bm:u32=board_mask
    i:int=(gpu.block.x*gpu.block.dim.x)+gpu.thread.x
    if i>=m:return

    markctrl:u32=markctrl_arr[i]
    jmark:u32=markctrl&u32(31)
    endm:u32=(markctrl>>u32(5))&u32(31)
    mark1:u32=(markctrl>>u32(10))&u32(31)
    mark2:u32=(markctrl>>u32(15))&u32(31)
    total:u64=u64(0)

    root_ld:u32=ld_arr[i]
    root_rd:u32=rd_arr[i]
    root_col:u32=col_arr[i]
    root_a:u32=free_arr[i]&bm
    if root_a==u32(0):
      results[i]=u64(0)
      return

    schedule_raw:u32=ctrl0_arr[i]
    schedule_depth:int=0
    pack_word:u32=u32(0)
    pack_word_index:int=0
    root_action:u32=u32(0)
    while True:
      schedule_fu:u32=schedule_raw&u32(31)
      schedule_rowv:u32=(schedule_raw>>u32(5))&u32(31)

      if ((IS_P5_MASK>>schedule_fu)&u32(1))!=u32(0):
        if schedule_rowv==mark1:
          schedule_fu=u32(meta_next[int(schedule_fu)])

      frame_action:u32=u32(0)
      frame_opcode:u32=u32(0)
      frame_raw:u32=u32(0)
      schedule_isbu:u32=(IS_BASE_MASK>>schedule_fu)&u32(1)
      if schedule_isbu!=u32(0) and schedule_rowv==endm:
        frame_action=u32(3) if schedule_fu==u32(14) else u32(2)
      else:
        schedule_ismu:u32=(IS_MARK_MASK>>schedule_fu)&u32(1)
        schedule_block_code:u32=u32(0)
        schedule_stepv:u32=u32(1)
        schedule_use_futureu:u32=u32(1)-schedule_ismu
        schedule_nextfidu:u32=schedule_fu

        if schedule_ismu!=u32(0):
          schedule_markv:u32=mark2 if ((SEL2_MASK>>schedule_fu)&u32(1))!=u32(0) else mark1
          if schedule_rowv==schedule_markv:
            schedule_block_code=(
              ((BLOCK_CODE_B0_MASK>>schedule_fu)&u32(1))
              |(((BLOCK_CODE_B1_MASK>>schedule_fu)&u32(1))<<u32(1))
              |(((BLOCK_CODE_B2_MASK>>schedule_fu)&u32(1))<<u32(2))
            )
            schedule_stepv=u32(2)+((OP_STEP3_MASK>>schedule_block_code)&u32(1))
            schedule_use_futureu=u32(0)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_isju:u32=(IS_JMARK_MASK>>schedule_fu)&u32(1)
        if schedule_isju!=u32(0):
          if schedule_rowv==jmark:
            frame_action=u32(1)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_child_rowu:u32=schedule_rowv+schedule_stepv
        schedule_fcvu:u32=u32(0)
        if schedule_use_futureu!=u32(0) and schedule_child_rowu<endm:
          schedule_fcvu=u32(1)
        frame_opcode=schedule_block_code|(schedule_fcvu<<u32(3))
        frame_raw=schedule_nextfidu|(schedule_child_rowu<<u32(5))

      if schedule_depth==0:
        root_action=frame_action
      else:
        parent_lane:int=(schedule_depth-1)&3
        parent_action_shift:u32=u32(parent_lane*8+4)
        pack_word|=frame_action<<parent_action_shift
        if (schedule_depth&3)==0:
          packed_schedule[pack_word_index]=pack_word
          pack_word_index+=1
          pack_word=u32(0)

      if frame_action>=u32(2):
        if schedule_depth>0 and (schedule_depth&3)!=0:
          packed_schedule[pack_word_index]=pack_word
        break

      opcode_shift:u32=u32((schedule_depth&3)*8)
      pack_word|=frame_opcode<<opcode_shift
      schedule_raw=frame_raw
      schedule_depth+=1

    if root_action==u32(2):
      results[i]=w_arr[i]
      return
    if root_action==u32(3):
      total+=u64(1) if ((root_a&~u32(1))!=u32(0)) else u64(0)
      results[i]=total*w_arr[i]
      return
    if root_action==u32(1):
      root_a&=~u32(1)
      if root_a==u32(0):
        results[i]=u64(0)
        return
      root_ld|=u32(1)

    sp:int=0
    ld[0]=root_ld
    rd[0]=root_rd
    col[0]=root_col
    avail[0]=root_a

    while True:
      a:u32=avail[sp]
      if a==u32(0):
        if sp==0:
          break
        sp-=1
        continue

      opcode_word:u32=packed_schedule[sp>>2]
      opcode:u32=(opcode_word>>u32((sp&3)*8))&u32(255)
      block_code:u32=opcode&u32(7)
      bit:u32=a&(u32(0)-a)
      avail[sp]=a^bit

      nld:u32=u32(0)
      nrd:u32=u32(0)
      if block_code!=u32(0):
        stepu:u32=u32(2)+((OP_STEP3_MASK>>block_code)&u32(1))
        addvu:u32=(OP_ADD1_MASK>>block_code)&u32(1)
        bLiu:u32=(
          ((OP_BL1_MASK>>block_code)&u32(1))
          |(((OP_BL2_MASK>>block_code)&u32(1))<<u32(1))
        )
        ktu:u32=(
          ((OP_KN3_MASK>>block_code)&u32(1))
          |(((OP_KN4_MASK>>block_code)&u32(1))<<u32(1))
        )
        bKu:u32=(n3&(u32(0)-(ktu&u32(1))))|(n4&(u32(0)-(ktu>>u32(1))))
        nld=((ld[sp]|bit)<<stepu)|addvu|bLiu
        nrd=((rd[sp]|bit)>>stepu)|bKu
      else:
        nld=(ld[sp]|bit)<<u32(1)
        nrd=(rd[sp]|bit)>>u32(1)
      ncol:u32=col[sp]|bit
      nf:u32=bm&~(nld|nrd|ncol)
      if nf==u32(0):
        continue
      if (opcode&u32(8))!=u32(0):
        if (bm&~((nld<<u32(1))|(nrd>>u32(1))|ncol))==u32(0):
          continue

      child_action:u32=(opcode>>u32(4))&u32(3)
      if child_action>=u32(2):
        if child_action==u32(2):
          total+=u64(1)
        else:
          total+=u64(1) if ((nf&~u32(1))!=u32(0)) else u64(0)
        continue
      if child_action==u32(1):
        nf&=~u32(1)
        if nf==u32(0):
          continue
        nld|=u32(1)

      sp+=1
      ld[sp]=nld
      rd[sp]=nrd
      col[sp]=ncol
      avail[sp]=nf
    results[i]=total*w_arr[i]

def launch_kernel_dfs_iter_gpu_static_maxd(
  selected_maxd:int,
  soa:TaskSoA,
  w_arr:List[u64],
  meta_next:List[u8],
  results:List[u64],
  m:int,
  board_mask_gpu:u32,
  n3_gpu:u32,
  n4_gpu:u32,
  grid_size:int,
  block_size:int
)->bool:
  if selected_maxd==14:
    kernel_dfs_iter_gpu_maxd14(
      gpu.raw(soa.ld_arr),gpu.raw(soa.rd_arr),gpu.raw(soa.col_arr),
      gpu.raw(soa.ctrl0_arr),gpu.raw(soa.free_arr),
      gpu.raw(soa.markctrl_arr),gpu.raw(w_arr),gpu.raw(meta_next),gpu.raw(results),
      m,board_mask_gpu,n3_gpu,n4_gpu,grid=grid_size,block=block_size
    )
    return True
  if selected_maxd==16:
    kernel_dfs_iter_gpu_maxd16(
      gpu.raw(soa.ld_arr),gpu.raw(soa.rd_arr),gpu.raw(soa.col_arr),
      gpu.raw(soa.ctrl0_arr),gpu.raw(soa.free_arr),
      gpu.raw(soa.markctrl_arr),gpu.raw(w_arr),gpu.raw(meta_next),gpu.raw(results),
      m,board_mask_gpu,n3_gpu,n4_gpu,grid=grid_size,block=block_size
    )
    return True
  if selected_maxd==18:
    kernel_dfs_iter_gpu_maxd18(
      gpu.raw(soa.ld_arr),gpu.raw(soa.rd_arr),gpu.raw(soa.col_arr),
      gpu.raw(soa.ctrl0_arr),gpu.raw(soa.free_arr),
      gpu.raw(soa.markctrl_arr),gpu.raw(w_arr),gpu.raw(meta_next),gpu.raw(results),
      m,board_mask_gpu,n3_gpu,n4_gpu,grid=grid_size,block=block_size
    )
    return True
  if selected_maxd==20:
    kernel_dfs_iter_gpu_maxd20(
      gpu.raw(soa.ld_arr),gpu.raw(soa.rd_arr),gpu.raw(soa.col_arr),
      gpu.raw(soa.ctrl0_arr),gpu.raw(soa.free_arr),
      gpu.raw(soa.markctrl_arr),gpu.raw(w_arr),gpu.raw(meta_next),gpu.raw(results),
      m,board_mask_gpu,n3_gpu,n4_gpu,grid=grid_size,block=block_size
    )
    return True
  if selected_maxd==21:
    kernel_dfs_iter_gpu_maxd21(
      gpu.raw(soa.ld_arr),gpu.raw(soa.rd_arr),gpu.raw(soa.col_arr),
      gpu.raw(soa.ctrl0_arr),gpu.raw(soa.free_arr),
      gpu.raw(soa.markctrl_arr),gpu.raw(w_arr),gpu.raw(meta_next),gpu.raw(results),
      m,board_mask_gpu,n3_gpu,n4_gpu,grid=grid_size,block=block_size
    )
    return True
  return False

def dfs_iter(
  meta:List[Tuple[int,int,int]],blockK:List[int],blockL:List[int],board_mask:int,
  functionid:int,ld:int,rd:int,col:int,row:int,free:int,
  jmark:int,endmark:int,mark1:int,mark2:int
)->u64:
  total:u64=u64(0)

  stack:List[Tuple[int,int,int,int,int,int]]=[(functionid,ld,rd,col,row,free)]

  while stack:
    functionid,ld,rd,col,row,free=stack.pop()

    if not free:
      continue

    next_funcid,funcptn,avail_flag=meta[functionid]
    avail:int=free

    if funcptn==5 and row==endmark:
      if functionid==14:
        total+=u64(1) if (avail>>1) else u64(0)
      else:
        total+=u64(1)
      continue

    step:int=1
    add1:int=0
    row_step:int=row+1

    use_blocks:bool=False
    use_future:bool=(avail_flag==1)

    local_next_funcid:int=functionid

    _blockK:int=0
    _blockL:int=0

    if funcptn in (0,1,2):
      at_mark:bool=(row==mark1) if funcptn in (0,2) else (row==mark2)

      if at_mark and avail:
        step=2 if funcptn in (0,1) else 3
        add1=1 if (funcptn==1 and functionid==20) else 0
        row_step=row+step

        _blockK=blockK[functionid]
        _blockL=blockL[functionid]

        use_blocks=True
        use_future=False
        local_next_funcid=next_funcid

    elif funcptn==3 and row==jmark:
      avail&=~1

      ld|=1

      local_next_funcid=next_funcid

      if not avail:
        continue

    elif funcptn==4 and row==mark1:
      stack.append((next_funcid,ld,rd,col,row,avail))
      continue

    if use_blocks:
      while avail:
        bit:int=avail&-avail
        avail&=avail-1

        nld:int=((ld|bit)<<step)|add1|_blockL
        nrd:int=((rd|bit)>>step)|_blockK
        ncol:int=col|bit

        nf:int=board_mask&~(nld|nrd|ncol)

        if nf:
          stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))

      continue

    if not use_future:
      while avail:
        bit:int=avail&-avail
        avail&=avail-1

        nld:int=(ld|bit)<<1
        nrd:int=(rd|bit)>>1
        ncol:int=col|bit

        nf:int=board_mask&~(nld|nrd|ncol)

        if nf:
          stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))

      continue

    if row_step>=endmark:
      while avail:
        bit:int=avail&-avail
        avail&=avail-1

        nld:int=(ld|bit)<<1
        nrd:int=(rd|bit)>>1
        ncol:int=col|bit

        nf:int=board_mask&~(nld|nrd|ncol)

        if nf:
          stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))

      continue

    while avail:
      bit:int=avail&-avail
      avail&=avail-1

      nld:int=(ld|bit)<<1
      nrd:int=(rd|bit)>>1
      ncol:int=col|bit

      nf:int=board_mask&~(nld|nrd|ncol)

      if not nf:
        continue

      if board_mask&~((nld<<1)|(nrd>>1)|ncol):
        stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))

  return total

def dfs(
    meta:List[Tuple[int,int,int]],
    blockK_by_funcid:List[int],blockL_by_funcid:List[int],
    board_mask:int,
    functionid:int,
    ld:int,rd:int,col:int,row:int,free:int,
    jmark:int,endmark:int,mark1:int,mark2:int)->u64:
  next_funcid,funcptn,avail_flag=meta[functionid]

  avail:int=free
  if not avail:
    return u64(0)

  total:u64=u64(0)

  if funcptn==5 and row==endmark:
    if functionid==14:
      return u64(1) if (avail>>1) else u64(0)
    return u64(1)

  step:int=1
  add1:int=0
  row_step:int=row+1

  use_blocks:bool=False
  use_future:bool=(avail_flag==1)

  local_next_funcid:int=functionid

  bK:int=0
  bL:int=0

  if funcptn in (0,1,2):
    at_mark:bool=(row==mark1) if funcptn in (0,2) else (row==mark2)

    if at_mark and avail:
      step=2 if funcptn in (0,1) else 3
      add1=1 if (funcptn==1 and functionid==20) else 0
      row_step=row+step

      bK=blockK_by_funcid[functionid]
      bL=blockL_by_funcid[functionid]

      use_blocks=True
      use_future=False
      local_next_funcid=next_funcid

  elif funcptn==3 and row==jmark:
    avail&=~1
    ld|=1
    local_next_funcid=next_funcid

    if not avail:
      return u64(0)

  elif funcptn==4 and row==mark1:
    return dfs(
      meta,
      blockK_by_funcid,
      blockL_by_funcid,
      board_mask,
      next_funcid,
      ld,rd,col,row,avail,
      jmark,endmark,mark1,mark2
    )

  if use_blocks:
    while avail:
      bit:int=avail&-avail
      avail&=avail-1

      nld:int=((ld|bit)<<step)|add1|bL
      nrd:int=((rd|bit)>>step)|bK
      ncol:int=col|bit

      nf:int=board_mask&~(nld|nrd|ncol)

      if nf:
        total+=dfs(
          meta,
          blockK_by_funcid,
          blockL_by_funcid,
          board_mask,
          local_next_funcid,
          nld,nrd,ncol,row_step,nf,
          jmark,endmark,mark1,mark2
        )

    return total

  if not use_future:
    while avail:
      bit:int=avail&-avail
      avail&=avail-1

      nld:int=(ld|bit)<<1
      nrd:int=(rd|bit)>>1
      ncol:int=col|bit

      nf:int=board_mask&~(nld|nrd|ncol)

      if nf:
        total+=dfs(
          meta,
          blockK_by_funcid,
          blockL_by_funcid,
          board_mask,
          local_next_funcid,
          nld,nrd,ncol,row_step,nf,
          jmark,endmark,mark1,mark2
        )

    return total

  if row_step>=endmark:
    while avail:
      bit:int=avail&-avail
      avail&=avail-1

      nld:int=(ld|bit)<<1
      nrd:int=(rd|bit)>>1
      ncol:int=col|bit

      nf:int=board_mask&~(nld|nrd|ncol)

      if nf:
        total+=dfs(
          meta,
          blockK_by_funcid,
          blockL_by_funcid,
          board_mask,
          local_next_funcid,
          nld,nrd,ncol,row_step,nf,
          jmark,endmark,mark1,mark2
        )

    return total

  while avail:
    bit:int=avail&-avail
    avail&=avail-1

    nld:int=(ld|bit)<<1
    nrd:int=(rd|bit)>>1
    ncol:int=col|bit

    nf:int=board_mask&~(nld|nrd|ncol)

    if not nf:
      continue

    if board_mask&~(((nld<<1)|(nrd>>1)|ncol)):
      total+=dfs(
        meta,
        blockK_by_funcid,
        blockL_by_funcid,
        board_mask,
        local_next_funcid,
        nld,nrd,ncol,row_step,nf,
        jmark,endmark,mark1,mark2
      )

  return total

def build_soa_for_range(
    N:int,
    constellations:List[Dict[str,int]],
    off:int,
    m:int,
    soa:TaskSoA,
    w_arr:List[u64]
)->Tuple[TaskSoA,List[u64]]:
    board_mask:int=(1<<N)-1

    small_mask:int=(1<<max(0,N-2))-1

    N1:int=N-1
    N2:int=N-2

    for t in range(m):
        constellation:Dict[str,int]=constellations[off+t]

        jmark:int=0
        mark1:int=0
        mark2:int=0

        start_ijkl:int=constellation["startijkl"]
        start:int=start_ijkl>>20
        ijkl:int=start_ijkl&((1<<20)-1)

        j,k,l=getj(ijkl),getk(ijkl),getl(ijkl)

        ld:int=constellation["ld"]>>1
        rd:int=constellation["rd"]>>1

        col:int=(constellation["col"]>>1)|(~small_mask)

        col&=board_mask

        LD:int=(1<<(N1-j))|(1<<(N1-l))

        ld|=LD>>(N-start)

        if start>k:
            rd|=(1<<(N1-(start-k+1)))

        if j>=2*N-33-start:
            rd|=(1<<(N1-j))<<(N2-start)

        free:int=board_mask&~(ld|rd|col)

        endmark:int=0
        target:int=0

        j_lt_N3:bool=(j<N-3)
        j_eq_N3:bool=(j==N-3)
        j_eq_N2:bool=(j==N-2)

        k_lt_l:bool=(k<l)
        start_lt_k:bool=(start<k)
        start_lt_l:bool=(start<l)

        l_eq_kp1:bool=(l==k+1)
        k_eq_lp1:bool=(k==l+1)

        j_gate:bool=(j>2*N-34-start)

        if j_lt_N3:
            jmark=j+1

            endmark=N2

            if j_gate:
                if k_lt_l:
                    mark1,mark2=k-1,l-1

                    if start_lt_l:
                        if start_lt_k:
                            target:int=0 if (not l_eq_kp1) else 4
                        else:
                            target=1
                    else:
                        target=2
                else:
                    mark1,mark2=l-1,k-1

                    if start_lt_k:
                        if start_lt_l:
                            target=5 if (not k_eq_lp1) else 7
                        else:
                            target=6
                    else:
                        target=2
            else:
                if k_lt_l:
                    mark1,mark2=k-1,l-1
                    target=8 if (not l_eq_kp1) else 9
                else:
                    mark1,mark2=l-1,k-1
                    target=10 if (not k_eq_lp1) else 11

        elif j_eq_N3:
            endmark=N2

            if k_lt_l:
                mark1,mark2=k-1,l-1

                if start_lt_l:
                    if start_lt_k:
                        target=12 if (not l_eq_kp1) else 15
                    else:
                        mark2=l-1
                        target=13
                else:
                    target=14
            else:
                mark1,mark2=l-1,k-1

                if start_lt_k:
                    if start_lt_l:
                        target=16 if (not k_eq_lp1) else 18
                    else:
                        mark2=k-1
                        target=17
                else:
                    target=14

        elif j_eq_N2:
            if k_lt_l:
                endmark=N2
                if start_lt_l:
                    if start_lt_k:
                        mark1=k-1
                        if not l_eq_kp1:
                            mark2=l-1
                            target=19
                        else:
                            target=22
                    else:
                        mark2=l-1
                        target=20
                else:
                    target=21
            else:
                if start_lt_k:
                    if start_lt_l:
                        if k<N2:
                            mark1,endmark=l-1,N2
                            if not k_eq_lp1:
                                mark2=k-1
                                target=23
                            else:
                                target=24
                        else:
                            if l!=(N-3):
                                mark2,endmark=l-1,N-3
                                target=20
                            else:
                                endmark=N-4
                                target=21
                    else:
                        if k!=N2:
                            mark2,endmark=k-1,N2
                            target=25
                        else:
                            endmark=N-3
                            target=21
                else:
                    endmark=N2
                    target=21

        else:
            endmark=N2
            if start>k:
                target=26
            else:
                mark1=k-1
                target=27

        soa.ld_arr[t]=u32(ld)
        soa.rd_arr[t]=u32(rd)
        soa.col_arr[t]=u32(col)
        soa.row_arr[t]=start
        soa.ctrl0_arr[t]=u32(target)|(u32(start)<<u32(5))
        soa.free_arr[t]=u32(free)
        soa.markctrl_arr[t]=(
          u32(jmark&31)
          |(u32(endmark&31)<<u32(5))
          |(u32(mark1&31)<<u32(10))
          |(u32(mark2&31)<<u32(15))
        )
        soa.jmark_arr[t]=jmark
        soa.end_arr[t]=endmark
        soa.mark1_arr[t]=mark1
        soa.mark2_arr[t]=mark2
        soa.funcid_arr[t]=target
        soa.ijkl_arr[t]=ijkl

    @par
    for t in range(m):
        w_arr[t]=symmetry(soa.ijkl_arr[t],N)

    return soa,w_arr

def bc_id(N:int,j:int)->int:
  if j<N-3:
    return 0   # B / normal
  if j==N-3:
    return 1   # SQd2
  if j==N-2:
    return 2   # SQd1
  return 3     # SQd0

def bc_name(cid:int,N:int)->str:
  if cid==0:
    return f"B(j<{N-3})"
  if cid==1:
    return f"SQd2(j={N-3})"
  if cid==2:
    return f"SQd1(j={N-2})"
  return f"SQd0(j>{N-2})"

def fid_name(fid:int)->str:
  names:List[str]=[
    "SQBkBlBjrB","SQBlBjrB","SQBjrB","SQB",
    "SQBklBjrB","SQBlBkBjrB","SQBkBjrB","SQBlkBjrB",
    "SQBjlBkBlBjrB","SQBjlBklBjrB","SQBjlBlBkBjrB","SQBjlBlkBjrB",
    "SQd2BkBlB","SQd2BlB","SQd2B","SQd2BklB","SQd2BlBkB",
    "SQd2BkB","SQd2BlkB","SQd1BkBlB","SQd1BlB","SQd1B",
    "SQd1BklB","SQd1BlBkB","SQd1BlkB","SQd1BkB","SQd0B","SQd0BkB"
  ]
  if fid>=0 and fid<28:
    return names[fid]
  return "UNKNOWN"

def bc_expected_fid(cid:int,fid:int)->bool:
  if cid==0:
    return fid>=0 and fid<=11
  if cid==1:
    return fid>=12 and fid<=18
  if cid==2:
    return fid>=19 and fid<=25
  if cid==3:
    return fid==26 or fid==27
  return False

def bc_expected_endmark(N:int,cid:int,endmark:int)->bool:
  if cid==0:
    return endmark==N-2
  if cid==1:
    return endmark==N-2
  if cid==2:
    return endmark==N-2 or endmark==N-3 or endmark==N-4
  if cid==3:
    return endmark==N-2
  return False

def diagnose_boundary_classification(N:int,constellations:List[Dict[str,int]])->None:
  m:int=len(constellations)
  if m==0:
    print(f"[bc-summary] N={N} constellations=0")
    return

  soa:TaskSoA=TaskSoA(m)
  w_arr:List[u64]=[u64(0)]*m
  soa,w_arr=build_soa_for_range(N,constellations,0,m,soa,w_arr)

  case_cnt:List[int]=[0]*4
  case_free0:List[int]=[0]*4
  case_w2:List[int]=[0]*4
  case_w4:List[int]=[0]*4
  case_w8:List[int]=[0]*4
  case_bad_fid:List[int]=[0]*4
  case_bad_end:List[int]=[0]*4
  fid_cnt:List[int]=[0]*28
  case_fid_cnt:List[int]=[0]*(4*28)
  case_start_cnt:List[int]=[0]*(4*(N+1))
  case_end_cnt:List[int]=[0]*(4*(N+1))

  bad_printed:int=0

  for idx in range(m):
    ijkl:int=soa.ijkl_arr[idx]
    j:int=getj(ijkl)
    cid:int=bc_id(N,j)
    fid:int=soa.funcid_arr[idx]
    endmark:int=soa.end_arr[idx]
    start:int=soa.row_arr[idx]

    case_cnt[cid]+=1

    if fid>=0 and fid<28:
      fid_cnt[fid]+=1
      case_fid_cnt[cid*28+fid]+=1
    else:
      case_bad_fid[cid]+=1

    if not bc_expected_fid(cid,fid):
      case_bad_fid[cid]+=1
      if bad_printed<20:
        print(f"[bc-error-fid] idx={idx} case={bc_name(cid,N)} i={geti(ijkl)} j={j} k={getk(ijkl)} l={getl(ijkl)} fid={fid} {fid_name(fid)} start={start} end={endmark}")
        bad_printed+=1

    if not bc_expected_endmark(N,cid,endmark):
      case_bad_end[cid]+=1
      if bad_printed<20:
        print(f"[bc-error-end] idx={idx} case={bc_name(cid,N)} i={geti(ijkl)} j={j} k={getk(ijkl)} l={getl(ijkl)} fid={fid} {fid_name(fid)} start={start} end={endmark}")
        bad_printed+=1

    if soa.free_arr[idx]==u32(0):
      case_free0[cid]+=1

    w:int=int(w_arr[idx])
    if w==2:
      case_w2[cid]+=1
    elif w==4:
      case_w4[cid]+=1
    elif w==8:
      case_w8[cid]+=1

    if start>=0 and start<=N:
      case_start_cnt[cid*(N+1)+start]+=1
    if endmark>=0 and endmark<=N:
      case_end_cnt[cid*(N+1)+endmark]+=1

  print(f"[bc-summary] N={N} constellations={m} N-3={N-3} N-2={N-2} signature_prune_disabled={1 if DISABLE_CONSTELLATION_SIGNATURE_PRUNE else 0}")

  for cid in range(4):
    print(f"[bc-case] {bc_name(cid,N)} count={case_cnt[cid]} free0={case_free0[cid]} w2={case_w2[cid]} w4={case_w4[cid]} w8={case_w8[cid]} bad_fid={case_bad_fid[cid]} bad_end={case_bad_end[cid]}")

    line:str="[bc-start] " + bc_name(cid,N)
    for r in range(N+1):
      v:int=case_start_cnt[cid*(N+1)+r]
      if v>0:
        line += f" r{r}={v}"
    print(line)

    line="[bc-end]   " + bc_name(cid,N)
    for r in range(N+1):
      v:int=case_end_cnt[cid*(N+1)+r]
      if v>0:
        line += f" e{r}={v}"
    print(line)

    for fid in range(28):
      c:int=case_fid_cnt[cid*28+fid]
      if c>0:
        print(f"[bc-fid] {bc_name(cid,N)} fid={fid} {fid_name(fid)} count={c}")

  print("[bc-fid-total]")
  for fid in range(28):
    if fid_cnt[fid]>0:
      print(f"[bc-fid-total] fid={fid} {fid_name(fid)} count={fid_cnt[fid]}")

def diagnose_solution_by_boundary(N:int,constellations:List[Dict[str,int]])->None:
  m:int=len(constellations)
  if m==0:
    print(f"[bc-sol-summary] N={N} constellations=0")
    return

  soa:TaskSoA=TaskSoA(m)
  w_arr:List[u64]=[u64(0)]*m
  soa,w_arr=build_soa_for_range(N,constellations,0,m,soa,w_arr)

  case_cnt:List[int]=[0]*4
  case_total:List[int]=[0]*4
  case_fid_cnt:List[int]=[0]*(4*28)
  case_fid_total:List[int]=[0]*(4*28)
  all_total:int=0

  for idx in range(m):
    ijkl:int=soa.ijkl_arr[idx]
    j:int=getj(ijkl)
    cid:int=bc_id(N,j)
    fid:int=soa.funcid_arr[idx]
    sol:int=constellations[idx]["solutions"]

    case_cnt[cid]+=1
    case_total[cid]+=sol
    all_total+=sol

    if fid>=0 and fid<28:
      case_fid_cnt[cid*28+fid]+=1
      case_fid_total[cid*28+fid]+=sol

  print(f"[bc-sol-summary] N={N} constellations={m} total={all_total}")
  for cid in range(4):
    print(f"[bc-sol-case] {bc_name(cid,N)} count={case_cnt[cid]} total={case_total[cid]}")
    for fid in range(28):
      c:int=case_fid_cnt[cid*28+fid]
      t:int=case_fid_total[cid*28+fid]
      if c>0 or t>0:
        print(f"[bc-sol-fid] {bc_name(cid,N)} fid={fid} {fid_name(fid)} count={c} total={t}")

def auto_sort_mode(N:int)->int:
  if N==20 or N==21:
    return 9
  return 0

def validate_chunk_range(label:str,start:int,end:int,total:int)->bool:
  ok:bool=True
  if start<0:
    print(f"[cross-stripe-safe][error] {label}: start < 0 start={start} total={total}")
    ok=False
  if end<start:
    print(f"[cross-stripe-safe][error] {label}: end < start start={start} end={end} total={total}")
    ok=False
  if end>total:
    print(f"[cross-stripe-safe][error] {label}: end > total start={start} end={end} total={total}")
    ok=False
  if ok and start==end:
    print(f"[cross-stripe-safe][warn] {label}: empty range start={start} end={end} total={total}")
  return ok

def validate_reordered_count(label:str,expected:int,actual:int)->bool:
  if expected!=actual:
    print(f"[stripe-reorder][error] {label}: reordered count mismatch expected={expected} actual={actual}")
    return False
  return True

def validate_reordered_indices(label:str,expected:int,idxs:List[int])->bool:
  if not validate_reordered_count(label,expected,len(idxs)):
    return False
  seen:List[int]=[0]*expected
  for v in idxs:
    if v<0 or v>=expected:
      print(f"[cross-stripe-safe][error] {label}: index out of range idx={v} expected={expected}")
      return False
    if seen[v]!=0:
      print(f"[cross-stripe-safe][error] {label}: duplicated index idx={v}")
      return False
    seen[v]=1
  missing:int=0
  first_missing:int=-1
  for i in range(expected):
    if seen[i]==0:
      missing+=1
      if first_missing<0:
        first_missing=i
  if missing!=0:
    print(f"[cross-stripe-safe][error] {label}: missing count={missing} first_missing={first_missing}")
    return False
  return True

def popcount_int(x:int)->int:
  c:int=0
  while x:
    x&=x-1
    c+=1
  return c

def exec_solutions(N:int,constellations:List[Dict[str,int]],use_gpu:bool,gpu_block:int=32,gpu_max_blocks:int=484,gpu_log_level:int=0,gpu_sort_mode:int=-1,cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,reorder_only:bool=False,chunk_only:bool=False,debug_chunk_start:int=0,debug_chunk_count:int=1)->None:
  N1:int=N-1
  N2:int=N-2
  board_mask:int=(1<<N)-1

  if gpu_sort_mode < 0:
    gpu_sort_mode = auto_sort_mode(N)

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
  F=len(func_meta)
  funcptn_by_fid:List[int]=[0]*F
  for f,(nxt,ptn,aflag) in enumerate(func_meta):
      funcptn_by_fid[f]=ptn
  is_base=[0]*F   # ptn==5
  is_jmark=[0]*F   # ptn==3
  is_mark=[0]*F   # ptn in {0,1,2}

  mark_sel=[0]*F  # 0:none 1:mark1 2:mark2
  mark_step=[1]*F  # 1 or 2 or 3
  mark_add1=[0]*F  # 0/1
  for f,(nxt,ptn,aflag) in enumerate(func_meta):
      if ptn==5:
          is_base[f]=1
      elif ptn==3:
          is_jmark[f]=1
      elif ptn==0 or ptn==1 or ptn==2:
          is_mark[f]=1
          if ptn==1:
              mark_sel[f]=2
              mark_step[f]=2
              if f==20:
                  mark_add1[f]=1
          else:
              mark_sel[f]=1
              mark_step[f]=2 if ptn==0 else 3

  n3=1<<max(0,N-3)   # 念のため負シフト防止
  n4=1<<max(0,N-4)   # N3,N4とは違います
  m=len(constellations)
  BLOCK=gpu_block
  MAX_BLOCKS=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS=BLOCK*MAX_BLOCKS
  m_all=len(constellations)

  if use_gpu:
    stripe_chunks:bool=(gpu_sort_mode==5 or gpu_sort_mode==6 or gpu_sort_mode==7 or gpu_sort_mode==8 or gpu_sort_mode==9 or gpu_sort_mode==10)
    balanced_stripe:bool=(gpu_sort_mode==7 or gpu_sort_mode==8)
    cross_stripe:bool=(gpu_sort_mode==9 or gpu_sort_mode==10)
    local_sort_mode:int=gpu_sort_mode
    work_constellations:List[Dict[str,int]]=constellations
    if reorder_only and not stripe_chunks:
      print(f"[reorder-only] N={N} sort_mode={gpu_sort_mode} stripe=0 original={m_all} steps={STEPS} ok")
      if m_all>0:
        constellations[0]["solutions"]=0
      return
    if stripe_chunks:
      n_chunks_est:int=(m_all + STEPS - 1)//STEPS
      reordered:List[Dict[str,int]]=[]
      validate_reorder:bool=(cross_stripe_safe or reorder_only)
      reordered_idx:List[int]=[]
      if cross_stripe:
        out_ch:int=0
        while out_ch<n_chunks_est:
          base:int=0
          while base*n_chunks_est<STEPS:
            src_ch:int=0
            while src_ch<n_chunks_est:
              rem:int=(src_ch + out_ch) % n_chunks_est
              within:int=base*n_chunks_est + rem
              if within<STEPS:
                idx:int=src_ch*STEPS+within
                if idx<m_all:
                  reordered.append(constellations[idx])
                  if validate_reorder:
                    reordered_idx.append(idx)
              src_ch+=1
            base+=1
          out_ch+=1
      elif balanced_stripe:
        out_ch:int=0
        while out_ch<n_chunks_est:
          slot:int=0
          while slot<STEPS:
            src_ch:int=slot % n_chunks_est
            base:int=slot // n_chunks_est
            within:int=(out_ch + base*n_chunks_est) % STEPS
            idx:int=src_ch*STEPS+within
            if idx<m_all:
              reordered.append(constellations[idx])
              if validate_reorder:
                reordered_idx.append(idx)
            slot+=1
          out_ch+=1
      else:
        within:int=0
        while within<STEPS:
          ch:int=0
          while ch<n_chunks_est:
            idx:int=ch*STEPS+within
            if idx<m_all:
              reordered.append(constellations[idx])
              if validate_reorder:
                reordered_idx.append(idx)
            ch+=1
          within+=1
      work_constellations=reordered
      if not validate_reordered_count("stripe_reorder",m_all,len(work_constellations)):
        return
      if validate_reorder:
        if not validate_reordered_indices("stripe_reorder",m_all,reordered_idx):
          return
      if reorder_only:
        print(f"[reorder-only] N={N} sort_mode={gpu_sort_mode} original={m_all} reordered={len(work_constellations)} chunks={n_chunks_est} steps={STEPS} ok")
        if m_all>0:
          constellations[0]["solutions"]=0
        return
      if gpu_sort_mode==5 or gpu_sort_mode==8 or gpu_sort_mode==10:
        local_sort_mode=4
      else:
        local_sort_mode=0
    if gpu_log_level>=1:
      print(f"[gpu-config] N={N} original=1 mixed32=1 hotpath=1 trunk75={BLOCK}x{MAX_BLOCKS} sort_mode={gpu_sort_mode} local_sort={local_sort_mode} stripe={1 if stripe_chunks else 0} balanced={1 if balanced_stripe else 0} cross={1 if cross_stripe else 0} cross_safe={1 if cross_stripe_safe else 0} reorder_only={1 if reorder_only else 0} chunk_only={1 if chunk_only else 0} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} log_level={gpu_log_level}")
    soa:TaskSoA=TaskSoA(STEPS)
    results:List[u64]=[u64(0)]*STEPS
    gpu_total:int=0
    w_arr:List[u64]=[u64(0)]*STEPS

    sort_soa:TaskSoA=TaskSoA(STEPS)
    sort_w_arr:List[u64]=[u64(0)]*STEPS
    order:List[int]=[0]*STEPS

    meta_next: List[u8] = [ u8(1),u8(2),u8(3),u8(3),u8(2),u8(6),u8(2),u8(2), u8(0),u8(4),u8(5),u8(7),u8(13),u8(14),u8(14),u8(14), u8(17),u8(14),u8(14),u8(20),u8(21),u8(21),u8(21),u8(25), u8(21),u8(21),u8(26),u8(26) ]
    off = 0
    board_mask_gpu:u32=u32(board_mask)
    n3_gpu:u32=u32(1)<<u32(N-3)
    n4_gpu:u32=u32(1)<<u32(N-4)
    chunks:int=0
    executed_chunks:int=0
    if chunk_only:
      if debug_chunk_start<0:
        debug_chunk_start=0
      if debug_chunk_count<=0:
        debug_chunk_count=1
      if gpu_log_level>=1:
        print(f"[chunk-only] mode=7 executes selected chunks only; start={debug_chunk_start} count={debug_chunk_count}")
    while off < m_all:
      m = min(STEPS, m_all - off)
      chunk_index:int=chunks
      chunks+=1
      if chunk_only:
        run_this_chunk:bool=(chunk_index>=debug_chunk_start and chunk_index<debug_chunk_start+debug_chunk_count)
        if not run_this_chunk:
          if gpu_log_level>=2:
            print(f"[gpu-chunk-skip] N={N} chunk={chunk_index} off={off} m={m}")
          off += m
          continue
      executed_chunks+=1
      if cross_stripe_safe:
        if not validate_chunk_range("gpu_chunk",off,off+m,m_all):
          return
      if gpu_log_level>=2:
        t0=datetime.now()
      build_soa_for_range(N,work_constellations, off, m,soa,w_arr)
      if gpu_log_level>=2:
        t1=datetime.now()
      use_sorted:bool=(local_sort_mode==1 or local_sort_mode==2 or local_sort_mode==3 or local_sort_mode==4)
      if gpu_log_level>=2:
        ts0=datetime.now()
      if use_sorted:
        nb:int=28
        if local_sort_mode==2:
          nb=6
        if local_sort_mode==3:
          nb=24  # funcptn 6種 x work bucket 4種
        if local_sort_mode==4:
          nb=48  # funcptn 6種 x work bucket 8種
        counts:List[int]=[0]*64
        pos:List[int]=[0]*64
        cur:List[int]=[0]*64
        for i in range(m):
          fid0:int=soa.funcid_arr[i]
          key:int=fid0
          if local_sort_mode==2:
            key=funcptn_by_fid[fid0]
          if local_sort_mode==3:
            ptn:int=funcptn_by_fid[fid0]
            depth:int=soa.end_arr[i]-soa.row_arr[i]
            if depth<0:
              depth=0
            pc:int=popcount_int(int(soa.free_arr[i]))
            wb:int=0
            if pc>=3:
              wb+=1
            if depth>=12:
              wb+=2
            key=ptn*4+wb
          if local_sort_mode==4:
            ptn:int=funcptn_by_fid[fid0]
            depth:int=soa.end_arr[i]-soa.row_arr[i]
            if depth<0:
              depth=0
            pc:int=popcount_int(int(soa.free_arr[i]))
            wb:int=0
            if pc>=2:
              wb+=1
            if pc>=4:
              wb+=2
            if depth>=10:
              wb+=4
            key=ptn*8+wb
          counts[key]+=1
        run:int=0
        for b in range(nb):
          pos[b]=run
          cur[b]=run
          run+=counts[b]
        for i in range(m):
          fid0:int=soa.funcid_arr[i]
          key:int=fid0
          if local_sort_mode==2:
            key=funcptn_by_fid[fid0]
          if local_sort_mode==3:
            ptn:int=funcptn_by_fid[fid0]
            depth:int=soa.end_arr[i]-soa.row_arr[i]
            if depth<0:
              depth=0
            pc:int=popcount_int(int(soa.free_arr[i]))
            wb:int=0
            if pc>=3:
              wb+=1
            if depth>=12:
              wb+=2
            key=ptn*4+wb
          if local_sort_mode==4:
            ptn:int=funcptn_by_fid[fid0]
            depth:int=soa.end_arr[i]-soa.row_arr[i]
            if depth<0:
              depth=0
            pc:int=popcount_int(int(soa.free_arr[i]))
            wb:int=0
            if pc>=2:
              wb+=1
            if pc>=4:
              wb+=2
            if depth>=10:
              wb+=4
            key=ptn*8+wb
          p:int=cur[key]
          order[p]=i
          cur[key]+=1
        for p in range(m):
          q:int=order[p]
          sort_soa.ld_arr[p]=soa.ld_arr[q]
          sort_soa.rd_arr[p]=soa.rd_arr[q]
          sort_soa.col_arr[p]=soa.col_arr[q]
          sort_soa.row_arr[p]=soa.row_arr[q]
          sort_soa.ctrl0_arr[p]=soa.ctrl0_arr[q]
          sort_soa.free_arr[p]=soa.free_arr[q]
          sort_soa.markctrl_arr[p]=soa.markctrl_arr[q]
          sort_soa.funcid_arr[p]=soa.funcid_arr[q]
          sort_soa.ijkl_arr[p]=soa.ijkl_arr[q]
          sort_w_arr[p]=w_arr[q]
      if gpu_log_level>=2:
        ts1=datetime.now()
      GRID = (m + BLOCK - 1) // BLOCK

      if use_sorted:
        required_maxd:int=max_schedule_depth_of_tasks(sort_soa,m,meta_next)
        selected_maxd:int=select_static_maxd(required_maxd)
        if gpu_log_level>=2:
          print(f"[maxd-dispatch] N={N} scope=exec chunk={chunk_index} m={m} required_maxd={required_maxd} selected_MAXD={selected_maxd} schedule_words={packed_schedule_words_for_maxd(selected_maxd)} stack_bytes_per_thread={packed_stack_bytes_per_thread(selected_maxd)} capacity_check=OK")
        if not launch_kernel_dfs_iter_gpu_static_maxd(selected_maxd,sort_soa,sort_w_arr,meta_next,results,m,board_mask_gpu,n3_gpu,n4_gpu,GRID,BLOCK):
          print(f"[maxd-error] unsupported required_maxd={required_maxd}; supported maximum is 21")
          return
      else:
        required_maxd:int=max_schedule_depth_of_tasks(soa,m,meta_next)
        selected_maxd:int=select_static_maxd(required_maxd)
        if gpu_log_level>=2:
          print(f"[maxd-dispatch] N={N} scope=exec chunk={chunk_index} m={m} required_maxd={required_maxd} selected_MAXD={selected_maxd} schedule_words={packed_schedule_words_for_maxd(selected_maxd)} stack_bytes_per_thread={packed_stack_bytes_per_thread(selected_maxd)} capacity_check=OK")
        if not launch_kernel_dfs_iter_gpu_static_maxd(selected_maxd,soa,w_arr,meta_next,results,m,board_mask_gpu,n3_gpu,n4_gpu,GRID,BLOCK):
          print(f"[maxd-error] unsupported required_maxd={required_maxd}; supported maximum is 21")
          return

      if gpu_log_level>=2:
        t2=datetime.now()
      chunk_total:int=0
      for i in range(m):
        chunk_total += int(results[i])
      gpu_total += chunk_total
      if gpu_log_level>=2:
        t3=datetime.now()
        print(f"[gpu-chunk] N={N} chunk={chunk_index} off={off} m={m} grid={GRID} sort={gpu_sort_mode}/{local_sort_mode} build={str(t1-t0)[:-3]} sort_time={str(ts1-ts0)[:-3]} kernel={str(t2-ts1)[:-3]} sum={str(t3-t2)[:-3]} partial_total={chunk_total}")
      off += m

    if m_all>0:
      constellations[0]["solutions"] = gpu_total
    if gpu_log_level>=1:
      print(f"[gpu-summary] N={N} constellations={m_all} chunks={chunks} executed_chunks={executed_chunks} direct_total=1 stripe={1 if stripe_chunks else 0} balanced={1 if balanced_stripe else 0} cross={1 if cross_stripe else 0} cross_safe={1 if cross_stripe_safe else 0} chunk_only={1 if chunk_only else 0}")
    return
  else:
    soa:TaskSoA = TaskSoA(m_all)
    results: List[u64] = [u64(0)] * m_all
    results_all: List[u64] = [u64(0)] * m_all
    w_arr: List[u64] = [u64(0)] * m_all

    size=max(FID.values())+1
    blockK_by_funcid=[0]*size
    blockL_by_funcid=[0,1,0,0,1,1,0,2,0,0,0,0,0,1,0,1,1,0,2,0,0,0,1,1,2,0,0,0]
    for fn,cat in FUNC_CATEGORY.items():# FUNC_CATEGORY: {関数名: 3 or 4 or 0}
      fid=FID[fn]
      blockK_by_funcid[fid]=n3 if cat==3 else (n4 if cat==4 else 0)

    m_all = len(constellations) # CPUは全件を1回で SoA + w_arr を作る（これがないと壊れる）
    if m_all == 0:
      return
    soa, w_arr = build_soa_for_range(N,constellations, 0, m_all, soa, w_arr)
    results:List[u64] = [u64(0)] * m_all
    @par
    for i in range(m_all):
      use_itter = True
      if use_itter:
        cnt:u64 = dfs_iter(
            func_meta,
            blockK_by_funcid,blockL_by_funcid,
            board_mask,
            soa.funcid_arr[i],
            int(soa.ld_arr[i]), int(soa.rd_arr[i]), int(soa.col_arr[i]),
            soa.row_arr[i],int(soa.free_arr[i]),
            soa.jmark_arr[i], soa.end_arr[i],
            soa.mark1_arr[i], soa.mark2_arr[i])
      else:
        cnt:u64 = dfs(
            func_meta,
            blockK_by_funcid,blockL_by_funcid,
            board_mask,
            soa.funcid_arr[i],
            int(soa.ld_arr[i]), int(soa.rd_arr[i]), int(soa.col_arr[i]),
            soa.row_arr[i],int(soa.free_arr[i]),
            soa.jmark_arr[i], soa.end_arr[i],
            soa.mark1_arr[i], soa.mark2_arr[i])
      results[i]=cnt*w_arr[i]
  out = results
  for i, constellation in enumerate(constellations):
    constellation["solutions"] = int(out[i])

def mix64(x:u64)->u64:
  x=(x^(x>>u64(30)))*u64(0xBF58476D1CE4E5B9)
  x=(x^(x>>u64(27)))*u64(0x94D049BB133111EB)
  x^=(x>>u64(31))
  return x

def gen_list(cnt:int,seed:u64)->List[u64]:
  out:List[u64]=[]
  s:u64=seed
  for _ in range(cnt):
    s=s+u64(0x9E3779B97F4A7C15)
    out.append(mix64(s))
  return out

def init_zobrist(N:int,zobrist_hash_tables: Dict[int, Dict[str, List[u64]]])->Dict[str,List[u64]]:
  if N in zobrist_hash_tables:
    return zobrist_hash_tables[N]
  base_seed:u64=(u64(0xC0D0_0000_0000_0000)^(u64(N)<<u64(32)))
  tbl:Dict[str,List[u64]]={
    'ld':gen_list(N,base_seed^u64(0x01)),
    'rd':gen_list(N,base_seed^u64(0x02)),
    'col':gen_list(N,base_seed^u64(0x03)),
    'LD':gen_list(N,base_seed^u64(0x04)),
    'RD':gen_list(N,base_seed^u64(0x05)),
    'row':gen_list(N,base_seed^u64(0x06)),
    'queens':gen_list(N,base_seed^u64(0x07)),
    'k':gen_list(N,base_seed^u64(0x08)),
    'l':gen_list(N,base_seed^u64(0x09)),
  }
  zobrist_hash_tables[N]=tbl
  return tbl

def zobrist_hash(N:int, ld: int, rd: int, col: int, row: int, queens: int, k: int, l: int, LD: int, RD: int,zobrist_hash_tables:Dict[int, Dict[str, List[u64]]]) -> u64:
  tbl: Dict[str, List[u64]] = init_zobrist(N,zobrist_hash_tables)

  ld_tbl  = tbl["ld"]    # List[u64]
  rd_tbl  = tbl["rd"]    # List[u64]
  col_tbl = tbl["col"]   # List[u64]
  LD_tbl  = tbl["LD"]    # List[u64]
  RD_tbl  = tbl["RD"]    # List[u64]
  row_tbl = tbl["row"]   # List[u64]
  q_tbl   = tbl["queens"]# List[u64]
  k_tbl   = tbl["k"]     # List[u64]
  l_tbl   = tbl["l"]     # List[u64]

  mask: u64 = (u64(1) << u64(N)) - u64(1)

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

def to_ijkl(i:int,j:int,k:int,l:int)->int:return (i<<15)+(j<<10)+(k<<5)+l
def mirvert(ijkl:int,N:int)->int:return to_ijkl(N-1-geti(ijkl),N-1-getj(ijkl),getl(ijkl),getk(ijkl))
def ffmin(a:int,b:int)->int:return min(a,b)
def geti(ijkl:int)->int:return (ijkl>>15)&0x1F
def getj(ijkl:int)->int:return (ijkl>>10)&0x1F
def getk(ijkl:int)->int:return (ijkl>>5)&0x1F
def getl(ijkl:int)->int:return ijkl&0x1F

def rot90(ijkl:int,N:int)->int:return ((N-1-getk(ijkl))<<15)+((N-1-getl(ijkl))<<10)+(getj(ijkl)<<5)+geti(ijkl)
def rot180(ijkl:int,N:int)->int:return ((N-1-getj(ijkl))<<15)+((N-1-geti(ijkl))<<10)+((N-1-getl(ijkl))<<5)+(N-1-getk(ijkl))
def symmetry(ijkl:int,N:int)->u64:return u64(2) if symmetry90(ijkl,N) else u64(4) if geti(ijkl)==N-1-getj(ijkl) and getk(ijkl)==N-1-getl(ijkl) else u64(8)
def symmetry90(ijkl:int,N:int)->bool:return ((geti(ijkl)<<15)+(getj(ijkl)<<10)+(getk(ijkl)<<5)+getl(ijkl))==(((N-1-getk(ijkl))<<15)+((N-1-getl(ijkl))<<10)+(getj(ijkl)<<5)+geti(ijkl))

def check_rotations(ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
  return any(rot in ijkl_list for rot in [((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i,((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k),(l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)])

jasmin_cache_global:Dict[Tuple[int,int],int]={}

def get_jasmin(N:int,c:int)->int:
  key=(c,N)
  if key in jasmin_cache_global:
    return jasmin_cache_global[key]
  result=jasmin(c,N)
  jasmin_cache_global[key]=result
  return result

def jasmin(ijkl:int,N:int)->int:
  arg=0
  min_val=ffmin(getj(ijkl),N-1-getj(ijkl))
  if ffmin(geti(ijkl),N-1-geti(ijkl))<min_val:
    arg=2
    min_val=ffmin(geti(ijkl),N-1-geti(ijkl))
  if ffmin(getk(ijkl),N-1-getk(ijkl))<min_val:
    arg=3
    min_val=ffmin(getk(ijkl),N-1-getk(ijkl))
  if ffmin(getl(ijkl),N-1-getl(ijkl))<min_val:
    arg=1
    min_val=ffmin(getl(ijkl),N-1-getl(ijkl))
  _rot90=rot90
  for _ in range(arg):
    ijkl=_rot90(ijkl,N)
  if getj(ijkl)<N-1-getj(ijkl):
    ijkl=mirvert(ijkl,N)
  return ijkl

def set_pre_queens_cached(
  N:int,
  ijkl_list:Set[int],
  subconst_cache:set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],
  ld:int, rd:int, col:int,
  k:int, l:int,
  row:int, queens:int,
  LD:int, RD:int,
  counter:List[int],
  constellations:List[Dict[str,int]],
  preset_queens:int,
  visited:Set[int],
  constellation_signatures:Set[Tuple[int,int,int,int,int,int]],
  zobrist_hash_tables: Dict[int, Dict[str, List[u64]]]
)->Tuple[Set[int], Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]], List[Dict[str,int]], int]:
  if preset_queens>=7:
    ijkl_list, subconst_cache, constellations, preset_queens = set_pre_queens(
      N, ijkl_list, subconst_cache,
      ld, rd, col,
      k, l,
      row, queens,
      LD, RD,
      counter, constellations, preset_queens,
      visited, constellation_signatures,
      zobrist_hash_tables
    )
    return ijkl_list, subconst_cache, constellations, preset_queens

  key:Tuple[int,int,int,int,int,int,int,int,int,int,int] = (
    ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens
  )

  if key in subconst_cache:
    return ijkl_list, subconst_cache, constellations, preset_queens

  subconst_cache.add(key)

  ijkl_list, subconst_cache, constellations, preset_queens = set_pre_queens(
    N, ijkl_list, subconst_cache,
    ld, rd, col,
    k, l,
    row, queens,
    LD, RD,
    counter, constellations, preset_queens,
    visited, constellation_signatures,
    zobrist_hash_tables
  )
  return ijkl_list, subconst_cache, constellations, preset_queens

use_visited_prune:bool=False
def set_pre_queens(N:int,ijkl_list:Set[int],subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:list,constellations:List[Dict[str,int]],preset_queens:int,visited:Set[int],constellation_signatures:Set[Tuple[int,int,int,int,int,int]],zobrist_hash_tables: Dict[int, Dict[str, List[u64]]])->Tuple[Set[int], Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]], List[Dict[str,int]], int]:
  board_mask= (1<<N)-1
  if use_visited_prune:
    h: int = int(zobrist_hash(N,ld & board_mask, rd & board_mask, col & board_mask, row, queens, k, l, LD & board_mask, RD & board_mask,zobrist_hash_tables))
    if h in visited:
      return ijkl_list, subconst_cache, constellations, preset_queens
    visited.add(h)

  if row==k or row==l:
    ijkl_list, subconst_cache, constellations, preset_queens = set_pre_queens_cached(N,ijkl_list,subconst_cache,ld<<1,rd>>1,col,k,l,row+1,queens,LD,RD,counter,constellations,preset_queens,visited,constellation_signatures,zobrist_hash_tables)
    return ijkl_list, subconst_cache, constellations, preset_queens
  if queens == preset_queens:
    if (not DISABLE_CONSTELLATION_SIGNATURE_PRUNE) and preset_queens <= 5:
      sig = (ld, rd, col, k, l, row)    # これが signature (tuple)
      if sig in constellation_signatures:
        return ijkl_list, subconst_cache, constellations, preset_queens
      constellation_signatures.add(sig)
    constellation={"ld":ld,"rd":rd,"col":col,"startijkl":row<<20,"solutions":0}
    constellations.append(constellation) #星座データ追加
    counter[0]+=1
    return ijkl_list, subconst_cache, constellations, preset_queens
  free=~(ld|rd|col|(LD>>(N-1-row))|(RD<<(N-1-row)))&board_mask
  while free:
    bit:int=free&-free
    free&=free-1
    ijkl_list, subconst_cache, constellations, preset_queens = set_pre_queens_cached(N,ijkl_list,subconst_cache,(ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,counter,constellations,preset_queens,visited,constellation_signatures,zobrist_hash_tables)

  return ijkl_list, subconst_cache, constellations, preset_queens

def gen_constellations(
  N:int,
  ijkl_list:Set[int],
  subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],
  constellations:List[Dict[str,int]],
  preset_queens:int
)->Tuple[
  Set[int],
  Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],
  List[Dict[str,int]],
  int
]:
  halfN = (N + 1) // 2        # N の半分（切り上げ）。開始星座生成の範囲を絞るために使う
  N1:int = N - 1              # 最終列 index
  N2:int = N - 2

  subconst_cache.clear()

  constellation_signatures: Set[Tuple[int,int,int,int,int,int]] = set()

  if N % 2 == 1:
    center = N // 2
    ijkl_list.update(
      to_ijkl(i, j, center, l)
      for l in range(center + 1, N1)
      for i in range(center + 1, N1)
      if i != (N1) - l
      for j in range(N - center - 2, 0, -1)
      if j != i and j != l
      if not check_rotations(ijkl_list, i, j, center, l, N)
    )

  ijkl_list.update(
    to_ijkl(i, j, k, l)
    for k in range(1, halfN)
    for l in range(k + 1, N1)
    for i in range(k + 1, N1)
    if i != (N1) - l
    for j in range(N - k - 2, 0, -1)
    if j != i and j != l
    if not check_rotations(ijkl_list, i, j, k, l, N)
  )

  ijkl_list.update({to_ijkl(0, j, 0, l) for j in range(1, N2) for l in range(j + 1, N1)})

  ijkl_list = {get_jasmin(N, c) for c in ijkl_list}

  L = 1 << (N1)

  for sc in ijkl_list:
    subconst_cache.clear()

    constellation_signatures = set()

    i, j, k, l = geti(sc), getj(sc), getk(sc), getl(sc)

    Lj = L >> j
    Li = L >> i
    Ll = L >> l

    ld = (((L >> (i - 1)) if i > 0 else 0) | (1 << (N - k)))
    rd = ((L >> (i + 1)) | (1 << (l - 1)))
    col = (1 | L | Li | Lj)

    LD = (Lj | Ll)
    RD = (Lj | (1 << k))

    counter: List[int] = [0]     # set_pre_queens 側が増やす
    visited: Set[int] = set()    # 枝刈り用 visited（hash を入れる設計）

    zobrist_hash_tables: Dict[int, Dict[str, List[u64]]] = {}

    ijkl_list, subconst_cache, constellations, preset_queens = set_pre_queens_cached(
      N, ijkl_list, subconst_cache,
      ld, rd, col,
      k, l,
      1,
      3 if j == N1 else 4,
      LD, RD,
      counter, constellations, preset_queens,
      visited, constellation_signatures,
      zobrist_hash_tables
    )

    base = to_ijkl(i, j, k, l)

    for a in range(counter[0]):
      constellations[-1 - a]["startijkl"] |= base

  return ijkl_list, subconst_cache, constellations, preset_queens

def validate_constellation_list(constellations:List[Dict[str,int]])->bool:
  return all(all(k in c for k in ("ld","rd","col","startijkl")) for c in constellations)

def read_uint32_le(b:str)->int:
  return (ord(b[0])&0xFF)|((ord(b[1])&0xFF)<<8)|((ord(b[2])&0xFF)<<16)|((ord(b[3])&0xFF)<<24)

def int_to_le_bytes(x:int)->List[int]:
  return [(x>>(8*i))&0xFF for i in range(4)]

def file_exists(fname:str)->bool:
  try:
    with open(fname,"rb"):
      return True
  except:
    return False

def validate_bin_file(fname:str)->bool:
  try:
    with open(fname,"rb") as f:
      f.seek(0,2)  # ファイル末尾に移動
      size=f.tell()
    return size%16==0
  except:
    return False

def u64_to_le_bytes(x: u64) -> List[int]:
  v:int = int(x)
  return [(v >> (8*i)) & 0xFF for i in range(8)]

def read_uint64_le( raw: str) -> u64:
  v:int = 0
  for i in range(8):
    v |= (ord(raw[i]) & 0xFF) << (8*i)
  return u64(v)

def save_solutions_txt(fname:str,constellations:List[Dict[str,int]]) -> None:
  f = open(fname, "w")
  f.write("startijkl,solutions\n")
  for d in constellations:
    f.write(str(d["startijkl"]))
    f.write(",")
    f.write(str(int(d["solutions"])))
    f.write("\n")
  f.close()

def save_solutions_bin_v2(fname:str,constellations:List[Dict[str,int]]) -> None:
  b8 = u64_to_le_bytes
  f = open(fname, "wb")
  for d in constellations:
    for x in (
      u64(d["startijkl"]),
      u64(d["ld"]),
      u64(d["rd"]),
      u64(d["col"]),
      u64(d["solutions"]),
    ):
      bb = b8(x)
      f.write("".join(chr(c) for c in bb))
  f.close()

def load_solutions_txt_into(fname:str,constellations:List[Dict[str,int]]) -> bool:
  try:
    f = open(fname, "r")
  except:
    return False
  text = f.read()
  f.close()
  if text is None:
    return False
  lines = text.split("\n")
  if len(lines) < 2:
    return False
  if lines[0].strip() != "startijkl,solutions":
    return False

  mp: Dict[int, int] = {}
  for idx in range(1, len(lines)):
    line = lines[idx].strip()
    if line == "":
      continue
    parts = line.split(",")
    if len(parts) != 2:
      return False
    k = int(parts[0])
    v = int(parts[1])
    mp[k] = v
  for d in constellations:
    s = d["startijkl"]
    if s not in mp:
      return False
    d["solutions"] = mp[s]

  return True

def load_solutions_bin_into_v2(fname:str,constellations:List[Dict[str,int]])->bool:
  try:
    f = open(fname, "rb")
  except:
    return False
  data = f.read()
  f.close()
  if data is None:
    return False
  rec:int = 40
  n:int = len(data)
  if n == 0 or (n % rec) != 0:
    return False
  nrec:int = n // rec
  r8 = read_uint64_le
  mp: Dict[Tuple[u64,u64,u64,u64], u64] = {}
  p:int = 0
  for _ in range(nrec):
    s  = r8(data[p:p+8]);   p += 8
    ld = r8(data[p:p+8]);   p += 8
    rd = r8(data[p:p+8]);   p += 8
    col= r8(data[p:p+8]);   p += 8
    sol= r8(data[p:p+8]);   p += 8
    mp[(s, ld, rd, col)] = sol
  for d in constellations:
    key = (u64(d["startijkl"]), u64(d["ld"]), u64(d["rd"]), u64(d["col"]))
    if key not in mp:
      print("[cache miss] startijkl=", int(key[0])," ld=", int(key[1]), " rd=", int(key[2]), " col=", int(key[3]))
      return False
    d["solutions"] = int(mp[key])

  return True

def load_or_build_solutions_txt(N:int,constellations:List[Dict[str,int]],preset_queens:int,use_gpu:bool,cache_tag:str = "") -> None:

  tag = "_" + cache_tag if cache_tag != "" else ""
  fname = "solutions_N" + str(N) + "_" + str(preset_queens) + tag + ".txt"

  if file_exists(fname):
    if load_solutions_txt_into(fname, constellations):
      return
    else:
      print("[警告] solutions txt キャッシュ不一致: " + fname + " を再生成します")

  exec_solutions(N,constellations,use_gpu)
  save_solutions_txt(fname, constellations)

def load_or_build_solutions_bin(N:int,constellations:List[Dict[str,int]],preset_queens:int,use_gpu:bool,cache_tag:str = "") -> None:

  tag = f"_{cache_tag}" if cache_tag != "" else ""
  fname = f"solutions_N{N}_{preset_queens}{tag}.bin"

  if file_exists(fname):
    if load_solutions_bin_into_v2(fname, constellations):
      return
    else:
      print(f"[警告] solutions キャッシュ不一致/破損: {fname} を再生成します")

  exec_solutions(N,constellations, use_gpu)
  save_solutions_bin_v2(fname, constellations)

def save_constellations_txt(path:str,constellations:List[Dict[str,int]])->None:
  with open(path,"w") as f:
    for c in constellations:
      ld=c["ld"]
      rd=c["rd"]
      col=c["col"]
      startijkl=c["startijkl"]
      solutions=c.get("solutions",0)
      f.write(f"{ld} {rd} {col} {startijkl} {solutions}\n")

def load_constellations_txt(path:str,constellations:List[Dict[str,int]])->List[Dict[str,int]]:
  with open(path,"r") as f:
    for line in f:
      parts=line.strip().split()
      if len(parts)!=5:
        continue
      ld=int(parts[0]);rd=int(parts[1]);col=int(parts[2])
      startijkl=int(parts[3]);solutions=int(parts[4])
      constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions": solutions})
  return constellations

def load_or_build_constellations_txt(N:int,ijkl_list:Set[int],subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],constellations:List[Dict[str,int]],preset_queens:int)->Tuple[Set[int],Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],List[Dict[str,int]],int]:

  fname=f"constellations_N{N}_{preset_queens}.txt"
  if file_exists(fname):
    try:
      constellations=load_constellations_txt(fname,constellations)
      if validate_constellation_list(constellations):
        return ijkl_list,subconst_cache,constellations,preset_queens
      else:
        print(f"[警告] 不正なキャッシュ形式: {fname} を再生成します")
    except Exception as e:
      print(f"[警告] キャッシュ読み込み失敗: {fname}, 理由: {e}")
  ijkl_list,subconst_cache,constellations,preset_queens=gen_constellations(N,ijkl_list,subconst_cache,constellations,preset_queens)
  save_constellations_txt(fname,constellations)
  return ijkl_list,subconst_cache,constellations,preset_queens

def save_constellations_bin(N:int,fname:str,constellations:List[Dict[str,int]])->None:
  with open(fname,"wb") as f:
    for d in constellations:
      for key in ["ld","rd","col","startijkl"]:
        b=int_to_le_bytes(d[key])
        f.write("".join(chr(c) for c in b))  # Codonでは str がバイト文字列扱い

def load_constellations_bin(N:int,fname:str,constellations:List[Dict[str,int]],)->List[Dict[str,int]]:
  _read_uint32_le=read_uint32_le
  with open(fname,"rb") as f:
    while True:
      raw:str=f.read(16)
      if len(raw)<16:
        break
      ld=read_uint32_le(raw[0:4])
      rd=read_uint32_le(raw[4:8])
      col=read_uint32_le(raw[8:12])
      startijkl=_read_uint32_le(raw[12:16])
      constellations.append({ "ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0 })
  return constellations

def load_or_build_constellations_bin(N:int,ijkl_list:Set[int],subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],constellations:List[Dict[str,int]],preset_queens:int)->Tuple[Set[int],Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],List[Dict[str,int]],int]:

  fname=f"constellations_N{N}_{preset_queens}.bin"
  if file_exists(fname):
    try:
      constellations=load_constellations_bin(N,fname,constellations)
      if validate_bin_file(fname) and validate_constellation_list(constellations):
        return ijkl_list,subconst_cache,constellations,preset_queens
      else:
        print(f"[警告] 不正なキャッシュ形式: {fname} を再生成します")
    except Exception as e:
      print(f"[警告] キャッシュ読み込み失敗: {fname}, 理由: {e}")
  ijkl_list,subconst_cache,constellations,preset_queens=gen_constellations(N,ijkl_list,subconst_cache,constellations,preset_queens)
  save_constellations_bin(N,fname,constellations)
  return ijkl_list,subconst_cache,constellations,preset_queens

def count_constellations_bin_records(fname:str)->int:
  try:
    with open(fname,"rb") as f:
      f.seek(0,2)
      size:int=f.tell()
    if size%16!=0:
      return 0
    return size//16
  except:
    return 0

def read_stream_done_count(fname:str)->int:
  try:
    with open(fname,"r") as f:
      text:str=f.read().strip()
    if text=="":
      return -1
    return int(text)
  except:
    return -1

def write_stream_done_count(fname:str,count:int)->None:
  with open(fname,"w") as f:
    f.write(str(count))
    f.write("\n")

def truncate_constellations_bin(fname:str)->None:
  with open(fname,"wb") as f:
    pass
  write_stream_done_count(fname+".done",0)

def append_constellations_bin(fname:str,constellations:List[Dict[str,int]])->None:
  with open(fname,"ab") as f:
    for d in constellations:
      for key in ["ld","rd","col","startijkl"]:
        b=int_to_le_bytes(d[key])
        f.write("".join(chr(c) for c in b))

def gen_constellations_stream_to_bin(
  N:int,
  ijkl_list:Set[int],
  subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],
  fname:str,
  preset_queens:int,
  gpu_log_level:int=0
)->Tuple[Set[int],Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],int,int]:

  halfN:int=(N+1)//2
  N1:int=N-1
  N2:int=N-2
  subconst_cache.clear()

  constellation_signatures:Set[Tuple[int,int,int,int,int,int]]=set()

  if N%2==1:
    center:int=N//2
    ijkl_list.update(
      to_ijkl(i,j,center,l)
      for l in range(center+1,N1)
      for i in range(center+1,N1)
      if i!=(N1)-l
      for j in range(N-center-2,0,-1)
      if j!=i and j!=l
      if not check_rotations(ijkl_list,i,j,center,l,N)
    )

  ijkl_list.update(
    to_ijkl(i,j,k,l)
    for k in range(1,halfN)
    for l in range(k+1,N1)
    for i in range(k+1,N1)
    if i!=(N1)-l
    for j in range(N-k-2,0,-1)
    if j!=i and j!=l
    if not check_rotations(ijkl_list,i,j,k,l,N)
  )

  ijkl_list.update({to_ijkl(0,j,0,l) for j in range(1,N2) for l in range(j+1,N1)})
  ijkl_list={get_jasmin(N,c) for c in ijkl_list}

  L:int=1<<(N1)
  total_count:int=0
  sc_index:int=0
  truncate_constellations_bin(fname)

  for sc in ijkl_list:
    subconst_cache.clear()
    constellation_signatures=set()

    i,j,k,l=geti(sc),getj(sc),getk(sc),getl(sc)
    Lj:int=L>>j
    Li:int=L>>i
    Ll:int=L>>l

    ld:int=(((L>>(i-1)) if i>0 else 0)|(1<<(N-k)))
    rd:int=((L>>(i+1))|(1<<(l-1)))
    col:int=(1|L|Li|Lj)
    LD:int=(Lj|Ll)
    RD:int=(Lj|(1<<k))

    counter:List[int]=[0]
    visited:Set[int]=set()
    zobrist_hash_tables:Dict[int,Dict[str,List[u64]]]={}
    sc_constellations:List[Dict[str,int]]=[]

    ijkl_list,subconst_cache,sc_constellations,preset_queens=set_pre_queens_cached(
      N,ijkl_list,subconst_cache,
      ld,rd,col,
      k,l,
      1,
      3 if j==N1 else 4,
      LD,RD,
      counter,sc_constellations,preset_queens,
      visited,constellation_signatures,
      zobrist_hash_tables
    )

    base:int=to_ijkl(i,j,k,l)
    for a in range(counter[0]):
      sc_constellations[-1-a]["startijkl"]|=base

    if counter[0]>0:
      append_constellations_bin(fname,sc_constellations)
      total_count+=counter[0]

    if gpu_log_level>=2:
      print(f"[stream-build-sc] N={N} sc_index={sc_index} added={counter[0]} total={total_count}")

    sc_index+=1

  write_stream_done_count(fname+".done",total_count)
  if gpu_log_level>=1:
    print(f"[stream-build-summary] N={N} preset_queens={preset_queens} sc={sc_index} records={total_count} bin={fname}")

  return ijkl_list,subconst_cache,total_count,preset_queens

def ensure_constellations_bin_stream(N:int,ijkl_list:Set[int],subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],preset_queens:int,gpu_log_level:int=0)->Tuple[Set[int],Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],int,int,str]:
  fname:str=f"constellations_N{N}_{preset_queens}.bin"
  records:int=count_constellations_bin_records(fname)
  done_count:int=read_stream_done_count(fname+".done")
  if records>0 and done_count==records:
    if gpu_log_level>=1:
      print(f"[stream-cache-hit] N={N} preset_queens={preset_queens} records={records} bin={fname}")
    return ijkl_list,subconst_cache,records,preset_queens,fname

  if file_exists(fname):
    print(f"[stream-cache-warning] invalid/incomplete bin cache: {fname}; records={records} done={done_count}; rebuilding")
  else:
    if gpu_log_level>=1:
      print(f"[stream-cache-miss] N={N} preset_queens={preset_queens} bin={fname}; building")

  ijkl_list,subconst_cache,records,preset_queens=gen_constellations_stream_to_bin(N,ijkl_list,subconst_cache,fname,preset_queens,gpu_log_level)
  return ijkl_list,subconst_cache,records,preset_queens,fname

def stream_elapsed_text_to_ms(elapsed_text:str)->int:
  s:str=elapsed_text
  days:int=0
  day_parts=s.split(",")
  if len(day_parts)>1:
    day_tokens=day_parts[0].strip().split()
    if len(day_tokens)>0:
      days=int(day_tokens[0])
    s=day_parts[1].strip()

  hms=s.split(":")
  if len(hms)<3:
    return 0
  hours:int=int(hms[0])
  minutes:int=int(hms[1])
  sec_ms=hms[2].split(".")
  seconds:int=int(sec_ms[0])
  millis:int=0
  if len(sec_ms)>1:
    ms_str:str=sec_ms[1]
    if len(ms_str)==1:
      millis=int(ms_str)*100
    elif len(ms_str)==2:
      millis=int(ms_str)*10
    elif len(ms_str)>=3:
      millis=int(ms_str[0:3])
  return (((days*24+hours)*60+minutes)*60+seconds)*1000+millis

def format_ratio_3(num:int,den:int)->str:
  if den<=0:
    return "0.000"
  scaled:int=(num*1000)//den
  whole:int=scaled//1000
  frac:int=scaled%1000
  frac_s:str=str(frac)
  if frac<10:
    frac_s="00"+frac_s
  elif frac<100:
    frac_s="0"+frac_s
  return str(whole)+"."+frac_s

def stream_measure2_progress_header()->str:
  h:str="N\tpreset\tchunk\toff\tm\tblock\tmax_blocks\tsteps\tsort_mode\telapsed\telapsed_ms\tchunk_total\tgpu_total\tdone_records\ttotal_records\tremaining_records"
  h+="\tfree_popcount_sum\tfree_popcount_avg\tfree_popcount_min\tfree_popcount_max"
  h+="\trow_sum\trow_avg\trow_min\trow_max"
  h+="\tend_sum\tend_avg\tend_min\tend_max"
  h+="\tdepth_sum\tdepth_avg\tdepth_min\tdepth_max"
  h+="\tscore_sum\tscore_avg\tscore_min\tscore_max"
  h+="\tw2_count\tw4_count\tw8_count"
  fid:int=0
  while fid<28:
    h+=f"\tfuncid_{fid}_count"
    fid+=1
  h+="\n"
  return h

def stream_measure2_stats_suffix(stats:List[int],m:int)->str:
  s:str=""
  s+=f"\t{stats[0]}\t{format_ratio_3(stats[0],m)}\t{stats[1]}\t{stats[2]}"
  s+=f"\t{stats[3]}\t{format_ratio_3(stats[3],m)}\t{stats[4]}\t{stats[5]}"
  s+=f"\t{stats[6]}\t{format_ratio_3(stats[6],m)}\t{stats[7]}\t{stats[8]}"
  s+=f"\t{stats[9]}\t{format_ratio_3(stats[9],m)}\t{stats[10]}\t{stats[11]}"
  s+=f"\t{stats[12]}\t{format_ratio_3(stats[12],m)}\t{stats[13]}\t{stats[14]}"
  s+=f"\t{stats[15]}\t{stats[16]}\t{stats[17]}"
  fid:int=0
  while fid<28:
    s+=f"\t{stats[18+fid]}"
    fid+=1
  return s

def analyze_stream_chunk_input_stats(N:int,chunk_constellations:List[Dict[str,int]])->List[int]:
  m:int=len(chunk_constellations)
  stats:List[int]=[0]*46
  if m<=0:
    return stats

  stats[1]=999999999
  stats[4]=999999999
  stats[7]=999999999
  stats[10]=999999999
  stats[13]=999999999

  soa:TaskSoA=TaskSoA(m)
  w_arr:List[u64]=[u64(0)]*m
  build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)

  i:int=0
  while i<m:
    pc:int=popcount_int(int(soa.free_arr[i]))
    rowv:int=soa.row_arr[i]
    endv:int=soa.end_arr[i]
    depth:int=endv-rowv
    if depth<0:
      depth=0
    score:int=pc*depth

    stats[0]+=pc
    if pc<stats[1]:
      stats[1]=pc
    if pc>stats[2]:
      stats[2]=pc

    stats[3]+=rowv
    if rowv<stats[4]:
      stats[4]=rowv
    if rowv>stats[5]:
      stats[5]=rowv

    stats[6]+=endv
    if endv<stats[7]:
      stats[7]=endv
    if endv>stats[8]:
      stats[8]=endv

    stats[9]+=depth
    if depth<stats[10]:
      stats[10]=depth
    if depth>stats[11]:
      stats[11]=depth

    stats[12]+=score
    if score<stats[13]:
      stats[13]=score
    if score>stats[14]:
      stats[14]=score

    w:int=int(w_arr[i])
    if w==2:
      stats[15]+=1
    elif w==4:
      stats[16]+=1
    elif w==8:
      stats[17]+=1

    fid:int=soa.funcid_arr[i]
    if fid>=0 and fid<28:
      stats[18+fid]+=1

    i+=1

  return stats

def append_stream_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,chunk_total:int,gpu_total:int,total_records:int,stats:List[int])->None:
  done_records:int=off+m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{chunk_total}\t{gpu_total}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,m))
    f.write("\n")

def funcid_reorder_bucket(fid:int)->int:
  if fid==26 or fid==27:
    return 0
  if fid==19 or fid==22 or fid==23 or fid==24:
    return 1
  if fid==20 or fid==21:
    return 2
  if fid==0 or fid==4 or fid==5 or fid==12 or fid==16 or fid==17 or fid==18:
    return 3
  return 4

def funcid_reorder_bucket_label(g:int)->str:
  if g==0:
    return "B"
  if g==1:
    return "A"
  if g==2:
    return "C"
  if g==3:
    return "G"
  return "O"

def funcid_reorder_bucket_fname(N:int,preset_queens:int,g:int)->str:
  return f"constellations_N{N}_{preset_queens}_funcid_reorder_v2_{funcid_reorder_bucket_label(g)}.bin"

def funcid_reorder_output_fname(N:int,preset_queens:int)->str:
  if FUNCID_REORDER_V2_SWEEP_TEMP_OUTPUT:
    return f"constellations_N{N}_{preset_queens}_funcid_reorder_v2_sweep_tmp.bin"
  return f"constellations_N{N}_{preset_queens}_funcid_reorder_v2_w{FUNCID_REORDER_V2_WINDOW_MULT}_j{FUNCID_REORDER_V2_PHASE_JUMP}.bin"

def truncate_plain_bin(fname:str)->None:
  with open(fname,"wb") as f:
    pass

def stream_funcid_reorder_progress_header()->str:
  h:str=stream_measure2_progress_header().strip()
  h+="\trisky_a_count\trisky_a_ratio"
  h+="\trisky_b_count\trisky_b_ratio"
  h+="\trisky_c_count\trisky_c_ratio"
  h+="\tgood_count\tgood_ratio"
  h+="\tother_count\tother_ratio"
  h+="\n"
  return h

def stream_funcid_reorder_risk_suffix(stats:List[int],m:int)->str:
  risky_a:int=stats[18+19]+stats[18+22]+stats[18+23]+stats[18+24]
  risky_b:int=stats[18+26]+stats[18+27]
  risky_c:int=stats[18+20]+stats[18+21]
  good:int=stats[18+0]+stats[18+4]+stats[18+5]+stats[18+12]+stats[18+16]+stats[18+17]+stats[18+18]
  other:int=m-risky_a-risky_b-risky_c-good
  if other<0:
    other=0
  s:str=""
  s+=f"\t{risky_a}\t{format_ratio_3(risky_a,m)}"
  s+=f"\t{risky_b}\t{format_ratio_3(risky_b,m)}"
  s+=f"\t{risky_c}\t{format_ratio_3(risky_c,m)}"
  s+=f"\t{good}\t{format_ratio_3(good,m)}"
  s+=f"\t{other}\t{format_ratio_3(other,m)}"
  return s

def profile_elapsed_ms_between(t0:datetime,t1:datetime)->int:
  return stream_elapsed_text_to_ms(str(t1-t0)[:-3])

def analyze_stream_chunk_input_stats_from_soa(soa:TaskSoA,w_arr:List[u64],m:int)->List[int]:
  stats:List[int]=[0]*46
  if m<=0:
    return stats

  stats[1]=999999999
  stats[4]=999999999
  stats[7]=999999999
  stats[10]=999999999
  stats[13]=999999999

  i:int=0
  while i<m:
    pc:int=popcount_int(int(soa.free_arr[i]))
    rowv:int=soa.row_arr[i]
    endv:int=soa.end_arr[i]
    depth:int=endv-rowv
    if depth<0:
      depth=0
    score:int=pc*depth

    stats[0]+=pc
    if pc<stats[1]:
      stats[1]=pc
    if pc>stats[2]:
      stats[2]=pc

    stats[3]+=rowv
    if rowv<stats[4]:
      stats[4]=rowv
    if rowv>stats[5]:
      stats[5]=rowv

    stats[6]+=endv
    if endv<stats[7]:
      stats[7]=endv
    if endv>stats[8]:
      stats[8]=endv

    stats[9]+=depth
    if depth<stats[10]:
      stats[10]=depth
    if depth>stats[11]:
      stats[11]=depth

    stats[12]+=score
    if score<stats[13]:
      stats[13]=score
    if score>stats[14]:
      stats[14]=score

    w:int=int(w_arr[i])
    if w==2:
      stats[15]+=1
    elif w==4:
      stats[16]+=1
    elif w==8:
      stats[17]+=1

    fid:int=soa.funcid_arr[i]
    if fid>=0 and fid<28:
      stats[18+fid]+=1

    i+=1

  return stats

def stream_funcid_reorder_profile_progress_header()->str:
  h:str=stream_funcid_reorder_progress_header().strip()
  h+="\tstage_read_ms"
  h+="\tstage_soa_ms"
  h+="\tstage_stats_ms"
  h+="\tstage_sort_ms"
  h+="\tstage_kernel_ms"
  h+="\tstage_reduce_ms"
  h+="\tstage_compute_ms"
  h+="\tstage_no_read_ms"
  h+="\tstage_total_ms"
  h+="\n"
  return h

def stream_funcid_reorder_profile_stage_suffix(stages:List[int])->str:
  s:str=""
  i:int=0
  while i<9:
    v:int=0
    if i<len(stages):
      v=stages[i]
    s+=f"\t{v}"
    i+=1
  return s

def append_stream_funcid_reorder_profile_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,chunk_total:int,gpu_total:int,total_records:int,stats:List[int],stages:List[int])->None:
  done_records:int=off+m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{chunk_total}\t{gpu_total}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,m))
    f.write(stream_funcid_reorder_risk_suffix(stats,m))
    f.write(stream_funcid_reorder_profile_stage_suffix(stages))
    f.write("\n")

def stream_funcid_reorder_chunksize_progress_header()->str:
  h:str=stream_funcid_reorder_profile_progress_header().strip()
  h+="\tbase_steps"
  h+="\tbase_chunk"
  h+="\tchunk_factor"
  h+="\trange_start_chunk"
  h+="\trange_end_chunk"
  h+="\trange_chunks"
  h+="\n"
  return h

def append_stream_funcid_reorder_chunksize_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,chunk_total:int,gpu_total:int,total_records:int,stats:List[int],stages:List[int],base_steps:int,base_chunk:int,chunk_factor:int,range_start_chunk:int,range_end_chunk:int,range_chunks:int)->None:
  done_records:int=off+m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{chunk_total}\t{gpu_total}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,m))
    f.write(stream_funcid_reorder_risk_suffix(stats,m))
    f.write(stream_funcid_reorder_profile_stage_suffix(stages))
    f.write(f"\t{base_steps}\t{base_chunk}\t{chunk_factor}\t{range_start_chunk}\t{range_end_chunk}\t{range_chunks}")
    f.write("\n")

def stream_funcid_reorder_funcid_target_progress_header()->str:
  h:str=stream_funcid_reorder_profile_progress_header().strip()
  h+="\tbase_steps"
  h+="\tbase_chunk"
  h+="\ttarget_group"
  h+="\tsource_m"
  h+="\ttarget_m"
  h+="\tstage_filter_ms"
  h+="\n"
  return h

def append_stream_funcid_reorder_funcid_target_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,target_m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,chunk_total:int,gpu_total:int,total_records:int,stats:List[int],stages:List[int],base_steps:int,base_chunk:int,target_group:str,source_m:int,stage_filter_ms:int)->None:
  done_records:int=off+source_m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{target_m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{chunk_total}\t{gpu_total}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,target_m))
    f.write(stream_funcid_reorder_risk_suffix(stats,target_m))
    f.write(stream_funcid_reorder_profile_stage_suffix(stages))
    f.write(f"\t{base_steps}\t{base_chunk}\t{target_group}\t{source_m}\t{target_m}\t{stage_filter_ms}")
    f.write("\n")

def stream_funcid_reorder_funcid_single_progress_header()->str:
  h:str=stream_funcid_reorder_profile_progress_header().strip()
  h+="	base_steps"
  h+="	base_chunk"
  h+="	target_funcid"
  h+="	source_m"
  h+="	target_m"
  h+="	stage_classify_ms"
  h+="	stage_filter_ms"
  h+="\n"
  return h

def append_stream_funcid_reorder_funcid_single_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,target_m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,chunk_total:int,gpu_total:int,total_records:int,stats:List[int],stages:List[int],base_steps:int,base_chunk:int,target_funcid:int,source_m:int,stage_classify_ms:int,stage_filter_ms:int)->None:
  done_records:int=off+source_m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}	{preset_queens}	{chunk_index}	{off}	{target_m}	{BLOCK}	{MAX_BLOCKS}	{STEPS}	{gpu_sort_mode}	{elapsed_text}	{elapsed_ms}	{chunk_total}	{gpu_total}	{done_records}	{total_records}	{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,target_m))
    f.write(stream_funcid_reorder_risk_suffix(stats,target_m))
    f.write(stream_funcid_reorder_profile_stage_suffix(stages))
    f.write(f"	{base_steps}	{base_chunk}	{target_funcid}	{source_m}	{target_m}	{stage_classify_ms}	{stage_filter_ms}")
    f.write("\n")

def stream_funcid_reorder_funcid_split_progress_header()->str:
  h:str=stream_funcid_reorder_profile_progress_header().strip()
  h+="	base_steps"
  h+="	base_chunk"
  h+="	split_group"
  h+="	source_m"
  h+="	target_m"
  h+="	stage_classify_ms"
  h+="	stage_filter_ms"
  h+="\n"
  return h

def append_stream_funcid_reorder_funcid_split_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,target_m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,chunk_total:int,gpu_total:int,total_records:int,stats:List[int],stages:List[int],base_steps:int,base_chunk:int,split_group:str,source_m:int,stage_classify_ms:int,stage_filter_ms:int)->None:
  done_records:int=off+source_m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}	{preset_queens}	{chunk_index}	{off}	{target_m}	{BLOCK}	{MAX_BLOCKS}	{STEPS}	{gpu_sort_mode}	{elapsed_text}	{elapsed_ms}	{chunk_total}	{gpu_total}	{done_records}	{total_records}	{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,target_m))
    f.write(stream_funcid_reorder_risk_suffix(stats,target_m))
    f.write(stream_funcid_reorder_profile_stage_suffix(stages))
    f.write(f"	{base_steps}	{base_chunk}	{split_group}	{source_m}	{target_m}	{stage_classify_ms}	{stage_filter_ms}")
    f.write("\n")

def stream_funcid_reorder_funcid_depth_progress_header()->str:
  h:str=stream_funcid_reorder_profile_progress_header().strip()
  h+="\tbase_steps"
  h+="\tbase_chunk"
  h+="\ttarget_funcid"
  h+="\tdepth_bucket"
  h+="\tsource_m"
  h+="\ttarget_m"
  h+="\tdepth_min"
  h+="\tdepth_max"
  h+="\tfree_pc_min"
  h+="\tfree_pc_max"
  h+="\tstage_classify_ms"
  h+="\tstage_filter_ms"
  h+="\n"
  return h

def append_stream_funcid_reorder_funcid_depth_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,target_m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,chunk_total:int,gpu_total:int,total_records:int,stats:List[int],stages:List[int],base_steps:int,base_chunk:int,target_funcid:int,depth_bucket:str,source_m:int,depth_min:int,depth_max:int,free_pc_min:int,free_pc_max:int,stage_classify_ms:int,stage_filter_ms:int)->None:
  done_records:int=off+source_m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{target_m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{chunk_total}\t{gpu_total}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,target_m))
    f.write(stream_funcid_reorder_risk_suffix(stats,target_m))
    f.write(stream_funcid_reorder_profile_stage_suffix(stages))
    f.write(f"\t{base_steps}\t{base_chunk}\t{target_funcid}\t{depth_bucket}\t{source_m}\t{target_m}\t{depth_min}\t{depth_max}\t{free_pc_min}\t{free_pc_max}\t{stage_classify_ms}\t{stage_filter_ms}")
    f.write("\n")

def stream_funcid_reorder_funcid_mark_progress_header()->str:
  h:str=stream_funcid_reorder_profile_progress_header().strip()
  h+="\tbase_steps"
  h+="\tbase_chunk"
  h+="\ttarget_funcid"
  h+="\tmark_bucket"
  h+="\tsource_m"
  h+="\ttarget_m"
  h+="\tmark1_min"
  h+="\tmark1_max"
  h+="\tmark2_min"
  h+="\tmark2_max"
  h+="\tmark_gap_min"
  h+="\tmark_gap_max"
  h+="\trow_to_mark1_min"
  h+="\trow_to_mark1_max"
  h+="\trow_to_mark2_min"
  h+="\trow_to_mark2_max"
  h+="\tjmark_min"
  h+="\tjmark_max"
  h+="\tendmark_min"
  h+="\tendmark_max"
  h+="\tstage_classify_ms"
  h+="\tstage_filter_ms"
  h+="\n"
  return h

def append_stream_funcid_reorder_funcid_mark_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,target_m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,chunk_total:int,gpu_total:int,total_records:int,stats:List[int],stages:List[int],base_steps:int,base_chunk:int,target_funcid:int,mark_bucket:str,source_m:int,mark1_min:int,mark1_max:int,mark2_min:int,mark2_max:int,mark_gap_min:int,mark_gap_max:int,row_to_mark1_min:int,row_to_mark1_max:int,row_to_mark2_min:int,row_to_mark2_max:int,jmark_min:int,jmark_max:int,endmark_min:int,endmark_max:int,stage_classify_ms:int,stage_filter_ms:int)->None:
  done_records:int=off+source_m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{target_m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{chunk_total}\t{gpu_total}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,target_m))
    f.write(stream_funcid_reorder_risk_suffix(stats,target_m))
    f.write(stream_funcid_reorder_profile_stage_suffix(stages))
    f.write(f"\t{base_steps}\t{base_chunk}\t{target_funcid}\t{mark_bucket}\t{source_m}\t{target_m}\t{mark1_min}\t{mark1_max}\t{mark2_min}\t{mark2_max}\t{mark_gap_min}\t{mark_gap_max}\t{row_to_mark1_min}\t{row_to_mark1_max}\t{row_to_mark2_min}\t{row_to_mark2_max}\t{jmark_min}\t{jmark_max}\t{endmark_min}\t{endmark_max}\t{stage_classify_ms}\t{stage_filter_ms}")
    f.write("\n")

def stream_funcid_reorder_funcid_markdist_progress_header()->str:
  h:str=stream_funcid_reorder_profile_progress_header().strip()
  h+="\tbase_steps"
  h+="\tbase_chunk"
  h+="\ttarget_funcid"
  h+="\tmarkdist_axis"
  h+="\tmarkdist_key"
  h+="\tsource_m"
  h+="\ttarget_m"
  h+="\tmark1_min"
  h+="\tmark1_max"
  h+="\tmark2_min"
  h+="\tmark2_max"
  h+="\tmark_gap_min"
  h+="\tmark_gap_max"
  h+="\td1_min"
  h+="\td1_max"
  h+="\td2_min"
  h+="\td2_max"
  h+="\tjmark_min"
  h+="\tjmark_max"
  h+="\tendmark_min"
  h+="\tendmark_max"
  h+="\tstage_classify_ms"
  h+="\tstage_filter_ms"
  h+="\n"
  return h

def append_stream_funcid_reorder_funcid_markdist_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,target_m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,chunk_total:int,gpu_total:int,total_records:int,stats:List[int],stages:List[int],base_steps:int,base_chunk:int,target_funcid:int,markdist_axis:str,markdist_key:str,source_m:int,mark1_min:int,mark1_max:int,mark2_min:int,mark2_max:int,mark_gap_min:int,mark_gap_max:int,d1_min:int,d1_max:int,d2_min:int,d2_max:int,jmark_min:int,jmark_max:int,endmark_min:int,endmark_max:int,stage_classify_ms:int,stage_filter_ms:int)->None:
  done_records:int=off+source_m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{target_m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{chunk_total}\t{gpu_total}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,target_m))
    f.write(stream_funcid_reorder_risk_suffix(stats,target_m))
    f.write(stream_funcid_reorder_profile_stage_suffix(stages))
    f.write(f"\t{base_steps}\t{base_chunk}\t{target_funcid}\t{markdist_axis}\t{markdist_key}\t{source_m}\t{target_m}\t{mark1_min}\t{mark1_max}\t{mark2_min}\t{mark2_max}\t{mark_gap_min}\t{mark_gap_max}\t{d1_min}\t{d1_max}\t{d2_min}\t{d2_max}\t{jmark_min}\t{jmark_max}\t{endmark_min}\t{endmark_max}\t{stage_classify_ms}\t{stage_filter_ms}")
    f.write("\n")

def append_stream_funcid_reorder_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,chunk_total:int,gpu_total:int,total_records:int,stats:List[int])->None:
  done_records:int=off+m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{chunk_total}\t{gpu_total}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,m))
    f.write(stream_funcid_reorder_risk_suffix(stats,m))
    f.write("\n")

def append_one_constellation_from_file(f:T,out:List[Dict[str,int]],T:type)->bool:
  raw:str=f.read(16)
  if len(raw)<16:
    return False
  ld:int=read_uint32_le(raw[0:4])
  rd:int=read_uint32_le(raw[4:8])
  col:int=read_uint32_le(raw[8:12])
  startijkl:int=read_uint32_le(raw[12:16])
  out.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
  return True

def build_funcid_reorder_bucket_bins(N:int,fname:str,preset_queens:int,BLOCK:int,MAX_BLOCKS:int,gpu_log_level:int=0)->List[int]:
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  g:int=0
  while g<5:
    truncate_plain_bin(funcid_reorder_bucket_fname(N,preset_queens,g))
    g+=1

  counts:List[int]=[0]*5
  chunk_index:int=0
  _read_uint32_le=read_uint32_le

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-bucket-config] N={N} bin={fname} steps={STEPS}")

  with open(fname,"rb") as f:
    while True:
      chunk_constellations:List[Dict[str,int]]=[]
      i:int=0
      while i<STEPS:
        raw:str=f.read(16)
        if len(raw)<16:
          break
        ld:int=_read_uint32_le(raw[0:4])
        rd:int=_read_uint32_le(raw[4:8])
        col:int=_read_uint32_le(raw[8:12])
        startijkl:int=_read_uint32_le(raw[12:16])
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1

      m:int=len(chunk_constellations)
      if m==0:
        break

      soa:TaskSoA=TaskSoA(m)
      w_arr:List[u64]=[u64(0)]*m
      build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)

      bucket_b:List[Dict[str,int]]=[]
      bucket_a:List[Dict[str,int]]=[]
      bucket_c:List[Dict[str,int]]=[]
      bucket_g:List[Dict[str,int]]=[]
      bucket_o:List[Dict[str,int]]=[]

      i=0
      while i<m:
        fid:int=soa.funcid_arr[i]
        bg:int=funcid_reorder_bucket(fid)
        if bg==0:
          bucket_b.append(chunk_constellations[i])
          counts[0]+=1
        elif bg==1:
          bucket_a.append(chunk_constellations[i])
          counts[1]+=1
        elif bg==2:
          bucket_c.append(chunk_constellations[i])
          counts[2]+=1
        elif bg==3:
          bucket_g.append(chunk_constellations[i])
          counts[3]+=1
        else:
          bucket_o.append(chunk_constellations[i])
          counts[4]+=1
        i+=1

      if len(bucket_b)>0:
        append_constellations_bin(funcid_reorder_bucket_fname(N,preset_queens,0),bucket_b)
      if len(bucket_a)>0:
        append_constellations_bin(funcid_reorder_bucket_fname(N,preset_queens,1),bucket_a)
      if len(bucket_c)>0:
        append_constellations_bin(funcid_reorder_bucket_fname(N,preset_queens,2),bucket_c)
      if len(bucket_g)>0:
        append_constellations_bin(funcid_reorder_bucket_fname(N,preset_queens,3),bucket_g)
      if len(bucket_o)>0:
        append_constellations_bin(funcid_reorder_bucket_fname(N,preset_queens,4),bucket_o)

      if gpu_log_level>=2:
        print(f"[funcid-reorder-v2-bucket-chunk] chunk={chunk_index} m={m} B={len(bucket_b)} A={len(bucket_a)} C={len(bucket_c)} G={len(bucket_g)} O={len(bucket_o)}")
      chunk_index+=1

  if gpu_log_level>=1:
    total:int=counts[0]+counts[1]+counts[2]+counts[3]+counts[4]
    print(f"[funcid-reorder-v2-bucket-summary] N={N} records={total} B={counts[0]} A={counts[1]} C={counts[2]} G={counts[3]} O={counts[4]}")

  return counts

def funcid_reorder_make_quotas(rem_counts:List[int],total_remaining:int,m_target:int)->List[int]:
  quotas:List[int]=[0]*5
  if total_remaining<=0 or m_target<=0:
    return quotas

  g:int=0
  quota_sum:int=0
  while g<5:
    q:int=(rem_counts[g]*m_target)//total_remaining
    if q>rem_counts[g]:
      q=rem_counts[g]
    quotas[g]=q
    quota_sum+=q
    g+=1

  while quota_sum<m_target:
    best:int=-1
    best_room:int=-1
    g=0
    while g<5:
      room:int=rem_counts[g]-quotas[g]
      if room>best_room:
        best_room=room
        best=g
      g+=1
    if best<0 or best_room<=0:
      break
    quotas[best]+=1
    quota_sum+=1

  return quotas

FUNCID_REORDER_V2_WINDOW_MULT:int=8
FUNCID_REORDER_V2_PHASE_JUMP:int=7
FUNCID_REORDER_V2_DEFAULT_REASON:str="N22 measured best baseline w8_j7"

MICROBENCH_DEFAULT_CHUNK_LIST:str="0,8,94,150,339,557,710,976,1073,1222,1225,1370,1469,1471,1587,1625,1644,1706,1772,1854"

CHUNKSIZE_DEFAULT_FACTOR_LIST:str="1,2,4"

FUNCID_TARGET_DEFAULT_CHUNK_LIST:str="94,1587,1625,1644,1772,1854"
FUNCID_TARGET_DEFAULT_GROUP_LIST:str="all;5;7;23;5+23;7+23;other"

FUNCID_SINGLE_DEFAULT_CHUNK_LIST:str="94,1587,1625,1644,1772,1854"
FUNCID_SINGLE_DEFAULT_FUNCID_LIST:str="0-27"

FUNCID_SPLIT_DEFAULT_CHUNK_LIST:str="94,1587,1625,1644,1772,1854"
FUNCID_SPLIT_DEFAULT_GROUP_LIST:str="all;heavy_tail;bulk_heavy;rest"

FUNCID_DEPTH_DEFAULT_CHUNK_LIST:str="94,1587,1625,1644,1772,1854"
FUNCID_DEPTH_DEFAULT_FUNCID_LIST:str="0,4,5,7"
FUNCID_DEPTH_DEFAULT_BUCKET_LIST:str="all;d0-8;d9-10;d11-12;d13-14;d15p;pc0-1;pc2;pc3;pc4p;d13p_pc3p"

FUNCID_MARK_DEFAULT_CHUNK_LIST:str="94,1587,1625,1644,1772,1854"
FUNCID_MARK_DEFAULT_FUNCID_LIST:str="0,1,4,5,7,15,19,22,24"
FUNCID_MARK_DEFAULT_BUCKET_LIST:str="all;gap0;gap1;gap2;gap3p;near_m1_0_2;near_m1_3p;near_m2_0_2;near_m2_3p"

FUNCID_MARKDIST_DEFAULT_CHUNK_LIST:str="94,1587,1625,1644,1772,1854"
FUNCID_MARKDIST_DEFAULT_FUNCID_LIST:str="0,1,4,5,7,15,19,22,24"
FUNCID_MARKDIST_DEFAULT_AXIS_LIST:str="all;gap;d1;d2;gap_d1;gap_d2;d1_d2;gap_d1_d2"

FUNCID_MARKDIST_RISK_REORDER_VERSION:str="v1"
FUNCID_MARKDIST_RISK_REORDER_DEFAULT_REASON:str="105Py exact mark-distance tail table fid_gap_d1"

BROAD_MARKDIST_REORDER_VERSION:str="v1"
BROAD_MARKDIST_REORDER_DEFAULT_REASON:str="107 broad funcid primary + exact mark-distance secondary 5x5 quota"

BROAD_MARKDIST_TAIL_REORDER_VERSION:str="v4"
BROAD_MARKDIST_TAIL_REORDER_DEFAULT_REASON:str="115 final default: 114 weekend ablation selected rotate_only for A10G single-GPU throughput"
BROAD_MARKDIST_TAIL_VARIANT:int=2
BROAD_MARKDIST_TAIL_PHASE_SALT:int=53
BROAD_MARKDIST_TAIL_CELL_SALT:int=17
BROAD_MARKDIST_TAIL_RISK_SALT:int=11

def broad_markdist_tail_variant_tag()->str:
  v:int=BROAD_MARKDIST_TAIL_VARIANT
  if v==0:
    return "v2base"
  if v==1:
    return "phase_only"
  if v==2:
    return "rotate_only"
  if v==3:
    return "wide_only"
  if v==4:
    return "phase_rotate"
  if v==5:
    return "wide_phase_rotate"
  return "unknown"

def broad_markdist_tail_variant_desc()->str:
  v:int=BROAD_MARKDIST_TAIL_VARIANT
  if v==0:
    return "112/111-like v2 baseline: boost=1, simple tail phase, fixed interleave"
  if v==1:
    return "phase only: boost=1, cell/risk-aware tail phase, fixed interleave"
  if v==2:
    return "rotate only: boost=1, simple tail phase, rotating F17/GH/R interleave"
  if v==3:
    return "wide only: boost=2, simple tail phase, fixed interleave"
  if v==4:
    return "phase+rotate: boost=1, cell/risk-aware tail phase, rotating interleave"
  if v==5:
    return "113-like full: boost=2, cell/risk-aware tail phase, rotating interleave"
  return "unknown broadmarktail variant"

def broad_markdist_tail_window_boost_value()->int:
  v:int=BROAD_MARKDIST_TAIL_VARIANT
  if v==3 or v==5:
    return 2
  return 1

def broad_markdist_tail_phase_salt_value()->int:
  v:int=BROAD_MARKDIST_TAIL_VARIANT
  if v==1 or v==4 or v==5:
    return 53
  return 31

def broad_markdist_tail_use_phase_mix()->bool:
  v:int=BROAD_MARKDIST_TAIL_VARIANT
  return v==1 or v==4 or v==5

def broad_markdist_tail_use_rotating_interleave()->bool:
  v:int=BROAD_MARKDIST_TAIL_VARIANT
  return v==2 or v==4 or v==5

def parse_chunk_list_spec(spec:str)->List[int]:
  out:List[int]=[]
  s:str=spec.strip()
  if s=="" or s=="-":
    return out
  parts:List[str]=s.split(",")
  for p in parts:
    t:str=p.strip()
    if t=="":
      continue
    v:int=int(t)
    if v>=0:
      out.append(v)
  return out

def parse_positive_int_list_spec(spec:str)->List[int]:
  out:List[int]=[]
  s:str=spec.strip()
  if s=="" or s=="-":
    return out
  parts:List[str]=s.split(",")
  for p in parts:
    t:str=p.strip()
    if t=="":
      continue
    v:int=int(t)
    if v>0:
      out.append(v)
  return out

def parse_funcid_target_group_list_spec(spec:str)->List[str]:
  out:List[str]=[]
  s:str=spec.strip()
  if s=="" or s=="-":
    return out
  parts:List[str]=s.split(";")
  for p in parts:
    t:str=p.strip()
    if t!="":
      out.append(t)
  return out

def parse_funcid_single_list_spec(spec:str)->List[int]:
  out:List[int]=[]
  s:str=spec.strip()
  if s=="" or s=="-" or s=="all" or s=="ALL" or s=="*":
    return out
  parts:List[str]=s.split(",")
  for p in parts:
    t:str=p.strip()
    if t=="":
      continue
    if len(t)>=2 and (t[0:1]=="f" or t[0:1]=="F"):
      t=t[1:]
    rparts:List[str]=t.split("-")
    if len(rparts)==2:
      a:int=int(rparts[0].strip())
      b:int=int(rparts[1].strip())
      if a>b:
        tmp:int=a
        a=b
        b=tmp
      v:int=a
      while v<=b:
        if v>=0 and v<28:
          out.append(v)
        v+=1
    else:
      v:int=int(t)
      if v>=0 and v<28:
        out.append(v)
  return out

def default_funcid_single_list()->List[int]:
  out:List[int]=[]
  i:int=0
  while i<28:
    out.append(i)
    i+=1
  return out

def funcid_list_to_string(fids:List[int])->str:
  s:str=""
  first:bool=True
  for v in fids:
    if first:
      s=str(v)
      first=False
    else:
      s+=","+str(v)
  return s

def funcid_split_group_match(group:str,fid:int)->bool:
  g:str=group.strip()
  if g=="all":
    return True
  if g=="heavy_tail" or g=="tail" or g=="heavy":
    return fid==0 or fid==4 or fid==7
  if g=="bulk_heavy" or g=="bulk" or g=="f5bulk":
    return fid==5
  if g=="rest" or g=="other":
    return not (fid==0 or fid==4 or fid==7 or fid==5)
  if g=="medium":
    return fid==1 or fid==12 or fid==15 or fid==18 or fid==19 or fid==20 or fid==22 or fid==24
  if g=="light":
    return fid==16 or fid==17 or fid==21 or fid==23 or fid==25 or fid==26 or fid==27
  return funcid_target_group_match(g,fid)

def filter_constellations_by_funcid_split_group_from_soa(source:List[Dict[str,int]],soa:TaskSoA,source_m:int,group:str)->List[Dict[str,int]]:
  out:List[Dict[str,int]]=[]
  if source_m<=0:
    return out
  g:str=group.strip()
  if g=="all":
    return source
  i:int=0
  while i<source_m:
    fid:int=soa.funcid_arr[i]
    if funcid_split_group_match(g,fid):
      out.append(source[i])
    i+=1
  return out

def parse_funcid_depth_bucket_list_spec(spec:str)->List[str]:
  out:List[str]=[]
  s:str=spec.strip()
  if s=="" or s=="-":
    return out
  parts:List[str]=s.split(";")
  for p in parts:
    t:str=p.strip()
    if t!="":
      out.append(t)
  return out

def funcid_depth_bucket_match(bucket:str,depth:int,free_pc:int)->bool:
  g:str=bucket.strip()
  if g=="" or g=="all" or g=="ALL" or g=="*":
    return True

  if g=="d0-8" or g=="depth0-8":
    return depth>=0 and depth<=8
  if g=="d9-10" or g=="depth9-10":
    return depth>=9 and depth<=10
  if g=="d11-12" or g=="depth11-12":
    return depth>=11 and depth<=12
  if g=="d13-14" or g=="depth13-14":
    return depth>=13 and depth<=14
  if g=="d15p" or g=="d15+" or g=="depth15p" or g=="depth15+":
    return depth>=15

  if g=="pc0-1" or g=="free0-1":
    return free_pc>=0 and free_pc<=1
  if g=="pc2" or g=="free2":
    return free_pc==2
  if g=="pc3" or g=="free3":
    return free_pc==3
  if g=="pc4p" or g=="pc4+" or g=="free4p" or g=="free4+":
    return free_pc>=4

  if g=="d13p_pc3p" or g=="d13+pc3+" or g=="d13p+pc3p":
    return depth>=13 and free_pc>=3
  if g=="d13p_pc4p" or g=="d13+pc4+" or g=="d13p+pc4p":
    return depth>=13 and free_pc>=4
  if g=="d15p_pc3p" or g=="d15+pc3+" or g=="d15p+pc3p":
    return depth>=15 and free_pc>=3
  if g=="d15p_pc4p" or g=="d15+pc4+" or g=="d15p+pc4p":
    return depth>=15 and free_pc>=4

  return False

def filter_constellations_by_funcid_depth_bucket_from_soa(source:List[Dict[str,int]],soa:TaskSoA,source_m:int,target_fid:int,bucket:str)->List[Dict[str,int]]:
  out:List[Dict[str,int]]=[]
  if source_m<=0:
    return out
  i:int=0
  while i<source_m:
    fid:int=soa.funcid_arr[i]
    if fid==target_fid:
      depth:int=soa.end_arr[i]-soa.row_arr[i]
      if depth<0:
        depth=0
      free_pc:int=popcount_int(int(soa.free_arr[i]))
      if funcid_depth_bucket_match(bucket,depth,free_pc):
        out.append(source[i])
    i+=1
  return out

def summarize_funcid_depth_bucket_from_soa(soa:TaskSoA,source_m:int,target_fid:int,bucket:str)->List[int]:
  count:int=0
  dmin:int=999999
  dmax:int=-1
  pcmin:int=999999
  pcmax:int=-1
  i:int=0
  while i<source_m:
    fid:int=soa.funcid_arr[i]
    if fid==target_fid:
      depth:int=soa.end_arr[i]-soa.row_arr[i]
      if depth<0:
        depth=0
      free_pc:int=popcount_int(int(soa.free_arr[i]))
      if funcid_depth_bucket_match(bucket,depth,free_pc):
        count+=1
        if depth<dmin:
          dmin=depth
        if depth>dmax:
          dmax=depth
        if free_pc<pcmin:
          pcmin=free_pc
        if free_pc>pcmax:
          pcmax=free_pc
    i+=1
  if count==0:
    dmin=0
    dmax=0
    pcmin=0
    pcmax=0
  return [count,dmin,dmax,pcmin,pcmax]

def parse_funcid_mark_bucket_list_spec(spec:str)->List[str]:
  out:List[str]=[]
  s:str=spec.strip()
  if s=="" or s=="-":
    return out
  parts:List[str]=s.split(";")
  for p in parts:
    t:str=p.strip()
    if t!="":
      out.append(t)
  return out

def funcid_mark_effective_values_from_soa(soa:TaskSoA,idx:int)->List[int]:
  ijkl:int=soa.ijkl_arr[idx]
  k:int=getk(ijkl)
  l:int=getl(ijkl)
  lo:int=k
  hi:int=l
  if lo>hi:
    tmp:int=lo
    lo=hi
    hi=tmp
  mark1:int=lo-1
  mark2:int=hi-1
  if mark1<0:
    mark1=0
  if mark2<0:
    mark2=0
  rowv:int=soa.row_arr[idx]
  mark_gap:int=mark2-mark1
  row_to_mark1:int=mark1-rowv
  row_to_mark2:int=mark2-rowv
  return [mark1,mark2,mark_gap,row_to_mark1,row_to_mark2,soa.jmark_arr[idx],soa.end_arr[idx]]

def is_nonnegative_decimal_string(s:str)->bool:
  if s=="":
    return False
  i:int=0
  while i<len(s):
    ch:int=ord(s[i:i+1])
    if ch<48 or ch>57:
      return False
    i+=1
  return True

def funcid_mark_bucket_match(bucket:str,mark1:int,mark2:int,mark_gap:int,row_to_mark1:int,row_to_mark2:int,jmark:int,endmark:int)->bool:
  g:str=bucket.strip()
  if g=="" or g=="all" or g=="ALL" or g=="*":
    return True

  if g=="gap0" or g=="mark_gap0":
    return mark_gap==0
  if g=="gap1" or g=="mark_gap1":
    return mark_gap==1
  if g=="gap2" or g=="mark_gap2":
    return mark_gap==2
  if g=="gap3p" or g=="gap3+" or g=="mark_gap3p" or g=="mark_gap3+":
    return mark_gap>=3

  if g=="near_m1_0_2" or g=="m1_0_2" or g=="row_to_m1_0_2" or g=="rtm1_0_2":
    return row_to_mark1>=0 and row_to_mark1<=2
  if g=="near_m1_3p" or g=="near_m1_3+" or g=="m1_3p" or g=="row_to_m1_3p" or g=="rtm1_3p":
    return row_to_mark1>=3

  if g=="near_m2_0_2" or g=="m2_0_2" or g=="row_to_m2_0_2" or g=="rtm2_0_2":
    return row_to_mark2>=0 and row_to_mark2<=2
  if g=="near_m2_3p" or g=="near_m2_3+" or g=="m2_3p" or g=="row_to_m2_3p" or g=="rtm2_3p":
    return row_to_mark2>=3

  if len(g)>3 and g[0:3]=="m1_":
    tail:str=g[3:]
    if is_nonnegative_decimal_string(tail):
      return mark1==int(tail)
  if len(g)>3 and g[0:3]=="m2_":
    tail=g[3:]
    if is_nonnegative_decimal_string(tail):
      return mark2==int(tail)
  if len(g)>5 and g[0:5]=="jmark":
    tail=g[5:]
    if is_nonnegative_decimal_string(tail):
      return jmark==int(tail)
  if len(g)>7 and g[0:7]=="endmark":
    tail=g[7:]
    if is_nonnegative_decimal_string(tail):
      return endmark==int(tail)
  if len(g)>3 and g[0:3]=="end":
    tail=g[3:]
    if is_nonnegative_decimal_string(tail):
      return endmark==int(tail)

  return False

def filter_constellations_by_funcid_mark_bucket_from_soa(source:List[Dict[str,int]],soa:TaskSoA,source_m:int,target_fid:int,bucket:str)->List[Dict[str,int]]:
  out:List[Dict[str,int]]=[]
  if source_m<=0:
    return out
  i:int=0
  while i<source_m:
    fid:int=soa.funcid_arr[i]
    if fid==target_fid:
      vals:List[int]=funcid_mark_effective_values_from_soa(soa,i)
      if funcid_mark_bucket_match(bucket,vals[0],vals[1],vals[2],vals[3],vals[4],vals[5],vals[6]):
        out.append(source[i])
    i+=1
  return out

def summarize_funcid_mark_bucket_from_soa(soa:TaskSoA,source_m:int,target_fid:int,bucket:str)->List[int]:
  count:int=0
  m1min:int=999999
  m1max:int=-999999
  m2min:int=999999
  m2max:int=-999999
  gapmin:int=999999
  gapmax:int=-999999
  rtm1min:int=999999
  rtm1max:int=-999999
  rtm2min:int=999999
  rtm2max:int=-999999
  jmin:int=999999
  jmax:int=-999999
  endmin:int=999999
  endmax:int=-999999
  i:int=0
  while i<source_m:
    fid:int=soa.funcid_arr[i]
    if fid==target_fid:
      vals:List[int]=funcid_mark_effective_values_from_soa(soa,i)
      if funcid_mark_bucket_match(bucket,vals[0],vals[1],vals[2],vals[3],vals[4],vals[5],vals[6]):
        count+=1
        if vals[0]<m1min:
          m1min=vals[0]
        if vals[0]>m1max:
          m1max=vals[0]
        if vals[1]<m2min:
          m2min=vals[1]
        if vals[1]>m2max:
          m2max=vals[1]
        if vals[2]<gapmin:
          gapmin=vals[2]
        if vals[2]>gapmax:
          gapmax=vals[2]
        if vals[3]<rtm1min:
          rtm1min=vals[3]
        if vals[3]>rtm1max:
          rtm1max=vals[3]
        if vals[4]<rtm2min:
          rtm2min=vals[4]
        if vals[4]>rtm2max:
          rtm2max=vals[4]
        if vals[5]<jmin:
          jmin=vals[5]
        if vals[5]>jmax:
          jmax=vals[5]
        if vals[6]<endmin:
          endmin=vals[6]
        if vals[6]>endmax:
          endmax=vals[6]
    i+=1
  if count==0:
    m1min=0
    m1max=0
    m2min=0
    m2max=0
    gapmin=0
    gapmax=0
    rtm1min=0
    rtm1max=0
    rtm2min=0
    rtm2max=0
    jmin=0
    jmax=0
    endmin=0
    endmax=0
  return [count,m1min,m1max,m2min,m2max,gapmin,gapmax,rtm1min,rtm1max,rtm2min,rtm2max,jmin,jmax,endmin,endmax]

def parse_funcid_markdist_axis_list_spec(spec:str)->List[str]:
  out:List[str]=[]
  s:str=spec.strip()
  if s=="" or s=="-":
    return out
  parts:List[str]=s.split(";")
  for p in parts:
    t:str=p.strip()
    if t!="":
      out.append(t)
  return out

def markdist_signed_key(prefix:str,v:int)->str:
  if v<0:
    return prefix+"n"+str(0-v)
  return prefix+str(v)

def funcid_markdist_key(axis:str,mark_gap:int,d1:int,d2:int)->str:
  a:str=axis.strip()
  if a=="" or a=="all" or a=="ALL" or a=="*":
    return "all"
  if a=="gap" or a=="g":
    return "g"+str(mark_gap)
  if a=="d1" or a=="row_to_mark1" or a=="rtm1":
    return markdist_signed_key("d1",d1)
  if a=="d2" or a=="row_to_mark2" or a=="rtm2":
    return markdist_signed_key("d2",d2)
  if a=="gap_d1" or a=="g_d1" or a=="gapx_d1":
    return "g"+str(mark_gap)+"_"+markdist_signed_key("d1",d1)
  if a=="gap_d2" or a=="g_d2" or a=="gapx_d2":
    return "g"+str(mark_gap)+"_"+markdist_signed_key("d2",d2)
  if a=="d1_d2" or a=="d1x_d2":
    return markdist_signed_key("d1",d1)+"_"+markdist_signed_key("d2",d2)
  if a=="gap_d1_d2" or a=="tuple" or a=="g_d1_d2" or a=="gapx_d1x_d2":
    return "g"+str(mark_gap)+"_"+markdist_signed_key("d1",d1)+"_"+markdist_signed_key("d2",d2)
  return "g"+str(mark_gap)+"_"+markdist_signed_key("d1",d1)+"_"+markdist_signed_key("d2",d2)

def markdist_key_list_contains(keys:List[str],key:str)->bool:
  i:int=0
  while i<len(keys):
    if keys[i]==key:
      return True
    i+=1
  return False

def collect_funcid_markdist_keys_from_soa(soa:TaskSoA,source_m:int,target_fid:int,axis:str)->List[str]:
  keys:List[str]=[]
  i:int=0
  while i<source_m:
    fid:int=soa.funcid_arr[i]
    if fid==target_fid:
      vals:List[int]=funcid_mark_effective_values_from_soa(soa,i)
      key:str=funcid_markdist_key(axis,vals[2],vals[3],vals[4])
      if not markdist_key_list_contains(keys,key):
        keys.append(key)
    i+=1
  return keys

def filter_constellations_by_funcid_markdist_key_from_soa(source:List[Dict[str,int]],soa:TaskSoA,source_m:int,target_fid:int,axis:str,key:str)->List[Dict[str,int]]:
  out:List[Dict[str,int]]=[]
  if source_m<=0:
    return out
  i:int=0
  while i<source_m:
    fid:int=soa.funcid_arr[i]
    if fid==target_fid:
      vals:List[int]=funcid_mark_effective_values_from_soa(soa,i)
      k:str=funcid_markdist_key(axis,vals[2],vals[3],vals[4])
      if k==key:
        out.append(source[i])
    i+=1
  return out

def summarize_funcid_markdist_key_from_soa(soa:TaskSoA,source_m:int,target_fid:int,axis:str,key:str)->List[int]:
  count:int=0
  m1min:int=999999
  m1max:int=-999999
  m2min:int=999999
  m2max:int=-999999
  gapmin:int=999999
  gapmax:int=-999999
  d1min:int=999999
  d1max:int=-999999
  d2min:int=999999
  d2max:int=-999999
  jmin:int=999999
  jmax:int=-999999
  endmin:int=999999
  endmax:int=-999999
  i:int=0
  while i<source_m:
    fid:int=soa.funcid_arr[i]
    if fid==target_fid:
      vals:List[int]=funcid_mark_effective_values_from_soa(soa,i)
      k:str=funcid_markdist_key(axis,vals[2],vals[3],vals[4])
      if k==key:
        count+=1
        if vals[0]<m1min:
          m1min=vals[0]
        if vals[0]>m1max:
          m1max=vals[0]
        if vals[1]<m2min:
          m2min=vals[1]
        if vals[1]>m2max:
          m2max=vals[1]
        if vals[2]<gapmin:
          gapmin=vals[2]
        if vals[2]>gapmax:
          gapmax=vals[2]
        if vals[3]<d1min:
          d1min=vals[3]
        if vals[3]>d1max:
          d1max=vals[3]
        if vals[4]<d2min:
          d2min=vals[4]
        if vals[4]>d2max:
          d2max=vals[4]
        if vals[5]<jmin:
          jmin=vals[5]
        if vals[5]>jmax:
          jmax=vals[5]
        if vals[6]<endmin:
          endmin=vals[6]
        if vals[6]>endmax:
          endmax=vals[6]
    i+=1
  if count==0:
    m1min=0
    m1max=0
    m2min=0
    m2max=0
    gapmin=0
    gapmax=0
    d1min=0
    d1max=0
    d2min=0
    d2max=0
    jmin=0
    jmax=0
    endmin=0
    endmax=0
  return [count,m1min,m1max,m2min,m2max,gapmin,gapmax,d1min,d1max,d2min,d2max,jmin,jmax,endmin,endmax]

def funcid_target_group_match(group:str,fid:int)->bool:
  g:str=group.strip()
  if g=="all":
    return True
  if g=="other" or g=="rest":
    return not (fid==5 or fid==7 or fid==23)
  if g=="risky_a":
    return fid==19 or fid==22 or fid==23 or fid==24
  if g=="risky_b":
    return fid==26 or fid==27
  if g=="risky_c":
    return fid==20 or fid==21
  if g=="good":
    return fid==0 or fid==4 or fid==5 or fid==12 or fid==16 or fid==17 or fid==18
  if g=="risk_other":
    return funcid_reorder_bucket(fid)==4

  parts:List[str]=g.split("+")
  if len(parts)<=1:
    parts=g.split(",")
  for p in parts:
    t:str=p.strip()
    if t=="":
      continue
    if len(t)>=2 and (t[0:1]=="f" or t[0:1]=="F"):
      t=t[1:]
    v:int=int(t)
    if fid==v:
      return True
  return False

def filter_constellations_by_funcid_target_group(N:int,source:List[Dict[str,int]],group:str)->List[Dict[str,int]]:
  out:List[Dict[str,int]]=[]
  m:int=len(source)
  if m<=0:
    return out
  g:str=group.strip()
  if g=="all":
    return source

  soa:TaskSoA=TaskSoA(m)
  w_arr:List[u64]=[u64(0)]*m
  build_soa_for_range(N,source,0,m,soa,w_arr)
  i:int=0
  while i<m:
    fid:int=soa.funcid_arr[i]
    if funcid_target_group_match(g,fid):
      out.append(source[i])
    i+=1
  return out

def filter_constellations_by_single_funcid_from_soa(source:List[Dict[str,int]],soa:TaskSoA,source_m:int,target_fid:int)->List[Dict[str,int]]:
  out:List[Dict[str,int]]=[]
  if target_fid<0 or target_fid>=28:
    return out
  i:int=0
  while i<source_m:
    if soa.funcid_arr[i]==target_fid:
      out.append(source[i])
    i+=1
  return out

def chunk_list_contains(chunks:List[int],chunk_index:int)->bool:
  for v in chunks:
    if v==chunk_index:
      return True
  return False

def chunk_list_max(chunks:List[int])->int:
  mx:int=-1
  for v in chunks:
    if v>mx:
      mx=v
  return mx

def chunk_list_to_string(chunks:List[int])->str:
  s:str=""
  first:bool=True
  for v in chunks:
    if first:
      s=str(v)
      first=False
    else:
      s+=","+str(v)
  return s

FUNCID_REORDER_V2_SWEEP_TEMP_OUTPUT:bool=False

def funcid_reorder_param_tag()->str:
  return f"w{FUNCID_REORDER_V2_WINDOW_MULT}_j{FUNCID_REORDER_V2_PHASE_JUMP}"

def stream_chunk_param_tag(block:int,max_blocks:int)->str:
  b:int=block
  mb:int=max_blocks
  if b<=0:
    b=32
  if mb<=0:
    mb=484
  steps:int=b*mb
  if steps<=0:
    steps=15488
  return f"b{b}_m{mb}_s{steps}"

def funcid_reorder_run_param_tag(block:int,max_blocks:int)->str:
  return f"{funcid_reorder_param_tag()}_{stream_chunk_param_tag(block,max_blocks)}"

def fill_constellation_buffer_from_file(f:T,buf:List[Dict[str,int]],target:int,T:type)->int:
  added:int=0
  if target<0:
    target=0
  while len(buf)<target:
    if append_one_constellation_from_file(f,buf):
      added+=1
    else:
      break
  return added

def take_striped_records_from_buffer(buf:List[Dict[str,int]],q:int,chunk_index:int,group_id:int)->Tuple[List[Dict[str,int]],List[Dict[str,int]]]:
  taken_records:List[Dict[str,int]]=[]
  n:int=len(buf)
  if q<=0 or n<=0:
    return taken_records,buf
  if q>n:
    q=n

  selected:List[bool]=[False]*n
  step:int=n//q
  if step<=0:
    step=1

  start:int=(chunk_index*FUNCID_REORDER_V2_PHASE_JUMP+group_id*3)%step
  idx:int=start
  taken:int=0
  guard:int=0
  guard_limit:int=n*2+q+16

  while taken<q and guard<guard_limit:
    if not selected[idx]:
      selected[idx]=True
      taken_records.append(buf[idx])
      taken+=1
    idx+=step
    if idx>=n:
      idx=idx%n
    guard+=1

  i:int=0
  while taken<q and i<n:
    if not selected[i]:
      selected[i]=True
      taken_records.append(buf[i])
      taken+=1
    i+=1

  newbuf:List[Dict[str,int]]=[]
  i=0
  while i<n:
    if not selected[i]:
      newbuf.append(buf[i])
    i+=1

  return taken_records,newbuf

def interleave_funcid_reorder_parts(part_b:List[Dict[str,int]],part_a:List[Dict[str,int]],part_c:List[Dict[str,int]],part_g:List[Dict[str,int]],part_o:List[Dict[str,int]],m_target:int)->List[Dict[str,int]]:
  out:List[Dict[str,int]]=[]
  ib:int=0
  ia:int=0
  ic:int=0
  ig:int=0
  io:int=0
  while len(out)<m_target:
    progressed:bool=False
    if ib<len(part_b):
      out.append(part_b[ib])
      ib+=1
      progressed=True
    if ig<len(part_g):
      out.append(part_g[ig])
      ig+=1
      progressed=True
    if ia<len(part_a):
      out.append(part_a[ia])
      ia+=1
      progressed=True
    if io<len(part_o):
      out.append(part_o[io])
      io+=1
      progressed=True
    if ic<len(part_c):
      out.append(part_c[ic])
      ic+=1
      progressed=True
    if not progressed:
      break
  return out

def build_funcid_reordered_bin(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1
)->Tuple[str,int,int]:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  counts:List[int]=build_funcid_reorder_bucket_bins(N,fname,preset_queens,BLOCK,MAX_BLOCKS,gpu_log_level)
  counted_records:int=counts[0]+counts[1]+counts[2]+counts[3]+counts[4]
  if counted_records!=total_records:
    print(f"[funcid-reorder-v2-warning] bucket count mismatch: counted={counted_records} total_records={total_records}")
    total_records=counted_records

  reorder_fname:str=funcid_reorder_output_fname(N,preset_queens)
  truncate_plain_bin(reorder_fname)

  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{funcid_reorder_param_tag()}_sim.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_funcid_reorder_progress_header())

  rem_counts:List[int]=[0]*5
  g:int=0
  while g<5:
    rem_counts[g]=counts[g]
    g+=1

  fb=open(funcid_reorder_bucket_fname(N,preset_queens,0),"rb")
  fa=open(funcid_reorder_bucket_fname(N,preset_queens,1),"rb")
  fc=open(funcid_reorder_bucket_fname(N,preset_queens,2),"rb")
  fg=open(funcid_reorder_bucket_fname(N,preset_queens,3),"rb")
  fo=open(funcid_reorder_bucket_fname(N,preset_queens,4),"rb")

  buf_b:List[Dict[str,int]]=[]
  buf_a:List[Dict[str,int]]=[]
  buf_c:List[Dict[str,int]]=[]
  buf_g:List[Dict[str,int]]=[]
  buf_o:List[Dict[str,int]]=[]

  off:int=0
  chunk_index:int=0
  total_remaining:int=total_records

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-build-config] N={N} records={total_records} steps={STEPS} output={reorder_fname} progress={progress_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP}")

  while total_remaining>0:
    m_target:int=STEPS
    if total_remaining<STEPS:
      m_target=total_remaining

    quotas:List[int]=funcid_reorder_make_quotas(rem_counts,total_remaining,m_target)
    qb:int=quotas[0]
    qa:int=quotas[1]
    qc:int=quotas[2]
    qg:int=quotas[3]
    qo:int=quotas[4]

    t0=datetime.now()

    target_b:int=qb*FUNCID_REORDER_V2_WINDOW_MULT
    target_a:int=qa*FUNCID_REORDER_V2_WINDOW_MULT
    target_c:int=qc*FUNCID_REORDER_V2_WINDOW_MULT
    target_g:int=qg*FUNCID_REORDER_V2_WINDOW_MULT
    target_o:int=qo*FUNCID_REORDER_V2_WINDOW_MULT
    if target_b<qb:
      target_b=qb
    if target_a<qa:
      target_a=qa
    if target_c<qc:
      target_c=qc
    if target_g<qg:
      target_g=qg
    if target_o<qo:
      target_o=qo
    if target_b>rem_counts[0]:
      target_b=rem_counts[0]
    if target_a>rem_counts[1]:
      target_a=rem_counts[1]
    if target_c>rem_counts[2]:
      target_c=rem_counts[2]
    if target_g>rem_counts[3]:
      target_g=rem_counts[3]
    if target_o>rem_counts[4]:
      target_o=rem_counts[4]

    fill_constellation_buffer_from_file(fb,buf_b,target_b)
    fill_constellation_buffer_from_file(fa,buf_a,target_a)
    fill_constellation_buffer_from_file(fc,buf_c,target_c)
    fill_constellation_buffer_from_file(fg,buf_g,target_g)
    fill_constellation_buffer_from_file(fo,buf_o,target_o)

    part_b:List[Dict[str,int]]=[]
    part_a:List[Dict[str,int]]=[]
    part_c:List[Dict[str,int]]=[]
    part_g:List[Dict[str,int]]=[]
    part_o:List[Dict[str,int]]=[]

    part_b,buf_b=take_striped_records_from_buffer(buf_b,qb,chunk_index,0)
    part_a,buf_a=take_striped_records_from_buffer(buf_a,qa,chunk_index,1)
    part_c,buf_c=take_striped_records_from_buffer(buf_c,qc,chunk_index,2)
    part_g,buf_g=take_striped_records_from_buffer(buf_g,qg,chunk_index,3)
    part_o,buf_o=take_striped_records_from_buffer(buf_o,qo,chunk_index,4)

    rem_counts[0]-=len(part_b)
    rem_counts[1]-=len(part_a)
    rem_counts[2]-=len(part_c)
    rem_counts[3]-=len(part_g)
    rem_counts[4]-=len(part_o)
    if rem_counts[0]<0:
      rem_counts[0]=0
    if rem_counts[1]<0:
      rem_counts[1]=0
    if rem_counts[2]<0:
      rem_counts[2]=0
    if rem_counts[3]<0:
      rem_counts[3]=0
    if rem_counts[4]<0:
      rem_counts[4]=0

    chunk_constellations:List[Dict[str,int]]=interleave_funcid_reorder_parts(part_b,part_a,part_c,part_g,part_o,m_target)
    m:int=len(chunk_constellations)
    if m==0:
      break

    append_constellations_bin(reorder_fname,chunk_constellations)
    stats:List[int]=analyze_stream_chunk_input_stats(N,chunk_constellations)
    t1:datetime=datetime.now()
    elapsed_text:str=str(t1-t0)[:-3]
    elapsed_ms:int=stream_elapsed_text_to_ms(elapsed_text)
    append_stream_funcid_reorder_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,0,0,total_records,stats)

    if gpu_log_level>=2:
      print(f"[funcid-reorder-v2-build-chunk] chunk={chunk_index} off={off} m={m} B={stats[18+26]+stats[18+27]} A={stats[18+19]+stats[18+22]+stats[18+23]+stats[18+24]} C={stats[18+20]+stats[18+21]} G={stats[18+0]+stats[18+4]+stats[18+5]+stats[18+12]+stats[18+16]+stats[18+17]+stats[18+18]}")

    off+=m
    chunk_index+=1
    total_remaining=total_records-off

  fb.close()
  fa.close()
  fc.close()
  fg.close()
  fo.close()

  write_stream_done_count(reorder_fname+".done",off)
  reordered_records:int=count_constellations_bin_records(reorder_fname)
  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-build-summary] N={N} records={reordered_records} chunks={chunk_index} output={reorder_fname} progress={progress_fname} valid={1 if validate_bin_file(reorder_fname) else 0}")

  return reorder_fname,reordered_records,chunk_index

def funcid_markdist_risk_reorder_bucket_label(g:int)->str:
  if g==0:
    return "X"   # extreme exact tail
  if g==1:
    return "T"   # tail
  if g==2:
    return "H"   # heavy
  if g==3:
    return "M"   # medium / known-heavy fallback
  return "O"     # other

def funcid_markdist_risk_reorder_bucket_fname(N:int,preset_queens:int,g:int)->str:
  return f"constellations_N{N}_{preset_queens}_markdist_risk_reorder_{FUNCID_MARKDIST_RISK_REORDER_VERSION}_{funcid_markdist_risk_reorder_bucket_label(g)}.bin"

def funcid_markdist_risk_reorder_output_fname(N:int,preset_queens:int)->str:
  return f"constellations_N{N}_{preset_queens}_markdist_risk_reorder_{FUNCID_MARKDIST_RISK_REORDER_VERSION}_{funcid_reorder_param_tag()}.bin"

def funcid_markdist_risk_score(fid:int,mark_gap:int,d1:int)->int:
  if fid==5:
    if mark_gap==5 and d1==4:
      return 368385
    if mark_gap==3 and d1==5:
      return 283091
    if mark_gap==8 and d1==0:
      return 263545
    if mark_gap==5 and d1==5:
      return 239714
    if mark_gap==4 and d1==5:
      return 212174
    if mark_gap==7 and d1==0:
      return 170038
    if mark_gap==4 and d1==4:
      return 160513
    if mark_gap==2 and d1==4:
      return 81086
    if mark_gap==4 and d1==2:
      return 52036
    return 1928

  if fid==0:
    if mark_gap==3 and d1==0:
      return 278909
    if mark_gap==2 and d1==1:
      return 229048
    if mark_gap==2 and d1==2:
      return 21911
    if mark_gap==2 and d1==0:
      return 7626
    return 12654

  if fid==1:
    if mark_gap==3 and d1==-2:
      return 197600
    if mark_gap==4 and d1==-2:
      return 30436
    if mark_gap==2 and d1==-2:
      return 15648
    return 23653

  if fid==4:
    if mark_gap==1 and d1==2:
      return 23208
    if mark_gap==1 and d1==3:
      return 16755
    if mark_gap==1 and d1==1:
      return 15180
    if mark_gap==1 and d1==0:
      return 13930
    return 10456

  if fid==7:
    if mark_gap==1 and d1==1:
      return 22703
    if mark_gap==1 and d1==4:
      return 21049
    if mark_gap==1 and d1==3:
      return 20638
    if mark_gap==1 and d1==2:
      return 13311
    return 9606

  if fid==15:
    if mark_gap==1 and d1==3:
      return 237308
    if mark_gap==1 and d1==1:
      return 12596
    return 19458

  if fid==19:
    if mark_gap==5 and d1==0:
      return 61714
    if mark_gap==2 and d1==3:
      return 40407
    if mark_gap==2 and d1==0:
      return 7701
    return 10278

  if fid==22:
    if mark_gap==1 and d1==0:
      return 34009
    if mark_gap==1 and d1==4:
      return 27746
    if mark_gap==1 and d1==1:
      return 8152
    return 12353

  if fid==24:
    if mark_gap==1 and d1==0:
      return 94216
    if mark_gap==1 and d1==5:
      return 30767
    if mark_gap==1 and d1==2:
      return 5466
    return 9473

  if fid==12 or fid==16 or fid==17 or fid==18:
    return 5000
  if fid==20 or fid==21 or fid==23 or fid==25 or fid==26 or fid==27:
    return 3000
  return 1000

def funcid_markdist_risk_bucket_from_score(score:int)->int:
  if score>=200000:
    return 0
  if score>=50000:
    return 1
  if score>=15000:
    return 2
  if score>=5000:
    return 3
  return 4

def funcid_markdist_risk_bucket(fid:int,mark_gap:int,d1:int)->int:
  return funcid_markdist_risk_bucket_from_score(funcid_markdist_risk_score(fid,mark_gap,d1))

def stream_funcid_markdist_risk_reorder_progress_header()->str:
  h:str=stream_funcid_reorder_progress_header().strip()
  h+="\tmarkrisk_x_count\tmarkrisk_x_ratio"
  h+="\tmarkrisk_t_count\tmarkrisk_t_ratio"
  h+="\tmarkrisk_h_count\tmarkrisk_h_ratio"
  h+="\tmarkrisk_m_count\tmarkrisk_m_ratio"
  h+="\tmarkrisk_o_count\tmarkrisk_o_ratio"
  h+="\tmarkrisk_score_sum\tmarkrisk_score_avg\tmarkrisk_score_min\tmarkrisk_score_max"
  h+="\n"
  return h

def analyze_markdist_risk_stats_from_soa(soa:TaskSoA,m:int)->List[int]:
  out:List[int]=[0]*8
  if m<=0:
    return out
  out[6]=999999999
  out[7]=0
  i:int=0
  while i<m:
    fid:int=soa.funcid_arr[i]
    vals:List[int]=funcid_mark_effective_values_from_soa(soa,i)
    score:int=funcid_markdist_risk_score(fid,vals[2],vals[3])
    b:int=funcid_markdist_risk_bucket_from_score(score)
    if b<0 or b>4:
      b=4
    out[b]+=1
    out[5]+=score
    if score<out[6]:
      out[6]=score
    if score>out[7]:
      out[7]=score
    i+=1
  if out[6]==999999999:
    out[6]=0
  return out

def analyze_markdist_risk_stats(N:int,chunk_constellations:List[Dict[str,int]])->List[int]:
  m:int=len(chunk_constellations)
  stats:List[int]=[0]*8
  if m<=0:
    return stats
  soa:TaskSoA=TaskSoA(m)
  w_arr:List[u64]=[u64(0)]*m
  build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)
  stats=analyze_markdist_risk_stats_from_soa(soa,m)
  return stats

def stream_funcid_markdist_risk_reorder_suffix(risk_stats:List[int],m:int)->str:
  s:str=""
  s+=f"\t{risk_stats[0]}\t{format_ratio_3(risk_stats[0],m)}"
  s+=f"\t{risk_stats[1]}\t{format_ratio_3(risk_stats[1],m)}"
  s+=f"\t{risk_stats[2]}\t{format_ratio_3(risk_stats[2],m)}"
  s+=f"\t{risk_stats[3]}\t{format_ratio_3(risk_stats[3],m)}"
  s+=f"\t{risk_stats[4]}\t{format_ratio_3(risk_stats[4],m)}"
  s+=f"\t{risk_stats[5]}\t{format_ratio_3(risk_stats[5],m)}\t{risk_stats[6]}\t{risk_stats[7]}"
  return s

def append_stream_funcid_markdist_risk_reorder_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,total_records:int,stats:List[int],risk_stats:List[int])->None:
  done_records:int=off+m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{0}\t{0}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,m))
    f.write(stream_funcid_reorder_risk_suffix(stats,m))
    f.write(stream_funcid_markdist_risk_reorder_suffix(risk_stats,m))
    f.write("\n")

def build_funcid_markdist_risk_reorder_bucket_bins(N:int,fname:str,preset_queens:int,BLOCK:int,MAX_BLOCKS:int,gpu_log_level:int=0)->List[int]:
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  g:int=0
  while g<5:
    truncate_plain_bin(funcid_markdist_risk_reorder_bucket_fname(N,preset_queens,g))
    g+=1

  counts:List[int]=[0]*5
  chunk_index:int=0
  _read_uint32_le=read_uint32_le

  if gpu_log_level>=1:
    print(f"[markdist-risk-reorder-bucket-config] N={N} bin={fname} steps={STEPS} reason={FUNCID_MARKDIST_RISK_REORDER_DEFAULT_REASON}")

  with open(fname,"rb") as f:
    while True:
      chunk_constellations:List[Dict[str,int]]=[]
      i:int=0
      while i<STEPS:
        raw:str=f.read(16)
        if len(raw)<16:
          break
        ld:int=_read_uint32_le(raw[0:4])
        rd:int=_read_uint32_le(raw[4:8])
        col:int=_read_uint32_le(raw[8:12])
        startijkl:int=_read_uint32_le(raw[12:16])
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1

      m:int=len(chunk_constellations)
      if m==0:
        break

      soa:TaskSoA=TaskSoA(m)
      w_arr:List[u64]=[u64(0)]*m
      build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)

      bucket_x:List[Dict[str,int]]=[]
      bucket_t:List[Dict[str,int]]=[]
      bucket_h:List[Dict[str,int]]=[]
      bucket_m:List[Dict[str,int]]=[]
      bucket_o:List[Dict[str,int]]=[]

      i=0
      while i<m:
        fid:int=soa.funcid_arr[i]
        vals:List[int]=funcid_mark_effective_values_from_soa(soa,i)
        bg:int=funcid_markdist_risk_bucket(fid,vals[2],vals[3])
        if bg==0:
          bucket_x.append(chunk_constellations[i])
          counts[0]+=1
        elif bg==1:
          bucket_t.append(chunk_constellations[i])
          counts[1]+=1
        elif bg==2:
          bucket_h.append(chunk_constellations[i])
          counts[2]+=1
        elif bg==3:
          bucket_m.append(chunk_constellations[i])
          counts[3]+=1
        else:
          bucket_o.append(chunk_constellations[i])
          counts[4]+=1
        i+=1

      if len(bucket_x)>0:
        append_constellations_bin(funcid_markdist_risk_reorder_bucket_fname(N,preset_queens,0),bucket_x)
      if len(bucket_t)>0:
        append_constellations_bin(funcid_markdist_risk_reorder_bucket_fname(N,preset_queens,1),bucket_t)
      if len(bucket_h)>0:
        append_constellations_bin(funcid_markdist_risk_reorder_bucket_fname(N,preset_queens,2),bucket_h)
      if len(bucket_m)>0:
        append_constellations_bin(funcid_markdist_risk_reorder_bucket_fname(N,preset_queens,3),bucket_m)
      if len(bucket_o)>0:
        append_constellations_bin(funcid_markdist_risk_reorder_bucket_fname(N,preset_queens,4),bucket_o)

      if gpu_log_level>=2:
        print(f"[markdist-risk-reorder-bucket-chunk] chunk={chunk_index} m={m} X={len(bucket_x)} T={len(bucket_t)} H={len(bucket_h)} M={len(bucket_m)} O={len(bucket_o)}")
      chunk_index+=1

  if gpu_log_level>=1:
    total:int=counts[0]+counts[1]+counts[2]+counts[3]+counts[4]
    print(f"[markdist-risk-reorder-bucket-summary] N={N} records={total} X={counts[0]} T={counts[1]} H={counts[2]} M={counts[3]} O={counts[4]}")

  return counts

def interleave_funcid_markdist_risk_reorder_parts(part_x:List[Dict[str,int]],part_t:List[Dict[str,int]],part_h:List[Dict[str,int]],part_m:List[Dict[str,int]],part_o:List[Dict[str,int]],m_target:int)->List[Dict[str,int]]:
  out:List[Dict[str,int]]=[]
  ix:int=0
  it:int=0
  ih:int=0
  im:int=0
  io:int=0
  while len(out)<m_target:
    progressed:bool=False
    if ix<len(part_x):
      out.append(part_x[ix])
      ix+=1
      progressed=True
    if io<len(part_o):
      out.append(part_o[io])
      io+=1
      progressed=True
    if it<len(part_t):
      out.append(part_t[it])
      it+=1
      progressed=True
    if im<len(part_m):
      out.append(part_m[im])
      im+=1
      progressed=True
    if ih<len(part_h):
      out.append(part_h[ih])
      ih+=1
      progressed=True
    if not progressed:
      break
  return out

def build_funcid_markdist_risk_reordered_bin(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1
)->Tuple[str,int,int]:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  counts:List[int]=build_funcid_markdist_risk_reorder_bucket_bins(N,fname,preset_queens,BLOCK,MAX_BLOCKS,gpu_log_level)
  counted_records:int=counts[0]+counts[1]+counts[2]+counts[3]+counts[4]
  if counted_records!=total_records:
    print(f"[markdist-risk-reorder-warning] bucket count mismatch: counted={counted_records} total_records={total_records}")
    total_records=counted_records

  reorder_fname:str=funcid_markdist_risk_reorder_output_fname(N,preset_queens)
  truncate_plain_bin(reorder_fname)

  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_markdist_risk_reorder_{FUNCID_MARKDIST_RISK_REORDER_VERSION}_{funcid_reorder_param_tag()}_sim.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_funcid_markdist_risk_reorder_progress_header())

  rem_counts:List[int]=[0]*5
  g:int=0
  while g<5:
    rem_counts[g]=counts[g]
    g+=1

  fx=open(funcid_markdist_risk_reorder_bucket_fname(N,preset_queens,0),"rb")
  ft=open(funcid_markdist_risk_reorder_bucket_fname(N,preset_queens,1),"rb")
  fh=open(funcid_markdist_risk_reorder_bucket_fname(N,preset_queens,2),"rb")
  fm=open(funcid_markdist_risk_reorder_bucket_fname(N,preset_queens,3),"rb")
  fo=open(funcid_markdist_risk_reorder_bucket_fname(N,preset_queens,4),"rb")

  buf_x:List[Dict[str,int]]=[]
  buf_t:List[Dict[str,int]]=[]
  buf_h:List[Dict[str,int]]=[]
  buf_m:List[Dict[str,int]]=[]
  buf_o:List[Dict[str,int]]=[]

  off:int=0
  chunk_index:int=0
  total_remaining:int=total_records

  if gpu_log_level>=1:
    print(f"[markdist-risk-reorder-build-config] N={N} records={total_records} steps={STEPS} output={reorder_fname} progress={progress_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} reason={FUNCID_MARKDIST_RISK_REORDER_DEFAULT_REASON}")

  while total_remaining>0:
    m_target:int=STEPS
    if total_remaining<STEPS:
      m_target=total_remaining

    quotas:List[int]=funcid_reorder_make_quotas(rem_counts,total_remaining,m_target)
    qx:int=quotas[0]
    qt:int=quotas[1]
    qh:int=quotas[2]
    qm:int=quotas[3]
    qo:int=quotas[4]

    t0:datetime=datetime.now()

    target_x:int=qx*FUNCID_REORDER_V2_WINDOW_MULT
    target_t:int=qt*FUNCID_REORDER_V2_WINDOW_MULT
    target_h:int=qh*FUNCID_REORDER_V2_WINDOW_MULT
    target_m2:int=qm*FUNCID_REORDER_V2_WINDOW_MULT
    target_o:int=qo*FUNCID_REORDER_V2_WINDOW_MULT
    if target_x<qx:
      target_x=qx
    if target_t<qt:
      target_t=qt
    if target_h<qh:
      target_h=qh
    if target_m2<qm:
      target_m2=qm
    if target_o<qo:
      target_o=qo
    if target_x>rem_counts[0]:
      target_x=rem_counts[0]
    if target_t>rem_counts[1]:
      target_t=rem_counts[1]
    if target_h>rem_counts[2]:
      target_h=rem_counts[2]
    if target_m2>rem_counts[3]:
      target_m2=rem_counts[3]
    if target_o>rem_counts[4]:
      target_o=rem_counts[4]

    fill_constellation_buffer_from_file(fx,buf_x,target_x)
    fill_constellation_buffer_from_file(ft,buf_t,target_t)
    fill_constellation_buffer_from_file(fh,buf_h,target_h)
    fill_constellation_buffer_from_file(fm,buf_m,target_m2)
    fill_constellation_buffer_from_file(fo,buf_o,target_o)

    part_x:List[Dict[str,int]]=[]
    part_t:List[Dict[str,int]]=[]
    part_h:List[Dict[str,int]]=[]
    part_m:List[Dict[str,int]]=[]
    part_o:List[Dict[str,int]]=[]

    part_x,buf_x=take_striped_records_from_buffer(buf_x,qx,chunk_index,0)
    part_t,buf_t=take_striped_records_from_buffer(buf_t,qt,chunk_index,1)
    part_h,buf_h=take_striped_records_from_buffer(buf_h,qh,chunk_index,2)
    part_m,buf_m=take_striped_records_from_buffer(buf_m,qm,chunk_index,3)
    part_o,buf_o=take_striped_records_from_buffer(buf_o,qo,chunk_index,4)

    rem_counts[0]-=len(part_x)
    rem_counts[1]-=len(part_t)
    rem_counts[2]-=len(part_h)
    rem_counts[3]-=len(part_m)
    rem_counts[4]-=len(part_o)
    if rem_counts[0]<0:
      rem_counts[0]=0
    if rem_counts[1]<0:
      rem_counts[1]=0
    if rem_counts[2]<0:
      rem_counts[2]=0
    if rem_counts[3]<0:
      rem_counts[3]=0
    if rem_counts[4]<0:
      rem_counts[4]=0

    chunk_constellations:List[Dict[str,int]]=interleave_funcid_markdist_risk_reorder_parts(part_x,part_t,part_h,part_m,part_o,m_target)
    m:int=len(chunk_constellations)
    if m==0:
      break

    append_constellations_bin(reorder_fname,chunk_constellations)
    stats:List[int]=analyze_stream_chunk_input_stats(N,chunk_constellations)
    risk_stats:List[int]=analyze_markdist_risk_stats(N,chunk_constellations)
    t1:datetime=datetime.now()
    elapsed_text:str=str(t1-t0)[:-3]
    elapsed_ms:int=stream_elapsed_text_to_ms(elapsed_text)
    append_stream_funcid_markdist_risk_reorder_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,total_records,stats,risk_stats)

    if gpu_log_level>=2:
      print(f"[markdist-risk-reorder-build-chunk] chunk={chunk_index} off={off} m={m} X={risk_stats[0]} T={risk_stats[1]} H={risk_stats[2]} M={risk_stats[3]} O={risk_stats[4]} score_avg={format_ratio_3(risk_stats[5],m)}")

    off+=m
    chunk_index+=1
    total_remaining=total_records-off

  fx.close()
  ft.close()
  fh.close()
  fm.close()
  fo.close()

  write_stream_done_count(reorder_fname+".done",off)
  reordered_records:int=count_constellations_bin_records(reorder_fname)
  if gpu_log_level>=1:
    print(f"[markdist-risk-reorder-build-summary] N={N} records={reordered_records} chunks={chunk_index} output={reorder_fname} progress={progress_fname} valid={1 if validate_bin_file(reorder_fname) else 0}")

  return reorder_fname,reordered_records,chunk_index

def broad_markdist_reorder_cell_fname(N:int,preset_queens:int,broad:int,risk:int)->str:
  return f"constellations_N{N}_{preset_queens}_broadmark_reorder_{BROAD_MARKDIST_REORDER_VERSION}_{funcid_reorder_bucket_label(broad)}_{funcid_markdist_risk_reorder_bucket_label(risk)}.bin"

def broad_markdist_reorder_output_fname(N:int,preset_queens:int)->str:
  return f"constellations_N{N}_{preset_queens}_broadmark_reorder_{BROAD_MARKDIST_REORDER_VERSION}_{funcid_reorder_param_tag()}.bin"

def broad_markdist_cell_index(broad:int,risk:int)->int:
  return broad*5+risk

def make_broad_markdist_cell_buffers()->List[List[Dict[str,int]]]:
  out:List[List[Dict[str,int]]]=[]
  i:int=0
  while i<25:
    one:List[Dict[str,int]]=[]
    out.append(one)
    i+=1
  return out

def fill_constellation_buffer_from_bin_range(fname:str,buf:List[Dict[str,int]],off_record:int,target:int)->Tuple[List[Dict[str,int]],int]:
  if target<0:
    target=0
  while len(buf)<target:
    need:int=target-len(buf)
    chunk:List[Dict[str,int]]=read_constellations_bin_range(fname,off_record,need)
    got:int=len(chunk)
    if got<=0:
      break
    i:int=0
    while i<got:
      buf.append(chunk[i])
      i+=1
    off_record+=got
  return buf,off_record

def analyze_broad_markdist_cell_stats_from_soa(soa:TaskSoA,m:int)->List[int]:
  out:List[int]=[0]*25
  i:int=0
  while i<m:
    fid:int=soa.funcid_arr[i]
    broad:int=funcid_reorder_bucket(fid)
    vals:List[int]=funcid_mark_effective_values_from_soa(soa,i)
    risk:int=funcid_markdist_risk_bucket(fid,vals[2],vals[3])
    if broad<0 or broad>4:
      broad=4
    if risk<0 or risk>4:
      risk=4
    out[broad_markdist_cell_index(broad,risk)]+=1
    i+=1
  return out

def analyze_broad_markdist_cell_stats(N:int,chunk_constellations:List[Dict[str,int]])->List[int]:
  m:int=len(chunk_constellations)
  out:List[int]=[0]*25
  if m<=0:
    return out
  soa:TaskSoA=TaskSoA(m)
  w_arr:List[u64]=[u64(0)]*m
  build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)
  out=analyze_broad_markdist_cell_stats_from_soa(soa,m)
  return out

def stream_broad_markdist_reorder_progress_header()->str:
  h:str=stream_funcid_markdist_risk_reorder_progress_header().strip()
  broad:int=0
  while broad<5:
    risk:int=0
    while risk<5:
      h+=f"\tcell_{funcid_reorder_bucket_label(broad)}_{funcid_markdist_risk_reorder_bucket_label(risk)}_count"
      risk+=1
    broad+=1
  h+="\n"
  return h

def stream_broad_markdist_cell_suffix(cell_stats:List[int])->str:
  s:str=""
  i:int=0
  while i<25:
    v:int=0
    if i<len(cell_stats):
      v=cell_stats[i]
    s+=f"\t{v}"
    i+=1
  return s

def append_stream_broad_markdist_reorder_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,total_records:int,stats:List[int],risk_stats:List[int],cell_stats:List[int])->None:
  done_records:int=off+m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{0}\t{0}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,m))
    f.write(stream_funcid_reorder_risk_suffix(stats,m))
    f.write(stream_funcid_markdist_risk_reorder_suffix(risk_stats,m))
    f.write(stream_broad_markdist_cell_suffix(cell_stats))
    f.write("\n")

def build_broad_markdist_reorder_cell_bins(N:int,fname:str,preset_queens:int,BLOCK:int,MAX_BLOCKS:int,gpu_log_level:int=0)->List[int]:
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  cell:int=0
  while cell<25:
    broad:int=cell//5
    risk:int=cell%5
    truncate_plain_bin(broad_markdist_reorder_cell_fname(N,preset_queens,broad,risk))
    cell+=1

  counts:List[int]=[0]*25
  chunk_index:int=0
  _read_uint32_le=read_uint32_le

  if gpu_log_level>=1:
    print(f"[broadmark-reorder-cell-config] N={N} bin={fname} steps={STEPS} reason={BROAD_MARKDIST_REORDER_DEFAULT_REASON}")

  with open(fname,"rb") as f:
    while True:
      chunk_constellations:List[Dict[str,int]]=[]
      i:int=0
      while i<STEPS:
        raw:str=f.read(16)
        if len(raw)<16:
          break
        ld:int=_read_uint32_le(raw[0:4])
        rd:int=_read_uint32_le(raw[4:8])
        col:int=_read_uint32_le(raw[8:12])
        startijkl:int=_read_uint32_le(raw[12:16])
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1

      m:int=len(chunk_constellations)
      if m==0:
        break

      soa:TaskSoA=TaskSoA(m)
      w_arr:List[u64]=[u64(0)]*m
      build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)

      buckets:List[List[Dict[str,int]]]=make_broad_markdist_cell_buffers()
      i=0
      while i<m:
        fid:int=soa.funcid_arr[i]
        broad:int=funcid_reorder_bucket(fid)
        vals:List[int]=funcid_mark_effective_values_from_soa(soa,i)
        risk:int=funcid_markdist_risk_bucket(fid,vals[2],vals[3])
        if broad<0 or broad>4:
          broad=4
        if risk<0 or risk>4:
          risk=4
        cell=broad_markdist_cell_index(broad,risk)
        buckets[cell].append(chunk_constellations[i])
        counts[cell]+=1
        i+=1

      cell=0
      while cell<25:
        if len(buckets[cell])>0:
          broad=cell//5
          risk=cell%5
          append_constellations_bin(broad_markdist_reorder_cell_fname(N,preset_queens,broad,risk),buckets[cell])
        cell+=1

      if gpu_log_level>=2:
        broad_b:int=0
        broad_a:int=0
        broad_c:int=0
        broad_g:int=0
        broad_o:int=0
        risk_x:int=0
        risk_t:int=0
        risk_h:int=0
        risk_m:int=0
        risk_o:int=0
        cell=0
        while cell<25:
          c:int=len(buckets[cell])
          broad=cell//5
          risk=cell%5
          if broad==0:
            broad_b+=c
          elif broad==1:
            broad_a+=c
          elif broad==2:
            broad_c+=c
          elif broad==3:
            broad_g+=c
          else:
            broad_o+=c
          if risk==0:
            risk_x+=c
          elif risk==1:
            risk_t+=c
          elif risk==2:
            risk_h+=c
          elif risk==3:
            risk_m+=c
          else:
            risk_o+=c
          cell+=1
        print(f"[broadmark-reorder-cell-chunk] chunk={chunk_index} m={m} B={broad_b} A={broad_a} C={broad_c} G={broad_g} O={broad_o} X={risk_x} T={risk_t} H={risk_h} M={risk_m} R={risk_o}")
      chunk_index+=1

  if gpu_log_level>=1:
    total:int=0
    cell=0
    while cell<25:
      total+=counts[cell]
      cell+=1
    print(f"[broadmark-reorder-cell-summary] N={N} records={total} B={counts[0]+counts[1]+counts[2]+counts[3]+counts[4]} A={counts[5]+counts[6]+counts[7]+counts[8]+counts[9]} C={counts[10]+counts[11]+counts[12]+counts[13]+counts[14]} G={counts[15]+counts[16]+counts[17]+counts[18]+counts[19]} O={counts[20]+counts[21]+counts[22]+counts[23]+counts[24]} X={counts[0]+counts[5]+counts[10]+counts[15]+counts[20]} T={counts[1]+counts[6]+counts[11]+counts[16]+counts[21]} H={counts[2]+counts[7]+counts[12]+counts[17]+counts[22]} M={counts[3]+counts[8]+counts[13]+counts[18]+counts[23]} R={counts[4]+counts[9]+counts[14]+counts[19]+counts[24]}")

  return counts

def broad_markdist_make_cell_quotas(rem_counts:List[int],total_remaining:int,m_target:int)->List[int]:
  quotas:List[int]=[0]*25
  if total_remaining<=0 or m_target<=0:
    return quotas

  broad_rem:List[int]=[0]*5
  broad:int=0
  while broad<5:
    risk:int=0
    while risk<5:
      broad_rem[broad]+=rem_counts[broad*5+risk]
      risk+=1
    broad+=1

  broad_quotas:List[int]=funcid_reorder_make_quotas(broad_rem,total_remaining,m_target)

  broad=0
  while broad<5:
    bq:int=broad_quotas[broad]
    if bq>0 and broad_rem[broad]>0:
      cell_rem:List[int]=[0]*5
      risk=0
      while risk<5:
        cell_rem[risk]=rem_counts[broad*5+risk]
        risk+=1
      cell_q:List[int]=funcid_reorder_make_quotas(cell_rem,broad_rem[broad],bq)
      risk=0
      while risk<5:
        quotas[broad*5+risk]=cell_q[risk]
        risk+=1
    broad+=1

  qsum:int=0
  cell:int=0
  while cell<25:
    qsum+=quotas[cell]
    cell+=1
  while qsum<m_target:
    best:int=-1
    best_room:int=-1
    cell=0
    while cell<25:
      room:int=rem_counts[cell]-quotas[cell]
      if room>best_room:
        best_room=room
        best=cell
      cell+=1
    if best<0 or best_room<=0:
      break
    quotas[best]+=1
    qsum+=1

  return quotas

def interleave_broad_markdist_secondary_parts(parts:List[List[Dict[str,int]]],broad_quotas:List[int],m_target:int)->List[Dict[str,int]]:
  part_b:List[Dict[str,int]]=interleave_funcid_markdist_risk_reorder_parts(parts[0],parts[1],parts[2],parts[3],parts[4],broad_quotas[0])
  part_a:List[Dict[str,int]]=interleave_funcid_markdist_risk_reorder_parts(parts[5],parts[6],parts[7],parts[8],parts[9],broad_quotas[1])
  part_c:List[Dict[str,int]]=interleave_funcid_markdist_risk_reorder_parts(parts[10],parts[11],parts[12],parts[13],parts[14],broad_quotas[2])
  part_g:List[Dict[str,int]]=interleave_funcid_markdist_risk_reorder_parts(parts[15],parts[16],parts[17],parts[18],parts[19],broad_quotas[3])
  part_o:List[Dict[str,int]]=interleave_funcid_markdist_risk_reorder_parts(parts[20],parts[21],parts[22],parts[23],parts[24],broad_quotas[4])
  return interleave_funcid_reorder_parts(part_b,part_a,part_c,part_g,part_o,m_target)

def build_broad_markdist_reordered_bin(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1
)->Tuple[str,int,int]:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  counts:List[int]=build_broad_markdist_reorder_cell_bins(N,fname,preset_queens,BLOCK,MAX_BLOCKS,gpu_log_level)
  counted_records:int=0
  cell:int=0
  while cell<25:
    counted_records+=counts[cell]
    cell+=1
  if counted_records!=total_records:
    print(f"[broadmark-reorder-warning] cell count mismatch: counted={counted_records} total_records={total_records}")
    total_records=counted_records

  reorder_fname:str=broad_markdist_reorder_output_fname(N,preset_queens)
  truncate_plain_bin(reorder_fname)

  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_broadmark_reorder_{BROAD_MARKDIST_REORDER_VERSION}_{funcid_reorder_param_tag()}_sim.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_broad_markdist_reorder_progress_header())

  rem_counts:List[int]=[0]*25
  read_offsets:List[int]=[0]*25
  cell_buffers:List[List[Dict[str,int]]]=make_broad_markdist_cell_buffers()
  cell=0
  while cell<25:
    rem_counts[cell]=counts[cell]
    read_offsets[cell]=0
    cell+=1

  off:int=0
  chunk_index:int=0
  total_remaining:int=total_records

  if gpu_log_level>=1:
    print(f"[broadmark-reorder-build-config] N={N} records={total_records} steps={STEPS} output={reorder_fname} progress={progress_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} reason={BROAD_MARKDIST_REORDER_DEFAULT_REASON}")

  while total_remaining>0:
    m_target:int=STEPS
    if total_remaining<STEPS:
      m_target=total_remaining

    quotas:List[int]=broad_markdist_make_cell_quotas(rem_counts,total_remaining,m_target)
    broad_quotas:List[int]=[0]*5
    cell=0
    while cell<25:
      broad:int=cell//5
      broad_quotas[broad]+=quotas[cell]
      cell+=1

    t0:datetime=datetime.now()

    parts:List[List[Dict[str,int]]]=make_broad_markdist_cell_buffers()
    cell=0
    while cell<25:
      q:int=quotas[cell]
      if q>0:
        broad=cell//5
        risk:int=cell%5
        target:int=q*FUNCID_REORDER_V2_WINDOW_MULT
        if target<q:
          target=q
        if target>rem_counts[cell]:
          target=rem_counts[cell]
        fname_cell:str=broad_markdist_reorder_cell_fname(N,preset_queens,broad,risk)
        newbuf:List[Dict[str,int]]=[]
        cell_buffers[cell],read_offsets[cell]=fill_constellation_buffer_from_bin_range(fname_cell,cell_buffers[cell],read_offsets[cell],target)
        parts[cell],newbuf=take_striped_records_from_buffer(cell_buffers[cell],q,chunk_index,cell)
        cell_buffers[cell]=newbuf
        rem_counts[cell]-=len(parts[cell])
        if rem_counts[cell]<0:
          rem_counts[cell]=0
      cell+=1

    chunk_constellations:List[Dict[str,int]]=interleave_broad_markdist_secondary_parts(parts,broad_quotas,m_target)
    m:int=len(chunk_constellations)
    if m==0:
      break

    append_constellations_bin(reorder_fname,chunk_constellations)
    stats:List[int]=analyze_stream_chunk_input_stats(N,chunk_constellations)
    risk_stats:List[int]=analyze_markdist_risk_stats(N,chunk_constellations)
    cell_stats:List[int]=analyze_broad_markdist_cell_stats(N,chunk_constellations)
    t1:datetime=datetime.now()
    elapsed_text:str=str(t1-t0)[:-3]
    elapsed_ms:int=stream_elapsed_text_to_ms(elapsed_text)
    append_stream_broad_markdist_reorder_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,total_records,stats,risk_stats,cell_stats)

    if gpu_log_level>=2:
      print(f"[broadmark-reorder-build-chunk] chunk={chunk_index} off={off} m={m} B={stats[18+26]+stats[18+27]} A={stats[18+19]+stats[18+22]+stats[18+23]+stats[18+24]} C={stats[18+20]+stats[18+21]} G={stats[18+0]+stats[18+4]+stats[18+5]+stats[18+12]+stats[18+16]+stats[18+17]+stats[18+18]} O={m-(stats[18+26]+stats[18+27]+stats[18+19]+stats[18+22]+stats[18+23]+stats[18+24]+stats[18+20]+stats[18+21]+stats[18+0]+stats[18+4]+stats[18+5]+stats[18+12]+stats[18+16]+stats[18+17]+stats[18+18])} X={risk_stats[0]} T={risk_stats[1]} H={risk_stats[2]} M={risk_stats[3]} R={risk_stats[4]} score_avg={format_ratio_3(risk_stats[5],m)}")

    off+=m
    chunk_index+=1
    total_remaining=total_records-off

  write_stream_done_count(reorder_fname+".done",off)
  reordered_records:int=count_constellations_bin_records(reorder_fname)
  if gpu_log_level>=1:
    print(f"[broadmark-reorder-build-summary] N={N} records={reordered_records} chunks={chunk_index} output={reorder_fname} progress={progress_fname} valid={1 if validate_bin_file(reorder_fname) else 0}")

  return reorder_fname,reordered_records,chunk_index

def broad_markdist_tail_label(tail:int)->str:
  if tail==0:
    return "F17"
  if tail==1:
    return "GH"
  return "R"

def broad_markdist_tail_reorder_subcell_fname(N:int,preset_queens:int,broad:int,risk:int,tail:int)->str:
  return f"constellations_N{N}_{preset_queens}_broadmarktail_reorder_{BROAD_MARKDIST_TAIL_REORDER_VERSION}_{funcid_reorder_bucket_label(broad)}_{funcid_markdist_risk_reorder_bucket_label(risk)}_{broad_markdist_tail_label(tail)}.bin"

def broad_markdist_tail_reorder_output_fname(N:int,preset_queens:int,block:int=32,max_blocks:int=484)->str:
  return f"constellations_N{N}_{preset_queens}_broadmarktail_reorder_{BROAD_MARKDIST_TAIL_REORDER_VERSION}_{broad_markdist_tail_variant_tag()}_{funcid_reorder_run_param_tag(block,max_blocks)}.bin"

CHUNKSHAPE148_REORDER_VERSION:str="scorestripe_v9_lanephase32_octetfirstpairlock29"
CHUNKSHAPE148_DEFAULT_REASON:str="232 cleanup-only reuses validated 231/217 scorestripe_v9 task order/cache; MAXD14 kernel unchanged"

def chunkshape148_reorder_output_fname(N:int,preset_queens:int,block:int=32,max_blocks:int=484)->str:
  return f"constellations_N{N}_{preset_queens}_chunkshape148_{CHUNKSHAPE148_REORDER_VERSION}_{BROAD_MARKDIST_TAIL_REORDER_VERSION}_{broad_markdist_tail_variant_tag()}_{funcid_reorder_run_param_tag(block,max_blocks)}.bin"

CHUNKSHAPE148_SCORE_KEY_MAX:int=32767
CHUNKSHAPE148_LANE_COUNT:int=32
CHUNKSHAPE148_LANE_MASK:int=31

def chunkshape148_score_key_from_soa(soa:TaskSoA,idx:int,global_idx:int)->int:
  fid:int=soa.funcid_arr[idx]
  pc:int=popcount_int(int(soa.free_arr[idx]))
  depth:int=soa.end_arr[idx]-soa.row_arr[idx]
  if depth<0:
    depth=0

  mark_gap:int=soa.mark2_arr[idx]-soa.mark1_arr[idx]
  if mark_gap<0:
    mark_gap=-mark_gap
  row_to_end:int=soa.end_arr[idx]-soa.row_arr[idx]
  if row_to_end<0:
    row_to_end=0
  row_to_mark1:int=soa.mark1_arr[idx]-soa.row_arr[idx]
  if row_to_mark1<0:
    row_to_mark1=0

  raw:int=0
  raw+=pc*12
  raw+=depth*7
  raw+=row_to_end*3

  if fid==26 or fid==27:
    raw+=96
  elif fid==19 or fid==22 or fid==23 or fid==24:
    raw+=72
  elif fid==20 or fid==21:
    raw+=56
  elif fid==17:
    raw+=42
  elif fid==14:
    raw+=36
  elif fid==0 or fid==4 or fid==5 or fid==12 or fid==16 or fid==18:
    raw+=20
  else:
    raw+=8

  if pc>=5:
    raw+=20
  elif pc>=4:
    raw+=12
  elif pc>=3:
    raw+=6

  if depth>=13:
    raw+=20
  elif depth>=11:
    raw+=12
  elif depth>=9:
    raw+=6

  if mark_gap>=3:
    raw+=8
  if row_to_mark1>=4:
    raw+=4

  tie:int=(global_idx*13 + soa.ijkl_arr[idx]*7 + fid*5 + pc*3 + depth) & 31
  key:int=raw*32 + tie
  if key<0:
    key=0
  if key>CHUNKSHAPE148_SCORE_KEY_MAX:
    key=CHUNKSHAPE148_SCORE_KEY_MAX
  return key

def chunkshape148_build_thresholds_from_hist(hist:List[int],total_records:int)->List[int]:
  thresholds:List[int]=[0]*8
  if total_records<=0:
    thresholds[7]=CHUNKSHAPE148_SCORE_KEY_MAX
    return thresholds

  b:int=0
  cum:int=0
  k:int=0
  while k<=CHUNKSHAPE148_SCORE_KEY_MAX:
    cum+=hist[k]
    while b<7:
      target:int=(total_records*(b+1))//8
      if target<=0:
        target=1
      if cum>=target:
        thresholds[b]=k
        b+=1
      else:
        break
    k+=1

  while b<7:
    thresholds[b]=CHUNKSHAPE148_SCORE_KEY_MAX
    b+=1
  thresholds[7]=CHUNKSHAPE148_SCORE_KEY_MAX
  return thresholds

def chunkshape148_bucket_from_key(key:int,thresholds:List[int])->int:
  b:int=0
  while b<7:
    if key<=thresholds[b]:
      return b
    b+=1
  return 7

def chunkshape148_lane_from_key(key:int)->int:
  lane:int=key&CHUNKSHAPE148_LANE_MASK
  if lane<0:
    lane=0
  if lane>=CHUNKSHAPE148_LANE_COUNT:
    lane=lane&CHUNKSHAPE148_LANE_MASK
  return lane

def chunkshape148_make_quotas(rem_counts:List[int],total_remaining:int,m_target:int)->List[int]:
  quotas:List[int]=[0]*8
  if total_remaining<=0 or m_target<=0:
    return quotas

  b:int=0
  quota_sum:int=0
  while b<8:
    q:int=(rem_counts[b]*m_target)//total_remaining
    if q>rem_counts[b]:
      q=rem_counts[b]
    quotas[b]=q
    quota_sum+=q
    b+=1

  while quota_sum<m_target:
    best:int=-1
    best_room:int=-1
    b=0
    while b<8:
      room:int=rem_counts[b]-quotas[b]
      if room>best_room:
        best_room=room
        best=b
      b+=1
    if best<0 or best_room<=0:
      break
    quotas[best]+=1
    quota_sum+=1

  return quotas

def build_chunkshape148_reordered_bin(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0
)->Tuple[str,int,int]:
  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  n_chunks:int=0
  if total_records>0:
    n_chunks=(total_records+STEPS-1)//STEPS

  out_fname:str=chunkshape148_reorder_output_fname(N,preset_queens,BLOCK,MAX_BLOCKS)
  progress_fname:str=f"progress_N{N}_{preset_queens}_chunkshape148_{CHUNKSHAPE148_REORDER_VERSION}_{BROAD_MARKDIST_TAIL_REORDER_VERSION}_{broad_markdist_tail_variant_tag()}_{funcid_reorder_run_param_tag(BLOCK,MAX_BLOCKS)}_sim.tsv"

  if gpu_log_level>=1:
    print(f"[chunkshape148-build-config] N={N} source={fname} records={total_records} chunks={n_chunks} steps={STEPS} output={out_fname} progress={progress_fname} reason={CHUNKSHAPE148_DEFAULT_REASON}")

  with open(progress_fname,"w") as pf:
    lane_header:str=""
    lane_i:int=0
    while lane_i<CHUNKSHAPE148_LANE_COUNT:
      lane_header+=f"\tl{lane_i}"
      lane_i+=1
    pf.write("N\tpreset\tchunk\tout_records\ttotal_records\telapsed\telapsed_ms\tb0\tb1\tb2\tb3\tb4\tb5\tb6\tb7"+lane_header+"\n")

  if total_records<=0:
    truncate_plain_bin(out_fname)
    write_stream_done_count(out_fname+".done",0)
    return out_fname,0,0

  data:str=""
  with open(fname,"rb") as f:
    data=f.read()

  expected_bytes:int=total_records*16
  if len(data)<expected_bytes:
    print(f"[chunkshape148-warning] source byte count shorter than expected: bytes={len(data)} expected={expected_bytes}")
    total_records=len(data)//16
    n_chunks=(total_records+STEPS-1)//STEPS

  truncate_plain_bin(out_fname)
  write_stream_done_count(out_fname+".done",0)

  score_key_by_idx:List[int]=[0]*total_records
  key_hist:List[int]=[0]*(CHUNKSHAPE148_SCORE_KEY_MAX+1)
  off:int=0
  classify_chunk:int=0
  key_min:int=CHUNKSHAPE148_SCORE_KEY_MAX
  key_max:int=0
  while off<total_records:
    m:int=STEPS
    remain:int=total_records-off
    if remain<m:
      m=remain

    chunk_constellations:List[Dict[str,int]]=[]
    i:int=0
    while i<m:
      p:int=(off+i)*16
      ld:int=read_uint32_le(data[p:p+4])
      rd:int=read_uint32_le(data[p+4:p+8])
      col:int=read_uint32_le(data[p+8:p+12])
      startijkl:int=read_uint32_le(data[p+12:p+16])
      chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
      i+=1

    soa:TaskSoA=TaskSoA(m)
    w_arr:List[u64]=[u64(0)]*m
    build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)

    i=0
    while i<m:
      key:int=chunkshape148_score_key_from_soa(soa,i,off+i)
      if key<0:
        key=0
      if key>CHUNKSHAPE148_SCORE_KEY_MAX:
        key=CHUNKSHAPE148_SCORE_KEY_MAX
      score_key_by_idx[off+i]=key
      key_hist[key]+=1
      if key<key_min:
        key_min=key
      if key>key_max:
        key_max=key
      i+=1

    if gpu_log_level>=2:
      print(f"[chunkshape148-classify-chunk] N={N} chunk={classify_chunk} off={off} m={m}")
    off+=m
    classify_chunk+=1

  score_thresholds:List[int]=chunkshape148_build_thresholds_from_hist(key_hist,total_records)
  bucket_counts:List[int]=[0]*8
  lane_counts:List[int]=[0]*CHUNKSHAPE148_LANE_COUNT
  idx_count:int=0
  while idx_count<total_records:
    b0:int=chunkshape148_bucket_from_key(score_key_by_idx[idx_count],score_thresholds)
    lane0:int=chunkshape148_lane_from_key(score_key_by_idx[idx_count])
    bucket_counts[b0]+=1
    lane_counts[lane0]+=1
    idx_count+=1

  if gpu_log_level>=1:
    print(f"[chunkshape148-score-key] N={N} records={total_records} key_min={key_min} key_max={key_max} t0={score_thresholds[0]} t1={score_thresholds[1]} t2={score_thresholds[2]} t3={score_thresholds[3]} t4={score_thresholds[4]} t5={score_thresholds[5]} t6={score_thresholds[6]}")
    print(f"[chunkshape148-score-buckets] N={N} records={total_records} b0={bucket_counts[0]} b1={bucket_counts[1]} b2={bucket_counts[2]} b3={bucket_counts[3]} b4={bucket_counts[4]} b5={bucket_counts[5]} b6={bucket_counts[6]} b7={bucket_counts[7]}")
    lane_counts_text:str=""
    lane_i:int=0
    while lane_i<CHUNKSHAPE148_LANE_COUNT:
      lane_counts_text+=f" l{lane_i}={lane_counts[lane_i]}"
      lane_i+=1
    print(f"[chunkshape148-lanes] N={N} records={total_records}{lane_counts_text}")

  bucket_lane_indices:List[List[int]]=[]
  flat:int=0
  while flat<(8*CHUNKSHAPE148_LANE_COUNT):
    one:List[int]=[]
    bucket_lane_indices.append(one)
    flat+=1

  out_ch:int=0
  staged_total:int=0
  while out_ch<n_chunks:
    base:int=0
    while base*n_chunks<STEPS:
      src_ch:int=0
      while src_ch<n_chunks:
        rem:int=(src_ch+out_ch)%n_chunks
        within:int=base*n_chunks+rem
        if within<STEPS:
          idx:int=src_ch*STEPS+within
          if idx<total_records:
            bucket:int=chunkshape148_bucket_from_key(score_key_by_idx[idx],score_thresholds)
            lane_val:int=chunkshape148_lane_from_key(score_key_by_idx[idx])
            bucket_lane_indices[bucket*CHUNKSHAPE148_LANE_COUNT+lane_val].append(idx)
            staged_total+=1
        src_ch+=1
      base+=1
    out_ch+=1

  if staged_total!=total_records:
    print(f"[chunkshape148-warning] staged count mismatch: staged={staged_total} source={total_records}")

  bucket_lane_pos:List[int]=[0]*(8*CHUNKSHAPE148_LANE_COUNT)
  bucket_lane_rem:List[int]=[0]*(8*CHUNKSHAPE148_LANE_COUNT)
  bucket_rem:List[int]=[0]*8
  b:int=0
  while b<8:
    lane:int=0
    while lane<CHUNKSHAPE148_LANE_COUNT:
      flat=b*CHUNKSHAPE148_LANE_COUNT+lane
      cnt:int=len(bucket_lane_indices[flat])
      bucket_lane_rem[flat]=cnt
      bucket_rem[b]+=cnt
      lane+=1
    b+=1

  interleave_order:List[int]=[7,0,6,1,5,2,4,3]

  written_total:int=0
  out_ch=0
  with open(out_fname,"ab") as out:
    while out_ch<n_chunks:
      t0:datetime=datetime.now()
      m_target:int=STEPS
      total_remaining:int=total_records-written_total
      if total_remaining<m_target:
        m_target=total_remaining

      quotas:List[int]=chunkshape148_make_quotas(bucket_rem,total_remaining,m_target)
      written_by_bucket:List[int]=[0]*8
      written_by_lane:List[int]=[0]*CHUNKSHAPE148_LANE_COUNT
      written_chunk:int=0

      quartet_index:int=out_ch//4
      octet_first_pair_phase_bias:int=0
      if (quartet_index&1)==0 and (out_ch&3)<2 and (((quartet_index+1)*4)+1)<n_chunks:
        octet_first_pair_phase_bias=29

      while written_chunk<m_target:
        made:int=0
        oi:int=0
        while oi<8:
          order_pos:int=(oi+out_ch)%8
          b=interleave_order[order_pos]
          if quotas[b]>0:
            pair_phase_bias:int=0
            if (out_ch&1)==0:
              pair_phase_bias=11 if (out_ch&3)==0 else 1
            phase_seed:int=(out_ch*11 + quartet_index*17 + b*13 + pair_phase_bias + octet_first_pair_phase_bias) & CHUNKSHAPE148_LANE_MASK
            start_lane:int=(phase_seed + written_by_bucket[b]*5) & CHUNKSHAPE148_LANE_MASK
            found:int=-1
            scan:int=0
            while scan<CHUNKSHAPE148_LANE_COUNT:
              lane=(start_lane+scan) & CHUNKSHAPE148_LANE_MASK
              flat=b*CHUNKSHAPE148_LANE_COUNT+lane
              if bucket_lane_rem[flat]>0:
                found=lane
                break
              scan+=1
            if found>=0:
              flat=b*CHUNKSHAPE148_LANE_COUNT+found
              pick_idx:int=bucket_lane_indices[flat][bucket_lane_pos[flat]]
              pick_p:int=pick_idx*16
              out.write(data[pick_p:pick_p+16])
              bucket_lane_pos[flat]+=1
              bucket_lane_rem[flat]-=1
              bucket_rem[b]-=1
              quotas[b]-=1
              written_by_bucket[b]+=1
              written_by_lane[found]+=1
              written_chunk+=1
              made+=1
              if written_chunk>=m_target:
                break
          oi+=1
        if made==0:
          break

      written_total+=written_chunk
      t1:datetime=datetime.now()
      elapsed_text:str=str(t1-t0)[:-3]
      elapsed_ms:int=stream_elapsed_text_to_ms(elapsed_text)
      with open(progress_fname,"a") as pf:
        lane_values:str=""
        lane_i:int=0
        while lane_i<CHUNKSHAPE148_LANE_COUNT:
          lane_values+=f"\t{written_by_lane[lane_i]}"
          lane_i+=1
        pf.write(f"{N}\t{preset_queens}\t{out_ch}\t{written_chunk}\t{total_records}\t{elapsed_text}\t{elapsed_ms}\t{written_by_bucket[0]}\t{written_by_bucket[1]}\t{written_by_bucket[2]}\t{written_by_bucket[3]}\t{written_by_bucket[4]}\t{written_by_bucket[5]}\t{written_by_bucket[6]}\t{written_by_bucket[7]}"+lane_values+"\n")
      if gpu_log_level>=2:
        lane_debug:str=""
        lane_i:int=0
        while lane_i<CHUNKSHAPE148_LANE_COUNT:
          lane_debug+=f" l{lane_i}={written_by_lane[lane_i]}"
          lane_i+=1
        print(f"[chunkshape148-build-chunk] N={N} chunk={out_ch} written={written_chunk} total_written={written_total} b0={written_by_bucket[0]} b1={written_by_bucket[1]} b2={written_by_bucket[2]} b3={written_by_bucket[3]} b4={written_by_bucket[4]} b5={written_by_bucket[5]} b6={written_by_bucket[6]} b7={written_by_bucket[7]}{lane_debug}")
      out_ch+=1

  write_stream_done_count(out_fname+".done",written_total)
  reordered_records:int=count_constellations_bin_records(out_fname)
  if written_total!=total_records:
    print(f"[chunkshape148-warning] record count mismatch: written={written_total} source={total_records}")
  if reordered_records!=written_total:
    print(f"[chunkshape148-warning] output count mismatch: file_records={reordered_records} written={written_total}")
  if gpu_log_level>=1:
    print(f"[chunkshape148-build-summary] N={N} records={reordered_records} chunks={n_chunks} output={out_fname} progress={progress_fname} valid={1 if validate_bin_file(out_fname) else 0}")

  return out_fname,reordered_records,n_chunks

def broad_markdist_tail_subcell_index(broad:int,risk:int,tail:int)->int:
  return (broad*5+risk)*3+tail

def make_broad_markdist_tail_subcell_buffers()->List[List[Dict[str,int]]]:
  out:List[List[Dict[str,int]]]=[]
  i:int=0
  while i<75:
    one:List[Dict[str,int]]=[]
    out.append(one)
    i+=1
  return out

def broad_markdist_tail_bucket(fid:int,broad:int,risk:int)->int:
  if fid==17:
    return 0
  if broad==3 and risk==2:
    return 1
  return 2

def broad_markdist_tail_phase_group_id(subcell:int,broad:int,risk:int,tail:int)->int:
  gid:int=subcell
  if broad_markdist_tail_use_phase_mix():
    gid+=broad*BROAD_MARKDIST_TAIL_CELL_SALT
    gid+=risk*BROAD_MARKDIST_TAIL_RISK_SALT
    if tail==0:
      gid+=broad_markdist_tail_phase_salt_value()
    elif tail==1:
      gid+=broad_markdist_tail_phase_salt_value()*2
  else:
    if tail==0 or tail==1:
      gid+=broad_markdist_tail_phase_salt_value()
  return gid

def analyze_broad_markdist_tail_subcell_stats_from_soa(soa:TaskSoA,m:int)->List[int]:
  out:List[int]=[0]*75
  i:int=0
  while i<m:
    fid:int=soa.funcid_arr[i]
    broad:int=funcid_reorder_bucket(fid)
    vals:List[int]=funcid_mark_effective_values_from_soa(soa,i)
    risk:int=funcid_markdist_risk_bucket(fid,vals[2],vals[3])
    if broad<0 or broad>4:
      broad=4
    if risk<0 or risk>4:
      risk=4
    tail:int=broad_markdist_tail_bucket(fid,broad,risk)
    out[broad_markdist_tail_subcell_index(broad,risk,tail)]+=1
    i+=1
  return out

def analyze_broad_markdist_tail_summary_from_soa(soa:TaskSoA,m:int)->List[int]:
  out:List[int]=[0]*4
  i:int=0
  while i<m:
    fid:int=soa.funcid_arr[i]
    broad:int=funcid_reorder_bucket(fid)
    vals:List[int]=funcid_mark_effective_values_from_soa(soa,i)
    risk:int=funcid_markdist_risk_bucket(fid,vals[2],vals[3])
    if broad<0 or broad>4:
      broad=4
    if risk<0 or risk>4:
      risk=4
    score:int=0
    if fid==17:
      out[0]+=1
      score+=4
    if broad==3 and risk==2:
      out[1]+=1
      score+=3
    if risk==2:
      out[2]+=1
      score+=1
    out[3]+=score
    i+=1
  return out

def analyze_broad_markdist_tail_summary(N:int,chunk_constellations:List[Dict[str,int]])->List[int]:
  m:int=len(chunk_constellations)
  out:List[int]=[0]*4
  if m<=0:
    return out
  soa:TaskSoA=TaskSoA(m)
  w_arr:List[u64]=[u64(0)]*m
  build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)
  out=analyze_broad_markdist_tail_summary_from_soa(soa,m)
  return out

def stream_broad_markdist_tail_reorder_progress_header()->str:
  h:str=stream_broad_markdist_reorder_progress_header().strip()
  h+="\ttail_funcid17_count"
  h+="\ttail_cell_G_H_count"
  h+="\ttail_markrisk_H_count"
  h+="\ttail_proxy_sum"
  h+="\ttail_proxy_avg"
  h+="\n"
  return h

def stream_broad_markdist_tail_summary_suffix(tail_stats:List[int],m:int)->str:
  f17:int=0
  gh:int=0
  hcnt:int=0
  proxy:int=0
  if len(tail_stats)>0:
    f17=tail_stats[0]
  if len(tail_stats)>1:
    gh=tail_stats[1]
  if len(tail_stats)>2:
    hcnt=tail_stats[2]
  if len(tail_stats)>3:
    proxy=tail_stats[3]
  return f"\t{f17}\t{gh}\t{hcnt}\t{proxy}\t{format_ratio_3(proxy,m)}"

def append_stream_broad_markdist_tail_reorder_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,total_records:int,stats:List[int],risk_stats:List[int],cell_stats:List[int],tail_stats:List[int])->None:
  done_records:int=off+m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{0}\t{0}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,m))
    f.write(stream_funcid_reorder_risk_suffix(stats,m))
    f.write(stream_funcid_markdist_risk_reorder_suffix(risk_stats,m))
    f.write(stream_broad_markdist_cell_suffix(cell_stats))
    f.write(stream_broad_markdist_tail_summary_suffix(tail_stats,m))
    f.write("\n")

def build_broad_markdist_tail_reorder_subcell_bins(N:int,fname:str,preset_queens:int,BLOCK:int,MAX_BLOCKS:int,gpu_log_level:int=0)->List[int]:
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  subcell:int=0
  while subcell<75:
    cell:int=subcell//3
    broad:int=cell//5
    risk:int=cell%5
    tail:int=subcell%3
    truncate_plain_bin(broad_markdist_tail_reorder_subcell_fname(N,preset_queens,broad,risk,tail))
    subcell+=1

  counts:List[int]=[0]*75
  chunk_index:int=0
  _read_uint32_le=read_uint32_le

  if gpu_log_level>=1:
    print(f"[broadmarktail-reorder-subcell-config] N={N} bin={fname} steps={STEPS} reason={BROAD_MARKDIST_TAIL_REORDER_DEFAULT_REASON}")

  with open(fname,"rb") as f:
    while True:
      chunk_constellations:List[Dict[str,int]]=[]
      i:int=0
      while i<STEPS:
        raw:str=f.read(16)
        if len(raw)<16:
          break
        ld:int=_read_uint32_le(raw[0:4])
        rd:int=_read_uint32_le(raw[4:8])
        col:int=_read_uint32_le(raw[8:12])
        startijkl:int=_read_uint32_le(raw[12:16])
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1

      m:int=len(chunk_constellations)
      if m==0:
        break

      soa:TaskSoA=TaskSoA(m)
      w_arr:List[u64]=[u64(0)]*m
      build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)

      buckets:List[List[Dict[str,int]]]=make_broad_markdist_tail_subcell_buffers()
      i=0
      while i<m:
        fid:int=soa.funcid_arr[i]
        broad=funcid_reorder_bucket(fid)
        vals:List[int]=funcid_mark_effective_values_from_soa(soa,i)
        risk=funcid_markdist_risk_bucket(fid,vals[2],vals[3])
        if broad<0 or broad>4:
          broad=4
        if risk<0 or risk>4:
          risk=4
        tail=broad_markdist_tail_bucket(fid,broad,risk)
        subcell=broad_markdist_tail_subcell_index(broad,risk,tail)
        buckets[subcell].append(chunk_constellations[i])
        counts[subcell]+=1
        i+=1

      subcell=0
      while subcell<75:
        if len(buckets[subcell])>0:
          cell=subcell//3
          broad=cell//5
          risk=cell%5
          tail=subcell%3
          append_constellations_bin(broad_markdist_tail_reorder_subcell_fname(N,preset_queens,broad,risk,tail),buckets[subcell])
        subcell+=1

      if gpu_log_level>=2:
        tail_stats:List[int]=analyze_broad_markdist_tail_summary_from_soa(soa,m)
        print(f"[broadmarktail-reorder-subcell-chunk] chunk={chunk_index} m={m} funcid17={tail_stats[0]} cell_G_H={tail_stats[1]} markrisk_H={tail_stats[2]} proxy_avg={format_ratio_3(tail_stats[3],m)}")
      chunk_index+=1

  if gpu_log_level>=1:
    total:int=0
    subcell=0
    while subcell<75:
      total+=counts[subcell]
      subcell+=1
    print(f"[broadmarktail-reorder-subcell-summary] N={N} records={total} funcid17={counts[broad_markdist_tail_subcell_index(3,0,0)]+counts[broad_markdist_tail_subcell_index(3,1,0)]+counts[broad_markdist_tail_subcell_index(3,2,0)]+counts[broad_markdist_tail_subcell_index(3,3,0)]+counts[broad_markdist_tail_subcell_index(3,4,0)]} cell_G_H={counts[broad_markdist_tail_subcell_index(3,2,0)]+counts[broad_markdist_tail_subcell_index(3,2,1)]+counts[broad_markdist_tail_subcell_index(3,2,2)]}")

  return counts

def broad_markdist_tail_make_subcell_quotas(rem_counts:List[int],total_remaining:int,m_target:int)->List[int]:
  quotas:List[int]=[0]*75
  if total_remaining<=0 or m_target<=0:
    return quotas

  cell_rem:List[int]=[0]*25
  subcell:int=0
  while subcell<75:
    cell:int=subcell//3
    cell_rem[cell]+=rem_counts[subcell]
    subcell+=1

  cell_quotas:List[int]=broad_markdist_make_cell_quotas(cell_rem,total_remaining,m_target)

  cell=0
  while cell<25:
    cq:int=cell_quotas[cell]
    if cq>0 and cell_rem[cell]>0:
      sub_rem:List[int]=[0]*5
      sub_rem[0]=rem_counts[cell*3]
      sub_rem[1]=rem_counts[cell*3+1]
      sub_rem[2]=rem_counts[cell*3+2]
      sub_q:List[int]=funcid_reorder_make_quotas(sub_rem,cell_rem[cell],cq)
      quotas[cell*3]=sub_q[0]
      quotas[cell*3+1]=sub_q[1]
      quotas[cell*3+2]=sub_q[2]
    cell+=1

  qsum:int=0
  subcell=0
  while subcell<75:
    qsum+=quotas[subcell]
    subcell+=1
  while qsum<m_target:
    best:int=-1
    best_room:int=-1
    subcell=0
    while subcell<75:
      room:int=rem_counts[subcell]-quotas[subcell]
      if room>best_room:
        best_room=room
        best=subcell
      subcell+=1
    if best<0 or best_room<=0:
      break
    quotas[best]+=1
    qsum+=1

  return quotas

def interleave_broad_markdist_tail_subparts(part_f17:List[Dict[str,int]],part_gh:List[Dict[str,int]],part_r:List[Dict[str,int]],m_target:int,cell:int,chunk_index:int)->List[Dict[str,int]]:
  out:List[Dict[str,int]]=[]
  i17:int=0
  ig:int=0
  ir:int=0
  phase:int=0
  if broad_markdist_tail_use_rotating_interleave():
    phase=(chunk_index+cell)%4
  while len(out)<m_target:
    progressed:bool=False

    if phase==0:
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
      if i17<len(part_f17) and len(out)<m_target:
        out.append(part_f17[i17])
        i17+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
      if ig<len(part_gh) and len(out)<m_target:
        out.append(part_gh[ig])
        ig+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
    elif phase==1:
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
      if ig<len(part_gh) and len(out)<m_target:
        out.append(part_gh[ig])
        ig+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
      if i17<len(part_f17) and len(out)<m_target:
        out.append(part_f17[i17])
        i17+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
    elif phase==2:
      if i17<len(part_f17) and len(out)<m_target:
        out.append(part_f17[i17])
        i17+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
      if ig<len(part_gh) and len(out)<m_target:
        out.append(part_gh[ig])
        ig+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
    else:
      if ig<len(part_gh) and len(out)<m_target:
        out.append(part_gh[ig])
        ig+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
      if i17<len(part_f17) and len(out)<m_target:
        out.append(part_f17[i17])
        i17+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True

    if not progressed:
      break
  return out

def build_broad_markdist_tail_reordered_bin(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1
)->Tuple[str,int,int]:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  counts:List[int]=build_broad_markdist_tail_reorder_subcell_bins(N,fname,preset_queens,BLOCK,MAX_BLOCKS,gpu_log_level)
  counted_records:int=0
  subcell:int=0
  while subcell<75:
    counted_records+=counts[subcell]
    subcell+=1
  if counted_records!=total_records:
    print(f"[broadmarktail-reorder-warning] subcell count mismatch: counted={counted_records} total_records={total_records}")
    total_records=counted_records

  reorder_fname:str=broad_markdist_tail_reorder_output_fname(N,preset_queens,BLOCK,MAX_BLOCKS)
  truncate_plain_bin(reorder_fname)

  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_broadmarktail_reorder_{BROAD_MARKDIST_TAIL_REORDER_VERSION}_{broad_markdist_tail_variant_tag()}_{funcid_reorder_run_param_tag(BLOCK,MAX_BLOCKS)}_sim.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_broad_markdist_tail_reorder_progress_header())

  rem_counts:List[int]=[0]*75
  read_offsets:List[int]=[0]*75
  subcell_buffers:List[List[Dict[str,int]]]=make_broad_markdist_tail_subcell_buffers()
  subcell=0
  while subcell<75:
    rem_counts[subcell]=counts[subcell]
    read_offsets[subcell]=0
    subcell+=1

  off:int=0
  chunk_index:int=0
  total_remaining:int=total_records

  if gpu_log_level>=1:
    print(f"[broadmarktail-reorder-build-config] N={N} records={total_records} steps={STEPS} output={reorder_fname} progress={progress_fname} param={funcid_reorder_run_param_tag(BLOCK,MAX_BLOCKS)} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} weak_tail_window_boost={broad_markdist_tail_window_boost_value()} tail_variant={broad_markdist_tail_variant_tag()} reason={BROAD_MARKDIST_TAIL_REORDER_DEFAULT_REASON}")

  while total_remaining>0:
    m_target:int=STEPS
    if total_remaining<STEPS:
      m_target=total_remaining

    quotas:List[int]=broad_markdist_tail_make_subcell_quotas(rem_counts,total_remaining,m_target)
    cell_quotas:List[int]=[0]*25
    broad_quotas:List[int]=[0]*5
    subcell=0
    while subcell<75:
      cell:int=subcell//3
      broad:int=cell//5
      cell_quotas[cell]+=quotas[subcell]
      broad_quotas[broad]+=quotas[subcell]
      subcell+=1

    t0:datetime=datetime.now()

    parts:List[List[Dict[str,int]]]=make_broad_markdist_tail_subcell_buffers()
    subcell=0
    while subcell<75:
      q:int=quotas[subcell]
      if q>0:
        cell=subcell//3
        broad=cell//5
        risk:int=cell%5
        tail:int=subcell%3
        target:int=q*FUNCID_REORDER_V2_WINDOW_MULT
        if tail==0 or tail==1:
          target=target*broad_markdist_tail_window_boost_value()
        if target<q:
          target=q
        if target>rem_counts[subcell]:
          target=rem_counts[subcell]
        fname_subcell:str=broad_markdist_tail_reorder_subcell_fname(N,preset_queens,broad,risk,tail)
        newbuf:List[Dict[str,int]]=[]
        subcell_buffers[subcell],read_offsets[subcell]=fill_constellation_buffer_from_bin_range(fname_subcell,subcell_buffers[subcell],read_offsets[subcell],target)
        group_id:int=broad_markdist_tail_phase_group_id(subcell,broad,risk,tail)
        parts[subcell],newbuf=take_striped_records_from_buffer(subcell_buffers[subcell],q,chunk_index,group_id)
        subcell_buffers[subcell]=newbuf
        rem_counts[subcell]-=len(parts[subcell])
        if rem_counts[subcell]<0:
          rem_counts[subcell]=0
      subcell+=1

    cell_parts:List[List[Dict[str,int]]]=make_broad_markdist_cell_buffers()
    cell=0
    while cell<25:
      qcell:int=cell_quotas[cell]
      cell_parts[cell]=interleave_broad_markdist_tail_subparts(parts[cell*3],parts[cell*3+1],parts[cell*3+2],qcell,cell,chunk_index)
      cell+=1

    chunk_constellations:List[Dict[str,int]]=interleave_broad_markdist_secondary_parts(cell_parts,broad_quotas,m_target)
    m:int=len(chunk_constellations)
    if m==0:
      break

    append_constellations_bin(reorder_fname,chunk_constellations)
    stats:List[int]=analyze_stream_chunk_input_stats(N,chunk_constellations)
    risk_stats:List[int]=analyze_markdist_risk_stats(N,chunk_constellations)
    cell_stats:List[int]=analyze_broad_markdist_cell_stats(N,chunk_constellations)
    tail_stats:List[int]=analyze_broad_markdist_tail_summary(N,chunk_constellations)
    t1:datetime=datetime.now()
    elapsed_text:str=str(t1-t0)[:-3]
    elapsed_ms:int=stream_elapsed_text_to_ms(elapsed_text)
    append_stream_broad_markdist_tail_reorder_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,total_records,stats,risk_stats,cell_stats,tail_stats)

    if gpu_log_level>=2:
      print(f"[broadmarktail-reorder-build-chunk] chunk={chunk_index} off={off} m={m} B={stats[18+26]+stats[18+27]} A={stats[18+19]+stats[18+22]+stats[18+23]+stats[18+24]} C={stats[18+20]+stats[18+21]} G={stats[18+0]+stats[18+4]+stats[18+5]+stats[18+12]+stats[18+16]+stats[18+17]+stats[18+18]} O={m-(stats[18+26]+stats[18+27]+stats[18+19]+stats[18+22]+stats[18+23]+stats[18+24]+stats[18+20]+stats[18+21]+stats[18+0]+stats[18+4]+stats[18+5]+stats[18+12]+stats[18+16]+stats[18+17]+stats[18+18])} X={risk_stats[0]} T={risk_stats[1]} H={risk_stats[2]} M={risk_stats[3]} R={risk_stats[4]} funcid17={tail_stats[0]} cell_G_H={tail_stats[1]} tail_proxy_avg={format_ratio_3(tail_stats[3],m)}")

    off+=m
    chunk_index+=1
    total_remaining=total_records-off

  write_stream_done_count(reorder_fname+".done",off)
  reordered_records:int=count_constellations_bin_records(reorder_fname)
  if gpu_log_level>=1:
    print(f"[broadmarktail-reorder-build-summary] N={N} records={reordered_records} chunks={chunk_index} output={reorder_fname} progress={progress_fname} valid={1 if validate_bin_file(reorder_fname) else 0}")

  return reorder_fname,reordered_records,chunk_index

def exec_solutions_gpu_bin_stream_funcid_reorder(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,
  chunk_only:bool=False,
  debug_chunk_start:int=0,
  debug_chunk_count:int=1,
  chunk_list_spec:str="",
  progress_suffix:str="",
  worker_id:int=0,
  worker_count:int=1
)->int:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  if worker_count<=0:
    print(f"[worker-warning] invalid worker_count={worker_count}; using 1")
    worker_count=1
  if worker_id<0:
    print(f"[worker-warning] invalid worker_id={worker_id}; using 0")
    worker_id=0
  if worker_id>=worker_count:
    print(f"[worker-error] worker_id must be smaller than worker_count: worker_id={worker_id} worker_count={worker_count}")
    return 0

  total_records:int=count_constellations_bin_records(fname)
  run_param_tag:str=funcid_reorder_run_param_tag(BLOCK,MAX_BLOCKS)
  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{run_param_tag}.tsv"
  if progress_suffix!="":
    progress_fname=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{run_param_tag}_{progress_suffix}.tsv"
  if worker_count>1:
    if progress_suffix!="":
      progress_fname=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{run_param_tag}_{progress_suffix}_worker{worker_id}of{worker_count}.tsv"
    else:
      progress_fname=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{run_param_tag}_worker{worker_id}of{worker_count}.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_funcid_reorder_progress_header())

  selected_chunks:List[int]=parse_chunk_list_spec(chunk_list_spec)
  use_chunk_list:bool=(len(selected_chunks)>0)
  if chunk_only:
    if debug_chunk_start<0:
      debug_chunk_start=0
    if debug_chunk_count<=0:
      debug_chunk_count=1

  stop_after_chunk:int=-1
  if use_chunk_list:
    stop_after_chunk=chunk_list_max(selected_chunks)
  elif chunk_only:
    stop_after_chunk=debug_chunk_start+debug_chunk_count-1

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-gpu-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} chunk_only={1 if (chunk_only or use_chunk_list) else 0} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={chunk_list_to_string(selected_chunks)} worker={worker_id}/{worker_count} progress={progress_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} inner_log_level=0")

  gpu_total:int=0
  off:int=0
  chunk_index:int=0
  executed_chunks:int=0
  _read_uint32_le=read_uint32_le

  with open(fname,"rb") as f:
    while True:
      if stop_after_chunk>=0 and chunk_index>stop_after_chunk:
        break
      chunk_constellations:List[Dict[str,int]]=[]
      i:int=0
      while i<STEPS:
        raw:str=f.read(16)
        if len(raw)<16:
          break
        ld:int=_read_uint32_le(raw[0:4])
        rd:int=_read_uint32_le(raw[4:8])
        col:int=_read_uint32_le(raw[8:12])
        startijkl:int=_read_uint32_le(raw[12:16])
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1

      m:int=len(chunk_constellations)
      if m==0:
        break

      if chunk_only or use_chunk_list:
        run_this_chunk:bool=True
        if use_chunk_list:
          run_this_chunk=chunk_list_contains(selected_chunks,chunk_index)
        else:
          run_this_chunk=(chunk_index>=debug_chunk_start and chunk_index<debug_chunk_start+debug_chunk_count)
        if not run_this_chunk:
          if gpu_log_level>=2:
            print(f"[funcid-reorder-v2-gpu-chunk-skip] N={N} chunk={chunk_index} off={off} m={m}")
          off+=m
          chunk_index+=1
          continue

      if worker_count>1:
        run_worker_chunk:bool=((chunk_index % worker_count)==worker_id)
        if not run_worker_chunk:
          if gpu_log_level>=2:
            print(f"[funcid-reorder-v2-gpu-worker-skip] N={N} worker={worker_id}/{worker_count} chunk={chunk_index} off={off} m={m}")
          off+=m
          chunk_index+=1
          continue

      stats:List[int]=analyze_stream_chunk_input_stats(N,chunk_constellations)

      t0=datetime.now()
      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-gpu-chunk-start] N={N} worker={worker_id}/{worker_count} chunk={chunk_index} off={off} m={m}")

      inner_gpu_log_level:int=0
      exec_solutions(N,chunk_constellations,True,gpu_block,gpu_max_blocks,inner_gpu_log_level,gpu_sort_mode,cross_stripe_safe)

      chunk_total:int=0
      if m>0:
        chunk_total=chunk_constellations[0]["solutions"]
      gpu_total+=chunk_total
      executed_chunks+=1
      t1=datetime.now()
      elapsed_text:str=str(t1-t0)[:-3]
      elapsed_ms:int=stream_elapsed_text_to_ms(elapsed_text)
      append_stream_funcid_reorder_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,chunk_total,gpu_total,total_records,stats)
      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-gpu-chunk-end] N={N} worker={worker_id}/{worker_count} chunk={chunk_index} off={off} m={m} elapsed={elapsed_text} elapsed_ms={elapsed_ms} chunk_total={chunk_total} gpu_total={gpu_total}")

      off+=m
      chunk_index+=1

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-gpu-summary] N={N} records={total_records} chunks={chunk_index} executed_chunks={executed_chunks} total={gpu_total} progress={progress_fname}")
    if worker_count>1:
      print(f"[worker-summary] N={N} worker={worker_id}/{worker_count} records={total_records} chunks={chunk_index} executed_chunks={executed_chunks} partial_total={gpu_total} progress={progress_fname}")

  return gpu_total

def exec_solutions_gpu_chunk_profile(
  N:int,
  chunk_constellations:List[Dict[str,int]],
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT
)->Tuple[int,List[int],List[int],str,int]:
  board_mask:int=(1<<N)-1
  if gpu_sort_mode<0:
    gpu_sort_mode=auto_sort_mode(N)

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  m:int=len(chunk_constellations)
  soa:TaskSoA=TaskSoA(STEPS)
  sort_soa:TaskSoA=TaskSoA(STEPS)
  w_arr:List[u64]=[u64(0)]*STEPS
  sort_w_arr:List[u64]=[u64(0)]*STEPS
  results:List[u64]=[u64(0)]*STEPS
  order:List[int]=[0]*STEPS

  funcptn_by_fid:List[int]=[0,1,3,5,2,0,1,2,4,4,4,4,0,1,5,2,0,1,2,0,1,5,2,0,2,1,5,0]
  meta_next:List[u8]=[u8(1),u8(2),u8(3),u8(3),u8(2),u8(6),u8(2),u8(2),u8(0),u8(4),u8(5),u8(7),u8(13),u8(14),u8(14),u8(14),u8(17),u8(14),u8(14),u8(20),u8(21),u8(21),u8(21),u8(25),u8(21),u8(21),u8(26),u8(26)]

  local_sort_mode:int=gpu_sort_mode
  if gpu_sort_mode==5 or gpu_sort_mode==8 or gpu_sort_mode==10:
    local_sort_mode=4
  if gpu_sort_mode==6 or gpu_sort_mode==7 or gpu_sort_mode==9:
    local_sort_mode=0

  t_soa0=datetime.now()
  build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)
  t_soa1=datetime.now()

  t_stats0=datetime.now()
  stats:List[int]=analyze_stream_chunk_input_stats_from_soa(soa,w_arr,m)
  t_stats1=datetime.now()

  t_sort0=datetime.now()
  use_sorted:bool=(local_sort_mode==1 or local_sort_mode==2 or local_sort_mode==3 or local_sort_mode==4)
  if use_sorted:
    nb:int=28
    if local_sort_mode==2:
      nb=6
    if local_sort_mode==3:
      nb=24
    if local_sort_mode==4:
      nb=48
    counts:List[int]=[0]*64
    pos:List[int]=[0]*64
    cur:List[int]=[0]*64
    i:int=0
    while i<m:
      fid0:int=soa.funcid_arr[i]
      key:int=fid0
      if local_sort_mode==2:
        key=funcptn_by_fid[fid0]
      if local_sort_mode==3:
        ptn:int=funcptn_by_fid[fid0]
        depth:int=soa.end_arr[i]-soa.row_arr[i]
        if depth<0:
          depth=0
        pc:int=popcount_int(int(soa.free_arr[i]))
        wb:int=0
        if pc>=3:
          wb+=1
        if depth>=12:
          wb+=2
        key=ptn*4+wb
      if local_sort_mode==4:
        ptn:int=funcptn_by_fid[fid0]
        depth:int=soa.end_arr[i]-soa.row_arr[i]
        if depth<0:
          depth=0
        pc:int=popcount_int(int(soa.free_arr[i]))
        wb:int=0
        if pc>=2:
          wb+=1
        if pc>=4:
          wb+=2
        if depth>=10:
          wb+=4
        key=ptn*8+wb
      counts[key]+=1
      i+=1
    run:int=0
    b:int=0
    while b<nb:
      pos[b]=run
      cur[b]=run
      run+=counts[b]
      b+=1
    i=0
    while i<m:
      fid0:int=soa.funcid_arr[i]
      key:int=fid0
      if local_sort_mode==2:
        key=funcptn_by_fid[fid0]
      if local_sort_mode==3:
        ptn:int=funcptn_by_fid[fid0]
        depth:int=soa.end_arr[i]-soa.row_arr[i]
        if depth<0:
          depth=0
        pc:int=popcount_int(int(soa.free_arr[i]))
        wb:int=0
        if pc>=3:
          wb+=1
        if depth>=12:
          wb+=2
        key=ptn*4+wb
      if local_sort_mode==4:
        ptn:int=funcptn_by_fid[fid0]
        depth:int=soa.end_arr[i]-soa.row_arr[i]
        if depth<0:
          depth=0
        pc:int=popcount_int(int(soa.free_arr[i]))
        wb:int=0
        if pc>=2:
          wb+=1
        if pc>=4:
          wb+=2
        if depth>=10:
          wb+=4
        key=ptn*8+wb
      p:int=cur[key]
      order[p]=i
      cur[key]+=1
      i+=1
    p:int=0
    while p<m:
      q:int=order[p]
      sort_soa.ld_arr[p]=soa.ld_arr[q]
      sort_soa.rd_arr[p]=soa.rd_arr[q]
      sort_soa.col_arr[p]=soa.col_arr[q]
      sort_soa.row_arr[p]=soa.row_arr[q]
      sort_soa.ctrl0_arr[p]=soa.ctrl0_arr[q]
      sort_soa.free_arr[p]=soa.free_arr[q]
      sort_soa.markctrl_arr[p]=soa.markctrl_arr[q]
      sort_soa.funcid_arr[p]=soa.funcid_arr[q]
      sort_soa.ijkl_arr[p]=soa.ijkl_arr[q]
      sort_w_arr[p]=w_arr[q]
      p+=1
  t_sort1=datetime.now()

  GRID:int=(m+BLOCK-1)//BLOCK
  board_mask_gpu:u32=u32(board_mask)
  n3_gpu:u32=u32(1)<<u32(N-3)
  n4_gpu:u32=u32(1)<<u32(N-4)

  t_kernel0=datetime.now()

  if use_sorted:
    required_maxd:int=max_schedule_depth_of_tasks(sort_soa,m,meta_next)
    selected_maxd:int=select_static_maxd(required_maxd)
    if not launch_kernel_dfs_iter_gpu_static_maxd(selected_maxd,sort_soa,sort_w_arr,meta_next,results,m,board_mask_gpu,n3_gpu,n4_gpu,GRID,BLOCK):
      print(f"[maxd-error] unsupported required_maxd={required_maxd}; supported maximum is 21")
      error_stages:List[int]=[0]*7
      return 0,stats,error_stages,"0:00:00.000",0
  else:
    required_maxd:int=max_schedule_depth_of_tasks(soa,m,meta_next)
    selected_maxd:int=select_static_maxd(required_maxd)
    if not launch_kernel_dfs_iter_gpu_static_maxd(selected_maxd,soa,w_arr,meta_next,results,m,board_mask_gpu,n3_gpu,n4_gpu,GRID,BLOCK):
      print(f"[maxd-error] unsupported required_maxd={required_maxd}; supported maximum is 21")
      error_stages:List[int]=[0]*7
      return 0,stats,error_stages,"0:00:00.000",0

  t_kernel1=datetime.now()

  t_reduce0=datetime.now()
  chunk_total:int=0
  i:int=0
  while i<m:
    chunk_total+=int(results[i])
    i+=1
  t_reduce1=datetime.now()

  stage_soa_ms:int=profile_elapsed_ms_between(t_soa0,t_soa1)
  stage_stats_ms:int=profile_elapsed_ms_between(t_stats0,t_stats1)
  stage_sort_ms:int=profile_elapsed_ms_between(t_sort0,t_sort1)
  stage_kernel_ms:int=profile_elapsed_ms_between(t_kernel0,t_kernel1)
  stage_reduce_ms:int=profile_elapsed_ms_between(t_reduce0,t_reduce1)
  stage_compute_ms:int=stage_soa_ms+stage_sort_ms+stage_kernel_ms+stage_reduce_ms
  stage_no_read_ms:int=stage_compute_ms+stage_stats_ms
  elapsed_text:str=str(t_reduce1-t_soa0)[:-3]
  elapsed_ms:int=stage_no_read_ms
  stages_inner:List[int]=[stage_soa_ms,stage_stats_ms,stage_sort_ms,stage_kernel_ms,stage_reduce_ms,stage_compute_ms,stage_no_read_ms]

  return chunk_total,stats,stages_inner,elapsed_text,elapsed_ms

def exec_solutions_gpu_chunk_split145(
  N:int,
  chunk_constellations:List[Dict[str,int]],
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,
  split_mode:int=0,
  gpu_log_level:int=0
)->Tuple[int,List[int],List[int],str,int]:
  board_mask:int=(1<<N)-1

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  m:int=len(chunk_constellations)
  soa:TaskSoA=TaskSoA(STEPS)
  w_arr:List[u64]=[u64(0)]*STEPS
  results:List[u64]=[u64(0)]*STEPS

  t0=datetime.now()
  build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)
  t1=datetime.now()
  stats:List[int]=analyze_stream_chunk_input_stats_from_soa(soa,w_arr,m)
  t2=datetime.now()

  d2base14_m:int=0
  d0_m:int=0
  i:int=0
  while i<m:
    fid:int=soa.funcid_arr[i]
    if fid==14:
      d2base14_m+=1
    if fid==26 or fid==27:
      d0_m+=1
    i+=1

  t3=datetime.now()

  board_mask_gpu:u32=u32(board_mask)
  n3_gpu:u32=u32(1)<<u32(N-3)
  n4_gpu:u32=u32(1)<<u32(N-4)
  chunk_total:int=0

  if gpu_log_level>=1:
    print(f"[split145-buckets] N={N} m={m} generic={m} d2base14={d2base14_m} d0={d0_m} rest=0 split_mode={split_mode} specialized=0")

  meta_next:List[u8]=[u8(1),u8(2),u8(3),u8(3),u8(2),u8(6),u8(2),u8(2),u8(0),u8(4),u8(5),u8(7),u8(13),u8(14),u8(14),u8(14),u8(17),u8(14),u8(14),u8(20),u8(21),u8(21),u8(21),u8(25),u8(21),u8(21),u8(26),u8(26)]

  if m>0:
    GRID:int=(m+BLOCK-1)//BLOCK
    required_maxd:int=max_schedule_depth_of_tasks(soa,m,meta_next)
    selected_maxd:int=select_static_maxd(required_maxd)
    if gpu_log_level>=1:
      print(f"[maxd-dispatch] N={N} scope=split145 m={m} required_maxd={required_maxd} selected_MAXD={selected_maxd} schedule_words={packed_schedule_words_for_maxd(selected_maxd)} stack_bytes_per_thread={packed_stack_bytes_per_thread(selected_maxd)} capacity_check=OK")
    if not launch_kernel_dfs_iter_gpu_static_maxd(selected_maxd,soa,w_arr,meta_next,results,m,board_mask_gpu,n3_gpu,n4_gpu,GRID,BLOCK):
      print(f"[maxd-error] unsupported required_maxd={required_maxd}; supported maximum is 21")
      error_stages:List[int]=[0]*10
      return 0,stats,error_stages,"0:00:00.000",0
    i=0
    while i<m:
      chunk_total+=int(results[i])
      i+=1

  t4=datetime.now()
  stage_soa_ms:int=profile_elapsed_ms_between(t0,t1)
  stage_stats_ms:int=profile_elapsed_ms_between(t1,t2)
  stage_split_ms:int=profile_elapsed_ms_between(t2,t3)
  stage_kernel_reduce_ms:int=profile_elapsed_ms_between(t3,t4)
  stage_compute_ms:int=stage_soa_ms+stage_split_ms+stage_kernel_reduce_ms
  stage_no_read_ms:int=stage_compute_ms+stage_stats_ms
  elapsed_text:str=str(t4-t0)[:-3]
  elapsed_ms:int=stage_no_read_ms
  stages:List[int]=[stage_soa_ms,stage_stats_ms,stage_split_ms,stage_kernel_reduce_ms,stage_compute_ms,stage_no_read_ms,0,d0_m,d2base14_m,m]
  return chunk_total,stats,stages,elapsed_text,elapsed_ms

def exec_solutions_gpu_bin_stream_split145(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,
  chunk_only:bool=False,
  debug_chunk_start:int=0,
  debug_chunk_count:int=1,
  chunk_list_spec:str="",
  progress_suffix:str="split145",
  worker_id:int=0,
  worker_count:int=1,
  split_mode:int=2
)->int:
  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488
  if worker_count<=0:
    print(f"[worker-warning] invalid worker_count={worker_count}; using 1")
    worker_count=1
  if worker_id<0:
    print(f"[worker-warning] invalid worker_id={worker_id}; using 0")
    worker_id=0
  if worker_id>=worker_count:
    print(f"[worker-error] worker_id must be smaller than worker_count: worker_id={worker_id} worker_count={worker_count}")
    return 0

  total_records:int=count_constellations_bin_records(fname)
  run_param_tag:str=funcid_reorder_run_param_tag(BLOCK,MAX_BLOCKS)
  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_split145_{run_param_tag}.tsv"
  if progress_suffix!="":
    progress_fname=f"progress_N{N}_{preset_queens}_stream_split145_{run_param_tag}_{progress_suffix}.tsv"
  if worker_count>1:
    if progress_suffix!="":
      progress_fname=f"progress_N{N}_{preset_queens}_stream_split145_{run_param_tag}_{progress_suffix}_worker{worker_id}of{worker_count}.tsv"
    else:
      progress_fname=f"progress_N{N}_{preset_queens}_stream_split145_{run_param_tag}_worker{worker_id}of{worker_count}.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_funcid_reorder_progress_header())

  selected_chunks:List[int]=parse_chunk_list_spec(chunk_list_spec)
  use_chunk_list:bool=(len(selected_chunks)>0)
  if chunk_only:
    if debug_chunk_start<0:
      debug_chunk_start=0
    if debug_chunk_count<=0:
      debug_chunk_count=1
  stop_after_chunk:int=-1
  if use_chunk_list:
    stop_after_chunk=chunk_list_max(selected_chunks)
  elif chunk_only:
    stop_after_chunk=debug_chunk_start+debug_chunk_count-1

  if gpu_log_level>=1:
    print(f"[split145-gpu-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} chunk_only={1 if (chunk_only or use_chunk_list) else 0} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={chunk_list_to_string(selected_chunks)} worker={worker_id}/{worker_count} split_mode={split_mode} progress={progress_fname}")

  gpu_total:int=0
  off:int=0
  chunk_index:int=0
  executed_chunks:int=0
  _read_uint32_le=read_uint32_le
  with open(fname,"rb") as f:
    while True:
      if stop_after_chunk>=0 and chunk_index>stop_after_chunk:
        break
      chunk_constellations:List[Dict[str,int]]=[]
      i:int=0
      while i<STEPS:
        raw:str=f.read(16)
        if len(raw)<16:
          break
        ld:int=_read_uint32_le(raw[0:4])
        rd:int=_read_uint32_le(raw[4:8])
        col:int=_read_uint32_le(raw[8:12])
        startijkl:int=_read_uint32_le(raw[12:16])
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1
      m:int=len(chunk_constellations)
      if m==0:
        break
      if chunk_only or use_chunk_list:
        run_this_chunk:bool=True
        if use_chunk_list:
          run_this_chunk=chunk_list_contains(selected_chunks,chunk_index)
        else:
          run_this_chunk=(chunk_index>=debug_chunk_start and chunk_index<debug_chunk_start+debug_chunk_count)
        if not run_this_chunk:
          if gpu_log_level>=2:
            print(f"[split145-gpu-chunk-skip] N={N} chunk={chunk_index} off={off} m={m}")
          off+=m; chunk_index+=1; continue
      if worker_count>1:
        run_worker_chunk:bool=((chunk_index % worker_count)==worker_id)
        if not run_worker_chunk:
          if gpu_log_level>=2:
            print(f"[split145-gpu-worker-skip] N={N} worker={worker_id}/{worker_count} chunk={chunk_index} off={off} m={m}")
          off+=m; chunk_index+=1; continue

      t0=datetime.now()
      if gpu_log_level>=1:
        print(f"[split145-gpu-chunk-start] N={N} worker={worker_id}/{worker_count} chunk={chunk_index} off={off} m={m}")
      chunk_total:int=0
      stats:List[int]=[0]*46
      stages_inner:List[int]=[0,0,0,0,0,0,0]
      elapsed_text:str="0:00:00.000"
      elapsed_ms:int=0
      chunk_total,stats,stages_inner,elapsed_text,elapsed_ms=exec_solutions_gpu_chunk_split145(N,chunk_constellations,BLOCK,MAX_BLOCKS,gpu_sort_mode,cross_stripe_safe,split_mode,gpu_log_level)
      gpu_total+=chunk_total
      executed_chunks+=1
      t1=datetime.now()
      elapsed_outer_text:str=str(t1-t0)[:-3]
      elapsed_outer_ms:int=stream_elapsed_text_to_ms(elapsed_outer_text)
      append_stream_funcid_reorder_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_outer_text,elapsed_outer_ms,chunk_total,gpu_total,total_records,stats)
      if gpu_log_level>=1:
        print(f"[split145-gpu-chunk-end] N={N} worker={worker_id}/{worker_count} chunk={chunk_index} off={off} m={m} elapsed={elapsed_outer_text} inner={elapsed_text} elapsed_ms={elapsed_outer_ms} chunk_total={chunk_total} gpu_total={gpu_total} soa_ms={stages_inner[0]} stats_ms={stages_inner[1]} split_ms={stages_inner[2]} kernel_reduce_ms={stages_inner[3]}")
      off+=m
      chunk_index+=1
  if gpu_log_level>=1:
    print(f"[split145-gpu-summary] N={N} records={total_records} chunks={chunk_index} executed_chunks={executed_chunks} total={gpu_total} split_mode={split_mode} progress={progress_fname}")
    if worker_count>1:
      print(f"[worker-summary] N={N} worker={worker_id}/{worker_count} records={total_records} chunks={chunk_index} executed_chunks={executed_chunks} partial_total={gpu_total} progress={progress_fname}")
  return gpu_total

def exec_solutions_gpu_bin_stream_funcid_reorder_profile(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,
  chunk_only:bool=True,
  debug_chunk_start:int=0,
  debug_chunk_count:int=1,
  chunk_list_spec:str="",
  progress_suffix:str="profile"
)->int:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{funcid_reorder_param_tag()}_{progress_suffix}.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_funcid_reorder_profile_progress_header())

  selected_chunks:List[int]=parse_chunk_list_spec(chunk_list_spec)
  use_chunk_list:bool=(len(selected_chunks)>0)
  if chunk_only:
    if debug_chunk_start<0:
      debug_chunk_start=0
    if debug_chunk_count<=0:
      debug_chunk_count=1

  stop_after_chunk:int=-1
  if use_chunk_list:
    stop_after_chunk=chunk_list_max(selected_chunks)
  elif chunk_only:
    stop_after_chunk=debug_chunk_start+debug_chunk_count-1

  if gpu_log_level>=1:
    chunk_mode:str="range"
    if use_chunk_list:
      chunk_mode="list"
    print(f"[funcid-reorder-v2-profile-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} chunk_mode={chunk_mode} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list_count={len(selected_chunks)} chunk_list={chunk_list_to_string(selected_chunks)} progress={progress_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP}")

  gpu_total:int=0
  off:int=0
  chunk_index:int=0
  executed_chunks:int=0
  _read_uint32_le=read_uint32_le

  with open(fname,"rb") as f:
    while True:
      if stop_after_chunk>=0 and chunk_index>stop_after_chunk:
        break
      t_read0=datetime.now()
      chunk_constellations:List[Dict[str,int]]=[]
      i:int=0
      while i<STEPS:
        raw:str=f.read(16)
        if len(raw)<16:
          break
        ld:int=_read_uint32_le(raw[0:4])
        rd:int=_read_uint32_le(raw[4:8])
        col:int=_read_uint32_le(raw[8:12])
        startijkl:int=_read_uint32_le(raw[12:16])
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1
      t_read1=datetime.now()
      stage_read_ms:int=profile_elapsed_ms_between(t_read0,t_read1)

      m:int=len(chunk_constellations)
      if m==0:
        break

      if chunk_only or use_chunk_list:
        run_this_chunk:bool=True
        if use_chunk_list:
          run_this_chunk=chunk_list_contains(selected_chunks,chunk_index)
        else:
          run_this_chunk=(chunk_index>=debug_chunk_start and chunk_index<debug_chunk_start+debug_chunk_count)
        if not run_this_chunk:
          if gpu_log_level>=2:
            print(f"[funcid-reorder-v2-profile-chunk-skip] N={N} chunk={chunk_index} off={off} m={m}")
          off+=m
          chunk_index+=1
          continue

      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-profile-chunk-start] N={N} chunk={chunk_index} off={off} m={m} read_ms={stage_read_ms}")

      chunk_total,stats,stages_inner,elapsed_text,elapsed_ms=exec_solutions_gpu_chunk_profile(N,chunk_constellations,gpu_block,gpu_max_blocks,gpu_sort_mode,cross_stripe_safe)

      gpu_total+=chunk_total
      executed_chunks+=1

      stage_soa_ms:int=stages_inner[0]
      stage_stats_ms:int=stages_inner[1]
      stage_sort_ms:int=stages_inner[2]
      stage_kernel_ms:int=stages_inner[3]
      stage_reduce_ms:int=stages_inner[4]
      stage_compute_ms:int=stages_inner[5]
      stage_no_read_ms:int=stages_inner[6]
      stage_total_ms:int=stage_read_ms+stage_no_read_ms
      stages:List[int]=[stage_read_ms,stage_soa_ms,stage_stats_ms,stage_sort_ms,stage_kernel_ms,stage_reduce_ms,stage_compute_ms,stage_no_read_ms,stage_total_ms]

      append_stream_funcid_reorder_profile_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,chunk_total,gpu_total,total_records,stats,stages)

      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-profile-chunk-end] N={N} chunk={chunk_index} off={off} m={m} elapsed={elapsed_text} elapsed_ms={elapsed_ms} chunk_total={chunk_total} gpu_total={gpu_total} read_ms={stage_read_ms} soa_ms={stage_soa_ms} stats_ms={stage_stats_ms} sort_ms={stage_sort_ms} kernel_ms={stage_kernel_ms} reduce_ms={stage_reduce_ms} compute_ms={stage_compute_ms} total_ms={stage_total_ms}")

      off+=m
      chunk_index+=1

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-profile-summary] N={N} records={total_records} chunks={chunk_index} executed_chunks={executed_chunks} total={gpu_total} progress={progress_fname}")

  return gpu_total

def read_constellations_bin_range(fname:str,off_record:int,max_records:int)->List[Dict[str,int]]:
  out:List[Dict[str,int]]=[]
  if off_record<0:
    off_record=0
  if max_records<=0:
    return out
  _read_uint32_le=read_uint32_le
  with open(fname,"rb") as f:
    f.seek(off_record*16,0)
    i:int=0
    while i<max_records:
      raw:str=f.read(16)
      if len(raw)<16:
        break
      ld:int=_read_uint32_le(raw[0:4])
      rd:int=_read_uint32_le(raw[4:8])
      col:int=_read_uint32_le(raw[8:12])
      startijkl:int=_read_uint32_le(raw[12:16])
      out.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
      i+=1
  return out

def build_chunk_range_list(start:int,count:int)->List[int]:
  out:List[int]=[]
  if start<0:
    start=0
  if count<=0:
    count=1
  i:int=0
  while i<count:
    out.append(start+i)
    i+=1
  return out

def exec_solutions_gpu_bin_stream_funcid_reorder_chunksize_profile(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,
  debug_chunk_start:int=0,
  debug_chunk_count:int=1,
  chunk_list_spec:str="",
  factor_list_spec:str=CHUNKSIZE_DEFAULT_FACTOR_LIST,
  progress_suffix:str="chunksize"
)->int:

  BLOCK:int=gpu_block
  BASE_MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if BASE_MAX_BLOCKS<=0:
    BASE_MAX_BLOCKS=484
  BASE_STEPS:int=BLOCK*BASE_MAX_BLOCKS
  if BASE_STEPS<=0:
    BASE_STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  selected_chunks:List[int]=parse_chunk_list_spec(chunk_list_spec)
  if len(selected_chunks)==0:
    selected_chunks=build_chunk_range_list(debug_chunk_start,debug_chunk_count)
  factors:List[int]=parse_positive_int_list_spec(factor_list_spec)
  if len(factors)==0:
    factors=parse_positive_int_list_spec(CHUNKSIZE_DEFAULT_FACTOR_LIST)

  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{funcid_reorder_param_tag()}_{progress_suffix}.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_funcid_reorder_chunksize_progress_header())

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-chunksize-config] N={N} records={total_records} bin={fname} block={BLOCK} base_max_blocks={BASE_MAX_BLOCKS} base_steps={BASE_STEPS} sort_mode={gpu_sort_mode} chunk_list_count={len(selected_chunks)} chunk_list={chunk_list_to_string(selected_chunks)} factor_list={factor_list_spec} progress={progress_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP}")

  gpu_total:int=0
  executed_cases:int=0
  fi:int=0
  while fi<len(factors):
    factor:int=factors[fi]
    if factor<=0:
      fi+=1
      continue
    FACTOR_MAX_BLOCKS:int=BASE_MAX_BLOCKS*factor
    FACTOR_STEPS:int=BLOCK*FACTOR_MAX_BLOCKS
    ci:int=0
    while ci<len(selected_chunks):
      base_chunk:int=selected_chunks[ci]
      if base_chunk<0:
        ci+=1
        continue
      off:int=base_chunk*BASE_STEPS
      if off>=total_records:
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-chunksize-skip] N={N} factor={factor} base_chunk={base_chunk} off={off} reason=off_ge_records")
        ci+=1
        continue
      target_m:int=FACTOR_STEPS
      remaining:int=total_records-off
      if target_m>remaining:
        target_m=remaining
      t_read0=datetime.now()
      chunk_constellations:List[Dict[str,int]]=read_constellations_bin_range(fname,off,target_m)
      t_read1=datetime.now()
      stage_read_ms:int=profile_elapsed_ms_between(t_read0,t_read1)
      m:int=len(chunk_constellations)
      if m==0:
        ci+=1
        continue

      range_chunks:int=(m + BASE_STEPS - 1)//BASE_STEPS
      range_start_chunk:int=base_chunk
      range_end_chunk:int=base_chunk+range_chunks-1

      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-chunksize-case-start] N={N} factor={factor} base_chunk={base_chunk} off={off} m={m} block={BLOCK} max_blocks={FACTOR_MAX_BLOCKS} steps={FACTOR_STEPS} read_ms={stage_read_ms} range_chunks={range_chunks}")

      chunk_total,stats,stages_inner,elapsed_text,elapsed_ms=exec_solutions_gpu_chunk_profile(N,chunk_constellations,BLOCK,FACTOR_MAX_BLOCKS,gpu_sort_mode,cross_stripe_safe)
      gpu_total+=chunk_total
      executed_cases+=1

      stage_soa_ms:int=stages_inner[0]
      stage_stats_ms:int=stages_inner[1]
      stage_sort_ms:int=stages_inner[2]
      stage_kernel_ms:int=stages_inner[3]
      stage_reduce_ms:int=stages_inner[4]
      stage_compute_ms:int=stages_inner[5]
      stage_no_read_ms:int=stages_inner[6]
      stage_total_ms:int=stage_read_ms+stage_no_read_ms
      stages:List[int]=[stage_read_ms,stage_soa_ms,stage_stats_ms,stage_sort_ms,stage_kernel_ms,stage_reduce_ms,stage_compute_ms,stage_no_read_ms,stage_total_ms]

      append_stream_funcid_reorder_chunksize_progress(progress_fname,N,preset_queens,base_chunk,off,m,BLOCK,FACTOR_MAX_BLOCKS,FACTOR_STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,chunk_total,gpu_total,total_records,stats,stages,BASE_STEPS,base_chunk,factor,range_start_chunk,range_end_chunk,range_chunks)

      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-chunksize-case-end] N={N} factor={factor} base_chunk={base_chunk} range={range_start_chunk}-{range_end_chunk} off={off} m={m} elapsed={elapsed_text} elapsed_ms={elapsed_ms} chunk_total={chunk_total} gpu_total={gpu_total} read_ms={stage_read_ms} soa_ms={stage_soa_ms} stats_ms={stage_stats_ms} sort_ms={stage_sort_ms} kernel_ms={stage_kernel_ms} reduce_ms={stage_reduce_ms} compute_ms={stage_compute_ms} total_ms={stage_total_ms}")

      ci+=1
    fi+=1

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-chunksize-summary] N={N} records={total_records} base_steps={BASE_STEPS} cases={executed_cases} total={gpu_total} progress={progress_fname}")

  return gpu_total

def exec_solutions_gpu_bin_stream_funcid_reorder_funcid_target_profile(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,
  debug_chunk_start:int=0,
  debug_chunk_count:int=1,
  chunk_list_spec:str=FUNCID_TARGET_DEFAULT_CHUNK_LIST,
  group_list_spec:str=FUNCID_TARGET_DEFAULT_GROUP_LIST,
  progress_suffix:str="funcidtarget"
)->int:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  selected_chunks:List[int]=parse_chunk_list_spec(chunk_list_spec)
  if len(selected_chunks)==0:
    selected_chunks=build_chunk_range_list(debug_chunk_start,debug_chunk_count)
  groups:List[str]=parse_funcid_target_group_list_spec(group_list_spec)
  if len(groups)==0:
    groups=parse_funcid_target_group_list_spec(FUNCID_TARGET_DEFAULT_GROUP_LIST)

  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{funcid_reorder_param_tag()}_{progress_suffix}.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_funcid_reorder_funcid_target_progress_header())

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-funcidtarget-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} chunk_list_count={len(selected_chunks)} chunk_list={chunk_list_to_string(selected_chunks)} group_list={group_list_spec} progress={progress_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP}")

  gpu_total:int=0
  executed_cases:int=0
  ci:int=0
  while ci<len(selected_chunks):
    base_chunk:int=selected_chunks[ci]
    if base_chunk<0:
      ci+=1
      continue
    off:int=base_chunk*STEPS
    if off>=total_records:
      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-funcidtarget-skip] N={N} base_chunk={base_chunk} off={off} reason=off_ge_records")
      ci+=1
      continue
    target_m_read:int=STEPS
    remaining:int=total_records-off
    if target_m_read>remaining:
      target_m_read=remaining

    t_read0=datetime.now()
    source_constellations:List[Dict[str,int]]=read_constellations_bin_range(fname,off,target_m_read)
    t_read1=datetime.now()
    stage_read_ms:int=profile_elapsed_ms_between(t_read0,t_read1)
    source_m:int=len(source_constellations)
    if source_m==0:
      ci+=1
      continue

    gi:int=0
    while gi<len(groups):
      group:str=groups[gi]
      t_filter0=datetime.now()
      target_constellations:List[Dict[str,int]]=filter_constellations_by_funcid_target_group(N,source_constellations,group)
      t_filter1=datetime.now()
      stage_filter_ms:int=profile_elapsed_ms_between(t_filter0,t_filter1)
      target_m:int=len(target_constellations)

      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-funcidtarget-case-start] N={N} base_chunk={base_chunk} group={group} off={off} source_m={source_m} target_m={target_m} read_ms={stage_read_ms} filter_ms={stage_filter_ms}")

      chunk_total:int=0
      stats:List[int]=[0]*46
      stages_inner:List[int]=[0,0,0,0,0,0,0]
      elapsed_text:str="0:00:00.000"
      elapsed_ms:int=0

      if target_m>0:
        chunk_total,stats,stages_inner,elapsed_text,elapsed_ms=exec_solutions_gpu_chunk_profile(N,target_constellations,BLOCK,MAX_BLOCKS,gpu_sort_mode,cross_stripe_safe)

      gpu_total+=chunk_total
      executed_cases+=1

      stage_soa_ms:int=stages_inner[0]
      stage_stats_ms:int=stages_inner[1]
      stage_sort_ms:int=stages_inner[2]
      stage_kernel_ms:int=stages_inner[3]
      stage_reduce_ms:int=stages_inner[4]
      stage_compute_ms:int=stages_inner[5]
      stage_no_read_ms:int=stages_inner[6]
      stage_total_ms:int=stage_read_ms+stage_filter_ms+stage_no_read_ms
      stages:List[int]=[stage_read_ms,stage_soa_ms,stage_stats_ms,stage_sort_ms,stage_kernel_ms,stage_reduce_ms,stage_compute_ms,stage_no_read_ms,stage_total_ms]

      append_stream_funcid_reorder_funcid_target_progress(progress_fname,N,preset_queens,base_chunk,off,target_m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,chunk_total,gpu_total,total_records,stats,stages,STEPS,base_chunk,group,source_m,stage_filter_ms)

      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-funcidtarget-case-end] N={N} base_chunk={base_chunk} group={group} off={off} source_m={source_m} target_m={target_m} elapsed={elapsed_text} elapsed_ms={elapsed_ms} chunk_total={chunk_total} gpu_total={gpu_total} read_ms={stage_read_ms} filter_ms={stage_filter_ms} soa_ms={stage_soa_ms} stats_ms={stage_stats_ms} sort_ms={stage_sort_ms} kernel_ms={stage_kernel_ms} reduce_ms={stage_reduce_ms} compute_ms={stage_compute_ms} total_ms={stage_total_ms}")

      gi+=1
    ci+=1

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-funcidtarget-summary] N={N} records={total_records} steps={STEPS} cases={executed_cases} total={gpu_total} progress={progress_fname}")

  return gpu_total

def exec_solutions_gpu_bin_stream_funcid_reorder_funcid_single_profile(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,
  debug_chunk_start:int=0,
  debug_chunk_count:int=1,
  chunk_list_spec:str=FUNCID_SINGLE_DEFAULT_CHUNK_LIST,
  funcid_list_spec:str=FUNCID_SINGLE_DEFAULT_FUNCID_LIST,
  progress_suffix:str="funcidsingle"
)->int:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  selected_chunks:List[int]=parse_chunk_list_spec(chunk_list_spec)
  if len(selected_chunks)==0:
    selected_chunks=build_chunk_range_list(debug_chunk_start,debug_chunk_count)
  fids:List[int]=parse_funcid_single_list_spec(funcid_list_spec)
  if len(fids)==0:
    fids=default_funcid_single_list()

  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{funcid_reorder_param_tag()}_{progress_suffix}.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_funcid_reorder_funcid_single_progress_header())

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-funcidsingle-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} chunk_list_count={len(selected_chunks)} chunk_list={chunk_list_to_string(selected_chunks)} funcid_list={funcid_list_to_string(fids)} progress={progress_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP}")

  gpu_total:int=0
  executed_cases:int=0
  ci:int=0
  while ci<len(selected_chunks):
    base_chunk:int=selected_chunks[ci]
    if base_chunk<0:
      ci+=1
      continue
    off:int=base_chunk*STEPS
    if off>=total_records:
      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-funcidsingle-skip] N={N} base_chunk={base_chunk} off={off} reason=off_ge_records")
      ci+=1
      continue
    target_m_read:int=STEPS
    remaining:int=total_records-off
    if target_m_read>remaining:
      target_m_read=remaining

    t_read0=datetime.now()
    source_constellations:List[Dict[str,int]]=read_constellations_bin_range(fname,off,target_m_read)
    t_read1=datetime.now()
    stage_read_ms:int=profile_elapsed_ms_between(t_read0,t_read1)
    source_m:int=len(source_constellations)
    if source_m==0:
      ci+=1
      continue

    t_classify0=datetime.now()
    source_soa:TaskSoA=TaskSoA(source_m)
    source_w_arr:List[u64]=[u64(0)]*source_m
    build_soa_for_range(N,source_constellations,0,source_m,source_soa,source_w_arr)
    t_classify1=datetime.now()
    stage_classify_ms:int=profile_elapsed_ms_between(t_classify0,t_classify1)

    fi:int=0
    while fi<len(fids):
      target_fid:int=fids[fi]
      t_filter0=datetime.now()
      target_constellations:List[Dict[str,int]]=filter_constellations_by_single_funcid_from_soa(source_constellations,source_soa,source_m,target_fid)
      t_filter1=datetime.now()
      stage_filter_ms:int=profile_elapsed_ms_between(t_filter0,t_filter1)
      target_m:int=len(target_constellations)

      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-funcidsingle-case-start] N={N} base_chunk={base_chunk} fid={target_fid} off={off} source_m={source_m} target_m={target_m} read_ms={stage_read_ms} classify_ms={stage_classify_ms} filter_ms={stage_filter_ms}")

      chunk_total:int=0
      stats:List[int]=[0]*46
      stages_inner:List[int]=[0,0,0,0,0,0,0]
      elapsed_text:str="0:00:00.000"
      elapsed_ms:int=0

      if target_m>0:
        chunk_total,stats,stages_inner,elapsed_text,elapsed_ms=exec_solutions_gpu_chunk_profile(N,target_constellations,BLOCK,MAX_BLOCKS,gpu_sort_mode,cross_stripe_safe)

      gpu_total+=chunk_total
      executed_cases+=1

      stage_soa_ms:int=stages_inner[0]
      stage_stats_ms:int=stages_inner[1]
      stage_sort_ms:int=stages_inner[2]
      stage_kernel_ms:int=stages_inner[3]
      stage_reduce_ms:int=stages_inner[4]
      stage_compute_ms:int=stages_inner[5]
      stage_no_read_ms:int=stages_inner[6]
      stage_total_ms:int=stage_read_ms+stage_classify_ms+stage_filter_ms+stage_no_read_ms
      stages:List[int]=[stage_read_ms,stage_soa_ms,stage_stats_ms,stage_sort_ms,stage_kernel_ms,stage_reduce_ms,stage_compute_ms,stage_no_read_ms,stage_total_ms]

      append_stream_funcid_reorder_funcid_single_progress(progress_fname,N,preset_queens,base_chunk,off,target_m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,chunk_total,gpu_total,total_records,stats,stages,STEPS,base_chunk,target_fid,source_m,stage_classify_ms,stage_filter_ms)

      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-funcidsingle-case-end] N={N} base_chunk={base_chunk} fid={target_fid} off={off} source_m={source_m} target_m={target_m} elapsed={elapsed_text} elapsed_ms={elapsed_ms} chunk_total={chunk_total} gpu_total={gpu_total} read_ms={stage_read_ms} classify_ms={stage_classify_ms} filter_ms={stage_filter_ms} soa_ms={stage_soa_ms} stats_ms={stage_stats_ms} sort_ms={stage_sort_ms} kernel_ms={stage_kernel_ms} reduce_ms={stage_reduce_ms} compute_ms={stage_compute_ms} total_ms={stage_total_ms}")

      fi+=1
    ci+=1

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-funcidsingle-summary] N={N} records={total_records} steps={STEPS} cases={executed_cases} total={gpu_total} progress={progress_fname}")

  return gpu_total

def exec_solutions_gpu_bin_stream_funcid_reorder_funcid_split_profile(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,
  debug_chunk_start:int=0,
  debug_chunk_count:int=1,
  chunk_list_spec:str=FUNCID_SPLIT_DEFAULT_CHUNK_LIST,
  split_group_list_spec:str=FUNCID_SPLIT_DEFAULT_GROUP_LIST,
  progress_suffix:str="funcidsplit"
)->int:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  selected_chunks:List[int]=parse_chunk_list_spec(chunk_list_spec)
  if len(selected_chunks)==0:
    selected_chunks=build_chunk_range_list(debug_chunk_start,debug_chunk_count)
  groups:List[str]=parse_funcid_target_group_list_spec(split_group_list_spec)
  if len(groups)==0:
    groups=parse_funcid_target_group_list_spec(FUNCID_SPLIT_DEFAULT_GROUP_LIST)

  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{funcid_reorder_param_tag()}_{progress_suffix}.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_funcid_reorder_funcid_split_progress_header())

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-funcidsplit-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} chunk_list_count={len(selected_chunks)} chunk_list={chunk_list_to_string(selected_chunks)} split_group_list={split_group_list_spec} progress={progress_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP}")

  gpu_total:int=0
  executed_cases:int=0
  ci:int=0
  while ci<len(selected_chunks):
    base_chunk:int=selected_chunks[ci]
    if base_chunk<0:
      ci+=1
      continue
    off:int=base_chunk*STEPS
    if off>=total_records:
      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-funcidsplit-skip] N={N} base_chunk={base_chunk} off={off} reason=off_ge_records")
      ci+=1
      continue
    target_m_read:int=STEPS
    remaining:int=total_records-off
    if target_m_read>remaining:
      target_m_read=remaining

    t_read0=datetime.now()
    source_constellations:List[Dict[str,int]]=read_constellations_bin_range(fname,off,target_m_read)
    t_read1=datetime.now()
    stage_read_ms:int=profile_elapsed_ms_between(t_read0,t_read1)
    source_m:int=len(source_constellations)
    if source_m==0:
      ci+=1
      continue

    t_classify0=datetime.now()
    source_soa:TaskSoA=TaskSoA(source_m)
    source_w_arr:List[u64]=[u64(0)]*source_m
    build_soa_for_range(N,source_constellations,0,source_m,source_soa,source_w_arr)
    t_classify1=datetime.now()
    stage_classify_ms:int=profile_elapsed_ms_between(t_classify0,t_classify1)

    if gpu_log_level>=1:
      print(f"[funcid-reorder-v2-funcidsplit-chunk] N={N} base_chunk={base_chunk} off={off} source_m={source_m} read_ms={stage_read_ms} classify_ms={stage_classify_ms} groups={split_group_list_spec}")

    gi:int=0
    while gi<len(groups):
      group:str=groups[gi]
      t_filter0=datetime.now()
      target_constellations:List[Dict[str,int]]=filter_constellations_by_funcid_split_group_from_soa(source_constellations,source_soa,source_m,group)
      t_filter1=datetime.now()
      stage_filter_ms:int=profile_elapsed_ms_between(t_filter0,t_filter1)
      target_m:int=len(target_constellations)

      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-funcidsplit-case-start] N={N} base_chunk={base_chunk} group={group} off={off} source_m={source_m} target_m={target_m} read_ms={stage_read_ms} classify_ms={stage_classify_ms} filter_ms={stage_filter_ms}")

      chunk_total:int=0
      stats:List[int]=[0]*46
      stages_inner:List[int]=[0,0,0,0,0,0,0]
      elapsed_text:str="0:00:00.000"
      elapsed_ms:int=0

      if target_m>0:
        chunk_total,stats,stages_inner,elapsed_text,elapsed_ms=exec_solutions_gpu_chunk_profile(N,target_constellations,BLOCK,MAX_BLOCKS,gpu_sort_mode,cross_stripe_safe)

      gpu_total+=chunk_total
      executed_cases+=1

      stage_soa_ms:int=stages_inner[0]
      stage_stats_ms:int=stages_inner[1]
      stage_sort_ms:int=stages_inner[2]
      stage_kernel_ms:int=stages_inner[3]
      stage_reduce_ms:int=stages_inner[4]
      stage_compute_ms:int=stages_inner[5]
      stage_no_read_ms:int=stages_inner[6]
      stage_total_ms:int=stage_read_ms+stage_classify_ms+stage_filter_ms+stage_no_read_ms
      stages:List[int]=[stage_read_ms,stage_soa_ms,stage_stats_ms,stage_sort_ms,stage_kernel_ms,stage_reduce_ms,stage_compute_ms,stage_no_read_ms,stage_total_ms]

      append_stream_funcid_reorder_funcid_split_progress(progress_fname,N,preset_queens,base_chunk,off,target_m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,chunk_total,gpu_total,total_records,stats,stages,STEPS,base_chunk,group,source_m,stage_classify_ms,stage_filter_ms)

      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-funcidsplit-case-end] N={N} base_chunk={base_chunk} group={group} off={off} source_m={source_m} target_m={target_m} elapsed={elapsed_text} elapsed_ms={elapsed_ms} chunk_total={chunk_total} gpu_total={gpu_total} read_ms={stage_read_ms} classify_ms={stage_classify_ms} filter_ms={stage_filter_ms} soa_ms={stage_soa_ms} stats_ms={stage_stats_ms} sort_ms={stage_sort_ms} kernel_ms={stage_kernel_ms} reduce_ms={stage_reduce_ms} compute_ms={stage_compute_ms} total_ms={stage_total_ms}")

      gi+=1
    ci+=1

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-funcidsplit-summary] N={N} records={total_records} steps={STEPS} cases={executed_cases} total={gpu_total} progress={progress_fname}")

  return gpu_total

def exec_solutions_gpu_bin_stream_funcid_reorder_funcid_depth_profile(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,
  debug_chunk_start:int=0,
  debug_chunk_count:int=1,
  chunk_list_spec:str=FUNCID_DEPTH_DEFAULT_CHUNK_LIST,
  funcid_list_spec:str=FUNCID_DEPTH_DEFAULT_FUNCID_LIST,
  bucket_list_spec:str=FUNCID_DEPTH_DEFAULT_BUCKET_LIST,
  progress_suffix:str="funciddepth"
)->int:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  selected_chunks:List[int]=parse_chunk_list_spec(chunk_list_spec)
  if len(selected_chunks)==0:
    selected_chunks=build_chunk_range_list(debug_chunk_start,debug_chunk_count)
  fids:List[int]=parse_funcid_single_list_spec(funcid_list_spec)
  if len(fids)==0:
    fids=parse_funcid_single_list_spec(FUNCID_DEPTH_DEFAULT_FUNCID_LIST)
  buckets:List[str]=parse_funcid_depth_bucket_list_spec(bucket_list_spec)
  if len(buckets)==0:
    buckets=parse_funcid_depth_bucket_list_spec(FUNCID_DEPTH_DEFAULT_BUCKET_LIST)

  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{funcid_reorder_param_tag()}_{progress_suffix}.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_funcid_reorder_funcid_depth_progress_header())

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-funciddepth-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} chunk_list_count={len(selected_chunks)} chunk_list={chunk_list_to_string(selected_chunks)} funcid_list={funcid_list_to_string(fids)} bucket_list={bucket_list_spec} progress={progress_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP}")

  gpu_total:int=0
  executed_cases:int=0
  ci:int=0
  while ci<len(selected_chunks):
    base_chunk:int=selected_chunks[ci]
    if base_chunk<0:
      ci+=1
      continue
    off:int=base_chunk*STEPS
    if off>=total_records:
      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-funciddepth-skip] N={N} base_chunk={base_chunk} off={off} reason=off_ge_records")
      ci+=1
      continue
    target_m_read:int=STEPS
    remaining:int=total_records-off
    if target_m_read>remaining:
      target_m_read=remaining

    t_read0=datetime.now()
    source_constellations:List[Dict[str,int]]=read_constellations_bin_range(fname,off,target_m_read)
    t_read1=datetime.now()
    stage_read_ms:int=profile_elapsed_ms_between(t_read0,t_read1)
    source_m:int=len(source_constellations)
    if source_m==0:
      ci+=1
      continue

    t_classify0=datetime.now()
    source_soa:TaskSoA=TaskSoA(source_m)
    source_w_arr:List[u64]=[u64(0)]*source_m
    build_soa_for_range(N,source_constellations,0,source_m,source_soa,source_w_arr)
    t_classify1=datetime.now()
    stage_classify_ms:int=profile_elapsed_ms_between(t_classify0,t_classify1)

    if gpu_log_level>=1:
      print(f"[funcid-reorder-v2-funciddepth-chunk] N={N} base_chunk={base_chunk} off={off} source_m={source_m} read_ms={stage_read_ms} classify_ms={stage_classify_ms} funcids={funcid_list_to_string(fids)} buckets={bucket_list_spec}")

    fi:int=0
    while fi<len(fids):
      target_fid:int=fids[fi]
      bi:int=0
      while bi<len(buckets):
        bucket:str=buckets[bi]
        t_filter0=datetime.now()
        target_constellations:List[Dict[str,int]]=filter_constellations_by_funcid_depth_bucket_from_soa(source_constellations,source_soa,source_m,target_fid,bucket)
        bucket_summary:List[int]=summarize_funcid_depth_bucket_from_soa(source_soa,source_m,target_fid,bucket)
        t_filter1=datetime.now()
        stage_filter_ms:int=profile_elapsed_ms_between(t_filter0,t_filter1)
        target_m:int=len(target_constellations)
        depth_min:int=bucket_summary[1]
        depth_max:int=bucket_summary[2]
        free_pc_min:int=bucket_summary[3]
        free_pc_max:int=bucket_summary[4]

        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-funciddepth-case-start] N={N} base_chunk={base_chunk} fid={target_fid} bucket={bucket} off={off} source_m={source_m} target_m={target_m} depth_min={depth_min} depth_max={depth_max} free_pc_min={free_pc_min} free_pc_max={free_pc_max} read_ms={stage_read_ms} classify_ms={stage_classify_ms} filter_ms={stage_filter_ms}")

        chunk_total:int=0
        stats:List[int]=[0]*46
        stages_inner:List[int]=[0,0,0,0,0,0,0]
        elapsed_text:str="0:00:00.000"
        elapsed_ms:int=0

        if target_m>0:
          chunk_total,stats,stages_inner,elapsed_text,elapsed_ms=exec_solutions_gpu_chunk_profile(N,target_constellations,BLOCK,MAX_BLOCKS,gpu_sort_mode,cross_stripe_safe)

        gpu_total+=chunk_total
        executed_cases+=1

        stage_soa_ms:int=stages_inner[0]
        stage_stats_ms:int=stages_inner[1]
        stage_sort_ms:int=stages_inner[2]
        stage_kernel_ms:int=stages_inner[3]
        stage_reduce_ms:int=stages_inner[4]
        stage_compute_ms:int=stages_inner[5]
        stage_no_read_ms:int=stages_inner[6]
        stage_total_ms:int=stage_read_ms+stage_classify_ms+stage_filter_ms+stage_no_read_ms
        stages:List[int]=[stage_read_ms,stage_soa_ms,stage_stats_ms,stage_sort_ms,stage_kernel_ms,stage_reduce_ms,stage_compute_ms,stage_no_read_ms,stage_total_ms]

        append_stream_funcid_reorder_funcid_depth_progress(progress_fname,N,preset_queens,base_chunk,off,target_m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,chunk_total,gpu_total,total_records,stats,stages,STEPS,base_chunk,target_fid,bucket,source_m,depth_min,depth_max,free_pc_min,free_pc_max,stage_classify_ms,stage_filter_ms)

        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-funciddepth-case-end] N={N} base_chunk={base_chunk} fid={target_fid} bucket={bucket} off={off} source_m={source_m} target_m={target_m} elapsed={elapsed_text} elapsed_ms={elapsed_ms} chunk_total={chunk_total} gpu_total={gpu_total} depth_min={depth_min} depth_max={depth_max} free_pc_min={free_pc_min} free_pc_max={free_pc_max} read_ms={stage_read_ms} classify_ms={stage_classify_ms} filter_ms={stage_filter_ms} soa_ms={stage_soa_ms} stats_ms={stage_stats_ms} sort_ms={stage_sort_ms} kernel_ms={stage_kernel_ms} reduce_ms={stage_reduce_ms} compute_ms={stage_compute_ms} total_ms={stage_total_ms}")

        bi+=1
      fi+=1
    ci+=1

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-funciddepth-summary] N={N} records={total_records} steps={STEPS} cases={executed_cases} total={gpu_total} progress={progress_fname}")

  return gpu_total

def exec_solutions_gpu_bin_stream_funcid_reorder_funcid_mark_profile(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,
  debug_chunk_start:int=0,
  debug_chunk_count:int=1,
  chunk_list_spec:str=FUNCID_MARK_DEFAULT_CHUNK_LIST,
  funcid_list_spec:str=FUNCID_MARK_DEFAULT_FUNCID_LIST,
  bucket_list_spec:str=FUNCID_MARK_DEFAULT_BUCKET_LIST,
  progress_suffix:str="funcidmark"
)->int:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  selected_chunks:List[int]=parse_chunk_list_spec(chunk_list_spec)
  if len(selected_chunks)==0:
    selected_chunks=build_chunk_range_list(debug_chunk_start,debug_chunk_count)
  fids:List[int]=parse_funcid_single_list_spec(funcid_list_spec)
  if len(fids)==0:
    fids=parse_funcid_single_list_spec(FUNCID_MARK_DEFAULT_FUNCID_LIST)
  buckets:List[str]=parse_funcid_mark_bucket_list_spec(bucket_list_spec)
  if len(buckets)==0:
    buckets=parse_funcid_mark_bucket_list_spec(FUNCID_MARK_DEFAULT_BUCKET_LIST)

  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{funcid_reorder_param_tag()}_{progress_suffix}.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_funcid_reorder_funcid_mark_progress_header())

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-funcidmark-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} chunk_list_count={len(selected_chunks)} chunk_list={chunk_list_to_string(selected_chunks)} funcid_list={funcid_list_to_string(fids)} bucket_list={bucket_list_spec} progress={progress_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP}")

  gpu_total:int=0
  executed_cases:int=0
  ci:int=0
  while ci<len(selected_chunks):
    base_chunk:int=selected_chunks[ci]
    if base_chunk<0:
      ci+=1
      continue
    off:int=base_chunk*STEPS
    if off>=total_records:
      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-funcidmark-skip] N={N} base_chunk={base_chunk} off={off} reason=off_ge_records")
      ci+=1
      continue
    target_m_read:int=STEPS
    remaining:int=total_records-off
    if target_m_read>remaining:
      target_m_read=remaining

    t_read0=datetime.now()
    source_constellations:List[Dict[str,int]]=read_constellations_bin_range(fname,off,target_m_read)
    t_read1=datetime.now()
    stage_read_ms:int=profile_elapsed_ms_between(t_read0,t_read1)
    source_m:int=len(source_constellations)
    if source_m==0:
      ci+=1
      continue

    t_classify0=datetime.now()
    source_soa:TaskSoA=TaskSoA(source_m)
    source_w_arr:List[u64]=[u64(0)]*source_m
    build_soa_for_range(N,source_constellations,0,source_m,source_soa,source_w_arr)
    t_classify1=datetime.now()
    stage_classify_ms:int=profile_elapsed_ms_between(t_classify0,t_classify1)

    if gpu_log_level>=1:
      print(f"[funcid-reorder-v2-funcidmark-chunk] N={N} base_chunk={base_chunk} off={off} source_m={source_m} read_ms={stage_read_ms} classify_ms={stage_classify_ms} funcids={funcid_list_to_string(fids)} buckets={bucket_list_spec}")

    fi:int=0
    while fi<len(fids):
      target_fid:int=fids[fi]
      bi:int=0
      while bi<len(buckets):
        bucket:str=buckets[bi]
        t_filter0=datetime.now()
        target_constellations:List[Dict[str,int]]=filter_constellations_by_funcid_mark_bucket_from_soa(source_constellations,source_soa,source_m,target_fid,bucket)
        bucket_summary:List[int]=summarize_funcid_mark_bucket_from_soa(source_soa,source_m,target_fid,bucket)
        t_filter1=datetime.now()
        stage_filter_ms:int=profile_elapsed_ms_between(t_filter0,t_filter1)
        target_m:int=len(target_constellations)
        mark1_min:int=bucket_summary[1]
        mark1_max:int=bucket_summary[2]
        mark2_min:int=bucket_summary[3]
        mark2_max:int=bucket_summary[4]
        mark_gap_min:int=bucket_summary[5]
        mark_gap_max:int=bucket_summary[6]
        row_to_mark1_min:int=bucket_summary[7]
        row_to_mark1_max:int=bucket_summary[8]
        row_to_mark2_min:int=bucket_summary[9]
        row_to_mark2_max:int=bucket_summary[10]
        jmark_min:int=bucket_summary[11]
        jmark_max:int=bucket_summary[12]
        endmark_min:int=bucket_summary[13]
        endmark_max:int=bucket_summary[14]

        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-funcidmark-case-start] N={N} base_chunk={base_chunk} fid={target_fid} bucket={bucket} off={off} source_m={source_m} target_m={target_m} mark1={mark1_min}-{mark1_max} mark2={mark2_min}-{mark2_max} gap={mark_gap_min}-{mark_gap_max} row_to_mark1={row_to_mark1_min}-{row_to_mark1_max} row_to_mark2={row_to_mark2_min}-{row_to_mark2_max} jmark={jmark_min}-{jmark_max} endmark={endmark_min}-{endmark_max} read_ms={stage_read_ms} classify_ms={stage_classify_ms} filter_ms={stage_filter_ms}")

        chunk_total:int=0
        stats:List[int]=[0]*46
        stages_inner:List[int]=[0,0,0,0,0,0,0]
        elapsed_text:str="0:00:00.000"
        elapsed_ms:int=0

        if target_m>0:
          chunk_total,stats,stages_inner,elapsed_text,elapsed_ms=exec_solutions_gpu_chunk_profile(N,target_constellations,BLOCK,MAX_BLOCKS,gpu_sort_mode,cross_stripe_safe)

        gpu_total+=chunk_total
        executed_cases+=1

        stage_soa_ms:int=stages_inner[0]
        stage_stats_ms:int=stages_inner[1]
        stage_sort_ms:int=stages_inner[2]
        stage_kernel_ms:int=stages_inner[3]
        stage_reduce_ms:int=stages_inner[4]
        stage_compute_ms:int=stages_inner[5]
        stage_no_read_ms:int=stages_inner[6]
        stage_total_ms:int=stage_read_ms+stage_classify_ms+stage_filter_ms+stage_no_read_ms
        stages:List[int]=[stage_read_ms,stage_soa_ms,stage_stats_ms,stage_sort_ms,stage_kernel_ms,stage_reduce_ms,stage_compute_ms,stage_no_read_ms,stage_total_ms]

        append_stream_funcid_reorder_funcid_mark_progress(progress_fname,N,preset_queens,base_chunk,off,target_m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,chunk_total,gpu_total,total_records,stats,stages,STEPS,base_chunk,target_fid,bucket,source_m,mark1_min,mark1_max,mark2_min,mark2_max,mark_gap_min,mark_gap_max,row_to_mark1_min,row_to_mark1_max,row_to_mark2_min,row_to_mark2_max,jmark_min,jmark_max,endmark_min,endmark_max,stage_classify_ms,stage_filter_ms)

        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-funcidmark-case-end] N={N} base_chunk={base_chunk} fid={target_fid} bucket={bucket} off={off} source_m={source_m} target_m={target_m} elapsed={elapsed_text} elapsed_ms={elapsed_ms} chunk_total={chunk_total} gpu_total={gpu_total} mark1={mark1_min}-{mark1_max} mark2={mark2_min}-{mark2_max} gap={mark_gap_min}-{mark_gap_max} row_to_mark1={row_to_mark1_min}-{row_to_mark1_max} row_to_mark2={row_to_mark2_min}-{row_to_mark2_max} jmark={jmark_min}-{jmark_max} endmark={endmark_min}-{endmark_max} read_ms={stage_read_ms} classify_ms={stage_classify_ms} filter_ms={stage_filter_ms} soa_ms={stage_soa_ms} stats_ms={stage_stats_ms} sort_ms={stage_sort_ms} kernel_ms={stage_kernel_ms} reduce_ms={stage_reduce_ms} compute_ms={stage_compute_ms} total_ms={stage_total_ms}")

        bi+=1
      fi+=1
    ci+=1

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-funcidmark-summary] N={N} records={total_records} steps={STEPS} cases={executed_cases} total={gpu_total} progress={progress_fname}")

  return gpu_total

def exec_solutions_gpu_bin_stream_funcid_reorder_funcid_markdist_profile(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,
  debug_chunk_start:int=0,
  debug_chunk_count:int=1,
  chunk_list_spec:str=FUNCID_MARKDIST_DEFAULT_CHUNK_LIST,
  funcid_list_spec:str=FUNCID_MARKDIST_DEFAULT_FUNCID_LIST,
  axis_list_spec:str=FUNCID_MARKDIST_DEFAULT_AXIS_LIST,
  progress_suffix:str="funcidmarkdist"
)->int:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  selected_chunks:List[int]=parse_chunk_list_spec(chunk_list_spec)
  if len(selected_chunks)==0:
    selected_chunks=build_chunk_range_list(debug_chunk_start,debug_chunk_count)
  fids:List[int]=parse_funcid_single_list_spec(funcid_list_spec)
  if len(fids)==0:
    fids=parse_funcid_single_list_spec(FUNCID_MARKDIST_DEFAULT_FUNCID_LIST)
  axes:List[str]=parse_funcid_markdist_axis_list_spec(axis_list_spec)
  if len(axes)==0:
    axes=parse_funcid_markdist_axis_list_spec(FUNCID_MARKDIST_DEFAULT_AXIS_LIST)

  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{funcid_reorder_param_tag()}_{progress_suffix}.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_funcid_reorder_funcid_markdist_progress_header())

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-funcidmarkdist-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} chunk_list_count={len(selected_chunks)} chunk_list={chunk_list_to_string(selected_chunks)} funcid_list={funcid_list_to_string(fids)} axis_list={axis_list_spec} progress={progress_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP}")

  gpu_total:int=0
  executed_cases:int=0
  ci:int=0
  while ci<len(selected_chunks):
    base_chunk:int=selected_chunks[ci]
    if base_chunk<0:
      ci+=1
      continue
    off:int=base_chunk*STEPS
    if off>=total_records:
      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-funcidmarkdist-skip] N={N} base_chunk={base_chunk} off={off} reason=off_ge_records")
      ci+=1
      continue
    target_m_read:int=STEPS
    remaining:int=total_records-off
    if target_m_read>remaining:
      target_m_read=remaining

    t_read0=datetime.now()
    source_constellations:List[Dict[str,int]]=read_constellations_bin_range(fname,off,target_m_read)
    t_read1=datetime.now()
    stage_read_ms:int=profile_elapsed_ms_between(t_read0,t_read1)
    source_m:int=len(source_constellations)
    if source_m==0:
      ci+=1
      continue

    t_classify0=datetime.now()
    source_soa:TaskSoA=TaskSoA(source_m)
    source_w_arr:List[u64]=[u64(0)]*source_m
    build_soa_for_range(N,source_constellations,0,source_m,source_soa,source_w_arr)
    t_classify1=datetime.now()
    stage_classify_ms:int=profile_elapsed_ms_between(t_classify0,t_classify1)

    if gpu_log_level>=1:
      print(f"[funcid-reorder-v2-funcidmarkdist-chunk] N={N} base_chunk={base_chunk} off={off} source_m={source_m} read_ms={stage_read_ms} classify_ms={stage_classify_ms} funcids={funcid_list_to_string(fids)} axes={axis_list_spec}")

    fi:int=0
    while fi<len(fids):
      target_fid:int=fids[fi]
      ai:int=0
      while ai<len(axes):
        axis:str=axes[ai]
        keys:List[str]=collect_funcid_markdist_keys_from_soa(source_soa,source_m,target_fid,axis)
        ki:int=0
        while ki<len(keys):
          key:str=keys[ki]
          t_filter0=datetime.now()
          target_constellations:List[Dict[str,int]]=filter_constellations_by_funcid_markdist_key_from_soa(source_constellations,source_soa,source_m,target_fid,axis,key)
          bucket_summary:List[int]=summarize_funcid_markdist_key_from_soa(source_soa,source_m,target_fid,axis,key)
          t_filter1=datetime.now()
          stage_filter_ms:int=profile_elapsed_ms_between(t_filter0,t_filter1)
          target_m:int=len(target_constellations)
          mark1_min:int=bucket_summary[1]
          mark1_max:int=bucket_summary[2]
          mark2_min:int=bucket_summary[3]
          mark2_max:int=bucket_summary[4]
          mark_gap_min:int=bucket_summary[5]
          mark_gap_max:int=bucket_summary[6]
          d1_min:int=bucket_summary[7]
          d1_max:int=bucket_summary[8]
          d2_min:int=bucket_summary[9]
          d2_max:int=bucket_summary[10]
          jmark_min:int=bucket_summary[11]
          jmark_max:int=bucket_summary[12]
          endmark_min:int=bucket_summary[13]
          endmark_max:int=bucket_summary[14]

          if gpu_log_level>=1:
            print(f"[funcid-reorder-v2-funcidmarkdist-case-start] N={N} base_chunk={base_chunk} fid={target_fid} axis={axis} key={key} off={off} source_m={source_m} target_m={target_m} mark1={mark1_min}-{mark1_max} mark2={mark2_min}-{mark2_max} gap={mark_gap_min}-{mark_gap_max} d1={d1_min}-{d1_max} d2={d2_min}-{d2_max} jmark={jmark_min}-{jmark_max} endmark={endmark_min}-{endmark_max} read_ms={stage_read_ms} classify_ms={stage_classify_ms} filter_ms={stage_filter_ms}")

          chunk_total:int=0
          stats:List[int]=[0]*46
          stages_inner:List[int]=[0,0,0,0,0,0,0]
          elapsed_text:str="0:00:00.000"
          elapsed_ms:int=0

          if target_m>0:
            chunk_total,stats,stages_inner,elapsed_text,elapsed_ms=exec_solutions_gpu_chunk_profile(N,target_constellations,BLOCK,MAX_BLOCKS,gpu_sort_mode,cross_stripe_safe)

          gpu_total+=chunk_total
          executed_cases+=1

          stage_soa_ms:int=stages_inner[0]
          stage_stats_ms:int=stages_inner[1]
          stage_sort_ms:int=stages_inner[2]
          stage_kernel_ms:int=stages_inner[3]
          stage_reduce_ms:int=stages_inner[4]
          stage_compute_ms:int=stages_inner[5]
          stage_no_read_ms:int=stages_inner[6]
          stage_total_ms:int=stage_read_ms+stage_classify_ms+stage_filter_ms+stage_no_read_ms
          stages:List[int]=[stage_read_ms,stage_soa_ms,stage_stats_ms,stage_sort_ms,stage_kernel_ms,stage_reduce_ms,stage_compute_ms,stage_no_read_ms,stage_total_ms]

          append_stream_funcid_reorder_funcid_markdist_progress(progress_fname,N,preset_queens,base_chunk,off,target_m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,chunk_total,gpu_total,total_records,stats,stages,STEPS,base_chunk,target_fid,axis,key,source_m,mark1_min,mark1_max,mark2_min,mark2_max,mark_gap_min,mark_gap_max,d1_min,d1_max,d2_min,d2_max,jmark_min,jmark_max,endmark_min,endmark_max,stage_classify_ms,stage_filter_ms)

          if gpu_log_level>=1:
            print(f"[funcid-reorder-v2-funcidmarkdist-case-end] N={N} base_chunk={base_chunk} fid={target_fid} axis={axis} key={key} off={off} source_m={source_m} target_m={target_m} elapsed={elapsed_text} elapsed_ms={elapsed_ms} chunk_total={chunk_total} gpu_total={gpu_total} mark1={mark1_min}-{mark1_max} mark2={mark2_min}-{mark2_max} gap={mark_gap_min}-{mark_gap_max} d1={d1_min}-{d1_max} d2={d2_min}-{d2_max} jmark={jmark_min}-{jmark_max} endmark={endmark_min}-{endmark_max} read_ms={stage_read_ms} classify_ms={stage_classify_ms} filter_ms={stage_filter_ms} soa_ms={stage_soa_ms} stats_ms={stage_stats_ms} sort_ms={stage_sort_ms} kernel_ms={stage_kernel_ms} reduce_ms={stage_reduce_ms} compute_ms={stage_compute_ms} total_ms={stage_total_ms}")

          ki+=1
        ai+=1
      fi+=1
    ci+=1

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-funcidmarkdist-summary] N={N} records={total_records} steps={STEPS} cases={executed_cases} total={gpu_total} progress={progress_fname}")

  return gpu_total

def exec_solutions_gpu_bin_stream_stats_only(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1
)->int:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_measure2_stats.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_measure2_progress_header())

  if gpu_log_level>=1:
    print(f"[stream-stats-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} progress={progress_fname} stats_only=1")

  off:int=0
  chunk_index:int=0
  _read_uint32_le=read_uint32_le

  with open(fname,"rb") as f:
    while True:
      chunk_constellations:List[Dict[str,int]]=[]
      i:int=0
      while i<STEPS:
        raw:str=f.read(16)
        if len(raw)<16:
          break
        ld=_read_uint32_le(raw[0:4])
        rd=_read_uint32_le(raw[4:8])
        col=_read_uint32_le(raw[8:12])
        startijkl=_read_uint32_le(raw[12:16])
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1

      m:int=len(chunk_constellations)
      if m==0:
        break

      t0=datetime.now()
      stats:List[int]=analyze_stream_chunk_input_stats(N,chunk_constellations)
      t1=datetime.now()
      elapsed_text:str=str(t1-t0)[:-3]
      elapsed_ms:int=stream_elapsed_text_to_ms(elapsed_text)
      append_stream_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,0,0,total_records,stats)

      if gpu_log_level>=2:
        print(f"[stream-stats-chunk] N={N} chunk={chunk_index} off={off} m={m} elapsed={elapsed_text} elapsed_ms={elapsed_ms} free_avg={format_ratio_3(stats[0],m)} depth_avg={format_ratio_3(stats[9],m)} score_avg={format_ratio_3(stats[12],m)}")

      off+=m
      chunk_index+=1

  if gpu_log_level>=1:
    print(f"[stream-stats-summary] N={N} records={total_records} chunks={chunk_index} progress={progress_fname} stats_only=1")

  return chunk_index

def exec_solutions_gpu_bin_stream(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,
  chunk_only:bool=False,
  debug_chunk_start:int=0,
  debug_chunk_count:int=1
)->int:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_measure2.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_measure2_progress_header())

  if chunk_only:
    if debug_chunk_start<0:
      debug_chunk_start=0
    if debug_chunk_count<=0:
      debug_chunk_count=1

  if gpu_log_level>=1:
    print(f"[stream-gpu-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} chunk_only={1 if chunk_only else 0} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} progress={progress_fname} inner_log_level=0 measure2=1")

  gpu_total:int=0
  off:int=0
  chunk_index:int=0
  executed_chunks:int=0
  _read_uint32_le=read_uint32_le

  with open(fname,"rb") as f:
    while True:
      chunk_constellations:List[Dict[str,int]]=[]
      i:int=0
      while i<STEPS:
        raw:str=f.read(16)
        if len(raw)<16:
          break
        ld=_read_uint32_le(raw[0:4])
        rd=_read_uint32_le(raw[4:8])
        col=_read_uint32_le(raw[8:12])
        startijkl=_read_uint32_le(raw[12:16])
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1

      m:int=len(chunk_constellations)
      if m==0:
        break

      if chunk_only:
        run_this_chunk:bool=(chunk_index>=debug_chunk_start and chunk_index<debug_chunk_start+debug_chunk_count)
        if not run_this_chunk:
          if gpu_log_level>=2:
            print(f"[stream-gpu-chunk-skip] N={N} chunk={chunk_index} off={off} m={m}")
          off+=m
          chunk_index+=1
          continue

      stats:List[int]=analyze_stream_chunk_input_stats(N,chunk_constellations)

      t0=datetime.now()
      if gpu_log_level>=1:
        print(f"[stream-gpu-chunk-start] N={N} chunk={chunk_index} off={off} m={m}")

      inner_gpu_log_level:int=0
      exec_solutions(N,chunk_constellations,True,gpu_block,gpu_max_blocks,inner_gpu_log_level,gpu_sort_mode,cross_stripe_safe)

      chunk_total:int=0
      if m>0:
        chunk_total=chunk_constellations[0]["solutions"]
      gpu_total+=chunk_total
      executed_chunks+=1
      t1=datetime.now()
      elapsed_text:str=str(t1-t0)[:-3]
      elapsed_ms:int=stream_elapsed_text_to_ms(elapsed_text)
      append_stream_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,chunk_total,gpu_total,total_records,stats)
      if gpu_log_level>=1:
        print(f"[stream-gpu-chunk-end] N={N} chunk={chunk_index} off={off} m={m} elapsed={elapsed_text} elapsed_ms={elapsed_ms} chunk_total={chunk_total} gpu_total={gpu_total}")

      off+=m
      chunk_index+=1

  if gpu_log_level>=1:
    print(f"[stream-gpu-summary] N={N} records={total_records} chunks={chunk_index} executed_chunks={executed_chunks} total={gpu_total} progress={progress_fname} measure2=1")

  return gpu_total

def select_dynamic_preset_queens(N:int,preset_queens:int)->int:
  if N>=5 and N<=17:
    return 5
  elif N>=18 and N<=21:
    return 6
  elif N>=22 and N<=24:
    return 7
  elif N>=25 and N<=27:
    return 8
  return preset_queens

def build_constellations_dynamicK(N: int, ijkl_list:Set[int],subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],constellations:List[Dict[str,int]],use_gpu: bool,preset_queens:int)->Tuple[Set[int],Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],List[Dict[str,int]],int]:

  preset_queens=select_dynamic_preset_queens(N,preset_queens)
  use_bin=True
  if use_bin:
    ijkl_list,subconst_cache,constellations,preset_queens=load_or_build_constellations_bin(N,ijkl_list,subconst_cache, constellations, preset_queens)
  else:
    ijkl_list,subconst_cache,constellations,preset_queens=load_or_build_constellations_txt(N,ijkl_list,subconst_cache, constellations, preset_queens)

  return  ijkl_list,subconst_cache,constellations,preset_queens

def _bit_total(N:int)->int:
  mask:int=(1<<N)-1
  def bt(row:int,left:int,down:int,right:int)->int:
    if row==N:
      return 1
    total:int=0
    bitmap:int=mask&~(left|down|right)
    while bitmap:
      bit:int=-bitmap&bitmap
      bitmap^=bit
      total+=bt(row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
    return total
  return bt(0,0,0,0)

def main()->None:
  global DISABLE_CONSTELLATION_SIGNATURE_PRUNE
  global FUNCID_REORDER_V2_WINDOW_MULT,FUNCID_REORDER_V2_PHASE_JUMP,FUNCID_REORDER_V2_SWEEP_TEMP_OUTPUT
  global BROAD_MARKDIST_TAIL_VARIANT

  expected:List[int]=[0,0,0,0,0,10,4,40,92,352,724,2680,14200,73712,365596,2279184,14772512,95815104,666090624,4968057848,39029188884,314666222712,2691008701644,24233937684440,227514171973736,2207893435808352,22317699616364044,234907967154122528]
  nmin:int=DEFAULT_RANGE_NMIN
  nmax:int=DEFAULT_RANGE_NMAX_EXCLUSIVE
  use_gpu:bool=False
  gpu_block:int=32
  gpu_max_blocks:int=484
  gpu_log_level:int=0
  gpu_sort_mode:int=-1
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT
  debug_chunk_start:int=0
  debug_chunk_count:int=1
  microbench_chunk_list_spec:str=MICROBENCH_DEFAULT_CHUNK_LIST
  chunksize_factor_list_spec:str=CHUNKSIZE_DEFAULT_FACTOR_LIST
  funcid_target_group_list_spec:str=FUNCID_TARGET_DEFAULT_GROUP_LIST
  funcid_single_list_spec:str=FUNCID_SINGLE_DEFAULT_FUNCID_LIST
  funcid_split_group_list_spec:str=FUNCID_SPLIT_DEFAULT_GROUP_LIST
  funcid_depth_bucket_list_spec:str=FUNCID_DEPTH_DEFAULT_BUCKET_LIST
  funcid_mark_bucket_list_spec:str=FUNCID_MARK_DEFAULT_BUCKET_LIST
  funcid_markdist_axis_list_spec:str=FUNCID_MARKDIST_DEFAULT_AXIS_LIST
  bench_mode:int=0  # 0:normal, 1:N20 warmup repeat, 2:N19 preheat, 3:N18+N19 preheat, 4:N20 repeat3 sweep, 5:N20 repeat2 benchmark, 6:reorder-only debug, 7:chunk-only debug, 8:boundary-classification-only, 9:boundary-solution-summary, 10:boundary-classification-only + signature prune disabled, 11:stream-bin-build-only, 13:stream-input-stats-only, 14:funcid-reorder-v2-sim-only, 15:funcid-reorder-v2-gpu, 16:funcid-reorder-v2-sim-sweep, 17:funcid-reorder-v2-microbench, 18:funcid-reorder-v2-profile, 19:funcid-reorder-v2-chunksize-profile, 20:funcid-reorder-v2-funcid-target-profile, 21:funcid-reorder-v2-funcid-single-profile, 22:funcid-reorder-v2-funcid-split-profile, 23:funcid-reorder-v2-funcid-depth-profile, 24:funcid-reorder-v2-funcid-mark-profile, 25:funcid-reorder-v2-exact-mark-distance-profile, 26:markdist-risk-reorder-sim-only, 27:markdist-risk-reorder-gpu, 28:broadmarktail-reorder-sim-only, 29:broadmarktail-reorder-gpu, 30:split145-generic-probe, 31:split145-generic-full-gpu
  reorder_window_mult:int=FUNCID_REORDER_V2_WINDOW_MULT
  reorder_phase_jump:int=FUNCID_REORDER_V2_PHASE_JUMP
  worker_id:int=0
  worker_count:int=1
  broadmark_tail_variant:int=BROAD_MARKDIST_TAIL_VARIANT
  preset_queens_arg:int=5
  requested_preset_arg:int=5
  argc:int=len(sys.argv)

  if argc == 1:
    use_gpu=False
    nmin=CPU_FINAL_DEFAULT_N
    nmax=CPU_FINAL_DEFAULT_N+1
    requested_preset_arg=5
    preset_queens_arg=5
    bench_mode=0
    print("CPU auto mode selected")
    print("[115-default] no arguments: CPU N22 default; use -g for A10G range mode")
  elif argc >= 2:
    arg = sys.argv[1]
    if arg == "-c":
      use_gpu = False
      print("CPU mode selected")
      if argc == 2:
        requested_preset_arg=5
        preset_queens_arg=5
        bench_mode=0
    elif arg == "-g":
      use_gpu = True
      print("GPU mode selected")
      if argc == 2:
        gpu_block=A10G_FINAL_DEFAULT_BLOCK
        gpu_max_blocks=A10G_FINAL_DEFAULT_MAX_BLOCKS
        gpu_log_level=A10G_FINAL_DEFAULT_LOG_LEVEL
        gpu_sort_mode=A10G_FINAL_DEFAULT_SORT_MODE
        requested_preset_arg=A10G_FINAL_DEFAULT_PRESET
        preset_queens_arg=A10G_FINAL_DEFAULT_PRESET
        bench_mode=A10G_FINAL_DEFAULT_BENCH_MODE
        reorder_window_mult=A10G_FINAL_DEFAULT_REORDER_WINDOW_MULT
        reorder_phase_jump=A10G_FINAL_DEFAULT_REORDER_PHASE_JUMP
        cross_stripe_safe=A10G_FINAL_DEFAULT_CROSS_STRIPE_SAFE
        worker_id=A10G_FINAL_DEFAULT_WORKER_ID
        worker_count=A10G_FINAL_DEFAULT_WORKER_COUNT
        broadmark_tail_variant=A10G_FINAL_DEFAULT_BROADMARK_VARIANT
    else:
      print(f"Unknown option: {arg}")
      print("Usage: nqueens [-c | -g] [nmin nmax] [gpu_block gpu_max_blocks log_level sort_mode] [preset_queens] [bench_mode] [cross_stripe_safe] [debug_chunk_start] [debug_chunk_count] [reorder_window_mult] [reorder_phase_jump]")
      return

    if argc >= 4:
      nmin=int(sys.argv[2])
      nmax=int(sys.argv[3])+1
    if argc >= 5:
      gpu_block=int(sys.argv[4])
    if argc >= 6:
      gpu_max_blocks=int(sys.argv[5])
    if argc >= 7:
      gpu_log_level=int(sys.argv[6])
    if argc >= 8:
      gpu_sort_mode=int(sys.argv[7])
    if argc >= 9:
      requested_preset_arg=int(sys.argv[8])
    if argc >= 10:
      bench_mode=int(sys.argv[9])
      if bench_mode<0 or (bench_mode>11 and bench_mode!=13 and bench_mode!=14 and bench_mode!=15 and bench_mode!=16 and bench_mode!=17 and bench_mode!=18 and bench_mode!=19 and bench_mode!=20 and bench_mode!=21 and bench_mode!=22 and bench_mode!=23 and bench_mode!=24 and bench_mode!=25 and bench_mode!=26 and bench_mode!=27 and bench_mode!=28 and bench_mode!=29 and bench_mode!=30 and bench_mode!=31):
        print(f"[warning] unknown bench_mode={bench_mode}; using 0")
        bench_mode=0
    if bench_mode>=8:
      preset_queens_arg=requested_preset_arg
    else:
      if requested_preset_arg!=5:
        print(f"[warning] preset_queens={requested_preset_arg} is disabled in 77 normal modes; using 5")
      preset_queens_arg=5
    if bench_mode==14 or bench_mode==15 or bench_mode==17 or bench_mode==18 or bench_mode==19 or bench_mode==20 or bench_mode==21 or bench_mode==22 or bench_mode==23 or bench_mode==24 or bench_mode==25 or bench_mode==26 or bench_mode==27 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31:
      if argc >= 11:
        reorder_window_mult=int(sys.argv[10])
      if argc >= 12:
        reorder_phase_jump=int(sys.argv[11])
      if argc >= 13:
        cross_stripe_safe=(int(sys.argv[12])!=0)
      if bench_mode==29 or bench_mode==31:
        if argc >= 14:
          worker_id=int(sys.argv[13])
        if argc >= 15:
          worker_count=int(sys.argv[14])
        if argc >= 16:
          broadmark_tail_variant=int(sys.argv[15])
      if bench_mode==28:
        if argc >= 14:
          broadmark_tail_variant=int(sys.argv[13])
      if bench_mode==17 or bench_mode==18 or bench_mode==19 or bench_mode==20 or bench_mode==21 or bench_mode==22 or bench_mode==23 or bench_mode==24 or bench_mode==25 or bench_mode==30:
        if argc >= 14:
          debug_chunk_start=int(sys.argv[13])
          microbench_chunk_list_spec=""
        if argc >= 15:
          debug_chunk_count=int(sys.argv[14])
        if argc >= 16:
          microbench_chunk_list_spec=sys.argv[15]
        if bench_mode==19 and argc >= 17:
          chunksize_factor_list_spec=sys.argv[16]
        if bench_mode==20 and argc >= 17:
          funcid_target_group_list_spec=sys.argv[16]
        if bench_mode==21 and argc >= 17:
          funcid_single_list_spec=sys.argv[16]
        if bench_mode==22 and argc >= 17:
          funcid_split_group_list_spec=sys.argv[16]
        if bench_mode==23 and argc >= 17:
          funcid_single_list_spec=sys.argv[16]
        if bench_mode==23 and argc >= 18:
          funcid_depth_bucket_list_spec=sys.argv[17]
        if bench_mode==24 and argc >= 17:
          funcid_single_list_spec=sys.argv[16]
        if bench_mode==24 and argc >= 18:
          funcid_mark_bucket_list_spec=sys.argv[17]
        if bench_mode==25 and argc >= 17:
          funcid_single_list_spec=sys.argv[16]
        if bench_mode==25 and argc >= 18:
          funcid_markdist_axis_list_spec=sys.argv[17]
        if bench_mode==30 and argc >= 17:
          broadmark_tail_variant=int(sys.argv[16])
        if (bench_mode==17 or bench_mode==18) and argc > 16:
          print("Too many arguments")
          print("Usage microbench/profile: nqueens -g nmin nmax block max_blocks log_level sort_mode preset_queens mode[17|18] [reorder_window_mult] [reorder_phase_jump] [cross_stripe_safe] [chunk_start] [chunk_count] [chunk_list]")
          return
        if bench_mode==19 and argc > 17:
          print("Too many arguments")
          print("Usage chunksize: nqueens -g nmin nmax block max_blocks log_level sort_mode preset_queens 19 [reorder_window_mult] [reorder_phase_jump] [cross_stripe_safe] [chunk_start] [chunk_count] [chunk_list] [factor_list]")
          return
        if bench_mode==20 and argc > 17:
          print("Too many arguments")
          print("Usage funcid-target: nqueens -g nmin nmax block max_blocks log_level sort_mode preset_queens 20 [reorder_window_mult] [reorder_phase_jump] [cross_stripe_safe] [chunk_start] [chunk_count] [chunk_list] [group_list]")
          return
        if bench_mode==21 and argc > 17:
          print("Too many arguments")
          print("Usage funcid-single: nqueens -g nmin nmax block max_blocks log_level sort_mode preset_queens 21 [reorder_window_mult] [reorder_phase_jump] [cross_stripe_safe] [chunk_start] [chunk_count] [chunk_list] [funcid_list]")
          return
        if bench_mode==22 and argc > 17:
          print("Too many arguments")
          print("Usage funcid-split: nqueens -g nmin nmax block max_blocks log_level sort_mode preset_queens 22 [reorder_window_mult] [reorder_phase_jump] [cross_stripe_safe] [chunk_start] [chunk_count] [chunk_list] [split_group_list]")
          return
        if bench_mode==23 and argc > 18:
          print("Too many arguments")
          print("Usage funcid-depth: nqueens -g nmin nmax block max_blocks log_level sort_mode preset_queens 23 [reorder_window_mult] [reorder_phase_jump] [cross_stripe_safe] [chunk_start] [chunk_count] [chunk_list] [funcid_list] [bucket_list]")
          return
        if bench_mode==24 and argc > 18:
          print("Too many arguments")
          print("Usage funcid-mark: nqueens -g nmin nmax block max_blocks log_level sort_mode preset_queens 24 [reorder_window_mult] [reorder_phase_jump] [cross_stripe_safe] [chunk_start] [chunk_count] [chunk_list] [funcid_list] [mark_bucket_list]")
          return
        if bench_mode==25 and argc > 18:
          print("Too many arguments")
          print("Usage funcid-markdist: nqueens -g nmin nmax block max_blocks log_level sort_mode preset_queens 25 [reorder_window_mult] [reorder_phase_jump] [cross_stripe_safe] [chunk_start] [chunk_count] [chunk_list] [funcid_list] [axis_list]")
          return
        if bench_mode==30 and argc > 17:
          print("Too many arguments")
          print("Usage split145-generic-probe: nqueens -g nmin nmax block max_blocks log_level sort_mode preset_queens 30 [reorder_window_mult] [reorder_phase_jump] [cross_stripe_safe] [chunk_start] [chunk_count] [chunk_list] [broadmark_tail_variant]")
          return
      else:
        if bench_mode==29 or bench_mode==31:
          if argc > 16:
            print("Too many arguments")
            print("Usage broadmarktail/split145-generic worker: nqueens -g nmin nmax block max_blocks log_level sort_mode preset_queens mode[29|31] [reorder_window_mult] [reorder_phase_jump] [cross_stripe_safe] [worker_id] [worker_count] [broadmark_tail_variant]")
            return
        elif bench_mode==28:
          if argc > 14:
            print("Too many arguments")
            print("Usage broadmarktail sim/build: nqueens -g nmin nmax block max_blocks log_level sort_mode preset_queens 28 [reorder_window_mult] [reorder_phase_jump] [cross_stripe_safe] [broadmark_tail_variant]")
            return
        else:
          if argc > 13:
            print("Too many arguments")
            print("Usage reorder modes: nqueens -g nmin nmax block max_blocks log_level sort_mode preset_queens bench_mode[14|15|26|27|30|31] [reorder_window_mult] [reorder_phase_jump] [cross_stripe_safe]")
            return
    elif bench_mode==16:
      if argc > 10:
        print("Too many arguments")
        print("Usage sweep sim: nqueens -g nmin nmax block max_blocks log_level sort_mode preset_queens 16")
        return
    else:
      if argc >= 11:
        cross_stripe_safe=(int(sys.argv[10])!=0)
      if argc >= 12:
        debug_chunk_start=int(sys.argv[11])
      if argc >= 13:
        debug_chunk_count=int(sys.argv[12])
      if argc > 13:
        print("Too many arguments")
        print("Usage: nqueens [-c | -g] [nmin nmax] [gpu_block gpu_max_blocks log_level sort_mode] [preset_queens] [bench_mode] [cross_stripe_safe] [debug_chunk_start] [debug_chunk_count]")
        return
  else:
    print("Usage: nqueens [-c | -g] [nmin nmax] [gpu_block gpu_max_blocks log_level sort_mode] [preset_queens] [bench_mode] [cross_stripe_safe] [debug_chunk_start] [debug_chunk_count] [reorder_window_mult] [reorder_phase_jump]")
    return

  if bench_mode==20 and microbench_chunk_list_spec==MICROBENCH_DEFAULT_CHUNK_LIST:
    microbench_chunk_list_spec=FUNCID_TARGET_DEFAULT_CHUNK_LIST
  if bench_mode==21 and microbench_chunk_list_spec==MICROBENCH_DEFAULT_CHUNK_LIST:
    microbench_chunk_list_spec=FUNCID_SINGLE_DEFAULT_CHUNK_LIST
  if bench_mode==22 and microbench_chunk_list_spec==MICROBENCH_DEFAULT_CHUNK_LIST:
    microbench_chunk_list_spec=FUNCID_SPLIT_DEFAULT_CHUNK_LIST
  if bench_mode==23 and microbench_chunk_list_spec==MICROBENCH_DEFAULT_CHUNK_LIST:
    microbench_chunk_list_spec=FUNCID_DEPTH_DEFAULT_CHUNK_LIST
  if bench_mode==24 and microbench_chunk_list_spec==MICROBENCH_DEFAULT_CHUNK_LIST:
    microbench_chunk_list_spec=FUNCID_MARK_DEFAULT_CHUNK_LIST
  if bench_mode==24 and funcid_single_list_spec==FUNCID_SINGLE_DEFAULT_FUNCID_LIST:
    funcid_single_list_spec=FUNCID_MARK_DEFAULT_FUNCID_LIST
  if bench_mode==25 and microbench_chunk_list_spec==MICROBENCH_DEFAULT_CHUNK_LIST:
    microbench_chunk_list_spec=FUNCID_MARKDIST_DEFAULT_CHUNK_LIST
  if bench_mode==25 and funcid_single_list_spec==FUNCID_SINGLE_DEFAULT_FUNCID_LIST:
    funcid_single_list_spec=FUNCID_MARKDIST_DEFAULT_FUNCID_LIST
  if bench_mode==30 and microbench_chunk_list_spec==MICROBENCH_DEFAULT_CHUNK_LIST:
    microbench_chunk_list_spec=""

  if reorder_window_mult<=0:
    print(f"[warning] reorder_window_mult={reorder_window_mult} is invalid; using 8")
    reorder_window_mult=8
  if reorder_phase_jump<=0:
    print(f"[warning] reorder_phase_jump={reorder_phase_jump} is invalid; using 7")
    reorder_phase_jump=7
  FUNCID_REORDER_V2_WINDOW_MULT=reorder_window_mult
  FUNCID_REORDER_V2_PHASE_JUMP=reorder_phase_jump

  if broadmark_tail_variant<0 or broadmark_tail_variant>5:
    print(f"[warning] broadmark_tail_variant={broadmark_tail_variant} is invalid; using 2")
    broadmark_tail_variant=2
  BROAD_MARKDIST_TAIL_VARIANT=broadmark_tail_variant

  if worker_count<=0:
    print(f"[warning] worker_count={worker_count} is invalid; using 1")
    worker_count=1
  if worker_id<0:
    print(f"[warning] worker_id={worker_id} is invalid; using 0")
    worker_id=0
  if worker_id>=worker_count:
    print(f"[warning] worker_id={worker_id} must be < worker_count={worker_count}; using worker_id=0 worker_count=1")
    worker_id=0
    worker_count=1

  if use_gpu and (nmin<5 or nmax>28):
    print(f"[error] 184 no-sibling tailcall GPU candidate supports N=5..27 only; requested N={nmin}..{nmax-1}")
    return

  if bench_mode==10:
    DISABLE_CONSTELLATION_SIGNATURE_PRUNE=True
  else:
    DISABLE_CONSTELLATION_SIGNATURE_PRUNE=False

  if use_gpu and gpu_log_level>=1:
    print(f"version        : {VERSION_TAG}")
    print(f"cross_stripe_safe: {1 if cross_stripe_safe else 0}")
    if bench_mode==29 or bench_mode==31:
      print(f"worker_split : worker={worker_id}/{worker_count}")
    if bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31:
      print(f"broadmarktail_variant: id={BROAD_MARKDIST_TAIL_VARIANT} tag={broad_markdist_tail_variant_tag()} desc={broad_markdist_tail_variant_desc()}")
    if bench_mode==7:
      print(f"chunk_only    : start={debug_chunk_start} count={debug_chunk_count}")
    if bench_mode==8 or bench_mode==9 or bench_mode==10:
      print(f"boundary_debug: mode={bench_mode} preset={preset_queens_arg} signature_prune_disabled={1 if DISABLE_CONSTELLATION_SIGNATURE_PRUNE else 0}")
    if bench_mode==11:
      print(f"stream_bin_only: mode={bench_mode} preset={preset_queens_arg}")
    if bench_mode==13:
      print(f"stream_stats_only: mode={bench_mode} preset={preset_queens_arg}")
    if bench_mode==14:
      print(f"funcid_reorder_v2_sim: mode={bench_mode} preset={preset_queens_arg}")
    if bench_mode==15:
      print(f"funcid_reorder_v2_gpu: mode={bench_mode} preset={preset_queens_arg}")
    if bench_mode==16:
      print(f"funcid_reorder_v2_sweep_sim: mode={bench_mode} preset={preset_queens_arg}")
    if bench_mode==17:
      print(f"funcid_reorder_v2_microbench: mode={bench_mode} preset={preset_queens_arg} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec}")
    if bench_mode==18:
      print(f"funcid_reorder_v2_profile: mode={bench_mode} preset={preset_queens_arg} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec}")
    if bench_mode==19:
      print(f"funcid_reorder_v2_chunksize: mode={bench_mode} preset={preset_queens_arg} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec} factor_list={chunksize_factor_list_spec}")
    if bench_mode==20:
      print(f"funcid_reorder_v2_funcidtarget: mode={bench_mode} preset={preset_queens_arg} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec} group_list={funcid_target_group_list_spec}")
    if bench_mode==21:
      print(f"funcid_reorder_v2_funcidsingle: mode={bench_mode} preset={preset_queens_arg} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec} funcid_list={funcid_single_list_spec}")
    if bench_mode==22:
      print(f"funcid_reorder_v2_funcidsplit: mode={bench_mode} preset={preset_queens_arg} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec} split_group_list={funcid_split_group_list_spec}")
    if bench_mode==23:
      print(f"funcid_reorder_v2_funciddepth: mode={bench_mode} preset={preset_queens_arg} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec} funcid_list={funcid_single_list_spec} bucket_list={funcid_depth_bucket_list_spec}")
    if bench_mode==24:
      print(f"funcid_reorder_v2_funcidmark: mode={bench_mode} preset={preset_queens_arg} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec} funcid_list={funcid_single_list_spec} bucket_list={funcid_mark_bucket_list_spec}")
    if bench_mode==25:
      print(f"funcid_reorder_v2_funcidmarkdist: mode={bench_mode} preset={preset_queens_arg} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec} funcid_list={funcid_single_list_spec} axis_list={funcid_markdist_axis_list_spec}")
    if bench_mode==26:
      print(f"markdist_risk_reorder_sim: mode={bench_mode} preset={preset_queens_arg}")
    if bench_mode==27:
      print(f"markdist_risk_reorder_gpu: mode={bench_mode} preset={preset_queens_arg}")
    if bench_mode==28:
      print(f"broadmarktail_reorder_sim: mode={bench_mode} preset={preset_queens_arg}")
    if bench_mode==29:
      print(f"broadmarktail_reorder_gpu: mode={bench_mode} preset={preset_queens_arg}")
    if bench_mode==30:
      print(f"split232_cleanup_only_probe: mode={bench_mode} preset={preset_queens_arg} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec}")
    if bench_mode==31:
      print(f"split232_cleanup_only_full_gpu: mode={bench_mode} preset={preset_queens_arg}")
  if gpu_log_level>=1 and (bench_mode==14 or bench_mode==15 or bench_mode==16 or bench_mode==17 or bench_mode==18 or bench_mode==19 or bench_mode==20 or bench_mode==21 or bench_mode==22 or bench_mode==23 or bench_mode==24 or bench_mode==25 or bench_mode==26 or bench_mode==27 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31):
    print(f"funcid_reorder_v2_params: window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} param={funcid_reorder_param_tag()} reason={FUNCID_REORDER_V2_DEFAULT_REASON}")
  if gpu_log_level>=1 and (bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31):
    print(f"broadmarktail_params: version={BROAD_MARKDIST_TAIL_REORDER_VERSION} variant={BROAD_MARKDIST_TAIL_VARIANT} tag={broad_markdist_tail_variant_tag()} window_boost={broad_markdist_tail_window_boost_value()} phase_mix={1 if broad_markdist_tail_use_phase_mix() else 0} rotate_interleave={1 if broad_markdist_tail_use_rotating_interleave() else 0} phase_salt={broad_markdist_tail_phase_salt_value()} reason={BROAD_MARKDIST_TAIL_REORDER_DEFAULT_REASON}")
  if gpu_log_level>=1 and (bench_mode==30 or bench_mode==31):
    print(f"chunkshape148_params: version={CHUNKSHAPE148_REORDER_VERSION} reason={CHUNKSHAPE148_DEFAULT_REASON}")
  print(" N:             Total           Unique         hh:mm:ss.ms")
  for N in range(nmin,nmax):
    override_elapsed_text:str=""
    start_time=datetime.now()
    if N<=5:

      total=_bit_total(N)

      dt=datetime.now()-start_time
      text=str(dt)[:-3]
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}")
      continue

    ijkl_list:Set[int]=set()
    constellations:List[Dict[str,int]]=[]
    subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]]=set()

    use_constellation_cache:bool = False

    preset_queens:int = preset_queens_arg # preset_queens CPUが担当する深さ
    preset_queens=select_dynamic_preset_queens(N,preset_queens)

    if gpu_log_level>=1:
      print(f"[dynamic-preset] N={N} preset_queens={preset_queens}")

    if bench_mode==11:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[stream-cache-only] N={N} preset_queens={preset_queens} records={stream_records} bin={stream_fname} valid={1 if validate_bin_file(stream_fname) else 0}")
      print(f"{N:2d}:{0:18d}{0:17d}{text:>21s}    stream-cache-only")
      continue

    if bench_mode==13:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      stats_chunks:int=exec_solutions_gpu_bin_stream_stats_only(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[stream-stats-only] N={N} preset_queens={preset_queens} records={stream_records} chunks={stats_chunks} bin={stream_fname} valid={1 if validate_bin_file(stream_fname) else 0}")
      print(f"{N:2d}:{0:18d}{0:17d}{text:>21s}    stream-stats-only")
      continue

    if bench_mode==14:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname,reorder_records,reorder_chunks=build_funcid_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[funcid-reorder-v2-sim-only] N={N} preset_queens={preset_queens} records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} valid={1 if validate_bin_file(reorder_fname) else 0}")
      print(f"{N:2d}:{0:18d}{0:17d}{text:>21s}    funcid-reorder-v2-sim-only")
      continue

    if bench_mode==26:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname,reorder_records,reorder_chunks=build_funcid_markdist_risk_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[markdist-risk-reorder-sim-only] N={N} preset_queens={preset_queens} records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} valid={1 if validate_bin_file(reorder_fname) else 0}")
      print(f"{N:2d}:{0:18d}{0:17d}{text:>21s}    markdist-risk-reorder-sim-only")
      continue

    if use_gpu and N>=21 and bench_mode==27:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname:str=funcid_markdist_risk_reorder_output_fname(N,preset_queens)
      reorder_records:int=count_constellations_bin_records(reorder_fname)
      steps_for_count:int=gpu_block*gpu_max_blocks
      if steps_for_count<=0:
        steps_for_count=15488
      reorder_chunks:int=0
      if reorder_records>0:
        reorder_chunks=(reorder_records + steps_for_count - 1)//steps_for_count
      done_count:int=read_stream_done_count(reorder_fname+".done")
      if reorder_records==stream_records and done_count==stream_records and validate_bin_file(reorder_fname):
        if gpu_log_level>=1:
          print(f"[markdist-risk-reorder-gpu-reuse] N={N} records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)}")
      else:
        if gpu_log_level>=1:
          print(f"[markdist-risk-reorder-gpu-build] N={N} stream_records={stream_records} existing_records={reorder_records} done_count={done_count} bin={reorder_fname}")
        reorder_fname,reorder_records,reorder_chunks=build_funcid_markdist_risk_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      total:int=exec_solutions_gpu_bin_stream_funcid_reorder(N,reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,False,debug_chunk_start,debug_chunk_count,"","markdistrisk_gpu")
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      status:str="ok" if expected[N]==total else f"ng({total}!={expected[N]})"
      print(f"[markdist-risk-reorder-gpu-done] N={N} source_records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP}")
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    {status}")
      continue

    if bench_mode==28:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname,reorder_records,reorder_chunks=build_broad_markdist_tail_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[broadmarktail-reorder-sim-only] N={N} preset_queens={preset_queens} records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} valid={1 if validate_bin_file(reorder_fname) else 0}")
      print(f"{N:2d}:{0:18d}{0:17d}{text:>21s}    broadmarktail-reorder-sim-only")
      continue

    if use_gpu and N>=21 and bench_mode==29:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname:str=broad_markdist_tail_reorder_output_fname(N,preset_queens,gpu_block,gpu_max_blocks)
      reorder_records:int=count_constellations_bin_records(reorder_fname)
      steps_for_count:int=gpu_block*gpu_max_blocks
      if steps_for_count<=0:
        steps_for_count=15488
      reorder_chunks:int=0
      if reorder_records>0:
        reorder_chunks=(reorder_records + steps_for_count - 1)//steps_for_count
      done_count:int=read_stream_done_count(reorder_fname+".done")
      if reorder_records==stream_records and done_count==stream_records and validate_bin_file(reorder_fname):
        if gpu_log_level>=1:
          print(f"[broadmarktail-reorder-gpu-reuse] N={N} records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)}")
      else:
        if gpu_log_level>=1:
          print(f"[broadmarktail-reorder-gpu-build] N={N} stream_records={stream_records} existing_records={reorder_records} done_count={done_count} bin={reorder_fname}")
        reorder_fname,reorder_records,reorder_chunks=build_broad_markdist_tail_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      progress_suffix:str=f"broadmarktail_{BROAD_MARKDIST_TAIL_REORDER_VERSION}_{broad_markdist_tail_variant_tag()}_gpu"
      total:int=exec_solutions_gpu_bin_stream_funcid_reorder(N,reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,False,debug_chunk_start,debug_chunk_count,"",progress_suffix,worker_id,worker_count)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      status:str="ok" if expected[N]==total else f"ng({total}!={expected[N]})"
      if worker_count>1:
        status=f"partial-worker-{worker_id}-of-{worker_count}"
      if gpu_log_level>=1:
        print(f"[broadmarktail-reorder-gpu-done] N={N} source_records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} variant={broad_markdist_tail_variant_tag()} worker={worker_id}/{worker_count} total={total}")
        if worker_count>1:
          print(f"[worker-done] N={N} worker={worker_id}/{worker_count} partial_total={total} expected_total={expected[N]}")
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    {status}")
      continue

    if use_gpu and N>=21 and bench_mode==30:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname:str=broad_markdist_tail_reorder_output_fname(N,preset_queens,gpu_block,gpu_max_blocks)
      reorder_records:int=count_constellations_bin_records(reorder_fname)
      steps_for_count:int=gpu_block*gpu_max_blocks
      if steps_for_count<=0:
        steps_for_count=15488
      reorder_chunks:int=0
      if reorder_records>0:
        reorder_chunks=(reorder_records + steps_for_count - 1)//steps_for_count
      done_count:int=read_stream_done_count(reorder_fname+".done")
      if reorder_records==stream_records and done_count==stream_records and validate_bin_file(reorder_fname):
        if gpu_log_level>=1:
          print(f"[split232-probe-reuse] N={N} records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)}")
      else:
        if gpu_log_level>=1:
          print(f"[split232-probe-build] N={N} stream_records={stream_records} existing_records={reorder_records} done_count={done_count} bin={reorder_fname}")
        reorder_fname,reorder_records,reorder_chunks=build_broad_markdist_tail_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      base_reorder_fname:str=reorder_fname
      base_reorder_records:int=reorder_records
      base_reorder_chunks:int=reorder_chunks
      shaped_fname:str=chunkshape148_reorder_output_fname(N,preset_queens,gpu_block,gpu_max_blocks)
      shaped_records:int=count_constellations_bin_records(shaped_fname)
      shaped_chunks:int=0
      if shaped_records>0:
        shaped_chunks=(shaped_records+steps_for_count-1)//steps_for_count
      shaped_done:int=read_stream_done_count(shaped_fname+".done")
      if shaped_records==base_reorder_records and shaped_done==base_reorder_records and validate_bin_file(shaped_fname):
        if gpu_log_level>=1:
          print(f"[chunkshape148-probe-reuse] N={N} records={shaped_records} chunks={shaped_chunks} bin={shaped_fname} source_bin={base_reorder_fname}")
      else:
        if gpu_log_level>=1:
          print(f"[chunkshape148-probe-build] N={N} source_records={base_reorder_records} existing_records={shaped_records} done_count={shaped_done} bin={shaped_fname}")
        shaped_fname,shaped_records,shaped_chunks=build_chunkshape148_reordered_bin(N,base_reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level)
      reorder_fname=shaped_fname
      reorder_records=shaped_records
      reorder_chunks=shaped_chunks
      progress_suffix:str=f"split232_probe_{CHUNKSHAPE148_REORDER_VERSION}_{BROAD_MARKDIST_TAIL_REORDER_VERSION}_{broad_markdist_tail_variant_tag()}"
      total:int=exec_solutions_gpu_bin_stream_split145(N,reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,True,debug_chunk_start,debug_chunk_count,microbench_chunk_list_spec,progress_suffix,0,1,0)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[split232-probe-done] N={N} source_records={stream_records} base_reordered_records={base_reorder_records} shaped_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} base_bin={base_reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)} chunkshape={CHUNKSHAPE148_REORDER_VERSION} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} variant={broad_markdist_tail_variant_tag()} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec} partial_total={total}")
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    split232-probe")
      continue

    if use_gpu and N>=21 and bench_mode==31:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname:str=broad_markdist_tail_reorder_output_fname(N,preset_queens,gpu_block,gpu_max_blocks)
      reorder_records:int=count_constellations_bin_records(reorder_fname)
      steps_for_count:int=gpu_block*gpu_max_blocks
      if steps_for_count<=0:
        steps_for_count=15488
      reorder_chunks:int=0
      if reorder_records>0:
        reorder_chunks=(reorder_records + steps_for_count - 1)//steps_for_count
      done_count:int=read_stream_done_count(reorder_fname+".done")
      if reorder_records==stream_records and done_count==stream_records and validate_bin_file(reorder_fname):
        if gpu_log_level>=1:
          print(f"[split232-full-reuse] N={N} records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)}")
      else:
        if gpu_log_level>=1:
          print(f"[split232-full-build] N={N} stream_records={stream_records} existing_records={reorder_records} done_count={done_count} bin={reorder_fname}")
        reorder_fname,reorder_records,reorder_chunks=build_broad_markdist_tail_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      base_reorder_fname:str=reorder_fname
      base_reorder_records:int=reorder_records
      base_reorder_chunks:int=reorder_chunks
      shaped_fname:str=chunkshape148_reorder_output_fname(N,preset_queens,gpu_block,gpu_max_blocks)
      shaped_records:int=count_constellations_bin_records(shaped_fname)
      shaped_chunks:int=0
      if shaped_records>0:
        shaped_chunks=(shaped_records+steps_for_count-1)//steps_for_count
      shaped_done:int=read_stream_done_count(shaped_fname+".done")
      if shaped_records==base_reorder_records and shaped_done==base_reorder_records and validate_bin_file(shaped_fname):
        if gpu_log_level>=1:
          print(f"[chunkshape148-full-reuse] N={N} records={shaped_records} chunks={shaped_chunks} bin={shaped_fname} source_bin={base_reorder_fname}")
      else:
        if gpu_log_level>=1:
          print(f"[chunkshape148-full-build] N={N} source_records={base_reorder_records} existing_records={shaped_records} done_count={shaped_done} bin={shaped_fname}")
        shaped_fname,shaped_records,shaped_chunks=build_chunkshape148_reordered_bin(N,base_reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level)
      reorder_fname=shaped_fname
      reorder_records=shaped_records
      reorder_chunks=shaped_chunks
      progress_suffix:str=f"split232_full_{CHUNKSHAPE148_REORDER_VERSION}_{BROAD_MARKDIST_TAIL_REORDER_VERSION}_{broad_markdist_tail_variant_tag()}"
      total:int=exec_solutions_gpu_bin_stream_split145(N,reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,False,debug_chunk_start,debug_chunk_count,"",progress_suffix,worker_id,worker_count,0)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      status:str="ok" if expected[N]==total else f"ng({total}!={expected[N]})"
      if worker_count>1:
        status=f"partial-worker-{worker_id}-of-{worker_count}"
      if gpu_log_level>=1:
        print(f"[split232-full-done] N={N} source_records={stream_records} base_reordered_records={base_reorder_records} shaped_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} base_bin={base_reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)} chunkshape={CHUNKSHAPE148_REORDER_VERSION} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} variant={broad_markdist_tail_variant_tag()} worker={worker_id}/{worker_count} total={total}")
        if worker_count>1:
          print(f"[worker-done] N={N} worker={worker_id}/{worker_count} partial_total={total} expected_total={expected[N]}")
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    {status}")
      continue

    if bench_mode==16:
      FUNCID_REORDER_V2_SWEEP_TEMP_OUTPUT=True
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      sweep_windows:List[int]=[8,16,32]
      sweep_phases:List[int]=[5,7,11]
      sweep_count:int=0
      wi:int=0
      while wi<len(sweep_windows):
        pj_i:int=0
        while pj_i<len(sweep_phases):
          FUNCID_REORDER_V2_WINDOW_MULT=sweep_windows[wi]
          FUNCID_REORDER_V2_PHASE_JUMP=sweep_phases[pj_i]
          one_t0=datetime.now()
          reorder_fname,reorder_records,reorder_chunks=build_funcid_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
          one_elapsed=datetime.now()-one_t0
          one_text=str(one_elapsed)[:-3]
          print(f"[funcid-reorder-v2-sweep-sim] N={N} preset_queens={preset_queens} records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} temporary_bin=1 param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} valid={1 if validate_bin_file(reorder_fname) else 0} elapsed={one_text}")
          sweep_count+=1
          pj_i+=1
        wi+=1
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[funcid-reorder-v2-sweep-summary] N={N} preset_queens={preset_queens} records={stream_records} cases={sweep_count} elapsed={text}")
      print(f"{N:2d}:{0:18d}{0:17d}{text:>21s}    funcid-reorder-v2-sweep-sim")
      continue

    if use_gpu and N>=21 and bench_mode==17:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname:str=funcid_reorder_output_fname(N,preset_queens)
      reorder_records:int=count_constellations_bin_records(reorder_fname)
      steps_for_count:int=gpu_block*gpu_max_blocks
      if steps_for_count<=0:
        steps_for_count=15488
      reorder_chunks:int=0
      if reorder_records>0:
        reorder_chunks=(reorder_records + steps_for_count - 1)//steps_for_count
      done_count:int=read_stream_done_count(reorder_fname+".done")
      if reorder_records==stream_records and done_count==stream_records and validate_bin_file(reorder_fname):
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-microbench-reuse] N={N} records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)}")
      else:
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-microbench-build] N={N} stream_records={stream_records} existing_records={reorder_records} done_count={done_count} bin={reorder_fname}")
        reorder_fname,reorder_records,reorder_chunks=build_funcid_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      total:int=exec_solutions_gpu_bin_stream_funcid_reorder(N,reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,True,debug_chunk_start,debug_chunk_count,microbench_chunk_list_spec,"microbench")
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[funcid-reorder-v2-microbench-done] N={N} source_records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec} partial_total={total}")
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    funcid-reorder-v2-microbench")
      continue

    if use_gpu and N>=21 and bench_mode==18:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname:str=funcid_reorder_output_fname(N,preset_queens)
      reorder_records:int=count_constellations_bin_records(reorder_fname)
      steps_for_count:int=gpu_block*gpu_max_blocks
      if steps_for_count<=0:
        steps_for_count=15488
      reorder_chunks:int=0
      if reorder_records>0:
        reorder_chunks=(reorder_records + steps_for_count - 1)//steps_for_count
      done_count:int=read_stream_done_count(reorder_fname+".done")
      if reorder_records==stream_records and done_count==stream_records and validate_bin_file(reorder_fname):
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-profile-reuse] N={N} records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)}")
      else:
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-profile-build] N={N} stream_records={stream_records} existing_records={reorder_records} done_count={done_count} bin={reorder_fname}")
        reorder_fname,reorder_records,reorder_chunks=build_funcid_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      total:int=exec_solutions_gpu_bin_stream_funcid_reorder_profile(N,reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,True,debug_chunk_start,debug_chunk_count,microbench_chunk_list_spec,"profile")
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[funcid-reorder-v2-profile-done] N={N} source_records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec} partial_total={total}")
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    funcid-reorder-v2-profile")
      continue

    if use_gpu and N>=21 and bench_mode==19:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname:str=funcid_reorder_output_fname(N,preset_queens)
      reorder_records:int=count_constellations_bin_records(reorder_fname)
      steps_for_count:int=gpu_block*gpu_max_blocks
      if steps_for_count<=0:
        steps_for_count=15488
      reorder_chunks:int=0
      if reorder_records>0:
        reorder_chunks=(reorder_records + steps_for_count - 1)//steps_for_count
      done_count:int=read_stream_done_count(reorder_fname+".done")
      if reorder_records==stream_records and done_count==stream_records and validate_bin_file(reorder_fname):
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-chunksize-reuse] N={N} records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)}")
      else:
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-chunksize-build] N={N} stream_records={stream_records} existing_records={reorder_records} done_count={done_count} bin={reorder_fname}")
        reorder_fname,reorder_records,reorder_chunks=build_funcid_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      total:int=exec_solutions_gpu_bin_stream_funcid_reorder_chunksize_profile(N,reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,debug_chunk_start,debug_chunk_count,microbench_chunk_list_spec,chunksize_factor_list_spec,"chunksize")
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[funcid-reorder-v2-chunksize-done] N={N} source_records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec} factor_list={chunksize_factor_list_spec} partial_total={total}")
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    funcid-reorder-v2-chunksize")
      continue

    if use_gpu and N>=21 and bench_mode==23:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname:str=funcid_reorder_output_fname(N,preset_queens)
      reorder_records:int=count_constellations_bin_records(reorder_fname)
      steps_for_count:int=gpu_block*gpu_max_blocks
      if steps_for_count<=0:
        steps_for_count=15488
      reorder_chunks:int=0
      if reorder_records>0:
        reorder_chunks=(reorder_records + steps_for_count - 1)//steps_for_count
      done_count:int=read_stream_done_count(reorder_fname+".done")
      if reorder_records==stream_records and done_count==stream_records and validate_bin_file(reorder_fname):
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-funciddepth-reuse] N={N} records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)}")
      else:
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-funciddepth-build] N={N} stream_records={stream_records} existing_records={reorder_records} done_count={done_count} bin={reorder_fname}")
        reorder_fname,reorder_records,reorder_chunks=build_funcid_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      total:int=exec_solutions_gpu_bin_stream_funcid_reorder_funcid_depth_profile(N,reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,debug_chunk_start,debug_chunk_count,microbench_chunk_list_spec,funcid_single_list_spec,funcid_depth_bucket_list_spec,"funciddepth")
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[funcid-reorder-v2-funciddepth-done] N={N} source_records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec} funcid_list={funcid_single_list_spec} bucket_list={funcid_depth_bucket_list_spec} partial_total={total}")
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    funcid-reorder-v2-funciddepth")
      continue

    if use_gpu and N>=21 and bench_mode==24:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname:str=funcid_reorder_output_fname(N,preset_queens)
      reorder_records:int=count_constellations_bin_records(reorder_fname)
      steps_for_count:int=gpu_block*gpu_max_blocks
      if steps_for_count<=0:
        steps_for_count=15488
      reorder_chunks:int=0
      if reorder_records>0:
        reorder_chunks=(reorder_records + steps_for_count - 1)//steps_for_count
      done_count:int=read_stream_done_count(reorder_fname+".done")
      if reorder_records==stream_records and done_count==stream_records and validate_bin_file(reorder_fname):
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-funcidmark-reuse] N={N} records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)}")
      else:
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-funcidmark-build] N={N} stream_records={stream_records} existing_records={reorder_records} done_count={done_count} bin={reorder_fname}")
        reorder_fname,reorder_records,reorder_chunks=build_funcid_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      total:int=exec_solutions_gpu_bin_stream_funcid_reorder_funcid_mark_profile(N,reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,debug_chunk_start,debug_chunk_count,microbench_chunk_list_spec,funcid_single_list_spec,funcid_mark_bucket_list_spec,"funcidmark")
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[funcid-reorder-v2-funcidmark-done] N={N} source_records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec} funcid_list={funcid_single_list_spec} bucket_list={funcid_mark_bucket_list_spec} partial_total={total}")
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    funcid-reorder-v2-funcidmark")
      continue

    if use_gpu and N>=21 and bench_mode==25:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname:str=funcid_reorder_output_fname(N,preset_queens)
      reorder_records:int=count_constellations_bin_records(reorder_fname)
      steps_for_count:int=gpu_block*gpu_max_blocks
      if steps_for_count<=0:
        steps_for_count=15488
      reorder_chunks:int=0
      if reorder_records>0:
        reorder_chunks=(reorder_records + steps_for_count - 1)//steps_for_count
      done_count:int=read_stream_done_count(reorder_fname+".done")
      if reorder_records==stream_records and done_count==stream_records and validate_bin_file(reorder_fname):
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-funcidmarkdist-reuse] N={N} records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)}")
      else:
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-funcidmarkdist-build] N={N} stream_records={stream_records} existing_records={reorder_records} done_count={done_count} bin={reorder_fname}")
        reorder_fname,reorder_records,reorder_chunks=build_funcid_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      total:int=exec_solutions_gpu_bin_stream_funcid_reorder_funcid_markdist_profile(N,reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,debug_chunk_start,debug_chunk_count,microbench_chunk_list_spec,funcid_single_list_spec,funcid_markdist_axis_list_spec,"funcidmarkdist")
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[funcid-reorder-v2-funcidmarkdist-done] N={N} source_records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec} funcid_list={funcid_single_list_spec} axis_list={funcid_markdist_axis_list_spec} partial_total={total}")
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    funcid-reorder-v2-funcidmarkdist")
      continue

    if use_gpu and N>=21 and bench_mode==22:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname:str=funcid_reorder_output_fname(N,preset_queens)
      reorder_records:int=count_constellations_bin_records(reorder_fname)
      steps_for_count:int=gpu_block*gpu_max_blocks
      if steps_for_count<=0:
        steps_for_count=15488
      reorder_chunks:int=0
      if reorder_records>0:
        reorder_chunks=(reorder_records + steps_for_count - 1)//steps_for_count
      done_count:int=read_stream_done_count(reorder_fname+".done")
      if reorder_records==stream_records and done_count==stream_records and validate_bin_file(reorder_fname):
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-funcidsplit-reuse] N={N} records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)}")
      else:
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-funcidsplit-build] N={N} stream_records={stream_records} existing_records={reorder_records} done_count={done_count} bin={reorder_fname}")
        reorder_fname,reorder_records,reorder_chunks=build_funcid_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      total:int=exec_solutions_gpu_bin_stream_funcid_reorder_funcid_split_profile(N,reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,debug_chunk_start,debug_chunk_count,microbench_chunk_list_spec,funcid_split_group_list_spec,"funcidsplit")
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[funcid-reorder-v2-funcidsplit-done] N={N} source_records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec} split_group_list={funcid_split_group_list_spec} partial_total={total}")
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    funcid-reorder-v2-funcidsplit")
      continue

    if use_gpu and N>=21 and bench_mode==21:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname:str=funcid_reorder_output_fname(N,preset_queens)
      reorder_records:int=count_constellations_bin_records(reorder_fname)
      steps_for_count:int=gpu_block*gpu_max_blocks
      if steps_for_count<=0:
        steps_for_count=15488
      reorder_chunks:int=0
      if reorder_records>0:
        reorder_chunks=(reorder_records + steps_for_count - 1)//steps_for_count
      done_count:int=read_stream_done_count(reorder_fname+".done")
      if reorder_records==stream_records and done_count==stream_records and validate_bin_file(reorder_fname):
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-funcidsingle-reuse] N={N} records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)}")
      else:
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-funcidsingle-build] N={N} stream_records={stream_records} existing_records={reorder_records} done_count={done_count} bin={reorder_fname}")
        reorder_fname,reorder_records,reorder_chunks=build_funcid_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      total:int=exec_solutions_gpu_bin_stream_funcid_reorder_funcid_single_profile(N,reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,debug_chunk_start,debug_chunk_count,microbench_chunk_list_spec,funcid_single_list_spec,"funcidsingle")
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[funcid-reorder-v2-funcidsingle-done] N={N} source_records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec} funcid_list={funcid_single_list_spec} partial_total={total}")
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    funcid-reorder-v2-funcidsingle")
      continue

    if use_gpu and N>=21 and bench_mode==20:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname:str=funcid_reorder_output_fname(N,preset_queens)
      reorder_records:int=count_constellations_bin_records(reorder_fname)
      steps_for_count:int=gpu_block*gpu_max_blocks
      if steps_for_count<=0:
        steps_for_count=15488
      reorder_chunks:int=0
      if reorder_records>0:
        reorder_chunks=(reorder_records + steps_for_count - 1)//steps_for_count
      done_count:int=read_stream_done_count(reorder_fname+".done")
      if reorder_records==stream_records and done_count==stream_records and validate_bin_file(reorder_fname):
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-funcidtarget-reuse] N={N} records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)}")
      else:
        if gpu_log_level>=1:
          print(f"[funcid-reorder-v2-funcidtarget-build] N={N} stream_records={stream_records} existing_records={reorder_records} done_count={done_count} bin={reorder_fname}")
        reorder_fname,reorder_records,reorder_chunks=build_funcid_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      total:int=exec_solutions_gpu_bin_stream_funcid_reorder_funcid_target_profile(N,reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,debug_chunk_start,debug_chunk_count,microbench_chunk_list_spec,funcid_target_group_list_spec,"funcidtarget")
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[funcid-reorder-v2-funcidtarget-done] N={N} source_records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={microbench_chunk_list_spec} group_list={funcid_target_group_list_spec} partial_total={total}")
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    funcid-reorder-v2-funcidtarget")
      continue

    if use_gpu and N>=21 and bench_mode==15:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname,reorder_records,reorder_chunks=build_funcid_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      total:int=exec_solutions_gpu_bin_stream_funcid_reorder(N,reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,False,debug_chunk_start,debug_chunk_count)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      status:str="ok" if expected[N]==total else f"ng({total}!={expected[N]})"
      print(f"[funcid-reorder-v2-gpu-done] N={N} source_records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP}")
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    {status}")
      continue

    if use_gpu and N>=21 and not (bench_mode==8 or bench_mode==9 or bench_mode==10 or bench_mode==14 or bench_mode==15 or bench_mode==16 or bench_mode==17 or bench_mode==18 or bench_mode==19 or bench_mode==20 or bench_mode==21 or bench_mode==22 or bench_mode==23 or bench_mode==24 or bench_mode==25 or bench_mode==26 or bench_mode==27 or bench_mode==30 or bench_mode==31):
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      stream_chunk_only:bool=(bench_mode==7)
      total:int=exec_solutions_gpu_bin_stream(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,stream_chunk_only,debug_chunk_start,debug_chunk_count)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      status:str="ok" if expected[N]==total else f"ng({total}!={expected[N]})"
      if stream_chunk_only:
        status="stream-chunk-only"
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    {status}")
      continue

    if use_constellation_cache:
      ijkl_list,subconst_cache,constellations,preset_queens= build_constellations_dynamicK(N,ijkl_list,subconst_cache,constellations, use_gpu,preset_queens)
    else:
      ijkl_list,subconst_cache,constellations,preset_queens=gen_constellations(N,ijkl_list,subconst_cache,constellations,preset_queens)

    if bench_mode==8 or bench_mode==10:
      print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)} signature_prune_disabled={1 if DISABLE_CONSTELLATION_SIGNATURE_PRUNE else 0}")
      diagnose_boundary_classification(N,constellations)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      status:str="boundary-only"
      if bench_mode==10:
        status="boundary-only-nosig"
      print(f"{N:2d}:{0:18d}{0:17d}{text:>21s}    {status}")
      continue

    use_solution_cache = False
    if use_solution_cache:
        load_or_build_solutions_bin(N,constellations, preset_queens, use_gpu, cache_tag="v2")
    else:
        if bench_mode==6 and use_gpu:
          if gpu_log_level>=1:
            print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)}")
            print(f"[reorder-only] mode=6 validates launch-order permutation only; GPU kernel is not executed")
          exec_solutions(N,constellations,use_gpu,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,True,True)
          override_elapsed_text="reorder-only"
        elif bench_mode==7 and use_gpu:
          if gpu_log_level>=1:
            print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)}")
            print(f"[chunk-only] mode=7 executes selected chunks only; GPU kernel runs only for the requested range")
          exec_solutions(N,constellations,use_gpu,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,False,True,debug_chunk_start,debug_chunk_count)
          override_elapsed_text="chunk-only"
        elif bench_mode==1 and use_gpu and N==20:
          if gpu_log_level>=1:
            print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)}")
            print(f"[bench-warmup] mode=1 first run is warmup; second run is measured")
          warm_start=datetime.now()
          exec_solutions(N,constellations,use_gpu,gpu_block,gpu_max_blocks,0,gpu_sort_mode,cross_stripe_safe)
          warm_total:int=sum(c['solutions'] for c in constellations if c['solutions']>0)
          warm_elapsed=datetime.now()-warm_start
          warm_text=str(warm_elapsed)[:-3]
          warm_status:str="ok" if expected[N]==warm_total else f"ng({warm_total}!={expected[N]})"
          print(f"[warmup] N={N} total={warm_total} elapsed={warm_text} {warm_status}")
          for c in constellations:
            c["solutions"]=0
          start_time=datetime.now()
          if gpu_log_level>=1:
            print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)}")
          exec_solutions(N,constellations,use_gpu,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe)
        elif bench_mode==5 and use_gpu and N==20:
          if gpu_log_level>=1:
            print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)}")
            print(f"[bench-repeat2] mode=5 run N=20 twice in the same process; run 2 is measured")
          for run_no in range(1,3):
            for c in constellations:
              c["solutions"]=0
            run_t0=datetime.now()
            run_log:int=gpu_log_level if run_no==2 else 0
            exec_solutions(N,constellations,use_gpu,gpu_block,gpu_max_blocks,run_log,gpu_sort_mode,cross_stripe_safe)
            run_total:int=sum(c['solutions'] for c in constellations if c['solutions']>0)
            run_elapsed=datetime.now()-run_t0
            run_text=str(run_elapsed)[:-3]
            if run_no==2:
              override_elapsed_text=run_text
            run_status:str="ok" if expected[N]==run_total else f"ng({run_total}!={expected[N]})"
            print(f"[repeat2] N={N} run={run_no} total={run_total} elapsed={run_text} {run_status}")
            if run_no<2:
              for c in constellations:
                c["solutions"]=0
        elif bench_mode==4 and use_gpu and N==20:
          if gpu_log_level>=1:
            print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)}")
            print(f"[bench-repeat] mode=4 run N=20 three times in the same process; run 3 is measured")
          for run_no in range(1,4):
            for c in constellations:
              c["solutions"]=0
            run_t0=datetime.now()
            run_log:int=gpu_log_level if run_no==3 else 0
            exec_solutions(N,constellations,use_gpu,gpu_block,gpu_max_blocks,run_log,gpu_sort_mode,cross_stripe_safe)
            run_total:int=sum(c['solutions'] for c in constellations if c['solutions']>0)
            run_elapsed=datetime.now()-run_t0
            run_text=str(run_elapsed)[:-3]
            if run_no==3:
              override_elapsed_text=run_text
            run_status:str="ok" if expected[N]==run_total else f"ng({run_total}!={expected[N]})"
            print(f"[repeat] N={N} run={run_no} total={run_total} elapsed={run_text} {run_status}")
            if run_no<3:
              for c in constellations:
                c["solutions"]=0
        elif (bench_mode==2 or bench_mode==3) and use_gpu and N==20:
          if gpu_log_level>=1:
            print(f"[bench-preheat] mode={bench_mode} preheat before measured N=20")
          pre_start_N:int=19
          if bench_mode==3:
            pre_start_N=18
          for PN in range(pre_start_N,20):
            pre_ijkl_list:Set[int]=set()
            pre_constellations:List[Dict[str,int]]=[]
            pre_subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]]=set()
            pre_preset_queens:int=5
            pre_t0=datetime.now()
            pre_ijkl_list,pre_subconst_cache,pre_constellations,pre_preset_queens=gen_constellations(PN,pre_ijkl_list,pre_subconst_cache,pre_constellations,pre_preset_queens)
            exec_solutions(PN,pre_constellations,use_gpu,gpu_block,gpu_max_blocks,0,gpu_sort_mode,cross_stripe_safe)
            pre_total:int=sum(c['solutions'] for c in pre_constellations if c['solutions']>0)
            pre_elapsed=datetime.now()-pre_t0
            pre_text=str(pre_elapsed)[:-3]
            pre_status:str="ok" if expected[PN]==pre_total else f"ng({pre_total}!={expected[PN]})"
            print(f"[preheat] N={PN} total={pre_total} elapsed={pre_text} {pre_status}")
          start_time=datetime.now()
          if gpu_log_level>=1:
            print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)}")
          exec_solutions(N,constellations,use_gpu,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe)
        else:
          if gpu_log_level>=1:
            print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)}")
          exec_solutions(N,constellations,use_gpu,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe)

    if bench_mode==9:
      if use_gpu:
        print("[bc-sol-warning] bench_mode=9 is intended for CPU. GPU direct_total stores only constellations[0].")
      else:
        diagnose_solution_by_boundary(N,constellations)

    total:int=sum(c['solutions'] for c in constellations if c['solutions']>0)
    time_elapsed=datetime.now()-start_time
    text=str(time_elapsed)[:-3]
    if override_elapsed_text != "":
      text=override_elapsed_text
    status:str="ok" if expected[N]==total else f"ng({total}!={expected[N]})"
    if bench_mode==6 and use_gpu:
      status="reorder-only"
    if bench_mode==7 and use_gpu:
      status="chunk-only"
    print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    {status}")

if __name__=="__main__":
  main()
