#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# NQ_UPDATE_MEMO
# 233 r3: fasttrim-inline main-entry fix。232 cleanup-onlyを親に、MAXD14 CUDA kernelは変更せず、N=21 split145 full実行のhost側hot pathを整理する。
#      build_soa内の小helper呼び出し、旧split145 chunk worker経由、per-record read helper呼び出しを本線へinlineし、旧診断関数を削減する。
#      cudacodon側の旧source混入検出に備え、旧chunk helper名のactive定義/active呼び出しとmain entryを静的検査する。
#      runtime tagはsplit233。futuremask、no-sibling、rootrestlate、root one/two preroll、host task order、MAXD14 208 bytes/threadを維持する.
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
ご指摘の方向で、**233Py fasttrim-inline** を作成しました。
232ログは正当性OKで、速度も良好でした。

```text
232Py N=21 full once
final total              314666222712 OK
progress rows            131 OK
duplicate/missing        0 / 0 OK
dispatch task sum        2025282 OK
required_maxd            全chunk 14
selected_MAXD            全chunk 14
schedule_words           0
stack_bytes_per_thread   208
warning/error            0 / 0
elapsed                  0:07:07.733
```

233Pyでは、**GPU kernel本文は変更せず**、ご提案どおり host 側の本線で小さな関数呼び出しを減らす方向にしました。

```text
232 source : 9432行 / 365682 bytes / 228 defs
233 source : 4104行 / 133920 bytes / 116 defs

削減:
  5328行
  231762 bytes
  112 defs
```

主な変更です。

```text
1. build_soa_for_range() 内の getj/getk/getl 呼び出しを直接bit展開へ変更
2. build_soa_for_range() 内の symmetry()/symmetry90() 呼び出しを @par loop 内へ直接展開
3. exec_solutions_gpu_chunk_split145() を削除し、split145 stream loop内へ同梱
4. split145 stream loop の read_uint32_le() 呼び出しを per-record decode内へ直接展開
5. 旧CPU DFS、境界診断、funcid/profile/chunksize/mark/markdist系の旧診断関数を削除
6. main() を N=21 split145 full 検証本線へ特化
```

確認済みの同一性です。

```text
MAXD14 kernel                         identical
MAXD16 fallback kernel                 identical
MAXD18 fallback kernel                 identical
MAXD20 fallback kernel                 identical
MAXD21 fallback kernel                 identical
launch_kernel_dfs_iter_gpu_static_maxd identical
```

こちらでは `STATIC_ONLY=1` を通しています。

```text
source_version_tag              OK
source_future_check_mask_guard  OK
source_nosibling_parent         OK
source_217_232_restore_shape    OK
source_split_tag                split145 split233 OK
source_fasttrim_inline          OK
```

cudacodon側では、まず静的確認です。

```bash
STATIC_ONLY=1 bash 233Py_fasttrim_inline_split145_rootrestlate_restore_fastest_maxd14_validate_full_once.sh
```

その後、full実行です。

```bash
bash 233Py_fasttrim_inline_split145_rootrestlate_restore_fastest_maxd14_validate_full_once.sh
```

注意点として、233Pyは **本線特化版** です。旧diagnostic modeを使う場合は232PyまたはArchive側を残してください。233Pyは、N=21 fullの見通しとhost hot pathの軽量化を優先しています。

作成物はこちらです。

* [233Py_fasttrim_inline_split145_rootrestlate_restore_fastest_maxd14_probe.py](sandbox:/mnt/data/233Py_fasttrim_inline_split145_rootrestlate_restore_fastest_maxd14_probe.py)
* [233Py_fasttrim_inline_split145_rootrestlate_restore_fastest_maxd14_validate_full_once.sh](sandbox:/mnt/data/233Py_fasttrim_inline_split145_rootrestlate_restore_fastest_maxd14_validate_full_once.sh)
* [README.md](sandbox:/mnt/data/README.md)
* [233_fasttrim_inline_equivalence_report.txt](sandbox:/mnt/data/233_fasttrim_inline_equivalence_report.txt)
* [233_sha256.txt](sandbox:/mnt/data/233_sha256.txt)
* [233Py fasttrim-inline まとめtar](sandbox:/mnt/data/233Py_fasttrim_inline_split145_rootrestlate_restore_fastest_maxd14_bundle_20260706.tar)

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

VERSION_TAG:str="233 fasttrim-inline: r3 main-entry fix; 232 cleanup-only plus split145 host hot-path inline/diagnostic trim; MAXD14 kernel unchanged"
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

        j:int=(ijkl>>10)&31
        k:int=(ijkl>>5)&31
        l:int=ijkl&31

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
        # 233 fasttrim-inline: inline symmetry() / symmetry90() and bit unpacking.
        v:int=soa.ijkl_arr[t]
        ii:int=(v>>15)&31
        jj:int=(v>>10)&31
        kk:int=(v>>5)&31
        ll:int=v&31
        rot90v:int=((N-1-kk)<<15)+((N-1-ll)<<10)+(jj<<5)+ii
        if v==rot90v:
          w_arr[t]=u64(2)
        elif ii==N-1-jj and kk==N-1-ll:
          w_arr[t]=u64(4)
        else:
          w_arr[t]=u64(8)

    return soa,w_arr

def popcount_int(x:int)->int:
  c:int=0
  while x:
    x&=x-1
    c+=1
  return c

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
CHUNKSHAPE148_DEFAULT_REASON:str="233 fasttrim-inline reuses validated 232/231/217 scorestripe_v9 task order/cache; MAXD14 kernel unchanged"

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
  """233 fasttrim-inline: N21 split145 full-run host hot path with chunk worker body inlined."""
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

  if chunk_only:
    if debug_chunk_start<0:
      debug_chunk_start=0
    if debug_chunk_count<=0:
      debug_chunk_count=1
  stop_after_chunk:int=-1
  if chunk_only:
    stop_after_chunk=debug_chunk_start+debug_chunk_count-1

  if gpu_log_level>=1:
    print(f"[split145-gpu-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} chunk_only={1 if chunk_only else 0} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} worker={worker_id}/{worker_count} split_mode={split_mode} progress={progress_fname} fasttrim_inline=1")

  gpu_total:int=0
  off:int=0
  chunk_index:int=0
  executed_chunks:int=0
  meta_next:List[u8]=[u8(1),u8(2),u8(3),u8(3),u8(2),u8(6),u8(2),u8(2),u8(0),u8(4),u8(5),u8(7),u8(13),u8(14),u8(14),u8(14),u8(17),u8(14),u8(14),u8(20),u8(21),u8(21),u8(21),u8(25),u8(21),u8(21),u8(26),u8(26)]
  board_mask_gpu:u32=u32(board_mask)
  n3_gpu:u32=u32(1)<<u32(N-3)
  n4_gpu:u32=u32(1)<<u32(N-4)

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
        # 233 fasttrim-inline: inline read_uint32_le() in the per-record read path.
        ld:int=(ord(raw[0])&0xFF)|((ord(raw[1])&0xFF)<<8)|((ord(raw[2])&0xFF)<<16)|((ord(raw[3])&0xFF)<<24)
        rd:int=(ord(raw[4])&0xFF)|((ord(raw[5])&0xFF)<<8)|((ord(raw[6])&0xFF)<<16)|((ord(raw[7])&0xFF)<<24)
        col:int=(ord(raw[8])&0xFF)|((ord(raw[9])&0xFF)<<8)|((ord(raw[10])&0xFF)<<16)|((ord(raw[11])&0xFF)<<24)
        startijkl:int=(ord(raw[12])&0xFF)|((ord(raw[13])&0xFF)<<8)|((ord(raw[14])&0xFF)<<16)|((ord(raw[15])&0xFF)<<24)
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1
      m:int=len(chunk_constellations)
      if m==0:
        break

      if chunk_only:
        run_this_chunk:bool=(chunk_index>=debug_chunk_start and chunk_index<debug_chunk_start+debug_chunk_count)
        if not run_this_chunk:
          if gpu_log_level>=2:
            print(f"[split145-gpu-chunk-skip] N={N} chunk={chunk_index} off={off} m={m}")
          off+=m
          chunk_index+=1
          continue
      if worker_count>1:
        run_worker_chunk:bool=((chunk_index % worker_count)==worker_id)
        if not run_worker_chunk:
          if gpu_log_level>=2:
            print(f"[split145-gpu-worker-skip] N={N} worker={worker_id}/{worker_count} chunk={chunk_index} off={off} m={m}")
          off+=m
          chunk_index+=1
          continue

      t0=datetime.now()
      if gpu_log_level>=1:
        print(f"[split145-gpu-chunk-start] N={N} worker={worker_id}/{worker_count} chunk={chunk_index} off={off} m={m}")

      soa:TaskSoA=TaskSoA(STEPS)
      w_arr:List[u64]=[u64(0)]*STEPS
      results:List[u64]=[u64(0)]*STEPS

      build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)
      t1=datetime.now()

      # 233 fasttrim-inline: inline chunk input statistics and split bucket count.
      stats:List[int]=[0]*46
      if m>0:
        stats[1]=999999999
        stats[4]=999999999
        stats[7]=999999999
        stats[10]=999999999
        stats[13]=999999999
      d2base14_m:int=0
      d0_m:int=0
      i=0
      while i<m:
        fid:int=soa.funcid_arr[i]
        if fid==14:
          d2base14_m+=1
        if fid==26 or fid==27:
          d0_m+=1
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
        if fid>=0 and fid<28:
          stats[18+fid]+=1
        i+=1
      t2=datetime.now()

      if gpu_log_level>=1:
        print(f"[split145-buckets] N={N} m={m} generic={m} d2base14={d2base14_m} d0={d0_m} rest=0 split_mode={split_mode} specialized=0")

      chunk_total:int=0
      if m>0:
        GRID:int=(m+BLOCK-1)//BLOCK
        required_maxd:int=max_schedule_depth_of_tasks(soa,m,meta_next)
        selected_maxd:int=select_static_maxd(required_maxd)
        if gpu_log_level>=1:
          print(f"[maxd-dispatch] N={N} scope=split145 m={m} required_maxd={required_maxd} selected_MAXD={selected_maxd} schedule_words={packed_schedule_words_for_maxd(selected_maxd)} stack_bytes_per_thread={packed_stack_bytes_per_thread(selected_maxd)} capacity_check=OK")
        if not launch_kernel_dfs_iter_gpu_static_maxd(selected_maxd,soa,w_arr,meta_next,results,m,board_mask_gpu,n3_gpu,n4_gpu,GRID,BLOCK):
          print(f"[maxd-error] unsupported required_maxd={required_maxd}; supported maximum is 21")
          return gpu_total
        i=0
        while i<m:
          chunk_total+=int(results[i])
          i+=1
      t3=datetime.now()

      gpu_total+=chunk_total
      executed_chunks+=1
      elapsed_outer_text:str=str(t3-t0)[:-3]
      elapsed_outer_ms:int=stream_elapsed_text_to_ms(elapsed_outer_text)
      append_stream_funcid_reorder_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_outer_text,elapsed_outer_ms,chunk_total,gpu_total,total_records,stats)
      if gpu_log_level>=1:
        stage_soa_ms:int=stream_elapsed_text_to_ms(str(t1-t0)[:-3])
        stage_stats_ms:int=stream_elapsed_text_to_ms(str(t2-t1)[:-3])
        stage_kernel_reduce_ms:int=stream_elapsed_text_to_ms(str(t3-t2)[:-3])
        print(f"[split145-gpu-chunk-end] N={N} worker={worker_id}/{worker_count} chunk={chunk_index} off={off} m={m} elapsed={elapsed_outer_text} inner={elapsed_outer_text} elapsed_ms={elapsed_outer_ms} chunk_total={chunk_total} gpu_total={gpu_total} soa_ms={stage_soa_ms} stats_ms={stage_stats_ms} split_ms=0 kernel_reduce_ms={stage_kernel_reduce_ms}")
      off+=m
      chunk_index+=1

  if gpu_log_level>=1:
    print(f"[split145-gpu-summary] N={N} records={total_records} chunks={chunk_index} executed_chunks={executed_chunks} total={gpu_total} split_mode={split_mode} progress={progress_fname}")
    if worker_count>1:
      print(f"[worker-summary] N={N} worker={worker_id}/{worker_count} records={total_records} chunks={chunk_index} executed_chunks={executed_chunks} partial_total={gpu_total} progress={progress_fname}")
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
  global FUNCID_REORDER_V2_WINDOW_MULT,FUNCID_REORDER_V2_PHASE_JUMP
  global BROAD_MARKDIST_TAIL_VARIANT

  expected:List[int]=[0,0,0,0,0,10,4,40,92,352,724,2680,14200,73712,365596,2279184,14772512,95815104,666090624,4968057848,39029188884,314666222712,2691008701644,24233937684440,227514171973736,2207893435808352,22317699616364044,234907967154122528]
  nmin:int=21
  nmax:int=22
  use_gpu:bool=False
  gpu_block:int=32
  gpu_max_blocks:int=484
  gpu_log_level:int=1
  gpu_sort_mode:int=0
  requested_preset_arg:int=7
  preset_queens_arg:int=7
  bench_mode:int=31
  reorder_window_mult:int=8
  reorder_phase_jump:int=7
  cross_stripe_safe:bool=False
  worker_id:int=0
  worker_count:int=1
  broadmark_tail_variant:int=2
  argc:int=len(sys.argv)

  if argc>=2:
    arg=sys.argv[1]
    if arg=="-g":
      use_gpu=True
      print("GPU mode selected")
    else:
      print("Usage: nqueens -g nmin nmax block max_blocks log_level sort_mode preset_queens 31 reorder_window_mult reorder_phase_jump cross_stripe_safe worker_id worker_count broadmark_variant")
      return
  else:
    use_gpu=True
    print("GPU mode selected")
    print("[233-default] no arguments: GPU N21 split145 full default")

  if argc>=4:
    nmin=int(sys.argv[2])
    nmax=int(sys.argv[3])+1
  if argc>=5:
    gpu_block=int(sys.argv[4])
  if argc>=6:
    gpu_max_blocks=int(sys.argv[5])
  if argc>=7:
    gpu_log_level=int(sys.argv[6])
  if argc>=8:
    gpu_sort_mode=int(sys.argv[7])
  if argc>=9:
    requested_preset_arg=int(sys.argv[8])
    preset_queens_arg=requested_preset_arg
  if argc>=10:
    bench_mode=int(sys.argv[9])
  if argc>=11:
    reorder_window_mult=int(sys.argv[10])
  if argc>=12:
    reorder_phase_jump=int(sys.argv[11])
  if argc>=13:
    cross_stripe_safe=(int(sys.argv[12])!=0)
  if argc>=14:
    worker_id=int(sys.argv[13])
  if argc>=15:
    worker_count=int(sys.argv[14])
  if argc>=16:
    broadmark_tail_variant=int(sys.argv[15])
  if argc>16:
    print("Too many arguments")
    return

  if bench_mode!=31:
    print(f"[error] 233 fasttrim-inline is N21 split145 full-run focused; requested bench_mode={bench_mode}")
    return
  if reorder_window_mult<=0:
    reorder_window_mult=8
  if reorder_phase_jump<=0:
    reorder_phase_jump=7
  FUNCID_REORDER_V2_WINDOW_MULT=reorder_window_mult
  FUNCID_REORDER_V2_PHASE_JUMP=reorder_phase_jump
  if broadmark_tail_variant<0 or broadmark_tail_variant>5:
    broadmark_tail_variant=2
  BROAD_MARKDIST_TAIL_VARIANT=broadmark_tail_variant
  if worker_count<=0:
    worker_count=1
  if worker_id<0:
    worker_id=0
  if worker_id>=worker_count:
    worker_id=0
    worker_count=1

  if use_gpu and (nmin<5 or nmax>28):
    print(f"[error] 233 fasttrim-inline GPU candidate supports N=5..27 only; requested N={nmin}..{nmax-1}")
    return

  if use_gpu and gpu_log_level>=1:
    print(f"version        : {VERSION_TAG}")
    print(f"cross_stripe_safe: {1 if cross_stripe_safe else 0}")
    print(f"worker_split : worker={worker_id}/{worker_count}")
    print(f"broadmarktail_variant: id={BROAD_MARKDIST_TAIL_VARIANT} tag={broad_markdist_tail_variant_tag()} desc={broad_markdist_tail_variant_desc()}")
    print(f"split233_fasttrim_full_gpu: mode={bench_mode} preset={preset_queens_arg}")
    print(f"funcid_reorder_v2_params: window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} param={funcid_reorder_param_tag()} reason={FUNCID_REORDER_V2_DEFAULT_REASON}")
    print(f"broadmarktail_params: version={BROAD_MARKDIST_TAIL_REORDER_VERSION} variant={BROAD_MARKDIST_TAIL_VARIANT} tag={broad_markdist_tail_variant_tag()} window_boost={broad_markdist_tail_window_boost_value()} phase_mix={1 if broad_markdist_tail_use_phase_mix() else 0} rotate_interleave={1 if broad_markdist_tail_use_rotating_interleave() else 0} phase_salt={broad_markdist_tail_phase_salt_value()} reason={BROAD_MARKDIST_TAIL_REORDER_DEFAULT_REASON}")
    print(f"chunkshape148_params: version={CHUNKSHAPE148_REORDER_VERSION} reason={CHUNKSHAPE148_DEFAULT_REASON}")

  print(" N:             Total           Unique         hh:mm:ss.ms")
  for N in range(nmin,nmax):
    start_time=datetime.now()
    if N<=5:
      total_small:int=_bit_total(N)
      text_small:str=str(datetime.now()-start_time)[:-3]
      print(f"{N:2d}:{total_small:18d}{0:17d}{text_small:>21s}")
      continue
    if not use_gpu or N<21:
      print(f"[error] 233 fasttrim-inline supports the validated GPU stream path for N>=21 only; N={N}")
      continue

    ijkl_list:Set[int]=set()
    subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]]=set()
    preset_queens:int=select_dynamic_preset_queens(N,preset_queens_arg)
    if gpu_log_level>=1:
      print(f"[dynamic-preset] N={N} preset_queens={preset_queens}")

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
        print(f"[split233-full-reuse] N={N} records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)}")
    else:
      if gpu_log_level>=1:
        print(f"[split233-full-build] N={N} stream_records={stream_records} existing_records={reorder_records} done_count={done_count} bin={reorder_fname}")
      reorder_fname,reorder_records,reorder_chunks=build_broad_markdist_tail_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)

    base_reorder_fname:str=reorder_fname
    base_reorder_records:int=reorder_records
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

    progress_suffix:str=f"split233_full_{CHUNKSHAPE148_REORDER_VERSION}_{BROAD_MARKDIST_TAIL_REORDER_VERSION}_{broad_markdist_tail_variant_tag()}"
    total:int=exec_solutions_gpu_bin_stream_split145(N,shaped_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,False,0,1,"",progress_suffix,worker_id,worker_count,0)
    text:str=str(datetime.now()-start_time)[:-3]
    status:str="ok" if expected[N]==total else f"ng({total}!={expected[N]})"
    if worker_count>1:
      status=f"partial-worker-{worker_id}-of-{worker_count}"
    if gpu_log_level>=1:
      print(f"[split233-full-done] N={N} source_records={stream_records} base_reordered_records={base_reorder_records} shaped_records={shaped_records} chunks={shaped_chunks} bin={shaped_fname} base_bin={base_reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)} chunkshape={CHUNKSHAPE148_REORDER_VERSION} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} variant={broad_markdist_tail_variant_tag()} worker={worker_id}/{worker_count} total={total}")
      if worker_count>1:
        print(f"[worker-done] N={N} worker={worker_id}/{worker_count} partial_total={total} expected_total={expected[N]}")
    print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    {status}")

if __name__=="__main__":
  main()
