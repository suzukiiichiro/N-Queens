#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
327_w_arr_loadsplit_probe.py

Standalone, throwaway diagnostic probe -- NOT part of the N-Queens solver
lineage. Does NOT touch kernel_dfs_iter_gpu_maxd14 or any 3xxPy revision
file. Sole purpose: verify the 323 hypothesis that `w_arr[idx]` (a
Ptr[u64] array read via a single idx index, once per outer-loop
iteration) compiles to TWO separate 32-bit LDG.E instructions instead of
one 64-bit load -- BEFORE touching the real kernel, learning the lesson
of 326 (a plausible hypothesis still caused a large, unexpected
regression when implemented directly).

=== ROUND 1 RESULT (confirmed) ===
w_probe_kernel's SASS showed exactly the pattern seen in
319_ncu_source.txt: two 32-bit LDG.E at offset +0/+4 of one base
register, no LDG.E.64 anywhere. This confirms the split is a general
Codon/NVPTX code-generation behavior for this access shape, not
something specific to the surrounding DFS complexity.

=== ROUND 2: why the split ALSO costs "excessive sectors", not just an
extra instruction ===
Both real-kernel LDG.E rows (319_ncu_source.txt) carried non-zero L2
Theoretical Sectors Global Excessive (92,928 each). A plain instruction
split alone would not explain that -- 32 threads each reading 4
contiguous bytes should coalesce perfectly on its own. The likely cause:
w_arr is 8-byte-strided, so for a fixed 32-bit half (say offset+0),
consecutive threads' addresses are 8 bytes apart, not 4 -- a stride-2
gap relative to plain 4-byte-per-thread packing. That gap is what wastes
L2 sector fetches, not the instruction count per se.

This round tests THREE kernel variants in one build/profile pass:
  w_probe_kernel        -- baseline, confirmed split (round 1)
  w_probe_kernel_tmpvar -- same read, but through a named local variable
                            before the multiply (cheap to rule out;
                            unlikely to change codegen, since the
                            underlying load shape is identical)
  w_probe_kernel_soa    -- w_arr split into two SEPARATE densely-packed
                            u32 arrays (w_lo_arr, w_hi_arr), each read
                            with plain contiguous 4-byte-per-thread
                            indexing (idx, not idx*2), then recombined:
                              lo:u32=w_lo_arr[idx]
                              hi:u32=w_hi_arr[idx]
                              combined:u64=u64(lo)|(u64(hi)<<u64(32))
                            If the stride-gap theory above is right, this
                            should let BOTH 32-bit loads coalesce cleanly
                            (consecutive threads now 4 bytes apart in
                            each array, not 8), eliminating the excess
                            sectors even though it's still two loads.

Build:
  codon build -release -o 327_w_arr_loadsplit_probe 327_w_arr_loadsplit_probe.py

Fastest check (no GPU execution needed):
  cuobjdump --dump-sass ./327_w_arr_loadsplit_probe > 327_probe_sass.txt
  grep -nE "LDG\.E" 327_probe_sass.txt

Fallback via ncu (same tool used throughout 318-321) -- this one matters
more this round, since we need the L2 Theoretical Sectors Global
Excessive column per kernel, which cuobjdump's static SASS dump alone
does not show (that requires the profiled PC-sampling/counter run):
  sudo ncu --launch-count 3 --section SourceCounters -f \
    -o 327_w_arr_probe_ncu_v2 \
    ./327_w_arr_loadsplit_probe
  /usr/local/cuda/bin/ncu --page source --import 327_w_arr_probe_ncu_v2.ncu-rep \
    2>&1 | tee 327_w_arr_probe_ncu_v2_source.txt
  (--launch-count 3 because this build now launches 3 kernels in one run;
  the report will contain a separate "Kernel Name" section for each --
  compare the LDG.E rows and their Excessive-sectors columns across all
  three sections.)

Interpreting the output -- for EACH kernel's LDG.E row(s), compare:
  - Still two 32-bit LDG.E with non-zero Excessive sectors in ALL THREE
    kernels -> the stride-gap theory is wrong (or Codon disregards SoA
    layout too); the split/excess is unavoidable via source rewrites in
    this toolchain, and the 11.79% lead is likely not cheaply reachable.
  - w_probe_kernel_soa's two LDG.E rows show Excessive sectors close to
    ZERO while baseline/tmpvar still show the ~92,928-style excess ->
    the stride-gap theory is confirmed; splitting w_arr into an SoA
    (w_lo_arr/w_hi_arr) pair at the HOST side becomes a well-motivated,
    reasonably low-risk next real-kernel change to propose (still
    outside the hot DFS loop, only touching the once-per-task read at
    line ~839 and its precompute step).
  - w_probe_kernel_tmpvar behaves identically to baseline (expected) --
    confirms the split isn't about source phrasing of the read itself,
    only isolates whether SoA layout (not just syntax) is what matters.

This file intentionally has no dependency on, and makes no changes to,
any part of the main solver. It is not tracked by the N=21 full
validation harness (that harness checks solver correctness/timing,
which is meaningless for this probe).
"""

import gpu

PROBE_N:Static[int]=200000
PROBE_STRIDE:Static[int]=16384

@gpu.kernel
def w_probe_kernel(results:Ptr[u64],w_arr:Ptr[u64],n:int,stride:int)->None:
  tid:int=(gpu.block.x*gpu.block.dim.x)+gpu.thread.x
  idx:int=tid
  thread_total:u64=u64(0)
  while idx<n:
    total:u64=u64(idx&15)+u64(1)
    thread_total+=total*w_arr[idx]
    idx+=stride
  results[tid]=thread_total

@gpu.kernel
def w_probe_kernel_tmpvar(results:Ptr[u64],w_arr:Ptr[u64],n:int,stride:int)->None:
  tid:int=(gpu.block.x*gpu.block.dim.x)+gpu.thread.x
  idx:int=tid
  thread_total:u64=u64(0)
  while idx<n:
    total:u64=u64(idx&15)+u64(1)
    wv:u64=w_arr[idx]
    thread_total+=total*wv
    idx+=stride
  results[tid]=thread_total

@gpu.kernel
def w_probe_kernel_soa(results:Ptr[u64],w_lo_arr:Ptr[u32],w_hi_arr:Ptr[u32],n:int,stride:int)->None:
  tid:int=(gpu.block.x*gpu.block.dim.x)+gpu.thread.x
  idx:int=tid
  thread_total:u64=u64(0)
  while idx<n:
    total:u64=u64(idx&15)+u64(1)
    lo:u32=w_lo_arr[idx]
    hi:u32=w_hi_arr[idx]
    combined:u64=u64(lo)|(u64(hi)<<u64(32))
    thread_total+=total*combined
    idx+=stride
  results[tid]=thread_total

def main()->None:
  w_arr:List[u64]=[u64(i*2654435761 % 1000000007) for i in range(PROBE_N)]
  w_lo_arr:List[u32]=[u32(v&u64(0xffffffff)) for v in w_arr]
  w_hi_arr:List[u32]=[u32(v>>u64(32)) for v in w_arr]
  results:List[u64]=[u64(0) for _ in range(PROBE_STRIDE)]
  results_tmpvar:List[u64]=[u64(0) for _ in range(PROBE_STRIDE)]
  results_soa:List[u64]=[u64(0) for _ in range(PROBE_STRIDE)]
  block_size:int=256
  grid_size:int=(PROBE_STRIDE+block_size-1)//block_size

  w_probe_kernel(gpu.raw(results),gpu.raw(w_arr),PROBE_N,PROBE_STRIDE,grid=grid_size,block=block_size)
  w_probe_kernel_tmpvar(gpu.raw(results_tmpvar),gpu.raw(w_arr),PROBE_N,PROBE_STRIDE,grid=grid_size,block=block_size)
  w_probe_kernel_soa(gpu.raw(results_soa),gpu.raw(w_lo_arr),gpu.raw(w_hi_arr),PROBE_N,PROBE_STRIDE,grid=grid_size,block=block_size)

  # sum every element so the compiler cannot prove any kernel's output
  # is unused and eliminate its load as dead code.
  checksum:u64=u64(0)
  for v in results:
    checksum+=v
  for v in results_tmpvar:
    checksum+=v
  for v in results_soa:
    checksum+=v
  print("checksum:",checksum)
  # sanity: baseline and soa must compute the SAME thread_total values
  # (same w_arr content, just split into two u32 arrays), so their sums
  # over all threads must be identical -- verifies the SoA reconstruction
  # (lo|hi<<32) is correct, not just "some other value that happens to
  # coalesce better".
  sum_base:u64=u64(0)
  sum_soa:u64=u64(0)
  for v in results:
    sum_base+=v
  for v in results_soa:
    sum_soa+=v
  print("sum_base:",sum_base,"sum_soa:",sum_soa,"match:",sum_base==sum_soa)

if __name__=="__main__":
  main()

