#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
325_gpu_inline_probe.py

Standalone, throwaway diagnostic probe -- NOT part of the N-Queens solver
lineage. Does NOT touch kernel_dfs_iter_gpu_maxd14 or any 3xxPy revision
file. Sole purpose: answer the mandatory pre-check from 324's Open
Objectives #1 -- does Codon's @gpu.kernel actually inline a plain
(non-@gpu.kernel) function called from within it, or does it emit a
real CALL/RET boundary in the compiled SASS?

Design: probe_kernel branches on a per-thread flag and calls one of two
DIFFERENT plain functions (variant_a / variant_b). Each contains a small
loop that cannot be trivially constant-folded away, so the two variants
are easy to tell apart in disassembly regardless of inlining behavior.

  - If Codon INLINES: probe_kernel's SASS will contain two distinct
    duplicated code blocks (one per branch), with NO CALL instruction
    and no separate variant_a/variant_b function symbols anywhere in
    the disassembly.
  - If Codon does NOT inline: the SASS will contain CALL instructions
    targeting separate function symbols, and variant_a/variant_b will
    each appear as their own disassembled subroutine ending in RET.

Build:
  codon build -release -o 325_gpu_inline_probe 325_gpu_inline_probe.py

Fastest check (no GPU execution needed -- pure static disassembly of
the compiled binary's embedded cubin/fatbin):
  cuobjdump --dump-sass ./325_gpu_inline_probe > 325_probe_sass.txt
  grep -nE "Function : |CALL|RET|variant_a|variant_b" 325_probe_sass.txt

Fallback / cross-check via ncu (same tool used throughout 318-321):
  sudo ncu --launch-count 1 --section SourceCounters -f \
    -o 325_inline_probe_ncu \
    ./325_gpu_inline_probe
  /usr/local/cuda/bin/ncu --page source --import 325_inline_probe_ncu.ncu-rep \
    2>&1 | tee 325_inline_probe_ncu_source.txt

Interpreting either output:
  - CALL/RET present + variant_a/variant_b appear as separate symbols
    -> Codon does NOT inline. The device-function specialization design
    in Open Objectives #1 needs a different approach (Codon generics,
    an explicit inline hint if one exists, or accept and directly
    measure call overhead) before touching kernel_dfs_iter_gpu_maxd14.
  - No CALL/RET, both variant bodies appear directly inlined into
    probe_kernel -> Codon DOES inline. The real design (splitting on
    future_check_mask==0, one axis only, per 324's plan) is safe to
    prototype next.

This file intentionally has no dependency on, and makes no changes to,
any part of the main solver. It is not tracked by the N=21 full
validation harness (that harness checks solver correctness/timing,
which is meaningless for this probe).
"""

import gpu

PROBE_N:Static[int]=4096

def variant_a(x:u32,y:u32)->u32:
  r:u32=x
  i:int=0
  while i<4:
    r=(r^(r<<u32(3)))+y
    i+=1
  return r

def variant_b(x:u32,y:u32)->u32:
  r:u32=x+y
  i:int=0
  while i<4:
    r=(r*u32(3))^y
    i+=1
  return r

@gpu.kernel
def probe_kernel(out:Ptr[u32],flag_arr:Ptr[u32],n:int)->None:
  tid:int=(gpu.block.x*gpu.block.dim.x)+gpu.thread.x
  if tid>=n:return
  flag:u32=flag_arr[tid]
  if flag==u32(0):
    out[tid]=variant_a(flag,u32(tid))
  else:
    out[tid]=variant_b(flag,u32(tid))

def main()->None:
  out:List[u32]=[u32(0) for _ in range(PROBE_N)]
  flag_arr:List[u32]=[u32(i%2) for i in range(PROBE_N)]
  block_size:int=256
  grid_size:int=(PROBE_N+block_size-1)//block_size
  probe_kernel(gpu.raw(out),gpu.raw(flag_arr),PROBE_N,grid=grid_size,block=block_size)
  # sum every element so the compiler cannot prove the kernel's output
  # is unused and eliminate it (and both branches) as dead code.
  checksum:u64=u64(0)
  for v in out:
    checksum+=u64(v)
  print("checksum:",checksum)

if __name__=="__main__":
  main()
