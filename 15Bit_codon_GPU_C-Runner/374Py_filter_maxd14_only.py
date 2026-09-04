#!/usr/bin/env python3
"""
374Py_filter_maxd14_only.py

kernel_dfs_iter_gpu_maxd14 is only ever launched, in real production,
on constellations whose required schedule depth (computed host-side by
max_schedule_depth_of_tasks(), via schedule_depth_for_task() per
record) is <=14 -- records needing more depth are routed to
kernel_dfs_iter_gpu_maxd16/18/20/21 instead. 361's dump was produced
by build_soa_for_range() run unconditionally over ALL 2,025,282 real
N=21 records, with no such filtering (dump_soa_reference_c_port() was
only ever meant to validate build_soa_for_range()+symmetry(), not to
double as maxd14-filtered kernel test input).

Feeding 361's full, unfiltered dump into the maxd14-only kernel port
is therefore testing outside that kernel's documented precondition:
some records genuinely need >14 schedule levels, and the schedule
precomputation loop (both in the original Codon kernel and in this C
port -- neither has a depth bound, since real dispatch is supposed to
guarantee it's never needed) does not terminate on those records.

This script applies the exact same filter production dispatch does:
literal port of schedule_depth_for_task() (626-667 of the Codon
source), run per record using each record's ctrl0/markctrl fields
already present in 361's dump, keeping only depth<=14 records and
writing them in the 7-field layout 363_kernel_maxd14.cu's CPU test
harness expects (ld, rd, col, ctrl0, free, markctrl, w_lo).

Usage:
  python3 374Py_filter_maxd14_only.py <361_dump.bin> <out_soa7_filtered.bin>
"""
# 374 RENAME NOTE: pure rename of 363_filter_maxd14_only.py to
# 374Py_filter_maxd14_only.py for the 374 consolidation package (see
# 374_README_append.md). schedule_depth_for_task() and every other line
# of logic below is BYTE-IDENTICAL to 363_filter_maxd14_only.py -- only
# the two filename self-references above changed.

import sys
import struct

IS_BASE_MASK_I  = 69222408
IS_JMARK_MASK_I = 4
IS_MARK_MASK_I  = 199209203
IS_P5_MASK_I    = 3840
SEL2_MASK_I     = 34742338
STP3_MASK_I     = 21266576

META_NEXT = [1,2,3,3,2,6,2,2,0,4,5,7,13,14,14,14,17,14,14,20,21,21,21,25,21,21,26,26]


def schedule_depth_for_task(ctrl0: int, markctrl: int) -> int:
    """Literal port of schedule_depth_for_task (626-667 of the Codon
    source). Values as plain Python ints; ctrl0/markctrl are already
    masked to 32 bits by construction (they came from a u32 dump)."""
    raw = ctrl0
    marks = markctrl
    jmark = marks & 31
    endm = (marks >> 5) & 31
    mark1 = (marks >> 10) & 31
    mark2 = (marks >> 15) & 31
    depth = 0

    while True:
        fu = raw & 31
        rowv = (raw >> 5) & 31

        if ((IS_P5_MASK_I >> fu) & 1) != 0 and rowv == mark1:
            fu = META_NEXT[fu]

        if ((IS_BASE_MASK_I >> fu) & 1) != 0 and rowv == endm:
            return depth

        stepv = 1
        nextfid = fu
        if ((IS_MARK_MASK_I >> fu) & 1) != 0:
            markv = mark2 if ((SEL2_MASK_I >> fu) & 1) != 0 else mark1
            if rowv == markv:
                stepv = 3 if ((STP3_MASK_I >> fu) & 1) != 0 else 2
                nextfid = META_NEXT[fu]

        if ((IS_JMARK_MASK_I >> fu) & 1) != 0 and rowv == jmark:
            nextfid = META_NEXT[fu]

        child_row = rowv + stepv
        depth += 1
        if depth > 21 or child_row > 31:
            return 22
        raw = nextfid | (child_row << 5)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <361_dump.bin> <out_soa7_filtered.bin>", file=sys.stderr)
        return 1
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, 'rb') as f:
        data = f.read()
    if len(data) % 40 != 0:  # 10 fields x 4 bytes = 40 bytes/record
        print(f"ERROR: input size {len(data)} is not a multiple of 40 "
              f"(expected 10 x u32 LE per record, the 361 dump layout)",
              file=sys.stderr)
        return 1
    n = len(data) // 40

    kept = 0
    depth_hist = {}
    with open(out_path, 'wb') as fout:
        for i in range(n):
            rec = struct.unpack_from('<10I', data, i * 40)
            ld, rd, col, row, ctrl0, free, markctrl, funcid, ijkl, w_lo = rec
            d = schedule_depth_for_task(ctrl0, markctrl)
            depth_hist[d] = depth_hist.get(d, 0) + 1
            if d <= 14:
                fout.write(struct.pack('<7I', ld, rd, col, ctrl0, free, markctrl, w_lo))
                kept += 1

    print(f"[filter-done] records_in={n} records_kept(depth<=14)={kept} "
          f"records_dropped={n - kept} out={out_path}")
    print("[filter-depth-histogram] " +
          " ".join(f"{d}:{c}" for d, c in sorted(depth_hist.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
