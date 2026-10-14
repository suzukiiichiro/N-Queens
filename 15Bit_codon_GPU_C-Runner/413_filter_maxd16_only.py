#!/usr/bin/env python3
"""
413_filter_maxd16_only.py -- the maxd16 counterpart of 363_filter_maxd14_only.py.

Reads a 361 reference dump (10 x u32 LE per record: ld, rd, col, row, ctrl0, free,
markctrl, funcid, ijkl, w_lo), computes each record's required schedule depth with the
literal port of schedule_depth_for_task() (identical to 363's), keeps every record whose
depth is <= MAXDEPTH (default 15 = PACK423_ANCESTOR + 1 of 413_kernel_maxd16.cu) and
writes the 7-field / 28-byte SoA7 layout the CRunner binaries read
(ld, rd, col, ctrl0, free, markctrl, w_lo).

Differences from 363:
  * threshold 15 instead of 14 (env MAXDEPTH overrides; 413_validate.sh pins it);
  * the depth is memoised on (ctrl0, markctrl) -- the depth depends on nothing else, and
    the number of distinct pairs is tiny next to 44M records, so the N=23 pass is minutes,
    not hours;
  * the summary line also carries max_depth and the record count, which the harness gates
    (max_depth <= MAXDEPTH and records_kept == records_in: for N=23 no record may be dropped,
    a dropped record would silently lose solutions).

Usage:
  python3 413_filter_maxd16_only.py <361_dump.bin> <out_soa7_filtered.bin>
"""
import sys, os, struct, array

IS_BASE_MASK_I  = 69222408
IS_JMARK_MASK_I = 4
IS_MARK_MASK_I  = 199209203
IS_P5_MASK_I    = 3840
SEL2_MASK_I     = 34742338
STP3_MASK_I     = 21266576
META_NEXT = [1,2,3,3,2,6,2,2,0,4,5,7,13,14,14,14,17,14,14,20,21,21,21,25,21,21,26,26]
MAXDEPTH = int(os.environ.get("MAXDEPTH", "15"))


def schedule_depth_for_task(ctrl0: int, markctrl: int) -> int:
    """Literal port of schedule_depth_for_task (626-667 of the Codon source); same as 363."""
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
    size = os.path.getsize(in_path)
    if size % 40 != 0:
        print(f"ERROR: input size {size} is not a multiple of 40 (10 x u32 LE per record, the 361 dump layout)",
              file=sys.stderr)
        return 1
    n = size // 40
    a = array.array('I')
    with open(in_path, 'rb') as f:
        a.fromfile(f, n * 10)
    if sys.byteorder != 'little':
        a.byteswap()

    cache = {}
    depth_hist = {}
    kept = 0
    max_depth = 0
    CHUNK = 1 << 16
    with open(out_path, 'wb') as fout:
        out = array.array('I')
        for i in range(n):
            b = i * 10
            ctrl0 = a[b + 4]; markctrl = a[b + 6]
            key = (ctrl0, markctrl)
            d = cache.get(key)
            if d is None:
                d = schedule_depth_for_task(ctrl0, markctrl)
                cache[key] = d
            depth_hist[d] = depth_hist.get(d, 0) + 1
            if d > max_depth:
                max_depth = d
            if d <= MAXDEPTH:
                out.extend((a[b], a[b + 1], a[b + 2], ctrl0, a[b + 5], markctrl, a[b + 9]))
                kept += 1
                if len(out) >= CHUNK * 7:
                    if sys.byteorder != 'little':
                        out.byteswap()
                    out.tofile(fout); out = array.array('I')
            if (i & 0x3FFFFF) == 0 and i:
                print(f"[filter-progress] {i}/{n} kept={kept} distinct_schedules={len(cache)}", file=sys.stderr, flush=True)
        if len(out):
            if sys.byteorder != 'little':
                out.byteswap()
            out.tofile(fout)

    print(f"[filter-done] records_in={n} records_kept(depth<={MAXDEPTH})={kept} "
          f"records_dropped={n - kept} max_depth={max_depth} distinct_schedules={len(cache)} out={out_path}")
    print("[filter-depth-histogram] " + " ".join(f"{d}:{c}" for d, c in sorted(depth_hist.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
