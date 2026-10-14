#!/usr/bin/env python3
"""
415_bitcheck.py -- bit-budget evidence for the 12-byte DFS frame, read off a 361 reference dump.

Reads a *.soa_ref_361.bin (10 x u32 LE per record: ld, rd, col, row, ctrl0, free, markctrl,
funcid, ijkl, w_lo) and prints, over ALL records, the OR and AND of ld / rd / col / free, the
highest set bit of each, and the per-bit population of rd above the board. It then checks the
invariants the 424 layout (N=24) relies on:

  I1  col: bits N-2 and N-1 are 1 in EVERY record         (AND(col) has both bits)
  I2  free: bits N-2 and N-1 are 0 in EVERY record        (OR(free) has neither bit)
  I3  rd: highest set bit <= 30                            (B = rd[1..30] | ld[0] fits a u32)
  I4  ld: (INFO) highest set bit -- the host ld DOES carry off-board bits (real N=24: bits 24..27,
      because the raw constellation ld is the preset queens' diagonals shifted past the board).
      They are harmless: the kernel reads ld only after a left shift, so every ld bit >= N-3 is
      dead (415 note); 403/423 already drop them on push and matched the oracle. Reported, not gated.
  I5  row (= start) < 32 and funcid < 28                   (schedule fields unchanged)

These are HOST-side facts (the root records). The in-kernel part of the argument (rd only moves
down, ld only moves up, col's two constant bits cover cur_x bits N-2/N-1) is in the 415 note and
does not need data. Usage:

  python3 415_bitcheck.py constellations_N24_7.bin.soa_ref_361.bin 24
  python3 415_bitcheck.py constellations_N23_7.bin.soa_ref_361.bin 23      # control

GPU zero, ~1 min per GB with numpy, ~10x slower without.
"""
import sys, os

def hb(x):
    return x.bit_length() - 1 if x else -1

def main():
    if len(sys.argv) != 3:
        print(__doc__); return 2
    path, N = sys.argv[1], int(sys.argv[2])
    size = os.path.getsize(path)
    if size % 40:
        print(f"ERROR: {size} is not a multiple of 40"); return 1
    n = size // 40
    names = ["ld", "rd", "col", "row", "ctrl0", "free", "markctrl", "funcid", "ijkl", "w_lo"]
    OR = [0] * 10; AND = [0xFFFFFFFF] * 10
    rd_hist = [0] * 32; ld_hist = [0] * 32
    MAX = [0] * 10
    try:
        import numpy as np
        CH = 1 << 22
        with open(path, "rb") as f:
            done = 0
            while done < n:
                m = min(CH, n - done)
                a = np.frombuffer(f.read(m * 40), dtype="<u4").reshape(m, 10)
                for k in range(10):
                    OR[k] |= int(np.bitwise_or.reduce(a[:, k]))
                    AND[k] &= int(np.bitwise_and.reduce(a[:, k]))
                    MAX[k] = max(MAX[k], int(a[:, k].max()))
                for b in range(32):
                    rd_hist[b] += int(np.count_nonzero((a[:, 1] >> b) & 1))
                    ld_hist[b] += int(np.count_nonzero((a[:, 0] >> b) & 1))
                done += m
                print(f"[bitcheck-progress] {done}/{n}", file=sys.stderr, flush=True)
    except ImportError:
        import array
        with open(path, "rb") as f:
            CH = 1 << 20
            done = 0
            while done < n:
                m = min(CH, n - done)
                a = array.array("I"); a.fromfile(f, m * 10)
                if sys.byteorder != "little": a.byteswap()
                for i in range(m):
                    b = i * 10
                    for k in range(10):
                        v = a[b + k]; OR[k] |= v; AND[k] &= v
                        if v > MAX[k]: MAX[k] = v
                    v = a[b + 1]
                    while v:
                        lb = v & -v; rd_hist[hb(lb)] += 1; v ^= lb
                    v = a[b]
                    while v:
                        lb = v & -v; ld_hist[hb(lb)] += 1; v ^= lb
                done += m
                print(f"[bitcheck-progress] {done}/{n}", file=sys.stderr, flush=True)

    print(f"[bitcheck] file={path} N={N} records={n}")
    for k, nm in enumerate(names):
        print(f"[bitcheck-field] {nm:8s} OR=0x{OR[k]:08x} (hb {hb(OR[k]):2d})  AND=0x{AND[k]:08x}  MAX={MAX[k]}")
    print("[bitcheck-rd-bits-above-board] " + " ".join(f"{b}:{rd_hist[b]}" for b in range(N, 32) if rd_hist[b]))
    print("[bitcheck-ld-bits] " + " ".join(f"{b}:{ld_hist[b]}" for b in range(32) if ld_hist[b]))

    hi2 = (1 << (N - 2)) | (1 << (N - 1))
    checks = [
        ("I1_col_bits_N-2_N-1_always_1", (AND[2] & hi2) == hi2),
        ("I2_free_bits_N-2_N-1_always_0", (OR[5] & hi2) == 0),
        ("I3_rd_highest_bit_le_30", hb(OR[1]) <= 30),
        ("I5_row_lt_32_funcid_lt_28", MAX[3] < 32 and MAX[7] < 28),
    ]
    print(f"[bitcheck-info] I4_ld_highest_bit={hb(OR[0])} (N-1={N-1}; bits >= N-3={N-3} are dead in the kernel: "
          f"{sum(ld_hist[b] for b in range(N-3, 32))} set-bit occurrences at those positions -- expected, not a failure)")
    bad = 0
    for nm, ok in checks:
        print(f"[bitcheck-{'ok' if ok else 'FAIL'}] {nm}")
        bad += (not ok)
    a_bits = 2 * (N - 2) + (N - 4)
    print(f"[bitcheck-budget] 424-style A = col {N-2} + avail {N-2} + ld[1..{N-4}] {N-4} = {a_bits} bits "
          f"({'fits' if a_bits <= 64 else 'does NOT fit'} 64); B = rd[1..30] + ld[0] = 31 bits")
    print(f"[bitcheck-done] N={N} records={n} failures={bad}")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
