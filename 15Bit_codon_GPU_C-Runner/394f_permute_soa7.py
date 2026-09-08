#!/usr/bin/env python3
# 394f_permute_soa7.py -- permute a SoA7 CRunner input (395a: sched key as one int) (28-byte records:
# ld, rd, col, ctrl0, free, markctrl, w_lo as u32 LE) into a new order.
#
# Every mode is a PERMUTATION: the record multiset is unchanged, so the
# N=21 oracle 314666222712 must still match on the GPU. The tool prints an
# order-independent checksum (record count + per-field u64 sums) for both
# input and output and refuses to write if they differ.
#
# Modes:
#   random <seed>   uniform shuffle (Fisher-Yates via random.Random(seed))
#                   -- the CONTROL: if the generator's natural order is good
#                   only because adjacent constellations are similar, a
#                   shuffle should be as bad as the broadmarktail base.
#   state           stable sort by (col, ld, rd): root-state similarity
#   free            stable sort by (-popcount(free), col, ld, rd): cost proxy
#                   first, then root state
#   sched           stable sort by (markctrl, ctrl0): schedule similarity,
#                   raw adjacency preserved within equal keys (this is the
#                   funcid grouping WITHOUT the broadmarktail interleave)
#
# Usage: python3 394f_permute_soa7.py <mode> <in.bin> <out.bin> [seed]
import sys, struct, array, random

REC = 28

def load(path):
    data = open(path, 'rb').read()
    if len(data) % REC != 0:
        sys.exit(f"ERROR: {path} size {len(data)} not a multiple of {REC}")
    n = len(data) // REC
    a = array.array('I')
    a.frombytes(data)
    if sys.byteorder != 'little':
        a.byteswap()
    return n, a

def checksum(n, a):
    sums = [0]*7
    for i in range(n):
        b = i*7
        for f in range(7):
            sums[f] += a[b+f]
    return (n, tuple(sums))

def field(a, i, f):
    return a[i*7+f]

def main():
    if len(sys.argv) < 4:
        sys.exit("Usage: 394f_permute_soa7.py <random|state|free|sched> <in.bin> <out.bin> [seed]")
    mode, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 394
    n, a = load(src)
    cs_in = checksum(n, a)
    idx = list(range(n))
    if mode == 'random':
        random.Random(seed).shuffle(idx)
    elif mode == 'state':
        idx.sort(key=lambda i: (field(a,i,2), field(a,i,0), field(a,i,1)))
    elif mode == 'free':
        idx.sort(key=lambda i: (-bin(field(a,i,4)).count('1'), field(a,i,2), field(a,i,0), field(a,i,1)))
    elif mode == 'sched':
        # 395a: single-int key (markctrl<<32 | ctrl0) == the (markctrl, ctrl0)
        # tuple order exactly (ctrl0 < 2^32), but ~5x less memory for the
        # 28.7M-record N=22 file. Stable sort: raw adjacency kept within
        # equal keys (this is what makes L3 -1.7% .. -4.9%).
        idx.sort(key=lambda i: (field(a,i,5) << 32) | field(a,i,3))
    else:
        sys.exit(f"ERROR: unknown mode {mode}")
    out = array.array('I', [0]*(n*7))
    for j, i in enumerate(idx):
        out[j*7:(j+1)*7] = a[i*7:(i+1)*7]
    cs_out = checksum(n, out)
    if cs_in != cs_out:
        sys.exit(f"ERROR: checksum mismatch after permutation: in={cs_in} out={cs_out}")
    if sys.byteorder != 'little':
        out.byteswap()
    open(dst, 'wb').write(out.tobytes())
    # how many records kept their raw neighbour? (adjacency retained)
    kept = sum(1 for j in range(1, n) if idx[j] == idx[j-1] + 1)
    print(f"[permute-done] mode={mode} seed={seed if mode=='random' else '-'} records={n} "
          f"raw_adjacent_pairs_kept={kept} ({100.0*kept/max(1,n-1):.1f}%) checksum_records={cs_in[0]} "
          f"checksum_fieldsums={'/'.join(str(x) for x in cs_in[1])} out={dst}")

if __name__ == '__main__':
    main()
