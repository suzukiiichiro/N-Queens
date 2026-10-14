#!/usr/bin/env python3
# 413_slice_soa7.py -- take COUNT equal-spaced records out of a SoA7 CRunner input
# (28-byte records: ld, rd, col, ctrl0, free, markctrl, w_lo as u32 LE) and write them,
# in their original relative order, as a smaller SoA7 file.
#
# Record i of the output is record floor(i * n / COUNT) of the input, so the slice is a
# stratified sample across the whole (schedule-sorted) file: every schedule group is
# represented in proportion. Absolute times on a slice are NOT representative of the full
# run (398-r2: subsets run 12-37% dear or 13-26% cheap); slices are for (a) per-thread
# byte identity between two binaries at the same stride and (b) relative comparisons of
# launch configs on the SAME slice. The oracle for a slice is not known in advance: the
# harness compares binaries against each other, not against expected[].
#
# Usage: python3 413_slice_soa7.py <in.bin> <out.bin> <count>
import sys, os, mmap

REC = 28

def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: 413_slice_soa7.py <in.bin> <out.bin> <count>")
    src, dst, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
    size = os.path.getsize(src)
    if size % REC != 0:
        sys.exit(f"ERROR: {src} size {size} not a multiple of {REC}")
    n = size // REC
    if count <= 0 or count > n:
        sys.exit(f"ERROR: count {count} out of range 1..{n}")
    with open(src, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm, open(dst, 'wb') as out:
        buf = bytearray()
        for i in range(count):
            j = (i * n) // count
            buf += mm[j * REC:(j + 1) * REC]
            if len(buf) >= (1 << 20):
                out.write(buf); buf = bytearray()
        if buf:
            out.write(buf)
    print(f"[slice-done] in={src} records_in={n} out={dst} records_out={count} step={n / count:.3f}")

if __name__ == '__main__':
    main()
