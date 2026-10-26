#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""418_split_sched.py -- cut a CRunner input (28 B/record, sched394f order) into K contiguous chunks
whose boundaries are multiples of STRIDE records, so that the per-thread results of the chunks add up
to the per-thread results of the whole file.

WHY THE STRIDE MATTERS. The CRunner kernel is one grid-stride launch: thread t sums records
t, t+stride, t+2*stride, ... (stride = BLOCK x MAX_BLOCKS = 32 x 960 = 30,720) and writes one u64 per
thread (results.bin, 245,760 B). If a chunk starts at record o with o % stride == 0, its local record j
is global record o+j and (o+j) % stride == j % stride, so chunk thread t holds exactly the terms of the
whole run's thread t that fall inside the chunk. Elementwise u64 addition of the chunk results.bin files
is therefore BYTE-IDENTICAL to the single-run results.bin -- the harness gates that (418_sum_results.py).

Chunk sizes: the ceil(m / stride) stride-groups are dealt out as evenly as possible (the first m%K chunks get
one group more), so every chunk but the last is a multiple of stride records and no chunk is empty for
K <= ceil(m / stride) (enforced). The record ORDER is untouched
(394f's neighbour pairs survive inside each chunk; only K-1 pairs are cut at the boundaries).

Usage:
  python3 418_split_sched.py <in.bin> <K> [--stride 30720] [--outdir DIR] [--prefix NAME] [--verify]
  python3 418_split_sched.py --selftest
Outputs <prefix>.part{k:02d}of{K:02d}.bin and <prefix>.split{K:02d}.manifest.tsv
(columns: part offset_records count_records bytes sha256). --verify re-reads every part and checks that
their concatenation hashes to the source (streamed, no RAM). Existing parts with the right size are reused.
"""
import argparse, hashlib, os, sys, tempfile

REC = 28

def sha256_file(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b: break
            h.update(b)
    return h.hexdigest()

def plan(m, k, stride):
    if m <= 0 or m % 1: raise ValueError("bad record count")
    nstr = (m + stride - 1) // stride
    if k < 1 or k > nstr: raise ValueError(f"K={k} must be in 1..{nstr} (= ceil(records/stride))")
    # distribute the ceil(m/stride) stride-groups as evenly as possible: the first (nstr % k) chunks get one more
    base, extra = divmod(nstr, k)
    parts = []
    off = 0
    for i in range(k):
        cnt = min((base + (1 if i < extra else 0)) * stride, m - off)
        if cnt <= 0:
            raise ValueError(f"K={k} leaves chunk {i} empty (records={m}, stride={stride}); use a smaller K")
        parts.append((off, cnt))
        off += cnt
    assert off == m, (off, m)
    return parts

def split(src, k, stride, outdir, prefix, verify):
    size = os.path.getsize(src)
    if size % REC: sys.exit(f"[split] ERROR: {src} size {size} is not a multiple of {REC}")
    m = size // REC
    parts = plan(m, k, stride)
    os.makedirs(outdir, exist_ok=True)
    rows = []
    hsrc = hashlib.sha256() if verify else None
    with open(src, "rb") as f:
        for i, (off, cnt) in enumerate(parts):
            name = os.path.join(outdir, f"{prefix}.part{i:02d}of{k:02d}.bin")
            nbytes = cnt * REC
            reuse = os.path.exists(name) and os.path.getsize(name) == nbytes
            f.seek(off * REC)
            h = hashlib.sha256()
            if reuse:
                # hash both the existing part and the source range, they must agree
                hp = sha256_file(name)
                left = nbytes
                while left:
                    b = f.read(min(left, 1 << 24)); left -= len(b); h.update(b)
                    if hsrc: hsrc.update(b)
                if hp != h.hexdigest(): sys.exit(f"[split] ERROR: existing {name} differs from the source range; delete it")
                print(f"[split-part] {i} reuse {name} records={cnt} offset={off}")
            else:
                tmp = name + ".tmp"
                with open(tmp, "wb") as g:
                    left = nbytes
                    while left:
                        b = f.read(min(left, 1 << 24))
                        if not b: sys.exit("[split] ERROR: short read")
                        left -= len(b); g.write(b); h.update(b)
                        if hsrc: hsrc.update(b)
                os.replace(tmp, name)
                print(f"[split-part] {i} wrote {name} records={cnt} offset={off}")
            rows.append((i, off, cnt, nbytes, h.hexdigest()))
    man = os.path.join(outdir, f"{prefix}.split{k:02d}.manifest.tsv")
    with open(man, "w") as g:
        g.write(f"# source={os.path.abspath(src)} records={m} stride={stride} K={k}\n")
        g.write("part\toffset_records\tcount_records\tbytes\tsha256\n")
        for r in rows: g.write("\t".join(str(x) for x in r) + "\n")
    if verify:
        full = sha256_file(src)
        ok = (full == hsrc.hexdigest())
        print(f"[split-verify] concat_sha={hsrc.hexdigest()[:16]} source_sha={full[:16]} {'MATCH' if ok else 'MISMATCH'}")
        if not ok: sys.exit(1)
    print(f"[split-done] records={m} stride={stride} K={k} per_chunk={parts[0][1]} last={parts[-1][1]} manifest={man}")
    return 0

def selftest():
    import random
    d = tempfile.mkdtemp(prefix="418_split_")
    stride = 8
    for m in (8, 9, 23, 64, 65, 1000, 1001):
        src = os.path.join(d, f"src{m}.bin")
        data = bytes(random.getrandbits(8) for _ in range(m * REC))
        open(src, "wb").write(data)
        for k in range(1, (m + stride - 1) // stride + 1):
            parts = plan(m, k, stride)
            assert all(off % stride == 0 for off, _ in parts)
            assert sum(c for _, c in parts) == m and all(c > 0 for _, c in parts)
            split(src, k, stride, d, f"t{m}", verify=False) if k <= 3 else None
            if k <= 3:
                cat = b"".join(open(os.path.join(d, f"t{m}.part{i:02d}of{k:02d}.bin"), "rb").read() for i in range(k))
                assert cat == data, (m, k)
        bad = False
        try: plan(m, (m + stride - 1) // stride + 1, stride)
        except ValueError: bad = True
        assert bad
    print("[selftest] ok")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?"); ap.add_argument("k", nargs="?", type=int)
    ap.add_argument("--stride", type=int, default=30720); ap.add_argument("--outdir", default=".")
    ap.add_argument("--prefix", default=None); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: return selftest()
    if not a.src or not a.k: ap.error("src and K are required")
    return split(a.src, a.k, a.stride, a.outdir, a.prefix or os.path.basename(a.src), a.verify)

if __name__ == "__main__":
    raise SystemExit(main())
