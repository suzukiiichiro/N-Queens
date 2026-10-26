#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""418_sum_results.py -- add CRunner per-thread results.bin files elementwise (u64 LE, one per thread) and
report the grand total; optionally compare the sum with a single-run results.bin and with an oracle.

The chunks of 418_split_sched.py start at multiples of the launch stride, so the elementwise sum of their
results.bin files is byte-identical to the single run's results.bin (see that script's docstring). This is
the strongest cheap check of a split run: not only the total, every one of the 30,720 thread sums agrees.

Usage:
  python3 418_sum_results.py --out SUM.bin [--expect ORACLE] [--ref SINGLE_RESULTS.bin] part00.bin part01.bin ...
  python3 418_sum_results.py --selftest
Exit 0 when every requested check passes; prints one "[sum-done] ..." line with the verdicts.
"""
import argparse, struct, sys, tempfile, os

def read_u64s(path):
    b = open(path, "rb").read()
    if len(b) % 8: sys.exit(f"[sum] ERROR: {path} size {len(b)} is not a multiple of 8")
    return list(struct.unpack("<%dQ" % (len(b) // 8), b))

def run(parts, out, expect, ref):
    acc = None
    per_part = []
    for p in parts:
        v = read_u64s(p)
        if acc is None: acc = [0] * len(v)
        if len(v) != len(acc): sys.exit(f"[sum] ERROR: {p} has {len(v)} threads, expected {len(acc)}")
        for i, x in enumerate(v): acc[i] += x
        per_part.append((p, sum(v)))
        print(f"[sum-part] {os.path.basename(p)} threads={len(v)} total={sum(v)}")
    if acc is None: sys.exit("[sum] ERROR: no parts")
    if max(acc) >= 1 << 64: sys.exit("[sum] ERROR: u64 overflow in a thread sum")
    total = sum(acc)
    if out:
        with open(out, "wb") as g: g.write(struct.pack("<%dQ" % len(acc), *acc))
    verdicts = []
    rc = 0
    if expect is not None:
        ok = (total == expect); rc |= (not ok)
        verdicts.append(f"oracle={'MATCH' if ok else 'MISMATCH'}(expected {expect})")
    if ref:
        r = read_u64s(ref)
        ok = (r == acc); rc |= (not ok)
        nd = sum(1 for a, b in zip(r, acc) if a != b) if len(r) == len(acc) else -1
        verdicts.append(f"ref_bytes={'IDENTICAL' if ok else 'DIFFER'}(threads_differing={nd})")
    print(f"[sum-done] parts={len(parts)} threads={len(acc)} total_sum={total} out={out or '-'} " + " ".join(verdicts))
    return int(rc)

def selftest():
    import random
    d = tempfile.mkdtemp(prefix="418_sum_")
    n = 16
    full = [random.getrandbits(40) for _ in range(n)]
    # split every thread sum into 3 random addends -> three "chunk" files
    parts = [[0] * n for _ in range(3)]
    for i, x in enumerate(full):
        a = random.randint(0, x); b = random.randint(0, x - a)
        parts[0][i], parts[1][i], parts[2][i] = a, b, x - a - b
    names = []
    for k, v in enumerate(parts):
        p = os.path.join(d, f"p{k}.bin"); open(p, "wb").write(struct.pack("<%dQ" % n, *v)); names.append(p)
    ref = os.path.join(d, "ref.bin"); open(ref, "wb").write(struct.pack("<%dQ" % n, *full))
    out = os.path.join(d, "sum.bin")
    assert run(names, out, sum(full), ref) == 0
    assert open(out, "rb").read() == open(ref, "rb").read()
    bad = full[:]; bad[3] += 1
    open(ref, "wb").write(struct.pack("<%dQ" % n, *bad))
    assert run(names, None, sum(full) + 1, ref) == 1
    print("[selftest] ok")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*"); ap.add_argument("--out"); ap.add_argument("--expect", type=int)
    ap.add_argument("--ref"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: return selftest()
    if not a.parts: ap.error("no part files")
    return run(a.parts, a.out, a.expect, a.ref)

if __name__ == "__main__":
    raise SystemExit(main())
