#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sum 111Py multi-GPU worker partial totals from worker logs.

Usage:
  python3 sum_111py_worker_totals.py 111Py_N22_worker*.log

It reads lines like:
  [worker-done] N=22 worker=2/4 partial_total=... expected_total=...
and verifies worker coverage and expected_total when available.
"""
import re
import sys
from pathlib import Path

pat = re.compile(r"\[worker-done\]\s+N=(\d+)\s+worker=(\d+)/(\d+)\s+partial_total=(\d+)\s+expected_total=(\d+)")

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 sum_111py_worker_totals.py 111Py_N22_worker*.log", file=sys.stderr)
        return 2
    rows = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        try:
            text = path.read_text(errors="replace")
        except Exception as e:
            print(f"[error] cannot read {path}: {e}", file=sys.stderr)
            return 2
        matches = pat.findall(text)
        if not matches:
            print(f"[warning] no [worker-done] line found in {path}", file=sys.stderr)
            continue
        n, wid, wcnt, total, expected = matches[-1]
        rows.append((int(n), int(wid), int(wcnt), int(total), int(expected), str(path)))

    if not rows:
        print("[error] no worker totals found", file=sys.stderr)
        return 1

    nset = {r[0] for r in rows}
    wcntset = {r[2] for r in rows}
    expset = {r[4] for r in rows}
    if len(nset) != 1 or len(wcntset) != 1 or len(expset) != 1:
        print("[error] inconsistent N / worker_count / expected_total among logs", file=sys.stderr)
        for r in rows:
            print(r, file=sys.stderr)
        return 1

    n = rows[0][0]
    worker_count = rows[0][2]
    expected = rows[0][4]
    by_worker = {}
    for _, wid, _, total, _, path in rows:
        if wid in by_worker:
            print(f"[error] duplicate worker_id={wid}: {by_worker[wid][1]} and {path}", file=sys.stderr)
            return 1
        by_worker[wid] = (total, path)

    missing = [i for i in range(worker_count) if i not in by_worker]
    if missing:
        print(f"[error] missing worker logs: {missing}", file=sys.stderr)
        return 1

    grand_total = sum(v[0] for v in by_worker.values())
    print(f"N={n}")
    print(f"worker_count={worker_count}")
    for i in range(worker_count):
        total, path = by_worker[i]
        print(f"worker{i}_partial_total={total}\t{path}")
    print(f"grand_total={grand_total}")
    print(f"expected_total={expected}")
    print("status=ok" if grand_total == expected else f"status=ng({grand_total}!={expected})")
    return 0 if grand_total == expected else 1

if __name__ == "__main__":
    raise SystemExit(main())
