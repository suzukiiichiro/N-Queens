#!/usr/bin/env python3
import re
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("usage: python3 236Py_sum_worker_totals.py worker*.log", file=sys.stderr)
    sys.exit(2)

total = 0
expected = None
seen = {}
for arg in sys.argv[1:]:
    path = Path(arg)
    text = path.read_text(errors="replace")
    m = re.findall(r"\[worker-done\]\s+N=(\d+)\s+worker=(\d+)/(\d+)\s+partial_total=(\d+)\s+expected_total=(\d+)", text)
    if m:
        n, wid, wc, part, exp = m[-1]
        key = (int(n), int(wid), int(wc))
        val = int(part)
        total += val
        expected = int(exp)
        seen[key] = (val, str(path))
        continue
    # fallback: final table line with partial-worker status
    m2 = re.findall(r"^\s*(\d+):\s*(\d+)\s+\d+\s+\S+\s+partial-worker-(\d+)-of-(\d+)\s*$", text, flags=re.M)
    if m2:
        n, part, wid, wc = m2[-1]
        key = (int(n), int(wid), int(wc))
        val = int(part)
        total += val
        seen[key] = (val, str(path))
    else:
        print(f"[sum-warning] no worker total found: {path}")

print("[worker-sum]")
for key in sorted(seen):
    val, path = seen[key]
    n, wid, wc = key
    print(f"N={n} worker={wid}/{wc} partial_total={val} file={path}")
print(f"combined_total={total}")
if expected is not None:
    print(f"expected_total={expected}")
    print("status=OK" if total == expected else f"status=FAIL delta={total-expected}")
    sys.exit(0 if total == expected else 1)
