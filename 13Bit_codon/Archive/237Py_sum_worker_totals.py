#!/usr/bin/env python3
import re
import sys
from pathlib import Path

EXPECTED = {
    21: 314666222712,
    22: 2691008701644,
    23: 24233937684440,
    24: 227514171973736,
    25: 2207893435808352,
    26: 22317699616364044,
    27: 234907967154122528,
}

def parse_log(path: Path):
    text = path.read_text(errors="replace")
    m = re.search(r"\[worker-done\]\s+N=(\d+)\s+worker=(\d+)/(\d+)\s+partial_total=(\d+)", text)
    if m:
        return tuple(map(int, m.groups()))  # N, worker, workers, total
    # Fallback: final table line may contain partial-worker status.
    rows = re.findall(r"^\s*(\d+):\s*(\d+)\s+\d+\s+[0-9:,\. day]+\s+partial-worker-(\d+)-of-(\d+)\s*$", text, flags=re.M)
    if rows:
        n, total, worker, workers = rows[-1]
        return int(n), int(worker), int(workers), int(total)
    raise ValueError(f"could not parse worker total from {path}")

def main(argv):
    if len(argv) < 2:
        print("Usage: python3 237Py_sum_worker_totals.py worker*.log", file=sys.stderr)
        return 2
    entries = []
    for arg in argv[1:]:
        p = Path(arg)
        n, worker, workers, total = parse_log(p)
        entries.append((n, worker, workers, total, p))
    entries.sort(key=lambda x: x[1])
    n_values = {e[0] for e in entries}
    workers_values = {e[2] for e in entries}
    if len(n_values) != 1 or len(workers_values) != 1:
        raise SystemExit(f"mismatched N/worker_count in logs: N={sorted(n_values)} workers={sorted(workers_values)}")
    n = entries[0][0]
    workers = entries[0][2]
    seen = {e[1] for e in entries}
    missing = [i for i in range(workers) if i not in seen]
    total = sum(e[3] for e in entries)
    for n0, worker, workers0, t, p in entries:
        print(f"worker={worker}/{workers0}\tpartial_total={t}\tlog={p}")
    print(f"sum_total={total}")
    exp = EXPECTED.get(n)
    if missing:
        print(f"missing_workers={','.join(map(str, missing))}")
        return 1
    if exp is not None:
        status = "OK" if total == exp else f"NG expected={exp}"
        print(f"N={n}\texpected={exp}\tstatus={status}")
        return 0 if total == exp else 1
    print(f"N={n}\texpected=unknown\tstatus=INFO")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
