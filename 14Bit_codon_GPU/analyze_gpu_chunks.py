#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse [gpu-chunk] lines from N-Queens GPU logs and print normalized chunk costs.

Usage:
  python3 analyze_gpu_chunks.py 20260423ChatGPT.txt
  ./binary -g 20 20 32 512 2 0 | python3 analyze_gpu_chunks.py -
"""

import re
import sys
from statistics import mean

CHUNK_RE = re.compile(
    r"\[gpu-chunk\]\s+N=(?P<N>\d+)\s+chunk=(?P<chunk>\d+)\s+off=(?P<off>\d+)\s+"
    r"m=(?P<m>\d+)\s+grid=(?P<grid>\d+)\s+sort=(?P<sort>\d+)\s+"
    r"build=(?P<build>\S+)\s+"
    r"(?:(?:sortprep=(?P<sortprep>\S+)\s+))?"
    r"kernel=(?P<kernel>\S+)\s+"
    r"(?:(?:span=(?P<span>\S+)\s+))?"
    r"copy=(?P<copy>\S+)"
)

def hms_to_seconds(s: str) -> float:
    # Accept H:MM:SS.mmm and D variants are not expected for chunk kernels.
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"unexpected duration: {s!r}")
    h = int(parts[0])
    m = int(parts[1])
    sec = float(parts[2])
    return h * 3600 + m * 60 + sec

def read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_gpu_chunks.py <logfile-or->", file=sys.stderr)
        return 2

    text = read_text(sys.argv[1])
    rows = []
    for line in text.splitlines():
        m = CHUNK_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        kernel = hms_to_seconds(d["kernel"])
        cnt = int(d["m"])
        rows.append({
            "N": int(d["N"]),
            "chunk": int(d["chunk"]),
            "off": int(d["off"]),
            "m": cnt,
            "grid": int(d["grid"]),
            "sort": int(d["sort"]),
            "kernel": kernel,
        })

    if not rows:
        print("No [gpu-chunk] lines found.", file=sys.stderr)
        return 1

    full_m = max(r["m"] for r in rows)
    total_kernel = sum(r["kernel"] for r in rows)

    print("N sort chunk off m grid kernel_sec sec_per_const normalized_to_full")
    for r in rows:
        sec_per = r["kernel"] / r["m"] if r["m"] else 0.0
        norm = sec_per * full_m
        print(
            f'{r["N"]} {r["sort"]} {r["chunk"]} {r["off"]} {r["m"]} {r["grid"]} '
            f'{r["kernel"]:.3f} {sec_per:.9f} {norm:.3f}'
        )

    first_full = [r["kernel"] for r in rows if r["m"] == full_m]
    tail = [r for r in rows if r["m"] != full_m]

    print()
    print(f"chunks={len(rows)} full_m={full_m} total_kernel_sec={total_kernel:.3f}")
    if first_full:
        print(f"full_chunk_avg_sec={mean(first_full):.3f}")
    for r in tail:
        norm = (r["kernel"] / r["m"]) * full_m if r["m"] else 0.0
        ratio = norm / mean(first_full) if first_full else 0.0
        print(
            f'tail chunk={r["chunk"]} m={r["m"]} kernel_sec={r["kernel"]:.3f} '
            f'normalized={norm:.3f} ratio_vs_full_avg={ratio:.3f}'
        )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
