#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
114Py weekend full-run collector.

Usage:
  python3 114py_collect_weekend_results.py
  python3 114py_collect_weekend_results.py --top 30 --glob 'progress_N22_7_stream_funcid_reorder_v2_*broadmarktail_v4_*_gpu.tsv'

Outputs:
  114py_weekend_summary.tsv
  114py_weekend_top_chunks.tsv

This script is intentionally stdlib-only so it can run on the Codon/GPU host.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple

EXPECTED_N22 = 2691008701644

VARIANT_TAGS = {
    "0": "v2base",
    "1": "phase_only",
    "2": "rotate_only",
    "3": "wide_only",
    "4": "phase_rotate",
    "5": "wide_phase_rotate",
}


def parse_duration_ms(text: str) -> Optional[int]:
    """Parse '3:35:30.017' or '1 day, 0:43:10.509' to milliseconds."""
    s = text.strip()
    if not s:
        return None
    days = 0
    if "," in s:
        left, right = s.split(",", 1)
        m = re.search(r"(\d+)\s+day", left)
        if m:
            days = int(m.group(1))
        s = right.strip()
    parts = s.split(":")
    if len(parts) != 3:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        sec_part = parts[2]
        if "." in sec_part:
            sec_s, ms_s = sec_part.split(".", 1)
            seconds = int(sec_s)
            ms = int((ms_s + "000")[:3])
        else:
            seconds = int(sec_part)
            ms = 0
    except ValueError:
        return None
    return (((days * 24 + hours) * 60 + minutes) * 60 + seconds) * 1000 + ms


def fmt_ms(ms: Optional[float]) -> str:
    if ms is None:
        return ""
    if isinstance(ms, float) and math.isnan(ms):
        return ""
    return f"{ms:.3f}"


def pct(values: List[int], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def to_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        v = row.get(key, "")
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def infer_variant_from_name(path: Path) -> str:
    name = path.name
    m = re.search(r"broadmarktail_v4_([^_]+(?:_[^_]+)*)_gpu(?:_worker\d+of\d+)?\.tsv$", name)
    if m:
        tag = m.group(1)
        # The greedy expression may include run tag pieces if the filename format changes.
        for known in sorted(VARIANT_TAGS.values(), key=len, reverse=True):
            if known in tag:
                return known
        return tag
    m = re.search(r"broadmarktail_reorder_v4_([^_]+(?:_[^_]+)*)_w\d+_j\d+_b\d+_m\d+_s\d+_sim\.tsv$", name)
    if m:
        tag = m.group(1)
        for known in sorted(VARIANT_TAGS.values(), key=len, reverse=True):
            if known in tag:
                return known
        return tag
    # Fallback: look for known tag in filename.
    for tag in sorted(VARIANT_TAGS.values(), key=len, reverse=True):
        if tag in name:
            return tag
    return "unknown"


def find_matching_log(variant: str, log_glob: str) -> Optional[Path]:
    candidates = [Path(p) for p in glob.glob(log_glob)]
    scored: List[Tuple[int, Path]] = []
    for p in candidates:
        name = p.name
        score = 0
        if variant != "unknown" and variant in name:
            score += 10
        if "single" in name or "32x484" in name:
            score += 2
        if "sim" in name or "build" in name:
            score -= 3
        if score > 0:
            scored.append((score, p))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)
    return scored[0][1]


def parse_log(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(errors="replace")
    out: Dict[str, str] = {"log_file": path.name}
    # Final result line, e.g.:
    # 22:     2691008701644                0          3:35:30.017    ok
    matches = re.findall(r"^\s*(\d+):\s+(\d+)\s+\d+\s+(.+?)\s+([A-Za-z_\-0-9()=!]+)\s*$", text, flags=re.M)
    if matches:
        n, total, elapsed, status = matches[-1]
        out.update({
            "log_N": n,
            "log_total": total,
            "log_elapsed": elapsed.strip(),
            "log_elapsed_ms": str(parse_duration_ms(elapsed.strip()) or ""),
            "log_status": status,
        })
    m = re.search(r"broadmarktail_variant:\s+id=(\d+)\s+tag=([^\s]+)\s+desc=(.+)", text)
    if m:
        out["variant_id"] = m.group(1)
        out["variant_tag_from_log"] = m.group(2)
        out["variant_desc"] = m.group(3).strip()
    m = re.search(r"broadmarktail_params:\s+version=([^\s]+)\s+variant=(\d+)\s+tag=([^\s]+)\s+window_boost=(\d+)\s+phase_mix=(\d+)\s+rotate_interleave=(\d+)", text)
    if m:
        out.update({
            "bt_version": m.group(1),
            "variant_id": m.group(2),
            "variant_tag_from_params": m.group(3),
            "window_boost": m.group(4),
            "phase_mix": m.group(5),
            "rotate_interleave": m.group(6),
        })
    return out


def summarize_progress(path: Path, top_n: int, log_glob: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    rows = read_tsv(path)
    variant = infer_variant_from_name(path)
    log_path = find_matching_log(variant, log_glob)
    log_info = parse_log(log_path)

    elapsed = [to_int(r, "elapsed_ms") for r in rows]
    elapsed = [x for x in elapsed if x >= 0]
    last = rows[-1] if rows else {}
    totals = [to_int(r, "gpu_total") for r in rows]
    chunk_totals = [to_int(r, "chunk_total") for r in rows]
    max_idx = max(range(len(rows)), key=lambda i: to_int(rows[i], "elapsed_ms")) if rows else -1
    max_row = rows[max_idx] if max_idx >= 0 else {}

    total_last = totals[-1] if totals else 0
    status = "ok" if total_last == EXPECTED_N22 else "ng"
    if log_info.get("log_status"):
        status = log_info["log_status"]

    summary: Dict[str, str] = {
        "variant": variant,
        "variant_id": log_info.get("variant_id", ""),
        "variant_desc": log_info.get("variant_desc", ""),
        "progress_file": path.name,
        "log_file": log_info.get("log_file", ""),
        "rows": str(len(rows)),
        "chunks": str(len(rows)),
        "total_last": str(total_last),
        "expected": str(EXPECTED_N22),
        "status": status,
        "log_elapsed": log_info.get("log_elapsed", ""),
        "log_elapsed_ms": log_info.get("log_elapsed_ms", ""),
        "chunk_ms_sum": str(sum(elapsed)) if elapsed else "",
        "chunk_ms_mean": fmt_ms(mean(elapsed)) if elapsed else "",
        "chunk_ms_median": fmt_ms(median(elapsed)) if elapsed else "",
        "chunk_ms_p90": fmt_ms(pct(elapsed, 0.90)),
        "chunk_ms_p95": fmt_ms(pct(elapsed, 0.95)),
        "chunk_ms_p99": fmt_ms(pct(elapsed, 0.99)),
        "chunk_ms_max": str(max(elapsed)) if elapsed else "",
        "chunks_ge_8000ms": str(sum(1 for x in elapsed if x >= 8000)),
        "chunks_ge_9000ms": str(sum(1 for x in elapsed if x >= 9000)),
        "chunks_ge_10000ms": str(sum(1 for x in elapsed if x >= 10000)),
        "top_chunk": max_row.get("chunk", ""),
        "top_elapsed_ms": max_row.get("elapsed_ms", ""),
        "last_done_records": last.get("done_records", ""),
        "last_total_records": last.get("total_records", ""),
        "window_boost": log_info.get("window_boost", ""),
        "phase_mix": log_info.get("phase_mix", ""),
        "rotate_interleave": log_info.get("rotate_interleave", ""),
    }

    # Include a few key counts from the slowest row.
    for key in ["funcid_4_count", "funcid_5_count", "funcid_7_count", "funcid_17_count", "funcid_19_count", "funcid_22_count", "funcid_23_count", "funcid_24_count", "score_avg", "free_popcount_avg", "tail_funcid17_count", "tail_cell_G_H_count", "tail_proxy_avg"]:
        if key in max_row:
            summary[f"top_{key}"] = max_row.get(key, "")

    top_rows: List[Dict[str, str]] = []
    if rows:
        ranked = sorted(rows, key=lambda r: to_int(r, "elapsed_ms"), reverse=True)[:top_n]
        for rank, r in enumerate(ranked, 1):
            out = {
                "variant": variant,
                "rank": str(rank),
                "progress_file": path.name,
                "chunk": r.get("chunk", ""),
                "off": r.get("off", ""),
                "m": r.get("m", ""),
                "elapsed": r.get("elapsed", ""),
                "elapsed_ms": r.get("elapsed_ms", ""),
                "chunk_total": r.get("chunk_total", ""),
                "gpu_total": r.get("gpu_total", ""),
                "score_avg": r.get("score_avg", ""),
                "free_popcount_avg": r.get("free_popcount_avg", ""),
                "depth_avg": r.get("depth_avg", ""),
                "funcid_4_count": r.get("funcid_4_count", ""),
                "funcid_5_count": r.get("funcid_5_count", ""),
                "funcid_7_count": r.get("funcid_7_count", ""),
                "funcid_17_count": r.get("funcid_17_count", ""),
                "tail_funcid17_count": r.get("tail_funcid17_count", ""),
                "tail_cell_G_H_count": r.get("tail_cell_G_H_count", ""),
                "tail_proxy_avg": r.get("tail_proxy_avg", ""),
            }
            top_rows.append(out)
    return summary, top_rows


def write_tsv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="progress_N22_7_stream_funcid_reorder_v2_*broadmarktail_v4_*_gpu*.tsv", help="progress TSV glob")
    ap.add_argument("--log-glob", default="114Py_N22*.log", help="114Py log glob")
    ap.add_argument("--top", type=int, default=20, help="top slow chunks per variant")
    ap.add_argument("--summary", default="114py_weekend_summary.tsv")
    ap.add_argument("--top-out", default="114py_weekend_top_chunks.tsv")
    args = ap.parse_args()

    progress_files = sorted(Path(p) for p in glob.glob(args.glob))
    if not progress_files:
        print(f"No progress files matched: {args.glob}")
        return

    summaries: List[Dict[str, str]] = []
    top_rows: List[Dict[str, str]] = []
    for p in progress_files:
        summary, tops = summarize_progress(p, args.top, args.log_glob)
        summaries.append(summary)
        top_rows.extend(tops)

    # Best first by chunk_ms_sum when available, otherwise filename order.
    def sort_key(row: Dict[str, str]) -> Tuple[int, int, str]:
        try:
            return (0, int(row.get("chunk_ms_sum", "999999999999")), row.get("variant", ""))
        except Exception:
            return (1, 999999999999, row.get("variant", ""))

    summaries.sort(key=sort_key)
    write_tsv(Path(args.summary), summaries)
    write_tsv(Path(args.top_out), top_rows)

    print(f"wrote {args.summary} ({len(summaries)} variants)")
    print(f"wrote {args.top_out} ({len(top_rows)} rows)")
    print("\nsummary:")
    for r in summaries:
        print(
            f"{r.get('variant',''):18s} status={r.get('status',''):>4s} "
            f"total={r.get('total_last','')} chunk_sum_ms={r.get('chunk_ms_sum','')} "
            f"mean={r.get('chunk_ms_mean','')} p99={r.get('chunk_ms_p99','')} max={r.get('chunk_ms_max','')} "
            f">=9s={r.get('chunks_ge_9000ms','')} log={r.get('log_elapsed','')}"
        )


if __name__ == "__main__":
    main()
