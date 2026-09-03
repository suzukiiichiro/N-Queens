## 369 — mem-probe: VmHWM checkpoints + record_limit sweep for N=22 (DIAGNOSTIC-ONLY)

**Status**: source + harness prepared this session, static checks (STATIC_ONLY=1)
pass locally (OK=27 FAIL=0 WARN=1[sudo, expected]). NOT yet built/run against
real codon/GPU hardware. Suzuki to run the sweep next.

### Why 369 exists (368's result forced a re-think)

368 (bench_mode=35) was designed to test whether `meta_next[fu]` indexing
out of the table's 28-element bounds caused N=22's segfault. It never got
the chance: **366/bench_mode=34 and 368/bench_mode=35 both crashed at the
identical instruction address** (`sudo dmesg`: `ip=0x412a9c error 6`, same
code bytes, same binary-relative offset `[402000+44000]`), inside the
shared, unmodified bin-load path
(`count_constellations_bin_records → read_constellations_bin_range →
build_soa_for_range`) — **before** either revision's new code (schedule
walk / bounds guard) was ever reached. The `[stream-cache-hit]` line prints,
then nothing.

A follow-up single-variable test raised 367's `MAX_MEM_PERCENT` from 70 to
95 (10GB → 14GB ulimit ceiling on this session's swapless 15GB-RAM host):
**no change** — same crash, same instruction address. This ruled out the
ulimit *setting* as the direct cause, but not memory pressure in general —
this session's host has **no swap** (`Swap: 0B` in `free -h`), so `ulimit -v`
was never the only ceiling in play; the real one may simply be the 15GB of
physical RAM itself. (Earlier in the investigation, the absence of an
OOM-killer entry in `dmesg` was read as evidence against memory exhaustion —
that reasoning was corrected: `ulimit -v` failures happen silently at the
`mmap`/`brk` syscall level and never appear in `dmesg`, so their absence
proves nothing either way.)

### What 369 does

Pure additive diagnostic on top of 368 (366/368 both remain byte-identical
and reachable via `bench_mode==34`/`35`). Six marked delta spans this time
(`===369-VARINIT===`, `===369-CLIGATE-COMMENT===`, `===369-PRESETGATE-COMMENT===`,
`===369-ARGPARSE-INSERT===`, `===369-INSERT===`, `===369-DISPATCH-INSERT===`).

Two new functions:
- `read_vmhwm_kb()` — parses `/proc/self/status` for `VmHWM` (peak resident
  set size so far, in KB). Returns `-1` on any failure (safe, non-fatal).
- `probe_partial_load_memory(N, fname, record_limit, gpu_log_level)` —
  checkpoints VmHWM at four points: start, after
  `count_constellations_bin_records`, after `read_constellations_bin_range`
  (loading only `min(record_limit, total_records)` records, not necessarily
  all of them), after `build_soa_for_range`. Returns the loaded count and
  all four checkpoints.

New `bench_mode==36` dispatch branch, always prints on completion:

```
[mem-probe-done] N=... src_bin=... record_limit=... loaded=...
  vmhwm_start_kb=... vmhwm_after_count_kb=... vmhwm_after_read_kb=...
  vmhwm_after_soa_kb=... delta_read_kb=... delta_soa_kb=... delta_total_kb=...
```

`record_limit` is CLI-controlled: reused `argv[13]` slot (the same one
`bench_mode==30` uses for `debug_chunk_start` — mutually exclusive per
invocation, unambiguous). Default `1,000,000` if not supplied (a known-safe
scale — well under N=21's full 2,025,282, which has run cleanly since 361).

### Validation harness: `369Py_mem_probe_validate_N22_sweep.sh`

Static-check section modeled on 366/368's, extended for 6 spans. Core-region
hash now compares against **368's own full hash**
(`362bb7fad3c026a625e7bb276c9b2e9b2de6ed0af29dda1934980890c0b85454`, 5997
lines, computed this session) rather than 366's. Adds checks that both 366's
*and* 368's original diagnostic functions are still present verbatim (this
revision must not regress either earlier diagnostic).

**New in this harness**: section 7 is a **sweep**, not a single run. It
builds once, then invokes the binary once per rung of a fixed
`record_limit` ladder (default: `1000000 5000000 10000000 15000000 20000000
25000000 28719035` — the last value is N=22's actual full record count),
strictly increasing, stopping at the **first rung that does not complete**.
For each successful rung it parses `delta_read_kb`/`delta_soa_kb`/
`delta_total_kb` out of the `[mem-probe-done]` line and accumulates a
results table. At the end:

- If every rung (including the full 28,719,035) completes: the bin-load
  path is **not** the cause after all, and 366/368's crash must come from
  something later (kernel-launch/dispatch machinery, SoA→GPU staging, or a
  driver interaction) — next revision should instrument past this point.
- If a rung fails: the last-successful and first-failing `record_limit`
  values bracket the actual threshold, plus a rough linear
  KB-per-record extrapolation to the full count (explicitly caveated as
  order-of-magnitude only — Dict/list overhead is not guaranteed linear).

Override the ladder with `RECORD_LIMITS="..."` (space-separated) if a finer
or coarser sweep is wanted.

`STATIC_ONLY=1` run in this session: **OK=27 FAIL=0 INFO=0 WARN=1** (WARN is
the expected non-fatal `sudo -n true` failure). The delta-extraction regex
(`delta_read_kb=`, `delta_soa_kb=`, `delta_total_kb=`) was also sanity-tested
against a synthetic log line in this session and parses correctly.

### Run instructions

```
STATIC_ONLY=1 bash 369Py_mem_probe_validate_N22_sweep.sh
./367_safe_run_wrapper.sh -- bash 369Py_mem_probe_validate_N22_sweep.sh
```

Recommend keeping `MAX_MEM_PERCENT` at its default (70) for this sweep,
since the point is to find where the ceiling bites, not to raise it further
first.

### How to read the result

- **Sweep completes fully (all rungs including 28,719,035)**: the
  out-of-bounds `meta_next` hypothesis from 368 is back on the table as
  worth re-testing properly — but only after finding what actually differs
  between this read-only probe and 366/368's crashing runs (they call the
  same bin-load functions; the difference is what runs *after*). Do not
  re-run 368 as-is expecting a different result; the crash site was never
  in 368's own new code.
- **Sweep fails partway**: report the bracket (last-OK / first-fail
  `record_limit`) and the table. Next revision's direction depends on the
  bracket's position — if it's a small fraction of 28,719,035, an
  instance-size or streaming-rewrite of `read_constellations_bin_range`
  becomes the likely next step; if it's very close to the full count, a
  smaller, more targeted fix (e.g. releasing `constellations` before/while
  building `soa`) may be enough.
