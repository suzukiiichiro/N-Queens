## 370 — mem_probe_v2: fix read_vmhwm_kb() (BUGFIX on 369's own new code)

**Status**: source + harness prepared this session, static checks (STATIC_ONLY=1)
pass locally (OK=12 FAIL=0 WARN=1[sudo, expected]). NOT yet built/run against
real codon/GPU hardware. Suzuki to run the sweep next.

### 369's real-hardware result (the trigger for 370)

The 369 sweep (default `MAX_MEM_PERCENT=70`, 10GB ulimit) ran:

- `record_limit=1,000,000` — completed
- `record_limit=5,000,000` — completed
- `record_limit=10,000,000` — completed
- `record_limit=15,000,000` — **did not complete** (no `[mem-probe-done]`
  line; sweep stopped here as designed)

**This bracket (10,000,000 OK / 15,000,000 FAIL) is real, independent
evidence that the crash scales with record count** — consistent with the
memory-exhaustion picture that's been building since the `MAX_MEM_PERCENT`
70→95 experiment (which showed the ulimit *value* wasn't the direct cause,
but didn't rule out memory pressure in general, especially given this
session's host has no swap).

However, every completed rung reported
`vmhwm_start_kb=vmhwm_after_count_kb=vmhwm_after_read_kb=vmhwm_after_soa_kb=-1`
and therefore `delta_read_kb=delta_soa_kb=delta_total_kb=0` — the VmHWM
instrumentation itself was silently non-functional throughout, so 369
produced a real threshold bracket but no usable memory numbers.

### Diagnosis

`/proc/self/status` is a procfs pseudo-file; `stat()` on it reports
`st_size=0`. 369's `read_vmhwm_kb()` used `f.read()` with no explicit size —
the same idiom `count_constellations_bin_records()` uses successfully, but
only for **real files** where `stat()` reports the correct size. Suspected:
a size-based read may preallocate a 0-byte buffer against the reported
0-byte size and return before the actual read syscall runs.

### What 370 does

**Single-function bugfix, not a new diagnostic.** The entire body of
`read_vmhwm_kb()` (369's own new function; nothing from 366/368 or earlier)
is replaced with an explicit fixed-size chunk-read loop (4096 bytes per
`read()` call, looping until an empty chunk), which does not depend on the
file's reported size at all — the standard technique for reading procfs
pseudo-files. Marked as a single `===370-VMHWM-FIX-BEGIN/END===` span (a
whole-function MODIFICATION, not a pure insertion — the harness checks this
the same way the CLI-gate 2-word edits are checked in earlier revisions,
just scaled up to a full function body).

Nothing else changes: `probe_partial_load_memory()`, the `bench_mode==36`
dispatch, the CLI/preset gates, and all of 366/368's own diagnostics remain
byte-identical to 369.

### Validation harness: `370Py_mem_probe_v2_validate_N22_sweep.sh`

Core-region check compares against **369's own full hash**
(`b977905f4f40fba74de9048cdfb4c4c325462ad74f7e724b2236e4658e7b36d2`, 6101
lines, computed this session) by reversing the single function-body swap.

**Bug found and fixed in this session's own harness code before delivery**:
the first draft's line-counting used `core2.count('\n')` inside a Python
heredoc, but the file as written contained the literal two-character escape
`\\n` at that spot (an artifact of the surrounding string literals, which
legitimately need `\\n` to represent Codon source's `"\n"`). This counted
occurrences of the substring "backslash-n" instead of real newline
characters, so the hash matched correctly but the reported line count was
wildly wrong (14 instead of 6101), causing a false `FAIL`. Fixed to
`core2.count(chr(10))`. Re-ran `STATIC_ONLY=1` after the fix: **OK=12
FAIL=0 INFO=0 WARN=1** (WARN is the expected non-fatal sudo warning).

**New check added**: `source_old_size_based_read_removed` — fails if the old
`text:str=f.read()$` (no explicit chunk size) pattern is still present
anywhere, so a future revision can't silently reintroduce the bug this one
fixes.

Section 6 (build + sweep) re-runs the same sweep mechanism as 369, with a
refined default ladder that re-confirms 369's known-good point and bisects
the 10,000,000–15,000,000 gap:

```
1000000 10000000 11000000 12000000 13000000 14000000 15000000 20000000 28719035
```

**New safety check**: after each completed rung, if
`vmhwm_after_soa_kb == -1`, the harness sets a flag and reports
`vmhwm_instrumentation_functional: FAIL` at the end — so if the fix somehow
doesn't work on real hardware either, the harness will say so explicitly
rather than silently presenting untrustworthy zeros again.

### Run instructions

```
STATIC_ONLY=1 bash 370Py_mem_probe_v2_validate_N22_sweep.sh
./367_safe_run_wrapper.sh -- bash 370Py_mem_probe_v2_validate_N22_sweep.sh
```

`MAX_MEM_PERCENT` left at its default (70), matching 369's run that produced
the bracket being narrowed here.

### How to read the result

- **If `vmhwm_instrumentation_functional` still fails** (some rung reports
  -1): report immediately; the fix hypothesis about procfs size-based reads
  was wrong or incomplete, and the delta numbers still can't be trusted.
- **If it passes**: the results table now has real `delta_total_kb` per
  rung. Compare the delta between the 10,000,000 and (whichever of
  11–15,000,000 is the last to succeed) rungs to estimate KB/record near the
  actual threshold, which is a much more reliable extrapolation basis than
  369's single-point guess would have been.
