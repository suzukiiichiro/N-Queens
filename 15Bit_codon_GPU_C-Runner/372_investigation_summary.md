# 372 — N=22 maxd-check crash investigation: summary and conclusion

**Type**: DESIGN-ONLY. No code changes. Zero-delta revision closing out the
368–371 investigation arc. Source files touched: none (366/368/369/370/371
all remain exactly as delivered).

**Status**: real-hardware findings only in this document; nothing here was
generated or verified by Claude beyond organizing what Suzuki-san ran and
reported this session.

---

## Headline result (the practical question is answered)

**N=22 has `required_maxd=14`, identical to N=21.** Confirmed 3/3 stable
via `366Py_maxd_check` (`bench_mode=34`, full 28,719,035 records):

```
[maxd-check] N=22 records=28719035 required_maxd=14 selected_maxd=14
             schedule_words=0 stack_bytes_per_thread=208
             supported=yes has_c_port=yes(maxd14)
```

**This means N=22 needs no new kernel work.** The existing C-ported
`kernel_dfs_iter_gpu_maxd14` (361–365) already covers it. Porting
`kernel_dfs_iter_gpu_maxd16/18/20/21` — anticipated as "substantial
additional work" when 366 was written — is **not required** for N=22.

368's motivating concern (that `meta_next`'s 28-element table, indexed by
`fu=raw&31` at three sites in `schedule_depth_for_task()`, might be read out
of bounds for N=22) is therefore **withdrawn as moot**: with
`required_maxd=14`, the schedule-walk never needed to visit the deeper
territory that concern was about. The out-of-bounds hypothesis was never
actually confirmed OR refuted on its own terms (368's own bounds-guarded
code was never reached, per 369/370's finding below) — it's moot because
the practical question it was meant to inform is now settled independently.

---

## What actually caused 366/368's original crashes: investigated, not fully resolved

This part of the investigation did **not** reach a fully confirmed root
cause. The chronology, honestly recorded:

| Rev | What ran | N=22 records | Result |
|---|---|---|---|
| 366 | `bench_mode=34` (real check) | 28,719,035 (full) | **segfault** (dmesg: `error 4` then `error 6`, in-binary) |
| 368 | `bench_mode=35` (bounds-guarded diag copy) | 28,719,035 (full) | **segfault**, identical instruction address to 366's 2nd crash — 368's own new code was never reached |
| 369 | `bench_mode=36` (mem-probe, `record_limit` swept) | up to 10,000,000 | sweep completed through 10M, **failed at 15,000,000** — but `read_vmhwm_kb()` was silently broken (bug), so no usable memory numbers |
| 370 | same, `read_vmhwm_kb()` fixed | up to 12,000,000 | sweep completed through 12M with real VmHWM deltas (~0.43 KB/record, near-linear); **failed at 13,000,000**, with a transient `OMP: Error #34 ... System error #11: Resource temporarily unavailable` observed at one 12M rung and a segfault (again near-identical instruction bytes to 366/368/369) confirmed via dmesg |
| 371 | fine-grain bisection + x2 repeats, no code change | 12.0M–13.0M | `record_limit=12,000,000`: 2/2 OK. `record_limit=12,200,000` through `13,000,000`: **4/4 deterministic FAIL** (repeats agreed with each other) |
| (manual) | `OMP_NUM_THREADS=1..5` on the 371 failure case (12.2M) | 12,200,000 | **all 5 succeeded** |
| (manual) | 366 (`bench_mode=34`, full 28.7M) with `OMP_NUM_THREADS=1`, then unset, then 3x unset in a row | 28,719,035 | **all runs succeeded** (5 total: 1 with `OMP_NUM_THREADS=1`, 1 unset, 3 more unset) |
| (manual) | 366 (`bench_mode=34`, full 28.7M) with ASLR disabled (`setarch -R`) | 28,719,035 | succeeded (uninformative — this case already succeeded unconditionally, so ASLR-off success here confirms nothing) |
| (manual) | 370 (`bench_mode=36`, `record_limit=12,200,000`, the deterministic-failure case) with ASLR disabled (`setarch -R`) | 12,200,000 | **succeeded** — the first condition change that flipped this specific case from FAIL to OK |

### What this data does and doesn't support

- **The crash is real, reproducible, and record-count-related** — but not
  in a simple "exceeds N bytes" way. A larger run (28.7M records,
  `bench_mode=34`) succeeded repeatedly and stably while a smaller run
  (12.2M records, `bench_mode=36`) failed deterministically (4/4). Data
  volume alone does not explain this.
- **`OMP_NUM_THREADS` does not appear to be the cause.** Setting it
  (1 through 5) made the previously-failing 12.2M case succeed every time
  it was tried — but leaving it unset also succeeded every time it was
  tried on the 28.7M case. No condition with `OMP_NUM_THREADS` unset was
  ever tested against the specific 12.2M failure case, so the two threads
  of evidence don't actually contradict each other, but they also don't
  establish `OMP_NUM_THREADS` as causal — the correlation observed earlier
  in the session was likely coincidental.
- **ASLR disabling flipped the one real failure case it was tested
  against** (12.2M, `bench_mode=36`) from FAIL to OK, in a single trial.
  This is suggestive of an address-space-layout/fragmentation-sensitive
  large allocation (consistent with the segfault's indexed-array-store
  instruction pattern seen in dmesg across 366/368/369/370, and with a
  large contiguous allocation being harder to satisfy depending on where
  ASLR happens to place existing mappings). **It is one trial, not a
  repeated/deterministic confirmation** — 371's own standard (repeat runs
  before trusting a result) was not applied here, by deliberate choice, to
  stop the investigation at a reasonable point once the practical question
  was already answered.

### Formal conclusion on root cause

**Not fully determined.** The strongest single piece of evidence points
toward an ASLR-sensitive, large-contiguous-allocation failure mode (plausibly
in Codon/its GC or in glibc's allocator, triggered by the unsized dynamic
growth of `constellations`/SoA arrays in `read_constellations_bin_range`/
`build_soa_for_range`), but this is not confirmed to the standard the rest
of this project holds itself to (single-variable discipline, repeated
confirmation, pre-registered predictions). Suzuki-san and Claude agreed to
stop pursuing further confirmation here, since the practical question this
whole arc was meant to answer (does N=22 need new kernel work?) is already
settled by the `required_maxd=14` result, independent of this open question.

**If this crash resurfaces** (e.g. when N=22 is later run through
`bench_mode=33` or further N=23+ work), the leads recorded here — ASLR
sensitivity, the specific indexed-array-store segfault signature, the
record-count non-monotonicity — are the starting point, not a solved
problem to route around by habit (e.g. don't assume "`OMP_NUM_THREADS=1`
fixes it" without retesting; that correlation did not hold up).

---

## Revisions closed by this document

- **368** (`bench_mode=35`, meta_next bounds-guard diagnostic): hypothesis
  moot given `required_maxd=14`; the bounds-guarded code path itself was
  never exercised in any run this session (all crashes occurred earlier,
  in the shared bin-load path). No further action.
- **369/370** (`bench_mode=36`, mem-probe + VmHWM instrumentation): tooling
  is sound (370 fixed the real bug in 369, verified working with real
  numbers). Retained as available diagnostic tooling for future memory
  investigations, not superseded.
- **371** (fine-grain bisection + determinism sweep): established that the
  12.0M/12.2M boundary is deterministic under back-to-back same-session
  repeats — later shown to not be deterministic across different sessions/
  conditions (ASLR). This nuance (session-local determinism vs. true
  determinism) is itself a useful methodological note for future
  crash-bisection work on this host.

## Next step (per Suzuki-san's direction)

Proceed to **N=22's actual single-shot run** (`bench_mode=33`, mirroring
365's protocol for N=21) to obtain the real solution count and compare
against the external 2023-11-22 benchmark's Total value referenced earlier
in this project's history. This is a full 28,719,035-record run through
`kernel_dfs_iter_gpu_maxd14` on real GPU hardware — the same scale that
succeeded repeatedly and stably in this session's `bench_mode=34` testing,
so no `367_safe_run_wrapper.sh`-level concern is expected, but running
under it regardless remains cheap insurance given the open root-cause
question above.
