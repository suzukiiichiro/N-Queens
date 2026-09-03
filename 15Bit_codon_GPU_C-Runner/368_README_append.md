## 368 — maxd-diag: bounds-safe survey of `fu` vs `meta_next` for N=22 (DIAGNOSTIC-ONLY)

**Status**: source + harness prepared this session, static checks (STATIC_ONLY=1)
pass locally (OK=24 FAIL=0 WARN=1[sudo, expected]). NOT yet built/run against
real codon/GPU hardware — no codon toolchain available in this environment.
Suzuki to run for N=22 next.

### Trigger

366 (bench_mode=34) segfaulted on real hardware for N=22
(records=28,719,035, preset_queens=7). `sudo dmesg` showed:

```
366Py_maxd_chec[57658]: segfault at 0 ... error 4 in 366Py_maxd_check[...]
366Py_maxd_chec[57965]: segfault at 0 ... error 6 in 366Py_maxd_check[...]
```

No OOM-killer entry, no allocation-failure message — ruling out the initial
memory-exhaustion hypothesis (367's 10GB ulimit ceiling on this session's
15GB-total host) as the direct cause of the crash itself (367's ceiling may
still be worth keeping as a general safety net, but it isn't what killed this
run).

### Hypothesis

`schedule_depth_for_task()` indexes `meta_next[fu]` at three call sites,
where `fu = raw&31` (0..31 by construction, a 5-bit mask). The `meta_next`
table built inside `check_required_maxd_for_N()` has exactly 28 elements
(indices 0..27). Nothing bounds-checks `fu < len(meta_next)` before the
three `meta_next[fu]` reads. N=21 apparently never produces `fu>=28`; N=22
(different preset_queens, different constellation/task shape) may.

### What 368 does

Pure additive diagnostic, byte-identical to 366 outside 4 marked spans
(`===368-INSERT===`, `===368-CLIGATE-COMMENT===`, `===368-PRESETGATE-COMMENT===`,
`===368-DISPATCH-INSERT===`). Does **not** touch 366's `check_required_maxd_for_N()`
or `schedule_depth_for_task()` — both remain reachable unchanged via
`bench_mode==34`.

Adds a parallel, bounds-safe **copy** of the schedule-walk:
- `class MaxdDiagStats` — accumulator (fu_min, fu_max, oob_count, first-occurrence
  repro: task_index/fu/ctrl0/markctrl, tasks_scanned).
- `schedule_depth_for_task_diag(...)` — same logic as the original, but checks
  `fu>=META_NEXT_LEN` before each of the 3 `meta_next[fu]` sites; on hit,
  records stats and returns a diagnostic sentinel (23/24/25, distinct from the
  original's existing 22-sentinel) instead of indexing out of bounds.
- `scan_maxd_diag_for_tasks(...)` — drives the diag walk over all `m` tasks.
- `check_required_maxd_for_N_diag(...)` — mirrors `check_required_maxd_for_N`
  (same bin loading / SoA build / same literal `meta_next` table), calls the
  diag scan, optionally verbose-prints `[maxd-diag]`.

New `bench_mode==35` dispatch branch (CLI whitelist + preset gate updated
preemptively, per the 361-r1 lesson already applied in 365/366). Always
prints `[maxd-diag-done] N=... records=... fu_min=... fu_max=... oob_count=...
first_oob_task_index=... first_oob_fu=... first_oob_ctrl0=... first_oob_markctrl=...`
on completion — this is the line the `.sh` harness greps for.

### Validation harness: `368Py_maxd_diag_validate_N22_once.sh`

Modeled directly on `366Py_maxd_check_validate_N22_once.sh`. Core-region
hash check now compares against **366's own full hash**
(`03fed068544019ffe1650ab3ba8bfdc4e95880eda20f9eed0bc163524935474f`,
5832 lines — 366's own docstring/VERSION_TAG-stripped source, computed this
session), not 365's. Added checks beyond the 366 pattern:
- `source_366_original_function_still_intact` — 366's `check_required_maxd_for_N`
  signature must still be present verbatim (368 must not remove/rename it).
- `source_diag_has_three_bounds_guards` — exactly 3 occurrences of
  `if fu>=META_NEXT_LEN:` inside the 368-INSERT span (one per `meta_next[fu]`
  call site).
- `negtest_guard_not_vacuous` — sanity check that `meta_next_len (28) <= 31`
  (max possible `fu`), so the guard can actually fire; would catch a future
  edit that widened `meta_next` to 32 elements and silently defeated the
  diagnostic.

`STATIC_ONLY=1` run in this session: **OK=24 FAIL=0 INFO=0 WARN=1** (WARN is
the expected non-fatal `sudo -n true` failure).

### Run instructions

```
STATIC_ONLY=1 bash 368Py_maxd_diag_validate_N22_once.sh
./367_safe_run_wrapper.sh -- bash 368Py_maxd_diag_validate_N22_once.sh
```

Recommend running under 367 as before — bin generation for N=22 (unchanged
from 366) is still resource-heavy if the bin was lost in the EBS reset.

### How to read the result

- **`oob_count==0`**: the out-of-bounds hypothesis is **refuted** for N=22.
  Do not proceed to a `meta_next`-widening fix; the segfault cause is still
  open and needs a different lead (e.g. re-examine `build_soa_for_range` for
  N=22, or bisect further).
- **`oob_count>0`**: the hypothesis is **confirmed**. `first_oob_task_index`
  / `first_oob_fu` / `first_oob_ctrl0` / `first_oob_markctrl` give a concrete
  repro case. The actual correctness fix (extending/reworking `meta_next`,
  or determining N=22 needs a structurally different table) is **out of
  scope for 368** — next revision, once the repro is in hand.
- If 368 itself crashes on N=22 (it shouldn't — that's the point of the
  bounds guard): that's significant new information on its own. Run
  `sudo dmesg | tail -30` immediately and report it; the out-of-bounds
  hypothesis would then need to be reconsidered too.
