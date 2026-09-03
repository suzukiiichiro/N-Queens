## 371 — fine-grain bisection + determinism-repeat sweep (DESIGN-ONLY, zero code change)

**Status**: harness prepared this session, static check (STATIC_ONLY=1)
passes locally (OK=3 FAIL=0 WARN=1[sudo, expected]). Sweep+determinism
logic dry-tested end-to-end against a fake `codon`/binary stub in this
session (not real hardware) — confirmed the per-run table and the
per-value determinism comparison both work correctly, including the
FAIL-vs-OK branching. Suzuki to run for real next.

### Why 371 is design-only

370's sweep found: near-linear `delta_total_kb` growth from 10M→12M
records (~0.43 KB/record), a transient `ValueError: invalid format
specifier` right at the 12M rung (right after its own `[mem-probe-done]`
printed), and a clean non-completion at 13M. Extrapolating the 10M–12M
slope to 13M would still leave several GB of headroom under the 10GB
ceiling — the near-linear steady-state number does NOT predict a hard wall
at 13M. dmesg confirmed 369 and 370 both segfaulted at nearly identical
instruction bytes, in what looks like an indexed array-store
(`[reg + reg*4]`) instruction.

Working hypothesis: not simply "steady-state memory exceeded 10GB" but a
**transient spike** — most likely a dynamic list reallocation (doubling
growth) that briefly requests much more than the eventual steady state,
and that transient request is what hits the ceiling.

371 tests this using **only the harness** — 370Py_mem_probe_v2.py is reused
completely unmodified (raw sha256 identity-checked:
`8152a0ef550f1c6d0ab4b949d8ba25c973117e8fa99f9d1510abd87fbf68f13d`). No new
marker spans, no core-hash-with-deltas-stripped procedure (there are no
deltas) — just a raw file-identity check, which is a stronger guarantee for
a genuinely zero-change revision.

### What the harness does differently

1. **Bisects the 12,000,000/13,000,000 gap**: default ladder
   `12000000 12200000 12500000 12800000 13000000` (the two endpoints
   re-included as anchors to reconfirm 370's result under the new
   protocol).
2. **Runs every rung twice in a row** (`REPEATS=2` by default) and
   explicitly compares completion status between the two runs of the same
   `record_limit`. If a value's repeats disagree (one OK, one FAIL), the
   harness flags it as `NON-DETERMINISTIC` — direct evidence for a
   timing/allocation-race explanation over a fixed hard threshold, which
   would point toward a different kind of fix than a pure capacity problem.

Output includes a per-run table (`record_limit / rep / status /
delta_total_kb`) and a per-value determinism verdict, ending with an
`INFO`-level summary (`nondeterminism_detected` or `all_values_deterministic`)
rather than a pass/fail judgment, since either outcome is informative, not
a defect.

### Run instructions

```
STATIC_ONLY=1 bash 371_finegrain_determinism_sweep.sh
./367_safe_run_wrapper.sh -- bash 371_finegrain_determinism_sweep.sh
```

Needs `370Py_mem_probe_v2.py` present in the working directory (unchanged
from 370's delivery). `MAX_MEM_PERCENT` left at 370's default (70).

### How to read the result

- **All values deterministic, threshold still sharp near a fixed point**:
  supports a genuine fixed-capacity explanation after all (perhaps the
  transient spike itself is deterministic given fixed input data) — next
  step would be sizing/estimating that spike directly (e.g. pre-allocating
  `constellations`/SoA arrays to the known record count instead of
  growing them dynamically, which is the standard fix for reallocation-
  spike problems and would need its own revision).
- **Some value(s) non-deterministic**: strengthens the reallocation-timing-
  race hypothesis; report which value(s) and their exact OK/FAIL pattern.
  This changes the fix direction — a race isn't resolved by raising a
  ceiling or picking a "safe" `record_limit", since the same value could
  fail on a different run.
