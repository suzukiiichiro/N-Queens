## 366 — maxd-check: cheap required_maxd diagnostic for N=22+ (CHECK-ONLY)

**Mode**: check-only. No GPU upload, no kernel launch, no solution total.

New `bench_mode=34`, new function `check_required_maxd_for_N()` — reuses
existing unmodified functions (`ensure_constellations_bin_stream`,
`build_soa_for_range`, `max_schedule_depth_of_tasks`, `select_static_maxd`)
to report `required_maxd`/`selected_maxd` for a given N before committing to
any further kernel-porting work. Motivation: `kernel_dfs_iter_gpu_maxd14`
(361–365) only covers `required_maxd<=14`; N=22+ was suspected to need more.

**Real-hardware history this session**:
- First attempt (`bench_mode=34`, full N=22, 28,719,035 records) —
  **segfault** (`dmesg`: `error 4` then `error 6`, in-binary). Initially
  misattributed to OOM (367's 10GB ulimit); later shown by dmesg to be a
  segfault, not an OOM-killer event, and ulimit changes made no difference.
- Root cause investigated across 368–372 (see below). **Final result (372):
  `required_maxd=14`, confirmed 3/3 stable** once the underlying crash
  stopped reproducing partway through the investigation — see 372 for the
  full, honest account of what was and wasn't resolved.

---

## 367 — safe_run_wrapper.sh: generic host-safety guard (INFRASTRUCTURE, not a numbered code revision)

Written after an N=23 incident: 366's on-demand constellation-bin
generation for N=23 grew unboundedly, exhausted host memory, and made the
host unresponsive over SSH and serial console, requiring a force-stop that
destroyed an ephemeral instance-store volume (bins lost; source safe in
Git). Pure shell-level guard, no Codon source changes — deliberately kept
outside the 361–372 single-variable code-delta sequence.

Three independent, always-on guards: memory ceiling (`ulimit -v`, default
70% of total system RAM), wall-clock timeout (`timeout`, default 2h), and a
live disk-space monitor (default floor 20GB, checked every 10s while the
wrapped command runs). Usage: `367_safe_run_wrapper.sh -- <command>
[args...]`.

**Note for this host**: 15GB total RAM, **no swap**. `ulimit -v` failures
are silent at the syscall level and never appear in `dmesg` — their absence
does not rule out memory pressure (this was initially misjudged, then
corrected, during the 368–370 investigation).

---

## 368 — maxd-diag: bounds-safe meta_next survey (DIAGNOSTIC-ONLY)

**Mode**: diagnostic-only, no GPU/kernel. New `bench_mode=35`.

Motivation: `schedule_depth_for_task()` indexes `meta_next[fu]` at three
sites, where `fu=raw&31` (0–31 by construction) against a 28-element table
(indices 0–27) — a structural range mismatch N=21 never triggers but N=22
was suspected to. 368 added a parallel, bounds-safe copy of the schedule
walk (`MaxdDiagStats`, `schedule_depth_for_task_diag`,
`scan_maxd_diag_for_tasks`, `check_required_maxd_for_N_diag`) that records
`fu_min`/`fu_max`/`oob_count`/first-occurrence repro instead of crashing on
an out-of-bounds access. 366's original functions left untouched.

**Real-hardware result**: crashed at the **identical instruction address**
as 366's crash (`dmesg`: `ip=0x412a9c`, `error 6`), inside the shared,
unmodified bin-load path — **before 368's own new bounds-guarded code was
ever reached**. The meta_next hypothesis was never actually tested by this
revision. (Later shown moot anyway once `required_maxd=14` was confirmed —
see 372.)

---

## 369 — mem_probe: VmHWM checkpoints + record_limit sweep (DIAGNOSTIC-ONLY)

**Mode**: diagnostic-only. New `bench_mode=36`.

Since 366 and 368 both crashed in the shared bin-load path
(`count_constellations_bin_records → read_constellations_bin_range →
build_soa_for_range`) before reaching either revision's own new code, 369
instrumented that shared path directly: `read_vmhwm_kb()` (parses
`/proc/self/status` VmHWM, peak RSS) and `probe_partial_load_memory()`
(checkpoints before/after each load stage), with the loaded record count
exposed as a CLI parameter (`record_limit`, reusing the `argv[13]` slot
`bench_mode==30` uses for `debug_chunk_start`). Validation harness sweeps a
record-count ladder, one process per rung, stopping at the first
non-completing rung.

**Real-hardware result**: sweep completed through `record_limit=10,000,000`,
failed at `15,000,000` — a real, record-count-related bracket. **But**
`read_vmhwm_kb()` was silently broken (`vmhwm_*_kb=-1` on every rung, all
deltas 0) — the instrumentation itself didn't work, even though the
completion/failure bracket it happened to produce was real.

---

## 370 — mem_probe_v2: fix read_vmhwm_kb() (BUGFIX on 369's own new code)

**Mode**: bugfix, single function. Nothing from 366/368/369-shared-code
touched.

Diagnosis: `/proc/self/status` is a procfs pseudo-file reporting
`st_size=0` via `stat()`; 369's size-based `f.read()` (the same idiom that
works for **real** files elsewhere in the codebase) likely preallocated a
0-byte buffer and returned before the actual read syscall ran. Fix: replace
the entire body of `read_vmhwm_kb()` with an explicit fixed-size chunk-read
loop (4096 bytes/call, looping to EOF) that doesn't depend on reported file
size.

**Session note**: a bug in this revision's own validation-harness code
(`core2.count('\n')` accidentally written as `core2.count('\\n')`, counting
literal backslash-n occurrences instead of real newlines) caused a false
`FAIL` on an otherwise-correct hash match; caught and fixed
(`core2.count(chr(10))`) before delivery.

**Real-hardware result**: VmHWM instrumentation now returns real numbers.
Sweep (ladder re-run + bisected: 1M/10M/11M/12M/13M+) showed near-linear
`delta_total_kb` growth from 10M→12M (~0.43 KB/record), a transient `OMP:
Error #34 ... Resource temporarily unavailable` at the 12M rung (right
after that rung's own success), and a clean non-completion at 13M. `dmesg`
confirmed a real segfault, again at near-identical instruction bytes to
366/368/369 (an indexed array-store instruction pattern), not an OOM-killer
event.

---

## 371 — fine-grain bisection + determinism-repeat sweep (DESIGN-ONLY, zero code change)

**Mode**: harness-only. Reuses 370's `.py` completely unmodified (raw
sha256 identity-checked instead of the usual marker/hash-delta procedure,
since there is no code delta).

Bisected the 12.0M/13.0M gap (`12.0M/12.2M/12.5M/12.8M/13.0M`) and ran every
rung **twice in a row**, comparing completion status between repeats, to
test whether the failure was a fixed threshold or a timing/allocation race.

**Real-hardware result**: `12,000,000` — 2/2 OK. `12,200,000` through
`13,000,000` — **4/4 deterministic FAIL** (repeats agreed with each other
within the session). This ruled out a same-session timing race — but see
372: this session-local determinism did **not** hold up across different
sessions/conditions (manual `OMP_NUM_THREADS` and ASLR trials afterward).

---

## 372 — N=22 crash investigation: summary and conclusion (DESIGN-ONLY)

**Mode**: documentation only. No code changes; closes out the 368–371
investigation arc. Full detail in `372_investigation_summary.md`.

**Headline result — the practical question is answered**: N=22 has
`required_maxd=14`, identical to N=21, confirmed 3/3 stable via 366
(`bench_mode=34`, full 28,719,035 records). **No new kernel-porting work
(maxd16/18/20/21) is needed for N=22.** 368's meta_next out-of-bounds
concern is withdrawn as moot — the schedule walk never needed the depth
that concern was about.

**Root cause of the original crashes — investigated, not fully resolved**,
recorded honestly rather than papered over:
- Not a simple "exceeds N bytes" story: a **larger** run (28.7M records,
  `bench_mode=34`) succeeded repeatedly and stably while a **smaller** run
  (12.2M records, `bench_mode=36`) failed deterministically (4/4 within one
  session).
- `OMP_NUM_THREADS` does **not** appear causal on closer inspection — the
  apparent correlation from earlier manual trials didn't hold up once the
  two failure/success conditions were compared on equal footing.
- **ASLR disabling flipped the one real deterministic-failure case it was
  tested against** (12.2M) from FAIL to OK — the strongest single lead, but
  based on **one trial**, not the repeated confirmation this project
  otherwise requires before trusting a result. Deliberately not chased
  further this session, since the practical (kernel-coverage) question was
  already settled independently.
- Consistent thread across 366/368/369/370's crashes: dmesg always showed a
  real segfault (never an OOM-killer event) at near-identical instruction
  bytes, resembling an indexed array-store — consistent with a large,
  ASLR-sensitive contiguous allocation failing intermittently, most likely
  inside Codon's runtime/GC or glibc's allocator during the unsized dynamic
  growth of `constellations`/SoA arrays.

**Revisions closed by 372**: 368 (hypothesis moot), 369/370 (tooling sound,
retained for future memory investigations), 371 (session-local determinism
confirmed, but shown non-general afterward — a methodological note for
future crash-bisection work on this host).

**Next step**: proceed to N=22's actual single-shot run (`bench_mode=33`,
mirroring 365's N=21 protocol) for the real solution count, at the same
28.7M-record scale that succeeded repeatedly and stably in this session's
`bench_mode=34` testing.
