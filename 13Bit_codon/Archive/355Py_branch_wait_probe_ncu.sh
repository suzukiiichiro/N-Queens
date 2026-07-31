#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# 355 branch/wait ncu probe (NO CODE CHANGE -- pure profiling procedure)
#
# WHAT 355 IS. No source edit. This script drives ncu against the unmodified
# 352 kernel to return to Open Objectives item 11, deferred through 353
# (workload census -> lane_tail 1.14-1.16x, ruled out the CUDA C port on
# imbalance grounds) and 354 (offline key search -> no candidate beat the
# current sort key). What is left is the divergence itself: wait at 45.6% and
# branch_resolving at 20.1% of N18 K1 stall samples, together 66%.
#
# TWO SEPARATE QUESTIONS, TWO SEPARATE SCOPES.
#   STRUCTURAL (which SASS instructions the wait stall attributes to):
#     taken at N=18, --section SourceCounters, ~9s, because registers/thread
#     and SASS structure are compile-time properties confirmed N-independent
#     in 352 (45 registers on both N=18 and N=21 for this kernel).
#   DYNAMIC (does the 66% figure hold at true production shape, and is
#     eligible-warp scarcity or something else the binding constraint):
#     taken at N=21, --section SchedulerStats --section WarpStateStats,
#     --launch-count 1 (chunk 0, launch 0 only), because N=18's small task
#     count runs K=1 (one constellation per thread) while N=21 production
#     runs K=48 (forty-eight constellations per thread, back to back through
#     the grid-stride loop) -- the same K-batching that 353 showed averages
#     away lane-to-lane workload imbalance can also change the DYNAMIC stall
#     mix, so N=18's ratios are not assumed to transfer and are retaken here.
#
# WHY NOT --section SourceCounters AT N=21: 5-pass replay, ~146s per pass per
# launch (352's own measurement), infeasible in reasonable time even scoped to
# one launch. SchedulerStats and WarpStateStats are single-pass hardware
# counter sections and do not carry that cost.
#
# PERMISSION CHECK FIRST, ALWAYS. 352 lost ~14 minutes by running two long ncu
# commands before confirming `sudo` worked non-interactively. This script
# checks `sudo -n` before any of the timed commands.
#
# NO ORACLE. No code changed, so there is no correctness value and no timing
# baseline to compare against. The only correctness claim this revision makes
# is that the kernel diff against 352 is empty; the harness proves that
# statically before touching ncu at all.
#
# NO FRAGILE TEXT PARSING. This script does not attempt to extract ncu's
# numeric output into summary.tsv beyond checking that the expected section
# headers are present. ncu's exact text formatting was not verified against a
# real run before this script was written, and treating an unverified format
# guess as ground truth is the same failure class as 345 r1 / 349 r1 / 354's
# comment-token near-miss: prose (or here, an assumed format) mistaken for
# something load-bearing. Raw logs are captured in full for manual reading.
#
# RUN ORDER.
#     STATIC_ONLY=1 bash 355Py_branch_wait_probe_ncu.sh
#     bash 355Py_branch_wait_probe_ncu.sh
#
# Send back the logdir (tar it) and I will read the raw ncu output with you.
# =============================================================================

SRC=${SRC:-./355Py_branch_wait_probe.py}
CAND=${CAND:-./355Py_branch_wait_probe}
AUTO_BUILD=${AUTO_BUILD:-1}
FORCE_REBUILD=${FORCE_REBUILD:-1}
STATIC_ONLY=${STATIC_ONLY:-0}
LOG_ROOT=${LOG_ROOT:-.}
LOCK_FILE=${LOCK_FILE:-/tmp/355Py_branch_wait_probe.lock}

N18=${N18:-18}
N21=${N21:-21}
BLOCK=${BLOCK:-32}
MAX_BLOCKS=${MAX_BLOCKS:-484}
LOG_LEVEL=${LOG_LEVEL:-1}
SORT_MODE=${SORT_MODE:-0}
PRESET_QUEENS=${PRESET_QUEENS:-7}
BENCH_MODE=${BENCH_MODE:-31}
REORDER_WINDOW_MULT=${REORDER_WINDOW_MULT:-3}
REORDER_PHASE_JUMP=${REORDER_PHASE_JUMP:-7}
CROSS_STRIPE_SAFE=${CROSS_STRIPE_SAFE:-0}
WORKER_ID=${WORKER_ID:-0}
WORKER_COUNT=${WORKER_COUNT:-1}
BROADMARK_VARIANT=${BROADMARK_VARIANT:-2}
CHUNKSHAPE148_BUCKET_RUN=${CHUNKSHAPE148_BUCKET_RUN:-2048}
CHUNKSHAPE148_ITER_SORT=${CHUNKSHAPE148_ITER_SORT:-9}

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="${LOG_ROOT%/}/355Py_branch_wait_probe_logs_${TS}"
SUMMARY="$LOGDIR/summary.tsv"
mkdir -p "$LOGDIR"
printf 'check\texpected\tactual\tstatus\n' > "$SUMMARY"

echo "[start] 355 branch/wait ncu probe"
echo "[source] $SRC"
echo "[candidate] $CAND"
echo "[logdir] $LOGDIR"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo "[abort] rc=$rc logdir=${LOGDIR:-unknown}" >&2; fi' EXIT

record_check() {
  local name=$1 expected=$2 actual=$3 status=FAIL
  if [[ "$actual" == "$expected" ]]; then status=OK; fi
  printf '%s\t%s\t%s\t%s\n' "$name" "$expected" "$actual" "$status" >> "$SUMMARY"
  [[ "$status" == OK ]]
}
failures=0
static_failures=0

if [[ ! -f "$SRC" ]]; then
  echo "[error] source not found: $SRC" >&2
  exit 66
fi

if grep -q '355 branch-wait-probe' "$SRC"; then
  printf 'source_version_tag\t355 branch-wait-probe\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_version_tag\t355 branch-wait-probe\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi

# ---- 355: prove NO code change at all (stricter than a diff fingerprint) ----
# 355 touches no logic, so the bar is full byte identity to 352 once
# docstrings and the two prose constants are stripped -- not "only these
# lines differ" like 353/354, but "nothing differs". FAILS rather than skips
# if 352 is absent, same reasoning as every prior identity check in this
# project: a check that silently skips is worse than no check.
REF352=${REF352:-./352Py_record_fix.py}
if [[ -f "$REF352" ]]; then
  IDENT=$(python3 - "$SRC" "$REF352" <<'PYIDENT355'
import re, sys
def code_only(p):
    s = open(p, encoding='utf-8').read()
    s = re.sub(r'"""[\s\S]*?"""', '', s)
    s = re.sub(r'^(VERSION_TAG|WHI_ELIM_REASON):str="[^"]*"$', '', s, flags=re.M)
    return s.encode('utf-8')
a = code_only(sys.argv[1]); b = code_only(sys.argv[2])
print('identical' if a == b else 'differs_%d_vs_%d_bytes' % (len(a), len(b)))
PYIDENT355
)
  record_check source_code_identical_to_352 identical "${IDENT:-compare_failed}" || static_failures=$((static_failures+1))
else
  printf 'source_code_identical_to_352\tidentical\t352 source not found at %s\tFAIL\n' "$REF352" >> "$SUMMARY"
  static_failures=$((static_failures+1))
fi

QUOTE_BAD=$(awk '/^[A-Za-z_][A-Za-z0-9_]*:str="/ { n=gsub(/"/,"\""); if (n!=2) bad++ } END { print bad+0 }' "$SRC")
record_check source_str_literal_quote_balance 0 "$QUOTE_BAD" || static_failures=$((static_failures+1))


if (( static_failures != 0 )); then
  echo "================================================================"
  echo "[static-summary]"
  cat "$SUMMARY"
  echo "[logdir] $LOGDIR"
  echo "[error] 355 source checks failed" >&2
  exit 66
fi
echo "[static-ok] 355 source is byte-identical to 352; no code change to build-validate beyond compiling"

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "================================================================"
  echo "[static-summary]"
  cat "$SUMMARY"
  echo "[logdir] $LOGDIR"
  exit 0
fi

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[error] another 355 probe holds: $LOCK_FILE" >&2
    exit 75
  fi
fi

# ---- permission check FIRST, before any timed command (the 352 lesson) ----
echo "[perm-check] sudo -n true"
if sudo -n true 2>/dev/null; then
  printf 'sudo_noninteractive\tworks\tworks\tOK\n' >> "$SUMMARY"
else
  printf 'sudo_noninteractive\tworks\tFAILS -- ncu hardware counters require sudo; fix this before anything else\tFAIL\n' >> "$SUMMARY"
  echo "================================================================"
  cat "$SUMMARY"
  echo "[error] sudo -n true failed. Fix sudo access (e.g. NOPASSWD for this user, or run this script itself under sudo) before re-running. Nothing timed has run yet." >&2
  exit 1
fi

if ! command -v ncu >/dev/null 2>&1; then
  printf 'ncu_present\tfound on PATH\tnot found\tFAIL\n' >> "$SUMMARY"
  echo "================================================================"
  cat "$SUMMARY"
  echo "[error] ncu not found on PATH" >&2
  exit 69
fi
printf 'ncu_present\tfound on PATH\tfound\tOK\n' >> "$SUMMARY"

# ---- build (only if needed) ----
need_build=0
if [[ "$FORCE_REBUILD" == "1" ]]; then
  need_build=1
elif [[ ! -x "$CAND" ]]; then
  need_build=1
elif [[ "$SRC" -nt "$CAND" ]]; then
  need_build=1
fi
BUILD_LOG="$LOGDIR/build.log"
if (( need_build )); then
  if [[ "$AUTO_BUILD" != "1" ]]; then echo "[error] stale/missing candidate and AUTO_BUILD=$AUTO_BUILD: $CAND" >&2; exit 66; fi
  if ! command -v codon >/dev/null 2>&1; then echo "[error] codon was not found; cannot build $SRC" >&2; exit 69; fi
  rm -f "$CAND"
  echo "[build] codon build -release $SRC" | tee "$BUILD_LOG"
  set +e; codon build -release "$SRC" 2>&1 | tee -a "$BUILD_LOG"; build_rc=${PIPESTATUS[0]}; set -e
  record_check build_exit 0 "$build_rc" || failures=$((failures+1))
  if (( build_rc != 0 )); then exit "$build_rc"; fi
else
  echo "[build] reuse executable: $CAND" | tee "$BUILD_LOG"
fi

# ---- STEP 1: structural probe at N=18 (SourceCounters, ~9s expected) ----
# Same argv shape as 352's own N=18 registers/thread and SourceCounters
# checks. bench_mode 31 (mode31 split145) so the SAME code path and the SAME
# compiled kernel_dfs_iter_gpu_maxd14 that N=21 production uses gets probed;
# only the task COUNT differs, which is exactly the point (K=1 here).
N18_LOG="$LOGDIR/ncu_n18_sourcecounters.log"
N18_CMD=(sudo ncu --section SourceCounters --page source --launch-count 1 --target-processes all -f -o "$LOGDIR/n18_sourcecounters" -- "$CAND" -g "$N18" "$N18" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS" "$BENCH_MODE" "$REORDER_WINDOW_MULT" "$REORDER_PHASE_JUMP" "$CROSS_STRIPE_SAFE" "$WORKER_ID" "$WORKER_COUNT" "$BROADMARK_VARIANT" "$CHUNKSHAPE148_BUCKET_RUN" "$CHUNKSHAPE148_ITER_SORT")
echo "[step1] structural probe, N=$N18, --section SourceCounters, expect ~9s"
echo "[step1-cmd] ${N18_CMD[*]}" | tee "$N18_LOG"
set +e; "${N18_CMD[@]}" 2>&1 | tee -a "$N18_LOG"; step1_rc=${PIPESTATUS[0]}; set -e
record_check step1_ncu_exit 0 "$step1_rc" || failures=$((failures+1))
if grep -qi 'ERR_NVGPUCTRPERM' "$N18_LOG"; then
  printf 'step1_permission_error\tabsent\tERR_NVGPUCTRPERM present -- sudo -n true passed but the counter attach itself was still denied; check nvidia driver NVreg_RestrictProfilingToAdminUsers\tFAIL\n' >> "$SUMMARY"
  failures=$((failures+1))
fi
STEP1_HAS_SOURCECOUNTERS=$(grep -c 'Source Counters' "$N18_LOG" || true)
record_check step1_section_header_present 1 "$( [[ "$STEP1_HAS_SOURCECOUNTERS" -ge 1 ]] && echo 1 || echo 0 )" || failures=$((failures+1))

# ---- STEP 2: dynamic probe at N=21 (SchedulerStats + WarpStateStats, chunk 0 launch 0 only) ----
# --launch-count 1 bounds this to the first kernel launch only (chunk 0's
# maxd14 launch), avoiding the full ~400s N=21 run. These two sections are
# single-pass hardware counter sections, not SourceCounters' 5-pass replay.
N21_LOG="$LOGDIR/ncu_n21_schedwarp.log"
N21_CMD=(sudo ncu --section SchedulerStats --section WarpStateStats --launch-count 1 --target-processes all -f -o "$LOGDIR/n21_schedwarp" -- "$CAND" -g "$N21" "$N21" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS" "$BENCH_MODE" "$REORDER_WINDOW_MULT" "$REORDER_PHASE_JUMP" "$CROSS_STRIPE_SAFE" "$WORKER_ID" "$WORKER_COUNT" "$BROADMARK_VARIANT" "$CHUNKSHAPE148_BUCKET_RUN" "$CHUNKSHAPE148_ITER_SORT")
echo "[step2] dynamic probe, N=$N21, --section SchedulerStats --section WarpStateStats, launch-count 1 (chunk 0 only)"
echo "[step2-cmd] ${N21_CMD[*]}" | tee "$N21_LOG"
set +e; "${N21_CMD[@]}" 2>&1 | tee -a "$N21_LOG"; step2_rc=${PIPESTATUS[0]}; set -e
record_check step2_ncu_exit 0 "$step2_rc" || failures=$((failures+1))
if grep -qi 'ERR_NVGPUCTRPERM' "$N21_LOG"; then
  printf 'step2_permission_error\tabsent\tERR_NVGPUCTRPERM present\tFAIL\n' >> "$SUMMARY"
  failures=$((failures+1))
fi
STEP2_HAS_SCHED=$(grep -c 'Scheduler Statistics' "$N21_LOG" || true)
STEP2_HAS_WARP=$(grep -c 'Warp State Statistics' "$N21_LOG" || true)
record_check step2_scheduler_section_present 1 "$( [[ "$STEP2_HAS_SCHED" -ge 1 ]] && echo 1 || echo 0 )" || failures=$((failures+1))
record_check step2_warpstate_section_present 1 "$( [[ "$STEP2_HAS_WARP" -ge 1 ]] && echo 1 || echo 0 )" || failures=$((failures+1))

printf 'no_oracle\tno code changed in this revision\t355 makes no source edit; there is no correctness value and no timing baseline. source_code_identical_to_352 is the only correctness claim.\tINFO\n' >> "$SUMMARY"
printf 'raw_logs\tread these directly, not the summary\t%s and %s\tINFO\n' "$N18_LOG" "$N21_LOG" >> "$SUMMARY"

echo
echo "================================================================"
echo "[summary]"
cat "$SUMMARY"
echo "[logdir]   $LOGDIR"
echo "================================================================"
if (( failures != 0 )); then
  echo "[validation-failed] failures=$failures" >&2
  exit 1
fi
echo "[validation-ok] 355 ran both ncu probes against the unmodified 352 kernel (source_code_identical_to_352 proves the kernel touched is exactly 352's). Section headers for SourceCounters (N=18) and SchedulerStats/WarpStateStats (N=21, chunk 0 launch 0 only) are present. THIS SCRIPT DOES NOT INTERPRET THE NUMBERS: read $LOGDIR/ncu_n18_sourcecounters.log and $LOGDIR/ncu_n21_schedwarp.log directly, or tar $LOGDIR and send it back."
