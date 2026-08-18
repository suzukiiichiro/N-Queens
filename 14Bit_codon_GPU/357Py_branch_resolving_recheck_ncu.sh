#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# 357 branch_resolving re-check (ncu probe, NO CODE CHANGE)
#
# WHAT 357 IS. 356 narrowed save_sp from Codon's 64-bit int to u32 and
# measured a consistent 3-chunk improvement (-0.83% / -0.93% / -2.00%
# against the 352 anchor). It did NOT touch the two unconditional BRA
# instructions 355 identified as responsible for 60.2% of the measured
# stall_branch_resolving total, both adjacent to BSYNC reconvergence
# markers in the stack push/pop machinery.
#
# 357 changes NO code. It reruns 355's ncu procedure against the 356
# kernel instead of the 352 kernel:
#   - N=18 (K=1), --section SourceCounters --page source (~9s, instruction
#     level breakdown, the same technique 304-319/355 used)
#   - N=21 (K=48, production shape), --section SchedulerStats
#     --section WarpStateStats, chunk0 only
#
# THE ONE PROCEDURAL CHANGE FROM 355: N=21 dynamic profiling uses
# bench_mode=30 (split145 probe/cache-build, debug_chunk_start=0
# debug_chunk_count=1) instead of 355's bench_mode=31 + --launch-count 1.
# 355 learned the hard way that --launch-count only scopes ncu
# instrumentation; bench_mode=31 still runs the full 3-chunk program to
# completion regardless (measured 24 minutes vs. the normal ~398s).
# bench_mode=30 actually stops after the requested chunk count.
#
# NO ORACLE IN THIS REVISION. No code changed and only chunk0 executes,
# so there is no 314666222712 full-run total to check. The only
# correctness criterion is byte-for-byte source identity with 356 outside
# VERSION_TAG (source_code_identical_to_356 below).
#
# NCU OUTPUT IS NOT AUTO-PARSED. Per 355's policy: the script only checks
# that expected section headings appear in the console dump, and keeps
# the raw .ncu-rep report files and console logs for manual reading.
# Numeric interpretation happens after the real hardware log is in hand.
#
#     STATIC_ONLY=1 bash 357Py_branch_resolving_recheck_ncu.sh
#                    bash 357Py_branch_resolving_recheck_ncu.sh
#
# Permission check runs FIRST, before any build or ncu invocation (352
# lesson: 14 minutes were wasted once by discovering a missing sudo grant
# only after a long build/run).
# =============================================================================

SRC=${SRC:-./357Py_branch_resolving_recheck_ncu.py}
CAND=${CAND:-./357Py_branch_resolving_recheck_ncu}
AUTO_BUILD=${AUTO_BUILD:-1}
FORCE_REBUILD=${FORCE_REBUILD:-1}
STATIC_ONLY=${STATIC_ONLY:-0}
LOG_ROOT=${LOG_ROOT:-.}
LOCK_FILE=${LOCK_FILE:-/tmp/357Py_branch_resolving_recheck_ncu.lock}

# ---- N=18 SourceCounters probe params (normal GPU mode, bench_mode=0) ----
N18=${N18:-18}
BLOCK=${BLOCK:-32}
MAX_BLOCKS=${MAX_BLOCKS:-484}
LOG_LEVEL=${LOG_LEVEL:-1}
SORT_MODE=${SORT_MODE:--1}
PRESET_QUEENS_N18=${PRESET_QUEENS_N18:-5}
BENCH_MODE_N18=${BENCH_MODE_N18:-0}

# ---- N=21 SchedulerStats/WarpStateStats probe params (bench_mode=30, chunk0 only) ----
N21=${N21:-21}
PRESET_QUEENS_N21=${PRESET_QUEENS_N21:-7}
BENCH_MODE_N21=${BENCH_MODE_N21:-30}
REORDER_WINDOW_MULT=${REORDER_WINDOW_MULT:-3}
REORDER_PHASE_JUMP=${REORDER_PHASE_JUMP:-7}
CROSS_STRIPE_SAFE=${CROSS_STRIPE_SAFE:-0}
DEBUG_CHUNK_START=${DEBUG_CHUNK_START:-0}
DEBUG_CHUNK_COUNT=${DEBUG_CHUNK_COUNT:-1}
SPLIT_PROBE_CHUNK_LIST_SPEC=${SPLIT_PROBE_CHUNK_LIST_SPEC:-}
BROADMARK_VARIANT=${BROADMARK_VARIANT:-2}

# ---- fixed reference: sha256 of 356's source, docstrings stripped AND the
# VERSION_TAG:str=... line removed. 357's candidate must match exactly;
# only VERSION_TAG (and the two free-form docstrings, already stripped) may
# differ between 356 and 357. Computed once, offline, against the actual
# 356 deliverable (356Py_savesp_narrow.py). ----
EXPECTED_SHA256_356_NODOC_NOVERSIONTAG="6db7de43fe05143352e9b8d5917b334c4031e488ed234417153fc74901fc09ca"
EXPECTED_LINES_356_NODOC_NOVERSIONTAG=5607

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="${LOG_ROOT%/}/357Py_branch_resolving_recheck_ncu_logs_${TS}"
BUILD_LOG="$LOGDIR/build.log"
SUMMARY="$LOGDIR/summary.tsv"
mkdir -p "$LOGDIR"
printf 'check\texpected\tactual\tstatus\n' > "$SUMMARY"

echo "[start] 357 branch_resolving recheck (ncu probe, no code change)"
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
static_failures=0

# ---- permission check FIRST, before touching source, build, or ncu ----
# ERR_NVGPUCTRPERM fires at ncu attach time regardless of section weight;
# this is the 318 lesson. Not gated on STATIC_ONLY: even a static-only run
# should surface a missing grant immediately rather than after the fact.
if sudo -n true 2>/dev/null; then
  printf 'sudo_permission_check\tsudo -n true succeeds\tsucceeded\tOK\n' >> "$SUMMARY"
else
  printf 'sudo_permission_check\tsudo -n true succeeds\tfailed (passwordless sudo not available)\tFAIL\n' >> "$SUMMARY"
  echo "================================================================"
  cat "$SUMMARY"
  echo "[error] sudo -n true failed; ncu requires elevated perf-counter access (ERR_NVGPUCTRPERM). Aborting before any build or ncu work." >&2
  exit 77
fi

if [[ ! -f "$SRC" ]]; then
  echo "[error] source not found: $SRC" >&2
  exit 66
fi

# ---- docstring-stripped copy, same safety-checked technique as 328 r4/345 r2 ----
SRC_NODOC="$LOGDIR/src_nodoc.py"
python3 - "$SRC" "$SRC_NODOC" <<'PYSTRIP'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding='utf-8').read()
s = re.sub(r'"""[\s\S]*?"""', '', s)
open(dst, 'w', encoding='utf-8').write(s)
PYSTRIP
if [[ -s "$SRC_NODOC" ]] \
   && grep -q '^def build_chunkshape148_reordered_bin(' "$SRC_NODOC" \
   && grep -q '^def main()->None:' "$SRC_NODOC" \
   && grep -q '^VERSION_TAG:str=' "$SRC_NODOC"; then
  NODOC_LINES=$(grep -c '' "$SRC_NODOC")
  SRC_LINES=$(grep -c '' "$SRC")
  printf 'source_nodoc_copy\tdocstrings stripped, code markers intact\t%s of %s lines kept\tOK\n' "$NODOC_LINES" "$SRC_LINES" >> "$SUMMARY"
else
  printf 'source_nodoc_copy\tdocstrings stripped, code markers intact\tstripped copy lost code markers -- check for an unbalanced triple quote in the docstring\tFAIL\n' >> "$SUMMARY"
  static_failures=$((static_failures+1))
  SRC_NODOC="$SRC"
fi

# ---- 339 r2 policy: Codon string-literal quote balance on module-level NAME:str="..." lines ----
QUOTE_BAD=$(awk '/^[A-Za-z_][A-Za-z0-9_]*:str="/ { n=gsub(/"/,"\""); if (n!=2) bad++ } END { print bad+0 }' "$SRC")
record_check source_str_literal_quote_balance 0 "$QUOTE_BAD" || static_failures=$((static_failures+1))
if [[ "$QUOTE_BAD" != "0" ]]; then
  echo "[error] a module-level NAME:str=\"...\" line contains an unescaped double quote:" >&2
  awk '/^[A-Za-z_][A-Za-z0-9_]*:str="/ { n=gsub(/"/,"\""); if (n!=2) printf "  line %d: %s (quotes=%d)\n", NR, substr($0,1,index($0,":")), n }' "$SRC" >&2
fi

# ---- version tag ----
if grep -q '357 branch-resolving-recheck-ncu' "$SRC"; then
  printf 'source_version_tag\t357 branch-resolving-recheck-ncu\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_version_tag\t357 branch-resolving-recheck-ncu\tmissing\tFAIL\n' >> "$SUMMARY"
  static_failures=$((static_failures+1))
fi

# ---- THE gating check: candidate is byte-identical to 356 outside VERSION_TAG ----
# Same fingerprint technique 353/354/355 established for "no code change"
# revisions, generalized: docstrings stripped (free-form prose, already
# excluded), VERSION_TAG:str=... line removed (expected to differ every
# revision), sha256 of what remains must match 356 exactly.
CAND_SHA=$(python3 - "$SRC_NODOC" <<'PYHASH'
import sys, hashlib
s = open(sys.argv[1], encoding='utf-8').read()
lines = [l for l in s.split('\n') if not l.startswith('VERSION_TAG:str=')]
print(hashlib.sha256('\n'.join(lines).encode('utf-8')).hexdigest())
PYHASH
)
CAND_LINES=$(python3 - "$SRC_NODOC" <<'PYLINES'
import sys
s = open(sys.argv[1], encoding='utf-8').read()
lines = [l for l in s.split('\n') if not l.startswith('VERSION_TAG:str=')]
print(len(lines))
PYLINES
)
if [[ "$CAND_SHA" == "$EXPECTED_SHA256_356_NODOC_NOVERSIONTAG" && "$CAND_LINES" == "$EXPECTED_LINES_356_NODOC_NOVERSIONTAG" ]]; then
  printf 'source_code_identical_to_356\tsha256=%s (lines=%s)\tsha256=%s (lines=%s)\tOK\n' \
    "$EXPECTED_SHA256_356_NODOC_NOVERSIONTAG" "$EXPECTED_LINES_356_NODOC_NOVERSIONTAG" \
    "$CAND_SHA" "$CAND_LINES" >> "$SUMMARY"
else
  printf 'source_code_identical_to_356\tsha256=%s (lines=%s)\tsha256=%s (lines=%s)\tFAIL\n' \
    "$EXPECTED_SHA256_356_NODOC_NOVERSIONTAG" "$EXPECTED_LINES_356_NODOC_NOVERSIONTAG" \
    "$CAND_SHA" "$CAND_LINES" >> "$SUMMARY"
  static_failures=$((static_failures+1))
  echo "[error] source_code_identical_to_356 FAILED -- 357 must not change any code outside VERSION_TAG. Run 'diff' against 356Py_savesp_narrow.py to find the stray edit." >&2
fi

# ---- negative-test note (run manually, not automated here): ----
#   SRC=./356Py_savesp_narrow.py STATIC_ONLY=1 bash 357Py_branch_resolving_recheck_ncu.sh
# must FAIL exactly source_version_tag (356's tag lacks "357 ..."), while
# source_code_identical_to_356 passes trivially (356 compared to itself
# after VERSION_TAG-line removal, which is a no-op on 356's own tag).

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "================================================================"
  echo "[static-summary]"
  cat "$SUMMARY"
  echo "[logdir] $LOGDIR"
  if (( static_failures != 0 )); then exit 1; fi
  exit 0
fi

if (( static_failures != 0 )); then
  echo "================================================================"
  echo "[static-summary]"
  cat "$SUMMARY"
  echo "[logdir] $LOGDIR"
  echo "[error] 357 source static checks failed" >&2
  exit 66
fi
echo "[static-ok] 357 source checks passed; proceeding to build/ncu"

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[error] another 357 run holds: $LOCK_FILE" >&2
    exit 75
  fi
fi

need_build=0
if [[ "$FORCE_REBUILD" == "1" ]]; then
  need_build=1
elif [[ ! -x "$CAND" ]]; then
  need_build=1
elif [[ "$SRC" -nt "$CAND" ]]; then
  need_build=1
fi
if (( need_build )); then
  if [[ "$AUTO_BUILD" != "1" ]]; then echo "[error] stale/missing candidate and AUTO_BUILD=$AUTO_BUILD: $CAND" >&2; exit 66; fi
  if ! command -v codon >/dev/null 2>&1; then echo "[error] codon was not found; cannot build $SRC" >&2; exit 69; fi
  rm -f "$CAND"
  echo "[build] codon build -release $SRC" | tee "$BUILD_LOG"
  set +e; codon build -release "$SRC" 2>&1 | tee -a "$BUILD_LOG"; build_rc=${PIPESTATUS[0]}; set -e
  record_check build_exit 0 "$build_rc"
  if (( build_rc != 0 )); then
    echo "================================================================"; cat "$SUMMARY"
    exit "$build_rc"
  fi
else
  echo "[build] reuse executable: $CAND" | tee "$BUILD_LOG"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu="timestamp,clocks.current.sm,clocks.max.sm,power.draw,power.limit" --format=csv > "$LOGDIR/gpu_pre_probe_snapshot.csv" 2>&1 || true
  echo "[telemetry] pre-probe snapshot (context only, not gated): $LOGDIR/gpu_pre_probe_snapshot.csv"
fi

# =============================================================================
# Probe 1: N=18 (K=1), instruction-level breakdown.
# Same technique 304-319/355 established: registers/thread and SASS
# structure are compile-time attributes, N-independent, so N=18's
# SourceCounters transfer to N=21's kernel_dfs_iter_gpu_maxd14 unchanged.
# Normal GPU mode (bench_mode=0) so N=18 falls through the same
# launch_kernel_dfs_iter_gpu_static_maxd path 352 verified. --launch-count 1
# is safe here because bench_mode 0/11/28/29 do not have 31's "always run
# to completion regardless of instrumentation scope" behavior; 18 is a
# trivially small N anyway (~seconds either way).
# =============================================================================
N18_REP="$LOGDIR/n18_sourcecounters"
N18_CONSOLE="$LOGDIR/n18_sourcecounters_console.log"
echo "[ncu] N=18 SourceCounters --page source (~9s expected)"
set +e
sudo ncu --section SourceCounters --page source --csv --target-processes all \
  --launch-count 1 -f -o "$N18_REP" \
  "$CAND" -g "$N18" "$N18" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS_N18" "$BENCH_MODE_N18" \
  > "$N18_CONSOLE" 2>&1
ncu_n18_rc=$?
set -e
printf 'ncu_n18_sourcecounters_exit\t0\t%s\t%s\n' "$ncu_n18_rc" "$([[ $ncu_n18_rc -eq 0 ]] && echo OK || echo FAIL)" >> "$SUMMARY"
if [[ -f "${N18_REP}.ncu-rep" ]]; then
  printf 'ncu_n18_report_present\tpresent\tpresent (%s.ncu-rep)\tOK\n' "$N18_REP" >> "$SUMMARY"
else
  printf 'ncu_n18_report_present\tpresent\tmissing\tFAIL\n' >> "$SUMMARY"
fi
# Loose, non-gating heading presence check only -- ncu output is never
# auto-parsed for values (355 policy). This just confirms the section ran.
if grep -qi 'SourceCounters' "$N18_CONSOLE" 2>/dev/null; then
  printf 'ncu_n18_sourcecounters_heading_seen\tpresent (informational only)\tpresent\tINFO\n' >> "$SUMMARY"
else
  printf 'ncu_n18_sourcecounters_heading_seen\tpresent (informational only)\tnot found in console dump; inspect %s and re-render with --page source if needed\tINFO\n' "$N18_CONSOLE" >> "$SUMMARY"
fi

# =============================================================================
# Probe 2: N=21 (K=48, production shape), chunk0 only.
# bench_mode=30 (split145 probe/cache-build): argv[13]=debug_chunk_start,
# argv[14]=debug_chunk_count, argv[15]=split_probe_chunk_list_spec,
# argv[16]=broadmark_tail_variant. chunk_start=0 chunk_count=1 executes
# ONLY chunk0's launch, then stops -- unlike 355's bench_mode=31 +
# --launch-count 1, which still ran the full 3-chunk program to completion
# under instrumentation (measured 24 minutes). SchedulerStats +
# WarpStateStats are single hardware-counter passes, no 5-pass replay, so
# they do not need N=18's SourceCounters shortcut.
# =============================================================================
N21_REP="$LOGDIR/n21_chunk0_schedulerwarp"
N21_CONSOLE="$LOGDIR/n21_chunk0_schedulerwarp_console.log"
echo "[ncu] N=21 SchedulerStats+WarpStateStats, bench_mode=30 chunk0 only"
set +e
sudo ncu --section SchedulerStats --section WarpStateStats --target-processes all \
  --launch-count 1 -f -o "$N21_REP" \
  "$CAND" -g "$N21" "$N21" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" 0 "$PRESET_QUEENS_N21" "$BENCH_MODE_N21" \
  "$REORDER_WINDOW_MULT" "$REORDER_PHASE_JUMP" "$CROSS_STRIPE_SAFE" \
  "$DEBUG_CHUNK_START" "$DEBUG_CHUNK_COUNT" "$SPLIT_PROBE_CHUNK_LIST_SPEC" "$BROADMARK_VARIANT" \
  > "$N21_CONSOLE" 2>&1
ncu_n21_rc=$?
set -e
printf 'ncu_n21_schedulerwarp_exit\t0\t%s\t%s\n' "$ncu_n21_rc" "$([[ $ncu_n21_rc -eq 0 ]] && echo OK || echo FAIL)" >> "$SUMMARY"
if [[ -f "${N21_REP}.ncu-rep" ]]; then
  printf 'ncu_n21_report_present\tpresent\tpresent (%s.ncu-rep)\tOK\n' "$N21_REP" >> "$SUMMARY"
else
  printf 'ncu_n21_report_present\tpresent\tmissing\tFAIL\n' >> "$SUMMARY"
fi
if grep -qi 'Scheduler Statistics\|SchedulerStats' "$N21_CONSOLE" 2>/dev/null; then
  printf 'ncu_n21_schedulerstats_heading_seen\tpresent (informational only)\tpresent\tINFO\n' >> "$SUMMARY"
else
  printf 'ncu_n21_schedulerstats_heading_seen\tpresent (informational only)\tnot found in console dump; inspect %s\tINFO\n' "$N21_CONSOLE" >> "$SUMMARY"
fi
if grep -qi 'Warp State Statistics\|WarpStateStats' "$N21_CONSOLE" 2>/dev/null; then
  printf 'ncu_n21_warpstatestats_heading_seen\tpresent (informational only)\tpresent\tINFO\n' >> "$SUMMARY"
else
  printf 'ncu_n21_warpstatestats_heading_seen\tpresent (informational only)\tnot found in console dump; inspect %s\tINFO\n' "$N21_CONSOLE" >> "$SUMMARY"
fi
if grep -q 'split291_final_probe' "$N21_CONSOLE" 2>/dev/null; then
  printf 'n21_probe_mode_confirmed\tsplit291_final_probe printed (bench_mode=30 took effect)\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'n21_probe_mode_confirmed\tsplit291_final_probe printed (bench_mode=30 took effect)\tnot found -- check LOG_LEVEL/argv wiring\tFAIL\n' >> "$SUMMARY"
fi

echo "================================================================"
echo "[summary]"
cat "$SUMMARY"
echo "[logdir] $LOGDIR"
echo "[note] no correctness oracle this revision (no code change, chunk0-only run)."
echo "[note] ncu output not auto-parsed. Read ${N18_CONSOLE} and ${N21_CONSOLE}, and re-render"
echo "[note] the .ncu-rep files with 'ncu -i <report>.ncu-rep --page source --csv' if a section"
echo "[note] needs to be seen again without re-measuring (--page source is easy to forget -- 355 lesson)."
