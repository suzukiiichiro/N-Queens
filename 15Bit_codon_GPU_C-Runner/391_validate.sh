#!/usr/bin/env bash
# 391_validate.sh
#
# rev391 -- Validation harness for the staged maxd16 GPU-vs-CPU cross-
# check (bench_mode=38), run BEFORE attempting N=23's full 44.27M-record
# run. This is also the first-ever real compile of 390's new maxd16
# kernel and 391's new process_one_task_maxd16_cpu() -- a build failure
# here is genuinely possible (neither has been compiled before) and
# should be reported clearly rather than treated as alarming in itself.
#
# Expected runtime: each of the four stages (10,000 / 50,000 / 100,000 /
# 200,000 records) is orders of magnitude smaller than N=21's full
# 2,025,282-record run (~3-4 minutes), so the whole staged sequence
# should complete in well under that -- likely low minutes total. If it
# doesn't, that mismatch between expectation and reality is itself
# useful information (see the note printed on FAIL below).
#
# This harness does NOT attempt N=23's full run -- that is explicitly
# the NEXT step, gated on this one passing.

set -u
PY_SRC="${PY_SRC:-391Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
BIN="${BIN:-391Py_kernel_maxd14_final}"
STATIC_ONLY="${STATIC_ONLY:-0}"
CODON="${CODON:-codon}"
N="${N:-23}"

PASS=0
FAIL=0
declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }

for f in "$PY_SRC" "$HELPER_SRC"; do
  if [[ ! -f "$f" ]]; then
    fail "file_present[$f]" "not found in $(pwd)"
    exit 1
  fi
  pass "file_present[$f]"
done

# ---------------------------------------------------------------------
# 1. static checks
# ---------------------------------------------------------------------
if grep -q "def process_one_task_maxd16_cpu" "$PY_SRC"; then
  pass "cpu_reference_function_present"
else
  fail "cpu_reference_function_present" "process_one_task_maxd16_cpu not found in $PY_SRC"
fi

if grep -q "def maxd16_staged_gpu_cpu_xcheck" "$PY_SRC"; then
  pass "staged_xcheck_function_present"
else
  fail "staged_xcheck_function_present" "maxd16_staged_gpu_cpu_xcheck not found in $PY_SRC"
fi

if grep -q "record_limits:List\[int\]=\[10000,50000,100000,200000\]" "$PY_SRC"; then
  pass "default_stages_match_requested_sequence"
else
  fail "default_stages_match_requested_sequence" "expected the default record_limits to be [10000,50000,100000,200000]"
fi

if [[ "$FAIL" -gt 0 ]]; then exit 1; fi

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "STATIC_ONLY=1: stopping after static checks."
  exit 0
fi

# ---------------------------------------------------------------------
# 2. codon build -- FIRST real compile of 390's kernel and 391's CPU
#    cross-check function together.
# ---------------------------------------------------------------------
if ! command -v "$CODON" >/dev/null 2>&1; then
  fail "codon_toolchain_present" "'codon' not found on PATH"
  exit 1
fi
pass "codon_toolchain_present"

echo "Building $PY_SRC with $CODON build -release (FIRST compile of 390's maxd16 kernel + 391's CPU cross-check)..."
rm -f "$BIN"
BUILD_LOG="391_build_$(date +%Y%m%d_%H%M%S).log"
"$CODON" build -release -o "$BIN" "$PY_SRC" 2>&1 | tee "$BUILD_LOG"

if [[ ! -x "$BIN" ]]; then
  fail "codon_build_succeeded" "binary $BIN was not produced -- see $BUILD_LOG for the exact error(s); this is genuinely possible, not just a formality"
  exit 1
fi
pass "codon_build_succeeded"

# ---------------------------------------------------------------------
# 3. THE staged cross-check itself, N=23, bench_mode=38.
# ---------------------------------------------------------------------
echo "Running: ./$BIN -g $N $N 32 484 1 0 7 38   (staged maxd16 cross-check, N=23)"
RUN_LOG="391_run_$(date +%Y%m%d_%H%M%S).log"
./"$BIN" -g "$N" "$N" 32 484 1 0 7 38 2>&1 | tee "$RUN_LOG"

STAGE_COUNT="$(grep -c '\[maxd16-xcheck\] N=' "$RUN_LOG" || true)"
if [[ "$STAGE_COUNT" -ge 1 ]]; then
  pass "at_least_one_stage_ran ($STAGE_COUNT stage line(s) seen)"
else
  fail "at_least_one_stage_ran" "no [maxd16-xcheck] lines at all -- see $RUN_LOG (possible crash before the first stage even started)"
  exit 1
fi

MATCH_COUNT="$(grep -c '\[maxd16-xcheck\].*match=YES' "$RUN_LOG" || true)"
MISMATCH_COUNT="$(grep -c '\[maxd16-xcheck\].*match=NO' "$RUN_LOG" || true)"

if [[ "$MISMATCH_COUNT" -gt 0 ]]; then
  fail "no_mismatches" "$MISMATCH_COUNT stage(s) reported match=NO -- see $RUN_LOG for [maxd16-xcheck] MISMATCH. This means 390_maxd16_kernel_port_spec.md's depth-invariance claims need re-examination before any further N=23 work, not a quick patch."
else
  pass "no_mismatches"
fi

if grep -q '\[maxd16-xcheck-done\].*all 4 stages matched' "$RUN_LOG"; then
  pass "all_four_stages_completed_and_matched"
else
  fail "all_four_stages_completed_and_matched" "did not see the final [maxd16-xcheck-done] line with all 4 stages -- see $RUN_LOG (could mean an early stop, a crash, or the run is still in progress if you Ctrl-C'd it deliberately)"
fi

echo ""
echo "===== 391 summary ====="
echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then
  echo "FAILED CHECKS:"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  exit 1
fi
echo "391 PASSED: the new maxd16 kernel matches an independent CPU"
echo "reference across all four ascending stages (10k/50k/100k/200k"
echo "records). This is real evidence the kernel is likely correct at"
echo "N=23's full scale, though not a substitute for actually running"
echo "it -- that remains the next, separate step."
exit 0
