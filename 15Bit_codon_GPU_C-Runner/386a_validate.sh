#!/usr/bin/env bash
# 386a_validate.sh
#
# rev386a -- Validation harness for removing bench_mode==28
# (broadmarktail-reorder-sim-only) and bench_mode==29
# (broadmarktail-reorder-gpu), confirmed unused by Suzuki. Two things
# to confirm on real hardware:
#
#   (1) bench_mode==28/29 are truly gone (CLI now rejects them with
#       the existing "[warning] bench_mode=... was removed" message,
#       same mechanism used for every prior removed mode -- no crash,
#       no silent wrong behavior).
#   (2) bench_mode==31 (the surviving pipeline that internally calls
#       the SAME build_broad_markdist_tail_reordered_bin() function
#       28/29 used to call standalone) still reproduces the same N=21
#       result as before -- i.e. removing the standalone debug entry
#       points did not disturb the shared function or the pipeline
#       that still depends on it.
#
# This does NOT re-run bench_mode=37's CRunner dispatch (385/386
# already covered that path twice); this harness is scoped to the
# thing 386a actually touched.

set -u
PY_SRC="${PY_SRC:-386aPy_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
BIN="${BIN:-386aPy_kernel_maxd14_final}"
STATIC_ONLY="${STATIC_ONLY:-0}"
CODON="${CODON:-codon}"
N="${N:-21}"
EXPECTED_ORACLE="${EXPECTED_ORACLE:-314666222712}"

PASS=0
FAIL=0
declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }

# ---------------------------------------------------------------------
# 0. files present
# ---------------------------------------------------------------------
for f in "$PY_SRC" "$HELPER_SRC"; do
  if [[ ! -f "$f" ]]; then
    fail "file_present[$f]" "not found in $(pwd)"
    exit 1
  fi
  pass "file_present[$f]"
done

# ---------------------------------------------------------------------
# 1. static checks: bench_mode==28/29 gone from CLIGATE/PRESETGATE/
#    dispatch; build_broad_markdist_tail_reordered_bin() (the shared
#    function) still present and still called by bench_mode==31.
# ---------------------------------------------------------------------
if grep -qE '^\s*(if|elif)\s.*bench_mode==28' "$PY_SRC"; then
  fail "bench_mode_28_removed" "a real 'if/elif ... bench_mode==28' conditional still exists in $PY_SRC (comments/VERSION_TAG mentions are expected and excluded by this check)"
else
  pass "bench_mode_28_removed"
fi
if grep -qE '^\s*(if|elif)\s.*bench_mode==29' "$PY_SRC"; then
  fail "bench_mode_29_removed" "a real 'if/elif ... bench_mode==29' conditional still exists in $PY_SRC (comments/VERSION_TAG mentions are expected and excluded by this check)"
else
  pass "bench_mode_29_removed"
fi
if grep -q "^def build_broad_markdist_tail_reordered_bin(" "$PY_SRC" && grep -q "bench_mode==30 or bench_mode==31" "$PY_SRC"; then
  pass "shared_function_and_bench31_pipeline_intact"
else
  fail "shared_function_and_bench31_pipeline_intact" "build_broad_markdist_tail_reordered_bin definition or the bench_mode==30/31 dispatch condition is missing -- 386a may have removed too much"
fi

if [[ "$FAIL" -gt 0 ]]; then exit 1; fi

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "STATIC_ONLY=1: stopping after static checks."
  exit 0
fi

# ---------------------------------------------------------------------
# 2. codon build
# ---------------------------------------------------------------------
if ! command -v "$CODON" >/dev/null 2>&1; then
  fail "codon_toolchain_present" "'codon' not found on PATH"
  exit 1
fi
pass "codon_toolchain_present"

echo "Building $PY_SRC with $CODON build -release..."
rm -f "$BIN"
BUILD_LOG="386a_build_$(date +%Y%m%d_%H%M%S).log"
"$CODON" build -release -o "$BIN" "$PY_SRC" 2>&1 | tee "$BUILD_LOG"

if [[ ! -x "$BIN" ]]; then
  fail "codon_build_succeeded" "binary $BIN was not produced -- see $BUILD_LOG"
  exit 1
fi
pass "codon_build_succeeded"

# ---------------------------------------------------------------------
# 3. confirm bench_mode==28 is now rejected cleanly (the existing
#    "[warning] bench_mode=... was removed" mechanism, same as every
#    prior removed mode -- not a crash, not silently ignored).
# ---------------------------------------------------------------------
echo "Confirming bench_mode==28 is rejected: ./$BIN -g 5 5 32 484 0 0 5 28"
REJECT_LOG="386a_reject_$(date +%Y%m%d_%H%M%S).log"
./"$BIN" -g 5 5 32 484 0 0 5 28 2>&1 | tee "$REJECT_LOG"
if grep -q '\[warning\] bench_mode=28 was removed' "$REJECT_LOG"; then
  pass "bench_mode_28_rejected_cleanly"
else
  fail "bench_mode_28_rejected_cleanly" "expected the existing [warning] bench_mode=28 was removed message -- see $REJECT_LOG"
fi

# ---------------------------------------------------------------------
# 4. regression: bench_mode==31 (the surviving pipeline) still
#    reproduces the same N=21 total it always has.
# ---------------------------------------------------------------------
echo "Running: ./$BIN -g $N $N 32 484 1 0 7 31"
RUN_LOG="386a_run_$(date +%Y%m%d_%H%M%S).log"
./"$BIN" -g "$N" "$N" 32 484 1 0 7 31 2>&1 | tee "$RUN_LOG"

if grep -qE "^${N}:\s*${EXPECTED_ORACLE}\s" "$RUN_LOG"; then
  pass "bench31_total_matches_oracle (expected=$EXPECTED_ORACLE)"
else
  fail "bench31_total_matches_oracle" "N=$N row did not show $EXPECTED_ORACLE -- see $RUN_LOG"
fi

if grep -qE "^${N}:.*[[:space:]]ok[[:space:]]*\$" "$RUN_LOG"; then
  pass "bench31_table_row_status_ok"
else
  fail "bench31_table_row_status_ok" "N=$N row did not print status 'ok' -- see $RUN_LOG"
fi

echo ""
echo "===== 386a summary ====="
echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then
  echo "FAILED CHECKS:"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  exit 1
fi
echo "386a PASSED: bench_mode==28/29 are cleanly gone (existing"
echo "[warning]-removed mechanism), and bench_mode==31 -- which still"
echo "calls the same shared build_broad_markdist_tail_reordered_bin()"
echo "function internally -- reproduces the same N=21 total as always."
exit 0
