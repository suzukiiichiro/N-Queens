#!/usr/bin/env bash
# 390_validate.sh
#
# rev390 -- Validation harness for the new kernel_dfs_iter_gpu_maxd16
# (390_maxd16_kernel_port_spec.md). This is the FIRST real-hardware test
# of this revision's code -- everything up to now was static analysis
# and a from-source diff proof, not a compile or a run.
#
# Test: bench_mode=33 (exec_solutions_gpu_single_shot(), already
# maxd-generic -- no new wiring needed) against N=23's real data. N=23's
# stream bin (44,271,796 records) should already be cached from 389's
# own bench_mode=34 run; if not, this will regenerate it (~1-2 minutes
# per 389's own measurement).
#
# Pass criterion: total=24233937684440 (N=23's published oracle, already
# present in this file's own expected[] array at index 23).

set -u
PY_SRC="${PY_SRC:-390Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
BIN="${BIN:-390Py_kernel_maxd14_final}"
STATIC_ONLY="${STATIC_ONLY:-0}"
CODON="${CODON:-codon}"
N="${N:-23}"
EXPECTED_ORACLE="${EXPECTED_ORACLE:-24233937684440}"

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
# 1. static checks: old stale kernel gone, new one present with the
#    right stack-size constant; sizing functions corrected.
# ---------------------------------------------------------------------
if grep -q "MAXD16_ANCESTOR:Static\[int\]=15" "$PY_SRC"; then
  pass "maxd16_ancestor_constant_present"
else
  fail "maxd16_ancestor_constant_present" "MAXD16_ANCESTOR:Static[int]=15 not found in $PY_SRC"
fi

if grep -q "stack=__array__\[u64\](MAXD16_ANCESTOR\*2)" "$PY_SRC"; then
  pass "maxd16_kernel_uses_new_ancestor_constant"
else
  fail "maxd16_kernel_uses_new_ancestor_constant" "expected the maxd16 kernel body to allocate its stack via MAXD16_ANCESTOR*2"
fi

if grep -q "kbatch_stride16" "$PY_SRC"; then
  pass "dispatcher_passes_stride_to_maxd16"
else
  fail "dispatcher_passes_stride_to_maxd16" "launch_kernel_dfs_iter_gpu_static_maxd()'s selected_maxd==16 branch does not appear to pass a stride argument"
fi

if [[ "$FAIL" -gt 0 ]]; then exit 1; fi

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "STATIC_ONLY=1: stopping after static checks."
  exit 0
fi

# ---------------------------------------------------------------------
# 2. codon build -- the actual first real-hardware test of this
#    revision's code. A build failure here is entirely plausible (this
#    file has never been compiled before) and not itself alarming --
#    report the log clearly either way, same spirit as 385/388's own
#    "first real compile" notes.
# ---------------------------------------------------------------------
if ! command -v "$CODON" >/dev/null 2>&1; then
  fail "codon_toolchain_present" "'codon' not found on PATH"
  exit 1
fi
pass "codon_toolchain_present"

echo "Building $PY_SRC with $CODON build -release (FIRST compile of the new maxd16 kernel)..."
rm -f "$BIN"
BUILD_LOG="390_build_$(date +%Y%m%d_%H%M%S).log"
"$CODON" build -release -o "$BIN" "$PY_SRC" 2>&1 | tee "$BUILD_LOG"

if [[ ! -x "$BIN" ]]; then
  fail "codon_build_succeeded" "binary $BIN was not produced -- see $BUILD_LOG for the exact error(s); this is genuinely possible given no prior real-hardware compile of this kernel exists"
  exit 1
fi
pass "codon_build_succeeded"

# ---------------------------------------------------------------------
# 3. THE real test: N=23 via bench_mode=33, gpu_log_level=1 so the
#    [single-shot-maxd-dispatch] line confirms selected_maxd=16 was
#    actually chosen (not silently falling back to something else).
# ---------------------------------------------------------------------
echo "Running: ./$BIN -g $N $N 32 484 1 0 7 33   (N=23, first-ever maxd16 real-hardware run)"
RUN_LOG="390_run_$(date +%Y%m%d_%H%M%S).log"
./"$BIN" -g "$N" "$N" 32 484 1 0 7 33 2>&1 | tee "$RUN_LOG"

if grep -q "\[single-shot-maxd-dispatch\].*selected_maxd=16" "$RUN_LOG"; then
  pass "selected_maxd_is_16 (confirmed the new kernel path was actually taken)"
else
  fail "selected_maxd_is_16" "expected selected_maxd=16 in [single-shot-maxd-dispatch] -- see $RUN_LOG"
fi

if grep -qE "^${N}:\s*${EXPECTED_ORACLE}\s.*[[:space:]]ok[[:space:]]*\$" "$RUN_LOG"; then
  pass "N23_matches_oracle_and_ok (expected=$EXPECTED_ORACLE)"
else
  fail "N23_matches_oracle_and_ok" "N=$N row missing, wrong total, or missing 'ok' -- see $RUN_LOG. A wrong (not crashed) total here would mean section 2 of 390_maxd16_kernel_port_spec.md's depth-invariance claim needs re-examination, not a quick patch."
fi

echo ""
echo "===== 390 summary ====="
echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then
  echo "FAILED CHECKS:"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  exit 1
fi
echo "390 PASSED: the new maxd16 kernel, derived mechanically from"
echo "maxd14's current optimized code with exactly one constant changed,"
echo "correctly computes N=23's real oracle total on real hardware."
echo "390_maxd16_kernel_port_spec.md's core depth-invariance claims are"
echo "now confirmed, not just argued."
exit 0
