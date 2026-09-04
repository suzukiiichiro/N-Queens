#!/usr/bin/env bash
# 377b_frame_kernel_validate.sh
#
# rev377b — Integration of the frame queue into a real (decoupled)
# DFS execution model, validated on SYNTHETIC data against an
# unmodified copy of process_one_task (see file header of
# 377b_frame_kernel.cu for the full rationale and the three
# concurrency bugs found and fixed this session). Does NOT touch
# 374Py_kernel_maxd14.cu. CPU/OpenMP only, no GPU/nvcc needed.

set -u
CUSRC="${CUSRC:-377b_frame_kernel.cu}"
BIN="${BIN:-377b_frame_kernel_test}"
STATIC_ONLY="${STATIC_ONLY:-0}"

PASS=0
FAIL=0
declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }

if [[ ! -f "$CUSRC" ]]; then
  fail "file_present[$CUSRC]" "not found in $(pwd)"
  exit 1
fi
pass "file_present[$CUSRC]"

for sym in process_one_task_reference seed_task step_one_frame fq_push fq_pop FQSlot; do
  if grep -q "$sym" "$CUSRC"; then
    pass "symbol_present[$sym]"
  else
    fail "symbol_present[$sym]" "'$sym' not found in $CUSRC"
  fi
done
if [[ "$FAIL" -gt 0 ]]; then exit 1; fi

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "STATIC_ONLY=1: stopping after static checks."
  exit 0
fi

CC="${CC:-gcc}"
command -v "$CC" >/dev/null 2>&1 || CC=cc
rm -f "$BIN"
"$CC" -x c -O2 -Wall -Wextra -fopenmp -o "$BIN" "$CUSRC" -lm 2>&1 | tee "${BIN}_build_$(date +%Y%m%d_%H%M%S).log"
if [[ ! -x "$BIN" ]]; then
  fail "build_succeeded" "binary $BIN was not produced"
  exit 1
fi
pass "build_succeeded"

BUILD_LOG=$(ls -t "${BIN}_build_"*.log 2>/dev/null | head -n1)
if [[ -n "$BUILD_LOG" ]] && grep -qi warning "$BUILD_LOG"; then
  fail "build_warning_free" "gcc emitted warnings, see $BUILD_LOG"
else
  pass "build_warning_free"
fi

NPROC=$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)
echo "Detected NPROC=$NPROC. Running with 1, 2, NPROC, NPROC*4 threads, 3 repeats each, n_tasks=20000..."
ALL_OK=1
for t in 1 2 "$NPROC" $((NPROC*4)); do
  for run in 1 2 3; do
    OUT=$(timeout 120 "./$BIN" "$t" 20000 2>&1)
    if ! echo "$OUT" | grep -q "===== 377b: PASS"; then
      ALL_OK=0
      echo "$OUT"
      fail "run[threads=$t,run=$run]" "did not report 377b PASS"
    fi
  done
done
if [[ "$ALL_OK" -eq 1 ]]; then
  pass "all_thread_counts_all_repeats_passed"
fi

echo ""
echo "===== 377b summary ====="
echo "OK=$PASS  FAIL=$FAIL"
[[ "$FAIL" -gt 0 ]] && exit 1
echo "377b PASSED: frame-queue based DFS (full decoupling: push sibling+child,"
echo "always return to queue) proven byte-identical to unmodified"
echo "process_one_task on synthetic data, under real multi-thread concurrency."
echo "No GPU kernel code was touched. Three concurrency bugs were found and"
echo "fixed in this revision (see file header for the full diagnostic trail)."
echo "Ready to discuss 377c (real N=21-slice data, then real GPU integration)."
exit 0
