#!/usr/bin/env bash
# 377a_frame_queue_validate.sh
#
# rev377a — Frame queue isolated correctness test (CPU/OpenMP only, no
# GPU/nvcc needed, mirrors 363's "cheapest possible check before a real
# device build" philosophy). Does NOT touch 374Py_kernel_maxd14.cu.
#
# IMPORTANT: the sandbox this was authored in has only 1 CPU core, so
# it cannot exercise true hardware-parallel memory-ordering races.
# Please also run this on cudacodon (or any real multi-core machine)
# with NPROC set to the real core count -- that is where a genuine
# race (if one exists) would actually be caught.

set -u
CUSRC="${CUSRC:-377a_frame_queue.cu}"
BIN="${BIN:-377a_frame_queue_test}"
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

for sym in fq_push fq_pop FrameQueue TaskSchedule test_a_no_overflow test_a2_exact_once test_b_overflow_detected; do
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
echo "Detected NPROC=$NPROC. Running with 1, 2, NPROC, and NPROC*4 threads, 3 repeats each..."
ALL_OK=1
for t in 1 2 "$NPROC" $((NPROC*4)); do
  for run in 1 2 3; do
    OUT=$("./$BIN" "$t" 2>&1)
    if ! echo "$OUT" | tail -1 | grep -q "ALL TESTS PASSED"; then
      ALL_OK=0
      echo "$OUT"
      fail "run[threads=$t,run=$run]" "did not report ALL TESTS PASSED"
    fi
  done
done
if [[ "$ALL_OK" -eq 1 ]]; then
  pass "all_thread_counts_all_repeats_passed"
fi

echo ""
echo "===== 377a summary ====="
echo "OK=$PASS  FAIL=$FAIL"
[[ "$FAIL" -gt 0 ]] && exit 1
echo "377a PASSED: frame queue push/pop primitives proven correct under"
echo "concurrent OpenMP access (exact-once round-trip + overflow detection)."
echo "No GPU kernel code was touched. Ready to design 377b (integration"
echo "into a persistent kernel) once this has also been confirmed on a"
echo "real multi-core machine (this sandbox has only 1 core)."
exit 0
