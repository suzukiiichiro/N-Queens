#!/usr/bin/env bash
# 379b_k_sweep_fixed_dataset.sh
#
# rev379b — Same 379a_hybrid_kernel.cu (no .cu changes at all in this
# revision). The point of 379b is methodological, not code: 379a's own
# sweep used a DIFFERENT record count per K value (small for low K, to
# keep runtime sane given low K's huge queue traffic), which confounds
# K's true effect with "which records happened to be in that subset."
# 379b fixes this by using ONE constant record set across every K
# value, and measuring wall-clock time alongside total_frames, so the
# comparison is a clean, apples-to-apples characterization of K.
#
# FIXED_RECORDS defaults to 200 -- chosen because K=1 (the most
# expensive setting) produced ~151,576 frames/task on 20 real records
# in 379a; scaling to 200 records already means substantial queue
# traffic even at K=1, while staying well within capacity and finishing
# in reasonable time. Override FIXED_RECORDS to try other fixed sizes,
# but be aware K=1 at large record counts can need very large
# FQ_CAPACITY_LOG2 (see 377c's finding: 1000 real records needed
# 400M+ shared-queue slots at effectively-K=0).

set -u
CUSRC="${CUSRC:-379a_hybrid_kernel_timed.cu}"
BIN="${BIN:-379a_hybrid_kernel_timed_test}"
STATIC_ONLY="${STATIC_ONLY:-0}"
SOA7_FILE="${SOA7_FILE:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
N="${N:-21}"
FQ_CAPACITY_LOG2="${FQ_CAPACITY_LOG2:-28}"
FIXED_RECORDS="${FIXED_RECORDS:-200}"
K_VALUES="${K_VALUES:-1 4 8 13 16 20 24}"

PASS=0
FAIL=0
declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }

if [[ ! -f "$CUSRC" ]]; then
  fail "file_present[$CUSRC]" "not found in $(pwd)"
  exit 1
fi
pass "file_present[$CUSRC]"

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

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "STATIC_ONLY=1: stopping after static checks."
  exit 0
fi

NPROC=$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)

if [[ -f "$SOA7_FILE" ]]; then
  DATA_ARGS=("$SOA7_FILE" "$N" "$FIXED_RECORDS")
  info "data_source" "using real data: $SOA7_FILE, FIXED $FIXED_RECORDS records for every K"
else
  DATA_ARGS=("--synthetic" "$FIXED_RECORDS")
  info "data_source" "$SOA7_FILE not found -- using --synthetic self-test, FIXED $FIXED_RECORDS tasks for every K"
fi

echo ""
echo "===== K sweep on a FIXED $FIXED_RECORDS-record dataset (NPROC=$NPROC threads) ====="
printf "%-4s %-10s %-12s %-16s %-12s %-10s\n" "K" "wall_sec" "worker_sec" "total_frames" "avg/task" "result"
ALL_OK=1
for K in $K_VALUES; do
  START=$(date +%s.%N)
  OUT=$(FQ_CAPACITY_LOG2="$FQ_CAPACITY_LOG2" K_THRESHOLD="$K" timeout 300 "./$BIN" "$NPROC" "${DATA_ARGS[@]}" 2>&1)
  END=$(date +%s.%N)
  WALL=$(awk -v s="$START" -v e="$END" 'BEGIN{printf "%.2f", e-s}')
  if ! echo "$OUT" | grep -q "379a: PASS"; then
    ALL_OK=0
    printf "%-4s %-10s %-12s %-16s %-12s %-10s\n" "$K" "$WALL" "-" "-" "-" "FAIL"
    echo "$OUT" | tail -5
    fail "K=$K" "did not report 379a PASS"
  else
    FRAMES=$(echo "$OUT" | grep -oE 'total_frames=[0-9]+' | head -n1 | cut -d= -f2)
    AVG=$(echo "$OUT" | grep -oE 'avg_frames_per_task=[0-9.]+' | head -n1 | cut -d= -f2)
    WORKER_SEC=$(echo "$OUT" | grep -oE 'worker_elapsed=[0-9.]+' | head -n1 | cut -d= -f2)
    printf "%-4s %-10s %-12s %-16s %-12s %-10s\n" "$K" "$WALL" "${WORKER_SEC:-N/A}" "$FRAMES" "$AVG" "PASS"
    pass "K=$K (wall=${WALL}s worker=${WORKER_SEC:-N/A}s frames=$FRAMES avg=$AVG)"
  fi
done
echo "=========================================================================="

if [[ "$ALL_OK" -eq 1 ]]; then
  pass "all_K_values_passed_on_fixed_dataset"
fi

echo ""
echo "===== 379b summary ====="
echo "OK=$PASS  FAIL=$FAIL"
[[ "$FAIL" -gt 0 ]] && exit 1
echo "379b PASSED: K sweep on a single fixed dataset, apples-to-apples."
echo "Look at the table above -- the K where wall_sec stops improving (or"
echo "starts getting worse) as K increases toward 374's near-zero-queue-usage"
echo "behavior is the practical candidate threshold for 379c's GPU port."
echo "No 374 kernel code was touched."
exit 0
