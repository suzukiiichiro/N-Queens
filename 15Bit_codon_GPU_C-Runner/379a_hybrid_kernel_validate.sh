#!/usr/bin/env bash
# 379a_hybrid_kernel_validate.sh
#
# rev379a — Hybrid local-stack + threshold-overflow-to-shared-queue
# (378 design), validated against process_one_task_reference on
# synthetic data first (363's own method), then against real data if
# available, across several K_THRESHOLD values to characterize how
# shared-queue traffic scales with K before any real GPU integration.

set -u
CUSRC="${CUSRC:-379a_hybrid_kernel.cu}"
BIN="${BIN:-379a_hybrid_kernel_test}"
STATIC_ONLY="${STATIC_ONLY:-0}"
SOA7_FILE="${SOA7_FILE:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
N="${N:-21}"
FQ_CAPACITY_LOG2="${FQ_CAPACITY_LOG2:-28}"

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

for sym in process_one_task_reference seed_task run_hybrid_episode LocalStack fq_push fq_pop FQSlot; do
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

if [[ -f "$SOA7_FILE" ]]; then
  info "data_source" "using real data (small slice): $SOA7_FILE"
  HAVE_REAL_DATA=1
else
  info "data_source" "$SOA7_FILE not found -- using --synthetic self-test"
  HAVE_REAL_DATA=0
  DATA_ARGS=("--synthetic" "20000")
fi

echo "Sweeping K_THRESHOLD={1,4,8,13} at NPROC=$NPROC threads, checking correctness"
echo "and reporting how total_frames (shared-queue traffic) scales with K..."
echo "NOTE: low K behaves like 377b/c's full-decoupling (huge frame counts on"
echo "real data -- 377c measured ~1.87M frames/task average), so low-K sweep"
echo "points use a much smaller record count than high-K points."
records_for_K() {
  case "$1" in
    1)  echo 20 ;;
    4)  echo 200 ;;
    8)  echo 2000 ;;
    13) echo 5000 ;;
    *)  echo 200 ;;
  esac
}

ALL_OK=1
for K in 1 4 8 13; do
  KREC=$(records_for_K "$K")
  if [[ "$HAVE_REAL_DATA" -eq 1 ]]; then
    DATA_ARGS=("$SOA7_FILE" "$N" "$KREC")
  fi
  OUT=$(FQ_CAPACITY_LOG2="$FQ_CAPACITY_LOG2" K_THRESHOLD="$K" timeout 120 "./$BIN" "$NPROC" "${DATA_ARGS[@]}" 2>&1)
  if ! echo "$OUT" | grep -q "379a: PASS"; then
    ALL_OK=0
    echo "$OUT"
    fail "K_THRESHOLD=$K" "did not report 379a PASS"
  else
    FRAMES=$(echo "$OUT" | grep -oE 'total_frames=[0-9]+' | head -n1)
    AVG=$(echo "$OUT" | grep -oE 'avg_frames_per_task=[0-9.]+' | head -n1)
    pass "K_THRESHOLD=$K (records=$KREC $FRAMES $AVG)"
  fi
done

echo ""
echo "Multi-thread reproducibility check at K_THRESHOLD=8, 3 repeats per thread count..."
echo "NOTE: uses records_for_K(4)'s smaller record count (not K=8's own, larger one)"
echo "specifically so single-thread runs finish within a sane timeout here -- this repro"
echo "check is only about push/pop correctness across thread counts, not about"
echo "measuring K=8's true queue-traffic scale (that needs 379b's fixed-dataset sweep)."
if [[ "$HAVE_REAL_DATA" -eq 1 ]]; then
  REPRO_ARGS=("$SOA7_FILE" "$N" "$(records_for_K 4)")
else
  REPRO_ARGS=("${DATA_ARGS[@]}")
fi
for t in 1 2 "$NPROC"; do
  for run in 1 2 3; do
    OUT=$(FQ_CAPACITY_LOG2="$FQ_CAPACITY_LOG2" K_THRESHOLD=8 timeout 300 "./$BIN" "$t" "${REPRO_ARGS[@]}" 2>&1)
    if ! echo "$OUT" | grep -q "379a: PASS"; then
      ALL_OK=0
      echo "$OUT"
      fail "repro[threads=$t,run=$run]" "did not report 379a PASS"
    fi
  done
done
if [[ "$ALL_OK" -eq 1 ]]; then
  pass "all_K_values_and_repro_passed"
fi

echo ""
echo "===== 379a summary ====="
echo "OK=$PASS  FAIL=$FAIL"
[[ "$FAIL" -gt 0 ]] && exit 1
echo "379a PASSED: hybrid local-stack + threshold-overflow design proven"
echo "correct across K_THRESHOLD={1,4,8,13} and multiple thread counts."
echo "Higher K should show total_frames approaching 374's near-zero shared-"
echo "queue usage; lower K approaches 377b's full-decoupling frame count."
echo "No 374 kernel code was touched. Next: 379b real-data-scale sweep."
exit 0
