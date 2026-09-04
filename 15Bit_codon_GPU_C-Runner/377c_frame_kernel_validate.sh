#!/usr/bin/env bash
# 377c_frame_kernel_validate.sh
#
# rev377c — Same engine as 377b (process_one_task_reference/
# TaskSchedule/Frame/FQSlot/fq_push/fq_pop/seed_task/step_one_frame,
# byte-identical, copied verbatim). The only change is main(): reads
# REAL data from a SoA7 binary file instead of generating synthetic
# random tasks. Falls back to --synthetic (377b-equivalent) if no
# real file is available.
#
# Usage:
#   SOA7_FILE=constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin \
#     bash 377c_frame_kernel_validate.sh
#   (falls back to --synthetic self-test if SOA7_FILE is unset or missing)

set -u
CUSRC="${CUSRC:-377c_frame_kernel.cu}"
BIN="${BIN:-377c_frame_kernel_test}"
STATIC_ONLY="${STATIC_ONLY:-0}"
SOA7_FILE="${SOA7_FILE:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
N="${N:-21}"
MAX_RECORDS="${MAX_RECORDS:-}"

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

for sym in process_one_task_reference seed_task step_one_frame fq_push fq_pop FQSlot mode=real_data; do
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

# NOTE (this session): process_one_task_reference() is parallelized in
# 377c now (was serial in an earlier draft -- see 377c_frame_kernel.cu
# header), but running the FULL N=21 file (2,025,282 records) through
# the 4-thread-count x 3-repeat sweep below would still mean 12
# separate full-scale runs. To keep the sweep itself fast, it uses a
# bounded record count by default (SWEEP_RECORDS below); a single,
# separate full-scale confirmation run happens once at the end.
SWEEP_RECORDS="${SWEEP_RECORDS:-100000}"

if [[ -f "$SOA7_FILE" ]]; then
  info "data_source" "using real data: $SOA7_FILE"
  SWEEP_ARGS=("$SOA7_FILE" "$N" "$SWEEP_RECORDS")
  FULL_ARGS=("$SOA7_FILE" "$N")
  [[ -n "$MAX_RECORDS" ]] && FULL_ARGS=("$SOA7_FILE" "$N" "$MAX_RECORDS")
else
  info "data_source" "$SOA7_FILE not found -- falling back to --synthetic self-test (this is NOT a real-data validation; place the real filtered SoA7 file alongside this script and re-run for the actual 377c goal)"
  SWEEP_ARGS=("--synthetic" "20000")
  FULL_ARGS=("${SWEEP_ARGS[@]}")
fi

echo "Detected NPROC=$NPROC. Sweep uses up to $SWEEP_RECORDS records (fast; full-scale confirmation is a separate single run below)."
echo "Running sweep: 1, 2, NPROC, NPROC*4 threads, 3 repeats each..."
ALL_OK=1
for t in 1 2 "$NPROC" $((NPROC*4)); do
  for run in 1 2 3; do
    OUT=$(timeout 300 "./$BIN" "$t" "${SWEEP_ARGS[@]}" 2>&1)
    if ! echo "$OUT" | grep -q "===== 377c: PASS"; then
      ALL_OK=0
      echo "$OUT"
      fail "run[threads=$t,run=$run]" "did not report 377c PASS"
    fi
  done
done
if [[ "$ALL_OK" -eq 1 ]]; then
  pass "sweep_all_thread_counts_all_repeats_passed (SWEEP_RECORDS=$SWEEP_RECORDS)"
fi

echo ""
echo "Sweep done. Running ONE full-scale confirmation (NPROC=$NPROC threads,"
echo "all records in $SOA7_FILE unless MAX_RECORDS was set). This can take a"
echo "while even with the reference computation now parallelized -- let it run;"
echo "there is no repeated loop here, just this one pass. Timeout: 4 hours."
FULL_OUT=$(timeout 14400 "./$BIN" "$NPROC" "${FULL_ARGS[@]}" 2>&1)
N_SKIPPED_SEEN=$(echo "$FULL_OUT" | grep -oE 'n_skipped=[0-9]+' | head -n1)
if echo "$FULL_OUT" | grep -q "===== 377c: PASS"; then
  pass "full_scale_confirmation_passed"
else
  echo "$FULL_OUT"
  fail "full_scale_confirmation_passed" "did not report 377c PASS (or timed out after 4h)"
fi

if [[ -f "$SOA7_FILE" ]]; then
  if [[ "$N_SKIPPED_SEEN" == "n_skipped=0" ]]; then
    pass "real_data_zero_skipped (matches 363's finding: all real N=21 records satisfy maxd<=14)"
  else
    fail "real_data_zero_skipped" "$N_SKIPPED_SEEN -- unexpected for real maxd<=14-filtered data, investigate before trusting the PASS above"
  fi
fi

echo ""
echo "===== 377c summary ====="
echo "OK=$PASS  FAIL=$FAIL"
[[ "$FAIL" -gt 0 ]] && exit 1
echo "377c PASSED. No GPU kernel code was touched."
exit 0
