#!/usr/bin/env bash
# 381a_validate.sh
#
# rev381c — First real GPU test of the seed_kernel + hybrid_kernel
# two-stage design (380 spec). K_THRESHOLD fixed at 13 (see file
# header of 381c_hybrid_kernel_lowcontention.cu): this run's only job is
# confirming the new queue-based plumbing reproduces 374's own oracle
# (314666222712) on real N=21 data. Speed is reported but NOT the
# pass/fail criterion here -- a small regression from 374's
# kernel_ms is expected (extra kernel-launch overhead) and acceptable
# at this stage. This script has NOT been run against real nvcc/GPU
# by the author (no CUDA toolkit in the authoring sandbox) -- this is
# its first real-hardware test.

set -u
CUSRC="${CUSRC:-381c_hybrid_kernel_lowcontention.cu}"
BIN="${BIN:-381c_hybrid_kernel_lowcontention}"
STATIC_ONLY="${STATIC_ONLY:-0}"
SOA7_FILE="${SOA7_FILE:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
N="${N:-21}"
EXPECTED_ORACLE="${EXPECTED_ORACLE:-314666222712}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
K_THRESHOLD="${K_THRESHOLD:-13}"
FQ_CAPACITY_LOG2="${FQ_CAPACITY_LOG2:-24}"

PASS=0
FAIL=0
declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }

# ---------------------------------------------------------------------
# 0. sudo not needed here (no ncu), skip that check from prior scripts.
# ---------------------------------------------------------------------
if [[ ! -f "$CUSRC" ]]; then
  fail "file_present[$CUSRC]" "not found in $(pwd)"
  exit 1
fi
pass "file_present[$CUSRC]"

if [[ ! -f "$SOA7_FILE" ]]; then
  fail "soa7_file_present[$SOA7_FILE]" "not found -- this test needs real N=21 data (from 374's earlier run)"
  exit 1
fi
pass "soa7_file_present[$SOA7_FILE]"

for sym in kernel_dfs_hybrid_maxd14 process_task_hybrid run_hybrid_episode fq_push fq_pop FQSlot LocalStack; do
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

# ---------------------------------------------------------------------
# 1. nvcc build. This is the FIRST real compilation of this file --
#    it was authored without a CUDA toolkit available, so build
#    errors here are expected to be a real possibility, not a sign
#    of a deeper design problem. Report the log clearly either way.
# ---------------------------------------------------------------------
if [[ ! -x "$NVCC" ]] && ! command -v nvcc >/dev/null 2>&1; then
  fail "nvcc_toolchain_present" "$NVCC not executable and 'nvcc' not on PATH"
  exit 1
fi
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
echo "Building $CUSRC with $NVCC -arch=$ARCH (FIRST real compile of this file)..."
rm -f "$BIN"
"$NVCC" -O3 -arch="$ARCH" -o "$BIN" "$CUSRC" 2>&1 | tee "${BIN}_build_$(date +%Y%m%d_%H%M%S).log"
if [[ ! -x "$BIN" ]]; then
  fail "nvcc_build_succeeded" "binary $BIN was not produced -- see build log above. This file was never nvcc-compiled before this run, so please share the exact error(s) if this fails."
  exit 1
fi
pass "nvcc_build_succeeded"

# ---------------------------------------------------------------------
# 2. Real GPU run against the real oracle. K_THRESHOLD=13 means the
#    local stack alone should suffice for every task (26 slots =
#    MAXD14_ANCESTOR*2, matching 374's own stack size exactly) -- the
#    shared queue should see traffic only from the initial seed, never
#    from overflow. If a queue-overflow FAIL happens even so, that
#    itself is a meaningful finding (something differs from the CPU
#    characterization in 379b), not just a capacity knob to crank up.
# ---------------------------------------------------------------------
echo "Running: K_THRESHOLD=$K_THRESHOLD FQ_CAPACITY_LOG2=$FQ_CAPACITY_LOG2 ./$BIN $N $SOA7_FILE /tmp/381a_results.bin $EXPECTED_ORACLE"
RUN_LOG="${BIN}_run_$(date +%Y%m%d_%H%M%S).log"
K_THRESHOLD="$K_THRESHOLD" FQ_CAPACITY_LOG2="$FQ_CAPACITY_LOG2" \
  "./$BIN" "$N" "$SOA7_FILE" /tmp/381a_results.bin "$EXPECTED_ORACLE" 2>&1 | tee "$RUN_LOG"

if grep -q '\[gpu-hybrid-run-done\]' "$RUN_LOG"; then
  pass "gpu_run_completed"
else
  fail "gpu_run_completed" "no [gpu-hybrid-run-done] line in $RUN_LOG -- check for crashes/overflow above"
  exit 1
fi

if grep -q '\[gpu-hybrid-run-correctness\] MATCH' "$RUN_LOG"; then
  pass "gpu_total_matches_oracle (expected=$EXPECTED_ORACLE)"
else
  fail "gpu_total_matches_oracle" "no MATCH line found -- see $RUN_LOG for MISMATCH or missing correctness line"
fi

echo ""
echo "===== 381c summary ====="
echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then
  echo "FAILED CHECKS:"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  exit 1
fi
echo "381c PASSED: kernel_dfs_hybrid_maxd14 (K_THRESHOLD=13) with low-"
echo "contention termination (active_static_workers decremented once per"
echo "THREAD, not once per TASK) reproduces 374's own oracle on real N=21"
echo "data. Compare kernel_ms above to 374 (~201,232ms), 381a (285,841ms +"
echo "seed 13,259ms), and 381b (613,041ms, the atomic-contention regression"
echo "this revision fixes). This should land closest to 374 of all four."
echo "381d+: sweep K_THRESHOLD downward on real GPU hardware for actual speed data."
exit 0
