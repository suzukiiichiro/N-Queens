#!/usr/bin/env bash
# 382_validate.sh
#
# rev382 — Validation harness for the CONSOLIDATED best-known-working
# state of the 375-381e CUDA C hybrid-kernel research track
# (382_kernel_dfs_hybrid.cu, K_THRESHOLD=13, real-hardware confirmed:
# total_sum=314666222712 MATCH, kernel_ms=263,003.750). No .cu code
# changes in this revision -- pure consolidation, same treatment as
# 374's own "374Py_" renaming pass. Speed comparison note: this is
# ~31% SLOWER than 374's own kernel_ms (~201,232ms) -- 382 is NOT a
# replacement for the production kernel, it preserves the K-sweep
# research infrastructure at a verified-correct checkpoint.

set -u
CUSRC="${CUSRC:-382_kernel_dfs_hybrid.cu}"
BIN="${BIN:-382_kernel_dfs_hybrid}"
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
echo "===== 382 summary ====="
echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then
  echo "FAILED CHECKS:"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  exit 1
fi
echo "382 PASSED: current best-known-working state of the 375-381e"
echo "hybrid-kernel research track (K_THRESHOLD=13, kernel_ms should be"
echo "~263,000ms). REMINDER: 382Py_kernel_maxd14_final.py + 374's own"
echo "364_kernel_maxd14.cu remain the actual PRODUCTION kernel"
echo "(kernel_ms~=201,232, faster than this file at K=13). This file"
echo "preserves the K-sweep research infrastructure for future work on"
echo "the original question: does K<13 actually improve on 374's static"
echo "grid-stride once real overflow-sharing kicks in?"
exit 0
