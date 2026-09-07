#!/usr/bin/env bash
# 392_validate.sh
#
# rev392 -- Build-only validation for both the maxd16 CUDA C port AND
# the .py rename (391Py -> 392Py, per Suzuki's request that the .py
# carry the same revision number as the .cu). This does NOT attempt to
# run either the GPU kernel or the CPU test harness against real N=23
# data -- deliberately deferred until after stepping back to N=21 for
# ncu-based speedup work (Suzuki's own stated plan).
#
# Deliberately does NOT register the .cu binary in
# crunner_dispatch_table() -- doing so would let bare -g attempt N=23's
# full run automatically, which is exactly what this plan defers.

set -u
CUSRC="${CUSRC:-392_kernel_maxd16.cu}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
GPU_BIN="${GPU_BIN:-392_kernel_maxd16}"
CPU_BIN="${CPU_BIN:-392_kernel_maxd16_cputest}"
GCC="${GCC:-gcc}"
PY_SRC="${PY_SRC:-392Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
PY_BIN="${PY_BIN:-392Py_kernel_maxd14_final}"
CODON="${CODON:-codon}"

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

if [[ ! -f "$PY_SRC" ]]; then
  fail "file_present[$PY_SRC]" "not found in $(pwd)"
  exit 1
fi
pass "file_present[$PY_SRC]"

if [[ ! -f "$HELPER_SRC" ]]; then
  fail "file_present[$HELPER_SRC]" "not found in $(pwd)"
  exit 1
fi
pass "file_present[$HELPER_SRC]"

if grep -qE '^REV_TAG:str="392"' "$PY_SRC"; then
  pass "rev_tag_is_392 (was caught stale at 388 since 388 itself)"
else
  fail "rev_tag_is_392" "expected REV_TAG:str=\"392\" in $PY_SRC"
fi

for sym in kernel_dfs_iter_gpu_maxd16 process_one_task MAXD16_ANCESTOR "gpu-run-done" "gpu-run-correctness"; do
  if grep -q -- "$sym" "$CUSRC"; then
    pass "symbol_present[$sym]"
  else
    fail "symbol_present[$sym]" "'$sym' not found in $CUSRC"
  fi
done
if grep -qE '^#define MAXD16_ANCESTOR 15$' "$CUSRC"; then
  pass "ancestor_constant_is_15"
else
  fail "ancestor_constant_is_15" "expected '#define MAXD16_ANCESTOR 15' in $CUSRC"
fi
if [[ "$FAIL" -gt 0 ]]; then exit 1; fi

# ---------------------------------------------------------------------
# 1. nvcc build (device). Build-only -- no run, no GPU time spent.
# ---------------------------------------------------------------------
if [[ ! -x "$NVCC" ]] && ! command -v nvcc >/dev/null 2>&1; then
  fail "nvcc_toolchain_present" "$NVCC not executable and 'nvcc' not on PATH"
  exit 1
fi
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
echo "Building $CUSRC (device) with $NVCC -arch=$ARCH (build-only, not run)..."
rm -f "$GPU_BIN"
GPU_BUILD_LOG="392_gpu_build_$(date +%Y%m%d_%H%M%S).log"
"$NVCC" -O3 -arch="$ARCH" -o "$GPU_BIN" "$CUSRC" 2>&1 | tee "$GPU_BUILD_LOG"
if [[ ! -x "$GPU_BIN" ]]; then
  fail "nvcc_build_succeeded" "binary $GPU_BIN was not produced -- see $GPU_BUILD_LOG"
  exit 1
fi
pass "nvcc_build_succeeded"

# ---------------------------------------------------------------------
# 2. gcc CPU-test build (host-only, no CUDA toolkit needed). -x c MUST
#    precede the source file -- see this file's own header note on the
#    pre-existing 364/388/389 documentation issue with the opposite
#    order.
# ---------------------------------------------------------------------
if ! command -v "$GCC" >/dev/null 2>&1; then
  fail "gcc_toolchain_present" "'$GCC' not found on PATH"
  exit 1
fi
pass "gcc_toolchain_present"
echo "Building $CUSRC (CPU test harness) with $GCC -x c (build-only, not run)..."
rm -f "$CPU_BIN"
CPU_BUILD_LOG="392_cpu_build_$(date +%Y%m%d_%H%M%S).log"
"$GCC" -O2 -Wall -Wextra -x c -o "$CPU_BIN" "$CUSRC" -lm 2>&1 | tee "$CPU_BUILD_LOG"
if [[ ! -x "$CPU_BIN" ]]; then
  fail "gcc_build_succeeded" "binary $CPU_BIN was not produced -- see $CPU_BUILD_LOG"
  exit 1
fi
pass "gcc_build_succeeded"

if [[ -s "$CPU_BUILD_LOG" ]] && grep -qi "warning" "$CPU_BUILD_LOG"; then
  fail "gcc_build_zero_warnings" "warnings present in $CPU_BUILD_LOG -- local sandbox build was clean under -Wall -Wextra, worth checking why this differs"
else
  pass "gcc_build_zero_warnings (matches the local off-hardware check)"
fi

# quick smoke test: usage message only, no real data needed
USAGE_OUT="$("./$CPU_BIN" 2>&1 || true)"
if echo "$USAGE_OUT" | grep -q "Usage:"; then
  pass "cpu_binary_runs_and_prints_usage"
else
  fail "cpu_binary_runs_and_prints_usage" "expected a Usage: message, got: $USAGE_OUT"
fi

# ---------------------------------------------------------------------
# 3. codon build (392Py). Build-only, same spirit as the C side --
#    confirms the pure rename (391Py -> 392Py) plus the REV_TAG fix
#    didn't break anything, without spending any GPU time.
# ---------------------------------------------------------------------
if ! command -v "$CODON" >/dev/null 2>&1; then
  fail "codon_toolchain_present" "'codon' not found on PATH"
  exit 1
fi
pass "codon_toolchain_present"
echo "Building $PY_SRC with $CODON build -release (build-only, not run)..."
rm -f "$PY_BIN"
PY_BUILD_LOG="392_py_build_$(date +%Y%m%d_%H%M%S).log"
"$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$PY_BUILD_LOG"
if [[ ! -x "$PY_BIN" ]]; then
  fail "codon_build_succeeded" "binary $PY_BIN was not produced -- see $PY_BUILD_LOG"
  exit 1
fi
pass "codon_build_succeeded"

echo ""
echo "===== 392 summary ====="
echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then
  echo "FAILED CHECKS:"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  exit 1
fi
echo "392 PASSED: the device (nvcc) and host-only CPU-test (gcc) builds"
echo "of the .cu succeed, AND the renamed 392Py builds under codon."
echo "Per plan, no run was attempted -- next step is stepping back to"
echo "N=21 with ncu, not N=23 execution."
exit 0
