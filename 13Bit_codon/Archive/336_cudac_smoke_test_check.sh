#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# 336_cudac_smoke_test_check.sh
#
# Standalone check for the FIRST concrete buried-idea C-2 (rev84 CUDA C
# runner) artifact. Completely independent of the main solver's N=21
# validate harness (336Py_cudac_smoke_test_validate_N21_full_once.sh) --
# does not build, run, or touch the Codon solver in any way.
#
# What this does:
#   1. Build 336_cudac_smoke_test.cu with nvcc, targeting the A10G's
#      confirmed Compute Capability (8.6 -> -arch=sm_86, measured in 335
#      via `nvidia-smi --query-gpu=compute_cap`).
#   2. Run the resulting binary (this DOES touch the GPU, unlike 334/335's
#      toolchain probes which were nvcc/cuobjdump/nvidia-smi metadata only).
#   3. Parse its self-verification output (match=True/False) and record a
#      simple PASS/FAIL summary.
#
# This script does NOT gate or affect the main N21 validate harness in any
# way. If the smoke test fails, that is itself useful design information
# for buried-idea C-2 and does not indicate any problem with the adopted
# w3_j7 solver (328-336 lineage), which is unaffected either way.
#
# nvcc path: uses `command -v nvcc` (334/335 confirmed this resolves to
# /usr/local/cuda/bin/nvcc on cudacodon), falling back to a direct path
# check if PATH does not have it.
# =============================================================================

SRC=${SRC:-./336_cudac_smoke_test.cu}
BIN=${BIN:-./336_cudac_smoke_test}
ARCH=${ARCH:-sm_86}
LOG_ROOT=${LOG_ROOT:-.}
TS=$(date -u +%Y%m%d_%H%M%S)
LOGDIR="${LOG_ROOT%/}/336_cudac_smoke_test_check_${TS}"
mkdir -p "$LOGDIR"
BUILD_LOG="$LOGDIR/build.log"
RUN_LOG="$LOGDIR/run.log"
SUMMARY="$LOGDIR/summary.tsv"
printf 'check\texpected\tactual\tstatus\n' > "$SUMMARY"

echo "[start] 336 cudac-smoke-test check script"
echo "[info] SRC=$SRC BIN=$BIN ARCH=$ARCH LOGDIR=$LOGDIR"

# ---- locate nvcc (PATH first, then the known 334/335 fallback path) ----
NVCC=""
if command -v nvcc >/dev/null 2>&1; then
  NVCC=$(command -v nvcc)
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  NVCC=/usr/local/cuda/bin/nvcc
  echo "[info] nvcc not on PATH; falling back to known location /usr/local/cuda/bin/nvcc (confirmed present in 335)" | tee -a "$BUILD_LOG"
fi

if [[ -z "$NVCC" ]]; then
  echo "[error] nvcc not found on PATH or at /usr/local/cuda/bin/nvcc" | tee -a "$BUILD_LOG" >&2
  printf 'nvcc_located\tpresent\tmissing\tFAIL\n' >> "$SUMMARY"
  printf 'build_exit\t0\tskipped\tFAIL\n' >> "$SUMMARY"
  printf 'run_exit\t0\tskipped\tFAIL\n' >> "$SUMMARY"
  printf 'smoke_match\tTrue\tskipped\tFAIL\n' >> "$SUMMARY"
  echo "[logdir] $LOGDIR"
  exit 1
fi
echo "[info] using nvcc: $NVCC" | tee -a "$BUILD_LOG"
printf 'nvcc_located\tpresent\t%s\tOK\n' "$NVCC" >> "$SUMMARY"

if [[ ! -f "$SRC" ]]; then
  echo "[error] source not found: $SRC" >&2
  printf 'source_present\tpresent\tmissing\tFAIL\n' >> "$SUMMARY"
  exit 1
fi
printf 'source_present\tpresent\tpresent\tOK\n' >> "$SUMMARY"

# ---- build ----
echo "[build] $NVCC -O3 -arch=$ARCH -o $BIN $SRC" | tee -a "$BUILD_LOG"
set +e
"$NVCC" -O3 -arch="$ARCH" -o "$BIN" "$SRC" >>"$BUILD_LOG" 2>&1
build_rc=$?
set -e
printf 'build_exit\t0\t%s\t%s\n' "$build_rc" "$([[ $build_rc -eq 0 ]] && echo OK || echo FAIL)" >> "$SUMMARY"
if (( build_rc != 0 )); then
  echo "[error] build failed (exit=$build_rc), see $BUILD_LOG" >&2
  cat "$SUMMARY"
  echo "[logdir] $LOGDIR"
  exit 1
fi

# ---- run ----
echo "[run] $BIN" | tee -a "$RUN_LOG"
set +e
"$BIN" >>"$RUN_LOG" 2>&1
run_rc=$?
set -e
cat "$RUN_LOG"
printf 'run_exit\t0\t%s\t%s\n' "$run_rc" "$([[ $run_rc -eq 0 ]] && echo OK || echo FAIL)" >> "$SUMMARY"

# ---- parse self-verification ----
if grep -q 'match=True' "$RUN_LOG"; then
  printf 'smoke_match\tTrue\tTrue\tOK\n' >> "$SUMMARY"
  MATCH_OK=1
elif grep -q 'match=False' "$RUN_LOG"; then
  printf 'smoke_match\tTrue\tFalse\tFAIL\n' >> "$SUMMARY"
  MATCH_OK=0
else
  printf 'smoke_match\tTrue\tnot found in run output\tFAIL\n' >> "$SUMMARY"
  MATCH_OK=0
fi

echo "================================================================"
echo "[summary]"
cat "$SUMMARY"
echo "[logdir] $LOGDIR"

if (( build_rc != 0 )) || (( run_rc != 0 )) || (( MATCH_OK != 1 )); then
  echo "[result] FAIL -- see $LOGDIR for details" >&2
  exit 1
fi

echo "[result] OK -- nvcc -arch=$ARCH build+run round-trip confirmed on this GPU"
