#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# 337_bin_format_reader_check.sh
#
# Standalone check for the SECOND concrete buried-idea C-2 (rev84 CUDA C
# runner) artifact. Completely independent of the main solver's N=21
# validate harness (337Py_bin_format_reader_design_validate_N21_full_once.sh)
# -- does not build, run, or touch the Codon solver in any way, and does
# NOT use the GPU at all (this is a pure host-side file-format check).
#
# What this does:
#   1. Build 337_bin_format_reader.cu with nvcc (host-only code; still
#      uses nvcc for toolchain consistency with 336, but no -arch/GPU
#      execution is actually exercised by this program).
#   2. Run the resulting binary against a .bin file. By default it looks
#      for one of the already-cached constellations_N21_*.bin files in
#      the current directory (produced by earlier revisions' N=21 runs);
#      a path can also be passed explicitly as $1.
#   3. Parse its output (size_mod16_ok / records_ok) and record a simple
#      PASS/FAIL summary, plus print the checksum for manual reference.
#
# This script does NOT gate or affect the main N21 validate harness in any
# way, and vice versa -- if no .bin file is found, that is reported as
# INFO/SKIP rather than a hard failure, since this is a spec-confirmation
# tool, not a correctness gate on the adopted w3_j7 solver.
# =============================================================================

SRC=${SRC:-./337_bin_format_reader.cu}
BIN=${BIN:-./337_bin_format_reader}
ARCH=${ARCH:-sm_86}
LOG_ROOT=${LOG_ROOT:-.}
BIN_FILE=${1:-}
TS=$(date -u +%Y%m%d_%H%M%S)
LOGDIR="${LOG_ROOT%/}/337_bin_format_reader_check_${TS}"
mkdir -p "$LOGDIR"
BUILD_LOG="$LOGDIR/build.log"
RUN_LOG="$LOGDIR/run.log"
SUMMARY="$LOGDIR/summary.tsv"
printf 'check\texpected\tactual\tstatus\n' > "$SUMMARY"

echo "[start] 337 bin-format-reader check script"
echo "[info] SRC=$SRC BIN=$BIN ARCH=$ARCH LOGDIR=$LOGDIR"

# ---- locate nvcc (PATH first, then the known 334/335 fallback path) ----
NVCC=""
if command -v nvcc >/dev/null 2>&1; then
  NVCC=$(command -v nvcc)
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  NVCC=/usr/local/cuda/bin/nvcc
  echo "[info] nvcc not on PATH; falling back to known location /usr/local/cuda/bin/nvcc" | tee -a "$BUILD_LOG"
fi

if [[ -z "$NVCC" ]]; then
  echo "[error] nvcc not found on PATH or at /usr/local/cuda/bin/nvcc" | tee -a "$BUILD_LOG" >&2
  printf 'nvcc_located\tpresent\tmissing\tFAIL\n' >> "$SUMMARY"
  printf 'build_exit\t0\tskipped\tFAIL\n' >> "$SUMMARY"
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

# ---- build (host-only code; -arch is harmless but kept for consistency) ----
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

# ---- locate a .bin file if none given ----
if [[ -z "$BIN_FILE" ]]; then
  BIN_FILE=$(ls -1 constellations_N21_*.bin 2>/dev/null | head -n 1 || true)
fi

if [[ -z "$BIN_FILE" ]]; then
  echo "[skip] no .bin file given and none of constellations_N21_*.bin found in $(pwd)" | tee -a "$RUN_LOG"
  printf 'bin_file_located\tpresent\tnone found (pass a path as \\$1 to run against a specific file)\tINFO\n' >> "$SUMMARY"
  echo "================================================================"
  echo "[summary]"
  cat "$SUMMARY"
  echo "[logdir] $LOGDIR"
  echo "[result] SKIPPED -- build succeeded but no .bin file was available to read; re-run with an explicit path: bash 337_bin_format_reader_check.sh /path/to/constellations_N21_6.bin"
  exit 0
fi

echo "[info] using bin file: $BIN_FILE"
printf 'bin_file_located\tpresent\t%s\tOK\n' "$BIN_FILE" >> "$SUMMARY"

# ---- run ----
echo "[run] $BIN $BIN_FILE" | tee -a "$RUN_LOG"
set +e
"$BIN" "$BIN_FILE" >>"$RUN_LOG" 2>&1
run_rc=$?
set -e
cat "$RUN_LOG"
printf 'run_exit\t0\t%s\t%s\n' "$run_rc" "$([[ $run_rc -eq 0 ]] && echo OK || echo FAIL)" >> "$SUMMARY"

# ---- parse self-verification ----
if grep -q 'size_mod16_ok=True' "$RUN_LOG"; then
  printf 'size_mod16_ok\tTrue\tTrue\tOK\n' >> "$SUMMARY"
  SIZE_OK=1
else
  printf 'size_mod16_ok\tTrue\tFalse or not found\tFAIL\n' >> "$SUMMARY"
  SIZE_OK=0
fi
if grep -q 'records_ok=True' "$RUN_LOG"; then
  printf 'records_ok\tTrue\tTrue\tOK\n' >> "$SUMMARY"
  RECORDS_OK=1
else
  printf 'records_ok\tTrue\tFalse or not found\tFAIL\n' >> "$SUMMARY"
  RECORDS_OK=0
fi
CHECKSUM=$(grep -o 'checksum_u64=[0-9]*' "$RUN_LOG" | head -n 1 | cut -d= -f2 || true)
printf 'checksum_u64\t(for manual cross-reference)\t%s\tINFO\n' "${CHECKSUM:-not found}" >> "$SUMMARY"

echo "================================================================"
echo "[summary]"
cat "$SUMMARY"
echo "[logdir] $LOGDIR"

if (( build_rc != 0 )) || (( run_rc != 0 )) || (( SIZE_OK != 1 )) || (( RECORDS_OK != 1 )); then
  echo "[result] FAIL -- see $LOGDIR for details" >&2
  exit 1
fi

echo "[result] OK -- bin format spec confirmed against real file $BIN_FILE, checksum_u64=$CHECKSUM"
