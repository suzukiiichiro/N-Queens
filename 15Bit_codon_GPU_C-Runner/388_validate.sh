#!/usr/bin/env bash
# 388_validate.sh
#
# rev388 -- Validation harness for the -d flag (r1: position-
# independence) AND the log/console separation fix (r2: bench_mode==37's
# own dispatch messages no longer print unconditionally to the console
# -- they always go to crunner_logs/dispatch.log, and only ALSO print
# to the console when -d is passed). r2 was a direct response to
# Suzuki's feedback on 387's real-hardware output.

set -u
PY_SRC="${PY_SRC:-388Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
BIN="${BIN:-388Py_kernel_maxd14_final}"
CRUNNER_SRC="${CRUNNER_SRC:-388_kernel_maxd14.cu}"
CRUNNER_BIN="${CRUNNER_BIN:-388_kernel_maxd14}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
STATIC_ONLY="${STATIC_ONLY:-0}"
CODON="${CODON:-codon}"
N="${N:-21}"
EXPECTED_ORACLE="${EXPECTED_ORACLE:-314666222712}"
REV_TAG="${REV_TAG:-388}"

PASS=0
FAIL=0
declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }

for f in "$PY_SRC" "$HELPER_SRC"; do
  if [[ ! -f "$f" ]]; then
    fail "file_present[$f]" "not found in $(pwd)"
    exit 1
  fi
  pass "file_present[$f]"
done

if [[ ! -x "./$CRUNNER_BIN" ]]; then
  # 388 r4: this binary's source (388_kernel_maxd14.cu) is a brand new
  # filename (pure rename of 364_kernel_maxd14.cu, code region confirmed
  # byte-identical by diff before shipping) -- it has never been built
  # under this name on real hardware, so build it here rather than
  # just failing if missing.
  if [[ ! -f "$CRUNNER_SRC" ]]; then
    fail "crunner_source_present[$CRUNNER_SRC]" "not found in $(pwd)"
    exit 1
  fi
  pass "crunner_source_present[$CRUNNER_SRC]"
  if [[ ! -x "$NVCC" ]] && ! command -v nvcc >/dev/null 2>&1; then
    fail "nvcc_toolchain_present" "$NVCC not executable and 'nvcc' not on PATH"
    exit 1
  fi
  [[ ! -x "$NVCC" ]] && NVCC="nvcc"
  echo "Building $CRUNNER_SRC with $NVCC -arch=$ARCH (first build under this filename)..."
  CRUNNER_BUILD_LOG="388_crunner_build_$(date +%Y%m%d_%H%M%S).log"
  "$NVCC" -O3 -arch="$ARCH" -o "$CRUNNER_BIN" "$CRUNNER_SRC" 2>&1 | tee "$CRUNNER_BUILD_LOG"
  if [[ ! -x "$CRUNNER_BIN" ]]; then
    fail "crunner_binary_present[$CRUNNER_BIN]" "nvcc build failed -- see $CRUNNER_BUILD_LOG"
    exit 1
  fi
fi
pass "crunner_binary_present[$CRUNNER_BIN]"

CRUNNER_INPUT="${CRUNNER_INPUT:-constellations_N${N}_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
if [[ -f "$CRUNNER_INPUT" ]]; then
  pass "crunner_input_present[$CRUNNER_INPUT]"
else
  fail "crunner_input_present[$CRUNNER_INPUT]" "maxd14-filtered SoA reference dump not found"
  exit 1
fi

if grep -qE '^\s*if\s+tok=="-d":' "$PY_SRC"; then
  pass "d_flag_filter_present"
else
  fail "d_flag_filter_present" "expected an argv-filtering loop matching tok==\"-d\" in $PY_SRC"
fi

if grep -q "def crunner_dispatch_log" "$HELPER_SRC"; then
  pass "crunner_dispatch_log_present_in_helper"
else
  fail "crunner_dispatch_log_present_in_helper" "crunner_dispatch_log() not found in $HELPER_SRC"
fi

if grep -qE "^REV_TAG:str=\"${REV_TAG}\"" "$PY_SRC"; then
  pass "rev_tag_constant_matches[$REV_TAG]"
else
  fail "rev_tag_constant_matches" "expected REV_TAG:str=\"$REV_TAG\" in $PY_SRC"
fi

if [[ "$FAIL" -gt 0 ]]; then exit 1; fi

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "STATIC_ONLY=1: stopping after static checks."
  exit 0
fi

if ! command -v "$CODON" >/dev/null 2>&1; then
  fail "codon_toolchain_present" "'codon' not found on PATH"
  exit 1
fi
pass "codon_toolchain_present"

echo "Building $PY_SRC with $CODON build -release..."
rm -f "$BIN"
BUILD_LOG="388_build_$(date +%Y%m%d_%H%M%S).log"
"$CODON" build -release -o "$BIN" "$PY_SRC" 2>&1 | tee "$BUILD_LOG"

if [[ ! -x "$BIN" ]]; then
  fail "codon_build_succeeded" "binary $BIN was not produced -- see $BUILD_LOG"
  exit 1
fi
pass "codon_build_succeeded"

# ---------------------------------------------------------------------
# 1. cheap N=6 checks: -d position-independence, unaffected by r2.
# ---------------------------------------------------------------------
echo "Running: ./$BIN -g 6 6 32 484 0 0 5 0 -d"
LOG1="388_run_trailing_$(date +%Y%m%d_%H%M%S).log"
./"$BIN" -g 6 6 32 484 0 0 5 0 -d 2>&1 | tee "$LOG1"

if grep -qE "^[[:space:]]*6:[[:space:]]*4[[:space:]]" "$LOG1"; then
  pass "trailing_-d_positional_args_unaffected (N=6 total=4)"
else
  fail "trailing_-d_positional_args_unaffected" "N=6 row missing or wrong -- see $LOG1"
fi

if grep -q '\[debug-mode\] enabled via -d' "$LOG1"; then
  pass "trailing_-d_banner_present"
else
  fail "trailing_-d_banner_present" "no [debug-mode] banner -- see $LOG1"
fi

echo "Running: ./$BIN -g 6 6 32 484 0 0 5 0   (no -d)"
LOG3="388_run_nodebug_$(date +%Y%m%d_%H%M%S).log"
./"$BIN" -g 6 6 32 484 0 0 5 0 2>&1 | tee "$LOG3"

if grep -q '\[debug-mode\]' "$LOG3"; then
  fail "no_d_means_no_banner" "unexpected [debug-mode] banner appeared without -d -- see $LOG3"
else
  pass "no_d_means_no_banner"
fi

# ---------------------------------------------------------------------
# 2. THE actual r2 check: N=21 via bench_mode=37, WITHOUT -d. The
#    console must show ONLY the clean table row (no [crunner-*] lines
#    at all), while crunner_logs/dispatch.log must contain the full
#    [crunner-dispatch-summary] line anyway. This is a real ~3-4
#    minute run (same CRunner dispatch path 385/386/387 already
#    proved correct on real hardware -- this check is about WHERE the
#    diagnostic text goes, not whether the computation is right).
# ---------------------------------------------------------------------
rm -rf "${REV_TAG}_crunner_logs"
echo "Running: ./$BIN -g $N $N 32 484 0 0 7 37   (no -d -- this is the real measurement-mode check)"
RUN_LOG="388_run_measure_$(date +%Y%m%d_%H%M%S).log"
./"$BIN" -g "$N" "$N" 32 484 0 0 7 37 2>&1 | tee "$RUN_LOG"

if grep -qE "^${N}:\s*${EXPECTED_ORACLE}\s.*[[:space:]]ok[[:space:]]*\$" "$RUN_LOG"; then
  pass "N21_table_row_correct_without_-d"
else
  fail "N21_table_row_correct_without_-d" "N=$N row missing, wrong total, or missing 'ok' -- see $RUN_LOG"
fi

if grep -q '\[crunner-dispatch-summary\]' "$RUN_LOG"; then
  fail "console_clean_without_-d" "[crunner-dispatch-summary] appeared on console WITHOUT -d -- see $RUN_LOG (r2's console/file separation did not take effect)"
else
  pass "console_clean_without_-d (no [crunner-*] diagnostic lines leaked to console)"
fi

if [[ -f "${REV_TAG}_crunner_logs/dispatch.log" ]] && grep -q '\[crunner-dispatch-summary\]' "${REV_TAG}_crunner_logs/dispatch.log"; then
  pass "dispatch_log_file_contains_summary (in the rev-numbered ${REV_TAG}_crunner_logs/ directory)"
else
  fail "dispatch_log_file_contains_summary" "${REV_TAG}_crunner_logs/dispatch.log missing or does not contain [crunner-dispatch-summary]"
fi

echo ""
echo "===== 388 summary ====="
echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then
  echo "FAILED CHECKS:"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  exit 1
fi
echo "388 PASSED: -d is position-independent, plain -g measurement runs"
echo "show ONLY the clean record-output table on the console, and every"
echo "[crunner-*] diagnostic line is still fully preserved in"
echo "${REV_TAG}_crunner_logs/dispatch.log regardless of -d."
exit 0

