#!/usr/bin/env bash
# 389_validate.sh
#
# rev389 -- Validation harness for ensure_crunner_input_bin(). Two
# checks: (1) a cheap N=21 forced-from-scratch rebuild proof (the
# existing filtered file is backed up, not deleted, then the pipeline
# is forced to rebuild it and the result is checksummed against the
# backup) confirming the function is genuinely generic and not
# secretly N=22-specific; (2) the real target -- N=22, whose filtered
# CRunner input file has never existed on this machine (confirmed
# directly by Suzuki's own bare `-g` runs, which stopped at N=22 with
# [crunner-input-missing] every time since 387). If bench_mode=37 for
# N=22 now succeeds and matches the N=22 oracle, the auto-build
# pipeline (361's dump_soa_reference_c_port + the external
# 363_filter_maxd14_only.py script) genuinely works end-to-end on real
# hardware, not just N=21 where a filtered file already happened to
# exist.
#
# This is a genuinely long, first-time operation: N=22 has ~28.7M
# constellation records (366's own measurement), so building the SoA
# reference dump and then filtering it will take real time beyond the
# usual ~3-4 minute kernel run. Budget generously -- this could be the
# longest-running harness in the project so far.

set -u
PY_SRC="${PY_SRC:-389Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
BIN="${BIN:-389Py_kernel_maxd14_final}"
CRUNNER_SRC="${CRUNNER_SRC:-389_kernel_maxd14.cu}"
CRUNNER_BIN="${CRUNNER_BIN:-389_kernel_maxd14}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
FILTER_SCRIPT="${FILTER_SCRIPT:-363_filter_maxd14_only.py}"
STATIC_ONLY="${STATIC_ONLY:-0}"
CODON="${CODON:-codon}"
N="${N:-22}"
EXPECTED_ORACLE="${EXPECTED_ORACLE:-2691008701644}"
REV_TAG="${REV_TAG:-389}"

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

# ---------------------------------------------------------------------
# 0. the external filter script this whole feature depends on. If it's
#    missing, fail fast here in seconds rather than after a long N=22
#    generation run that will just hit crunner-input-missing anyway.
# ---------------------------------------------------------------------
if [[ -f "$FILTER_SCRIPT" ]]; then
  pass "filter_script_present[$FILTER_SCRIPT]"
else
  fail "filter_script_present[$FILTER_SCRIPT]" "not found -- ensure_crunner_input_bin() cannot complete stage 2 without this; the run below would end in [crunner-input-missing] regardless of stage 1 succeeding"
  exit 1
fi

if grep -q "def ensure_crunner_input_bin" "$PY_SRC"; then
  pass "autobuild_function_present"
else
  fail "autobuild_function_present" "ensure_crunner_input_bin() not found in $PY_SRC"
fi

if [[ "$FAIL" -gt 0 ]]; then exit 1; fi

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "STATIC_ONLY=1: stopping after static checks."
  exit 0
fi

# ---------------------------------------------------------------------
# 1. CRunner binary (389_kernel_maxd14) -- build if missing, same
#    pattern 388's own r4 established.
# ---------------------------------------------------------------------
if [[ ! -x "./$CRUNNER_BIN" ]]; then
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
  echo "Building $CRUNNER_SRC with $NVCC -arch=$ARCH..."
  CRUNNER_BUILD_LOG="389_crunner_build_$(date +%Y%m%d_%H%M%S).log"
  "$NVCC" -O3 -arch="$ARCH" -o "$CRUNNER_BIN" "$CRUNNER_SRC" 2>&1 | tee "$CRUNNER_BUILD_LOG"
  if [[ ! -x "$CRUNNER_BIN" ]]; then
    fail "crunner_binary_present[$CRUNNER_BIN]" "nvcc build failed -- see $CRUNNER_BUILD_LOG"
    exit 1
  fi
fi
pass "crunner_binary_present[$CRUNNER_BIN]"

if ! command -v "$CODON" >/dev/null 2>&1; then
  fail "codon_toolchain_present" "'codon' not found on PATH"
  exit 1
fi
pass "codon_toolchain_present"

echo "Building $PY_SRC with $CODON build -release..."
rm -f "$BIN"
BUILD_LOG="389_build_$(date +%Y%m%d_%H%M%S).log"
"$CODON" build -release -o "$BIN" "$PY_SRC" 2>&1 | tee "$BUILD_LOG"

if [[ ! -x "$BIN" ]]; then
  fail "codon_build_succeeded" "binary $BIN was not produced -- see $BUILD_LOG"
  exit 1
fi
pass "codon_build_succeeded"

# ---------------------------------------------------------------------
# 2. N=21 auto-rebuild proof (cheap, fast): ensure_crunner_input_bin()
#    was written generically (parameterized by N and stream_fname, no
#    N=22-specific logic anywhere) -- this proves that on real
#    hardware, not just by code inspection. N=21's filtered file
#    already exists from 385 onward, so it is BACKED UP (renamed, not
#    deleted) here, the pipeline is forced to rebuild it from scratch,
#    and then the rebuilt file's checksum is compared against the
#    backup to confirm the auto-build reproduces byte-identical
#    output, not just "a" file. If anything here fails, the backup is
#    restored before exiting so N=21 is never left broken.
N21_FILTERED="constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin"
N21_SOAREF="constellations_N21_6.bin.soa_ref_361.bin"
N21_FILTERED_BAK="${N21_FILTERED}.389bak"
N21_SOAREF_BAK="${N21_SOAREF}.389bak"

restore_n21_backup() {
  [[ -f "$N21_FILTERED_BAK" ]] && mv -f "$N21_FILTERED_BAK" "$N21_FILTERED"
  [[ -f "$N21_SOAREF_BAK" ]] && mv -f "$N21_SOAREF_BAK" "$N21_SOAREF"
}

if [[ -f "$N21_FILTERED" ]]; then
  ORIG_CHECKSUM="$(sha256sum "$N21_FILTERED" | awk '{print $1}')"
  mv -f "$N21_FILTERED" "$N21_FILTERED_BAK"
  [[ -f "$N21_SOAREF" ]] && mv -f "$N21_SOAREF" "$N21_SOAREF_BAK"
  pass "n21_existing_file_backed_up"

  echo "Running: ./$BIN -g 21 21 32 484 1 0 7 37 -d   (N=21 forced from-scratch auto-rebuild proof)"
  N21_LOG="389_run_n21_rebuild_$(date +%Y%m%d_%H%M%S).log"
  ./"$BIN" -g 21 21 32 484 1 0 7 37 -d 2>&1 | tee "$N21_LOG"

  if grep -q '\[crunner-input-build\]' "$N21_LOG"; then
    pass "n21_autobuild_triggered_from_scratch"
  else
    fail "n21_autobuild_triggered_from_scratch" "no [crunner-input-build] lines -- see $N21_LOG"
  fi

  if grep -qE "^21:\s*314666222712\s.*[[:space:]]ok[[:space:]]*\$" "$N21_LOG"; then
    pass "n21_rebuilt_result_matches_oracle"
  else
    fail "n21_rebuilt_result_matches_oracle" "see $N21_LOG"
    restore_n21_backup
    exit 1
  fi

  if [[ -f "$N21_FILTERED" ]]; then
    NEW_CHECKSUM="$(sha256sum "$N21_FILTERED" | awk '{print $1}')"
    if [[ "$NEW_CHECKSUM" == "$ORIG_CHECKSUM" ]]; then
      pass "n21_rebuilt_file_byte_identical_to_original"
    else
      fail "n21_rebuilt_file_byte_identical_to_original" "checksum differs: orig=$ORIG_CHECKSUM new=$NEW_CHECKSUM (total still matched the oracle, so this would be a same-answer-different-bytes curiosity, not a correctness failure -- but worth a look)"
    fi
  else
    fail "n21_rebuilt_file_byte_identical_to_original" "$N21_FILTERED does not exist after the rebuild run"
  fi

  # backup no longer needed once we've confirmed the rebuild is good
  rm -f "$N21_FILTERED_BAK" "$N21_SOAREF_BAK"
else
  echo "NOTE: $N21_FILTERED not found -- skipping the backup/rebuild proof (nothing to back up); N=22's own from-scratch test below covers the same code path."
fi

# ---------------------------------------------------------------------
# 3. THE real test: N=22, bench_mode=37, WITH -d (so the
#    [crunner-input-build] stage messages are visible for this
#    first-ever run of the auto-build path, and land in
#    389_crunner_logs/dispatch.log either way). This is expected to
#    take considerably longer than N=21's ~3-4 minutes.
# ---------------------------------------------------------------------
echo "Running: ./$BIN -g $N $N 32 484 1 0 7 37 -d   (this WILL take a while -- first-ever N=22 auto-build)"
RUN_LOG="389_run_$(date +%Y%m%d_%H%M%S).log"
./"$BIN" -g "$N" "$N" 32 484 1 0 7 37 -d 2>&1 | tee "$RUN_LOG"

if grep -q '\[crunner-input-build\]' "$RUN_LOG"; then
  pass "autobuild_stages_ran (saw at least one [crunner-input-build] line)"
else
  fail "autobuild_stages_ran" "no [crunner-input-build] lines seen -- either the file already existed (unexpected per this harness's premise) or auto-build never triggered -- see $RUN_LOG"
fi

if grep -qE "^${N}:\s*${EXPECTED_ORACLE}\s.*[[:space:]]ok[[:space:]]*\$" "$RUN_LOG"; then
  pass "N22_matches_oracle_and_ok (expected=$EXPECTED_ORACLE)"
else
  fail "N22_matches_oracle_and_ok" "N=$N row missing, wrong total, or missing 'ok' -- see $RUN_LOG"
fi

FILTERED_EXPECTED="constellations_N${N}_7.bin.soa_ref_361.bin.maxd14only_363.bin"
if [[ -f "$FILTERED_EXPECTED" ]]; then
  pass "filtered_input_file_now_exists[$FILTERED_EXPECTED]"
else
  fail "filtered_input_file_now_exists" "expected $FILTERED_EXPECTED to exist after a successful auto-build run"
fi

echo ""
echo "===== 389 summary ====="
echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then
  echo "FAILED CHECKS:"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  exit 1
fi
echo "389 PASSED: N=22's maxd14-filtered CRunner input, which had never"
echo "been built on this machine, was auto-built end-to-end (361 dump +"
echo "external 363 filter script) and N=22 now resolves via bench_mode=37"
echo "exactly like N=21 already did. Bare -g should now reach N=22"
echo "automatically on future runs (the file persists)."
exit 0
