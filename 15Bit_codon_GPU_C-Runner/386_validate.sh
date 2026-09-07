#!/usr/bin/env bash
# 386_validate.sh
#
# rev386 -- Validation harness for the validation-helper file split
# (12 pure verification/diagnostic functions relocated from
# 385Py_kernel_maxd14_final.py into rev386_validation_helpers.py, no
# logic changes). Two things to confirm on real hardware:
#
#   (1) codon build -release still succeeds with the local import in
#       play (386's version of 384's Q1, now against the real
#       production file instead of a toy probe).
#   (2) the SAME bench_mode=37 N=21 CRunner-dispatch check 385 already
#       passed still passes byte-for-byte the same way -- i.e. the
#       relocation did not change behavior. This is NOT a new feature
#       test; it's a regression check that the split was truly inert.

set -u
PY_SRC="${PY_SRC:-386Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
BIN="${BIN:-386Py_kernel_maxd14_final}"
CRUNNER_BIN="${CRUNNER_BIN:-364_kernel_maxd14}"
STATIC_ONLY="${STATIC_ONLY:-0}"
CODON="${CODON:-codon}"
N="${N:-21}"
EXPECTED_ORACLE="${EXPECTED_ORACLE:-314666222712}"

PASS=0
FAIL=0
declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }

# ---------------------------------------------------------------------
# 0. files present, including the CRunner binary and its filtered
#    input (same real-hardware-hazard lesson as 385: check this BEFORE
#    a long build+run, not after).
# ---------------------------------------------------------------------
for f in "$PY_SRC" "$HELPER_SRC"; do
  if [[ ! -f "$f" ]]; then
    fail "file_present[$f]" "not found in $(pwd)"
    exit 1
  fi
  pass "file_present[$f]"
done

if [[ ! -x "./$CRUNNER_BIN" ]]; then
  fail "crunner_binary_present[$CRUNNER_BIN]" "not found or not executable"
  exit 1
fi
pass "crunner_binary_present[$CRUNNER_BIN]"

CRUNNER_INPUT="${CRUNNER_INPUT:-constellations_N${N}_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
if [[ -f "$CRUNNER_INPUT" ]]; then
  pass "crunner_input_present[$CRUNNER_INPUT]"
else
  fail "crunner_input_present[$CRUNNER_INPUT]" "maxd14-filtered SoA reference dump not found -- see 385_README_append.md for what builds this"
  exit 1
fi

# ---------------------------------------------------------------------
# 1. static checks: relocated functions must be GONE from the main
#    file (only a [rev386] pointer comment should remain) and PRESENT
#    in the helper file, and the import statement must be there.
# ---------------------------------------------------------------------
for sym in validate_chunk_range validate_reordered_count validate_reordered_indices file_exists validate_bin_file count_constellations_bin_records read_stream_done_count write_stream_done_count read_vmhwm_kb crunner_parse_result crunner_input_fname crunner_input_valid; do
  if grep -q "^def ${sym}(" "$PY_SRC"; then
    fail "relocated[$sym]" "still defined in $PY_SRC -- split did not remove it"
  else
    pass "relocated[$sym] (no longer defined in $PY_SRC)"
  fi
  if grep -q "^def ${sym}(" "$HELPER_SRC"; then
    pass "present_in_helper[$sym]"
  else
    fail "present_in_helper[$sym]" "not found in $HELPER_SRC"
  fi
done

if grep -q "^import rev386_validation_helpers" "$PY_SRC"; then
  pass "import_statement_present"
else
  fail "import_statement_present" "no 'import rev386_validation_helpers' line in $PY_SRC"
fi

if [[ "$FAIL" -gt 0 ]]; then exit 1; fi

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "STATIC_ONLY=1: stopping after static checks."
  exit 0
fi

# ---------------------------------------------------------------------
# 2. codon build (question 1: local import still survives with the
#    real production file, not just 384's toy probe)
# ---------------------------------------------------------------------
if ! command -v "$CODON" >/dev/null 2>&1; then
  fail "codon_toolchain_present" "'codon' not found on PATH"
  exit 1
fi
pass "codon_toolchain_present"

echo "Building $PY_SRC (imports $HELPER_SRC) with $CODON build -release..."
rm -f "$BIN"
BUILD_LOG="386_build_$(date +%Y%m%d_%H%M%S).log"
"$CODON" build -release -o "$BIN" "$PY_SRC" 2>&1 | tee "$BUILD_LOG"

if [[ ! -x "$BIN" ]]; then
  fail "codon_build_succeeded" "binary $BIN was not produced -- see $BUILD_LOG"
  exit 1
fi
pass "codon_build_succeeded"

# ---------------------------------------------------------------------
# 3. regression check: same bench_mode=37 N=21 run 385 already passed.
#    If the split changed behavior, this is where it would show up.
# ---------------------------------------------------------------------
echo "Running: ./$BIN -g $N $N 32 484 1 0 7 37"
RUN_LOG="386_run_$(date +%Y%m%d_%H%M%S).log"
./"$BIN" -g "$N" "$N" 32 484 1 0 7 37 2>&1 | tee "$RUN_LOG"

if grep -q "\[crunner-dispatch-summary\].*total=${EXPECTED_ORACLE}" "$RUN_LOG"; then
  pass "crunner_total_matches_oracle (expected=$EXPECTED_ORACLE, unchanged from 385)"
else
  fail "crunner_total_matches_oracle" "total= did not match $EXPECTED_ORACLE -- see $RUN_LOG (this would mean the split changed behavior)"
fi

if grep -qE "^${N}:.*[[:space:]]ok[[:space:]]*\$" "$RUN_LOG"; then
  pass "table_row_status_ok"
else
  fail "table_row_status_ok" "N=$N row did not print status 'ok' -- see $RUN_LOG"
fi

echo ""
echo "===== 386 summary ====="
echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then
  echo "FAILED CHECKS:"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  exit 1
fi
echo "386 PASSED: the 12 relocated functions build and run correctly"
echo "from rev386_validation_helpers.py, and the bench_mode=37 N=21"
echo "CRunner dispatch check reproduces 385's exact result -- the split"
echo "was behaviorally inert."
exit 0
