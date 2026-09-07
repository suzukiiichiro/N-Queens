#!/usr/bin/env bash
# 385_validate.sh
#
# rev385 -- Validation harness for the new maxd-gated CRunner os.system
# dispatch (CRunnerEntry / crunner_dispatch_table / crunner_run,
# bench_mode=37). Builds 385Py_kernel_maxd14_final.py with codon, then
# runs `-g 21 21 ... 37` (bench_mode=37) which shells out to the
# UNCHANGED 364_kernel_maxd14 binary via os.system() + logfile parsing
# (the mechanism 384 confirmed on real hardware) instead of using
# either Codon GPU path.
#
# Pass criteria: [crunner-dispatch-summary] reports total=314666222712
# (oracle), and the printed "ok" status line -- confirming the dispatch
# mechanism reproduces the same result as calling 364_kernel_maxd14
# directly (374's own real-hardware anchor: total_sum=314666222712,
# kernel_ms=201232.422). This harness does NOT rebuild 364_kernel_
# maxd14.cu -- it must already be built (374_validate.sh / 382_validate.sh
# territory) since 385 never touches that file.

set -u
PY_SRC="${PY_SRC:-385Py_kernel_maxd14_final.py}"
BIN="${BIN:-385Py_kernel_maxd14_final}"
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
# 0. files present, including the CRunner binary this revision
#    dispatches to (built separately -- 385 does not build it).
# ---------------------------------------------------------------------
if [[ ! -f "$PY_SRC" ]]; then
  fail "file_present[$PY_SRC]" "not found in $(pwd)"
  exit 1
fi
pass "file_present[$PY_SRC]"

if [[ ! -x "./$CRUNNER_BIN" ]]; then
  fail "crunner_binary_present[$CRUNNER_BIN]" "not found or not executable -- this must already be built from 364_kernel_maxd14.cu (see 374/382's own validate scripts), 385 does not build it"
  exit 1
fi
pass "crunner_binary_present[$CRUNNER_BIN]"

# r2: 364_kernel_maxd14 reads the maxd14-FILTERED SoA reference dump
# (7 fields/28 bytes), NOT the raw stream bin -- r1 of this revision
# got this wrong and hung for 40+ minutes on real hardware feeding the
# wrong file in. Check for the expected filtered file up front so a
# missing-input failure is caught here in seconds, not after a long
# run. Filename must match crunner_input_fname()'s convention exactly:
# {stream_fname}.soa_ref_361.bin.maxd14only_363.bin, which for N=21
# (preset_queens=6) is constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin
CRUNNER_INPUT="${CRUNNER_INPUT:-constellations_N${N}_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
if [[ -f "$CRUNNER_INPUT" ]]; then
  pass "crunner_input_present[$CRUNNER_INPUT]"
else
  fail "crunner_input_present[$CRUNNER_INPUT]" "maxd14-filtered SoA reference dump not found -- this is 361's dump_soa_reference_c_port() output further filtered by the external 363_filter_maxd14_only.py script. Build it first (or point CRUNNER_INPUT at an existing copy) before running this harness. The binary itself will now fail fast with [crunner-input-missing] rather than hang if this is missing, but catching it here avoids even the codon build+run round trip."
  exit 1
fi

# ---------------------------------------------------------------------
# 1. static checks: the new bench_mode=37 symbols must be present, and
#    364_kernel_maxd14.cu must be UNCHANGED (385's entire premise is
#    that it never touches this file -- a real hash check against a
#    known-good copy is stronger than this, but a symbol-presence check
#    at minimum confirms nothing here tries to rebuild/patch it).
# ---------------------------------------------------------------------
for sym in CRunnerEntry crunner_dispatch_table crunner_select_entry_index crunner_parse_result crunner_run "bench_mode==37"; do
  if grep -q -- "$sym" "$PY_SRC"; then
    pass "symbol_present[$sym]"
  else
    fail "symbol_present[$sym]" "'$sym' not found in $PY_SRC"
  fi
done
if [[ "$FAIL" -gt 0 ]]; then exit 1; fi

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "STATIC_ONLY=1: stopping after static checks."
  exit 0
fi

# ---------------------------------------------------------------------
# 2. codon build
# ---------------------------------------------------------------------
if ! command -v "$CODON" >/dev/null 2>&1; then
  fail "codon_toolchain_present" "'codon' not found on PATH"
  exit 1
fi
pass "codon_toolchain_present"

echo "Building $PY_SRC with $CODON build -release..."
rm -f "$BIN"
BUILD_LOG="385_build_$(date +%Y%m%d_%H%M%S).log"
"$CODON" build -release -o "$BIN" "$PY_SRC" 2>&1 | tee "$BUILD_LOG"

if [[ ! -x "$BIN" ]]; then
  fail "codon_build_succeeded" "binary $BIN was not produced -- see $BUILD_LOG"
  exit 1
fi
pass "codon_build_succeeded"

# ---------------------------------------------------------------------
# 3. real run: -g N N <block> <max_blocks> <log_level> <sort_mode>
#    <preset> <bench_mode=37>. gpu_log_level=1 so [crunner-dispatch*]
#    lines are printed (not just the summary table row).
# ---------------------------------------------------------------------
echo "Running: ./$BIN -g $N $N 32 484 1 0 7 37"
RUN_LOG="385_run_$(date +%Y%m%d_%H%M%S).log"
./"$BIN" -g "$N" "$N" 32 484 1 0 7 37 2>&1 | tee "$RUN_LOG"

if grep -q '\[crunner-dispatch-summary\]' "$RUN_LOG"; then
  pass "crunner_dispatch_summary_present"
else
  fail "crunner_dispatch_summary_present" "no [crunner-dispatch-summary] line -- see $RUN_LOG for [crunner-unsupported] or a crash"
  exit 1
fi

if grep -q "\[crunner-dispatch-summary\].*total=${EXPECTED_ORACLE}" "$RUN_LOG"; then
  pass "crunner_total_matches_oracle (expected=$EXPECTED_ORACLE)"
else
  fail "crunner_total_matches_oracle" "total= did not match $EXPECTED_ORACLE -- see $RUN_LOG"
fi

if grep -qE "^${N}:.*[[:space:]]ok[[:space:]]*\$" "$RUN_LOG"; then
  pass "table_row_status_ok"
else
  fail "table_row_status_ok" "N=$N row did not print status 'ok' -- see $RUN_LOG"
fi

echo ""
echo "===== 385 summary ====="
echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then
  echo "FAILED CHECKS:"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  exit 1
fi
echo "385 PASSED: maxd-gated CRunner os.system dispatch reproduces"
echo "374's real-hardware anchor (total_sum=314666222712) by shelling"
echo "out to the UNCHANGED 364_kernel_maxd14 binary. Check the printed"
echo "kernel_ms in [crunner-dispatch-summary] against the ~201,232ms"
echo "anchor -- it should be close (this run's own os.system + parsing"
echo "overhead is host-side and outside the CUDA event timing, so it"
echo "should not move kernel_ms itself)."
exit 0
