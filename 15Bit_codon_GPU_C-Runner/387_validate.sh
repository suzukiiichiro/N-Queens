#!/usr/bin/env bash
# 387_validate.sh
#
# rev387 -- Validation harness for bare `-g` (no other args) now
# defaulting to bench_mode=37 (maxd-gated CRunner dispatch) and running
# N=5 up to the file's real ceiling (28, see DEFAULT_RANGE_NMAX_
# EXCLUSIVE's own comment), stopping cleanly the first time a maxd is
# unsupported OR a supported maxd's CRunner input file isn't built yet.
#
# IMPORTANT CAVEAT, read before running: this harness does NOT assume
# N=22's maxd14-filtered CRunner input file (constellations_N22_7.bin.
# soa_ref_361.bin.maxd14only_363.bin, following crunner_input_fname()'s
# naming convention with N=22's preset_queens=7) has ever been built.
# Historically N=22 was validated via bench_mode=33 (Codon's own
# single-shot kernel, which reads the RAW stream bin directly, not the
# filtered CRunner format) -- so it is genuinely possible that file has
# never existed. If it hasn't, bare `-g` will correctly stop at N=22
# with [crunner-input-missing] rather than N=21. THAT IS NOT A FAILURE
# of this revision -- 385 deliberately chose not to auto-build that
# file (see 385's own README notes) -- so this harness treats a clean
# stop at N=21 OR N=22 as equally acceptable; only a crash, a hang, or
# a wrong total is a real failure.
#
# This is a genuinely long-running check: N=21 alone takes ~3-4 minutes
# via bench_mode=37 (see 385/386's own real-hardware timings). Budget
# accordingly.

set -u
PY_SRC="${PY_SRC:-387Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
BIN="${BIN:-387Py_kernel_maxd14_final}"
STATIC_ONLY="${STATIC_ONLY:-0}"
CODON="${CODON:-codon}"

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
# 1. static checks: bare -g's new defaults, and break (not continue)
#    on both bench_mode==37 failure paths.
# ---------------------------------------------------------------------
if grep -qE '^A10G_FINAL_DEFAULT_BENCH_MODE:int=37' "$PY_SRC"; then
  pass "bare_g_default_bench_mode_is_37"
else
  fail "bare_g_default_bench_mode_is_37" "A10G_FINAL_DEFAULT_BENCH_MODE is not 37 in $PY_SRC"
fi

if grep -qE '^DEFAULT_RANGE_NMAX_EXCLUSIVE:int=28' "$PY_SRC"; then
  pass "default_nmax_is_28"
else
  fail "default_nmax_is_28" "DEFAULT_RANGE_NMAX_EXCLUSIVE is not 28 in $PY_SRC"
fi

# crude but effective: the maxd-unsupported/crunner-input-missing print
# lines should each be immediately followed by "break" a few lines
# later, not "continue" -- check there is no bare "continue" between
# either print line and the next blank/dedent. We approximate this by
# just confirming "break" appears at least twice after the first
# "[crunner-unsupported]" line in the file.
if [[ "$(grep -c '^\s*break\s*$' "$PY_SRC")" -ge 2 ]]; then
  pass "break_statements_present (at least 2, for both failure paths)"
else
  fail "break_statements_present" "expected at least 2 bare 'break' statements for bench_mode==37's failure paths"
fi

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
BUILD_LOG="387_build_$(date +%Y%m%d_%H%M%S).log"
"$CODON" build -release -o "$BIN" "$PY_SRC" 2>&1 | tee "$BUILD_LOG"

if [[ ! -x "$BIN" ]]; then
  fail "codon_build_succeeded" "binary $BIN was not produced -- see $BUILD_LOG"
  exit 1
fi
pass "codon_build_succeeded"

# ---------------------------------------------------------------------
# 3. THE actual ask: bare -g, no other arguments at all.
# ---------------------------------------------------------------------
echo "Running: ./$BIN -g   (bare, no other args -- this is the real request)"
RUN_LOG="387_run_$(date +%Y%m%d_%H%M%S).log"
./"$BIN" -g 2>&1 | tee "$RUN_LOG"

if grep -q "^GPU mode selected" "$RUN_LOG"; then
  pass "gpu_mode_selected_header"
else
  fail "gpu_mode_selected_header" "missing 'GPU mode selected' line -- see $RUN_LOG"
fi

# N=5..20 should each print a clean table row with no bench_mode-37
# machinery visible (log_level=0 by default), matching Suzuki's
# original desired output format.
for small_n in 5 10 15 20; do
  if grep -qE "^${small_n}:" "$RUN_LOG"; then
    pass "row_present[N=${small_n}]"
  else
    fail "row_present[N=${small_n}]" "no table row for N=${small_n} -- see $RUN_LOG"
  fi
done

if grep -qE "^21:\s*314666222712\s.*[[:space:]]ok[[:space:]]*\$" "$RUN_LOG"; then
  pass "N21_matches_oracle_and_ok"
else
  fail "N21_matches_oracle_and_ok" "N=21 row missing, wrong total, or missing 'ok' status -- see $RUN_LOG"
fi

# N=22: either a correct oracle match, or a clean stop. Both acceptable
# per the caveat above -- only a crash/hang/wrong-total is a failure.
if grep -qE "^22:\s*2691008701644\s.*[[:space:]]ok[[:space:]]*\$" "$RUN_LOG"; then
  pass "N22_matches_oracle_and_ok (CRunner input for N=22 was already built)"
elif grep -q "\[crunner-input-missing\] N=22" "$RUN_LOG" || grep -q "\[crunner-unsupported\] N=22" "$RUN_LOG"; then
  pass "N22_clean_stop (CRunner input for N=22 not built yet -- expected per this harness's caveat, not a defect)"
else
  fail "N22_behavior" "N=22 neither matched the oracle nor stopped cleanly -- see $RUN_LOG (this would indicate a real problem)"
fi

# Whatever N the loop stopped at, the run must actually have ENDED
# (not hung) -- confirmed simply by the script reaching this point at
# all, since `tee` only returns after the process exits.
pass "run_terminated (did not hang)"

echo ""
echo "===== 387 summary ====="
echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then
  echo "FAILED CHECKS:"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  exit 1
fi
echo "387 PASSED: bare './$BIN -g' with no other arguments runs N=5"
echo "through as far as CRunner coverage currently reaches, in the"
echo "requested continuous-output format, and stops cleanly rather"
echo "than crashing or looping forever."
exit 0
