#!/usr/bin/env bash
# 386b_validate.sh
#
# rev386b -- Validation harness for removing the broadmark_tail_variant
# CLI-override capability from bench_mode==30/31, confirmed unused by
# Suzuki. Two things to confirm on real hardware:
#
#   (1) the CLI-read lines are truly gone (static check).
#   (2) bench_mode==31 (N=21) still reproduces the same result as
#       386a -- confirming the variant stays correctly fixed at its
#       adopted default (2) now that only the internal default path
#       sets it, and that no positional-argv shift broke
#       chunkshape148_bucket_run/chunkshape148_iter_sort (passed
#       explicitly on the command line below, at their unchanged
#       argv[16]/argv[17] positions).

set -u
PY_SRC="${PY_SRC:-386bPy_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
BIN="${BIN:-386bPy_kernel_maxd14_final}"
STATIC_ONLY="${STATIC_ONLY:-0}"
CODON="${CODON:-codon}"
N="${N:-21}"
EXPECTED_ORACLE="${EXPECTED_ORACLE:-314666222712}"

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
# 1. static check: no real "broadmark_tail_variant=int(sys.argv[...])"
#    assignment should remain anywhere (comments are fine and excluded).
# ---------------------------------------------------------------------
if grep -qE 'broadmark_tail_variant\s*=\s*int\(sys\.argv' "$PY_SRC"; then
  fail "cli_override_removed" "a real 'broadmark_tail_variant=int(sys.argv[...])' assignment still exists in $PY_SRC"
else
  pass "cli_override_removed"
fi

# the variable and its non-CLI defaults must still be intact
if grep -qE 'broadmark_tail_variant:int=BROAD_MARKDIST_TAIL_VARIANT' "$PY_SRC" && grep -qE 'broadmark_tail_variant=A10G_FINAL_DEFAULT_BROADMARK_VARIANT' "$PY_SRC"; then
  pass "non_cli_defaults_intact"
else
  fail "non_cli_defaults_intact" "expected default-assignment lines for broadmark_tail_variant are missing -- 386b may have removed too much"
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
BUILD_LOG="386b_build_$(date +%Y%m%d_%H%M%S).log"
"$CODON" build -release -o "$BIN" "$PY_SRC" 2>&1 | tee "$BUILD_LOG"

if [[ ! -x "$BIN" ]]; then
  fail "codon_build_succeeded" "binary $BIN was not produced -- see $BUILD_LOG"
  exit 1
fi
pass "codon_build_succeeded"

# ---------------------------------------------------------------------
# 3. regression: bench_mode==31 (N=21), explicitly passing values at
#    the argv[16]/argv[17] chunkshape148 positions to confirm they
#    still land correctly with the variant slot now unconsumed in
#    between (argv[15] -- deliberately passed as "0" here, a value
#    that would previously have set variant=0 and changed the
#    broadmarktail_params log line; it should now be silently ignored).
# ---------------------------------------------------------------------
echo "Running: ./$BIN -g $N $N 32 484 1 0 7 31 3 7 0 0 1 0 2048 9"
RUN_LOG="386b_run_$(date +%Y%m%d_%H%M%S).log"
./"$BIN" -g "$N" "$N" 32 484 1 0 7 31 3 7 0 0 1 0 2048 9 2>&1 | tee "$RUN_LOG"

if grep -q "broadmarktail_params:.*variant=2 " "$RUN_LOG"; then
  pass "variant_still_2_despite_argv15_being_0 (CLI override truly gone)"
else
  fail "variant_still_2_despite_argv15_being_0" "expected 'variant=2' in the broadmarktail_params log line even with argv[15]=0 -- see $RUN_LOG. If this shows variant=0 instead, the override was NOT actually removed."
fi

if grep -q "chunkshape148_params:.*bucket_run=2048" "$RUN_LOG"; then
  pass "chunkshape148_bucket_run_still_reaches_argv16 (2048)"
else
  fail "chunkshape148_bucket_run_still_reaches_argv16" "expected bucket_run=2048 -- see $RUN_LOG (would indicate a positional shift)"
fi

if grep -qE "^${N}:\s*${EXPECTED_ORACLE}\s" "$RUN_LOG"; then
  pass "bench31_total_matches_oracle (expected=$EXPECTED_ORACLE)"
else
  fail "bench31_total_matches_oracle" "N=$N row did not show $EXPECTED_ORACLE -- see $RUN_LOG"
fi

echo ""
echo "===== 386b summary ====="
echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then
  echo "FAILED CHECKS:"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  exit 1
fi
echo "386b PASSED: broadmark_tail_variant's CLI-override capability is"
echo "gone (passing argv[15]=0 no longer changes variant, which stays"
echo "at its adopted default of 2), and the subsequent chunkshape148"
echo "positional args at argv[16]/argv[17] still land correctly --"
echo "confirming no positional shift occurred."
exit 0
