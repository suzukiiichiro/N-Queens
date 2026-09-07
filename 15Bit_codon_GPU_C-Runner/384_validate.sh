#!/usr/bin/env bash
# 384_validate.sh
#
# rev384 -- Build+run harness for the multi-file-import + os.system
# feasibility probe. This is NOT a kernel/GPU validation script (no
# nvcc, no ncu, no oracle) -- it answers exactly two yes/no questions
# on real hardware before 383's real implementation is attempted:
#
#   Q1: does `import rev384_helper_probe` (a local .py file in the
#       same directory) survive `codon build -release`?
#   Q2: does os.system(cmd) + reading the resulting log file back
#       with Codon's native file I/O work as expected?
#
# See 384_README_append.md for the pre-registered predictions this
# harness is checking against.

set -u
MAIN_SRC="${MAIN_SRC:-384Py_multifile_osexec_probe.py}"
HELPER_SRC="${HELPER_SRC:-rev384_helper_probe.py}"
BIN="${BIN:-384_probe}"
CODON="${CODON:-codon}"

PASS=0
FAIL=0
declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }

# ---------------------------------------------------------------------
# 0. files present
# ---------------------------------------------------------------------
for f in "$MAIN_SRC" "$HELPER_SRC"; do
  if [[ ! -f "$f" ]]; then
    fail "file_present[$f]" "not found in $(pwd)"
    exit 1
  fi
  pass "file_present[$f]"
done

if ! command -v "$CODON" >/dev/null 2>&1; then
  fail "codon_toolchain_present" "'codon' not found on PATH"
  exit 1
fi
pass "codon_toolchain_present"

# ---------------------------------------------------------------------
# 1. codon build -release, with the local import in play. This is
#    Q1's answer: if this produces a binary, the import survived.
# ---------------------------------------------------------------------
echo "Building $MAIN_SRC (imports $HELPER_SRC) with $CODON build -release..."
rm -f "$BIN"
BUILD_LOG="384_build_$(date +%Y%m%d_%H%M%S).log"
"$CODON" build -release -o "$BIN" "$MAIN_SRC" 2>&1 | tee "$BUILD_LOG"

if [[ ! -x "$BIN" ]]; then
  fail "codon_build_succeeded" "binary $BIN was not produced -- this IS question (1)'s answer: local multi-file import did NOT survive codon build -release as written. See $BUILD_LOG for the exact error."
  exit 1
fi
pass "codon_build_succeeded (Q1: local multi-file import SURVIVED codon build -release)"

# ---------------------------------------------------------------------
# 2. run it and check the marker lines it prints (Q1 exercised for
#    real by calling the imported function; Q2 exercised by the
#    os.system + logfile round trip).
# ---------------------------------------------------------------------
RUN_LOG="384_run_$(date +%Y%m%d_%H%M%S).log"
rm -rf 384_probe_logs
./"$BIN" 2>&1 | tee "$RUN_LOG"

if grep -q '\[384-import-probe\] PASS' "$RUN_LOG"; then
  pass "import_probe (Q1: imported function actually callable, returns real data)"
else
  fail "import_probe" "no PASS line -- see $RUN_LOG"
fi

if grep -q '\[384-osexec-probe\] PASS' "$RUN_LOG"; then
  pass "osexec_probe (Q2: os.system + native-file-I/O logfile round-trip)"
else
  fail "osexec_probe" "no PASS line -- see $RUN_LOG"
fi

if grep -q '\[384-probe-summary\].*overall=PASS' "$RUN_LOG"; then
  pass "overall_summary_pass"
else
  fail "overall_summary_pass" "see $RUN_LOG"
fi

echo ""
echo "===== 384 summary ====="
echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then
  echo "FAILED CHECKS:"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  exit 1
fi
echo "384 PASSED: both open questions answered YES on real hardware."
echo "-> 383's real implementation (maxd-gated os.system dispatch to"
echo "   CRunner binaries, plus splitting the validation-helper"
echo "   functions into their own file) can proceed on this confirmed"
echo "   foundation. If either question had failed, 383's design would"
echo "   need to fall back to a single-file layout and/or a different"
echo "   process-launch mechanism before any real code was written."
exit 0
