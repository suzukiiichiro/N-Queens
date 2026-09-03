#!/usr/bin/env bash
# 371_finegrain_determinism_sweep.sh
#
# rev371 — DESIGN-ONLY / HARNESS-ONLY. Zero code changes: reuses
# 370Py_mem_probe_v2.py verbatim (same binary, same bench_mode=36
# mem-probe mode). Nothing in 366/368/369/370 is touched. This
# revision's only content is a new validation-harness procedure, in
# the same spirit as 338/360/362's design-only revisions.
#
# Motivation: 370's sweep (record_limit ladder 1M/10M/11M/12M/13M/...)
# showed:
#   - delta_total_kb near-linear across 10M/11M/12M (~0.43 KB/record)
#   - a mid-run `ValueError: invalid format specifier` exception at
#     the 12M rung (right after mem-probe-done printed, while
#     formatting the next progress line) -- suggestive of a transient
#     allocation failure unrelated to the probe's own steady-state
#     memory use
#   - a clean, total failure to complete at 13M (no [mem-probe-done])
#   - dmesg confirmed a real segfault for both 369 and 370's runs, at
#     nearly identical instruction bytes, in what looks like an
#     indexed array-store instruction (`[reg + reg*4]` addressing)
# The near-linear delta_total_kb up to 12M does not, by itself,
# predict a hard failure at 13M -- extrapolating the 10M-12M slope to
# 13M would still leave several GB of headroom under the 10GB ceiling.
# This is consistent with a working hypothesis: the failure is not
# simply "steady-state memory exceeded 10GB" but a TRANSIENT spike --
# most likely a dynamic array/list reallocation (doubling growth,
# typical of unsized list construction) that briefly requests much
# more memory than the eventual steady state, and that transient
# request is what actually hits the ceiling.
#
# 371 does two things to test this, using ONLY the harness (no code
# change):
#   1. Bisects the 12,000,000/13,000,000 gap with a finer ladder
#      (12,200,000 / 12,500,000 / 12,800,000) to narrow the threshold.
#   2. Runs EVERY rung TWICE in a row (back-to-back) and compares
#      completion status between the two runs of the same
#      record_limit. If the failure is a reallocation-timing race (as
#      opposed to a deterministic, fixed threshold), repeated runs at
#      values near the boundary may disagree with each other --
#      itself diagnostic information, not a script bug.
#
# IMPORTANT: this harness does NOT verify markers or reconstruct a
# core hash against a prior revision the way 366/368/369/370's
# harnesses do, because there is no code delta to check -- it instead
# verifies the source file's raw content hash is byte-identical to
# 370's as-delivered file, which is a stronger and simpler guarantee
# for a zero-change revision.

set -u
SRC="${SRC:-370Py_mem_probe_v2.py}"
STATIC_ONLY="${STATIC_ONLY:-0}"
BIN="${BIN:-371_finegrain}"

PASS=0
FAIL=0
INFO=0
WARN=0
declare -a FAILED_CHECKS=()

pass()  { PASS=$((PASS+1));  echo "OK    $1"; }
fail()  { FAIL=$((FAIL+1));  FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info()  { INFO=$((INFO+1));  echo "INFO  $1: $2"; }
warn()  { WARN=$((WARN+1));  echo "WARN  $1: $2"; }

# ---------------------------------------------------------------------
# 0. sudo check FIRST (352 lesson). Non-fatal.
# ---------------------------------------------------------------------
if sudo -n true 2>/dev/null; then
  pass "sudo_permission_check"
else
  warn "sudo_permission_check" "sudo -n true failed (non-fatal; if a rung crashes, run 'sudo dmesg | tail -30' by hand afterward)"
fi

# ---------------------------------------------------------------------
# 1. Presence + raw identity check (zero-delta revision: 371 must use
#    370's source completely unmodified).
# ---------------------------------------------------------------------
if [[ ! -f "$SRC" ]]; then
  fail "source_py_present" "$SRC not found in $(pwd)"
  echo "Cannot continue without source file. Aborting."
  exit 1
fi
pass "source_py_present"

REF_HASH_370_RAW="8152a0ef550f1c6d0ab4b949d8ba25c973117e8fa99f9d1510abd87fbf68f13d"
ACTUAL_HASH=$(sha256sum "$SRC" | awk '{print $1}')
if [[ "$ACTUAL_HASH" == "$REF_HASH_370_RAW" ]]; then
  pass "source_identical_to_370_raw (hash=$ACTUAL_HASH)"
else
  fail "source_identical_to_370_raw" "expected raw hash=$REF_HASH_370_RAW, got $ACTUAL_HASH -- 371 must use 370's source byte-for-byte unmodified (this is a design-only/harness-only revision)"
fi

if grep -q 'VERSION_TAG:str="370' "$SRC"; then
  pass "source_version_tag_370_confirmed"
else
  fail "source_version_tag_370_confirmed" "VERSION_TAG for 370 not found in $SRC"
fi

echo ""
echo "===== 371 static-check summary ====="
echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
if [[ "$FAIL" -gt 0 ]]; then
  echo "FAILED CHECKS:"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
fi
echo "====================================="
echo ""

if [[ "$FAIL" -gt 0 ]]; then
  echo "Static checks failed. Not proceeding to build/run."
  exit 1
fi

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "STATIC_ONLY=1: stopping after static checks (dry run complete)."
  exit 0
fi

# ---------------------------------------------------------------------
# 2. Build once, then run the fine-grain x2-repeat sweep.
#
#    RECOMMENDED: run wrapped in 367_safe_run_wrapper.sh, same as
#    before.
# ---------------------------------------------------------------------
N="${N:-22}"
BLOCK="${BLOCK:-32}"
MAX_BLOCKS="${MAX_BLOCKS:-484}"
LOG_LEVEL="${LOG_LEVEL:-1}"
SORT_MODE="${SORT_MODE:-0}"
PRESET_QUEENS="${PRESET_QUEENS:-7}"
BENCH_MODE="${BENCH_MODE:-36}"
REORDER_WINDOW_MULT="${REORDER_WINDOW_MULT:-3}"
REORDER_PHASE_JUMP="${REORDER_PHASE_JUMP:-7}"
CROSS_STRIPE_SAFE="${CROSS_STRIPE_SAFE:-0}"
WORKER_COUNT="${WORKER_COUNT:-1}"
BROADMARK_VARIANT="${BROADMARK_VARIANT:-2}"
CHUNKSHAPE148_BUCKET_RUN="${CHUNKSHAPE148_BUCKET_RUN:-2048}"
CHUNKSHAPE148_ITER_SORT="${CHUNKSHAPE148_ITER_SORT:-9}"
# Fine-grain ladder bisecting 370's 12,000,000 (OK) / 13,000,000
# (FAIL) bracket. 12,000,000 and 13,000,000 themselves are re-included
# as anchors to re-confirm 370's result under the x2-repeat protocol.
RECORD_LIMITS="${RECORD_LIMITS:-12000000 12200000 12500000 12800000 13000000}"
REPEATS="${REPEATS:-2}"

echo "Building 371 (reusing 370's source, codon)..."
if ! command -v codon >/dev/null 2>&1; then
  fail "codon_toolchain_present" "codon not found on PATH"
  echo ""
  echo "===== updated summary ====="
  echo "OK=$PASS  FAIL=$((FAIL+1))  INFO=$INFO  WARN=$WARN"
  exit 1
fi

CAND="./${SRC%.py}"
rm -f "$CAND"
codon build -release "$SRC" 2>&1 | tee "${BIN}_build_$(date +%Y%m%d_%H%M%S).log"
if [[ ! -x "$CAND" ]]; then
  fail "py_build_succeeded" "binary $CAND was not produced"
  exit 1
fi
pass "py_build_succeeded"

echo ""
echo "===== 371 sweep: record_limit ladder = $RECORD_LIMITS, repeats=$REPEATS each ====="
echo ""

declare -a ROW_LIMIT=()
declare -a ROW_REP=()
declare -a ROW_STATUS=()
declare -a ROW_DELTA_TOTAL=()

for RL in $RECORD_LIMITS; do
  REP=1
  while [[ $REP -le $REPEATS ]]; do
    echo "--- rung: record_limit=$RL  rep=$REP/$REPEATS ---"
    CMD=("$CAND" -g "$N" "$N" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS" "$BENCH_MODE" "$REORDER_WINDOW_MULT" "$REORDER_PHASE_JUMP" "$CROSS_STRIPE_SAFE" "$RL" "$WORKER_COUNT" "$BROADMARK_VARIANT" "$CHUNKSHAPE148_BUCKET_RUN" "$CHUNKSHAPE148_ITER_SORT")
    echo "Running: ${CMD[*]}"
    PYLOG="${BIN}_N${N}_rl${RL}_rep${REP}_$(date +%Y%m%d_%H%M%S).log"
    stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "$PYLOG"

    DONE_LINE=$(grep '^\[mem-probe-done\]' "$PYLOG" | tail -n1)
    ROW_LIMIT+=("$RL")
    ROW_REP+=("$REP")
    if [[ -z "$DONE_LINE" ]]; then
      echo "!!! record_limit=$RL rep=$REP did NOT complete."
      ROW_STATUS+=("FAIL")
      ROW_DELTA_TOTAL+=("-")
    else
      DT=$(echo "$DONE_LINE" | grep -oE 'delta_total_kb=-?[0-9]+' | cut -d= -f2)
      ROW_STATUS+=("OK")
      ROW_DELTA_TOTAL+=("${DT:-?}")
    fi
    echo ""
    REP=$((REP+1))
  done
done

echo ""
echo "===== 371 sweep: per-run results ====="
printf "%-14s %-6s %-6s %-16s\n" "record_limit" "rep" "status" "delta_total_kb"
IDX=0
while [[ $IDX -lt ${#ROW_LIMIT[@]} ]]; do
  printf "%-14s %-6s %-6s %-16s\n" "${ROW_LIMIT[$IDX]}" "${ROW_REP[$IDX]}" "${ROW_STATUS[$IDX]}" "${ROW_DELTA_TOTAL[$IDX]}"
  IDX=$((IDX+1))
done
echo "======================================"
echo ""

echo "===== 371 sweep: per-value determinism check ====="
NONDETERMINISTIC=0
for RL in $RECORD_LIMITS; do
  STATUSES=""
  IDX=0
  while [[ $IDX -lt ${#ROW_LIMIT[@]} ]]; do
    if [[ "${ROW_LIMIT[$IDX]}" == "$RL" ]]; then
      STATUSES="$STATUSES ${ROW_STATUS[$IDX]}"
    fi
    IDX=$((IDX+1))
  done
  UNIQ=$(echo "$STATUSES" | tr ' ' '\n' | sort -u | tr '\n' ' ')
  UNIQ_COUNT=$(echo "$STATUSES" | tr ' ' '\n' | sort -u | grep -c .)
  if [[ "$UNIQ_COUNT" -gt 1 ]]; then
    echo "record_limit=$RL: NON-DETERMINISTIC (statuses:$STATUSES) -- same input, different outcomes across repeats. This itself supports a timing/allocation-race explanation over a fixed hard threshold."
    NONDETERMINISTIC=1
  else
    echo "record_limit=$RL: deterministic (all repeats: $UNIQ)"
  fi
done
echo "======================================"
echo ""

if [[ "$NONDETERMINISTIC" -eq 1 ]]; then
  info "nondeterminism_detected" "at least one record_limit value produced different outcomes across repeated runs -- report this explicitly, it changes the shape of the eventual fix (a timing/allocation-race issue is not fixed the same way as a fixed memory ceiling)"
else
  info "all_values_deterministic" "every tested record_limit gave the same outcome across all repeats"
fi

echo ""
echo "===== final summary ====="
echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
echo ""
echo "Report the per-run table and the determinism check above."
exit 0
