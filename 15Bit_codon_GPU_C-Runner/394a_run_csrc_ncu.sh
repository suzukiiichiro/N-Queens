#!/usr/bin/env bash
# 394a_run_csrc_ncu.sh
#
# rev394a -- ncu SourceCounters on the PRODUCTION CUDA C kernel.
#
# WHY THIS EXISTS
# ---------------
# 393 stage 3 established that the Codon and C kernels are structurally
# different code, not two builds of the same thing:
#
#     SASS instructions   Codon 664   ->   C 472   (-28.9%)
#     BRA                       68            37   (-45.6%)
#     LOP3                     150            83   (-44.7%)
#     stack push/pop     STL.64 x4 / LDL.64 x2  ->  STL.128 x2 / LDL.128 x1
#
# So 393 stage 1's per-instruction breakdown (long_sb 99.77% on one
# instruction, push/pop/reconverge = 47.15% of samples, pop path running
# at 2 lanes of 32) describes the Codon kernel. The production N=21 path
# is the C kernel. 394a takes the same measurement on the C side so the
# two are directly comparable, before 394b/394c change anything.
#
# Bonus that 393 could not have: the C source is built with -lineinfo,
# so ncu's source page attributes samples to actual .cu LINE NUMBERS.
# Codon can never do this -- its -debug emits invalid PTX (319-321).
#
# METHOD (375's, reused verbatim)
# -------------------------------
# The C runner's GPU path takes no record limit (its argv is
# <N> <in_soa7_bin> <out_results_bin> [expected_total]), so the input
# file itself is truncated instead: the first 15,488 records of the
# 374-confirmed filtered bin. 15,488 = BLOCK*MAX_BLOCKS = exactly one
# grid, so the grid-stride loop does not go round even once. N stays 21
# because the data is N=21 (board_mask/n3/n4 must be computed for 21).
#
# CORRECTNESS GATE
# ----------------
# A truncated run has no oracle. What it does have is an equality that
# must hold: the -lineinfo build and the plain build, given the same
# truncated input, must produce the IDENTICAL partial total. That is
# this revision's gate. If -lineinfo perturbed codegen enough to change
# a result, the whole measurement is void. The SASS instruction count is
# cross-checked against 393 stage 3's 472 as a second, non-gating signal.
#
# REPRESENTATIVENESS
# ------------------
# The filtered bin is in RAW order (393-10: the CRunner path never sees
# the chunkshape148 reordering), so "the first 15,488 records" is an
# arbitrary slice, not a designed one. 375 accepted this. 394a spends a
# further ~15 seconds taking a SECOND profile from a mid-file offset and
# comparing the two, so the caveat is measured instead of assumed.
#
# USAGE
#   STATIC_ONLY=1 bash 394a_run_csrc_ncu.sh     # checks + derive, no GPU
#                 bash 394a_run_csrc_ncu.sh     # full (~2-3 min)
#   OFFSET_STAGE=0 bash 394a_run_csrc_ncu.sh    # skip the 2nd slice
#
# sudo is checked FIRST, before any build (352's 14-minute lesson).

set -u

REV="394a"
SRC_CU="${SRC_CU:-389_kernel_maxd14.cu}"
REV_CU="${REV_CU:-394a_kernel_maxd14.cu}"
LINEINFO_BIN="${LINEINFO_BIN:-394a_kernel_maxd14_lineinfo}"
PLAIN_BIN="${PLAIN_BIN:-394a_kernel_maxd14}"
FULL_BIN="${FULL_BIN:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
RECORDS="${RECORDS:-15488}"        # BLOCK*MAX_BLOCKS -- exactly one grid
RECBYTES="${RECBYTES:-28}"         # SoA7: 7 x u32 LE
OFFSET_RECORDS="${OFFSET_RECORDS:-1000000}"   # 2nd slice start
OFFSET_STAGE="${OFFSET_STAGE:-1}"
STATIC_ONLY="${STATIC_ONLY:-0}"
STAGE_PAUSE="${STAGE_PAUSE:-5}"

# 6102's lesson: resolve ncu to an ABSOLUTE path and hand THAT to sudo.
NCU="${NCU:-$(command -v ncu 2>/dev/null)}"
[[ -z "$NCU" && -x /usr/local/cuda/bin/ncu ]] && NCU="/usr/local/cuda/bin/ncu"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_csrc_ncu_${TS}}"
SLICE_BIN="${REV}_input_head${RECORDS}.bin"
SLICE2_BIN="${REV}_input_off${OFFSET_RECORDS}_${RECORDS}.bin"
SLICE_BYTES=$(( RECORDS * RECBYTES ))
OFFSET_BYTES=$(( OFFSET_RECORDS * RECBYTES ))

# Number of lines in the header block this script prepends to $REV_CU.
# Used both to build the header and to verify the derivation, so the two
# can never drift apart.
HDR_LINES=15

PASS=0
FAIL=0
declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }

# ---------------------------------------------------------------------
# 0. sudo FIRST.
# ---------------------------------------------------------------------
if sudo -n true 2>/dev/null; then
  pass "sudo_noninteractive_available"
else
  fail "sudo_noninteractive_available" "'sudo -n true' failed -- ncu cannot read hardware counters without it. Stopping before any build."
  exit 1
fi

# ---------------------------------------------------------------------
# 1. Static checks
# ---------------------------------------------------------------------
for f in "$SRC_CU" "$FULL_BIN"; do
  if [[ -f "$f" ]]; then pass "file_present[$f]"
  else fail "file_present[$f]" "not found in $(pwd)"; fi
done

if [[ -n "$NCU" && -x "$NCU" ]]; then pass "ncu_present[$NCU]"
else fail "ncu_present" "ncu not found on PATH nor at /usr/local/cuda/bin/ncu"; fi

if [[ ! -x "$NVCC" ]] && ! command -v nvcc >/dev/null 2>&1; then
  fail "nvcc_present" "$NVCC not executable and 'nvcc' not on PATH"
else
  [[ ! -x "$NVCC" ]] && NVCC="nvcc"
  pass "nvcc_present[$NVCC]"
fi

if [[ -f "$FULL_BIN" ]]; then
  FULL_SIZE=$(stat -c%s "$FULL_BIN")
  if (( FULL_SIZE % RECBYTES == 0 )); then
    pass "input_record_aligned ($((FULL_SIZE / RECBYTES)) records of ${RECBYTES}B)"
  else
    fail "input_record_aligned" "$FULL_BIN is $FULL_SIZE bytes, not a multiple of $RECBYTES"
  fi
  if (( FULL_SIZE >= OFFSET_BYTES + SLICE_BYTES )); then
    pass "input_large_enough_for_both_slices"
  else
    info "input_large_enough_for_both_slices" "file too small for the offset slice -- 2nd stage will be skipped"
    OFFSET_STAGE=0
  fi
fi

if [[ "$FAIL" -gt 0 ]]; then
  echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  exit 1
fi

# ---------------------------------------------------------------------
# 2. Derive the rev-numbered .cu from 389's.
#
#    Per the standing per-revision-self-contained policy (388 r3/r4:
#    each revision references a file carrying its OWN number, never a
#    past one), 394a gets its own .cu. It is 389's file with a header
#    comment block PREPENDED and nothing else touched -- prepending
#    rather than editing in place is what makes the derivation exactly
#    verifiable: dropping the first $HDR_LINES lines must reproduce
#    389_kernel_maxd14.cu byte for byte.
#
#    NOTE ON LINE NUMBERS: -lineinfo will attribute to THIS file's line
#    numbers. To map a reported line back to 389_kernel_maxd14.cu,
#    subtract $HDR_LINES. The offset is printed below and recorded in
#    the log directory.
# ---------------------------------------------------------------------
banner "Deriving $REV_CU from $SRC_CU (header prepend only)"
{
  echo "/* ==================================================================="
  echo " * ${REV}_kernel_maxd14.cu"
  echo " *"
  echo " * Derived mechanically from ${SRC_CU} by 394a_run_csrc_ncu.sh:"
  echo " * this ${HDR_LINES}-line comment block is PREPENDED and nothing else is"
  echo " * changed. Dropping the first ${HDR_LINES} lines reproduces ${SRC_CU}"
  echo " * byte for byte (checked below, gating)."
  echo " *"
  echo " * Purpose: a -lineinfo diagnostic build for ncu SourceCounters."
  echo " * This is NOT a production binary -- crunner_dispatch_table() in"
  echo " * 394aPy deliberately still points at ./389_kernel_maxd14, the same"
  echo " * separation 320 kept between its debug build and its timing build."
  echo " *"
  echo " * LINE NUMBER OFFSET vs ${SRC_CU}: +${HDR_LINES}"
  echo " * =================================================================== */"
  cat "$SRC_CU"
} > "$REV_CU"

ACTUAL_HDR=$(( $(wc -l < "$REV_CU") - $(wc -l < "$SRC_CU") ))
if [[ "$ACTUAL_HDR" -eq "$HDR_LINES" ]]; then
  pass "derived_header_length_is_${HDR_LINES}"
else
  fail "derived_header_length_is_${HDR_LINES}" "prepended $ACTUAL_HDR lines, expected $HDR_LINES -- the header text and HDR_LINES have drifted apart"
fi

SRC_SHA=$(sha256sum "$SRC_CU" | cut -d' ' -f1)
DER_SHA=$(tail -n +$((HDR_LINES + 1)) "$REV_CU" | sha256sum | cut -d' ' -f1)
if [[ "$SRC_SHA" == "$DER_SHA" ]]; then
  pass "derived_code_region_identical_to_389 (sha256 ${SRC_SHA:0:16}...)"
else
  fail "derived_code_region_identical_to_389" "sha256 mismatch: 389=${SRC_SHA:0:16}... derived=${DER_SHA:0:16}..."
fi

if [[ "$FAIL" -gt 0 ]]; then
  echo; echo "===== ${REV} derivation summary ====="; echo "OK=$PASS FAIL=$FAIL"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  exit 1
fi

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo; echo "===== ${REV} STATIC_ONLY summary ====="
  echo "OK=$PASS  FAIL=$FAIL"
  echo "$REV_CU derived and verified. Re-run without STATIC_ONLY=1 to build and profile."
  exit 0
fi

mkdir -p "$LOGDIR"
pass "logdir_created[$LOGDIR]"
echo "line-number offset from $REV_CU to $SRC_CU: -$HDR_LINES" > "$LOGDIR/00_lineinfo_offset.txt"
cp "$REV_CU" "$LOGDIR/" 2>/dev/null || true

{
  echo "=== $REV environment capture (pre) $(date -Is) ==="
  uname -a
  nvidia-smi 2>&1
  nvidia-smi -q -d CLOCK 2>&1
  "$NCU" --version 2>&1
  "$NVCC" --version 2>&1
  sha256sum "$SRC_CU" "$REV_CU" "$FULL_BIN" 2>&1
} > "$LOGDIR/01_env_pre.txt" 2>&1
SM_CLOCK="$(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader 2>/dev/null || echo unavailable)"
info "sm_clock" "$SM_CLOCK  (393 measured at 1710MHz; a different value invalidates cross-revision comparison)"

MARKER="$LOGDIR/.owner_marker"; touch "$MARKER"
reclaim_ownership() {
  find . -maxdepth 1 -newer "$MARKER" -user root -print0 2>/dev/null \
    | xargs -0 -r sudo chown "$(id -u):$(id -g)" 2>/dev/null || true
  sudo chown -R "$(id -u):$(id -g)" "$LOGDIR" 2>/dev/null || true
}

# ---------------------------------------------------------------------
# 3. Two builds: -lineinfo (for ncu) and plain (as the codegen control).
# ---------------------------------------------------------------------
banner "Building $REV_CU twice: -lineinfo and plain"
rm -f "$LINEINFO_BIN" "$PLAIN_BIN"
"$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$LINEINFO_BIN" "$REV_CU" 2>&1 | tee "$LOGDIR/02_build_lineinfo.log"
"$NVCC" -O3 -arch="$ARCH"            -o "$PLAIN_BIN"    "$REV_CU" 2>&1 | tee "$LOGDIR/03_build_plain.log"
for b in "$LINEINFO_BIN" "$PLAIN_BIN"; do
  if [[ -x "$b" ]]; then pass "build_succeeded[$b]"; else fail "build_succeeded[$b]" "binary not produced"; fi
done
[[ "$FAIL" -gt 0 ]] && { reclaim_ownership; exit 1; }

# ---------------------------------------------------------------------
# 4. Truncate the input to exactly one grid.
# ---------------------------------------------------------------------
head -c "$SLICE_BYTES" "$FULL_BIN" > "$SLICE_BIN"
if [[ "$(stat -c%s "$SLICE_BIN")" -eq "$SLICE_BYTES" ]]; then
  pass "slice_built[$SLICE_BIN] ($RECORDS records, $SLICE_BYTES bytes)"
else
  fail "slice_built[$SLICE_BIN]" "expected $SLICE_BYTES bytes"
  reclaim_ownership; exit 1
fi

# ---------------------------------------------------------------------
# 5. CORRECTNESS GATE: plain and -lineinfo builds must agree exactly.
# ---------------------------------------------------------------------
banner "Correctness gate: plain vs -lineinfo on the same slice"
"./$PLAIN_BIN"    "$NQ" "$SLICE_BIN" "/tmp/${REV}_plain.bin"    2>&1 | tee "$LOGDIR/10_run_plain.log"
"./$LINEINFO_BIN" "$NQ" "$SLICE_BIN" "/tmp/${REV}_lineinfo.bin" 2>&1 | tee "$LOGDIR/11_run_lineinfo.log"

get_total() { grep -o 'total_sum=[0-9]*' "$1" | head -1 | cut -d= -f2; }
get_kms()   { grep -o 'kernel_ms=[0-9.]*' "$1" | head -1 | cut -d= -f2; }
T_PLAIN="$(get_total "$LOGDIR/10_run_plain.log" || true)"
T_LINE="$(get_total "$LOGDIR/11_run_lineinfo.log" || true)"
K_PLAIN="$(get_kms "$LOGDIR/10_run_plain.log" || true)"
K_LINE="$(get_kms "$LOGDIR/11_run_lineinfo.log" || true)"

if [[ -n "${T_PLAIN:-}" && "$T_PLAIN" == "${T_LINE:-}" ]]; then
  pass "lineinfo_total_matches_plain (partial total=$T_PLAIN)"
else
  fail "lineinfo_total_matches_plain" "plain='${T_PLAIN:-<none>}' lineinfo='${T_LINE:-<none>}' -- -lineinfo changed the RESULT; this measurement is void"
fi
info "partial_kernel_ms" "plain=${K_PLAIN:-?}  lineinfo=${K_LINE:-?}  (a large gap here also means -lineinfo perturbed codegen)"
# Also compare the result blobs byte-for-byte -- stronger than the total.
if cmp -s "/tmp/${REV}_plain.bin" "/tmp/${REV}_lineinfo.bin"; then
  pass "lineinfo_result_blob_byte_identical"
else
  fail "lineinfo_result_blob_byte_identical" "per-thread result arrays differ between the two builds"
fi
[[ "$FAIL" -gt 0 ]] && { reclaim_ownership; exit 1; }

export_report() {
  local rep="$1" stem="$2"
  [[ -f "$rep" ]] || { info "report_missing[$rep]" "nothing to re-export"; return 0; }
  "$NCU" --import "$rep" --page details --print-details all > "${stem}_details.txt" 2>&1 || true
  "$NCU" --import "$rep" --page source                      > "${stem}_source.txt"  2>&1 || true
  "$NCU" --import "$rep" --page source --csv                > "${stem}_source.csv"  2>&1 || true
  info "report_exported[$(basename "$stem")]" "details/source/csv written"
}

# ---------------------------------------------------------------------
# 6. SourceCounters, head slice.
# ---------------------------------------------------------------------
banner "ncu SourceCounters -- head slice ($RECORDS records)"
echo "Ctrl-C now to skip. Starting in ${STAGE_PAUSE}s..."
sleep "$STAGE_PAUSE"
S1_REP="$LOGDIR/${REV}_csrc_head_sourcecounters"
sudo "$NCU" --section SourceCounters --page source -f -o "$S1_REP" \
    "./$LINEINFO_BIN" "$NQ" "$SLICE_BIN" "/tmp/${REV}_ncu_head.bin" 2>&1 \
    | tee "$LOGDIR/20_ncu_head.log"
reclaim_ownership
if grep -qi "Source Counters" "$LOGDIR/20_ncu_head.log"; then
  info "head_sourcecounters_heading" "present"
else
  info "head_sourcecounters_heading" "NOT found -- check the raw log (not gating)"
fi
export_report "${S1_REP}.ncu-rep" "$S1_REP"

# Cross-check against 393 stage 3's SASS instruction count.
if [[ -f "${S1_REP}_source.csv" ]]; then
  NINST=$(( $(wc -l < "${S1_REP}_source.csv") - 2 ))
  if [[ "$NINST" -ge 470 && "$NINST" -le 474 ]]; then
    pass "sass_instruction_count_matches_393_stage3 ($NINST, expected 472 +/-2)"
  else
    info "sass_instruction_count_matches_393_stage3" "got $NINST, 393 stage3 saw 472 -- -lineinfo may have perturbed codegen; treat the attribution with care"
  fi
fi

# ---------------------------------------------------------------------
# 7. SourceCounters, offset slice -- is the head slice representative?
#    The filtered bin is in RAW order (393-10), so this is measured
#    rather than assumed.
# ---------------------------------------------------------------------
if [[ "$OFFSET_STAGE" == "1" ]]; then
  banner "ncu SourceCounters -- offset slice (records ${OFFSET_RECORDS}..$((OFFSET_RECORDS+RECORDS)))"
  echo "Ctrl-C now to skip. Starting in ${STAGE_PAUSE}s..."
  sleep "$STAGE_PAUSE"
  dd if="$FULL_BIN" of="$SLICE2_BIN" bs="$RECBYTES" skip="$OFFSET_RECORDS" count="$RECORDS" status=none
  if [[ "$(stat -c%s "$SLICE2_BIN")" -eq "$SLICE_BYTES" ]]; then
    pass "offset_slice_built[$SLICE2_BIN]"
    S2_REP="$LOGDIR/${REV}_csrc_off${OFFSET_RECORDS}_sourcecounters"
    sudo "$NCU" --section SourceCounters --page source -f -o "$S2_REP" \
        "./$LINEINFO_BIN" "$NQ" "$SLICE2_BIN" "/tmp/${REV}_ncu_off.bin" 2>&1 \
        | tee "$LOGDIR/30_ncu_offset.log"
    reclaim_ownership
    export_report "${S2_REP}.ncu-rep" "$S2_REP"
    pass "offset_stage_completed"
  else
    info "offset_slice_built" "dd produced the wrong size -- offset stage skipped"
  fi
fi

# ---------------------------------------------------------------------
# 8. Wrap up
# ---------------------------------------------------------------------
{
  echo "=== $REV environment capture (post) $(date -Is) ==="
  nvidia-smi -q -d CLOCK 2>&1
} > "$LOGDIR/99_env_post.txt" 2>&1

cp "$SLICE_BIN" "$LOGDIR/" 2>/dev/null || true
rm -f "$MARKER"
reclaim_ownership

TARBALL="${LOGDIR}.tar.gz"
tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" \
  || info "tarball_created" "tar failed -- send $LOGDIR/ as-is"

echo
echo "===== ${REV} summary ====="
echo "OK=$PASS  FAIL=$FAIL"
echo "SM clock       : $SM_CLOCK"
echo "partial total  : ${T_PLAIN:-?}  (plain == lineinfo, gated above)"
echo "line offset    : $REV_CU line N  ->  $SRC_CU line N-$HDR_LINES"
echo "logs           : $LOGDIR/"
echo "tarball        : $TARBALL"
if [[ "$FAIL" -gt 0 ]]; then
  echo "FAILED CHECKS:"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  exit 1
fi
echo "${REV} PASSED. Send $TARBALL for analysis."
exit 0
