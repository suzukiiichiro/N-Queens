#!/usr/bin/env bash
# 375_lineinfo_ncu_probe_N18.sh
#
# rev375 — MEASUREMENT-ONLY revision (mirrors 355's discipline: zero
# code changes anywhere, .cu/.py both byte-identical to 374). Purpose:
# confirm, on the actual CUDA C port (374Py_kernel_maxd14.cu) rather
# than the original Codon kernel, whether the two BRA/BSYNC hotspots
# identified in 329/355/357 (stall_branch_resolving concentration,
# warp-occupancy-collapse/tail-effect signature: Divergent Branches=0,
# Avg Threads Executed 2-3/32) land on the same source lines after the
# 1:1 translation to C, or whether nvcc's codegen has moved them.
#
# r2 CORRECTION (found this session): the original plan was to
# generate a fresh N=18 dump via 374Py's bench_mode=32, mirroring how
# 355/357 profiled N=18 on the ORIGINAL Codon kernel. This does not
# work -- 370Py's dispatch gates bench_mode==32 (and every other
# bench_mode in the 32-36 range) behind `N>=21`, because that whole
# streaming-bin pipeline (ensure_constellations_bin_stream and
# everything downstream of it) was built exclusively for the real
# N=21 production dataset. N=18 falls through to the ORIGINAL
# in-memory gen_constellations()/exec_solutions() path instead, which
# has no dump hook at all -- confirmed by this session's real run
# producing a normal "18: 666090624 ... ok" solve instead of a
# [soa-ref-dump-done] line.
#
# FIX: rather than touching .py dispatch code (out of scope for a
# measurement-only revision), this script instead slices the FIRST
# STRIDE (=BLOCK*MAX_BLOCKS=15488) records off 374's own
# real-hardware-confirmed N=21 filtered SoA7 file
# (constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin). At
# exactly one stride's worth of records, every thread does exactly one
# task and the grid-stride loop never repeats -- the fastest possible
# full-occupancy single-pass run, structurally equivalent in spirit to
# the "N=18 trick" (352/353: registers/occupancy are compile-time
# attributes, N-independent; the trick's actual value was always
# "small record count, fast ncu turnaround", not the literal number
# 18). N stays 21 here (board_mask/n3/n4 must match the data's true
# origin) -- this is a small-slice-of-N21 probe, not a true N=18 run.
#
# IMPORTANT CAVEAT carried forward from 352/353/README.md: static data
# (SASS structure, per-line hotspot location, register count) is
# known to be N-independent. DYNAMIC stall RATIOS are NOT guaranteed
# to transfer to the full N=21 run without separate empirical
# cross-check (355/357 validated their specific N=18-vs-N=21 case by
# also taking N=21 chunk0 SchedulerStats/WarpStateStats -- this script
# does not attempt that cross-check; treat 375's ratios as a location
# hint, not a final verdict, until a follow-up revision replicates
# that dual validation).

set -u
CUSRC="${CUSRC:-374Py_kernel_maxd14.cu}"
SRC="${SRC:-374Py_kernel_maxd14_final.py}"
FILTER_PY="${FILTER_PY:-374Py_filter_maxd14_only.py}"
LINEINFO_BIN="${LINEINFO_BIN:-375_kernel_maxd14_lineinfo}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
STATIC_ONLY="${STATIC_ONLY:-0}"

# NOTE: N is fixed at 21 -- see FIX note above. The probe data is a
# small slice of REAL N=21 data, not a synthetic or N=18 dataset.
N=21
FULL_FILTERED_N21="${FULL_FILTERED_N21:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
BLOCK="${BLOCK:-32}"
MAX_BLOCKS="${MAX_BLOCKS:-484}"
STRIDE=$((BLOCK * MAX_BLOCKS))
SLICE_RECORDS="${SLICE_RECORDS:-$STRIDE}"
RECORD_BYTES=28
SLICE_BYTES=$((SLICE_RECORDS * RECORD_BYTES))
SLICE_BIN="${SLICE_BIN:-${FULL_FILTERED_N21}.slice${SLICE_RECORDS}.bin}"

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
# 0. sudo check FIRST (352 lesson). FATAL here (unlike prior
#    revisions) -- ncu hardware counters require it, and this whole
#    revision's purpose is to take that measurement.
# ---------------------------------------------------------------------
if sudo -n true 2>/dev/null; then
  pass "sudo_permission_check"
else
  fail "sudo_permission_check" "sudo -n true failed -- ncu hardware counters require sudo, and this revision cannot do anything useful without them"
fi

# ---------------------------------------------------------------------
# 1. Presence + sha256 identity to 374 (this revision must not touch
#    the .cu or .py -- measurement only).
# ---------------------------------------------------------------------
for f in "$CUSRC" "$SRC" "$FILTER_PY"; do
  if [[ ! -f "$f" ]]; then
    fail "file_present[$f]" "$f not found in $(pwd)"
  else
    pass "file_present[$f]"
  fi
done
if [[ "$FAIL" -gt 0 ]]; then
  echo "Cannot continue. Aborting."
  exit 1
fi

REF_HASH_CU="f08fb6b7e506ff737c59be70bacf7db1ec2980b80d8b42833ec4d650df2d8ebf"
ACTUAL_HASH_CU=$(sha256sum "$CUSRC" | awk '{print $1}')
if [[ "$ACTUAL_HASH_CU" == "$REF_HASH_CU" ]]; then
  pass "cu_file_identical_to_374 (hash=$ACTUAL_HASH_CU) -- confirms zero code drift before profiling"
else
  fail "cu_file_identical_to_374" "expected hash=$REF_HASH_CU, got hash=$ACTUAL_HASH_CU -- $CUSRC has drifted since 374; this revision must not proceed with an unverified kernel"
fi

if [[ "$FAIL" -gt 0 ]]; then
  echo "Static checks failed. Not proceeding."
  exit 1
fi

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "STATIC_ONLY=1: stopping after static checks (dry run complete)."
  exit 0
fi

# ---------------------------------------------------------------------
# 2. -lineinfo build. Kept SEPARATE from 374's own gpu_build_succeeded
#    binary (374Py_kernel_maxd14) -- this binary exists only to be
#    profiled, and is never used for a correctness/timing claim.
# ---------------------------------------------------------------------
if [[ ! -x "$NVCC" ]] && ! command -v nvcc >/dev/null 2>&1; then
  fail "nvcc_toolchain_present" "$NVCC not executable and 'nvcc' not on PATH"
  exit 1
fi
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
echo "Building $CUSRC WITH -lineinfo (profiling build, separate from 374's binary)..."
rm -f "$LINEINFO_BIN"
"$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$LINEINFO_BIN" "$CUSRC" 2>&1 | tee "${LINEINFO_BIN}_build_$(date +%Y%m%d_%H%M%S).log"
if [[ ! -x "$LINEINFO_BIN" ]]; then
  fail "lineinfo_build_succeeded" "binary $LINEINFO_BIN was not produced"
  exit 1
fi
pass "lineinfo_build_succeeded"

# ---------------------------------------------------------------------
# 3. Slice: take the first STRIDE (=15488) records off 374's own
#    real-hardware-confirmed N=21 filtered SoA7 file. No Codon dispatch
#    involved at all -- pure byte-offset slicing of already-validated
#    data, so there is no correctness question to resolve here (374
#    already confirmed the source file's provenance and correctness).
# ---------------------------------------------------------------------
if [[ ! -f "$FULL_FILTERED_N21" ]]; then
  fail "full_filtered_n21_present" "$FULL_FILTERED_N21 not found -- this should already exist from 374's run. Re-run 374Py_validate_N21_full_once.sh first if it was cleaned up."
  exit 1
fi
pass "full_filtered_n21_present ($FULL_FILTERED_N21)"

FULL_BYTES=$(stat -c%s "$FULL_FILTERED_N21" 2>/dev/null || stat -f%z "$FULL_FILTERED_N21")
if [[ "$FULL_BYTES" -lt "$SLICE_BYTES" ]]; then
  fail "slice_size_sane" "$FULL_FILTERED_N21 is only $FULL_BYTES bytes, smaller than the requested slice ($SLICE_BYTES bytes for $SLICE_RECORDS records)"
  exit 1
fi

if [[ ! -f "$SLICE_BIN" ]]; then
  echo "Slicing first $SLICE_RECORDS records ($SLICE_BYTES bytes) from $FULL_FILTERED_N21..."
  dd if="$FULL_FILTERED_N21" of="$SLICE_BIN" bs=28 count="$SLICE_RECORDS" status=none
fi
SLICE_ACTUAL_BYTES=$(stat -c%s "$SLICE_BIN" 2>/dev/null || stat -f%z "$SLICE_BIN")
if [[ "$SLICE_ACTUAL_BYTES" -eq "$SLICE_BYTES" ]]; then
  pass "slice_created ($SLICE_BIN, $SLICE_RECORDS records)"
else
  fail "slice_created" "expected $SLICE_BYTES bytes, got $SLICE_ACTUAL_BYTES in $SLICE_BIN"
  exit 1
fi

# ---------------------------------------------------------------------
# 4. ncu SourceCounters, per-line (the whole point of this revision --
#    this granularity was impossible on Codon per 338's finding that
#    Codon's -debug produces invalid PTX for ncu source correlation).
# ---------------------------------------------------------------------
REPORT="${LINEINFO_BIN}_sourcecounters_N21slice${SLICE_RECORDS}_$(date +%Y%m%d_%H%M%S)"
echo "Running: sudo ncu --section SourceCounters --page source -o $REPORT -f ./$LINEINFO_BIN $N $SLICE_BIN /tmp/375_slice_results.bin"
NCU_LOG="${REPORT}.console.log"
sudo ncu --section SourceCounters --page source -o "$REPORT" -f \
  "./$LINEINFO_BIN" "$N" "$SLICE_BIN" /tmp/375_slice_results.bin \
  2>&1 | tee "$NCU_LOG"

if [[ -f "${REPORT}.ncu-rep" ]]; then
  pass "ncu_report_produced (${REPORT}.ncu-rep)"
else
  fail "ncu_report_produced" "${REPORT}.ncu-rep was not created -- check $NCU_LOG for ncu errors (sudo/permissions/lineinfo issues are the usual cause)"
fi

echo ""
echo "===== final summary ====="
echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
if [[ "$FAIL" -eq 0 ]]; then
  echo "375 PASSED (measurement-only): ${REPORT}.ncu-rep contains per-line"
  echo "SourceCounters for kernel_dfs_iter_gpu_maxd14 on N=18. Open it with"
  echo "'ncu-ui ${REPORT}.ncu-rep' or 'ncu --import ${REPORT}.ncu-rep --page source'"
  echo "and compare the hottest lines against 355/357's Codon-side finding"
  echo "(2 BRA/BSYNC sites carrying ~60-66% of branch_resolving stalls)."
  echo "No source files were modified in this revision."
fi
[[ "$FAIL" -gt 0 ]] && exit 1
exit 0
