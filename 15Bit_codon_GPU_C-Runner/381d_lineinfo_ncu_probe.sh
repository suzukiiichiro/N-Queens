#!/usr/bin/env bash
# 381d_lineinfo_ncu_probe.sh
#
# rev381e (measurement) — 381d closed most of the gap to 374
# (569,157ms -> 265,403ms) by eliminating the spin-wait. ~32%
# (265,403 vs 374's 201,232) remains unexplained. Hypothesis: at
# K_THRESHOLD=13, every task is fully handled by a single thread
# (never touches the shared queue), yet atomic_add_result() still
# does a real atomicAdd to results[task_id] on EVERY terminal hit
# within run_hybrid_episode's DFS loop -- unlike 374's process_
# one_task, which accumulates into a plain register-resident local
# variable and writes results[tid] exactly ONCE per thread's entire
# grid-stride loop. Given 377c's finding of ~1.87M DFS steps per task
# on average, this could mean millions of real atomicAdd operations
# where 374 has none. This script does NOT change 381d_hybrid_kernel_
# nospin.cu -- it only profiles it with -lineinfo + ncu SourceCounters
# to confirm or refute this hypothesis with real hardware data before
# writing any fix, same "prove don't assume" discipline as 375/381c.

set -u
CUSRC="${CUSRC:-381d_hybrid_kernel_nospin.cu}"
LINEINFO_BIN="${LINEINFO_BIN:-381d_lineinfo}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
STATIC_ONLY="${STATIC_ONLY:-0}"
SOA7_FILE="${SOA7_FILE:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
N="${N:-21}"
SLICE_RECORDS="${SLICE_RECORDS:-15488}"   # one stride's worth, same as 375
K_THRESHOLD="${K_THRESHOLD:-13}"
FQ_CAPACITY_LOG2="${FQ_CAPACITY_LOG2:-22}"

PASS=0
FAIL=0
declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }

if sudo -n true 2>/dev/null; then
  pass "sudo_permission_check"
else
  fail "sudo_permission_check" "sudo -n true failed -- ncu needs it"
fi

if [[ ! -f "$CUSRC" ]]; then
  fail "file_present[$CUSRC]" "not found in $(pwd)"
  exit 1
fi
pass "file_present[$CUSRC]"

if [[ ! -f "$SOA7_FILE" ]]; then
  fail "soa7_file_present[$SOA7_FILE]" "not found"
  exit 1
fi
pass "soa7_file_present[$SOA7_FILE]"

if [[ "$FAIL" -gt 0 ]]; then exit 1; fi

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "STATIC_ONLY=1: stopping after static checks."
  exit 0
fi

if [[ ! -x "$NVCC" ]] && ! command -v nvcc >/dev/null 2>&1; then
  fail "nvcc_toolchain_present" "$NVCC not executable and 'nvcc' not on PATH"
  exit 1
fi
[[ ! -x "$NVCC" ]] && NVCC="nvcc"

echo "Building $CUSRC WITH -lineinfo (profiling build)..."
rm -f "$LINEINFO_BIN"
"$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$LINEINFO_BIN" "$CUSRC" 2>&1 | tee "${LINEINFO_BIN}_build_$(date +%Y%m%d_%H%M%S).log"
if [[ ! -x "$LINEINFO_BIN" ]]; then
  fail "lineinfo_build_succeeded" "binary $LINEINFO_BIN was not produced"
  exit 1
fi
pass "lineinfo_build_succeeded"

# ---------------------------------------------------------------------
# Slice: reuse 375's exact technique -- cut a small prefix off the
# full 28-byte-per-record file, small enough for a fast ncu pass.
# NOTE: this kernel's own results array is sized to the RECORD COUNT
# in the file it reads (m = filesize/28), so slicing the file itself
# (not a separate --max-records flag) is the correct way to shrink
# the run for this binary -- unlike 374Py_kernel_maxd14, this binary
# takes only <N> <in_soa7_bin> <out_results_bin> [expected_total] with
# no record-count-limiting argument.
# ---------------------------------------------------------------------
SLICE_BIN="${SOA7_FILE}.slice${SLICE_RECORDS}.bin"
if [[ ! -f "$SLICE_BIN" ]]; then
  echo "Slicing first $SLICE_RECORDS records from $SOA7_FILE..."
  dd if="$SOA7_FILE" of="$SLICE_BIN" bs=28 count="$SLICE_RECORDS" status=none
fi
SLICE_BYTES=$(stat -c%s "$SLICE_BIN" 2>/dev/null || stat -f%z "$SLICE_BIN")
EXPECTED_BYTES=$((SLICE_RECORDS * 28))
if [[ "$SLICE_BYTES" -eq "$EXPECTED_BYTES" ]]; then
  pass "slice_created ($SLICE_BIN, $SLICE_RECORDS records)"
else
  fail "slice_created" "expected $EXPECTED_BYTES bytes, got $SLICE_BYTES"
  exit 1
fi

NCU="${NCU:-$(command -v ncu 2>/dev/null)}"
if [[ -z "$NCU" ]]; then
  fail "ncu_resolved" "'ncu' not found on PATH"
  exit 1
fi
pass "ncu_resolved ($NCU)"

REPORT="${LINEINFO_BIN}_sourcecounters_$(date +%Y%m%d_%H%M%S)"
echo "Running: sudo $NCU --section SourceCounters --page source -o $REPORT -f ./$LINEINFO_BIN $N $SLICE_BIN /tmp/381c_slice_results.bin"
NCU_LOG="${REPORT}.console.log"
K_THRESHOLD="$K_THRESHOLD" FQ_CAPACITY_LOG2="$FQ_CAPACITY_LOG2" \
  sudo -E "$NCU" --section SourceCounters --page source -o "$REPORT" -f \
  "./$LINEINFO_BIN" "$N" "$SLICE_BIN" /tmp/381c_slice_results.bin \
  2>&1 | tee "$NCU_LOG"

if [[ -f "${REPORT}.ncu-rep" ]]; then
  pass "ncu_report_produced (${REPORT}.ncu-rep)"
else
  fail "ncu_report_produced" "${REPORT}.ncu-rep was not created -- check $NCU_LOG"
fi

# Also produce a cuobjdump -sass dump for offset-mapping, same
# technique 375 used to translate ncu runtime addresses to source
# lines when live source correlation isn't shown in the console.
CUOBJDUMP="${CUOBJDUMP:-$(dirname "$NVCC")/cuobjdump}"
if [[ -x "$CUOBJDUMP" ]]; then
  "$CUOBJDUMP" -sass "$LINEINFO_BIN" > "${LINEINFO_BIN}_sass.txt" 2>&1
  echo "Wrote ${LINEINFO_BIN}_sass.txt (for offset-mapping hot addresses, same technique as 375)."
fi

echo ""
echo "===== 381d (measurement) summary ====="
echo "OK=$PASS  FAIL=$FAIL"
[[ "$FAIL" -gt 0 ]] && exit 1
echo "Measurement done. Please share:"
echo "  1. ${REPORT}.console.log (or the .ncu-rep if you can open it locally)"
echo "  2. ${LINEINFO_BIN}_sass.txt"
echo "so hotspots can be identified the same way 375 found the original"
echo "kernel's BSYNC/BRA concentration -- no more blind guessing."
exit 0
