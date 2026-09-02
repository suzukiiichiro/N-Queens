#!/usr/bin/env bash
# 363Py_kernel_maxd14_validate_N21_full_once.sh
#
# rev363 — kernel_dfs_iter_gpu_maxd14 CUDA C implementation. 363Py's
# code region is byte-identical to 362 (and therefore 361 r2) — this
# revision's real deliverables are 363_kernel_maxd14.cu,
# 363_filter_maxd14_only.py, and 363_kernel_reference_sim.py, validated
# here WITHOUT a GPU or nvcc.
#
# r2 CORRECTION (found this session): kernel_dfs_iter_gpu_maxd14 is
# only ever launched, in real production, on records whose required
# schedule depth is <=14 (deeper records go to kernel_dfs_iter_gpu_
# maxd16/18/20/21 instead). 361's dump has ALL 2,025,282 real records
# unconditionally, with no such filtering. The r1 version of this
# script fed that unfiltered dump straight into the maxd14-only C
# kernel and compared the total against the real correctness oracle
# 314666222712 -- both wrong: (a) the kernel's schedule-precompute loop
# has no depth bound (matching the original Codon kernel exactly, which
# relies entirely on upstream dispatch to guarantee it's never needed),
# so unfiltered real data caused an indefinite hang; (b) even had it
# not hung, 314666222712 is the sum across ALL FIVE kernels, not this
# one alone, so a maxd14-only subset could never match it anyway.
#
# r2 filters 361's dump to the maxd<=14 subset first (363_filter_
# maxd14_only.py, a literal port of schedule_depth_for_task(), the same
# filter production dispatch uses), then cross-checks 363_kernel_
# maxd14.cu's CPU-test build against 363_kernel_reference_sim.py (an
# independent standalone re-execution of the literal Codon kernel
# source) on that SAME filtered subset — both must agree byte-for-byte.
#
# This script does NOT invoke nvcc, does NOT touch the GPU, and does
# NOT run codon's full N=21 GPU pipeline — it is deliberately the
# cheapest possible real-data check before committing to a real device
# build in 364.

set -u
SRC="${SRC:-363Py_kernel_maxd14.py}"
CUSRC="${CUSRC:-363_kernel_maxd14.cu}"
EXTRACT_PY="${EXTRACT_PY:-363_extract_kernel_input_from_361_dump.py}"
FILTER_PY="${FILTER_PY:-363_filter_maxd14_only.py}"
REFSIM_PY="${REFSIM_PY:-363_kernel_reference_sim.py}"
DUMP_361="${DUMP_361:-constellations_N21_6.bin.soa_ref_361.bin}"
STATIC_ONLY="${STATIC_ONLY:-0}"
CPUTEST_BIN="${CPUTEST_BIN:-363_kernel_maxd14_cputest}"

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
# 0. sudo check FIRST (352 lesson). Non-fatal (no ncu here).
# ---------------------------------------------------------------------
if sudo -n true 2>/dev/null; then
  pass "sudo_permission_check"
else
  warn "sudo_permission_check" "sudo -n true failed (non-fatal, no ncu is invoked in this revision)"
fi

# ---------------------------------------------------------------------
# 1. Presence.
# ---------------------------------------------------------------------
for f in "$SRC" "$CUSRC" "$FILTER_PY" "$REFSIM_PY"; do
  if [[ ! -f "$f" ]]; then
    fail "file_present[$f]" "$f not found in $(pwd)"
  else
    pass "file_present[$f]"
  fi
done
if [[ "$FAIL" -gt 0 ]]; then
  echo "Cannot continue without all three files. Aborting."
  exit 1
fi

# ---------------------------------------------------------------------
# 2. Docstring-stripped .py copy, same 3-block convention as 361/362.
# ---------------------------------------------------------------------
NODOC="${SRC%.py}_nodoc.py"
python3 - "$SRC" "$NODOC" << 'PYEOF'
import sys
src_path, out_path = sys.argv[1], sys.argv[2]
with open(src_path, 'r', encoding='utf-8') as f:
    text = f.read()
parts = text.split('"""')
if len(parts) >= 7:
    rest = '"""'.join(parts[6:])
else:
    rest = text
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(rest)
PYEOF
if [[ -f "$NODOC" ]]; then
  pass "source_docstring_stripped_copy_created"
else
  fail "source_docstring_stripped_copy_created" "python3 stripping step did not produce $NODOC"
  exit 1
fi

NOTAG=$(mktemp)
grep -v '^VERSION_TAG:str="363' "$NODOC" > "$NOTAG"

# ---------------------------------------------------------------------
# 3. Full-region hash: 363Py's code region must be byte-identical to
#    362's (and therefore 361 r2's) code region. No new Codon code in
#    this revision.
# ---------------------------------------------------------------------
REF_HASH_362="2b7688b8af2db194ad3f8f60041acac7ccffbb58ef5fc74b669c261053f537ff"
REF_LINES_362=5682

ACTUAL_HASH=$(sha256sum "$NOTAG" | awk '{print $1}')
ACTUAL_LINES=$(wc -l < "$NOTAG")

if [[ "$ACTUAL_HASH" == "$REF_HASH_362" && "$ACTUAL_LINES" -eq "$REF_LINES_362" ]]; then
  pass "source_code_identical_to_362 (hash=$ACTUAL_HASH, lines=$ACTUAL_LINES)"
else
  fail "source_code_identical_to_362" "expected hash=$REF_HASH_362 lines=$REF_LINES_362, got hash=$ACTUAL_HASH lines=$ACTUAL_LINES -- code has drifted"
fi
rm -f "$NODOC" "$NOTAG"

# ---------------------------------------------------------------------
# 4. C source sanity checks.
# ---------------------------------------------------------------------
for sym in process_one_task kernel_dfs_iter_gpu_maxd14 META_NEXT IS_BASE_MASK schedule_lo schedule_hi child_jmark_mask future_check_mask terminal_depth terminal_base14; do
  if grep -q "$sym" "$CUSRC"; then
    pass "cu_symbol_present[$sym]"
  else
    fail "cu_symbol_present[$sym]" "'$sym' not found in $CUSRC"
  fi
done

# The bug found and fixed this session: process_one_task must actually
# USE its meta_next parameter (not silently fall back to the global
# META_NEXT array). Check both call sites inside process_one_task.
METAUSE_COUNT=$(grep -c 'meta_next\[schedule_fu\]' "$CUSRC")
if [[ "$METAUSE_COUNT" -ge 3 ]]; then
  pass "cu_meta_next_parameter_actually_used (found $METAUSE_COUNT usages)"
else
  fail "cu_meta_next_parameter_actually_used" "expected >=3 usages of meta_next[schedule_fu] inside process_one_task, found $METAUSE_COUNT -- this is the exact bug found and fixed this session (process_one_task silently used the global META_NEXT instead of its parameter)"
fi

if grep -qE '#ifndef __CUDACC__' "$CUSRC" && grep -qE '#ifdef __CUDACC__' "$CUSRC"; then
  pass "cu_dual_build_guards_present"
else
  fail "cu_dual_build_guards_present" "expected both #ifdef __CUDACC__ (real kernel) and #ifndef __CUDACC__ (CPU test main()) guards"
fi

if grep -qE 'schedule_depth > 1000000' "$CUSRC"; then
  pass "cu_schedule_precompute_diagnostic_cap_present"
else
  fail "cu_schedule_precompute_diagnostic_cap_present" "expected a diagnostic iteration cap on the schedule-precompute loop (CPU-test build only) -- this is what would have caught the r1 hang quickly instead of silently spinning for an hour"
fi

if grep -q 'def schedule_depth_for_task' "$FILTER_PY"; then
  pass "filter_py_has_schedule_depth_for_task"
else
  fail "filter_py_has_schedule_depth_for_task" "$FILTER_PY is missing its literal port of schedule_depth_for_task()"
fi

if grep -q 'def process_one_task' "$REFSIM_PY"; then
  pass "refsim_py_has_process_one_task"
else
  fail "refsim_py_has_process_one_task" "$REFSIM_PY is missing its standalone process_one_task() re-execution"
fi

# ---------------------------------------------------------------------
# 5. Negative test: tamper with the .py core -> hash check must FAIL.
# ---------------------------------------------------------------------
NEGTMP=$(mktemp)
python3 - "$SRC" "$NEGTMP" << 'PYEOF'
import sys
with open(sys.argv[1], encoding='utf-8') as f:
    text = f.read()
parts = text.split('"""')
rest = '"""'.join(parts[6:])
lines = [l for l in rest.split('\n') if not l.startswith('VERSION_TAG:str="363')]
notag = '\n'.join(lines)
notag = notag.replace(
    "def auto_sort_mode(N:int)->int:",
    "def auto_sort_mode(N:int)->int:  # tampered",
    1
)
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    f.write(notag)
PYEOF
NEG_HASH=$(sha256sum "$NEGTMP" | awk '{print $1}')
if [[ "$NEG_HASH" != "$REF_HASH_362" ]]; then
  pass "negtest_core_tamper_detected"
else
  fail "negtest_core_tamper_detected" "tampering did not change the hash (test harness bug)"
fi
rm -f "$NEGTMP"

# ---------------------------------------------------------------------
# 6. Summary.
# ---------------------------------------------------------------------
echo ""
echo "===== 363 static-check summary ====="
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
# 7. Build the CPU-only test harness with gcc (no CUDA toolkit
#    required — this is the whole point of the HOSTDEV/#ifdef
#    __CUDACC__ dual-build structure).
# ---------------------------------------------------------------------
echo "Building CPU-only test harness (gcc, no CUDA toolkit needed)..."
CC="${CC:-gcc}"
if ! command -v "$CC" >/dev/null 2>&1; then
  CC=cc
fi
if ! command -v "$CC" >/dev/null 2>&1; then
  fail "c_toolchain_present" "neither gcc nor cc found on PATH"
  exit 1
fi
rm -f "$CPUTEST_BIN"
"$CC" -x c -O2 -Wall -Wextra -o "$CPUTEST_BIN" "$CUSRC" 2>&1 | tee "${CPUTEST_BIN}_build_$(date +%Y%m%d_%H%M%S).log"
if [[ ! -x "$CPUTEST_BIN" ]]; then
  fail "cputest_build_succeeded" "binary $CPUTEST_BIN was not produced"
  exit 1
fi
pass "cputest_build_succeeded"

# The build must be warning-free (this is exactly how the meta_next
# bug was caught this session via -Wunused-parameter).
BUILD_LOG=$(ls -t "${CPUTEST_BIN}_build_"*.log 2>/dev/null | head -n1)
if [[ -n "$BUILD_LOG" ]] && grep -qi 'warning' "$BUILD_LOG"; then
  fail "cputest_build_warning_free" "gcc emitted warnings, see $BUILD_LOG -- do not ignore these, they have already caught one real bug this session"
else
  pass "cputest_build_warning_free"
fi

# ---------------------------------------------------------------------
# 8. Real N=21 data check.
#
#    IMPORTANT CORRECTION (found this session): kernel_dfs_iter_gpu_
#    maxd14 is only ever launched, in real production, on records whose
#    required schedule depth is <=14 -- deeper records are routed to
#    kernel_dfs_iter_gpu_maxd16/18/20/21 instead. 361's dump contains
#    ALL 2,025,282 real records unconditionally (build_soa_for_range()
#    doesn't filter by depth), so feeding it straight into this
#    maxd14-only kernel is testing outside the kernel's documented
#    precondition -- some records need >14 schedule levels, and this
#    kernel's schedule-precompute loop (matching the original Codon
#    kernel exactly) has no depth bound of its own, since production
#    dispatch is supposed to guarantee it's never needed. This is what
#    caused the r1 harness to hang indefinitely on Suzuki's real data.
#
#    Consequently the real correctness oracle 314666222712 (the sum
#    across ALL FIVE kernels) is NOT the right target for a maxd14-only
#    subset -- that subset's true total is smaller and not independently
#    known here. The correct check is: (a) filter 361's dump down to
#    only the maxd<=14 subset using the exact same filter production
#    dispatch uses (363_filter_maxd14_only.py, a literal port of
#    schedule_depth_for_task()); (b) run BOTH 363_kernel_maxd14.cu's
#    CPU-test build AND 363_kernel_reference_sim.py (an independent
#    standalone re-execution of the literal Codon kernel source, not
#    derived from the C port) on that SAME filtered subset; (c) the two
#    must agree byte-for-byte and in total, whatever that total is.
# ---------------------------------------------------------------------
if [[ ! -f "$DUMP_361" ]]; then
  info "real_data_check" "361's dump file '$DUMP_361' not found in $(pwd) -- skipping the real-data check. Place 361's confirmed dump (constellations_N21_6.bin.soa_ref_361.bin, 2,025,282 records) alongside this script and re-run to complete this step before moving to 364 (nvcc/GPU build)."
else
  pass "dump_361_found"

  FILTERED="${DUMP_361}.maxd14only_363.bin"
  echo "Filtering $DUMP_361 down to the maxd<=14 subset (production dispatch's own precondition)..."
  python3 "$FILTER_PY" "$DUMP_361" "$FILTERED" 2>&1 | tee "${CPUTEST_BIN}_filter_$(date +%Y%m%d_%H%M%S).log"
  if [[ ! -f "$FILTERED" ]]; then
    fail "filter_succeeded" "$FILTERED was not produced"
  else
    pass "filter_succeeded"

    C_RESULTS="${FILTERED}.c_results.bin"
    PY_RESULTS="${FILTERED}.py_results.bin"

    echo "Running C kernel (CPU build) on the filtered maxd<=14 subset..."
    "./$CPUTEST_BIN" 21 "$FILTERED" "$C_RESULTS" 2>&1 | tee "${CPUTEST_BIN}_realdata_$(date +%Y%m%d_%H%M%S).log"
    C_LOG=$(ls -t "${CPUTEST_BIN}_realdata_"*.log 2>/dev/null | head -n1)
    C_TOTAL=$(grep -oE 'total_sum=[0-9]+' "$C_LOG" | tail -n1 | cut -d= -f2)

    echo "Running independent Python reference simulation on the same subset..."
    python3 "$REFSIM_PY" "$FILTERED" "$PY_RESULTS" 21 2>&1 | tee "${CPUTEST_BIN}_refsim_$(date +%Y%m%d_%H%M%S).log"
    PY_LOG=$(ls -t "${CPUTEST_BIN}_refsim_"*.log 2>/dev/null | head -n1)
    PY_TOTAL=$(grep -oE 'total_sum=[0-9]+' "$PY_LOG" | tail -n1 | cut -d= -f2)

    if [[ -n "$C_TOTAL" && "$C_TOTAL" == "$PY_TOTAL" ]]; then
      pass "c_vs_python_total_match (total=$C_TOTAL)"
    else
      fail "c_vs_python_total_match" "C total=${C_TOTAL:-missing}, Python total=${PY_TOTAL:-missing} -- these must agree; do not proceed to 364 until they do"
    fi

    if cmp -s "$C_RESULTS" "$PY_RESULTS"; then
      pass "c_vs_python_byte_identical"
    else
      fail "c_vs_python_byte_identical" "cmp reported a difference: $(cmp "$C_RESULTS" "$PY_RESULTS" 2>&1)"
    fi
  fi
fi

echo ""
echo "===== final summary ====="
echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
if [[ "$FAIL" -eq 0 && "$INFO" -eq 0 ]]; then
  echo "363 PASSED end-to-end: the CUDA C kernel port agrees byte-for-byte"
  echo "with an independent Python re-execution of the real Codon kernel"
  echo "source, on the real N=21 maxd<=14 subset, computed entirely on"
  echo "CPU with no GPU or nvcc. Ready to proceed to 364 (real nvcc"
  echo "device build, GPU launch, and +-3% timing comparison)."
fi
[[ "$FAIL" -gt 0 ]] && exit 1
exit 0
