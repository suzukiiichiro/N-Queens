#!/usr/bin/env bash
# 364Py_kernel_maxd14_gpu_validate_N21_full_once.sh
#
# rev364 — real nvcc device build + GPU launch of kernel_dfs_iter_gpu_
# maxd14 on real N=21 data. 364Py's code region is byte-identical to
# 363 (r2) — this revision's real deliverable is the host-side GPU
# runner added to 364_kernel_maxd14.cu (a copy of 363's file plus one
# new main(), guarded by #ifdef __CUDACC__).
#
# CORRECTED UNDERSTANDING (from 363's investigation): every real N=21
# record is exactly schedule-depth 14 (0 records excluded by the
# maxd<=14 filter), so the maxd14-only subset IS the full real N=21
# dataset. total_sum==314666222712 (the real correctness oracle) is
# therefore the correct and meaningful target for this GPU run.
#
# This script requires a real CUDA toolkit (nvcc) and GPU — it cannot
# run in a sandbox without one. It builds nvcc's device binary, runs it
# against the already-filtered SoA7 file from 363 (or produces one from
# 361's dump + 363_filter_maxd14_only.py if not already present), and
# checks the GPU-computed total against the real oracle.

set -u
SRC="${SRC:-364Py_kernel_maxd14_gpu.py}"
CUSRC="${CUSRC:-364_kernel_maxd14.cu}"
FILTER_PY="${FILTER_PY:-363_filter_maxd14_only.py}"
DUMP_361="${DUMP_361:-constellations_N21_6.bin.soa_ref_361.bin}"
FILTERED_363="${FILTERED_363:-${DUMP_361}.maxd14only_363.bin}"
STATIC_ONLY="${STATIC_ONLY:-0}"
GPU_BIN="${GPU_BIN:-364_kernel_maxd14}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
EXPECTED_ORACLE="${EXPECTED_ORACLE:-314666222712}"

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
# 0. sudo check FIRST (352 lesson). Non-fatal (no ncu in this revision
#    — that comes once correctness here is confirmed).
# ---------------------------------------------------------------------
if sudo -n true 2>/dev/null; then
  pass "sudo_permission_check"
else
  warn "sudo_permission_check" "sudo -n true failed (non-fatal, no ncu is invoked in this revision)"
fi

# ---------------------------------------------------------------------
# 1. Presence.
# ---------------------------------------------------------------------
for f in "$SRC" "$CUSRC" "$FILTER_PY"; do
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
# 2. Docstring-stripped .py copy.
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
grep -v '^VERSION_TAG:str="364' "$NODOC" > "$NOTAG"

# ---------------------------------------------------------------------
# 3. Full-region hash: 364Py's code region must be byte-identical to
#    363's (and therefore 362/361 r2's) code region. No new Codon code.
# ---------------------------------------------------------------------
REF_HASH_363="2b7688b8af2db194ad3f8f60041acac7ccffbb58ef5fc74b669c261053f537ff"
REF_LINES_363=5682

ACTUAL_HASH=$(sha256sum "$NOTAG" | awk '{print $1}')
ACTUAL_LINES=$(wc -l < "$NOTAG")

if [[ "$ACTUAL_HASH" == "$REF_HASH_363" && "$ACTUAL_LINES" -eq "$REF_LINES_363" ]]; then
  pass "source_code_identical_to_363 (hash=$ACTUAL_HASH, lines=$ACTUAL_LINES)"
else
  fail "source_code_identical_to_363" "expected hash=$REF_HASH_363 lines=$REF_LINES_363, got hash=$ACTUAL_HASH lines=$ACTUAL_LINES -- code has drifted"
fi
rm -f "$NODOC" "$NOTAG"

# ---------------------------------------------------------------------
# 4. .cu source sanity checks: everything from 363 must still be
#    present, plus the new GPU runner.
# ---------------------------------------------------------------------
for sym in process_one_task kernel_dfs_iter_gpu_maxd14 META_NEXT IS_BASE_MASK; do
  if grep -q "$sym" "$CUSRC"; then
    pass "cu_symbol_present[$sym]"
  else
    fail "cu_symbol_present[$sym]" "'$sym' not found in $CUSRC"
  fi
done

for sym in cudaMalloc cudaMemcpy cudaEventCreate cudaEventElapsedTime "kernel_dfs_iter_gpu_maxd14<<<" CUDA_CHECK; do
  if grep -qF "$sym" "$CUSRC"; then
    pass "cu_gpu_runner_present[$sym]"
  else
    fail "cu_gpu_runner_present[$sym]" "'$sym' not found in $CUSRC -- the host-side GPU runner appears incomplete"
  fi
done

if grep -qE '^\s*int main\(int argc, char \*\*argv\) \{' "$CUSRC"; then
  MAIN_COUNT=$(grep -cE '^\s*int main\(int argc, char \*\*argv\) \{' "$CUSRC")
  if [[ "$MAIN_COUNT" -eq 2 ]]; then
    pass "cu_has_exactly_two_mains (one CPU-test, one GPU runner)"
  else
    fail "cu_has_exactly_two_mains" "expected exactly 2 main() definitions (CPU-test + GPU runner), found $MAIN_COUNT"
  fi
else
  fail "cu_has_exactly_two_mains" "no main() found at all"
fi

METAUSE_COUNT=$(grep -c 'meta_next\[schedule_fu\]' "$CUSRC")
if [[ "$METAUSE_COUNT" -ge 3 ]]; then
  pass "cu_meta_next_parameter_actually_used (found $METAUSE_COUNT usages)"
else
  fail "cu_meta_next_parameter_actually_used" "expected >=3 usages, found $METAUSE_COUNT -- regression of the r2 fix"
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
lines = [l for l in rest.split('\n') if not l.startswith('VERSION_TAG:str="364')]
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
if [[ "$NEG_HASH" != "$REF_HASH_363" ]]; then
  pass "negtest_core_tamper_detected"
else
  fail "negtest_core_tamper_detected" "tampering did not change the hash (test harness bug)"
fi
rm -f "$NEGTMP"

# ---------------------------------------------------------------------
# 6. Summary.
# ---------------------------------------------------------------------
echo ""
echo "===== 364 static-check summary ====="
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
# 7. nvcc device build. This step requires a real CUDA toolkit and GPU
#    — it cannot succeed in a sandbox without one.
# ---------------------------------------------------------------------
if [[ ! -x "$NVCC" ]] && ! command -v nvcc >/dev/null 2>&1; then
  fail "nvcc_toolchain_present" "$NVCC not executable and 'nvcc' not on PATH -- this step requires a real CUDA toolkit"
  echo ""
  echo "===== updated summary ====="
  echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
  exit 1
fi
if [[ ! -x "$NVCC" ]]; then
  NVCC="nvcc"
fi

echo "Building with $NVCC -arch=$ARCH ..."
rm -f "$GPU_BIN"
"$NVCC" -O3 -arch="$ARCH" -o "$GPU_BIN" "$CUSRC" 2>&1 | tee "${GPU_BIN}_build_$(date +%Y%m%d_%H%M%S).log"
if [[ ! -x "$GPU_BIN" ]]; then
  fail "gpu_build_succeeded" "binary $GPU_BIN was not produced"
  exit 1
fi
pass "gpu_build_succeeded"

# ---------------------------------------------------------------------
# 8. Ensure the filtered maxd<=14 SoA7 input exists (reuse 363's if
#    already present, else regenerate from 361's dump).
# ---------------------------------------------------------------------
if [[ ! -f "$FILTERED_363" ]]; then
  if [[ ! -f "$DUMP_361" ]]; then
    fail "filtered_input_available" "neither $FILTERED_363 nor $DUMP_361 found -- cannot proceed without real N=21 data. Place 361's confirmed dump alongside this script and re-run."
    echo ""
    echo "===== updated summary ====="
    echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
    exit 1
  fi
  echo "Filtered input not found, regenerating from $DUMP_361..."
  python3 "$FILTER_PY" "$DUMP_361" "$FILTERED_363"
fi
if [[ -f "$FILTERED_363" ]]; then
  pass "filtered_input_available ($FILTERED_363)"
else
  fail "filtered_input_available" "$FILTERED_363 still not present after regeneration attempt"
  exit 1
fi

# ---------------------------------------------------------------------
# 9. Real GPU run, with the real correctness oracle as the expected
#    total (see header comment for why this is now the correct target).
# ---------------------------------------------------------------------
GPU_RESULTS="${FILTERED_363}.gpu_results.bin"
echo "Running on real GPU: ./$GPU_BIN 21 $FILTERED_363 $GPU_RESULTS $EXPECTED_ORACLE"
GPU_LOG="${GPU_BIN}_run_$(date +%Y%m%d_%H%M%S).log"
"./$GPU_BIN" 21 "$FILTERED_363" "$GPU_RESULTS" "$EXPECTED_ORACLE" 2>&1 | tee "$GPU_LOG"

DONE_LINE=$(grep '^\[gpu-run-done\]' "$GPU_LOG" | tail -n1)
CORRECTNESS_LINE=$(grep '^\[gpu-run-correctness\]' "$GPU_LOG" | tail -n1)

if [[ -z "$DONE_LINE" ]]; then
  fail "gpu_run_completed" "no [gpu-run-done] line in $GPU_LOG -- the GPU run did not complete as expected (check for CUDA errors above)"
else
  pass "gpu_run_completed"
  echo "$DONE_LINE"
fi

if [[ "$CORRECTNESS_LINE" == *"MATCH"* && "$CORRECTNESS_LINE" != *"MISMATCH"* ]]; then
  pass "gpu_total_matches_oracle (expected=$EXPECTED_ORACLE)"
elif [[ -n "$CORRECTNESS_LINE" ]]; then
  fail "gpu_total_matches_oracle" "$CORRECTNESS_LINE -- do not proceed to a timing comparison until this is resolved"
else
  fail "gpu_total_matches_oracle" "no [gpu-run-correctness] line found -- was EXPECTED_ORACLE passed correctly?"
fi

echo ""
echo "===== final summary ====="
echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
if [[ "$FAIL" -eq 0 ]]; then
  echo "364 PASSED: the real nvcc-built, GPU-executed kernel reproduces"
  echo "the correctness oracle (314666222712) on real N=21 data."
  echo "Ready to proceed to 365: wiring this into the 3-chunk measure2"
  echo "protocol for a rigorous +-3% timing comparison against the 356"
  echo "anchor (393.404s)."
fi
[[ "$FAIL" -gt 0 ]] && exit 1
exit 0
