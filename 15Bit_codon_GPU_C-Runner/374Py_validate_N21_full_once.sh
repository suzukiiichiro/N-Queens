#!/usr/bin/env bash
# 374_validate_N21_full_once.sh
#
# rev374 — CONSOLIDATION revision. No Codon/.cu code changes anywhere
# in this revision (mirrors 371/372/373's discipline: 371 confirmed
# 370's code unchanged via sha256, 372 was investigation docs only,
# 373 was a single hardcoded constant update). 374's actual
# deliverable is THIS SCRIPT: a single integrated validation harness
# that formally packages the CUDA C port's current final state as one
# reproducible entry point, replacing the need to manually chain
# 361/363/364/365's separate per-revision scripts together.
#
# Per Suzuki-san's confirmation this session, the three source files
# below are the current final state and are used UNCHANGED:
#   - 370Py_mem_probe_v2.py   (the final Codon source; 371 added no
#                              code on top of it, confirmed via sha256
#                              in that revision)
#   - 364_kernel_maxd14.cu    (kernel + CPU test harness + GPU host
#                              runner; unchanged since 364, confirmed
#                              unchanged through the g5.2xlarge
#                              environment migration and 373)
#   - 363_filter_maxd14_only.py (unchanged since 363)
#
# 374 additionally formally adds 363_filter_maxd14_only.py to the
# tracked deliverable set (previously only referenced ad hoc by
# 363/364's own scripts) and makes the full chain of prerequisite bin
# files self-generating: if the maxd14-filtered SoA7 input is not
# present, this script produces it from the 361-style dump; if THAT
# dump is not present either, this script runs 370Py's bench_mode=32
# to produce it (which in turn transparently generates the raw
# constellations_N21_*.bin via ensure_constellations_bin_stream() if
# that doesn't exist yet either -- unchanged since 361). A fresh
# checkout with no cached bin files at all can therefore run this
# single script start to finish.
#
# This script requires a real CUDA toolkit (nvcc), a GPU, and codon --
# it cannot complete end-to-end in a sandbox without them (static
# checks and STATIC_ONLY=1 do not require any of these).

set -u
SRC="${SRC:-374Py_kernel_maxd14_final.py}"
BIN="${BIN:-374Py_kernel_maxd14_final}"
CUSRC="${CUSRC:-374Py_kernel_maxd14.cu}"
GPU_BIN="${GPU_BIN:-374Py_kernel_maxd14}"
FILTER_PY="${FILTER_PY:-374Py_filter_maxd14_only.py}"
DUMP_361="${DUMP_361:-constellations_N21_6.bin.soa_ref_361.bin}"
FILTERED_363="${FILTERED_363:-${DUMP_361}.maxd14only_363.bin}"
STATIC_ONLY="${STATIC_ONLY:-0}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
EXPECTED_ORACLE="${EXPECTED_ORACLE:-314666222712}"
N="${N:-21}"

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
# 0. sudo check FIRST (352 lesson). Non-fatal (no ncu in this revision).
# ---------------------------------------------------------------------
if sudo -n true 2>/dev/null; then
  pass "sudo_permission_check"
else
  warn "sudo_permission_check" "sudo -n true failed (non-fatal, no ncu is invoked in this revision)"
fi

# ---------------------------------------------------------------------
# 1. Presence of all three tracked source files.
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
# 2. 370Py core-region hash (docstring-stripped, VERSION_TAG-excluded,
#    same convention as 361/363/365). Confirms the actual final Codon
#    source in hand is byte-identical to the 370 state Suzuki-san
#    confirmed this session, with zero drift since.
# ---------------------------------------------------------------------
NODOC="${SRC%.py}_nodoc.py"
python3 - "$SRC" "$NODOC" << 'PYEOF'
import sys
src_path, out_path = sys.argv[1], sys.argv[2]
with open(src_path, 'r', encoding='utf-8') as f:
    text = f.read()
parts = text.split('"""')
rest = '"""'.join(parts[6:]) if len(parts) >= 7 else text
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
grep -v '^VERSION_TAG:str="374' "$NODOC" > "$NOTAG"
rm -f "$NODOC"

REF_HASH_370="17c61ccfa5ae4890b6a484035483387a321f3713551dfbb0e0214c8c4f62df26"  # == 374Py's own core hash: proves zero logic drift across the 370->374 rename
REF_LINES_370=6123
ACTUAL_HASH=$(sha256sum "$NOTAG" | awk '{print $1}')
ACTUAL_LINES=$(wc -l < "$NOTAG")
if [[ "$ACTUAL_HASH" == "$REF_HASH_370" && "$ACTUAL_LINES" -eq "$REF_LINES_370" ]]; then
  pass "source_core_identical_to_370_and_374 (hash=$ACTUAL_HASH, lines=$ACTUAL_LINES)"
else
  fail "source_core_identical_to_370_and_374" "expected hash=$REF_HASH_370 lines=$REF_LINES_370, got hash=$ACTUAL_HASH lines=$ACTUAL_LINES -- code has drifted since 370/371/374"
fi

# Negative test: tampering must change the hash (harness self-check).
NEGTMP=$(mktemp)
sed 's/def auto_sort_mode(N:int)->int:/def auto_sort_mode(N:int)->int:  # tampered/' "$NOTAG" > "$NEGTMP"
NEG_HASH=$(sha256sum "$NEGTMP" | awk '{print $1}')
if [[ "$NEG_HASH" != "$REF_HASH_370" ]]; then
  pass "negtest_370_core_tamper_detected"
else
  fail "negtest_370_core_tamper_detected" "tampering did not change the hash (test harness bug)"
fi
rm -f "$NOTAG" "$NEGTMP"

# Targeted checks: the three dispatch branches this script actually
# depends on must still be present and reachable.
for pat in \
  'if use_gpu and N>=21 and bench_mode==32:' \
  'if use_gpu and N>=21 and bench_mode==33:' \
  'def ensure_constellations_bin_stream' \
  'def exec_solutions_gpu_single_shot' \
  'def dump_soa_reference_c_port' \
; do
  if grep -qF "$pat" "$SRC"; then
    pass "source_symbol_present[$pat]"
  else
    fail "source_symbol_present[$pat]" "'$pat' not found in $SRC"
  fi
done

# ---------------------------------------------------------------------
# 3. .cu raw-file hash (whole-file, no docstring stripping -- unlike
#    the .py, this file has no docstring-block convention). Confirms
#    the kernel + CPU test harness + 364 GPU host runner are all still
#    exactly the bytes confirmed this session.
# ---------------------------------------------------------------------
REF_HASH_CU="f08fb6b7e506ff737c59be70bacf7db1ec2980b80d8b42833ec4d650df2d8ebf"  # 374Py_kernel_maxd14.cu (renamed from 364, header comments only differ)
ACTUAL_HASH_CU=$(sha256sum "$CUSRC" | awk '{print $1}')
if [[ "$ACTUAL_HASH_CU" == "$REF_HASH_CU" ]]; then
  pass "cu_file_identical_to_374 (hash=$ACTUAL_HASH_CU)"
else
  fail "cu_file_identical_to_374" "expected hash=$REF_HASH_CU, got hash=$ACTUAL_HASH_CU -- $CUSRC has drifted since 374"
fi

for sym in process_one_task kernel_dfs_iter_gpu_maxd14 META_NEXT \
           cudaMalloc cudaEventCreate cudaEventElapsedTime CUDA_CHECK \
           'kernel_dfs_iter_gpu_maxd14<<<'; do
  if grep -qF "$sym" "$CUSRC"; then
    pass "cu_symbol_present[$sym]"
  else
    fail "cu_symbol_present[$sym]" "'$sym' not found in $CUSRC"
  fi
done

MAIN_COUNT=$(grep -cE '^\s*int main\(int argc, char \*\*argv\) \{' "$CUSRC")
if [[ "$MAIN_COUNT" -eq 2 ]]; then
  pass "cu_has_exactly_two_mains (CPU-test + GPU runner)"
else
  fail "cu_has_exactly_two_mains" "expected exactly 2 main() definitions, found $MAIN_COUNT"
fi

# ---------------------------------------------------------------------
# 4. filter script raw-file hash + sanity check.
# ---------------------------------------------------------------------
REF_HASH_FILTER="4eaff0bd43e51d7c4c2c462a91b7c93a4e4f9121602ac33ea3f59e385eab46c5"  # 374Py_filter_maxd14_only.py (renamed from 363, docstring only differs)
ACTUAL_HASH_FILTER=$(sha256sum "$FILTER_PY" | awk '{print $1}')
if [[ "$ACTUAL_HASH_FILTER" == "$REF_HASH_FILTER" ]]; then
  pass "filter_py_identical_to_374 (hash=$ACTUAL_HASH_FILTER)"
else
  fail "filter_py_identical_to_374" "expected hash=$REF_HASH_FILTER, got hash=$ACTUAL_HASH_FILTER -- $FILTER_PY has drifted since 374"
fi
if grep -q 'def schedule_depth_for_task' "$FILTER_PY"; then
  pass "filter_py_has_schedule_depth_for_task"
else
  fail "filter_py_has_schedule_depth_for_task" "$FILTER_PY is missing its literal port of schedule_depth_for_task()"
fi

# ---------------------------------------------------------------------
# 5. Summary of static checks.
# ---------------------------------------------------------------------
echo ""
echo "===== 374 static-check summary ====="
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
# 6. Build 370Py (codon) and 364_kernel_maxd14.cu (nvcc). Both are
#    required from here on -- this step needs real toolchains.
# ---------------------------------------------------------------------
if ! command -v codon >/dev/null 2>&1; then
  fail "codon_toolchain_present" "codon not found on PATH"
  echo "" ; echo "===== updated summary =====" ; echo "OK=$PASS  FAIL=$((FAIL+1))  INFO=$INFO  WARN=$WARN"
  exit 1
fi
echo "Building 370Py (codon)..."
CAND="./${SRC%.py}"
rm -f "$CAND"
codon build -release "$SRC" 2>&1 | tee "${BIN}_build_$(date +%Y%m%d_%H%M%S).log"
if [[ ! -x "$CAND" ]]; then
  fail "py_build_succeeded" "binary $CAND was not produced"
  exit 1
fi
pass "py_build_succeeded"

if [[ ! -x "$NVCC" ]] && ! command -v nvcc >/dev/null 2>&1; then
  fail "nvcc_toolchain_present" "$NVCC not executable and 'nvcc' not on PATH"
  echo "" ; echo "===== updated summary =====" ; echo "OK=$PASS  FAIL=$((FAIL+1))  INFO=$INFO  WARN=$WARN"
  exit 1
fi
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
echo "Building $CUSRC with $NVCC -arch=$ARCH ..."
rm -f "$GPU_BIN"
"$NVCC" -O3 -arch="$ARCH" -o "$GPU_BIN" "$CUSRC" 2>&1 | tee "${GPU_BIN}_build_$(date +%Y%m%d_%H%M%S).log"
if [[ ! -x "$GPU_BIN" ]]; then
  fail "gpu_build_succeeded" "binary $GPU_BIN was not produced"
  exit 1
fi
pass "gpu_build_succeeded"

# ---------------------------------------------------------------------
# 7. NEW IN 374: full self-generating bin chain. If the filtered
#    maxd<=14 SoA7 input is missing, produce it -- regenerating the
#    361-style dump first via 370Py's bench_mode=32 if THAT is missing
#    too. bench_mode=32's own first step is ensure_constellations_bin_
#    stream(), which transparently (re)generates the raw
#    constellations_N21_*.bin from scratch if it isn't already cached
#    (unchanged since 361) -- so this one branch covers all three
#    levels of the dependency chain (raw bin -> 361 dump -> filtered
#    bin) with no manual intervention needed on a clean checkout.
# ---------------------------------------------------------------------
if [[ ! -f "$FILTERED_363" ]]; then
  if [[ ! -f "$DUMP_361" ]]; then
    info "dump_361_missing" "$DUMP_361 not found -- generating via 370Py bench_mode=32 (this also generates the raw constellations bin first if needed)"
    DUMP_LOG="${BIN}_bench32_$(date +%Y%m%d_%H%M%S).log"
    stdbuf -oL -eL "$CAND" -g "$N" "$N" 32 484 1 0 7 32 3 7 0 0 1 2 2048 9 2>&1 | tee "$DUMP_LOG"
    if [[ ! -f "$DUMP_361" ]]; then
      fail "dump_361_generated" "$DUMP_361 still not present after bench_mode=32 run -- check $DUMP_LOG"
      echo "" ; echo "===== updated summary =====" ; echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
      exit 1
    fi
    pass "dump_361_generated ($DUMP_361)"
  else
    pass "dump_361_already_present ($DUMP_361)"
  fi
  echo "Filtering $DUMP_361 down to the maxd<=14 subset..."
  python3 "$FILTER_PY" "$DUMP_361" "$FILTERED_363" 2>&1 | tee "${GPU_BIN}_filter_$(date +%Y%m%d_%H%M%S).log"
fi
if [[ -f "$FILTERED_363" ]]; then
  pass "filtered_input_available ($FILTERED_363)"
else
  fail "filtered_input_available" "$FILTERED_363 still not present after generation attempts"
  exit 1
fi

# ---------------------------------------------------------------------
# 8. Real GPU run against the real correctness oracle. Every real N=21
#    record is exactly schedule-depth 14 (0 dropped by the filter, per
#    363's finding), so the maxd14-only subset IS the full N=21
#    dataset and 314666222712 is the correct target.
# ---------------------------------------------------------------------
GPU_RESULTS="${FILTERED_363}.gpu_results.bin"
echo "Running on real GPU: ./$GPU_BIN $N $FILTERED_363 $GPU_RESULTS $EXPECTED_ORACLE"
GPU_LOG="${GPU_BIN}_run_$(date +%Y%m%d_%H%M%S).log"
"./$GPU_BIN" "$N" "$FILTERED_363" "$GPU_RESULTS" "$EXPECTED_ORACLE" 2>&1 | tee "$GPU_LOG"

DONE_LINE=$(grep '^\[gpu-run-done\]' "$GPU_LOG" | tail -n1)
CORRECTNESS_LINE=$(grep '^\[gpu-run-correctness\]' "$GPU_LOG" | tail -n1)

if [[ -z "$DONE_LINE" ]]; then
  fail "gpu_run_completed" "no [gpu-run-done] line in $GPU_LOG"
else
  pass "gpu_run_completed"
  echo "$DONE_LINE"
fi

if [[ "$CORRECTNESS_LINE" == *"MATCH"* && "$CORRECTNESS_LINE" != *"MISMATCH"* ]]; then
  pass "gpu_total_matches_oracle (expected=$EXPECTED_ORACLE)"
elif [[ -n "$CORRECTNESS_LINE" ]]; then
  fail "gpu_total_matches_oracle" "$CORRECTNESS_LINE"
else
  fail "gpu_total_matches_oracle" "no [gpu-run-correctness] line found"
fi

echo ""
echo "===== final summary ====="
echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
if [[ "$FAIL" -eq 0 ]]; then
  echo "374 PASSED: 370Py + 364_kernel_maxd14.cu + 363_filter_maxd14_only.py"
  echo "confirmed byte-identical to their reference states and, together,"
  echo "reproduce the correctness oracle (314666222712) on real N=21 data"
  echo "via a single self-contained entry point (bin files auto-generated"
  echo "from a clean checkout if not already cached)."
fi
[[ "$FAIL" -gt 0 ]] && exit 1
exit 0
