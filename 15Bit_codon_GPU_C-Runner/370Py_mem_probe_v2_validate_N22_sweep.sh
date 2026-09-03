#!/usr/bin/env bash
# 370Py_mem_probe_v2_validate_N22_sweep.sh
#
# rev370 — BUGFIX on 369's OWN new code (read_vmhwm_kb()), not on
# 366/368. Nothing from 366/368/earlier is touched.
#
# Context: 369's real-hardware sweep for N=22 ran successfully through
# record_limit=10,000,000 and failed (no [mem-probe-done] line) at
# record_limit=15,000,000 -- a genuine, record-count-proportional
# crash bracket that independently confirms the memory-scaling
# hypothesis, regardless of the bug below. But every rung, including
# the successful ones, reported vmhwm_*_kb=-1 (all deltas 0) --
# read_vmhwm_kb() was silently non-functional throughout. Suspected
# cause: /proc/self/status is a procfs pseudo-file reporting st_size=0
# via stat(); 369's size-based f.read() (the same idiom
# count_constellations_bin_records uses successfully, but only for
# REAL files with correct stat sizes) may have preallocated a 0-byte
# buffer and returned before the actual read syscall ran.
#
# 370's ONLY delta versus 369's executable code region is a single
# MODIFICATION (not a pure insertion): the entire body of
# read_vmhwm_kb() is replaced with an explicit fixed-size chunk-read
# loop (===370-VMHWM-FIX-BEGIN/END===), which does not depend on the
# file's reported size at all.
#
# IMPORTANT: the reference this script checks 370's core against is
# 369's OWN FULL code region (369's deltas from 361/365/366/368
# intact) -- 370 is built on top of 369.

set -u
SRC="${SRC:-370Py_mem_probe_v2.py}"
STATIC_ONLY="${STATIC_ONLY:-0}"
BIN="${BIN:-370Py_mem_probe_v2}"

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
  warn "sudo_permission_check" "sudo -n true failed (non-fatal; if a sweep rung crashes, run 'sudo dmesg | tail -30' by hand afterward)"
fi

# ---------------------------------------------------------------------
# 1. Presence.
# ---------------------------------------------------------------------
if [[ ! -f "$SRC" ]]; then
  fail "source_py_present" "$SRC not found in $(pwd)"
  echo "Cannot continue without source file. Aborting."
  exit 1
fi
pass "source_py_present"

# ---------------------------------------------------------------------
# 2. Docstring-stripped source copy.
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
grep -v '^VERSION_TAG:str="370' "$NODOC" > "$NOTAG"

# ---------------------------------------------------------------------
# 3. Core-region hash: reverse the single whole-function MODIFICATION,
#    and compare against 369's own FULL code region hash (369's deltas
#    from 361/365/366/368 intact -- NOT 369's core-with-its-own-deltas
#    -stripped hash).
# ---------------------------------------------------------------------
REF_HASH_369_FULL="b977905f4f40fba74de9048cdfb4c4c325462ad74f7e724b2236e4658e7b36d2"
REF_LINES_369_FULL=6101

CORE_RESULT=$(python3 - "$NOTAG" << 'PYEOF'
import sys, hashlib
path = sys.argv[1]
with open(path, encoding='utf-8') as f:
    s = f.read()

def strip_span(s, b_m, e_m):
    b = s.find(b_m); e = s.find(e_m)
    if b == -1 or e == -1:
        return None
    e_end = e + len(e_m)
    return s[:b] + s[e_end:].lstrip('\n')

NEW_FN = '''def read_vmhwm_kb()->int:
  try:
    text:str=""
    with open("/proc/self/status","r") as f:
      while True:
        chunk:str=f.read(4096)
        if chunk=="":
          break
        text+=chunk
    lines:List[str]=text.split("\\n")
    for line in lines:
      if line.startswith("VmHWM:"):
        parts:List[str]=line.split()
        if len(parts)>=2:
          return int(parts[1])
    return -1
  except:
    return -1'''

OLD_FN = '''def read_vmhwm_kb()->int:
  try:
    with open("/proc/self/status","r") as f:
      text:str=f.read()
    lines:List[str]=text.split("\\n")
    for line in lines:
      if line.startswith("VmHWM:"):
        parts:List[str]=line.split()
        if len(parts)>=2:
          return int(parts[1])
    return -1
  except:
    return -1'''

core = strip_span(s, '# ===370-VMHWM-FIX-BEGIN===', '# ===370-VMHWM-FIX-END===')
if core is None:
    print("MARKER_MISSING:370-VMHWM-FIX")
    sys.exit(0)

# strip_span removed the whole marked block (comment + NEW_FN) as a
# unit; now re-insert OLD_FN in its place so the result should equal
# 369's own full source verbatim.
b = s.find('# ===370-VMHWM-FIX-BEGIN===')
e = s.find('# ===370-VMHWM-FIX-END===')
e_end = e + len('# ===370-VMHWM-FIX-END===')
inner = s[b:e_end]
if NEW_FN not in inner:
    print("NEW_FN_NOT_FOUND")
    sys.exit(0)
cnt = s.count(NEW_FN)
if cnt != 1:
    print(f"NEW_FN_COUNT_WRONG:{cnt}")
    sys.exit(0)

core2 = s.replace(inner, OLD_FN)
core3 = strip_span(core2, '# ===370-VMHWM-FIX-BEGIN===', '# ===370-VMHWM-FIX-END===')
# core2 should no longer contain the markers at all (they were part of
# `inner`, which we replaced wholesale) -- confirm that, then hash.
if '===370-VMHWM-FIX' in core2:
    print("MARKER_RESIDUE")
    sys.exit(0)

h = hashlib.sha256(core2.encode('utf-8')).hexdigest()
lines = core2.count(chr(10))
print(f"{h} {lines}")
PYEOF
)

if [[ "$CORE_RESULT" == MARKER_MISSING:* ]]; then
  fail "source_core_identical_to_369" "insertion marker missing or malformed: ${CORE_RESULT#MARKER_MISSING:}"
elif [[ "$CORE_RESULT" == NEW_FN_NOT_FOUND ]]; then
  fail "source_core_identical_to_369" "the expected new read_vmhwm_kb() body was not found verbatim inside the 370-VMHWM-FIX span"
elif [[ "$CORE_RESULT" == NEW_FN_COUNT_WRONG:* ]]; then
  fail "source_core_identical_to_369" "the new read_vmhwm_kb() body was not found exactly once: ${CORE_RESULT#NEW_FN_COUNT_WRONG:}"
elif [[ "$CORE_RESULT" == MARKER_RESIDUE ]]; then
  fail "source_core_identical_to_369" "370-VMHWM-FIX markers still present after substitution (harness bug)"
else
  CORE_HASH=$(echo "$CORE_RESULT" | awk '{print $1}')
  CORE_LINES=$(echo "$CORE_RESULT" | awk '{print $2}')
  if [[ "$CORE_HASH" == "$REF_HASH_369_FULL" && "$CORE_LINES" -eq "$REF_LINES_369_FULL" ]]; then
    pass "source_core_identical_to_369 (hash=$CORE_HASH, lines=$CORE_LINES)"
  else
    fail "source_core_identical_to_369" "expected hash=$REF_HASH_369_FULL lines=$REF_LINES_369_FULL (369's full code, deltas intact), got hash=$CORE_HASH lines=$CORE_LINES -- code outside the marked 370 fix has drifted"
  fi
fi

# ---------------------------------------------------------------------
# 4. Targeted content checks.
# ---------------------------------------------------------------------
if grep -q 'while True:' "$NOTAG" && grep -q 'chunk:str=f.read(4096)' "$NOTAG"; then
  pass "source_chunked_read_present"
else
  fail "source_chunked_read_present" "expected chunked-read loop (chunk:str=f.read(4096)) not found"
fi

if grep -q 'text:str=f.read()$' "$NOTAG"; then
  fail "source_old_size_based_read_removed" "the old size-based f.read() (no explicit chunk size) is still present -- the bug this revision fixes would still be live"
else
  pass "source_old_size_based_read_removed"
fi

# 366/368/369's other functions must remain untouched.
if grep -q 'def check_required_maxd_for_N(N:int,fname:str,gpu_log_level:int=0)->Tuple\[int,int,int\]:' "$NOTAG"; then
  pass "source_366_original_function_still_intact"
else
  fail "source_366_original_function_still_intact" "366's check_required_maxd_for_N() signature is missing"
fi
if grep -q 'def check_required_maxd_for_N_diag(N:int,fname:str,gpu_log_level:int=0)->MaxdDiagStats:' "$NOTAG"; then
  pass "source_368_original_function_still_intact"
else
  fail "source_368_original_function_still_intact" "368's check_required_maxd_for_N_diag() signature is missing"
fi
if grep -q 'def probe_partial_load_memory(N:int,fname:str,record_limit:int,gpu_log_level:int=0)->Tuple\[int,int,int,int,int\]:' "$NOTAG"; then
  pass "source_369_probe_function_still_intact"
else
  fail "source_369_probe_function_still_intact" "369's probe_partial_load_memory() signature is missing"
fi
if grep -qE '^\s*if use_gpu and N>=21 and bench_mode==36:' "$NOTAG"; then
  pass "source_bench_mode_36_dispatch_still_present"
else
  fail "source_bench_mode_36_dispatch_still_present" "bench_mode==36 dispatch branch is missing"
fi

for marker in '# ===370-VMHWM-FIX-BEGIN===' '# ===370-VMHWM-FIX-END===' ; do
  cnt=$(grep -cF "$marker" "$NOTAG")
  if [[ "$cnt" -eq 1 ]]; then
    pass "marker_count_exactly_one[$marker]"
  else
    fail "marker_count_exactly_one[$marker]" "expected exactly 1 occurrence, found $cnt"
  fi
done

if grep -q 'VERSION_TAG:str="370' "$SRC"; then
  pass "source_version_tag_370"
else
  fail "source_version_tag_370" "VERSION_TAG for 370 not found"
fi

rm -f "$NOTAG" "$NODOC"

# ---------------------------------------------------------------------
# 5. Summary.
# ---------------------------------------------------------------------
echo ""
echo "===== 370 static-check summary ====="
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
# 6. Build once, then SWEEP record_limit. Default ladder re-confirms
#    369's two known points (1,000,000 and the last known-good
#    10,000,000) with (this time) real VmHWM numbers, then bisects the
#    10,000,000/15,000,000 gap 369 found, then re-tries the original
#    15,000,000 failure point and the full 28,719,035 for completeness.
#    Single-variable discipline: record_limit is still the only thing
#    that changes across the sweep's runs; this revision's own
#    variable (the read_vmhwm_kb fix) is fixed for the whole sweep.
#
#    RECOMMENDED: run this whole script wrapped in
#    367_safe_run_wrapper.sh, same as before.
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
RECORD_LIMITS="${RECORD_LIMITS:-1000000 10000000 11000000 12000000 13000000 14000000 15000000 20000000 28719035}"

echo "Building 370Py (codon)..."
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
echo "===== 370 sweep: record_limit ladder = $RECORD_LIMITS ====="
echo ""

declare -a SWEEP_LIMIT=()
declare -a SWEEP_VMHWM_TOTAL=()
declare -a SWEEP_DELTA_READ=()
declare -a SWEEP_DELTA_SOA=()
declare -a SWEEP_DELTA_TOTAL=()
SWEEP_FAILED_AT=""
VMHWM_STILL_BROKEN=0

for RL in $RECORD_LIMITS; do
  echo "--- rung: record_limit=$RL ---"
  CMD=("$CAND" -g "$N" "$N" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS" "$BENCH_MODE" "$REORDER_WINDOW_MULT" "$REORDER_PHASE_JUMP" "$CROSS_STRIPE_SAFE" "$RL" "$WORKER_COUNT" "$BROADMARK_VARIANT" "$CHUNKSHAPE148_BUCKET_RUN" "$CHUNKSHAPE148_ITER_SORT")
  echo "Running: ${CMD[*]}"
  PYLOG="${BIN}_N${N}_rl${RL}_$(date +%Y%m%d_%H%M%S).log"
  stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "$PYLOG"

  DONE_LINE=$(grep '^\[mem-probe-done\]' "$PYLOG" | tail -n1)
  if [[ -z "$DONE_LINE" ]]; then
    echo ""
    echo "!!! rung record_limit=$RL did NOT complete (no [mem-probe-done] line in $PYLOG)."
    echo "!!! Stopping sweep here. Run 'sudo dmesg | tail -30' now."
    SWEEP_FAILED_AT="$RL"
    break
  fi
  echo "$DONE_LINE"

  VT=$(echo "$DONE_LINE" | grep -oE 'vmhwm_after_soa_kb=-?[0-9]+' | cut -d= -f2)
  DR=$(echo "$DONE_LINE" | grep -oE 'delta_read_kb=-?[0-9]+' | cut -d= -f2)
  DS=$(echo "$DONE_LINE" | grep -oE 'delta_soa_kb=-?[0-9]+' | cut -d= -f2)
  DT=$(echo "$DONE_LINE" | grep -oE 'delta_total_kb=-?[0-9]+' | cut -d= -f2)
  if [[ "${VT:-0}" == "-1" ]]; then
    VMHWM_STILL_BROKEN=1
  fi
  SWEEP_LIMIT+=("$RL")
  SWEEP_VMHWM_TOTAL+=("${VT:-0}")
  SWEEP_DELTA_READ+=("${DR:-0}")
  SWEEP_DELTA_SOA+=("${DS:-0}")
  SWEEP_DELTA_TOTAL+=("${DT:-0}")
  echo ""
done

echo ""
echo "===== 370 sweep: results table ====="
printf "%-14s %-18s %-16s %-16s %-16s\n" "record_limit" "vmhwm_after_soa_kb" "delta_read_kb" "delta_soa_kb" "delta_total_kb"
IDX=0
while [[ $IDX -lt ${#SWEEP_LIMIT[@]} ]]; do
  printf "%-14s %-18s %-16s %-16s %-16s\n" "${SWEEP_LIMIT[$IDX]}" "${SWEEP_VMHWM_TOTAL[$IDX]}" "${SWEEP_DELTA_READ[$IDX]}" "${SWEEP_DELTA_SOA[$IDX]}" "${SWEEP_DELTA_TOTAL[$IDX]}"
  IDX=$((IDX+1))
done
echo "======================================"
echo ""

if [[ "$VMHWM_STILL_BROKEN" -eq 1 ]]; then
  fail "vmhwm_instrumentation_functional" "at least one completed rung still reported vmhwm_after_soa_kb=-1 -- the read_vmhwm_kb() fix did not resolve the issue; report this immediately, do not trust the delta_*_kb numbers above"
else
  if [[ ${#SWEEP_LIMIT[@]} -gt 0 ]]; then
    pass "vmhwm_instrumentation_functional (all completed rungs reported real, non-(-1) vmhwm values)"
  fi
fi

if [[ -n "$SWEEP_FAILED_AT" ]]; then
  fail "mem_probe_sweep_completed_fully" "sweep stopped at record_limit=$SWEEP_FAILED_AT (did not complete)"
  if [[ ${#SWEEP_LIMIT[@]} -gt 0 ]]; then
    LAST_OK_IDX=$((${#SWEEP_LIMIT[@]}-1))
    echo "Threshold bracket: last successful record_limit=${SWEEP_LIMIT[$LAST_OK_IDX]} (delta_total_kb=${SWEEP_DELTA_TOTAL[$LAST_OK_IDX]}), first failing record_limit=$SWEEP_FAILED_AT."
  fi
else
  pass "mem_probe_sweep_completed_fully (all rungs completed)"
fi

echo ""
echo "===== final summary ====="
echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
[[ "$FAIL" -gt 0 ]] && exit 1
exit 0
