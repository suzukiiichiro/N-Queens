#!/usr/bin/env bash
# 369Py_mem_probe_validate_N22_sweep.sh
#
# rev369 — DIAGNOSTIC-ONLY. Does not modify anything from 366/368;
# both remain byte-identical and reachable via bench_mode==34/35.
#
# Motivation: 366/bench_mode=34 and 368/bench_mode=35 both crashed for
# N=22 (records=28,719,035) at the IDENTICAL instruction address
# (dmesg: ip=0x412a9c, error 6), inside the shared, unmodified bin-load
# path (count_constellations_bin_records -> read_constellations_bin_
# range -> build_soa_for_range), BEFORE ever reaching
# schedule_depth_for_task(_diag). So 368's meta_next out-of-bounds
# hypothesis was never actually exercised. Raising 367's
# MAX_MEM_PERCENT from 70 to 95 (10GB -> 14GB ulimit ceiling on this
# session's swapless 15GB-RAM host) made no difference: same crash,
# same address.
#
# 369 instruments that shared load path with /proc/self/status VmHWM
# (peak RSS) checkpoints (after count, after the Dict-list read, after
# SoA build) and exposes the loaded record count as a CLI-controlled
# variable (bench_mode==36, record_limit via the same argv[13] slot
# bench_mode==30 uses for debug_chunk_start). This script then SWEEPS
# record_limit across a fixed ladder, in increasing order, one binary
# invocation per rung, stopping at the first rung that does not
# complete -- so the sweep itself brackets the actual memory
# threshold on real N=22 data, without guessing further.
#
# 369Py's delta versus 368Py's executable code region is six clearly
# delimited spans (extending the 361/365/366/368 pattern):
#   1. ===369-VARINIT-BEGIN/END===       new record_limit_arg default,
#      pure insertion (1 line).
#   2. ===369-CLIGATE-COMMENT-BEGIN/END===   comment (pure insertion)
#      above a 2-word MODIFICATION ("or bench_mode==36") to the CLI
#      whitelist gate -- applied preemptively per the 361-r1 lesson.
#   3. ===369-PRESETGATE-COMMENT-BEGIN/END===   same pattern for the
#      adjacent preset_queens gating condition.
#   4. ===369-ARGPARSE-INSERT-BEGIN/END===   new bench_mode==36 argv
#      parsing block, pure insertion.
#   5. ===369-INSERT-BEGIN/END===        new functions read_vmhwm_kb()
#      + probe_partial_load_memory(), pure insertion.
#   6. ===369-DISPATCH-INSERT-BEGIN/END===   new bench_mode==36
#      dispatch branch, pure insertion.
#
# IMPORTANT: the reference this script checks 369's core against is
# 368's OWN FULL code region (368's deltas from 361/365/366 intact,
# NOT stripped) -- 369 is built on top of 368, so 368's additions must
# remain byte-for-byte present, not be treated as removable.

set -u
SRC="${SRC:-369Py_mem_probe.py}"
STATIC_ONLY="${STATIC_ONLY:-0}"
BIN="${BIN:-369Py_mem_probe}"

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
grep -v '^VERSION_TAG:str="369' "$NODOC" > "$NOTAG"

# ---------------------------------------------------------------------
# 3. Core-region hash: strip the six pure-insertion spans, reverse the
#    two targeted 2-word CLI-gate modifications, and compare against
#    368's own FULL code region hash (368's deltas from 361/365/366
#    intact -- NOT 368's core-with-its-own-deltas-stripped hash).
# ---------------------------------------------------------------------
REF_HASH_368_FULL="362bb7fad3c026a625e7bb276c9b2e9b2de6ed0af29dda1934980890c0b85454"
REF_LINES_368_FULL=5997

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

core = s
for b_m, e_m, label in [
    ('  # ===369-VARINIT-BEGIN===', '  # ===369-VARINIT-END===', '369-VARINIT'),
    ('      # ===369-CLIGATE-COMMENT-BEGIN===', '      # ===369-CLIGATE-COMMENT-END===', '369-CLIGATE-COMMENT'),
    ('    # ===369-PRESETGATE-COMMENT-BEGIN===', '    # ===369-PRESETGATE-COMMENT-END===', '369-PRESETGATE-COMMENT'),
    ('    # ===369-ARGPARSE-INSERT-BEGIN===', '    # ===369-ARGPARSE-INSERT-END===', '369-ARGPARSE-INSERT'),
    ('# ===369-INSERT-BEGIN===', '# ===369-INSERT-END===', '369-INSERT'),
    ('    # ===369-DISPATCH-INSERT-BEGIN===', '    # ===369-DISPATCH-INSERT-END===', '369-DISPATCH-INSERT'),
]:
    core2 = strip_span(core, b_m, e_m)
    if core2 is None:
        print(f"MARKER_MISSING:{label}")
        sys.exit(0)
    core = core2

mods = [
    ("if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34 or bench_mode==35 or bench_mode==36):",
     "if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34 or bench_mode==35):",
     "369-CLIGATE-LINE"),
    ("if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34 or bench_mode==35 or bench_mode==36:",
     "if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34 or bench_mode==35:",
     "369-PRESETGATE-LINE"),
]
for modified, original, label in mods:
    cnt = core.count(modified)
    if cnt != 1:
        print(f"MOD_LINE_COUNT_WRONG:{label}:{cnt}")
        sys.exit(0)
    core = core.replace(modified, original)

h = hashlib.sha256(core.encode('utf-8')).hexdigest()
lines = core.count('\n')
print(f"{h} {lines}")
PYEOF
)

if [[ "$CORE_RESULT" == MARKER_MISSING:* ]]; then
  fail "source_core_identical_to_368" "insertion marker missing or malformed: ${CORE_RESULT#MARKER_MISSING:}"
elif [[ "$CORE_RESULT" == MOD_LINE_COUNT_WRONG:* ]]; then
  fail "source_core_identical_to_368" "targeted 2-word modification line not found exactly once: ${CORE_RESULT#MOD_LINE_COUNT_WRONG:}"
else
  CORE_HASH=$(echo "$CORE_RESULT" | awk '{print $1}')
  CORE_LINES=$(echo "$CORE_RESULT" | awk '{print $2}')
  if [[ "$CORE_HASH" == "$REF_HASH_368_FULL" && "$CORE_LINES" -eq "$REF_LINES_368_FULL" ]]; then
    pass "source_core_identical_to_368 (hash=$CORE_HASH, lines=$CORE_LINES)"
  else
    fail "source_core_identical_to_368" "expected hash=$REF_HASH_368_FULL lines=$REF_LINES_368_FULL (368's full code, deltas intact), got hash=$CORE_HASH lines=$CORE_LINES -- code outside the marked 369 deltas has drifted"
  fi
fi

# ---------------------------------------------------------------------
# 4. Targeted content checks.
# ---------------------------------------------------------------------
if grep -q 'def read_vmhwm_kb()->int:' "$NOTAG"; then
  pass "source_read_vmhwm_function_present"
else
  fail "source_read_vmhwm_function_present" "read_vmhwm_kb() not found"
fi

if grep -q 'def probe_partial_load_memory(N:int,fname:str,record_limit:int,gpu_log_level:int=0)->Tuple\[int,int,int,int,int\]:' "$NOTAG"; then
  pass "source_probe_partial_load_function_present"
else
  fail "source_probe_partial_load_function_present" "probe_partial_load_memory() signature not found or changed"
fi

# 366/368's original functions must remain untouched.
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

if grep -qE '^\s*if use_gpu and N>=21 and bench_mode==36:' "$NOTAG"; then
  pass "source_bench_mode_36_dispatch_present"
else
  fail "source_bench_mode_36_dispatch_present" "bench_mode==36 dispatch branch not found"
fi

if grep -qE '^\s*if not \(bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34 or bench_mode==35 or bench_mode==36\):' "$NOTAG"; then
  pass "source_cligate_whitelist_includes_36"
else
  fail "source_cligate_whitelist_includes_36" "the bench_mode CLI whitelist does not include bench_mode==36 -- this is the exact 361-r1 bug class"
fi

if grep -qE '^\s*if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34 or bench_mode==35 or bench_mode==36:' "$NOTAG"; then
  pass "source_presetgate_includes_36"
else
  fail "source_presetgate_includes_36" "the preset_queens gating condition does not include bench_mode==36"
fi

if grep -qE '^\s*if bench_mode==36:' "$NOTAG"; then
  pass "source_argparse_bench_mode_36_present"
else
  fail "source_argparse_bench_mode_36_present" "the bench_mode==36 argv-parsing block for record_limit_arg was not found"
fi

# No kernel launch, no GPU upload primitives: this mode must stay a
# CPU-side, read-only memory probe.
DISPATCH_BLOCK=$(sed -n '/# ===369-DISPATCH-INSERT-BEGIN===/,/# ===369-DISPATCH-INSERT-END===/p' "$NOTAG")
if echo "$DISPATCH_BLOCK" | grep -q 'launch_kernel_dfs_iter_gpu_static_maxd'; then
  fail "source_no_kernel_launch_in_mem_probe" "the mem-probe dispatch branch appears to call the kernel launch primitive -- this mode is supposed to be launch-free"
else
  pass "source_no_kernel_launch_in_mem_probe"
fi

for marker in '  # ===369-VARINIT-BEGIN===' '  # ===369-VARINIT-END===' '      # ===369-CLIGATE-COMMENT-BEGIN===' '      # ===369-CLIGATE-COMMENT-END===' '    # ===369-PRESETGATE-COMMENT-BEGIN===' '    # ===369-PRESETGATE-COMMENT-END===' '    # ===369-ARGPARSE-INSERT-BEGIN===' '    # ===369-ARGPARSE-INSERT-END===' '# ===369-INSERT-BEGIN===' '# ===369-INSERT-END===' '    # ===369-DISPATCH-INSERT-BEGIN===' '    # ===369-DISPATCH-INSERT-END===' ; do
  cnt=$(grep -cF "$marker" "$NOTAG")
  if [[ "$cnt" -eq 1 ]]; then
    pass "marker_count_exactly_one[$marker]"
  else
    fail "marker_count_exactly_one[$marker]" "expected exactly 1 occurrence, found $cnt"
  fi
done

if grep -q 'VERSION_TAG:str="369' "$SRC"; then
  pass "source_version_tag_369"
else
  fail "source_version_tag_369" "VERSION_TAG for 369 not found"
fi

# ---------------------------------------------------------------------
# 5. Negative tests.
# ---------------------------------------------------------------------
NEGTMP=$(mktemp)
cp "$NOTAG" "$NEGTMP"
sed -i 's/def auto_sort_mode(N:int)->int:/def auto_sort_mode(N:int)->int:  # tampered/' "$NEGTMP"
NEG_RESULT=$(python3 - "$NEGTMP" << 'PYEOF'
import sys, hashlib
path = sys.argv[1]
with open(path, encoding='utf-8') as f:
    s = f.read()
def strip_span(s, b_m, e_m):
    b = s.find(b_m); e = s.find(e_m)
    if b == -1 or e == -1: return None
    e_end = e + len(e_m)
    return s[:b] + s[e_end:].lstrip('\n')
core = s
for b_m, e_m in [
    ('  # ===369-VARINIT-BEGIN===', '  # ===369-VARINIT-END==='),
    ('      # ===369-CLIGATE-COMMENT-BEGIN===', '      # ===369-CLIGATE-COMMENT-END==='),
    ('    # ===369-PRESETGATE-COMMENT-BEGIN===', '    # ===369-PRESETGATE-COMMENT-END==='),
    ('    # ===369-ARGPARSE-INSERT-BEGIN===', '    # ===369-ARGPARSE-INSERT-END==='),
    ('# ===369-INSERT-BEGIN===', '# ===369-INSERT-END==='),
    ('    # ===369-DISPATCH-INSERT-BEGIN===', '    # ===369-DISPATCH-INSERT-END==='),
]:
    core = strip_span(core, b_m, e_m)
mods = [
    ("if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34 or bench_mode==35 or bench_mode==36):",
     "if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34 or bench_mode==35):"),
    ("if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34 or bench_mode==35 or bench_mode==36:",
     "if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34 or bench_mode==35:"),
]
for modified, original in mods:
    core = core.replace(modified, original)
print(hashlib.sha256(core.encode('utf-8')).hexdigest())
PYEOF
)
if [[ "$NEG_RESULT" != "$REF_HASH_368_FULL" ]]; then
  pass "negtest_core_tamper_detected"
else
  fail "negtest_core_tamper_detected" "tampering did not change the hash (test harness bug)"
fi
rm -f "$NEGTMP"

BUG_REPRO=$(python3 -c "
bench_mode = 36
if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34 or bench_mode==35):
    bench_mode = 0
print(bench_mode)
")
if [[ "$BUG_REPRO" == "0" ]]; then
  pass "negtest_361_style_bug_reproduced (confirms the PRE-369 whitelist line would have silently rejected bench_mode=36)"
else
  fail "negtest_361_style_bug_reproduced" "reverting to the pre-369 whitelist line did not reproduce the expected rejection (test harness bug)"
fi

rm -f "$NOTAG" "$NODOC"

# ---------------------------------------------------------------------
# 6. Summary.
# ---------------------------------------------------------------------
echo ""
echo "===== 369 static-check summary ====="
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
# 7. Build once, then SWEEP record_limit across a fixed ladder for
#    N=22, one binary invocation per rung, increasing order, stopping
#    at the first rung that does not complete. Single-variable
#    discipline: record_limit is the only thing that changes across
#    the sweep's runs.
#
#    RECOMMENDED: run this whole script wrapped in
#    367_safe_run_wrapper.sh, same as 366/368 -- bin generation (if
#    needed) and each sweep rung remain resource-heavy for N=22.
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
WORKER_ID="${WORKER_ID:-0}"
WORKER_COUNT="${WORKER_COUNT:-1}"
BROADMARK_VARIANT="${BROADMARK_VARIANT:-2}"
CHUNKSHAPE148_BUCKET_RUN="${CHUNKSHAPE148_BUCKET_RUN:-2048}"
CHUNKSHAPE148_ITER_SORT="${CHUNKSHAPE148_ITER_SORT:-9}"
# Sweep ladder: record_limit values, increasing. 28719035 is N=22's
# actual full record count (the value that has crashed twice so far).
# Override with RECORD_LIMITS="1000000 2000000 ..." (space-separated)
# if a finer or coarser ladder is wanted.
RECORD_LIMITS="${RECORD_LIMITS:-1000000 5000000 10000000 15000000 20000000 25000000 28719035}"

echo "Building 369Py (codon)..."
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
echo "===== 369 sweep: record_limit ladder = $RECORD_LIMITS ====="
echo ""

declare -a SWEEP_LIMIT=()
declare -a SWEEP_DELTA_READ=()
declare -a SWEEP_DELTA_SOA=()
declare -a SWEEP_DELTA_TOTAL=()
SWEEP_FAILED_AT=""

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
    echo "!!! Stopping sweep here -- this rung brackets the memory threshold together"
    echo "!!! with the last successful rung below. Run 'sudo dmesg | tail -30' now."
    SWEEP_FAILED_AT="$RL"
    break
  fi
  echo "$DONE_LINE"

  DR=$(echo "$DONE_LINE" | grep -oE 'delta_read_kb=-?[0-9]+' | cut -d= -f2)
  DS=$(echo "$DONE_LINE" | grep -oE 'delta_soa_kb=-?[0-9]+' | cut -d= -f2)
  DT=$(echo "$DONE_LINE" | grep -oE 'delta_total_kb=-?[0-9]+' | cut -d= -f2)
  SWEEP_LIMIT+=("$RL")
  SWEEP_DELTA_READ+=("${DR:-0}")
  SWEEP_DELTA_SOA+=("${DS:-0}")
  SWEEP_DELTA_TOTAL+=("${DT:-0}")
  echo ""
done

echo ""
echo "===== 369 sweep: results table ====="
printf "%-14s %-16s %-16s %-16s\n" "record_limit" "delta_read_kb" "delta_soa_kb" "delta_total_kb"
IDX=0
while [[ $IDX -lt ${#SWEEP_LIMIT[@]} ]]; do
  printf "%-14s %-16s %-16s %-16s\n" "${SWEEP_LIMIT[$IDX]}" "${SWEEP_DELTA_READ[$IDX]}" "${SWEEP_DELTA_SOA[$IDX]}" "${SWEEP_DELTA_TOTAL[$IDX]}"
  IDX=$((IDX+1))
done
echo "======================================"
echo ""

if [[ -n "$SWEEP_FAILED_AT" ]]; then
  fail "mem_probe_sweep_completed_fully" "sweep stopped at record_limit=$SWEEP_FAILED_AT (did not complete)"
  if [[ ${#SWEEP_LIMIT[@]} -gt 0 ]]; then
    LAST_OK_IDX=$((${#SWEEP_LIMIT[@]}-1))
    echo "Threshold bracket: last successful record_limit=${SWEEP_LIMIT[$LAST_OK_IDX]} (delta_total_kb=${SWEEP_DELTA_TOTAL[$LAST_OK_IDX]}), first failing record_limit=$SWEEP_FAILED_AT."
    if [[ "${SWEEP_LIMIT[$LAST_OK_IDX]}" -gt 0 ]]; then
      KB_PER_RECORD=$(python3 -c "print(${SWEEP_DELTA_TOTAL[$LAST_OK_IDX]}/${SWEEP_LIMIT[$LAST_OK_IDX]})")
      EXTRAP_KB=$(python3 -c "print(int($KB_PER_RECORD*28719035))")
      echo "Rough linear extrapolation from the last successful rung: ~${KB_PER_RECORD} KB/record -> ~${EXTRAP_KB} KB (~$((EXTRAP_KB/1024/1024)) GB) for the full 28,719,035 records. This is a rough estimate (Dict/list overhead is not guaranteed linear); treat as an order-of-magnitude guide only."
    fi
  else
    echo "Sweep failed at the very first rung ($SWEEP_FAILED_AT records) -- even the smallest tested scale did not complete. This changes the picture significantly; report immediately rather than trying larger MAX_MEM_PERCENT values."
  fi
else
  pass "mem_probe_sweep_completed_fully (all rungs including the full 28,719,035-record count completed)"
  echo "The full N=22 record count completed under bench_mode=36's read-only probe."
  echo "If 366/368's kernel-launch modes still crash at the same address, the cause"
  echo "is likely NOT the bin-load path itself but something further along (e.g. the"
  echo "kernel-launch/dispatch machinery, SoA-to-GPU staging, or interaction with the"
  echo "GPU driver) -- next revision should instrument past this point, not repeat"
  echo "this one's checkpoints."
fi

echo ""
echo "===== final summary ====="
echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
[[ "$FAIL" -gt 0 ]] && exit 1
exit 0
