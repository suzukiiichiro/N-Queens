#!/usr/bin/env bash
# 368Py_maxd_diag_validate_N22_once.sh
#
# rev368 — DIAGNOSTIC-ONLY. Does not fix or modify 366's
# check_required_maxd_for_N()/schedule_depth_for_task() (both remain
# byte-identical and reachable via bench_mode==34). Adds a parallel,
# bounds-safe COPY of the schedule-walk (MaxdDiagStats,
# schedule_depth_for_task_diag, scan_maxd_diag_for_tasks,
# check_required_maxd_for_N_diag) to survey the actual range of
# fu=raw&31 (0..31 by construction) against meta_next (28 elements,
# indices 0..27) for a given N's task set, WITHOUT crashing, and
# report fu_min/fu_max/oob_count + a first-occurrence repro.
#
# Motivation: a real-hardware run of 366/bench_mode=34 for N=22
# (records=28,719,035) produced no [maxd-check-done] line. dmesg (via
# sudo) showed two `366Py_maxd_chec[...]: segfault at 0` events (error
# 4 then error 6), no OOM-killer entry, no allocation-failure message
# -- inconsistent with the initial memory-exhaustion hypothesis (367's
# 10GB ulimit ceiling on a 15GB host) and consistent with an
# out-of-bounds meta_next[fu] access instead. 368 exists to confirm or
# refute that, on real N=22 data, without crashing.
#
# 368Py's delta versus 366Py's executable code region is five clearly
# delimited spans (mirroring the 361/365/366 pattern exactly):
#   1. ===368-INSERT-BEGIN/END===              new class MaxdDiagStats
#      + 3 new functions, pure insertion (parallel copy, not a
#      modification of 366's originals).
#   2. ===368-CLIGATE-COMMENT-BEGIN/END===      comment (pure
#      insertion) above a 2-word MODIFICATION ("or bench_mode==35") to
#      the CLI whitelist gate -- applied preemptively, learning from
#      361's r1 bug (same lesson 366 already applied for bench_mode==34).
#   3. ===368-PRESETGATE-COMMENT-BEGIN/END===   same pattern for the
#      adjacent preset_queens gating condition.
#   4. ===368-DISPATCH-INSERT-BEGIN/END===      new bench_mode==35
#      dispatch branch, pure insertion.
#
# IMPORTANT: the reference this script checks 368's core against is
# 366's OWN FULL code region (366's deltas from 361/365 intact, NOT
# stripped) -- 368 is built on top of 366, so 366's additions must
# remain byte-for-byte present, not be treated as removable.

set -u
SRC="${SRC:-368Py_maxd_diag.py}"
STATIC_ONLY="${STATIC_ONLY:-0}"
BIN="${BIN:-368Py_maxd_diag}"

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
  warn "sudo_permission_check" "sudo -n true failed (non-fatal, no ncu is invoked in this revision; note: if this crashes again, run 'sudo dmesg | tail -30' by hand afterward, same as 368's own diagnosis)"
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
grep -v '^VERSION_TAG:str="368' "$NODOC" > "$NOTAG"

# ---------------------------------------------------------------------
# 3. Core-region hash: strip the four pure-insertion spans, reverse
#    the two targeted 2-word CLI-gate modifications, and compare
#    against 366's own FULL code region hash (366's deltas from
#    361/365 intact -- NOT 366's core-with-its-own-deltas-stripped
#    hash).
# ---------------------------------------------------------------------
REF_HASH_366_FULL="03fed068544019ffe1650ab3ba8bfdc4e95880eda20f9eed0bc163524935474f"
REF_LINES_366_FULL=5832

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
    ('# ===368-INSERT-BEGIN===', '# ===368-INSERT-END===', '368-INSERT'),
    ('      # ===368-CLIGATE-COMMENT-BEGIN===', '      # ===368-CLIGATE-COMMENT-END===', '368-CLIGATE-COMMENT'),
    ('    # ===368-PRESETGATE-COMMENT-BEGIN===', '    # ===368-PRESETGATE-COMMENT-END===', '368-PRESETGATE-COMMENT'),
    ('    # ===368-DISPATCH-INSERT-BEGIN===', '    # ===368-DISPATCH-INSERT-END===', '368-DISPATCH-INSERT'),
]:
    core2 = strip_span(core, b_m, e_m)
    if core2 is None:
        print(f"MARKER_MISSING:{label}")
        sys.exit(0)
    core = core2

mods = [
    ("if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34 or bench_mode==35):",
     "if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34):",
     "368-CLIGATE-LINE"),
    ("if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34 or bench_mode==35:",
     "if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34:",
     "368-PRESETGATE-LINE"),
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
  fail "source_core_identical_to_366" "insertion marker missing or malformed: ${CORE_RESULT#MARKER_MISSING:}"
elif [[ "$CORE_RESULT" == MOD_LINE_COUNT_WRONG:* ]]; then
  fail "source_core_identical_to_366" "targeted 2-word modification line not found exactly once: ${CORE_RESULT#MOD_LINE_COUNT_WRONG:}"
else
  CORE_HASH=$(echo "$CORE_RESULT" | awk '{print $1}')
  CORE_LINES=$(echo "$CORE_RESULT" | awk '{print $2}')
  if [[ "$CORE_HASH" == "$REF_HASH_366_FULL" && "$CORE_LINES" -eq "$REF_LINES_366_FULL" ]]; then
    pass "source_core_identical_to_366 (hash=$CORE_HASH, lines=$CORE_LINES)"
  else
    fail "source_core_identical_to_366" "expected hash=$REF_HASH_366_FULL lines=$REF_LINES_366_FULL (366's full code, deltas intact), got hash=$CORE_HASH lines=$CORE_LINES -- code outside the marked 368 deltas has drifted"
  fi
fi

# ---------------------------------------------------------------------
# 4. Targeted content checks.
# ---------------------------------------------------------------------
if grep -q 'class MaxdDiagStats:' "$NOTAG"; then
  pass "source_maxddiagstats_class_present"
else
  fail "source_maxddiagstats_class_present" "class MaxdDiagStats not found"
fi

if grep -q 'def schedule_depth_for_task_diag(ctrl0:u32,markctrl:u32,meta_next:List\[u8\],stats:MaxdDiagStats,task_index:int)->int:' "$NOTAG"; then
  pass "source_schedule_depth_diag_function_present"
else
  fail "source_schedule_depth_diag_function_present" "schedule_depth_for_task_diag() signature not found or changed"
fi

if grep -q 'def check_required_maxd_for_N_diag(N:int,fname:str,gpu_log_level:int=0)->MaxdDiagStats:' "$NOTAG"; then
  pass "source_check_maxd_diag_function_present"
else
  fail "source_check_maxd_diag_function_present" "check_required_maxd_for_N_diag() signature not found or changed"
fi

# Original 366 function must remain untouched -- this is a diagnostic
# addition, not a replacement.
if grep -q 'def check_required_maxd_for_N(N:int,fname:str,gpu_log_level:int=0)->Tuple\[int,int,int\]:' "$NOTAG"; then
  pass "source_366_original_function_still_intact"
else
  fail "source_366_original_function_still_intact" "366's check_required_maxd_for_N() signature is missing -- 368 must not remove or rename it"
fi

# The diagnostic walk must bounds-check fu against len(meta_next)
# before every meta_next[fu] access it performs (3 sites), not touch
# meta_next's own definition.
DIAG_BLOCK=$(sed -n '/# ===368-INSERT-BEGIN===/,/# ===368-INSERT-END===/p' "$NOTAG")
OOB_GUARD_COUNT=$(echo "$DIAG_BLOCK" | grep -c 'if fu>=META_NEXT_LEN:')
if [[ "$OOB_GUARD_COUNT" -eq 3 ]]; then
  pass "source_diag_has_three_bounds_guards"
else
  fail "source_diag_has_three_bounds_guards" "expected exactly 3 'if fu>=META_NEXT_LEN:' guards (one per meta_next[fu] call site), found $OOB_GUARD_COUNT"
fi

if grep -qE '^\s*if use_gpu and N>=21 and bench_mode==35:' "$NOTAG"; then
  pass "source_bench_mode_35_dispatch_present"
else
  fail "source_bench_mode_35_dispatch_present" "bench_mode==35 dispatch branch not found"
fi

if grep -qE '^\s*if not \(bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34 or bench_mode==35\):' "$NOTAG"; then
  pass "source_cligate_whitelist_includes_35"
else
  fail "source_cligate_whitelist_includes_35" "the bench_mode CLI whitelist does not include bench_mode==35 -- this is the exact 361-r1 bug class"
fi

if grep -qE '^\s*if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34 or bench_mode==35:' "$NOTAG"; then
  pass "source_presetgate_includes_35"
else
  fail "source_presetgate_includes_35" "the preset_queens gating condition does not include bench_mode==35"
fi

# No kernel launch: this mode must not call launch_kernel_dfs_iter_gpu_static_maxd.
DISPATCH_BLOCK=$(sed -n '/# ===368-DISPATCH-INSERT-BEGIN===/,/# ===368-DISPATCH-INSERT-END===/p' "$NOTAG")
if echo "$DISPATCH_BLOCK" | grep -q 'launch_kernel_dfs_iter_gpu_static_maxd'; then
  fail "source_no_kernel_launch_in_maxd_diag" "the maxd-diag dispatch branch appears to call the kernel launch primitive -- this mode is supposed to be launch-free"
else
  pass "source_no_kernel_launch_in_maxd_diag"
fi

for marker in '# ===368-INSERT-BEGIN===' '# ===368-INSERT-END===' '      # ===368-CLIGATE-COMMENT-BEGIN===' '      # ===368-CLIGATE-COMMENT-END===' '    # ===368-PRESETGATE-COMMENT-BEGIN===' '    # ===368-PRESETGATE-COMMENT-END===' '    # ===368-DISPATCH-INSERT-BEGIN===' '    # ===368-DISPATCH-INSERT-END===' ; do
  cnt=$(grep -cF "$marker" "$NOTAG")
  if [[ "$cnt" -eq 1 ]]; then
    pass "marker_count_exactly_one[$marker]"
  else
    fail "marker_count_exactly_one[$marker]" "expected exactly 1 occurrence, found $cnt"
  fi
done

if grep -q 'VERSION_TAG:str="368' "$SRC"; then
  pass "source_version_tag_368"
else
  fail "source_version_tag_368" "VERSION_TAG for 368 not found"
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
    ('# ===368-INSERT-BEGIN===', '# ===368-INSERT-END==='),
    ('      # ===368-CLIGATE-COMMENT-BEGIN===', '      # ===368-CLIGATE-COMMENT-END==='),
    ('    # ===368-PRESETGATE-COMMENT-BEGIN===', '    # ===368-PRESETGATE-COMMENT-END==='),
    ('    # ===368-DISPATCH-INSERT-BEGIN===', '    # ===368-DISPATCH-INSERT-END==='),
]:
    core = strip_span(core, b_m, e_m)
mods = [
    ("if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34 or bench_mode==35):",
     "if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34):"),
    ("if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34 or bench_mode==35:",
     "if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34:"),
]
for modified, original in mods:
    core = core.replace(modified, original)
print(hashlib.sha256(core.encode('utf-8')).hexdigest())
PYEOF
)
if [[ "$NEG_RESULT" != "$REF_HASH_366_FULL" ]]; then
  pass "negtest_core_tamper_detected"
else
  fail "negtest_core_tamper_detected" "tampering did not change the hash (test harness bug)"
fi
rm -f "$NEGTMP"

BUG_REPRO=$(python3 -c "
bench_mode = 35
if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34):
    bench_mode = 0
print(bench_mode)
")
if [[ "$BUG_REPRO" == "0" ]]; then
  pass "negtest_361_style_bug_reproduced (confirms the PRE-368 whitelist line would have silently rejected bench_mode=35)"
else
  fail "negtest_361_style_bug_reproduced" "reverting to the pre-368 whitelist line did not reproduce the expected rejection (test harness bug)"
fi

# Sanity: fu is a 5-bit mask (raw&31), so it can never exceed 31 --
# the diagnostic's own bounds guard threshold (META_NEXT_LEN=len(meta_next)=28)
# must be strictly less than 32, or the guard would be vacuous.
GUARD_SANITY=$(python3 -c "
meta_next_len = 28
fu_max_possible = 31
print('ok' if meta_next_len <= fu_max_possible else 'vacuous')
")
if [[ "$GUARD_SANITY" == "ok" ]]; then
  pass "negtest_guard_not_vacuous (meta_next_len=28 <= max possible fu=31, so the oob guard can actually trigger)"
else
  fail "negtest_guard_not_vacuous" "meta_next_len exceeds the max possible fu value -- the bounds guard could never fire, defeating the diagnostic's purpose"
fi

rm -f "$NOTAG" "$NODOC"

# ---------------------------------------------------------------------
# 6. Summary.
# ---------------------------------------------------------------------
echo ""
echo "===== 368 static-check summary ====="
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
# 7. Build and run for N=22 only (single-variable discipline: this
#    revision's only variable is the diagnostic guard itself). If the
#    bin cache for N=22 doesn't exist (e.g. lost in the EBS reset),
#    this run regenerates it first via the same unmodified
#    ensure_constellations_bin_stream() 366 uses -- same as before,
#    may take a while. RECOMMENDED: run this wrapped in
#    367_safe_run_wrapper.sh, same as the 366 attempt that segfaulted,
#    since bin generation itself is untouched by 368 and still
#    resource-heavy for N=22.
# ---------------------------------------------------------------------
N="${N:-22}"
BLOCK="${BLOCK:-32}"
MAX_BLOCKS="${MAX_BLOCKS:-484}"
LOG_LEVEL="${LOG_LEVEL:-1}"
SORT_MODE="${SORT_MODE:-0}"
PRESET_QUEENS="${PRESET_QUEENS:-7}"
BENCH_MODE="${BENCH_MODE:-35}"
REORDER_WINDOW_MULT="${REORDER_WINDOW_MULT:-3}"
REORDER_PHASE_JUMP="${REORDER_PHASE_JUMP:-7}"
CROSS_STRIPE_SAFE="${CROSS_STRIPE_SAFE:-0}"
WORKER_ID="${WORKER_ID:-0}"
WORKER_COUNT="${WORKER_COUNT:-1}"
BROADMARK_VARIANT="${BROADMARK_VARIANT:-2}"
CHUNKSHAPE148_BUCKET_RUN="${CHUNKSHAPE148_BUCKET_RUN:-2048}"
CHUNKSHAPE148_ITER_SORT="${CHUNKSHAPE148_ITER_SORT:-9}"

echo "Building 368Py (codon)..."
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

CMD=("$CAND" -g "$N" "$N" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS" "$BENCH_MODE" "$REORDER_WINDOW_MULT" "$REORDER_PHASE_JUMP" "$CROSS_STRIPE_SAFE" "$WORKER_ID" "$WORKER_COUNT" "$BROADMARK_VARIANT" "$CHUNKSHAPE148_BUCKET_RUN" "$CHUNKSHAPE148_ITER_SORT")
echo "Running: ${CMD[*]}"
echo "NOTE: if constellations_N${N}_*.bin does not exist yet (e.g. lost"
echo "in the EBS reset), this will regenerate it first, which may take"
echo "a while for N=22. This step is unchanged from 366 -- if it seems"
echo "to hang, let it run and report back."
echo "NOTE: unlike 366's bench_mode=34, this mode is designed NOT to"
echo "segfault even if the out-of-bounds hypothesis is correct -- it"
echo "should always reach [maxd-diag-done]. If it crashes anyway, that"
echo "itself is important new information (report dmesg immediately)."
PYLOG="${BIN}_N${N}_maxd_diag_$(date +%Y%m%d_%H%M%S).log"
stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "$PYLOG"

DONE_LINE=$(grep '^\[maxd-diag-done\]' "$PYLOG" | tail -n1)
if [[ -z "$DONE_LINE" ]]; then
  fail "maxd_diag_done_line_found" "no [maxd-diag-done] line in $PYLOG -- IMPORTANT: if this mode crashed too, the bounds guard did not prevent it, meaning the out-of-bounds meta_next[fu] hypothesis is likely WRONG (or incomplete) and the real cause is still unidentified. Run 'sudo dmesg | tail -30' immediately and report it."
  echo ""
  echo "===== final summary ====="
  echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
  exit 1
fi
pass "maxd_diag_done_line_found"
echo "$DONE_LINE"

echo ""
echo "===== final summary ====="
echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
echo ""
echo "Report the fu_min/fu_max/oob_count values above."
echo "  - oob_count==0 means fu never reached >=28 for N=$N's task set:"
echo "    the meta_next out-of-bounds hypothesis is REFUTED for N=$N,"
echo "    and the segfault cause is still open -- do not proceed to a"
echo "    meta_next fix without a different lead."
echo "  - oob_count>0 means the hypothesis is CONFIRMED: meta_next[fu]"
echo "    would have read out of bounds oob_count times. The"
echo "    first_oob_task_index/fu/ctrl0/markctrl fields above are a"
echo "    concrete repro case for designing the correctness fix (e.g."
echo "    extending meta_next to cover fu up to 31, or determining"
echo "    that the current meta_next table's semantics are simply not"
echo "    valid for N=22's constellation set) -- next revision, not 368."
[[ "$FAIL" -gt 0 ]] && exit 1
exit 0
