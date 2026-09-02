#!/usr/bin/env bash
# 366Py_maxd_check_validate_N22_once.sh
#
# rev366 — cheap, kernel-launch-free check of the required schedule
# depth (maxd) for N=22 onward, before committing to porting more
# kernels (maxd16/18/20/21) to C. Reuses existing unmodified functions
# only (ensure_constellations_bin_stream, build_soa_for_range,
# max_schedule_depth_of_tasks, select_static_maxd) -- no new
# algorithmic code, no GPU upload, no kernel launch.
#
# 366Py's delta versus 365Py's executable code region is five clearly
# delimited spans (mirroring the 361/365 pattern exactly):
#   1. ===366-INSERT-BEGIN/END===              new function
#      check_required_maxd_for_N(), pure insertion.
#   2. ===366-CLIGATE-COMMENT-BEGIN/END===      comment (pure
#      insertion) above a 2-word MODIFICATION ("or bench_mode==34") to
#      the CLI whitelist gate -- applied preemptively, learning from
#      361's r1 bug.
#   3. ===366-PRESETGATE-COMMENT-BEGIN/END===   same pattern for the
#      adjacent preset_queens gating condition.
#   4. ===366-DISPATCH-INSERT-BEGIN/END===      new bench_mode==34
#      dispatch branch, pure insertion.
#
# IMPORTANT: the reference this script checks 366's core against is
# 365's OWN FULL code region (365's deltas from 361 intact, NOT
# stripped) -- 366 is built on top of 365, so 365's additions must
# remain byte-for-byte present, not be treated as removable.

set -u
SRC="${SRC:-366Py_maxd_check.py}"
STATIC_ONLY="${STATIC_ONLY:-0}"
BIN="${BIN:-366Py_maxd_check}"

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
  warn "sudo_permission_check" "sudo -n true failed (non-fatal, no ncu is invoked in this revision)"
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
grep -v '^VERSION_TAG:str="366' "$NODOC" > "$NOTAG"

# ---------------------------------------------------------------------
# 3. Core-region hash: strip the four pure-insertion spans, reverse
#    the two targeted 2-word CLI-gate modifications, and compare
#    against 365's own FULL code region hash (365's deltas from 361
#    intact -- NOT 365's core-with-its-own-deltas-stripped hash).
# ---------------------------------------------------------------------
REF_HASH_365_FULL="701a7f0ac38fc6ca5ba639a3530ef011420ed99831ec31f6cd3a2b1fd9e20b57"
REF_LINES_365_FULL=5769

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
    ('# ===366-INSERT-BEGIN===', '# ===366-INSERT-END===', '366-INSERT'),
    ('      # ===366-CLIGATE-COMMENT-BEGIN===', '      # ===366-CLIGATE-COMMENT-END===', '366-CLIGATE-COMMENT'),
    ('    # ===366-PRESETGATE-COMMENT-BEGIN===', '    # ===366-PRESETGATE-COMMENT-END===', '366-PRESETGATE-COMMENT'),
    ('    # ===366-DISPATCH-INSERT-BEGIN===', '    # ===366-DISPATCH-INSERT-END===', '366-DISPATCH-INSERT'),
]:
    core2 = strip_span(core, b_m, e_m)
    if core2 is None:
        print(f"MARKER_MISSING:{label}")
        sys.exit(0)
    core = core2

mods = [
    ("if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34):",
     "if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33):",
     "366-CLIGATE-LINE"),
    ("if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34:",
     "if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33:",
     "366-PRESETGATE-LINE"),
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
  fail "source_core_identical_to_365" "insertion marker missing or malformed: ${CORE_RESULT#MARKER_MISSING:}"
elif [[ "$CORE_RESULT" == MOD_LINE_COUNT_WRONG:* ]]; then
  fail "source_core_identical_to_365" "targeted 2-word modification line not found exactly once: ${CORE_RESULT#MOD_LINE_COUNT_WRONG:}"
else
  CORE_HASH=$(echo "$CORE_RESULT" | awk '{print $1}')
  CORE_LINES=$(echo "$CORE_RESULT" | awk '{print $2}')
  if [[ "$CORE_HASH" == "$REF_HASH_365_FULL" && "$CORE_LINES" -eq "$REF_LINES_365_FULL" ]]; then
    pass "source_core_identical_to_365 (hash=$CORE_HASH, lines=$CORE_LINES)"
  else
    fail "source_core_identical_to_365" "expected hash=$REF_HASH_365_FULL lines=$REF_LINES_365_FULL (365's full code, deltas intact), got hash=$CORE_HASH lines=$CORE_LINES -- code outside the marked 366 deltas has drifted"
  fi
fi

# ---------------------------------------------------------------------
# 4. Targeted content checks.
# ---------------------------------------------------------------------
if grep -q 'def check_required_maxd_for_N(N:int,fname:str,gpu_log_level:int=0)->Tuple\[int,int,int\]:' "$NOTAG"; then
  pass "source_maxd_check_function_present"
else
  fail "source_maxd_check_function_present" "check_required_maxd_for_N() signature not found or changed"
fi

if grep -qE '^\s*if use_gpu and N>=21 and bench_mode==34:' "$NOTAG"; then
  pass "source_bench_mode_34_dispatch_present"
else
  fail "source_bench_mode_34_dispatch_present" "bench_mode==34 dispatch branch not found"
fi

if grep -qE '^\s*if not \(bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34\):' "$NOTAG"; then
  pass "source_cligate_whitelist_includes_34"
else
  fail "source_cligate_whitelist_includes_34" "the bench_mode CLI whitelist does not include bench_mode==34 -- this is the exact 361-r1 bug class"
fi

if grep -qE '^\s*if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34:' "$NOTAG"; then
  pass "source_presetgate_includes_34"
else
  fail "source_presetgate_includes_34" "the preset_queens gating condition does not include bench_mode==34"
fi

# No kernel launch: this mode must not call launch_kernel_dfs_iter_gpu_static_maxd.
DISPATCH_BLOCK=$(sed -n '/# ===366-DISPATCH-INSERT-BEGIN===/,/# ===366-DISPATCH-INSERT-END===/p' "$NOTAG")
if echo "$DISPATCH_BLOCK" | grep -q 'launch_kernel_dfs_iter_gpu_static_maxd'; then
  fail "source_no_kernel_launch_in_maxd_check" "the maxd-check dispatch branch appears to call the kernel launch primitive -- this mode is supposed to be launch-free"
else
  pass "source_no_kernel_launch_in_maxd_check"
fi

for marker in '# ===366-INSERT-BEGIN===' '# ===366-INSERT-END===' '      # ===366-CLIGATE-COMMENT-BEGIN===' '      # ===366-CLIGATE-COMMENT-END===' '    # ===366-PRESETGATE-COMMENT-BEGIN===' '    # ===366-PRESETGATE-COMMENT-END===' '    # ===366-DISPATCH-INSERT-BEGIN===' '    # ===366-DISPATCH-INSERT-END===' ; do
  cnt=$(grep -cF "$marker" "$NOTAG")
  if [[ "$cnt" -eq 1 ]]; then
    pass "marker_count_exactly_one[$marker]"
  else
    fail "marker_count_exactly_one[$marker]" "expected exactly 1 occurrence, found $cnt"
  fi
done

if grep -q 'VERSION_TAG:str="366' "$SRC"; then
  pass "source_version_tag_366"
else
  fail "source_version_tag_366" "VERSION_TAG for 366 not found"
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
    ('# ===366-INSERT-BEGIN===', '# ===366-INSERT-END==='),
    ('      # ===366-CLIGATE-COMMENT-BEGIN===', '      # ===366-CLIGATE-COMMENT-END==='),
    ('    # ===366-PRESETGATE-COMMENT-BEGIN===', '    # ===366-PRESETGATE-COMMENT-END==='),
    ('    # ===366-DISPATCH-INSERT-BEGIN===', '    # ===366-DISPATCH-INSERT-END==='),
]:
    core = strip_span(core, b_m, e_m)
mods = [
    ("if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34):",
     "if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33):"),
    ("if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33 or bench_mode==34:",
     "if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33:"),
]
for modified, original in mods:
    core = core.replace(modified, original)
print(hashlib.sha256(core.encode('utf-8')).hexdigest())
PYEOF
)
if [[ "$NEG_RESULT" != "$REF_HASH_365_FULL" ]]; then
  pass "negtest_core_tamper_detected"
else
  fail "negtest_core_tamper_detected" "tampering did not change the hash (test harness bug)"
fi
rm -f "$NEGTMP"

BUG_REPRO=$(python3 -c "
bench_mode = 34
if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33):
    bench_mode = 0
print(bench_mode)
")
if [[ "$BUG_REPRO" == "0" ]]; then
  pass "negtest_361_style_bug_reproduced (confirms the PRE-366 whitelist line would have silently rejected bench_mode=34)"
else
  fail "negtest_361_style_bug_reproduced" "reverting to the pre-366 whitelist line did not reproduce the expected rejection (test harness bug)"
fi

rm -f "$NOTAG" "$NODOC"

# ---------------------------------------------------------------------
# 6. Summary.
# ---------------------------------------------------------------------
echo ""
echo "===== 366 static-check summary ====="
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
# 7. Build and run for N=22 only (single-variable discipline: confirm
#    N=22 before attempting N=23-26). If the bin cache for N=22 doesn't
#    exist yet, this run generates it first -- may take noticeably
#    longer than N=21's cache-hit runs did.
# ---------------------------------------------------------------------
N="${N:-22}"
BLOCK="${BLOCK:-32}"
MAX_BLOCKS="${MAX_BLOCKS:-484}"
LOG_LEVEL="${LOG_LEVEL:-1}"
SORT_MODE="${SORT_MODE:-0}"
PRESET_QUEENS="${PRESET_QUEENS:-7}"
BENCH_MODE="${BENCH_MODE:-34}"
REORDER_WINDOW_MULT="${REORDER_WINDOW_MULT:-3}"
REORDER_PHASE_JUMP="${REORDER_PHASE_JUMP:-7}"
CROSS_STRIPE_SAFE="${CROSS_STRIPE_SAFE:-0}"
WORKER_ID="${WORKER_ID:-0}"
WORKER_COUNT="${WORKER_COUNT:-1}"
BROADMARK_VARIANT="${BROADMARK_VARIANT:-2}"
CHUNKSHAPE148_BUCKET_RUN="${CHUNKSHAPE148_BUCKET_RUN:-2048}"
CHUNKSHAPE148_ITER_SORT="${CHUNKSHAPE148_ITER_SORT:-9}"

echo "Building 366Py (codon)..."
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
echo "NOTE: if constellations_N${N}_*.bin does not exist yet, this will"
echo "generate it first, which may take a while for N=22. If it seems"
echo "to hang with no output, let it run and report back -- do not"
echo "assume it's the 363-style bug (that was CPU DFS work; this step"
echo "is bin generation, a different code path)."
PYLOG="${BIN}_N${N}_maxd_check_$(date +%Y%m%d_%H%M%S).log"
stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "$PYLOG"

DONE_LINE=$(grep '^\[maxd-check-done\]' "$PYLOG" | tail -n1)
if [[ -z "$DONE_LINE" ]]; then
  fail "maxd_check_done_line_found" "no [maxd-check-done] line in $PYLOG"
  echo ""
  echo "===== final summary ====="
  echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
  exit 1
fi
pass "maxd_check_done_line_found"
echo "$DONE_LINE"

echo ""
echo "===== final summary ====="
echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
echo ""
echo "Report the required_maxd/selected_maxd values above. If"
echo "selected_maxd==14, N=$N already fits the existing C port with no"
echo "new kernel work. If selected_maxd is 16/18/20/21, that Codon"
echo "kernel already exists but has no C port yet (future work). If"
echo "selected_maxd==0 (required_maxd>21), N=$N is unsupported by any"
echo "existing kernel, Codon or C -- a bigger question outside this"
echo "project's current scope."
[[ "$FAIL" -gt 0 ]] && exit 1
exit 0
