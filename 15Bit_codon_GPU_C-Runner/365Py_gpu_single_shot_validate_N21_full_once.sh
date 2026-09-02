#!/usr/bin/env bash
# 365Py_gpu_single_shot_validate_N21_full_once.sh
#
# rev365 — single-shot (non-chunked) Codon-side GPU baseline, mirroring
# 364's CUDA C host runner protocol exactly, per Suzuki's decision to
# adopt 364's simpler single-shot configuration as the new standard
# (rather than replicating 356's exact 3-chunk measure2 protocol).
#
# 365Py's delta versus 364Py's executable code region is five clearly
# delimited spans:
#   1. ===365-INSERT-BEGIN/END===             new function
#      exec_solutions_gpu_single_shot(), pure insertion.
#   2. ===365-CLIGATE-COMMENT-BEGIN/END===     comment (pure insertion)
#      above a 2-word MODIFICATION ("or bench_mode==33") to the CLI
#      whitelist gate -- applied preemptively this time, learning from
#      361's r1 bug (that exact gate silently reset bench_mode==32 to
#      0 for an hour before anyone noticed).
#   3. ===365-PRESETGATE-COMMENT-BEGIN/END===  same pattern for the
#      adjacent preset_queens gating condition.
#   4. ===365-DISPATCH-INSERT-BEGIN/END===     new bench_mode==33
#      dispatch branch, pure insertion.
# This script's static check strips the pure-insertion spans and
# reverses the two 2-word CLI-gate modifications, then hashes the
# result against 364's own reference hash for the identical
# docstring-stripped/VERSION_TAG-excluded code region.

set -u
SRC="${SRC:-365Py_gpu_single_shot.py}"
STATIC_ONLY="${STATIC_ONLY:-0}"
BIN="${BIN:-365Py_gpu_single_shot}"

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
grep -v '^VERSION_TAG:str="365' "$NODOC" > "$NOTAG"

# ---------------------------------------------------------------------
# 3. Core-region hash: strip the three pure-insertion spans, reverse
#    the two targeted 2-word CLI-gate modifications, and compare
#    against 364's own reference hash for the identical region.
# ---------------------------------------------------------------------
REF_HASH_364="2b7688b8af2db194ad3f8f60041acac7ccffbb58ef5fc74b669c261053f537ff"
REF_LINES_364=5682

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
    ('# ===365-INSERT-BEGIN===', '# ===365-INSERT-END===', '365-INSERT'),
    ('      # ===365-CLIGATE-COMMENT-BEGIN===', '      # ===365-CLIGATE-COMMENT-END===', '365-CLIGATE-COMMENT'),
    ('    # ===365-PRESETGATE-COMMENT-BEGIN===', '    # ===365-PRESETGATE-COMMENT-END===', '365-PRESETGATE-COMMENT'),
    ('    # ===365-DISPATCH-INSERT-BEGIN===', '    # ===365-DISPATCH-INSERT-END===', '365-DISPATCH-INSERT'),
]:
    core2 = strip_span(core, b_m, e_m)
    if core2 is None:
        print(f"MARKER_MISSING:{label}")
        sys.exit(0)
    core = core2

mods = [
    ("if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33):",
     "if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32):",
     "365-CLIGATE-LINE"),
    ("if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33:",
     "if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32:",
     "365-PRESETGATE-LINE"),
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
  fail "source_core_identical_to_364" "insertion marker missing or malformed: ${CORE_RESULT#MARKER_MISSING:}"
elif [[ "$CORE_RESULT" == MOD_LINE_COUNT_WRONG:* ]]; then
  fail "source_core_identical_to_364" "targeted 2-word modification line not found exactly once: ${CORE_RESULT#MOD_LINE_COUNT_WRONG:}"
else
  CORE_HASH=$(echo "$CORE_RESULT" | awk '{print $1}')
  CORE_LINES=$(echo "$CORE_RESULT" | awk '{print $2}')
  if [[ "$CORE_HASH" == "$REF_HASH_364" && "$CORE_LINES" -eq "$REF_LINES_364" ]]; then
    pass "source_core_identical_to_364 (hash=$CORE_HASH, lines=$CORE_LINES)"
  else
    fail "source_core_identical_to_364" "expected hash=$REF_HASH_364 lines=$REF_LINES_364, got hash=$CORE_HASH lines=$CORE_LINES -- code outside the marked deltas has drifted"
  fi
fi

# ---------------------------------------------------------------------
# 4. Targeted content checks.
# ---------------------------------------------------------------------
if grep -q 'def exec_solutions_gpu_single_shot(N:int,fname:str,preset_queens:int' "$NOTAG"; then
  pass "source_single_shot_function_present"
else
  fail "source_single_shot_function_present" "exec_solutions_gpu_single_shot() signature not found or changed"
fi

if grep -qE '^\s*if use_gpu and N>=21 and bench_mode==33:' "$NOTAG"; then
  pass "source_bench_mode_33_dispatch_present"
else
  fail "source_bench_mode_33_dispatch_present" "bench_mode==33 dispatch branch not found"
fi

if grep -qE '^\s*if not \(bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33\):' "$NOTAG"; then
  pass "source_cligate_whitelist_includes_33"
else
  fail "source_cligate_whitelist_includes_33" "the bench_mode CLI whitelist does not include bench_mode==33 -- this is the exact 361-r1 bug class; bench_mode=33 would be silently reset to 0"
fi

if grep -qE '^\s*if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33:' "$NOTAG"; then
  pass "source_presetgate_includes_33"
else
  fail "source_presetgate_includes_33" "the preset_queens gating condition does not include bench_mode==33"
fi

for marker in '# ===365-INSERT-BEGIN===' '# ===365-INSERT-END===' '      # ===365-CLIGATE-COMMENT-BEGIN===' '      # ===365-CLIGATE-COMMENT-END===' '    # ===365-PRESETGATE-COMMENT-BEGIN===' '    # ===365-PRESETGATE-COMMENT-END===' '    # ===365-DISPATCH-INSERT-BEGIN===' '    # ===365-DISPATCH-INSERT-END===' ; do
  cnt=$(grep -cF "$marker" "$NOTAG")
  if [[ "$cnt" -eq 1 ]]; then
    pass "marker_count_exactly_one[$marker]"
  else
    fail "marker_count_exactly_one[$marker]" "expected exactly 1 occurrence, found $cnt"
  fi
done

if grep -q 'VERSION_TAG:str="365' "$SRC"; then
  pass "source_version_tag_365"
else
  fail "source_version_tag_365" "VERSION_TAG for 365 not found"
fi

# ---------------------------------------------------------------------
# 5. Negative tests.
# ---------------------------------------------------------------------
# (a) tamper with the core region -> hash check must FAIL.
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
    ('# ===365-INSERT-BEGIN===', '# ===365-INSERT-END==='),
    ('      # ===365-CLIGATE-COMMENT-BEGIN===', '      # ===365-CLIGATE-COMMENT-END==='),
    ('    # ===365-PRESETGATE-COMMENT-BEGIN===', '    # ===365-PRESETGATE-COMMENT-END==='),
    ('    # ===365-DISPATCH-INSERT-BEGIN===', '    # ===365-DISPATCH-INSERT-END==='),
]:
    core = strip_span(core, b_m, e_m)
mods = [
    ("if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33):",
     "if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32):"),
    ("if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32 or bench_mode==33:",
     "if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32:"),
]
for modified, original in mods:
    core = core.replace(modified, original)
print(hashlib.sha256(core.encode('utf-8')).hexdigest())
PYEOF
)
if [[ "$NEG_RESULT" != "$REF_HASH_364" ]]; then
  pass "negtest_core_tamper_detected"
else
  fail "negtest_core_tamper_detected" "tampering the core region did not change the hash (test harness bug)"
fi
rm -f "$NEGTMP"

# (b) revert the whitelist line to the pre-365 (364) form -> confirms
#     bench_mode=33 would have been rejected, i.e. the fix is real and
#     load-bearing (mirrors 361's negtest_r1_bug_reproduced).
BUG_REPRO=$(python3 -c "
bench_mode = 33
if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32):
    bench_mode = 0
print(bench_mode)
")
if [[ "$BUG_REPRO" == "0" ]]; then
  pass "negtest_361_style_bug_reproduced (confirms the PRE-365 whitelist line would have silently rejected bench_mode=33)"
else
  fail "negtest_361_style_bug_reproduced" "reverting to the pre-365 whitelist line did not reproduce the expected rejection (test harness bug)"
fi

rm -f "$NOTAG" "$NODOC"

# ---------------------------------------------------------------------
# 6. Summary.
# ---------------------------------------------------------------------
echo ""
echo "===== 365 static-check summary ====="
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
# 7. Build and run on real hardware.
# ---------------------------------------------------------------------
N="${N:-21}"
BLOCK="${BLOCK:-32}"
MAX_BLOCKS="${MAX_BLOCKS:-484}"
LOG_LEVEL="${LOG_LEVEL:-1}"
SORT_MODE="${SORT_MODE:-0}"
PRESET_QUEENS="${PRESET_QUEENS:-7}"
BENCH_MODE="${BENCH_MODE:-33}"
REORDER_WINDOW_MULT="${REORDER_WINDOW_MULT:-3}"
REORDER_PHASE_JUMP="${REORDER_PHASE_JUMP:-7}"
CROSS_STRIPE_SAFE="${CROSS_STRIPE_SAFE:-0}"
WORKER_ID="${WORKER_ID:-0}"
WORKER_COUNT="${WORKER_COUNT:-1}"
BROADMARK_VARIANT="${BROADMARK_VARIANT:-2}"
CHUNKSHAPE148_BUCKET_RUN="${CHUNKSHAPE148_BUCKET_RUN:-2048}"
CHUNKSHAPE148_ITER_SORT="${CHUNKSHAPE148_ITER_SORT:-9}"

echo "Building 365Py (codon)..."
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
PYLOG="${BIN}_N21_single_shot_$(date +%Y%m%d_%H%M%S).log"
stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "$PYLOG"

DONE_LINE=$(grep '^\[single-shot-done\]' "$PYLOG" | tail -n1)
if [[ -z "$DONE_LINE" ]]; then
  fail "py_single_shot_done_line_found" "no [single-shot-done] line in $PYLOG"
  echo ""
  echo "===== final summary ====="
  echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
  exit 1
fi
pass "py_single_shot_done_line_found"

PY_TOTAL=$(echo "$DONE_LINE" | sed -n 's/.*total=\([0-9]*\).*/\1/p')
PY_KERNEL_MS=$(echo "$DONE_LINE" | sed -n 's/.*kernel_elapsed_ms=\([0-9]*\).*/\1/p')

EXPECTED_ORACLE=314666222712
if [[ "$PY_TOTAL" == "$EXPECTED_ORACLE" ]]; then
  pass "py_total_matches_oracle (total=$PY_TOTAL)"
else
  fail "py_total_matches_oracle" "got total=$PY_TOTAL, expected $EXPECTED_ORACLE"
fi

# ---------------------------------------------------------------------
# 8. +-3% comparison against 364's kernel_ms=260685.062ms.
# ---------------------------------------------------------------------
REF_364_KERNEL_MS="260685.062"
if [[ -n "$PY_KERNEL_MS" ]]; then
  PCT=$(python3 -c "
ref = $REF_364_KERNEL_MS
got = $PY_KERNEL_MS
pct = (got - ref) / ref * 100.0
print(f'{pct:.4f}')
")
  echo "365 (Codon single-shot) kernel_elapsed_ms=$PY_KERNEL_MS vs 364 (C single-shot) kernel_ms=$REF_364_KERNEL_MS -> ${PCT}%"
  WITHIN=$(python3 -c "print('yes' if abs(float('$PCT')) <= 3.0 else 'no')")
  if [[ "$WITHIN" == "yes" ]]; then
    pass "timing_within_3pct_of_364 (${PCT}%)"
  else
    info "timing_within_3pct_of_364" "${PCT}% is outside +-3%% -- not a failure by itself (this is the FIRST same-protocol comparison ever made; report the number, do not force a PASS/FAIL verdict pre-registration was not done for this specific check)"
  fi
else
  fail "timing_within_3pct_of_364" "could not parse kernel_elapsed_ms from $DONE_LINE"
fi

echo ""
echo "===== final summary ====="
echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
[[ "$FAIL" -gt 0 ]] && exit 1
exit 0
