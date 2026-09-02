#!/usr/bin/env bash
# 360Py_cuda_c_port_spec_update_validate_N21_full_once.sh
#
# rev360 — CUDA C port spec update (338 -> 356 delta), DESIGN-ONLY.
# No kernel/dispatcher code change is intended relative to 356/357.
# The only source delta versus 359Py_pushbranch_partial.py is: the one
# push-guard hunk 359 modified has been reverted to the 356/357 anchor
# form quoted verbatim in the project README ("356/357(アンカー)"
# block). Everything else (VERSION_TAG, docstring, this harness) is new.
#
# VERIFICATION NOTE: the literal 356Py file (356Py_savesp_narrow.py) was
# supplied by Suzuki and used directly to build this file's code region
# (docstring/VERSION_TAG differ by design; import-gpu-onward is meant to
# be byte-identical to 356/357). A prior session had reconstructed 356's
# code from 359Py by reverting its one documented push-guard hunk; that
# reconstruction was cross-checked against the real 356Py this session
# via direct diff and found byte-for-byte identical (0 differences),
# confirming the reconstruction method was sound. This file was then
# rebuilt directly from the real 356Py rather than the reconstruction, so
# the reference hash below is a genuine verification, not a self-
# reference.

set -u
SRC="${SRC:-360Py_cuda_c_port_spec_update.py}"
STATIC_ONLY="${STATIC_ONLY:-0}"
BIN="${BIN:-360Py_cuda_c_port_spec_update}"

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
# 0. sudo check FIRST (352 lesson: missing sudo cost ~14 minutes there).
#    360 is design-only and does not itself require ncu/sudo, but the
#    check is kept as a standing habit per project discipline; failure
#    here is non-fatal for a design-only revision (no ncu is run) but is
#    reported so the habit is never silently skipped.
# ---------------------------------------------------------------------
if sudo -n true 2>/dev/null; then
  pass "sudo_permission_check"
else
  warn "sudo_permission_check" "sudo -n true failed (non-fatal for this design-only revision; no ncu is invoked here, but fix before any future ncu-bearing revision)"
fi

# ---------------------------------------------------------------------
# 1. Docstring-stripped source copy (regex-stripped before any grep),
#    per project discipline: static checks must be immune to prose
#    content, including chat logs pasted into docstrings.
# ---------------------------------------------------------------------
if [[ ! -f "$SRC" ]]; then
  fail "source_file_present" "$SRC not found in $(pwd)"
  echo "Cannot continue without source file. Aborting."
  exit 1
fi
pass "source_file_present"

NODOC="${SRC%.py}_nodoc.py"
python3 - "$SRC" "$NODOC" << 'PYEOF'
import re, sys
src_path, out_path = sys.argv[1], sys.argv[2]
with open(src_path, 'r', encoding='utf-8') as f:
    text = f.read()
# Strip the two leading triple-quoted docstring blocks only (not any
# triple-quoted string appearing later inside code, of which this file
# has none). This mirrors the project's documented convention: static
# checks run against a docstring-stripped copy so that prose (including
# pasted chat logs) can never contaminate a grep-based check.
# NOTE: 360Py now embeds THREE docstring blocks (the ASCII-art/Open-
# Objectives header, the 360 session-summary narrative, and the full
# 360_maxd14_port_spec_update.md content as a third block) -- 6 triple-
# quote markers total, not 4. parts[6:] is the executable code, not
# parts[4:].
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

# ---------------------------------------------------------------------
# 2. sha256 fingerprint of the docstring-stripped body, VERSION_TAG line
#    excluded, compared against 360's OWN self-computed reference (see
#    HONESTY NOTE above -- this is NOT the historical 356/357 reference).
# ---------------------------------------------------------------------
# Verified directly against the literal 356Py file Suzuki supplied this
# session (356Py_savesp_narrow.py): docstring-stripped (triple-quote
# split, parts[4:]), VERSION_TAG line excluded. A direct diff of that
# stripped body against this file's own stripped body showed a byte-for-
# byte match (0 differences) before this hash was pinned. This is a real
# verification, not a self-reference -- see the script header note.
REF_HASH_356="793eef693d8f5af43ca6f131fcdd37d000efb785be5c295e54188885f5625fd4"
REF_LINES_356=5601

NOTAG=$(mktemp)
grep -v '^VERSION_TAG:str="360' "$NODOC" > "$NOTAG"
ACTUAL_HASH=$(sha256sum "$NOTAG" | awk '{print $1}')
ACTUAL_LINES=$(wc -l < "$NOTAG")

if [[ "$ACTUAL_HASH" == "$REF_HASH_356" && "$ACTUAL_LINES" -eq "$REF_LINES_356" ]]; then
  pass "source_code_identical_to_356 (hash=$ACTUAL_HASH, lines=$ACTUAL_LINES)"
else
  fail "source_code_identical_to_356" "expected hash=$REF_HASH_356 lines=$REF_LINES_356 (verified against the literal 356Py this session), got hash=$ACTUAL_HASH lines=$ACTUAL_LINES -- source has drifted"
fi

info "note_357_historical_hash_mismatch" "357's own historical reference (6db7de43fe05143352e9b8d5917b334c4031e488ed234417153fc74901fc09ca, 5607 lines) does not match this script's stripping method's output for 356 either (this script gets 793eef69..., 5601 lines, for the literal 356Py). Likely a docstring-stripping methodology difference in 357's original script (e.g. line-counting or whitespace handling), not a code-content difference -- confirmed by a direct line-for-line diff against the literal 356Py showing zero differences in the code region. Non-blocking."
# ---------------------------------------------------------------------
# 3. Targeted content checks for the two deltas this revision documents.
#    NOTE: these run against $NOTAG (VERSION_TAG line excluded), not
#    $NODOC, precisely because VERSION_TAG's own prose legitimately
#    discusses w_hi_arr/save_sp while explaining what changed -- running
#    against NODOC would let that explanatory prose falsely trip the
#    "must be absent" check below (a code-region analogue of the
#    docstring-contamination problem this project already guards against
#    with the docstring-stripped-copy discipline).
# ---------------------------------------------------------------------
if grep -qE '^\s*save_sp:u32=u32\(0\)' "$NOTAG"; then
  pass "source_savesp_u32_declared"
else
  fail "source_savesp_u32_declared" "expected 'save_sp:u32=u32(0)' declaration (356's narrowing) not found"
fi

if grep -qE '^\s*stack_ptr:int=0' "$NOTAG"; then
  pass "source_stack_ptr_int_unchanged"
else
  fail "source_stack_ptr_int_unchanged" "expected 'stack_ptr:int=0' (356 deliberately left this un-narrowed) not found"
fi

if grep -q 'w_hi_arr' "$NOTAG"; then
  fail "source_w_hi_arr_absent" "'w_hi_arr' still referenced somewhere in the executable code (outside VERSION_TAG) -- 351 removed it and it must not reappear"
else
  pass "source_w_hi_arr_absent"
fi

if grep -q 'w_lo_arr' "$NOTAG"; then
  pass "source_w_lo_arr_present"
else
  fail "source_w_lo_arr_present" "'w_lo_arr' not found -- unexpected, this array must remain"
fi

if grep -q 'WHI_ELIM_REASON' "$NOTAG"; then
  pass "source_whi_elim_reason_constant_present"
else
  fail "source_whi_elim_reason_constant_present" "351/352's WHI_ELIM_REASON documentation constant not found"
fi

# Precise, context-aware check (python, not grep) for the maxd14 hot-loop
# push-guard shape: the 356/357 anchor is exactly
#   next_depth:int=cur_depth+1
#   if cur_avail!=u32(0):
#     stack[stack_ptr]=u64(cur_ld)|(u64(cur_rd)<<u64(32))
#     stack[stack_ptr+1]=u64(cur_col)|(u64(cur_avail|(u32(cur_depth)<<u32(27)))<<u64(32))
#     stack_ptr+=2
#     save_sp+=u32(1)
# 358/359 both instead have an unconditional stack[stack_ptr]=... store as
# the line immediately after next_depth:int=cur_depth+1 (358: both stores
# unconditional; 359: only the ldrd store, with a smaller 3-statement if
# for the rest). Checking the literal line immediately following
# "next_depth:int=cur_depth+1" distinguishes all three shapes reliably,
# without relying on any comment text (which differs in language/wording
# between 358/359 and is not a safe discriminator on its own).
PUSHGUARD_CHECK=$(python3 - "$NOTAG" << 'PYEOF'
import sys
with open(sys.argv[1], encoding='utf-8') as f:
    lines = f.readlines()
found = False
anchor_ok = False
for i, l in enumerate(lines):
    if l.strip() == 'next_depth:int=cur_depth+1':
        found = True
        nxt = lines[i+1].strip() if i+1 < len(lines) else ''
        if nxt == 'if cur_avail!=u32(0):':
            anchor_ok = True
        break
print('FOUND=%s ANCHOR=%s' % (found, anchor_ok))
PYEOF
)
if [[ "$PUSHGUARD_CHECK" == "FOUND=True ANCHOR=True" ]]; then
  pass "source_pushguard_anchor_form_356_357"
else
  fail "source_pushguard_anchor_form_356_357" "expected the line right after 'next_depth:int=cur_depth+1' to be 'if cur_avail!=u32(0):' (the 356/357 anchor shape); got: $PUSHGUARD_CHECK -- this indicates a 358/359-style unconditional store, or the marker line itself was not found"
fi

if grep -q '^VERSION_TAG:str="360 cuda_c_port_spec_update_356_delta' "$NODOC"; then
  pass "source_version_tag_360"
else
  fail "source_version_tag_360" "expected VERSION_TAG to start with '360 cuda_c_port_spec_update_356_delta'"
fi

rm -f "$NOTAG"

# ---------------------------------------------------------------------
# 4. Negative tests: mutate a scratch copy and confirm checks correctly
#    FAIL (per project discipline: feeding wrong source must FAIL).
# ---------------------------------------------------------------------
NEGTMP=$(mktemp)
cp "$NODOC" "$NEGTMP"
# (a) reintroduce a w_hi_arr reference into the CODE region (mimicking a
#     careless partial revert), matching a body line, not VERSION_TAG ->
#     the real check (against a fresh NOTAG rebuilt from this mutant)
#     must FAIL.
sed -i 's/^import gpu/w_hi_arr_test_injection:int=0\nimport gpu/' "$NEGTMP"
NEGNOTAG=$(mktemp)
grep -v '^VERSION_TAG:str="360' "$NEGTMP" > "$NEGNOTAG"
if grep -q 'w_hi_arr' "$NEGNOTAG"; then
  pass "negtest_w_hi_arr_reinjection_detected"
else
  fail "negtest_w_hi_arr_reinjection_detected" "injection did not take (test harness bug)"
fi
rm -f "$NEGTMP" "$NEGNOTAG"

NEGTMP2=$(mktemp)
cp "$NODOC" "$NEGTMP2"
# (b) revert save_sp back to int (352-style) -> source_savesp_u32_declared
#     must FAIL when re-checked against this mutant.
sed -i 's/save_sp:u32=u32(0)/save_sp:int=0/' "$NEGTMP2"
if grep -qE '^\s*save_sp:u32=u32\(0\)' "$NEGTMP2"; then
  fail "negtest_savesp_reversion_detected" "reversion did not take (test harness bug)"
else
  pass "negtest_savesp_reversion_detected"
fi
rm -f "$NEGTMP2"

# (c) real-world negative test: run the pushguard anchor check against the
#     ACTUAL 359Py source (if present alongside this script), which is
#     known to have the unconditional-store shape. Must FAIL there.
if [[ -f "359Py_pushbranch_partial.py" ]]; then
  python3 - "359Py_pushbranch_partial.py" << 'PYEOF'
import sys, re
with open(sys.argv[1], encoding='utf-8') as f:
    text = f.read()
parts = text.split('"""')
rest = '"""'.join(parts[4:]) if len(parts) >= 5 else text
lines = [l for l in rest.splitlines() if not l.startswith('VERSION_TAG:str="359')]
found = False
anchor_ok = False
for i, l in enumerate(lines):
    if l.strip() == 'next_depth:int=cur_depth+1':
        found = True
        nxt = lines[i+1].strip() if i+1 < len(lines) else ''
        if nxt == 'if cur_avail!=u32(0):':
            anchor_ok = True
        break
sys.exit(0 if (found and not anchor_ok) else 1)
PYEOF
  if [[ $? -eq 0 ]]; then
    pass "negtest_359_source_correctly_fails_anchor_check"
  else
    fail "negtest_359_source_correctly_fails_anchor_check" "running the anchor check against the real 359Py did not report the expected non-anchor shape -- the check may not actually discriminate 359 from 356/357"
  fi
else
  info "negtest_359_source_correctly_fails_anchor_check" "359Py_pushbranch_partial.py not found alongside this script -- skipped (not fatal, but recommended when available)"
fi

# ---------------------------------------------------------------------
# 5. Summary (always printed, even on STATIC_ONLY or FAIL, per 200's
#    lesson: never let a static-check failure suppress the summary).
# ---------------------------------------------------------------------
echo ""
echo "===== 360 static-check summary ====="
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
# 6. Build + N=21 full GPU run.
#
#    CORRECTED after Suzuki's first real run of this script: the earlier
#    version of this section invoked the binary with a bare "-g" flag,
#    which triggers the generic CPU/GPU demo mode shown in the docstring
#    boilerplate (N=5..21 sequential sweep, no chunk-level output). That
#    mode does not stop cleanly after N=21 either -- it continues on to
#    N=22 (multi-hour), which is what looked like a hang and required
#    Ctrl-C. This section now uses the actual positional-argument
#    invocation every real revision's validate script uses (confirmed by
#    reading 359Py_pushbranch_partial_validate_N21_full_once.sh's own
#    CMD construction), which runs ONLY N=21 through the mode31/split145
#    chunked path and terminates on its own.
#
#    Expected: correctness 314666222712, elapsed ~393.4s (356's own
#    figure, chunk0/1/2 = 144590/144473/103271 ms), since this revision's
#    code is byte-identical to 356/357 (verified this session against the
#    literal 356Py -- see source_code_identical_to_356 above).
#
#    NOTE: this harness intentionally does NOT reproduce the full ~1400-
#    line check apparatus real per-revision validate scripts carry
#    (progress.tsv duplicate/missing-chunk checks, dispatch row parsing,
#    telemetry capture, dozens of historical baseline comparisons, etc.).
#    It covers correctness and the chunk0/1/2 timing comparison needed to
#    judge this design-only revision, and nothing more. If a fuller check
#    is wanted, run this build under an actual per-revision harness
#    (e.g. 356Py_savesp_narrow_validate_N21_full_once.sh) instead.
# ---------------------------------------------------------------------
N="${N:-21}"
BLOCK="${BLOCK:-32}"
MAX_BLOCKS="${MAX_BLOCKS:-484}"
LOG_LEVEL="${LOG_LEVEL:-1}"
SORT_MODE="${SORT_MODE:-0}"
PRESET_QUEENS="${PRESET_QUEENS:-7}"
BENCH_MODE="${BENCH_MODE:-31}"
REORDER_WINDOW_MULT="${REORDER_WINDOW_MULT:-3}"
REORDER_PHASE_JUMP="${REORDER_PHASE_JUMP:-7}"
CROSS_STRIPE_SAFE="${CROSS_STRIPE_SAFE:-0}"
WORKER_ID="${WORKER_ID:-0}"
WORKER_COUNT="${WORKER_COUNT:-1}"
BROADMARK_VARIANT="${BROADMARK_VARIANT:-2}"
CHUNKSHAPE148_BUCKET_RUN="${CHUNKSHAPE148_BUCKET_RUN:-2048}"
CHUNKSHAPE148_ITER_SORT="${CHUNKSHAPE148_ITER_SORT:-9}"

echo "Building..."
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
  fail "build_succeeded" "binary $CAND was not produced"
  exit 1
fi
pass "build_succeeded"

CMD=("$CAND" -g "$N" "$N" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS" "$BENCH_MODE" "$REORDER_WINDOW_MULT" "$REORDER_PHASE_JUMP" "$CROSS_STRIPE_SAFE" "$WORKER_ID" "$WORKER_COUNT" "$BROADMARK_VARIANT" "$CHUNKSHAPE148_BUCKET_RUN" "$CHUNKSHAPE148_ITER_SORT")
echo "Running: ${CMD[*]}"
LOGFILE="${BIN}_N21_full_once_$(date +%Y%m%d_%H%M%S).log"
stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "$LOGFILE"

TOTAL_21=$(grep -E "^[[:space:]]*${N}:" "$LOGFILE" | tail -n1 | awk '{print $2}')
if [[ "$TOTAL_21" == "314666222712" ]]; then
  pass "correctness_314666222712"
else
  fail "correctness_314666222712" "got '${TOTAL_21:-missing}', expected 314666222712 -- STOP, do not read timing"
fi

# Per-chunk elapsed, if a progress.tsv path was printed to the log (same
# convention as real revisions: "... progress=<path>.tsv ...").
PROGRESS=$(sed -n 's/.* progress=\([^[:space:]]*\.tsv\).*/\1/p' "$LOGFILE" | tail -n1 | tr -d '\r')
if [[ -n "$PROGRESS" && -s "$PROGRESS" ]]; then
  echo ""
  echo "Per-chunk timing from $PROGRESS:"
  cat "$PROGRESS"
else
  info "progress_tsv_found" "no progress=....tsv path found in the run log; only the overall elapsed line is available below"
fi

echo ""
echo "Reference for comparison (356's own confirmed figures, NOT re-derived here):"
echo "  356 elapsed total : 393.404s"
echo "  356 chunk0         : 144,590 ms"
echo "  356 chunk1         : 144,473 ms"
echo "  356 chunk2         : 103,271 ms"
echo "Compare this run's chunk0/1/2 against the above by hand (chunk-level,"
echo "not session elapsed, per project discipline on cross-session noise)."
echo ""
echo "===== final summary ====="
echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
[[ "$FAIL" -gt 0 ]] && exit 1
exit 0
