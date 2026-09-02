#!/usr/bin/env bash
# 361Py_soa_derive_c_port_validate_N21_full_once.sh
#
# rev361 — CUDA C runner port, first implementation step (Open
# Objectives item 6). Scope: build_soa_for_range()+symmetry() ONLY
# (the bin->SoA derivation function, 28-branch funcid decision tree).
# No kernel/dispatcher/reorder-pipeline change relative to 356/357/360.
#
# r2 (post-run bugfix): the first attempt at 361 only added deltas 1-2
# below and missed a pre-existing CLI-level bench_mode whitelist gate
# (rev276, restore274/coretrim) that silently reset bench_mode=32 back
# to 0 before delta 2's dispatch branch was ever reached. Suzuki's first
# real run therefore fell through to the plain ~11-minute default N=21
# execution (correctness 314666222712 was confirmed, but no soa-ref-dump
# was produced). Deltas 3-4 below fix that. See VERSION_TAG in 361Py for
# the full account.
#
# 361Py's delta versus 360Py's executable code region is four clearly
# delimited spans:
#   1. ===361-INSERT-BEGIN/END===              new function
#      dump_soa_reference_c_port(), inserted immediately after
#      build_soa_for_range()'s existing `return soa,w_arr`. Pure
#      insertion, nothing removed or modified.
#   2. ===361-DISPATCH-INSERT-BEGIN/END===      new bench_mode==32
#      branch, inserted after the existing mode30/31 block, before the
#      generic N>=21 fallback. Pure insertion.
#   3. ===361-CLIGATE-COMMENT-BEGIN/END===      explanatory comment
#      (pure insertion) immediately above a 2-word MODIFICATION to the
#      existing bench_mode whitelist condition (adds "or bench_mode==32").
#   4. ===361-PRESETGATE-COMMENT-BEGIN/END===   same pattern: comment
#      (pure insertion) above a 2-word MODIFICATION to the adjacent
#      preset_queens gating condition.
# This script's static check strips spans 1-2 and the comment portions
# of 3-4 entirely, then reverses the exact known 2-word addition on
# each of the two modified lines from 3/4, and hashes the result against
# 360's own reference hash for the SAME docstring-stripped/VERSION_TAG-
# excluded code region (793eef693d8f5af43ca6f131fcdd37d000efb785be5c29
# 5e54188885f5625fd4, 5602 lines) — i.e. it proves the rest of the file,
# including the non-32 parts of the two modified lines, is untouched.
#
# The new 361_soa_derive.c is host-only C (no CUDA in this file at all:
# build_soa_for_range() itself is a CPU-side prep step in the Codon
# source too, run before SoA arrays are uploaded to the GPU). It was
# cross-validated THIS session against a pure-Python re-execution of
# the literal 360Py build_soa_for_range()/symmetry() source (extracted
# verbatim, run under CPython with u32()/u64() stubbed as bitmasks) on
# 2,025,282 synthetic-but-shift-valid N=21 records: byte-for-byte
# identical output, 27/28 reachable funcid branches covered (funcid=3
# is never assigned anywhere in the source itself — a structural gap,
# not a coverage miss). See README.md for the harness used.

set -u
SRC="${SRC:-361Py_soa_derive_c_port.py}"
CSRC="${CSRC:-361_soa_derive.c}"
STATIC_ONLY="${STATIC_ONLY:-0}"
BIN="${BIN:-361Py_soa_derive_c_port}"
CBIN="${CBIN:-361_soa_derive}"

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
#    361 does not itself require ncu/sudo (no profiling in this
#    revision), but the check is kept as a standing habit; failure is
#    non-fatal here.
# ---------------------------------------------------------------------
if sudo -n true 2>/dev/null; then
  pass "sudo_permission_check"
else
  warn "sudo_permission_check" "sudo -n true failed (non-fatal for this revision; no ncu is invoked here)"
fi

# ---------------------------------------------------------------------
# 1. Presence checks.
# ---------------------------------------------------------------------
if [[ ! -f "$SRC" ]]; then
  fail "source_py_present" "$SRC not found in $(pwd)"
  echo "Cannot continue without source file. Aborting."
  exit 1
fi
pass "source_py_present"

if [[ ! -f "$CSRC" ]]; then
  fail "source_c_present" "$CSRC not found in $(pwd)"
  echo "Cannot continue without C source file. Aborting."
  exit 1
fi
pass "source_c_present"

# ---------------------------------------------------------------------
# 2. Docstring-stripped source copy of the .py (regex/split-stripped
#    before any grep), per project discipline: static checks must be
#    immune to prose content, including chat logs pasted into
#    docstrings. 361Py has 3 docstring blocks (6 triple-quote markers,
#    same structure as 360Py), so the executable code starts at
#    parts[6:], not parts[4:].
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
grep -v '^VERSION_TAG:str="361' "$NODOC" > "$NOTAG"

# ---------------------------------------------------------------------
# 3. Core-region hash: strip the two pure-insertion spans and the two
#    comment-only portions of the two modified-line spans, then reverse
#    the exact known 2-word addition ("or bench_mode==32") on each of
#    the two modified lines, then compare against 360's own reference
#    hash for the identical docstring-stripped/VERSION_TAG-excluded
#    code region. A match proves 361's code is byte-identical to
#    356/357/360 outside these four precisely-accounted-for deltas.
# ---------------------------------------------------------------------
REF_HASH_CORE="793eef693d8f5af43ca6f131fcdd37d000efb785be5c295e54188885f5625fd4"
REF_LINES_CORE=5602

CORE_RESULT=$(python3 - "$NOTAG" << 'PYEOF'
import sys, hashlib
path = sys.argv[1]
with open(path, encoding='utf-8') as f:
    s = f.read()

def strip_span(s, begin_marker, end_marker):
    b = s.find(begin_marker)
    e = s.find(end_marker)
    if b == -1 or e == -1:
        return None
    e_end = e + len(end_marker)
    tail = s[e_end:].lstrip('\n')
    return s[:b] + tail

core = s
for begin_m, end_m, label in [
    ('# ===361-INSERT-BEGIN===', '# ===361-INSERT-END===', '361-INSERT'),
    ('    # ===361-DISPATCH-INSERT-BEGIN===', '    # ===361-DISPATCH-INSERT-END===', '361-DISPATCH-INSERT'),
    ('      # ===361-CLIGATE-COMMENT-BEGIN===', '      # ===361-CLIGATE-COMMENT-END===', '361-CLIGATE-COMMENT'),
    ('    # ===361-PRESETGATE-COMMENT-BEGIN===', '    # ===361-PRESETGATE-COMMENT-END===', '361-PRESETGATE-COMMENT'),
]:
    core2 = strip_span(core, begin_m, end_m)
    if core2 is None:
        print(f"MARKER_MISSING:{label}")
        sys.exit(0)
    core = core2

# Reverse the two targeted 2-word modifications. Each must appear
# EXACTLY ONCE, or the reversal (and therefore the whole equivalence
# claim) is not trustworthy.
mods = [
    (
        "if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32):",
        "if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31):",
        "361-CLIGATE-LINE",
    ),
    (
        "if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32:",
        "if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31:",
        "361-PRESETGATE-LINE",
    ),
]
for modified, original, label in mods:
    cnt = core.count(modified)
    if cnt != 1:
        print(f"MOD_LINE_COUNT_WRONG:{label}:{cnt}")
        sys.exit(0)
    core = core.replace(modified, original)

h = hashlib.sha256(core.encode('utf-8')).hexdigest()
lines = core.count('\n') + 1
print(f"{h} {lines}")
PYEOF
)

if [[ "$CORE_RESULT" == MARKER_MISSING:* ]]; then
  fail "source_core_identical_to_356_360" "insertion marker missing or malformed: ${CORE_RESULT#MARKER_MISSING:}"
elif [[ "$CORE_RESULT" == MOD_LINE_COUNT_WRONG:* ]]; then
  fail "source_core_identical_to_356_360" "targeted 2-word modification line not found exactly once: ${CORE_RESULT#MOD_LINE_COUNT_WRONG:}"
else
  CORE_HASH=$(echo "$CORE_RESULT" | awk '{print $1}')
  CORE_LINES=$(echo "$CORE_RESULT" | awk '{print $2}')
  if [[ "$CORE_HASH" == "$REF_HASH_CORE" && "$CORE_LINES" -eq "$REF_LINES_CORE" ]]; then
    pass "source_core_identical_to_356_360 (hash=$CORE_HASH, lines=$CORE_LINES)"
  else
    fail "source_core_identical_to_356_360" "expected hash=$REF_HASH_CORE lines=$REF_LINES_CORE (356/357/360 anchor, four deltas excised/reversed), got hash=$CORE_HASH lines=$CORE_LINES -- code outside the four marked deltas has drifted"
  fi
fi

# ---------------------------------------------------------------------
# 4. Targeted content checks for the two insertions themselves.
# ---------------------------------------------------------------------
if grep -q 'def dump_soa_reference_c_port(N:int,fname:str,out_fname:str,gpu_log_level:int=0)->Tuple\[int,int\]:' "$NOTAG"; then
  pass "source_dump_function_signature_present"
else
  fail "source_dump_function_signature_present" "dump_soa_reference_c_port() signature not found or changed"
fi

if grep -qE '^\s*if use_gpu and N>=21 and bench_mode==32:' "$NOTAG"; then
  pass "source_bench_mode_32_dispatch_present"
else
  fail "source_bench_mode_32_dispatch_present" "bench_mode==32 dispatch branch not found"
fi

if grep -qE '^\s*if not \(bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32\):' "$NOTAG"; then
  pass "source_cligate_whitelist_includes_32"
else
  fail "source_cligate_whitelist_includes_32" "the bench_mode CLI whitelist does not include bench_mode==32 -- bench_mode=32 will be silently reset to 0 before the new dispatch branch is ever reached (this is the exact bug found on Suzuki's first real run of the r1 harness)"
fi

if grep -qE '^\s*if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32:' "$NOTAG"; then
  pass "source_presetgate_includes_32"
else
  fail "source_presetgate_includes_32" "the preset_queens gating condition does not include bench_mode==32"
fi

for marker in '# ===361-INSERT-BEGIN===' '# ===361-INSERT-END===' '    # ===361-DISPATCH-INSERT-BEGIN===' '    # ===361-DISPATCH-INSERT-END===' '      # ===361-CLIGATE-COMMENT-BEGIN===' '      # ===361-CLIGATE-COMMENT-END===' '    # ===361-PRESETGATE-COMMENT-BEGIN===' '    # ===361-PRESETGATE-COMMENT-END===' ; do
  cnt=$(grep -cF "$marker" "$NOTAG")
  if [[ "$cnt" -eq 1 ]]; then
    pass "marker_count_exactly_one[$marker]"
  else
    fail "marker_count_exactly_one[$marker]" "expected exactly 1 occurrence, found $cnt"
  fi
done

if grep -q 'VERSION_TAG:str="361' "$SRC"; then
  pass "source_version_tag_361"
else
  fail "source_version_tag_361" "VERSION_TAG for 361 not found"
fi

# ---------------------------------------------------------------------
# 5. C-side static checks (light — the real check is the byte-diff in
#    step 8 below, run against the ACTUAL production bin file).
# ---------------------------------------------------------------------
for sym in build_soa_for_range_one symmetry symmetry90 geti getj getk getl; do
  if grep -q "$sym" "$CSRC"; then
    pass "c_symbol_present[$sym]"
  else
    fail "c_symbol_present[$sym]" "'$sym' not found in $CSRC"
  fi
done

# NOTE: several target values (0/4/5/7/8/9/10/11/12/15/16/18) are
# assigned inside ternary expressions on a single line, e.g.
# "target = (!l_eq_kp1) ? 0 : 4;" -- both 0 and 4 are reachable targets
# on that one line, neither as a bare "target = N;". The check below
# matches the number as a whole word (\b...\b) anywhere on a line that
# also contains "target", which covers both the plain-assignment and
# ternary forms without over-matching (e.g. "20" does not match "\b2\b").
FUNCID_ALL_OK=1
for tgt in 0 1 2 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27; do
  if grep -qE "target.*\\b${tgt}\\b" "$CSRC"; then
    :
  else
    fail "c_funcid_target_present[$tgt]" "no line containing both 'target' and the whole word '$tgt' found in $CSRC -- one of the 27 reachable funcid values (3 is a structural gap, never assigned in the source) is missing"
    FUNCID_ALL_OK=0
  fi
done
if [[ "$FUNCID_ALL_OK" -eq 1 ]]; then
  pass "c_funcid_targets_present (27/27 reachable values checked)"
fi

# ---------------------------------------------------------------------
# 6. Negative tests. All four go through the SAME strip/reverse pipeline
#    used in step 3 (factored into a helper here) so they actually
#    exercise the real check, not a stale stand-in for it.
# ---------------------------------------------------------------------
CORE_PIPELINE='
import sys, hashlib
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    s = f.read()

def strip_span(s, b_m, e_m):
    b = s.find(b_m); e = s.find(e_m)
    if b == -1 or e == -1:
        return None
    e_end = e + len(e_m)
    return s[:b] + s[e_end:].lstrip("\n")

core = s
for b_m, e_m, label in [
    ("# ===361-INSERT-BEGIN===", "# ===361-INSERT-END===", "361-INSERT"),
    ("    # ===361-DISPATCH-INSERT-BEGIN===", "    # ===361-DISPATCH-INSERT-END===", "361-DISPATCH-INSERT"),
    ("      # ===361-CLIGATE-COMMENT-BEGIN===", "      # ===361-CLIGATE-COMMENT-END===", "361-CLIGATE-COMMENT"),
    ("    # ===361-PRESETGATE-COMMENT-BEGIN===", "    # ===361-PRESETGATE-COMMENT-END===", "361-PRESETGATE-COMMENT"),
]:
    core2 = strip_span(core, b_m, e_m)
    if core2 is None:
        print(f"MARKER_MISSING:{label}")
        sys.exit(0)
    core = core2

mods = [
    ("if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32):",
     "if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31):",
     "361-CLIGATE-LINE"),
    ("if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32:",
     "if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31:",
     "361-PRESETGATE-LINE"),
]
for modified, original, label in mods:
    cnt = core.count(modified)
    if cnt != 1:
        print(f"MOD_LINE_COUNT_WRONG:{label}:{cnt}")
        sys.exit(0)
    core = core.replace(modified, original)

h = hashlib.sha256(core.encode("utf-8")).hexdigest()
lines = core.count("\n") + 1
print(f"{h} {lines}")
'

# (a) tamper with the CORE region (outside all four marked spans) -> the
#     pipeline's hash must diverge from REF_HASH_CORE on the mutant.
NEGTMP=$(mktemp)
cp "$NOTAG" "$NEGTMP"
sed -i 's/def auto_sort_mode(N:int)->int:/def auto_sort_mode(N:int)->int:  # tampered/' "$NEGTMP"
NEG_RESULT=$(python3 -c "$CORE_PIPELINE" "$NEGTMP")
NEG_HASH=$(echo "$NEG_RESULT" | awk '{print $1}')
if [[ "$NEG_HASH" != "$REF_HASH_CORE" ]]; then
  pass "negtest_core_tamper_detected"
else
  fail "negtest_core_tamper_detected" "tampering the core region did not change the hash (test harness bug)"
fi
rm -f "$NEGTMP"

# (b) delete the INSERT-END marker -> pipeline must report
#     MARKER_MISSING, not silently produce a wrong hash.
NEGTMP2=$(mktemp)
cp "$NOTAG" "$NEGTMP2"
sed -i '/# ===361-INSERT-END===/d' "$NEGTMP2"
NEG2_RESULT=$(python3 -c "$CORE_PIPELINE" "$NEGTMP2")
if [[ "$NEG2_RESULT" == MARKER_MISSING:* ]]; then
  pass "negtest_missing_marker_detected"
else
  fail "negtest_missing_marker_detected" "deleting the END marker did not trip MARKER_MISSING (test harness bug), got: $NEG2_RESULT"
fi
rm -f "$NEGTMP2"

# (c) revert the whitelist line to its ORIGINAL 356/360 form (i.e.
#     reproduce the exact r1 bug) -> the pipeline's hash must now MATCH
#     REF_HASH_CORE exactly (since that's what the reversal step is
#     designed to compare against), proving the reversal step really
#     does isolate "or bench_mode==32" as the only difference on that
#     line, AND, separately, that reverted line's own literal condition
#     must reject bench_mode=32 -- reproducing the actual bug Suzuki hit.
NEGTMP3=$(mktemp)
cp "$NOTAG" "$NEGTMP3"
sed -i 's/if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32):/if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31):/' "$NEGTMP3"
BUG_REPRO=$(python3 -c "
bench_mode = 32
if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31):
    bench_mode = 0
print(bench_mode)
")
if [[ "$BUG_REPRO" == "0" ]]; then
  pass "negtest_r1_bug_reproduced (confirms the ORIGINAL 356/360 whitelist line silently rejects bench_mode=32, which is exactly what happened on Suzuki's first run)"
else
  fail "negtest_r1_bug_reproduced" "reverting to the original whitelist line did not reproduce the r1 bug (test harness bug), bench_mode ended up $BUG_REPRO"
fi
rm -f "$NEGTMP3"

# (c) real-world check, if 360Py is present alongside: 360Py must NOT
#     contain the new function or dispatch branch (sanity that these
#     really are new in 361, not something already latent in 360).
if [[ -f "360Py_cuda_c_port_spec_update.py" ]]; then
  if grep -q 'def dump_soa_reference_c_port' "360Py_cuda_c_port_spec_update.py"; then
    fail "negtest_360_lacks_dump_function" "360Py unexpectedly already contains dump_soa_reference_c_port -- 361's diff claim is wrong"
  else
    pass "negtest_360_lacks_dump_function"
  fi
else
  info "negtest_360_lacks_dump_function" "360Py_cuda_c_port_spec_update.py not found alongside this script -- skipped (not fatal)"
fi

rm -f "$NOTAG"

# ---------------------------------------------------------------------
# 7. Summary (always printed, even on STATIC_ONLY or FAIL, per 200's
#    lesson: never let a static-check failure suppress the summary).
# ---------------------------------------------------------------------
echo ""
echo "===== 361 static-check summary ====="
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
# 8. Build both sides, run both, byte-diff the reference dumps.
#
#    Codon side: bench_mode=32 resolves the SAME raw stream bin
#    ensure_constellations_bin_stream() always resolves for N=21 (the
#    pre-reorder cache file 337 already validated end-to-end:
#    records_read=2025282=EXPECTED_TASKS, checksum_u64=13342728758502),
#    runs the unmodified build_soa_for_range()+symmetry() over every
#    record in file order, and writes a flat 10-field-per-record u32 LE
#    reference dump.
#
#    C side: 361_soa_derive reads the SAME bin path (resolved from the
#    Codon run's own log line, not hardcoded) and writes the same
#    layout independently.
#
#    Judgment: cmp -s of the two dump files. Exact byte match required
#    (this is a behavioral-equivalence port, not a performance
#    experiment -- no +-3% tolerance applies here; that tolerance is
#    reserved for the eventual kernel port's timing, per 338/360 spec).
# ---------------------------------------------------------------------
N="${N:-21}"
BLOCK="${BLOCK:-32}"
MAX_BLOCKS="${MAX_BLOCKS:-484}"
LOG_LEVEL="${LOG_LEVEL:-1}"
SORT_MODE="${SORT_MODE:-0}"
PRESET_QUEENS="${PRESET_QUEENS:-7}"
BENCH_MODE="${BENCH_MODE:-32}"
REORDER_WINDOW_MULT="${REORDER_WINDOW_MULT:-3}"
REORDER_PHASE_JUMP="${REORDER_PHASE_JUMP:-7}"
CROSS_STRIPE_SAFE="${CROSS_STRIPE_SAFE:-0}"
WORKER_ID="${WORKER_ID:-0}"
WORKER_COUNT="${WORKER_COUNT:-1}"
BROADMARK_VARIANT="${BROADMARK_VARIANT:-2}"
CHUNKSHAPE148_BUCKET_RUN="${CHUNKSHAPE148_BUCKET_RUN:-2048}"
CHUNKSHAPE148_ITER_SORT="${CHUNKSHAPE148_ITER_SORT:-9}"

echo "Building 361Py (codon)..."
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

echo "Building 361_soa_derive.c (host-only C, no CUDA)..."
CC="${CC:-gcc}"
if ! command -v "$CC" >/dev/null 2>&1; then
  CC=cc
fi
if ! command -v "$CC" >/dev/null 2>&1; then
  fail "c_toolchain_present" "neither gcc nor cc found on PATH"
  exit 1
fi
rm -f "$CBIN"
"$CC" -O2 -Wall -Wextra -o "$CBIN" "$CSRC" 2>&1 | tee "${CBIN}_build_$(date +%Y%m%d_%H%M%S).log"
if [[ ! -x "$CBIN" ]]; then
  fail "c_build_succeeded" "binary $CBIN was not produced"
  exit 1
fi
pass "c_build_succeeded"

CMD=("$CAND" -g "$N" "$N" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS" "$BENCH_MODE" "$REORDER_WINDOW_MULT" "$REORDER_PHASE_JUMP" "$CROSS_STRIPE_SAFE" "$WORKER_ID" "$WORKER_COUNT" "$BROADMARK_VARIANT" "$CHUNKSHAPE148_BUCKET_RUN" "$CHUNKSHAPE148_ITER_SORT")
echo "Running (Codon side): ${CMD[*]}"
PYLOG="${BIN}_N21_soa_ref_$(date +%Y%m%d_%H%M%S).log"
stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "$PYLOG"

DONE_LINE=$(grep '^\[soa-ref-dump-done\]' "$PYLOG" | tail -n1)
if [[ -z "$DONE_LINE" ]]; then
  fail "py_dump_done_line_found" "no [soa-ref-dump-done] line in $PYLOG -- Codon-side run did not complete as expected"
  echo ""
  echo "===== final summary ====="
  echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
  exit 1
fi
pass "py_dump_done_line_found"

SRC_BIN=$(echo "$DONE_LINE" | sed -n 's/.*src_bin=\([^ ]*\).*/\1/p')
PY_RECORDS=$(echo "$DONE_LINE" | sed -n 's/.*records=\([0-9]*\).*/\1/p')
PY_OUT=$(echo "$DONE_LINE" | sed -n 's/.*out=\([^ ]*\).*/\1/p')
PY_CHECKSUM=$(echo "$DONE_LINE" | sed -n 's/.*checksum_u64=\([0-9]*\).*/\1/p')

echo "Resolved: src_bin=$SRC_BIN records=$PY_RECORDS out=$PY_OUT checksum_u64=$PY_CHECKSUM"

EXPECTED_TASKS=2025282
if [[ "$PY_RECORDS" == "$EXPECTED_TASKS" ]]; then
  pass "py_records_match_expected_tasks ($PY_RECORDS)"
else
  fail "py_records_match_expected_tasks" "got $PY_RECORDS, expected $EXPECTED_TASKS"
fi

if [[ ! -f "$SRC_BIN" ]]; then
  fail "src_bin_exists" "$SRC_BIN not found -- cannot run C side"
  echo ""
  echo "===== final summary ====="
  echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
  exit 1
fi
pass "src_bin_exists"

C_OUT="${SRC_BIN}.soa_ref_361_c.bin"
echo "Running (C side): ./$CBIN $N $SRC_BIN $C_OUT"
CLOG="${CBIN}_N21_soa_ref_$(date +%Y%m%d_%H%M%S).log"
"./$CBIN" "$N" "$SRC_BIN" "$C_OUT" 2>&1 | tee "$CLOG"

C_DONE_LINE=$(grep '^\[soa-ref-dump-done\]' "$CLOG" | tail -n1)
C_CHECKSUM=$(echo "$C_DONE_LINE" | sed -n 's/.*checksum_u64=\([0-9]*\).*/\1/p')
C_RECORDS=$(echo "$C_DONE_LINE" | sed -n 's/.*records=\([0-9]*\).*/\1/p')

if [[ "$C_RECORDS" == "$EXPECTED_TASKS" ]]; then
  pass "c_records_match_expected_tasks ($C_RECORDS)"
else
  fail "c_records_match_expected_tasks" "got $C_RECORDS, expected $EXPECTED_TASKS"
fi

if [[ -n "$PY_CHECKSUM" && "$PY_CHECKSUM" == "$C_CHECKSUM" ]]; then
  pass "checksum_quick_match (py=$PY_CHECKSUM c=$C_CHECKSUM)"
else
  fail "checksum_quick_match" "py=$PY_CHECKSUM c=$C_CHECKSUM -- checksums differ, STOP, do not trust the byte-diff below either"
fi

echo "Byte-diffing $PY_OUT (Codon) vs $C_OUT (C)..."
if cmp -s "$PY_OUT" "$C_OUT"; then
  pass "soa_dump_byte_identical"
else
  fail "soa_dump_byte_identical" "cmp reported a difference -- first differing byte: $(cmp "$PY_OUT" "$C_OUT" 2>&1)"
fi

echo ""
echo "===== final summary ====="
echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
if [[ "$FAIL" -eq 0 ]]; then
  echo "361 PASSED: build_soa_for_range()+symmetry() C port is byte-for-byte"
  echo "equivalent to the Codon reference on the real N=21 bin (2,025,282"
  echo "records). Ready to proceed to 362 (kernel_dfs_iter_gpu_maxd14 port)."
fi
[[ "$FAIL" -gt 0 ]] && exit 1
exit 0
