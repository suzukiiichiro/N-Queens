#!/usr/bin/env bash
# 362Py_kernel_port_spec_validate_N21_full_once.sh
#
# rev362 — kernel_dfs_iter_gpu_maxd14 C-port SPEC (design-only, no code
# change). Deliverable is 362_kernel_port_spec.md, embedded in this
# file's docstring. Executable code region (import gpu onward) is
# byte-identical to 361 (r2) — this script's only real job is to prove
# that, the same way 338/360's harnesses proved their own zero-code-
# change claims.
#
# Unlike 338/360 (which had no prior additive delta to preserve), 362
# must remain byte-identical to 361's FULL code region including all
# four of 361's marked deltas (two pure insertions, two targeted 2-word
# modifications) — i.e. the reference here is 361's own full
# docstring-stripped/VERSION_TAG-excluded hash, not the older 356/360
# "core" hash. No span-stripping is needed this time since nothing new
# was added on top of 361.

set -u
SRC="${SRC:-362Py_kernel_port_spec.py}"
STATIC_ONLY="${STATIC_ONLY:-0}"
BIN="${BIN:-362Py_kernel_port_spec}"

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
# 0. sudo check FIRST (352 lesson). Non-fatal for this design-only
#    revision (no ncu is invoked here).
# ---------------------------------------------------------------------
if sudo -n true 2>/dev/null; then
  pass "sudo_permission_check"
else
  warn "sudo_permission_check" "sudo -n true failed (non-fatal for this design-only revision)"
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
# 2. Docstring-stripped source copy (same 3-docstring-block / parts[6:]
#    convention as 360/361).
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
grep -v '^VERSION_TAG:str="362' "$NODOC" > "$NOTAG"

# ---------------------------------------------------------------------
# 3. Full-region hash: 362's code region must be byte-identical to
#    361's own code region (all four of 361's deltas preserved intact,
#    nothing added, nothing removed).
# ---------------------------------------------------------------------
REF_HASH_361="2b7688b8af2db194ad3f8f60041acac7ccffbb58ef5fc74b669c261053f537ff"
REF_LINES_361=5682

ACTUAL_HASH=$(sha256sum "$NOTAG" | awk '{print $1}')
ACTUAL_LINES=$(wc -l < "$NOTAG")

if [[ "$ACTUAL_HASH" == "$REF_HASH_361" && "$ACTUAL_LINES" -eq "$REF_LINES_361" ]]; then
  pass "source_code_identical_to_361 (hash=$ACTUAL_HASH, lines=$ACTUAL_LINES)"
else
  fail "source_code_identical_to_361" "expected hash=$REF_HASH_361 lines=$REF_LINES_361 (361 r2 code region), got hash=$ACTUAL_HASH lines=$ACTUAL_LINES -- code has drifted from 361"
fi

# Sanity: 361's own four markers/modifications must still be present
# (this is a subset check of the hash above, but gives a more specific
# failure message if something did drift).
if grep -qE '^\s*if not \(bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32\):' "$NOTAG"; then
  pass "source_361_cligate_fix_preserved"
else
  fail "source_361_cligate_fix_preserved" "361's bench_mode==32 CLI whitelist fix is missing -- 362 must not regress this"
fi

if grep -q 'def dump_soa_reference_c_port' "$NOTAG"; then
  pass "source_361_dump_function_preserved"
else
  fail "source_361_dump_function_preserved" "361's dump_soa_reference_c_port() is missing"
fi

# ---------------------------------------------------------------------
# 4. Spec-document sanity checks (light — this is a design doc, not
#    code; the real review happens in README/chat).
# ---------------------------------------------------------------------
for phrase in "362_kernel_port_spec.md" "kernel_dfs_iter_gpu_maxd14" "IS_BASE_MASK" "meta_next" "stack_ptr" "save_sp" "future_check_mask" "child_jmark_mask" "terminal_depth" "393.404"; do
  if grep -qF "$phrase" "$SRC"; then
    pass "spec_contains[$phrase]"
  else
    fail "spec_contains[$phrase]" "expected phrase not found in $SRC docstring content"
  fi
done

# ---------------------------------------------------------------------
# 5. Negative test: tamper with the code region -> hash check must
#    FAIL on the mutant.
# ---------------------------------------------------------------------
NEGTMP=$(mktemp)
cp "$NOTAG" "$NEGTMP"
sed -i 's/def auto_sort_mode(N:int)->int:/def auto_sort_mode(N:int)->int:  # tampered/' "$NEGTMP"
NEG_HASH=$(sha256sum "$NEGTMP" | awk '{print $1}')
if [[ "$NEG_HASH" != "$REF_HASH_361" ]]; then
  pass "negtest_core_tamper_detected"
else
  fail "negtest_core_tamper_detected" "tampering did not change the hash (test harness bug)"
fi
rm -f "$NEGTMP" "$NODOC" "$NOTAG"

# ---------------------------------------------------------------------
# 6. Summary.
# ---------------------------------------------------------------------
echo ""
echo "===== 362 static-check summary ====="
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
# 7. Reuse-run: since 362 is code-identical to 361, re-running 361's
#    soa-ref-dump mode (bench_mode=32) should reproduce the exact same
#    checksum as 361's confirmed run. This is not a new capability
#    check, just a "nothing broke" rebuild+rerun, matching 338/360's
#    own reuse-verification pattern.
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

echo "Building 362Py (codon)..."
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
LOGFILE="${BIN}_N21_soa_ref_$(date +%Y%m%d_%H%M%S).log"
stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "$LOGFILE"

DONE_LINE=$(grep '^\[soa-ref-dump-done\]' "$LOGFILE" | tail -n1)
if [[ -z "$DONE_LINE" ]]; then
  fail "dump_done_line_found" "no [soa-ref-dump-done] line in $LOGFILE"
else
  pass "dump_done_line_found"
  echo "$DONE_LINE"
  REPRO_CHECKSUM=$(echo "$DONE_LINE" | sed -n 's/.*checksum_u64=\([0-9]*\).*/\1/p')
  REF_CHECKSUM="7905625137249"
  if [[ "$REPRO_CHECKSUM" == "$REF_CHECKSUM" ]]; then
    pass "checksum_matches_361_confirmed_run ($REPRO_CHECKSUM)"
  else
    fail "checksum_matches_361_confirmed_run" "got $REPRO_CHECKSUM, expected $REF_CHECKSUM (361's confirmed run) -- unexpected since 362 is code-identical to 361"
  fi
fi

echo ""
echo "===== final summary ====="
echo "OK=$PASS  FAIL=$FAIL  INFO=$INFO  WARN=$WARN"
[[ "$FAIL" -gt 0 ]] && exit 1
exit 0
