#!/usr/bin/env bash
# 395c_r5_validate.sh
#
# rev395c-r5 -- take the -0.30% found in r4 to the PRODUCTION -g path.
#
# WHERE WE ARE (395c / r2 / r3 / r4; every cell oracle-MATCH, every cell
# 1710 MHz, in-session repeat noise 0.012-0.022%)
#
#   r4 resolved the model. Contexts are a THRESHOLD, not a count:
#     free_mb 22018:  1 ctx 147,807 (S255) | 2 ctx 133,561 (D1) / 133,563 (M1)
#     free_mb 21762:  2 ctx 133,181 (Y510) | 3 ctx 133,201 (X2b)   <- same
#     free_mb ~21.2k: 2 ctx ~134,0xx        | 5 ctx 138,174 | 7 ctx 137,404
#   So: one context is slow, two or three is the optimum, five or more is bad
#   again. And an idle context created in OUR OWN process by cuCtxCreate is
#   indistinguishable from a foreign holder: M1 133,562.562 vs D1 133,561.062,
#   a difference of 1.5 ms in 133,561 (+0.0011%).
#
#   At two contexts, extra occupied memory has a shallow optimum:
#     +0 MB 133,561 | +255 MB 133,181 | +512 MB 133,854 | +1024 MB 134,310
#     | +2048 MB 147,635 (the cliff sits between 1024 and 2048)
#
#   Production today is Codon parent (255 MiB) + CRunner = 2 contexts,
#   free_mb 22,018, 133,585.188. The optimum is free_mb ~21,763, about
#   400 ms below it. Reaching it needs ONE more context -- and no explicit
#   allocation at all, because a context already brings its own 255 MiB.
#
# WHAT THIS REVISION MEASURES
#   Two independent routes to that state, on the real -g path:
#     Gh  an idle holder process alongside -g        -- ZERO code change
#     Gx  NQ_EXTRA_CTX=1 reaching the CRunner        -- no helper process
#   Both should land on free_mb 21,763 with 3 contexts, so they should give
#   the SAME number. That agreement is itself a check on the model.
#
#   No kernel change, and no host change either: the binary is r4's, and the
#   Codon source is a pure rename of 395cPy whose only code-region edits are
#   VERSION_TAG, REV_TAG and the dispatch-table entry (removed=3 added=9).
#   env_prefix stays EMPTY so one table entry serves both routes and
#   NQ_EXTRA_CTX can come from the caller's environment (os.system inherits).
#
# CELLS
#   G0     -g 21 21, nothing extra          anchor; free_mb must be 22,018
#   Gh     -g 21 21 + 1 idle holder         free_mb must be 21,763
#   Gx     -g 21 21 + NQ_EXTRA_CTX=1        free_mb must be 21,763
#   Gxb    replicate Gx                     the number we would deploy on
#   F128   direct, 1 holder x 128 MB        bracket the +255 MB optimum
#   F384   direct, 1 holder x 384 MB        bracket it from the other side
#
# PRE-REGISTERED (395c_r5_README_append.md; written before execution)
#   R1  G0 within +-0.5% of 133,585 AND free_mb 22,018 -> the rename and the
#       r4 binary are inert on the production path. If this fails, nothing
#       below is interpretable.
#   R2  Gh within +-0.15% of 133,201 -> the zero-code-change route delivers.
#   R3  Gx within +-0.15% of 133,190, and the CRunner log shows
#       [gpu-ctx] extra_ctx=1 -> the env var reaches the child and the
#       in-process route delivers. Falsified if extra_ctx=0 appears: the
#       dispatcher does not pass the environment through, and route (b)
#       needs the table's env_prefix instead.
#   R4  |Gx - Gh| <= 0.05% AND both report free_mb 21,763 -> the two routes
#       reach the identical device state, as the model says they must.
#       A disagreement larger than that means the model is still missing
#       something about WHICH process owns the extra context.
#   R5  Gxb within +-0.05% of Gx.
#   R6  F128 >= 133,181 and F384 >= 133,181 -> +255 MB really is the local
#       optimum. If either lands below 133,150, the minimum is elsewhere and
#       a finer sweep is worth its time.
#
#   CLAIM RULE, fixed in advance: the gain is claimed only if
#   (G0 - min(Gh,Gx,Gxb)) / G0 >= 0.20% AND Gxb replicates Gx within 0.05%.
#   Below that, r4's -0.30% was a between-session artifact and production
#   stays as it is.
#
# USAGE
#   STATIC_ONLY=1 bash 395c_r5_validate.sh
#                 bash 395c_r5_validate.sh     # ~21 min incl. the codon build

set -u

REV="395c_r5"
PY_SRC="${PY_SRC:-395c_r5Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-395c_r5Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-395cPy_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-395c_r4_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-395c_r4_kernel_maxd14}"
HOLDER_SRC="${HOLDER_SRC:-395c_ctx_holder.cu}"
HOLDER_BIN="${HOLDER_BIN:-395c_ctx_holder}"
HOLDER_SECS="${HOLDER_SECS:-1200}"
CRLOG_DIR="${CRLOG_DIR:-395c_r5_crunner_logs}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
ORACLE="${ORACLE:-314666222712}"
IN_RAW="${IN_RAW:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
IN_PROD="${IN_PROD:-${IN_RAW}.sched394f.bin}"
EXPECTED_RECORDS="${EXPECTED_RECORDS:-2025282}"
MB_PROD="${MB_PROD:-800}"
HOLD_F1="${HOLD_F1:-128}"
HOLD_F2="${HOLD_F2:-384}"
# references from 395c / r2 / r3 / r4
REF_G0_MS="${REF_G0_MS:-133585}"
REF_GH_MS="${REF_GH_MS:-133201}"
REF_GX_MS="${REF_GX_MS:-133190}"
REF_Y510_MS="${REF_Y510_MS:-133181}"
FREE_G0="${FREE_G0:-22018}"
FREE_G3="${FREE_G3:-21763}"
KERNEL_SHA_395C="${KERNEL_SHA_395C:-ebd7f523deb591347eab1a352a9555c2e4a6c0b4a456d80517322c933a86e68a}"
COOLDOWN="${COOLDOWN:-20}"
STATIC_ONLY="${STATIC_ONLY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"
TSV="$LOGDIR/${REV}_results.tsv"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }

# ---------------------------------------------------------------------
# 1. Static
# ---------------------------------------------------------------------
for f in "$PY_SRC" "$HELPER_SRC" "$CU_SRC" "$HOLDER_SRC" "$IN_RAW"; do
  [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"
done
if [[ -f "$IN_PROD" ]]; then
  sz="$(stat -c%s "$IN_PROD")"
  [[ "$sz" -eq $((EXPECTED_RECORDS*28)) ]] && pass "sched_input_present_and_sized[$((sz/28)) records]" \
    || fail "sched_input_present_and_sized" "$IN_PROD is $sz bytes, expected $((EXPECTED_RECORDS*28))"
else fail "sched_input_present_and_sized" "$IN_PROD missing -- r5 must NOT regenerate it inside a timed session"; fi
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }

# The code region is everything from the first top-level `import` onward.
# The file's head is documentation: the 700-line Open Objectives docstring
# plus two empty """ """ slots kept for revision notes. Stripping only the
# FIRST triple-quote pair (the pre-r5 method) left those slots inside the
# "code", so pasting notes into one moved the fingerprint by dozens of lines
# and, if a paste ever contained """ itself, shifted the string boundary and
# made real code parse as text. Anchoring on `import` is immune to both.
py_code_region() {
  python3 -c "
import sys
lines=open(sys.argv[1],encoding='utf-8').read().split('\n')
i=next(i for i,l in enumerate(lines) if l.startswith('import '))
sys.stdout.write('\n'.join(l for l in lines[i:] if not l.lstrip().startswith('#')))
" "$1"
}
py_note_region_quote_count() {
  python3 -c "
import sys
lines=open(sys.argv[1],encoding='utf-8').read().split('\n')
i=next(i for i,l in enumerate(lines) if l.startswith('import '))
print('\n'.join(lines[:i]).count('\"'*3))
" "$1"
}
py_code_region "$PY_SRC" > "/tmp/${REV}_code_only.py"
CODE="/tmp/${REV}_code_only.py"
# The revision notes belong IN the source, as # comment lines, and this gate
# is what makes that stick: if a future revision ships without them, the run
# stops here instead of quietly losing the record. Comment lines cost nothing
# -- they are excluded from the fingerprint below -- so there is no reason to
# leave them out.
NOTE_LINES=$(awk '/^# =+$/{f=1} f&&/^#/{n++} END{print n+0}' "$PY_SRC")
if grep -q "^# ${REV//_/-} " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; then
  pass "py_revision_notes_present ($NOTE_LINES comment lines in the note block; the ${REV//_/-} record is in the source)"
else
  fail "py_revision_notes_present" "no '# ${REV//_/-} ...' note block (>=20 comment lines) in $PY_SRC -- paste the revision rationale, cells and pre-registered predictions in as '#' comment lines above the first import"
fi
NQ_=$(py_note_region_quote_count "$PY_SRC")
if [[ $((NQ_ % 2)) -eq 0 ]]; then
  pass "py_note_region_quotes_balanced ($NQ_ triple-quotes before the first import -- every note slot closes)"
else
  fail "py_note_region_quotes_balanced" "$NQ_ triple-quotes before the first import: a note slot does not close, so real code below is being parsed as text"
fi
grep -qE '^REV_TAG:str="395c_r5"' "$CODE" && pass "source_rev_tag_is_395c_r5" || fail "source_rev_tag_is_395c_r5" "wrong REV_TAG"
grep -q 'CRunnerEntry(14,"./395c_r4_kernel_maxd14",""' "$CODE" && pass "source_dispatch_table_references_r4_binary" || fail "source_dispatch_table_references_r4_binary" "table does not point at $CU_BIN"
[[ "$(grep -c 'CRunnerEntry(14,"./395c_kernel_maxd14",""' "$CODE")" == "0" ]] && pass "source_old_395c_entry_removed" || fail "source_old_395c_entry_removed" "the 395c entry is still live -- the run would use the wrong binary"
grep -qE '^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=800' "$CODE" && pass "source_default_max_blocks_800" || fail "source_default_max_blocks_800" "not 800"
grep -q '      if argc <= 4:' "$CODE" && [[ "$(grep -c '      if argc == 2:' "$CODE")" == "0" ]] && pass "source_defaults_apply_for_argc_le_4" || fail "source_defaults_apply_for_argc_le_4" "the A10G_FINAL defaults gate is not argc<=4"
grep -qE '^CRUNNER_INPUT_ORDER:str="sched"' "$CODE" && pass "source_input_order_sched" || fail "source_input_order_sched" "not sched"
grep -q 'NQ_MAX_BLOCKS={gpu_max_blocks} {entry_base37.env_prefix}' "$CODE" && pass "source_mode37_stride_coupling" || fail "source_mode37_stride_coupling" "mode 37 does not pass NQ_MAX_BLOCKS"
grep -q '> {log_path} 2>&1' "$CODE" && pass "source_crunner_stderr_is_captured (the [gpu-ctx] line will reach the crunner log)" || fail "source_crunner_stderr_is_captured" "crunner_run does not redirect stderr -- Gx could not be verified"
[[ "$(grep -c 'os.system(f"python3' "$CODE")" == "2" && "$(grep 'os.system(f"python3' "$CODE" | grep -vc '2>&1')" == "0" ]] && pass "source_external_tools_output_redirected" || fail "source_external_tools_output_redirected" "an external python3 os.system call is not redirected"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  # Fingerprint on the WHOLE file, not the docstring-stripped one: the strip
  # depends on where the first triple-quote pair happens to fall, so a change
  # anywhere in the docstring can move the boundary and produce a nonsense
  # count. Whole-file diff is deterministic. Expected: 4 removed / 10 added
  # = VERSION_TAG, REV_TAG, the table entry, the docstring date line, plus
  # the 6-line table comment.
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true)
  PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  if [[ "$PR_" == "3" && "$PA_" == "3" ]]; then
    pass "py_diff_fingerprint_vs_395cPy (removed=3 added=3 EXECUTABLE lines: VERSION_TAG, REV_TAG, table entry. Comments and notes are not counted -- annotate freely.)"
  else
    fail "py_diff_fingerprint_vs_395cPy" "removed=$PR_ added=$PA_, expected 3/3 executable lines -- this revision must be a pure rename. Comment lines and everything above the first import are already excluded, so this counts real code only."
    echo "      --- the actual CODE difference (first 40 lines) -------------------"
    diff "/tmp/${REV}_prev_code.py" "$CODE" | head -40 | sed 's/^/      /' | cut -c1-140
    echo "      --- $PREV_PY: $(wc -l < "$PREV_PY") lines, sha $(sha256sum "$PREV_PY" | cut -c1-16)"
    echo "      --- $PY_SRC: $(wc -l < "$PY_SRC") lines, sha $(sha256sum "$PY_SRC" | cut -c1-16)"
    echo "      If $PREV_PY is NEWER than the copy r5 was derived from, r5 would"
    echo "      silently revert those edits. Regenerate r5Py from the on-disk file."
    echo "      -------------------------------------------------------------------"
  fi
else info "py_diff_fingerprint_vs_395cPy" "skipped ($PREV_PY absent)"; fi

# --- the .cu is r4's, unchanged ---
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
awk 'f||/^#include/{f=1;print}' "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA_395C" ]] && pass "cu_kernel_region_sha_unchanged_since_395a_r2 (${KB:0:16}...)" || fail "cu_kernel_region_sha_unchanged_since_395a_r2" "got ${KB:0:16}..."
grep -q 'CUDA_VERSION >= 13000' "$CU_SRC" && pass "cuCtxCreate_v4_signature_handled" || fail "cuCtxCreate_v4_signature_handled" "no CUDA 13 branch"
grep -q 'CU_CHECK(cuCtxPopCurrent(&popped));' "$CU_SRC" && pass "extra_ctx_is_popped" || fail "extra_ctx_is_popped" "the created context is left current"
grep -q '    int MAX_BLOCKS = 800;' "$CU_SRC" && pass "cu_binary_default_max_blocks_800" || fail "cu_binary_default_max_blocks_800" "not 800"
[[ "$(grep -c 'uint64_t top0' "$CU_SRC")" == "0" ]] && pass "cu_395b_register_top_absent" || fail "cu_395b_register_top_absent" "395b's top0/top1 is back"
grep -q 'RISK NOTE' "$CU_SRC" && grep -q 'if (cur_avail != 0u) {' "$CU_SRC" && pass "cu_push_guard_untouched" || fail "cu_push_guard_untouched" "the 358/359 push guard form changed"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; nvidia-smi -q -d CLOCK 2>&1
  sha256sum "$PY_SRC" "$HELPER_SRC" "$CU_SRC" "$HOLDER_SRC" "$IN_RAW" "$IN_PROD" 2>&1; [[ -f "$PREV_PY" ]] && sha256sum "$PREV_PY"; } > "$LOGDIR/00_env_pre.txt" 2>&1

# ---------------------------------------------------------------------
# 2. Builds
# ---------------------------------------------------------------------
banner "Building $CU_SRC (with -lcuda), $HOLDER_SRC and $PY_SRC"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -o "$CU_BIN" "$CU_SRC" -lcuda 2>&1 | tee "$LOGDIR/05_nvcc_build.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build_succeeded[$CU_BIN]" || { fail "nvcc_build_succeeded" "no binary"; exit 1; }
"$NVCC" -O2 -arch="$ARCH" -o "$HOLDER_BIN" "$HOLDER_SRC" 2>&1 | tee "$LOGDIR/05a_nvcc_holder.log"
[[ -x "$HOLDER_BIN" ]] && pass "ctx_holder_build_succeeded" || { fail "ctx_holder_build_succeeded" "no $HOLDER_BIN"; exit 1; }
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/06_codon_build.log"
[[ -x "$PY_BIN" ]] && pass "codon_build_succeeded[$PY_BIN]" || { fail "codon_build_succeeded" "see $LOGDIR/06_codon_build.log"; exit 1; }
head -c $((15488*28)) "$IN_RAW" > "/tmp/${REV}_probe.bin"
env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX "./$CU_BIN" "$NQ" "/tmp/${REV}_probe.bin" "/tmp/${REV}_probe_out.bin" > "$LOGDIR/05b_default_config_probe.log" 2>&1 || true
grep -q 'MAX_BLOCKS=800 stride=25600' "$LOGDIR/05b_default_config_probe.log" && pass "binary_default_config_is_800" || fail "binary_default_config_is_800" "see 05b log"
grep -q '\[gpu-ctx\] extra_ctx=0' "$LOGDIR/05b_default_config_probe.log" && pass "extra_ctx_inert_when_unset" || fail "extra_ctx_inert_when_unset" "see 05b log"

# ---------------------------------------------------------------------
# 3. Runs
# ---------------------------------------------------------------------
printf 'cell\tpath\tholders\textra_ctx\tkernel_ms\tfree_mb\ttotal_sum\tmatch\tstride_actual\tsm_clock_mhz\ttemp_c\tstart\n' > "$TSV"
declare -A KMS FREEMB CLK
SAMPLER_PID=""; APPS_PID=""
start_samplers() {
  local cell="$1" out="$LOGDIR/clk_${1}.tsv" apps="$LOGDIR/apps_${1}.tsv"
  : > "$out"; : > "$apps"
  ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits >> "$out" 2>/dev/null; sleep 5; done ) &
  SAMPLER_PID=$!
  ( while true; do t="$(date +%H:%M:%S)"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed "s|^|$t, |" >> "$apps"; sleep 2; done ) &
  APPS_PID=$!
}
stop_samplers() {
  local cell="$1"
  [[ -n "$SAMPLER_PID" ]] && { kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; }; SAMPLER_PID=""
  [[ -n "$APPS_PID" ]] && { kill "$APPS_PID" 2>/dev/null; wait "$APPS_PID" 2>/dev/null; }; APPS_PID=""
  CLK[$cell]="$(awk -F', *' 'NF>=6 && $2+0>0 && $6+0>50 {n++; s+=$2; if(min==""||$2<min)min=$2; p+=$4} END{if(n) printf "sm_mean=%.0f sm_min=%.0f power_mean=%.1fW n=%d", s/n, min, p/n, n; else print "no-samples"}' "$LOGDIR/clk_${cell}.tsv")"
  info "in_run_clock[$cell]" "${CLK[$cell]}"
  info "concurrent_procs[$cell]" "$(awk -F', *' '{c[$1]++} END{m=0; for(t in c) if(c[t]>m) m=c[t]; print "max="m}' "$LOGDIR/apps_${cell}.tsv")"
}
record_row() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$TSV"; }

HPID=""
start_holder() {  # cell mb
  local cell="$1" mb="$2"
  "./$HOLDER_BIN" "$mb" "$HOLDER_SECS" > "$LOGDIR/holder_${cell}.log" 2>&1 &
  HPID=$!; sleep 5; cat "$LOGDIR/holder_${cell}.log"
  grep -q '\[ctx-holder\] context up' "$LOGDIR/holder_${cell}.log" && pass "holder_up[$cell] (${mb} MB)" || { fail "holder_up[$cell]" "no live context"; return 1; }
}
stop_holder() { [[ -n "$HPID" ]] && { kill "$HPID" 2>/dev/null; wait "$HPID" 2>/dev/null || true; }; HPID=""; sleep 3; }
gpu_occupancy_gate() {  # cell expected
  local cell="$1" want="$2" n
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null > "$LOGDIR/apps_before_${cell}.txt"
  n="$(grep -c '^[0-9]' "$LOGDIR/apps_before_${cell}.txt" || true)"; cat "$LOGDIR/apps_before_${cell}.txt"
  [[ "$n" == "$want" ]] && pass "gpu_occupancy_as_intended[$cell] ($n foreign)" || { fail "gpu_occupancy_as_intended[$cell]" "$n foreign, expected $want"; return 1; }
}
free_gate() {  # cell expected tol
  local cell="$1" want="$2" tol="${3:-3}" got="${FREEMB[$1]:-}"
  [[ -z "$got" || "$got" == "?" ]] && { info "free_mb_as_intended[$cell]" "not captured"; return 0; }
  awk -v a="$got" -v b="$want" -v t="$tol" 'BEGIN{exit !((a-b<=t)&&(b-a<=t))}' \
    && pass "free_mb_as_intended[$cell] ($got, expected $want)" \
    || fail "free_mb_as_intended[$cell]" "$got MiB free, expected $want -- the device state is not the one this cell is meant to test"
}

run_dispatch() {  # cell extra_ctx holders_desc
  local cell="$1" xctx="$2" hdesc="$3"
  banner "cell $cell: ./$PY_BIN -g $NQ $NQ   NQ_EXTRA_CTX=$xctx  holders=$hdesc"
  local clk tmp start gcr total kms stride_act match freemb xseen
  clk="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  tmp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  start="$(date -Is)"; gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${NQ}.log"
  rm -f "$gcr"
  start_samplers "$cell"
  if [[ "$xctx" == "0" ]]; then
    env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX "./$PY_BIN" -g "$NQ" "$NQ" 2>&1 | tee "$LOGDIR/1_${cell}_console.log"
  else
    env -u NQ_MAX_BLOCKS -u NQ_PAD_MB NQ_EXTRA_CTX="$xctx" "./$PY_BIN" -g "$NQ" "$NQ" 2>&1 | tee "$LOGDIR/1_${cell}_console.log"
  fi
  stop_samplers "$cell"
  cp "$gcr" "$LOGDIR/1_${cell}_crunner.log" 2>/dev/null || true
  cp "$CRLOG_DIR/dispatch.log" "$LOGDIR/dispatch_after_${cell}.log" 2>/dev/null || true
  if [[ ! -f "$gcr" ]]; then fail "crunner_path_taken[$cell]" "no $gcr"; return 1; fi
  total="$(grep -o 'total_sum=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"
  kms="$(grep -o 'kernel_ms=[0-9.]*' "$gcr" | head -1 | cut -d= -f2)"
  stride_act="$(grep -o 'stride=[0-9]*' "$gcr" | tail -1 | cut -d= -f2)"
  freemb="$(grep -o 'free_mb=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"
  xseen="$(grep -o 'extra_ctx=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"
  match=0; grep -q '\[gpu-run-correctness\] MATCH' "$gcr" && match=1
  record_row "$cell" dispatch "$hdesc" "${xseen:-?}" "${kms:-?}" "${freemb:-?}" "${total:-?}" "$match" "${stride_act:-?}" "$clk" "$tmp" "$start"
  if [[ "$match" -eq 1 && "${total:-}" == "$ORACLE" ]]; then pass "oracle_match[$cell]"; else fail "oracle_match[$cell]" "total_sum='${total:-<none>}' match=$match"; return 1; fi
  [[ "${stride_act:-}" == "$((32*MB_PROD))" ]] && pass "stride_as_intended[$cell]" || { fail "stride_as_intended[$cell]" "got ${stride_act:-?}"; return 1; }
  [[ "${xseen:-}" == "$xctx" ]] && pass "extra_ctx_reached_the_child[$cell] (extra_ctx=$xseen)" \
    || { fail "extra_ctx_reached_the_child[$cell]" "the CRunner reported extra_ctx=${xseen:-?}, the shell asked for $xctx -- the dispatcher is not passing the environment through; use the table env_prefix instead"; return 1; }
  KMS[$cell]="$kms"; FREEMB[$cell]="${freemb:-?}"
  info "cell[$cell]" "kernel_ms=$kms  free_mb=${freemb:-?}"
  return 0
}
run_direct() {  # cell holders_desc
  local cell="$1" hdesc="$2"
  banner "cell $cell: direct ./$CU_BIN  holders=$hdesc"
  local clk tmp start log total kms stride_act match freemb
  clk="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  tmp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  start="$(date -Is)"; log="$LOGDIR/1_${cell}.log"
  start_samplers "$cell"
  env -u NQ_EXTRA_CTX -u NQ_PAD_MB NQ_MAX_BLOCKS="$MB_PROD" "./$CU_BIN" "$NQ" "$IN_PROD" "/tmp/${REV}_${cell}.bin" "$ORACLE" 2>&1 | tee "$log"
  stop_samplers "$cell"
  total="$(grep -o 'total_sum=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  kms="$(grep -o 'kernel_ms=[0-9.]*' "$log" | head -1 | cut -d= -f2)"
  stride_act="$(grep -o 'stride=[0-9]*' "$log" | tail -1 | cut -d= -f2)"
  freemb="$(grep -o 'free_mb=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  match=0; grep -q '\[gpu-run-correctness\] MATCH' "$log" && match=1
  record_row "$cell" direct "$hdesc" 0 "${kms:-?}" "${freemb:-?}" "${total:-?}" "$match" "${stride_act:-?}" "$clk" "$tmp" "$start"
  if [[ "$match" -eq 1 && "${total:-}" == "$ORACLE" ]]; then pass "oracle_match[$cell]"; else fail "oracle_match[$cell]" "total_sum='${total:-<none>}' match=$match"; return 1; fi
  KMS[$cell]="$kms"; FREEMB[$cell]="${freemb:-?}"
  info "cell[$cell]" "kernel_ms=$kms  free_mb=${freemb:-?}"
  return 0
}
cool() { echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; }

banner "6 cells: G0 Gh Gx Gxb F128 F384  (~19 min)"

gpu_occupancy_gate G0 0 || exit 1
run_dispatch G0 0 "none" || exit 1
free_gate G0 "$FREE_G0" 2
cool

start_holder Gh 0 || exit 1
gpu_occupancy_gate Gh 1 || { stop_holder; exit 1; }
run_dispatch Gh 0 "1x0MB" || { stop_holder; exit 1; }
stop_holder; free_gate Gh "$FREE_G3" 3
cool

gpu_occupancy_gate Gx 0 || exit 1
run_dispatch Gx 1 "none" || exit 1
free_gate Gx "$FREE_G3" 3
cool

gpu_occupancy_gate Gxb 0 || exit 1
run_dispatch Gxb 1 "none" || exit 1
free_gate Gxb "$FREE_G3" 3
cool

start_holder F128 "$HOLD_F1" || exit 1
gpu_occupancy_gate F128 1 || { stop_holder; exit 1; }
run_direct F128 "1x${HOLD_F1}MB" || { stop_holder; exit 1; }
stop_holder; cool

start_holder F384 "$HOLD_F2" || exit 1
gpu_occupancy_gate F384 1 || { stop_holder; exit 1; }
run_direct F384 "1x${HOLD_F2}MB" || { stop_holder; exit 1; }
stop_holder

# ---------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------
banner "Evaluation"
absdev() { awk -v a="$1" -v r="$2" 'BEGIN{d=(a-r)/r*100; printf "%.3f",(d<0?-d:d)}'; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%+.3f",(b-a)/a*100}'; }
near() { awk -v d="$1" -v t="$2" 'BEGIN{exit !(d<=t)}'; }

banner "R1 -- is the renamed dispatcher + r4 binary inert on the production path?"
if [[ -n "${KMS[G0]:-}" ]]; then
  d="$(absdev "${KMS[G0]}" "$REF_G0_MS")"
  near "$d" 0.5 && pass "R1_G0_reproduces_production (${KMS[G0]}, ${d}% from 133,585)" \
    || fail "R1_G0_reproduces_production" "${KMS[G0]} is ${d}% off 133,585 -- the rename or the r4 binary is NOT inert; nothing below is interpretable"
fi

banner "R2 -- the zero-code-change route"
[[ -n "${KMS[Gh]:-}" ]] && { d="$(absdev "${KMS[Gh]}" "$REF_GH_MS")"; near "$d" 0.15 && pass "R2_Gh_hits_the_optimum (${KMS[Gh]}, ${d}% from 133,201)" || fail "R2_Gh_hits_the_optimum" "${KMS[Gh]} is ${d}% off 133,201"; }

banner "R3 -- the no-helper route"
[[ -n "${KMS[Gx]:-}" ]] && { d="$(absdev "${KMS[Gx]}" "$REF_GX_MS")"; near "$d" 0.15 && pass "R3_Gx_hits_the_optimum (${KMS[Gx]}, ${d}% from 133,190)" || fail "R3_Gx_hits_the_optimum" "${KMS[Gx]} is ${d}% off 133,190"; }

banner "R4 -- do the two routes reach the same device state?"
if [[ -n "${KMS[Gx]:-}" && -n "${KMS[Gh]:-}" ]]; then
  d="$(absdev "${KMS[Gx]}" "${KMS[Gh]}")"; info "Gx_vs_Gh" "$(pct "${KMS[Gh]}" "${KMS[Gx]}")%  free_mb ${FREEMB[Gx]:-?} vs ${FREEMB[Gh]:-?}"
  near "$d" 0.05 && pass "R4_routes_agree (${d}%)" \
    || fail "R4_routes_agree" "${d}% apart -- the model says an extra context is an extra context regardless of owner; it is missing something"
fi

banner "R5 -- does the deployment candidate replicate?"
if [[ -n "${KMS[Gxb]:-}" && -n "${KMS[Gx]:-}" ]]; then
  d="$(absdev "${KMS[Gxb]}" "${KMS[Gx]}")"
  near "$d" 0.05 && pass "R5_Gxb_replicates_Gx (${KMS[Gxb]}, ${d}%)" || fail "R5_Gxb_replicates_Gx" "${d}% apart"
fi

banner "R6 -- is +255 MB really the local optimum?"
for c in F128 F384; do
  [[ -z "${KMS[$c]:-}" ]] && continue
  info "$c" "kernel_ms=${KMS[$c]}  free_mb=${FREEMB[$c]:-?}  ($(pct "$REF_Y510_MS" "${KMS[$c]}")% vs the +255 MB point 133,181)"
  awk -v x="${KMS[$c]}" 'BEGIN{exit !(x<133150)}' && info "R6_new_minimum[$c]" "below 133,150 -- the optimum is NOT at +255 MB; a finer sweep pays"
done
if [[ -n "${KMS[F128]:-}" && -n "${KMS[F384]:-}" ]]; then
  awk -v a="${KMS[F128]}" -v b="${KMS[F384]}" -v r="$REF_Y510_MS" 'BEGIN{exit !(a>=r && b>=r)}' \
    && pass "R6_plus255MB_is_the_local_optimum (both neighbours are slower)" \
    || info "R6_optimum_moved" "a neighbour beat 133,181 -- sweep +160..+320 MB in r6"
fi

banner "CLAIM RULE"
cand=""; candv=""
for c in Gh Gx Gxb; do
  [[ -z "${KMS[$c]:-}" ]] && continue
  if [[ -z "$candv" ]] || awk -v a="${KMS[$c]}" -v b="$candv" 'BEGIN{exit !(a<b)}'; then cand="$c"; candv="${KMS[$c]}"; fi
done
if [[ -n "${KMS[G0]:-}" && -n "$candv" ]]; then
  g="$(awk -v a="${KMS[G0]}" -v b="$candv" 'BEGIN{printf "%.3f",(a-b)/a*100}')"
  info "measured_production_gain" "$cand = $candv vs G0 ${KMS[G0]}  ->  -${g}%  ($(awk -v a="${KMS[G0]}" -v b="$candv" 'BEGIN{printf "%.0f",a-b}') ms)"
  r5ok=0; [[ -n "${KMS[Gxb]:-}" && -n "${KMS[Gx]:-}" ]] && near "$(absdev "${KMS[Gxb]}" "${KMS[Gx]}")" 0.05 && r5ok=1
  if awk -v x="$g" 'BEGIN{exit !(x>=0.20)}' && [[ "$r5ok" == "1" ]]; then
    pass "CLAIM_ALLOWED (>=0.20% and replicated) -- record the gain and pick a route in r6"
  else
    info "CLAIM_WITHHELD" "gain ${g}% (rule: >=0.20%) or the replicate failed -- r4's -0.30% does not survive to the production path; production stays as it is"
  fi
fi
for c in G0 Gh Gx Gxb F128 F384; do [[ -n "${CLK[$c]:-}" ]] && info "clock[$c]" "${CLK[$c]}"; done

{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi -q -d CLOCK 2>&1; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs" 2>/dev/null || true
echo; python3 - "$TSV" <<'EOF' 2>/dev/null || cat "$TSV"
import csv,sys
rows=list(csv.reader(open(sys.argv[1]),delimiter='\t'))
w=[max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
for r in rows: print('  '.join(c.ljust(w[i]) for i,c in enumerate(r)))
EOF
TARBALL="${LOGDIR}.tar.gz"
tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed -- send $LOGDIR/"
echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; echo "tsv: $TSV"; echo "tarball: $TARBALL"
if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
echo "${REV} PASSED. Send $TARBALL for analysis."
exit 0
