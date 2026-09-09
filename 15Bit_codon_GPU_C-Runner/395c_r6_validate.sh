#!/usr/bin/env bash
# 395c_r6_validate.sh
#
# rev395c-r6 -- adopt the 413 ms that 395c-r5 measured, and verify it as a
#               production change rather than as an experiment.
#
# WHAT r5 ESTABLISHED (2026-09-09, all six pre-registered predictions held,
# every cell oracle-MATCH, every cell 1710 MHz)
#   G0   -g 21 21, nothing extra      133,585.359   free_mb 22,018
#   Gh   + one idle holder process    133,179.047   free_mb 21,762
#   Gx   + NQ_EXTRA_CTX=1             133,206.172   free_mb 21,762
#   Gxb  replicate of Gx              133,172.016   free_mb 21,762
#   |Gx - Gh| = 0.020%. A context in another process and a context in ours
#   are indistinguishable on the production path. Gain over G0: 0.309%,
#   413 ms. The claim rule fixed before that run (>=0.20%, replicated within
#   0.05%) was met. G0 reproduced the previous day's Gp to +0.0001%.
#
# WHAT r6 CHANGES -- exactly one executable line
#   The dispatch table entry becomes
#     CRunnerEntry(14,"./395c_r6_kernel_maxd14","NQ_EXTRA_CTX=1 ",...)
#   The treatment sits in env_prefix, NOT in the binary's default, so that
#   [crunner-config] env_prefix= and dispatch.log record the state of every
#   run. NQ_EXTRA_CTX stays 0 when unset, so direct runs are unchanged and
#   remain usable as a control.
#   The .cu is r4's, renamed: byte-identical below the first #include, gated
#   here by a sha256 of the whole code region, not just the kernel.
#
# CELLS -- alternating, so that any within-session drift cancels
#   Gb1  ./395c_r5Py_kernel_maxd14_final -g 21 21   (no treatment)
#   Gn1  ./395c_r6Py_kernel_maxd14_final -g 21 21   (env_prefix)
#   Gb2  r5 again
#   Gn2  r6 again
#   ~11 minutes. The r5 dispatcher is the baseline artifact because it is
#   the one that produced G0; using it avoids inventing a new control.
#
# PRE-REGISTERED (395c_r6_README_append.md; written before execution)
#   S1  |Gb1 - Gb2| <= 0.05%, and both within +-0.1% of 133,585, with
#       free_mb 22,018 and extra_ctx 0. The baseline is stable and is the
#       same state r5 measured.
#   S2  |Gn1 - Gn2| <= 0.05%, with free_mb 21,763 and extra_ctx 1.
#   S3  mean(Gn) <= mean(Gb) - 0.25%.
#   S4  dispatch.log shows env_prefix=NQ_EXTRA_CTX=1 and the CRunner log
#       shows [gpu-ctx] extra_ctx=1. Falsified if the prefix is present but
#       extra_ctx=0 reaches the binary: the table's env_prefix is not being
#       applied and the change is inert.
#
#   ADOPTION RULE, fixed in advance: adopt only if S1-S4 all hold AND the
#   measured gain is >= 0.25%. Otherwise revert the table entry to the r5
#   form. A gain that does not survive an alternating within-session A/B is
#   not a gain.
#
#   NOT COVERED HERE: correctness across other N. Run
#     SMALL_N=1 bash 395c_r6_validate.sh
#   once before adopting, which adds a `-g 5 15` correctness-only cell.
#   It is off by default because uncached small-N inputs make its duration
#   unpredictable and this harness is meant to stay at ~11 minutes.
#
# USAGE
#   STATIC_ONLY=1 bash 395c_r6_validate.sh
#                 bash 395c_r6_validate.sh          # ~13 min incl. builds

set -u

REV="395c_r6"
PY_SRC="${PY_SRC:-395c_r6Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-395c_r6Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-395c_r5Py_kernel_maxd14_final.py}"
BASE_PY_BIN="${BASE_PY_BIN:-395c_r5Py_kernel_maxd14_final}"
BASE_CU_BIN="${BASE_CU_BIN:-395c_r4_kernel_maxd14}"
BASE_CRLOG_DIR="${BASE_CRLOG_DIR:-395c_r5_crunner_logs}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-395c_r6_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-395c_r6_kernel_maxd14}"
PREV_CU="${PREV_CU:-395c_r4_kernel_maxd14.cu}"
CRLOG_DIR="${CRLOG_DIR:-395c_r6_crunner_logs}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
ORACLE="${ORACLE:-314666222712}"
IN_RAW="${IN_RAW:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
IN_PROD="${IN_PROD:-${IN_RAW}.sched394f.bin}"
EXPECTED_RECORDS="${EXPECTED_RECORDS:-2025282}"
MB_PROD="${MB_PROD:-800}"
REF_BASE_MS="${REF_BASE_MS:-133585}"
FREE_BASE="${FREE_BASE:-22018}"
FREE_NEW="${FREE_NEW:-21763}"
MIN_GAIN_PCT="${MIN_GAIN_PCT:-0.25}"
KERNEL_SHA_395C="${KERNEL_SHA_395C:-ebd7f523deb591347eab1a352a9555c2e4a6c0b4a456d80517322c933a86e68a}"
COOLDOWN="${COOLDOWN:-20}"
SMALL_N="${SMALL_N:-0}"
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
for f in "$PY_SRC" "$HELPER_SRC" "$CU_SRC" "$IN_RAW"; do
  [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"
done
[[ -x "$BASE_PY_BIN" ]] && pass "baseline_dispatcher_present[$BASE_PY_BIN] (the artifact that produced r5's G0)" \
  || fail "baseline_dispatcher_present" "$BASE_PY_BIN missing -- rebuild it from $PREV_PY; r6 needs it as the control"
[[ -x "$BASE_CU_BIN" ]] && pass "baseline_binary_present[$BASE_CU_BIN]" || fail "baseline_binary_present" "$BASE_CU_BIN missing -- the r5 table points at it"
if [[ -f "$IN_PROD" ]]; then
  sz="$(stat -c%s "$IN_PROD")"
  [[ "$sz" -eq $((EXPECTED_RECORDS*28)) ]] && pass "sched_input_present_and_sized[$((sz/28)) records]" || fail "sched_input_present_and_sized" "$IN_PROD is $sz bytes"
else fail "sched_input_present_and_sized" "$IN_PROD missing"; fi
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }

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

NOTE_LINES=$(awk '/^# =+$/{f=1} f&&/^#/{n++} END{print n+0}' "$PY_SRC")
if grep -q "^# ${REV//_/-} " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; then
  pass "py_revision_notes_present ($NOTE_LINES comment lines; the ${REV//_/-} record is in the source)"
else
  fail "py_revision_notes_present" "no '# ${REV//_/-} ...' note block (>=20 comment lines) in $PY_SRC -- the revision rationale, cells and pre-registered predictions belong in the source as '#' comment lines"
fi
NQ_=$(py_note_region_quote_count "$PY_SRC")
[[ $((NQ_ % 2)) -eq 0 ]] && pass "py_note_region_quotes_balanced ($NQ_ triple-quotes before the first import)" \
  || fail "py_note_region_quotes_balanced" "$NQ_ triple-quotes before the first import: a note slot does not close"

grep -qE '^REV_TAG:str="395c_r6"' "$CODE" && pass "source_rev_tag_is_395c_r6" || fail "source_rev_tag_is_395c_r6" "wrong REV_TAG"
grep -q 'CRunnerEntry(14,"./395c_r6_kernel_maxd14","NQ_EXTRA_CTX=1 "' "$CODE" \
  && pass "source_table_carries_the_treatment (binary + env_prefix NQ_EXTRA_CTX=1)" \
  || fail "source_table_carries_the_treatment" "the table entry is not the r6 form -- this revision has no effect"
[[ "$(grep -c 'CRunnerEntry(14,' "$CODE")" == "1" ]] && pass "source_single_maxd14_entry" || fail "source_single_maxd14_entry" "more than one maxd14 entry is live"
grep -qE '^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=800' "$CODE" && pass "source_default_max_blocks_800" || fail "source_default_max_blocks_800" "not 800"
grep -q '      if argc <= 4:' "$CODE" && [[ "$(grep -c '      if argc == 2:' "$CODE")" == "0" ]] && pass "source_defaults_apply_for_argc_le_4" || fail "source_defaults_apply_for_argc_le_4" "the defaults gate is not argc<=4"
grep -qE '^CRUNNER_INPUT_ORDER:str="sched"' "$CODE" && pass "source_input_order_sched" || fail "source_input_order_sched" "not sched"
grep -q 'NQ_MAX_BLOCKS={gpu_max_blocks} {entry_base37.env_prefix}' "$CODE" && pass "source_mode37_stride_coupling (and env_prefix is interpolated after it, so the treatment reaches the command line)" || fail "source_mode37_stride_coupling" "mode 37 does not pass NQ_MAX_BLOCKS"
grep -q '> {log_path} 2>&1' "$CODE" && pass "source_crunner_stderr_is_captured" || fail "source_crunner_stderr_is_captured" "crunner_run does not redirect stderr"
[[ "$(grep -c 'os.system(f"python3' "$CODE")" == "2" && "$(grep 'os.system(f"python3' "$CODE" | grep -vc '2>&1')" == "0" ]] && pass "source_external_tools_output_redirected" || fail "source_external_tools_output_redirected" "an external python3 os.system call is not redirected"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true)
  PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  if [[ "$PR_" == "3" && "$PA_" == "3" ]]; then
    pass "py_diff_fingerprint_vs_r5Py (removed=3 added=3 EXECUTABLE lines: VERSION_TAG, REV_TAG, table entry. Comments and notes are not counted -- annotate freely.)"
  else
    fail "py_diff_fingerprint_vs_r5Py" "removed=$PR_ added=$PA_, expected 3/3 executable lines"
    echo "      --- the actual CODE difference (first 40 lines) -------------------"
    diff "/tmp/${REV}_prev_code.py" "$CODE" | head -40 | sed 's/^/      /' | cut -c1-140
    echo "      -------------------------------------------------------------------"
  fi
else info "py_diff_fingerprint_vs_r5Py" "skipped ($PREV_PY absent)"; fi

# --- the .cu must be r4's, renamed and nothing else ---
cu_code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA_395C" ]] && pass "cu_kernel_region_sha_unchanged_since_395a_r2 (${KB:0:16}...)" || fail "cu_kernel_region_sha_unchanged_since_395a_r2" "got ${KB:0:16}..."
if [[ -f "$PREV_CU" ]]; then
  cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  CA="$(sha256sum < "/tmp/${REV}_prev_code.cu" | cut -d' ' -f1)"; CB="$(sha256sum < "/tmp/${REV}_cur_code.cu" | cut -d' ' -f1)"
  [[ "$CA" == "$CB" ]] && pass "cu_whole_code_region_identical_to_r4 (${CB:0:16}... -- r6 renamed the file and added a header note, nothing else)" \
    || fail "cu_whole_code_region_identical_to_r4" "the code region differs from $PREV_CU; r6 must be a rename"
else info "cu_whole_code_region_identical_to_r4" "skipped ($PREV_CU absent)"; fi
grep -q 'CUDA_VERSION >= 13000' "$CU_SRC" && pass "cuCtxCreate_v4_signature_handled" || fail "cuCtxCreate_v4_signature_handled" "no CUDA 13 branch"
grep -q 'CU_CHECK(cuCtxPopCurrent(&popped));' "$CU_SRC" && pass "extra_ctx_is_popped" || fail "extra_ctx_is_popped" "the created context is left current"
grep -q '    int MAX_BLOCKS = 800;' "$CU_SRC" && pass "cu_binary_default_max_blocks_800" || fail "cu_binary_default_max_blocks_800" "not 800"
[[ "$(grep -c 'uint64_t top0' "$CU_SRC")" == "0" ]] && pass "cu_395b_register_top_absent" || fail "cu_395b_register_top_absent" "395b's top0/top1 is back"
grep -q 'RISK NOTE' "$CU_SRC" && grep -q 'if (cur_avail != 0u) {' "$CU_SRC" && pass "cu_push_guard_untouched" || fail "cu_push_guard_untouched" "the 358/359 push guard form changed"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; nvidia-smi -q -d CLOCK 2>&1
  sha256sum "$PY_SRC" "$PREV_PY" "$HELPER_SRC" "$CU_SRC" "$PREV_CU" "$IN_RAW" "$IN_PROD" 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1

# ---------------------------------------------------------------------
# 2. Builds
# ---------------------------------------------------------------------
banner "Building $CU_SRC (with -lcuda) and $PY_SRC"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -o "$CU_BIN" "$CU_SRC" -lcuda 2>&1 | tee "$LOGDIR/05_nvcc_build.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build_succeeded[$CU_BIN]" || { fail "nvcc_build_succeeded" "no binary"; exit 1; }
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/06_codon_build.log"
[[ -x "$PY_BIN" ]] && pass "codon_build_succeeded[$PY_BIN]" || { fail "codon_build_succeeded" "see $LOGDIR/06_codon_build.log"; exit 1; }
head -c $((15488*28)) "$IN_RAW" > "/tmp/${REV}_probe.bin"
env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX "./$CU_BIN" "$NQ" "/tmp/${REV}_probe.bin" "/tmp/${REV}_probe_out.bin" > "$LOGDIR/05b_default_config_probe.log" 2>&1 || true
grep -q 'MAX_BLOCKS=800 stride=25600' "$LOGDIR/05b_default_config_probe.log" && pass "binary_default_config_is_800" || fail "binary_default_config_is_800" "see 05b log"
grep -q '\[gpu-ctx\] extra_ctx=0' "$LOGDIR/05b_default_config_probe.log" && pass "binary_default_extra_ctx_is_0 (direct runs stay unchanged and remain a control)" || fail "binary_default_extra_ctx_is_0" "see 05b log"

# ---------------------------------------------------------------------
# 3. Runs -- alternating baseline / treatment
# ---------------------------------------------------------------------
printf 'cell\tdispatcher\tbinary\textra_ctx\tkernel_ms\tfree_mb\ttotal_sum\tmatch\tstride_actual\tsm_clock_mhz\ttemp_c\tstart\n' > "$TSV"
declare -A KMS FREEMB CLK
SAMPLER_PID=""
start_sampler() { local out="$LOGDIR/clk_${1}.tsv"; : > "$out"
  ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits >> "$out" 2>/dev/null; sleep 5; done ) &
  SAMPLER_PID=$!; }
stop_sampler() { local cell="$1"
  [[ -n "$SAMPLER_PID" ]] && { kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; }; SAMPLER_PID=""
  CLK[$cell]="$(awk -F', *' 'NF>=6 && $2+0>0 && $6+0>50 {n++; s+=$2; if(min==""||$2<min)min=$2; p+=$4} END{if(n) printf "sm_mean=%.0f sm_min=%.0f power_mean=%.1fW n=%d", s/n, min, p/n, n; else print "no-samples"}' "$LOGDIR/clk_${cell}.tsv")"
  info "in_run_clock[$cell]" "${CLK[$cell]}"; }
record_row() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$TSV"; }
gpu_occupancy_gate() { local cell="$1" n
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null > "$LOGDIR/apps_before_${cell}.txt"
  n="$(grep -c '^[0-9]' "$LOGDIR/apps_before_${cell}.txt" || true)"
  [[ "$n" == "0" ]] && pass "gpu_idle_before[$cell]" || { fail "gpu_idle_before[$cell]" "$n process(es) already on the GPU -- would shift the result by up to 6%"; return 1; }; }
free_gate() { local cell="$1" want="$2" got="${FREEMB[$1]:-}"
  [[ -z "$got" || "$got" == "?" ]] && { info "free_mb[$cell]" "not captured"; return 0; }
  awk -v a="$got" -v b="$want" 'BEGIN{exit !((a-b<=3)&&(b-a<=3))}' && pass "free_mb_as_intended[$cell] ($got, expected $want)" \
    || fail "free_mb_as_intended[$cell]" "$got MiB free, expected $want"; }

run_cell() {  # cell  dispatcher_bin  crunner_bin_name  crlog_dir  expected_extra_ctx
  local cell="$1" disp="$2" cbin="$3" cdir="$4" xwant="$5"
  banner "cell $cell: ./$disp -g $NQ $NQ   (expects extra_ctx=$xwant)"
  local clk tmp start gcr total kms stride_act match freemb xseen
  clk="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1)"
  tmp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)"
  start="$(date -Is)"; gcr="$cdir/crunner_${cbin}_N${NQ}.log"; rm -f "$gcr"
  start_sampler "$cell"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX "./$disp" -g "$NQ" "$NQ" 2>&1 | tee "$LOGDIR/1_${cell}_console.log"
  stop_sampler "$cell"
  cp "$gcr" "$LOGDIR/1_${cell}_crunner.log" 2>/dev/null || true
  cp "$cdir/dispatch.log" "$LOGDIR/dispatch_after_${cell}.log" 2>/dev/null || true
  if [[ ! -f "$gcr" ]]; then fail "crunner_path_taken[$cell]" "no $gcr"; return 1; fi
  total="$(grep -o 'total_sum=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"
  kms="$(grep -o 'kernel_ms=[0-9.]*' "$gcr" | head -1 | cut -d= -f2)"
  stride_act="$(grep -o 'stride=[0-9]*' "$gcr" | tail -1 | cut -d= -f2)"
  freemb="$(grep -o 'free_mb=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"
  xseen="$(grep -o 'extra_ctx=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"
  match=0; grep -q '\[gpu-run-correctness\] MATCH' "$gcr" && match=1
  record_row "$cell" "$disp" "$cbin" "${xseen:-?}" "${kms:-?}" "${freemb:-?}" "${total:-?}" "$match" "${stride_act:-?}" "$clk" "$tmp" "$start"
  if [[ "$match" -eq 1 && "${total:-}" == "$ORACLE" ]]; then pass "oracle_match[$cell]"; else fail "oracle_match[$cell]" "total_sum='${total:-<none>}' match=$match"; return 1; fi
  [[ "${stride_act:-}" == "$((32*MB_PROD))" ]] && pass "stride_as_intended[$cell]" || { fail "stride_as_intended[$cell]" "got ${stride_act:-?}"; return 1; }
  [[ "${xseen:-}" == "$xwant" ]] && pass "extra_ctx_as_intended[$cell] (extra_ctx=$xseen)" \
    || { fail "extra_ctx_as_intended[$cell]" "the CRunner reported extra_ctx=${xseen:-?}, expected $xwant -- the table's env_prefix is not reaching the command line, so this revision is inert"; return 1; }
  KMS[$cell]="$kms"; FREEMB[$cell]="${freemb:-?}"
  info "cell[$cell]" "kernel_ms=$kms  free_mb=${freemb:-?}"
  return 0
}
cool() { echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; }

banner "4 cells, alternating: Gb1 Gn1 Gb2 Gn2  (~11 min)"
gpu_occupancy_gate Gb1 || exit 1; run_cell Gb1 "$BASE_PY_BIN" "$BASE_CU_BIN" "$BASE_CRLOG_DIR" 0 || exit 1; free_gate Gb1 "$FREE_BASE"; cool
gpu_occupancy_gate Gn1 || exit 1; run_cell Gn1 "$PY_BIN"      "$CU_BIN"      "$CRLOG_DIR"      1 || exit 1; free_gate Gn1 "$FREE_NEW";  cool
gpu_occupancy_gate Gb2 || exit 1; run_cell Gb2 "$BASE_PY_BIN" "$BASE_CU_BIN" "$BASE_CRLOG_DIR" 0 || exit 1; free_gate Gb2 "$FREE_BASE"; cool
gpu_occupancy_gate Gn2 || exit 1; run_cell Gn2 "$PY_BIN"      "$CU_BIN"      "$CRLOG_DIR"      1 || exit 1; free_gate Gn2 "$FREE_NEW"

if [[ "$SMALL_N" == "1" ]]; then
  banner "SMALL_N: ./$PY_BIN -g 5 15  (correctness only, no timing claim)"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX "./$PY_BIN" -g 5 15 2>&1 | tee "$LOGDIR/2_smalln_console.log"
  if grep -qiE 'mismatch|FAIL|error' "$LOGDIR/2_smalln_console.log"; then
    fail "small_n_correctness" "a mismatch or error appeared in -g 5 15 -- see $LOGDIR/2_smalln_console.log"
  else pass "small_n_correctness (-g 5 15 clean)"; fi
fi

# ---------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------
banner "Evaluation"
absdev() { awk -v a="$1" -v r="$2" 'BEGIN{d=(a-r)/r*100; printf "%.3f",(d<0?-d:d)}'; }
mean2() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f",(a+b)/2}'; }
near() { awk -v d="$1" -v t="$2" 'BEGIN{exit !(d<=t)}'; }

S1=0; S2=0; S3=0
if [[ -n "${KMS[Gb1]:-}" && -n "${KMS[Gb2]:-}" ]]; then
  d="$(absdev "${KMS[Gb2]}" "${KMS[Gb1]}")"; e1="$(absdev "${KMS[Gb1]}" "$REF_BASE_MS")"; e2="$(absdev "${KMS[Gb2]}" "$REF_BASE_MS")"
  info "baseline" "Gb1=${KMS[Gb1]}  Gb2=${KMS[Gb2]}  spread=${d}%  vs 133,585: ${e1}% / ${e2}%"
  if near "$d" 0.05 && near "$e1" 0.1 && near "$e2" 0.1; then S1=1; pass "S1_baseline_stable_and_matches_r5"
  else fail "S1_baseline_stable_and_matches_r5" "spread ${d}% (<=0.05%), deviations ${e1}%/${e2}% (<=0.1%)"; fi
fi
if [[ -n "${KMS[Gn1]:-}" && -n "${KMS[Gn2]:-}" ]]; then
  d="$(absdev "${KMS[Gn2]}" "${KMS[Gn1]}")"
  info "treatment" "Gn1=${KMS[Gn1]}  Gn2=${KMS[Gn2]}  spread=${d}%"
  if near "$d" 0.05; then S2=1; pass "S2_treatment_replicates"; else fail "S2_treatment_replicates" "spread ${d}% (<=0.05%)"; fi
fi
if [[ -n "${KMS[Gb1]:-}" && -n "${KMS[Gb2]:-}" && -n "${KMS[Gn1]:-}" && -n "${KMS[Gn2]:-}" ]]; then
  MB_="$(mean2 "${KMS[Gb1]}" "${KMS[Gb2]}")"; MN_="$(mean2 "${KMS[Gn1]}" "${KMS[Gn2]}")"
  GAIN="$(awk -v a="$MB_" -v b="$MN_" 'BEGIN{printf "%.3f",(a-b)/a*100}')"
  MS="$(awk -v a="$MB_" -v b="$MN_" 'BEGIN{printf "%.0f",a-b}')"
  info "means" "baseline=$MB_  treatment=$MN_  gain=${GAIN}%  (${MS} ms)"
  awk -v g="$GAIN" 'BEGIN{exit !(g>=0.25)}' && { S3=1; pass "S3_gain_reaches_the_threshold (${GAIN}% >= 0.25%)"; } \
    || fail "S3_gain_reaches_the_threshold" "${GAIN}% < 0.25%"
fi
S4=1
for c in Gn1 Gn2; do
  if [[ -f "$LOGDIR/dispatch_after_${c}.log" ]]; then
    grep -q 'env_prefix=NQ_EXTRA_CTX=1' "$LOGDIR/dispatch_after_${c}.log" && pass "S4_env_prefix_recorded[$c]" \
      || { fail "S4_env_prefix_recorded[$c]" "dispatch.log does not record env_prefix=NQ_EXTRA_CTX=1"; S4=0; }
  fi
done

banner "ADOPTION RULE"
if [[ "$S1" == "1" && "$S2" == "1" && "$S3" == "1" && "$S4" == "1" && "$FAIL" -eq 0 ]]; then
  pass "ADOPT (S1-S4 all held and the gain reached ${MIN_GAIN_PCT}%) -- 395c-r6 becomes the production dispatcher"
else
  info "DO_NOT_ADOPT" "S1=$S1 S2=$S2 S3=$S3 S4=$S4 -- revert the table entry to the r5 form (binary ./395c_r4_kernel_maxd14, env_prefix empty). A gain that does not survive an alternating within-session A/B is not a gain."
fi
for c in Gb1 Gn1 Gb2 Gn2; do [[ -n "${CLK[$c]:-}" ]] && info "clock[$c]" "${CLK[$c]}"; done

{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi -q -d CLOCK 2>&1; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs_r6" 2>/dev/null || true
cp -r "$BASE_CRLOG_DIR" "$LOGDIR/crunner_logs_base" 2>/dev/null || true
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
