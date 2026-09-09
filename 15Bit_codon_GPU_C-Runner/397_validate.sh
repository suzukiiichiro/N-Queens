#!/usr/bin/env bash
# 397_validate.sh
#
# rev397 -- first kernel change since 395a r2. Repack the DFS stack frame
#           from 16 bytes to 12, and test ONE hypothesis with it.
#
# WHAT 396 ESTABLISHED
#   ptxas   208 bytes stack frame, 0 spill stores, 0 spill loads, 37 registers
#           -> the frame is the declared array uint64_t stack[26], not spills
#   Occupancy (real N=21)  theoretical 33.33%, achieved 10.32%, 4.95 warps/SM
#           BLOCK=32 is one warp per block against a 16-block-per-SM limit;
#           Block Limit Registers is 48, so registers are NOT the cap
#   Scheduler  No Eligible 61.63%, 2.41 active warps per scheduler out of 12
#   SpeedOfLight, one-round proxy (NOT representative)
#           L1/TEX 52.96%, L2 46.73%, DRAM 16.09%, Compute (SM) 19.41%
#
# THE FORK
#   (a) VOLUME     local-memory bytes moved per push and pop
#   (b) DIVERGENCE threads sit at different depths, so warp addresses scatter
#                  and each request splits into many sectors regardless of
#                  how few bytes are live
#   397 attacks (a) alone: 25% fewer bytes per frame, with the number of local
#   accesses held at exactly two per push and two per pop. Request count and
#   divergence are untouched, so the two hypotheses are separated.
#
#   word A (uint64_t)  ld(21) | rd(21)<<21 | col(21)<<42
#   word B (uint32_t)  avail | depth<<27, unchanged
#   16 -> 12 bytes per frame; sectors per push/pop 16 -> 12.
#
# WHY THIS IS GATED HARDER THAN USUAL
#   The packing drops bits at or above 2^21 of ld and rd. The argument that
#   they are dead for N <= 21 is a reasoning argument about every use site,
#   and reasoning arguments are what this project gates rather than trusts.
#   So before any timing is read, both binaries run the FULL 2,025,282-record
#   input and their per-record outputs are compared byte for byte. If that
#   fails, the harness stops and no number is reported.
#
# PRE-REGISTERED (397_README_append.md; written before execution)
#   V1  ptxas: 396-r2 reports 208 bytes; 397 reports 140-168 bytes, spills
#       still 0/0, registers <= 42. If the frame did not shrink, the repack
#       did not take and nothing below is worth running.
#   V2  byte-for-byte equality of the per-record output over the full input,
#       both oracle MATCH. HARD GATE.
#   V3  if (a) holds: Gn <= Gb - 1.0%.
#       STATED PREDICTION: |Gn - Gb| <= 0.3%, i.e. (a) is REFUTED. 320
#       threads/SM x 208 B is 66.6 KB against a 128 KB L1, so capacity does
#       not look exceeded. Confidence is low, which is why this is worth
#       running: either outcome decides where 398 goes.
#   V4  a direct run with NQ_EXTRA_CTX=2 (three contexts, free_mb 21763)
#       lands within +-0.15% of 133,18x -- the production device state
#       reproduced without the dispatcher, which shortens every later
#       measurement by one dispatcher run.
#
# USAGE
#   STATIC_ONLY=1 bash 397_validate.sh
#                 bash 397_validate.sh          # ~20 min

set -u

REV="397"
PY_SRC="${PY_SRC:-397Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-397Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-396_r2Py_kernel_maxd14_final.py}"
BASE_PY_BIN="${BASE_PY_BIN:-396_r2Py_kernel_maxd14_final}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-397_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-397_kernel_maxd14}"
PREV_CU="${PREV_CU:-396_r2_kernel_maxd14.cu}"
BASE_CU_BIN="${BASE_CU_BIN:-396_r2_kernel_maxd14}"
CRLOG_DIR="${CRLOG_DIR:-397_crunner_logs}"
BASE_CRLOG_DIR="${BASE_CRLOG_DIR:-396_r2_crunner_logs}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
ORACLE="${ORACLE:-314666222712}"
IN_RAW="${IN_RAW:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
IN_PROD="${IN_PROD:-${IN_RAW}.sched394f.bin}"
EXPECTED_RECORDS="${EXPECTED_RECORDS:-2025282}"
MB_PROD="${MB_PROD:-800}"
REF_PROD_MS="${REF_PROD_MS:-133192.071}"
REF_3CTX_MS="${REF_3CTX_MS:-133185}"
FRAME_OLD="${FRAME_OLD:-208}"
FRAME_MIN="${FRAME_MIN:-140}"
FRAME_MAX="${FRAME_MAX:-168}"
REG_MAX="${REG_MAX:-42}"
KERNEL_SHA_395A="${KERNEL_SHA_395A:-ebd7f523deb591347eab1a352a9555c2e4a6c0b4a456d80517322c933a86e68a}"
GAIN_THRESHOLD="${GAIN_THRESHOLD:-1.0}"
NULL_BAND="${NULL_BAND:-0.3}"
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
for f in "$PY_SRC" "$HELPER_SRC" "$CU_SRC" "$PREV_CU" "$IN_RAW"; do
  [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"
done
if [[ -f "$IN_PROD" ]]; then
  sz="$(stat -c%s "$IN_PROD")"
  [[ "$sz" -eq $((EXPECTED_RECORDS*28)) ]] && pass "sched_input_present_and_sized[$((sz/28)) records]" || fail "sched_input_present_and_sized" "$IN_PROD is $sz bytes"
else fail "sched_input_present_and_sized" "$IN_PROD missing"; fi
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }

py_code_region() { python3 -c "
import sys
lines=open(sys.argv[1],encoding='utf-8').read().split('\n')
i=next(i for i,l in enumerate(lines) if l.startswith('import '))
sys.stdout.write('\n'.join(l for l in lines[i:] if not l.lstrip().startswith('#')))
" "$1"; }
py_note_quotes() { python3 -c "
import sys
lines=open(sys.argv[1],encoding='utf-8').read().split('\n')
i=next(i for i,l in enumerate(lines) if l.startswith('import '))
print('\n'.join(lines[:i]).count('\"'*3))
" "$1"; }
py_code_region "$PY_SRC" > "/tmp/${REV}_code_only.py"; CODE="/tmp/${REV}_code_only.py"
NOTE_LINES=$(awk '/^# =+$/{f=1} f&&/^#/{n++} END{print n+0}' "$PY_SRC")
{ grep -q "^# ${REV} " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" \
  || fail "py_revision_notes_present" "no '# ${REV} ...' note block (>=20 comment lines) in $PY_SRC"
NQ_=$(py_note_quotes "$PY_SRC"); [[ $((NQ_ % 2)) -eq 0 ]] && pass "py_note_region_quotes_balanced ($NQ_)" || fail "py_note_region_quotes_balanced" "$NQ_ triple-quotes"
grep -qE '^REV_TAG:str="397"' "$CODE" && pass "source_rev_tag_is_397" || fail "source_rev_tag_is_397" "wrong REV_TAG"
grep -q 'CRunnerEntry(14,"./397_kernel_maxd14","NQ_EXTRA_CTX=1 "' "$CODE" && pass "source_table_points_at_397_and_keeps_the_treatment" || fail "source_table_points_at_397_and_keeps_the_treatment" "table entry wrong"
grep -qE '^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=800' "$CODE" && pass "source_default_max_blocks_800" || fail "source_default_max_blocks_800" "not 800"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_396_r2Py (removed=3 added=3 EXECUTABLE lines)" \
    || { fail "py_diff_fingerprint_vs_396_r2Py" "removed=$PR_ added=$PA_, expected 3/3"; diff "/tmp/${REV}_prev_code.py" "$CODE" | head -30 | sed 's/^/      /'; }
else info "py_diff_fingerprint_vs_396_r2Py" "skipped"; fi

# --- the kernel DID change; check that it changed in the intended way only ---
cu_code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"; cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" != "$KERNEL_SHA_395A" ]] && pass "cu_kernel_region_changed_as_intended (new sha ${KB:0:16}..., was ${KERNEL_SHA_395A:0:16}... since 395a r2)" \
  || fail "cu_kernel_region_changed_as_intended" "the kernel region is unchanged -- the repack did not land"
CR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true)
CA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
[[ "$CR_" == "18" && "$CA_" == "58" ]] && pass "cu_diff_fingerprint_vs_396_r2 (removed=18 added=58: declaration, two guards, two pushes, one pop, the mask constant, the N guard)" \
  || { fail "cu_diff_fingerprint_vs_396_r2" "removed=$CR_ added=$CA_, expected 18/58"; diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | head -40 | sed 's/^/      /' | cut -c1-140; }
[[ "$(grep -c 'stack\[stack_ptr' "$CU_SRC")" == "0" ]] && pass "cu_old_stack_array_fully_replaced" || fail "cu_old_stack_array_fully_replaced" "a reference to the old stack[] remains"
grep -q 'if (N > (int64_t)MAXD14_PACK_BITS)' "$CU_SRC" && [[ "$(grep -c 'if (N > (int64_t)MAXD14_PACK_BITS)' "$CU_SRC")" == "2" ]] \
  && pass "cu_N_guard_in_both_mains (the 21-bit packing refuses N>21)" || fail "cu_N_guard_in_both_mains" "the N<=21 guard is missing from one of the two mains"
grep -q '    int MAX_BLOCKS = 800;' "$CU_SRC" && pass "cu_binary_default_max_blocks_800" || fail "cu_binary_default_max_blocks_800" "not 800"
grep -q 'RISK NOTE' "$CU_SRC" && grep -q 'if (cur_avail != 0u) {' "$CU_SRC" && pass "cu_push_guard_untouched" || fail "cu_push_guard_untouched" "the push guard form changed"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; sha256sum "$PY_SRC" "$CU_SRC" "$PREV_CU" "$IN_PROD" 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1

# ---------------------------------------------------------------------
# 2. V1 -- ptxas: did the frame actually shrink?
# ---------------------------------------------------------------------
banner "V1 ptxas: stack frame, spills, registers"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
ptxas_of() { "$NVCC" -O3 -arch="$ARCH" -Xptxas -v -c "$1" -o /dev/null 2>&1 | grep -A3 'kernel_dfs_iter_gpu_maxd14'; }
ptxas_of "$PREV_CU" > "$LOGDIR/02_ptxas_396_r2.txt" 2>&1
ptxas_of "$CU_SRC"  > "$LOGDIR/02_ptxas_397.txt" 2>&1
frame_of() { grep -o '[0-9]* bytes stack frame' "$1" | head -1 | cut -d' ' -f1; }
regs_of()  { grep -o 'Used [0-9]* registers' "$1" | head -1 | cut -d' ' -f2; }
spill_of() { grep -o '[0-9]* bytes spill stores' "$1" | head -1 | cut -d' ' -f1; }
FO="$(frame_of "$LOGDIR/02_ptxas_396_r2.txt")"; FN="$(frame_of "$LOGDIR/02_ptxas_397.txt")"
RO="$(regs_of  "$LOGDIR/02_ptxas_396_r2.txt")"; RN="$(regs_of  "$LOGDIR/02_ptxas_397.txt")"
SN="$(spill_of "$LOGDIR/02_ptxas_397.txt")"
info "ptxas[396-r2]" "frame=${FO:-?} B  registers=${RO:-?}"
info "ptxas[397]"    "frame=${FN:-?} B  registers=${RN:-?}  spill_stores=${SN:-?} B"
[[ "${FO:-0}" == "$FRAME_OLD" ]] && pass "V1_baseline_frame_is_${FRAME_OLD}B" || fail "V1_baseline_frame_is_${FRAME_OLD}B" "got ${FO:-?} -- the baseline is not what 396 measured"
{ [[ -n "${FN:-}" ]] && [[ "$FN" -ge "$FRAME_MIN" ]] && [[ "$FN" -le "$FRAME_MAX" ]]; } \
  && pass "V1_frame_shrank (${FO} -> ${FN} B, $(awk -v a="$FO" -v b="$FN" 'BEGIN{printf "%.1f",(a-b)/a*100}')% fewer bytes per thread)" \
  || { fail "V1_frame_shrank" "397 frame is ${FN:-?} B, expected ${FRAME_MIN}-${FRAME_MAX} -- the repack did not take; stopping before spending GPU time"; exit 1; }
[[ "${SN:-0}" == "0" ]] && pass "V1_no_spills_introduced" || fail "V1_no_spills_introduced" "397 spills ${SN} bytes"
{ [[ -n "${RN:-}" ]] && [[ "$RN" -le "$REG_MAX" ]]; } && pass "V1_registers_within_budget (${RN} <= $REG_MAX)" || fail "V1_registers_within_budget" "397 uses ${RN:-?} registers; above $REG_MAX would cut occupancy and confound the test"

# ---------------------------------------------------------------------
# 3. Builds
# ---------------------------------------------------------------------
banner "Building both CRunners and both dispatchers"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$CU_BIN" "$CU_SRC" -lcuda 2>&1 | tee "$LOGDIR/05_nvcc_397.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build[397]" || { fail "nvcc_build[397]" "no binary"; exit 1; }
if [[ ! -x "$BASE_CU_BIN" ]]; then
  "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$BASE_CU_BIN" "$PREV_CU" -lcuda 2>&1 | tee "$LOGDIR/05_nvcc_396_r2.log"
fi
[[ -x "$BASE_CU_BIN" ]] && pass "baseline_binary_present[$BASE_CU_BIN]" || { fail "baseline_binary_present" "cannot build $BASE_CU_BIN"; exit 1; }
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/06_codon_397.log"
[[ -x "$PY_BIN" ]] && pass "codon_build[397]" || { fail "codon_build[397]" "see log"; exit 1; }
if [[ ! -x "$BASE_PY_BIN" ]]; then "$CODON" build -release -o "$BASE_PY_BIN" "$PREV_PY" 2>&1 | tee "$LOGDIR/06_codon_396_r2.log"; fi
[[ -x "$BASE_PY_BIN" ]] && pass "baseline_dispatcher_present[$BASE_PY_BIN]" || { fail "baseline_dispatcher_present" "cannot build $BASE_PY_BIN"; exit 1; }
env -u NQ_MAX_BLOCKS -u NQ_EXTRA_CTX "./$CU_BIN" 22 "$IN_PROD" /tmp/${REV}_nguard.bin > "$LOGDIR/07_n_guard_probe.log" 2>&1 || true
grep -q 'N must be <= 21' "$LOGDIR/07_n_guard_probe.log" && pass "N_guard_refuses_N22_at_runtime" || fail "N_guard_refuses_N22_at_runtime" "N=22 was not refused; see $LOGDIR/07_n_guard_probe.log"

# ---------------------------------------------------------------------
# 4. V2 -- byte-for-byte equivalence on the FULL input. HARD GATE.
# ---------------------------------------------------------------------
printf 'cell\tpath\tbinary\textra_ctx\tkernel_ms\tfree_mb\ttotal_sum\tmatch\tstride_actual\tsm_clock_mhz\ttemp_c\tstart\n' > "$TSV"
declare -A KMS FREEMB
idle_gate() { local c="$1" n; n="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
  [[ "$n" == "0" ]] && pass "gpu_idle_before[$c]" || { fail "gpu_idle_before[$c]" "$n process(es) on the GPU"; return 1; }; }
run_direct() {  # cell binary extra_ctx outbin
  local cell="$1" bin="$2" x="$3" outb="$4"
  banner "cell $cell: ./$bin  NQ_EXTRA_CTX=$x  (full input)"
  local clk tmp start log
  clk="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1)"
  tmp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)"
  start="$(date -Is)"; log="$LOGDIR/1_${cell}.log"
  env -u NQ_PAD_MB NQ_EXTRA_CTX="$x" NQ_MAX_BLOCKS="$MB_PROD" "./$bin" "$NQ" "$IN_PROD" "$outb" "$ORACLE" 2>&1 | tee "$log"
  local total kms stride_act match freemb xseen
  total="$(grep -o 'total_sum=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  kms="$(grep -o 'kernel_ms=[0-9.]*' "$log" | head -1 | cut -d= -f2)"
  stride_act="$(grep -o 'stride=[0-9]*' "$log" | tail -1 | cut -d= -f2)"
  freemb="$(grep -o 'free_mb=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  xseen="$(grep -o 'extra_ctx=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  match=0; grep -q '\[gpu-run-correctness\] MATCH' "$log" && match=1
  printf '%s\tdirect\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$bin" "${xseen:-?}" "${kms:-?}" "${freemb:-?}" "${total:-?}" "$match" "${stride_act:-?}" "$clk" "$tmp" "$start" >> "$TSV"
  [[ "$match" -eq 1 && "${total:-}" == "$ORACLE" ]] && pass "oracle_match[$cell]" || { fail "oracle_match[$cell]" "total_sum='${total:-<none>}'"; return 1; }
  [[ "${xseen:-}" == "$x" ]] && pass "extra_ctx_as_intended[$cell]" || { fail "extra_ctx_as_intended[$cell]" "got ${xseen:-?}"; return 1; }
  KMS[$cell]="$kms"; FREEMB[$cell]="${freemb:-?}"; info "cell[$cell]" "kernel_ms=$kms free_mb=${freemb:-?}"; }
cool() { echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; }

banner "V2 equivalence: both binaries, full input, per-record output compared byte for byte"
idle_gate E0 || exit 1; run_direct E0 "$BASE_CU_BIN" 2 "/tmp/${REV}_out_base.bin" || exit 1; cool
idle_gate E1 || exit 1; run_direct E1 "$CU_BIN"      2 "/tmp/${REV}_out_new.bin"  || exit 1; cool
if cmp -s "/tmp/${REV}_out_base.bin" "/tmp/${REV}_out_new.bin"; then
  pass "V2_per_record_output_identical ($EXPECTED_RECORDS records, byte for byte)"
else
  fail "V2_per_record_output_identical" "396-r2 and 397 differ on the full input. The argument that bits >= 2^21 of ld and rd are dead is WRONG. No timing is reported; revert the packing or keep ld/rd at full width."
  cmp "/tmp/${REV}_out_base.bin" "/tmp/${REV}_out_new.bin" | head -3 | sed 's/^/      /'
  echo "OK=$PASS FAIL=$FAIL"; exit 1
fi
d="$(awk -v a="${KMS[E0]}" -v r="$REF_3CTX_MS" 'BEGIN{x=(a-r)/r*100;printf "%.3f",(x<0?-x:x)}')"
awk -v d="$d" 'BEGIN{exit !(d<=0.15)}' && pass "V4_direct_with_two_extra_contexts_reproduces_production (${KMS[E0]}, ${d}% from $REF_3CTX_MS)" \
  || info "V4_not_reproduced" "${KMS[E0]} is ${d}% from $REF_3CTX_MS -- the dispatcher-free shortcut is not equivalent; keep using -g for timing"
info "V2_by_product_delta" "E1 vs E0 (direct, 3 contexts): $(awk -v a="${KMS[E0]}" -v b="${KMS[E1]}" 'BEGIN{printf "%+.3f",(b-a)/a*100}')%"

# ---------------------------------------------------------------------
# 5. V3 -- alternating A/B on the production path
# ---------------------------------------------------------------------
run_dispatch() {  # cell dispatcher crunner crlogdir
  local cell="$1" disp="$2" cbin="$3" cdir="$4"
  banner "cell $cell: ./$disp -g $NQ $NQ"
  local clk tmp start gcr total kms stride_act match freemb xseen
  clk="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1)"
  tmp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)"
  start="$(date -Is)"; gcr="$cdir/crunner_${cbin}_N${NQ}.log"; rm -f "$gcr"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX "./$disp" -g "$NQ" "$NQ" 2>&1 | tee "$LOGDIR/1_${cell}_console.log"
  cp "$gcr" "$LOGDIR/1_${cell}_crunner.log" 2>/dev/null || true
  [[ -f "$gcr" ]] || { fail "crunner_path_taken[$cell]" "no $gcr"; return 1; }
  total="$(grep -o 'total_sum=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"
  kms="$(grep -o 'kernel_ms=[0-9.]*' "$gcr" | head -1 | cut -d= -f2)"
  stride_act="$(grep -o 'stride=[0-9]*' "$gcr" | tail -1 | cut -d= -f2)"
  freemb="$(grep -o 'free_mb=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"
  xseen="$(grep -o 'extra_ctx=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"
  match=0; grep -q '\[gpu-run-correctness\] MATCH' "$gcr" && match=1
  printf '%s\tdispatch\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$cbin" "${xseen:-?}" "${kms:-?}" "${freemb:-?}" "${total:-?}" "$match" "${stride_act:-?}" "$clk" "$tmp" "$start" >> "$TSV"
  [[ "$match" -eq 1 && "${total:-}" == "$ORACLE" ]] && pass "oracle_match[$cell]" || { fail "oracle_match[$cell]" "total_sum='${total:-<none>}'"; return 1; }
  KMS[$cell]="$kms"; FREEMB[$cell]="${freemb:-?}"; info "cell[$cell]" "kernel_ms=$kms free_mb=${freemb:-?}"; }

banner "V3 alternating A/B on the -g path: Gb1 Gn1 Gb2 Gn2"
idle_gate Gb1 || exit 1; run_dispatch Gb1 "$BASE_PY_BIN" "$BASE_CU_BIN" "$BASE_CRLOG_DIR" || exit 1; cool
idle_gate Gn1 || exit 1; run_dispatch Gn1 "$PY_BIN"      "$CU_BIN"      "$CRLOG_DIR"      || exit 1; cool
idle_gate Gb2 || exit 1; run_dispatch Gb2 "$BASE_PY_BIN" "$BASE_CU_BIN" "$BASE_CRLOG_DIR" || exit 1; cool
idle_gate Gn2 || exit 1; run_dispatch Gn2 "$PY_BIN"      "$CU_BIN"      "$CRLOG_DIR"      || exit 1

banner "Evaluation"
mean2() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f",(a+b)/2}'; }
if [[ -n "${KMS[Gb1]:-}" && -n "${KMS[Gb2]:-}" && -n "${KMS[Gn1]:-}" && -n "${KMS[Gn2]:-}" ]]; then
  SB="$(awk -v a="${KMS[Gb1]}" -v b="${KMS[Gb2]}" 'BEGIN{x=(b-a)/a*100;printf "%.3f",(x<0?-x:x)}')"
  SN2="$(awk -v a="${KMS[Gn1]}" -v b="${KMS[Gn2]}" 'BEGIN{x=(b-a)/a*100;printf "%.3f",(x<0?-x:x)}')"
  MB_="$(mean2 "${KMS[Gb1]}" "${KMS[Gb2]}")"; MN_="$(mean2 "${KMS[Gn1]}" "${KMS[Gn2]}")"
  DELTA="$(awk -v a="$MB_" -v b="$MN_" 'BEGIN{printf "%.3f",(b-a)/a*100}')"
  info "baseline"  "Gb1=${KMS[Gb1]} Gb2=${KMS[Gb2]} spread=${SB}%  mean=$MB_"
  info "treatment" "Gn1=${KMS[Gn1]} Gn2=${KMS[Gn2]} spread=${SN2}%  mean=$MN_"
  info "delta"     "${DELTA}%  ($(awk -v a="$MB_" -v b="$MN_" 'BEGIN{printf "%+.0f",b-a}') ms)"
  awk -v a="$SB" -v b="$SN2" 'BEGIN{exit !(a<=0.05 && b<=0.05)}' && pass "replicates_within_noise (both spreads <=0.05%)" || fail "replicates_within_noise" "spreads ${SB}% / ${SN2}%"
  if awk -v d="$DELTA" -v t="$GAIN_THRESHOLD" 'BEGIN{exit !(d <= -t)}'; then
    pass "V3_HYPOTHESIS_A_CONFIRMED (${DELTA}%) -- local-memory byte volume is what costs"
    info "next" "398 pushes further on volume: a tighter packing, and revisit BLOCK size now that the frame is smaller"
  elif awk -v d="$DELTA" -v n="$NULL_BAND" 'BEGIN{x=(d<0?-d:d); exit !(x<=n)}'; then
    pass "V3_HYPOTHESIS_A_REFUTED (${DELTA}%, inside the +-${NULL_BAND}% null band) -- 25% fewer local bytes buys nothing"
    info "next" "398 goes after (b): the divergence of stack addresses across a warp. Measure l1tex mem_local sectors per request on a representative N=20 workload before changing anything."
  else
    info "V3_inconclusive" "${DELTA}% falls between the null band and the gain threshold -- report it, do not act on it"
  fi
fi
for c in E0 E1 Gb1 Gn1 Gb2 Gn2; do [[ -n "${FREEMB[$c]:-}" ]] && info "free_mb[$c]" "${FREEMB[$c]}"; done
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs_397" 2>/dev/null || true
echo; python3 - "$TSV" <<'EOF' 2>/dev/null || cat "$TSV"
import csv,sys
rows=list(csv.reader(open(sys.argv[1]),delimiter='\t'))
w=[max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
for r in rows: print('  '.join(c.ljust(w[i]) for i,c in enumerate(r)))
EOF
TARBALL="${LOGDIR}.tar.gz"
tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; echo "tarball: $TARBALL"
if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
echo "${REV} PASSED. Send $TARBALL for analysis."
exit 0
