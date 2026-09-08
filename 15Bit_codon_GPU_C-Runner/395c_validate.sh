#!/usr/bin/env bash
# 395c_validate.sh
#
# rev395c -- (1) REVERT 395b's kernel change (kernel region = 395a r2);
#            (2) explain the dispatcher-vs-direct ~4%.
#
# 395b RESULTS (2026-09-08, all oracle-MATCH):
#   I0  cmp 394f_input_L3_sched.bin vs ...maxd14only_363.bin.sched394f.bin: IDENTICAL
#   I1  395a bin, direct, prod sched, 800   139,166      I2 (394f L3 file) 139,181
#   K1  395b bin, direct, prod sched, 800   144,032  (+3.5% vs I1)  <- REGRESSION
#   G   395b bin via dispatcher (-g 21 21)  138,650  (-3.7% vs K1: the path effect again)
#   A   395b bin via dispatcher, 484        201,705  (+5.2% vs 395a's 191,780)
# => the two sched files are the same; the ~4% is the INVOCATION PATH, not
#    the file. And the register-resident top made the kernel slower: two new
#    if(save_sp!=0) guards around push/pop = a control-flow shape change in
#    the hottest region, the class that has now failed 7/7 times.
#
# THIS REVISION
#   kernel: 395a r2 restored (sha-checked); host default 800 kept (policy).
#   the 4%: same binary, same file, same stride --
#     D0  direct, nothing else on the GPU                         (~139.2 expected)
#     D1  direct, with 395c_ctx_holder (context only, 0 MB) alive
#     D2  direct, with 395c_ctx_holder (context + 2048 MB touched) alive
#     G   via the dispatcher: ./395cPy -g 21 21                   (~133.6 expected)
#   A background sampler records SM clock / power / temperature every 5 s
#   DURING each cell (nvidia-smi), so a clock difference cannot hide.
#   PRE-REGISTERED (395c_README_append.md):
#     G within 1.5% of 133,577 (395a's G)   -> the revert restores 395a
#     D0 within 1% of 139,166 (395b I1)
#     H_ctx  (stated): D1 <= D0 - 3%  -> the presence of another CUDA context
#            is the cause; D2 vs D1 then tells whether device-memory
#            placement (H_place) matters on top: |D2-D1| >= 1.5% => yes
#     H_clock: mean in-run SM clock differs between D0 and G by >= 2%
#     falsified (H_ctx) if D1 within +-1% of D0 -> the cause is something the
#            Codon parent does beyond holding a context (env, cwd, stdio,
#            scheduling) -> 395c-r2 bisects the dispatcher command line
#
# USAGE
#   STATIC_ONLY=1 bash 395c_validate.sh
#                 bash 395c_validate.sh          # CPU equiv -> D0 D1 D2 G A (~15 min)

set -u

REV="395c"
PY_SRC="${PY_SRC:-395cPy_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-395cPy_kernel_maxd14_final}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
FILTER_SCRIPT="${FILTER_SCRIPT:-363_filter_maxd14_only.py}"
PERMUTE="${PERMUTE:-394f_permute_soa7.py}"
CU_SRC="${CU_SRC:-395c_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-395c_kernel_maxd14}"
PREV_CU="${PREV_CU:-395a_kernel_maxd14.cu}"
HOLDER_SRC="${HOLDER_SRC:-395c_ctx_holder.cu}"
HOLDER_BIN="${HOLDER_BIN:-395c_ctx_holder}"
HOLDER_MB="${HOLDER_MB:-2048}"
PREV_BIN="${PREV_BIN:-395a_kernel_maxd14}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
ORACLE="${ORACLE:-314666222712}"
IN_RAW="${IN_RAW:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
IN_L3="${IN_L3:-394f_input_L3_sched.bin}"
MB_PROD="${MB_PROD:-800}"
REF_G_MS="${REF_G_MS:-133577}"       # 395a G (production sched, 800): 133,580 / 133,574
REF_D0_MS="${REF_D0_MS:-139166}"     # 395b I1 (395a bin, direct, prod sched, 800)
IN_PROD="${IN_PROD:-${IN_RAW}.sched394f.bin}"
REF_A_MS="${REF_A_MS:-191781}"       # 395a A (production sched, 484)
CPU_CHECK_RECORDS="${CPU_CHECK_RECORDS:-2048}"
NCU_STAGE="${NCU:-0}"
FULL_G="${FULL_G:-0}"
ORACLE22="${ORACLE22:-2691008701644}"
COOLDOWN="${COOLDOWN:-20}"
STATIC_ONLY="${STATIC_ONLY:-0}"
CRLOG_DIR="${REV}_crunner_logs"

NCU_BIN="${NCU_BIN:-$(command -v ncu 2>/dev/null)}"
[[ -z "$NCU_BIN" && -x /usr/local/cuda/bin/ncu ]] && NCU_BIN="/usr/local/cuda/bin/ncu"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"
TSV="$LOGDIR/${REV}_results.tsv"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }

if [[ "$NCU_STAGE" == "1" ]]; then
  sudo -n true 2>/dev/null && pass "sudo_noninteractive_available" || { fail "sudo_noninteractive_available" "NCU=1 needs sudo"; exit 1; }
fi

# ---------------------------------------------------------------------
# 1. Static
# ---------------------------------------------------------------------
for f in "$PY_SRC" "$HELPER_SRC" "$FILTER_SCRIPT" "$PERMUTE" "$CU_SRC" "$IN_RAW" "$HOLDER_SRC"; do
  [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"
done
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }

python3 - "$PY_SRC" > "/tmp/${REV}_code_only.py" <<'EOF'
import re,sys
s=open(sys.argv[1],encoding='utf-8').read()
m=re.search(r'"""',s); e=s.index('"""',m.end()); print(s[:m.start()]+s[e+3:])
EOF
CODE="/tmp/${REV}_code_only.py"
grep -qE '^REV_TAG:str="395c"' "$CODE" && pass "source_rev_tag_is_395c" || fail "source_rev_tag_is_395c" "wrong REV_TAG"
grep -qE '^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=800' "$CODE" && pass "source_default_max_blocks_800" || fail "source_default_max_blocks_800" "not 800"
grep -q '      if argc <= 4:' "$CODE" && [[ "$(grep -c '      if argc == 2:' "$CODE")" == "0" ]] && pass "source_defaults_apply_for_argc_le_4 (-g and -g nmin nmax both get the confirmed parameters)" || fail "source_defaults_apply_for_argc_le_4" "the A10G_FINAL defaults gate is not argc<=4"
grep -qE '^CRUNNER_INPUT_ORDER:str="sched"' "$CODE" && pass "source_input_order_sched" || fail "source_input_order_sched" "not sched"
grep -q 'ensure_crunner_input_bin(N,stream_fname,gpu_log_level,CRUNNER_INPUT_ORDER)' "$CODE" && pass "source_mode37_passes_order" || fail "source_mode37_passes_order" "mode 37 does not pass the order"
grep -q 'NQ_MAX_BLOCKS={gpu_max_blocks} {entry_base37.env_prefix}' "$CODE" && pass "source_mode37_stride_coupling" || fail "source_mode37_stride_coupling" "mode 37 does not pass NQ_MAX_BLOCKS"
[[ "$(grep -c 'ensure_crunner_input_bin(N,shaped_fname,gpu_log_level)' "$CODE")" == "1" ]] && pass "source_mode39_unchanged_raw_order" || fail "source_mode39_unchanged_raw_order" "mode 39 call changed"
grep -q '^def _ensure_crunner_filtered_bin(' "$CODE" && pass "source_389_three_stage_factored" || fail "source_389_three_stage_factored" "helper missing"
grep -qE 'if not \(bench_mode==0 .*bench_mode==39\):' "$CODE" && pass "source_bench_mode_39_in_cli_whitelist" || fail "source_bench_mode_39_in_cli_whitelist" "39 missing"
grep -q 'CRunnerEntry(14,"./395c_kernel_maxd14",""' "$CODE" && pass "source_dispatch_table_references_395c_binary" || fail "source_dispatch_table_references_395c_binary" "table wrong"
[[ "$(grep -c 'os.system(f"python3' "$CODE")" == "2" && "$(grep 'os.system(f"python3' "$CODE" | grep -vc '2>&1')" == "0" ]] && pass "source_external_tools_output_redirected (r3: 363 filter + 394f permute)" || fail "source_external_tools_output_redirected" "an external python3 os.system call is not redirected -- bare -g would leak its output"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF

# .cu: the kernel region (process_one_task .. end of the __global__ kernel) must be
# byte-identical to 395a r2; the ONLY code-region difference is the host default 800.
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
if [[ -f "$PREV_CU" ]]; then
  awk 'f||/^#include/{f=1;print}' "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  awk 'f||/^#include/{f=1;print}' "$CU_SRC"  > "/tmp/${REV}_cur_code.cu"
  KA=$(extract_kernel "/tmp/${REV}_prev_code.cu" | sha256sum | cut -d' ' -f1); KB=$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)
  [[ -n "$KA" && "$KA" == "$KB" ]] && pass "cu_kernel_region_identical_to_395a_r2 (${KB:0:16}...)" || fail "cu_kernel_region_identical_to_395a_r2" "395b was not fully reverted"
  NR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true); NA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
  [[ "$NR_" == "3" && "$NA_" == "6" ]] && pass "cu_diff_fingerprint_vs_395a (removed=3 added=6: host default 800 only)" || fail "cu_diff_fingerprint_vs_395a" "removed=$NR_ added=$NA_, expected 3/6"
  grep -q '    int MAX_BLOCKS = 800;' "$CU_SRC" && pass "cu_binary_default_max_blocks_800" || fail "cu_binary_default_max_blocks_800" "not 800"
  [[ "$(grep -c 'uint64_t top0' "$CU_SRC")" == "0" ]] && pass "cu_395b_register_top_absent" || fail "cu_395b_register_top_absent" "395b's top0/top1 still present"
else info "cu_kernel_region_identical_to_395a_r2" "skipped ($PREV_CU absent)"; fi
grep -q 'RISK NOTE' "$CU_SRC" && grep -q 'if (cur_avail != 0u) {' "$CU_SRC" && pass "cu_push_guard_untouched" || fail "cu_push_guard_untouched" "the 358/359 push guard form changed"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; nvidia-smi -q -d CLOCK 2>&1
  sha256sum "$PY_SRC" "$HELPER_SRC" "$FILTER_SCRIPT" "$PERMUTE" "$CU_SRC" "$IN_RAW" 2>&1; [[ -f "$PREV_CU" ]] && sha256sum "$PREV_CU"; } > "$LOGDIR/00_env_pre.txt" 2>&1

# ---------------------------------------------------------------------
# 2. CPU-harness equivalence gate (no GPU): 395a vs 395c on real records (must be identical: same kernel)
# ---------------------------------------------------------------------
banner "CPU-harness equivalence: process_one_task() 395a vs 395c on the first $CPU_CHECK_RECORDS real records"
if [[ -f "$PREV_CU" ]] && command -v "$GCC" >/dev/null 2>&1; then
  "$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_prev" "$PREV_CU" -lm 2>"$LOGDIR/03_gcc_prev.log" && "$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"$LOGDIR/03_gcc_cur.log"
  if [[ -x "/tmp/${REV}_cpu_prev" && -x "/tmp/${REV}_cpu_cur" ]]; then
    head -c $((CPU_CHECK_RECORDS*28)) "$IN_RAW" > "/tmp/${REV}_cpu_in.bin"
    export OMP_NUM_THREADS="$(nproc)"
    "/tmp/${REV}_cpu_prev" "$NQ" "/tmp/${REV}_cpu_in.bin" "/tmp/${REV}_cpu_prev.bin" 2>&1 | tail -1 | tee "$LOGDIR/04_cpu_prev.log"
    "/tmp/${REV}_cpu_cur"  "$NQ" "/tmp/${REV}_cpu_in.bin" "/tmp/${REV}_cpu_cur.bin"  2>&1 | tail -1 | tee "$LOGDIR/04_cpu_cur.log"
    if cmp -s "/tmp/${REV}_cpu_prev.bin" "/tmp/${REV}_cpu_cur.bin"; then pass "cpu_harness_per_record_results_identical ($CPU_CHECK_RECORDS records)"; else fail "cpu_harness_per_record_results_identical" "395a and 395c differ on CPU -- the revert is not clean; stopping before the GPU"; exit 1; fi
  else fail "cpu_harness_build" "see $LOGDIR/03_gcc_*.log"; exit 1; fi
else info "cpu_harness_equivalence" "skipped ($PREV_CU or gcc absent)"; fi

# ---------------------------------------------------------------------
# 3. Builds
# ---------------------------------------------------------------------
banner "Building $CU_SRC, $PREV_CU (control) and $PY_SRC"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -o "$CU_BIN" "$CU_SRC" 2>&1 | tee "$LOGDIR/05_nvcc_build.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build_succeeded[$CU_BIN]" || { fail "nvcc_build_succeeded" "no binary"; exit 1; }
"$NVCC" -O2 -arch="$ARCH" -o "$HOLDER_BIN" "$HOLDER_SRC" 2>&1 | tee "$LOGDIR/05a_nvcc_holder.log"
[[ -x "$HOLDER_BIN" ]] && pass "ctx_holder_build_succeeded" || { fail "ctx_holder_build_succeeded" "no $HOLDER_BIN"; exit 1; }
# policy check on the binary itself: with NQ_MAX_BLOCKS unset it must run the confirmed config (800x32)
head -c $((15488*28)) "$IN_RAW" > "/tmp/${REV}_probe.bin"
env -u NQ_MAX_BLOCKS "./$CU_BIN" "$NQ" "/tmp/${REV}_probe.bin" "/tmp/${REV}_probe_out.bin" > "$LOGDIR/05b_default_config_probe.log" 2>&1 || true
grep -q 'MAX_BLOCKS=800 stride=25600' "$LOGDIR/05b_default_config_probe.log" && pass "binary_default_config_is_800 (NQ_MAX_BLOCKS unset)" || fail "binary_default_config_is_800" "expected '[gpu-config] BLOCK=32 MAX_BLOCKS=800 stride=25600' -- see $LOGDIR/05b_default_config_probe.log"
[[ "$(grep -o 'total_sum=[0-9]*' "$LOGDIR/05b_default_config_probe.log" | head -1 | cut -d= -f2)" == "2196649880" ]] && pass "head_slice_total_reproduces (2196649880 at 800x32 too)" || info "head_slice_total_reproduces" "head-slice total differs from 2196649880 -- expected only if the slice is not the 394a one"
if [[ -f "$PREV_CU" && ! -x "$PREV_BIN" ]]; then "$NVCC" -O3 -arch="$ARCH" -o "$PREV_BIN" "$PREV_CU" 2>&1 | tee "$LOGDIR/05_nvcc_build_prev.log"; fi
[[ -x "$PREV_BIN" ]] && pass "control_binary_present[$PREV_BIN]" || info "control_binary_present" "$PREV_BIN absent -- K0 will be skipped"
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/06_codon_build.log"
[[ -x "$PY_BIN" ]] && pass "codon_build_succeeded" || { fail "codon_build_succeeded" "see $LOGDIR/06_codon_build.log"; exit 1; }
if [[ -f "$IN_L3" && "$(stat -c%s "$IN_L3")" -eq $((2025282*28)) ]]; then info "l3_input" "reusing $IN_L3"; else python3 "$PERMUTE" sched "$IN_RAW" "$IN_L3" 2>&1 | tee "$LOGDIR/07_permute_L3.log"; fi
[[ -f "$IN_L3" ]] && pass "l3_input_present" || { fail "l3_input_present" "missing"; exit 1; }

# ---------------------------------------------------------------------
# 4. Runs
# ---------------------------------------------------------------------
printf 'cell\tpath\tbinary\tinput\tmax_blocks\tstride_actual\ttotal_sum\tmatch\tkernel_ms\tsm_clock_mhz\ttemp_c\tstart\n' > "$TSV"
declare -A KMS CLK
SAMPLER_PID=""
start_sampler() {  # $1 = cell; samples SM clock / power / temp every 5 s into the log dir
  local out="$LOGDIR/clk_${1}.tsv"; : > "$out"
  ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits >> "$out" 2>/dev/null; sleep 5; done ) &
  SAMPLER_PID=$!
}
stop_sampler() {   # $1 = cell; prints mean/min SM clock and mean power over the cell
  [[ -n "$SAMPLER_PID" ]] && { kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; }; SAMPLER_PID=""
  CLK[$1]="$(awk -F', *' 'NF>=6 && $2+0>0 {n++; s+=$2; if(min==""||$2<min)min=$2; p+=$4} END{if(n) printf "sm_mean=%.0f sm_min=%.0f power_mean=%.0fW n=%d", s/n, min, p/n, n; else print "no-samples"}' "$LOGDIR/clk_${1}.tsv")"
  info "in_run_clock[$1]" "${CLK[$1]}"
}
record_row() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$TSV"; }
run_direct() {   # cell bin input mb
  local cell="$1"; local bin="$2"; local src="$3"; local mb="$4"
  banner "cell $cell: direct ./$bin on $(basename "$src" | cut -c1-60) NQ_MAX_BLOCKS=$mb"
  local clk tmp start log total kms stride_act match
  clk="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  tmp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  start="$(date -Is)"; log="$LOGDIR/1_${cell}.log"
  start_sampler "$cell"
  NQ_MAX_BLOCKS="$mb" "./$bin" "$NQ" "$src" "/tmp/${REV}_${cell}.bin" "$ORACLE" 2>&1 | tee "$log"
  stop_sampler "$cell"
  total="$(grep -o 'total_sum=[0-9]*' "$log" | head -1 | cut -d= -f2)"; kms="$(grep -o 'kernel_ms=[0-9.]*' "$log" | head -1 | cut -d= -f2)"
  stride_act="$(grep -o 'stride=[0-9]*' "$log" | tail -1 | cut -d= -f2)"; match=0; grep -q '\[gpu-run-correctness\] MATCH' "$log" && match=1
  record_row "$cell" direct "$bin" "$(basename "$src")" "$mb" "${stride_act:-?}" "${total:-?}" "$match" "${kms:-?}" "$clk" "$tmp" "$start"
  if [[ "$match" -eq 1 && "${total:-}" == "$ORACLE" ]]; then pass "oracle_match[$cell]"; else fail "oracle_match[$cell]" "total_sum='${total:-<none>}' match=$match"; return 1; fi
  [[ "${stride_act:-}" == "$((32*mb))" ]] && pass "stride_as_intended[$cell]" || { fail "stride_as_intended[$cell]" "got ${stride_act:-?}"; return 1; }
  KMS[$cell]="$kms"; info "cell[$cell]" "kernel_ms=$kms"; return 0
}
run_dispatch() {   # cell "args..." expect_mb expect_input_suffix
  local cell="$1"; local args="$2"; local emb="$3"; local esuf="$4"
  banner "cell $cell: ./$PY_BIN $args   (dispatcher path${QUIET_EXPECTED:+, console must stay quiet})"
  local clk tmp start log crlog total kms stride_act match
  clk="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  tmp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  start="$(date -Is)"; log="$LOGDIR/1_${cell}_console.log"
  rm -f "$CRLOG_DIR/crunner_${CU_BIN}_N${NQ}.log"
  start_sampler "$cell"
  # shellcheck disable=SC2086
  env -u NQ_MAX_BLOCKS "./$PY_BIN" $args 2>&1 | tee "$log"
  stop_sampler "$cell"
  crlog="$CRLOG_DIR/crunner_${CU_BIN}_N${NQ}.log"; cp "$crlog" "$LOGDIR/1_${cell}_crunner.log" 2>/dev/null || true; cp "$CRLOG_DIR/dispatch.log" "$LOGDIR/dispatch_after_${cell}.log" 2>/dev/null || true
  cp "$CRLOG_DIR/permute_sched_N${NQ}.log" "$LOGDIR/permute_sched_N${NQ}_after_${cell}.log" 2>/dev/null || true
  if [[ ! -f "$crlog" ]]; then
    fail "crunner_path_taken[$cell]" "no $crlog was written: the dispatcher did not run the CRunner (bench_mode not 37/39? a non-CRunner path?). Console row: $(grep '^21:' "$log" | head -1)"; return 1
  fi
  total="$(grep -o 'total_sum=[0-9]*' "$crlog" 2>/dev/null | head -1 | cut -d= -f2)"; kms="$(grep -o 'kernel_ms=[0-9.]*' "$crlog" 2>/dev/null | head -1 | cut -d= -f2)"
  stride_act="$(grep -o 'stride=[0-9]*' "$crlog" 2>/dev/null | tail -1 | cut -d= -f2)"; match=0; grep -q '\[gpu-run-correctness\] MATCH' "$crlog" 2>/dev/null && match=1
  local src; src="$(grep -o 'src=[^ ]*' "$crlog" 2>/dev/null | head -1 | cut -d= -f2)"
  record_row "$cell" dispatch "$CU_BIN" "$(basename "${src:-?}")" "$emb" "${stride_act:-?}" "${total:-?}" "$match" "${kms:-?}" "$clk" "$tmp" "$start"
  if [[ "$match" -eq 1 && "${total:-}" == "$ORACLE" ]]; then pass "oracle_match[$cell]"; else fail "oracle_match[$cell]" "total_sum='${total:-<none>}' match=$match"; return 1; fi
  grep '^21:' "$log" | grep -q ' ok$' && pass "dispatcher_row_ok[$cell]" || { fail "dispatcher_row_ok[$cell]" "N=21 row not ok"; return 1; }
  [[ "${stride_act:-}" == "$((32*emb))" ]] && pass "stride_as_intended[$cell]=$stride_act" || { fail "stride_as_intended[$cell]" "C ran stride ${stride_act:-?}, expected $((32*emb))"; return 1; }
  [[ "${src:-}" == *"$esuf" ]] && pass "input_order_as_intended[$cell] (...$esuf)" || { fail "input_order_as_intended[$cell]" "C input '${src:-?}' does not end with $esuf"; return 1; }
  grep -q '\[crunner-config\]' "$LOGDIR/dispatch_after_${cell}.log" 2>/dev/null && pass "crunner_config_logged[$cell]" || info "crunner_config_logged[$cell]" "no [crunner-config] line in dispatch.log"
  if [[ "${QUIET_EXPECTED:-0}" == "1" ]]; then
    # r3 contract: without -d the console shows only "GPU mode selected", the
    # header, and table rows. Any [permute-done]/[filter-]/[crunner-] line is a leak.
    if grep -qE '^\[(permute-done|filter-|crunner-|debug-mode)' "$log"; then fail "console_quiet_without_d[$cell]" "leaked: $(grep -E '^\[(permute-done|filter-|crunner-|debug-mode)' "$log" | head -1 | cut -c1-100)"; return 1; else pass "console_quiet_without_d[$cell] (sched stage ran, nothing leaked)"; fi
  fi
  KMS[$cell]="$kms"; info "cell[$cell]" "kernel_ms=$kms"; return 0
}

# ---- D0: direct, GPU otherwise idle ----
run_direct D0 "$CU_BIN" "$IN_PROD" "$MB_PROD" || exit 1
echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"
# ---- D1: direct, with a bare CUDA context held by another process ----
banner "starting $HOLDER_BIN (context only) for D1"
"./$HOLDER_BIN" 0 900 > "$LOGDIR/holder_D1.log" 2>&1 &
HPID=$!; sleep 3; cat "$LOGDIR/holder_D1.log"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null | tee "$LOGDIR/apps_before_D1.txt"
run_direct D1 "$CU_BIN" "$IN_PROD" "$MB_PROD" || { kill "$HPID" 2>/dev/null; exit 1; }
kill "$HPID" 2>/dev/null; wait "$HPID" 2>/dev/null || true
echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"
# ---- D2: direct, with a context + a large touched allocation held ----
banner "starting $HOLDER_BIN (context + ${HOLDER_MB} MB) for D2"
"./$HOLDER_BIN" "$HOLDER_MB" 900 > "$LOGDIR/holder_D2.log" 2>&1 &
HPID=$!; sleep 5; cat "$LOGDIR/holder_D2.log"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null | tee "$LOGDIR/apps_before_D2.txt"
run_direct D2 "$CU_BIN" "$IN_PROD" "$MB_PROD" || { kill "$HPID" 2>/dev/null; exit 1; }
kill "$HPID" 2>/dev/null; wait "$HPID" 2>/dev/null || true
echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"
# (D0 is the direct reference for this revision)
# G runs WITHOUT -d and with the cached sched file removed first, so the
# 4th pipeline stage actually executes and the console contract of bare -g
# ("table rows only") is exercised, not just assumed.
SCHED_CACHE="${IN_RAW}.sched394f.bin"
if [[ -f "$SCHED_CACHE" ]]; then info "G_prep" "removing cached $SCHED_CACHE so the sched stage runs during G (rebuilt in ~30 s)"; rm -f "$SCHED_CACHE"; fi
# 395b: "-g 21 21" now applies the A10G_FINAL defaults (argc<=4). This IS the
# production path restricted to N=21 -- no positional overrides at all.
QUIET_EXPECTED=1 run_dispatch G "-g $NQ $NQ" "$MB_PROD" ".sched394f.bin" || exit 1
[[ -f "$SCHED_CACHE" ]] && pass "G_rebuilt_sched_cache" || fail "G_rebuilt_sched_cache" "the sched stage did not produce $SCHED_CACHE"
echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"
run_dispatch A "-g $NQ $NQ 32 484 0 0 7 37 -d" 484 ".sched394f.bin" || exit 1

# ---------------------------------------------------------------------
# 4b. Optional: the true bare -g (N=5..27). N=22 builds its sched
#     permutation on first use (28.7M records, minutes). ~40 min total.
# ---------------------------------------------------------------------
if [[ "$FULL_G" == "1" ]]; then
  banner "FULL_G: bare ./$PY_BIN -g  (N=5..27, production defaults end to end)"
  rm -f "$CRLOG_DIR/crunner_${CU_BIN}_N21.log" "$CRLOG_DIR/crunner_${CU_BIN}_N22.log"
  START="$(date -Is)"
  env -u NQ_MAX_BLOCKS "./$PY_BIN" -g 2>&1 | tee "$LOGDIR/2_FULLG_console.log"
  for nn in 21 22; do
    cl="$CRLOG_DIR/crunner_${CU_BIN}_N${nn}.log"; cp "$cl" "$LOGDIR/2_FULLG_crunner_N${nn}.log" 2>/dev/null || true
    exp="$ORACLE"; [[ "$nn" == "22" ]] && exp="$ORACLE22"
    t="$(grep -o 'total_sum=[0-9]*' "$cl" 2>/dev/null | head -1 | cut -d= -f2)"; k="$(grep -o 'kernel_ms=[0-9.]*' "$cl" 2>/dev/null | head -1 | cut -d= -f2)"
    st="$(grep -o 'stride=[0-9]*' "$cl" 2>/dev/null | tail -1 | cut -d= -f2)"; sr="$(grep -o 'src=[^ ]*' "$cl" 2>/dev/null | head -1 | cut -d= -f2)"
    m=0; grep -q '\[gpu-run-correctness\] MATCH' "$cl" 2>/dev/null && m=1
    record_row "FULLG_N$nn" bare-g "$CU_BIN" "$(basename "${sr:-?}")" "$MB_PROD" "${st:-?}" "${t:-?}" "$m" "${k:-?}" "?" "?" "$START"
    if [[ "$m" -eq 1 && "${t:-}" == "$exp" ]]; then pass "fullg_oracle_match[N=$nn] total_sum=$t kernel_ms=${k:-?}"; else fail "fullg_oracle_match[N=$nn]" "total_sum='${t:-<none>}' match=$m (expected $exp)"; fi
    [[ "${st:-}" == "$((32*MB_PROD))" ]] && pass "fullg_stride[N=$nn]=$st" || fail "fullg_stride[N=$nn]" "got ${st:-?}"
    [[ "${sr:-}" == *".sched394f.bin" ]] && pass "fullg_input_sched[N=$nn]" || fail "fullg_input_sched[N=$nn]" "input '${sr:-?}'"
    [[ "$nn" == "21" && -n "${k:-}" ]] && KMS[FULLG21]="$k"
    [[ "$nn" == "22" && -n "${k:-}" ]] && info "N22_production" "kernel_ms=$k = $(awk -v k="$k" 'BEGIN{s=k/1000; printf "%d:%04.1f", int(s/60), s-60*int(s/60)}')  (389-era 484/raw: ~1,699,210 = 28:19)"
  done
  cp "$CRLOG_DIR/dispatch.log" "$LOGDIR/dispatch_after_FULLG.log" 2>/dev/null || true
fi

# ---------------------------------------------------------------------
# 5. Optional SASS check (NCU=1)
# ---------------------------------------------------------------------
if [[ "$NCU_STAGE" == "1" ]]; then
  banner "ncu SourceCounters on the head slice with a -lineinfo build: SASS count and the 2899xx duplicate"
  "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "${CU_BIN}_lineinfo" "$CU_SRC" 2>&1 | tee "$LOGDIR/08_nvcc_lineinfo.log"
  head -c $((15488*28)) "$IN_RAW" > "/tmp/${REV}_head.bin"
  REP="$LOGDIR/${REV}_head_sourcecounters"; MARKER="$LOGDIR/.owner_marker"; touch "$MARKER"
  sudo "$NCU_BIN" --section SourceCounters --page source -f -o "$REP" "./${CU_BIN}_lineinfo" "$NQ" "/tmp/${REV}_head.bin" "/tmp/${REV}_head_ncu.bin" 2>&1 | tee "$LOGDIR/09_ncu_head.log"
  find . -maxdepth 1 -newer "$MARKER" -user root -print0 2>/dev/null | xargs -0 -r sudo chown "$(id -u):$(id -g)" 2>/dev/null || true; sudo chown -R "$(id -u):$(id -g)" "$LOGDIR" 2>/dev/null || true; rm -f "$MARKER"
  "$NCU_BIN" --import "${REP}.ncu-rep" --page source --csv > "${REP}_source.csv" 2>&1 || true
  "$NCU_BIN" --import "${REP}.ncu-rep" --page source --print-source cuda,sass --resolve-source-file "$(pwd)/$CU_SRC" --csv > "${REP}_cuda_sass.csv" 2>&1 || true
  NINST=$(( $(wc -l < "${REP}_source.csv") - 2 ))
  if (( NINST >= 460 && NINST <= 469 )); then pass "sass_count_dropped ($NINST, 394a saw 472)"; elif (( NINST >= 470 && NINST <= 474 )); then info "sass_count_dropped" "$NINST -- unchanged from 472: the compiler re-created the duplicate; see 395a-r2 (64-bit schedule shift)"; else info "sass_count_dropped" "$NINST (unexpected)"; fi
  python3 395b_sass_summary.py "${REP}_source.csv" 2>/dev/null | sed 's/^/    /' || true
fi

# ---------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------
banner "Evaluation"
absdev() { awk -v a="$1" -v r="$2" 'BEGIN{d=(a-r)/r*100; printf "%.3f",(d<0?-d:d)}'; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%+.3f",(b-a)/a*100}'; }
if [[ -n "${KMS[D0]:-}" ]]; then d="$(absdev "${KMS[D0]}" "$REF_D0_MS")"; awk -v d="$d" 'BEGIN{exit !(d<=1.0)}' && pass "D0_reproduces_395b_I1 (${KMS[D0]}, ${d}%)" || info "D0_reproduces_395b_I1" "${KMS[D0]} is ${d}% off 139,166"; fi
if [[ -n "${KMS[D0]:-}" && -n "${KMS[D1]:-}" ]]; then
  d="$(pct "${KMS[D0]}" "${KMS[D1]}")"; info "H_ctx: D1_vs_D0 (context held by another process)" "${d}%  (pre-registered: <= -3 => the context presence is the cause)"
  if awk -v a="$d" 'BEGIN{exit !(a<=-3)}'; then info "verdict_H_ctx" "CONFIRMED: another resident CUDA context makes the same run ${d}% faster"
  elif awk -v a="$d" 'BEGIN{exit !(a>=-1 && a<=1)}'; then info "verdict_H_ctx" "REFUTED: a bare context does nothing -- the cause is something else the Codon parent does (395c-r2: bisect the dispatcher command line)"
  else info "verdict_H_ctx" "partial (${d}%)"; fi
fi
if [[ -n "${KMS[D1]:-}" && -n "${KMS[D2]:-}" ]]; then d="$(pct "${KMS[D1]}" "${KMS[D2]}")"; info "H_place: D2_vs_D1 (+${HOLDER_MB} MB touched by the holder)" "${d}%  (|delta| >= 1.5 => device-memory placement matters on top of the context)"; fi
if [[ -n "${KMS[D0]:-}" && -n "${KMS[G]:-}" ]]; then d="$(pct "${KMS[D0]}" "${KMS[G]}")"; info "path_effect: G_vs_D0 (dispatcher vs direct, same binary+file+stride)" "${d}%  (395a: -4.0%, 395b: -3.7%)"; fi
for c in D0 D1 D2 G A; do [[ -n "${CLK[$c]:-}" ]] && info "clock[$c]" "${CLK[$c]}"; done
if [[ -n "${CLK[D0]:-}" && -n "${CLK[G]:-}" ]]; then
  c0=$(echo "${CLK[D0]}" | grep -o 'sm_mean=[0-9]*' | cut -d= -f2); cg=$(echo "${CLK[G]}" | grep -o 'sm_mean=[0-9]*' | cut -d= -f2)
  [[ -n "$c0" && -n "$cg" ]] && info "H_clock: mean SM clock D0 vs G" "$c0 vs $cg MHz ($(pct "$c0" "$cg")%)  (>= 2% apart => a clock effect, not a memory/context effect)"
fi
if [[ -n "${KMS[G]:-}" ]]; then d="$(absdev "${KMS[G]}" "$REF_G_MS")"; awk -v d="$d" 'BEGIN{exit !(d<=1.5)}' && pass "revert_restores_395a_G (${KMS[G]}, ${d}%)" || fail "revert_restores_395a_G" "${KMS[G]} is ${d}% off 133,577"; fi
# (395c: G is expected to differ from the direct run by ~-4% -- that IS the effect under study, so no equality gate here)
if [[ -n "${KMS[A]:-}" ]]; then d="$(absdev "${KMS[A]}" "$REF_A_MS")"; awk -v d="$d" 'BEGIN{exit !(d<=1.5)}' && pass "explicit_484_matches_395a_A (${KMS[A]}, ${d}%)" || info "explicit_484_matches_395a_A" "${KMS[A]} is ${d}% off 191,781"; fi
[[ -n "${KMS[G]:-}" ]] && info "production_now" "production config (800 x sched, via -g 21 21) N=21: kernel_ms=${KMS[G]} = $(awk -v k="${KMS[G]}" 'BEGIN{s=k/1000; printf "%d:%04.1f", int(s/60), s-60*int(s/60)}')  ($(pct 201237 "${KMS[G]}")% vs the 389 anchor 201,237; target 2:02.52 = 122,520)"

{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi -q -d CLOCK 2>&1; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
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
