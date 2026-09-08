#!/usr/bin/env bash
# 395b_validate.sh
#
# rev395b -- register-resident top of the explicit stack (kernel), plus an
#            investigation of an unexplained 4.2% seen in 395a.
#
# 395a r2 RESULT: K1 (395a binary, direct, 394f_input_L3_sched.bin, 800) =
# 139,179 / 139,158 vs K0 (394g binary) 139,437 / 139,443: -0.2%. SASS still
# 472; the predicated cur_depth<8 pair is gone but nvcc still rematerialises
# the extraction (2 x SHF.R.U64). Safe, neutral, kept.
#
# THE 4.2% PUZZLE: G (395a binary via the dispatcher, on the production
# sched file constellations_...maxd14only_363.bin.sched394f.bin, 800) =
# 133,580 / 133,574 -- 4.2% faster than K1, same binary, same stride, and
# an input that should be byte-identical to 394f_input_L3_sched.bin (same
# stable sort of the same raw file). A (484) shows the same: 191,780 vs
# 394f's L3@484 197,792 (-3.0%). Either the two files differ, or the
# invocation path matters. Stage I settles it without theorising:
#   I0  cmp the two sched files                            (1 s)
#   I1  395a binary, DIRECT, production sched file, 800    (2.3 min)
#   I2  395a binary, DIRECT, 394f_input_L3_sched.bin, 800  (= 395a K1 repeat)
#   file differs  <=> cmp says differ and I1 ~= 133.6
#   path matters  <=> cmp says identical and I1 ~= 139.2
#
# THE KERNEL CHANGE (395b, three sites in process_one_task): the top frame
# of the explicit stack lives in registers; push spills the old top
# (STL.128) and installs the new frame; pop reads top from registers and
# refills it with an LDL.128 that is not consumed until the NEXT pop/push
# (a whole node expansion later). 394a: the pop's LDL consumer carried
# 100.0% of long_scoreboard = 6.8% of samples with 2 instructions of slack.
# This targets latency on the critical path, not instruction count -- the
# lesson of 395a is that in this stall-bound regime (IPC 0.28) removing
# ALU work moves samples rather than time.
#   K0  395a binary, direct, production sched, 800   (= I1, reused)
#   K1  395b binary, direct, production sched, 800   => K1 vs K0 = kernel change alone
#   G   ./395bPy -g 21 21   (quiet, sched cache removed first; from 395b the
#       A10G_FINAL defaults apply for argc<=4, so this is the production config)
#   A   ./395bPy -g 21 21 32 484 0 0 7 37 -d
#
# PRE-REGISTERED (395b_README_append.md)
#   K1 vs K0: -2% .. -5%   (falsified if K1 >= K0 - 0.5%)
#   CPU-harness per-record equality 395a vs 395b on 2,048 real records (gate)
#   SASS (NCU=1): long_scoreboard share falls from ~6.8% to <= 2%
#
# USAGE
#   STATIC_ONLY=1 bash 395b_validate.sh
#                 bash 395b_validate.sh              # I0 I1 I2 K1 G A (~15 min)
#   NCU=1         bash 395b_validate.sh
#   FULL_G=1      bash 395b_validate.sh

set -u

REV="395b"
PY_SRC="${PY_SRC:-395bPy_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-395bPy_kernel_maxd14_final}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
FILTER_SCRIPT="${FILTER_SCRIPT:-363_filter_maxd14_only.py}"
PERMUTE="${PERMUTE:-394f_permute_soa7.py}"
CU_SRC="${CU_SRC:-395b_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-395b_kernel_maxd14}"
PREV_CU="${PREV_CU:-395a_kernel_maxd14.cu}"
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
REF_K0_MS="${REF_K0_MS:-133577}"      # 395a G (production sched, 800): 133,580 / 133,574
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
for f in "$PY_SRC" "$HELPER_SRC" "$FILTER_SCRIPT" "$PERMUTE" "$CU_SRC" "$IN_RAW" 395b_order_diff.py 395b_sass_summary.py; do
  [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"
done
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }

python3 - "$PY_SRC" > "/tmp/${REV}_code_only.py" <<'EOF'
import re,sys
s=open(sys.argv[1],encoding='utf-8').read()
m=re.search(r'"""',s); e=s.index('"""',m.end()); print(s[:m.start()]+s[e+3:])
EOF
CODE="/tmp/${REV}_code_only.py"
grep -qE '^REV_TAG:str="395b"' "$CODE" && pass "source_rev_tag_is_395b" || fail "source_rev_tag_is_395b" "wrong REV_TAG"
grep -qE '^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=800' "$CODE" && pass "source_default_max_blocks_800" || fail "source_default_max_blocks_800" "not 800"
grep -q '      if argc <= 4:' "$CODE" && [[ "$(grep -c '      if argc == 2:' "$CODE")" == "0" ]] && pass "source_defaults_apply_for_argc_le_4 (-g and -g nmin nmax both get the confirmed parameters)" || fail "source_defaults_apply_for_argc_le_4" "the A10G_FINAL defaults gate is not argc<=4"
grep -qE '^CRUNNER_INPUT_ORDER:str="sched"' "$CODE" && pass "source_input_order_sched" || fail "source_input_order_sched" "not sched"
grep -q 'ensure_crunner_input_bin(N,stream_fname,gpu_log_level,CRUNNER_INPUT_ORDER)' "$CODE" && pass "source_mode37_passes_order" || fail "source_mode37_passes_order" "mode 37 does not pass the order"
grep -q 'NQ_MAX_BLOCKS={gpu_max_blocks} {entry_base37.env_prefix}' "$CODE" && pass "source_mode37_stride_coupling" || fail "source_mode37_stride_coupling" "mode 37 does not pass NQ_MAX_BLOCKS"
[[ "$(grep -c 'ensure_crunner_input_bin(N,shaped_fname,gpu_log_level)' "$CODE")" == "1" ]] && pass "source_mode39_unchanged_raw_order" || fail "source_mode39_unchanged_raw_order" "mode 39 call changed"
grep -q '^def _ensure_crunner_filtered_bin(' "$CODE" && pass "source_389_three_stage_factored" || fail "source_389_three_stage_factored" "helper missing"
grep -qE 'if not \(bench_mode==0 .*bench_mode==39\):' "$CODE" && pass "source_bench_mode_39_in_cli_whitelist" || fail "source_bench_mode_39_in_cli_whitelist" "39 missing"
grep -q 'CRunnerEntry(14,"./395b_kernel_maxd14",""' "$CODE" && pass "source_dispatch_table_references_395b_binary" || fail "source_dispatch_table_references_395b_binary" "table wrong"
[[ "$(grep -c 'os.system(f"python3' "$CODE")" == "2" && "$(grep 'os.system(f"python3' "$CODE" | grep -vc '2>&1')" == "0" ]] && pass "source_external_tools_output_redirected (r3: 363 filter + 394f permute)" || fail "source_external_tools_output_redirected" "an external python3 os.system call is not redirected -- bare -g would leak its output"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF

# .cu: the code-region diff vs 395a must be EXACTLY the register-top change (15 removed / 40 added lines)
if [[ -f "$PREV_CU" ]]; then
  awk 'f||/^#include/{f=1;print}' "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  awk 'f||/^#include/{f=1;print}' "$CU_SRC"  > "/tmp/${REV}_cur_code.cu"
  NR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true)
  NA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
  if [[ "$NR_" == "18" && "$NA_" == "46" ]]; then pass "cu_diff_fingerprint_vs_395a (removed=18 added=46: register top + host default 800)"; else fail "cu_diff_fingerprint_vs_395a" "removed=$NR_ added=$NA_, expected 18/46 -- the change is not the reviewed one"; fi
  grep -q '    int MAX_BLOCKS = 800;' "$CU_SRC" && pass "cu_binary_default_max_blocks_800 (policy: every entry point defaults to the confirmed parameters)" || fail "cu_binary_default_max_blocks_800" "the binary default is not 800"
  grep -q 'uint64_t top0 = 0, top1 = 0;' "$CU_SRC" && pass "cu_register_top_present" || fail "cu_register_top_present" "top0/top1 missing"
  [[ "$(grep -c 'top0 = stack\[stack_ptr\];' "$CU_SRC")" == "1" && "$(grep -c 'stack\[stack_ptr\]   = top0;' "$CU_SRC")" == "1" ]] && pass "cu_top_refill_and_spill_sites" || fail "cu_top_refill_and_spill_sites" "refill/spill sites not as reviewed"
  [[ "$(grep -c 'if (save_sp != 0u) {' "$CU_SRC")" == "2" ]] && pass "cu_two_save_sp_guards (pop refill + push spill)" || fail "cu_two_save_sp_guards" "expected exactly 2"
  grep -q 'uint64_t stack\[MAXD14_ANCESTOR \* 2\];' "$CU_SRC" && pass "cu_stack_array_unchanged" || fail "cu_stack_array_unchanged" "the 26-slot array changed"
  grep -q 'const uint32_t nibble_op  = (uint32_t)(schedule64 >> (cur_depth \* 4)) & 15u;' "$CU_SRC" && pass "cu_395a_r2_extraction_kept" || fail "cu_395a_r2_extraction_kept" "395a r2 extraction missing"
else info "cu_diff_fingerprint_vs_395a" "skipped ($PREV_CU absent)"; fi
grep -q 'RISK NOTE' "$CU_SRC" && grep -q 'if (cur_avail != 0u) {' "$CU_SRC" && pass "cu_push_guard_untouched" || fail "cu_push_guard_untouched" "the 358/359 push guard form changed"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; nvidia-smi -q -d CLOCK 2>&1
  sha256sum "$PY_SRC" "$HELPER_SRC" "$FILTER_SCRIPT" "$PERMUTE" "$CU_SRC" "$IN_RAW" 2>&1; [[ -f "$PREV_CU" ]] && sha256sum "$PREV_CU"; } > "$LOGDIR/00_env_pre.txt" 2>&1

# ---------------------------------------------------------------------
# 2. CPU-harness equivalence gate (no GPU): 395a vs 395b on real records
# ---------------------------------------------------------------------
banner "CPU-harness equivalence: process_one_task() 395a vs 395b on the first $CPU_CHECK_RECORDS real records"
if [[ -f "$PREV_CU" ]] && command -v "$GCC" >/dev/null 2>&1; then
  "$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_prev" "$PREV_CU" -lm 2>"$LOGDIR/03_gcc_prev.log" && "$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"$LOGDIR/03_gcc_cur.log"
  if [[ -x "/tmp/${REV}_cpu_prev" && -x "/tmp/${REV}_cpu_cur" ]]; then
    head -c $((CPU_CHECK_RECORDS*28)) "$IN_RAW" > "/tmp/${REV}_cpu_in.bin"
    export OMP_NUM_THREADS="$(nproc)"
    "/tmp/${REV}_cpu_prev" "$NQ" "/tmp/${REV}_cpu_in.bin" "/tmp/${REV}_cpu_prev.bin" 2>&1 | tail -1 | tee "$LOGDIR/04_cpu_prev.log"
    "/tmp/${REV}_cpu_cur"  "$NQ" "/tmp/${REV}_cpu_in.bin" "/tmp/${REV}_cpu_cur.bin"  2>&1 | tail -1 | tee "$LOGDIR/04_cpu_cur.log"
    if cmp -s "/tmp/${REV}_cpu_prev.bin" "/tmp/${REV}_cpu_cur.bin"; then pass "cpu_harness_per_record_results_identical ($CPU_CHECK_RECORDS records)"; else fail "cpu_harness_per_record_results_identical" "395a and 395b differ on CPU -- the kernel change is NOT semantics-preserving; stopping before the GPU"; exit 1; fi
  else fail "cpu_harness_build" "see $LOGDIR/03_gcc_*.log"; exit 1; fi
else info "cpu_harness_equivalence" "skipped ($PREV_CU or gcc absent)"; fi

# ---------------------------------------------------------------------
# 3. Builds
# ---------------------------------------------------------------------
banner "Building $CU_SRC, $PREV_CU (control) and $PY_SRC"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -o "$CU_BIN" "$CU_SRC" 2>&1 | tee "$LOGDIR/05_nvcc_build.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build_succeeded[$CU_BIN]" || { fail "nvcc_build_succeeded" "no binary"; exit 1; }
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
declare -A KMS
record_row() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$TSV"; }
run_direct() {   # cell bin input mb
  local cell="$1"; local bin="$2"; local src="$3"; local mb="$4"
  banner "cell $cell: direct ./$bin on $(basename "$src" | cut -c1-60) NQ_MAX_BLOCKS=$mb"
  local clk tmp start log total kms stride_act match
  clk="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  tmp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  start="$(date -Is)"; log="$LOGDIR/1_${cell}.log"
  NQ_MAX_BLOCKS="$mb" "./$bin" "$NQ" "$src" "/tmp/${REV}_${cell}.bin" "$ORACLE" 2>&1 | tee "$log"
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
  # shellcheck disable=SC2086
  env -u NQ_MAX_BLOCKS "./$PY_BIN" $args 2>&1 | tee "$log"
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

# ---- Stage I: the 4.2% puzzle ----
banner "I0: cmp $IN_L3 vs $IN_PROD"
if [[ -f "$IN_L3" && -f "$IN_PROD" ]]; then
  if cmp -s "$IN_L3" "$IN_PROD"; then info "I0_sched_files" "IDENTICAL -- the 4.2% is NOT the file; it is the invocation path"
  else FIRST=$(cmp "$IN_L3" "$IN_PROD" 2>/dev/null | head -1); info "I0_sched_files" "DIFFER ($FIRST) -- the two 'sched' orders are not the same; see I1/I2"
       python3 395b_order_diff.py "$IN_L3" "$IN_PROD" 2>/dev/null | tee "$LOGDIR/I0_order_diff.txt" || true
  fi
else info "I0_sched_files" "one of the files is missing ($IN_L3 / $IN_PROD)"; fi
[[ -x "$PREV_BIN" ]] && { run_direct I1 "$PREV_BIN" "$IN_PROD" "$MB_PROD" || exit 1; echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; }
[[ -x "$PREV_BIN" ]] && { run_direct I2 "$PREV_BIN" "$IN_L3" "$MB_PROD" || exit 1; echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; }
[[ -n "${KMS[I1]:-}" ]] && KMS[K0]="${KMS[I1]}"
# ---- Stage K: the kernel change alone ----
run_direct K1 "$CU_BIN" "$IN_PROD" "$MB_PROD" || exit 1
echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"
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
if [[ -n "${KMS[I1]:-}" && -n "${KMS[I2]:-}" ]]; then
  d="$(pct "${KMS[I2]}" "${KMS[I1]}")"; info "I1_vs_I2 (prod sched vs 394f L3, same binary, direct)" "${d}%  (395a saw the prod file 4.2% faster via the dispatcher)"
  if awk -v a="$d" 'BEGIN{exit !(a<=-3)}'; then info "verdict_4pct" "the FILE: the production sched order is faster than 394f_input_L3_sched.bin even when run directly"
  elif awk -v a="$d" 'BEGIN{exit !(a>=-1 && a<=1)}'; then info "verdict_4pct" "the PATH: same file speed when run directly -> the dispatcher invocation is ~4% faster; investigate the Codon parent (CUDA context?)"
  else info "verdict_4pct" "partial (${d}%)"; fi
fi
if [[ -n "${KMS[K0]:-}" ]]; then d="$(absdev "${KMS[K0]}" "$REF_K0_MS")"; awk -v d="$d" 'BEGIN{exit !(d<=1.5)}' && pass "K0_reproduces_395a_G (${KMS[K0]}, ${d}%)" || info "K0_reproduces_395a_G" "${KMS[K0]} is ${d}% off 133,577"; fi
if [[ -n "${KMS[K0]:-}" && -n "${KMS[K1]:-}" ]]; then
  d="$(pct "${KMS[K0]}" "${KMS[K1]}")"; info "prereg_K1_vs_K0 (kernel change alone)" "${d}%  (395b pre-registered -2 .. -5; falsified if >= -0.5)"
  awk -v a="$d" 'BEGIN{exit !(a<=-0.5)}' && pass "kernel_change_is_a_measured_gain (${d}%)" || info "kernel_change_is_a_measured_gain" "${d}% -- not a gain at the 0.5% threshold; oracle still holds, so the change is SAFE but not effective"
fi
if [[ -n "${KMS[K1]:-}" && -n "${KMS[G]:-}" ]]; then d="$(absdev "${KMS[G]}" "${KMS[K1]}")"; awk -v d="$d" 'BEGIN{exit !(d<=1.0)}' && pass "bare_g_matches_direct_K1 (${KMS[G]} vs ${KMS[K1]}, ${d}%)" || fail "bare_g_matches_direct_K1" "${d}% apart -- the dispatcher path is not running the same config as the direct run"; fi
if [[ -n "${KMS[A]:-}" ]]; then d="$(absdev "${KMS[A]}" "$REF_A_MS")"; awk -v d="$d" 'BEGIN{exit !(d<=1.5)}' && pass "explicit_484_matches_395a_A (${KMS[A]}, ${d}%)" || info "explicit_484_matches_395a_A" "${KMS[A]} is ${d}% off 191,781"; fi
[[ -n "${KMS[G]:-}" ]] && info "production_now" "production config (800 x sched) N=21: kernel_ms=${KMS[G]} = $(awk -v k="${KMS[G]}" 'BEGIN{s=k/1000; printf "%d:%04.1f", int(s/60), s-60*int(s/60)}')  ($(pct 201237 "${KMS[G]}")% vs the 389 anchor 201,237; target 2:02.52 = 122,520)"

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
