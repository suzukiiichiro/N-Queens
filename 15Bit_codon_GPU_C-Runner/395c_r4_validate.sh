#!/usr/bin/env bash
# 395c_r4_validate.sh
#
# rev395c-r4 -- is the gain COUNT or AMOUNT, and does the extra context have
#               to live in another process?
#
# WHERE WE ARE (395c / r2 / r3; every cell oracle-MATCH, every cell 1710 MHz,
# in-session repeat noise 0.012%, anchors reproduce across sessions to 0.007%)
#
#   total contexts (incl. ours)   foreign MiB   kernel_ms      vs A0
#     1                                0        139,185.0        --
#     2  (= production)               250       133,561.1     -4.041%
#     3  (2 idle holders)             500       133,171.9     -4.320%   <- record
#     2  (1 holder + 512 MB)          762       133,854.0     -3.830%
#     2  (1 holder + 2048 MB)        2298       147,634.8     +6.071%
#   our OWN 255 MiB pad, 1 ctx                  147,807.0     +6.195%
#   our OWN 2303 MiB pad, 1 ctx                 142,339.5     +2.266%
#   production -g 21 21                         133,585.2     -4.023%
#
#   r3 killed the two-factor model. Foreign memory and our own memory have
#   OPPOSITE signs, and neither is monotonic: S255 (free_mb 22,017) and D1
#   (free_mb 22,018) differ by 10.7%, so free memory is not the variable.
#   r3 also confirmed by direct sampling that the Codon dispatcher holds
#   exactly 250 MiB throughout G (66/66 samples) -- a bare context, nothing
#   recoverable there.
#
# THE TWO QUESTIONS THIS REVISION ANSWERS
#   (1) X2 (2 holders, 500 MiB, 3 contexts) beat B512 (1 holder, 762 MiB,
#       2 contexts). Was that the COUNT or the AMOUNT? Y510 holds the amount
#       fixed at exactly X2's 510 MiB in a SINGLE holder, so free_mb matches
#       X2 to within a MiB and only the context count differs.
#   (2) Must the extra context be in another PROCESS? NQ_EXTRA_CTX makes our
#       own process create idle contexts via cuCtxCreate and pop them, so the
#       runtime keeps using the primary context. If that reproduces the gain,
#       production needs no helper process at all.
#
# CELLS  (order is not a variable -- r2 settled that)
#   X2b    2 holders x 0 MB          replicate the record before claiming it
#   Y510   1 holder  x 255 MB        same 510 MiB as X2b, ONE context
#   M1     NQ_EXTRA_CTX=1, no holder our own second context
#   X4     4 holders x 0 MB          extend the count ladder (1020 MiB)
#   B1024  1 holder  x 1024 MB       bracket the cliff (762 ok, 2298 bad)
#   X6     6 holders x 0 MB          count vs the memory penalty (1530 MiB)
#
# PRE-REGISTERED (395c_r4_README_append.md; written before execution)
#   Q1  X2b within +-0.1% of 133,172. The record replicates, or it was not a
#       record. (Repeat noise is 0.012%, so 0.1% is eight times the floor.)
#   Q2 (stated)  Y510 >= X2b + 0.25%  -> the axis is the CONTEXT COUNT.
#       Reasoning: on an amount-only curve, 250 MiB gives 133,561 and 762 MiB
#       gives 133,854, so 510 MiB should interpolate to about 133,70x -- well
#       above X2b's 133,172.
#       (alternative)  |Y510 - X2b| <= 0.1% -> it is the AMOUNT, there is an
#       optimum near 500 MiB, and X4/X6 below are pointless.
#   Q3 (stated)  M1 within +-0.5% of 139,180 -> a context in our own process
#       does nothing; the extra context must be foreign. Precedent: our own
#       memory pad behaved with the opposite sign to a foreign one.
#       (alternative, and the valuable one)  M1 <= 133,600 -> the effect is
#       about contexts on the DEVICE, not about processes. Production then
#       gets it with one env var and no helper process, and NQ_EXTRA_CTX
#       becomes a sweepable production knob.
#   Q4  X4 <= X2b and X6 <= X4, each step smaller than the last.
#       Falsified if X4 > X2b: the memory penalty already dominates at
#       1020 MiB and the count axis is capped at 3 contexts.
#   Q5  B1024 lands between 134,000 and 140,000 -> the +10.5% seen at
#       2298 MiB has not begun at 1279 MiB. If B1024 >= 145,000 the cliff
#       is below 1279 MiB and production must stay under it deliberately.
#
#   NOT A GOAL: no production change is made here. If Q2 says count and Q3
#   says foreign, r5 wires a holder into the dispatcher and measures -g.
#
# USAGE
#   STATIC_ONLY=1 bash 395c_r4_validate.sh
#                 bash 395c_r4_validate.sh     # CPU equiv -> 6 cells (~19 min)

set -u

REV="395c_r4"
CU_SRC="${CU_SRC:-395c_r4_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-395c_r4_kernel_maxd14}"
PREV_CU="${PREV_CU:-395c_r3_kernel_maxd14.cu}"
HOLDER_SRC="${HOLDER_SRC:-395c_ctx_holder.cu}"
HOLDER_BIN="${HOLDER_BIN:-395c_ctx_holder}"
HOLDER_SECS="${HOLDER_SECS:-1200}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
ORACLE="${ORACLE:-314666222712}"
IN_RAW="${IN_RAW:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
IN_PROD="${IN_PROD:-${IN_RAW}.sched394f.bin}"
EXPECTED_RECORDS="${EXPECTED_RECORDS:-2025282}"
MB_PROD="${MB_PROD:-800}"
HOLD_Y="${HOLD_Y:-255}"      # 255 + the 255 MiB context = the 510 MiB X2 occupies
HOLD_B="${HOLD_B:-1024}"
# references from 395c / r2 / r3
REF_X2_MS="${REF_X2_MS:-133172}"
REF_D1_MS="${REF_D1_MS:-133561}"
REF_A0_MS="${REF_A0_MS:-139180}"
FREE_X2="${FREE_X2:-21762}"
KERNEL_SHA_395C="${KERNEL_SHA_395C:-ebd7f523deb591347eab1a352a9555c2e4a6c0b4a456d80517322c933a86e68a}"
CPU_CHECK_RECORDS="${CPU_CHECK_RECORDS:-2048}"
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
for f in "$CU_SRC" "$HOLDER_SRC" "$IN_RAW"; do
  [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"
done
if [[ -f "$IN_PROD" ]]; then
  sz="$(stat -c%s "$IN_PROD")"
  [[ "$sz" -eq $((EXPECTED_RECORDS*28)) ]] && pass "sched_input_present_and_sized[$((sz/28)) records]" \
    || fail "sched_input_present_and_sized" "$IN_PROD is $sz bytes, expected $((EXPECTED_RECORDS*28))"
else fail "sched_input_present_and_sized" "$IN_PROD missing -- generate it BEFORE the run"; fi
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }

extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA_395C" ]] && pass "cu_kernel_region_sha_unchanged_since_395a_r2 (${KB:0:16}...)" \
  || fail "cu_kernel_region_sha_unchanged_since_395a_r2" "got ${KB:0:16}... -- r4 is a host-side-only change"
if [[ -f "$PREV_CU" ]]; then
  code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  NR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true)
  NA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
  [[ "$NR_" == "0" && "$NA_" == "61" ]] && pass "cu_diff_fingerprint_vs_r3 (removed=0 added=61: the extra-context block only)" \
    || fail "cu_diff_fingerprint_vs_r3" "removed=$NR_ added=$NA_, expected 0/61"
else info "cu_diff_fingerprint_vs_r3" "skipped ($PREV_CU absent)"; fi

# --- the new treatment must be inert unless asked for, and must not disturb
#     the primary context the runtime API uses ---
grep -q 'CU_CHECK(cuCtxPopCurrent(&popped));' "$CU_SRC" && pass "extra_ctx_is_popped (every runtime call still uses the primary context)" \
  || fail "extra_ctx_is_popped" "the created context is left current -- every cudaMalloc below would run on it"
grep -q 'for (long i = 0; i < want; i++) {' "$CU_SRC" && pass "extra_ctx_creation_loop_present" \
  || fail "extra_ctx_creation_loop_present" "the want-bounded creation loop is missing"
grep -q 'CUDA_VERSION >= 13000' "$CU_SRC" && grep -q 'cuCtxCreate(&c, NULL, 0, cu_dev)' "$CU_SRC" && pass "cuCtxCreate_v4_signature_handled (CUDA 13 takes 4 args)" \
  || fail "cuCtxCreate_v4_signature_handled" "no CUDA 13 branch -- nvcc 13 maps cuCtxCreate to _v4 and the 3-arg form will not compile"
grep -q 'if (want > 0) {' "$CU_SRC" && pass "extra_ctx_inert_when_unset_statically" || fail "extra_ctx_inert_when_unset_statically" "no if (want > 0) guard"
grep -q 'cuCtxDestroy(extra_ctx\[_i\])' "$CU_SRC" && pass "extra_ctx_destroyed" || fail "extra_ctx_destroyed" "no cuCtxDestroy"
L_CTX=$(grep -n 'CU_CHECK(cuCtxCreate' "$CU_SRC" | head -1 | cut -d: -f1)
L_PAD=$(grep -n 'cudaMalloc(&d_pad' "$CU_SRC" | head -1 | cut -d: -f1)
L_LD=$(grep -n 'cudaMalloc(&d_ld' "$CU_SRC" | head -1 | cut -d: -f1)
L_H2D=$(grep -n 'cudaEventRecord(ev_h2d_start)' "$CU_SRC" | head -1 | cut -d: -f1)
if [[ -n "$L_CTX" && -n "$L_PAD" && -n "$L_LD" && -n "$L_H2D" ]] && (( L_CTX < L_PAD && L_PAD < L_LD && L_LD < L_H2D )); then
  pass "extra_ctx@$L_CTX < pad@$L_PAD < working_allocs@$L_LD < first_timed_event@$L_H2D"
else fail "treatment_ordering" "ctx@${L_CTX:-?} pad@${L_PAD:-?} ld@${L_LD:-?} h2d@${L_H2D:-?}"; fi
# --- invariants carried forward ---
[[ "$(grep -c 'CUDA_CHECK(cudaMalloc' "$CU_SRC")" == "10" ]] && pass "alloc_count_is_9_working_plus_1_pad" \
  || fail "alloc_count_is_9_working_plus_1_pad" "$(grep -c 'CUDA_CHECK(cudaMalloc' "$CU_SRC") calls, expected 10"
grep -A1 'if (pad_mb > 0) {' "$CU_SRC" | grep -q 'cudaMalloc(&d_pad' && pass "r3_pad_still_guarded" || fail "r3_pad_still_guarded" "the pad guard changed"
grep -q '\[gpu-mem-base\] ld=0x%llx' "$CU_SRC" && pass "r2_placement_probe_retained" || fail "r2_placement_probe_retained" "the [gpu-mem-base] probe was dropped"
grep -q '    int MAX_BLOCKS = 800;' "$CU_SRC" && pass "cu_binary_default_max_blocks_800" || fail "cu_binary_default_max_blocks_800" "not 800"
[[ "$(grep -c 'uint64_t top0' "$CU_SRC")" == "0" ]] && pass "cu_395b_register_top_absent" || fail "cu_395b_register_top_absent" "395b's top0/top1 is back"
grep -q 'RISK NOTE' "$CU_SRC" && grep -q 'if (cur_avail != 0u) {' "$CU_SRC" && pass "cu_push_guard_untouched" || fail "cu_push_guard_untouched" "the 358/359 push guard form changed"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; nvidia-smi -q -d CLOCK 2>&1
  sha256sum "$CU_SRC" "$HOLDER_SRC" "$IN_RAW" "$IN_PROD" 2>&1; [[ -f "$PREV_CU" ]] && sha256sum "$PREV_CU"; } > "$LOGDIR/00_env_pre.txt" 2>&1

# ---------------------------------------------------------------------
# 2. CPU-harness equivalence: r3 vs r4 must be byte-identical
# ---------------------------------------------------------------------
banner "CPU-harness equivalence: r3 vs r4 on the first $CPU_CHECK_RECORDS real records"
if [[ -f "$PREV_CU" ]] && command -v "$GCC" >/dev/null 2>&1; then
  "$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_prev" "$PREV_CU" -lm 2>"$LOGDIR/03_gcc_prev.log" \
    && "$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"$LOGDIR/03_gcc_cur.log"
  if [[ -x "/tmp/${REV}_cpu_prev" && -x "/tmp/${REV}_cpu_cur" ]]; then
    head -c $((CPU_CHECK_RECORDS*28)) "$IN_RAW" > "/tmp/${REV}_cpu_in.bin"
    export OMP_NUM_THREADS="$(nproc)"
    "/tmp/${REV}_cpu_prev" "$NQ" "/tmp/${REV}_cpu_in.bin" "/tmp/${REV}_cpu_prev.bin" 2>&1 | tail -1 | tee "$LOGDIR/04_cpu_prev.log"
    "/tmp/${REV}_cpu_cur"  "$NQ" "/tmp/${REV}_cpu_in.bin" "/tmp/${REV}_cpu_cur.bin"  2>&1 | tail -1 | tee "$LOGDIR/04_cpu_cur.log"
    if cmp -s "/tmp/${REV}_cpu_prev.bin" "/tmp/${REV}_cpu_cur.bin"; then pass "cpu_harness_per_record_results_identical ($CPU_CHECK_RECORDS records)"
    else fail "cpu_harness_per_record_results_identical" "r3 and r4 differ on CPU -- stopping before the GPU"; exit 1; fi
  else fail "cpu_harness_build" "see $LOGDIR/03_gcc_*.log"; exit 1; fi
else info "cpu_harness_equivalence" "skipped ($PREV_CU or gcc absent)"; fi

# ---------------------------------------------------------------------
# 3. Builds  (-lcuda: the driver API is needed for cuCtxCreate)
# ---------------------------------------------------------------------
banner "Building $CU_SRC (with -lcuda) and $HOLDER_SRC"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -o "$CU_BIN" "$CU_SRC" -lcuda 2>&1 | tee "$LOGDIR/05_nvcc_build.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build_succeeded[$CU_BIN]" || { fail "nvcc_build_succeeded" "no binary -- if the link failed on -lcuda, add -L/usr/lib/x86_64-linux-gnu"; exit 1; }
"$NVCC" -O2 -arch="$ARCH" -o "$HOLDER_BIN" "$HOLDER_SRC" 2>&1 | tee "$LOGDIR/05a_nvcc_holder.log"
[[ -x "$HOLDER_BIN" ]] && pass "ctx_holder_build_succeeded" || { fail "ctx_holder_build_succeeded" "no $HOLDER_BIN"; exit 1; }
head -c $((15488*28)) "$IN_RAW" > "/tmp/${REV}_probe.bin"
env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX "./$CU_BIN" "$NQ" "/tmp/${REV}_probe.bin" "/tmp/${REV}_probe_out.bin" > "$LOGDIR/05b_default_config_probe.log" 2>&1 || true
grep -q 'MAX_BLOCKS=800 stride=25600' "$LOGDIR/05b_default_config_probe.log" && pass "binary_default_config_is_800" || fail "binary_default_config_is_800" "see 05b log"
grep -q '\[gpu-ctx\] extra_ctx=0' "$LOGDIR/05b_default_config_probe.log" && pass "extra_ctx_inert_when_unset_at_runtime" || fail "extra_ctx_inert_when_unset_at_runtime" "$(grep -o '\[gpu-ctx\].*' "$LOGDIR/05b_default_config_probe.log" | head -1)"
grep -q '\[gpu-pad\] pad_mb=0' "$LOGDIR/05b_default_config_probe.log" && pass "pad_inert_when_unset" || fail "pad_inert_when_unset" "see 05b log"
# a real cuCtxCreate must actually work before we spend 19 minutes on it
env -u NQ_MAX_BLOCKS NQ_EXTRA_CTX=1 "./$CU_BIN" "$NQ" "/tmp/${REV}_probe.bin" "/tmp/${REV}_probe_out2.bin" > "$LOGDIR/05c_extra_ctx_probe.log" 2>&1 || true
grep -q '\[gpu-ctx\] extra_ctx=1' "$LOGDIR/05c_extra_ctx_probe.log" && grep -q 'gpu-run-done' "$LOGDIR/05c_extra_ctx_probe.log" \
  && pass "extra_ctx_probe_runs (NQ_EXTRA_CTX=1 creates a context and the run still completes)" \
  || { fail "extra_ctx_probe_runs" "see $LOGDIR/05c_extra_ctx_probe.log"; exit 1; }

# ---------------------------------------------------------------------
# 4. Runs
# ---------------------------------------------------------------------
printf 'cell\tholders\textra_ctx\ttotal_ctx\tkernel_ms\tfree_mb\td_ld\ttotal_sum\tmatch\tstride_actual\tsm_clock_mhz\ttemp_c\tstart\n' > "$TSV"
declare -A KMS FREEMB LD CLK
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
record_row() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$TSV"; }

declare -a HPIDS=()
start_holders() {  # cell count mb
  local cell="$1" n="$2" mb="$3" i
  HPIDS=()
  banner "starting $n holder(s) x ${mb} MB for $cell"
  for (( i=1; i<=n; i++ )); do
    "./$HOLDER_BIN" "$mb" "$HOLDER_SECS" > "$LOGDIR/holder_${cell}_$i.log" 2>&1 &
    HPIDS+=("$!"); sleep 4
    grep -q '\[ctx-holder\] context up' "$LOGDIR/holder_${cell}_$i.log" || { fail "holder_up[$cell#$i]" "no live context"; return 1; }
  done
  pass "holders_up[$cell] ($n x ${mb} MB)"
}
stop_holders() { local p; for p in "${HPIDS[@]:-}"; do [[ -n "$p" ]] && { kill "$p" 2>/dev/null; wait "$p" 2>/dev/null || true; }; done; HPIDS=(); sleep 3; }
gpu_occupancy_gate() {  # cell expected_foreign_count
  local cell="$1" want="$2" n
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null > "$LOGDIR/apps_before_${cell}.txt"
  n="$(grep -c '^[0-9]' "$LOGDIR/apps_before_${cell}.txt" || true)"; cat "$LOGDIR/apps_before_${cell}.txt"
  [[ "$n" == "$want" ]] && pass "gpu_occupancy_as_intended[$cell] ($n foreign)" \
    || { fail "gpu_occupancy_as_intended[$cell]" "$n foreign process(es), expected $want"; return 1; }
}
run_direct() {   # cell holders_desc extra_ctx total_ctx
  local cell="$1" hdesc="$2" xctx="$3" tctx="$4"
  banner "cell $cell: holders=$hdesc  NQ_EXTRA_CTX=$xctx  (total contexts on device: $tctx)"
  local clk tmp start log total kms stride_act match ldv freemb xseen
  clk="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  tmp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  start="$(date -Is)"; log="$LOGDIR/1_${cell}.log"
  start_samplers "$cell"
  if [[ "$xctx" == "0" ]]; then
    env -u NQ_EXTRA_CTX -u NQ_PAD_MB NQ_MAX_BLOCKS="$MB_PROD" "./$CU_BIN" "$NQ" "$IN_PROD" "/tmp/${REV}_${cell}.bin" "$ORACLE" 2>&1 | tee "$log"
  else
    env -u NQ_PAD_MB NQ_EXTRA_CTX="$xctx" NQ_MAX_BLOCKS="$MB_PROD" "./$CU_BIN" "$NQ" "$IN_PROD" "/tmp/${REV}_${cell}.bin" "$ORACLE" 2>&1 | tee "$log"
  fi
  stop_samplers "$cell"
  total="$(grep -o 'total_sum=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  kms="$(grep -o 'kernel_ms=[0-9.]*' "$log" | head -1 | cut -d= -f2)"
  stride_act="$(grep -o 'stride=[0-9]*' "$log" | tail -1 | cut -d= -f2)"
  ldv="$(grep -o 'ld=0x[0-9a-f]*' "$log" | head -1 | cut -d= -f2)"
  freemb="$(grep -o 'free_mb=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  xseen="$(grep -o 'extra_ctx=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  match=0; grep -q '\[gpu-run-correctness\] MATCH' "$log" && match=1
  record_row "$cell" "$hdesc" "${xseen:-?}" "$tctx" "${kms:-?}" "${freemb:-?}" "${ldv:-?}" "${total:-?}" "$match" "${stride_act:-?}" "$clk" "$tmp" "$start"
  if [[ "$match" -eq 1 && "${total:-}" == "$ORACLE" ]]; then pass "oracle_match[$cell]"; else fail "oracle_match[$cell]" "total_sum='${total:-<none>}' match=$match"; return 1; fi
  [[ "${stride_act:-}" == "$((32*MB_PROD))" ]] && pass "stride_as_intended[$cell]" || { fail "stride_as_intended[$cell]" "got ${stride_act:-?}"; return 1; }
  [[ "${xseen:-}" == "$xctx" ]] && pass "extra_ctx_as_intended[$cell] (extra_ctx=$xseen)" || { fail "extra_ctx_as_intended[$cell]" "binary reported extra_ctx=${xseen:-?}, asked for $xctx"; return 1; }
  KMS[$cell]="$kms"; FREEMB[$cell]="${freemb:-?}"; LD[$cell]="${ldv:-?}"
  info "cell[$cell]" "kernel_ms=$kms  free_mb=${freemb:-?}"
  return 0
}
cool() { echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; }

banner "6 cells: X2b Y510 M1 X4 B1024 X6  (~19 min)"

# ---- X2b: replicate the record ----
start_holders X2b 2 0 || exit 1
gpu_occupancy_gate X2b 2 || { stop_holders; exit 1; }
run_direct X2b "2x0MB" 0 3 || { stop_holders; exit 1; }
stop_holders; cool
# ---- Y510: the SAME 510 MiB as X2b, in ONE holder. Count vs amount. ----
start_holders Y510 1 "$HOLD_Y" || exit 1
gpu_occupancy_gate Y510 1 || { stop_holders; exit 1; }
run_direct Y510 "1x${HOLD_Y}MB" 0 2 || { stop_holders; exit 1; }
stop_holders; cool
# ---- M1: our own second context, no helper process ----
gpu_occupancy_gate M1 0 || exit 1
run_direct M1 "none" 1 2 || exit 1
cool
# ---- X4: extend the count ladder ----
start_holders X4 4 0 || exit 1
gpu_occupancy_gate X4 4 || { stop_holders; exit 1; }
run_direct X4 "4x0MB" 0 5 || { stop_holders; exit 1; }
stop_holders; cool
# ---- B1024: bracket the cliff ----
start_holders B1024 1 "$HOLD_B" || exit 1
gpu_occupancy_gate B1024 1 || { stop_holders; exit 1; }
run_direct B1024 "1x${HOLD_B}MB" 0 2 || { stop_holders; exit 1; }
stop_holders; cool
# ---- X6: count against the memory penalty ----
start_holders X6 6 0 || exit 1
gpu_occupancy_gate X6 6 || { stop_holders; exit 1; }
run_direct X6 "6x0MB" 0 7 || { stop_holders; exit 1; }
stop_holders

# ---------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------
banner "Evaluation"
absdev() { awk -v a="$1" -v r="$2" 'BEGIN{d=(a-r)/r*100; printf "%.3f",(d<0?-d:d)}'; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%+.3f",(b-a)/a*100}'; }
near() { awk -v d="$1" -v t="$2" 'BEGIN{exit !(d<=t)}'; }

banner "Q1 -- does the record replicate?"
if [[ -n "${KMS[X2b]:-}" ]]; then
  d="$(absdev "${KMS[X2b]}" "$REF_X2_MS")"
  near "$d" 0.1 && pass "Q1_record_replicates (${KMS[X2b]}, ${d}% from 133,172)" \
    || fail "Q1_record_replicates" "${KMS[X2b]} is ${d}% off 133,172 -- the 3-context number was not stable; everything below is read with that in mind"
  [[ -n "${FREEMB[X2b]:-}" ]] && info "free_mb[X2b]" "${FREEMB[X2b]} (r3 saw $FREE_X2)"
fi

banner "Q2 -- COUNT or AMOUNT? (Y510 holds the amount fixed at X2b's)"
if [[ -n "${KMS[Y510]:-}" && -n "${KMS[X2b]:-}" ]]; then
  p="$(pct "${KMS[X2b]}" "${KMS[Y510]}")"; d="$(absdev "${KMS[Y510]}" "${KMS[X2b]}")"
  info "Y510_vs_X2b" "${p}%   free_mb ${FREEMB[Y510]:-?} vs ${FREEMB[X2b]:-?}"
  if awk -v x="$p" 'BEGIN{exit !(x>=0.25)}'; then
    pass "Q2_CONFIRMED_the_axis_is_CONTEXT_COUNT (same 510 MiB, one fewer context, ${p}% slower)"
    info "next" "r5: wire an idle holder into the dispatcher and measure -g 21 21 end to end"
  elif near "$d" 0.1; then
    pass "Q2_ALTERNATIVE_the_axis_is_the_AMOUNT (context count is irrelevant at fixed MiB)"
    info "next" "sweep foreign MiB finely around 500; X4/X6 below carry no information"
  else info "Q2_partial" "${p}% -- between the two pre-registered outcomes"; fi
fi

banner "Q3 -- must the extra context be in another PROCESS?"
if [[ -n "${KMS[M1]:-}" ]]; then
  dA="$(absdev "${KMS[M1]}" "$REF_A0_MS")"; pD="$(pct "$REF_D1_MS" "${KMS[M1]}")"
  info "M1" "kernel_ms=${KMS[M1]}  free_mb=${FREEMB[M1]:-?}  (${dA}% from the 1-context 139,180; ${pD}% vs the 2-context 133,561)"
  if awk -v x="${KMS[M1]}" 'BEGIN{exit !(x<=133600)}'; then
    pass "Q3_ALTERNATIVE_A_CONTEXT_IN_OUR_OWN_PROCESS_WORKS"
    info "next" "no helper process is needed: NQ_EXTRA_CTX becomes a production knob, and sweeping it is the cheapest remaining axis"
  elif near "$dA" 0.5; then
    pass "Q3_CONFIRMED_the_extra_context_must_be_FOREIGN (our own buys nothing)"
    info "next" "the lever is a helper process; r5 wires one into the dispatcher"
  else info "Q3_partial" "M1 sits between the references"; fi
fi

banner "Q4 -- how far does the count ladder go?"
for c in X2b X4 X6; do [[ -n "${KMS[$c]:-}" ]] && info "ladder[$c]" "kernel_ms=${KMS[$c]}  free_mb=${FREEMB[$c]:-?}"; done
if [[ -n "${KMS[X4]:-}" && -n "${KMS[X2b]:-}" ]]; then
  p="$(pct "${KMS[X2b]}" "${KMS[X4]}")"; info "X4_vs_X2b" "${p}%"
  awk -v x="$p" 'BEGIN{exit !(x<=0)}' && pass "Q4_ladder_still_descending_at_5_contexts" \
    || info "Q4_ladder_capped" "X4 is ${p}% SLOWER than X2b -- the memory penalty already dominates at ~1020 MiB; 3 contexts is the practical optimum"
fi
if [[ -n "${KMS[X6]:-}" && -n "${KMS[X4]:-}" ]]; then info "X6_vs_X4" "$(pct "${KMS[X4]}" "${KMS[X6]}")%"; fi
best=""; bestv=""
for c in X2b Y510 M1 X4 B1024 X6; do
  [[ -z "${KMS[$c]:-}" ]] && continue
  if [[ -z "$bestv" ]] || awk -v a="${KMS[$c]}" -v b="$bestv" 'BEGIN{exit !(a<b)}'; then best="$c"; bestv="${KMS[$c]}"; fi
done
[[ -n "$best" ]] && info "best_cell_this_revision" "$best = $bestv  ($(pct 133585 "$bestv")% vs production 133,585; $(pct 133172 "$bestv")% vs the r3 record 133,172)"

banner "Q5 -- where is the memory cliff?"
if [[ -n "${KMS[B1024]:-}" ]]; then
  info "B1024" "kernel_ms=${KMS[B1024]}  free_mb=${FREEMB[B1024]:-?}  (762 MiB gave 133,854; 2298 MiB gave 147,635)"
  if awk -v x="${KMS[B1024]}" 'BEGIN{exit !(x>=134000 && x<=140000)}'; then pass "Q5_cliff_is_above_1279_MiB"
  elif awk -v x="${KMS[B1024]}" 'BEGIN{exit !(x>=145000)}'; then fail "Q5_cliff_is_BELOW_1279_MiB" "${KMS[B1024]} -- production must stay under this deliberately; document it as a hard constraint"
  else info "Q5_other" "${KMS[B1024]} -- outside both pre-registered bands"; fi
fi

for c in X2b Y510 M1 X4 B1024 X6; do [[ -n "${CLK[$c]:-}" ]] && info "clock[$c]" "${CLK[$c]}"; done

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
