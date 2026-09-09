#!/usr/bin/env bash
# 395c_r2_validate.sh
#
# rev395c-r2 -- kill the ONE confound left in 395c: run order.
#
# 395c RESULT (2026-09-08/09, all oracle-MATCH, all cells 1710 MHz):
#   D0  direct, GPU idle                 139,188.266
#   D1  direct, foreign ctx (0 MB)       133,578.562   -4.030% vs D0   <- H_ctx
#   D2  direct, foreign ctx + 2048 MB    147,621.453  +10.513% vs D1   <- H_place
#   G   dispatcher -g 21 21              133,576.281   -0.002% vs D1   <- the 4% is fully explained
#   A   dispatcher, 484                  191,765.484
#   H_clock REJECTED: sm_mean = sm_min = 1710 in every cell.
#
# THE PROBLEM WITH 395c
#   The cells ran in the fixed order D0 -> D1 -> D2 -> G -> A, and D0 was the
#   only cold-start cell (30 C at t=0). Clock is excluded, but "the first run
#   of a session is slow" is NOT excluded by 395c's design. If that is what
#   happened, H_ctx and H_place are both artifacts.
#
# THIS REVISION: the same three direct cells, ORDER REVERSED, plus repeats.
#   D2r   ctx + 2048 MB   <- now the COLD first cell (was 3rd and warm in 395c)
#   D1r   ctx only
#   D0r   GPU idle        <- now the 3rd, warm cell (was 1st and cold in 395c)
#   D0r2  GPU idle        (immediate repeat: within-session drift of the slow state)
#   D1r2  ctx only        (late repeat: is the fast state still reachable when hot?)
#
#   The binary is 395c plus ONE host-side diagnostic block that logs where the
#   device buffers landed ([gpu-mem] / [gpu-mem-base]). It runs after every
#   cudaMalloc and before ev_h2d_start, and allocates nothing, so it changes
#   neither placement nor any timed region -- gated statically below. This
#   turns "D1 is faster" into a question that can be answered by ADDRESS:
#   if d_ld is the same in D0r and D1r, placement cannot be the mechanism.
#
#   No dispatcher cell: 395c reproduced G to 0.0005% and D1 to 0.002%. G is
#   not a variable here, and dropping it also drops the Codon build.
#
# PRE-REGISTERED (395c_r2_README_append.md; written before execution)
#   P1  D0r  within +-0.5% of 139,188  AND  D0r2 within +-0.5% of 139,188
#       -> the slow state is a property of the CONDITION, not of being first.
#   P2  D2r  within +-1.0% of 147,621, even though it is now the cold cell
#       -> the slowest cell is slow when run first, too. This is the single
#          sharpest test in this revision: order and treatment are now
#          maximally anti-correlated.
#   P3  D1r <= D0r - 3%  AND  D1r2 <= D0r2 - 3%
#       -> H_ctx replicates twice under reversed order.
#   FALSIFIED if D1r is within +-1% of D0r, or if D2r lands near 139,000
#       instead of 147,600 -> order/thermal dominates, 395c's H_ctx and
#       H_place are void, and the 4% goes back to being unexplained.
#   SECONDARY (no threshold, evidence only): [gpu-mem-base] ld= per cell.
#       same address in D0r and D1r  -> placement is NOT the mechanism of the
#         D0/D1 gap; "a foreign context exists" is doing the work by itself.
#       different address             -> consistent with placement; feeds the
#         self-padding sweep (block 4).
#   BY-PRODUCT: |D0r2-D0r| and |D1r2-D1r| are this session's true repeat
#       noise floor for a 2:13 run. 394c's +-1.6% between sessions is the
#       number this has to be compared against.
#
# USAGE
#   STATIC_ONLY=1 bash 395c_r2_validate.sh
#                 bash 395c_r2_validate.sh     # CPU equiv -> D2r D1r D0r D0r2 D1r2 (~16 min)

set -u

REV="395c_r2"
CU_SRC="${CU_SRC:-395c_r2_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-395c_r2_kernel_maxd14}"
PREV_CU="${PREV_CU:-395c_kernel_maxd14.cu}"
HOLDER_SRC="${HOLDER_SRC:-395c_ctx_holder.cu}"
HOLDER_BIN="${HOLDER_BIN:-395c_ctx_holder}"
HOLDER_MB="${HOLDER_MB:-2048}"
HOLDER_SECS="${HOLDER_SECS:-900}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
ORACLE="${ORACLE:-314666222712}"
IN_RAW="${IN_RAW:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
IN_PROD="${IN_PROD:-${IN_RAW}.sched394f.bin}"
EXPECTED_RECORDS="${EXPECTED_RECORDS:-2025282}"
MB_PROD="${MB_PROD:-800}"
# 395c measured values -- the references this revision must reproduce
REF_D0_MS="${REF_D0_MS:-139188}"
REF_D1_MS="${REF_D1_MS:-133579}"
REF_D2_MS="${REF_D2_MS:-147621}"
TOL_D0="${TOL_D0:-0.5}"
TOL_D1="${TOL_D1:-0.5}"
TOL_D2="${TOL_D2:-1.0}"
# 395c kernel-region sha256 (== 395a r2). Hard-coded so the gate holds even
# if 395c_kernel_maxd14.cu is not on disk.
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
else
  fail "sched_input_present_and_sized" "$IN_PROD missing -- 395c's G rebuilt it; regenerate with 394f_permute_soa7.py sched before running r2 (this revision must NOT rebuild it mid-session)"
fi
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }

# --- the kernel must be untouched: sha256 of the kernel region ---
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA_395C" ]] && pass "cu_kernel_region_sha_matches_395c_and_395a_r2 (${KB:0:16}...)" \
  || fail "cu_kernel_region_sha_matches_395c_and_395a_r2" "got ${KB:0:16}..., expected ${KERNEL_SHA_395C:0:16}... -- the kernel was touched; r2 must be a host-side-only change"
if [[ -f "$PREV_CU" ]]; then
  code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  NR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true)
  NA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
  [[ "$NR_" == "0" && "$NA_" == "20" ]] && pass "cu_diff_fingerprint_vs_395c (removed=0 added=20: the placement probe only)" \
    || fail "cu_diff_fingerprint_vs_395c" "removed=$NR_ added=$NA_, expected 0/20 -- more than the probe changed"
else info "cu_diff_fingerprint_vs_395c" "skipped ($PREV_CU absent)"; fi

# --- the probe must be non-perturbing: position and contents ---
L_ALLOC=$(grep -n 'cudaMalloc(&d_results' "$CU_SRC" | head -1 | cut -d: -f1)
L_PROBE=$(grep -n '\[gpu-mem\] ld=%p' "$CU_SRC" | head -1 | cut -d: -f1)
L_H2D=$(grep -n 'cudaEventRecord(ev_h2d_start)' "$CU_SRC" | head -1 | cut -d: -f1)
if [[ -n "$L_ALLOC" && -n "$L_PROBE" && -n "$L_H2D" ]] && (( L_ALLOC < L_PROBE && L_PROBE < L_H2D )); then
  pass "probe_is_after_all_allocs_and_before_first_timed_event (malloc@$L_ALLOC < probe@$L_PROBE < h2d@$L_H2D)"
else
  fail "probe_is_after_all_allocs_and_before_first_timed_event" "malloc@${L_ALLOC:-?} probe@${L_PROBE:-?} h2d@${L_H2D:-?}"
fi
[[ "$(grep -c 'CUDA_CHECK(cudaMalloc' "$CU_SRC")" == "9" ]] && pass "probe_allocates_nothing (still exactly 9 CUDA_CHECK(cudaMalloc) calls, as in 395c)" \
  || fail "probe_allocates_nothing" "$(grep -c 'CUDA_CHECK(cudaMalloc' "$CU_SRC") allocation calls, expected 9 -- the probe added an allocation and is no longer non-perturbing"
grep -q '\[gpu-mem-base\] ld=0x%llx' "$CU_SRC" && pass "probe_logs_base_address" || fail "probe_logs_base_address" "no [gpu-mem-base] line"
grep -q 'cudaMemGetInfo' "$CU_SRC" && pass "probe_logs_free_memory" || fail "probe_logs_free_memory" "no cudaMemGetInfo"
# --- unchanged invariants carried from 395c ---
grep -q '    int MAX_BLOCKS = 800;' "$CU_SRC" && pass "cu_binary_default_max_blocks_800" || fail "cu_binary_default_max_blocks_800" "not 800"
[[ "$(grep -c 'uint64_t top0' "$CU_SRC")" == "0" ]] && pass "cu_395b_register_top_absent" || fail "cu_395b_register_top_absent" "395b's top0/top1 is back"
grep -q 'RISK NOTE' "$CU_SRC" && grep -q 'if (cur_avail != 0u) {' "$CU_SRC" && pass "cu_push_guard_untouched" || fail "cu_push_guard_untouched" "the 358/359 push guard form changed"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; nvidia-smi -q -d CLOCK 2>&1
  sha256sum "$CU_SRC" "$HOLDER_SRC" "$IN_RAW" "$IN_PROD" 2>&1; [[ -f "$PREV_CU" ]] && sha256sum "$PREV_CU"; } > "$LOGDIR/00_env_pre.txt" 2>&1

# ---------------------------------------------------------------------
# 2. CPU-harness equivalence gate: 395c vs r2 must be byte-identical
#    (the kernel is untouched, so any difference means the probe leaked)
# ---------------------------------------------------------------------
banner "CPU-harness equivalence: 395c vs 395c-r2 on the first $CPU_CHECK_RECORDS real records"
if [[ -f "$PREV_CU" ]] && command -v "$GCC" >/dev/null 2>&1; then
  "$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_prev" "$PREV_CU" -lm 2>"$LOGDIR/03_gcc_prev.log" \
    && "$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"$LOGDIR/03_gcc_cur.log"
  if [[ -x "/tmp/${REV}_cpu_prev" && -x "/tmp/${REV}_cpu_cur" ]]; then
    head -c $((CPU_CHECK_RECORDS*28)) "$IN_RAW" > "/tmp/${REV}_cpu_in.bin"
    export OMP_NUM_THREADS="$(nproc)"
    "/tmp/${REV}_cpu_prev" "$NQ" "/tmp/${REV}_cpu_in.bin" "/tmp/${REV}_cpu_prev.bin" 2>&1 | tail -1 | tee "$LOGDIR/04_cpu_prev.log"
    "/tmp/${REV}_cpu_cur"  "$NQ" "/tmp/${REV}_cpu_in.bin" "/tmp/${REV}_cpu_cur.bin"  2>&1 | tail -1 | tee "$LOGDIR/04_cpu_cur.log"
    if cmp -s "/tmp/${REV}_cpu_prev.bin" "/tmp/${REV}_cpu_cur.bin"; then pass "cpu_harness_per_record_results_identical ($CPU_CHECK_RECORDS records)"
    else fail "cpu_harness_per_record_results_identical" "395c and r2 differ on CPU -- stopping before the GPU"; exit 1; fi
  else fail "cpu_harness_build" "see $LOGDIR/03_gcc_*.log"; exit 1; fi
else info "cpu_harness_equivalence" "skipped ($PREV_CU or gcc absent)"; fi

# ---------------------------------------------------------------------
# 3. Builds
# ---------------------------------------------------------------------
banner "Building $CU_SRC and $HOLDER_SRC"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -o "$CU_BIN" "$CU_SRC" 2>&1 | tee "$LOGDIR/05_nvcc_build.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build_succeeded[$CU_BIN]" || { fail "nvcc_build_succeeded" "no binary"; exit 1; }
"$NVCC" -O2 -arch="$ARCH" -o "$HOLDER_BIN" "$HOLDER_SRC" 2>&1 | tee "$LOGDIR/05a_nvcc_holder.log"
[[ -x "$HOLDER_BIN" ]] && pass "ctx_holder_build_succeeded" || { fail "ctx_holder_build_succeeded" "no $HOLDER_BIN"; exit 1; }
head -c $((15488*28)) "$IN_RAW" > "/tmp/${REV}_probe.bin"
env -u NQ_MAX_BLOCKS "./$CU_BIN" "$NQ" "/tmp/${REV}_probe.bin" "/tmp/${REV}_probe_out.bin" > "$LOGDIR/05b_default_config_probe.log" 2>&1 || true
grep -q 'MAX_BLOCKS=800 stride=25600' "$LOGDIR/05b_default_config_probe.log" && pass "binary_default_config_is_800 (NQ_MAX_BLOCKS unset)" \
  || fail "binary_default_config_is_800" "see $LOGDIR/05b_default_config_probe.log"
grep -q '\[gpu-mem\]' "$LOGDIR/05b_default_config_probe.log" && pass "probe_emits_at_runtime" || fail "probe_emits_at_runtime" "no [gpu-mem] line in the probe run"

# ---------------------------------------------------------------------
# 4. Runs -- REVERSED ORDER
# ---------------------------------------------------------------------
printf 'cell\tholder_mb\tbinary\tinput\tmax_blocks\tstride_actual\ttotal_sum\tmatch\tkernel_ms\td_ld\tfree_mb\tsm_clock_mhz\ttemp_c\tstart\n' > "$TSV"
declare -A KMS CLK LD FREEMB
SAMPLER_PID=""
start_sampler() {
  local out="$LOGDIR/clk_${1}.tsv"; : > "$out"
  ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits >> "$out" 2>/dev/null; sleep 5; done ) &
  SAMPLER_PID=$!
}
stop_sampler() {
  [[ -n "$SAMPLER_PID" ]] && { kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; }; SAMPLER_PID=""
  CLK[$1]="$(awk -F', *' 'NF>=6 && $2+0>0 && $6+0>50 {n++; s+=$2; if(min==""||$2<min)min=$2; p+=$4} END{if(n) printf "sm_mean=%.0f sm_min=%.0f power_mean=%.1fW n=%d", s/n, min, p/n, n; else print "no-samples"}' "$LOGDIR/clk_${1}.tsv")"
  info "in_run_clock[$1]" "${CLK[$1]}"
}
record_row() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$TSV"; }

# 395c lesson (D2): a leftover process on the GPU moves the result by up to 6%.
# Occupancy is now a gated pre-condition of every cell, not an observation.
HPID=""
start_holder() {  # $1 = cell, $2 = MB (empty/0 => context only)
  local cell="$1" mb="${2:-0}"
  banner "starting $HOLDER_BIN (alloc_mb=$mb) for $cell"
  "./$HOLDER_BIN" "$mb" "$HOLDER_SECS" > "$LOGDIR/holder_${cell}.log" 2>&1 &
  HPID=$!; sleep 5; cat "$LOGDIR/holder_${cell}.log"
  grep -q '\[ctx-holder\] context up' "$LOGDIR/holder_${cell}.log" && pass "holder_up[$cell] (alloc_mb=$mb)" \
    || { fail "holder_up[$cell]" "holder did not report a live context"; return 1; }
}
stop_holder() { [[ -n "$HPID" ]] && { kill "$HPID" 2>/dev/null; wait "$HPID" 2>/dev/null || true; }; HPID=""; sleep 2; }
gpu_occupancy_gate() {  # $1 = cell, $2 = expected number of foreign compute processes
  local cell="$1" want="$2" n
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null > "$LOGDIR/apps_before_${cell}.txt"
  n="$(grep -c '^[0-9]' "$LOGDIR/apps_before_${cell}.txt" || true)"
  cat "$LOGDIR/apps_before_${cell}.txt"
  [[ "$n" == "$want" ]] && pass "gpu_occupancy_as_intended[$cell] ($n foreign process(es))" \
    || { fail "gpu_occupancy_as_intended[$cell]" "$n foreign process(es) on the GPU, expected $want -- a stray process shifts the result by up to 6% (395c D2)"; return 1; }
}
run_direct() {   # cell holder_mb
  local cell="$1"; local hmb="$2"
  banner "cell $cell: direct ./$CU_BIN  (holder_mb=$hmb)  NQ_MAX_BLOCKS=$MB_PROD"
  local clk tmp start log total kms stride_act match ldv freemb
  clk="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  tmp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  start="$(date -Is)"; log="$LOGDIR/1_${cell}.log"
  start_sampler "$cell"
  NQ_MAX_BLOCKS="$MB_PROD" "./$CU_BIN" "$NQ" "$IN_PROD" "/tmp/${REV}_${cell}.bin" "$ORACLE" 2>&1 | tee "$log"
  stop_sampler "$cell"
  total="$(grep -o 'total_sum=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  kms="$(grep -o 'kernel_ms=[0-9.]*' "$log" | head -1 | cut -d= -f2)"
  stride_act="$(grep -o 'stride=[0-9]*' "$log" | tail -1 | cut -d= -f2)"
  ldv="$(grep -o 'ld=0x[0-9a-f]*' "$log" | head -1 | cut -d= -f2)"
  freemb="$(grep -o 'free_mb=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  match=0; grep -q '\[gpu-run-correctness\] MATCH' "$log" && match=1
  record_row "$cell" "$hmb" "$CU_BIN" "$(basename "$IN_PROD")" "$MB_PROD" "${stride_act:-?}" "${total:-?}" "$match" "${kms:-?}" "${ldv:-?}" "${freemb:-?}" "$clk" "$tmp" "$start"
  if [[ "$match" -eq 1 && "${total:-}" == "$ORACLE" ]]; then pass "oracle_match[$cell]"; else fail "oracle_match[$cell]" "total_sum='${total:-<none>}' match=$match"; return 1; fi
  [[ "${stride_act:-}" == "$((32*MB_PROD))" ]] && pass "stride_as_intended[$cell]" || { fail "stride_as_intended[$cell]" "got ${stride_act:-?}"; return 1; }
  KMS[$cell]="$kms"; LD[$cell]="${ldv:-?}"; FREEMB[$cell]="${freemb:-?}"
  info "cell[$cell]" "kernel_ms=$kms  d_ld=${ldv:-?}  free_mb=${freemb:-?}"
  return 0
}
cool() { echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; }

banner "5 direct cells in REVERSED order: D2r D1r D0r D0r2 D1r2  (~16 min)"

# ---- D2r: the 395c-slowest condition, now run FIRST and cold ----
start_holder D2r "$HOLDER_MB" || exit 1
gpu_occupancy_gate D2r 1 || { stop_holder; exit 1; }
run_direct D2r "$HOLDER_MB" || { stop_holder; exit 1; }
stop_holder; cool

# ---- D1r: context only ----
start_holder D1r 0 || exit 1
gpu_occupancy_gate D1r 1 || { stop_holder; exit 1; }
run_direct D1r 0 || { stop_holder; exit 1; }
stop_holder; cool

# ---- D0r: GPU idle, now the 3rd and warm cell ----
gpu_occupancy_gate D0r 0 || exit 1
run_direct D0r "-" || exit 1
cool

# ---- D0r2: immediate repeat of the idle condition ----
gpu_occupancy_gate D0r2 0 || exit 1
run_direct D0r2 "-" || exit 1
cool

# ---- D1r2: late repeat of the fast condition, at the session's hottest ----
start_holder D1r2 0 || exit 1
gpu_occupancy_gate D1r2 1 || { stop_holder; exit 1; }
run_direct D1r2 0 || { stop_holder; exit 1; }
stop_holder

# ---------------------------------------------------------------------
# 5. Evaluation against the pre-registered predictions
# ---------------------------------------------------------------------
banner "Evaluation"
absdev() { awk -v a="$1" -v r="$2" 'BEGIN{d=(a-r)/r*100; printf "%.3f",(d<0?-d:d)}'; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%+.3f",(b-a)/a*100}'; }
within() { awk -v d="$1" -v t="$2" 'BEGIN{exit !(d<=t)}'; }

P1=1; P2=1; P3=1
for c in D0r D0r2; do
  if [[ -n "${KMS[$c]:-}" ]]; then
    d="$(absdev "${KMS[$c]}" "$REF_D0_MS")"
    within "$d" "$TOL_D0" && pass "P1[$c]_reproduces_395c_D0 (${KMS[$c]}, ${d}%)" || { fail "P1[$c]_reproduces_395c_D0" "${KMS[$c]} is ${d}% off $REF_D0_MS (tol ${TOL_D0}%)"; P1=0; }
  else P1=0; fi
done
if [[ -n "${KMS[D2r]:-}" ]]; then
  d="$(absdev "${KMS[D2r]}" "$REF_D2_MS")"
  within "$d" "$TOL_D2" && pass "P2[D2r]_slow_even_when_cold_and_first (${KMS[D2r]}, ${d}%)" || { fail "P2[D2r]_slow_even_when_cold_and_first" "${KMS[D2r]} is ${d}% off $REF_D2_MS (tol ${TOL_D2}%)"; P2=0; }
else P2=0; fi
for c in D1r D1r2; do
  if [[ -n "${KMS[$c]:-}" ]]; then
    d="$(absdev "${KMS[$c]}" "$REF_D1_MS")"
    within "$d" "$TOL_D1" && pass "P3[$c]_reproduces_395c_D1 (${KMS[$c]}, ${d}%)" || { fail "P3[$c]_reproduces_395c_D1" "${KMS[$c]} is ${d}% off $REF_D1_MS (tol ${TOL_D1}%)"; P3=0; }
  else P3=0; fi
done
for pair in "D0r:D1r" "D0r2:D1r2"; do
  a="${pair%%:*}"; b="${pair##*:}"
  if [[ -n "${KMS[$a]:-}" && -n "${KMS[$b]:-}" ]]; then
    d="$(pct "${KMS[$a]}" "${KMS[$b]}")"
    info "H_ctx[$b vs $a]" "${d}%  (pre-registered: <= -3%)"
    awk -v x="$d" 'BEGIN{exit !(x<=-3)}' || { fail "H_ctx_replicated[$b vs $a]" "${d}% does not reach -3%"; P3=0; }
    awk -v x="$d" 'BEGIN{exit !(x>=-1 && x<=1)}' && info "verdict" "REFUTED at [$b vs $a]: with the order reversed the foreign context does nothing -- 395c's H_ctx was a first-run artifact"
  fi
done
# repeat noise floor (the by-product that makes every later sweep readable)
[[ -n "${KMS[D0r]:-}" && -n "${KMS[D0r2]:-}" ]] && info "repeat_noise[D0r->D0r2]" "$(pct "${KMS[D0r]}" "${KMS[D0r2]}")%  (394c between-session spread was +-1.6%)"
[[ -n "${KMS[D1r]:-}" && -n "${KMS[D1r2]:-}" ]] && info "repeat_noise[D1r->D1r2]" "$(pct "${KMS[D1r]}" "${KMS[D1r2]}")%"
# placement evidence: same address in D0r and D1r => placement is NOT the mechanism
for c in D2r D1r D0r D0r2 D1r2; do [[ -n "${LD[$c]:-}" ]] && info "placement[$c]" "d_ld=${LD[$c]} free_mb=${FREEMB[$c]:-?}"; done
if [[ -n "${LD[D0r]:-}" && -n "${LD[D1r]:-}" ]]; then
  if [[ "${LD[D0r]}" == "${LD[D1r]}" ]]; then
    info "verdict_placement" "d_ld IDENTICAL in D0r and D1r (${LD[D0r]}) -> device-memory placement of our buffers is NOT the mechanism of the D0/D1 gap; the foreign context is doing the work some other way"
  else
    info "verdict_placement" "d_ld DIFFERS (${LD[D0r]} vs ${LD[D1r]}) -> consistent with placement; the self-padding sweep is the right next move"
  fi
fi
for c in D2r D1r D0r D0r2 D1r2; do [[ -n "${CLK[$c]:-}" ]] && info "clock[$c]" "${CLK[$c]}"; done
if [[ "$P1" == "1" && "$P2" == "1" && "$P3" == "1" ]]; then
  pass "ORDER_CONFOUND_EXCLUDED (all of P1/P2/P3 held with the order reversed; 395c's H_ctx and H_place stand)"
else
  info "verdict_395c" "at least one pre-registered prediction failed (P1=$P1 P2=$P2 P3=$P3) -- do NOT proceed to the padding sweep; 395c-r3 must re-open the order/thermal question"
fi

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
