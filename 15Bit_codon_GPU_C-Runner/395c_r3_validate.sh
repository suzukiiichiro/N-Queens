#!/usr/bin/env bash
# 395c_r3_validate.sh
#
# rev395c-r3 -- separate the two factors 395c and 395c-r2 could not.
#
# WHERE WE ARE (395c 2026-09-08, r2 2026-09-09; every cell oracle-MATCH,
# every cell sm_mean = sm_min = 1710 MHz)
#   foreign occupancy   free_mb   kernel_ms    vs no-occupancy
#     0 MiB             22273     139,177.4        --
#   255 MiB (ctx only)  22018     133,561.1     -4.035%
#  2303 MiB             19970     147,634.8     +6.077%
#   dispatcher -g 21 21           133,576.3     (= the 255 MiB row, to 0.002%)
#
#   r2 reversed the cell order and reproduced all five cells to <=0.015%, at
#   a session temperature 10 C higher, so ORDER and THERMAL are excluded.
#   In-session repeat noise is 0.012% / 0.003%.
#
#   r2's placement probe REFUTED placement: D0r and D0r2 have different
#   relative buffer layouts (ctrl0 at -0x10800000 vs -0x16800000) and agree
#   to 0.012%, while D0r2 and D2r share a layout and differ by 6%. Virtual
#   address does not predict time. Foreign occupancy does, and it does so
#   NON-MONOTONICALLY (0 slow, 255 MiB fast, 2303 MiB slowest) -- which no
#   single factor can produce. Two factors:
#     A  a second CUDA context exists      -4.04%   (binary)
#     B  device memory is occupied        +10.54% per 2 GB  (monotonic)
#
# THE FORK THIS REVISION RESOLVES
#   A foreign holder changes BOTH at once: it creates a context AND occupies
#   ~255 MiB. So "-4.04%" might be either. NQ_PAD_MB makes OUR OWN process
#   occupy the same memory with NO second context, changing only factor B.
#     S255 lands near 139,177  -> occupancy alone does nothing; the second
#          CONTEXT is the cause. The lever is a helper process (which
#          production already has) and an in-process pad sweep is pointless.
#     S255 lands near 133,561  -> occupancy IS the cause. The lever is inside
#          our own process, and a pad sweep for a point better than 133,561
#          becomes the obvious next move.
#   S2303 then asks whether factor B cares WHO holds the memory.
#   X2 asks whether factor A scales with the NUMBER of contexts (free speed
#   if it does: production would just spawn one more idle holder).
#   B512 puts a point in the middle of factor B's gradient.
#   Gp finally looks at what the Codon dispatcher actually does to the GPU.
#
# CELLS (order is not a variable any more -- r2 settled that)
#   S255   direct, NQ_PAD_MB=255,  no foreign process   free_mb must be 22018
#   S2303  direct, NQ_PAD_MB=2303, no foreign process   free_mb must be 19970
#   A0     direct, no pad, no foreign process           free_mb must be 22273
#   X2     direct, TWO holders (0 MB each)
#   B512   direct, ONE holder with 512 MB
#   Gp     ./395cPy_kernel_maxd14_final -g 21 21, with a 2 s compute-apps
#          sampler running throughout (the 395c binary, unchanged)
#
# PRE-REGISTERED (395c_r3_README_append.md; written before execution)
#   P1 (stated)  S255 within +-0.5% of 139,177  -> factor A is the CONTEXT.
#      (alternative, explicitly named) S255 within +-0.5% of 133,561
#                                      -> factor A is OCCUPANCY, self-controllable.
#      Anything else -> neither; r4 re-opens the question.
#   P2  S2303 >= S255 + 8%  -> factor B does not care who owns the memory.
#       S2303 within +-0.5% of S255 -> factor B is foreign-only too.
#   P3 (stated)  X2 within +-0.5% of 133,561 -> factor A saturates at >=1
#       context. (alternative) X2 <= 132,225 (-1% below D1) -> it scales, and
#       a new and nearly free axis opens.
#   P4  B512 within +-1.5% of 137,080 = 133,561 x (1 + 0.1054 x 512/2048)
#       -> factor B is linear in foreign MiB over this range.
#   P5 (stated)  During Gp, nvidia-smi shows TWO concurrent compute processes
#       -- the Codon dispatcher and the CRunner -- and the dispatcher holds
#       ~250 MiB, i.e. a bare context and nothing more. Gp within +-0.5% of
#       133,576.
#       Falsified if only ONE process is ever visible: then nothing holds a
#       second context during G, D1 == G is a coincidence, and the whole
#       two-factor model has to be rebuilt.
#       If the dispatcher holds MUCH more than 250 MiB, that excess is
#       recoverable speed on factor B's gradient -- the one cell in this
#       revision that could pay out today.
#
# USAGE
#   STATIC_ONLY=1 bash 395c_r3_validate.sh
#                 bash 395c_r3_validate.sh     # CPU equiv -> 6 cells (~19 min)

set -u

REV="395c_r3"
CU_SRC="${CU_SRC:-395c_r3_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-395c_r3_kernel_maxd14}"
PREV_CU="${PREV_CU:-395c_r2_kernel_maxd14.cu}"
HOLDER_SRC="${HOLDER_SRC:-395c_ctx_holder.cu}"
HOLDER_BIN="${HOLDER_BIN:-395c_ctx_holder}"
HOLDER_SECS="${HOLDER_SECS:-900}"
# Gp uses the EXISTING 395c dispatcher and the EXISTING 395c binary: this
# revision changes no Python at all, so the dispatcher path stays exactly the
# one that produced G = 133,576.281.
PY_BIN="${PY_BIN:-395cPy_kernel_maxd14_final}"
G_CU_BIN="${G_CU_BIN:-395c_kernel_maxd14}"
CRLOG_DIR="${CRLOG_DIR:-395c_crunner_logs}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
ORACLE="${ORACLE:-314666222712}"
IN_RAW="${IN_RAW:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
IN_PROD="${IN_PROD:-${IN_RAW}.sched394f.bin}"
EXPECTED_RECORDS="${EXPECTED_RECORDS:-2025282}"
MB_PROD="${MB_PROD:-800}"
PAD_A="${PAD_A:-255}"     # matches the 255 MiB a bare foreign context costs
PAD_B="${PAD_B:-2303}"    # matches the 2303 MiB of the D2 holder
HOLD_B512="${HOLD_B512:-512}"
# references, from 395c + r2 (means where two cells exist)
REF_A0_MS="${REF_A0_MS:-139177}"
REF_D1_MS="${REF_D1_MS:-133561}"
REF_D2_MS="${REF_D2_MS:-147635}"
REF_G_MS="${REF_G_MS:-133576}"
REF_B512_MS="${REF_B512_MS:-137080}"
FREE_A0="${FREE_A0:-22273}"
FREE_D1="${FREE_D1:-22018}"
FREE_D2="${FREE_D2:-19970}"
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
else fail "sched_input_present_and_sized" "$IN_PROD missing -- generate it BEFORE the run; r3 must not build it mid-session"; fi
[[ -x "$PY_BIN" ]] && pass "dispatcher_present[$PY_BIN] (unchanged from 395c: this revision touches no Python)" \
  || fail "dispatcher_present" "$PY_BIN missing -- Gp cannot run"
[[ -x "$G_CU_BIN" ]] && pass "395c_binary_present[$G_CU_BIN] (the dispatch table points at it)" \
  || fail "395c_binary_present" "$G_CU_BIN missing -- rebuild it from 395c_kernel_maxd14.cu"
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }

extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA_395C" ]] && pass "cu_kernel_region_sha_unchanged_since_395a_r2 (${KB:0:16}...)" \
  || fail "cu_kernel_region_sha_unchanged_since_395a_r2" "got ${KB:0:16}... -- the kernel was touched; r3 is a host-side-only change"
if [[ -f "$PREV_CU" ]]; then
  code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  NR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true)
  NA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
  [[ "$NR_" == "0" && "$NA_" == "31" ]] && pass "cu_diff_fingerprint_vs_r2 (removed=0 added=31: the self-pad only)" \
    || fail "cu_diff_fingerprint_vs_r2" "removed=$NR_ added=$NA_, expected 0/31"
else info "cu_diff_fingerprint_vs_r2" "skipped ($PREV_CU absent)"; fi

# --- the pad must be inert unless asked for, and must precede the buffers ---
[[ "$(grep -c 'CUDA_CHECK(cudaMalloc' "$CU_SRC")" == "10" ]] && pass "alloc_count_is_9_working_plus_1_pad" \
  || fail "alloc_count_is_9_working_plus_1_pad" "$(grep -c 'CUDA_CHECK(cudaMalloc' "$CU_SRC") allocation calls, expected 10"
grep -A1 'if (pad_mb > 0) {' "$CU_SRC" | grep -q 'cudaMalloc(&d_pad' && pass "pad_alloc_is_guarded_by_pad_mb_gt_0 (NQ_PAD_MB unset => nothing is allocated)" \
  || fail "pad_alloc_is_guarded_by_pad_mb_gt_0" "the pad allocation is not directly inside if (pad_mb > 0)"
L_PAD=$(grep -n 'cudaMalloc(&d_pad' "$CU_SRC" | head -1 | cut -d: -f1)
L_LD=$(grep -n 'cudaMalloc(&d_ld' "$CU_SRC" | head -1 | cut -d: -f1)
L_PROBE=$(grep -n '\[gpu-mem\] ld=%p' "$CU_SRC" | head -1 | cut -d: -f1)
L_H2D=$(grep -n 'cudaEventRecord(ev_h2d_start)' "$CU_SRC" | head -1 | cut -d: -f1)
if [[ -n "$L_PAD" && -n "$L_LD" && -n "$L_PROBE" && -n "$L_H2D" ]] && (( L_PAD < L_LD && L_LD < L_PROBE && L_PROBE < L_H2D )); then
  pass "pad@$L_PAD < working_allocs@$L_LD < probe@$L_PROBE < first_timed_event@$L_H2D"
else fail "pad_ordering" "pad@${L_PAD:-?} ld@${L_LD:-?} probe@${L_PROBE:-?} h2d@${L_H2D:-?}"; fi
grep -q 'if (d_pad) cudaFree(d_pad);' "$CU_SRC" && pass "pad_is_freed" || fail "pad_is_freed" "no cudaFree(d_pad)"
grep -q '\[gpu-pad\] pad_mb=%ld' "$CU_SRC" && pass "pad_is_logged" || fail "pad_is_logged" "no [gpu-pad] line"
grep -q '\[gpu-mem-base\] ld=0x%llx' "$CU_SRC" && pass "r2_placement_probe_retained" || fail "r2_placement_probe_retained" "the [gpu-mem-base] probe was dropped"
grep -q '    int MAX_BLOCKS = 800;' "$CU_SRC" && pass "cu_binary_default_max_blocks_800" || fail "cu_binary_default_max_blocks_800" "not 800"
[[ "$(grep -c 'uint64_t top0' "$CU_SRC")" == "0" ]] && pass "cu_395b_register_top_absent" || fail "cu_395b_register_top_absent" "395b's top0/top1 is back"
grep -q 'RISK NOTE' "$CU_SRC" && grep -q 'if (cur_avail != 0u) {' "$CU_SRC" && pass "cu_push_guard_untouched" || fail "cu_push_guard_untouched" "the 358/359 push guard form changed"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; nvidia-smi -q -d CLOCK 2>&1
  sha256sum "$CU_SRC" "$HOLDER_SRC" "$IN_RAW" "$IN_PROD" "$PY_BIN" 2>&1; [[ -f "$PREV_CU" ]] && sha256sum "$PREV_CU"; } > "$LOGDIR/00_env_pre.txt" 2>&1

# ---------------------------------------------------------------------
# 2. CPU-harness equivalence: r2 vs r3 must be byte-identical
# ---------------------------------------------------------------------
banner "CPU-harness equivalence: r2 vs r3 on the first $CPU_CHECK_RECORDS real records"
if [[ -f "$PREV_CU" ]] && command -v "$GCC" >/dev/null 2>&1; then
  "$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_prev" "$PREV_CU" -lm 2>"$LOGDIR/03_gcc_prev.log" \
    && "$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"$LOGDIR/03_gcc_cur.log"
  if [[ -x "/tmp/${REV}_cpu_prev" && -x "/tmp/${REV}_cpu_cur" ]]; then
    head -c $((CPU_CHECK_RECORDS*28)) "$IN_RAW" > "/tmp/${REV}_cpu_in.bin"
    export OMP_NUM_THREADS="$(nproc)"
    "/tmp/${REV}_cpu_prev" "$NQ" "/tmp/${REV}_cpu_in.bin" "/tmp/${REV}_cpu_prev.bin" 2>&1 | tail -1 | tee "$LOGDIR/04_cpu_prev.log"
    "/tmp/${REV}_cpu_cur"  "$NQ" "/tmp/${REV}_cpu_in.bin" "/tmp/${REV}_cpu_cur.bin"  2>&1 | tail -1 | tee "$LOGDIR/04_cpu_cur.log"
    if cmp -s "/tmp/${REV}_cpu_prev.bin" "/tmp/${REV}_cpu_cur.bin"; then pass "cpu_harness_per_record_results_identical ($CPU_CHECK_RECORDS records)"
    else fail "cpu_harness_per_record_results_identical" "r2 and r3 differ on CPU -- stopping before the GPU"; exit 1; fi
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
env -u NQ_MAX_BLOCKS -u NQ_PAD_MB "./$CU_BIN" "$NQ" "/tmp/${REV}_probe.bin" "/tmp/${REV}_probe_out.bin" > "$LOGDIR/05b_default_config_probe.log" 2>&1 || true
grep -q 'MAX_BLOCKS=800 stride=25600' "$LOGDIR/05b_default_config_probe.log" && pass "binary_default_config_is_800" || fail "binary_default_config_is_800" "see 05b log"
grep -q '\[gpu-pad\] pad_mb=0' "$LOGDIR/05b_default_config_probe.log" && pass "pad_inert_when_unset (NQ_PAD_MB unset => pad_mb=0, nothing allocated)" \
  || fail "pad_inert_when_unset" "$(grep -o '\[gpu-pad\][^\"]*' "$LOGDIR/05b_default_config_probe.log" | head -1)"
grep -q '\[gpu-mem\]' "$LOGDIR/05b_default_config_probe.log" && pass "probe_emits_at_runtime" || fail "probe_emits_at_runtime" "no [gpu-mem] line"

# ---------------------------------------------------------------------
# 4. Runs
# ---------------------------------------------------------------------
printf 'cell\tpath\tpad_mb\tholders\tkernel_ms\tfree_mb\td_ld\ttotal_sum\tmatch\tstride_actual\tsm_clock_mhz\ttemp_c\tstart\n' > "$TSV"
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
  info "concurrent_procs[$cell]" "$(awk -F', *' '{c[$1]++} END{m=0; for(t in c) if(c[t]>m) m=c[t]; print "max="m" concurrent compute process(es) at any sample"}' "$LOGDIR/apps_${cell}.tsv")"
}
record_row() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$TSV"; }

declare -a HPIDS=()
start_holders() {  # $1 = cell, $2 = count, $3 = MB each
  local cell="$1" n="$2" mb="$3" i
  HPIDS=()
  for (( i=1; i<=n; i++ )); do
    "./$HOLDER_BIN" "$mb" "$HOLDER_SECS" > "$LOGDIR/holder_${cell}_$i.log" 2>&1 &
    HPIDS+=("$!"); sleep 4; cat "$LOGDIR/holder_${cell}_$i.log"
    grep -q '\[ctx-holder\] context up' "$LOGDIR/holder_${cell}_$i.log" || { fail "holder_up[$cell#$i]" "no live context"; return 1; }
  done
  pass "holders_up[$cell] ($n x ${mb} MB)"
}
stop_holders() { local p; for p in "${HPIDS[@]:-}"; do [[ -n "$p" ]] && { kill "$p" 2>/dev/null; wait "$p" 2>/dev/null || true; }; done; HPIDS=(); sleep 2; }
gpu_occupancy_gate() {  # $1 = cell, $2 = expected foreign process count
  local cell="$1" want="$2" n
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null > "$LOGDIR/apps_before_${cell}.txt"
  n="$(grep -c '^[0-9]' "$LOGDIR/apps_before_${cell}.txt" || true)"; cat "$LOGDIR/apps_before_${cell}.txt"
  [[ "$n" == "$want" ]] && pass "gpu_occupancy_as_intended[$cell] ($n foreign)" \
    || { fail "gpu_occupancy_as_intended[$cell]" "$n foreign process(es), expected $want"; return 1; }
}
run_direct() {   # cell pad_mb holders_desc
  local cell="$1" pad="$2" hdesc="$3"
  banner "cell $cell: direct ./$CU_BIN  NQ_PAD_MB=$pad  holders=$hdesc"
  local clk tmp start log total kms stride_act match ldv freemb padseen
  clk="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  tmp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  start="$(date -Is)"; log="$LOGDIR/1_${cell}.log"
  start_samplers "$cell"
  if [[ "$pad" == "0" ]]; then
    env -u NQ_PAD_MB NQ_MAX_BLOCKS="$MB_PROD" "./$CU_BIN" "$NQ" "$IN_PROD" "/tmp/${REV}_${cell}.bin" "$ORACLE" 2>&1 | tee "$log"
  else
    NQ_PAD_MB="$pad" NQ_MAX_BLOCKS="$MB_PROD" "./$CU_BIN" "$NQ" "$IN_PROD" "/tmp/${REV}_${cell}.bin" "$ORACLE" 2>&1 | tee "$log"
  fi
  stop_samplers "$cell"
  total="$(grep -o 'total_sum=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  kms="$(grep -o 'kernel_ms=[0-9.]*' "$log" | head -1 | cut -d= -f2)"
  stride_act="$(grep -o 'stride=[0-9]*' "$log" | tail -1 | cut -d= -f2)"
  ldv="$(grep -o 'ld=0x[0-9a-f]*' "$log" | head -1 | cut -d= -f2)"
  freemb="$(grep -o 'free_mb=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  padseen="$(grep -o 'pad_mb=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  match=0; grep -q '\[gpu-run-correctness\] MATCH' "$log" && match=1
  record_row "$cell" direct "${padseen:-?}" "$hdesc" "${kms:-?}" "${freemb:-?}" "${ldv:-?}" "${total:-?}" "$match" "${stride_act:-?}" "$clk" "$tmp" "$start"
  if [[ "$match" -eq 1 && "${total:-}" == "$ORACLE" ]]; then pass "oracle_match[$cell]"; else fail "oracle_match[$cell]" "total_sum='${total:-<none>}' match=$match"; return 1; fi
  [[ "${stride_act:-}" == "$((32*MB_PROD))" ]] && pass "stride_as_intended[$cell]" || { fail "stride_as_intended[$cell]" "got ${stride_act:-?}"; return 1; }
  [[ "${padseen:-}" == "$pad" ]] && pass "pad_applied_as_intended[$cell] (pad_mb=$padseen)" || { fail "pad_applied_as_intended[$cell]" "binary reported pad_mb=${padseen:-?}, asked for $pad"; return 1; }
  KMS[$cell]="$kms"; FREEMB[$cell]="${freemb:-?}"; LD[$cell]="${ldv:-?}"
  info "cell[$cell]" "kernel_ms=$kms  free_mb=${freemb:-?}  d_ld=${ldv:-?}"
  return 0
}
free_gate() {  # cell expected_free_mb  -- proves the treatment reached the device
  local cell="$1" want="$2" got="${FREEMB[$1]:-}"
  [[ -z "$got" || "$got" == "?" ]] && { info "free_mb_as_intended[$cell]" "not captured"; return 0; }
  awk -v a="$got" -v b="$want" 'BEGIN{exit !((a-b<=2)&&(b-a<=2))}' \
    && pass "free_mb_as_intended[$cell] ($got, expected $want)" \
    || fail "free_mb_as_intended[$cell]" "$got MiB free, expected $want -- the treatment did not land as designed"
}
cool() { echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; }

banner "6 cells: S255 S2303 A0 X2 B512 Gp  (~19 min)"

# ---- S255: our own 255 MiB, no second context. THE decisive cell. ----
gpu_occupancy_gate S255 0 || exit 1
run_direct S255 "$PAD_A" "none" || exit 1
free_gate S255 "$FREE_D1"
cool
# ---- S2303: our own 2303 MiB, no second context ----
gpu_occupancy_gate S2303 0 || exit 1
run_direct S2303 "$PAD_B" "none" || exit 1
free_gate S2303 "$FREE_D2"
cool
# ---- A0: in-session anchor ----
gpu_occupancy_gate A0 0 || exit 1
run_direct A0 0 "none" || exit 1
free_gate A0 "$FREE_A0"
cool
# ---- X2: does factor A scale with the NUMBER of contexts? ----
start_holders X2 2 0 || exit 1
gpu_occupancy_gate X2 2 || { stop_holders; exit 1; }
run_direct X2 0 "2x0MB" || { stop_holders; exit 1; }
stop_holders; cool
# ---- B512: a middle point on factor B's gradient ----
start_holders B512 1 "$HOLD_B512" || exit 1
gpu_occupancy_gate B512 1 || { stop_holders; exit 1; }
run_direct B512 0 "1x${HOLD_B512}MB" || { stop_holders; exit 1; }
stop_holders; cool
# ---- Gp: what does the Codon dispatcher actually put on the GPU? ----
gpu_occupancy_gate Gp 0 || exit 1
banner "cell Gp: ./$PY_BIN -g $NQ $NQ  (395c dispatcher + 395c binary, unchanged; compute-apps sampled every 2 s)"
GSTART="$(date -Is)"; GCLK="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1)"
GTMP="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)"
rm -f "$CRLOG_DIR/crunner_${G_CU_BIN}_N${NQ}.log"
start_samplers Gp
env -u NQ_MAX_BLOCKS -u NQ_PAD_MB "./$PY_BIN" -g "$NQ" "$NQ" 2>&1 | tee "$LOGDIR/1_Gp_console.log"
stop_samplers Gp
GCR="$CRLOG_DIR/crunner_${G_CU_BIN}_N${NQ}.log"; cp "$GCR" "$LOGDIR/1_Gp_crunner.log" 2>/dev/null || true
cp "$CRLOG_DIR/dispatch.log" "$LOGDIR/dispatch_after_Gp.log" 2>/dev/null || true
if [[ -f "$GCR" ]]; then
  gt="$(grep -o 'total_sum=[0-9]*' "$GCR" | head -1 | cut -d= -f2)"; gk="$(grep -o 'kernel_ms=[0-9.]*' "$GCR" | head -1 | cut -d= -f2)"
  gs="$(grep -o 'stride=[0-9]*' "$GCR" | tail -1 | cut -d= -f2)"; gm=0; grep -q '\[gpu-run-correctness\] MATCH' "$GCR" && gm=1
  record_row Gp dispatch "-" "codon-parent?" "${gk:-?}" "-" "-" "${gt:-?}" "$gm" "${gs:-?}" "$GCLK" "$GTMP" "$GSTART"
  if [[ "$gm" -eq 1 && "${gt:-}" == "$ORACLE" ]]; then pass "oracle_match[Gp]"; else fail "oracle_match[Gp]" "total_sum='${gt:-<none>}' match=$gm"; fi
  KMS[Gp]="$gk"; info "cell[Gp]" "kernel_ms=${gk:-?}"
else fail "crunner_path_taken[Gp]" "no $GCR"; fi

# ---------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------
banner "Evaluation"
absdev() { awk -v a="$1" -v r="$2" 'BEGIN{d=(a-r)/r*100; printf "%.3f",(d<0?-d:d)}'; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%+.3f",(b-a)/a*100}'; }
near() { awk -v d="$1" -v t="$2" 'BEGIN{exit !(d<=t)}'; }

[[ -n "${KMS[A0]:-}" ]] && { d="$(absdev "${KMS[A0]}" "$REF_A0_MS")"; near "$d" 0.5 && pass "anchor_A0_reproduces (${KMS[A0]}, ${d}%)" || fail "anchor_A0_reproduces" "${KMS[A0]} is ${d}% off $REF_A0_MS -- the session baseline moved; read every other cell with care"; }

banner "P1 -- the fork: is factor A occupancy, or a second context?"
if [[ -n "${KMS[S255]:-}" ]]; then
  dA="$(absdev "${KMS[S255]}" "$REF_A0_MS")"; dD="$(absdev "${KMS[S255]}" "$REF_D1_MS")"
  info "S255" "kernel_ms=${KMS[S255]}  (${dA}% from the no-occupancy 139,177; ${dD}% from the foreign-context 133,561)"
  if near "$dA" 0.5; then
    pass "P1_CONFIRMED_factor_A_is_THE_CONTEXT (our own 255 MiB buys nothing)"
    info "next" "the lever is a helper PROCESS, not an in-process pad; skip the pad sweep, sweep the holder instead (count, and whether an idle kernel/stream changes it)"
  elif near "$dD" 0.5; then
    pass "P1_ALTERNATIVE_factor_A_is_OCCUPANCY (self-controllable)"
    info "next" "sweep NQ_PAD_MB for a point better than 133,561 -- this is now an in-process, production-deployable knob"
  else
    info "P1_NEITHER" "S255 sits between the two references -- factor A is not cleanly either; r4 must re-open it"
  fi
fi

banner "P2 -- does factor B care who owns the memory?"
if [[ -n "${KMS[S2303]:-}" && -n "${KMS[S255]:-}" ]]; then
  d="$(pct "${KMS[S255]}" "${KMS[S2303]}")"
  info "S2303_vs_S255" "${d}%  (>= +8% => factor B is owner-agnostic; ~0% => it needs a foreign owner)"
  info "S2303_vs_D2_reference" "$(pct "$REF_D2_MS" "${KMS[S2303]}")% from 147,635"
fi

banner "P3 -- does factor A scale with the number of contexts?"
if [[ -n "${KMS[X2]:-}" ]]; then
  d="$(absdev "${KMS[X2]}" "$REF_D1_MS")"; p="$(pct "$REF_D1_MS" "${KMS[X2]}")"
  info "X2_vs_one_context" "${KMS[X2]} is ${p}% vs 133,561"
  if near "$d" 0.5; then pass "P3_CONFIRMED_factor_A_saturates_at_one_context"
  elif awk -v x="$p" 'BEGIN{exit !(x<=-1)}'; then pass "P3_ALTERNATIVE_factor_A_SCALES (${p}%) -- a new and nearly free axis: sweep the holder count"
  else info "P3_other" "${p}% -- neither saturation nor the -1% threshold"; fi
fi

banner "P4 -- is factor B linear over 0..2 GB?"
if [[ -n "${KMS[B512]:-}" ]]; then
  d="$(absdev "${KMS[B512]}" "$REF_B512_MS")"
  near "$d" 1.5 && pass "P4_factor_B_is_linear (${KMS[B512]} vs the linear prediction $REF_B512_MS, ${d}%)" \
    || info "P4_factor_B_is_NOT_linear" "${KMS[B512]} vs linear $REF_B512_MS (${d}%) -- the gradient has structure; a finer sweep is worth it"
  info "B512_vs_one_bare_context" "$(pct "$REF_D1_MS" "${KMS[B512]}")%"
fi

banner "P5 -- what does the dispatcher put on the GPU?"
if [[ -n "${KMS[Gp]:-}" ]]; then
  d="$(absdev "${KMS[Gp]}" "$REF_G_MS")"
  near "$d" 0.5 && pass "Gp_reproduces_395c_G (${KMS[Gp]}, ${d}%)" || fail "Gp_reproduces_395c_G" "${KMS[Gp]} is ${d}% off $REF_G_MS"
fi
if [[ -s "$LOGDIR/apps_Gp.tsv" ]]; then
  python3 - "$LOGDIR/apps_Gp.tsv" > "$LOGDIR/apps_Gp_summary.txt" <<'EOF'
import sys, collections
rows=[[c.strip() for c in l.split(',')] for l in open(sys.argv[1]) if l.strip()]
per_t=collections.defaultdict(list)
for r in rows:
    if len(r)>=4: per_t[r[0]].append((r[1], r[2], r[3]))
mx=max((len(v) for v in per_t.values()), default=0)
mem=collections.defaultdict(list); name={}
for v in per_t.values():
    for pid,nm,mb in v:
        name[pid]=nm
        try: mem[pid].append(int(mb))
        except ValueError: pass
print(f"INFO  Gp_max_concurrent_compute_processes: {mx}")
for pid,vals in sorted(mem.items(), key=lambda x:-max(x[1])):
    vals=sorted(vals)
    print(f"INFO  Gp_process[{pid}]: {name.get(pid,'?')}  min={vals[0]} med={vals[len(vals)//2]} max={vals[-1]} MiB  samples={len(vals)}")
if mx>=2:
    print("OK    P5_two_contexts_during_G (the dispatcher does hold a context while the CRunner runs)")
else:
    print("FAIL  P5_two_contexts_during_G: only one compute process was ever visible -- nothing holds a second context during G, so D1==G is a coincidence and the two-factor model must be rebuilt")
EOF
  grep -vE "^(OK|FAIL) +P5" "$LOGDIR/apps_Gp_summary.txt"
  if grep -q "^OK    P5_two_contexts_during_G" "$LOGDIR/apps_Gp_summary.txt"; then
    pass "P5_two_contexts_during_G (the dispatcher holds a context while the CRunner runs)"
  else
    fail "P5_two_contexts_during_G" "only one compute process was ever visible during G -- nothing holds a second context, so D1 == G is a coincidence and the two-factor model must be rebuilt"
  fi
fi

for c in S255 S2303 A0 X2 B512 Gp; do [[ -n "${CLK[$c]:-}" ]] && info "clock[$c]" "${CLK[$c]}"; done
for c in S255 S2303 A0 X2 B512; do [[ -n "${LD[$c]:-}" ]] && info "placement[$c]" "d_ld=${LD[$c]} free_mb=${FREEMB[$c]:-?}"; done

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
