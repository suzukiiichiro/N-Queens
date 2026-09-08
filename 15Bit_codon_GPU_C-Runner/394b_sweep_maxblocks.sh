#!/usr/bin/env bash
# 394b_sweep_maxblocks.sh
#
# rev394b -- MAX_BLOCKS sweep on the production CUDA C kernel.
#
# WHY (393-9)
# -----------
# ncu measured Active Warps Per Scheduler = 1.51 on N=21, against a
# hardware maximum of 12 and a 4-warp ceiling for this BLOCK=32 launch
# config. 1.51 = 484 blocks / (80 SMs x 4 schedulers) exactly: the grid
# requests 12.6% of the machine's warp slots and gets exactly that.
# Inter-warp imbalance is ~0 (1.51/1.5125 = 99.8% -- every warp lives to
# the end), so the loss is not load balance, it is residency. 484x32
# predates K-batching (292); only K has been swept since.
#
# 394a added the mechanism: 16.3% of all samples are `wait` on a
# predicated BRA/BREAK (ISETP -> branch latency), 23.7% branch_resolving,
# 6.8% long_scoreboard -- all latencies that more resident warps overlap.
#
# WHAT CHANGES
# ------------
# ONE knob, at run time: NQ_MAX_BLOCKS (394b_kernel_maxd14.cu reads it;
# default 484 = 389 behaviour). BLOCK=32 and K=48 are untouched. The
# kernel SASS is identical for every point (stride is a kernel argument).
#
# SWEEP: 484 (anchor) -> 968 -> 1280 -> 1936 -> 484 (anchor again, drift
# check). 1936 is the deliberate UPPER BRACKET and is pre-registered to be
# WORSE than 1280 (61,952 threads x 208 B = 156 KB of stack per SM > the
# 128 KB L1). Each point is a full N=21 run, gated on the oracle.
#
# PRE-REGISTERED (393-11, restated so it travels with the harness):
#   kernel_ms at 1280 vs 484 anchor: -15% .. -35%
#   falsification: |delta| <= 3% at 1280 => not warp-parallelism-bound,
#                  close this axis
#   1936 worse than 1280
#
# COST: 6 runs x ~3.5 min (faster if the prediction holds) ~= 20 min.
#
# USAGE
#   STATIC_ONLY=1 bash 394b_sweep_maxblocks.sh
#                 bash 394b_sweep_maxblocks.sh
#   POINTS="484 1280 484" bash 394b_sweep_maxblocks.sh   # custom
#
# No sudo needed (no ncu). nvidia-smi is read for clocks only.

set -u

REV="394b"
SRC_CU="${SRC_CU:-394b_kernel_maxd14.cu}"
BIN="${BIN:-394b_kernel_maxd14}"
REF_CU="${REF_CU:-389_kernel_maxd14.cu}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
ORACLE="${ORACLE:-314666222712}"
INPUT="${INPUT:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
POINTS="${POINTS:-484 968 1280 1936 484}"
ANCHOR_MB="${ANCHOR_MB:-484}"
REF_KERNEL_MS="${REF_KERNEL_MS:-201237}"     # 389 real-hardware anchor (bench_mode=37, 2026-09-07)
COOLDOWN="${COOLDOWN:-20}"
STATIC_ONLY="${STATIC_ONLY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_sweep_${TS}}"
TSV="$LOGDIR/${REV}_sweep_results.tsv"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }

# ---------------------------------------------------------------------
# 1. Static checks
# ---------------------------------------------------------------------
for f in "$SRC_CU" "$INPUT"; do
  if [[ -f "$f" ]]; then pass "file_present[$f]"; else fail "file_present[$f]" "not found in $(pwd)"; fi
done

if [[ ! -x "$NVCC" ]] && ! command -v nvcc >/dev/null 2>&1; then
  fail "nvcc_present" "$NVCC not executable and 'nvcc' not on PATH"
else
  [[ ! -x "$NVCC" ]] && NVCC="nvcc"
  pass "nvcc_present[$NVCC]"
fi

# The single-variable claim, checked statically: strip both files' leading
# header comment (everything before the first #include), then the ONLY
# differences must be the two host-side blocks. Kernel + process_one_task
# must be identical. Done by diffing and counting changed lines that are
# not in the two expected hunks.
if [[ -f "$REF_CU" && -f "$SRC_CU" ]]; then
  awk 'f||/^#include/{f=1;print}' "$REF_CU" > "/tmp/${REV}_ref_code.cu"
  awk 'f||/^#include/{f=1;print}' "$SRC_CU" > "/tmp/${REV}_src_code.cu"
  # process_one_task + __global__ kernel region: from "static uint64_t process_one_task" to the line before "#ifdef __CUDACC__" that opens main()
  extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
  REF_K=$(extract_kernel "/tmp/${REV}_ref_code.cu" | sha256sum | cut -d' ' -f1)
  SRC_K=$(extract_kernel "/tmp/${REV}_src_code.cu" | sha256sum | cut -d' ' -f1)
  if [[ -n "$REF_K" && "$REF_K" == "$SRC_K" ]]; then
    pass "kernel_region_identical_to_389 (sha256 ${SRC_K:0:16}...)"
  else
    fail "kernel_region_identical_to_389" "process_one_task/__global__ region differs from $REF_CU -- 394b must not touch the kernel"
  fi
  NCHG=$(diff "/tmp/${REV}_ref_code.cu" "/tmp/${REV}_src_code.cu" | grep -c '^[<>]' || true)
  info "host_side_changed_lines_vs_389" "$NCHG (expected ~35: the NQ_MAX_BLOCKS block + the gpu-run-done line)"
else
  info "kernel_region_identical_to_389" "skipped -- $REF_CU not present for comparison"
fi

if grep -q 'getenv("NQ_MAX_BLOCKS")' "$SRC_CU" 2>/dev/null; then
  pass "source_reads_NQ_MAX_BLOCKS"
else
  fail "source_reads_NQ_MAX_BLOCKS" "the env override is missing from $SRC_CU"
fi

for p in $POINTS; do
  if [[ "$p" =~ ^[0-9]+$ ]] && (( p >= 1 && p <= 65535 )); then :; else fail "point_valid[$p]" "not an integer in [1,65535]"; fi
done
[[ "$FAIL" -eq 0 ]] && pass "sweep_points_valid[$POINTS]"

if [[ "$FAIL" -gt 0 ]]; then
  echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1
fi
if [[ "$STATIC_ONLY" == "1" ]]; then
  echo; echo "===== ${REV} STATIC_ONLY summary ====="; echo "OK=$PASS  FAIL=$FAIL"
  echo "Re-run without STATIC_ONLY=1 to build and sweep."; exit 0
fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{
  echo "=== $REV environment capture (pre) $(date -Is) ==="
  uname -a; nvidia-smi 2>&1; nvidia-smi -q -d CLOCK 2>&1; "$NVCC" --version 2>&1
  sha256sum "$SRC_CU" "$INPUT" 2>&1; [[ -f "$REF_CU" ]] && sha256sum "$REF_CU" 2>&1
  echo "POINTS=$POINTS ANCHOR_MB=$ANCHOR_MB REF_KERNEL_MS=$REF_KERNEL_MS COOLDOWN=$COOLDOWN"
} > "$LOGDIR/00_env_pre.txt" 2>&1
cp "$SRC_CU" "$LOGDIR/" 2>/dev/null || true

# ---------------------------------------------------------------------
# 2. Build (once -- the same binary serves every point)
# ---------------------------------------------------------------------
banner "Building $SRC_CU -> $BIN"
rm -f "$BIN"
"$NVCC" -O3 -arch="$ARCH" -o "$BIN" "$SRC_CU" 2>&1 | tee "$LOGDIR/01_build.log"
if [[ -x "$BIN" ]]; then pass "nvcc_build_succeeded"; else fail "nvcc_build_succeeded" "binary not produced"; exit 1; fi

# Default-path equivalence: with NQ_MAX_BLOCKS unset the binary must
# report the 389 configuration. Checked from its own [gpu-config] line
# on a trivially small run (head 15,488 records -> ~3.5 s).
head -c $((15488*28)) "$INPUT" > "/tmp/${REV}_head.bin"
"./$BIN" "$NQ" "/tmp/${REV}_head.bin" "/tmp/${REV}_head_out.bin" > "$LOGDIR/02_default_config_probe.log" 2>&1 || true
if grep -q 'BLOCK=32 MAX_BLOCKS=484 stride=15488' "$LOGDIR/02_default_config_probe.log"; then
  pass "default_config_is_389 (BLOCK=32 MAX_BLOCKS=484 stride=15488 with NQ_MAX_BLOCKS unset)"
else
  fail "default_config_is_389" "expected '[gpu-config] BLOCK=32 MAX_BLOCKS=484 stride=15488' -- see $LOGDIR/02_default_config_probe.log"
  exit 1
fi
HEAD_TOTAL="$(grep -o 'total_sum=[0-9]*' "$LOGDIR/02_default_config_probe.log" | head -1 | cut -d= -f2)"
if [[ "${HEAD_TOTAL:-}" == "2196649880" ]]; then
  pass "head_slice_total_matches_394a (2196649880)"
else
  info "head_slice_total_matches_394a" "got '${HEAD_TOTAL:-<none>}', 394a saw 2196649880 -- check before trusting the sweep"
fi

# ---------------------------------------------------------------------
# 3. Sweep
# ---------------------------------------------------------------------
printf 'point\tmax_blocks\tstride\tthreads\tk_max\ttotal_sum\tmatch\tkernel_ms\th2d_ms\td2h_ms\tdelta_vs_first_anchor_pct\tdelta_vs_389_ref_pct\tsm_clock_mhz\ttemp_c\tstart\n' > "$TSV"
FIRST_ANCHOR_MS=""
i=0
for MB in $POINTS; do
  i=$((i+1))
  banner "point $i/$(echo $POINTS | wc -w): NQ_MAX_BLOCKS=$MB  (N=$NQ full, oracle-gated)"
  if (( i > 1 )); then echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; fi
  CLK="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  TMP="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  START="$(date -Is)"
  RUNLOG="$LOGDIR/1${i}_mb${MB}.log"
  NQ_MAX_BLOCKS="$MB" "./$BIN" "$NQ" "$INPUT" "/tmp/${REV}_mb${MB}_results.bin" "$ORACLE" 2>&1 | tee "$RUNLOG"

  TOTAL="$(grep -o 'total_sum=[0-9]*' "$RUNLOG" | head -1 | cut -d= -f2)"
  KMS="$(grep -o 'kernel_ms=[0-9.]*' "$RUNLOG" | head -1 | cut -d= -f2)"
  H2D="$(grep -o 'h2d_ms=[0-9.]*' "$RUNLOG" | head -1 | cut -d= -f2)"
  D2H="$(grep -o 'd2h_ms=[0-9.]*' "$RUNLOG" | head -1 | cut -d= -f2)"
  STRIDE="$(grep -o 'stride=[0-9]*' "$RUNLOG" | head -1 | cut -d= -f2)"
  KMAX="$(grep -o 'k_per_thread_max=[0-9]*' "$RUNLOG" | head -1 | cut -d= -f2)"
  MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$RUNLOG" && MATCH=1

  # ORACLE GATE, per point. A point that fails correctness contributes no
  # timing -- it is recorded and the sweep stops.
  if [[ "$MATCH" -eq 1 && "${TOTAL:-}" == "$ORACLE" ]]; then
    pass "oracle_match[mb=$MB] total_sum=$TOTAL"
  else
    fail "oracle_match[mb=$MB]" "total_sum='${TOTAL:-<none>}' match=$MATCH -- correctness broke under this launch config; stopping"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$i" "$MB" "${STRIDE:-?}" "${STRIDE:-?}" "${KMAX:-?}" "${TOTAL:-?}" "$MATCH" "${KMS:-?}" "${H2D:-?}" "${D2H:-?}" "NA" "NA" "$CLK" "$TMP" "$START" >> "$TSV"
    break
  fi

  DELTA_A="NA"; DELTA_R="NA"
  if [[ -n "${KMS:-}" ]]; then
    if [[ -z "$FIRST_ANCHOR_MS" && "$MB" == "$ANCHOR_MB" ]]; then FIRST_ANCHOR_MS="$KMS"; fi
    [[ -n "$FIRST_ANCHOR_MS" ]] && DELTA_A="$(awk -v a="$FIRST_ANCHOR_MS" -v k="$KMS" 'BEGIN{printf "%+.3f",(k-a)/a*100}')"
    DELTA_R="$(awk -v a="$REF_KERNEL_MS" -v k="$KMS" 'BEGIN{printf "%+.3f",(k-a)/a*100}')"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$i" "$MB" "${STRIDE:-?}" "${STRIDE:-?}" "${KMAX:-?}" "$TOTAL" "$MATCH" "${KMS:-?}" "${H2D:-?}" "${D2H:-?}" "$DELTA_A" "$DELTA_R" "$CLK" "$TMP" "$START" >> "$TSV"
  info "point[$MB]" "kernel_ms=${KMS:-?}  vs first anchor ${DELTA_A}%  vs 389 ref ${DELTA_R}%  (SM ${CLK}MHz, ${TMP}C)"
done

# ---------------------------------------------------------------------
# 4. Anchor reproduction + drift, and the pre-registered checks
# ---------------------------------------------------------------------
banner "Evaluation"
if [[ -n "$FIRST_ANCHOR_MS" ]]; then
  DEV="$(awk -v a="$FIRST_ANCHOR_MS" -v r="$REF_KERNEL_MS" 'BEGIN{d=(a-r)/r*100; printf "%.3f", (d<0?-d:d)}')"
  if awk -v d="$DEV" 'BEGIN{exit !(d<=3.0)}'; then
    pass "anchor_reproduces_389_within_3pct (${FIRST_ANCHOR_MS}ms vs ${REF_KERNEL_MS}ms, ${DEV}%)"
  else
    fail "anchor_reproduces_389_within_3pct" "first 484 point is ${DEV}% off the 389 reference -- session baseline shifted; compare within-session only"
  fi
fi
ANCHORS=$(awk -F'\t' -v mb="$ANCHOR_MB" 'NR>1 && $2==mb && $8!="?" {print $8}' "$TSV")
NA=$(echo "$ANCHORS" | grep -c . || true)
if (( NA >= 2 )); then
  A1=$(echo "$ANCHORS" | head -1); A2=$(echo "$ANCHORS" | tail -1)
  DRIFT="$(awk -v a="$A1" -v b="$A2" 'BEGIN{printf "%+.3f",(b-a)/a*100}')"
  info "anchor_drift_first_vs_last" "${A1}ms -> ${A2}ms = ${DRIFT}%  (noise floor is 0.03-0.29%; beyond ~1% suspect thermal/clock drift)"
fi
P1280=$(awk -F'\t' 'NR>1 && $2==1280 && $8!="?" {print $8; exit}' "$TSV")
P1936=$(awk -F'\t' 'NR>1 && $2==1936 && $8!="?" {print $8; exit}' "$TSV")
if [[ -n "$FIRST_ANCHOR_MS" && -n "${P1280:-}" ]]; then
  D="$(awk -v a="$FIRST_ANCHOR_MS" -v k="$P1280" 'BEGIN{printf "%+.2f",(k-a)/a*100}')"
  info "prereg_1280_vs_anchor" "${D}%  (pre-registered: -15 .. -35; falsified if |delta| <= 3)"
fi
if [[ -n "${P1280:-}" && -n "${P1936:-}" ]]; then
  D="$(awk -v a="$P1280" -v k="$P1936" 'BEGIN{printf "%+.2f",(k-a)/a*100}')"
  info "prereg_1936_worse_than_1280" "1936 vs 1280 = ${D}%  (pre-registered: positive = worse)"
fi

{ echo "=== $REV environment capture (post) $(date -Is) ==="; nvidia-smi -q -d CLOCK 2>&1; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1

echo; echo "===== ${REV} results ====="; column -t -s $'\t' "$TSV" 2>/dev/null || cat "$TSV"
TARBALL="${LOGDIR}.tar.gz"
tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed -- send $LOGDIR/ as-is"
echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; echo "tsv: $TSV"; echo "tarball: $TARBALL"
if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
echo "${REV} PASSED. Send $TARBALL for analysis."
exit 0
