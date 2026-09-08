#!/usr/bin/env bash
# 394b_r2_cliff_probe.sh
#
# rev394b r2 -- characterise the 968 -> 1280 cliff, then map the curve
# between them. Same binary, same single variable (MAX_BLOCKS).
#
# WHAT r1 FOUND (2026-09-08, all five points oracle-MATCH, anchors 201,231
# / 201,237 ms = +0.003% drift):
#     484   201,231 ms   anchor
#     968   163,185 ms   -18.91%   <- real, but not where predicted
#    1280   252,207 ms   +25.33%   <- pre-registered -15..-35, REFUTED
#    1936   261,153 ms   +29.78%
# The axis works; the optimum is lower than predicted and there is a
# CLIFF between 968 and 1280 (+54.6% from one to the next).
#
# TWO HYPOTHESES, DISCRIMINATED HERE WITHOUT GUESSING
# ---------------------------------------------------
# H_L1   The per-thread local stack (208 B; 6.65 KB per warp) stops
#        fitting the L1 slice available to it. sm_86 reserves 1 KB of
#        shared memory per resident block (16 blocks -> 16 KB carveout
#        -> L1 = 112 KB). 12 warps x 6.65 = 80 KB fits; 16 x 6.65 = 106 KB
#        does not, once global-load lines are also in there.
# H_wave The A10G may have 72 SMs, not 80. Then 1280 blocks / 72 = 17.8
#        per SM > the 16-block hardware limit: a second partial wave,
#        i.e. a launch-level tail.
# Each hypothesis makes a different, cheap, directly measurable
# prediction. This harness measures rather than argues:
#
#   STAGE 1  -Xptxas -v rebuild (identical binary, plus the resource
#            report): registers, STACK FRAME BYTES, spill bytes. If the
#            frame is not 208 B or spills are nonzero, every L1 number
#            below shifts and must be recomputed. ~10 s.
#   STAGE 2a ncu --section LaunchStats --section Occupancy on a 204,800-
#            record slice at 484 / 968 / 1280. LaunchStats prints "# SMs",
#            "Waves Per SM", "Shared Memory Configuration Size", the
#            per-limit block caps and Theoretical/Achieved Occupancy --
#            the exact numbers both hypotheses hinge on. sudo, ~2 min/pt.
#   STAGE 2b ncu --metrics for the LOCAL-memory L1 hit rate at 968 vs
#            1280 on the same slice. H_L1 predicts a collapse (>=95% ->
#            well under 80%); H_wave predicts no change. sudo, ~2 min/pt.
#            Non-fatal: an unknown metric name logs and continues.
#   STAGE 3  Timing sweep 968 -> 1024 -> 1088 -> 1152 -> 1216 -> 968,
#            full N=21, oracle-gated, no sudo. ~6 x 2.8 min.
#
# PRE-REGISTERED (see 394b_README_append.md, r2 section):
#   ptxas: stack frame = 208 B, spill = 0, regs 45..48
#   LaunchStats@1280: Waves/SM = 1.00 (80 SMs) or ~1.11 (72 SMs)
#   L1 local hit @968 >= 95%; @1280 < 80% under H_L1
#   sweep: optimum in 968..1088, 1216 worse than 1152
#
# USAGE
#   STATIC_ONLY=1 bash 394b_r2_cliff_probe.sh
#                 bash 394b_r2_cliff_probe.sh            # stages 1,2a,2b,3
#   STAGES=1,3    bash 394b_r2_cliff_probe.sh            # no ncu / no sudo
#   FINE_POINTS="968 1024 1088 1152 1216 968" (default)

set -u

REV="394b_r2"
SRC_CU="${SRC_CU:-394b_kernel_maxd14.cu}"
BIN="${BIN:-394b_kernel_maxd14}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
ORACLE="${ORACLE:-314666222712}"
INPUT="${INPUT:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
STAGES="${STAGES:-1,2a,2b,3}"
SLICE_RECORDS="${SLICE_RECORDS:-204800}"      # = 5 x 40,960: >= 5 grid-stride iterations at every point
NCU_POINTS="${NCU_POINTS:-484 968 1280}"
L1_POINTS="${L1_POINTS:-968 1280}"
FINE_POINTS="${FINE_POINTS:-968 1024 1088 1152 1216 968}"
FINE_ANCHOR="${FINE_ANCHOR:-968}"
R1_968_MS="${R1_968_MS:-163184.547}"
R1_484_MS="${R1_484_MS:-201231.234}"
COOLDOWN="${COOLDOWN:-20}"
STATIC_ONLY="${STATIC_ONLY:-0}"
STAGE_PAUSE="${STAGE_PAUSE:-5}"

NCU="${NCU:-$(command -v ncu 2>/dev/null)}"
[[ -z "$NCU" && -x /usr/local/cuda/bin/ncu ]] && NCU="/usr/local/cuda/bin/ncu"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_cliff_${TS}}"
SLICE_BIN="/tmp/${REV}_slice${SLICE_RECORDS}.bin"
TSV="$LOGDIR/${REV}_fine_sweep.tsv"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }
want() { [[ ",$STAGES," == *",$1,"* ]]; }

# ---------------------------------------------------------------------
# 0. sudo -- only needed for the ncu stages, but checked FIRST when they
#    are requested (352's lesson), before any build.
# ---------------------------------------------------------------------
if want 2a || want 2b; then
  if sudo -n true 2>/dev/null; then pass "sudo_noninteractive_available"
  else fail "sudo_noninteractive_available" "'sudo -n true' failed and STAGES includes an ncu stage. Re-run with STAGES=1,3 or fix sudo."; exit 1; fi
fi

for f in "$SRC_CU" "$INPUT"; do
  if [[ -f "$f" ]]; then pass "file_present[$f]"; else fail "file_present[$f]" "not found in $(pwd)"; fi
done
if [[ ! -x "$NVCC" ]] && ! command -v nvcc >/dev/null 2>&1; then fail "nvcc_present" "$NVCC missing"; else [[ ! -x "$NVCC" ]] && NVCC="nvcc"; pass "nvcc_present[$NVCC]"; fi
if want 2a || want 2b; then
  if [[ -n "$NCU" && -x "$NCU" ]]; then pass "ncu_present[$NCU]"; else fail "ncu_present" "ncu not found"; fi
fi
grep -q 'getenv("NQ_MAX_BLOCKS")' "$SRC_CU" 2>/dev/null && pass "source_reads_NQ_MAX_BLOCKS" || fail "source_reads_NQ_MAX_BLOCKS" "wrong .cu"
if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; nvidia-smi -q -d CLOCK 2>&1; "$NVCC" --version 2>&1
  [[ -n "$NCU" ]] && "$NCU" --version 2>&1; sha256sum "$SRC_CU" "$INPUT" 2>&1
  echo "STAGES=$STAGES SLICE_RECORDS=$SLICE_RECORDS NCU_POINTS=$NCU_POINTS L1_POINTS=$L1_POINTS FINE_POINTS=$FINE_POINTS"; } > "$LOGDIR/00_env_pre.txt" 2>&1
SM_CLOCK="$(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader 2>/dev/null || echo unavailable)"
info "sm_clock" "$SM_CLOCK"

MARKER="$LOGDIR/.owner_marker"; touch "$MARKER"
reclaim_ownership() {
  find . -maxdepth 1 -newer "$MARKER" -user root -print0 2>/dev/null | xargs -0 -r sudo chown "$(id -u):$(id -g)" 2>/dev/null || true
  sudo chown -R "$(id -u):$(id -g)" "$LOGDIR" 2>/dev/null || true
}

# =====================================================================
# STAGE 1 -- rebuild with -Xptxas -v: registers / stack frame / spills
# =====================================================================
if want 1; then
  banner "STAGE 1  nvcc -Xptxas -v (resource usage; binary content unchanged)"
  rm -f "$BIN"
  "$NVCC" -O3 -arch="$ARCH" -Xptxas -v -o "$BIN" "$SRC_CU" 2>&1 | tee "$LOGDIR/01_build_ptxas_v.log"
  [[ -x "$BIN" ]] && pass "nvcc_build_succeeded" || { fail "nvcc_build_succeeded" "no binary"; exit 1; }
  # The kernel's line looks like:
  #   ptxas info : Function properties for _Z26kernel_dfs_iter_gpu_maxd14...
  #       208 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  #   ptxas info : Used 45 registers, ...
  FRAME="$(grep -A2 'kernel_dfs_iter_gpu_maxd14' "$LOGDIR/01_build_ptxas_v.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"
  SPILL_ST="$(grep -A2 'kernel_dfs_iter_gpu_maxd14' "$LOGDIR/01_build_ptxas_v.log" | grep -o '[0-9]* bytes spill stores' | head -1 | cut -d' ' -f1)"
  SPILL_LD="$(grep -A2 'kernel_dfs_iter_gpu_maxd14' "$LOGDIR/01_build_ptxas_v.log" | grep -o '[0-9]* bytes spill loads' | head -1 | cut -d' ' -f1)"
  REGS="$(grep -A3 'kernel_dfs_iter_gpu_maxd14' "$LOGDIR/01_build_ptxas_v.log" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"
  info "ptxas_kernel_resources" "registers=${REGS:-?} stack_frame_bytes=${FRAME:-?} spill_stores=${SPILL_ST:-?} spill_loads=${SPILL_LD:-?}"
  if [[ "${FRAME:-}" == "208" ]]; then pass "ptxas_stack_frame_is_208B"; else info "ptxas_stack_frame_is_208B" "got '${FRAME:-?}' -- recompute the L1 footprint table with this value"; fi
  if [[ "${SPILL_ST:-1}" == "0" && "${SPILL_LD:-1}" == "0" ]]; then pass "ptxas_zero_spill"; else info "ptxas_zero_spill" "spills present (${SPILL_ST:-?}/${SPILL_LD:-?}) -- local footprint > stack frame"; fi

  # Equivalence gate: the rebuilt binary must reproduce r1's head-slice total.
  head -c $((15488*28)) "$INPUT" > "/tmp/${REV}_head.bin"
  "./$BIN" "$NQ" "/tmp/${REV}_head.bin" "/tmp/${REV}_head_out.bin" > "$LOGDIR/02_head_probe.log" 2>&1 || true
  T="$(grep -o 'total_sum=[0-9]*' "$LOGDIR/02_head_probe.log" | head -1 | cut -d= -f2)"
  [[ "${T:-}" == "2196649880" ]] && pass "head_slice_total_reproduces (2196649880)" || { fail "head_slice_total_reproduces" "got '${T:-<none>}'"; exit 1; }
fi
[[ -x "$BIN" ]] || { fail "binary_present[$BIN]" "run STAGES including 1 first"; exit 1; }

# medium slice for the ncu stages
head -c $((SLICE_RECORDS*28)) "$INPUT" > "$SLICE_BIN"
[[ "$(stat -c%s "$SLICE_BIN")" -eq $((SLICE_RECORDS*28)) ]] && pass "slice_built[$SLICE_RECORDS records]" || { fail "slice_built" "size mismatch"; exit 1; }

export_report() {
  local rep="$1" stem="$2"
  [[ -f "$rep" ]] || { info "report_missing[$(basename "$rep")]" "nothing to export"; return 0; }
  "$NCU" --import "$rep" --page details --print-details all > "${stem}_details.txt" 2>&1 || true
  "$NCU" --import "$rep" --page raw                          > "${stem}_raw.txt"     2>&1 || true
}

# =====================================================================
# STAGE 2a -- LaunchStats + Occupancy at NCU_POINTS (the numbers both
#             hypotheses hinge on: # SMs, Waves/SM, shared-mem config,
#             occupancy limits, theoretical vs achieved)
# =====================================================================
if want 2a; then
  banner "STAGE 2a  ncu LaunchStats+Occupancy on ${SLICE_RECORDS}-record slice at: $NCU_POINTS"
  echo "Ctrl-C to skip. Starting in ${STAGE_PAUSE}s..."; sleep "$STAGE_PAUSE"
  for MB in $NCU_POINTS; do
    REP="$LOGDIR/${REV}_launchstats_mb${MB}"
    sudo env NQ_MAX_BLOCKS="$MB" "$NCU" --launch-count 1 --section LaunchStats --section Occupancy -f -o "$REP" \
        "./$BIN" "$NQ" "$SLICE_BIN" "/tmp/${REV}_ls_mb${MB}.bin" 2>&1 | tee "$LOGDIR/20_launchstats_mb${MB}.log"
    reclaim_ownership
    export_report "${REP}.ncu-rep" "$REP"
    for key in '# SMs' 'Waves Per SM' 'Shared Memory Configuration Size' 'Registers Per Thread' 'Block Limit' 'Theoretical Occupancy' 'Achieved Occupancy' 'Achieved Active Warps Per SM' 'Driver Shared Memory Per Block'; do
      grep -h -- "$key" "${REP}_details.txt" 2>/dev/null | head -4 | sed "s/^/    [mb=$MB] /"
    done
  done
  pass "stage2a_completed"
fi

# =====================================================================
# STAGE 2b -- LOCAL-memory L1 hit rate at L1_POINTS. Non-fatal.
# =====================================================================
if want 2b; then
  banner "STAGE 2b  ncu --metrics local-memory L1 hit rate at: $L1_POINTS"
  echo "Ctrl-C to skip. Starting in ${STAGE_PAUSE}s..."; sleep "$STAGE_PAUSE"
  METRICS="l1tex__t_sector_hit_rate.pct,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_hit.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,dram__bytes_read.sum,dram__bytes_write.sum,smsp__warps_active.avg.per_cycle_active,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct"
  for MB in $L1_POINTS; do
    REP="$LOGDIR/${REV}_l1local_mb${MB}"
    sudo env NQ_MAX_BLOCKS="$MB" "$NCU" --launch-count 1 --metrics "$METRICS" -f -o "$REP" \
        "./$BIN" "$NQ" "$SLICE_BIN" "/tmp/${REV}_l1_mb${MB}.bin" 2>&1 | tee "$LOGDIR/30_l1local_mb${MB}.log"
    reclaim_ownership
    if grep -q 'No metrics to collect\|Could not find metric\|ERROR' "$LOGDIR/30_l1local_mb${MB}.log"; then
      info "stage2b[mb=$MB]" "ncu reported a metric error -- see the log; the metric list may need adjusting for this ncu version. Not gating."
    else
      export_report "${REP}.ncu-rep" "$REP"
      "$NCU" --import "${REP}.ncu-rep" --page raw --csv > "${REP}_raw.csv" 2>/dev/null || true
      grep -h -E 'l1tex__t_sector_hit_rate|local_op_ld|local_op_st|warps_active|long_scoreboard|stalled_wait|dram__bytes' "${REP}_raw.txt" 2>/dev/null | sed "s/^/    [mb=$MB] /"
    fi
  done
  pass "stage2b_completed"
fi

# =====================================================================
# STAGE 3 -- fine timing sweep between the r1 optimum and the cliff
# =====================================================================
if want 3; then
  banner "STAGE 3  fine timing sweep: $FINE_POINTS  (full N=$NQ, oracle-gated, no sudo)"
  printf 'point\tmax_blocks\tstride\tk_max\ttotal_sum\tmatch\tkernel_ms\tdelta_vs_first_968_pct\tdelta_vs_r1_968_pct\tdelta_vs_r1_484_pct\tsm_clock_mhz\ttemp_c\tstart\n' > "$TSV"
  FIRST=""; i=0
  for MB in $FINE_POINTS; do
    i=$((i+1))
    banner "fine point $i/$(echo $FINE_POINTS | wc -w): NQ_MAX_BLOCKS=$MB"
    (( i > 1 )) && { echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; }
    CLK="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
    TMP="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
    START="$(date -Is)"; RUNLOG="$LOGDIR/4${i}_mb${MB}.log"
    NQ_MAX_BLOCKS="$MB" "./$BIN" "$NQ" "$INPUT" "/tmp/${REV}_mb${MB}_results.bin" "$ORACLE" 2>&1 | tee "$RUNLOG"
    TOTAL="$(grep -o 'total_sum=[0-9]*' "$RUNLOG" | head -1 | cut -d= -f2)"
    KMS="$(grep -o 'kernel_ms=[0-9.]*' "$RUNLOG" | head -1 | cut -d= -f2)"
    STRIDE="$(grep -o 'stride=[0-9]*' "$RUNLOG" | head -1 | cut -d= -f2)"
    KMAX="$(grep -o 'k_per_thread_max=[0-9]*' "$RUNLOG" | head -1 | cut -d= -f2)"
    MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$RUNLOG" && MATCH=1
    if [[ "$MATCH" -eq 1 && "${TOTAL:-}" == "$ORACLE" ]]; then pass "oracle_match[mb=$MB]"
    else fail "oracle_match[mb=$MB]" "total_sum='${TOTAL:-<none>}' match=$MATCH -- stopping"
         printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tNA\tNA\tNA\t%s\t%s\t%s\n' "$i" "$MB" "${STRIDE:-?}" "${KMAX:-?}" "${TOTAL:-?}" "$MATCH" "${KMS:-?}" "$CLK" "$TMP" "$START" >> "$TSV"; break; fi
    [[ -z "$FIRST" && "$MB" == "$FINE_ANCHOR" ]] && FIRST="$KMS"
    DA="NA"; [[ -n "$FIRST" ]] && DA="$(awk -v a="$FIRST" -v k="$KMS" 'BEGIN{printf "%+.3f",(k-a)/a*100}')"
    D968="$(awk -v a="$R1_968_MS" -v k="$KMS" 'BEGIN{printf "%+.3f",(k-a)/a*100}')"
    D484="$(awk -v a="$R1_484_MS" -v k="$KMS" 'BEGIN{printf "%+.3f",(k-a)/a*100}')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$i" "$MB" "${STRIDE:-?}" "${KMAX:-?}" "$TOTAL" "$MATCH" "$KMS" "$DA" "$D968" "$D484" "$CLK" "$TMP" "$START" >> "$TSV"
    info "point[$MB]" "kernel_ms=$KMS  vs in-session 968 ${DA}%  vs r1 968 ${D968}%  vs r1 484 ${D484}%"
  done
  echo; python3 - "$TSV" <<'EOF' 2>/dev/null || cat "$TSV"
import csv,sys
rows=list(csv.reader(open(sys.argv[1]),delimiter='\t'))
w=[max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
for r in rows: print('  '.join(c.ljust(w[i]) for i,c in enumerate(r)))
EOF
  BEST="$(awk -F'\t' 'NR>1 && $7!="?" && $7!="" {if(min==""||$7<min){min=$7;mb=$2}} END{print mb" "min}' "$TSV")"
  info "fine_sweep_best" "MAX_BLOCKS=${BEST}"
fi

{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi -q -d CLOCK 2>&1; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
rm -f "$MARKER"; reclaim_ownership
TARBALL="${LOGDIR}.tar.gz"
tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed -- send $LOGDIR/"
echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; echo "SM clock: $SM_CLOCK"; echo "logs: $LOGDIR/"; echo "tarball: $TARBALL"
if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
echo "${REV} PASSED. Send $TARBALL for analysis."
exit 0
