#!/usr/bin/env bash
# 415_r3_full24.sh -- the full N=24 run on the production path (./415Py_kernel_maxd16_final -g 24 24).
#
# Precondition: 415_r2_validate.sh passed (Y2c real-data CPU equivalence; the 16:34 GPU cells already held:
# C484 == R392 on 154,880 real records, D960/D880/D960r totals equal, W1 -> MB 960).
# This script produces the N=24 production value. No code change; no wrapper (cgroup is not needed: the
# Codon side only reads the 1.87 GB sched file), a plain timeout instead.
#   G24  ./415Py -g 24 24  -> dispatch.log env_prefix "NQ_MAX_BLOCKS=960 NQ_HELPER_CTX=1 NQ_HELPER_MB=128",
#        crunner log: layout=424, helpers=1, extra_ctx=0, BLOCK=32 MAX_BLOCKS=960, records=66,737,631,
#        oracle 227,514,171,973,736 MATCH                                                   HARD
#   Clock sampled every 5 s; norm_ms = kernel_ms * sm_mean / 1710 reported alongside (expect 1680-1695 MHz
#   at the 300 W cap for the whole run, as N=23 did).
#   Expectation (slice B D960 scaled): 92,050 s = 25.6 h; N=23's slice ran 6% dear, so ~24 h. Timeout 36 h.
#   /tmp/crunner_N24_results.bin is kept as the N=24 per-thread reference (bins/G24_results.bin).
#   Disk: the harness writes only logs; the GPU run writes 245,760 B of results.
#
# USAGE   nohup bash 415_r3_full24.sh > 415_r3_full24_stdout.log 2>&1 &     # overnight; tail -f the log

set -u
REV="415_r3"
PY_BIN="${PY_BIN:-415Py_kernel_maxd16_final}"
CU_BIN="${CU_BIN:-415_kernel_maxd16}"
CRLOG="${CRLOG:-415_crunner_logs}"
TIMEOUT_DURATION="${TIMEOUT_DURATION:-36h}"
CLK_INTERVAL="${CLK_INTERVAL:-5}"
ORACLE24=227514171973736
EXPECTED24=66737631
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_full24_${TS}}"
PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; [[ "$FAIL" -gt 0 ]] && exit 1; exit 0; }

[[ -x "$PY_BIN" ]] && pass "py_binary_present[$PY_BIN]" || { fail "py_binary" "$PY_BIN missing (codon build -release 415Py_kernel_maxd16_final.py)"; summary_exit; }
[[ -x "$CU_BIN" ]] && pass "cu_binary_present[$CU_BIN]" || { fail "cu_binary" "$CU_BIN missing (built by 415_validate.sh)"; summary_exit; }
SCHED24="constellations_N24_7.bin.soa_ref_361.bin.d16only_415.bin.sched394f.bin"
[[ -s "$SCHED24" && $(( $(stat -c %s "$SCHED24") / 28 )) -eq "$EXPECTED24" ]] && pass "sched24_present ($EXPECTED24 records)" || { fail "sched24" "$SCHED24 missing/wrong size (415_validate.sh hard-links it from the all_414 file)"; summary_exit; }
FREE_GB="$(df -BG . | awk 'NR==2{gsub("G","",$4); print $4}')"; [[ "${FREE_GB:-0}" -ge 5 ]] && pass "disk_free_${FREE_GB}G" || { fail "disk" "only ${FREE_GB} GB free"; summary_exit; }
mkdir -p "$LOGDIR/bins" "$CRLOG"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; df -h . 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"
[[ -z "$apps" ]] && pass "gpu_empty_before" || { fail "gpu_empty_before" "compute process(es) present: $apps"; summary_exit; }

dlog="$CRLOG/dispatch.log"; clog="$CRLOG/crunner_${CU_BIN}_N24.log"; before=0; [[ -f "$dlog" ]] && before="$(wc -l < "$dlog")"
echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$LOGDIR/clk_G24.tsv"
( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$LOGDIR/clk_G24.tsv" 2>/dev/null & CLKPID=$!
START="$(date -Is)"; info "G24_start" "$START  (./$PY_BIN -g 24 24, timeout $TIMEOUT_DURATION)"
env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX -u NQ_LAYOUT -u NQ_HELPER_CTX -u NQ_HELPER_MB -u NQ_MAX_BLOCKS -u NQ_BLOCK \
  timeout --signal=TERM --kill-after=60s "$TIMEOUT_DURATION" "./$PY_BIN" -g 24 24 > "$LOGDIR/3_G24.stdout" 2>&1; rc=$?
kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true
info "G24_end" "$(date -Is) rc=$rc"
[[ -f "$clog" ]] && cp "$clog" "$LOGDIR/3_G24.log"; [[ -f "$dlog" ]] && tail -n +"$((before+1))" "$dlog" > "$LOGDIR/3_G24_dispatch.log"
cp -f /tmp/crunner_N24_results.bin "$LOGDIR/bins/G24_results.bin" 2>/dev/null || true
apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; [[ -z "$apps" ]] && pass "gpu_residue_none" || fail "gpu_residue" "$apps"
[[ "$rc" == "0" ]] && pass "G24_rc0" || fail "G24_rc" "rc=$rc ($( [[ "$rc" == "124" ]] && echo timeout || tail -2 "$LOGDIR/3_G24.stdout" | tr '\n' ' '))"
cat "$LOGDIR/3_G24.stdout"
LG="$LOGDIR/3_G24.log"
if [[ -f "$LG" ]]; then
  KMS="$(grep -o 'kernel_ms=[0-9.]*' "$LG" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$LG" | head -1 | cut -d= -f2)"
  REC="$(grep -o 'records=[0-9]*' "$LG" | head -1 | cut -d= -f2)"; FREE="$(grep -o 'free_mb=[0-9]*' "$LG" | head -1 | cut -d= -f2)"
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 && $6+0>50 && $4+0>150 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.1f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_G24.tsv")
  NORMV="$(awk -v k="${KMS:-0}" -v s="${SMMEAN/\?/0}" 'BEGIN{ if(s>0) printf "%.1f", k*s/1710; else print "?" }')"
  grep -q '\[gpu-layout\] N=24 requested=auto layout=424' "$LG" && pass "G24_layout_424" || fail "G24_layout" "$(grep -o '\[gpu-layout\].*' "$LG" | head -1)"
  grep -q 'helpers=1' "$LG" && grep -q 'extra_ctx=0' "$LG" && pass "G24_ctx_state_helper1_extra0" || fail "G24_ctx_state" "$(grep -oE 'helpers=[0-9]+|extra_ctx=[0-9]+' "$LG" | tr '\n' ' ')"
  grep -q '\[gpu-config\] BLOCK=32 MAX_BLOCKS=960' "$LG" && pass "G24_config_32x960" || fail "G24_config" "$(grep -o '\[gpu-config\].*' "$LG" | head -1)"
  [[ "$REC" == "$EXPECTED24" ]] && pass "G24_records_$EXPECTED24" || fail "G24_records" "records=${REC:-?}"
  grep -q 'env_prefix=NQ_MAX_BLOCKS=960 NQ_HELPER_CTX=1 NQ_HELPER_MB=128' "$LOGDIR/3_G24_dispatch.log" 2>/dev/null && pass "G24_dispatch_env_prefix" || fail "G24_dispatch_env_prefix" "$(grep -o 'env_prefix=.*' "$LOGDIR/3_G24_dispatch.log" 2>/dev/null | head -1)"
  if grep -q '\[gpu-run-correctness\] MATCH' "$LG" && [[ "$TOT" == "$ORACLE24" ]]; then
    pass "G24_oracle_MATCH total_sum=$TOT"
    info "G24_PRODUCTION_VALUE" "N=24 kernel_ms=$KMS ($(awk -v k="$KMS" 'BEGIN{printf "%d:%02d:%04.1f", int(k/3600000), int(k/60000)%60, (k/1000)%60}')) norm_ms=$NORMV sm_mean=$SMMEAN sm_min=$SMMIN temp_max=${TMAX}C free_mb=$FREE"
  else fail "G24_oracle" "total_sum=${TOT:-?} expected $ORACLE24 ($(grep -o '\[gpu-run-correctness\].*' "$LG" | head -1))"; fi
  printf 'cell\tbinary\tN\tlayout\tblock\tmax_blocks\trecords\tfree_mb\tkernel_ms\tnorm_ms\ttotal_sum\tsm_mean\tsm_min\ttemp_max\tstart\nG24\t%s\t24\t424\t32\t960\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$PY_BIN" "${REC:-?}" "${FREE:-?}" "${KMS:-?}" "$NORMV" "${TOT:-?}" "$SMMEAN" "$SMMIN" "$TMAX" "$START" > "$LOGDIR/${REV}_results.tsv"
  sha256sum "$LOGDIR/bins/G24_results.bin" 2>/dev/null | cut -c1-16 | sed 's/^/INFO  G24_results_bin_sha: /'
else fail "G24_log" "no crunner log at $clog"; fi
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
summary_exit
