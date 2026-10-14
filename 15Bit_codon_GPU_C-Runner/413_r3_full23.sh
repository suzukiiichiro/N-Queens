#!/usr/bin/env bash
# 413_r3_full23.sh -- the full N=23 run on the production path (./413Py_kernel_maxd16_final -g 23 23).
#
# 413 (2026-10-08 18:21, OK=65 FAIL=0): every HARD gate held, 413 is the production binary/.py.
# This script produces the N=23 production value. No code change; no wrapper (367's ulimit -v
# segfaults the Codon binary at CUDA init), a plain timeout instead.
#   G23  ./413Py -g 23 23  -> dispatch.log env_prefix "NQ_MAX_BLOCKS=960 NQ_HELPER_CTX=1 NQ_HELPER_MB=128",
#        crunner log: layout=423, helpers=1, extra_ctx=0, BLOCK=32 MAX_BLOCKS=960, records=44,271,796,
#        oracle 24,233,937,684,440 MATCH                                                    HARD
#   Clock sampled every 5 s; norm_ms = kernel_ms * sm_mean / 1710 reported alongside.
#   Expectation (slice B scaled, 398-r2 caveat +-30%): ~8,200 s = 2 h 16 min; timeout 5 h.
#   /tmp/crunner_N23_results.bin is kept as the N=23 per-thread reference (bins/G23_results.bin).
#
# USAGE   bash 413_r3_full23.sh 2>&1 | tee 413_r3_full23_stdout.log

set -u
REV="413_r3"
PY_BIN="${PY_BIN:-413Py_kernel_maxd16_final}"
CU_BIN="${CU_BIN:-413_kernel_maxd16}"
CRLOG="${CRLOG:-413_crunner_logs}"
TIMEOUT_DURATION="${TIMEOUT_DURATION:-5h}"
CLK_INTERVAL="${CLK_INTERVAL:-5}"
ORACLE23=24233937684440
EXPECTED23=44271796
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_full23_${TS}}"
PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; [[ "$FAIL" -gt 0 ]] && exit 1; exit 0; }

[[ -x "$PY_BIN" ]] && pass "py_binary_present[$PY_BIN]" || { fail "py_binary" "$PY_BIN missing (codon build -release 413Py_kernel_maxd16_final.py)"; summary_exit; }
[[ -x "$CU_BIN" ]] && pass "cu_binary_present[$CU_BIN]" || { fail "cu_binary" "$CU_BIN missing (built by 413_validate.sh)"; summary_exit; }
SCHED23="constellations_N23_7.bin.soa_ref_361.bin.maxd16only_413.bin.sched394f.bin"
[[ -s "$SCHED23" && $(( $(stat -c %s "$SCHED23") / 28 )) -eq "$EXPECTED23" ]] && pass "sched23_present ($EXPECTED23 records)" || { fail "sched23" "$SCHED23 missing/wrong size"; summary_exit; }
mkdir -p "$LOGDIR/bins" "$CRLOG"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; df -h . 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"
[[ -z "$apps" ]] && pass "gpu_empty_before" || { fail "gpu_empty_before" "compute process(es) present: $apps"; summary_exit; }

dlog="$CRLOG/dispatch.log"; clog="$CRLOG/crunner_${CU_BIN}_N23.log"; before=0; [[ -f "$dlog" ]] && before="$(wc -l < "$dlog")"
echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$LOGDIR/clk_G23.tsv"
( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$LOGDIR/clk_G23.tsv" 2>/dev/null & CLKPID=$!
START="$(date -Is)"; info "G23_start" "$START  (./$PY_BIN -g 23 23, timeout $TIMEOUT_DURATION)"
env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX -u NQ_LAYOUT -u NQ_HELPER_CTX -u NQ_HELPER_MB -u NQ_MAX_BLOCKS -u NQ_BLOCK \
  timeout --signal=TERM --kill-after=60s "$TIMEOUT_DURATION" "./$PY_BIN" -g 23 23 > "$LOGDIR/3_G23.stdout" 2>&1; rc=$?
kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true
info "G23_end" "$(date -Is) rc=$rc"
[[ -f "$clog" ]] && cp "$clog" "$LOGDIR/3_G23.log"; [[ -f "$dlog" ]] && tail -n +"$((before+1))" "$dlog" > "$LOGDIR/3_G23_dispatch.log"
cp -f /tmp/crunner_N23_results.bin "$LOGDIR/bins/G23_results.bin" 2>/dev/null || true
apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; [[ -z "$apps" ]] && pass "gpu_residue_none" || fail "gpu_residue" "$apps"
[[ "$rc" == "0" ]] && pass "G23_rc0" || fail "G23_rc" "rc=$rc ($( [[ "$rc" == "124" ]] && echo timeout || tail -2 "$LOGDIR/3_G23.stdout" | tr '\n' ' '))"
cat "$LOGDIR/3_G23.stdout"
LG="$LOGDIR/3_G23.log"
if [[ -f "$LG" ]]; then
  KMS="$(grep -o 'kernel_ms=[0-9.]*' "$LG" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$LG" | head -1 | cut -d= -f2)"
  REC="$(grep -o 'records=[0-9]*' "$LG" | head -1 | cut -d= -f2)"; FREE="$(grep -o 'free_mb=[0-9]*' "$LG" | head -1 | cut -d= -f2)"
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 && $6+0>50 && $4+0>150 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.1f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_G23.tsv")
  NORMV="$(awk -v k="${KMS:-0}" -v s="${SMMEAN/\?/0}" 'BEGIN{ if(s>0) printf "%.1f", k*s/1710; else print "?" }')"
  grep -q '\[gpu-layout\] N=23 requested=auto layout=423' "$LG" && pass "G23_layout_423" || fail "G23_layout" "$(grep -o '\[gpu-layout\].*' "$LG" | head -1)"
  grep -q 'helpers=1' "$LG" && grep -q 'extra_ctx=0' "$LG" && pass "G23_ctx_state_helper1_extra0" || fail "G23_ctx_state" "$(grep -oE 'helpers=[0-9]+|extra_ctx=[0-9]+' "$LG" | tr '\n' ' ')"
  grep -q '\[gpu-config\] BLOCK=32 MAX_BLOCKS=960' "$LG" && pass "G23_config_32x960" || fail "G23_config" "$(grep -o '\[gpu-config\].*' "$LG" | head -1)"
  [[ "$REC" == "$EXPECTED23" ]] && pass "G23_records_$EXPECTED23" || fail "G23_records" "records=${REC:-?}"
  grep -q 'env_prefix=NQ_MAX_BLOCKS=960 NQ_HELPER_CTX=1 NQ_HELPER_MB=128' "$LOGDIR/3_G23_dispatch.log" 2>/dev/null && pass "G23_dispatch_env_prefix" || fail "G23_dispatch_env_prefix" "$(grep -o 'env_prefix=.*' "$LOGDIR/3_G23_dispatch.log" 2>/dev/null | head -1)"
  if grep -q '\[gpu-run-correctness\] MATCH' "$LG" && [[ "$TOT" == "$ORACLE23" ]]; then
    pass "G23_oracle_MATCH total_sum=$TOT"
    info "G23_PRODUCTION_VALUE" "N=23 kernel_ms=$KMS ($(awk -v k="$KMS" 'BEGIN{printf "%d:%02d:%04.1f", int(k/3600000), int(k/60000)%60, (k/1000)%60}')) norm_ms=$NORMV sm_mean=$SMMEAN sm_min=$SMMIN temp_max=${TMAX}C free_mb=$FREE"
  else fail "G23_oracle" "total_sum=${TOT:-?} expected $ORACLE23 ($(grep -o '\[gpu-run-correctness\].*' "$LG" | head -1))"; fi
  printf 'cell\tbinary\tN\tlayout\tblock\tmax_blocks\trecords\tfree_mb\tkernel_ms\tnorm_ms\ttotal_sum\tsm_mean\tsm_min\ttemp_max\tstart\nG23\t%s\t23\t423\t32\t960\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$PY_BIN" "${REC:-?}" "${FREE:-?}" "${KMS:-?}" "$NORMV" "${TOT:-?}" "$SMMEAN" "$SMMIN" "$TMAX" "$START" > "$LOGDIR/${REV}_results.tsv"
  sha256sum "$LOGDIR/bins/G23_results.bin" 2>/dev/null | cut -c1-16 | sed 's/^/INFO  G23_results_bin_sha: /'
else fail "G23_log" "no crunner log at $clog"; fi
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
summary_exit
