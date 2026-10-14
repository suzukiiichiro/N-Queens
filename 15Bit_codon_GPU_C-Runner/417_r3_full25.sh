#!/usr/bin/env bash
# 417_r3_full25.sh -- the full N=25 run on the production path (./417Py_kernel_maxd16_final -g 25 25).
#
# Precondition: 417_validate.sh passed (2026-10-14 15:50: all HARD -- Y2e 417(425) == 417_ref_dfs on 1,024 real N=25
# records, Y2r ref == 392 on real N=24, <402>/<403>/<423>/<424> SASS = 415, <425> 192 B / loop 111; D960/D880/D960r
# totals equal and D960r == D960 byte-identical; W1 D880/D960 = 1.0641 -> MB 960 stays).
# This script produces the N=25 production value. No code change; no wrapper; a plain timeout.
#   G25  ./417Py -g 25 25  -> dispatch.log env_prefix "NQ_MAX_BLOCKS=960 NQ_HELPER_CTX=1 NQ_HELPER_MB=128",
#        crunner log: layout=425, helpers=1, extra_ctx=0, BLOCK=32 MAX_BLOCKS=960, records=98,138,478,
#        oracle 2,207,893,435,808,352 MATCH                                                 HARD
#   Clock sampled every 5 s; norm_ms = kernel_ms * sm_mean / 1710 reported alongside (slice B ran at 1699-1702 MHz
#   under the 300 W cap; expect the same for the whole run).
#   Expectation: slice B D960 3,109,581 ms / 307,200 records scaled to 98,138,478 = 993,400 s = 276 h if the slice were
#   representative; equal-spaced slices ran 6% (N=23) and 26% (N=24) dear, so 220-260 h (9-11 days). Timeout 300 h.
#   /tmp/crunner_N25_results.bin is kept as the N=25 per-thread reference (bins/G25_results.bin, 245,760 B).
#
# USAGE (inside screen; detach with Ctrl+A then D -- NOT Ctrl+Z, which suspends the job)
#   NMIN=5 bash 417_r3_full25.sh 2>&1 | tee 417_r3_full25_stdout.log     # -g 5 25: the whole table N=5..25 (+ ~23 h for N<=24), ~11 days
#          bash 417_r3_full25.sh 2>&1 | tee 417_r3_full25_stdout.log     # -g 25 25 only
#   or without screen: NMIN=5 nohup bash 417_r3_full25.sh > 417_r3_full25_stdout.log 2>&1 &

set -u
REV="417_r3"
PY_BIN="${PY_BIN:-417Py_kernel_maxd16_final}"
CU_BIN="${CU_BIN:-417_kernel_maxd16}"
CRLOG="${CRLOG:-417_crunner_logs}"
TIMEOUT_DURATION="${TIMEOUT_DURATION:-300h}"
NMIN="${NMIN:-25}"              # NMIN=5 lists N=5..25 in one -g run (the table of the README's 実行例); the N=25 checks below are unchanged. NEVER let nmax exceed 25: N=26 would start generating constellations_N26_8.bin (~1e9+ records) before the maxd check stops it
CLK_INTERVAL="${CLK_INTERVAL:-5}"
ORACLE25=2207893435808352
EXPECTED25=98138478
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_full25_${TS}}"
PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; [[ "$FAIL" -gt 0 ]] && exit 1; exit 0; }

[[ -x "$PY_BIN" ]] && pass "py_binary_present[$PY_BIN]" || { fail "py_binary" "$PY_BIN missing (codon build -release 417Py_kernel_maxd16_final.py)"; summary_exit; }
[[ -x "$CU_BIN" ]] && pass "cu_binary_present[$CU_BIN]" || { fail "cu_binary" "$CU_BIN missing (built by 417_validate.sh)"; summary_exit; }
SCHED25="constellations_N25_7.bin.soa_ref_361.bin.d17only_417.bin.sched394f.bin"
[[ -s "$SCHED25" && $(( $(stat -c %s "$SCHED25") / 28 )) -eq "$EXPECTED25" ]] && pass "sched25_present ($EXPECTED25 records)" || { fail "sched25" "$SCHED25 missing/wrong size (417_build_input25.sh)"; summary_exit; }
[[ -s "constellations_N25_7.bin" ]] && pass "stream25_present (the -g path re-scans it for required_maxd)" || { fail "stream25" "constellations_N25_7.bin missing"; summary_exit; }
FREE_GB="$(df -BG . | awk 'NR==2{gsub("G","",$4); print $4}')"; [[ "${FREE_GB:-0}" -ge 5 ]] && pass "disk_free_${FREE_GB}G" || { fail "disk" "only ${FREE_GB} GB free"; summary_exit; }
mkdir -p "$LOGDIR/bins" "$CRLOG"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; df -h . 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"
[[ -z "$apps" ]] && pass "gpu_empty_before" || { fail "gpu_empty_before" "compute process(es) present: $apps"; summary_exit; }

dlog="$CRLOG/dispatch.log"; clog="$CRLOG/crunner_${CU_BIN}_N25.log"; before=0; [[ -f "$dlog" ]] && before="$(wc -l < "$dlog")"
echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$LOGDIR/clk_G25.tsv"
( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$LOGDIR/clk_G25.tsv" 2>/dev/null & CLKPID=$!
START="$(date -Is)"; info "G25_start" "$START  (./$PY_BIN -g $NMIN 25, timeout $TIMEOUT_DURATION)"
# The -g table rows are streamed to this script's stdout (tee it) AND to $LOGDIR/3_G25.stdout. The binary is given a
# pseudo-terminal by `script` so its stdout stays line-buffered (through a plain pipe or file it would be block-buffered
# and the rows could sit in the buffer for days). -e passes the child's exit code through; -f flushes every write.
if command -v script >/dev/null 2>&1; then
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX -u NQ_LAYOUT -u NQ_HELPER_CTX -u NQ_HELPER_MB -u NQ_MAX_BLOCKS -u NQ_BLOCK \
    script -qfec "timeout --signal=TERM --kill-after=60s $TIMEOUT_DURATION ./$PY_BIN -g $NMIN 25" "$LOGDIR/3_G25.stdout"; rc=$?
else
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX -u NQ_LAYOUT -u NQ_HELPER_CTX -u NQ_HELPER_MB -u NQ_MAX_BLOCKS -u NQ_BLOCK \
    timeout --signal=TERM --kill-after=60s "$TIMEOUT_DURATION" "./$PY_BIN" -g "$NMIN" 25 2>&1 | tee "$LOGDIR/3_G25.stdout"; rc=${PIPESTATUS[0]}
fi
kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true
info "G25_end" "$(date -Is) rc=$rc"
[[ -f "$clog" ]] && cp "$clog" "$LOGDIR/3_G25.log"; [[ -f "$dlog" ]] && tail -n +"$((before+1))" "$dlog" > "$LOGDIR/3_G25_dispatch.log"
cp -f /tmp/crunner_N25_results.bin "$LOGDIR/bins/G25_results.bin" 2>/dev/null || true
apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; [[ -z "$apps" ]] && pass "gpu_residue_none" || fail "gpu_residue" "$apps"
[[ "$rc" == "0" ]] && pass "G25_rc0" || fail "G25_rc" "rc=$rc ($( [[ "$rc" == "124" ]] && echo timeout || tail -2 "$LOGDIR/3_G25.stdout" | tr '\n' ' '))"
echo "----- $LOGDIR/3_G25.stdout (the -g table, again) -----"; tr -d '\r' < "$LOGDIR/3_G25.stdout" | grep -vE '^Script (started|done)'
LG="$LOGDIR/3_G25.log"
if [[ -f "$LG" ]]; then
  KMS="$(grep -o 'kernel_ms=[0-9.]*' "$LG" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$LG" | head -1 | cut -d= -f2)"
  REC="$(grep -o 'records=[0-9]*' "$LG" | head -1 | cut -d= -f2)"; FREE="$(grep -o 'free_mb=[0-9]*' "$LG" | head -1 | cut -d= -f2)"
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 && $6+0>50 && $4+0>150 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.1f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_G25.tsv")
  NORMV="$(awk -v k="${KMS:-0}" -v s="${SMMEAN/\?/0}" 'BEGIN{ if(s>0) printf "%.1f", k*s/1710; else print "?" }')"
  grep -q '\[gpu-layout\] N=25 requested=auto layout=425' "$LG" && pass "G25_layout_425" || fail "G25_layout" "$(grep -o '\[gpu-layout\].*' "$LG" | head -1)"
  grep -q 'helpers=1' "$LG" && grep -q 'extra_ctx=0' "$LG" && pass "G25_ctx_state_helper1_extra0" || fail "G25_ctx_state" "$(grep -oE 'helpers=[0-9]+|extra_ctx=[0-9]+' "$LG" | tr '\n' ' ')"
  grep -q '\[gpu-config\] BLOCK=32 MAX_BLOCKS=960' "$LG" && pass "G25_config_32x960" || fail "G25_config" "$(grep -o '\[gpu-config\].*' "$LG" | head -1)"
  [[ "$REC" == "$EXPECTED25" ]] && pass "G25_records_$EXPECTED25" || fail "G25_records" "records=${REC:-?}"
  grep -q 'env_prefix=NQ_MAX_BLOCKS=960 NQ_HELPER_CTX=1 NQ_HELPER_MB=128' "$LOGDIR/3_G25_dispatch.log" 2>/dev/null && pass "G25_dispatch_env_prefix" || fail "G25_dispatch_env_prefix" "$(grep -o 'env_prefix=.*' "$LOGDIR/3_G25_dispatch.log" 2>/dev/null | head -1)"
  if grep -q '\[gpu-run-correctness\] MATCH' "$LG" && [[ "$TOT" == "$ORACLE25" ]]; then
    pass "G25_oracle_MATCH total_sum=$TOT"
    info "G25_PRODUCTION_VALUE" "N=25 kernel_ms=$KMS ($(awk -v k="$KMS" 'BEGIN{printf "%d:%02d:%04.1f", int(k/3600000), int(k/60000)%60, (k/1000)%60}')) norm_ms=$NORMV sm_mean=$SMMEAN sm_min=$SMMIN temp_max=${TMAX}C free_mb=$FREE"
  else fail "G25_oracle" "total_sum=${TOT:-?} expected $ORACLE25 ($(grep -o '\[gpu-run-correctness\].*' "$LG" | head -1))"; fi
  printf 'cell\tbinary\tN\tlayout\tblock\tmax_blocks\trecords\tfree_mb\tkernel_ms\tnorm_ms\ttotal_sum\tsm_mean\tsm_min\ttemp_max\tstart\nG25\t%s\t25\t425\t32\t960\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$PY_BIN" "${REC:-?}" "${FREE:-?}" "${KMS:-?}" "$NORMV" "${TOT:-?}" "$SMMEAN" "$SMMIN" "$TMAX" "$START" > "$LOGDIR/${REV}_results.tsv"
  sha256sum "$LOGDIR/bins/G25_results.bin" 2>/dev/null | cut -c1-16 | sed 's/^/INFO  G25_results_bin_sha: /'
else fail "G25_log" "no crunner log at $clog"; fi
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
summary_exit
