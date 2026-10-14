#!/usr/bin/env bash
# 408_ab_extra_ctx.sh -- A/B: does NQ_EXTRA_CTX=1 in the production env_prefix cost anything?
#
# WHAT: 408_kernel_maxd14, N=21 direct, layout 403, BLOCK 32 / MB 960 / helper 1 + 128 MB, three cells:
#   A0  NQ_EXTRA_CTX unset       (= every direct cell of 405..408_validate)
#   A1  NQ_EXTRA_CTX=1           (= the production -g path: env_prefix "NQ_EXTRA_CTX=1 NQ_HELPER_CTX=1 NQ_HELPER_MB=128 ")
#   A2  NQ_EXTRA_CTX unset again (brackets the within-session scatter)
#   ~7 min GPU. Production stays 408 whatever comes out; this decides only whether the next .py
#   revision drops the token from env_prefix.
#
# WHY: NQ_EXTRA_CTX=1 makes the runner create one extra in-process CUDA context (395c-r4) and was
#   called "inert since 9 Sep" when the forked helper became the external holder (402_r5), but it
#   was never measured against its absence on the same binary in one session. It also costs device
#   memory (free_mb 21,379 on G21 vs 21,890 direct).
#
# CLOCK NORMALISATION (new, from the 10/5-10/6 logs): same-SASS times moved 0.1-0.4% and the SM
#   clock mean moved the same amount the other way (408 D1 98,393 @1702.5 MHz, B1 98,100.6 @1707,
#   G21 98,024.9 @1708.6: time ratio 1.0038, clock ratio 1.0036; the GPU reached 70-74 C today
#   and dipped to 1695 MHz, vs 63-65 C and a flat 1710 yesterday). Every cell reports
#   norm_ms = kernel_ms * sm_mean / 1710 as INFO; the A/B is read on norm_ms.
#
# PRE-REGISTERED READING (fixed before running):
#   d = A1_norm / mean(A0_norm, A2_norm) - 1 ; s = |A0_norm - A2_norm| / mean
#   |d| <= max(0.10%, s)  -> INERT: drop NQ_EXTRA_CTX=1 from env_prefix in the next .py revision
#                            (one token, .py only, no kernel change; G21/G22 re-measured there)
#   d  >  +0.20%          -> COSTS TIME: same action, sooner (own .py-only revision)
#   d  <  -0.20%          -> HELPS: keep it, and that is a finding to explain (extra context = faster?)
#   otherwise             -> grey: keep as is, repeat once on another day
# HARD: A0, A1, A2 results.bin identical, oracle 314666222712, layout=403, 1710 +-2% (mean).
#
# USAGE: bash 408_ab_extra_ctx.sh          (needs 408_kernel_maxd14 or 408_kernel_maxd14.cu, N=21 input)

set -u
REV="408_ab"
CU_SRC="${CU_SRC:-408_kernel_maxd14.cu}"; CU_BIN="${CU_BIN:-408_kernel_maxd14}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"; ARCH="${ARCH:-sm_86}"
BLOCK="${BLOCK:-32}"; MB="${MB:-960}"; HMB="${HMB:-128}"; COOLDOWN="${COOLDOWN:-15}"; CLK_INTERVAL="${CLK_INTERVAL:-5}"
ORACLE21=314666222712
TS="$(date +%Y%m%d_%H%M%S)"; LOGDIR="${LOGDIR:-${REV}_${TS}}"; TSV="$LOGDIR/${REV}_results.tsv"; mkdir -p "$LOGDIR/bins"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f",(a-b)/b*100}'; }
absdiff_le() { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=t)}'; }
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi; exit 0; }

[[ ! -x "$NVCC" ]] && NVCC="nvcc"
if [[ ! -x "$CU_BIN" ]]; then [[ -f "$CU_SRC" ]] || { fail "binary_or_source" "neither $CU_BIN nor $CU_SRC"; summary_exit; }; "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc.log" 2>&1; fi
[[ -x "$CU_BIN" ]] && pass "binary[$CU_BIN]" || { fail "binary" "build failed"; summary_exit; }
find_input() { local n="$1" c; for lg in $(ls -t 40*_crunner_logs/crunner_*_N${n}.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { echo "$c"; return; }; done; ls -t constellations_N${n}_*.sched394f.bin 2>/dev/null | head -1; }
IN21="$(find_input 21)"; [[ -f "${IN21:-/nonexistent}" ]] && pass "input21_located ($IN21)" || { fail "input21_located" "no N=21 sched input"; summary_exit; }
{ echo "=== $REV env (pre) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1

printf 'cell\textra_ctx\tkernel_ms\tnorm_ms\ttotal_sum\tmatch\tlayout\tfree_mb\tsm_mean\tsm_min\ttemp_min\ttemp_max\tstart\n' > "$TSV"
CLKPID=""
gpu_gate() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; [[ -z "$apps" ]] && return 0; fail "gpu_empty_before[$1]" "$(echo "$apps" | tr '\n' ' ')"; return 1; }
gpu_after() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; [[ -z "$apps" ]] || fail "gpu_residue_after[$1]" "$(echo "$apps" | tr '\n' ' ')"; }
clk_start() { local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"; ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPID=$!; }
clk_stop() { [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""
  # busy samples only (util > 50%) so idle tail samples do not dilute the mean
  read -r SMMEAN SMMIN TMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 && $6+0>50 {s+=$2; n++; if(min==""||$2<min)min=$2; if(tmin==""||$5<tmin)tmin=$5; if($5>t)t=$5} END{if(n) printf "%.1f %d %d %d\n", s/n, min, tmin, t; else print "? ? ? ?"}' "$LOGDIR/clk_$1.tsv"); }
declare -A VAL NORM SMM
run_cell() {  # cell extra_ctx_value_or_empty
  local cell="$1" ectx="$2"; gpu_gate "$cell" || return 1
  local lg="$LOGDIR/3_${cell}.log" start; start="$(date -Is)"; clk_start "$cell"
  if [[ -z "$ectx" ]]; then
    env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX NQ_LAYOUT=auto NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$MB" NQ_HELPER_CTX=1 NQ_HELPER_MB="$HMB" "./$CU_BIN" 21 "$IN21" "$LOGDIR/bins/${cell}_results.bin" "$ORACLE21" > "$lg" 2>&1 || true
  else
    env -u NQ_PAD_MB -u NQ_CARVEOUT NQ_EXTRA_CTX="$ectx" NQ_LAYOUT=auto NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$MB" NQ_HELPER_CTX=1 NQ_HELPER_MB="$HMB" "./$CU_BIN" 21 "$IN21" "$LOGDIR/bins/${cell}_results.bin" "$ORACLE21" > "$lg" 2>&1 || true
  fi
  clk_stop "$cell"; gpu_after "$cell"
  local KMS TOT MATCH FREE LAYV CTXL NORMV
  KMS="$(grep -o 'kernel_ms=[0-9.]*' "$lg" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$lg" && MATCH=1
  FREE="$(grep -o 'free_mb=[0-9]*' "$lg" | head -1 | cut -d= -f2)"; CTXL="$(grep -o 'extra_ctx=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  LAYV="$(grep -o '\[gpu-layout\] N=[0-9]* requested=[a-z0-9]* layout=[0-9]*' "$lg" | head -1 | grep -o 'layout=[0-9]*' | cut -d= -f2)"
  NORMV="$(awk -v k="${KMS:-0}" -v s="${SMMEAN/\?/0}" 'BEGIN{ if(s>0) printf "%.1f", k*s/1710; else print "?" }')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "${ectx:-unset}" "${KMS:-?}" "$NORMV" "${TOT:-?}" "$MATCH" "${LAYV:-?}" "${FREE:-?}" "$SMMEAN" "$SMMIN" "$TMIN" "$TMAX" "$start" >> "$TSV"
  [[ "${LAYV:-}" == "403" ]] || { fail "layout[$cell]" "layout=${LAYV:-?}"; return 1; }
  [[ "$MATCH" == "1" && "${TOT:-}" == "$ORACLE21" ]] || { fail "oracle[$cell]" "total_sum='${TOT:-<none>}' match=$MATCH"; return 1; }
  VAL[$cell]="$KMS"; NORM[$cell]="$NORMV"; SMM[$cell]="$SMMEAN"
  info "$cell" "NQ_EXTRA_CTX=${ectx:-unset} runner extra_ctx=${CTXL:-?} free_mb=$FREE kernel_ms=$KMS sm_mean=$SMMEAN (min $SMMIN, temp $TMIN-$TMAX C) norm_ms=$NORMV"
}
banner "A0 unset / A1 NQ_EXTRA_CTX=1 / A2 unset  ($CU_BIN N=21 layout 403)"
run_cell A0 ""  || summary_exit
sleep "$COOLDOWN"; run_cell A1 "1" || summary_exit
sleep "$COOLDOWN"; run_cell A2 ""  || summary_exit
for c in A1 A2; do cmp -s "$LOGDIR/bins/A0_results.bin" "$LOGDIR/bins/${c}_results.bin" && pass "${c}_results_identical_to_A0" || fail "${c}_results_DIFFER" "results.bin $c != A0"; done
grep -q 'extra_ctx=1' "$LOGDIR/3_A1.log" && pass "A1_runner_reports_extra_ctx=1" || fail "A1_runner_reports_extra_ctx=1" "$(grep -o 'extra_ctx=[0-9]*' "$LOGDIR/3_A1.log" | head -1)"
grep -q 'extra_ctx=0' "$LOGDIR/3_A0.log" && pass "A0_runner_reports_extra_ctx=0" || fail "A0_runner_reports_extra_ctx=0" "$(grep -o 'extra_ctx=[0-9]*' "$LOGDIR/3_A0.log" | head -1)"
CLK_OK=1; for c in A0 A1 A2; do absdiff_le "${SMM[$c]}" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM ${SMM[$c]} MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "sm_clock_1710_pm2pct_all_cells" || fail "sm_clock_1710_pm2pct" "see clk_*.tsv"

banner "Reading (registered)"
read -r D S VERDICT < <(awk -v a0="${NORM[A0]}" -v a1="${NORM[A1]}" -v a2="${NORM[A2]}" 'BEGIN{
  m=(a0+a2)/2; d=(a1/m-1)*100; s=(a0-a2)/m*100; if(s<0)s=-s; ad=(d<0)?-d:d; thr=(s>0.10)?s:0.10;
  v="GREY_repeat_another_day"; if(ad<=thr) v="INERT_drop_token_in_next_py_revision"; else if(d>0.20) v="COSTS_TIME_drop_token_in_own_py_revision"; else if(d<-0.20) v="HELPS_keep_and_explain";
  printf "%.3f %.3f %s\n", d, s, v }')
info "raw_ms"  "A0 ${VAL[A0]}  A1 ${VAL[A1]}  A2 ${VAL[A2]}   (A1 vs A0 $(pct "${VAL[A1]}" "${VAL[A0]}")%, A2 vs A0 $(pct "${VAL[A2]}" "${VAL[A0]}")%)"
info "norm_ms" "A0 ${NORM[A0]}  A1 ${NORM[A1]}  A2 ${NORM[A2]}   (x sm_mean/1710)"
info "A/B"     "d = A1/mean(A0,A2) - 1 = ${D}%  ; within-session scatter s = ${S}%  ->  $VERDICT"
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
column -t -s $'\t' "$TSV" | tee "$LOGDIR/9_table.txt"
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball" "tar failed"
summary_exit
