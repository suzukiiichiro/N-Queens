#!/usr/bin/env bash
# 421_multigpu_run.sh -- the split CRunner run: one input, K chunks, G GPUs on this machine, W machines, resumable.
# (418_multigpu_run.sh on the 419 kernel, plus WORKER_ID/WORKER_TOTAL, RESUME and SUM_ONLY.)
#
# MECHANISM (unchanged from 418). The CRunner kernel is one grid-stride launch: thread t sums records t, t+S, t+2S, ...
# (S = BLOCK x MAX_BLOCKS = 32 x 960 = 30,720) into results.bin[t]. 421_split_sched.py cuts the input at multiples of
# S, so chunk thread t holds exactly the whole run's thread-t terms that fall inside the chunk, and the ELEMENTWISE
# SUM of the chunk results.bin files is byte-identical to the single run's results.bin (421_sum_results.py checks the
# oracle and, when REF_RESULTS is given, that identity). Each chunk is an ordinary invocation of the production
# runner on its own input file, pinned to one GPU with CUDA_VISIBLE_DEVICES. Workers take chunks from a queue (mkdir
# lock). A chunk is DONE when its log has "[gpu-run-done]" and its results.bin is S x 8 bytes; re-running with the
# same LOGDIR skips done chunks -- the checkpoint/restart a 9-day single run never had.
#
# WHY 421. 418_kernel_maxd16 is the 417 runner: it carries the schedule-word overflow that made 417-r3 run3 count
# N=25 0.16 % short. 421_kernel_maxd16 is 419 renamed (kernel sha 3b67e26c, SASS gated identical to 419).
# The N=25 single-run reference is 419_r3_full25_20261104_232738/bins/G25_results.bin (sha bd2de957).
#
# NEW IN 421
#   WORKER_ID=i WORKER_TOTAL=W   this machine owns chunk k iff k % W == i (default 0 / 1 = all chunks). Every
#                                machine splits the same input the same way (the plan is deterministic), runs only
#                                its own chunks and stops with "partial"; nothing is shared at run time.
#   SUM_ONLY=1 LOGDIR=<dir>      no GPU, no split: check and sum the part results collected in <dir>
#                                (rsync every machine's <logdir>/bins/part*_results.bin, 3_part*.log and
#                                2_chunk_events*.tsv into one directory first). INPUT or RECORDS=<n> gives the plan.
#   RESUME=1                     reuse the newest ${REV}_multigpu_N<N>_x<K>_* directory instead of creating one
#                                (what a restarted spot instance runs from its boot script).
#   Cost model: the bill is GPU-hours x unit price; G GPUs or W machines shorten the wall clock, not the total.
#
# DEFAULTS are the N=25 run on a g5.12xlarge (4 x A10G):
#   N=25  INPUT=constellations_N25_7.bin.soa_ref_361.bin.d17only_417.bin.sched394f.bin  CHUNKS=8  GPUS=all
#   ORACLE=2,207,893,435,808,352  HARD: elementwise-sum total == ORACLE
#   REF_RESULTS=419_r3_full25_20261104_232738/bins/G25_results.bin adds the byte-identity check (HARD when given).
#   INFO: sum of chunk kernel_ms (the single-GPU-equivalent time; the single run took 728,149,056 ms),
#         wall-clock, per-chunk kernel_ms spread, per-GPU clock/power (clk_*gpu<g>.tsv, 5 s).
#
# USAGE (inside screen; Ctrl+A then D to detach)
#   set -o pipefail; bash 421_multigpu_run.sh 2>&1 | tee 421_multigpu25_stdout.log    # N=25, 8 chunks, all GPUs
#   RESUME=1 bash 421_multigpu_run.sh 2>&1 | tee -a 421_multigpu25_stdout.log         # continue after a stop
#   LOGDIR=421_multigpu_N25_x08_<ts> bash 421_multigpu_run.sh                         # same, naming the directory
#   WORKER_ID=2 WORKER_TOTAL=4 bash 421_multigpu_run.sh                               # machine 2 of 4 (1 GPU each)
#   SUM_ONLY=1 LOGDIR=<collected> REF_RESULTS=<G25_results.bin> bash 421_multigpu_run.sh
#   N=23 INPUT=<sched23> CHUNKS=4 ORACLE=24233937684440 bash 421_multigpu_run.sh      # what the harness does
# Every chunk's log, results.bin, the sum, the manifest copy and this summary end up in $LOGDIR (+ tarball).

set -u
REV="421"
N="${N:-25}"
INPUT="${INPUT:-constellations_N25_7.bin.soa_ref_361.bin.d17only_417.bin.sched394f.bin}"
CHUNKS="${CHUNKS:-8}"
GPUS="${GPUS:-auto}"
ORACLE="${ORACLE:-2207893435808352}"
CU_BIN="${CU_BIN:-./421_kernel_maxd16}"
BLOCK="${BLOCK:-32}"; MB="${MB:-960}"; HMB="${HMB:-128}"
STRIDE=$((BLOCK*MB))
SPLITTER="${SPLITTER:-421_split_sched.py}"
SUMMER="${SUMMER:-421_sum_results.py}"
REF_RESULTS="${REF_RESULTS:-}"
CHUNK_TIMEOUT="${CHUNK_TIMEOUT:-150h}"
CLK_INTERVAL="${CLK_INTERVAL:-5}"
WORKER_ID="${WORKER_ID:-0}"; WORKER_TOTAL="${WORKER_TOTAL:-1}"
SUM_ONLY="${SUM_ONLY:-0}"; RESUME="${RESUME:-0}"
KK="$(printf %02d "$CHUNKS")"
TS="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${LOGDIR:-}" && "$RESUME" == "1" ]]; then LOGDIR="$(ls -dt "${REV}_multigpu_N${N}_x${KK}_"*/ 2>/dev/null | head -1)"; LOGDIR="${LOGDIR%/}"; fi
LOGDIR="${LOGDIR:-${REV}_multigpu_N${N}_x${KK}_${TS}}"
CHUNKDIR="${CHUNKDIR:-${INPUT}.split${KK}}"
PREFIX="$(basename "$INPUT")"
case "$N" in 25) WANT_LAYOUT=425;; 24) WANT_LAYOUT=424;; 23) WANT_LAYOUT=423;; *) WANT_LAYOUT=403;; esac
WTAG=""; [[ "$WORKER_TOTAL" != "1" ]] && WTAG="_w${WORKER_ID}of${WORKER_TOTAL}"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
now() { date -Is; }
summary_exit() { echo; echo "===== ${REV}_multigpu N=$N x$CHUNKS${WTAG} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; [[ "$FAIL" -gt 0 ]] && exit 1; exit 0; }
own() { (( $1 % WORKER_TOTAL == WORKER_ID )); }

# ---- preconditions
[[ -f "$SUMMER" ]] && pass "summer_present[$SUMMER]" || { fail "tools" "$SUMMER missing"; summary_exit; }
if [[ -s "$INPUT" ]]; then
  [[ $(( $(stat -c %s "$INPUT") % 28 )) -eq 0 ]] && pass "input_present ($INPUT, $(( $(stat -c %s "$INPUT") / 28 )) records)" || { fail "input" "$INPUT is not a multiple of 28 B"; summary_exit; }
  RECORDS=$(( $(stat -c %s "$INPUT") / 28 ))
elif [[ "$SUM_ONLY" == "1" && -n "${RECORDS:-}" ]]; then info "input" "absent; plan from RECORDS=$RECORDS"
else fail "input" "$INPUT missing (SUM_ONLY=1 accepts RECORDS=<n> instead)"; summary_exit; fi
NSTR=$(( (RECORDS + STRIDE - 1) / STRIDE ))
[[ "$CHUNKS" -ge 1 && "$CHUNKS" -le "$NSTR" ]] && pass "chunks_${CHUNKS}_within_1..${NSTR}" || { fail "chunks" "CHUNKS=$CHUNKS must be 1..$NSTR (= ceil(records/stride))"; summary_exit; }
[[ "$WORKER_TOTAL" -ge 1 && "$WORKER_ID" -ge 0 && "$WORKER_ID" -lt "$WORKER_TOTAL" ]] && pass "worker_${WORKER_ID}_of_${WORKER_TOTAL}" || { fail "worker" "WORKER_ID=$WORKER_ID must be 0..$((WORKER_TOTAL-1))"; summary_exit; }
# the splitter's plan, in bash: the ceil(m/S) stride-groups dealt out as evenly as possible (first NSTR%K chunks get one more)
declare -a PLAN_CNT=(); _off=0
for ((k=0;k<CHUNKS;k++)); do _c=$(( (NSTR / CHUNKS + (k < NSTR % CHUNKS ? 1 : 0)) * STRIDE )); (( _c > RECORDS - _off )) && _c=$(( RECORDS - _off )); PLAN_CNT+=("$_c"); _off=$(( _off + _c )); done
[[ "$_off" == "$RECORDS" ]] || { fail "plan" "plan covers $_off of $RECORDS records"; summary_exit; }
OWN_N=0; OWN_LIST=""; for ((k=0;k<CHUNKS;k++)); do own "$k" && { OWN_N=$((OWN_N+1)); OWN_LIST+="$k "; }; done
part_file() { printf '%s/%s.part%02dof%02d.bin' "$CHUNKDIR" "$PREFIX" "$1" "$CHUNKS"; }
part_log() { printf '%s/3_part%02d.log' "$LOGDIR" "$1"; }
part_res() { printf '%s/bins/part%02d_results.bin' "$LOGDIR" "$1"; }
chunk_done() { [[ -f "$(part_res "$1")" && $(stat -c %s "$(part_res "$1")" 2>/dev/null || echo 0) -eq $((STRIDE*8)) ]] && grep -q '\[gpu-run-done\]' "$(part_log "$1")" 2>/dev/null; }

if [[ "$SUM_ONLY" == "1" ]]; then
  [[ -d "$LOGDIR/bins" ]] && pass "sum_only_logdir[$LOGDIR]" || { fail "sum_only" "LOGDIR=$LOGDIR has no bins/ (collect the part results there first)"; summary_exit; }
  RUN_START="$(now)"; RUN_END="$RUN_START"; WALL_S=0; GPUS=0
else
  [[ "$OWN_N" -ge 1 ]] && pass "own_chunks_${OWN_N}_of_${CHUNKS} (${OWN_LIST% })" || { fail "own_chunks" "worker $WORKER_ID of $WORKER_TOTAL owns no chunk with CHUNKS=$CHUNKS"; summary_exit; }
  [[ -x "$CU_BIN" ]] && pass "cu_binary_present[$CU_BIN]" || { fail "cu_binary" "$CU_BIN missing"; summary_exit; }
  [[ -f "$SPLITTER" ]] && pass "splitter_present[$SPLITTER]" || { fail "tools" "$SPLITTER missing"; summary_exit; }
  NGPU="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')"
  [[ "$GPUS" == "auto" ]] && GPUS="$NGPU"
  [[ "$GPUS" -ge 1 && "$GPUS" -le "$NGPU" ]] && pass "gpus_${GPUS}_of_${NGPU}" || { fail "gpus" "GPUS=$GPUS but nvidia-smi lists $NGPU"; summary_exit; }
  apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"
  [[ -z "$apps" ]] && pass "gpu_empty_before" || { fail "gpu_empty_before" "compute process(es) present: $apps"; summary_exit; }
  FREE_GB="$(df -BG . | awk 'NR==2{gsub("G","",$4); print $4}')"; NEED_GB=$(( $(stat -c %s "$INPUT") / 1000000000 + 2 ))
  [[ "${FREE_GB:-0}" -ge "$NEED_GB" ]] && pass "disk_free_${FREE_GB}G (chunks need ~${NEED_GB} GB once)" || { fail "disk" "only ${FREE_GB} GB free, chunks need ~${NEED_GB} GB"; summary_exit; }
  mkdir -p "$LOGDIR/bins" "$LOGDIR/lock"; rm -rf "$LOGDIR"/lock/*; pass "logdir[$LOGDIR] (reuse = resume)"
  { echo "=== $REV multigpu env (pre) $(now) N=$N INPUT=$INPUT CHUNKS=$CHUNKS GPUS=$GPUS WORKER=$WORKER_ID/$WORKER_TOTAL ==="; uname -a; nvidia-smi 2>&1; df -h . 2>&1; free -g 2>&1; } > "$LOGDIR/00_env_pre${WTAG}.txt" 2>&1

  # ---- split (idempotent; existing parts are re-hashed against the source)
  T0=$(date +%s)
  python3 "$SPLITTER" "$INPUT" "$CHUNKS" --stride "$STRIDE" --outdir "$CHUNKDIR" --prefix "$PREFIX" --verify > "$LOGDIR/1_split${WTAG}.log" 2>&1; rc=$?
  grep -E '^\[split-(verify|done)\]' "$LOGDIR/1_split${WTAG}.log"
  MAN="$CHUNKDIR/${PREFIX}.split${KK}.manifest.tsv"
  [[ "$rc" == "0" && -f "$MAN" ]] && pass "split_verified (concatenation sha == source, $(( $(date +%s)-T0 )) s)" || { fail "split" "$(tail -2 "$LOGDIR/1_split${WTAG}.log" | tr '\n' ' ')"; summary_exit; }
  cp -f "$MAN" "$LOGDIR/1_manifest.tsv"
  for ((k=0;k<CHUNKS;k++)); do [[ $(( $(stat -c %s "$(part_file "$k")") / 28 )) -eq "${PLAN_CNT[$k]}" ]] || { fail "plan_vs_split" "chunk $k: part file has $(( $(stat -c %s "$(part_file "$k")") / 28 )) records, plan ${PLAN_CNT[$k]}"; summary_exit; }; done
  pass "plan_matches_split_parts"
  PRE_DONE=0; for ((k=0;k<CHUNKS;k++)); do own "$k" && chunk_done "$k" && PRE_DONE=$((PRE_DONE+1)); done
  info "resume" "$PRE_DONE of $OWN_N own chunks already done in $LOGDIR"

  # ---- clock samplers, one per GPU
  declare -a CLKPIDS=()
  for ((g=0;g<GPUS;g++)); do
    f="$LOGDIR/clk${WTAG}_gpu${g}.tsv"; [[ -f "$f" ]] || echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"
    ( while true; do nvidia-smi -i "$g" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPIDS+=($!)
  done
  cleanup() { for p in "${CLKPIDS[@]}"; do kill "$p" 2>/dev/null; done; }
  trap cleanup EXIT

  # ---- workers: each takes the next unclaimed, not-done chunk this machine owns until none is left
  EVENTS="$LOGDIR/2_chunk_events${WTAG}.tsv"; [[ -f "$EVENTS" ]] || printf 'chunk\tgpu\tstart\tend\trc\tkernel_ms\ttotal_sum\trecords\n' > "$EVENTS"
  worker() {
    local g="$1" k
    for ((k=0;k<CHUNKS;k++)); do
      own "$k" || continue
      mkdir "$LOGDIR/lock/$k" 2>/dev/null || continue
      if chunk_done "$k"; then echo "[chunk-skip] chunk=$k already done"; continue; fi
      local in; in="$(part_file "$k")"; local recs=$(( $(stat -c %s "$in") / 28 ))
      local start; start="$(now)"; echo "[chunk-start] $start chunk=$k gpu=$g records=$recs"
      rm -f "$(part_res "$k")"
      env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX CUDA_VISIBLE_DEVICES="$g" NQ_LAYOUT=auto NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$MB" NQ_HELPER_CTX=1 NQ_HELPER_MB="$HMB" \
        timeout --signal=TERM --kill-after=60s "$CHUNK_TIMEOUT" "$CU_BIN" "$N" "$in" "$(part_res "$k")" > "$(part_log "$k")" 2>&1; local rc=$?
      local kms tot; kms="$(grep -o 'kernel_ms=[0-9.]*' "$(part_log "$k")" | head -1 | cut -d= -f2)"; tot="$(grep -o 'total_sum=[0-9]*' "$(part_log "$k")" | head -1 | cut -d= -f2)"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$k" "$g" "$start" "$(now)" "$rc" "${kms:-?}" "${tot:-?}" "$recs" >> "$EVENTS"
      echo "[chunk-done] $(now) chunk=$k gpu=$g rc=$rc kernel_ms=${kms:-?} ($(awk -v k="${kms:-0}" 'BEGIN{printf "%d:%02d:%02d", int(k/3600000), int(k/60000)%60, int(k/1000)%60}')) total_sum=${tot:-?}"
    done
  }
  RUN_START="$(now)"; T_RUN0=$(date +%s)
  info "run_start" "$RUN_START  N=$N chunks=$CHUNKS own=$OWN_N gpus=$GPUS stride=$STRIDE binary=$CU_BIN timeout/chunk=$CHUNK_TIMEOUT"
  declare -a WPIDS=()
  for ((g=0;g<GPUS;g++)); do worker "$g" & WPIDS+=($!); sleep 2; done
  for p in "${WPIDS[@]}"; do wait "$p"; done
  cleanup; trap - EXIT
  WALL_S=$(( $(date +%s)-T_RUN0 )); RUN_END="$(now)"
  info "run_end" "$RUN_END wall=${WALL_S}s ($(awk -v s="$WALL_S" 'BEGIN{printf "%d:%02d:%02d", int(s/3600), int(s/60)%60, s%60}'))"
  apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; [[ -z "$apps" ]] && pass "gpu_residue_none" || fail "gpu_residue" "$apps"
fi

# ---- per-chunk checks (every chunk present in $LOGDIR, whoever ran it)
ALL_PRESENT=1; OWN_DONE=1; NPRESENT=0; declare -a PARTS=()
for ((k=0;k<CHUNKS;k++)); do
  lg="$(part_log "$k")"
  if chunk_done "$k"; then PARTS+=("$(part_res "$k")"); NPRESENT=$((NPRESENT+1))
  else ALL_PRESENT=0; if [[ "$SUM_ONLY" != "1" ]] && own "$k"; then OWN_DONE=0; fail "chunk_${k}_incomplete" "$(tail -1 "$lg" 2>/dev/null | cut -c1-120)"; fi; continue; fi
  grep -q "\[gpu-layout\] N=$N requested=auto layout=$WANT_LAYOUT" "$lg" || fail "chunk_${k}_layout" "$(grep -o '\[gpu-layout\].*' "$lg" | head -1)"
  grep -q "\[gpu-config\] BLOCK=$BLOCK MAX_BLOCKS=$MB stride=$STRIDE" "$lg" || fail "chunk_${k}_config" "$(grep -o '\[gpu-config\].*' "$lg" | head -1)"
  grep -q 'helpers=1' "$lg" && grep -q 'extra_ctx=0' "$lg" || fail "chunk_${k}_ctx_state" "$(grep -oE 'helpers=[0-9]+|extra_ctx=[0-9]+' "$lg" | tr '\n' ' ')"
  got_recs="$(grep -o 'records=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  [[ "$got_recs" == "${PLAN_CNT[$k]}" ]] || fail "chunk_${k}_records" "records=$got_recs != plan ${PLAN_CNT[$k]}"
done
if [[ "$SUM_ONLY" != "1" ]]; then
  [[ "$OWN_DONE" == "1" ]] && pass "own_${OWN_N}_chunks_done (${OWN_LIST% })" || { info "resume_hint" "re-run with LOGDIR=$LOGDIR (or RESUME=1) to finish the missing chunks"; summary_exit; }
fi
if [[ "$ALL_PRESENT" != "1" ]]; then
  if [[ "$SUM_ONLY" == "1" ]]; then fail "sum_only_incomplete" "$NPRESENT of $CHUNKS chunks present in $LOGDIR"; summary_exit; fi
  info "partial" "$NPRESENT of $CHUNKS chunks present here (worker $WORKER_ID of $WORKER_TOTAL). Collect every machine's bins/part*_results.bin, 3_part*.log and 2_chunk_events*.tsv into one directory, then: SUM_ONLY=1 LOGDIR=<dir> bash $0"
  TARBALL="${LOGDIR}${WTAG}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
  summary_exit
fi
pass "all_${CHUNKS}_chunks_present"
[[ "$FAIL" -eq 0 ]] && pass "chunk_logs_layout_${WANT_LAYOUT}_config_${BLOCK}x${MB}_helper1_records"

# ---- elementwise sum: oracle (HARD) and, when given, byte-identity to the single run's results.bin (HARD)
SUMBIN="$LOGDIR/bins/N${N}_results_sum.bin"
SUMARGS=(--out "$SUMBIN" --expect "$ORACLE"); [[ -n "$REF_RESULTS" ]] && SUMARGS+=(--ref "$REF_RESULTS")
python3 "$SUMMER" "${SUMARGS[@]}" "${PARTS[@]}" > "$LOGDIR/4_sum.log" 2>&1; rc=$?
grep '\[sum-done\]' "$LOGDIR/4_sum.log"
TOT="$(grep -o 'total_sum=[0-9]*' "$LOGDIR/4_sum.log" | head -1 | cut -d= -f2)"
[[ "$TOT" == "$ORACLE" ]] && pass "oracle_MATCH total_sum=$TOT" || fail "oracle" "total_sum=${TOT:-?} expected $ORACLE"
if [[ -n "$REF_RESULTS" ]]; then grep -q 'ref_bytes=IDENTICAL' "$LOGDIR/4_sum.log" && pass "sum_results_byte_identical_to[$REF_RESULTS]" || fail "sum_results_vs_ref" "$(grep -o 'ref_bytes=[^ ]*' "$LOGDIR/4_sum.log")"; fi
[[ "$rc" == "0" ]] || fail "sum_tool_rc" "rc=$rc"

# ---- timing (information): sum of chunk kernel_ms = single-GPU-equivalent, wall, spread, clocks
# one row per chunk: the LAST successful row of any events file (resumed runs and other workers append / add files)
read -r KSUM KMIN KMAX KN < <(cat "$LOGDIR"/2_chunk_events*.tsv 2>/dev/null | awk -F'\t' '$1!="chunk" && $5=="0" && $6+0>0 {v[$1]=$6} END{for(k in v){s+=v[k]; n++; if(min==""||v[k]<min)min=v[k]; if(v[k]>max)max=v[k]} printf "%.1f %.1f %.1f %d\n", s, min, max, n}')
for f in "$LOGDIR"/clk*_gpu*.tsv "$LOGDIR"/clk_gpu*.tsv; do [[ -f "$f" ]] || continue; [[ "${SEEN_CLK:-}" == *"|$f|"* ]] && continue; SEEN_CLK="${SEEN_CLK:-}|$f|"
  read -r SMM SMN TM PW < <(awk -F, 'NR>1 && $2+0>0 && $6+0>50 && $4+0>150 {s+=$2; n++; p+=$4; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.1f %d %d %.0f\n", s/n, min, t, p/n; else print "? ? ? ?"}' "$f")
  info "$(basename "$f" .tsv)" "sm_mean=$SMM sm_min=$SMN temp_max=${TM}C power_mean=${PW}W (loaded samples)"
done
info "TIMING" "sum_kernel_ms=$KSUM ($(awk -v k="$KSUM" 'BEGIN{printf "%d:%02d:%04.1f", int(k/3600000), int(k/60000)%60, (k/1000)%60}')) chunks=$KN min=$KMIN max=$KMAX spread=$(awk -v a="$KMAX" -v b="$KMIN" 'BEGIN{if(b>0)printf "%.1f%%",(a/b-1)*100; else print "?"}') wall_s=$WALL_S speedup_vs_sum=$(awk -v k="$KSUM" -v w="$WALL_S" 'BEGIN{if(w>0)printf "%.2fx",k/1000/w; else print "?"}')"
[[ "$KN" == "$CHUNKS" ]] || info "TIMING_note" "kernel_ms known for $KN of $CHUNKS chunks (events files of the other machines not collected?)"
printf 'N\tchunks\tgpus\trecords\tstride\tsum_kernel_ms\tmin_chunk_ms\tmax_chunk_ms\twall_s\ttotal_sum\toracle\tstart\tend\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$N" "$CHUNKS" "$GPUS" "$RECORDS" "$STRIDE" "$KSUM" "$KMIN" "$KMAX" "$WALL_S" "${TOT:-?}" "$ORACLE" "$RUN_START" "$RUN_END" > "$LOGDIR/${REV}_multigpu_results.tsv"
sha256sum "$SUMBIN" 2>/dev/null | cut -c1-16 | sed 's/^/INFO  sum_results_sha: /'
[[ "$SUM_ONLY" == "1" ]] || { echo "=== $REV multigpu env (post) $(now) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post${WTAG}.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
summary_exit
