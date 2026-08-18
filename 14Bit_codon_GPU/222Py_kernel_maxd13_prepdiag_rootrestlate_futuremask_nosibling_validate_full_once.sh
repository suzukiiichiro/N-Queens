#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# NQ_UPDATE_MEMO
# 222: 221Py rootrestlate_inlinexor は正当性OKだが217最速を上回らず不採用。
#     222Pyは217Py rootrestlate新最速基準相当へ戻し、MAXD14探索ロジックを
#     変えず、MAXD13化の可否を見るため save_sp / cur_depth / terminal_depth
#     診断を追加する。
# Full update history: see README.md
# =============================================================================

# 222 runtime validation harness
#
# Parent:
#   217Py rootrestlate fastest baseline. 218/219/220/221 recombinations were
#   correctness OK but did not beat 217, so 222 does not continue predicate
#   recombination.
#
# Experiment:
#   MAXD14 kernel only: keep rootrestlate/root_first late extraction,
#   root_after_second scalar predicate, future_check_mask zero guard,
#   no-sibling spill elision, and root one/two preroll. Add diagnostic output
#   for max_save_sp, save_sp13_count, max_cur_depth, max_terminal_depth, and
#   root_pc_max to judge whether a later MAXD13 kernel can be safely attempted.
#
# Expected N=21 dispatch:
#   required=14, selected MAXD14, schedule_words=0, stack=208 bytes/thread.
#
# Validation:
#   one N=21 full GPU run + progress TSV reconstruction + diagnostic rows.

SRC=${SRC:-./222Py_kernel_maxd13_prepdiag_rootrestlate_futuremask_nosibling_probe.py}
CAND=${CAND:-./222Py_kernel_maxd13_prepdiag_rootrestlate_futuremask_nosibling_probe}
AUTO_BUILD=${AUTO_BUILD:-1}
STATIC_ONLY=${STATIC_ONLY:-0}
LOCK_FILE=${LOCK_FILE:-/tmp/222Py_kernel_maxd13_prepdiag_rootrestlate_futuremask_nosibling_validation.lock}
LOG_ROOT=${LOG_ROOT:-.}

N=${N:-21}
BLOCK=${BLOCK:-32}
MAX_BLOCKS=${MAX_BLOCKS:-484}
LOG_LEVEL=${LOG_LEVEL:-1}
SORT_MODE=${SORT_MODE:-0}
PRESET_QUEENS=${PRESET_QUEENS:-7}
BENCH_MODE=${BENCH_MODE:-31}
REORDER_WINDOW_MULT=${REORDER_WINDOW_MULT:-8}
REORDER_PHASE_JUMP=${REORDER_PHASE_JUMP:-7}
CROSS_STRIPE_SAFE=${CROSS_STRIPE_SAFE:-0}
WORKER_ID=${WORKER_ID:-0}
WORKER_COUNT=${WORKER_COUNT:-1}
BROADMARK_VARIANT=${BROADMARK_VARIANT:-2}

EXPECTED_CHUNKS=131
EXPECTED_TASKS=2025282
EXPECTED_REQUIRED_MAXD=14
EXPECTED_SELECTED_MAXD=14
EXPECTED_SCHEDULE_WORDS=0
EXPECTED_STACK_BYTES=208
EXP01=16879968420
EXP02=19219113480
EXP03=16935522136
EXP04=16230260724
EXP05=79383179384
FULL_TOTAL=314666222712

BASELINE_221_SECONDS=${BASELINE_221_SECONDS:-428.033}
BASELINE_220_SECONDS=${BASELINE_220_SECONDS:-427.959}
BASELINE_219_SECONDS=${BASELINE_219_SECONDS:-427.776}
BASELINE_218_SECONDS=${BASELINE_218_SECONDS:-427.825}
BASELINE_217_SECONDS=${BASELINE_217_SECONDS:-427.709}
BASELINE_204_SECONDS=${BASELINE_204_SECONDS:-427.795}
BASELINE_197_SECONDS=${BASELINE_197_SECONDS:-428.202}
BASELINE_188_SECONDS=${BASELINE_188_SECONDS:-430.137}
BASELINE_184_SECONDS=${BASELINE_184_SECONDS:-435.635}
BASELINE_175_SECONDS=${BASELINE_175_SECONDS:-471.106}

if [[ "$N" != "21" || "$BENCH_MODE" != "31" || "$WORKER_ID" != "0" || "$WORKER_COUNT" != "1" ]]; then
  echo "[error] fixed validation totals require N=21, BENCH_MODE=31, WORKER_ID=0, WORKER_COUNT=1" >&2
  exit 64
fi

if [[ "$STATIC_ONLY" != "1" ]] && command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[error] another 222 validation holds: $LOCK_FILE" >&2
    exit 75
  fi
fi

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="${LOG_ROOT%/}/222Py_kernel_maxd13_prepdiag_rootrestlate_futuremask_nosibling_logs_N21_full_once_${TS}"
RUN_LOG="$LOGDIR/full_once.log"
BUILD_LOG="$LOGDIR/build.log"
SUMMARY="$LOGDIR/summary.tsv"
METRICS="$LOGDIR/metrics.env"
DISPATCH_METRICS="$LOGDIR/dispatch_metrics.env"
DIAG_METRICS="$LOGDIR/maxd13_diag_metrics.env"
ARCHIVED_PROGRESS="$LOGDIR/progress_full.tsv"
mkdir -p "$LOGDIR"
printf 'check\texpected\tactual\tstatus\n' > "$SUMMARY"

record_check() {
  local name=$1 expected=$2 actual=$3
  local status=FAIL
  if [[ "$actual" == "$expected" ]]; then status=OK; fi
  printf '%s\t%s\t%s\t%s\n' "$name" "$expected" "$actual" "$status" >> "$SUMMARY"
  [[ "$status" == OK ]]
}

if [[ ! -f "$SRC" ]]; then
  echo "[error] source not found: $SRC" >&2
  exit 66
fi

static_failures=0
if grep -q 'future_check_mask:u32=u32(0)' "$SRC" && grep -q 'if future_check_mask!=u32(0):' "$SRC"; then
  printf 'source_future_check_mask_guard\tpresent\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_future_check_mask_guard\tpresent\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q 'save_sp:int=0' "$SRC" && grep -q 'cur_depth:int=0' "$SRC" && grep -q 'avail\[save_sp\]=cur_avail|(u32(cur_depth)<<u32(27))' "$SRC"; then
  printf 'source_nosibling_parent\t188-style save_sp/cur_depth\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_nosibling_parent\t188-style save_sp/cur_depth\tcheck source\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q 'root_after_second:u32=root_rest\^root_second' "$SRC" && grep -q 'if root_after_second==u32(0):' "$SRC" && grep -q 'cur_avail=root_rest' "$SRC"; then
  printf 'source_rootrestlate_scalar_predicate\tpresent\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_rootrestlate_scalar_predicate\tpresent\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q 'diag_save_sp:Ptr\[u32\]' "$SRC" && grep -q '\[maxd13-diag\]' "$SRC" && grep -q 'max_save_sp' "$SRC" && grep -q 'save_sp13_count' "$SRC"; then
  printf 'source_maxd13_diag\tpresent\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_maxd13_diag\tpresent\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if ! grep -q 'root_three_preroll' "$SRC" && ! grep -q 'root_rest_tail:u32=root_rest&(root_rest-u32(1))' "$SRC"; then
  printf 'source_195_rootthree_removed\tabsent\tabsent\tOK\n' >> "$SUMMARY"
else
  printf 'source_195_rootthree_removed\tabsent\tpresent\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if ! grep -q 'block_check_mask:u32' "$SRC" && ! grep -q 'if block_check_mask' "$SRC"; then
  printf 'source_190_blockmask_guard_removed\tabsent\tabsent\tOK\n' >> "$SUMMARY"
else
  printf 'source_190_blockmask_guard_removed\tabsent\tpresent\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if ! grep -q 'cur_ld\^bit' "$SRC" && ! grep -q 'cur_rd\^bit' "$SRC" && ! grep -q 'cur_col\^bit' "$SRC"; then
  printf 'source_191_place_xor_removed\tabsent\tabsent\tOK\n' >> "$SUMMARY"
else
  printf 'source_191_place_xor_removed\tabsent\tpresent\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if ! grep -q 'fast_nibble_op:u32=u32(0)' "$SRC" && ! grep -q 'fast_child_jmark:u32=' "$SRC"; then
  printf 'source_189_forced_chain_removed\tabsent\tabsent\tOK\n' >> "$SUMMARY"
else
  printf 'source_189_forced_chain_removed\tabsent\tpresent\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if ! grep -q 'if child_jmark_mask!=u32(0):' "$SRC"; then
  printf 'source_187_jmark_guard_removed\tabsent\tabsent\tOK\n' >> "$SUMMARY"
else
  printf 'source_187_jmark_guard_removed\tabsent\tpresent\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if ! grep -q 'if cur_depth==13:' "$SRC"; then
  printf 'source_terminal13_literal_removed\tabsent\tabsent\tOK\n' >> "$SUMMARY"
else
  printf 'source_terminal13_literal_removed\tabsent\tpresent\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q 'split222' "$SRC"; then
  printf 'source_split_tag\tsplit222 active preferred\tsplit222\tOK\n' >> "$SUMMARY"
else
  printf 'source_split_tag\tsplit222 active preferred\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "================================================================"
  echo "[static-summary]"
  column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
  echo "[logdir] $LOGDIR"
  if (( static_failures != 0 )); then exit 1; fi
  exit 0
fi

if (( static_failures != 0 )); then
  echo "================================================================"
  echo "[static-summary]"
  column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
  echo "[logdir] $LOGDIR"
  echo "[error] 222 source static checks failed" >&2
  exit 66
fi

need_build=0
if [[ ! -x "$CAND" ]]; then
  need_build=1
elif [[ "$SRC" -nt "$CAND" ]]; then
  need_build=1
fi

if (( need_build )); then
  if [[ "$AUTO_BUILD" != "1" ]]; then
    echo "[error] candidate is missing/stale and AUTO_BUILD=$AUTO_BUILD: $CAND" >&2
    exit 66
  fi
  if ! command -v codon >/dev/null 2>&1; then
    echo "[error] codon was not found; cannot build $SRC" >&2
    exit 69
  fi
  echo "[build] codon build -release $SRC" | tee "$BUILD_LOG"
  set +e
  codon build -release "$SRC" 2>&1 | tee -a "$BUILD_LOG"
  build_rc=${PIPESTATUS[0]}
  set -e
  printf 'build_exit\t0\t%d\t%s\n' "$build_rc" "$([[ $build_rc -eq 0 ]] && echo OK || echo FAIL)" >> "$SUMMARY"
  if (( build_rc != 0 )); then exit "$build_rc"; fi
else
  echo "[build] reuse executable: $CAND" | tee "$BUILD_LOG"
fi

if [[ ! -x "$CAND" ]]; then
  echo "[error] executable not found after build: $CAND" >&2
  exit 66
fi

CMD=(
  "$CAND"
  -g "$N" "$N"
  "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS"
  "$BENCH_MODE" "$REORDER_WINDOW_MULT" "$REORDER_PHASE_JUMP"
  "$CROSS_STRIPE_SAFE" "$WORKER_ID" "$WORKER_COUNT" "$BROADMARK_VARIANT"
)

{
  echo "================================================================"
  echo "candidate : $CAND"
  echo "source    : $SRC"
  echo "date      : $(date -Is)"
  echo "cwd       : $(pwd)"
  if command -v sha256sum >/dev/null 2>&1; then
    echo "source_sha256: $(sha256sum "$SRC" | awk '{print $1}')"
  fi
  printf 'command   :'; printf ' %q' "${CMD[@]}"; echo
  echo "validation: one full run; cases 01-05 reconstructed from its progress TSV"
  echo "diagnostic: [maxd13-diag] rows are informational; max_save_sp<=12 would support a later MAXD13-only candidate, max_save_sp==13 requires fallback or further proof"
  echo "================================================================"
} | tee "$RUN_LOG"

set +e
stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "$RUN_LOG"
run_rc=${PIPESTATUS[0]}
set -e
printf 'run_exit\t0\t%d\t%s\n' "$run_rc" "$([[ $run_rc -eq 0 ]] && echo OK || echo FAIL)" >> "$SUMMARY"
if (( run_rc != 0 )); then exit "$run_rc"; fi

awk '
  /^\[maxd-dispatch\] N=21 scope=split145 / {
    rows++; m=0; req=-1; sel=-1; words=-1; bytes=-1; cap=""
    for (i=1;i<=NF;i++) { split($i,a,"=");
      if (a[1]=="m") m=a[2]+0; else if (a[1]=="required_maxd") req=a[2]+0;
      else if (a[1]=="selected_MAXD") sel=a[2]+0; else if (a[1]=="schedule_words") words=a[2]+0;
      else if (a[1]=="stack_bytes_per_thread") bytes=a[2]+0; else if (a[1]=="capacity_check") cap=a[2]
    }
    tasks+=m; if (rows==1 || req<minreq) minreq=req; if (req>maxreq) maxreq=req;
    if (req!=14) badreq++; if (sel!=14) badsel++; if (words!=0) badwords++; if (bytes!=208) badbytes++; if (cap!="OK") badcap++; if (req>sel) under++
  }
  END {
    printf "DISPATCH_ROWS=%d\n",rows+0; printf "DISPATCH_TASKS=%.0f\n",tasks+0;
    printf "DISPATCH_MIN_REQUIRED=%d\n",minreq+0; printf "DISPATCH_MAX_REQUIRED=%d\n",maxreq+0;
    printf "DISPATCH_NON14=%d\n",badreq+0; printf "DISPATCH_NON_SELECTED14=%d\n",badsel+0;
    printf "DISPATCH_NON0WORDS=%d\n",badwords+0; printf "DISPATCH_NON208=%d\n",badbytes+0;
    printf "DISPATCH_BAD_CAPACITY=%d\n",badcap+0; printf "DISPATCH_UNDERSIZED=%d\n",under+0
  }
' "$RUN_LOG" > "$DISPATCH_METRICS"
source "$DISPATCH_METRICS"

failures=0
record_check "dispatch_launch_rows" "$EXPECTED_CHUNKS" "$DISPATCH_ROWS" || failures=$((failures+1))
record_check "dispatch_task_sum" "$EXPECTED_TASKS" "$DISPATCH_TASKS" || failures=$((failures+1))
record_check "dispatch_min_required" "$EXPECTED_REQUIRED_MAXD" "$DISPATCH_MIN_REQUIRED" || failures=$((failures+1))
record_check "dispatch_max_required" "$EXPECTED_REQUIRED_MAXD" "$DISPATCH_MAX_REQUIRED" || failures=$((failures+1))
record_check "dispatch_non_required14" "0" "$DISPATCH_NON14" || failures=$((failures+1))
record_check "dispatch_non_MAXD14" "0" "$DISPATCH_NON_SELECTED14" || failures=$((failures+1))
record_check "dispatch_non_0_schedule_words" "0" "$DISPATCH_NON0WORDS" || failures=$((failures+1))
record_check "dispatch_non_208_bytes" "0" "$DISPATCH_NON208" || failures=$((failures+1))
record_check "dispatch_bad_capacity_flag" "0" "$DISPATCH_BAD_CAPACITY" || failures=$((failures+1))
record_check "dispatch_undersized_launch" "0" "$DISPATCH_UNDERSIZED" || failures=$((failures+1))

awk '
  /^\[maxd13-diag\] N=21 scope=split145 / {
    rows++; m=0; ms=-1; c13=0; cd=-1; td=-1; rpc=-1; sel=-1;
    for (i=1;i<=NF;i++) { split($i,a,"=");
      if (a[1]=="m") m=a[2]+0; else if (a[1]=="selected_MAXD") sel=a[2]+0;
      else if (a[1]=="max_save_sp") ms=a[2]+0; else if (a[1]=="save_sp13_count") c13=a[2]+0;
      else if (a[1]=="max_cur_depth") cd=a[2]+0; else if (a[1]=="max_terminal_depth") td=a[2]+0; else if (a[1]=="root_pc_max") rpc=a[2]+0
    }
    tasks+=m; sp13+=c13; if (ms>maxsp) maxsp=ms; if (cd>maxcur) maxcur=cd; if (td>maxterm) maxterm=td; if (rpc>maxroot) maxroot=rpc; if (sel!=14) badsel++
  }
  END {
    printf "DIAG_ROWS=%d\n",rows+0; printf "DIAG_TASKS=%.0f\n",tasks+0; printf "DIAG_MAX_SAVE_SP=%d\n",maxsp+0;
    printf "DIAG_SAVE_SP13_COUNT=%.0f\n",sp13+0; printf "DIAG_MAX_CUR_DEPTH=%d\n",maxcur+0;
    printf "DIAG_MAX_TERMINAL_DEPTH=%d\n",maxterm+0; printf "DIAG_ROOT_PC_MAX=%d\n",maxroot+0; printf "DIAG_BAD_SELECTED=%d\n",badsel+0
  }
' "$RUN_LOG" > "$DIAG_METRICS"
source "$DIAG_METRICS"
record_check "maxd13_diag_rows" "$EXPECTED_CHUNKS" "$DIAG_ROWS" || failures=$((failures+1))
record_check "maxd13_diag_task_sum" "$EXPECTED_TASKS" "$DIAG_TASKS" || failures=$((failures+1))
record_check "maxd13_diag_non_MAXD14" "0" "$DIAG_BAD_SELECTED" || failures=$((failures+1))
printf 'maxd13_diag_max_save_sp\tINFO\t%s\tINFO\n' "$DIAG_MAX_SAVE_SP" >> "$SUMMARY"
printf 'maxd13_diag_save_sp13_count\tINFO\t%s\tINFO\n' "$DIAG_SAVE_SP13_COUNT" >> "$SUMMARY"
printf 'maxd13_diag_max_cur_depth\tINFO\t%s\tINFO\n' "$DIAG_MAX_CUR_DEPTH" >> "$SUMMARY"
printf 'maxd13_diag_max_terminal_depth\tINFO\t%s\tINFO\n' "$DIAG_MAX_TERMINAL_DEPTH" >> "$SUMMARY"
printf 'maxd13_diag_root_pc_max\tINFO\t%s\tINFO\n' "$DIAG_ROOT_PC_MAX" >> "$SUMMARY"
if [[ "$DIAG_MAX_SAVE_SP" -le 12 ]]; then
  printf 'maxd13_candidate_readiness\tINFO\tmax_save_sp<=12: MAXD13 candidate may be attempted with fallback validation\tINFO\n' >> "$SUMMARY"
else
  printf 'maxd13_candidate_readiness\tINFO\tmax_save_sp>=13: do not remove MAXD14 fallback yet\tINFO\n' >> "$SUMMARY"
fi

DYNAMIC_PRESET=$(sed -n 's/^\[dynamic-preset\] N=21 preset_queens=\([0-9][0-9]*\)$/\1/p' "$RUN_LOG" | tail -n1)
record_check "dynamic_preset_N21" "6" "${DYNAMIC_PRESET:-missing}" || failures=$((failures+1))

PROGRESS=$(sed -n 's/.* progress=\([^[:space:]]*\.tsv\).*/\1/p' "$RUN_LOG" | tail -n 1 | tr -d '\r')
if [[ -z "$PROGRESS" || ! -s "$PROGRESS" ]]; then
  printf 'progress_file\tpresent\t%s\tFAIL\n' "${PROGRESS:-missing}" >> "$SUMMARY"
  echo "[error] full-run progress TSV was not found" >&2
  exit 65
fi
cp -f "$PROGRESS" "$ARCHIVED_PROGRESS"
printf 'progress_file\tpresent\t%s\tOK\n' "$PROGRESS" >> "$SUMMARY"

awk -F '\t' -v expected_chunks="$EXPECTED_CHUNKS" '
  NR==1 { for (i=1;i<=NF;i++) { if ($i=="chunk") chunk_col=i; else if ($i=="chunk_total") total_col=i; else if ($i=="gpu_total") gpu_col=i } if (!chunk_col || !total_col || !gpu_col) { print "PARSE_OK=0"; exit 2 } next }
  { chunk=$(chunk_col)+0; value=$(total_col)+0; gpu=$(gpu_col)+0; rows++; full+=value; last_gpu=gpu; if (seen[chunk]++) duplicates++;
    if (chunk==0 || chunk==20 || chunk==40 || chunk==60 || chunk==80 || chunk==100 || chunk==120) p1+=value;
    if (chunk==35 || chunk==40 || chunk==41 || chunk==42 || chunk==47 || chunk==48 || chunk==52 || chunk==53) p2+=value;
    if (chunk==20 || chunk==40 || chunk==55 || chunk==56 || chunk==57 || chunk==58 || chunk==60) p3+=value;
    if (chunk==100 || chunk==105 || chunk==110 || chunk==115 || chunk==120 || chunk==125 || chunk==130) p4+=value;
    if ((chunk%4)==0) p5+=value }
  END { if (!chunk_col || !total_col || !gpu_col) exit; for (i=0;i<expected_chunks;i++) if (!(i in seen)) missing++;
    printf "PARSE_OK=1\nROWS=%.0f\nDUPLICATES=%.0f\nMISSING=%.0f\n",rows,duplicates,missing;
    printf "P1=%.0f\nP2=%.0f\nP3=%.0f\nP4=%.0f\nP5=%.0f\n",p1,p2,p3,p4,p5;
    printf "FULL=%.0f\nLAST_GPU=%.0f\n",full,last_gpu }
' "$ARCHIVED_PROGRESS" > "$METRICS"
source "$METRICS"

record_check "progress_parse" "1" "$PARSE_OK" || failures=$((failures+1))
record_check "progress_rows" "$EXPECTED_CHUNKS" "$ROWS" || failures=$((failures+1))
record_check "duplicate_chunks" "0" "$DUPLICATES" || failures=$((failures+1))
record_check "missing_chunks" "0" "$MISSING" || failures=$((failures+1))
record_check "01_standard_7chunk" "$EXP01" "$P1" || failures=$((failures+1))
record_check "02_heavy_band" "$EXP02" "$P2" || failures=$((failures+1))
record_check "03_d2base14_density" "$EXP03" "$P3" || failures=$((failures+1))
record_check "04_late_tail" "$EXP04" "$P4" || failures=$((failures+1))
record_check "05_worker0of4_derived" "$EXP05" "$P5" || failures=$((failures+1))
record_check "06_full_chunk_sum" "$FULL_TOTAL" "$FULL" || failures=$((failures+1))
record_check "06_full_last_gpu" "$FULL_TOTAL" "$LAST_GPU" || failures=$((failures+1))

FINAL_LINE=$(grep -E "^[[:space:]]*${N}:" "$RUN_LOG" | tail -n1 || true)
if [[ "$FINAL_LINE" == *"$FULL_TOTAL"* && "$FINAL_LINE" == *"ok"* ]]; then
  printf 'final_output\t%s ... ok\t%s\tOK\n' "$FULL_TOTAL" "$FINAL_LINE" >> "$SUMMARY"
else
  printf 'final_output\t%s ... ok\t%s\tFAIL\n' "$FULL_TOTAL" "${FINAL_LINE:-missing}" >> "$SUMMARY"; failures=$((failures+1))
fi

ERROR_HITS=$(grep -Eic '\[(.*-)?error\]|mismatch|ng\(' "$RUN_LOG" || true)
record_check "error_or_mismatch_hits" "0" "$ERROR_HITS" || failures=$((failures+1))

ELAPSED_TEXT=$(awk -v n="$N" '$0 ~ "^[[:space:]]*" n ":" {v=$(NF-1)} END {print v}' "$RUN_LOG")
if [[ -n "$ELAPSED_TEXT" ]]; then
  ELAPSED_SECONDS=$(awk -F: '{printf "%.3f",($1*3600)+($2*60)+$3}' <<< "$ELAPSED_TEXT")
  for pair in "221:$BASELINE_221_SECONDS" "220:$BASELINE_220_SECONDS" "219:$BASELINE_219_SECONDS" "218:$BASELINE_218_SECONDS" "217:$BASELINE_217_SECONDS" "204:$BASELINE_204_SECONDS" "197:$BASELINE_197_SECONDS" "188:$BASELINE_188_SECONDS" "184:$BASELINE_184_SECONDS" "175:$BASELINE_175_SECONDS"; do
    label=${pair%%:*}; base=${pair#*:}
    perf=$(awk -v base="$base" -v now="$ELAPSED_SECONDS" 'BEGIN {d=base-now;p=(base>0)?d/base*100:0;printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%",now,base,d,p}')
    printf 'timing_vs_%s\t%s_full=%ss\t%s\tINFO\n' "$label" "$label" "$base" "$perf" >> "$SUMMARY"
  done
fi

WARN_HITS=$(grep -Eic '\[.*warning\]' "$RUN_LOG" || true)
printf 'warning_hits\t0 preferred\t%s\tINFO\n' "$WARN_HITS" >> "$SUMMARY"

echo
echo "================================================================"
echo "[summary]"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo "[progress] $ARCHIVED_PROGRESS"
echo "[logdir]   $LOGDIR"
echo "================================================================"

if (( failures != 0 )); then
  echo "[validation-failed] failures=$failures" >&2
  exit 1
fi

echo "[validation-ok] 222 MAXD13 preparation diagnostics reproduced cases 01-06 with required=14, MAXD14, 208 bytes/thread"
