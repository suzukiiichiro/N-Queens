#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# NQ_UPDATE_MEMO
# 234 validation: 233 fasttrim-inline を親にした N=21/cache-hot/MAXD14 direct 版を検証する。
#      既存 shaped bin を直接読み、fallback kernels / schedule depth scan / MAXD select wrapper を削除。
#      CUDA MAXD14 kernel本文は233と同一。N=21 full once + progress TSV reconstruction で検証する。
# =============================================================================

SRC=${SRC:-./234Py_cachehot_maxd14_direct_split145_rootrestlate_fastest_probe.py}
CAND=${CAND:-./234Py_cachehot_maxd14_direct_split145_rootrestlate_fastest_probe}
AUTO_BUILD=${AUTO_BUILD:-1}
STATIC_ONLY=${STATIC_ONLY:-0}
LOCK_FILE=${LOCK_FILE:-/tmp/234Py_cachehot_maxd14_direct_split145_rootrestlate_fastest_validation.lock}
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

BASELINE_233_SECONDS=${BASELINE_233_SECONDS:-427.762}
BASELINE_231_SECONDS=${BASELINE_231_SECONDS:-427.818}
BASELINE_217_SECONDS=${BASELINE_217_SECONDS:-427.709}
BASELINE_204_SECONDS=${BASELINE_204_SECONDS:-427.795}
BASELINE_230_SECONDS=${BASELINE_230_SECONDS:-428.848}

if [[ "$N" != "21" || "$BENCH_MODE" != "31" || "$SORT_MODE" != "0" || "$PRESET_QUEENS" != "7" || "$REORDER_WINDOW_MULT" != "8" || "$REORDER_PHASE_JUMP" != "7" || "$CROSS_STRIPE_SAFE" != "0" || "$BROADMARK_VARIANT" != "2" ]]; then
  echo "[error] 234 fixed validation requires N=21, SORT_MODE=0, PRESET_QUEENS=7, BENCH_MODE=31, w8_j7, cross=0, variant=2" >&2
  exit 64
fi
if [[ "$WORKER_ID" != "0" || "$WORKER_COUNT" != "1" ]]; then
  echo "[error] fixed full validation totals require WORKER_ID=0, WORKER_COUNT=1" >&2
  exit 64
fi

if [[ "$STATIC_ONLY" != "1" ]] && command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[error] another 234 validation holds: $LOCK_FILE" >&2
    exit 75
  fi
fi

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="${LOG_ROOT%/}/234Py_cachehot_maxd14_direct_split145_rootrestlate_fastest_logs_N21_full_once_${TS}"
RUN_LOG="$LOGDIR/full_once.log"
BUILD_LOG="$LOGDIR/build.log"
SUMMARY="$LOGDIR/summary.tsv"
METRICS="$LOGDIR/metrics.env"
DISPATCH_METRICS="$LOGDIR/dispatch_metrics.env"
ARCHIVED_PROGRESS="$LOGDIR/progress_full.tsv"
mkdir -p "$LOGDIR"
printf 'check\texpected\tactual\tstatus\n' > "$SUMMARY"

record_check() {
  local name=$1 expected=$2 actual=$3 status=FAIL
  if [[ "$actual" == "$expected" ]]; then status=OK; fi
  printf '%s\t%s\t%s\t%s\n' "$name" "$expected" "$actual" "$status" >> "$SUMMARY"
  [[ "$status" == OK ]]
}

if [[ ! -f "$SRC" ]]; then
  echo "[error] source not found: $SRC" >&2
  exit 66
fi

static_failures=0
if grep -q '234 cachehot-maxd14-direct' "$SRC"; then
  printf 'source_version_tag\t234 cachehot-maxd14-direct\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_version_tag\t234 cachehot-maxd14-direct\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q 'if __name__=="__main__":' "$SRC" && grep -q '  main()' "$SRC"; then
  printf 'source_main_entry\t__main__ calls main()\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_main_entry\t__main__ calls main()\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q 'future_check_mask:u32=u32(0)' "$SRC" && grep -q 'if future_check_mask!=u32(0):' "$SRC"; then
  printf 'source_future_check_mask_guard\tpresent\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_future_check_mask_guard\tpresent\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q 'save_sp:int=0' "$SRC" && grep -q 'cur_depth:int=0' "$SRC" && grep -q 'avail\[save_sp\]=cur_avail|(u32(cur_depth)<<u32(27))' "$SRC"; then
  printf 'source_nosibling_parent\t188-style save_sp/cur_depth code\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_nosibling_parent\t188-style save_sp/cur_depth code\tcheck source\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if python3 - "$SRC" <<'PYCHECK'
import sys,re
s=open(sys.argv[1],encoding='utf-8').read()
compact=''.join(line for line in s.splitlines() if not line.lstrip().startswith('#'))
checks={
  'maxd14_kernel_present':'def kernel_dfs_iter_gpu_maxd14(' in s,
  'fallback16_absent':'def kernel_dfs_iter_gpu_maxd16(' not in s,
  'fallback18_absent':'def kernel_dfs_iter_gpu_maxd18(' not in s,
  'fallback20_absent':'def kernel_dfs_iter_gpu_maxd20(' not in s,
  'fallback21_absent':'def kernel_dfs_iter_gpu_maxd21(' not in s,
  'launch_wrapper_absent':'def launch_kernel_dfs_iter_gpu_static_maxd(' not in s,
  'schedule_depth_scan_absent':'def schedule_depth_for_task(' not in s and 'def max_schedule_depth_of_tasks(' not in s,
  'maxd_selector_absent':'def select_static_maxd(' not in s,
  'direct_launch_present':'kernel_dfs_iter_gpu_maxd14(' in s and 'grid=GRID,block=BLOCK' in compact,
  'static_dispatch_constant_log':'required_maxd=14 selected_MAXD=14 schedule_words=0 stack_bytes_per_thread=208' in s,
  'cachehot_marker':'cachehot_maxd14_direct=1' in s,
  'chunk_dict_removed':'chunk_constellations' not in s,
  'soa_trimmed':'row_arr' not in s and 'funcid_arr' not in s and 'ijkl_arr' not in s,
}
if all(checks.values()):
    sys.exit(0)
print('source_cachehot_maxd14_pycheck_failed=' + ','.join(k for k,v in checks.items() if not v))
sys.exit(1)
PYCHECK
then
  printf 'source_cachehot_maxd14_direct\tfallback-free direct launch\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_cachehot_maxd14_direct\tfallback-free direct launch\tmissing_or_old_shape\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi

actual_split=$(python3 - "$SRC" <<'PYSPLIT'
import re,sys
s=open(sys.argv[1],encoding='utf-8').read()
tags=sorted(set(re.findall(r'split\d+',s)))
bad=sorted(set(tags)-{'split145','split234'})
print(' '.join(tags) if tags else 'none')
sys.exit(0 if set(tags)=={'split145','split234'} and not bad else 1)
PYSPLIT
) && split_ok=1 || split_ok=0
if [[ "$split_ok" == 1 ]]; then
  printf 'source_split_tag\tsplit145 split234 active\t%s\tOK\n' "$actual_split" >> "$SUMMARY"
else
  printf 'source_split_tag\tsplit145 split234 active\t%s\tFAIL\n' "$actual_split" >> "$SUMMARY"; static_failures=$((static_failures+1))
fi

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "================================================================"; echo "[static-summary]"; cat "$SUMMARY"; echo "[logdir] $LOGDIR"
  if (( static_failures != 0 )); then exit 1; fi
  exit 0
fi

if (( static_failures != 0 )); then
  echo "================================================================"; echo "[static-summary]"; cat "$SUMMARY"; echo "[logdir] $LOGDIR"
  echo "[error] 234 source static checks failed" >&2
  exit 66
fi

need_build=0
if [[ ! -x "$CAND" ]]; then need_build=1; elif [[ "$SRC" -nt "$CAND" ]]; then need_build=1; fi
if (( need_build )); then
  if [[ "$AUTO_BUILD" != "1" ]]; then echo "[error] stale/missing candidate and AUTO_BUILD=$AUTO_BUILD: $CAND" >&2; exit 66; fi
  if ! command -v codon >/dev/null 2>&1; then echo "[error] codon was not found; cannot build $SRC" >&2; exit 69; fi
  echo "[build] codon build -release $SRC" | tee "$BUILD_LOG"
  set +e; codon build -release "$SRC" 2>&1 | tee -a "$BUILD_LOG"; build_rc=${PIPESTATUS[0]}; set -e
  printf 'build_exit\t0\t%d\t%s\n' "$build_rc" "$([[ $build_rc -eq 0 ]] && echo OK || echo FAIL)" >> "$SUMMARY"
  if (( build_rc != 0 )); then exit "$build_rc"; fi
else
  echo "[build] reuse executable: $CAND" | tee "$BUILD_LOG"
fi

CMD=("$CAND" -g "$N" "$N" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS" "$BENCH_MODE" "$REORDER_WINDOW_MULT" "$REORDER_PHASE_JUMP" "$CROSS_STRIPE_SAFE" "$WORKER_ID" "$WORKER_COUNT" "$BROADMARK_VARIANT")
{
  echo "================================================================"
  echo "candidate : $CAND"
  echo "source    : $SRC"
  echo "date      : $(date -Is)"
  echo "cwd       : $(pwd)"
  command -v sha256sum >/dev/null 2>&1 && echo "source_sha256: $(sha256sum "$SRC" | awk '{print $1}')"
  printf 'command   :'; printf ' %q' "${CMD[@]}"; echo
  echo "validation: one N=21 full run; cases 01-06 reconstructed from progress TSV"
  echo "dispatch  : fixed required=14, selected MAXD14, schedule_words=0, stack=208 bytes/thread"
  echo "cache     : requires existing N21 shaped scorestripe/chunkshape148 bin"
  echo "================================================================"
} | tee "$RUN_LOG"

set +e; stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "$RUN_LOG"; run_rc=${PIPESTATUS[0]}; set -e
printf 'run_exit\t0\t%d\t%s\n' "$run_rc" "$([[ $run_rc -eq 0 ]] && echo OK || echo FAIL)" >> "$SUMMARY"
if (( run_rc != 0 )); then exit "$run_rc"; fi

awk '
  /^\[maxd-dispatch\] N=21 scope=split145 / {
    rows++; m=0; req=-1; sel=-1; words=-1; bytes=-1; cap=""
    for (i=1;i<=NF;i++) { split($i,a,"="); if(a[1]=="m")m=a[2]+0; else if(a[1]=="required_maxd")req=a[2]+0; else if(a[1]=="selected_MAXD")sel=a[2]+0; else if(a[1]=="schedule_words")words=a[2]+0; else if(a[1]=="stack_bytes_per_thread")bytes=a[2]+0; else if(a[1]=="capacity_check")cap=a[2] }
    tasks+=m; if(rows==1||req<minreq)minreq=req; if(req>maxreq)maxreq=req; if(req!=14)badreq++; if(sel!=14)badsel++; if(words!=0)badwords++; if(bytes!=208)badbytes++; if(cap!="OK")badcap++; if(req>sel)under++
  }
  END { printf "DISPATCH_ROWS=%d\n",rows+0; printf "DISPATCH_TASKS=%.0f\n",tasks+0; printf "DISPATCH_MIN_REQUIRED=%d\n",minreq+0; printf "DISPATCH_MAX_REQUIRED=%d\n",maxreq+0; printf "DISPATCH_NON14=%d\n",badreq+0; printf "DISPATCH_NON_SELECTED14=%d\n",badsel+0; printf "DISPATCH_NON0WORDS=%d\n",badwords+0; printf "DISPATCH_NON208=%d\n",badbytes+0; printf "DISPATCH_BAD_CAPACITY=%d\n",badcap+0; printf "DISPATCH_UNDERSIZED=%d\n",under+0 }
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

DYNAMIC_PRESET=$(sed -n 's/^\[dynamic-preset\] N=21 preset_queens=\([0-9][0-9]*\)$/\1/p' "$RUN_LOG" | tail -n1)
record_check "dynamic_preset_N21" "6" "${DYNAMIC_PRESET:-missing}" || failures=$((failures+1))
CACHEHOT_REUSE=$(grep -c '^\[split234-full-reuse\]' "$RUN_LOG" || true)
record_check "cachehot_shaped_reuse" "1" "$CACHEHOT_REUSE" || failures=$((failures+1))

PROGRESS=$(sed -n 's/.* progress=\([^[:space:]]*\.tsv\).*/\1/p' "$RUN_LOG" | tail -n1 | tr -d '\r')
if [[ -z "$PROGRESS" || ! -s "$PROGRESS" ]]; then printf 'progress_file\tpresent\t%s\tFAIL\n' "${PROGRESS:-missing}" >> "$SUMMARY"; exit 65; fi
cp -f "$PROGRESS" "$ARCHIVED_PROGRESS"
printf 'progress_file\tpresent\t%s\tOK\n' "$PROGRESS" >> "$SUMMARY"

awk -F '\t' -v expected_chunks="$EXPECTED_CHUNKS" '
  NR==1 { for(i=1;i<=NF;i++){ if($i=="chunk")chunk_col=i; else if($i=="chunk_total")total_col=i; else if($i=="gpu_total")gpu_col=i } if(!chunk_col||!total_col||!gpu_col){print "PARSE_OK=0"; exit 2} next }
  { chunk=$(chunk_col)+0; value=$(total_col)+0; gpu=$(gpu_col)+0; rows++; full+=value; last_gpu=gpu; if(seen[chunk]++)duplicates++; if(chunk==0||chunk==20||chunk==40||chunk==60||chunk==80||chunk==100||chunk==120)p1+=value; if(chunk==35||chunk==40||chunk==41||chunk==42||chunk==47||chunk==48||chunk==52||chunk==53)p2+=value; if(chunk==20||chunk==40||chunk==55||chunk==56||chunk==57||chunk==58||chunk==60)p3+=value; if(chunk==100||chunk==105||chunk==110||chunk==115||chunk==120||chunk==125||chunk==130)p4+=value; if((chunk%4)==0)p5+=value }
  END { if(!chunk_col||!total_col||!gpu_col)exit; for(i=0;i<expected_chunks;i++)if(!(i in seen))missing++; printf "PARSE_OK=1\nROWS=%.0f\nDUPLICATES=%.0f\nMISSING=%.0f\n",rows,duplicates,missing; printf "P1=%.0f\nP2=%.0f\nP3=%.0f\nP4=%.0f\nP5=%.0f\n",p1,p2,p3,p4,p5; printf "FULL=%.0f\nLAST_GPU=%.0f\n",full,last_gpu }
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
if [[ "$FINAL_LINE" == *"$FULL_TOTAL"* && "$FINAL_LINE" == *"ok"* ]]; then printf 'final_output\t%s ... ok\t%s\tOK\n' "$FULL_TOTAL" "$FINAL_LINE" >> "$SUMMARY"; else printf 'final_output\t%s ... ok\t%s\tFAIL\n' "$FULL_TOTAL" "${FINAL_LINE:-missing}" >> "$SUMMARY"; failures=$((failures+1)); fi
ERROR_HITS=$(grep -Eic '\[(.*-)?error\]|mismatch|ng\(' "$RUN_LOG" || true)
record_check "error_or_mismatch_hits" "0" "$ERROR_HITS" || failures=$((failures+1))
ELAPSED_TEXT=$(awk -v n="$N" '$0 ~ "^[[:space:]]*" n ":" {v=$(NF-1)} END {print v}' "$RUN_LOG")
if [[ -n "$ELAPSED_TEXT" ]]; then
  ELAPSED_SECONDS=$(awk -F: '{printf "%.3f",($1*3600)+($2*60)+$3}' <<< "$ELAPSED_TEXT")
  for pair in "233:$BASELINE_233_SECONDS" "231:$BASELINE_231_SECONDS" "217:$BASELINE_217_SECONDS" "204:$BASELINE_204_SECONDS" "230:$BASELINE_230_SECONDS"; do
    label=${pair%%:*}; base=${pair#*:}; perf=$(awk -v base="$base" -v now="$ELAPSED_SECONDS" 'BEGIN{d=base-now;p=(base>0)?d/base*100:0;printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%",now,base,d,p}')
    printf 'timing_vs_%s\t%s_full=%ss\t%s\tINFO\n' "$label" "$label" "$base" "$perf" >> "$SUMMARY"
  done
fi
WARN_HITS=$(grep -Eic '\[.*warning\]' "$RUN_LOG" || true)
printf 'warning_hits\t0 preferred\t%s\tINFO\n' "$WARN_HITS" >> "$SUMMARY"

echo; echo "================================================================"; echo "[summary]"; cat "$SUMMARY"; echo "[progress] $ARCHIVED_PROGRESS"; echo "[logdir]   $LOGDIR"; echo "================================================================"
if (( failures != 0 )); then echo "[validation-failed] failures=$failures" >&2; exit 1; fi
echo "[validation-ok] 234 cachehot MAXD14 direct reproduced cases 01-06 with fixed required=14, MAXD14, 208 bytes/thread"
