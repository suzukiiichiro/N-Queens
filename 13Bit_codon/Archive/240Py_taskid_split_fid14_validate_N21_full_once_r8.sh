#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# 240 r8 N=21 full validation harness
# Parent: 239 n27coretrim-keepfeatures.
# Experiment: split145 fid=14 launch-split probe. CUDA kernel bodies are not
#             changed yet; this measures launch-split overhead and correctness.
# Expected: mode31 split145+chunkshape148, two launches/chunk when fid14 exists:
#           rest + fid14, required=14, selected MAXD14, schedule_words=0,
#           stack=208 bytes/thread.
# =============================================================================

SRC=${SRC:-./240Py_taskid_split_fid14_probe_r8.py}
CAND=${CAND:-./240Py_taskid_split_fid14_probe_r8}
AUTO_BUILD=${AUTO_BUILD:-1}
STATIC_ONLY=${STATIC_ONLY:-0}
LOG_ROOT=${LOG_ROOT:-.}
LOCK_FILE=${LOCK_FILE:-/tmp/240Py_taskid_split_fid14_N21_full.lock}

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
EXPECTED_DISPATCH_ROWS=262
EXPECTED_TASKS=2025282
EXPECTED_FID14_TASKS=8214
FULL_TOTAL=314666222712
BASELINE_239_SECONDS=${BASELINE_239_SECONDS:-427.703}
BASELINE_238_SECONDS=${BASELINE_238_SECONDS:-427.710}
BASELINE_237_SECONDS=${BASELINE_237_SECONDS:-427.834}
BASELINE_217_SECONDS=${BASELINE_217_SECONDS:-427.709}

if [[ "$N" != "21" || "$BENCH_MODE" != "31" || "$WORKER_ID" != "0" || "$WORKER_COUNT" != "1" ]]; then
  echo "[error] fixed validation totals require N=21, BENCH_MODE=31, WORKER_ID=0, WORKER_COUNT=1" >&2
  exit 64
fi

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="${LOG_ROOT%/}/240Py_taskid_split_fid14_logs_N21_full_once_${TS}"
RUN_LOG="$LOGDIR/full_once.log"
BUILD_LOG="$LOGDIR/build.log"
SUMMARY="$LOGDIR/summary.tsv"
PROGRESS_COPY="$LOGDIR/progress_full.tsv"
mkdir -p "$LOGDIR"
printf 'check\texpected\tactual\tstatus\n' > "$SUMMARY"

record_check() {
  local name=$1 expected=$2 actual=$3 status=FAIL
  if [[ "$actual" == "$expected" ]]; then status=OK; fi
  printf '%s\t%s\t%s\t%s\n' "$name" "$expected" "$actual" "$status" >> "$SUMMARY"
  [[ "$status" == OK ]]
}
failures=0
static_failures=0

if [[ ! -f "$SRC" ]]; then
  echo "[error] source not found: $SRC" >&2
  exit 66
fi

if grep -q '240 taskid-split-fid14' "$SRC"; then
  printf 'source_version_tag\t240 taskid-split-fid14\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_version_tag\t240 taskid-split-fid14\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q 'if __name__=="__main__"' "$SRC" && grep -q '  main()' "$SRC"; then
  printf 'source_main_entry\tpresent\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_main_entry\tpresent\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q 'A10G_FINAL_DEFAULT_BENCH_MODE:int=31' "$SRC"; then
  printf 'bare_g_fastdefault_mode31\t31\t31\tOK\n' >> "$SUMMARY"
else
  printf 'bare_g_fastdefault_mode31\t31\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q 'scope=split145-fid14' "$SRC" && grep -q 'scope=split145-rest' "$SRC" && grep -q 'split=fid14_launch' "$SRC" && grep -q 'split_mode==14' "$SRC"; then
  printf 'source_fid14_launch_split\tpresent\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_fid14_launch_split\tpresent\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q 'elif N>=25 and N<=27:' "$SRC" && grep -q '234907967154122528' "$SRC"; then
  printf 'gpu_range_n27_dynamic_preset\tN25..N27 preset8 and N27 total\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'gpu_range_n27_dynamic_preset\tN25..N27 preset8 and N27 total\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q '^FUNCID_REORDER_V2_WINDOW_MULT:int=8' "$SRC" \
   && grep -q '^BROAD_MARKDIST_TAIL_REORDER_VERSION:str="v4"' "$SRC" \
   && grep -q '^BROAD_MARKDIST_TAIL_VARIANT:int=2' "$SRC"; then
  printf 'source_runtime_globals\tfuncid/broadmarktail constants\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_runtime_globals\tfuncid/broadmarktail constants\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi

python3 - "$SRC" "$SUMMARY" <<'PYCHECK'
import re, sys
src, summary = sys.argv[1], sys.argv[2]
s = open(src, encoding='utf-8').read()
def has_def(name): return re.search(r'^def\s+' + re.escape(name) + r'\b', s, re.M) is not None
def has_kernel(name): return re.search(r'^@gpu\.kernel\s*\n^def\s+' + re.escape(name) + r'\b', s, re.M) is not None
checks=[]
required_defs = [
    'kernel_dfs_iter_gpu_maxd14','kernel_dfs_iter_gpu_maxd16','kernel_dfs_iter_gpu_maxd18','kernel_dfs_iter_gpu_maxd20','kernel_dfs_iter_gpu_maxd21',
    'launch_kernel_dfs_iter_gpu_static_maxd','ensure_constellations_bin_stream','build_broad_markdist_tail_reordered_bin',
    'build_chunkshape148_reordered_bin','exec_solutions_gpu_bin_stream_funcid_reorder','exec_solutions_gpu_bin_stream_split145',
    'exec_solutions_gpu_chunk_split145','dfs_iter'
]
missing=[x for x in required_defs if not (has_def(x) or has_kernel(x))]
checks.append(('required_runtime_defs','all present','missing='+','.join(missing) if missing else 'all present', not missing))
checks.append(('removed_cpu_recursive_dfs','absent','present' if has_def('dfs') else 'absent', not has_def('dfs')))
checks.append(('removed_use_itter_branch','absent','present' if 'use_itter' in s else 'absent', 'use_itter' not in s))
removed_modes = ['bench_mode==17','bench_mode==18','bench_mode==19','bench_mode==20','bench_mode==21','bench_mode==22','bench_mode==23','bench_mode==24','bench_mode==25','bench_mode==26','bench_mode==27']
left_modes=[x for x in removed_modes if x in s]
checks.append(('removed_diag_modes','17..27 absent','present='+','.join(left_modes) if left_modes else '17..27 absent', not left_modes))
keep_modes=['bench_mode==28','bench_mode==29','bench_mode==30','bench_mode==31']
missing_modes=[x for x in keep_modes if x not in s]
checks.append(('kept_core_modes','28/29/30/31 present','missing='+','.join(missing_modes) if missing_modes else '28/29/30/31 present', not missing_modes))
checks.append(('source_split_tag','split240 active','split240' if ('split240' in s and 'split239' not in s and 'split238' not in s) else 'old split tag present or split240 missing', ('split240' in s and 'split239' not in s and 'split238' not in s)))
try:
    st=s.index('def exec_solutions_gpu_chunk_split145')
    en=s.index('\ndef exec_solutions_gpu_bin_stream_split145', st)
    split145_chunk=s[st:en]
except ValueError:
    split145_chunk=''
stale_split_names=[x for x in ['required_maxd','selected_MAXD','selected_maxd'] if x in split145_chunk]
checks.append(('source_split145_no_stale_maxd_names','absent','present='+','.join(stale_split_names) if stale_split_names else 'absent', not stale_split_names))
checks.append(('worker_split_args','present','present' if ('worker_id:int=0' in s and 'worker_count:int=1' in s and 'worker_id}/{worker_count}' in s) else 'missing', ('worker_id:int=0' in s and 'worker_count:int=1' in s and 'worker_id}/{worker_count}' in s)))

# r8 buildfix guard: the split145 body must not contain a bare required_maxd token
# or the old f-string literal that triggered Codon's misleading name error.
m = re.search(r'^def\s+exec_solutions_gpu_chunk_split145\b', s, re.M)
n = re.search(r'^def\s+exec_solutions_gpu_bin_stream_split145\b', s, re.M)
body = s[m.start():n.start()] if m and n else ''
bad_split = ('required_maxd' in body) or (' selected_MAXD=' in body and 'reqmaxd=' not in body)
checks.append(('source_split145_reqmaxd_buildfix','no required_maxd in split145 body','OK' if not bad_split else 'required_maxd_or_old_label_present', not bad_split))
fail=0
with open(summary,'a',encoding='utf-8') as f:
    for name, exp, actual, ok in checks:
        f.write(f"{name}\t{exp}\t{actual}\t{'OK' if ok else 'FAIL'}\n")
        if not ok: fail+=1
sys.exit(1 if fail else 0)
PYCHECK
py_rc=$?
if (( py_rc != 0 )); then static_failures=$((static_failures+1)); fi

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "================================================================"
  echo "[static-summary]"
  cat "$SUMMARY"
  echo "[logdir] $LOGDIR"
  if (( static_failures != 0 )); then exit 1; fi
  exit 0
fi

if (( static_failures != 0 )); then
  echo "================================================================"
  echo "[static-summary]"
  cat "$SUMMARY"
  echo "[logdir] $LOGDIR"
  echo "[error] 240 source static checks failed" >&2
  exit 66
fi

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[error] another 240 validation holds: $LOCK_FILE" >&2
    exit 75
  fi
fi

need_build=0
if [[ ! -x "$CAND" ]]; then need_build=1; elif [[ "$SRC" -nt "$CAND" ]]; then need_build=1; fi
if (( need_build )); then
  if [[ "$AUTO_BUILD" != "1" ]]; then echo "[error] stale/missing candidate and AUTO_BUILD=$AUTO_BUILD: $CAND" >&2; exit 66; fi
  if ! command -v codon >/dev/null 2>&1; then echo "[error] codon was not found; cannot build $SRC" >&2; exit 69; fi
  echo "[build] codon build -release $SRC" | tee "$BUILD_LOG"
  set +e; codon build -release "$SRC" 2>&1 | tee -a "$BUILD_LOG"; build_rc=${PIPESTATUS[0]}; set -e
  record_check build_exit 0 "$build_rc" || failures=$((failures+1))
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
  echo "validation: one N=21 full run through 240 fid14/rest launch-split mode31 split145"
  echo "dispatch  : expected rows=$EXPECTED_DISPATCH_ROWS, tasks=$EXPECTED_TASKS, fid14_tasks=$EXPECTED_FID14_TASKS"
  echo "================================================================"
} | tee "$RUN_LOG"

set +e; stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "$RUN_LOG"; run_rc=${PIPESTATUS[0]}; set -e
record_check run_exit 0 "$run_rc" || failures=$((failures+1))
if (( run_rc != 0 )); then exit "$run_rc"; fi

DYNAMIC_PRESET=$(sed -n 's/^\[dynamic-preset\] N=21 preset_queens=\([0-9][0-9]*\)$/\1/p' "$RUN_LOG" | tail -n1)
record_check dynamic_preset_N21 6 "${DYNAMIC_PRESET:-missing}" || failures=$((failures+1))

awk '
  /^\[maxd-dispatch\] N=21 scope=split145/ {
    rows++; m=0; req=-1; sel=-1; words=-1; bytes=-1; cap=""; scope=""
    for (i=1;i<=NF;i++) { split($i,a,"="); if(a[1]=="scope")scope=a[2]; else if(a[1]=="m")m=a[2]+0; else if(a[1]=="required_maxd" || a[1]=="reqmaxd")req=a[2]+0; else if(a[1]=="selected_MAXD" || a[1]=="selMAXD")sel=a[2]+0; else if(a[1]=="schedule_words")words=a[2]+0; else if(a[1]=="stack_bytes_per_thread")bytes=a[2]+0; else if(a[1]=="capacity_check")cap=a[2] }
    tasks+=m; if(scope=="split145-fid14"){fid14_rows++; fid14_tasks+=m} if(scope=="split145-rest"){rest_rows++; rest_tasks+=m}
    if(req!=14)badreq++; if(sel!=14)badsel++; if(words!=0)badwords++; if(bytes!=208)badbytes++; if(cap!="OK")badcap++
  }
  END { printf "rows=%d\ntasks=%.0f\nfid14_rows=%d\nrest_rows=%d\nfid14_tasks=%.0f\nrest_tasks=%.0f\nbadreq=%d\nbadsel=%d\nbadwords=%d\nbadbytes=%d\nbadcap=%d\n",rows+0,tasks+0,fid14_rows+0,rest_rows+0,fid14_tasks+0,rest_tasks+0,badreq+0,badsel+0,badwords+0,badbytes+0,badcap+0 }
' "$RUN_LOG" > "$LOGDIR/dispatch.env"
source "$LOGDIR/dispatch.env"
record_check dispatch_launch_rows "$EXPECTED_DISPATCH_ROWS" "$rows" || failures=$((failures+1))
record_check dispatch_task_sum "$EXPECTED_TASKS" "$tasks" || failures=$((failures+1))
record_check dispatch_fid14_launch_rows "$EXPECTED_CHUNKS" "$fid14_rows" || failures=$((failures+1))
record_check dispatch_rest_launch_rows "$EXPECTED_CHUNKS" "$rest_rows" || failures=$((failures+1))
record_check dispatch_fid14_task_sum "$EXPECTED_FID14_TASKS" "$fid14_tasks" || failures=$((failures+1))
record_check dispatch_non_required14 0 "$badreq" || failures=$((failures+1))
record_check dispatch_non_MAXD14 0 "$badsel" || failures=$((failures+1))
record_check dispatch_non_0_schedule_words 0 "$badwords" || failures=$((failures+1))
record_check dispatch_non_208_bytes 0 "$badbytes" || failures=$((failures+1))
record_check dispatch_bad_capacity_flag 0 "$badcap" || failures=$((failures+1))

PROGRESS=$(sed -n 's/.* progress=\([^[:space:]]*\.tsv\).*/\1/p' "$RUN_LOG" | tail -n1 | tr -d '\r')
if [[ -n "$PROGRESS" && -s "$PROGRESS" ]]; then
  cp -f "$PROGRESS" "$PROGRESS_COPY"
  record_check progress_file present present || failures=$((failures+1))
  awk -F '\t' -v expected_chunks="$EXPECTED_CHUNKS" '
    NR==1 { for(i=1;i<=NF;i++){ if($i=="chunk")chunk_col=i; else if($i=="chunk_total")total_col=i; else if($i=="gpu_total")gpu_col=i } next }
    { chunk=$(chunk_col)+0; value=$(total_col)+0; gpu=$(gpu_col)+0; rows++; full+=value; last_gpu=gpu; if(seen[chunk]++)dup++ }
    END { for(i=0;i<expected_chunks;i++)if(!(i in seen))missing++; printf "ROWS=%.0f\nDUP=%.0f\nMISS=%.0f\nFULL=%.0f\nLAST_GPU=%.0f\n",rows,dup,missing,full,last_gpu }
  ' "$PROGRESS_COPY" > "$LOGDIR/progress_metrics.env"
  source "$LOGDIR/progress_metrics.env"
  record_check progress_rows "$EXPECTED_CHUNKS" "$ROWS" || failures=$((failures+1))
  record_check duplicate_chunks 0 "$DUP" || failures=$((failures+1))
  record_check missing_chunks 0 "$MISS" || failures=$((failures+1))
  record_check full_chunk_sum "$FULL_TOTAL" "$FULL" || failures=$((failures+1))
  record_check full_last_gpu "$FULL_TOTAL" "$LAST_GPU" || failures=$((failures+1))
else
  record_check progress_file present missing || failures=$((failures+1))
fi

FINAL_LINE=$(grep -E "^[[:space:]]*${N}:" "$RUN_LOG" | tail -n1 || true)
if [[ "$FINAL_LINE" == *"$FULL_TOTAL"* && "$FINAL_LINE" == *"ok"* ]]; then
  printf 'final_output\t%s ... ok\t%s\tOK\n' "$FULL_TOTAL" "$FINAL_LINE" >> "$SUMMARY"
else
  printf 'final_output\t%s ... ok\t%s\tFAIL\n' "$FULL_TOTAL" "${FINAL_LINE:-missing}" >> "$SUMMARY"; failures=$((failures+1))
fi
ERROR_HITS=$(grep -Eic '\[(.*-)?error\]|mismatch|ng\(' "$RUN_LOG" || true)
record_check error_or_mismatch_hits 0 "$ERROR_HITS" || failures=$((failures+1))

ELAPSED_TEXT=$(awk -v n="$N" '$0 ~ "^[[:space:]]*" n ":" {v=$(NF-1)} END {print v}' "$RUN_LOG")
if [[ -n "$ELAPSED_TEXT" ]]; then
  ELAPSED_SECONDS=$(awk -F: '{printf "%.3f",($1*3600)+($2*60)+$3}' <<< "$ELAPSED_TEXT")
  for pair in "239:$BASELINE_239_SECONDS" "238:$BASELINE_238_SECONDS" "237:$BASELINE_237_SECONDS" "217:$BASELINE_217_SECONDS"; do
    label=${pair%%:*}; base=${pair#*:}
    perf=$(awk -v base="$base" -v now="$ELAPSED_SECONDS" 'BEGIN{d=base-now;p=(base>0)?d/base*100:0;printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%",now,base,d,p}')
    printf 'timing_vs_%s\tbaseline=%ss\t%s\tINFO\n' "$label" "$base" "$perf" >> "$SUMMARY"
  done
fi

echo
echo "================================================================"
echo "[summary]"
cat "$SUMMARY"
echo "[progress] ${PROGRESS_COPY:-missing}"
echo "[logdir]   $LOGDIR"
echo "================================================================"
if (( failures != 0 )); then
  echo "[validation-failed] failures=$failures" >&2
  exit 1
fi
echo "[validation-ok] 240 N=21 mode31 split145 fid14/rest launch-split reproduced total with required=14, MAXD14, 208 bytes/thread"
