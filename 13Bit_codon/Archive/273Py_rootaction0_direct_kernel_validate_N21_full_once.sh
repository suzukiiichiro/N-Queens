#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# 273 N=21 full validation harness
# Parent: 271 final-fastest-complete / 272 root_action distribution result.
# Experiment: add a direct MAXD14 root0 kernel without schedule precompute.
#             If root_action_nonzero_count==0 for a chunk, dispatch the
#             root_action-free kernel; otherwise fallback to 271 MAXD14.
# Expected: mode31 split145+chunkshape148, one launch per chunk, required=14,
#           selected MAXD14, schedule_words=0, stack=208 bytes/thread,
#           131 root0 dispatch rows and zero fallback rows for N=21.
# =============================================================================

SRC=${SRC:-./273Py_rootaction0_direct_kernel_probe.py}
CAND=${CAND:-./273Py_rootaction0_direct_kernel_probe}
AUTO_BUILD=${AUTO_BUILD:-1}
# 241 r3: visible startup + force release rebuild by default so a previous non-release
# `codon build` executable is not reused and cannot trigger CUDA_ERROR_INVALID_PTX.
# 241 r3: print start/status early and prevent static pycheck from failing silently under set -e.
FORCE_REBUILD=${FORCE_REBUILD:-1}
STATIC_ONLY=${STATIC_ONLY:-0}
LOG_ROOT=${LOG_ROOT:-.}
LOCK_FILE=${LOCK_FILE:-/tmp/273Py_rootaction0_direct_kernel_N21_full.lock}

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
FULL_TOTAL=314666222712
EXPECTED_REQUIRED_MAXD=14
EXPECTED_SELECTED_MAXD=14
EXPECTED_SCHEDULE_WORDS=0
EXPECTED_STACK_BYTES=208
BASELINE_273_PARENT271_SECONDS=${BASELINE_273_PARENT271_SECONDS:-427.705}
BASELINE_272_DIAG_SECONDS=${BASELINE_272_DIAG_SECONDS:-427.728}
BASELINE_271_PARENT_SECONDS=${BASELINE_271_PARENT_SECONDS:-427.788}
BASELINE_270_REJECT_SECONDS=${BASELINE_270_REJECT_SECONDS:-428.824}
BASELINE_267_SECONDS=${BASELINE_267_SECONDS:-428.056}
BASELINE_239_SECONDS=${BASELINE_239_SECONDS:-427.703}
BASELINE_240_REJECT_SECONDS=${BASELINE_240_REJECT_SECONDS:-568.451}
BASELINE_238_SECONDS=${BASELINE_238_SECONDS:-427.710}
BASELINE_237_SECONDS=${BASELINE_237_SECONDS:-427.834}
BASELINE_232_SECONDS=${BASELINE_232_SECONDS:-427.733}
BASELINE_217_SECONDS=${BASELINE_217_SECONDS:-427.709}

if [[ "$N" != "21" || "$BENCH_MODE" != "31" || "$WORKER_ID" != "0" || "$WORKER_COUNT" != "1" ]]; then
  echo "[error] fixed validation totals require N=21, BENCH_MODE=31, WORKER_ID=0, WORKER_COUNT=1" >&2
  exit 64
fi

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="${LOG_ROOT%/}/273Py_rootaction0_direct_kernel_logs_N21_full_once_${TS}"
RUN_LOG="$LOGDIR/full_once.log"
BUILD_LOG="$LOGDIR/build.log"
SUMMARY="$LOGDIR/summary.tsv"
PROGRESS_COPY="$LOGDIR/progress_full.tsv"
mkdir -p "$LOGDIR"
printf 'check\texpected\tactual\tstatus\n' > "$SUMMARY"

echo "[start] 273 rootaction0-direct-kernel validation script"
echo "[source] $SRC"
echo "[candidate] $CAND"
echo "[logdir] $LOGDIR"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo "[abort] rc=$rc logdir=${LOGDIR:-unknown}" >&2; fi' EXIT
echo "[validation-start] 273 rootaction0-direct-kernel SRC=$SRC CAND=$CAND STATIC_ONLY=$STATIC_ONLY FORCE_REBUILD=$FORCE_REBUILD LOGDIR=$LOGDIR"

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

# ---- source static checks ----
if grep -q '273 rootaction0-direct-kernel' "$SRC"; then
  printf 'source_version_tag\t273 rootaction0-direct-kernel\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_version_tag\t273 rootaction0-direct-kernel\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
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
if grep -q 'elif N>=25 and N<=27:' "$SRC" && grep -q '234907967154122528' "$SRC"; then
  printf 'gpu_range_n27_dynamic_preset	N25..N27 preset8 and N27 total	present	OK
' >> "$SUMMARY"
else
  printf 'gpu_range_n27_dynamic_preset	N25..N27 preset8 and N27 total	missing	FAIL
' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q '^FUNCID_REORDER_V2_WINDOW_MULT:int=8' "$SRC"    && grep -q '^FUNCID_REORDER_V2_PHASE_JUMP:int=7' "$SRC"    && grep -q '^FUNCID_REORDER_V2_DEFAULT_REASON:str=' "$SRC"    && grep -q '^BROAD_MARKDIST_TAIL_REORDER_VERSION:str="v4"' "$SRC"    && grep -q '^BROAD_MARKDIST_TAIL_VARIANT:int=2' "$SRC"    && grep -q '^BROAD_MARKDIST_TAIL_CELL_SALT:int=17' "$SRC"    && grep -q '^BROAD_MARKDIST_TAIL_RISK_SALT:int=11' "$SRC"    && grep -q '^BROAD_MARKDIST_TAIL_PHASE_SALT:int=53' "$SRC"; then
  printf 'source_runtime_globals	funcid/broadmarktail constants	present	OK
' >> "$SUMMARY"
else
  printf 'source_runtime_globals	funcid/broadmarktail constants	missing	FAIL
' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi

set +e
python3 - "$SRC" "$SUMMARY" <<'PYCHECK'
import re, sys
src, summary = sys.argv[1], sys.argv[2]
s = open(src, encoding='utf-8').read()
checks = []
def has_def(name):
    return re.search(r'^def\s+' + re.escape(name) + r'\b', s, re.M) is not None
def has_kernel(name):
    return re.search(r'^@gpu\.kernel\s*\n^def\s+' + re.escape(name) + r'\b', s, re.M) is not None
required_defs = [
    'kernel_dfs_iter_gpu_maxd14','kernel_dfs_iter_gpu_maxd14_root0','kernel_dfs_iter_gpu_maxd16','kernel_dfs_iter_gpu_maxd18','kernel_dfs_iter_gpu_maxd20','kernel_dfs_iter_gpu_maxd21',
    'launch_kernel_dfs_iter_gpu_static_maxd','ensure_constellations_bin_stream','build_broad_markdist_tail_reordered_bin',
    'build_chunkshape148_reordered_bin','exec_solutions_gpu_bin_stream_funcid_reorder','exec_solutions_gpu_bin_stream_split145',
    'exec_solutions_gpu_chunk_split145','stream_funcid_reorder_risk_suffix','funcid_reorder_make_quotas',
    'interleave_funcid_reorder_parts','exec_solutions','dfs_iter','root_action_for_task'
]
missing = [x for x in required_defs if not (has_def(x) or has_kernel(x))]
checks.append(('required_runtime_defs', 'all present', 'missing=' + ','.join(missing) if missing else 'all present', not missing))
removed_defs = [
    'diagnose_boundary_classification','diagnose_solution_by_boundary','bc_id','bc_name','fid_name',
    'exec_solutions_gpu_bin_stream_funcid_reorder_profile','exec_solutions_gpu_bin_stream_funcid_reorder_chunksize_profile',
    'exec_solutions_gpu_bin_stream_funcid_reorder_funcid_target_profile','exec_solutions_gpu_bin_stream_funcid_reorder_funcid_single_profile',
    'exec_solutions_gpu_bin_stream_funcid_reorder_funcid_split_profile','exec_solutions_gpu_bin_stream_funcid_reorder_funcid_depth_profile',
    'exec_solutions_gpu_bin_stream_funcid_reorder_funcid_mark_profile','exec_solutions_gpu_bin_stream_funcid_reorder_funcid_markdist_profile',
    'build_funcid_reordered_bin','build_funcid_markdist_risk_reordered_bin','exec_solutions_gpu_bin_stream_stats_only'
]
left = [x for x in removed_defs if has_def(x)]
checks.append(('removed_diag_defs', 'absent', 'present=' + ','.join(left) if left else 'absent', not left))
checks.append(('removed_cpu_recursive_dfs', 'absent', 'present' if has_def('dfs') else 'absent', not has_def('dfs')))
# Check active code only; comments/docstrings may mention old use_itter history.
code_no_comments=[]
in_triple=None
for line in s.splitlines():
    stripped=line.strip()
    if in_triple:
        if in_triple in stripped:
            in_triple=None
        continue
    if stripped.startswith('\"\"\"') or stripped.startswith("'''"):
        if stripped.startswith('\"\"\"'):
            if stripped.count('\"\"\"') < 2:
                in_triple='\"\"\"'
        else:
            if stripped.count("'''") < 2:
                in_triple="'''"
        continue
    line=line.split('#',1)[0]
    code_no_comments.append(line)
active='\n'.join(code_no_comments)
checks.append(('removed_use_itter_branch', 'active absent', 'active present' if 'use_itter' in active else 'active absent', 'use_itter' not in active))
removed_modes = ['bench_mode==17','bench_mode==18','bench_mode==19','bench_mode==20','bench_mode==21','bench_mode==22','bench_mode==23','bench_mode==24','bench_mode==25','bench_mode==26','bench_mode==27']
left_modes = [x for x in removed_modes if x in s]
checks.append(('removed_diag_modes', '17..27 absent', 'present=' + ','.join(left_modes) if left_modes else '17..27 absent', not left_modes))
keep_modes = ['bench_mode==28','bench_mode==29','bench_mode==30','bench_mode==31']
missing_modes = [x for x in keep_modes if x not in s]
checks.append(('kept_core_modes', '28/29/30/31 present', 'missing=' + ','.join(missing_modes) if missing_modes else '28/29/30/31 present', not missing_modes))
if 'split273' in s and 'split272_' not in s and 'split271_' not in s and 'split270_' not in s and 'split240' not in s and 'split239_' not in s:
    checks.append(('source_split_tag', 'split273 active; 271 parent kept; 270/240 rejected', 'split273', True))
else:
    checks.append(('source_split_tag', 'split273 active; 271 parent kept; 270/240 rejected', 'old split runtime present or split273 missing', False))
if 'worker_id:int=0' in s and 'worker_count:int=1' in s and 'worker_id}/{worker_count}' in s:
    checks.append(('worker_split_args', 'present', 'present', True))
else:
    checks.append(('worker_split_args', 'present', 'missing', False))

if 'split=fid14_launch' in s or 'split145-fid14' in s or 'split145-rest' in s or 'source_fid14_launch_split' in s:
    checks.append(('source_fid14_split_rejected', 'absent', 'fid14 split marker present', False))
else:
    checks.append(('source_fid14_split_rejected', 'absent', 'absent', True))

root0 = re.search(r'^@gpu\.kernel\s*\n^def\s+kernel_dfs_iter_gpu_maxd14_root0\b.*?(?=^@gpu\.kernel\s*\n^def\s+kernel_dfs_iter_gpu_maxd16\b)', s, re.M | re.S)
if not root0:
    checks.append(('source_root0_direct_kernel', 'present', 'missing', False))
else:
    rbody = root0.group(0)
    # active-code only for root0 body
    active_lines=[]
    for line in rbody.splitlines():
        active_lines.append(line.split('#',1)[0])
    ractive='\n'.join(active_lines)
    bad = ('root_action:u32' in ractive or 'root_action=' in ractive or 'root_action==' in ractive or 'root_action!=u32' in ractive)
    ok = (not bad and 'root_after_second:u32=root_rest^root_second' in ractive and 'if root_after_second==u32(0):' in ractive and 'cur_avail=root_rest' in ractive)
    checks.append(('source_root0_direct_kernel', 'root0 kernel without root_action active code', 'present' if ok else 'bad root0 body', ok))
launch_start = s.find('def launch_kernel_dfs_iter_gpu_static_maxd')
launch_end = s.find('if selected_maxd==16:', launch_start)
launch_body = s[launch_start:launch_end] if launch_start >= 0 and launch_end > launch_start else ''
root0_dispatch_ok = ('if soa.root_action_nonzero_count==0:' in launch_body and 'kernel_dfs_iter_gpu_maxd14_root0(' in launch_body and 'kernel_dfs_iter_gpu_maxd14(' in launch_body)
checks.append(('source_root0_dispatch', 'cached count root0/fallback dispatch', 'present' if root0_dispatch_ok else 'missing', root0_dispatch_ok))
host_count_ok = ('self.root_action_nonzero_count:int=0' in s and 'root_action_for_task(soa.ctrl0_arr[t],soa.markctrl_arr[t],meta_next_sched)' in s and 'soa.root_action_nonzero_count+=1' in s and 'sort_soa.root_action_nonzero_count=soa.root_action_nonzero_count' in s and '[root0-dispatch]' in s)
checks.append(('source_root_action_count_cache', 'host count cache and log present', 'present' if host_count_ok else 'missing', host_count_ok))
precompute_markers = ['root_pre2_flag','precompute_maxd14_schedule_fields','sched_hi_arr:Ptr','root_action_arr:Ptr','sched_lo_arr','sched_terminal_arr','sched_mask_arr']
active_precompute = [x for x in precompute_markers if x in active]
checks.append(('source_schedule_precompute_rejected', 'active absent', 'present=' + ','.join(active_precompute) if active_precompute else 'active absent', not active_precompute))
fail = 0
with open(summary, 'a', encoding='utf-8') as f:
    for name, exp, actual, ok in checks:
        f.write(f"{name}\t{exp}\t{actual}\t{'OK' if ok else 'FAIL'}\n")
        if not ok: fail += 1
sys.exit(1 if fail else 0)
PYCHECK
py_rc=$?
set -e
if (( py_rc != 0 )); then static_failures=$((static_failures+1)); fi

printf 'release_build_policy\tforce -release rebuild by default\tFORCE_REBUILD=%s\tOK\n' "$FORCE_REBUILD" >> "$SUMMARY"

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
  echo "[error] 273 source static checks failed" >&2
  exit 66
fi
echo "[static-ok] 273 source checks passed; proceeding to release build/run"

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[error] another 241 validation holds: $LOCK_FILE" >&2
    exit 75
  fi
fi

need_build=0
if [[ "$FORCE_REBUILD" == "1" ]]; then
  need_build=1
elif [[ ! -x "$CAND" ]]; then
  need_build=1
elif [[ "$SRC" -nt "$CAND" ]]; then
  need_build=1
fi
if (( need_build )); then
  if [[ "$AUTO_BUILD" != "1" ]]; then echo "[error] stale/missing candidate and AUTO_BUILD=$AUTO_BUILD: $CAND" >&2; exit 66; fi
  if ! command -v codon >/dev/null 2>&1; then echo "[error] codon was not found; cannot build $SRC" >&2; exit 69; fi
  rm -f "$CAND"
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
  echo "build_mode: codon build -release; FORCE_REBUILD=$FORCE_REBUILD"
  command -v sha256sum >/dev/null 2>&1 && echo "source_sha256: $(sha256sum "$SRC" | awk '{print $1}')"
  printf 'command   :'; printf ' %q' "${CMD[@]}"; echo
  echo "validation: one N=21 full run through 273 rootaction0-direct-kernel mode31 split145"
  echo "dispatch  : required=14, selected MAXD14, schedule_words=0, stack=208 bytes/thread"
  echo "================================================================"
} | tee "$RUN_LOG"

set +e; stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "$RUN_LOG"; run_rc=${PIPESTATUS[0]}; set -e
record_check run_exit 0 "$run_rc" || failures=$((failures+1))
if (( run_rc != 0 )); then exit "$run_rc"; fi

DYNAMIC_PRESET=$(sed -n 's/^\[dynamic-preset\] N=21 preset_queens=\([0-9][0-9]*\)$/\1/p' "$RUN_LOG" | tail -n1)
record_check dynamic_preset_N21 6 "${DYNAMIC_PRESET:-missing}" || failures=$((failures+1))

awk '
  /^\[maxd-dispatch\] N=21 scope=split145 / {
    rows++; m=0; req=-1; sel=-1; words=-1; bytes=-1; cap=""
    for (i=1;i<=NF;i++) { split($i,a,"="); if(a[1]=="m")m=a[2]+0; else if(a[1]=="required_maxd")req=a[2]+0; else if(a[1]=="selected_MAXD")sel=a[2]+0; else if(a[1]=="schedule_words")words=a[2]+0; else if(a[1]=="stack_bytes_per_thread")bytes=a[2]+0; else if(a[1]=="capacity_check")cap=a[2] }
    tasks+=m; if(req!=14)badreq++; if(sel!=14)badsel++; if(words!=0)badwords++; if(bytes!=208)badbytes++; if(cap!="OK")badcap++
  }
  END { printf "rows=%d\ntasks=%.0f\nbadreq=%d\nbadsel=%d\nbadwords=%d\nbadbytes=%d\nbadcap=%d\n",rows+0,tasks+0,badreq+0,badsel+0,badwords+0,badbytes+0,badcap+0 }
' "$RUN_LOG" > "$LOGDIR/dispatch.env"
source "$LOGDIR/dispatch.env"
record_check dispatch_launch_rows "$EXPECTED_CHUNKS" "$rows" || failures=$((failures+1))
record_check dispatch_task_sum "$EXPECTED_TASKS" "$tasks" || failures=$((failures+1))
record_check dispatch_non_required14 0 "$badreq" || failures=$((failures+1))
record_check dispatch_non_MAXD14 0 "$badsel" || failures=$((failures+1))
record_check dispatch_non_0_schedule_words 0 "$badwords" || failures=$((failures+1))
record_check dispatch_non_208_bytes 0 "$badbytes" || failures=$((failures+1))
record_check dispatch_bad_capacity_flag 0 "$badcap" || failures=$((failures+1))

awk '
  /^\[root0-dispatch\] N=21 scope=split145 / {
    rows++; m=0; nz=-1; r0=-1; fb=-1; sel=-1;
    for (i=1;i<=NF;i++) { split($i,a,"="); if(a[1]=="m")m=a[2]+0; else if(a[1]=="root_action_nonzero_count")nz=a[2]+0; else if(a[1]=="root0_kernel")r0=a[2]+0; else if(a[1]=="fallback_kernel")fb=a[2]+0; else if(a[1]=="selected_MAXD")sel=a[2]+0 }
    tasks+=m; nonzero+=nz; r0rows+=r0; fbrows+=fb; if(sel!=14)badsel++; if(nz!=0)badnz++; if(r0!=1)badroot0++; if(fb!=0)badfb++;
  }
  END { printf "ROOT0_ROWS=%d\nROOT0_TASKS=%.0f\nROOT0_NONZERO=%.0f\nROOT0_KERNEL_ROWS=%.0f\nROOT0_FALLBACK_ROWS=%.0f\nROOT0_BADSEL=%d\nROOT0_BADNZ=%d\nROOT0_BADROOT0=%d\nROOT0_BADFB=%d\n",rows+0,tasks+0,nonzero+0,r0rows+0,fbrows+0,badsel+0,badnz+0,badroot0+0,badfb+0 }
' "$RUN_LOG" > "$LOGDIR/root0_dispatch.env"
source "$LOGDIR/root0_dispatch.env"
record_check root0_dispatch_rows "$EXPECTED_CHUNKS" "$ROOT0_ROWS" || failures=$((failures+1))
record_check root0_dispatch_task_sum "$EXPECTED_TASKS" "$ROOT0_TASKS" || failures=$((failures+1))
record_check root0_dispatch_nonzero_sum 0 "$ROOT0_NONZERO" || failures=$((failures+1))
record_check root0_dispatch_root0_rows "$EXPECTED_CHUNKS" "$ROOT0_KERNEL_ROWS" || failures=$((failures+1))
record_check root0_dispatch_fallback_rows 0 "$ROOT0_FALLBACK_ROWS" || failures=$((failures+1))
record_check root0_dispatch_bad_selected 0 "$ROOT0_BADSEL" || failures=$((failures+1))
record_check root0_dispatch_bad_nonzero 0 "$ROOT0_BADNZ" || failures=$((failures+1))
record_check root0_dispatch_bad_root0 0 "$ROOT0_BADROOT0" || failures=$((failures+1))
record_check root0_dispatch_bad_fallback 0 "$ROOT0_BADFB" || failures=$((failures+1))

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
  for pair in "273parent271:$BASELINE_273_PARENT271_SECONDS" "272diag:$BASELINE_272_DIAG_SECONDS" "271parent241:$BASELINE_271_PARENT_SECONDS" "270rejected:$BASELINE_270_REJECT_SECONDS" "267:$BASELINE_267_SECONDS" "239:$BASELINE_239_SECONDS" "240rejected:$BASELINE_240_REJECT_SECONDS" "238:$BASELINE_238_SECONDS" "237:$BASELINE_237_SECONDS" "232:$BASELINE_232_SECONDS" "217:$BASELINE_217_SECONDS"; do
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
echo "[validation-ok] 273 N=21 mode31 split145 full reproduced total with required=14, MAXD14, 208 bytes/thread and root0 direct kernel dispatch"
