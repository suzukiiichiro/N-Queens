#!/usr/bin/env bash

set -Eeuo pipefail

# =============================================================================
# NQ_UPDATE_MEMO
# 205: 更新メモ: 204は正当性OKかつ197比で0.407秒高速のため新最速基準として採用。204 root_rest clear-lowbitを親に、MAXD14 kernelのみroot_after_second一時scalarを削り、if条件へ(root_rest^root_second)を直接inlineする微差実験を検証する。
# Full update history: see README.md
# =============================================================================

# 205 runtime-only single-pass validation harness
#
# One N=21 full GPU run reconstructs cases 01-05 from the same 131-row
# progress TSV and verifies the complete total. Candidate 205 starts from
# validated 204 because 204 is correct and numerically faster than the 197 direct-if baseline.
# 198/199/200/201/202/203 were correct but did not beat 197 and remain not adopted.
#
# Kernel arithmetic keeps 197/196/194/192/188 bitboard/schedule semantics:
# register-held current frame, 13-slot ancestor stack, scalar 4-bit nibble
# schedule, no-sibling spill elision, future_check_mask zero guard,
# schedule_words=0, and 208 local stack bytes/thread. The experiment is
# deliberately narrow: if post-root-action root_avail contains one or two bits,
# process the first depth-0 candidate once before the generic DFS loop. 3-bit
# and wider roots remain generic. The root<=2 predicate stays the validated
# 204 second-lowbit direct-if form with clear-lowbit root_rest. Only the
# root_after_second temporary scalar is removed: 205 tests
# (root_rest ^ root_second) == 0 directly in the if condition. Host task order,
# dispatch, cache, progress verification, and README-managed history remain unchanged.
#
# Shell policy:
#   - STATIC_ONLY=1 does not take the full-run lock.
#   - source static-check failure prints the summary before exiting.
#   - root-afterinline source check is whitespace/comment robust.
SRC=${SRC:-./205Py_kernel_roottwo_rootafterinline_futuremask_nosibling_maxd14_probe.py}
CAND=${CAND:-./205Py_kernel_roottwo_rootafterinline_futuremask_nosibling_maxd14_probe}
AUTO_BUILD=${AUTO_BUILD:-1}
STATIC_ONLY=${STATIC_ONLY:-0}
LOCK_FILE=${LOCK_FILE:-/tmp/205Py_kernel_roottwo_rootafterinline_futuremask_nosibling_maxd14_validation.lock}
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

BASELINE_204_SECONDS=${BASELINE_204_SECONDS:-427.795}
BASELINE_203_SECONDS=${BASELINE_203_SECONDS:-428.344}
BASELINE_202_SECONDS=${BASELINE_202_SECONDS:-428.482}
BASELINE_201_SECONDS=${BASELINE_201_SECONDS:-428.383}
BASELINE_200_SECONDS=${BASELINE_200_SECONDS:-429.520}
BASELINE_199_SECONDS=${BASELINE_199_SECONDS:-428.293}
BASELINE_198_SECONDS=${BASELINE_198_SECONDS:-428.239}
BASELINE_197_SECONDS=${BASELINE_197_SECONDS:-428.202}
BASELINE_196_SECONDS=${BASELINE_196_SECONDS:-428.239}
BASELINE_195_SECONDS=${BASELINE_195_SECONDS:-428.738}
BASELINE_194_SECONDS=${BASELINE_194_SECONDS:-428.288}
BASELINE_193_SECONDS=${BASELINE_193_SECONDS:-437.870}
BASELINE_192_SECONDS=${BASELINE_192_SECONDS:-430.118}
BASELINE_191_SECONDS=${BASELINE_191_SECONDS:-430.144}
BASELINE_190_SECONDS=${BASELINE_190_SECONDS:-473.357}
BASELINE_189_SECONDS=${BASELINE_189_SECONDS:-897.665}
BASELINE_188_SECONDS=${BASELINE_188_SECONDS:-430.137}
BASELINE_187_SECONDS=${BASELINE_187_SECONDS:-436.241}
BASELINE_186_SECONDS=${BASELINE_186_SECONDS:-554.022}
BASELINE_185_SECONDS=${BASELINE_185_SECONDS:-444.682}
BASELINE_184_SECONDS=${BASELINE_184_SECONDS:-435.635}
BASELINE_183_SECONDS=${BASELINE_183_SECONDS:-471.091}
BASELINE_182_SECONDS=${BASELINE_182_SECONDS:-471.064}
BASELINE_181_SECONDS=${BASELINE_181_SECONDS:-471.068}
BASELINE_175_SECONDS=${BASELINE_175_SECONDS:-471.106}
BASELINE_174_SECONDS=${BASELINE_174_SECONDS:-501.387}
BASELINE_173_SECONDS=${BASELINE_173_SECONDS:-748.469}
BASELINE_172_SECONDS=${BASELINE_172_SECONDS:-491.190}
BASELINE_171_SECONDS=${BASELINE_171_SECONDS:-491.231}
BASELINE_170_SECONDS=${BASELINE_170_SECONDS:-558.483}
BASELINE_169_SECONDS=${BASELINE_169_SECONDS:-573.503}
BASELINE_168_SECONDS=${BASELINE_168_SECONDS:-560.261}
BASELINE_167_SECONDS=${BASELINE_167_SECONDS:-633.039}
BASELINE_166_SECONDS=${BASELINE_166_SECONDS:-670.976}
BASELINE_165_SECONDS=${BASELINE_165_SECONDS:-667.471}
BASELINE_164_SECONDS=${BASELINE_164_SECONDS:-633.526}
BASELINE_162_SECONDS=${BASELINE_162_SECONDS:-891.060}

if [[ "$N" != "21" || "$BENCH_MODE" != "31" || "$WORKER_ID" != "0" || "$WORKER_COUNT" != "1" ]]; then
  echo "[error] fixed validation totals require N=21, BENCH_MODE=31, WORKER_ID=0, WORKER_COUNT=1" >&2
  exit 64
fi

if [[ "$STATIC_ONLY" != "1" ]] && command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[error] another 205 validation holds: $LOCK_FILE" >&2
    exit 75
  fi
fi

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="${LOG_ROOT%/}/205Py_kernel_roottwo_rootafterinline_futuremask_nosibling_maxd14_logs_N21_full_once_${TS}"
RUN_LOG="$LOGDIR/full_once.log"
BUILD_LOG="$LOGDIR/build.log"
SUMMARY="$LOGDIR/summary.tsv"
METRICS="$LOGDIR/metrics.env"
DISPATCH_METRICS="$LOGDIR/dispatch_metrics.env"
MODEL_METRICS="$LOGDIR/model_metrics.env"
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

# 205 RUNTIME-ONLY VALIDATION:
#   The long Python source/model assertion block is intentionally disabled.
#   It was useful while editing generated source, but on the cudacodon side it
#   can fail before the real Codon/CUDA validation starts if the local 205Py source
#   has a harmless header/string/layout difference. This 205 harness validates the
#   actual full GPU run, dispatch logs, chunk totals, and final answer only.
printf 'source_model_validation	skipped in runtime-only harness	not enforced	INFO
' >> "$SUMMARY"
printf 'source_revision_memo_check	skipped in runtime-only harness	not enforced	INFO
' >> "$SUMMARY"
printf 'source_history_header_check	skipped in runtime-only harness	not enforced	INFO
' >> "$SUMMARY"
printf 'source_parent_baseline	validated 204 root-rest clear-lowbit futuremask/no-sibling MAXD14 dispatch; 204 is new fastest over 197; 198 tailtest, 199 eqsecond, 200 prfuturebit, 201 prjmarkdirect, 202 rootsecondsave, and 203 rootdepth0save correct but not adopted; 195 rootthree and 193/189/190/191 not adopted	runtime-only validation	INFO
' >> "$SUMMARY"
printf 'kernel_shape_expectation	MAXD14 nibble schedule + current frame register + ancestor13 stack + no-sibling spill elision + future_check_mask zero guard + root one/two-candidate direct-if predicate + root_rest clear-lowbit + root_after_second inline if	runtime dispatch/progress checked	INFO
' >> "$SUMMARY"
printf 'schedule_words_MAXD14	0	runtime dispatch checked	INFO
' >> "$SUMMARY"
printf 'stack_bytes_MAXD14	208	runtime dispatch checked	INFO
' >> "$SUMMARY"

static_failures=0
if grep -q 'future_check_mask:u32=u32(0)' "$SRC" && grep -q 'if future_check_mask!=u32(0):' "$SRC"; then
  printf 'source_future_check_mask_guard	present	present	OK
' >> "$SUMMARY"
else
  printf 'source_future_check_mask_guard	present	missing	FAIL
' >> "$SUMMARY"
  static_failures=$((static_failures+1))
fi
if grep -q 'no-sibling tail-call descent' "$SRC" && grep -q 'save_sp:int=0' "$SRC" && grep -q 'cur_depth:int=0' "$SRC"; then
  printf 'source_nosibling_parent	188-style save_sp/cur_depth	present	OK
' >> "$SUMMARY"
else
  printf 'source_nosibling_parent	188-style save_sp/cur_depth	check source	FAIL
' >> "$SUMMARY"
  static_failures=$((static_failures+1))
fi
if python3 - "$SRC" <<'PYCHECK'
import sys
p=sys.argv[1]
s=open(p, encoding='utf-8').read()
compact=''.join(s.split())
marker='205root_afterinline:'
root_pos=compact.find(marker)
if root_pos < 0:
    root_pos=compact.find('root_first:u32=cur_avail&(u32(0)-cur_avail)')
gen_pos=compact.find('whileTrue:', root_pos)
root_block=compact[root_pos:gen_pos if gen_pos >= 0 else len(compact)]
generic_block=compact[gen_pos if gen_pos >= 0 else 0:]
checks = {
    'root_clearrest_present': 'root_rest:u32=cur_avail&(cur_avail-u32(1))' in compact,
    'root_xor_rest_removed': 'root_rest:u32=cur_avail^root_first' not in compact,
    'root_after_scalar_absent': 'root_after_second:u32=' not in compact,
    'root_inline_if_present': 'if(root_rest^root_second)==u32(0):' in compact,
    'root_old_direct_if_absent': 'ifroot_after_second==u32(0):' not in compact,
    'root_preroll_flag_absent': 'root_preroll:u32' not in s,
    # 205 keeps 204/197 save payload; 203 depth0-save is not adopted.
    'root_depth_pack_present': 'avail[save_sp]=cur_avail|(u32(cur_depth)<<u32(27))' in root_block,
    'generic_depth_pack_present': 'avail[save_sp]=cur_avail|(u32(cur_depth)<<u32(27))' in generic_block,
    # 205 keeps 204/197 payload source; 202 root_second payload is not adopted.
    'root_second_save_payload_absent': 'avail[save_sp]=root_second|(u32(cur_depth)<<u32(27))' not in compact,
    'root_second_active_payload_absent': 'cur_avail=root_second' not in compact,
    # 205 keeps 204/197 root-preroll child_jmark scalar; 201 direct bit0 is not adopted.
    'pr_child_jmark_scalar_present': 'pr_child_jmark:u32=child_jmark_mask&u32(1)' in compact,
    'direct_child_jmark_bit0_absent': 'if(child_jmark_mask&u32(1))!=u32(0):' not in compact,
    # 205 keeps 204/197 root-preroll future guard; 200 direct pr_nibble bit8 is not adopted.
    'pr_future_mask_guard_present': 'ifpr_descend!=u32(0):iffuture_check_mask!=u32(0):if(pr_nibble_op&u32(8))!=u32(0):' in compact,
    'prfuturebit_direct_shape_absent': 'rootpr_futurebit' not in s,
    'tailtest_absent': 'root_rest_more:u32' not in s,
}
# For 199's eqsecond experiment, reject only an active predicate, not prose.
checks['eqsecond_active_absent'] = 'ifroot_rest==root_second:' not in compact
if all(checks.values()):
    sys.exit(0)
print('source_rootafterinline_pycheck_failed=' + ','.join(k for k,v in checks.items() if not v))
sys.exit(1)
PYCHECK
then
  printf 'source_rootafterinline\tpresent_inline_root_after_predicate\tpresent_inline_root_after_predicate\tOK
' >> "$SUMMARY"
else
  printf 'source_rootafterinline\tpresent_inline_root_after_predicate\tmissing_or_old_root_after_scalar\tFAIL
' >> "$SUMMARY"
  static_failures=$((static_failures+1))
fi
if ! grep -q 'root_three_preroll' "$SRC" && ! grep -q 'root_rest_tail:u32=root_rest&(root_rest-u32(1))' "$SRC"; then
  printf 'source_195_rootthree_removed	absent	absent	OK
' >> "$SUMMARY"
else
  printf 'source_195_rootthree_removed	absent	present	FAIL
' >> "$SUMMARY"
  static_failures=$((static_failures+1))
fi
if ! grep -q 'block_check_mask:u32' "$SRC" && ! grep -q 'if block_check_mask' "$SRC"; then
  printf 'source_190_blockmask_guard_removed	absent	absent	OK
' >> "$SUMMARY"
else
  printf 'source_190_blockmask_guard_removed	absent	present	FAIL
' >> "$SUMMARY"
  static_failures=$((static_failures+1))
fi
if ! grep -q 'cur_ld\^bit' "$SRC" && ! grep -q 'cur_rd\^bit' "$SRC" && ! grep -q 'cur_col\^bit' "$SRC"; then
  printf 'source_191_place_xor_removed	absent	absent	OK
' >> "$SUMMARY"
else
  printf 'source_191_place_xor_removed	absent	present	FAIL
' >> "$SUMMARY"
  static_failures=$((static_failures+1))
fi
if ! grep -q 'fast_nibble_op:u32=u32(0)' "$SRC" && ! grep -q 'fast_child_jmark:u32=' "$SRC"; then
  printf 'source_189_forced_chain_removed	absent	absent	OK
' >> "$SUMMARY"
else
  printf 'source_189_forced_chain_removed	absent	present	FAIL
' >> "$SUMMARY"
  static_failures=$((static_failures+1))
fi
if ! grep -q 'if child_jmark_mask!=u32(0):' "$SRC"; then
  printf 'source_187_jmark_guard_removed	absent	absent	OK
' >> "$SUMMARY"
else
  printf 'source_187_jmark_guard_removed	absent	present	FAIL
' >> "$SUMMARY"
  static_failures=$((static_failures+1))
fi
if ! grep -q 'if cur_depth==13:' "$SRC"; then
  printf 'source_terminal13_literal_removed	absent	absent	OK
' >> "$SUMMARY"
else
  printf 'source_terminal13_literal_removed	absent	present	FAIL
' >> "$SUMMARY"
  static_failures=$((static_failures+1))
fi
if python3 - "$SRC" <<'PYSPLIT'
import re, sys
s=open(sys.argv[1], encoding='utf-8').read()
tags=set(re.findall(r'split\d+', s))
allowed={'split145','split205'}
bad=sorted(tags-allowed)
if 'split205' in tags and not bad:
    sys.exit(0)
print('bad=' + ','.join(bad) + ' tags=' + ','.join(sorted(tags)))
sys.exit(1)
PYSPLIT
then
  actual_split=$(python3 - "$SRC" <<'PYSPLIT2'
import re, sys
s=open(sys.argv[1], encoding='utf-8').read()
tags=sorted(set(re.findall(r'split\d+', s)))
print(' '.join(tags))
PYSPLIT2
)
  printf 'source_split_tag	split205 active preferred	%s	OK
' "$actual_split" >> "$SUMMARY"
else
  actual_split=$(python3 - "$SRC" <<'PYSPLIT3'
import re, sys
s=open(sys.argv[1], encoding='utf-8').read()
tags=sorted(set(re.findall(r'split\d+', s)))
print(' '.join(tags) if tags else 'none')
PYSPLIT3
)
  printf 'source_split_tag	split205 active preferred	%s	FAIL
' "$actual_split" >> "$SUMMARY"
  static_failures=$((static_failures+1))
fi

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "================================================================"
  echo "[static-summary]"
  column -t -s $'	' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
  echo "[logdir] $LOGDIR"
  echo "[static-validation-skipped] 205 runtime-only harness disabled source/model assertions"
  if (( static_failures != 0 )); then exit 1; fi
  exit 0
fi

if (( static_failures != 0 )); then
  echo "================================================================"
  echo "[static-summary]"
  column -t -s $'	' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
  echo "[logdir] $LOGDIR"
  echo "[static-validation-skipped] 205 runtime-only harness disabled source/model assertions"
  echo "[error] source roottwo-rootafterinline static checks failed" >&2
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
  echo "dispatch  : required=14, selected MAXD14, schedule_words=0, stack=208 bytes/thread, 205 roottwo-rootafterinline futuremask/no-sibling kernel + README history policy + tail-flatness metrics"
  echo "================================================================"
} | tee "$RUN_LOG"

set +e
stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "$RUN_LOG"
run_rc=${PIPESTATUS[0]}
set -e
printf 'run_exit\t0\t%d\t%s\n' "$run_rc" "$([[ $run_rc -eq 0 ]] && echo OK || echo FAIL)" >> "$SUMMARY"
if (( run_rc != 0 )); then exit "$run_rc"; fi

# Runtime dispatch proof from the same 131 launches.
awk '
  /^\[maxd-dispatch\] N=21 scope=split145 / {
    rows++
    m=0; req=-1; sel=-1; words=-1; bytes=-1; cap=""
    for (i=1;i<=NF;i++) {
      split($i,a,"=")
      if (a[1]=="m") m=a[2]+0
      else if (a[1]=="required_maxd") req=a[2]+0
      else if (a[1]=="selected_MAXD") sel=a[2]+0
      else if (a[1]=="schedule_words") words=a[2]+0
      else if (a[1]=="stack_bytes_per_thread") bytes=a[2]+0
      else if (a[1]=="capacity_check") cap=a[2]
    }
    tasks+=m
    if (rows==1 || req<minreq) minreq=req
    if (req>maxreq) maxreq=req
    if (req!=14) badreq++
    if (sel!=14) badsel++
    if (words!=0) badwords++
    if (bytes!=208) badbytes++
    if (cap!="OK") badcap++
    if (req>sel) under++
  }
  END {
    printf "DISPATCH_ROWS=%d\n",rows+0
    printf "DISPATCH_TASKS=%.0f\n",tasks+0
    printf "DISPATCH_MIN_REQUIRED=%d\n",minreq+0
    printf "DISPATCH_MAX_REQUIRED=%d\n",maxreq+0
    printf "DISPATCH_NON14=%d\n",badreq+0
    printf "DISPATCH_NON_SELECTED14=%d\n",badsel+0
    printf "DISPATCH_NON0WORDS=%d\n",badwords+0
    printf "DISPATCH_NON208=%d\n",badbytes+0
    printf "DISPATCH_BAD_CAPACITY=%d\n",badcap+0
    printf "DISPATCH_UNDERSIZED=%d\n",under+0
  }
' "$RUN_LOG" > "$DISPATCH_METRICS"
# shellcheck disable=SC1090
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

PROGRESS=$(sed -n 's/.* progress=\([^[:space:]]*\.tsv\).*/\1/p' "$RUN_LOG" | tail -n 1 | tr -d '\r')
if [[ -z "$PROGRESS" || ! -s "$PROGRESS" ]]; then
  printf 'progress_file\tpresent\t%s\tFAIL\n' "${PROGRESS:-missing}" >> "$SUMMARY"
  echo "[error] full-run progress TSV was not found" >&2
  exit 65
fi
cp -f "$PROGRESS" "$ARCHIVED_PROGRESS"
printf 'progress_file\tpresent\t%s\tOK\n' "$PROGRESS" >> "$SUMMARY"

awk -F '\t' -v expected_chunks="$EXPECTED_CHUNKS" '
  NR==1 {
    for (i=1;i<=NF;i++) {
      if ($i=="chunk") chunk_col=i
      else if ($i=="chunk_total") total_col=i
      else if ($i=="gpu_total") gpu_col=i
    }
    if (!chunk_col || !total_col || !gpu_col) { print "PARSE_OK=0"; exit 2 }
    next
  }
  {
    chunk=$(chunk_col)+0; value=$(total_col)+0; gpu=$(gpu_col)+0
    rows++; full+=value; last_gpu=gpu
    if (seen[chunk]++) duplicates++
    if (chunk==0 || chunk==20 || chunk==40 || chunk==60 || chunk==80 || chunk==100 || chunk==120) p1+=value
    if (chunk==35 || chunk==40 || chunk==41 || chunk==42 || chunk==47 || chunk==48 || chunk==52 || chunk==53) p2+=value
    if (chunk==20 || chunk==40 || chunk==55 || chunk==56 || chunk==57 || chunk==58 || chunk==60) p3+=value
    if (chunk==100 || chunk==105 || chunk==110 || chunk==115 || chunk==120 || chunk==125 || chunk==130) p4+=value
    if ((chunk%4)==0) p5+=value
  }
  END {
    if (!chunk_col || !total_col || !gpu_col) exit
    for (i=0;i<expected_chunks;i++) if (!(i in seen)) missing++
    printf "PARSE_OK=1\nROWS=%.0f\nDUPLICATES=%.0f\nMISSING=%.0f\n",rows,duplicates,missing
    printf "P1=%.0f\nP2=%.0f\nP3=%.0f\nP4=%.0f\nP5=%.0f\n",p1,p2,p3,p4,p5
    printf "FULL=%.0f\nLAST_GPU=%.0f\n",full,last_gpu
  }
' "$ARCHIVED_PROGRESS" > "$METRICS"
# shellcheck disable=SC1090
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

# 201 keeps the validated 197 host/chunk structure, then derives chunk-tail diagnostics from
# the existing progress TSV without changing GPU arithmetic.
TAIL_METRICS="$LOGDIR/chunk_tail_metrics.env"
awk -F '\t' '
  NR==1 {
    for (i=1;i<=NF;i++) {
      if ($i=="elapsed_ms") elapsed_col=i
      else if ($i=="chunk") chunk_col=i
    }
    next
  }
  {
    n++; e=$elapsed_col+0; c=$chunk_col+0; vals[n]=e; chunks[n]=c; sum+=e
    if (n==1 || e<min) { min=e; minc=c }
    if (e>max) { max=e; maxc=c }
  }
  END {
    for (i=1;i<=n;i++) {
      for (j=i+1;j<=n;j++) {
        if (vals[j]<vals[i]) {
          te=vals[i]; vals[i]=vals[j]; vals[j]=te
          tc=chunks[i]; chunks[i]=chunks[j]; chunks[j]=tc
        }
      }
    }
    p50=vals[int((n+1)*50/100)]; p90=vals[int((n+1)*90/100)]; p95=vals[int((n+1)*95/100)]
    printf "CHUNK_ELAPSED_ROWS=%d\n",n+0
    printf "CHUNK_ELAPSED_SUM_MS=%d\n",sum+0
    printf "CHUNK_ELAPSED_MIN_MS=%d\n",min+0
    printf "CHUNK_ELAPSED_MIN_CHUNK=%d\n",minc+0
    printf "CHUNK_ELAPSED_MAX_MS=%d\n",max+0
    printf "CHUNK_ELAPSED_MAX_CHUNK=%d\n",maxc+0
    printf "CHUNK_ELAPSED_P50_MS=%d\n",p50+0
    printf "CHUNK_ELAPSED_P90_MS=%d\n",p90+0
    printf "CHUNK_ELAPSED_P95_MS=%d\n",p95+0
  }
' "$ARCHIVED_PROGRESS" > "$TAIL_METRICS"
# shellcheck disable=SC1090
source "$TAIL_METRICS"
printf 'chunk_elapsed_sum_ms\tINFO\t%s\tINFO\n' "$CHUNK_ELAPSED_SUM_MS" >> "$SUMMARY"
printf 'chunk_elapsed_min_ms\tINFO\t%s@chunk%s\tINFO\n' "$CHUNK_ELAPSED_MIN_MS" "$CHUNK_ELAPSED_MIN_CHUNK" >> "$SUMMARY"
printf 'chunk_elapsed_max_ms\tINFO\t%s@chunk%s\tINFO\n' "$CHUNK_ELAPSED_MAX_MS" "$CHUNK_ELAPSED_MAX_CHUNK" >> "$SUMMARY"
printf 'chunk_elapsed_p50_p90_p95_ms\tINFO\t%s/%s/%s\tINFO\n' "$CHUNK_ELAPSED_P50_MS" "$CHUNK_ELAPSED_P90_MS" "$CHUNK_ELAPSED_P95_MS" >> "$SUMMARY"
TAIL_RANGE_MS=$((CHUNK_ELAPSED_MAX_MS-CHUNK_ELAPSED_MIN_MS))
TAIL_P95_OVER_P50_PERMILLE=$(awk -v p95="$CHUNK_ELAPSED_P95_MS" -v p50="$CHUNK_ELAPSED_P50_MS" 'BEGIN { if (p50>0) printf "%d", int((p95*1000)/p50); else printf "0" }')
printf 'tail_range_ms\tINFO\t%s\tINFO\n' "$TAIL_RANGE_MS" >> "$SUMMARY"
printf 'tail_p95_over_p50_permille\tINFO\t%s/1000\tINFO\n' "$TAIL_P95_OVER_P50_PERMILLE" >> "$SUMMARY"


# 201: richer tail diagnostics from the already-validated progress TSV.
# This does not touch GPU arithmetic or task order. It writes the slowest chunks
# and a suggested chunk-list for the next focused probe.
TAIL_TOP="$LOGDIR/tail_top_chunks.tsv"
TAIL_BUCKETS="$LOGDIR/tail_bucket_summary.tsv"
TAIL_EXTENDED="$LOGDIR/tail_extended_metrics.env"
python3 - "$ARCHIVED_PROGRESS" "$TAIL_TOP" "$TAIL_BUCKETS" "$TAIL_EXTENDED" <<'PYTAIL'
import csv, sys
from pathlib import Path

progress=Path(sys.argv[1])
top_path=Path(sys.argv[2])
bucket_path=Path(sys.argv[3])
env_path=Path(sys.argv[4])

with progress.open(newline='') as f:
    rows=list(csv.DictReader(f, delimiter='\t'))

def to_int(row, key, default=0):
    try:
        return int(float(row.get(key, default) or default))
    except Exception:
        return default

def to_float(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default

items=[]
for r in rows:
    chunk=to_int(r,'chunk')
    elapsed=to_int(r,'elapsed_ms')
    m=to_int(r,'m')
    chunk_total=to_int(r,'chunk_total')
    score_avg=to_float(r,'score_avg')
    free_avg=to_float(r,'free_popcount_avg')
    depth_avg=to_float(r,'depth_avg')
    best_fid=-1
    best_count=-1
    for fid in range(28):
        c=to_int(r,f'funcid_{fid}_count')
        if c>best_count:
            best_count=c
            best_fid=fid
    items.append({
        'chunk':chunk,'elapsed_ms':elapsed,'m':m,'chunk_total':chunk_total,
        'score_avg':score_avg,'free_popcount_avg':free_avg,'depth_avg':depth_avg,
        'dominant_funcid':best_fid,'dominant_funcid_count':best_count,
    })

items_sorted=sorted(items, key=lambda x:(x['elapsed_ms'], x['chunk']), reverse=True)
with top_path.open('w', newline='') as f:
    w=csv.writer(f, delimiter='\t')
    w.writerow(['rank','chunk','elapsed_ms','m','chunk_total','score_avg','free_popcount_avg','depth_avg','dominant_funcid','dominant_funcid_count'])
    for rank,item in enumerate(items_sorted[:25],1):
        w.writerow([rank,item['chunk'],item['elapsed_ms'],item['m'],item['chunk_total'],f"{item['score_avg']:.3f}",f"{item['free_popcount_avg']:.3f}",f"{item['depth_avg']:.3f}",item['dominant_funcid'],item['dominant_funcid_count']])

buckets=[(0,31),(32,63),(64,95),(96,130)]
with bucket_path.open('w', newline='') as f:
    w=csv.writer(f, delimiter='\t')
    w.writerow(['chunk_range','rows','elapsed_sum_ms','elapsed_avg_ms','elapsed_max_ms','max_chunk'])
    for a,b in buckets:
        sub=[x for x in items if a<=x['chunk']<=b]
        if not sub:
            w.writerow([f'{a}-{b}',0,0,'0.000',0,-1])
            continue
        s=sum(x['elapsed_ms'] for x in sub)
        mx=max(sub, key=lambda x:x['elapsed_ms'])
        w.writerow([f'{a}-{b}',len(sub),s,f'{s/len(sub):.3f}',mx['elapsed_ms'],mx['chunk']])

total_elapsed=sum(x['elapsed_ms'] for x in items)
top5=sum(x['elapsed_ms'] for x in items_sorted[:5])
top10=sum(x['elapsed_ms'] for x in items_sorted[:10])
slow_threshold=items_sorted[max(0, min(len(items_sorted)-1, int(len(items_sorted)*0.05)-1))]['elapsed_ms'] if items_sorted else 0
suggested=','.join(str(x['chunk']) for x in items_sorted[:16])

with env_path.open('w') as f:
    f.write(f"TAIL_TOP1_CHUNK={items_sorted[0]['chunk'] if items_sorted else -1}\n")
    f.write(f"TAIL_TOP1_MS={items_sorted[0]['elapsed_ms'] if items_sorted else 0}\n")
    f.write(f"TAIL_TOP5_SUM_MS={top5}\n")
    f.write(f"TAIL_TOP10_SUM_MS={top10}\n")
    f.write(f"TAIL_TOTAL_ELAPSED_MS={total_elapsed}\n")
    f.write(f"TAIL_TOP5_SHARE_PERMILLE={(top5*1000//total_elapsed) if total_elapsed else 0}\n")
    f.write(f"TAIL_TOP10_SHARE_PERMILLE={(top10*1000//total_elapsed) if total_elapsed else 0}\n")
    f.write(f"TAIL_SLOW_THRESHOLD_MS={slow_threshold}\n")
    f.write(f"TAIL_SUGGESTED_CHUNKS={suggested}\n")
PYTAIL
# shellcheck disable=SC1090
source "$TAIL_EXTENDED"
printf 'tail_top1_ms\tINFO\t%s@chunk%s\tINFO\n' "$TAIL_TOP1_MS" "$TAIL_TOP1_CHUNK" >> "$SUMMARY"
printf 'tail_top5_share_permille\tINFO\t%s/1000\tINFO\n' "$TAIL_TOP5_SHARE_PERMILLE" >> "$SUMMARY"
printf 'tail_top10_share_permille\tINFO\t%s/1000\tINFO\n' "$TAIL_TOP10_SHARE_PERMILLE" >> "$SUMMARY"
printf 'tail_suggested_chunks\tINFO\t%s\tINFO\n' "$TAIL_SUGGESTED_CHUNKS" >> "$SUMMARY"
printf 'tail_top_chunks_file\tINFO\t%s\tINFO\n' "$TAIL_TOP" >> "$SUMMARY"
printf 'tail_bucket_summary_file\tINFO\t%s\tINFO\n' "$TAIL_BUCKETS" >> "$SUMMARY"

FINAL_LINE=$(grep -E "^[[:space:]]*${N}:" "$RUN_LOG" | tail -n1 || true)
if [[ "$FINAL_LINE" == *"$FULL_TOTAL"* && "$FINAL_LINE" == *"ok"* ]]; then
  printf 'final_output\t%s ... ok\t%s\tOK\n' "$FULL_TOTAL" "$FINAL_LINE" >> "$SUMMARY"
else
  printf 'final_output\t%s ... ok\t%s\tFAIL\n' "$FULL_TOTAL" "${FINAL_LINE:-missing}" >> "$SUMMARY"
  failures=$((failures+1))
fi

ERROR_HITS=$(grep -Eic '\[(.*-)?error\]|mismatch|ng\(' "$RUN_LOG" || true)
record_check "error_or_mismatch_hits" "0" "$ERROR_HITS" || failures=$((failures+1))

ELAPSED_TEXT=$(awk -v n="$N" '$0 ~ "^[[:space:]]*" n ":" {v=$(NF-1)} END {print v}' "$RUN_LOG")
if [[ -n "$ELAPSED_TEXT" ]]; then
  ELAPSED_SECONDS=$(awk -F: '{printf "%.3f",($1*3600)+($2*60)+$3}' <<< "$ELAPSED_TEXT")
  for pair in "204:$BASELINE_204_SECONDS" "203:$BASELINE_203_SECONDS" "202:$BASELINE_202_SECONDS" "201:$BASELINE_201_SECONDS" "200:$BASELINE_200_SECONDS" "199:$BASELINE_199_SECONDS" "198:$BASELINE_198_SECONDS" "197:$BASELINE_197_SECONDS" "196:$BASELINE_196_SECONDS" "195:$BASELINE_195_SECONDS" "194:$BASELINE_194_SECONDS" "193:$BASELINE_193_SECONDS" "192:$BASELINE_192_SECONDS" "191:$BASELINE_191_SECONDS" "190:$BASELINE_190_SECONDS" "189:$BASELINE_189_SECONDS" "188:$BASELINE_188_SECONDS" "187:$BASELINE_187_SECONDS" "186:$BASELINE_186_SECONDS" "185:$BASELINE_185_SECONDS" "184:$BASELINE_184_SECONDS" "183:$BASELINE_183_SECONDS" "182:$BASELINE_182_SECONDS" "181:$BASELINE_181_SECONDS" "175:$BASELINE_175_SECONDS" "174:$BASELINE_174_SECONDS" "173:$BASELINE_173_SECONDS" "172:$BASELINE_172_SECONDS" "171:$BASELINE_171_SECONDS" "170:$BASELINE_170_SECONDS" "169:$BASELINE_169_SECONDS" "168:$BASELINE_168_SECONDS" "167:$BASELINE_167_SECONDS" "166:$BASELINE_166_SECONDS" "165:$BASELINE_165_SECONDS" "164:$BASELINE_164_SECONDS" "162:$BASELINE_162_SECONDS"; do
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

echo "[validation-ok] 205 roottwo-rootafterinline futuremask/no-sibling reproduced cases 01-06 with required=14, MAXD14, 208 bytes/thread"
