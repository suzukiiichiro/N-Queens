#!/usr/bin/env bash

set -Eeuo pipefail

# 156 single-pass validation harness
#
# Replaces both 148Py1-6.sh and 148Py1-6_locked.sh.
# It runs the N=21 full 131-chunk job exactly once, then reconstructs all
# former cases 01-05 from the full-run progress TSV:
#   01-04: fixed selected-chunk sums
#   05:    worker 0-of-4 sum (chunk_index % 4 == 0)
#   06:    full N=21 total
#
# The source intentionally reuses the validated 148 reordered-bin cache.
# Candidate 156 retains the validated 155 ctrl0:u32 prepack and all prior DFS logic.
# Its sole experiment is to keep all eight fields written into initialized ctrl
# words as u32 before the final OR/shift pack. rowv/stepv and mark/end comparisons
# remain int so the next mark/end-u32 experiment stays independently measurable.
# A real flock guard prevents two copies of this harness from competing.

SRC=${SRC:-./156Py_ctrl_fields_u32_probe.py}
CAND=${CAND:-./156Py_ctrl_fields_u32_probe}
AUTO_BUILD=${AUTO_BUILD:-1}
LOCK_FILE=${LOCK_FILE:-/tmp/156Py_ctrl_fields_u32_validation.lock}
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
EXP01=16879968420
EXP02=19219113480
EXP03=16935522136
EXP04=16230260724
EXP05=79383179384
FULL_TOTAL=314666222712

# Direct validated 155/154/153/152/151/150/149 full runs and historical 148 mean (informational only).
BASELINE_155_SECONDS=${BASELINE_155_SECONDS:-964.864}
BASELINE_154_SECONDS=${BASELINE_154_SECONDS:-963.903}
BASELINE_153_SECONDS=${BASELINE_153_SECONDS:-965.305}
BASELINE_152_SECONDS=${BASELINE_152_SECONDS:-970.212}
BASELINE_151_SECONDS=${BASELINE_151_SECONDS:-976.854}
BASELINE_150_SECONDS=${BASELINE_150_SECONDS:-976.988}
BASELINE_149_SECONDS=${BASELINE_149_SECONDS:-976.932}
BASELINE_148_SECONDS=${BASELINE_148_SECONDS:-976.825}

if [[ "$N" != "21" || "$BENCH_MODE" != "31" || "$WORKER_ID" != "0" || "$WORKER_COUNT" != "1" ]]; then
  echo "[error] fixed validation totals require N=21, BENCH_MODE=31, WORKER_ID=0, WORKER_COUNT=1" >&2
  exit 64
fi

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[error] another 156 validation holds: $LOCK_FILE" >&2
    exit 75
  fi
fi

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="${LOG_ROOT%/}/156Py_ctrl_fields_u32_logs_N21_full_once_${TS}"
RUN_LOG="$LOGDIR/full_once.log"
BUILD_LOG="$LOGDIR/build.log"
SUMMARY="$LOGDIR/summary.tsv"
METRICS="$LOGDIR/metrics.env"
ARCHIVED_PROGRESS="$LOGDIR/progress_full.tsv"
mkdir -p "$LOGDIR"

printf 'check\texpected\tactual\tstatus\n' > "$SUMMARY"

record_check() {
  local name=$1 expected=$2 actual=$3
  local status=FAIL
  if [[ "$actual" == "$expected" ]]; then
    status=OK
  fi
  printf '%s\t%s\t%s\t%s\n' "$name" "$expected" "$actual" "$status" >> "$SUMMARY"
  [[ "$status" == OK ]]
}

if [[ -f "$SRC" ]]; then
  source_failures=0
  KERNEL_SNIP="$LOGDIR/kernel_source.tmp"
  awk '
    /^@gpu\.kernel$/ { in_kernel=1 }
    in_kernel { print }
    in_kernel && /^####################################################################################################$/ { exit }
  ' "$SRC" > "$KERNEL_SNIP"

  # Retain the validated 154 u32 GPU-input layout.
  grep -Fq 'self.ld_arr:List[u32]' "$SRC" || source_failures=$((source_failures+1))
  grep -Fq 'self.rd_arr:List[u32]' "$SRC" || source_failures=$((source_failures+1))
  grep -Fq 'self.col_arr:List[u32]' "$SRC" || source_failures=$((source_failures+1))
  grep -Fq 'self.free_arr:List[u32]' "$SRC" || source_failures=$((source_failures+1))
  grep -Fq 'm:int,board_mask:u32' "$SRC" || source_failures=$((source_failures+1))
  grep -Fq 'n3:u32,n4:u32' "$SRC" || source_failures=$((source_failures+1))

  # Retain the validated 155 ctrl0:u32 host prepack and one-pointer GPU input.
  grep -Fq 'self.ctrl0_arr:List[u32]=[u32(0)]*m' "$SRC" || source_failures=$((source_failures+1))
  grep -Fq 'col_arr:Ptr[u32],ctrl0_arr:Ptr[u32],free_arr:Ptr[u32]' "$SRC" || source_failures=$((source_failures+1))
  grep -Fq 'soa.ctrl0_arr[t]=u32(target)|(u32(start)<<u32(5))' "$SRC" || source_failures=$((source_failures+1))
  grep -Fq 'ctrl[0]=ctrl0_arr[i]' "$SRC" || source_failures=$((source_failures+1))

  ctrl0_launch_refs=$(grep -Ec 'gpu\.raw\((sort_)?soa\.ctrl0_arr\)' "$SRC" || true)
  sort_ctrl0_copies=$(grep -Fc 'sort_soa.ctrl0_arr[p]=soa.ctrl0_arr[q]' "$SRC" || true)
  [[ "$ctrl0_launch_refs" == "5" ]] || source_failures=$((source_failures+1))
  [[ "$sort_ctrl0_copies" == "2" ]] || source_failures=$((source_failures+1))

  if grep -Eq 'gpu\.raw\((sort_)?soa\.(row_arr|funcid_arr)\)' "$SRC" || \
     grep -Fq 'ctrl[0]=u32(funcid_arr[i] | (row_arr[i]<<5))' "$SRC"; then
    source_failures=$((source_failures+1))
  fi

  # 156 sole experiment: all initialized-ctrl fields are u32 before final packing.
  for decl in \
    'nextfidu:u32' \
    'child_rowu:u32' \
    'frame_stepu:u32' \
    'frame_addu:u32' \
    'frame_blocksu:u32' \
    'frame_fcvu:u32' \
    'frame_bLiu:u32' \
    'frame_ktu:u32'
  do
    grep -Fq "$decl" "$KERNEL_SNIP" || source_failures=$((source_failures+1))
  done

  # The final pack must be a pure u32 OR/shift expression, with typed shift counts.
  for expr in \
    'u32(INIT_MASK)' \
    '|nextfidu' \
    '|(child_rowu<<u32(5))' \
    '|(frame_stepu<<u32(10))' \
    '|(frame_addu<<u32(12))' \
    '|(frame_blocksu<<u32(13))' \
    '|(frame_fcvu<<u32(14))' \
    '|(frame_bLiu<<u32(15))' \
    '|(frame_ktu<<u32(17))'
  do
    grep -Fq "$expr" "$KERNEL_SNIP" || source_failures=$((source_failures+1))
  done

  # 156 deliberately leaves row/step and mark/end comparison values as int.
  grep -Fq 'rowv:int=(cv0i>>5)&31' "$KERNEL_SNIP" || source_failures=$((source_failures+1))
  grep -Fq 'stepv:int=1' "$KERNEL_SNIP" || source_failures=$((source_failures+1))
  grep -Fq 'jmark:int=jmark_arr[i]' "$KERNEL_SNIP" || source_failures=$((source_failures+1))
  grep -Fq 'endm:int=end_arr[i]' "$KERNEL_SNIP" || source_failures=$((source_failures+1))
  grep -Fq 'mark1:int=mark1_arr[i]' "$KERNEL_SNIP" || source_failures=$((source_failures+1))
  grep -Fq 'mark2:int=mark2_arr[i]' "$KERNEL_SNIP" || source_failures=$((source_failures+1))

  # Reject the legacy all-int ctrl pack or int declarations for ctrl fields.
  if grep -Fq 'cv=u32(524288 | nextfv | (child_row<<5)' "$KERNEL_SNIP" || \
     grep -Fq 'markctrl_arr' "$KERNEL_SNIP" || \
     grep -Eq '^[[:space:]]*(nextfv|addv|use_blocks|fcv|bLv|ktype|child_row):int=' "$KERNEL_SNIP"; then
    source_failures=$((source_failures+1))
  fi

  # Exhaustively verify the 20-bit ctrl layout for all valid field combinations.
  if ! pack_cases=$(awk 'BEGIN {
      cases=0
      for (nextfid=0; nextfid<28; nextfid++)
      for (child=0; child<32; child++)
      for (step=1; step<=3; step++)
      for (add1=0; add1<=1; add1++)
      for (blocks=0; blocks<=1; blocks++)
      for (future=0; future<=1; future++)
      for (blockL=0; blockL<4; blockL++)
      for (ktype=0; ktype<3; ktype++) {
        oldpack=524288 + nextfid + child*32 + step*1024 + add1*4096 + blocks*8192 + future*16384 + blockL*32768 + ktype*131072
        newpack=524288 + nextfid + child*32 + step*1024 + add1*4096 + blocks*8192 + future*16384 + blockL*32768 + ktype*131072
        if (oldpack != newpack) exit 1
        if ((newpack % 32) != nextfid) exit 1
        if ((int(newpack/32) % 32) != child) exit 1
        if ((int(newpack/1024) % 4) != step) exit 1
        if ((int(newpack/4096) % 2) != add1) exit 1
        if ((int(newpack/8192) % 2) != blocks) exit 1
        if ((int(newpack/16384) % 2) != future) exit 1
        if ((int(newpack/32768) % 4) != blockL) exit 1
        if ((int(newpack/131072) % 4) != ktype) exit 1
        if ((int(newpack/524288) % 2) != 1) exit 1
        if (newpack < 0 || newpack >= 1048576) exit 1
        cases++
      }
      print cases
    }'); then
    source_failures=$((source_failures+1))
    pack_cases=0
  fi

  if (( source_failures != 0 )); then
    printf 'source_ctrl_fields_u32_shape\trequested layout\t%d check failures\tFAIL\n' "$source_failures" >> "$SUMMARY"
    echo "[error] source does not match the requested 156 initialized-ctrl field-u32 experiment" >&2
    exit 65
  fi
  printf 'source_ctrl_fields_u32_shape\trequested layout\tverified\tOK\n' >> "$SUMMARY"
  printf 'ctrl_field_pack_equivalence\t258048 cases\t%s cases\tOK\n' "$pack_cases" >> "$SUMMARY"
  rm -f "$KERNEL_SNIP"
fi

need_build=0
if [[ ! -x "$CAND" ]]; then
  need_build=1
elif [[ -f "$SRC" && "$SRC" -nt "$CAND" ]]; then
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
  if (( build_rc != 0 )); then
    printf 'build_exit\t0\t%d\tFAIL\n' "$build_rc" >> "$SUMMARY"
    exit "$build_rc"
  fi
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
  if command -v sha256sum >/dev/null 2>&1 && [[ -f "$SRC" ]]; then
    echo "source_sha256: $(sha256sum "$SRC" | awk '{print $1}')"
  fi
  printf 'command   :'
  printf ' %q' "${CMD[@]}"
  echo
  echo "validation: one full run; cases 01-05 reconstructed from its progress TSV"
  echo "================================================================"
} | tee "$RUN_LOG"

set +e
stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "$RUN_LOG"
run_rc=${PIPESTATUS[0]}
set -e
printf 'run_exit\t0\t%d\t%s\n' "$run_rc" "$([[ $run_rc -eq 0 ]] && echo OK || echo FAIL)" >> "$SUMMARY"
if (( run_rc != 0 )); then
  echo "[error] candidate exited with $run_rc" >&2
  exit "$run_rc"
fi

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
    for (i=1; i<=NF; i++) {
      if ($i=="chunk") chunk_col=i
      else if ($i=="chunk_total") total_col=i
      else if ($i=="gpu_total") gpu_col=i
    }
    if (!chunk_col || !total_col || !gpu_col) {
      print "PARSE_OK=0"
      exit 2
    }
    next
  }
  {
    chunk=$(chunk_col)+0
    value=$(total_col)+0
    gpu=$(gpu_col)+0
    rows++
    full+=value
    last_gpu=gpu
    if (seen[chunk]++) duplicates++

    if (chunk==0 || chunk==20 || chunk==40 || chunk==60 || chunk==80 || chunk==100 || chunk==120) p1+=value
    if (chunk==35 || chunk==40 || chunk==41 || chunk==42 || chunk==47 || chunk==48 || chunk==52 || chunk==53) p2+=value
    if (chunk==20 || chunk==40 || chunk==55 || chunk==56 || chunk==57 || chunk==58 || chunk==60) p3+=value
    if (chunk==100 || chunk==105 || chunk==110 || chunk==115 || chunk==120 || chunk==125 || chunk==130) p4+=value
    if ((chunk % 4)==0) p5+=value
  }
  END {
    if (!chunk_col || !total_col || !gpu_col) exit
    missing=0
    for (i=0; i<expected_chunks; i++) if (!(i in seen)) missing++
    printf "PARSE_OK=1\n"
    printf "ROWS=%.0f\n", rows
    printf "DUPLICATES=%.0f\n", duplicates
    printf "MISSING=%.0f\n", missing
    printf "P1=%.0f\n", p1
    printf "P2=%.0f\n", p2
    printf "P3=%.0f\n", p3
    printf "P4=%.0f\n", p4
    printf "P5=%.0f\n", p5
    printf "FULL=%.0f\n", full
    printf "LAST_GPU=%.0f\n", last_gpu
  }
' "$ARCHIVED_PROGRESS" > "$METRICS"

# metrics.env contains only fixed KEY=integer lines emitted by the awk block above.
# shellcheck disable=SC1090
source "$METRICS"

failures=0
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

FINAL_LINE=$(grep -E "^[[:space:]]*${N}:" "$RUN_LOG" | tail -n 1 || true)
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
  ELAPSED_SECONDS=$(awk -F: '{printf "%.3f", ($1*3600)+($2*60)+$3}' <<< "$ELAPSED_TEXT")
  PERF155=$(awk -v base="$BASELINE_155_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF154=$(awk -v base="$BASELINE_154_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF153=$(awk -v base="$BASELINE_153_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF152=$(awk -v base="$BASELINE_152_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF151=$(awk -v base="$BASELINE_151_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF150=$(awk -v base="$BASELINE_150_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF149=$(awk -v base="$BASELINE_149_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF148=$(awk -v base="$BASELINE_148_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  printf 'timing_vs_155\t155_full=%ss\t%s\tINFO\n' "$BASELINE_155_SECONDS" "$PERF155" >> "$SUMMARY"
  printf 'timing_vs_154\t154_full=%ss\t%s\tINFO\n' "$BASELINE_154_SECONDS" "$PERF154" >> "$SUMMARY"
  printf 'timing_vs_153\t153_full=%ss\t%s\tINFO\n' "$BASELINE_153_SECONDS" "$PERF153" >> "$SUMMARY"
  printf 'timing_vs_152\t152_full=%ss\t%s\tINFO\n' "$BASELINE_152_SECONDS" "$PERF152" >> "$SUMMARY"
  printf 'timing_vs_151\t151_full=%ss\t%s\tINFO\n' "$BASELINE_151_SECONDS" "$PERF151" >> "$SUMMARY"
  printf 'timing_vs_150\t150_full=%ss\t%s\tINFO\n' "$BASELINE_150_SECONDS" "$PERF150" >> "$SUMMARY"
  printf 'timing_vs_149\t149_full=%ss\t%s\tINFO\n' "$BASELINE_149_SECONDS" "$PERF149" >> "$SUMMARY"
  printf 'timing_vs_148\t148_mean=%ss\t%s\tINFO\n' "$BASELINE_148_SECONDS" "$PERF148" >> "$SUMMARY"
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

echo "[validation-ok] 156 one full run reproduced cases 01-06"
