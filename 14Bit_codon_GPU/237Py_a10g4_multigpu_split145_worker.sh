#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# 237 A10G multi-GPU split145 launch helper
# Builds/reuses the 237 general runner, builds broadmarktail reorder bin once,
# builds chunkshape148/split145 shaped bin once, then launches worker_id/worker_count
# split jobs on CUDA_VISIBLE_DEVICES.  Defaults mirror the historic 4xA10G pattern;
# set N=21..27/GPU_IDS/GPU_COUNT as needed.
# =============================================================================

SRC=${SRC:-./237Py_restore232_fastdefault_keepfeatures_probe.py}
CAND=${CAND:-./237Py_restore232_fastdefault_keepfeatures_probe}
AUTO_BUILD=${AUTO_BUILD:-1}
N=${N:-22}
BLOCK=${BLOCK:-32}
MAX_BLOCKS=${MAX_BLOCKS:-484}
LOG_LEVEL=${LOG_LEVEL:-1}
SORT_MODE=${SORT_MODE:-0}
PRESET_QUEENS=${PRESET_QUEENS:-7}
WINDOW=${WINDOW:-8}
PHASE=${PHASE:-7}
CROSS_STRIPE_SAFE=${CROSS_STRIPE_SAFE:-0}
BROADMARK_VARIANT=${BROADMARK_VARIANT:-2}
GPU_IDS=${GPU_IDS:-0,1,2,3}
GPU_COUNT=${GPU_COUNT:-4}
LOG_ROOT=${LOG_ROOT:-.}
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR=${LOGDIR:-${LOG_ROOT%/}/237Py_multigpu_split145_N${N}_${GPU_COUNT}gpu_${TS}}
mkdir -p "$LOGDIR"

if [[ ! -f "$SRC" ]]; then echo "[error] source not found: $SRC" >&2; exit 66; fi
if (( N < 21 || N > 27 )); then echo "[error] this helper is intended for GPU stream/reorder N=21..27; N=$N" >&2; exit 64; fi

need_build=0
if [[ ! -x "$CAND" ]]; then need_build=1; elif [[ "$SRC" -nt "$CAND" ]]; then need_build=1; fi
if (( need_build )); then
  if [[ "$AUTO_BUILD" != "1" ]]; then echo "[error] candidate missing/stale and AUTO_BUILD=$AUTO_BUILD: $CAND" >&2; exit 66; fi
  if ! command -v codon >/dev/null 2>&1; then echo "[error] codon not found" >&2; exit 69; fi
  echo "[build] codon build -release $SRC" | tee "$LOGDIR/build.log"
  codon build -release "$SRC" 2>&1 | tee -a "$LOGDIR/build.log"
else
  echo "[build] reuse executable: $CAND" | tee "$LOGDIR/build.log"
fi

# Step 1: build/reuse broadmarktail reorder bin once. This avoids 4 workers racing on the same .bin.
BUILD_BROAD_CMD=("$CAND" -g "$N" "$N" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS" 28 "$WINDOW" "$PHASE" "$CROSS_STRIPE_SAFE" "$BROADMARK_VARIANT")
printf '[broad-cache-build-command]'; printf ' %q' "${BUILD_BROAD_CMD[@]}"; echo | tee "$LOGDIR/cache_build.log"
"${BUILD_BROAD_CMD[@]}" 2>&1 | tee -a "$LOGDIR/cache_build.log"

# Step 2: build/reuse chunkshape148 shaped bin once. mode30 executes only one probe chunk by default,
# but its important role here is to materialize the shaped bin before parallel workers start.
BUILD_SHAPED_CMD=("$CAND" -g "$N" "$N" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS" 30 "$WINDOW" "$PHASE" "$CROSS_STRIPE_SAFE" 0 1 "" "$BROADMARK_VARIANT")
printf '[shaped-cache-build-command]'; printf ' %q' "${BUILD_SHAPED_CMD[@]}"; echo | tee -a "$LOGDIR/cache_build.log"
"${BUILD_SHAPED_CMD[@]}" 2>&1 | tee -a "$LOGDIR/cache_build.log"

IFS=',' read -r -a IDS <<< "$GPU_IDS"
if (( ${#IDS[@]} != GPU_COUNT )); then
  echo "[warning] GPU_IDS count=${#IDS[@]} differs from GPU_COUNT=$GPU_COUNT; worker_count stays $GPU_COUNT" | tee -a "$LOGDIR/launch.log"
fi

echo "[launch] N=$N gpu_count=$GPU_COUNT gpu_ids=$GPU_IDS block=$BLOCK max_blocks=$MAX_BLOCKS preset=$PRESET_QUEENS mode=31 split145 window=$WINDOW phase=$PHASE variant=$BROADMARK_VARIANT" | tee "$LOGDIR/launch.log"

pids=()
idx=0
for gid in "${IDS[@]}"; do
  worker_log="$LOGDIR/237Py_N${N}_worker${idx}of${GPU_COUNT}_gpu${gid}.log"
  CMD=("$CAND" -g "$N" "$N" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS" 31 "$WINDOW" "$PHASE" "$CROSS_STRIPE_SAFE" "$idx" "$GPU_COUNT" "$BROADMARK_VARIANT")
  { printf '[worker-command] CUDA_VISIBLE_DEVICES=%q' "$gid"; printf ' %q' "${CMD[@]}"; echo; } | tee -a "$LOGDIR/launch.log"
  CUDA_VISIBLE_DEVICES="$gid" stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "$worker_log" &
  pids+=("$!")
  idx=$((idx+1))
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then rc=1; fi
done

if [[ -f ./237Py_sum_worker_totals.py ]]; then
  python3 ./237Py_sum_worker_totals.py "$LOGDIR"/237Py_N${N}_worker*of${GPU_COUNT}_gpu*.log | tee "$LOGDIR/sum_workers.log" || rc=1
elif [[ -f ./236Py_sum_worker_totals.py ]]; then
  python3 ./236Py_sum_worker_totals.py "$LOGDIR"/237Py_N${N}_worker*of${GPU_COUNT}_gpu*.log | tee "$LOGDIR/sum_workers.log" || rc=1
else
  echo "[info] sum helper not found: ./237Py_sum_worker_totals.py" | tee -a "$LOGDIR/launch.log"
fi

echo "[done] logdir=$LOGDIR rc=$rc"
exit "$rc"
