#!/usr/bin/env bash
set -Eeuo pipefail


# N22
# N=22 GPU_COUNT=4 GPU_IDS=0,1,2,3 \
# bash 236Py_a10g4_multigpu_broadmarktail_worker.sh

# N27
# N=27 GPU_COUNT=4 GPU_IDS=0,1,2,3 \
# bash 236Py_a10g4_multigpu_broadmarktail_worker.sh

# =============================================================================
# 236 A10G multi-GPU launch helper
# Builds/reuses the 236 general runner, builds broadmarktail reorder bin once,
# then launches worker_id/worker_count split jobs on CUDA_VISIBLE_DEVICES.
# Defaults reproduce the historic N22 4xA10G worker pattern; set N=21..27/GPU_IDS/GPU_COUNT as needed.
# =============================================================================

SRC=${SRC:-./236Py_restore232_general_cleanup_keepfeatures_probe.py}
CAND=${CAND:-./236Py_restore232_general_cleanup_keepfeatures_probe}
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
LOGDIR=${LOGDIR:-${LOG_ROOT%/}/236Py_multigpu_N${N}_${GPU_COUNT}gpu_${TS}}
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

BUILD_CACHE_CMD=("$CAND" -g "$N" "$N" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS" 28 "$WINDOW" "$PHASE" "$CROSS_STRIPE_SAFE" "$BROADMARK_VARIANT")
printf '[cache-build-command]'; printf ' %q' "${BUILD_CACHE_CMD[@]}"; echo | tee "$LOGDIR/cache_build.log"
"${BUILD_CACHE_CMD[@]}" 2>&1 | tee -a "$LOGDIR/cache_build.log"

IFS=',' read -r -a IDS <<< "$GPU_IDS"
if (( ${#IDS[@]} != GPU_COUNT )); then
  echo "[warning] GPU_IDS count=${#IDS[@]} differs from GPU_COUNT=$GPU_COUNT; worker_count stays $GPU_COUNT" | tee -a "$LOGDIR/launch.log"
fi

echo "[launch] N=$N gpu_count=$GPU_COUNT gpu_ids=$GPU_IDS block=$BLOCK max_blocks=$MAX_BLOCKS preset=$PRESET_QUEENS mode=29 window=$WINDOW phase=$PHASE variant=$BROADMARK_VARIANT" | tee "$LOGDIR/launch.log"

pids=()
idx=0
for gid in "${IDS[@]}"; do
  worker_log="$LOGDIR/236Py_N${N}_worker${idx}of${GPU_COUNT}_gpu${gid}.log"
  CMD=("$CAND" -g "$N" "$N" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS" 29 "$WINDOW" "$PHASE" "$CROSS_STRIPE_SAFE" "$idx" "$GPU_COUNT" "$BROADMARK_VARIANT")
  { printf '[worker-command] CUDA_VISIBLE_DEVICES=%q' "$gid"; printf ' %q' "${CMD[@]}"; echo; } | tee -a "$LOGDIR/launch.log"
  CUDA_VISIBLE_DEVICES="$gid" stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "$worker_log" &
  pids+=("$!")
  idx=$((idx+1))
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then rc=1; fi
done

if [[ -f ./236Py_sum_worker_totals.py ]]; then
  python3 ./236Py_sum_worker_totals.py "$LOGDIR"/236Py_N${N}_worker*of${GPU_COUNT}_gpu*.log | tee "$LOGDIR/sum_workers.log" || rc=1
else
  echo "[info] sum helper not found: ./236Py_sum_worker_totals.py" | tee -a "$LOGDIR/launch.log"
fi

echo "[done] logdir=$LOGDIR rc=$rc"
exit "$rc"
