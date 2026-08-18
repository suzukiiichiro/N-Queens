#!/usr/bin/env bash
# 114Py weekend ablation runner for N22 / A10G single GPU.
# Default matrix runs all six 114 variants sequentially.
# To run a smaller set, override VARIANTS, for example:
#   VARIANTS="0:v2base 1:phase_only 4:phase_rotate 5:wide_phase_rotate" bash 114py_weekend_run_commands.sh

set -u

SRC="114Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_broadmark_funcid17_gh_tail_ablation.py"
BIN="./114Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_broadmark_funcid17_gh_tail_ablation"

# id:tag
VARIANTS="${VARIANTS:-0:v2base 1:phase_only 2:rotate_only 3:wide_only 4:phase_rotate 5:wide_phase_rotate}"

# Fixed N22 baseline parameters.
NMIN=22
NMAX=22
BLOCK=32
MAX_BLOCKS=484
LOG_LEVEL=1
SORT_MODE=0
PRESET=7
MODE_BUILD=28
MODE_RUN=29
WINDOW=8
PHASE=7
CROSS_SAFE=0
WORKER_ID=0
WORKER_COUNT=1

printf '[114py-weekend] start: '; date
printf '[114py-weekend] host: '; hostname || true
uname -a || true
nvidia-smi || true

printf '\n[114py-weekend] build source\n'
codon build -release "$SRC" 2>&1 | tee 114Py_build.log

for item in $VARIANTS; do
  vid="${item%%:*}"
  tag="${item#*:}"
  prefix="114Py_N22_v4_${vid}_${tag}_32x484"

  printf '\n[114py-weekend] variant=%s tag=%s build reorder bin\n' "$vid" "$tag"
  "$BIN" \
    -g "$NMIN" "$NMAX" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET" "$MODE_BUILD" \
    "$WINDOW" "$PHASE" "$CROSS_SAFE" "$vid" \
    2>&1 | tee "${prefix}_sim_build.log"

  printf '\n[114py-weekend] variant=%s tag=%s full GPU run\n' "$vid" "$tag"
  "$BIN" \
    -g "$NMIN" "$NMAX" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET" "$MODE_RUN" \
    "$WINDOW" "$PHASE" "$CROSS_SAFE" "$WORKER_ID" "$WORKER_COUNT" "$vid" \
    2>&1 | tee "${prefix}_single_gpu.log"

  printf '[114py-weekend] variant=%s tag=%s done: ' "$vid" "$tag"; date
done

printf '\n[114py-weekend] collect summary\n'
python3 114py_collect_weekend_results.py 2>&1 | tee 114py_collect_weekend_results.log

printf '[114py-weekend] all done: '; date
