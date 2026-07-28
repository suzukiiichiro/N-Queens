#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# 340_bucket_run_sweep_driver.sh
#
# Standalone driver for the 340 CHUNKSHAPE148_BUCKET_RUN sweep. It does NOT
# reimplement any validation: it calls the ordinary single-point harness
# (340Py_bucket_run_sweep_validate_N21_full_once.sh) once per sweep point
# with a different CHUNKSHAPE148_BUCKET_RUN, and aggregates the resulting
# summary.tsv files into one table. Same separation of concerns as 337's
# standalone bin-format check: the validate harness stays untouched.
#
# WHY A SWEEP AT ALL, AND WHY THESE POINTS
#   339 measured CHUNKSHAPE148_BUCKET_RUN=32 at 447.116s against 338's
#   450.329s (-0.713%, about 4.8x the entire 333-338 noise band, with
#   chunkshape148_cache_state=reuse so the timing is comparable). The open
#   question is not "is more better" but "why does 32 work". Three
#   discriminating points answer that:
#
#     RUN=16  below warp width -- one warp-iteration straddles two score
#             buckets. If the intra-warp model is right this is WORSE than 32.
#     RUN=48  LARGER than 32 but not a multiple of it, so warp boundaries and
#             bucket boundaries misalign. THIS IS THE DECISIVE POINT: if a
#             larger run length still regresses, the mechanism is warp-width
#             ALIGNMENT rather than simply longer runs.
#     RUN=64  a multiple of 32 -- intra-warp homogeneity is preserved while
#             inter-warp mixing coarsens. Model predicts roughly equal to 32.
#
#   RUN=1 (the 276-338 control) and RUN=32 (the 339 anchor) are re-measured
#   in this same session so the comparison never depends on cross-session
#   stability, and RUN=32 is measured FIRST and LAST so session drift is
#   itself visible in the table.
#
# TWO-PASS PROTOCOL (handled automatically)
#   A BUCKET_RUN value with no cached shaped bin forces a rebuild, and that
#   rebuild lands inside the N=21 elapsed line. So for each value this driver
#   runs the harness twice -- once to build the cache, once to time it -- and
#   only the second (cache_state=reuse) run is recorded as the timing. If the
#   cache file for a value already exists on disk, the build pass is skipped
#   automatically, so RUN=1 and RUN=32 normally cost a single run each.
#
# CORRECTNESS IS CHECKED AT EVERY POINT. Reordering is a permutation, so
# every sweep point must still produce 314666222712. A point that fails
# correctness is recorded as FAIL and the sweep continues, so one bad point
# does not throw away the rest of the run; the driver exits non-zero at the
# end if any point failed.
#
# COST: roughly 6 timed N=21 runs (~45 min) plus 3 shaped-bin rebuilds for
# the new values. Each new value also leaves a ~32MB cached .bin on disk.
#
# USAGE
#   bash 340_bucket_run_sweep_driver.sh                  # default sweep
#   SWEEP="1 32 16 48 64 32" bash 340_bucket_run_sweep_driver.sh
#   SWEEP="16 64" bash 340_bucket_run_sweep_driver.sh    # trim if short on time
#   DRY_RUN=1 bash 340_bucket_run_sweep_driver.sh        # print the plan only
# =============================================================================

HARNESS=${HARNESS:-./340Py_bucket_run_sweep_validate_N21_full_once.sh}
SRC=${SRC:-./340Py_bucket_run_sweep.py}
SWEEP=${SWEEP:-"1 32 16 48 64 32"}
DRY_RUN=${DRY_RUN:-0}
N=${N:-21}
PRESET_QUEENS=${PRESET_QUEENS:-7}
# the filename carries the EFFECTIVE preset the solver resolved for N=21,
# which the 339 log shows as 6 (summary row dynamic_preset_N21). This is only
# used to guess whether a cache already exists; guessing wrong merely costs one
# extra build pass, it cannot corrupt a measurement.
PRESET_EFFECTIVE=${PRESET_EFFECTIVE:-6}
BLOCK=${BLOCK:-32}
MAX_BLOCKS=${MAX_BLOCKS:-484}
FULL_TOTAL=${FULL_TOTAL:-314666222712}

# reference points, for the delta column only
BASELINE_339_RUN32=${BASELINE_339_RUN32:-447.116}
BASELINE_338_RUN1=${BASELINE_338_RUN1:-450.329}

TS=$(date -u +%Y%m%d_%H%M%S)
SWEEPDIR="./340_bucket_run_sweep_${TS}"
mkdir -p "$SWEEPDIR"
TABLE="$SWEEPDIR/sweep_table.tsv"
PLAN="$SWEEPDIR/plan.txt"
printf 'order\tbucket_run\tpasses\tcache_state\tcorrectness\telapsed_s\tvs_run1_pct\tvs_339run32_pct\tstatus\tlogdir\n' > "$TABLE"

echo "[start] 340 bucket-run sweep driver"
echo "[info]  harness=$HARNESS src=$SRC"
echo "[info]  sweep=$SWEEP"
echo "[info]  sweepdir=$SWEEPDIR"

if [[ ! -f "$HARNESS" ]]; then echo "[error] harness not found: $HARNESS" >&2; exit 66; fi
if [[ ! -f "$SRC" ]];     then echo "[error] source not found: $SRC"   >&2; exit 66; fi

# ---- shaped-bin filename for a given run value (mirrors ----
# ---- chunkshape148_reorder_output_fname + chunkshape148_bucket_run_tag) ----
shaped_bin_for() {
  local r=$1 suffix=""
  if [[ "$r" != "1" ]]; then suffix="_run${r}"; fi
  printf 'constellations_N%s_%s_chunkshape148_scorestripe_v9_lanephase32_octetfirstpairlock29_v4_rotate_only_w3_j7_b%s_m%s_s%s%s.bin' \
    "$N" "$PRESET_EFFECTIVE" "$BLOCK" "$MAX_BLOCKS" "$((BLOCK * MAX_BLOCKS))" "$suffix"
}

# ---- plan ----
{
  echo "sweep points : $SWEEP"
  echo "harness      : $HARNESS"
  for r in $SWEEP; do
    b=$(shaped_bin_for "$r")
    if [[ -f "$b" ]]; then echo "  RUN=$r  cache PRESENT -> 1 timed run        ($b)"
    else                   echo "  RUN=$r  cache ABSENT  -> build pass + timed ($b)"; fi
  done
} | tee "$PLAN"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] plan written to $PLAN; nothing executed."
  exit 0
fi

# ---- one harness invocation; echoes the logdir it produced ----
run_once() {
  local r=$1 label=$2 force=$3
  local out rc logdir
  set +e
  out=$(CHUNKSHAPE148_BUCKET_RUN="$r" FORCE_REBUILD="$force" \
        bash "$HARNESS" 2>&1 | tee "$SWEEPDIR/run_${label}.log")
  rc=${PIPESTATUS[0]}
  set -e
  logdir=$(grep -oE '\[logdir\] .*' <<< "$out" | tail -n1 | sed 's/^\[logdir\] //')
  printf '%s\t%s\n' "$rc" "$logdir"
}

first=1
order=0
failed=0
for r in $SWEEP; do
  order=$((order + 1))
  echo "================================================================"
  echo "[sweep $order] CHUNKSHAPE148_BUCKET_RUN=$r"
  bin=$(shaped_bin_for "$r")
  passes=1
  force=0
  if (( first )); then force=1; first=0; fi   # build the codon binary once

  if [[ ! -f "$bin" ]]; then
    echo "[sweep $order] cache absent -> build pass first (its timing is discarded)"
    res=$(run_once "$r" "${order}_run${r}_build" "$force")
    force=0
    passes=2
    if [[ "${res%%	*}" != "0" ]]; then
      echo "[sweep $order] build pass FAILED" >&2
      printf '%d\t%s\t%d\tbuild-pass-failed\tunknown\t-\t-\t-\tFAIL\t%s\n' \
        "$order" "$r" "$passes" "${res#*	}" >> "$TABLE"
      failed=$((failed + 1))
      continue
    fi
  else
    echo "[sweep $order] cache present -> single timed run"
  fi

  res=$(run_once "$r" "${order}_run${r}_timed" "$force")
  rc=${res%%	*}
  logdir=${res#*	}

  status=OK
  correctness=unknown
  cache_state=unknown
  elapsed=-
  if [[ -n "$logdir" && -f "$logdir/summary.tsv" ]]; then
    if awk -F'\t' -v t="$FULL_TOTAL" '$1=="full_chunk_sum" && $2==t && $3==t && $4=="OK"{f=1} END{exit !f}' "$logdir/summary.tsv"; then
      correctness=$FULL_TOTAL
    else
      correctness=MISMATCH; status=FAIL
    fi
    cache_state=$(awk -F'\t' '$1=="chunkshape148_cache_state"{print $3}' "$logdir/summary.tsv")
    elapsed=$(awk -F'\t' '$1 ~ /^timing_vs_/ {print $3; exit}' "$logdir/summary.tsv" \
              | sed -n 's/.*elapsed=\([0-9.]*\).*/\1/p')
    if awk -F'\t' '$4=="FAIL"{f=1} END{exit !f}' "$logdir/summary.tsv"; then status=FAIL; fi
  else
    status=FAIL
  fi
  (( rc != 0 )) && status=FAIL
  [[ "$cache_state" != "reuse" ]] && status="${status}/NOT-COMPARABLE"
  [[ "$status" == FAIL* ]] && failed=$((failed + 1))

  d1="-"; d2="-"
  if [[ -n "$elapsed" && "$elapsed" != "-" ]]; then
    d1=$(awk -v e="$elapsed" -v b="$BASELINE_338_RUN1"   'BEGIN{printf "%+.3f", (e-b)/b*100}')
    d2=$(awk -v e="$elapsed" -v b="$BASELINE_339_RUN32" 'BEGIN{printf "%+.3f", (e-b)/b*100}')
  fi

  printf '%d\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$order" "$r" "$passes" "${cache_state:-unknown}" "$correctness" "${elapsed:--}" \
    "$d1" "$d2" "$status" "$logdir" >> "$TABLE"

  echo "[sweep $order] RUN=$r elapsed=${elapsed:--}s cache=${cache_state:-unknown} status=$status"
done

echo "================================================================"
echo "[sweep-table] $TABLE"
column -t -s $'\t' "$TABLE" 2>/dev/null || cat "$TABLE"
echo "================================================================"
echo "[how to read it]"
echo "  vs_run1_pct     : negative = faster than the 276-338 order (338 = ${BASELINE_338_RUN1}s)"
echo "  vs_339run32_pct : negative = faster than 339's RUN=32 (${BASELINE_339_RUN32}s)"
echo "  order 2 and order 6 are BOTH RUN=32; their difference is this session's drift."
echo "  Only rows with cache_state=reuse are comparable."
echo
echo "[what the model predicts]"
echo "  RUN=16 worse than 32; RUN=48 worse than 32 DESPITE being larger; RUN=64 ~ equal to 32."
echo "  If RUN=48 lands between 32 and 64 instead, the mechanism is run LENGTH, not warp alignment,"
echo "  and 341 should sweep upward (96/128) rather than refine bucket granularity."

if (( failed > 0 )); then
  echo "[result] $failed sweep point(s) FAILED or were not comparable -- see $TABLE" >&2
  exit 1
fi
echo "[result] OK -- all sweep points completed, correctness $FULL_TOTAL at every point"
