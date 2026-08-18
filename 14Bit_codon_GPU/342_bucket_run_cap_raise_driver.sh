#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# 342_bucket_run_cap_raise_driver.sh
#
# Standalone driver for the 341 CHUNKSHAPE148_BUCKET_RUN sweep. It calls the
# ordinary single-point harness once per sweep point with a different
# CHUNKSHAPE148_BUCKET_RUN and aggregates the results into one table. It
# reimplements no validation logic of its own.
#
# ---------------------------------------------------------------------------
# FIXES CARRIED IN FROM THE 340 DRIVER
#   340's driver executed all nine passes correctly, but its table came out
#   with every field "unknown". Two defects:
#     1. The harness prints "[logdir]   <path>" with THREE spaces; the sed
#        stripped only one, so the captured path kept leading whitespace and
#        every subsequent [[ -f "$logdir/summary.tsv" ]] test failed.
#     2. rc=${PIPESTATUS[0]} was read after out=$(... | tee ...), i.e. after
#        a command substitution rather than after a pipeline, so it never
#        reflected the harness exit code.
#   341 does not rely on that capture at all. It records a marker before the
#   sweep starts and builds the final table by scanning every harness log
#   directory newer than the marker, reading each one's summary.tsv and
#   progress_full.tsv directly. That path cannot be broken by output
#   formatting. AGGREGATE_ONLY=1 re-runs just the aggregation, so a table can
#   always be rebuilt after the fact without repeating any GPU work.
# ---------------------------------------------------------------------------
#
# WHY THIS SWEEP
#   340 settled the mechanism: RUN=48 is LARGER than 32 yet regressed to
#   448.995s, level with RUN=16's 448.931s, so what matters is that the run
#   length is a MULTIPLE OF THE WARP WIDTH -- bucket boundaries aligning with
#   warp boundaries -- not run length itself. One residual is unexplained:
#   RUN=64 (446.390s) is a further 0.176% faster than RUN=32 (447.177s avg),
#   about 11x the 0.016% session drift, even though every multiple of 32
#   should give identical intra-warp homogeneity. 341 extends upward through
#   the multiples and re-tests alignment at a larger scale:
#     RUN=80   2.5x -- NON-MULTIPLE CONTROL; must regress if alignment governs
#     RUN=96   3x
#     RUN=128  4x
#     RUN=256  8x -- the CHUNKSHAPE148_BUCKET_RUN_MAX ceiling in the source
#   RUN=64 anchors the sweep first and last so drift stays visible.
#
# TWO-PASS PROTOCOL (automatic): a value with no cached shaped bin needs one
# run to build the cache (its elapsed includes the rebuild and is discarded)
# and one to time. If the cache file already exists, only the timed run
# happens. The codon binary is built once, on the first invocation.
#
# CORRECTNESS IS CHECKED AT EVERY POINT (reordering is a permutation, so
# 314666222712 must hold everywhere). A failing point is recorded and the
# sweep continues; the driver exits non-zero at the end if any point failed.
#
# COST: about 10 timed runs (~75 min) plus 4 shaped-bin rebuilds, and roughly
# 32MB of cache per new value.
#
# USAGE
#   bash 342_bucket_run_cap_raise_driver.sh
#   SWEEP="64 96 256 64" bash 342_bucket_run_cap_raise_driver.sh   # quick version
#   DRY_RUN=1        bash 342_bucket_run_cap_raise_driver.sh       # plan only
#   AGGREGATE_ONLY=1 SWEEPDIR=./342_bucket_run_cap_raise_<ts> \
#                    bash 342_bucket_run_cap_raise_driver.sh       # rebuild table
# =============================================================================

HARNESS=${HARNESS:-./342Py_bucket_run_cap_raise_validate_N21_full_once.sh}
SRC=${SRC:-./342Py_bucket_run_cap_raise.py}
LOGPREFIX=${LOGPREFIX:-./342Py_bucket_run_cap_raise_logs_N21_full_once_}
SWEEP=${SWEEP:-"256 512 1024 2048 384 256"}
DRY_RUN=${DRY_RUN:-0}
AGGREGATE_ONLY=${AGGREGATE_ONLY:-0}
N=${N:-21}
PRESET_EFFECTIVE=${PRESET_EFFECTIVE:-6}
BLOCK=${BLOCK:-32}
MAX_BLOCKS=${MAX_BLOCKS:-484}
FULL_TOTAL=${FULL_TOTAL:-314666222712}

# 340 in-session reference points, used only for the delta columns
REF_RUN1=${REF_RUN1:-450.183}
REF_RUN64=${REF_RUN64:-439.835}  # 342: RUN=256 reference

TS=$(date -u +%Y%m%d_%H%M%S)
SWEEPDIR=${SWEEPDIR:-./342_bucket_run_cap_raise_${TS}}
mkdir -p "$SWEEPDIR"
MARKER="$SWEEPDIR/.sweep_start_marker"
TABLE="$SWEEPDIR/sweep_table.tsv"
PLAN="$SWEEPDIR/plan.txt"

shaped_bin_for() {
  local r=$1 suffix=""
  [[ "$r" != "1" ]] && suffix="_run${r}"
  printf 'constellations_N%s_%s_chunkshape148_scorestripe_v9_lanephase32_octetfirstpairlock29_v4_rotate_only_w3_j7_b%s_m%s_s%s%s.bin' \
    "$N" "$PRESET_EFFECTIVE" "$BLOCK" "$MAX_BLOCKS" "$((BLOCK * MAX_BLOCKS))" "$suffix"
}

# ---------------------------------------------------------------------------
# aggregation: read the log directories directly. Independent of any stdout
# capture, so it works even if the harness changes its banner formatting.
# ---------------------------------------------------------------------------
aggregate() {
  printf 'seq\tbucket_run\tcache_state\tcorrectness\tfails\telapsed_s\tvs_run1_pct\tvs_run64_pct\tchunk0_ms\tchunk1_ms\tchunk2_ms\tstatus\tlogdir\n' > "$TABLE"
  local seq=0 d br cs ok fails el c0 c1 c2 status d1 d2
  while IFS= read -r d; do
    [[ -f "$d/summary.tsv" ]] || continue
    seq=$((seq + 1))
    br=$(awk -F'\t' '$1=="runtime_chunkshape148_bucket_run"{print $3}' "$d/summary.tsv")
    cs=$(awk -F'\t' '$1=="chunkshape148_cache_state"{print $3}' "$d/summary.tsv")
    fails=$(awk -F'\t' '$4=="FAIL"' "$d/summary.tsv" | wc -l | tr -d ' ')
    if awk -F'\t' -v t="$FULL_TOTAL" '$1=="full_chunk_sum" && $2==t && $3==t && $4=="OK"{f=1} END{exit !f}' "$d/summary.tsv"; then
      ok=$FULL_TOTAL
    else
      ok=MISMATCH
    fi
    el=$(awk -F'\t' '$1 ~ /^timing_vs_/ {print $3; exit}' "$d/summary.tsv" \
         | sed -n 's/.*elapsed=\([0-9.]*\).*/\1/p')
    c0=$(awk -F'\t' 'NR==2{print $11}' "$d/progress_full.tsv" 2>/dev/null || true)
    c1=$(awk -F'\t' 'NR==3{print $11}' "$d/progress_full.tsv" 2>/dev/null || true)
    c2=$(awk -F'\t' 'NR==4{print $11}' "$d/progress_full.tsv" 2>/dev/null || true)

    status=OK
    [[ "$ok" == MISMATCH ]] && status=FAIL
    [[ "$fails" != "0" ]] && status=FAIL
    [[ "$cs" != "reuse" ]] && status="${status}/CACHE-BUILD-NOT-COMPARABLE"

    d1="-"; d2="-"
    if [[ -n "$el" ]]; then
      d1=$(awk -v e="$el" -v b="$REF_RUN1"  'BEGIN{printf "%+.3f", (e-b)/b*100}')
      d2=$(awk -v e="$el" -v b="$REF_RUN64" 'BEGIN{printf "%+.3f", (e-b)/b*100}')
    fi
    printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$seq" "${br:-?}" "${cs:-?}" "$ok" "$fails" "${el:--}" "$d1" "$d2" \
      "${c0:--}" "${c1:--}" "${c2:--}" "$status" "$d" >> "$TABLE"
  done < <(find . -maxdepth 1 -type d -name "$(basename "$LOGPREFIX")*" \
             ${MARKER_EXISTS:+-newer "$MARKER"} | sort)
}

if [[ "$AGGREGATE_ONLY" == "1" ]]; then
  echo "[aggregate-only] rebuilding table from log directories under $(pwd)"
  aggregate
  echo "[sweep-table] $TABLE"
  { column -t -s $'\t' "$TABLE" 2>/dev/null || cat "$TABLE"; }
  exit 0
fi

echo "[start] 342 bucket-run cap-raise driver"
echo "[info]  harness=$HARNESS src=$SRC"
echo "[info]  sweep=$SWEEP"
echo "[info]  sweepdir=$SWEEPDIR"
[[ -f "$HARNESS" ]] || { echo "[error] harness not found: $HARNESS" >&2; exit 66; }
[[ -f "$SRC" ]]     || { echo "[error] source not found: $SRC"   >&2; exit 66; }

{
  echo "sweep points : $SWEEP"
  for r in $SWEEP; do
    b=$(shaped_bin_for "$r")
    if [[ -f "$b" ]]; then echo "  RUN=$r  cache PRESENT -> 1 timed run"
    else                   echo "  RUN=$r  cache ABSENT  -> build pass + timed run"; fi
    echo "        $b"
  done
} | tee "$PLAN"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] plan written to $PLAN; nothing executed."
  exit 0
fi

touch "$MARKER"
export MARKER_EXISTS=1
sleep 1   # ensure log dirs are strictly newer than the marker

first=1
order=0
hard_fail=0
for r in $SWEEP; do
  order=$((order + 1))
  echo "================================================================"
  echo "[sweep $order] CHUNKSHAPE148_BUCKET_RUN=$r"
  bin=$(shaped_bin_for "$r")
  force=0
  if (( first )); then force=1; first=0; fi

  if [[ ! -f "$bin" ]]; then
    echo "[sweep $order] cache absent -> build pass (timing discarded)"
    set +e
    CHUNKSHAPE148_BUCKET_RUN="$r" FORCE_REBUILD="$force" \
      bash "$HARNESS" 2>&1 | tee "$SWEEPDIR/run_${order}_run${r}_build.log"
    rc=${PIPESTATUS[0]}
    set -e
    force=0
    if (( rc != 0 )); then
      echo "[sweep $order] build pass FAILED rc=$rc" >&2
      hard_fail=$((hard_fail + 1))
      continue
    fi
  else
    echo "[sweep $order] cache present -> single timed run"
  fi

  set +e
  CHUNKSHAPE148_BUCKET_RUN="$r" FORCE_REBUILD="$force" \
    bash "$HARNESS" 2>&1 | tee "$SWEEPDIR/run_${order}_run${r}_timed.log"
  rc=${PIPESTATUS[0]}
  set -e
  (( rc != 0 )) && { echo "[sweep $order] timed run FAILED rc=$rc" >&2; hard_fail=$((hard_fail + 1)); }
done

echo "================================================================"
aggregate
echo "[sweep-table] $TABLE"
{ column -t -s $'\t' "$TABLE" 2>/dev/null || cat "$TABLE"; }
echo "================================================================"
echo "[how to read it]"
echo "  Every harness run of this session appears, build passes included."
echo "  Only rows with cache_state=reuse are comparable; build rows are the"
echo "  discarded cache-construction passes and will look slow."
echo "  vs_run1_pct  : negative = faster than 340's in-session RUN=1 ($REF_RUN1 s)"
echo "  vs_run64_pct : negative = faster than 340's RUN=64 ($REF_RUN64 s)"
echo "  The first and last RUN=64 rows bracket the session; their difference"
echo "  is this session's drift (340 measured 0.016% between its two anchors)."
echo
echo "[what the model predicts]"
echo "  RUN=80 (not a multiple of 32) must REGRESS if alignment still governs."
echo "  RUN=96/128/256 should be at least as good as 64. If they keep improving"
echo "  monotonically, the residual is not alignment but something that grows"
echo "  with run length, and 342 should raise CHUNKSHAPE148_BUCKET_RUN_MAX"
echo "  (currently 256) rather than adopt a value straight away."

if (( hard_fail > 0 )); then
  echo "[result] $hard_fail harness invocation(s) exited non-zero -- see $TABLE and the run_*.log files" >&2
  exit 1
fi
if awk -F'\t' 'NR>1 && $12 ~ /FAIL/ {f=1} END{exit !f}' "$TABLE"; then
  echo "[result] some sweep points are FAIL or not comparable -- see $TABLE" >&2
  exit 1
fi
echo "[result] OK -- all sweep points completed, correctness $FULL_TOTAL at every point"
