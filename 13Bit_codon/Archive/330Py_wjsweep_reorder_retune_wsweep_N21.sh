#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# 330 N=21 funcid_reorder W-sweep harness (wjsweep-reorder-retune)
# Base: 328/329 (adopted SoA baseline; kernel logic identical; 329's own
#   N=21 run reconfirmed 314666222712 / 456.187s, -0.033% vs 328's
#   456.036s, within noise).
#
# *** THIS REVISION CONTAINS NO KERNEL-LOGIC CHANGE AND NO SOURCE-LOGIC
#   CHANGE. *** The 330 .py differs from 329 only in the header docstring
#   / VERSION_TAG / reason string. This harness differs in PURPOSE from
#   the usual single-run validate script: it is a PARAMETER SWEEP over
#   the funcid_reorder window_mult (W) runtime argument.
#
# WHY: buried-idea C-3 from the 2026-07-23 archive survey
#   (330_buried_ideas_survey.md). The active w8_j7 was chosen in the
#   rev94-99 era, measured on N22 ("N22 measured best baseline w8_j7"),
#   and has never been re-validated across chunkshape148 (128-148),
#   nibble schedule (168), root-preroll (196-232), K-batching (292-308),
#   variant2 (311), and SoA (328). The rev95 sweep itself covered
#   w10/12/16/32 x j5/7/11 -- w8 was the inherited baseline, not a
#   sweep winner for the current pipeline or for N21.
#
# ALSO CLOSED IN 330 (documented in the .py header and README): the
#   rev292 open question "does Codon's @gpu.kernel support atomics" --
#   real-machine grep on cudacodon found no CUDA device atomics in
#   Codon's gpu module (the 2 hits in internal/gpu.codon are
#   internal.gc.atomic, a GC allocator type-trait, unrelated). 292's
#   plan-C-1 (persistent kernel + atomic work queue) is impossible in
#   pure Codon; the rev84 CUDA C runner is the remaining route.
#
# SWEEP DESIGN:
#   J=7 fixed. W in {4, 6, 8, 10, 12, 16} (override: SWEEP_W env var).
#   One full N=21 GPU run per W, sequential, same binary, built once.
#   W=8 doubles as the in-session control point (expected ~456s).
#   Correctness (314666222712) is checked for EVERY point; any mismatch
#   aborts the whole sweep immediately.
#   Metrics recorded per point:
#     - elapsed_line : the program's own final-line N=21 time
#     - kernel_ms    : sum of kernel_reduce_ms across the 3 chunk-end
#                      lines (GPU-pure time, immune to first-run
#                      reorder-bin generation cost)
#   DECISION RULE: adopt a new W only if kernel_ms clearly beats W=8
#   beyond session noise (reference noise band: 328 vs 329 = 0.03%,
#   327 vs 319 = 0.12%). Otherwise keep w8_j7 and close this axis
#   for N21.
#
# DISK NOTE: each new W value generates 2 new reorder bins
#   (broadmarktail + chunkshape), several hundred MB each. 5 new W
#   values => a few GB. Check free space first; bins are reusable
#   caches and can be deleted afterwards if space is tight.
#
# EXPECTED_CHUNKS=3, K=48, BLOCK=32, MAX_BLOCKS=484, VARIANT=2 --
#   all unchanged from the adopted baseline.
# =============================================================================

SRC=${SRC:-./330Py_wjsweep_reorder_retune.py}
CAND=${CAND:-./330Py_wjsweep_reorder_retune}
AUTO_BUILD=${AUTO_BUILD:-1}
FORCE_REBUILD=${FORCE_REBUILD:-1}
STATIC_ONLY=${STATIC_ONLY:-0}
LOG_ROOT=${LOG_ROOT:-.}
LOCK_FILE=${LOCK_FILE:-/tmp/330Py_wjsweep_reorder_retune_N21_wsweep.lock}
COOLDOWN_SECONDS=${COOLDOWN_SECONDS:-30}

N=${N:-21}
BLOCK=${BLOCK:-32}
MAX_BLOCKS=${MAX_BLOCKS:-484}
LOG_LEVEL=${LOG_LEVEL:-1}
SORT_MODE=${SORT_MODE:-0}
PRESET_QUEENS=${PRESET_QUEENS:-7}
BENCH_MODE=${BENCH_MODE:-31}
REORDER_PHASE_JUMP=${REORDER_PHASE_JUMP:-7}
CROSS_STRIPE_SAFE=${CROSS_STRIPE_SAFE:-0}
WORKER_ID=${WORKER_ID:-0}
WORKER_COUNT=${WORKER_COUNT:-1}
BROADMARK_VARIANT=${BROADMARK_VARIANT:-2}

SWEEP_W=${SWEEP_W:-"4 6 8 10 12 16"}

FULL_TOTAL=314666222712
EXPECTED_CHUNKS=3
BASELINE_W8_KNOWN_ELAPSED="456.036 (328) / 456.187 (329)"

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="${LOG_ROOT%/}/330Py_wjsweep_reorder_retune_logs_N21_wsweep_${TS}"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.tsv"
SWEEP_TSV="$LOGDIR/wsweep_results.tsv"

echo "[start] 330 wjsweep-reorder-retune W-sweep script"
echo "[sweep-config] J=$REORDER_PHASE_JUMP W_list={$SWEEP_W} N=$N block=$BLOCK max_blocks=$MAX_BLOCKS variant=$BROADMARK_VARIANT"
echo "[disk-check] free space in cwd:"
df -h . | tail -1

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[abort] another sweep/validation appears to be running (lock: $LOCK_FILE)"
  exit 66
fi

printf 'check\texpected\tactual\tstatus\n' > "$SUMMARY"

# ---- source static checks (light: version tag + SoA invariants) ----
static_failures=0
if grep -q '330 wjsweep-reorder-retune' "$SRC"; then
  printf 'source_version_tag\t330 wjsweep-reorder-retune\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_version_tag\t330 wjsweep-reorder-retune\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi

set +e
python3 - "$SRC" "$SUMMARY" << 'PYCHECK'
import re, sys
src, summary = sys.argv[1], sys.argv[2]
s = open(src, encoding='utf-8').read()
checks = []

soa_sig_count = 0
old_sig_count = 0
for kname in ['maxd14', 'maxd16', 'maxd18', 'maxd20', 'maxd21']:
    sig_m = re.search(r'^def\s+kernel_dfs_iter_gpu_' + kname + r'\(\n(.*?)\n\)->None:', s, re.M | re.S)
    if sig_m is None:
        continue
    sig_text = sig_m.group(1)
    if 'w_lo_arr:Ptr[u32],w_hi_arr:Ptr[u32]' in sig_text and 'w_arr:Ptr[u64]' not in sig_text:
        soa_sig_count += 1
    elif 'w_arr:Ptr[u64]' in sig_text:
        old_sig_count += 1
ok_soa = (soa_sig_count == 5) and (old_sig_count == 0)
checks.append(('source_warr_soa_split_signatures', '5 kernels use w_lo_arr/w_hi_arr, 0 old-style',
               f'{soa_sig_count} converted, {old_sig_count} old-style', ok_soa))

dispatcher_m = re.search(r'^def launch_kernel_dfs_iter_gpu_static_maxd\(.*?\n  return False', s, re.M | re.S)
ok_dispatcher = (dispatcher_m is not None
                 and 'w_lo_arr:List[u32]=' in dispatcher_m.group(0)
                 and 'w_hi_arr:List[u32]=' in dispatcher_m.group(0)
                 and dispatcher_m.group(0).count('gpu.raw(w_lo_arr),gpu.raw(w_hi_arr)') == 5
                 and 'gpu.raw(w_arr)' not in dispatcher_m.group(0))
checks.append(('source_warr_soa_split_dispatcher', 'derives and passes to all 5 launches',
               'present' if ok_dispatcher else 'missing or incomplete', ok_dispatcher))

ok_main = ('if __name__=="__main__"' in s and '  main()' in s)
checks.append(('source_main_entry', 'present', 'present' if ok_main else 'missing', ok_main))

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

if (( static_failures > 0 )); then
  echo "[abort] static checks failed ($static_failures); see $SUMMARY"
  exit 65
fi
echo "[static-ok] version tag + SoA invariants confirmed"
if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "[static-only] exiting before build/run as requested"
  exit 0
fi

# ---- build (once) ----
if [[ "$AUTO_BUILD" == "1" ]]; then
  if [[ "$FORCE_REBUILD" == "1" || ! -x "$CAND" || "$SRC" -nt "$CAND" ]]; then
    rm -f "$CAND"
    echo "[build] codon build -release $SRC"
    codon build -release "$SRC" 2>&1 | tee "$LOGDIR/build.log"
    if [[ ! -x "$CAND" ]]; then
      echo "[abort] build did not produce $CAND"
      printf 'build_exit\t0\tno-binary\tFAIL\n' >> "$SUMMARY"
      exit 64
    fi
  fi
fi
printf 'build_exit\t0\t0\tOK\n' >> "$SUMMARY"

# ---- pre-run GPU snapshot (clock cap status for the record) ----
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.limit,power.default_limit,temperature.gpu \
    --format=csv > "$LOGDIR/gpu_pre_run_snapshot.csv" 2>/dev/null || true
  cat "$LOGDIR/gpu_pre_run_snapshot.csv" || true
fi

# ---- sweep ----
printf 'W\tJ\ttotal\tcorrect\telapsed_line\telapsed_s\tkernel_ms\tchunks_seen\tlog\n' > "$SWEEP_TSV"
sweep_failures=0

for W in $SWEEP_W; do
  RUN_LOG="$LOGDIR/full_once_w${W}_j${REORDER_PHASE_JUMP}.log"
  echo "================================================================"
  echo "[sweep-run] W=$W J=$REORDER_PHASE_JUMP -> $RUN_LOG"
  echo "[sweep-run] known W=8 baseline elapsed: $BASELINE_W8_KNOWN_ELAPSED"
  CMD=("$CAND" -g "$N" "$N" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS" "$BENCH_MODE" "$W" "$REORDER_PHASE_JUMP" "$CROSS_STRIPE_SAFE" "$WORKER_ID" "$WORKER_COUNT" "$BROADMARK_VARIANT")
  echo "[sweep-cmd] ${CMD[*]}"
  set +e
  stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "$RUN_LOG"
  run_rc=${PIPESTATUS[0]}
  set -e
  if (( run_rc != 0 )); then
    echo "[sweep-abort] W=$W run exit code $run_rc"
    printf '%s\t%s\t-\tRUN_FAIL\t-\t-\t-\t-\t%s\n' "$W" "$REORDER_PHASE_JUMP" "$RUN_LOG" >> "$SWEEP_TSV"
    sweep_failures=$((sweep_failures+1))
    break
  fi

  # correctness: final line must contain FULL_TOTAL and 'ok'
  FINAL_LINE=$(grep -E "^${N}: " "$RUN_LOG" | tail -1 || true)
  if ! echo "$FINAL_LINE" | grep -q "$FULL_TOTAL"; then
    echo "[sweep-abort] W=$W final total mismatch: '$FINAL_LINE' (expected $FULL_TOTAL)"
    printf '%s\t%s\tMISMATCH\tFAIL\t-\t-\t-\t-\t%s\n' "$W" "$REORDER_PHASE_JUMP" "$RUN_LOG" >> "$SWEEP_TSV"
    sweep_failures=$((sweep_failures+1))
    break
  fi
  if ! echo "$FINAL_LINE" | grep -q "ok"; then
    echo "[sweep-abort] W=$W final line lacks 'ok': '$FINAL_LINE'"
    printf '%s\t%s\t%s\tNO_OK\t-\t-\t-\t-\t%s\n' "$W" "$REORDER_PHASE_JUMP" "$FULL_TOTAL" "$RUN_LOG" >> "$SWEEP_TSV"
    sweep_failures=$((sweep_failures+1))
    break
  fi

  # metrics
  ELAPSED_HMS=$(echo "$FINAL_LINE" | grep -oE '[0-9]+:[0-9]{2}:[0-9]{2}\.[0-9]+' | tail -1)
  ELAPSED_S=$(python3 -c "
h,m,s=('$ELAPSED_HMS'.split(':'))
print(f'{int(h)*3600+int(m)*60+float(s):.3f}')
")
  KERNEL_MS=$(grep -oE 'kernel_reduce_ms=[0-9]+' "$RUN_LOG" | awk -F= '{s+=$2} END{print s+0}')
  CHUNKS_SEEN=$(grep -c 'split145-gpu-chunk-end' "$RUN_LOG" || true)
  if [[ "$CHUNKS_SEEN" != "$EXPECTED_CHUNKS" ]]; then
    echo "[sweep-warn] W=$W chunk-end count $CHUNKS_SEEN != expected $EXPECTED_CHUNKS (kernel_ms may be off)"
  fi
  printf '%s\t%s\t%s\tOK\t%s\t%s\t%s\t%s\t%s\n' \
    "$W" "$REORDER_PHASE_JUMP" "$FULL_TOTAL" "$ELAPSED_HMS" "$ELAPSED_S" "$KERNEL_MS" "$CHUNKS_SEEN" "$RUN_LOG" >> "$SWEEP_TSV"
  echo "[sweep-point-ok] W=$W total=$FULL_TOTAL elapsed=$ELAPSED_HMS (${ELAPSED_S}s) kernel_ms=$KERNEL_MS chunks=$CHUNKS_SEEN"

  if (( COOLDOWN_SECONDS > 0 )); then
    echo "[cooldown] ${COOLDOWN_SECONDS}s"
    sleep "$COOLDOWN_SECONDS"
  fi
done

echo "================================================================"
echo "[sweep-table] $SWEEP_TSV"
column -t -s $'\t' "$SWEEP_TSV" || cat "$SWEEP_TSV"

if (( sweep_failures > 0 )); then
  printf 'sweep_all_points\t%s points OK\tfailure at some point\tFAIL\n' "$(wc -w <<< "$SWEEP_W")" >> "$SUMMARY"
  echo "[validation-fail] 330 W-sweep aborted after a failed point; DO NOT compare timings from a failed sweep"
  exit 63
fi
printf 'sweep_all_points\tall correct (%s)\tall correct\tOK\n' "$FULL_TOTAL" >> "$SUMMARY"

# ---- ranking summary ----
python3 - "$SWEEP_TSV" << 'PYRANK'
import sys
rows = [l.rstrip('\n').split('\t') for l in open(sys.argv[1], encoding='utf-8')][1:]
ok = [r for r in rows if r[3] == 'OK']
ok.sort(key=lambda r: int(r[6]) if r[6].isdigit() else 1 << 60)
print("[sweep-ranking] by kernel_ms (GPU-pure, ascending):")
base = next((r for r in ok if r[0] == '8'), None)
for r in ok:
    mark = ''
    if base and r is not base and base[6].isdigit() and r[6].isdigit():
        d = (int(r[6]) - int(base[6])) / int(base[6]) * 100.0
        mark = f"  ({d:+.3f}% vs W=8)"
    if r is base:
        mark = "  <= in-session control (w8_j7, current adopted value)"
    print(f"  W={r[0]:>2} kernel_ms={r[6]:>9} elapsed={r[4]}{mark}")
print("[sweep-note] adopt a new W only if it beats W=8 clearly beyond noise (0.03-0.12% band); otherwise keep w8_j7 and close this axis for N21.")
PYRANK

echo "[validation-ok] 330 W-sweep complete: J=$REORDER_PHASE_JUMP, W in {$SWEEP_W}, all points reproduced $FULL_TOTAL; see $SWEEP_TSV; kernel logic identical to 328/329 (adopted SoA baseline); Codon-atomic question closed (no device atomics; rev292 plan-C-1 impossible in pure Codon; rev84 CUDA C runner is the remaining tail-effect route)"
echo "[archive-hint] tar cf 330Py_wjsweep_reorder_retune_logs_N21_wsweep_${TS}.tar -C \"${LOG_ROOT%/}\" \"$(basename "$LOGDIR")\""
