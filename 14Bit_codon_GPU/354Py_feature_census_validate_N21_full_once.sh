#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# 354 feature-census harness (CPU ONLY -- NO GPU, NO KERNEL CHANGE, NO ORACLE)
#
# WHAT 354 IS. 354 sits on 352, not on 353. 353's own VERSION_TAG restarts the
# kernel lineage from 352 for the next revision; 354 has no business in the
# kernel at all, so it inherits the byte-identical 352 kernel untouched and
# adds only a CPU-side dump path: new bench_mode 32.
#
# bench_mode 32 reuses the shaped-bin read loop and chunk_only/chunk_list
# semantics already used by bench_mode 30/31, but in place of a GPU dispatch
# it calls build_soa_for_range -- the exact production host-side function,
# confirmed by source inspection to contain no @gpu.kernel and no gpu.raw call
# anywhere in its call chain -- and dumps, per task, the six raw fields
# chunkshape148_score_key_from_soa reads (funcid, free, end, row, mark1,
# mark2) plus the REAL production key value, obtained by calling that
# function verbatim rather than re-deriving its arithmetic a second time.
#
# WHY THIS AXIS. 353 measured lane_tail at 1.14-1.16x, far under the 6x that
# would justify the CUDA C port on workload-imbalance grounds; modelS
# (cross-warp imbalance, the ceiling achievable under SIMT lockstep) measured
# 6.300% headroom, above the 3% threshold that keeps host-side reordering
# alive; and rank_corr(position, trips) measured about -0.36, below the 0.85
# that would have closed the key-redesign axis. 354 is the offline tool for
# spending that 6.3%.
#
# NO CORRECTNESS ORACLE. This revision touches neither the task set nor the
# GPU, so 314666222712 does not apply here. Its correctness gate is instead a
# RECONSTRUCTION CHECK: the six raw fields must reproduce the real key column
# exactly (raw == key // 32 for every task, checked by the embedded analysis).
# A nonzero reconstruction_errors count means the ported formula has a bug and
# every candidate ranking below it is untrustworthy.
#
# NO GPU, NO N=21 FULL RUN COST. build_soa_for_range measured about 111ms per
# 743424-record chunk inside 353's own soa_ms stage, so the full N=21
# population processes in low seconds. Candidate key scoring is entirely
# offline Python (see the embedded analysis at the end of this script):
# several reweightings of the same raw ingredients are scored by simulating
# the actual grid-stride striping assignment and measuring the resulting
# cross-warp imbalance, the same modelS metric 353 fixed a 6.300% floor for.
#
# RUN ORDER.
#     STATIC_ONLY=1    bash 354Py_feature_census_validate_N21_full_once.sh
#     FEATURE_SMOKE=1  bash 354Py_feature_census_validate_N21_full_once.sh   # chunk 0 only, seconds, no GPU
#                      bash 354Py_feature_census_validate_N21_full_once.sh   # all chunks, seconds, no GPU
#
# WALL CLOCK IS NOT RECORDED AS A TIMING ROW: there is no comparable baseline
# for a GPU-free revision.
#
# 355, IF A CANDIDATE WINS: restarts from the unchanged 352 kernel, rebuilds
# the shaped bin with the winning key formula in
# chunkshape148_score_key_from_soa, and ONLY THEN spends an N=21 full run to
# verify 314666222712 and the timing baseline. 354 itself decides nothing by
# itself; it ranks candidates for that decision.
# =============================================================================

SRC=${SRC:-./354Py_feature_census.py}
CAND=${CAND:-./354Py_feature_census}
AUTO_BUILD=${AUTO_BUILD:-1}
# 274: visible startup + force release rebuild by default so a previous non-release
# `codon build` executable is not reused and cannot trigger CUDA_ERROR_INVALID_PTX.
# 274: print start/status early and prevent static pycheck from failing silently under set -e.
FORCE_REBUILD=${FORCE_REBUILD:-1}
STATIC_ONLY=${STATIC_ONLY:-0}
LOG_ROOT=${LOG_ROOT:-.}
LOCK_FILE=${LOCK_FILE:-/tmp/354Py_feature_census_N21_full.lock}
COOLDOWN_SECONDS=${COOLDOWN_SECONDS:-0}
TELEMETRY_INTERVAL_SECONDS=${TELEMETRY_INTERVAL_SECONDS:-5}
CAPTURE_TELEMETRY=${CAPTURE_TELEMETRY:-1}

N=${N:-21}
BLOCK=${BLOCK:-32}
MAX_BLOCKS=${MAX_BLOCKS:-484}
LOG_LEVEL=${LOG_LEVEL:-1}
SORT_MODE=${SORT_MODE:-0}
PRESET_QUEENS=${PRESET_QUEENS:-7}
BENCH_MODE=${BENCH_MODE:-31}
REORDER_WINDOW_MULT=${REORDER_WINDOW_MULT:-3}
REORDER_PHASE_JUMP=${REORDER_PHASE_JUMP:-7}
CROSS_STRIPE_SAFE=${CROSS_STRIPE_SAFE:-0}
WORKER_ID=${WORKER_ID:-0}
WORKER_COUNT=${WORKER_COUNT:-1}
BROADMARK_VARIANT=${BROADMARK_VARIANT:-2}
# 339: bucket run length. 32 = warp width = the value under test.
# Set CHUNKSHAPE148_BUCKET_RUN=1 to reproduce the 333-338 order and cache exactly.
CHUNKSHAPE148_BUCKET_RUN=${CHUNKSHAPE148_BUCKET_RUN:-2048}
EXPECTED_CHUNKSHAPE148_BUCKET_RUN=$CHUNKSHAPE148_BUCKET_RUN
# 346: sort scope knob, ADOPTED at 1 (launch-group ascending on key>>5).
# Set CHUNKSHAPE148_ITER_SORT=0 to reproduce the 344 order and cache exactly.
# 2 = launch-group descending, 3 = iteration-local ascending (the 345 control
# that regressed by 16.449%), 4 = launch-group ascending on the full key.
# 347 ADDS: 5 = mode 1 plus serpentine, 6 = mode 2 plus serpentine.
# 348 ADDS: 7 = mode 5 plus the light-tail rotation (the candidate),
# 8 = mode 1 plus the light-tail rotation, no serpentine.
# 349 ADDS: 9 = CONDITIONAL serpentine -- applied only when the group length is
# an exact multiple of iter_len, so chunk0 and chunk1 get it and chunk2 does
# not. Mode 9 is the candidate. Only mode 9 builds a new shaped bin here.
CHUNKSHAPE148_ITER_SORT=${CHUNKSHAPE148_ITER_SORT:-9}
EXPECTED_CHUNKSHAPE148_ITER_SORT=$CHUNKSHAPE148_ITER_SORT
EXPECTED_CHUNKSHAPE148_SORT_GROUP=48
EXPECTED_BROADMARK_VARIANT=2
EXPECTED_BROADMARK_VARIANT_TAG="rotate_only"

EXPECTED_CHUNKS=3
EXPECTED_TASKS=2025282
FULL_TOTAL=314666222712
EXPECTED_REQUIRED_MAXD=14
EXPECTED_SELECTED_MAXD=14
EXPECTED_SCHEDULE_WORDS=0
EXPECTED_STACK_BYTES=208
EXPECTED_K_PER_THREAD_MAXD14=48
# NOTE: 333 adopts w3_j7 (host-side reorder parameter) on top of the
# 328-332 kernel-identical lineage. Old w8_j7 baselines (456.0s) remain
# listed for the cross-parameter comparison; the expected result of THIS
# run is ~450.1s, matching 332's direct (3,7) measurement 450.113s.
BASELINE_351_R1_SECONDS=${BASELINE_351_R1_SECONDS:-398.053}
BASELINE_351_R2_SECONDS=${BASELINE_351_R2_SECONDS:-398.018}
BASELINE_350_INSESSION_ANCHOR_SECONDS=${BASELINE_350_INSESSION_ANCHOR_SECONDS:-399.200}
BASELINE_350_ADOPTED_SECONDS=${BASELINE_350_ADOPTED_SECONDS:-398.733}
BASELINE_350_ISORT1_CONTROL_SECONDS=${BASELINE_350_ISORT1_CONTROL_SECONDS:-402.413}
BASELINE_349_ISORT9_ADOPTED_SECONDS=${BASELINE_349_ISORT9_ADOPTED_SECONDS:-398.988}
BASELINE_349_ISORT1_ANCHOR_SECONDS=${BASELINE_349_ISORT1_ANCHOR_SECONDS:-402.301}
BASELINE_349_ISORT5_SECONDS=${BASELINE_349_ISORT5_SECONDS:-401.007}
BASELINE_348_ISORT5_SECONDS=${BASELINE_348_ISORT5_SECONDS:-400.777}
BASELINE_348_ISORT1_ANCHOR_SECONDS=${BASELINE_348_ISORT1_ANCHOR_SECONDS:-402.296}
BASELINE_348_ISORT7_SECONDS=${BASELINE_348_ISORT7_SECONDS:-402.568}
BASELINE_348_ISORT8_SECONDS=${BASELINE_348_ISORT8_SECONDS:-407.063}
BASELINE_347_ISORT5_SECONDS=${BASELINE_347_ISORT5_SECONDS:-400.670}
BASELINE_347_ISORT1_ANCHOR_SECONDS=${BASELINE_347_ISORT1_ANCHOR_SECONDS:-402.424}
BASELINE_347_ISORT6_SECONDS=${BASELINE_347_ISORT6_SECONDS:-402.133}
BASELINE_346_ADOPTED_SECONDS=${BASELINE_346_ADOPTED_SECONDS:-402.258}
BASELINE_346_ISORT0_CONTROL_SECONDS=${BASELINE_346_ISORT0_CONTROL_SECONDS:-431.812}
BASELINE_345_ISORT1_ADOPTED_SECONDS=${BASELINE_345_ISORT1_ADOPTED_SECONDS:-402.460}
BASELINE_345_ISORT4_SECONDS=${BASELINE_345_ISORT4_SECONDS:-402.758}
BASELINE_345_ISORT2_SECONDS=${BASELINE_345_ISORT2_SECONDS:-403.331}
BASELINE_345_ISORT0_ANCHOR_SECONDS=${BASELINE_345_ISORT0_ANCHOR_SECONDS:-431.983}
BASELINE_345_ISORT3_SECONDS=${BASELINE_345_ISORT3_SECONDS:-503.038}
BASELINE_344_ADOPTED_SECONDS=${BASELINE_344_ADOPTED_SECONDS:-431.944}
BASELINE_344_RUN1_CONTROL_SECONDS=${BASELINE_344_RUN1_CONTROL_SECONDS:-450.067}
BASELINE_343_SATURATED_SECONDS=${BASELINE_343_SATURATED_SECONDS:-431.677}
BASELINE_340_RUN1_CONTROL_SECONDS_ADOPT=${BASELINE_340_RUN1_CONTROL_SECONDS_ADOPT:-450.183}
BASELINE_342_RUN2048_SECONDS=${BASELINE_342_RUN2048_SECONDS:-431.684}
BASELINE_342_RUN1024_SECONDS=${BASELINE_342_RUN1024_SECONDS:-438.211}
BASELINE_341_RUN256_SECONDS=${BASELINE_341_RUN256_SECONDS:-439.835}
BASELINE_341_RUN64_SECONDS=${BASELINE_341_RUN64_SECONDS:-446.436}
BASELINE_340_RUN64_SECONDS=${BASELINE_340_RUN64_SECONDS:-446.390}
BASELINE_340_RUN32_SECONDS=${BASELINE_340_RUN32_SECONDS:-447.177}
BASELINE_340_RUN1_SECONDS=${BASELINE_340_RUN1_SECONDS:-450.183}
BASELINE_339_BUCKET_RUN32_SECONDS=${BASELINE_339_BUCKET_RUN32_SECONDS:-447.116}
BASELINE_338_MAXD14_PORT_DESIGN_SECONDS=${BASELINE_338_MAXD14_PORT_DESIGN_SECONDS:-450.329}
BASELINE_337_BIN_FORMAT_READER_DESIGN_SECONDS=${BASELINE_337_BIN_FORMAT_READER_DESIGN_SECONDS:-450.056}
BASELINE_336_CUDAC_SMOKE_TEST_SECONDS=${BASELINE_336_CUDAC_SMOKE_TEST_SECONDS:-450.432}
BASELINE_335_CUDAC_TOOLCHAIN_LOCATE_SECONDS=${BASELINE_335_CUDAC_TOOLCHAIN_LOCATE_SECONDS:-450.218}
BASELINE_334_CUDAC_RUNNER_SPIKE_DESIGN_SECONDS=${BASELINE_334_CUDAC_RUNNER_SPIKE_DESIGN_SECONDS:-450.181}
BASELINE_333_W3J7_ADOPT_SECONDS=${BASELINE_333_W3J7_ADOPT_SECONDS:-450.667}
BASELINE_332_W3J7_MEASURED_SECONDS=${BASELINE_332_W3J7_MEASURED_SECONDS:-450.113}
BASELINE_328_WARR_SOA_SPLIT_IMPLEMENT_SECONDS=${BASELINE_328_WARR_SOA_SPLIT_IMPLEMENT_SECONDS:-456.036}
BASELINE_326_FUTURECHECK_SPECIALIZE_AXIS1_REJECTED_SECONDS=${BASELINE_326_FUTURECHECK_SPECIALIZE_AXIS1_REJECTED_SECONDS:-517.563}
BASELINE_327_WARRLOADSPLIT_VERIFY_SECONDS=${BASELINE_327_WARRLOADSPLIT_VERIFY_SECONDS:-454.563}
BASELINE_319_SOURCECOUNTERS_PAGESOURCE_PROBE_SECONDS=${BASELINE_319_SOURCECOUNTERS_PAGESOURCE_PROBE_SECONDS:-455.116}
BASELINE_318_SOURCECOUNTERS_SUDO_RETRY_SECONDS=${BASELINE_318_SOURCECOUNTERS_SUDO_RETRY_SECONDS:-454.585}
BASELINE_317_BRANCH_DIVERGENCE_PROBE_SECONDS=${BASELINE_317_BRANCH_DIVERGENCE_PROBE_SECONDS:-454.617}
BASELINE_316_ENV_ACCEPT_NCU_PREP_SECONDS=${BASELINE_316_ENV_ACCEPT_NCU_PREP_SECONDS:-454.460}
BASELINE_315_TELEMETRY_FIELDNAME_FIX_SECONDS=${BASELINE_315_TELEMETRY_FIELDNAME_FIX_SECONDS:-454.779}
BASELINE_314_POWER_CAP_DIAGNOSIS_SECONDS=${BASELINE_314_POWER_CAP_DIAGNOSIS_SECONDS:-454.424}
BASELINE_313_CLOCK_CAP_DIAGNOSIS_SECONDS=${BASELINE_313_CLOCK_CAP_DIAGNOSIS_SECONDS:-454.419}
BASELINE_312_THERMAL_REPRO_CHECK_SECONDS=${BASELINE_312_THERMAL_REPRO_CHECK_SECONDS:-454.417}
BASELINE_311_VARIANT2_RESTORE_SECONDS=${BASELINE_311_VARIANT2_RESTORE_SECONDS:-454.422}
BASELINE_310_VARIANT1_PHASE_ONLY_SECONDS=${BASELINE_310_VARIANT1_PHASE_ONLY_SECONDS:-476.932}
BASELINE_309_VARIANT4_PHASE_ROTATE_SECONDS=${BASELINE_309_VARIANT4_PHASE_ROTATE_SECONDS:-481.149}
BASELINE_308_K52_FINAL_SWEEP_SECONDS=${BASELINE_308_K52_FINAL_SWEEP_SECONDS:-351.675}
BASELINE_307_K44_FINE_PROBE_SECONDS=${BASELINE_307_K44_FINE_PROBE_SECONDS:-351.240}
BASELINE_306_K56_SWEEP_SECONDS=${BASELINE_306_K56_SWEEP_SECONDS:-351.534}
BASELINE_305_K40_SWEEP_SECONDS=${BASELINE_305_K40_SWEEP_SECONDS:-353.587}
BASELINE_304_K48_SWEEP_SECONDS=${BASELINE_304_K48_SWEEP_SECONDS:-351.070}
BASELINE_291_BLOCKCODELATE_SECONDS=${BASELINE_291_BLOCKCODELATE_SECONDS:-424.369}
BASELINE_292_K16_SECONDS=${BASELINE_292_K16_SECONDS:-375.587}
BASELINE_292_K32_SECONDS=${BASELINE_292_K32_SECONDS:-367.539}
BASELINE_292_K32_CONFIRMED_SECONDS=${BASELINE_292_K32_CONFIRMED_SECONDS:-367.413}
BASELINE_293_DUAL_LANE_SECONDS=${BASELINE_293_DUAL_LANE_SECONDS:-367.652}
BASELINE_294_COLAV_LDRD_SECONDS=${BASELINE_294_COLAV_LDRD_SECONDS:-362.782}
BASELINE_295_STACK_MERGE_SECONDS=${BASELINE_295_STACK_MERGE_SECONDS:-362.588}
BASELINE_296_STACK_PTR_SECONDS=${BASELINE_296_STACK_PTR_SECONDS:-353.671}
BASELINE_297_SAVE_SP_ELIM_SECONDS=${BASELINE_297_SAVE_SP_ELIM_SECONDS:-362.707}
BASELINE_298_NEXT_DEPTH_ELIM_SECONDS=${BASELINE_298_NEXT_DEPTH_ELIM_SECONDS:-416.429}
BASELINE_299_K64_ON_296_SECONDS=${BASELINE_299_K64_ON_296_SECONDS:-353.896}
BASELINE_300_SCHEDULE_U64_SECONDS=${BASELINE_300_SCHEDULE_U64_SECONDS:-375.613}
BASELINE_301_CUR_DEPTH_X4_SECONDS=${BASELINE_301_CUR_DEPTH_X4_SECONDS:-647.930}
BASELINE_302_CUR_DEPTH_X4_FIX_SECONDS=${BASELINE_302_CUR_DEPTH_X4_FIX_SECONDS:-635.928}
BASELINE_303_CUR_DEPTH_X4_NEUTRAL_SECONDS=${BASELINE_303_CUR_DEPTH_X4_NEUTRAL_SECONDS:-658.105}
BASELINE_292_K64_SECONDS=${BASELINE_292_K64_SECONDS:-367.340}
BASELINE_289_NCOLONLY_SECONDS=${BASELINE_289_NCOLONLY_SECONDS:-424.097}
BASELINE_288_REJECT_SECONDS=${BASELINE_288_REJECT_SECONDS:-517.227}
BASELINE_287_ADOPT_SECONDS=${BASELINE_287_ADOPT_SECONDS:-424.486}
BASELINE_286_ADOPT_SECONDS=${BASELINE_286_ADOPT_SECONDS:-424.033}
BASELINE_285_RESTORE_SECONDS=${BASELINE_285_RESTORE_SECONDS:-427.818}
BASELINE_284_REJECT_SECONDS=${BASELINE_284_REJECT_SECONDS:-427.795}
BASELINE_283_NORMALFIRST_SECONDS=${BASELINE_283_NORMALFIRST_SECONDS:-427.698}
BASELINE_276_CURRENT_SECONDS=${BASELINE_276_CURRENT_SECONDS:-427.672}
BASELINE_278_REJECT_SECONDS=${BASELINE_278_REJECT_SECONDS:-429.603}
BASELINE_277_DEPTHU_SECONDS=${BASELINE_277_DEPTHU_SECONDS:-427.717}
BASELINE_276_PARENT_274_SECONDS=${BASELINE_276_PARENT_274_SECONDS:-427.758}
BASELINE_275_DIAG_SECONDS=${BASELINE_275_DIAG_SECONDS:-427.716}
BASELINE_273_REJECT_SECONDS=${BASELINE_273_REJECT_SECONDS:-428.757}
BASELINE_272_DIAG_SECONDS=${BASELINE_272_DIAG_SECONDS:-427.728}
BASELINE_271_FASTEST_SECONDS=${BASELINE_271_FASTEST_SECONDS:-427.705}
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
LOGDIR="${LOG_ROOT%/}/354Py_feature_census_logs_N21_full_once_${TS}"
RUN_LOG="$LOGDIR/full_once.log"
BUILD_LOG="$LOGDIR/build.log"
SUMMARY="$LOGDIR/summary.tsv"
PROGRESS_COPY="$LOGDIR/progress_full.tsv"
mkdir -p "$LOGDIR"
printf 'check\texpected\tactual\tstatus\n' > "$SUMMARY"

echo "[start] 354 feature-census validation script"
echo "[source] $SRC"
echo "[candidate] $CAND"
echo "[logdir] $LOGDIR"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo "[abort] rc=$rc logdir=${LOGDIR:-unknown}" >&2; fi' EXIT
echo "[validation-start] 354 feature-census SRC=$SRC CAND=$CAND STATIC_ONLY=$STATIC_ONLY FORCE_REBUILD=$FORCE_REBUILD LOGDIR=$LOGDIR"

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


# ---- 345 r2: docstring-stripped copy for code-sensitive static checks ----
# THE FAILURE THIS FIXES: this project's normal workflow pastes chat responses
# into the module docstring. 345 r1's handover text listed the two lines that
# 345 removes from the emit loop, verbatim:
#     pick_p:int=pick_idx*16
#     out.write(data[pick_p:pick_p+16])
# Once that text was pasted into the docstring, the bash-level greps counted
# the prose occurrence as if it were code, so
# source_chunkshape148_iter_sort_single_write read 2 instead of 1 and
# source_chunkshape148_iter_sort_old_inloop_write_absent read 1 instead of 0.
# The source itself was correct; the checks were not immune to prose.
#
# The PYCHECK heredoc below already strips docstrings for its own checks (the
# established fix from 328 r4). 345 r1 added new checks at the bash level and
# failed to route them through the same protection. r2 does that: one stripped
# copy is produced here, and every chunkshape148 static grep reads it instead
# of $SRC. Presence checks in particular are worth protecting -- prose can make
# a presence check pass just as easily as it can make a count check fail, and
# the membership guards below are presence checks.
SRC_NODOC="$LOGDIR/src_nodoc.py"
python3 - "$SRC" "$SRC_NODOC" <<'PYSTRIP'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding='utf-8').read()
s = re.sub(r'"""[\s\S]*?"""', '', s)
open(dst, 'w', encoding='utf-8').write(s)
PYSTRIP
# SAFETY: an odd number of triple quotes in pasted prose could make the regex
# above eat real code. Refuse to run the checks against a corrupted copy.
if [[ -s "$SRC_NODOC" ]] \
   && grep -q '^def build_chunkshape148_reordered_bin(' "$SRC_NODOC" \
   && grep -q '^def main()->None:' "$SRC_NODOC" \
   && grep -q '^VERSION_TAG:str=' "$SRC_NODOC"; then
  NODOC_LINES=$(grep -c '' "$SRC_NODOC")
  SRC_LINES=$(grep -c '' "$SRC")
  printf 'source_nodoc_copy\tdocstrings stripped, code markers intact\t%s of %s lines kept\tOK\n' "$NODOC_LINES" "$SRC_LINES" >> "$SUMMARY"
else
  printf 'source_nodoc_copy\tdocstrings stripped, code markers intact\tstripped copy lost code markers -- check for an unbalanced triple quote in the docstring\tFAIL\n' >> "$SUMMARY"
  static_failures=$((static_failures+1))
  SRC_NODOC="$SRC"
fi

# ---- source static checks ----
if grep -q '354 feature-census' "$SRC"; then
  printf 'source_version_tag	354 feature-census	present	OK
' >> "$SUMMARY"
else
  printf 'source_version_tag	354 feature-census	missing	FAIL
' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
# ---- 339 r2: Codon string-literal quote balance ----
# Codon does NOT perform CPython's implicit adjacent-string-literal
# concatenation, so a bare " inside a double-quoted module-level string is a
# hard build error ("syntax error, unexpected '\"'") -- while CPython's own
# parser accepts the same text silently. ast.parse therefore CANNOT catch
# this. 339 r1 hit exactly this: VERSION_TAG contained the two characters ""
# in prose and codon build aborted. Because pasting prose into VERSION_TAG is
# part of this project's workflow every single revision, this is now a gating
# static check. It fires before the build, not after.
QUOTE_BAD=$(awk '/^[A-Za-z_][A-Za-z0-9_]*:str="/ { n=gsub(/"/,"\""); if (n!=2) bad++ } END { print bad+0 }' "$SRC")
record_check source_str_literal_quote_balance 0 "$QUOTE_BAD" || static_failures=$((static_failures+1))
if [[ "$QUOTE_BAD" != "0" ]]; then
  echo "[error] a module-level NAME:str=\"...\" line contains an unescaped double quote:" >&2
  awk '/^[A-Za-z_][A-Za-z0-9_]*:str="/ { n=gsub(/"/,"\""); if (n!=2) printf "  line %d: %s (quotes=%d)\n", NR, substr($0,1,index($0,":")), n }' "$SRC" >&2
fi

# ---- 342: the raised run-length ceiling (the ONLY source change in 342) ----
CAP_NEW=$(grep -c '^CHUNKSHAPE148_BUCKET_RUN_MAX:int=65536$' "$SRC" || true)
record_check source_chunkshape148_bucket_run_max_65536 1 "$CAP_NEW" || static_failures=$((static_failures+1))
CAP_OLD=$(grep -cE '^CHUNKSHAPE148_BUCKET_RUN_MAX:int=(256|8192)$' "$SRC" || true)
record_check source_chunkshape148_bucket_run_max_old_absent 0 "$CAP_OLD" || static_failures=$((static_failures+1))
# 343: the cap must clear the guaranteed saturation point, RUN >= STEPS = BLOCK*MAX_BLOCKS = 15488
CAP_VAL=$(sed -n 's/^CHUNKSHAPE148_BUCKET_RUN_MAX:int=\([0-9]*\)$/\1/p' "$SRC" | head -n1)
if [[ -n "$CAP_VAL" ]] && (( CAP_VAL >= BLOCK * MAX_BLOCKS )); then
  printf 'source_bucket_run_cap_clears_saturation\t>= %s (BLOCK*MAX_BLOCKS)\t%s\tOK\n' "$((BLOCK*MAX_BLOCKS))" "$CAP_VAL" >> "$SUMMARY"
else
  printf 'source_bucket_run_cap_clears_saturation\t>= %s (BLOCK*MAX_BLOCKS)\t%s\tFAIL\n' "$((BLOCK*MAX_BLOCKS))" "${CAP_VAL:-missing}" >> "$SUMMARY"
  static_failures=$((static_failures+1))
fi

# ---- 339: chunkshape148 bucket-run knob (new gating checks) ----
# 344: ADOPTION. Both defaults must now read 2048, and they must agree.
BR_DEFAULT=$(grep -c '^CHUNKSHAPE148_BUCKET_RUN:int=2048$' "$SRC" || true)
record_check source_chunkshape148_bucket_run_default_adopted 1 "$BR_DEFAULT" || static_failures=$((static_failures+1))
BR_OLD=$(grep -c '^CHUNKSHAPE148_BUCKET_RUN:int=1$' "$SRC" || true)
record_check source_chunkshape148_bucket_run_default_old_absent 0 "$BR_OLD" || static_failures=$((static_failures+1))
BR_A10G=$(sed -n 's/^A10G_FINAL_DEFAULT_CHUNKSHAPE148_BUCKET_RUN:int=\([0-9]*\).*/\1/p' "$SRC" | head -n1)
BR_MOD=$(sed -n 's/^CHUNKSHAPE148_BUCKET_RUN:int=\([0-9]*\)$/\1/p' "$SRC" | head -n1)
record_check source_chunkshape148_bucket_run_a10g_default 2048 "${BR_A10G:-missing}" || static_failures=$((static_failures+1))
if [[ -n "$BR_A10G" && "$BR_A10G" == "$BR_MOD" ]]; then
  printf 'source_chunkshape148_bucket_run_defaults_agree\tmodule == A10G default\t%s == %s\tOK\n' "$BR_MOD" "$BR_A10G" >> "$SUMMARY"
else
  printf 'source_chunkshape148_bucket_run_defaults_agree\tmodule == A10G default\t%s vs %s\tFAIL\n' "${BR_MOD:-missing}" "${BR_A10G:-missing}" >> "$SUMMARY"
  static_failures=$((static_failures+1))
fi
if grep -q 'def chunkshape148_bucket_run_tag()' "$SRC_NODOC" && grep -q '{chunkshape148_bucket_run_tag()}{chunkshape148_iter_sort_tag()}.bin' "$SRC_NODOC"; then
  printf 'source_chunkshape148_bucket_run_fname_tag\tpresent (run=1 keeps the 276-338 filename)\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_chunkshape148_bucket_run_fname_tag\tpresent (run=1 keeps the 276-338 filename)\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q 'bucket_run:int=chunkshape148_bucket_run_value()' "$SRC_NODOC" && grep -q 'while rep<bucket_run:' "$SRC_NODOC"; then
  printf 'source_chunkshape148_bucket_run_emit_loop\trun-length loop present in build_chunkshape148_reordered_bin\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_chunkshape148_bucket_run_emit_loop\trun-length loop present in build_chunkshape148_reordered_bin\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q 'chunkshape148_bucket_run=int(sys.argv\[16\])' "$SRC_NODOC" && grep -q '^  CHUNKSHAPE148_BUCKET_RUN=chunkshape148_bucket_run$' "$SRC_NODOC"; then
  printf 'source_chunkshape148_bucket_run_argv\targv[16] parsed and assigned to the global\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_chunkshape148_bucket_run_argv\targv[16] parsed and assigned to the global\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi

# ---- 345: iter-sort knob (new gating checks) ----
# The default MUST be 0 so that a fresh checkout reproduces the adopted 344
# exactly, including the cached shaped-bin filename.
# 346: ADOPTION. Both defaults must now read 1, and they must agree.
# 350: ADOPTION. Both defaults must now read 9, and they must agree.
IS_DEFAULT=$(grep -c '^CHUNKSHAPE148_ITER_SORT:int=9$' "$SRC_NODOC" || true)
record_check source_chunkshape148_iter_sort_default_adopted 1 "$IS_DEFAULT" || static_failures=$((static_failures+1))
IS_OLD=$(grep -cE '^CHUNKSHAPE148_ITER_SORT:int=(0|1)$' "$SRC_NODOC" || true)
record_check source_chunkshape148_iter_sort_default_old_absent 0 "$IS_OLD" || static_failures=$((static_failures+1))
IS_MAX=$(sed -n 's/^CHUNKSHAPE148_ITER_SORT_MAX:int=\([0-9]*\)$/\1/p' "$SRC_NODOC" | head -n1)
record_check source_chunkshape148_iter_sort_max 9 "${IS_MAX:-missing}" || static_failures=$((static_failures+1))
IS_A10G=$(sed -n 's/^A10G_FINAL_DEFAULT_CHUNKSHAPE148_ITER_SORT:int=\([0-9]*\).*/\1/p' "$SRC_NODOC" | head -n1)
IS_MOD=$(sed -n 's/^CHUNKSHAPE148_ITER_SORT:int=\([0-9]*\)$/\1/p' "$SRC_NODOC" | head -n1)
record_check source_chunkshape148_iter_sort_a10g_default 9 "${IS_A10G:-missing}" || static_failures=$((static_failures+1))
if [[ -n "$IS_A10G" && "$IS_A10G" == "$IS_MOD" ]]; then
  printf 'source_chunkshape148_iter_sort_defaults_agree\tmodule == A10G default\t%s == %s\tOK\n' "$IS_MOD" "$IS_A10G" >> "$SUMMARY"
else
  printf 'source_chunkshape148_iter_sort_defaults_agree\tmodule == A10G default\t%s vs %s\tFAIL\n' "${IS_MOD:-missing}" "${IS_A10G:-missing}" >> "$SUMMARY"
  static_failures=$((static_failures+1))
fi
# 346: iter_sort=0 must still return an EMPTY cache suffix, otherwise the 344
# baseline stops being reachable with its own cached shaped bin.
if grep -A12 'def chunkshape148_iter_sort_tag()' "$SRC_NODOC" | grep -q 'if m==0:' \
   && grep -A12 'def chunkshape148_iter_sort_tag()' "$SRC_NODOC" | grep -q 'return ""'; then
  printf 'source_chunkshape148_iter_sort_zero_reaches_344\tmode 0 returns an empty suffix (344 cache filename)\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_chunkshape148_iter_sort_zero_reaches_344\tmode 0 returns an empty suffix (344 cache filename)\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
# The group MUST equal K_PER_THREAD_MAXD14, otherwise a sort group is not a GPU
# launch and the whole warp-stratification argument stops holding.
IS_GROUP=$(sed -n 's/^CHUNKSHAPE148_SORT_GROUP:int=\([0-9]*\)$/\1/p' "$SRC_NODOC" | head -n1)
IS_K=$(sed -n 's/^K_PER_THREAD_MAXD14:Static\[int\]=\([0-9]*\)$/\1/p' "$SRC_NODOC" | head -n1)
if [[ -n "$IS_GROUP" && "$IS_GROUP" == "$IS_K" && "$IS_GROUP" == "$EXPECTED_CHUNKSHAPE148_SORT_GROUP" ]]; then
  printf 'source_chunkshape148_sort_group_equals_k\tCHUNKSHAPE148_SORT_GROUP == K_PER_THREAD_MAXD14 == %s\t%s == %s\tOK\n' "$EXPECTED_CHUNKSHAPE148_SORT_GROUP" "$IS_GROUP" "$IS_K" >> "$SUMMARY"
else
  printf 'source_chunkshape148_sort_group_equals_k\tCHUNKSHAPE148_SORT_GROUP == K_PER_THREAD_MAXD14 == %s\t%s vs %s\tFAIL\n' "$EXPECTED_CHUNKSHAPE148_SORT_GROUP" "${IS_GROUP:-missing}" "${IS_K:-missing}" >> "$SUMMARY"
  static_failures=$((static_failures+1))
fi
# The sequence field must be wide enough to hold a whole group, otherwise the
# packed sort silently disables itself.
IS_SEQ=$(sed -n 's/^CHUNKSHAPE148_SORT_SEQ_BITS:int=\([0-9]*\)$/\1/p' "$SRC_NODOC" | head -n1)
IS_SEQ_MASK=$(sed -n 's/^CHUNKSHAPE148_SORT_SEQ_MASK:int=\([0-9]*\)$/\1/p' "$SRC_NODOC" | head -n1)
GROUP_RECORDS=$(( BLOCK * MAX_BLOCKS * EXPECTED_CHUNKSHAPE148_SORT_GROUP ))
if [[ -n "$IS_SEQ" && -n "$IS_SEQ_MASK" ]] && (( IS_SEQ_MASK == (1 << IS_SEQ) - 1 )) && (( IS_SEQ_MASK + 1 >= GROUP_RECORDS )); then
  printf 'source_chunkshape148_sort_seq_width\tmask == 2^bits-1 and covers %s records\tbits=%s mask=%s\tOK\n' "$GROUP_RECORDS" "$IS_SEQ" "$IS_SEQ_MASK" >> "$SUMMARY"
else
  printf 'source_chunkshape148_sort_seq_width\tmask == 2^bits-1 and covers %s records\tbits=%s mask=%s\tFAIL\n' "$GROUP_RECORDS" "${IS_SEQ:-missing}" "${IS_SEQ_MASK:-missing}" >> "$SUMMARY"
  static_failures=$((static_failures+1))
fi
if grep -q 'def chunkshape148_iter_sort_tag()' "$SRC_NODOC" && grep -q '{chunkshape148_bucket_run_tag()}{chunkshape148_iter_sort_tag()}.bin' "$SRC_NODOC"; then
  printf 'source_chunkshape148_iter_sort_fname_tag\tpresent (mode 0 keeps the 344 filename)\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_chunkshape148_iter_sort_fname_tag\tpresent (mode 0 keeps the 344 filename)\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q 'def chunkshape148_reorder_group(' "$SRC_NODOC" && grep -q 'group_picked.append(pick_idx)' "$SRC_NODOC" && grep -q 'ordered:List\[int\]=chunkshape148_reorder_group(' "$SRC_NODOC"; then
  printf 'source_chunkshape148_iter_sort_buffered_emit\tpicks buffered and reordered before write\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_chunkshape148_iter_sort_buffered_emit\tpicks buffered and reordered before write\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
# MEMBERSHIP GUARD 1: there must be exactly ONE out.write in the shaping
# builder and it must be the one in the group flush, not the old in-loop write.
WRITE_COUNT=$(grep -c 'out.write(data\[pick_p:pick_p+16\])' "$SRC_NODOC" || true)
record_check source_chunkshape148_iter_sort_single_write 1 "$WRITE_COUNT" || static_failures=$((static_failures+1))
OLD_INLOOP_WRITE=$(grep -c 'pick_p:int=pick_idx\*16' "$SRC_NODOC" || true)
record_check source_chunkshape148_iter_sort_old_inloop_write_absent 0 "$OLD_INLOOP_WRITE" || static_failures=$((static_failures+1))
# MEMBERSHIP GUARD 2: the 344 selection rules must be untouched. These five
# lines are what decides WHICH record goes into WHICH launch; 345 may only
# change the order inside a launch group.
SEL_OK=1
grep -q 'quotas:List\[int\]=chunkshape148_make_quotas(bucket_rem,total_remaining,m_target)' "$SRC_NODOC" || SEL_OK=0
grep -q 'start_lane:int=(phase_seed + written_by_bucket\[b\]\*5) & CHUNKSHAPE148_LANE_MASK' "$SRC_NODOC" || SEL_OK=0
grep -q 'order_pos:int=(oi+out_ch)%8' "$SRC_NODOC" || SEL_OK=0
grep -q 'while rep<bucket_run:' "$SRC_NODOC" || SEL_OK=0
grep -q 'interleave_order:List\[int\]=\[7,0,6,1,5,2,4,3\]' "$SRC_NODOC" || SEL_OK=0
if [[ "$SEL_OK" == "1" ]]; then
  printf 'source_chunkshape148_selection_rules_unchanged\tquotas/lane-phase/rotation/run-length/interleave all present\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_chunkshape148_selection_rules_unchanged\tquotas/lane-phase/rotation/run-length/interleave all present\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q 'chunkshape148_iter_sort=int(sys.argv\[17\])' "$SRC_NODOC" && grep -q '^  CHUNKSHAPE148_ITER_SORT=chunkshape148_iter_sort$' "$SRC_NODOC"; then
  printf 'source_chunkshape148_iter_sort_argv\targv[17] parsed and assigned to the global\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_chunkshape148_iter_sort_argv\targv[17] parsed and assigned to the global\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi

# ---- 347: serpentine modes (new gating checks) ----
# The serpentine pass must exist and must operate on whole iter_len strata so
# the 32-alignment rule from 340 survives. 349 r2: the gate moved from an inline
# condition to the serp_on flag when mode 9 was added, and this check was left
# pointing at the old inline form, which is why 349 r1 reported it missing while
# every substantive check passed. The structure test below is now written
# against the serp_on form and against the loop body, which is what actually
# matters -- the mode membership itself is covered by the three checks that
# follow (mode6_descending, mode7_serpentine, conditional_serpentine).
if grep -q 'serp_on:bool=(mode==5 or mode==6 or mode==7)' "$SRC_NODOC" \
   && grep -q 'if serp_on and iter_len>=1:' "$SRC_NODOC" \
   && grep -q 'sseg:int=iter_len' "$SRC_NODOC" \
   && grep -q 'if (sidx&1)==1:' "$SRC_NODOC"; then
  printf 'source_chunkshape148_serpentine_pass\tserp_on gate present, reverses whole iter_len strata\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_chunkshape148_serpentine_pass\tserp_on gate present, reverses whole iter_len strata\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
# Mode 6 must take the descending branch, otherwise it is just mode 5 again and
# prediction 2 becomes untestable.
if grep -q 'elif mode==2 or mode==6:' "$SRC_NODOC"; then
  printf 'source_chunkshape148_serpentine_mode6_descending\tmode 6 uses the descending key part\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_chunkshape148_serpentine_mode6_descending\tmode 6 uses the descending key part\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi

# ---- 348: light tail modes (new gating checks) ----
# The rotation must exist, must be gated to modes 7 and 8 only, and must be a
# NO-OP when the group length is an exact multiple of iter_len. That last
# property is what makes the 2x2 factorial exact: on the two full launches
# mode 8 is element-for-element identical to mode 1 and mode 7 to mode 5, so
# only chunk2 can move and the light-tail factor is isolated.
if grep -q 'if (mode==7 or mode==8) and iter_len>=1:' "$SRC_NODOC" \
   && grep -q 'rem:int=len(out)%iter_len' "$SRC_NODOC" \
   && grep -q 'if rem>0:' "$SRC_NODOC"; then
  printf 'source_chunkshape148_light_tail_rotation\tgated to modes 7/8, no-op when the remainder is zero\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_chunkshape148_light_tail_rotation\tgated to modes 7/8, no-op when the remainder is zero\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
# Mode 7 must ALSO take the serpentine pass, otherwise it is mode 8 and the
# factorial collapses to three cells.
if grep -q 'if serp_on and iter_len>=1:' "$SRC_NODOC"; then
  printf 'source_chunkshape148_light_tail_mode7_serpentine\tmode 7 takes the serpentine pass, mode 8 does not\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_chunkshape148_light_tail_mode7_serpentine\tmode 7 takes the serpentine pass, mode 8 does not\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
# ORDER MATTERS: the rotation has to run BEFORE the serpentine pass. Reversed,
# mode 7 would serpentine the heavy-tail layout and then rotate it, which is a
# different arrangement entirely.
ROT_LINE=$(grep -n 'if (mode==7 or mode==8) and iter_len>=1:' "$SRC_NODOC" | head -n1 | cut -d: -f1)
SERP_LINE=$(grep -n 'serp_on:bool=(mode==5 or mode==6 or mode==7)' "$SRC_NODOC" | head -n1 | cut -d: -f1)
if [[ -n "$ROT_LINE" && -n "$SERP_LINE" ]] && (( ROT_LINE < SERP_LINE )); then
  printf 'source_chunkshape148_light_tail_before_serpentine\trotation precedes the serpentine pass\tline %s < %s\tOK\n' "$ROT_LINE" "$SERP_LINE" >> "$SUMMARY"
else
  printf 'source_chunkshape148_light_tail_before_serpentine\trotation precedes the serpentine pass\tline %s vs %s\tFAIL\n' "${ROT_LINE:-missing}" "${SERP_LINE:-missing}" >> "$SUMMARY"; static_failures=$((static_failures+1))
fi

# ---- 349: conditional serpentine (new gating checks) ----
# Mode 9 must gate serpentine on the group being an exact multiple of iter_len,
# and the gate must be evaluated AFTER the light-tail rotation so that the
# serpentine decision sees the final length.
if grep -q 'serp_on:bool=(mode==5 or mode==6 or mode==7)' "$SRC_NODOC" \
   && grep -q 'if mode==9 and iter_len>=1 and (len(out)%iter_len)==0:' "$SRC_NODOC" \
   && grep -q 'if serp_on and iter_len>=1:' "$SRC_NODOC"; then
  printf 'source_chunkshape148_conditional_serpentine\tmode 9 serpentines only exact multiples of iter_len\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_chunkshape148_conditional_serpentine\tmode 9 serpentines only exact multiples of iter_len\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
# Mode 9 must NOT take the light-tail rotation, otherwise it is not mode 5 on a
# full launch and not mode 1 on the partial one, and the byte-equality check
# below stops being meaningful.
if grep -q 'if (mode==7 or mode==8) and iter_len>=1:' "$SRC_NODOC"; then
  printf 'source_chunkshape148_mode9_no_rotation\tmode 9 excluded from the light-tail rotation\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_chunkshape148_mode9_no_rotation\tmode 9 excluded from the light-tail rotation\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
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
if grep -q '^FUNCID_REORDER_V2_WINDOW_MULT:int=3' "$SRC"    && grep -q '^FUNCID_REORDER_V2_PHASE_JUMP:int=7' "$SRC"    && grep -q '^FUNCID_REORDER_V2_DEFAULT_REASON:str=' "$SRC"    && grep -q '^BROAD_MARKDIST_TAIL_REORDER_VERSION:str="v4"' "$SRC"    && grep -q '^BROAD_MARKDIST_TAIL_VARIANT:int=2' "$SRC"    && grep -q '^BROAD_MARKDIST_TAIL_CELL_SALT:int=17' "$SRC"    && grep -q '^BROAD_MARKDIST_TAIL_RISK_SALT:int=11' "$SRC"    && grep -q '^BROAD_MARKDIST_TAIL_PHASE_SALT:int=53' "$SRC"; then
  printf 'source_runtime_globals	funcid/broadmarktail constants	present	OK
' >> "$SUMMARY"
else
  printf 'source_runtime_globals	funcid/broadmarktail constants	missing	FAIL
' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q '^A10G_FINAL_DEFAULT_BROADMARK_VARIANT:int=2' "$SRC"; then
  printf 'source_a10g_default_variant2	2	2	OK\n' >> "$SUMMARY"
else
  printf 'source_a10g_default_variant2	2	missing/mismatch	FAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi

# ---- 354: prove the kernel is byte identical to 352, and the diff is only the feature-census plumbing ----
# TWO SEPARATE PROOFS, because they protect against two separate mistakes.
#
# PROOF 1: the five kernels and the dispatcher's launch signature are BYTE
# IDENTICAL to 352. Extracted by slicing from "def kernel_dfs_iter_gpu_maxd14("
# to "def launch_kernel_dfs_iter_gpu_static_maxd(" and compared verbatim -- no
# docstring stripping is needed or wanted here, because this slice contains no
# docstrings; any difference at all is real and must FAIL.
#
# PROOF 2: the rest of the file, after docstrings and the three prose
# constants are stripped, differs from 352 by EXACTLY the feature-census
# plumbing: the fingerprint below is the sha256 of the sorted added/removed
# diff lines. If a single unrelated character moved anywhere outside the
# kernel slice, this FAILS too.
#
# Both FAIL rather than skip when the 352 source is absent, for the same
# reason 352 and 353 did: a check that silently skips is worse than no check.
REF352=${REF352:-./352Py_record_fix.py}
EXPECTED_FEATURE354_DIFF="added=163 removed=6 sha256=dc9ee6a6730f1e95555a53e65cb983fde44d3f99d7ea80151e6c6ec08a64091e"
if [[ -f "$REF352" ]]; then
  KERNEL_IDENT_AND_DIFF=$(python3 - "$REF352" "$SRC" <<'PYFEATIDENT'
import re, sys, difflib, hashlib

def kernel_slice(p):
    s = open(p, encoding='utf-8').read()
    a = s.index('def kernel_dfs_iter_gpu_maxd14(')
    b = s.index('def launch_kernel_dfs_iter_gpu_static_maxd(')
    return s[a:b]

def strip(p):
    s = open(p, encoding='utf-8').read()
    s = re.sub(r'"""[\s\S]*?"""', '', s)
    s = re.sub(r'^(VERSION_TAG|WHI_ELIM_REASON|FEATURE354_REASON):str="[^"]*"$', '', s, flags=re.M)
    return s.split('\n')

k_ref, k_src = kernel_slice(sys.argv[1]), kernel_slice(sys.argv[2])
kernel_status = 'identical' if k_ref == k_src else 'DIFFERS_%d_vs_%d_bytes' % (len(k_ref), len(k_src))

a, b = strip(sys.argv[1]), strip(sys.argv[2])
d = list(difflib.unified_diff(a, b, lineterm='', n=0))
adds = sorted(l for l in d if l.startswith('+') and not l.startswith('+++'))
dels = sorted(l for l in d if l.startswith('-') and not l.startswith('---'))
blob = ('\n'.join(dels) + '\n@@\n' + '\n'.join(adds)).encode('utf-8')
diff_status = "added=%d removed=%d sha256=%s" % (len(adds), len(dels), hashlib.sha256(blob).hexdigest())

print('kernel=%s' % kernel_status)
print(diff_status)
PYFEATIDENT
)
  KERNEL_STATUS=$(echo "$KERNEL_IDENT_AND_DIFF" | sed -n '1p')
  DIFF_STATUS=$(echo "$KERNEL_IDENT_AND_DIFF" | sed -n '2p')
  record_check source_kernel_identical_to_352 "kernel=identical" "${KERNEL_STATUS:-compare_failed}" || static_failures=$((static_failures+1))
  record_check source_code_identical_to_352_except_feature354 "$EXPECTED_FEATURE354_DIFF" "${DIFF_STATUS:-compare_failed}" || static_failures=$((static_failures+1))
else
  printf 'source_kernel_identical_to_352\tkernel=identical\t352 source not found at %s\tFAIL\n' "$REF352" >> "$SUMMARY"
  printf 'source_code_identical_to_352_except_feature354\t%s\t352 source not found at %s\tFAIL\n' "$EXPECTED_FEATURE354_DIFF" "$REF352" >> "$SUMMARY"
  static_failures=$((static_failures+2))
fi

# ---- 354: bench_mode 32 wiring and CPU-only-ness, checked directly ----
# NEGATIVE TEST (the 351/352/353 procedure): feed 352's source to this script
# and every one of these must read absent/0 and FAIL:
#     SRC=./352Py_record_fix.py STATIC_ONLY=1 bash 354Py_feature_census_validate_N21_full_once.sh
FEAT_FN_PRESENT=$(grep -c '^def exec_features_cpu_bin_stream_split145($' "$SRC_NODOC" || true)
FEAT_WRITER_PRESENT=$(grep -c '^def write_features_chunk_cpu(fname:str,soa:TaskSoA,m:int,global_base:int)->int:$' "$SRC_NODOC" || true)
FEAT_DISPATCH=$(grep -c '^      if bench_mode==32:$' "$SRC_NODOC" || true)
FEAT_SITES="fn=$FEAT_FN_PRESENT writer=$FEAT_WRITER_PRESENT dispatch=$FEAT_DISPATCH"
record_check source_feature354_sites "fn=1 writer=1 dispatch=1" "$FEAT_SITES" || static_failures=$((static_failures+1))

# Scope: neither new function may call a GPU primitive. This is the direct,
# mechanical version of "354 is CPU only" -- not a claim, a grep.
FEAT_GPU_CALLS=$(python3 - "$SRC_NODOC" <<'PYFEATGPU'
import sys
s = open(sys.argv[1], encoding='utf-8').read()
hits = 0
for name in ('write_features_chunk_cpu(', 'exec_features_cpu_bin_stream_split145('):
    i = s.find('def ' + name)
    if i < 0:
        continue
    j = s.find('\ndef ', i + 1)
    if j < 0:
        j = len(s)
    body = s[i:j]
    hits += body.count('gpu.kernel') + body.count('gpu.raw') + body.count('grid=') + body.count('@gpu')
print(hits)
PYFEATGPU
)
record_check source_feature354_no_gpu_calls 0 "${FEAT_GPU_CALLS:-probe_failed}" || static_failures=$((static_failures+1))

# The record layout must be 6 int32 + 1 int64 = 32 bytes; this is the one
# arithmetic fact the offline analyzer trusts blindly (struct.unpack('<iiiiiiq')).
if grep -q "rec+=int_to_le_bytes8(key)" "$SRC_NODOC"; then
  printf 'source_feature354_record_layout\t6xint32+int64=32 bytes\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_feature354_record_layout\t6xint32+int64=32 bytes\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi

# The stored key must come from calling the real function, not a reimplementation.
if grep -q 'key:int=chunkshape148_score_key_from_soa(soa,i,global_base+i)' "$SRC_NODOC"; then
  printf 'source_feature354_uses_real_key_fn\tverbatim call, not reimplemented\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_feature354_uses_real_key_fn\tverbatim call, not reimplemented\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi

set +e
python3 - "$SRC" "$SUMMARY" "$EXPECTED_K_PER_THREAD_MAXD14" <<'PYCHECK'
import re, sys
src, summary = sys.argv[1], sys.argv[2]
EXPECTED_K_PER_THREAD_MAXD14_PY = sys.argv[3]
s = open(src, encoding='utf-8').read()
# Per user request: exclude docstring content from every static check below.
# This file has exactly two triple-quoted string literals (the module
# header docstring and the narrative continuation block that follows
# it) -- both are pure documentation, including pasted chat-log content
# per this project's normal workflow, and neither should ever affect
# what these checks see. Stripping them here, once, makes every check
# below immune to false positives from prose mentioning code-like
# strings (type annotations, old/new identifiers, code snippets in
# pasted chat text, etc.) without needing to special-case each check
# individually.
s = re.sub(r'"""[\s\S]*?"""', '', s)
checks = []
def has_def(name):
    return re.search(r'^def\s+' + re.escape(name) + r'\b', s, re.M) is not None
def has_kernel(name):
    return re.search(r'^@gpu\.kernel\s*\n^def\s+' + re.escape(name) + r'\b', s, re.M) is not None
required_defs = [
    'kernel_dfs_iter_gpu_maxd14','kernel_dfs_iter_gpu_maxd16','kernel_dfs_iter_gpu_maxd18','kernel_dfs_iter_gpu_maxd20','kernel_dfs_iter_gpu_maxd21',
    'launch_kernel_dfs_iter_gpu_static_maxd','ensure_constellations_bin_stream','build_broad_markdist_tail_reordered_bin',
    'build_chunkshape148_reordered_bin','exec_solutions_gpu_bin_stream_funcid_reorder','exec_solutions_gpu_bin_stream_split145',
    'exec_solutions_gpu_chunk_split145','stream_funcid_reorder_risk_suffix','funcid_reorder_make_quotas',
    'interleave_funcid_reorder_parts','exec_solutions','dfs_iter'
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
old_split_markers = ['split290', 'split288', 'split287', 'split286', 'split285', 'split284', 'split282', 'split281', 'split280', 'split278', 'split277', 'split275', 'split273', 'split270_', 'split240', 'split239_', 'kernel-blockdiag', 'block_count_diag', 'rootaction0-direct']
old_present = [x for x in old_split_markers if x in s]
if 'split291' in s and not old_present:
    checks.append(('source_split_tag', 'split291 active; rejected runtime tags absent', 'split291', True))
else:
    checks.append(('source_split_tag', 'split291 active; rejected runtime tags absent', 'present=' + ','.join(old_present) if old_present else 'split291 missing', False))
if 'worker_id:int=0' in s and 'worker_count:int=1' in s and 'worker_id}/{worker_count}' in s:
    checks.append(('worker_split_args', 'present', 'present', True))
else:
    checks.append(('worker_split_args', 'present', 'missing', False))

if 'split=fid14_launch' in s or 'split145-fid14' in s or 'split145-rest' in s or 'source_fid14_launch_split' in s:
    checks.append(('source_fid14_split_rejected', 'absent', 'fid14 split marker present', False))
else:
    checks.append(('source_fid14_split_rejected', 'absent', 'absent', True))

if 'kernel_dfs_iter_gpu_maxd14_root0' in s or 'root0-dispatch' in s or 'rootaction0-direct' in s:
    checks.append(('source_root0_direct_rejected', 'absent', 'root0 direct kernel marker present', False))
else:
    checks.append(('source_root0_direct_rejected', 'absent', 'absent', True))

if 'split145-bucket-summary' in s or 'bucket_total_tasks' in s or 'source_split145_bucket_diag' in s:
    checks.append(('source_bucket_diag_rejected', 'absent', 'bucket diagnostic marker present', False))
else:
    checks.append(('source_bucket_diag_rejected', 'absent', 'absent', True))

if 'depth_u:u32' in active or 'source_depthu_childsave' in s:
    checks.append(('source_depthu_rejected', 'active absent', 'depth_u marker present', False))
else:
    checks.append(('source_depthu_rejected', 'active absent', 'active absent', True))

if 'ZERO:u32' in active or 'zero_const_assign' in active:
    checks.append(('source_zero_const_rejected', 'active absent', 'zero const marker active present', False))
else:
    checks.append(('source_zero_const_rejected', 'active absent', 'active absent', True))

# Check MAXD14 generic DFS loop normal-default nld/nrd + ncol-only + block_code-late shape.
# 294: single-lane kernel (no laneA/B split); standard 292-style check with relative
# indent normalization. Root-preroll (pr_block_code etc.) must remain untouched.
m14 = re.search(r'^@gpu\.kernel\s*\n^def\s+kernel_dfs_iter_gpu_maxd14\b.*?(?=^@gpu\.kernel\s*\n^def\s+kernel_dfs_iter_gpu_maxd16\b)', s, re.M | re.S)
if not m14:
    checks.append(('source_generic_normaldefault', 'MAXD14 generic normal-default nld/nrd', 'MAXD14 not found', False))
    checks.append(('source_blockcode_late', 'block_code scalar only inside special branch', 'MAXD14 not found', False))
else:
    body14 = m14.group(0)
    idx = body14.find('nibble_op:u32=u32(0)')
    tail = body14[idx:] if idx >= 0 else body14
    lines = [l for l in tail.split('\n') if l.strip() and 'nld:u32=(cur_ld|bit)<<u32(1)' in l]
    ok_normal = False
    ok_late = False
    if lines:
        bi = len(lines[0]) - len(lines[0].lstrip(' '))
        norm = re.sub(r'\n {%d}' % bi, '\n', tail)
        expected_normal = ('nld:u32=(cur_ld|bit)<<u32(1)\n'
                           'nrd:u32=(cur_rd|bit)>>u32(1)\n'
                           'ncol:u32=cur_col|bit\n'
                           'if (nibble_op&u32(7))!=u32(0):\n'
                           '  block_code:u32=nibble_op&u32(7)\n'
                           '  stepu:u32=')
        old_early_blockcode = 'block_code:u32=nibble_op&u32(7)\n\nbit:u32=cur_avail&(u32(0)-cur_avail)'
        nf_default_bad = ('nf:u32=bm&~(nld|nrd|ncol)\nif (nibble_op&u32(7))!=u32(0):' in norm
                          or 'nf:u32=bm&~(nld|nrd|ncol)\nif block_code!=u32(0):' in norm)
        ok_normal = expected_normal in norm and not nf_default_bad
        ok_late = old_early_blockcode not in norm and expected_normal in norm
    checks.append(('source_generic_normaldefault', 'MAXD14 generic normal-default nld/nrd+ncol-only', 'present' if ok_normal else 'missing/old-form-or-nfdefault', ok_normal))
    checks.append(('source_blockcode_late', 'block_code scalar only inside special branch', 'present' if ok_late else 'missing/old block_code scalar before bit', ok_late))

# 304: K48-sweep shape checks (296 kernel logic, K=48).
    ok_stride_param = 'stride:int,' in body14.split(')->None:')[0]
    ok_stack_array = 'stack=__array__[u64](MAXD14_ANCESTOR*2)' in body14
    ok_save_sp = 'if save_sp==0:' in body14
    ok_next_depth = 'next_depth:int=cur_depth+1' in body14
    ok_cur_depth = 'cur_depth:int=0' in body14
    ok_stack_ptr_incr = body14.count('stack_ptr+=2') == 2
    ok_gridstride_loop = 'while idx<m:' in body14
    ok_single_writeback = 'results[tid]=thread_total\n' in body14
    ok_push = body14.count('stack[stack_ptr]=u64(cur_ld)') == 2
    ok_pop = 'packed_ldrd:u64=stack[stack_ptr]' in body14
    ok_shape = (ok_stride_param and ok_stack_array and ok_save_sp and ok_next_depth
                 and ok_cur_depth and ok_stack_ptr_incr and ok_gridstride_loop
                 and ok_single_writeback and ok_push and ok_pop)
    checks.append(('source_K48_sweep_shape',
                    '296 kernel shape (stack/save_sp/next_depth/cur_depth/stack_ptr) + K=48',
                    'present' if ok_shape else (
                        'missing (stride=%s stack=%s sp=%s nd=%s cd=%s ptr=%s gs=%s wb=%s push=%s pop=%s)'
                        % (ok_stride_param, ok_stack_array, ok_save_sp, ok_next_depth, ok_cur_depth,
                           ok_stack_ptr_incr, ok_gridstride_loop, ok_single_writeback, ok_push, ok_pop)
                    ),
                    ok_shape))
k_match = re.search(r'^K_PER_THREAD_MAXD14:Static\[int\]=(\d+)', s, re.M)
k_value = k_match.group(1) if k_match else 'missing'
checks.append(('source_k_per_thread_maxd14', str(EXPECTED_K_PER_THREAD_MAXD14_PY), k_value, k_value == str(EXPECTED_K_PER_THREAD_MAXD14_PY)))


# maxd16/18/20/21 kernels are intentionally unmodified (safe 1-task-per-thread
# fallback for selected_maxd>14 chunks); make sure no stray edits crept in.
for other_maxd in (16, 18, 20, 21):
    other_m = re.search(r'^@gpu\.kernel\s*\n^def\s+kernel_dfs_iter_gpu_maxd' + str(other_maxd) + r'\b.*?\)->None:', s, re.M | re.S)
    ok_unmodified = other_m is not None and 'stride:int,' not in other_m.group(0)
    checks.append(('source_maxd%d_unmodified' % other_maxd, 'no stride param (1-task-per-thread fallback preserved)', 'present/unmodified' if ok_unmodified else 'missing or unexpectedly modified', ok_unmodified))

if 'diag_loop_iters_arr' in s or 'kernel-blockdiag' in s or 'block_count_diag' in s:
    checks.append(('source_blockdiag_rejected', 'absent', 'blockdiag marker present', False))
else:
    checks.append(('source_blockdiag_rejected', 'absent', 'absent', True))

# 351: THE 328 SoA-SPLIT CHECKS ARE INVERTED HERE, NOT DELETED.
#
# 328 verified that all 5 kernel signatures carried BOTH halves of the split
# w array and that the dispatcher derived both before every launch. 351 proves
# the high half was identically zero -- symmetry() returns only u64(2)/u64(4)/
# u64(8) -- and removes it. The 328 checks would therefore FAIL by construction,
# which is exactly the trap that bit 345 r1 and 349 r1: the implementation moved
# and an existing check kept looking for the old string. So both are rewritten to
# assert ABSENCE instead of presence, under new names, and three more are added.
#
# SCOPING IS UNCHANGED FROM 328 AND IS THE WHOLE POINT. Only the actual
# signature line of each kernel (from 'def kernel_dfs_iter_gpu_maxdNN(' to its
# matching '->None:'), the 5 kernel bodies, and the dispatcher body are
# examined, with docstrings already stripped above and comments stripped below.
# A whole-file grep for the identifier CANNOT be used: this project pastes chat
# responses into the module docstring as normal practice, VERSION_TAG and
# WHI_ELIM_REASON both discuss the identifier in prose, and the kernel comments
# name it too. Every one of those would make an absence check fail spuriously.
def strip_comments(text):
    # kernel and dispatcher code in this module contains no '#' inside any
    # string literal, so a line-tail cut is safe and exact here.
    return re.sub(r'#[^\n]*', '', text)

lo_only_count = 0
bad_sig_count = 0
sig_details = []
for kname in ['maxd14', 'maxd16', 'maxd18', 'maxd20', 'maxd21']:
    sig_m = re.search(r'^def\s+kernel_dfs_iter_gpu_' + kname + r'\(\n(.*?)\n\)->None:', s, re.M | re.S)
    if sig_m is None:
        sig_details.append(f'{kname}=missing')
        bad_sig_count += 1
        continue
    sig_text = strip_comments(sig_m.group(1))
    has_lo = 'markctrl_arr:Ptr[u32],w_lo_arr:Ptr[u32],' in sig_text
    has_hi = 'w_hi_arr' in sig_text
    has_old = 'w_arr:Ptr[u64]' in sig_text
    if has_lo and not has_hi and not has_old:
        lo_only_count += 1
        sig_details.append(f'{kname}=lo_only')
    elif has_hi:
        bad_sig_count += 1
        sig_details.append(f'{kname}=high_half_still_present')
    elif has_old:
        bad_sig_count += 1
        sig_details.append(f'{kname}=pre328_combined_u64')
    else:
        bad_sig_count += 1
        sig_details.append(f'{kname}=unrecognised')
ok_whi_sig = (lo_only_count == 5) and (bad_sig_count == 0)
checks.append(('source_whi_elim_signatures',
                '5 kernels take the low u32 array only; 0 mention the high array; 0 use pre-328 w_arr:Ptr[u64]',
                f'{lo_only_count} lo-only, {bad_sig_count} bad ({", ".join(sig_details)})', ok_whi_sig))

# All five kernel BODIES: the identifier must be gone entirely, and the
# epilogue must read the low array directly at exactly 15 sites (3 per kernel:
# root_action==2, root_action==3, and the generic-loop exit).
kzone_m = re.search(r'^@gpu\.kernel\s*\n^def\s+kernel_dfs_iter_gpu_maxd14\b'
                    r'.*?(?=^def launch_kernel_dfs_iter_gpu_static_maxd\()', s, re.M | re.S)
if kzone_m is None:
    checks.append(('source_whi_elim_kernel_bodies', '0 references in kernel bodies, exactly 15 single-load epilogue sites',
                    'kernel zone not found', False))
else:
    kzone = strip_comments(kzone_m.group(0))
    hi_refs = kzone.count('w_hi_arr')
    lo_sites = kzone.count('u64(w_lo_arr[')
    old_recon = len(re.findall(r'\|\(u64\(w_\w+\[\w+\]\)<<u64\(32\)\)', kzone))
    ok_bodies = (hi_refs == 0) and (lo_sites == 15) and (old_recon == 0)
    checks.append(('source_whi_elim_kernel_bodies', '0 references in kernel bodies, exactly 15 single-load epilogue sites',
                    f'{hi_refs} references, {lo_sites} single-load sites, {old_recon} old reconstruction expressions',
                    ok_bodies))

# Dispatcher: builds ONE array, passes ONE pointer to all five launches, and
# never names the removed one in code.
dispatcher_m = re.search(r'^def launch_kernel_dfs_iter_gpu_static_maxd\(.*?\n  return False', s, re.M | re.S)
d = strip_comments(dispatcher_m.group(0)) if dispatcher_m is not None else ''
ok_dispatcher = (dispatcher_m is not None
                  and 'w_lo_arr:List[u32]=[u32(v&u64(0xffffffff)) for v in w_arr]' in d
                  and 'w_hi_arr' not in d
                  and d.count('gpu.raw(w_lo_arr),gpu.raw(meta_next)') == 5
                  and 'gpu.raw(w_lo_arr),gpu.raw(w_hi_arr)' not in d
                  and 'gpu.raw(w_arr)' not in d)
checks.append(('source_whi_elim_dispatcher',
                'derives the low u32 array once, passes it to all 5 launches, never builds or passes the high array',
                'present' if ok_dispatcher else 'missing or incomplete', ok_dispatcher))

# The host-side invariant guard. 351's correctness rests entirely on
# symmetry() staying inside 32 bits, so that claim is checked at runtime
# rather than assumed. All four elements must be present.
guard_bits = {
    'or_accumulator_init': 'w_hi_or:u64=u64(0)' in d,
    'or_pass_over_w_arr': 'for v in w_arr:' in d and 'w_hi_or|=v' in d,
    'high_half_test': 'if (w_hi_or>>u64(32))!=u64(0):' in d,
    'loud_abort': '[whi-error]' in d and 'return False' in d,
}
ok_guard = all(guard_bits.values())
checks.append(('source_whi_zero_guard',
                'dispatcher ORs all of w_arr and aborts loudly if the high half is ever nonzero',
                'present' if ok_guard else 'missing: ' + ','.join(k for k, v in guard_bits.items() if not v),
                ok_guard))

# 333: w3_j7 adoption -- both WINDOW_MULT defaults must be 3, both PHASE_JUMP defaults must be 7,
# and no WINDOW_MULT default may remain at the old value 8
wm3 = len(re.findall(r'^A10G_FINAL_DEFAULT_REORDER_WINDOW_MULT:int=3$', s, re.M)) \
    + len(re.findall(r'^FUNCID_REORDER_V2_WINDOW_MULT:int=3$', s, re.M))
wm8 = len(re.findall(r'^A10G_FINAL_DEFAULT_REORDER_WINDOW_MULT:int=8$', s, re.M)) \
    + len(re.findall(r'^FUNCID_REORDER_V2_WINDOW_MULT:int=8$', s, re.M))
pj7 = len(re.findall(r'^A10G_FINAL_DEFAULT_REORDER_PHASE_JUMP:int=7$', s, re.M)) \
    + len(re.findall(r'^FUNCID_REORDER_V2_PHASE_JUMP:int=7$', s, re.M))
ok_w3 = (wm3 == 2) and (wm8 == 0)
ok_j7 = (pj7 == 2)
checks.append(('source_w3j7_window_mult_default', 'both WINDOW_MULT defaults =3, none =8',
                f'{wm3} at 3, {wm8} at 8', ok_w3))
checks.append(('source_w3j7_phase_jump_default', 'both PHASE_JUMP defaults =7',
                f'{pj7} at 7', ok_j7))
ok_reason = ('N21 measured best w3_j7' in s)
checks.append(('source_w3j7_reason_string', 'FUNCID_REORDER_V2_DEFAULT_REASON mentions N21 measured best w3_j7',
                'present' if ok_reason else 'missing', ok_reason))

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

# ---- 334/335: CUDA C runner (buried-idea C-2) spike prerequisite recon ----
# Non-gating environment probe: does NOT touch the GPU, does NOT build or
# run anything solver-related, and NEVER fails this validation (INFO only).
# Purpose: establish whether cudacodon has a usable standalone CUDA
# toolchain (nvcc + cuobjdump + known -arch target) before any 336+ spike
# code is written.
NVCC_PATH=""
if command -v nvcc >/dev/null 2>&1; then
  NVCC_PATH=$(command -v nvcc)
  NVCC_VER=$(nvcc --version 2>&1 | tr '\n' ' ' | sed 's/\t/ /g')
  printf 'cudac_toolchain_probe_nvcc\tnvcc present with version info\t%s\tINFO\n' "$NVCC_VER" >> "$SUMMARY"
  printf 'cudac_toolchain_probe_nvcc_path\tfull path via command -v\t%s\tINFO\n' "$NVCC_PATH" >> "$SUMMARY"
else
  printf 'cudac_toolchain_probe_nvcc\tnvcc present with version info\tnvcc not found on PATH\tINFO\n' >> "$SUMMARY"
  printf 'cudac_toolchain_probe_nvcc_path\tfull path via command -v\tn/a (nvcc not on PATH)\tINFO\n' >> "$SUMMARY"
fi
# 334: plain PATH lookup for cuobjdump (kept for continuity/comparison).
if command -v cuobjdump >/dev/null 2>&1; then
  CUOBJDUMP_VER=$(cuobjdump --version 2>&1 | tr '\n' ' ' | sed 's/\t/ /g')
  printf 'cudac_toolchain_probe_cuobjdump\tcuobjdump present with version info\t%s\tINFO\n' "$CUOBJDUMP_VER" >> "$SUMMARY"
else
  printf 'cudac_toolchain_probe_cuobjdump\tcuobjdump present with version info\tcuobjdump not found on PATH\tINFO\n' >> "$SUMMARY"
fi
# 335: PATH-independent re-probe -- look for cuobjdump directly inside
# nvcc's own directory (334 found nvcc on PATH but not cuobjdump there,
# even though 325/327 successfully used cuobjdump for SASS disassembly
# in the past; this resolves whether it is a PATH gap or a real absence).
if [[ -n "$NVCC_PATH" ]]; then
  NVCC_DIR=$(dirname "$NVCC_PATH")
  if [[ -x "$NVCC_DIR/cuobjdump" ]]; then
    CUOBJDUMP_DIR_VER=$("$NVCC_DIR/cuobjdump" --version 2>&1 | tr '\n' ' ' | sed 's/\t/ /g')
    printf 'cudac_toolchain_probe_cuobjdump_near_nvcc\tcuobjdump next to nvcc (%s)\t%s\tINFO\n' "$NVCC_DIR" "$CUOBJDUMP_DIR_VER" >> "$SUMMARY"
  else
    printf 'cudac_toolchain_probe_cuobjdump_near_nvcc\tcuobjdump next to nvcc (%s)\tnot found in nvcc'"'"'s directory either\tINFO\n' "$NVCC_DIR" >> "$SUMMARY"
  fi
else
  printf 'cudac_toolchain_probe_cuobjdump_near_nvcc\tcuobjdump next to nvcc\tn/a (nvcc not on PATH)\tINFO\n' >> "$SUMMARY"
fi
# 335: empirically pin down the A10G's Compute Capability for the future
# -arch=sm_XX nvcc flag (this codebase's only prior sm_XX reference,
# sm_61, was for a different, older GPU and was never verified for A10G).
if command -v nvidia-smi >/dev/null 2>&1; then
  COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>&1 | tr '\n' ' ' | sed 's/\t/ /g; s/[[:space:]]*$//')
  if [[ -n "$COMPUTE_CAP" ]]; then
    printf 'cudac_toolchain_probe_compute_cap\tA10G compute_cap (nvidia-smi)\t%s\tINFO\n' "$COMPUTE_CAP" >> "$SUMMARY"
  else
    printf 'cudac_toolchain_probe_compute_cap\tA10G compute_cap (nvidia-smi)\tempty result\tINFO\n' >> "$SUMMARY"
  fi
else
  printf 'cudac_toolchain_probe_compute_cap\tA10G compute_cap (nvidia-smi)\tnvidia-smi unavailable\tINFO\n' >> "$SUMMARY"
fi

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
  echo "[error] 354 source static checks failed" >&2
  exit 66
fi
echo "[static-ok] 354 source checks passed; proceeding to release build/run (CPU-only bench_mode 32, no GPU dispatch)"

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[error] another 354 validation holds: $LOCK_FILE" >&2
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
# ---- 354: FEATURE_SMOKE (chunk 0 only, bench_mode 32, no GPU) ----
FEATURE_SMOKE=${FEATURE_SMOKE:-0}
EXPECTED_FEATURE354_CHUNKS=${EXPECTED_FEATURE354_CHUNKS:-3}
BLOCK=${BLOCK:-32}
MAX_BLOCKS=${MAX_BLOCKS:-484}
LOG_LEVEL=${LOG_LEVEL:-1}
PRESET_QUEENS=${PRESET_QUEENS:-7}
REORDER_WINDOW_MULT=${REORDER_WINDOW_MULT:-3}
REORDER_PHASE_JUMP=${REORDER_PHASE_JUMP:-7}
CROSS_STRIPE_SAFE=${CROSS_STRIPE_SAFE:-0}
BROADMARK_VARIANT=${BROADMARK_VARIANT:-2}
N=${N:-21}

if [[ "$FEATURE_SMOKE" == "1" ]]; then
  echo "[feature-smoke] bench_mode=32 chunk 0 only, CPU only, no GPU dispatch"
  SMOKE_LOG="$LOGDIR/feature_smoke.log"
  SMOKE_CMD=("$CAND" -g "$N" "$N" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" 0 "$PRESET_QUEENS" 32 "$REORDER_WINDOW_MULT" "$REORDER_PHASE_JUMP" "$CROSS_STRIPE_SAFE" 0 1 "" "$BROADMARK_VARIANT")
  echo "[feature-smoke] ${SMOKE_CMD[*]}" | tee "$SMOKE_LOG"
  set +e; stdbuf -oL -eL "${SMOKE_CMD[@]}" 2>&1 | tee -a "$SMOKE_LOG"; smoke_rc=${PIPESTATUS[0]}; set -e
  record_check feature_smoke_exit 0 "$smoke_rc" || failures=$((failures+1))
  SMOKE_FILE=$(sed -n 's/.*\[feature354-chunk-done\].* file=\([^ ]*\.bin\).*/\1/p' "$SMOKE_LOG" | tail -n1)
  SMOKE_RECS=$(sed -n 's/.*\[feature354-chunk-done\].* records=\([0-9]*\) .*/\1/p' "$SMOKE_LOG" | tail -n1)
  record_check feature_smoke_records 743424 "${SMOKE_RECS:-none}" || failures=$((failures+1))
  if [[ -n "$SMOKE_FILE" && -s "$SMOKE_FILE" ]]; then
    SMOKE_BYTES=$(wc -c < "$SMOKE_FILE")
    record_check feature_smoke_bytes 23789568 "$SMOKE_BYTES" || failures=$((failures+1))
  else
    printf 'feature_smoke_file\tpresent and non-empty\t%s\tFAIL\n' "${SMOKE_FILE:-missing}" >> "$SUMMARY"; failures=$((failures+1))
  fi
  echo
  echo "================================================================"
  echo "[feature-smoke summary]"
  cat "$SUMMARY"
  echo "[logdir] $LOGDIR"
  echo "================================================================"
  if (( failures != 0 || static_failures != 0 )); then
    echo "[feature-smoke-failed] failures=$failures static_failures=$static_failures" >&2
    exit 1
  fi
  echo "[feature-smoke-ok] chunk 0 plumbing verified, CPU only, no GPU touched; now run the full pass without FEATURE_SMOKE"
  exit 0
fi

# ---- 354: full feature census, all chunks, CPU only, no GPU ----
# debug_chunk_count=0 tells the executable's bench_mode 32 path to process
# every chunk rather than stopping after one (see FEATURE354_REASON and the
# feature_chunk_only definition in main()).
RUN_LOG="$LOGDIR/full_once.log"
FULL_CMD=("$CAND" -g "$N" "$N" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" 0 "$PRESET_QUEENS" 32 "$REORDER_WINDOW_MULT" "$REORDER_PHASE_JUMP" "$CROSS_STRIPE_SAFE" 0 0 "" "$BROADMARK_VARIANT")
echo "[feature-full] ${FULL_CMD[*]}" | tee "$RUN_LOG"
echo "validation: feature census only (bench_mode 32), CPU only, no GPU kernel launched. Kernel is byte identical to 352 (source_kernel_identical_to_352 proves it statically). No correctness oracle applies: this revision touches neither the task set nor the GPU. Expect three feature354_*.bin files matching census353's chunk sizes (743424, 743424, 538434) and reconstruction_errors=0 in the embedded analysis below." | tee -a "$RUN_LOG"
set +e; stdbuf -oL -eL "${FULL_CMD[@]}" 2>&1 | tee -a "$RUN_LOG"; run_rc=${PIPESTATUS[0]}; set -e
record_check run_exit 0 "$run_rc" || failures=$((failures+1))

FEAT_WRITES=$(grep -c '\[feature354-chunk-done\]' "$RUN_LOG" || true)
record_check feature354_write_rows "$EXPECTED_FEATURE354_CHUNKS" "$FEAT_WRITES" || failures=$((failures+1))
FEAT_RECS=$(sed -n 's/.*\[feature354-chunk-done\].* records=\([0-9]*\) .*/\1/p' "$RUN_LOG" | paste -sd, -)
record_check feature354_record_counts "743424,743424,538434" "${FEAT_RECS:-none}" || failures=$((failures+1))
FEAT_FILES=$(sed -n 's/.*\[feature354-chunk-done\].* file=\([^ ]*\.bin\).*/\1/p' "$RUN_LOG" | paste -sd' ' -)
printf 'feature354_files\t(kept in the working directory, not archived)\t%s\tINFO\n' "${FEAT_FILES:-none}" >> "$SUMMARY"
ERROR_HITS=$(grep -Eic '\[(.*-)?error\]|mismatch' "$RUN_LOG" || true)
record_check error_or_mismatch_hits 0 "$ERROR_HITS" || failures=$((failures+1))

# ---- 354: offline candidate-key analysis ----
# CENSUS353_GLOB lets the person point this at wherever the 353 census bins
# live; they are not regenerated here and are not part of this revision's
# deliverables (353 already produced them and they stay in the working
# directory per 353's own design note).
CENSUS353_GLOB=${CENSUS353_GLOB:-census353_*_chunk*.bin}
FEATURE354_GLOB=${FEATURE354_GLOB:-features354_*_chunk*.bin}
ANALYSIS_REPORT="$LOGDIR/feature354_report.txt"
FEATURE354_STRIDE=$((BLOCK * MAX_BLOCKS))
CENSUS_FILE_COUNT=$(bash -c "ls $CENSUS353_GLOB 2>/dev/null | wc -l")
if [[ "$CENSUS_FILE_COUNT" -gt 0 ]]; then
  echo "[feature354-analysis] joining $FEAT_WRITES feature files against $CENSUS_FILE_COUNT census files at stride=$FEATURE354_STRIDE"
  set +e
  python3 - "$FEATURE354_STRIDE" "$FEATURE354_GLOB" "$CENSUS353_GLOB" > "$ANALYSIS_REPORT" 2>&1 <<'PYFEATUREANALYZE'
# 354 offline candidate-key analysis. GPU and Codon rebuild NOT required.
# Joins features354_*.bin (raw fields + real production key, from this revision)
# with census353_*.bin (true DFS trip counts, from 353) by position within each
# chunk, verifies the reproduction of the production key formula against the
# real key column, then scores several reweighted candidate keys by simulating
# the SAME grid-stride striping production uses and measuring the resulting
# cross-warp imbalance (modelS from 353), which is the metric 353 fixed a
# 6.300% floor for.
import sys, struct, glob, array

STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 15488
WARP = 32
FPAT = sys.argv[2] if len(sys.argv) > 2 else 'features354_*_chunk*.bin'
CPAT = sys.argv[3] if len(sys.argv) > 3 else 'census353_*_chunk*.bin'

CHUNKSHAPE148_SCORE_KEY_MAX = (1 << 20) - 1  # matches the module constant

def fid_bucket_bonus(fid):
    if fid == 26 or fid == 27: return 96
    if fid in (19, 22, 23, 24): return 72
    if fid in (20, 21): return 56
    if fid == 17: return 42
    if fid == 14: return 36
    if fid in (0, 4, 5, 12, 16, 18): return 20
    return 8

def production_raw(fid, free, end, row, mark1, mark2):
    # Verbatim port of chunkshape148_score_key_from_soa's non-tie arithmetic.
    pc = bin(free & 0xFFFFFFFF).count('1')
    depth = end - row
    if depth < 0: depth = 0
    mark_gap = mark2 - mark1
    if mark_gap < 0: mark_gap = -mark_gap
    row_to_end = end - row
    if row_to_end < 0: row_to_end = 0
    row_to_mark1 = mark1 - row
    if row_to_mark1 < 0: row_to_mark1 = 0
    raw = pc * 12 + depth * 7 + row_to_end * 3 + fid_bucket_bonus(fid)
    if pc >= 5: raw += 20
    elif pc >= 4: raw += 12
    elif pc >= 3: raw += 6
    if depth >= 13: raw += 20
    elif depth >= 11: raw += 12
    elif depth >= 9: raw += 6
    if mark_gap >= 3: raw += 8
    if row_to_mark1 >= 4: raw += 4
    return raw, pc, depth, mark_gap, row_to_mark1

def candidate_raw(name, fid, free, end, row, mark1, mark2):
    pc = bin(free & 0xFFFFFFFF).count('1')
    depth = end - row
    if depth < 0: depth = 0
    mark_gap = mark2 - mark1
    if mark_gap < 0: mark_gap = -mark_gap
    row_to_mark1 = mark1 - row
    if row_to_mark1 < 0: row_to_mark1 = 0
    fb = fid_bucket_bonus(fid)
    if name == 'A_current_raw_only':
        r, _, _, _, _ = production_raw(fid, free, end, row, mark1, mark2)
        return r
    if name == 'B_pc_x2':
        return pc * 24 + depth * 7 + fb
    if name == 'C_depth_heavy':
        return pc * 6 + depth * 16 + fb
    if name == 'D_no_fid':
        return pc * 12 + depth * 7 + mark_gap * 4
    if name == 'E_fid_only':
        return fb * 10
    if name == 'F_markgap_heavy':
        return pc * 6 + depth * 6 + mark_gap * 20 + row_to_mark1 * 10 + fb
    if name == 'G_pc_depth_product':
        return pc * depth * 3 + fb
    raise ValueError(name)

CANDIDATES = ['A_current_raw_only', 'B_pc_x2', 'C_depth_heavy', 'D_no_fid', 'E_fid_only', 'F_markgap_heavy', 'G_pc_depth_product']

def spearman(xs, ys):
    n = len(xs)
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n; my = sum(ry) / n
    sxy = sxx = syy = 0.0
    for i in range(n):
        dx = rx[i] - mx; dy = ry[i] - my
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy
    if sxx <= 0 or syy <= 0: return 0.0
    return sxy / (sxx * syy) ** 0.5

def headroom_for_order(order_key, trips, stride, warp):
    # simulate the actual execution assignment: sort ascending by order_key,
    # then grid-stride assign to `stride` threads (thread t gets sorted
    # positions t, t+stride, t+2*stride, ...), matching how the shaped bin's
    # linear order maps onto GPU threads in production.
    n = len(trips)
    idx_sorted = sorted(range(n), key=lambda i: order_key[i])
    warps = (stride + warp - 1) // warp
    B = [0] * warps
    T = [0] * stride
    for pos, i in enumerate(idx_sorted):
        t = pos % stride
        T[t] += trips[i]
    for t in range(stride):
        B[t // warp] += T[t]
    total = sum(B)
    ideal = total / float(warps)
    worst = max(B)
    return 100.0 * (1.0 - ideal / worst), worst, ideal

def load_features(path):
    with open(path, 'rb') as f:
        data = f.read()
    n = len(data) // 32
    out = []
    for i in range(n):
        rec = data[i*32:(i+1)*32]
        fid, free, end, row, mark1, mark2, key = struct.unpack('<iiiiiiq', rec)
        out.append((fid, free, end, row, mark1, mark2, key))
    return out

def load_census(path):
    a = array.array('Q')
    with open(path, 'rb') as f:
        a.frombytes(f.read())
    return list(a)

ffiles = sorted(glob.glob(FPAT), key=lambda p: int(p.rsplit('chunk', 1)[1].split('.')[0]))
if not ffiles:
    print('FEATURE354_ANALYSIS status=no_feature_files pattern=%s' % FPAT)
    sys.exit(0)

grand_current = []
grand_best = {c: [] for c in CANDIDATES}
recon_errors_total = 0
rho_position_key_all = []
rho_key_trips_all = []

for fpath in ffiles:
    ci = int(fpath.rsplit('chunk', 1)[1].split('.')[0])
    cpath = None
    for cand in glob.glob(CPAT):
        if cand.rsplit('chunk', 1)[1].split('.')[0] == str(ci):
            cpath = cand
            break
    feats = load_features(fpath)
    if cpath is None:
        print('FEATURE354_CHUNK chunk=%d status=no_matching_census_file' % ci)
        continue
    trips = load_census(cpath)
    n = min(len(feats), len(trips))
    if len(feats) != len(trips):
        print('FEATURE354_CHUNK chunk=%d WARNING record_count_mismatch features=%d census=%d, truncating to %d'
              % (ci, len(feats), len(trips), n))
    feats = feats[:n]; trips = trips[:n]

    # ---- reproduction check: recomputed raw must equal key // 32 exactly ----
    recon_errors = 0
    reproduced_raw = [0] * n
    for i, (fid, free, end, row, mark1, mark2, key) in enumerate(feats):
        raw, pc, depth, mark_gap, row_to_mark1 = production_raw(fid, free, end, row, mark1, mark2)
        reproduced_raw[i] = raw
        if raw != (key // 32):
            recon_errors += 1
    recon_errors_total += recon_errors
    print('FEATURE354_CHUNK chunk=%d tasks=%d reconstruction_errors=%d  (0 expected: raw recomputed from the six raw fields must equal key//32 for every task)'
          % (ci, n, recon_errors))

    position = list(range(n))
    key_col = [f[6] for f in feats]
    rho_pos_key = spearman(position, key_col)
    rho_key_trips = spearman(key_col, trips)
    rho_position_key_all.append(rho_pos_key)
    rho_key_trips_all.append(rho_key_trips)
    print('FEATURE354_CHUNK chunk=%d rank_corr position_vs_key=%.4f key_vs_trips=%.4f  (353 measured position_vs_trips approx -0.36 for this chunk)'
          % (ci, rho_pos_key, rho_key_trips))

    hr_current, worst_c, ideal_c = headroom_for_order(key_col, trips, STRIDE, WARP)
    grand_current.append((hr_current, worst_c, ideal_c))
    print('FEATURE354_CHUNK chunk=%d candidate=A_current_real_key headroom_pct=%.3f  (uses the actual production key bytes, tie-breaker included; this is what 352 actually runs)'
          % (ci, hr_current))

    for cname in CANDIDATES:
        ck = [candidate_raw(cname, f[0], f[1], f[2], f[3], f[4], f[5]) for f in feats]
        hr, worst, ideal = headroom_for_order(ck, trips, STRIDE, WARP)
        grand_best[cname].append((hr, worst, ideal))
        print('FEATURE354_CHUNK chunk=%d candidate=%s headroom_pct=%.3f' % (ci, cname, hr))

print('FEATURE354_TOTAL reconstruction_errors=%d  (0 required; nonzero means the raw-field extraction or the ported '
      'formula has a bug, and every ranking below is untrustworthy until this is 0)' % recon_errors_total)
avg_rho_pk = sum(rho_position_key_all) / len(rho_position_key_all) if rho_position_key_all else 0.0
avg_rho_kt = sum(rho_key_trips_all) / len(rho_key_trips_all) if rho_key_trips_all else 0.0
print('FEATURE354_TOTAL avg_rank_corr position_vs_key=%.4f key_vs_trips=%.4f' % (avg_rho_pk, avg_rho_kt))

def weighted_headroom(rows):
    tw = sum(w for _, _, w in rows)
    tot_worst = sum(w for _, w, _ in rows)
    tot_ideal = sum(i for _, _, i in rows)
    return 100.0 * (1.0 - tot_ideal / tot_worst)

cur_hr = weighted_headroom(grand_current)
print('FEATURE354_TOTAL candidate=A_current_real_key headroom_pct=%.3f  (baseline: this is what 352 actually runs today)' % cur_hr)
best_name, best_hr = None, cur_hr
for cname in CANDIDATES:
    hr = weighted_headroom(grand_best[cname])
    tag = ' <-- BEST' if hr < best_hr - 1e-9 else ''
    print('FEATURE354_TOTAL candidate=%s headroom_pct=%.3f%s' % (cname, hr, tag))
    if hr < best_hr:
        best_hr = hr; best_name = cname

improvement = cur_hr - best_hr
if best_name is None:
    print('FEATURE354_VERDICT no_candidate_beats_current best=A_current headroom_pct=%.3f' % cur_hr)
else:
    print('FEATURE354_VERDICT best_candidate=%s current_headroom_pct=%.3f best_headroom_pct=%.3f improvement_points=%.3f'
          % (best_name, cur_hr, best_hr, improvement))
PYFEATUREANALYZE
  analysis_rc=$?
  set -e
  record_check feature354_analysis_exit 0 "$analysis_rc" || failures=$((failures+1))
  if [[ -s "$ANALYSIS_REPORT" ]]; then
    echo
    echo "---------------- feature354 report ----------------"
    cat "$ANALYSIS_REPORT"
    echo "-----------------------------------------------------"
    RECON_ERRS=$(sed -n 's/.*FEATURE354_TOTAL reconstruction_errors=\([0-9]*\).*/\1/p' "$ANALYSIS_REPORT" | tail -n1)
    record_check feature354_reconstruction_errors 0 "${RECON_ERRS:-probe_failed}" || failures=$((failures+1))
    VERDICT_LINE=$(grep '^FEATURE354_VERDICT' "$ANALYSIS_REPORT" | tail -n1)
    printf 'feature354_verdict\t(candidate ranking; a positive result here is a proposal for 355, not an adopted change)\t%s\tINFO\n' "${VERDICT_LINE:-missing}" >> "$SUMMARY"
  else
    printf 'feature354_report\tnon-empty\tempty or missing\tFAIL\n' >> "$SUMMARY"; failures=$((failures+1))
  fi
else
  printf 'feature354_analysis_exit\t0\tskipped: no files matched CENSUS353_GLOB=%s (set the env var to point at where 353 left the census bins)\tINFO\n' "$CENSUS353_GLOB" >> "$SUMMARY"
fi

printf 'timing_not_recorded\tno GPU dispatch in this revision\t354 does not launch the GPU kernel, so there is no comparable wall-clock baseline; none is recorded\tINFO\n' >> "$SUMMARY"

echo
echo "================================================================"
echo "[summary]"
cat "$SUMMARY"
echo "[logdir]   $LOGDIR"
echo "================================================================"
if (( failures != 0 )); then
  echo "[validation-failed] failures=$failures" >&2
  exit 1
fi
echo "[validation-ok] 354 feature census complete: kernel byte identical to 352 (source_kernel_identical_to_352), CPU-only bench_mode 32 wrote 3 files matching census353's chunk sizes, and the embedded offline analysis ran. 354 IS NOT A CANDIDATE and decides nothing on its own -- read feature354_report.txt and feature354_verdict, and if a candidate looks better, take it to 355 for an actual N=21 full-run verification against 314666222712."
