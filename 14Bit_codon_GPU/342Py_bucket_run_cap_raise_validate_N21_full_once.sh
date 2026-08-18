#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# 342 N=21 full validation harness (bucket-run-cap-raise)
#
# 342's ONLY source change is one constant: CHUNKSHAPE148_BUCKET_RUN_MAX
# raised from 256 to 8192. Kernels, dispatcher, K=48, SoA w_lo/w_hi, w3_j7,
# the 339 run-length emit loop and the source default
# CHUNKSHAPE148_BUCKET_RUN=1 are all byte-for-byte identical to 341/340/339.
# Raising the cap cannot change behaviour for any RUN <= 256, so every
# shaped-bin cache built in 339-341 remains valid and reusable.
#
# 340+341 RESULTS (in-session RUN=1 control = 450.183s; every point correct
# at 314666222712, tasks 2025282, cache_state=reuse, zero failures; session
# drift +0.0161% in 340 and +0.0027% in 341):
#
#   RUN   x32   2^n   elapsed     vs RUN=1    chunk0    chunk1    chunk2
#     1    -     -    450.183s    +0.000%     (ref)     (ref)     (ref)
#    16    NO    yes  448.931s    -0.278%
#    32    yes   yes  447.141s    -0.676%     +0.276%   -0.744%   -1.850%
#    48    NO    -    448.995s    -0.264%
#    64    yes   yes  446.436s    -0.832%     +0.280%   -0.738%   -2.471%
#    80    NO    -    448.671s    -0.336%     +0.273%   -0.738%   -0.597%
#    96    yes   -    447.443s    -0.609%     +0.770%   -0.903%   -2.078%
#   128    yes   yes  446.502s    -0.818%     +0.273%   -0.742%   -2.404%
#   256    yes   yes  439.835s    -2.299%     -2.333%   -2.449%   -1.930%
#    64    yes   yes  446.448s    -0.830%     +0.271%   -0.726%   -2.484%
#
# TWO FINDINGS:
#   1. The warp-alignment rule from 340 holds at larger magnitude. The
#      non-aligned control RUN=80 landed at -0.336%, with RUN=16 and RUN=48.
#      The pre-registered condition for abandoning the model did not fire.
#   2. RUN=256 is an outlier by a wide margin (-2.299%, nearly 3x the best
#      previous point) AND it works differently. From 32 through 128 the
#      per-chunk figures were frozen at about +0.27% (chunk0) and -0.74%
#      (chunk1) and ALL variation sat in chunk2 -- the knob only ever moved
#      chunk2. At 256, chunk0 and chunk1 moved for the first time (-2.333%,
#      -2.449%) while chunk2 gave a little back. A second mechanism switches
#      on at 256; it is not yet identified (L2 locality is the leading
#      candidate, untested).
#
# THE PROBLEM 342 FIXES: 256 WAS the ceiling. 341 stopped measuring exactly
# where the interesting behaviour started.
#
# THIS SCRIPT is the ordinary single-point validate harness; its default is
# now CHUNKSHAPE148_BUCKET_RUN=256 (341's best). The 342 experiment is the
# sweep:
#
#     bash 342_bucket_run_cap_raise_driver.sh
#
# 342 SWEEP POINTS (default: 256 512 1024 2048 384 256):
#   256   anchor, measured first AND last so session drift stays visible
#   512, 1024, 2048   powers of two -- where does the curve peak or saturate?
#   384 = 12x32   warp-aligned but NOT a power of two. RUN=96 (also aligned,
#         not a power of two) underperformed its neighbours 64 and 128, so
#         being a power of two may matter on top of being a multiple of 32.
#         384 tests that at a larger magnitude.
#   Optional: add 4096. The per-iteration bucket quota is about
#         STEPS/8 = 15488/8 = 1936, and a run is truncated by the quota, so
#         2048 and 4096 SHOULD be indistinguishable. If they differ by more
#         than drift, the quota-saturation model is wrong.
#         SWEEP="256 512 1024 2048 4096 384 256"
#
# ADOPTION IS DEFERRED TO 343. It was planned for 342, but a best point
# sitting exactly on an artificial ceiling is not an optimum.
#
# MANDATORY VALIDATION ORDER:
#   1. STATIC_ONLY=1 first. Two new gating checks cover the raised cap.
#   2. Full N=21 run; confirm 314666222712 FIRST, before any timing.
#   3. Then the sweep via 342_bucket_run_cap_raise_driver.sh.
# EXPECTED_CHUNKS=3, K=48 (unchanged).
# =============================================================================

SRC=${SRC:-./342Py_bucket_run_cap_raise.py}
CAND=${CAND:-./342Py_bucket_run_cap_raise}
AUTO_BUILD=${AUTO_BUILD:-1}
# 274: visible startup + force release rebuild by default so a previous non-release
# `codon build` executable is not reused and cannot trigger CUDA_ERROR_INVALID_PTX.
# 274: print start/status early and prevent static pycheck from failing silently under set -e.
FORCE_REBUILD=${FORCE_REBUILD:-1}
STATIC_ONLY=${STATIC_ONLY:-0}
LOG_ROOT=${LOG_ROOT:-.}
LOCK_FILE=${LOCK_FILE:-/tmp/342Py_bucket_run_cap_raise_N21_full.lock}
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
CHUNKSHAPE148_BUCKET_RUN=${CHUNKSHAPE148_BUCKET_RUN:-256}
EXPECTED_CHUNKSHAPE148_BUCKET_RUN=$CHUNKSHAPE148_BUCKET_RUN
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
LOGDIR="${LOG_ROOT%/}/342Py_bucket_run_cap_raise_logs_N21_full_once_${TS}"
RUN_LOG="$LOGDIR/full_once.log"
BUILD_LOG="$LOGDIR/build.log"
SUMMARY="$LOGDIR/summary.tsv"
PROGRESS_COPY="$LOGDIR/progress_full.tsv"
mkdir -p "$LOGDIR"
printf 'check\texpected\tactual\tstatus\n' > "$SUMMARY"

echo "[start] 342 bucket-run-cap-raise validation script"
echo "[source] $SRC"
echo "[candidate] $CAND"
echo "[logdir] $LOGDIR"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo "[abort] rc=$rc logdir=${LOGDIR:-unknown}" >&2; fi' EXIT
echo "[validation-start] 342 bucket-run-cap-raise SRC=$SRC CAND=$CAND STATIC_ONLY=$STATIC_ONLY FORCE_REBUILD=$FORCE_REBUILD LOGDIR=$LOGDIR"

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
if grep -q '342 bucket-run-cap-raise' "$SRC"; then
  printf 'source_version_tag	342 bucket-run-cap-raise	present	OK
' >> "$SUMMARY"
else
  printf 'source_version_tag	342 bucket-run-cap-raise	missing	FAIL
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
CAP_NEW=$(grep -c '^CHUNKSHAPE148_BUCKET_RUN_MAX:int=8192$' "$SRC" || true)
record_check source_chunkshape148_bucket_run_max_8192 1 "$CAP_NEW" || static_failures=$((static_failures+1))
CAP_OLD=$(grep -c '^CHUNKSHAPE148_BUCKET_RUN_MAX:int=256$' "$SRC" || true)
record_check source_chunkshape148_bucket_run_max_old_absent 0 "$CAP_OLD" || static_failures=$((static_failures+1))

# ---- 339: chunkshape148 bucket-run knob (new gating checks) ----
BR_DEFAULT=$(grep -c '^CHUNKSHAPE148_BUCKET_RUN:int=1$' "$SRC" || true)
record_check source_chunkshape148_bucket_run_default 1 "$BR_DEFAULT" || static_failures=$((static_failures+1))
if grep -q 'def chunkshape148_bucket_run_tag()' "$SRC" && grep -q '{chunkshape148_bucket_run_tag()}.bin' "$SRC"; then
  printf 'source_chunkshape148_bucket_run_fname_tag\tpresent (run=1 keeps the 276-338 filename)\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_chunkshape148_bucket_run_fname_tag\tpresent (run=1 keeps the 276-338 filename)\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q 'bucket_run:int=chunkshape148_bucket_run_value()' "$SRC" && grep -q 'while rep<bucket_run:' "$SRC"; then
  printf 'source_chunkshape148_bucket_run_emit_loop\trun-length loop present in build_chunkshape148_reordered_bin\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_chunkshape148_bucket_run_emit_loop\trun-length loop present in build_chunkshape148_reordered_bin\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
fi
if grep -q 'chunkshape148_bucket_run=int(sys.argv\[16\])' "$SRC" && grep -q '^  CHUNKSHAPE148_BUCKET_RUN=chunkshape148_bucket_run$' "$SRC"; then
  printf 'source_chunkshape148_bucket_run_argv\targv[16] parsed and assigned to the global\tpresent\tOK\n' >> "$SUMMARY"
else
  printf 'source_chunkshape148_bucket_run_argv\targv[16] parsed and assigned to the global\tmissing\tFAIL\n' >> "$SUMMARY"; static_failures=$((static_failures+1))
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

# 328: verify the w_arr SoA split (w_lo_arr/w_hi_arr) is present in all
# 5 kernel signatures, that no kernel still declares the old combined
# w_arr:Ptr[u64] parameter, and that the dispatcher derives w_lo_arr/
# w_hi_arr from w_arr before every launch.
# 328 (redesigned after a false-positive caused by prose in this file's
# own docstring/chat-log content matching the old whole-file substring
# count): check ONLY the actual signature line of each of the 5 kernels
# (from 'def kernel_dfs_iter_gpu_maxdNN(' to its matching '->None:'),
# never the surrounding docstring/comment text. This makes the check
# immune to any prose -- including pasted chat log content, which is
# this project's normal documentation practice -- that happens to
# mention the old or new type syntax in passing.
soa_sig_count = 0
old_sig_count = 0
sig_details = []
for kname in ['maxd14', 'maxd16', 'maxd18', 'maxd20', 'maxd21']:
    sig_m = re.search(r'^def\s+kernel_dfs_iter_gpu_' + kname + r'\(\n(.*?)\n\)->None:', s, re.M | re.S)
    if sig_m is None:
        sig_details.append(f'{kname}=missing')
        continue
    sig_text = sig_m.group(1)
    has_new = 'w_lo_arr:Ptr[u32],w_hi_arr:Ptr[u32]' in sig_text
    has_old = 'w_arr:Ptr[u64]' in sig_text
    if has_new and not has_old:
        soa_sig_count += 1
        sig_details.append(f'{kname}=soa')
    elif has_old:
        old_sig_count += 1
        sig_details.append(f'{kname}=old')
    else:
        sig_details.append(f'{kname}=neither')
ok_soa_sig = (soa_sig_count == 5) and (old_sig_count == 0)
checks.append(('source_warr_soa_split_signatures', '5 kernels use w_lo_arr/w_hi_arr, 0 use old w_arr:Ptr[u64]',
                f'{soa_sig_count} kernels converted, {old_sig_count} old-style remaining ({", ".join(sig_details)})', ok_soa_sig))

dispatcher_m = re.search(r'^def launch_kernel_dfs_iter_gpu_static_maxd\(.*?\n  return False', s, re.M | re.S)
ok_dispatcher = (dispatcher_m is not None
                  and 'w_lo_arr:List[u32]=' in dispatcher_m.group(0)
                  and 'w_hi_arr:List[u32]=' in dispatcher_m.group(0)
                  and dispatcher_m.group(0).count('gpu.raw(w_lo_arr),gpu.raw(w_hi_arr)') == 5
                  and 'gpu.raw(w_arr)' not in dispatcher_m.group(0))
checks.append(('source_warr_soa_split_dispatcher', 'derives w_lo_arr/w_hi_arr once, passes to all 5 kernel launches',
                'present' if ok_dispatcher else 'missing or incomplete', ok_dispatcher))

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
  echo "[error] 304 source static checks failed" >&2
  exit 66
fi
echo "[static-ok] 304 source checks passed; proceeding to release build/run"

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[error] another 304 validation holds: $LOCK_FILE" >&2
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

TELEMETRY_LOG="$LOGDIR/gpu_telemetry.csv"
TELEMETRY_PID=""
NVSMI_FIELDS="timestamp,temperature.gpu,clocks.current.sm,clocks.current.memory,clocks.max.sm,clocks.max.memory,clocks.applications.graphics,clocks.applications.memory,power.draw,power.limit,power.default_limit,power.min_limit,power.max_limit,utilization.gpu,clocks_event_reasons.active"
if [[ "$CAPTURE_TELEMETRY" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu="$NVSMI_FIELDS" --format=csv > "$LOGDIR/gpu_pre_run_snapshot.csv" 2>&1 || true
  echo "[telemetry] pre-run snapshot: $LOGDIR/gpu_pre_run_snapshot.csv"

  SNAPSHOT_HEADER_OK=0
  if head -n1 "$LOGDIR/gpu_pre_run_snapshot.csv" | grep -q '^timestamp'; then
    SNAPSHOT_HEADER_OK=1
  else
    echo "[WARNING] gpu_pre_run_snapshot.csv does not look like valid nvidia-smi CSV output:" >&2
    head -n1 "$LOGDIR/gpu_pre_run_snapshot.csv" >&2
  fi

  # 313/314: clock-cap smoking-gun check (unchanged from 313). Compare the
  # currently-reported SM clock against the max supported SM clock; a large,
  # persistent gap with the GPU cool and idle is the signature of a clock
  # lock/cap, not thermal throttling.
  if (( SNAPSHOT_HEADER_OK )); then
    CUR_SM=$(awk -F', *' 'NR==2{gsub(/ MHz/,"",$3); print $3}' "$LOGDIR/gpu_pre_run_snapshot.csv" 2>/dev/null || true)
    MAX_SM=$(awk -F', *' 'NR==2{gsub(/ MHz/,"",$5); print $5}' "$LOGDIR/gpu_pre_run_snapshot.csv" 2>/dev/null || true)
  else
    CUR_SM=""; MAX_SM=""
  fi
  if [[ -n "$CUR_SM" && -n "$MAX_SM" && "$CUR_SM" =~ ^[0-9]+$ && "$MAX_SM" =~ ^[0-9]+$ && "$MAX_SM" -gt 0 ]]; then
    printf 'gpu_clock_cap_check\tcurrent_sm>=90%%_of_max_sm\tcurrent_sm=%sMHz max_sm=%sMHz\t%s\n' \
      "$CUR_SM" "$MAX_SM" \
      "$(awk -v c="$CUR_SM" -v m="$MAX_SM" 'BEGIN{print (c/m>=0.90)?"OK":"WARN-CAPPED"}')" >> "$SUMMARY"
    if (( CUR_SM * 100 < MAX_SM * 90 )); then
      echo "[WARNING] pre-run idle SM clock (${CUR_SM}MHz) is well below max supported (${MAX_SM}MHz)." >&2
      echo "[WARNING] This is consistent with a persistent clock lock/cap, not thermal throttling." >&2
      echo "[WARNING] Consider (manually, may need sudo): nvidia-smi -rgc ; nvidia-smi -rac" >&2
    fi
  elif (( SNAPSHOT_HEADER_OK )); then
    printf 'gpu_clock_cap_check\tcurrent_sm>=90%%_of_max_sm\tunavailable (max_sm/current_sm not reported by this nvidia-smi/driver)\tINFO\n' >> "$SUMMARY"
  else
    printf 'gpu_clock_cap_check\tcurrent_sm>=90%%_of_max_sm\tFAIL: nvidia-smi query error, see gpu_pre_run_snapshot.csv\tFAIL\n' >> "$SUMMARY"
    failures=$((failures+1))
  fi

  # 314: power-cap smoking-gun check. -rgc had no effect in 313's manual
  # follow-up (Clocks.SM stayed 1320MHz), so check whether the power limit
  # itself has been lowered below the card's default.
  if (( SNAPSHOT_HEADER_OK )); then
    POWER_LIMIT=$(awk -F', *' 'NR==2{gsub(/ W/,"",$10); print $10}' "$LOGDIR/gpu_pre_run_snapshot.csv" 2>/dev/null || true)
    POWER_DEFAULT=$(awk -F', *' 'NR==2{gsub(/ W/,"",$11); print $11}' "$LOGDIR/gpu_pre_run_snapshot.csv" 2>/dev/null || true)
  else
    POWER_LIMIT=""; POWER_DEFAULT=""
  fi
  if [[ -n "$POWER_LIMIT" && -n "$POWER_DEFAULT" ]] && \
     [[ "$POWER_LIMIT" =~ ^[0-9]+(\.[0-9]+)?$ ]] && [[ "$POWER_DEFAULT" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    printf 'gpu_power_cap_check\tpower.limit>=power.default_limit\tpower.limit=%sW power.default_limit=%sW\t%s\n' \
      "$POWER_LIMIT" "$POWER_DEFAULT" \
      "$(awk -v l="$POWER_LIMIT" -v d="$POWER_DEFAULT" 'BEGIN{print (l+0>=d+0-0.5)?"OK":"WARN-POWER-CAPPED"}')" >> "$SUMMARY"
    if awk -v l="$POWER_LIMIT" -v d="$POWER_DEFAULT" 'BEGIN{exit !(l+0 < d+0-0.5)}'; then
      echo "[WARNING] power.limit (${POWER_LIMIT}W) is below power.default_limit (${POWER_DEFAULT}W)." >&2
      echo "[WARNING] Consider (manually, requires sudo): sudo nvidia-smi -pl ${POWER_DEFAULT}" >&2
    fi
  elif (( SNAPSHOT_HEADER_OK )); then
    printf 'gpu_power_cap_check\tpower.limit>=power.default_limit\tunavailable (power.limit/power.default_limit not reported by this nvidia-smi/driver)\tINFO\n' >> "$SUMMARY"
  else
    printf 'gpu_power_cap_check\tpower.limit>=power.default_limit\tFAIL: nvidia-smi query error, see gpu_pre_run_snapshot.csv\tFAIL\n' >> "$SUMMARY"
    failures=$((failures+1))
  fi
else
  echo "[telemetry] nvidia-smi not available or CAPTURE_TELEMETRY=$CAPTURE_TELEMETRY; skipping pre-run snapshot" >&2
  printf 'gpu_clock_cap_check\tcurrent_sm>=90%%_of_max_sm\tnvidia-smi unavailable\tINFO\n' >> "$SUMMARY"
  printf 'gpu_power_cap_check\tpower.limit>=power.default_limit\tnvidia-smi unavailable\tINFO\n' >> "$SUMMARY"
fi

if (( COOLDOWN_SECONDS > 0 )); then
  echo "[cooldown] waiting ${COOLDOWN_SECONDS}s before full run (thermal drift precaution)"
  sleep "$COOLDOWN_SECONDS"
fi

if [[ "$CAPTURE_TELEMETRY" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu="$NVSMI_FIELDS" --format=csv -l "$TELEMETRY_INTERVAL_SECONDS" > "$TELEMETRY_LOG" 2>&1 &
  TELEMETRY_PID=$!
  echo "[telemetry] background capture started: pid=$TELEMETRY_PID interval=${TELEMETRY_INTERVAL_SECONDS}s log=$TELEMETRY_LOG"
fi

CMD=("$CAND" -g "$N" "$N" "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS" "$BENCH_MODE" "$REORDER_WINDOW_MULT" "$REORDER_PHASE_JUMP" "$CROSS_STRIPE_SAFE" "$WORKER_ID" "$WORKER_COUNT" "$BROADMARK_VARIANT" "$CHUNKSHAPE148_BUCKET_RUN")
{
  echo "================================================================"
  echo "candidate : $CAND"
  echo "source    : $SRC"
  echo "date      : $(date -Is)"
  echo "cwd       : $(pwd)"
  echo "build_mode: codon build -release; FORCE_REBUILD=$FORCE_REBUILD"
  command -v sha256sum >/dev/null 2>&1 && echo "source_sha256: $(sha256sum "$SRC" | awk '{print $1}')"
  printf 'command   :'; printf ' %q' "${CMD[@]}"; echo
  echo "validation: one N=21 full run through 342 bucket-run-cap-raise (kernel-identical to 338/337/336/335/334/333/328) mode31 split145 CHUNKSHAPE148_BUCKET_RUN=$CHUNKSHAPE148_BUCKET_RUN"
  echo "dispatch  : required=14, selected MAXD14, schedule_words=0, stack=208 bytes/thread"
  echo "================================================================"
} | tee "$RUN_LOG"

set +e; stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "$RUN_LOG"; run_rc=${PIPESTATUS[0]}; set -e

if [[ -n "$TELEMETRY_PID" ]]; then
  kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
  wait "$TELEMETRY_PID" 2>/dev/null || true
  echo "[telemetry] background capture stopped: $TELEMETRY_LOG"
fi

record_check run_exit 0 "$run_rc" || failures=$((failures+1))
if (( run_rc != 0 )); then exit "$run_rc"; fi

if [[ -s "$TELEMETRY_LOG" ]] && head -n1 "$TELEMETRY_LOG" | grep -q '^timestamp'; then
  printf 'gpu_telemetry_captured\tpresent\tpresent (%s rows)\tOK\n' "$(($(wc -l < "$TELEMETRY_LOG") - 1))" >> "$SUMMARY"
elif [[ -s "$TELEMETRY_LOG" ]]; then
  # File exists and is non-empty but doesn't look like the expected CSV --
  # most likely an nvidia-smi query error (e.g. an invalid field name).
  # This is a real failure, not just "telemetry unavailable".
  printf 'gpu_telemetry_captured\tpresent\tFAIL: unexpected content (%s)\tFAIL\n' \
    "$(head -n1 "$TELEMETRY_LOG" | tr -d '\t')" >> "$SUMMARY"
  failures=$((failures+1))
  echo "[WARNING] gpu_telemetry.csv does not start with the expected 'timestamp' header:" >&2
  head -n1 "$TELEMETRY_LOG" >&2
else
  printf 'gpu_telemetry_captured\tpresent\tabsent (nvidia-smi unavailable or CAPTURE_TELEMETRY=0)\tINFO\n' >> "$SUMMARY"
fi

DYNAMIC_PRESET=$(sed -n 's/^\[dynamic-preset\] N=21 preset_queens=\([0-9][0-9]*\)$/\1/p' "$RUN_LOG" | tail -n1)
record_check dynamic_preset_N21 6 "${DYNAMIC_PRESET:-missing}" || failures=$((failures+1))

RUNTIME_VARIANT=$(sed -n 's/.*\bvariant=\([0-9][0-9]*\).*/\1/p' "$RUN_LOG" | tail -n1)
record_check runtime_broadmark_variant "$EXPECTED_BROADMARK_VARIANT" "${RUNTIME_VARIANT:-missing}" || failures=$((failures+1))
RUNTIME_VARIANT_TAG=$(sed -n 's/.*\btag=\([a-z_]*\).*/\1/p' "$RUN_LOG" | tail -n1)
record_check runtime_broadmark_variant_tag "$EXPECTED_BROADMARK_VARIANT_TAG" "${RUNTIME_VARIANT_TAG:-missing}" || failures=$((failures+1))

# ---- 339: bucket_run actually in effect, and whether the shaped bin was rebuilt ----
RUNTIME_BUCKET_RUN=$(sed -n 's/.*\bbucket_run=\([0-9][0-9]*\).*/\1/p' "$RUN_LOG" | tail -n1)
record_check runtime_chunkshape148_bucket_run "$EXPECTED_CHUNKSHAPE148_BUCKET_RUN" "${RUNTIME_BUCKET_RUN:-missing}" || failures=$((failures+1))
if grep -q '\[chunkshape148-reuse\]' "$RUN_LOG"; then
  CACHE_STATE=reuse
elif grep -q '\[chunkshape148-build' "$RUN_LOG"; then
  CACHE_STATE=build
else
  CACHE_STATE=unknown
fi
printf 'chunkshape148_cache_state\treuse (timing comparable) or build (timing NOT comparable)\t%s\tINFO\n' "$CACHE_STATE" >> "$SUMMARY"
if [[ "$CACHE_STATE" == build ]]; then
  printf 'timing_comparability\tcached run required for comparison\tthis run REBUILT the shaped bin; elapsed includes cache construction -- re-run once more before comparing\tINFO\n' >> "$SUMMARY"
else
  printf 'timing_comparability\tcached run required for comparison\tcached shaped bin reused; elapsed is comparable\tINFO\n' >> "$SUMMARY"
fi
SHAPED_BIN=$(sed -n 's/.*\bbin=\(constellations_[^ ]*\.bin\).*/\1/p' "$RUN_LOG" | tail -n1)
printf 'chunkshape148_shaped_bin\t(for manual reference)\t%s\tINFO\n' "${SHAPED_BIN:-not found}" >> "$SUMMARY"

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
  for pair in "341run256:$BASELINE_341_RUN256_SECONDS" "341run64:$BASELINE_341_RUN64_SECONDS" "340run64:$BASELINE_340_RUN64_SECONDS" "340run32:$BASELINE_340_RUN32_SECONDS" "340run1:$BASELINE_340_RUN1_SECONDS" "339bucketrun32:$BASELINE_339_BUCKET_RUN32_SECONDS" "338maxd14portdesign:$BASELINE_338_MAXD14_PORT_DESIGN_SECONDS" "337binformatreaderdesign:$BASELINE_337_BIN_FORMAT_READER_DESIGN_SECONDS" "336cudacsmoketest:$BASELINE_336_CUDAC_SMOKE_TEST_SECONDS" "335cudactoolchainlocate:$BASELINE_335_CUDAC_TOOLCHAIN_LOCATE_SECONDS" "334cudacrunnerspikedesign:$BASELINE_334_CUDAC_RUNNER_SPIKE_DESIGN_SECONDS" "333w3j7adopt:$BASELINE_333_W3J7_ADOPT_SECONDS" "332w3j7measured:$BASELINE_332_W3J7_MEASURED_SECONDS" "328warrsoasplitimplement:$BASELINE_328_WARR_SOA_SPLIT_IMPLEMENT_SECONDS" "326futurecheckspecializeaxis1rejected:$BASELINE_326_FUTURECHECK_SPECIALIZE_AXIS1_REJECTED_SECONDS" "327warrloadsplitverify:$BASELINE_327_WARRLOADSPLIT_VERIFY_SECONDS" "319sourcecounterspagesourceprobe:$BASELINE_319_SOURCECOUNTERS_PAGESOURCE_PROBE_SECONDS" "318sourcecounterssudoretry:$BASELINE_318_SOURCECOUNTERS_SUDO_RETRY_SECONDS" "317branchdivergenceprobe:$BASELINE_317_BRANCH_DIVERGENCE_PROBE_SECONDS" "316envacceptncuprep:$BASELINE_316_ENV_ACCEPT_NCU_PREP_SECONDS" "315telemetryfieldnamefix:$BASELINE_315_TELEMETRY_FIELDNAME_FIX_SECONDS" "314powercapdiagnosis:$BASELINE_314_POWER_CAP_DIAGNOSIS_SECONDS" "313clockcapdiagnosis:$BASELINE_313_CLOCK_CAP_DIAGNOSIS_SECONDS" "312thermalreprocheck:$BASELINE_312_THERMAL_REPRO_CHECK_SECONDS" "311variant2restore:$BASELINE_311_VARIANT2_RESTORE_SECONDS" "310variant1phaseonly:$BASELINE_310_VARIANT1_PHASE_ONLY_SECONDS" "309variant4phaserotate:$BASELINE_309_VARIANT4_PHASE_ROTATE_SECONDS" "308K52finalsweep:$BASELINE_308_K52_FINAL_SWEEP_SECONDS" "307K44fineprobe:$BASELINE_307_K44_FINE_PROBE_SECONDS" "306K56sweep:$BASELINE_306_K56_SWEEP_SECONDS" "305K40sweep:$BASELINE_305_K40_SWEEP_SECONDS" "304K48sweep:$BASELINE_304_K48_SWEEP_SECONDS" "303curdepthx4neutral:$BASELINE_303_CUR_DEPTH_X4_NEUTRAL_SECONDS" "302curdepthx4fix:$BASELINE_302_CUR_DEPTH_X4_FIX_SECONDS" "301curdepthx4:$BASELINE_301_CUR_DEPTH_X4_SECONDS" "300scheduleu64:$BASELINE_300_SCHEDULE_U64_SECONDS" "299K64on296:$BASELINE_299_K64_ON_296_SECONDS" "298nextdepthelim:$BASELINE_298_NEXT_DEPTH_ELIM_SECONDS" "297savespelim:$BASELINE_297_SAVE_SP_ELIM_SECONDS" "296stackptr:$BASELINE_296_STACK_PTR_SECONDS" "295stackmerge:$BASELINE_295_STACK_MERGE_SECONDS" "294colavldrd:$BASELINE_294_COLAV_LDRD_SECONDS" "293duallane:$BASELINE_293_DUAL_LANE_SECONDS" "292k32confirmed:$BASELINE_292_K32_CONFIRMED_SECONDS" "291blockcodelate:$BASELINE_291_BLOCKCODELATE_SECONDS" "292k16:$BASELINE_292_K16_SECONDS" "292k32:$BASELINE_292_K32_SECONDS" "292k64:$BASELINE_292_K64_SECONDS" "289ncolonly:$BASELINE_289_NCOLONLY_SECONDS" "288rejected:$BASELINE_288_REJECT_SECONDS" "287adopt:$BASELINE_287_ADOPT_SECONDS" "286adopt:$BASELINE_286_ADOPT_SECONDS" "285restore:$BASELINE_285_RESTORE_SECONDS" "284rejected:$BASELINE_284_REJECT_SECONDS" "283normalfirst:$BASELINE_283_NORMALFIRST_SECONDS" "276current:427.672" "271fastest:427.705" "239:427.703" "217:427.709"; do
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
echo "[validation-ok] 342 N=21 mode31 split145 full reproduced total with required=14, MAXD14, 208 bytes/thread, K_PER_THREAD_MAXD14=48 (3 dispatch/progress rows), BROADMARK_VARIANT=2 (rotate_only, unchanged); the only source change in 342 is CHUNKSHAPE148_BUCKET_RUN_MAX 256 to 8192, which cannot affect any RUN <= 256. This run used CHUNKSHAPE148_BUCKET_RUN=$CHUNKSHAPE148_BUCKET_RUN. 341 measured RUN=256 at 439.835s (-2.299% vs the RUN=1 control 450.183s), the best point so far and sitting exactly on the old ceiling. CHECK chunkshape148_cache_state BEFORE COMPARING TIMING: only reuse rows are comparable. The 342 experiment proper is the sweep -- run bash 342_bucket_run_cap_raise_driver.sh"
