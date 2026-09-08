#!/usr/bin/env bash
# 393_ncu_profile.sh
#
# rev393 -- ncu profiling harness for the N=21 speedup work, driven
# entirely through the 392 Codon binary with the 388 `-d` debug flag.
#
# SCOPE / WHAT THIS IS NOT
# ------------------------
# This revision changes NO source. There is no correctness oracle for
# it: stage 1 and stage 2 are partial executions (N=18 launch 0, and
# N=21 chunk0 only), so 314666222712 cannot be confirmed here. The
# only correctness criterion is that the profiled binary is the
# real-hardware-confirmed 392 binary, unmodified -- checked statically
# below via REV_TAG plus a recorded sha256 of the .py.
#
# Following 355/357: this script does NOT parse ncu numeric output. It
# checks only that section headings appear (INFO, non-gating) and saves
# the raw logs and the .ncu-rep files whole. Reading and interpreting
# the numbers happens after the logs come back.
#
# STAGES (ascending cost, each individually skippable, Ctrl-C safe
# between them -- the 391 staged pattern)
# -----------------------------------------------------------------
#   1  N=18, bench_mode=0, --section SourceCounters --page source
#      Structural. ~9s of work + 5-pass replay. Same compiled
#      kernel_dfs_iter_gpu_maxd14 via launch_kernel_dfs_iter_gpu_
#      static_maxd, so SASS/registers transfer to N=21 (proved in 352).
#      This is the "N=18 trick".
#
#   2  N=21, bench_mode=30 chunk0 only, --section SchedulerStats
#      --section WarpStateStats
#      Dynamic, production K=48 shape. bench_mode=30 (NOT 31) because
#      --launch-count 1 only scopes ncu's instrumentation, it does not
#      stop the program -- 355 lost 24 minutes to exactly that, 357
#      fixed it with 30 + debug_chunk_start=0 debug_chunk_count=1.
#      357 measured this stage at 19m40s. Budget 25 minutes.
#
#   3  OPT-IN ONLY (STAGES must list 3). N=21 via bench_mode=37, i.e.
#      the CRunner os.system dispatch into ./389_kernel_maxd14, with
#      --target-processes all so ncu follows into the C child process.
#      This profiles the PRODUCTION kernel (the one at kernel_ms ~
#      201,237), not Codon's. It runs N=21 to completion -- the C
#      binary's CLI has no record limit on the GPU path -- so budget
#      ~10 minutes, and note it needs the filtered CRunner input bin
#      (389's ensure_crunner_input_bin() builds it if absent).
#
# Default is STAGES="1,2". See the accompanying note on which of the
# two kernels rev393 should actually be optimising.
#
# USAGE
#   STATIC_ONLY=1 bash 393_ncu_profile.sh     # checks only, no GPU
#                 bash 393_ncu_profile.sh     # stages 1,2
#   STAGES=1      bash 393_ncu_profile.sh     # N=18 structural only
#   STAGES=1,2,3  bash 393_ncu_profile.sh     # + production C kernel
#
# sudo is required for hardware counters (established since 318; 316's
# "PC sampling is blocked in this environment" conclusion turned out to
# be nothing but a missing sudo). `sudo -n true` is the FIRST thing
# checked, before any build or any GPU time -- 352 burned 14 minutes
# discovering this at the end instead of the start.

set -u

REV="393"
PY_SRC="${PY_SRC:-392Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-392Py_kernel_maxd14_final}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CRUNNER_BIN="${CRUNNER_BIN:-./389_kernel_maxd14}"
# 392 fixed REV_TAG (it had been stale at 388 since 388), so the
# dispatch log dir the binary writes to is now 392_crunner_logs.
CRUNNER_LOGDIR="${CRUNNER_LOGDIR:-392_crunner_logs}"
CODON="${CODON:-codon}"
STAGES="${STAGES:-1,2}"
STATIC_ONLY="${STATIC_ONLY:-0}"
BUILD_IF_MISSING="${BUILD_IF_MISSING:-1}"
STAGE_PAUSE="${STAGE_PAUSE:-8}"

# 6102's lesson: resolve ncu to an ABSOLUTE path here and hand that to
# sudo. `sudo ncu` alone can fail on root's PATH even when ncu is on
# the user's.
NCU="${NCU:-$(command -v ncu 2>/dev/null)}"
[[ -z "$NCU" && -x /usr/local/cuda/bin/ncu ]] && NCU="/usr/local/cuda/bin/ncu"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_ncu_${TS}}"

PASS=0
FAIL=0
declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }

want_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

# ---------------------------------------------------------------------
# 0. sudo FIRST. Nothing else happens until this passes.
# ---------------------------------------------------------------------
if sudo -n true 2>/dev/null; then
  pass "sudo_noninteractive_available"
else
  fail "sudo_noninteractive_available" "'sudo -n true' failed -- ncu cannot read hardware counters without it. Stopping before any build or GPU time (352's 14-minute lesson)."
  exit 1
fi

# ---------------------------------------------------------------------
# 1. Static checks
# ---------------------------------------------------------------------
for f in "$PY_SRC" "$HELPER_SRC"; do
  if [[ -f "$f" ]]; then pass "file_present[$f]"
  else fail "file_present[$f]" "not found in $(pwd)"; fi
done

if grep -qE '^REV_TAG:str="392"' "$PY_SRC" 2>/dev/null; then
  pass "source_rev_tag_is_392"
else
  fail "source_rev_tag_is_392" "expected REV_TAG:str=\"392\" in $PY_SRC -- this harness profiles the 392 binary specifically"
fi

if grep -q '^@gpu.kernel' "$PY_SRC" 2>/dev/null && grep -q 'def kernel_dfs_iter_gpu_maxd14' "$PY_SRC" 2>/dev/null; then
  pass "source_kernel_maxd14_present"
else
  fail "source_kernel_maxd14_present" "kernel_dfs_iter_gpu_maxd14 not found in $PY_SRC"
fi

# The -d flag itself: 388 implemented it as an argv filter + log_level
# bump. Confirm both halves are still there before relying on it.
if grep -q 'if tok=="-d":' "$PY_SRC" 2>/dev/null && grep -q 'if debug_mode and gpu_log_level<1:' "$PY_SRC" 2>/dev/null; then
  pass "source_debug_flag_intact"
else
  fail "source_debug_flag_intact" "the 388 -d argv filter and/or its log_level bump is missing from $PY_SRC"
fi

# bench_mode=30 must survive the CLI whitelist gate. This gate has
# silently reset new modes to 0 four separate times (361, 365, 366,
# 368/369) -- 30 is old and should be fine, but check, don't assume.
if grep -q 'bench_mode==30' "$PY_SRC" 2>/dev/null && \
   grep -qE 'if not \(bench_mode==0 .*bench_mode==30' "$PY_SRC" 2>/dev/null; then
  pass "source_bench_mode_30_whitelisted"
else
  fail "source_bench_mode_30_whitelisted" "bench_mode==30 is not in the CLI whitelist gate -- it would be silently reset to 0 and stage 2 would run the wrong path"
fi

if [[ -n "$NCU" && -x "$NCU" ]]; then
  pass "ncu_present[$NCU]"
else
  fail "ncu_present" "ncu not found on PATH nor at /usr/local/cuda/bin/ncu"
fi

if want_stage 3; then
  if [[ -x "$CRUNNER_BIN" ]]; then
    pass "crunner_binary_present[$CRUNNER_BIN]"
  else
    fail "crunner_binary_present[$CRUNNER_BIN]" "stage 3 was requested but the CRunner binary is not built. Build it from 389_kernel_maxd14.cu first."
  fi
fi

if [[ ! -x "$PY_BIN" ]]; then
  if [[ "$BUILD_IF_MISSING" == "1" && "$STATIC_ONLY" != "1" ]]; then
    info "binary_missing" "$PY_BIN not found -- building (codon build -release), same command 392_validate.sh used"
    "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "${REV}_codon_build_${TS}.log"
  fi
fi
if [[ -x "$PY_BIN" ]]; then
  pass "binary_present[$PY_BIN]"
else
  if [[ "$STATIC_ONLY" == "1" ]]; then
    info "binary_present[$PY_BIN]" "absent, but STATIC_ONLY=1 so not building"
  else
    fail "binary_present[$PY_BIN]" "not found and could not be built"
  fi
fi

if [[ "$FAIL" -gt 0 ]]; then
  echo
  echo "===== ${REV} static summary ====="
  echo "OK=$PASS  FAIL=$FAIL"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  exit 1
fi

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo
  echo "===== ${REV} STATIC_ONLY summary ====="
  echo "OK=$PASS  FAIL=$FAIL"
  echo "Static checks passed. Re-run without STATIC_ONLY=1 to profile."
  exit 0
fi

mkdir -p "$LOGDIR"
pass "logdir_created[$LOGDIR]"

# ---------------------------------------------------------------------
# 2. Environment capture. SM clock is a first-class measurement
#    variable in this project (earlier sessions ran clock-limited at
#    1320MHz against a 1710MHz maximum), so record it BEFORE and AFTER
#    rather than discovering a mismatch when comparing to 355/357.
# ---------------------------------------------------------------------
{
  echo "=== $REV environment capture (pre) $(date -Is) ==="
  echo "--- host ---";        uname -a
  echo "--- nvidia-smi ---";  nvidia-smi 2>&1
  echo "--- clocks ---";      nvidia-smi -q -d CLOCK 2>&1
  echo "--- ncu ---";         "$NCU" --version 2>&1
  echo "--- nvcc ---";        (nvcc --version 2>&1 || /usr/local/cuda/bin/nvcc --version 2>&1)
  echo "--- source sha256 (docstrings NOT stripped; raw file) ---"
  sha256sum "$PY_SRC" "$HELPER_SRC" 2>&1
  [[ -x "$PY_BIN" ]] && sha256sum "$PY_BIN" 2>&1
  [[ -x "$CRUNNER_BIN" ]] && sha256sum "$CRUNNER_BIN" 2>&1
  echo "--- invocation ---"
  echo "STAGES=$STAGES PY_BIN=$PY_BIN NCU=$NCU"
} > "$LOGDIR/00_env_pre.txt" 2>&1
pass "env_captured_pre"

SM_CLOCK="$(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader 2>/dev/null || echo 'unavailable')"
info "sm_clock" "$SM_CLOCK  (relevant when comparing against 355/357 numbers)"

# Anything the profiled run creates under sudo lands root-owned and
# will trip up the next non-sudo run. Hand it back afterwards.
MARKER="$LOGDIR/.owner_marker"
touch "$MARKER"
reclaim_ownership() {
  find . -maxdepth 1 -newer "$MARKER" -user root -print0 2>/dev/null \
    | xargs -0 -r sudo chown "$(id -u):$(id -g)" 2>/dev/null || true
  sudo chown -R "$(id -u):$(id -g)" "$LOGDIR" 2>/dev/null || true
}

# Re-export a .ncu-rep to human-readable text. Never gating: 318 and
# 319 both established that the right page flag is discovered
# empirically, and a re-import costs nothing (no re-measurement).
export_report() {
  local rep="$1" stem="$2"
  [[ -f "$rep" ]] || { info "report_missing[$rep]" "nothing to re-export"; return 0; }
  "$NCU" --import "$rep" --page details --print-details all > "${stem}_details.txt" 2>&1 || true
  "$NCU" --import "$rep" --page source                      > "${stem}_source.txt"  2>&1 || true
  "$NCU" --import "$rep" --page source --csv                > "${stem}_source.csv"  2>&1 || true
  info "report_exported[$stem]" "details/source/csv written"
}

heading_seen() {
  local file="$1" needle="$2" label="$3"
  if grep -qi -- "$needle" "$file" 2>/dev/null; then
    info "$label" "heading present"
  else
    info "$label" "heading NOT found -- not gating, but worth a look in the raw log"
  fi
}

# =====================================================================
# STAGE 1 -- N=18 structural (SourceCounters)
# =====================================================================
if want_stage 1; then
  banner "STAGE 1  N=18 SourceCounters (structural, ~1 min with replay)"
  echo "Ctrl-C now to skip. Starting in ${STAGE_PAUSE}s..."
  sleep "$STAGE_PAUSE"

  S1_REP="$LOGDIR/${REV}_n18_sourcecounters"
  S1_LOG="$LOGDIR/10_stage1_n18_sourcecounters.log"
  # -g 18 18 32 484 0 0 5 0 -d
  #        |  |  |   |  | | | |  `-- 388 debug flag: bumps log_level to 1
  #        |  |  |   |  | | | `----- bench_mode=0 (N=18 is below the
  #        |  |  |   |  | | |        N>=21 gate on modes 30/31 anyway)
  #        |  |  |   |  | | `------- preset_queens=5 (the default; any
  #        |  |  |   |  | |          other value is force-reset to 5
  #        |  |  |   |  | |          with a warning in normal modes)
  #        |  |  |   |  | `--------- sort_mode=0
  #        |  |  |   |  `----------- log_level=0, deliberately: -d is
  #        |  |  |   |               what raises it, which is exactly
  #        |  |  |   |               the flag being exercised here
  #        |  |  |   `-------------- MAX_BLOCKS=484
  #        |  |  `------------------ BLOCK=32
  #        `--`--------------------- nmin=nmax=18
  set -x
  sudo "$NCU" --launch-count 1 --section SourceCounters --page source -f \
      -o "$S1_REP" \
      "./$PY_BIN" -g 18 18 32 484 0 0 5 0 -d 2>&1 | tee "$S1_LOG"
  set +x
  reclaim_ownership

  if grep -q '\[debug-mode\] enabled via -d' "$S1_LOG"; then
    pass "stage1_debug_flag_engaged"
  else
    fail "stage1_debug_flag_engaged" "the -d banner did not appear in $S1_LOG -- the flag did not take effect"
  fi
  heading_seen "$S1_LOG" "Source Counters" "stage1_sourcecounters_heading"
  export_report "${S1_REP}.ncu-rep" "$LOGDIR/${REV}_n18_sourcecounters"
  pass "stage1_completed"
fi

# =====================================================================
# STAGE 2 -- N=21 dynamic, chunk0 only (SchedulerStats + WarpStateStats)
# =====================================================================
if want_stage 2; then
  banner "STAGE 2  N=21 chunk0, SchedulerStats+WarpStateStats (budget ~25 min)"
  echo "357 measured this at 19m40s. Ctrl-C now to skip. Starting in ${STAGE_PAUSE}s..."
  sleep "$STAGE_PAUSE"

  S2_REP="$LOGDIR/${REV}_n21_chunk0_scheduler_warpstate"
  S2_LOG="$LOGDIR/20_stage2_n21_chunk0.log"
  # -g 21 21 32 484 0 0 7 30 3 7 0 0 1 -d
  #                         |  | | | | |
  #                         |  | | | | `-- argv[14] debug_chunk_count=1
  #                         |  | | | `---- argv[13] debug_chunk_start=0
  #                         |  | | `------ argv[12] cross_stripe_safe=0
  #                         |  | `-------- argv[11] phase_jump=7   (adopted 333)
  #                         |  `---------- argv[10] window_mult=3  (adopted 333)
  #                         `------------- bench_mode=30: probe mode,
  #                                        stops at the chunk boundary.
  #                                        NOT 31 -- --launch-count 1
  #                                        scopes instrumentation only,
  #                                        it does not stop the program.
  set -x
  sudo "$NCU" --launch-count 1 \
      --section SchedulerStats --section WarpStateStats -f \
      -o "$S2_REP" \
      "./$PY_BIN" -g 21 21 32 484 0 0 7 30 3 7 0 0 1 -d 2>&1 | tee "$S2_LOG"
  set +x
  reclaim_ownership

  if grep -q '\[debug-mode\] enabled via -d' "$S2_LOG"; then
    pass "stage2_debug_flag_engaged"
  else
    fail "stage2_debug_flag_engaged" "the -d banner did not appear in $S2_LOG"
  fi
  # 357's own gate: proof we really are in probe mode and not mode 31.
  if grep -q 'split291_final_probe' "$S2_LOG"; then
    pass "stage2_probe_mode_confirmed"
  else
    fail "stage2_probe_mode_confirmed" "'split291_final_probe' absent from $S2_LOG -- bench_mode may have been reset to 0 by the CLI whitelist gate, or log_level never reached 1"
  fi
  heading_seen "$S2_LOG" "Scheduler Statistics"   "stage2_schedulerstats_heading"
  heading_seen "$S2_LOG" "Warp State Statistics"  "stage2_warpstatestats_heading"
  export_report "${S2_REP}.ncu-rep" "$LOGDIR/${REV}_n21_chunk0_scheduler_warpstate"
  pass "stage2_completed"
fi

# =====================================================================
# STAGE 3 (opt-in) -- N=21 production path, C kernel via CRunner
# =====================================================================
if want_stage 3; then
  banner "STAGE 3  N=21 via bench_mode=37 -> $CRUNNER_BIN (opt-in, budget ~10 min)"
  echo "This runs N=21 TO COMPLETION -- the C GPU path takes no record limit."
  echo "Ctrl-C now to skip. Starting in ${STAGE_PAUSE}s..."
  sleep "$STAGE_PAUSE"

  S3_REP="$LOGDIR/${REV}_n21_crunner_scheduler_warpstate"
  S3_LOG="$LOGDIR/30_stage3_n21_crunner.log"
  # --target-processes all is what makes ncu follow os.system() into
  # the C child. Without it ncu profiles only the Codon parent, which
  # launches no kernel at all in bench_mode=37.
  set -x
  sudo "$NCU" --target-processes all --launch-count 1 \
      --section SchedulerStats --section WarpStateStats -f \
      -o "$S3_REP" \
      "./$PY_BIN" -g 21 21 32 484 0 0 7 37 -d 2>&1 | tee "$S3_LOG"
  set +x
  reclaim_ownership

  if grep -q '\[gpu-run-done\]' "$S3_LOG"; then
    pass "stage3_crunner_reached_c_binary"
  else
    fail "stage3_crunner_reached_c_binary" "'[gpu-run-done]' absent -- the CRunner dispatch did not reach $CRUNNER_BIN (check for crunner-input-missing / maxd-unsupported in $S3_LOG)"
  fi
  # Stage 3 is the one stage that DOES run to completion, so the
  # oracle is available here. Use it.
  if grep -q '314666222712' "$S3_LOG"; then
    pass "stage3_oracle_314666222712_seen"
  else
    info "stage3_oracle_314666222712_seen" "oracle not seen in the console log -- check the crunner_logs dispatch.log before trusting any timing from this stage"
  fi
  heading_seen "$S3_LOG" "Scheduler Statistics"  "stage3_schedulerstats_heading"
  heading_seen "$S3_LOG" "Warp State Statistics" "stage3_warpstatestats_heading"
  export_report "${S3_REP}.ncu-rep" "$LOGDIR/${REV}_n21_crunner_scheduler_warpstate"
  pass "stage3_completed"
fi

# ---------------------------------------------------------------------
# 3. Post-run capture + tarball
# ---------------------------------------------------------------------
{
  echo "=== $REV environment capture (post) $(date -Is) ==="
  echo "--- clocks ---"; nvidia-smi -q -d CLOCK 2>&1
  echo "--- nvidia-smi ---"; nvidia-smi 2>&1
} > "$LOGDIR/99_env_post.txt" 2>&1

# The rev-numbered CRunner log dir the binary writes to, if stage 3 ran.
[[ -d "$CRUNNER_LOGDIR" ]] && cp -r "$CRUNNER_LOGDIR" "$LOGDIR/" 2>/dev/null || true

rm -f "$MARKER"
reclaim_ownership

TARBALL="${LOGDIR}.tar.gz"
tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" \
  || info "tarball_created" "tar failed -- send $LOGDIR/ as-is"

echo
echo "===== ${REV} summary ====="
echo "OK=$PASS  FAIL=$FAIL"
echo "stages run: $STAGES"
echo "SM clock  : $SM_CLOCK"
echo "logs      : $LOGDIR/"
echo "tarball   : $TARBALL"
if [[ "$FAIL" -gt 0 ]]; then
  echo "FAILED CHECKS:"
  for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  echo "(the .ncu-rep files are still saved -- a failed check here does"
  echo " not mean the profile itself is unusable)"
  exit 1
fi
echo "${REV} PASSED. Send $TARBALL for analysis."
exit 0
