#!/usr/bin/bash

set -u

# 132 locked validation harness
# - Candidate: 132Py_scorestripe_chunkshape_probe
# - Kernel path remains generic-only; 132 is a 131 validated carry-forward with isolated 132 labels/cache names.
# - Locked after the first N21 validation run on 2026-06-16.
# - This script checks fixed partial_total values plus full correctness.

BASE=${BASE:-./115Py_range_default_clean_cg_v2}
CAND=${CAND:-./132Py_scorestripe_chunkshape_probe}
RUN_FULL=${RUN_FULL:-1}

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="132Py_logs_N21_locked_validation_${TS}"
SUMMARY="${LOGDIR}/summary.tsv"

mkdir -p "$LOGDIR"

echo -e "case\texpected_pattern\tstatus\texit_code\tlogfile" > "$SUMMARY"

run_case() {
  local name="$1"
  local expected="$2"
  shift 2

  local logfile="${LOGDIR}/${name}.log"

  echo
  echo "================================================================"
  echo "[run] ${name}"
  echo "[log] ${logfile}"
  echo "================================================================"

  {
    echo "================================================================"
    echo "case      : ${name}"
    echo "date      : $(date -Is)"
    echo "cwd       : $(pwd)"
    echo "BASE      : ${BASE}"
    echo "CAND      : ${CAND}"
    echo "RUN_FULL  : ${RUN_FULL}"
    echo "expected  : ${expected}"
    echo -n "command   :"
    printf ' %q' "$@"
    echo
    echo "----------------------------------------------------------------"

    stdbuf -oL -eL "$@"
    rc=$?

    echo "----------------------------------------------------------------"
    echo "exit_code : ${rc}"
    echo "end_date  : $(date -Is)"
    echo "================================================================"

    exit "$rc"
  } 2>&1 | tee "$logfile"

  local rc=${PIPESTATUS[0]}
  local status="CHECK"

  # Inspect only the command-output body.
  # The logfile also contains the expected line, so grepping the full file can false-pass.
  local body_tmp="${LOGDIR}/${name}.body.tmp"

  awk '
    /^----------------------------------------------------------------$/ {
      sep += 1
      next
    }
    sep == 1 {
      print
    }
  ' "$logfile" > "$body_tmp"

  if [ "$rc" -ne 0 ]; then
    status="FAIL_EXIT"
  elif [ -n "$expected" ] && grep -q "$expected" "$body_tmp"; then
    status="OK"
  elif [ -z "$expected" ]; then
    status="NO_EXPECTED"
  else
    status="FAIL_EXPECTED"
  fi

  rm -f "$body_tmp"

  echo -e "${name}\t${expected}\t${status}\t${rc}\t${logfile}" >> "$SUMMARY"
  echo "[result] ${name}: ${status}"
}

skip_case() {
  local name="$1"
  local reason="$2"
  echo
  echo "================================================================"
  echo "[skip] ${name}: ${reason}"
  echo "================================================================"
  echo -e "${name}\t${reason}\tSKIP\t0\t-" >> "$SUMMARY"
}

echo "[validation-start] ${TS}"
echo "[logdir] ${LOGDIR}"
echo "[BASE] ${BASE}"
echo "[CAND] ${CAND}"
echo "[RUN_FULL] ${RUN_FULL}"

# ================================================================
# 1) 132 selected-chunk smoke: standard spread
# locked partial_total = 16775605854
# ================================================================
run_case \
  "01_cand132_mode30_standard_7chunk" \
  "partial_total=16775605854" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "0,20,40,60,80,100,120"

# ================================================================
# 2) 132 selected-chunk stress: heavy band
# locked partial_total = 19270188328
# ================================================================
run_case \
  "02_cand132_mode30_heavy_band" \
  "partial_total=19270188328" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "35,40,41,42,47,48,52,53"

# ================================================================
# 3) 132 selected-chunk stress: d2base14-density band
# locked partial_total = 16808879476
# ================================================================
run_case \
  "03_cand132_mode30_d2base14_density" \
  "partial_total=16808879476" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "20,40,55,56,57,58,60"

# ================================================================
# 4) 132 selected-chunk stress: late/tail band
# locked partial_total = 16269647518
# ================================================================
run_case \
  "04_cand132_mode30_late_tail" \
  "partial_total=16269647518" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "100,105,110,115,120,125,130"

# ================================================================
# 5) Candidate worker split quick check
# worker 0/4 locked partial_total = 79323742766
# ================================================================
run_case \
  "05_cand132_mode31_worker0of4_quick" \
  "partial_total=79323742766 expected_total=314666222712" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 31 8 7 0 0 4 2

# ================================================================
# 6) Full N21 correctness check
# Default RUN_FULL=1. Set RUN_FULL=0 to skip the long full run.
# ================================================================
if [ "${RUN_FULL}" = "1" ]; then
  run_case \
    "06_cand132_mode31_full_correctness" \
    "314666222712.*ok" \
    "$CAND" \
      -g 21 21 32 484 1 0 7 31 8 7 0 0 1 2
else
  skip_case \
    "06_cand132_mode31_full_correctness" \
    "RUN_FULL=0"
fi

echo
echo "================================================================"
echo "[summary]"
cat "$SUMMARY"
echo "================================================================"
echo "[done] logs are under: ${LOGDIR}"
