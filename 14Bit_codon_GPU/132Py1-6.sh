#!/usr/bin/bash

set -u

# 132 validation harness
# - Candidate: 132Py_scorestripe_chunkshape_probe
# - Kernel path remains generic-only; 132 is a 131 validated carry-forward with isolated 132 labels/cache names.
# - The selected-chunk partial totals are intentionally not hard-coded for the first run,
#   because percentile score-aware shaping changes which records belong to each chunk.
# - Full correctness is checked in case 6.

BASE=${BASE:-./115Py_range_default_clean_cg_v2}
CAND=${CAND:-./132Py_scorestripe_chunkshape_probe}
RUN_FULL=${RUN_FULL:-1}

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="132Py_logs_N21_validation_${TS}"
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

  # command output 部分だけを検査する。
  # logfile には expected 行も含まれるため、logfile 全体を grep すると false OK になる。
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
# chunkshape132 changes chunk composition, so only partial_total emission is checked.
# ================================================================
run_case \
  "01_cand132_mode30_standard_7chunk" \
  "partial_total=" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "0,20,40,60,80,100,120"

# ================================================================
# 2) 132 selected-chunk stress: former heavy band
# ================================================================
run_case \
  "02_cand132_mode30_heavy_band" \
  "partial_total=" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "35,40,41,42,47,48,52,53"

# ================================================================
# 3) 132 selected-chunk stress: former d2base14-density band
# ================================================================
run_case \
  "03_cand132_mode30_d2base14_density" \
  "partial_total=" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "20,40,55,56,57,58,60"

# ================================================================
# 4) 132 selected-chunk stress: late/tail band
# ================================================================
run_case \
  "04_cand132_mode30_late_tail" \
  "partial_total=" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "100,105,110,115,120,125,130"

# ================================================================
# 5) Candidate worker split quick check
# worker 0/4 only.  The exact worker subtotal changes after chunk shaping,
# but the expected full total is still printed as a guard.
# ================================================================
run_case \
  "05_cand132_mode31_worker0of4_quick" \
  "expected_total=314666222712" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 31 8 7 0 0 4 2

# ================================================================
# 6) Full N21 correctness check
# Default RUN_FULL=1.  Set RUN_FULL=0 to skip the long full run.
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
