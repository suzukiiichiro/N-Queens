#!/usr/bin/bash

set -u

# 137 locked validation harness
# - Candidate: 137Py_scorestripe_chunkshape_probe
# - Kernel path remains generic-only.
# - 137 is scorestripe_v5_lanephase32_phasejump from the validated 136 scorestripe_v4_lanephase32 line.
# - Because 137 changes host-side permutation, fixed selected-chunk partial_total values are learned
#   from the latest 137Py1-6.sh validation log directory before this script runs.
# - Set LOCK_SOURCE_DIR=/path/to/137Py_logs_N21_validation_YYYYMMDD_HHMMSS to pin a specific seed run.

BASE=${BASE:-./115Py_range_default_clean_cg_v2}
CAND=${CAND:-./137Py_scorestripe_chunkshape_probe}
RUN_FULL=${RUN_FULL:-1}
LOCK_SOURCE_DIR=${LOCK_SOURCE_DIR:-}

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="137Py_logs_N21_locked_validation_${TS}"
SUMMARY="${LOGDIR}/summary.tsv"

mkdir -p "$LOGDIR"

echo -e "case\texpected_pattern\tstatus\texit_code\tlogfile" > "$SUMMARY"

find_lock_source_dir() {
  if [ -n "$LOCK_SOURCE_DIR" ]; then
    echo "$LOCK_SOURCE_DIR"
    return 0
  fi
  ls -dt 137Py_logs_N21_validation_* 2>/dev/null | head -n 1
}

extract_partial_total() {
  local logfile="$1"
  if [ ! -f "$logfile" ]; then
    echo ""
    return 0
  fi
  grep -Eo 'partial_total=[0-9]+' "$logfile" | tail -n 1 | sed 's/^partial_total=//'
}

LOCK_DIR=$(find_lock_source_dir)
if [ -z "$LOCK_DIR" ] || [ ! -d "$LOCK_DIR" ]; then
  echo "[lock-error] 137 validation log directory not found. Run ./137Py1-6.sh first, or set LOCK_SOURCE_DIR." >&2
  echo -e "lock_source\tmissing\tFAIL_LOCK_SOURCE\t2\t-" >> "$SUMMARY"
  exit 2
fi

EXP01=$(extract_partial_total "${LOCK_DIR}/01_cand137_mode30_standard_7chunk.log")
EXP02=$(extract_partial_total "${LOCK_DIR}/02_cand137_mode30_heavy_band.log")
EXP03=$(extract_partial_total "${LOCK_DIR}/03_cand137_mode30_d2base14_density.log")
EXP04=$(extract_partial_total "${LOCK_DIR}/04_cand137_mode30_late_tail.log")
EXP05=$(extract_partial_total "${LOCK_DIR}/05_cand137_mode31_worker0of4_quick.log")

if [ -z "$EXP01" ] || [ -z "$EXP02" ] || [ -z "$EXP03" ] || [ -z "$EXP04" ] || [ -z "$EXP05" ]; then
  echo "[lock-error] could not extract all 137 fixed partial_total values from ${LOCK_DIR}" >&2
  echo "[lock-error] EXP01=${EXP01} EXP02=${EXP02} EXP03=${EXP03} EXP04=${EXP04} EXP05=${EXP05}" >&2
  echo -e "lock_source\tparse_failed\tFAIL_LOCK_SOURCE\t2\t${LOCK_DIR}" >> "$SUMMARY"
  exit 2
fi

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
    echo "LOCK_DIR  : ${LOCK_DIR}"
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
echo "[lock-source] ${LOCK_DIR}"
echo "[BASE] ${BASE}"
echo "[CAND] ${CAND}"
echo "[RUN_FULL] ${RUN_FULL}"
echo "[locked-values] 01=${EXP01} 02=${EXP02} 03=${EXP03} 04=${EXP04} 05=${EXP05} full=314666222712"

# ================================================================
# 1) 137 selected-chunk smoke: standard spread
# ================================================================
run_case \
  "01_cand137_mode30_standard_7chunk" \
  "partial_total=${EXP01}" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "0,20,40,60,80,100,120"

# ================================================================
# 2) 137 selected-chunk stress: heavy band
# ================================================================
run_case \
  "02_cand137_mode30_heavy_band" \
  "partial_total=${EXP02}" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "35,40,41,42,47,48,52,53"

# ================================================================
# 3) 137 selected-chunk stress: d2base14-density band
# ================================================================
run_case \
  "03_cand137_mode30_d2base14_density" \
  "partial_total=${EXP03}" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "20,40,55,56,57,58,60"

# ================================================================
# 4) 137 selected-chunk stress: late/tail band
# ================================================================
run_case \
  "04_cand137_mode30_late_tail" \
  "partial_total=${EXP04}" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "100,105,110,115,120,125,130"

# ================================================================
# 5) Candidate worker split quick check
# ================================================================
run_case \
  "05_cand137_mode31_worker0of4_quick" \
  "partial_total=${EXP05} expected_total=314666222712" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 31 8 7 0 0 4 2

# ================================================================
# 6) Full N21 correctness check
# Default RUN_FULL=1. Set RUN_FULL=0 to skip the long full run.
# ================================================================
if [ "${RUN_FULL}" = "1" ]; then
  run_case \
    "06_cand137_mode31_full_correctness" \
    "314666222712.*ok" \
    "$CAND" \
      -g 21 21 32 484 1 0 7 31 8 7 0 0 1 2
else
  skip_case \
    "06_cand137_mode31_full_correctness" \
    "RUN_FULL=0"
fi

echo
echo "================================================================"
echo "[summary]"
cat "$SUMMARY"
echo "================================================================"
echo "[done] logs are under: ${LOGDIR}"
