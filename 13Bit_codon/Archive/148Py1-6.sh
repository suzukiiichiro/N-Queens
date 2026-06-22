#!/usr/bin/bash

set -u

# 148 validation harness
# - Candidate: 148Py_scorestripe_chunkshape_probe
# - Base: 147 mark-select candidate, MAXD=21, five u32 DFS stacks, unchanged task order.
# - Sole kernel experiment: keep nld/nrd in u32 through calculation, tests, and stack write-back.
# - Task order and solution arithmetic are unchanged, so cases 1-5 use the fixed 146/147 partial totals;
#   case 6 checks the full N=21 total.

BASE=${BASE:-./115Py_range_default_clean_cg_v2}
CAND=${CAND:-./148Py_scorestripe_chunkshape_probe}
RUN_FULL=${RUN_FULL:-1}

EXP01=16879968420
EXP02=19219113480
EXP03=16935522136
EXP04=16230260724
EXP05=79383179384
FULL_TOTAL=314666222712

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="148Py_logs_N21_validation_${TS}"
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
echo "[146/147-fixed-values] 01=${EXP01} 02=${EXP02} 03=${EXP03} 04=${EXP04} 05=${EXP05} full=${FULL_TOTAL}"

run_case \
  "01_cand148_mode30_standard_7chunk" \
  "partial_total=${EXP01}" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "0,20,40,60,80,100,120"

run_case \
  "02_cand148_mode30_heavy_band" \
  "partial_total=${EXP02}" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "35,40,41,42,47,48,52,53"

run_case \
  "03_cand148_mode30_d2base14_density" \
  "partial_total=${EXP03}" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "20,40,55,56,57,58,60"

run_case \
  "04_cand148_mode30_late_tail" \
  "partial_total=${EXP04}" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "100,105,110,115,120,125,130"

run_case \
  "05_cand148_mode31_worker0of4_quick" \
  "partial_total=${EXP05} expected_total=${FULL_TOTAL}" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 31 8 7 0 0 4 2

if [ "${RUN_FULL}" = "1" ]; then
  run_case \
    "06_cand148_mode31_full_correctness" \
    "${FULL_TOTAL}.*ok" \
    "$CAND" \
      -g 21 21 32 484 1 0 7 31 8 7 0 0 1 2
else
  skip_case \
    "06_cand148_mode31_full_correctness" \
    "RUN_FULL=0"
fi

echo
echo "================================================================"
echo "[summary]"
cat "$SUMMARY"
echo "================================================================"
echo "[done] logs are under: ${LOGDIR}"
