#!/usr/bin/bash

set -u

# 144 locked validation harness
# - Candidate: 144Py_scorestripe_chunkshape_probe
# - 144 keeps the validated 143 task permutation and all five u32 DFS stacks unchanged.
# - The sole experiment is kernel_dfs_iter_gpu MAXD: 32 -> 24.
# - Independent repeat using the fixed 143 partial totals.
# - No values are learned from the first 144 run: a repeatable depth reduction must reproduce
#   all five subtotals and the full N=21 total exactly.

BASE=${BASE:-./115Py_range_default_clean_cg_v2}
CAND=${CAND:-./144Py_scorestripe_chunkshape_probe}
RUN_FULL=${RUN_FULL:-1}

EXP01=16879968420
EXP02=19219113480
EXP03=16935522136
EXP04=16230260724
EXP05=79383179384
FULL_TOTAL=314666222712

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="144Py_logs_N21_locked_validation_${TS}"
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

echo "[locked-validation-start] ${TS}"
echo "[logdir] ${LOGDIR}"
echo "[BASE] ${BASE}"
echo "[CAND] ${CAND}"
echo "[RUN_FULL] ${RUN_FULL}"
echo "[143-fixed-values] 01=${EXP01} 02=${EXP02} 03=${EXP03} 04=${EXP04} 05=${EXP05} full=${FULL_TOTAL}"

run_case \
  "01_cand144_mode30_standard_7chunk" \
  "partial_total=${EXP01}" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "0,20,40,60,80,100,120"

run_case \
  "02_cand144_mode30_heavy_band" \
  "partial_total=${EXP02}" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "35,40,41,42,47,48,52,53"

run_case \
  "03_cand144_mode30_d2base14_density" \
  "partial_total=${EXP03}" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "20,40,55,56,57,58,60"

run_case \
  "04_cand144_mode30_late_tail" \
  "partial_total=${EXP04}" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "100,105,110,115,120,125,130"

run_case \
  "05_cand144_mode31_worker0of4_quick" \
  "partial_total=${EXP05} expected_total=${FULL_TOTAL}" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 31 8 7 0 0 4 2

if [ "${RUN_FULL}" = "1" ]; then
  run_case \
    "06_cand144_mode31_full_correctness" \
    "${FULL_TOTAL}.*ok" \
    "$CAND" \
      -g 21 21 32 484 1 0 7 31 8 7 0 0 1 2
else
  skip_case \
    "06_cand144_mode31_full_correctness" \
    "RUN_FULL=0"
fi

echo
echo "================================================================"
echo "[summary]"
cat "$SUMMARY"
echo "================================================================"
echo "[done] logs are under: ${LOGDIR}"
