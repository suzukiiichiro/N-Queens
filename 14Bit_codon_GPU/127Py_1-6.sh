#!/usr/bin/bash

set -u

# BASE=${BASE:-./115Py_range_default_clean_cg_v2}
# CAND=${CAND:-./126Py_d2base14_only_adoption}
BASE=./115Py_range_default_clean_cg_v2
CAND=./127Py_generic_split_harness_probe

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs_N21_validation_${TS}"
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

echo "[validation-start] ${TS}"
echo "[logdir] ${LOGDIR}"
echo "[BASE] ${BASE}"
echo "[CAND] ${CAND}"

# ================================================================
# 1) Standard 7-chunk smoke/regression
# expected partial_total = 16337097758
# ================================================================
run_case \
  "01_cand_mode30_standard_7chunk" \
  "partial_total=16337097758" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "0,20,40,60,80,100,120"

# ================================================================
# 2) Heavy-band chunk stress
# expected partial_total = 22253483640
# ================================================================
run_case \
  "02_cand_mode30_heavy_band" \
  "partial_total=22253483640" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "35,40,41,42,47,48,52,53"

# ================================================================
# 3) d2base14-density stress
# expected partial_total = 16871798494
# ================================================================
run_case \
  "03_cand_mode30_d2base14_density" \
  "partial_total=16871798494" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "20,40,55,56,57,58,60"

# ================================================================
# 4) Late/tail stress
# expected partial_total = 16202368694
# ================================================================
run_case \
  "04_cand_mode30_late_tail" \
  "partial_total=16202368694" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 30 8 7 0 0 1 "100,105,110,115,120,125,130"

# ================================================================
# 5-light) Candidate worker split quick check
# worker 0/4 only
# expected worker 0/4 = 79322573818
# ================================================================
run_case \
  "05_cand_mode31_worker0of4_quick" \
  "79322573818" \
  "$CAND" \
    -g 21 21 32 484 1 0 7 31 8 7 0 0 4 2

# ================================================================
# 6) Reorder-bin build/reuse validation only
# expected: records=2025282, chunks=131, rotate_only path
# ================================================================
run_case \
  "06_base_mode28_reorder_build_reuse" \
  "records=2025282" \
  "$BASE" \
    -g 21 21 32 484 1 0 7 28 8 7 0 2

echo
echo "================================================================"
echo "[summary]"
cat "$SUMMARY"
echo "================================================================"
echo "[done] logs are under: ${LOGDIR}"