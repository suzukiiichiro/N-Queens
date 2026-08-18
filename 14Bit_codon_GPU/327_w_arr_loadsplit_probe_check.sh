#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# 327_w_arr_loadsplit_probe_check.sh (round 2: 3 kernels)
#
# Builds and inspects 327_w_arr_loadsplit_probe.py, which now launches
# THREE kernels (w_probe_kernel baseline, w_probe_kernel_tmpvar,
# w_probe_kernel_soa) in one run, to test whether splitting w_arr into
# two separate densely-packed u32 arrays (SoA layout) eliminates the
# "excessive sectors" cost seen on the split 32-bit loads, not just the
# instruction count.
#
# Standalone diagnostic, unrelated to the N=21 full validation harness
# used for the main solver. Does not check solver correctness/timing.
#
# This round needs the ncu path (not just cuobjdump's static SASS dump),
# because we need the L2 Theoretical Sectors Global Excessive column per
# kernel, which only the profiled/counter-based report shows.
#
# Usage:
#   bash 327_w_arr_loadsplit_probe_check.sh
# =============================================================================

SRC=${SRC:-./327_w_arr_loadsplit_probe.py}
CAND=${CAND:-./327_w_arr_loadsplit_probe}

if [[ ! -f "$SRC" ]]; then
  echo "[error] source not found: $SRC" >&2
  exit 66
fi

echo "[build] codon build -release -o $CAND $SRC"
codon build -release -o "$CAND" "$SRC"
echo "[build] ok"

echo "[quick-check] cuobjdump --dump-sass (instruction shapes only, no Excessive-sectors data)"
if command -v cuobjdump >/dev/null 2>&1; then
  cuobjdump --dump-sass "$CAND" > 327_probe_sass_v2.txt 2>&1 || true
  echo "[quick-check] wrote 327_probe_sass_v2.txt; LDG.E lines:"
  grep -nE "LDG\.E" 327_probe_sass_v2.txt || true
  echo
fi

echo "[main-check] sudo ncu --launch-count 3 --section SourceCounters -f -o 327_w_arr_probe_ncu_v2 $CAND"
sudo ncu --launch-count 3 --section SourceCounters -f -o 327_w_arr_probe_ncu_v2 "$CAND" 2>&1 | tee 327_w_arr_probe_ncu_v2_run.log
echo "[main-check] program stdout (checksum/sum_base/sum_soa/match) captured in 327_w_arr_probe_ncu_v2_run.log"
/usr/local/cuda/bin/ncu --page source --import 327_w_arr_probe_ncu_v2.ncu-rep 2>&1 | tee 327_w_arr_probe_ncu_v2_source.txt

echo
echo "[result] 327_w_arr_probe_ncu_v2_source.txt now contains THREE 'Kernel Name' sections"
echo "         (w_probe_kernel, w_probe_kernel_tmpvar, w_probe_kernel_soa)."
echo "         For each, find the LDG.E row(s) and compare the L2 Theoretical Sectors"
echo "         Global Excessive column. See the header comment in"
echo "         327_w_arr_loadsplit_probe.py for how to interpret the comparison."
echo "         Also check 327_w_arr_probe_ncu_v2_run.log for the program's own stdout"
echo "         ('checksum:', 'sum_base:', 'sum_soa:', 'match:') to confirm the SoA"
echo "         reconstruction produced identical results to the baseline."

