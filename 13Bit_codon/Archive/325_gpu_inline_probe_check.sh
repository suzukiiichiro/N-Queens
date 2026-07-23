#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# 325_gpu_inline_probe_check.sh
#
# Builds and inspects 325_gpu_inline_probe.py to answer one question:
# does Codon inline a plain function called from inside a @gpu.kernel?
#
# This is a standalone diagnostic, unrelated to the N=21 full validation
# harness used for the main solver (325Py_*_validate_N21_full_once.sh).
# It does not check correctness or timing of the N-Queens solver.
#
# Usage:
#   bash 325_gpu_inline_probe_check.sh
#
# What it does:
#   1. codon build -release the probe.
#   2. Run cuobjdump --dump-sass on the resulting binary (static
#      disassembly, no GPU execution required).
#   3. Report whether CALL/RET and separate variant_a/variant_b symbols
#      are present.
#   4. If cuobjdump is unavailable, fall back to a single sudo ncu
#      --section SourceCounters + --page source run (this DOES execute
#      on the GPU, and needs sudo per the 318 findings).
# =============================================================================

SRC=${SRC:-./325_gpu_inline_probe.py}
CAND=${CAND:-./325_gpu_inline_probe}

if [[ ! -f "$SRC" ]]; then
  echo "[error] source not found: $SRC" >&2
  exit 66
fi

echo "[build] codon build -release -o $CAND $SRC"
codon build -release -o "$CAND" "$SRC"
echo "[build] ok"

SASS_TXT="325_probe_sass.txt"

if command -v cuobjdump >/dev/null 2>&1; then
  echo "[check] cuobjdump --dump-sass $CAND"
  cuobjdump --dump-sass "$CAND" > "$SASS_TXT" 2>&1 || true

  if [[ -s "$SASS_TXT" ]]; then
    echo "[check] wrote $SASS_TXT ($(wc -l < "$SASS_TXT") lines)"
    FUNC_COUNT=$(grep -cE "^\s*Function : " "$SASS_TXT" || true)
    CALL_COUNT=$(grep -cE "\bCALL\b" "$SASS_TXT" || true)
    VARIANT_A=$(grep -c "variant_a" "$SASS_TXT" || true)
    VARIANT_B=$(grep -c "variant_b" "$SASS_TXT" || true)
    echo "[result] distinct 'Function :' symbols found: $FUNC_COUNT"
    echo "[result] CALL instruction occurrences:         $CALL_COUNT"
    echo "[result] 'variant_a' symbol occurrences:        $VARIANT_A"
    echo "[result] 'variant_b' symbol occurrences:        $VARIANT_B"
    echo
    if [[ "$CALL_COUNT" -gt 0 && ( "$VARIANT_A" -gt 0 || "$VARIANT_B" -gt 0 ) ]]; then
      echo "[conclusion] CALL present + variant symbols present -> Codon does NOT inline device functions in @gpu.kernel."
      echo "             The device-function specialization design needs a different approach before touching kernel_dfs_iter_gpu_maxd14."
    elif [[ "$FUNC_COUNT" -le 1 && "$CALL_COUNT" -eq 0 ]]; then
      echo "[conclusion] No CALL, single function body -> Codon DOES inline device functions in @gpu.kernel."
      echo "             Safe to prototype the real future_check_mask==0 split next, per 324's plan."
    else
      echo "[conclusion] Ambiguous -- inspect $SASS_TXT manually before drawing a conclusion."
    fi
    exit 0
  else
    echo "[warn] cuobjdump produced no output; falling back to ncu" >&2
  fi
else
  echo "[warn] cuobjdump not found on PATH; falling back to ncu" >&2
fi

echo "[check] sudo ncu --launch-count 1 --section SourceCounters -f -o 325_inline_probe_ncu $CAND"
sudo ncu --launch-count 1 --section SourceCounters -f -o 325_inline_probe_ncu "$CAND"
/usr/local/cuda/bin/ncu --page source --import 325_inline_probe_ncu.ncu-rep 2>&1 | tee 325_inline_probe_ncu_source.txt
echo
echo "[result] Inspect 325_inline_probe_ncu_source.txt for CALL/RET and variant_a/variant_b symbols (see header comment in 325_gpu_inline_probe.py for how to interpret)."
