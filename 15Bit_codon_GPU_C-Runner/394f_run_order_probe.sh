#!/usr/bin/env bash
# 394f_run_order_probe.sh
#
# rev394f -- is the raw order good BECAUSE adjacent constellations are
# similar? And can a simple sort do better?
#
# WHAT 394e FOUND (2026-09-08, all six cells oracle-MATCH)
# --------------------------------------------------------
#   raw   @484 201,234   @485 200,061 (-0.58%)        raw is stride-insensitive
#   base  @484 296,899   @485 282,374   @483 290,851   +40..+47% survives the shift
#   shaped@485 260,624 (= @484 to -0.17%)
# => M1: the penalty lives in the 32-record warp groups, not in the
#    per-thread sequence. M2 (launch tail) is worth <= ~7 points.
#
# WHY (from the source): the broadmarktail base lays each chunk out by
# interleave_broad_markdist_tail_subparts(), a RECORD-BY-RECORD round-robin
# of three parts (F17 / GH / R), i.e. it deliberately spreads the heavy
# records evenly over every warp (a rev-111-113 inter-warp balancing
# idea). The C kernel reconverges all 32 lanes at the end of EVERY
# grid-stride iteration (BSYNC B4 at the outer loop), so a warp's time
# per iteration is its slowest lane: a warp with 6 heavy lanes and 26
# light ones is paced by the 6. The generator's natural order puts
# adjacent (i,j,k,l) constellations -- near-isomorphic subtrees -- in the
# same warp, so max ~= mean and the wait is small. That is the hidden
# advantage the reorder destroys.
#
# THIS REVISION (kernel untouched, no Codon, no sudo)
# ---------------------------------------------------
# 394f_permute_soa7.py builds permutations of the RAW filtered input:
#   X   random shuffle           the CONTROL. If adjacency is the lever,
#                                X is as bad as base (+30..+50%).
#   L1  sort by (col, ld, rd)    root-state similarity
#   L2  sort by (-popcount(free), col, ld, rd)   cost proxy first
#   L3  sort by (markctrl, ctrl0) schedule similarity, raw adjacency kept
#                                within equal keys = funcid grouping
#                                WITHOUT the interleave
# plus R (raw, anchor). Each is oracle-gated (a permutation cannot change
# the total; the tool also checks an order-independent checksum before
# writing). The per-thread results[] arrays are copied and summarised
# (max/mean, CV) as a free inter-thread-balance proxy.
#
# PRE-REGISTERED (394f_README_append.md):
#   R within 1% of 201,237
#   X >= +30% vs R   (falsified if within +-5%: adjacency is NOT the lever)
#   L1 <= R - 3%     (stated prediction; falsified if L1 >= R)
#   L3 <= R          (schedule grouping without interleave is harmless or better)
#   L2 between L1 and R
#
# USAGE
#   STATIC_ONLY=1 bash 394f_run_order_probe.sh
#                 bash 394f_run_order_probe.sh              # R X L1 L2 L3 (~20 min)
#   CELLS="R X" bash 394f_run_order_probe.sh                # control only

set -u

REV="394f"
CU_SRC="${CU_SRC:-394f_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-394f_kernel_maxd14}"
REF_CU="${REF_CU:-394e_kernel_maxd14.cu}"
PERMUTE="${PERMUTE:-394f_permute_soa7.py}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
ORACLE="${ORACLE:-314666222712}"
MB="${MB:-484}"
IN_RAW="${IN_RAW:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
CELLS="${CELLS:-R X L1 L2 L3}"
SEED="${SEED:-394}"
REF_RAW_MS="${REF_RAW_MS:-201237}"
COOLDOWN="${COOLDOWN:-20}"
STATIC_ONLY="${STATIC_ONLY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_order_${TS}}"
TSV="$LOGDIR/${REV}_order.tsv"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }

# ---------------------------------------------------------------------
# 1. Static
# ---------------------------------------------------------------------
for f in "$CU_SRC" "$PERMUTE" "$IN_RAW"; do [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"; done
[[ -f "$IN_RAW" ]] && { sz=$(stat -c%s "$IN_RAW"); (( sz == 2025282*28 )) && pass "raw_input_full (2025282 records)" || fail "raw_input_full" "size $sz"; }
if [[ -f "$REF_CU" ]]; then
  A=$(awk 'f||/^#include/{f=1;print}' "$REF_CU" | sha256sum | cut -d' ' -f1); B=$(awk 'f||/^#include/{f=1;print}' "$CU_SRC" | sha256sum | cut -d' ' -f1)
  [[ "$A" == "$B" ]] && pass "cu_code_region_identical_to_394e (${B:0:16}...)" || fail "cu_code_region_identical_to_394e" "differs"
else info "cu_code_region_identical_to_394e" "skipped"; fi
python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$PERMUTE" 2>/dev/null && pass "permute_tool_parses" || fail "permute_tool_parses" "syntax error in $PERMUTE"
[[ ! -x "$NVCC" ]] && ! command -v nvcc >/dev/null 2>&1 && fail "nvcc_present" "missing"; [[ ! -x "$NVCC" ]] && NVCC="nvcc"
for c in $CELLS; do [[ "$c" =~ ^(R|X|L1|L2|L3)$ ]] || fail "cell_name_valid[$c]" "expected R|X|L1|L2|L3"; done
if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; nvidia-smi -q -d CLOCK 2>&1
  sha256sum "$CU_SRC" "$PERMUTE" "$IN_RAW" 2>&1; echo "CELLS=$CELLS MB=$MB SEED=$SEED"; } > "$LOGDIR/00_env_pre.txt" 2>&1

banner "Building $CU_SRC"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -o "$CU_BIN" "$CU_SRC" 2>&1 | tee "$LOGDIR/01_nvcc_build.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build_succeeded" || { fail "nvcc_build_succeeded" "no binary"; exit 1; }
head -c $((15488*28)) "$IN_RAW" > "/tmp/${REV}_head.bin"
"./$CU_BIN" "$NQ" "/tmp/${REV}_head.bin" "/tmp/${REV}_head_out.bin" > "$LOGDIR/02_head_probe.log" 2>&1 || true
[[ "$(grep -o 'total_sum=[0-9]*' "$LOGDIR/02_head_probe.log" | head -1 | cut -d= -f2)" == "2196649880" ]] && pass "head_slice_total_reproduces" || { fail "head_slice_total_reproduces" "see log"; exit 1; }

# ---------------------------------------------------------------------
# 2. Build the permutations (seconds each; checksum-guarded by the tool)
# ---------------------------------------------------------------------
banner "Building permutations of $IN_RAW"
declare -A INPUT
INPUT[R]="$IN_RAW"
build_perm() {
  local cell="$1"
  local mode="$2"
  local out="${REV}_input_${cell}_${mode}.bin"
  if [[ -f "$out" && "$(stat -c%s "$out")" -eq $((2025282*28)) ]]; then info "perm[$cell]" "reusing $out"; else
    python3 "$PERMUTE" "$mode" "$IN_RAW" "$out" "$SEED" 2>&1 | tee "$LOGDIR/03_permute_${cell}.log"
    [[ -f "$out" ]] && grep -q '\[permute-done\]' "$LOGDIR/03_permute_${cell}.log" && pass "perm_built[$cell:$mode]" || { fail "perm_built[$cell:$mode]" "tool did not produce $out"; return 1; }
  fi
  INPUT[$cell]="$out"
}
for c in $CELLS; do
  case $c in X) build_perm X random || exit 1;; L1) build_perm L1 state || exit 1;; L2) build_perm L2 free || exit 1;; L3) build_perm L3 sched || exit 1;; esac
done

# ---------------------------------------------------------------------
# 3. Runs
# ---------------------------------------------------------------------
printf 'cell\tinput\tmax_blocks\tstride\ttotal_sum\tmatch\tkernel_ms\tdelta_vs_R_pct\tdelta_vs_389_ref_pct\tthread_max_over_mean\tthread_cv_pct\tsm_clock_mhz\ttemp_c\tstart\n' > "$TSV"
declare -A KMS
thread_stats() {  # per-thread solution totals from the results bin: max/mean and CV (%)
  python3 - "$1" <<'EOF'
import sys,array,math
a=array.array('Q'); a.frombytes(open(sys.argv[1],'rb').read())
n=len(a); m=sum(a)/n; mx=max(a); var=sum((x-m)**2 for x in a)/n
print(f"{mx/m:.4f} {100*math.sqrt(var)/m:.2f}")
EOF
}
run_cell() {
  local cell="$1" src="${INPUT[$1]}"
  banner "cell $cell: input=$(basename "$src" | cut -c1-70) NQ_MAX_BLOCKS=$MB"
  local clk tmp start log total kms stride_act match rb stats
  clk="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  tmp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  start="$(date -Is)"; log="$LOGDIR/1_${cell}.log"; rb="/tmp/${REV}_${cell}_results.bin"
  NQ_MAX_BLOCKS="$MB" "./$CU_BIN" "$NQ" "$src" "$rb" "$ORACLE" 2>&1 | tee "$log"
  total="$(grep -o 'total_sum=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  kms="$(grep -o 'kernel_ms=[0-9.]*' "$log" | head -1 | cut -d= -f2)"
  stride_act="$(grep -o 'stride=[0-9]*' "$log" | tail -1 | cut -d= -f2)"
  match=0; grep -q '\[gpu-run-correctness\] MATCH' "$log" && match=1
  stats="$(thread_stats "$rb" 2>/dev/null || echo '? ?')"; cp "$rb" "$LOGDIR/results_${cell}.bin" 2>/dev/null || true
  local dR="NA" dRef="NA"
  [[ -n "${KMS[R]:-}" && -n "${kms:-}" ]] && dR="$(awk -v a="${KMS[R]}" -v k="$kms" 'BEGIN{printf "%+.3f",(k-a)/a*100}')"
  [[ -n "${kms:-}" ]] && dRef="$(awk -v a="$REF_RAW_MS" -v k="$kms" 'BEGIN{printf "%+.3f",(k-a)/a*100}')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$(basename "$src")" "$MB" "${stride_act:-?}" "${total:-?}" "$match" "${kms:-?}" "$dR" "$dRef" "${stats% *}" "${stats#* }" "$clk" "$tmp" "$start" >> "$TSV"
  if [[ "$match" -eq 1 && "${total:-}" == "$ORACLE" ]]; then pass "oracle_match[$cell]"; else fail "oracle_match[$cell]" "total_sum='${total:-<none>}' match=$match"; return 1; fi
  KMS[$cell]="$kms"; info "cell[$cell]" "kernel_ms=$kms vs R ${dR}%  thread max/mean=${stats% *} cv=${stats#* }%"
  return 0
}
first=1
for c in $CELLS; do (( first )) || { echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; }; first=0; run_cell "$c" || break; done

# ---------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------
banner "Evaluation"
absdev() { awk -v a="$1" -v r="$2" 'BEGIN{d=(a-r)/r*100; printf "%.3f",(d<0?-d:d)}'; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%+.2f",(b-a)/a*100}'; }
if [[ -n "${KMS[R]:-}" ]]; then d="$(absdev "${KMS[R]}" "$REF_RAW_MS")"; awk -v d="$d" 'BEGIN{exit !(d<=1.0)}' && pass "R_reproduces_389 (${KMS[R]}, ${d}%)" || fail "R_reproduces_389" "${d}% off"
  for c in X L1 L2 L3; do [[ -n "${KMS[$c]:-}" ]] && info "delta_vs_R[$c]" "$(pct "${KMS[R]}" "${KMS[$c]}")%"; done
  if [[ -n "${KMS[X]:-}" ]]; then dx="$(pct "${KMS[R]}" "${KMS[X]}")"
    if awk -v a="$dx" 'BEGIN{exit !(a>=30)}'; then info "verdict_control" "X ${dx}%: adjacency IS the lever (random order is as bad as the reorder)"
    elif awk -v a="$dx" 'BEGIN{exit !(a<=5 && a>=-5)}'; then info "verdict_control" "X ${dx}%: adjacency is NOT the lever -- the broadmarktail interleave is bad for a specific reason"
    else info "verdict_control" "X ${dx}%: partial -- adjacency matters but explains only part of the base penalty"; fi
  fi
  for c in L1 L2 L3; do [[ -n "${KMS[$c]:-}" ]] && awk -v a="$(pct "${KMS[R]}" "${KMS[$c]}")" 'BEGIN{exit !(a<=-3)}' && info "note[$c]" "BELOW raw by >=3%: a sort beats the generator order -- new lever"; done
fi
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi -q -d CLOCK 2>&1; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
echo; python3 - "$TSV" <<'EOF' 2>/dev/null || cat "$TSV"
import csv,sys
rows=list(csv.reader(open(sys.argv[1]),delimiter='\t'))
w=[max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
for r in rows: print('  '.join(c.ljust(w[i]) for i,c in enumerate(r)))
EOF
TARBALL="${LOGDIR}.tar.gz"
tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed -- send $LOGDIR/"
echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; echo "tsv: $TSV"; echo "tarball: $TARBALL"
if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
echo "${REV} PASSED. Send $TARBALL for analysis."
exit 0
