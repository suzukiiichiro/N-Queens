#!/usr/bin/env bash
# 394g_sweep_fine_and_L3.sh
#
# rev394g -- bracket the occupancy optimum below 968, and confirm the L3
# (schedule-sorted, stable) input composes with it.
#
# WHAT WE KNOW (all real hardware, all oracle-MATCH, 2026-09-08)
# --------------------------------------------------------------
#   MAX_BLOCKS on raw:  484 201,231 | 968 163,185 | 1024 187,719 | 1280 252,207
#                       -> steep L1-pressure ramp starts right above 968;
#                          nothing measured between 484 and 968 yet.
#   order at 484:       raw 201,238 | random +44.2% | (col,ld,rd) +20.8% |
#                       free-count +21.2% | L3 schedule-stable -1.71%
#                       -> the generator's sibling adjacency is worth ~44%;
#                          L3 keeps 99.4% of it and adds schedule homogeneity.
#
# DESIGN: 5 points x {raw, L3}, paired so thermal drift hits both alike
# ----------------------------------------------------------------------
#   484 (anchor, raw only)  704  800  864  928  968   (warps/SM 6.05 .. 12.1)
# 11 full N=21 runs, ~3.5 min each, cooldown 15 s -> ~40 min. No sudo.
# The L3 input is rebuilt with 394f_permute_soa7.py if absent (seconds).
#
# PRE-REGISTERED (394g_README_append.md)
#   484 raw within 1% of 201,237; 968 raw within 2% of 160,528..163,197
#   MB* (raw optimum) is in [864, 968]  (falsified if 704 or 800 is best)
#   L3 vs raw at every point: -0.7% .. -2.7%  (i.e. -1.7 +- 1), same sign
#   at all points -> the two effects compose (multiplicatively)
#
# USAGE
#   STATIC_ONLY=1 bash 394g_sweep_fine_and_L3.sh
#                 bash 394g_sweep_fine_and_L3.sh
#   POINTS="864 928" bash 394g_sweep_fine_and_L3.sh

set -u

REV="394g"
CU_SRC="${CU_SRC:-394g_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-394g_kernel_maxd14}"
REF_CU="${REF_CU:-394f_kernel_maxd14.cu}"
PERMUTE="${PERMUTE:-394f_permute_soa7.py}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
ORACLE="${ORACLE:-314666222712}"
IN_RAW="${IN_RAW:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
IN_L3="${IN_L3:-394f_input_L3_sched.bin}"
POINTS="${POINTS:-704 800 864 928 968}"
ANCHOR_MB="${ANCHOR_MB:-484}"
REF_484_MS="${REF_484_MS:-201237}"
REF_968_MS="${REF_968_MS:-161900}"      # midpoint of the 160,528..163,197 band seen so far
COOLDOWN="${COOLDOWN:-15}"
STATIC_ONLY="${STATIC_ONLY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_fine_${TS}}"
TSV="$LOGDIR/${REV}_fine.tsv"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }

# ---------------------------------------------------------------------
# 1. Static
# ---------------------------------------------------------------------
for f in "$CU_SRC" "$PERMUTE" "$IN_RAW"; do [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"; done
[[ -f "$IN_RAW" ]] && { sz=$(stat -c%s "$IN_RAW"); (( sz == 2025282*28 )) && pass "raw_input_full" || fail "raw_input_full" "size $sz"; }
if [[ -f "$REF_CU" ]]; then
  A=$(awk 'f||/^#include/{f=1;print}' "$REF_CU" | sha256sum | cut -d' ' -f1); B=$(awk 'f||/^#include/{f=1;print}' "$CU_SRC" | sha256sum | cut -d' ' -f1)
  [[ "$A" == "$B" ]] && pass "cu_code_region_identical_to_394f (${B:0:16}...)" || fail "cu_code_region_identical_to_394f" "differs"
else info "cu_code_region_identical_to_394f" "skipped"; fi
[[ ! -x "$NVCC" ]] && ! command -v nvcc >/dev/null 2>&1 && fail "nvcc_present" "missing"; [[ ! -x "$NVCC" ]] && NVCC="nvcc"
for p in $POINTS; do [[ "$p" =~ ^[0-9]+$ ]] && (( p>=1 && p<=65535 )) || fail "point_valid[$p]" "bad"; done
if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; nvidia-smi -q -d CLOCK 2>&1
  sha256sum "$CU_SRC" "$PERMUTE" "$IN_RAW" 2>&1; [[ -f "$IN_L3" ]] && sha256sum "$IN_L3"; echo "POINTS=$POINTS ANCHOR_MB=$ANCHOR_MB"; } > "$LOGDIR/00_env_pre.txt" 2>&1

banner "Building $CU_SRC"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -o "$CU_BIN" "$CU_SRC" 2>&1 | tee "$LOGDIR/01_nvcc_build.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build_succeeded" || { fail "nvcc_build_succeeded" "no binary"; exit 1; }
head -c $((15488*28)) "$IN_RAW" > "/tmp/${REV}_head.bin"
"./$CU_BIN" "$NQ" "/tmp/${REV}_head.bin" "/tmp/${REV}_head_out.bin" > "$LOGDIR/02_head_probe.log" 2>&1 || true
[[ "$(grep -o 'total_sum=[0-9]*' "$LOGDIR/02_head_probe.log" | head -1 | cut -d= -f2)" == "2196649880" ]] && pass "head_slice_total_reproduces" || { fail "head_slice_total_reproduces" "see log"; exit 1; }

# L3 input (schedule-sorted, stable) -- rebuild if absent
if [[ -f "$IN_L3" && "$(stat -c%s "$IN_L3")" -eq $((2025282*28)) ]]; then info "l3_input" "reusing $IN_L3"; else
  python3 "$PERMUTE" sched "$IN_RAW" "$IN_L3" 2>&1 | tee "$LOGDIR/03_permute_L3.log"
  [[ -f "$IN_L3" ]] && grep -q '\[permute-done\]' "$LOGDIR/03_permute_L3.log" && pass "l3_input_built" || { fail "l3_input_built" "permute tool failed"; exit 1; }
fi

# ---------------------------------------------------------------------
# 2. Runs
# ---------------------------------------------------------------------
printf 'idx\tmax_blocks\tinput\tstride\tk_max\ttotal_sum\tmatch\tkernel_ms\tdelta_vs_484raw_pct\tL3_vs_raw_same_point_pct\tsm_clock_mhz\ttemp_c\tstart\n' > "$TSV"
declare -A KMS
run_one() {
  local idx="$1" mb="$2" kind="$3"
  local src="$IN_RAW"; [[ "$kind" == "L3" ]] && src="$IN_L3"
  banner "run $idx: MAX_BLOCKS=$mb input=$kind"
  local clk tmp start log total kms stride_act kmax match
  clk="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  tmp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  start="$(date -Is)"; log="$LOGDIR/1${idx}_mb${mb}_${kind}.log"
  NQ_MAX_BLOCKS="$mb" "./$CU_BIN" "$NQ" "$src" "/tmp/${REV}_mb${mb}_${kind}.bin" "$ORACLE" 2>&1 | tee "$log"
  total="$(grep -o 'total_sum=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  kms="$(grep -o 'kernel_ms=[0-9.]*' "$log" | head -1 | cut -d= -f2)"
  stride_act="$(grep -o 'stride=[0-9]*' "$log" | tail -1 | cut -d= -f2)"
  kmax="$(grep -o 'k_per_thread_max=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  match=0; grep -q '\[gpu-run-correctness\] MATCH' "$log" && match=1
  local dA="NA" dL="NA"
  [[ -n "${KMS[${ANCHOR_MB}_raw]:-}" && -n "${kms:-}" ]] && dA="$(awk -v a="${KMS[${ANCHOR_MB}_raw]}" -v k="$kms" 'BEGIN{printf "%+.3f",(k-a)/a*100}')"
  [[ "$kind" == "L3" && -n "${KMS[${mb}_raw]:-}" && -n "${kms:-}" ]] && dL="$(awk -v a="${KMS[${mb}_raw]}" -v k="$kms" 'BEGIN{printf "%+.3f",(k-a)/a*100}')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$idx" "$mb" "$kind" "${stride_act:-?}" "${kmax:-?}" "${total:-?}" "$match" "${kms:-?}" "$dA" "$dL" "$clk" "$tmp" "$start" >> "$TSV"
  if [[ "$match" -eq 1 && "${total:-}" == "$ORACLE" ]]; then pass "oracle_match[$mb/$kind]"; else fail "oracle_match[$mb/$kind]" "total_sum='${total:-<none>}' match=$match"; return 1; fi
  [[ "${stride_act:-}" == "$((32*mb))" ]] && pass "stride_as_intended[$mb]" || { fail "stride_as_intended[$mb]" "got ${stride_act:-?}"; return 1; }
  KMS[${mb}_${kind}]="$kms"; info "run[$mb/$kind]" "kernel_ms=$kms  vs 484raw ${dA}%  L3-vs-raw@point ${dL}%"
  return 0
}
i=0
run_one $((++i)) "$ANCHOR_MB" raw || exit 1
for mb in $POINTS; do
  for kind in raw L3; do
    echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"
    run_one $((++i)) "$mb" "$kind" || exit 1
  done
done

# ---------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------
banner "Evaluation"
absdev() { awk -v a="$1" -v r="$2" 'BEGIN{d=(a-r)/r*100; printf "%.3f",(d<0?-d:d)}'; }
if [[ -n "${KMS[484_raw]:-}" ]]; then d="$(absdev "${KMS[484_raw]}" "$REF_484_MS")"; awk -v d="$d" 'BEGIN{exit !(d<=1.0)}' && pass "484_raw_reproduces_389 (${KMS[484_raw]}, ${d}%)" || fail "484_raw_reproduces_389" "${d}% off"; fi
if [[ -n "${KMS[968_raw]:-}" ]]; then d="$(absdev "${KMS[968_raw]}" "$REF_968_MS")"; awk -v d="$d" 'BEGIN{exit !(d<=2.0)}' && pass "968_raw_within_2pct_of_band (${KMS[968_raw]})" || info "968_raw_within_2pct_of_band" "${KMS[968_raw]} is ${d}% off the 160,528..163,197 midpoint"; fi
echo; printf '  %-6s %12s %12s %9s %9s\n' "MB" "raw_ms" "L3_ms" "raw/484" "L3/raw"
best_raw=""; best_raw_mb=""; best_l3=""; best_l3_mb=""
for mb in $ANCHOR_MB $POINTS; do
  r="${KMS[${mb}_raw]:-}"; l="${KMS[${mb}_L3]:-}"
  [[ -n "$r" ]] || continue
  dr="$(awk -v a="${KMS[484_raw]}" -v k="$r" 'BEGIN{printf "%+.2f%%",(k-a)/a*100}')"
  dl="NA"; [[ -n "$l" ]] && dl="$(awk -v a="$r" -v k="$l" 'BEGIN{printf "%+.2f%%",(k-a)/a*100}')"
  printf '  %-6s %12s %12s %9s %9s\n' "$mb" "$r" "${l:-NA}" "$dr" "$dl"
  if [[ -z "$best_raw" ]] || awk -v a="$r" -v b="$best_raw" 'BEGIN{exit !(a<b)}'; then best_raw="$r"; best_raw_mb="$mb"; fi
  if [[ -n "$l" ]] && { [[ -z "$best_l3" ]] || awk -v a="$l" -v b="$best_l3" 'BEGIN{exit !(a<b)}'; }; then best_l3="$l"; best_l3_mb="$mb"; fi
done
[[ -n "$best_raw" ]] && info "best_raw" "MAX_BLOCKS=$best_raw_mb kernel_ms=$best_raw ($(awk -v a="$REF_484_MS" -v k="$best_raw" 'BEGIN{printf "%+.2f",(k-a)/a*100}')% vs 389 anchor)"
[[ -n "$best_l3" ]] && info "best_L3" "MAX_BLOCKS=$best_l3_mb kernel_ms=$best_l3 ($(awk -v a="$REF_484_MS" -v k="$best_l3" 'BEGIN{printf "%+.2f",(k-a)/a*100}')% vs 389 anchor) -> N=21 ~ $(awk -v k="$best_l3" 'BEGIN{s=k/1000; printf "%d:%04.1f", int(s/60), s-60*int(s/60)}')"
if [[ -n "$best_raw_mb" ]]; then
  if [[ "$best_raw_mb" == "704" || "$best_raw_mb" == "800" ]]; then info "prereg_optimum_in_864_968" "FALSIFIED: best raw point is $best_raw_mb"; else info "prereg_optimum_in_864_968" "held: best raw point is $best_raw_mb"; fi
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
