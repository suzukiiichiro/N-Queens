#!/usr/bin/env bash
# 394e_run_stride_decouple.sh
#
# rev394e -- is the reorder's damage intra-warp or inter-thread?
#
# WHAT 394d FOUND (2026-09-08, all six rungs oracle-MATCH, 484x32)
# ----------------------------------------------------------------
#   A   raw                            201,238 ms
#   E1  broadmarktail base only        296,902 ms   +47.5%   <- the jump
#   E2  + scorestripe                  295,850 ms   +47.0%
#   E3  + bucket_run=2048              288,149 ms   +43.2%
#   E4  + iter_sort=1                  258,831 ms   +28.6%
#   E5  + isort9 (= 394c B)            261,063 ms   +29.7%   (B reproduced to 0.001%)
# H_isort refuted: the whole penalty is in the broadmarktail base
# (funcid w3_j7 reorder + rotate); every later stage only repairs it.
#
# TWO MECHANISMS, ONE CHEAP DISCRIMINATOR
# --------------------------------------
# M1 intra-warp: the base reorder makes the 32 lanes of a warp more
#    divergent than the generator's natural order.
# M2 inter-thread: the base reorder makes per-THREAD cumulative work
#    (131 tasks) uneven; the rotation meant to balance it does not,
#    and the launch ends on a long tail. (Raw has 99.8% residency per
#    393 s3, i.e. essentially no tail -- so there is room for one.)
#
# The C runner maps record i -> thread (i mod stride), lane (i mod 32).
# For ANY stride that is a multiple of 32, the 32-record warp groups are
# identical; only the per-thread SEQUENCE changes. So running the SAME
# 484-shaped file at MAX_BLOCKS=485 (stride 15,520) or 483 (15,456)
# keeps M1's cause fixed and scrambles M2's: each thread then walks a
# diagonal through the design's [iteration x warp] grid instead of
# staying in one column.
#   M1 => E1@483 ~= E1@484 ~= E1@485 (all ~ +47%)
#   M2 => E1@483 and E1@485 collapse toward raw; possibly BELOW raw if
#         warp homogeneity is worth anything once the tail is gone.
#
# CELLS (raw, base, shaped inputs already built by 394c/394d; driven
# DIRECTLY, no Codon, no shaping, no sudo)
#   R484   raw    @484   anchor
#   R485   raw    @485   control: raw must be stride-insensitive
#   B484   base   @484   repeat of 394d E1 (296,902)
#   B485   base   @485   the test
#   B483   base   @483   the test, other direction
#   S485   shaped @485   the full pipeline under the same trick
# ~6 x 3.4..5 min ~= 25 min.
#
# PRE-REGISTERED (394e_README_append.md): stated prediction is M2.
#   R485 within 1% of R484
#   B484 within 1% of 296,902
#   M2 if B485 and B483 are both <= +10% vs R484;  M1 if both >= +40%
#   anything in between: mixed, and the ncu stage (NCU=1, sudo) decides
#
# OPTIONAL (NCU=1, sudo): SchedulerStats + WarpStateStats on the first
# 743,424 records (48 iterations) of raw and base at 484. Active Warps
# Per Scheduler falling from ~1.51 = tail (M2); Avg Active Threads Per
# Warp falling from ~10 = lanes (M1). ~9 min each.
#
# USAGE
#   STATIC_ONLY=1 bash 394e_run_stride_decouple.sh
#                 bash 394e_run_stride_decouple.sh
#   CELLS="R484 B484 B485" bash 394e_run_stride_decouple.sh
#   NCU=1         bash 394e_run_stride_decouple.sh

set -u

REV="394e"
CU_SRC="${CU_SRC:-394e_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-394e_kernel_maxd14}"
REF_CU="${REF_CU:-394d_kernel_maxd14.cu}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
ORACLE="${ORACLE:-314666222712}"
IN_RAW="${IN_RAW:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
IN_BASE="${IN_BASE:-constellations_N21_6_broadmarktail_reorder_v4_rotate_only_w3_j7_b32_m484_s15488.bin.soa_ref_361.bin.maxd14only_363.bin}"
IN_SHAPED="${IN_SHAPED:-constellations_N21_6_chunkshape148_scorestripe_v9_lanephase32_octetfirstpairlock29_v4_rotate_only_w3_j7_b32_m484_s15488_run2048_isort9.bin.soa_ref_361.bin.maxd14only_363.bin}"
CELLS="${CELLS:-R484 R485 B484 B485 B483 S485}"
NCU_STAGE="${NCU:-0}"
REF_RAW_MS="${REF_RAW_MS:-201237}"
REF_BASE_MS="${REF_BASE_MS:-296902}"
COOLDOWN="${COOLDOWN:-20}"
STATIC_ONLY="${STATIC_ONLY:-0}"

NCU_BIN="${NCU_BIN:-$(command -v ncu 2>/dev/null)}"
[[ -z "$NCU_BIN" && -x /usr/local/cuda/bin/ncu ]] && NCU_BIN="/usr/local/cuda/bin/ncu"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_stride_${TS}}"
TSV="$LOGDIR/${REV}_stride.tsv"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }
want() { [[ " $CELLS " == *" $1 "* ]]; }

if [[ "$NCU_STAGE" == "1" ]]; then
  sudo -n true 2>/dev/null && pass "sudo_noninteractive_available" || { fail "sudo_noninteractive_available" "NCU=1 needs sudo"; exit 1; }
fi

# ---------------------------------------------------------------------
# 1. Static
# ---------------------------------------------------------------------
[[ -f "$CU_SRC" ]] && pass "file_present[$CU_SRC]" || fail "file_present[$CU_SRC]" "missing"
for f in "$IN_RAW" "$IN_BASE" "$IN_SHAPED"; do
  if [[ -f "$f" ]]; then
    sz=$(stat -c%s "$f"); if (( sz == 2025282*28 )); then pass "input_present_and_full[$(echo $f | cut -c1-60)...]"; else fail "input_present_and_full[$f]" "size $sz != $((2025282*28))"; fi
  else fail "input_present[$f]" "not found -- 394c/394d should have built it"; fi
done
if [[ -f "$REF_CU" ]]; then
  A=$(awk 'f||/^#include/{f=1;print}' "$REF_CU" | sha256sum | cut -d' ' -f1); B=$(awk 'f||/^#include/{f=1;print}' "$CU_SRC" | sha256sum | cut -d' ' -f1)
  [[ "$A" == "$B" ]] && pass "cu_code_region_identical_to_394d (${B:0:16}...)" || fail "cu_code_region_identical_to_394d" "differs"
else info "cu_code_region_identical_to_394d" "skipped"; fi
grep -q 'getenv("NQ_MAX_BLOCKS")' "$CU_SRC" && pass "cu_reads_NQ_MAX_BLOCKS" || fail "cu_reads_NQ_MAX_BLOCKS" "missing"
[[ ! -x "$NVCC" ]] && ! command -v nvcc >/dev/null 2>&1 && fail "nvcc_present" "missing"; [[ ! -x "$NVCC" ]] && NVCC="nvcc"
for c in $CELLS; do [[ "$c" =~ ^[RBS](48[345])$ ]] || fail "cell_name_valid[$c]" "expected R|B|S followed by 483|484|485"; done
if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; nvidia-smi -q -d CLOCK 2>&1
  sha256sum "$CU_SRC" "$IN_RAW" "$IN_BASE" "$IN_SHAPED" 2>&1; echo "CELLS=$CELLS NCU=$NCU_STAGE"; } > "$LOGDIR/00_env_pre.txt" 2>&1

banner "Building $CU_SRC"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -o "$CU_BIN" "$CU_SRC" 2>&1 | tee "$LOGDIR/01_nvcc_build.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build_succeeded" || { fail "nvcc_build_succeeded" "no binary"; exit 1; }
head -c $((15488*28)) "$IN_RAW" > "/tmp/${REV}_head.bin"
"./$CU_BIN" "$NQ" "/tmp/${REV}_head.bin" "/tmp/${REV}_head_out.bin" > "$LOGDIR/02_head_probe.log" 2>&1 || true
[[ "$(grep -o 'total_sum=[0-9]*' "$LOGDIR/02_head_probe.log" | head -1 | cut -d= -f2)" == "2196649880" ]] && pass "head_slice_total_reproduces" || { fail "head_slice_total_reproduces" "see $LOGDIR/02_head_probe.log"; exit 1; }

# ---------------------------------------------------------------------
# 2. Cells
# ---------------------------------------------------------------------
printf 'cell\tinput\tmax_blocks\tstride_actual\tk_max\ttotal_sum\tmatch\tkernel_ms\tdelta_vs_R484_pct\tdelta_vs_389_ref_pct\tsm_clock_mhz\ttemp_c\tstart\n' > "$TSV"
declare -A KMS
run_cell() {
  local cell="$1"; local kind="${cell:0:1}" mb="${cell:1}"
  local src label
  case "$kind" in R) src="$IN_RAW"; label=raw;; B) src="$IN_BASE"; label=base;; S) src="$IN_SHAPED"; label=shaped;; esac
  banner "cell $cell: input=$label NQ_MAX_BLOCKS=$mb (stride $((32*mb)))"
  local clk tmp start log total kms stride_act kmax match
  clk="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  tmp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  start="$(date -Is)"; log="$LOGDIR/1_${cell}.log"
  NQ_MAX_BLOCKS="$mb" "./$CU_BIN" "$NQ" "$src" "/tmp/${REV}_${cell}.bin" "$ORACLE" 2>&1 | tee "$log"
  total="$(grep -o 'total_sum=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  kms="$(grep -o 'kernel_ms=[0-9.]*' "$log" | head -1 | cut -d= -f2)"
  stride_act="$(grep -o 'stride=[0-9]*' "$log" | tail -1 | cut -d= -f2)"
  kmax="$(grep -o 'k_per_thread_max=[0-9]*' "$log" | head -1 | cut -d= -f2)"
  match=0; grep -q '\[gpu-run-correctness\] MATCH' "$log" && match=1
  local dR="NA" dRef="NA"
  [[ -n "${KMS[R484]:-}" && -n "${kms:-}" ]] && dR="$(awk -v a="${KMS[R484]}" -v k="$kms" 'BEGIN{printf "%+.3f",(k-a)/a*100}')"
  [[ -n "${kms:-}" ]] && dRef="$(awk -v a="$REF_RAW_MS" -v k="$kms" 'BEGIN{printf "%+.3f",(k-a)/a*100}')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$label" "$mb" "${stride_act:-?}" "${kmax:-?}" "${total:-?}" "$match" "${kms:-?}" "$dR" "$dRef" "$clk" "$tmp" "$start" >> "$TSV"
  if [[ "$match" -eq 1 && "${total:-}" == "$ORACLE" ]]; then pass "oracle_match[$cell]"; else fail "oracle_match[$cell]" "total_sum='${total:-<none>}' match=$match"; return 1; fi
  [[ "${stride_act:-}" == "$((32*mb))" ]] && pass "stride_as_intended[$cell]=$stride_act" || { fail "stride_as_intended[$cell]" "got ${stride_act:-?}"; return 1; }
  KMS[$cell]="$kms"; info "cell[$cell]" "kernel_ms=$kms  vs R484 ${dR}%  vs 389 ref ${dRef}%"
  return 0
}
first=1
for cell in $CELLS; do
  (( first )) || { echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; }; first=0
  run_cell "$cell" || break
done

# ---------------------------------------------------------------------
# 3. Optional ncu (raw vs base, chunk0 slice, 484)
# ---------------------------------------------------------------------
if [[ "$NCU_STAGE" == "1" ]]; then
  banner "ncu SchedulerStats+WarpStateStats, chunk0 slice (743,424 records = 48 iterations), raw vs base @484"
  MARKER="$LOGDIR/.owner_marker"; touch "$MARKER"
  for tag in raw base; do
    src="$IN_RAW"; [[ "$tag" == "base" ]] && src="$IN_BASE"
    head -c $((743424*28)) "$src" > "/tmp/${REV}_chunk0_${tag}.bin"
    REP="$LOGDIR/${REV}_ncu_chunk0_${tag}"
    sudo env NQ_MAX_BLOCKS=484 "$NCU_BIN" --launch-count 1 --section SchedulerStats --section WarpStateStats -f -o "$REP" \
        "./$CU_BIN" "$NQ" "/tmp/${REV}_chunk0_${tag}.bin" "/tmp/${REV}_ncu_${tag}_out.bin" 2>&1 | tee "$LOGDIR/3_ncu_${tag}.log"
    find . -maxdepth 1 -newer "$MARKER" -user root -print0 2>/dev/null | xargs -0 -r sudo chown "$(id -u):$(id -g)" 2>/dev/null || true
    sudo chown -R "$(id -u):$(id -g)" "$LOGDIR" 2>/dev/null || true
    "$NCU_BIN" --import "${REP}.ncu-rep" --page details --print-details all > "${REP}_details.txt" 2>&1 || true
    grep -h -E 'Active Warps Per Scheduler|Eligible Warps Per Scheduler|No Eligible|Avg. Active Threads Per Warp|Warp Cycles Per Issued|Stall Wait|Stall Branch Resolving|Stall Long Scoreboard' "${REP}_details.txt" 2>/dev/null | sed "s/^/    [$tag] /"
  done
  rm -f "$MARKER"
fi

# ---------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------
banner "Evaluation"
absdev() { awk -v a="$1" -v r="$2" 'BEGIN{d=(a-r)/r*100; printf "%.3f",(d<0?-d:d)}'; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%+.2f",(b-a)/a*100}'; }
if [[ -n "${KMS[R484]:-}" ]]; then d="$(absdev "${KMS[R484]}" "$REF_RAW_MS")"; awk -v d="$d" 'BEGIN{exit !(d<=1.0)}' && pass "R484_reproduces_389 (${KMS[R484]}, ${d}%)" || fail "R484_reproduces_389" "${d}% off"; fi
if [[ -n "${KMS[R484]:-}" && -n "${KMS[R485]:-}" ]]; then d="$(absdev "${KMS[R485]}" "${KMS[R484]}")"; awk -v d="$d" 'BEGIN{exit !(d<=1.0)}' && pass "raw_is_stride_insensitive (R485 vs R484 ${d}%)" || info "raw_is_stride_insensitive" "R485 differs from R484 by ${d}% -- interpret B485 relative to R485"; fi
if [[ -n "${KMS[B484]:-}" ]]; then d="$(absdev "${KMS[B484]}" "$REF_BASE_MS")"; awk -v d="$d" 'BEGIN{exit !(d<=1.0)}' && pass "B484_reproduces_394d_E1 (${KMS[B484]}, ${d}%)" || fail "B484_reproduces_394d_E1" "${d}% off"; fi
if [[ -n "${KMS[R484]:-}" ]]; then
  for c in B484 B485 B483 S485; do [[ -n "${KMS[$c]:-}" ]] && info "delta_vs_R484[$c]" "$(pct "${KMS[R484]}" "${KMS[$c]}")%"; done
  if [[ -n "${KMS[B485]:-}" && -n "${KMS[B483]:-}" ]]; then
    d5="$(pct "${KMS[R484]}" "${KMS[B485]}")"; d3="$(pct "${KMS[R484]}" "${KMS[B483]}")"
    if awk -v a="$d5" -v b="$d3" 'BEGIN{exit !(a<=10 && b<=10)}'; then info "verdict" "M2 (inter-thread tail): the same warp groups at a shifted stride recover to within +10% of raw (B485 ${d5}%, B483 ${d3}%)"
    elif awk -v a="$d5" -v b="$d3" 'BEGIN{exit !(a>=40 && b>=40)}'; then info "verdict" "M1 (intra-warp): the penalty survives the stride shift (B485 ${d5}%, B483 ${d3}%)"
    else info "verdict" "MIXED (B485 ${d5}%, B483 ${d3}%): run NCU=1 to split it"; fi
    if awk -v a="$d5" 'BEGIN{exit !(a<0)}'; then info "note" "B485 is BELOW raw: warp homogeneity has value once the tail is removed -- a new lever"; fi
  fi
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
