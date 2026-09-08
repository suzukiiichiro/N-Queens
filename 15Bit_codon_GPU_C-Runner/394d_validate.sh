#!/usr/bin/env bash
# 394d_validate.sh
#
# rev394d -- decompose 394c's cell B rung by rung.
#
# WHAT 394c FOUND (2026-09-08, all four cells oracle-MATCH)
# ---------------------------------------------------------
#                    raw (mode 37)      reordered (mode 39)
#   MAX_BLOCKS=484   A 201,236 ms       B 261,060 ms   (+29.7%)  <- ??
#   MAX_BLOCKS=968   C 160,528 ms       D 158,955 ms   ( -1.0%)
#
# Pre-registered for B vs A was -3..-6%. What came back was the wrong
# sign at 5x the magnitude: the single largest effect seen in this
# campaign, from DATA ORDER alone, kernel untouched. And D's shaping ran
# into "[chunkshape148-warning] group too large for stable packing ...
# iter_sort disabled" at stride 30,976, so stride and iter_sort are
# CONFOUNDED between B and D. This harness un-confounds them at the
# production stride (15,488), one pipeline stage per rung.
#
# THE LADDER (all at MAX_BLOCKS=484, stride 15,488)
# ------------------------------------------------
#   A   mode 37                                   raw stream order    (anchor)
#   E1  mode 39  input_stage=1                    broadmarktail base only
#                                                 (funcid w3_j7 reorder + rotate; no chunkshape)
#   E2  mode 39  bucket_run=1     iter_sort=0     chunkshape 276-338 order (scorestripe only)
#   E3  mode 39  bucket_run=2048  iter_sort=0     344 order
#   E4  mode 39  bucket_run=2048  iter_sort=1     346 order
#   E5  mode 39  bucket_run=2048  iter_sort=9     350 order = 394c cell B (in-session repeat)
# Each rung adds exactly one pipeline component. Whichever rung first
# jumps by >= +20% names the culprit.
#
# PRE-REGISTERED (394d_README_append.md):
#   A within 1% of 201,237;  E5 within 1% of 261,060
#   H_isort  (stated prediction): E1, E2, E3 all within +-3% of A;
#            the jump appears at E4 and/or E5.  Basis: D at 968 had
#            iter_sort force-disabled and came in at -1.0%.
#   falsified if E3 is already >= +20%  (culprit = scorestripe/bucket_run)
#   or E1 already >= +20%               (culprit = broadmarktail)
#
# COST: 6 full N=21 runs (~3.4..4.4 min each) + 3 chunkshape builds
# (~1 min each incl. the 361 dump + 363 filter) ~= 30 min. No sudo.
#
# OPTIONAL (NCU=1, sudo): SchedulerStats+WarpStateStats on the first
# 743,424 records (= chunk0 = 48 grid-stride iterations, the shaping's
# own period) for raw vs E5. Avg Active Threads/Warp and Active Warps/
# Scheduler then say whether the +30% is intra-warp (lane utilisation
# drops) or inter-warp (residency drops = a launch tail). ~9 min each.
#
# USAGE
#   STATIC_ONLY=1 bash 394d_validate.sh
#                 bash 394d_validate.sh                 # A E1 E2 E3 E4 E5
#   STAGES=A,E3,E5 bash 394d_validate.sh                # subset
#   NCU=1         bash 394d_validate.sh                 # + the two ncu slices

set -u

REV="394d"
PY_SRC="${PY_SRC:-394dPy_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-394dPy_kernel_maxd14_final}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
FILTER_SCRIPT="${FILTER_SCRIPT:-363_filter_maxd14_only.py}"
CU_SRC="${CU_SRC:-394d_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-394d_kernel_maxd14}"
REF_CU="${REF_CU:-394c_kernel_maxd14.cu}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
ORACLE="${ORACLE:-314666222712}"
MB="${MB:-484}"
STAGES="${STAGES:-A,E1,E2,E3,E4,E5}"
NCU_STAGE="${NCU:-0}"
REF_A_MS="${REF_A_MS:-201237}"
REF_B_MS="${REF_B_MS:-261060}"
COOLDOWN="${COOLDOWN:-20}"
STATIC_ONLY="${STATIC_ONLY:-0}"
CRLOG_DIR="${REV}_crunner_logs"

NCU_BIN="${NCU_BIN:-$(command -v ncu 2>/dev/null)}"
[[ -z "$NCU_BIN" && -x /usr/local/cuda/bin/ncu ]] && NCU_BIN="/usr/local/cuda/bin/ncu"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_ladder_${TS}}"
TSV="$LOGDIR/${REV}_ladder.tsv"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }
want() { [[ ",$STAGES," == *",$1,"* ]]; }

if [[ "$NCU_STAGE" == "1" ]]; then
  sudo -n true 2>/dev/null && pass "sudo_noninteractive_available" || { fail "sudo_noninteractive_available" "NCU=1 needs sudo"; exit 1; }
fi

# ---------------------------------------------------------------------
# 1. Static checks
# ---------------------------------------------------------------------
for f in "$PY_SRC" "$HELPER_SRC" "$FILTER_SCRIPT" "$CU_SRC"; do
  [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "not found in $(pwd)"
done
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }
python3 - "$PY_SRC" > "/tmp/${REV}_code_only.py" <<'EOF'
import re,sys
s=open(sys.argv[1],encoding='utf-8').read()
m=re.search(r'"""',s); e=s.index('"""',m.end()); print(s[:m.start()]+s[e+3:])
EOF
CODE="/tmp/${REV}_code_only.py"
grep -qE '^REV_TAG:str="394d"' "$CODE" && pass "source_rev_tag_is_394d" || fail "source_rev_tag_is_394d" "REV_TAG is not 394d"
grep -qE 'if not \(bench_mode==0 .*bench_mode==39\):' "$CODE" && pass "source_bench_mode_39_in_cli_whitelist" || fail "source_bench_mode_39_in_cli_whitelist" "39 missing from CLI whitelist"
grep -qE '^    if bench_mode==11 .*bench_mode==39:' "$CODE" && pass "source_bench_mode_39_in_preset_gate" || fail "source_bench_mode_39_in_preset_gate" "39 missing from preset gate"
grep -q '    if bench_mode==39:' "$CODE" && grep -q 'crunner_input_stage=int(argv\[15\])' "$CODE" && pass "source_mode_39_ladder_knobs_parsed" || fail "source_mode_39_ladder_knobs_parsed" "argv[10..15] parser for mode 39 missing"
grep -q 'chunkshape148_bucket_run=int(argv\[13\])' "$CODE" && grep -q 'chunkshape148_iter_sort=int(argv\[14\])' "$CODE" && pass "source_mode_39_bucket_run_iter_sort_slots" || fail "source_mode_39_bucket_run_iter_sort_slots" "argv[13]/[14] slots missing"
grep -q 'if crunner_input_stage>=2:' "$CODE" && pass "source_mode_39_input_stage_honoured" || fail "source_mode_39_input_stage_honoured" "input_stage branch missing in dispatch"
[[ "$(grep -c 'ensure_crunner_input_bin(N,stream_fname,gpu_log_level)' "$CODE")" == "1" ]] && pass "source_mode_37_still_raw_input" || fail "source_mode_37_still_raw_input" "mode 37 changed"
grep -q 'NQ_MAX_BLOCKS={gpu_max_blocks}' "$CODE" && pass "source_mode_39_stride_coupling" || fail "source_mode_39_stride_coupling" "missing"
grep -q 'CRunnerEntry(14,"./394d_kernel_maxd14",""' "$CODE" && pass "source_dispatch_table_references_394d_binary" || fail "source_dispatch_table_references_394d_binary" "table wrong"
if [[ -f "$REF_CU" ]]; then
  A=$(awk 'f||/^#include/{f=1;print}' "$REF_CU" | sha256sum | cut -d' ' -f1); B=$(awk 'f||/^#include/{f=1;print}' "$CU_SRC" | sha256sum | cut -d' ' -f1)
  [[ "$A" == "$B" ]] && pass "cu_code_region_identical_to_394c (${B:0:16}...)" || fail "cu_code_region_identical_to_394c" "differs"
else info "cu_code_region_identical_to_394c" "skipped ($REF_CU absent)"; fi
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY summary ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; nvidia-smi -q -d CLOCK 2>&1
  sha256sum "$PY_SRC" "$HELPER_SRC" "$FILTER_SCRIPT" "$CU_SRC" 2>&1; echo "STAGES=$STAGES MB=$MB NCU=$NCU_STAGE"; } > "$LOGDIR/00_env_pre.txt" 2>&1

banner "Building $CU_SRC and $PY_SRC"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -o "$CU_BIN" "$CU_SRC" 2>&1 | tee "$LOGDIR/01_nvcc_build.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build_succeeded" || { fail "nvcc_build_succeeded" "no binary"; exit 1; }
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/02_codon_build.log"
[[ -x "$PY_BIN" ]] && pass "codon_build_succeeded" || { fail "codon_build_succeeded" "see $LOGDIR/02_codon_build.log"; exit 1; }

# ---------------------------------------------------------------------
# 2. The ladder
# ---------------------------------------------------------------------
printf 'cell\tmode\tinput_stage\tbucket_run\titer_sort\tmax_blocks\tstride_actual\ttotal_sum\tmatch\tkernel_ms\tdelta_vs_A_pct\tsm_clock_mhz\ttemp_c\tstart\n' > "$TSV"
declare -A KMS
run_cell() {
  local cell="$1" mode="$2" stage="$3" brun="$4" isort="$5"
  banner "cell $cell: mode=$mode input_stage=$stage bucket_run=$brun iter_sort=$isort MAX_BLOCKS=$MB"
  local clk tmp start log crlog total kms stride_act match
  clk="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  tmp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  start="$(date -Is)"; log="$LOGDIR/1${cell}_mode${mode}.log"
  rm -f "$CRLOG_DIR/crunner_${CU_BIN}_N${NQ}.log"
  if [[ "$mode" == "37" ]]; then
    env -u NQ_MAX_BLOCKS "./$PY_BIN" -g "$NQ" "$NQ" 32 "$MB" 0 0 7 37 -d 2>&1 | tee "$log"
  else
    # argv[10..15] = window_mult phase_jump cross_stripe_safe bucket_run iter_sort input_stage
    env -u NQ_MAX_BLOCKS "./$PY_BIN" -g "$NQ" "$NQ" 32 "$MB" 0 0 7 39 3 7 0 "$brun" "$isort" "$stage" -d 2>&1 | tee "$log"
  fi
  crlog="$CRLOG_DIR/crunner_${CU_BIN}_N${NQ}.log"
  cp "$crlog" "$LOGDIR/1${cell}_crunner.log" 2>/dev/null || true
  cp "$CRLOG_DIR/dispatch.log" "$LOGDIR/dispatch_after_${cell}.log" 2>/dev/null || true
  total="$(grep -o 'total_sum=[0-9]*' "$crlog" 2>/dev/null | head -1 | cut -d= -f2)"
  kms="$(grep -o 'kernel_ms=[0-9.]*' "$crlog" 2>/dev/null | head -1 | cut -d= -f2)"
  stride_act="$(grep -o 'stride=[0-9]*' "$crlog" 2>/dev/null | tail -1 | cut -d= -f2)"
  match=0; grep -q '\[gpu-run-correctness\] MATCH' "$crlog" 2>/dev/null && match=1
  local dA="NA"; [[ -n "${KMS[A]:-}" && -n "${kms:-}" ]] && dA="$(awk -v a="${KMS[A]}" -v k="$kms" 'BEGIN{printf "%+.3f",(k-a)/a*100}')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$mode" "$stage" "$brun" "$isort" "$MB" "${stride_act:-?}" "${total:-?}" "$match" "${kms:-?}" "$dA" "$clk" "$tmp" "$start" >> "$TSV"
  if [[ "$match" -eq 1 && "${total:-}" == "$ORACLE" ]]; then pass "oracle_match[$cell]"; else fail "oracle_match[$cell]" "total_sum='${total:-<none>}' match=$match"; return 1; fi
  grep '^21:' "$log" | grep -q ' ok$' && pass "dispatcher_row_ok[$cell]" || { fail "dispatcher_row_ok[$cell]" "row not ok -- see $log"; return 1; }
  [[ "${stride_act:-}" == "$((32*MB))" ]] && pass "stride_as_intended[$cell]=$stride_act" || { fail "stride_as_intended[$cell]" "got ${stride_act:-?}"; return 1; }
  if [[ "$mode" == "39" ]]; then
    # the params line the .py prints for mode 39 must echo the knobs we passed
    if grep -q "crunner_reordered_params: input_stage=$stage .*bucket_run=$brun .*iter_sort=$isort " "$log"; then pass "knobs_echoed[$cell]"; else fail "knobs_echoed[$cell]" "the .py did not report input_stage=$stage bucket_run=$brun iter_sort=$isort -- a knob was not parsed"; return 1; fi
    if [[ "$stage" == "1" ]]; then grep -q 'crunner-reordered-base-only' "$log" && pass "base_only_path_taken[$cell]" || fail "base_only_path_taken[$cell]" "chunkshape stage was not skipped"; fi
    if grep -q 'chunkshape148-warning' "$log"; then info "chunkshape_warning[$cell]" "$(grep 'chunkshape148-warning' "$log" | head -1 | cut -c1-140)"; fi
  fi
  KMS[$cell]="$kms"; info "cell[$cell]" "kernel_ms=$kms  vs A ${dA}%"
  return 0
}
first=1
for cell in A E1 E2 E3 E4 E5; do
  want "$cell" || continue
  (( first )) || { echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; }; first=0
  case "$cell" in
    A)  run_cell A  37 2 2048 9 || break ;;
    E1) run_cell E1 39 1 2048 9 || break ;;    # base only (bucket_run/iter_sort unused)
    E2) run_cell E2 39 2 1    0 || break ;;
    E3) run_cell E3 39 2 2048 0 || break ;;
    E4) run_cell E4 39 2 2048 1 || break ;;
    E5) run_cell E5 39 2 2048 9 || break ;;
  esac
done

# ---------------------------------------------------------------------
# 3. Optional ncu: raw vs E5 on the chunk0 slice (48 iterations)
# ---------------------------------------------------------------------
if [[ "$NCU_STAGE" == "1" ]]; then
  banner "ncu SchedulerStats+WarpStateStats on the chunk0 slice (743,424 records), raw vs E5"
  RAW_IN="constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin"
  E5_IN="$(ls constellations_N21_6_chunkshape148_*_b32_m${MB}_s$((32*MB))_run2048_isort9.bin.soa_ref_361.bin.maxd14only_363.bin 2>/dev/null | head -1)"
  MARKER="$LOGDIR/.owner_marker"; touch "$MARKER"
  for tag in raw e5; do
    src="$RAW_IN"; [[ "$tag" == "e5" ]] && src="$E5_IN"
    [[ -f "${src:-}" ]] || { info "ncu[$tag]" "input not found, skipped"; continue; }
    head -c $((743424*28)) "$src" > "/tmp/${REV}_chunk0_${tag}.bin"
    REP="$LOGDIR/${REV}_ncu_chunk0_${tag}"
    sudo env NQ_MAX_BLOCKS="$MB" "$NCU_BIN" --launch-count 1 --section SchedulerStats --section WarpStateStats -f -o "$REP" \
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
if [[ -n "${KMS[A]:-}" ]]; then d="$(absdev "${KMS[A]}" "$REF_A_MS")"; awk -v d="$d" 'BEGIN{exit !(d<=1.0)}' && pass "A_reproduces_389_anchor (${KMS[A]}, ${d}%)" || fail "A_reproduces_389_anchor" "${d}% off"; fi
if [[ -n "${KMS[E5]:-}" ]]; then d="$(absdev "${KMS[E5]}" "$REF_B_MS")"; awk -v d="$d" 'BEGIN{exit !(d<=1.0)}' && pass "E5_reproduces_394c_B (${KMS[E5]}, ${d}%)" || fail "E5_reproduces_394c_B" "${d}% off"; fi
if [[ -n "${KMS[A]:-}" ]]; then
  echo; echo "  rung                              kernel_ms      vs A"
  for c in E1 E2 E3 E4 E5; do
    [[ -n "${KMS[$c]:-}" ]] || continue
    d="$(awk -v a="${KMS[A]}" -v k="${KMS[$c]}" 'BEGIN{printf "%+7.2f",(k-a)/a*100}')"
    case $c in E1) l="E1 base only (w3_j7 + rotate)";; E2) l="E2 + scorestripe (run=1, isort=0)";; E3) l="E3 + bucket_run=2048 (isort=0)";; E4) l="E4 + iter_sort=1";; E5) l="E5 + iter_sort=9 (= 394c B)";; esac
    printf '  %-32s %12s %8s%%\n' "$l" "${KMS[$c]}" "$d"
  done
  # first rung with >= +20% names the culprit
  for c in E1 E2 E3 E4 E5; do
    [[ -n "${KMS[$c]:-}" ]] || continue
    if awk -v a="${KMS[A]}" -v k="${KMS[$c]}" 'BEGIN{exit !((k-a)/a*100>=20)}'; then info "first_rung_with_jump_ge_20pct" "$c"; break; fi
  done
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
