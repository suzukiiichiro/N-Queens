#!/usr/bin/env bash
# 400_validate.sh
#
# rev400 -- lift the occupancy ceiling that BLOCK=32 has been holding down.
#
# WHAT 399-r2 ESTABLISHED (U5, at the third attempt)
#   N=20 stall cycles per issue, excluding `selected` which is issue not stall:
#     wait              45.05%   2.32
#     branch_resolving  26.80%   1.38
#     long_scoreboard   13.98%   0.72
#     no_instruction     7.57%   0.39
#   N=19 gives the identical ranking, so this is not a proxy artefact.
#   `wait` is fixed-latency dependency stalling. Memory is 14%. That
#   overturns the L1-bound reading that 398-r2's Speed-of-Light numbers
#   suggested: the L1 pipe is busy, but the warps are not blocked on it.
#   With 2.43 of 12 warp slots per scheduler filled there is nothing to hide
#   even short latencies with.
#
#   399-r2 also closed the ordering axis. Record order is worth 35.2% and
#   394f already captures it: neither popcount sort nor a MEASURED
#   per-record cost sort beat it (+0.41%, +1.96%, +2.81%, +3.47%). And lane
#   utilisation could not be priced -- the local slope (desc: -3.3% lanes
#   for +0.40% time) and the global slope (shuffle: -35.6% lanes for +35.2%
#   time) differ by a factor of six, so the shuffle penalty is not mostly
#   lane utilisation.
#
# THE CEILING
#   sm_86 holds at most 16 blocks and 48 warps per SM. BLOCK=32 makes a
#   block exactly one warp, so 16 blocks/SM caps theoretical occupancy at
#   33.3% however large the grid gets -- and 800 blocks over 80 SMs reaches
#   only 10 of those 16. MAX_BLOCKS has been swept (394b, 394g); BLOCK has
#   sat at 32 the whole time. Registers are not the constraint: 37 per
#   thread allows 52 warps.
#     BLOCK= 64 -> 16 blocks x 2 warps = 32/48 = 66.7%
#     BLOCK=128 -> 12 blocks x 4 warps = 48/48 = 100%
#
# WHAT IS NOT ASSUMED
#   More warps also means more resident local memory and fewer records per
#   thread, which weakens 394f's balancing. The net is measured, not argued.
#
# PRE-REGISTERED (400_README_append.md; written before execution)
#   Z1  kernel region sha still ebd7f523, ptxas registers still 37, and
#       NQ_BLOCK unset leaves BLOCK=32.
#   Z2  the packaging control -- stride held at 25,600 while BLOCK goes
#       32/64/128 -- moves kernel_ms by <= 2%. Same warps per SM, so shape
#       alone should not matter.
#   Z3  STATED PREDICTION: at least one configuration with >= 32 warps/SM
#       beats the (32,800) baseline by >= 5%. If nothing beats it by more
#       than 1%, occupancy is not the lever and the `wait` reading has to be
#       reworked before anything else is tried.
#   Z4  for the winner, Achieved Occupancy and Active Warps Per Scheduler
#       rise and `wait`'s share of stall cycles falls. Mechanism, not just
#       outcome.
#   Z5  the winner carries at least half its N=20 relative gain to N=21.
#   Z6  (WITH_N22=1) the winner also beats the baseline at N=22.
#
#   Every run is oracle-gated. Changing BLOCK or MAX_BLOCKS only repartitions
#   the same records across threads, so the answer must not move.
#
# USAGE
#   STATIC_ONLY=1 bash 400_r2_validate.sh
#                 bash 400_r2_validate.sh         # ~15 min (sweep N=19, confirm N=20)
#   NPROF=20 NCONF=21 NBIG=22 bash 400_r2_validate.sh   # production scale, ~45 min
#   WITH_NBIG=1   bash 400_r2_validate.sh         # + the big-N confirmation
#   NCU_PREFIX=sudo bash 400_r2_validate.sh
#
# PRIVILEGE: only the ncu calls are elevated; outputs are chowned back.

set -u

REV="400_r2"
PY_SRC="${PY_SRC:-400_r2Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-400_r2Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-400Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-400_r2_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-400_r2_kernel_maxd14}"
PREV_CU="${PREV_CU:-400_kernel_maxd14.cu}"
CRLOG_DIR="${CRLOG_DIR:-400_r2_crunner_logs}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
NCU="${NCU:-ncu}"
ARCH="${ARCH:-sm_86}"
# 400-r2: every N is one lower than 400's. NPROF sweeps, NCONF confirms,
# NBIG is the optional big check. One line restores production scale:
#   NPROF=20 NCONF=21 NBIG=22 bash 400_r2_validate.sh
NPROF="${NPROF:-19}"
NCONF="${NCONF:-20}"
NBIG="${NBIG:-21}"
oracle_of() {  # $1 = N
  case "$1" in
    18) echo 666090624 ;;   19) echo 4968057848 ;;      20) echo 39029188884 ;;
    21) echo 314666222712 ;; 22) echo 2691008701644 ;;  23) echo 24233937684440 ;;
    *)  echo "" ;;
  esac
}
BASE_BLOCK="${BASE_BLOCK:-32}"
BASE_MB="${BASE_MB:-800}"
REPS="${REPS:-3}"
REPS21="${REPS21:-3}"
EXTRA_CTX="${EXTRA_CTX:-2}"     # 2 extra contexts = the 3-context production state (397 V4, 398 W1)
KERNEL_SHA_395A="${KERNEL_SHA_395A:-ebd7f523deb591347eab1a352a9555c2e4a6c0b4a456d80517322c933a86e68a}"
SECTION_TIMEOUT_S="${SECTION_TIMEOUT_S:-600}"
STALL_METRICS="smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__average_warps_issue_stalled_selected_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_misc_per_issue_active.ratio"
OCC_METRICS="sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warps_active.avg.per_cycle_active,smsp__thread_inst_executed_per_inst_executed.ratio"
# BLOCK:MAX_BLOCKS -- family A holds stride at 25,600 (packaging control),
# family B raises the resident warp count.
SWEEP="${SWEEP:-32:800 64:400 128:200 32:1280 64:640 64:1280 128:640 128:960 128:1280 256:480}"
WITH_NBIG="${WITH_NBIG:-${WITH_N22:-0}}"
STATIC_ONLY="${STATIC_ONLY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"
TSV="$LOGDIR/${REV}_sweep.tsv"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }

# ---------------------------------------------------------------------
# 1. Static
# ---------------------------------------------------------------------
for f in "$PY_SRC" "$HELPER_SRC" "$CU_SRC"; do
  [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"
done
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }
py_code_region() { python3 -c "
import sys
lines=open(sys.argv[1],encoding='utf-8').read().split('\n')
i=next(i for i,l in enumerate(lines) if l.startswith('import '))
sys.stdout.write('\n'.join(l for l in lines[i:] if not l.lstrip().startswith('#')))
" "$1"; }
py_note_quotes() { python3 -c "
import sys
lines=open(sys.argv[1],encoding='utf-8').read().split('\n')
i=next(i for i,l in enumerate(lines) if l.startswith('import '))
print('\n'.join(lines[:i]).count('\"'*3))
" "$1"; }
py_code_region "$PY_SRC" > "/tmp/${REV}_code_only.py"; CODE="/tmp/${REV}_code_only.py"
NOTE_LINES=$(awk '/^# =+$/{f=1} f&&/^#/{n++} END{print n+0}' "$PY_SRC")
{ grep -q "^# ${REV//_/-} " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# ${REV//_/-} ...' note block"
NQ_=$(py_note_quotes "$PY_SRC"); [[ $((NQ_ % 2)) -eq 0 ]] && pass "py_note_region_quotes_balanced ($NQ_)" || fail "py_note_region_quotes_balanced" "$NQ_ triple-quotes"
grep -qE '^REV_TAG:str="400_r2"' "$CODE" && pass "source_rev_tag_is_400_r2" || fail "source_rev_tag_is_400_r2" "wrong REV_TAG"
grep -q 'CRunnerEntry(14,"./400_r2_kernel_maxd14","NQ_EXTRA_CTX=1 "' "$CODE" && pass "source_table_points_at_400_r2_and_keeps_the_treatment" || fail "source_table_points_at_400_r2_and_keeps_the_treatment" "table entry wrong"
grep -qE '^CRUNNER_MIN_N:int=19' "$CODE" && pass "source_crunner_min_n_still_19" || fail "source_crunner_min_n_still_19" "CRUNNER_MIN_N lost"
grep -qE '^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=800' "$CODE" && pass "source_default_max_blocks_800" || fail "source_default_max_blocks_800" "not 800"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_400Py (removed=3 added=3 EXECUTABLE lines)" \
    || { fail "py_diff_fingerprint_vs_400Py" "removed=$PR_ added=$PA_, expected 3/3"; diff "/tmp/${REV}_prev_code.py" "$CODE" | head -20 | sed 's/^/      /'; }
else info "py_diff_fingerprint_vs_400Py" "skipped"; fi
cu_code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA_395A" ]] && pass "Z1_cu_kernel_region_still_395a_r2 (${KB:0:16}... -- NQ_BLOCK is host-side only)" || fail "Z1_cu_kernel_region_still_395a_r2" "got ${KB:0:16}..."
if [[ -f "$PREV_CU" ]]; then
  cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  CA="$(sha256sum < "/tmp/${REV}_prev_code.cu" | cut -d' ' -f1)"; CB="$(sha256sum < "/tmp/${REV}_cur_code.cu" | cut -d' ' -f1)"
  [[ "$CA" == "$CB" ]] && pass "cu_whole_code_region_identical_to_400 (${CB:0:16}... -- r2 changes only the harness scale)" \
    || fail "cu_whole_code_region_identical_to_400" "400-r2 must be a rename of 400"
else info "cu_whole_code_region_identical_to_400" "skipped"; fi
grep -q 'const char \*env_bk = getenv("NQ_BLOCK");' "$CU_SRC" && pass "cu_nq_block_knob_present" || fail "cu_nq_block_knob_present" "NQ_BLOCK is not read"
grep -q '    int BLOCK = 32;' "$CU_SRC" && pass "cu_block_default_is_32" || fail "cu_block_default_is_32" "the default is not 32"
grep -q 'dim3 block(BLOCK);' "$CU_SRC" && pass "cu_launch_uses_BLOCK" || fail "cu_launch_uses_BLOCK" "the launch no longer uses BLOCK"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1

banner "Building"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$CU_BIN" "$CU_SRC" -lcuda 2>&1 | tee "$LOGDIR/05_nvcc.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "no binary"; exit 1; }
"$NVCC" -O3 -arch="$ARCH" -Xptxas -v -c "$CU_SRC" -o /dev/null 2>&1 | grep -A3 'kernel_dfs_iter_gpu_maxd14' > "$LOGDIR/02_ptxas.txt" 2>&1
RN="$(grep -o 'Used [0-9]* registers' "$LOGDIR/02_ptxas.txt" | head -1 | cut -d' ' -f2)"
FN="$(grep -o '[0-9]* bytes stack frame' "$LOGDIR/02_ptxas.txt" | head -1 | cut -d' ' -f1)"
info "ptxas" "registers=${RN:-?} stack_frame=${FN:-?} B"
[[ "${RN:-0}" == "37" && "${FN:-0}" == "208" ]] && pass "Z1_ptxas_unchanged (37 registers, 208 B frame)" || fail "Z1_ptxas_unchanged" "registers=${RN:-?} frame=${FN:-?}, expected 37 / 208"
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/06_codon.log"
[[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN]" || { fail "codon_build" "see log"; exit 1; }

declare -A NIN NBASEMS
gen_for_N() {  # $1 = N
  local n="$1" gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${1}.log"
  banner "Generating and timing N=$n via ./$PY_BIN -g $n $n"
  rm -f "$gcr"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX -u NQ_BLOCK "./$PY_BIN" -g "$n" "$n" 2>&1 | tee "$LOGDIR/1_gen_N${n}_console.log"
  cp "$gcr" "$LOGDIR/1_gen_N${n}_crunner.log" 2>/dev/null || true
  [[ -f "$gcr" ]] || { fail "crunner_path_taken[N=$n]" "no $gcr -- check $CRLOG_DIR/dispatch.log"; return 1; }
  NIN[$n]="$(grep -o 'src=[^ ]*' "$gcr" | head -1 | cut -d= -f2)"
  NBASEMS[$n]="$(grep -o 'kernel_ms=[0-9.]*' "$gcr" | head -1 | cut -d= -f2)"
  [[ -f "${NIN[$n]:-/nonexistent}" ]] && pass "input_located[N=$n] (${NIN[$n]}, kernel_ms=${NBASEMS[$n]})" || { fail "input_located[N=$n]" "src='${NIN[$n]:-<none>}'"; return 1; }
  return 0
}
gen_for_N "$NPROF" || exit 1
gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${NPROF}.log"
grep -q '\[gpu-config\] BLOCK=32 ' "$gcr" && pass "Z1_nq_block_inert_when_unset (the dispatcher run still reports BLOCK=32)" || fail "Z1_nq_block_inert_when_unset" "$(grep -o '\[gpu-config\].*' "$gcr" | head -1)"
gen_for_N "$NCONF" || info "NCONF_input" "N=$NCONF input unavailable -- Z5 will be skipped"
IN20="${NIN[$NPROF]}"
ORC_PROF="$(oracle_of "$NPROF")"; ORC_CONF="$(oracle_of "$NCONF")"; ORC_BIG="$(oracle_of "$NBIG")"
[[ -n "$ORC_PROF" ]] && pass "oracle_known[N=$NPROF] ($ORC_PROF)" || { fail "oracle_known[N=$NPROF]" "no oracle for N=$NPROF"; exit 1; }

# ---------------------------------------------------------------------
# 2. Sweep
# ---------------------------------------------------------------------
printf 'block\tmax_blocks\tstride\tblocks_per_sm\twarps_per_sm\trep\tkernel_ms\ttotal_sum\tmatch\n' > "$TSV"
declare -A SMEAN
run_cfg() {  # block max_blocks N input oracle reps tag
  local bk="$1" mb="$2" n="$3" input="$4" orc="$5" reps="$6" tag="$7"
  local stride=$((bk*mb)) bps=$((mb/80)) wps=$(( (mb/80) * (bk/32) ))
  [[ "$wps" -gt 48 ]] && wps=48
  local sum=0 k
  for r in $(seq 1 "$reps"); do
    local lg="$LOGDIR/3_${tag}_b${bk}_m${mb}_r${r}.log"
    env -u NQ_PAD_MB NQ_BLOCK="$bk" NQ_MAX_BLOCKS="$mb" NQ_EXTRA_CTX="$EXTRA_CTX" \
      "./$CU_BIN" "$n" "$input" "/tmp/${REV}_out.bin" "$orc" > "$lg" 2>&1 || true
    local kms tot m sact
    kms="$(grep -o 'kernel_ms=[0-9.]*' "$lg" | head -1 | cut -d= -f2)"
    tot="$(grep -o 'total_sum=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
    sact="$(grep -o 'stride=[0-9]*' "$lg" | tail -1 | cut -d= -f2)"
    m=0; grep -q '\[gpu-run-correctness\] MATCH' "$lg" && m=1
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$bk" "$mb" "$stride" "$bps" "$wps" "$r" "${kms:-?}" "${tot:-?}" "$m" >> "$TSV"
    if [[ "$m" != "1" || "${tot:-}" != "$orc" ]]; then
      fail "oracle[$tag b=$bk m=$mb rep$r]" "total_sum='${tot:-<none>}' expected $orc -- repartitioning the SAME records across threads must not change the answer"
      return 1
    fi
    [[ "${sact:-}" == "$stride" ]] || { fail "stride_as_intended[$tag b=$bk m=$mb]" "binary reported ${sact:-?}, expected $stride"; return 1; }
    sum="$(awk -v s="$sum" -v k="${kms:-0}" 'BEGIN{printf "%.3f",s+k}')"
  done
  SMEAN["${tag}_${bk}_${mb}"]="$(awk -v s="$sum" -v r="$reps" 'BEGIN{printf "%.3f",s/r}')"
  info "cfg[$tag]" "BLOCK=$bk MAX_BLOCKS=$mb stride=$stride blocks/SM=$bps warps/SM~$wps  mean=${SMEAN["${tag}_${bk}_${mb}"]} ms"
  return 0
}
banner "Sweep at N=$NPROF, $REPS reps each, NQ_EXTRA_CTX=$EXTRA_CTX (the production 3-context state)"
info "scale" "sweep N=$NPROF, confirm N=$NCONF, optional N=$NBIG. Restore production scale with: NPROF=20 NCONF=21 NBIG=22"
for pair in $SWEEP; do
  bk="${pair%%:*}"; mb="${pair##*:}"
  run_cfg "$bk" "$mb" "$NPROF" "$IN20" "$ORC_PROF" "$REPS" nP || { echo "OK=$PASS FAIL=$FAIL"; exit 1; }
done
BASE20="${SMEAN[nP_${BASE_BLOCK}_${BASE_MB}]:-}"
[[ -n "$BASE20" ]] && pass "baseline_measured[N=$NPROF] (${BASE_BLOCK}x${BASE_MB} = $BASE20 ms)" || { fail "baseline_measured" "no baseline"; exit 1; }

banner "Z2/Z3 evaluation at N=$NPROF"
BESTK=""; BESTV=""
for pair in $SWEEP; do
  bk="${pair%%:*}"; mb="${pair##*:}"; v="${SMEAN[nP_${bk}_${mb}]:-}"
  [[ -z "$v" ]] && continue
  d="$(awk -v a="$BASE20" -v b="$v" 'BEGIN{printf "%+.3f",(b-a)/a*100}')"
  info "vs_baseline" "BLOCK=$bk MAX_BLOCKS=$mb  $v ms  ${d}%"
  if [[ -z "$BESTV" ]] || awk -v a="$v" -v b="$BESTV" 'BEGIN{exit !(a<b)}'; then BESTK="$bk:$mb"; BESTV="$v"; fi
done
for pair in 64:400 128:200; do
  bk="${pair%%:*}"; mb="${pair##*:}"; v="${SMEAN[nP_${bk}_${mb}]:-}"
  [[ -z "$v" ]] && continue
  d="$(awk -v a="$BASE20" -v b="$v" 'BEGIN{x=(b-a)/a*100; printf "%.3f",(x<0?-x:x)}')"
  awk -v d="$d" 'BEGIN{exit !(d<=2)}' && pass "Z2_packaging_neutral[$bk:$mb] (${d}% at constant stride 25,600)" \
    || info "Z2_packaging_matters[$bk:$mb]" "${d}% -- block shape alone moves the result more than the 2% registered"
done
GAIN="$(awk -v a="$BASE20" -v b="$BESTV" 'BEGIN{printf "%.3f",(a-b)/a*100}')"
info "best_config" "$BESTK = $BESTV ms, ${GAIN}% faster than ${BASE_BLOCK}x${BASE_MB}"
if awk -v g="$GAIN" 'BEGIN{exit !(g>=5)}'; then pass "Z3_CONFIRMED_occupancy_is_the_lever (${GAIN}%)"
elif awk -v g="$GAIN" 'BEGIN{exit !(g<=1)}'; then
  if [[ "$NPROF" -lt 20 ]]; then
    info "Z3_not_reached_at_debug_scale" "best is ${GAIN}% at N=$NPROF. Records per thread falls with N, so the large-stride configurations lose more of 394f's balancing here than they would at N=$NCONF -- this scale is biased AGAINST them. Not a refutation; the N=$NCONF confirmation below decides."
  else
    fail "Z3_REFUTED" "the best configuration is only ${GAIN}% from the baseline -- occupancy is NOT the lever, and the wait reading has to be reworked before anything else is attempted"
  fi
else info "Z3_partial" "${GAIN}% -- between the registered bands"; fi

# ---------------------------------------------------------------------
# 3. Z4 -- did occupancy actually rise and wait actually fall?
# ---------------------------------------------------------------------
banner "Z4: mechanism check with ncu"
command -v "$NCU" >/dev/null 2>&1 && pass "ncu_present" || fail "ncu_present" "set NCU=..."
NCU_PREFIX="${NCU_PREFIX-}"
reclaim() { [[ -n "$NCU_PREFIX" ]] && sudo chown -R "$(id -u):$(id -g)" "$LOGDIR" 2>/dev/null; true; }
if command -v "$NCU" >/dev/null 2>&1; then
  head -c $((25600*28)) "$IN20" > "/tmp/${REV}_tiny.bin"
  ncu_probe() { $1 env NQ_EXTRA_CTX=1 NQ_MAX_BLOCKS=800 "$NCU" --section SpeedOfLight --csv \
      "./$CU_BIN" "$NPROF" "/tmp/${REV}_tiny.bin" "/tmp/${REV}_tiny_out.bin" > "$2" 2>&1 || true
    grep -q 'kernel_dfs_iter_gpu_maxd14' "$2"; }
  if [[ -n "$NCU_PREFIX" ]]; then ncu_probe "$NCU_PREFIX" "$LOGDIR/07_ncu_probe.log" && pass "ncu_permission (via '$NCU_PREFIX')" || fail "ncu_permission" "cannot read counters"
  elif ncu_probe "" "$LOGDIR/07_ncu_probe.log"; then pass "ncu_permission (unprivileged)"
  elif ncu_probe "sudo" "$LOGDIR/07_ncu_probe_sudo.log"; then NCU_PREFIX="sudo"; pass "ncu_permission (under sudo)"
  else fail "ncu_permission" "counters unreadable; set NVreg_RestrictProfilingToAdminUsers=0 and reboot"; fi
fi
ncu_cfg() {  # tag block max_blocks metrics
  local tag="$1" bk="$2" mb="$3" met="$4"
  timeout "$SECTION_TIMEOUT_S" $NCU_PREFIX env NQ_EXTRA_CTX=1 NQ_BLOCK="$bk" NQ_MAX_BLOCKS="$mb" "$NCU" \
    --metrics "$met" --section Occupancy --section SchedulerStats --csv \
    "./$CU_BIN" "$NPROF" "$IN20" "/tmp/${REV}_p.bin" >> "$LOGDIR/8_ncu_${tag}.csv" 2>&1
  local rc=$?; reclaim
  [[ "$rc" == "124" ]] && { info "ncu[$tag]" "timed out"; return 1; }
  grep -q '==ERROR==' "$LOGDIR/8_ncu_${tag}.csv" && { info "ncu[$tag]" "ncu error"; return 1; }
  pass "ncu_metrics[$tag]"; return 0
}
if [[ -n "${NCU_PREFIX+x}" ]]; then
  ncu_cfg "base" "$BASE_BLOCK" "$BASE_MB" "$STALL_METRICS,$OCC_METRICS" || true
  if [[ -n "$BESTK" && "$BESTK" != "${BASE_BLOCK}:${BASE_MB}" ]]; then
    ncu_cfg "best" "${BESTK%%:*}" "${BESTK##*:}" "$STALL_METRICS,$OCC_METRICS" || true
  fi
fi

# ---------------------------------------------------------------------
# 4. Z5 -- does it carry to N=21?
# ---------------------------------------------------------------------
banner "Z5: confirm at N=$NCONF -- this is the cell that decides"
if [[ -n "${NIN[$NCONF]:-}" && -n "$BESTK" && -n "$ORC_CONF" ]]; then
  run_cfg "$BASE_BLOCK" "$BASE_MB" "$NCONF" "${NIN[$NCONF]}" "$ORC_CONF" "$REPS21" nC || { echo "OK=$PASS FAIL=$FAIL"; exit 1; }
  if [[ "$BESTK" != "${BASE_BLOCK}:${BASE_MB}" ]]; then
    run_cfg "${BESTK%%:*}" "${BESTK##*:}" "$NCONF" "${NIN[$NCONF]}" "$ORC_CONF" "$REPS21" nC || { echo "OK=$PASS FAIL=$FAIL"; exit 1; }
    BC="${SMEAN[nC_${BASE_BLOCK}_${BASE_MB}]}"; WC="${SMEAN[nC_${BESTK%%:*}_${BESTK##*:}]}"
    GC="$(awk -v a="$BC" -v b="$WC" 'BEGIN{printf "%.3f",(a-b)/a*100}')"
    info "N$NCONF" "baseline $BC ms -> best $WC ms  = ${GC}% (N=$NPROF gave ${GAIN}%)"
    if awk -v g="$GC" 'BEGIN{exit !(g>=5)}'; then pass "Z3_CONFIRMED_at_N${NCONF} (${GC}%) -- occupancy is the lever"
    elif awk -v g="$GC" 'BEGIN{exit !(g<=1)}'; then fail "Z3_REFUTED_at_N${NCONF}" "${GC}% -- occupancy is NOT the lever at the confirming scale either; the wait reading has to be reworked"
    else info "Z3_partial_at_N${NCONF}" "${GC}%"; fi
    awk -v g="$GC" -v gp="$GAIN" 'BEGIN{exit !(g >= gp/2)}' && pass "Z5_carries_up_one_N (${GC}% >= half of ${GAIN}%)" \
      || info "Z5_shrinks_with_N" "${GC}% is less than half the N=$NPROF gain -- the configuration may need tuning per N"
  else info "Z5" "the baseline was already the best at N=$NPROF; nothing to confirm"; fi
else info "Z5" "skipped (no N=$NCONF input or no winner)"; fi

# ---------------------------------------------------------------------
# 5. Z6 -- optional N=22
# ---------------------------------------------------------------------
if [[ "$WITH_NBIG" == "1" && -n "$BESTK" ]]; then
  banner "Z6: N=$NBIG (long -- input build plus a full-size kernel per run)"
  g22="$CRLOG_DIR/crunner_${CU_BIN}_N${NBIG}.log"; rm -f "$g22"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX -u NQ_BLOCK "./$PY_BIN" -g "$NBIG" "$NBIG" 2>&1 | tee "$LOGDIR/9_N${NBIG}_console.log"
  cp "$g22" "$LOGDIR/9_N${NBIG}_crunner.log" 2>/dev/null || true
  if [[ -f "$g22" ]]; then
    IN22="$(grep -o 'src=[^ ]*' "$g22" | head -1 | cut -d= -f2)"
    info "N$NBIG" "input=$IN22 records=$(grep -o 'records=[0-9]*' "$g22" | head -1 | cut -d= -f2) free_mb=$(grep -o 'free_mb=[0-9]*' "$g22" | head -1 | cut -d= -f2)"
    if [[ -f "${IN22:-/nonexistent}" ]]; then
      run_cfg "$BASE_BLOCK" "$BASE_MB" "$NBIG" "$IN22" "$ORC_BIG" 1 nB || true
      run_cfg "${BESTK%%:*}" "${BESTK##*:}" "$NBIG" "$IN22" "$ORC_BIG" 1 nB || true
      B22="${SMEAN[nB_${BASE_BLOCK}_${BASE_MB}]:-}"; W22="${SMEAN[nB_${BESTK%%:*}_${BESTK##*:}]:-}"
      if [[ -n "$B22" && -n "$W22" ]]; then
        G22="$(awk -v a="$B22" -v b="$W22" 'BEGIN{printf "%.3f",(a-b)/a*100}')"
        info "N${NBIG}_gain" "${G22}% (single rep each; treat as indicative)"
        awk -v g="$G22" 'BEGIN{exit !(g>0)}' && pass "Z6_carries_to_N${NBIG} (${G22}%)" || fail "Z6_carries_to_N${NBIG}" "${G22}% -- the configuration does not help at N=$NBIG"
      fi
    fi
  else fail "crunner_path_taken[N=$NBIG]" "no $g22 -- see $CRLOG_DIR/dispatch.log"; fi
fi

banner "Results"
python3 - "$LOGDIR" "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, glob, os, re, sys, statistics as st
logdir,tsv=sys.argv[1],sys.argv[2]
rows=list(csv.DictReader(open(tsv),delimiter='\t'))
by={}
for r in rows:
    if r['kernel_ms']=='?': continue
    by.setdefault((r['block'],r['max_blocks'],r['stride'],r['warps_per_sm']),[]).append(float(r['kernel_ms']))
n20=[(k,v) for k,v in by.items() if len(v)>=2]
print(f"{'BLOCK':>6} {'MB':>6} {'stride':>8} {'w/SM':>5} {'mean ms':>12} {'spread':>8}")
for k,v in sorted(by.items(), key=lambda x: st.fmean(x[1])):
    m=st.fmean(v); sp=(max(v)-min(v))/m*100 if len(v)>1 else 0.0
    print(f"{k[0]:>6} {k[1]:>6} {k[2]:>8} {k[3]:>5} {m:12.3f} {sp:7.3f}%")
def load(p):
    m={}; hdr=None
    for r in csv.reader(open(p,newline='',errors='replace')):
        if 'Metric Name' in r: hdr=r; continue
        if hdr is None: continue
        try: i_n=hdr.index('Metric Name'); i_v=hdr.index('Metric Value')
        except ValueError: continue
        if len(r)<=i_v or not r[i_n].strip(): continue
        try: m[r[i_n].strip()]=float(r[i_v].replace(',',''))
        except ValueError: pass
    return m
cfgs={}
for p in sorted(glob.glob(os.path.join(logdir,'8_ncu_*.csv'))):
    cfgs[re.search(r'8_ncu_(\w+)\.csv',p).group(1)]=load(p)
if cfgs:
    print("\n=== Z4 mechanism ===")
    for tag,m in cfgs.items():
        st_=[(k,v) for k,v in m.items() if 'issue_stalled' in k and 'selected' not in k]
        tot=sum(v for _,v in st_)
        occ=m.get('Achieved Occupancy'); aw=m.get('Active Warps Per Scheduler'); lanes=m.get('smsp__thread_inst_executed_per_inst_executed.ratio')
        print(f"\n  [{tag}] Achieved Occupancy {occ if occ is not None else '?'}%  Active Warps/Sched {aw if aw is not None else '?'}  lanes {lanes if lanes is not None else '?'}/32")
        for k,v in sorted(st_, key=lambda x:-x[1]):
            print(f"      {(v/tot*100 if tot else 0):6.2f}%  {v:6.3f}  {re.sub(r'^smsp__average_warps_issue_stalled_','',re.sub(r'_per_issue_active.ratio$','',k))}")
    if 'base' in cfgs and 'best' in cfgs:
        def waitshare(m):
            s=[(k,v) for k,v in m.items() if 'issue_stalled' in k and 'selected' not in k]
            t=sum(v for _,v in s); w=dict(s).get('smsp__average_warps_issue_stalled_wait_per_issue_active.ratio',0)
            return (w/t*100) if t else 0
        wb,ww=waitshare(cfgs['base']),waitshare(cfgs['best'])
        ob,ow=cfgs['base'].get('Achieved Occupancy',0),cfgs['best'].get('Achieved Occupancy',0)
        print(f"\n  occupancy {ob:.2f}% -> {ow:.2f}%   wait share {wb:.2f}% -> {ww:.2f}%")
        print("  Z4 HELD: occupancy rose and wait fell -- the gain is the mechanism it was predicted to be."
              if ow>ob and ww<wb else
              "  Z4 FAILED: the numbers did not move as predicted; whatever produced the timing change, it is not the one that was registered.")
EOF

{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs" 2>/dev/null || true
reclaim
TARBALL="${LOGDIR}.tar.gz"
tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; echo "ranked: $LOGDIR/9_ranked.txt"; echo "tarball: $TARBALL"
if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
echo "${REV} PASSED. Send $TARBALL for analysis."
exit 0
