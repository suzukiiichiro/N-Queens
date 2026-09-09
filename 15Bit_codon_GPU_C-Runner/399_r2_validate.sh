#!/usr/bin/env bash
# 399_validate.sh
#
# rev399 -- finish 398's homework, then open the lane-utilisation axis.
#
# WHAT 398-r2 MEASURED (N=20 and N=19, each with its own 394f schedule)
#   Avg. Active Threads Per Warp  10.53/32 (N=19 10.25)
#   sectors per local request      4.690 ld, 4.691 st (N=19 4.588)
#   L1/TEX 70.64%  Compute 31.55%  DRAM 25.22%
#   No Eligible 60.53% (N=21: 61.63%)   Active Warps/Sched 2.43 (N=21: 2.41)
#   Two multiplicative losses per local access: lanes 32.9%, sector
#   efficiency 56.1%, combined 18.5%.
#   The (a) volume vs (b) divergence fork was a false dichotomy -- both are
#   the same wall. 397's idea was sound; only its masking of rd was invalid.
#
# WHAT IS STILL OWED
#   U5. WarpStateStats completed but its SECTION only reports four summary
#   metrics; the per-reason breakdown needs the smsp__ metrics named
#   explicitly. That is a harness defect, and phase A repays it.
#
# THE NEW AXIS
#   Lane utilisation of 32.9% is the largest single inefficiency measured so
#   far. Thread tid takes records tid, tid+stride, tid+2*stride, ..., so at
#   step k the 32 lanes of warp w hold records [k*stride + w*32 .. +31]:
#   thirty-two CONSECUTIVE records in the scheduled file. Which records
#   share a warp is therefore a property of the host-side ordering.
#   And reordering is safe by construction: records are independent tasks
#   and the answer is their sum, so every permutation must produce the same
#   total_sum. Every ordering below is still gated on the oracle.
#
# PHASES
#   A  explicit stall metrics at N=20 and N=19        ~10 min
#   B  grouping analysis of the scheduled file        no GPU, ~1 min
#   C  ordering experiment at N=20, 4 orderings x 3   ~5 min
#   D  lane utilisation of the best ordering vs C0    ~8 min
#
# PRE-REGISTERED (399_README_append.md; written before execution)
#   X1  the leading stall reason is long-scoreboard (memory latency) at
#       >= 25%. STATED AGAINST the 394a-era belief that branch resolving
#       dominates: L1 is the top pipe at 70.64%, issue costs 6.14 warp
#       cycles, and there are 164,000 local accesses per record. If branch
#       resolving still leads, the lane-utilisation reading needs rework.
#   X2  within-group variance / total variance >= 0.8 for the cost proxies,
#       i.e. 394f does NOT group similar records into a warp. <= 0.3 would
#       mean it already does and this axis has little room. No GPU needed.
#   X3  a random shuffle is >= 2% SLOWER than the 394f order at N=20.
#       This is the control that says ordering matters at all. If the two
#       are within 0.5%, ordering is irrelevant and phase C's axis is dead
#       -- a valuable negative that would save a lot of wasted effort.
#   X4  STATED PREDICTION: no proxy sort beats C0 by more than 1%. 394f was
#       designed for exactly this problem. The value of this phase is X3
#       and the lane-utilisation measurement in phase D, not a win.
#   X5  any ordering whose kernel_ms moves by >= 1% moves Avg Active
#       Threads Per Warp in the same direction, tying the metric to the
#       mechanism.
#
#   PERMUTATION INVARIANCE IS A HARD GATE. Every ordering must produce
#   total_sum 39,029,188,884 at N=20. If one does not, records are not
#   independent and a premise of this whole project is wrong; the harness
#   stops there rather than reporting a time.
#
# USAGE
#   STATIC_ONLY=1 bash 399_validate.sh
#                 bash 399_validate.sh          # ~35 min
#   NCU_PREFIX=sudo bash 399_validate.sh
#
# PRIVILEGE: do not run the whole harness under sudo. Only the ncu calls
# are elevated and their outputs are chowned back.

set -u

REV="399_r2"
PY_SRC="${PY_SRC:-399_r2Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-399_r2Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-399Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-399_r2_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-399_r2_kernel_maxd14}"
PREV_CU="${PREV_CU:-399_kernel_maxd14.cu}"
CRLOG_DIR="${CRLOG_DIR:-399_r2_crunner_logs}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
NCU="${NCU:-ncu}"
ARCH="${ARCH:-sm_86}"
NPROF="${NPROF:-20}"
NALT="${NALT:-19}"
N20_ORACLE="${N20_ORACLE:-39029188884}"
N19_ORACLE="${N19_ORACLE:-4968057848}"
MB_PROD="${MB_PROD:-800}"
STRIDE=$((32*MB_PROD))
REPS="${REPS:-3}"
KERNEL_SHA_395A="${KERNEL_SHA_395A:-ebd7f523deb591347eab1a352a9555c2e4a6c0b4a456d80517322c933a86e68a}"
SECTION_TIMEOUT_S="${SECTION_TIMEOUT_S:-600}"
STALL_METRICS="smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio,smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio,smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_misc_per_issue_active.ratio,smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__average_warps_issue_stalled_selected_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio,smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
LANE_METRICS="smsp__thread_inst_executed_per_inst_executed.ratio,smsp__average_warps_active_per_issue_active.ratio"
STATIC_ONLY="${STATIC_ONLY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"
TSV="$LOGDIR/${REV}_orderings.tsv"

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
grep -qE '^REV_TAG:str="399_r2"' "$CODE" && pass "source_rev_tag_is_399_r2" || fail "source_rev_tag_is_399_r2" "wrong REV_TAG"
grep -q 'CRunnerEntry(14,"./399_r2_kernel_maxd14","NQ_EXTRA_CTX=1 "' "$CODE" && pass "source_table_points_at_399_r2_and_keeps_the_treatment" || fail "source_table_points_at_399_r2_and_keeps_the_treatment" "table entry wrong"
grep -qE '^CRUNNER_MIN_N:int=19' "$CODE" && pass "source_crunner_min_n_still_19 (carried from 398-r2)" || fail "source_crunner_min_n_still_19" "CRUNNER_MIN_N lost"
grep -qE '^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=800' "$CODE" && pass "source_default_max_blocks_800" || fail "source_default_max_blocks_800" "not 800"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_399Py (removed=3 added=3 EXECUTABLE lines)" \
    || { fail "py_diff_fingerprint_vs_399Py" "removed=$PR_ added=$PA_, expected 3/3"; diff "/tmp/${REV}_prev_code.py" "$CODE" | head -20 | sed 's/^/      /'; }
else info "py_diff_fingerprint_vs_399Py" "skipped"; fi
cu_code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA_395A" ]] && pass "cu_kernel_region_still_395a_r2 (${KB:0:16}...)" || fail "cu_kernel_region_still_395a_r2" "got ${KB:0:16}..."
if [[ -f "$PREV_CU" ]]; then
  cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  CA="$(sha256sum < "/tmp/${REV}_prev_code.cu" | cut -d' ' -f1)"; CB="$(sha256sum < "/tmp/${REV}_cur_code.cu" | cut -d' ' -f1)"
  [[ "$CA" == "$CB" ]] && pass "cu_whole_code_region_identical_to_399 (${CB:0:16}...)" || fail "cu_whole_code_region_identical_to_399" "399-r2 must be a rename"
else info "cu_whole_code_region_identical_to_399" "skipped"; fi

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; "$NCU" --version 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1

banner "Building"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$CU_BIN" "$CU_SRC" -lcuda 2>&1 | tee "$LOGDIR/05_nvcc.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "no binary"; exit 1; }
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/06_codon.log"
[[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN]" || { fail "codon_build" "see log"; exit 1; }

banner "ncu permission probe (before anything expensive)"
# 399 REGRESSION, fixed here: this block had been abbreviated to a bare
# `command -v ncu`, so every profiling pass ran unprivileged and died on
# ERR_NVGPUCTRPERM. The probe below is the one that worked in 398-r2.
command -v "$NCU" >/dev/null 2>&1 && pass "ncu_present ($("$NCU" --version 2>&1 | head -1))" || { fail "ncu_present" "set NCU=/usr/local/cuda/bin/ncu"; exit 1; }
NCU_PREFIX="${NCU_PREFIX-}"
reclaim() { [[ -n "$NCU_PREFIX" ]] && sudo chown -R "$(id -u):$(id -g)" "$LOGDIR" 2>/dev/null; true; }
PROBE_IN=""
for cand in constellations_N20_6.bin.soa_ref_361.bin.maxd14only_363.bin.sched394f.bin \
            constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin.sched394f.bin; do
  [[ -f "$cand" ]] && { PROBE_IN="$cand"; break; }
done
if [[ -n "$PROBE_IN" ]]; then
  head -c $((25600*28)) "$PROBE_IN" > "/tmp/${REV}_tiny.bin"
  ncu_probe() { $1 env NQ_EXTRA_CTX=1 NQ_MAX_BLOCKS="$MB_PROD" "$NCU" --section SpeedOfLight --csv \
      "./$CU_BIN" 21 "/tmp/${REV}_tiny.bin" "/tmp/${REV}_tiny_out.bin" > "$2" 2>&1 || true
    grep -q 'kernel_dfs_iter_gpu_maxd14' "$2"; }
  if [[ -n "$NCU_PREFIX" ]]; then
    ncu_probe "$NCU_PREFIX" "$LOGDIR/02_ncu_probe.log" && pass "ncu_permission (via '$NCU_PREFIX')" || { fail "ncu_permission" "NCU_PREFIX='$NCU_PREFIX' cannot read counters"; exit 1; }
  elif ncu_probe "" "$LOGDIR/02_ncu_probe.log"; then
    pass "ncu_permission (unprivileged)"
  else
    if grep -qi 'ERR_NVGPUCTRPERM\|permission' "$LOGDIR/02_ncu_probe.log"; then
      info "ncu_permission" "admin-restricted unprivileged; retrying under sudo (only the ncu calls are elevated)"
      if ncu_probe "sudo" "$LOGDIR/02_ncu_probe_sudo.log"; then NCU_PREFIX="sudo"; pass "ncu_permission (under sudo; outputs chowned back)"
      else fail "ncu_permission" "sudo did not help. Set it permanently: echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiler.conf && sudo update-initramfs -u && reboot"; exit 1; fi
    else fail "ncu_permission" "ncu ran but did not report the kernel; see $LOGDIR/02_ncu_probe.log"; exit 1; fi
  fi
else
  fail "ncu_permission" "no scheduled input on disk to probe with -- run 398-r2 or 399 first so the N=20/N=21 inputs exist"
  exit 1
fi

# ---------------------------------------------------------------------
# 2. Inputs: let the dispatcher build and time N=20 and N=19
# ---------------------------------------------------------------------
declare -A NIN NMS NREC
for n in "$NPROF" "$NALT"; do
  banner "Generating and timing N=$n via ./$PY_BIN -g $n $n"
  gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${n}.log"; rm -f "$gcr"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX "./$PY_BIN" -g "$n" "$n" 2>&1 | tee "$LOGDIR/1_gen_N${n}_console.log"
  cp "$gcr" "$LOGDIR/1_gen_N${n}_crunner.log" 2>/dev/null || true
  if [[ ! -f "$gcr" ]]; then fail "crunner_path_taken[N=$n]" "no $gcr -- check $CRLOG_DIR/dispatch.log"; continue; fi
  NIN[$n]="$(grep -o 'src=[^ ]*' "$gcr" | head -1 | cut -d= -f2)"
  NMS[$n]="$(grep -o 'kernel_ms=[0-9.]*' "$gcr" | head -1 | cut -d= -f2)"
  NREC[$n]="$(grep -o 'records=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"
  [[ -f "${NIN[$n]:-/nonexistent}" ]] && pass "input_located[N=$n] (${NIN[$n]}, ${NREC[$n]} records, kernel_ms=${NMS[$n]})" || fail "input_located[N=$n]" "src='${NIN[$n]:-<none>}'"
done
[[ -n "${NIN[$NPROF]:-}" ]] || { fail "profiling_input_required" "N=$NPROF is the primary target"; exit 1; }

# ---------------------------------------------------------------------
# 3. Phase A -- explicit stall metrics (398's homework)
# ---------------------------------------------------------------------
banner "Phase A: explicit stall metrics"
ncu_run() {  # tag N input args...
  local tag="$1" n="$2" input="$3"; shift 3
  local t0 t1 rc
  t0=$(date +%s)
  timeout "$SECTION_TIMEOUT_S" $NCU_PREFIX env NQ_EXTRA_CTX=1 NQ_MAX_BLOCKS="$MB_PROD" "$NCU" "$@" --csv \
    "./$CU_BIN" "$n" "$input" "/tmp/${REV}_prof_out.bin" >> "$LOGDIR/3_ncu_${tag}.csv" 2>&1
  rc=$?; t1=$(date +%s); reclaim
  if [[ "$rc" == "124" ]]; then info "ncu[$tag]" "timed out after ${SECTION_TIMEOUT_S}s"; return 1; fi
  if grep -q '==ERROR==' "$LOGDIR/3_ncu_${tag}.csv"; then info "ncu[$tag]" "ncu error after $((t1-t0))s"; return 1; fi
  info "ncu[$tag]" "completed in $((t1-t0))s"; return 0
}
for n in "$NPROF" "$NALT"; do
  [[ -z "${NIN[$n]:-}" ]] && continue
  ncu_run "stall_N${n}" "$n" "${NIN[$n]}" --metrics "$STALL_METRICS" && pass "ncu_stall_metrics[N=$n]" || fail "ncu_stall_metrics[N=$n]" "see 3_ncu_stall_N${n}.csv"
done

# ---------------------------------------------------------------------
# 4. Phase B -- grouping analysis of the scheduled file (no GPU)
# ---------------------------------------------------------------------
banner "Phase B: does 394f group similar records into a warp? (no GPU)"
python3 - "${NIN[$NPROF]}" "$STRIDE" > "$LOGDIR/4_grouping.txt" 2>&1 <<'EOF'
import sys, struct, statistics
path, stride = sys.argv[1], int(sys.argv[2])
raw=open(path,'rb').read()
n=len(raw)//28
print(f"records={n}  stride={stride}  warp group = 32 consecutive records")
flds=['ld','rd','col','ctrl0','free','markctrl','w_lo']
cols=[[] for _ in flds]
for i in range(n):
    v=struct.unpack_from('<7I', raw, i*28)
    for j in range(7): cols[j].append(v[j])
def pc(x): return bin(x).count('1')
proxies={
  'popcount(free)': [pc(x) for x in cols[4]],
  'popcount(col)' : [pc(x) for x in cols[2]],
  'w_lo'          : cols[6],
}
ng=n//32
print(f"{'proxy':16s} {'total var':>14s} {'mean within-32 var':>20s} {'ratio':>8s}")
for name,vals in proxies.items():
    tv=statistics.pvariance(vals)
    wv=[]
    for g in range(ng):
        blk=vals[g*32:(g+1)*32]
        if len(blk)==32: wv.append(statistics.pvariance(blk))
    mwv=sum(wv)/len(wv) if wv else 0.0
    r=(mwv/tv) if tv else float('nan')
    print(f"{name:16s} {tv:14.4f} {mwv:20.4f} {r:8.3f}")
    if r>=0.8:   print(f"    -> X2 holds for this proxy: the 32 records in a warp are as varied as the file at large; 394f does not group by it.")
    elif r<=0.3: print(f"    -> already grouped by this proxy; little room on this axis.")
    else:        print(f"    -> partial grouping.")
EOF
cat "$LOGDIR/4_grouping.txt"

# ---------------------------------------------------------------------
# 5. Phase C -- ordering experiment. Permutations cannot change the answer.
# ---------------------------------------------------------------------
banner "Phase C: orderings at N=$NPROF (every one gated on the oracle)"
mk_order() {  # mode outfile
  python3 - "${NIN[$NPROF]}" "$2" "$1" <<'EOF'
import sys, struct, random
src,dst,mode=sys.argv[1],sys.argv[2],sys.argv[3]
raw=open(src,'rb').read(); n=len(raw)//28
idx=list(range(n))
if mode=='asis': pass
elif mode=='shuffle':
    random.seed(399); random.shuffle(idx)
else:
    def key(i):
        v=struct.unpack_from('<7I', raw, i*28)
        return bin(v[4]).count('1')          # popcount(free): root branching factor
    idx.sort(key=key, reverse=(mode=='desc'))
with open(dst,'wb') as g:
    for i in idx: g.write(raw[i*28:(i+1)*28])
EOF
}
printf 'ordering\trep\tkernel_ms\ttotal_sum\tmatch\n' > "$TSV"
declare -A OMEAN
for mode in asis shuffle asc desc; do
  ob="/tmp/${REV}_order_${mode}.bin"
  mk_order "$mode" "$ob" || { fail "order_build[$mode]" "permutation failed"; continue; }
  [[ "$(stat -c%s "$ob")" == "$(stat -c%s "${NIN[$NPROF]}")" ]] && pass "order_is_a_permutation[$mode] (same byte count)" || { fail "order_is_a_permutation[$mode]" "size changed"; continue; }
  sum=0; ok=1
  for r in $(seq 1 "$REPS"); do
    lg="$LOGDIR/5_order_${mode}_r${r}.log"
    env -u NQ_PAD_MB NQ_EXTRA_CTX=1 NQ_MAX_BLOCKS="$MB_PROD" "./$CU_BIN" "$NPROF" "$ob" "/tmp/${REV}_o_out.bin" "$N20_ORACLE" > "$lg" 2>&1 || true
    kms="$(grep -o 'kernel_ms=[0-9.]*' "$lg" | head -1 | cut -d= -f2)"
    tot="$(grep -o 'total_sum=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
    m=0; grep -q '\[gpu-run-correctness\] MATCH' "$lg" && m=1
    printf '%s\t%s\t%s\t%s\t%s\n' "$mode" "$r" "${kms:-?}" "${tot:-?}" "$m" >> "$TSV"
    if [[ "$m" != "1" || "${tot:-}" != "$N20_ORACLE" ]]; then
      fail "permutation_invariance[$mode rep$r]" "total_sum='${tot:-<none>}' expected $N20_ORACLE -- a permutation changed the answer, so records are NOT independent and a premise of this project is wrong. Stopping."
      ok=0; break
    fi
    sum="$(awk -v s="$sum" -v k="${kms:-0}" 'BEGIN{printf "%.3f",s+k}')"
  done
  [[ "$ok" == "1" ]] || { echo "OK=$PASS FAIL=$FAIL"; exit 1; }
  OMEAN[$mode]="$(awk -v s="$sum" -v r="$REPS" 'BEGIN{printf "%.3f",s/r}')"
  pass "oracle_match_all_reps[$mode] (mean kernel_ms=${OMEAN[$mode]})"
done
BASE="${OMEAN[asis]:-}"
if [[ -n "$BASE" ]]; then
  for mode in shuffle asc desc; do
    [[ -z "${OMEAN[$mode]:-}" ]] && continue
    info "ordering[$mode]" "$(awk -v a="$BASE" -v b="${OMEAN[$mode]}" 'BEGIN{printf "%.3f ms  %+.3f%% vs the 394f order",b,(b-a)/a*100}')"
  done
  if [[ -n "${OMEAN[shuffle]:-}" ]]; then
    d="$(awk -v a="$BASE" -v b="${OMEAN[shuffle]}" 'BEGIN{printf "%.3f",(b-a)/a*100}')"
    if awk -v d="$d" 'BEGIN{exit !(d>=2)}'; then pass "X3_ordering_matters (a random shuffle is ${d}% slower than 394f)"
    elif awk -v d="$d" 'BEGIN{x=(d<0?-d:d); exit !(x<=0.5)}'; then
      fail "X3_ordering_matters" "shuffle is only ${d}% from the 394f order -- ordering is IRRELEVANT at this scale, so the lane-utilisation axis cannot be reached by reordering. That is a real result: stop pursuing phase C's direction and take lane utilisation from the kernel side instead."
    else info "X3_partial" "shuffle is ${d}% -- between the pre-registered bands"; fi
  fi
  BEST="asis"
  for mode in shuffle asc desc; do
    [[ -z "${OMEAN[$mode]:-}" ]] && continue
    awk -v a="${OMEAN[$mode]}" -v b="${OMEAN[$BEST]}" 'BEGIN{exit !(a<b)}' && BEST="$mode"
  done
  info "best_ordering" "$BEST (${OMEAN[$BEST]} ms)"
  g="$(awk -v a="$BASE" -v b="${OMEAN[$BEST]}" 'BEGIN{printf "%.3f",(a-b)/a*100}')"
  awk -v g="$g" 'BEGIN{exit !(g>1)}' && info "X4_refuted" "$BEST beats the 394f order by ${g}% -- more than the 1% I predicted was impossible" \
    || pass "X4_no_proxy_sort_beats_394f_by_more_than_1pct (best gain ${g}%)"
fi

# ---------------------------------------------------------------------
# 6. Phase D -- lane utilisation of the best ordering vs the 394f order
# ---------------------------------------------------------------------
banner "Phase D: Y2/Y3 -- put a slope on the lane-utilisation axis"
# Three points, not one. asis and shuffle are the ends of a measured 35.2%
# time swing, so their lane utilisations calibrate how much a live lane is
# worth; desc sits in between and tests monotonicity.
for mode in asis shuffle desc; do
  ob="/tmp/${REV}_order_${mode}.bin"
  [[ -f "$ob" ]] || continue
  ncu_run "lane_${mode}" "$NPROF" "$ob" --metrics "$LANE_METRICS" && pass "ncu_lane_metrics[$mode]" || fail "ncu_lane_metrics[$mode]" "see 3_ncu_lane_${mode}.csv"
done

# ---------------------------------------------------------------------
# 7. Phase E -- a MEASURED cost proxy instead of a guessed one
# ---------------------------------------------------------------------
banner "Phase E: per-record solution counts as a cost proxy"
REC="${NREC[$NPROF]:-0}"
if [[ "$REC" -gt 0 ]]; then
  WIDE_BLOCKS=$(( (REC + 31) / 32 ))
  info "phaseE" "records=$REC -> NQ_MAX_BLOCKS=$WIDE_BLOCKS gives stride=$((WIDE_BLOCKS*32)) >= records, so each thread takes exactly one record and the output becomes per-RECORD counts"
  env -u NQ_PAD_MB NQ_EXTRA_CTX=1 NQ_MAX_BLOCKS="$WIDE_BLOCKS" "./$CU_BIN" "$NPROF" "${NIN[$NPROF]}" "/tmp/${REV}_percord.bin" "$N20_ORACLE" > "$LOGDIR/7_percord.log" 2>&1 || true
  if grep -q '\[gpu-run-correctness\] MATCH' "$LOGDIR/7_percord.log"; then
    pass "per_record_pass_oracle_match"
    for mode in mcost_asc mcost_desc; do
      ob="/tmp/${REV}_order_${mode}.bin"
      python3 - "${NIN[$NPROF]}" "/tmp/${REV}_percord.bin" "$ob" "$mode" <<'PYE'
import sys, struct
src,cost,dst,mode=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
raw=open(src,'rb').read(); n=len(raw)//28
c=open(cost,'rb').read()
w=[struct.unpack_from('<Q', c, i*8)[0] for i in range(n)]
idx=sorted(range(n), key=lambda i: w[i], reverse=(mode=='mcost_desc'))
with open(dst,'wb') as g:
    for i in idx: g.write(raw[i*28:(i+1)*28])
PYE
      if [[ ! -f "$ob" ]] || [[ "$(stat -c%s "$ob")" != "$(stat -c%s "${NIN[$NPROF]}")" ]]; then
        fail "order_is_a_permutation[$mode]" "size changed or build failed"; continue
      fi
      pass "order_is_a_permutation[$mode]"
      sum=0
      for r in $(seq 1 "$REPS"); do
        lg="$LOGDIR/7_order_${mode}_r${r}.log"
        env -u NQ_PAD_MB NQ_EXTRA_CTX=1 NQ_MAX_BLOCKS="$MB_PROD" "./$CU_BIN" "$NPROF" "$ob" "/tmp/${REV}_o_out.bin" "$N20_ORACLE" > "$lg" 2>&1 || true
        kms="$(grep -o 'kernel_ms=[0-9.]*' "$lg" | head -1 | cut -d= -f2)"
        tot="$(grep -o 'total_sum=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
        m=0; grep -q '\[gpu-run-correctness\] MATCH' "$lg" && m=1
        printf '%s\t%s\t%s\t%s\t%s\n' "$mode" "$r" "${kms:-?}" "${tot:-?}" "$m" >> "$TSV"
        [[ "$m" == "1" && "${tot:-}" == "$N20_ORACLE" ]] || { fail "permutation_invariance[$mode rep$r]" "total_sum='${tot:-<none>}'"; echo "OK=$PASS FAIL=$FAIL"; exit 1; }
        sum="$(awk -v s="$sum" -v k="${kms:-0}" 'BEGIN{printf "%.3f",s+k}')"
      done
      OMEAN[$mode]="$(awk -v s="$sum" -v r="$REPS" 'BEGIN{printf "%.3f",s/r}')"
      d="$(awk -v a="$BASE" -v b="${OMEAN[$mode]}" 'BEGIN{printf "%.3f",(b-a)/a*100}')"
      info "ordering[$mode]" "${OMEAN[$mode]} ms  ${d}% vs the 394f order"
      awk -v d="$d" 'BEGIN{exit !(d <= -1)}' \
        && pass "Y4_REFUTED_measured_cost_sort_wins[$mode] (${d}%) -- the first real gain from the ordering axis" \
        || info "Y4_holds[$mode]" "${d}% -- within the +-1% band I predicted"
    done
  else
    fail "per_record_pass_oracle_match" "the wide-stride pass did not match the oracle; see $LOGDIR/7_percord.log"
  fi
else info "phaseE" "skipped: record count unknown"; fi

banner "Ranked results"
python3 - "$LOGDIR" <<'EOF' | tee "$LOGDIR/6_ranked.txt"
import csv, glob, os, re, sys
logdir=sys.argv[1]
def load(p):
    m={}; hdr=None
    for r in csv.reader(open(p,newline='',errors='replace')):
        if 'Metric Name' in r: hdr=r; continue
        if hdr is None: continue
        try: i_n=hdr.index('Metric Name'); i_v=hdr.index('Metric Value'); i_u=hdr.index('Metric Unit')
        except ValueError: continue
        if len(r)<=i_v or not r[i_n].strip(): continue
        try: v=float(r[i_v].replace(',',''))
        except ValueError: continue
        m[r[i_n].strip()]=(v, r[i_u].strip() if len(r)>i_u else '')
    return m
for p in sorted(glob.glob(os.path.join(logdir,'3_ncu_stall_N*.csv'))):
    n=re.search(r'stall_N(\d+)',p).group(1); m=load(p)
    st=[(k,v) for k,(v,u) in m.items() if 'issue_stalled' in k]
    if not st: print(f"\nN={n}: no stall metrics parsed from {os.path.basename(p)}"); continue
    tot=sum(v for _,v in st); st.sort(key=lambda x:-x[1])
    print(f"\n=== X1 warp stall reasons, N={n} (total {tot:.3f} warp cycles per issue) ===")
    for k,v in st:
        short=re.sub(r'^smsp__average_warps_issue_stalled_','',re.sub(r'_per_issue_active\.ratio$','',k))
        print(f"  {(v/tot*100 if tot else 0):6.2f}%  {v:8.4f}  {short}")
    lead=re.sub(r'^smsp__average_warps_issue_stalled_','',re.sub(r'_per_issue_active\.ratio$','',st[0][0]))
    share=st[0][1]/tot*100 if tot else 0
    print(f"  dominant: {lead} at {share:.2f}%")
    if n=='20':
        if 'long_scoreboard' in lead and share>=25:
            print("  X1 HELD: memory latency leads. The 394a-era 'branch resolving dominates' picture is superseded.")
        elif 'branch' in lead:
            print("  X1 FALSIFIED: branch resolving still leads. The lane-utilisation reading has to be reworked before acting on it.")
        else:
            print(f"  X1 neither: {lead} leads. Read the whole table before choosing 400's direction.")
lanes={}
for p in sorted(glob.glob(os.path.join(logdir,'3_ncu_lane_*.csv'))):
    mode=re.search(r'lane_([a-z]+)\.csv',p).group(1); m=load(p)
    v=m.get('smsp__thread_inst_executed_per_inst_executed.ratio',(None,''))[0]
    if v is not None: lanes[mode]=v
if lanes:
    print("\n=== X5 lane utilisation (active threads per executed instruction, out of 32) ===")
    for k,v in lanes.items(): print(f"  {k:8s} {v:6.2f} / 32 = {v/32*100:5.1f}%   (398-r2 measured 10.53 on the 394f order)")
    if 'asis' in lanes and len(lanes)>1:
        for k,v in lanes.items():
            if k!='asis': print(f"  {k} vs asis: {v-lanes['asis']:+.2f} threads per warp")
    if 'shuffle' in lanes:
        sh=lanes['shuffle']; ai=lanes['asis']
        print(f"\n  Y2: predicted shuffle 7.79/32 from inverse proportionality; measured {sh:.2f}")
        print("  Y2 HELD: time and lane utilisation are inversely proportional to first order." if 7.0<=sh<=8.6
              else "  Y2 FAILED: the relation is not simple inverse proportionality -- the axis cannot be priced this way yet.")
        if ai>sh:
            print(f"  slope: {(22285.509-16482.743)/16482.743*100:.2f}% of time per {ai-sh:.2f} lanes = {((22285.509-16482.743)/16482.743*100)/(ai-sh):.2f}% per live lane")
            print(f"  extrapolation (linear, treat with suspicion): 32/32 lanes would be {(32-ai)*((22285.509-16482.743)/16482.743*100)/(ai-sh):.1f}% faster than the 394f order")
EOF

{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs" 2>/dev/null || true
reclaim
echo; cat "$TSV"
TARBALL="${LOGDIR}.tar.gz"
tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; echo "ranked: $LOGDIR/6_ranked.txt"; echo "tarball: $TARBALL"
if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
echo "${REV} PASSED. Send $TARBALL for analysis."
exit 0
