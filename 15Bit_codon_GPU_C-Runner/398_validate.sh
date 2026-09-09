#!/usr/bin/env bash
# 398_validate.sh
#
# rev398 -- measure instead of build. Profile N=20 and N=19, answer the warp
#           stall question, and settle volume vs divergence.
#
# WHERE THIS STANDS
#   397 tried to test the volume hypothesis by shrinking the DFS stack frame.
#   ptxas confirmed 208 -> 160 bytes with no new spills, and the equivalence
#   gate then rejected it before any timing was read: total_sum came out
#   427,786,151,034 against the oracle 314,666,222,712.
#   The reason is structural. Every use of cur_ld is a LEFT shift and every
#   use of cur_rd is a RIGHT shift, so bits of rd above the board mask move
#   DOWN into range as the search descends and are live constraints. Masking
#   rd drops them. With true widths -- ld 20, rd 32, col 21, avail 21,
#   depth 4 -- a frame needs 98 bits and twelve bytes is 96. The packing
#   route is two bits short and 398 does not pursue it.
#
#   397 was written before the diagnosis was finished. 398 finishes it.
#
# WHAT IS STILL UNKNOWN
#   U5  the dominant warp stall reason. WarpStateStats has never completed:
#       396 tried it twice on the full N=21 kernel and ncu errored out both
#       times. It DID succeed on a 1.5 s input, so kernel duration is the
#       blocker, not permissions.
#   (a) vs (b)  is the local-memory cost byte VOLUME or address DIVERGENCE?
#       This decides whether any packing work is worth attempting at all.
#
# WHY N=20 AND N=19
#   Record subsets are not usable: 396 and 396-r2 both measured that
#   truncation runs 13-26% cheap per record and stratified sampling 12-37%
#   dear, because 394f balances cumulative per-thread work over the WHOLE
#   set and any subset breaks that balance. A single round alone costs
#   1.74x what it costs inside the full run.
#   N=20 and N=19 each carry their OWN 394f schedule, so they are balanced
#   by construction. N=20 is about 16 s and N=19 about 2 s -- both inside
#   the range where ncu has been observed to work.
#   Representativeness is not assumed. N=21's Occupancy and Scheduler
#   Statistics were measured by 396, so W6 below compares them directly.
#
# PRE-REGISTERED (398_README_append.md; written before execution)
#   W1  the anchor (direct run, NQ_EXTRA_CTX=2) lands within +-0.15% of
#       133,185. 397 established this shortcut (E0 = 133,155.344, -0.022%),
#       so no dispatcher run is needed and every measurement is 2.5 min
#       shorter.
#   W2  N=20 kernel_ms in 14,000-20,000; N=19 in 1,500-3,000. Informational.
#   W3  the dominant warp stall reason at N=20 is branch resolving, at
#       >= 25% of warp stall cycles. Falsified -> the bottleneck has moved
#       and 399 changes optimisation family.
#   W4  local-memory sectors per request.
#       STATED PREDICTION: >= 2, i.e. (b) DIVERGENCE dominates. Local memory
#       is interleaved across threads, so a warp whose lanes all touch the
#       same word index coalesces perfectly; DFS depth differs per lane, the
#       indices scatter, and the request splits. Depth divergence is
#       inherent to this algorithm.
#       <= 1.2 would mean coalescing is fine and volume is what costs -- and
#       only then is it worth working out how to fit 98 bits into 96.
#   W5  the top three stall reasons agree in order between N=19 and N=20.
#       If they do, the ranking is adopted as the answer for N=21. If not,
#       the ranking is N-dependent and N=21 has to be profiled overnight.
#   W6  representativeness, checked rather than assumed: N=20's achieved
#       occupancy within +-3 points of 10.32%, and No Eligible within
#       +-5 points of 61.63%.
#
# USAGE
#   STATIC_ONLY=1 bash 398_validate.sh
#                 bash 398_validate.sh          # ~35 min
#   NCU_PREFIX=sudo bash 398_validate.sh        # force elevation for ncu only
#
# PRIVILEGE: do not run the whole harness under sudo -- it would leave the
# logs, binaries and tarball owned by root. Only the ncu calls are elevated,
# and their outputs are chowned back.

set -u

REV="398"
PY_SRC="${PY_SRC:-398Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-398Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-396_r2Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-398_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-398_kernel_maxd14}"
PREV_CU="${PREV_CU:-396_r2_kernel_maxd14.cu}"
CRLOG_DIR="${CRLOG_DIR:-398_crunner_logs}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
NCU="${NCU:-ncu}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
ORACLE="${ORACLE:-314666222712}"
IN_RAW="${IN_RAW:-constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin}"
IN_PROD="${IN_PROD:-${IN_RAW}.sched394f.bin}"
EXPECTED_RECORDS="${EXPECTED_RECORDS:-2025282}"
MB_PROD="${MB_PROD:-800}"
REF_3CTX_MS="${REF_3CTX_MS:-133185}"
TOL_ANCHOR="${TOL_ANCHOR:-0.15}"
PROFILE_NS="${PROFILE_NS:-20 19}"
KERNEL_SHA_395A="${KERNEL_SHA_395A:-ebd7f523deb591347eab1a352a9555c2e4a6c0b4a456d80517322c933a86e68a}"
# N=21 reference values, measured by 396 -- used by W6
REF_ACHIEVED_OCC="${REF_ACHIEVED_OCC:-10.32}"
REF_NO_ELIGIBLE="${REF_NO_ELIGIBLE:-61.63}"
SECTION_TIMEOUT_S="${SECTION_TIMEOUT_S:-600}"
NCU_BUDGET_S="${NCU_BUDGET_S:-1800}"
LOCAL_METRICS="l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,l1tex__t_requests_pipe_lsu_mem_local_op_st.sum"
STATIC_ONLY="${STATIC_ONLY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }

# ---------------------------------------------------------------------
# 1. Static
# ---------------------------------------------------------------------
for f in "$PY_SRC" "$HELPER_SRC" "$CU_SRC" "$IN_RAW"; do
  [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"
done
[[ -f "$IN_PROD" ]] && pass "sched_input_present[$IN_PROD]" || fail "sched_input_present" "$IN_PROD missing"
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
{ grep -q "^# ${REV} " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" \
  || fail "py_revision_notes_present" "no '# ${REV} ...' note block in $PY_SRC"
NQ_=$(py_note_quotes "$PY_SRC"); [[ $((NQ_ % 2)) -eq 0 ]] && pass "py_note_region_quotes_balanced ($NQ_)" || fail "py_note_region_quotes_balanced" "$NQ_ triple-quotes"
grep -qE '^REV_TAG:str="398"' "$CODE" && pass "source_rev_tag_is_398" || fail "source_rev_tag_is_398" "wrong REV_TAG"
grep -q 'CRunnerEntry(14,"./398_kernel_maxd14","NQ_EXTRA_CTX=1 "' "$CODE" && pass "source_table_points_at_398_and_keeps_the_treatment" || fail "source_table_points_at_398_and_keeps_the_treatment" "table entry wrong"
grep -qE '^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=800' "$CODE" && pass "source_default_max_blocks_800" || fail "source_default_max_blocks_800" "not 800"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_396_r2Py (removed=3 added=3 EXECUTABLE lines)" \
    || { fail "py_diff_fingerprint_vs_396_r2Py" "removed=$PR_ added=$PA_, expected 3/3"; diff "/tmp/${REV}_prev_code.py" "$CODE" | head -30 | sed 's/^/      /'; }
else info "py_diff_fingerprint_vs_396_r2Py" "skipped"; fi
cu_code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA_395A" ]] && pass "cu_kernel_region_back_to_395a_r2 (${KB:0:16}... -- 397 is discarded)" \
  || fail "cu_kernel_region_back_to_395a_r2" "got ${KB:0:16}..., expected ${KERNEL_SHA_395A:0:16}...; 398 must not carry 397's packing"
if [[ -f "$PREV_CU" ]]; then
  cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  CA="$(sha256sum < "/tmp/${REV}_prev_code.cu" | cut -d' ' -f1)"; CB="$(sha256sum < "/tmp/${REV}_cur_code.cu" | cut -d' ' -f1)"
  [[ "$CA" == "$CB" ]] && pass "cu_whole_code_region_identical_to_396_r2 (${CB:0:16}...)" || fail "cu_whole_code_region_identical_to_396_r2" "398 must be a rename"
else info "cu_whole_code_region_identical_to_396_r2" "skipped"; fi
[[ "$(grep -c 'MAXD14_PACK_MASK' "$CU_SRC")" == "0" ]] && pass "cu_397_packing_absent" || fail "cu_397_packing_absent" "397's packing is still in the file"
grep -q '    int MAX_BLOCKS = 800;' "$CU_SRC" && pass "cu_binary_default_max_blocks_800" || fail "cu_binary_default_max_blocks_800" "not 800"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; "$NCU" --version 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1

banner "Building $CU_SRC and $PY_SRC"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$CU_BIN" "$CU_SRC" -lcuda 2>&1 | tee "$LOGDIR/05_nvcc.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "no binary"; exit 1; }
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/06_codon.log"
[[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN]" || { fail "codon_build" "see log"; exit 1; }

# ---------------------------------------------------------------------
# 2. ncu permission -- before anything expensive
# ---------------------------------------------------------------------
banner "ncu availability and permission probe (before anything expensive)"
command -v "$NCU" >/dev/null 2>&1 && pass "ncu_present ($("$NCU" --version 2>&1 | head -1))" || { fail "ncu_present" "set NCU=/usr/local/cuda/bin/ncu"; exit 1; }
head -c $((25600*28)) "$IN_PROD" > "/tmp/${REV}_tiny.bin"
ncu_probe() { $1 env NQ_EXTRA_CTX=1 NQ_MAX_BLOCKS="$MB_PROD" "$NCU" --section SpeedOfLight --csv \
    "./$CU_BIN" "$NQ" "/tmp/${REV}_tiny.bin" "/tmp/${REV}_tiny_out.bin" > "$2" 2>&1 || true
  grep -q 'kernel_dfs_iter_gpu_maxd14' "$2"; }
NCU_PREFIX="${NCU_PREFIX-}"
if [[ -n "$NCU_PREFIX" ]]; then
  ncu_probe "$NCU_PREFIX" "$LOGDIR/03_ncu_probe.log" && pass "ncu_permission (via '$NCU_PREFIX')" || { fail "ncu_permission" "NCU_PREFIX='$NCU_PREFIX' cannot read counters"; exit 1; }
elif ncu_probe "" "$LOGDIR/03_ncu_probe.log"; then
  pass "ncu_permission (unprivileged)"
else
  if grep -qi 'ERR_NVGPUCTRPERM\|permission' "$LOGDIR/03_ncu_probe.log"; then
    info "ncu_permission" "admin-restricted unprivileged; retrying under sudo (only the ncu calls are elevated)"
    if ncu_probe "sudo" "$LOGDIR/03_ncu_probe_sudo.log"; then NCU_PREFIX="sudo"; pass "ncu_permission (under sudo; outputs chowned back)"
    else fail "ncu_permission" "sudo did not help. Set it permanently: echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiler.conf && sudo update-initramfs -u && reboot"; exit 1; fi
  else fail "ncu_permission" "ncu ran but did not report the kernel; see $LOGDIR/03_ncu_probe.log"; exit 1; fi
fi
reclaim() { [[ -n "$NCU_PREFIX" ]] && sudo chown -R "$(id -u):$(id -g)" "$LOGDIR" 2>/dev/null; true; }

# ---------------------------------------------------------------------
# 3. W1 anchor -- direct, NQ_EXTRA_CTX=2 (the 397 shortcut)
# ---------------------------------------------------------------------
banner "W1 anchor: ./$CU_BIN N=$NQ direct with NQ_EXTRA_CTX=2"
n="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
[[ "$n" == "0" ]] && pass "gpu_idle_before_anchor" || { fail "gpu_idle_before_anchor" "$n process(es) on the GPU"; exit 1; }
env -u NQ_PAD_MB NQ_EXTRA_CTX=2 NQ_MAX_BLOCKS="$MB_PROD" "./$CU_BIN" "$NQ" "$IN_PROD" "/tmp/${REV}_anchor.bin" "$ORACLE" 2>&1 | tee "$LOGDIR/1_anchor.log"
AK="$(grep -o 'kernel_ms=[0-9.]*' "$LOGDIR/1_anchor.log" | head -1 | cut -d= -f2)"
AT="$(grep -o 'total_sum=[0-9]*' "$LOGDIR/1_anchor.log" | head -1 | cut -d= -f2)"
AF="$(grep -o 'free_mb=[0-9]*' "$LOGDIR/1_anchor.log" | head -1 | cut -d= -f2)"
[[ "${AT:-}" == "$ORACLE" ]] && pass "oracle_match[anchor]" || { fail "oracle_match[anchor]" "total_sum='${AT:-<none>}'"; exit 1; }
d="$(awk -v a="${AK:-0}" -v r="$REF_3CTX_MS" 'BEGIN{x=(a-r)/r*100;printf "%.4f",(x<0?-x:x)}')"
awk -v d="$d" -v t="$TOL_ANCHOR" 'BEGIN{exit !(d<=t)}' && pass "W1_anchor_reproduces_production (${AK}, ${d}%, free_mb=${AF:-?})" \
  || { fail "W1_anchor_reproduces_production" "${AK} is ${d}% off $REF_3CTX_MS -- the session is not in the production state"; exit 1; }

# ---------------------------------------------------------------------
# 4. W2 -- generate and time N=20 and N=19 via the dispatcher
# ---------------------------------------------------------------------
declare -A NIN NMS
for n20 in $PROFILE_NS; do
  banner "W2 generate + time N=$n20 via ./$PY_BIN -g $n20 $n20"
  gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${n20}.log"; rm -f "$gcr"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX "./$PY_BIN" -g "$n20" "$n20" 2>&1 | tee "$LOGDIR/2_gen_N${n20}_console.log"
  cp "$gcr" "$LOGDIR/2_gen_N${n20}_crunner.log" 2>/dev/null || true
  if [[ ! -f "$gcr" ]]; then fail "crunner_path_taken[N=$n20]" "no $gcr -- N=$n20 may not take the maxd14 entry"; continue; fi
  src="$(grep -o 'src=[^ ]*' "$gcr" | head -1 | cut -d= -f2)"
  kms="$(grep -o 'kernel_ms=[0-9.]*' "$gcr" | head -1 | cut -d= -f2)"
  recs="$(grep -o 'records=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"
  if [[ -n "${src:-}" && -f "$src" ]]; then
    NIN[$n20]="$src"; NMS[$n20]="$kms"
    pass "input_located[N=$n20] ($src, $recs records, kernel_ms=$kms)"
  else fail "input_located[N=$n20]" "could not resolve the scheduled input from '${src:-<none>}'"; fi
done
[[ -n "${NIN[20]:-}" ]] || { fail "N20_input_required" "N=20 is the primary profiling target"; exit 1; }

# ---------------------------------------------------------------------
# 5. Profile
# ---------------------------------------------------------------------
SPENT=0
run_ncu() {  # tag  input  args...
  local tag="$1" input="$2"; shift 2
  [[ "$SPENT" -ge "$NCU_BUDGET_S" ]] && { info "budget_exhausted" "skipping $tag"; return 1; }
  local t0 t1 dt rc
  t0=$(date +%s)
  timeout "$SECTION_TIMEOUT_S" $NCU_PREFIX env NQ_EXTRA_CTX=1 NQ_MAX_BLOCKS="$MB_PROD" "$NCU" "$@" --csv \
    "./$CU_BIN" "${input%%::*}" "${input#*::}" "/tmp/${REV}_prof_out.bin" >> "$LOGDIR/5_ncu_${tag}.csv" 2>&1
  rc=$?; t1=$(date +%s); dt=$((t1-t0)); SPENT=$((SPENT+dt)); reclaim
  if [[ "$rc" == "124" ]]; then info "ncu[$tag]" "timed out after ${SECTION_TIMEOUT_S}s"; return 1; fi
  if grep -q '==ERROR==' "$LOGDIR/5_ncu_${tag}.csv"; then info "ncu[$tag]" "ncu reported an error after ${dt}s (see 5_ncu_${tag}.csv)"; return 1; fi
  info "ncu[$tag]" "completed in ${dt}s (spent ${SPENT}s of ${NCU_BUDGET_S}s)"; return 0
}
for n20 in $PROFILE_NS; do
  [[ -z "${NIN[$n20]:-}" ]] && continue
  banner "Profiling N=$n20  (${NMS[$n20]:-?} ms native)"
  IN="${n20}::${NIN[$n20]}"
  run_ncu "N${n20}_warp"  "$IN" --section WarpStateStats                 && pass "ncu_WarpStateStats[N=$n20]"
  run_ncu "N${n20}_local" "$IN" --metrics "$LOCAL_METRICS"               && pass "ncu_local_memory_counters[N=$n20]"
  run_ncu "N${n20}_mem"   "$IN" --section MemoryWorkloadAnalysis         && pass "ncu_MemoryWorkloadAnalysis[N=$n20]"
  run_ncu "N${n20}_sol"   "$IN" --section SpeedOfLight --section Occupancy --section SchedulerStats && pass "ncu_SoL_Occ_Sched[N=$n20]"
done

# ---------------------------------------------------------------------
# 6. Rank and compare
# ---------------------------------------------------------------------
banner "Ranked results"
python3 - "$LOGDIR" "$REF_ACHIEVED_OCC" "$REF_NO_ELIGIBLE" <<'EOF' | tee "$LOGDIR/6_ranked.txt"
import csv, glob, os, re, sys
logdir, ref_occ, ref_noelig = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
def load(path):
    m={}
    hdr=None
    for r in csv.reader(open(path, newline='', errors='replace')):
        if 'Metric Name' in r: hdr=r; continue
        if hdr is None: continue
        try:
            i_n=hdr.index('Metric Name'); i_v=hdr.index('Metric Value'); i_u=hdr.index('Metric Unit')
        except ValueError: continue
        if len(r)<=i_v: continue
        try: v=float(r[i_v].replace(',',''))
        except ValueError: continue
        m[r[i_n].strip()]=(v, r[i_u].strip() if len(r)>i_u else '')
    return m
per_n={}
for p in sorted(glob.glob(os.path.join(logdir,'5_ncu_N*.csv'))):
    n=re.search(r'5_ncu_N(\d+)_',os.path.basename(p)).group(1)
    per_n.setdefault(n,{}).update(load(p))
if not per_n:
    print("no ncu metrics parsed -- inspect 5_ncu_*.csv by hand"); sys.exit(0)
tops={}
for n,m in sorted(per_n.items(), key=lambda x:-int(x[0])):
    print(f"\n================ N={n} ================")
    st=[(k,v) for k,(v,u) in m.items() if 'stalled' in k.lower()]
    if st:
        tot=sum(v for _,v in st); st.sort(key=lambda x:-x[1])
        print("--- W3 warp stall reasons, ranked ---")
        for k,v in st:
            print(f"  {(v/tot*100 if tot else 0):6.2f}%  {v:9.3f}  {re.sub(r'^smsp__average_warps_issue_stalled_','',k)}")
        tops[n]=[re.sub(r'^smsp__average_warps_issue_stalled_','',k) for k,_ in st[:3]]
        lead, lv = st[0][0], st[0][1]/tot*100 if tot else 0
        print(f"  dominant: {re.sub(r'^smsp__average_warps_issue_stalled_','',lead)} at {lv:.2f}%")
        if n=='20':
            if 'branch' in lead.lower() and lv>=25: print("  W3 HELD: branch resolving leads at >=25%. 399 stays in the branch/divergence family.")
            else: print("  W3 FALSIFIED: the bottleneck has MOVED. Read the whole ranking before choosing 399's direction.")
    else:
        print("--- no stall metrics (WarpStateStats did not complete for this N) ---")
    ls=m.get('l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',(0,''))[0]
    lr=m.get('l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum',(0,''))[0]
    ss=m.get('l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',(0,''))[0]
    sr=m.get('l1tex__t_requests_pipe_lsu_mem_local_op_st.sum',(0,''))[0]
    if lr or sr:
        print("--- W4 local memory: sectors per request ---")
        if lr: print(f"  load : {ls:,.0f} sectors / {lr:,.0f} requests = {ls/lr:.3f}")
        if sr: print(f"  store: {ss:,.0f} sectors / {sr:,.0f} requests = {ss/sr:.3f}")
        spr=max(ls/lr if lr else 0, ss/sr if sr else 0)
        if spr>=2:   print(f"  W4 -> (b) DIVERGENCE dominates ({spr:.2f} sectors per request). Byte-count work is not the lever; 399 attacks the access pattern.")
        elif spr<=1.2: print(f"  W4 -> (b) refuted ({spr:.2f}). Coalescing is fine, so VOLUME is what costs -- fitting 98 bits into 96 becomes worth solving.")
        else:        print(f"  W4 -> inconclusive ({spr:.2f}), between the two pre-registered bands.")
    occ=m.get('Achieved Occupancy',(None,''))[0]; ne=m.get('No Eligible',(None,''))[0]
    if occ is not None or ne is not None:
        print("--- W6 representativeness against the measured N=21 values ---")
        if occ is not None:
            print(f"  Achieved Occupancy {occ:.2f}%  vs N=21 {ref_occ:.2f}%  (delta {occ-ref_occ:+.2f} pts, band +-3)")
        if ne is not None:
            print(f"  No Eligible        {ne:.2f}%  vs N=21 {ref_noelig:.2f}%  (delta {ne-ref_noelig:+.2f} pts, band +-5)")
        ok=(occ is None or abs(occ-ref_occ)<=3) and (ne is None or abs(ne-ref_noelig)<=5)
        print("  W6 HELD: N="+n+" is representative on the metrics we can compare directly." if ok
              else "  W6 FAILED: this N does not reproduce N=21's occupancy/scheduler picture; treat its ranking with suspicion.")
if len(tops)>=2:
    ns=sorted(tops, key=lambda x:-int(x))
    print(f"\n--- W5 ranking agreement between N={ns[0]} and N={ns[1]} ---")
    print(f"  N={ns[0]}: {tops[ns[0]]}")
    print(f"  N={ns[1]}: {tops[ns[1]]}")
    print("  W5 HELD: same top three in the same order -> adopt as the answer for N=21."
          if tops[ns[0]]==tops[ns[1]] else
          "  W5 FAILED: the ranking is N-dependent -> N=21 has to be profiled directly, overnight.")
EOF

{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs" 2>/dev/null || true
reclaim
TARBALL="${LOGDIR}.tar.gz"
tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; echo "ranked: $LOGDIR/6_ranked.txt"; echo "tarball: $TARBALL"
if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
echo "${REV} PASSED. Send $TARBALL for analysis."
exit 0
