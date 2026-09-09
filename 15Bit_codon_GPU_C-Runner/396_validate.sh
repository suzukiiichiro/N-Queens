#!/usr/bin/env bash
# 396_validate.sh
#
# rev396 -- re-measure the bottleneck. No functional change; the deliverable
#           is a ranked warp-stall breakdown, not a faster number.
#
# WHY NOW
#   395c closed with production at 133,192.071 ms (2:13.19). The 400 ms it
#   won matters less than what it did to the ruler: the noise floor is now
#   0.009-0.029% within a session and 0.005% across sessions. Every earlier
#   optimisation decision on this kernel was taken with a ruler somewhere
#   between 0.3% and 1.6%. The picture of where the time goes is therefore
#   itself due a re-measurement, before anything else is attempted.
#
# THE TWO PROBLEMS THIS HARNESS SOLVES BEFORE IT PROFILES
#
#   1. IS THE PROXY REPRESENTATIVE?
#      A full N=21 run is 133 s and cannot be replayed under ncu. The
#      profile therefore uses a truncated record set -- but truncating
#      changes k_per_thread (production: 2,025,282 / 25,600 = 79.1 records
#      per thread) and with it the loop trip count and the divergence
#      pattern. So the harness sweeps k = 1,2,4,8,16 WITHOUT ncu and picks
#      the smallest k whose microseconds-per-record match the full run
#      within TOL_PROXY. If none does, it stops: an unrepresentative proxy
#      produces a confident, wrong ranking, which is worse than no ranking.
#
#   2. WHAT WILL THE MEASUREMENT COST?
#      ncu replays the kernel once per pass and the pass count is not
#      predictable from the section list. A one-section pilot runs first,
#      its wall time is measured, and the full run is extrapolated from it.
#      If the projection exceeds NCU_BUDGET_S the harness steps down to a
#      smaller proxy and says so, instead of hanging for an hour.
#
# A CAVEAT RECORDED DELIBERATELY
#   ncu holds its own CUDA context and device memory. 395c established that
#   context count and device occupancy move THIS kernel by 0.3% to 10%, so a
#   profiled run is not in the production memory state. Its wall clock is
#   evidence about nothing. 396 reads per-kernel counters and ratios only,
#   and records ncu's own GPU footprint so that this stays visible.
#
# PRE-REGISTERED (396_README_append.md; written before execution)
#   T1  us/record converges to within +-1% of 65.765 at some k <= 16.
#       Otherwise the proxy fails and no profile is taken.
#   T2  the anchor -g 21 21 lands within +-0.1% of 133,192. If it does not,
#       the session is not in the production state and nothing below is
#       believed.
#   T3  Memory Throughput < 5% of peak. The kernel's entire DRAM traffic is
#       one pass over a 57 MB input in 133 s -- 0.4 MB/s effective. This is
#       the counter-level confirmation that it is not bandwidth-bound.
#   T4  the dominant warp stall reason is branch resolving, at >= 25% of
#       warp stall cycles -- carrying forward what 394a found. Falsified if
#       something else leads: the bottleneck has moved and 397 must change
#       optimisation family. This is the highest-information outcome here.
#
# USAGE
#   STATIC_ONLY=1 bash 396_validate.sh
#                 bash 396_validate.sh          # ~20-30 min
#   SKIP_ANCHOR=1 bash 396_validate.sh          # skip the 2.5 min anchor

set -u

REV="396"
PY_SRC="${PY_SRC:-396Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-396Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-395c_r6Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-396_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-396_kernel_maxd14}"
PREV_CU="${PREV_CU:-395c_r6_kernel_maxd14.cu}"
CRLOG_DIR="${CRLOG_DIR:-396_crunner_logs}"
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
STRIDE=$((32*MB_PROD))
REF_PROD_MS="${REF_PROD_MS:-133192.071}"
TOL_ANCHOR="${TOL_ANCHOR:-0.1}"
TOL_PROXY="${TOL_PROXY:-1.0}"
K_LIST="${K_LIST:-1 2 4 8 16}"
NCU_BUDGET_S="${NCU_BUDGET_S:-900}"
KERNEL_SHA_395C="${KERNEL_SHA_395C:-ebd7f523deb591347eab1a352a9555c2e4a6c0b4a456d80517322c933a86e68a}"
SKIP_ANCHOR="${SKIP_ANCHOR:-0}"
STATIC_ONLY="${STATIC_ONLY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"
TSV="$LOGDIR/${REV}_results.tsv"

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
if [[ -f "$IN_PROD" ]]; then
  sz="$(stat -c%s "$IN_PROD")"
  [[ "$sz" -eq $((EXPECTED_RECORDS*28)) ]] && pass "sched_input_present_and_sized[$((sz/28)) records]" || fail "sched_input_present_and_sized" "$IN_PROD is $sz bytes"
else fail "sched_input_present_and_sized" "$IN_PROD missing"; fi
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }

py_code_region() {
  python3 -c "
import sys
lines=open(sys.argv[1],encoding='utf-8').read().split('\n')
i=next(i for i,l in enumerate(lines) if l.startswith('import '))
sys.stdout.write('\n'.join(l for l in lines[i:] if not l.lstrip().startswith('#')))
" "$1"
}
py_note_quotes() {
  python3 -c "
import sys
lines=open(sys.argv[1],encoding='utf-8').read().split('\n')
i=next(i for i,l in enumerate(lines) if l.startswith('import '))
print('\n'.join(lines[:i]).count('\"'*3))
" "$1"
}
py_code_region "$PY_SRC" > "/tmp/${REV}_code_only.py"; CODE="/tmp/${REV}_code_only.py"
NOTE_LINES=$(awk '/^# =+$/{f=1} f&&/^#/{n++} END{print n+0}' "$PY_SRC")
if grep -q "^# ${REV} " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; then
  pass "py_revision_notes_present ($NOTE_LINES comment lines; the ${REV} record is in the source)"
else
  fail "py_revision_notes_present" "no '# ${REV} ...' note block (>=20 comment lines) in $PY_SRC"
fi
NQ_=$(py_note_quotes "$PY_SRC"); [[ $((NQ_ % 2)) -eq 0 ]] && pass "py_note_region_quotes_balanced ($NQ_)" || fail "py_note_region_quotes_balanced" "$NQ_ triple-quotes: a note slot does not close"
grep -qE '^REV_TAG:str="396"' "$CODE" && pass "source_rev_tag_is_396" || fail "source_rev_tag_is_396" "wrong REV_TAG"
grep -q 'CRunnerEntry(14,"./396_kernel_maxd14","NQ_EXTRA_CTX=1 "' "$CODE" && pass "source_table_keeps_the_r6_treatment (env_prefix NQ_EXTRA_CTX=1 retained)" || fail "source_table_keeps_the_r6_treatment" "the adopted treatment was dropped"
grep -qE '^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=800' "$CODE" && pass "source_default_max_blocks_800" || fail "source_default_max_blocks_800" "not 800"
grep -qE '^CRUNNER_INPUT_ORDER:str="sched"' "$CODE" && pass "source_input_order_sched" || fail "source_input_order_sched" "not sched"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  if [[ "$PR_" == "3" && "$PA_" == "3" ]]; then pass "py_diff_fingerprint_vs_r6Py (removed=3 added=3 EXECUTABLE lines; comments and notes are not counted)"
  else fail "py_diff_fingerprint_vs_r6Py" "removed=$PR_ added=$PA_, expected 3/3"
       diff "/tmp/${REV}_prev_code.py" "$CODE" | head -40 | sed 's/^/      /' | cut -c1-140; fi
else info "py_diff_fingerprint_vs_r6Py" "skipped"; fi

cu_code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA_395C" ]] && pass "cu_kernel_region_sha_unchanged_since_395a_r2 (${KB:0:16}...)" || fail "cu_kernel_region_sha_unchanged_since_395a_r2" "got ${KB:0:16}..."
if [[ -f "$PREV_CU" ]]; then
  cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  CA="$(sha256sum < "/tmp/${REV}_prev_code.cu" | cut -d' ' -f1)"; CB="$(sha256sum < "/tmp/${REV}_cur_code.cu" | cut -d' ' -f1)"
  [[ "$CA" == "$CB" ]] && pass "cu_whole_code_region_identical_to_r6 (${CB:0:16}...)" || fail "cu_whole_code_region_identical_to_r6" "396 must be a rename"
else info "cu_whole_code_region_identical_to_r6" "skipped"; fi
grep -q '    int MAX_BLOCKS = 800;' "$CU_SRC" && pass "cu_binary_default_max_blocks_800" || fail "cu_binary_default_max_blocks_800" "not 800"
grep -q 'RISK NOTE' "$CU_SRC" && grep -q 'if (cur_avail != 0u) {' "$CU_SRC" && pass "cu_push_guard_untouched" || fail "cu_push_guard_untouched" "the push guard form changed"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; "$NCU" --version 2>&1; sha256sum "$PY_SRC" "$CU_SRC" "$IN_PROD" 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1

# ---------------------------------------------------------------------
# 2. Builds
# ---------------------------------------------------------------------
banner "Building $CU_SRC (with -lcuda) and $PY_SRC"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$CU_BIN" "$CU_SRC" -lcuda 2>&1 | tee "$LOGDIR/05_nvcc_build.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build_succeeded[$CU_BIN] (-lineinfo added so ncu can attribute stalls to source lines)" || { fail "nvcc_build_succeeded" "no binary"; exit 1; }
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/06_codon_build.log"
[[ -x "$PY_BIN" ]] && pass "codon_build_succeeded[$PY_BIN]" || { fail "codon_build_succeeded" "see log"; exit 1; }

# ---------------------------------------------------------------------
# 3. T2 -- anchor on the adopted production number
# ---------------------------------------------------------------------
if [[ "$SKIP_ANCHOR" == "0" ]]; then
  banner "T2 anchor: ./$PY_BIN -g $NQ $NQ  (must reproduce $REF_PROD_MS)"
  n="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
  [[ "$n" == "0" ]] && pass "gpu_idle_before_anchor" || { fail "gpu_idle_before_anchor" "$n process(es) already on the GPU"; exit 1; }
  gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${NQ}.log"; rm -f "$gcr"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX "./$PY_BIN" -g "$NQ" "$NQ" 2>&1 | tee "$LOGDIR/1_anchor_console.log"
  cp "$gcr" "$LOGDIR/1_anchor_crunner.log" 2>/dev/null || true
  AK="$(grep -o 'kernel_ms=[0-9.]*' "$gcr" 2>/dev/null | head -1 | cut -d= -f2)"
  AT="$(grep -o 'total_sum=[0-9]*' "$gcr" 2>/dev/null | head -1 | cut -d= -f2)"
  AX="$(grep -o 'extra_ctx=[0-9]*' "$gcr" 2>/dev/null | head -1 | cut -d= -f2)"
  [[ "${AT:-}" == "$ORACLE" ]] && pass "oracle_match[anchor]" || { fail "oracle_match[anchor]" "total_sum='${AT:-<none>}'"; exit 1; }
  [[ "${AX:-}" == "1" ]] && pass "anchor_extra_ctx_is_1 (the r6 treatment is live)" || fail "anchor_extra_ctx_is_1" "extra_ctx=${AX:-?}"
  d="$(awk -v a="${AK:-0}" -v r="$REF_PROD_MS" 'BEGIN{x=(a-r)/r*100; printf "%.4f",(x<0?-x:x)}')"
  awk -v d="$d" -v t="$TOL_ANCHOR" 'BEGIN{exit !(d<=t)}' \
    && pass "T2_anchor_reproduces_production (${AK}, ${d}%)" \
    || { fail "T2_anchor_reproduces_production" "${AK} is ${d}% off $REF_PROD_MS -- the session is not in the production state; the profile below would describe a different machine"; exit 1; }
else info "T2_anchor" "skipped by SKIP_ANCHOR=1 -- the profile is then unanchored"; fi

# ---------------------------------------------------------------------
# 4. T1 -- calibrate the proxy WITHOUT ncu
# ---------------------------------------------------------------------
banner "T1 proxy calibration: sweep k, match us/record against the full run"
REF_US_PER_REC="$(awk -v ms="$REF_PROD_MS" -v n="$EXPECTED_RECORDS" 'BEGIN{printf "%.4f", ms*1000.0/n}')"
info "reference" "$REF_PROD_MS ms / $EXPECTED_RECORDS records = $REF_US_PER_REC us/record (k_per_thread = $(awk -v n="$EXPECTED_RECORDS" -v s="$STRIDE" 'BEGIN{printf "%.1f",n/s}'))"
printf 'k\trecords\tkernel_ms\tus_per_record\tdev_pct\n' > "$LOGDIR/proxy_calibration.tsv"
K_OK=""; K_OK_MS=""
for k in $K_LIST; do
  recs=$((STRIDE*k))
  [[ "$recs" -gt "$EXPECTED_RECORDS" ]] && { info "proxy[k=$k]" "skipped: $recs records exceeds the input"; continue; }
  head -c $((recs*28)) "$IN_PROD" > "/tmp/${REV}_proxy_k${k}.bin"
  NQ_EXTRA_CTX=1 NQ_MAX_BLOCKS="$MB_PROD" "./$CU_BIN" "$NQ" "/tmp/${REV}_proxy_k${k}.bin" "/tmp/${REV}_proxy_out.bin" > "$LOGDIR/2_proxy_k${k}.log" 2>&1 || true
  ms="$(grep -o 'kernel_ms=[0-9.]*' "$LOGDIR/2_proxy_k${k}.log" | head -1 | cut -d= -f2)"
  [[ -z "${ms:-}" ]] && { fail "proxy_run[k=$k]" "no kernel_ms; see $LOGDIR/2_proxy_k${k}.log"; continue; }
  upr="$(awk -v ms="$ms" -v n="$recs" 'BEGIN{printf "%.4f", ms*1000.0/n}')"
  dev="$(awk -v a="$upr" -v r="$REF_US_PER_REC" 'BEGIN{x=(a-r)/r*100; printf "%.3f",x}')"
  printf '%s\t%s\t%s\t%s\t%s\n' "$k" "$recs" "$ms" "$upr" "$dev" >> "$LOGDIR/proxy_calibration.tsv"
  info "proxy[k=$k]" "records=$recs kernel_ms=$ms  ${upr} us/record  dev=${dev}%"
  if [[ -z "$K_OK" ]] && awk -v d="$dev" -v t="$TOL_PROXY" 'BEGIN{x=(d<0?-d:d); exit !(x<=t)}'; then K_OK="$k"; K_OK_MS="$ms"; fi
done
if [[ -n "$K_OK" ]]; then
  pass "T1_proxy_calibrated (k=$K_OK, $((STRIDE*K_OK)) records, ${K_OK_MS} ms, within ${TOL_PROXY}% of the full run per record)"
else
  fail "T1_proxy_calibrated" "no k in [$K_LIST] reproduced $REF_US_PER_REC us/record within ${TOL_PROXY}% -- truncation is not a valid reduction for this kernel. Do NOT profile a proxy that failed this; design a different reduction (e.g. a stratified record subset) in 396-r2."
  echo; cat "$LOGDIR/proxy_calibration.tsv"; echo "OK=$PASS FAIL=$FAIL"; exit 1
fi
PROXY="/tmp/${REV}_proxy_k${K_OK}.bin"

# ---------------------------------------------------------------------
# 5. ncu availability and permission
# ---------------------------------------------------------------------
banner "ncu availability and permission probe"
command -v "$NCU" >/dev/null 2>&1 && pass "ncu_present ($("$NCU" --version 2>&1 | head -1))" \
  || { fail "ncu_present" "ncu not on PATH -- set NCU=/usr/local/cuda/bin/ncu"; exit 1; }
head -c $((STRIDE*28)) "$IN_PROD" > "/tmp/${REV}_tiny.bin"
NQ_EXTRA_CTX=1 NQ_MAX_BLOCKS="$MB_PROD" "$NCU" --section SpeedOfLight --csv \
  "./$CU_BIN" "$NQ" "/tmp/${REV}_tiny.bin" "/tmp/${REV}_tiny_out.bin" > "$LOGDIR/3_ncu_permission_probe.log" 2>&1 || true
if grep -qi 'ERR_NVGPUCTRPERM\|permission' "$LOGDIR/3_ncu_permission_probe.log"; then
  fail "ncu_permission" "profiling counters are admin-restricted. Either run this harness under sudo, or set the driver option permanently: echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiler.conf, then reboot. See $LOGDIR/3_ncu_permission_probe.log"
  exit 1
fi
grep -q 'kernel_dfs_iter_gpu_maxd14' "$LOGDIR/3_ncu_permission_probe.log" && pass "ncu_permission (counters readable, the kernel was seen)" \
  || { fail "ncu_permission" "ncu ran but did not report the kernel; see $LOGDIR/3_ncu_permission_probe.log"; exit 1; }

# ---------------------------------------------------------------------
# 6. Cost pilot -- measure the measurement before committing to it
# ---------------------------------------------------------------------
banner "Cost pilot: one section on the calibrated proxy, then extrapolate"
SECTIONS="SpeedOfLight WarpStateStats SchedulerStats InstructionStats ComputeWorkloadAnalysis MemoryWorkloadAnalysis Occupancy"
NSEC=$(echo $SECTIONS | wc -w)
t0=$(date +%s)
NQ_EXTRA_CTX=1 NQ_MAX_BLOCKS="$MB_PROD" "$NCU" --section SpeedOfLight --csv \
  "./$CU_BIN" "$NQ" "$PROXY" "/tmp/${REV}_pilot_out.bin" > "$LOGDIR/4_ncu_pilot.log" 2>&1 || true
t1=$(date +%s); PILOT=$((t1-t0))
PROJ=$((PILOT*NSEC))
info "cost_pilot" "one section on k=$K_OK took ${PILOT}s; $NSEC sections project to ~${PROJ}s (budget ${NCU_BUDGET_S}s)"
while [[ "$PROJ" -gt "$NCU_BUDGET_S" && "$K_OK" -gt 1 ]]; do
  K_OK=$((K_OK/2)); PROXY="/tmp/${REV}_proxy_k${K_OK}.bin"; PILOT=$((PILOT/2)); PROJ=$((PILOT*NSEC))
  info "cost_stepdown" "projection exceeded the budget; stepping the proxy down to k=$K_OK (~${PROJ}s). NOTE: this k may sit outside the ${TOL_PROXY}% calibration band -- check proxy_calibration.tsv before trusting ratios that depend on loop trip count."
done
[[ -f "$PROXY" ]] || { fail "proxy_available_after_stepdown" "$PROXY missing"; exit 1; }
pass "cost_pilot_within_budget (profiling k=$K_OK, projected ~${PROJ}s)"

# ---------------------------------------------------------------------
# 7. The profile
# ---------------------------------------------------------------------
banner "Profiling k=$K_OK with $NSEC sections"
SECARGS=""; for s in $SECTIONS; do SECARGS="$SECARGS --section $s"; done
( while true; do echo "$(date +%H:%M:%S), $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')" >> "$LOGDIR/apps_during_ncu.tsv"; sleep 5; done ) &
APPS_PID=$!
NQ_EXTRA_CTX=1 NQ_MAX_BLOCKS="$MB_PROD" "$NCU" $SECARGS --csv -f -o "$LOGDIR/${REV}_profile" \
  "./$CU_BIN" "$NQ" "$PROXY" "/tmp/${REV}_prof_out.bin" > "$LOGDIR/5_ncu_full.csv" 2>&1 || true
kill "$APPS_PID" 2>/dev/null; wait "$APPS_PID" 2>/dev/null || true
[[ -s "$LOGDIR/5_ncu_full.csv" ]] && pass "ncu_profile_produced" || { fail "ncu_profile_produced" "empty output"; exit 1; }
info "ncu_own_footprint" "$(tail -1 "$LOGDIR/apps_during_ncu.tsv" 2>/dev/null || echo 'not sampled') -- recorded because context count and occupancy move this kernel by 0.3-10%; the wall clock under ncu is not comparable with production"

# ---------------------------------------------------------------------
# 8. Extract and rank
# ---------------------------------------------------------------------
banner "Ranked results"
python3 - "$LOGDIR/5_ncu_full.csv" "$LOGDIR" <<'EOF' | tee "$LOGDIR/6_ranked.txt"
import csv, sys, re
path, logdir = sys.argv[1], sys.argv[2]
rows=[]
with open(path, newline='', errors='replace') as f:
    for r in csv.reader(f):
        if len(r) >= 2: rows.append(r)
hdr=None
for r in rows:
    if 'Metric Name' in r: hdr=r; break
if hdr is None:
    print("could not find a CSV header in the ncu output; inspect 5_ncu_full.csv by hand"); sys.exit(0)
iname=hdr.index('Metric Name'); iunit=hdr.index('Metric Unit') if 'Metric Unit' in hdr else None
ival=hdr.index('Metric Value')
metrics={}
for r in rows:
    if r is hdr or len(r)<=max(iname,ival): continue
    n=r[iname].strip()
    try: v=float(r[ival].replace(',',''))
    except ValueError: continue
    u=r[iunit].strip() if iunit is not None and len(r)>iunit else ''
    metrics[n]=(v,u)

def show(title, pred):
    sel=[(n,v,u) for n,(v,u) in metrics.items() if pred(n)]
    if not sel: return
    sel.sort(key=lambda x:-x[1])
    print(f"\n--- {title} ---")
    for n,v,u in sel[:20]: print(f"  {v:12,.3f} {u:<12} {n}")

stalls=[(n,v,u) for n,(v,u) in metrics.items() if 'stalled' in n.lower()]
if stalls:
    tot=sum(v for _,v,_ in stalls)
    stalls.sort(key=lambda x:-x[1])
    print("\n=== T4: warp stall reasons, ranked ===")
    for n,v,u in stalls:
        share = (v/tot*100) if tot else 0
        short = re.sub(r'^smsp__average_warps_issue_stalled_','',n)
        print(f"  {share:6.2f}%  {v:10.3f} {u:<10} {short}")
    top = stalls[0]
    tshare = top[1]/tot*100 if tot else 0
    print(f"\n  dominant: {re.sub(r'^smsp__average_warps_issue_stalled_','',top[0])} at {tshare:.2f}% of warp stall cycles")
    if 'branch' in top[0].lower() and tshare>=25:
        print("  T4 HELD: branch resolving still leads at >=25%. 397 stays in the branch/divergence family.")
    else:
        print("  T4 FALSIFIED: the bottleneck has MOVED. 397 must change optimisation family -- this is the")
        print("  most informative outcome available from this revision; read the full ranking above before choosing.")
else:
    print("no stall metrics found -- WarpStateStats may not have been collected; check 5_ncu_full.csv")

show("Speed of Light / utilisation", lambda n: 'sol' in n.lower() or 'throughput' in n.lower())
show("Occupancy", lambda n: 'occupancy' in n.lower())
show("Issue / instruction", lambda n: 'inst_executed' in n or 'issue' in n.lower() or 'ipc' in n.lower())
show("Branch", lambda n: 'branch' in n.lower())
show("DRAM / memory", lambda n: n.startswith('dram__') or 'l1tex' in n or 'lts__' in n)
EOF

{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs" 2>/dev/null || true
echo; cat "$LOGDIR/proxy_calibration.tsv"
TARBALL="${LOGDIR}.tar.gz"
tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed -- send $LOGDIR/"
echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; echo "ranked: $LOGDIR/6_ranked.txt"; echo "tarball: $TARBALL"
if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
echo "${REV} PASSED. Send $TARBALL for analysis."
exit 0
