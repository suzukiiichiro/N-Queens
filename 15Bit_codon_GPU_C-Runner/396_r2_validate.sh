#!/usr/bin/env bash
# 396_r2_validate.sh
#
# rev396-r2 -- 396's proxy failed its own calibration gate. Fix the proxy,
#              and make sure the stall ranking gets produced either way.
#
# WHAT 396 FOUND (and why that was the gate working, not the harness failing)
#   Truncating the scheduled input to its first k rounds gave, against the
#   full run's 65.7647 us/record:
#     k=1  57.467 (-12.6%)   k=2  51.387 (-21.9%)   k=4  48.898 (-25.6%)
#     k=8  55.263 (-16.0%)   k=16 51.910 (-21.1%)
#   Too cheap, and non-monotonic. Differencing the prefix sums says why:
#     round 0      1471.2 ms       rounds 4-7    1577.7 ms/round
#     round 1      1159.9 ms       rounds 8-15   1243.1 ms/round
#     rounds 2-3   1188.1 ms       rounds 16-79  1773.5 ms/round
#     global mean  1683.6 ms/round
#   The 394f schedule orders records, so the head of the file is its cheap
#   part and the tail carries the cost. A prefix is not a sample of a sorted
#   file. Profiling that proxy would have produced a confident wrong ranking.
#
# WHAT r2 DOES INSTEAD
#   STRATIFIED ROUND SAMPLING. A round is exactly 25,600 records -- one per
#   thread at stride 25,600 -- so lifting whole rounds preserves the
#   record-to-thread mapping, and with it the within-warp divergence pattern,
#   exactly. Only the trip count changes. r2 takes k rounds spread evenly
#   across all 79 instead of the first k.
#
#   And a FALLBACK. If stratified sampling also fails calibration, the
#   harness profiles the REAL full input, one section per invocation, under
#   a time budget, starting with WarpStateStats. The deliverable of 396 is
#   the stall ranking; a proxy-design problem should not be allowed to
#   withhold it. Whatever completes inside the budget is reported.
#
# PHASES
#   0  static gates, builds
#   1  anchor -g 21 21                        ~2.5 min
#   2  per-round cost probe, rounds 0..70     ~30 s   (U2: diagnosis)
#   3  stratified calibration, k=2,4,8,16     ~2 min  (U1)
#   4a calibrated -> ncu on the proxy
#   4b not calibrated -> ncu on the full input, one section at a time
#
# PRE-REGISTERED (396_r2_README_append.md; written before execution)
#   U1  stratified k in {4,8,16} lands within +-1% of 65.7647 us/record.
#       From the round-cost model above the predictions are k=4 66.33
#       (+0.85%), k=8 65.21 (-0.84%), k=16 65.47 (-0.45%), and k=2 misses
#       at -3.64%. The model treats rounds 16-79 as uniform, so a miss is
#       evidence about the model's coarseness, not necessarily about the
#       method -- which is why phase 2 measures the profile directly.
#   U2  single-round cost is not monotonic in position, and rounds >= 16
#       are consistently dearer than rounds < 16.
#   U3  the anchor reproduces 133,192.071 within +-0.1%, extra_ctx=1.
#   U4  Memory Throughput < 5% of peak.
#   U5  the dominant warp stall reason is branch resolving at >= 25% of
#       warp stall cycles. Falsified -> the bottleneck has moved and 397
#       changes optimisation family. This is the point of the revision.
#
# USAGE
#   STATIC_ONLY=1 bash 396_r2_validate.sh
#                 bash 396_r2_validate.sh
#   FORCE_FULL=1  bash 396_r2_validate.sh   # skip the proxy, profile the real input
#   SKIP_ANCHOR=1 bash 396_r2_validate.sh
#   NCU_PREFIX=sudo bash 396_r2_validate.sh  # force elevation for the ncu calls only
#
# NOTE ON PRIVILEGE: do NOT run this whole harness under sudo. It would leave
# the log directory, the binaries, 396_r2_crunner_logs/ and the tarball owned
# by root and break the next unprivileged run. Only the ncu invocations are
# elevated, and their outputs are chowned back. The permission probe now runs
# straight after the builds, so a privilege problem costs three seconds
# instead of the five minutes it cost in the first attempt.

set -u

REV="396_r2"
PY_SRC="${PY_SRC:-396_r2Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-396_r2Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-396Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-396_r2_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-396_r2_kernel_maxd14}"
PREV_CU="${PREV_CU:-396_kernel_maxd14.cu}"
CRLOG_DIR="${CRLOG_DIR:-396_r2_crunner_logs}"
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
K_LIST="${K_LIST:-2 4 8 16}"
PROBE_ROUNDS="${PROBE_ROUNDS:-0 10 20 30 40 50 60 70}"
NCU_BUDGET_S="${NCU_BUDGET_S:-2400}"
SECTION_TIMEOUT_S="${SECTION_TIMEOUT_S:-900}"
KERNEL_SHA_395C="${KERNEL_SHA_395C:-ebd7f523deb591347eab1a352a9555c2e4a6c0b4a456d80517322c933a86e68a}"
SKIP_ANCHOR="${SKIP_ANCHOR:-0}"
FORCE_FULL="${FORCE_FULL:-0}"
STATIC_ONLY="${STATIC_ONLY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }

# ---------------------------------------------------------------------
# 0. Static
# ---------------------------------------------------------------------
for f in "$PY_SRC" "$HELPER_SRC" "$CU_SRC" "$IN_RAW"; do
  [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"
done
if [[ -f "$IN_PROD" ]]; then
  sz="$(stat -c%s "$IN_PROD")"
  [[ "$sz" -eq $((EXPECTED_RECORDS*28)) ]] && pass "sched_input_present_and_sized[$((sz/28)) records]" || fail "sched_input_present_and_sized" "$IN_PROD is $sz bytes"
else fail "sched_input_present_and_sized" "$IN_PROD missing"; fi
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
if grep -q "^# ${REV//_/-} " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; then
  pass "py_revision_notes_present ($NOTE_LINES comment lines)"
else fail "py_revision_notes_present" "no '# ${REV//_/-} ...' note block (>=20 comment lines) in $PY_SRC"; fi
NQ_=$(py_note_quotes "$PY_SRC"); [[ $((NQ_ % 2)) -eq 0 ]] && pass "py_note_region_quotes_balanced ($NQ_)" || fail "py_note_region_quotes_balanced" "$NQ_ triple-quotes"
grep -qE '^REV_TAG:str="396_r2"' "$CODE" && pass "source_rev_tag_is_396_r2" || fail "source_rev_tag_is_396_r2" "wrong REV_TAG"
grep -q 'CRunnerEntry(14,"./396_r2_kernel_maxd14","NQ_EXTRA_CTX=1 "' "$CODE" && pass "source_table_keeps_the_r6_treatment" || fail "source_table_keeps_the_r6_treatment" "the adopted treatment or the binary name is wrong"
grep -qE '^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=800' "$CODE" && pass "source_default_max_blocks_800" || fail "source_default_max_blocks_800" "not 800"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_396Py (removed=3 added=3 EXECUTABLE lines)" \
    || { fail "py_diff_fingerprint_vs_396Py" "removed=$PR_ added=$PA_, expected 3/3"; diff "/tmp/${REV}_prev_code.py" "$CODE" | head -30 | sed 's/^/      /' | cut -c1-140; }
else info "py_diff_fingerprint_vs_396Py" "skipped"; fi
cu_code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA_395C" ]] && pass "cu_kernel_region_sha_unchanged_since_395a_r2 (${KB:0:16}...)" || fail "cu_kernel_region_sha_unchanged_since_395a_r2" "got ${KB:0:16}..."
if [[ -f "$PREV_CU" ]]; then
  cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  CA="$(sha256sum < "/tmp/${REV}_prev_code.cu" | cut -d' ' -f1)"; CB="$(sha256sum < "/tmp/${REV}_cur_code.cu" | cut -d' ' -f1)"
  [[ "$CA" == "$CB" ]] && pass "cu_whole_code_region_identical_to_396 (${CB:0:16}...)" || fail "cu_whole_code_region_identical_to_396" "r2 must be a rename"
else info "cu_whole_code_region_identical_to_396" "skipped"; fi
grep -q '    int MAX_BLOCKS = 800;' "$CU_SRC" && pass "cu_binary_default_max_blocks_800" || fail "cu_binary_default_max_blocks_800" "not 800"
grep -q 'RISK NOTE' "$CU_SRC" && grep -q 'if (cur_avail != 0u) {' "$CU_SRC" && pass "cu_push_guard_untouched" || fail "cu_push_guard_untouched" "the push guard form changed"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; "$NCU" --version 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1

banner "Building $CU_SRC and $PY_SRC"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$CU_BIN" "$CU_SRC" -lcuda 2>&1 | tee "$LOGDIR/05_nvcc_build.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build_succeeded[$CU_BIN]" || { fail "nvcc_build_succeeded" "no binary"; exit 1; }
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/06_codon_build.log"
[[ -x "$PY_BIN" ]] && pass "codon_build_succeeded[$PY_BIN]" || { fail "codon_build_succeeded" "see log"; exit 1; }

REF_US="$(awk -v ms="$REF_PROD_MS" -v n="$EXPECTED_RECORDS" 'BEGIN{printf "%.4f", ms*1000.0/n}')"
TOTAL_ROUNDS=$((EXPECTED_RECORDS/STRIDE))
info "reference" "$REF_US us/record; $TOTAL_ROUNDS whole rounds of $STRIDE records (+$((EXPECTED_RECORDS-TOTAL_ROUNDS*STRIDE)) leftover)"

# rounds -> file. Lifting WHOLE rounds keeps record->thread mapping intact.
make_rounds() {  # $1=out  $2...=round indices
  local out="$1"; shift
  python3 - "$IN_PROD" "$out" "$STRIDE" "$@" <<'EOF'
import sys
src,dst,stride = sys.argv[1], sys.argv[2], int(sys.argv[3])
rounds=[int(x) for x in sys.argv[4:]]
rec=28
with open(src,'rb') as f, open(dst,'wb') as g:
    for r in rounds:
        f.seek(r*stride*rec)
        buf=f.read(stride*rec)
        if len(buf)!=stride*rec: raise SystemExit(f"short read at round {r}")
        g.write(buf)
EOF
}
run_proxy() {  # $1=bin-in  $2=tag -> echoes kernel_ms
  local pin="$1" tag="$2"
  NQ_EXTRA_CTX=1 NQ_MAX_BLOCKS="$MB_PROD" "./$CU_BIN" "$NQ" "$pin" "/tmp/${REV}_out.bin" > "$LOGDIR/$tag.log" 2>&1 || true
  grep -o 'kernel_ms=[0-9.]*' "$LOGDIR/$tag.log" | head -1 | cut -d= -f2
}

banner "ncu availability and permission probe (before anything expensive)"
command -v "$NCU" >/dev/null 2>&1 && pass "ncu_present ($("$NCU" --version 2>&1 | head -1))" || { fail "ncu_present" "set NCU=/usr/local/cuda/bin/ncu"; exit 1; }
mkdir -p "$LOGDIR"
make_rounds "/tmp/${REV}_tiny.bin" 0
ncu_probe() {  # $1 = prefix ("" or "sudo"), $2 = log
  $1 env NQ_EXTRA_CTX=1 NQ_MAX_BLOCKS="$MB_PROD" "$NCU" --section SpeedOfLight --csv \
    "./$CU_BIN" "$NQ" "/tmp/${REV}_tiny.bin" "/tmp/${REV}_tiny_out.bin" > "$2" 2>&1 || true
  grep -q 'kernel_dfs_iter_gpu_maxd14' "$2"
}
NCU_PREFIX="${NCU_PREFIX-}"
if [[ -n "$NCU_PREFIX" ]]; then
  ncu_probe "$NCU_PREFIX" "$LOGDIR/4_ncu_permission_probe.log" \
    && pass "ncu_permission (counters readable via '$NCU_PREFIX', as requested)" \
    || { fail "ncu_permission" "NCU_PREFIX='$NCU_PREFIX' still cannot read counters; see $LOGDIR/4_ncu_permission_probe.log"; exit 1; }
elif ncu_probe "" "$LOGDIR/4_ncu_permission_probe.log"; then
  pass "ncu_permission (counters readable unprivileged)"
else
  if grep -qi 'ERR_NVGPUCTRPERM\|permission' "$LOGDIR/4_ncu_permission_probe.log"; then
    info "ncu_permission" "counters are admin-restricted unprivileged; retrying the probe under sudo (you may be prompted). ONLY the ncu calls are elevated -- the anchor, the proxy runs and every file this harness writes stay under your own user."
    if ncu_probe "sudo" "$LOGDIR/4_ncu_permission_probe_sudo.log"; then
      NCU_PREFIX="sudo"
      pass "ncu_permission (counters readable under sudo; ncu calls will be elevated, outputs chowned back)"
    else
      fail "ncu_permission" "sudo did not help either. Set it permanently instead: echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiler.conf && sudo update-initramfs -u && reboot. Logs: $LOGDIR/4_ncu_permission_probe*.log"
      exit 1
    fi
  else
    fail "ncu_permission" "ncu ran but did not report the kernel; see $LOGDIR/4_ncu_permission_probe.log"
    exit 1
  fi
fi
reclaim() { [[ -n "$NCU_PREFIX" ]] && sudo chown -R "$(id -u):$(id -g)" "$LOGDIR" 2>/dev/null; sudo chown "$(id -u):$(id -g)" /tmp/${REV}_*.bin 2>/dev/null; true; }

# ---------------------------------------------------------------------
# 1. Anchor
# ---------------------------------------------------------------------
if [[ "$SKIP_ANCHOR" == "0" ]]; then
  banner "U3 anchor: ./$PY_BIN -g $NQ $NQ  (must reproduce $REF_PROD_MS)"
  n="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
  [[ "$n" == "0" ]] && pass "gpu_idle_before_anchor" || { fail "gpu_idle_before_anchor" "$n process(es) on the GPU"; exit 1; }
  gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${NQ}.log"; rm -f "$gcr"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX "./$PY_BIN" -g "$NQ" "$NQ" 2>&1 | tee "$LOGDIR/1_anchor_console.log"
  cp "$gcr" "$LOGDIR/1_anchor_crunner.log" 2>/dev/null || true
  AK="$(grep -o 'kernel_ms=[0-9.]*' "$gcr" 2>/dev/null | head -1 | cut -d= -f2)"
  AT="$(grep -o 'total_sum=[0-9]*' "$gcr" 2>/dev/null | head -1 | cut -d= -f2)"
  AX="$(grep -o 'extra_ctx=[0-9]*' "$gcr" 2>/dev/null | head -1 | cut -d= -f2)"
  [[ "${AT:-}" == "$ORACLE" ]] && pass "oracle_match[anchor]" || { fail "oracle_match[anchor]" "total_sum='${AT:-<none>}'"; exit 1; }
  [[ "${AX:-}" == "1" ]] && pass "anchor_extra_ctx_is_1" || fail "anchor_extra_ctx_is_1" "extra_ctx=${AX:-?}"
  d="$(awk -v a="${AK:-0}" -v r="$REF_PROD_MS" 'BEGIN{x=(a-r)/r*100; printf "%.4f",(x<0?-x:x)}')"
  awk -v d="$d" -v t="$TOL_ANCHOR" 'BEGIN{exit !(d<=t)}' && pass "U3_anchor_reproduces_production (${AK}, ${d}%)" \
    || { fail "U3_anchor_reproduces_production" "${AK} is ${d}% off $REF_PROD_MS"; exit 1; }
else info "U3_anchor" "skipped by SKIP_ANCHOR=1"; fi


# ---------------------------------------------------------------------
# 1. Anchor
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2. U2 -- per-round cost probe (the diagnosis 396 could only infer)
# ---------------------------------------------------------------------
banner "U2 per-round cost probe: one round at positions $PROBE_ROUNDS"
printf 'round\tkernel_ms\tus_per_record\n' > "$LOGDIR/round_profile.tsv"
for r in $PROBE_ROUNDS; do
  [[ "$r" -ge "$TOTAL_ROUNDS" ]] && continue
  make_rounds "/tmp/${REV}_round_${r}.bin" "$r" || { fail "round_extract[$r]" "extraction failed"; continue; }
  ms="$(run_proxy "/tmp/${REV}_round_${r}.bin" "2_round_${r}")"
  [[ -z "${ms:-}" ]] && { fail "round_probe[$r]" "no kernel_ms"; continue; }
  upr="$(awk -v ms="$ms" -v n="$STRIDE" 'BEGIN{printf "%.4f", ms*1000.0/n}')"
  printf '%s\t%s\t%s\n' "$r" "$ms" "$upr" >> "$LOGDIR/round_profile.tsv"
  info "round[$r]" "kernel_ms=$ms  ${upr} us/record"
  rm -f "/tmp/${REV}_round_${r}.bin"
done
python3 - "$LOGDIR/round_profile.tsv" <<'EOF'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
if len(rows)>=4:
    early=[float(r['kernel_ms']) for r in rows if int(r['round'])<16]
    late=[float(r['kernel_ms']) for r in rows if int(r['round'])>=16]
    if early and late:
        e=sum(early)/len(early); l=sum(late)/len(late)
        print(f"INFO  round_profile: mean(round<16)={e:.1f} ms  mean(round>=16)={l:.1f} ms  ratio={l/e:.3f}")
        print("OK    U2_late_rounds_are_dearer" if l>e else "FAIL  U2_late_rounds_are_dearer: the prefix bias is not positional -- rethink the reduction")
EOF

# ---------------------------------------------------------------------
# 3. U1 -- stratified calibration
# ---------------------------------------------------------------------
K_OK=""; PROXY=""
if [[ "$FORCE_FULL" == "1" ]]; then
  info "U1_stratified_calibration" "skipped by FORCE_FULL=1 -- profiling the real input"
else
  banner "U1 stratified calibration: k rounds spread across all $TOTAL_ROUNDS"
  printf 'k\trounds\trecords\tkernel_ms\tus_per_record\tdev_pct\n' > "$LOGDIR/proxy_calibration.tsv"
  for k in $K_LIST; do
    IDX="$(python3 -c "
import sys
k=int(sys.argv[1]); T=int(sys.argv[2])
print(' '.join(str(round(i*(T-1)/(k-1))) for i in range(k)) if k>1 else '0')
" "$k" "$TOTAL_ROUNDS")"
    out="/tmp/${REV}_strat_k${k}.bin"
    make_rounds "$out" $IDX || { fail "strat_extract[k=$k]" "extraction failed"; continue; }
    ms="$(run_proxy "$out" "3_strat_k${k}")"
    [[ -z "${ms:-}" ]] && { fail "strat_run[k=$k]" "no kernel_ms"; continue; }
    recs=$((STRIDE*k))
    upr="$(awk -v ms="$ms" -v n="$recs" 'BEGIN{printf "%.4f", ms*1000.0/n}')"
    dev="$(awk -v a="$upr" -v r="$REF_US" 'BEGIN{printf "%.3f",(a-r)/r*100}')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$k" "$IDX" "$recs" "$ms" "$upr" "$dev" >> "$LOGDIR/proxy_calibration.tsv"
    info "strat[k=$k]" "rounds=[$IDX] records=$recs kernel_ms=$ms ${upr} us/record dev=${dev}%"
    if [[ -z "$K_OK" ]] && awk -v d="$dev" -v t="$TOL_PROXY" 'BEGIN{x=(d<0?-d:d); exit !(x<=t)}'; then K_OK="$k"; PROXY="$out"; fi
  done
  if [[ -n "$K_OK" ]]; then pass "U1_stratified_proxy_calibrated (k=$K_OK, within ${TOL_PROXY}% per record)"
  else info "U1_NOT_calibrated" "no k in [$K_LIST] hit ${TOL_PROXY}% -- falling back to profiling the REAL full input, one section at a time. See proxy_calibration.tsv; the stall ranking is still produced."; fi
fi

# ---------------------------------------------------------------------
# 4. ncu
# ---------------------------------------------------------------------
SECTIONS="WarpStateStats SpeedOfLight Occupancy SchedulerStats InstructionStats ComputeWorkloadAnalysis MemoryWorkloadAnalysis"
( while true; do echo "$(date +%H:%M:%S), $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')" >> "$LOGDIR/apps_during_ncu.tsv"; sleep 5; done ) &
APPS_PID=$!
: > "$LOGDIR/5_ncu_full.csv"
if [[ -n "$K_OK" ]]; then
  banner "Profiling the calibrated stratified proxy (k=$K_OK)"
  $NCU_PREFIX env NQ_EXTRA_CTX=1 NQ_MAX_BLOCKS="$MB_PROD" "$NCU" $(for s in $SECTIONS; do printf -- "--section %s " "$s"; done) \
    --csv -f -o "$LOGDIR/${REV}_profile" "./$CU_BIN" "$NQ" "$PROXY" "/tmp/${REV}_prof_out.bin" >> "$LOGDIR/5_ncu_full.csv" 2>&1 || true
  reclaim
  pass "profiled_on_calibrated_proxy (k=$K_OK)"
else
  banner "Fallback: profiling the REAL full input, one section at a time (budget ${NCU_BUDGET_S}s)"
  SPENT=0; DONE_SECS=""
  for s in $SECTIONS; do
    [[ "$SPENT" -ge "$NCU_BUDGET_S" ]] && { info "budget_exhausted" "stopping before $s; completed: ${DONE_SECS:-none}"; break; }
    t0=$(date +%s)
    timeout "$SECTION_TIMEOUT_S" $NCU_PREFIX env NQ_EXTRA_CTX=1 NQ_MAX_BLOCKS="$MB_PROD" "$NCU" --section "$s" --csv \
      "./$CU_BIN" "$NQ" "$IN_PROD" "/tmp/${REV}_prof_out.bin" >> "$LOGDIR/5_ncu_full.csv" 2>&1
      rc=$?
    t1=$(date +%s); dt=$((t1-t0)); SPENT=$((SPENT+dt)); reclaim
    if [[ "$rc" == "124" ]]; then info "section[$s]" "timed out after ${SECTION_TIMEOUT_S}s -- skipped"
    else DONE_SECS="$DONE_SECS $s"; info "section[$s]" "completed in ${dt}s (spent ${SPENT}s of ${NCU_BUDGET_S}s)"; fi
  done
  [[ -n "${DONE_SECS// /}" ]] && pass "profiled_on_full_input (sections:$DONE_SECS)" || { fail "profiled_on_full_input" "no section completed inside the budget -- raise NCU_BUDGET_S/SECTION_TIMEOUT_S or profile overnight"; }
fi
kill "$APPS_PID" 2>/dev/null; wait "$APPS_PID" 2>/dev/null || true
info "ncu_own_footprint" "$(tail -1 "$LOGDIR/apps_during_ncu.tsv" 2>/dev/null || echo 'not sampled') -- context count and occupancy move this kernel by 0.3-10%, so the wall clock under ncu is not comparable with production"

# ---------------------------------------------------------------------
# 5. Rank
# ---------------------------------------------------------------------
banner "Ranked results"
python3 - "$LOGDIR/5_ncu_full.csv" <<'EOF' | tee "$LOGDIR/6_ranked.txt"
import csv, sys, re
rows=[]
with open(sys.argv[1], newline='', errors='replace') as f:
    for r in csv.reader(f):
        if len(r)>=2: rows.append(r)
metrics={}
hdr=None
for r in rows:
    if 'Metric Name' in r: hdr=r; continue
    if hdr is None or len(r)<len(hdr): continue
    try:
        n=r[hdr.index('Metric Name')].strip()
        v=float(r[hdr.index('Metric Value')].replace(',',''))
        u=r[hdr.index('Metric Unit')].strip() if 'Metric Unit' in hdr else ''
    except (ValueError, IndexError): continue
    metrics[n]=(v,u)
if not metrics:
    print("no metrics parsed -- inspect 5_ncu_full.csv by hand"); sys.exit(0)
stalls=[(n,v,u) for n,(v,u) in metrics.items() if 'stalled' in n.lower()]
if stalls:
    tot=sum(v for _,v,_ in stalls); stalls.sort(key=lambda x:-x[1])
    print("\n=== U5: warp stall reasons, ranked ===")
    for n,v,u in stalls:
        print(f"  {(v/tot*100 if tot else 0):6.2f}%  {v:10.3f} {u:<10} {re.sub(r'^smsp__average_warps_issue_stalled_','',n)}")
    top=stalls[0]; ts=top[1]/tot*100 if tot else 0
    lead=re.sub(r'^smsp__average_warps_issue_stalled_','',top[0])
    print(f"\n  dominant: {lead} at {ts:.2f}%")
    if 'branch' in top[0].lower() and ts>=25:
        print("  U5 HELD: branch resolving still leads at >=25%. 397 stays in the branch/divergence family.")
    else:
        print("  U5 FALSIFIED: the bottleneck has MOVED. 397 must change optimisation family --")
        print("  the most informative outcome available here. Read the full ranking before choosing.")
else:
    print("no stall metrics -- WarpStateStats did not complete; see 5_ncu_full.csv")
def show(t,p):
    sel=sorted([(n,v,u) for n,(v,u) in metrics.items() if p(n)], key=lambda x:-x[1])
    if sel:
        print(f"\n--- {t} ---")
        for n,v,u in sel[:20]: print(f"  {v:12,.3f} {u:<12} {n}")
show("Speed of Light / utilisation", lambda n:'sol' in n.lower() or 'throughput' in n.lower())
show("Occupancy", lambda n:'occupancy' in n.lower())
show("Issue / instruction", lambda n:'inst_executed' in n or 'issue' in n.lower() or 'ipc' in n.lower())
show("Branch", lambda n:'branch' in n.lower())
show("DRAM / memory", lambda n:n.startswith('dram__') or 'l1tex' in n or 'lts__' in n)
EOF

{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs" 2>/dev/null || true
echo; cat "$LOGDIR/round_profile.tsv" 2>/dev/null; echo; cat "$LOGDIR/proxy_calibration.tsv" 2>/dev/null
TARBALL="${LOGDIR}.tar.gz"
tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed -- send $LOGDIR/"
echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; echo "ranked: $LOGDIR/6_ranked.txt"; echo "tarball: $TARBALL"
if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
echo "${REV} PASSED. Send $TARBALL for analysis."
exit 0
