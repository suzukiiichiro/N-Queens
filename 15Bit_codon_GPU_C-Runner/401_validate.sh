#!/usr/bin/env bash
# 401_validate.sh
#
# rev401 -- map the cliff before building anything.
#
# WHAT 400-r3 SETTLED
#   The carveout knob is inert: the driver already maximises L1.
#     default effective_pref=-1, NQ_CARVEOUT=0 -> effective_pref=0
#     stride 81,920: 10,729.669 -> 10,729.866 ms (+0.002%)
#     L1 hit rate  :     85.96% ->     85.92%
#   But the diagnostic it carried located the wall exactly:
#     stride 25,600   320 thr/SM   65 KB   L1 hit 99.48%   L2 read    478 M
#     stride 81,920  1024 thr/SM  208 KB   L1 hit 85.96%   L2 read 12,298 M
#   Miss rate 0.52% -> 14.04% (27x) and L2 traffic 25.7x agree, and the
#   dominant stall inverts with them:
#                       stride 25,600     stride 81,920
#     long_scoreboard        12.9%            94.5%
#     wait                   48.0%             3.1%
#     branch_resolving       28.5%             1.8%
#   Occupancy did rise as intended, warps per scheduler 2.44 -> 7.71. The
#   DFS stack fell out of L1 the moment it did. The 208-byte frame is the
#   gatekeeper of the occupancy axis -- and at the production point local
#   memory is not a problem at all.
#
# THE ONE QUESTION 401 ASKS
#   400-r2 and r3 only ever sampled strides far past the cliff. Nobody has
#   looked at the near side. Before writing a smaller frame it is worth
#   knowing whether more warps help AT ALL while the working set still fits:
#
#     is ANY configuration with more than 10 warps/SM faster than the
#     320-thread baseline?
#
#   Yes -> halving the frame extends a range that is known to pay, and 402
#   builds it. No -> occupancy never helps for this algorithm and the
#   packing would be wasted however well it were written. Ten minutes, no
#   code change, and 397's mistake (building before the diagnosis finished)
#   is not repeated.
#
# THE GRID -- BLOCK=32, MAX_BLOCKS in multiples of 80 so blocks/SM stays a
# whole number (400-r2 measured a 6.5% penalty for 200 blocks over 80 SMs)
#     MB    thr/SM  warps/SM  footprint     MB    thr/SM  warps/SM  footprint
#    800     320      10        65.0 KB    1280     512      16       104.0 KB
#    880     352      11        71.5 KB    1440     576      18       117.0 KB
#    960     384      12        78.0 KB    1600     640      20       130.0 KB
#   1040     416      13        84.5 KB    1920     768      24       156.0 KB
#   1120     448      14        91.0 KB    2560    1024      32       208.0 KB
#   1200     480      15        97.5 KB
#
# PRE-REGISTERED (401_README_append.md; written before execution)
#   L1  MB=800 lands within +-0.5% of 2,153.8 ms -- the session anchor.
#   L2  time rises monotonically with footprint. A non-monotonic curve means
#       something other than capacity is mixed in.
#   L3  L1 hit rate falls monotonically and crosses 99% between 80 and
#       120 KB.
#   L4  the footprint where time first exceeds baseline+5% is within one
#       sweep step of where hit rate first drops below 99% -- the slowdown
#       is accounted for by misses, not by something else.
#   L5  THE DECISION. STATED PREDICTION: at least one configuration above
#       10 warps/SM beats the baseline, by something in the 0.5-3% range.
#       If none does, the occupancy axis is dead regardless of frame size
#       and 402 must NOT write the packing.
#
#   Every run is oracle-gated: changing MAX_BLOCKS only repartitions the
#   same records, so the answer must not move.
#
# USAGE
#   STATIC_ONLY=1 bash 401_validate.sh
#                 bash 401_validate.sh          # ~12 min at N=19
#   NPROF=20 NCONF=21 bash 401_validate.sh      # production scale
#   NCU_PREFIX=sudo bash 401_validate.sh

set -u

REV="401"
PY_SRC="${PY_SRC:-401Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-401Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-400_r3Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-401_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-401_kernel_maxd14}"
PREV_CU="${PREV_CU:-400_r3_kernel_maxd14.cu}"
CRLOG_DIR="${CRLOG_DIR:-401_crunner_logs}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
NCU="${NCU:-ncu}"
ARCH="${ARCH:-sm_86}"
NPROF="${NPROF:-19}"
NCONF="${NCONF:-20}"
BASE_MB="${BASE_MB:-800}"
BLOCK="${BLOCK:-32}"
SMS="${SMS:-80}"
FRAME_B="${FRAME_B:-208}"
TARGET_FRAME_B="${TARGET_FRAME_B:-104}"
REPS="${REPS:-3}"
REPS_CONF="${REPS_CONF:-3}"
EXTRA_CTX="${EXTRA_CTX:-2}"
MB_LIST="${MB_LIST:-800 880 960 1040 1120 1200 1280 1440 1600 1920 2560}"
NCU_MB_LIST="${NCU_MB_LIST:-800 960 1120 1280 1600 2560}"
KERNEL_SHA_395A="${KERNEL_SHA_395A:-ebd7f523deb591347eab1a352a9555c2e4a6c0b4a456d80517322c933a86e68a}"
SECTION_TIMEOUT_S="${SECTION_TIMEOUT_S:-600}"
L1_METRICS="l1tex__t_sector_hit_rate.pct,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum,lts__t_sectors_srcunit_tex_op_read.sum,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio"
STATIC_ONLY="${STATIC_ONLY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"
TSV="$LOGDIR/${REV}_footprint.tsv"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }
oracle_of() { case "$1" in 18) echo 666090624;; 19) echo 4968057848;; 20) echo 39029188884;;
  21) echo 314666222712;; 22) echo 2691008701644;; *) echo "";; esac; }

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
{ grep -q "^# ${REV} " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# ${REV} ...' note block"
NQ_=$(py_note_quotes "$PY_SRC"); [[ $((NQ_ % 2)) -eq 0 ]] && pass "py_note_region_quotes_balanced ($NQ_)" || fail "py_note_region_quotes_balanced" "$NQ_ triple-quotes"
grep -qE '^REV_TAG:str="401"' "$CODE" && pass "source_rev_tag_is_401" || fail "source_rev_tag_is_401" "wrong REV_TAG"
grep -q 'CRunnerEntry(14,"./401_kernel_maxd14","NQ_EXTRA_CTX=1 "' "$CODE" && pass "source_table_points_at_401_and_keeps_the_treatment" || fail "source_table_points_at_401_and_keeps_the_treatment" "table entry wrong"
grep -qE '^CRUNNER_MIN_N:int=19' "$CODE" && pass "source_crunner_min_n_still_19" || fail "source_crunner_min_n_still_19" "CRUNNER_MIN_N lost"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_400_r3Py (removed=3 added=3 EXECUTABLE lines)" \
    || { fail "py_diff_fingerprint_vs_400_r3Py" "removed=$PR_ added=$PA_, expected 3/3"; diff "/tmp/${REV}_prev_code.py" "$CODE" | head -20 | sed 's/^/      /'; }
else info "py_diff_fingerprint_vs_400_r3Py" "skipped"; fi
cu_code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA_395A" ]] && pass "cu_kernel_region_still_395a_r2 (${KB:0:16}...)" || fail "cu_kernel_region_still_395a_r2" "got ${KB:0:16}..."
if [[ -f "$PREV_CU" ]]; then
  cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  CA="$(sha256sum < "/tmp/${REV}_prev_code.cu" | cut -d' ' -f1)"; CB="$(sha256sum < "/tmp/${REV}_cur_code.cu" | cut -d' ' -f1)"
  [[ "$CA" == "$CB" ]] && pass "cu_whole_code_region_identical_to_400_r3 (${CB:0:16}... -- 401 is a rename; the sweep needs no code)" \
    || fail "cu_whole_code_region_identical_to_400_r3" "401 must be a rename"
else info "cu_whole_code_region_identical_to_400_r3" "skipped"; fi

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
banner "Building"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$CU_BIN" "$CU_SRC" -lcuda 2>&1 | tee "$LOGDIR/05_nvcc.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "no binary"; exit 1; }
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/06_codon.log"
[[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN]" || { fail "codon_build" "see log"; exit 1; }

declare -A NIN
gen_for_N() {
  local n="$1" gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${1}.log"
  banner "Generating and timing N=$n"
  rm -f "$gcr"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX -u NQ_BLOCK -u NQ_CARVEOUT "./$PY_BIN" -g "$n" "$n" 2>&1 | tee "$LOGDIR/1_gen_N${n}_console.log"
  cp "$gcr" "$LOGDIR/1_gen_N${n}_crunner.log" 2>/dev/null || true
  [[ -f "$gcr" ]] || { fail "crunner_path_taken[N=$n]" "no $gcr"; return 1; }
  NIN[$n]="$(grep -o 'src=[^ ]*' "$gcr" | head -1 | cut -d= -f2)"
  [[ -f "${NIN[$n]:-/nonexistent}" ]] && pass "input_located[N=$n] (${NIN[$n]})" || { fail "input_located[N=$n]" "src='${NIN[$n]:-<none>}'"; return 1; }
}
gen_for_N "$NPROF" || exit 1
gen_for_N "$NCONF" || info "NCONF_input" "N=$NCONF unavailable; the confirmation will be skipped"
ORC_P="$(oracle_of "$NPROF")"; ORC_C="$(oracle_of "$NCONF")"

# ---------------------------------------------------------------------
# 2. The sweep
# ---------------------------------------------------------------------
printf 'N\tmax_blocks\tstride\tthreads_per_sm\twarps_per_sm\tfootprint_kb\trep\tkernel_ms\ttotal_sum\tmatch\n' > "$TSV"
declare -A MEAN
run_mb() {  # N input oracle reps mb tag
  local n="$1" input="$2" orc="$3" reps="$4" mb="$5" tag="$6"
  local stride=$((BLOCK*mb)) tps=$((BLOCK*mb/SMS)) sum=0 r
  local wps=$((tps/32)); local fkb
  fkb="$(awk -v t="$tps" -v f="$FRAME_B" 'BEGIN{printf "%.1f", t*f/1024}')"
  for r in $(seq 1 "$reps"); do
    local lg="$LOGDIR/3_${tag}_m${mb}_r${r}.log"
    env -u NQ_PAD_MB -u NQ_CARVEOUT NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$mb" NQ_EXTRA_CTX="$EXTRA_CTX" \
      "./$CU_BIN" "$n" "$input" "/tmp/${REV}_out.bin" "$orc" > "$lg" 2>&1 || true
    local kms tot m sact
    kms="$(grep -o 'kernel_ms=[0-9.]*' "$lg" | head -1 | cut -d= -f2)"
    tot="$(grep -o 'total_sum=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
    sact="$(grep -o 'stride=[0-9]*' "$lg" | tail -1 | cut -d= -f2)"
    m=0; grep -q '\[gpu-run-correctness\] MATCH' "$lg" && m=1
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$mb" "$stride" "$tps" "$wps" "$fkb" "$r" "${kms:-?}" "${tot:-?}" "$m" >> "$TSV"
    [[ "$m" == "1" && "${tot:-}" == "$orc" ]] || { fail "oracle[$tag mb=$mb rep$r]" "total_sum='${tot:-<none>}' expected $orc"; return 1; }
    [[ "${sact:-}" == "$stride" ]] || { fail "stride_as_intended[$tag mb=$mb]" "got ${sact:-?}"; return 1; }
    sum="$(awk -v s="$sum" -v k="${kms:-0}" 'BEGIN{printf "%.3f",s+k}')"
  done
  MEAN["${tag}_${mb}"]="$(awk -v s="$sum" -v r="$reps" 'BEGIN{printf "%.3f",s/r}')"
  info "mb[$mb]" "warps/SM=$wps footprint=${fkb} KB  mean=${MEAN["${tag}_${mb}"]} ms"
  return 0
}
banner "Footprint sweep at N=$NPROF, $REPS reps, BLOCK=$BLOCK, NQ_EXTRA_CTX=$EXTRA_CTX"
for mb in $MB_LIST; do
  run_mb "$NPROF" "${NIN[$NPROF]}" "$ORC_P" "$REPS" "$mb" nP || { echo "OK=$PASS FAIL=$FAIL"; exit 1; }
done
BASE="${MEAN[nP_${BASE_MB}]:-}"
[[ -n "$BASE" ]] && pass "baseline_measured (MB=$BASE_MB = $BASE ms)" || { fail "baseline_measured" "no baseline"; exit 1; }
if [[ "$NPROF" == "19" ]]; then
  d="$(awk -v a="$BASE" 'BEGIN{x=(a-2153.8)/2153.8*100; printf "%.3f",(x<0?-x:x)}')"
  awk -v d="$d" 'BEGIN{exit !(d<=0.5)}' && pass "L1_session_anchor ($BASE ms, ${d}% from 400-r3's 2,153.8)" \
    || fail "L1_session_anchor" "$BASE is ${d}% from 2,153.8 -- the session is not comparable with 400-r3"
fi

# ---------------------------------------------------------------------
# 3. L1 hit rate at a subset
# ---------------------------------------------------------------------
banner "L1 hit rate at a subset of the grid"
command -v "$NCU" >/dev/null 2>&1 && pass "ncu_present" || fail "ncu_present" "set NCU=..."
NCU_PREFIX="${NCU_PREFIX-}"
reclaim() { [[ -n "$NCU_PREFIX" ]] && sudo chown -R "$(id -u):$(id -g)" "$LOGDIR" 2>/dev/null; true; }
if command -v "$NCU" >/dev/null 2>&1; then
  head -c $((25600*28)) "${NIN[$NPROF]}" > "/tmp/${REV}_tiny.bin"
  ncu_probe() { $1 env NQ_EXTRA_CTX=1 NQ_MAX_BLOCKS=800 "$NCU" --section SpeedOfLight --csv \
      "./$CU_BIN" "$NPROF" "/tmp/${REV}_tiny.bin" "/tmp/${REV}_t.bin" > "$2" 2>&1 || true
    grep -q 'kernel_dfs_iter_gpu_maxd14' "$2"; }
  if [[ -n "$NCU_PREFIX" ]]; then ncu_probe "$NCU_PREFIX" "$LOGDIR/07_probe.log" && pass "ncu_permission (via '$NCU_PREFIX')" || fail "ncu_permission" "cannot read counters"
  elif ncu_probe "" "$LOGDIR/07_probe.log"; then pass "ncu_permission (unprivileged)"
  elif ncu_probe "sudo" "$LOGDIR/07_probe_sudo.log"; then NCU_PREFIX="sudo"; pass "ncu_permission (under sudo)"
  else fail "ncu_permission" "counters unreadable"; fi
  for mb in $NCU_MB_LIST; do
    timeout "$SECTION_TIMEOUT_S" $NCU_PREFIX env NQ_EXTRA_CTX=1 NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$mb" "$NCU" \
      --metrics "$L1_METRICS" --section Occupancy --section SchedulerStats --csv \
      "./$CU_BIN" "$NPROF" "${NIN[$NPROF]}" "/tmp/${REV}_p.bin" >> "$LOGDIR/8_ncu_m${mb}.csv" 2>&1
    rc=$?; reclaim
    [[ "$rc" == "124" ]] && info "ncu[mb=$mb]" "timed out" || { grep -q '==ERROR==' "$LOGDIR/8_ncu_m${mb}.csv" && info "ncu[mb=$mb]" "ncu error" || pass "ncu_metrics[mb=$mb]"; }
  done
fi

# ---------------------------------------------------------------------
# 4. L5 -- the decision, and the confirmation
# ---------------------------------------------------------------------
banner "L5: is any configuration above 10 warps/SM faster than the baseline?"
BESTMB="$BASE_MB"; BESTV="$BASE"
for mb in $MB_LIST; do
  v="${MEAN[nP_${mb}]:-}"; [[ -z "$v" ]] && continue
  d="$(awk -v a="$BASE" -v b="$v" 'BEGIN{printf "%+.3f",(b-a)/a*100}')"
  tps=$((BLOCK*mb/SMS)); fkb="$(awk -v t="$tps" -v f="$FRAME_B" 'BEGIN{printf "%.1f", t*f/1024}')"
  info "point" "MB=$mb warps/SM=$((tps/32)) footprint=${fkb} KB  $v ms  ${d}%"
  [[ "$mb" == "$BASE_MB" ]] && continue
  if awk -v a="$v" -v b="$BESTV" 'BEGIN{exit !(a<b)}'; then BESTMB="$mb"; BESTV="$v"; fi
done
if [[ "$BESTMB" != "$BASE_MB" ]]; then
  g="$(awk -v a="$BASE" -v b="$BESTV" 'BEGIN{printf "%.3f",(a-b)/a*100}')"
  pass "L5_CONFIRMED_more_warps_can_help (MB=$BESTMB, $((BLOCK*BESTMB/SMS/32)) warps/SM, ${g}% faster) -- halving the frame extends a range that pays, so 402 should build the packing"
else
  fail "L5_REFUTED" "no configuration above 10 warps/SM beats MB=$BASE_MB. Occupancy does not help this algorithm at any footprint that fits, so a smaller frame would buy nothing and 402 must NOT write the packing. Look elsewhere for the wait 48% -- shorter dependency chains in the inner loop, not more warps."
fi
if [[ -n "${NIN[$NCONF]:-}" && -n "$ORC_C" && "$BESTMB" != "$BASE_MB" ]]; then
  banner "Confirm the winner at N=$NCONF"
  run_mb "$NCONF" "${NIN[$NCONF]}" "$ORC_C" "$REPS_CONF" "$BASE_MB" nC || true
  run_mb "$NCONF" "${NIN[$NCONF]}" "$ORC_C" "$REPS_CONF" "$BESTMB"  nC || true
  bc="${MEAN[nC_${BASE_MB}]:-}"; wc="${MEAN[nC_${BESTMB}]:-}"
  if [[ -n "$bc" && -n "$wc" ]]; then
    gc="$(awk -v a="$bc" -v b="$wc" 'BEGIN{printf "%.3f",(a-b)/a*100}')"
    info "N$NCONF" "baseline $bc -> MB=$BESTMB $wc = ${gc}%"
    awk -v g="$gc" 'BEGIN{exit !(g>0)}' && pass "winner_carries_to_N${NCONF} (${gc}%)" || fail "winner_carries_to_N${NCONF}" "${gc}% -- the winner is specific to N=$NPROF"
  fi
fi

banner "Results"
python3 - "$LOGDIR" "$TSV" "$FRAME_B" "$TARGET_FRAME_B" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, glob, os, re, sys, statistics as st
logdir,tsv,frame,target=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
rows=[r for r in csv.DictReader(open(tsv),delimiter='\t') if r['kernel_ms'] not in ('?','')]
by={}
for r in rows: by.setdefault((r['N'],int(r['max_blocks'])),[]).append(float(r['kernel_ms']))
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
hits={}
for p in sorted(glob.glob(os.path.join(logdir,'8_ncu_m*.csv'))):
    mb=int(re.search(r'8_ncu_m(\d+)\.csv',p).group(1)); m=load(p)
    hits[mb]=(m.get('l1tex__t_sector_hit_rate.pct'), m.get('Achieved Occupancy'),
              m.get('Active Warps Per Scheduler'), m.get('lts__t_sectors_srcunit_tex_op_read.sum'),
              m.get('smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio'),
              m.get('smsp__average_warps_issue_stalled_wait_per_issue_active.ratio'))
Ns=sorted({k[0] for k in by})
for N in Ns:
    pts=sorted([(mb,st.fmean(v)) for (n,mb),v in by.items() if n==N])
    if not pts: continue
    base=dict(pts).get(800, pts[0][1])
    print(f"\n=== N={N} ===")
    print(f"{'MB':>6}{'w/SM':>6}{'KB':>8}{'mean ms':>12}{'vs base':>10}{'L1 hit':>9}{'occ':>8}{'w/sched':>9}{'lsb%':>7}{'wait%':>7}")
    for mb,ms in pts:
        tps=32*mb//80; fkb=tps*frame/1024
        h=hits.get(mb)
        hs=f"{h[0]:8.2f}" if h and h[0] is not None else "       -"
        oc=f"{h[1]:7.2f}" if h and h[1] is not None else "      -"
        ws=f"{h[2]:8.2f}" if h and h[2] is not None else "       -"
        tot=(h[4] or 0)+(h[5] or 0) if h else 0
        lsb=f"{(h[4]/tot*100):6.1f}" if h and tot else "     -"
        wt =f"{(h[5]/tot*100):6.1f}" if h and tot else "     -"
        print(f"{mb:>6}{tps//32:>6}{fkb:>8.1f}{ms:>12.3f}{(ms-base)/base*100:>+9.2f}%{hs}{oc}{ws}{lsb}{wt}")
    if N==Ns[0]:
        mono=all(pts[i][1]<=pts[i+1][1]+1e-9 for i in range(len(pts)-1))
        print(f"\n  L2 {'HELD' if mono else 'FAILED'}: time is {'monotone' if mono else 'NOT monotone'} in footprint"
              + ("" if mono else " -- something other than capacity is mixed in"))
        cross=[mb for mb,_ in pts if hits.get(mb) and hits[mb][0] is not None and hits[mb][0] < 99.0]
        if cross:
            mb0=min(cross); kb0=32*mb0//80*frame/1024
            prev=[mb for mb,_ in pts if mb<mb0]
            kbp=(32*max(prev)//80*frame/1024) if prev else 0
            print(f"  L3: L1 hit first falls below 99% between {kbp:.1f} and {kb0:.1f} KB")
            print("  L3 HELD" if 80<=kb0<=120 else "  L3 outside the registered 80-120 KB band")
            budget=kbp if kbp else kb0
            print(f"\n  BUDGET: {budget:.1f} KB is the largest footprint measured at >=99% hit.")
            print(f"  With a {target}-byte frame that budget holds {budget*1024/target:.0f} threads/SM "
                  f"= {budget*1024/target/32:.1f} warps/SM, against {32*800//80//32} today.")
        else:
            print("  L3: no measured point fell below 99% -- the cliff is above the swept range")
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
