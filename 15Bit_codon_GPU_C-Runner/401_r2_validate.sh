#!/usr/bin/env bash
# 401_r2_validate.sh
#
# rev401-r2 -- run 401's ladder at three values of N, so that warps per SM
#              can be raised without records per thread collapsing.
#
# TWO HARNESS ERRORS OF MINE THAT 401'S READING DEPENDED ON
#   (1) The footprint model ignored the hardware limit of 16 blocks per SM.
#       With BLOCK=32 a block is one warp, so MAX_BLOCKS above 1280 adds no
#       resident warps -- the extra blocks queue. Every row above MB=1280
#       reported a footprint it never had, and "L2 FAILED: time is not
#       monotone" came entirely from that. Over 800..1280, where the
#       footprint really changes, the curve IS monotone: 2154.7, 2301.0,
#       2279.5, 2433.5, 2854.0, 3032.6, 3304.5 ms. ncu agrees: MB=2560
#       reported 30.25% occupancy and 3.75 warps per scheduler, i.e. 15
#       warps/SM, not 32.
#   (2) warps/SM and k_per_thread cannot be separated at one N -- both come
#       from stride. At N=19, 10 -> 16 warps/SM also drops k from 35.0 to
#       21.9, so the +53% penalty mixes an L1 effect with a tail effect and
#       L5's refutation is confounded.
#   Both are fixed here: blocks/SM is capped, and k is a printed column.
#
# WHAT SURVIVES FROM 401
#   The L1 curve, a per-SM capacity property independent of N:
#     65.0 KB -> 99.51% hit,  78.0 KB -> 98.56%,  104.0 KB -> 97.68%.
#   The production point is already at the edge; effective L1 for local is
#   about 70 KB. My L3 prediction (99% crossing between 80 and 120 KB) was
#   wrong: it is between 71.5 and 78.0 KB.
#
# THE LADDER
#     MB   warps/SM  footprint   k(N=19)  k(N=20)  k(N=21)
#    800      10      65.0 KB      35.0     53.5     79.1
#    960      12      78.0 KB      29.2     44.6     65.9
#   1120      14      91.0 KB      25.0     38.2     56.5
#   1280      16     104.0 KB      21.9     33.4     49.4
#   At N=21 the 16-warp rung has MORE records per thread than N=19's
#   baseline rung has. That is the point of running all three.
#
# PRE-REGISTERED (401_r2_README_append.md; written before execution)
#   M1  N=19 MB=800 lands within +-0.5% of 2,154.677 ms -- session anchor.
#   M2  the MB=1280 penalty shrinks as N rises:
#         penalty(N=19) > penalty(N=20) > penalty(N=21).
#       This tests that the confound is real. If it does not shrink
#       monotonically, "k is what hurts" is the wrong explanation and
#       something else is going on.
#   M3  THE DECISION. STATED PREDICTION: at N=21, MB=1280 is still slower
#       than MB=800, by less than 20%. If it is FASTER, L5's refutation was
#       a debug-scale artefact and the 104-byte packing is back on.
#   M4  if the N=21 penalty is <= 5%, the trend says a smaller frame could
#       flip it and one more revision is justified. If >= 20%, the
#       occupancy axis is dead and `wait` must be attacked by shortening
#       dependency chains instead.
#
#   No ncu: 396 established that ncu cannot finish the 133 s N=21 kernel,
#   and 401's L1 curve already answers the capacity question.
#   Every run is oracle-gated.
#
# USAGE
#   STATIC_ONLY=1 bash 401_r2_validate.sh
#                 bash 401_r2_validate.sh        # ~25 min (N=19,20,21)
#   N_LIST="19 20" bash 401_r2_validate.sh       # ~4 min, no decision cell
#   REPS=1 bash 401_r2_validate.sh               # halve it; noise is 0.02%

set -u

REV="401_r2"
PY_SRC="${PY_SRC:-401_r2Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-401_r2Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-401Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-401_r2_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-401_r2_kernel_maxd14}"
PREV_CU="${PREV_CU:-401_kernel_maxd14.cu}"
CRLOG_DIR="${CRLOG_DIR:-401_r2_crunner_logs}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
N_LIST="${N_LIST:-19 20 21}"
DECIDE_N="${DECIDE_N:-21}"
MB_LIST="${MB_LIST:-800 960 1120 1280}"
BASE_MB="${BASE_MB:-800}"
BLOCK="${BLOCK:-32}"
SMS="${SMS:-80}"
MAX_BLOCKS_PER_SM="${MAX_BLOCKS_PER_SM:-16}"   # sm_86 hardware limit -- the thing 401 forgot
MAX_WARPS_PER_SM="${MAX_WARPS_PER_SM:-48}"
FRAME_B="${FRAME_B:-208}"
TARGET_FRAME_B="${TARGET_FRAME_B:-104}"
REPS="${REPS:-2}"
EXTRA_CTX="${EXTRA_CTX:-2}"
ANCHOR_N19="${ANCHOR_N19:-2154.677}"
KERNEL_SHA_395A="${KERNEL_SHA_395A:-ebd7f523deb591347eab1a352a9555c2e4a6c0b4a456d80517322c933a86e68a}"
STATIC_ONLY="${STATIC_ONLY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"
TSV="$LOGDIR/${REV}_ladder.tsv"

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
{ grep -q "^# ${REV//_/-} " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# ${REV//_/-} ...' note block"
NQ_=$(py_note_quotes "$PY_SRC"); [[ $((NQ_ % 2)) -eq 0 ]] && pass "py_note_region_quotes_balanced ($NQ_)" || fail "py_note_region_quotes_balanced" "$NQ_ triple-quotes"
grep -qE '^REV_TAG:str="401_r2"' "$CODE" && pass "source_rev_tag_is_401_r2" || fail "source_rev_tag_is_401_r2" "wrong REV_TAG"
grep -q 'CRunnerEntry(14,"./401_r2_kernel_maxd14","NQ_EXTRA_CTX=1 "' "$CODE" && pass "source_table_points_at_401_r2_and_keeps_the_treatment" || fail "source_table_points_at_401_r2_and_keeps_the_treatment" "table entry wrong"
grep -qE '^CRUNNER_MIN_N:int=19' "$CODE" && pass "source_crunner_min_n_still_19" || fail "source_crunner_min_n_still_19" "CRUNNER_MIN_N lost"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_401Py (removed=3 added=3 EXECUTABLE lines)" \
    || { fail "py_diff_fingerprint_vs_401Py" "removed=$PR_ added=$PA_, expected 3/3"; diff "/tmp/${REV}_prev_code.py" "$CODE" | head -20 | sed 's/^/      /'; }
else info "py_diff_fingerprint_vs_401Py" "skipped"; fi
cu_code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA_395A" ]] && pass "cu_kernel_region_still_395a_r2 (${KB:0:16}...)" || fail "cu_kernel_region_still_395a_r2" "got ${KB:0:16}..."
if [[ -f "$PREV_CU" ]]; then
  cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  CA="$(sha256sum < "/tmp/${REV}_prev_code.cu" | cut -d' ' -f1)"; CB="$(sha256sum < "/tmp/${REV}_cur_code.cu" | cut -d' ' -f1)"
  [[ "$CA" == "$CB" ]] && pass "cu_whole_code_region_identical_to_401 (${CB:0:16}...)" || fail "cu_whole_code_region_identical_to_401" "401-r2 must be a rename"
else info "cu_whole_code_region_identical_to_401" "skipped"; fi

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

declare -A NIN NREC
for n in $N_LIST; do
  banner "Generating and timing N=$n"
  gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${n}.log"; rm -f "$gcr"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX -u NQ_BLOCK -u NQ_CARVEOUT "./$PY_BIN" -g "$n" "$n" 2>&1 | tee "$LOGDIR/1_gen_N${n}_console.log"
  cp "$gcr" "$LOGDIR/1_gen_N${n}_crunner.log" 2>/dev/null || true
  [[ -f "$gcr" ]] || { fail "crunner_path_taken[N=$n]" "no $gcr"; continue; }
  NIN[$n]="$(grep -o 'src=[^ ]*' "$gcr" | head -1 | cut -d= -f2)"
  NREC[$n]="$(grep -o 'records=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"
  [[ -f "${NIN[$n]:-/nonexistent}" ]] && pass "input_located[N=$n] (${NREC[$n]} records)" || fail "input_located[N=$n]" "src='${NIN[$n]:-<none>}'"
done

# ---------------------------------------------------------------------
# 2. The ladder
# ---------------------------------------------------------------------
printf 'N\tmax_blocks\tblocks_per_sm\twarps_per_sm\tfootprint_kb\tk_per_thread\trep\tkernel_ms\ttotal_sum\tmatch\n' > "$TSV"
declare -A MEAN
resident_of() {  # $1 = max_blocks -> "blocks_per_sm warps_per_sm footprint_kb"
  python3 -c "
mb=int('$1'); blk=$BLOCK; sms=$SMS; capb=$MAX_BLOCKS_PER_SM; capw=$MAX_WARPS_PER_SM; fr=$FRAME_B
bps=min(mb//sms, capb)                      # 401 forgot this cap
w=min(bps*(blk//32), capw)
print(bps, w, round(w*32*fr/1024,1))"
}
run_point() {  # N mb
  local n="$1" mb="$2" orc; orc="$(oracle_of "$n")"
  local rr; rr=($(resident_of "$mb")); local bps="${rr[0]}" wps="${rr[1]}" fkb="${rr[2]}"
  local kpt; kpt="$(awk -v r="${NREC[$n]:-0}" -v s="$((BLOCK*mb))" 'BEGIN{printf "%.1f", (s?r/s:0)}')"
  local sum=0 r
  for r in $(seq 1 "$REPS"); do
    local lg="$LOGDIR/3_N${n}_m${mb}_r${r}.log"
    env -u NQ_PAD_MB -u NQ_CARVEOUT NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$mb" NQ_EXTRA_CTX="$EXTRA_CTX" \
      "./$CU_BIN" "$n" "${NIN[$n]}" "/tmp/${REV}_out.bin" "$orc" > "$lg" 2>&1 || true
    local kms tot m
    kms="$(grep -o 'kernel_ms=[0-9.]*' "$lg" | head -1 | cut -d= -f2)"
    tot="$(grep -o 'total_sum=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
    m=0; grep -q '\[gpu-run-correctness\] MATCH' "$lg" && m=1
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$mb" "$bps" "$wps" "$fkb" "$kpt" "$r" "${kms:-?}" "${tot:-?}" "$m" >> "$TSV"
    [[ "$m" == "1" && "${tot:-}" == "$orc" ]] || { fail "oracle[N=$n mb=$mb rep$r]" "total_sum='${tot:-<none>}' expected $orc"; return 1; }
    sum="$(awk -v s="$sum" -v k="${kms:-0}" 'BEGIN{printf "%.3f",s+k}')"
  done
  MEAN["${n}_${mb}"]="$(awk -v s="$sum" -v r="$REPS" 'BEGIN{printf "%.3f",s/r}')"
  info "N=$n mb=$mb" "blocks/SM=$bps warps/SM=$wps footprint=${fkb} KB k=${kpt}  mean=${MEAN["${n}_${mb}"]} ms"
  return 0
}
for n in $N_LIST; do
  [[ -z "${NIN[$n]:-}" ]] && continue
  banner "Ladder at N=$n ($REPS reps, NQ_EXTRA_CTX=$EXTRA_CTX)"
  for mb in $MB_LIST; do
    run_point "$n" "$mb" || { echo "OK=$PASS FAIL=$FAIL"; exit 1; }
  done
done

# ---------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------
banner "Evaluation"
if [[ -n "${MEAN[19_${BASE_MB}]:-}" ]]; then
  d="$(awk -v a="${MEAN[19_${BASE_MB}]}" -v r="$ANCHOR_N19" 'BEGIN{x=(a-r)/r*100; printf "%.3f",(x<0?-x:x)}')"
  awk -v d="$d" 'BEGIN{exit !(d<=0.5)}' && pass "M1_session_anchor (${MEAN[19_${BASE_MB}]} ms, ${d}% from $ANCHOR_N19)" \
    || fail "M1_session_anchor" "${MEAN[19_${BASE_MB}]} is ${d}% from $ANCHOR_N19 -- not comparable with 401"
fi
declare -A PEN
for n in $N_LIST; do
  b="${MEAN[${n}_${BASE_MB}]:-}"; w="${MEAN[${n}_1280]:-}"
  [[ -z "$b" || -z "$w" ]] && continue
  PEN[$n]="$(awk -v a="$b" -v c="$w" 'BEGIN{printf "%.3f",(c-a)/a*100}')"
  info "penalty[N=$n]" "MB=800 $b ms -> MB=1280 $w ms = ${PEN[$n]}%"
done
NS=($N_LIST); MONO=1
for ((i=0;i<${#NS[@]}-1;i++)); do
  a="${PEN[${NS[$i]}]:-}"; b="${PEN[${NS[$((i+1))]}]:-}"
  [[ -z "$a" || -z "$b" ]] && continue
  awk -v x="$a" -v y="$b" 'BEGIN{exit !(y<x)}' || MONO=0
done
[[ "$MONO" == "1" ]] && pass "M2_penalty_shrinks_with_N (the warps/k confound is real and measurable)" \
  || fail "M2_penalty_shrinks_with_N" "the MB=1280 penalty does not fall monotonically with N -- 'k is what hurts' is the wrong explanation and the mechanism is still unidentified"

banner "M3: the decision, at N=$DECIDE_N"
PD="${PEN[$DECIDE_N]:-}"
if [[ -n "$PD" ]]; then
  if awk -v p="$PD" 'BEGIN{exit !(p<0)}'; then
    pass "M3_REFUTED_MORE_WARPS_WINS (${PD}% at N=$DECIDE_N) -- 401's L5 was a debug-scale artefact. The 104-byte packing is back on: 402 should build it, with the full 2,025,282-record byte-equality gate that stopped 397."
  elif awk -v p="$PD" 'BEGIN{exit !(p<=5)}'; then
    pass "M3_HELD_but_close (${PD}%) -- still a penalty, but small enough that halving the frame could flip it. One more revision is justified; see M4."
  elif awk -v p="$PD" 'BEGIN{exit !(p<20)}'; then
    pass "M3_HELD_as_predicted (${PD}%, under the 20% registered)"
    info "next" "the penalty is real but bounded. Whether a 104-byte frame closes it is a judgement call: the L1 budget says 22 warps/SM would fit, but nothing measured says more warps pays."
  else
    fail "M4_axis_is_dead" "${PD}% at N=$DECIDE_N -- more warps loses badly even with k=49.4, which is more records per thread than N=19's baseline had. Occupancy does not help this algorithm. Do NOT write the packing; attack wait 48% by shortening the dependency chains in the inner loop instead."
  fi
fi

banner "Results"
python3 - "$TSV" "$FRAME_B" "$TARGET_FRAME_B" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys, statistics as st
tsv,frame,target=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
rows=[r for r in csv.DictReader(open(tsv),delimiter='\t') if r['kernel_ms'] not in ('?','')]
by={}
for r in rows: by.setdefault((int(r['N']),int(r['max_blocks'])),[]).append(float(r['kernel_ms']))
meta={(int(r['N']),int(r['max_blocks'])):(r['blocks_per_sm'],r['warps_per_sm'],r['footprint_kb'],r['k_per_thread']) for r in rows}
for N in sorted({k[0] for k in by}):
    pts=sorted([(mb,st.fmean(v)) for (n,mb),v in by.items() if n==N])
    base=dict(pts).get(800, pts[0][1])
    print(f"\n=== N={N} ===")
    print(f"{'MB':>6}{'b/SM':>6}{'w/SM':>6}{'KB':>8}{'k':>8}{'mean ms':>13}{'spread':>9}{'vs MB=800':>11}")
    for mb,ms in pts:
        b,w,kb,k=meta[(N,mb)]
        v=by[(N,mb)]; sp=(max(v)-min(v))/ms*100 if len(v)>1 else 0.0
        print(f"{mb:>6}{b:>6}{w:>6}{kb:>8}{k:>8}{ms:>13.3f}{sp:>8.3f}%{(ms-base)/base*100:>+10.2f}%")
print("\n=== the confound, made visible ===")
print(f"{'N':>4}{'k at 10 w/SM':>15}{'k at 16 w/SM':>15}{'penalty':>10}")
for N in sorted({k[0] for k in by}):
    if (N,800) in by and (N,1280) in by:
        k10=meta[(N,800)][3]; k16=meta[(N,1280)][3]
        p=(st.fmean(by[(N,1280)])-st.fmean(by[(N,800)]))/st.fmean(by[(N,800)])*100
        print(f"{N:>4}{k10:>15}{k16:>15}{p:>+9.2f}%")
print("\n  401 measured the L1 budget at 71.5 KB for >=99% hit (65.0 KB -> 99.51%,")
print(f"  78.0 KB -> 98.56%). With a {target}-byte frame that budget holds")
print(f"  {71.5*1024/target:.0f} threads/SM = {71.5*1024/target/32:.1f} warps/SM, against 10 today --")
print("  but that only matters if the table above shows more warps paying at all.")
EOF

{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs" 2>/dev/null || true
TARBALL="${LOGDIR}.tar.gz"
tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; echo "ranked: $LOGDIR/9_ranked.txt"; echo "tarball: $TARBALL"
if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
echo "${REV} PASSED. Send $TARBALL for analysis."
exit 0
