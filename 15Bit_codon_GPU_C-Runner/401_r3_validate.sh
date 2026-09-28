#!/usr/bin/env bash
# 401_r3_validate.sh
#
# rev401-r3 -- close 401-r2 on measurement, not extrapolation. Zero code change.
#
# WHAT 401-r2 LEFT OPEN
#   401-r2 ran with N_LIST="19 20" only, so the decision cell (M3: N=21,
#   MB=800 vs 1280) never ran. What did run refuted M2: the MB=1280 penalty
#   GROWS with N (N=19 +53.33%, N=20 +71.72%). records-per-thread is not the
#   mechanism.
#   Its logs also hold an unexplained gap at N=20: same binary, same input,
#   3 contexts and free_mb=21776 in both, yet
#       dispatcher  -g 20 20              16,480.3 ms
#       direct      NQ_EXTRA_CTX=2        17,378.4 ms   (+5.5%)
#   397's V4 ("direct + NQ_EXTRA_CTX=2 reproduces the production state") was
#   only ever confirmed at N=21.
#
# THREE QUESTIONS
#   (A) M3 at N=21: MB=800 vs MB=1280.
#   (B) The premise of 402 (104-byte packing): is the slope INSIDE L1 still
#       positive at N=21 with the current binary? 394g measured 484->704->800
#       ->968 as -23.7% / -4.6% / +11.3%. r3 puts MB=720 (9 warps/SM, 58.5 KB)
#       and MB=880 (11 warps/SM, 71.5 KB) on either side of 800.
#   (C) Does the N=20 gap reproduce, and which side does an external holder
#       take?  Gd = -g 20 20 | Dx2 = direct NQ_EXTRA_CTX=2 |
#       Dh = direct NQ_EXTRA_CTX=1 + 395c_ctx_holder (0 MB).  Interleaved.
#
# PRE-REGISTERED (401_r3_README_append.md; written before execution)
#   R0  G21 (-g 21 21) within +-0.10% of 133,192 ms, extra_ctx=1.
#   R1  direct N=21 MB=800 within +-0.15% of 133,197 ms, free_mb=21762,
#       extra_ctx=2.  HARD: if it fails, R2-R4 are not read.
#   R2  N=21 MB=1280 penalty >= 20%  (point guess ~85%).  Closes M3/M4.
#   R3  N=21 MB=720 slower than MB=800 by +1.5..+6%.  Refuted if within
#       +-0.3% or faster -> the inside-L1 slope has flattened; 402's premise
#       is weakened.
#   R4  N=21 MB=880 slower than MB=800.  If faster by >0.3%, record as an
#       A/B candidate (do not adopt from one run).
#   R5  N=20: Dx2 - Gd >= +3% reproduces.  Stated (weak): Dh sides with Gd.
#       Falsified if all three agree within 0.1%.
#       Decision rule: if a >=1% gap persists, 402's N=19/20 cells must use
#       whichever direct method matched Gd, or the dispatcher itself.
#
#   GPU must be empty before every cell (the holder is the only allowed
#   compute process, and only during Dh). Every run is oracle-gated.
#
# USAGE
#   STATIC_ONLY=1 bash 401_r3_validate.sh      # OK=14
#                 bash 401_r3_validate.sh      # ~20 min (B: 3 min, A: 13 min, build/gen 4)
#   SKIP_B=1      bash 401_r3_validate.sh      # N=21 only
#   SKIP_A=1      bash 401_r3_validate.sh      # N=20 only (~5 min)
#   MB21_LIST="800 1280" bash 401_r3_validate.sh   # decision cell only

set -u

REV="401_r3"
PY_SRC="${PY_SRC:-401_r3Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-401_r3Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-401_r2Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-401_r3_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-401_r3_kernel_maxd14}"
PREV_CU="${PREV_CU:-401_r2_kernel_maxd14.cu}"
HOLDER_SRC="${HOLDER_SRC:-395c_ctx_holder.cu}"
HOLDER_BIN="${HOLDER_BIN:-395c_ctx_holder}"
CRLOG_DIR="${CRLOG_DIR:-401_r3_crunner_logs}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
BLOCK="${BLOCK:-32}"
SMS="${SMS:-80}"
MAX_BLOCKS_PER_SM="${MAX_BLOCKS_PER_SM:-16}"
FRAME_B="${FRAME_B:-208}"
# phase B (N=20)
NB="${NB:-20}"
MB_B="${MB_B:-800}"
REPS_B="${REPS_B:-2}"
SKIP_B="${SKIP_B:-0}"
# phase A (N=21)
NA="${NA:-21}"
MB21_LIST="${MB21_LIST:-720 800 880 1280}"
BASE_MB="${BASE_MB:-800}"
REPS_A="${REPS_A:-1}"
SKIP_A="${SKIP_A:-0}"
EXTRA_CTX_DIRECT="${EXTRA_CTX_DIRECT:-2}"
# anchors
ANCHOR_G21="${ANCHOR_G21:-133192}"        # production, 395c-r6 .. 401-r2
ANCHOR_D21="${ANCHOR_D21:-133197.422}"    # 398 W1 / 401-r2Py note, direct + NQ_EXTRA_CTX=2
ANCHOR_G20="${ANCHOR_G20:-16480.264}"     # 401-r2 1_gen_N20_crunner.log
ANCHOR_D20="${ANCHOR_D20:-17378.408}"     # 401-r2 ladder N=20 MB=800
FREE_MB_21="${FREE_MB_21:-21762}"
KERNEL_SHA_395A="${KERNEL_SHA_395A:-ebd7f523deb591347eab1a352a9555c2e4a6c0b4a456d80517322c933a86e68a}"
COOLDOWN="${COOLDOWN:-10}"
STATIC_ONLY="${STATIC_ONLY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"
TSV="$LOGDIR/${REV}_results.tsv"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }
oracle_of() { case "$1" in 18) echo 666090624;; 19) echo 4968057848;; 20) echo 39029188884;;
  21) echo 314666222712;; 22) echo 2691008701644;; *) echo "";; esac; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f",(a-b)/b*100}'; }        # (a-b)/b %
abspct() { awk -v a="$1" -v b="$2" 'BEGIN{x=(a-b)/b*100; printf "%.3f",(x<0?-x:x)}'; }
le() { awk -v x="$1" -v y="$2" 'BEGIN{exit !(x<=y)}'; }
ge() { awk -v x="$1" -v y="$2" 'BEGIN{exit !(x>=y)}'; }

# ---------------------------------------------------------------------
# 1. Static
# ---------------------------------------------------------------------
for f in "$PY_SRC" "$HELPER_SRC" "$CU_SRC" "$HOLDER_SRC"; do
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
grep -qE '^REV_TAG:str="401_r3"' "$CODE" && pass "source_rev_tag_is_401_r3" || fail "source_rev_tag_is_401_r3" "wrong REV_TAG"
grep -q 'CRunnerEntry(14,"./401_r3_kernel_maxd14","NQ_EXTRA_CTX=1 "' "$CODE" && pass "source_table_points_at_401_r3_and_keeps_the_treatment" || fail "source_table_points_at_401_r3_and_keeps_the_treatment" "table entry wrong"
grep -qE '^CRUNNER_MIN_N:int=19' "$CODE" && pass "source_crunner_min_n_still_19" || fail "source_crunner_min_n_still_19" "CRUNNER_MIN_N lost"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_401_r2Py (removed=3 added=3 EXECUTABLE lines)" \
    || { fail "py_diff_fingerprint_vs_401_r2Py" "removed=$PR_ added=$PA_, expected 3/3"; diff "/tmp/${REV}_prev_code.py" "$CODE" | head -20 | sed 's/^/      /'; }
else info "py_diff_fingerprint_vs_401_r2Py" "skipped"; fi
cu_code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA_395A" ]] && pass "cu_kernel_region_still_395a_r2 (${KB:0:16}...)" || fail "cu_kernel_region_still_395a_r2" "got ${KB:0:16}..."
if [[ -f "$PREV_CU" ]]; then
  cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  CA="$(sha256sum < "/tmp/${REV}_prev_code.cu" | cut -d' ' -f1)"; CB="$(sha256sum < "/tmp/${REV}_cur_code.cu" | cut -d' ' -f1)"
  [[ "$CA" == "$CB" ]] && pass "cu_whole_code_region_identical_to_401_r2 (${CB:0:16}...)" || fail "cu_whole_code_region_identical_to_401_r2" "401-r3 must be a rename"
else info "cu_whole_code_region_identical_to_401_r2" "skipped"; fi
grep -q "rev401-r3" "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev401-r3 note in header"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Build
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
banner "Building"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$CU_BIN" "$CU_SRC" -lcuda 2>&1 | tee "$LOGDIR/05_nvcc.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "no binary"; exit 1; }
if [[ ! -x "$HOLDER_BIN" ]]; then "$NVCC" -O2 -arch="$ARCH" -o "$HOLDER_BIN" "$HOLDER_SRC" 2>&1 | tee "$LOGDIR/05a_nvcc_holder.log"; fi
[[ -x "$HOLDER_BIN" ]] && pass "ctx_holder_available[$HOLDER_BIN]" || { fail "ctx_holder_available" "no $HOLDER_BIN"; exit 1; }
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/06_codon.log"
[[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN]" || { fail "codon_build" "see log"; exit 1; }

# ---------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------
printf 'cell\tN\tpath\tmax_blocks\twarps_per_sm\tfootprint_kb\textra_ctx\tfree_mb\trep\tkernel_ms\ttotal_sum\tmatch\tstart\n' > "$TSV"
HPID=""
gpu_gate() {  # $1 cell. The GPU must hold no compute process except our holder.
  local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"
  echo "$apps" > "$LOGDIR/apps_before_$1.txt"
  local others; others="$(echo "$apps" | awk -F, -v h="${HPID:-none}" 'NF && $1!=h {print}')"
  if [[ -n "$others" ]]; then
    fail "gpu_empty_before[$1]" "foreign compute process(es) on the GPU: $(echo "$others" | tr '\n' ' ')"
    echo "ABORT: a leftover process moves this kernel by 4-6% (395c). Clear it and rerun."; return 1
  fi
  return 0
}
fields_of() {  # $1 log -> kms tot match extra free
  local lg="$1"
  KMS="$(grep -o 'kernel_ms=[0-9.]*' "$lg" | head -1 | cut -d= -f2)"
  TOT="$(grep -o 'total_sum=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$lg" && MATCH=1
  XCTX="$(grep -o 'extra_ctx=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  FREE="$(grep -o 'free_mb=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
}
resident_of() {  # $1 max_blocks -> "warps_per_sm footprint_kb"
  python3 -c "
mb=int('$1'); blk=$BLOCK; sms=$SMS; capb=$MAX_BLOCKS_PER_SM; fr=$FRAME_B
bps=min(mb//sms, capb); w=bps*(blk//32); print(w, round(w*32*fr/1024,1))"
}
record() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$TSV"; }
declare -A NIN NREC MEAN CNT XC FM
add_mean() {  # key value
  local k="$1" v="$2"
  MEAN[$k]="$(awk -v s="${MEAN[$k]:-0}" -v x="$v" 'BEGIN{printf "%.3f",s+x}')"; CNT[$k]=$(( ${CNT[$k]:-0} + 1 ))
}
finish_mean() { local k; for k in "${!MEAN[@]}"; do MEAN[$k]="$(awk -v s="${MEAN[$k]}" -v n="${CNT[$k]}" 'BEGIN{printf "%.3f",s/n}')"; done; }

run_dispatch() {  # cell N rep  -> bare -g N N through the dispatcher (production path)
  local cell="$1" n="$2" rep="$3" orc; orc="$(oracle_of "$n")"
  local gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${n}.log"; rm -f "$gcr"
  gpu_gate "${cell}_r${rep}" || return 1
  local start; start="$(date -Is)"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX -u NQ_BLOCK -u NQ_CARVEOUT "./$PY_BIN" -g "$n" "$n" > "$LOGDIR/2_${cell}_r${rep}_console.log" 2>&1
  cp "$gcr" "$LOGDIR/2_${cell}_r${rep}_crunner.log" 2>/dev/null || { fail "crunner_path_taken[$cell r$rep]" "no $gcr"; return 1; }
  fields_of "$gcr"
  NIN[$n]="$(grep -o 'src=[^ ]*' "$gcr" | head -1 | cut -d= -f2)"; NREC[$n]="$(grep -o 'records=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"
  local rr; rr=($(resident_of 800))
  record "$cell" "$n" dispatch 800 "${rr[0]}" "${rr[1]}" "${XCTX:-?}" "${FREE:-?}" "$rep" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$start"
  [[ "$MATCH" == "1" && "${TOT:-}" == "$orc" ]] || { fail "oracle[$cell r$rep]" "total_sum='${TOT:-<none>}' expected $orc"; return 1; }
  add_mean "$cell" "$KMS"; XC[$cell]="$XCTX"; FM[$cell]="$FREE"
  info "$cell r$rep" "kernel_ms=$KMS extra_ctx=$XCTX free_mb=$FREE"
}
run_direct() {  # cell N mb extra_ctx rep
  local cell="$1" n="$2" mb="$3" xc="$4" rep="$5" orc; orc="$(oracle_of "$n")"
  [[ -f "${NIN[$n]:-/nonexistent}" ]] || { fail "input_located[$cell]" "no input for N=$n (dispatcher cell must run first)"; return 1; }
  gpu_gate "${cell}_r${rep}" || return 1
  local lg="$LOGDIR/3_${cell}_r${rep}.log" start; start="$(date -Is)"
  env -u NQ_PAD_MB -u NQ_CARVEOUT NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$mb" NQ_EXTRA_CTX="$xc" \
    "./$CU_BIN" "$n" "${NIN[$n]}" "/tmp/${REV}_out.bin" "$orc" > "$lg" 2>&1 || true
  fields_of "$lg"
  local rr; rr=($(resident_of "$mb"))
  record "$cell" "$n" direct "$mb" "${rr[0]}" "${rr[1]}" "${XCTX:-?}" "${FREE:-?}" "$rep" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$start"
  [[ "$MATCH" == "1" && "${TOT:-}" == "$orc" ]] || { fail "oracle[$cell r$rep]" "total_sum='${TOT:-<none>}' expected $orc"; return 1; }
  add_mean "$cell" "$KMS"; XC[$cell]="$XCTX"; FM[$cell]="$FREE"
  info "$cell r$rep" "mb=$mb warps/SM=${rr[0]} footprint=${rr[1]}KB kernel_ms=$KMS extra_ctx=$XCTX free_mb=$FREE"
}
holder_up() {
  "./$HOLDER_BIN" 0 900 > "$LOGDIR/holder_$1.log" 2>&1 & HPID=$!; sleep 4; cat "$LOGDIR/holder_$1.log"
}
holder_down() { [[ -n "$HPID" ]] && { kill "$HPID" 2>/dev/null; wait "$HPID" 2>/dev/null || true; }; HPID=""; sleep 2; }
bail() { holder_down; echo "OK=$PASS FAIL=$FAIL"; exit 1; }

# ---------------------------------------------------------------------
# 4. Phase B -- N=20, three paths interleaved
# ---------------------------------------------------------------------
if [[ "$SKIP_B" != "1" ]]; then
  banner "Phase B: N=$NB  Gd (dispatcher) / Dx2 (direct, ctx=2) / Dh (direct, ctx=1 + holder), x$REPS_B"
  for r in $(seq 1 "$REPS_B"); do
    run_dispatch Gd "$NB" "$r" || bail
    sleep "$COOLDOWN"
    run_direct Dx2 "$NB" "$MB_B" 2 "$r" || bail
    sleep "$COOLDOWN"
    holder_up "Dh_r$r"
    run_direct Dh "$NB" "$MB_B" 1 "$r" || bail
    holder_down
    sleep "$COOLDOWN"
  done
fi

# ---------------------------------------------------------------------
# 5. Phase A -- N=21, production anchor then the ladder around 800
# ---------------------------------------------------------------------
if [[ "$SKIP_A" != "1" ]]; then
  banner "Phase A: N=$NA  G21 anchor, then direct MB in {$MB21_LIST} x$REPS_A (NQ_EXTRA_CTX=$EXTRA_CTX_DIRECT)"
  run_dispatch G21 "$NA" 1 || bail
  sleep "$COOLDOWN"
  for r in $(seq 1 "$REPS_A"); do
    for mb in $MB21_LIST; do
      run_direct "D21_m${mb}" "$NA" "$mb" "$EXTRA_CTX_DIRECT" "$r" || bail
      sleep "$COOLDOWN"
    done
  done
fi
finish_mean

# ---------------------------------------------------------------------
# 6. Evaluation (R0..R5)
# ---------------------------------------------------------------------
banner "Evaluation"
if [[ "$SKIP_A" != "1" ]]; then
  g="${MEAN[G21]:-}"
  if [[ -n "$g" ]]; then
    d="$(abspct "$g" "$ANCHOR_G21")"
    { le "$d" 0.10 && [[ "${XC[G21]:-}" == "1" ]]; } && pass "R0_production_anchor (G21=$g ms, ${d}% from $ANCHOR_G21, extra_ctx=${XC[G21]:-?})" \
      || fail "R0_production_anchor" "G21=$g is ${d}% from $ANCHOR_G21 (extra_ctx=${XC[G21]:-?}) -- the session is not in the production state"
  fi
  b="${MEAN[D21_m${BASE_MB}]:-}"
  R1=0
  if [[ -n "$b" ]]; then
    d="$(abspct "$b" "$ANCHOR_D21")"
    if le "$d" 0.15 && [[ "${FM[D21_m${BASE_MB}]:-}" == "$FREE_MB_21" && "${XC[D21_m${BASE_MB}]:-}" == "$EXTRA_CTX_DIRECT" ]]; then
      pass "R1_direct_anchor_HARD (D21_m800=$b ms, ${d}% from $ANCHOR_D21, free_mb=${FM[D21_m${BASE_MB}]}, extra_ctx=${XC[D21_m${BASE_MB}]})"; R1=1
    else
      fail "R1_direct_anchor_HARD" "D21_m800=$b is ${d}% from $ANCHOR_D21, free_mb=${FM[D21_m${BASE_MB}]:-?} (want $FREE_MB_21), extra_ctx=${XC[D21_m${BASE_MB}]:-?} -- N=21 direct cells are NOT comparable; R2-R4 not read"
    fi
  fi
  if [[ "$R1" == "1" ]]; then
    for mb in $MB21_LIST; do
      [[ "$mb" == "$BASE_MB" ]] && continue
      v="${MEAN[D21_m${mb}]:-}"; [[ -z "$v" ]] && continue
      p="$(pct "$v" "$b")"; info "N=$NA MB=$mb vs $BASE_MB" "${p}%  ($v vs $b ms)"
      case "$mb" in
        1280)
          if ge "$p" 20; then pass "R2_M3_closed_axis_dead_outside_L1 (+${p}% at N=$NA, k=49.4 -- 401-r2 M4: do not write the packing FOR occupancy beyond L1)"
          elif ge "$p" 0; then fail "R2_penalty_under_20pct" "+${p}% -- smaller than registered; by 401-r2 M4 one more revision on this rung is justified before closing"
          else fail "R2_REFUTED_more_warps_wins" "${p}% -- MB=1280 is FASTER at N=$NA; 401's L5 was a debug-scale artefact"; fi ;;
        720)
          if ge "$p" 1.5 && le "$p" 6; then pass "R3_inside_L1_slope_positive (MB=720 is +${p}% slower: fewer warps inside L1 still costs; 402's premise holds at N=$NA)"
          elif ge "$p" 0.3; then pass "R3_inside_L1_slope_positive_but_small (+${p}%, below the +1.5% registered; the slope is flattening toward 10 warps)"
          else fail "R3_REFUTED_slope_flat_inside_L1" "${p}% -- 9 warps/SM is not slower than 10; more warps do not pay even inside L1 at this point. Re-plan 402 before writing the packing"; fi ;;
        880)
          if ge "$p" 0.3; then pass "R4_800_is_top_of_L1 (MB=880 +${p}% at 71.5 KB: budget the 104-byte frame against 65 KB -> 20 warps/SM, not 22)"
          elif ge "$p" -0.3; then pass "R4_flat_at_880 (${p}%: 71.5 KB is neither better nor worse within 0.3%; treat 800..880 as the plateau)"
          else fail "R4_REFUTED_880_faster" "${p}% -- 11 warps/SM beats 10 at N=$NA. 800 was not the top of L1. A/B candidate for a small revision; not adopted from one run"; fi ;;
      esac
    done
  fi
fi
if [[ "$SKIP_B" != "1" ]]; then
  gd="${MEAN[Gd]:-}"; dx="${MEAN[Dx2]:-}"; dh="${MEAN[Dh]:-}"
  if [[ -n "$gd" && -n "$dx" && -n "$dh" ]]; then
    gap_x="$(pct "$dx" "$gd")"; gap_h="$(pct "$dh" "$gd")"; gap_hx="$(pct "$dh" "$dx")"
    info "N=$NB" "Gd=$gd  Dx2=$dx (${gap_x}% vs Gd)  Dh=$dh (${gap_h}% vs Gd, ${gap_hx}% vs Dx2)"
    info "N=$NB free_mb/extra_ctx" "Gd ${FM[Gd]:-?}/${XC[Gd]:-?}  Dx2 ${FM[Dx2]:-?}/${XC[Dx2]:-?}  Dh ${FM[Dh]:-?}/${XC[Dh]:-?}"
    info "N=$NB vs 401-r2" "Gd was $ANCHOR_G20 ($(pct "$gd" "$ANCHOR_G20")%), Dx2 was $ANCHOR_D20 ($(pct "$dx" "$ANCHOR_D20")%)"
    ax="$(abspct "$dx" "$gd")"; ah="$(abspct "$dh" "$gd")"; ahx="$(abspct "$dh" "$dx")"
    if le "$ax" 0.1 && le "$ah" 0.1; then
      fail "R5_REFUTED_all_three_agree" "gaps ${gap_x}% / ${gap_h}% -- 401-r2's 17,378 at N=$NB was a session artefact; V4 holds at N=$NB too"
    elif ge "$gap_x" 3; then
      if le "$ah" 0.3; then pass "R5_gap_reproduces_holder_sides_with_dispatcher (Dx2 +${gap_x}%, Dh ${gap_h}% vs Gd) -- at N=$NB an in-process cuCtxCreate context is NOT equivalent to another process's; 395c-r4's M1=D1 was N=21-specific. RULE: debug-scale direct cells use the holder, not NQ_EXTRA_CTX=2"
      elif le "$ahx" 0.3; then pass "R5_gap_reproduces_holder_sides_with_direct (Dx2 +${gap_x}%, Dh +${gap_h}% vs Gd) -- both direct methods are slow; the dispatcher path carries something beyond context count and free_mb. RULE: debug-scale timing goes through -g until that is found"
      else pass "R5_gap_reproduces_three_way (Dx2 +${gap_x}%, Dh +${gap_h}% vs Gd) -- three distinct states; context ownership matters at N=$NB in a way it did not at N=21"; fi
    else
      fail "R5_partial" "Dx2 is ${gap_x}% vs Gd (registered >= +3%), Dh ${gap_h}% -- the gap is smaller than r2 measured; report, do not conclude"
    fi
  fi
fi

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys, statistics as st
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
by={}
for r in rows: by.setdefault(r['cell'],[]).append(r)
print(f"{'cell':<10}{'N':>3}{'path':>10}{'MB':>6}{'w/SM':>6}{'KB':>8}{'ctx':>5}{'free_mb':>9}{'n':>3}{'mean ms':>13}{'spread':>9}")
for c,rs in by.items():
    v=[float(r['kernel_ms']) for r in rs]; m=st.fmean(v); sp=(max(v)-min(v))/m*100 if len(v)>1 else 0.0
    r=rs[0]; print(f"{c:<10}{r['N']:>3}{r['path']:>10}{r['max_blocks']:>6}{r['warps_per_sm']:>6}{r['footprint_kb']:>8}{r['extra_ctx']:>5}{r['free_mb']:>9}{len(v):>3}{m:>13.3f}{sp:>8.3f}%")
n21={c:st.fmean([float(r['kernel_ms']) for r in rs]) for c,rs in by.items() if c.startswith('D21_m')}
if 'D21_m800' in n21:
    b=n21['D21_m800']; print("\n=== N=21 ladder vs MB=800 ===")
    for c in sorted(n21, key=lambda x:int(x[5:])): print(f"  MB={c[5:]:>5}  {n21[c]:>12.3f} ms  {(n21[c]-b)/b*100:>+8.3f}%")
EOF

{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs" 2>/dev/null || true
TARBALL="${LOGDIR}.tar.gz"
tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; echo "results: $TSV"; echo "tarball: $TARBALL"
if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS (some are registered refutations, not errors -- read the label):"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
echo "${REV} PASSED. Send $TARBALL for analysis."
exit 0
