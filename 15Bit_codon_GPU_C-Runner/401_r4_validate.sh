#!/usr/bin/env bash
# 401_r4_validate.sh
#
# rev401-r4 -- remeasure what 401-r3 could not read. The GPU state is made by
#              an EXTERNAL context holder (395c_ctx_holder), never by
#              NQ_EXTRA_CTX. Zero code change.
#
# WHY (401-r3, 2026-09-28)
#   Direct N=21 MB=800 with NQ_EXTRA_CTX=2 gave 139,181 ms = 395c's D0
#   (one context, 139,188) to 0.005%, although the log said extra_ctx=2 and
#   free_mb=21762. At N=20 the same method was +5.44% while an external
#   holder matched the dispatcher to 0.008%. So an in-process cuCtxCreate
#   context currently does not count for the ~4% effect; another process's
#   context does. r3's ladder was therefore measured in the wrong state.
#
# CELLS (all N=21; holder = 395c_ctx_holder 0 MB; MB=800 unless noted)
#   G21   -g 21 21                          production anchor
#   H0    1 holder,  NQ_EXTRA_CTX=0         2 ctx, free_mb 22018  <- anchor
#   H1    1 holder,  NQ_EXTRA_CTX=1         does an in-process ctx add?
#   H2    2 holders, NQ_EXTRA_CTX=0         3 real ctx, free_mb 21762
#   L720 L800 L880 L1280   1 holder, ctx=0  the ladder, in the right state
#   nvidia-smi SM clock / power / temperature sampled every 5 s per cell.
#
# PRE-REGISTERED (401_r4_README_append.md; fixed before execution)
#   P0  G21 within +-0.15% of 133,567, extra_ctx=1, free_mb 21762.
#   P1  HARD: H0 within +-0.15% of 133,561, free_mb 22018+-2, extra_ctx=0.
#   P2  stated: |H1 - H0| <= 0.10%.   Alt: H1 <= H0 - 0.25%.
#   P3  stated (weak): H2 <= H0 - 0.25%.   Refuted if within 0.10%.
#   P4  L1280 >= H0 + 20%.
#   P5  L720 vs L800 in +1.5..+6%.  Refuted if <= +0.3%.
#   P6  L880 slower than L800.  Faster by >0.3% -> A/B candidate only.
#   P7  |L800 - H0| <= 0.05%.
#   P8  mean SM clock within 2% of 1710 in every cell.
#
# USAGE
#   STATIC_ONLY=1 bash 401_r4_validate.sh        # OK=14
#                 bash 401_r4_validate.sh        # ~24 min
#   CELLS="G21 H0 H1 H2" bash 401_r4_validate.sh # state probes only (~10 min)
#   CELLS="H0 L720 L800 L880 L1280" bash 401_r4_validate.sh   # ladder only

set -u

REV="401_r4"
PY_SRC="${PY_SRC:-401_r4Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-401_r4Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-401_r3Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-401_r4_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-401_r4_kernel_maxd14}"
PREV_CU="${PREV_CU:-401_r3_kernel_maxd14.cu}"
HOLDER_SRC="${HOLDER_SRC:-395c_ctx_holder.cu}"
HOLDER_BIN="${HOLDER_BIN:-395c_ctx_holder}"
CRLOG_DIR="${CRLOG_DIR:-401_r4_crunner_logs}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
BLOCK="${BLOCK:-32}"
SMS="${SMS:-80}"
MAX_BLOCKS_PER_SM="${MAX_BLOCKS_PER_SM:-16}"
FRAME_B="${FRAME_B:-208}"
CELLS="${CELLS:-G21 H0 H1 H2 L720 L800 L880 L1280}"
ANCHOR_G21="${ANCHOR_G21:-133566.969}"    # 401-r3 G21 (today's production state)
ANCHOR_H0="${ANCHOR_H0:-133561.062}"      # 395c-r4 D1: holder + CRunner, 2 ctx
FREE_H0="${FREE_H0:-22018}"
FREE_3CTX="${FREE_3CTX:-21762}"
KERNEL_SHA_395A="${KERNEL_SHA_395A:-ebd7f523deb591347eab1a352a9555c2e4a6c0b4a456d80517322c933a86e68a}"
COOLDOWN="${COOLDOWN:-10}"
CLK_INTERVAL="${CLK_INTERVAL:-5}"
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
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f",(a-b)/b*100}'; }
abspct() { awk -v a="$1" -v b="$2" 'BEGIN{x=(a-b)/b*100; printf "%.3f",(x<0?-x:x)}'; }
le() { awk -v x="$1" -v y="$2" 'BEGIN{exit !(x<=y)}'; }
ge() { awk -v x="$1" -v y="$2" 'BEGIN{exit !(x>=y)}'; }
absdiff_le() { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=t)}'; }

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
grep -qE '^REV_TAG:str="401_r4"' "$CODE" && pass "source_rev_tag_is_401_r4" || fail "source_rev_tag_is_401_r4" "wrong REV_TAG"
grep -q 'CRunnerEntry(14,"./401_r4_kernel_maxd14","NQ_EXTRA_CTX=1 "' "$CODE" && pass "source_table_points_at_401_r4_and_keeps_the_treatment" || fail "source_table_points_at_401_r4_and_keeps_the_treatment" "table entry wrong"
grep -qE '^CRUNNER_MIN_N:int=19' "$CODE" && pass "source_crunner_min_n_still_19" || fail "source_crunner_min_n_still_19" "CRUNNER_MIN_N lost"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_401_r3Py (removed=3 added=3 EXECUTABLE lines)" \
    || { fail "py_diff_fingerprint_vs_401_r3Py" "removed=$PR_ added=$PA_, expected 3/3"; diff "/tmp/${REV}_prev_code.py" "$CODE" | head -20 | sed 's/^/      /'; }
else info "py_diff_fingerprint_vs_401_r3Py" "skipped"; fi
cu_code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA_395A" ]] && pass "cu_kernel_region_still_395a_r2 (${KB:0:16}...)" || fail "cu_kernel_region_still_395a_r2" "got ${KB:0:16}..."
if [[ -f "$PREV_CU" ]]; then
  cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  CA="$(sha256sum < "/tmp/${REV}_prev_code.cu" | cut -d' ' -f1)"; CB="$(sha256sum < "/tmp/${REV}_cur_code.cu" | cut -d' ' -f1)"
  [[ "$CA" == "$CB" ]] && pass "cu_whole_code_region_identical_to_401_r3 (${CB:0:16}...)" || fail "cu_whole_code_region_identical_to_401_r3" "401-r4 must be a rename"
else info "cu_whole_code_region_identical_to_401_r3" "skipped"; fi
grep -q "rev401-r4" "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev401-r4 note in header"

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
printf 'cell\tN\tpath\tmax_blocks\twarps_per_sm\tfootprint_kb\tholders\tnq_extra_ctx\textra_ctx\tfree_mb\tkernel_ms\ttotal_sum\tmatch\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
declare -a HPIDS=()
CLKPID=""
gpu_gate() {  # $1 cell. Only our holders may be on the GPU.
  local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"
  echo "$apps" > "$LOGDIR/apps_before_$1.txt"
  local allowed=" ${HPIDS[*]:-} "; local others=""
  while IFS=, read -r pid mem; do
    [[ -z "$pid" ]] && continue
    [[ "$allowed" == *" $pid "* ]] || others+="$pid($mem) "
  done <<< "$apps"
  if [[ -n "$others" ]]; then
    fail "gpu_empty_before[$1]" "foreign compute process(es): $others"
    echo "ABORT: a leftover process moves this kernel by 4-6% (395c). Clear it and rerun."; return 1
  fi
  return 0
}
clk_start() {  # $1 cell
  local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"
  ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null &
  CLKPID=$!
}
clk_stop() {  # $1 cell -> SMMEAN SMMIN TMAX
  [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.0f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_$1.tsv")
}
fields_of() {  # $1 log
  KMS="$(grep -o 'kernel_ms=[0-9.]*' "$1" | head -1 | cut -d= -f2)"
  TOT="$(grep -o 'total_sum=[0-9]*' "$1" | head -1 | cut -d= -f2)"
  MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$1" && MATCH=1
  XCTX="$(grep -o 'extra_ctx=[0-9]*' "$1" | head -1 | cut -d= -f2)"
  FREE="$(grep -o 'free_mb=[0-9]*' "$1" | head -1 | cut -d= -f2)"
}
resident_of() { python3 -c "
mb=int('$1'); blk=$BLOCK; sms=$SMS; capb=$MAX_BLOCKS_PER_SM; fr=$FRAME_B
bps=min(mb//sms, capb); w=bps*(blk//32); print(w, round(w*32*fr/1024,1))"; }
record() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$TSV"; }
declare -A VAL XC FM SMM
IN=""; ORC="$(oracle_of "$NQ")"
holders_up() {  # $1 count $2 cell
  local i; for i in $(seq 1 "$1"); do
    "./$HOLDER_BIN" 0 900 > "$LOGDIR/holder_$2_$i.log" 2>&1 & HPIDS+=("$!")
  done
  sleep 4; cat "$LOGDIR"/holder_"$2"_*.log
}
holders_down() { local p; for p in "${HPIDS[@]:-}"; do [[ -n "$p" ]] && { kill "$p" 2>/dev/null; wait "$p" 2>/dev/null || true; }; done; HPIDS=(); sleep 2; }
bail() { holders_down; [[ -n "$CLKPID" ]] && kill "$CLKPID" 2>/dev/null; echo "OK=$PASS FAIL=$FAIL"; exit 1; }
locate_input() {  # from a crunner log, never guessed
  local gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${NQ}.log"
  [[ -f "$gcr" ]] && IN="$(grep -o 'src=[^ ]*' "$gcr" | head -1 | cut -d= -f2)"
  if [[ ! -f "${IN:-/nonexistent}" ]]; then
    # no r4 log yet: borrow the path from the newest earlier crunner log, then verify size (2,025,282 x 28 B)
    local any; any="$(ls -t 401_r*_crunner_logs/crunner_*_N${NQ}.log 2>/dev/null | head -1)"
    [[ -n "$any" ]] && IN="$(grep -o 'src=[^ ]*' "$any" | head -1 | cut -d= -f2)"
  fi
  [[ -f "${IN:-/nonexistent}" ]] && [[ "$(stat -c %s "$IN")" == "56707896" ]] && { pass "input_located (${IN})"; return 0; }
  return 1
}
run_dispatch() {  # cell
  local cell="$1" gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${NQ}.log"; rm -f "$gcr"
  gpu_gate "$cell" || return 1
  local start; start="$(date -Is)"; clk_start "$cell"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX -u NQ_BLOCK -u NQ_CARVEOUT "./$PY_BIN" -g "$NQ" "$NQ" > "$LOGDIR/2_${cell}_console.log" 2>&1
  clk_stop "$cell"
  cp "$gcr" "$LOGDIR/2_${cell}_crunner.log" 2>/dev/null || { fail "crunner_path_taken[$cell]" "no $gcr"; return 1; }
  fields_of "$gcr"; local rr; rr=($(resident_of 800))
  record "$cell" "$NQ" dispatch 800 "${rr[0]}" "${rr[1]}" 0 - "${XCTX:-?}" "${FREE:-?}" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start"
  [[ "$MATCH" == "1" && "${TOT:-}" == "$ORC" ]] || { fail "oracle[$cell]" "total_sum='${TOT:-<none>}' expected $ORC"; return 1; }
  VAL[$cell]="$KMS"; XC[$cell]="$XCTX"; FM[$cell]="$FREE"; SMM[$cell]="$SMMEAN"
  info "$cell" "kernel_ms=$KMS extra_ctx=$XCTX free_mb=$FREE sm_mean=$SMMEAN sm_min=$SMMIN temp_max=$TMAX"
}
run_direct() {  # cell mb holders nq_extra_ctx
  local cell="$1" mb="$2" nh="$3" xc="$4"
  holders_up "$nh" "$cell"
  gpu_gate "$cell" || { holders_down; return 1; }
  local lg="$LOGDIR/3_${cell}.log" start; start="$(date -Is)"; clk_start "$cell"
  env -u NQ_PAD_MB -u NQ_CARVEOUT NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$mb" NQ_EXTRA_CTX="$xc" \
    "./$CU_BIN" "$NQ" "$IN" "/tmp/${REV}_out.bin" "$ORC" > "$lg" 2>&1 || true
  clk_stop "$cell"; holders_down
  fields_of "$lg"; local rr; rr=($(resident_of "$mb"))
  record "$cell" "$NQ" direct "$mb" "${rr[0]}" "${rr[1]}" "$nh" "$xc" "${XCTX:-?}" "${FREE:-?}" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start"
  [[ "$MATCH" == "1" && "${TOT:-}" == "$ORC" ]] || { fail "oracle[$cell]" "total_sum='${TOT:-<none>}' expected $ORC"; return 1; }
  VAL[$cell]="$KMS"; XC[$cell]="$XCTX"; FM[$cell]="$FREE"; SMM[$cell]="$SMMEAN"
  info "$cell" "mb=$mb holders=$nh NQ_EXTRA_CTX=$xc -> extra_ctx=$XCTX free_mb=$FREE kernel_ms=$KMS sm_mean=$SMMEAN sm_min=$SMMIN temp_max=$TMAX"
}

# ---------------------------------------------------------------------
# 4. Cells
# ---------------------------------------------------------------------
for cell in $CELLS; do
  banner "Cell $cell"
  case "$cell" in
    G21)   run_dispatch G21 || bail; locate_input || { fail "input_located" "no sched input after -g"; bail; } ;;
    *)     [[ -f "${IN:-/nonexistent}" ]] || locate_input || { fail "input_located" "run G21 first (or keep an earlier 401_r*_crunner_logs)"; bail; } ;;
  esac
  case "$cell" in
    G21)   : ;;
    H0)    run_direct H0 800 1 0 || bail ;;
    H1)    run_direct H1 800 1 1 || bail ;;
    H2)    run_direct H2 800 2 0 || bail ;;
    L*)    run_direct "$cell" "${cell#L}" 1 0 || bail ;;
    *)     fail "unknown_cell" "$cell"; bail ;;
  esac
  sleep "$COOLDOWN"
done

# ---------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------
banner "Evaluation"
if [[ -n "${VAL[G21]:-}" ]]; then
  d="$(abspct "${VAL[G21]}" "$ANCHOR_G21")"
  { le "$d" 0.15 && [[ "${XC[G21]}" == "1" && "${FM[G21]}" == "$FREE_3CTX" ]]; } && pass "P0_production_anchor (G21=${VAL[G21]}, ${d}% from $ANCHOR_G21)" \
    || fail "P0_production_anchor" "G21=${VAL[G21]} is ${d}% from $ANCHOR_G21, extra_ctx=${XC[G21]}, free_mb=${FM[G21]}"
fi
P1=0
if [[ -n "${VAL[H0]:-}" ]]; then
  d="$(abspct "${VAL[H0]}" "$ANCHOR_H0")"
  if le "$d" 0.15 && absdiff_le "${FM[H0]}" "$FREE_H0" 2 && [[ "${XC[H0]}" == "0" ]]; then
    pass "P1_HARD_2ctx_anchor (H0=${VAL[H0]}, ${d}% from $ANCHOR_H0, free_mb=${FM[H0]}, extra_ctx=0)"; P1=1
  else
    fail "P1_HARD_2ctx_anchor" "H0=${VAL[H0]} is ${d}% from $ANCHOR_H0, free_mb=${FM[H0]} (want $FREE_H0), extra_ctx=${XC[H0]} -- the holder does not reproduce the 2-context state; nothing below is read"
  fi
fi
if [[ "$P1" == "1" ]]; then
  h0="${VAL[H0]}"
  if [[ -n "${VAL[H1]:-}" ]]; then
    p="$(pct "${VAL[H1]}" "$h0")"; a="$(abspct "${VAL[H1]}" "$h0")"
    if le "$a" 0.10; then pass "P2_in_process_ctx_adds_nothing (H1 ${p}% vs H0, free_mb=${FM[H1]}, extra_ctx=${XC[H1]}) -- NQ_EXTRA_CTX is inert now; keep env_prefix for the record only"
    elif le "$p" -0.25; then fail "P2_REFUTED_in_process_ctx_counts_again" "H1 ${p}% vs H0 -- the effect is time-varying; r3's D21 anomaly must be re-examined"
    else fail "P2_indeterminate" "H1 ${p}% vs H0 -- between the two registered outcomes; report, do not conclude"; fi
  fi
  if [[ -n "${VAL[H2]:-}" ]]; then
    p="$(pct "${VAL[H2]}" "$h0")"; a="$(abspct "${VAL[H2]}" "$h0")"
    if le "$p" -0.25; then pass "P3_notch_exists_with_a_real_3rd_ctx (H2=${VAL[H2]}, ${p}% vs H0, free_mb=${FM[H2]}) -- the r6 gain is recoverable by an external helper; A/B on the -g path in a later revision"
    elif le "$a" 0.10; then fail "P3_REFUTED_notch_gone" "H2 ${p}% vs H0 at free_mb=${FM[H2]} -- a real 3rd context does nothing either; 133,56x is the production number"
    else fail "P3_indeterminate" "H2 ${p}% vs H0 -- report, do not conclude"; fi
  fi
  l8="${VAL[L800]:-}"
  if [[ -n "$l8" ]]; then
    a="$(abspct "$l8" "$h0")"
    le "$a" 0.05 && pass "P7_replicate_noise_floor (L800 vs H0 ${a}%)" || fail "P7_replicate_noise_floor" "L800 vs H0 ${a}% (> 0.05%): the session drifted; read the ladder with that in mind"
  fi
  base="${l8:-$h0}"
  if [[ -n "${VAL[L1280]:-}" ]]; then
    p="$(pct "${VAL[L1280]}" "$h0")"; info "L1280 vs H0" "${p}%"
    if ge "$p" 20; then pass "P4_M3_closed_axis_dead_outside_L1 (+${p}% at N=$NQ, 16 warps/SM, k=49.4)"
    elif ge "$p" 0; then fail "P4_penalty_under_20pct" "+${p}% -- smaller than registered; 401-r2 M4 grants one more revision on this rung"
    else fail "P4_REFUTED_more_warps_wins" "${p}%"; fi
  fi
  if [[ -n "${VAL[L720]:-}" ]]; then
    p="$(pct "${VAL[L720]}" "$base")"; info "L720 vs L800" "${p}%"
    if ge "$p" 1.5 && le "$p" 6; then pass "P5_inside_L1_slope_positive (L720 +${p}%: 9 warps/SM costs; 402's premise holds)"
    elif ge "$p" 0.3; then pass "P5_slope_positive_but_small (+${p}%, under the +1.5% registered)"
    else fail "P5_REFUTED_slope_flat_inside_L1" "${p}% -- 9 warps/SM is not slower than 10; re-plan 402 before writing the packing"; fi
  fi
  if [[ -n "${VAL[L880]:-}" ]]; then
    p="$(pct "${VAL[L880]}" "$base")"; info "L880 vs L800" "${p}%"
    if ge "$p" 0.3; then pass "P6_800_is_top_of_L1 (L880 +${p}% at 71.5 KB -> budget the 104-byte frame against 65 KB: 20 warps/SM)"
    elif ge "$p" -0.3; then pass "P6_flat_at_880 (${p}%)"
    else fail "P6_REFUTED_880_faster" "${p}% -- A/B candidate; not adopted from one run"; fi
  fi
fi
CLK_OK=1
for c in $CELLS; do
  m="${SMM[$c]:-}"; [[ -z "$m" || "$m" == "?" ]] && continue
  absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM clock $m MHz"; }
done
[[ "$CLK_OK" == "1" ]] && pass "P8_sm_clock_1710_all_cells" || fail "P8_sm_clock_1710_all_cells" "at least one cell ran off 1710 MHz -- see clk_*.tsv; a clock effect is mixed in"

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
print(f"{'cell':<7}{'path':>9}{'MB':>6}{'w/SM':>5}{'KB':>7}{'hold':>5}{'NQ_X':>5}{'ctx':>4}{'free_mb':>8}{'kernel_ms':>13}{'sm_mean':>8}{'sm_min':>7}{'tmax':>5}")
for r in rows:
    print(f"{r['cell']:<7}{r['path']:>9}{r['max_blocks']:>6}{r['warps_per_sm']:>5}{r['footprint_kb']:>7}{r['holders']:>5}{r['nq_extra_ctx']:>5}{r['extra_ctx']:>4}{r['free_mb']:>8}{float(r['kernel_ms']):>13.3f}{r['sm_mean']:>8}{r['sm_min']:>7}{r['temp_max']:>5}")
v={r['cell']:float(r['kernel_ms']) for r in rows}
if 'H0' in v:
    print("\n=== vs H0 (1 holder, 2 contexts) ===")
    for c in rows:
        if c['cell']!='H0': print(f"  {c['cell']:<7}{v[c['cell']]:>13.3f}  {(v[c['cell']]-v['H0'])/v['H0']*100:>+8.3f}%")
EOF

{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs" 2>/dev/null || true
TARBALL="${LOGDIR}.tar.gz"
tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; echo "results: $TSV"; echo "tarball: $TARBALL"
if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS (registered refutations are listed here too -- read the label):"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
echo "${REV} PASSED. Send $TARBALL for analysis."
exit 0
