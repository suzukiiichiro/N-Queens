#!/usr/bin/env bash
# 402_r2_validate.sh
#
# rev402-r2 -- promote 402's winner by interleaved A/B, then measure the
#              production path (-g 21 21) with A10G_FINAL_DEFAULT_MAX_BLOCKS=960.
#              .cu is a rename of 402 (byte-identical code region).
#
# CELLS (direct: 1 holder, NQ_EXTRA_CTX=0, N=21, 402-r2 binary)
#   round 1: A800 A880 A960 A1040      round 2: A1040 A960 A880 A800
#   then G21: ./402_r2Py_kernel_maxd14_final -g 21 21 (dispatcher, default 960)
# PRE-REGISTERED (402_r2_README_append.md; fixed before execution)
#   Q1  HARD: mean A800 within +-0.15% of 129,524 (402 E1), free_mb 22018
#   Q2  each config's two runs within 0.05%
#   Q3  stated: A960 minimum; A880 between A800 and A960; A1040 +-0.3% of 116,102
#   Q4  ADOPTION: winner <= A800 - 10% and Q2 holds. Winner 960 -> this .py stands.
#       Winner 880/1040 -> adopted, default changed in 402-r3.
#   Q5  G21 within +-0.30% of the winner's mean (if winner is 960), MAX_BLOCKS=960
#       in the CRunner log, NQ_MAX_BLOCKS=960 in dispatch.log
#   Q6  mean SM clock within 2% of 1710 in every cell
#
# USAGE
#   STATIC_ONLY=1 bash 402_r2_validate.sh     # OK=16
#                 bash 402_r2_validate.sh     # ~22 min
#   ROUNDS=1      bash 402_r2_validate.sh     # single pass (Q2 not evaluable)
#   SKIP_G21=1    bash 402_r2_validate.sh

set -u

REV="402_r2"
PY_SRC="${PY_SRC:-402_r2Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-402_r2Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-402Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-402_r2_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-402_r2_kernel_maxd14}"
PREV_CU="${PREV_CU:-402_kernel_maxd14.cu}"
HOLDER_SRC="${HOLDER_SRC:-395c_ctx_holder.cu}"
HOLDER_BIN="${HOLDER_BIN:-395c_ctx_holder}"
CRLOG_DIR="${CRLOG_DIR:-402_r2_crunner_logs}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
BLOCK="${BLOCK:-32}"
SMS="${SMS:-80}"
MAX_BLOCKS_PER_SM="${MAX_BLOCKS_PER_SM:-16}"
FRAME_B="${FRAME_B:-160}"
MB_LIST="${MB_LIST:-800 880 960 1040}"
ROUNDS="${ROUNDS:-2}"
SKIP_G21="${SKIP_G21:-0}"
DEFAULT_MB="${DEFAULT_MB:-960}"
ANCHOR_A800="${ANCHOR_A800:-129523.508}"   # 402 E1
ANCHOR_A960="${ANCHOR_A960:-111997.031}"   # 402 P960
ANCHOR_A1040="${ANCHOR_A1040:-116102.156}" # 402 P1040
FREE_2CTX="${FREE_2CTX:-22018}"
KERNEL_SHA_402="${KERNEL_SHA_402:-}"       # optional pin; the rename gate below is what matters
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
oracle_of() { case "$1" in 19) echo 4968057848;; 20) echo 39029188884;; 21) echo 314666222712;; 22) echo 2691008701644;; *) echo "";; esac; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f",(a-b)/b*100}'; }
abspct() { awk -v a="$1" -v b="$2" 'BEGIN{x=(a-b)/b*100; printf "%.3f",(x<0?-x:x)}'; }
le() { awk -v x="$1" -v y="$2" 'BEGIN{exit !(x<=y)}'; }
ge() { awk -v x="$1" -v y="$2" 'BEGIN{exit !(x>=y)}'; }
absdiff_le() { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=t)}'; }
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS (registered refutations are listed here too -- read the label):"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi; exit 0; }

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
grep -qE '^REV_TAG:str="402_r2"' "$CODE" && pass "source_rev_tag_is_402_r2" || fail "source_rev_tag_is_402_r2" "wrong REV_TAG"
grep -q 'CRunnerEntry(14,"./402_r2_kernel_maxd14","NQ_EXTRA_CTX=1 "' "$CODE" && pass "source_table_points_at_402_r2_and_keeps_the_treatment" || fail "source_table_points_at_402_r2_and_keeps_the_treatment" "table entry wrong"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=${DEFAULT_MB}\b" "$CODE" && pass "source_default_max_blocks_is_${DEFAULT_MB}" || fail "source_default_max_blocks_is_${DEFAULT_MB}" "A10G_FINAL_DEFAULT_MAX_BLOCKS is not $DEFAULT_MB"
grep -qE '^CRUNNER_MIN_N:int=19' "$CODE" && pass "source_crunner_min_n_still_19" || fail "source_crunner_min_n_still_19" "CRUNNER_MIN_N lost"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "4" && "$PA_" == "4" ]] && pass "py_diff_fingerprint_vs_402Py (removed=4 added=4 EXECUTABLE lines: tags, table, MAX_BLOCKS default)" \
    || { fail "py_diff_fingerprint_vs_402Py" "removed=$PR_ added=$PA_, expected 4/4"; diff "/tmp/${REV}_prev_code.py" "$CODE" | head -20 | sed 's/^/      /'; }
else info "py_diff_fingerprint_vs_402Py" "skipped"; fi
cu_code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
if [[ -f "$PREV_CU" ]]; then
  cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  CA="$(sha256sum < "/tmp/${REV}_prev_code.cu" | cut -d' ' -f1)"; CB="$(sha256sum < "/tmp/${REV}_cur_code.cu" | cut -d' ' -f1)"
  [[ "$CA" == "$CB" ]] && pass "cu_whole_code_region_identical_to_402 (${CB:0:16}...)" || fail "cu_whole_code_region_identical_to_402" "402-r2 must be a rename"
else info "cu_whole_code_region_identical_to_402" "skipped"; fi
{ grep -q 'uint64_t stack_a\[MAXD14_ANCESTOR\];' "/tmp/${REV}_cur_code.cu" && grep -q 'uint64_t stack_depth = 0;' "/tmp/${REV}_cur_code.cu"; } && pass "cu_12B_frame_present" || fail "cu_12B_frame_present" "the 402 packing is missing"
[[ "$(grep -c 'if (N > PACK402_WIDTH) {' "/tmp/${REV}_cur_code.cu")" == "2" ]] && pass "cu_N_gt_21_refused_in_both_mains" || fail "cu_N_gt_21_refused_in_both_mains" "guard count != 2"
grep -q "rev402-r2" "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev402-r2 note in header"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Build
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
banner "Building"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc.log" 2>&1; grep -A3 'kernel_dfs_iter_gpu_maxd14' "$LOGDIR/05_nvcc.log" | grep -o '[0-9]* bytes stack frame.*\|Used [0-9]* registers' | head -2
[[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "no binary"; summary_exit; }
FR="$(grep -A3 'kernel_dfs_iter_gpu_maxd14' "$LOGDIR/05_nvcc.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"
[[ "$FR" == "160" ]] && pass "ptxas_frame_160B" || fail "ptxas_frame_160B" "frame=${FR:-?} (402 built to 160)"
if [[ ! -x "$HOLDER_BIN" ]]; then "$NVCC" -O2 -arch="$ARCH" -o "$HOLDER_BIN" "$HOLDER_SRC" > "$LOGDIR/05a_nvcc_holder.log" 2>&1; fi
[[ -x "$HOLDER_BIN" ]] && pass "ctx_holder_available[$HOLDER_BIN]" || { fail "ctx_holder_available" "no $HOLDER_BIN"; summary_exit; }
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/06_codon.log"
[[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN]" || { fail "codon_build" "see log"; summary_exit; }

# ---------------------------------------------------------------------
# 3. Input (from a crunner log, never guessed)
# ---------------------------------------------------------------------
IN=""; ORC="$(oracle_of "$NQ")"
for lg in $(ls -t 40*_crunner_logs/crunner_*_N${NQ}.log 2>/dev/null); do
  c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { IN="$c"; break; }
done
[[ -f "${IN:-/nonexistent}" && "$(stat -c %s "$IN")" == "56707896" ]] && pass "input_located ($IN)" || { fail "input_located" "no sched input via 40*_crunner_logs"; summary_exit; }

# ---------------------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------------------
printf 'cell\tround\tpath\tmax_blocks\twarps_per_sm\tfootprint_kb\textra_ctx\tfree_mb\tkernel_ms\ttotal_sum\tmatch\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
declare -a HPIDS=(); CLKPID=""
gpu_gate() {
  local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"
  echo "$apps" > "$LOGDIR/apps_before_$1.txt"
  local allowed=" ${HPIDS[*]:-} "; local others=""
  while IFS=, read -r pid mem; do [[ -z "$pid" ]] && continue; [[ "$allowed" == *" $pid "* ]] || others+="$pid($mem) "; done <<< "$apps"
  [[ -z "$others" ]] && return 0
  fail "gpu_empty_before[$1]" "foreign compute process(es): $others"; return 1
}
clk_start() { local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"
  ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPID=$!; }
clk_stop() { [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.0f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_$1.tsv"); }
fields_of() { KMS="$(grep -o 'kernel_ms=[0-9.]*' "$1" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$1" | head -1 | cut -d= -f2)"
  MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$1" && MATCH=1; XCTX="$(grep -o 'extra_ctx=[0-9]*' "$1" | head -1 | cut -d= -f2)"; FREE="$(grep -o 'free_mb=[0-9]*' "$1" | head -1 | cut -d= -f2)"
  CFGMB="$(grep -o 'MAX_BLOCKS=[0-9]*' "$1" | head -1 | cut -d= -f2)"; }
resident_of() { python3 -c "
mb=int('$1'); blk=$BLOCK; sms=$SMS; capb=$MAX_BLOCKS_PER_SM; fr=$FRAME_B
bps=min(mb//sms, capb); w=bps*(blk//32); print(w, round(w*32*fr/1024,1))"; }
holders_up() { local i; for i in $(seq 1 "$1"); do "./$HOLDER_BIN" 0 900 > "$LOGDIR/holder_$2_$i.log" 2>&1 & HPIDS+=("$!"); done; sleep 4; }
holders_down() { local p; for p in "${HPIDS[@]:-}"; do [[ -n "$p" ]] && { kill "$p" 2>/dev/null; wait "$p" 2>/dev/null || true; }; done; HPIDS=(); sleep 2; }
bail() { holders_down; [[ -n "$CLKPID" ]] && kill "$CLKPID" 2>/dev/null; summary_exit; }
declare -A SUM CNT VALS FM SMM MINV MAXV
acc() { local k="$1" v="$2"; SUM[$k]="$(awk -v s="${SUM[$k]:-0}" -v x="$v" 'BEGIN{printf "%.3f",s+x}')"; CNT[$k]=$(( ${CNT[$k]:-0}+1 )); VALS[$k]="${VALS[$k]:-}$v "
  [[ -z "${MINV[$k]:-}" ]] || le "$v" "${MINV[$k]}" && MINV[$k]="$v"; [[ -z "${MAXV[$k]:-}" ]] || ge "$v" "${MAXV[$k]}" && MAXV[$k]="$v"; }
mean_of() { awk -v s="${SUM[$1]}" -v n="${CNT[$1]}" 'BEGIN{printf "%.3f",s/n}'; }
spread_of() { awk -v a="${MAXV[$1]}" -v b="${MINV[$1]}" -v s="${SUM[$1]}" -v n="${CNT[$1]}" 'BEGIN{m=s/n; printf "%.3f",(a-b)/m*100}'; }
run_direct() {  # cell mb round
  local cell="$1" mb="$2" rd="$3"
  holders_up 1 "${cell}_r${rd}"; gpu_gate "${cell}_r${rd}" || { holders_down; return 1; }
  local lg="$LOGDIR/3_${cell}_r${rd}.log" start; start="$(date -Is)"; clk_start "${cell}_r${rd}"
  env -u NQ_PAD_MB -u NQ_CARVEOUT NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$mb" NQ_EXTRA_CTX=0 "./$CU_BIN" "$NQ" "$IN" "/tmp/${REV}_out.bin" "$ORC" > "$lg" 2>&1 || true
  clk_stop "${cell}_r${rd}"; holders_down
  fields_of "$lg"; local rr; rr=($(resident_of "$mb"))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$rd" direct "$mb" "${rr[0]}" "${rr[1]}" "${XCTX:-?}" "${FREE:-?}" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start" >> "$TSV"
  [[ "$MATCH" == "1" && "${TOT:-}" == "$ORC" ]] || { fail "oracle[$cell r$rd]" "total_sum='${TOT:-<none>}' expected $ORC"; return 1; }
  acc "$cell" "$KMS"; FM[$cell]="$FREE"; SMM["${cell}_r${rd}"]="$SMMEAN"
  info "$cell r$rd" "mb=$mb warps/SM=${rr[0]} footprint=${rr[1]}KB kernel_ms=$KMS free_mb=$FREE sm_mean=$SMMEAN temp_max=$TMAX"
}
run_g21() {
  local gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${NQ}.log"; rm -f "$gcr"
  gpu_gate G21 || return 1
  local start; start="$(date -Is)"; clk_start G21
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX -u NQ_BLOCK -u NQ_CARVEOUT "./$PY_BIN" -g "$NQ" "$NQ" > "$LOGDIR/2_G21_console.log" 2>&1
  clk_stop G21
  cp "$gcr" "$LOGDIR/2_G21_crunner.log" 2>/dev/null || { fail "crunner_path_taken[G21]" "no $gcr"; return 1; }
  cp "$CRLOG_DIR/dispatch.log" "$LOGDIR/2_G21_dispatch.log" 2>/dev/null || true
  fields_of "$gcr"; local rr; rr=($(resident_of "${CFGMB:-0}"))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' G21 1 dispatch "${CFGMB:-?}" "${rr[0]}" "${rr[1]}" "${XCTX:-?}" "${FREE:-?}" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start" >> "$TSV"
  [[ "$MATCH" == "1" && "${TOT:-}" == "$ORC" ]] || { fail "oracle[G21]" "total_sum='${TOT:-<none>}' expected $ORC"; return 1; }
  acc G21 "$KMS"; FM[G21]="$FREE"; SMM[G21]="$SMMEAN"; G21MB="${CFGMB:-?}"
  info "G21" "-g $NQ $NQ: MAX_BLOCKS=$G21MB kernel_ms=$KMS extra_ctx=$XCTX free_mb=$FREE sm_mean=$SMMEAN"
}

# ---------------------------------------------------------------------
# 5. Cells: interleaved rounds (odd rounds forward, even rounds reversed), then G21
# ---------------------------------------------------------------------
REV_LIST="$(echo "$MB_LIST" | tr ' ' '\n' | tac | tr '\n' ' ')"
for rd in $(seq 1 "$ROUNDS"); do
  order="$MB_LIST"; (( rd % 2 == 0 )) && order="$REV_LIST"
  banner "Round $rd: $order"
  for mb in $order; do run_direct "A$mb" "$mb" "$rd" || bail; sleep "$COOLDOWN"; done
done
G21MB="?"
if [[ "$SKIP_G21" != "1" ]]; then banner "G21: production path with default MAX_BLOCKS=$DEFAULT_MB"; run_g21 || bail; fi

# ---------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------
banner "Evaluation"
declare -A MEAN; for k in "${!SUM[@]}"; do MEAN[$k]="$(mean_of "$k")"; done
a800="${MEAN[A800]:-}"; Q1=0
if [[ -n "$a800" ]]; then d="$(abspct "$a800" "$ANCHOR_A800")"
  if le "$d" 0.15 && absdiff_le "${FM[A800]}" "$FREE_2CTX" 2; then pass "Q1_HARD_anchor (A800=$a800, ${d}% from $ANCHOR_A800, free_mb=${FM[A800]})"; Q1=1
  else fail "Q1_HARD_anchor" "A800=$a800 is ${d}% from $ANCHOR_A800, free_mb=${FM[A800]} -- not the 402 state; nothing below is read"; fi
fi
if [[ "$Q1" == "1" ]]; then
  Q2=1
  for mb in $MB_LIST; do k="A$mb"; [[ "${CNT[$k]:-0}" -ge 2 ]] || { Q2=2; continue; }; sp="$(spread_of "$k")"; info "spread[$k]" "${sp}% over ${CNT[$k]} runs (${VALS[$k]})"; le "$sp" 0.05 || Q2=0; done
  case "$Q2" in 1) pass "Q2_replicates_within_0.05pct";; 0) fail "Q2_replicates_drift" "a config's two runs differ by more than 0.05%; the A/B is still readable but the adoption bar (Q4) is not met";; *) info "Q2" "single round -- not evaluable";; esac
  best=""; bestmb=""; for mb in $MB_LIST; do v="${MEAN[A$mb]:-}"; [[ -z "$v" ]] && continue; [[ -z "$best" ]] || le "$v" "$best" && { best="$v"; bestmb="$mb"; }; done
  for mb in $MB_LIST; do v="${MEAN[A$mb]:-}"; [[ -z "$v" ]] && continue; info "A$mb" "$v ms  $(pct "$v" "$a800")% vs A800"; done
  q3=1
  [[ "$bestmb" == "960" ]] || q3=0
  if [[ -n "${MEAN[A880]:-}" && -n "${MEAN[A960]:-}" ]]; then { le "${MEAN[A880]}" "$a800" && ge "${MEAN[A880]}" "${MEAN[A960]}"; } || q3=0; fi
  if [[ -n "${MEAN[A1040]:-}" ]]; then le "$(abspct "${MEAN[A1040]}" "$ANCHOR_A1040")" 0.3 || q3=0; fi
  [[ "$q3" == "1" ]] && pass "Q3_stated_shape_holds (winner A960=${MEAN[A960]:-?}; A880 between A800 and A960; A1040 reproduces 402)" \
    || { [[ "$bestmb" == "880" ]] && fail "Q3_ALT_optimum_is_11_warps" "A880=${MEAN[A880]} beats A960=${MEAN[A960]:-?}" || fail "Q3_shape_differs" "winner A$bestmb=$best; A880=${MEAN[A880]:-?} A960=${MEAN[A960]:-?} A1040=${MEAN[A1040]:-?}"; }
  gain="$(pct "$best" "$a800")"
  if le "$gain" -10 && [[ "$Q2" != "0" ]]; then
    if [[ "$bestmb" == "$DEFAULT_MB" ]]; then pass "Q4_ADOPT (A$bestmb=$best, ${gain}% vs A800) -- A10G_FINAL_DEFAULT_MAX_BLOCKS=$DEFAULT_MB in this .py stands"
    else pass "Q4_ADOPT_but_default_must_move (A$bestmb=$best, ${gain}% vs A800) -- 402-r3: set A10G_FINAL_DEFAULT_MAX_BLOCKS=$bestmb (3 executable lines + one G21)"; fi
  else fail "Q4_not_adopted" "best A$bestmb ${gain}% vs A800 (bar: <= -10% and Q2) -- keep 800 until r3 settles it"; fi
  if [[ "$SKIP_G21" != "1" && -n "${MEAN[G21]:-}" ]]; then
    ref="${MEAN[A$DEFAULT_MB]:-$best}"; d="$(abspct "${MEAN[G21]}" "$ref")"
    prefix_ok=0; grep -q "NQ_MAX_BLOCKS=$DEFAULT_MB " "$LOGDIR/2_G21_dispatch.log" 2>/dev/null && prefix_ok=1
    if [[ "$G21MB" == "$DEFAULT_MB" && "$prefix_ok" == "1" ]] && le "$d" 0.30; then pass "Q5_production_path (G21=${MEAN[G21]} ms, ${d}% from direct A$DEFAULT_MB; MAX_BLOCKS=$G21MB carried by the dispatcher)"
    else fail "Q5_production_path" "G21=${MEAN[G21]} (${d}% from A$DEFAULT_MB), CRunner MAX_BLOCKS=$G21MB, prefix_in_dispatch_log=$prefix_ok -- the default is not (or not cleanly) on the -g path"; fi
  fi
fi
CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "Q6_sm_clock_1710_all_cells" || fail "Q6_sm_clock_1710_all_cells" "see clk_*.tsv"

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys, statistics as st
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
by={}
for r in rows: by.setdefault(r['cell'],[]).append(r)
base=st.fmean(float(r['kernel_ms']) for r in by.get('A800',[])) if 'A800' in by else None
print(f"{'cell':<6}{'path':>9}{'MB':>6}{'w/SM':>5}{'KB':>7}{'free_mb':>8}{'n':>3}{'mean ms':>13}{'spread':>8}{'vs A800':>9}{'mm:ss.s':>9}")
for c,rs in by.items():
    v=[float(r['kernel_ms']) for r in rs]; m=st.fmean(v); sp=(max(v)-min(v))/m*100 if len(v)>1 else 0.0
    rel=f"{(m-base)/base*100:+.2f}%" if base else "-"
    print(f"{c:<6}{rs[0]['path']:>9}{rs[0]['max_blocks']:>6}{rs[0]['warps_per_sm']:>5}{rs[0]['footprint_kb']:>7}{rs[0]['free_mb']:>8}{len(v):>3}{m:>13.3f}{sp:>7.3f}%{rel:>9}{int(m//60000):>5}:{(m%60000)/1000:04.1f}")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs" 2>/dev/null || true
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
