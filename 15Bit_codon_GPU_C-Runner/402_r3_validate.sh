#!/usr/bin/env bash
# 402_r3_validate.sh
#
# rev402-r3 -- does the third-context notch exist at MB=960? Zero code change;
#              every GPU state is made by external holders (395c_ctx_holder).
#
# CELLS (N=21, default MB=960, 402-r3 binary)
#   round 1: G0 H0 H2 Gh H3 H2b H2c      round 2: H0 H2 Gh
#   G0  -g 21 21                    H0  1 holder (0 MB)      H2  2 holders (0,0)
#   Gh  -g 21 21 + 1 holder alive   H3  3 holders (0,0,0)
#   H2b holders (0,128)             H2c holders (0,384)
# PRE-REGISTERED (402_r3_README_append.md)
#   S1 HARD: mean H0 within +-0.15% of 112,001, free_mb 22018
#   S2 stated (weak): H2 <= H0 - 0.20%.  Alt: |H2 - H0| <= 0.10%.
#   S3 |Gh - H2| <= 0.10% and Gh <= G0 - 0.20%
#   S4 H3 not better than H2 by > 0.10%
#   S5 H2b, H2c not better than H2 by > 0.10% (exploratory)
#   S6 mean SM clock within 2% of 1710 in every cell
#   Nothing is adopted from this run.
#
# USAGE
#   STATIC_ONLY=1 bash 402_r3_validate.sh     # OK=15
#                 bash 402_r3_validate.sh     # ~22 min
#   ROUND2=0      bash 402_r3_validate.sh     # single pass (~15 min)

set -u

REV="402_r3"
PY_SRC="${PY_SRC:-402_r3Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-402_r3Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-402_r2Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-402_r3_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-402_r3_kernel_maxd14}"
PREV_CU="${PREV_CU:-402_r2_kernel_maxd14.cu}"
HOLDER_SRC="${HOLDER_SRC:-395c_ctx_holder.cu}"
HOLDER_BIN="${HOLDER_BIN:-395c_ctx_holder}"
CRLOG_DIR="${CRLOG_DIR:-402_r3_crunner_logs}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
BLOCK="${BLOCK:-32}"
MB="${MB:-960}"
ROUND2="${ROUND2:-1}"
ANCHOR_H0="${ANCHOR_H0:-112001.438}"   # 402-r2 A960 (1 holder, ctx=0)
ANCHOR_G0="${ANCHOR_G0:-112000.352}"   # 402-r2 G21
FREE_2CTX="${FREE_2CTX:-22018}"
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
grep -qE '^REV_TAG:str="402_r3"' "$CODE" && pass "source_rev_tag_is_402_r3" || fail "source_rev_tag_is_402_r3" "wrong REV_TAG"
grep -q 'CRunnerEntry(14,"./402_r3_kernel_maxd14","NQ_EXTRA_CTX=1 "' "$CODE" && pass "source_table_points_at_402_r3_and_keeps_the_treatment" || fail "source_table_points_at_402_r3_and_keeps_the_treatment" "table entry wrong"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=${MB}\b" "$CODE" && pass "source_default_max_blocks_is_${MB}" || fail "source_default_max_blocks_is_${MB}" "default not $MB"
grep -qE '^CRUNNER_MIN_N:int=19' "$CODE" && pass "source_crunner_min_n_still_19" || fail "source_crunner_min_n_still_19" "CRUNNER_MIN_N lost"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_402_r2Py (removed=3 added=3 EXECUTABLE lines)" \
    || { fail "py_diff_fingerprint_vs_402_r2Py" "removed=$PR_ added=$PA_, expected 3/3"; diff "/tmp/${REV}_prev_code.py" "$CODE" | head -20 | sed 's/^/      /'; }
else info "py_diff_fingerprint_vs_402_r2Py" "skipped"; fi
cu_code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
if [[ -f "$PREV_CU" ]]; then
  cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  CA="$(sha256sum < "/tmp/${REV}_prev_code.cu" | cut -d' ' -f1)"; CB="$(sha256sum < "/tmp/${REV}_cur_code.cu" | cut -d' ' -f1)"
  [[ "$CA" == "$CB" ]] && pass "cu_whole_code_region_identical_to_402_r2 (${CB:0:16}...)" || fail "cu_whole_code_region_identical_to_402_r2" "402-r3 must be a rename"
else info "cu_whole_code_region_identical_to_402_r2" "skipped"; fi
grep -q 'uint64_t stack_depth = 0;' "/tmp/${REV}_cur_code.cu" && pass "cu_12B_frame_present" || fail "cu_12B_frame_present" "the 402 packing is missing"
grep -q "rev402-r3" "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev402-r3 note in header"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Build
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
banner "Building"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc.log" 2>&1
[[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "no binary"; summary_exit; }
FR="$(grep -A3 'kernel_dfs_iter_gpu_maxd14' "$LOGDIR/05_nvcc.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"
[[ "$FR" == "160" ]] && pass "ptxas_frame_160B" || fail "ptxas_frame_160B" "frame=${FR:-?}"
if [[ ! -x "$HOLDER_BIN" ]]; then "$NVCC" -O2 -arch="$ARCH" -o "$HOLDER_BIN" "$HOLDER_SRC" > "$LOGDIR/05a_nvcc_holder.log" 2>&1; fi
[[ -x "$HOLDER_BIN" ]] && pass "ctx_holder_available[$HOLDER_BIN]" || { fail "ctx_holder_available" "no $HOLDER_BIN"; summary_exit; }
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/06_codon.log"
[[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN]" || { fail "codon_build" "see log"; summary_exit; }

IN=""; ORC="$(oracle_of "$NQ")"
for lg in $(ls -t 40*_crunner_logs/crunner_*_N${NQ}.log 2>/dev/null); do
  c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { IN="$c"; break; }
done
[[ -f "${IN:-/nonexistent}" && "$(stat -c %s "$IN")" == "56707896" ]] && pass "input_located ($IN)" || { fail "input_located" "no sched input via 40*_crunner_logs"; summary_exit; }

# ---------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------
printf 'cell\tround\tpath\tholders_mb\tmax_blocks\textra_ctx\tfree_mb\tkernel_ms\ttotal_sum\tmatch\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
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
holders_up() {  # tag mb1 [mb2 ...]
  local tag="$1"; shift; local i=0 m
  for m in "$@"; do i=$((i+1)); "./$HOLDER_BIN" "$m" 900 > "$LOGDIR/holder_${tag}_$i.log" 2>&1 & HPIDS+=("$!"); done
  sleep 5; cat "$LOGDIR"/holder_"$tag"_*.log 2>/dev/null
}
holders_down() { local p; for p in "${HPIDS[@]:-}"; do [[ -n "$p" ]] && { kill "$p" 2>/dev/null; wait "$p" 2>/dev/null || true; }; done; HPIDS=(); sleep 2; }
bail() { holders_down; [[ -n "$CLKPID" ]] && kill "$CLKPID" 2>/dev/null; summary_exit; }
declare -A SUM CNT VALS FM SMM MINV MAXV
acc() { local k="$1" v="$2"; SUM[$k]="$(awk -v s="${SUM[$k]:-0}" -v x="$v" 'BEGIN{printf "%.3f",s+x}')"; CNT[$k]=$(( ${CNT[$k]:-0}+1 )); VALS[$k]="${VALS[$k]:-}$v "
  [[ -z "${MINV[$k]:-}" ]] || le "$v" "${MINV[$k]}" && MINV[$k]="$v"; [[ -z "${MAXV[$k]:-}" ]] || ge "$v" "${MAXV[$k]}" && MAXV[$k]="$v"; }
mean_of() { awk -v s="${SUM[$1]}" -v n="${CNT[$1]}" 'BEGIN{printf "%.3f",s/n}'; }
spread_of() { awk -v a="${MAXV[$1]}" -v b="${MINV[$1]}" -v s="${SUM[$1]}" -v n="${CNT[$1]}" 'BEGIN{m=s/n; printf "%.3f",(a-b)/m*100}'; }
run_direct() {  # cell round holders_mb...
  local cell="$1" rd="$2"; shift 2; local tag="${cell}_r${rd}"
  holders_up "$tag" "$@"; gpu_gate "$tag" || { holders_down; return 1; }
  local lg="$LOGDIR/3_${tag}.log" start; start="$(date -Is)"; clk_start "$tag"
  env -u NQ_PAD_MB -u NQ_CARVEOUT NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$MB" NQ_EXTRA_CTX=0 "./$CU_BIN" "$NQ" "$IN" "/tmp/${REV}_out.bin" "$ORC" > "$lg" 2>&1 || true
  clk_stop "$tag"; holders_down; fields_of "$lg"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$rd" direct "$*" "$MB" "${XCTX:-?}" "${FREE:-?}" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start" >> "$TSV"
  [[ "$MATCH" == "1" && "${TOT:-}" == "$ORC" ]] || { fail "oracle[$tag]" "total_sum='${TOT:-<none>}' expected $ORC"; return 1; }
  acc "$cell" "$KMS"; FM[$cell]="$FREE"; SMM[$tag]="$SMMEAN"
  info "$tag" "holders=[$*] kernel_ms=$KMS free_mb=$FREE sm_mean=$SMMEAN temp_max=$TMAX"
}
run_g() {  # cell round holders_mb...
  local cell="$1" rd="$2"; shift 2; local tag="${cell}_r${rd}"
  local gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${NQ}.log"; rm -f "$gcr"
  [[ $# -gt 0 ]] && holders_up "$tag" "$@"; gpu_gate "$tag" || { holders_down; return 1; }
  local start; start="$(date -Is)"; clk_start "$tag"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX -u NQ_BLOCK -u NQ_CARVEOUT "./$PY_BIN" -g "$NQ" "$NQ" > "$LOGDIR/2_${tag}_console.log" 2>&1
  clk_stop "$tag"; holders_down
  cp "$gcr" "$LOGDIR/2_${tag}_crunner.log" 2>/dev/null || { fail "crunner_path_taken[$tag]" "no $gcr"; return 1; }
  fields_of "$gcr"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$rd" dispatch "${*:-none}" "${CFGMB:-?}" "${XCTX:-?}" "${FREE:-?}" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start" >> "$TSV"
  [[ "$MATCH" == "1" && "${TOT:-}" == "$ORC" ]] || { fail "oracle[$tag]" "total_sum='${TOT:-<none>}' expected $ORC"; return 1; }
  [[ "${CFGMB:-}" == "$MB" ]] || { fail "g_max_blocks[$tag]" "CRunner ran MAX_BLOCKS=${CFGMB:-?}, expected $MB"; return 1; }
  acc "$cell" "$KMS"; FM[$cell]="$FREE"; SMM[$tag]="$SMMEAN"
  info "$tag" "-g $NQ $NQ holders=[${*:-none}] MAX_BLOCKS=$CFGMB kernel_ms=$KMS free_mb=$FREE sm_mean=$SMMEAN"
}

# ---------------------------------------------------------------------
# 4. Cells
# ---------------------------------------------------------------------
banner "Round 1"
run_g      G0  1        || bail; sleep "$COOLDOWN"
run_direct H0  1 0      || bail; sleep "$COOLDOWN"
run_direct H2  1 0 0    || bail; sleep "$COOLDOWN"
run_g      Gh  1 0      || bail; sleep "$COOLDOWN"
run_direct H3  1 0 0 0  || bail; sleep "$COOLDOWN"
run_direct H2b 1 0 128  || bail; sleep "$COOLDOWN"
run_direct H2c 1 0 384  || bail; sleep "$COOLDOWN"
if [[ "$ROUND2" == "1" ]]; then
  banner "Round 2"
  run_direct H0 2 0     || bail; sleep "$COOLDOWN"
  run_direct H2 2 0 0   || bail; sleep "$COOLDOWN"
  run_g      Gh 2 0     || bail; sleep "$COOLDOWN"
fi

# ---------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------
banner "Evaluation"
declare -A MEAN; for k in "${!SUM[@]}"; do MEAN[$k]="$(mean_of "$k")"; done
for k in G0 H0 H2 Gh H3 H2b H2c; do [[ -n "${MEAN[$k]:-}" ]] && info "$k" "mean=${MEAN[$k]} n=${CNT[$k]} spread=$( [[ ${CNT[$k]} -ge 2 ]] && spread_of "$k" || echo -)% free_mb=${FM[$k]}"; done
h0="${MEAN[H0]}"; S1=0
d="$(abspct "$h0" "$ANCHOR_H0")"
if le "$d" 0.15 && absdiff_le "${FM[H0]}" "$FREE_2CTX" 2; then pass "S1_HARD_anchor (H0=$h0, ${d}% from $ANCHOR_H0, free_mb=${FM[H0]})"; S1=1
else fail "S1_HARD_anchor" "H0=$h0 is ${d}% from $ANCHOR_H0, free_mb=${FM[H0]} -- not the 402-r2 state; nothing below is read"; fi
if [[ "$S1" == "1" ]]; then
  h2="${MEAN[H2]}"; p="$(pct "$h2" "$h0")"; a="$(abspct "$h2" "$h0")"
  if le "$p" -0.20; then pass "S2_notch_present_at_960 (H2 ${p}% vs H0, free_mb=${FM[H2]})"; S2=1
  elif le "$a" 0.10; then fail "S2_ALT_notch_gone_at_960" "H2 ${p}% vs H0 -- the third context does nothing at MB=960; the notch was specific to 800's layout. Axis closed, nothing to build"; S2=0
  else fail "S2_indeterminate" "H2 ${p}% vs H0 -- between the registered outcomes; report only"; S2=0; fi
  gh="${MEAN[Gh]}"; g0="${MEAN[G0]}"
  pgh="$(pct "$gh" "$h2")"; agh="$(abspct "$gh" "$h2")"; pg="$(pct "$gh" "$g0")"
  if le "$agh" 0.10 && le "$pg" -0.20; then pass "S3_helper_beside_g_reproduces_3ctx (Gh ${pgh}% vs H2; Gh ${pg}% vs G0=$g0)"; S3=1
  else fail "S3_helper_beside_g" "Gh=$gh: ${pgh}% vs H2 (want <=0.10% abs), ${pg}% vs G0 (want <= -0.20%)"; S3=0; fi
  if [[ -n "${MEAN[H3]:-}" ]]; then p="$(pct "${MEAN[H3]}" "$h2")"; le "$p" -0.10 && fail "S4_REFUTED_4ctx_better" "H3 ${p}% vs H2 -- a fourth context helps further; sweep the count in a follow-up" || pass "S4_4ctx_not_better (H3 ${p}% vs H2)"; fi
  for c in H2b H2c; do [[ -n "${MEAN[$c]:-}" ]] || continue; p="$(pct "${MEAN[$c]}" "$h2")"; le "$p" -0.10 && info "S5_exploratory[$c]" "${p}% vs H2 -- BETTER than +255; the valley is elsewhere at 960 (report only)" || info "S5_exploratory[$c]" "${p}% vs H2 (not better than +255)"; done
  adopt=0
  if [[ "$ROUND2" == "1" && "${S2:-0}" == "1" && "${S3:-0}" == "1" ]]; then
    sg="$(spread_of Gh)"; sh="$(spread_of H2)"
    if le "$sg" 0.05 && le "$sh" 0.05; then adopt=1; fi
    [[ "$adopt" == "1" ]] && pass "ADOPTION_RULE_MET (Gh spread ${sg}%, H2 spread ${sh}%, gain ${pg}% on the -g path) -- 402-r4: implement NQ_HELPER_CTX (fork a context-holding child) in the binary, gated by an interleaved -g A/B" \
      || fail "ADOPTION_RULE_not_met" "Gh spread ${sg}%, H2 spread ${sh}% (bar 0.05%) -- the gain is real but not yet reproducible enough to build on"
  elif [[ "${S2:-0}" == "1" && "${S3:-0}" == "1" ]]; then info "ADOPTION" "single pass -- rerun with ROUND2=1 before building anything"; fi
fi
CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "S6_sm_clock_1710_all_cells" || fail "S6_sm_clock_1710_all_cells" "see clk_*.tsv"

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys, statistics as st
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
by={}
for r in rows: by.setdefault(r['cell'],[]).append(r)
h0=st.fmean(float(r['kernel_ms']) for r in by['H0']) if 'H0' in by else None
print(f"{'cell':<5}{'path':>9}{'holders(MB)':>13}{'MB':>5}{'ctx':>4}{'free_mb':>8}{'n':>3}{'mean ms':>13}{'spread':>8}{'vs H0':>9}")
for c,rs in by.items():
    v=[float(r['kernel_ms']) for r in rs]; m=st.fmean(v); sp=(max(v)-min(v))/m*100 if len(v)>1 else 0.0
    rel=f"{(m-h0)/h0*100:+.3f}%" if h0 else "-"
    print(f"{c:<5}{rs[0]['path']:>9}{rs[0]['holders_mb']:>13}{rs[0]['max_blocks']:>5}{rs[0]['extra_ctx']:>4}{rs[0]['free_mb']:>8}{len(v):>3}{m:>13.3f}{sp:>7.3f}%{rel:>9}")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs" 2>/dev/null || true
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
