#!/usr/bin/env bash
# 402_r4_validate.sh
#
# rev402-r4 -- NQ_HELPER_CTX / NQ_HELPER_MB: the external context holder moves
#              into the binary as forked child processes. Host-side only; the
#              kernel region is 402's. A/B on the production -g path.
#
# CELLS (N=21, MB=960 default, 402-r4 binary; -g cells inherit NQ_HELPER_*
# from this harness's environment through os.system; table prefix unchanged)
#   round 1: G0 G1 G1m G2 D1 D2       round 2: G2 G1m G1 G0
#   G0  -g, no helpers      G1  -g, HELPER_CTX=1     G1m -g, CTX=1 MB=128
#   G2  -g, HELPER_CTX=2    D1 direct HELPER_CTX=1   D2 direct HELPER_CTX=2
#   (D1/D2 have NO external holder: the child must play the holder's part)
# PRE-REGISTERED (402_r4_README_append.md)
#   T1 HARD: G0 within +-0.15% of 111,965; [gpu-helper] helpers=0
#   T2 HARD: D1 within +-0.15% of 111,961 (r3 H0); D2 within +-0.15% of 109,969 (r3 H2)
#   T3 stated: G1 <= G0 - 1.5%      T4 stated: G1m, G2 <= G0 - 2.0%, within 0.15% of each other
#   T5 each -g config's two runs within 0.05%    T6 no straggler processes   T7 SM clock 1710
#   ADOPTION: winner <= G0 - 1.5% and T5; tie-break |G1m - G2| <= 0.15% -> G1m. Adopted in 402-r5.
#
# USAGE
#   STATIC_ONLY=1 bash 402_r4_validate.sh     # OK=18
#                 bash 402_r4_validate.sh     # ~22 min
#   ROUND2=0      bash 402_r4_validate.sh

set -u

REV="402_r4"
PY_SRC="${PY_SRC:-402_r4Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-402_r4Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-402_r3Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-402_r4_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-402_r4_kernel_maxd14}"
PREV_CU="${PREV_CU:-402_r3_kernel_maxd14.cu}"
CRLOG_DIR="${CRLOG_DIR:-402_r4_crunner_logs}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
BLOCK="${BLOCK:-32}"
MB="${MB:-960}"
ROUND2="${ROUND2:-1}"
KERNEL_SHA_402="${KERNEL_SHA_402:-122b331980495f8b}"   # prefix is enough; full value printed
EXPECT_REMOVED="${EXPECT_REMOVED:-0}"
EXPECT_ADDED="${EXPECT_ADDED:-87}"
ANCHOR_G0="${ANCHOR_G0:-111964.648}"    # 402-r3 G0
ANCHOR_H0="${ANCHOR_H0:-111961.433}"    # 402-r3 H0 (1 external holder)
ANCHOR_H2="${ANCHOR_H2:-109969.332}"    # 402-r3 H2 (2 external holders)
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
for f in "$PY_SRC" "$HELPER_SRC" "$CU_SRC" "$PREV_CU"; do
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
grep -qE '^REV_TAG:str="402_r4"' "$CODE" && pass "source_rev_tag_is_402_r4" || fail "source_rev_tag_is_402_r4" "wrong REV_TAG"
grep -q 'CRunnerEntry(14,"./402_r4_kernel_maxd14","NQ_EXTRA_CTX=1 "' "$CODE" && pass "source_table_points_at_402_r4_prefix_unchanged" || fail "source_table_points_at_402_r4_prefix_unchanged" "table entry wrong (prefix must still be exactly NQ_EXTRA_CTX=1 in r4)"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=${MB}\b" "$CODE" && pass "source_default_max_blocks_is_${MB}" || fail "source_default_max_blocks_is_${MB}" "default not $MB"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_402_r3Py (removed=3 added=3 EXECUTABLE lines)" \
    || { fail "py_diff_fingerprint_vs_402_r3Py" "removed=$PR_ added=$PA_, expected 3/3"; }
else info "py_diff_fingerprint_vs_402_r3Py" "skipped"; fi
cu_code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"; cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"; KP="$(extract_kernel "/tmp/${REV}_prev_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KP" && "$KB" == "$KERNEL_SHA_402"* ]] && pass "cu_kernel_region_unchanged_from_402 (${KB:0:16}...)" || fail "cu_kernel_region_unchanged_from_402" "cur ${KB:0:16} prev ${KP:0:16} -- r4 must not touch the kernel"
CR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true); CA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
[[ "$CR_" == "$EXPECT_REMOVED" && "$CA_" == "$EXPECT_ADDED" ]] && pass "cu_diff_fingerprint_vs_402_r3 (removed=$CR_ added=$CA_)" || fail "cu_diff_fingerprint_vs_402_r3" "removed=$CR_ added=$CA_, expected $EXPECT_REMOVED/$EXPECT_ADDED"
read -r LH LF LC < <(awk '/getenv\("NQ_HELPER_CTX"\)/{h=NR} /fopen\(in_path/{if(!f)f=NR} /cudaFuncGetAttributes\(&fa/{c=NR} END{print h+0, f+0, c+0}' "/tmp/${REV}_cur_code.cu")
[[ "$LH" -gt 0 && "$LH" -lt "$LF" && "$LF" -lt "$LC" ]] && pass "cu_fork_precedes_first_file_and_cuda_call (helper@$LH fopen@$LF cuda@$LC)" || fail "cu_fork_precedes_first_file_and_cuda_call" "helper@$LH fopen@$LF cuda@$LC -- the fork must come before any CUDA call in the parent"
grep -q 'if (want > 0) {' "/tmp/${REV}_cur_code.cu" && grep -q 'pid_t pid = fork();' "/tmp/${REV}_cur_code.cu" && grep -q '_exit(st ? 0 : 1);' "/tmp/${REV}_cur_code.cu" && grep -q 'prctl(PR_SET_PDEATHSIG, SIGKILL);' "/tmp/${REV}_cur_code.cu" && pass "cu_helper_guarded_child_exits_and_pdeathsig" || fail "cu_helper_guarded_child_exits_and_pdeathsig" "helper block incomplete"
grep -q 'waitpid(helper_pid\[_i\], &_st, 0);' "/tmp/${REV}_cur_code.cu" && pass "cu_helpers_reaped_at_exit" || fail "cu_helpers_reaped_at_exit" "no waitpid of helpers"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"/tmp/${REV}_gcc.log" && pass "cu_cpu_harness_still_builds (host section is behind __CUDACC__)" || { fail "cu_cpu_harness_still_builds" "$(head -3 /tmp/${REV}_gcc.log)"; }
grep -q "rev402-r4" "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev402-r4 note in header"

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
[[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "$(grep -i error "$LOGDIR/05_nvcc.log" | head -3)"; summary_exit; }
FR="$(grep -A3 'kernel_dfs_iter_gpu_maxd14' "$LOGDIR/05_nvcc.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"
[[ "$FR" == "160" ]] && pass "ptxas_frame_160B" || fail "ptxas_frame_160B" "frame=${FR:-?}"
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
printf 'cell\tround\tpath\thelper_ctx\thelper_mb\thelpers_logged\tmax_blocks\tfree_mb\tkernel_ms\ttotal_sum\tmatch\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
CLKPID=""
gpu_gate() {  # nothing at all may be on the GPU (no external holders in r4)
  local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"
  echo "$apps" > "$LOGDIR/apps_before_$1.txt"
  [[ -z "$apps" ]] && return 0
  fail "gpu_empty_before[$1]" "compute process(es) present: $(echo "$apps" | tr '\n' ' ') -- a previous cell's helper did not exit (T6) or something else is running"; return 1
}
clk_start() { local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"
  ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPID=$!; }
clk_stop() { [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.0f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_$1.tsv"); }
fields_of() { KMS="$(grep -o 'kernel_ms=[0-9.]*' "$1" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$1" | head -1 | cut -d= -f2)"
  MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$1" && MATCH=1; FREE="$(grep -o 'free_mb=[0-9]*' "$1" | head -1 | cut -d= -f2)"
  CFGMB="$(grep -o 'MAX_BLOCKS=[0-9]*' "$1" | head -1 | cut -d= -f2)"; HLOG="$(grep -o 'helpers=[0-9]*' "$1" | head -1 | cut -d= -f2)"; }
bail() { [[ -n "$CLKPID" ]] && kill "$CLKPID" 2>/dev/null; summary_exit; }
declare -A SUM CNT VALS FM SMM MINV MAXV HL
acc() { local k="$1" v="$2"; SUM[$k]="$(awk -v s="${SUM[$k]:-0}" -v x="$v" 'BEGIN{printf "%.3f",s+x}')"; CNT[$k]=$(( ${CNT[$k]:-0}+1 )); VALS[$k]="${VALS[$k]:-}$v "
  [[ -z "${MINV[$k]:-}" ]] || le "$v" "${MINV[$k]}" && MINV[$k]="$v"; [[ -z "${MAXV[$k]:-}" ]] || ge "$v" "${MAXV[$k]}" && MAXV[$k]="$v"; }
mean_of() { awk -v s="${SUM[$1]}" -v n="${CNT[$1]}" 'BEGIN{printf "%.3f",s/n}'; }
spread_of() { awk -v a="${MAXV[$1]}" -v b="${MINV[$1]}" -v s="${SUM[$1]}" -v n="${CNT[$1]}" 'BEGIN{m=s/n; printf "%.3f",(a-b)/m*100}'; }
after_cell() {  # T6: nothing may remain
  sleep 2; local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"
  echo "$apps" > "$LOGDIR/apps_after_$1.txt"
  [[ -z "$apps" ]] || { STRAGGLER=1; info "T6_straggler_after[$1]" "$(echo "$apps" | tr '\n' ' ')"; }
}
STRAGGLER=0
run_g() {  # cell round hctx hmb
  local cell="$1" rd="$2" hc="$3" hm="$4" tag="${1}_r${2}"
  local gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${NQ}.log"; rm -f "$gcr"
  gpu_gate "$tag" || return 1
  local start; start="$(date -Is)"; clk_start "$tag"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX -u NQ_BLOCK -u NQ_CARVEOUT NQ_HELPER_CTX="$hc" NQ_HELPER_MB="$hm" "./$PY_BIN" -g "$NQ" "$NQ" > "$LOGDIR/2_${tag}_console.log" 2>&1
  clk_stop "$tag"; after_cell "$tag"
  cp "$gcr" "$LOGDIR/2_${tag}_crunner.log" 2>/dev/null || { fail "crunner_path_taken[$tag]" "no $gcr"; return 1; }
  fields_of "$gcr"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$rd" dispatch "$hc" "$hm" "${HLOG:-?}" "${CFGMB:-?}" "${FREE:-?}" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start" >> "$TSV"
  [[ "$MATCH" == "1" && "${TOT:-}" == "$ORC" ]] || { fail "oracle[$tag]" "total_sum='${TOT:-<none>}' expected $ORC"; return 1; }
  [[ "${CFGMB:-}" == "$MB" ]] || { fail "g_max_blocks[$tag]" "MAX_BLOCKS=${CFGMB:-?}, expected $MB"; return 1; }
  [[ "${HLOG:-}" == "$hc" ]] || { fail "helper_env_reached_binary[$tag]" "[gpu-helper] helpers=${HLOG:-?}, expected $hc -- the environment did not pass through os.system"; return 1; }
  acc "$cell" "$KMS"; FM[$cell]="$FREE"; SMM[$tag]="$SMMEAN"; HL[$cell]="$HLOG"
  info "$tag" "-g HELPER_CTX=$hc MB=$hm -> helpers=$HLOG kernel_ms=$KMS free_mb=$FREE sm_mean=$SMMEAN temp_max=$TMAX"
}
run_direct() {  # cell round hctx hmb
  local cell="$1" rd="$2" hc="$3" hm="$4" tag="${1}_r${2}"
  gpu_gate "$tag" || return 1
  local lg="$LOGDIR/3_${tag}.log" start; start="$(date -Is)"; clk_start "$tag"
  env -u NQ_PAD_MB -u NQ_CARVEOUT NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$MB" NQ_EXTRA_CTX=0 NQ_HELPER_CTX="$hc" NQ_HELPER_MB="$hm" "./$CU_BIN" "$NQ" "$IN" "/tmp/${REV}_out.bin" "$ORC" > "$lg" 2>&1 || true
  clk_stop "$tag"; after_cell "$tag"; fields_of "$lg"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$rd" direct "$hc" "$hm" "${HLOG:-?}" "$MB" "${FREE:-?}" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start" >> "$TSV"
  [[ "$MATCH" == "1" && "${TOT:-}" == "$ORC" ]] || { fail "oracle[$tag]" "total_sum='${TOT:-<none>}' expected $ORC"; return 1; }
  [[ "${HLOG:-}" == "$hc" ]] || { fail "helper_count[$tag]" "helpers=${HLOG:-?}, expected $hc"; return 1; }
  acc "$cell" "$KMS"; FM[$cell]="$FREE"; SMM[$tag]="$SMMEAN"; HL[$cell]="$HLOG"
  info "$tag" "direct HELPER_CTX=$hc MB=$hm -> helpers=$HLOG kernel_ms=$KMS free_mb=$FREE sm_mean=$SMMEAN temp_max=$TMAX"
}

# ---------------------------------------------------------------------
# 4. Cells
# ---------------------------------------------------------------------
banner "Round 1"
run_g      G0  1 0 0   || bail; sleep "$COOLDOWN"
run_g      G1  1 1 0   || bail; sleep "$COOLDOWN"
run_g      G1m 1 1 128 || bail; sleep "$COOLDOWN"
run_g      G2  1 2 0   || bail; sleep "$COOLDOWN"
run_direct D1  1 1 0   || bail; sleep "$COOLDOWN"
run_direct D2  1 2 0   || bail; sleep "$COOLDOWN"
if [[ "$ROUND2" == "1" ]]; then
  banner "Round 2 (reversed)"
  run_g G2  2 2 0   || bail; sleep "$COOLDOWN"
  run_g G1m 2 1 128 || bail; sleep "$COOLDOWN"
  run_g G1  2 1 0   || bail; sleep "$COOLDOWN"
  run_g G0  2 0 0   || bail; sleep "$COOLDOWN"
fi

# ---------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------
banner "Evaluation"
declare -A MEAN; for k in "${!SUM[@]}"; do MEAN[$k]="$(mean_of "$k")"; done
for k in G0 G1 G1m G2 D1 D2; do [[ -n "${MEAN[$k]:-}" ]] && info "$k" "mean=${MEAN[$k]} n=${CNT[$k]} spread=$( [[ ${CNT[$k]} -ge 2 ]] && spread_of "$k" || echo -)% free_mb=${FM[$k]} helpers=${HL[$k]}"; done
g0="${MEAN[G0]}"; T1=0
d="$(abspct "$g0" "$ANCHOR_G0")"
if le "$d" 0.15 && [[ "${HL[G0]}" == "0" ]]; then pass "T1_HARD_inert_default (G0=$g0, ${d}% from $ANCHOR_G0, helpers=0)"; T1=1
else fail "T1_HARD_inert_default" "G0=$g0 is ${d}% from $ANCHOR_G0, helpers=${HL[G0]} -- nothing below is read"; fi
if [[ "$T1" == "1" ]]; then
  d1="$(abspct "${MEAN[D1]}" "$ANCHOR_H0")"; d2="$(abspct "${MEAN[D2]}" "$ANCHOR_H2")"
  if le "$d1" 0.15 && le "$d2" 0.15; then pass "T2_HARD_child_equals_external_holder (D1=${MEAN[D1]} ${d1}% from r3 H0; D2=${MEAN[D2]} ${d2}% from r3 H2)"; T2=1
  else fail "T2_HARD_child_is_NOT_a_holder" "D1=${MEAN[D1]} (${d1}% from $ANCHOR_H0), D2=${MEAN[D2]} (${d2}% from $ANCHOR_H2) -- a forked child does not reproduce the external-holder state; production must keep using an external process"; T2=0; fi
  p1="$(pct "${MEAN[G1]}" "$g0")"
  le "$p1" -1.5 && pass "T3_one_helper_on_g (G1 ${p1}% vs G0)" || fail "T3_one_helper_on_g" "G1 ${p1}% vs G0 (registered <= -1.5%)"
  pm="$(pct "${MEAN[G1m]}" "$g0")"; p2="$(pct "${MEAN[G2]}" "$g0")"; a12="$(abspct "${MEAN[G1m]}" "${MEAN[G2]}")"
  if le "$pm" -2.0 && le "$p2" -2.0 && le "$a12" 0.15; then pass "T4_plateau_on_g (G1m ${pm}%, G2 ${p2}% vs G0; |G1m-G2|=${a12}%)"
  else fail "T4_plateau_on_g" "G1m ${pm}%, G2 ${p2}% vs G0, |G1m-G2|=${a12}% -- the r3 plateau does not reproduce as registered on the -g path"; fi
  T5=1
  if [[ "$ROUND2" == "1" ]]; then for k in G0 G1 G1m G2; do sp="$(spread_of "$k")"; le "$sp" 0.05 || T5=0; done
    [[ "$T5" == "1" ]] && pass "T5_replicates_within_0.05pct" || fail "T5_replicates_drift" "a -g config's two runs differ by more than 0.05%"; else T5=2; info "T5" "single round"; fi
  # winner and adoption
  best=""; bestk=""; for k in G1 G1m G2; do v="${MEAN[$k]}"; [[ -z "$best" ]] || le "$v" "$best" && { best="$v"; bestk="$k"; }; done
  if le "$a12" 0.15 && le "$pm" -1.5; then bestk="G1m"; best="${MEAN[G1m]}"; fi   # tie-break: fewer contexts, one child
  gain="$(pct "$best" "$g0")"
  case "$bestk" in G1) setting="NQ_HELPER_CTX=1";; G1m) setting="NQ_HELPER_CTX=1 NQ_HELPER_MB=128";; G2) setting="NQ_HELPER_CTX=2";; esac
  if [[ "${T2:-0}" == "1" && "$T5" == "1" ]] && le "$gain" -1.5; then
    pass "ADOPTION_RULE_MET ($bestk=$best, ${gain}% vs G0) -- 402-r5: put '$setting' into the table env_prefix and re-measure -g 21 21"
  elif [[ "$T5" == "2" ]]; then info "ADOPTION" "single round -- rerun with ROUND2=1 before adopting"
  else fail "ADOPTION_RULE_not_met" "$bestk ${gain}% vs G0, T2=${T2:-0}, T5=$T5"; fi
fi
[[ "$STRAGGLER" == "0" ]] && pass "T6_no_straggler_processes" || fail "T6_stragglers_seen" "see apps_after_*.txt -- helpers outlived the parent"
CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "T7_sm_clock_1710_all_cells" || fail "T7_sm_clock_1710_all_cells" "see clk_*.tsv"

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys, statistics as st
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
by={}
for r in rows: by.setdefault(r['cell'],[]).append(r)
g0=st.fmean(float(r['kernel_ms']) for r in by['G0']) if 'G0' in by else None
print(f"{'cell':<5}{'path':>9}{'hctx':>5}{'hmb':>5}{'free_mb':>8}{'n':>3}{'mean ms':>13}{'spread':>8}{'vs G0':>9}{'mm:ss.s':>9}")
for c,rs in by.items():
    v=[float(r['kernel_ms']) for r in rs]; m=st.fmean(v); sp=(max(v)-min(v))/m*100 if len(v)>1 else 0.0
    rel=f"{(m-g0)/g0*100:+.3f}%" if g0 else "-"
    print(f"{c:<5}{rs[0]['path']:>9}{rs[0]['helper_ctx']:>5}{rs[0]['helper_mb']:>5}{rs[0]['free_mb']:>8}{len(v):>3}{m:>13.3f}{sp:>7.3f}%{rel:>9}{int(m//60000):>5}:{(m%60000)/1000:04.1f}")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs" 2>/dev/null || true
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
