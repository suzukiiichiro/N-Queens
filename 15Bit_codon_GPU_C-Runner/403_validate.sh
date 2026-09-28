#!/usr/bin/env bash
# 403_validate.sh
#
# rev403 -- the 12-byte DFS frame re-laid out for N <= 22 (col 22 | avail 22 |
#           ld 21 | rd 31, using ld's dead top bit and rd's dead bottom bit).
#
# GATES (in order; each a hard stop)
#   W1  ptxas frame 160 B, spill 0/0, regs <= 44
#   W2  CPU harness per-record equality: 402_r5 vs 403 on N=21 real records,
#       401_r4 vs 403 on N=22 real records
#   W3  N=23 refused (rc=3)
#   W4  N=21 full input @960 with helper: 402_r5 vs 403 per-thread byte-identical
# PRE-REGISTERED
#   W5 stated |E1 - E0| <= 0.3%     W6 HARD: -g 22 22 MATCH 2,691,008,701,644
#   W7 stated G22 <= 969,840 ms      W8 info: free_mb, k_per_thread at N=22
#   W9 SM clock 1710 in every cell
#   FULL22=1: 401_r4 direct N=22 @960 with helper; per-thread equality vs G22 (+20 min)
#
# USAGE
#   STATIC_ONLY=1 bash 403_validate.sh        # OK=20
#                 bash 403_validate.sh        # ~30 min (+ N=22 sched generation if absent)
#   FULL22=1      bash 403_validate.sh        # ~50 min
#   GATES_ONLY=1  bash 403_validate.sh        # W1..W4 + W5, no N=22

set -u

REV="403"
PY_SRC="${PY_SRC:-403Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-403Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-402_r5Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-403_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-403_kernel_maxd14}"
PREV_CU="${PREV_CU:-402_r5_kernel_maxd14.cu}"
PREV_BIN="${PREV_BIN:-402_r5_kernel_maxd14}"
OLD_CU="${OLD_CU:-401_r4_kernel_maxd14.cu}"       # unpacked frame: the N=22 reference
OLD_BIN="${OLD_BIN:-401_r4_kernel_maxd14}"
CRLOG_DIR="${CRLOG_DIR:-403_crunner_logs}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
BLOCK="${BLOCK:-32}"
MB="${MB:-960}"
HCTX="${HCTX:-1}"; HMB="${HMB:-128}"              # production helper state
CPU_CHECK_RECORDS="${CPU_CHECK_RECORDS:-8192}"
CPU_CHECK_RECORDS_22="${CPU_CHECK_RECORDS_22:-16384}"
ANCHOR_E0="${ANCHOR_E0:-109437.2}"                # 402-r5 G21 mean
BASE22="${BASE22:-1140988.375}"                   # 9 Sep, 401-r2 binary, MB=800, no helper
KERNEL_SHA_402="${KERNEL_SHA_402:-122b331980495f8b}"
EXPECT_REMOVED="${EXPECT_REMOVED:-21}"
EXPECT_ADDED="${EXPECT_ADDED:-33}"
GATES_ONLY="${GATES_ONLY:-0}"
FULL22="${FULL22:-0}"
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
absdiff_le() { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=t)}'; }
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS (registered refutations are listed here too -- read the label):"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi; exit 0; }

# ---------------------------------------------------------------------
# 1. Static
# ---------------------------------------------------------------------
for f in "$PY_SRC" "$HELPER_SRC" "$CU_SRC" "$PREV_CU" "$OLD_CU"; do
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
grep -qE '^REV_TAG:str="403"' "$CODE" && pass "source_rev_tag_is_403" || fail "source_rev_tag_is_403" "wrong REV_TAG"
grep -qF 'CRunnerEntry(14,"./403_kernel_maxd14","NQ_EXTRA_CTX=1 NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_points_at_403_keeps_helper_prefix" || fail "source_table_points_at_403_keeps_helper_prefix" "table entry wrong"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=${MB}\b" "$CODE" && pass "source_default_max_blocks_is_${MB}" || fail "source_default_max_blocks_is_${MB}" "default not $MB"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_402_r5Py (removed=3 added=3 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_402_r5Py" "removed=$PR_ added=$PA_, expected 3/3"
else info "py_diff_fingerprint_vs_402_r5Py" "skipped"; fi
cu_code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"; cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" != "$KERNEL_SHA_402"* ]] && pass "cu_kernel_region_CHANGED_from_402 (now ${KB:0:16}...)" || fail "cu_kernel_region_CHANGED_from_402" "kernel region is still 402's -- the new layout is not in this file"
CR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true); CA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
[[ "$CR_" == "$EXPECT_REMOVED" && "$CA_" == "$EXPECT_ADDED" ]] && pass "cu_diff_fingerprint_vs_402_r5 (removed=$CR_ added=$CA_)" || fail "cu_diff_fingerprint_vs_402_r5" "removed=$CR_ added=$CA_, expected $EXPECT_REMOVED/$EXPECT_ADDED"
grep -q 'PACK402' "/tmp/${REV}_cur_code.cu" && fail "cu_no_402_layout_left" "PACK402 still referenced" || pass "cu_no_402_layout_left"
[[ "$(grep -c 'stack_b\[stack_ptr\] = (cur_rd & ~1u) | ((cur_ld >> 20) & 1u);' "/tmp/${REV}_cur_code.cu")" == "2" ]] && pass "cu_word_B_rd31_plus_ld20_at_both_pushes" || fail "cu_word_B_rd31_plus_ld20_at_both_pushes" "word B layout missing at a push site"
grep -q 'cur_ld  = ((uint32_t)(packed_a >> 44) & PACK403_LDLO) | ((packed_b & 1u) << 20);' "/tmp/${REV}_cur_code.cu" && grep -q 'cur_rd  = packed_b & ~1u;' "/tmp/${REV}_cur_code.cu" && pass "cu_pop_restores_ld21_rd31" || fail "cu_pop_restores_ld21_rd31" "pop layout missing"
[[ "$(grep -c 'if (N > PACK403_WIDTH) {' "/tmp/${REV}_cur_code.cu")" == "2" ]] && pass "cu_N_gt_22_refused_in_both_mains" || fail "cu_N_gt_22_refused_in_both_mains" "guard count != 2"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"/tmp/${REV}_gcc.log" && pass "cu_cpu_harness_builds" || fail "cu_cpu_harness_builds" "$(head -3 /tmp/${REV}_gcc.log)"
grep -q "rev403" "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev403 note in header"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Build + W1
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
banner "Building"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
ptxas_of() { FRAME="$(grep -A3 'kernel_dfs_iter_gpu_maxd14' "$1" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"
  SPILLS="$(grep -A3 'kernel_dfs_iter_gpu_maxd14' "$1" | grep -o '[0-9]* bytes spill stores, [0-9]* bytes spill loads' | head -1)"
  REGS="$(grep -A3 'kernel_dfs_iter_gpu_maxd14' "$1" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"; }
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc_403.log" 2>&1
[[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "$(grep -i error "$LOGDIR/05_nvcc_403.log" | head -3)"; summary_exit; }
for pair in "$PREV_CU:$PREV_BIN" "$OLD_CU:$OLD_BIN"; do src="${pair%%:*}"; bin="${pair##*:}"
  if [[ ! -x "$bin" ]]; then "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$bin" "$src" -lcuda > "$LOGDIR/05_nvcc_${bin}.log" 2>&1; fi
  [[ -x "$bin" ]] && pass "reference_binary_available[$bin]" || { fail "reference_binary_available[$bin]" "build failed"; summary_exit; }
done
ptxas_of "$LOGDIR/05_nvcc_403.log"; CF="${FRAME:-?}"; CREG="${REGS:-?}"; CS="${SPILLS:-?}"
info "ptxas 403" "frame=${CF}B regs=${CREG} ${CS}"
if [[ "$CF" == "160" && "$CS" == "0 bytes spill stores, 0 bytes spill loads" && "$CREG" =~ ^[0-9]+$ && "$CREG" -le 44 ]]; then pass "W1_ptxas_frame160_spill0_regs${CREG}"
elif [[ "$CF" =~ ^[0-9]+$ && "$CF" -le 168 ]]; then pass "W1_frame_ok_outside_band (frame=${CF}B regs=${CREG} ${CS}) -- continuing"
else fail "W1_frame_grew" "frame=${CF}B regs=${CREG} ${CS}"; summary_exit; fi
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/06_codon.log"
[[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN]" || { fail "codon_build" "see log"; summary_exit; }

# ---------------------------------------------------------------------
# 3. Inputs (from crunner logs / known names), W2, W3
# ---------------------------------------------------------------------
IN21=""; for lg in $(ls -t 40*_crunner_logs/crunner_*_N21.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { IN21="$c"; break; }; done
[[ -f "${IN21:-/nonexistent}" && "$(stat -c %s "$IN21")" == "56707896" ]] && pass "input21_located ($IN21)" || { fail "input21_located" "no N=21 sched input via 40*_crunner_logs"; summary_exit; }
IN22=""; for lg in $(ls -t 40*_crunner_logs/crunner_*_N22.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { IN22="$c"; break; }; done
[[ -f "${IN22:-/nonexistent}" ]] || IN22="$(ls -t constellations_N22_*.sched394f.bin 2>/dev/null | head -1)"
if [[ -f "${IN22:-/nonexistent}" ]]; then info "input22_located" "$IN22 ($(stat -c %s "$IN22") bytes, $(( $(stat -c %s "$IN22") / 28 )) records)"; else info "input22" "not present -- W2 at N=22 will be skipped, -g 22 22 will generate it (minutes)"; fi

banner "W2: CPU-harness per-record equality"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_prev" "$PREV_CU" -lm 2>"$LOGDIR/03_gcc_prev.log" && "$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_old" "$OLD_CU" -lm 2>"$LOGDIR/03_gcc_old.log" || { fail "W2_cpu_reference_builds" "see 03_gcc_*.log"; summary_exit; }
head -c $((CPU_CHECK_RECORDS*28)) "$IN21" > "/tmp/${REV}_cpu_in21.bin"
"/tmp/${REV}_cpu_prev" 21 "/tmp/${REV}_cpu_in21.bin" "/tmp/${REV}_cpu_prev21.bin" 2>&1 | tail -1 | tee "$LOGDIR/04_cpu_prev21.log"
"/tmp/${REV}_cpu_cur"  21 "/tmp/${REV}_cpu_in21.bin" "/tmp/${REV}_cpu_cur21.bin"  2>&1 | tail -1 | tee "$LOGDIR/04_cpu_cur21.log"
if cmp -s "/tmp/${REV}_cpu_prev21.bin" "/tmp/${REV}_cpu_cur21.bin" && [[ -s "/tmp/${REV}_cpu_cur21.bin" ]]; then pass "W2a_cpu_N21_identical_402_r5_vs_403 ($CPU_CHECK_RECORDS records)"; else fail "W2a_cpu_N21_DIFFER" "403 changes N=21 results on CPU -- stopping"; summary_exit; fi
if [[ -f "${IN22:-/nonexistent}" ]]; then
  head -c $((CPU_CHECK_RECORDS_22*28)) "$IN22" > "/tmp/${REV}_cpu_in22.bin"
  "/tmp/${REV}_cpu_old" 22 "/tmp/${REV}_cpu_in22.bin" "/tmp/${REV}_cpu_old22.bin" 2>&1 | tail -1 | tee "$LOGDIR/04_cpu_old22.log"
  "/tmp/${REV}_cpu_cur" 22 "/tmp/${REV}_cpu_in22.bin" "/tmp/${REV}_cpu_cur22.bin" 2>&1 | tail -1 | tee "$LOGDIR/04_cpu_cur22.log"
  if cmp -s "/tmp/${REV}_cpu_old22.bin" "/tmp/${REV}_cpu_cur22.bin" && [[ -s "/tmp/${REV}_cpu_cur22.bin" ]]; then pass "W2b_cpu_N22_identical_401_r4_vs_403 ($CPU_CHECK_RECORDS_22 records)"; else fail "W2b_cpu_N22_DIFFER" "403 changes N=22 results on CPU -- a dead-bit claim is wrong; stopping"; summary_exit; fi
else info "W2b_cpu_N22" "skipped (no N=22 input yet)"; W2B_PENDING=1; fi

banner "W3: N=23 must be refused"
"./$CU_BIN" 23 "$IN21" "/tmp/${REV}_n23.bin" > "$LOGDIR/07_n23_refusal.log" 2>&1; rc=$?
{ [[ "$rc" == "3" ]] && grep -q '\[403-pack\] N=23 unsupported' "$LOGDIR/07_n23_refusal.log"; } && pass "W3_N23_refused (rc=3)" || { fail "W3_N23_refused" "rc=$rc"; summary_exit; }

# ---------------------------------------------------------------------
# 4. GPU cells
# ---------------------------------------------------------------------
printf 'cell\tN\tbinary\tpath\tmax_blocks\thelpers\thelper_mb\tfree_mb\tk_per_thread_max\tkernel_ms\ttotal_sum\tmatch\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
CLKPID=""
gpu_gate() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_before_$1.txt"
  [[ -z "$apps" ]] && return 0; fail "gpu_empty_before[$1]" "compute process(es) present: $(echo "$apps" | tr '\n' ' ')"; return 1; }
clk_start() { local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"
  ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPID=$!; }
clk_stop() { [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.0f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_$1.tsv"); }
fields_of() { KMS="$(grep -o 'kernel_ms=[0-9.]*' "$1" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$1" | head -1 | cut -d= -f2)"
  MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$1" && MATCH=1; FREE="$(grep -o 'free_mb=[0-9]*' "$1" | head -1 | cut -d= -f2)"
  CFGMB="$(grep -o 'MAX_BLOCKS=[0-9]*' "$1" | head -1 | cut -d= -f2)"; HL="$(grep -o 'helpers=[0-9]*' "$1" | head -1 | cut -d= -f2)"; HM="$(grep -o 'helper_mb=[0-9]*' "$1" | head -1 | cut -d= -f2)"
  KPT="$(grep -o 'k_per_thread_max=[0-9]*' "$1" | head -1 | cut -d= -f2)"; }
declare -A VAL SMM
record() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$TSV"; }
run_direct() {  # cell N bin input mb out
  local cell="$1" n="$2" bin="$3" in="$4" mb="$5" out="$6" orc; orc="$(oracle_of "$n")"
  gpu_gate "$cell" || return 1
  local lg="$LOGDIR/3_${cell}.log" start; start="$(date -Is)"; clk_start "$cell"
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$mb" NQ_HELPER_CTX="$HCTX" NQ_HELPER_MB="$HMB" "./$bin" "$n" "$in" "$out" "$orc" > "$lg" 2>&1 || true
  clk_stop "$cell"; fields_of "$lg"
  record "$cell" "$n" "$bin" direct "$mb" "${HL:-?}" "${HM:-?}" "${FREE:-?}" "${KPT:-?}" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start"
  [[ "$MATCH" == "1" && "${TOT:-}" == "$orc" ]] || { fail "oracle[$cell]" "total_sum='${TOT:-<none>}' expected $orc"; return 1; }
  VAL[$cell]="$KMS"; SMM[$cell]="$SMMEAN"
  info "$cell" "$bin N=$n mb=$mb helpers=$HL/$HM kernel_ms=$KMS free_mb=$FREE k=$KPT sm_mean=$SMMEAN"
}
run_g() {  # cell N
  local cell="$1" n="$2" orc; orc="$(oracle_of "$n")"; local gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${n}.log"; rm -f "$gcr"
  gpu_gate "$cell" || return 1
  local start; start="$(date -Is)"; clk_start "$cell"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX -u NQ_BLOCK -u NQ_CARVEOUT -u NQ_HELPER_CTX -u NQ_HELPER_MB "./$PY_BIN" -g "$n" "$n" > "$LOGDIR/2_${cell}_console.log" 2>&1
  clk_stop "$cell"
  cp "$gcr" "$LOGDIR/2_${cell}_crunner.log" 2>/dev/null || { fail "crunner_path_taken[$cell]" "no $gcr"; return 1; }
  cp "$CRLOG_DIR/dispatch.log" "$LOGDIR/2_${cell}_dispatch.log" 2>/dev/null || true
  fields_of "$gcr"
  record "$cell" "$n" "$CU_BIN" dispatch "${CFGMB:-?}" "${HL:-?}" "${HM:-?}" "${FREE:-?}" "${KPT:-?}" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start"
  [[ "$MATCH" == "1" && "${TOT:-}" == "$orc" ]] || { fail "oracle[$cell]" "total_sum='${TOT:-<none>}' expected $orc"; return 1; }
  VAL[$cell]="$KMS"; SMM[$cell]="$SMMEAN"; G_IN="$(grep -o 'src=[^ ]*' "$gcr" | head -1 | cut -d= -f2)"
  info "$cell" "-g $n $n: MAX_BLOCKS=$CFGMB helpers=$HL/$HM kernel_ms=$KMS free_mb=$FREE k=$KPT sm_mean=$SMMEAN src=$G_IN"
}

banner "W4: N=21 full input @$MB, helper $HCTX+$HMB MB -- 402_r5 vs 403 must be byte-identical"
run_direct E0 21 "$PREV_BIN" "$IN21" "$MB" "/tmp/${REV}_E0.bin" || summary_exit
sleep "$COOLDOWN"
run_direct E1 21 "$CU_BIN"   "$IN21" "$MB" "/tmp/${REV}_E1.bin" || summary_exit
if cmp -s "/tmp/${REV}_E0.bin" "/tmp/${REV}_E1.bin" && [[ -s "/tmp/${REV}_E1.bin" ]]; then pass "W4_N21_full_input_per_thread_identical ($(stat -c %s "/tmp/${REV}_E1.bin") bytes)"; else fail "W4_N21_full_input_DIFFER" "timings are NOT reported"; summary_exit; fi
p="$(pct "${VAL[E1]}" "${VAL[E0]}")"; a="$(abspct "${VAL[E1]}" "${VAL[E0]}")"
le "$a" 0.3 && pass "W5_layout_cost_free_at_N21 (E1 ${p}% vs E0=${VAL[E0]})" || fail "W5_layout_cost" "E1 ${p}% vs E0 (registered |.| <= 0.3%)"
d="$(abspct "${VAL[E0]}" "$ANCHOR_E0")"; le "$d" 0.15 && pass "E0_anchor (${d}% from 402-r5 G21 mean)" || info "E0_anchor" "${d}% from $ANCHOR_E0 -- read E1 relative to E0"
sleep "$COOLDOWN"

if [[ "$GATES_ONLY" != "1" ]]; then
  banner "G22: -g 22 22 through the dispatcher (403 table: MB=$MB, helper) -- W6/W7"
  run_g G22 22 || summary_exit
  pass "W6_N22_oracle_MATCH (G22 kernel_ms=${VAL[G22]})"
  [[ -f "${IN22:-/nonexistent}" ]] || IN22="$G_IN"
  if [[ "${W2B_PENDING:-0}" == "1" && -f "${IN22:-/nonexistent}" ]]; then
    banner "W2b (deferred): CPU per-record equality at N=22 on the input -g just produced"
    head -c $((CPU_CHECK_RECORDS_22*28)) "$IN22" > "/tmp/${REV}_cpu_in22.bin"
    "/tmp/${REV}_cpu_old" 22 "/tmp/${REV}_cpu_in22.bin" "/tmp/${REV}_cpu_old22.bin" 2>&1 | tail -1 | tee "$LOGDIR/04_cpu_old22.log"
    "/tmp/${REV}_cpu_cur" 22 "/tmp/${REV}_cpu_in22.bin" "/tmp/${REV}_cpu_cur22.bin" 2>&1 | tail -1 | tee "$LOGDIR/04_cpu_cur22.log"
    cmp -s "/tmp/${REV}_cpu_old22.bin" "/tmp/${REV}_cpu_cur22.bin" && [[ -s "/tmp/${REV}_cpu_cur22.bin" ]] && pass "W2b_cpu_N22_identical_401_r4_vs_403 ($CPU_CHECK_RECORDS_22 records)" || fail "W2b_cpu_N22_DIFFER" "403 changes N=22 results on CPU"
  fi
  r="$(awk -v g="${VAL[G22]}" -v b="$BASE22" 'BEGIN{printf "%.4f", g/b}')"
  if le "${VAL[G22]}" 969840; then pass "W7_N22_gains_carry (G22=${VAL[G22]} ms = ${r} x 9-Sep baseline $BASE22; $(pct "${VAL[G22]}" "$BASE22")%)"
  elif le "${VAL[G22]}" 1050000; then pass "W7_partial (G22=${VAL[G22]} ms = ${r} x baseline; between the registered outcomes)"
  else fail "W7_REFUTED_N22_behaves_differently" "G22=${VAL[G22]} ms = ${r} x baseline $BASE22"; fi
  if [[ "$FULL22" == "1" ]]; then
    sleep "$COOLDOWN"; banner "FULL22: 401_r4 direct N=22 @$MB with helper, per-thread equality vs G22"
    run_direct R22 22 "$OLD_BIN" "$IN22" "$MB" "/tmp/${REV}_R22.bin" || summary_exit
    if cmp -s "/tmp/${REV}_R22.bin" "/tmp/crunner_N22_results.bin" 2>/dev/null; then pass "FULL22_N22_per_thread_identical_401_r4_vs_403"; else fail "FULL22_N22_per_thread_DIFFER_or_missing" "/tmp/crunner_N22_results.bin vs R22 -- both sums matched the oracle; inspect"; fi
    info "N22_frame_effect" "401_r4 (208 B) @$MB = ${VAL[R22]} ms vs 403 (160 B) @$MB = ${VAL[G22]} ms ($(pct "${VAL[G22]}" "${VAL[R22]}")%)"
  fi
fi
CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "W9_sm_clock_1710_all_cells" || fail "W9_sm_clock_1710_all_cells" "see clk_*.tsv"

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
print(f"{'cell':<5}{'N':>3}{'binary':<22}{'path':>9}{'MB':>5}{'help':>6}{'free_mb':>8}{'k':>6}{'kernel_ms':>14}{'match':>6}{'sm':>6}{'h:mm:ss':>10}")
for r in rows:
    v=float(r['kernel_ms']); s=v/1000
    print(f"{r['cell']:<5}{r['N']:>3}{r['binary']:<22}{r['path']:>9}{r['max_blocks']:>5}{r['helpers']+'/'+r['helper_mb']:>6}{r['free_mb']:>8}{r['k_per_thread_max']:>6}{v:>14.3f}{r['match']:>6}{r['sm_mean']:>6}{int(s//3600):>4}:{int(s%3600//60):02d}:{s%60:04.1f}")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs" 2>/dev/null || true
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
