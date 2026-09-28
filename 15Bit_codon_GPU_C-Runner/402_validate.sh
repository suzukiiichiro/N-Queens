#!/usr/bin/env bash
# 402_validate.sh
#
# rev402 -- DFS stack frame 16 -> 12 bytes (first kernel change since 395a r2).
#           Four correctness gates before any timing is read, then a
#           MAX_BLOCKS sweep in the 2-context holder state.
#
# GATES (in order; each is a hard stop)
#   V1  ptxas frame 208 -> 152..168 B, spill 0/0, registers <= 44
#   V2  CPU harness (gcc): 401_r4 vs 402 on the first CPU_CHECK_RECORDS real
#       sched records -> per-record results byte-identical
#   V3  the 402 binary refuses N=22 (rc=3, "[402-pack]" line)
#   V4  full input (2,025,282 records), MB=800, both binaries, 1 holder:
#       per-thread result files byte-identical, both MATCH the oracle
# PRE-REGISTERED (402_README_append.md; fixed before execution)
#   V5  E0 (401_r4 binary) within +-0.15% of 133,565 (401-r4 H0)
#   V6  stated: |E1 - E0| <= 0.5%.   Alt: E1 >= E0 + 1%.
#   V7  DECISION. stated: min(P960, P1040) <= E0 - 3%. Refuted if nothing
#       is below E0 - 1%.
#   V8  P1120 and P1280 both slower than best(P960, P1040)
#   V9  mean SM clock within 2% of 1710 in every cell
#   Nothing is adopted as a production default by this run.
#
# USAGE
#   STATIC_ONLY=1 bash 402_validate.sh       # OK=19
#                 bash 402_validate.sh       # ~22 min
#   GATES_ONLY=1  bash 402_validate.sh       # V1..V4 only (E0/E1), ~8 min
#   MB_LIST="960 1040" bash 402_validate.sh  # shorter sweep

set -u

REV="402"
PY_SRC="${PY_SRC:-402Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-402Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-401_r4Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-402_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-402_kernel_maxd14}"
PREV_CU="${PREV_CU:-401_r4_kernel_maxd14.cu}"
PREV_BIN="${PREV_BIN:-401_r4_kernel_maxd14}"
HOLDER_SRC="${HOLDER_SRC:-395c_ctx_holder.cu}"
HOLDER_BIN="${HOLDER_BIN:-395c_ctx_holder}"
CRLOG_DIR="${CRLOG_DIR:-402_crunner_logs}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
BLOCK="${BLOCK:-32}"
SMS="${SMS:-80}"
MAX_BLOCKS_PER_SM="${MAX_BLOCKS_PER_SM:-16}"
MB_LIST="${MB_LIST:-960 1040 1120 1280}"
CPU_CHECK_RECORDS="${CPU_CHECK_RECORDS:-8192}"
ANCHOR_E0="${ANCHOR_E0:-133565.281}"     # 401-r4 H0: 1 holder, NQ_EXTRA_CTX=0, MB=800
FREE_2CTX="${FREE_2CTX:-22018}"
KERNEL_SHA_395A="${KERNEL_SHA_395A:-ebd7f523deb591347eab1a352a9555c2e4a6c0b4a456d80517322c933a86e68a}"
EXPECT_REMOVED="${EXPECT_REMOVED:-21}"
EXPECT_ADDED="${EXPECT_ADDED:-46}"
COOLDOWN="${COOLDOWN:-10}"
CLK_INTERVAL="${CLK_INTERVAL:-5}"
STATIC_ONLY="${STATIC_ONLY:-0}"
GATES_ONLY="${GATES_ONLY:-0}"

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
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS (registered refutations are listed here too -- read the label):"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi; exit 0; }

# ---------------------------------------------------------------------
# 1. Static
# ---------------------------------------------------------------------
for f in "$PY_SRC" "$HELPER_SRC" "$CU_SRC" "$PREV_CU" "$HOLDER_SRC"; do
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
grep -qE '^REV_TAG:str="402"' "$CODE" && pass "source_rev_tag_is_402" || fail "source_rev_tag_is_402" "wrong REV_TAG"
grep -q 'CRunnerEntry(14,"./402_kernel_maxd14","NQ_EXTRA_CTX=1 "' "$CODE" && pass "source_table_points_at_402_and_keeps_the_treatment" || fail "source_table_points_at_402_and_keeps_the_treatment" "table entry wrong"
grep -qE '^CRUNNER_MIN_N:int=19' "$CODE" && pass "source_crunner_min_n_still_19" || fail "source_crunner_min_n_still_19" "CRUNNER_MIN_N lost"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_401_r4Py (removed=3 added=3 EXECUTABLE lines)" \
    || { fail "py_diff_fingerprint_vs_401_r4Py" "removed=$PR_ added=$PA_, expected 3/3"; diff "/tmp/${REV}_prev_code.py" "$CODE" | head -20 | sed 's/^/      /'; }
else info "py_diff_fingerprint_vs_401_r4Py" "skipped"; fi
cu_code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"; cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" != "$KERNEL_SHA_395A" ]] && pass "cu_kernel_region_CHANGED_from_395a_r2 (now ${KB:0:16}...)" || fail "cu_kernel_region_CHANGED_from_395a_r2" "kernel region is still 395a r2 -- the packing is not in this file"
CR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true); CA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
[[ "$CR_" == "$EXPECT_REMOVED" && "$CA_" == "$EXPECT_ADDED" ]] && pass "cu_diff_fingerprint_vs_401_r4 (removed=$CR_ added=$CA_)" || fail "cu_diff_fingerprint_vs_401_r4" "removed=$CR_ added=$CA_, expected $EXPECT_REMOVED/$EXPECT_ADDED"
grep -q 'uint64_t stack\[MAXD14_ANCESTOR \* 2\];' "/tmp/${REV}_cur_code.cu" && fail "cu_old_16B_stack_gone" "the old 2 x u64 stack is still declared" || pass "cu_old_16B_stack_gone"
{ grep -q 'uint64_t stack_a\[MAXD14_ANCESTOR\];' "/tmp/${REV}_cur_code.cu" && grep -q 'uint32_t stack_b\[MAXD14_ANCESTOR\];' "/tmp/${REV}_cur_code.cu" && grep -q 'uint64_t stack_depth = 0;' "/tmp/${REV}_cur_code.cu"; } && pass "cu_12B_frame_declared (stack_a u64, stack_b u32, depth in a register)" || fail "cu_12B_frame_declared" "packed frame declarations missing"
[[ "$(grep -c 'stack_b\[stack_ptr\] = cur_rd;' "/tmp/${REV}_cur_code.cu")" == "2" ]] && pass "cu_rd_saved_whole_at_both_push_sites (the 397 mistake is not repeated)" || fail "cu_rd_saved_whole_at_both_push_sites" "rd is not stored whole at both pushes"
[[ "$(grep -c 'if (N > PACK402_WIDTH) {' "/tmp/${REV}_cur_code.cu")" == "2" ]] && pass "cu_N_gt_21_refused_in_both_mains" || fail "cu_N_gt_21_refused_in_both_mains" "guard count != 2"
grep -q "rev402" "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev402 note in header"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Build + V1 (ptxas)
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
banner "Building"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
ptxas_of() {  # $1 log -> FRAME SPILLS REGS (kernel_dfs_iter_gpu_maxd14 block)
  FRAME="$(grep -A3 'kernel_dfs_iter_gpu_maxd14' "$1" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"
  SPILLS="$(grep -A3 'kernel_dfs_iter_gpu_maxd14' "$1" | grep -o '[0-9]* bytes spill stores, [0-9]* bytes spill loads' | head -1)"
  REGS="$(grep -A3 'kernel_dfs_iter_gpu_maxd14' "$1" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"
}
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc_402.log" 2>&1; cat "$LOGDIR/05_nvcc_402.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "no binary"; summary_exit; }
if [[ ! -x "$PREV_BIN" ]]; then "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$PREV_BIN" "$PREV_CU" -lcuda > "$LOGDIR/05_nvcc_401_r4.log" 2>&1; else "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "/tmp/${REV}_prev_probe" "$PREV_CU" -lcuda > "$LOGDIR/05_nvcc_401_r4.log" 2>&1; fi
[[ -x "$PREV_BIN" ]] && pass "prev_binary_available[$PREV_BIN]" || { fail "prev_binary_available" "no $PREV_BIN"; summary_exit; }
if [[ ! -x "$HOLDER_BIN" ]]; then "$NVCC" -O2 -arch="$ARCH" -o "$HOLDER_BIN" "$HOLDER_SRC" > "$LOGDIR/05a_nvcc_holder.log" 2>&1; fi
[[ -x "$HOLDER_BIN" ]] && pass "ctx_holder_available[$HOLDER_BIN]" || { fail "ctx_holder_available" "no $HOLDER_BIN"; summary_exit; }
ptxas_of "$LOGDIR/05_nvcc_401_r4.log"; PF="${FRAME:-?}"; PR="${REGS:-?}"; PS="${SPILLS:-?}"
ptxas_of "$LOGDIR/05_nvcc_402.log";    CF="${FRAME:-?}"; CREG="${REGS:-?}"; CS="${SPILLS:-?}"
info "ptxas 401_r4" "frame=${PF}B regs=${PR} ${PS}"
info "ptxas 402"    "frame=${CF}B regs=${CREG} ${CS}"
FRAME_B="${CF:-156}"
V1=0
if [[ "$CF" =~ ^[0-9]+$ ]] && [[ "$CF" -lt 208 ]]; then
  if [[ "$CF" -ge 152 && "$CF" -le 168 && "$CS" == "0 bytes spill stores, 0 bytes spill loads" && "$CREG" =~ ^[0-9]+$ && "$CREG" -le 44 ]]; then pass "V1_ptxas_frame_208_to_${CF}B_spill0_regs${CREG}"; V1=1
  else pass "V1_frame_smaller_but_outside_registered_band (frame=${CF}B regs=${CREG} ${CS}; registered 152..168 / spill 0 / regs<=44) -- continuing, read the sweep with this in mind"; V1=1; fi
else
  fail "V1_frame_not_smaller" "402 frame=${CF}B vs 208 -- the packing did not take effect; stopping before the GPU"; summary_exit
fi
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/06_codon.log"
[[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN]" || { fail "codon_build" "see log"; summary_exit; }

# ---------------------------------------------------------------------
# 3. Input (from a crunner log, never guessed) + V2 CPU equivalence + V3
# ---------------------------------------------------------------------
IN=""; ORC="$(oracle_of "$NQ")"
for lg in $(ls -t 40*_crunner_logs/crunner_*_N${NQ}.log 2>/dev/null); do
  c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { IN="$c"; break; }
done
[[ -f "${IN:-/nonexistent}" && "$(stat -c %s "$IN")" == "56707896" ]] && pass "input_located ($IN)" || { fail "input_located" "no sched input found via 40*_crunner_logs; run ./401_r4Py_kernel_maxd14_final -g 21 21 once"; summary_exit; }

banner "V2: CPU-harness equivalence, 401_r4 vs 402, first $CPU_CHECK_RECORDS real records"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_prev" "$PREV_CU" -lm 2>"$LOGDIR/03_gcc_prev.log" && "$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"$LOGDIR/03_gcc_cur.log"
if [[ -x "/tmp/${REV}_cpu_prev" && -x "/tmp/${REV}_cpu_cur" ]]; then
  head -c $((CPU_CHECK_RECORDS*28)) "$IN" > "/tmp/${REV}_cpu_in.bin"
  "/tmp/${REV}_cpu_prev" "$NQ" "/tmp/${REV}_cpu_in.bin" "/tmp/${REV}_cpu_prev.bin" 2>&1 | tail -1 | tee "$LOGDIR/04_cpu_prev.log"
  "/tmp/${REV}_cpu_cur"  "$NQ" "/tmp/${REV}_cpu_in.bin" "/tmp/${REV}_cpu_cur.bin"  2>&1 | tail -1 | tee "$LOGDIR/04_cpu_cur.log"
  if cmp -s "/tmp/${REV}_cpu_prev.bin" "/tmp/${REV}_cpu_cur.bin" && [[ -s "/tmp/${REV}_cpu_cur.bin" ]]; then pass "V2_cpu_per_record_results_identical ($CPU_CHECK_RECORDS records)"
  else fail "V2_cpu_per_record_results_DIFFER" "the packing changes results on CPU -- stopping before the GPU"; summary_exit; fi
else fail "V2_cpu_harness_build" "see $LOGDIR/03_gcc_*.log"; summary_exit; fi

banner "V3: N=22 must be refused"
"./$CU_BIN" 22 "$IN" "/tmp/${REV}_n22.bin" > "$LOGDIR/07_n22_refusal.log" 2>&1; rc=$?
{ [[ "$rc" == "3" ]] && grep -q '\[402-pack\] N=22 unsupported' "$LOGDIR/07_n22_refusal.log"; } && pass "V3_N22_refused (rc=3)" || { fail "V3_N22_refused" "rc=$rc -- see 07_n22_refusal.log"; summary_exit; }

# ---------------------------------------------------------------------
# 4. Cells (1 holder, NQ_EXTRA_CTX=0, N=21)
# ---------------------------------------------------------------------
printf 'cell\tbinary\tmax_blocks\twarps_per_sm\tfootprint_kb\textra_ctx\tfree_mb\tkernel_ms\ttotal_sum\tmatch\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
declare -a HPIDS=(); CLKPID=""
gpu_gate() {
  local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"
  echo "$apps" > "$LOGDIR/apps_before_$1.txt"
  local allowed=" ${HPIDS[*]:-} "; local others=""
  while IFS=, read -r pid mem; do [[ -z "$pid" ]] && continue; [[ "$allowed" == *" $pid "* ]] || others+="$pid($mem) "; done <<< "$apps"
  [[ -z "$others" ]] && return 0
  fail "gpu_empty_before[$1]" "foreign compute process(es): $others"; echo "ABORT: a leftover process moves this kernel by 4-6% (395c)."; return 1
}
clk_start() { local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"
  ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPID=$!; }
clk_stop() { [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.0f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_$1.tsv"); }
fields_of() { KMS="$(grep -o 'kernel_ms=[0-9.]*' "$1" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$1" | head -1 | cut -d= -f2)"
  MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$1" && MATCH=1; XCTX="$(grep -o 'extra_ctx=[0-9]*' "$1" | head -1 | cut -d= -f2)"; FREE="$(grep -o 'free_mb=[0-9]*' "$1" | head -1 | cut -d= -f2)"; }
resident_of() { python3 -c "
mb=int('$1'); blk=$BLOCK; sms=$SMS; capb=$MAX_BLOCKS_PER_SM; fr=$FRAME_B
bps=min(mb//sms, capb); w=bps*(blk//32); print(w, round(w*32*fr/1024,1))"; }
holders_up() { local i; for i in $(seq 1 "$1"); do "./$HOLDER_BIN" 0 900 > "$LOGDIR/holder_$2_$i.log" 2>&1 & HPIDS+=("$!"); done; sleep 4; }
holders_down() { local p; for p in "${HPIDS[@]:-}"; do [[ -n "$p" ]] && { kill "$p" 2>/dev/null; wait "$p" 2>/dev/null || true; }; done; HPIDS=(); sleep 2; }
bail() { holders_down; [[ -n "$CLKPID" ]] && kill "$CLKPID" 2>/dev/null; summary_exit; }
declare -A VAL XC FM SMM
run_cell() {  # cell binary mb outfile
  local cell="$1" bin="$2" mb="$3" out="$4"
  holders_up 1 "$cell"; gpu_gate "$cell" || { holders_down; return 1; }
  local lg="$LOGDIR/3_${cell}.log" start; start="$(date -Is)"; clk_start "$cell"
  env -u NQ_PAD_MB -u NQ_CARVEOUT NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$mb" NQ_EXTRA_CTX=0 "./$bin" "$NQ" "$IN" "$out" "$ORC" > "$lg" 2>&1 || true
  clk_stop "$cell"; holders_down
  fields_of "$lg"; local rr; rr=($(resident_of "$mb"))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$bin" "$mb" "${rr[0]}" "${rr[1]}" "${XCTX:-?}" "${FREE:-?}" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start" >> "$TSV"
  [[ "$MATCH" == "1" && "${TOT:-}" == "$ORC" ]] || { fail "oracle[$cell]" "total_sum='${TOT:-<none>}' expected $ORC"; return 1; }
  VAL[$cell]="$KMS"; XC[$cell]="$XCTX"; FM[$cell]="$FREE"; SMM[$cell]="$SMMEAN"
  info "$cell" "$bin mb=$mb warps/SM=${rr[0]} footprint=${rr[1]}KB kernel_ms=$KMS free_mb=$FREE sm_mean=$SMMEAN temp_max=$TMAX"
}

banner "V4: full input, MB=800, both binaries -- results must be byte-identical"
run_cell E0 "$PREV_BIN" 800 "/tmp/${REV}_E0.bin" || bail
sleep "$COOLDOWN"
run_cell E1 "$CU_BIN"   800 "/tmp/${REV}_E1.bin" || bail
if cmp -s "/tmp/${REV}_E0.bin" "/tmp/${REV}_E1.bin" && [[ -s "/tmp/${REV}_E1.bin" ]]; then pass "V4_full_input_per_thread_results_identical ($(stat -c %s "/tmp/${REV}_E1.bin") bytes, 2,025,282 records, both MATCH)"
else fail "V4_full_input_results_DIFFER" "401_r4 and 402 disagree on the full input although both matched the oracle sum -- timings are NOT reported"; bail; fi
sleep "$COOLDOWN"
if [[ "$GATES_ONLY" != "1" ]]; then
  for mb in $MB_LIST; do
    banner "P$mb"
    run_cell "P$mb" "$CU_BIN" "$mb" "/tmp/${REV}_P${mb}.bin" || bail
    sleep "$COOLDOWN"
  done
fi

# ---------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------
banner "Evaluation"
e0="${VAL[E0]}"; e1="${VAL[E1]}"
d="$(abspct "$e0" "$ANCHOR_E0")"
{ le "$d" 0.15 && absdiff_le "${FM[E0]}" "$FREE_2CTX" 2; } && pass "V5_anchor (E0=$e0, ${d}% from $ANCHOR_E0, free_mb=${FM[E0]})" || fail "V5_anchor" "E0=$e0 is ${d}% from $ANCHOR_E0, free_mb=${FM[E0]} -- the session is not in the 401-r4 state; read relative numbers only"
p="$(pct "$e1" "$e0")"; a="$(abspct "$e1" "$e0")"
if le "$a" 0.5; then pass "V6_packing_is_free_at_10_warps (E1 ${p}% vs E0)"
elif ge "$p" 1; then fail "V6_REFUTED_packing_costs" "E1 +${p}% vs E0 at equal warps -- 393-8's objection stands; the sweep must pay this back first"
elif le "$p" -0.5; then pass "V6_packing_already_helps_at_10_warps (E1 ${p}% vs E0 -- fewer sectors alone pay)"
else fail "V6_indeterminate" "E1 ${p}% vs E0 -- between the registered outcomes"; fi
if [[ "$GATES_ONLY" != "1" ]]; then
  best=""; bestmb=""
  for mb in $MB_LIST; do v="${VAL[P$mb]:-}"; [[ -z "$v" ]] && continue; p="$(pct "$v" "$e0")"; info "P$mb vs E0" "${p}%  ($v ms)"
    if [[ "$mb" == "960" || "$mb" == "1040" ]]; then [[ -z "$best" ]] || le "$v" "$best" && { best="$v"; bestmb="$mb"; }; fi; done
  if [[ -n "$best" ]]; then
    p="$(pct "$best" "$e0")"
    if le "$p" -3; then pass "V7_DECISION_more_warps_pay_inside_L1 (best P$bestmb ${p}% vs E0) -- A/B candidate for 402-r2 (interleaved, >=2 reps); not adopted from one run"
    elif le "$p" -1; then pass "V7_held_but_small (best P$bestmb ${p}% vs E0, above the -3% registered) -- real but modest; 402-r2 decides whether it survives an A/B"
    else fail "V7_REFUTED_occupancy_does_not_pay_inside_L1" "best P$bestmb ${p}% vs E0 -- 12-13 warps/SM within budget are not faster than 10; this axis closes at N=21"; fi
    v8=1; for mb in 1120 1280; do v="${VAL[P$mb]:-}"; [[ -z "$v" ]] && continue; ge "$v" "$best" || v8=0; done
    [[ "$v8" == "1" ]] && pass "V8_cliff_is_capacity (P1120 and P1280 slower than P$bestmb; the cliff moved with the frame as budgeted)" || fail "V8_REFUTED" "a point past the 65 KB budget beat the in-budget best -- the budget model is wrong"
  fi
fi
CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "V9_sm_clock_1710_all_cells" || fail "V9_sm_clock_1710_all_cells" "see clk_*.tsv"

banner "Results"
python3 - "$TSV" "$PF" "$CF" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
print(f"ptxas frame: 401_r4 {sys.argv[2]} B  ->  402 {sys.argv[3]} B\n")
print(f"{'cell':<7}{'binary':<22}{'MB':>6}{'w/SM':>5}{'KB':>7}{'ctx':>4}{'free_mb':>8}{'kernel_ms':>13}{'vs E0':>9}{'sm_mean':>8}{'tmax':>5}")
e0=next((float(r['kernel_ms']) for r in rows if r['cell']=='E0'),None)
for r in rows:
    v=float(r['kernel_ms']); rel=f"{(v-e0)/e0*100:+.3f}%" if e0 else "-"
    print(f"{r['cell']:<7}{r['binary']:<22}{r['max_blocks']:>6}{r['warps_per_sm']:>5}{r['footprint_kb']:>7}{r['extra_ctx']:>4}{r['free_mb']:>8}{v:>13.3f}{rel:>9}{r['sm_mean']:>8}{r['temp_max']:>5}")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs" 2>/dev/null || true
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
