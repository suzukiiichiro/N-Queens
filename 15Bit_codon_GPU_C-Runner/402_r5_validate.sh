#!/usr/bin/env bash
# 402_r5_validate.sh
#
# rev402-r5 -- adopt the 402-r4 winner on the production path:
#              env_prefix = "NQ_EXTRA_CTX=1 NQ_HELPER_CTX=1 NQ_HELPER_MB=128 ".
#              .cu is a rename of 402-r4. Confirm with bare -g.
#
# CELLS
#   G21a, G21b   ./402_r5Py_kernel_maxd14_final -g 21 21   (twice)
#   G1921        ./402_r5Py_kernel_maxd14_final -g 19 21   (N=19, 20, 21, oracle each)
# PRE-REGISTERED (402_r5_README_append.md)
#   U1 HARD: dispatch.log env_prefix carries NQ_HELPER_CTX=1 NQ_HELPER_MB=128 (and
#      NQ_MAX_BLOCKS=960); CRunner log [gpu-helper] helpers=1 helper_mb=128, MAX_BLOCKS=960
#   U2 G21a, G21b within +-0.15% of 109,459 and within 0.05% of each other
#   U3 -g 19 21: N=19/20/21 all MATCH (hard); N=20 within +-1.0% of 16,482 (informational)
#   U4 mean SM clock within 2% of 1710 in every cell
#
# USAGE
#   STATIC_ONLY=1 bash 402_r5_validate.sh     # OK=14
#                 bash 402_r5_validate.sh     # ~9 min

set -u

REV="402_r5"
PY_SRC="${PY_SRC:-402_r5Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-402_r5Py_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-402_r4Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-402_r5_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-402_r5_kernel_maxd14}"
PREV_CU="${PREV_CU:-402_r4_kernel_maxd14.cu}"
CRLOG_DIR="${CRLOG_DIR:-402_r5_crunner_logs}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
MB="${MB:-960}"
PREFIX_EXPECT="${PREFIX_EXPECT:-NQ_EXTRA_CTX=1 NQ_HELPER_CTX=1 NQ_HELPER_MB=128 }"
ANCHOR_G21="${ANCHOR_G21:-109458.973}"   # 402-r4 G1m mean
ANCHOR_G20="${ANCHOR_G20:-16481.972}"    # 401-r3 Gd (no helper) -- informational
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
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi; exit 0; }

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
grep -qE '^REV_TAG:str="402_r5"' "$CODE" && pass "source_rev_tag_is_402_r5" || fail "source_rev_tag_is_402_r5" "wrong REV_TAG"
grep -qF "CRunnerEntry(14,\"./402_r5_kernel_maxd14\",\"${PREFIX_EXPECT}\"" "$CODE" && pass "source_table_adopts_helper_prefix" || fail "source_table_adopts_helper_prefix" "expected env_prefix \"${PREFIX_EXPECT}\""
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
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_402_r4Py (removed=3 added=3 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_402_r4Py" "removed=$PR_ added=$PA_, expected 3/3"
else info "py_diff_fingerprint_vs_402_r4Py" "skipped"; fi
cu_code_region() { awk 'f||/^#include/{f=1;print}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
if [[ -f "$PREV_CU" ]]; then
  cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
  CA="$(sha256sum < "/tmp/${REV}_prev_code.cu" | cut -d' ' -f1)"; CB="$(sha256sum < "/tmp/${REV}_cur_code.cu" | cut -d' ' -f1)"
  [[ "$CA" == "$CB" ]] && pass "cu_whole_code_region_identical_to_402_r4 (${CB:0:16}...)" || fail "cu_whole_code_region_identical_to_402_r4" "402-r5 must be a rename"
else info "cu_whole_code_region_identical_to_402_r4" "skipped"; fi
grep -q 'getenv("NQ_HELPER_CTX")' "/tmp/${REV}_cur_code.cu" && pass "cu_helper_present" || fail "cu_helper_present" "no NQ_HELPER_CTX in the binary"
grep -q "rev402-r5" "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev402-r5 note in header"

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
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/06_codon.log"
[[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN]" || { fail "codon_build" "see log"; summary_exit; }

# ---------------------------------------------------------------------
# 3. Cells
# ---------------------------------------------------------------------
printf 'cell\tN\tprefix_ok\thelpers\thelper_mb\tmax_blocks\tfree_mb\tkernel_ms\ttotal_sum\tmatch\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
CLKPID=""
gpu_gate() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_before_$1.txt"
  [[ -z "$apps" ]] && return 0; fail "gpu_empty_before[$1]" "compute process(es) present: $(echo "$apps" | tr '\n' ' ')"; return 1; }
clk_start() { local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"
  ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPID=$!; }
clk_stop() { [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.0f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_$1.tsv"); }
declare -A VAL SMM
STRAG=0
after_cell() { sleep 2; local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_after_$1.txt"; [[ -z "$apps" ]] || { STRAG=1; info "straggler_after[$1]" "$apps"; }; }
read_n() {  # cell N -> records one row, returns 0 on oracle+prefix+helper ok
  local cell="$1" n="$2" gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${n}.log" orc; orc="$(oracle_of "$n")"
  cp "$gcr" "$LOGDIR/2_${cell}_N${n}_crunner.log" 2>/dev/null || { fail "crunner_path_taken[$cell N=$n]" "no $gcr"; return 1; }
  local kms tot m hl hm cmb fr pok=0
  kms="$(grep -o 'kernel_ms=[0-9.]*' "$gcr" | head -1 | cut -d= -f2)"; tot="$(grep -o 'total_sum=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"
  m=0; grep -q '\[gpu-run-correctness\] MATCH' "$gcr" && m=1
  hl="$(grep -o 'helpers=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"; hm="$(grep -o 'helper_mb=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"
  cmb="$(grep -o 'MAX_BLOCKS=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"; fr="$(grep -o 'free_mb=[0-9]*' "$gcr" | head -1 | cut -d= -f2)"
  grep -q "\[crunner-config\] N=$n .*env_prefix=NQ_MAX_BLOCKS=$MB ${PREFIX_EXPECT% }" "$CRLOG_DIR/dispatch.log" 2>/dev/null && pok=1
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$n" "$pok" "${hl:-?}" "${hm:-?}" "${cmb:-?}" "${fr:-?}" "${kms:-?}" "${tot:-?}" "$m" "$SMMEAN" "$SMMIN" "$TMAX" "$START" >> "$TSV"
  [[ "$m" == "1" && "${tot:-}" == "$orc" ]] || { fail "oracle[$cell N=$n]" "total_sum='${tot:-<none>}' expected $orc"; return 1; }
  [[ "$pok" == "1" && "${hl:-}" == "1" && "${hm:-}" == "128" && "${cmb:-}" == "$MB" ]] || { fail "U1_prefix_on_path[$cell N=$n]" "prefix_in_dispatch_log=$pok helpers=${hl:-?} helper_mb=${hm:-?} MAX_BLOCKS=${cmb:-?}"; return 1; }
  VAL["${cell}_N${n}"]="$kms"; info "$cell N=$n" "kernel_ms=$kms helpers=$hl mb=$hm MAX_BLOCKS=$cmb free_mb=$fr sm_mean=$SMMEAN"
  return 0
}
run_g() {  # cell nlo nhi
  local cell="$1" nlo="$2" nhi="$3"; local n
  for n in $(seq "$nlo" "$nhi"); do rm -f "$CRLOG_DIR/crunner_${CU_BIN}_N${n}.log"; done
  gpu_gate "$cell" || return 1
  START="$(date -Is)"; clk_start "$cell"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX -u NQ_BLOCK -u NQ_CARVEOUT -u NQ_HELPER_CTX -u NQ_HELPER_MB "./$PY_BIN" -g "$nlo" "$nhi" > "$LOGDIR/2_${cell}_console.log" 2>&1
  clk_stop "$cell"; after_cell "$cell"; SMM[$cell]="$SMMEAN"
  cp "$CRLOG_DIR/dispatch.log" "$LOGDIR/2_${cell}_dispatch.log" 2>/dev/null || true
  local ok=0; for n in $(seq "$nlo" "$nhi"); do read_n "$cell" "$n" || ok=1; done
  return $ok
}
banner "G21a"; run_g G21a 21 21 || summary_exit; sleep "$COOLDOWN"
banner "G21b"; run_g G21b 21 21 || summary_exit; sleep "$COOLDOWN"
banner "G1921 (-g 19 21)"; run_g G1921 19 21 || summary_exit

# ---------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------
banner "Evaluation"
pass "U1_prefix_and_helper_on_production_path (checked per run above)"
a="${VAL[G21a_N21]}"; b="${VAL[G21b_N21]}"
da="$(abspct "$a" "$ANCHOR_G21")"; db="$(abspct "$b" "$ANCHOR_G21")"; dab="$(abspct "$a" "$b")"
mean="$(awk -v x="$a" -v y="$b" 'BEGIN{printf "%.3f",(x+y)/2}')"
if le "$da" 0.15 && le "$db" 0.15 && le "$dab" 0.05; then pass "U2_production_confirmed (G21a=$a G21b=$b mean=$mean ms; ${da}%/${db}% from $ANCHOR_G21; |a-b|=${dab}%)"
else fail "U2_production_not_confirmed" "G21a=$a (${da}%), G21b=$b (${db}%), |a-b|=${dab}%"; fi
g20="${VAL[G1921_N20]:-}"; g19="${VAL[G1921_N19]:-}"; g21c="${VAL[G1921_N21]:-}"
if [[ -n "$g20" ]]; then d20="$(pct "$g20" "$ANCHOR_G20")"; info "U3_N20_with_helper" "$g20 ms (${d20}% vs 401-r3 Gd without helper)"; absdiff_le "$d20" 0 1.0 && pass "U3_N20_within_1pct_of_no_helper" || info "U3_N20" "outside +-1.0% -- informational; the helper state at N=20 is now measured"; fi
[[ -n "$g19" && -n "$g21c" ]] && info "U3_N19_N21" "N=19 $g19 ms, N=21 $g21c ms ($(pct "$g21c" "$mean")% vs G21 mean)"
pass "U3_oracles_N19_N20_N21 (all MATCH, checked per run above)"
[[ "$STRAG" == "0" ]] && pass "no_straggler_processes" || fail "straggler_processes" "see apps_after_*.txt"
CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "U4_sm_clock_1710_all_cells" || fail "U4_sm_clock_1710_all_cells" "see clk_*.tsv"

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
print(f"{'cell':<7}{'N':>3}{'prefix':>7}{'helpers':>8}{'MB':>5}{'MAX_BLOCKS':>11}{'free_mb':>8}{'kernel_ms':>13}{'match':>6}{'sm':>6}{'mm:ss.s':>9}")
for r in rows:
    v=float(r['kernel_ms'])
    print(f"{r['cell']:<7}{r['N']:>3}{r['prefix_ok']:>7}{r['helpers']:>8}{r['helper_mb']:>5}{r['max_blocks']:>11}{r['free_mb']:>8}{v:>13.3f}{r['match']:>6}{r['sm_mean']:>6}{int(v//60000):>5}:{(v%60000)/1000:04.1f}")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs" 2>/dev/null || true
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
