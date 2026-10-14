#!/usr/bin/env bash
# 410_diag.sh -- diagnostic cells for the 410_r2 search (N=21 only, ~8 min GPU)
#
# WHAT IS MEASURED (base 408; variants generated from 408_kernel_maxd14.cu by the 410_r2 search's
# generator, SASS sha checked against the search):
#   D0  408                                        <403> e940cb96 / 445 / 108 / MOV 8   anchor (408 A0 97,913.1 @1710)
#   D1  X2  403 pop: cur_x from the frame words,   5bad3239 / 446 / 109 / MOV 8   CANDIDATE
#           merges pinned as lop3 (3 stages after the LDLs instead of 5), registers 36
#   D2  X   same, C spelling (ptxas's own LOP3s)   3ebaa943 / 447 / 110 / MOV 8   candidate (+1 vs X2)
#
# WHY (410_r2 search 14:15, GPU zero; all 160 B / 0 spill / CPU equal to 408 x3):
#   pops are 26.9% of steps and the next step's nf = ~((cur_x | bit) | ...) waits for the pop's cur_x,
#   which 408 forms 5 dependent stages after the loads (a>>43 -> ld LOP3 -> <<1 -> | rd>>1 -> | col).
#   (ld << 1) == ((a >> 42) & ~3) | ((b << 1) & 2), so cur_x can be formed in 3 stages directly from
#   the frame words; cur_ld keeps its own path (next used only at the descent). No loop-carried
#   variable is added (410's top-of-stack cache added 11 MOVs and is closed). The gain, if any, is
#   latency: no instruction count shows it (K: +9.2% with +19; Pu: +63% with -2), so it is measured.
#   With BLOCK 32 x MB 960 (~12 warps / SM, 3 per scheduler) dependent-chain length is exposed.
#
# PRE-REGISTERED READING (no direction asserted; same as 405..408_diag), on CLOCK-NORMALISED time
#   norm_ms = kernel_ms * sm_mean / 1710 (10/6 finding: same-SASS scatter 0.1-0.4% = SM clock
#   1702..1710 MHz under 70-74 C; A/B 13:58 at a flat 1710 reproduced normalised values to 0.03%):
#   Dk/D0 <= 0.998  candidate      0.998 < Dk/D0 < 1.002  no effect      Dk/D0 >= 1.002  loses
#   Expected shape (not a claim): D1 <= D2 (X2 is X minus one instruction on the same chain).
#   The next revision is ONE measured cell; sums are not assumed.
# HARD: all cells results.bin == D0, oracle 314666222712, layout=403, SASS sha == search, 1710 +-2%.
# Production stays 408 whatever comes out.
#
# USAGE
#   bash 410_diag.sh                 # D0..D2, ~8 min
#   CELLS="D0 D1" bash 410_diag.sh
# Prerequisites: 408_kernel_maxd14.cu (+ 408_kernel_maxd14, rebuilt if absent), 410_r2_compile_search.sh
#   (its generator is reused), N=21 input.

set -u
REV="410_diag"
BASE_CU="${BASE_CU:-408_kernel_maxd14.cu}"
BASE_BIN="${BASE_BIN:-408_kernel_maxd14}"
SEARCH_SH="${SEARCH_SH:-410_r2_compile_search.sh}"
CELLS="${CELLS:-D0 D1 D2}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"; CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"; ARCH="${ARCH:-sm_86}"
BLOCK="${BLOCK:-32}"; MB="${MB:-960}"; HMB="${HMB:-128}"; COOLDOWN="${COOLDOWN:-15}"; CLK_INTERVAL="${CLK_INTERVAL:-5}"
ANCHOR_D0="${ANCHOR_D0:-97913.1}"      # 408 A/B A0 (N=21 layout 403, direct, SM flat 1710)
ORACLE21=314666222712
TS="$(date +%Y%m%d_%H%M%S)"; LOGDIR="${LOGDIR:-${REV}_${TS}}"; TSV="$LOGDIR/${REV}_results.tsv"
mkdir -p "$LOGDIR/sass" "$LOGDIR/bins" "$LOGDIR/variants"

# cell -> ops ; expected <403> sha / real / loop (from the search)
declare -A OPS=( [D0]="0" [D1]="X2" [D2]="X" )
declare -A SHA=( [D0]="e940cb96e5e65fc4" [D1]="5bad3239a689f599" [D2]="3ebaa9432f1bc801" )
declare -A REAL=( [D0]=445 [D1]=446 [D2]=447 )
declare -A LOOP=( [D0]=108 [D1]=109 [D2]=110 )
declare -A LABEL=( [D0]="408" [D1]="X2 pop cur_x in 3 stages (lop3 pinned)" [D2]="X pop cur_x in 3 stages (C spelling)" )

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f",(a-b)/b*100}'; }
ratio() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.4f",a/b}'; }
le() { awk -v x="$1" -v y="$2" 'BEGIN{exit !(x<=y)}'; }
ge() { awk -v x="$1" -v y="$2" 'BEGIN{exit !(x>=y)}'; }
absdiff_le() { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=t)}'; }
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi; exit 0; }

for f in "$BASE_CU" "$SEARCH_SH"; do [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"; done
[[ "$FAIL" -gt 0 ]] && summary_exit
[[ ! -x "$NVCC" ]] && NVCC="nvcc"; [[ ! -x "$CUOBJDUMP" ]] && CUOBJDUMP="cuobjdump"
# reuse the search's generator and SASS tool verbatim
eval "$(sed -n '/^gen_variant() {/,/^}/p' "$SEARCH_SH")"
eval "$(sed -n '/^sass_stat() {/,/^}/p' "$SEARCH_SH")"
declare -f gen_variant >/dev/null && declare -f sass_stat >/dev/null && pass "generator_and_sass_tool_loaded_from[$SEARCH_SH]" || { fail "generator_load" "could not extract gen_variant/sass_stat from $SEARCH_SH"; summary_exit; }

find_input() { local n="$1" c; for lg in $(ls -t 40*_crunner_logs/crunner_*_N${n}.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { echo "$c"; return; }; done; ls -t constellations_N${n}_*.sched394f.bin 2>/dev/null | head -1; }
IN21="$(find_input 21)"; [[ -f "${IN21:-/nonexistent}" ]] && pass "input21_located ($IN21)" || { fail "input21_located" "no N=21 sched input"; summary_exit; }
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1

# ---------------------------------------------------------------------
# build each cell's binary, check SASS against the search
# ---------------------------------------------------------------------
banner "Building cells: $CELLS"
declare -A BIN
for C in $CELLS; do
  ops="${OPS[$C]:-}"; [[ -n "$ops" ]] || { fail "cell[$C]" "unknown cell"; continue; }
  if [[ "$C" == "D0" ]]; then
    vcu="$BASE_CU"; bin="$BASE_BIN"
    if [[ ! -x "$bin" ]]; then "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$bin" "$vcu" -lcuda > "$LOGDIR/05_nvcc_${C}.log" 2>&1; fi
  else
    vcu="$LOGDIR/variants/${REV}_${C}_${ops}.cu"; bin="$LOGDIR/${REV}_${C}_kernel_maxd14"
    gen_variant "$BASE_CU" "$vcu" "$ops" 2>"$LOGDIR/variants/gen_${C}.log" || { fail "gen[$C]" "$(cat "$LOGDIR/variants/gen_${C}.log")"; continue; }
    "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$bin" "$vcu" -lcuda > "$LOGDIR/05_nvcc_${C}.log" 2>&1
  fi
  [[ -x "$bin" ]] || { fail "nvcc[$C]" "$(grep -i error "$LOGDIR/05_nvcc_${C}.log" 2>/dev/null | head -3)"; continue; }
  if [[ -f "$LOGDIR/05_nvcc_${C}.log" ]]; then
    FR="$(grep -A3 'kernel_dfs_iter_gpu_maxd14ILi403E' "$LOGDIR/05_nvcc_${C}.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"
    RG="$(grep -A3 'kernel_dfs_iter_gpu_maxd14ILi403E' "$LOGDIR/05_nvcc_${C}.log" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"
    SP="$(grep -A3 'kernel_dfs_iter_gpu_maxd14ILi403E' "$LOGDIR/05_nvcc_${C}.log" | grep -o '[0-9]* bytes spill stores' | head -1 | cut -d' ' -f1)"
    [[ "$FR" == "160" && "$SP" == "0" && "${RG:-99}" -le 44 ]] && pass "ptxas[$C]_160B_spill0_regs${RG}" || { fail "ptxas[$C]" "frame=${FR}B regs=${RG} spill=${SP}"; continue; }
  fi
  "$CUOBJDUMP" -sass "./$bin" > "$LOGDIR/sass/${C}.txt" 2>&1
  read -r S R L T _ _ _ _ _ _ _ <<< "$(sass_stat "$LOGDIR/sass/${C}.txt" kernel_dfs_iter_gpu_maxd14ILi403E "$LOGDIR/sass/${C}_loop403.txt")"
  [[ "$S" == "${SHA[$C]}" && "$R" == "${REAL[$C]}" && "$L" == "${LOOP[$C]}" ]] && pass "sass_403[$C]_matches_search (sha $S, real $R, loop $L)" || { fail "sass_403[$C]_differs_from_search" "sha $S real $R loop $L != ${SHA[$C]} / ${REAL[$C]} / ${LOOP[$C]}"; continue; }
  BIN[$C]="$bin"
done
[[ "$FAIL" -gt 0 ]] && summary_exit

# ---------------------------------------------------------------------
# cells (N=21 direct, layout 403, BLOCK 32 / MB 960 / helper 1 + 128 MB)
# ---------------------------------------------------------------------
printf 'cell\tops\tbinary\tN\tlayout\thelpers\tfree_mb\tkernel_ms\tnorm_ms\ttotal_sum\tmatch\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
CLKPID=""
gpu_gate() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_before_$1.txt"; [[ -z "$apps" ]] && return 0; fail "gpu_empty_before[$1]" "compute process(es) present: $(echo "$apps" | tr '\n' ' ')"; return 1; }
gpu_after() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_after_$1.txt"; [[ -z "$apps" ]] || fail "gpu_residue_after[$1]" "$(echo "$apps" | tr '\n' ' ')"; }
clk_start() { local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"; ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPID=$!; }
clk_stop() { [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 && $6+0>50 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.1f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_$1.tsv"); }
declare -A VAL NORM SMM
run_cell() {  # cell
  local cell="$1" bin="${BIN[$1]}"; gpu_gate "$cell" || return 1
  local lg="$LOGDIR/3_${cell}.log" start; start="$(date -Is)"; clk_start "$cell"
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX NQ_LAYOUT=auto NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$MB" NQ_HELPER_CTX=1 NQ_HELPER_MB="$HMB" "./$bin" 21 "$IN21" "$LOGDIR/bins/${cell}_results.bin" "$ORACLE21" > "$lg" 2>&1 || true
  clk_stop "$cell"; gpu_after "$cell"
  local KMS TOT MATCH FREE HL LAYV
  KMS="$(grep -o 'kernel_ms=[0-9.]*' "$lg" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$lg" && MATCH=1
  FREE="$(grep -o 'free_mb=[0-9]*' "$lg" | head -1 | cut -d= -f2)"; HL="$(grep -o 'helpers=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  LAYV="$(grep -o '\[gpu-layout\] N=[0-9]* requested=[a-z0-9]* layout=[0-9]*' "$lg" | head -1 | grep -o 'layout=[0-9]*' | cut -d= -f2)"
  local NORMV; NORMV="$(awk -v k="${KMS:-0}" -v s="${SMMEAN/\?/0}" 'BEGIN{ if(s>0) printf "%.1f", k*s/1710; else print "?" }')"
  printf '%s\t%s\t%s\t21\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "${OPS[$cell]}" "$bin" "${LAYV:-?}" "${HL:-?}" "${FREE:-?}" "${KMS:-?}" "$NORMV" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start" >> "$TSV"
  [[ "${LAYV:-}" == "403" ]] || { fail "layout[$cell]" "layout=${LAYV:-?}, expected 403"; return 1; }
  [[ "$MATCH" == "1" && "${TOT:-}" == "$ORACLE21" ]] || { fail "oracle[$cell]" "total_sum='${TOT:-<none>}' match=$MATCH"; return 1; }
  VAL[$cell]="$KMS"; NORM[$cell]="$NORMV"; SMM[$cell]="$SMMEAN"
  info "$cell" "${LABEL[$cell]} kernel_ms=$KMS norm_ms=$NORMV (sm_mean $SMMEAN, min $SMMIN, temp max $TMAX C) free_mb=$FREE"
}
same_bin() { if [[ -s "$LOGDIR/bins/D0_results.bin" ]] && cmp -s "$LOGDIR/bins/D0_results.bin" "$LOGDIR/bins/${1}_results.bin"; then pass "${1}_per_thread_results_identical_to_D0"; else fail "${1}_per_thread_results_DIFFER" "results.bin $1 != D0"; fi; }

first=1
for C in $CELLS; do
  [[ -n "${BIN[$C]:-}" ]] || continue
  [[ "$first" == "1" ]] || sleep "$COOLDOWN"; first=0
  banner "$C: ${LABEL[$C]}"
  run_cell "$C" || summary_exit
  [[ "$C" == "D0" ]] || same_bin "$C"
done
[[ -n "${VAL[D0]:-}" ]] && info "D0_vs_408_A0" "raw $(pct "${VAL[D0]}" "$ANCHOR_D0")% / normalised $(pct "${NORM[D0]}" "$ANCHOR_D0")% vs $ANCHOR_D0 (same SASS, measured at a flat 1710 MHz)"

banner "Reading on norm_ms (registered: <= 0.998 candidate / 0.998..1.002 no effect / >= 1.002 loses)"
for C in $CELLS; do
  [[ "$C" == "D0" || -z "${VAL[$C]:-}" || -z "${VAL[D0]:-}" ]] && continue
  r="$(ratio "${NORM[$C]}" "${NORM[D0]}")"; p="$(pct "${NORM[$C]}" "${NORM[D0]}")"; praw="$(pct "${VAL[$C]}" "${VAL[D0]}")"
  if le "$r" 0.998; then v="CANDIDATE"; elif ge "$r" 1.002; then v="LOSES"; else v="no effect"; fi
  info "$C/D0" "normalised $r (${p}%; raw ${praw}%)  -> $v   [${LABEL[$C]}]"
done
CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "sm_clock_1710_all_cells" || fail "sm_clock_1710_all_cells" "see clk_*.tsv"

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
ref={r['cell']:float(r['kernel_ms']) for r in rows}; d0=ref.get('D0')
n0=float(next(r['norm_ms'] for r in rows if r['cell']=='D0')) if d0 else None
print(f"{'cell':<5}{'ops':<6}{'lay':>4}{'help':>5}{'free_mb':>8}{'kernel_ms':>14}{'vs D0':>10}{'norm_ms':>12}{'vs D0n':>10}{'match':>6}{'sm':>8}{'m:ss':>8}")
for r in rows:
    v=float(r['kernel_ms']); s=v/1000; nv=float(r['norm_ms']) if r['norm_ms'] not in ('?','') else None
    vs=f"{(v/d0-1)*100:>+9.3f}%" if d0 else f"{'':>10}"
    vn=f"{(nv/n0-1)*100:>+9.3f}%" if (n0 and nv) else f"{'':>10}"
    print(f"{r['cell']:<5}{r['ops']:<6}{r['layout']:>4}{r['helpers']:>5}{r['free_mb']:>8}{v:>14.3f}{vs}{(nv if nv else 0):>12.1f}{vn}{r['match']:>6}{r['sm_mean']:>8}{int(s//60):>5}:{s%60:04.1f}")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd14" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
