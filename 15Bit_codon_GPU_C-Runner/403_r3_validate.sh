#!/usr/bin/env bash
# 403_r3_validate.sh
#
# rev403-r3 -- NO code change. N=22 sweep on the adopted 403_r2d code:
#   MAX_BLOCKS 800 / 960 / 1040 x helper {1 + 128 MB, none}, direct runs,
#   plus a replicate of the anchor cell for the N=22 noise floor.
#
#   Static: code region byte-identical to 403_r2d; ptxas 160 B / 0 spill /
#           regs <= 44 both instantiations; SASS of <402> == 402_r5 and
#           <403> == 403 (the binary IS the production code)      HARD
#   Cells (N=22, BLOCK=32, sched order, helper state fixed within a ladder):
#     H960 H800 H1040   helper 1 + 128 MB      (H960 = anchor 987,079)
#     N960 N800 N1040   no helper (NQ_HELPER_CTX=0)   [SKIP_NOHELPER=1 skips]
#     H960r             replicate of H960, last
#   Q1 STATED: helper ladder minimum at 960, both neighbours >= +2%
#              (refuted if either < +1%)
#   Q2 STATED: N960 - H960 in [+1%, +3%]  (refuted if |diff| < 0.5%)
#   Q3 info:   |H960r - H960|  (expected <= 0.1%)
#   Q4 HARD:   every cell oracle MATCH; Q5 1710 MHz; GPU empty before each
#   Nothing is adopted here.
#
# USAGE
#   STATIC_ONLY=1   bash 403_r3_validate.sh        # OK=13
#                   bash 403_r3_validate.sh        # 7 runs, ~2 h
#   SKIP_NOHELPER=1 bash 403_r3_validate.sh        # 4 runs, ~70 min

set -u

REV="403_r3"
PY_SRC="${PY_SRC:-403_r3Py_kernel_maxd14_final.py}"
PREV_PY="${PREV_PY:-403_r2dPy_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-403_r3_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-403_r3_kernel_maxd14}"
PREV_CU="${PREV_CU:-403_r2d_kernel_maxd14.cu}"
P402_BIN="${P402_BIN:-402_r5_kernel_maxd14}"
P403_BIN="${P403_BIN:-403_kernel_maxd14}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
BLOCK="${BLOCK:-32}"
HMB="${HMB:-128}"
KERNEL_SHA_403_R2D="${KERNEL_SHA_403_R2D:-14cba3d91d8e6288}"
ANCHOR_H960="${ANCHOR_H960:-987079.25}"     # 403-r2d G22
SKIP_NOHELPER="${SKIP_NOHELPER:-0}"
COOLDOWN="${COOLDOWN:-15}"
CLK_INTERVAL="${CLK_INTERVAL:-10}"
STATIC_ONLY="${STATIC_ONLY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"
TSV="$LOGDIR/${REV}_results.tsv"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f",(a-b)/b*100}'; }
abspct() { awk -v a="$1" -v b="$2" 'BEGIN{x=(a-b)/b*100; printf "%.3f",(x<0?-x:x)}'; }
le() { awk -v x="$1" -v y="$2" 'BEGIN{exit !(x<=y)}'; }
ge() { awk -v x="$1" -v y="$2" 'BEGIN{exit !(x>=y)}'; }
absdiff_le() { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=t)}'; }
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS (registered refutations are listed here too -- read the label):"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi; exit 0; }
ORACLE22=2691008701644

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
py_code_region "$PY_SRC" > "/tmp/${REV}_code_only.py"; CODE="/tmp/${REV}_code_only.py"
NOTE_LINES=$(awk '/^# =+$/{f=1} f&&/^#/{n++} END{print n+0}' "$PY_SRC")
{ grep -q "^# 403-r3 " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 403-r3 ...' note block"
grep -qE '^REV_TAG:str="403_r3"' "$CODE" && pass "source_rev_tag_is_403_r3" || fail "source_rev_tag_is_403_r3" "wrong REV_TAG"
grep -qF 'CRunnerEntry(14,"./403_r3_kernel_maxd14","NQ_EXTRA_CTX=1 NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_points_at_403_r3_keeps_helper_prefix" || fail "source_table_points_at_403_r3_keeps_helper_prefix" "table entry wrong"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=960\b" "$CODE" && pass "source_default_max_blocks_is_960" || fail "source_default_max_blocks_is_960" "default not 960"
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_403_r2dPy (removed=3 added=3 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_403_r2dPy" "removed=$PR_ added=$PA_, expected 3/3"
else info "py_diff_fingerprint_vs_403_r2dPy" "skipped"; fi
cu_code_region() { python3 - "$1" <<'PYEOF'
import sys
def strip(src):
    out=[]; i=0; n=len(src); in_str=None
    while i<n:
        c=src[i]
        if in_str:
            out.append(c)
            if c=='\\' and i+1<n: out.append(src[i+1]); i+=2; continue
            if c==in_str: in_str=None
            i+=1; continue
        if c in ('"',"'"): in_str=c; out.append(c); i+=1; continue
        if src.startswith('/*',i):
            j=src.find('*/',i+2); j=n if j<0 else j+2
            out.append('\n'*src[i:j].count('\n')); i=j; continue
        if src.startswith('//',i):
            j=src.find('\n',i); j=n if j<0 else j
            i=j; continue
        out.append(c); i+=1
    return ''.join(out)
lines=strip(open(sys.argv[1],encoding='utf-8').read()).split('\n')
try: k=next(i for i,l in enumerate(lines) if l.startswith('#include'))
except StopIteration: k=0
sys.stdout.write('\n'.join(l.rstrip() for l in lines[k:] if l.strip())+'\n')
PYEOF
}
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"; cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA_403_R2D"* ]] && pass "cu_kernel_region_IDENTICAL_to_403_r2d (${KB:0:16})" || fail "cu_kernel_region_IDENTICAL_to_403_r2d" "sha ${KB:0:16} != $KERNEL_SHA_403_R2D -- r3 must not change code"
cmp -s "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" && pass "cu_code_region_byte_identical_to_403_r2d" || fail "cu_code_region_byte_identical_to_403_r2d" "code region differs"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"/tmp/${REV}_gcc.log" && pass "cu_cpu_harness_builds" || fail "cu_cpu_harness_builds" "$(head -3 /tmp/${REV}_gcc.log)"
grep -q "rev403-r3" "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev403-r3 note in header"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Build, ptxas, SASS identity to the adopted production code
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR/sass"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
[[ ! -x "$NVCC" ]] && NVCC="nvcc"; [[ ! -x "$CUOBJDUMP" ]] && CUOBJDUMP="cuobjdump"
banner "Building"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc_${REV}.log" 2>&1
[[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "$(grep -i error "$LOGDIR/05_nvcc_${REV}.log" | head -3)"; summary_exit; }
for L in 402 403; do
  FR="$(grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${L}E" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"
  SP="$(grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${L}E" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes spill stores, [0-9]* bytes spill loads' | head -1)"
  RG="$(grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${L}E" "$LOGDIR/05_nvcc_${REV}.log" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"
  [[ "$FR" == "160" && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "ptxas_L${L}_frame160_spill0_regs${RG}" || { fail "ptxas_L${L}" "frame=${FR}B regs=${RG} ${SP}"; summary_exit; }
done
sass_cmp() { python3 - "$@" <<'PYEOF'
import sys, re
label, fa, fragA, fb, fragB = sys.argv[1:6]
def funcs(path):
    out = {}; cur = None
    for line in open(path, encoding='utf-8', errors='replace'):
        m = re.match(r'\s*Function\s*:\s*(\S+)', line)
        if m: cur = m.group(1); out[cur] = []; continue
        if cur is None: continue
        m = re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s*(.*?);\s*(/\*.*\*/)?\s*$', line)
        if m: out[cur].append(m.group(2).strip())
    return out
A = funcs(fa); B = funcs(fb)
ka = [k for k in A if fragA in k]; kb = [k for k in B if fragB in k]
if len(ka) != 1 or len(kb) != 1: print(f"{label}: CANNOT_LOCATE A={ka} B={kb}"); sys.exit(2)
ia, ib = A[ka[0]], B[kb[0]]
if ia == ib: print(f"{label}: IDENTICAL ({len(ia)} instructions)"); sys.exit(0)
print(f"{label}: DIFFER {sum(1 for x, y in zip(ia, ib) if x != y)} of {len(ia)}/{len(ib)}"); sys.exit(1)
PYEOF
}
if [[ -x "$P402_BIN" && -x "$P403_BIN" ]]; then
  "$CUOBJDUMP" -sass "./$CU_BIN" > "$LOGDIR/sass/${CU_BIN}.txt" 2>&1; "$CUOBJDUMP" -sass "./$P402_BIN" > "$LOGDIR/sass/${P402_BIN}.txt" 2>&1; "$CUOBJDUMP" -sass "./$P403_BIN" > "$LOGDIR/sass/${P403_BIN}.txt" 2>&1
  sass_cmp S402 "$LOGDIR/sass/${P402_BIN}.txt" "kernel_dfs_iter_gpu_maxd14PKj" "$LOGDIR/sass/${CU_BIN}.txt" "kernel_dfs_iter_gpu_maxd14ILi402E" >/dev/null && pass "sass_402_identical_to_402_r5" || { fail "sass_402_differs_from_402_r5" "this is not the adopted code"; summary_exit; }
  sass_cmp S403 "$LOGDIR/sass/${P403_BIN}.txt" "kernel_dfs_iter_gpu_maxd14PKj" "$LOGDIR/sass/${CU_BIN}.txt" "kernel_dfs_iter_gpu_maxd14ILi403E" >/dev/null && pass "sass_403_identical_to_403" || { fail "sass_403_differs_from_403" "this is not the adopted code"; summary_exit; }
else info "sass_identity" "reference binaries $P402_BIN / $P403_BIN not present -- skipped"; fi

IN22=""; for lg in $(ls -t 40*_crunner_logs/crunner_*_N22.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { IN22="$c"; break; }; done
[[ -f "${IN22:-/nonexistent}" ]] || IN22="$(ls -t constellations_N22_*.sched394f.bin 2>/dev/null | head -1)"
[[ -f "${IN22:-/nonexistent}" ]] && pass "input22_located ($IN22, $(( $(stat -c %s "$IN22") / 28 )) records)" || { fail "input22_located" "no N=22 sched input (run -g 22 22 once)"; summary_exit; }

# ---------------------------------------------------------------------
# 3. Cells
# ---------------------------------------------------------------------
printf 'cell\tmax_blocks\thelpers\thelper_mb\tfree_mb\tk_per_thread_max\tkernel_ms\ttotal_sum\tmatch\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
CLKPID=""
gpu_gate() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_before_$1.txt"; [[ -z "$apps" ]] && return 0; fail "gpu_empty_before[$1]" "compute process(es) present: $(echo "$apps" | tr '\n' ' ')"; return 1; }
clk_start() { local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"; ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPID=$!; }
clk_stop() { [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.0f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_$1.tsv"); }
declare -A VAL SMM
run_cell() {  # cell mb helper(0|1)
  local cell="$1" mb="$2" h="$3"; gpu_gate "$cell" || return 1
  local lg="$LOGDIR/3_${cell}.log" start; start="$(date -Is)"; clk_start "$cell"
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX -u NQ_LAYOUT NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$mb" NQ_HELPER_CTX="$h" NQ_HELPER_MB="$HMB" "./$CU_BIN" 22 "$IN22" "/tmp/${REV}_${cell}.bin" "$ORACLE22" > "$lg" 2>&1 || true
  clk_stop "$cell"
  local KMS TOT MATCH FREE HL HM KPT
  KMS="$(grep -o 'kernel_ms=[0-9.]*' "$lg" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$lg" | head -1 | cut -d= -f2)"; MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$lg" && MATCH=1
  FREE="$(grep -o 'free_mb=[0-9]*' "$lg" | head -1 | cut -d= -f2)"; HL="$(grep -o 'helpers=[0-9]*' "$lg" | head -1 | cut -d= -f2)"; HM="$(grep -o 'helper_mb=[0-9]*' "$lg" | head -1 | cut -d= -f2)"; KPT="$(grep -o 'k_per_thread_max=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$mb" "${HL:-$h}" "${HM:-?}" "${FREE:-?}" "${KPT:-?}" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start" >> "$TSV"
  grep -q '\[gpu-layout\] N=22 requested=auto layout=403' "$lg" || { fail "layout[$cell]" "layout line missing"; return 1; }
  [[ "$MATCH" == "1" && "${TOT:-}" == "$ORACLE22" ]] || { fail "Q4_oracle[$cell]" "total_sum='${TOT:-<none>}'"; return 1; }
  VAL[$cell]="$KMS"; SMM[$cell]="$SMMEAN"
  info "$cell" "MB=$mb helpers=${HL:-$h}/${HM:-?} kernel_ms=$KMS free_mb=$FREE k=$KPT sm_mean=$SMMEAN"
}
banner "Helper ladder (1 + $HMB MB): H960 H800 H1040"
run_cell H960 960 1 || summary_exit
d="$(abspct "${VAL[H960]}" "$ANCHOR_H960")"; le "$d" 0.2 && pass "H960_anchor (${d}% from 403-r2d G22 $ANCHOR_H960)" || info "H960_anchor" "${d}% from $ANCHOR_H960 -- ladder still read relative to H960"
sleep "$COOLDOWN"; run_cell H800 800 1 || summary_exit
sleep "$COOLDOWN"; run_cell H1040 1040 1 || summary_exit
p800="$(pct "${VAL[H800]}" "${VAL[H960]}")"; p1040="$(pct "${VAL[H1040]}" "${VAL[H960]}")"
if ge "$p800" 2 && ge "$p1040" 2; then pass "Q1_960_sharp_minimum_at_N22 (800 ${p800}%, 1040 ${p1040}%)"
elif ge "$p800" 1 && ge "$p1040" 1; then fail "Q1_grey_zone" "800 ${p800}%, 1040 ${p1040}% (stated >= +2%, refuted < +1%)"
else fail "Q1_REFUTED_neighbour_within_1pct" "800 ${p800}%, 1040 ${p1040}% -- the N=22 minimum is not 960-sharp; 403-r3b replicates before any table change"; fi
if [[ "$SKIP_NOHELPER" != "1" ]]; then
  banner "No-helper ladder: N960 N800 N1040"
  sleep "$COOLDOWN"; run_cell N960 960 0 || summary_exit
  sleep "$COOLDOWN"; run_cell N800 800 0 || summary_exit
  sleep "$COOLDOWN"; run_cell N1040 1040 0 || summary_exit
  ph="$(pct "${VAL[N960]}" "${VAL[H960]}")"; ah="$(abspct "${VAL[N960]}" "${VAL[H960]}")"
  if ge "$ph" 1 && le "$ph" 3; then pass "Q2_helper_effect_present_at_N22 (N960 ${ph}% vs H960; N=21 was +2.26%)"
  elif le "$ah" 0.5; then fail "Q2_REFUTED_no_helper_effect_at_N22" "N960 ${ph}% vs H960 (|diff| < 0.5%): N=22 memory state is outside the N=21 notch"
  else fail "Q2_outside_band" "N960 ${ph}% vs H960 (stated [+1%, +3%])"; fi
  n800="$(pct "${VAL[N800]}" "${VAL[N960]}")"; n1040="$(pct "${VAL[N1040]}" "${VAL[N960]}")"
  info "no_helper_ladder" "800 ${n800}%, 1040 ${n1040}% vs N960 (same-state comparison only)"
fi
banner "Replicate: H960r"
sleep "$COOLDOWN"; run_cell H960r 960 1 || summary_exit
r="$(abspct "${VAL[H960r]}" "${VAL[H960]}")"; le "$r" 0.1 && pass "Q3_N22_replicate_noise (${r}%)" || info "Q3_N22_replicate_noise" "${r}% (> 0.1%: N=22 noise floor is wider than N=21's 0.04%)"
CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "Q5_sm_clock_1710_all_cells" || fail "Q5_sm_clock_1710_all_cells" "see clk_*.tsv"
best=""; for c in "${!VAL[@]}"; do [[ "$c" == "H960" || "$c" == "H960r" ]] && continue; g="$(pct "${VAL[$c]}" "${VAL[H960]}")"; le "$g" -1 && best="$best $c(${g}%)"; done
[[ -z "$best" ]] && info "adoption" "no cell beats H960 by > 1%; production stays 960 + helper" || info "adoption" "cells beating H960 by > 1%:$best -- 403-r3b must replicate before any table change"

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
ref=float(next(r['kernel_ms'] for r in rows if r['cell']=='H960'))
print(f"{'cell':<7}{'MB':>5}{'help':>7}{'free_mb':>8}{'k':>5}{'kernel_ms':>14}{'vs H960':>9}{'match':>6}{'sm':>6}{'h:mm:ss':>10}")
for r in rows:
    v=float(r['kernel_ms']); s=v/1000
    print(f"{r['cell']:<7}{r['max_blocks']:>5}{r['helpers']+'/'+r['helper_mb']:>7}{r['free_mb']:>8}{r['k_per_thread_max']:>5}{v:>14.3f}{(v/ref-1)*100:>+8.3f}%{r['match']:>6}{r['sm_mean']:>6}{int(s//3600):>4}:{int(s%3600//60):02d}:{s%60:04.1f}")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
