#!/usr/bin/env bash
# 403_r5b_validate.sh -- NO code change. Re-registration of 403-r5's adoption.
#
# 403-r5 (2026-09-30 12:31) passed every HARD gate (S2 <402> SASS == 402_r5,
# <403> 504 -> 496 instructions, Y2 CPU equality x3, Y3, X1 results.bin ==
# X0 byte for byte, 5/5 oracle, 1710 MHz) and measured
#   Z5/Z3 = 0.9959 (N=21 layout 403)   X1/X0 = 0.9956 (N=22 direct)
#   G22 = 983,142.25 (dispatcher; 403-r2d G22 987,079 -0.40%)
# P1/P2 were registered as "stated <= 0.985, refuted >= 0.997" and landed in
# the grey zone, so r5 was not adopted by that rule. The effect is 150x the
# N=22 floor (0.003%), has the same sign and size at N=21 and N=22, and the
# results are byte-identical, so the size estimate was wrong, not the effect.
# Suzuki decided (2026-09-30) to adopt after ONE replicate under this rule,
# fixed here before the run:
#
#   R0 HARD: the binaries are the ones r5 measured: 403_r5_kernel_maxd14.cu
#      kernel sha 7799b9e4; the existing 403_r5_kernel_maxd14 binary's SASS
#      <402> instruction list sha 76aabe60 (== 402_r5) and <403> sha fd6f41b2
#      (496). If a binary is missing it is rebuilt (nvcc is deterministic) and
#      must reproduce those shas.
#   R1 HARD: G22r ./403_r5Py -g 22 22: oracle 2,691,008,701,644, layout=403,
#      helpers=1, dispatch.log env_prefix unchanged, 1710 MHz.
#   R2 STATED (adoption): |G22r - 983,142.25| <= 0.05%.  Refuted: > 0.15%
#      (then the r5 G22 value is not stable and r5 is NOT adopted; between:
#      grey, one more run before deciding).
#   ADOPT (R0, R1, R2 held): 403_r5_kernel_maxd14 + 403_r5Py are production.
#      N=22 production value = mean(G22, G22r) = 983,1xx ms; N=21 unchanged
#      (same SASS as 402_r5, 109,4xx). Update the README production table.
#
# USAGE:  bash 403_r5b_validate.sh        # one N=22 run, ~17 min

set -u
REV="403_r5b"
CU_SRC="${CU_SRC:-403_r5_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-403_r5_kernel_maxd14}"
PY_SRC="${PY_SRC:-403_r5Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-${PY_SRC%.py}}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"; [[ -x "$NVCC" ]] || NVCC="nvcc"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"; [[ -x "$CUOBJDUMP" ]] || CUOBJDUMP="cuobjdump"
CODON="${CODON:-codon}"
ARCH="${ARCH:-sm_86}"
KERNEL_SHA="7799b9e456356748"
SASS402_SHA="76aabe60e055a22f"
SASS403_SHA="fd6f41b228f02d54"
G22_R5="983142.25"
ORACLE22=2691008701644
PREFIX22="NQ_MAX_BLOCKS=960 NQ_EXTRA_CTX=1 NQ_HELPER_CTX=1 NQ_HELPER_MB=128"
CLK_INTERVAL="${CLK_INTERVAL:-10}"
TS="$(date +%Y%m%d_%H%M%S)"; LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"; mkdir -p "$LOGDIR/sass"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
abspct() { awk -v a="$1" -v b="$2" 'BEGIN{x=(a-b)/b*100; printf "%.3f",(x<0?-x:x)}'; }
le() { awk -v x="$1" -v y="$2" 'BEGIN{exit !(x<=y)}'; }
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]:-}"; do [[ -n "$c" ]] && echo "  - $c"; done; tar czf "${LOGDIR}.tar.gz" "$LOGDIR" 2>/dev/null && echo "tarball: ${LOGDIR}.tar.gz"; [[ "$FAIL" -gt 0 ]] && exit 1; exit 0; }

# ---- R0: same binaries as r5 ----
for f in "$CU_SRC" "$PY_SRC"; do [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"; done
[[ "$FAIL" -gt 0 ]] && summary_exit
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
KB="$(cu_code_region "$CU_SRC" | awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' | sha256sum | cut -c1-16)"
[[ "$KB" == "$KERNEL_SHA" ]] && pass "R0_cu_kernel_sha_is_r5 ($KB)" || { fail "R0_cu_kernel_sha" "$KB != $KERNEL_SHA"; summary_exit; }
if [[ ! -x "$CU_BIN" ]]; then info "R0" "$CU_BIN missing -- rebuilding from $CU_SRC"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc.log" 2>&1; fi
[[ -x "$CU_BIN" ]] || { fail "R0_binary" "no $CU_BIN"; summary_exit; }
"$CUOBJDUMP" -sass "./$CU_BIN" > "$LOGDIR/sass/${CU_BIN}.txt" 2>&1
read -r S402 S403 N403 < <(python3 - "$LOGDIR/sass/${CU_BIN}.txt" <<'PYEOF'
import sys,re,hashlib
out={};cur=None
for line in open(sys.argv[1],errors='replace'):
    m=re.match(r'\s*Function\s*:\s*(\S+)',line)
    if m: cur=m.group(1); out[cur]=[]; continue
    if cur is None: continue
    m=re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s*(.*?);\s*(/\*.*\*/)?\s*$',line)
    if m: out[cur].append(m.group(2).strip())
h=lambda k:hashlib.sha256('\n'.join(out[k]).encode()).hexdigest()[:16]
k2=[k for k in out if 'ILi402E' in k][0]; k3=[k for k in out if 'ILi403E' in k][0]
print(h(k2),h(k3),len(out[k3]))
PYEOF
)
[[ "$S402" == "$SASS402_SHA" ]] && pass "R0_sass_402_is_402_r5 ($S402)" || { fail "R0_sass_402" "$S402 != $SASS402_SHA -- not the r5 binary"; summary_exit; }
[[ "$S403" == "$SASS403_SHA" ]] && pass "R0_sass_403_is_r5 ($S403, $N403 instructions)" || { fail "R0_sass_403" "$S403 != $SASS403_SHA -- not the r5 binary"; summary_exit; }
if [[ ! -x "$PY_BIN" ]]; then "$CODON" build -release "$PY_SRC" > "$LOGDIR/06_codon.log" 2>&1; fi
[[ -x "$PY_BIN" ]] && pass "py_binary[$PY_BIN]" || { fail "py_binary" "codon build failed"; summary_exit; }

# ---- R1/R2: G22r ----
apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_before_G22r.txt"
[[ -z "$apps" ]] && pass "gpu_empty_before[G22r]" || { fail "gpu_empty_before[G22r]" "$apps"; summary_exit; }
dlog="403_r5_crunner_logs/dispatch.log"; before=0; [[ -f "$dlog" ]] && before="$(wc -l < "$dlog")"
CLK="$LOGDIR/clk_G22r.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$CLK"
( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$CLK" 2>/dev/null & CLKPID=$!
START="$(date -Is)"
env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX -u NQ_LAYOUT -u NQ_HELPER_CTX -u NQ_HELPER_MB -u NQ_MAX_BLOCKS -u NQ_BLOCK "./$PY_BIN" -g 22 22 > "$LOGDIR/3_G22r.stdout" 2>&1; rc=$?
kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ' > "$LOGDIR/apps_after_G22r.txt"; [[ -s "$LOGDIR/apps_after_G22r.txt" ]] && fail "gpu_residue_after[G22r]" "$(cat "$LOGDIR/apps_after_G22r.txt")"
cp "403_r5_crunner_logs/crunner_${CU_BIN}_N22.log" "$LOGDIR/3_G22r.log" 2>/dev/null; tail -n +"$((before+1))" "$dlog" > "$LOGDIR/3_G22r_dispatch.log" 2>/dev/null
LG="$LOGDIR/3_G22r.log"
KMS="$(grep -o 'kernel_ms=[0-9.]*' "$LG" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$LG" | head -1 | cut -d= -f2)"
MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$LG" && MATCH=1
LAYV="$(grep -o '\[gpu-layout\] N=22 requested=auto layout=[0-9]*' "$LG" | head -1 | grep -o 'layout=[0-9]*' | cut -d= -f2)"
HL="$(grep -o 'helpers=[0-9]*' "$LG" | head -1 | cut -d= -f2)"; FREE="$(grep -o 'free_mb=[0-9]*' "$LG" | head -1 | cut -d= -f2)"
EP="$(grep '\[crunner-config\] N=22 ' "$LOGDIR/3_G22r_dispatch.log" | tail -1 | sed 's/.*env_prefix=//')"
read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.0f %d %d\n", s/n, min, t; else print "? ? ?"}' "$CLK")
printf 'cell\tN\tlayout\thelpers\tfree_mb\tkernel_ms\ttotal_sum\tmatch\tenv_prefix\tsm_mean\tsm_min\ttemp_max\tstart\nG22r\t22\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${LAYV:-?}" "${HL:-?}" "${FREE:-?}" "${KMS:-?}" "${TOT:-?}" "$MATCH" "${EP:-?}" "$SMMEAN" "$SMMIN" "$TMAX" "$START" > "$LOGDIR/${REV}_results.tsv"
[[ "$rc" == "0" && "$MATCH" == "1" && "${TOT:-}" == "$ORACLE22" ]] && pass "R1_oracle[G22r]" || { fail "R1_oracle[G22r]" "rc=$rc total_sum='${TOT:-<none>}' match=$MATCH"; summary_exit; }
[[ "${LAYV:-}" == "403" && "${HL:-}" == "1" ]] && pass "R1_layout403_helpers1" || fail "R1_layout_helpers" "layout=${LAYV:-?} helpers=${HL:-?}"
[[ "$EP" == "$PREFIX22" ]] && pass "R1_dispatch_env_prefix_unchanged" || fail "R1_dispatch_env_prefix" "[$EP]"
awk -v m="$SMMEAN" 'BEGIN{exit !(m>=1676 && m<=1744)}' && pass "R1_sm_clock_1710 ($SMMEAN)" || fail "R1_sm_clock" "$SMMEAN"
d="$(abspct "$KMS" "$G22_R5")"; MEAN="$(awk -v a="$KMS" -v b="$G22_R5" 'BEGIN{printf "%.1f",(a+b)/2}')"
if le "$d" 0.05; then pass "R2_G22r_replicates_r5_G22 (${d}%; G22r=$KMS, G22=$G22_R5)"
elif le "$d" 0.15; then fail "R2_grey_zone" "${d}% -- one more run before deciding"
else fail "R2_REFUTED_not_stable" "${d}% -- r5 G22 value not reproducible; not adopted"; fi
if [[ "$FAIL" == "0" ]]; then info "ADOPTION" "403_r5_kernel_maxd14 + 403_r5Py are PRODUCTION. N=22 = $MEAN ms (mean of G22 $G22_R5 and G22r $KMS); N=21 unchanged (same SASS as 402_r5)."
else info "ADOPTION" "not adopted yet -- see FAIL lines"; fi
info "G22r" "kernel_ms=$KMS free_mb=$FREE sm=$SMMEAN vs 403-r2d G22 987,079: $(awk -v a="$KMS" 'BEGIN{printf "%+.3f%%",(a-987079.25)/987079.25*100}')"
summary_exit
