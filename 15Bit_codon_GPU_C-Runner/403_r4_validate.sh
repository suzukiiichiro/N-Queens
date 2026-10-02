#!/usr/bin/env bash
# 403_r4_validate.sh
#
# rev403-r4 -- .py only: the production -g path drops the helper context for
#   N >= 22 (NQ_HELPER_CTX=1 -> 0 in the dispatched command line when N > 21;
#   A10G_FINAL_HELPER_MAX_N=21, crunner_env_prefix_for_n()). The .cu is a
#   pure rename of 403_r3 (= 403_r2d, the adopted production code).
#
#   Static: py notes present; REV_TAG 403_r4; table row = 403_r4 binary with
#           the UNCHANGED helper prefix; new constant + function + mode-37
#           call site present; py diff vs 403_r3Py removed=4 added=10;
#           .cu code region byte-identical to 403_r3 (kernel sha 14cba3d9);
#           CPU harness builds; header note.                        HARD
#   Build:  nvcc; ptxas 160 B / 0 spill / regs <= 44 both instantiations;
#           SASS <402> == 402_r5, <403> == 403 (V1)                 HARD
#           codon build of the .py
#   Cells (all through the dispatcher, i.e. the production path):
#     G21   ./403_r4Py  -g 21 21   layout 402, helper 1 + 128 MB  (unchanged)
#     G22a  ./403_r4Py  -g 22 22   layout 403, helper 0           (the change)
#     C22   ./403_r2dPy -g 22 22   control: helper 1, same session
#                                  [SKIP_CONTROL=1 skips; anchor -> 987,079]
#     G22b  ./403_r4Py  -g 22 22   replicate of G22a, last
#   V2 HARD:   G21 oracle, layout=402, helpers=1, dispatch.log env_prefix
#              identical to 403-r2d's, within +-0.15% of 109,437.
#              STATED: within +-0.05% of 109,460.
#   V3 HARD:   G22a/G22b oracle, layout=403, helpers=0, dispatch.log shows
#              NQ_HELPER_CTX=0 for N=22.
#   V4 STATED: mean(G22a,G22b) <= anchor - 0.08%   (expected ~ -0.125%)
#              refuted if |diff| <= 0.04%; between = grey (not adopted)
#   V5 info:   |G22a - G22b| <= 0.02%.  V6: 1710 MHz all cells.
#   ADOPT (V1-V3 hard, V4 held, V5 <= 0.05%): 403_r4Py is the production .py.
#
# USAGE
#   STATIC_ONLY=1  bash 403_r4_validate.sh        # OK=17
#                  bash 403_r4_validate.sh        # 4 runs, ~52 min
#   SKIP_CONTROL=1 bash 403_r4_validate.sh        # 3 runs, ~36 min

set -u

REV="403_r4"
PY_SRC="${PY_SRC:-403_r4Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-${PY_SRC%.py}}"
PREV_PY="${PREV_PY:-403_r3Py_kernel_maxd14_final.py}"
CTRL_PY_SRC="${CTRL_PY_SRC:-403_r2dPy_kernel_maxd14_final.py}"
CTRL_PY_BIN="${CTRL_PY_BIN:-${CTRL_PY_SRC%.py}}"
CTRL_CU_SRC="${CTRL_CU_SRC:-403_r2d_kernel_maxd14.cu}"
CTRL_CU_BIN="${CTRL_CU_BIN:-403_r2d_kernel_maxd14}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-403_r4_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-403_r4_kernel_maxd14}"
PREV_CU="${PREV_CU:-403_r3_kernel_maxd14.cu}"
[[ -f "$PREV_CU" ]] || PREV_CU="403_r2d_kernel_maxd14.cu"
P402_BIN="${P402_BIN:-402_r5_kernel_maxd14}"
P403_BIN="${P403_BIN:-403_kernel_maxd14}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
CODON="${CODON:-codon}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
KERNEL_SHA_403_R2D="${KERNEL_SHA_403_R2D:-14cba3d91d8e6288}"
ANCHOR_G21="${ANCHOR_G21:-109437}"          # production N=21 (402-r5 G21a/b mean)
ANCHOR_G21_R2D="${ANCHOR_G21_R2D:-109459.8}" # 403-r2d G21 (same SASS, same env as G21 here)
ANCHOR_G22="${ANCHOR_G22:-987079.25}"        # 403-r2d G22 (dispatcher, helper 1) -- fallback if C22 is skipped
SKIP_CONTROL="${SKIP_CONTROL:-0}"
COOLDOWN="${COOLDOWN:-15}"
CLK_INTERVAL="${CLK_INTERVAL:-10}"
STATIC_ONLY="${STATIC_ONLY:-0}"
ORACLE21=314666222712
ORACLE22=2691008701644
PREFIX21="NQ_MAX_BLOCKS=960 NQ_EXTRA_CTX=1 NQ_HELPER_CTX=1 NQ_HELPER_MB=128"
PREFIX22="NQ_MAX_BLOCKS=960 NQ_EXTRA_CTX=1 NQ_HELPER_CTX=0 NQ_HELPER_MB=128"

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
{ grep -q "^# 403-r4 " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 403-r4 ...' note block"
grep -qE '^REV_TAG:str="403_r4"' "$CODE" && pass "source_rev_tag_is_403_r4" || fail "source_rev_tag_is_403_r4" "wrong REV_TAG"
grep -qF 'CRunnerEntry(14,"./403_r4_kernel_maxd14","NQ_EXTRA_CTX=1 NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_points_at_403_r4_row_prefix_UNCHANGED" || fail "source_table_points_at_403_r4_row_prefix_UNCHANGED" "table entry wrong (the row string must stay the N=21 production string)"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=960\b" "$CODE" && pass "source_default_max_blocks_is_960" || fail "source_default_max_blocks_is_960" "default not 960"
grep -qE "^A10G_FINAL_HELPER_MAX_N:int=21\b" "$CODE" && pass "source_helper_max_n_is_21" || fail "source_helper_max_n_is_21" "constant missing or not 21"
{ grep -qE '^def crunner_env_prefix_for_n\(base_env_prefix:str,N:int\)->str:' "$CODE" && grep -qF 'return base_env_prefix.replace("NQ_HELPER_CTX=1 ","NQ_HELPER_CTX=0 ")' "$CODE"; } && pass "source_env_prefix_for_n_defined" || fail "source_env_prefix_for_n_defined" "function missing or wrong replacement"
grep -qF 'f"NQ_MAX_BLOCKS={gpu_max_blocks} {crunner_env_prefix_for_n(entry_base37.env_prefix,N)}"' "$CODE" && pass "source_mode37_site_uses_env_prefix_for_n" || fail "source_mode37_site_uses_env_prefix_for_n" "mode-37 dispatch site not routed through crunner_env_prefix_for_n"
NUSE="$(grep -cE '^def crunner_env_prefix_for_n\(|\{crunner_env_prefix_for_n\(' "$CODE")"
[[ "$NUSE" == "2" ]] && pass "source_env_prefix_for_n_used_exactly_once (def + mode-37 site)" || fail "source_env_prefix_for_n_used_exactly_once" "found $NUSE def/call lines, expected 2 (bench_mode=39 must stay untouched)"
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "4" && "$PA_" == "10" ]] && pass "py_diff_fingerprint_vs_403_r3Py (removed=4 added=10 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_403_r3Py" "removed=$PR_ added=$PA_, expected 4/10"
else info "py_diff_fingerprint_vs_403_r3Py" "skipped ($PREV_PY absent)"; fi
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
[[ "$KB" == "$KERNEL_SHA_403_R2D"* ]] && pass "cu_kernel_region_IDENTICAL_to_403_r2d (${KB:0:16})" || fail "cu_kernel_region_IDENTICAL_to_403_r2d" "sha ${KB:0:16} != $KERNEL_SHA_403_R2D -- r4 must not change code"
cmp -s "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" && pass "cu_code_region_byte_identical_to_${PREV_CU%.cu}" || fail "cu_code_region_byte_identical_to_${PREV_CU%.cu}" "code region differs"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"/tmp/${REV}_gcc.log" && pass "cu_cpu_harness_builds" || fail "cu_cpu_harness_builds" "$(head -3 /tmp/${REV}_gcc.log)"
grep -q "rev403-r4" "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev403-r4 note in header"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Build: nvcc, ptxas, SASS identity (V1); codon
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
  sass_cmp S402 "$LOGDIR/sass/${P402_BIN}.txt" "kernel_dfs_iter_gpu_maxd14PKj" "$LOGDIR/sass/${CU_BIN}.txt" "kernel_dfs_iter_gpu_maxd14ILi402E" >/dev/null && pass "V1_sass_402_identical_to_402_r5" || { fail "V1_sass_402_differs_from_402_r5" "this is not the adopted code"; summary_exit; }
  sass_cmp S403 "$LOGDIR/sass/${P403_BIN}.txt" "kernel_dfs_iter_gpu_maxd14PKj" "$LOGDIR/sass/${CU_BIN}.txt" "kernel_dfs_iter_gpu_maxd14ILi403E" >/dev/null && pass "V1_sass_403_identical_to_403" || { fail "V1_sass_403_differs_from_403" "this is not the adopted code"; summary_exit; }
else info "V1_sass_identity" "reference binaries $P402_BIN / $P403_BIN not present -- skipped"; fi

rm -f "$PY_BIN"; "$CODON" build -release "$PY_SRC" > "$LOGDIR/06_codon_${REV}.log" 2>&1
[[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN]" || { fail "codon_build" "$(tail -3 "$LOGDIR/06_codon_${REV}.log")"; summary_exit; }

CTRL_OK=0
if [[ "$SKIP_CONTROL" != "1" ]]; then
  if [[ ! -x "$CTRL_CU_BIN" && -f "$CTRL_CU_SRC" ]]; then "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$CTRL_CU_BIN" "$CTRL_CU_SRC" -lcuda > "$LOGDIR/07_nvcc_ctrl.log" 2>&1; fi
  if [[ ! -x "$CTRL_PY_BIN" && -f "$CTRL_PY_SRC" ]]; then "$CODON" build -release "$CTRL_PY_SRC" > "$LOGDIR/08_codon_ctrl.log" 2>&1; fi
  [[ -x "$CTRL_CU_BIN" && -x "$CTRL_PY_BIN" ]] && { CTRL_OK=1; pass "control_available[$CTRL_PY_BIN + $CTRL_CU_BIN]"; } || info "control" "$CTRL_PY_BIN / $CTRL_CU_BIN not available -- C22 skipped, anchor = $ANCHOR_G22"
fi

# ---------------------------------------------------------------------
# 3. Cells (dispatcher -g path)
# ---------------------------------------------------------------------
printf 'cell\tN\tpy\tlayout\thelpers\thelper_mb\tfree_mb\tkernel_ms\ttotal_sum\tmatch\tenv_prefix\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
CLKPID=""
gpu_gate() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_before_$1.txt"; [[ -z "$apps" ]] && return 0; fail "gpu_empty_before[$1]" "compute process(es) present: $(echo "$apps" | tr '\n' ' ')"; return 1; }
gpu_after() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_after_$1.txt"; [[ -z "$apps" ]] || fail "gpu_residue_after[$1]" "$(echo "$apps" | tr '\n' ' ')"; }
clk_start() { local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"; ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPID=$!; }
clk_stop() { [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.0f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_$1.tsv"); }
declare -A VAL SMM PFX HLP LAY
run_g() {  # cell N pybin cu_bin_stem
  local cell="$1" N="$2" pyb="$3" stem="$4"; gpu_gate "$cell" || return 1
  local rtag="${stem%_kernel_maxd14}" dlog="${stem%_kernel_maxd14}_crunner_logs/dispatch.log" clog="${stem%_kernel_maxd14}_crunner_logs/crunner_${stem}_N${N}.log"
  local before=0; [[ -f "$dlog" ]] && before="$(wc -l < "$dlog")"
  local lg="$LOGDIR/3_${cell}.stdout" start; start="$(date -Is)"; clk_start "$cell"
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX -u NQ_LAYOUT -u NQ_HELPER_CTX -u NQ_HELPER_MB -u NQ_MAX_BLOCKS -u NQ_BLOCK "./$pyb" -g "$N" "$N" > "$lg" 2>&1; local rc=$?
  clk_stop "$cell"; gpu_after "$cell"
  [[ -f "$clog" ]] && cp "$clog" "$LOGDIR/3_${cell}_crunner.log"; [[ -f "$dlog" ]] && tail -n +"$((before+1))" "$dlog" > "$LOGDIR/3_${cell}_dispatch.log"
  local KMS TOT MATCH FREE HL HM LAYV EP
  KMS="$(grep -o 'kernel_ms=[0-9.]*' "$LOGDIR/3_${cell}_crunner.log" 2>/dev/null | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$LOGDIR/3_${cell}_crunner.log" 2>/dev/null | head -1 | cut -d= -f2)"
  MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$LOGDIR/3_${cell}_crunner.log" 2>/dev/null && MATCH=1
  FREE="$(grep -o 'free_mb=[0-9]*' "$LOGDIR/3_${cell}_crunner.log" 2>/dev/null | head -1 | cut -d= -f2)"; HL="$(grep -o 'helpers=[0-9]*' "$LOGDIR/3_${cell}_crunner.log" 2>/dev/null | head -1 | cut -d= -f2)"; HM="$(grep -o 'helper_mb=[0-9]*' "$LOGDIR/3_${cell}_crunner.log" 2>/dev/null | head -1 | cut -d= -f2)"
  LAYV="$(grep -o '\[gpu-layout\] N=[0-9]* requested=[a-z0-9]* layout=[0-9]*' "$LOGDIR/3_${cell}_crunner.log" 2>/dev/null | head -1 | grep -o 'layout=[0-9]*' | cut -d= -f2)"
  EP="$(grep "\[crunner-config\] N=${N} " "$LOGDIR/3_${cell}_dispatch.log" 2>/dev/null | tail -1 | sed 's/.*env_prefix=//')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$N" "$pyb" "${LAYV:-?}" "${HL:-?}" "${HM:-?}" "${FREE:-?}" "${KMS:-?}" "${TOT:-?}" "$MATCH" "${EP:-?}" "$SMMEAN" "$SMMIN" "$TMAX" "$start" >> "$TSV"
  local want; want="$([[ "$N" -le 21 ]] && echo "$ORACLE21" || echo "$ORACLE22")"
  [[ "$rc" == "0" && "$MATCH" == "1" && "${TOT:-}" == "$want" ]] || { fail "oracle[$cell]" "rc=$rc total_sum='${TOT:-<none>}' match=$MATCH"; return 1; }
  VAL[$cell]="$KMS"; SMM[$cell]="$SMMEAN"; PFX[$cell]="$EP"; HLP[$cell]="${HL:-?}"; LAY[$cell]="${LAYV:-?}"
  info "$cell" "N=$N layout=${LAYV:-?} helpers=${HL:-?}/${HM:-?} kernel_ms=$KMS free_mb=$FREE sm_mean=$SMMEAN env_prefix=[$EP]"
}
check_env() {  # cell layout helpers prefix label
  local c="$1"; [[ "${LAY[$c]}" == "$2" ]] && pass "$5_layout${2}[$c]" || fail "$5_layout[$c]" "layout=${LAY[$c]}, expected $2"
  [[ "${HLP[$c]}" == "$3" ]] && pass "$5_helpers${3}[$c]" || fail "$5_helpers[$c]" "helpers=${HLP[$c]}, expected $3"
  [[ "${PFX[$c]}" == "$4" ]] && pass "$5_dispatch_env_prefix[$c]" || fail "$5_dispatch_env_prefix[$c]" "got [${PFX[$c]}], expected [$4]"
}

banner "G21: -g 21 21 (must be untouched)"
run_g G21 21 "$PY_BIN" "$CU_BIN" || summary_exit
check_env G21 402 1 "$PREFIX21" V2
d="$(abspct "${VAL[G21]}" "$ANCHOR_G21")"; le "$d" 0.15 && pass "V2_G21_within_0.15pct_of_production (${d}% from $ANCHOR_G21)" || { fail "V2_G21_outside_0.15pct" "${d}% from $ANCHOR_G21 -- the N=21 side moved; stop"; summary_exit; }
d2="$(abspct "${VAL[G21]}" "$ANCHOR_G21_R2D")"; le "$d2" 0.05 && pass "V2_stated_G21_within_0.05pct_of_r2d_G21 (${d2}%)" || info "V2_stated_G21" "${d2}% from r2d G21 $ANCHOR_G21_R2D (stated <= 0.05%; N=21 floor is 0.04%)"

banner "G22a: -g 22 22 (helper off -- the change)"
sleep "$COOLDOWN"; run_g G22a 22 "$PY_BIN" "$CU_BIN" || summary_exit
check_env G22a 403 0 "$PREFIX22" V3

if [[ "$CTRL_OK" == "1" ]]; then
  banner "C22: control, 403_r2dPy -g 22 22 (helper on, same session)"
  sleep "$COOLDOWN"; run_g C22 22 "$CTRL_PY_BIN" "$CTRL_CU_BIN" || summary_exit
  check_env C22 403 1 "$PREFIX21" C22
  ANCH="${VAL[C22]}"; ANCH_LABEL="C22 (same session)"
  info "C22_vs_r2d_G22" "$(pct "${VAL[C22]}" "$ANCHOR_G22")% (day-to-day drift of the helper-on dispatcher value)"
else ANCH="$ANCHOR_G22"; ANCH_LABEL="403-r2d G22 $ANCHOR_G22 (yesterday)"; fi

banner "G22b: replicate"
sleep "$COOLDOWN"; run_g G22b 22 "$PY_BIN" "$CU_BIN" || summary_exit
check_env G22b 403 0 "$PREFIX22" V3

MEAN22="$(awk -v a="${VAL[G22a]}" -v b="${VAL[G22b]}" 'BEGIN{printf "%.3f",(a+b)/2}')"
p4="$(pct "$MEAN22" "$ANCH")"; a4="$(abspct "$MEAN22" "$ANCH")"
V4=0
if le "$p4" -0.08; then pass "V4_helper_off_gain_at_N22 (mean G22 ${p4}% vs $ANCH_LABEL; 403-r3 predicted about -0.125%)"; V4=1
elif le "$a4" 0.04; then fail "V4_REFUTED_no_effect_on_dispatcher_path" "mean G22 ${p4}% vs $ANCH_LABEL (|diff| <= 0.04%)"
elif le "$p4" 0; then fail "V4_grey_zone" "mean G22 ${p4}% vs $ANCH_LABEL (stated <= -0.08%, refuted |diff| <= 0.04%) -- not adopted"
else fail "V4_G22_SLOWER" "mean G22 ${p4}% vs $ANCH_LABEL -- helper off is worse here; not adopted"; fi
r="$(abspct "${VAL[G22b]}" "${VAL[G22a]}")"; le "$r" 0.02 && pass "V5_N22_replicate (${r}%)" || info "V5_N22_replicate" "${r}% (> 0.02%; floor was 0.003%)"
CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "V6_sm_clock_1710_all_cells" || fail "V6_sm_clock_1710_all_cells" "see clk_*.tsv"
if [[ "$V4" == "1" ]] && le "$r" 0.05 && [[ "$FAIL" == "0" ]]; then
  info "ADOPTION" "403_r4Py is the production .py: N=21 ${VAL[G21]} (unchanged), N=22 $MEAN22 ms (helper off). Update the README production value for N=22."
else
  info "ADOPTION" "NOT adopted -- production .py stays 403_r2d/403_r3 (helper on for every N). See FAIL lines."
fi

banner "Results"
python3 - "$TSV" "$ANCH" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
anch=float(sys.argv[2])
print(f"{'cell':<6}{'N':>3}{'lay':>5}{'help':>7}{'free_mb':>8}{'kernel_ms':>14}{'vs anchor':>11}{'match':>6}{'sm':>6}{'h:mm:ss':>10}  env_prefix")
for r in rows:
    v=float(r['kernel_ms']); s=v/1000; ref=109437.0 if r['N']=='21' else anch
    print(f"{r['cell']:<6}{r['N']:>3}{r['layout']:>5}{r['helpers']+'/'+r['helper_mb']:>7}{r['free_mb']:>8}{v:>14.3f}{(v/ref-1)*100:>+10.3f}%{r['match']:>6}{r['sm_mean']:>6}{int(s//3600):>4}:{int(s%3600//60):02d}:{s%60:04.1f}  {r['env_prefix']}")
print(f"(N=21 rows vs 109,437; N=22 rows vs anchor {anch:.3f})")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
