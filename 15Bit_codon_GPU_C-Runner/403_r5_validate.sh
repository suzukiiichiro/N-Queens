#!/usr/bin/env bash
# 403_r5_validate.sh
#
# rev403-r5 -- the 403 layout re-spelled ("403b"): split ld at the BOTTOM.
#   A = col | avail<<22 | (ld>>1)<<44 ; B = (rd & ~1) | (ld & 1) ; pop reads
#   rd back unmasked. Only the three 403 branches change (+ PACK403_LDLO
#   retired); the 402 branches are byte-identical to 403_r3 / 402_r5.
#
#   S0 static: 402 branch statements identical to 403_r3; the six new 403
#      statements present; code-region fingerprint removed=7 added=6;
#      kernel sha 7799b9e4; gcc CPU harness; notes                    HARD
#   S1 ptxas: both instantiations 160 B / 0 spill / regs <= 44        HARD
#   S2 <402> SASS identical to 402_r5 (N=21 production untouched)     HARD
#   S3 STATED: <403> instruction count < 504 (403_r2d/r3/r4: 504)
#   Y2 CPU per-record equality vs 403_r3 on real sched records:
#      N=22 auto, N=21 NQ_LAYOUT=403, N=21 auto (NREC records each)   HARD
#   Y3 N=23 -> rc=3 ; NQ_LAYOUT=402 @ N=22 -> rc=3                      HARD
#   Cells (direct = BLOCK 32, MAX_BLOCKS 960, helper 1 + 128 MB):
#     Z3  403_r3  N=21 NQ_LAYOUT=403 direct     (the +2.79% penalty, old)
#     Z5  403_r5  N=21 NQ_LAYOUT=403 direct     (the penalty, new spelling)
#     X0  403_r3  N=22 direct                   (anchor; r3 H960 987,831)
#     X1  403_r5  N=22 direct                   (results.bin == X0, HARD)
#     G22 ./403_r5Py -g 22 22                   (production path)  [SKIP_G22=1]
#   P1 STATED: X1/X0 <= 0.985   (refuted if >= 0.997; between = grey)
#   P2 STATED: Z5/Z3 <= 0.986   (refuted if >= 0.997)
#   P3 HARD:   G22 oracle MATCH, layout=403 ; info: G22 vs 987,079
#   P4:        1710 MHz all cells
#   ADOPT: S0-S2, Y2, Y3, X1 identity, P1 held, P3 -> 403_r5 is production.
#
# USAGE
#   STATIC_ONLY=1 bash 403_r5_validate.sh      # OK=15
#                 bash 403_r5_validate.sh      # ~60 min
#   SKIP_G22=1    bash 403_r5_validate.sh      # ~43 min

set -u

REV="403_r5"
PY_SRC="${PY_SRC:-403_r5Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-${PY_SRC%.py}}"
PREV_PY="${PREV_PY:-403_r3Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-403_r5_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-403_r5_kernel_maxd14}"
PREV_CU="${PREV_CU:-403_r3_kernel_maxd14.cu}"
PREV_BIN="${PREV_BIN:-403_r3_kernel_maxd14}"
[[ -x "$PREV_BIN" || -f "$PREV_CU" ]] || { PREV_CU="403_r2d_kernel_maxd14.cu"; PREV_BIN="403_r2d_kernel_maxd14"; }
P402_BIN="${P402_BIN:-402_r5_kernel_maxd14}"
P403_BIN="${P403_BIN:-403_kernel_maxd14}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
CODON="${CODON:-codon}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
BLOCK="${BLOCK:-32}"
MB="${MB:-960}"
HMB="${HMB:-128}"
NREC="${NREC:-2048}"
KERNEL_SHA_403_R5="${KERNEL_SHA_403_R5:-7799b9e456356748}"
ANCHOR_X0="${ANCHOR_X0:-987830.56}"        # 403-r3 H960 (direct, helper 1+128)
ANCHOR_G22="${ANCHOR_G22:-987079.25}"      # 403-r2d G22 (dispatcher)
ANCHOR_Z3="${ANCHOR_Z3:-112472.7}"         # 403-r2d X3 (N=21 layout 403, direct)
SKIP_G22="${SKIP_G22:-0}"
COOLDOWN="${COOLDOWN:-15}"
CLK_INTERVAL="${CLK_INTERVAL:-10}"
STATIC_ONLY="${STATIC_ONLY:-0}"
ORACLE21=314666222712
ORACLE22=2691008701644

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
ratio() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.4f",a/b}'; }
le() { awk -v x="$1" -v y="$2" 'BEGIN{exit !(x<=y)}'; }
ge() { awk -v x="$1" -v y="$2" 'BEGIN{exit !(x>=y)}'; }
absdiff_le() { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=t)}'; }
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS (registered refutations are listed here too -- read the label):"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi; exit 0; }

# ---------------------------------------------------------------------
# 1. Static (S0)
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
{ grep -q "^# 403-r5 " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 403-r5 ...' note block"
grep -qE '^REV_TAG:str="403_r5"' "$CODE" && pass "source_rev_tag_is_403_r5" || fail "source_rev_tag_is_403_r5" "wrong REV_TAG"
grep -qF 'CRunnerEntry(14,"./403_r5_kernel_maxd14","NQ_EXTRA_CTX=1 NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_points_at_403_r5_keeps_helper_prefix" || fail "source_table_points_at_403_r5_keeps_helper_prefix" "table entry wrong"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=960\b" "$CODE" && pass "source_default_max_blocks_is_960" || fail "source_default_max_blocks_is_960" "default not 960"
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_403_r3Py (removed=3 added=3 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_403_r3Py" "removed=$PR_ added=$PA_, expected 3/3"
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
[[ "$KB" == "$KERNEL_SHA_403_R5"* ]] && pass "cu_kernel_region_sha_matches_delivered_r5 (${KB:0:16})" || fail "cu_kernel_region_sha" "sha ${KB:0:16} != $KERNEL_SHA_403_R5 -- not the delivered r5 source"
CR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true); CA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
[[ "$CR_" == "7" && "$CA_" == "6" ]] && pass "cu_code_region_fingerprint_vs_${PREV_CU%.cu} (removed=7 added=6)" || fail "cu_code_region_fingerprint" "removed=$CR_ added=$CA_, expected 7/6"
# 402 branch statements: every line mentioning the 402 pack must be byte-identical (order and count)
grep -E 'PACK402_MASK|stack_b\[stack_ptr\] = cur_rd;|cur_rd  = stack_b\[stack_ptr\];' "/tmp/${REV}_prev_code.cu" > "/tmp/${REV}_402_prev.txt"
grep -E 'PACK402_MASK|stack_b\[stack_ptr\] = cur_rd;|cur_rd  = stack_b\[stack_ptr\];' "/tmp/${REV}_cur_code.cu"  > "/tmp/${REV}_402_cur.txt"
{ cmp -s "/tmp/${REV}_402_prev.txt" "/tmp/${REV}_402_cur.txt" && [[ "$(wc -l < "/tmp/${REV}_402_cur.txt")" -ge 8 ]]; } && pass "cu_402_branch_statements_identical_to_${PREV_CU%.cu} ($(wc -l < "/tmp/${REV}_402_cur.txt") lines)" || fail "cu_402_branch_statements" "the 402 layout statements changed -- <402> SASS will not be 402_r5's"
N403=0
grep -cF '| ((uint64_t)cur_avail << 22) | ((uint64_t)(cur_ld >> 1) << 44);' "/tmp/${REV}_cur_code.cu" | grep -q '^2$' && N403=$((N403+1))
grep -cF 'stack_b[stack_ptr] = (cur_rd & ~1u) | (cur_ld & 1u);' "/tmp/${REV}_cur_code.cu" | grep -q '^2$' && N403=$((N403+1))
grep -qF 'cur_ld  = ((uint32_t)(packed_a >> 43) & ~1u) | (packed_b & 1u);' "/tmp/${REV}_cur_code.cu" && N403=$((N403+1))
grep -qF 'cur_rd  = packed_b;' "/tmp/${REV}_cur_code.cu" && N403=$((N403+1))
grep -q 'PACK403_LDLO' "/tmp/${REV}_cur_code.cu" || N403=$((N403+1))
[[ "$N403" == "5" ]] && pass "cu_403b_statements_present (2 pushes, pop ld/rd, PACK403_LDLO retired)" || fail "cu_403b_statements_present" "$N403 of 5 checks"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"/tmp/${REV}_gcc.log" && pass "cu_cpu_harness_builds" || fail "cu_cpu_harness_builds" "$(head -3 /tmp/${REV}_gcc.log)"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_prev" "$PREV_CU" -lm 2>"/tmp/${REV}_gcc_prev.log" || fail "cu_cpu_harness_prev_builds" "$(head -3 /tmp/${REV}_gcc_prev.log)"
grep -q "rev403-r5" "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev403-r5 note in header"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Build, ptxas (S1), SASS (S2/S3), codon, inputs
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR/sass" "$LOGDIR/bins"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
[[ ! -x "$NVCC" ]] && NVCC="nvcc"; [[ ! -x "$CUOBJDUMP" ]] && CUOBJDUMP="cuobjdump"
banner "Building"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc_${REV}.log" 2>&1
[[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "$(grep -i error "$LOGDIR/05_nvcc_${REV}.log" | head -3)"; summary_exit; }
for L in 402 403; do
  FR="$(grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${L}E" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"
  SP="$(grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${L}E" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes spill stores, [0-9]* bytes spill loads' | head -1)"
  RG="$(grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${L}E" "$LOGDIR/05_nvcc_${REV}.log" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"
  [[ "$FR" == "160" && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "S1_ptxas_L${L}_frame160_spill0_regs${RG}" || { fail "S1_ptxas_L${L}" "frame=${FR}B regs=${RG} ${SP}"; summary_exit; }
done
if [[ ! -x "$PREV_BIN" ]]; then "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$PREV_BIN" "$PREV_CU" -lcuda > "$LOGDIR/05b_nvcc_prev.log" 2>&1; fi
[[ -x "$PREV_BIN" ]] && pass "prev_binary_available[$PREV_BIN]" || { fail "prev_binary" "cannot build $PREV_BIN from $PREV_CU"; summary_exit; }
sass_tool() { python3 - "$@" <<'PYEOF'
import sys, re
mode = sys.argv[1]
def funcs(path):
    out = {}; cur = None
    for line in open(path, encoding='utf-8', errors='replace'):
        m = re.match(r'\s*Function\s*:\s*(\S+)', line)
        if m: cur = m.group(1); out[cur] = []; continue
        if cur is None: continue
        m = re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s*(.*?);\s*(/\*.*\*/)?\s*$', line)
        if m: out[cur].append(m.group(2).strip())
    return out
if mode == "cmp":
    label, fa, fragA, fb, fragB = sys.argv[2:7]
    A = funcs(fa); B = funcs(fb)
    ka = [k for k in A if fragA in k]; kb = [k for k in B if fragB in k]
    if len(ka) != 1 or len(kb) != 1: print(f"{label}: CANNOT_LOCATE A={ka} B={kb}"); sys.exit(2)
    ia, ib = A[ka[0]], B[kb[0]]
    if ia == ib: print(f"{label}: IDENTICAL ({len(ia)} instructions)"); sys.exit(0)
    print(f"{label}: DIFFER {sum(1 for x, y in zip(ia, ib) if x != y)} of {len(ia)}/{len(ib)}"); sys.exit(1)
else:  # count
    fa, frag = sys.argv[2:4]
    A = funcs(fa); ka = [k for k in A if frag in k]
    print(len(A[ka[0]]) if len(ka) == 1 else -1)
PYEOF
}
"$CUOBJDUMP" -sass "./$CU_BIN" > "$LOGDIR/sass/${CU_BIN}.txt" 2>&1
if [[ -x "$P402_BIN" ]]; then
  "$CUOBJDUMP" -sass "./$P402_BIN" > "$LOGDIR/sass/${P402_BIN}.txt" 2>&1
  sass_tool cmp S2 "$LOGDIR/sass/${P402_BIN}.txt" "kernel_dfs_iter_gpu_maxd14PKj" "$LOGDIR/sass/${CU_BIN}.txt" "kernel_dfs_iter_gpu_maxd14ILi402E" >/dev/null && pass "S2_sass_402_identical_to_402_r5" || { fail "S2_sass_402_differs_from_402_r5" "the N=21 side moved -- stop"; summary_exit; }
else
  "$CUOBJDUMP" -sass "./$PREV_BIN" > "$LOGDIR/sass/${PREV_BIN}.txt" 2>&1
  sass_tool cmp S2 "$LOGDIR/sass/${PREV_BIN}.txt" "kernel_dfs_iter_gpu_maxd14ILi402E" "$LOGDIR/sass/${CU_BIN}.txt" "kernel_dfs_iter_gpu_maxd14ILi402E" >/dev/null && pass "S2_sass_402_identical_to_${PREV_BIN}_402 (402_r5 binary absent)" || { fail "S2_sass_402_differs" "the N=21 side moved -- stop"; summary_exit; }
fi
N403I="$(sass_tool count "$LOGDIR/sass/${CU_BIN}.txt" "kernel_dfs_iter_gpu_maxd14ILi403E")"
[[ "$N403I" -gt 0 && "$N403I" -lt 504 ]] && pass "S3_403_instruction_count_below_504 ($N403I; predicted about 498)" || info "S3_403_instruction_count" "$N403I (stated < 504)"
[[ -x "$P403_BIN" ]] && { "$CUOBJDUMP" -sass "./$P403_BIN" > "$LOGDIR/sass/${P403_BIN}.txt" 2>&1; info "S3_403_vs_403_binary" "$(sass_tool cmp S3 "$LOGDIR/sass/${P403_BIN}.txt" "kernel_dfs_iter_gpu_maxd14PKj" "$LOGDIR/sass/${CU_BIN}.txt" "kernel_dfs_iter_gpu_maxd14ILi403E")"; }

rm -f "$PY_BIN"; "$CODON" build -release "$PY_SRC" > "$LOGDIR/06_codon_${REV}.log" 2>&1
[[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN]" || { fail "codon_build" "$(tail -3 "$LOGDIR/06_codon_${REV}.log")"; summary_exit; }

find_input() { local n="$1" c; for lg in $(ls -t 40*_crunner_logs/crunner_*_N${n}.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { echo "$c"; return; }; done; ls -t constellations_N${n}_*.sched394f.bin 2>/dev/null | head -1; }
IN21="$(find_input 21)"; IN22="$(find_input 22)"
[[ -f "${IN21:-/nonexistent}" ]] && pass "input21_located ($IN21)" || { fail "input21_located" "no N=21 sched input"; summary_exit; }
[[ -f "${IN22:-/nonexistent}" ]] && pass "input22_located ($IN22)" || { fail "input22_located" "no N=22 sched input"; summary_exit; }

# ---------------------------------------------------------------------
# 3. CPU equivalence (Y2) and refusals (Y3)
# ---------------------------------------------------------------------
banner "Y2: CPU per-record equality vs ${PREV_CU%.cu} ($NREC records)"
cpu_pair() {  # label N input env...
  local label="$1" n="$2" in="$3"; shift 3
  env "$@" "/tmp/${REV}_cpu_prev" "$n" "$in" "/tmp/${REV}_cpu_${label}_prev.bin" "$NREC" > "$LOGDIR/2_cpu_${label}_prev.log" 2>&1; local r1=$?
  env "$@" "/tmp/${REV}_cpu_cur"  "$n" "$in" "/tmp/${REV}_cpu_${label}_cur.bin"  "$NREC" > "$LOGDIR/2_cpu_${label}_cur.log"  2>&1; local r2=$?
  [[ "$r1" == "0" && "$r2" == "0" ]] && cmp -s "/tmp/${REV}_cpu_${label}_prev.bin" "/tmp/${REV}_cpu_${label}_cur.bin" && [[ -s "/tmp/${REV}_cpu_${label}_cur.bin" ]] \
    && pass "Y2_cpu_identical[$label] ($(( $(stat -c %s "/tmp/${REV}_cpu_${label}_cur.bin") / 8 )) records)" || { fail "Y2_cpu_identical[$label]" "rc=$r1/$r2 or per-record results differ"; summary_exit; }
}
cpu_pair N22_auto 22 "$IN22" NQ_LAYOUT=auto
cpu_pair N21_l403 21 "$IN21" NQ_LAYOUT=403
cpu_pair N21_auto 21 "$IN21" NQ_LAYOUT=auto
"/tmp/${REV}_cpu_cur" 23 "$IN22" /tmp/${REV}_y3a.bin 8 > "$LOGDIR/2_y3_n23.log" 2>&1; rc=$?
[[ "$rc" == "3" ]] && pass "Y3_N23_refused_rc3" || fail "Y3_N23_refused_rc3" "rc=$rc"
NQ_LAYOUT=402 "/tmp/${REV}_cpu_cur" 22 "$IN22" /tmp/${REV}_y3b.bin 8 > "$LOGDIR/2_y3_l402_n22.log" 2>&1; rc=$?
[[ "$rc" == "3" ]] && pass "Y3_layout402_at_N22_refused_rc3" || fail "Y3_layout402_at_N22_refused_rc3" "rc=$rc"

# ---------------------------------------------------------------------
# 4. Cells
# ---------------------------------------------------------------------
printf 'cell\tbinary\tN\tlayout\thelpers\tfree_mb\tkernel_ms\ttotal_sum\tmatch\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
CLKPID=""
gpu_gate() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_before_$1.txt"; [[ -z "$apps" ]] && return 0; fail "gpu_empty_before[$1]" "compute process(es) present: $(echo "$apps" | tr '\n' ' ')"; return 1; }
gpu_after() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_after_$1.txt"; [[ -z "$apps" ]] || fail "gpu_residue_after[$1]" "$(echo "$apps" | tr '\n' ' ')"; }
clk_start() { local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"; ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPID=$!; }
clk_stop() { [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.0f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_$1.tsv"); }
declare -A VAL SMM
parse_log() {  # cell binlabel N logfile  -> sets KMS TOT MATCH FREE HL LAYV, appends TSV, checks oracle+layout
  local cell="$1" b="$2" n="$3" lg="$4" start="$5" want_layout="$6"
  KMS="$(grep -o 'kernel_ms=[0-9.]*' "$lg" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$lg" && MATCH=1
  FREE="$(grep -o 'free_mb=[0-9]*' "$lg" | head -1 | cut -d= -f2)"; HL="$(grep -o 'helpers=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  LAYV="$(grep -o '\[gpu-layout\] N=[0-9]* requested=[a-z0-9]* layout=[0-9]*' "$lg" | head -1 | grep -o 'layout=[0-9]*' | cut -d= -f2)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$b" "$n" "${LAYV:-?}" "${HL:-?}" "${FREE:-?}" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start" >> "$TSV"
  local want; want="$([[ "$n" -le 21 ]] && echo "$ORACLE21" || echo "$ORACLE22")"
  [[ "${LAYV:-}" == "$want_layout" ]] || { fail "layout[$cell]" "layout=${LAYV:-?}, expected $want_layout"; return 1; }
  [[ "$MATCH" == "1" && "${TOT:-}" == "$want" ]] || { fail "oracle[$cell]" "total_sum='${TOT:-<none>}' match=$MATCH"; return 1; }
  VAL[$cell]="$KMS"; SMM[$cell]="$SMMEAN"
  info "$cell" "$b N=$n layout=$LAYV helpers=${HL:-?} kernel_ms=$KMS free_mb=$FREE sm_mean=$SMMEAN"
}
run_direct() {  # cell binary N layout_env want_layout
  local cell="$1" bin="$2" n="$3" lenv="$4" want="$5"; gpu_gate "$cell" || return 1
  local lg="$LOGDIR/3_${cell}.log" start; start="$(date -Is)"; clk_start "$cell"
  local in; in="$([[ "$n" -le 21 ]] && echo "$IN21" || echo "$IN22")"; local oracle; oracle="$([[ "$n" -le 21 ]] && echo "$ORACLE21" || echo "$ORACLE22")"
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX NQ_LAYOUT="$lenv" NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$MB" NQ_HELPER_CTX=1 NQ_HELPER_MB="$HMB" "./$bin" "$n" "$in" "$LOGDIR/bins/${cell}_results.bin" "$oracle" > "$lg" 2>&1 || true
  clk_stop "$cell"; gpu_after "$cell"
  parse_log "$cell" "$bin" "$n" "$lg" "$start" "$want"
}
run_g() {  # cell N
  local cell="$1" n="$2"; gpu_gate "$cell" || return 1
  local dlog="${REV}_crunner_logs/dispatch.log" clog="${REV}_crunner_logs/crunner_${CU_BIN}_N${n}.log" before=0; [[ -f "$dlog" ]] && before="$(wc -l < "$dlog")"
  local start; start="$(date -Is)"; clk_start "$cell"
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX -u NQ_LAYOUT -u NQ_HELPER_CTX -u NQ_HELPER_MB -u NQ_MAX_BLOCKS -u NQ_BLOCK "./$PY_BIN" -g "$n" "$n" > "$LOGDIR/3_${cell}.stdout" 2>&1; local rc=$?
  clk_stop "$cell"; gpu_after "$cell"
  [[ -f "$clog" ]] && cp "$clog" "$LOGDIR/3_${cell}.log"; [[ -f "$dlog" ]] && tail -n +"$((before+1))" "$dlog" > "$LOGDIR/3_${cell}_dispatch.log"
  [[ "$rc" == "0" ]] || { fail "rc[$cell]" "rc=$rc"; return 1; }
  parse_log "$cell" "$PY_BIN" "$n" "$LOGDIR/3_${cell}.log" "$start" "403"
}

banner "Z3 / Z5: N=21 with NQ_LAYOUT=403 (the +2.79% penalty), old vs new spelling"
run_direct Z3 "$PREV_BIN" 21 403 403 || summary_exit
d="$(abspct "${VAL[Z3]}" "$ANCHOR_Z3")"; le "$d" 0.3 && pass "Z3_anchor (${d}% from 403-r2d X3 $ANCHOR_Z3)" || info "Z3_anchor" "${d}% from $ANCHOR_Z3 -- P2 still read as Z5/Z3"
sleep "$COOLDOWN"; run_direct Z5 "$CU_BIN" 21 403 403 || summary_exit
rz="$(ratio "${VAL[Z5]}" "${VAL[Z3]}")"
if le "$rz" 0.986; then pass "P2_penalty_recovered_at_N21 (Z5/Z3 = $rz)"
elif ge "$rz" 0.997; then fail "P2_REFUTED_no_recovery_at_N21" "Z5/Z3 = $rz -- the 403 penalty is not the instruction count"
else fail "P2_grey_zone" "Z5/Z3 = $rz (stated <= 0.986, refuted >= 0.997)"; fi

banner "X0: ${PREV_BIN} N=22 direct (anchor)"
sleep "$COOLDOWN"; run_direct X0 "$PREV_BIN" 22 auto 403 || summary_exit
d="$(abspct "${VAL[X0]}" "$ANCHOR_X0")"; le "$d" 0.2 && pass "X0_anchor (${d}% from 403-r3 H960 $ANCHOR_X0)" || info "X0_anchor" "${d}% from $ANCHOR_X0 -- P1 still read as X1/X0"
banner "X1: ${CU_BIN} N=22 direct"
sleep "$COOLDOWN"; run_direct X1 "$CU_BIN" 22 auto 403 || summary_exit
cmp -s "$LOGDIR/bins/X0_results.bin" "$LOGDIR/bins/X1_results.bin" && [[ -s "$LOGDIR/bins/X1_results.bin" ]] && pass "X1_per_thread_results_identical_to_X0 ($(stat -c %s "$LOGDIR/bins/X1_results.bin") bytes)" || { fail "X1_per_thread_results_DIFFER" "results.bin X1 != X0 -- not adoptable"; }
rx="$(ratio "${VAL[X1]}" "${VAL[X0]}")"; px="$(pct "${VAL[X1]}" "${VAL[X0]}")"
P1=0
if le "$rx" 0.985; then pass "P1_403b_faster_at_N22 (X1/X0 = $rx, ${px}%)"; P1=1
elif ge "$rx" 0.997; then fail "P1_REFUTED_no_gain_at_N22" "X1/X0 = $rx (${px}%) -- fewer instructions, same time"
else fail "P1_grey_zone" "X1/X0 = $rx (${px}%) (stated <= 0.985, refuted >= 0.997)"; fi

if [[ "$SKIP_G22" != "1" ]]; then
  banner "G22: ./$PY_BIN -g 22 22 (production path)"
  sleep "$COOLDOWN"; run_g G22 22 || summary_exit
  pass "P3_G22_oracle_layout403"
  info "G22_vs_r2d_G22" "$(pct "${VAL[G22]}" "$ANCHOR_G22")% vs $ANCHOR_G22 (dispatcher, helper 1)"
fi
CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "P4_sm_clock_1710_all_cells" || fail "P4_sm_clock_1710_all_cells" "see clk_*.tsv"
if [[ "$P1" == "1" && "$FAIL" == "0" ]]; then
  info "ADOPTION" "403_r5 is the production binary/.py: N=21 unchanged (same SASS), N=22 = ${VAL[G22]:-<G22 skipped: run it before updating the README>} ms"
else
  info "ADOPTION" "NOT adopted -- production stays 403_r2d/403_r3. See FAIL lines."
fi

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
ref={r['cell']:float(r['kernel_ms']) for r in rows}
print(f"{'cell':<5}{'binary':<32}{'N':>3}{'lay':>5}{'help':>5}{'free_mb':>8}{'kernel_ms':>14}{'vs ref':>10}{'match':>6}{'sm':>6}{'h:mm:ss':>10}")
for r in rows:
    v=float(r['kernel_ms']); s=v/1000; c=r['cell']
    base=ref.get('Z3') if c in ('Z3','Z5') else ref.get('X0') if c in ('X0','X1') else 987079.25
    print(f"{c:<5}{r['binary'][:31]:<32}{r['N']:>3}{r['layout']:>5}{r['helpers']:>5}{r['free_mb']:>8}{v:>14.3f}{(v/base-1)*100:>+9.3f}%{r['match']:>6}{r['sm_mean']:>6}{int(s//3600):>4}:{int(s%3600//60):02d}:{s%60:04.1f}")
print("(Z5 vs Z3, X1 vs X0, G22 vs 403-r2d G22 987,079)")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
