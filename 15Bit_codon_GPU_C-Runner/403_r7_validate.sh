#!/usr/bin/env bash
# 403_r7_validate.sh
#
# rev403-r7 -- push side of the 403b layout: 7 packing instructions -> 5
#   (the searched variant w3: A's ld part as shl + masked OR in C, B as inline
#   PTX lop3.b32 0xD8), at both push sites. Pop and 402 branches untouched.
#   <403>: real 487 -> 483, hot loop 143 -> 141 (<402>: 482 / 140).
#   NOTE: counts are REAL instructions (tail alignment NOPs excluded); the
#   "496" of earlier revisions was the padded count.
#
#   S0 static: r7 statements present at both sites; 402 statements identical
#      to 403_r7; code-region fingerprint removed=6 added=18; kernel sha; gcc
#   S1 ptxas 160 B / 0 spill / regs <= 44                              HARD
#   S2 <402> SASS identical to 402_r5                                   HARD
#   S3 <403> SASS sha == db7e0ec2 (searched w3), real 483, loop 141      HARD
#   Y2 CPU per-record equality vs 403_r7: N=22 auto, N=21 l403, N=21 auto HARD
#      (the CPU side runs the new C spelling of A; B's asm is checked on the
#       GPU side by X7 byte identity and the oracle)
#   Y3 rc=3 cases                                                       HARD
#   Cells (direct = BLOCK 32 / MB 960 / helper 1 + 128 MB):
#     Z7  403_r7  N=21 NQ_LAYOUT=403   (anchor 403-r7 Z7 111,208)
#     Z7  403_r7  N=21 NQ_LAYOUT=403
#     X7  403_r7  N=22 direct          (anchor 403-r7 X7 977,058)  [SKIP_X6=1]
#     X7  403_r7  N=22 direct          results.bin == X7 (or r6's X7)  HARD
#     G22 ./403_r7Py -g 22 22                                        [SKIP_G22=1]
#   P1 STATED: X7/X7 <= 0.992 (2 loop instructions at 0.4-0.7% each)
#      <= 0.998 real but below the per-instruction rate (info, still adopt)
#      refuted >= 0.9997 ; between grey
#   P2 STATED: Z7/Z7 <= 0.992 ; same bands
#   P3 HARD: G22 oracle, layout=403.  P4: 1710 MHz all cells.
#   ADOPT: S0-S3, Y2, Y3, X7 identity, X7/X7 <= 0.998, P3 -> 403_r7 is production.
#
# USAGE
#   STATIC_ONLY=1 bash 403_r7_validate.sh      # OK=15
#                 bash 403_r7_validate.sh      # ~55 min
#   SKIP_X6=1     bash 403_r7_validate.sh      # ~38 min (X7 vs r6's X7 977,058)

set -u

REV="403_r7"
PY_SRC="${PY_SRC:-403_r7Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-${PY_SRC%.py}}"
PREV_PY="${PREV_PY:-403_r6Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-403_r7_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-403_r7_kernel_maxd14}"
PREV_CU="${PREV_CU:-403_r6_kernel_maxd14.cu}"
PREV_BIN="${PREV_BIN:-403_r6_kernel_maxd14}"
SASS403_SHA="${SASS403_SHA:-db7e0ec2c3261e09}"
SASS403_REAL="${SASS403_REAL:-483}"
SASS403_LOOP="${SASS403_LOOP:-141}"
SKIP_X6="${SKIP_X6:-0}"
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
KERNEL_SHA_403_R7="${KERNEL_SHA_403_R7:-77b1a90b3e501748}"
ANCHOR_X6="${ANCHOR_X6:-977058.5}"         # 403-r6 X6 (direct, helper 1+128)
ANCHOR_G22="${ANCHOR_G22:-976391.2}"       # 403-r6 G22 (dispatcher)
ANCHOR_Z6="${ANCHOR_Z6:-111208.1}"         # 403-r6 Z6 (N=21 layout 403, direct)
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
{ grep -q "^# 403-r7 " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 403-r7 ...' note block"
grep -qE '^REV_TAG:str="403_r7"' "$CODE" && pass "source_rev_tag_is_403_r7" || fail "source_rev_tag_is_403_r7" "wrong REV_TAG"
grep -qF 'CRunnerEntry(14,"./403_r7_kernel_maxd14","NQ_EXTRA_CTX=1 NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_points_at_403_r7_keeps_helper_prefix" || fail "source_table_points_at_403_r7_keeps_helper_prefix" "table entry wrong"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=960\b" "$CODE" && pass "source_default_max_blocks_is_960" || fail "source_default_max_blocks_is_960" "default not 960"
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_403_r6Py (removed=3 added=3 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_403_r6Py" "removed=$PR_ added=$PA_, expected 3/3"
else info "py_diff_fingerprint_vs_403_r6Py" "skipped ($PREV_PY absent)"; fi
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
[[ "$KB" == "$KERNEL_SHA_403_R7"* ]] && pass "cu_kernel_region_sha_matches_delivered_r7 (${KB:0:16})" || fail "cu_kernel_region_sha" "sha ${KB:0:16} != $KERNEL_SHA_403_R7 -- not the delivered r7 source"
CR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true); CA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
[[ "$CR_" == "6" && "$CA_" == "18" ]] && pass "cu_code_region_fingerprint_vs_${PREV_CU%.cu} (removed=6 added=18)" || fail "cu_code_region_fingerprint" "removed=$CR_ added=$CA_, expected 6/18"
# 402 branch statements: every line mentioning the 402 pack must be byte-identical (order and count)
grep -E 'PACK402_MASK|stack_b\[stack_ptr\] = cur_rd;|cur_rd  = stack_b\[stack_ptr\];' "/tmp/${REV}_prev_code.cu" > "/tmp/${REV}_402_prev.txt"
grep -E 'PACK402_MASK|stack_b\[stack_ptr\] = cur_rd;|cur_rd  = stack_b\[stack_ptr\];' "/tmp/${REV}_cur_code.cu"  > "/tmp/${REV}_402_cur.txt"
{ cmp -s "/tmp/${REV}_402_prev.txt" "/tmp/${REV}_402_cur.txt" && [[ "$(wc -l < "/tmp/${REV}_402_cur.txt")" -ge 8 ]]; } && pass "cu_402_branch_statements_identical_to_${PREV_CU%.cu} ($(wc -l < "/tmp/${REV}_402_cur.txt") lines)" || fail "cu_402_branch_statements" "the 402 layout statements changed -- <402> SASS will not be 402_r5's"
N403=0
cnt() { grep -cF -- "$1" "/tmp/${REV}_cur_code.cu"; }
[[ "$(cnt 'stack_a[stack_ptr] = (uint64_t)cur_col | ((uint64_t)cur_avail << 22)')" == "2" ]] && N403=$((N403+1))
[[ "$(cnt '| ((uint64_t)((cur_ld << 11) & 0xFFFFF000u) << 32);')" == "2" ]] && N403=$((N403+1))
[[ "$(cnt 'asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(pb403) : "r"(cur_rd), "r"(cur_ld), "r"(1u));')" == "2" ]] && N403=$((N403+1))
[[ "$(cnt 'pb403 = (cur_rd & ~1u) | (cur_ld & 1u);')" == "2" ]] && N403=$((N403+1))
[[ "$(cnt 'stack_b[stack_ptr] = pb403; }')" == "2" ]] && N403=$((N403+1))
[[ "$(cnt '(cur_ld >> 1) << 44')" == "0" ]] && N403=$((N403+1))
[[ "$(cnt 'asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(cur_ld) : "r"(a43), "r"(packed_b), "r"(1u));')" == "1" ]] && N403=$((N403+1))
[[ "$(cnt 'cur_ld  = (a43 & ~1u) | (packed_b & 1u);')" == "1" ]] && N403=$((N403+1))
[[ "$N403" == "8" ]] && pass "cu_r7_statements_present (push: shl+masked-OR A and asm lop3 B at 2 sites, old spelling gone; r6 pop unchanged)" || fail "cu_r7_statements_present" "$N403 of 8 checks"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"/tmp/${REV}_gcc.log" && pass "cu_cpu_harness_builds" || fail "cu_cpu_harness_builds" "$(head -3 /tmp/${REV}_gcc.log)"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_prev" "$PREV_CU" -lm 2>"/tmp/${REV}_gcc_prev.log" || fail "cu_cpu_harness_prev_builds" "$(head -3 /tmp/${REV}_gcc_prev.log)"
grep -q "rev403-r7" "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev403-r7 note in header"

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
S403="$(python3 - "$LOGDIR/sass/${CU_BIN}.txt" <<'PYEOF'
import sys,re,hashlib
out={};cur=None
for line in open(sys.argv[1],errors='replace'):
    m=re.match(r'\s*Function\s*:\s*(\S+)',line)
    if m: cur=m.group(1); out[cur]=[]; continue
    if cur is None: continue
    m=re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s*(.*?);\s*(/\*.*\*/)?\s*$',line)
    if m: out[cur].append((int(m.group(1),16),m.group(2).strip()))
I=out[[k for k in out if 'ILi403E' in k][0]]
real=sum(1 for _,s in I if not s.startswith('NOP'))
best=None  # hot loop = smallest backward-BRA span containing an LDL.64
for a,s in I:
    m=re.match(r'(@!?P\d\s+)?BRA\s+(0x[0-9a-f]+)$',s)
    if not m: continue
    t=int(m.group(2),16)
    if t>=a: continue
    b=[x for x in I if t<=x[0]<=a]
    if any(x[1].startswith('LDL.64') for x in b) and (best is None or len(b)<len(best)): best=b
print(hashlib.sha256('\n'.join(s for _,s in I).encode()).hexdigest()[:16], real, len(best) if best else -1, len(I))
PYEOF
)"
read -r S_SHA S_REAL S_LOOP S_TOT <<< "$S403"
[[ "$S_SHA" == "$SASS403_SHA" && "$S_REAL" == "$SASS403_REAL" && "$S_LOOP" == "$SASS403_LOOP" ]] && pass "S3_sass_403_is_the_searched_w3 (sha $S_SHA, real $S_REAL, loop $S_LOOP, padded $S_TOT)" || { fail "S3_sass_403_differs_from_search" "sha $S_SHA real $S_REAL loop $S_LOOP != $SASS403_SHA / $SASS403_REAL / $SASS403_LOOP -- ptxas produced something else; stop and read the SASS"; summary_exit; }

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

banner "Z6 / Z7: N=21 NQ_LAYOUT=403 direct, 403_r6 vs 403_r7"
run_direct Z6 "$PREV_BIN" 21 403 403 || summary_exit
d="$(abspct "${VAL[Z6]}" "$ANCHOR_Z6")"; le "$d" 0.3 && pass "Z6_anchor (${d}% from 403-r6 Z6 $ANCHOR_Z6)" || info "Z6_anchor" "${d}% from $ANCHOR_Z6 -- P2 still read as Z7/Z6"
sleep "$COOLDOWN"; run_direct Z7 "$CU_BIN" 21 403 403 || summary_exit
rz="$(ratio "${VAL[Z7]}" "${VAL[Z6]}")"
if le "$rz" 0.992; then pass "P2_gain_as_stated_at_N21 (Z7/Z6 = $rz <= 0.992)"
elif le "$rz" 0.998; then info "P2_gain_below_per_instruction_rate" "Z7/Z6 = $rz (real, but stated <= 0.992: part of r6's gain was chain depth)"
elif ge "$rz" 0.9997; then fail "P2_REFUTED_no_gain_at_N21" "Z7/Z6 = $rz"
else fail "P2_grey_zone" "Z7/Z6 = $rz (stated <= 0.992, adopt <= 0.998, refuted >= 0.9997)"; fi

X6REF=""; XLABEL=""
if [[ "$SKIP_X6" != "1" ]]; then
  banner "X6: ${PREV_BIN} N=22 direct (same-session anchor)"
  sleep "$COOLDOWN"; run_direct X6 "$PREV_BIN" 22 auto 403 || summary_exit
  d="$(abspct "${VAL[X6]}" "$ANCHOR_X6")"; le "$d" 0.2 && pass "X6_anchor (${d}% from 403-r6 X6 $ANCHOR_X6)" || info "X6_anchor" "${d}% from $ANCHOR_X6 -- P1 still read as X7/X6"
  X6REF="${VAL[X6]}"; XLABEL="X6 (same session)"; X6BIN="$LOGDIR/bins/X6_results.bin"
else
  X6REF="$ANCHOR_X6"; XLABEL="403-r6 X6 $ANCHOR_X6"; X6BIN="$(ls -t 403_r6_validate_*/bins/X6_results.bin 2>/dev/null | head -1)"
fi
banner "X7: ${CU_BIN} N=22 direct"
sleep "$COOLDOWN"; run_direct X7 "$CU_BIN" 22 auto 403 || summary_exit
if [[ -n "${X6BIN:-}" && -s "$X6BIN" ]]; then
  cmp -s "$X6BIN" "$LOGDIR/bins/X7_results.bin" && pass "X7_per_thread_results_identical ($(stat -c %s "$LOGDIR/bins/X7_results.bin") bytes vs $X6BIN)" || fail "X7_per_thread_results_DIFFER" "results.bin X7 != $X6BIN -- not adoptable"
else fail "X7_per_thread_results_reference_missing" "no X6 run and no 403_r6_validate_*/bins/X6_results.bin"; fi
rx="$(ratio "${VAL[X7]}" "$X6REF")"; px="$(pct "${VAL[X7]}" "$X6REF")"
P1=0
if le "$rx" 0.992; then pass "P1_gain_as_stated_at_N22 (X7 vs $XLABEL = $rx, ${px}% ; <= 0.992)"; P1=1
elif le "$rx" 0.998; then info "P1_gain_below_per_instruction_rate" "X7 vs $XLABEL = $rx (${px}%) (real and adoptable, but stated <= 0.992: part of r6's gain was chain depth)"; P1=1
elif ge "$rx" 0.9997; then fail "P1_REFUTED_no_gain_at_N22" "X7 vs $XLABEL = $rx (${px}%)"
else fail "P1_grey_zone" "X7 vs $XLABEL = $rx (${px}%) (stated <= 0.992, adopt <= 0.998, refuted >= 0.9997) -- Suzuki decides"; fi

if [[ "$SKIP_G22" != "1" ]]; then
  banner "G22: ./$PY_BIN -g 22 22 (production path)"
  sleep "$COOLDOWN"; run_g G22 22 || summary_exit
  pass "P3_G22_oracle_layout403"
  info "G22_vs_r6_G22" "$(pct "${VAL[G22]}" "$ANCHOR_G22")% vs $ANCHOR_G22 (403-r6 G22, dispatcher)"
fi
CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "P4_sm_clock_1710_all_cells" || fail "P4_sm_clock_1710_all_cells" "see clk_*.tsv"
if [[ "$P1" == "1" && "$FAIL" == "0" ]]; then
  info "ADOPTION" "403_r7 is the production binary/.py: N=21 unchanged (same SASS), N=22 = ${VAL[G22]:-<G22 skipped: run it before updating the README>} ms"
else
  info "ADOPTION" "NOT adopted (or grey) -- production stays 403_r6. See FAIL lines."
fi

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
ref={r['cell']:float(r['kernel_ms']) for r in rows}
print(f"{'cell':<5}{'binary':<32}{'N':>3}{'lay':>5}{'help':>5}{'free_mb':>8}{'kernel_ms':>14}{'vs ref':>10}{'match':>6}{'sm':>6}{'h:mm:ss':>10}")
for r in rows:
    v=float(r['kernel_ms']); s=v/1000; c=r['cell']
    base=ref.get('Z6') if c in ('Z6','Z7') else (ref.get('X6') or 977058.5) if c in ('X6','X7') else 976391.2
    print(f"{c:<5}{r['binary'][:31]:<32}{r['N']:>3}{r['layout']:>5}{r['helpers']:>5}{r['free_mb']:>8}{v:>14.3f}{(v/base-1)*100:>+9.3f}%{r['match']:>6}{r['sm_mean']:>6}{int(s//3600):>4}:{int(s%3600//60):02d}:{s%60:04.1f}")
print("(Z7 vs Z6, X7 vs X6 or r6 X6 977,058, G22 vs 403-r6 G22 976,391)")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd14" --exclude="*_cpu" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
