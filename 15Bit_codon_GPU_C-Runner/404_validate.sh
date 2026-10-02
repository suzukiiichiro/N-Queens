#!/usr/bin/env bash
# 404_validate.sh
#
# rev404 -- two bookkeeping removals in the DFS loop, both layouts:
#   A  save_sp dropped (always == stack_ptr)
#   B  4-bit depth LIFO dropped (cur_depth = popc(cur_col) - popc(root_col))
#   = the searched variant d3. <402> real 482 -> 463, loop 140 -> 130;
#   <403> 483 -> 465, 141 -> 131. Counts are REAL instructions (no tail NOPs).
#   <402> no longer equals 402_r5: N=21 is guarded by per-thread byte identity
#   against 403_r7 and the oracle.
#
#   S0 static: code region == generator(A+B) applied to 403_r7; save_sp and
#      stack_depth gone; pack statements identical to 403_r7; kernel sha; gcc
#   S1 ptxas 160 B / 0 spill / regs <= 44                               HARD
#   S2 <402> SASS sha dd619aaf, real 463, loop 130 (searched d3)         HARD
#   S3 <403> SASS sha 6a17e634, real 465, loop 131                       HARD
#   SD diagnostic binaries d1 (A) and d5 (A+B+C), generated from 403_r7,
#      <402> sha b03db4da / ab5608e9 (as searched)                       HARD
#   Y2 CPU per-record equality vs 403_r7: N=22 auto, N=21 l403, N=21 auto HARD
#   Y3 rc=3 cases                                                        HARD
#   Cells (direct = BLOCK 32 / MB 960 / helper 1 + 128 MB):
#     L0  403_r7   N=21 layout 402   (anchor 109,437)
#     L1  404_d1   N=21 layout 402   A only            results.bin == L0  HARD
#     L3  404      N=21 layout 402   A+B (candidate)   results.bin == L0  HARD
#     L5  404_d5   N=21 layout 402   A+B+C diagnostic  results.bin == L0  HARD
#     Z7  403_r7   N=21 NQ_LAYOUT=403 (anchor 403-r7 Z7 110,339)
#     Z4  404      N=21 NQ_LAYOUT=403                  results.bin == Z7  HARD
#     X7  403_r7   N=22 direct, only with RUN_X7=1 (anchor 403-r7 X7 968,932)
#     X4  404      N=22 direct       results.bin == X7 (or 403-r7's X7)   HARD
#     G21, G22  ./404Py -g 21 21 / -g 22 22                      [SKIP_G=1]
#   PA STATED: L1/L0 <= 0.994 ; <= 0.998 real but smaller ; refuted >= 0.9997
#   PB STATED: L3/L1 <= 0.980 ; <= 0.998 real, chain cost visible ;
#              >= 1.000 B loses (404-r2 would ship d1 only)
#   PX STATED: X4/X7 <= 0.985 ; <= 0.998 real but smaller ; refuted >= 0.9997
#   PC: L5/L3, information only (rev242 revisited).
#   P3 HARD: G21 oracle layout=402, G22 oracle layout=403.  P4: 1710 MHz.
#   ADOPT: hard gates, identities, L3/L0 <= 0.998, X4/X7 <= 0.998, P3
#          -> 404 is production (N=21 = G21, N=22 = G22).
#
# USAGE
#   STATIC_ONLY=1 bash 404_validate.sh      # OK=16
#                 bash 404_validate.sh      # ~50 min
#   SKIP_G=1      bash 404_validate.sh      # ~32 min (no dispatcher runs)
#   RUN_X7=1      bash 404_validate.sh      # +16.5 min (same-session N=22 anchor)

set -u

REV="404"
PY_SRC="${PY_SRC:-404Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-${PY_SRC%.py}}"
PREV_PY="${PREV_PY:-403_r7Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-404_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-404_kernel_maxd14}"
PREV_CU="${PREV_CU:-403_r7_kernel_maxd14.cu}"
PREV_BIN="${PREV_BIN:-403_r7_kernel_maxd14}"
SASS402_SHA="${SASS402_SHA:-dd619aaf2118d56c}"; SASS402_REAL="${SASS402_REAL:-463}"; SASS402_LOOP="${SASS402_LOOP:-130}"
SASS403_SHA="${SASS403_SHA:-6a17e634dd083882}"; SASS403_REAL="${SASS403_REAL:-465}"; SASS403_LOOP="${SASS403_LOOP:-131}"
D1_BIN="${D1_BIN:-404_d1_kernel_maxd14}"; D1_SHA402="${D1_SHA402:-b03db4da2b28dd5d}"
D5_BIN="${D5_BIN:-404_d5_kernel_maxd14}"; D5_SHA402="${D5_SHA402:-ab5608e947e586bc}"
RUN_X7="${RUN_X7:-0}"
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
KERNEL_SHA_404="${KERNEL_SHA_404:-6e3311adf7a4c450}"
ANCHOR_X7="${ANCHOR_X7:-968931.6}"         # 403-r7 X7 (N=22 direct, helper 1+128)
ANCHOR_G22="${ANCHOR_G22:-968205.9}"       # 403-r7 G22 (dispatcher)
ANCHOR_G21="${ANCHOR_G21:-109437.0}"       # 402-r5 G21 mean (<402> SASS unchanged through 403_r7)
ANCHOR_L0="${ANCHOR_L0:-109437.0}"
ANCHOR_Z7="${ANCHOR_Z7:-110339.0}"         # 403-r7 Z7 (N=21 layout 403, direct)
SKIP_G="${SKIP_G:-0}"
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
{ grep -q "^# 404 " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 404 ...' note block"
grep -qE '^REV_TAG:str="404"' "$CODE" && pass "source_rev_tag_is_404" || fail "source_rev_tag_is_404" "wrong REV_TAG"
grep -qF 'CRunnerEntry(14,"./404_kernel_maxd14","NQ_EXTRA_CTX=1 NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_points_at_404_keeps_helper_prefix" || fail "source_table_points_at_404_keeps_helper_prefix" "table entry wrong"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=960\b" "$CODE" && pass "source_default_max_blocks_is_960" || fail "source_default_max_blocks_is_960" "default not 960"
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_403_r7Py (removed=3 added=3 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_403_r7Py" "removed=$PR_ added=$PA_, expected 3/3"
else info "py_diff_fingerprint_vs_403_r7Py" "skipped ($PREV_PY absent)"; fi
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
[[ "$KB" == "$KERNEL_SHA_404"* ]] && pass "cu_kernel_region_sha_matches_delivered_404 (${KB:0:16})" || fail "cu_kernel_region_sha" "sha ${KB:0:16} != $KERNEL_SHA_404 -- not the delivered 404 source"
CR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true); CA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
[[ "$CR_" == "11" && "$CA_" == "4" ]] && pass "cu_code_region_fingerprint_vs_${PREV_CU%.cu} (removed=11 added=4)" || fail "cu_code_region_fingerprint" "removed=$CR_ added=$CA_, expected 11/4"
# 402 branch statements: every line mentioning the 402 pack must be byte-identical (order and count)
grep -E 'PACK402_MASK|stack_b\[stack_ptr\] = cur_rd;|cur_rd  = stack_b\[stack_ptr\];' "/tmp/${REV}_prev_code.cu" > "/tmp/${REV}_402_prev.txt"
grep -E 'PACK402_MASK|stack_b\[stack_ptr\] = cur_rd;|cur_rd  = stack_b\[stack_ptr\];' "/tmp/${REV}_cur_code.cu"  > "/tmp/${REV}_402_cur.txt"
{ cmp -s "/tmp/${REV}_402_prev.txt" "/tmp/${REV}_402_cur.txt" && [[ "$(wc -l < "/tmp/${REV}_402_cur.txt")" -ge 8 ]]; } && pass "cu_402_branch_statements_identical_to_${PREV_CU%.cu} ($(wc -l < "/tmp/${REV}_402_cur.txt") lines)" || fail "cu_402_branch_statements" "the 402 layout statements changed -- <402> SASS will not be 402_r5's"
# the search's generator: A = drop save_sp, B = depth from popc(col), C = future check from the depth bit
gen_variant() {  # base.cu out.cu letters(e.g. AB)
python3 - "$1" "$2" "$3" <<'PYEOF'
import sys, re
base, outp, letters = sys.argv[1:4]
src = open(base, encoding='utf-8').read()
def sub(s, pat, rep, n, what):
    out, k = re.subn(pat, rep, s, flags=re.M)
    if k != n: sys.exit("generator: %s matched %d times, expected %d" % (what, k, n))
    return out
def A(s):
    s = sub(s, r'^    uint32_t save_sp  = 0;\n', '', 1, 'save_sp decl')
    s = sub(s, r'^[ ]+save_sp   \+= 1u;\n', '', 2, 'save_sp += 1')
    s = sub(s, r'^([ ]+)if \(save_sp == 0u\) \{\n', r'\1if (stack_ptr == 0) {\n', 1, 'save_sp == 0')
    s = sub(s, r'^[ ]+save_sp -= 1u;\n', '', 1, 'save_sp -= 1')
    s = sub(s, r'\(long long\)debug_idx, stack_ptr, save_sp, cur_depth, cur_avail, terminal_depth,',
               r'(long long)debug_idx, stack_ptr, (unsigned)stack_ptr, cur_depth, cur_avail, terminal_depth,', 1, 'debug print')
    return s
def B(s):
    s = sub(s, r'^    uint64_t stack_depth = 0;\n', '    const int depth_base = (int)__builtin_popcount(root_col);\n', 1, 'stack_depth decl')
    s = sub(s, r'^[ ]+stack_depth = \(stack_depth << 4\) \| \(uint64_t\)\(uint32_t\)cur_depth;\n', '', 2, 'stack_depth push')
    s = sub(s, r'^([ ]+)cur_depth = \(int\)\(stack_depth & 15u\);\n[ ]+stack_depth >>= 4;\n',
            r'\1cur_depth = (int)__builtin_popcount(cur_col) - depth_base;\n', 1, 'stack_depth pop')
    return s
def C(s):
    return sub(s, r'^([ ]+)if \(future_check_mask != 0u\) \{\n([ ]+)if \(fc_flag != 0u\) \{[^\n]*\n',
               r'\1{\n\2if (((future_check_mask >> cur_depth) & 1u) != 0u) {\n', 1, 'future check')
for ch in letters: src = {'A': A, 'B': B, 'C': C}[ch](src)
open(outp, 'w', encoding='utf-8').write(src)
PYEOF
}
if gen_variant "$PREV_CU" "/tmp/${REV}_gen_d3.cu" AB 2>"/tmp/${REV}_gen.log"; then
  cu_code_region "/tmp/${REV}_gen_d3.cu" > "/tmp/${REV}_gen_d3_code.cu"
  cmp -s "/tmp/${REV}_gen_d3_code.cu" "/tmp/${REV}_cur_code.cu" && pass "cu_code_region_equals_generator_A+B_on_${PREV_CU%.cu}" || fail "cu_code_region_equals_generator" "the delivered .cu is not generator(A+B) applied to $PREV_CU (comment-stripped)"
else fail "cu_generator_runs_on_${PREV_CU%.cu}" "$(cat /tmp/${REV}_gen.log)"; fi
N404=0
cnt() { grep -cF -- "$1" "/tmp/${REV}_cur_code.cu"; }
[[ "$(grep -cE 'save_sp +[-+=]|uint32_t save_sp|, save_sp,' "/tmp/${REV}_cur_code.cu")" == "0" ]] && N404=$((N404+1))   # the CPU debug format string still prints the name
[[ "$(cnt 'stack_depth')" == "0" ]] && N404=$((N404+1))
[[ "$(cnt 'if (stack_ptr == 0) {')" == "1" ]] && N404=$((N404+1))
[[ "$(cnt 'const int depth_base = (int)__builtin_popcount(root_col);')" == "1" ]] && N404=$((N404+1))
[[ "$(cnt 'cur_depth = (int)__builtin_popcount(cur_col) - depth_base;')" == "1" ]] && N404=$((N404+1))
[[ "$(cnt 'if (fc_flag != 0u) {')" == "1" ]] && N404=$((N404+1))
grep -E 'stack_a\[stack_ptr\] =|stack_b\[stack_ptr\] =|PACK403_MASK|pb403|a43|packed_b' "/tmp/${REV}_prev_code.cu" > "/tmp/${REV}_pack_prev.txt"
grep -E 'stack_a\[stack_ptr\] =|stack_b\[stack_ptr\] =|PACK403_MASK|pb403|a43|packed_b' "/tmp/${REV}_cur_code.cu"  > "/tmp/${REV}_pack_cur.txt"
cmp -s "/tmp/${REV}_pack_prev.txt" "/tmp/${REV}_pack_cur.txt" && N404=$((N404+1))
[[ "$N404" == "7" ]] && pass "cu_404_statements (save_sp and stack_depth gone, popc depth present, future check untouched, 403 pack statements identical)" || fail "cu_404_statements" "$N404 of 7 checks"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"/tmp/${REV}_gcc.log" && pass "cu_cpu_harness_builds" || fail "cu_cpu_harness_builds" "$(head -3 /tmp/${REV}_gcc.log)"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_prev" "$PREV_CU" -lm 2>"/tmp/${REV}_gcc_prev.log" || fail "cu_cpu_harness_prev_builds" "$(head -3 /tmp/${REV}_gcc_prev.log)"
grep -q "rev404 " "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev404 note in header"

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
sass_stat() { python3 - "$1" "$2" <<'PYEOF'
import sys,re,hashlib
out={};cur=None
for line in open(sys.argv[1],errors='replace'):
    m=re.match(r'\s*Function\s*:\s*(\S+)',line)
    if m: cur=m.group(1); out[cur]=[]; continue
    if cur is None: continue
    m=re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s*(.*?);\s*(/\*.*\*/)?\s*$',line)
    if m: out[cur].append((int(m.group(1),16),m.group(2).strip()))
k=[k for k in out if sys.argv[2] in k]
if len(k)!=1: print("CANNOT_LOCATE -1 -1 -1"); sys.exit(0)
I=out[k[0]]
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
}
"$CUOBJDUMP" -sass "./$CU_BIN" > "$LOGDIR/sass/${CU_BIN}.txt" 2>&1
read -r S_SHA S_REAL S_LOOP S_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_dfs_iter_gpu_maxd14ILi402E)"
[[ "$S_SHA" == "$SASS402_SHA" && "$S_REAL" == "$SASS402_REAL" && "$S_LOOP" == "$SASS402_LOOP" ]] && pass "S2_sass_402_is_the_searched_d3 (sha $S_SHA, real $S_REAL, loop $S_LOOP, padded $S_TOT)" || { fail "S2_sass_402_differs_from_search" "sha $S_SHA real $S_REAL loop $S_LOOP != $SASS402_SHA / $SASS402_REAL / $SASS402_LOOP -- ptxas produced something else; stop and read the SASS"; summary_exit; }
read -r S_SHA S_REAL S_LOOP S_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_dfs_iter_gpu_maxd14ILi403E)"
[[ "$S_SHA" == "$SASS403_SHA" && "$S_REAL" == "$SASS403_REAL" && "$S_LOOP" == "$SASS403_LOOP" ]] && pass "S3_sass_403_is_the_searched_d3 (sha $S_SHA, real $S_REAL, loop $S_LOOP, padded $S_TOT)" || { fail "S3_sass_403_differs_from_search" "sha $S_SHA real $S_REAL loop $S_LOOP != $SASS403_SHA / $SASS403_REAL / $SASS403_LOOP -- stop and read the SASS"; summary_exit; }
# diagnostic binaries for the ladder: d1 = A, d5 = A+B+C, generated from the previous production source
mkdir -p "$LOGDIR/diag"
for spec in "d1:A:$D1_BIN:$D1_SHA402" "d5:ABC:$D5_BIN:$D5_SHA402"; do
  IFS=: read -r dv dl dbin dsha <<< "$spec"
  gen_variant "$PREV_CU" "$LOGDIR/diag/${REV}_${dv}_kernel_maxd14.cu" "$dl" 2>"$LOGDIR/diag/gen_${dv}.log" || { fail "SD_generate_${dv}" "$(cat "$LOGDIR/diag/gen_${dv}.log")"; summary_exit; }
  rm -f "$dbin"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$dbin" "$LOGDIR/diag/${REV}_${dv}_kernel_maxd14.cu" -lcuda > "$LOGDIR/05_nvcc_${REV}_${dv}.log" 2>&1
  [[ -x "$dbin" ]] || { fail "SD_build_${dv}" "$(grep -i error "$LOGDIR/05_nvcc_${REV}_${dv}.log" | head -2)"; summary_exit; }
  "$CUOBJDUMP" -sass "./$dbin" > "$LOGDIR/sass/${dbin}.txt" 2>&1
  read -r S_SHA S_REAL S_LOOP S_TOT <<< "$(sass_stat "$LOGDIR/sass/${dbin}.txt" kernel_dfs_iter_gpu_maxd14ILi402E)"
  [[ "$S_SHA" == "$dsha" ]] && pass "SD_diag_${dv}_402_sha_as_searched ($S_SHA, real $S_REAL, loop $S_LOOP)" || { fail "SD_diag_${dv}_sha" "<402> sha $S_SHA != $dsha"; summary_exit; }
  rm -f "$LOGDIR/diag/${REV}_${dv}_kernel_maxd14.cu"
done

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
run_g() {  # cell N want_layout
  local cell="$1" n="$2" wantl="$3"; gpu_gate "$cell" || return 1
  local dlog="${REV}_crunner_logs/dispatch.log" clog="${REV}_crunner_logs/crunner_${CU_BIN}_N${n}.log" before=0; [[ -f "$dlog" ]] && before="$(wc -l < "$dlog")"
  local start; start="$(date -Is)"; clk_start "$cell"
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX -u NQ_LAYOUT -u NQ_HELPER_CTX -u NQ_HELPER_MB -u NQ_MAX_BLOCKS -u NQ_BLOCK "./$PY_BIN" -g "$n" "$n" > "$LOGDIR/3_${cell}.stdout" 2>&1; local rc=$?
  clk_stop "$cell"; gpu_after "$cell"
  [[ -f "$clog" ]] && cp "$clog" "$LOGDIR/3_${cell}.log"; [[ -f "$dlog" ]] && tail -n +"$((before+1))" "$dlog" > "$LOGDIR/3_${cell}_dispatch.log"
  [[ "$rc" == "0" ]] || { fail "rc[$cell]" "rc=$rc"; return 1; }
  parse_log "$cell" "$PY_BIN" "$n" "$LOGDIR/3_${cell}.log" "$start" "$wantl"
}

ADOPT=1
same_bin() {  # cell refcell-or-file label
  local ref="$2"; [[ -f "$ref" ]] || ref="$LOGDIR/bins/${2}_results.bin"
  if [[ -s "$ref" ]] && cmp -s "$ref" "$LOGDIR/bins/${1}_results.bin"; then pass "${1}_per_thread_results_identical ($(stat -c %s "$LOGDIR/bins/${1}_results.bin") bytes vs $ref)"
  else fail "${1}_per_thread_results_DIFFER" "results.bin $1 != $ref -- not adoptable"; ADOPT=0; fi
}
banner "N=21 ladder, direct, layout 402: L0 $PREV_BIN / L1 $D1_BIN (A) / L3 $CU_BIN (A+B) / L5 $D5_BIN (A+B+C)"
run_direct L0 "$PREV_BIN" 21 auto 402 || summary_exit
d="$(abspct "${VAL[L0]}" "$ANCHOR_L0")"; le "$d" 0.15 && pass "L0_anchor (${d}% from 109,437)" || info "L0_anchor" "${d}% from $ANCHOR_L0 -- ratios are still read within this session"
sleep "$COOLDOWN"; run_direct L1 "$D1_BIN" 21 auto 402 || summary_exit; same_bin L1 L0
sleep "$COOLDOWN"; run_direct L3 "$CU_BIN" 21 auto 402 || summary_exit; same_bin L3 L0
sleep "$COOLDOWN"; run_direct L5 "$D5_BIN" 21 auto 402 || summary_exit; same_bin L5 L0
rA="$(ratio "${VAL[L1]}" "${VAL[L0]}")"; rB="$(ratio "${VAL[L3]}" "${VAL[L1]}")"; rAB="$(ratio "${VAL[L3]}" "${VAL[L0]}")"; rC="$(ratio "${VAL[L5]}" "${VAL[L3]}")"
if le "$rA" 0.994; then pass "PA_gain_as_stated (L1/L0 = $rA <= 0.994)"
elif le "$rA" 0.998; then info "PA_gain_smaller_than_stated" "L1/L0 = $rA (real, stated <= 0.994)"
elif ge "$rA" 0.9997; then fail "PA_REFUTED_save_sp_removal_no_gain" "L1/L0 = $rA"
else fail "PA_grey_zone" "L1/L0 = $rA (stated <= 0.994, real <= 0.998, refuted >= 0.9997)"; fi
if le "$rB" 0.980; then pass "PB_gain_as_stated (L3/L1 = $rB <= 0.980)"
elif le "$rB" 0.998; then info "PB_gain_smaller_than_stated" "L3/L1 = $rB (real; the longer depth chain after a pop is visible)"
elif ge "$rB" 1.000; then fail "PB_REFUTED_popc_depth_loses" "L3/L1 = $rB -- 404-r2 would ship d1 (A) only"
else fail "PB_grey_zone" "L3/L1 = $rB (stated <= 0.980, real <= 0.998, loses >= 1.000)"; fi
info "PC_future_check_depth_bit" "L5/L3 = $rC ($(pct "${VAL[L5]}" "${VAL[L3]}")%) -- information only (rev242 lost 1.3% with this form)"
le "$rAB" 0.998 && pass "N21_adoption_condition (L3/L0 = $rAB, $(pct "${VAL[L3]}" "${VAL[L0]}")%)" || { fail "N21_adoption_condition" "L3/L0 = $rAB > 0.998"; ADOPT=0; }

banner "Z7 / Z4: N=21 NQ_LAYOUT=403 direct, $PREV_BIN vs $CU_BIN"
sleep "$COOLDOWN"; run_direct Z7 "$PREV_BIN" 21 403 403 || summary_exit
d="$(abspct "${VAL[Z7]}" "$ANCHOR_Z7")"; le "$d" 0.3 && pass "Z7_anchor (${d}% from 403-r7 Z7 $ANCHOR_Z7)" || info "Z7_anchor" "${d}% from $ANCHOR_Z7"
sleep "$COOLDOWN"; run_direct Z4 "$CU_BIN" 21 403 403 || summary_exit; same_bin Z4 Z7
info "Z4_vs_Z7" "Z4/Z7 = $(ratio "${VAL[Z4]}" "${VAL[Z7]}") ($(pct "${VAL[Z4]}" "${VAL[Z7]}")%)"

X7REF="$ANCHOR_X7"; XLABEL="403-r7 X7 $ANCHOR_X7"; X7BIN="$(ls -t 403_r7_validate_*/bins/X7_results.bin 2>/dev/null | head -1)"
if [[ "$RUN_X7" == "1" ]]; then
  banner "X7: ${PREV_BIN} N=22 direct (same-session anchor)"
  sleep "$COOLDOWN"; run_direct X7 "$PREV_BIN" 22 auto 403 || summary_exit
  X7REF="${VAL[X7]}"; XLABEL="X7 (same session)"; X7BIN="$LOGDIR/bins/X7_results.bin"
fi
banner "X4: ${CU_BIN} N=22 direct"
sleep "$COOLDOWN"; run_direct X4 "$CU_BIN" 22 auto 403 || summary_exit
if [[ -n "${X7BIN:-}" && -s "$X7BIN" ]]; then same_bin X4 "$X7BIN"
else fail "X4_per_thread_results_reference_missing" "no X7 run and no 403_r7_validate_*/bins/X7_results.bin (use RUN_X7=1)"; ADOPT=0; fi
rx="$(ratio "${VAL[X4]}" "$X7REF")"; px="$(pct "${VAL[X4]}" "$X7REF")"
if le "$rx" 0.985; then pass "PX_gain_as_stated_at_N22 (X4 vs $XLABEL = $rx, ${px}%)"
elif le "$rx" 0.998; then info "PX_gain_smaller_than_stated" "X4 vs $XLABEL = $rx (${px}%) (real and adoptable; stated <= 0.985)"
elif ge "$rx" 0.9997; then fail "PX_REFUTED_no_gain_at_N22" "X4 vs $XLABEL = $rx (${px}%)"; ADOPT=0
else fail "PX_grey_zone" "X4 vs $XLABEL = $rx (${px}%) -- Suzuki decides"; ADOPT=0; fi

if [[ "$SKIP_G" != "1" ]]; then
  banner "G21: ./$PY_BIN -g 21 21 (production path)"
  sleep "$COOLDOWN"; run_g G21 21 402 || summary_exit
  info "G21_vs_109437" "$(pct "${VAL[G21]}" "$ANCHOR_G21")% vs $ANCHOR_G21 ; vs L3 $(pct "${VAL[G21]}" "${VAL[L3]}")%"
  banner "G22: ./$PY_BIN -g 22 22 (production path)"
  sleep "$COOLDOWN"; run_g G22 22 403 || summary_exit
  pass "P3_G21_G22_oracle_and_layout"
  info "G22_vs_r7_G22" "$(pct "${VAL[G22]}" "$ANCHOR_G22")% vs $ANCHOR_G22 (403-r7 G22, dispatcher)"
fi
CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "P4_sm_clock_1710_all_cells" || { fail "P4_sm_clock_1710_all_cells" "see clk_*.tsv"; ADOPT=0; }
if [[ "$ADOPT" == "1" ]]; then
  info "ADOPTION" "404 is the production binary/.py: N=21 = ${VAL[G21]:-<G21 skipped: run it before updating the README>} ms, N=22 = ${VAL[G22]:-<G22 skipped>} ms"
else
  info "ADOPTION" "NOT adopted (or grey) -- production stays 403_r7. See FAIL lines."
fi

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
ref={r['cell']:float(r['kernel_ms']) for r in rows}
print(f"{'cell':<5}{'binary':<32}{'N':>3}{'lay':>5}{'help':>5}{'free_mb':>8}{'kernel_ms':>14}{'vs ref':>10}{'match':>6}{'sm':>6}{'h:mm:ss':>10}")
for r in rows:
    v=float(r['kernel_ms']); s=v/1000; c=r['cell']
    base=ref.get('L0') if c.startswith('L') else ref.get('Z7') if c.startswith('Z') else (ref.get('X7') or 968931.6) if c.startswith('X') else 109437.0 if c=='G21' else 968205.9
    print(f"{c:<5}{r['binary'][:31]:<32}{r['N']:>3}{r['layout']:>5}{r['helpers']:>5}{r['free_mb']:>8}{v:>14.3f}{(v/base-1)*100:>+9.3f}%{r['match']:>6}{r['sm_mean']:>6}{int(s//3600):>4}:{int(s%3600//60):02d}:{s%60:04.1f}")
print("(L* vs L0, Z4 vs Z7, X4 vs X7 or 403-r7 X7 968,932, G21 vs 109,437, G22 vs 403-r7 G22 968,206)")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd14" --exclude="*_cpu" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
