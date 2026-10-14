#!/usr/bin/env bash
# 408_validate.sh
#
# rev408 -- T2: the mark arm reads one 64-bit per-CODE table
#   (mt = MARK_T64 >> (block_code*8); step = mt&3; lo2 = (mt>>2)&3; kt = (mt>>4)&3;
#    nld = ((ld|bit) << step) | lo2; nrd = ((rd|bit) >> step) | kt*n4).
#   Kernel == the measured diagnostic D1 (generator T2 on 407).
#   <403> real 458 -> 445, loop 121 -> 108, hot-loop MOV 14 -> 8; <402> real 459 -> 446, loop 122 -> 109.
#   408_diag measured D1/D0 = 0.9906 at N=21 (-0.94%). Pu (push w/o branch region) +63% -> closed.
#
#   S0 static: code region == generator(T2) applied to 407; gcc
#   S1 ptxas 160 B / 0 spill / regs <= 44                               HARD
#   S2 <402> SASS sha 821302f2, real 446, loop 109                       HARD
#   S3 <403> SASS sha e940cb96, real 445, loop 108                       HARD
#   Y2 CPU per-record equality vs 407: N=22 auto, N=21 l402, N=21 auto    HARD
#   Y3 rc=3 cases                                                        HARD
#   Cells (direct = BLOCK 32 / MB 960 / helper 1 + 128 MB):
#     B0  407  N=21 auto (403)     anchor 407 B1 99,230.2
#     B1  408  N=21 auto (403)     results.bin == B0                     HARD
#     B2  408  N=21 NQ_LAYOUT=402  results.bin == B0 ; time is info      HARD
#     X8  407  N=22 direct         only with RUN_X8=1 (anchor 407 X8 868,716.5)
#     X9  408  N=22 direct         results.bin == X8 (or 407's X8)       HARD
#     G21, G22  ./408Py -g 21 21 / -g 22 22                      [SKIP_G=1]
#   R1 STATED: B1/B0 <= 0.992 ; <= 0.998 adoptable ; refuted >= 0.9997
#   R2 STATED: X9/X8 <= 0.993 ; <= 0.998 adoptable ; refuted >= 0.9997
#      (marked steps are 8.1% of all steps at N=22 vs 8.9% at N=21: a slightly smaller gain)
#   R3 HARD: G21, G22 oracle, layout=403.   R4: 1710 MHz.
#   ADOPT: hard gates, identities, B1/B0 <= 0.998 AND X9/X8 <= 0.998, R3
#          -> 408 is production (N=21 = G21, N=22 = G22).
#          If the two N disagree the harness reports NOT adopted; Suzuki decides.
#
# USAGE
#   STATIC_ONLY=1 bash 408_validate.sh      # OK=15
#                 bash 408_validate.sh      # ~41 min
#   SKIP_G=1      bash 408_validate.sh      # ~24 min
#   RUN_X8=1      bash 408_validate.sh      # +14.5 min (same-session N=22 anchor)

set -u

REV="408"
PY_SRC="${PY_SRC:-408Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-${PY_SRC%.py}}"
PREV_PY="${PREV_PY:-407Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-408_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-408_kernel_maxd14}"
PREV_CU="${PREV_CU:-407_kernel_maxd14.cu}"
PREV_BIN="${PREV_BIN:-407_kernel_maxd14}"
SASS402_SHA="${SASS402_SHA:-821302f227dcd160}"; SASS402_REAL="${SASS402_REAL:-446}"; SASS402_LOOP="${SASS402_LOOP:-109}"
SASS403_SHA="${SASS403_SHA:-e940cb96e5e65fc4}"; SASS403_REAL="${SASS403_REAL:-445}"; SASS403_LOOP="${SASS403_LOOP:-108}"
RUN_X8="${RUN_X8:-0}"
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
KERNEL_SHA_408="${KERNEL_SHA_408:-ab483e70328c39c6}"
ANCHOR_G22="${ANCHOR_G22:-868287.1}"       # 407 G22 (dispatcher)
ANCHOR_B0="${ANCHOR_B0:-99230.2}"         # 407 B1 (N=21 layout 403, direct)
ANCHOR_D1="${ANCHOR_D1:-98393.0}"         # 408_diag D1 (T2, N=21 layout 403, direct; same kernel as 408)
ANCHOR_B2="${ANCHOR_B2:-99172.0}"         # 407 B2 (407 under NQ_LAYOUT=402)
ANCHOR_X8="${ANCHOR_X8:-868716.5}"        # 407 X8 (N=22 direct)
ANCHOR_G21="${ANCHOR_G21:-99214.4}"       # 407 G21
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
{ grep -q "^# 408 " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 408 ...' note block"
grep -qE '^REV_TAG:str="408"' "$CODE" && pass "source_rev_tag_is_408" || fail "source_rev_tag_is_408" "wrong REV_TAG"
grep -qF 'CRunnerEntry(14,"./408_kernel_maxd14","NQ_EXTRA_CTX=1 NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_points_at_408_keeps_helper_prefix" || fail "source_table_points_at_408_keeps_helper_prefix" "table entry wrong"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=960\b" "$CODE" && pass "source_default_max_blocks_is_960" || fail "source_default_max_blocks_is_960" "default not 960"
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_407Py (removed=3 added=3 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_407Py" "removed=$PR_ added=$PA_, expected 3/3"
else info "py_diff_fingerprint_vs_407Py" "skipped ($PREV_PY absent)"; fi
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
[[ "$KB" == "$KERNEL_SHA_408"* ]] && pass "cu_kernel_region_sha_matches_delivered_408 (${KB:0:16})" || fail "cu_kernel_region_sha" "sha ${KB:0:16} != $KERNEL_SHA_408 -- not the delivered 408 source"
CR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true); CA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
[[ "$CR_" == "11" && "$CA_" == "5" ]] && pass "cu_code_region_fingerprint_vs_${PREV_CU%.cu} (removed=11 added=5)" || fail "cu_code_region_fingerprint" "removed=$CR_ added=$CA_, expected 11/5"
# the search's generator (408_compile_search.sh): this revision is T2
gen_variant() {  # base.cu out.cu ops
python3 - "$1" "$2" "$3" <<'PYEOF'
import sys
base, outp, letters = sys.argv[1:4]
src = open(base, encoding='utf-8').read()
def lit(s, old, new, n, what):
    k = s.count(old)
    if k != n: sys.exit("generator: %s matched %d times, expected %d" % (what, k, n))
    return s.replace(old, new)

# the per-code tables, computed from the OP_* masks of the source (checked against it)
STEP3, ADD1, BL1, BL2, KN3, KN4 = 24, 32, 12, 16, 18, 8
for name, val in (('OP_STEP3_MASK', STEP3), ('OP_ADD1_MASK', ADD1), ('OP_BL1_MASK', BL1),
                  ('OP_BL2_MASK', BL2), ('OP_KN3_MASK', KN3), ('OP_KN4_MASK', KN4)):
    import re
    m = re.search(r'static const uint32_t %s\s*=\s*(\d+)u;' % name, src)
    if not m or int(m.group(1)) != val:
        sys.exit("generator: %s != %d in the source" % (name, val))
TA = TB = T64 = 0
for c in range(8):
    step = 2 + ((STEP3 >> c) & 1)
    lo2  = ((ADD1 >> c) & 1) | ((BL1 >> c) & 1) | (((BL2 >> c) & 1) << 1)
    kt   = (((KN3 >> c) & 1) << 1) | ((KN4 >> c) & 1)
    TA  |= ((lo2 << 2) | step) << (4 * c)
    TB  |= kt << (2 * c)
    T64 |= ((kt << 4) | (lo2 << 2) | step) << (8 * c)
KN4_LINE = 'static const uint32_t OP_KN4_MASK         = 8u;    /* code 3 */\n'

MARK_OLD = ('            uint32_t stepu = 2u + ((OP_STEP3_MASK >> block_code) & 1u);\n'
            '            uint32_t addvu = (OP_ADD1_MASK >> block_code) & 1u;\n'
            '            uint32_t bLiu =\n'
            '                ((OP_BL1_MASK >> block_code) & 1u)\n'
            '                | (((OP_BL2_MASK >> block_code) & 1u) << 1);\n'
            '            uint32_t ktu =\n'
            '                ((OP_KN3_MASK >> block_code) & 1u)\n'
            '                | (((OP_KN4_MASK >> block_code) & 1u) << 1);\n'
            '            uint32_t bKu =\n'
            '                (n3 & (0u - (ktu & 1u))) | (n4 & (0u - (ktu >> 1)));\n'
            '            nld = ((cur_ld | bit) << stepu) | addvu | bLiu;\n'
            '            nrd = ((cur_rd | bit) >> stepu) | bKu;\n')

def T1(s):
    # 408 (T1): the mark's operands depend on block_code only (3 bits): two 32-bit
    # per-code tables. bK = kn3*n3 + kn4*n4 = (2*kn3 + kn4) * n4 since n3 == 2*n4.
    tab = KN4_LINE + (
        '/* 408 (T1): per-code tables. TA nibble(code) = (addv | bLi) << 2 | step ; TB 2 bits(code) = kn3 << 1 | kn4.\n'
        ' * Built from the six OP_* masks above (408_compile_search.sh generator). */\n'
        'static const uint32_t MARK_TA = 0x%08Xu;\n'
        'static const uint32_t MARK_TB = 0x%08Xu;\n' % (TA, TB))
    s = lit(s, KN4_LINE, tab, 1, 'OP_KN4 line')
    new = ('            /* 408 (T1): tables indexed by block_code; bK = kt * n4 (n3 == 2 * n4 on both hosts). */\n'
           '            const uint32_t mta = MARK_TA >> (block_code * 4u);\n'
           '            const uint32_t stepu = mta & 3u;\n'
           '            const uint32_t bKu = ((MARK_TB >> (block_code * 2u)) & 3u) * n4;\n'
           '            nld = ((cur_ld | bit) << stepu) | ((mta >> 2) & 3u);\n'
           '            nrd = ((cur_rd | bit) >> stepu) | bKu;\n')
    return lit(s, MARK_OLD, new, 1, 'mark body')

def T2(s):
    # 408 (T2): one 64-bit per-code table, byte(code) = kt << 4 | lo2 << 2 | step.
    tab = KN4_LINE + (
        '/* 408 (T2): per-code table, byte(code) = (kn3 << 1 | kn4) << 4 | (addv | bLi) << 2 | step.\n'
        ' * Built from the six OP_* masks above (408_compile_search.sh generator). */\n'
        'static const uint64_t MARK_T64 = 0x%016XULL;\n' % T64)
    s = lit(s, KN4_LINE, tab, 1, 'OP_KN4 line')
    new = ('            /* 408 (T2): one table indexed by block_code; bK = kt * n4 (n3 == 2 * n4 on both hosts). */\n'
           '            const uint32_t mt = (uint32_t)(MARK_T64 >> (block_code * 8u));\n'
           '            const uint32_t stepu = mt & 3u;\n'
           '            const uint32_t bKu = ((mt >> 4) & 3u) * n4;\n'
           '            nld = ((cur_ld | bit) << stepu) | ((mt >> 2) & 3u);\n'
           '            nrd = ((cur_rd | bit) >> stepu) | bKu;\n')
    return lit(s, MARK_OLD, new, 1, 'mark body')

def Pu(s):
    # 408 (Pu): the frame is written on every descent; only the increment is conditional.
    # Main-loop push only (the root fast path's push is indented deeper and untouched).
    old_head = ('        if (cur_avail != 0u) {\n'
                '            /* RISK NOTE: literal 1:1 push-guard translation, see file header. */\n')
    new_head = ('        {\n'
                '            /* 408 (Pu): unconditional frame write, conditional increment -- no branch region.\n'
                '             * In bounds: at a descent stack_ptr <= cur_depth <= terminal_depth - 1 <= 12. */\n')
    s = lit(s, old_head, new_head, 1, 'push head')
    old_tail = ('            stack_ptr += 1;\n'
                '        }\n'
                '        cur_ld = nld;\n')
    new_tail = ('            stack_ptr += (cur_avail != 0u) ? 1 : 0;\n'
                '        }\n'
                '        cur_ld = nld;\n')
    return lit(s, old_tail, new_tail, 1, 'push tail')

OPS = {'T1': T1, 'T2': T2, 'Pu': Pu}
for tok in [t for t in letters.split('+') if t and t != '0']:
    if tok not in OPS: sys.exit("generator: unknown op " + tok)
    src = OPS[tok](src)
open(outp, 'w', encoding='utf-8').write(src)
PYEOF
}
if gen_variant "$PREV_CU" "/tmp/${REV}_gen.cu" "T2" 2>"/tmp/${REV}_gen.log"; then
  cu_code_region "/tmp/${REV}_gen.cu" > "/tmp/${REV}_gen_code.cu"
  cmp -s "/tmp/${REV}_gen_code.cu" "/tmp/${REV}_cur_code.cu" && pass "cu_code_region_equals_generator_T2_on_${PREV_CU%.cu}" || fail "cu_code_region_equals_generator" "the delivered .cu is not generator(T2) applied to $PREV_CU (comment-stripped)"
else fail "cu_generator_runs_on_${PREV_CU%.cu}" "$(cat /tmp/${REV}_gen.log)"; fi
NR8=0
cnt() { grep -cF -- "$1" "/tmp/${REV}_cur_code.cu"; }
[[ "$(cnt 'static const uint64_t MARK_T64 = 0x0202062B17062202ULL;')" == "1" ]] && NR8=$((NR8+1))                                 # the table (generator value)
[[ "$(cnt 'const uint32_t mt = (uint32_t)(MARK_T64 >> (block_code * 8u));')" == "1" ]] && NR8=$((NR8+1))                      # one table read in the main loop
[[ "$(cnt 'const uint32_t bKu = ((mt >> 4) & 3u) * n4;')" == "1" ]] && NR8=$((NR8+1))                                           # bK = kt * n4
[[ "$(cnt '(OP_STEP3_MASK >> block_code)')" == "0" ]] && NR8=$((NR8+1))                                                         # old spelling gone from the main loop
[[ "$(cnt '(OP_STEP3_MASK >> pr_block_code)')" == "1" ]] && NR8=$((NR8+1))                                                      # root fast path untouched
[[ "$(cnt 'nf &= ~child_jmark;')" == "1" && "$(cnt 'nld |= child_jmark;')" == "1" ]] && NR8=$((NR8+1))                          # 407 J untouched
[[ "$(cnt 'asm("lop3.b32 %0, %1, %2, %3, 0x01;" : "=r"(nf) : "r"(xb), "r"(b1), "r"(b2));')" == "1" ]] && NR8=$((NR8+1))        # 406's nf arm untouched
[[ "$(cnt 'cur_x = (nld << 1) | (nrd >> 1) | ncol;')" == "1" && "$(cnt 'if ((bm & ~((nld << 1) | (nrd >> 1) | ncol)) == 0u) {')" == "1" ]] && NR8=$((NR8+1))  # cur_x, future check untouched
[[ "$(cnt 'layout = PACK_LAYOUT_403;')" == "2" ]] && NR8=$((NR8+1))
[[ "$NR8" == "9" ]] && pass "cu_408_statements (per-code table in the mark arm; root path, nf arms, cur_x, future check, j-mark, layout selection untouched)" || fail "cu_408_statements" "$NR8 of 9 checks"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"/tmp/${REV}_gcc.log" && pass "cu_cpu_harness_builds" || fail "cu_cpu_harness_builds" "$(head -3 /tmp/${REV}_gcc.log)"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_prev" "$PREV_CU" -lm 2>"/tmp/${REV}_gcc_prev.log" || fail "cu_cpu_harness_prev_builds" "$(head -3 /tmp/${REV}_gcc_prev.log)"
grep -q "rev408 " "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev408 note in header"

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
[[ "$S_SHA" == "$SASS402_SHA" && "$S_REAL" == "$SASS402_REAL" && "$S_LOOP" == "$SASS402_LOOP" ]] && pass "S2_sass_402_is_the_measured_D1 (sha $S_SHA, real $S_REAL, loop $S_LOOP)" || { fail "S2_sass_402_differs_from_D1" "sha $S_SHA real $S_REAL loop $S_LOOP != $SASS402_SHA / $SASS402_REAL / $SASS402_LOOP -- stop and read the SASS"; summary_exit; }
read -r S_SHA S_REAL S_LOOP S_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_dfs_iter_gpu_maxd14ILi403E)"
[[ "$S_SHA" == "$SASS403_SHA" && "$S_REAL" == "$SASS403_REAL" && "$S_LOOP" == "$SASS403_LOOP" ]] && pass "S3_sass_403_is_the_measured_D1 (sha $S_SHA, real $S_REAL, loop $S_LOOP)" || { fail "S3_sass_403_differs_from_D1" "sha $S_SHA real $S_REAL loop $S_LOOP != $SASS403_SHA / $SASS403_REAL / $SASS403_LOOP -- stop and read the SASS"; summary_exit; }

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
cpu_pair N21_l402 21 "$IN21" NQ_LAYOUT=402
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
same_bin() {  # cell refcell-or-file
  local ref="$2"; [[ -f "$ref" ]] || ref="$LOGDIR/bins/${2}_results.bin"
  if [[ -s "$ref" ]] && cmp -s "$ref" "$LOGDIR/bins/${1}_results.bin"; then pass "${1}_per_thread_results_identical ($(stat -c %s "$LOGDIR/bins/${1}_results.bin") bytes vs $ref)"
  else fail "${1}_per_thread_results_DIFFER" "results.bin $1 != $ref -- not adoptable"; ADOPT=0; fi
}
banner "N=21 direct: B0 $PREV_BIN auto(403) / B1 $CU_BIN auto(403) / B2 $CU_BIN NQ_LAYOUT=402"
run_direct B0 "$PREV_BIN" 21 auto 403 || summary_exit
d="$(abspct "${VAL[B0]}" "$ANCHOR_B0")"; le "$d" 0.15 && pass "B0_anchor (${d}% from 407 B1 $ANCHOR_B0)" || info "B0_anchor" "${d}% from $ANCHOR_B0 -- ratios are still read within this session"
sleep "$COOLDOWN"; run_direct B1 "$CU_BIN" 21 auto 403 || summary_exit; same_bin B1 B0
sleep "$COOLDOWN"; run_direct B2 "$CU_BIN" 21 402 402 || summary_exit; same_bin B2 B0
r1="$(ratio "${VAL[B1]}" "${VAL[B0]}")"; OK21=0
if le "$r1" 0.992; then pass "R1_gain_as_stated_at_N21 (B1/B0 = $r1, $(pct "${VAL[B1]}" "${VAL[B0]}")%)"; OK21=1
elif le "$r1" 0.998; then info "R1_gain_smaller_than_stated" "B1/B0 = $r1 (real and adoptable; stated <= 0.992)"; OK21=1
elif ge "$r1" 0.9997; then fail "R1_REFUTED_no_gain_at_N21" "B1/B0 = $r1"
else fail "R1_grey_zone" "B1/B0 = $r1 (stated <= 0.992, adopt <= 0.998, refuted >= 0.9997)"; fi
info "B1_vs_408diag_D1" "$(pct "${VAL[B1]}" "$ANCHOR_D1")% vs $ANCHOR_D1 (same kernel SASS, layout 403)"
info "B2_vs_407_B2" "$(pct "${VAL[B2]}" "$ANCHOR_B2")% vs $ANCHOR_B2 (407 under layout 402) ; B2/B1 = $(ratio "${VAL[B2]}" "${VAL[B1]}")"

X5REF="$ANCHOR_X8"; XLABEL="407 X8 $ANCHOR_X8"; X5BIN="$(ls -t 407_validate_*/bins/X8_results.bin 2>/dev/null | head -1)"
if [[ "$RUN_X8" == "1" ]]; then
  banner "X8: ${PREV_BIN} N=22 direct (same-session anchor)"
  sleep "$COOLDOWN"; run_direct X8 "$PREV_BIN" 22 auto 403 || summary_exit
  X5REF="${VAL[X8]}"; XLABEL="X8 (same session)"; X5BIN="$LOGDIR/bins/X8_results.bin"
fi
banner "X9: ${CU_BIN} N=22 direct"
sleep "$COOLDOWN"; run_direct X9 "$CU_BIN" 22 auto 403 || summary_exit
if [[ -n "${X5BIN:-}" && -s "$X5BIN" ]]; then same_bin X9 "$X5BIN"
else fail "X9_per_thread_results_reference_missing" "no X8 run and no 407_validate_*/bins/X8_results.bin (use RUN_X8=1)"; ADOPT=0; fi
rx="$(ratio "${VAL[X9]}" "$X5REF")"; px="$(pct "${VAL[X9]}" "$X5REF")"; OK22=0
if le "$rx" 0.993; then pass "R2_gain_as_stated_at_N22 (X9 vs $XLABEL = $rx, ${px}%)"; OK22=1
elif le "$rx" 0.998; then info "R2_gain_smaller_than_stated" "X9 vs $XLABEL = $rx (${px}%) (real and adoptable; stated <= 0.993)"; OK22=1
elif ge "$rx" 0.9997; then fail "R2_REFUTED_no_gain_at_N22" "X9 vs $XLABEL = $rx (${px}%)"
else fail "R2_grey_zone" "X9 vs $XLABEL = $rx (${px}%) (stated <= 0.993, adopt <= 0.998, refuted >= 0.9997)"; fi
if [[ "$OK21" != "$OK22" ]]; then info "N21_and_N22_DISAGREE" "N=21 adoptable=$OK21, N=22 adoptable=$OK22 -- registered: reported as NOT adopted, Suzuki decides"; fi
[[ "$OK21" == "1" && "$OK22" == "1" ]] || ADOPT=0

if [[ "$SKIP_G" != "1" ]]; then
  banner "G21: ./$PY_BIN -g 21 21 (production path)"
  sleep "$COOLDOWN"; run_g G21 21 403 || summary_exit
  info "G21_vs_407_G21" "$(pct "${VAL[G21]}" "$ANCHOR_G21")% vs $ANCHOR_G21 ; vs B1 $(pct "${VAL[G21]}" "${VAL[B1]}")%"
  banner "G22: ./$PY_BIN -g 22 22 (production path)"
  sleep "$COOLDOWN"; run_g G22 22 403 || summary_exit
  pass "R3_G21_G22_oracle_layout403"
  info "G22_vs_407_G22" "$(pct "${VAL[G22]}" "$ANCHOR_G22")% vs $ANCHOR_G22 (407 G22, dispatcher)"
fi
CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "R4_sm_clock_1710_all_cells" || { fail "R4_sm_clock_1710_all_cells" "see clk_*.tsv"; ADOPT=0; }
if [[ "$ADOPT" == "1" ]]; then
  info "ADOPTION" "408 is the production binary/.py: N=21 = ${VAL[G21]:-<G21 skipped: run it before updating the README>} ms, N=22 = ${VAL[G22]:-<G22 skipped>} ms"
else
  info "ADOPTION" "NOT adopted (or Suzuki decides) -- production stays 407. See FAIL / INFO lines."
fi

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
ref={r['cell']:float(r['kernel_ms']) for r in rows}
print(f"{'cell':<5}{'binary':<32}{'N':>3}{'lay':>5}{'help':>5}{'free_mb':>8}{'kernel_ms':>14}{'vs ref':>10}{'match':>6}{'sm':>6}{'h:mm:ss':>10}")
for r in rows:
    v=float(r['kernel_ms']); s=v/1000; c=r['cell']
    base=ref.get('B0') if c.startswith('B') else (ref.get('X8') or 868716.5) if c.startswith('X') else 99214.4 if c=='G21' else 868287.1
    print(f"{c:<5}{r['binary'][:31]:<32}{r['N']:>3}{r['layout']:>5}{r['helpers']:>5}{r['free_mb']:>8}{v:>14.3f}{(v/base-1)*100:>+9.3f}%{r['match']:>6}{r['sm_mean']:>6}{int(s//3600):>4}:{int(s%3600//60):02d}:{s%60:04.1f}")
print("(B* vs B0, X9 vs X8 or 407 X8 868,717, G21 vs 407 G21 99,214, G22 vs 407 G22 868,287)")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd14" --exclude="*_cpu" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
