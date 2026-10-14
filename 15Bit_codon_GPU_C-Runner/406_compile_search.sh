#!/usr/bin/env bash
# 406_compile_search.sh
#
# rev406 search -- COMPILE + CPU ONLY, NO GPU RUN. Base: 405_kernel_maxd14.cu.
#
# 405 was adopted with the instruction count unchanged: what moved was the
# depth of the nf == 0 branch. Static read of 405's <403> step (no mark):
#     s1  -avail                      IMAD.MOV
#     s2  bit = avail & -avail        LOP3
#     s3  ld|bit, rd|bit, col|bit     LOP3 x3
#     s4  << 1, >> 1                  IMAD.SHL, SHF
#     s5  nf = ~(nld|nrd|ncol)        LOP3 0x01
#     s6  nf != 0                     ISETP          s7  BRA
# s3 + s4 can be one stage: shifts distribute over OR, so with a carried
#     cur_x = (ld << 1) | (rd >> 1) | col        (changes only on descent / pop)
#     nf    = ~((cur_x | bit) | (bit << 1) | (bit >> 1))
# nf is two stages after bit instead of three. The price: cur_x is rebuilt on
# every descent and every pop (2 shifts + 1 LOP3 each).
#
#   H1  carried cur_x; nld / nrd / ncol stay where they are; the two arms
#       (mark / no mark) each form nf; one shared nf == 0 test.
#   H2  carried cur_x; the no-mark arm tests nf BEFORE forming nld / nrd
#       (a dead end never pays for them); the mark arm keeps its own test.
#       This changes the control-flow shape of the step (two exits).
#
# PATH PROFILE (new, CPU, single thread): the same source with counters,
# run on the first NREC_PROF real records of N=21 and N=22. It answers what
# the static read cannot: how many steps are dead ends, how many descend,
# how many are mark steps, pops per step. The algorithm is identical in all
# variants, so one profile weighs them all. It is NOT a timing.
#
#  Output per variant and instantiation: frame / spill / registers, SASS sha,
#  REAL instructions, hot loop, push, pop, rest, opcode-multiset diff vs v0,
#  loop listings. CPU equality vs v0: N=21 auto, N=21 l402, N=22 auto.
#  Hard check: v0 must reproduce 405's SASS (530de5ce / 3fd01fac).
#
# USAGE
#   bash 406_compile_search.sh            # a few minutes, GPU idle

set -u

REV="406_search"
BASE_CU="${BASE_CU:-405_kernel_maxd14.cu}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
NREC="${NREC:-1024}"
VARIANTS="${VARIANTS:-0 H1 H2}"
NREC_PROF="${NREC_PROF:-4096}"
BASE402_SHA="${BASE402_SHA:-530de5ce72cf920c}"
BASE403_SHA="${BASE403_SHA:-3fd01fac2e9aedd3}"
SKIP_CPU="${SKIP_CPU:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_${TS}}"
TSV="$LOGDIR/${REV}_results.tsv"
PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }

[[ -f "$BASE_CU" ]] && pass "file_present[$BASE_CU]" || { fail "file_present[$BASE_CU]" "missing"; exit 1; }
[[ ! -x "$NVCC" ]] && NVCC="nvcc"; [[ ! -x "$CUOBJDUMP" ]] && CUOBJDUMP="cuobjdump"
mkdir -p "$LOGDIR/src" "$LOGDIR/sass" "$LOGDIR/loops" "$LOGDIR/logs" "$LOGDIR/bin"

# ---------------------------------------------------------------------
# generator: ops joined by '+', applied left to right to the base .cu
# ---------------------------------------------------------------------
gen_variant() {  # base.cu out.cu ops
python3 - "$1" "$2" "$3" <<'PYEOF'
import sys, re
base, outp, letters = sys.argv[1:4]
src = open(base, encoding='utf-8').read()
def lit(s, old, new, n, what):
    k = s.count(old)
    if k != n: sys.exit("generator: %s matched %d times, expected %d" % (what, k, n))
    return s.replace(old, new)
LOOP_HEAD = ('#ifndef __CUDACC__\n    uint64_t debug_iter_count = 0;\n    const uint64_t DEBUG_ITER_CAP = 50000000ULL;\n#endif\n    for (;;) {\n')
STEP_OLD = ('        uint32_t nld = (cur_ld | bit) << 1;\n'
            '        uint32_t nrd = (cur_rd | bit) >> 1;\n'
            '        uint32_t ncol = cur_col | bit;\n'
            '        if (block_code != 0u) {\n')
MARK_TAIL = ('            nld = ((cur_ld | bit) << stepu) | addvu | bLiu;\n'
             '            nrd = ((cur_rd | bit) >> stepu) | bKu;\n'
             '        }\n')
CMT = ('        /* 405 (F): 0x01 = ~(a | b | c). ncol holds ~bm, so this IS bm & ~(nld | nrd | ncol):\n'
       '         * one LOP3 instead of OR3 / NOT / AND. ptxas does not fuse the C spelling (405 search, Fc). */\n')
NF_OLD = (CMT + '        uint32_t nf;\n'
          '#ifdef __CUDACC__\n'
          '        asm("lop3.b32 %0, %1, %2, %3, 0x01;" : "=r"(nf) : "r"(nld), "r"(nrd), "r"(ncol));\n'
          '#else\n'
          '        nf = ~(nld | nrd | ncol);\n'
          '#endif\n'
          '        if (nf == 0u) {\n'
          '            continue;\n'
          '        }\n')
LOP = ('#ifdef __CUDACC__\n'
       '%(i)sasm("lop3.b32 %%0, %%1, %%2, %%3, 0x01;" : "=r"(nf) : "r"(%(a)s), "r"(%(b)s), "r"(%(c)s));\n'
       '#else\n'
       '%(i)snf = ~(%(a)s | %(b)s | %(c)s);\n'
       '#endif\n')
def X(s):
    # carried state cur_x = (ld << 1) | (rd >> 1) | col  (col already holds ~bm). On a step WITHOUT a mark,
    #   nf = ~(((ld|bit) << 1) | ((rd|bit) >> 1) | col | bit) = ~((cur_x | bit) | (bit << 1) | (bit >> 1))
    # so nf is two stages after bit (OR / shifts, then LOP3) instead of three (OR, shift, LOP3).
    s = lit(s, LOOP_HEAD, '    uint32_t cur_x = (cur_ld << 1) | (cur_rd >> 1) | cur_col;\n' + LOOP_HEAD, 1, 'loop head')
    s = lit(s, '             * 405 (F): taken from the masked field inside each layout branch above. */\n            continue;\n',
               '             * 405 (F): taken from the masked field inside each layout branch above. */\n'
               '            cur_x = (cur_ld << 1) | (cur_rd >> 1) | cur_col;\n            continue;\n', 1, 'pop tail')
    s = lit(s, '        cur_ld = nld;\n        cur_rd = nrd;\n        cur_col = ncol;\n        cur_avail = nf;\n',
               '        cur_ld = nld;\n        cur_rd = nrd;\n        cur_col = ncol;\n        cur_x = (nld << 1) | (nrd >> 1) | ncol;\n        cur_avail = nf;\n', 1, 'descent')
    return s
def H1(s):   # both arms compute nf, one shared nf == 0 test; nld/nrd/ncol stay where they are
    s = X(s)
    k = s.count(MARK_TAIL + NF_OLD)
    if k != 1: sys.exit("generator: mark tail + nf matched %d times" % k)
    new = (MARK_TAIL + CMT + '        uint32_t nf;\n        if (block_code != 0u) {\n'
           + LOP % dict(i='            ', a='nld', b='nrd', c='ncol')
           + '        } else {\n            const uint32_t xb = cur_x | bit, b1 = bit << 1, b2 = bit >> 1;\n'
           + LOP % dict(i='            ', a='xb', b='b1', c='b2')
           + '        }\n        if (nf == 0u) {\n            continue;\n        }\n')
    return s.replace(MARK_TAIL + NF_OLD, new)
def H2(s):   # no-mark arm tests nf BEFORE nld/nrd are formed; the mark arm keeps its own test
    s = X(s)
    a = s.index(STEP_OLD); b = s.index(MARK_TAIL + NF_OLD)
    if s.count(STEP_OLD) != 1 or s.count(MARK_TAIL + NF_OLD) != 1: sys.exit("generator: H2 anchors")
    mark_body = s[a + len(STEP_OLD): b]
    new = ('        uint32_t nld, nrd, nf;\n'
           '        const uint32_t ncol = cur_col | bit;\n'
           '        if (block_code == 0u) {\n'
           '            const uint32_t xb = cur_x | bit, b1 = bit << 1, b2 = bit >> 1;\n'
           + LOP % dict(i='            ', a='xb', b='b1', c='b2') +
           '            if (nf == 0u) {\n                continue;\n            }\n'
           '            nld = (cur_ld | bit) << 1;\n'
           '            nrd = (cur_rd | bit) >> 1;\n'
           '        } else {\n'
           + mark_body +
           '            nld = ((cur_ld | bit) << stepu) | addvu | bLiu;\n'
           '            nrd = ((cur_rd | bit) >> stepu) | bKu;\n'
           + LOP % dict(i='            ', a='nld', b='nrd', c='ncol') +
           '            if (nf == 0u) {\n                continue;\n            }\n'
           '        }\n')
    return s[:a] + new + s[b + len(MARK_TAIL + NF_OLD):]
PROF = ['pop', 'step', 'mark_step', 'dead_nf', 'fc_tested', 'fc_dead', 'terminal', 'jmark_tested', 'jmark_dead', 'descend', 'push', 'task_end']
def Z(s):    # CPU-only path counters (single thread). The GPU build is untouched by the macro.
    decl = ('#define LAYOUT_IF(c)      if (c)\n#endif\n'
            '#ifndef __CUDACC__\n#include <stdlib.h>\nstatic unsigned long long nq_prof[16];\n#define NQP(i) (nq_prof[i]++)\n'
            'static void nq_prof_dump(void) {\n    static const char *nm[] = {' + ', '.join('"%s"' % n for n in PROF) + '};\n'
            '    int i; for (i = 0; i < %d; ++i) fprintf(stderr, "[nq-prof] %%s %%llu\\n", nm[i], nq_prof[i]);\n}\n' % len(PROF) +
            '__attribute__((constructor)) static void nq_prof_init(void) { atexit(nq_prof_dump); }\n#else\n#define NQP(i)\n#endif\n')
    s = lit(s, '#define LAYOUT_IF(c)      if (c)\n#endif\n', decl, 1, 'prof decl')
    s = lit(s, '            if (stack_ptr == 0) {\n                break;\n            }\n            stack_ptr -= 1;\n',
               '            if (stack_ptr == 0) {\n                NQP(11);\n                break;\n            }\n            NQP(0);\n            stack_ptr -= 1;\n', 1, 'pop')
    s = lit(s, '        const uint32_t nibble_op  = (uint32_t)(schedule64 >> (cur_depth * 4)) & 15u;\n',
               '        NQP(1);\n        const uint32_t nibble_op  = (uint32_t)(schedule64 >> (cur_depth * 4)) & 15u;\n', 1, 'step')
    s = lit(s, '        if (block_code != 0u) {\n            uint32_t stepu', '        if (block_code != 0u) {\n            NQP(2);\n            uint32_t stepu', 1, 'mark')
    s = lit(s, '#endif\n        if (nf == 0u) {\n            continue;\n', '#endif\n        if (nf == 0u) {\n            NQP(3);\n            continue;\n', 1, 'dead')
    s = lit(s, '            if (((future_check_mask >> cur_depth) & 1u) != 0u) {\n                if ((bm & ~((nld << 1) | (nrd >> 1) | ncol)) == 0u) {\n',
               '            if (((future_check_mask >> cur_depth) & 1u) != 0u) {\n                NQP(4);\n                if ((bm & ~((nld << 1) | (nrd >> 1) | ncol)) == 0u) {\n                    NQP(5);\n', 1, 'fc')
    s = lit(s, '        if (cur_depth == terminal_depth) {\n            if (terminal_base14 == 0u) {', '        if (cur_depth == terminal_depth) {\n            NQP(6);\n            if (terminal_base14 == 0u) {', 1, 'terminal')
    s = lit(s, '        if (child_jmark != 0u) {\n            nf &= ~1u;\n            if (nf == 0u) {\n',
               '        if (child_jmark != 0u) {\n            NQP(7);\n            nf &= ~1u;\n            if (nf == 0u) {\n                NQP(8);\n', 1, 'jmark')
    s = lit(s, '        int next_depth = cur_depth + 1;\n        if (cur_avail != 0u) {\n', '        NQP(9);\n        int next_depth = cur_depth + 1;\n        if (cur_avail != 0u) {\n            NQP(10);\n', 1, 'descend')
    return s
OPS = {'H1': H1, 'H2': H2, 'Z': Z}
for tok in [t for t in letters.split('+') if t and t != '0']:
    if tok not in OPS: sys.exit("generator: unknown op " + tok)
    src = OPS[tok](src)
open(outp, 'w', encoding='utf-8').write(src)
PYEOF
}

# ---------------------------------------------------------------------
# SASS statistics: sha, real, loop, push, pop ; writes the loop listing
# ---------------------------------------------------------------------
sass_stat() {  # sass.txt function-fragment loop_out.txt  ->  "sha real loop push pop total"
python3 - "$1" "$2" "$3" <<'PYEOF'
import sys, re, hashlib
out = {}; cur = None
for line in open(sys.argv[1], errors='replace'):
    m = re.match(r'\s*Function\s*:\s*(\S+)', line)
    if m: cur = m.group(1); out[cur] = []; continue
    if cur is None: continue
    m = re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s*(.*?);\s*(/\*.*\*/)?\s*$', line)
    if m: out[cur].append((int(m.group(1), 16), m.group(2).strip()))
k = [k for k in out if sys.argv[2] in k]
if len(k) != 1: print("CANNOT_LOCATE -1 -1 -1 -1 -1"); sys.exit(0)
I = out[k[0]]
real = sum(1 for _, s in I if not s.startswith('NOP'))
best = None   # hot loop = smallest backward-BRA span containing an LDL.64
for a, s in I:
    m = re.match(r'(@!?P\d\s+)?BRA\s+(0x[0-9a-f]+)$', s)
    if not m: continue
    t = int(m.group(2), 16)
    if t >= a: continue
    b = [x for x in I if t <= x[0] <= a]
    if any(x[1].startswith('LDL.64') for x in b) and (best is None or len(b) < len(best)): best = b
sha = hashlib.sha256('\n'.join(s for _, s in I).encode()).hexdigest()[:16]
if not best: print(sha, real, -1, -1, -1, len(I)); sys.exit(0)
L = best
def ctrl(s): return re.match(r'(@!?P\d\s+)?(BRA|BSYNC|BSSY|BREAK)\b', s) is not None
push = pop = -1
st = [i for i, (_, s) in enumerate(L) if s.startswith('STL')]
if st:   # push = straight-line run that ends at the last store
    i = st[0]
    while i > 0 and not ctrl(L[i-1][1]): i -= 1
    push = st[-1] - i + 1
ld = [i for i, (_, s) in enumerate(L) if s.startswith('LDL')]
if ld:   # pop = from the stack-empty test (the run of control instructions before the loads) to the last straight-line instruction
    i = ld[0]
    while i > 0 and not ctrl(L[i-1][1]): i -= 1
    while i > 0 and ctrl(L[i-1][1]): i -= 1
    while i > 0 and not ctrl(L[i-1][1]): i -= 1
    j = ld[-1]
    while j + 1 < len(L) and not ctrl(L[j+1][1]): j += 1
    pop = j - i + 1
with open(sys.argv[3], 'w') as f:
    for a, s in L: f.write("%04x  %s\n" % (a, s))
print(sha, real, len(L), push, pop, len(I))
PYEOF
}
loop_diff() {  # v0_loop.txt v_loop.txt -> opcode-multiset difference (one line)
python3 - "$1" "$2" <<'PYEOF'
import sys, re, collections
def ops(p):
    c = collections.Counter()
    for line in open(p):
        s = line.split(None, 1)[1].strip()
        s = re.sub(r'^@!?P\d\s+', '', s)
        op = s.split()[0]
        if op.startswith('LOP3'):
            m = re.search(r'(0x[0-9a-f]+), !PT$', s); op = 'LOP3:' + (m.group(1) if m else '?')
        c[op] += 1
    return c
a, b = ops(sys.argv[1]), ops(sys.argv[2])
d = ["%s%+d" % (k, b[k] - a[k]) for k in sorted(set(a) | set(b)) if a[k] != b[k]]
print(' '.join(d) if d else '(same multiset)')
PYEOF
}

find_input() { local n="$1" c; for lg in $(ls -t 40*_crunner_logs/crunner_*_N${n}.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { echo "$c"; return; }; done; ls -t constellations_N${n}_*.sched394f.bin 2>/dev/null | head -1; }
IN21=""; IN22=""
if [[ "$SKIP_CPU" != "1" ]]; then
  IN21="$(find_input 21)"; IN22="$(find_input 22)"
  [[ -f "${IN21:-/nonexistent}" ]] && pass "input21_located ($IN21)" || { fail "input21_located" "no N=21 sched input (SKIP_CPU=1 to compile only)"; exit 1; }
  [[ -f "${IN22:-/nonexistent}" ]] && pass "input22_located ($IN22)" || { fail "input22_located" "no N=22 sched input (SKIP_CPU=1 to compile only)"; exit 1; }
fi
cpu_run() {  # bin label N input layout -> writes $LOGDIR/bin/<label>.bin ; rc
  NQ_LAYOUT="$5" "$1" "$3" "$4" "$LOGDIR/bin/$2.bin" "$NREC" > "$LOGDIR/logs/cpu_$2.log" 2>&1
}

printf 'variant\tinst\tframe\tspill\tregs\tsha\treal\tloop\tpush\tpop\trest\tcpu\tloop_diff_vs_v0\n' > "$TSV"
echo; echo "base: $BASE_CU   variants: $VARIANTS"
for V in $VARIANTS; do
  NAME="v_$(echo "$V" | tr -d '+')"; SRC="$LOGDIR/src/${NAME}.cu"; BIN="$LOGDIR/bin/${NAME}"
  echo; echo "---- $V ----"
  if ! gen_variant "$BASE_CU" "$SRC" "$V" 2> "$LOGDIR/logs/gen_${NAME}.log"; then
    fail "generate[$V]" "$(cat "$LOGDIR/logs/gen_${NAME}.log")"; printf '%s\t-\t-\t-\t-\tGEN_FAILED\t-\t-\t-\t-\t-\t-\t-\n' "$V" >> "$TSV"; continue; fi
  "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$BIN" "$SRC" -lcuda > "$LOGDIR/logs/nvcc_${NAME}.log" 2>&1
  if [[ ! -x "$BIN" ]]; then
    fail "nvcc[$V]" "$(grep -i 'error' "$LOGDIR/logs/nvcc_${NAME}.log" | head -2 | tr '\n' ' ')"; printf '%s\t-\t-\t-\t-\tNVCC_FAILED\t-\t-\t-\t-\t-\t-\t-\n' "$V" >> "$TSV"; continue; fi
  "$CUOBJDUMP" -sass "$BIN" > "$LOGDIR/sass/${NAME}.txt" 2>&1
  rm -f "$BIN"
  # CPU equivalence vs v0 (per-record results)
  CPU="-"
  if [[ "$SKIP_CPU" != "1" ]]; then
    if "$GCC" -O2 -fopenmp -x c -o "$LOGDIR/bin/${NAME}_cpu" "$SRC" -lm 2> "$LOGDIR/logs/gcc_${NAME}.log"; then
      CPU="ok"
      for C in "N21_auto 21 $IN21 auto" "N21_l402 21 $IN21 402" "N22_auto 22 $IN22 auto"; do
        set -- $C
        cpu_run "$LOGDIR/bin/${NAME}_cpu" "${NAME}_$1" "$2" "$3" "$4"; rc=$?
        if [[ "$rc" != "0" || ! -s "$LOGDIR/bin/${NAME}_$1.bin" ]]; then CPU="rc=$rc[$1]"; break; fi
        if [[ "$V" != "0" ]]; then
          if [[ ! -s "$LOGDIR/bin/v_0_$1.bin" ]]; then CPU="no_v0_ref"; break; fi
          cmp -s "$LOGDIR/bin/v_0_$1.bin" "$LOGDIR/bin/${NAME}_$1.bin" || { CPU="DIFF[$1]"; break; }
        fi
      done
      rm -f "$LOGDIR/bin/${NAME}_cpu"
    else CPU="gcc_failed"; fi
    [[ "$CPU" == "ok" ]] && pass "cpu_equal_to_v0[$V] (3 cases x $NREC records)" || fail "cpu_equal_to_v0[$V]" "$CPU"
  fi
  for L in 402 403; do
    FR="$(grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${L}E" "$LOGDIR/logs/nvcc_${NAME}.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"
    SP="$(grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${L}E" "$LOGDIR/logs/nvcc_${NAME}.log" | grep -o '[0-9]* bytes spill stores' | head -1 | cut -d' ' -f1)"
    RG="$(grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${L}E" "$LOGDIR/logs/nvcc_${NAME}.log" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"
    read -r S_SHA S_REAL S_LOOP S_PUSH S_POP S_TOT <<< "$(sass_stat "$LOGDIR/sass/${NAME}.txt" "kernel_dfs_iter_gpu_maxd14ILi${L}E" "$LOGDIR/loops/${NAME}_${L}.txt")"
    REST="-"; [[ "$S_LOOP" -gt 0 && "$S_PUSH" -ge 0 && "$S_POP" -ge 0 ]] && REST=$((S_LOOP - S_PUSH - S_POP))
    LD="-"; [[ -s "$LOGDIR/loops/v_0_${L}.txt" && -s "$LOGDIR/loops/${NAME}_${L}.txt" ]] && LD="$(loop_diff "$LOGDIR/loops/v_0_${L}.txt" "$LOGDIR/loops/${NAME}_${L}.txt")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$V" "$L" "${FR:-?}" "${SP:-?}" "${RG:-?}" "$S_SHA" "$S_REAL" "$S_LOOP" "$S_PUSH" "$S_POP" "$REST" "$CPU" "$LD" >> "$TSV"
    if [[ "$V" == "0" ]]; then
      WANT="$BASE402_SHA"; [[ "$L" == "403" ]] && WANT="$BASE403_SHA"
      [[ "$S_SHA" == "$WANT" ]] && pass "v0_reproduces_405_sass_<$L> (sha $S_SHA, real $S_REAL, loop $S_LOOP, push $S_PUSH, pop $S_POP)" \
        || fail "v0_reproduces_405_sass_<$L>" "sha $S_SHA != $WANT -- toolchain or base differs; the table is not comparable to 405"
    fi
    info "$V <$L>" "frame=${FR:-?} spill=${SP:-?} regs=${RG:-?} sha=$S_SHA real=$S_REAL loop=$S_LOOP push=$S_PUSH pop=$S_POP rest=$REST"
  done
done

# ---------------------------------------------------------------------
# path profile (CPU, single thread, counters only)
# ---------------------------------------------------------------------
if [[ "$SKIP_CPU" != "1" ]]; then
  echo; echo "---- path profile (first $NREC_PROF records, single thread) ----"
  if gen_variant "$BASE_CU" "$LOGDIR/src/v_prof.cu" "Z" 2> "$LOGDIR/logs/gen_prof.log" && "$GCC" -O2 -fopenmp -x c -o "$LOGDIR/bin/v_prof_cpu" "$LOGDIR/src/v_prof.cu" -lm 2> "$LOGDIR/logs/gcc_prof.log"; then
    for C in "21 $IN21" "22 $IN22"; do
      set -- $C
      OMP_NUM_THREADS=1 NQ_LAYOUT=auto "$LOGDIR/bin/v_prof_cpu" "$1" "$2" "$LOGDIR/bin/prof_$1.bin" "$NREC_PROF" > "$LOGDIR/logs/prof_N$1.log" 2>&1
      if grep -q '^\[nq-prof\] step' "$LOGDIR/logs/prof_N$1.log"; then pass "path_profile[N=$1]"; else fail "path_profile[N=$1]" "no counters (see logs/prof_N$1.log)"; fi
    done
    python3 - "$LOGDIR/logs/prof_N21.log" "$LOGDIR/logs/prof_N22.log" <<'EOF' | tee "$LOGDIR/8_profile.txt"
import sys, re
cols = []
for p in sys.argv[1:]:
    d = {}
    try:
        for line in open(p, errors='replace'):
            m = re.match(r'\[nq-prof\] (\S+) (\d+)', line)
            if m: d[m.group(1)] = int(m.group(2))
    except OSError: pass
    cols.append(d)
names = ['step', 'dead_nf', 'fc_tested', 'fc_dead', 'terminal', 'jmark_tested', 'jmark_dead', 'descend', 'push', 'pop', 'mark_step', 'task_end']
print("%-14s %16s %9s   %16s %9s" % ('event', 'N=21', '/step', 'N=22', '/step'))
for n in names:
    row = "%-14s" % n
    for d in cols:
        s = d.get('step', 0); v = d.get(n, 0)
        row += " %16d %9s  " % (v, ("%.4f" % (v / s)) if s else '-')
    print(row)
for i, d in enumerate(cols):
    s = d.get('step', 0)
    if s: print("N=%d: iterations of the loop = step + pop + task_end = %d ; steps per task = %.0f" % (21 + i, s + d.get('pop', 0) + d.get('task_end', 0), s / max(1, d.get('task_end', 1))))
print("(dead_nf = nf == 0 ; descend = reached the state update ; push = descend with siblings left ; counts, not time)")
EOF
    rm -f "$LOGDIR/bin/v_prof_cpu"
  else fail "path_profile_build" "$(cat "$LOGDIR/logs/gen_prof.log" "$LOGDIR/logs/gcc_prof.log" 2>/dev/null | head -3)"; fi
fi

echo; echo "======================================================"; echo "Results (real = NOPs excluded; rest = loop - push - pop)"; echo "======================================================"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_table.txt"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
ref = {r['inst']: r for r in rows if r['variant'] == '0'}
for inst in ('403', '402'):
    print("<%s>" % inst)
    print("  %-7s %5s %5s %4s  %-16s %5s %5s %5s %5s %5s %6s  %-6s %s" % ('variant', 'frame', 'spill', 'regs', 'sha', 'real', 'loop', 'push', 'pop', 'rest', 'dloop', 'cpu', 'loop opcode diff vs v0'))
    for r in rows:
        if r['inst'] != inst: continue
        try: d = "%+d" % (int(r['loop']) - int(ref[inst]['loop']))
        except Exception: d = '?'
        print("  %-7s %5s %5s %4s  %-16s %5s %5s %5s %5s %5s %6s  %-6s %s" % (r['variant'], r['frame'], r['spill'], r['regs'], r['sha'], r['real'], r['loop'], r['push'], r['pop'], r['rest'], d, r['cpu'], r['loop_diff_vs_v0']))
for r in rows:
    if r['inst'] == '-': print("  %-7s %s" % (r['variant'], r['sha']))
print("(405 with this tool: <403> 466 / 130 / 12 / 16 / 102)")
EOF
rm -f "$LOGDIR"/bin/*.bin; rmdir "$LOGDIR/bin" 2>/dev/null
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL] ($(stat -c %s "$TARBALL") bytes)" || info "tarball_created" "tar failed"
command -v zip >/dev/null 2>&1 && zip -qr "${LOGDIR}.zip" "$LOGDIR" && info "zip_created" "${LOGDIR}.zip"
echo "results: $TSV"; echo "tarball: $TARBALL"
echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
exit 0
