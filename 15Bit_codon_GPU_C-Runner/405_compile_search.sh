#!/usr/bin/env bash
# 405_compile_search.sh
#
# rev405 spelling search -- COMPILE + CPU EQUIVALENCE ONLY, NO GPU RUN.
# Base: 404_r3_kernel_maxd14.cu (production). Every variant is generated from
# it by the embedded generator; nothing here is a production candidate yet.
#
# Two questions, both from the 2026-10-05 static read of the <403> hot loop
# (464 real / loop 130 / push 12 / pop 15):
#
#  P  Stack bookkeeping. push and pop each spend 3 instructions on
#     IMAD (addr A) + IMAD (addr B) + IADD3 (stack_ptr). With two running
#     pointers (sp_a: 8-byte stride, sp_b: 4-byte stride) that is 2 pointer
#     adds: push -1, pop -1. The compiler may turn the pointers back into an
#     index, or ptxas may do 64-bit pointer arithmetic (worse) -- hence a search.
#       Pa  empty test `sp_a == stack_a`     Pb  empty test `sp_b == stack_b`
#     Does not touch the dependency chain of the DFS state.
#
#  nf The step computes nf = bm & ~(nld | nrd | ncol) in THREE instructions:
#       1600 LOP3 0xfe (3-input OR) / 1620 LOP3 0x33 (NOT) / 1630 LOP3 0xc0 (& bm)
#     ptxas does not fold the NOT into either neighbour here (it does fold it
#     in the future check at 16e0, LUT 0x0c). This sits on the loop-carried
#     chain avail -> bit -> nld/nrd -> nf -> avail.
#       E1  inline-PTX lop3 0x0C:  nf = ~or3 & bm           (3 -> 2, chain -1)
#       E2  inline-PTX lop3 0x02:  nf = ~(nld|nrd) & ~ncol & bm   (3 -> 2)
#       Fa  col carries the off-board bits (cur_col | ~bm) as state, so
#           nf = ~(nld | nrd | ncol) is ONE lop3 0x01 (3 -> 1, chain -2).
#           pop ORs ~bm back into col, push masks col to its field; depth_base
#           becomes popc(root_col | ~bm). Every other value is unchanged.
#       Fc  same as Fa with the plain C spelling (does ptxas fuse it itself?)
#     E/F change the dependency chain: by the standing rule their effect is
#     NOT predictable from the instruction count; they would be measured.
#
#  Variants: v0 (control) Pa Pb E1 E2 Fa Fc Pa+E1 Pa+Fa
#
#  Per variant and per instantiation (<402>, <403>): frame / spill / registers,
#  SASS sha, REAL instructions (alignment NOPs excluded), hot loop, push, pop,
#  rest; opcode-multiset diff of the loop vs v0; loop listings saved.
#  CPU equivalence vs v0 (gcc, same process_one_task): N=21 auto(403),
#  N=21 NQ_LAYOUT=402, N=22 auto, NREC real records each.
#  Hard check: v0 must reproduce 404_r3's SASS (ab5608e9 / 7c67c3aa), else the
#  toolchain differs and the table is not comparable.
#
# USAGE
#   bash 405_compile_search.sh            # a few minutes, GPU idle
#   VARIANTS="0 Pa Fa" bash 405_compile_search.sh

set -u

REV="405_search"
BASE_CU="${BASE_CU:-404_r3_kernel_maxd14.cu}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
NREC="${NREC:-1024}"
VARIANTS="${VARIANTS:-0 Pa Pb E1 E2 Fa Fc Pa+E1 Pa+Fa}"
BASE402_SHA="${BASE402_SHA:-ab5608e947e586bc}"
BASE403_SHA="${BASE403_SHA:-7c67c3aa44cb7a4d}"
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
def sub(s, pat, rep, n, what):
    out, k = re.subn(pat, rep, s, flags=re.M)
    if k != n: sys.exit("generator: %s matched %d times, expected %d" % (what, k, n))
    return out
def lit(s, old, new, n, what):
    k = s.count(old)
    if k != n: sys.exit("generator: %s matched %d times, expected %d" % (what, k, n))
    return s.replace(old, new)
def P(s, test):   # two stack pointers instead of one index; empty test on sp_a ('a') or sp_b ('b')
    s = sub(s, r'^    int      stack_ptr = 0;[^\n]*\n', '', 1, 'stack_ptr decl')
    s = lit(s, '    uint32_t stack_b[MAXD14_ANCESTOR];\n',
               '    uint32_t stack_b[MAXD14_ANCESTOR];\n    uint64_t *sp_a = stack_a;\n    uint32_t *sp_b = stack_b;\n', 1, 'stack_b decl')
    s = lit(s, 'stack_a[stack_ptr] = ', '*sp_a = ', 4, 'push A')
    s = lit(s, 'stack_b[stack_ptr] = ', '*sp_b = ', 4, 'push B')
    s = sub(s, r'^([ ]+)stack_ptr \+= 1;\n', r'\1sp_a += 1;\n\1sp_b += 1;\n', 2, 'stack_ptr += 1')
    s = lit(s, 'if (stack_ptr == 0) {', 'if (sp_a == stack_a) {' if test == 'a' else 'if (sp_b == stack_b) {', 1, 'empty test')
    s = sub(s, r'^([ ]+)stack_ptr -= 1;\n', r'\1sp_a -= 1;\n\1sp_b -= 1;\n', 1, 'stack_ptr -= 1')
    s = sub(s, r'^([ ]+)uint64_t packed_a = stack_a\[stack_ptr\];', r'\1uint64_t packed_a = *sp_a;', 1, 'pop A')
    s = sub(s, r'^([ ]+)(cur_rd  = |uint32_t packed_b = )stack_b\[stack_ptr\];', r'\1\2*sp_b;', 2, 'pop B')
    # CPU-only debug code (#ifndef __CUDACC__)
    s = lit(s, 'if (stack_ptr >= MAXD14_ANCESTOR) {', 'if ((int)(sp_b - stack_b) >= MAXD14_ANCESTOR) {', 2, 'overflow guard')
    s = lit(s, '(long long)debug_idx, stack_ptr, cur_depth,', '(long long)debug_idx, (int)(sp_b - stack_b), cur_depth,', 2, 'overflow print')
    s = lit(s, '(long long)debug_idx, stack_ptr, (unsigned)stack_ptr, cur_depth,',
               '(long long)debug_idx, (int)(sp_b - stack_b), (unsigned)(sp_b - stack_b), cur_depth,', 1, 'iter-cap print')
    return s
NF_OLD = '        uint32_t nf = bm & ~(nld | nrd | ncol);\n'
def E(s, form):   # nf's NOT + AND as one inline-PTX LOP3 (state unchanged)
    if form == 1:
        new = ('        uint32_t nf;\n'
               '        { const uint32_t or3 = nld | nrd | ncol;\n'
               '#ifdef __CUDACC__\n'
               '          asm("lop3.b32 %0, %1, %2, %3, 0x0C;" : "=r"(nf) : "r"(or3), "r"(bm), "r"(0u));\n'
               '#else\n'
               '          nf = bm & ~or3;\n'
               '#endif\n'
               '        }\n')
    else:
        new = ('        uint32_t nf;\n'
               '        { const uint32_t or2 = nld | nrd;\n'
               '#ifdef __CUDACC__\n'
               '          asm("lop3.b32 %0, %1, %2, %3, 0x02;" : "=r"(nf) : "r"(or2), "r"(ncol), "r"(bm));\n'
               '#else\n'
               '          nf = bm & ~(or2 | ncol);\n'
               '#endif\n'
               '        }\n')
    return lit(s, NF_OLD, new, 1, 'nf')
def F(s, use_asm):  # col carries the off-board bits (col | ~bm): nf = ~(nld | nrd | ncol), no bm
    s = lit(s, '    uint32_t cur_col = root_col;\n',
               '    const uint32_t nbm = ~bm;\n    uint32_t cur_col = root_col | nbm;\n', 1, 'cur_col init')
    s = lit(s, '(int)__builtin_popcount(root_col);', '(int)__builtin_popcount(root_col | nbm);', 1, 'depth_base')
    if use_asm:
        new = ('        uint32_t nf;\n'
               '#ifdef __CUDACC__\n'
               '        asm("lop3.b32 %0, %1, %2, %3, 0x01;" : "=r"(nf) : "r"(nld), "r"(nrd), "r"(ncol));\n'
               '#else\n'
               '        nf = ~(nld | nrd | ncol);\n'
               '#endif\n')
    else:
        new = '        uint32_t nf = ~(nld | nrd | ncol);\n'
    s = lit(s, NF_OLD, new, 1, 'nf')
    s = lit(s, 'cur_col = (uint32_t)packed_a & PACK403_MASK;', 'cur_col = ((uint32_t)packed_a & PACK403_MASK) | nbm;', 1, 'pop col 403')
    s = lit(s, 'cur_col = (uint32_t)(packed_a >> 21) & PACK402_MASK;', 'cur_col = ((uint32_t)(packed_a >> 21) & PACK402_MASK) | nbm;', 1, 'pop col 402')
    s = lit(s, '(uint64_t)cur_col | ((uint64_t)cur_avail << 22)', '(uint64_t)(cur_col & PACK403_MASK) | ((uint64_t)cur_avail << 22)', 2, 'push col 403')
    s = lit(s, '((uint64_t)cur_col << 21)', '((uint64_t)(cur_col & PACK402_MASK) << 21)', 2, 'push col 402')
    return s
OPS = {'Pa': lambda s: P(s, 'a'), 'Pb': lambda s: P(s, 'b'), 'E1': lambda s: E(s, 1), 'E2': lambda s: E(s, 2),
       'Fa': lambda s: F(s, True), 'Fc': lambda s: F(s, False)}
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
      [[ "$S_SHA" == "$WANT" ]] && pass "v0_reproduces_404_r3_sass_<$L> (sha $S_SHA, real $S_REAL, loop $S_LOOP, push $S_PUSH, pop $S_POP)" \
        || fail "v0_reproduces_404_r3_sass_<$L>" "sha $S_SHA != $WANT -- toolchain or base differs; the table is not comparable to 404_r3"
    fi
    info "$V <$L>" "frame=${FR:-?} spill=${SP:-?} regs=${RG:-?} sha=$S_SHA real=$S_REAL loop=$S_LOOP push=$S_PUSH pop=$S_POP rest=$REST"
  done
done

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
print("(404_r3 with this tool: <403> 464 / 130 / 12 / 15 / 103, <402> 462 / 129 / 11 / 15 / 103; the 404 search counted <402> push as 12 -- boundary only)")
EOF
rm -f "$LOGDIR"/bin/*.bin; rmdir "$LOGDIR/bin" 2>/dev/null
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL] ($(stat -c %s "$TARBALL") bytes)" || info "tarball_created" "tar failed"
command -v zip >/dev/null 2>&1 && zip -qr "${LOGDIR}.zip" "$LOGDIR" && info "zip_created" "${LOGDIR}.zip"
echo "results: $TSV"; echo "tarball: $TARBALL"
echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
exit 0
