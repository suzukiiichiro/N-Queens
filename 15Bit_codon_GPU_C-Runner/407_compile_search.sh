#!/usr/bin/env bash
# 407_compile_search.sh
#
# rev407 search -- COMPILE + CPU EQUALITY ONLY, NO GPU RUN. Base: 406_kernel_maxd14.cu.
#
# Static read of 406 (H1) against 405, <403>, per path (no mark):
#                        instr  control(BSSY/BSYNC/BREAK/BRA)  MOV
#   405 descend + push     69            23                    12
#   406 descend + push     65            20                     5
#   405 dead end (nf==0)   29            11                     4
#   406 dead end           28            11                     0
# 406 also turned the future check into one branch (flag and test combined in
# `@!P0 BRA P2`, no BSSY region, no BREAK). None of these counts explains
# -5.9%; what they suggest is the direction: fewer saved-state MOVs, fewer
# separate conditional regions. Three independent candidates:
#
#   K  The future check's (nld << 1) | (nrd >> 1) | ncol IS the child's
#      cur_x. 406 forms part of it twice (SHL + OR again on every descent,
#      61% of steps). K forms it once (nx), tests it, carries it; the j-mark's
#      nld |= 1 becomes nx |= 2.
#   J  j-mark without a branch region: nf &= ~jm; nld |= jm; one rarely taken
#      nf == 0 exit (j-mark is tested on 3.6% of steps, dead on 0.1%).
#      Today it is a BSSY region with a depth-dependent branch for every step.
#   M  The mark's operands (shift 2/3, low bits, n3/n4) depend on the DEPTH
#      only. 406 decodes them from block_code on every mark step (the mark arm
#      is 28 instructions against 6 without a mark; 8-9% of steps are marks).
#      M decodes them once per task into two words (a nibble and 2 bits per
#      depth) and the mark step reads them: bK = kt * n4.
#
#  Variants: 0  K  J  M  K+J  K+M  K+J+M
#  Output per variant and instantiation: frame / spill / registers, SASS sha,
#  REAL instructions, hot loop, push, pop, rest, opcode-multiset diff vs v0,
#  loop listings. CPU equality vs v0: N=21 auto, N=21 l402, N=22 auto.
#  Hard check: v0 must reproduce 406's SASS (00a0d569 / 1ee5dbe1).
#  The counts choose what to measure; they do not predict the time
#  (H1 and H2: same executed count, 7.4% apart).
#
# USAGE
#   bash 407_compile_search.sh            # a few minutes, GPU idle

set -u

REV="407_search"
BASE_CU="${BASE_CU:-406_kernel_maxd14.cu}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
NREC="${NREC:-1024}"
VARIANTS="${VARIANTS:-0 K J M K+J K+M K+J+M}"
BASE402_SHA="${BASE402_SHA:-00a0d569cfd32bf0}"
BASE403_SHA="${BASE403_SHA:-1ee5dbe112d12936}"
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
FC_OLD = ('            if (((future_check_mask >> cur_depth) & 1u) != 0u) {\n'
          '                if ((bm & ~((nld << 1) | (nrd >> 1) | ncol)) == 0u) {\n')
JM_OLD = ('        uint32_t child_jmark = (child_jmark_mask >> cur_depth) & 1u;\n'
          '        if (child_jmark != 0u) {\n'
          '            nf &= ~1u;\n'
          '            if (nf == 0u) {\n'
          '                continue;\n'
          '            }\n'
          '            nld |= 1u;\n'
          '        }\n')
JM_K   = JM_OLD.replace('            nld |= 1u;\n', '            nld |= 1u;\n            nx |= 2u;\n')
def K(s):
    # the future check's (nld << 1) | (nrd >> 1) | ncol IS the child's cur_x: form it once (nx),
    # test it, and carry it; the j-mark's nld |= 1 becomes nx |= 2.
    s = lit(s, FC_OLD, ('            if (((future_check_mask >> cur_depth) & 1u) != 0u) {\n'
                        '                if ((bm & ~nx) == 0u) {\n'), 1, 'future check')
    s = lit(s, '        if (nf == 0u) {\n            continue;\n        }\n        {\n',
               '        if (nf == 0u) {\n            continue;\n        }\n        uint32_t nx = (nld << 1) | (nrd >> 1) | ncol;\n        {\n', 1, 'nx decl')
    s = lit(s, JM_OLD, JM_K, 1, 'jmark')
    s = lit(s, '        cur_x = (nld << 1) | (nrd >> 1) | ncol;\n', '        cur_x = nx;\n', 1, 'descent')
    return s
def J(s):
    # j-mark without a branch region: nf &= ~jm; nld |= jm; one (rarely taken) nf == 0 exit.
    withk = JM_K in s
    old = JM_K if withk else JM_OLD
    new = ('        uint32_t child_jmark = (child_jmark_mask >> cur_depth) & 1u;\n'
           '        nf &= ~child_jmark;\n'
           '        if (nf == 0u) {\n'
           '            continue;\n'
           '        }\n'
           '        nld |= child_jmark;\n' + ('        nx |= (child_jmark << 1);\n' if withk else ''))
    return lit(s, old, new, 1, 'jmark')
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
SCHED = '    const uint64_t schedule64 = ((uint64_t)schedule_hi << 32) | (uint64_t)schedule_lo;\n'
def M(s):
    # the mark's operands depend on the depth only: decode them once per task into two words
    #   mk_a64: nibble per depth = step3 << 2 | (addv | bLi)     mk_b: 2 bits per depth = kn3 << 1 | kn4
    # bK = n3*kn3 + n4*kn4 = (2*kn3 + kn4) * n4   (both hosts pass n3 = 1 << (N-3), n4 = 1 << (N-4))
    tab = (SCHED +
           '    uint64_t mk_a64 = 0;\n    uint32_t mk_b = 0;\n'
           '    { int mk_d;\n'
           '      for (mk_d = 0; mk_d < 14; ++mk_d) {\n'
           '          const uint32_t mk_bc = (uint32_t)(schedule64 >> (mk_d * 4)) & 7u;\n'
           '          const uint32_t mk_lo = ((OP_ADD1_MASK >> mk_bc) & 1u) | ((OP_BL1_MASK >> mk_bc) & 1u)\n'
           '                               | (((OP_BL2_MASK >> mk_bc) & 1u) << 1);\n'
           '          const uint32_t mk_kt = (((OP_KN3_MASK >> mk_bc) & 1u) << 1) | ((OP_KN4_MASK >> mk_bc) & 1u);\n'
           '          mk_a64 |= (uint64_t)((((OP_STEP3_MASK >> mk_bc) & 1u) << 2) | mk_lo) << (mk_d * 4);\n'
           '          mk_b   |= mk_kt << (mk_d * 2);\n'
           '      }\n'
           '    }\n')
    s = lit(s, SCHED, tab, 1, 'schedule64')
    new = ('            const uint32_t mka = (uint32_t)(mk_a64 >> (cur_depth * 4));\n'
           '            const uint32_t stepu = 2u + ((mka >> 2) & 1u);\n'
           '            const uint32_t bKu = ((mk_b >> (cur_depth * 2)) & 3u) * n4;\n'
           '            nld = ((cur_ld | bit) << stepu) | (mka & 3u);\n'
           '            nrd = ((cur_rd | bit) >> stepu) | bKu;\n')
    return lit(s, MARK_OLD, new, 1, 'mark body')
OPS = {'K': K, 'J': J, 'M': M}
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
      [[ "$S_SHA" == "$WANT" ]] && pass "v0_reproduces_406_sass_<$L> (sha $S_SHA, real $S_REAL, loop $S_LOOP, push $S_PUSH, pop $S_POP)" \
        || fail "v0_reproduces_406_sass_<$L>" "sha $S_SHA != $WANT -- toolchain or base differs; the table is not comparable to 406"
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
print("(406 with this tool: <403> 459 / 123 / 10 / 20 / 93)")
EOF
rm -f "$LOGDIR"/bin/*.bin; rmdir "$LOGDIR/bin" 2>/dev/null
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL] ($(stat -c %s "$TARBALL") bytes)" || info "tarball_created" "tar failed"
command -v zip >/dev/null 2>&1 && zip -qr "${LOGDIR}.zip" "$LOGDIR" && info "zip_created" "${LOGDIR}.zip"
echo "results: $TSV"; echo "tarball: $TARBALL"
echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
exit 0
