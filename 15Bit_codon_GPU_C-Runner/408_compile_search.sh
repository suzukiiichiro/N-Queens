#!/usr/bin/env bash
# 408_compile_search.sh
#
# COMPILE-ONLY SEARCH on top of 407 (GPU time zero). Compiles every variant with
# nvcc, compares SASS against the control (= 407), builds the gcc CPU harness of
# every variant and checks per-record equality against 407 on real input.
#
# WHY (static read of the 407 <403> hot loop, 121 instructions, 2026-10-06):
#   The remaining branch regions (BSSY..BSYNC) in the loop are
#     B5  mark / no-mark          mark arm 28 instr (8.9% of steps), no-mark arm 6
#     B6  push guard              10 instr + BSSY/BSYNC/BRA (push 26.9%, skipped 34.2%)
#     B5  descent region          4 register MOVs remain (avail, col, rd, ld); they
#                                 stay because the push reads the PARENT state after
#                                 the child state exists -- not touched here.
#   15 of the mark arm's 28 instructions are five copies of
#   "MOV constant -> SHF by block_code -> & 1". 407's M replaced them with a
#   per-DEPTH table built once per task (arm 28 -> 14) but the 14-iteration table
#   loop unrolled and took 55 registers (M: -0.89%; M2 inside the decode loop:
#   +1.43%). block_code is 3 bits (codes 1..5 occur), so a per-CODE table is a
#   32-bit (or 64-bit) IMMEDIATE: no precompute, no extra registers, and the
#   nibble extraction stays shared (block_code is still used -- M's "5-instruction
#   (7 << 4d) & schedule64 test" came from block_code becoming dead).
#
# VARIANTS (generator below; letters combine with '+'):
#   0    control = 407 (must reproduce <403> 7864364a / 458 / 121, <402> 999fdf33 / 459 / 122)
#   T1   mark arm from two 32-bit per-code tables:
#          TA nibble(code) = lo2 << 2 | step      (step 2|3 ; lo2 = addv | bLi)
#          TB 2 bits(code) = kn3 << 1 | kn4       (bK = kt * n4, n3 == 2 * n4 on both hosts)
#   T2   mark arm from one 64-bit per-code table (byte(code) = kt << 4 | lo2 << 2 | step)
#   Pu   push without a branch region: the frame is written to stack_a/b[stack_ptr]
#        on EVERY descent, only the increment is conditional
#        (stack_ptr += (cur_avail != 0)). In-bounds: at a descent stack_ptr <= cur_depth
#        <= terminal_depth - 1 <= 12 < MAXD14_ANCESTOR (13); the CPU harness asserts it.
#        This is NOT the closed "push-guard removal" (push-then-pop); pops are unchanged.
#        Cost: 2 extra STL on the 34.2% of steps that descend without pushing.
#   T1+Pu, T2+Pu
#
# READ (registered before running):
#   - a variant whose hot-loop MOV count is ABOVE the control is NOT a candidate,
#     whatever its instruction count (permanent rule, 407 K)
#   - instruction counts narrow the field; they do not predict time. Variants whose
#     control shape changes (all of these) go to 408_diag cells (N=21, 2 min each)
#   - expected: T1/T2 mark arm 28 -> 13..15, registers <= 40, frame 160 B;
#     Pu loop -3 control instructions (BSSY/BSYNC/BRA), +0 arithmetic, STL unconditional
#
# USAGE
#   bash 408_compile_search.sh                 # all variants, ~5 min, GPU zero
#   VARIANTS="0 T1 Pu" bash 408_compile_search.sh
#   NREC=4096 bash 408_compile_search.sh
#
# Prerequisites: 407_kernel_maxd14.cu, N=21 / N=22 inputs (located like 407_validate.sh).

set -u

REV="408"
BASE_CU="${BASE_CU:-407_kernel_maxd14.cu}"
VARIANTS="${VARIANTS:-0 T1 T2 Pu T1+Pu T2+Pu}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
NREC="${NREC:-2048}"
CTRL403_SHA="${CTRL403_SHA:-7864364ac0c83a1d}"; CTRL403_REAL=458; CTRL403_LOOP=121
CTRL402_SHA="${CTRL402_SHA:-999fdf338ebc8db7}"; CTRL402_REAL=459; CTRL402_LOOP=122
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_search_${TS}}"
mkdir -p "$LOGDIR/variants" "$LOGDIR/sass" "$LOGDIR/loops" "$LOGDIR/cpu"
TSV="$LOGDIR/${REV}_search.tsv"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }

[[ -f "$BASE_CU" ]] && pass "base_present[$BASE_CU]" || { fail "base_present" "$BASE_CU missing"; exit 1; }
[[ ! -x "$NVCC" ]] && NVCC="nvcc"; [[ ! -x "$CUOBJDUMP" ]] && CUOBJDUMP="cuobjdump"

# ---------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------
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

# SASS: sha / real / loop / total, plus the hot loop's instruction-kind histogram and dump
sass_stat() {  # sass.txt frag loopdump.txt  -> "sha real loop total MOV BSSY BSYNC BREAK BRA STL LDL"
python3 - "$1" "$2" "$3" <<'PYEOF'
import sys,re,hashlib
out={};cur=None
for line in open(sys.argv[1],errors='replace'):
    m=re.match(r'\s*Function\s*:\s*(\S+)',line)
    if m: cur=m.group(1); out[cur]=[]; continue
    if cur is None: continue
    m=re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s*(.*?);\s*(/\*.*\*/)?\s*$',line)
    if m: out[cur].append((int(m.group(1),16),m.group(2).strip()))
k=[k for k in out if sys.argv[2] in k]
if len(k)!=1: print("CANNOT_LOCATE -1 -1 -1 -1 -1 -1 -1 -1 -1 -1"); sys.exit(0)
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
sha=hashlib.sha256('\n'.join(s for _,s in I).encode()).hexdigest()[:16]
if not best: print(sha, real, -1, len(I), *([-1]*7)); sys.exit(0)
def op(s):
    s=re.sub(r'^@!?P\d\s+','',s); return s.split()[0]
ops=[op(s) for _,s in best]
mov=sum(1 for o in ops if o.startswith('MOV') or o.startswith('IMAD.MOV'))
cnt=lambda p: sum(1 for o in ops if o==p or o.startswith(p+'.'))
with open(sys.argv[3],'w') as f:
    for a,s in best: f.write("%04x  %s\n"%(a,s))
    f.write("\n# kinds:\n")
    from collections import Counter
    for o,n in sorted(Counter(ops).items(), key=lambda x:(-x[1],x[0])): f.write("# %4d  %s\n"%(n,o))
print(sha, real, len(best), len(I), mov, cnt('BSSY'), cnt('BSYNC'), cnt('BREAK'), cnt('BRA'), cnt('STL'), cnt('LDL'))
PYEOF
}

find_input() { local n="$1" c; for lg in $(ls -t 40*_crunner_logs/crunner_*_N${n}.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { echo "$c"; return; }; done; ls -t constellations_N${n}_*.sched394f.bin 2>/dev/null | head -1; }
IN21="$(find_input 21)"; IN22="$(find_input 22)"
[[ -f "${IN21:-/nonexistent}" ]] && pass "input21_located ($IN21)" || { fail "input21_located" "no N=21 sched input"; exit 1; }
[[ -f "${IN22:-/nonexistent}" ]] && pass "input22_located ($IN22)" || { fail "input22_located" "no N=22 sched input"; exit 1; }

cu_code_region "$BASE_CU" > "$LOGDIR/variants/base_code.cu"
"$GCC" -O2 -fopenmp -x c -o "$LOGDIR/cpu/cpu_base" "$BASE_CU" -lm 2>"$LOGDIR/cpu/gcc_base.log" && pass "gcc_base" || { fail "gcc_base" "$(head -3 "$LOGDIR/cpu/gcc_base.log")"; exit 1; }
banner "CPU reference (407) on $NREC records x 3 conditions"
run_cpu() {  # bin label N input env...
  local bin="$1" label="$2" n="$3" in="$4"; shift 4
  env "$@" "$bin" "$n" "$in" "$LOGDIR/cpu/${label}.bin" "$NREC" > "$LOGDIR/cpu/${label}.log" 2>&1
}
run_cpu "$LOGDIR/cpu/cpu_base" base_N22_auto 22 "$IN22" NQ_LAYOUT=auto || { fail "cpu_base_N22" "rc=$?"; exit 1; }
run_cpu "$LOGDIR/cpu/cpu_base" base_N21_l402 21 "$IN21" NQ_LAYOUT=402  || { fail "cpu_base_N21_l402" "rc=$?"; exit 1; }
run_cpu "$LOGDIR/cpu/cpu_base" base_N21_auto 21 "$IN21" NQ_LAYOUT=auto || { fail "cpu_base_N21_auto" "rc=$?"; exit 1; }
pass "cpu_base_3_conditions"

printf 'variant\tremoved\tadded\tframe402\tregs402\tspill402\tframe403\tregs403\tspill403\tsha402\treal402\tloop402\tsha403\treal403\tloop403\tMOV\tBSSY\tBSYNC\tBREAK\tBRA\tSTL\tLDL\tcpu_eq\n' > "$TSV"
declare -A ROW
for V in $VARIANTS; do
  banner "variant $V"
  tag="${V//+/_}"; vcu="$LOGDIR/variants/${REV}_${tag}.cu"
  if ! gen_variant "$BASE_CU" "$vcu" "$V" 2>"$LOGDIR/variants/gen_${tag}.log"; then fail "gen[$V]" "$(cat "$LOGDIR/variants/gen_${tag}.log")"; continue; fi
  cu_code_region "$vcu" > "$LOGDIR/variants/${tag}_code.cu"
  R_=$(diff "$LOGDIR/variants/base_code.cu" "$LOGDIR/variants/${tag}_code.cu" | grep -c '^<' || true); A_=$(diff "$LOGDIR/variants/base_code.cu" "$LOGDIR/variants/${tag}_code.cu" | grep -c '^>' || true)
  info "code_region_diff[$V]" "removed=$R_ added=$A_"
  # CPU equality vs 407
  EQ="ok"
  if "$GCC" -O2 -fopenmp -x c -o "$LOGDIR/cpu/cpu_${tag}" "$vcu" -lm 2>"$LOGDIR/cpu/gcc_${tag}.log"; then
    for cond in "N22_auto 22 $IN22 NQ_LAYOUT=auto" "N21_l402 21 $IN21 NQ_LAYOUT=402" "N21_auto 21 $IN21 NQ_LAYOUT=auto"; do
      set -- $cond; lbl="$1"; n="$2"; in="$3"; e="$4"
      if run_cpu "$LOGDIR/cpu/cpu_${tag}" "${tag}_${lbl}" "$n" "$in" "$e" && cmp -s "$LOGDIR/cpu/base_${lbl}.bin" "$LOGDIR/cpu/${tag}_${lbl}.bin" && [[ -s "$LOGDIR/cpu/${tag}_${lbl}.bin" ]]; then :; else EQ="DIFF[$lbl]"; fi
    done
  else EQ="gcc_fail"; fi
  [[ "$EQ" == "ok" ]] && pass "cpu_equal_vs_407[$V]" || fail "cpu_equal_vs_407[$V]" "$EQ"
  # nvcc
  bin="$LOGDIR/${REV}_${tag}_kernel_maxd14"
  "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$bin" "$vcu" -lcuda > "$LOGDIR/nvcc_${tag}.log" 2>&1
  if [[ ! -x "$bin" ]]; then fail "nvcc[$V]" "$(grep -i error "$LOGDIR/nvcc_${tag}.log" | head -3)"; continue; fi
  declare -A FR RG SP
  for L in 402 403; do
    FR[$L]="$(grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${L}E" "$LOGDIR/nvcc_${tag}.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"
    RG[$L]="$(grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${L}E" "$LOGDIR/nvcc_${tag}.log" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"
    SP[$L]="$(grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${L}E" "$LOGDIR/nvcc_${tag}.log" | grep -o '[0-9]* bytes spill stores' | head -1 | cut -d' ' -f1)"
  done
  "$CUOBJDUMP" -sass "$bin" > "$LOGDIR/sass/${tag}.txt" 2>&1
  read -r S402 R402 L402 T402 _ _ _ _ _ _ _ <<< "$(sass_stat "$LOGDIR/sass/${tag}.txt" kernel_dfs_iter_gpu_maxd14ILi402E "$LOGDIR/loops/${tag}_402.txt")"
  read -r S403 R403 L403 T403 MOV BSSY BSYNC BREAK BRA STL LDL <<< "$(sass_stat "$LOGDIR/sass/${tag}.txt" kernel_dfs_iter_gpu_maxd14ILi403E "$LOGDIR/loops/${tag}_403.txt")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$V" "$R_" "$A_" "${FR[402]}" "${RG[402]}" "${SP[402]}" "${FR[403]}" "${RG[403]}" "${SP[403]}" "$S402" "$R402" "$L402" "$S403" "$R403" "$L403" "$MOV" "$BSSY" "$BSYNC" "$BREAK" "$BRA" "$STL" "$LDL" "$EQ" >> "$TSV"
  info "$V" "<402> $S402 real $R402 loop $L402 | <403> $S403 real $R403 loop $L403 regs ${RG[403]} frame ${FR[403]}B spill ${SP[403]} | loop403 MOV=$MOV BSSY=$BSSY BSYNC=$BSYNC BREAK=$BREAK BRA=$BRA STL=$STL LDL=$LDL"
  if [[ "$V" == "0" ]]; then
    [[ "$S403" == "$CTRL403_SHA" && "$R403" == "$CTRL403_REAL" && "$L403" == "$CTRL403_LOOP" ]] && pass "control_403_reproduces_407" || fail "control_403_reproduces_407" "$S403 / $R403 / $L403 != $CTRL403_SHA / $CTRL403_REAL / $CTRL403_LOOP -- toolchain differs, read the numbers relatively"
    [[ "$S402" == "$CTRL402_SHA" && "$R402" == "$CTRL402_REAL" && "$L402" == "$CTRL402_LOOP" ]] && pass "control_402_reproduces_407" || fail "control_402_reproduces_407" "$S402 / $R402 / $L402 != $CTRL402_SHA / $CTRL402_REAL / $CTRL402_LOOP"
    CTRL_MOV="$MOV"
  fi
  for L in 402 403; do
    [[ "${FR[$L]}" == "160" && "${SP[$L]}" == "0" && "${RG[$L]:-99}" -le 44 ]] && pass "ptxas[$V]_L${L}_160B_spill0_regs${RG[$L]}" || info "ptxas[$V]_L${L}" "frame=${FR[$L]}B regs=${RG[$L]} spill=${SP[$L]} -- outside the gate (160 B / 0 / <= 44)"
  done
done

banner "Results (<403> hot loop; control = 407)"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_table.txt"
import csv, sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
ctrl=next((r for r in rows if r['variant']=='0'),None)
print(f"{'variant':<8}{'rm/add':>8}{'frame':>6}{'regs':>5}{'spill':>6} {'sha403':<17}{'real':>5}{'loop':>5}{'MOV':>5}{'BSSY':>5}{'BSYNC':>6}{'BREAK':>6}{'BRA':>5}{'STL':>4}{'LDL':>4}  {'real402':>7}{'loop402':>8}  cpu_eq  note")
for r in rows:
    note=[]
    if ctrl and r['variant']!='0':
        try:
            if int(r['MOV'])>int(ctrl['MOV']): note.append("MOV+%d -> NOT a candidate"%(int(r['MOV'])-int(ctrl['MOV'])))
            d=int(r['loop403'])-int(ctrl['loop403']); note.append("loop %+d"%d)
        except ValueError: pass
    if r['cpu_eq']!='ok': note.append("CPU "+r['cpu_eq'])
    print(f"{r['variant']:<8}{r['removed']+'/'+r['added']:>8}{r['frame403']:>6}{r['regs403']:>5}{r['spill403']:>6} {r['sha403']:<17}{r['real403']:>5}{r['loop403']:>5}{r['MOV']:>5}{r['BSSY']:>5}{r['BSYNC']:>6}{r['BREAK']:>6}{r['BRA']:>5}{r['STL']:>4}{r['LDL']:>4}  {r['real402']:>7}{r['loop402']:>8}  {r['cpu_eq']:<6}  {'; '.join(note)}")
print("\nhot-loop dumps with instruction-kind histograms: loops/<variant>_403.txt ; diff two with: diff loops/0_403.txt loops/T1_403.txt")
EOF

TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd14" --exclude="cpu_*" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball" "tar failed"
echo; echo "===== ${REV} search summary ====="; echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
exit 0
