#!/usr/bin/env bash
# 409_compile_search.sh
#
# COMPILE-ONLY SEARCH on top of 408 (GPU time zero): the mark arm (14 instructions in 408's
# <403> loop, 8.9% of steps), spelled shorter. Compiles every variant with nvcc, compares
# SASS against the control (= 408), builds the gcc CPU harness of every variant and checks
# per-record equality against 408 on real input.
#
# WHY (static read of 408's mark arm, 2026-10-06):
#   1430 IMAD.SHL  raw*8          \ code*8 from the funnel-shifted schedule (2)
#   1450 LOP3      & 0x38         /
#   1440 MOV       table lo       \ the 64-bit table read (2; hi word is an immediate)
#   1460 SHF.R.U64 >> code*8      /
#   1480 LOP3      step = mt & 3        the three field extractions (4): upper bytes of mt are
#   1470 SHF.R     mt >> 4               other codes' entries, so every field needs an AND
#   1490 LOP3      kt = .. & 3
#   14b0 SHF.R     mt >> 2
#   14a0 SHF.L / 14d0 SHF.R  by step     shifts (2)
#   14c0 IMAD      kt * n4              (1)
#   14e0 LOP3 nld / 14f0 LOP3 nrd       (2)        + 1500 LOP3 nf (shared with the no-mark arm)
#   Two levers:
#   (a) PRMT (__byte_perm(lo, hi, code)) reads ONE byte zero-extended: no code*8, no shift,
#       and no AND to clear the other codes' bytes -- provided byte(code 0) == 0 (code 0 never
#       enters the mark arm, so its entry is free).
#   (b) field order inside the byte: with a zero-extended byte the TOP field needs no AND
#       (mt >> 4) and the BOTTOM field folds into the OR (LOP3 shl | (mt & 3) is one
#       instruction). Put step on top and lo2 at the bottom; kt in the middle as
#       bK = (mt & 0xC) * (n4 >> 2) (n4 >> 2 is loop-invariant; n4 = 1 << (N-4), N >= 8).
#
# VARIANTS (generator below; base = 408; byte(0) = 0 in every new table):
#   0    control = 408 (must reproduce <403> e940cb96 / 445 / 108 / MOV 8, <402> 821302f2 / 446 / 109)
#   P1   (a) only: PRMT read, 408's byte layout (step | lo2<<2 | kt<<4), extractions unchanged
#   P2   (a)+(b): PRMT read, layout lo2 | kt<<2 | step<<4; step = mt >> 4; nld |= mt & 3;
#        bK = (mt & 0xC) * n4q, n4q = n4 >> 2 hoisted before the main loop
#   P3   P2 with bK = ((mt >> 2) & 3) * n4 (no hoisted n4q: does n4q cost a register / a MOV?)
#   S2   (b) only: shift read as in 408 plus `& 0xFF`, layout and extractions as P2
#   Expected mark arm (incl. nf): 408 14; P1 ~12; P2 ~10; P3 ~11; S2 ~13.
#   Risk registered: PRMT takes its two table words in REGISTERS. ptxas may (i) hoist them
#   (+2 registers, MOV -1 in the loop) or (ii) rematerialize them per iteration (MOV +1 vs 408).
#   The MOV column decides: a variant with more hot-loop MOVs than the control is NOT a
#   candidate (permanent rule), whatever its instruction count.
#   CPU side (gcc): __byte_perm is spelled (T64 >> (code*8)) & 0xFF -- identical value.
#
# USAGE
#   bash 409_compile_search.sh                 # all variants, ~4 min, GPU zero
#   VARIANTS="0 P2" bash 409_compile_search.sh
#
# Prerequisites: 408_kernel_maxd14.cu, N=21 / N=22 inputs (located like 408_validate.sh).

set -u

REV="409"
BASE_CU="${BASE_CU:-408_kernel_maxd14.cu}"
VARIANTS="${VARIANTS:-0 P1 P2 P3 S2}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
NREC="${NREC:-2048}"
CTRL403_SHA="${CTRL403_SHA:-e940cb96e5e65fc4}"; CTRL403_REAL=445; CTRL403_LOOP=108
CTRL402_SHA="${CTRL402_SHA:-821302f227dcd160}"; CTRL402_REAL=446; CTRL402_LOOP=109
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

# tables, computed from the OP_* masks of the source (checked against it); byte(code 0) = 0
import re
STEP3, ADD1, BL1, BL2, KN3, KN4 = 24, 32, 12, 16, 18, 8
for name, val in (('OP_STEP3_MASK', STEP3), ('OP_ADD1_MASK', ADD1), ('OP_BL1_MASK', BL1),
                  ('OP_BL2_MASK', BL2), ('OP_KN3_MASK', KN3), ('OP_KN4_MASK', KN4)):
    m = re.search(r'static const uint32_t %s\s*=\s*(\d+)u;' % name, src)
    if not m or int(m.group(1)) != val:
        sys.exit("generator: %s != %d in the source" % (name, val))
def fields(c):
    step = 2 + ((STEP3 >> c) & 1)
    lo2  = ((ADD1 >> c) & 1) | ((BL1 >> c) & 1) | (((BL2 >> c) & 1) << 1)
    kt   = (((KN3 >> c) & 1) << 1) | ((KN4 >> c) & 1)
    return step, lo2, kt
T_L1 = T_L2 = 0        # L1 = 408's order (step | lo2<<2 | kt<<4) ; L2 = lo2 | kt<<2 | step<<4
for c in range(1, 8):
    step, lo2, kt = fields(c)
    T_L1 |= ((kt << 4) | (lo2 << 2) | step) << (8 * c)
    T_L2 |= ((step << 4) | (kt << 2) | lo2) << (8 * c)
T408_LINE = 'static const uint64_t MARK_T64 = 0x0202062B17062202ULL;\n'
def table_lines(t, tag):
    return ('/* 409 (%s): byte(code) as described in 409_compile_search.sh; byte(code 0) = 0 (code 0 never marks). */\n'
            'static const uint64_t MARK_T64 = 0x%016XULL;\n'
            'static const uint32_t MARK_T64_LO = 0x%08Xu;\n'
            'static const uint32_t MARK_T64_HI = 0x%08Xu;\n' % (tag, t, t & 0xFFFFFFFF, t >> 32))
MARK_OLD = ('            const uint32_t mt = (uint32_t)(MARK_T64 >> (block_code * 8u));\n'
            '            const uint32_t stepu = mt & 3u;\n'
            '            const uint32_t bKu = ((mt >> 4) & 3u) * n4;\n'
            '            nld = ((cur_ld | bit) << stepu) | ((mt >> 2) & 3u);\n'
            '            nrd = ((cur_rd | bit) >> stepu) | bKu;\n')
READ_PRMT = ('#ifdef __CUDACC__\n'
             '            const uint32_t mt = __byte_perm(MARK_T64_LO, MARK_T64_HI, block_code);\n'
             '#else\n'
             '            const uint32_t mt = (uint32_t)(MARK_T64 >> (block_code * 8u)) & 0xFFu;\n'
             '#endif\n')
READ_SHIFT = '            const uint32_t mt = (uint32_t)(MARK_T64 >> (block_code * 8u)) & 0xFFu;\n'
L1_EXTRACT = ('            const uint32_t stepu = mt & 3u;\n'
              '            const uint32_t bKu = ((mt >> 4) & 3u) * n4;\n'
              '            nld = ((cur_ld | bit) << stepu) | ((mt >> 2) & 3u);\n'
              '            nrd = ((cur_rd | bit) >> stepu) | bKu;\n')
L2_EXTRACT_Q = ('            const uint32_t stepu = mt >> 4;\n'
                '            const uint32_t bKu = (mt & 0xCu) * n4q;\n'
                '            nld = ((cur_ld | bit) << stepu) | (mt & 3u);\n'
                '            nrd = ((cur_rd | bit) >> stepu) | bKu;\n')
L2_EXTRACT_N = ('            const uint32_t stepu = mt >> 4;\n'
                '            const uint32_t bKu = ((mt >> 2) & 3u) * n4;\n'
                '            nld = ((cur_ld | bit) << stepu) | (mt & 3u);\n'
                '            nrd = ((cur_rd | bit) >> stepu) | bKu;\n')
CURX_LINE = '    uint32_t cur_x = (cur_ld << 1) | (cur_rd >> 1) | cur_col;\n'
N4Q = ('    /* 409: n4 >> 2 is loop-invariant (n4 = 1 << (N-4), N >= 8); bK = (kt << 2) * (n4 >> 2) = kt * n4. */\n'
       '    const uint32_t n4q = n4 >> 2;\n')
def P1(s):
    s = lit(s, T408_LINE, table_lines(T_L1, 'P1'), 1, 'table')
    return lit(s, MARK_OLD, '            /* 409 (P1): one byte, zero-extended, by PRMT. */\n' + READ_PRMT + L1_EXTRACT, 1, 'mark body')
def P2(s):
    s = lit(s, T408_LINE, table_lines(T_L2, 'P2'), 1, 'table')
    s = lit(s, CURX_LINE, N4Q + CURX_LINE, 1, 'cur_x line')
    return lit(s, MARK_OLD, '            /* 409 (P2): PRMT byte, step on top (no AND), lo2 at the bottom (folds into the OR). */\n' + READ_PRMT + L2_EXTRACT_Q, 1, 'mark body')
def P3(s):
    s = lit(s, T408_LINE, table_lines(T_L2, 'P3'), 1, 'table')
    return lit(s, MARK_OLD, '            /* 409 (P3): as P2 without the hoisted n4 >> 2. */\n' + READ_PRMT + L2_EXTRACT_N, 1, 'mark body')
def S2(s):
    s = lit(s, T408_LINE, table_lines(T_L2, 'S2'), 1, 'table')
    s = lit(s, CURX_LINE, N4Q + CURX_LINE, 1, 'cur_x line')
    return lit(s, MARK_OLD, '            /* 409 (S2): shift read + & 0xFF, fields as P2. */\n' + READ_SHIFT + L2_EXTRACT_Q, 1, 'mark body')

OPS = {'P1': P1, 'P2': P2, 'P3': P3, 'S2': S2}
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
banner "CPU reference (408) on $NREC records x 3 conditions"
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
  # CPU equality vs 408
  EQ="ok"
  if "$GCC" -O2 -fopenmp -x c -o "$LOGDIR/cpu/cpu_${tag}" "$vcu" -lm 2>"$LOGDIR/cpu/gcc_${tag}.log"; then
    for cond in "N22_auto 22 $IN22 NQ_LAYOUT=auto" "N21_l402 21 $IN21 NQ_LAYOUT=402" "N21_auto 21 $IN21 NQ_LAYOUT=auto"; do
      set -- $cond; lbl="$1"; n="$2"; in="$3"; e="$4"
      if run_cpu "$LOGDIR/cpu/cpu_${tag}" "${tag}_${lbl}" "$n" "$in" "$e" && cmp -s "$LOGDIR/cpu/base_${lbl}.bin" "$LOGDIR/cpu/${tag}_${lbl}.bin" && [[ -s "$LOGDIR/cpu/${tag}_${lbl}.bin" ]]; then :; else EQ="DIFF[$lbl]"; fi
    done
  else EQ="gcc_fail"; fi
  [[ "$EQ" == "ok" ]] && pass "cpu_equal_vs_408[$V]" || fail "cpu_equal_vs_408[$V]" "$EQ"
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
    [[ "$S403" == "$CTRL403_SHA" && "$R403" == "$CTRL403_REAL" && "$L403" == "$CTRL403_LOOP" ]] && pass "control_403_reproduces_408" || fail "control_403_reproduces_408" "$S403 / $R403 / $L403 != $CTRL403_SHA / $CTRL403_REAL / $CTRL403_LOOP -- toolchain differs, read the numbers relatively"
    [[ "$S402" == "$CTRL402_SHA" && "$R402" == "$CTRL402_REAL" && "$L402" == "$CTRL402_LOOP" ]] && pass "control_402_reproduces_408" || fail "control_402_reproduces_408" "$S402 / $R402 / $L402 != $CTRL402_SHA / $CTRL402_REAL / $CTRL402_LOOP"
    CTRL_MOV="$MOV"
  fi
  for L in 402 403; do
    [[ "${FR[$L]}" == "160" && "${SP[$L]}" == "0" && "${RG[$L]:-99}" -le 44 ]] && pass "ptxas[$V]_L${L}_160B_spill0_regs${RG[$L]}" || info "ptxas[$V]_L${L}" "frame=${FR[$L]}B regs=${RG[$L]} spill=${SP[$L]} -- outside the gate (160 B / 0 / <= 44)"
  done
done

banner "Results (<403> hot loop; control = 408)"
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
print("\nhot-loop dumps with instruction-kind histograms: loops/<variant>_403.txt ; diff two with: diff loops/0_403.txt loops/P2_403.txt")
EOF

TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd14" --exclude="cpu_*" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball" "tar failed"
echo; echo "===== ${REV} search summary ====="; echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
exit 0
