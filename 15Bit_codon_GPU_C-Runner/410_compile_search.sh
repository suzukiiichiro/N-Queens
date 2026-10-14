#!/usr/bin/env bash
# 410_compile_search.sh
#
# COMPILE-ONLY SEARCH on top of 408 (GPU time zero): the pop path's load latency.
# Compiles every variant with nvcc, compares SASS against the control (= 408), builds the gcc
# CPU harness of every variant and checks per-record equality against 408 on real input.
#
# WHY (static read of 408's pop, <403> loop, 16 instructions + control; pops are 26.9% of steps):
#   19b0 IMAD  &stack_a[sp]      19c0 IMAD  &stack_b[sp]
#   19d0 LDL.64 packed_a         19e0 LDL   packed_b           <- the two loads
#   1a00 LOP3 colm = a & 0x3fffff      -> 1a20 POPC -> 1ab0 IADD depth        (load -> 3 stages)
#   1a10 SHF a>>11 -> 1a40 LOP3 ld -> 1a70 SHL -> 1a80 LOP3 -> 1aa0 LOP3 cur_x  (load -> 5 stages)
#   1a50 SHF.R.U64 avail -> 1a90 LOP3 & bm  -> loop top: bit = avail & -avail  (load -> 2 stages)
#   Every consumer of the popped frame waits for the LDL (L1 hit, tens of cycles). The run has
#   BLOCK 32 x MB 960 on 80 SMs = ~12 warps per SM, 3 per scheduler: little to hide that latency
#   behind. K (+9.2%) and Pu (+63%) showed how sensitive this kernel is to chain length and LSU.
#
# IDEA (C, "top-of-stack in registers"): keep frame sp-1 in registers (top_a, top_b); memory holds
#   frames 0..sp-2. pop: sp--, decode top (registers), then PREFETCH top <- mem[sp-1] for the next
#   pop -- the load is issued but nothing waits on it until the next pop or push. push: spill the
#   old top to mem[sp-1] (store data = registers, no packing), pack the new frame into top, sp++.
#   Index trick: slot k holds frame k-1, slot 0 is a dummy, so the spill writes mem[sp] and the
#   prefetch reads mem[sp] after the decrement -- the SAME address arithmetic as today, no clamp,
#   no branch; at sp == 0 the dummy slot absorbs the write / read. Slots 0..12 = 13 = MAXD14_ANCESTOR
#   (at a push sp <= cur_depth <= terminal_depth - 1 <= 12); the CPU harness keeps its overflow abort.
#   Cost: +3 live registers (37 -> ~40, gate <= 44); push/pop instruction counts unchanged on paper.
#   Risk registered: the pop decodes the OLD top while the LDLs write the NEW top -- ptxas may
#   rename cleanly or add register MOVs at the loop back-edge. The MOV column decides (rule: more
#   hot-loop MOVs than the control -> not a candidate). The gain, if any, is latency, which no
#   instruction count shows: a candidate goes to a diagnostic cell.
#
# VARIANTS (generator below; base = 408):
#   0    control = 408 (must reproduce <403> e940cb96 / 445 / 108 / MOV 8, <402> 821302f2 / 446 / 109)
#   C    top-of-stack in registers, dummy slot 0 (spill mem[sp] = top; pop: top <- mem[sp] after sp--)
#   Cc   same, clamp spelling instead of the dummy slot (mem[max(sp-1, 0)]): does ptxas spell
#        the clamp as IMNMX, and does the slot choice change anything else?
#   Cr   C with the root fast path's push also spelled as spill + top (C sets top only there,
#        since sp == 0 at the root): is the root path's shape visible in the loop?
#   Expected: C loop ~108 (same), MOV 8 (if renamed cleanly) .. 11 (if 3 back-edge MOVs appear),
#   registers ~40, frame 160 B.
#
# USAGE
#   bash 410_compile_search.sh                 # all variants, ~3 min, GPU zero
#   VARIANTS="0 C" bash 410_compile_search.sh
#
# Prerequisites: 408_kernel_maxd14.cu, N=21 / N=22 inputs (located like 408_validate.sh).

set -u

REV="410"
BASE_CU="${BASE_CU:-408_kernel_maxd14.cu}"
VARIANTS="${VARIANTS:-0 C Cc Cr}"
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

DECL_OLD = '    uint32_t stack_b[MAXD14_ANCESTOR];\n'
DECL_NEW = (DECL_OLD +
            '    /* 410 (C): the top frame (sp-1) lives in registers; slot k of the arrays holds frame k-1\n'
            '     * (slot 0 is a dummy), so spill and prefetch both address stack_*[stack_ptr]. */\n'
            '    uint64_t top_a = 0;\n'
            '    uint32_t top_b = 0;\n')
# push heads (root: 16 spaces, main: 12 spaces) -- the LAYOUT_IF line plus its first assignment line
ROOT_PUSH_HEAD = ('                LAYOUT_IF (layout == PACK_LAYOUT_402) {\n'
                  '                    stack_a[stack_ptr] = (uint64_t)(cur_ld & PACK402_MASK)\n')
MAIN_PUSH_HEAD = ('            LAYOUT_IF (layout == PACK_LAYOUT_402) {\n'
                  '                stack_a[stack_ptr] = (uint64_t)(cur_ld & PACK402_MASK)\n')
SPILL_DUMMY = ('            /* 410 (C): spill the old top into slot sp (= frame sp-1); at sp == 0 this is the dummy slot. */\n'
               '            stack_a[stack_ptr] = top_a;\n'
               '            stack_b[stack_ptr] = top_b;\n')
SPILL_CLAMP = ('            /* 410 (Cc): spill the old top into frame sp-1, clamped to 0 (slot 0 = frame 0 here). */\n'
               '            { const int spill = (stack_ptr > 0) ? stack_ptr - 1 : 0; stack_a[spill] = top_a; stack_b[spill] = top_b; }\n')
ROOT_SPILL_DUMMY = ('                /* 410 (Cr): spill at the root too (sp == 0: the dummy slot), for a uniform push. */\n'
                    '                stack_a[stack_ptr] = top_a;\n'
                    '                stack_b[stack_ptr] = top_b;\n')
POP_OLD = ('            stack_ptr -= 1;\n'
           '            uint64_t packed_a = stack_a[stack_ptr];\n')
POP_DUMMY = ('            stack_ptr -= 1;\n'
             '            /* 410 (C): decode the top from registers; prefetch the frame below (slot sp = frame sp-1;\n'
             '             * at sp == 0 the dummy slot) into top for the NEXT pop -- nothing waits on these loads here. */\n'
             '            const uint64_t packed_a = top_a;\n'
             '            const uint32_t packed_b0 = top_b;\n'
             '            top_a = stack_a[stack_ptr];\n'
             '            top_b = stack_b[stack_ptr];\n')
POP_CLAMP = ('            stack_ptr -= 1;\n'
             '            /* 410 (Cc): decode the top from registers; prefetch frame sp-1 (clamped to 0) into top. */\n'
             '            const uint64_t packed_a = top_a;\n'
             '            const uint32_t packed_b0 = top_b;\n'
             '            { const int below = (stack_ptr > 0) ? stack_ptr - 1 : 0; top_a = stack_a[below]; top_b = stack_b[below]; }\n')
def tos(s, spill, pop, root_spill):
    s = lit(s, DECL_OLD, DECL_NEW, 1, 'stack decl')
    # the four frame writes (root + main) now target the registers
    s = lit(s, 'stack_a[stack_ptr] = (uint64_t)(cur_ld & PACK402_MASK)', 'top_a = (uint64_t)(cur_ld & PACK402_MASK)', 2, 'push a 402')
    s = lit(s, 'stack_b[stack_ptr] = cur_rd;', 'top_b = cur_rd;', 2, 'push b 402')
    s = lit(s, 'stack_a[stack_ptr] = (uint64_t)(cur_col & PACK403_MASK)', 'top_a = (uint64_t)(cur_col & PACK403_MASK)', 2, 'push a 403')
    s = lit(s, 'stack_b[stack_ptr] = pb403; }', 'top_b = pb403; }', 2, 'push b 403')
    main_head = MAIN_PUSH_HEAD.replace('stack_a[stack_ptr] =', 'top_a =')
    root_head = ROOT_PUSH_HEAD.replace('stack_a[stack_ptr] =', 'top_a =')
    s = lit(s, main_head, spill + main_head, 1, 'main push head')
    if root_spill:
        s = lit(s, root_head, root_spill + root_head, 1, 'root push head')
    # pop
    s = lit(s, POP_OLD, pop, 1, 'pop head')
    s = lit(s, '                cur_rd  = stack_b[stack_ptr];\n', '                cur_rd  = packed_b0;\n', 1, 'pop b 402')
    s = lit(s, '                uint32_t packed_b = stack_b[stack_ptr];\n', '                uint32_t packed_b = packed_b0;\n', 1, 'pop b 403')
    return s
def C(s):  return tos(s, SPILL_DUMMY, POP_DUMMY, None)
def Cc(s): return tos(s, SPILL_CLAMP, POP_CLAMP, None)
def Cr(s): return tos(s, SPILL_DUMMY, POP_DUMMY, ROOT_SPILL_DUMMY)

OPS = {'C': C, 'Cc': Cc, 'Cr': Cr}
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
print("\nhot-loop dumps with instruction-kind histograms: loops/<variant>_403.txt ; diff two with: diff loops/0_403.txt loops/C_403.txt")
EOF

TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd14" --exclude="cpu_*" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball" "tar failed"
echo; echo "===== ${REV} search summary ====="; echo "OK=$PASS  FAIL=$FAIL"
if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
exit 0
