#!/usr/bin/env bash
# 405_diag.sh
#
# rev405 DIAGNOSTIC cells -- not a production candidate, nothing is adopted here.
#
# The two compile searches (2026-10-05 12:54, 13:14) found NO spelling that
# shortens the <403> hot loop (130). Three variants keep the count at 130 but
# change the dependency chains; by the standing rule a chain change cannot be
# read statically, so they are measured, one variable each, against 404_r3:
#
#   E1     nf = ~or3 & bm as one inline-PTX LOP3 (0x0c) + ISETP.
#          nf's value one stage earlier; the nf == 0 branch at the same depth.
#   Fa+Fm  col carries the off-board bits: nf = LOP3 0x01 + ISETP (step -1,
#          nf two stages earlier, branch one stage earlier); pop +1 (OR of ~bm,
#          beside POPC, not in front of it).
#   Pa+Ta  two running stack pointers, 32-bit empty test: IMAD x2 + IADD3
#          becomes IADD3 x3 on push and on pop; the pop's loads no longer wait
#          for an IMAD (address ready one stage earlier).
#
#   S  each diagnostic binary is generated from 404_r3's .cu and its SASS sha
#      must equal the search's (both instantiations)                    HARD
#   Cells, N=21 auto (layout 403), direct = BLOCK 32 / MB 960 / helper 1 + 128 MB:
#     D0  404_r3   anchor (404-r3 B1 107,325.0)
#     D1  E1       results.bin == D0                                    HARD
#     D2  Fa+Fm    results.bin == D0                                    HARD
#     D3  Pa+Ta    results.bin == D0                                    HARD
#   Pre-registered reading (fixed before the run; NO direction is asserted --
#   the count is unchanged and only the chains move):
#     Dk/D0 <= 0.998          candidate: next step is a four-file revision
#                             that also measures N=22
#     0.998 < Dk/D0 < 1.002   no effect at this resolution; closed
#     Dk/D0 >= 1.002          loses; closed, recorded
#   If more than one is a candidate they are NOT assumed additive: the
#   combination is searched and measured separately.
#
# USAGE
#   bash 405_diag.sh          # ~10 min (4 cells x 2 min + builds)

set -u

REV="405_diag"
BASE_CU="${BASE_CU:-404_r3_kernel_maxd14.cu}"
BASE_BIN="${BASE_BIN:-404_r3_kernel_maxd14}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
ARCH="${ARCH:-sm_86}"
BLOCK="${BLOCK:-32}"
MB="${MB:-960}"
HMB="${HMB:-128}"
COOLDOWN="${COOLDOWN:-15}"
CLK_INTERVAL="${CLK_INTERVAL:-10}"
ANCHOR_D0="${ANCHOR_D0:-107325.0}"
ORACLE21=314666222712
ORACLE22=2691008701644
# cell : generator ops : binary name : <402> sha : <403> sha   (shas from 405_search / 405_r2_search)
SPECS=(
  "D1:E1:405d_E1_kernel_maxd14:49bee351b9314178:2382d464a7f6b546"
  "D2:Fa+Fm:405d_FaFm_kernel_maxd14:530de5ce72cf920c:3fd01fac2e9aedd3"
  "D3:Pa+Ta:405d_PaTa_kernel_maxd14:12c2b2f9b42e29ae:d7f50318642869d1"
)
BASE402_SHA="ab5608e947e586bc"; BASE403_SHA="7c67c3aa44cb7a4d"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_${TS}}"
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
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; if [[ "$FAIL" -gt 0 ]]; then for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi; exit 0; }

[[ -f "$BASE_CU" ]] && pass "file_present[$BASE_CU]" || { fail "file_present[$BASE_CU]" "missing"; exit 1; }
[[ ! -x "$NVCC" ]] && NVCC="nvcc"; [[ ! -x "$CUOBJDUMP" ]] && CUOBJDUMP="cuobjdump"
mkdir -p "$LOGDIR/sass" "$LOGDIR/bins" "$LOGDIR/src"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1

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
def T(s, which):  # after Pa/Pb: compare only the low 32 bits of the pointer (local addresses are 32-bit)
    if which == 'a':
        return lit(s, 'if (sp_a == stack_a) {', 'if ((uint32_t)(uintptr_t)sp_a == (uint32_t)(uintptr_t)stack_a) {', 1, 'empty test a')
    return lit(s, 'if (sp_b == stack_b) {', 'if ((uint32_t)(uintptr_t)sp_b == (uint32_t)(uintptr_t)stack_b) {', 1, 'empty test b')
def Fm(s):  # after Fa/Fc: depth from the masked col (popc next to the OR of ~bm, not behind it)
    s = lit(s, '(int)__builtin_popcount(root_col | nbm);',
               '(int)__builtin_popcount((root_col | nbm) & ((layout == PACK_LAYOUT_402) ? PACK402_MASK : PACK403_MASK));', 1, 'depth_base masked')
    s = lit(s, 'cur_col = ((uint32_t)packed_a & PACK403_MASK) | nbm;',
               '{ const uint32_t colm = (uint32_t)packed_a & PACK403_MASK; cur_col = colm | nbm; cur_depth = (int)__builtin_popcount(colm) - depth_base; }', 1, 'pop col 403 masked')
    s = lit(s, 'cur_col = ((uint32_t)(packed_a >> 21) & PACK402_MASK) | nbm;',
               '{ const uint32_t colm = (uint32_t)(packed_a >> 21) & PACK402_MASK; cur_col = colm | nbm; cur_depth = (int)__builtin_popcount(colm) - depth_base; }', 1, 'pop col 402 masked')
    s = sub(s, r'^[ ]+cur_depth = \(int\)__builtin_popcount\(cur_col\) - depth_base;\n', '', 1, 'common depth line')
    return s
OPS = {'Pa': lambda s: P(s, 'a'), 'Pb': lambda s: P(s, 'b'), 'E1': lambda s: E(s, 1), 'E2': lambda s: E(s, 2),
       'Fa': lambda s: F(s, True), 'Fc': lambda s: F(s, False),
       'Ta': lambda s: T(s, 'a'), 'Tb': lambda s: T(s, 'b'), 'Fm': Fm}
for tok in [t for t in letters.split('+') if t and t != '0']:
    if tok not in OPS: sys.exit("generator: unknown op " + tok)
    src = OPS[tok](src)
open(outp, 'w', encoding='utf-8').write(src)
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
find_input() { local n="$1" c; for lg in $(ls -t 40*_crunner_logs/crunner_*_N${n}.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { echo "$c"; return; }; done; ls -t constellations_N${n}_*.sched394f.bin 2>/dev/null | head -1; }

check_sass() {  # label binary want402 want403
  "$CUOBJDUMP" -sass "./$2" > "$LOGDIR/sass/$2.txt" 2>&1
  local L want
  for L in 402 403; do
    want="$3"; [[ "$L" == "403" ]] && want="$4"
    read -r S_SHA S_REAL S_LOOP S_TOT <<< "$(sass_stat "$LOGDIR/sass/$2.txt" "kernel_dfs_iter_gpu_maxd14ILi${L}E")"
    [[ "$S_SHA" == "$want" ]] && pass "S_sass_<$L>[$1] (sha $S_SHA, real $S_REAL, loop $S_LOOP)" || { fail "S_sass_<$L>[$1]" "sha $S_SHA != $want (real $S_REAL loop $S_LOOP) -- not the binary the search read; stop"; summary_exit; }
  done
}
banner "Building"
if [[ ! -x "$BASE_BIN" ]]; then "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$BASE_BIN" "$BASE_CU" -lcuda > "$LOGDIR/05_nvcc_base.log" 2>&1; fi
[[ -x "$BASE_BIN" ]] && pass "base_binary_available[$BASE_BIN]" || { fail "base_binary" "cannot build $BASE_BIN"; summary_exit; }
check_sass D0 "$BASE_BIN" "$BASE402_SHA" "$BASE403_SHA"
for SPEC in "${SPECS[@]}"; do
  IFS=: read -r CELL OPSV BINV W402 W403 <<< "$SPEC"
  gen_variant "$BASE_CU" "$LOGDIR/src/${BINV}.cu" "$OPSV" 2> "$LOGDIR/gen_${CELL}.log" || { fail "generate[$CELL $OPSV]" "$(cat "$LOGDIR/gen_${CELL}.log")"; summary_exit; }
  rm -f "$BINV"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$BINV" "$LOGDIR/src/${BINV}.cu" -lcuda > "$LOGDIR/05_nvcc_${CELL}.log" 2>&1
  [[ -x "$BINV" ]] && pass "nvcc_build[$CELL $OPSV -> $BINV]" || { fail "nvcc_build[$CELL]" "$(grep -i error "$LOGDIR/05_nvcc_${CELL}.log" | head -3)"; summary_exit; }
  check_sass "$CELL" "$BINV" "$W402" "$W403"
done
IN21="$(find_input 21)"; IN22=""
[[ -f "${IN21:-/nonexistent}" ]] && pass "input21_located ($IN21)" || { fail "input21_located" "no N=21 sched input"; summary_exit; }

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
same_bin() {  # cell refcell
  local ref="$LOGDIR/bins/${2}_results.bin"
  if [[ -s "$ref" ]] && cmp -s "$ref" "$LOGDIR/bins/${1}_results.bin"; then pass "${1}_per_thread_results_identical ($(stat -c %s "$LOGDIR/bins/${1}_results.bin") bytes vs $2)"
  else fail "${1}_per_thread_results_DIFFER" "results.bin $1 != $2 -- the variant is WRONG on the GPU; its time means nothing"; fi
}
banner "N=21 direct, layout 403: D0 $BASE_BIN, then the three diagnostics"
run_direct D0 "$BASE_BIN" 21 auto 403 || summary_exit
d="$(abspct "${VAL[D0]}" "$ANCHOR_D0")"; le "$d" 0.15 && pass "D0_anchor (${d}% from 404-r3 B1 $ANCHOR_D0)" || info "D0_anchor" "${d}% from $ANCHOR_D0 -- ratios are still read within this session"
REFB1="$(ls -t 404_r3_validate_*/bins/B1_results.bin 2>/dev/null | head -1)"
if [[ -n "${REFB1:-}" && -s "$REFB1" ]]; then cmp -s "$REFB1" "$LOGDIR/bins/D0_results.bin" && pass "D0_results_identical_to_404r3_B1" || fail "D0_results_differ_from_404r3_B1" "$REFB1"; fi
for SPEC in "${SPECS[@]}"; do
  IFS=: read -r CELL OPSV BINV W402 W403 <<< "$SPEC"
  sleep "$COOLDOWN"; run_direct "$CELL" "$BINV" 21 auto 403 || summary_exit
  same_bin "$CELL" D0
  r="$(ratio "${VAL[$CELL]}" "${VAL[D0]}")"; p="$(pct "${VAL[$CELL]}" "${VAL[D0]}")"
  if le "$r" 0.998; then info "READ[$CELL $OPSV]" "$CELL/D0 = $r (${p}%) -- CANDIDATE (registered: <= 0.998)"
  elif ge "$r" 1.002; then info "READ[$CELL $OPSV]" "$CELL/D0 = $r (${p}%) -- LOSES (registered: >= 1.002), closed"
  else info "READ[$CELL $OPSV]" "$CELL/D0 = $r (${p}%) -- no effect at this resolution, closed"; fi
done
CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "sm_clock_1710_all_cells" || fail "sm_clock_1710_all_cells" "see clk_*.tsv -- ratios are not readable"

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
base=next((float(r['kernel_ms']) for r in rows if r['cell']=='D0'),None)
print(f"{'cell':<5}{'binary':<30}{'N':>3}{'lay':>5}{'kernel_ms':>14}{'vs D0':>10}{'match':>6}{'sm':>6}")
for r in rows:
    v=float(r['kernel_ms'])
    print(f"{r['cell']:<5}{r['binary'][:29]:<30}{r['N']:>3}{r['layout']:>5}{v:>14.3f}{(v/base-1)*100:>+9.3f}%{r['match']:>6}{r['sm_mean']:>6}")
print("(diagnostic only: D1 = E1, D2 = Fa+Fm, D3 = Pa+Ta; production stays 404_r3)")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd14" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
command -v zip >/dev/null 2>&1 && zip -qr "${LOGDIR}.zip" "$LOGDIR" -x '*_kernel_maxd14' && info "zip_created" "${LOGDIR}.zip"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
