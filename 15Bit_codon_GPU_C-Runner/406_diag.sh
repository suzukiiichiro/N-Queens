#!/usr/bin/env bash
# 406_diag.sh
#
# rev406 DIAGNOSTIC cells -- not a production candidate, nothing is adopted here.
#
# 406_search (2026-10-05 14:58, base 405): a carried cur_x = (ld<<1)|(rd>>1)|col
# lets nf be formed two stages after `bit` instead of three. <403>:
#     0   466 real / loop 130 / 35 regs
#     H1  459 / 123 / 38   shared nf == 0 test; ptxas dropped 16 register MOVs
#     H2  465 / 129 / 37   no-mark arm tests nf before forming nld / nrd
# Path profile (first 4,096 real records, counts per 100 steps, N=21 / N=22):
#     dead (nf == 0) 19.4 / 18.8   future-check dead 19.6 / 19.7
#     descend 60.8 / 61.3 (push 26.8 / 26.5)   pop 26.8 / 26.5
#     mark steps 0, jmark 0  <-- the sample never enters the mark block
# Instructions executed per path (no mark; read off the listings):
#               dead  fc-dead  descend  descend+push  pop   weighted (N=21)
#     405        29     41       55        69          21     5648 / 100 steps
#     H1         28     37       55        65          25     5550  (-1.7%)
#     H2         24     38       56        66          25     5553  (-1.7%)
# Executed count is NOT time (Pa+Ta: same count, +0.82%), and both variants
# change the kinds of instructions and the control flow. So: measure.
#
#   S  each diagnostic binary is generated from 405's .cu and its SASS sha
#      must equal the search's (both instantiations)                    HARD
#   Cells, N=21 auto (layout 403), direct = BLOCK 32 / MB 960 / helper 1 + 128 MB:
#     D0  405   anchor (405 B1 106,881.6)
#     D1  H1    results.bin == D0                                       HARD
#     D2  H2    results.bin == D0                                       HARD
#   Pre-registered reading (fixed before the run; no direction asserted):
#     Dk/D0 <= 0.998          candidate: next step is a four-file revision
#                             that also measures N=22
#     0.998 < Dk/D0 < 1.002   no effect at this resolution; closed
#     Dk/D0 >= 1.002          loses; closed, recorded
#     Information only: if time followed the executed count, both would land
#     near 0.983. If both are candidates, the faster one is taken forward.
#   After the GPU cells (CPU, GPU idle): the path profile again on a STRIDE
#   sample (NREC_PROF records spread over the whole input) for N=21 and N=22,
#   to see whether "mark steps = 0" is a property of the first records only.
#   SKIP_PROF=1 skips it.
#
# USAGE
#   bash 406_diag.sh          # ~8 min GPU (3 cells x 2 min + builds) + profile

set -u

REV="406_diag"
BASE_CU="${BASE_CU:-405_kernel_maxd14.cu}"
BASE_BIN="${BASE_BIN:-405_kernel_maxd14}"
GCC="${GCC:-gcc}"
NREC_PROF="${NREC_PROF:-4096}"
SKIP_PROF="${SKIP_PROF:-0}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
ARCH="${ARCH:-sm_86}"
BLOCK="${BLOCK:-32}"
MB="${MB:-960}"
HMB="${HMB:-128}"
COOLDOWN="${COOLDOWN:-15}"
CLK_INTERVAL="${CLK_INTERVAL:-10}"
ANCHOR_D0="${ANCHOR_D0:-106881.6}"
ORACLE21=314666222712
ORACLE22=2691008701644
# cell : generator ops : binary name : <402> sha : <403> sha   (shas from 406_search)
SPECS=(
  "D1:H1:406d_H1_kernel_maxd14:00a0d569cfd32bf0:1ee5dbe112d12936"
  "D2:H2:406d_H2_kernel_maxd14:b10b86fa8d17cf38:ec03ed3f341a730f"
)
BASE402_SHA="530de5ce72cf920c"; BASE403_SHA="3fd01fac2e9aedd3"

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
IN21="$(find_input 21)"; IN22="$(find_input 22)"
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
banner "N=21 direct, layout 403: D0 $BASE_BIN, then the diagnostics"
run_direct D0 "$BASE_BIN" 21 auto 403 || summary_exit
d="$(abspct "${VAL[D0]}" "$ANCHOR_D0")"; le "$d" 0.15 && pass "D0_anchor (${d}% from 405 B1 $ANCHOR_D0)" || info "D0_anchor" "${d}% from $ANCHOR_D0 -- ratios are still read within this session"
REFB1="$(ls -t 405_validate_*/bins/B1_results.bin 2>/dev/null | head -1)"
if [[ -n "${REFB1:-}" && -s "$REFB1" ]]; then cmp -s "$REFB1" "$LOGDIR/bins/D0_results.bin" && pass "D0_results_identical_to_405_B1" || fail "D0_results_differ_from_405_B1" "$REFB1"; fi
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
print("(diagnostic only: D1 = H1, D2 = H2; production stays 405)")
EOF
if [[ "$SKIP_PROF" != "1" ]]; then
  banner "Path profile on a stride sample (CPU, single thread, GPU idle)"
  if gen_variant "$BASE_CU" "$LOGDIR/src/v_prof.cu" "Z" 2> "$LOGDIR/gen_prof.log" && "$GCC" -O2 -fopenmp -x c -o "$LOGDIR/v_prof_cpu" "$LOGDIR/src/v_prof.cu" -lm 2> "$LOGDIR/gcc_prof.log"; then
    for C in "21 $IN21" "22 ${IN22:-}"; do
      set -- $C; [[ -f "${2:-/nonexistent}" ]] || { info "profile[N=$1]" "no input, skipped"; continue; }
      python3 - "$2" "/tmp/${REV}_stride_N$1.bin" "$NREC_PROF" <<'EOF' > "$LOGDIR/prof_sample_N$1.txt" 2>&1
import sys, os
src, dst, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
REC = 28                                  # 7 x uint32 per record
n = os.path.getsize(src) // REC
step = max(1, n // want)
with open(src, 'rb') as f, open(dst, 'wb') as g:
    k = 0
    for i in range(0, n, step):
        if k >= want: break
        f.seek(i * REC); g.write(f.read(REC)); k += 1
print("records=%d stride=%d sampled=%d bytes_mod=%d" % (n, step, k, os.path.getsize(src) % REC))
EOF
      info "profile_sample[N=$1]" "$(cat "$LOGDIR/prof_sample_N$1.txt")"
      OMP_NUM_THREADS=1 NQ_LAYOUT=auto "$LOGDIR/v_prof_cpu" "$1" "/tmp/${REV}_stride_N$1.bin" "/tmp/${REV}_prof_N$1.out" > "$LOGDIR/prof_N$1.log" 2>&1
      grep -q '^\[nq-prof\] step' "$LOGDIR/prof_N$1.log" && pass "path_profile_stride[N=$1]" || info "path_profile_stride[N=$1]" "no counters (see prof_N$1.log) -- information only"
    done
    python3 - "$LOGDIR/prof_N21.log" "$LOGDIR/prof_N22.log" <<'EOF' | tee "$LOGDIR/8_profile_stride.txt"
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
print("(stride sample over the whole input; 406_search's first-4,096 sample had mark_step = 0)")
EOF
    rm -f "$LOGDIR/v_prof_cpu" /tmp/${REV}_stride_N2?.bin
  else info "path_profile_build" "failed (information only): $(cat "$LOGDIR/gen_prof.log" "$LOGDIR/gcc_prof.log" 2>/dev/null | head -2)"; fi
fi

{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd14" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
command -v zip >/dev/null 2>&1 && zip -qr "${LOGDIR}.zip" "$LOGDIR" -x '*_kernel_maxd14' && info "zip_created" "${LOGDIR}.zip"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
