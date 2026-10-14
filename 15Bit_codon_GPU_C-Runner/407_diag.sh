#!/usr/bin/env bash
# 407_diag.sh
#
# rev407 DIAGNOSTIC cells -- not a production candidate, nothing is adopted here.
#
# 407_search (2026-10-05 16:56, base 406). <403> real / loop / registers:
#     0      459 / 123 / 38
#     J      458 / 121 / 37   j-mark without a branch region (BSSY, BSYNC, BREAK, BRA -1 each)
#     M      706 / 112 / 55   mark arm 28 -> 14 instructions, BUT the 14-iteration table loop
#                             was unrolled outside the hot loop: 55 registers (gate is <= 44)
#     K      480 / 142 / 39   future check and cur_x shared: 18 register MOVs and 2 BREAKs
#                             came back (+19 in the loop)
#     K+J 476/137   K+M 725/131   K+J+M 720/126
# Cells, one variable each against 406:
#     D1 J     candidate
#     D2 M     what 55 registers cost (or do not)
#     D3 K     SEPARATION: K puts back the MOVs / BREAKs that 406 removed while keeping
#              406's nf path. If K loses several %, the -5.9% of 406 was the MOV / control
#              shape rather than the branch depth.
#     D4 M2    NOT searched: M's two words filled inside the schedule-decode loop instead of
#              by a separate loop (aim: same mark arm, registers back under 44)
#     D5 J+M2  NOT searched
#   For D1-D3 the SASS sha must equal the search's (both instantiations)          HARD
#   For D4-D5 the harness records sha / real / loop / registers / frame (nothing to compare
#   with), requires 160 B and 0 spill, and checks CPU per-record equality vs 406
#   (N=21 auto, N=22 auto) before the GPU cell.                                   HARD
#   Every cell: results.bin == D0, oracle, 1710 MHz.                              HARD
#   Pre-registered reading (fixed before the run; no direction asserted for any cell):
#     Dk/D0 <= 0.998 candidate ; 0.998 < Dk/D0 < 1.002 no effect ; >= 1.002 loses.
#     Candidates are NOT assumed additive; the next revision takes ONE measured cell.
#
# USAGE
#   bash 407_diag.sh          # ~14 min (6 cells x 2 min + builds)

set -u

REV="407_diag"
BASE_CU="${BASE_CU:-406_kernel_maxd14.cu}"
BASE_BIN="${BASE_BIN:-406_kernel_maxd14}"
GCC="${GCC:-gcc}"
NREC="${NREC:-1024}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
ARCH="${ARCH:-sm_86}"
BLOCK="${BLOCK:-32}"
MB="${MB:-960}"
HMB="${HMB:-128}"
COOLDOWN="${COOLDOWN:-15}"
CLK_INTERVAL="${CLK_INTERVAL:-10}"
ANCHOR_D0="${ANCHOR_D0:-100612.1}"
ORACLE21=314666222712
ORACLE22=2691008701644
# cell : generator ops : binary name : <402> sha : <403> sha   (shas from 407_search; "-" = not searched)
SPECS=(
  "D1:J:407d_J_kernel_maxd14:999fdf338ebc8db7:7864364ac0c83a1d"
  "D2:M:407d_M_kernel_maxd14:224795e3e64e4100:560aa78484979a02"
  "D3:K:407d_K_kernel_maxd14:b06ee8eecc563da9:be635538f8ea1469"
  "D4:M2:407d_M2_kernel_maxd14:-:-"
  "D5:J+M2:407d_JM2_kernel_maxd14:-:-"
)
BASE402_SHA="00a0d569cfd32bf0"; BASE403_SHA="1ee5dbe112d12936"

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
def M2(s):
    # M with the two words filled INSIDE the schedule-decode loop (one frame at a time), not by a
    # separate 14-iteration loop: 407_search's M unrolled that loop (real 459 -> 706, 55 registers).
    s = lit(s, '    uint32_t schedule_lo = 0, schedule_hi = 0;\n',
               '    uint32_t schedule_lo = 0, schedule_hi = 0;\n    uint32_t mk_alo = 0, mk_ahi = 0, mk_b = 0;\n', 1, 'decl')
    old = ('        if (schedule_depth < 8) {\n'
           '            schedule_lo |= frame_nibble << (schedule_depth * 4);\n'
           '        } else {\n'
           '            schedule_hi |= frame_nibble << ((schedule_depth - 8) * 4);\n'
           '        }\n')
    new = ('        { const uint32_t mk_bc = frame_nibble & 7u;\n'
           '          const uint32_t mk_nib = (((OP_STEP3_MASK >> mk_bc) & 1u) << 2)\n'
           '                                | ((OP_ADD1_MASK >> mk_bc) & 1u) | ((OP_BL1_MASK >> mk_bc) & 1u)\n'
           '                                | (((OP_BL2_MASK >> mk_bc) & 1u) << 1);\n'
           '          const uint32_t mk_kt = (((OP_KN3_MASK >> mk_bc) & 1u) << 1) | ((OP_KN4_MASK >> mk_bc) & 1u);\n'
           '          mk_b |= mk_kt << (schedule_depth * 2);\n'
           '          if (schedule_depth < 8) {\n'
           '              schedule_lo |= frame_nibble << (schedule_depth * 4);\n'
           '              mk_alo |= mk_nib << (schedule_depth * 4);\n'
           '          } else {\n'
           '              schedule_hi |= frame_nibble << ((schedule_depth - 8) * 4);\n'
           '              mk_ahi |= mk_nib << ((schedule_depth - 8) * 4);\n'
           '          }\n'
           '        }\n')
    s = lit(s, old, new, 1, 'nibble store')
    s = lit(s, SCHED, SCHED + '    const uint64_t mk_a64 = ((uint64_t)mk_ahi << 32) | (uint64_t)mk_alo;\n', 1, 'schedule64')
    new = ('            const uint32_t mka = (uint32_t)(mk_a64 >> (cur_depth * 4));\n'
           '            const uint32_t stepu = 2u + ((mka >> 2) & 1u);\n'
           '            const uint32_t bKu = ((mk_b >> (cur_depth * 2)) & 3u) * n4;\n'
           '            nld = ((cur_ld | bit) << stepu) | (mka & 3u);\n'
           '            nrd = ((cur_rd | bit) >> stepu) | bKu;\n')
    return lit(s, MARK_OLD, new, 1, 'mark body')
OPS = {'K': K, 'J': J, 'M': M, 'M2': M2}
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
    if [[ "$want" == "-" ]]; then info "S_sass_<$L>[$1] (not searched)" "sha $S_SHA real $S_REAL loop $S_LOOP"
    elif [[ "$S_SHA" == "$want" ]]; then pass "S_sass_<$L>[$1] (sha $S_SHA, real $S_REAL, loop $S_LOOP)"
    else fail "S_sass_<$L>[$1]" "sha $S_SHA != $want (real $S_REAL loop $S_LOOP) -- not the binary the search read; stop"; summary_exit; fi
  done
}
banner "Building"
if [[ ! -x "$BASE_BIN" ]]; then "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$BASE_BIN" "$BASE_CU" -lcuda > "$LOGDIR/05_nvcc_base.log" 2>&1; fi
[[ -x "$BASE_BIN" ]] && pass "base_binary_available[$BASE_BIN]" || { fail "base_binary" "cannot build $BASE_BIN"; summary_exit; }
check_sass D0 "$BASE_BIN" "$BASE402_SHA" "$BASE403_SHA"
IN21="$(find_input 21)"; IN22="$(find_input 22)"
[[ -f "${IN21:-/nonexistent}" ]] && pass "input21_located ($IN21)" || { fail "input21_located" "no N=21 sched input"; summary_exit; }
declare -A SKIPCELL
for SPEC in "${SPECS[@]}"; do
  IFS=: read -r CELL OPSV BINV W402 W403 <<< "$SPEC"
  gen_variant "$BASE_CU" "$LOGDIR/src/${BINV}.cu" "$OPSV" 2> "$LOGDIR/gen_${CELL}.log" || { fail "generate[$CELL $OPSV]" "$(cat "$LOGDIR/gen_${CELL}.log")"; SKIPCELL[$CELL]=1; continue; }
  rm -f "$BINV"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$BINV" "$LOGDIR/src/${BINV}.cu" -lcuda > "$LOGDIR/05_nvcc_${CELL}.log" 2>&1
  if [[ ! -x "$BINV" ]]; then fail "nvcc_build[$CELL $OPSV]" "$(grep -i error "$LOGDIR/05_nvcc_${CELL}.log" | head -3 | tr '\n' ' ')"; SKIPCELL[$CELL]=1; continue; fi
  pass "nvcc_build[$CELL $OPSV -> $BINV]"
  PT="$(grep -A3 'kernel_dfs_iter_gpu_maxd14ILi403E' "$LOGDIR/05_nvcc_${CELL}.log" | tr '\n' ' ')"
  FR="$(echo "$PT" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"; SP="$(echo "$PT" | grep -o '[0-9]* bytes spill stores' | head -1 | cut -d' ' -f1)"; RG="$(echo "$PT" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"
  info "ptxas_<403>[$CELL $OPSV]" "frame=${FR:-?} spill=${SP:-?} registers=${RG:-?}$([[ "${RG:-0}" -gt 44 ]] && echo '  (> 44: measured anyway, this is a diagnostic)')"
  check_sass "$CELL" "$BINV" "$W402" "$W403"
  if [[ "$W403" == "-" ]]; then
    [[ "${FR:-}" == "160" && "${SP:-}" == "0" ]] || { fail "ptxas_frame_spill[$CELL]" "frame=${FR:-?} spill=${SP:-?} (need 160 / 0)"; SKIPCELL[$CELL]=1; continue; }
    CPUOK=1
    if "$GCC" -O2 -fopenmp -x c -o "$LOGDIR/cpu_cur" "$LOGDIR/src/${BINV}.cu" -lm 2> "$LOGDIR/gcc_${CELL}.log" && "$GCC" -O2 -fopenmp -x c -o "$LOGDIR/cpu_base" "$BASE_CU" -lm 2>> "$LOGDIR/gcc_${CELL}.log"; then
      for C in "21 $IN21" "22 ${IN22:-}"; do
        set -- $C; [[ -f "${2:-/nonexistent}" ]] || continue
        NQ_LAYOUT=auto "$LOGDIR/cpu_base" "$1" "$2" "/tmp/${REV}_base_$1.bin" "$NREC" > "$LOGDIR/2_cpu_${CELL}_N$1_base.log" 2>&1
        NQ_LAYOUT=auto "$LOGDIR/cpu_cur"  "$1" "$2" "/tmp/${REV}_cur_$1.bin"  "$NREC" > "$LOGDIR/2_cpu_${CELL}_N$1_cur.log" 2>&1
        [[ -s "/tmp/${REV}_base_$1.bin" ]] && cmp -s "/tmp/${REV}_base_$1.bin" "/tmp/${REV}_cur_$1.bin" || CPUOK=0
      done
    else CPUOK=0; fi
    rm -f "$LOGDIR/cpu_cur" "$LOGDIR/cpu_base"
    [[ "$CPUOK" == "1" ]] && pass "cpu_equal_to_406[$CELL $OPSV] ($NREC records, N=21 and N=22)" || { fail "cpu_equal_to_406[$CELL $OPSV]" "differs or did not build -- cell skipped"; SKIPCELL[$CELL]=1; continue; }
  fi
done

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
d="$(abspct "${VAL[D0]}" "$ANCHOR_D0")"; le "$d" 0.15 && pass "D0_anchor (${d}% from 406 B1 $ANCHOR_D0)" || info "D0_anchor" "${d}% from $ANCHOR_D0 -- ratios are still read within this session"
REFB1="$(ls -t 406_validate_*/bins/B1_results.bin 2>/dev/null | head -1)"
if [[ -n "${REFB1:-}" && -s "$REFB1" ]]; then cmp -s "$REFB1" "$LOGDIR/bins/D0_results.bin" && pass "D0_results_identical_to_406_B1" || fail "D0_results_differ_from_406_B1" "$REFB1"; fi
for SPEC in "${SPECS[@]}"; do
  IFS=: read -r CELL OPSV BINV W402 W403 <<< "$SPEC"
  [[ "${SKIPCELL[$CELL]:-0}" == "1" ]] && { info "cell[$CELL $OPSV]" "skipped (see FAIL above)"; continue; }
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
print("(diagnostic only: D1 = J, D2 = M, D3 = K, D4 = M2, D5 = J+M2; production stays 406)")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd14" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
command -v zip >/dev/null 2>&1 && zip -qr "${LOGDIR}.zip" "$LOGDIR" -x '*_kernel_maxd14' && info "zip_created" "${LOGDIR}.zip"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
