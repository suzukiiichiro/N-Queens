#!/usr/bin/env bash
# 411_validate.sh
#
# rev411 -- CONSOLIDATION: kernel == 408 byte for byte; the .py drops NQ_EXTRA_CTX=1 from
#   env_prefix (A/B 2026-10-06 13:58: A0 unset 97,913.1 / A1 =1 97,906.3 / A2 unset 97,908.6,
#   d = -0.005% -> INERT as registered; 256 MB of device memory returned).
#
#   S0 static: .cu code region == 408's (cmp); .py = 408Py + 3 executable lines; table entry
#      "./411_kernel_maxd14" with env_prefix "NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "; no table entry
#      carries NQ_EXTRA_CTX; gcc; notes
#   S1 ptxas 160 B / 0 spill / regs <= 44                               HARD
#   S2 <402> SASS sha 821302f2, real 446, loop 109  (== 408)            HARD
#   S3 <403> SASS sha e940cb96, real 445, loop 108  (== 408)            HARD
#   Y3 rc=3 cases                                                        HARD
#   Cells (direct = BLOCK 32 / MB 960 / helper 1 + 128 MB, NQ_EXTRA_CTX unset):
#     B0  408  N=21 auto (403)     anchor 408 A0 97,913.1 (flat 1710)
#     B1  411  N=21 auto (403)     results.bin == B0                     HARD (same SASS)
#     G21, G22  ./411Py -g 21 21 / -g 22 22   (production path)          [SKIP_G=1 for a dry run]
#         dispatch.log env_prefix has no NQ_EXTRA_CTX                    HARD
#         crunner log "[gpu-ctx] extra_ctx=0"                            HARD
#         oracle, layout=403                                             HARD
#         /tmp/crunner_N21_results.bin vs B0                             INFO
#   R1 (no gain claimed): |G21_norm/B1_norm - 1| <= 0.3% AND |G22_norm/854,582.1 - 1| <= 0.5%
#      (norm = kernel_ms * sm_mean / 1710); G21 free_mb >= 21,379 + 200
#   ADOPT: all HARD + R1 -> 411 is production (N=21 = G21, N=22 = G22).
#
# USAGE
#   STATIC_ONLY=1 bash 411_validate.sh      # OK=15
#                 bash 411_validate.sh      # ~21 min
#   SKIP_G=1      bash 411_validate.sh      # ~5 min (B0, B1 only; not adoptable)

set -u

REV="411"
PY_SRC="${PY_SRC:-411Py_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-${PY_SRC%.py}}"
PREV_PY="${PREV_PY:-408Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-411_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-411_kernel_maxd14}"
PREV_CU="${PREV_CU:-408_kernel_maxd14.cu}"
PREV_BIN="${PREV_BIN:-408_kernel_maxd14}"
SASS402_SHA="${SASS402_SHA:-821302f227dcd160}"; SASS402_REAL="${SASS402_REAL:-446}"; SASS402_LOOP="${SASS402_LOOP:-109}"
SASS403_SHA="${SASS403_SHA:-e940cb96e5e65fc4}"; SASS403_REAL="${SASS403_REAL:-445}"; SASS403_LOOP="${SASS403_LOOP:-108}"
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
KERNEL_SHA_411="${KERNEL_SHA_411:-ab483e70328c39c6}"   # == 408
ANCHOR_G22="${ANCHOR_G22:-854582.1}"       # 408 G22 (dispatcher)
ANCHOR_B0="${ANCHOR_B0:-97913.1}"         # 408 A/B A0 (N=21 layout 403, direct, flat 1710)
ANCHOR_G21="${ANCHOR_G21:-98024.9}"       # 408 G21
ANCHOR_G21_FREE="${ANCHOR_G21_FREE:-21379}"  # 408 G21 free_mb (with the extra context)
SKIP_G="${SKIP_G:-0}"
COOLDOWN="${COOLDOWN:-15}"
CLK_INTERVAL="${CLK_INTERVAL:-5}"
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
{ grep -q "^# 411 " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 411 ...' note block"
grep -qE '^REV_TAG:str="411"' "$CODE" && pass "source_rev_tag_is_411" || fail "source_rev_tag_is_411" "wrong REV_TAG"
grep -qF 'CRunnerEntry(14,"./411_kernel_maxd14","NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_points_at_411_helper_prefix_without_extra_ctx" || fail "source_table_points_at_411" "table entry wrong"
{ ! grep -E 'CRunnerEntry\(' "$CODE" | grep -q 'NQ_EXTRA_CTX'; } && pass "source_no_table_entry_carries_NQ_EXTRA_CTX" || fail "source_no_table_entry_carries_NQ_EXTRA_CTX" "$(grep -E 'CRunnerEntry\(' "$CODE" | grep 'NQ_EXTRA_CTX' | head -2)"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=960\b" "$CODE" && pass "source_default_max_blocks_is_960" || fail "source_default_max_blocks_is_960" "default not 960"
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_408Py (removed=3 added=3 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_408Py" "removed=$PR_ added=$PA_, expected 3/3"
else info "py_diff_fingerprint_vs_408Py" "skipped ($PREV_PY absent)"; fi
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
[[ "$KB" == "$KERNEL_SHA_411"* ]] && pass "cu_kernel_region_sha_is_408s (${KB:0:16})" || fail "cu_kernel_region_sha" "sha ${KB:0:16} != $KERNEL_SHA_411 -- not 408's kernel"
CR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true); CA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
[[ "$CR_" == "0" && "$CA_" == "0" ]] && pass "cu_code_region_fingerprint_vs_${PREV_CU%.cu} (removed=0 added=0)" || fail "cu_code_region_fingerprint" "removed=$CR_ added=$CA_, expected 0/0"
cmp -s "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" && pass "cu_code_region_identical_to_${PREV_CU%.cu}" || fail "cu_code_region_identical" "the .cu code region differs from $PREV_CU (comment-stripped) -- 411 must not change the kernel"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"/tmp/${REV}_gcc.log" && pass "cu_cpu_harness_builds" || fail "cu_cpu_harness_builds" "$(head -3 /tmp/${REV}_gcc.log)"
grep -q "rev411 " "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev411 note in header"

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
[[ "$S_SHA" == "$SASS402_SHA" && "$S_REAL" == "$SASS402_REAL" && "$S_LOOP" == "$SASS402_LOOP" ]] && pass "S2_sass_402_is_408s (sha $S_SHA, real $S_REAL, loop $S_LOOP)" || { fail "S2_sass_402_differs_from_408" "sha $S_SHA real $S_REAL loop $S_LOOP != $SASS402_SHA / $SASS402_REAL / $SASS402_LOOP -- stop and read the SASS"; summary_exit; }
read -r S_SHA S_REAL S_LOOP S_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_dfs_iter_gpu_maxd14ILi403E)"
[[ "$S_SHA" == "$SASS403_SHA" && "$S_REAL" == "$SASS403_REAL" && "$S_LOOP" == "$SASS403_LOOP" ]] && pass "S3_sass_403_is_408s (sha $S_SHA, real $S_REAL, loop $S_LOOP)" || { fail "S3_sass_403_differs_from_408" "sha $S_SHA real $S_REAL loop $S_LOOP != $SASS403_SHA / $SASS403_REAL / $SASS403_LOOP -- stop and read the SASS"; summary_exit; }

rm -f "$PY_BIN"; "$CODON" build -release "$PY_SRC" > "$LOGDIR/06_codon_${REV}.log" 2>&1
[[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN]" || { fail "codon_build" "$(tail -3 "$LOGDIR/06_codon_${REV}.log")"; summary_exit; }

find_input() { local n="$1" c; for lg in $(ls -t 40*_crunner_logs/crunner_*_N${n}.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { echo "$c"; return; }; done; ls -t constellations_N${n}_*.sched394f.bin 2>/dev/null | head -1; }
IN21="$(find_input 21)"; IN22="$(find_input 22)"
[[ -f "${IN21:-/nonexistent}" ]] && pass "input21_located ($IN21)" || { fail "input21_located" "no N=21 sched input"; summary_exit; }
[[ -f "${IN22:-/nonexistent}" ]] && pass "input22_located ($IN22)" || { fail "input22_located" "no N=22 sched input"; summary_exit; }

# ---------------------------------------------------------------------
# 3. CPU equivalence (Y2) and refusals (Y3)
# ---------------------------------------------------------------------
banner "Y3: refusals (kernel identical to 408, Y2 not repeated)"
"/tmp/${REV}_cpu_cur" 23 "$IN22" /tmp/${REV}_y3a.bin 8 > "$LOGDIR/2_y3_n23.log" 2>&1; rc=$?
[[ "$rc" == "3" ]] && pass "Y3_N23_refused_rc3" || fail "Y3_N23_refused_rc3" "rc=$rc"
NQ_LAYOUT=402 "/tmp/${REV}_cpu_cur" 22 "$IN22" /tmp/${REV}_y3b.bin 8 > "$LOGDIR/2_y3_l402_n22.log" 2>&1; rc=$?
[[ "$rc" == "3" ]] && pass "Y3_layout402_at_N22_refused_rc3" || fail "Y3_layout402_at_N22_refused_rc3" "rc=$rc"

# ---------------------------------------------------------------------
# 4. Cells
# ---------------------------------------------------------------------
printf 'cell\tbinary\tN\tlayout\thelpers\tfree_mb\tkernel_ms\tnorm_ms\ttotal_sum\tmatch\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
CLKPID=""
gpu_gate() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_before_$1.txt"; [[ -z "$apps" ]] && return 0; fail "gpu_empty_before[$1]" "compute process(es) present: $(echo "$apps" | tr '\n' ' ')"; return 1; }
gpu_after() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_after_$1.txt"; [[ -z "$apps" ]] || fail "gpu_residue_after[$1]" "$(echo "$apps" | tr '\n' ' ')"; }
clk_start() { local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"; ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPID=$!; }
clk_stop() { [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 && $6+0>50 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.1f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_$1.tsv"); }
declare -A VAL NORM SMM FREEMB
parse_log() {  # cell binlabel N logfile  -> sets KMS TOT MATCH FREE HL LAYV, appends TSV, checks oracle+layout
  local cell="$1" b="$2" n="$3" lg="$4" start="$5" want_layout="$6"
  KMS="$(grep -o 'kernel_ms=[0-9.]*' "$lg" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$lg" && MATCH=1
  FREE="$(grep -o 'free_mb=[0-9]*' "$lg" | head -1 | cut -d= -f2)"; HL="$(grep -o 'helpers=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  LAYV="$(grep -o '\[gpu-layout\] N=[0-9]* requested=[a-z0-9]* layout=[0-9]*' "$lg" | head -1 | grep -o 'layout=[0-9]*' | cut -d= -f2)"
  local NORMV; NORMV="$(awk -v k="${KMS:-0}" -v s="${SMMEAN/\?/0}" 'BEGIN{ if(s>0) printf "%.1f", k*s/1710; else print "?" }')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$b" "$n" "${LAYV:-?}" "${HL:-?}" "${FREE:-?}" "${KMS:-?}" "$NORMV" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start" >> "$TSV"
  local want; want="$([[ "$n" -le 21 ]] && echo "$ORACLE21" || echo "$ORACLE22")"
  [[ "${LAYV:-}" == "$want_layout" ]] || { fail "layout[$cell]" "layout=${LAYV:-?}, expected $want_layout"; return 1; }
  [[ "$MATCH" == "1" && "${TOT:-}" == "$want" ]] || { fail "oracle[$cell]" "total_sum='${TOT:-<none>}' match=$MATCH"; return 1; }
  VAL[$cell]="$KMS"; NORM[$cell]="$NORMV"; SMM[$cell]="$SMMEAN"; FREEMB[$cell]="${FREE:-0}"
  info "$cell" "$b N=$n layout=$LAYV helpers=${HL:-?} kernel_ms=$KMS norm_ms=$NORMV (sm_mean $SMMEAN) free_mb=$FREE"
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
banner "N=21 direct: B0 $PREV_BIN auto(403) / B1 $CU_BIN auto(403)"
run_direct B0 "$PREV_BIN" 21 auto 403 || summary_exit
info "B0_anchor" "$(pct "${VAL[B0]}" "$ANCHOR_B0")% vs 408 A0 $ANCHOR_B0 (same SASS; normalised $(pct "${NORM[B0]}" "$ANCHOR_B0")%)"
sleep "$COOLDOWN"; run_direct B1 "$CU_BIN" 21 auto 403 || summary_exit; same_bin B1 B0
info "B1_vs_B0" "raw $(pct "${VAL[B1]}" "${VAL[B0]}")% / normalised $(pct "${NORM[B1]}" "${NORM[B0]}")% (identical kernels: this is the session's scatter)"

OKG=0
if [[ "$SKIP_G" != "1" ]]; then
  banner "G21: ./$PY_BIN -g 21 21 (production path, env_prefix without NQ_EXTRA_CTX)"
  sleep "$COOLDOWN"; run_g G21 21 403 || summary_exit
  cp -f /tmp/crunner_N21_results.bin "$LOGDIR/bins/G21_results.bin" 2>/dev/null || true
  grep -q 'env_prefix=' "$LOGDIR/3_G21_dispatch.log" && pass "G21_dispatch_log_has_env_prefix" || { fail "G21_dispatch_log_has_env_prefix" "no env_prefix line"; ADOPT=0; }
  { grep 'env_prefix=' "$LOGDIR/3_G21_dispatch.log" | grep -q 'NQ_EXTRA_CTX'; } && { fail "G21_env_prefix_still_has_NQ_EXTRA_CTX" "$(grep 'env_prefix=' "$LOGDIR/3_G21_dispatch.log" | head -1)"; ADOPT=0; } || pass "G21_env_prefix_without_NQ_EXTRA_CTX ($(grep -o 'env_prefix=.*' "$LOGDIR/3_G21_dispatch.log" | head -1))"
  grep -q '\[gpu-ctx\] extra_ctx=0' "$LOGDIR/3_G21.log" && pass "G21_runner_extra_ctx_0" || { fail "G21_runner_extra_ctx_0" "$(grep -o 'extra_ctx=[0-9]*' "$LOGDIR/3_G21.log" | head -1)"; ADOPT=0; }
  grep -q 'helpers=1' "$LOGDIR/3_G21.log" && pass "G21_helper_present" || { fail "G21_helper_present" "no helpers=1"; ADOPT=0; }
  if [[ -s "$LOGDIR/bins/G21_results.bin" ]]; then cmp -s "$LOGDIR/bins/G21_results.bin" "$LOGDIR/bins/B0_results.bin" && info "G21_results_vs_B0" "identical" || info "G21_results_vs_B0" "differ (input order / stride may differ on the -g path; informational)"; fi
  ge "${FREEMB[G21]}" "$((ANCHOR_G21_FREE + 200))" && pass "G21_free_mb_recovered (${FREEMB[G21]} >= $ANCHOR_G21_FREE + 200)" || { fail "G21_free_mb_recovered" "free_mb ${FREEMB[G21]} (408 G21 had $ANCHOR_G21_FREE with the extra context)"; ADOPT=0; }
  info "G21_vs_408_G21" "$(pct "${VAL[G21]}" "$ANCHOR_G21")% vs $ANCHOR_G21 ; vs B1 raw $(pct "${VAL[G21]}" "${VAL[B1]}")% / normalised $(pct "${NORM[G21]}" "${NORM[B1]}")%"
  banner "G22: ./$PY_BIN -g 22 22 (production path)"
  sleep "$COOLDOWN"; run_g G22 22 403 || summary_exit
  { grep 'env_prefix=' "$LOGDIR/3_G22_dispatch.log" | grep -q 'NQ_EXTRA_CTX'; } && { fail "G22_env_prefix_still_has_NQ_EXTRA_CTX" "see dispatch log"; ADOPT=0; } || pass "G22_env_prefix_without_NQ_EXTRA_CTX"
  grep -q '\[gpu-ctx\] extra_ctx=0' "$LOGDIR/3_G22.log" && pass "G22_runner_extra_ctx_0" || { fail "G22_runner_extra_ctx_0" "$(grep -o 'extra_ctx=[0-9]*' "$LOGDIR/3_G22.log" | head -1)"; ADOPT=0; }
  pass "R3_G21_G22_oracle_layout403"
  info "G22_vs_408_G22" "raw $(pct "${VAL[G22]}" "$ANCHOR_G22")% / normalised $(pct "${NORM[G22]}" "$ANCHOR_G22")% vs $ANCHOR_G22 (408 G22, dispatcher)"
  d21="$(abspct "${NORM[G21]}" "${NORM[B1]}")"; d22="$(abspct "${NORM[G22]}" "$ANCHOR_G22")"
  if le "$d21" 0.3 && le "$d22" 0.5; then pass "R1_no_change_within_band (G21 vs B1 ${d21}% <= 0.3, G22 vs 408 ${d22}% <= 0.5, normalised)"; OKG=1
  else fail "R1_outside_band" "G21 vs B1 ${d21}% (<= 0.3), G22 vs 408 G22 ${d22}% (<= 0.5) -- read the clock logs before deciding"; fi
else
  info "SKIP_G" "production-path cells skipped: not adoptable"
fi
[[ "$OKG" == "1" ]] || ADOPT=0
CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "R4_sm_clock_1710_all_cells" || { fail "R4_sm_clock_1710_all_cells" "see clk_*.tsv"; ADOPT=0; }
if [[ "$ADOPT" == "1" ]]; then
  info "ADOPTION" "411 is the production binary/.py: N=21 = ${VAL[G21]} ms (norm ${NORM[G21]}), N=22 = ${VAL[G22]} ms (norm ${NORM[G22]}); env_prefix without NQ_EXTRA_CTX"
else
  info "ADOPTION" "NOT adopted -- production stays 408. See FAIL / INFO lines."
fi

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
ref={r['cell']:float(r['kernel_ms']) for r in rows}
print(f"{'cell':<5}{'binary':<32}{'N':>3}{'lay':>5}{'help':>5}{'free_mb':>8}{'kernel_ms':>14}{'vs ref':>10}{'norm_ms':>12}{'match':>6}{'sm':>8}{'h:mm:ss':>10}")
for r in rows:
    v=float(r['kernel_ms']); s=v/1000; c=r['cell']
    base=ref.get('B0') if c.startswith('B') else 98024.9 if c=='G21' else 854582.1
    print(f"{c:<5}{r['binary'][:31]:<32}{r['N']:>3}{r['layout']:>5}{r['helpers']:>5}{r['free_mb']:>8}{v:>14.3f}{(v/base-1)*100:>+9.3f}%{r['norm_ms']:>12}{r['match']:>6}{r['sm_mean']:>8}{int(s//3600):>4}:{int(s%3600//60):02d}:{s%60:04.1f}")
print("(B1 vs B0, G21 vs 408 G21 98,025, G22 vs 408 G22 854,582)")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd14" --exclude="*_cpu" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
