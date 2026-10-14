#!/usr/bin/env bash
# 413_validate.sh
#
# rev413 -- N=23: one binary (413_kernel_maxd16) for N <= 23. N <= 22 keeps 408/411's kernel
#   (<402>/<403> SASS identical, HARD); N = 23 runs the new 12-byte layout 423 with 14 ancestor
#   frames (168 B/thread). No production value changes in this revision; N=23's full run is
#   413-r3 (or RUN_FULL23=1 here).
#
#   S0 static: .py = 411Py + 5 executable lines (VERSION_TAG, REV_TAG, CRunnerEntry(15, ./413_kernel_maxd16),
#      N>=23 filter script + suffix); .cu fingerprint vs 411 (removed=23 added=87), kernel sha 101cbdbb;
#      PACK423_ANCESTOR 14; filter MAXDEPTH 15; notes; scripts present; gcc builds (413 / 411 / 392 CPU)
#   S1 ptxas: <402>,<403> 160 B / spill 0 / regs <= 44 ; <423> frame <= 176 B / spill 0 / regs <= 44    HARD
#   S2/S3 <402> 821302f2 / 446 / 109, <403> e940cb96 / 445 / 108  (== 408)                             HARD
#   S4 <423> SASS stats (sha / real / loop) recorded; stated: loop <= 110 (info)
#   N=23 input (first run only, 40-60 min, under 367_safe_run_wrapper.sh when present):
#      361 dump (413Py bench_mode=32) -> 413 filter (depth <= 15) -> 394f sched
#   H1 filter: records_kept == records_in and max_depth <= 15 (= PACK423_ANCESTOR + 1)                  HARD
#   H2 sched file has EXPECTED23 = 44,271,796 records                                                    HARD
#   Y2 CPU equivalence (8 threads): N=21 auto 413 == 411 (2,048) ; N=21 NQ_LAYOUT=423 413 == 411 (2,048) ;
#      N=23 413 (423) == 392_kernel_maxd16 CPU harness on 2,048 equal-spaced records                     HARD
#   Y3 rc=3: N=24 ; NQ_LAYOUT=403 at N=23 ; NQ_LAYOUT=402 at N=22                                       HARD
#   Cells (direct = BLOCK 32, helper 1 + 128 MB, NQ_EXTRA_CTX unset):
#     B0   411 N=21 auto(403) MB 960           results.bin == REF21 (411 B1 1bfab090)                   HARD
#     B1   413 N=21 auto(403) MB 960           results.bin == B0 (same SASS)                             HARD
#     Z1   413 N=21 NQ_LAYOUT=423 MB 960       results.bin == B0 ; time = cost of 423 at N=21 (info)      HARD
#     R392 392 N=23 slice A (309,760 = 20/thread @ 15,488), fixed 484x32, no helper
#     C484 413 N=23 slice A, NQ_MAX_BLOCKS=484  results.bin == R392 (same stride)                         HARD
#     C960 / C880 / C960r  413 N=23 slice B (3,072,000 = 100/thread @ 30,720)  total_sum all equal       HARD
#   PRE-REGISTERED: no gain claimed. Z1/B1 <= 1.01 (stated, info). W1 C880/C960: no direction stated;
#     <= 0.998 -> N=23 runs at 880 (413-r2 puts an N-dependent MB in the .py); otherwise 960.
#     W2 |C960r - C960| <= 0.3% (normalised). W3 all cells 1710 +- 2%.
#   ADOPT: all HARD -> 413 is the production binary/.py (N=21/22 values stay 411's; N=23's value = 413-r3).
#   RUN_FULL23=1: G23 = ./413Py -g 23 23 under the wrapper, oracle 24,233,937,684,440 HARD (est. 40-70 min).
#
# USAGE
#   STATIC_ONLY=1 bash 413_validate.sh      # OK=25, GPU untouched
#                 bash 413_validate.sh      # + build (first time: + N=23 input), CPU ~10 min, GPU ~25 min
#   RUN_FULL23=1  bash 413_validate.sh      # + the full N=23 run at the end

set -u

REV="413"
PY_SRC="${PY_SRC:-413Py_kernel_maxd16_final.py}"
PY_BIN="${PY_BIN:-${PY_SRC%.py}}"
PREV_PY="${PREV_PY:-411Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-413_kernel_maxd16.cu}"
CU_BIN="${CU_BIN:-413_kernel_maxd16}"
PREV_CU="${PREV_CU:-411_kernel_maxd14.cu}"
PREV_BIN="${PREV_BIN:-411_kernel_maxd14}"
REF_CU="${REF_CU:-392_kernel_maxd16.cu}"
REF_BIN="${REF_BIN:-392_kernel_maxd16}"
FILTER="${FILTER:-413_filter_maxd16_only.py}"
SLICER="${SLICER:-413_slice_soa7.py}"
PERMUTE="${PERMUTE:-394f_permute_soa7.py}"
WRAPPER="${WRAPPER:-367_safe_run_wrapper.sh}"
REF21="${REF21:-411_validate_20261006_150938/bins/B1_results.bin}"
REF21_SHA="${REF21_SHA:-1bfab090}"
STREAM23="${STREAM23:-constellations_N23_7.bin}"
EXPECTED23="${EXPECTED23:-44271796}"
SASS402_SHA="${SASS402_SHA:-821302f227dcd160}"; SASS402_REAL="${SASS402_REAL:-446}"; SASS402_LOOP="${SASS402_LOOP:-109}"
SASS403_SHA="${SASS403_SHA:-e940cb96e5e65fc4}"; SASS403_REAL="${SASS403_REAL:-445}"; SASS403_LOOP="${SASS403_LOOP:-108}"
KERNEL_SHA_413="${KERNEL_SHA_413:-101cbdbbcb6316d0}"
KERNEL_SHA_411="${KERNEL_SHA_411:-ab483e70328c39c6}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
CODON="${CODON:-codon}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
HMB="${HMB:-128}"
NREC="${NREC:-2048}"
CPU23="${CPU23:-2048}"
SLICE_A="${SLICE_A:-309760}"      # 20 records per thread at stride 15,488 (392's fixed 484 x 32)
SLICE_B="${SLICE_B:-3072000}"     # 100 records per thread at stride 30,720
COOLDOWN="${COOLDOWN:-15}"
CLK_INTERVAL="${CLK_INTERVAL:-5}"
STATIC_ONLY="${STATIC_ONLY:-0}"
RUN_FULL23="${RUN_FULL23:-0}"
OMP_THREADS="${OMP_THREADS:-8}"
ORACLE21=314666222712
ORACLE22=2691008701644
ORACLE23=24233937684440

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"
TSV="$LOGDIR/${REV}_results.tsv"
CRLOG="${REV}_crunner_logs"

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
for f in "$PY_SRC" "$HELPER_SRC" "$CU_SRC" "$PREV_CU" "$REF_CU" "$FILTER" "$SLICER" "$PERMUTE"; do
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
{ grep -q "^# 413 " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 413 ...' note block"
grep -qE '^REV_TAG:str="413"' "$CODE" && pass "source_rev_tag_is_413" || fail "source_rev_tag_is_413" "wrong REV_TAG"
grep -qF 'CRunnerEntry(15,"./413_kernel_maxd16","NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_entry_maxd15_413_binary_helper_prefix" || fail "source_table_entry" "table entry wrong"
{ ! grep -E 'CRunnerEntry\(' "$CODE" | grep -q 'NQ_EXTRA_CTX'; } && pass "source_no_table_entry_carries_NQ_EXTRA_CTX" || fail "source_no_table_entry_carries_NQ_EXTRA_CTX" "found"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=960\b" "$CODE" && pass "source_default_max_blocks_is_960" || fail "source_default_max_blocks_is_960" "default not 960"
grep -qF 'else "413_filter_maxd16_only.py"' "$CODE" && grep -qF 'maxd16only_413.bin' "$CODE" && pass "source_n23_filter_and_suffix_wired" || fail "source_n23_filter_and_suffix_wired" "missing"
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "5" && "$PA_" == "5" ]] && pass "py_diff_fingerprint_vs_411Py (removed=5 added=5 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_411Py" "removed=$PR_ added=$PA_, expected 5/5"
else info "py_diff_fingerprint_vs_411Py" "skipped ($PREV_PY absent)"; fi
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
[[ "$KB" == "$KERNEL_SHA_413"* ]] && pass "cu_kernel_region_sha (${KB:0:16})" || fail "cu_kernel_region_sha" "sha ${KB:0:16} != $KERNEL_SHA_413"
KP="$(extract_kernel "/tmp/${REV}_prev_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KP" == "$KERNEL_SHA_411"* ]] && pass "prev_cu_is_411 (${KP:0:16})" || fail "prev_cu_is_411" "sha ${KP:0:16} != $KERNEL_SHA_411"
CR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true); CA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
[[ "$CR_" == "23" && "$CA_" == "87" ]] && pass "cu_code_region_fingerprint_vs_411 (removed=23 added=87)" || fail "cu_code_region_fingerprint" "removed=$CR_ added=$CA_, expected 23/87"
grep -q '^#define PACK423_ANCESTOR 14' "$CU_SRC" && pass "cu_pack423_ancestor_is_14" || fail "cu_pack423_ancestor_is_14" "missing"
grep -q '^#define MAXD14_ANCESTOR 13' "$CU_SRC" && pass "cu_maxd14_ancestor_still_13" || fail "cu_maxd14_ancestor_still_13" "changed"
grep -q 'MAXDEPTH = int(os.environ.get("MAXDEPTH", "15"))' "$FILTER" && pass "filter_maxdepth_default_15" || fail "filter_maxdepth_default_15" "wrong default"
grep -q "rev413 " "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev413 note in header"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"/tmp/${REV}_gcc.log" && pass "cu_cpu_harness_builds[413]" || fail "cu_cpu_harness_builds[413]" "$(head -3 /tmp/${REV}_gcc.log)"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_prev" "$PREV_CU" -lm 2>"/tmp/${REV}_gcc_prev.log" && pass "cu_cpu_harness_builds[411]" || fail "cu_cpu_harness_builds[411]" "$(head -3 /tmp/${REV}_gcc_prev.log)"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_ref" "$REF_CU" -lm 2>"/tmp/${REV}_gcc_ref.log" && pass "cu_cpu_harness_builds[392]" || fail "cu_cpu_harness_builds[392]" "$(head -3 /tmp/${REV}_gcc_ref.log)"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Build, ptxas (S1), SASS (S2-S4), codon, reference binaries
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR/sass" "$LOGDIR/bins" "$CRLOG"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; df -h . 2>&1; free -g 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
[[ ! -x "$NVCC" ]] && NVCC="nvcc"; [[ ! -x "$CUOBJDUMP" ]] && CUOBJDUMP="cuobjdump"
banner "Building"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc_${REV}.log" 2>&1
[[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "$(grep -i error "$LOGDIR/05_nvcc_${REV}.log" | head -3)"; summary_exit; }
ptx_stats() {  # mangled-fragment -> FR SP RG
  FR="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"
  SP="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes spill stores, [0-9]* bytes spill loads' | head -1)"
  RG="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"
}
for L in 402 403; do
  ptx_stats "kernel_dfs_iter_gpu_maxd16ILi${L}E"
  [[ "$FR" == "160" && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "S1_ptxas_L${L}_frame160_spill0_regs${RG}" || { fail "S1_ptxas_L${L}" "frame=${FR}B regs=${RG} ${SP}"; summary_exit; }
done
ptx_stats "kernel_dfs_iter_gpu_maxd16ILi423E"
[[ -n "$FR" && "$FR" -le 176 && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "S1_ptxas_L423_frame${FR}_spill0_regs${RG} (<= 176 B: 12 blocks/SM = $((FR*384)) B of local)" || { fail "S1_ptxas_L423" "frame=${FR}B regs=${RG} ${SP} (expected <= 176 / 0 / <= 44)"; summary_exit; }
FRAME423="$FR"
if [[ ! -x "$PREV_BIN" ]]; then "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$PREV_BIN" "$PREV_CU" -lcuda > "$LOGDIR/05b_nvcc_prev.log" 2>&1; fi
[[ -x "$PREV_BIN" ]] && pass "prev_binary_available[$PREV_BIN]" || { fail "prev_binary" "cannot build $PREV_BIN"; summary_exit; }
if [[ ! -x "$REF_BIN" ]]; then "$NVCC" -O3 -arch="$ARCH" -o "$REF_BIN" "$REF_CU" > "$LOGDIR/05c_nvcc_ref.log" 2>&1; fi
[[ -x "$REF_BIN" ]] && pass "ref_binary_available[$REF_BIN]" || { fail "ref_binary" "cannot build $REF_BIN from $REF_CU: $(grep -i error "$LOGDIR/05c_nvcc_ref.log" | head -2)"; summary_exit; }
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
best=None
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
read -r S_SHA S_REAL S_LOOP S_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_dfs_iter_gpu_maxd16ILi402E)"
[[ "$S_SHA" == "$SASS402_SHA" && "$S_REAL" == "$SASS402_REAL" && "$S_LOOP" == "$SASS402_LOOP" ]] && pass "S2_sass_402_is_408s (sha $S_SHA, real $S_REAL, loop $S_LOOP)" || { fail "S2_sass_402_differs_from_408" "sha $S_SHA real $S_REAL loop $S_LOOP != $SASS402_SHA / $SASS402_REAL / $SASS402_LOOP -- the ANC/mask macros perturbed the <402> instantiation: stop and read the SASS"; summary_exit; }
read -r S_SHA S_REAL S_LOOP S_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_dfs_iter_gpu_maxd16ILi403E)"
[[ "$S_SHA" == "$SASS403_SHA" && "$S_REAL" == "$SASS403_REAL" && "$S_LOOP" == "$SASS403_LOOP" ]] && pass "S3_sass_403_is_408s (sha $S_SHA, real $S_REAL, loop $S_LOOP)" || { fail "S3_sass_403_differs_from_408" "sha $S_SHA real $S_REAL loop $S_LOOP != $SASS403_SHA / $SASS403_REAL / $SASS403_LOOP -- stop and read the SASS"; summary_exit; }
read -r S_SHA S_REAL S_LOOP S_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_dfs_iter_gpu_maxd16ILi423E)"
[[ "$S_SHA" != "CANNOT_LOCATE" ]] && pass "S4_sass_423_recorded (sha $S_SHA, real $S_REAL, loop $S_LOOP, frame ${FRAME423} B)" || { fail "S4_sass_423_missing" "no <423> instantiation in the SASS"; summary_exit; }
if [[ "$S_LOOP" -le 110 ]]; then info "S4_loop_423" "loop $S_LOOP <= 110 (stated: <403> 108 + 0..2)"; else info "S4_loop_423_above_statement" "loop $S_LOOP > 110 -- read the pop/push of <423> before 413-r2"; fi

rm -f "$PY_BIN"; "$CODON" build -release "$PY_SRC" > "$LOGDIR/06_codon_${REV}.log" 2>&1
[[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN]" || { fail "codon_build" "$(tail -3 "$LOGDIR/06_codon_${REV}.log")"; summary_exit; }

find_input() { local n="$1" c; for lg in $(ls -t 4*_crunner_logs/crunner_*_N${n}.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { echo "$c"; return; }; done; ls -t constellations_N${n}_*.sched394f.bin 2>/dev/null | head -1; }
IN21="$(find_input 21)"
[[ -f "${IN21:-/nonexistent}" ]] && pass "input21_located ($IN21)" || { fail "input21_located" "no N=21 sched input"; summary_exit; }
IN22="$(find_input 22)"; [[ -f "${IN22:-/nonexistent}" ]] && info "input22_located" "$IN22" || info "input22_located" "none (only needed for the Y3 rc=3 case; a 28-byte stub is used instead)"
if [[ -f "$REF21" ]]; then
  RS="$(sha256sum "$REF21" | cut -c1-8)"
  [[ "$RS" == "$REF21_SHA" ]] && pass "ref21_is_411_B1 ($REF21 sha $RS)" || fail "ref21_is_411_B1" "sha $RS != $REF21_SHA"
else fail "ref21_present" "$REF21 missing -- pass REF21=<411 B1 results.bin>"; fi

# ---------------------------------------------------------------------
# 3. N=23 input: 361 dump -> 413 filter -> 394f sched (first run only)
# ---------------------------------------------------------------------
banner "N=23 input"
[[ -f "$STREAM23" && -f "$STREAM23.done" ]] && pass "stream23_present ($STREAM23, $(( $(stat -c %s "$STREAM23") / 16 )) records)" || { fail "stream23_present" "$STREAM23 / .done missing"; summary_exit; }
SOA23="${STREAM23}.soa_ref_361.bin"; FILT23="${SOA23}.maxd16only_413.bin"; SCHED23="${FILT23}.sched394f.bin"
FLOG="$CRLOG/filter_413_N23.log"
wrap() { if [[ -f "$WRAPPER" ]]; then MAX_MEM_PERCENT="${MAX_MEM_PERCENT:-70}" TIMEOUT_DURATION="${TIMEOUT_DURATION:-3h}" MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-10}" bash "$WRAPPER" -- "$@"; else "$@"; fi; }
if [[ ! -s "$SOA23" ]]; then
  info "n23_soa_ref" "building $SOA23 with $PY_BIN bench_mode=32 (Codon 361 dump, ~1.8 GB, 20-40 min)"
  wrap "./$PY_BIN" -g 23 23 32 960 1 0 7 32 > "$LOGDIR/10_soa_ref_N23.log" 2>&1 || true
  grep -h 'soa-ref-dump-done' "$LOGDIR/10_soa_ref_N23.log" 367_wrapped_*.log 2>/dev/null | tail -1
fi
[[ -s "$SOA23" && $(( $(stat -c %s "$SOA23") % 40 )) -eq 0 ]] && pass "n23_soa_ref_present ($(( $(stat -c %s "$SOA23") / 40 )) records)" || { fail "n23_soa_ref" "$SOA23 missing or not a multiple of 40 bytes"; summary_exit; }
if [[ ! -s "$FILT23" || ! -f "$FLOG" ]]; then
  info "n23_filter" "running python3 $FILTER $SOA23 $FILT23 (minutes)"
  wrap python3 "$FILTER" "$SOA23" "$FILT23" > "$FLOG" 2>&1 || true
  grep -h '^\[filter-' "$FLOG" 367_wrapped_*.log 2>/dev/null | grep -v progress | tail -2
  cp -f "$FLOG" "$LOGDIR/11_filter_N23.log" 2>/dev/null || true
fi
FD="$(grep -h '^\[filter-done\]' "$FLOG" 367_wrapped_*.log 2>/dev/null | tail -1)"
F_IN="$(echo "$FD" | grep -o 'records_in=[0-9]*' | cut -d= -f2)"; F_KEPT="$(echo "$FD" | grep -o 'records_kept([^)]*)=[0-9]*' | grep -o '[0-9]*$')"; F_MAX="$(echo "$FD" | grep -o 'max_depth=[0-9]*' | cut -d= -f2)"
[[ -n "$F_IN" && "$F_IN" == "$F_KEPT" && "$F_IN" == "$EXPECTED23" ]] && pass "H1_filter_kept_every_record ($F_KEPT of $F_IN)" || { fail "H1_filter_kept_every_record" "$FD"; summary_exit; }
[[ -n "$F_MAX" && "$F_MAX" -le 15 ]] && pass "H1_max_schedule_depth_${F_MAX}_le_15 (= PACK423_ANCESTOR + 1)" || { fail "H1_max_schedule_depth" "max_depth='${F_MAX:-?}' > 15: PACK423_ANCESTOR=14 is NOT enough -- do not run the GPU"; summary_exit; }
info "H1_depth_histogram" "$(grep -h '^\[filter-depth-histogram\]' "$FLOG" 367_wrapped_*.log 2>/dev/null | tail -1)"
if [[ ! -s "$SCHED23" ]]; then
  info "n23_sched" "running python3 $PERMUTE sched $FILT23 $SCHED23 (44M-record sort, minutes, several GB of RAM)"
  wrap python3 "$PERMUTE" sched "$FILT23" "$SCHED23" > "$CRLOG/permute_sched_N23.log" 2>&1 || true
  grep -h '^\[permute-done\]' "$CRLOG/permute_sched_N23.log" 367_wrapped_*.log 2>/dev/null | tail -1
  cp -f "$CRLOG/permute_sched_N23.log" "$LOGDIR/12_permute_N23.log" 2>/dev/null || true
fi
[[ -s "$SCHED23" && $(( $(stat -c %s "$SCHED23") / 28 )) -eq "$EXPECTED23" && $(( $(stat -c %s "$SCHED23") % 28 )) -eq 0 ]] && pass "H2_sched23_has_${EXPECTED23}_records ($SCHED23)" || { fail "H2_sched23" "$SCHED23 missing or wrong size"; summary_exit; }
IN23="$SCHED23"
SLA="${REV}_n23_sliceA_${SLICE_A}.bin"; SLB="${REV}_n23_sliceB_${SLICE_B}.bin"; SLC="${REV}_n23_sliceC_${CPU23}.bin"
for pair in "$SLA:$SLICE_A" "$SLB:$SLICE_B" "$SLC:$CPU23"; do f="${pair%%:*}"; c="${pair##*:}"; if [[ ! -s "$f" || $(( $(stat -c %s "$f") / 28 )) -ne "$c" ]]; then python3 "$SLICER" "$IN23" "$f" "$c" > "$LOGDIR/13_slice_${c}.log" 2>&1 || true; fi; [[ -s "$f" && $(( $(stat -c %s "$f") / 28 )) -eq "$c" ]] && pass "slice_ready[$f]" || { fail "slice[$f]" "$(tail -1 "$LOGDIR/13_slice_${c}.log" 2>/dev/null)"; summary_exit; }; done

# ---------------------------------------------------------------------
# 4. CPU equivalence (Y2) and refusals (Y3)
# ---------------------------------------------------------------------
banner "Y2: CPU equivalence ($OMP_THREADS threads)"
export OMP_NUM_THREADS="$OMP_THREADS"
head -c $((NREC*28)) "$IN21" > "/tmp/${REV}_in21_head.bin"
"/tmp/${REV}_cpu_prev" 21 "/tmp/${REV}_in21_head.bin" "/tmp/${REV}_y2_411_21.bin" "$NREC" > "$LOGDIR/2_cpu_N21_411.log" 2>&1
"/tmp/${REV}_cpu_cur"  21 "/tmp/${REV}_in21_head.bin" "/tmp/${REV}_y2_413_21.bin" "$NREC" > "$LOGDIR/2_cpu_N21_413auto.log" 2>&1
cmp -s "/tmp/${REV}_y2_411_21.bin" "/tmp/${REV}_y2_413_21.bin" && pass "Y2a_cpu_N21_auto_413_equals_411 ($NREC records)" || { fail "Y2a_cpu_N21_auto" "per-record outputs differ"; summary_exit; }
NQ_LAYOUT=423 "/tmp/${REV}_cpu_cur" 21 "/tmp/${REV}_in21_head.bin" "/tmp/${REV}_y2_413x_21.bin" "$NREC" > "$LOGDIR/2_cpu_N21_413l423.log" 2>&1
grep -q 'layout=423' "$LOGDIR/2_cpu_N21_413l423.log" && cmp -s "/tmp/${REV}_y2_411_21.bin" "/tmp/${REV}_y2_413x_21.bin" && pass "Y2b_cpu_N21_layout423_equals_411 ($NREC records)" || { fail "Y2b_cpu_N21_layout423" "differ or layout not 423"; summary_exit; }
T0=$(date +%s)
"/tmp/${REV}_cpu_ref" 23 "$SLC" "/tmp/${REV}_y2_392_23.bin" > "$LOGDIR/2_cpu_N23_392.log" 2>&1; T1=$(date +%s)
"/tmp/${REV}_cpu_cur" 23 "$SLC" "/tmp/${REV}_y2_413_23.bin" > "$LOGDIR/2_cpu_N23_413.log" 2>&1; T2=$(date +%s)
grep -q 'layout=423' "$LOGDIR/2_cpu_N23_413.log" && cmp -s "/tmp/${REV}_y2_392_23.bin" "/tmp/${REV}_y2_413_23.bin" && pass "Y2c_cpu_N23_413_equals_392 ($CPU23 equal-spaced records; 392 $((T1-T0)) s, 413 $((T2-T1)) s; total_sum $(grep -o 'total_sum=[0-9]*' "$LOGDIR/2_cpu_N23_413.log" | head -1 | cut -d= -f2))" || { fail "Y2c_cpu_N23_413_vs_392" "per-record outputs differ (or abort: $(grep -o '\[[A-Z-]*\]' "$LOGDIR/2_cpu_N23_413.log" | head -1)) -- the 423 layout or the 14-frame bound is wrong; do not run the GPU"; summary_exit; }
banner "Y3: refusals"
"/tmp/${REV}_cpu_cur" 24 "$SLC" /tmp/${REV}_y3a.bin 8 > "$LOGDIR/2_y3_n24.log" 2>&1; rc=$?
[[ "$rc" == "3" ]] && pass "Y3_N24_refused_rc3" || fail "Y3_N24_refused_rc3" "rc=$rc"
NQ_LAYOUT=403 "/tmp/${REV}_cpu_cur" 23 "$SLC" /tmp/${REV}_y3b.bin 8 > "$LOGDIR/2_y3_l403_n23.log" 2>&1; rc=$?
[[ "$rc" == "3" ]] && pass "Y3_layout403_at_N23_refused_rc3" || fail "Y3_layout403_at_N23_refused_rc3" "rc=$rc"
STUB22="${IN22:-/tmp/${REV}_in21_head.bin}"
NQ_LAYOUT=402 "/tmp/${REV}_cpu_cur" 22 "$STUB22" /tmp/${REV}_y3c.bin 8 > "$LOGDIR/2_y3_l402_n22.log" 2>&1; rc=$?
[[ "$rc" == "3" ]] && pass "Y3_layout402_at_N22_refused_rc3" || fail "Y3_layout402_at_N22_refused_rc3" "rc=$rc"

# ---------------------------------------------------------------------
# 5. Cells
# ---------------------------------------------------------------------
printf 'cell\tbinary\tN\tlayout\tblock\tmax_blocks\trecords\thelpers\tfree_mb\tkernel_ms\tnorm_ms\ttotal_sum\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
CLKPID=""
gpu_gate() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_before_$1.txt"; [[ -z "$apps" ]] && return 0; fail "gpu_empty_before[$1]" "compute process(es) present: $(echo "$apps" | tr '\n' ' ')"; return 1; }
gpu_after() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_after_$1.txt"; [[ -z "$apps" ]] || fail "gpu_residue_after[$1]" "$(echo "$apps" | tr '\n' ' ')"; }
clk_start() { local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"; ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPID=$!; }
clk_stop() { [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 && $6+0>50 && $4+0>150 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.1f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_$1.tsv"); }
declare -A VAL NORM SMM TOTAL
parse_log() {  # cell binlabel N logfile start want_layout block mb records
  local cell="$1" b="$2" n="$3" lg="$4" start="$5" want_layout="$6" blk="$7" mb="$8" recs="$9"
  local KMS TOT MATCHL FREE HL LAYV XC CFG NORMV
  KMS="$(grep -o 'kernel_ms=[0-9.]*' "$lg" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  FREE="$(grep -o 'free_mb=[0-9]*' "$lg" | head -1 | cut -d= -f2)"; HL="$(grep -o 'helpers=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  LAYV="$(grep -o '\[gpu-layout\] N=[0-9]* requested=[a-z0-9]* layout=[0-9]*' "$lg" | head -1 | grep -o 'layout=[0-9]*' | cut -d= -f2)"
  XC="$(grep -o 'extra_ctx=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  CFG="$(grep -o '\[gpu-config\] BLOCK=[0-9]* MAX_BLOCKS=[0-9]*' "$lg" | head -1)"
  NORMV="$(awk -v k="${KMS:-0}" -v s="${SMMEAN/\?/0}" 'BEGIN{ if(s>0) printf "%.1f", k*s/1710; else print "?" }')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$b" "$n" "${LAYV:-?}" "$blk" "$mb" "$recs" "${HL:-?}" "${FREE:-?}" "${KMS:-?}" "$NORMV" "${TOT:-?}" "$SMMEAN" "$SMMIN" "$TMAX" "$start" >> "$TSV"
  [[ -z "$want_layout" || "${LAYV:-}" == "$want_layout" ]] || { fail "layout[$cell]" "layout=${LAYV:-?}, expected $want_layout"; return 1; }
  [[ -z "$want_layout" || ( "${HL:-}" == "1" && "${XC:-}" == "0" ) ]] || { fail "ctx_state[$cell]" "helpers=${HL:-?} extra_ctx=${XC:-?}"; return 1; }
  [[ -z "$want_layout" || "$CFG" == "[gpu-config] BLOCK=$blk MAX_BLOCKS=$mb" ]] || { fail "config[$cell]" "'${CFG:-<none>}' != BLOCK=$blk MAX_BLOCKS=$mb"; return 1; }
  [[ -n "${KMS:-}" && -n "${TOT:-}" ]] || { fail "run[$cell]" "no kernel_ms/total_sum in $lg: $(tail -2 "$lg" | tr '\n' ' ')"; return 1; }
  VAL[$cell]="$KMS"; NORM[$cell]="$NORMV"; SMM[$cell]="$SMMEAN"; TOTAL[$cell]="$TOT"
  info "$cell" "$b N=$n layout=${LAYV:-?} BLOCK=$blk MB=$mb records=$recs kernel_ms=$KMS norm_ms=$NORMV (sm_mean $SMMEAN, ${TMAX}C) total_sum=$TOT free_mb=${FREE:-?}"
}
run_413() {  # cell binary N input layout_env block mb want_layout [oracle]
  local cell="$1" bin="$2" n="$3" in="$4" lenv="$5" blk="$6" mb="$7" want="$8" oracle="${9:-}"; gpu_gate "$cell" || return 1
  local lg="$LOGDIR/3_${cell}.log" start recs; start="$(date -Is)"; recs=$(( $(stat -c %s "$in") / 28 )); clk_start "$cell"
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX NQ_LAYOUT="$lenv" NQ_BLOCK="$blk" NQ_MAX_BLOCKS="$mb" NQ_HELPER_CTX=1 NQ_HELPER_MB="$HMB" "./$bin" "$n" "$in" "$LOGDIR/bins/${cell}_results.bin" $oracle > "$lg" 2>&1 || true
  clk_stop "$cell"; gpu_after "$cell"
  parse_log "$cell" "$bin" "$n" "$lg" "$start" "$want" "$blk" "$mb" "$recs"
}
run_392() {  # cell input   (392: fixed 484 x 32, no env knobs, no helper)
  local cell="$1" in="$2"; gpu_gate "$cell" || return 1
  local lg="$LOGDIR/3_${cell}.log" start recs; start="$(date -Is)"; recs=$(( $(stat -c %s "$in") / 28 )); clk_start "$cell"
  "./$REF_BIN" 23 "$in" "$LOGDIR/bins/${cell}_results.bin" > "$lg" 2>&1 || true
  clk_stop "$cell"; gpu_after "$cell"
  parse_log "$cell" "$REF_BIN" 23 "$lg" "$start" "" 32 484 "$recs"
}
same_bin() { local a="$LOGDIR/bins/${1}_results.bin" b="$2"; [[ -f "$b" ]] || b="$LOGDIR/bins/${2}_results.bin"; cmp -s "$a" "$b" && pass "${1}_per_thread_results_identical_to_${2} ($(stat -c %s "$a") bytes)" || fail "${1}_per_thread_results_DIFFER_from_${2}" "not adoptable"; }
oracle_ok() { [[ "${TOTAL[$1]:-}" == "$2" ]] && pass "oracle[$1]" || fail "oracle[$1]" "total_sum=${TOTAL[$1]:-?} != $2"; }

banner "N=21: B0 $PREV_BIN / B1 $CU_BIN / Z1 $CU_BIN NQ_LAYOUT=423"
run_413 B0 "$PREV_BIN" 21 "$IN21" auto 32 960 403 "$ORACLE21" || summary_exit; oracle_ok B0 "$ORACLE21"; same_bin B0 "$REF21"
sleep "$COOLDOWN"; run_413 B1 "$CU_BIN" 21 "$IN21" auto 32 960 403 "$ORACLE21" || summary_exit; oracle_ok B1 "$ORACLE21"; same_bin B1 B0
info "B1_vs_B0" "raw $(pct "${VAL[B1]}" "${VAL[B0]}")% / normalised $(pct "${NORM[B1]}" "${NORM[B0]}")% (identical SASS: session scatter)"
sleep "$COOLDOWN"; run_413 Z1 "$CU_BIN" 21 "$IN21" 423 32 960 423 "$ORACLE21" || summary_exit; oracle_ok Z1 "$ORACLE21"; same_bin Z1 B0
r="$(ratio "${NORM[Z1]}" "${NORM[B1]}")"; info "Z1_423_at_N21" "Z1/B1 = $r (normalised; stated <= 1.01: the 423 pop/push and the 168 B frame at N=21, informational)"
le "$r" 1.01 && pass "Z1_stated_band_held ($r <= 1.01)" || info "Z1_stated_band_missed" "$r > 1.01 -- read the <423> loop before 413-r2"

banner "N=23 slice A ($SLICE_A records): R392 $REF_BIN (484x32) / C484 $CU_BIN @484"
sleep "$COOLDOWN"; run_392 R392 "$SLA" || summary_exit
sleep "$COOLDOWN"; run_413 C484 "$CU_BIN" 23 "$SLA" auto 32 484 423 || summary_exit
[[ "${TOTAL[C484]}" == "${TOTAL[R392]}" ]] && pass "C484_total_equals_R392 (${TOTAL[C484]})" || fail "C484_total_differs_from_R392" "${TOTAL[C484]} != ${TOTAL[R392]} -- not adoptable"
same_bin C484 R392
info "C484_vs_R392_time" "413@484 $(pct "${NORM[C484]}" "${NORM[R392]}")% vs 392 (old 240 B frame, 1 context; informational only)"

banner "N=23 slice B ($SLICE_B records): C960 / C880 / C960r"
sleep "$COOLDOWN"; run_413 C960 "$CU_BIN" 23 "$SLB" auto 32 960 423 || summary_exit
sleep "$COOLDOWN"; run_413 C880 "$CU_BIN" 23 "$SLB" auto 32 880 423 || summary_exit
sleep "$COOLDOWN"; run_413 C960r "$CU_BIN" 23 "$SLB" auto 32 960 423 || summary_exit
[[ "${TOTAL[C880]}" == "${TOTAL[C960]}" && "${TOTAL[C960r]}" == "${TOTAL[C960]}" ]] && pass "sliceB_totals_equal_across_MB (${TOTAL[C960]})" || fail "sliceB_totals_differ" "C960 ${TOTAL[C960]} C880 ${TOTAL[C880]} C960r ${TOTAL[C960r]} -- not adoptable"
same_bin C960r C960
r="$(ratio "${NORM[C880]}" "${NORM[C960]}")"
if le "$r" 0.998; then info "W1_880_faster" "C880/C960 = $r -> N=23 runs at MB 880 (413-r2: N-dependent MB in the .py)"
elif le "$r" 1.002; then info "W1_no_difference" "C880/C960 = $r -> 960 stays"
else info "W1_960_faster" "C880/C960 = $r (+$(pct "${NORM[C880]}" "${NORM[C960]}")%) -> 960 stays"; fi
d="$(abspct "${NORM[C960r]}" "${NORM[C960]}")"; le "$d" 0.3 && pass "W2_C960_repeat_within_0.3pct (${d}%)" || fail "W2_C960_repeat_drift" "${d}% > 0.3% -- read clk_C960*.tsv"
info "N23_estimate" "full run at 960 ~ $(awk -v k="${VAL[C960]}" -v s="$SLICE_B" -v e="$EXPECTED23" 'BEGIN{printf "%.0f s (%.1f min)", k/1000*e/s, k/1000*e/s/60}') if the slice were representative (398-r2: it is not, +-30%)"

G23_DONE=0
if [[ "$RUN_FULL23" == "1" ]]; then
  banner "G23: ./$PY_BIN -g 23 23 (production path, full N=23, oracle $ORACLE23)"
  gpu_gate G23 || summary_exit
  dlog="$CRLOG/dispatch.log"; clog="$CRLOG/crunner_${CU_BIN}_N23.log"; before=0; [[ -f "$dlog" ]] && before="$(wc -l < "$dlog")"
  start="$(date -Is)"; clk_start G23
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX -u NQ_LAYOUT -u NQ_HELPER_CTX -u NQ_HELPER_MB -u NQ_MAX_BLOCKS -u NQ_BLOCK TIMEOUT_DURATION="${TIMEOUT_DURATION:-4h}" wrap "./$PY_BIN" -g 23 23 > "$LOGDIR/3_G23.stdout" 2>&1; rc=$?
  clk_stop G23; gpu_after G23
  [[ -f "$clog" ]] && cp "$clog" "$LOGDIR/3_G23.log"; [[ -f "$dlog" ]] && tail -n +"$((before+1))" "$dlog" > "$LOGDIR/3_G23_dispatch.log"
  [[ "$rc" == "0" ]] || fail "rc[G23]" "rc=$rc"
  parse_log G23 "$PY_BIN" 23 "$LOGDIR/3_G23.log" "$start" 423 32 960 "$EXPECTED23" && G23_DONE=1
  grep -q '\[gpu-run-correctness\] MATCH' "$LOGDIR/3_G23.log" 2>/dev/null && [[ "${TOTAL[G23]:-}" == "$ORACLE23" ]] && pass "G23_oracle_MATCH (${VAL[G23]} ms = $(awk -v k="${VAL[G23]}" 'BEGIN{printf "%d:%02d:%04.1f", int(k/3600000), int(k/60000)%60, (k/1000)%60}'))" || fail "G23_oracle" "total_sum=${TOTAL[G23]:-?} (expected $ORACLE23)"
  cp -f /tmp/crunner_N23_results.bin "$LOGDIR/bins/G23_results.bin" 2>/dev/null || true
fi

CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "W3_sm_clock_1710_all_cells" || fail "W3_sm_clock_1710_all_cells" "see clk_*.tsv"
if [[ "$FAIL" -eq 0 ]]; then info "ADOPTION" "413 is the production binary/.py (N=21/22 values stay 411's; N=23 production value = G23 when RUN_FULL23=1, else 413-r3)"; else info "ADOPTION" "NOT adopted -- see FAIL lines"; fi

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
nm={r['cell']:float(r['norm_ms']) for r in rows if r['norm_ms'] not in ('?','')}
print(f"{'cell':<6}{'binary':<24}{'N':>3}{'lay':>5}{'blk':>4}{'MB':>5}{'records':>10}{'free_mb':>8}{'kernel_ms':>12}{'norm_ms':>12}{'vs':>9}{'sm':>8}{'C':>4}{'h:mm:ss':>10}")
for r in rows:
    c=r['cell']; v=float(r['kernel_ms']); n=nm.get(c,v); s=v/1000
    base={'B1':'B0','Z1':'B1','C484':'R392','C880':'C960','C960r':'C960'}.get(c)
    d=f"{(n/nm[base]-1)*100:+.3f}%" if base and base in nm else ""
    print(f"{c:<6}{r['binary'][:23]:<24}{r['N']:>3}{r['layout']:>5}{r['block']:>4}{r['max_blocks']:>5}{r['records']:>10}{r['free_mb']:>8}{v:>12.1f}{n:>12.1f}{d:>9}{r['sm_mean']:>8}{r['temp_max']:>4}{int(s//3600):>4}:{int(s%3600//60):02d}:{s%60:04.1f}")
print("(vs: B1->B0, Z1->B1, C484->R392, C880/C960r->C960; normalised to 1710 MHz)")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd1[46]" --exclude="*_cpu" --exclude="*slice*.bin" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
