#!/usr/bin/env bash
# 415_r2b_validate.sh
#
# r2b (2026-10-13): r2 stopped at S0 -- the .py on disk had lost its revision-note block (code lines unchanged:
#   REV_TAG 415, diff 6/7). r2b (1) reports the note as INFO instead of FAIL, (2) with SKIP_BUILD=1 (default)
#   does NOT rm/rebuild 415_kernel_maxd16 or the Codon binary -- it reads the existing binary for S2-S5 -- so it
#   can run while 415_r3_full24.sh is using those binaries on the GPU; (3) everything else is r2 (valid Y2, raised
#   CPU iteration cap, GPU cells skipped by default). Y2 uses 8 CPU threads; the r3 host process is idle during
#   the kernel, so they do not compete. Run: bash 415_r2b_validate.sh 2>&1 | tee 415_r2b_validate_stdout.log
#
# r2 (2026-10-09, after the 16:34 run): the Y2c check PASSED VACUOUSLY. Both CPU harnesses (392 and 415) aborted on
#   real N=24 records with [ITER-CAP-HIT] (the CPU-only debug cap of 50,000,000 loop iterations per record is below
#   what real N=24 tasks need); both left an EMPTY output file, and `cmp -s` of two empty files is "identical".
#   Real-data equivalence of 415 vs 392 is nevertheless established by the GPU cell C484 == R392 (per-thread sums
#   identical on 154,880 real records, same stride) -- but Y2c as a gate was void. r2 fixes the harness only:
#   (1) every Y2 comparison requires the [kernel-cputest-done] line in BOTH logs and output size == records x 8;
#   (2) the CPU harnesses are built from /tmp copies whose DEBUG_ITER_CAP is raised to ITER_CAP (default 4e9);
#   (3) the GPU cells are skipped by default (they passed at 16:34; RUN_CELLS=1 repeats them).
#   No .py/.cu change. Expected: S0-S5 ~3 min, Y2 ~10 min (N=24 CPU records are ~6x N=23's).
#
# rev415 -- N=24: one binary (415_kernel_maxd16) for N <= 24. N <= 22 keeps 408's kernel, N = 23 keeps 413's
#   423 layout (<402>/<403>/<423> SASS identical, HARD); N = 24 runs the new 12-byte layout 424 with 15
#   ancestor frames (180 B/thread). No production value changes here; N=24's full run is 415-r3.
#
#   S0 static: .py = 414Py + 5 executable lines (VERSION_TAG, REV_TAG, CRunnerEntry(16, ./415_kernel_maxd16),
#      N>=24 filter env MAXDEPTH=16 + suffix d16only_415; diff 6/7); .cu fingerprint vs 414 (removed=17 added=74),
#      kernel sha 1f1200c4; PACK424_ANCESTOR 15; PACK423_ANCESTOR 14; notes; scripts present; gcc builds (415/414/392)
#   S1 ptxas: <402>,<403> 160 B ; <423> <= 176 B ; <424> <= 192 B ; spill 0 ; regs <= 44                 HARD
#   S2/S3 <402> 821302f2 / 446 / 109, <403> e940cb96 / 445 / 108  (== 408)                             HARD
#   S4 <423> sha c8316e18 (== 413)                                                                       HARD
#   S5 <424> SASS stats (sha / real / loop) recorded; stated P1: loop <= 110 (info)
#   N=24 input: 414 built constellations_N24_7.bin.soa_ref_361.bin.all_414.bin(.sched394f.bin) (66,737,631
#      records, all depth 16). This harness hard-links them to the 415 names (*.d16only_415.bin) -- no recompute.
#   H2 N=24 sched file has EXPECTED24 = 66,737,631 records                                               HARD
#   Y2 CPU equivalence (8 threads): N=21 415 == 414 (2,048 head) ; N=23 415 == 414 (2,048 equal-spaced) ;
#      N=24 415 (424) == 392_kernel_maxd16 CPU harness on 2,048 equal-spaced records                     HARD
#   Y3 rc=3: N=25 ; NQ_LAYOUT=423 at N=24 ; NQ_LAYOUT=403 at N=23                                       HARD
#   Cells (direct = BLOCK 32, helper 1 + 128 MB, NQ_EXTRA_CTX unset):
#     B0   414 N=21 auto(403) MB 960           results.bin == REF21 (411 B1 1bfab090)                   HARD
#     B1   415 N=21 auto(403) MB 960           results.bin == B0 (same SASS)                             HARD
#     Z1   415 N=21 NQ_LAYOUT=424 MB 960       results.bin == B0 ; time = cost of 424/180 B at N=21 (info) HARD
#     R392 392 N=24 slice A (154,880 = 10/thread @ 15,488), fixed 484x32, no helper
#     C484 415 N=24 slice A, NQ_MAX_BLOCKS=484  results.bin == R392 (same stride), total equal              HARD
#     D960 / D880 / D960r  415 N=24 slice B (614,400 = 20/thread @ 30,720)  total_sum all equal, D960r==D960 HARD
#   PRE-REGISTERED: no gain claimed. P1 <424> loop <= 110 (stated). P2 <424> frame <= 192 B, spill 0 (HARD).
#     No statement on Z1/B1 (176 B gave +2.1% at N=21; 180 B is read as +2..4%, informational).
#     W1 D880/D960: no direction stated (180 B x 11 blocks = 63.4 KB fits L1, x 12 = 69.1 KB does not; 413's
#     176 B x 12 still won). <= 0.998 -> N=24 runs at 880 (415-r2: N-dependent MB in the .py); otherwise 960.
#     W2 |D960r - D960| <= 0.3% (normalised). W3 all cells 1710 +- 2%.
#   ADOPT: all HARD -> 415 is the production binary/.py (N=21/22/23 values stay 413's; N=24's value = 415-r3).
#   RUN_FULL24=1: G24 = ./415Py -g 24 24, oracle 227,514,171,973,736 HARD, timeout 36h (est. 20-25 h). Prefer 415-r3.
#
# USAGE
#   bash 415_r2_validate.sh 2>&1 | tee 415_r2_validate_stdout.log   # S0-S5 + Y2/Y3 (GPU zero, ~15 min)
#   RUN_CELLS=1 bash 415_r2_validate.sh                              # + the GPU cells again (~45 min)

set -u

REV="415_r2b"
SKIP_BUILD="${SKIP_BUILD:-1}"
PY_SRC="${PY_SRC:-415Py_kernel_maxd16_final.py}"
PY_BIN="${PY_BIN:-${PY_SRC%.py}}"
PREV_PY="${PREV_PY:-414Py_kernel_maxd16_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-415_kernel_maxd16.cu}"
CU_BIN="${CU_BIN:-415_kernel_maxd16}"
PREV_CU="${PREV_CU:-414_kernel_maxd16.cu}"
PREV_BIN="${PREV_BIN:-414_kernel_maxd16}"
REF_CU="${REF_CU:-392_kernel_maxd16.cu}"
REF_BIN="${REF_BIN:-392_kernel_maxd16}"
FILTER="${FILTER:-413_filter_maxd16_only.py}"
SLICER="${SLICER:-413_slice_soa7.py}"
PERMUTE="${PERMUTE:-394f_permute_soa7.py}"
REF21="${REF21:-411_validate_20261006_150938/bins/B1_results.bin}"
REF21_SHA="${REF21_SHA:-1bfab090}"
REF23B="${REF23B:-$(ls -t 413_validate_*/bins/C960_results.bin 2>/dev/null | head -1)}"
SLB23="${SLB23:-413_n23_sliceB_3072000.bin}"
STREAM24="${STREAM24:-constellations_N24_7.bin}"
ALL24="${ALL24:-${STREAM24}.soa_ref_361.bin.all_414.bin}"
EXPECTED24="${EXPECTED24:-66737631}"
SASS402_SHA="${SASS402_SHA:-821302f227dcd160}"; SASS402_REAL="${SASS402_REAL:-446}"; SASS402_LOOP="${SASS402_LOOP:-109}"
SASS403_SHA="${SASS403_SHA:-e940cb96e5e65fc4}"; SASS403_REAL="${SASS403_REAL:-445}"; SASS403_LOOP="${SASS403_LOOP:-108}"
SASS423_SHA="${SASS423_SHA:-c8316e18}"; SASS423_REAL="${SASS423_REAL:-447}"; SASS423_LOOP="${SASS423_LOOP:-107}"
KERNEL_SHA_415="${KERNEL_SHA_415:-1f1200c4249266a3}"
KERNEL_SHA_414="${KERNEL_SHA_414:-101cbdbbcb6316d0}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
CODON="${CODON:-codon}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
HMB="${HMB:-128}"
NREC="${NREC:-2048}"
CPU24="${CPU24:-2048}"
SLICE_A="${SLICE_A:-154880}"      # 10 records per thread at stride 15,488 (392's fixed 484 x 32)
SLICE_B="${SLICE_B:-614400}"      # 20 records per thread at stride 30,720 (~1.1 ms/record expected -> ~11 min/cell)
COOLDOWN="${COOLDOWN:-15}"
CLK_INTERVAL="${CLK_INTERVAL:-5}"
STATIC_ONLY="${STATIC_ONLY:-0}"
RUN_FULL24="${RUN_FULL24:-0}"
RUN_K23="${RUN_K23:-0}"
RUN_CELLS="${RUN_CELLS:-0}"
ITER_CAP="${ITER_CAP:-4000000000ULL}"
OMP_THREADS="${OMP_THREADS:-8}"
ORACLE21=314666222712
ORACLE23=24233937684440
ORACLE24=227514171973736

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
{ grep -q "^# 415 " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || info "py_revision_notes_MISSING" "no '# 415 ...' note block in $PY_SRC (the 16:34 copy had it; code lines are unchanged -- r2b reports, does not gate)"
grep -qE '^REV_TAG:str="415"' "$CODE" && pass "source_rev_tag_is_415" || fail "source_rev_tag_is_415" "wrong REV_TAG"
grep -qF 'CRunnerEntry(16,"./415_kernel_maxd16","NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_entry_maxd16_415_binary_helper_prefix" || fail "source_table_entry" "table entry wrong"
{ ! grep -E 'CRunnerEntry\(' "$CODE" | grep -q 'NQ_EXTRA_CTX'; } && pass "source_no_table_entry_carries_NQ_EXTRA_CTX" || fail "source_no_table_entry_carries_NQ_EXTRA_CTX" "found"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=960\b" "$CODE" && pass "source_default_max_blocks_is_960" || fail "source_default_max_blocks_is_960" "default not 960"
grep -qF 'd16only_415.bin' "$CODE" && grep -qF 'filter_env:str="" if N<=23 else "MAXDEPTH=16 "' "$CODE" && grep -qF 'maxd16only_413.bin" if N==23' "$CODE" && pass "source_n24_filter_env_and_suffix_wired (N=23 keeps maxd16only_413)" || fail "source_n24_filter_env_and_suffix_wired" "missing"
grep -qE '^STREAM_SCAN_CHUNK:int=1<<21$' "$CODE" && pass "source_414_chunked_scans_kept" || fail "source_414_chunked_scans_kept" "STREAM_SCAN_CHUNK missing"
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "6" && "$PA_" == "7" ]] && pass "py_diff_fingerprint_vs_414Py (removed=6 added=7 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_414Py" "removed=$PR_ added=$PA_, expected 6/7"
else info "py_diff_fingerprint_vs_414Py" "skipped ($PREV_PY absent)"; fi
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
[[ "$KB" == "$KERNEL_SHA_415"* ]] && pass "cu_kernel_region_sha (${KB:0:16})" || fail "cu_kernel_region_sha" "sha ${KB:0:16} != $KERNEL_SHA_415"
KP="$(extract_kernel "/tmp/${REV}_prev_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KP" == "$KERNEL_SHA_414"* ]] && pass "prev_cu_is_414_(=413) (${KP:0:16})" || fail "prev_cu_is_414" "sha ${KP:0:16} != $KERNEL_SHA_414"
CR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true); CA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
[[ "$CR_" == "17" && "$CA_" == "74" ]] && pass "cu_code_region_fingerprint_vs_414 (removed=17 added=74)" || fail "cu_code_region_fingerprint" "removed=$CR_ added=$CA_, expected 17/74"
grep -q '^#define PACK424_ANCESTOR 15' "$CU_SRC" && pass "cu_pack424_ancestor_is_15" || fail "cu_pack424_ancestor_is_15" "missing"
grep -q '^#define PACK424_WIDTH   22' "$CU_SRC" && pass "cu_pack424_width_is_22" || fail "cu_pack424_width_is_22" "missing"
grep -q '^#define PACK423_ANCESTOR 14' "$CU_SRC" && pass "cu_pack423_ancestor_still_14" || fail "cu_pack423_ancestor_still_14" "changed"
grep -q '^#define MAXD14_ANCESTOR 13' "$CU_SRC" && pass "cu_maxd14_ancestor_still_13" || fail "cu_maxd14_ancestor_still_13" "changed"
grep -q 'MAXDEPTH = int(os.environ.get("MAXDEPTH", "15"))' "$FILTER" && pass "filter_maxdepth_default_15 (415 passes MAXDEPTH=16 in the env)" || fail "filter_maxdepth_default_15" "wrong default"
grep -q "rev415 " "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev415 note in header"
# r2: the CPU-only debug cap (50,000,000 iterations per record) is below real N=24 tasks. The harnesses are built from
# /tmp copies with the cap raised to $ITER_CAP; the sources on disk are untouched (the cap lives in #ifndef __CUDACC__ code).
patched() { sed "s/const uint64_t DEBUG_ITER_CAP = 50000000ULL;/const uint64_t DEBUG_ITER_CAP = ${ITER_CAP};/" "$1" > "/tmp/${REV}_$(basename "$1")"; grep -c "DEBUG_ITER_CAP = ${ITER_CAP}" "/tmp/${REV}_$(basename "$1")"; }
[[ "$(patched "$CU_SRC")" == "1" && "$(patched "$PREV_CU")" == "1" && "$(patched "$REF_CU")" == "1" ]] && pass "cpu_iter_cap_patched_in_tmp_copies ($ITER_CAP)" || fail "cpu_iter_cap_patch" "DEBUG_ITER_CAP line not found in one of the three .cu"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "/tmp/${REV}_$(basename "$CU_SRC")" -lm 2>"/tmp/${REV}_gcc.log" && pass "cu_cpu_harness_builds[415]" || fail "cu_cpu_harness_builds[415]" "$(head -3 /tmp/${REV}_gcc.log)"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_prev" "/tmp/${REV}_$(basename "$PREV_CU")" -lm 2>"/tmp/${REV}_gcc_prev.log" && pass "cu_cpu_harness_builds[414]" || fail "cu_cpu_harness_builds[414]" "$(head -3 /tmp/${REV}_gcc_prev.log)"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_ref" "/tmp/${REV}_$(basename "$REF_CU")" -lm 2>"/tmp/${REV}_gcc_ref.log" && pass "cu_cpu_harness_builds[392]" || fail "cu_cpu_harness_builds[392]" "$(head -3 /tmp/${REV}_gcc_ref.log)"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Build, ptxas (S1), SASS (S2-S5), codon, reference binaries
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR/sass" "$LOGDIR/bins" "$CRLOG"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; df -h . 2>&1; free -g 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
[[ ! -x "$NVCC" ]] && NVCC="nvcc"; [[ ! -x "$CUOBJDUMP" ]] && CUOBJDUMP="cuobjdump"
banner "Building"
if [[ "$SKIP_BUILD" == "1" ]]; then
  [[ -x "$CU_BIN" ]] && pass "existing_binary_used[$CU_BIN] ($(stat -c %y "$CU_BIN" | cut -c1-19); SKIP_BUILD=1, r3 may be running on it)" || { fail "cu_binary_missing" "$CU_BIN absent and SKIP_BUILD=1"; summary_exit; }
  : > "$LOGDIR/05_nvcc_${REV}.log"
else
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc_${REV}.log" 2>&1
[[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "$(grep -i error "$LOGDIR/05_nvcc_${REV}.log" | head -3)"; summary_exit; }
fi
ptx_stats() {  # mangled-fragment -> FR SP RG
  FR="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"
  SP="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes spill stores, [0-9]* bytes spill loads' | head -1)"
  RG="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"
}
if [[ "$SKIP_BUILD" == "1" ]]; then info "S1_ptxas" "skipped (no fresh nvcc log; the 16:34 run recorded 160/160/176/184 B, spill 0, regs 37/35/36/36)"; FRAME424="184(16:34)"; else
for L in 402 403; do
  ptx_stats "kernel_dfs_iter_gpu_maxd16ILi${L}E"
  [[ "$FR" == "160" && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "S1_ptxas_L${L}_frame160_spill0_regs${RG}" || { fail "S1_ptxas_L${L}" "frame=${FR}B regs=${RG} ${SP}"; summary_exit; }
done
ptx_stats "kernel_dfs_iter_gpu_maxd16ILi423E"
[[ -n "$FR" && "$FR" -le 176 && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "S1_ptxas_L423_frame${FR}_spill0_regs${RG}" || { fail "S1_ptxas_L423" "frame=${FR}B regs=${RG} ${SP}"; summary_exit; }
ptx_stats "kernel_dfs_iter_gpu_maxd16ILi424E"
[[ -n "$FR" && "$FR" -le 192 && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "S1_P2_ptxas_L424_frame${FR}_spill0_regs${RG} (15 x 12 = 180 B; 12 blocks/SM = $((FR*384)) B of local)" || { fail "S1_P2_ptxas_L424" "frame=${FR}B regs=${RG} ${SP} (expected <= 192 / 0 / <= 44)"; summary_exit; }
FRAME424="$FR"
fi
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
[[ "$S_SHA" == "$SASS402_SHA" && "$S_REAL" == "$SASS402_REAL" && "$S_LOOP" == "$SASS402_LOOP" ]] && pass "S2_sass_402_is_408s (sha $S_SHA, real $S_REAL, loop $S_LOOP)" || { fail "S2_sass_402_differs_from_408" "sha $S_SHA real $S_REAL loop $S_LOOP != $SASS402_SHA / $SASS402_REAL / $SASS402_LOOP -- stop and read the SASS"; summary_exit; }
read -r S_SHA S_REAL S_LOOP S_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_dfs_iter_gpu_maxd16ILi403E)"
[[ "$S_SHA" == "$SASS403_SHA" && "$S_REAL" == "$SASS403_REAL" && "$S_LOOP" == "$SASS403_LOOP" ]] && pass "S3_sass_403_is_408s (sha $S_SHA, real $S_REAL, loop $S_LOOP)" || { fail "S3_sass_403_differs_from_408" "sha $S_SHA real $S_REAL loop $S_LOOP != $SASS403_SHA / $SASS403_REAL / $SASS403_LOOP -- stop and read the SASS"; summary_exit; }
read -r S_SHA S_REAL S_LOOP S_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_dfs_iter_gpu_maxd16ILi423E)"
[[ "$S_SHA" == "$SASS423_SHA"* && "$S_REAL" == "$SASS423_REAL" && "$S_LOOP" == "$SASS423_LOOP" ]] && pass "S4_sass_423_is_413s (sha $S_SHA, real $S_REAL, loop $S_LOOP)" || { fail "S4_sass_423_differs_from_413" "sha $S_SHA real $S_REAL loop $S_LOOP != $SASS423_SHA* / $SASS423_REAL / $SASS423_LOOP -- the 424 branch perturbed <423>: stop and read the SASS"; summary_exit; }
read -r S_SHA S_REAL S_LOOP S_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_dfs_iter_gpu_maxd16ILi424E)"
[[ "$S_SHA" != "CANNOT_LOCATE" ]] && pass "S5_sass_424_recorded (sha $S_SHA, real $S_REAL, loop $S_LOOP, frame ${FRAME424} B)" || { fail "S5_sass_424_missing" "no <424> instantiation in the SASS"; summary_exit; }
if [[ "$S_LOOP" -le 110 ]]; then pass "P1_loop_424_le_110 (loop $S_LOOP; stated: <423> 107 + 0)"; else info "P1_loop_424_above_statement" "loop $S_LOOP > 110 -- read the pop/push of <424> before 415-r2 (informational)"; fi

if [[ "$SKIP_BUILD" == "1" ]]; then [[ -x "$PY_BIN" ]] && pass "existing_py_binary_used[$PY_BIN]" || { fail "py_binary_missing" "$PY_BIN absent and SKIP_BUILD=1"; summary_exit; }
else
rm -f "$PY_BIN"; "$CODON" build -release "$PY_SRC" > "$LOGDIR/06_codon_${REV}.log" 2>&1
[[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN]" || { fail "codon_build" "$(tail -3 "$LOGDIR/06_codon_${REV}.log")"; summary_exit; }
fi

find_input() { local n="$1" c; for lg in $(ls -t 4*_crunner_logs/crunner_*_N${n}.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { echo "$c"; return; }; done; ls -t constellations_N${n}_*.sched394f.bin 2>/dev/null | head -1; }
IN21="$(find_input 21)"; [[ -f "${IN21:-/nonexistent}" ]] && pass "input21_located ($IN21)" || { fail "input21_located" "no N=21 sched input"; summary_exit; }
IN23="$(find_input 23)"; [[ -f "${IN23:-/nonexistent}" ]] && pass "input23_located ($IN23)" || { fail "input23_located" "no N=23 sched input (413 built constellations_N23_7.bin...maxd16only_413.bin.sched394f.bin)"; summary_exit; }
if [[ -f "$REF21" ]]; then
  RS="$(sha256sum "$REF21" | cut -c1-8)"
  [[ "$RS" == "$REF21_SHA" ]] && pass "ref21_is_411_B1 ($REF21 sha $RS)" || fail "ref21_is_411_B1" "sha $RS != $REF21_SHA"
else fail "ref21_present" "$REF21 missing -- pass REF21=<411 B1 results.bin>"; fi

# ---------------------------------------------------------------------
# 3. N=24 input: 414's all_414 files, hard-linked to the 415 names (no recompute)
# ---------------------------------------------------------------------
banner "N=24 input"
SOA24="${STREAM24}.soa_ref_361.bin"; FILT24="${SOA24}.d16only_415.bin"; SCHED24="${FILT24}.sched394f.bin"
[[ -s "$ALL24" && -s "$ALL24.sched394f.bin" ]] && pass "n24_414_files_present ($ALL24 + .sched394f.bin)" || { fail "n24_414_files" "$ALL24(.sched394f.bin) missing: run 414_validate.sh (N24 stages) first"; summary_exit; }
[[ -e "$FILT24" ]] || ln "$ALL24" "$FILT24" 2>/dev/null || cp "$ALL24" "$FILT24"
[[ -e "$SCHED24" ]] || ln "$ALL24.sched394f.bin" "$SCHED24" 2>/dev/null || cp "$ALL24.sched394f.bin" "$SCHED24"
cmp -s "$ALL24.sched394f.bin" "$SCHED24" && pass "n24_sched_linked ($SCHED24)" || { fail "n24_sched_linked" "$SCHED24 differs from $ALL24.sched394f.bin"; summary_exit; }
[[ $(( $(stat -c %s "$SCHED24") / 28 )) -eq "$EXPECTED24" && $(( $(stat -c %s "$SCHED24") % 28 )) -eq 0 ]] && pass "H2_sched24_has_${EXPECTED24}_records" || { fail "H2_sched24" "$SCHED24 wrong size"; summary_exit; }
IN24="$SCHED24"
SLA="${REV}_n24_sliceA_${SLICE_A}.bin"; SLB="${REV}_n24_sliceB_${SLICE_B}.bin"; SLC="${REV}_n24_sliceC_${CPU24}.bin"; SLC23="${REV}_n23_sliceC_${NREC}.bin"
for pair in "$SLA:$SLICE_A:$IN24" "$SLB:$SLICE_B:$IN24" "$SLC:$CPU24:$IN24" "$SLC23:$NREC:$IN23"; do IFS=: read -r f c src <<< "$pair"; if [[ ! -s "$f" || $(( $(stat -c %s "$f") / 28 )) -ne "$c" ]]; then python3 "$SLICER" "$src" "$f" "$c" > "$LOGDIR/13_slice_${f}.log" 2>&1 || true; fi; [[ -s "$f" && $(( $(stat -c %s "$f") / 28 )) -eq "$c" ]] && pass "slice_ready[$f]" || { fail "slice[$f]" "$(tail -1 "$LOGDIR/13_slice_${f}.log" 2>/dev/null)"; summary_exit; }; done

# ---------------------------------------------------------------------
# 4. CPU equivalence (Y2) and refusals (Y3)
# ---------------------------------------------------------------------
banner "Y2: CPU equivalence ($OMP_THREADS threads)"
export OMP_NUM_THREADS="$OMP_THREADS"
# r2: a comparison is valid only when BOTH runs finished ([kernel-cputest-done] with the expected record count) and
# BOTH outputs have records x 8 bytes. An abort ([ITER-CAP-HIT], [STACK-OVERFLOW]) leaves an empty file: never compare those.
y2_valid() {  # log out records label
  local lg="$1" out="$2" n="$3" lab="$4"
  grep -q "\[kernel-cputest-done\] N=[0-9]* records=${n} " "$lg" || { fail "Y2_invalid[$lab]" "no done line (abort: $(grep -o '\[[A-Z-]*\]' "$lg" | grep -v progress | tail -1))"; return 1; }
  [[ -f "$out" && $(stat -c %s "$out") -eq $((n*8)) ]] || { fail "Y2_invalid[$lab]" "output size $(stat -c %s "$out" 2>/dev/null) != $((n*8))"; return 1; }
  return 0; }
head -c $((NREC*28)) "$IN21" > "/tmp/${REV}_in21_head.bin"
"/tmp/${REV}_cpu_prev" 21 "/tmp/${REV}_in21_head.bin" "/tmp/${REV}_y2_414_21.bin" "$NREC" > "$LOGDIR/2_cpu_N21_414.log" 2>&1
"/tmp/${REV}_cpu_cur"  21 "/tmp/${REV}_in21_head.bin" "/tmp/${REV}_y2_415_21.bin" "$NREC" > "$LOGDIR/2_cpu_N21_415auto.log" 2>&1
y2_valid "$LOGDIR/2_cpu_N21_414.log" "/tmp/${REV}_y2_414_21.bin" "$NREC" 414_21 && y2_valid "$LOGDIR/2_cpu_N21_415auto.log" "/tmp/${REV}_y2_415_21.bin" "$NREC" 415_21 || summary_exit
cmp -s "/tmp/${REV}_y2_414_21.bin" "/tmp/${REV}_y2_415_21.bin" && pass "Y2a_cpu_N21_auto_415_equals_414 ($NREC records)" || { fail "Y2a_cpu_N21_auto" "per-record outputs differ"; summary_exit; }
T0=$(date +%s)
"/tmp/${REV}_cpu_prev" 23 "$SLC23" "/tmp/${REV}_y2_414_23.bin" > "$LOGDIR/2_cpu_N23_414.log" 2>&1
"/tmp/${REV}_cpu_cur"  23 "$SLC23" "/tmp/${REV}_y2_415_23.bin" > "$LOGDIR/2_cpu_N23_415.log" 2>&1; T1=$(date +%s)
y2_valid "$LOGDIR/2_cpu_N23_414.log" "/tmp/${REV}_y2_414_23.bin" "$NREC" 414_23 && y2_valid "$LOGDIR/2_cpu_N23_415.log" "/tmp/${REV}_y2_415_23.bin" "$NREC" 415_23 || summary_exit
grep -q 'layout=423' "$LOGDIR/2_cpu_N23_415.log" && cmp -s "/tmp/${REV}_y2_414_23.bin" "/tmp/${REV}_y2_415_23.bin" && pass "Y2b_cpu_N23_415_equals_414 ($NREC equal-spaced records, $((T1-T0)) s)" || { fail "Y2b_cpu_N23_415_vs_414" "differ or layout not 423"; summary_exit; }
T0=$(date +%s)
"/tmp/${REV}_cpu_ref" 24 "$SLC" "/tmp/${REV}_y2_392_24.bin" > "$LOGDIR/2_cpu_N24_392.log" 2>&1; T1=$(date +%s)
"/tmp/${REV}_cpu_cur" 24 "$SLC" "/tmp/${REV}_y2_415_24.bin" > "$LOGDIR/2_cpu_N24_415.log" 2>&1; T2=$(date +%s)
y2_valid "$LOGDIR/2_cpu_N24_392.log" "/tmp/${REV}_y2_392_24.bin" "$CPU24" 392_24 && y2_valid "$LOGDIR/2_cpu_N24_415.log" "/tmp/${REV}_y2_415_24.bin" "$CPU24" 415_24 || summary_exit
grep -q 'layout=424' "$LOGDIR/2_cpu_N24_415.log" && cmp -s "/tmp/${REV}_y2_392_24.bin" "/tmp/${REV}_y2_415_24.bin" && pass "Y2c_cpu_N24_415_equals_392 ($CPU24 equal-spaced records; 392 $((T1-T0)) s, 415 $((T2-T1)) s; total_sum $(grep -o 'total_sum=[0-9]*' "$LOGDIR/2_cpu_N24_415.log" | head -1 | cut -d= -f2))" || { fail "Y2c_cpu_N24_415_vs_392" "per-record outputs differ (or abort: $(grep -o '\[[A-Z-]*\]' "$LOGDIR/2_cpu_N24_415.log" | head -1)) -- the 424 layout or the 15-frame bound is wrong; do not run the GPU"; summary_exit; }
banner "Y3: refusals"
"/tmp/${REV}_cpu_cur" 25 "$SLC" /tmp/${REV}_y3a.bin 8 > "$LOGDIR/2_y3_n25.log" 2>&1; rc=$?
[[ "$rc" == "3" ]] && pass "Y3_N25_refused_rc3" || fail "Y3_N25_refused_rc3" "rc=$rc"
NQ_LAYOUT=423 "/tmp/${REV}_cpu_cur" 24 "$SLC" /tmp/${REV}_y3b.bin 8 > "$LOGDIR/2_y3_l423_n24.log" 2>&1; rc=$?
[[ "$rc" == "3" ]] && pass "Y3_layout423_at_N24_refused_rc3" || fail "Y3_layout423_at_N24_refused_rc3" "rc=$rc"
NQ_LAYOUT=403 "/tmp/${REV}_cpu_cur" 23 "$SLC23" /tmp/${REV}_y3c.bin 8 > "$LOGDIR/2_y3_l403_n23.log" 2>&1; rc=$?
[[ "$rc" == "3" ]] && pass "Y3_layout403_at_N23_refused_rc3" || fail "Y3_layout403_at_N23_refused_rc3" "rc=$rc"

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
  local KMS TOT FREE HL LAYV XC CFG NORMV
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
run_cell() {  # cell binary N input layout_env block mb want_layout [oracle]
  local cell="$1" bin="$2" n="$3" in="$4" lenv="$5" blk="$6" mb="$7" want="$8" oracle="${9:-}"; gpu_gate "$cell" || return 1
  local lg="$LOGDIR/3_${cell}.log" start recs; start="$(date -Is)"; recs=$(( $(stat -c %s "$in") / 28 )); clk_start "$cell"
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX NQ_LAYOUT="$lenv" NQ_BLOCK="$blk" NQ_MAX_BLOCKS="$mb" NQ_HELPER_CTX=1 NQ_HELPER_MB="$HMB" "./$bin" "$n" "$in" "$LOGDIR/bins/${cell}_results.bin" $oracle > "$lg" 2>&1 || true
  clk_stop "$cell"; gpu_after "$cell"
  parse_log "$cell" "$bin" "$n" "$lg" "$start" "$want" "$blk" "$mb" "$recs"
}
run_392() {  # cell N input   (392: fixed 484 x 32, no env knobs, no helper)
  local cell="$1" n="$2" in="$3"; gpu_gate "$cell" || return 1
  local lg="$LOGDIR/3_${cell}.log" start recs; start="$(date -Is)"; recs=$(( $(stat -c %s "$in") / 28 )); clk_start "$cell"
  "./$REF_BIN" "$n" "$in" "$LOGDIR/bins/${cell}_results.bin" > "$lg" 2>&1 || true
  clk_stop "$cell"; gpu_after "$cell"
  parse_log "$cell" "$REF_BIN" "$n" "$lg" "$start" "" 32 484 "$recs"
}
same_bin() { local a="$LOGDIR/bins/${1}_results.bin" b="$2"; [[ -f "$b" ]] || b="$LOGDIR/bins/${2}_results.bin"; cmp -s "$a" "$b" && pass "${1}_per_thread_results_identical_to_${2} ($(stat -c %s "$a") bytes)" || fail "${1}_per_thread_results_DIFFER_from_${2}" "not adoptable"; }
oracle_ok() { [[ "${TOTAL[$1]:-}" == "$2" ]] && pass "oracle[$1]" || fail "oracle[$1]" "total_sum=${TOTAL[$1]:-?} != $2"; }

if [[ "$RUN_CELLS" != "1" ]]; then
  info "cells" "skipped (RUN_CELLS=1 repeats B0/B1/Z1/R392/C484/D960/D880/D960r; they passed in 415_validate_20261009_163442)"
  if [[ "$FAIL" -eq 0 ]]; then info "ADOPTION" "Y2c now holds on real data: with the 16:34 GPU cells, 415 is the production binary/.py (N=24 production value = 415-r3)"; else info "ADOPTION" "NOT adopted -- see FAIL lines"; fi
  { echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
  TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
  summary_exit
fi
banner "N=21: B0 $PREV_BIN / B1 $CU_BIN / Z1 $CU_BIN NQ_LAYOUT=424"
run_cell B0 "$PREV_BIN" 21 "$IN21" auto 32 960 403 "$ORACLE21" || summary_exit; oracle_ok B0 "$ORACLE21"; same_bin B0 "$REF21"
sleep "$COOLDOWN"; run_cell B1 "$CU_BIN" 21 "$IN21" auto 32 960 403 "$ORACLE21" || summary_exit; oracle_ok B1 "$ORACLE21"; same_bin B1 B0
info "B1_vs_B0" "raw $(pct "${VAL[B1]}" "${VAL[B0]}")% / normalised $(pct "${NORM[B1]}" "${NORM[B0]}")% (identical SASS: session scatter)"
sleep "$COOLDOWN"; run_cell Z1 "$CU_BIN" 21 "$IN21" 424 32 960 424 "$ORACLE21" || summary_exit; oracle_ok Z1 "$ORACLE21"; same_bin Z1 B0
info "Z1_424_at_N21" "Z1/B1 = $(ratio "${NORM[Z1]}" "${NORM[B1]}") (normalised; cost of the 424 pop/push and the ${FRAME424} B frame at N=21; 413's 423/176 B gave 1.0211; informational)"

if [[ "$RUN_K23" == "1" ]]; then
  banner "K23: $CU_BIN on 413's N=23 slice B == 413 C960 (same SASS, same stride)"
  if [[ -s "$SLB23" && -f "${REF23B:-/nonexistent}" ]]; then
    sleep "$COOLDOWN"; run_cell K23 "$CU_BIN" 23 "$SLB23" auto 32 960 423 || summary_exit; same_bin K23 "$REF23B"
  else info "K23" "skipped ($SLB23 or REF23B missing)"; fi
fi

banner "N=24 slice A ($SLICE_A records): R392 $REF_BIN (484x32) / C484 $CU_BIN @484"
sleep "$COOLDOWN"; run_392 R392 24 "$SLA" || summary_exit
sleep "$COOLDOWN"; run_cell C484 "$CU_BIN" 24 "$SLA" auto 32 484 424 || summary_exit
[[ "${TOTAL[C484]}" == "${TOTAL[R392]}" ]] && pass "C484_total_equals_R392 (${TOTAL[C484]})" || fail "C484_total_differs_from_R392" "${TOTAL[C484]} != ${TOTAL[R392]} -- not adoptable"
same_bin C484 R392
info "C484_vs_R392_time" "415@484 $(pct "${NORM[C484]}" "${NORM[R392]}")% vs 392 (old 240 B frame, 1 context; informational only)"

banner "N=24 slice B ($SLICE_B records): D960 / D880 / D960r"
sleep "$COOLDOWN"; run_cell D960 "$CU_BIN" 24 "$SLB" auto 32 960 424 || summary_exit
sleep "$COOLDOWN"; run_cell D880 "$CU_BIN" 24 "$SLB" auto 32 880 424 || summary_exit
sleep "$COOLDOWN"; run_cell D960r "$CU_BIN" 24 "$SLB" auto 32 960 424 || summary_exit
[[ "${TOTAL[D880]}" == "${TOTAL[D960]}" && "${TOTAL[D960r]}" == "${TOTAL[D960]}" ]] && pass "sliceB_totals_equal_across_MB (${TOTAL[D960]})" || fail "sliceB_totals_differ" "D960 ${TOTAL[D960]} D880 ${TOTAL[D880]} D960r ${TOTAL[D960r]} -- not adoptable"
same_bin D960r D960
r="$(ratio "${NORM[D880]}" "${NORM[D960]}")"
if le "$r" 0.998; then info "W1_880_faster" "D880/D960 = $r -> N=24 runs at MB 880 (415-r2: N-dependent MB in the .py)"
elif le "$r" 1.002; then info "W1_no_difference" "D880/D960 = $r -> 960 stays"
else info "W1_960_faster" "D880/D960 = $r (+$(pct "${NORM[D880]}" "${NORM[D960]}")%) -> 960 stays"; fi
d="$(abspct "${NORM[D960r]}" "${NORM[D960]}")"; le "$d" 0.3 && pass "W2_D960_repeat_within_0.3pct (${d}%)" || fail "W2_D960_repeat_drift" "${d}% > 0.3% -- read clk_D960*.tsv"
info "N24_estimate" "full run at 960 ~ $(awk -v k="${VAL[D960]}" -v s="$SLICE_B" -v e="$EXPECTED24" 'BEGIN{printf "%.0f s (%.1f h)", k/1000*e/s, k/1000*e/s/3600}') if the slice were representative (398-r2 / 413-r3: it is not, 6-37%)"

if [[ "$RUN_FULL24" == "1" ]]; then
  banner "G24: ./$PY_BIN -g 24 24 (production path, full N=24, oracle $ORACLE24, timeout ${TIMEOUT_DURATION:-36h})"
  gpu_gate G24 || summary_exit
  dlog="$CRLOG/dispatch.log"; clog="$CRLOG/crunner_${CU_BIN}_N24.log"; before=0; [[ -f "$dlog" ]] && before="$(wc -l < "$dlog")"
  start="$(date -Is)"; clk_start G24
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX -u NQ_LAYOUT -u NQ_HELPER_CTX -u NQ_HELPER_MB -u NQ_MAX_BLOCKS -u NQ_BLOCK timeout --signal=TERM --kill-after=60s "${TIMEOUT_DURATION:-36h}" "./$PY_BIN" -g 24 24 > "$LOGDIR/3_G24.stdout" 2>&1; rc=$?
  clk_stop G24; gpu_after G24
  [[ -f "$clog" ]] && cp "$clog" "$LOGDIR/3_G24.log"; [[ -f "$dlog" ]] && tail -n +"$((before+1))" "$dlog" > "$LOGDIR/3_G24_dispatch.log"
  [[ "$rc" == "0" ]] || fail "rc[G24]" "rc=$rc"
  parse_log G24 "$PY_BIN" 24 "$LOGDIR/3_G24.log" "$start" 424 32 960 "$EXPECTED24" || true
  grep -q '\[gpu-run-correctness\] MATCH' "$LOGDIR/3_G24.log" 2>/dev/null && [[ "${TOTAL[G24]:-}" == "$ORACLE24" ]] && pass "G24_oracle_MATCH (${VAL[G24]} ms = $(awk -v k="${VAL[G24]}" 'BEGIN{printf "%d:%02d:%04.1f", int(k/3600000), int(k/60000)%60, (k/1000)%60}'))" || fail "G24_oracle" "total_sum=${TOTAL[G24]:-?} (expected $ORACLE24)"
  cp -f /tmp/crunner_N24_results.bin "$LOGDIR/bins/G24_results.bin" 2>/dev/null || true
fi

CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "W3_sm_clock_1710_all_cells" || fail "W3_sm_clock_1710_all_cells" "see clk_*.tsv (power-capped cells read 1680-1695; compare norm_ms)"
if [[ "$FAIL" -eq 0 ]]; then info "ADOPTION" "415 is the production binary/.py (N=21/22/23 values stay 413's; N=24 production value = G24 when RUN_FULL24=1, else 415-r3)"; else info "ADOPTION" "NOT adopted -- see FAIL lines"; fi

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
nm={r['cell']:float(r['norm_ms']) for r in rows if r['norm_ms'] not in ('?','')}
print(f"{'cell':<6}{'binary':<24}{'N':>3}{'lay':>5}{'blk':>4}{'MB':>5}{'records':>10}{'free_mb':>8}{'kernel_ms':>12}{'norm_ms':>12}{'vs':>9}{'sm':>8}{'C':>4}{'h:mm:ss':>10}")
for r in rows:
    c=r['cell']; v=float(r['kernel_ms']); n=nm.get(c,v); s=v/1000
    base={'B1':'B0','Z1':'B1','C484':'R392','D880':'D960','D960r':'D960'}.get(c)
    d=f"{(n/nm[base]-1)*100:+.3f}%" if base and base in nm else ""
    print(f"{c:<6}{r['binary'][:23]:<24}{r['N']:>3}{r['layout']:>5}{r['block']:>4}{r['max_blocks']:>5}{r['records']:>10}{r['free_mb']:>8}{v:>12.1f}{n:>12.1f}{d:>9}{r['sm_mean']:>8}{r['temp_max']:>4}{int(s//3600):>4}:{int(s%3600//60):02d}:{s%60:04.1f}")
print("(vs: B1->B0, Z1->B1, C484->R392, D880/D960r->D960; normalised to 1710 MHz)")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd1[46]" --exclude="*_cpu" --exclude="*slice*.bin" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
