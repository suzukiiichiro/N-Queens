#!/usr/bin/env bash
# 417_validate.sh -- N=25 on layout 425 with 16 ancestor frames (192 B): correctness against an independent
# recursive reference, then the MB sweep on an N=25 slice. Production values for N <= 24 do not change.
#
# WHERE 416 STANDS (2026-10-14 11:40): adopted. <402>/<403>/<423>/<424> SASS = 415; <425> bcbd388a / 452 / 111, 184 B;
#   E425/E424 = 1.0292 on the N=24 slice B (results.bin byte-identical), Z2 +2.61% at N=21. 425 costs +2.9% at N=24.
#   Production: N=21 97,946 / N=22 849,096 / N=23 7,717,284 / N=24 73,112,528 ms on 416_kernel_maxd16 / 416Py.
#
# rev417 -- three constants in the .cu (PACK425_ANCESTOR 15 -> 16, PACK425_MAXN 25, auto N=25 -> 425) and 7 executable
#   lines in the .py (tags, CRunnerEntry(17, ./417_kernel_maxd16), preset_queens 7 at N=25, MAXDEPTH=17 filter into
#   *.d17only_417.bin). The N=25 input is built by 417_build_input25.sh (GPU zero, cgroup) BEFORE this harness.
#   392 (16 B frames, 15 of them) cannot run N=25, so the CPU reference is 417_ref_dfs.c: a plain recursive DFS that
#   shares the schedule precompute and the one-step arithmetic but none of the stack mechanics. Its own validity is
#   gated here on real N=24 data against 392 (Y2r) before it is used at N=25 (Y2e).
#
#   S0 static: .py diff 7/7 vs 416Py; .cu code-region diff removed=8 added=14 vs 416; kernel sha b8f82d27 (prev 04f3b881);
#      PACK425_ANCESTOR 16 / MAXN 25 / WIDTH 27; PACK424/423/MAXD14 constants unchanged; auto line; notes; gcc builds (417/416/392/ref)
#   S1 ptxas: <402>,<403> 160 B ; <423> 176 B ; <424> 184 B ; <425> <= 200 B (192 expected) ; spill 0 ; regs <= 44   HARD
#   S2/S3 <402> 821302f2 / 446 / 109, <403> e940cb96 / 445 / 108  (== 408)                                      HARD
#   S4 <423> sha c8316e18 / 447 / 107 (== 413)   S5 <424> sha f60ecee7 / 447 / 107 (== 415)                     HARD
#   S6 <425> SASS stats recorded (the frame size moves the sha); stated P1: loop 111 +- 1 (info)
#   H2 N=25 sched file (*.d17only_417.bin.sched394f.bin) record count == the 361 dump's                           HARD
#   Y2 CPU equivalence (8 threads, DEBUG_ITER_CAP raised to ITER_CAP in /tmp copies, y2_valid):
#      Y2a N=21 417 == 416 (2,048 head) ; Y2c N=24 417 auto(424) == 416 (2,048 equal-spaced) ;
#      Y2r N=24 ref == 392 (same 2,048: the reference validated on real data)                                   HARD
#      Y2e N=25 417 auto(425, 16 frames) == ref (CPU25 = 1,024 equal-spaced records)                            HARD
#   Y3 rc=3: N=26 ; NQ_LAYOUT=424 at N=25 ; NQ_LAYOUT=423 at N=24                                                HARD
#   Cells (direct = BLOCK 32, helper 1 + 128 MB, NQ_EXTRA_CTX unset):
#     B0    416 N=21 auto(403) MB 960           results.bin == REF21 (411 B1 1bfab090)                          HARD
#     B1    417 N=21 auto(403) MB 960           results.bin == B0 (same SASS)                                    HARD
#     D960 / D880 / D960r  417 N=25 slice B (SLICE_B = 307,200 = 10/thread @ 30,720) auto(425) MB 960 / 880 / 960
#                                               total_sum all equal, D960r == D960 byte-identical                HARD
#   PRE-REGISTERED: no gain claimed. P1 <425> loop 111 +- 1 (stated). P2 <425> frame <= 200 B, spill 0, regs <= 44 (HARD).
#     W1 D880/D960: no direction stated (192 B x 12 = 73.7 KB vs x 11 = 67.6 KB; 413 +5.7%, 415 +0.75% -- the margin shrinks).
#        <= 0.998 -> N=25 runs at MB 880 (417-r2: N-dependent MB in the .py); otherwise 960 stays.
#     W2 |D960r - D960| <= 0.3% (normalised). W3 all cells 1710 +- 2% (read norm_ms).
#   ADOPT: all HARD -> 417 is the production binary/.py (N <= 24 values unchanged); N=25's value = 417-r3 (full run,
#     oracle 2,207,893,435,808,352, ~206 h expected, timeout 300 h).
#
# USAGE
#   STATIC_ONLY=1 bash 417_validate.sh                                # S0, GPU zero, ~1 min
#   bash 417_validate.sh 2>&1 | tee 417_validate_stdout.log           # build + S1-S6 + Y2/Y3 (~45 min) + cells (~2 h)
#   SKIP_CELLS=1 bash 417_validate.sh                                 # everything but the GPU cells

set -u

REV="417"
SKIP_BUILD="${SKIP_BUILD:-0}"
PY_SRC="${PY_SRC:-417Py_kernel_maxd16_final.py}"
PY_BIN="${PY_BIN:-${PY_SRC%.py}}"
PREV_PY="${PREV_PY:-416Py_kernel_maxd16_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-417_kernel_maxd16.cu}"
CU_BIN="${CU_BIN:-417_kernel_maxd16}"
PREV_CU="${PREV_CU:-416_kernel_maxd16.cu}"
PREV_BIN="${PREV_BIN:-416_kernel_maxd16}"
REFDFS_SRC="${REFDFS_SRC:-417_ref_dfs.c}"
REF_CU="${REF_CU:-392_kernel_maxd16.cu}"
REF_BIN="${REF_BIN:-392_kernel_maxd16}"
FILTER="${FILTER:-413_filter_maxd16_only.py}"
SLICER="${SLICER:-413_slice_soa7.py}"
PERMUTE="${PERMUTE:-394f_permute_soa7.py}"
REF21="${REF21:-411_validate_20261006_150938/bins/B1_results.bin}"
REF21_SHA="${REF21_SHA:-1bfab090}"
STREAM24="${STREAM24:-constellations_N24_7.bin}"
SCHED24="${SCHED24:-${STREAM24}.soa_ref_361.bin.d16only_415.bin.sched394f.bin}"
EXPECTED24="${EXPECTED24:-66737631}"
STREAM25="${STREAM25:-constellations_N25_7.bin}"
DUMP25="${DUMP25:-${STREAM25}.soa_ref_361.bin}"
SCHED25="${SCHED25:-${DUMP25}.d17only_417.bin.sched394f.bin}"
SASS402_SHA="${SASS402_SHA:-821302f227dcd160}"; SASS402_REAL="${SASS402_REAL:-446}"; SASS402_LOOP="${SASS402_LOOP:-109}"
SASS403_SHA="${SASS403_SHA:-e940cb96e5e65fc4}"; SASS403_REAL="${SASS403_REAL:-445}"; SASS403_LOOP="${SASS403_LOOP:-108}"
SASS423_SHA="${SASS423_SHA:-c8316e18}"; SASS423_REAL="${SASS423_REAL:-447}"; SASS423_LOOP="${SASS423_LOOP:-107}"
SASS424_SHA="${SASS424_SHA:-f60ecee7}"; SASS424_REAL="${SASS424_REAL:-447}"; SASS424_LOOP="${SASS424_LOOP:-107}"
KERNEL_SHA_417="${KERNEL_SHA_417:-b8f82d27baa3044d}"
KERNEL_SHA_416="${KERNEL_SHA_416:-04f3b8818ee02bc7}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
CODON="${CODON:-codon}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
HMB="${HMB:-128}"
NREC="${NREC:-2048}"
CPU24="${CPU24:-2048}"
CPU25="${CPU25:-1024}"
SLICE_B="${SLICE_B:-307200}"      # N=25: 10 records per thread at stride 30,720 (~6.9 ms/record expected -> ~35 min/cell)
COOLDOWN="${COOLDOWN:-15}"
CLK_INTERVAL="${CLK_INTERVAL:-5}"
STATIC_ONLY="${STATIC_ONLY:-0}"
SKIP_CELLS="${SKIP_CELLS:-0}"
ITER_CAP="${ITER_CAP:-4000000000ULL}"
OMP_THREADS="${OMP_THREADS:-8}"
ORACLE21=314666222712
ORACLE23=24233937684440
ORACLE24=227514171973736
ORACLE25=2207893435808352

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
for f in "$PY_SRC" "$HELPER_SRC" "$CU_SRC" "$PREV_CU" "$REF_CU" "$REFDFS_SRC" "$FILTER" "$SLICER" "$PERMUTE"; do
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
{ grep -q "^# 417 " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 417 ...' note block in $PY_SRC (rule 8)"
grep -qE '^REV_TAG:str="417"' "$CODE" && pass "source_rev_tag_is_417" || fail "source_rev_tag_is_417" "wrong REV_TAG"
grep -qF 'CRunnerEntry(17,"./417_kernel_maxd16","NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_entry_maxd17_417_binary_helper_prefix" || fail "source_table_entry" "table entry wrong (maxd_max must be 17)"
grep -qF 'elif N>=22 and N<=25:' "$CODE" && grep -qF 'elif N>=26 and N<=27:' "$CODE" && pass "source_preset_queens_7_at_N25" || fail "source_preset_queens_7_at_N25" "select_dynamic_preset_queens not updated"
grep -qF 'd17only_417.bin' "$CODE" && grep -qF '"MAXDEPTH=16 " if N==24 else "MAXDEPTH=17 "' "$CODE" && pass "source_n25_filter_env_and_suffix_wired" || fail "source_n25_filter_env_and_suffix_wired" "missing"
{ ! grep -E 'CRunnerEntry\(' "$CODE" | grep -q 'NQ_EXTRA_CTX'; } && pass "source_no_table_entry_carries_NQ_EXTRA_CTX" || fail "source_no_table_entry_carries_NQ_EXTRA_CTX" "found"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=960\b" "$CODE" && pass "source_default_max_blocks_is_960" || fail "source_default_max_blocks_is_960" "default not 960"
grep -qF 'd16only_415.bin" if N==24' "$CODE" && grep -qF 'maxd16only_413.bin" if N==23' "$CODE" && pass "source_n23_n24_filter_names_kept" || fail "source_n23_n24_filter_names_kept" "missing"
grep -qE '^STREAM_SCAN_CHUNK:int=1<<21$' "$CODE" && pass "source_414_chunked_scans_kept" || fail "source_414_chunked_scans_kept" "STREAM_SCAN_CHUNK missing"
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "7" && "$PA_" == "7" ]] && pass "py_diff_fingerprint_vs_416Py (removed=7 added=7 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_416Py" "removed=$PR_ added=$PA_, expected 7/7"
else info "py_diff_fingerprint_vs_416Py" "skipped ($PREV_PY absent)"; fi
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
[[ "$KB" == "$KERNEL_SHA_417"* ]] && pass "cu_kernel_region_sha (${KB:0:16})" || fail "cu_kernel_region_sha" "sha ${KB:0:16} != $KERNEL_SHA_417"
KP="$(extract_kernel "/tmp/${REV}_prev_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KP" == "$KERNEL_SHA_416"* ]] && pass "prev_cu_is_416 (${KP:0:16})" || fail "prev_cu_is_416" "sha ${KP:0:16} != $KERNEL_SHA_416"
CR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true); CA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
[[ "$CR_" == "8" && "$CA_" == "14" ]] && pass "cu_code_region_fingerprint_vs_416 (removed=8 added=14)" || fail "cu_code_region_fingerprint" "removed=$CR_ added=$CA_, expected 8/14"
grep -q '^#define PACK425_WIDTH   27' "$CU_SRC" && grep -q '^#define PACK425_IDXSHIFT 27' "$CU_SRC" && pass "cu_pack425_width_27_idxshift_27" || fail "cu_pack425_width_27_idxshift_27" "missing"
grep -q '^#define PACK425_ANCESTOR 16' "$CU_SRC" && grep -q '^#define PACK425_MAXN    25' "$CU_SRC" && pass "cu_pack425_ancestor_16_maxn_25" || fail "cu_pack425_ancestor_16_maxn_25" "missing"
grep -q '^#define PACK424_ANCESTOR 15' "$CU_SRC" && grep -q '^#define PACK424_WIDTH   22' "$CU_SRC" && pass "cu_pack424_constants_unchanged" || fail "cu_pack424_constants_unchanged" "changed"
grep -q '^#define PACK423_ANCESTOR 14' "$CU_SRC" && pass "cu_pack423_ancestor_still_14" || fail "cu_pack423_ancestor_still_14" "changed"
grep -q '^#define MAXD14_ANCESTOR 13' "$CU_SRC" && pass "cu_maxd14_ancestor_still_13" || fail "cu_maxd14_ancestor_still_13" "changed"
grep -q 'MAXDEPTH = int(os.environ.get("MAXDEPTH", "15"))' "$FILTER" && pass "filter_maxdepth_default_15" || fail "filter_maxdepth_default_15" "wrong default"
grep -q "rev417 " "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev417 note in header"
grep -qF 'layout = (N > PACK424_MAXN) ? PACK_LAYOUT_425 : (N > PACK423_MAXN) ? PACK_LAYOUT_424 : (N > PACK403_WIDTH) ? PACK_LAYOUT_423 : PACK_LAYOUT_403;' "$CU_SRC" && pass "cu_auto_layout_n25_is_425 (N=24 -> 424 unchanged)" || fail "cu_auto_layout_n25_is_425" "auto selection line not as expected"
# r2: the CPU-only debug cap (50,000,000 iterations per record) is below real N=24 tasks. The harnesses are built from
# /tmp copies with the cap raised to $ITER_CAP; the sources on disk are untouched (the cap lives in #ifndef __CUDACC__ code).
patched() { sed "s/const uint64_t DEBUG_ITER_CAP = 50000000ULL;/const uint64_t DEBUG_ITER_CAP = ${ITER_CAP};/" "$1" > "/tmp/${REV}_$(basename "$1")"; grep -c "DEBUG_ITER_CAP = ${ITER_CAP}" "/tmp/${REV}_$(basename "$1")"; }
[[ "$(patched "$CU_SRC")" == "1" && "$(patched "$PREV_CU")" == "1" && "$(patched "$REF_CU")" == "1" ]] && pass "cpu_iter_cap_patched_in_tmp_copies ($ITER_CAP)" || fail "cpu_iter_cap_patch" "DEBUG_ITER_CAP line not found in one of the three .cu"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "/tmp/${REV}_$(basename "$CU_SRC")" -lm 2>"/tmp/${REV}_gcc.log" && pass "cu_cpu_harness_builds[417]" || fail "cu_cpu_harness_builds[417]" "$(head -3 /tmp/${REV}_gcc.log)"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_prev" "/tmp/${REV}_$(basename "$PREV_CU")" -lm 2>"/tmp/${REV}_gcc_prev.log" && pass "cu_cpu_harness_builds[416]" || fail "cu_cpu_harness_builds[416]" "$(head -3 /tmp/${REV}_gcc_prev.log)"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_ref" "/tmp/${REV}_$(basename "$REF_CU")" -lm 2>"/tmp/${REV}_gcc_ref.log" && pass "cu_cpu_harness_builds[392]" || fail "cu_cpu_harness_builds[392]" "$(head -3 /tmp/${REV}_gcc_ref.log)"
"$GCC" -O2 -fopenmp -o "/tmp/${REV}_refdfs" "$REFDFS_SRC" 2>"/tmp/${REV}_gcc_refdfs.log" && pass "ref_dfs_builds[417_ref_dfs]" || fail "ref_dfs_builds" "$(head -3 /tmp/${REV}_gcc_refdfs.log)"

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
  [[ -x "$CU_BIN" ]] && pass "existing_binary_used[$CU_BIN] ($(stat -c %y "$CU_BIN" | cut -c1-19); SKIP_BUILD=1)" || { fail "cu_binary_missing" "$CU_BIN absent and SKIP_BUILD=1"; summary_exit; }
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
if [[ "$SKIP_BUILD" == "1" ]]; then info "S1_ptxas" "skipped (no fresh nvcc log)"; FRAME425="?(SKIP_BUILD)"; else
for L in 402 403; do
  ptx_stats "kernel_dfs_iter_gpu_maxd16ILi${L}E"
  [[ "$FR" == "160" && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "S1_ptxas_L${L}_frame160_spill0_regs${RG}" || { fail "S1_ptxas_L${L}" "frame=${FR}B regs=${RG} ${SP}"; summary_exit; }
done
ptx_stats "kernel_dfs_iter_gpu_maxd16ILi423E"
[[ "$FR" == "176" && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "S1_ptxas_L423_frame176_spill0_regs${RG}" || { fail "S1_ptxas_L423" "frame=${FR}B regs=${RG} ${SP} (expected 176 / 0)"; summary_exit; }
ptx_stats "kernel_dfs_iter_gpu_maxd16ILi424E"
[[ "$FR" == "184" && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "S1_ptxas_L424_frame184_spill0_regs${RG}" || { fail "S1_ptxas_L424" "frame=${FR}B regs=${RG} ${SP} (expected 184 / 0, as 415)"; summary_exit; }
ptx_stats "kernel_dfs_iter_gpu_maxd16ILi425E"
[[ -n "$FR" && "$FR" -le 200 && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "S1_P2_ptxas_L425_frame${FR}_spill0_regs${RG} (16 x 12 = 192 B; 12 blocks/SM = $((FR*384)) B of local)" || { fail "S1_P2_ptxas_L425" "frame=${FR}B regs=${RG} ${SP} (expected <= 200 / 0 / <= 44)"; summary_exit; }
FRAME425="$FR"
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
[[ "$S_SHA" == "$SASS424_SHA"* && "$S_REAL" == "$SASS424_REAL" && "$S_LOOP" == "$SASS424_LOOP" ]] && pass "S5_sass_424_is_415s (sha $S_SHA, real $S_REAL, loop $S_LOOP)" || { fail "S5_sass_424_differs_from_415" "sha $S_SHA real $S_REAL loop $S_LOOP != $SASS424_SHA* / $SASS424_REAL / $SASS424_LOOP -- the 425 branch perturbed <424>: stop and read the SASS"; summary_exit; }
read -r S_SHA S_REAL S_LOOP S_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_dfs_iter_gpu_maxd16ILi425E)"
[[ "$S_SHA" != "CANNOT_LOCATE" ]] && pass "S6_sass_425_recorded (sha $S_SHA, real $S_REAL, loop $S_LOOP, frame ${FRAME425} B)" || { fail "S6_sass_425_missing" "no <425> instantiation in the SASS"; summary_exit; }
if [[ "$S_LOOP" -ge 110 && "$S_LOOP" -le 112 ]]; then pass "P1_loop_425_111pm1 (loop $S_LOOP; 416: 111)"; else info "P1_loop_425_outside_statement" "loop $S_LOOP not in 110..112 -- the frame count changed the loop body? read the SASS (informational)"; fi

if [[ "$SKIP_BUILD" == "1" ]]; then [[ -x "$PY_BIN" ]] && pass "existing_py_binary_used[$PY_BIN]" || { fail "py_binary_missing" "$PY_BIN absent and SKIP_BUILD=1"; summary_exit; }
else
rm -f "$PY_BIN"; "$CODON" build -release "$PY_SRC" > "$LOGDIR/06_codon_${REV}.log" 2>&1
[[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN] (no production run in this harness; the binary is the deliverable)" || { fail "codon_build" "$(tail -3 "$LOGDIR/06_codon_${REV}.log")"; summary_exit; }
fi

find_input() { local n="$1" c; for lg in $(ls -t 4*_crunner_logs/crunner_*_N${n}.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { echo "$c"; return; }; done; ls -t constellations_N${n}_*.sched394f.bin 2>/dev/null | head -1; }
IN21="$(find_input 21)"; [[ -f "${IN21:-/nonexistent}" ]] && pass "input21_located ($IN21)" || { fail "input21_located" "no N=21 sched input"; summary_exit; }
if [[ -f "$REF21" ]]; then
  RS="$(sha256sum "$REF21" | cut -c1-8)"
  [[ "$RS" == "$REF21_SHA" ]] && pass "ref21_is_411_B1 ($REF21 sha $RS)" || fail "ref21_is_411_B1" "sha $RS != $REF21_SHA"
else fail "ref21_present" "$REF21 missing -- pass REF21=<411 B1 results.bin>"; fi

# ---------------------------------------------------------------------
# 3. N=24 input: 415's hard links of 414's all_414 files (no recompute)
# ---------------------------------------------------------------------
banner "N=24 input (415's file) and N=25 input (417_build_input25.sh)"
[[ -s "$SCHED24" ]] && pass "n24_415_sched_present ($SCHED24)" || { fail "n24_415_sched" "$SCHED24 missing"; summary_exit; }
[[ -s "$DUMP25" && -s "$SCHED25" ]] && pass "n25_files_present ($SCHED25)" || { fail "n25_files" "$DUMP25 / $SCHED25 missing: run 417_build_input25.sh first"; summary_exit; }
EXPECTED25=$(( $(stat -c %s "$DUMP25") / 40 ))
[[ $(( $(stat -c %s "$SCHED25") / 28 )) -eq "$EXPECTED25" && $(( $(stat -c %s "$SCHED25") % 28 )) -eq 0 && "$EXPECTED25" -gt 0 ]] && pass "H2_sched25_has_${EXPECTED25}_records (= the 361 dump)" || { fail "H2_sched25" "$SCHED25 size does not match the dump ($EXPECTED25 records)"; summary_exit; }
IN25="$SCHED25"
[[ $(( $(stat -c %s "$SCHED24") / 28 )) -eq "$EXPECTED24" && $(( $(stat -c %s "$SCHED24") % 28 )) -eq 0 ]] && pass "sched24_has_${EXPECTED24}_records" || { fail "sched24" "$SCHED24 wrong size"; summary_exit; }
IN24="$SCHED24"
SLB="${REV}_n25_sliceB_${SLICE_B}.bin"; SLC25="${REV}_n25_sliceC_${CPU25}.bin"; SLC="${REV}_n24_sliceC_${CPU24}.bin"
for pair in "$SLB:$SLICE_B:$IN25" "$SLC25:$CPU25:$IN25" "$SLC:$CPU24:$IN24"; do IFS=: read -r f c src <<< "$pair"; if [[ ! -s "$f" || $(( $(stat -c %s "$f") / 28 )) -ne "$c" ]]; then python3 "$SLICER" "$src" "$f" "$c" > "$LOGDIR/13_slice_${f}.log" 2>&1 || true; fi; [[ -s "$f" && $(( $(stat -c %s "$f") / 28 )) -eq "$c" ]] && pass "slice_ready[$f]" || { fail "slice[$f]" "$(tail -1 "$LOGDIR/13_slice_${f}.log" 2>/dev/null)"; summary_exit; }; done

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
"/tmp/${REV}_cpu_prev" 21 "/tmp/${REV}_in21_head.bin" "/tmp/${REV}_y2_416_21.bin" "$NREC" > "$LOGDIR/2_cpu_N21_416.log" 2>&1
"/tmp/${REV}_cpu_cur"  21 "/tmp/${REV}_in21_head.bin" "/tmp/${REV}_y2_417_21.bin" "$NREC" > "$LOGDIR/2_cpu_N21_417auto.log" 2>&1
y2_valid "$LOGDIR/2_cpu_N21_416.log" "/tmp/${REV}_y2_416_21.bin" "$NREC" 416_21 && y2_valid "$LOGDIR/2_cpu_N21_417auto.log" "/tmp/${REV}_y2_417_21.bin" "$NREC" 417_21 || summary_exit
cmp -s "/tmp/${REV}_y2_416_21.bin" "/tmp/${REV}_y2_417_21.bin" && pass "Y2a_cpu_N21_auto_417_equals_416 ($NREC records)" || { fail "Y2a_cpu_N21_auto" "per-record outputs differ"; summary_exit; }
T0=$(date +%s)
"/tmp/${REV}_cpu_ref"  24 "$SLC" "/tmp/${REV}_y2_392_24.bin" > "$LOGDIR/2_cpu_N24_392.log" 2>&1; T1=$(date +%s)
"/tmp/${REV}_cpu_prev" 24 "$SLC" "/tmp/${REV}_y2_416_24.bin" > "$LOGDIR/2_cpu_N24_416.log" 2>&1; T2=$(date +%s)
"/tmp/${REV}_cpu_cur"  24 "$SLC" "/tmp/${REV}_y2_417_24.bin" > "$LOGDIR/2_cpu_N24_417auto.log" 2>&1; T3=$(date +%s)
"/tmp/${REV}_refdfs"   24 "$SLC" "/tmp/${REV}_y2_ref_24.bin" > "$LOGDIR/2_cpu_N24_refdfs.log" 2>&1; T4=$(date +%s)
y2_valid "$LOGDIR/2_cpu_N24_392.log" "/tmp/${REV}_y2_392_24.bin" "$CPU24" 392_24 && y2_valid "$LOGDIR/2_cpu_N24_416.log" "/tmp/${REV}_y2_416_24.bin" "$CPU24" 416_24 \
  && y2_valid "$LOGDIR/2_cpu_N24_417auto.log" "/tmp/${REV}_y2_417_24.bin" "$CPU24" 417_24 && y2_valid "$LOGDIR/2_cpu_N24_refdfs.log" "/tmp/${REV}_y2_ref_24.bin" "$CPU24" ref_24 || summary_exit
cmp -s "/tmp/${REV}_y2_392_24.bin" "/tmp/${REV}_y2_416_24.bin" && pass "Y2c0_cpu_N24_416_equals_392 (control, 392 $((T1-T0)) s / 416 $((T2-T1)) s)" || { fail "Y2c0_cpu_N24_416_vs_392" "the control pair differs"; summary_exit; }
grep -q 'layout=424' "$LOGDIR/2_cpu_N24_417auto.log" && cmp -s "/tmp/${REV}_y2_416_24.bin" "/tmp/${REV}_y2_417_24.bin" && pass "Y2c_cpu_N24_417auto_equals_416 ($CPU24 equal-spaced records, $((T3-T2)) s)" || { fail "Y2c_cpu_N24_417auto_vs_416" "differ or layout not 424; do not run the GPU"; summary_exit; }
cmp -s "/tmp/${REV}_y2_392_24.bin" "/tmp/${REV}_y2_ref_24.bin" && pass "Y2r_cpu_N24_refdfs_equals_392 (the recursive reference holds on $CPU24 real records, $((T4-T3)) s)" || { fail "Y2r_cpu_N24_refdfs_vs_392" "417_ref_dfs differs from 392 on real N=24 data -- the reference is wrong, not the kernel; stop"; summary_exit; }
T0=$(date +%s)
"/tmp/${REV}_cpu_cur" 25 "$SLC25" "/tmp/${REV}_y2_417_25.bin" > "$LOGDIR/2_cpu_N25_417auto.log" 2>&1; T1=$(date +%s)
"/tmp/${REV}_refdfs"  25 "$SLC25" "/tmp/${REV}_y2_ref_25.bin" > "$LOGDIR/2_cpu_N25_refdfs.log" 2>&1; T2=$(date +%s)
y2_valid "$LOGDIR/2_cpu_N25_417auto.log" "/tmp/${REV}_y2_417_25.bin" "$CPU25" 417_25 && y2_valid "$LOGDIR/2_cpu_N25_refdfs.log" "/tmp/${REV}_y2_ref_25.bin" "$CPU25" ref_25 || summary_exit
grep -q 'layout=425' "$LOGDIR/2_cpu_N25_417auto.log" && cmp -s "/tmp/${REV}_y2_417_25.bin" "/tmp/${REV}_y2_ref_25.bin" && pass "Y2e_cpu_N25_417auto425_equals_refdfs ($CPU25 equal-spaced records; 417 $((T1-T0)) s, ref $((T2-T1)) s; total_sum $(grep -o 'total_sum=[0-9]*' "$LOGDIR/2_cpu_N25_417auto.log" | head -1 | cut -d= -f2))" || { fail "Y2e_cpu_N25_417_vs_refdfs" "per-record outputs differ (or abort: $(grep -o '\[[A-Z-]*\]' "$LOGDIR/2_cpu_N25_417auto.log" | head -1)) -- 16 frames / N=25 bit positions: do not run the GPU"; summary_exit; }
banner "Y3: refusals"
"/tmp/${REV}_cpu_cur" 26 "$SLC25" /tmp/${REV}_y3a.bin 8 > "$LOGDIR/2_y3_n26.log" 2>&1; rc=$?
[[ "$rc" == "3" ]] && pass "Y3_N26_refused_rc3" || fail "Y3_N26_refused_rc3" "rc=$rc"
NQ_LAYOUT=424 "/tmp/${REV}_cpu_cur" 25 "$SLC25" /tmp/${REV}_y3b.bin 8 > "$LOGDIR/2_y3_l424_n25.log" 2>&1; rc=$?
[[ "$rc" == "3" ]] && pass "Y3_layout424_at_N25_refused_rc3" || fail "Y3_layout424_at_N25_refused_rc3" "rc=$rc"
NQ_LAYOUT=423 "/tmp/${REV}_cpu_cur" 24 "$SLC" /tmp/${REV}_y3c.bin 8 > "$LOGDIR/2_y3_l423_n24.log" 2>&1; rc=$?
[[ "$rc" == "3" ]] && pass "Y3_layout423_at_N24_refused_rc3" || fail "Y3_layout423_at_N24_refused_rc3" "rc=$rc"

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
same_bin() { local a="$LOGDIR/bins/${1}_results.bin" b="$2"; [[ -f "$b" ]] || b="$LOGDIR/bins/${2}_results.bin"; cmp -s "$a" "$b" && pass "${1}_per_thread_results_identical_to_${2} ($(stat -c %s "$a") bytes)" || fail "${1}_per_thread_results_DIFFER_from_${2}" "not adoptable"; }
oracle_ok() { [[ "${TOTAL[$1]:-}" == "$2" ]] && pass "oracle[$1]" || fail "oracle[$1]" "total_sum=${TOTAL[$1]:-?} != $2"; }

if [[ "$SKIP_CELLS" == "1" ]]; then
  info "cells" "skipped (SKIP_CELLS=1)"
  if [[ "$FAIL" -eq 0 ]]; then info "ADOPTION" "static/CPU gates hold; the GPU cells (B0/B1/D960/D880/D960r) decide -- run without SKIP_CELLS"; else info "ADOPTION" "NOT adopted -- see FAIL lines"; fi
  { echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
  TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
  summary_exit
fi
banner "N=21: B0 $PREV_BIN / B1 $CU_BIN"
run_cell B0 "$PREV_BIN" 21 "$IN21" auto 32 960 403 "$ORACLE21" || summary_exit; oracle_ok B0 "$ORACLE21"; same_bin B0 "$REF21"
sleep "$COOLDOWN"; run_cell B1 "$CU_BIN" 21 "$IN21" auto 32 960 403 "$ORACLE21" || summary_exit; oracle_ok B1 "$ORACLE21"; same_bin B1 B0
info "B1_vs_B0" "raw $(pct "${VAL[B1]}" "${VAL[B0]}")% / normalised $(pct "${NORM[B1]}" "${NORM[B0]}")% (identical SASS: session scatter)"

banner "N=25 slice B ($SLICE_B records): D960 / D880 / D960r (auto = 425, 16 frames)"
sleep "$COOLDOWN"; run_cell D960 "$CU_BIN" 25 "$SLB" auto 32 960 425 || summary_exit
sleep "$COOLDOWN"; run_cell D880 "$CU_BIN" 25 "$SLB" auto 32 880 425 || summary_exit
sleep "$COOLDOWN"; run_cell D960r "$CU_BIN" 25 "$SLB" auto 32 960 425 || summary_exit
[[ "${TOTAL[D880]}" == "${TOTAL[D960]}" && "${TOTAL[D960r]}" == "${TOTAL[D960]}" ]] && pass "sliceB_totals_equal_across_MB (${TOTAL[D960]})" || fail "sliceB_totals_differ" "D960 ${TOTAL[D960]} D880 ${TOTAL[D880]} D960r ${TOTAL[D960r]} -- not adoptable"
same_bin D960r D960
r="$(ratio "${NORM[D880]}" "${NORM[D960]}")"
if le "$r" 0.998; then info "W1_880_faster" "D880/D960 = $r -> N=25 runs at MB 880 (417-r2: N-dependent MB in the .py)"
elif le "$r" 1.002; then info "W1_no_difference" "D880/D960 = $r -> 960 stays"
else info "W1_960_faster" "D880/D960 = $r (+$(pct "${NORM[D880]}" "${NORM[D960]}")%) -> 960 stays"; fi
d="$(abspct "${NORM[D960r]}" "${NORM[D960]}")"; le "$d" 0.3 && pass "W2_D960_repeat_within_0.3pct (${d}%)" || fail "W2_D960_repeat_drift" "${d}% > 0.3% -- read clk_D960*.tsv"
info "N25_estimate" "full run at 960 ~ $(awk -v k="${VAL[D960]}" -v s="$SLICE_B" -v e="$EXPECTED25" 'BEGIN{printf "%.0f s (%.1f h)", k/1000*e/s, k/1000*e/s/3600}') if the slice were representative (it runs 6-26% dear: 413-r3 +6%, 415-r3 +26%)"

CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "W3_sm_clock_1710_all_cells" || fail "W3_sm_clock_1710_all_cells" "see clk_*.tsv (power-capped cells read 1680-1695; compare norm_ms)"
if [[ "$FAIL" -eq 0 ]]; then info "ADOPTION" "417 is the production binary/.py (N <= 24 values unchanged, production SASS identical); N=25's value = 417-r3 (full run)"; else info "ADOPTION" "NOT adopted -- see FAIL lines; 416 stays"; fi

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
nm={r['cell']:float(r['norm_ms']) for r in rows if r['norm_ms'] not in ('?','')}
print(f"{'cell':<6}{'binary':<24}{'N':>3}{'lay':>5}{'blk':>4}{'MB':>5}{'records':>10}{'free_mb':>8}{'kernel_ms':>12}{'norm_ms':>12}{'vs':>9}{'sm':>8}{'C':>4}{'h:mm:ss':>10}")
for r in rows:
    c=r['cell']; v=float(r['kernel_ms']); n=nm.get(c,v); s=v/1000
    base={'B1':'B0','D880':'D960','D960r':'D960'}.get(c)
    d=f"{(n/nm[base]-1)*100:+.3f}%" if base and base in nm else ""
    print(f"{c:<6}{r['binary'][:23]:<24}{r['N']:>3}{r['layout']:>5}{r['block']:>4}{r['max_blocks']:>5}{r['records']:>10}{r['free_mb']:>8}{v:>12.1f}{n:>12.1f}{d:>9}{r['sm_mean']:>8}{r['temp_max']:>4}{int(s//3600):>4}:{int(s%3600//60):02d}:{s%60:04.1f}")
print("(vs: B1->B0, D880/D960r->D960; normalised to 1710 MHz)")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd1[46]" --exclude="*_cpu" --exclude="*slice*.bin" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
