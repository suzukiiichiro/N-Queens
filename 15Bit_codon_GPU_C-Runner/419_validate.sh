#!/usr/bin/env bash
# 419_validate.sh -- the N=25 fix: layout 425's per-depth schedule word (3-bit block codes, 21 depths) replaces
# the 4-bit-nibble word (16 depths) that N=25 schedules (depth 17) overflowed in 417. Single GPU (NQ_CRunner).
#
# WHERE 417 STANDS (2026-11-04): 417-r3 run3 finished (10/26 15:20 -> 11/4 01:34, 202.2 h, rc=0) with
#   total_sum 2,204,354,806,741,818 against the oracle 2,207,893,435,808,352 (-3,538,629,066,534, -0.16 %).
#   N=21..24 values unchanged (reproduced by run1). Cause: see the rev419 header note in 419_kernel_maxd16.cu.
#
# rev419 -- .cu: layout-425-only changes inside process_one_task (3-bit codes in the same hi:lo pair, cur_depth*3
#   extraction, root fc from the mask; non-425 statements verbatim) + a CPU-harness [SCHEDULE-WORD-OVERFLOW] abort;
#   code-region diff vs 417 removed=5 added=25, kernel sha 3b67e26c. r1 (if-constexpr blocks + a new variable) re-ordered
#   <402>'s SASS (sha 109788c8, same 446/109) and was withdrawn at S2. .py: 3 executable lines (tags, CRunnerEntry -> ./419_kernel_maxd16). 419_ref_dfs.c: block codes in a
#   plain array (no packing -- the 417 reference shared the overflow). 419_sched_scan.c: host scan of the schedules.
#
#   S0 static: .py diff 3/3 vs 417Py; .cu diff 7/34 vs 417, kernel sha 3be9c958; PACK constants unchanged; notes;
#      overflow guard present; gcc builds 419/417/392 CPU harnesses, both references, the scan                   HARD
#   S1 ptxas: <402>,<403> 160 B ; <423> 176 B ; <424> 184 B ; <425> <= 200 B ; spill 0 ; regs <= 44             HARD
#   S2-S5 <402>/<403>/<423>/<424> SASS identical to 417 (the fix must not touch them)                            HARD
#   S6 <425> SASS recorded; P1 loop 111 +- 1 (stated: the *4 shift becomes a *3 IMAD)                            info
#   SCAN 419_sched_scan over the whole N=25 sched: runaway 0, max depth <= 20, histogram (17: all)               HARD
#        A = records with a non-zero block code at depth 16 (the affected class; info) -> slice T (1,024 of them)
#   Y2 CPU equivalence (8 threads, DEBUG_ITER_CAP raised in /tmp copies, done line + size checked):
#      Y2a N=21 419 == 417 (2,048 head)                                                                          HARD
#      Y2c N=24 419 auto(424) == 392 (2,048 equal-spaced)                                                        HARD
#      Y2e N=25 C25 (1,024 equal-spaced): 419 == ref419                                                          HARD
#      Y2t N=25 T (1,024 affected): 419 == ref419                                                                HARD
#      Y2x N=25 T: 417 != 419 on at least one record (the diagnosis' refutation test)                           HARD
#      Y2o N=25 T: 417 == ref417 (both carry the overflow; info)
#   Y3 rc=3: N=26 ; NQ_LAYOUT=424 at N=25 ; NQ_LAYOUT=423 at N=24                                                HARD
#   Cells (BLOCK 32, MB 960, helper 1 + 128 MB):
#     B0    417 N=21 auto(403)                 results.bin == REF21 (411 B1 1bfab090)                            HARD
#     B1    419 N=21 auto(403)                 results.bin == B0                                                  HARD
#     T419  419 N=25 slice T (one record per thread)  results.bin[0..2047] == Y2t's 419 CPU output              HARD
#     T417  417 N=25 slice T                   results.bin[0..2047] == Y2o's 417 CPU output (GPU carries the same error; info)
#     D960  419 N=25 slice B (307,200)         total_sum recorded; W1 vs 417's D960 (3,109,581.5 ms, norm 3,090,305.7)
#   PRE-REGISTERED: no gain claimed. P1 loop 111 +- 1. P2 <425> frame 192 B, spill 0 (HARD). W1 D960(419)/D960(417)
#     normalised in 0.98..1.02. Refutations: Y2x (417 == 419 on T -> the cause is elsewhere, stop); S2-S5 changed.
#   ADOPT: all HARD -> 419 is the production binary/.py (N <= 24 unchanged); N=25's value = 419-r3 (419_r3_full25.sh).
#
# USAGE (NQ_CRunner, after run3; ~3 h: static+build 3 min, scan 5 min, Y2 ~1.5 h (N=25 CPU records take ~1 s each
#        per thread), cells ~1 h 10 (D960 52 min). RUN_REF417=1 adds the 417 reference on T (info only, +20 min).
#   STATIC_ONLY=1 bash 419_validate.sh
#   bash 419_validate.sh 2>&1 | tee 419_validate_stdout.log
#   set -o pipefail; bash 419_validate.sh 2>&1 | tee 419_validate_stdout.log && bash 419_r3_full25.sh 2>&1 | tee 419_r3_full25_stdout.log
#     (pipefail, or the && gates on tee's exit status, not the harness's)
#   SKIP_CELLS=1 ...   SKIP_BUILD=1 ...   SKIP_D960=1 (drop the 52-min slice B cell)

set -u
REV="419"
SKIP_BUILD="${SKIP_BUILD:-0}"; STATIC_ONLY="${STATIC_ONLY:-0}"; SKIP_CELLS="${SKIP_CELLS:-0}"; SKIP_D960="${SKIP_D960:-0}"; RUN_REF417="${RUN_REF417:-0}"
PY_SRC="${PY_SRC:-419Py_kernel_maxd16_final.py}"; PY_BIN="${PY_BIN:-${PY_SRC%.py}}"
PREV_PY="${PREV_PY:-417Py_kernel_maxd16_final.py}"
CU_SRC="${CU_SRC:-419_kernel_maxd16.cu}"; CU_BIN="${CU_BIN:-419_kernel_maxd16}"
PREV_CU="${PREV_CU:-417_kernel_maxd16.cu}"; PREV_BIN="${PREV_BIN:-417_kernel_maxd16}"
REF_CU="${REF_CU:-392_kernel_maxd16.cu}"
REFDFS_SRC="${REFDFS_SRC:-419_ref_dfs.c}"; PREV_REFDFS_SRC="${PREV_REFDFS_SRC:-417_ref_dfs.c}"
SCAN_SRC="${SCAN_SRC:-419_sched_scan.c}"
SLICER="${SLICER:-413_slice_soa7.py}"
REF21="${REF21:-411_validate_20261006_150938/bins/B1_results.bin}"; REF21_SHA="${REF21_SHA:-1bfab090}"
SCHED24="${SCHED24:-constellations_N24_7.bin.soa_ref_361.bin.d16only_415.bin.sched394f.bin}"; EXPECTED24=66737631
SCHED25="${SCHED25:-constellations_N25_7.bin.soa_ref_361.bin.d17only_417.bin.sched394f.bin}"; EXPECTED25=98138478
SASS402_SHA="821302f227dcd160"; SASS402_REAL=446; SASS402_LOOP=109
SASS403_SHA="e940cb96e5e65fc4"; SASS403_REAL=445; SASS403_LOOP=108
SASS423_SHA="c8316e18"; SASS423_REAL=447; SASS423_LOOP=107
SASS424_SHA="f60ecee7"; SASS424_REAL=447; SASS424_LOOP=107
KERNEL_SHA_419="3b67e26c23bf3454"; KERNEL_SHA_417="b8f82d27baa3044d"
D960_417_MS="${D960_417_MS:-3109581.5}"; D960_417_NORM="${D960_417_NORM:-3090305.7}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"; CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"; CODON="${CODON:-codon}"; GCC="${GCC:-gcc}"; ARCH="${ARCH:-sm_86}"
HMB="${HMB:-128}"; NREC="${NREC:-2048}"; CPU24="${CPU24:-2048}"; CPU25="${CPU25:-1024}"; TREC="${TREC:-1024}"; SLICE_B="${SLICE_B:-307200}"
COOLDOWN="${COOLDOWN:-15}"; CLK_INTERVAL="${CLK_INTERVAL:-5}"; ITER_CAP="${ITER_CAP:-4000000000ULL}"; OMP_THREADS="${OMP_THREADS:-8}"
ORACLE21=314666222712; ORACLE25=2207893435808352
TS="$(date +%Y%m%d_%H%M%S)"; LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"; TSV="$LOGDIR/${REV}_results.tsv"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f",(a-b)/b*100}'; }
ratio() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.4f",a/b}'; }
le() { awk -v x="$1" -v y="$2" 'BEGIN{exit !(x<=y)}'; }
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS (registered refutations are listed here too -- read the label):"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi; exit 0; }

# ---------------------------------------------------------------------
# 1. Static (S0)
# ---------------------------------------------------------------------
for f in "$PY_SRC" "$CU_SRC" "$PREV_CU" "$REF_CU" "$REFDFS_SRC" "$PREV_REFDFS_SRC" "$SCAN_SRC" "$SLICER"; do [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"; done
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }
py_code_region() { python3 -c "
import sys
lines=open(sys.argv[1],encoding='utf-8').read().split('\n')
i=next(i for i,l in enumerate(lines) if l.startswith('import '))
sys.stdout.write('\n'.join(l for l in lines[i:] if not l.lstrip().startswith('#')))
" "$1"; }
py_code_region "$PY_SRC" > "/tmp/${REV}_code_only.py"; CODE="/tmp/${REV}_code_only.py"
NOTE_LINES=$(awk '/^# =+$/{f=1} f&&/^#/{n++} END{print n+0}' "$PY_SRC")
{ grep -q "^# 419 " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 419 ...' note block (rule 8)"
grep -qE '^REV_TAG:str="419"' "$CODE" && pass "source_rev_tag_is_419" || fail "source_rev_tag_is_419" "wrong REV_TAG"
grep -qF 'CRunnerEntry(17,"./419_kernel_maxd16","NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_entry_maxd17_419_binary_helper_prefix" || fail "source_table_entry" "table entry wrong"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=960\b" "$CODE" && pass "source_default_max_blocks_is_960" || fail "source_default_max_blocks_is_960" "default not 960"
grep -qF 'elif N>=22 and N<=25:' "$CODE" && grep -qF 'd17only_417.bin' "$CODE" && pass "source_417_n25_wiring_kept" || fail "source_417_n25_wiring_kept" "missing"
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_417Py (removed=3 added=3 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_417Py" "removed=$PR_ added=$PA_, expected 3/3"
else info "py_diff_fingerprint_vs_417Py" "skipped ($PREV_PY absent)"; fi
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
[[ "$KB" == "$KERNEL_SHA_419"* ]] && pass "cu_kernel_region_sha (${KB:0:16})" || fail "cu_kernel_region_sha" "sha ${KB:0:16} != $KERNEL_SHA_419"
KP="$(extract_kernel "/tmp/${REV}_prev_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KP" == "$KERNEL_SHA_417"* ]] && pass "prev_cu_is_417 (${KP:0:16})" || fail "prev_cu_is_417" "sha ${KP:0:16} != $KERNEL_SHA_417"
CR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true); CA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
[[ "$CR_" == "5" && "$CA_" == "25" ]] && pass "cu_code_region_fingerprint_vs_417 (removed=5 added=25)" || fail "cu_code_region_fingerprint" "removed=$CR_ added=$CA_, expected 5/25"
grep -q 'const int sh425 = schedule_depth \* 3;' "$CU_SRC" && grep -q 'cur_depth \* ((layout == PACK_LAYOUT_425) ? 3 : 4)' "$CU_SRC" && grep -q '(schedule_lo & 7u) | ((future_check_mask & 1u) << 3)' "$CU_SRC" && pass "cu_425_3bit_schedule_word_present (pack / extract / root fc from the mask)" || fail "cu_425_3bit_schedule_word" "pack/extract lines not found"
grep -q 'SCHEDULE-WORD-OVERFLOW' "$CU_SRC" && pass "cu_cpu_overflow_guard_present" || fail "cu_cpu_overflow_guard" "missing"
grep -q '^#define PACK425_ANCESTOR 16' "$CU_SRC" && grep -q '^#define PACK425_MAXN    25' "$CU_SRC" && grep -q '^#define PACK424_ANCESTOR 15' "$CU_SRC" && grep -q '^#define PACK423_ANCESTOR 14' "$CU_SRC" && pass "cu_pack_constants_unchanged" || fail "cu_pack_constants_unchanged" "changed"
grep -q "rev419 " "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev419 note"
grep -q 'bc\[sdepth\] = (uint8_t)(nibble & 7u);' "$REFDFS_SRC" && ! grep -q 'schedule64' "$REFDFS_SRC" && pass "ref_dfs_keeps_codes_in_an_array (no packed word)" || fail "ref_dfs_representation" "419_ref_dfs.c still packs the schedule"
patched() { sed "s/const uint64_t DEBUG_ITER_CAP = 50000000ULL;/const uint64_t DEBUG_ITER_CAP = ${ITER_CAP};/" "$1" > "/tmp/${REV}_$(basename "$1")"; grep -c "DEBUG_ITER_CAP = ${ITER_CAP}" "/tmp/${REV}_$(basename "$1")"; }
[[ "$(patched "$CU_SRC")" == "1" && "$(patched "$PREV_CU")" == "1" && "$(patched "$REF_CU")" == "1" ]] && pass "cpu_iter_cap_patched_in_tmp_copies ($ITER_CAP)" || fail "cpu_iter_cap_patch" "DEBUG_ITER_CAP line not found"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur"  "/tmp/${REV}_$(basename "$CU_SRC")" -lm 2>"/tmp/${REV}_gcc.log" && pass "cu_cpu_harness_builds[419]" || fail "cu_cpu_harness_builds[419]" "$(head -3 /tmp/${REV}_gcc.log)"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_prev" "/tmp/${REV}_$(basename "$PREV_CU")" -lm 2>"/tmp/${REV}_gcc_prev.log" && pass "cu_cpu_harness_builds[417]" || fail "cu_cpu_harness_builds[417]" "$(head -3 /tmp/${REV}_gcc_prev.log)"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_ref"  "/tmp/${REV}_$(basename "$REF_CU")" -lm 2>"/tmp/${REV}_gcc_ref.log" && pass "cu_cpu_harness_builds[392]" || fail "cu_cpu_harness_builds[392]" "$(head -3 /tmp/${REV}_gcc_ref.log)"
"$GCC" -O2 -fopenmp -o "/tmp/${REV}_refdfs" "$REFDFS_SRC" 2>"/tmp/${REV}_gcc_refdfs.log" && pass "ref_dfs_builds[419_ref_dfs]" || fail "ref_dfs_builds" "$(head -3 /tmp/${REV}_gcc_refdfs.log)"
"$GCC" -O2 -fopenmp -o "/tmp/${REV}_refdfs417" "$PREV_REFDFS_SRC" 2>"/tmp/${REV}_gcc_refdfs417.log" && pass "ref_dfs_builds[417_ref_dfs]" || fail "ref_dfs_builds_417" "$(head -3 /tmp/${REV}_gcc_refdfs417.log)"
"$GCC" -O2 -o "/tmp/${REV}_scan" "$SCAN_SRC" 2>"/tmp/${REV}_gcc_scan.log" && pass "sched_scan_builds" || fail "sched_scan_builds" "$(head -3 /tmp/${REV}_gcc_scan.log)"
if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Build, ptxas (S1), SASS (S2-S6), codon
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR/sass" "$LOGDIR/bins"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; df -h . 2>&1; free -g 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
[[ ! -x "$NVCC" ]] && NVCC="nvcc"; [[ ! -x "$CUOBJDUMP" ]] && CUOBJDUMP="cuobjdump"
banner "Building"
FRAME425="?"
if [[ "$SKIP_BUILD" == "1" ]]; then
  [[ -x "$CU_BIN" ]] && pass "existing_binary_used[$CU_BIN] ($(stat -c %y "$CU_BIN" | cut -c1-19); SKIP_BUILD=1)" || { fail "cu_binary_missing" "$CU_BIN absent and SKIP_BUILD=1"; summary_exit; }
  info "S1_ptxas" "skipped (no fresh nvcc log)"
else
  rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc_${REV}.log" 2>&1
  [[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "$(grep -i error "$LOGDIR/05_nvcc_${REV}.log" | head -3)"; summary_exit; }
  ptx_stats() { FR="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"; SP="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes spill stores, [0-9]* bytes spill loads' | head -1)"; RG="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"; }
  for pair in 402:160 403:160 423:176 424:184; do IFS=: read -r L want <<< "$pair"; ptx_stats "kernel_dfs_iter_gpu_maxd16ILi${L}E"
    [[ "$FR" == "$want" && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "S1_ptxas_L${L}_frame${want}_spill0_regs${RG}" || { fail "S1_ptxas_L${L}" "frame=${FR}B regs=${RG} ${SP} (expected $want / 0 / <= 44)"; summary_exit; }; done
  ptx_stats "kernel_dfs_iter_gpu_maxd16ILi425E"
  [[ -n "$FR" && "$FR" -le 200 && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "S1_P2_ptxas_L425_frame${FR}_spill0_regs${RG} (192 expected)" || { fail "S1_P2_ptxas_L425" "frame=${FR}B regs=${RG} ${SP}"; summary_exit; }
  FRAME425="$FR"
fi
[[ -x "$PREV_BIN" ]] && pass "prev_binary_available[$PREV_BIN] (417, run3's binary)" || { "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$PREV_BIN" "$PREV_CU" -lcuda > "$LOGDIR/05b_nvcc_prev.log" 2>&1; [[ -x "$PREV_BIN" ]] && pass "prev_binary_built[$PREV_BIN]" || { fail "prev_binary" "cannot build $PREV_BIN"; summary_exit; }; }
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
sass_gate() { local L="$1" sha="$2" real="$3" loop="$4" tag="$5"; read -r S_SHA S_REAL S_LOOP S_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" "kernel_dfs_iter_gpu_maxd16ILi${L}E")"
  [[ "$S_SHA" == "$sha"* && "$S_REAL" == "$real" && "$S_LOOP" == "$loop" ]] && pass "${tag}_sass_${L}_is_417s (sha $S_SHA, real $S_REAL, loop $S_LOOP)" || { fail "${tag}_sass_${L}_differs_from_417" "sha $S_SHA real $S_REAL loop $S_LOOP != $sha* / $real / $loop -- the 425-only edit leaked into <$L>: stop"; summary_exit; }; }
sass_gate 402 "$SASS402_SHA" "$SASS402_REAL" "$SASS402_LOOP" S2
sass_gate 403 "$SASS403_SHA" "$SASS403_REAL" "$SASS403_LOOP" S3
sass_gate 423 "$SASS423_SHA" "$SASS423_REAL" "$SASS423_LOOP" S4
sass_gate 424 "$SASS424_SHA" "$SASS424_REAL" "$SASS424_LOOP" S5
read -r S_SHA S_REAL S_LOOP S_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_dfs_iter_gpu_maxd16ILi425E)"
[[ "$S_SHA" != "CANNOT_LOCATE" ]] && pass "S6_sass_425_recorded (sha $S_SHA, real $S_REAL, loop $S_LOOP, frame ${FRAME425} B; 417: ae5129bc / loop 111)" || { fail "S6_sass_425_missing" "no <425> instantiation"; summary_exit; }
if [[ "$S_LOOP" -ge 110 && "$S_LOOP" -le 112 ]]; then pass "P1_loop_425_111pm1 (loop $S_LOOP)"; else info "P1_loop_425_outside_statement" "loop $S_LOOP not in 110..112 -- read the SASS (informational; W1 decides the cost)"; fi
if [[ "$SKIP_BUILD" == "1" ]]; then [[ -x "$PY_BIN" ]] && pass "existing_py_binary_used[$PY_BIN]" || { fail "py_binary_missing" "$PY_BIN absent and SKIP_BUILD=1"; summary_exit; }
else rm -f "$PY_BIN"; "$CODON" build -release "$PY_SRC" > "$LOGDIR/06_codon_${REV}.log" 2>&1
  [[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN] (the -g path names ./$CU_BIN; 419_r3_full25.sh runs it)" || { fail "codon_build" "$(tail -3 "$LOGDIR/06_codon_${REV}.log")"; summary_exit; }; fi

find_input() { local n="$1" c; for lg in $(ls -t 4*_crunner_logs/crunner_*_N${n}.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { echo "$c"; return; }; done; ls -t constellations_N${n}_*.sched394f.bin 2>/dev/null | head -1; }
IN21="$(find_input 21)"; [[ -f "${IN21:-/nonexistent}" ]] && pass "input21_located ($IN21)" || { fail "input21_located" "no N=21 sched input"; summary_exit; }
[[ -s "$SCHED24" && $(( $(stat -c %s "$SCHED24") / 28 )) -eq "$EXPECTED24" ]] && pass "sched24_present" || { fail "sched24" "$SCHED24 missing/wrong size"; summary_exit; }
[[ -s "$SCHED25" && $(( $(stat -c %s "$SCHED25") / 28 )) -eq "$EXPECTED25" ]] && pass "sched25_present ($EXPECTED25 records)" || { fail "sched25" "$SCHED25 missing/wrong size"; summary_exit; }
if [[ -f "$REF21" ]]; then RS="$(sha256sum "$REF21" | cut -c1-8)"; [[ "$RS" == "$REF21_SHA" ]] && pass "ref21_is_411_B1 (sha $RS)" || fail "ref21_is_411_B1" "sha $RS != $REF21_SHA"; else fail "ref21_present" "$REF21 missing"; fi

# ---------------------------------------------------------------------
# 3. SCAN: every N=25 schedule, the affected class, slice T
# ---------------------------------------------------------------------
banner "SCAN: 419_sched_scan over $SCHED25 (two passes, ~2 min each)"
T0=$(date +%s); "/tmp/${REV}_scan" 25 "$SCHED25" > "$LOGDIR/1_scan.log" 2>&1; cat "$LOGDIR/1_scan.log"
RUNAWAY="$(grep -o 'runaway=[0-9]*' "$LOGDIR/1_scan.log" | cut -d= -f2)"; HIST="$(grep -o '\[sched-scan-histogram\].*' "$LOGDIR/1_scan.log" | cut -d']' -f2)"
MAXD="$(echo "$HIST" | tr ' ' '\n' | grep -o '^[0-9]*' | sort -n | tail -1)"
AFF="$(grep -o 'records_with_nonzero_code_at_depth>=16=[0-9]*' "$LOGDIR/1_scan.log" | cut -d= -f2)"
[[ "$RUNAWAY" == "0" && -n "$MAXD" && "$MAXD" -le 20 ]] && pass "SCAN_runaway0_maxdepth${MAXD}_le20 (histogram:$HIST; $(( $(date +%s)-T0 )) s)" || { fail "SCAN" "runaway=$RUNAWAY maxdepth=${MAXD:-?} (425's word holds depths 0..20)"; summary_exit; }
[[ "$HIST" == " 17:$EXPECTED25" ]] && pass "SCAN_every_record_depth_17 (as the 417 filter histogram)" || info "SCAN_histogram" "$HIST (not all 17)"
info "SCAN_affected_class" "A=$AFF records have a non-zero block code at depth 16 ($(awk -v a="$AFF" -v m="$EXPECTED25" 'BEGIN{printf "%.4f", 100*a/m}')%); their 417 counts are wrong"
[[ "${AFF:-0}" -ge 1 ]] || { fail "SCAN_no_affected_records" "A=0 -- the overflow would be harmless, the diagnosis is wrong: stop"; summary_exit; }
SLT="${REV}_n25_sliceT_${TREC}.bin"; EVERY=$(( AFF / TREC )); [[ "$EVERY" -lt 1 ]] && EVERY=1
"/tmp/${REV}_scan" 25 "$SCHED25" --extract "/tmp/${REV}_sliceT_all.bin" --every "$EVERY" > "$LOGDIR/1_scan_extract.log" 2>&1
head -c $((TREC*28)) "/tmp/${REV}_sliceT_all.bin" > "$SLT"; rm -f "/tmp/${REV}_sliceT_all.bin"
TN=$(( $(stat -c %s "$SLT") / 28 )); [[ "$TN" -ge 1 ]] && pass "sliceT_ready[$SLT] ($TN affected records, every ${EVERY}-th)" || { fail "sliceT" "empty"; summary_exit; }
SLC25="${REV}_n25_sliceC_${CPU25}.bin"; SLC="${REV}_n24_sliceC_${CPU24}.bin"; SLB="${REV}_n25_sliceB_${SLICE_B}.bin"
for pair in "$SLC25:$CPU25:$SCHED25" "$SLC:$CPU24:$SCHED24" "$SLB:$SLICE_B:$SCHED25"; do IFS=: read -r f c src <<< "$pair"; if [[ ! -s "$f" || $(( $(stat -c %s "$f") / 28 )) -ne "$c" ]]; then python3 "$SLICER" "$src" "$f" "$c" > "$LOGDIR/13_slice_${f}.log" 2>&1 || true; fi; [[ -s "$f" && $(( $(stat -c %s "$f") / 28 )) -eq "$c" ]] && pass "slice_ready[$f]" || { fail "slice[$f]" "$(tail -1 "$LOGDIR/13_slice_${f}.log" 2>/dev/null)"; summary_exit; }; done

# ---------------------------------------------------------------------
# 4. CPU equivalence (Y2) and refusals (Y3)
# ---------------------------------------------------------------------
banner "Y2: CPU equivalence ($OMP_THREADS threads)"
export OMP_NUM_THREADS="$OMP_THREADS"
y2_valid() { local lg="$1" out="$2" n="$3" lab="$4"
  grep -q "\[kernel-cputest-done\] N=[0-9]* records=${n} " "$lg" || { fail "Y2_invalid[$lab]" "no done line (abort: $(grep -o '\[[A-Z-]*\]' "$lg" | grep -v progress | tail -1))"; return 1; }
  [[ -f "$out" && $(stat -c %s "$out") -eq $((n*8)) ]] || { fail "Y2_invalid[$lab]" "output size $(stat -c %s "$out" 2>/dev/null) != $((n*8))"; return 1; }; return 0; }
head -c $((NREC*28)) "$IN21" > "/tmp/${REV}_in21_head.bin"
"/tmp/${REV}_cpu_prev" 21 "/tmp/${REV}_in21_head.bin" "/tmp/${REV}_y2_417_21.bin" "$NREC" > "$LOGDIR/2_cpu_N21_417.log" 2>&1
"/tmp/${REV}_cpu_cur"  21 "/tmp/${REV}_in21_head.bin" "/tmp/${REV}_y2_419_21.bin" "$NREC" > "$LOGDIR/2_cpu_N21_419.log" 2>&1
y2_valid "$LOGDIR/2_cpu_N21_417.log" "/tmp/${REV}_y2_417_21.bin" "$NREC" 417_21 && y2_valid "$LOGDIR/2_cpu_N21_419.log" "/tmp/${REV}_y2_419_21.bin" "$NREC" 419_21 || summary_exit
cmp -s "/tmp/${REV}_y2_417_21.bin" "/tmp/${REV}_y2_419_21.bin" && pass "Y2a_cpu_N21_419_equals_417 ($NREC records)" || { fail "Y2a_cpu_N21" "differ -- the 425-only edit changed layout 403: stop"; summary_exit; }
T0=$(date +%s)
"/tmp/${REV}_cpu_ref" 24 "$SLC" "/tmp/${REV}_y2_392_24.bin" > "$LOGDIR/2_cpu_N24_392.log" 2>&1; T1=$(date +%s)
"/tmp/${REV}_cpu_cur" 24 "$SLC" "/tmp/${REV}_y2_419_24.bin" > "$LOGDIR/2_cpu_N24_419.log" 2>&1; T2=$(date +%s)
y2_valid "$LOGDIR/2_cpu_N24_392.log" "/tmp/${REV}_y2_392_24.bin" "$CPU24" 392_24 && y2_valid "$LOGDIR/2_cpu_N24_419.log" "/tmp/${REV}_y2_419_24.bin" "$CPU24" 419_24 || summary_exit
grep -q 'layout=424' "$LOGDIR/2_cpu_N24_419.log" && cmp -s "/tmp/${REV}_y2_392_24.bin" "/tmp/${REV}_y2_419_24.bin" && pass "Y2c_cpu_N24_419auto424_equals_392 ($CPU24 records; 392 $((T1-T0)) s, 419 $((T2-T1)) s)" || { fail "Y2c_cpu_N24" "differ or layout not 424: stop"; summary_exit; }
T0=$(date +%s)
"/tmp/${REV}_cpu_cur" 25 "$SLC25" "/tmp/${REV}_y2_419_25c.bin" > "$LOGDIR/2_cpu_N25C_419.log" 2>&1; T1=$(date +%s)
"/tmp/${REV}_refdfs"  25 "$SLC25" "/tmp/${REV}_y2_ref_25c.bin" > "$LOGDIR/2_cpu_N25C_ref419.log" 2>&1; T2=$(date +%s)
y2_valid "$LOGDIR/2_cpu_N25C_419.log" "/tmp/${REV}_y2_419_25c.bin" "$CPU25" 419_25c && y2_valid "$LOGDIR/2_cpu_N25C_ref419.log" "/tmp/${REV}_y2_ref_25c.bin" "$CPU25" ref_25c || summary_exit
grep -q 'layout=425' "$LOGDIR/2_cpu_N25C_419.log" && cmp -s "/tmp/${REV}_y2_419_25c.bin" "/tmp/${REV}_y2_ref_25c.bin" && pass "Y2e_cpu_N25_C_419auto425_equals_ref419 ($CPU25 equal-spaced; 419 $((T1-T0)) s, ref $((T2-T1)) s)" || { fail "Y2e_cpu_N25_C" "differ (or abort: $(grep -o '\[[A-Z-]*\]' "$LOGDIR/2_cpu_N25C_419.log" | head -1)): stop"; summary_exit; }
banner "Y2t/Y2x/Y2o: slice T ($TN affected records) -- 419 == ref419 ; 417 != 419 ; 417 == ref417"
T0=$(date +%s)
"/tmp/${REV}_cpu_cur"    25 "$SLT" "/tmp/${REV}_y2_419_T.bin"    > "$LOGDIR/2_cpu_N25T_419.log" 2>&1; T1=$(date +%s)
"/tmp/${REV}_refdfs"     25 "$SLT" "/tmp/${REV}_y2_ref_T.bin"    > "$LOGDIR/2_cpu_N25T_ref419.log" 2>&1; T2=$(date +%s)
"/tmp/${REV}_cpu_prev"   25 "$SLT" "/tmp/${REV}_y2_417_T.bin"    > "$LOGDIR/2_cpu_N25T_417.log" 2>&1; T3=$(date +%s)
if [[ "$RUN_REF417" == "1" ]]; then "/tmp/${REV}_refdfs417" 25 "$SLT" "/tmp/${REV}_y2_ref417_T.bin" > "$LOGDIR/2_cpu_N25T_ref417.log" 2>&1; fi; T4=$(date +%s)
y2_valid "$LOGDIR/2_cpu_N25T_419.log" "/tmp/${REV}_y2_419_T.bin" "$TN" 419_T && y2_valid "$LOGDIR/2_cpu_N25T_ref419.log" "/tmp/${REV}_y2_ref_T.bin" "$TN" ref_T || summary_exit
cmp -s "/tmp/${REV}_y2_419_T.bin" "/tmp/${REV}_y2_ref_T.bin" && pass "Y2t_cpu_N25_T_419_equals_ref419 ($TN affected records; 419 $((T1-T0)) s, ref $((T2-T1)) s)" || { fail "Y2t_cpu_N25_T" "419 differs from the array reference on the affected class: the fix is wrong, stop"; summary_exit; }
if y2_valid "$LOGDIR/2_cpu_N25T_417.log" "/tmp/${REV}_y2_417_T.bin" "$TN" 417_T; then
  ND="$(python3 - "/tmp/${REV}_y2_417_T.bin" "/tmp/${REV}_y2_419_T.bin" <<'EOF'
import sys,struct
a=open(sys.argv[1],'rb').read(); b=open(sys.argv[2],'rb').read(); n=len(a)//8
A=struct.unpack('<%dQ'%n,a); B=struct.unpack('<%dQ'%n,b)
d=sum(1 for x,y in zip(A,B) if x!=y); print(d, sum(A)-sum(B), sum(B))
EOF
)"
  read -r NDIFF DSUM S419 <<< "$ND"
  [[ "$NDIFF" -ge 1 ]] && pass "Y2x_cpu_N25_T_417_differs_from_419 ($NDIFF of $TN records differ; 417-419 sum $DSUM; 419 total $S419)" || { fail "Y2x_REFUTED_417_equals_419_on_affected_records" "the overflow does not change these counts -- the cause is elsewhere: stop"; summary_exit; }
  [[ "$RUN_REF417" == "1" ]] && y2_valid "$LOGDIR/2_cpu_N25T_ref417.log" "/tmp/${REV}_y2_ref417_T.bin" "$TN" ref417_T && { cmp -s "/tmp/${REV}_y2_417_T.bin" "/tmp/${REV}_y2_ref417_T.bin" && info "Y2o" "417 == ref417 on T (both carry the overflow, as diagnosed; $((T4-T3)) s)" || info "Y2o" "417 != ref417 on T (the two undefined shifts differ on this host; still both wrong vs ref419: $(cmp -s "/tmp/${REV}_y2_ref417_T.bin" "/tmp/${REV}_y2_ref_T.bin" && echo 'ref417 == ref419 ?!' || echo 'ref417 != ref419')"; }
else info "Y2x" "417 CPU aborted on T ($(grep -o '\[[A-Z-]*\]' "$LOGDIR/2_cpu_N25T_417.log" | head -1)) -- the T417 GPU cell takes over the comparison"; NDIFF=-1; fi
banner "Y3: refusals"
"/tmp/${REV}_cpu_cur" 26 "$SLC25" /tmp/${REV}_y3a.bin 8 > "$LOGDIR/2_y3_n26.log" 2>&1; rc=$?; [[ "$rc" == "3" ]] && pass "Y3_N26_refused_rc3" || fail "Y3_N26_refused_rc3" "rc=$rc"
NQ_LAYOUT=424 "/tmp/${REV}_cpu_cur" 25 "$SLC25" /tmp/${REV}_y3b.bin 8 > "$LOGDIR/2_y3_l424_n25.log" 2>&1; rc=$?; [[ "$rc" == "3" ]] && pass "Y3_layout424_at_N25_refused_rc3" || fail "Y3_layout424_at_N25_refused_rc3" "rc=$rc"
NQ_LAYOUT=423 "/tmp/${REV}_cpu_cur" 24 "$SLC" /tmp/${REV}_y3c.bin 8 > "$LOGDIR/2_y3_l423_n24.log" 2>&1; rc=$?; [[ "$rc" == "3" ]] && pass "Y3_layout423_at_N24_refused_rc3" || fail "Y3_layout423_at_N24_refused_rc3" "rc=$rc"

# ---------------------------------------------------------------------
# 5. Cells
# ---------------------------------------------------------------------
printf 'cell\tbinary\tN\tlayout\tblock\tmax_blocks\trecords\thelpers\tfree_mb\tkernel_ms\tnorm_ms\ttotal_sum\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
CLKPID=""
gpu_gate() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; [[ -z "$apps" ]] && return 0; fail "gpu_empty_before[$1]" "$(echo "$apps" | tr '\n' ' ')"; return 1; }
gpu_after() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; [[ -z "$apps" ]] || fail "gpu_residue_after[$1]" "$(echo "$apps" | tr '\n' ' ')"; }
clk_start() { local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"; ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPID=$!; }
clk_stop() { [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 && $6+0>50 && $4+0>150 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.1f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_$1.tsv"); }
declare -A VAL NORM TOTAL
parse_log() { local cell="$1" b="$2" n="$3" lg="$4" start="$5" want_layout="$6" blk="$7" mb="$8" recs="$9"; local KMS TOT FREE HL LAYV XC CFG NORMV
  KMS="$(grep -o 'kernel_ms=[0-9.]*' "$lg" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  FREE="$(grep -o 'free_mb=[0-9]*' "$lg" | head -1 | cut -d= -f2)"; HL="$(grep -o 'helpers=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  LAYV="$(grep -o '\[gpu-layout\] N=[0-9]* requested=[a-z0-9]* layout=[0-9]*' "$lg" | head -1 | grep -o 'layout=[0-9]*' | cut -d= -f2)"; XC="$(grep -o 'extra_ctx=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  CFG="$(grep -o '\[gpu-config\] BLOCK=[0-9]* MAX_BLOCKS=[0-9]*' "$lg" | head -1)"
  NORMV="$(awk -v k="${KMS:-0}" -v s="${SMMEAN/\?/0}" 'BEGIN{ if(s>0) printf "%.1f", k*s/1710; else print "?" }')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$b" "$n" "${LAYV:-?}" "$blk" "$mb" "$recs" "${HL:-?}" "${FREE:-?}" "${KMS:-?}" "$NORMV" "${TOT:-?}" "$SMMEAN" "$SMMIN" "$TMAX" "$start" >> "$TSV"
  [[ "${LAYV:-}" == "$want_layout" ]] || { fail "layout[$cell]" "layout=${LAYV:-?}, expected $want_layout"; return 1; }
  [[ "${HL:-}" == "1" && "${XC:-}" == "0" ]] || { fail "ctx_state[$cell]" "helpers=${HL:-?} extra_ctx=${XC:-?}"; return 1; }
  [[ "$CFG" == "[gpu-config] BLOCK=$blk MAX_BLOCKS=$mb" ]] || { fail "config[$cell]" "'${CFG:-<none>}'"; return 1; }
  [[ -n "${KMS:-}" && -n "${TOT:-}" ]] || { fail "run[$cell]" "no kernel_ms/total_sum: $(tail -2 "$lg" | tr '\n' ' ')"; return 1; }
  VAL[$cell]="$KMS"; NORM[$cell]="$NORMV"; TOTAL[$cell]="$TOT"
  info "$cell" "$b N=$n layout=${LAYV:-?} MB=$mb records=$recs kernel_ms=$KMS norm_ms=$NORMV (sm_mean $SMMEAN, ${TMAX}C) total_sum=$TOT"; }
run_cell() { local cell="$1" bin="$2" n="$3" in="$4" blk="$5" mb="$6" want="$7" oracle="${8:-}"; gpu_gate "$cell" || return 1
  local lg="$LOGDIR/3_${cell}.log" start recs; start="$(date -Is)"; recs=$(( $(stat -c %s "$in") / 28 )); clk_start "$cell"
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX NQ_LAYOUT=auto NQ_BLOCK="$blk" NQ_MAX_BLOCKS="$mb" NQ_HELPER_CTX=1 NQ_HELPER_MB="$HMB" "./$bin" "$n" "$in" "$LOGDIR/bins/${cell}_results.bin" $oracle > "$lg" 2>&1 || true
  clk_stop "$cell"; gpu_after "$cell"; parse_log "$cell" "$bin" "$n" "$lg" "$start" "$want" "$blk" "$mb" "$recs"; }
same_bin() { local a="$LOGDIR/bins/${1}_results.bin" b="$2"; [[ -f "$b" ]] || b="$LOGDIR/bins/${2}_results.bin"; cmp -s "$a" "$b" && pass "${1}_per_thread_results_identical_to_${2}" || fail "${1}_per_thread_results_DIFFER_from_${2}" "not adoptable"; }
oracle_ok() { [[ "${TOTAL[$1]:-}" == "$2" ]] && pass "oracle[$1]" || fail "oracle[$1]" "total_sum=${TOTAL[$1]:-?} != $2"; }
if [[ "$SKIP_CELLS" == "1" ]]; then info "cells" "skipped (SKIP_CELLS=1)"; TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]"; summary_exit; fi

banner "N=21: B0 $PREV_BIN / B1 $CU_BIN (same SASS -> byte-identical)"
run_cell B0 "$PREV_BIN" 21 "$IN21" 32 960 403 "$ORACLE21" || summary_exit; oracle_ok B0 "$ORACLE21"; same_bin B0 "$REF21"
sleep "$COOLDOWN"; run_cell B1 "$CU_BIN" 21 "$IN21" 32 960 403 "$ORACLE21" || summary_exit; oracle_ok B1 "$ORACLE21"; same_bin B1 B0
info "B1_vs_B0" "raw $(pct "${VAL[B1]}" "${VAL[B0]}")% / normalised $(pct "${NORM[B1]}" "${NORM[B0]}")% (identical SASS: session scatter)"

banner "T419 / T417: slice T on the GPU, one record per thread -- per-record equality with the CPU"
sleep "$COOLDOWN"; run_cell T419 "$CU_BIN" 25 "$SLT" 32 960 425 || summary_exit
head -c $((TN*8)) "$LOGDIR/bins/T419_results.bin" > "/tmp/${REV}_T419_head.bin"
cmp -s "/tmp/${REV}_T419_head.bin" "/tmp/${REV}_y2_419_T.bin" && pass "T419_gpu_per_record_equals_419_cpu ($TN records; GPU == CPU == ref419 on the affected class)" || fail "T419_gpu_vs_cpu" "GPU per-record results differ from the 419 CPU harness on T -- stop"
python3 - "$LOGDIR/bins/T419_results.bin" "$TN" <<'EOF' | sed 's/^/INFO  T419_tail: /'
import sys,struct
b=open(sys.argv[1],'rb').read(); n=int(sys.argv[2]); v=struct.unpack('<%dQ'%(len(b)//8),b)
print(f"threads beyond the slice all zero: {all(x==0 for x in v[n:])} (stride {len(v)})")
EOF
sleep "$COOLDOWN"; run_cell T417 "$PREV_BIN" 25 "$SLT" 32 960 425 || summary_exit
head -c $((TN*8)) "$LOGDIR/bins/T417_results.bin" > "/tmp/${REV}_T417_head.bin"
if [[ -f "/tmp/${REV}_y2_417_T.bin" && $(stat -c %s "/tmp/${REV}_y2_417_T.bin") -eq $((TN*8)) ]]; then cmp -s "/tmp/${REV}_T417_head.bin" "/tmp/${REV}_y2_417_T.bin" && info "T417" "417 GPU == 417 CPU on T (the GPU carries the same overflow)" || info "T417" "417 GPU != 417 CPU on T (the undefined shifts differ between sm_86 and x86 -- both wrong)"; fi
cmp -s "/tmp/${REV}_T417_head.bin" "/tmp/${REV}_T419_head.bin" && fail "T417_equals_T419_on_affected_records" "the GPU 417 agrees with 419 on T: the diagnosis does not explain the GPU deficit -- stop" || pass "T417_gpu_differs_from_T419_gpu (417 total ${TOTAL[T417]} vs 419 ${TOTAL[T419]}: 417-419 = $(( TOTAL[T417] - TOTAL[T419] )))"
info "deficit_extrapolation" "T is every ${EVERY}-th affected record: (417-419 on T) x $EVERY = $(( (TOTAL[T417] - TOTAL[T419]) * EVERY )) vs run3's deficit -3,538,629,066,534 (order-of-magnitude check only)"

if [[ "$SKIP_D960" != "1" ]]; then
  banner "D960: 419 on the N=25 slice B ($SLICE_B records; 417's D960 3,109,581.5 ms / norm 3,090,305.7, ~52 min)"
  sleep "$COOLDOWN"; run_cell D960 "$CU_BIN" 25 "$SLB" 32 960 425 || summary_exit
  r="$(ratio "${NORM[D960]}" "$D960_417_NORM")"
  if le 0.98 "$r" && le "$r" 1.02; then pass "W1_D960_419_vs_417_within_2pct ($r; raw $(ratio "${VAL[D960]}" "$D960_417_MS"))"; else info "W1_D960_outside_statement" "419/417 = $r (raw $(ratio "${VAL[D960]}" "$D960_417_MS")) -- read the SASS loop; the fix stands regardless"; fi
  info "D960_total" "419 total_sum ${TOTAL[D960]} (417's D960: 6,899,707,878,380; the difference is the affected records inside slice B)"
fi
CLK_OK=1; for c in B0 B1 T419 T417 D960; do [[ -n "${VAL[$c]:-}" ]] || continue; m="$(awk -F'\t' -v c="$c" '$1==c{print $13}' "$TSV" | tail -1)"; [[ "$m" == "?" || -z "$m" ]] && continue; awk -v x="$m" 'BEGIN{d=x-1710; if(d<0)d=-d; exit !(d<=34)}' || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "W3_sm_clock_1710_all_cells" || info "W3_sm_clock" "power-capped cells read below 1676; compare norm_ms"
if [[ "$FAIL" -eq 0 ]]; then info "ADOPTION" "419 is the production binary/.py (N <= 24 unchanged); N=25's value = 419-r3 (bash 419_r3_full25.sh, ~202 h, oracle $ORACLE25)"; else info "ADOPTION" "NOT adopted -- see FAIL lines"; fi
banner "Results"; python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import sys
rows=[l.rstrip('\n').split('\t') for l in open(sys.argv[1])]
w=[max(len(r[i]) for r in rows if i<len(r)) for i in range(len(rows[0]))]
for r in rows: print('  '.join(c.ljust(w[i]) for i,c in enumerate(r)))
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -f "/tmp/${REV}_y2_419_T.bin" "$LOGDIR/bins/Y2t_419_cpu_T.bin" 2>/dev/null; cp -f "/tmp/${REV}_y2_ref_T.bin" "$LOGDIR/bins/Y2t_ref419_T.bin" 2>/dev/null; cp -f "/tmp/${REV}_y2_417_T.bin" "$LOGDIR/bins/Y2o_417_cpu_T.bin" 2>/dev/null; cp -f "$SLT" "$LOGDIR/bins/" 2>/dev/null
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd1[46]" --exclude="*slice[BC]*.bin" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
