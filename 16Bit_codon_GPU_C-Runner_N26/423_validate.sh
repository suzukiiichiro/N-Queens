#!/usr/bin/env bash
# 423_validate.sh -- the logic trial: NQ_LOGIC=mrv (dynamic row choice) against the 419 search, N=25, one A10G (NQ_CRunner).
#
# WHERE 419 STANDS: production, N=21 97,946 / N=22 849,096 / N=23 7,717,284 / N=24 73,112,528 / N=25 728,149,056 ms (202.3 h).
#   N=26 extrapolates to ~2,000 GPU-hours (A10G x 4: ~21 days), N=27 to ~21,000. 2026-11-18 CPU probe: 92.5 % of the DFS steps
#   are inside subtrees with no solution; choosing the row dynamically cut the steps to 26 % and the CPU time 1.61x.
#
# rev423 -- .cu: 419 + process_one_task_mrv / kernel_mrv_gpu, selected by NQ_LOGIC=mrv. Unset = 419 exactly (kernel region sha
#   3b67e26c unchanged; the five instantiations' SASS must equal the 419 binary's). Code region vs 419: removed=3 added=379.
#   .py: 3 executable lines (tags, CRunnerEntry -> ./423_kernel_maxd16); the -g path does not set NQ_LOGIC.
#   423_rows_scan.c: host scan proving the leaf count (see the .cu note).
#
#   S0 static: .py 3/3 vs 419Py; kernel region sha == 419's; .cu 3/379; notes; gcc builds the CPU harness, the reference, the scan   HARD
#   S1 ptxas: <402>,<403> 160 B; <423> 176; <424> 184; <425> <= 200; spill 0; regs <= 44                                               HARD
#            mrv: frame <= 128 B, spill 0 (clang route on Claude's side: 112 B, 45 registers)                                         HARD
#   S2-S6 the five instantiations: SASS == the 419 binary's (sha / real / loop). A failure here does NOT stop the trial cells -- it
#         only means 423 cannot replace 419 on the -g path as it is (PROD line in the summary).                                       HARD(prod)
#   S7 mrv SASS recorded: sha, real instructions, shortest backward loop holding a local load (= the row loop)                       info
#   SCAN 423_rows_scan over the whole N=25 input: no record whose rows can use more columns than there are rows; rows <= 18; pos <= 31  HARD
#   Y2 CPU (8 threads; done line + size checked before any cmp):
#      Y2c N=25 slice C (1,024 equal-spaced): mrv == 419_ref_dfs                                                                      HARD
#      Y2t N=25 slice T (1,024 with a mark at depth 16): mrv == 419_ref_dfs                                                          HARD
#      Y2d N=25 first 64 of C: dfs (NQ_LOGIC unset) == 419_ref_dfs                                                                    HARD
#   Y3 rc=3: N=26 ; NQ_LOGIC=bad                                                                                                      HARD
#   Cells (BLOCK 32, helper 1 + 128 MB; MB 960 unless noted):
#     B1     423 dfs N=21                      results.bin == REF21 (411 B1 1bfab090), oracle                                         HARD
#     TM     423 mrv N=25 slice T (one record per thread)   results.bin[0..1023] == Y2t's CPU output                                  HARD
#     DM960  423 mrv N=25 slice B (307,200)    total_sum 6,910,807,391,840 ; results.bin sha c881e856 (= 419's D960, per thread)      HARD
#     DD960  423 dfs N=25 slice B              only with RUN_DREF=1 (+52 min): same-session anchor, results.bin == DM960 (HARD)
#     M<mb>  423 mrv N=25 slice B at MAX_BLOCKS in SWEEP_MB (default "1280"): total_sum HARD, time info. Skipped when P1 is refuted
#            or SKIP_SWEEP=1. (1280 = 16 one-warp blocks per SM, the sm_86 ceiling; 16 x 32 x 112 B = 57 KB stays inside L1.)
#   PRE-REGISTERED: P1 DM960 / 419's D960 (3,099,327 ms raw; normalised when 419_validate's tsv is found) in 0.40 .. 0.65.
#     Refuted: >= 0.90. ADOPTION (agreed 2026-11-19): every HARD and the ratio <= 0.50 -> 424 puts NQ_LOGIC=mrv on the -g path.
#     Otherwise: recorded and closed, 419 stays production.
#
# USAGE (NQ_CRunner, GPU idle; about 1.5-2 h: static+build 3 min, scan 3 min, Y2 20-40 min, cells 50-75 min)
#   STATIC_ONLY=1 bash 423_validate.sh
#   bash 423_validate.sh 2>&1 | tee 423_validate_stdout.log          (inside screen; detach with Ctrl+A then D)
#   r2 (2026-11-19 14:30, harness only): codon build is opt-in (RUN_CODON=1); a missing N=21 input / REF21 skips B1 instead of stopping (IN21=, REF21= override).
#   SKIP_CELLS=1 ...   SKIP_BUILD=1 ...   SKIP_Y2=1 (cells only; TM then compares nothing)   SKIP_SWEEP=1   RUN_DREF=1   SWEEP_MB="1120 1280"

set -u
REV="423"
SKIP_BUILD="${SKIP_BUILD:-0}"; STATIC_ONLY="${STATIC_ONLY:-0}"; SKIP_CELLS="${SKIP_CELLS:-0}"; SKIP_Y2="${SKIP_Y2:-0}"; SKIP_SWEEP="${SKIP_SWEEP:-0}"; RUN_DREF="${RUN_DREF:-0}"; RUN_CODON="${RUN_CODON:-0}"
PY_SRC="${PY_SRC:-423Py_kernel_maxd16_final.py}"; PY_BIN="${PY_BIN:-${PY_SRC%.py}}"
PREV_PY="${PREV_PY:-419Py_kernel_maxd16_final.py}"
CU_SRC="${CU_SRC:-423_kernel_maxd16.cu}"; CU_BIN="${CU_BIN:-423_kernel_maxd16}"
PREV_CU="${PREV_CU:-419_kernel_maxd16.cu}"; PREV_BIN="${PREV_BIN:-419_kernel_maxd16}"
REFDFS_SRC="${REFDFS_SRC:-419_ref_dfs.c}"; ROWS_SRC="${ROWS_SRC:-423_rows_scan.c}"
REF21="${REF21:-411_validate_20261006_150938/bins/B1_results.bin}"; REF21_SHA="${REF21_SHA:-1bfab090}"
SCHED25="${SCHED25:-constellations_N25_7.bin.soa_ref_361.bin.d17only_417.bin.sched394f.bin}"; EXPECTED25="${EXPECTED25:-98138478}"
SLC25="${SLC25:-419_n25_sliceC_1024.bin}"; SLT="${SLT:-419_n25_sliceT_1024.bin}"; SLB="${SLB:-419_n25_sliceB_307200.bin}"
SLC25_SHA="${SLC25_SHA:-253e101a17f5e3a2}"; SLT_SHA="${SLT_SHA:-5671650fa938a7cf}"
SASS402_SHA="821302f227dcd160"; SASS403_SHA="e940cb96e5e65fc4"; SASS423_SHA="c8316e18"; SASS424_SHA="f60ecee7"
KERNEL_SHA_419="3b67e26c23bf3454"
D960_419_MS="${D960_419_MS:-3099327}"; D960_419_NORM="${D960_419_NORM:-}"; D960_419_TOTAL="${D960_419_TOTAL:-6910807391840}"; D960_419_BIN_SHA="${D960_419_BIN_SHA:-c881e856}"
SWEEP_MB="${SWEEP_MB:-1280}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"; CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"; CODON="${CODON:-codon}"; GCC="${GCC:-gcc}"; ARCH="${ARCH:-sm_86}"
HMB="${HMB:-128}"; DFS_CHECK="${DFS_CHECK:-64}"; NSL="${NSL:-1024}"; SLICE_B="${SLICE_B:-307200}"
COOLDOWN="${COOLDOWN:-15}"; CLK_INTERVAL="${CLK_INTERVAL:-5}"; ITER_CAP="${ITER_CAP:-4000000000ULL}"; OMP_THREADS="${OMP_THREADS:-8}"
ORACLE21=314666222712
TS="$(date +%Y%m%d_%H%M%S)"; LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"; TSV="$LOGDIR/${REV}_results.tsv"

PASS=0; FAIL=0; PROD_OK=1; declare -a FAILED_CHECKS=()
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
for f in "$PY_SRC" "$CU_SRC" "$PREV_CU" "$REFDFS_SRC" "$ROWS_SRC"; do [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"; done
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }
py_code_region() { python3 -c "
import sys
lines=open(sys.argv[1],encoding='utf-8').read().split('\n')
i=next(i for i,l in enumerate(lines) if l.startswith('import '))
sys.stdout.write('\n'.join(l for l in lines[i:] if not l.lstrip().startswith('#')))
" "$1"; }
py_code_region "$PY_SRC" > "/tmp/${REV}_code_only.py"; CODE="/tmp/${REV}_code_only.py"
NOTE_LINES=$(awk '/^# =+$/{f=1} f&&/^#/{n++} END{print n+0}' "$PY_SRC")
{ grep -q "^# 423 " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 423 ...' note block (rule 8)"
grep -qE '^REV_TAG:str="423"' "$CODE" && pass "source_rev_tag_is_423" || fail "source_rev_tag_is_423" "wrong REV_TAG"
grep -qF 'CRunnerEntry(17,"./423_kernel_maxd16","NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_entry_maxd17_423_binary_helper_prefix" || fail "source_table_entry" "table entry wrong"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=960\b" "$CODE" && pass "source_default_max_blocks_is_960" || fail "source_default_max_blocks_is_960" "default not 960"
! grep -F 'CRunnerEntry(' "$CODE" | grep -q 'NQ_LOGIC' && pass "py_does_not_set_NQ_LOGIC (the -g path runs the 419 search)" || fail "py_sets_NQ_LOGIC" "the .py must not select the trial search in this revision"
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_419Py (removed=3 added=3 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_419Py" "removed=$PR_ added=$PA_, expected 3/3"
else info "py_diff_fingerprint_vs_419Py" "skipped ($PREV_PY absent)"; fi
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
[[ "$KB" == "$KERNEL_SHA_419"* ]] && pass "cu_kernel_region_sha_is_419s (${KB:0:16}: process_one_task .. template kernel untouched)" || fail "cu_kernel_region_sha" "sha ${KB:0:16} != $KERNEL_SHA_419 -- the 419 search was edited"
KP="$(extract_kernel "/tmp/${REV}_prev_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KP" == "$KERNEL_SHA_419"* ]] && pass "prev_cu_is_419 (${KP:0:16})" || fail "prev_cu_is_419" "sha ${KP:0:16} != $KERNEL_SHA_419"
CR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true); CA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
[[ "$CR_" == "3" && "$CA_" == "379" ]] && pass "cu_code_region_fingerprint_vs_419 (removed=3 added=379)" || fail "cu_code_region_fingerprint" "removed=$CR_ added=$CA_, expected 3/379"
grep -q '^static uint64_t process_one_task_mrv(' "$CU_SRC" && grep -q '^__global__ void kernel_mrv_gpu(' "$CU_SRC" && grep -q 'getenv("NQ_LOGIC")' "$CU_SRC" && pass "cu_mrv_search_present (process_one_task_mrv / kernel_mrv_gpu / NQ_LOGIC)" || fail "cu_mrv_search_present" "missing"
grep -q '^#define MRV_MAXROWS 18' "$CU_SRC" && grep -q '^#define MRV_SWITCH  0' "$CU_SRC" && grep -q '^#define MRV_FORCED  1' "$CU_SRC" && pass "cu_mrv_constants (18 rows, switch 0, forced 1)" || fail "cu_mrv_constants" "changed"
grep -q "rev423 " "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev423 note"
patched() { sed "s/const uint64_t DEBUG_ITER_CAP = 50000000ULL;/const uint64_t DEBUG_ITER_CAP = ${ITER_CAP};/" "$1" > "/tmp/${REV}_$(basename "$1")"; grep -c "DEBUG_ITER_CAP = ${ITER_CAP}" "/tmp/${REV}_$(basename "$1")"; }
[[ "$(patched "$CU_SRC")" == "1" ]] && pass "cpu_iter_cap_patched_in_tmp_copy ($ITER_CAP; the dfs check Y2d needs it)" || fail "cpu_iter_cap_patch" "DEBUG_ITER_CAP line not found"
{ "$GCC" -O2 -mpopcnt -fopenmp -x c -o "/tmp/${REV}_cpu_cur"  "/tmp/${REV}_$(basename "$CU_SRC")" -lm 2>"/tmp/${REV}_gcc.log" || "$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur"  "/tmp/${REV}_$(basename "$CU_SRC")" -lm 2>"/tmp/${REV}_gcc.log"; } && pass "cu_cpu_harness_builds[423] (-mpopcnt when the host has it: the mrv search counts bits on every row)" || fail "cu_cpu_harness_builds[423]" "$(head -3 /tmp/${REV}_gcc.log)"
"$GCC" -O2 -fopenmp -o "/tmp/${REV}_refdfs" "$REFDFS_SRC" 2>"/tmp/${REV}_gcc_refdfs.log" && pass "ref_dfs_builds[419_ref_dfs]" || fail "ref_dfs_builds" "$(head -3 /tmp/${REV}_gcc_refdfs.log)"
"$GCC" -O2 -o "/tmp/${REV}_rows_scan" "$ROWS_SRC" 2>"/tmp/${REV}_gcc_scan.log" && pass "rows_scan_builds" || fail "rows_scan_builds" "$(head -3 /tmp/${REV}_gcc_scan.log)"
for pair in "$SLC25:$NSL:$SLC25_SHA" "$SLT:$NSL:$SLT_SHA" "$SLB:$SLICE_B:"; do IFS=: read -r f c want <<< "$pair"
  if [[ -s "$f" && $(( $(stat -c %s "$f") / 28 )) -eq "$c" ]]; then s="$(sha256sum "$f" | cut -c1-16)"
    if [[ -z "$want" || "$s" == "$want" ]]; then pass "slice_present[$f] ($c records, sha $s)"; else fail "slice_sha[$f]" "sha $s != $want (the file Claude checked on 2026-11-19)"; fi
  else fail "slice_present[$f]" "missing or not $c records (419_validate.sh builds it)"; fi; done
if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Build, ptxas (S1), SASS (S2-S7), codon
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR/sass" "$LOGDIR/bins"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; df -h . 2>&1; free -g 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
[[ ! -x "$NVCC" ]] && NVCC="nvcc"; [[ ! -x "$CUOBJDUMP" ]] && CUOBJDUMP="cuobjdump"
banner "Building"
FRAME_MRV="?"; REGS_MRV="?"
if [[ "$SKIP_BUILD" == "1" ]]; then
  [[ -x "$CU_BIN" ]] && pass "existing_binary_used[$CU_BIN] ($(stat -c %y "$CU_BIN" | cut -c1-19); SKIP_BUILD=1)" || { fail "cu_binary_missing" "$CU_BIN absent and SKIP_BUILD=1"; summary_exit; }
  info "S1_ptxas" "skipped (no fresh nvcc log)"
else
  rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc_${REV}.log" 2>&1
  [[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "$(grep -i error "$LOGDIR/05_nvcc_${REV}.log" | head -3)"; summary_exit; }
  ptx_stats() { FR="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"; SP="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes spill stores, [0-9]* bytes spill loads' | head -1)"; RG="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"; }
  for pair in 402:160 403:160 423:176 424:184; do IFS=: read -r L want <<< "$pair"; ptx_stats "kernel_dfs_iter_gpu_maxd16ILi${L}E"
    [[ "$FR" == "$want" && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "S1_ptxas_L${L}_frame${want}_spill0_regs${RG}" || { fail "S1_ptxas_L${L}" "frame=${FR}B regs=${RG} ${SP} (expected $want / 0 / <= 44)"; PROD_OK=0; }; done
  ptx_stats "kernel_dfs_iter_gpu_maxd16ILi425E"
  [[ -n "$FR" && "$FR" -le 200 && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "S1_ptxas_L425_frame${FR}_spill0_regs${RG} (192 expected)" || { fail "S1_ptxas_L425" "frame=${FR}B regs=${RG} ${SP}"; PROD_OK=0; }
  ptx_stats "kernel_mrv_gpu"; FRAME_MRV="${FR:-?}"; REGS_MRV="${RG:-?}"
  [[ -n "$FR" && "$FR" -le 128 && "$SP" == "0 bytes spill stores, 0 bytes spill loads" ]] && pass "S1_ptxas_mrv_frame${FR}_spill0_regs${RG} (clang route: 112 B / 45; <425> is 192 B)" || { fail "S1_ptxas_mrv" "frame=${FR:-?}B regs=${RG:-?} ${SP:-no ptxas line} (expected <= 128 B, 0 spill)"; summary_exit; }
fi
[[ -x "$PREV_BIN" ]] && pass "prev_binary_available[$PREV_BIN] (419, the production binary)" || { "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$PREV_BIN" "$PREV_CU" -lcuda > "$LOGDIR/05b_nvcc_prev.log" 2>&1; [[ -x "$PREV_BIN" ]] && pass "prev_binary_built[$PREV_BIN]" || { fail "prev_binary" "cannot build $PREV_BIN"; summary_exit; }; }
# sass_stat <dump> <function substring> [load mnemonic prefix]: sha16 / real instructions / shortest backward loop holding that load / all lines
sass_stat() { python3 - "$1" "$2" "${3:-LDL.64}" <<'PYEOF'
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
    if any(x[1].startswith(sys.argv[3]) for x in b) and (best is None or len(b)<len(best)): best=b
print(hashlib.sha256('\n'.join(s for _,s in I).encode()).hexdigest()[:16], real, len(best) if best else -1, len(I))
PYEOF
}
"$CUOBJDUMP" -sass "./$CU_BIN" > "$LOGDIR/sass/${CU_BIN}.txt" 2>&1
"$CUOBJDUMP" -sass "./$PREV_BIN" > "$LOGDIR/sass/${PREV_BIN}.txt" 2>&1
for trip in "402:$SASS402_SHA:S2" "403:$SASS403_SHA:S3" "423:$SASS423_SHA:S4" "424:$SASS424_SHA:S5" "425::S6"; do IFS=: read -r L known tag <<< "$trip"
  read -r N_SHA N_REAL N_LOOP N_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" "kernel_dfs_iter_gpu_maxd16ILi${L}E")"
  read -r P_SHA P_REAL P_LOOP P_TOT <<< "$(sass_stat "$LOGDIR/sass/${PREV_BIN}.txt" "kernel_dfs_iter_gpu_maxd16ILi${L}E")"
  if [[ "$N_SHA" != "CANNOT_LOCATE" && "$N_SHA" == "$P_SHA" && "$N_REAL" == "$P_REAL" && "$N_LOOP" == "$P_LOOP" ]]; then pass "${tag}_sass_${L}_is_419s (sha $N_SHA, real $N_REAL, loop $N_LOOP)"
  else fail "${tag}_sass_${L}_differs_from_419" "423: $N_SHA / $N_REAL / $N_LOOP ; 419 binary: $P_SHA / $P_REAL / $P_LOOP -- the added search moved <$L>; the trial cells below are still valid, the -g path must stay on 419"; PROD_OK=0; fi
  [[ -n "$known" && "$P_SHA" != "CANNOT_LOCATE" && "$P_SHA" != "$known"* ]] && info "${tag}_prev_binary_sha" "419 binary's <$L> is $P_SHA, the recorded value is $known (different toolkit or build flags? the comparison above is binary against binary)"
done
read -r M_SHA M_REAL M_LOOP M_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_mrv_gpu LDL)"
[[ "$M_SHA" != "CANNOT_LOCATE" ]] && pass "S7_sass_mrv_recorded (sha $M_SHA, real $M_REAL, shortest loop with a local load $M_LOOP, frame ${FRAME_MRV} B, regs ${REGS_MRV}; <425>: real 454, loop 111, 192 B)" || { fail "S7_sass_mrv_missing" "no kernel_mrv_gpu in the binary"; summary_exit; }
if [[ "$RUN_CODON" != "1" ]]; then info "codon_build" "not run (r2: -g stays on 419Py in this revision; RUN_CODON=1 builds $PY_SRC, which needs rev386_validation_helpers.py beside it)"
else rm -f "$PY_BIN"; "$CODON" build -release "$PY_SRC" > "$LOGDIR/06_codon_${REV}.log" 2>&1
  [[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN] (not run here: -g stays on 419Py until 424)" || fail "codon_build" "$(tail -3 "$LOGDIR/06_codon_${REV}.log")"; fi

find_input() { local n="$1" c; for lg in $(ls -t 4*_crunner_logs/crunner_*_N${n}.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { echo "$c"; return; }; done; ls -t constellations_N${n}_*.sched394f.bin 2>/dev/null | head -1; }
IN21="${IN21:-$(find_input 21)}"; HAVE21=1
[[ -f "${IN21:-/nonexistent}" ]] && pass "input21_located ($IN21)" || { HAVE21=0; PROD_OK=0; info "input21_not_found" "no N=21 sched input in this directory (set IN21=<path>): B1 is skipped, so the default path is checked by SASS only"; }
SCAN_IN="$SCHED25"; SCAN_N="$EXPECTED25"
[[ -s "$SCHED25" && $(( $(stat -c %s "$SCHED25") / 28 )) -eq "$EXPECTED25" ]] && pass "sched25_present ($EXPECTED25 records)" || { SCAN_IN="$SLB"; SCAN_N="$SLICE_B"; info "sched25_not_found" "$SCHED25 is not in this directory (set SCHED25=<path>): SCAN runs on slice B ($SLICE_B records) instead -- the whole-input scan is still owed before any full N=25 run"; }
HAVEREF21=0
if [[ -f "$REF21" ]]; then RS="$(sha256sum "$REF21" | cut -c1-8)"; [[ "$RS" == "$REF21_SHA" ]] && { HAVEREF21=1; pass "ref21_is_411_B1 (sha $RS)"; } || fail "ref21_is_411_B1" "sha $RS != $REF21_SHA"; else info "ref21_not_found" "$REF21 missing (set REF21=<path>): B1 is checked against the oracle only"; fi
if [[ -z "$D960_419_NORM" ]]; then
  for t in $(ls -t 419_validate_*/419_results.tsv 2>/dev/null); do v="$(awk -F'\t' '$1=="D960"{print $11}' "$t" | tail -1)"; k="$(awk -F'\t' '$1=="D960"{print $10}' "$t" | tail -1)"
    if [[ -n "$v" && "$v" != "?" ]]; then D960_419_NORM="$v"; info "D960_419_anchor" "from $t: kernel_ms=$k norm_ms=$v"; break; fi; done
fi
[[ -z "$D960_419_NORM" ]] && info "D960_419_anchor" "419_validate's tsv not found: the ratio is raw only (419 D960 kernel_ms $D960_419_MS; set D960_419_NORM= to normalise)"

# ---------------------------------------------------------------------
# 3. SCAN: every N=25 record -- the condition the leaf count rests on
# ---------------------------------------------------------------------
banner "SCAN: 423_rows_scan over $SCAN_IN (one pass)"
T0=$(date +%s); "/tmp/${REV}_rows_scan" 25 "$SCAN_IN" > "$LOGDIR/1_rows_scan.log" 2>&1; SCRC=$?; cat "$LOGDIR/1_rows_scan.log"
rsv() { grep -o "$1=[0-9]*" "$LOGDIR/1_rows_scan.log" | head -1 | cut -d= -f2; }
[[ "$SCRC" == "0" && "$(rsv records)" == "$SCAN_N" && "$(rsv cover_gt_rows)" == "0" && "$(rsv runaway)" == "0" ]] && pass "SCAN_no_record_can_use_more_columns_than_rows (records $(rsv records), max_rows $(rsv max_rows), max_pos $(rsv max_pos); $(( $(date +%s)-T0 )) s)" || { fail "SCAN_rows" "rc=$SCRC cover_gt_rows=$(rsv cover_gt_rows) max_rows=$(rsv max_rows) max_pos=$(rsv max_pos) runaway=$(rsv runaway) -- the mrv leaf count is not proven for this input: stop"; summary_exit; }
info "SCAN_detail" "decided_at_root=$(rsv decided_at_root) empty_static_row=$(rsv empty_static_row) cover_lt_rows=$(rsv cover_lt_rows) free_ne_rows=$(rsv free_ne_rows) col0_free_but_unused=$(rsv col0_free_but_unused)"

# ---------------------------------------------------------------------
# 4. CPU equivalence (Y2) and refusals (Y3)
# ---------------------------------------------------------------------
export OMP_NUM_THREADS="$OMP_THREADS"
y2_valid() { local lg="$1" out="$2" n="$3" lab="$4"
  grep -q "\[kernel-cputest-done\] N=[0-9]* records=${n} " "$lg" || { fail "Y2_invalid[$lab]" "no done line (abort: $(grep -o '\[[A-Z-]*\]' "$lg" | grep -v progress | tail -1))"; return 1; }
  [[ -f "$out" && $(stat -c %s "$out") -eq $((n*8)) ]] || { fail "Y2_invalid[$lab]" "output size $(stat -c %s "$out" 2>/dev/null) != $((n*8))"; return 1; }; return 0; }
TN="$NSL"; Y2T_OK=0
if [[ "$SKIP_Y2" == "1" ]]; then info "Y2" "skipped (SKIP_Y2=1): TM has no CPU output to compare with"
else
  banner "Y2: CPU equivalence ($OMP_THREADS threads): mrv == 419_ref_dfs on slices C and T; dfs == ref on $DFS_CHECK"
  for sl in C T; do [[ "$sl" == "C" ]] && in="$SLC25" || in="$SLT"
    T0=$(date +%s); "/tmp/${REV}_refdfs" 25 "$in" "/tmp/${REV}_y2_ref_${sl}.bin" > "$LOGDIR/2_cpu_N25${sl}_ref419.log" 2>&1; T1=$(date +%s)
    env NQ_LOGIC=mrv "/tmp/${REV}_cpu_cur" 25 "$in" "/tmp/${REV}_y2_mrv_${sl}.bin" > "$LOGDIR/2_cpu_N25${sl}_mrv.log" 2>&1; T2=$(date +%s)
    y2_valid "$LOGDIR/2_cpu_N25${sl}_ref419.log" "/tmp/${REV}_y2_ref_${sl}.bin" "$NSL" "ref_$sl" && y2_valid "$LOGDIR/2_cpu_N25${sl}_mrv.log" "/tmp/${REV}_y2_mrv_${sl}.bin" "$NSL" "mrv_$sl" || summary_exit
    tot="$(grep -o 'total_sum=[0-9]*' "$LOGDIR/2_cpu_N25${sl}_mrv.log" | tail -1 | cut -d= -f2)"
    if grep -q 'logic=mrv' "$LOGDIR/2_cpu_N25${sl}_mrv.log" && cmp -s "/tmp/${REV}_y2_ref_${sl}.bin" "/tmp/${REV}_y2_mrv_${sl}.bin"; then pass "Y2$(echo "$sl" | tr CT ct)_cpu_N25_${sl}_mrv_equals_ref419 ($NSL records, total $tot; ref $((T1-T0)) s, mrv $((T2-T1)) s = $(ratio $((T2-T1)) $((T1-T0 > 0 ? T1-T0 : 1)))x of ref)"
    else fail "Y2_cpu_N25_${sl}" "mrv differs from the reference on slice $sl (or NQ_LOGIC was not honoured): stop"; summary_exit; fi
  done
  Y2T_OK=1
  head -c $((DFS_CHECK*28)) "$SLC25" > "/tmp/${REV}_c_head.bin"; head -c $((DFS_CHECK*8)) "/tmp/${REV}_y2_ref_C.bin" > "/tmp/${REV}_y2_ref_head.bin"
  env -u NQ_LOGIC "/tmp/${REV}_cpu_cur" 25 "/tmp/${REV}_c_head.bin" "/tmp/${REV}_y2_dfs_head.bin" > "$LOGDIR/2_cpu_N25C_dfs.log" 2>&1
  y2_valid "$LOGDIR/2_cpu_N25C_dfs.log" "/tmp/${REV}_y2_dfs_head.bin" "$DFS_CHECK" dfs_head || summary_exit
  grep -q 'logic=dfs' "$LOGDIR/2_cpu_N25C_dfs.log" && grep -q 'layout=425' "$LOGDIR/2_cpu_N25C_dfs.log" && cmp -s "/tmp/${REV}_y2_dfs_head.bin" "/tmp/${REV}_y2_ref_head.bin" && pass "Y2d_cpu_N25_dfs_default_equals_ref419 ($DFS_CHECK records; NQ_LOGIC unset = the 419 search)" || { fail "Y2d_cpu_N25_dfs" "the default path differs from the reference: stop"; summary_exit; }
fi
banner "Y3: refusals"
"/tmp/${REV}_cpu_cur" 26 "$SLC25" /tmp/${REV}_y3a.bin 8 > "$LOGDIR/2_y3_n26.log" 2>&1; rc=$?; [[ "$rc" == "3" ]] && pass "Y3_N26_refused_rc3" || fail "Y3_N26_refused_rc3" "rc=$rc"
NQ_LOGIC=bad "/tmp/${REV}_cpu_cur" 25 "$SLC25" /tmp/${REV}_y3b.bin 8 > "$LOGDIR/2_y3_logic_bad.log" 2>&1; rc=$?; [[ "$rc" == "3" ]] && pass "Y3_NQ_LOGIC_bad_refused_rc3" || fail "Y3_NQ_LOGIC_bad_refused_rc3" "rc=$rc"

# ---------------------------------------------------------------------
# 5. Cells
# ---------------------------------------------------------------------
printf 'cell\tbinary\tN\tlogic\tblock\tmax_blocks\trecords\thelpers\tfree_mb\tkernel_ms\tnorm_ms\ttotal_sum\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
CLKPID=""
gpu_gate() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; [[ -z "$apps" ]] && return 0; fail "gpu_empty_before[$1]" "$(echo "$apps" | tr '\n' ' ')"; return 1; }
gpu_after() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; [[ -z "$apps" ]] || fail "gpu_residue_after[$1]" "$(echo "$apps" | tr '\n' ' ')"; }
clk_start() { local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"; ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPID=$!; }
clk_stop() { [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 && $6+0>50 && $4+0>150 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.1f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_$1.tsv"); }
declare -A VAL NORM TOTAL
parse_log() { local cell="$1" b="$2" n="$3" lg="$4" start="$5" want_logic="$6" blk="$7" mb="$8" recs="$9"; local KMS TOT FREE HL LOGV XC CFG NORMV
  KMS="$(grep -o 'kernel_ms=[0-9.]*' "$lg" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  FREE="$(grep -o 'free_mb=[0-9]*' "$lg" | head -1 | cut -d= -f2)"; HL="$(grep -o 'helpers=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  LOGV="$(grep -o '\[gpu-logic\] N=[0-9]* requested=[a-z0-9]* logic=[a-z]*' "$lg" | head -1 | grep -o 'logic=[a-z]*' | cut -d= -f2)"; XC="$(grep -o 'extra_ctx=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  CFG="$(grep -o '\[gpu-config\] BLOCK=[0-9]* MAX_BLOCKS=[0-9]*' "$lg" | head -1)"
  NORMV="$(awk -v k="${KMS:-0}" -v s="${SMMEAN/\?/0}" 'BEGIN{ if(s>0) printf "%.1f", k*s/1710; else print "?" }')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$b" "$n" "${LOGV:-?}" "$blk" "$mb" "$recs" "${HL:-?}" "${FREE:-?}" "${KMS:-?}" "$NORMV" "${TOT:-?}" "$SMMEAN" "$SMMIN" "$TMAX" "$start" >> "$TSV"
  [[ "${LOGV:-}" == "$want_logic" ]] || { fail "logic[$cell]" "logic=${LOGV:-?}, expected $want_logic"; return 1; }
  [[ "${HL:-}" == "1" && "${XC:-}" == "0" ]] || { fail "ctx_state[$cell]" "helpers=${HL:-?} extra_ctx=${XC:-?}"; return 1; }
  [[ "$CFG" == "[gpu-config] BLOCK=$blk MAX_BLOCKS=$mb" ]] || { fail "config[$cell]" "'${CFG:-<none>}'"; return 1; }
  [[ -n "${KMS:-}" && -n "${TOT:-}" ]] || { fail "run[$cell]" "no kernel_ms/total_sum: $(tail -2 "$lg" | tr '\n' ' ')"; return 1; }
  VAL[$cell]="$KMS"; NORM[$cell]="$NORMV"; TOTAL[$cell]="$TOT"
  info "$cell" "$b N=$n logic=${LOGV:-?} MB=$mb records=$recs kernel_ms=$KMS norm_ms=$NORMV (sm_mean $SMMEAN, ${TMAX}C; $(grep -o 'local_bytes_per_thread=[0-9]*' "$lg" | head -1) $(grep -o 'registers=[0-9]*' "$lg" | head -1)) total_sum=$TOT"; }
# run_cell <cell> <binary> <N> <input> <block> <max_blocks> <logic dfs|mrv> [oracle]
run_cell() { local cell="$1" bin="$2" n="$3" in="$4" blk="$5" mb="$6" logic="$7" oracle="${8:-}"; gpu_gate "$cell" || return 1
  local lg="$LOGDIR/3_${cell}.log" start recs; start="$(date -Is)"; recs=$(( $(stat -c %s "$in") / 28 )); clk_start "$cell"
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX NQ_LAYOUT=auto NQ_LOGIC="$logic" NQ_BLOCK="$blk" NQ_MAX_BLOCKS="$mb" NQ_HELPER_CTX=1 NQ_HELPER_MB="$HMB" "./$bin" "$n" "$in" "$LOGDIR/bins/${cell}_results.bin" $oracle > "$lg" 2>&1 || true
  clk_stop "$cell"; gpu_after "$cell"; parse_log "$cell" "$bin" "$n" "$lg" "$start" "$logic" "$blk" "$mb" "$recs"; }
if [[ "$SKIP_CELLS" == "1" ]]; then info "cells" "skipped (SKIP_CELLS=1)"; TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]"; summary_exit; fi

banner "B1: $CU_BIN, NQ_LOGIC=dfs, N=21 -- the 419 search inside the 423 binary"
if [[ "$HAVE21" == "1" ]]; then
run_cell B1 "$CU_BIN" 21 "$IN21" 32 960 dfs "$ORACLE21" || summary_exit
[[ "${TOTAL[B1]:-}" == "$ORACLE21" ]] && pass "oracle[B1]" || { fail "oracle[B1]" "total_sum=${TOTAL[B1]:-?} != $ORACLE21"; PROD_OK=0; }
if [[ "$HAVEREF21" == "1" ]]; then cmp -s "$LOGDIR/bins/B1_results.bin" "$REF21" && pass "B1_per_thread_results_identical_to_411_B1" || { fail "B1_per_thread_results_DIFFER_from_411_B1" "the default path changed"; PROD_OK=0; }; else PROD_OK=0; fi
else info "B1" "skipped (no N=21 input)"; fi

banner "TM: mrv on slice T, one record per thread -- per-record equality with the CPU"
sleep "$COOLDOWN"; run_cell TM "$CU_BIN" 25 "$SLT" 32 960 mrv || summary_exit
if [[ "$Y2T_OK" == "1" ]]; then head -c $((TN*8)) "$LOGDIR/bins/TM_results.bin" > "/tmp/${REV}_TM_head.bin"
  cmp -s "/tmp/${REV}_TM_head.bin" "/tmp/${REV}_y2_mrv_T.bin" && pass "TM_gpu_per_record_equals_mrv_cpu ($TN records; GPU == CPU == ref419 on slice T)" || { fail "TM_gpu_vs_cpu" "GPU per-record results differ from the CPU harness on slice T -- stop"; summary_exit; }
else info "TM" "not compared (Y2 skipped); total_sum ${TOTAL[TM]}"; fi

banner "DM960: mrv on the N=25 slice B (307,200 records) -- the judged cell (419's D960: $D960_419_MS ms, about 52 min)"
sleep "$COOLDOWN"; run_cell DM960 "$CU_BIN" 25 "$SLB" 32 960 mrv || summary_exit
[[ "${TOTAL[DM960]}" == "$D960_419_TOTAL" ]] && pass "DM960_total_sum_equals_419s ($D960_419_TOTAL)" || { fail "DM960_total_sum" "${TOTAL[DM960]} != $D960_419_TOTAL -- the search is wrong on slice B: stop"; summary_exit; }
DMS="$(sha256sum "$LOGDIR/bins/DM960_results.bin" | cut -c1-8)"; D960_FILE="$(ls -t 419_validate_*/bins/D960_results.bin 2>/dev/null | head -1)"
if [[ "$DMS" == "$D960_419_BIN_SHA" ]]; then pass "DM960_per_thread_results_identical_to_419_D960 (results.bin sha $DMS, 30,720 thread sums over $SLICE_B records)"
elif [[ -n "$D960_FILE" ]] && cmp -s "$LOGDIR/bins/DM960_results.bin" "$D960_FILE"; then pass "DM960_per_thread_results_identical_to_419_D960 (cmp with $D960_FILE; its sha is $DMS, the recorded $D960_419_BIN_SHA is not a sha256 prefix)"
else fail "DM960_per_thread_results" "results.bin sha $DMS != $D960_419_BIN_SHA${D960_FILE:+ and cmp with $D960_FILE differs} (the total matched: compare the files)"; fi
R_RAW="$(ratio "${VAL[DM960]}" "$D960_419_MS")"; R_USE="$R_RAW"; R_KIND="raw"
if [[ -n "$D960_419_NORM" && "${NORM[DM960]}" != "?" ]]; then R_USE="$(ratio "${NORM[DM960]}" "$D960_419_NORM")"; R_KIND="normalised"; fi
info "DM960_vs_419_D960" "ratio $R_USE ($R_KIND; raw $R_RAW) = $(awk -v r="$R_USE" 'BEGIN{printf "%.2f", 1/r}')x faster"
if [[ "$RUN_DREF" == "1" ]]; then
  banner "DD960: the 419 search on slice B in this session (RUN_DREF=1, about 52 min)"
  sleep "$COOLDOWN"; run_cell DD960 "$CU_BIN" 25 "$SLB" 32 960 dfs || summary_exit
  cmp -s "$LOGDIR/bins/DD960_results.bin" "$LOGDIR/bins/DM960_results.bin" && pass "DD960_per_thread_results_identical_to_DM960" || fail "DD960_vs_DM960" "per-thread results differ"
  R_USE="$(ratio "${NORM[DM960]}" "${NORM[DD960]}")"; R_KIND="normalised, same session"
  info "DM960_vs_DD960" "ratio $R_USE ($R_KIND; raw $(ratio "${VAL[DM960]}" "${VAL[DD960]}")); DD960 vs the recorded D960: raw $(ratio "${VAL[DD960]}" "$D960_419_MS")"
fi
if le 0.90 "$R_USE"; then fail "P1_REFUTED_mrv_not_faster" "DM960 / D960 = $R_USE >= 0.90 -- the work model does not carry to the GPU; recorded and closed (read S7's loop and the lane utilisation before any second attempt)"
elif le 0.40 "$R_USE" && le "$R_USE" 0.65; then pass "P1_ratio_in_0.40_0.65 ($R_USE, $R_KIND)"
else info "P1_outside_statement" "DM960 / D960 = $R_USE ($R_KIND): outside 0.40..0.65, below the refutation line 0.90"; fi

if [[ "$SKIP_SWEEP" != "1" ]] && ! le 0.90 "$R_USE"; then
  for mb in $SWEEP_MB; do
    banner "M$mb: mrv on slice B at MAX_BLOCKS=$mb ($(( mb / 80 )) one-warp blocks per SM x 32 x ${FRAME_MRV} B)"
    sleep "$COOLDOWN"; run_cell "M$mb" "$CU_BIN" 25 "$SLB" 32 "$mb" mrv || summary_exit
    [[ "${TOTAL[M$mb]}" == "$D960_419_TOTAL" ]] && pass "M${mb}_total_sum_equals_419s" || fail "M${mb}_total_sum" "${TOTAL[M$mb]} != $D960_419_TOTAL"
    info "M${mb}_vs_DM960" "normalised $(pct "${NORM[M$mb]}" "${NORM[DM960]}")% (raw $(pct "${VAL[M$mb]}" "${VAL[DM960]}")%); against 419's D960: raw $(ratio "${VAL[M$mb]}" "$D960_419_MS")  (the frame is ${FRAME_MRV} B: 960 was the optimum for 192 B)"
  done
else info "sweep" "skipped"; fi
CLK_OK=1; for c in "${!VAL[@]}"; do m="$(awk -F'\t' -v c="$c" '$1==c{print $13}' "$TSV" | tail -1)"; [[ "$m" == "?" || -z "$m" ]] && continue; awk -v x="$m" 'BEGIN{d=x-1710; if(d<0)d=-d; exit !(d<=34)}' || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "W3_sm_clock_1710_all_cells" || info "W3_sm_clock" "power-capped cells read below 1676; compare norm_ms"
[[ "$PROD_OK" == "1" ]] && info "PROD" "the 423 binary's default path is 419's (SASS, N=21 per-thread results): it may stand in for 419 on the -g path" || info "PROD" "the 423 binary's default path is NOT proven identical to 419 -- keep 419 on the -g path (see the S1-S6 / B1 FAIL lines)"
if [[ "$FAIL" -eq 0 ]] && le "$R_USE" 0.50; then info "ADOPTION" "mrv is >= 2x on slice B with every HARD gate passed ($R_USE): 424 puts NQ_LOGIC=mrv on the -g path (N=21..25, oracle-gated)"
elif [[ "$FAIL" -eq 0 ]]; then info "ADOPTION" "NOT adopted by the agreed rule: ratio $R_USE > 0.50 (correct, but under 2x) -- Suzuki decides whether the sweep cell or a second revision is worth it"
else info "ADOPTION" "NOT adopted -- see FAIL lines"; fi
banner "Results"; python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import sys
rows=[l.rstrip('\n').split('\t') for l in open(sys.argv[1])]
w=[max(len(r[i]) for r in rows if i<len(r)) for i in range(len(rows[0]))]
for r in rows: print('  '.join(c.ljust(w[i]) for i,c in enumerate(r)))
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -f "/tmp/${REV}_y2_mrv_T.bin" "$LOGDIR/bins/Y2t_mrv_cpu_T.bin" 2>/dev/null; cp -f "/tmp/${REV}_y2_ref_T.bin" "$LOGDIR/bins/Y2t_ref419_T.bin" 2>/dev/null; cp -f "/tmp/${REV}_y2_mrv_C.bin" "$LOGDIR/bins/Y2c_mrv_cpu_C.bin" 2>/dev/null
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd1[46]" --exclude="*slice[BC]*.bin" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
