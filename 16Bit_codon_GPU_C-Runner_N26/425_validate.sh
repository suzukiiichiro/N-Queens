#!/usr/bin/env bash
# 425_validate.sh -- a shared queue for the refill search (NQ_LOGIC=queue): a thread that needs a record takes the next
# unclaimed record of the whole grid, so no thread is left with a longer list than the others. One A10G (NQ_CRunner).
#
# WHERE 424 ENDED (2026-11-20, OK=63 FAIL=0): lane refill with FIXED lists (tid, tid + stride, ...) is correct and nvcc kept
#   one loop, but R21 / Z21 = 1.059, R22 / Z22 = 1.040, DR960 / D960 = 0.913 -> not adopted. Power traces: R22 finishes the bulk
#   8 % sooner than Z22 (plateau 730 s against 795 s) and then runs a 185 s tail at 70-130 W; DR960 is 32 % tail (905 s).
#   The heaviest threads run on alone. The as-built kernel has a short tail because its warps advance round by round.
#
# rev425 -- .cu: 424 + process_queue_refill / kernel_queue_gpu (the 424 search; "idx += stride" became an atomicAdd on a
#   grid-wide counter; one result per RECORD, folded by the host into the usual per-thread sums at the launch stride).
#   NQ_LOGIC unset = 419 exactly. Code region vs 419: removed=3 added=1111. .py: 3 executable lines vs 419Py.
#
#   S0 static + every input located (cwd first, then find under SEARCH_ROOT)                                                HARD
#   S1 ptxas: five instantiations as 419; mrv <= 128 B; refill, queue <= 200 B; spill 0                                     HARD
#   S2-S6 five instantiations: SASS == the 419 binary's                                                                     HARD(prod)
#   S7 mrv / refill SASS == 423's / 424's (info)        S8 queue SASS recorded (loops holding a 64-bit local load)          info
#   Y2 CPU (8 threads draining ONE shared counter; per-record outputs compared by sha256 with the 419_ref_dfs outputs):
#      Y2c N=25 slice C (1,024) ; Y2t N=25 slice T (1,024) ; Y2d first 64 of C with NQ_LOGIC unset                           HARD
#   Y3 rc=3: N=26 with NQ_LOGIC=queue ; NQ_LOGIC=bad                                                                        HARD
#   Cells (BLOCK 32, MAX_BLOCKS 960, helper 1 + 128 MB). results.bin of a queue cell = per-record results folded per thread:
#     B1    dfs   N=21 auto(403)    results.bin == 411 B1, oracle                                                           HARD(prod)
#     TQ    queue N=25 slice T      per-RECORD results (NQ_REC_OUT) == the reference, sha fe262a34                          HARD
#     Q21   queue N=21              oracle ; results.bin sha 1bfab090 (= 411 B1 = 424's Z21, per thread)                    HARD
#     Q22   queue N=22              oracle ; results.bin sha 9a6ca49e (= 424's Z22, per thread)                             HARD
#     DQ960 queue N=25 slice B      total 6,910,807,391,840 ; results.bin sha c881e856 (= 419's D960, per thread)           HARD
#   ANCHORS (not re-run: 424 measured them on 2026-11-20 in this binary's layout; RUN_Z=1 re-runs Z21 and Z22, +17 min):
#     Z21 dfs at layout 425: 100,675.4 ms     Z22: 884,175.9 ms     419's D960: 3,099,327 ms
#     production (layout 403): N=21 97,946 ms, N=22 849,096 ms -- printed next to each ratio.
#   PRE-REGISTERED (from 424's power traces: plateau time + the tail's work at plateau rate; ratios are queue / anchor):
#     P1  Q21 / Z21     0.80 .. 0.92   (centre 0.86)
#     P2  Q22 / Z22     0.80 .. 0.92   (centre 0.86)
#     P3  DQ960 / D960  0.65 .. 0.80   (centre 0.70; a slice-B record runs for minutes, so the end of the queue still idles lanes)
#     Refuted: P2 >= 0.97.
#   ADOPTION: every HARD, Q21 <= 0.95 x Z21 and Q22 <= 0.95 x Z22 -> 426 puts NQ_LOGIC=queue on the -g path for N=25 and
#     decides N=21..24 against the layout-403/423/424 production values (the queue kernel is layout 425 only).
#
# USAGE (NQ_CRunner, GPU idle; about 1 h 15: static+build 3 min, Y2 ~15 min, cells B1 2 + TQ 1 + Q21 2 + Q22 ~13 + DQ960 ~37 min)
#   STATIC_ONLY=1 bash 425_validate.sh
#   bash 425_validate.sh 2>&1 | tee 425_validate_stdout.log          (inside screen; detach with Ctrl+A then D)
#   SKIP_CELLS=1   SKIP_BUILD=1   SKIP_Y2=1   SKIP_N22=1   SKIP_DQ960=1   RUN_Z=1   SEARCH_ROOT=<dir>   IN21= IN22= REF21= ...

set -u
REV="425"
SKIP_BUILD="${SKIP_BUILD:-0}"; STATIC_ONLY="${STATIC_ONLY:-0}"; SKIP_CELLS="${SKIP_CELLS:-0}"; SKIP_Y2="${SKIP_Y2:-0}"; SKIP_N22="${SKIP_N22:-0}"; SKIP_DQ960="${SKIP_DQ960:-0}"; RUN_Z="${RUN_Z:-0}"; RUN_CODON="${RUN_CODON:-0}"
SEARCH_ROOT="${SEARCH_ROOT:-$HOME}"
PY_SRC="${PY_SRC:-425Py_kernel_maxd16_final.py}"; PY_BIN="${PY_BIN:-${PY_SRC%.py}}"
PREV_PY="${PREV_PY:-419Py_kernel_maxd16_final.py}"
CU_SRC="${CU_SRC:-425_kernel_maxd16.cu}"; CU_BIN="${CU_BIN:-425_kernel_maxd16}"
PREV_CU="${PREV_CU:-419_kernel_maxd16.cu}"; PREV_BIN="${PREV_BIN:-419_kernel_maxd16}"; MRV_BIN="${MRV_BIN:-423_kernel_maxd16}"
REF21="${REF21:-411_validate_20261006_150938/bins/B1_results.bin}"; REF21_SHA="${REF21_SHA:-1bfab090}"
SLC25="${SLC25:-419_n25_sliceC_1024.bin}"; SLT="${SLT:-419_n25_sliceT_1024.bin}"; SLB="${SLB:-419_n25_sliceB_307200.bin}"
SLC25_SHA="${SLC25_SHA:-253e101a17f5e3a2}"; SLT_SHA="${SLT_SHA:-5671650fa938a7cf}"; SLB_SHA="${SLB_SHA:-76dd265fac94c7c5}"; NSL="${NSL:-1024}"; SLICE_B="${SLICE_B:-307200}"
REFC_SHA="${REFC_SHA:-f71f9d88ca32a900}"; REFT_SHA="${REFT_SHA:-fe262a347b7a5b2b}"; REFT_S32_SHA="${REFT_S32_SHA:-68a73bed207cb8ca}"; REFC_TOTAL="${REFC_TOTAL:-23378440174}"; REFT_TOTAL="${REFT_TOTAL:-1498221984}"; S32="${S32:-32}"
KERNEL_SHA_419="3b67e26c23bf3454"
D960_419_MS="${D960_419_MS:-3099327}"; D960_419_NORM="${D960_419_NORM:-}"; D960_419_TOTAL="${D960_419_TOTAL:-6910807391840}"; D960_419_BIN_SHA="${D960_419_BIN_SHA:-c881e856}"
P1LO=0.80; P1HI=0.92; P2LO=0.80; P2HI=0.92; P3LO=0.65; P3HI=0.80
Z21_NORM="${Z21_NORM:-100675.4}"; Z22_NORM="${Z22_NORM:-884175.9}"; PROD21=97946; PROD22=849096
Q21_BIN_SHA="${Q21_BIN_SHA:-1bfab0900918dd44}"; Q22_BIN_SHA="${Q22_BIN_SHA:-9a6ca49e165bde40}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"; CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"; CODON="${CODON:-codon}"; GCC="${GCC:-gcc}"; ARCH="${ARCH:-sm_86}"
HMB="${HMB:-128}"; DFS_CHECK="${DFS_CHECK:-64}"
COOLDOWN="${COOLDOWN:-15}"; CLK_INTERVAL="${CLK_INTERVAL:-5}"; ITER_CAP="${ITER_CAP:-4000000000ULL}"; OMP_THREADS="${OMP_THREADS:-8}"
ORACLE21="${ORACLE21:-314666222712}"; ORACLE22="${ORACLE22:-2691008701644}"
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
# 1. Static (S0) -- and every input this run will need, located NOW
# ---------------------------------------------------------------------
# locate <variable> <required 0|1> <find -name / -path pattern> : keeps the value if it exists, else searches SEARCH_ROOT
locate() { local var="$1" req="$2" pat="$3" cur="${!1}" hit=""
  if [[ -e "$cur" ]]; then pass "located[$var] $cur"; return 0; fi
  if [[ "$pat" == */* ]]; then hit="$(find "$SEARCH_ROOT" -path "*$pat" 2>/dev/null | head -1)"; else hit="$(find "$SEARCH_ROOT" -name "$pat" -size +0 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)"; fi
  if [[ -n "$hit" ]]; then printf -v "$var" '%s' "$hit"; pass "located[$var] $hit (not in this directory; found under $SEARCH_ROOT)"; return 0; fi
  if [[ "$req" == "1" ]]; then fail "located[$var]" "'$cur' not here and nothing matching '$pat' under $SEARCH_ROOT (set $var=<path>)"; else info "not_found[$var]" "'$cur' / '$pat' -- the checks that need it are skipped (set $var=<path>)"; printf -v "$var" '%s' ""; fi; return 1; }
for f in "$PY_SRC" "$CU_SRC"; do [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"; done
locate PREV_CU 1 "419_kernel_maxd16.cu"
locate SLC25 1 "419_n25_sliceC_1024.bin"; locate SLT 1 "419_n25_sliceT_1024.bin"; locate SLB 1 "419_n25_sliceB_307200.bin"
IN21="${IN21:-}"; [[ -z "$IN21" ]] && IN21="constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin.sched394f.bin"; locate IN21 1 "constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin.sched394f.bin"
IN22="${IN22:-}"; [[ -z "$IN22" ]] && IN22="constellations_N22_7.sched394f.bin"; locate IN22 0 "constellations_N22_*sched394f.bin"
locate REF21 0 "411_validate_20261006_150938/bins/B1_results.bin"
locate PREV_PY 0 "419Py_kernel_maxd16_final.py"
if [[ "$STATIC_ONLY" != "1" || -e "$PREV_BIN" ]]; then locate PREV_BIN 0 "419_kernel_maxd16"; fi
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }
py_code_region() { python3 -c "
import sys
lines=open(sys.argv[1],encoding='utf-8').read().split('\n')
i=next(i for i,l in enumerate(lines) if l.startswith('import '))
sys.stdout.write('\n'.join(l for l in lines[i:] if not l.lstrip().startswith('#')))
" "$1"; }
py_code_region "$PY_SRC" > "/tmp/${REV}_code_only.py"; CODE="/tmp/${REV}_code_only.py"
NOTE_LINES=$(awk '/^# =+$/{f=1} f&&/^#/{n++} END{print n+0}' "$PY_SRC")
{ grep -q "^# 425 " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 425 ...' note block (rule 8)"
grep -qE '^REV_TAG:str="425"' "$CODE" && pass "source_rev_tag_is_425" || fail "source_rev_tag_is_425" "wrong REV_TAG"
grep -qF 'CRunnerEntry(17,"./425_kernel_maxd16","NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_entry_maxd17_425_binary_helper_prefix" || fail "source_table_entry" "table entry wrong"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=960\b" "$CODE" && pass "source_default_max_blocks_is_960" || fail "source_default_max_blocks_is_960" "default not 960"
! grep -F 'CRunnerEntry(' "$CODE" | grep -q 'NQ_LOGIC' && pass "py_does_not_set_NQ_LOGIC (the -g path runs the 419 search)" || fail "py_sets_NQ_LOGIC" "the .py must not select a trial search in this revision"
if [[ -n "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_419Py (removed=3 added=3 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_419Py" "removed=$PR_ added=$PA_, expected 3/3"
fi
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
extract_kernel() { awk '/^static uint64_t process_one_task\(/{f=1} f{print} /^    results\[tid\] = thread_total;/{g=1} g&&/^}/{exit}' "$1"; }
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"; cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA_419"* ]] && pass "cu_kernel_region_sha_is_419s (${KB:0:16}: process_one_task .. template kernel untouched)" || fail "cu_kernel_region_sha" "sha ${KB:0:16} != $KERNEL_SHA_419 -- the 419 search was edited"
KP="$(extract_kernel "/tmp/${REV}_prev_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KP" == "$KERNEL_SHA_419"* ]] && pass "prev_cu_is_419 (${KP:0:16})" || fail "prev_cu_is_419" "sha ${KP:0:16} != $KERNEL_SHA_419"
CR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true); CA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
[[ "$CR_" == "3" && "$CA_" == "1111" ]] && pass "cu_code_region_fingerprint_vs_419 (removed=3 added=1111)" || fail "cu_code_region_fingerprint" "removed=$CR_ added=$CA_, expected 3/1111"
grep -q '^static uint64_t process_stream_refill(' "$CU_SRC" && grep -q '^__global__ void kernel_refill_gpu(' "$CU_SRC" && grep -q 'strcmp(req, "refill")' "$CU_SRC" && pass "cu_refill_search_kept (424)" || fail "cu_refill_search_kept" "missing"
grep -q '^static uint64_t process_queue_refill(' "$CU_SRC" && grep -q '^__global__ void kernel_queue_gpu(' "$CU_SRC" && grep -q 'strcmp(req, "queue")' "$CU_SRC" && grep -q '#define QUEUE_CLAIM(p) atomicAdd' "$CU_SRC" && pass "cu_queue_search_present (process_queue_refill / kernel_queue_gpu / NQ_LOGIC=queue / atomicAdd)" || fail "cu_queue_search_present" "missing"
grep -q '^static uint64_t process_one_task_mrv(' "$CU_SRC" && grep -q '^#define MRV_SWITCH  0' "$CU_SRC" && pass "cu_mrv_search_kept (423)" || fail "cu_mrv_search_kept" "missing"
grep -q "rev425 " "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev425 note"
patched() { sed "s/const uint64_t DEBUG_ITER_CAP = 50000000ULL;/const uint64_t DEBUG_ITER_CAP = ${ITER_CAP};/" "$1" > "/tmp/${REV}_$(basename "$1")"; grep -c "DEBUG_ITER_CAP = ${ITER_CAP}" "/tmp/${REV}_$(basename "$1")"; }
[[ "$(patched "$CU_SRC")" == "1" ]] && pass "cpu_iter_cap_patched_in_tmp_copy ($ITER_CAP; the dfs check Y2d needs it)" || fail "cpu_iter_cap_patch" "DEBUG_ITER_CAP line not found"
{ "$GCC" -O2 -mpopcnt -fopenmp -x c -o "/tmp/${REV}_cpu_cur"  "/tmp/${REV}_$(basename "$CU_SRC")" -lm 2>"/tmp/${REV}_gcc.log" || "$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur"  "/tmp/${REV}_$(basename "$CU_SRC")" -lm 2>"/tmp/${REV}_gcc.log"; } && pass "cu_cpu_harness_builds[425]" || fail "cu_cpu_harness_builds[425]" "$(head -3 /tmp/${REV}_gcc.log)"
for trip in "$SLC25:$NSL:$SLC25_SHA" "$SLT:$NSL:$SLT_SHA" "$SLB:$SLICE_B:$SLB_SHA"; do IFS=: read -r f c want <<< "$trip"
  s="$(sha256sum "$f" | cut -c1-16)"; [[ $(( $(stat -c %s "$f") / 28 )) -eq "$c" && "$s" == "$want" ]] && pass "slice_is_the_checked_file[$(basename "$f")] ($c records, sha $s)" || fail "slice_sha[$f]" "$(( $(stat -c %s "$f") / 28 )) records, sha $s; expected $c / $want"; done
if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Build, ptxas (S1), SASS (S2-S8)
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR/sass" "$LOGDIR/bins"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; df -h . 2>&1; free -g 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
[[ ! -x "$NVCC" ]] && NVCC="nvcc"; [[ ! -x "$CUOBJDUMP" ]] && CUOBJDUMP="cuobjdump"
banner "Building"
if [[ "$SKIP_BUILD" == "1" ]]; then
  [[ -x "$CU_BIN" ]] && pass "existing_binary_used[$CU_BIN] ($(stat -c %y "$CU_BIN" | cut -c1-19); SKIP_BUILD=1)" || { fail "cu_binary_missing" "$CU_BIN absent and SKIP_BUILD=1"; summary_exit; }
  info "S1_ptxas" "skipped (no fresh nvcc log)"
else
  rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc_${REV}.log" 2>&1
  [[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "$(grep -i error "$LOGDIR/05_nvcc_${REV}.log" | head -3)"; summary_exit; }
  ptx_stats() { FR="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"; SP="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes spill stores, [0-9]* bytes spill loads' | head -1)"; RG="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"; }
  for pair in 402:160 403:160 423:176 424:184 425:192; do IFS=: read -r L want <<< "$pair"; ptx_stats "kernel_dfs_iter_gpu_maxd16ILi${L}E"
    [[ "$FR" == "$want" && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "S1_ptxas_L${L}_frame${want}_spill0_regs${RG}" || { fail "S1_ptxas_L${L}" "frame=${FR}B regs=${RG} ${SP} (expected $want / 0 / <= 44)"; PROD_OK=0; }; done
  ptx_stats "kernel_mrv_gpu"; [[ -n "$FR" && "$FR" -le 128 && "$SP" == "0 bytes spill stores, 0 bytes spill loads" ]] && pass "S1_ptxas_mrv_frame${FR}_spill0_regs${RG} (423: 112 / 38)" || fail "S1_ptxas_mrv" "frame=${FR:-?}B regs=${RG:-?} ${SP:-}"
  ptx_stats "kernel_refill_gpu"; [[ -n "$FR" && "$FR" -le 200 && "$SP" == "0 bytes spill stores, 0 bytes spill loads" ]] && pass "S1_ptxas_refill_frame${FR}_spill0_regs${RG} (424: 192 / 47)" || fail "S1_ptxas_refill" "frame=${FR:-?}B regs=${RG:-?} ${SP:-}"
  ptx_stats "kernel_queue_gpu"
  [[ -n "$FR" && "$FR" -le 200 && "$SP" == "0 bytes spill stores, 0 bytes spill loads" ]] && pass "S1_ptxas_queue_frame${FR}_spill0_regs${RG} (clang route: 192 B / 45; refill is 192 B / 47)" || { fail "S1_ptxas_queue" "frame=${FR:-?}B regs=${RG:-?} ${SP:-no ptxas line} (expected <= 200 B, 0 spill)"; summary_exit; }
fi
if [[ -z "$PREV_BIN" || ! -x "$PREV_BIN" ]]; then PREV_BIN="419_kernel_maxd16"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$PREV_BIN" "$PREV_CU" -lcuda > "$LOGDIR/05b_nvcc_prev.log" 2>&1; [[ -x "$PREV_BIN" ]] && pass "prev_binary_built[$PREV_BIN]" || { fail "prev_binary" "cannot build $PREV_BIN"; summary_exit; }; fi
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
# sass_loops <dump> <function substring>: every backward loop that holds a 64-bit local load, as "len@start" (innermost first)
sass_loops() { python3 - "$1" "$2" <<'PYEOF'
import sys,re
out={};cur=None
for line in open(sys.argv[1],errors='replace'):
    m=re.match(r'\s*Function\s*:\s*(\S+)',line)
    if m: cur=m.group(1); out[cur]=[]; continue
    if cur is None: continue
    m=re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s*(.*?);\s*(/\*.*\*/)?\s*$',line)
    if m: out[cur].append((int(m.group(1),16),m.group(2).strip()))
k=[k for k in out if sys.argv[2] in k]
if len(k)!=1: print("CANNOT_LOCATE"); sys.exit(0)
I=out[k[0]]; idx={a:i for i,(a,_) in enumerate(I)}; L=[]
for i,(a,s) in enumerate(I):
    m=re.search(r'BRA\s+(?:P\d,\s*)?(0x[0-9a-f]+)$',s)
    if not m: continue
    t=int(m.group(1),16)
    if t<a and t in idx:
        b=I[idx[t]:i+1]
        if any(x[1].startswith('LDL.64') for x in b): L.append((len(b),t))
L.sort(); print(' '.join(f'{n}@{t:#x}' for n,t in L) if L else 'none', '| BSSY', sum(1 for _,s in I if s.startswith('BSSY')))
PYEOF
}
"$CUOBJDUMP" -sass "./$CU_BIN" > "$LOGDIR/sass/${CU_BIN}.txt" 2>&1
"$CUOBJDUMP" -sass "$(dirname "$PREV_BIN")/$(basename "$PREV_BIN")" > "$LOGDIR/sass/419_kernel_maxd16.txt" 2>&1
for trip in "402:S2" "403:S3" "423:S4" "424:S5" "425:S6"; do IFS=: read -r L tag <<< "$trip"
  read -r N_SHA N_REAL N_LOOP N_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" "kernel_dfs_iter_gpu_maxd16ILi${L}E")"
  read -r P_SHA P_REAL P_LOOP P_TOT <<< "$(sass_stat "$LOGDIR/sass/419_kernel_maxd16.txt" "kernel_dfs_iter_gpu_maxd16ILi${L}E")"
  if [[ "$N_SHA" != "CANNOT_LOCATE" && "$N_SHA" == "$P_SHA" && "$N_REAL" == "$P_REAL" && "$N_LOOP" == "$P_LOOP" ]]; then pass "${tag}_sass_${L}_is_419s (sha $N_SHA, real $N_REAL, loop $N_LOOP)"
  else fail "${tag}_sass_${L}_differs_from_419" "425: $N_SHA / $N_REAL / $N_LOOP ; 419 binary: $P_SHA / $P_REAL / $P_LOOP -- the trial cells below are still valid, the -g path must stay on 419"; PROD_OK=0; fi
done
read -r M_SHA M_REAL M_LOOP M_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_mrv_gpu LDL)"
[[ "$M_SHA" == "500eb8a37ec40f49" ]] && pass "S7_sass_mrv_is_423s (sha $M_SHA, real $M_REAL)" || info "S7_sass_mrv" "sha $M_SHA real $M_REAL (423's: 500eb8a37ec40f49 / 464) -- mrv is not run here"
read -r R_SHA R_REAL R_LOOP R_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_refill_gpu)"
[[ "$R_SHA" == "1fa851e7544fbab9" ]] && pass "S7_sass_refill_is_424s (sha $R_SHA, real $R_REAL)" || info "S7_sass_refill" "sha $R_SHA real $R_REAL (424's: 1fa851e7544fbab9 / 363) -- refill is not run here"
read -r Q_SHA Q_REAL Q_LOOP Q_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_queue_gpu)"
[[ "$Q_SHA" != "CANNOT_LOCATE" ]] && pass "S8_sass_queue_recorded (sha $Q_SHA, real $Q_REAL, innermost loop with a 64-bit local load $Q_LOOP; refill (424): real 363, loop 324)" || { fail "S8_sass_queue_missing" "no kernel_queue_gpu in the binary"; summary_exit; }
info "S8_loops_with_stack_loads" "queue: $(sass_loops "$LOGDIR/sass/${CU_BIN}.txt" kernel_queue_gpu)   ;   refill (424): 324@0x1b0 | BSSY 10   ;   atomics in queue: $(awk '/Function : .*kernel_queue_gpu/{f=1;next} /Function :/{f=0} f' "$LOGDIR/sass/${CU_BIN}.txt" | grep -c 'ATOM')"
if [[ "$RUN_CODON" == "1" ]]; then rm -f "$PY_BIN"; "$CODON" build -release "$PY_SRC" > "$LOGDIR/06_codon_${REV}.log" 2>&1; [[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN]" || fail "codon_build" "$(tail -3 "$LOGDIR/06_codon_${REV}.log")"; else info "codon_build" "not run (-g stays on 419Py in this revision; RUN_CODON=1 builds $PY_SRC in a directory that has rev386_validation_helpers.py)"; fi
HAVEREF21=0; if [[ -n "$REF21" ]]; then RS="$(sha256sum "$REF21" | cut -c1-8)"; [[ "$RS" == "$REF21_SHA" ]] && { HAVEREF21=1; pass "ref21_is_411_B1 (sha $RS)"; } || fail "ref21_is_411_B1" "sha $RS != $REF21_SHA"; fi
if [[ -z "$D960_419_NORM" ]]; then
  for t in $(find "$SEARCH_ROOT" -path '*419_validate_*/419_results.tsv' 2>/dev/null | xargs -r ls -t 2>/dev/null); do v="$(awk -F'\t' '$1=="D960"{print $11}' "$t" | tail -1)"; k="$(awk -F'\t' '$1=="D960"{print $10}' "$t" | tail -1)"
    if [[ -n "$v" && "$v" != "?" ]]; then D960_419_NORM="$v"; info "D960_419_anchor" "from $t: kernel_ms=$k norm_ms=$v"; break; fi; done
fi
[[ -z "$D960_419_NORM" ]] && info "D960_419_anchor" "419_validate's tsv not found: P1 uses raw times (419 D960 kernel_ms $D960_419_MS)"

# ---------------------------------------------------------------------
# 4. CPU equivalence (Y2) and refusals (Y3)
# ---------------------------------------------------------------------
export OMP_NUM_THREADS="$OMP_THREADS"
y2_valid() { local lg="$1" out="$2" n="$3" nout="$4" lab="$5"
  grep -q "\[kernel-cputest-done\] N=[0-9]* records=${n} " "$lg" || { fail "Y2_invalid[$lab]" "no done line (abort: $(grep -o '\[[A-Z-]*\]' "$lg" | grep -v progress | tail -1))"; return 1; }
  [[ -f "$out" && $(stat -c %s "$out") -eq $((nout*8)) ]] || { fail "Y2_invalid[$lab]" "output size $(stat -c %s "$out" 2>/dev/null) != $((nout*8))"; return 1; }; return 0; }
if [[ "$SKIP_Y2" == "1" ]]; then info "Y2" "skipped (SKIP_Y2=1)"
else
  banner "Y2: CPU equivalence ($OMP_THREADS threads draining one shared counter): queue per record == the 419_ref_dfs outputs (sha256)"
  for trip in "c:C:$SLC25:$REFC_SHA:$REFC_TOTAL" "t:T:$SLT:$REFT_SHA:$REFT_TOTAL"; do IFS=: read -r lab sl in want wtot <<< "$trip"
    T0=$(date +%s); env NQ_LOGIC=queue "/tmp/${REV}_cpu_cur" 25 "$in" "/tmp/${REV}_y2_queue_${sl}.bin" > "$LOGDIR/2_cpu_N25${sl}_queue.log" 2>&1; T1=$(date +%s)
    y2_valid "$LOGDIR/2_cpu_N25${sl}_queue.log" "/tmp/${REV}_y2_queue_${sl}.bin" "$NSL" "$NSL" "queue_$sl" || summary_exit
    s="$(sha256sum "/tmp/${REV}_y2_queue_${sl}.bin" | cut -c1-16)"; tot="$(grep -o 'total_sum=[0-9]*' "$LOGDIR/2_cpu_N25${sl}_queue.log" | tail -1 | cut -d= -f2)"
    if grep -q 'logic=queue' "$LOGDIR/2_cpu_N25${sl}_queue.log" && [[ "$s" == "$want" && "$tot" == "$wtot" ]]; then pass "Y2${lab}_cpu_N25_${sl}_queue_per_record_equals_ref419 ($NSL records, total $tot, sha $s, $(grep -o 'claims=[0-9]*' "$LOGDIR/2_cpu_N25${sl}_queue.log" | tail -1); $((T1-T0)) s)"
    else fail "Y2${lab}_cpu_N25_${sl}" "sha $s total $tot, expected $want / $wtot: queue differs from the reference on slice $sl -- stop"; summary_exit; fi
  done
  head -c $((DFS_CHECK*28)) "$SLC25" > "/tmp/${REV}_c_head.bin"; head -c $((DFS_CHECK*8)) "/tmp/${REV}_y2_queue_C.bin" > "/tmp/${REV}_y2_ref_head.bin"
  env -u NQ_LOGIC "/tmp/${REV}_cpu_cur" 25 "/tmp/${REV}_c_head.bin" "/tmp/${REV}_y2_dfs_head.bin" > "$LOGDIR/2_cpu_N25C_dfs.log" 2>&1
  y2_valid "$LOGDIR/2_cpu_N25C_dfs.log" "/tmp/${REV}_y2_dfs_head.bin" "$DFS_CHECK" "$DFS_CHECK" dfs_head || summary_exit
  grep -q 'logic=dfs' "$LOGDIR/2_cpu_N25C_dfs.log" && cmp -s "/tmp/${REV}_y2_dfs_head.bin" "/tmp/${REV}_y2_ref_head.bin" && pass "Y2d_cpu_N25_dfs_default_equals_ref419 ($DFS_CHECK records; NQ_LOGIC unset = the 419 search)" || { fail "Y2d_cpu_N25_dfs" "the default path differs from the reference: stop"; summary_exit; }
fi
banner "Y3: refusals"
NQ_LOGIC=queue "/tmp/${REV}_cpu_cur" 26 "$SLC25" /tmp/${REV}_y3a.bin 8 > "$LOGDIR/2_y3_n26.log" 2>&1; rc=$?; [[ "$rc" == "3" ]] && pass "Y3_N26_queue_refused_rc3" || fail "Y3_N26_queue_refused_rc3" "rc=$rc"
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
# run_cell <cell> <binary> <N> <input> <block> <max_blocks> <logic dfs|mrv|refill> [oracle]   (CELL_LAYOUT=425 forces the layout)
run_cell() { local cell="$1" bin="$2" n="$3" in="$4" blk="$5" mb="$6" logic="$7" oracle="${8:-}"; gpu_gate "$cell" || return 1
  local lg="$LOGDIR/3_${cell}.log" start recs; start="$(date -Is)"; recs=$(( $(stat -c %s "$in") / 28 )); clk_start "$cell"
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX NQ_LAYOUT="${CELL_LAYOUT:-auto}" NQ_LOGIC="$logic" NQ_BLOCK="$blk" NQ_MAX_BLOCKS="$mb" NQ_HELPER_CTX=1 NQ_HELPER_MB="$HMB" "./$bin" "$n" "$in" "$LOGDIR/bins/${cell}_results.bin" $oracle > "$lg" 2>&1 || true
  clk_stop "$cell"; gpu_after "$cell"; parse_log "$cell" "$bin" "$n" "$lg" "$start" "$logic" "$blk" "$mb" "$recs"; }
if [[ "$SKIP_CELLS" == "1" ]]; then info "cells" "skipped (SKIP_CELLS=1)"; TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]"; summary_exit; fi
in_band() { le "$2" "$1" && le "$1" "$3"; }
nv() { if [[ "${NORM[$1]}" != "?" ]]; then echo "${NORM[$1]}"; else echo "${VAL[$1]}"; fi; }
power_line() { local f="$LOGDIR/clk_$1.tsv"; [[ -f "$f" ]] || return 0
  info "power[$1]" "$(python3 - "$f" <<'PYEOF'
import sys,statistics
p=[]
for l in open(sys.argv[1]).read().split('\n')[1:]:
    c=l.split(',')
    try:
        if len(c)>=6 and float(c[5])>50: p.append(float(c[3]))
    except ValueError: pass
n=len(p)
if n<4: print("too short")
else:
    med=statistics.median(p[:max(3,n//2)]); last=max(i for i,x in enumerate(p) if x>=0.9*med); t=p[last+1:]
    print(f"plateau {med:.0f} W for {(last+1)*5} s ({(last+1)/n*100:.0f} % of the run), tail {len(t)*5} s at {(sum(t)/len(t) if t else 0):.0f} W")
PYEOF
) (424: R22 plateau 80 %, tail 185 s; Z22 91 %, 80 s)"; }

banner "B1: $CU_BIN, NQ_LOGIC=dfs, N=21, layout auto -- the production path inside the 425 binary"
run_cell B1 "$CU_BIN" 21 "$IN21" 32 960 dfs "$ORACLE21" || summary_exit
[[ "${TOTAL[B1]:-}" == "$ORACLE21" ]] && pass "oracle[B1]" || { fail "oracle[B1]" "total_sum=${TOTAL[B1]:-?} != $ORACLE21"; PROD_OK=0; }
if [[ "$HAVEREF21" == "1" ]]; then cmp -s "$LOGDIR/bins/B1_results.bin" "$REF21" && pass "B1_per_thread_results_identical_to_411_B1" || { fail "B1_per_thread_results_DIFFER_from_411_B1" "the default path changed"; PROD_OK=0; }; else PROD_OK=0; fi
power_line B1

banner "TQ: queue on slice T -- per-RECORD results from the GPU against the reference"
sleep "$COOLDOWN"; NQ_REC_OUT="$LOGDIR/bins/TQ_per_record.bin" run_cell TQ "$CU_BIN" 25 "$SLT" 32 960 queue || summary_exit
s="$(sha256sum "$LOGDIR/bins/TQ_per_record.bin" 2>/dev/null | cut -c1-16)"
[[ "$s" == "$REFT_SHA" && "${TOTAL[TQ]}" == "$REFT_TOTAL" ]] && pass "TQ_gpu_per_record_equals_ref419 ($NSL records, total ${TOTAL[TQ]}, sha $s; $(grep -o 'claims=[0-9]*' "$LOGDIR/3_TQ.log" | head -1))" || { fail "TQ_gpu_per_record" "sha ${s:-<no file>} total ${TOTAL[TQ]}, expected $REFT_SHA / $REFT_TOTAL -- the queue kernel is wrong on the GPU: stop"; summary_exit; }

if [[ "$RUN_Z" == "1" ]]; then
  banner "Z21 (RUN_Z=1): dfs at layout 425, this session's anchor"
  sleep "$COOLDOWN"; CELL_LAYOUT=425 run_cell Z21 "$CU_BIN" 21 "$IN21" 32 960 dfs "$ORACLE21" || summary_exit; Z21_NORM="$(nv Z21)"; power_line Z21
fi
banner "Q21: queue, N=21 (anchor Z21 = $Z21_NORM ms at layout 425; production at layout 403 = $PROD21 ms)"
sleep "$COOLDOWN"; run_cell Q21 "$CU_BIN" 21 "$IN21" 32 960 queue "$ORACLE21" || summary_exit
[[ "${TOTAL[Q21]:-}" == "$ORACLE21" ]] && pass "oracle[Q21]" || { fail "oracle[Q21]" "total_sum=${TOTAL[Q21]:-?} != $ORACLE21 -- stop"; summary_exit; }
s="$(sha256sum "$LOGDIR/bins/Q21_results.bin" | cut -c1-16)"; [[ "$s" == "$Q21_BIN_SHA" ]] && pass "Q21_folded_per_thread_results_identical_to_411_B1 (sha $s, 30,720 thread sums)" || { fail "Q21_per_thread_results" "sha $s != $Q21_BIN_SHA (the total matched): stop"; summary_exit; }
R21="$(ratio "$(nv Q21)" "$Z21_NORM")"; info "Q21_vs_Z21" "ratio $R21 = $(awk -v r="$R21" 'BEGIN{printf "%.2f", 1/r}')x ; against production N=21 ($PROD21 ms, layout 403): $(ratio "$(nv Q21)" "$PROD21")"
if in_band "$R21" "$P1LO" "$P1HI"; then pass "P1_Q21_over_Z21_in_${P1LO}_${P1HI} ($R21)"; else info "P1_outside_statement" "Q21 / Z21 = $R21, stated $P1LO .. $P1HI"; fi
power_line Q21

R22="?"
if [[ "$SKIP_N22" != "1" && -n "$IN22" ]]; then
  if [[ "$RUN_Z" == "1" ]]; then sleep "$COOLDOWN"; CELL_LAYOUT=425 run_cell Z22 "$CU_BIN" 22 "$IN22" 32 960 dfs "$ORACLE22" || summary_exit; Z22_NORM="$(nv Z22)"; power_line Z22; fi
  banner "Q22: queue, N=22 (anchor Z22 = $Z22_NORM ms at layout 425; production at layout 403 = $PROD22 ms)"
  sleep "$COOLDOWN"; run_cell Q22 "$CU_BIN" 22 "$IN22" 32 960 queue "$ORACLE22" || summary_exit
  [[ "${TOTAL[Q22]:-}" == "$ORACLE22" ]] && pass "oracle[Q22]" || { fail "oracle[Q22]" "total_sum=${TOTAL[Q22]:-?} != $ORACLE22 -- stop"; summary_exit; }
  s="$(sha256sum "$LOGDIR/bins/Q22_results.bin" | cut -c1-16)"; [[ "$s" == "$Q22_BIN_SHA" ]] && pass "Q22_folded_per_thread_results_identical_to_424_Z22 (sha $s, 30,720 thread sums)" || { fail "Q22_per_thread_results" "sha $s != $Q22_BIN_SHA (the total matched): stop"; summary_exit; }
  R22="$(ratio "$(nv Q22)" "$Z22_NORM")"; info "Q22_vs_Z22" "ratio $R22 = $(awk -v r="$R22" 'BEGIN{printf "%.2f", 1/r}')x ; against production N=22 ($PROD22 ms, layout 403): $(ratio "$(nv Q22)" "$PROD22")"
  if le 0.97 "$R22"; then fail "P2_REFUTED_queue_not_faster_at_N22" "Q22 / Z22 = $R22 >= 0.97 -- read the power line below (is there still a tail?) and S8"
  elif in_band "$R22" "$P2LO" "$P2HI"; then pass "P2_Q22_over_Z22_in_${P2LO}_${P2HI} ($R22)"; else info "P2_outside_statement" "Q22 / Z22 = $R22, stated $P2LO .. $P2HI"; fi
  power_line Q22
else info "N22" "skipped (SKIP_N22=1 or no N=22 input found)"; fi

R_USE="?"
if [[ "$SKIP_DQ960" != "1" ]]; then
  banner "DQ960: queue on the N=25 slice B (307,200 records) -- 419's D960: $D960_419_MS ms ; 424's DR960: 2,838,746 ms"
  sleep "$COOLDOWN"; run_cell DQ960 "$CU_BIN" 25 "$SLB" 32 960 queue || summary_exit
  [[ "${TOTAL[DQ960]}" == "$D960_419_TOTAL" ]] && pass "DQ960_total_sum_equals_419s ($D960_419_TOTAL)" || { fail "DQ960_total_sum" "${TOTAL[DQ960]} != $D960_419_TOTAL -- the queue search is wrong on slice B: stop"; summary_exit; }
  DMS="$(sha256sum "$LOGDIR/bins/DQ960_results.bin" | cut -c1-8)"
  [[ "$DMS" == "$D960_419_BIN_SHA" ]] && pass "DQ960_folded_per_thread_results_identical_to_419_D960 (results.bin sha $DMS, 30,720 thread sums over 307,200 records)" || fail "DQ960_per_thread_results" "results.bin sha $DMS != $D960_419_BIN_SHA (the total matched)"
  R_RAW="$(ratio "${VAL[DQ960]}" "$D960_419_MS")"; R_USE="$R_RAW"; R_KIND="raw"
  if [[ -n "$D960_419_NORM" && "${NORM[DQ960]}" != "?" ]]; then R_USE="$(ratio "${NORM[DQ960]}" "$D960_419_NORM")"; R_KIND="normalised"; fi
  info "DQ960_vs_419_D960" "ratio $R_USE ($R_KIND; raw $R_RAW) = $(awk -v r="$R_USE" 'BEGIN{printf "%.2f", 1/r}')x faster"
  if in_band "$R_USE" "$P3LO" "$P3HI"; then pass "P3_ratio_in_${P3LO}_${P3HI} ($R_USE, $R_KIND)"; else info "P3_outside_statement" "DQ960 / D960 = $R_USE ($R_KIND), stated $P3LO .. $P3HI"; fi
  power_line DQ960
fi
CLK_OK=1; for c in "${!VAL[@]}"; do m="$(awk -F'\t' -v c="$c" '$1==c{print $13}' "$TSV" | tail -1)"; [[ "$m" == "?" || -z "$m" ]] && continue; awk -v x="$m" 'BEGIN{d=x-1710; if(d<0)d=-d; exit !(d<=34)}' || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "W3_sm_clock_1710_all_cells" || info "W3_sm_clock" "power-capped cells read below 1676; ratios above use norm_ms"
[[ "$PROD_OK" == "1" ]] && info "PROD" "the 425 binary's default path is 419's (SASS, N=21 per-thread results)" || info "PROD" "the 425 binary's default path is NOT proven identical to 419 -- keep 419 on the -g path (see the S1-S6 / B1 lines)"
if [[ "$FAIL" -eq 0 && "$R22" != "?" ]] && le "$R21" 0.95 && le "$R22" 0.95; then info "ADOPTION" "queue: every HARD gate passed, N=21 $R21, N=22 $R22, N=25 slice B $R_USE -> 426 puts NQ_LOGIC=queue on the -g path (N=25 first; N=21..24 against their layout-403/423/424 production values)"
elif [[ "$FAIL" -eq 0 ]]; then info "ADOPTION" "NOT adopted by the stated rule (N=21 $R21, N=22 $R22, N=25 slice B $R_USE) -- correct, the gain is short of the rule; Suzuki decides"
else info "ADOPTION" "NOT adopted -- see FAIL lines"; fi
banner "Results"; python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import sys
rows=[l.rstrip('\n').split('\t') for l in open(sys.argv[1])]
w=[max(len(r[i]) for r in rows if i<len(r)) for i in range(len(rows[0]))]
for r in rows: print('  '.join(c.ljust(w[i]) for i,c in enumerate(r)))
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -f "/tmp/${REV}_y2_queue_T.bin" "$LOGDIR/bins/Y2t_queue_cpu_T.bin" 2>/dev/null; cp -f "/tmp/${REV}_y2_queue_C.bin" "$LOGDIR/bins/Y2c_queue_cpu_C.bin" 2>/dev/null
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd1[46]" --exclude="*slice[BC]*.bin" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
