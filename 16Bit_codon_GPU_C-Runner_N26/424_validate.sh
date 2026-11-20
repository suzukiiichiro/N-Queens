#!/usr/bin/env bash
# 424_validate.sh -- lane refill for the 419 search (NQ_LOGIC=refill): a thread that finishes a record takes its next record
# inside the same loop instead of waiting for the warp's longest record. One A10G (NQ_CRunner).
#
# WHERE 423 ENDED (2026-11-19): the mrv trial search is correct (slice B per-thread sha c881e856, slice T per record, rows scan
#   over all 98,138,478 records, B1 == 411 B1) but DM960 / D960 = 0.9407 -> not adopted; 419 stays production.
#   Diagnosis (423_warp_sim.c, reproduces ncu's 32.9 % lane utilisation as 34.8 % and 423's 0.941 as 0.951): the idle lanes of
#   419 come from (a) waiting at the end of each record for the warp's longest record and (b) different paths inside an iteration.
#
# rev424 -- .cu: 423 + process_stream_refill / kernel_refill_gpu (layout 425 statements, N <= 25). NQ_LOGIC unset = 419 exactly
#   (kernel region sha 3b67e26c; five instantiations' SASS == the 419 binary's). Code region: vs 419 removed=3 added=729.
#   .py: 3 executable lines vs 419Py. The -g path does not set NQ_LOGIC.
#
#   S0 static + every input located (cwd first, then find under SEARCH_ROOT) -- nothing later stops for a missing file    HARD
#   S1 ptxas: five instantiations as 419; mrv <= 128 B; refill <= 200 B; spill 0                                           HARD
#   S2-S6 five instantiations: SASS == the 419 binary's                                                                    HARD(prod)
#   S7 mrv SASS == the 423 binary's (when ./423_kernel_maxd16 exists)                                                      info
#   S8 refill SASS recorded: real instructions, every backward loop holding a 64-bit local load (419 <425> has two: the
#      111-instruction search loop and the record loop around it; refill should show ONE)                                   info
#   Y2 CPU (8 threads; outputs compared by sha256 with the 419_ref_dfs outputs Claude computed on 2026-11-19, which the
#      423 run reproduced on this machine: totals 23,378,440,174 / 1,498,221,984):
#      Y2c N=25 slice C: refill, one record per thread == reference (1,024 records)                                         HARD
#      Y2t N=25 slice T: refill, one record per thread == reference (1,024 records)                                         HARD
#      Y2s N=25 slice T at stride 32 (32 records per thread): per-thread sums == sums of the reference                      HARD
#      Y2d N=25 first 64 of C: NQ_LOGIC unset == reference                                                                  HARD
#   Y3 rc=3: N=26 with NQ_LOGIC=refill ; NQ_LOGIC=bad                                                                       HARD
#   Cells (BLOCK 32, MAX_BLOCKS 960, helper 1 + 128 MB):
#     B1    dfs    N=21 auto(403)   results.bin == 411 B1 (when found), oracle                                              HARD(prod)
#     Z21   dfs    N=21 layout 425  the anchor: same layout, same 192 B frame, as built (waits)                             oracle HARD
#     R21   refill N=21             results.bin == Z21 (per thread), oracle                                                 HARD
#     Z22 / R22   the same pair at N=22 (about 930 records per thread; 2 x ~14 min) when an N=22 input is found; SKIP_N22=1
#     DR960 refill N=25 slice B     total 6,910,807,391,840 ; results.bin sha c881e856 (= 419's D960)                       HARD
#   PRE-REGISTERED (from the warp model; a ratio is refill / as-built, normalised by the SM clock when available):
#     P1  DR960 / 419's D960 (3,099,327 ms)   in 0.70 .. 0.90     (10 records per thread: the tail of each lane's stream still idles)
#     P2  R21 / Z21                           in 0.62 .. 0.88     (66 records per thread)
#     P3  R22 / Z22                           in 0.60 .. 0.85     (about 930 records per thread: closest to a production run)
#     Refuted: P1 >= 0.97 (the waiting is not what the model says, or nvcc did not keep one loop -- read S8).
#   ADOPTION: every HARD, P1 <= 0.95 and R21 <= Z21 -> 425 moves the -g path to the refill kernel (N=25 first), oracle-gated.
#
# USAGE (NQ_CRunner, GPU idle; about 1.5-2 h: static+build 3 min, Y2 ~20 min, cells N=21 6 min, N=22 ~25 min, DR960 <= 52 min)
#   STATIC_ONLY=1 bash 424_validate.sh        # also prints where every input was found
#   bash 424_validate.sh 2>&1 | tee 424_validate_stdout.log          (inside screen; detach with Ctrl+A then D)
#   SKIP_CELLS=1   SKIP_BUILD=1   SKIP_Y2=1   SKIP_N22=1   SKIP_DR960=1   SEARCH_ROOT=<dir>   IN21= IN22= REF21= PREV_BIN= ...

set -u
REV="424"
SKIP_BUILD="${SKIP_BUILD:-0}"; STATIC_ONLY="${STATIC_ONLY:-0}"; SKIP_CELLS="${SKIP_CELLS:-0}"; SKIP_Y2="${SKIP_Y2:-0}"; SKIP_N22="${SKIP_N22:-0}"; SKIP_DR960="${SKIP_DR960:-0}"; RUN_CODON="${RUN_CODON:-0}"
SEARCH_ROOT="${SEARCH_ROOT:-$HOME}"
PY_SRC="${PY_SRC:-424Py_kernel_maxd16_final.py}"; PY_BIN="${PY_BIN:-${PY_SRC%.py}}"
PREV_PY="${PREV_PY:-419Py_kernel_maxd16_final.py}"
CU_SRC="${CU_SRC:-424_kernel_maxd16.cu}"; CU_BIN="${CU_BIN:-424_kernel_maxd16}"
PREV_CU="${PREV_CU:-419_kernel_maxd16.cu}"; PREV_BIN="${PREV_BIN:-419_kernel_maxd16}"; MRV_BIN="${MRV_BIN:-423_kernel_maxd16}"
REF21="${REF21:-411_validate_20261006_150938/bins/B1_results.bin}"; REF21_SHA="${REF21_SHA:-1bfab090}"
SLC25="${SLC25:-419_n25_sliceC_1024.bin}"; SLT="${SLT:-419_n25_sliceT_1024.bin}"; SLB="${SLB:-419_n25_sliceB_307200.bin}"
SLC25_SHA="${SLC25_SHA:-253e101a17f5e3a2}"; SLT_SHA="${SLT_SHA:-5671650fa938a7cf}"; SLB_SHA="${SLB_SHA:-76dd265fac94c7c5}"; NSL="${NSL:-1024}"; SLICE_B="${SLICE_B:-307200}"
REFC_SHA="${REFC_SHA:-f71f9d88ca32a900}"; REFT_SHA="${REFT_SHA:-fe262a347b7a5b2b}"; REFT_S32_SHA="${REFT_S32_SHA:-68a73bed207cb8ca}"; REFC_TOTAL="${REFC_TOTAL:-23378440174}"; REFT_TOTAL="${REFT_TOTAL:-1498221984}"; S32="${S32:-32}"
KERNEL_SHA_419="3b67e26c23bf3454"
D960_419_MS="${D960_419_MS:-3099327}"; D960_419_NORM="${D960_419_NORM:-}"; D960_419_TOTAL="${D960_419_TOTAL:-6910807391840}"; D960_419_BIN_SHA="${D960_419_BIN_SHA:-c881e856}"
P1LO=0.70; P1HI=0.90; P2LO=0.62; P2HI=0.88; P3LO=0.60; P3HI=0.85
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
{ grep -q "^# 424 " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 424 ...' note block (rule 8)"
grep -qE '^REV_TAG:str="424"' "$CODE" && pass "source_rev_tag_is_424" || fail "source_rev_tag_is_424" "wrong REV_TAG"
grep -qF 'CRunnerEntry(17,"./424_kernel_maxd16","NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_entry_maxd17_424_binary_helper_prefix" || fail "source_table_entry" "table entry wrong"
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
[[ "$CR_" == "3" && "$CA_" == "729" ]] && pass "cu_code_region_fingerprint_vs_419 (removed=3 added=729)" || fail "cu_code_region_fingerprint" "removed=$CR_ added=$CA_, expected 3/729"
grep -q '^static uint64_t process_stream_refill(' "$CU_SRC" && grep -q '^__global__ void kernel_refill_gpu(' "$CU_SRC" && grep -q 'strcmp(req, "refill")' "$CU_SRC" && pass "cu_refill_search_present (process_stream_refill / kernel_refill_gpu / NQ_LOGIC=refill)" || fail "cu_refill_search_present" "missing"
grep -q '^static uint64_t process_one_task_mrv(' "$CU_SRC" && grep -q '^#define MRV_SWITCH  0' "$CU_SRC" && pass "cu_mrv_search_kept (423)" || fail "cu_mrv_search_kept" "missing"
grep -q "rev424 " "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev424 note"
patched() { sed "s/const uint64_t DEBUG_ITER_CAP = 50000000ULL;/const uint64_t DEBUG_ITER_CAP = ${ITER_CAP};/" "$1" > "/tmp/${REV}_$(basename "$1")"; grep -c "DEBUG_ITER_CAP = ${ITER_CAP}" "/tmp/${REV}_$(basename "$1")"; }
[[ "$(patched "$CU_SRC")" == "1" ]] && pass "cpu_iter_cap_patched_in_tmp_copy ($ITER_CAP; the dfs check Y2d needs it)" || fail "cpu_iter_cap_patch" "DEBUG_ITER_CAP line not found"
{ "$GCC" -O2 -mpopcnt -fopenmp -x c -o "/tmp/${REV}_cpu_cur"  "/tmp/${REV}_$(basename "$CU_SRC")" -lm 2>"/tmp/${REV}_gcc.log" || "$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur"  "/tmp/${REV}_$(basename "$CU_SRC")" -lm 2>"/tmp/${REV}_gcc.log"; } && pass "cu_cpu_harness_builds[424]" || fail "cu_cpu_harness_builds[424]" "$(head -3 /tmp/${REV}_gcc.log)"
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
  ptx_stats "kernel_refill_gpu"
  [[ -n "$FR" && "$FR" -le 200 && "$SP" == "0 bytes spill stores, 0 bytes spill loads" ]] && pass "S1_ptxas_refill_frame${FR}_spill0_regs${RG} (clang route: 192 B / 47; <425> is 192 B / 36)" || { fail "S1_ptxas_refill" "frame=${FR:-?}B regs=${RG:-?} ${SP:-no ptxas line} (expected <= 200 B, 0 spill)"; summary_exit; }
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
  else fail "${tag}_sass_${L}_differs_from_419" "424: $N_SHA / $N_REAL / $N_LOOP ; 419 binary: $P_SHA / $P_REAL / $P_LOOP -- the trial cells below are still valid, the -g path must stay on 419"; PROD_OK=0; fi
done
read -r M_SHA M_REAL M_LOOP M_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_mrv_gpu LDL)"
[[ "$M_SHA" == "500eb8a37ec40f49" ]] && pass "S7_sass_mrv_is_423s (sha $M_SHA, real $M_REAL)" || info "S7_sass_mrv" "sha $M_SHA real $M_REAL (423's: 500eb8a37ec40f49 / 464) -- mrv is not run here"
read -r R_SHA R_REAL R_LOOP R_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_refill_gpu)"
[[ "$R_SHA" != "CANNOT_LOCATE" ]] && pass "S8_sass_refill_recorded (sha $R_SHA, real $R_REAL, innermost loop with a 64-bit local load $R_LOOP; <425>: real 454, loop 111)" || { fail "S8_sass_refill_missing" "no kernel_refill_gpu in the binary"; summary_exit; }
info "S8_loops_with_stack_loads" "refill: $(sass_loops "$LOGDIR/sass/${CU_BIN}.txt" kernel_refill_gpu)   ;   419 <425>: $(sass_loops "$LOGDIR/sass/419_kernel_maxd16.txt" kernel_dfs_iter_gpu_maxd16ILi425E)   (refill should have ONE such loop: the record loop is gone)"
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
Y2T_OK=0
if [[ "$SKIP_Y2" == "1" ]]; then info "Y2" "skipped (SKIP_Y2=1)"
else
  banner "Y2: CPU equivalence ($OMP_THREADS threads): refill == the 419_ref_dfs outputs (sha256), per record and per thread"
  for trip in "c:C:$SLC25:$REFC_SHA:$REFC_TOTAL" "t:T:$SLT:$REFT_SHA:$REFT_TOTAL"; do IFS=: read -r lab sl in want wtot <<< "$trip"
    T0=$(date +%s); env NQ_LOGIC=refill NQ_CPU_STRIDE="$NSL" "/tmp/${REV}_cpu_cur" 25 "$in" "/tmp/${REV}_y2_refill_${sl}.bin" > "$LOGDIR/2_cpu_N25${sl}_refill.log" 2>&1; T1=$(date +%s)
    y2_valid "$LOGDIR/2_cpu_N25${sl}_refill.log" "/tmp/${REV}_y2_refill_${sl}.bin" "$NSL" "$NSL" "refill_$sl" || summary_exit
    s="$(sha256sum "/tmp/${REV}_y2_refill_${sl}.bin" | cut -c1-16)"; tot="$(grep -o 'total_sum=[0-9]*' "$LOGDIR/2_cpu_N25${sl}_refill.log" | tail -1 | cut -d= -f2)"
    if grep -q 'logic=refill' "$LOGDIR/2_cpu_N25${sl}_refill.log" && [[ "$s" == "$want" && "$tot" == "$wtot" ]]; then pass "Y2${lab}_cpu_N25_${sl}_refill_per_record_equals_ref419 (1,024 records, total $tot, sha $s; $((T1-T0)) s)"
    else fail "Y2${lab}_cpu_N25_${sl}" "sha $s total $tot, expected $want / $wtot: refill differs from the reference on slice $sl -- stop"; summary_exit; fi
  done
  Y2T_OK=1
  env NQ_LOGIC=refill NQ_CPU_STRIDE="$S32" "/tmp/${REV}_cpu_cur" 25 "$SLT" "/tmp/${REV}_y2_refill_T_s32.bin" > "$LOGDIR/2_cpu_N25T_refill_s32.log" 2>&1
  y2_valid "$LOGDIR/2_cpu_N25T_refill_s32.log" "/tmp/${REV}_y2_refill_T_s32.bin" "$NSL" "$S32" refill_T_s32 || summary_exit
  s="$(sha256sum "/tmp/${REV}_y2_refill_T_s32.bin" | cut -c1-16)"
  [[ "$s" == "$REFT_S32_SHA" ]] && pass "Y2s_cpu_N25_T_refill_stride32_per_thread_sums_equal_ref419 (32 threads x 32 records in a row, sha $s)" || { fail "Y2s_cpu_N25_T_stride32" "sha $s != $REFT_S32_SHA: a thread's stream of records does not add up to the reference -- stop"; summary_exit; }
  head -c $((DFS_CHECK*28)) "$SLC25" > "/tmp/${REV}_c_head.bin"; head -c $((DFS_CHECK*8)) "/tmp/${REV}_y2_refill_C.bin" > "/tmp/${REV}_y2_ref_head.bin"
  env -u NQ_LOGIC "/tmp/${REV}_cpu_cur" 25 "/tmp/${REV}_c_head.bin" "/tmp/${REV}_y2_dfs_head.bin" > "$LOGDIR/2_cpu_N25C_dfs.log" 2>&1
  y2_valid "$LOGDIR/2_cpu_N25C_dfs.log" "/tmp/${REV}_y2_dfs_head.bin" "$DFS_CHECK" "$DFS_CHECK" dfs_head || summary_exit
  grep -q 'logic=dfs' "$LOGDIR/2_cpu_N25C_dfs.log" && cmp -s "/tmp/${REV}_y2_dfs_head.bin" "/tmp/${REV}_y2_ref_head.bin" && pass "Y2d_cpu_N25_dfs_default_equals_ref419 ($DFS_CHECK records; NQ_LOGIC unset = the 419 search)" || { fail "Y2d_cpu_N25_dfs" "the default path differs from the reference: stop"; summary_exit; }
fi
banner "Y3: refusals"
NQ_LOGIC=refill "/tmp/${REV}_cpu_cur" 26 "$SLC25" /tmp/${REV}_y3a.bin 8 > "$LOGDIR/2_y3_n26.log" 2>&1; rc=$?; [[ "$rc" == "3" ]] && pass "Y3_N26_refill_refused_rc3" || fail "Y3_N26_refill_refused_rc3" "rc=$rc"
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
use_ratio() { if [[ "${NORM[$1]}" != "?" && "${NORM[$2]}" != "?" ]]; then ratio "${NORM[$1]}" "${NORM[$2]}"; else ratio "${VAL[$1]}" "${VAL[$2]}"; fi; }
in_band() { le "$2" "$1" && le "$1" "$3"; }

banner "B1: $CU_BIN, NQ_LOGIC=dfs, N=21, layout auto -- the production path inside the 424 binary"
run_cell B1 "$CU_BIN" 21 "$IN21" 32 960 dfs "$ORACLE21" || summary_exit
[[ "${TOTAL[B1]:-}" == "$ORACLE21" ]] && pass "oracle[B1]" || { fail "oracle[B1]" "total_sum=${TOTAL[B1]:-?} != $ORACLE21"; PROD_OK=0; }
if [[ "$HAVEREF21" == "1" ]]; then cmp -s "$LOGDIR/bins/B1_results.bin" "$REF21" && pass "B1_per_thread_results_identical_to_411_B1" || { fail "B1_per_thread_results_DIFFER_from_411_B1" "the default path changed"; PROD_OK=0; }; else PROD_OK=0; fi

pair() { local n="$1" in="$2" oracle="$3" z="Z$1" r="R$1"
  banner "$z / $r: N=$n at layout 425 -- as built (dfs) against refill, same session, same input, same stride"
  sleep "$COOLDOWN"; CELL_LAYOUT=425 run_cell "$z" "$CU_BIN" "$n" "$in" 32 960 dfs "$oracle" || return 1
  [[ "${TOTAL[$z]:-}" == "$oracle" ]] && pass "oracle[$z]" || { fail "oracle[$z]" "total_sum=${TOTAL[$z]:-?} != $oracle"; return 1; }
  sleep "$COOLDOWN"; CELL_LAYOUT=425 run_cell "$r" "$CU_BIN" "$n" "$in" 32 960 refill "$oracle" || return 1
  [[ "${TOTAL[$r]:-}" == "$oracle" ]] && pass "oracle[$r]" || { fail "oracle[$r]" "total_sum=${TOTAL[$r]:-?} != $oracle -- the refill search is wrong at N=$n: stop"; return 1; }
  cmp -s "$LOGDIR/bins/${r}_results.bin" "$LOGDIR/bins/${z}_results.bin" && pass "${r}_per_thread_results_identical_to_${z} (30,720 thread sums)" || { fail "${r}_per_thread_results_DIFFER_from_${z}" "same total, different thread sums: stop"; return 1; }
  return 0; }
pair 21 "$IN21" "$ORACLE21" || summary_exit
R21="$(use_ratio R21 Z21)"; info "R21_vs_Z21" "ratio $R21 (raw $(ratio "${VAL[R21]}" "${VAL[Z21]}")) = $(awk -v r="$R21" 'BEGIN{printf "%.2f", 1/r}')x ; Z21 against B1 (layout 425 against 403, as built): raw $(ratio "${VAL[Z21]}" "${VAL[B1]}")"
if in_band "$R21" "$P2LO" "$P2HI"; then pass "P2_R21_over_Z21_in_${P2LO}_${P2HI} ($R21)"; else info "P2_outside_statement" "R21 / Z21 = $R21, stated $P2LO .. $P2HI"; fi
R22="?"
if [[ "$SKIP_N22" != "1" && -n "$IN22" ]]; then
  pair 22 "$IN22" "$ORACLE22" || summary_exit
  R22="$(use_ratio R22 Z22)"; info "R22_vs_Z22" "ratio $R22 (raw $(ratio "${VAL[R22]}" "${VAL[Z22]}")) = $(awk -v r="$R22" 'BEGIN{printf "%.2f", 1/r}')x"
  if in_band "$R22" "$P3LO" "$P3HI"; then pass "P3_R22_over_Z22_in_${P3LO}_${P3HI} ($R22)"; else info "P3_outside_statement" "R22 / Z22 = $R22, stated $P3LO .. $P3HI"; fi
else info "N22" "skipped (SKIP_N22=1 or no N=22 input found)"; fi

R_USE="?"
if [[ "$SKIP_DR960" != "1" ]]; then
  banner "DR960: refill on the N=25 slice B (307,200 records) -- 419's D960: $D960_419_MS ms (about 52 min as built)"
  sleep "$COOLDOWN"; run_cell DR960 "$CU_BIN" 25 "$SLB" 32 960 refill || summary_exit
  [[ "${TOTAL[DR960]}" == "$D960_419_TOTAL" ]] && pass "DR960_total_sum_equals_419s ($D960_419_TOTAL)" || { fail "DR960_total_sum" "${TOTAL[DR960]} != $D960_419_TOTAL -- the refill search is wrong on slice B: stop"; summary_exit; }
  DMS="$(sha256sum "$LOGDIR/bins/DR960_results.bin" | cut -c1-8)"
  [[ "$DMS" == "$D960_419_BIN_SHA" ]] && pass "DR960_per_thread_results_identical_to_419_D960 (results.bin sha $DMS, 30,720 thread sums over 307,200 records)" || fail "DR960_per_thread_results" "results.bin sha $DMS != $D960_419_BIN_SHA (the total matched)"
  R_RAW="$(ratio "${VAL[DR960]}" "$D960_419_MS")"; R_USE="$R_RAW"; R_KIND="raw"
  if [[ -n "$D960_419_NORM" && "${NORM[DR960]}" != "?" ]]; then R_USE="$(ratio "${NORM[DR960]}" "$D960_419_NORM")"; R_KIND="normalised"; fi
  info "DR960_vs_419_D960" "ratio $R_USE ($R_KIND; raw $R_RAW) = $(awk -v r="$R_USE" 'BEGIN{printf "%.2f", 1/r}')x faster"
  if le 0.97 "$R_USE"; then fail "P1_REFUTED_refill_not_faster" "DR960 / D960 = $R_USE >= 0.97 -- read S8 (did nvcc keep one loop?) and the power trace clk_DR960.tsv"
  elif in_band "$R_USE" "$P1LO" "$P1HI"; then pass "P1_ratio_in_${P1LO}_${P1HI} ($R_USE, $R_KIND)"
  else info "P1_outside_statement" "DR960 / D960 = $R_USE ($R_KIND), stated $P1LO .. $P1HI, refutation line 0.97"; fi
fi
for c in "${!VAL[@]}"; do f="$LOGDIR/clk_$c.tsv"; [[ -f "$f" ]] && info "power[$c]" "$(awk -F, 'NR>1 && $6+0>50 {n++; p+=$4; if($4>mx)mx=$4} END{if(n) printf "mean %.0f W, max %.0f W over %d samples (419 as built sits at the 299 W cap)", p/n, mx, n; else print "no samples"}' "$f")"; done
CLK_OK=1; for c in "${!VAL[@]}"; do m="$(awk -F'\t' -v c="$c" '$1==c{print $13}' "$TSV" | tail -1)"; [[ "$m" == "?" || -z "$m" ]] && continue; awk -v x="$m" 'BEGIN{d=x-1710; if(d<0)d=-d; exit !(d<=34)}' || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "W3_sm_clock_1710_all_cells" || info "W3_sm_clock" "power-capped cells read below 1676; ratios above use norm_ms"
[[ "$PROD_OK" == "1" ]] && info "PROD" "the 424 binary's default path is 419's (SASS, N=21 per-thread results)" || info "PROD" "the 424 binary's default path is NOT proven identical to 419 -- keep 419 on the -g path (see the S1-S6 / B1 lines)"
if [[ "$FAIL" -eq 0 && "$R_USE" != "?" ]] && le "$R_USE" 0.95 && le "$R21" 1.0; then info "ADOPTION" "refill: every HARD gate passed, N=25 slice B $R_USE, N=21 $R21, N=22 $R22 -> 425 moves the -g path to the refill kernel (N=25 first, oracle-gated)"
elif [[ "$FAIL" -eq 0 ]]; then info "ADOPTION" "NOT adopted by the stated rule (N=25 slice B $R_USE, N=21 $R21, N=22 $R22) -- correct, the gain is short of the rule; Suzuki decides"
else info "ADOPTION" "NOT adopted -- see FAIL lines"; fi
banner "Results"; python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import sys
rows=[l.rstrip('\n').split('\t') for l in open(sys.argv[1])]
w=[max(len(r[i]) for r in rows if i<len(r)) for i in range(len(rows[0]))]
for r in rows: print('  '.join(c.ljust(w[i]) for i,c in enumerate(r)))
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -f "/tmp/${REV}_y2_refill_T.bin" "$LOGDIR/bins/Y2t_refill_cpu_T.bin" 2>/dev/null; cp -f "/tmp/${REV}_y2_refill_C.bin" "$LOGDIR/bins/Y2c_refill_cpu_C.bin" 2>/dev/null; cp -f "/tmp/${REV}_y2_refill_T_s32.bin" "$LOGDIR/bins/Y2s_refill_cpu_T_s32.bin" 2>/dev/null
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd1[46]" --exclude="*slice[BC]*.bin" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
