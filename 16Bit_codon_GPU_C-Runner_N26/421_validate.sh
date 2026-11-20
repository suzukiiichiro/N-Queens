#!/usr/bin/env bash
# 421_validate.sh -- the split / multi-GPU / multi-machine run ON THE 419 KERNEL, validated on a 4-GPU instance
# (g5.12xlarge, 4 x A10G). Production values do not change; no kernel change (the 419 runner renamed and rebuilt).
#
# WHERE 419 STANDS (2026-11-16): adopted 11/4 (419_validate OK=71 FAIL=0). N=25 production value 728,149,056 ms
#   (419-r3, 11/4 23:27 -> 11/13 09:44, oracle 2,207,893,435,808,352 MATCH, results.bin sha bd2de957).
#   418 (10/26, never run) wraps the 417 runner, which counts N=25 0.16 % short (schedule-word overflow): it must not
#   be used at N >= 25. N=26 is ~2,000 GPU-hours (x9.96 extrapolation) and needs the split run.
#
# rev421 -- .cu: 419 renamed (header note only). .py: 3 executable lines (tags, CRunnerEntry -> ./421_kernel_maxd16).
#   421_split_sched.py / 421_sum_results.py: 418's, renamed. 421_multigpu_run.sh: 418's driver + WORKER_ID/WORKER_TOTAL
#   (chunk k runs on the machine with k % W == i), SUM_ONLY=1 (check + add collected part results, no GPU), RESUME=1.
#
#   S0 static: .py diff 3/3 vs 419Py; .cu code-region diff 0/0 vs 419; kernel sha 3b67e26c; notes; constants; the
#      3-bit schedule word and the overflow guard still there; splitter/summer selftests; driver knobs; bash -n; gcc   HARD
#   S1 ptxas: <402>,<403> 160 B ; <423> 176 B ; <424> 184 B ; <425> 192 B ; spill 0 ; regs <= 44                  HARD
#   S2-S6 all five SASS instantiations identical to 419 (<402> 821302f2/446/109, <403> e940cb96/445/108,
#      <423> c8316e18/447/107, <424> f60ecee7/447/107, <425> 4e3b9e8f/454/111)                                    HARD
#   Cells (BLOCK 32, MB 960, helper 1 + 128 MB, NQ_EXTRA_CTX unset; every cell checks its total):
#     G0..G3  N=21 single run on GPU 0,1,2,3 in turn   results.bin == REF21 (411 B1 1bfab090)                 HARD
#     P21x4   N=21 split 4 chunks, all GPUs (the driver) sum == G0 results.bin byte-identical, oracle             HARD
#     P21w    same LOGDIR twice: WORKER 1/2 (chunks 1,3; must end "partial"), then no WORKER (skips 2, runs 0,2)
#             -> sum == G0 byte-identical, oracle (multi-machine partition + resume)                              HARD
#     P21s    SUM_ONLY=1 on P21w's LOGDIR -> sum == G0 byte-identical (the collect-and-add path)                  HARD
#     P25Bx4  N=25 slice B (307,200 equal-spaced records) split 4 chunks -> sum == 419's D960 results.bin
#             (419_validate_20261104_214321/bins/D960_results.bin, sha c881e856, total 6,910,807,391,840):
#             depth 17, the 419 kernel, another machine, split -- end to end                                      HARD
#     S23     N=23 single run on GPU 0                   oracle 24,233,937,684,440 (kernel_ms vs 7,717,284: info) HARD
#     P23x4   N=23 split 4 chunks                        sum == S23 byte-identical, oracle                        HARD
#     P23x8   N=23 split 8 chunks (queue)                sum == S23 byte-identical, oracle                        HARD
#   PRE-REGISTERED (no gain claimed; production values unchanged):
#     W1 chunking cost = sum of chunk kernel_ms / single kernel_ms at N=23: x4 in 1.02..1.07, x8 in 1.04..1.10 (as 418;
#        the N=25 tail turned out to be 7 % of 202 h, not 5 %, so expect the upper half). Refutation: x8 > 1.15.
#     W2 wall-clock speedup on 4 GPUs = single kernel_ms / wall >= 3.4x (both N=23 split cells).
#     W3 G0..G3 kernel_ms within 1 % of each other (four equal A10Gs).
#     W4 (info) P25Bx4 sum of chunk kernel_ms / 3,099,327 (419's D960 on NQ_CRunner, another machine) in 1.00..1.10.
#   ADOPT: all HARD -> 421 is the production binary/.py (SASS identical to 419, values unchanged) and
#     421_multigpu_run.sh is the driver for N >= 25. N=26 itself (17 ancestor frames, input) is 422.
#
# NEEDS on this machine (besides the git sources): the N=21 / N=23 / N=25 sched394f inputs, 413_slice_soa7.py,
#   411_validate_20261006_150938/bins/B1_results.bin, 419_validate_20261104_214321/bins/D960_results.bin,
#   419_kernel_maxd16.cu and 419Py_kernel_maxd16_final.py (the diff bases).
#
# USAGE (on the 4-GPU instance)
#   STATIC_ONLY=1 bash 421_validate.sh                                # S0, ~1 min, GPU zero
#   set -o pipefail; bash 421_validate.sh 2>&1 | tee 421_validate_stdout.log   # ~3.8 h (S23 2h08, P23 x4/x8 ~35 min each, P25Bx4 ~20 min)
#   SKIP_N23=1 bash 421_validate.sh                                   # everything but N=23 (~35 min; no W1/W2)
#   SKIP_N25B=1 ...   SKIP_BUILD=1 ...   SKIP_CELLS=1 ...

set -u
REV="421"
SKIP_BUILD="${SKIP_BUILD:-0}"; STATIC_ONLY="${STATIC_ONLY:-0}"; SKIP_N23="${SKIP_N23:-0}"; SKIP_CELLS="${SKIP_CELLS:-0}"; SKIP_N25B="${SKIP_N25B:-0}"
PY_SRC="${PY_SRC:-421Py_kernel_maxd16_final.py}"; PY_BIN="${PY_BIN:-${PY_SRC%.py}}"
PREV_PY="${PREV_PY:-419Py_kernel_maxd16_final.py}"
CU_SRC="${CU_SRC:-421_kernel_maxd16.cu}"; CU_BIN="${CU_BIN:-421_kernel_maxd16}"
PREV_CU="${PREV_CU:-419_kernel_maxd16.cu}"
SPLITTER="${SPLITTER:-421_split_sched.py}"; SUMMER="${SUMMER:-421_sum_results.py}"; DRIVER="${DRIVER:-421_multigpu_run.sh}"
REF21="${REF21:-411_validate_20261006_150938/bins/B1_results.bin}"; REF21_SHA="${REF21_SHA:-1bfab090}"
SASS402_SHA="821302f227dcd160"; SASS402_REAL=446; SASS402_LOOP=109
SASS403_SHA="e940cb96e5e65fc4"; SASS403_REAL=445; SASS403_LOOP=108
SASS423_SHA="c8316e18"; SASS423_REAL=447; SASS423_LOOP=107
SASS424_SHA="f60ecee7"; SASS424_REAL=447; SASS424_LOOP=107
SASS425_SHA="${SASS425_SHA:-4e3b9e8f}"; SASS425_REAL=454; SASS425_LOOP=111
KERNEL_SHA="3b67e26c23bf3454"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"; CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"; CODON="${CODON:-codon}"; GCC="${GCC:-gcc}"; ARCH="${ARCH:-sm_86}"
HMB="${HMB:-128}"; COOLDOWN="${COOLDOWN:-15}"; CLK_INTERVAL="${CLK_INTERVAL:-5}"
ORACLE21=314666222712; ORACLE23=24233937684440; PROD23_MS=7717284
SCHED25="${SCHED25:-constellations_N25_7.bin.soa_ref_361.bin.d17only_417.bin.sched394f.bin}"; EXPECTED25=98138478
SLICER="${SLICER:-413_slice_soa7.py}"; SLICE_B="${SLICE_B:-307200}"; SLB="${SLB:-${REV}_n25_sliceB_${SLICE_B}.bin}"
REF25B="${REF25B:-419_validate_20261104_214321/bins/D960_results.bin}"; REF25B_SHA="${REF25B_SHA:-c881e856}"; TOTAL25B=6910807391840; D960_419_MS=3099327.25
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
for f in "$PY_SRC" "$CU_SRC" "$PREV_CU" "$SPLITTER" "$SUMMER" "$DRIVER"; do [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"; done
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }
py_code_region() { python3 -c "
import sys
lines=open(sys.argv[1],encoding='utf-8').read().split('\n')
i=next(i for i,l in enumerate(lines) if l.startswith('import '))
sys.stdout.write('\n'.join(l for l in lines[i:] if not l.lstrip().startswith('#')))
" "$1"; }
py_code_region "$PY_SRC" > "/tmp/${REV}_code_only.py"; CODE="/tmp/${REV}_code_only.py"
NOTE_LINES=$(awk '/^# =+$/{f=1} f&&/^#/{n++} END{print n+0}' "$PY_SRC")
{ grep -q "^# 421 " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 421 ...' note block in $PY_SRC (rule 8)"
grep -qE '^REV_TAG:str="421"' "$CODE" && pass "source_rev_tag_is_421" || fail "source_rev_tag_is_421" "wrong REV_TAG"
grep -qF 'CRunnerEntry(17,"./421_kernel_maxd16","NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_entry_maxd17_421_binary_helper_prefix" || fail "source_table_entry" "table entry wrong"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=960\b" "$CODE" && pass "source_default_max_blocks_is_960" || fail "source_default_max_blocks_is_960" "default not 960"
grep -qF 'elif N>=22 and N<=25:' "$CODE" && grep -qF 'd17only_417.bin' "$CODE" && pass "source_417_n25_wiring_kept" || fail "source_417_n25_wiring_kept" "missing"
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
[[ "$KB" == "$KERNEL_SHA"* ]] && pass "cu_kernel_region_sha_is_419s (${KB:0:16})" || fail "cu_kernel_region_sha" "sha ${KB:0:16} != $KERNEL_SHA"
CR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true); CA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
[[ "$CR_" == "0" && "$CA_" == "0" ]] && pass "cu_code_region_identical_to_419 (removed=0 added=0)" || fail "cu_code_region_fingerprint" "removed=$CR_ added=$CA_, expected 0/0 (no code change in 421)"
grep -q '^#define PACK425_ANCESTOR 16' "$CU_SRC" && grep -q '^#define PACK425_MAXN    25' "$CU_SRC" && grep -q '^#define PACK424_ANCESTOR 15' "$CU_SRC" && pass "cu_pack_constants_unchanged" || fail "cu_pack_constants_unchanged" "changed"
grep -q "rev421 " "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev421 note in header"
grep -q 'const int sh425 = schedule_depth \* 3;' "$CU_SRC" && grep -q 'cur_depth \* ((layout == PACK_LAYOUT_425) ? 3 : 4)' "$CU_SRC" && grep -q 'SCHEDULE-WORD-OVERFLOW' "$CU_SRC" && pass "cu_is_the_419_kernel (3-bit schedule word for 425 + overflow guard)" || fail "cu_is_the_419_kernel" "the 419 fix is not in $CU_SRC -- this is not a 419-based source"
python3 -I "$SPLITTER" --selftest > "/tmp/${REV}_split_selftest.log" 2>&1 && pass "splitter_selftest (plan boundaries at stride multiples, concat == source, K bounds)" || fail "splitter_selftest" "$(tail -2 /tmp/${REV}_split_selftest.log | tr '\n' ' ')"
python3 -I "$SUMMER" --selftest > "/tmp/${REV}_sum_selftest.log" 2>&1 && pass "summer_selftest (elementwise u64 sum, oracle and ref verdicts)" || fail "summer_selftest" "$(tail -2 /tmp/${REV}_sum_selftest.log | tr '\n' ' ')"
grep -q 'STRIDE=\$((BLOCK\*MB))' "$DRIVER" && grep -q 'mkdir "\$LOGDIR/lock/\$k"' "$DRIVER" && pass "driver_splits_at_stride_and_queues_chunks" || fail "driver_static" "stride/queue lines not found in $DRIVER"
grep -q 'own() { (( \$1 % WORKER_TOTAL == WORKER_ID )); }' "$DRIVER" && grep -q 'SUM_ONLY="\${SUM_ONLY:-0}"; RESUME="\${RESUME:-0}"' "$DRIVER" && grep -q 'CU_BIN="\${CU_BIN:-./421_kernel_maxd16}"' "$DRIVER" && pass "driver_has_worker_partition_sum_only_resume_and_names_the_421_binary" || fail "driver_knobs" "WORKER/SUM_ONLY/RESUME/CU_BIN lines not found in $DRIVER"
bash -n "$DRIVER" 2>"/tmp/${REV}_bashn.log" && pass "driver_bash_syntax" || fail "driver_bash_syntax" "$(head -2 /tmp/${REV}_bashn.log)"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"/tmp/${REV}_gcc.log" && pass "cu_cpu_harness_builds[421]" || fail "cu_cpu_harness_builds[421]" "$(head -3 /tmp/${REV}_gcc.log)"
if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Build, ptxas (S1), SASS (S2-S6), codon
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR/sass" "$LOGDIR/bins"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; df -h . 2>&1; free -g 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
[[ ! -x "$NVCC" ]] && NVCC="nvcc"; [[ ! -x "$CUOBJDUMP" ]] && CUOBJDUMP="cuobjdump"
NGPU="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')"; [[ "$NGPU" -ge 1 ]] && pass "gpus_visible_${NGPU}" || { fail "gpus" "nvidia-smi lists no GPU"; summary_exit; }
[[ "$NGPU" -ge 4 ]] || info "gpus" "only $NGPU GPU(s): the split cells run with GPUS=$NGPU (the byte-identity checks still hold; W2 needs 4)"
banner "Building"
if [[ "$SKIP_BUILD" == "1" ]]; then
  [[ -x "$CU_BIN" ]] && pass "existing_binary_used[$CU_BIN] ($(stat -c %y "$CU_BIN" | cut -c1-19); SKIP_BUILD=1)" || { fail "cu_binary_missing" "$CU_BIN absent and SKIP_BUILD=1"; summary_exit; }
  info "S1_ptxas" "skipped (no fresh nvcc log)"
else
  rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc_${REV}.log" 2>&1
  [[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "$(grep -i error "$LOGDIR/05_nvcc_${REV}.log" | head -3)"; summary_exit; }
  ptx_stats() { FR="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"; SP="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes spill stores, [0-9]* bytes spill loads' | head -1)"; RG="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"; }
  for pair in 402:160 403:160 423:176 424:184 425:192; do IFS=: read -r L want <<< "$pair"; ptx_stats "kernel_dfs_iter_gpu_maxd16ILi${L}E"
    [[ "$FR" == "$want" && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "S1_ptxas_L${L}_frame${want}_spill0_regs${RG}" || { fail "S1_ptxas_L${L}" "frame=${FR}B regs=${RG} ${SP} (expected $want / 0 / <= 44)"; summary_exit; }; done
fi
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
  if [[ "$real" == "-" ]]; then [[ "$S_SHA" == "$sha"* && "$S_LOOP" == "$loop" ]] && pass "${tag}_sass_${L}_is_419s (sha $S_SHA, real $S_REAL, loop $S_LOOP)" || { fail "${tag}_sass_${L}_differs_from_419" "sha $S_SHA real $S_REAL loop $S_LOOP != $sha* / loop $loop -- a renamed, unchanged .cu must give the same SASS: stop"; summary_exit; }
  else [[ "$S_SHA" == "$sha"* && "$S_REAL" == "$real" && "$S_LOOP" == "$loop" ]] && pass "${tag}_sass_${L}_is_419s (sha $S_SHA, real $S_REAL, loop $S_LOOP)" || { fail "${tag}_sass_${L}_differs_from_419" "sha $S_SHA real $S_REAL loop $S_LOOP != $sha* / $real / $loop -- stop"; summary_exit; }; fi; }
sass_gate 402 "$SASS402_SHA" "$SASS402_REAL" "$SASS402_LOOP" S2
sass_gate 403 "$SASS403_SHA" "$SASS403_REAL" "$SASS403_LOOP" S3
sass_gate 423 "$SASS423_SHA" "$SASS423_REAL" "$SASS423_LOOP" S4
sass_gate 424 "$SASS424_SHA" "$SASS424_REAL" "$SASS424_LOOP" S5
sass_gate 425 "$SASS425_SHA" "$SASS425_REAL" "$SASS425_LOOP" S6
if [[ "$SKIP_BUILD" == "1" ]]; then [[ -x "$PY_BIN" ]] && pass "existing_py_binary_used[$PY_BIN]" || { fail "py_binary_missing" "$PY_BIN absent and SKIP_BUILD=1"; summary_exit; }
else rm -f "$PY_BIN"; "$CODON" build -release "$PY_SRC" > "$LOGDIR/06_codon_${REV}.log" 2>&1
  [[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN] (the -g path now names ./$CU_BIN; no production run here)" || { fail "codon_build" "$(tail -3 "$LOGDIR/06_codon_${REV}.log")"; summary_exit; }; fi

find_input() { local n="$1" c; for lg in $(ls -t 4*_crunner_logs/crunner_*_N${n}.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { echo "$c"; return; }; done; ls -t constellations_N${n}_*.sched394f.bin 2>/dev/null | head -1; }
IN21="$(find_input 21)"; [[ -f "${IN21:-/nonexistent}" ]] && pass "input21_located ($IN21, $(( $(stat -c %s "$IN21") / 28 )) records)" || { fail "input21_located" "no N=21 sched input"; summary_exit; }
if [[ "$SKIP_N23" != "1" ]]; then IN23="$(find_input 23)"; [[ -f "${IN23:-/nonexistent}" ]] && pass "input23_located ($IN23, $(( $(stat -c %s "$IN23") / 28 )) records)" || { fail "input23_located" "no N=23 sched input (SKIP_N23=1 to skip)"; summary_exit; }; fi
if [[ -f "$REF21" ]]; then RS="$(sha256sum "$REF21" | cut -c1-8)"; [[ "$RS" == "$REF21_SHA" ]] && pass "ref21_is_411_B1 ($REF21 sha $RS)" || fail "ref21_is_411_B1" "sha $RS != $REF21_SHA"; else fail "ref21_present" "$REF21 missing -- pass REF21=<411 B1 results.bin>"; fi
if [[ "$SKIP_N25B" != "1" ]]; then
  if [[ -f "$REF25B" ]]; then RS="$(sha256sum "$REF25B" | cut -c1-8)"; [[ "$RS" == "$REF25B_SHA" ]] && pass "ref25B_is_419_D960 ($REF25B sha $RS)" || fail "ref25B_is_419_D960" "sha $RS != $REF25B_SHA"; else fail "ref25B_present" "$REF25B missing -- copy it from NQ_CRunner (or SKIP_N25B=1)"; fi
  if [[ ! -s "$SLB" || $(( $(stat -c %s "$SLB" 2>/dev/null || echo 0) / 28 )) -ne "$SLICE_B" ]]; then
    [[ -s "$SCHED25" && $(( $(stat -c %s "$SCHED25") / 28 )) -eq "$EXPECTED25" ]] && pass "sched25_present ($EXPECTED25 records)" || { fail "sched25" "$SCHED25 missing/wrong size (needed to cut slice B; or pass SLB=<419_n25_sliceB_307200.bin>, or SKIP_N25B=1)"; summary_exit; }
    python3 "$SLICER" "$SCHED25" "$SLB" "$SLICE_B" > "$LOGDIR/13_slice_B.log" 2>&1 || true
  fi
  [[ -s "$SLB" && $(( $(stat -c %s "$SLB") / 28 )) -eq "$SLICE_B" ]] && pass "sliceB_ready[$SLB] ($SLICE_B records, $SLICER as in 419_validate)" || { fail "sliceB" "$(tail -1 "$LOGDIR/13_slice_B.log" 2>/dev/null)"; summary_exit; }
fi
[[ "$FAIL" -gt 0 ]] && summary_exit
[[ "$SKIP_CELLS" == "1" ]] && { info "cells" "skipped"; summary_exit; }

# ---------------------------------------------------------------------
# 3. Cells
# ---------------------------------------------------------------------
printf 'cell\tgpu\tN\tchunks\trecords\tkernel_ms\tnorm_ms\twall_s\ttotal_sum\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
declare -A VAL NORM TOTAL WALL
gpu_gate() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; [[ -z "$apps" ]] && return 0; fail "gpu_empty_before[$1]" "$(echo "$apps" | tr '\n' ' ')"; return 1; }
run_single() {  # cell gpu N input oracle
  local cell="$1" g="$2" n="$3" in="$4" oracle="$5"; gpu_gate "$cell" || return 1
  local lg="$LOGDIR/3_${cell}.log" f="$LOGDIR/clk_${cell}.tsv" start t0; start="$(date -Is)"; t0=$(date +%s)
  echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"
  ( while true; do nvidia-smi -i "$g" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & local cp=$!
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX CUDA_VISIBLE_DEVICES="$g" NQ_LAYOUT=auto NQ_BLOCK=32 NQ_MAX_BLOCKS=960 NQ_HELPER_CTX=1 NQ_HELPER_MB="$HMB" "./$CU_BIN" "$n" "$in" "$LOGDIR/bins/${cell}_results.bin" "$oracle" > "$lg" 2>&1 || true
  kill "$cp" 2>/dev/null; wait "$cp" 2>/dev/null || true; local wall=$(( $(date +%s)-t0 ))
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 && $6+0>50 && $4+0>150 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.1f %d %d\n", s/n, min, t; else print "? ? ?"}' "$f")
  local KMS TOT; KMS="$(grep -o 'kernel_ms=[0-9.]*' "$lg" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  local NORMV; NORMV="$(awk -v k="${KMS:-0}" -v s="${SMMEAN/\?/0}" 'BEGIN{ if(s>0) printf "%.1f", k*s/1710; else print "?" }')"
  printf '%s\t%s\t%s\t1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$g" "$n" "$(( $(stat -c %s "$in") / 28 ))" "${KMS:-?}" "$NORMV" "$wall" "${TOT:-?}" "$SMMEAN" "$SMMIN" "$TMAX" "$start" >> "$TSV"
  [[ -n "${KMS:-}" && "${TOT:-}" == "$oracle" ]] || { fail "run[$cell]" "kernel_ms=${KMS:-?} total_sum=${TOT:-?} (oracle $oracle): $(tail -1 "$lg" | cut -c1-100)"; return 1; }
  VAL[$cell]="$KMS"; NORM[$cell]="$NORMV"; TOTAL[$cell]="$TOT"; WALL[$cell]="$wall"
  pass "oracle[$cell]"; info "$cell" "gpu=$g N=$n kernel_ms=$KMS norm_ms=$NORMV wall=${wall}s (sm_mean $SMMEAN, ${TMAX}C) results=$(sha256sum "$LOGDIR/bins/${cell}_results.bin" | cut -c1-8)"
}
run_split() {  # cell N input chunks oracle ref_results
  local cell="$1" n="$2" in="$3" k="$4" oracle="$5" ref="$6"; gpu_gate "$cell" || return 1
  local ld="$LOGDIR/${cell}_driver" t0; t0=$(date +%s)
  env N="$n" INPUT="$in" CHUNKS="$k" GPUS="$NGPU" ORACLE="$oracle" CU_BIN="./$CU_BIN" REF_RESULTS="$ref" LOGDIR="$ld" CHUNKDIR="$LOGDIR/chunks_${cell}" SPLITTER="$SPLITTER" SUMMER="$SUMMER" \
    WORKER_ID=0 WORKER_TOTAL=1 SUM_ONLY=0 RESUME=0 BLOCK=32 MB=960 HMB="$HMB" bash "$DRIVER" > "$LOGDIR/3_${cell}.log" 2>&1; local rc=$?
  grep -E '^(OK|FAIL|INFO  TIMING|INFO  clk)' "$LOGDIR/3_${cell}.log" | sed "s/^/  [$cell] /"
  local wall=$(( $(date +%s)-t0 )); local KSUM; KSUM="$(grep -o 'sum_kernel_ms=[0-9.]*' "$LOGDIR/3_${cell}.log" | head -1 | cut -d= -f2)"
  local TOT; TOT="$(grep -o 'oracle_MATCH total_sum=[0-9]*' "$LOGDIR/3_${cell}.log" | grep -o '[0-9]*$')"
  local DW; DW="$(grep -o 'wall_s=[0-9]*' "$LOGDIR/3_${cell}.log" | head -1 | cut -d= -f2)"
  printf '%s\t0-%s\t%s\t%s\t%s\t%s\t?\t%s\t%s\t?\t?\t?\t%s\n' "$cell" "$((NGPU-1))" "$n" "$k" "$(( $(stat -c %s "$in") / 28 ))" "${KSUM:-?}" "${DW:-$wall}" "${TOT:-?}" "$(date -Is)" >> "$TSV"
  cp -f "$ld/bins/N${n}_results_sum.bin" "$LOGDIR/bins/${cell}_results_sum.bin" 2>/dev/null || true
  [[ "$rc" == "0" ]] && grep -q 'OK    oracle_MATCH' "$LOGDIR/3_${cell}.log" && grep -q 'OK    sum_results_byte_identical_to' "$LOGDIR/3_${cell}.log" || { fail "split[$cell]" "driver rc=$rc: $(grep '^FAIL' "$LOGDIR/3_${cell}.log" | head -2 | tr '\n' ' ')"; return 1; }
  VAL[$cell]="$KSUM"; TOTAL[$cell]="$TOT"; WALL[$cell]="${DW:-$wall}"
  pass "${cell}_oracle_and_sum_byte_identical_to_single_run (x$k over $NGPU GPUs)"
}

banner "G0..G$((NGPU-1)): N=21 single run on each GPU"
for ((g=0; g<NGPU; g++)); do
  run_single "G$g" "$g" 21 "$IN21" "$ORACLE21" || summary_exit
  cmp -s "$LOGDIR/bins/G${g}_results.bin" "$REF21" && pass "G${g}_results_identical_to_REF21" || fail "G${g}_results_DIFFER_from_REF21" "GPU $g does not reproduce 411 B1 -- do not use this GPU"
  [[ $g -lt $((NGPU-1)) ]] && sleep "$COOLDOWN"
done
if [[ "$NGPU" -ge 2 ]]; then
  GMIN="${VAL[G0]}"; GMAX="${VAL[G0]}"; for ((g=1; g<NGPU; g++)); do le "${VAL[G$g]}" "$GMIN" && GMIN="${VAL[G$g]}"; le "$GMAX" "${VAL[G$g]}" && GMAX="${VAL[G$g]}"; done
  GS="$(pct "$GMAX" "$GMIN")"; le "$GS" 1.0 && pass "W3_gpus_within_1pct (max/min spread ${GS}%)" || info "W3_gpu_spread" "max/min spread ${GS}% > 1% -- the GPUs are not equally fast; read the clk files (chunk queueing absorbs it)"
fi
sleep "$COOLDOWN"
banner "P21x4: N=21 split into 4 chunks over $NGPU GPUs (the production driver)"
run_split P21x4 21 "$IN21" 4 "$ORACLE21" "$LOGDIR/bins/G0_results.bin" || summary_exit
info "P21x4_vs_G0" "sum_kernel_ms/kernel_ms = $(ratio "${VAL[P21x4]}" "${VAL[G0]}") (+$(pct "${VAL[P21x4]}" "${VAL[G0]}")%), wall ${WALL[P21x4]}s vs single ${WALL[G0]}s (N=21 is too short for W1/W2: 1 chunk = 25 s)"

sleep "$COOLDOWN"
banner "P21w / P21s: worker partition (1 of 2), resume in the same LOGDIR, then SUM_ONLY"
gpu_gate P21w || summary_exit
LDW="$LOGDIR/P21w_driver"
drv21() { env N=21 INPUT="$IN21" CHUNKS=4 GPUS="$NGPU" ORACLE="$ORACLE21" CU_BIN="./$CU_BIN" REF_RESULTS="$LOGDIR/bins/G0_results.bin" LOGDIR="$LDW" CHUNKDIR="$LOGDIR/chunks_P21x4" SPLITTER="$SPLITTER" SUMMER="$SUMMER" \
    WORKER_ID=0 WORKER_TOTAL=1 SUM_ONLY=0 RESUME=0 BLOCK=32 MB=960 HMB="$HMB" "$@" bash "$DRIVER"; }
drv21 WORKER_ID=1 WORKER_TOTAL=2 > "$LOGDIR/3_P21w_a.log" 2>&1; rca=$?
if [[ "$rca" == "0" ]] && grep -q 'OK    own_2_chunks_done (1 3)' "$LOGDIR/3_P21w_a.log" && grep -q 'INFO  partial: 2 of 4 chunks present' "$LOGDIR/3_P21w_a.log" && [[ ! -f "$LDW/bins/N21_results_sum.bin" && ! -f "$LDW/bins/part00_results.bin" && ! -f "$LDW/bins/part02_results.bin" ]]; then
  pass "P21w_a_worker_1_of_2_ran_chunks_1_3_only_and_stopped_partial"
else fail "P21w_a" "rc=$rca: $(grep -E '^(FAIL|INFO  partial|OK    own)' "$LOGDIR/3_P21w_a.log" | head -3 | tr '\n' ' ')"; summary_exit; fi
sleep "$COOLDOWN"; gpu_gate P21w_b || summary_exit
drv21 > "$LOGDIR/3_P21w_b.log" 2>&1; rcb=$?
NST="$(grep -c '^\[chunk-start\]' "$LOGDIR/3_P21w_b.log" || true)"; NSK="$(grep -c '^\[chunk-skip\]' "$LOGDIR/3_P21w_b.log" || true)"
if [[ "$rcb" == "0" && "$NST" == "2" && "$NSK" == "2" ]] && grep -q 'INFO  resume: 2 of 4 own chunks already done' "$LOGDIR/3_P21w_b.log" && grep -q 'OK    oracle_MATCH' "$LOGDIR/3_P21w_b.log" && grep -q 'OK    sum_results_byte_identical_to' "$LOGDIR/3_P21w_b.log"; then
  pass "P21w_b_resume_skipped_2_ran_2_sum_byte_identical_to_G0_and_oracle"
else fail "P21w_b" "rc=$rcb started=$NST skipped=$NSK: $(grep '^FAIL' "$LOGDIR/3_P21w_b.log" | head -2 | tr '\n' ' ')"; summary_exit; fi
drv21 SUM_ONLY=1 > "$LOGDIR/3_P21s.log" 2>&1; rcs=$?
if [[ "$rcs" == "0" ]] && grep -q 'OK    all_4_chunks_present' "$LOGDIR/3_P21s.log" && grep -q 'OK    oracle_MATCH' "$LOGDIR/3_P21s.log" && grep -q 'OK    sum_results_byte_identical_to' "$LOGDIR/3_P21s.log" && ! grep -q 'chunk-start' "$LOGDIR/3_P21s.log"; then
  pass "P21s_sum_only_byte_identical_to_G0_and_oracle (no GPU work)"
else fail "P21s" "rc=$rcs: $(grep '^FAIL' "$LOGDIR/3_P21s.log" | head -2 | tr '\n' ' ')"; summary_exit; fi

if [[ "$SKIP_N25B" != "1" ]]; then
  sleep "$COOLDOWN"; banner "P25Bx4: N=25 slice B ($SLICE_B records) split into 4 chunks -- sum vs 419's D960 results.bin (depth 17, end to end)"
  run_split P25Bx4 25 "$SLB" 4 "$TOTAL25B" "$REF25B" || summary_exit
  W4="$(ratio "${VAL[P25Bx4]}" "$D960_419_MS")"
  if le 1.00 "$W4" && le "$W4" 1.10; then pass "W4_P25Bx4_chunk_sum_vs_419_D960_within_stated ($W4; stated 1.00..1.10, another machine)"; else info "W4_outside_statement" "sum_kernel_ms / 3,099,327 = $W4 (information: another machine, 3+3+2+2 stride-groups)"; fi
  info "P25Bx4" "wall ${WALL[P25Bx4]}s vs 3,099 s single (NQ_CRunner) -> $(awk -v w="${WALL[P25Bx4]}" 'BEGIN{printf "%.2fx", 3099.3/w}')"
fi

if [[ "$SKIP_N23" != "1" ]]; then
  sleep "$COOLDOWN"; banner "S23: N=23 single run on GPU 0 (~2 h 08)"
  run_single S23 0 23 "$IN23" "$ORACLE23" || summary_exit
  info "S23_vs_production" "kernel_ms ${VAL[S23]} vs 7,717,284 (413-r3, g5.2xlarge): $(pct "${VAL[S23]}" "$PROD23_MS")% (norm $(pct "${NORM[S23]}" "$PROD23_MS")%)"
  sleep "$COOLDOWN"; banner "P23x4: N=23 split into 4 chunks"
  run_split P23x4 23 "$IN23" 4 "$ORACLE23" "$LOGDIR/bins/S23_results.bin" || summary_exit
  sleep "$COOLDOWN"; banner "P23x8: N=23 split into 8 chunks (queue over $NGPU GPUs)"
  run_split P23x8 23 "$IN23" 8 "$ORACLE23" "$LOGDIR/bins/S23_results.bin" || summary_exit
  W1_4="$(ratio "${VAL[P23x4]}" "${VAL[S23]}")"; W1_8="$(ratio "${VAL[P23x8]}" "${VAL[S23]}")"
  if le "$W1_4" 1.07; then pass "W1_x4_chunking_cost_within_stated (${W1_4}; stated 1.02..1.07)"; else info "W1_x4_outside_statement" "$W1_4 > 1.07"; fi
  if le "$W1_8" 1.10; then pass "W1_x8_chunking_cost_within_stated (${W1_8}; stated 1.04..1.10)"; elif le "$W1_8" 1.15; then info "W1_x8_outside_statement" "$W1_8 in 1.10..1.15"; else fail "W1_x8_REFUTED" "$W1_8 > 1.15 -- read the per-chunk kernel_ms spread in P23x8_driver/2_chunk_events.tsv"; fi
  S2_4="$(awk -v k="${VAL[S23]}" -v w="${WALL[P23x4]}" 'BEGIN{printf "%.2f", k/1000/w}')"; S2_8="$(awk -v k="${VAL[S23]}" -v w="${WALL[P23x8]}" 'BEGIN{printf "%.2f", k/1000/w}')"
  if [[ "$NGPU" -ge 4 ]]; then le 3.4 "$S2_4" && le 3.4 "$S2_8" && pass "W2_speedup_ge_3.4x (x4 ${S2_4}x, x8 ${S2_8}x)" || info "W2_speedup_below_statement" "x4 ${S2_4}x, x8 ${S2_8}x (< 3.4x: chunk imbalance -- compare max/min chunk kernel_ms)"; else info "W2" "x4 ${S2_4}x, x8 ${S2_8}x on $NGPU GPU(s)"; fi
  FINER=no; le "$W1_8" "$(awk -v a="$W1_4" 'BEGIN{printf "%.4f", a+0.03}')" && FINER=yes
  info "CHUNKS_hint" "W1 x4 $W1_4, x8 $W1_8 -> finer chunks cost <= 0.03 more: $FINER (N=26: 2 chunks per GPU when yes, 1 per GPU when no; decided in 422). An N=25 4-GPU run would take 202.3 h x $( [[ $FINER == yes ]] && echo "$W1_8" || echo "$W1_4" ) / 4 = $(awk -v w="$( [[ $FINER == yes ]] && echo "$W1_8" || echo "$W1_4" )" 'BEGIN{printf "%.0f h", 202.3*w/4}')"
  rm -rf "$LOGDIR"/chunks_P23x4 "$LOGDIR"/chunks_P23x8; info "cleanup" "N=23 chunk files removed (2.5 GB; the results and logs stay)"
fi
if [[ "$FAIL" -eq 0 ]]; then info "ADOPTION" "421 is the production binary/.py (SASS identical to 419; production values unchanged); 421_multigpu_run.sh is the driver for N >= 25 (N=26 kernel + input: 422)"; else info "ADOPTION" "NOT adopted -- see FAIL lines; 419 stays"; fi
banner "Results"; python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import sys
rows=[l.rstrip('\n').split('\t') for l in open(sys.argv[1])]
w=[max(len(r[i]) for r in rows if i<len(r)) for i in range(len(rows[0]))]
for r in rows: print('  '.join(c.ljust(w[i]) for i,c in enumerate(r)))
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="chunks_*" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
