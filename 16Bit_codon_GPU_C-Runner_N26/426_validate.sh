#!/usr/bin/env bash
# 426_validate.sh -- PRODUCTION SWITCH: ./426Py_kernel_maxd16_final -g runs the queue search (NQ_LOGIC=queue in the table's
# env prefix). The kernel source is 425's (code region byte-identical); this harness gates the switch on the real -g path.
#
# WHERE 425 ENDED (2026-11-20, OK=67 FAIL=0): queue correct (slice T per record on the GPU, N=21 / N=22 oracle + per-thread sha,
#   slice B c881e856); Q21 87,039.8 ms (0.889 x production 97,946), Q22 761,678.8 ms (0.897 x production 849,096).
#
#   S0 static: .py notes, tags, table entry with NQ_LOGIC=queue, 3/3 vs 419Py (when found); .cu code region sha == 425's (c81c259f);
#      THIS DIRECTORY holds rev386_validation_helpers.py and the N=NMIN..NMAX CRunner inputs (the -g path would otherwise
#      start building inputs here)                                                                                         HARD
#   S1 ptxas: five instantiations as 419; queue 192 B / spill 0                                                           HARD
#   S2-S8 SASS of every kernel == the values recorded from the 425 binary (five 419 instantiations, queue, refill, mrv)   HARD
#   CB codon build of 426Py                                                                                               HARD
#   G  ./426Py_kernel_maxd16_final -g NMIN NMAX (default 21 23; about 2 h 10: N=21 87 s, N=22 13 min, N=23 ~1.9 h)
#      per N: crunner log has logic=queue and the oracle MATCH; dispatch.log names NQ_LOGIC=queue                          HARD
#      N=21 / N=22: /tmp/crunner_N<N>_results.bin sha 1bfab090 / 9a6ca49e (per-thread sums = 411 B1 / 424 Z22)             HARD
#   PRE-REGISTERED: G21 within +-3 % of 425's Q21 (87,039.8 ms); G22 within +-3 % of Q22 (761,678.8 ms);
#      G23 / production (7,717,284 ms, layout 423) in 0.86 .. 0.93. Refuted: G23 >= 0.97.
#   ADOPTION: every HARD -> 426 is the production .py / binary; production values for N=21..23 become this run's.
#      N=24 / N=25 keep their 415-r3 / 419-r3 values (as-built search) until they are next run.
#
# USAGE -- in the PRODUCTION directory (the one with the constellations_N2x_* inputs and rev386_validation_helpers.py):
#   STATIC_ONLY=1 bash 426_validate.sh
#   bash 426_validate.sh 2>&1 | tee 426_validate_stdout.log          (inside screen; detach with Ctrl+A then D -- NOT Ctrl+Z)
#   NMAX=22 ...  (15 min instead of 2 h 10)     SKIP_BUILD=1 (binaries already built)     TIMEOUT_DURATION=6h

set -u
REV="426"
SKIP_BUILD="${SKIP_BUILD:-0}"; STATIC_ONLY="${STATIC_ONLY:-0}"; NMIN="${NMIN:-21}"; NMAX="${NMAX:-23}"; TIMEOUT_DURATION="${TIMEOUT_DURATION:-6h}"
SEARCH_ROOT="${SEARCH_ROOT:-$HOME}"
PY_SRC="${PY_SRC:-426Py_kernel_maxd16_final.py}"; PY_BIN="${PY_BIN:-${PY_SRC%.py}}"
PREV_PY="${PREV_PY:-419Py_kernel_maxd16_final.py}"
CU_SRC="${CU_SRC:-426_kernel_maxd16.cu}"; CU_BIN="${CU_BIN:-426_kernel_maxd16}"
CRLOG="${CRLOG:-426_crunner_logs}"
KERNEL_SHA_419="3b67e26c23bf3454"; CODE_SHA_425="${CODE_SHA_425:-c81c259f0ad72b76}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"; CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"; CODON="${CODON:-codon}"; ARCH="${ARCH:-sm_86}"
CLK_INTERVAL="${CLK_INTERVAL:-5}"; THREADS="${THREADS:-30720}"; SASS_CHECK="${SASS_CHECK:-1}"
declare -A ORACLE=([21]=314666222712 [22]=2691008701644 [23]=24233937684440 [24]=227514171973736 [25]=2207893435808352)
declare -A PROD=([21]=97946 [22]=849096 [23]=7717284 [24]=73112528 [25]=728149056)
declare -A BINSHA=([21]="${BINSHA21:-1bfab0900918dd44}" [22]="${BINSHA22:-9a6ca49e165bde40}")
declare -A STREAM=([21]="constellations_N21_6.bin" [22]="constellations_N22_7.bin" [23]="constellations_N23_7.bin" [24]="constellations_N24_7.bin" [25]="constellations_N25_7.bin")
Q21_MS=87039.8; Q22_MS=761678.8
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
for f in "$PY_SRC" "$CU_SRC"; do [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"; done
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }
where() { find "$SEARCH_ROOT" -name "$1" -size +0 2>/dev/null | head -1 | xargs -r dirname; }
if [[ -s rev386_validation_helpers.py ]]; then pass "helper_module_here[rev386_validation_helpers.py]"; else fail "helper_module_here" "rev386_validation_helpers.py is not in this directory (found in: $(where rev386_validation_helpers.py)) -- run 426 in the production directory: copy $PY_SRC, $CU_SRC and this script there"; fi
for n in $(seq "$NMIN" "$NMAX"); do
  s="${STREAM[$n]:-}"; in="$(ls -t ${s}.soa_ref_361.bin.*sched394f.bin 2>/dev/null | head -1)"
  if [[ -n "$s" && -s "$s" && -n "$in" ]]; then pass "inputs_here[N=$n] $s + $in ($(( $(stat -c %s "$in") / 28 )) records)"
  else fail "inputs_here[N=$n]" "need ${s:-?} and ${s:-?}.soa_ref_361.bin.*sched394f.bin in THIS directory (found in: $(where "${s}.soa_ref_361.bin.*sched394f.bin")) -- otherwise -g starts building the N=$n input here"; fi
done
[[ -n "${ORACLE[$NMAX]:-}" && "$NMIN" -ge 21 && "$NMAX" -le 25 && "$NMIN" -le "$NMAX" ]] && pass "range_N${NMIN}_N${NMAX}_supported" || fail "range" "NMIN=$NMIN NMAX=$NMAX must lie in 21..25 (the queue kernel is layout 425, N <= 25; -g must never reach N=26)"
py_code_region() { python3 -c "
import sys
lines=open(sys.argv[1],encoding='utf-8').read().split('\n')
i=next(i for i,l in enumerate(lines) if l.startswith('import '))
sys.stdout.write('\n'.join(l for l in lines[i:] if not l.lstrip().startswith('#')))
" "$1"; }
py_code_region "$PY_SRC" > "/tmp/${REV}_code_only.py"; CODE="/tmp/${REV}_code_only.py"
NOTE_LINES=$(awk '/^# =+$/{f=1} f&&/^#/{n++} END{print n+0}' "$PY_SRC")
{ grep -q "^# 426 " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 426 ...' note block (rule 8)"
grep -qE '^REV_TAG:str="426"' "$CODE" && pass "source_rev_tag_is_426" || fail "source_rev_tag_is_426" "wrong REV_TAG"
grep -qF 'CRunnerEntry(17,"./426_kernel_maxd16","NQ_LOGIC=queue NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_entry_maxd17_426_binary_NQ_LOGIC_queue_helper_prefix" || fail "source_table_entry" "table entry wrong"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=960\b" "$CODE" && pass "source_default_max_blocks_is_960" || fail "source_default_max_blocks_is_960" "default not 960"
[[ -f "$PREV_PY" ]] || PREV_PY="$(find "$SEARCH_ROOT" -name 419Py_kernel_maxd16_final.py 2>/dev/null | head -1)"
if [[ -n "$PREV_PY" && -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_419Py (removed=3 added=3 EXECUTABLE lines; $PREV_PY)" || fail "py_diff_fingerprint_vs_419Py" "removed=$PR_ added=$PA_, expected 3/3"
else info "py_diff_fingerprint_vs_419Py" "skipped (419Py not found)"; fi
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
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA_419"* ]] && pass "cu_kernel_region_sha_is_419s (${KB:0:16})" || fail "cu_kernel_region_sha" "sha ${KB:0:16} != $KERNEL_SHA_419"
CS="$(sha256sum "/tmp/${REV}_cur_code.cu" | cut -c1-16)"
[[ "$CS" == "$CODE_SHA_425" ]] && pass "cu_code_region_is_425s (sha $CS: every executable line is 425-r2's, comments aside)" || fail "cu_code_region_is_425s" "sha $CS != $CODE_SHA_425 -- the source differs from the one 425 validated"
grep -q "rev426 " "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev426 note"
FREE_GB="$(df -BG . | awk 'NR==2{gsub("G","",$4); print $4}')"; [[ "${FREE_GB:-0}" -ge 5 ]] && pass "disk_free_${FREE_GB}G" || fail "disk" "only ${FREE_GB} GB free"
if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Build, ptxas (S1), SASS (S2-S8), codon (CB)
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR/sass" "$LOGDIR/bins" "$CRLOG"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; df -h . 2>&1; free -g 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
[[ ! -x "$NVCC" ]] && NVCC="nvcc"; [[ ! -x "$CUOBJDUMP" ]] && CUOBJDUMP="cuobjdump"
banner "Building"
if [[ "$SKIP_BUILD" == "1" ]]; then
  [[ -x "$CU_BIN" && -x "$PY_BIN" ]] && pass "existing_binaries_used[$CU_BIN, $PY_BIN] (SKIP_BUILD=1)" || { fail "binaries_missing" "$CU_BIN / $PY_BIN absent and SKIP_BUILD=1"; summary_exit; }
else
  rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc_${REV}.log" 2>&1
  [[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "$(grep -i error "$LOGDIR/05_nvcc_${REV}.log" | head -3)"; summary_exit; }
  ptx_stats() { FR="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"; SP="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o '[0-9]* bytes spill stores, [0-9]* bytes spill loads' | head -1)"; RG="$(grep -A3 "$1" "$LOGDIR/05_nvcc_${REV}.log" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"; }
  for pair in 402:160 403:160 423:176 424:184 425:192; do IFS=: read -r L want <<< "$pair"; ptx_stats "kernel_dfs_iter_gpu_maxd16ILi${L}E"
    [[ "$FR" == "$want" && "$SP" == "0 bytes spill stores, 0 bytes spill loads" && "${RG:-99}" -le 44 ]] && pass "S1_ptxas_L${L}_frame${want}_spill0_regs${RG}" || fail "S1_ptxas_L${L}" "frame=${FR}B regs=${RG} ${SP} (expected $want / 0 / <= 44)"; done
  ptx_stats "kernel_queue_gpu"
  [[ "$FR" == "192" && "$SP" == "0 bytes spill stores, 0 bytes spill loads" ]] && pass "S1_ptxas_queue_frame192_spill0_regs${RG} (425: 192 / 47)" || { fail "S1_ptxas_queue" "frame=${FR:-?}B regs=${RG:-?} ${SP:-no ptxas line}"; summary_exit; }
fi
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
if [[ "$SASS_CHECK" == "1" ]]; then
for trip in "kernel_dfs_iter_gpu_maxd16ILi402E:821302f227dcd160:446:S2_402" "kernel_dfs_iter_gpu_maxd16ILi403E:e940cb96e5e65fc4:445:S3_403" "kernel_dfs_iter_gpu_maxd16ILi423E:c8316e183df391d7:447:S4_423" "kernel_dfs_iter_gpu_maxd16ILi424E:f60ecee7eab9450b:447:S5_424" "kernel_dfs_iter_gpu_maxd16ILi425E:4e3b9e8f0366db1a:454:S6_425" "kernel_queue_gpu:66d7b91957692186:355:S8_queue"; do IFS=: read -r fn want wreal tag <<< "$trip"
  read -r N_SHA N_REAL N_LOOP N_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" "$fn")"
  [[ "$N_SHA" == "$want" && "$N_REAL" == "$wreal" ]] && pass "${tag}_sass_is_425s (sha $N_SHA, real $N_REAL)" || { fail "${tag}_sass_differs_from_425" "sha $N_SHA real $N_REAL, recorded $want / $wreal -- not the binary 425 validated: stop"; summary_exit; }
done
for trip in "kernel_refill_gpu:1fa851e7544fbab9:S7_refill" "kernel_mrv_gpu:500eb8a37ec40f49:S7_mrv"; do IFS=: read -r fn want tag <<< "$trip"
  read -r N_SHA N_REAL N_LOOP N_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" "$fn" LDL)"; [[ "$N_SHA" == "$want" ]] && pass "${tag}_sass_is_425s (sha $N_SHA)" || info "${tag}_sass" "sha $N_SHA, recorded $want (not on the -g path)"; done
fi
if [[ "$SKIP_BUILD" != "1" ]]; then rm -f "$PY_BIN"; "$CODON" build -release "$PY_SRC" > "$LOGDIR/06_codon_${REV}.log" 2>&1
  [[ -x "$PY_BIN" ]] && pass "CB_codon_build[$PY_BIN]" || { fail "CB_codon_build" "$(tail -3 "$LOGDIR/06_codon_${REV}.log")"; summary_exit; }; fi

# ---------------------------------------------------------------------
# 3. G: the production path
# ---------------------------------------------------------------------
apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"
[[ -z "$apps" ]] && pass "gpu_empty_before" || { fail "gpu_empty_before" "compute process(es) present: $apps"; summary_exit; }
banner "G: ./$PY_BIN -g $NMIN $NMAX  (timeout $TIMEOUT_DURATION)"
dlog="$CRLOG/dispatch.log"; before=0; [[ -f "$dlog" ]] && before="$(wc -l < "$dlog")"
for n in $(seq "$NMIN" "$NMAX"); do rm -f "$CRLOG/crunner_${CU_BIN}_N${n}.log" "/tmp/crunner_N${n}_results.bin"; done
echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$LOGDIR/clk_G.tsv"
( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$LOGDIR/clk_G.tsv" 2>/dev/null & CLKPID=$!
START="$(date -Is)"; info "G_start" "$START"
if command -v script >/dev/null 2>&1; then
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX -u NQ_LAYOUT -u NQ_LOGIC -u NQ_REC_OUT -u NQ_HELPER_CTX -u NQ_HELPER_MB -u NQ_MAX_BLOCKS -u NQ_BLOCK \
    script -qfec "timeout --signal=TERM --kill-after=60s $TIMEOUT_DURATION ./$PY_BIN -g $NMIN $NMAX" "$LOGDIR/3_G.stdout"; rc=$?
else
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX -u NQ_LAYOUT -u NQ_LOGIC -u NQ_REC_OUT -u NQ_HELPER_CTX -u NQ_HELPER_MB -u NQ_MAX_BLOCKS -u NQ_BLOCK \
    timeout --signal=TERM --kill-after=60s "$TIMEOUT_DURATION" "./$PY_BIN" -g "$NMIN" "$NMAX" 2>&1 | tee "$LOGDIR/3_G.stdout"; rc=${PIPESTATUS[0]}
fi
kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true
info "G_end" "$(date -Is) rc=$rc"
[[ -f "$dlog" ]] && tail -n +"$((before+1))" "$dlog" > "$LOGDIR/3_G_dispatch.log"
apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; [[ -z "$apps" ]] && pass "gpu_residue_none" || fail "gpu_residue" "$apps"
[[ "$rc" == "0" ]] && pass "G_rc0" || fail "G_rc" "rc=$rc ($( [[ "$rc" == "124" ]] && echo timeout || tail -2 "$LOGDIR/3_G.stdout" | tr -d '\r' | tr '\n' ' '))"
printf 'cell\tN\tlogic\trecords\tkernel_ms\ttotal_sum\tproduction_ms\tratio_vs_production\tresults_sha\tstart\n' > "$TSV"
declare -A GMS
for n in $(seq "$NMIN" "$NMAX"); do
  src="$CRLOG/crunner_${CU_BIN}_N${n}.log"; LG="$LOGDIR/3_G${n}.log"
  if [[ ! -f "$src" ]]; then fail "G${n}_log" "no crunner log at $src (the -g run did not reach N=$n)"; continue; fi
  cp "$src" "$LG"; cp -f "/tmp/crunner_N${n}_results.bin" "$LOGDIR/bins/G${n}_results.bin" 2>/dev/null || true
  KMS="$(grep -o 'kernel_ms=[0-9.]*' "$LG" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$LG" | head -1 | cut -d= -f2)"; REC="$(grep -o 'records=[0-9]*' "$LG" | head -1 | cut -d= -f2)"
  LOGV="$(grep -o '\[gpu-logic\] N=[0-9]* requested=[a-z0-9]* logic=[a-z]*' "$LG" | head -1 | grep -o 'logic=[a-z]*' | cut -d= -f2)"
  SHA="$(sha256sum "$LOGDIR/bins/G${n}_results.bin" 2>/dev/null | cut -c1-16)"; GMS[$n]="${KMS:-}"
  R="$(awk -v a="${KMS:-0}" -v b="${PROD[$n]}" 'BEGIN{printf "%.4f", a/b}')"
  printf 'G%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$n" "${LOGV:-?}" "${REC:-?}" "${KMS:-?}" "${TOT:-?}" "${PROD[$n]}" "$R" "${SHA:-?}" "$START" >> "$TSV"
  [[ "${LOGV:-}" == "queue" ]] && pass "G${n}_logic_is_queue" || fail "G${n}_logic" "logic=${LOGV:-?} -- the -g path did not select the queue search"
  grep -q "N=${n} .*NQ_LOGIC=queue" "$LOGDIR/3_G_dispatch.log" 2>/dev/null && pass "G${n}_dispatch_names_NQ_LOGIC_queue" || fail "G${n}_dispatch" "$(grep "N=${n} " "$LOGDIR/3_G_dispatch.log" 2>/dev/null | head -1 | cut -c1-200)"
  grep -q '\[gpu-config\] BLOCK=32 MAX_BLOCKS=960' "$LG" && grep -q 'helpers=1' "$LG" && pass "G${n}_config_32x960_helper1" || fail "G${n}_config" "$(grep -o '\[gpu-config\].*' "$LG" | head -1)"
  grep -q "claims=$(( ${REC:-0} + THREADS )) " "$LG" && pass "G${n}_queue_drained (claims = records + $THREADS)" || fail "G${n}_claims" "$(grep -o '\[gpu-queue\] claims.*' "$LG" | head -1)"
  if grep -q '\[gpu-run-correctness\] MATCH' "$LG" && [[ "$TOT" == "${ORACLE[$n]}" ]]; then pass "G${n}_oracle_MATCH total_sum=$TOT"; else fail "G${n}_oracle" "total_sum=${TOT:-?} expected ${ORACLE[$n]}"; fi
  if [[ -n "${BINSHA[$n]:-}" ]]; then [[ "$SHA" == "${BINSHA[$n]}" ]] && pass "G${n}_per_thread_results_sha_${SHA:0:8} (the same 30,720 sums as the as-built kernel)" || fail "G${n}_per_thread_results" "sha ${SHA:-?} != ${BINSHA[$n]}"; else info "G${n}_results_sha" "${SHA:-?} (no earlier per-thread reference at N=$n: this becomes it)"; fi
  info "G${n}_PRODUCTION_VALUE" "N=$n kernel_ms=$KMS ($(awk -v k="${KMS:-0}" 'BEGIN{printf "%d:%02d:%04.1f", int(k/3600000), int(k/60000)%60, (k/1000)%60}')) = $R x the as-built production value ${PROD[$n]} ms"
done
band() { awk -v x="$1" -v lo="$2" -v hi="$3" 'BEGIN{exit !(x>=lo && x<=hi)}'; }
[[ -n "${GMS[21]:-}" ]] && { r="$(awk -v a="${GMS[21]}" -v b="$Q21_MS" 'BEGIN{printf "%.4f", a/b}')"; band "$r" 0.97 1.03 && pass "P_G21_within_3pct_of_Q21 ($r)" || info "P_G21_outside_statement" "G21 / Q21 = $r"; }
[[ -n "${GMS[22]:-}" ]] && { r="$(awk -v a="${GMS[22]}" -v b="$Q22_MS" 'BEGIN{printf "%.4f", a/b}')"; band "$r" 0.97 1.03 && pass "P_G22_within_3pct_of_Q22 ($r)" || info "P_G22_outside_statement" "G22 / Q22 = $r"; }
[[ -n "${GMS[23]:-}" ]] && { r="$(awk -v a="${GMS[23]}" -v b="${PROD[23]}" 'BEGIN{printf "%.4f", a/b}')"; if band "$r" 0.97 9; then fail "P_G23_REFUTED_not_faster_than_production" "G23 / production = $r >= 0.97"; elif band "$r" 0.86 0.93; then pass "P_G23_over_production_in_0.86_0.93 ($r)"; else info "P_G23_outside_statement" "G23 / production = $r, stated 0.86 .. 0.93"; fi; }
info "power[G]" "$(python3 - "$LOGDIR/clk_G.tsv" <<'PYEOF'
import sys
p=[]
for l in open(sys.argv[1]).read().split('\n')[1:]:
    c=l.split(',')
    try:
        if len(c)>=6 and float(c[5])>50: p.append((float(c[3]),float(c[1])))
    except ValueError: pass
if len(p)<4: print("too short")
else:
    lo=sum(1 for w,_ in p if w<200); print(f"{len(p)*5} s under load: mean {sum(w for w,_ in p)/len(p):.0f} W, {lo*5} s below 200 W ({lo/len(p)*100:.1f} %), SM clock mean {sum(c for _,c in p)/len(p):.0f} MHz, min {min(c for _,c in p):.0f}")
PYEOF
)"
echo "----- the -g table -----"; tr -d '\r' < "$LOGDIR/3_G.stdout" | grep -vE '^Script (started|done)'
if [[ "$FAIL" -eq 0 ]]; then info "ADOPTION" "426 is the production .py / binary: -g runs the queue search. Production values N=${NMIN}..${NMAX} are the G lines above; N above $NMAX keep their as-built values until next run"; else info "ADOPTION" "NOT adopted -- see FAIL lines (419Py / 419 binary remain production)"; fi
banner "Results"; python3 - "$TSV" <<'PYEOF' | tee "$LOGDIR/9_ranked.txt"
import sys
rows=[l.rstrip('\n').split('\t') for l in open(sys.argv[1])]
w=[max(len(r[i]) for r in rows if i<len(r)) for i in range(len(rows[0]))]
for r in rows: print('  '.join(c.ljust(w[i]) for i,c in enumerate(r)))
PYEOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
