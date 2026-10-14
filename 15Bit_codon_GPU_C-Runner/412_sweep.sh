#!/usr/bin/env bash
# 412_sweep.sh
#
# rev412 -- BLOCK / MAX_BLOCKS re-sweep on the 411 (= 408) kernel. NO code change, NO adoption
#   in this harness. Production stays 411 (N=21 97,946 / N=22 849,096 ms, adopted 2026-10-08).
#
#   WHY: the "sharp minimum at 960" (402-r2: A800 129,513 / A880 117,122 / A960 112,001 /
#   A1040 116,081) was measured on the 402 kernel (loop 140, 40 regs). The kernel has since
#   lost 32 hot-loop instructions (108) and 3 registers (37/35), but the 160 B frame -- the
#   quantity that sets the L1 working set (12 blocks/SM x 32 thr x 160 B = 61,440 B) -- is
#   unchanged. Registers never capped occupancy (37 regs allow 52 warps). So the registered
#   expectation is that 960 is still the minimum and the ladder has the same shape.
#   BLOCK=64 at the same stride is the one open question: 400-r2 (N=19, frame 208) measured
#   -1.07% for BLOCK=64 at fixed stride, 401-r2 (N=21, MB=400) measured +6.17%. Neither was
#   on this kernel. BLOCK=128 at fixed stride was +6.54% (400-r2).
#
#   CELLS (all N=21 direct, 411 binary, helper 1 + 128 MB, NQ_EXTRA_CTX unset, auto -> 403):
#     A960      BLOCK 32  MB  960   stride 30,720  12 blk/SM   anchor (411 B1 97,936 / B0 97,821)
#     A880      BLOCK 32  MB  880   stride 28,160  11 blk/SM   402-r2: +4.6%
#     A1040     BLOCK 32  MB 1040   stride 33,280  13 blk/SM   402-r2: +3.6%
#     B64_480   BLOCK 64  MB  480   stride 30,720   6 blk/SM   same threads as A960, 2 warps/blk
#     B128_240  BLOCK 128 MB  240   stride 30,720   3 blk/SM   same threads, 4 warps/blk
#     A960r     BLOCK 32  MB  960   repeat of A960 (drift of the session)
#   HARD on every cell: oracle 314,666,222,712, layout=403, helpers=1, extra_ctx=0,
#     results.bin size == stride x 8, SM clock 1710 +-2%.
#   HARD on A960 / A960r: results.bin byte-identical to REF_BIN (411 B1, sha 1bfab090...).
#     Other strides map records to threads differently, so their results.bin is checked by
#     size + oracle only.
#   All comparisons use norm_ms = kernel_ms * sm_mean / 1710 (5 s samples, busy only).
#
#   PRE-REGISTERED (fixed before any cell runs):
#     W1 (stated)  A960 is the minimum of the BLOCK-32 ladder; A880 >= +3% and A1040 >= +2%.
#                  Refuted if either is < +1% -- then the L1 ceiling moved and 412-r2 sweeps
#                  finer (920 / 1000) before anything else.
#     W2 (no direction claimed)  B64_480 / A960: <= 0.998 -> candidate -> 412-r2 = .py-only
#                  revision changing the table's BLOCK (NQ_BLOCK=64 in env_prefix, MB 480)
#                  with an alternating -g A/B;  0.998..1.002 -> no effect, closed;
#                  >= 1.002 -> loses, closed (consistent with 401-r2).
#     W3 (stated)  B128_240 / A960 >= 1.02 (400-r2: +6.54%). Refuted if < 1.005.
#     W4           |A960r - A960| <= 0.15% (normalised). Larger -> read clk_*.tsv first.
#     W5           all cells oracle + 1710 +- 2%.
#   ADOPTION: none here. Production stays 411 whatever the numbers say.
#
# USAGE
#   STATIC_ONLY=1 bash 412_sweep.sh      # OK=7, GPU untouched
#                 bash 412_sweep.sh      # 6 cells, ~14 min
#   REF_BIN=path/to/B1_results.bin       # default 411_validate_20261006_150938/bins/B1_results.bin

set -u

REV="412"
CU_SRC="${CU_SRC:-411_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-411_kernel_maxd14}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
REF_BIN="${REF_BIN:-411_validate_20261006_150938/bins/B1_results.bin}"
REF_BIN_SHA="${REF_BIN_SHA:-1bfab090}"
SASS402_SHA="${SASS402_SHA:-821302f227dcd160}"; SASS402_REAL="${SASS402_REAL:-446}"; SASS402_LOOP="${SASS402_LOOP:-109}"
SASS403_SHA="${SASS403_SHA:-e940cb96e5e65fc4}"; SASS403_REAL="${SASS403_REAL:-445}"; SASS403_LOOP="${SASS403_LOOP:-108}"
KERNEL_SHA="${KERNEL_SHA:-ab483e70328c39c6}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
ARCH="${ARCH:-sm_86}"
HMB="${HMB:-128}"
COOLDOWN="${COOLDOWN:-15}"
CLK_INTERVAL="${CLK_INTERVAL:-5}"
STATIC_ONLY="${STATIC_ONLY:-0}"
ANCHOR_B1="${ANCHOR_B1:-97936.1}"     # 411 B1 (N=21 direct, flat 1710)
ORACLE21=314666222712
N=21

# cell  block  max_blocks
CELLS="${CELLS:-A960:32:960 A880:32:880 A1040:32:1040 B64_480:64:480 B128_240:128:240 A960r:32:960}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_sweep_${TS}}"
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
# 1. Static: the binary is 411 (= 408 SASS); reference results.bin present
# ---------------------------------------------------------------------
for f in "$CU_SRC" "$HELPER_SRC"; do
  [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"
done
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }
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
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" == "$KERNEL_SHA"* ]] && pass "cu_kernel_region_sha_is_408s (${KB:0:16})" || fail "cu_kernel_region_sha" "sha ${KB:0:16} != $KERNEL_SHA"
grep -q 'getenv("NQ_BLOCK")' "$CU_SRC" && pass "cu_reads_NQ_BLOCK" || fail "cu_reads_NQ_BLOCK" "no NQ_BLOCK knob in $CU_SRC"
grep -q 'getenv("NQ_MAX_BLOCKS")' "$CU_SRC" && pass "cu_reads_NQ_MAX_BLOCKS" || fail "cu_reads_NQ_MAX_BLOCKS" "no NQ_MAX_BLOCKS knob"
if [[ -f "$REF_BIN" ]]; then
  RS="$(sha256sum "$REF_BIN" | cut -c1-8)"; RZ="$(stat -c %s "$REF_BIN")"
  [[ "$RS" == "$REF_BIN_SHA" && "$RZ" == "245760" ]] && pass "ref_bin_is_411_B1 ($REF_BIN sha $RS, $RZ bytes)" || fail "ref_bin_is_411_B1" "sha $RS size $RZ (expected $REF_BIN_SHA / 245760)"
else fail "ref_bin_present" "$REF_BIN missing -- pass REF_BIN=... (411 B1 or any 960/32 results.bin of the 408 kernel)"; fi
ncells=0; for c in $CELLS; do IFS=: read -r cn cb cm <<< "$c"; case "$cb" in 32|64|128|256|512|1024) ;; *) fail "cell_block[$cn]" "BLOCK=$cb not allowed"; esac; ncells=$((ncells+1)); done
[[ "$ncells" -ge 2 ]] && pass "cell_list_ok ($ncells cells: $CELLS)" || fail "cell_list_ok" "need >= 2 cells"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Binary + SASS (must be 408's), input
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR/sass" "$LOGDIR/bins"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
[[ ! -x "$NVCC" ]] && NVCC="nvcc"; [[ ! -x "$CUOBJDUMP" ]] && CUOBJDUMP="cuobjdump"
if [[ ! -x "$CU_BIN" ]]; then banner "Building $CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc_${REV}.log" 2>&1; fi
[[ -x "$CU_BIN" ]] && pass "binary_available[$CU_BIN]" || { fail "binary_available" "cannot build $CU_BIN"; summary_exit; }
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
read -r S_SHA S_REAL S_LOOP S_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_dfs_iter_gpu_maxd14ILi402E)"
[[ "$S_SHA" == "$SASS402_SHA" && "$S_REAL" == "$SASS402_REAL" && "$S_LOOP" == "$SASS402_LOOP" ]] && pass "S2_sass_402_is_408s (sha $S_SHA, real $S_REAL, loop $S_LOOP)" || { fail "S2_sass_402" "sha $S_SHA real $S_REAL loop $S_LOOP != $SASS402_SHA / $SASS402_REAL / $SASS402_LOOP -- wrong binary, stop"; summary_exit; }
read -r S_SHA S_REAL S_LOOP S_TOT <<< "$(sass_stat "$LOGDIR/sass/${CU_BIN}.txt" kernel_dfs_iter_gpu_maxd14ILi403E)"
[[ "$S_SHA" == "$SASS403_SHA" && "$S_REAL" == "$SASS403_REAL" && "$S_LOOP" == "$SASS403_LOOP" ]] && pass "S3_sass_403_is_408s (sha $S_SHA, real $S_REAL, loop $S_LOOP)" || { fail "S3_sass_403" "sha $S_SHA real $S_REAL loop $S_LOOP != $SASS403_SHA / $SASS403_REAL / $SASS403_LOOP -- wrong binary, stop"; summary_exit; }

find_input() { local n="$1" c; for lg in $(ls -t 4*_crunner_logs/crunner_*_N${n}.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { echo "$c"; return; }; done; ls -t constellations_N${n}_*.sched394f.bin 2>/dev/null | head -1; }
IN21="$(find_input 21)"
[[ -f "${IN21:-/nonexistent}" ]] && pass "input21_located ($IN21)" || { fail "input21_located" "no N=21 sched input"; summary_exit; }

# ---------------------------------------------------------------------
# 3. Cells
# ---------------------------------------------------------------------
printf 'cell\tblock\tmax_blocks\tstride\tblk_per_sm\tlayout\thelpers\tfree_mb\tkernel_ms\tnorm_ms\ttotal_sum\tmatch\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
CLKPID=""
gpu_gate() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_before_$1.txt"; [[ -z "$apps" ]] && return 0; fail "gpu_empty_before[$1]" "compute process(es) present: $(echo "$apps" | tr '\n' ' ')"; return 1; }
gpu_after() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_after_$1.txt"; [[ -z "$apps" ]] || fail "gpu_residue_after[$1]" "$(echo "$apps" | tr '\n' ' ')"; }
clk_start() { local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"; ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPID=$!; }
clk_stop() { [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 && $6+0>50 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.1f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_$1.tsv"); }
declare -A VAL NORM SMM
run_cell() {  # cell block max_blocks
  local cell="$1" blk="$2" mb="$3"; gpu_gate "$cell" || return 1
  local stride=$((blk*mb)) bps; bps="$(awk -v m="$mb" 'BEGIN{printf "%.2f", m/80}')"
  local lg="$LOGDIR/3_${cell}.log" start; start="$(date -Is)"; clk_start "$cell"
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX NQ_LAYOUT=auto NQ_BLOCK="$blk" NQ_MAX_BLOCKS="$mb" NQ_HELPER_CTX=1 NQ_HELPER_MB="$HMB" "./$CU_BIN" "$N" "$IN21" "$LOGDIR/bins/${cell}_results.bin" "$ORACLE21" > "$lg" 2>&1 || true
  clk_stop "$cell"; gpu_after "$cell"
  local KMS TOT MATCH FREE HL LAYV CFG XC
  KMS="$(grep -o 'kernel_ms=[0-9.]*' "$lg" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$lg" && MATCH=1
  FREE="$(grep -o 'free_mb=[0-9]*' "$lg" | head -1 | cut -d= -f2)"; HL="$(grep -o 'helpers=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  LAYV="$(grep -o '\[gpu-layout\] N=[0-9]* requested=[a-z0-9]* layout=[0-9]*' "$lg" | head -1 | grep -o 'layout=[0-9]*' | cut -d= -f2)"
  CFG="$(grep -o '\[gpu-config\] BLOCK=[0-9]* MAX_BLOCKS=[0-9]* stride=[0-9]*' "$lg" | head -1)"
  XC="$(grep -o 'extra_ctx=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
  local NORMV; NORMV="$(awk -v k="${KMS:-0}" -v s="${SMMEAN/\?/0}" 'BEGIN{ if(s>0) printf "%.1f", k*s/1710; else print "?" }')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$blk" "$mb" "$stride" "$bps" "${LAYV:-?}" "${HL:-?}" "${FREE:-?}" "${KMS:-?}" "$NORMV" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start" >> "$TSV"
  [[ "$CFG" == "[gpu-config] BLOCK=$blk MAX_BLOCKS=$mb stride=$stride" ]] || { fail "config[$cell]" "'${CFG:-<none>}' != BLOCK=$blk MAX_BLOCKS=$mb stride=$stride"; return 1; }
  [[ "${LAYV:-}" == "403" ]] || { fail "layout[$cell]" "layout=${LAYV:-?}, expected 403"; return 1; }
  [[ "${HL:-}" == "1" && "${XC:-}" == "0" ]] || { fail "ctx_state[$cell]" "helpers=${HL:-?} extra_ctx=${XC:-?}, expected 1 / 0"; return 1; }
  [[ "$MATCH" == "1" && "${TOT:-}" == "$ORACLE21" ]] || { fail "oracle[$cell]" "total_sum='${TOT:-<none>}' match=$MATCH"; return 1; }
  local sz; sz="$(stat -c %s "$LOGDIR/bins/${cell}_results.bin" 2>/dev/null || echo 0)"
  [[ "$sz" == "$((stride*8))" ]] && pass "results_bin_size[$cell] ($sz = $stride x 8)" || { fail "results_bin_size[$cell]" "$sz != $((stride*8))"; return 1; }
  if [[ "$blk" == "32" && "$mb" == "960" ]]; then
    cmp -s "$REF_BIN" "$LOGDIR/bins/${cell}_results.bin" && pass "${cell}_per_thread_results_identical_to_411_B1" || fail "${cell}_per_thread_results_DIFFER" "results.bin != $REF_BIN -- same stride, same kernel: stop and read"
  fi
  VAL[$cell]="$KMS"; NORM[$cell]="$NORMV"; SMM[$cell]="$SMMEAN"
  info "$cell" "BLOCK=$blk MB=$mb ($bps blk/SM) kernel_ms=$KMS norm_ms=$NORMV (sm_mean $SMMEAN, min $SMMIN, ${TMAX}C) free_mb=$FREE"
}

banner "N=21 direct sweep on $CU_BIN: $CELLS"
first=1
for c in $CELLS; do
  IFS=: read -r cn cb cm <<< "$c"
  [[ "$first" == "1" ]] || sleep "$COOLDOWN"; first=0
  run_cell "$cn" "$cb" "$cm" || summary_exit
done

# ---------------------------------------------------------------------
# 4. Pre-registered readings (W1..W5)
# ---------------------------------------------------------------------
banner "Readings (normalised to 1710 MHz)"
A="${NORM[A960]:-}"
if [[ -n "$A" ]]; then
  info "A960_vs_411_B1" "raw $(pct "${VAL[A960]}" "$ANCHOR_B1")% vs $ANCHOR_B1 (same SASS, same stride: session scatter)"
  if [[ -n "${NORM[A880]:-}" && -n "${NORM[A1040]:-}" ]]; then
    d880="$(pct "${NORM[A880]}" "$A")"; d1040="$(pct "${NORM[A1040]}" "$A")"
    if ge "$d880" 3 && ge "$d1040" 2; then pass "W1_960_still_the_minimum (A880 +${d880}%, A1040 +${d1040}%)"
    elif ge "$d880" 1 && ge "$d1040" 1; then info "W1_grey" "A880 +${d880}%, A1040 +${d1040}% -- 960 still minimum but the walls are lower than 402-r2 (+4.6 / +3.6)"
    else fail "W1_REFUTED_ceiling_moved" "A880 +${d880}%, A1040 +${d1040}% (registered: both >= +1%) -- 412-r2 sweeps 920 / 1000"; fi
  fi
  if [[ -n "${NORM[B64_480]:-}" ]]; then
    r="$(ratio "${NORM[B64_480]}" "$A")"
    if le "$r" 0.998; then info "W2_B64_480_CANDIDATE" "B64_480/A960 = $r ($(pct "${NORM[B64_480]}" "$A")%) -> 412-r2: .py-only revision (NQ_BLOCK=64, MB 480) with alternating -g A/B"
    elif le "$r" 1.002; then info "W2_B64_480_no_effect" "B64_480/A960 = $r -- closed"
    else info "W2_B64_480_loses" "B64_480/A960 = $r (+$(pct "${NORM[B64_480]}" "$A")%) -- closed, consistent with 401-r2"; fi
  fi
  if [[ -n "${NORM[B128_240]:-}" ]]; then
    r="$(ratio "${NORM[B128_240]}" "$A")"
    if ge "$r" 1.02; then pass "W3_B128_240_slower_as_registered ($r)"
    elif ge "$r" 1.005; then info "W3_grey" "B128_240/A960 = $r (registered >= 1.02)"
    else fail "W3_REFUTED_B128_240_not_slower" "B128_240/A960 = $r (registered >= 1.02, refuted < 1.005) -- re-measure before reading"; fi
  fi
  if [[ -n "${NORM[A960r]:-}" ]]; then
    d="$(abspct "${NORM[A960r]}" "$A")"
    le "$d" 0.15 && pass "W4_A960_repeat_within_0.15pct (${d}%)" || fail "W4_A960_repeat_drift" "${d}% > 0.15% -- read clk_A960*.tsv before reading any other cell"
  fi
fi
CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "W5_sm_clock_1710_all_cells" || fail "W5_sm_clock_1710_all_cells" "see clk_*.tsv"
info "ADOPTION" "none in this harness -- production stays 411 (BLOCK 32 / MB 960). A W2 candidate goes to 412-r2."

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
base=next((float(r['norm_ms']) for r in rows if r['cell']=='A960' and r['norm_ms'] not in ('?','')), None)
print(f"{'cell':<9}{'blk':>4}{'MB':>6}{'stride':>8}{'blk/SM':>7}{'free_mb':>8}{'kernel_ms':>12}{'norm_ms':>12}{'vs A960':>9}{'match':>6}{'sm':>8}{'C':>4}{'m:ss':>8}")
for r in rows:
    v=float(r['kernel_ms']); nm=float(r['norm_ms']) if r['norm_ms'] not in ('?','') else v; s=v/1000
    d=f"{(nm/base-1)*100:+.3f}%" if base else "?"
    print(f"{r['cell']:<9}{r['block']:>4}{r['max_blocks']:>6}{r['stride']:>8}{r['blk_per_sm']:>7}{r['free_mb']:>8}{v:>12.1f}{nm:>12.1f}{d:>9}{r['match']:>6}{r['sm_mean']:>8}{r['temp_max']:>4}{int(s//60):>4}:{s%60:04.1f}")
print("(vs A960 = normalised; 411 B1 97,936.1 / B0 97,820.9 for the day-to-day scale)")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" --exclude="*_kernel_maxd14" --exclude="*_cpu" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
