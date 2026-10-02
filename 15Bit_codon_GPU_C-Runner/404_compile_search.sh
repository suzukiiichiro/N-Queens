#!/usr/bin/env bash
# =====================================================================
# 404_compile_search.sh -- compile-only search + CPU equality (GPU time zero)
#
# NOTE (2026-10-02, from the session text):
#   403-r7 RESULT (2026-10-01 17:37) -- ADOPTED. S2 <402> == 402_r5, S3 <403>
#   sha db7e0ec2 / real 483 / loop 141, X7 results.bin == r6's X6, 4 runs
#   oracle + 1710 MHz. Z7/Z6 = 0.9922 (110,339.0 vs 111,203.2), X7 vs r6 X6
#   = 0.9917 (968,931.6 vs 977,058.5), G22 968,205.9. Production = 403_r7,
#   N=21 109,4xx unchanged, N=22 968,206 ms (16:08). Two push instructions
#   = -0.78..-0.83% -> ~0.4% per loop instruction (r5: 0.40%; r6's 0.71%
#   = ~0.4% count + ~0.3% chain depth).
#
#   Static read of the <402> hot loop (140): push 16 = pack 4, depth LIFO 3,
#   save_sp 1, stack_ptr 1, addresses 2, stores 2, MOV 2, BRA 1; pop 17 =
#   unpack 5, depth LIFO 3, save_sp 1, stack_ptr 1, addresses 2, loads 2,
#   branch 3. Less than half is the board itself. Candidates:
#   A  save_sp is always equal to stack_ptr (redundant since 402): drop it.
#      push -1, pop -1. README: no precedent.
#   B  cur_depth == popc(cur_col) - popc(root_col) (every step adds exactly
#      one bit to col): drop the 4-bit depth LIFO. push -3, pop -1, two
#      registers freed. README: no precedent. Caveat: after a pop the depth
#      chain grows from 1 to 4 ops (col 2 -> POPC -> SUB).
#   C  future check read from the depth bit of future_check_mask instead of
#      the re-materialised nibble (SHL+SHF+LOP3+ISETP -> 2). step -2.
#      PRECEDENT: rev242 tried exactly this (Codon JIT, one 7-minute run,
#      no SASS) and lost 1.3%. Included for the instruction count only.
#   Not touched: register MOVs (ptxas allocation; 395b lost 3.5% there) and
#   the push guard (closed axis).
#   Touching <402> ends "<402> SASS == 402_r5"; from here N=21 is guarded by
#   measurement and per-thread byte identity.
#
# Variants (base = 403_r7_kernel_maxd14.cu, both instantiations change):
#   d0 as is (control: <402> == 402_r5, <403> sha db7e0ec2c3261e09)
#   d1 A        d2 B        d3 A+B        d4 C        d5 A+B+C
# Per variant: gcc + nvcc, frame/regs, and for <402> and <403>: real
# instructions, hot loop, push region, pop region, rest of loop, sha;
# CPU per-record equality vs d0 (N=21 auto, N=21 NQ_LAYOUT=403, N=22 auto).
#
# Usage:  bash 404_compile_search.sh         (NREC=1024 records per CPU run)
#         SKIP_CPU=1 bash 404_compile_search.sh
# Needs:  403_r7_kernel_maxd14.cu, 402_r5_kernel_maxd14 (or its .cu),
#         N=21 / N=22 sched inputs (for the CPU equality)
# =====================================================================
set -u
REV="404"
BASE_CU="${BASE_CU:-403_r7_kernel_maxd14.cu}"
P402_BIN="${P402_BIN:-402_r5_kernel_maxd14}"
P402_CU="${P402_CU:-402_r5_kernel_maxd14.cu}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
NREC="${NREC:-1024}"
SKIP_NVCC="${SKIP_NVCC:-0}"
SKIP_CPU="${SKIP_CPU:-0}"
BASE_SHA403="${BASE_SHA403:-db7e0ec2c3261e09}"
VARIANTS="d0 d1 d2 d3 d4 d5"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_search_${TS}}"
mkdir -p "$LOGDIR/src" "$LOGDIR/sass" "$LOGDIR/bin" "$LOGDIR/loops"
[[ -x "$NVCC" ]] || NVCC="nvcc"; [[ -x "$CUOBJDUMP" ]] || CUOBJDUMP="cuobjdump"
[[ -f "$BASE_CU" ]] || { echo "FAIL  base source $BASE_CU not found"; exit 1; }

# ---- generate the variants ------------------------------------------
python3 - "$BASE_CU" "$LOGDIR/src" "$REV" <<'PYEOF' || { echo "FAIL  variant generation"; exit 1; }
import sys, re
base, outdir, rev = sys.argv[1:4]
src = open(base, encoding='utf-8').read()
def sub(s, pat, rep, n, what):
    out, k = re.subn(pat, rep, s, flags=re.M)
    if k != n: sys.exit("variant generation: %s matched %d times, expected %d" % (what, k, n))
    return out
def A(s):   # drop save_sp (always == stack_ptr)
    s = sub(s, r'^    uint32_t save_sp  = 0;\n', '', 1, 'save_sp decl')
    s = sub(s, r'^[ ]+save_sp   \+= 1u;\n', '', 2, 'save_sp += 1')
    s = sub(s, r'^([ ]+)if \(save_sp == 0u\) \{\n', r'\1if (stack_ptr == 0) {\n', 1, 'save_sp == 0')
    s = sub(s, r'^[ ]+save_sp -= 1u;\n', '', 1, 'save_sp -= 1')
    s = sub(s, r'\(long long\)debug_idx, stack_ptr, save_sp, cur_depth, cur_avail, terminal_depth,',
               r'(long long)debug_idx, stack_ptr, (unsigned)stack_ptr, cur_depth, cur_avail, terminal_depth,', 1, 'debug print')
    return s
def B(s):   # depth from popc(col); drop the 4-bit LIFO
    s = sub(s, r'^    uint64_t stack_depth = 0;\n',
            '    const int depth_base = (int)__builtin_popcount(root_col);\n', 1, 'stack_depth decl')
    s = sub(s, r'^[ ]+stack_depth = \(stack_depth << 4\) \| \(uint64_t\)\(uint32_t\)cur_depth;\n', '', 2, 'stack_depth push')
    s = sub(s, r'^([ ]+)cur_depth = \(int\)\(stack_depth & 15u\);\n[ ]+stack_depth >>= 4;\n',
            r'\1cur_depth = (int)__builtin_popcount(cur_col) - depth_base;\n', 1, 'stack_depth pop')
    return s
def C(s):   # future check from the depth bit of future_check_mask (rev242's form)
    s = sub(s, r'^([ ]+)if \(future_check_mask != 0u\) \{\n([ ]+)if \(fc_flag != 0u\) \{[^\n]*\n',
            r'\1{\n\2if (((future_check_mask >> cur_depth) & 1u) != 0u) {\n', 1, 'future check')
    return s
V = {"d0": [], "d1": [A], "d2": [B], "d3": [A, B], "d4": [C], "d5": [A, B, C]}
for name, fs in V.items():
    out = src
    for f in fs: out = f(out)
    if name == "d0" and out != src: sys.exit("d0 is not byte-identical to the base")
    open(f"{outdir}/{rev}_{name}_kernel_maxd14.cu", "w", encoding='utf-8').write(out)
print("generated:", ' '.join(V))
PYEOF

# ---- SASS reader -----------------------------------------------------
sass_read() { python3 - "$@" <<'PYEOF'
import sys, re, hashlib
path, frag, mode = sys.argv[1:4]
F = {}; cur = None
for line in open(path, encoding='utf-8', errors='replace'):
    m = re.match(r'\s*Function\s*:\s*(\S+)', line)
    if m: cur = m.group(1); F[cur] = []; continue
    if cur is None: continue
    m = re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s*(.*?);\s*(/\*.*\*/)?\s*$', line)
    if m: F[cur].append((int(m.group(1), 16), m.group(2).strip()))
k = [k for k in F if frag in k]
if len(k) != 1: print("CANNOT_LOCATE"); sys.exit(2)
I = F[k[0]]
if mode == "list": print('\n'.join(s for _, s in I)); sys.exit(0)
real = sum(1 for x in I if not x[1].startswith('NOP'))
sha = hashlib.sha256('\n'.join(s for _, s in I).encode()).hexdigest()[:16]
best = None   # hot loop = smallest backward-BRA span containing an LDL.64
for a, s in I:
    m = re.match(r'(@!?P\d\s+)?BRA\s+(0x[0-9a-f]+)$', s)
    if not m: continue
    t = int(m.group(2), 16)
    if t >= a: continue
    body = [x for x in I if t <= x[0] <= a]
    if any(x[1].startswith('LDL.64') for x in body) and (best is None or len(body) < len(best)): best = body
loop = len(best) if best else -1; push = pop = -1; pseg = []; qseg = []
if best:
    st = [i for i, x in enumerate(best) if x[1].startswith('STL.64')]
    if st:   # push = after the BSYNC before the first STL.64, through the next BRA
        j = st[0]
        while j > 0 and not best[j][1].startswith('BSYNC'): j -= 1
        e = st[0]
        while e < len(best) and not best[e][1].startswith('BRA'): e += 1
        pseg = best[j + 1:e + 1]; push = len(pseg)
    ld = [i for i, x in enumerate(best) if x[1].startswith('LDL')]
    if ld:   # pop = after the last unconditional BRA before the first LDL, up to the closing BSYNC
        j = ld[0]
        while j > 0 and not best[j][1].startswith('BRA'): j -= 1
        e = len(best) - 1
        while e > ld[-1] and not best[e][1].startswith('BSYNC'): e -= 1
        qseg = best[j + 1:e]; pop = len(qseg)
if mode == "stat": print(real, loop, push, pop, (loop - push - pop) if loop > 0 else -1, sha)
elif mode == "loop":
    print('\n'.join('%04x  %s' % x for x in (best or [])))
elif mode == "pp":
    print("  push:"); print('\n'.join('    ' + s for _, s in pseg)); print("  pop:"); print('\n'.join('    ' + s for _, s in qseg))
PYEOF
}

# ---- reference <402> --------------------------------------------------
REF402=""
if [[ "$SKIP_NVCC" != "1" ]]; then
  if [[ ! -x "$P402_BIN" && -f "$P402_CU" ]]; then "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$P402_BIN" "$P402_CU" -lcuda > "$LOGDIR/05_nvcc_402_r5.log" 2>&1; fi
  if [[ -x "$P402_BIN" ]]; then
    "$CUOBJDUMP" -sass "./$P402_BIN" > "$LOGDIR/sass/402_r5_kernel_maxd14.txt" 2>&1
    sass_read "$LOGDIR/sass/402_r5_kernel_maxd14.txt" "kernel_dfs_iter_gpu_maxd14PKj" list > "$LOGDIR/sass/ref402.list"; REF402="$LOGDIR/sass/ref402.list"
  else echo "INFO  402_r5 binary/source absent: the d0 control cannot check <402> == 402_r5"; fi
fi

# ---- CPU inputs --------------------------------------------------------
find_input() { local n="$1" c lg; for lg in $(ls -t 40*_crunner_logs/crunner_*_N${n}.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { echo "$c"; return; }; done; ls -t constellations_N${n}_*.sched394f.bin 2>/dev/null | head -1; }
IN21=""; IN22=""
if [[ "$SKIP_CPU" != "1" ]]; then
  IN21="$(find_input 21)"; IN22="$(find_input 22)"
  [[ -f "${IN21:-/nonexistent}" && -f "${IN22:-/nonexistent}" ]] && echo "INFO  CPU inputs: $IN21 / $IN22 ($NREC records each)" || { echo "INFO  CPU equality skipped: sched inputs not found (N=21 '$IN21', N=22 '$IN22')"; SKIP_CPU=1; }
fi
cpu_run() {  # variant label N input env
  env "$5" "$LOGDIR/bin/${REV}_${1}_cpu" "$3" "$4" "/tmp/${REV}_${1}_${2}.bin" "$NREC" > "$LOGDIR/2_cpu_${1}_${2}.log" 2>&1
}
cpu_eq() {   # variant -> "ok" | "DIFF[labels]" | "skipped"
  local v="$1" bad="" l
  [[ "$SKIP_CPU" == "1" ]] && { echo "skipped"; return; }
  cpu_run "$v" N21a 21 "$IN21" NQ_LAYOUT=auto; cpu_run "$v" N21l403 21 "$IN21" NQ_LAYOUT=403; cpu_run "$v" N22a 22 "$IN22" NQ_LAYOUT=auto
  [[ "$v" == "d0" ]] && { echo "ref"; return; }
  for l in N21a N21l403 N22a; do { [[ -s "/tmp/${REV}_${v}_${l}.bin" ]] && cmp -s "/tmp/${REV}_d0_${l}.bin" "/tmp/${REV}_${v}_${l}.bin"; } || bad="$bad,$l"; done
  [[ -z "$bad" ]] && echo "ok" || echo "DIFF[${bad#,}]"
}

TSV="$LOGDIR/${REV}_variants.tsv"
printf 'variant\tnvcc\tgcc\tcpu_eq\tframe\tregs\tinst\treal\tloop\tpush\tpop\trest\tsha\n' > "$TSV"
for v in $VARIANTS; do
  name="${REV}_${v}"; cu="$LOGDIR/src/${name}_kernel_maxd14.cu"; bin="$LOGDIR/bin/${name}_kernel_maxd14"
  G="ok"; "$GCC" -O2 -fopenmp -x c -o "$LOGDIR/bin/${name}_cpu" "$cu" -lm > "$LOGDIR/05_gcc_${name}.log" 2>&1 || G="FAIL"
  CE="-"; [[ "$G" == "ok" ]] && CE="$(cpu_eq "$v")"
  if [[ "$SKIP_NVCC" == "1" ]]; then printf '%s\tskipped\t%s\t%s\n' "$name" "$G" "$CE" >> "$TSV"; continue; fi
  NV="ok"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$bin" "$cu" -lcuda > "$LOGDIR/05_nvcc_${name}.log" 2>&1 || NV="FAIL"
  if [[ "$NV" != "ok" || ! -x "$bin" ]]; then
    printf '%s\tFAIL\t%s\t%s\n' "$name" "$G" "$CE" >> "$TSV"; echo "FAIL  $name nvcc: $(grep -i 'error' "$LOGDIR/05_nvcc_${name}.log" | head -2 | tr '\n' ' ')"; continue
  fi
  "$CUOBJDUMP" -sass "$bin" > "$LOGDIR/sass/${name}.txt" 2>&1
  for L in 402 403; do
    fr="$(grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${L}E" "$LOGDIR/05_nvcc_${name}.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"
    rg="$(grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${L}E" "$LOGDIR/05_nvcc_${name}.log" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"
    read -r NR LP PU PO RS SH <<< "$(sass_read "$LOGDIR/sass/${name}.txt" "kernel_dfs_iter_gpu_maxd14ILi${L}E" stat)"
    sass_read "$LOGDIR/sass/${name}.txt" "kernel_dfs_iter_gpu_maxd14ILi${L}E" loop > "$LOGDIR/loops/${name}.${L}.loop.txt"
    sass_read "$LOGDIR/sass/${name}.txt" "kernel_dfs_iter_gpu_maxd14ILi${L}E" pp   > "$LOGDIR/loops/${name}.${L}.pushpop.txt"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t<%s>\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$NV" "$G" "$CE" "${fr:-?}" "${rg:-?}" "$L" "$NR" "$LP" "$PU" "$PO" "$RS" "$SH" >> "$TSV"
    if [[ "$v" == "d0" && "$L" == "403" ]]; then
      [[ "$SH" == "$BASE_SHA403" ]] && echo "OK    d0 control: <403> sha = $SH (403_r7)" || echo "FAIL  d0 control: <403> sha $SH != $BASE_SHA403 -- toolchain or base file changed; read before trusting the rest"
    fi
    if [[ "$v" == "d0" && "$L" == "402" && -n "$REF402" ]]; then
      sass_read "$LOGDIR/sass/${name}.txt" "kernel_dfs_iter_gpu_maxd14ILi402E" list > "$LOGDIR/sass/${name}.402.list"
      cmp -s "$REF402" "$LOGDIR/sass/${name}.402.list" && echo "OK    d0 control: <402> == 402_r5" || echo "FAIL  d0 control: <402> differs from 402_r5"
    fi
  done
done

echo; echo "===== ${REV} compile search ====="
column -t -s $'\t' "$TSV" 2>/dev/null || cat "$TSV"
if [[ "$SKIP_NVCC" != "1" ]]; then
  echo; echo "expected d0: <402> real 482 loop 140 push 16 pop 17 rest 107 ; <403> real 483 loop 141 push 17 pop 17 rest 107"
  echo "aims       : d1 push -1 pop -1 ; d2 push -3 pop -1 ; d3 push -4 pop -2 ; d4 rest -2 ; d5 all ; cpu_eq ok ; frame 160"
  for v in $VARIANTS; do f="$LOGDIR/loops/${REV}_${v}.402.pushpop.txt"; [[ -s "$f" ]] && { echo; echo "--- <402> ${REV}_${v}"; cat "$f"; }; done
fi
tar czf "${LOGDIR}_tar.gz" --exclude="$LOGDIR/bin" --exclude="$LOGDIR/src" "$LOGDIR" && echo && echo "log: ${LOGDIR}_tar.gz ($(du -h "${LOGDIR}_tar.gz" | cut -f1); executables and variant sources excluded)"
