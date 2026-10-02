#!/usr/bin/env bash
# =====================================================================
# 403_r7_compile_search.sh -- compile-only search (GPU time zero)
#
# NOTE (2026-10-01, from the session text):
#   Static read of the 403-r6 SASS, register-normalised, <402> vs <403>:
#   - "496 instructions in both" counted the tail alignment NOPs. Real
#     instructions: <402> 482, <403> r5 488, <403> r6 487. Hot loop:
#     <402> 140, r5 144, r6 143.
#   - The loop bodies agree section by section (53/53, 40/40, 10/10,
#     pop 17/17; only reorderings inside the mark block and one swap).
#     The whole +3 is in the push: 16 vs 19.
#       B = (rd & ~1) | (ld & 1)          -> 2 LOP3   (402: 0)
#       (ld >> 1) << 44                   -> SHF+SHL  (402: mask fused)
#   - r5 -> r6 was one loop instruction for -0.71% (N=21) / -0.68%
#     (N=22); the remaining 3 instructions carry +1.6%. One loop
#     instruction on the push/pop path ~ 0.5-0.7%.
#   This search looks for a spelling of the 403 push with 5 packing
#   instructions instead of 7 (loop 143 -> 141, root push -2 as well).
#   Push values do not feed the next iteration, so 403-r7 will also
#   separate "instruction count" from "dependency-chain depth".
#
# Variants (both push sites get the same replacement; pop and the 402
# branch are untouched; base = 403_r6_kernel_maxd14.cu):
#   w0  as is (control: <403> sha must be a8c1fffa5b1e84dc)
#   w1  B via inline PTX lop3.b32 0xD8 (select by mask 1)        A as is
#   w2  B as is     A high part as C: ((ld << 11) & 0xFFFFF000) << 32
#   w3  w1 + w2
#   w4  w1 + A high part via inline PTX lop3.b32 0xF8 (hi | (t & mask))
#   w5  w1 + A as 64-bit C: ((uint64_t)ld << 43) & 0xFFFFF00000000000
# Target: loop <= 141 with <402> identical to 402_r5.
#
# Usage:  bash 403_r7_compile_search.sh
# Needs:  403_r6_kernel_maxd14.cu, 402_r5_kernel_maxd14 (or its .cu)
# =====================================================================
set -u
REV="403_r7"
BASE_CU="${BASE_CU:-403_r6_kernel_maxd14.cu}"
P402_BIN="${P402_BIN:-402_r5_kernel_maxd14}"
P402_CU="${P402_CU:-402_r5_kernel_maxd14.cu}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
SKIP_NVCC="${SKIP_NVCC:-0}"     # 1 = generate + gcc only (no CUDA toolchain)
BASE_SHA403="${BASE_SHA403:-a8c1fffa5b1e84dc}"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_search_${TS}}"
mkdir -p "$LOGDIR/src" "$LOGDIR/sass" "$LOGDIR/bin"
[[ -x "$NVCC" ]] || NVCC="nvcc"; [[ -x "$CUOBJDUMP" ]] || CUOBJDUMP="cuobjdump"
[[ -f "$BASE_CU" ]] || { echo "FAIL  base source $BASE_CU not found"; exit 1; }

# ---- generate the variants ------------------------------------------
python3 - "$BASE_CU" "$LOGDIR/src" "$REV" <<'PYEOF' || { echo "FAIL  variant generation"; exit 1; }
import sys, re
base, outdir, rev = sys.argv[1:4]
src = open(base, encoding='utf-8').read()
pat = re.compile(
    r'^(?P<i>[ ]+)stack_a\[stack_ptr\] = \(uint64_t\)cur_col\n'
    r'[ ]+\| \(\(uint64_t\)cur_avail << 22\) \| \(\(uint64_t\)\(cur_ld >> 1\) << 44\);\n'
    r'[ ]+stack_b\[stack_ptr\] = \(cur_rd & ~1u\) \| \(cur_ld & 1u\);\n', re.M)
if len(pat.findall(src)) != 2:
    sys.exit("expected exactly 2 push sites of layout 403, found %d" % len(pat.findall(src)))
A_OLD = ["stack_a[stack_ptr] = (uint64_t)cur_col",
         "                    | ((uint64_t)cur_avail << 22) | ((uint64_t)(cur_ld >> 1) << 44);"]
B_OLD = ["stack_b[stack_ptr] = (cur_rd & ~1u) | (cur_ld & 1u);"]
B_ASM = ["{ uint32_t pb403;",
         "#ifdef __CUDACC__",
         "  asm(\"lop3.b32 %0, %1, %2, %3, 0xD8;\" : \"=r\"(pb403) : \"r\"(cur_rd), \"r\"(cur_ld), \"r\"(1u));",
         "#else",
         "  pb403 = (cur_rd & ~1u) | (cur_ld & 1u);",
         "#endif",
         "  stack_b[stack_ptr] = pb403; }"]
A_C32 = ["stack_a[stack_ptr] = (uint64_t)cur_col | ((uint64_t)cur_avail << 22)",
         "                    | ((uint64_t)((cur_ld << 11) & 0xFFFFF000u) << 32);"]
A_C64 = ["stack_a[stack_ptr] = (uint64_t)cur_col | ((uint64_t)cur_avail << 22)",
         "                    | (((uint64_t)cur_ld << 43) & 0xFFFFF00000000000ull);"]
A_ASM = ["{ const uint64_t pw403 = (uint64_t)cur_col | ((uint64_t)cur_avail << 22);",
         "  const uint32_t pl403 = cur_ld << 11;",
         "  uint32_t ph403;",
         "#ifdef __CUDACC__",
         "  asm(\"lop3.b32 %0, %1, %2, %3, 0xF8;\" : \"=r\"(ph403) : \"r\"((uint32_t)(pw403 >> 32)), \"r\"(pl403), \"r\"(0xFFFFF000u));",
         "#else",
         "  ph403 = (uint32_t)(pw403 >> 32) | (pl403 & 0xFFFFF000u);",
         "#endif",
         "  stack_a[stack_ptr] = ((uint64_t)ph403 << 32) | (uint64_t)(uint32_t)pw403; }"]
V = {"w0": (A_OLD, B_OLD), "w1": (A_OLD, B_ASM), "w2": (A_C32, B_OLD),
     "w3": (A_C32, B_ASM), "w4": (A_ASM, B_ASM), "w5": (A_C64, B_ASM)}
for name, (a, b) in V.items():
    def rep(m):
        i = m.group('i')
        return ''.join((i + l if not l.startswith('#') else l) + '\n' for l in a + b)
    out = pat.sub(rep, src)
    if name == "w0" and out != src: sys.exit("w0 is not byte-identical to the base")
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
if mode == "list":
    print('\n'.join(s for _, s in I)); sys.exit(0)
real = [x for x in I if not x[1].startswith('NOP')]
sha = hashlib.sha256('\n'.join(s for _, s in I).encode()).hexdigest()[:16]
# hot loop = the smallest backward-BRA span that contains an LDL.64
best = None
for a, s in I:
    m = re.match(r'(@!?P\d\s+)?BRA\s+(0x[0-9a-f]+)$', s)
    if not m: continue
    t = int(m.group(2), 16)
    if t >= a: continue
    body = [x for x in I if t <= x[0] <= a]
    if any(x[1].startswith('LDL.64') for x in body) and (best is None or len(body) < len(best)): best = body
loop = len(best) if best else -1
push = -1; ptxt = []
if best:
    st = [i for i, x in enumerate(best) if x[1].startswith('STL.64')]
    if st:
        j = st[0]
        while j > 0 and not best[j][1].startswith('BSYNC'): j -= 1
        e = st[0]
        while e < len(best) and not best[e][1].startswith('BRA'): e += 1
        seg = best[j + 1:e + 1]; push = len(seg); ptxt = [s for _, s in seg]
if mode == "stat": print(len(I), len(real), loop, push, sha)
elif mode == "push": print('\n'.join('    ' + s for s in ptxt))
PYEOF
}

# ---- reference <402> --------------------------------------------------
REF402=""
if [[ "$SKIP_NVCC" != "1" ]]; then
  if [[ ! -x "$P402_BIN" && -f "$P402_CU" ]]; then "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$P402_BIN" "$P402_CU" -lcuda > "$LOGDIR/05_nvcc_402_r5.log" 2>&1; fi
  if [[ -x "$P402_BIN" ]]; then
    "$CUOBJDUMP" -sass "./$P402_BIN" > "$LOGDIR/sass/402_r5_kernel_maxd14.txt" 2>&1
    sass_read "$LOGDIR/sass/402_r5_kernel_maxd14.txt" "kernel_dfs_iter_gpu_maxd14PKj" list > "$LOGDIR/sass/ref402.list"
    REF402="$LOGDIR/sass/ref402.list"
    echo "INFO  reference <402>: $P402_BIN  $(sass_read "$LOGDIR/sass/402_r5_kernel_maxd14.txt" "kernel_dfs_iter_gpu_maxd14PKj" stat)  (total real loop push sha)"
  else
    echo "INFO  402_r5 binary/source absent: <402> is compared against variant w0 instead"
  fi
fi

TSV="$LOGDIR/${REV}_variants.tsv"
printf 'variant\tnvcc\tgcc\tframe402/403\tregs402/403\tsass402\tn403_total\tn403_real\tloop403\tpush403\tsass403_sha\n' > "$TSV"
for v in w0 w1 w2 w3 w4 w5; do
  name="${REV}_${v}"; cu="$LOGDIR/src/${name}_kernel_maxd14.cu"; bin="$LOGDIR/bin/${name}_kernel_maxd14"
  G="ok"; "$GCC" -O2 -fopenmp -x c -o "$LOGDIR/bin/${name}_cpu" "$cu" -lm > "$LOGDIR/05_gcc_${name}.log" 2>&1 || G="FAIL"
  if [[ "$SKIP_NVCC" == "1" ]]; then printf '%s\tskipped\t%s\n' "$name" "$G" >> "$TSV"; continue; fi
  NV="ok"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$bin" "$cu" -lcuda > "$LOGDIR/05_nvcc_${name}.log" 2>&1 || NV="FAIL"
  if [[ "$NV" != "ok" || ! -x "$bin" ]]; then
    printf '%s\tFAIL\t%s\t-\t-\t-\t-\t-\t-\t-\t-\n' "$name" "$G" >> "$TSV"
    echo "FAIL  $name nvcc: $(grep -i 'error' "$LOGDIR/05_nvcc_${name}.log" | head -2 | tr '\n' ' ')"; continue
  fi
  fr() { grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${1}E" "$LOGDIR/05_nvcc_${name}.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1; }
  rg() { grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${1}E" "$LOGDIR/05_nvcc_${name}.log" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*'; }
  "$CUOBJDUMP" -sass "$bin" > "$LOGDIR/sass/${name}.txt" 2>&1
  sass_read "$LOGDIR/sass/${name}.txt" "kernel_dfs_iter_gpu_maxd14ILi402E" list > "$LOGDIR/sass/${name}.402.list"
  [[ -z "$REF402" && "$v" == "w0" ]] && REF402="$LOGDIR/sass/${name}.402.list"
  if cmp -s "$REF402" "$LOGDIR/sass/${name}.402.list"; then E="IDENTICAL"; else E="DIFFER"; fi
  read -r NT NR LP PU SH <<< "$(sass_read "$LOGDIR/sass/${name}.txt" "kernel_dfs_iter_gpu_maxd14ILi403E" stat)"
  sass_read "$LOGDIR/sass/${name}.txt" "kernel_dfs_iter_gpu_maxd14ILi403E" push > "$LOGDIR/push_${name}.txt"
  printf '%s\tok\t%s\t%s/%s\t%s/%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$G" "$(fr 402)" "$(fr 403)" "$(rg 402)" "$(rg 403)" "$E" "$NT" "$NR" "$LP" "$PU" "$SH" >> "$TSV"
  if [[ "$v" == "w0" ]]; then
    [[ "$SH" == "$BASE_SHA403" ]] && echo "OK    w0 control: <403> sha = $SH (403_r6)" || echo "FAIL  w0 control: <403> sha $SH != $BASE_SHA403 -- the toolchain or the base file changed; read before trusting the rest"
  fi
done

echo; echo "===== ${REV} compile search ====="
column -t -s $'\t' "$TSV" 2>/dev/null || cat "$TSV"
if [[ "$SKIP_NVCC" != "1" ]]; then
  echo; echo "reference: 403_r6 = w0 (real 487, loop 143, push 19 expected); <402> loop 140, push 16"
  echo "target   : sass402 IDENTICAL, frame 160, loop403 <= 141"
  for v in w0 w1 w2 w3 w4 w5; do f="$LOGDIR/push_${REV}_${v}.txt"; [[ -s "$f" ]] && { echo; echo "--- push region (hot loop) ${REV}_${v}"; cat "$f"; }; done
fi
tar czf "${LOGDIR}_tar.gz" --exclude="$LOGDIR/bin" "$LOGDIR" && echo && echo "log: ${LOGDIR}_tar.gz ($(du -h "${LOGDIR}_tar.gz" | cut -f1), executables excluded)"
