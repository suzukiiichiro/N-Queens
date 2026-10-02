#!/usr/bin/env bash
# 403_r6_compile_search.sh -- GPU time ZERO. Spelling search for the pop-side
# ld reconstruction of the 403b layout.
#
# 403-r5's pop (SASS of the <403> instantiation, 496 instructions):
#     SHF.R.U32.HI R2, RZ, 0xb, R3          ; a = hi >> 11  (= packed_a >> 43)
#     LOP3.LUT     R2, R2, 0x1ffffe, RZ     ; a & ~1   <-- separate op
#     LOP3.LUT     R5, R2, 0x1, R7, 0xf8    ; | (B & 1)
# i.e. ld is 3 dependent ops after LDL.64 while cur_avail is 2, so ld has
# become the longest pop chain (402: ld 1 op). (a & ~1) | (b & 1) is ONE
# 3-input LOP3 (immLut 0xD8 with c = 1), and ptxas did not fold it.
#
# This script builds V variants of the ONE pop line from 403_r5's .cu, and
# for each reports: <402> SASS identical to 402_r5 (must hold -- the line
# is inside the 403 branch, `if constexpr`), <403> instruction count, the
# pop region (LDL .. loop back) with the op count after the loads, gcc CPU
# build. No binary is run. Pick the variant with the shortest ld chain
# (target: SHF + one LOP3) and the smallest count; that becomes 403-r6.
#
#   v0  r5 as is:            ((uint32_t)(packed_a >> 43) & ~1u) | (packed_b & 1u)
#   v1  xor-select:          a43 ^ ((a43 ^ packed_b) & 1u)
#   v2  explicit mask:       ((uint32_t)(packed_a >> 43) & 0xfffffffeu) | (packed_b & 1u)
#   v3  shift-back:          ((uint32_t)(packed_a >> 44) << 1) | (packed_b & 1u)
#   v4  select in 64-bit:    (uint32_t)(((packed_a >> 43) & ~1ull) | (uint64_t)(packed_b & 1u))
#   v5  inline PTX lop3:     lop3.b32 %0, a43, packed_b, 1, 0xD8   (C fallback = v0 for gcc)
#   v6  mask via variable:   m = 1u; (a43 & ~m) | (packed_b & m)
#
# USAGE:  bash 403_r6_compile_search.sh        # ~5 min, writes 403_r6_search_<ts>/

set -u
BASE_CU="${BASE_CU:-403_r5_kernel_maxd14.cu}"
P402_BIN="${P402_BIN:-402_r5_kernel_maxd14}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"; [[ -x "$NVCC" ]] || NVCC="nvcc"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"; [[ -x "$CUOBJDUMP" ]] || CUOBJDUMP="cuobjdump"
GCC="${GCC:-gcc}"; ARCH="${ARCH:-sm_86}"
TS="$(date +%Y%m%d_%H%M%S)"; OUT="${OUT:-403_r6_search_${TS}}"; mkdir -p "$OUT/sass" "$OUT/src"
[[ -f "$BASE_CU" ]] || { echo "missing $BASE_CU"; exit 1; }
OLD='                cur_ld  = ((uint32_t)(packed_a >> 43) & ~1u) | (packed_b & 1u);'
grep -qF "$OLD" "$BASE_CU" || { echo "pop line not found in $BASE_CU"; exit 1; }
[[ "$(grep -cF "$OLD" "$BASE_CU")" == "1" ]] || { echo "pop line not unique"; exit 1; }

python3 - "$BASE_CU" "$OUT/src" <<'PYEOF'
import sys
base, outdir = sys.argv[1], sys.argv[2]
src = open(base, encoding='utf-8').read()
OLD = '                cur_ld  = ((uint32_t)(packed_a >> 43) & ~1u) | (packed_b & 1u);\n'
I = '                '
V = {
 'v0': OLD,
 'v1': I+'{ const uint32_t a43 = (uint32_t)(packed_a >> 43);\n'+I+'  cur_ld  = a43 ^ ((a43 ^ packed_b) & 1u); }\n',
 'v2': I+'cur_ld  = ((uint32_t)(packed_a >> 43) & 0xfffffffeu) | (packed_b & 1u);\n',
 'v3': I+'cur_ld  = ((uint32_t)(packed_a >> 44) << 1) | (packed_b & 1u);\n',
 'v4': I+'cur_ld  = (uint32_t)(((packed_a >> 43) & ~1ull) | (uint64_t)(packed_b & 1u));\n',
 'v5': I+'{ const uint32_t a43 = (uint32_t)(packed_a >> 43);\n'
       + '#ifdef __CUDACC__\n'
       + I+'  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(cur_ld) : "r"(a43), "r"(packed_b), "r"(1u));\n'
       + '#else\n'
       + I+'  cur_ld  = (a43 & ~1u) | (packed_b & 1u);\n'
       + '#endif\n'
       + I+'}\n',
 'v6': I+'{ const uint32_t a43 = (uint32_t)(packed_a >> 43); const uint32_t m = 1u;\n'+I+'  cur_ld  = (a43 & ~m) | (packed_b & m); }\n',
}
assert src.count(OLD) == 1
for k, rep in V.items():
    open(f'{outdir}/403_r6_{k}_kernel_maxd14.cu', 'w', encoding='utf-8').write(src.replace(OLD, rep))
print('variants:', ' '.join(V))
PYEOF

sass_tool() { python3 - "$@" <<'PYEOF'
import sys, re, hashlib
mode = sys.argv[1]
def funcs(path):
    out = {}; cur = None
    for line in open(path, encoding='utf-8', errors='replace'):
        m = re.match(r'\s*Function\s*:\s*(\S+)', line)
        if m: cur = m.group(1); out[cur] = []; continue
        if cur is None: continue
        m = re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s*(.*?);\s*(/\*.*\*/)?\s*$', line)
        if m: out[cur].append(m.group(2).strip())
    return out
if mode == "cmp":
    fa, fragA, fb, fragB = sys.argv[2:6]
    A = funcs(fa); B = funcs(fb)
    ka = [k for k in A if fragA in k]; kb = [k for k in B if fragB in k]
    if len(ka) != 1 or len(kb) != 1: print("CANNOT_LOCATE"); sys.exit(2)
    print("IDENTICAL" if A[ka[0]] == B[kb[0]] else "DIFFER"); sys.exit(0)
if mode == "pop":   # count + pop region of the 403 instantiation
    fa = sys.argv[2]; A = funcs(fa); k = [k for k in A if 'ILi403E' in k][0]; ins = A[k]
    ldl = [i for i, x in enumerate(ins) if x.startswith('LDL')]
    if not ldl: print(f"{len(ins)}\t?\t?"); sys.exit(0)
    s = ldl[0]; e = s
    while e < len(ins) and not ins[e].startswith('BSYNC'): e += 1
    region = ins[s:e]
    # ops after the loads, excluding the ones that only touch stack_depth / counters (heuristic: keep all, report raw)
    post = [x for x in region if not x.startswith('LDL')]
    # ld chain: the last LOP3/SHF producing the register later used with 0x1 select is hard to name; print region
    print(f"{len(ins)}\t{len(post)}\t{hashlib.sha256(chr(10).join(ins).encode()).hexdigest()[:16]}")
    for x in ins[max(0,s-2):e+1]: print("   ", x)
PYEOF
}

[[ -x "$P402_BIN" ]] && "$CUOBJDUMP" -sass "./$P402_BIN" > "$OUT/sass/${P402_BIN}.txt" 2>&1
printf 'variant\tnvcc\tgcc\tframe402/403\tregs402/403\tsass402_eq_402_r5\tn403\tpop_ops_after_LDL\tsass403_sha\n' > "$OUT/403_r6_variants.tsv"
for cu in "$OUT"/src/403_r6_v*_kernel_maxd14.cu; do
  v="$(basename "$cu" _kernel_maxd14.cu)"; bin="$OUT/${v}_kernel_maxd14"
  echo "=== $v"
  "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$bin" "$cu" -lcuda > "$OUT/05_nvcc_${v}.log" 2>&1 && NV=ok || NV=FAIL
  "$GCC" -O2 -fopenmp -x c -o "$OUT/${v}_cpu" "$cu" -lm > "$OUT/05_gcc_${v}.log" 2>&1 && GC=ok || GC=FAIL
  FR=""; RG=""
  for L in 402 403; do
    FR="$FR$(grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${L}E" "$OUT/05_nvcc_${v}.log" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)/"
    RG="$RG$(grep -A3 "kernel_dfs_iter_gpu_maxd14ILi${L}E" "$OUT/05_nvcc_${v}.log" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')/"
  done
  EQ="?"; N403="?"; POPS="?"; SHA="?"
  if [[ "$NV" == "ok" ]]; then
    "$CUOBJDUMP" -sass "$bin" > "$OUT/sass/${v}.txt" 2>&1
    [[ -x "$P402_BIN" ]] && EQ="$(sass_tool cmp "$OUT/sass/${P402_BIN}.txt" kernel_dfs_iter_gpu_maxd14PKj "$OUT/sass/${v}.txt" kernel_dfs_iter_gpu_maxd14ILi402E)"
    sass_tool pop "$OUT/sass/${v}.txt" > "$OUT/pop_${v}.txt"
    read -r N403 POPS SHA < <(head -1 "$OUT/pop_${v}.txt")
    tail -n +2 "$OUT/pop_${v}.txt"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$v" "$NV" "$GC" "${FR%/}" "${RG%/}" "$EQ" "$N403" "$POPS" "$SHA" >> "$OUT/403_r6_variants.tsv"
done
echo; cat "$OUT/403_r6_variants.tsv"
tar czf "${OUT}.tar.gz" "$OUT" && echo "tarball: ${OUT}.tar.gz"
