#!/usr/bin/env bash
# 403_r2c_validate.sh
#
# rev403-r2c -- COMPILE-ONLY SEARCH: which spelling of the two-layout source
#   makes the <402> kernel's SASS IDENTICAL to 402_r5's? (403-r2b: same 496
#   instructions, same opcodes, same branch targets; 63 instructions differ
#   only in register names / commutative operand order / intra-block order,
#   concentrated around the 402 push/pop -> ptxas tie-breaking, +0.123%.)
#
#   The delivered 403_r2c_kernel_maxd14.cu is 403_r2's code, unchanged. The
#   harness generates spelling variants (python patch of the delivered file),
#   builds each with the production flags, and compares SASS against 402_r5
#   (and against 403 for the <403> kernel):
#     C0  control: rebuild 402_r5 from source -> must be SASS-identical to the
#         existing 402_r5 binary (nvcc determinism)                    HARD
#     P1  PTX: is the <402> entry of V0 already different from 402_r5's PTX
#         (canonical register/label names), or is it ptxas alone?     info
#     V0  as delivered              V1  no __forceinline__
#     V2  explicit instantiation <402> then <403>   V3  <403> then <402>
#     V4  two plain kernels (the 402 one keeps 402_r5's exact name)
#     V5  V4 without __forceinline__
#     V6  DIAG single <402> instantiation           V7  V6 without __forceinline__
#   WINNER = first of V1..V5 whose <402> SASS is identical (branch targets
#   included) to 402_r5; a variant whose <403> SASS is also identical to
#   403's is preferred. V6/V7 are diagnostics (they cannot run N=22) and
#   never win. The winner source is written to 403_r2c_kernel_maxd14.winner.cu
#   and built as 403_r2c_kernel_maxd14 (the binary the .py table names).
#
# PRE-REGISTERED
#   R1 STATED: C0 identical.  R2 STATED: at least one of V1..V5 is identical
#      (refuted -> no winner; adopt 403_r2 as-is at +0.12% or keep two binaries).
#   R3 (MEASURE=1 only, ~4 min): X0 = 402_r5 vs X1w = winner, N=21 @960 with
#      helper: per-thread byte-identical (HARD) and |X1w - X0| <= 0.05%
#      (STATED: identical SASS => inside the 0.04% noise floor).
#
# USAGE
#   STATIC_ONLY=1 bash 403_r2c_validate.sh        # OK=17
#                 bash 403_r2c_validate.sh        # ~3 min of nvcc, no GPU
#   MEASURE=1     bash 403_r2c_validate.sh        # + X0/X1w if a winner exists (~4 min GPU)

set -u

REV="403_r2c"
PY_SRC="${PY_SRC:-403_r2cPy_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-403_r2cPy_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-403_r2bPy_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-403_r2c_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-403_r2c_kernel_maxd14}"
WINNER_SRC="${WINNER_SRC:-403_r2c_kernel_maxd14.winner.cu}"
PREV_CU="${PREV_CU:-403_r2_kernel_maxd14.cu}"
P402_CU="${P402_CU:-402_r5_kernel_maxd14.cu}"
P402_BIN="${P402_BIN:-402_r5_kernel_maxd14}"
P403_CU="${P403_CU:-403_kernel_maxd14.cu}"
P403_BIN="${P403_BIN:-403_kernel_maxd14}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
BLOCK="${BLOCK:-32}"
MB="${MB:-960}"
HCTX="${HCTX:-1}"; HMB="${HMB:-128}"
KERNEL_SHA_403_R2="${KERNEL_SHA_403_R2:-4b9ce994c8199b14}"
ANCHOR_X0="${ANCHOR_X0:-109437.2}"
MEASURE="${MEASURE:-0}"
COOLDOWN="${COOLDOWN:-10}"
CLK_INTERVAL="${CLK_INTERVAL:-5}"
STATIC_ONLY="${STATIC_ONLY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"
TSV="$LOGDIR/${REV}_results.tsv"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }
oracle_of() { case "$1" in 21) echo 314666222712;; 22) echo 2691008701644;; *) echo "";; esac; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f",(a-b)/b*100}'; }
abspct() { awk -v a="$1" -v b="$2" 'BEGIN{x=(a-b)/b*100; printf "%.3f",(x<0?-x:x)}'; }
le() { awk -v x="$1" -v y="$2" 'BEGIN{exit !(x<=y)}'; }
absdiff_le() { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=t)}'; }
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS (registered refutations are listed here too -- read the label):"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi; exit 0; }

# ---------------------------------------------------------------------
# 1. Static
# ---------------------------------------------------------------------
for f in "$PY_SRC" "$HELPER_SRC" "$CU_SRC" "$PREV_CU" "$P402_CU" "$P403_CU"; do
  [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"
done
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }
py_code_region() { python3 -c "
import sys
lines=open(sys.argv[1],encoding='utf-8').read().split('\n')
i=next(i for i,l in enumerate(lines) if l.startswith('import '))
sys.stdout.write('\n'.join(l for l in lines[i:] if not l.lstrip().startswith('#')))
" "$1"; }
py_code_region "$PY_SRC" > "/tmp/${REV}_code_only.py"; CODE="/tmp/${REV}_code_only.py"
NOTE_LINES=$(awk '/^# =+$/{f=1} f&&/^#/{n++} END{print n+0}' "$PY_SRC")
{ grep -q "^# 403-r2c " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 403-r2c ...' note block"
grep -qE '^REV_TAG:str="403_r2c"' "$CODE" && pass "source_rev_tag_is_403_r2c" || fail "source_rev_tag_is_403_r2c" "wrong REV_TAG"
grep -qF 'CRunnerEntry(14,"./403_r2c_kernel_maxd14","NQ_EXTRA_CTX=1 NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_points_at_403_r2c_keeps_helper_prefix" || fail "source_table_points_at_403_r2c_keeps_helper_prefix" "table entry wrong"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=${MB}\b" "$CODE" && pass "source_default_max_blocks_is_${MB}" || fail "source_default_max_blocks_is_${MB}" "default not $MB"
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_403_r2bPy (removed=3 added=3 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_403_r2bPy" "removed=$PR_ added=$PA_, expected 3/3"
else info "py_diff_fingerprint_vs_403_r2bPy" "skipped"; fi
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
[[ "$KB" == "$KERNEL_SHA_403_R2"* ]] && pass "cu_kernel_region_IDENTICAL_to_403_r2 (${KB:0:16})" || fail "cu_kernel_region_IDENTICAL_to_403_r2" "sha ${KB:0:16} != $KERNEL_SHA_403_R2 -- the delivered r2c .cu must be 403_r2's code"
cmp -s "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" && pass "cu_code_region_byte_identical_to_403_r2" || fail "cu_code_region_byte_identical_to_403_r2" "code region differs"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"/tmp/${REV}_gcc.log" && pass "cu_cpu_harness_builds" || fail "cu_cpu_harness_builds" "$(head -3 /tmp/${REV}_gcc.log)"
grep -q "rev403-r2c" "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev403-r2c note in header"

# variant generator (patches the delivered source; each variant is checked to still build as plain C)
gen_variants() { python3 - "$CU_SRC" "$1" <<'PYEOF'
import sys, os
base = open(sys.argv[1], encoding='utf-8').read(); outdir = sys.argv[2]; os.makedirs(outdir, exist_ok=True)
KT_START = 'template <int LAYOUT>\n__global__ void kernel_dfs_iter_gpu_maxd14('
i = base.index(KT_START); j = base.index('\n}\n', i) + 3
kernel_tmpl = base[i:j]
EXPLICIT = ('template __global__ void kernel_dfs_iter_gpu_maxd14<%s>('
            'const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, '
            'const uint32_t*, const uint32_t*, const uint8_t*, uint64_t*, int64_t, uint32_t, uint32_t, uint32_t, int64_t);\n')
def noinline(s):
    a = '#define POT_FORCEINLINE __forceinline__'; assert s.count(a) == 1; return s.replace(a, '#define POT_FORCEINLINE')
def explicit(s, first, second):
    return s.replace(kernel_tmpl, kernel_tmpl + EXPLICIT % first + EXPLICIT % second, 1)
def nontemplate(s):
    body = kernel_tmpl[len('template <int LAYOUT>\n'):]
    k402 = body.replace('n3, n4, LAYOUT', 'n3, n4, PACK_LAYOUT_402')
    k403 = body.replace('kernel_dfs_iter_gpu_maxd14(', 'kernel_dfs_iter_gpu_maxd14_l403(', 1).replace('n3, n4, LAYOUT', 'n3, n4, PACK_LAYOUT_403')
    s = s.replace(kernel_tmpl, k402 + '\n' + k403, 1)
    s = s.replace('kernel_dfs_iter_gpu_maxd14<PACK_LAYOUT_402>', 'kernel_dfs_iter_gpu_maxd14')
    return s.replace('kernel_dfs_iter_gpu_maxd14<PACK_LAYOUT_403>', 'kernel_dfs_iter_gpu_maxd14_l403')
def only402(s):
    return s.replace('kernel_dfs_iter_gpu_maxd14<PACK_LAYOUT_403>', 'kernel_dfs_iter_gpu_maxd14<PACK_LAYOUT_402>')
variants = [
    ('V0_base',                 'kernel_dfs_iter_gpu_maxd14ILi402E', 'kernel_dfs_iter_gpu_maxd14ILi403E', 'as delivered (403_r2)',                        lambda s: s),
    ('V1_noinline',             'kernel_dfs_iter_gpu_maxd14ILi402E', 'kernel_dfs_iter_gpu_maxd14ILi403E', 'POT_FORCEINLINE empty',                        noinline),
    ('V2_inst402first',         'kernel_dfs_iter_gpu_maxd14ILi402E', 'kernel_dfs_iter_gpu_maxd14ILi403E', 'explicit instantiation <402> then <403>',      lambda s: explicit(s, 'PACK_LAYOUT_402', 'PACK_LAYOUT_403')),
    ('V3_inst403first',         'kernel_dfs_iter_gpu_maxd14ILi402E', 'kernel_dfs_iter_gpu_maxd14ILi403E', 'explicit instantiation <403> then <402>',      lambda s: explicit(s, 'PACK_LAYOUT_403', 'PACK_LAYOUT_402')),
    ('V4_nontemplate',          'kernel_dfs_iter_gpu_maxd14PKj',     'kernel_dfs_iter_gpu_maxd14_l403',   'two plain kernels; 402 keeps 402_r5 name',    nontemplate),
    ('V5_nontemplate_noinline', 'kernel_dfs_iter_gpu_maxd14PKj',     'kernel_dfs_iter_gpu_maxd14_l403',   'V4 + POT_FORCEINLINE empty',                   lambda s: noinline(nontemplate(s))),
    ('V6_only402',              'kernel_dfs_iter_gpu_maxd14ILi402E', '',                                  'DIAG single <402> instantiation',              only402),
    ('V7_only402_noinline',     'kernel_dfs_iter_gpu_maxd14ILi402E', '',                                  'DIAG V6 + POT_FORCEINLINE empty',              lambda s: noinline(only402(s))),
]
for name, f402, f403, desc, fn in variants:
    open(os.path.join(outdir, name + '.cu'), 'w', encoding='utf-8').write(fn(base))
    print(f'{name}\t{f402}\t{f403}\t{desc}')
PYEOF
}
mkdir -p "/tmp/${REV}_variants"; gen_variants "/tmp/${REV}_variants" > "/tmp/${REV}_variants/list.tsv"
NV=$(wc -l < "/tmp/${REV}_variants/list.tsv"); [[ "$NV" == "8" ]] && pass "variants_generated ($NV)" || fail "variants_generated" "got $NV"
VB=0; while IFS=$'\t' read -r name f402 f403 desc; do "$GCC" -O2 -fopenmp -x c -o /dev/null "/tmp/${REV}_variants/$name.cu" -lm 2>/dev/null || { VB=1; fail "variant_cpu_build[$name]" "does not build as plain C"; }; done < "/tmp/${REV}_variants/list.tsv"
[[ "$VB" == "0" ]] && pass "variants_all_build_as_plain_C"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Tools
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR/variants" "$LOGDIR/sass" "$LOGDIR/ptx"; pass "logdir_created[$LOGDIR]"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"; [[ ! -x "$CUOBJDUMP" ]] && CUOBJDUMP="cuobjdump"
cp "/tmp/${REV}_variants/"*.cu "/tmp/${REV}_variants/list.tsv" "$LOGDIR/variants/"
{ echo "=== $REV env $(date -Is) ==="; "$NVCC" --version; } > "$LOGDIR/00_env.txt" 2>&1

# sass_cmp label fileA fragA fileB fragB  -> rc 0 identical / 1 differ / 2 cannot locate ; prints one summary line + details
sass_cmp() { python3 - "$@" <<'PYEOF'
import sys, re
label, fa, fragA, fb, fragB = sys.argv[1:6]
def funcs(path):
    out = {}; cur = None
    for line in open(path, encoding='utf-8', errors='replace'):
        m = re.match(r'\s*Function\s*:\s*(\S+)', line)
        if m: cur = m.group(1); out[cur] = []; continue
        if cur is None: continue
        m = re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s*(.*?);\s*(/\*.*\*/)?\s*$', line)
        if m: out[cur].append(m.group(2).strip())
    return out
def pick(fs, frag):
    c = [k for k in fs if frag in k and not (frag.endswith('PKj') and '_l403' in k)]
    return c[0] if len(c) == 1 else None
A = funcs(fa); B = funcs(fb); ka = pick(A, fragA); kb = pick(B, fragB)
if ka is None or kb is None: print(f"{label}: CANNOT_LOCATE A={ka} B={kb} B_has={list(B)}"); sys.exit(2)
ia, ib = A[ka], B[kb]
if ia == ib: print(f"{label}: IDENTICAL ({len(ia)} instructions, branch targets included)"); sys.exit(0)
diff = [i for i, (x, y) in enumerate(zip(ia, ib)) if x != y]
print(f"{label}: DIFFER {len(diff)} of {len(ia)}/{len(ib)} instructions (first at #{diff[0] if diff else min(len(ia),len(ib))})")
for i in diff[:8]: print(f"    #{i:5d}  A: {ia[i]:<58s} B: {ib[i]}")
sys.exit(1)
PYEOF
}
# ptx_cmp label fileA fragA fileB fragB : canonical register + label renaming, .loc/.file/comments stripped
ptx_cmp() { python3 - "$@" <<'PYEOF'
import sys, re
label, fa, fragA, fb, fragB = sys.argv[1:6]
def entries(path):
    txt = open(path, encoding='utf-8', errors='replace').read()
    out = {}
    for m in re.finditer(r'\.visible \.entry (\S+)\(', txt):
        name = m.group(1); i = txt.index('{', m.start()); depth = 0; j = i
        while True:
            c = txt[j]; depth += (c == '{') - (c == '}'); j += 1
            if depth == 0: break
        lines = []
        for l in txt[i:j].split('\n'):
            l = re.sub(r'//.*', '', l).strip()
            if not l or l.startswith('.loc') or l.startswith('.file') or l.startswith('.pragma'): continue
            lines.append(l)
        out[name] = lines
    return out
def canon(lines):
    regs = {}; labs = {}; counts = {}
    def rr(m):
        k = m.group(0)
        if k not in regs:
            t = re.match(r'%([a-z]+)', k).group(1); counts[t] = counts.get(t, 0) + 1; regs[k] = f'%{t}#{counts[t]}'
        return regs[k]
    def rl(m):
        k = m.group(0); return labs.setdefault(k, f'$L#{len(labs)}')
    out = []
    for l in lines:
        l = re.sub(r'\$L__[A-Za-z0-9_]+', rl, l)
        l = re.sub(r'%[a-z]+\d+', rr, l)
        out.append(l)
    return out
A = entries(fa); B = entries(fb)
ka = [k for k in A if fragA in k]; kb = [k for k in B if fragB in k and not (fragB.endswith('PKj') and '_l403' in k)]
if len(ka) != 1 or len(kb) != 1: print(f"{label}: CANNOT_LOCATE A={ka} B={kb}"); sys.exit(2)
ca, cb = canon(A[ka[0]]), canon(B[kb[0]])
if ca == cb: print(f"{label}: PTX IDENTICAL modulo names ({len(ca)} lines)"); sys.exit(0)
diff = [i for i, (x, y) in enumerate(zip(ca, cb)) if x != y]
print(f"{label}: PTX DIFFERS {len(diff)} of {len(ca)}/{len(cb)} lines (first at #{diff[0] if diff else min(len(ca),len(cb))})")
for i in diff[:6]: print(f"    #{i:5d}  A: {ca[i]:<58s} B: {cb[i]}")
sys.exit(1)
PYEOF
}
build_sass() {  # src bin log -> $LOGDIR/sass/<bin>.txt ; sets PTXAS_SUM
  rm -f "$2"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$2" "$1" -lcuda > "$3" 2>&1 || return 1
  "$CUOBJDUMP" -sass "$2" > "$LOGDIR/sass/$(basename "$2").txt" 2>&1 || return 1
  local E FR RG; E=$(grep -c 'Function properties for' "$3"); FR=$(grep -o '[0-9]* bytes stack frame' "$3" | sort -u | tr '\n' ' '); RG=$(grep -o 'Used [0-9]* registers' "$3" | sort -u | tr '\n' ' ')
  PTXAS_SUM="entries=$E frames=[${FR% }] regs=[${RG% }]"
}

# ---------------------------------------------------------------------
# 3. C0 control: nvcc determinism on 402_r5 itself
# ---------------------------------------------------------------------
banner "C0: rebuild 402_r5 from source, compare to the existing binary"
[[ -x "$P402_BIN" ]] || { "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$P402_BIN" "$P402_CU" -lcuda > "$LOGDIR/05_nvcc_${P402_BIN}.log" 2>&1; info "reference_binary_built[$P402_BIN]" "was missing"; }
[[ -x "$P403_BIN" ]] || { "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$P403_BIN" "$P403_CU" -lcuda > "$LOGDIR/05_nvcc_${P403_BIN}.log" 2>&1; info "reference_binary_built[$P403_BIN]" "was missing"; }
"$CUOBJDUMP" -sass "./$P402_BIN" > "$LOGDIR/sass/ref_402_r5.txt" 2>&1 && "$CUOBJDUMP" -sass "./$P403_BIN" > "$LOGDIR/sass/ref_403.txt" 2>&1 || { fail "cuobjdump_references" "see $LOGDIR/sass"; summary_exit; }
build_sass "$P402_CU" "/tmp/${REV}_c0_402_r5" "$LOGDIR/05_nvcc_C0.log" || { fail "C0_build" "see 05_nvcc_C0.log"; summary_exit; }
sass_cmp C0 "$LOGDIR/sass/ref_402_r5.txt" "kernel_dfs_iter_gpu_maxd14PKj" "$LOGDIR/sass/${REV}_c0_402_r5.txt" "kernel_dfs_iter_gpu_maxd14PKj" | tee "$LOGDIR/C0.txt"; rc=${PIPESTATUS[0]}
[[ "$rc" == "0" ]] && pass "R1_C0_nvcc_deterministic_on_402_r5" || { fail "R1_REFUTED_C0_402_r5_rebuild_differs" "nvcc is not deterministic here -- the search is moot; see C0.txt"; summary_exit; }

# ---------------------------------------------------------------------
# 4. P1: PTX of V0 <402> vs 402_r5
# ---------------------------------------------------------------------
banner "P1: PTX comparison (before ptxas)"
"$NVCC" -O3 -arch="$ARCH" -ptx -o "$LOGDIR/ptx/402_r5.ptx" "$P402_CU" > "$LOGDIR/ptx/nvcc_402_r5.log" 2>&1 && "$NVCC" -O3 -arch="$ARCH" -ptx -o "$LOGDIR/ptx/V0.ptx" "/tmp/${REV}_variants/V0_base.cu" > "$LOGDIR/ptx/nvcc_V0.log" 2>&1 || { fail "P1_ptx_build" "see $LOGDIR/ptx"; summary_exit; }
ptx_cmp P1 "$LOGDIR/ptx/402_r5.ptx" "kernel_dfs_iter_gpu_maxd14PKj" "$LOGDIR/ptx/V0.ptx" "kernel_dfs_iter_gpu_maxd14ILi402E" | tee "$LOGDIR/P1.txt"; rc=${PIPESTATUS[0]}
if [[ "$rc" == "0" ]]; then info "P1" "PTX identical modulo names -> the 63-instruction difference is ptxas alone (register allocation / scheduling tie-breaks)"
elif [[ "$rc" == "1" ]]; then info "P1" "PTX already differs -> the difference starts in the front end (template / inline context); see P1.txt"
else info "P1" "could not locate entries; see P1.txt"; fi

# ---------------------------------------------------------------------
# 5. Variants
# ---------------------------------------------------------------------
banner "Variants: build with production flags, compare SASS"
printf 'variant\tdesc\tptxas\tsass402_vs_402_r5\tsass403_vs_403\n' > "$TSV"
WINNER=""; WINNER_BOTH=""
while IFS=$'\t' read -r name f402 f403 desc; do
  bin="/tmp/${REV}_$name"
  if ! build_sass "/tmp/${REV}_variants/$name.cu" "$bin" "$LOGDIR/05_nvcc_$name.log"; then
    printf '%s\t%s\tBUILD_FAIL\t-\t-\n' "$name" "$desc" >> "$TSV"; info "$name" "BUILD FAILED ($(grep -i error "$LOGDIR/05_nvcc_$name.log" | head -1))"; continue
  fi
  sass_cmp "$name/402" "$LOGDIR/sass/ref_402_r5.txt" "kernel_dfs_iter_gpu_maxd14PKj" "$LOGDIR/sass/${REV}_$name.txt" "$f402" > "$LOGDIR/${name}_402.txt"; r=$?
  case "$r" in 0) s402="IDENTICAL";; 1) s402="$(head -1 "$LOGDIR/${name}_402.txt" | sed 's/.*: //; s/ instructions.*//')";; *) s402="CANNOT_LOCATE";; esac
  if [[ -n "$f403" ]]; then
    sass_cmp "$name/403" "$LOGDIR/sass/ref_403.txt" "kernel_dfs_iter_gpu_maxd14PKj" "$LOGDIR/sass/${REV}_$name.txt" "$f403" > "$LOGDIR/${name}_403.txt"; r=$?
    case "$r" in 0) s403="IDENTICAL";; 1) s403="$(head -1 "$LOGDIR/${name}_403.txt" | sed 's/.*: //; s/ instructions.*//')";; *) s403="CANNOT_LOCATE";; esac
  else s403="n/a"; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$desc" "$PTXAS_SUM" "$s402" "$s403" >> "$TSV"
  info "$name" "$PTXAS_SUM | 402: $s402 | 403: $s403"
  case "$name" in V[1-5]*)
    if [[ "$s402" == "IDENTICAL" ]]; then
      [[ -z "$WINNER" ]] && WINNER="$name"
      [[ "$s403" == "IDENTICAL" && -z "$WINNER_BOTH" ]] && WINNER_BOTH="$name"
    fi;;
  esac
done < "/tmp/${REV}_variants/list.tsv"
[[ -n "$WINNER_BOTH" ]] && WINNER="$WINNER_BOTH"
echo; column -t -s$'\t' "$TSV" | tee "$LOGDIR/9_matrix.txt"
if [[ -n "$WINNER" ]]; then
  pass "R2_winner_found[$WINNER] (402 SASS identical to 402_r5$([[ "$WINNER" == "$WINNER_BOTH" ]] && echo ', 403 SASS identical to 403'))"
  cp "/tmp/${REV}_variants/$WINNER.cu" "$WINNER_SRC"; cp "/tmp/${REV}_$WINNER" "$CU_BIN"; cp "$WINNER_SRC" "$LOGDIR/"
  diff "$CU_SRC" "$WINNER_SRC" > "$LOGDIR/winner_diff_vs_delivered.txt" || true
  info "winner" "source -> $WINNER_SRC, binary -> $CU_BIN; diff vs delivered .cu: $(grep -c '^[<>]' "$LOGDIR/winner_diff_vs_delivered.txt") lines (in $LOGDIR/winner_diff_vs_delivered.txt)"
else
  fail "R2_REFUTED_no_variant_reaches_SASS_identity" "see $LOGDIR/9_matrix.txt -- decide: adopt 403_r2 at +0.12%, or keep two binaries"
fi

# ---------------------------------------------------------------------
# 6. MEASURE=1: X0 (402_r5) vs X1w (winner), N=21 @MB with helper
# ---------------------------------------------------------------------
if [[ "$MEASURE" == "1" && -n "$WINNER" ]]; then
  banner "R3: X0 402_r5 vs X1w $WINNER -- per-thread identity, |X1w - X0| <= 0.05%"
  IN21=""; for lg in $(ls -t 40*_crunner_logs/crunner_*_N21.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { IN21="$c"; break; }; done
  [[ -f "${IN21:-/nonexistent}" && "$(stat -c %s "$IN21")" == "56707896" ]] && pass "input21_located ($IN21)" || { fail "input21_located" "no N=21 sched input via 40*_crunner_logs"; summary_exit; }
  printf 'cell\tbinary\tkernel_ms\ttotal_sum\tmatch\tsm_mean\tfree_mb\n' > "$LOGDIR/measure.tsv"
  CLKPID=""
  gpu_gate() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_before_$1.txt"; [[ -z "$apps" ]] && return 0; fail "gpu_empty_before[$1]" "compute process(es) present"; return 1; }
  clk_start() { local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"; ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPID=$!; }
  clk_stop() { [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""; SMMEAN=$(awk -F, 'NR>1 && $2+0>0 {s+=$2; n++} END{if(n) printf "%.0f", s/n; else print "?"}' "$LOGDIR/clk_$1.tsv"); }
  declare -A VAL
  run_direct() {  # cell bin out
    local cell="$1" bin="$2" out="$3" orc; orc="$(oracle_of 21)"; gpu_gate "$cell" || return 1
    local lg="$LOGDIR/3_${cell}.log"; clk_start "$cell"
    env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX NQ_LAYOUT=auto NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$MB" NQ_HELPER_CTX="$HCTX" NQ_HELPER_MB="$HMB" "./$bin" 21 "$IN21" "$out" "$orc" > "$lg" 2>&1 || true
    clk_stop "$cell"
    local KMS TOT MATCH FREE; KMS="$(grep -o 'kernel_ms=[0-9.]*' "$lg" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$lg" | head -1 | cut -d= -f2)"; MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$lg" && MATCH=1; FREE="$(grep -o 'free_mb=[0-9]*' "$lg" | head -1 | cut -d= -f2)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$bin" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$SMMEAN" "${FREE:-?}" >> "$LOGDIR/measure.tsv"
    [[ "$MATCH" == "1" && "${TOT:-}" == "$orc" ]] || { fail "oracle[$cell]" "total_sum='${TOT:-<none>}'"; return 1; }
    VAL[$cell]="$KMS"; info "$cell" "$bin kernel_ms=$KMS free_mb=$FREE sm_mean=$SMMEAN"
    absdiff_le "$SMMEAN" 1710 34 || fail "clock[$cell]" "mean SM $SMMEAN MHz"
  }
  run_direct X0 "$P402_BIN" "/tmp/${REV}_X0.bin" || summary_exit
  sleep "$COOLDOWN"
  run_direct X1w "$CU_BIN" "/tmp/${REV}_X1w.bin" || summary_exit
  grep -q '\[gpu-layout\] N=21 requested=auto layout=402' "$LOGDIR/3_X1w.log" && pass "X1w_auto_selected_402" || fail "X1w_auto_selected_402" "layout line missing"
  if cmp -s "/tmp/${REV}_X0.bin" "/tmp/${REV}_X1w.bin" && [[ -s "/tmp/${REV}_X1w.bin" ]]; then pass "R3a_X1w_per_thread_identical_to_X0"; else fail "R3a_X1w_DIFFER" "timing not reported"; summary_exit; fi
  p="$(pct "${VAL[X1w]}" "${VAL[X0]}")"; a="$(abspct "${VAL[X1w]}" "${VAL[X0]}")"
  le "$a" 0.05 && pass "R3b_identical_SASS_identical_time (X1w ${p}% vs X0=${VAL[X0]})" || fail "R3b_REFUTED_time_differs_despite_identical_SASS" "X1w ${p}% vs X0 (stated <= 0.05%)"
  d="$(abspct "${VAL[X0]}" "$ANCHOR_X0")"; le "$d" 0.15 && pass "X0_anchor (${d}% from 109,437)" || info "X0_anchor" "${d}% from $ANCHOR_X0"
elif [[ "$MEASURE" == "1" ]]; then info "MEASURE" "skipped: no winner"; fi

TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "matrix: $LOGDIR/9_matrix.txt"; echo "tarball: $TARBALL"
summary_exit
