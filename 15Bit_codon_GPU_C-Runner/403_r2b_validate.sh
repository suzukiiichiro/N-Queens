#!/usr/bin/env bash
# 403_r2b_validate.sh
#
# rev403-r2b -- DIAGNOSTIC, NO code change, NO GPU time.
#   Does 403_r2's <402> instantiation compile to the same SASS as 402_r5's
#   kernel? (403-r2 measured the 402 layout at +0.123% inside 403_r2.)
#
#   Z1  HARD static: kernel region sha identical to 403_r2 (comment-stripped)
#   Z2  build 403_r2b with the production nvcc flags; reference binaries
#       402_r5_kernel_maxd14 and 403_kernel_maxd14 must exist (built by the
#       403 / 403-r2 harnesses with the same flags) or are built here
#   Z3  STATED: SASS(402_r5 kernel) == SASS(403_r2b <402>)   [normalised]
#   Z4  STATED: SASS(403 kernel)    == SASS(403_r2b <403>)   [control]
#   Z5  info: cubin .text sections (size/offset) of all three binaries
#
# USAGE
#   STATIC_ONLY=1 bash 403_r2b_validate.sh        # OK=15
#                 bash 403_r2b_validate.sh        # ~1 min (nvcc + cuobjdump)

set -u

REV="403_r2b"
PY_SRC="${PY_SRC:-403_r2bPy_kernel_maxd14_final.py}"
PREV_PY="${PREV_PY:-403_r2Py_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-403_r2b_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-403_r2b_kernel_maxd14}"
PREV_CU="${PREV_CU:-403_r2_kernel_maxd14.cu}"
P402_CU="${P402_CU:-402_r5_kernel_maxd14.cu}"
P402_BIN="${P402_BIN:-402_r5_kernel_maxd14}"
P403_CU="${P403_CU:-403_kernel_maxd14.cu}"
P403_BIN="${P403_BIN:-403_kernel_maxd14}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
MB="${MB:-960}"
KERNEL_SHA_403_R2="${KERNEL_SHA_403_R2:-4b9ce994c8199b14}"   # comment-stripped kernel region of 403_r2
STATIC_ONLY="${STATIC_ONLY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS (registered refutations are listed here too -- read the label):"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi; exit 0; }

# ---------------------------------------------------------------------
# 1. Static (Z1)
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
{ grep -q "^# 403-r2b " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 403-r2b ...' note block"
grep -qE '^REV_TAG:str="403_r2b"' "$CODE" && pass "source_rev_tag_is_403_r2b" || fail "source_rev_tag_is_403_r2b" "wrong REV_TAG"
grep -qF 'CRunnerEntry(14,"./403_r2b_kernel_maxd14","NQ_EXTRA_CTX=1 NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_points_at_403_r2b_keeps_helper_prefix" || fail "source_table_points_at_403_r2b_keeps_helper_prefix" "table entry wrong"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=${MB}\b" "$CODE" && pass "source_default_max_blocks_is_${MB}" || fail "source_default_max_blocks_is_${MB}" "default not $MB"
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_403_r2Py (removed=3 added=3 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_403_r2Py" "removed=$PR_ added=$PA_, expected 3/3"
else info "py_diff_fingerprint_vs_403_r2Py" "skipped"; fi
# Code region = from the first #include on, C comments and blank lines removed (standing rule).
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
[[ "$KB" == "$KERNEL_SHA_403_R2"* ]] && pass "Z1_cu_kernel_region_IDENTICAL_to_403_r2 (${KB:0:16})" || fail "Z1_cu_kernel_region_IDENTICAL_to_403_r2" "sha ${KB:0:16} != $KERNEL_SHA_403_R2 -- r2b must not change code"
cmp -s "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" && pass "Z1_cu_code_region_byte_identical_to_403_r2" || fail "Z1_cu_code_region_byte_identical_to_403_r2" "code region differs: $(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^[<>]') lines"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"/tmp/${REV}_gcc.log" && pass "cu_cpu_harness_builds" || fail "cu_cpu_harness_builds" "$(head -3 /tmp/${REV}_gcc.log)"
grep -q "rev403-r2b" "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev403-r2b note in header"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Build (Z2)
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"; [[ ! -x "$CUOBJDUMP" ]] && CUOBJDUMP="cuobjdump"
{ echo "=== $REV env $(date -Is) ==="; "$NVCC" --version; "$CUOBJDUMP" --version 2>&1 | head -3; } > "$LOGDIR/00_env.txt" 2>&1
banner "Building"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc_${REV}.log" 2>&1
[[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "$(grep -i error "$LOGDIR/05_nvcc_${REV}.log" | head -3)"; summary_exit; }
for pair in "$P402_CU:$P402_BIN" "$P403_CU:$P403_BIN"; do src="${pair%%:*}"; bin="${pair##*:}"
  if [[ ! -x "$bin" ]]; then "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$bin" "$src" -lcuda > "$LOGDIR/05_nvcc_${bin}.log" 2>&1; info "reference_binary_built[$bin]" "was missing -- rebuilt with the production flags"; fi
  [[ -x "$bin" ]] && pass "reference_binary_available[$bin]" || { fail "reference_binary_available[$bin]" "build failed"; summary_exit; }
done

# ---------------------------------------------------------------------
# 3. SASS (Z3, Z4, Z5)
# ---------------------------------------------------------------------
banner "SASS extraction"
for bin in "$P402_BIN" "$P403_BIN" "$CU_BIN"; do
  "$CUOBJDUMP" -sass "./$bin" > "$LOGDIR/sass_${bin}.txt" 2>&1 || { fail "cuobjdump[$bin]" "see $LOGDIR/sass_${bin}.txt"; summary_exit; }
done
pass "cuobjdump_sass_extracted (3 binaries)"

sass_cmp() {  # label fileA funcfragA fileB funcfragB
python3 - "$@" <<'PYEOF'
import sys, re, collections
label, fa, fragA, fb, fragB = sys.argv[1:6]
def funcs(path):
    out = {}; cur = None
    for line in open(path, encoding='utf-8', errors='replace'):
        m = re.match(r'\s*Function\s*:\s*(\S+)', line)
        if m: cur = m.group(1); out[cur] = []; continue
        if cur is None: continue
        m = re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s*(.*?);\s*(/\*.*\*/)?\s*$', line)
        if m: out[cur].append(m.group(2).strip())
        elif re.match(r'\s*\.L_\S+:', line): out[cur].append(line.strip())
    return out
def pick(fs, frag, avoid=None):
    c = [k for k in fs if frag in k and (avoid is None or avoid not in k)]
    return c[0] if len(c) == 1 else None
A = funcs(fa); B = funcs(fb)
ka = pick(A, fragA); kb = pick(B, fragB)
if ka is None or kb is None:
    print(f"CANNOT_LOCATE A={ka} B={kb} A_has={list(A)} B_has={list(B)}"); sys.exit(2)
ia = [x for x in A[ka] if not x.startswith('.L')]; ib = [x for x in B[kb] if not x.startswith('.L')]
BR = ('BRA', 'BRX', 'JMP', 'JMX', 'CALL', 'RET', 'BSSY', 'BSYNC', 'WARPSYNC')
def opcode(x):
    t = x.split()
    if t and t[0].startswith('@'): t = t[1:]
    return t[0] if t else ''
def mask_targets(x):  # branch targets are absolute offsets: identical code => identical targets, but mask them for the second-level verdict
    return re.sub(r'\b0x[0-9a-fA-F]+\b', '0xTGT', x) if opcode(x) in BR else x
na = ia; nb = ib
ha = collections.Counter(opcode(x) for x in ia); hb = collections.Counter(opcode(x) for x in ib)
print(f"{label}: A={ka} ({len(ia)} instr)  B={kb} ({len(ib)} instr)")
if na == nb:
    print(f"{label}: IDENTICAL instruction sequence ({len(na)} instructions, branch targets included)"); sys.exit(0)
ma = [mask_targets(x) for x in ia]; mb = [mask_targets(x) for x in ib]
if ma == mb:
    print(f"{label}: identical modulo branch targets ({len(na)} instructions) -- same code, different offsets"); 
first = next((i for i, (x, y) in enumerate(zip(na, nb)) if x != y), min(len(na), len(nb)))
print(f"{label}: DIFFER -- first difference at instruction #{first} (A has {len(na)}, B has {len(nb)})")
for i in range(max(0, first - 3), min(len(na), len(nb), first + 12)):
    mark = '  ' if na[i] == nb[i] else '!!'
    print(f"  {mark} #{i:5d}  A: {ia[i]:<60s} B: {ib[i]}")
delta = {op: hb[op] - ha[op] for op in set(ha) | set(hb) if hb[op] != ha[op]}
print(f"{label}: opcode histogram delta (B-A): " + (', '.join(f'{k}{v:+d}' for k, v in sorted(delta.items(), key=lambda kv: -abs(kv[1]))) or 'none (same multiset, different order/operands)'))
sys.exit(1)
PYEOF
}

banner "Z3: 402_r5 kernel vs 403_r2b <402>"
sass_cmp Z3 "$LOGDIR/sass_${P402_BIN}.txt" "kernel_dfs_iter_gpu_maxd14PKj" "$LOGDIR/sass_${CU_BIN}.txt" "kernel_dfs_iter_gpu_maxd14ILi402E" | tee "$LOGDIR/Z3_402pair.txt"; rc=${PIPESTATUS[0]}
if [[ "$rc" == "0" ]]; then pass "Z3_402_pair_SASS_IDENTICAL (the +0.12% is not the code -> adopt 403_r2; see Z5 for placement)"
elif [[ "$rc" == "1" ]]; then fail "Z3_REFUTED_402_pair_SASS_DIFFERS" "see $LOGDIR/Z3_402pair.txt -- 403-r2c changes the inlining spelling and re-measures X0/X1"
else fail "Z3_cannot_locate_kernels" "see $LOGDIR/Z3_402pair.txt"; fi

banner "Z4 (control): 403 kernel vs 403_r2b <403>"
sass_cmp Z4 "$LOGDIR/sass_${P403_BIN}.txt" "kernel_dfs_iter_gpu_maxd14PKj" "$LOGDIR/sass_${CU_BIN}.txt" "kernel_dfs_iter_gpu_maxd14ILi403E" | tee "$LOGDIR/Z4_403pair.txt"; rc=${PIPESTATUS[0]}
if [[ "$rc" == "0" ]]; then pass "Z4_403_pair_SASS_IDENTICAL"
elif [[ "$rc" == "1" ]]; then fail "Z4_REFUTED_403_pair_SASS_DIFFERS" "see $LOGDIR/Z4_403pair.txt"
else fail "Z4_cannot_locate_kernels" "see $LOGDIR/Z4_403pair.txt"; fi

banner "Z5: cubin .text sections (readelf on the extracted cubins)"
for bin in "$P402_BIN" "$P403_BIN" "$CU_BIN"; do
  d="$LOGDIR/cubin_${bin}"; mkdir -p "$d"; ( cd "$d" && "$CUOBJDUMP" -xelf all "../../$bin" > extract.log 2>&1 ) || true
  echo "--- $bin"; for c in "$d"/*.cubin; do [[ -f "$c" ]] || continue; echo "  $(basename "$c")"; readelf -SW "$c" 2>/dev/null | grep -E '\.text\.|Name\s+Type' | sed 's/^/    /'; done
done | tee "$LOGDIR/Z5_sections.txt"
pass "Z5_sections_listed (see $LOGDIR/Z5_sections.txt; informational)"

TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "tarball: $TARBALL"
summary_exit
