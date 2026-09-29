#!/usr/bin/env bash
# 403_r2d_validate.sh
#
# rev403-r2d -- the two-layout source spelled so that the 402 instantiation
#   is 402_r5's code for the front end: process_one_task<int layout> with
#   `if constexpr` at the three stack sites (plain-C CPU harness: runtime int
#   + plain if, as in 403-r2). 403-r2c showed every spelling that keeps a
#   folded runtime `if` gives the same 63-instruction SASS difference (+0.12%),
#   and P1 showed the PTX already differs (loop-carried movs re-ordered).
#
# GATES (in order; each a hard stop)
#   S0  static: 402_r5 push/pop verbatim, LAYOUT_IF at all 3 sites, template
#       plumbing, guards, gcc build, fingerprints (comment-stripped)
#   S1  ptxas: both instantiations frame 160 B, spill 0/0, regs <= 44, 2 entries
#   S2  SASS: <402> instantiation IDENTICAL to 402_r5's kernel (branch targets
#       included) -- HARD; nothing GPU runs if this fails
#   S3  SASS: <403> instantiation vs 403's kernel -- STATED identical
#   Y2  CPU per-record equality: N=21 auto == 402_r5 ; N=21 NQ_LAYOUT=403 == 403 ;
#       N=22 auto == 403
#   Y3  N=23 refused ; NQ_LAYOUT=402 at N=22 refused
#   Y4  N=21 full input @960 helper: X1 and X3 per-thread identical to X0
# PRE-REGISTERED
#   Y5' stated |X1 - X0| <= 0.05% (identical SASS => inside the 0.04% noise)
#   Y6' stated X3/X0 in [1.025, 1.031] (403's +2.79% now with 403's exact SASS)
#   Y7  HARD -g 21 21 MATCH + layout=402; stated within 0.15% of 109,437
#   Y8  HARD -g 22 22 MATCH + layout=403; stated within 0.5% of 987,065 (403's own SASS)
#   Y9  SM clock 1710 in every cell
#   ADOPTION: S0-S2, Y2-Y5', Y7 all hold -> 403_r2d is the production binary.
#
# USAGE
#   STATIC_ONLY=1 bash 403_r2d_validate.sh        # OK=25
#                 bash 403_r2d_validate.sh        # ~27 min
#   SKIP22=1      bash 403_r2d_validate.sh        # ~10 min, no -g 22 22
#   ALLOW_402_TEXT_MISMATCH=1 ...                # proceed past S0's text gate

set -u

REV="403_r2d"
PY_SRC="${PY_SRC:-403_r2dPy_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-403_r2dPy_kernel_maxd14_final}"
PREV_PY="${PREV_PY:-403_r2cPy_kernel_maxd14_final.py}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
CU_SRC="${CU_SRC:-403_r2d_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-403_r2d_kernel_maxd14}"
PREV_CU="${PREV_CU:-403_r2_kernel_maxd14.cu}"          # fingerprint base
P403_CU="${P403_CU:-403_kernel_maxd14.cu}"            # the 403 layout reference (N=22, CPU + SASS)
P403_BIN="${P403_BIN:-403_kernel_maxd14}"
P402_CU="${P402_CU:-402_r5_kernel_maxd14.cu}"          # the 402 layout reference (N=21, production)
P402_BIN="${P402_BIN:-402_r5_kernel_maxd14}"
CRLOG_DIR="${CRLOG_DIR:-403_r2d_crunner_logs}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
GCC="${GCC:-gcc}"
ARCH="${ARCH:-sm_86}"
BLOCK="${BLOCK:-32}"
MB="${MB:-960}"
HCTX="${HCTX:-1}"; HMB="${HMB:-128}"              # production helper state
CPU_CHECK_RECORDS="${CPU_CHECK_RECORDS:-8192}"
CPU_CHECK_RECORDS_22="${CPU_CHECK_RECORDS_22:-16384}"
ANCHOR_X0="${ANCHOR_X0:-109437.2}"                # 402-r5 G21 mean (production)
ANCHOR_G22="${ANCHOR_G22:-987064.75}"             # 403 G22
KERNEL_SHA_403_R2="${KERNEL_SHA_403_R2:-4b9ce994c8199b14}"   # comment-stripped kernel region of 403_r2
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
EXPECT_REMOVED="${EXPECT_REMOVED:-9}"
EXPECT_ADDED="${EXPECT_ADDED:-23}"    # comment-stripped code lines vs 403_r2
SKIP22="${SKIP22:-0}"
ALLOW_402_TEXT_MISMATCH="${ALLOW_402_TEXT_MISMATCH:-0}"
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
oracle_of() { case "$1" in 19) echo 4968057848;; 20) echo 39029188884;; 21) echo 314666222712;; 22) echo 2691008701644;; *) echo "";; esac; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f",(a-b)/b*100}'; }
abspct() { awk -v a="$1" -v b="$2" 'BEGIN{x=(a-b)/b*100; printf "%.3f",(x<0?-x:x)}'; }
ratio() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.4f",a/b}'; }
le() { awk -v x="$1" -v y="$2" 'BEGIN{exit !(x<=y)}'; }
absdiff_le() { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=t)}'; }
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS (registered refutations are listed here too -- read the label):"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi; exit 0; }

# ---------------------------------------------------------------------
# 1. Static (Y0)
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
py_note_quotes() { python3 -c "
import sys
lines=open(sys.argv[1],encoding='utf-8').read().split('\n')
i=next(i for i,l in enumerate(lines) if l.startswith('import '))
print('\n'.join(lines[:i]).count('\"'*3))
" "$1"; }
py_code_region "$PY_SRC" > "/tmp/${REV}_code_only.py"; CODE="/tmp/${REV}_code_only.py"
NOTE_LINES=$(awk '/^# =+$/{f=1} f&&/^#/{n++} END{print n+0}' "$PY_SRC")
{ grep -q "^# 403-r2d " "$PY_SRC" && [[ "$NOTE_LINES" -ge 20 ]]; } && pass "py_revision_notes_present ($NOTE_LINES comment lines)" || fail "py_revision_notes_present" "no '# 403-r2d ...' note block"
NQ_=$(py_note_quotes "$PY_SRC"); [[ $((NQ_ % 2)) -eq 0 ]] && pass "py_note_region_quotes_balanced ($NQ_)" || fail "py_note_region_quotes_balanced" "$NQ_ triple-quotes"
grep -qE '^REV_TAG:str="403_r2d"' "$CODE" && pass "source_rev_tag_is_403_r2d" || fail "source_rev_tag_is_403_r2d" "wrong REV_TAG"
grep -qF 'CRunnerEntry(14,"./403_r2d_kernel_maxd14","NQ_EXTRA_CTX=1 NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "' "$CODE" && pass "source_table_points_at_403_r2d_keeps_helper_prefix" || fail "source_table_points_at_403_r2d_keeps_helper_prefix" "table entry wrong"
grep -qE "^A10G_FINAL_DEFAULT_MAX_BLOCKS:int=${MB}\b" "$CODE" && pass "source_default_max_blocks_is_${MB}" || fail "source_default_max_blocks_is_${MB}" "default not $MB"
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "embedded quote in a module-level str literal"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF
if [[ -f "$PREV_PY" ]]; then
  py_code_region "$PREV_PY" > "/tmp/${REV}_prev_code.py"
  PR_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^<' || true); PA_=$(diff "/tmp/${REV}_prev_code.py" "$CODE" | grep -c '^>' || true)
  [[ "$PR_" == "3" && "$PA_" == "3" ]] && pass "py_diff_fingerprint_vs_403_r2cPy (removed=3 added=3 EXECUTABLE lines)" || fail "py_diff_fingerprint_vs_403_r2cPy" "removed=$PR_ added=$PA_, expected 3/3"
else info "py_diff_fingerprint_vs_403_r2cPy" "skipped"; fi
# Code region = from the first #include on, with ALL C comments (/* */ and //) and blank
# lines removed BEFORE any diff/sha. Comment lines -- including notes suzuki adds by hand --
# never move a fingerprint (standing rule; the 403 harness still diffed comments: fixed here).
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
cu_code_region "$CU_SRC" > "/tmp/${REV}_cur_code.cu"; cu_code_region "$PREV_CU" > "/tmp/${REV}_prev_code.cu"; cu_code_region "$P402_CU" > "/tmp/${REV}_402_code.cu"
KB="$(extract_kernel "/tmp/${REV}_cur_code.cu" | sha256sum | cut -d' ' -f1)"
[[ "$KB" != "$KERNEL_SHA_403_R2"* ]] && pass "cu_kernel_region_CHANGED_from_403_r2 (now ${KB:0:16}...)" || fail "cu_kernel_region_CHANGED_from_403_r2" "kernel region is still 403_r2's"
CR_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^<' || true); CA_=$(diff "/tmp/${REV}_prev_code.cu" "/tmp/${REV}_cur_code.cu" | grep -c '^>' || true)
[[ "$CR_" == "$EXPECT_REMOVED" && "$CA_" == "$EXPECT_ADDED" ]] && pass "cu_diff_fingerprint_vs_403_r2 (removed=$CR_ added=$CA_)" || fail "cu_diff_fingerprint_vs_403_r2" "removed=$CR_ added=$CA_, expected $EXPECT_REMOVED/$EXPECT_ADDED"
# Y0a: the 402 branch must be 402_r5's push/pop text, not a paraphrase.
python3 - "/tmp/${REV}_402_code.cu" "/tmp/${REV}_cur_code.cu" > "/tmp/${REV}_402_text.txt" 2>&1 <<'PYEOF'
import re, sys
def stmts(path):
    lines = open(path, encoding='utf-8').read().split('\n')
    out = []; buf = None
    for l in lines:
        t = l.strip()
        if buf is None:
            if re.match(r'(stack_[ab]\[stack_ptr\]|cur_(ld|rd|col|avail))\s*=', t) and ('packed_' in t or 'stack_' in t or buf is not None):
                buf = t
            else:
                continue
        else:
            buf += ' ' + t
        if buf.endswith(';'):
            if 'stack_' in buf or 'packed_' in buf:
                out.append(re.sub(r'\s+', '', buf))
            buf = None
    return out
ref = sorted(set(stmts(sys.argv[1])))
cur = set(stmts(sys.argv[2]))
missing = [r for r in ref if r not in cur]
for r in ref:
    print(('MISSING ' if r in missing else 'present ') + r)
print(f"{len(ref)-len(missing)}/{len(ref)} 402_r5 stack statements present")
sys.exit(1 if missing or not ref else 0)
PYEOF
rc=$?
if [[ "$rc" == "0" ]]; then pass "cu_402_branch_is_402_r5_text_verbatim ($(tail -1 "/tmp/${REV}_402_text.txt"))"
elif [[ "$ALLOW_402_TEXT_MISMATCH" == "1" ]]; then info "cu_402_branch_TEXT_MISMATCH_ALLOWED" "$(grep MISSING "/tmp/${REV}_402_text.txt" | tr '\n' ' ')"
else fail "cu_402_branch_is_402_r5_text_verbatim" "$(grep -c MISSING "/tmp/${REV}_402_text.txt") of 402_r5's statements are not in $CU_SRC (see /tmp/${REV}_402_text.txt) -- paste the MISSING lines back to Claude, or ALLOW_402_TEXT_MISMATCH=1 to rely on the CPU/GPU equality gates alone"; fi
[[ "$(extract_kernel "/tmp/${REV}_cur_code.cu" | grep -c 'LAYOUT_IF (layout == PACK_LAYOUT_402) {')" == "3" && "$(grep -c 'if (layout == PACK_LAYOUT_402) {' "/tmp/${REV}_cur_code.cu")" == "3" ]] && pass "cu_LAYOUT_IF_at_all_3_stack_sites (+3 host selections)" || fail "cu_LAYOUT_IF_at_all_3_stack_sites" "kernel LAYOUT_IF $(extract_kernel "/tmp/${REV}_cur_code.cu" | grep -c 'LAYOUT_IF (layout == PACK_LAYOUT_402) {') (expected 3), host if $(grep -c 'if (layout == PACK_LAYOUT_402) {' "/tmp/${REV}_cur_code.cu") (expected 3)"
{ grep -q '^#define LAYOUT_IF(c)      if constexpr (c)' "/tmp/${REV}_cur_code.cu" && grep -q '^#define POT_TEMPLATE      template <int layout>' "/tmp/${REV}_cur_code.cu" && grep -q 'process_one_task POT_TARGS(LAYOUT) (' "/tmp/${REV}_cur_code.cu"; } && pass "cu_if_constexpr_template_plumbing" || fail "cu_if_constexpr_template_plumbing" "macros/calls missing"
[[ "$(grep -c 'stack_b\[stack_ptr\] = (cur_rd & ~1u) | ((cur_ld >> 20) & 1u);' "/tmp/${REV}_cur_code.cu")" == "2" ]] && pass "cu_403_word_B_kept_at_both_pushes" || fail "cu_403_word_B_kept_at_both_pushes" "403 push text missing"
grep -q 'cur_ld  = ((uint32_t)(packed_a >> 44) & PACK403_LDLO) | ((packed_b & 1u) << 20);' "/tmp/${REV}_cur_code.cu" && pass "cu_403_pop_kept" || fail "cu_403_pop_kept" "403 pop text missing"
{ grep -q '^template <int LAYOUT>' "/tmp/${REV}_cur_code.cu" && [[ "$(grep -c 'kernel_dfs_iter_gpu_maxd14<PACK_LAYOUT_402>' "/tmp/${REV}_cur_code.cu")" -ge 3 && "$(grep -c 'kernel_dfs_iter_gpu_maxd14<PACK_LAYOUT_403>' "/tmp/${REV}_cur_code.cu")" -ge 3 ]]; } && pass "cu_kernel_template_instantiated_402_and_403" || fail "cu_kernel_template_instantiated_402_and_403" "template/instantiation missing"
grep -q 'n3, n4 POT_LAYOUT_ARG(LAYOUT)' "/tmp/${REV}_cur_code.cu" && grep -q 'POT_FORCEINLINE' "/tmp/${REV}_cur_code.cu" && pass "cu_layout_is_template_constant_and_forceinlined" || fail "cu_layout_is_template_constant_and_forceinlined" "LAYOUT not passed as template arg / no forceinline"
{ [[ "$(grep -c 'if (N > PACK403_WIDTH) {' "/tmp/${REV}_cur_code.cu")" == "2" ]] && [[ "$(grep -c 'select_pack_layout(N, ' "/tmp/${REV}_cur_code.cu")" == "2" ]]; } && pass "cu_N_gt_22_refused_and_layout_selected_in_both_mains" || fail "cu_N_gt_22_refused_and_layout_selected_in_both_mains" "guard/selection count != 2"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_cur" "$CU_SRC" -lm 2>"/tmp/${REV}_gcc.log" && pass "cu_cpu_harness_builds" || fail "cu_cpu_harness_builds" "$(head -3 /tmp/${REV}_gcc.log)"
grep -q "rev403-r2d" "$CU_SRC" && pass "cu_header_note_present" || fail "cu_header_note_present" "no rev403-r2d note in header"

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

# ---------------------------------------------------------------------
# 2. Build + Y1
# ---------------------------------------------------------------------
mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; } > "$LOGDIR/00_env_pre.txt" 2>&1
banner "Building"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
ptxas_of() {  # log mangled-name-fragment
  FRAME="$(grep -A3 "$2" "$1" | grep -o '[0-9]* bytes stack frame' | head -1 | cut -d' ' -f1)"
  SPILLS="$(grep -A3 "$2" "$1" | grep -o '[0-9]* bytes spill stores, [0-9]* bytes spill loads' | head -1)"
  REGS="$(grep -A3 "$2" "$1" | grep -o 'Used [0-9]* registers' | head -1 | grep -o '[0-9]*')"; }
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -lineinfo -Xptxas -v -o "$CU_BIN" "$CU_SRC" -lcuda > "$LOGDIR/05_nvcc_${REV}.log" 2>&1
[[ -x "$CU_BIN" ]] && pass "nvcc_build[$CU_BIN]" || { fail "nvcc_build" "$(grep -i error "$LOGDIR/05_nvcc_${REV}.log" | head -3)"; summary_exit; }
for pair in "$P402_CU:$P402_BIN" "$P403_CU:$P403_BIN"; do src="${pair%%:*}"; bin="${pair##*:}"
  if [[ ! -x "$bin" ]]; then "$NVCC" -O3 -arch="$ARCH" -lineinfo -o "$bin" "$src" -lcuda > "$LOGDIR/05_nvcc_${bin}.log" 2>&1; fi
  [[ -x "$bin" ]] && pass "reference_binary_available[$bin]" || { fail "reference_binary_available[$bin]" "build failed"; summary_exit; }
done
Y1_OK=1
for L in 402 403; do
  ptxas_of "$LOGDIR/05_nvcc_${REV}.log" "kernel_dfs_iter_gpu_maxd14ILi${L}E"; CF="${FRAME:-?}"; CREG="${REGS:-?}"; CS="${SPILLS:-?}"
  info "ptxas ${REV}<${L}>" "frame=${CF}B regs=${CREG} ${CS}"
  if [[ "$CF" == "160" && "$CS" == "0 bytes spill stores, 0 bytes spill loads" && "$CREG" =~ ^[0-9]+$ && "$CREG" -le 44 ]]; then pass "S1_ptxas_L${L}_frame160_spill0_regs${CREG}"
  else fail "S1_ptxas_L${L}" "frame=${CF}B regs=${CREG} ${CS}"; Y1_OK=0; fi
done
[[ "$(grep -c 'Function properties for' "$LOGDIR/05_nvcc_${REV}.log")" == "2" ]] && pass "S1_exactly_two_ptxas_entries (process_one_task inlined into both)" || { fail "S1_ptxas_entry_count" "expected 2 entries, got $(grep -c 'Function properties for' "$LOGDIR/05_nvcc_${REV}.log") -- process_one_task not inlined?"; Y1_OK=0; }
[[ "$Y1_OK" == "1" ]] || summary_exit
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/06_codon.log"
[[ -x "$PY_BIN" ]] && pass "codon_build[$PY_BIN]" || { fail "codon_build" "see log"; summary_exit; }

banner "S2/S3: SASS identity (nothing runs on the GPU unless S2 holds)"
[[ ! -x "$CUOBJDUMP" ]] && CUOBJDUMP="cuobjdump"
mkdir -p "$LOGDIR/sass"
"$CUOBJDUMP" -sass "./$CU_BIN" > "$LOGDIR/sass/${CU_BIN}.txt" 2>&1 && "$CUOBJDUMP" -sass "./$P402_BIN" > "$LOGDIR/sass/${P402_BIN}.txt" 2>&1 && "$CUOBJDUMP" -sass "./$P403_BIN" > "$LOGDIR/sass/${P403_BIN}.txt" 2>&1 || { fail "cuobjdump" "see $LOGDIR/sass"; summary_exit; }
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
A = funcs(fa); B = funcs(fb)
ka = [k for k in A if fragA in k]; kb = [k for k in B if fragB in k]
if len(ka) != 1 or len(kb) != 1: print(f"{label}: CANNOT_LOCATE A={ka} B={kb}"); sys.exit(2)
ia, ib = A[ka[0]], B[kb[0]]
if ia == ib: print(f"{label}: IDENTICAL ({len(ia)} instructions, branch targets included)"); sys.exit(0)
diff = [i for i, (x, y) in enumerate(zip(ia, ib)) if x != y]
print(f"{label}: DIFFER {len(diff)} of {len(ia)}/{len(ib)} instructions (first at #{diff[0] if diff else min(len(ia),len(ib))})")
for i in diff[:8]: print(f"    #{i:5d}  A: {ia[i]:<58s} B: {ib[i]}")
sys.exit(1)
PYEOF
}
sass_cmp S2 "$LOGDIR/sass/${P402_BIN}.txt" "kernel_dfs_iter_gpu_maxd14PKj" "$LOGDIR/sass/${CU_BIN}.txt" "kernel_dfs_iter_gpu_maxd14ILi402E" | tee "$LOGDIR/S2_402.txt"; rc=${PIPESTATUS[0]}
[[ "$rc" == "0" ]] && pass "S2_402_instantiation_SASS_IDENTICAL_to_402_r5" || { fail "S2_402_instantiation_SASS_DIFFERS" "if constexpr did not restore 402_r5's code; see S2_402.txt -- no GPU time spent"; summary_exit; }
sass_cmp S3 "$LOGDIR/sass/${P403_BIN}.txt" "kernel_dfs_iter_gpu_maxd14PKj" "$LOGDIR/sass/${CU_BIN}.txt" "kernel_dfs_iter_gpu_maxd14ILi403E" | tee "$LOGDIR/S3_403.txt"; rc=${PIPESTATUS[0]}
[[ "$rc" == "0" ]] && pass "S3_403_instantiation_SASS_IDENTICAL_to_403" || fail "S3_403_instantiation_SASS_DIFFERS_stated_identical" "N=22 timing then reads relative to 403-r2 (983,926), not 403; see S3_403.txt"

# ---------------------------------------------------------------------
# 3. Inputs, Y2, Y3
# ---------------------------------------------------------------------
IN21=""; for lg in $(ls -t 40*_crunner_logs/crunner_*_N21.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { IN21="$c"; break; }; done
[[ -f "${IN21:-/nonexistent}" && "$(stat -c %s "$IN21")" == "56707896" ]] && pass "input21_located ($IN21)" || { fail "input21_located" "no N=21 sched input via 40*_crunner_logs"; summary_exit; }
IN22=""; for lg in $(ls -t 40*_crunner_logs/crunner_*_N22.log 2>/dev/null); do c="$(grep -o 'src=[^ ]*' "$lg" | head -1 | cut -d= -f2)"; [[ -f "$c" ]] && { IN22="$c"; break; }; done
[[ -f "${IN22:-/nonexistent}" ]] || IN22="$(ls -t constellations_N22_*.sched394f.bin 2>/dev/null | head -1)"
if [[ -f "${IN22:-/nonexistent}" ]]; then info "input22_located" "$IN22 ($(( $(stat -c %s "$IN22") / 28 )) records)"; else info "input22" "not present -- Y2c deferred to after G22"; fi

banner "Y2: CPU-harness per-record equality"
"$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_402" "$P402_CU" -lm 2>"$LOGDIR/03_gcc_402.log" && "$GCC" -O2 -fopenmp -x c -o "/tmp/${REV}_cpu_403" "$P403_CU" -lm 2>"$LOGDIR/03_gcc_403.log" || { fail "Y2_cpu_reference_builds" "see 03_gcc_*.log"; summary_exit; }
head -c $((CPU_CHECK_RECORDS*28)) "$IN21" > "/tmp/${REV}_cpu_in21.bin"
"/tmp/${REV}_cpu_402" 21 "/tmp/${REV}_cpu_in21.bin" "/tmp/${REV}_cpu_402_21.bin" 2>&1 | tail -1 | tee "$LOGDIR/04_cpu_402_21.log"
"/tmp/${REV}_cpu_403" 21 "/tmp/${REV}_cpu_in21.bin" "/tmp/${REV}_cpu_403_21.bin" 2>&1 | tail -1 | tee "$LOGDIR/04_cpu_403_21.log"
env -u NQ_LAYOUT "/tmp/${REV}_cpu_cur" 21 "/tmp/${REV}_cpu_in21.bin" "/tmp/${REV}_cpu_cur21_auto.bin" 2>&1 | grep -E 'layout|cputest-done' | tee "$LOGDIR/04_cpu_cur21_auto.log"
NQ_LAYOUT=403 "/tmp/${REV}_cpu_cur" 21 "/tmp/${REV}_cpu_in21.bin" "/tmp/${REV}_cpu_cur21_l403.bin" 2>&1 | grep -E 'layout|cputest-done' | tee "$LOGDIR/04_cpu_cur21_l403.log"
grep -q '\[cpu-layout\] N=21 requested=auto layout=402' "$LOGDIR/04_cpu_cur21_auto.log" && pass "Y2_cpu_auto_selects_402_at_N21" || { fail "Y2_cpu_auto_selects_402_at_N21" "layout line missing/wrong"; summary_exit; }
if cmp -s "/tmp/${REV}_cpu_402_21.bin" "/tmp/${REV}_cpu_cur21_auto.bin" && [[ -s "/tmp/${REV}_cpu_cur21_auto.bin" ]]; then pass "Y2a_cpu_N21_auto_identical_402_r5 ($CPU_CHECK_RECORDS records)"; else fail "Y2a_cpu_N21_auto_DIFFER" "the 402 branch is not 402_r5 -- stopping"; summary_exit; fi
if cmp -s "/tmp/${REV}_cpu_403_21.bin" "/tmp/${REV}_cpu_cur21_l403.bin" && [[ -s "/tmp/${REV}_cpu_cur21_l403.bin" ]]; then pass "Y2b_cpu_N21_layout403_identical_403 ($CPU_CHECK_RECORDS records)"; else fail "Y2b_cpu_N21_layout403_DIFFER" "the 403 branch is not 403 -- stopping"; summary_exit; fi
y2c() {
  head -c $((CPU_CHECK_RECORDS_22*28)) "$IN22" > "/tmp/${REV}_cpu_in22.bin"
  "/tmp/${REV}_cpu_403" 22 "/tmp/${REV}_cpu_in22.bin" "/tmp/${REV}_cpu_403_22.bin" 2>&1 | tail -1 | tee "$LOGDIR/04_cpu_403_22.log"
  env -u NQ_LAYOUT "/tmp/${REV}_cpu_cur" 22 "/tmp/${REV}_cpu_in22.bin" "/tmp/${REV}_cpu_cur22_auto.bin" 2>&1 | grep -E 'layout|cputest-done' | tee "$LOGDIR/04_cpu_cur22_auto.log"
  grep -q '\[cpu-layout\] N=22 requested=auto layout=403' "$LOGDIR/04_cpu_cur22_auto.log" || { fail "Y2_cpu_auto_selects_403_at_N22" "layout line missing/wrong"; return 1; }
  if cmp -s "/tmp/${REV}_cpu_403_22.bin" "/tmp/${REV}_cpu_cur22_auto.bin" && [[ -s "/tmp/${REV}_cpu_cur22_auto.bin" ]]; then pass "Y2c_cpu_N22_auto_identical_403 ($CPU_CHECK_RECORDS_22 records)"; return 0; else fail "Y2c_cpu_N22_auto_DIFFER" "N=22 results changed"; return 1; fi
}
Y2C_PENDING=0
if [[ -f "${IN22:-/nonexistent}" ]]; then y2c || summary_exit; else Y2C_PENDING=1; fi

banner "Y3: refusals"
env -u NQ_LAYOUT "./$CU_BIN" 23 "$IN21" "/tmp/${REV}_n23.bin" > "$LOGDIR/07_n23_refusal.log" 2>&1; rc=$?
{ [[ "$rc" == "3" ]] && grep -q '\[403-pack\] N=23 unsupported' "$LOGDIR/07_n23_refusal.log"; } && pass "Y3a_N23_refused (rc=3)" || { fail "Y3a_N23_refused" "rc=$rc"; summary_exit; }
NQ_LAYOUT=402 "./$CU_BIN" 22 "$IN21" "/tmp/${REV}_l402n22.bin" > "$LOGDIR/07_l402_n22_refusal.log" 2>&1; rc=$?
{ [[ "$rc" == "3" ]] && grep -q '\[403-r2-layout\] N=22 unsupported with NQ_LAYOUT=402' "$LOGDIR/07_l402_n22_refusal.log"; } && pass "Y3b_layout402_at_N22_refused (rc=3)" || { fail "Y3b_layout402_at_N22_refused" "rc=$rc"; summary_exit; }

# ---------------------------------------------------------------------
# 4. GPU cells
# ---------------------------------------------------------------------
printf 'cell\tN\tbinary\tlayout\tpath\tmax_blocks\thelpers\thelper_mb\tfree_mb\tk_per_thread_max\tkernel_ms\ttotal_sum\tmatch\tsm_mean\tsm_min\ttemp_max\tstart\n' > "$TSV"
CLKPID=""
gpu_gate() { local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ')"; echo "$apps" > "$LOGDIR/apps_before_$1.txt"
  [[ -z "$apps" ]] && return 0; fail "gpu_empty_before[$1]" "compute process(es) present: $(echo "$apps" | tr '\n' ' ')"; return 1; }
clk_start() { local f="$LOGDIR/clk_$1.tsv"; echo "time,sm_mhz,mem_mhz,power_w,temp_c,util_pct" > "$f"
  ( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep "$CLK_INTERVAL"; done ) >> "$f" 2>/dev/null & CLKPID=$!; }
clk_stop() { [[ -n "$CLKPID" ]] && { kill "$CLKPID" 2>/dev/null; wait "$CLKPID" 2>/dev/null || true; }; CLKPID=""
  read -r SMMEAN SMMIN TMAX < <(awk -F, 'NR>1 && $2+0>0 {s+=$2; n++; if(min==""||$2<min)min=$2; if($5>t)t=$5} END{if(n) printf "%.0f %d %d\n", s/n, min, t; else print "? ? ?"}' "$LOGDIR/clk_$1.tsv"); }
fields_of() { KMS="$(grep -o 'kernel_ms=[0-9.]*' "$1" | head -1 | cut -d= -f2)"; TOT="$(grep -o 'total_sum=[0-9]*' "$1" | head -1 | cut -d= -f2)"
  MATCH=0; grep -q '\[gpu-run-correctness\] MATCH' "$1" && MATCH=1; FREE="$(grep -o 'free_mb=[0-9]*' "$1" | head -1 | cut -d= -f2)"
  CFGMB="$(grep -o 'MAX_BLOCKS=[0-9]*' "$1" | head -1 | cut -d= -f2)"; HL="$(grep -o 'helpers=[0-9]*' "$1" | head -1 | cut -d= -f2)"; HM="$(grep -o 'helper_mb=[0-9]*' "$1" | head -1 | cut -d= -f2)"
  KPT="$(grep -o 'k_per_thread_max=[0-9]*' "$1" | head -1 | cut -d= -f2)"; LAY="$(grep -o '\[gpu-layout\] N=[0-9]* requested=[a-z0-9]* layout=[0-9]*' "$1" | head -1 | sed 's/.*layout=//')"; }
declare -A VAL SMM
record() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$TSV"; }
run_direct() {  # cell N bin input mb out layout(auto|402|403)
  local cell="$1" n="$2" bin="$3" in="$4" mb="$5" out="$6" lay="${7:-auto}" orc; orc="$(oracle_of "$n")"
  gpu_gate "$cell" || return 1
  local lg="$LOGDIR/3_${cell}.log" start; start="$(date -Is)"; clk_start "$cell"
  env -u NQ_PAD_MB -u NQ_CARVEOUT -u NQ_EXTRA_CTX NQ_LAYOUT="$lay" NQ_BLOCK="$BLOCK" NQ_MAX_BLOCKS="$mb" NQ_HELPER_CTX="$HCTX" NQ_HELPER_MB="$HMB" "./$bin" "$n" "$in" "$out" "$orc" > "$lg" 2>&1 || true
  clk_stop "$cell"; fields_of "$lg"
  record "$cell" "$n" "$bin" "${LAY:-n/a}" direct "$mb" "${HL:-?}" "${HM:-?}" "${FREE:-?}" "${KPT:-?}" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start"
  [[ "$MATCH" == "1" && "${TOT:-}" == "$orc" ]] || { fail "oracle[$cell]" "total_sum='${TOT:-<none>}' expected $orc"; return 1; }
  VAL[$cell]="$KMS"; SMM[$cell]="$SMMEAN"
  info "$cell" "$bin N=$n layout=${LAY:-n/a} mb=$mb helpers=$HL/$HM kernel_ms=$KMS free_mb=$FREE k=$KPT sm_mean=$SMMEAN"
}
run_g() {  # cell N
  local cell="$1" n="$2" orc; orc="$(oracle_of "$n")"; local gcr="$CRLOG_DIR/crunner_${CU_BIN}_N${n}.log"; rm -f "$gcr"
  gpu_gate "$cell" || return 1
  local start; start="$(date -Is)"; clk_start "$cell"
  env -u NQ_MAX_BLOCKS -u NQ_PAD_MB -u NQ_EXTRA_CTX -u NQ_BLOCK -u NQ_CARVEOUT -u NQ_HELPER_CTX -u NQ_HELPER_MB -u NQ_LAYOUT "./$PY_BIN" -g "$n" "$n" > "$LOGDIR/2_${cell}_console.log" 2>&1
  clk_stop "$cell"
  cp "$gcr" "$LOGDIR/2_${cell}_crunner.log" 2>/dev/null || { fail "crunner_path_taken[$cell]" "no $gcr"; return 1; }
  cp "$CRLOG_DIR/dispatch.log" "$LOGDIR/2_${cell}_dispatch.log" 2>/dev/null || true
  fields_of "$gcr"
  record "$cell" "$n" "$CU_BIN" "${LAY:-n/a}" dispatch "${CFGMB:-?}" "${HL:-?}" "${HM:-?}" "${FREE:-?}" "${KPT:-?}" "${KMS:-?}" "${TOT:-?}" "$MATCH" "$SMMEAN" "$SMMIN" "$TMAX" "$start"
  [[ "$MATCH" == "1" && "${TOT:-}" == "$orc" ]] || { fail "oracle[$cell]" "total_sum='${TOT:-<none>}' expected $orc"; return 1; }
  VAL[$cell]="$KMS"; SMM[$cell]="$SMMEAN"; G_IN="$(grep -o 'src=[^ ]*' "$gcr" | head -1 | cut -d= -f2)"
  info "$cell" "-g $n $n: layout=${LAY:-n/a} MAX_BLOCKS=$CFGMB helpers=$HL/$HM kernel_ms=$KMS free_mb=$FREE k=$KPT sm_mean=$SMMEAN src=$G_IN"
}

banner "Y4: N=21 full input @$MB, helper $HCTX+$HMB MB -- X0 (402_r5) vs X1 (403_r2 auto) vs X3 (403_r2 layout 403)"
run_direct X0 21 "$P402_BIN" "$IN21" "$MB" "/tmp/${REV}_X0.bin" auto || summary_exit
sleep "$COOLDOWN"
run_direct X1 21 "$CU_BIN" "$IN21" "$MB" "/tmp/${REV}_X1.bin" auto || summary_exit
[[ "$(grep -c '\[gpu-layout\] N=21 requested=auto layout=402' "$LOGDIR/3_X1.log")" == "1" ]] && pass "X1_auto_selected_402" || { fail "X1_auto_selected_402" "layout line missing"; summary_exit; }
if cmp -s "/tmp/${REV}_X0.bin" "/tmp/${REV}_X1.bin" && [[ -s "/tmp/${REV}_X1.bin" ]]; then pass "Y4a_X1_per_thread_identical_to_X0 ($(stat -c %s "/tmp/${REV}_X1.bin") bytes)"; else fail "Y4a_X1_DIFFER" "timings are NOT reported"; summary_exit; fi
sleep "$COOLDOWN"
run_direct X3 21 "$CU_BIN" "$IN21" "$MB" "/tmp/${REV}_X3.bin" 403 || summary_exit
[[ "$(grep -c '\[gpu-layout\] N=21 requested=403 layout=403' "$LOGDIR/3_X3.log")" == "1" ]] && pass "X3_forced_403" || { fail "X3_forced_403" "layout line missing"; summary_exit; }
if cmp -s "/tmp/${REV}_X0.bin" "/tmp/${REV}_X3.bin"; then pass "Y4b_X3_per_thread_identical_to_X0"; else fail "Y4b_X3_DIFFER" "timings are NOT reported"; summary_exit; fi
p="$(pct "${VAL[X1]}" "${VAL[X0]}")"; a="$(abspct "${VAL[X1]}" "${VAL[X0]}")"
if le "$a" 0.05; then pass "Y5p_identical_SASS_identical_time (X1 ${p}% vs X0=${VAL[X0]})"
elif le "$a" 0.15; then fail "Y5p_grey_zone" "X1 ${p}% vs X0 (stated <= 0.05% with identical SASS; 403-r2 measured +0.123% with different SASS)"
else fail "Y5p_REFUTED_time_differs_despite_identical_SASS" "X1 ${p}% vs X0 -- do NOT adopt"; fi
r="$(ratio "${VAL[X3]}" "${VAL[X0]}")"
{ le 1.025 "$r" && le "$r" 1.031; } && pass "Y6p_403_SASS_reproduces_403_cost (X3/X0=${r}, 403 measured 1.0279, 403-r2 1.0240)" || fail "Y6p_403_cost_outside_band" "X3/X0=${r} (stated [1.025, 1.031])"
d="$(abspct "${VAL[X0]}" "$ANCHOR_X0")"; le "$d" 0.15 && pass "X0_anchor (${d}% from 402-r5 G21 mean)" || info "X0_anchor" "${d}% from $ANCHOR_X0 -- read X1/X3 relative to X0"
sleep "$COOLDOWN"

banner "G21: -g 21 21 through the dispatcher (403_r2d table: MB=$MB, helper) -- Y7"
run_g G21 21 || summary_exit
grep -q '\[gpu-layout\] N=21 requested=auto layout=402' "$LOGDIR/2_G21_crunner.log" && pass "Y7_G21_oracle_MATCH_layout402 (kernel_ms=${VAL[G21]})" || { fail "Y7_G21_layout_line" "crunner log lacks layout=402"; summary_exit; }
d="$(abspct "${VAL[G21]}" "$ANCHOR_X0")"; le "$d" 0.15 && pass "Y7_G21_production_restored_on_one_binary (${d}% from 109,437)" || fail "Y7_G21_off_anchor" "${d}% from $ANCHOR_X0 (stated <= 0.15%)"

if [[ "$SKIP22" != "1" ]]; then
  sleep "$COOLDOWN"
  banner "G22: -g 22 22 through the dispatcher -- Y8"
  run_g G22 22 || summary_exit
  grep -q '\[gpu-layout\] N=22 requested=auto layout=403' "$LOGDIR/2_G22_crunner.log" && pass "Y8_G22_oracle_MATCH_layout403 (kernel_ms=${VAL[G22]})" || { fail "Y8_G22_layout_line" "crunner log lacks layout=403"; summary_exit; }
  d="$(abspct "${VAL[G22]}" "$ANCHOR_G22")"; le "$d" 0.5 && pass "Y8_G22_replicates_403 (${d}% from 987,065)" || fail "Y8_G22_off_403" "${d}% from $ANCHOR_G22 (stated <= 0.5%; 403-r2 with reshuffled SASS gave 983,926)"
  [[ -f "${IN22:-/nonexistent}" ]] || IN22="$G_IN"
  if [[ "$Y2C_PENDING" == "1" && -f "${IN22:-/nonexistent}" ]]; then banner "Y2c (deferred): CPU equality at N=22"; y2c || true; fi
else info "G22" "skipped (SKIP22=1)"; fi
CLK_OK=1; for c in "${!SMM[@]}"; do m="${SMM[$c]}"; [[ "$m" == "?" ]] && continue; absdiff_le "$m" 1710 34 || { CLK_OK=0; info "clock[$c]" "mean SM $m MHz"; }; done
[[ "$CLK_OK" == "1" ]] && pass "Y9_sm_clock_1710_all_cells" || fail "Y9_sm_clock_1710_all_cells" "see clk_*.tsv"

banner "Results"
python3 - "$TSV" <<'EOF' | tee "$LOGDIR/9_ranked.txt"
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['kernel_ms'] not in ('?','')]
print(f"{'cell':<5}{'N':>3}{'binary':<24}{'lay':>4}{'path':>9}{'MB':>5}{'help':>6}{'free_mb':>8}{'k':>6}{'kernel_ms':>14}{'match':>6}{'sm':>6}{'h:mm:ss':>10}")
for r in rows:
    v=float(r['kernel_ms']); s=v/1000
    print(f"{r['cell']:<5}{r['N']:>3}{r['binary']:<24}{r['layout']:>4}{r['path']:>9}{r['max_blocks']:>5}{r['helpers']+'/'+r['helper_mb']:>6}{r['free_mb']:>8}{r['k_per_thread_max']:>6}{v:>14.3f}{r['match']:>6}{r['sm_mean']:>6}{int(s//3600):>4}:{int(s%3600//60):02d}:{s%60:04.1f}")
EOF
{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
cp -r "$CRLOG_DIR" "$LOGDIR/crunner_logs" 2>/dev/null || true
TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed"
echo "results: $TSV"; echo "tarball: $TARBALL"
summary_exit
