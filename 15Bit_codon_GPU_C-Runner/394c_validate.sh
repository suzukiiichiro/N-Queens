#!/usr/bin/env bash
# 394c_validate.sh
#
# rev394c -- feed the production CRunner the chunkshape148-REORDERED
# input (bench_mode=39) and measure it against the raw-order production
# path (bench_mode=37), through the Codon dispatcher, oracle-gated.
#
# WHY (393-10)
# ------------
# The production C path has never seen the 330-350 host-side reordering:
# ensure_crunner_input_bin() is fed the RAW stream bin. The reordering
# (w3_j7 -1.29%, bucket_run=2048 -4.11%, isort9 -0.82%, all measured on
# the Codon path) is lane-aware ("lanephase32"): it decides which 32
# tasks share a warp. 393-9/394a established that the loss is entirely
# intra-warp, so this is the mechanism it should address. The reordering
# is a permutation of the task set (333): the oracle is unchanged.
#
# DESIGN: a 2x2 factorial, one variable per axis
# ---------------------------------------------
#                      raw input (mode 37)     reordered input (mode 39)
#   MAX_BLOCKS=484     A  (= 389 anchor)       B
#   MAX_BLOCKS=968     C  (= 394b r1 point)    D
#
#   A vs B : reorder effect at the production stride  (394c's question)
#   C vs D : reorder effect at the 394b r1 optimum
#   A vs C : MAX_BLOCKS effect, raw    (replicates 394b)
#   B vs D : MAX_BLOCKS effect, reordered
#   A vs D : both together
# 348 used the same factorial shape. Default runs all four (~15 min +
# ~3 min to shape the 968 bin the first time); STAGES=A,B for 394c's
# question alone.
#
# HOW THE STRIDE STAYS CONSISTENT
# -------------------------------
# Mode 39 shapes for gpu_block*gpu_max_blocks (argv[4]*argv[5]) and
# passes NQ_MAX_BLOCKS=argv[5] to the C binary itself, so argv[5] is the
# single source of truth for both. Mode 37 ignores argv[5] for the C
# launch, so for C the harness exports NQ_MAX_BLOCKS=968 in the shell
# (os.system inherits it). Every run's [gpu-run-done] line carries the
# stride the C binary actually used (394b onward); the harness checks it
# against the intended value -- a stale binary that ignores the env var
# would be caught here, not misread as a result.
#
# PRE-REGISTERED (394c_README_append.md):
#   A reproduces 201,237 +-1%;  C reproduces 163,185 +-1%
#   B vs A: -3 .. -6%   (falsified if |delta| <= 1%)
#   D vs C: same sign as B vs A, magnitude within +-3 points of it
#
# USAGE
#   STATIC_ONLY=1 bash 394c_validate.sh
#                 bash 394c_validate.sh            # A B C D
#   STAGES=A,B    bash 394c_validate.sh            # production stride only

set -u

REV="394c"
PY_SRC="${PY_SRC:-394cPy_kernel_maxd14_final.py}"
PY_BIN="${PY_BIN:-394cPy_kernel_maxd14_final}"
HELPER_SRC="${HELPER_SRC:-rev386_validation_helpers.py}"
FILTER_SCRIPT="${FILTER_SCRIPT:-363_filter_maxd14_only.py}"
CU_SRC="${CU_SRC:-394c_kernel_maxd14.cu}"
CU_BIN="${CU_BIN:-394c_kernel_maxd14}"
REF_CU_394B="${REF_CU_394B:-394b_kernel_maxd14.cu}"
CODON="${CODON:-codon}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_86}"
NQ="${NQ:-21}"
ORACLE="${ORACLE:-314666222712}"
STAGES="${STAGES:-A,B,C,D}"
MB_PROD="${MB_PROD:-484}"
MB_OPT="${MB_OPT:-968}"
REF_A_MS="${REF_A_MS:-201237}"       # 389 anchor
REF_C_MS="${REF_C_MS:-163185}"       # 394b r1 968 point
COOLDOWN="${COOLDOWN:-20}"
STATIC_ONLY="${STATIC_ONLY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_factorial_${TS}}"
TSV="$LOGDIR/${REV}_factorial.tsv"
CRLOG_DIR="${REV}_crunner_logs"      # = f"{REV_TAG}_crunner_logs" in the .py

PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
banner() { echo; echo "======================================================"; echo "$*"; echo "======================================================"; }
want() { [[ ",$STAGES," == *",$1,"* ]]; }

# ---------------------------------------------------------------------
# 1. Static checks
# ---------------------------------------------------------------------
for f in "$PY_SRC" "$HELPER_SRC" "$FILTER_SCRIPT" "$CU_SRC"; do
  if [[ -f "$f" ]]; then pass "file_present[$f]"; else fail "file_present[$f]" "not found in $(pwd)"; fi
done
[[ "$FAIL" -gt 0 ]] && { echo "OK=$PASS FAIL=$FAIL"; exit 1; }

# Strip the module docstring (the project's convention: static checks are
# immune to prose) before grepping code.
python3 - "$PY_SRC" > "/tmp/${REV}_code_only.py" <<'EOF'
import re,sys
s=open(sys.argv[1],encoding='utf-8').read()
m=re.search(r'"""',s); e=s.index('"""',m.end()); print(s[:m.start()]+s[e+3:])
EOF
CODE="/tmp/${REV}_code_only.py"

grep -qE '^REV_TAG:str="394c"' "$CODE" && pass "source_rev_tag_is_394c" || fail "source_rev_tag_is_394c" "REV_TAG is not 394c"

# THE recurring trap (361/365/366/368/369/385/391): a new bench_mode that
# is not in the CLI whitelist is silently reset to 0.
if grep -qE 'if not \(bench_mode==0 .*bench_mode==39\):' "$CODE"; then
  pass "source_bench_mode_39_in_cli_whitelist"
else
  fail "source_bench_mode_39_in_cli_whitelist" "bench_mode==39 missing from the CLI whitelist gate -- mode 39 would silently run as mode 0"
fi
if grep -qE '^    if bench_mode==11 .*bench_mode==39:' "$CODE"; then
  pass "source_bench_mode_39_in_preset_gate"
else
  fail "source_bench_mode_39_in_preset_gate" "bench_mode==39 missing from the preset gate"
fi
grep -q 'if use_gpu and N>=21 and bench_mode==39:' "$CODE" && pass "source_mode_39_dispatch_present" || fail "source_mode_39_dispatch_present" "dispatch block missing"

# Mode 39 must be reached before the generic N>=21 fallthrough.
L39=$(grep -n 'if use_gpu and N>=21 and bench_mode==39:' "$CODE" | head -1 | cut -d: -f1)
LGEN=$(grep -n '^    if use_gpu and N>=21:$' "$CODE" | head -1 | cut -d: -f1)
if [[ -n "$L39" && -n "$LGEN" && "$L39" -lt "$LGEN" ]]; then pass "source_mode_39_before_generic_fallthrough"; else fail "source_mode_39_before_generic_fallthrough" "mode 39 block at line ${L39:-?} is not before the generic N>=21 block at ${LGEN:-?}"; fi

# Single-variable: mode 37 must still feed the RAW stream bin, mode 39 the shaped bin.
[[ "$(grep -c 'ensure_crunner_input_bin(N,stream_fname,gpu_log_level)' "$CODE")" == "1" ]] && pass "source_mode_37_still_raw_input" || fail "source_mode_37_still_raw_input" "mode 37's input call changed"
[[ "$(grep -c 'ensure_crunner_input_bin(N,shaped_fname,gpu_log_level)' "$CODE")" == "1" ]] && pass "source_mode_39_uses_shaped_input" || fail "source_mode_39_uses_shaped_input" "mode 39 does not feed shaped_fname"
grep -q 'NQ_MAX_BLOCKS={gpu_max_blocks}' "$CODE" && pass "source_mode_39_couples_stride_via_env_prefix" || fail "source_mode_39_couples_stride_via_env_prefix" "env_prefix coupling missing"
grep -q 'if shaped_records!=stream_records:' "$CODE" && pass "source_mode_39_permutation_guard" || fail "source_mode_39_permutation_guard" "record-count guard missing"
grep -q 'CRunnerEntry(14,"./394c_kernel_maxd14",""' "$CODE" && pass "source_dispatch_table_references_394c_binary" || fail "source_dispatch_table_references_394c_binary" "table does not reference ./394c_kernel_maxd14 with empty env_prefix"

# .cu: code region identical to 394b (header-only rename).
if [[ -f "$REF_CU_394B" ]]; then
  A=$(awk 'f||/^#include/{f=1;print}' "$REF_CU_394B" | sha256sum | cut -d' ' -f1)
  B=$(awk 'f||/^#include/{f=1;print}' "$CU_SRC" | sha256sum | cut -d' ' -f1)
  [[ "$A" == "$B" ]] && pass "cu_code_region_identical_to_394b (${B:0:16}...)" || fail "cu_code_region_identical_to_394b" "394c .cu code region differs from 394b's"
else
  info "cu_code_region_identical_to_394b" "skipped ($REF_CU_394B not present)"
fi
grep -q 'getenv("NQ_MAX_BLOCKS")' "$CU_SRC" && pass "cu_reads_NQ_MAX_BLOCKS" || fail "cu_reads_NQ_MAX_BLOCKS" "binary would ignore the stride coupling"

# VERSION_TAG quoting (the Codon adjacent-literal trap, permanent fixture).
python3 - "$CODE" <<'EOF' && pass "source_str_literal_quote_balance" || fail "source_str_literal_quote_balance" "a module-level str literal contains an embedded double quote"
import re,sys
bad=[i for i,l in enumerate(open(sys.argv[1],encoding='utf-8'),1) if (m:=re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*:\s*str\s*=\s*"(.*)"\s*$',l.rstrip('\n'))) and '"' in m.group(1)]
sys.exit(1 if bad else 0)
EOF

if [[ "$FAIL" -gt 0 ]]; then echo; echo "===== ${REV} static summary ====="; echo "OK=$PASS FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
if [[ "$STATIC_ONLY" == "1" ]]; then echo; echo "===== ${REV} STATIC_ONLY summary ====="; echo "OK=$PASS FAIL=$FAIL"; exit 0; fi

mkdir -p "$LOGDIR"; pass "logdir_created[$LOGDIR]"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; nvidia-smi 2>&1; nvidia-smi -q -d CLOCK 2>&1
  sha256sum "$PY_SRC" "$HELPER_SRC" "$FILTER_SCRIPT" "$CU_SRC" 2>&1; echo "STAGES=$STAGES MB_PROD=$MB_PROD MB_OPT=$MB_OPT"; } > "$LOGDIR/00_env_pre.txt" 2>&1

# ---------------------------------------------------------------------
# 2. Builds
# ---------------------------------------------------------------------
banner "Building $CU_SRC and $PY_SRC"
[[ ! -x "$NVCC" ]] && NVCC="nvcc"
rm -f "$CU_BIN"; "$NVCC" -O3 -arch="$ARCH" -o "$CU_BIN" "$CU_SRC" 2>&1 | tee "$LOGDIR/01_nvcc_build.log"
[[ -x "$CU_BIN" ]] && pass "nvcc_build_succeeded" || { fail "nvcc_build_succeeded" "no binary"; exit 1; }
rm -f "$PY_BIN"; "$CODON" build -release -o "$PY_BIN" "$PY_SRC" 2>&1 | tee "$LOGDIR/02_codon_build.log"
[[ -x "$PY_BIN" ]] && pass "codon_build_succeeded" || { fail "codon_build_succeeded" "no binary -- see $LOGDIR/02_codon_build.log"; exit 1; }

# ---------------------------------------------------------------------
# 3. The four cells
# ---------------------------------------------------------------------
printf 'cell\tmode\tinput\tmax_blocks\tstride_intended\tstride_actual\ttotal_sum\tmatch\tkernel_ms\tsm_clock_mhz\ttemp_c\tstart\n' > "$TSV"
declare -A KMS
run_cell() {
  local cell="$1" mode="$2" mb="$3" label="$4"
  banner "cell $cell: bench_mode=$mode ($label) MAX_BLOCKS=$mb"
  local clk tmp start log crlog stride_int stride_act total kms match
  clk="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  tmp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
  start="$(date -Is)"; log="$LOGDIR/1${cell}_mode${mode}_mb${mb}.log"
  stride_int=$((32*mb))
  rm -f "$CRLOG_DIR/crunner_${CU_BIN}_N${NQ}.log"
  # Mode 39 sets NQ_MAX_BLOCKS itself (from argv[5]). Mode 37 does not read
  # argv[5] for the launch, so export it here for the 968 cell; for the 484
  # cell leave it unset so A is exactly the 389 invocation.
  if [[ "$mode" == "37" && "$mb" != "484" ]]; then
    NQ_MAX_BLOCKS="$mb" "./$PY_BIN" -g "$NQ" "$NQ" 32 "$mb" 0 0 7 "$mode" -d 2>&1 | tee "$log"
  else
    env -u NQ_MAX_BLOCKS "./$PY_BIN" -g "$NQ" "$NQ" 32 "$mb" 0 0 7 "$mode" -d 2>&1 | tee "$log"
  fi
  crlog="$CRLOG_DIR/crunner_${CU_BIN}_N${NQ}.log"
  cp "$crlog" "$LOGDIR/1${cell}_crunner_mode${mode}_mb${mb}.log" 2>/dev/null || true
  cp "$CRLOG_DIR/dispatch.log" "$LOGDIR/dispatch_after_${cell}.log" 2>/dev/null || true
  total="$(grep -o 'total_sum=[0-9]*' "$crlog" 2>/dev/null | head -1 | cut -d= -f2)"
  kms="$(grep -o 'kernel_ms=[0-9.]*' "$crlog" 2>/dev/null | head -1 | cut -d= -f2)"
  stride_act="$(grep -o 'stride=[0-9]*' "$crlog" 2>/dev/null | tail -1 | cut -d= -f2)"
  match=0; grep -q '\[gpu-run-correctness\] MATCH' "$crlog" 2>/dev/null && match=1
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$mode" "$label" "$mb" "$stride_int" "${stride_act:-?}" "${total:-?}" "$match" "${kms:-?}" "$clk" "$tmp" "$start" >> "$TSV"
  # gates
  if [[ "$match" -eq 1 && "${total:-}" == "$ORACLE" ]]; then pass "oracle_match[$cell]"; else fail "oracle_match[$cell]" "total_sum='${total:-<none>}' match=$match"; return 1; fi
  if grep -q '^21:' "$log" && grep '^21:' "$log" | grep -q ' ok$'; then pass "dispatcher_row_ok[$cell]"; else fail "dispatcher_row_ok[$cell]" "the N=21 table row is not 'ok' -- see $log"; return 1; fi
  if [[ "${stride_act:-}" == "$stride_int" ]]; then pass "stride_as_intended[$cell]=$stride_act"; else fail "stride_as_intended[$cell]" "C binary ran stride=${stride_act:-?}, intended $stride_int -- env coupling did not take effect (stale binary?)"; return 1; fi
  if [[ "$mode" == "39" ]]; then
    grep -q 'crunner-reordered-dispatch' "$log" && pass "mode39_reordered_path_taken[$cell]" || fail "mode39_reordered_path_taken[$cell]" "no [crunner-reordered-dispatch] line -- mode 39 did not run its own block (whitelist gate?)"
    grep -q 'maxd14only_363.bin' "$log" && grep -q 'chunkshape148' "$log" && pass "mode39_input_is_shaped[$cell]" || info "mode39_input_is_shaped[$cell]" "could not confirm the shaped input name from the console log; check $LOGDIR/dispatch_after_${cell}.log"
  fi
  KMS[$cell]="$kms"
  info "cell[$cell]" "kernel_ms=$kms stride=$stride_act"
  return 0
}
first=1
for cell in A B C D; do
  want "$cell" || continue
  (( first )) || { echo "cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; }; first=0
  case "$cell" in
    A) run_cell A 37 "$MB_PROD" raw       || break ;;
    B) run_cell B 39 "$MB_PROD" reordered || break ;;
    C) run_cell C 37 "$MB_OPT"  raw       || break ;;
    D) run_cell D 39 "$MB_OPT"  reordered || break ;;
  esac
done

# ---------------------------------------------------------------------
# 4. Evaluation against the pre-registration
# ---------------------------------------------------------------------
banner "Evaluation"
pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%+.3f",(b-a)/a*100}'; }
absdev() { awk -v a="$1" -v r="$2" 'BEGIN{d=(a-r)/r*100; printf "%.3f",(d<0?-d:d)}'; }
if [[ -n "${KMS[A]:-}" ]]; then
  d="$(absdev "${KMS[A]}" "$REF_A_MS")"; awk -v d="$d" 'BEGIN{exit !(d<=1.0)}' && pass "A_reproduces_389_anchor_within_1pct (${KMS[A]} vs $REF_A_MS, ${d}%)" || fail "A_reproduces_389_anchor_within_1pct" "${d}% off -- compare within-session only"
fi
if [[ -n "${KMS[C]:-}" ]]; then
  d="$(absdev "${KMS[C]}" "$REF_C_MS")"; awk -v d="$d" 'BEGIN{exit !(d<=1.0)}' && pass "C_reproduces_394b_968_within_1pct (${KMS[C]} vs $REF_C_MS, ${d}%)" || fail "C_reproduces_394b_968_within_1pct" "${d}% off"
fi
[[ -n "${KMS[A]:-}" && -n "${KMS[B]:-}" ]] && info "prereg_B_vs_A (reorder @484)" "$(pct "${KMS[A]}" "${KMS[B]}")%  (pre-registered -3..-6; falsified if |delta|<=1)"
[[ -n "${KMS[C]:-}" && -n "${KMS[D]:-}" ]] && info "prereg_D_vs_C (reorder @968)" "$(pct "${KMS[C]}" "${KMS[D]}")%  (pre-registered: same sign as B vs A, within +-3 points)"
[[ -n "${KMS[A]:-}" && -n "${KMS[C]:-}" ]] && info "A_vs_C (MAX_BLOCKS, raw)" "$(pct "${KMS[A]}" "${KMS[C]}")%  (394b saw -18.9)"
[[ -n "${KMS[B]:-}" && -n "${KMS[D]:-}" ]] && info "B_vs_D (MAX_BLOCKS, reordered)" "$(pct "${KMS[B]}" "${KMS[D]}")%"
[[ -n "${KMS[A]:-}" && -n "${KMS[D]:-}" ]] && info "A_vs_D (both)" "$(pct "${KMS[A]}" "${KMS[D]}")%   -> N=21 at D = $(awk -v k="${KMS[D]}" 'BEGIN{s=k/1000; printf "%d:%04.1f", int(s/60), s-60*int(s/60)}')"

{ echo "=== $REV env (post) $(date -Is) ==="; nvidia-smi -q -d CLOCK 2>&1; nvidia-smi 2>&1; } > "$LOGDIR/99_env_post.txt" 2>&1
echo; python3 - "$TSV" <<'EOF' 2>/dev/null || cat "$TSV"
import csv,sys
rows=list(csv.reader(open(sys.argv[1]),delimiter='\t'))
w=[max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
for r in rows: print('  '.join(c.ljust(w[i]) for i,c in enumerate(r)))
EOF
TARBALL="${LOGDIR}.tar.gz"
tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball_created" "tar failed -- send $LOGDIR/"
echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; echo "tsv: $TSV"; echo "tarball: $TARBALL"
if [[ "$FAIL" -gt 0 ]]; then echo "FAILED CHECKS:"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; exit 1; fi
echo "${REV} PASSED. Send $TARBALL for analysis."
exit 0
