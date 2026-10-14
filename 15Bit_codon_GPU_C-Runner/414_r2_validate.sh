#!/usr/bin/env bash
# 414_r2_validate.sh -- 414 = 413Py with chunked stream-bin scans (host-side only, GPU ZERO).
#
# r2 (2026-10-09, after the 13:06 run OK=30 FAIL=2): harness-only fixes, no .py/.cu change.
#   (1) S1 links with -lcuda (the .cu uses the driver API cuGetErrorString; 413 was built the same way).
#   (2) S0 py fingerprint strips triple-quoted text before counting code lines: suzuki appends the
#       conversation into the header docstring as well as into # lines, and prose is not code.
#   (3) SKIP_DUMPS=1 skips D21-D23/R23/M21 (already passed in the 13:06 run), so S0+S1 alone take ~2 min.
#   (4) per-line timestamps on N24a dropped (the Codon binary buffers stdout; all lines got the end time).
#
# What 414 changes (one change): dump_soa_reference_c_port() and check_required_maxd_for_N()
# read the raw constellations bin STREAM_SCAN_CHUNK (2,097,152) records at a time instead of
# all records as one List[Dict]. Output bytes / checksum / required_maxd must be IDENTICAL.
# 414_kernel_maxd16.cu is 413_kernel_maxd16.cu renamed (code region 0/0, same SASS).
#
# Gates (HARD unless marked INFO):
#   S0  static: py code-line diff vs 413Py removed=34 added=54 (comment lines excluded), REV_TAG 414,
#       VERSION_TAG has 414, "414 リビジョンノート" present, table row ./414_kernel_maxd16, STREAM_SCAN_CHUNK
#       1<<21, .cu code region sha == 413 .cu, helper scripts present.
#   S1  nvcc -O3 -arch=sm_86 -lcuda build of 414 .cu; normalized `cuobjdump -sass` == existing ./413_kernel_maxd16
#       (nvcc is deterministic: 403-r2b C0). The built binary becomes ./414_kernel_maxd16.
#   B   codon build -release of 414Py when the binary is missing or older than the .py.
#   D21/D22/D23  `./414Py -g N N 32 960 1 0 <preset> 32` re-creates constellations_N*_*.bin.soa_ref_361.bin;
#       sha256 must equal the existing file (backed up first; restored on mismatch). RSS and time recorded.
#   R23 (INFO, SKIP_R23=1 skips) the same N=23 dump with 413Py (unchunked) for the RSS comparison.
#   M21 `./414Py -g 21 21 32 960 1 0 6 34` -> [maxd-check] required_maxd=14.
#   N24 (default ON, SKIP_N24=1 skips) under `systemd-run --user --scope -p MemoryMax=$MEMMAX` when available:
#       N24a gen+dump (bench_mode=32)  -> records, gen time, dump time, RSS    (records > 0, .done == records)
#       N24b dump only (2nd bench_mode=32 run; cache hit) -> dump RSS isolated; sha == N24a dump (determinism)
#       N24c MAXDEPTH=21 413 filter   -> all kept (HARD), max_depth (INFO; pre-registered 16), histogram
#       N24d 394f sched               -> time, RSS, record count == N24c
#   Pre-registration (N=24, stated directions): records 65-110M; required_maxd 16; D23 dump RSS <= 1/4 of R23;
#   N24b dump RSS <= 8 GB. Generation time: unknown, record only. No timing claim anywhere.
#
# USAGE  SKIP_DUMPS=1 SKIP_N24=1 bash 414_r2_validate.sh 2>&1 | tee 414_r2_validate_stdout.log   # S0+S1 (+B), about 2 min
#        bash 414_r2_validate.sh                 # everything (D21-23/R23/M21 ~10 min, N24 ~8 min with the cached bin)
#        STATIC_ONLY=1 bash 414_r2_validate.sh   # S0 only
#        MEMMAX=26G    (cgroup cap for the N=24 stages; needs systemd-run --user; falls back to no cap with INFO)

set -u
REV="414_r2"
PY_SRC="${PY_SRC:-414Py_kernel_maxd16_final.py}"
PY_BASE="${PY_BASE:-413Py_kernel_maxd16_final.py}"
PY_BIN="${PY_BIN:-414Py_kernel_maxd16_final}"
PY_BASE_BIN="${PY_BASE_BIN:-413Py_kernel_maxd16_final}"
CU_SRC="${CU_SRC:-414_kernel_maxd16.cu}"
CU_BASE="${CU_BASE:-413_kernel_maxd16.cu}"
CU_BIN="${CU_BIN:-414_kernel_maxd16}"
CU_BASE_BIN="${CU_BASE_BIN:-413_kernel_maxd16}"
FILTER="${FILTER:-413_filter_maxd16_only.py}"
PERMUTE="${PERMUTE:-394f_permute_soa7.py}"
STRIP="${STRIP:-414_strip_c_comments.py}"
NVCC_FLAGS="${NVCC_FLAGS:--O3 -arch=sm_86 -lcuda}"
MEMMAX="${MEMMAX:-26G}"
EXP_REMOVED=34; EXP_ADDED=54
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${REV}_validate_${TS}}"
PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
summary_exit() {
  { echo "=== $REV env (post) $(date -Is) ==="; free -g; df -h . ; } > "$LOGDIR/99_env_post.txt" 2>&1
  TARBALL="${LOGDIR}.tar.gz"; tar czf "$TARBALL" "$LOGDIR" 2>/dev/null && pass "tarball_created[$TARBALL]" || info "tarball" "tar failed"
  echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done
  [[ "$FAIL" -gt 0 ]] && exit 1; exit 0; }
mkdir -p "$LOGDIR"
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; free -g; df -h .; nvcc --version 2>&1 | tail -1; codon --version 2>&1; cuobjdump --version 2>&1 | tail -1; } > "$LOGDIR/00_env_pre.txt" 2>&1

# ---------------------------------------------------------------- S0 static
for f in "$PY_SRC" "$PY_BASE" "$CU_SRC" "$CU_BASE" "$FILTER" "$STRIP"; do
  [[ -s "$f" ]] && pass "present[$f]" || { fail "present" "$f missing"; summary_exit; }
done
read -r REMOVED ADDED < <(python3 - "$PY_BASE" "$PY_SRC" <<'EOF'
import sys,difflib,re
def code(p):
    s=re.sub(r'"""[\s\S]*?"""','""',open(p,encoding='utf-8').read())  # triple-quoted prose is not code
    return [l for l in s.split('\n') if l.strip() and not l.lstrip().startswith('#')]
a,b=code(sys.argv[1]),code(sys.argv[2])
d=list(difflib.unified_diff(a,b,lineterm='',n=0))
print(sum(1 for l in d if l.startswith('-') and not l.startswith('---')), sum(1 for l in d if l.startswith('+') and not l.startswith('+++')))
EOF
)
[[ "$REMOVED" == "$EXP_REMOVED" && "$ADDED" == "$EXP_ADDED" ]] && pass "S0_py_code_diff removed=$REMOVED added=$ADDED" || fail "S0_py_code_diff" "removed=$REMOVED added=$ADDED (expected $EXP_REMOVED/$EXP_ADDED)"
grep -q '^REV_TAG:str="414"' "$PY_SRC" && pass "S0_py_REV_TAG_414" || fail "S0_py_REV_TAG" "$(grep -o '^REV_TAG:str=.*' "$PY_SRC")"
grep -q '^VERSION_TAG:str="414 ' "$PY_SRC" && pass "S0_py_VERSION_TAG_414" || fail "S0_py_VERSION_TAG" "missing"
grep -q '^# 414 リビジョンノート' "$PY_SRC" && pass "S0_py_note_present" || fail "S0_py_note" "414 リビジョンノート not found"
grep -q 'CRunnerEntry(15,"./414_kernel_maxd16"' "$PY_SRC" && pass "S0_py_table_row_414" || fail "S0_py_table_row" "missing"
grep -q '^STREAM_SCAN_CHUNK:int=1<<21$' "$PY_SRC" && pass "S0_py_STREAM_SCAN_CHUNK_2097152" || fail "S0_py_STREAM_SCAN_CHUNK" "$(grep -o '^STREAM_SCAN_CHUNK.*' "$PY_SRC")"
python3 - "$PY_SRC" <<'EOF2' && pass "S0_py_both_scans_chunked" || fail "S0_py_scans" "dump_soa_reference_c_port / check_required_maxd_for_N still read the whole file, or lack STREAM_SCAN_CHUNK"
import sys
s=open(sys.argv[1],encoding='utf-8').read()
ok=True
for name in ('dump_soa_reference_c_port','check_required_maxd_for_N'):
    i=s.index('def '+name+'('); j=s.index('\ndef ',i+10); body=s[i:j]
    ok &= ('read_constellations_bin_range(fname,0,' not in body) and ('STREAM_SCAN_CHUNK' in body)
sys.exit(0 if ok else 1)
EOF2
CU_SHA_NEW="$(python3 "$STRIP" "$CU_SRC" | sha256sum | cut -c1-16)"; CU_SHA_OLD="$(python3 "$STRIP" "$CU_BASE" | sha256sum | cut -c1-16)"
[[ "$CU_SHA_NEW" == "$CU_SHA_OLD" ]] && pass "S0_cu_code_region_identical sha=$CU_SHA_NEW" || fail "S0_cu_code_region" "$CU_SHA_NEW != $CU_SHA_OLD"
grep -q 'rev414 --' "$CU_SRC" && pass "S0_cu_header_note" || fail "S0_cu_header_note" "missing"
[[ "${STATIC_ONLY:-0}" == "1" ]] && summary_exit

# ---------------------------------------------------------------- S1 nvcc + SASS
sass_norm() { cuobjdump -sass "$1" 2>/dev/null | sed -E 's#/\*[0-9a-f]{4,}\*/##g; /^[[:space:]]*\/\* 0x[0-9a-f]+ \*\/[[:space:]]*$/d; s/[[:space:]]+$//' ; }
if [[ -x "$CU_BASE_BIN" ]]; then
  if nvcc $NVCC_FLAGS -o "$CU_BIN" "$CU_SRC" > "$LOGDIR/1_nvcc.log" 2>&1; then
    pass "S1_nvcc_build[$CU_BIN]"
    sass_norm "$CU_BIN" > "$LOGDIR/1_sass_414.txt"; sass_norm "$CU_BASE_BIN" > "$LOGDIR/1_sass_413.txt"
    if cmp -s "$LOGDIR/1_sass_414.txt" "$LOGDIR/1_sass_413.txt"; then pass "S1_sass_identical_to_413 ($(wc -l < "$LOGDIR/1_sass_414.txt") lines)"
    else fail "S1_sass" "differs from ./$CU_BASE_BIN (set NVCC_FLAGS to the flags 413 was built with; diff in $LOGDIR/1_sass.diff)"; diff "$LOGDIR/1_sass_413.txt" "$LOGDIR/1_sass_414.txt" | head -40 > "$LOGDIR/1_sass.diff"; fi
  else fail "S1_nvcc_build" "$(tail -3 "$LOGDIR/1_nvcc.log" | tr '\n' ' ')"; fi
else fail "S1_base_binary" "./$CU_BASE_BIN missing"; fi

# ---------------------------------------------------------------- B codon build
if [[ ! -x "$PY_BIN" || "$PY_SRC" -nt "$PY_BIN" ]]; then
  if codon build -release -o "$PY_BIN" "$PY_SRC" > "$LOGDIR/2_codon.log" 2>&1; then pass "B_codon_build[$PY_BIN]"; else fail "B_codon_build" "$(tail -5 "$LOGDIR/2_codon.log" | tr '\n' ' ')"; summary_exit; fi
else pass "B_py_binary_present[$PY_BIN]"; fi

TIMEV=""; [[ -x /usr/bin/time ]] && TIMEV="/usr/bin/time -v"
rss_of() { grep -o 'Maximum resident set size (kbytes): [0-9]*' "$1" 2>/dev/null | awk '{printf "%.2f", $NF/1048576}'; }  # GB
wall_of() { grep -o 'Elapsed (wall clock) time (h:mm:ss or m:ss): .*' "$1" 2>/dev/null | sed 's/.*: //'; }

# ---------------------------------------------------------------- D21/D22/D23 (+R23)
dump_cell() {  # cell bin N preset tag
  local cell="$1" bin="$2" N="$3" p="$4" tag="$5"
  local stream="constellations_N${N}_${p}.bin" ref="constellations_N${N}_${p}.bin.soa_ref_361.bin"
  [[ -s "$stream" && -s "$stream.done" ]] || { fail "${cell}_input" "$stream(.done) missing"; return; }
  local sha_old="(none)" size_old=0
  if [[ -s "$ref" ]]; then sha_old="$(sha256sum "$ref" | cut -c1-16)"; size_old="$(stat -c %s "$ref")"; mv "$ref" "$ref.bak414"
  else info "${cell}_reference" "$ref not present: the dump is produced and recorded, identity cannot be gated for this N"; fi
  $TIMEV -o "$LOGDIR/${cell}_time.txt" "./$bin" -g "$N" "$N" 32 960 1 0 "$p" 32 > "$LOGDIR/${cell}_$tag.stdout" 2>&1; local rc=$?
  local sha_new="?"; [[ -s "$ref" ]] && sha_new="$(sha256sum "$ref" | cut -c1-16)"
  local rss; rss="$(rss_of "$LOGDIR/${cell}_time.txt")"; local wall; wall="$(wall_of "$LOGDIR/${cell}_time.txt")"
  if [[ "$sha_old" == "(none)" ]]; then
    [[ "$rc" == "0" && -s "$ref" ]] && info "${cell}_dump_created" "sha=$sha_new bytes=$(stat -c %s "$ref")" || fail "${cell}_dump" "rc=$rc, no output"
  elif [[ "$rc" == "0" && "$sha_new" == "$sha_old" ]]; then
    rm -f "$ref.bak414"; pass "${cell}_dump_identical[$bin N=$N] sha=$sha_new bytes=$size_old"
  else
    mv -f "$ref.bak414" "$ref"; fail "${cell}_dump" "rc=$rc sha_new=$sha_new sha_old=$sha_old (original restored)"
  fi
  info "${cell}_rss_wall" "max_rss_gb=${rss:-?} wall=${wall:-?} $(grep -o '\[soa-ref-dump-done\].*' "$LOGDIR/${cell}_$tag.stdout" | head -1)"
  echo "${rss:-?}" > "$LOGDIR/${cell}_rss_gb.txt"
}
if [[ "${SKIP_DUMPS:-0}" == "1" ]]; then info "D21_D23_R23_M21" "skipped (SKIP_DUMPS=1; they passed in 414_validate_20261009_130611)"; else
dump_cell D21 "$PY_BIN" 21 6 414
dump_cell D22 "$PY_BIN" 22 7 414
dump_cell D23 "$PY_BIN" 23 7 414
if [[ "${SKIP_R23:-0}" != "1" ]]; then
  if [[ -x "$PY_BASE_BIN" ]]; then
    dump_cell R23 "$PY_BASE_BIN" 23 7 413
    R="$(cat "$LOGDIR/R23_rss_gb.txt" 2>/dev/null)"; D="$(cat "$LOGDIR/D23_rss_gb.txt" 2>/dev/null)"
    if [[ -n "$R" && -n "$D" && "$R" != "?" && "$D" != "?" ]]; then
      awk -v r="$R" -v d="$D" 'BEGIN{exit !(d*4<=r)}' && pass "P1_D23_rss_le_quarter_of_R23 (414 ${D} GB vs 413 ${R} GB, ratio $(awk -v r="$R" -v d="$D" 'BEGIN{printf "%.3f", d/r}'))" \
        || info "P1_D23_rss" "stated <=1/4 NOT met: 414 ${D} GB vs 413 ${R} GB (information; D23 identity is the gate)"
    fi
  else info "R23" "skipped: ./$PY_BASE_BIN missing"; fi
fi

# ---------------------------------------------------------------- M21
$TIMEV -o "$LOGDIR/M21_time.txt" "./$PY_BIN" -g 21 21 32 960 1 0 6 34 > "$LOGDIR/M21.stdout" 2>&1
grep -q '\[maxd-check\] N=21 records=2025282 required_maxd=14 ' "$LOGDIR/M21.stdout" && pass "M21_required_maxd_14" || fail "M21_required_maxd" "$(grep -o '\[maxd-check\].*' "$LOGDIR/M21.stdout" | head -1)"
fi

# ---------------------------------------------------------------- N24
[[ "${SKIP_N24:-0}" == "1" ]] && { info "N24" "skipped (SKIP_N24=1)"; summary_exit; }
FREE_GB="$(df -BG . | awk 'NR==2{gsub("G","",$4); print $4}')"
[[ "${FREE_GB:-0}" -ge 15 ]] && pass "N24_disk_free_${FREE_GB}G" || { fail "N24_disk" "only ${FREE_GB} GB free (need 15)"; summary_exit; }
CAP=""
if command -v systemd-run >/dev/null 2>&1 && systemd-run --user --scope -q -p MemoryMax="$MEMMAX" true >/dev/null 2>&1; then
  CAP="systemd-run --user --scope -q -p MemoryMax=$MEMMAX"; info "N24_memcap" "$CAP"
else info "N24_memcap" "systemd-run --user --scope unavailable; running WITHOUT a memory cap (watch free -g)"; fi
STREAM24="constellations_N24_7.bin"; REF24="$STREAM24.soa_ref_361.bin"; ALL24="$REF24.all_414.bin"; SCHED24="$ALL24.sched394f.bin"
if [[ -s "$STREAM24" && -s "$STREAM24.done" ]]; then info "N24a" "$STREAM24 already present ($(( $(stat -c %s "$STREAM24") / 16 )) records): generation is skipped by the cache, timings below are dump-only"; fi
T0="$(date +%s)"; info "N24a_start" "$(date -Is) gen+dump: $CAP ./$PY_BIN -g 24 24 32 960 1 0 7 32"
$CAP $TIMEV -o "$LOGDIR/N24a_time.txt" "./$PY_BIN" -g 24 24 32 960 1 0 7 32 > "$LOGDIR/N24a.stdout" 2>&1; rc=$?
T1="$(date +%s)"
REC24=0; [[ -s "$STREAM24" ]] && REC24=$(( $(stat -c %s "$STREAM24") / 16 )); DONE24="$(cat "$STREAM24.done" 2>/dev/null | tr -dc '0-9')"
[[ "$rc" == "0" && "$REC24" -gt 0 && "$DONE24" == "$REC24" ]] && pass "N24a_gen_dump_rc0 records=$REC24 (.done=$DONE24)" || fail "N24a_gen_dump" "rc=$rc records=$REC24 done=${DONE24:-?} ($(tail -2 "$LOGDIR/N24a.stdout" | tr '\n' ' '))"
info "N24a_times" "total=$((T1-T0))s max_rss_gb=$(rss_of "$LOGDIR/N24a_time.txt") $(grep -o 'stream-build-summary.*' "$LOGDIR/N24a.stdout" | head -1)"
if [[ "$REC24" -ge 65000000 && "$REC24" -le 110000000 ]]; then pass "P2_records_in_65M_110M ($REC24)"; else info "P2_records" "stated 65-110M NOT met: $REC24 (information)"; fi
[[ -s "$REF24" ]] && info "N24a_dump" "$REF24 bytes=$(stat -c %s "$REF24") records=$(( $(stat -c %s "$REF24") / 40 )) sha=$(sha256sum "$REF24" | cut -c1-16)"
# N24b: dump only (stream cache hit), RSS of the chunked dump in isolation, determinism
if [[ "$rc" == "0" && -s "$REF24" ]]; then
  SHA_A="$(sha256sum "$REF24" | cut -c1-16)"; mv "$REF24" "$REF24.n24a"
  $CAP $TIMEV -o "$LOGDIR/N24b_time.txt" "./$PY_BIN" -g 24 24 32 960 1 0 7 32 > "$LOGDIR/N24b.stdout" 2>&1; rcb=$?
  SHA_B="?"; [[ -s "$REF24" ]] && SHA_B="$(sha256sum "$REF24" | cut -c1-16)"
  RSSB="$(rss_of "$LOGDIR/N24b_time.txt")"
  if [[ "$rcb" == "0" && "$SHA_A" == "$SHA_B" ]]; then rm -f "$REF24.n24a"; pass "N24b_dump_deterministic sha=$SHA_B"; else mv -f "$REF24.n24a" "$REF24"; fail "N24b_dump" "rc=$rcb sha_a=$SHA_A sha_b=$SHA_B"; fi
  info "N24b_dump_only" "max_rss_gb=${RSSB:-?} wall=$(wall_of "$LOGDIR/N24b_time.txt")"
  awk -v r="${RSSB:-999}" 'BEGIN{exit !(r<=8.0)}' && pass "P3_dump_rss_le_8GB (${RSSB} GB)" || info "P3_dump_rss" "stated <=8 GB NOT met: ${RSSB} GB (information)"
  # N24c: filter, keep everything, histogram -> required_maxd
  MAXDEPTH=21 $CAP $TIMEV -o "$LOGDIR/N24c_time.txt" python3 "$FILTER" "$REF24" "$ALL24" > "$LOGDIR/N24c.stdout" 2>&1; rcc=$?
  FD="$(grep -o '\[filter-done\].*' "$LOGDIR/N24c.stdout" | head -1)"; IN="$(grep -o 'records_in=[0-9]*' <<<"$FD" | cut -d= -f2)"; KEPT="$(grep -o 'records_kept([^)]*)=[0-9]*' <<<"$FD" | sed 's/.*=//')"; MD="$(grep -o 'max_depth=[0-9]*' <<<"$FD" | cut -d= -f2)"
  [[ "$rcc" == "0" && -n "$IN" && "$IN" == "$KEPT" && "$IN" == "$REC24" ]] && pass "N24c_filter_all_kept records=$IN" || fail "N24c_filter" "rc=$rcc $FD"
  info "N24c_required_maxd" "max_depth=${MD:-?} (pre-registered 16; >=17 means the N=24 kernel needs >15 frames) wall=$(wall_of "$LOGDIR/N24c_time.txt") max_rss_gb=$(rss_of "$LOGDIR/N24c_time.txt")"
  info "N24c_histogram" "$(grep -o '\[filter-depth-histogram\].*' "$LOGDIR/N24c.stdout" | head -1)"
  [[ "${MD:-0}" == "16" ]] && pass "P4_required_maxd_16" || info "P4_required_maxd" "stated 16 NOT met: ${MD:-?} (information)"
  # N24d: 394f sched
  if [[ -s "$PERMUTE" && "$rcc" == "0" ]]; then
    $CAP $TIMEV -o "$LOGDIR/N24d_time.txt" python3 "$PERMUTE" sched "$ALL24" "$SCHED24" > "$LOGDIR/N24d.stdout" 2>&1; rcd=$?
    SREC=0; [[ -s "$SCHED24" ]] && SREC=$(( $(stat -c %s "$SCHED24") / 28 ))
    [[ "$rcd" == "0" && "$SREC" == "$REC24" ]] && pass "N24d_sched_records_$SREC" || fail "N24d_sched" "rc=$rcd records=$SREC ($(tail -2 "$LOGDIR/N24d.stdout" | tr '\n' ' '))"
    info "N24d_sched" "wall=$(wall_of "$LOGDIR/N24d_time.txt") max_rss_gb=$(rss_of "$LOGDIR/N24d_time.txt") $(grep -o '\[permute-done\].*' "$LOGDIR/N24d.stdout" | head -1)"
  else info "N24d" "skipped ($PERMUTE missing or filter failed)"; fi
fi
info "N24_files" "$(ls -la "$STREAM24" "$REF24" "$ALL24" "$SCHED24" 2>/dev/null | awk '{print $5, $9}' | tr '\n' ';')"
summary_exit
