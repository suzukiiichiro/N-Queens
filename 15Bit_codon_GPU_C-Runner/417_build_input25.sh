#!/usr/bin/env bash
# 417_build_input25.sh -- builds the N=25 CRunner input, GPU zero, every Codon stage under a cgroup (rule 33).
#
#   N25a  stream      ./417Py -g 25 25 32 960 1 0 7 11   -> constellations_N25_7.bin (+ .done); preset_queens MUST be 7
#   N25b  maxd check  ./417Py -g 25 25 32 960 1 0 7 34   -> [maxd-check-done] required_maxd=17                      HARD
#   N25c  361 dump    ./417Py -g 25 25 32 960 1 0 7 32   -> constellations_N25_7.bin.soa_ref_361.bin (40 B/record)
#   N25d  bitcheck    python3 416_bitcheck.py <dump> 25  -> I1..I6 all ok (col/free constant bits, rd <= bit 30,
#                                                           free == bm & ~(ld|rd|col): the 425 preconditions)     HARD
#   N25e  filter      MAXDEPTH=17 python3 413_filter_maxd16_only.py <dump> <dump>.d17only_417.bin  (28 B/record, histogram max 17)
#   N25f  sched       python3 394f_permute_soa7.py sched <filtered> <filtered>.sched394f.bin        (RSS ~14 GB expected)
#
# Every stage is skipped when its output already exists with the right size, so the script can be re-run.
# Expected sizes for ~1e8 records: dump ~4 GB, filtered ~2.8 GB, sched ~2.8 GB (plus the stream bin). Needs ~12 GB free.
# The stream generation time is unknown (N=24's was not timed); each stage prints its wall time.
#
# USAGE   nohup bash 417_build_input25.sh > 417_build_input25_stdout.log 2>&1 &

set -u
REV="417_input25"
PY_SRC="${PY_SRC:-417Py_kernel_maxd16_final.py}"
PY_BIN="${PY_BIN:-${PY_SRC%.py}}"
CODON="${CODON:-codon}"
FILTER="${FILTER:-413_filter_maxd16_only.py}"
PERMUTE="${PERMUTE:-394f_permute_soa7.py}"
BITCHECK="${BITCHECK:-416_bitcheck.py}"
STREAM25="${STREAM25:-constellations_N25_7.bin}"
MEM_MAX="${MEM_MAX:-26G}"
EXPECT_MAXD="${EXPECT_MAXD:-17}"
TS="$(date +%Y%m%d_%H%M%S)"; LOGDIR="${LOGDIR:-${REV}_${TS}}"; mkdir -p "$LOGDIR"
PASS=0; FAIL=0; declare -a FAILED_CHECKS=()
pass() { PASS=$((PASS+1)); echo "OK    $1"; }
fail() { FAIL=$((FAIL+1)); FAILED_CHECKS+=("$1"); echo "FAIL  $1: $2"; }
info() { echo "INFO  $1: $2"; }
summary_exit() { echo; echo "===== ${REV} summary ====="; echo "OK=$PASS  FAIL=$FAIL"; for c in "${FAILED_CHECKS[@]}"; do echo "  - $c"; done; [[ "$FAIL" -gt 0 ]] && exit 1; exit 0; }
stage() { local name="$1"; shift; local t0; t0=$(date +%s); info "${name}_start" "$(date -Is)  $*"; "$@"; local rc=$?; info "${name}_end" "rc=$rc $(( $(date +%s)-t0 )) s"; return $rc; }
# cgroup wrapper (rule 33): systemd-run --scope; falls back to no wrapper with a warning (never ulimit -v, rule 26)
if WRAPPED="$(systemd-run --scope -p MemoryMax="$MEM_MAX" --quiet true 2>&1)"; then WRAP=(systemd-run --scope -p "MemoryMax=$MEM_MAX" --quiet); pass "cgroup_wrapper_available (MemoryMax=$MEM_MAX)"
elif WRAPPED="$(systemd-run --user --scope -p MemoryMax="$MEM_MAX" --quiet true 2>&1)"; then WRAP=(systemd-run --user --scope -p "MemoryMax=$MEM_MAX" --quiet); pass "cgroup_wrapper_available (--user, MemoryMax=$MEM_MAX)"
else WRAP=(); info "cgroup_wrapper" "systemd-run unavailable ($WRAPPED) -- running WITHOUT a memory ceiling (RAM 30 GB, 394f expected ~14 GB)"; fi

for f in "$PY_SRC" "$FILTER" "$PERMUTE" "$BITCHECK" rev386_validation_helpers.py; do [[ -f "$f" ]] && pass "file_present[$f]" || fail "file_present[$f]" "missing"; done
[[ "$FAIL" -gt 0 ]] && summary_exit
FREE_GB="$(df -BG . | awk 'NR==2{gsub("G","",$4); print $4}')"; [[ "${FREE_GB:-0}" -ge 15 ]] && pass "disk_free_${FREE_GB}G" || { fail "disk" "only ${FREE_GB} GB free (need ~12 GB for the N=25 files)"; summary_exit; }
{ echo "=== $REV env (pre) $(date -Is) ==="; uname -a; free -g; df -h .; } > "$LOGDIR/00_env_pre.txt" 2>&1
if [[ ! -x "$PY_BIN" ]]; then "$CODON" build -release "$PY_SRC" > "$LOGDIR/06_codon.log" 2>&1; fi
[[ -x "$PY_BIN" ]] && pass "py_binary[$PY_BIN]" || { fail "codon_build" "$(tail -3 "$LOGDIR/06_codon.log")"; summary_exit; }
grep -q 'REV_TAG:str="417"' "$PY_SRC" && pass "py_is_417" || fail "py_is_417" "wrong REV_TAG"

# ---- N25a: stream
DUMP25="${STREAM25}.soa_ref_361.bin"; FILT25="${DUMP25}.d17only_417.bin"; SCHED25="${FILT25}.sched394f.bin"
done_count() { [[ -f "$1.done" ]] && tr -dc '0-9' < "$1.done" || echo 0; }
if [[ -s "$STREAM25" && "$(done_count "$STREAM25")" -gt 0 ]]; then info "N25a_stream" "reuse $STREAM25 (done=$(done_count "$STREAM25"))"
else
  stage N25a "${WRAP[@]}" "./$PY_BIN" -g 25 25 32 960 1 0 7 11 > "$LOGDIR/1_N25a_stream.stdout" 2>&1
  grep -E 'stream-build-summary|stream-cache' "$LOGDIR/1_N25a_stream.stdout" | tail -2
fi
grep -qE 'preset_queens=7' "$LOGDIR/1_N25a_stream.stdout" 2>/dev/null && pass "N25a_preset_queens_7" || { [[ -s "$STREAM25" ]] && info "N25a_preset" "stream reused; preset is in the file name ($STREAM25)" || { fail "N25a_preset_queens_7" "$(grep -o 'preset_queens=[0-9]*' "$LOGDIR/1_N25a_stream.stdout" | head -1)"; summary_exit; }; }
[[ -s "$STREAM25" ]] && pass "N25a_stream_present ($(stat -c %s "$STREAM25") B, done=$(done_count "$STREAM25"))" || { fail "N25a_stream" "$STREAM25 missing"; summary_exit; }

# ---- N25b: required_maxd (chunked scan of the stream, 414)
stage N25b "${WRAP[@]}" "./$PY_BIN" -g 25 25 32 960 1 0 7 34 > "$LOGDIR/2_N25b_maxd.stdout" 2>&1
MAXD="$(grep -o 'required_maxd=[0-9]*' "$LOGDIR/2_N25b_maxd.stdout" | head -1 | cut -d= -f2)"; REC25="$(grep -o '\[maxd-check-done\].*records=[0-9]*' "$LOGDIR/2_N25b_maxd.stdout" | grep -o 'records=[0-9]*' | cut -d= -f2)"
grep '\[maxd-check-done\]' "$LOGDIR/2_N25b_maxd.stdout"
[[ "$MAXD" == "$EXPECT_MAXD" ]] && pass "N25b_required_maxd_${EXPECT_MAXD} (records=$REC25)" || { fail "N25b_required_maxd" "required_maxd=${MAXD:-?} != $EXPECT_MAXD -- PACK425_ANCESTOR (16 frames) does not match; stop"; summary_exit; }

# ---- N25c: 361 dump
if [[ -s "$DUMP25" && $(( $(stat -c %s "$DUMP25") / 40 )) -eq "$REC25" ]]; then info "N25c_dump" "reuse $DUMP25"
else stage N25c "${WRAP[@]}" "./$PY_BIN" -g 25 25 32 960 1 0 7 32 > "$LOGDIR/3_N25c_dump.stdout" 2>&1; grep '\[soa-ref-dump-done\]' "$LOGDIR/3_N25c_dump.stdout"; fi
[[ -s "$DUMP25" && $(( $(stat -c %s "$DUMP25") / 40 )) -eq "$REC25" && $(( $(stat -c %s "$DUMP25") % 40 )) -eq 0 ]] && pass "N25c_dump_records_${REC25}" || { fail "N25c_dump" "$DUMP25 size $(stat -c %s "$DUMP25" 2>/dev/null) != ${REC25} x 40"; summary_exit; }

# ---- N25d: bit invariants (425 preconditions) over every record
stage N25d python3 "$BITCHECK" "$DUMP25" 25 > "$LOGDIR/4_N25d_bitcheck.log" 2> "$LOGDIR/4_N25d_bitcheck.stderr"; rc=$?
grep -E '^\[bitcheck-(field|ok|FAIL|i6|info|done)' "$LOGDIR/4_N25d_bitcheck.log"
[[ "$rc" == "0" ]] && grep -q 'failures=0' "$LOGDIR/4_N25d_bitcheck.log" && pass "N25d_bitcheck_I1_I6_all_ok" || { fail "N25d_bitcheck" "$(grep 'bitcheck-FAIL' "$LOGDIR/4_N25d_bitcheck.log" | tr '\n' ' ')"; summary_exit; }

# ---- N25e: filter (depth <= 17, histogram)
if [[ -s "$FILT25" && $(( $(stat -c %s "$FILT25") / 28 )) -eq "$REC25" ]]; then info "N25e_filter" "reuse $FILT25"
else stage N25e env MAXDEPTH=17 python3 "$FILTER" "$DUMP25" "$FILT25" > "$LOGDIR/5_N25e_filter.log" 2>&1; fi
grep -E '^\[filter-' "$LOGDIR/5_N25e_filter.log" 2>/dev/null | head -5
[[ -s "$FILT25" && $(( $(stat -c %s "$FILT25") / 28 )) -eq "$REC25" && $(( $(stat -c %s "$FILT25") % 28 )) -eq 0 ]] && pass "N25e_filtered_records_${REC25} (every record kept)" || { fail "N25e_filter" "$FILT25 size $(stat -c %s "$FILT25" 2>/dev/null) != ${REC25} x 28 (see $LOGDIR/5_N25e_filter.log)"; summary_exit; }
HMAX="$(grep -o '\[filter-depth-histogram\].*' "$LOGDIR/5_N25e_filter.log" 2>/dev/null | grep -oE '(^|[^0-9])1[0-9]:[0-9]+' | cut -d: -f1 | tr -dc '0-9\n' | sort -n | tail -1)"
[[ -n "$HMAX" ]] && info "N25e_histogram_max_depth" "$HMAX (expected $EXPECT_MAXD; the filter aborts above MAXDEPTH=17)" || info "N25e_histogram" "$(grep -o '\[filter-depth-histogram\].*' "$LOGDIR/5_N25e_filter.log" 2>/dev/null | head -1)"

# ---- N25f: 394f schedule order
if [[ -s "$SCHED25" && $(( $(stat -c %s "$SCHED25") / 28 )) -eq "$REC25" ]]; then info "N25f_sched" "reuse $SCHED25"
else stage N25f "${WRAP[@]}" python3 "$PERMUTE" sched "$FILT25" "$SCHED25" > "$LOGDIR/6_N25f_sched.log" 2>&1; grep '\[permute-done\]' "$LOGDIR/6_N25f_sched.log"; fi
[[ -s "$SCHED25" && $(( $(stat -c %s "$SCHED25") / 28 )) -eq "$REC25" ]] && pass "N25f_sched_records_${REC25} ($SCHED25)" || { fail "N25f_sched" "$SCHED25 size $(stat -c %s "$SCHED25" 2>/dev/null) != ${REC25} x 28"; summary_exit; }

{ echo "=== $REV env (post) $(date -Is) ==="; free -g; df -h .; ls -la "$STREAM25"* ; } > "$LOGDIR/99_env_post.txt" 2>&1
info "N25_input_ready" "records=$REC25 required_maxd=$MAXD sched=$SCHED25 disk_free_now=$(df -BG . | awk 'NR==2{print $4}')"
summary_exit
