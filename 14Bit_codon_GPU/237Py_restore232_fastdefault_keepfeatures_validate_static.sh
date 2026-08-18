#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# NQ_UPDATE_MEMO
# 237 static validation: 236Py汎用本線を親に、bare -g がsplit145 mode31を
# A10G既定として使うことを検査する。N21専用化、cache-hot専用化、
# helper分離は不可。N=5..27、-c/-g bare range、cache生成、MAXD fallback、
# bench_mode 28/29/31、worker splitを残す。
# =============================================================================

SRC=${SRC:-./237Py_restore232_fastdefault_keepfeatures_probe.py}
SUMMARY=${SUMMARY:-./237Py_restore232_fastdefault_keepfeatures_static_summary.tsv}
printf 'check\texpected\tactual\tstatus\n' > "$SUMMARY"
failures=0

ok_line() { printf '%s\t%s\t%s\tOK\n' "$1" "$2" "$3" >> "$SUMMARY"; }
fail_line() { printf '%s\t%s\t%s\tFAIL\n' "$1" "$2" "$3" >> "$SUMMARY"; failures=$((failures+1)); }

if [[ ! -f "$SRC" ]]; then
  echo "[error] source not found: $SRC" >&2
  exit 66
fi

if grep -q '237 restore232-fastdefault-keepfeatures' "$SRC"; then ok_line source_version_tag '237 restore232-fastdefault-keepfeatures' present; else fail_line source_version_tag '237 restore232-fastdefault-keepfeatures' missing; fi
if grep -q 'if __name__=="__main__":' "$SRC" && grep -q '  main()' "$SRC"; then ok_line source_main_entry '__main__ calls main()' present; else fail_line source_main_entry '__main__ calls main()' missing; fi

python3 - "$SRC" <<'PY' > /tmp/237_static_checks.$$ || true
import sys,re
s=open(sys.argv[1],encoding='utf-8').read()
checks={
  'default_range_5_23':'DEFAULT_RANGE_NMIN:int=5' in s and 'DEFAULT_RANGE_NMAX_EXCLUSIVE:int=24' in s,
  'bare_c_range_kept':'if arg == "-c"' in s and 'CPU mode selected' in s and 'DEFAULT_RANGE_NMIN:int=5' in s and 'DEFAULT_RANGE_NMAX_EXCLUSIVE:int=24' in s,
  'bare_g_a10g_kept':'if arg == "-g"' in s and 'gpu_block=A10G_FINAL_DEFAULT_BLOCK' in s and 'bench_mode=A10G_FINAL_DEFAULT_BENCH_MODE' in s,
  'bare_g_fastdefault_mode31':'A10G_FINAL_DEFAULT_BENCH_MODE:int=31' in s and 'bare -g defaults to split145 mode31' in s,
  'a10g_best_params':'A10G_FINAL_DEFAULT_BLOCK:int=32' in s and 'A10G_FINAL_DEFAULT_MAX_BLOCKS:int=484' in s and 'A10G_FINAL_DEFAULT_PRESET:int=7' in s and 'A10G_FINAL_DEFAULT_BENCH_MODE:int=31' in s and 'A10G_FINAL_DEFAULT_REORDER_WINDOW_MULT:int=8' in s and 'A10G_FINAL_DEFAULT_REORDER_PHASE_JUMP:int=7' in s and 'A10G_FINAL_DEFAULT_BROADMARK_VARIANT:int=2' in s,
  'gpu_range_5_27':'nmin<5 or nmax>28' in s and 'supports N=5..27' in s,
  'dynamic_preset_n27':'elif N>=25 and N<=27:' in s and 'return 8' in s,
  'fallback_kernels':'def kernel_dfs_iter_gpu_maxd14(' in s and 'def kernel_dfs_iter_gpu_maxd16(' in s and 'def kernel_dfs_iter_gpu_maxd18(' in s and 'def kernel_dfs_iter_gpu_maxd20(' in s and 'def kernel_dfs_iter_gpu_maxd21(' in s,
  'maxd_dispatch_wrapper':'def launch_kernel_dfs_iter_gpu_static_maxd(' in s and 'select_static_maxd' in s and 'max_schedule_depth_of_tasks' in s,
  'cache_generation':'def ensure_constellations_bin_stream(' in s and 'def gen_constellations_stream_to_bin(' in s and 'def build_broad_markdist_tail_reordered_bin(' in s and 'def build_chunkshape148_reordered_bin(' in s,
  'broadmarktail_modes':'bench_mode==28' in s and 'bench_mode==29' in s and 'broadmarktail-reorder-sim-only' in s and 'broadmarktail-reorder-gpu' in s,
  'split145_mode31':'bench_mode==31' in s and 'exec_solutions_gpu_bin_stream_split145' in s and 'split237_full' in s,
  'worker_args':'worker_id=int(sys.argv[13])' in s and 'worker_count=int(sys.argv[14])' in s and 'worker_split : worker=' in s and 'worker-done' in s,
  'rootrestlate_futuremask_nosibling':'root_rest:u32=cur_avail&(cur_avail-u32(1))' in s and 'root_after_second:u32=root_rest^root_second' in s and 'future_check_mask:u32=u32(0)' in s and 'avail[save_sp]=cur_avail|(u32(cur_depth)<<u32(27))' in s,
  'not_cachehot_n21_only':'cachehot' not in s and 'fixed to: -g 21 21' not in s,
  'cpu_path_kept':'def dfs_iter(' in s and 'def dfs(' in s and 'CPU mode selected' in s,
}
for k,v in checks.items():
    print(f'{k}\tOK' if v else f'{k}\tFAIL')
sys.exit(0 if all(checks.values()) else 1)
PY
py_status=0
while IFS=$'\t' read -r name status; do
  [[ -z "$name" ]] && continue
  if [[ "$status" == OK ]]; then ok_line "$name" present present; else fail_line "$name" present missing; fi
done < /tmp/237_static_checks.$$
rm -f /tmp/237_static_checks.$$

actual_split=$(python3 - "$SRC" <<'PY'
import re,sys
s=open(sys.argv[1],encoding='utf-8').read()
tags=sorted(set(re.findall(r'split\d+',s)))
print(' '.join(tags) if tags else 'none')
sys.exit(0 if 'split237' in tags and 'split232' not in tags and 'split234' not in tags and 'split235' not in tags and 'split236' not in tags else 1)
PY
) && split_status=OK || split_status=FAIL
if [[ "$split_status" == OK ]]; then ok_line source_split_tag 'split237 active; no split232/234/235/236' "$actual_split"; else fail_line source_split_tag 'split237 active; no split232/234/235/236' "$actual_split"; fi

echo "================================================================"
echo "[static-summary]"
cat "$SUMMARY"
echo "================================================================"
if (( failures != 0 )); then
  echo "[validation-failed] static failures=$failures" >&2
  exit 1
fi
echo "[validation-ok] 237 fastdefault keeps 236 generic CPU/GPU range, cache, fallback kernels, worker split, and bare -g mode31"
