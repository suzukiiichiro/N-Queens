/*
 * 363_kernel_maxd14.cu
 *
 * rev395c-r6 -- NO code change at all. Byte-identical to
 * 395c_r4_kernel_maxd14.cu below the first #include; only this header note
 * differs, and the harness gates that with a sha256 of the whole code
 * region (not just the kernel). Renamed so that the four r6 artifacts carry
 * one revision number, and because the r6 dispatch table now names this
 * binary with a non-empty env_prefix.
 *
 * WHY r6 EXISTS
 *   395c-r5 measured, on the real -g path and with all six pre-registered
 *   predictions holding:
 *     G0  -g 21 21, nothing extra          133,585.359  free_mb 22,018
 *     Gh  + one idle holder process        133,179.047  free_mb 21,762
 *     Gx  + NQ_EXTRA_CTX=1                 133,206.172  free_mb 21,762
 *     Gxb replicate of Gx                  133,172.016  free_mb 21,762
 *   |Gx - Gh| = 0.020%: a context in another process and a context in ours
 *   are indistinguishable on the production path too. The gain over G0 is
 *   0.309%, i.e. 413 ms, and the claim rule fixed before the run
 *   (>=0.20% and replicated within 0.05%) was met.
 *
 *   r6 turns that into the shipped default by putting NQ_EXTRA_CTX=1 in the
 *   dispatch table's env_prefix -- in the table, not in this binary's
 *   default, so that every run records which state it was in via
 *   [crunner-config] env_prefix= and dispatch.log. NQ_EXTRA_CTX stays 0 when
 *   unset, so a direct run is unchanged and stays usable as a control.
 *
 * KNOWN FRAGILITY, recorded deliberately
 *   r5 also swept the neighbourhood and found the fast window is a NARROW
 *   NOTCH, not a basin:
 *     +0 MB 133,561 | +128 MB 133,993 | +255 MB 133,181 | +384 MB 134,074
 *     | +512 MB 133,854 | +1024 MB 134,310
 *   Both neighbours of the optimum are slower than adding nothing at all,
 *   and the window is under 128 MiB wide. What lands us in it is that one
 *   CUDA context happens to cost 255 MiB on this driver. So this 413 ms is
 *   a property of THIS configuration, not a principle:
 *     - a driver update that changes per-context overhead moves the notch;
 *     - ANY change to this binary's own device footprint moves us out of it.
 *   maxd16 for N=22/23 will allocate substantially more, so the notch must
 *   be re-measured there before any of this is assumed to carry over.
 *
 * rev395c-r4 -- IDENTICAL to 395c_r3_kernel_maxd14.cu except for the
 * NQ_EXTRA_CTX treatment (search for EXTRA CONTEXTS below), which is inert
 * unless NQ_EXTRA_CTX is set to a positive value. Needs -lcuda at link time
 * (driver API). The __global__ kernel and process_one_task() remain
 * byte-identical to 395a r2 / 395c / r2 / r3.
 *
 * rev395c-r3 -- IDENTICAL to 395c_r2_kernel_maxd14.cu except for the
 * NQ_PAD_MB self-pad (search for SELF-PAD below), which is inert unless
 * NQ_PAD_MB is set to a positive value. The __global__ kernel and
 * process_one_task() remain byte-identical to 395a r2 / 395c / r2.
 *
 * rev395c-r2 -- IDENTICAL to 395c_kernel_maxd14.cu except for one added
 * host-side diagnostic block ("[gpu-mem]" / "[gpu-mem-base]", search for
 * PLACEMENT PROBE below). The __global__ kernel and process_one_task()
 * are byte-identical to 395c, which is byte-identical to 395a r2; the
 * harness gates this with a sha256 of the kernel region. The probe runs
 * after every cudaMalloc and before ev_h2d_start, allocates nothing, and
 * therefore changes neither placement nor any timed region. Its purpose
 * is to let the D0/D1/D2 cells be distinguished by device ADDRESS, not
 * only by elapsed time.
 *
 * rev363 -- CUDA C port of kernel_dfs_iter_gpu_maxd14 (Open Objectives
 * item 6), per 362_kernel_port_spec.md. This is the sole GPU kernel
 * used on the selected_maxd==14 execution path; kernel_dfs_iter_gpu_
 * maxd16/18/20/21 remain out of scope (334-onward convention).
 *
 * The per-task body (720-1028 of 360Py/361Py/362Py) is factored into
 * process_one_task(), qualified HOSTDEV (see below) so that:
 *   - under nvcc (__CUDACC__ defined): HOSTDEV expands to
 *     "__host__ __device__", and the real __global__ kernel below
 *     calls it once per constellation inside the grid-stride loop.
 *   - under plain gcc/cc (__CUDACC__ undefined, no CUDA toolkit
 *     required): HOSTDEV expands to nothing, __global__/__device__/
 *     threadIdx/blockIdx are stubbed out by the #else branch below,
 *     and main() (also #ifndef __CUDACC__) drives process_one_task()
 *     directly over a dumped SoA input file for CPU-side cross-
 *     validation against a Python re-execution of the literal Codon
 *     kernel source, mirroring the method 361 already used
 *     successfully for build_soa_for_range()+symmetry().
 * This is a single source of truth: the exact same per-task logic
 * ships to the GPU and is what gets tested here, not a hand-copied
 * approximation of it.
 *
 * Every variable name below is kept identical to the Codon source
 * (schedule_lo, schedule_hi, child_jmark_mask, future_check_mask,
 * terminal_parent_depth, terminal_is_base14, root_action, pr_*, etc.)
 * so the two can be diffed side-by-side by eye, per project
 * discipline (see 361_soa_derive.c for the established pattern).
 *
 * RISK NOTE (358/359, carried forward from 362 spec section 2): the
 * push guard `if (cur_avail != 0)` immediately before every stack
 * push is ported here as a completely literal 1:1 translation, with
 * NO shape change. Do not touch this branch's form until +-3%
 * equivalence against the 356 anchor (393.404s) is confirmed on real
 * hardware, and even then treat any nvcc-side experiment as
 * unverified by the Codon-side findings (n=2 observations on a
 * different backend).
 *
 * Build (device, on cudacodon):
 *   /usr/local/cuda/bin/nvcc -O3 -arch=sm_86 -o 395c_kernel_maxd14 395c_kernel_maxd14.cu
 * Run (device):
 *   ./395c_kernel_maxd14 <N> <in_soa7_bin> <out_results_bin> [expected_total]
 *   e.g. ./395c_kernel_maxd14 21 constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin \
 *          /tmp/gpu_results.bin 314666222712
 *   395a: NQ_MAX_BLOCKS=800 ./395c_kernel_maxd14 21 <in> <out> 314666222712
 * Build (host-only CPU test, no CUDA toolkit needed, this sandbox):
 *   gcc -O2 -Wall -Wextra -x c -o 395c_kernel_maxd14_cputest 395c_kernel_maxd14.cu -lm
 * Build (host-only CPU test, OpenMP-parallelized outer loop):
 *   gcc -O2 -fopenmp -Wall -Wextra -x c -o 395c_kernel_maxd14_cputest_omp 395c_kernel_maxd14.cu -lm
 *   (394b: "-x c" moved BEFORE the source file in both gcc lines -- the
 *   389 header's order fails on gcc 13.3.0 with "file format not
 *   recognized"; noticed and recorded in 392, fixed in the header here.)
 *
 * r2 DIAGNOSTIC INSTRUMENTATION (CPU-test build only, #ifndef
 * __CUDACC__, zero effect on the real __global__ kernel): the r1
 * CPU-test binary hung indefinitely on Suzuki's real N=21 data
 * (2,025,282 records) after passing synthetic-only cross-validation.
 * The synthetic test data's validity filter only checked the three
 * shift-safety conditions build_soa_for_range() itself needs (see
 * 361_soa_derive.c), never verifying that the resulting SCHEDULE DEPTH
 * (child_jmark_mask/terminal_depth's precursor, walked in the
 * precompute phase above) actually reaches values near MAXD14=14 --
 * i.e. the exact regime this kernel exists to handle, and the exact
 * regime the r1 test suite never exercised. r2 adds: (1) a bounds
 * check immediately before each of the two stack pushes, aborting
 * with the offending record's index and field values if stack_ptr
 * would exceed the 26-slot (13-ancestor) array; (2) a 1,000,000-
 * iteration hard cap on the schedule-precompute loop and a
 * 50,000,000-iteration cap on the main DFS loop, each aborting with
 * full diagnostic state if hit; (3) wall-clock timing (elapsed/rate)
 * on the progress heartbeat, plus a per-record slow-record warning.
 * None of this changes process_one_task()'s actual arithmetic or
 * control flow on the non-error path. It turned out the real N=21
 * hang was NOT a bug at all -- the schedule-precompute cap never
 * fired (confirmed via 363_filter_maxd14_only.py: every real N=21
 * record is exactly depth=14, none dropped), and the main-loop cap
 * never fired either: it was genuinely still running, just extremely
 * slowly, because a single CPU thread doing serially what the GPU
 * does across 15,488 parallel threads is fundamentally ~4-5 orders of
 * magnitude slower for this workload. r2 also adds an optional OpenMP
 * build (#ifdef _OPENMP) parallelizing the CPU-test harness's outer
 * per-record loop (process_one_task() itself is untouched) as a
 * partial mitigation, purely for faster CPU-side validation runs --
 * this has no bearing on the real kernel's behavior or performance.
 *
 * 364 ADDITION: a host-side GPU runner (main(), #ifdef __CUDACC__
 * only) that uploads a SoA7 input file (same 7-field format the CPU
 * test harness and 363_filter_maxd14_only.py use) plus META_NEXT to
 * device memory, launches kernel_dfs_iter_gpu_maxd14 with the
 * unchanged production 32x484 grid/block config, downloads results,
 * sums them, times each phase (H2D/kernel/D2H) via cudaEvent, and
 * optionally checks the total against an expected value. This is a
 * single-shot, non-chunked run: it establishes real-hardware
 * correctness first; wiring it into the exact 3-chunk measure2
 * protocol for a true +-3% comparison against the 356 anchor
 * (393.404s) is deferred to a later revision once correctness here is
 * confirmed. Nothing outside this new host-side main() differs from
 * 363 (r2) -- process_one_task() and the __global__ kernel itself are
 * byte-identical.
 *
 * 388 RENAME (pure rename, zero code change): per Suzuki's explicit
 * request, CRunner binaries referenced by crunner_dispatch_table()
 * (385Py onward) should carry the CURRENT revision's number rather
 * than a past revision's, so each revision stays self-contained
 * instead of silently depending on an old file by reference. This
 * file is renamed from 364_kernel_maxd14.cu to 388_kernel_maxd14.cu
 * with ONLY the header's illustrative build/run command examples
 * updated to match (5 occurrences, all in this comment block, lines
 * 42-50) -- everything from the first #include onward, including
 * process_one_task() and the __global__ kernel, is confirmed
 * byte-identical to 364_kernel_maxd14.cu by direct diff. Real-hardware
 * confirmed 2026-09-07: nvcc build under this filename succeeded,
 * N=21 total=314666222712 kernel_ms~=201237, matching 364's own
 * anchor.
 *
 * 389 RENAME (pure rename, zero code change, same reasoning as 388's
 * own rename note directly above -- this is now the standing policy
 * for every future revision, not a one-off): renamed from
 * 388_kernel_maxd14.cu to 389_kernel_maxd14.cu. Only the header's
 * illustrative build/run command examples were updated (the same 5
 * occurrences), plus this paragraph. Code region confirmed byte-
 * identical to 388_kernel_maxd14.cu (and therefore to 364's) by direct
 * diff. NOT YET real-hardware built under this filename.
 *
 * 394b CHANGE (host-side launch configuration only; kernel byte-
 * identical): derived from 389_kernel_maxd14.cu. Three edits, all in
 * main() under #ifdef __CUDACC__ -- process_one_task() and the
 * __global__ kernel are untouched and compile to the same 472 SASS
 * instructions 393 stage3 / 394a observed:
 *   (1) MAX_BLOCKS becomes an int initialised to 484 and overridable
 *       at run time via the environment variable NQ_MAX_BLOCKS
 *       (range-checked, [1,65535]; unset or empty = 484 = 389 behaviour
 *       exactly). stride = BLOCK*MAX_BLOCKS follows, as before. stride
 *       was ALREADY a kernel argument, so no kernel change is needed.
 *   (2) a "[gpu-config] BLOCK=.. MAX_BLOCKS=.. stride=.. threads=..
 *       k_per_thread_max=.." line is printed to stderr right after
 *       "[gpu-run]", so every log is self-describing.
 *   (3) "block= max_blocks= stride=" are APPENDED to the end of the
 *       "[gpu-run-done]" stdout line. The existing total_sum= /
 *       kernel_ms= fields are unchanged and stay in place, so the
 *       Codon side's key=value parser (crunner_parse_result) is
 *       unaffected. 394b's sweep harness drives this binary directly
 *       and does not go through the dispatcher at all.
 * Why: 393-9. ncu measured Active Warps Per Scheduler = 1.51 against a
 * hardware maximum of 12 and a 4-warp ceiling for this BLOCK=32 config.
 * 1.51 = 484 blocks / (80 SMs x 4 schedulers) exactly -- the grid is
 * simply too small to occupy the machine. 484x32 dates from before
 * K-batching (292) decoupled grid size from chunk size; only K has been
 * swept since. 394b sweeps MAX_BLOCKS in {484, 968, 1280, 1936} with
 * K=48 untouched. 1280x32 = 40,960 threads is the L1-footprint-limited
 * target (16 warps/SM x 32 x 208 B = 104 KB of 128 KB) and happens to
 * make 40,960 x 48 = 1,966,080 ~= all of N=21 in one launch.
 * Real-hardware confirmed 2026-09-08 (394b r1/r2): default config
 * reproduces 389 (201,231 / 201,237 ms, oracle MATCH); NQ_MAX_BLOCKS=968
 * gives 163,060..163,197 ms (-18.9%, three runs within 0.08%).
 *
 * 394c RENAME (header only, zero code change): 394b_kernel_maxd14.cu ->
 * 394c_kernel_maxd14.cu. The code region is byte-identical to 394b's
 * (and the kernel region to 389's). 394c's change is on the Codon side
 * (bench_mode=39 feeds this binary the chunkshape148-REORDERED input and
 * sets NQ_MAX_BLOCKS to the shaping stride); nothing here needed to
 * move. Real-hardware confirmed 2026-09-08 (394c factorial): A=201,236
 * (raw, 484), B=261,060 (reordered, 484, +29.7%), C=160,528 (raw, 968),
 * D=158,955 (reordered-without-iter_sort, 968). All four oracle MATCH.
 *
 * 394d RENAME (header only, zero code change): 394c_kernel_maxd14.cu ->
 * 394d_kernel_maxd14.cu. Code region byte-identical to 394c/394b, kernel
 * region to 389. 394d's change is on the Codon side (bench_mode=39 gains
 * bucket_run / iter_sort / input_stage knobs to decompose cell B).
 * Real-hardware confirmed 2026-09-08 (394d ladder, 484): raw 201,238;
 * base-only 296,902 (+47.5%); +scorestripe 295,850; +bucket_run 288,149;
 * +iter_sort=1 258,831; +isort9 261,063 (= 394c B to 0.001%). All MATCH.
 *
 * 394e RENAME (header only, zero code change): 394d_kernel_maxd14.cu ->
 * 394e_kernel_maxd14.cu. Code region byte-identical to 394d/394c/394b,
 * kernel region to 389. 394e drives this binary directly at
 * NQ_MAX_BLOCKS 483/484/485 on inputs shaped for 484, to separate an
 * intra-warp mechanism from an inter-thread (launch-tail) one.
 * Real-hardware confirmed 2026-09-08 (394e): raw 201,234 / 200,061 at
 * 484/485; base 296,899 / 282,374 / 290,851 at 484/485/483 -> the
 * penalty survives the stride shift: intra-warp (M1).
 *
 * 394f RENAME (header only, zero code change): 394e_kernel_maxd14.cu ->
 * 394f_kernel_maxd14.cu. Code region byte-identical to 394e..394b, kernel
 * region to 389. 394f drives this binary on permutations of the raw
 * input (random / state-sorted / free-sorted / schedule-sorted).
 * Real-hardware confirmed 2026-09-08 (394f, 484): raw 201,238; random
 * +44.2%; (col,ld,rd)-sorted +20.8%; free-count-sorted +21.2%;
 * schedule-sorted (stable) -1.71% -- the first order to beat raw.
 *
 * 394g RENAME (header only, zero code change): 394f_kernel_maxd14.cu ->
 * 394g_kernel_maxd14.cu. Code region byte-identical to 394f..394b,
 * kernel region to 389. 394g sweeps MAX_BLOCKS 704/800/864/928 (+484/968
 * anchors) on raw and on the L3 input.
 * Real-hardware confirmed 2026-09-08 (394g): raw 484 201,240 | 704
 * 153,619 | 800 146,585 | 864 158,390 | 928 151,444 | 968 163,133; L3 at
 * 800 = 139,404 (-30.7% vs the 389 anchor). All MATCH.
 *
 * 395a -- FIRST KERNEL CHANGE SINCE 364. process_one_task() main loop:
 * the nibble is split into `block_code = nibble_op & 7u` and
 * `fc_flag = nibble_op & 8u` immediately after extraction, and those two
 * scalars are what the block_code branch and the future check consume.
 * Nothing else in the kernel moves. Motivation and the evidence are in
 * the comment at the change site and in 394a_README_append.md (the
 * duplicated extraction carried 4.09% of samples). Semantics are
 * identical: both consumers read the same bits of the same nibble.
 * r1 RESULT (2026-09-08, real hardware): oracle MATCH, CPU-harness
 * per-record results identical to 394g on 2,048 real records, but
 * kernel_ms 139,527 / 139,542 vs the 394g control's 139,443 / 139,396
 * (+0.07%): the hoist alone did NOT change what nvcc emits.
 * r2: the schedule is materialised once per task as a 64-bit value
 * (schedule64, loop-invariant) and the main loop extracts its nibble
 * with a single funnel shift + AND, replacing the cur_depth<8 test, two
 * shifts and a select. This removes the predicated pair regardless of
 * the compiler's rematerialisation choice. Semantics unchanged
 * (schedule_depth <= 14 -> shift amount <= 52). The root fast path's
 * `schedule_lo & 15u` (depth 0) is untouched. Push guard untouched.
 * 395a_validate.sh checks the SASS with ncu when sudo is available and
 * always checks the N=21 oracle.
 * r2 RESULT (2026-09-08): oracle MATCH; SASS still 472; the cur_depth<8
 * predication is gone but nvcc still rematerialises the extraction
 * (2 x SHF.R.U64); -0.2% vs the 394g control. Safe, neutral, kept.
 *
 * 395b (REVERTED here) tried a register-resident top of the explicit
 * stack: +3.5% at 800x32 (144,032 vs 139,166 direct), +5.2% at 484. The
 * two extra `if (save_sp != 0u)` guards around push/pop are control-flow
 * shape changes in the hottest region -- the same class that failed in
 * 240, 266-269, 273, 326, 358 and 359. Seven for seven: kernel-local
 * control-flow edits are CLOSED as an axis for this kernel.
 *
 * 395c: kernel region is byte-identical to 395a r2 (the last confirmed-
 * safe kernel). Host side keeps the 395b policy default (NQ_MAX_BLOCKS
 * default 800). 395c's measurement is not about the kernel at all: it
 * isolates why the same binary on the same file runs ~4% faster when
 * launched by the Codon dispatcher than directly (395a: 133,580 vs
 * 139,179; 395b: 138,650 vs 144,032) -- see 395c_validate.sh.
 * NOT YET real-hardware built under this filename.
 * Also carries the 394b/394c/394f/394g production parameters, which live
 * on the Codon side (395aPy: MAX_BLOCKS 800 by default, passed via
 * NQ_MAX_BLOCKS; input in the L3 schedule-sorted order). This binary's
 * own NQ_MAX_BLOCKS default stays 484 for direct-invocation continuity.
 * NOT YET real-hardware built under this filename.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#define HOSTDEV __host__ __device__
#else
#define HOSTDEV
/* Stubs so this file parses as plain C when not compiled by nvcc.
 * None of these are exercised outside the #ifdef __CUDACC__ guarded
 * __global__ kernel and its launch machinery below. */
#define __restrict__
#endif

/* ---------------------------------------------------------------------
 * Constants -- 13 bitmasks (362 spec section 3) and the 28-element
 * meta_next table (362 spec section 4). Values unchanged from Codon.
 * ------------------------------------------------------------------- */
static const uint32_t IS_BASE_MASK        = 69222408u;
static const uint32_t IS_JMARK_MASK       = 4u;
static const uint32_t IS_MARK_MASK        = 199209203u;
static const uint32_t IS_P5_MASK          = 3840u;
static const uint32_t SEL2_MASK           = 34742338u;

static const uint32_t BLOCK_CODE_B0_MASK  = 173707345u;
static const uint32_t BLOCK_CODE_B1_MASK  = 12689458u;
static const uint32_t BLOCK_CODE_B2_MASK  = 18088064u;

static const uint32_t OP_STEP3_MASK       = 24u;   /* codes 3,4 */
static const uint32_t OP_ADD1_MASK        = 32u;   /* code 5 */
static const uint32_t OP_BL1_MASK         = 12u;   /* codes 2,3 */
static const uint32_t OP_BL2_MASK         = 16u;   /* code 4 */
static const uint32_t OP_KN3_MASK         = 18u;   /* codes 1,4 */
static const uint32_t OP_KN4_MASK         = 8u;    /* code 3 */

static const uint8_t META_NEXT[28] = {
    1,2,3,3,2,6,2,2,0,4,5,7,13,14,14,14,17,14,14,20,21,21,21,25,21,21,26,26
};

#define MAXD14_ANCESTOR 13

/* ---------------------------------------------------------------------
 * process_one_task -- the per-constellation body (720-1028 of the
 * Codon source), factored out so it is callable from both the real
 * __global__ kernel (device) and a CPU test harness (host). Returns
 * the task's contribution to thread_total, i.e. total*w_lo, exactly
 * matching what the Codon kernel accumulates per idx before idx+=stride.
 * ------------------------------------------------------------------- */
HOSTDEV
static uint64_t process_one_task(
    uint32_t root_ld, uint32_t root_rd, uint32_t root_col,
    uint32_t root_a_in, uint32_t ctrl0, uint32_t markctrl, uint32_t w_lo,
    const uint8_t* __restrict__ meta_next,
    uint32_t bm, uint32_t n3, uint32_t n4
#ifndef __CUDACC__
    , int64_t debug_idx
#endif
) {
    uint32_t jmark = markctrl & 31u;
    uint32_t endm  = (markctrl >> 5) & 31u;
    uint32_t mark1 = (markctrl >> 10) & 31u;
    uint32_t mark2 = (markctrl >> 15) & 31u;
    uint64_t total = 0;

    uint32_t root_a = root_a_in & bm;
    if (root_a == 0u) {
        return 0;
    }

    /* --- schedule precomputation phase (362 spec section 5) --- */
    uint32_t schedule_raw = ctrl0;
    int      schedule_depth = 0;
    uint32_t schedule_lo = 0, schedule_hi = 0;
    uint32_t child_jmark_mask = 0;
    uint32_t future_check_mask = 0;
    int      terminal_parent_depth = 0;
    uint32_t terminal_is_base14 = 0;
    uint32_t root_action = 0;

    for (;;) {
#ifndef __CUDACC__
        if (schedule_depth > 1000000) {
            fprintf(stderr, "[SCHEDULE-PRECOMPUTE-RUNAWAY] debug_idx=%lld schedule_depth=%d schedule_raw=%u "
                    "ctrl0=%u markctrl=%u jmark=%u endm=%u mark1=%u mark2=%u -- this record's schedule "
                    "never reaches a terminal (IS_BASE_MASK) state within 1,000,000 steps. The original "
                    "Codon kernel has no bound here either -- it relies entirely on upstream dispatch "
                    "routing only required_maxd<=14 records to this kernel. This almost certainly means "
                    "the input record does not belong in a maxd14-only test set (see "
                    "363_filter_maxd14_only.py).\n",
                    (long long)debug_idx, schedule_depth, schedule_raw, ctrl0, markctrl, jmark, endm, mark1, mark2);
            fflush(stderr);
            abort();
        }
#endif
        uint32_t schedule_fu   = schedule_raw & 31u;
        uint32_t schedule_rowv = (schedule_raw >> 5) & 31u;

        if (((IS_P5_MASK >> schedule_fu) & 1u) != 0u) {
            if (schedule_rowv == mark1) {
                schedule_fu = (uint32_t)meta_next[schedule_fu];
            }
        }

        uint32_t frame_action = 0;
        uint32_t frame_nibble = 0;
        uint32_t frame_raw = 0;
        uint32_t schedule_fcvu = 0; /* set inside the else branch below when applicable */
        uint32_t schedule_isbu = (IS_BASE_MASK >> schedule_fu) & 1u;

        if (schedule_isbu != 0u && schedule_rowv == endm) {
            frame_action = (schedule_fu == 14u) ? 3u : 2u;
        } else {
            uint32_t schedule_ismu = (IS_MARK_MASK >> schedule_fu) & 1u;
            uint32_t schedule_block_code = 0;
            uint32_t schedule_stepv = 1;
            uint32_t schedule_use_futureu = 1u - schedule_ismu;
            uint32_t schedule_nextfidu = schedule_fu;

            if (schedule_ismu != 0u) {
                uint32_t schedule_markv =
                    (((SEL2_MASK >> schedule_fu) & 1u) != 0u) ? mark2 : mark1;
                if (schedule_rowv == schedule_markv) {
                    schedule_block_code =
                        ((BLOCK_CODE_B0_MASK >> schedule_fu) & 1u)
                        | (((BLOCK_CODE_B1_MASK >> schedule_fu) & 1u) << 1)
                        | (((BLOCK_CODE_B2_MASK >> schedule_fu) & 1u) << 2);
                    schedule_stepv = 2u + ((OP_STEP3_MASK >> schedule_block_code) & 1u);
                    schedule_use_futureu = 0;
                    schedule_nextfidu = (uint32_t)meta_next[schedule_fu];
                }
            }

            uint32_t schedule_isju = (IS_JMARK_MASK >> schedule_fu) & 1u;
            if (schedule_isju != 0u) {
                if (schedule_rowv == jmark) {
                    frame_action = 1u;
                    schedule_nextfidu = (uint32_t)meta_next[schedule_fu];
                }
            }

            uint32_t schedule_child_rowu = schedule_rowv + schedule_stepv;
            if (schedule_use_futureu != 0u && schedule_child_rowu < endm) {
                schedule_fcvu = 1u;
            }
            frame_nibble = schedule_block_code | (schedule_fcvu << 3);
            frame_raw = schedule_nextfidu | (schedule_child_rowu << 5);
        }

        if (schedule_depth == 0) {
            root_action = frame_action;
        } else {
            int parent_depth = schedule_depth - 1;
            if (frame_action == 1u) {
                child_jmark_mask |= (1u << parent_depth);
            } else if (frame_action >= 2u) {
                terminal_parent_depth = parent_depth;
                terminal_is_base14 = (frame_action == 3u) ? 1u : 0u;
            }
        }

        if (frame_action >= 2u) {
            break;
        }

        if (schedule_fcvu != 0u) {
            future_check_mask |= (1u << schedule_depth);
        }

        if (schedule_depth < 8) {
            schedule_lo |= frame_nibble << (schedule_depth * 4);
        } else {
            schedule_hi |= frame_nibble << ((schedule_depth - 8) * 4);
        }
        schedule_raw = frame_raw;
        schedule_depth += 1;
    }

    if (root_action == 2u) {
        return (uint64_t)w_lo;
    }
    if (root_action == 3u) {
        total += ((root_a & ~1u) != 0u) ? 1u : 0u;
        return total * (uint64_t)w_lo;
    }
    if (root_action == 1u) {
        root_a &= ~1u;
        if (root_a == 0u) {
            return 0;
        }
        root_ld |= 1u;
    }

    int      terminal_depth  = terminal_parent_depth;
    uint32_t terminal_base14 = terminal_is_base14;

    /* 395a r2: the schedule as ONE loop-invariant 64-bit value. The main
     * loop below extracts its nibble with a single funnel shift instead of
     * a cur_depth<8 test, two shifts and a select (schedule_depth <= 14,
     * so cur_depth*4 <= 52 < 64 -- the shift is always defined). */
    const uint64_t schedule64 = ((uint64_t)schedule_hi << 32) | (uint64_t)schedule_lo;

    uint32_t save_sp  = 0;
    int      stack_ptr = 0;
    int      cur_depth = 0;
    uint32_t cur_ld = root_ld;
    uint32_t cur_rd = root_rd;
    uint32_t cur_col = root_col;
    uint32_t cur_avail = root_a;

    uint64_t stack[MAXD14_ANCESTOR * 2];

    uint32_t root_rest = cur_avail & (cur_avail - 1u);
    uint32_t root_second = root_rest & (0u - root_rest);
    uint32_t root_after_second = root_rest ^ root_second;

    /* --- root 1-or-2-candidate fast path (362 spec section 6) --- */
    if (root_after_second == 0u) {
        uint32_t root_first = cur_avail & (0u - cur_avail);
        uint32_t pr_nibble_op = schedule_lo & 15u;
        uint32_t pr_block_code = pr_nibble_op & 7u;
        uint32_t pr_bit = root_first;

        uint32_t pr_nld, pr_nrd;
        if (pr_block_code != 0u) {
            uint32_t pr_stepu = 2u + ((OP_STEP3_MASK >> pr_block_code) & 1u);
            uint32_t pr_addvu = (OP_ADD1_MASK >> pr_block_code) & 1u;
            uint32_t pr_bLiu =
                ((OP_BL1_MASK >> pr_block_code) & 1u)
                | (((OP_BL2_MASK >> pr_block_code) & 1u) << 1);
            uint32_t pr_ktu =
                ((OP_KN3_MASK >> pr_block_code) & 1u)
                | (((OP_KN4_MASK >> pr_block_code) & 1u) << 1);
            uint32_t pr_bKu =
                (n3 & (0u - (pr_ktu & 1u))) | (n4 & (0u - (pr_ktu >> 1)));
            pr_nld = ((cur_ld | pr_bit) << pr_stepu) | pr_addvu | pr_bLiu;
            pr_nrd = ((cur_rd | pr_bit) >> pr_stepu) | pr_bKu;
        } else {
            pr_nld = (cur_ld | pr_bit) << 1;
            pr_nrd = (cur_rd | pr_bit) >> 1;
        }
        uint32_t pr_ncol = cur_col | pr_bit;
        uint32_t pr_nf = bm & ~(pr_nld | pr_nrd | pr_ncol);
        uint32_t pr_descend = 1u;
        if (pr_nf == 0u) {
            pr_descend = 0u;
        }
        if (pr_descend != 0u) {
            if (future_check_mask != 0u) {
                if ((pr_nibble_op & 8u) != 0u) {
                    if ((bm & ~((pr_nld << 1) | (pr_nrd >> 1) | pr_ncol)) == 0u) {
                        pr_descend = 0u;
                    }
                }
            }
        }
        if (pr_descend != 0u) {
            if (terminal_depth == 0) {
                if (terminal_base14 == 0u) {
                    total += 1u;
                } else {
                    total += ((pr_nf & ~1u) != 0u) ? 1u : 0u;
                }
                pr_descend = 0u;
            }
        }
        if (pr_descend != 0u) {
            uint32_t pr_child_jmark = child_jmark_mask & 1u;
            if (pr_child_jmark != 0u) {
                pr_nf &= ~1u;
                if (pr_nf == 0u) {
                    pr_descend = 0u;
                } else {
                    pr_nld |= 1u;
                }
            }
        }

        cur_avail = root_rest;
        if (pr_descend != 0u) {
            if (cur_avail != 0u) {
                /* RISK NOTE: literal 1:1 push-guard translation, see file header. */
#ifndef __CUDACC__
                if (stack_ptr + 1 >= MAXD14_ANCESTOR * 2) {
                    fprintf(stderr, "[STACK-OVERFLOW] debug_idx=%lld stack_ptr=%d cur_depth=%d "
                            "(root fast path) ld=%u rd=%u col=%u ctrl0=%u markctrl=%u\n",
                            (long long)debug_idx, stack_ptr, cur_depth, root_ld, root_rd, root_col, ctrl0, markctrl);
                    fflush(stderr);
                    abort();
                }
#endif
                stack[stack_ptr]   = (uint64_t)cur_ld | ((uint64_t)cur_rd << 32);
                stack[stack_ptr+1] = (uint64_t)cur_col
                                    | (((uint64_t)(cur_avail | ((uint32_t)cur_depth << 27))) << 32);
                stack_ptr += 2;
                save_sp   += 1u;
            }
            cur_ld = pr_nld;
            cur_rd = pr_nrd;
            cur_col = pr_ncol;
            cur_avail = pr_nf;
            cur_depth = 1;
        }
    }

    /* --- main explicit-stack DFS loop (362 spec section 7) --- */
#ifndef __CUDACC__
    uint64_t debug_iter_count = 0;
    const uint64_t DEBUG_ITER_CAP = 50000000ULL;
#endif
    for (;;) {
#ifndef __CUDACC__
        debug_iter_count++;
        if (debug_iter_count > DEBUG_ITER_CAP) {
            fprintf(stderr, "[ITER-CAP-HIT] debug_idx=%lld stack_ptr=%d save_sp=%u cur_depth=%d "
                    "cur_avail=%u terminal_depth=%d schedule_lo=%u schedule_hi=%u "
                    "ld=%u rd=%u col=%u ctrl0=%u markctrl=%u\n",
                    (long long)debug_idx, stack_ptr, save_sp, cur_depth, cur_avail, terminal_depth,
                    schedule_lo, schedule_hi, root_ld, root_rd, root_col, ctrl0, markctrl);
            fflush(stderr);
            abort();
        }
#endif
        if (cur_avail == 0u) {
            if (save_sp == 0u) {
                break;
            }
            save_sp -= 1u;
            stack_ptr -= 2;
            uint64_t packed_ldrd  = stack[stack_ptr];
            uint64_t packed_colav = stack[stack_ptr+1];
            cur_ld  = (uint32_t)packed_ldrd;
            cur_rd  = (uint32_t)(packed_ldrd  >> 32);
            cur_col = (uint32_t)packed_colav;
            uint32_t saved_avail = (uint32_t)(packed_colav >> 32);
            cur_avail = saved_avail & bm;
            cur_depth = (int)(saved_avail >> 27);
            continue;
        }

        /* 395a r1 (hoist only, kept): 394a's -lineinfo SourceCounters showed
         * nvcc rematerialising the nibble extraction (cur_depth<8 test,
         * shift, select: 5 SASS at 2899xx) for the future-check test,
         * because the first extraction's register was overwritten by
         * `nibble_op & 7u`. That duplicate carried 4.09% of all samples.
         * r1 split the nibble into block_code / fc_flag once, here -- and
         * measured +0.07%: the compiler still rematerialised.
         * 395a r2: make the extraction itself branchless and cheap -- one
         * funnel shift of the loop-invariant schedule64 plus an AND -- so
         * that whether or not the compiler chooses to recompute it, the
         * cur_depth<8 predication and the second shift are gone. */
        const uint32_t nibble_op  = (uint32_t)(schedule64 >> (cur_depth * 4)) & 15u;
        const uint32_t block_code = nibble_op & 7u;
        const uint32_t fc_flag    = nibble_op & 8u;
        uint32_t bit = cur_avail & (0u - cur_avail);
        cur_avail = cur_avail ^ bit;

        uint32_t nld = (cur_ld | bit) << 1;
        uint32_t nrd = (cur_rd | bit) >> 1;
        uint32_t ncol = cur_col | bit;
        if (block_code != 0u) {
            uint32_t stepu = 2u + ((OP_STEP3_MASK >> block_code) & 1u);
            uint32_t addvu = (OP_ADD1_MASK >> block_code) & 1u;
            uint32_t bLiu =
                ((OP_BL1_MASK >> block_code) & 1u)
                | (((OP_BL2_MASK >> block_code) & 1u) << 1);
            uint32_t ktu =
                ((OP_KN3_MASK >> block_code) & 1u)
                | (((OP_KN4_MASK >> block_code) & 1u) << 1);
            uint32_t bKu =
                (n3 & (0u - (ktu & 1u))) | (n4 & (0u - (ktu >> 1)));
            nld = ((cur_ld | bit) << stepu) | addvu | bLiu;
            nrd = ((cur_rd | bit) >> stepu) | bKu;
        }
        uint32_t nf = bm & ~(nld | nrd | ncol);
        if (nf == 0u) {
            continue;
        }
        if (future_check_mask != 0u) {
            if (fc_flag != 0u) {   /* 395a: was (nibble_op & 8u), see above */
                if ((bm & ~((nld << 1) | (nrd >> 1) | ncol)) == 0u) {
                    continue;
                }
            }
        }

        if (cur_depth == terminal_depth) {
            if (terminal_base14 == 0u) {
                total += 1u;
            } else {
                total += ((nf & ~1u) != 0u) ? 1u : 0u;
            }
            continue;
        }

        uint32_t child_jmark = (child_jmark_mask >> cur_depth) & 1u;
        if (child_jmark != 0u) {
            nf &= ~1u;
            if (nf == 0u) {
                continue;
            }
            nld |= 1u;
        }

        int next_depth = cur_depth + 1;
        if (cur_avail != 0u) {
            /* RISK NOTE: literal 1:1 push-guard translation, see file header. */
#ifndef __CUDACC__
            if (stack_ptr + 1 >= MAXD14_ANCESTOR * 2) {
                fprintf(stderr, "[STACK-OVERFLOW] debug_idx=%lld stack_ptr=%d cur_depth=%d "
                        "(main loop) ld=%u rd=%u col=%u ctrl0=%u markctrl=%u\n",
                        (long long)debug_idx, stack_ptr, cur_depth, root_ld, root_rd, root_col, ctrl0, markctrl);
                fflush(stderr);
                abort();
            }
#endif
            stack[stack_ptr]   = (uint64_t)cur_ld | ((uint64_t)cur_rd << 32);
            stack[stack_ptr+1] = (uint64_t)cur_col
                                | (((uint64_t)(cur_avail | ((uint32_t)cur_depth << 27))) << 32);
            stack_ptr += 2;
            save_sp   += 1u;
        }
        cur_ld = nld;
        cur_rd = nrd;
        cur_col = ncol;
        cur_avail = nf;
        cur_depth = next_depth;
    }

    return total * (uint64_t)w_lo;
}

#ifdef __CUDACC__
/* ---------------------------------------------------------------------
 * The real GPU kernel. Signature matches 362 spec section 1 exactly.
 * Grid-stride loop over K=ceil(m/stride) constellations per thread,
 * unchanged from the Codon source (292's design).
 * ------------------------------------------------------------------- */
__global__ void kernel_dfs_iter_gpu_maxd14(
    const uint32_t* __restrict__ ld_arr,
    const uint32_t* __restrict__ rd_arr,
    const uint32_t* __restrict__ col_arr,
    const uint32_t* __restrict__ ctrl0_arr,
    const uint32_t* __restrict__ free_arr,
    const uint32_t* __restrict__ markctrl_arr,
    const uint32_t* __restrict__ w_lo_arr,
    const uint8_t*  __restrict__ meta_next,
    uint64_t* __restrict__ results,
    int64_t m, uint32_t board_mask,
    uint32_t n3, uint32_t n4,
    int64_t stride
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= stride) return;

    uint64_t thread_total = 0;
    int64_t idx = tid;
    while (idx < m) {
        uint32_t root_a = free_arr[idx] & board_mask;
        if (root_a == 0u) {
            idx += stride;
            continue;
        }
        thread_total += process_one_task(
            ld_arr[idx], rd_arr[idx], col_arr[idx], root_a,
            ctrl0_arr[idx], markctrl_arr[idx], w_lo_arr[idx],
            meta_next, board_mask, n3, n4
        );
        idx += stride;
    }
    results[tid] = thread_total;
}

/* ---------------------------------------------------------------------
 * 364: host-side runner (real nvcc build only). Reads the same 7-field
 * SoA input format the CPU test harness uses (ld, rd, col, ctrl0, free,
 * markctrl, w_lo -- produced by 363_filter_maxd14_only.py from 361's
 * dump), uploads it plus META_NEXT to device memory, launches
 * kernel_dfs_iter_gpu_maxd14 with the unchanged production 32x484
 * grid/block config (stride = 484*32 = 15488, matching 292's K-batching
 * design), downloads the per-thread results, sums them on the host,
 * and reports both the total (for correctness) and elapsed time (for
 * later comparison against the 356 anchor once this is wired into the
 * same 3-chunk measure2 protocol -- 364 itself is a single-shot,
 * non-chunked run: correctness first, timing protocol parity later).
 * ------------------------------------------------------------------- */
/* 395c-r4: driver API, needed only for NQ_EXTRA_CTX (cuCtxCreate). This
 * include sits inside the #ifdef __CUDACC__ host-runner section, so the
 * plain-gcc CPU test build never sees it. Requires -lcuda at link time. */
#include <cuda.h>
#define CU_CHECK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char *_s = NULL; cuGetErrorString(_r, &_s); \
        fprintf(stderr, "[CU-ERROR] %s:%d: %s failed: %s\n", \
                __FILE__, __LINE__, #call, _s ? _s : "?"); \
        exit(1); \
    } \
} while (0)

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[CUDA-ERROR] %s:%d: %s failed: %s\n", \
                __FILE__, __LINE__, #call, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

static uint32_t gpu_read_u32_le(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s <N> <in_soa7_bin> <out_results_bin> [expected_total]\n", argv[0]);
        return 1;
    }
    int64_t N = atoll(argv[1]);
    const char *in_path = argv[2];
    const char *out_path = argv[3];
    int have_expected = (argc == 5);
    unsigned long long expected_total = have_expected ? strtoull(argv[4], NULL, 10) : 0ULL;

    FILE *fin = fopen(in_path, "rb");
    if (!fin) {
        fprintf(stderr, "ERROR: cannot open input '%s'\n", in_path);
        return 1;
    }
    fseek(fin, 0, SEEK_END);
    long fsize = ftell(fin);
    if (fsize < 0 || fsize % 28 != 0) {
        fprintf(stderr, "ERROR: input size %ld not a multiple of 28\n", fsize);
        return 1;
    }
    rewind(fin);
    int64_t m = fsize / 28;
    fprintf(stderr, "[gpu-run] N=%lld records=%lld src=%s\n", (long long)N, (long long)m, in_path);

    uint32_t *h_ld = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *h_rd = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *h_col = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *h_ctrl0 = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *h_free = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *h_markctrl = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *h_wlo = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    if (!h_ld || !h_rd || !h_col || !h_ctrl0 || !h_free || !h_markctrl || !h_wlo) {
        fprintf(stderr, "ERROR: host allocation failed for %lld records\n", (long long)m);
        return 1;
    }

    unsigned char buf[28];
    for (int64_t idx = 0; idx < m; idx++) {
        if (fread(buf, 1, 28, fin) != 28) {
            fprintf(stderr, "ERROR: short read at record %lld\n", (long long)idx);
            return 1;
        }
        h_ld[idx]       = gpu_read_u32_le(buf + 0);
        h_rd[idx]       = gpu_read_u32_le(buf + 4);
        h_col[idx]      = gpu_read_u32_le(buf + 8);
        h_ctrl0[idx]    = gpu_read_u32_le(buf + 12);
        h_free[idx]     = gpu_read_u32_le(buf + 16);
        h_markctrl[idx] = gpu_read_u32_le(buf + 20);
        h_wlo[idx]      = gpu_read_u32_le(buf + 24);
    }
    fclose(fin);

    /* 394b: MAX_BLOCKS is now overridable at RUN time via the environment
     * variable NQ_MAX_BLOCKS (default 800 since 395b -- was 484 = the 389
     * value -- a plain invocation now runs the confirmed config). This is a
     * pure launch-configuration knob: stride is already a kernel ARGUMENT,
     * so the compiled kernel (process_one_task + __global__) is identical
     * for every value -- only grid.x, stride and the results buffer size
     * change on the host. See 394b_README_append.md for why this axis is
     * being swept (393-9: Active Warps/Scheduler=1.51 of a hardware 12). */
    const int BLOCK = 32;
    int MAX_BLOCKS = 800;   /* 395b: was 484. Standing policy: every entry point
                             * defaults to the CONFIRMED parameters (394g: 800).
                             * NQ_MAX_BLOCKS still overrides; the Codon dispatcher
                             * always passes it explicitly. */
    {
        const char *env_mb = getenv("NQ_MAX_BLOCKS");
        if (env_mb != NULL && env_mb[0] != '\0') {
            long v = strtol(env_mb, NULL, 10);
            if (v >= 1 && v <= 65535) {
                MAX_BLOCKS = (int)v;
            } else {
                fprintf(stderr, "ERROR: NQ_MAX_BLOCKS='%s' out of range [1,65535]\n", env_mb);
                return 1;
            }
        }
    }
    const int64_t stride = (int64_t)BLOCK * MAX_BLOCKS; /* 15488 by default, matches 292's K-batching */
    fprintf(stderr, "[gpu-config] BLOCK=%d MAX_BLOCKS=%d stride=%lld threads=%lld k_per_thread_max=%lld\n",
            BLOCK, MAX_BLOCKS, (long long)stride, (long long)stride,
            (long long)((m + stride - 1) / stride));

    uint32_t board_mask = (uint32_t)((1ULL << N) - 1);
    uint32_t n3 = (uint32_t)(1ULL << (N - 3));
    uint32_t n4 = (uint32_t)(1ULL << (N - 4));

    /* 395c-r4 EXTRA CONTEXTS -- the ONE new treatment in this revision.
     * NQ_EXTRA_CTX=<n> makes THIS process create n additional CUDA contexts
     * on the device and hold them, idle, until the end. Each is popped off
     * the thread immediately, so every runtime API call below still runs on
     * the primary context exactly as before -- the extra contexts only exist.
     * They are created before the primary context (the first cudaMalloc) and
     * before any buffer, mirroring a holder process that was already running.
     * r3 showed that the CONTEXT COUNT is a real axis (1 -> 2 -> 3 contexts:
     * 139,185 -> 133,561 -> 133,172). This asks whether the extra context has
     * to live in ANOTHER PROCESS, or whether ours will do -- which decides
     * whether production needs a helper process at all.
     * With NQ_EXTRA_CTX unset or 0, nothing is created and the code path is
     * identical to 395c-r3. */
    CUcontext extra_ctx[16];
    int extra_n = 0;
    {
        long want = 0;
        const char *env_ctx = getenv("NQ_EXTRA_CTX");
        if (env_ctx != NULL && env_ctx[0] != '\0') {
            long v = strtol(env_ctx, NULL, 10);
            if (v < 0 || v > 16) {
                fprintf(stderr, "ERROR: NQ_EXTRA_CTX='%s' out of range [0,16]\n", env_ctx);
                return 1;
            }
            want = v;
        }
        if (want > 0) {
            CUdevice cu_dev;
            CU_CHECK(cuInit(0));
            CU_CHECK(cuDeviceGet(&cu_dev, 0));
            for (long i = 0; i < want; i++) {
                CUcontext c = NULL, popped = NULL;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
                /* CUDA 13 promoted cuCtxCreate to _v4: (pctx, params, flags, dev).
                 * NULL params = the default context, i.e. exactly what _v2 made. */
                CU_CHECK(cuCtxCreate(&c, NULL, 0, cu_dev));
#else
                CU_CHECK(cuCtxCreate(&c, 0, cu_dev));
#endif
                CU_CHECK(cuCtxPopCurrent(&popped));  /* leave the runtime on the primary context */
                extra_ctx[extra_n++] = c;
            }
        }
        fprintf(stderr, "[gpu-ctx] extra_ctx=%d\n", extra_n);
    }

    /* 395c-r3 SELF-PAD -- the ONE new treatment in this revision.
     * NQ_PAD_MB=<n> makes THIS process occupy n MiB of device memory, touched,
     * before any working buffer is allocated, and hold it until after the
     * kernel. It exists to separate the two factors 395c/r2 could not:
     *   "device memory is occupied"  vs  "a second CUDA context exists".
     * A foreign holder changes both at once; this changes only the first.
     * With NQ_PAD_MB unset or 0 NOTHING is allocated and the code path is
     * identical to 395c-r2 -- the harness gates that both statically (the
     * cudaMalloc is inside if (pad_mb > 0)) and at runtime ([gpu-pad]
     * pad_mb=0 in the default-config probe). */
    void *d_pad = NULL;
    long pad_mb = 0;
    {
        const char *env_pad = getenv("NQ_PAD_MB");
        if (env_pad != NULL && env_pad[0] != '\0') {
            long v = strtol(env_pad, NULL, 10);
            if (v < 0 || v > 20000) {
                fprintf(stderr, "ERROR: NQ_PAD_MB='%s' out of range [0,20000]\n", env_pad);
                return 1;
            }
            pad_mb = v;
        }
        if (pad_mb > 0) {
            CUDA_CHECK(cudaMalloc(&d_pad, (size_t)pad_mb << 20));
            CUDA_CHECK(cudaMemset(d_pad, 0, (size_t)pad_mb << 20));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        fprintf(stderr, "[gpu-pad] pad_mb=%ld ptr=%p\n", pad_mb, d_pad);
    }

    uint32_t *d_ld, *d_rd, *d_col, *d_ctrl0, *d_free, *d_markctrl, *d_wlo;
    uint8_t *d_meta_next;
    uint64_t *d_results;

    CUDA_CHECK(cudaMalloc(&d_ld,       (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_rd,       (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_col,      (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ctrl0,    (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_free,     (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_markctrl, (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_wlo,      (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_meta_next, 28 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_results,  (size_t)stride * sizeof(uint64_t)));

    /* 395c-r2 PLACEMENT PROBE -- diagnostic only, provably non-perturbing.
     * It runs AFTER every cudaMalloc (so it cannot change where anything
     * lands) and BEFORE ev_h2d_start (so it is outside every timed region).
     * It allocates nothing: cudaMemGetInfo is a pure query. Its whole job is
     * to record, per cell, WHERE our buffers landed, so that D0/D1/D2 can be
     * told apart by address and not only by wall time. */
    {
        size_t mem_free = 0, mem_total = 0;
        CUDA_CHECK(cudaMemGetInfo(&mem_free, &mem_total));
        fprintf(stderr, "[gpu-mem] ld=%p rd=%p col=%p ctrl0=%p free=%p markctrl=%p wlo=%p meta=%p results=%p\n",
                (void*)d_ld, (void*)d_rd, (void*)d_col, (void*)d_ctrl0,
                (void*)d_free, (void*)d_markctrl, (void*)d_wlo,
                (void*)d_meta_next, (void*)d_results);
        fprintf(stderr, "[gpu-mem-base] ld=0x%llx ld_off2m=0x%llx span=%lld free_mb=%lld total_mb=%lld\n",
                (unsigned long long)(uintptr_t)d_ld,
                (unsigned long long)((uintptr_t)d_ld & 0x1FFFFFULL),
                (long long)((uintptr_t)d_results - (uintptr_t)d_ld),
                (long long)(mem_free >> 20), (long long)(mem_total >> 20));
    }

    cudaEvent_t ev_h2d_start, ev_h2d_end, ev_kernel_start, ev_kernel_end, ev_d2h_end;
    CUDA_CHECK(cudaEventCreate(&ev_h2d_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d_end));
    CUDA_CHECK(cudaEventCreate(&ev_kernel_start));
    CUDA_CHECK(cudaEventCreate(&ev_kernel_end));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_end));

    CUDA_CHECK(cudaEventRecord(ev_h2d_start));
    CUDA_CHECK(cudaMemcpy(d_ld, h_ld, (size_t)m * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rd, h_rd, (size_t)m * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col, h_col, (size_t)m * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ctrl0, h_ctrl0, (size_t)m * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_free, h_free, (size_t)m * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_markctrl, h_markctrl, (size_t)m * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wlo, h_wlo, (size_t)m * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_meta_next, META_NEXT, 28 * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev_h2d_end));

    dim3 grid(MAX_BLOCKS);
    dim3 block(BLOCK);
    CUDA_CHECK(cudaEventRecord(ev_kernel_start));
    kernel_dfs_iter_gpu_maxd14<<<grid, block>>>(
        d_ld, d_rd, d_col, d_ctrl0, d_free, d_markctrl, d_wlo,
        d_meta_next, d_results,
        m, board_mask, n3, n4, stride
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_kernel_end));
    CUDA_CHECK(cudaEventSynchronize(ev_kernel_end));

    uint64_t *h_results = (uint64_t*)malloc((size_t)stride * sizeof(uint64_t));
    if (!h_results) {
        fprintf(stderr, "ERROR: host allocation failed for results (%lld)\n", (long long)stride);
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(h_results, d_results, (size_t)stride * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_d2h_end));
    CUDA_CHECK(cudaEventSynchronize(ev_d2h_end));

    float ms_h2d = 0, ms_kernel = 0, ms_d2h = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, ev_h2d_start, ev_h2d_end));
    CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, ev_kernel_start, ev_kernel_end));
    CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, ev_kernel_end, ev_d2h_end));

    unsigned long long total_sum = 0;
    for (int64_t t = 0; t < stride; t++) {
        total_sum += h_results[t];
    }

    FILE *fout = fopen(out_path, "wb");
    if (fout) {
        for (int64_t t = 0; t < stride; t++) {
            unsigned char outb[8];
            for (int b = 0; b < 8; b++) {
                outb[b] = (unsigned char)((h_results[t] >> (8*b)) & 0xFF);
            }
            fwrite(outb, 1, 8, fout);
        }
        fclose(fout);
    }

    printf("[gpu-run-done] N=%lld records=%lld total_sum=%llu "
           "h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f total_ms=%.3f "
           "block=%d max_blocks=%d stride=%lld\n",
           (long long)N, (long long)m, total_sum,
           ms_h2d, ms_kernel, ms_d2h, (double)ms_h2d + ms_kernel + ms_d2h,
           BLOCK, MAX_BLOCKS, (long long)stride);

    if (have_expected) {
        if (total_sum == expected_total) {
            printf("[gpu-run-correctness] MATCH expected=%llu\n", expected_total);
        } else {
            printf("[gpu-run-correctness] MISMATCH expected=%llu got=%llu\n", expected_total, total_sum);
        }
    }

    cudaFree(d_ld); cudaFree(d_rd); cudaFree(d_col); cudaFree(d_ctrl0);
    cudaFree(d_free); cudaFree(d_markctrl); cudaFree(d_wlo);
    cudaFree(d_meta_next); cudaFree(d_results);
    if (d_pad) cudaFree(d_pad);
    for (int _i = 0; _i < extra_n; _i++) cuCtxDestroy(extra_ctx[_i]);
    free(h_ld); free(h_rd); free(h_col); free(h_ctrl0);
    free(h_free); free(h_markctrl); free(h_wlo); free(h_results);

    return (have_expected && total_sum != expected_total) ? 1 : 0;
}
#endif /* __CUDACC__ */

#ifndef __CUDACC__
/* ---------------------------------------------------------------------
 * CPU-only test harness (no CUDA toolkit required). Reads a flat
 * binary dump of SoA input arrays (produced by the companion Python
 * simulation of the literal Codon kernel source) and computes
 * per-task contributions via process_one_task(), then a total sum,
 * for cross-validation. Input record layout, one record per
 * constellation, 7 x uint32_t fields in this exact order:
 *   ld  rd  col  ctrl0  free  markctrl  w_lo
 * ------------------------------------------------------------------- */
static uint32_t read_u32_le(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s <N> <in_soa7_bin> <out_results_bin> [max_records]\n", argv[0]);
        return 1;
    }
    int64_t N = atoll(argv[1]);
    const char *in_path = argv[2];
    const char *out_path = argv[3];
    int64_t max_records = (argc == 5) ? atoll(argv[4]) : -1;

    FILE *fin = fopen(in_path, "rb");
    if (!fin) {
        fprintf(stderr, "ERROR: cannot open input '%s'\n", in_path);
        return 1;
    }
    fseek(fin, 0, SEEK_END);
    long fsize = ftell(fin);
    if (fsize < 0 || fsize % 28 != 0) { /* 7 x u32 = 28 bytes/record */
        fprintf(stderr, "ERROR: input size %ld not a multiple of 28\n", fsize);
        return 1;
    }
    rewind(fin);
    int64_t m = fsize / 28;
    if (max_records >= 0 && max_records < m) {
        m = max_records;
        fprintf(stderr, "[limited-run] processing only the first %lld of %ld records\n",
                (long long)m, fsize / 28);
    }

    /* Load all needed records into memory upfront so the per-record work
     * below can be split across threads with OpenMP (when built with
     * -fopenmp; falls back to an ordinary serial loop otherwise). This
     * only restructures the test harness's I/O, not process_one_task()'s
     * logic, and each record is fully independent (no shared state),
     * so this is a safe, embarrassingly-parallel speedup for validation
     * purposes on real data, where a single CPU thread is far slower
     * than the 15,488-way GPU parallelism the kernel is designed for. */
    uint32_t *ld_a = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *rd_a = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *col_a = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *ctrl0_a = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *free_a = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *markctrl_a = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *wlo_a = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint64_t *contrib_a = (uint64_t*)malloc((size_t)m * sizeof(uint64_t));
    if (!ld_a || !rd_a || !col_a || !ctrl0_a || !free_a || !markctrl_a || !wlo_a || !contrib_a) {
        fprintf(stderr, "ERROR: out of memory allocating %lld records\n", (long long)m);
        return 1;
    }

    unsigned char buf[28];
    for (int64_t idx = 0; idx < m; idx++) {
        if (fread(buf, 1, 28, fin) != 28) {
            fprintf(stderr, "ERROR: short read at record %lld\n", (long long)idx);
            return 1;
        }
        ld_a[idx]       = read_u32_le(buf + 0);
        rd_a[idx]       = read_u32_le(buf + 4);
        col_a[idx]      = read_u32_le(buf + 8);
        ctrl0_a[idx]    = read_u32_le(buf + 12);
        free_a[idx]     = read_u32_le(buf + 16);
        markctrl_a[idx] = read_u32_le(buf + 20);
        wlo_a[idx]      = read_u32_le(buf + 24);
    }
    fclose(fin);

    FILE *fout = fopen(out_path, "wb");
    if (!fout) {
        fprintf(stderr, "ERROR: cannot open output '%s'\n", out_path);
        return 1;
    }

    uint32_t board_mask = (uint32_t)((1ULL << N) - 1);
    uint32_t n3 = (uint32_t)(1ULL << (N - 3));
    uint32_t n4 = (uint32_t)(1ULL << (N - 4));

    uint64_t total_sum = 0;
    clock_t t_start = clock();

#ifdef _OPENMP
    fprintf(stderr, "[openmp] running with %d threads\n", omp_get_max_threads());
    int64_t completed = 0;
    #pragma omp parallel for schedule(dynamic, 64) reduction(+:total_sum)
    for (int64_t idx = 0; idx < m; idx++) {
        uint32_t root_a = free_a[idx] & board_mask;
        uint64_t contribution;
        if (root_a == 0u) {
            contribution = 0;
        } else {
            contribution = process_one_task(
                ld_a[idx], rd_a[idx], col_a[idx], root_a, ctrl0_a[idx], markctrl_a[idx], wlo_a[idx],
                META_NEXT, board_mask, n3, n4,
                idx
            );
        }
        contrib_a[idx] = contribution;
        total_sum += contribution;
        #pragma omp atomic
        completed++;
        if (idx % 20000 == 0) {
            double elapsed = (double)(clock() - t_start) / CLOCKS_PER_SEC;
            fprintf(stderr, "[progress] completed~%lld/%lld elapsed=%.1fs (wall, /thread count not shown)\n",
                    (long long)completed, (long long)m, elapsed);
            fflush(stderr);
        }
    }
#else
    for (int64_t idx = 0; idx < m; idx++) {
        uint32_t root_a = free_a[idx] & board_mask;
        uint64_t contribution;
        clock_t t_rec_start = clock();
        if (root_a == 0u) {
            contribution = 0;
        } else {
            contribution = process_one_task(
                ld_a[idx], rd_a[idx], col_a[idx], root_a, ctrl0_a[idx], markctrl_a[idx], wlo_a[idx],
                META_NEXT, board_mask, n3, n4,
                idx
            );
        }
        double rec_seconds = (double)(clock() - t_rec_start) / CLOCKS_PER_SEC;
        if (rec_seconds > 2.0) {
            fprintf(stderr, "[SLOW-RECORD] idx=%lld took %.2fs alone -- ld=%u rd=%u col=%u ctrl0=%u markctrl=%u w_lo=%u\n",
                    (long long)idx, rec_seconds, ld_a[idx], rd_a[idx], col_a[idx], ctrl0_a[idx], markctrl_a[idx], wlo_a[idx]);
            fflush(stderr);
        }
        contrib_a[idx] = contribution;
        total_sum += contribution;
        if (idx % 1000 == 0) {
            double elapsed = (double)(clock() - t_start) / CLOCKS_PER_SEC;
            double rate = (idx > 0) ? (double)idx / elapsed : 0.0;
            fprintf(stderr, "[progress] idx=%lld/%lld total_sum_so_far=%llu elapsed=%.1fs rate=%.0f rec/s\n",
                    (long long)idx, (long long)m, (unsigned long long)total_sum, elapsed, rate);
            fflush(stderr);
        }
    }
#endif

    for (int64_t idx = 0; idx < m; idx++) {
        uint64_t contribution = contrib_a[idx];
        unsigned char outb[8];
        for (int b = 0; b < 8; b++) {
            outb[b] = (unsigned char)((contribution >> (8*b)) & 0xFF);
        }
        if (fwrite(outb, 1, 8, fout) != 8) {
            fprintf(stderr, "ERROR: short write at record %lld\n", (long long)idx);
            return 1;
        }
    }

    fclose(fout);
    free(ld_a); free(rd_a); free(col_a); free(ctrl0_a);
    free(free_a); free(markctrl_a); free(wlo_a); free(contrib_a);

    printf("[kernel-cputest-done] N=%lld records=%lld out=%s total_sum=%llu\n",
           (long long)N, (long long)m, out_path, (unsigned long long)total_sum);
    return 0;
}
#endif /* !__CUDACC__ */
