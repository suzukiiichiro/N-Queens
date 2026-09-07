/*
 * 392_kernel_maxd16.cu
 *
 * rev392 -- CUDA C port of kernel_dfs_iter_gpu_maxd16, following the
 * exact same structure as 389_kernel_maxd14.cu (itself a pure rename
 * of 364_kernel_maxd14.cu). This is the maxd16 counterpart needed for
 * N=23 (required_maxd=15, selected_maxd=16, confirmed on real hardware
 * by 389's own bench_mode=34 run).
 *
 * Provenance chain for the maxd16 Codon kernel this file ports:
 *   - 390_maxd16_kernel_port_spec.md: design analysis showing maxd16
 *     needs only ONE structural change from maxd14 (the explicit
 *     stack's ancestor count, 13->15), because the schedule_lo/
 *     schedule_hi nibble registers (2x u32 = 64 bits = exactly 16
 *     four-bit slots) and all mask/opcode constants are depth-
 *     invariant up to depth 16.
 *   - 390 (Codon): kernel_dfs_iter_gpu_maxd16 implemented per the spec,
 *     replacing a stale pre-292 duplicate that had never been updated.
 *   - 391 (Codon, real hardware, 2026-09-07): a staged GPU-vs-CPU
 *     cross-check (process_one_task_maxd16_cpu, mechanically derived
 *     from the same kernel body) matched across all 4 ascending stages
 *     (10/50/200/1000 records) on real N=23 data -- gpu_total==
 *     cpu_total every time.
 *   - 392 (this file): the same per-task logic, ported to C following
 *     363/364's own method exactly -- process_one_task() factored out
 *     with HOSTDEV so the identical logic runs on both the real
 *     __global__ kernel (device) and a CPU test harness (host, no CUDA
 *     toolkit required). Every variable name is kept identical to the
 *     Codon source, same discipline as 364's own header states.
 *
 * The only code-level change from 389_kernel_maxd14.cu (mechanical,
 * scripted substitution, not hand-retyped): MAXD14_ANCESTOR(13) ->
 * MAXD16_ANCESTOR(15), and the kernel/function name. Everything else
 * -- the bitmask constants, META_NEXT table, schedule-precompute loop,
 * root fast-path, main DFS loop, push/pop packing, host runner
 * structure (CLI, file I/O, cudaEvent timing, CUDA_CHECK, output
 * markers) -- is untouched, matching 390_maxd16_kernel_port_spec.md's
 * finding that these are depth-invariant.
 *
 * Output markers deliberately kept IDENTICAL to 389's own
 * ([gpu-run-done] / [gpu-run-correctness]) rather than a distinct tag
 * (contrast 382_kernel_dfs_hybrid.cu's [gpu-hybrid-run-*], which is a
 * genuinely different host-runner architecture) -- this is the same
 * single-shot grid-stride architecture, just a different maxd, and
 * crunner_dispatch_table() already namespaces each CRunner binary's
 * own log file by filename, so no new parsing logic is needed to
 * register this as a maxd16 entry once real-hardware verified.
 *
 * NOTE on CPU-test-harness input filtering: the CPU-only path below
 * still logs a message referencing "a future maxd16 equivalent of
 * 363_filter_maxd14_only.py" for its runaway-schedule diagnostic --
 * that script does not exist yet. Building the actual maxd16 CRunner
 * input pipeline (the 361-dump + 363-filter-equivalent 2-stage process
 * 385/389 already established for maxd14) is out of scope for this
 * file and deferred to whenever crunner_dispatch_table() registration
 * is attempted (390_maxd16_kernel_port_spec.md section 6/7).
 *
 * Build (device, on cudacodon):
 *   /usr/local/cuda/bin/nvcc -O3 -arch=sm_86 -o 392_kernel_maxd16 392_kernel_maxd16.cu
 * Run (device):
 *   ./392_kernel_maxd16 <N> <in_soa7_bin> <out_results_bin> [expected_total]
 *   e.g. ./392_kernel_maxd16 23 <maxd16-filtered-input> \
 *          /tmp/gpu_results.bin 24233937684440
 * Build (host-only CPU test, no CUDA toolkit needed):
 *   gcc -O2 -Wall -Wextra -x c -o 392_kernel_maxd16_cputest 392_kernel_maxd16.cu -lm
 * Build (host-only CPU test, OpenMP-parallelized outer loop):
 *   gcc -O2 -fopenmp -Wall -Wextra -x c -o 392_kernel_maxd16_cputest_omp 392_kernel_maxd16.cu -lm
 * NOTE: -x c must come BEFORE the .cu source file on the command line
 * to force C-language interpretation of the .cu extension -- 364/388/
 * 389's own header documented "... -o out file.cu -x c -lm" (source
 * before -x c), which this sandbox's gcc 13.3.0 actually rejects
 * ("file format not recognized; treating as linker script"). Verified
 * directly: 389_kernel_maxd14.cu fails identically under its own
 * documented command and succeeds once -x c is moved before the
 * source. This appears to be a pre-existing documentation issue
 * inherited across 364/388/389's renames, not something new to this
 * file -- flagged here rather than silently worked around, since it
 * likely affects anyone following those files' own header instructions
 * verbatim on a similar gcc version.
 *
 * STATUS: NOT YET real-hardware built or run under this filename --
 * this is a from-source port, mechanically derived, not yet compiled.
 * Per Suzuki's stated plan, N=23's full run is deliberately deferred
 * until AFTER stepping back to N=21 with ncu for speedup work, so this
 * file's own first real test is expected to be a small/medium-scale
 * correctness check (mirroring 391's staged approach, or 364's own
 * single-shot N=21-scale validation pattern) rather than N=23's full
 * 44.27M-record run.
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

#define MAXD16_ANCESTOR 15

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
                    "the input record does not belong in a maxd<=16 test set (see "
                    "a future maxd16 equivalent of 363_filter_maxd14_only.py (does not exist yet)).\n",
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

    uint32_t save_sp  = 0;
    int      stack_ptr = 0;
    int      cur_depth = 0;
    uint32_t cur_ld = root_ld;
    uint32_t cur_rd = root_rd;
    uint32_t cur_col = root_col;
    uint32_t cur_avail = root_a;

    uint64_t stack[MAXD16_ANCESTOR * 2];

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
                if (stack_ptr + 1 >= MAXD16_ANCESTOR * 2) {
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

        uint32_t nibble_op;
        if (cur_depth < 8) {
            nibble_op = (schedule_lo >> (cur_depth * 4)) & 15u;
        } else {
            nibble_op = (schedule_hi >> ((cur_depth - 8) * 4)) & 15u;
        }
        uint32_t bit = cur_avail & (0u - cur_avail);
        cur_avail = cur_avail ^ bit;

        uint32_t nld = (cur_ld | bit) << 1;
        uint32_t nrd = (cur_rd | bit) >> 1;
        uint32_t ncol = cur_col | bit;
        if ((nibble_op & 7u) != 0u) {
            uint32_t block_code = nibble_op & 7u;
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
            if ((nibble_op & 8u) != 0u) {
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
            if (stack_ptr + 1 >= MAXD16_ANCESTOR * 2) {
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
__global__ void kernel_dfs_iter_gpu_maxd16(
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
 * markctrl, w_lo -- produced by a future maxd16 equivalent of 363_filter_maxd14_only.py (does not exist yet) from 361's
 * dump), uploads it plus META_NEXT to device memory, launches
 * kernel_dfs_iter_gpu_maxd16 with the unchanged production 32x484
 * grid/block config (stride = 484*32 = 15488, matching 292's K-batching
 * design), downloads the per-thread results, sums them on the host,
 * and reports both the total (for correctness) and elapsed time (for
 * later comparison against the 356 anchor once this is wired into the
 * same 3-chunk measure2 protocol -- 364 itself is a single-shot,
 * non-chunked run: correctness first, timing protocol parity later).
 * ------------------------------------------------------------------- */
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

    /* Unchanged production launch config: BLOCK=32, MAX_BLOCKS=484. */
    const int BLOCK = 32;
    const int MAX_BLOCKS = 484;
    const int64_t stride = (int64_t)BLOCK * MAX_BLOCKS; /* 15488, matches 292's K-batching */

    uint32_t board_mask = (uint32_t)((1ULL << N) - 1);
    uint32_t n3 = (uint32_t)(1ULL << (N - 3));
    uint32_t n4 = (uint32_t)(1ULL << (N - 4));

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
    kernel_dfs_iter_gpu_maxd16<<<grid, block>>>(
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
           "h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f total_ms=%.3f\n",
           (long long)N, (long long)m, total_sum,
           ms_h2d, ms_kernel, ms_d2h, (double)ms_h2d + ms_kernel + ms_d2h);

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
