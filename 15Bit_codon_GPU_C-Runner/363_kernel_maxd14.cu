/*
 * 363_kernel_maxd14.cu
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
 *   /usr/local/cuda/bin/nvcc -O3 -arch=sm_86 -o 363_kernel_maxd14 363_kernel_maxd14.cu
 * Build (host-only CPU test, no CUDA toolkit needed, this sandbox):
 *   gcc -O2 -Wall -Wextra -o 363_kernel_maxd14_cputest 363_kernel_maxd14.cu -x c -lm
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
 * would exceed the 26-slot (13-ancestor) array; (2) a 50,000,000-
 * iteration hard cap on the main DFS loop per task, aborting with full
 * diagnostic state if hit, as a safety net in case the root cause is
 * a genuine infinite loop rather than an overflow; (3) a progress
 * heartbeat printed every 50,000 records in the CPU test harness, so
 * a hang's location is visible immediately instead of after an hour
 * of silence. None of this changes process_one_task()'s actual
 * arithmetic or control flow on the non-error path.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

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
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <N> <in_soa7_bin> <out_results_bin>\n", argv[0]);
        return 1;
    }
    int64_t N = atoll(argv[1]);
    const char *in_path = argv[2];
    const char *out_path = argv[3];

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

    FILE *fout = fopen(out_path, "wb");
    if (!fout) {
        fprintf(stderr, "ERROR: cannot open output '%s'\n", out_path);
        return 1;
    }

    uint32_t board_mask = (uint32_t)((1ULL << N) - 1);
    uint32_t n3 = (uint32_t)(1ULL << (N - 3));
    uint32_t n4 = (uint32_t)(1ULL << (N - 4));

    uint64_t total_sum = 0;
    unsigned char buf[28];
    for (int64_t idx = 0; idx < m; idx++) {
        if (fread(buf, 1, 28, fin) != 28) {
            fprintf(stderr, "ERROR: short read at record %lld\n", (long long)idx);
            return 1;
        }
        uint32_t ld       = read_u32_le(buf + 0);
        uint32_t rd       = read_u32_le(buf + 4);
        uint32_t col      = read_u32_le(buf + 8);
        uint32_t ctrl0    = read_u32_le(buf + 12);
        uint32_t free_v   = read_u32_le(buf + 16);
        uint32_t markctrl = read_u32_le(buf + 20);
        uint32_t w_lo     = read_u32_le(buf + 24);

        uint32_t root_a = free_v & board_mask;
        uint64_t contribution;
        if (root_a == 0u) {
            contribution = 0;
        } else {
            contribution = process_one_task(
                ld, rd, col, root_a, ctrl0, markctrl, w_lo,
                META_NEXT, board_mask, n3, n4,
                idx
            );
        }
        total_sum += contribution;

        if (idx % 50000 == 0) {
            fprintf(stderr, "[progress] idx=%lld/%lld total_sum_so_far=%llu\n",
                    (long long)idx, (long long)m, (unsigned long long)total_sum);
            fflush(stderr);
        }

        unsigned char outb[8];
        for (int b = 0; b < 8; b++) {
            outb[b] = (unsigned char)((contribution >> (8*b)) & 0xFF);
        }
        if (fwrite(outb, 1, 8, fout) != 8) {
            fprintf(stderr, "ERROR: short write at record %lld\n", (long long)idx);
            return 1;
        }
    }

    fclose(fin);
    fclose(fout);

    printf("[kernel-cputest-done] N=%lld records=%lld out=%s total_sum=%llu\n",
           (long long)N, (long long)m, out_path, (unsigned long long)total_sum);
    return 0;
}
#endif /* !__CUDACC__ */
