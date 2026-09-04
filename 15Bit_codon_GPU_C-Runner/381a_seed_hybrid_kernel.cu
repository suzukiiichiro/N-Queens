/*
 * 381a_seed_hybrid_kernel.cu
 *
 * rev381a — First real GPU integration of the 378/379a hybrid design
 * (local-stack-primary DFS + threshold-K overflow to a shared Vyukov
 * queue), per 380_gpu_hybrid_integration_spec.md. Two kernels replace
 * 374's single kernel_dfs_iter_gpu_maxd14:
 *
 *   1. seed_kernel_maxd14      -- grid-stride over all M records,
 *      calls seed_task() (verbatim from 379a) per record, filling
 *      TaskSchedule[i] and pushing 0-2 initial Frames into the shared
 *      queue. Implicit CUDA kernel-completion sync (host waits for
 *      this launch before issuing the next) stands in for a grid-wide
 *      barrier, avoiding cooperative-groups complexity.
 *
 *   2. kernel_dfs_hybrid_maxd14 -- launched with the SAME 32x484
 *      thread count as 374 (stride=15488), but NO grid-stride loop
 *      over tasks anymore: each thread is a persistent worker that
 *      pops Frames from the shared queue and runs run_hybrid_episode()
 *      (verbatim from 379a) using its own private LocalStack, until
 *      the queue is empty and no worker is mid-episode.
 *
 * SCOPE OF THIS REVISION (per 380 spec section 5): K_THRESHOLD is
 * fixed at 13 (K_slots=26=MAXD14_ANCESTOR*2, i.e. local capacity
 * exactly matches 374's own stack size) -- at this K, run_hybrid_
 * episode() should NEVER need to touch the shared queue after the
 * initial seed (the local stack alone is already sufficient for any
 * valid maxd<=14 task, exactly as 374 relies on). This makes 381a a
 * pure INFRASTRUCTURE test: does the seed+hybrid two-kernel queue-
 * based plumbing reproduce the SAME oracle (314666222712) as 374's
 * single grid-stride kernel? Speed is NOT evaluated here -- a small
 * regression from 374 (extra kernel-launch overhead) is expected and
 * acceptable at this stage. Sweeping K downward for real speed gains
 * is 381b+'s job, only after this correctness baseline is confirmed.
 *
 * process_one_task_reference()/seed_task()/run_hybrid_episode()/the
 * Vyukov queue (fq_push/fq_pop/FQSlot/FrameQueue) are copied VERBATIM
 * from 377a/379a -- zero logic changes, only the __global__ kernel
 * wrappers and host-side 2-launch sequence are new in this file.
 *
 * IMPORTANT: this file has NOT been compiled with nvcc (no CUDA
 * toolkit available in the authoring sandbox) -- it is unverified
 * device code. 381a_validate.sh's first job is confirming it builds
 * and matches the oracle on real hardware before anything else.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#ifdef __CUDACC__
/* BUG FIX (found on first real nvcc build, thanks to Suzuki-san's
 * report): atomicAdd/atomicExch/atomicCAS are __device__-only
 * intrinsics. Calling them from a __host__ __device__ function is
 * illegal even inside an #ifdef __CUDACC__ guard, because __CUDACC__
 * is true during BOTH nvcc's host-side and device-side compilation
 * passes of a dual-target function -- the guard doesn't distinguish
 * which pass is active. Unlike 377a/379a (which had a real CPU test
 * harness calling these functions from host code), 381a has NO host-
 * side caller at all -- everything runs via the two __global__
 * kernels. So HOSTDEV here is simply __device__, not __host__
 * __device__; this fixes all four call sites at once. */
#define HOSTDEV __device__
#else
#define HOSTDEV
#define __restrict__
#endif

/* ---------------------------------------------------------------------
 * Constants -- byte-identical to 374Py_kernel_maxd14.cu.
 * ------------------------------------------------------------------- */
static const uint32_t IS_BASE_MASK        = 69222408u;
static const uint32_t IS_JMARK_MASK       = 4u;
static const uint32_t IS_MARK_MASK        = 199209203u;
static const uint32_t IS_P5_MASK          = 3840u;
static const uint32_t SEL2_MASK           = 34742338u;

static const uint32_t BLOCK_CODE_B0_MASK  = 173707345u;
static const uint32_t BLOCK_CODE_B1_MASK  = 12689458u;
static const uint32_t BLOCK_CODE_B2_MASK  = 18088064u;

static const uint32_t OP_STEP3_MASK       = 24u;
static const uint32_t OP_ADD1_MASK        = 32u;
static const uint32_t OP_BL1_MASK         = 12u;
static const uint32_t OP_BL2_MASK         = 16u;
static const uint32_t OP_KN3_MASK         = 18u;
static const uint32_t OP_KN4_MASK         = 8u;

static const uint8_t META_NEXT[28] = {
    1,2,3,3,2,6,2,2,0,4,5,7,13,14,14,14,17,14,14,20,21,21,21,25,21,21,26,26
};

#define MAXD14_ANCESTOR 13

/* =======================================================================
 * Vyukov MPMC bounded queue -- verbatim from 377a/379a (BUG FIX #1/#2/#3
 * already applied there; see those files' headers for the diagnostic
 * trail). Zero changes here.
 * ===================================================================== */
typedef struct {
    uint32_t schedule_lo, schedule_hi;
    int      terminal_depth;
    uint32_t terminal_base14;
    uint32_t child_jmark_mask;
    uint32_t future_check_mask;
    uint64_t w_lo;
} TaskSchedule;

typedef struct {
    uint32_t task_id;
    uint32_t cur_ld, cur_rd, cur_col, cur_avail;
    uint32_t cur_depth;
} Frame;

typedef struct {
    Frame    data;
    uint64_t seq;
} FQSlot;

typedef struct {
    FQSlot*  buf;
    uint32_t capacity_mask;
    uint64_t enqueue_pos;
    uint64_t dequeue_pos;
} FrameQueue;

/* Per-slot seq-number initialization happens directly in main() below
 * (written into a host staging buffer, then cudaMemcpy'd to device) --
 * no separate host-side FrameQueue-init helper is needed here. */

HOSTDEV static uint64_t fq_atomic_load_u64(volatile uint64_t* addr) {
#ifdef __CUDACC__
    return atomicAdd((unsigned long long*)addr, 0ULL);
#else
    return __atomic_load_n(addr, __ATOMIC_ACQUIRE);
#endif
}
HOSTDEV static void fq_atomic_store_u64(uint64_t* addr, uint64_t val) {
#ifdef __CUDACC__
    atomicExch((unsigned long long*)addr, (unsigned long long)val);
#else
    __atomic_store_n(addr, val, __ATOMIC_RELEASE);
#endif
}
HOSTDEV static int fq_cas_u64(uint64_t* addr, uint64_t expected, uint64_t desired) {
#ifdef __CUDACC__
    unsigned long long old = atomicCAS((unsigned long long*)addr,
                                        (unsigned long long)expected, (unsigned long long)desired);
    return old == (unsigned long long)expected;
#else
    return __sync_bool_compare_and_swap(addr, expected, desired);
#endif
}

HOSTDEV static int fq_push(FrameQueue* q, const Frame* f) {
    uint64_t pos = fq_atomic_load_u64(&q->enqueue_pos);
    FQSlot* slot;
    for (;;) {
        slot = &q->buf[pos & q->capacity_mask];
        uint64_t seq = fq_atomic_load_u64(&slot->seq);
        int64_t dif = (int64_t)seq - (int64_t)pos;
        if (dif == 0) {
            if (fq_cas_u64(&q->enqueue_pos, pos, pos + 1)) break;
            pos = fq_atomic_load_u64(&q->enqueue_pos);
        } else if (dif < 0) {
            return 0;
        } else {
            pos = fq_atomic_load_u64(&q->enqueue_pos);
        }
    }
    slot->data = *f;
    fq_atomic_store_u64(&slot->seq, pos + 1);
    return 1;
}

HOSTDEV static int fq_pop(FrameQueue* q, Frame* out) {
    uint64_t pos = fq_atomic_load_u64(&q->dequeue_pos);
    FQSlot* slot;
    for (;;) {
        slot = &q->buf[pos & q->capacity_mask];
        uint64_t seq = fq_atomic_load_u64(&slot->seq);
        int64_t dif = (int64_t)seq - (int64_t)(pos + 1);
        if (dif == 0) {
            if (fq_cas_u64(&q->dequeue_pos, pos, pos + 1)) break;
            pos = fq_atomic_load_u64(&q->dequeue_pos);
        } else if (dif < 0) {
            return 0;
        } else {
            pos = fq_atomic_load_u64(&q->dequeue_pos);
        }
    }
    *out = slot->data;
    fq_atomic_store_u64(&slot->seq, pos + q->capacity_mask + 1);
    return 1;
}

HOSTDEV static void atomic_add_result(uint64_t* results, uint32_t task_id, uint64_t contrib) {
    if (contrib == 0) return;
#ifdef __CUDACC__
    atomicAdd((unsigned long long*)&results[task_id], (unsigned long long)contrib);
#else
    __sync_fetch_and_add(&results[task_id], contrib);
#endif
}

/* =======================================================================
 * seed_task -- verbatim from 379a/377b. Schedule precompute + root
 * fast paths; resolves trivial tasks directly via atomic_add_result,
 * or seeds 1-2 initial Frames into the shared queue.
 * ===================================================================== */
HOSTDEV static void seed_task(uint32_t task_id,
                               uint32_t root_ld, uint32_t root_rd, uint32_t root_col,
                               uint32_t root_a_in, uint32_t ctrl0, uint32_t markctrl, uint32_t w_lo,
                               const uint8_t* __restrict__ meta_next,
                               uint32_t bm, uint32_t n3, uint32_t n4,
                               TaskSchedule* ts_out, FrameQueue* q, uint64_t* results, int* push_overflow) {
    uint32_t jmark = markctrl & 31u;
    uint32_t endm  = (markctrl >> 5) & 31u;
    uint32_t mark1 = (markctrl >> 10) & 31u;
    uint32_t mark2 = (markctrl >> 15) & 31u;

    uint32_t root_a = root_a_in & bm;
    ts_out->w_lo = w_lo;
    if (root_a == 0u) return;

    uint32_t schedule_raw = ctrl0;
    int      schedule_depth = 0;
    uint32_t schedule_lo = 0, schedule_hi = 0;
    uint32_t child_jmark_mask = 0;
    uint32_t future_check_mask = 0;
    int      terminal_parent_depth = 0;
    uint32_t terminal_is_base14 = 0;
    uint32_t root_action = 0;

    for (;;) {
        uint32_t schedule_fu = schedule_raw & 31u;
        uint32_t schedule_rowv = (schedule_raw >> 5) & 31u;
        if (((IS_P5_MASK >> schedule_fu) & 1u) != 0u) {
            if (schedule_rowv == mark1) schedule_fu = meta_next[schedule_fu];
        }
        uint32_t frame_action = 0, frame_nibble = 0, frame_raw = 0, schedule_fcvu = 0;
        uint32_t schedule_isbu = (IS_BASE_MASK >> schedule_fu) & 1u;
        if (schedule_isbu != 0u && schedule_rowv == endm) {
            frame_action = (schedule_fu == 14u) ? 3u : 2u;
        } else {
            uint32_t schedule_ismu = (IS_MARK_MASK >> schedule_fu) & 1u;
            uint32_t schedule_block_code = 0;
            uint32_t schedule_stepv = 1u;
            uint32_t schedule_use_futureu = 1u - schedule_ismu;
            uint32_t schedule_nextfidu = schedule_fu;
            if (schedule_ismu != 0u) {
                uint32_t schedule_markv = ((SEL2_MASK >> schedule_fu) & 1u) != 0u ? mark2 : mark1;
                if (schedule_rowv == schedule_markv) {
                    schedule_block_code =
                        ((BLOCK_CODE_B0_MASK >> schedule_fu) & 1u)
                        | (((BLOCK_CODE_B1_MASK >> schedule_fu) & 1u) << 1)
                        | (((BLOCK_CODE_B2_MASK >> schedule_fu) & 1u) << 2);
                    schedule_stepv = 2u + ((OP_STEP3_MASK >> schedule_block_code) & 1u);
                    schedule_use_futureu = 0u;
                    schedule_nextfidu = meta_next[schedule_fu];
                }
            }
            uint32_t schedule_isju = (IS_JMARK_MASK >> schedule_fu) & 1u;
            if (schedule_isju != 0u && schedule_rowv == jmark) {
                frame_action = 1u;
                schedule_nextfidu = meta_next[schedule_fu];
            }
            uint32_t schedule_child_rowu = schedule_rowv + schedule_stepv;
            if (schedule_use_futureu != 0u && schedule_child_rowu < endm) schedule_fcvu = 1u;
            frame_nibble = schedule_block_code | (schedule_fcvu << 3);
            frame_raw = schedule_nextfidu | (schedule_child_rowu << 5);
        }
        if (schedule_depth == 0) {
            root_action = frame_action;
        } else {
            int parent_depth = schedule_depth - 1;
            if (frame_action == 1u) child_jmark_mask |= (1u << parent_depth);
            else if (frame_action >= 2u) {
                terminal_parent_depth = parent_depth;
                terminal_is_base14 = (frame_action == 3u) ? 1u : 0u;
            }
        }
        if (frame_action >= 2u) break;
        if (schedule_fcvu != 0u) future_check_mask |= (1u << schedule_depth);
        if (schedule_depth < 8) schedule_lo |= frame_nibble << (schedule_depth * 4);
        else schedule_hi |= frame_nibble << ((schedule_depth - 8) * 4);
        schedule_raw = frame_raw;
        schedule_depth += 1;
    }

    ts_out->schedule_lo = schedule_lo;
    ts_out->schedule_hi = schedule_hi;
    ts_out->terminal_depth = terminal_parent_depth;
    ts_out->terminal_base14 = terminal_is_base14;
    ts_out->child_jmark_mask = child_jmark_mask;
    ts_out->future_check_mask = future_check_mask;

    if (root_action == 2u) { atomic_add_result(results, task_id, w_lo); return; }
    if (root_action == 3u) {
        uint64_t contrib = ((root_a & ~1u) != 0u) ? (uint64_t)w_lo : 0u;
        atomic_add_result(results, task_id, contrib);
        return;
    }
    if (root_action == 1u) {
        root_a &= ~1u;
        if (root_a == 0u) return;
        root_ld |= 1u;
    }

    int      terminal_depth  = terminal_parent_depth;
    uint32_t terminal_base14 = terminal_is_base14;
    uint32_t cur_ld = root_ld, cur_rd = root_rd, cur_col = root_col, cur_avail = root_a;
    int cur_depth = 0;

    uint32_t root_rest = cur_avail & (cur_avail - 1u);
    uint32_t root_second = root_rest & (0u - root_rest);
    uint32_t root_after_second = root_rest ^ root_second;

    if (root_after_second == 0u) {
        uint32_t root_first = cur_avail & (0u - cur_avail);
        uint32_t pr_nibble_op = schedule_lo & 15u;
        uint32_t pr_block_code = pr_nibble_op & 7u;
        uint32_t pr_bit = root_first;
        uint32_t pr_nld, pr_nrd;
        if (pr_block_code != 0u) {
            uint32_t pr_stepu = 2u + ((OP_STEP3_MASK >> pr_block_code) & 1u);
            uint32_t pr_addvu = (OP_ADD1_MASK >> pr_block_code) & 1u;
            uint32_t pr_bLiu = ((OP_BL1_MASK >> pr_block_code) & 1u) | (((OP_BL2_MASK >> pr_block_code) & 1u) << 1);
            uint32_t pr_ktu  = ((OP_KN3_MASK >> pr_block_code) & 1u) | (((OP_KN4_MASK >> pr_block_code) & 1u) << 1);
            uint32_t pr_bKu  = (n3 & (0u - (pr_ktu & 1u))) | (n4 & (0u - (pr_ktu >> 1)));
            pr_nld = ((cur_ld | pr_bit) << pr_stepu) | pr_addvu | pr_bLiu;
            pr_nrd = ((cur_rd | pr_bit) >> pr_stepu) | pr_bKu;
        } else {
            pr_nld = (cur_ld | pr_bit) << 1;
            pr_nrd = (cur_rd | pr_bit) >> 1;
        }
        uint32_t pr_ncol = cur_col | pr_bit;
        uint32_t pr_nf = bm & ~(pr_nld | pr_nrd | pr_ncol);
        uint32_t pr_descend = (pr_nf == 0u) ? 0u : 1u;
        if (pr_descend != 0u && future_check_mask != 0u && (pr_nibble_op & 8u) != 0u) {
            if ((bm & ~((pr_nld << 1) | (pr_nrd >> 1) | pr_ncol)) == 0u) pr_descend = 0u;
        }
        if (pr_descend != 0u && terminal_depth == 0) {
            uint64_t contrib = (terminal_base14 == 0u) ? (uint64_t)w_lo
                                                         : (((pr_nf & ~1u) != 0u) ? (uint64_t)w_lo : 0u);
            atomic_add_result(results, task_id, contrib);
            pr_descend = 0u;
        }
        if (pr_descend != 0u) {
            uint32_t pr_child_jmark = child_jmark_mask & 1u;
            if (pr_child_jmark != 0u) {
                pr_nf &= ~1u;
                if (pr_nf == 0u) pr_descend = 0u; else pr_nld |= 1u;
            }
        }
        cur_avail = root_rest;
        if (pr_descend != 0u) {
            if (cur_avail != 0u) {
                Frame sib;
                sib.task_id = task_id; sib.cur_ld = cur_ld; sib.cur_rd = cur_rd;
                sib.cur_col = cur_col; sib.cur_avail = cur_avail; sib.cur_depth = (uint32_t)cur_depth;
                if (!fq_push(q, &sib)) { *push_overflow = 1; return; }
            }
            Frame child;
            child.task_id = task_id; child.cur_ld = pr_nld; child.cur_rd = pr_nrd;
            child.cur_col = pr_ncol; child.cur_avail = pr_nf; child.cur_depth = 1;
            if (!fq_push(q, &child)) { *push_overflow = 1; return; }
        }
        return;
    }

    Frame f0;
    f0.task_id = task_id; f0.cur_ld = cur_ld; f0.cur_rd = cur_rd;
    f0.cur_col = cur_col; f0.cur_avail = cur_avail; f0.cur_depth = (uint32_t)cur_depth;
    if (!fq_push(q, &f0)) { *push_overflow = 1; }
}

/* =======================================================================
 * LocalStack + run_hybrid_episode -- verbatim from 379a.
 * ===================================================================== */
typedef struct {
    uint64_t stack[MAXD14_ANCESTOR * 2];
    int      stack_ptr;
    uint32_t save_sp;
} LocalStack;

HOSTDEV static void run_hybrid_episode(
    uint32_t task_id, uint32_t cur_ld, uint32_t cur_rd, uint32_t cur_col, uint32_t cur_avail, int cur_depth,
    const TaskSchedule* ts, uint32_t bm, uint32_t n3, uint32_t n4,
    LocalStack* local, int K_slots,
    FrameQueue* q, uint64_t* results, int* push_overflow)
{
    uint32_t schedule_lo = ts->schedule_lo, schedule_hi = ts->schedule_hi;
    int terminal_depth = ts->terminal_depth;
    uint32_t terminal_base14 = ts->terminal_base14;
    uint32_t child_jmark_mask = ts->child_jmark_mask;
    uint32_t future_check_mask = ts->future_check_mask;

    for (;;) {
        if (cur_avail == 0u) {
            if (local->save_sp == 0u) return;
            local->save_sp -= 1u;
            local->stack_ptr -= 2;
            uint64_t packed_ldrd  = local->stack[local->stack_ptr];
            uint64_t packed_colav = local->stack[local->stack_ptr + 1];
            cur_ld  = (uint32_t)packed_ldrd;
            cur_rd  = (uint32_t)(packed_ldrd  >> 32);
            cur_col = (uint32_t)packed_colav;
            uint32_t saved_avail = (uint32_t)(packed_colav >> 32);
            cur_avail = saved_avail & bm;
            cur_depth = (int)(saved_avail >> 27);
            continue;
        }
        uint32_t nibble_op = (cur_depth < 8) ? ((schedule_lo >> (cur_depth*4)) & 15u)
                                              : ((schedule_hi >> ((cur_depth-8)*4)) & 15u);
        uint32_t bit = cur_avail & (0u - cur_avail);
        cur_avail = cur_avail ^ bit;
        uint32_t nld = (cur_ld | bit) << 1;
        uint32_t nrd = (cur_rd | bit) >> 1;
        uint32_t ncol = cur_col | bit;
        if ((nibble_op & 7u) != 0u) {
            uint32_t block_code = nibble_op & 7u;
            uint32_t stepu = 2u + ((OP_STEP3_MASK >> block_code) & 1u);
            uint32_t addvu = (OP_ADD1_MASK >> block_code) & 1u;
            uint32_t bLiu = ((OP_BL1_MASK >> block_code) & 1u) | (((OP_BL2_MASK >> block_code) & 1u) << 1);
            uint32_t ktu  = ((OP_KN3_MASK >> block_code) & 1u) | (((OP_KN4_MASK >> block_code) & 1u) << 1);
            uint32_t bKu  = (n3 & (0u - (ktu & 1u))) | (n4 & (0u - (ktu >> 1)));
            nld = ((cur_ld | bit) << stepu) | addvu | bLiu;
            nrd = ((cur_rd | bit) >> stepu) | bKu;
        }
        uint32_t nf = bm & ~(nld | nrd | ncol);
        if (nf == 0u) continue;
        if (future_check_mask != 0u && (nibble_op & 8u) != 0u) {
            if ((bm & ~((nld << 1) | (nrd >> 1) | ncol)) == 0u) continue;
        }
        if (cur_depth == terminal_depth) {
            uint64_t contrib = (terminal_base14 == 0u) ? ts->w_lo : (((nf & ~1u) != 0u) ? ts->w_lo : 0u);
            atomic_add_result(results, task_id, contrib);
            continue;
        }
        uint32_t child_jmark = (child_jmark_mask >> cur_depth) & 1u;
        if (child_jmark != 0u) {
            nf &= ~1u;
            if (nf == 0u) continue;
            nld |= 1u;
        }
        int next_depth = cur_depth + 1;
        if (cur_avail != 0u) {
            if (local->stack_ptr + 1 < K_slots) {
                local->stack[local->stack_ptr]   = (uint64_t)cur_ld | ((uint64_t)cur_rd << 32);
                local->stack[local->stack_ptr+1] = (uint64_t)cur_col
                                    | (((uint64_t)(cur_avail | ((uint32_t)cur_depth << 27))) << 32);
                local->stack_ptr += 2;
                local->save_sp   += 1u;
            } else {
                Frame sib;
                sib.task_id = task_id; sib.cur_ld = cur_ld; sib.cur_rd = cur_rd;
                sib.cur_col = cur_col; sib.cur_avail = cur_avail; sib.cur_depth = (uint32_t)cur_depth;
                if (!fq_push(q, &sib)) { *push_overflow = 1; return; }
            }
        }
        cur_ld = nld; cur_rd = nrd; cur_col = ncol; cur_avail = nf; cur_depth = next_depth;
    }
}

#ifdef __CUDACC__
/* ---------------------------------------------------------------------
 * Kernel 1: seed_kernel_maxd14. Grid-stride over all M records.
 * ------------------------------------------------------------------- */
__global__ void seed_kernel_maxd14(
    const uint32_t* __restrict__ ld_arr,
    const uint32_t* __restrict__ rd_arr,
    const uint32_t* __restrict__ col_arr,
    const uint32_t* __restrict__ ctrl0_arr,
    const uint32_t* __restrict__ free_arr,
    const uint32_t* __restrict__ markctrl_arr,
    const uint32_t* __restrict__ w_lo_arr,
    const uint8_t* __restrict__ meta_next,
    TaskSchedule* __restrict__ schedules,
    uint64_t* __restrict__ results,
    int64_t m, uint32_t board_mask, uint32_t n3, uint32_t n4,
    FrameQueue* q, int* push_overflow_flag
) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < m; idx += stride) {
        int of = 0;
        seed_task((uint32_t)idx, ld_arr[idx], rd_arr[idx], col_arr[idx], free_arr[idx],
                   ctrl0_arr[idx], markctrl_arr[idx], w_lo_arr[idx], meta_next, board_mask, n3, n4,
                   &schedules[idx], q, results, &of);
        if (of) atomicExch(push_overflow_flag, 1);
    }
}

/* ---------------------------------------------------------------------
 * Kernel 2: kernel_dfs_hybrid_maxd14. Launched with the same 32x484
 * thread count as 374 -- each thread is a persistent worker, NOT a
 * grid-stride task owner.
 * ------------------------------------------------------------------- */
__global__ void kernel_dfs_hybrid_maxd14(
    TaskSchedule* __restrict__ schedules,
    uint64_t* __restrict__ results,
    uint32_t board_mask, uint32_t n3, uint32_t n4,
    FrameQueue* q, int K_slots,
    int* active_workers, int* push_overflow_flag
) {
    LocalStack local;
    local.stack_ptr = 0;
    local.save_sp = 0;
    for (;;) {
        Frame f;
        if (fq_pop(q, &f)) {
            atomicAdd(active_workers, 1);
            int of = 0;
            run_hybrid_episode(f.task_id, f.cur_ld, f.cur_rd, f.cur_col, f.cur_avail, (int)f.cur_depth,
                                &schedules[f.task_id], board_mask, n3, n4, &local, K_slots, q, results, &of);
            if (of) atomicExch(push_overflow_flag, 1);
            atomicAdd(active_workers, -1);
        } else {
            uint64_t cur_enq = fq_atomic_load_u64(&q->enqueue_pos);
            uint64_t cur_deq = fq_atomic_load_u64(&q->dequeue_pos);
            if (cur_deq >= cur_enq && atomicAdd(active_workers, 0) == 0) break;
        }
    }
}

/* ---------------------------------------------------------------------
 * Host runner. Structure mirrors 374Py_kernel_maxd14.cu's host main()
 * (same CUDA_CHECK macro, same file-reading, same cudaEvent timing
 * pattern) with the additions needed for the queue + 2-launch
 * sequence. K_THRESHOLD is fixed to 13 in this revision per 380 spec
 * section 5 (381a's job is correctness-of-infrastructure, not speed).
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

    int K_threshold = 13;
    { const char* e = getenv("K_THRESHOLD"); if (e) K_threshold = atoi(e); }
    int K_slots = K_threshold * 2;
    if (K_slots > MAXD14_ANCESTOR * 2) K_slots = MAXD14_ANCESTOR * 2;

    int cap_log2 = 24; /* default 16.7M slots * 32B = 536MB; override via FQ_CAPACITY_LOG2 */
    { const char* e = getenv("FQ_CAPACITY_LOG2"); if (e) cap_log2 = atoi(e); }
    uint32_t capacity = 1u << cap_log2;

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
    fprintf(stderr, "[gpu-hybrid-run] N=%lld records=%lld src=%s K_THRESHOLD=%d capacity=%u (%.1f MB)\n",
            (long long)N, (long long)m, in_path, K_threshold, capacity,
            (double)(sizeof(FQSlot) * (size_t)capacity) / (1024.0*1024.0));

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

    /* Unchanged production config for kernel 2's launch shape (32x484
     * = 15488 persistent workers; unlike 374, no host-side "stride"
     * variable is needed here since kernel_dfs_hybrid_maxd14 doesn't
     * grid-stride over an index array -- the launch dimensions alone
     * (dim3 grid/block below) fully determine the worker count). */
    const int BLOCK = 32;
    const int MAX_BLOCKS = 484;

    uint32_t board_mask = (uint32_t)((1ULL << N) - 1);
    uint32_t n3 = (uint32_t)(1ULL << (N - 3));
    uint32_t n4 = (uint32_t)(1ULL << (N - 4));

    uint32_t *d_ld, *d_rd, *d_col, *d_ctrl0, *d_free, *d_markctrl, *d_wlo;
    uint8_t *d_meta_next;
    uint64_t *d_results;
    TaskSchedule *d_schedules;
    FQSlot *d_qbuf;
    FrameQueue *d_q;
    int *d_active_workers, *d_push_overflow;

    CUDA_CHECK(cudaMalloc(&d_ld,       (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_rd,       (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_col,      (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ctrl0,    (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_free,     (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_markctrl, (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_wlo,      (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_meta_next, 28 * sizeof(uint8_t)));
    /* results is now PER-TASK (size m), not per-thread (size stride)
     * as in 374 -- accumulation happens via atomic_add_result() from
     * whichever thread's episode reaches a terminal state for that
     * task, not a single grid-strided owner thread. */
    CUDA_CHECK(cudaMalloc(&d_results,  (size_t)m * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_results, 0, (size_t)m * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_schedules, (size_t)m * sizeof(TaskSchedule)));
    CUDA_CHECK(cudaMalloc(&d_qbuf, (size_t)capacity * sizeof(FQSlot)));
    CUDA_CHECK(cudaMalloc(&d_q, sizeof(FrameQueue)));
    CUDA_CHECK(cudaMalloc(&d_active_workers, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_push_overflow, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_active_workers, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_push_overflow, 0, sizeof(int)));

    /* Initialize the queue's per-slot sequence numbers on the HOST,
     * then upload -- avoids needing a separate init kernel. */
    {
        FQSlot* h_qinit = (FQSlot*)malloc((size_t)capacity * sizeof(FQSlot));
        if (!h_qinit) { fprintf(stderr, "ERROR: host alloc failed for queue init (%u slots)\n", capacity); return 1; }
        for (uint32_t i = 0; i < capacity; i++) h_qinit[i].seq = i;
        CUDA_CHECK(cudaMemcpy(d_qbuf, h_qinit, (size_t)capacity * sizeof(FQSlot), cudaMemcpyHostToDevice));
        free(h_qinit);

        FrameQueue h_q;
        h_q.buf = d_qbuf; /* device pointer, valid once copied to device struct */
        h_q.capacity_mask = capacity - 1;
        h_q.enqueue_pos = 0;
        h_q.dequeue_pos = 0;
        CUDA_CHECK(cudaMemcpy(d_q, &h_q, sizeof(FrameQueue), cudaMemcpyHostToDevice));
    }

    cudaEvent_t ev_h2d_start, ev_h2d_end, ev_seed_start, ev_seed_end,
                ev_kernel_start, ev_kernel_end, ev_d2h_end;
    CUDA_CHECK(cudaEventCreate(&ev_h2d_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d_end));
    CUDA_CHECK(cudaEventCreate(&ev_seed_start));
    CUDA_CHECK(cudaEventCreate(&ev_seed_end));
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

    /* Kernel 1: seed. Generic launch shape (independent of the
     * hybrid kernel's fixed 32x484 persistent-worker shape). */
    int seed_block = 256;
    int seed_grid = (int)((m + seed_block - 1) / seed_block);
    CUDA_CHECK(cudaEventRecord(ev_seed_start));
    seed_kernel_maxd14<<<seed_grid, seed_block>>>(
        d_ld, d_rd, d_col, d_ctrl0, d_free, d_markctrl, d_wlo, d_meta_next,
        d_schedules, d_results, m, board_mask, n3, n4, d_q, d_push_overflow
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_seed_end));
    CUDA_CHECK(cudaEventSynchronize(ev_seed_end));

    int h_push_overflow = 0;
    CUDA_CHECK(cudaMemcpy(&h_push_overflow, d_push_overflow, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_push_overflow) {
        fprintf(stderr, "[gpu-hybrid-run] FAIL: queue overflow during seeding (capacity=%u too small)\n", capacity);
        return 1;
    }

    /* Kernel 2: persistent hybrid workers, same 32x484 shape as 374. */
    dim3 grid(MAX_BLOCKS);
    dim3 block(BLOCK);
    CUDA_CHECK(cudaEventRecord(ev_kernel_start));
    kernel_dfs_hybrid_maxd14<<<grid, block>>>(
        d_schedules, d_results, board_mask, n3, n4, d_q, K_slots,
        d_active_workers, d_push_overflow
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_kernel_end));
    CUDA_CHECK(cudaEventSynchronize(ev_kernel_end));

    CUDA_CHECK(cudaMemcpy(&h_push_overflow, d_push_overflow, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_push_overflow) {
        fprintf(stderr, "[gpu-hybrid-run] FAIL: queue overflow during hybrid worker phase "
                "(capacity=%u too small for K_THRESHOLD=%d)\n", capacity, K_threshold);
        return 1;
    }

    uint64_t *h_results = (uint64_t*)malloc((size_t)m * sizeof(uint64_t));
    if (!h_results) {
        fprintf(stderr, "ERROR: host allocation failed for results (%lld)\n", (long long)m);
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(h_results, d_results, (size_t)m * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_d2h_end));
    CUDA_CHECK(cudaEventSynchronize(ev_d2h_end));

    float ms_h2d = 0, ms_seed = 0, ms_kernel = 0, ms_d2h = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, ev_h2d_start, ev_h2d_end));
    CUDA_CHECK(cudaEventElapsedTime(&ms_seed, ev_seed_start, ev_seed_end));
    CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, ev_kernel_start, ev_kernel_end));
    CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, ev_kernel_end, ev_d2h_end));

    unsigned long long total_sum = 0;
    for (int64_t t = 0; t < m; t++) total_sum += h_results[t];

    FILE *fout = fopen(out_path, "wb");
    if (fout) {
        for (int64_t t = 0; t < m; t++) {
            unsigned char outb[8];
            for (int b = 0; b < 8; b++) outb[b] = (unsigned char)((h_results[t] >> (8*b)) & 0xFF);
            fwrite(outb, 1, 8, fout);
        }
        fclose(fout);
    }

    printf("[gpu-hybrid-run-done] N=%lld records=%lld K_THRESHOLD=%d total_sum=%llu "
           "h2d_ms=%.3f seed_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f total_ms=%.3f\n",
           (long long)N, (long long)m, K_threshold, total_sum,
           ms_h2d, ms_seed, ms_kernel, ms_d2h,
           (double)ms_h2d + ms_seed + ms_kernel + ms_d2h);

    if (have_expected) {
        if (total_sum == expected_total) {
            printf("[gpu-hybrid-run-correctness] MATCH expected=%llu\n", expected_total);
        } else {
            printf("[gpu-hybrid-run-correctness] MISMATCH expected=%llu got=%llu\n", expected_total, total_sum);
        }
    }

    cudaFree(d_ld); cudaFree(d_rd); cudaFree(d_col); cudaFree(d_ctrl0);
    cudaFree(d_free); cudaFree(d_markctrl); cudaFree(d_wlo); cudaFree(d_meta_next);
    cudaFree(d_results); cudaFree(d_schedules); cudaFree(d_qbuf); cudaFree(d_q);
    cudaFree(d_active_workers); cudaFree(d_push_overflow);
    free(h_ld); free(h_rd); free(h_col); free(h_ctrl0);
    free(h_free); free(h_markctrl); free(h_wlo); free(h_results);

    return (have_expected && total_sum != expected_total) ? 1 : 0;
}
#endif /* __CUDACC__ */
