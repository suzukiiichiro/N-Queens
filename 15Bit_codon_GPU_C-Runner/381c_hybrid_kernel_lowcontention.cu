/*
 * 381c_hybrid_kernel_lowcontention.cu
 *
 * rev381c — CORRECTS a real design flaw found in 381b's real-GPU
 * results: 381b bracketed EVERY static-task acquisition (idx<m
 * branch, ~2M times) with atomicAdd(active_workers,1)/atomicAdd(
 * active_workers,-1) on a SINGLE shared global counter, contended by
 * all 15,488 threads. That is a vastly different contention regime
 * than 379b's 8-CPU-thread characterization -- and it shows: 381b
 * measured kernel_ms=613,041, over 3x WORSE than 374's 201,232 and
 * even worse than 381a's flawed 285,841. 380 spec section 4 flagged
 * this exact risk ("atomic競合の桁違いの差") as unverifiable without
 * real hardware; this is that verification, and the risk was real.
 *
 * FIX: active_workers is now touched ONLY during the queue-draining
 * fallback phase (idx>=m), not during ordinary static-task
 * processing. A SEPARATE counter, active_static_workers, is
 * decremented EXACTLY ONCE per thread (not per task) the moment a
 * thread exhausts its static task supply -- 15,488 total atomics for
 * the whole kernel, instead of ~4 million. Termination now requires
 * BOTH active_static_workers==0 (no thread could still be mid-task
 * in the static phase, which could still push overflow) AND
 * active_workers==0 (no thread mid-episode in the fallback phase)
 * AND the queue being empty. This preserves full correctness for any
 * K_THRESHOLD (a thread mid-static-task-processing is still
 * "counted" as potentially active via active_static_workers, just
 * via a single one-time decrement rather than per-task bracketing)
 * while removing the massive contention 381b introduced.
 *
 * Everything else (process_task_hybrid, run_hybrid_episode, the
 * Vyukov queue) is UNCHANGED from 381b.
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

/* =======================================================================
 * process_task_hybrid -- NEW in 381b. Combines seed_task's schedule-
 * precompute + root-fast-path arithmetic (copied verbatim, byte-
 * identical) with LOCAL-STACK-FIRST sibling handling (matching
 * run_hybrid_episode's own K_slots check), then falls through into
 * run_hybrid_episode() for the main DFS loop. This is what a thread
 * calls when it grabs a FRESH task via the free grid-stride index --
 * the shared queue (q) is passed through only for the rare overflow
 * case, exactly as run_hybrid_episode already does; it is NOT touched
 * for the common case where a task's whole tree fits in local capacity.
 * ===================================================================== */
HOSTDEV static void process_task_hybrid(
    uint32_t task_id,
    uint32_t root_ld, uint32_t root_rd, uint32_t root_col,
    uint32_t root_a_in, uint32_t ctrl0, uint32_t markctrl, uint32_t w_lo,
    const uint8_t* __restrict__ meta_next,
    uint32_t bm, uint32_t n3, uint32_t n4,
    TaskSchedule* ts_out, LocalStack* local, int K_slots,
    FrameQueue* q, uint64_t* results, int* push_overflow)
{
    uint32_t jmark = markctrl & 31u;
    uint32_t endm  = (markctrl >> 5) & 31u;
    uint32_t mark1 = (markctrl >> 10) & 31u;
    uint32_t mark2 = (markctrl >> 15) & 31u;

    uint32_t root_a = root_a_in & bm;
    ts_out->w_lo = w_lo;
    if (root_a == 0u) return;

    /* --- schedule precompute: byte-identical to seed_task() --- */
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
                /* SIBLING: local stack first (matches run_hybrid_episode's
                 * own K_slots check), shared queue only on genuine overflow.
                 * This is the entire point of 381b's fix over 381a. */
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
            /* CHILD: continue immediately via run_hybrid_episode -- no
             * queue involvement at all for the common case. */
            run_hybrid_episode(task_id, pr_nld, pr_nrd, pr_ncol, pr_nf, 1,
                                ts_out, bm, n3, n4, local, K_slots, q, results, push_overflow);
        }
        return;
    }

    /* root_after_second != 0: no fast path, hand straight to
     * run_hybrid_episode with the raw root state (cur_depth=0). */
    run_hybrid_episode(task_id, cur_ld, cur_rd, cur_col, cur_avail, cur_depth,
                        ts_out, bm, n3, n4, local, K_slots, q, results, push_overflow);
}

#ifdef __CUDACC__
/* ---------------------------------------------------------------------
 * kernel_dfs_hybrid_maxd14 -- 381b's single unified kernel, replacing
 * 381a's seed_kernel_maxd14 + kernel_dfs_hybrid_maxd14 pair.
 *
 * Launched with 374's own 32x484 thread count. Each thread claims
 * tasks via the SAME free, atomic-free grid-stride index 374 itself
 * uses (idx, idx+stride, idx+2*stride, ...) -- process_task_hybrid()
 * handles each one using ONLY the thread's own LocalStack, touching
 * the shared queue only on genuine K_slots overflow. Once a thread's
 * static supply is exhausted (idx>=m), it falls into a second phase:
 * helping drain any overflow OTHER threads published, until the
 * queue is empty and no thread anywhere is still mid-task (the
 * active_workers protocol from 379a/381a, unchanged).
 * ------------------------------------------------------------------- */
__global__ void kernel_dfs_hybrid_maxd14(
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
    int64_t m, uint32_t board_mask, uint32_t n3, uint32_t n4, int64_t stride,
    FrameQueue* q, int K_slots,
    int* active_static_workers, int* active_workers, int* push_overflow_flag
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    LocalStack local;
    local.stack_ptr = 0;
    local.save_sp = 0;
    int64_t idx = tid;
    int announced_done = 0; /* becomes 1 the moment this thread exhausts idx>=m */

    for (;;) {
        if (idx < m) {
            uint32_t task_id = (uint32_t)idx;
            uint32_t root_a = free_arr[task_id] & board_mask;
            idx += stride;
            if (root_a == 0u) continue; /* matches 374's own skip-if-empty check */
            /* NO active_workers atomics here -- this thread is already
             * accounted for by active_static_workers (still nonzero,
             * decremented exactly once below when idx finally exceeds
             * m). Bracketing every one of ~2M tasks with a shared-
             * counter atomic (as 381b did) caused massive contention
             * across all 15,488 threads; this path is now atomic-free
             * except for the rare genuine fq_push on real overflow
             * inside process_task_hybrid/run_hybrid_episode. */
            int of = 0;
            process_task_hybrid(task_id, ld_arr[task_id], rd_arr[task_id], col_arr[task_id], root_a,
                                 ctrl0_arr[task_id], markctrl_arr[task_id], w_lo_arr[task_id], meta_next,
                                 board_mask, n3, n4, &schedules[task_id], &local, K_slots, q, results, &of);
            if (of) atomicExch(push_overflow_flag, 1);
            continue;
        }
        if (!announced_done) {
            announced_done = 1;
            atomicAdd(active_static_workers, -1); /* ONE atomic per thread total, not per task */
        }
        /* Static task supply exhausted -- help drain shared-queue
         * overflow published by any thread (possibly a different
         * warp/block), until truly nothing is left anywhere. */
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
            if (cur_deq >= cur_enq
                && atomicAdd(active_workers, 0) == 0
                && atomicAdd(active_static_workers, 0) == 0) break;
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

    /* Unchanged production config, 32x484 = 15488 threads. stride is
     * needed again in 381b (unlike 381a) since the unified kernel
     * grid-strides over the static task index itself, exactly as 374
     * does. */
    const int BLOCK = 32;
    const int MAX_BLOCKS = 484;
    const int64_t stride = (int64_t)BLOCK * MAX_BLOCKS; /* 15488 */

    uint32_t board_mask = (uint32_t)((1ULL << N) - 1);
    uint32_t n3 = (uint32_t)(1ULL << (N - 3));
    uint32_t n4 = (uint32_t)(1ULL << (N - 4));

    uint32_t *d_ld, *d_rd, *d_col, *d_ctrl0, *d_free, *d_markctrl, *d_wlo;
    uint8_t *d_meta_next;
    uint64_t *d_results;
    TaskSchedule *d_schedules;
    FQSlot *d_qbuf;
    FrameQueue *d_q;
    int *d_active_static_workers, *d_active_workers, *d_push_overflow;

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
    CUDA_CHECK(cudaMalloc(&d_active_static_workers, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active_workers, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_push_overflow, sizeof(int)));
    {
        /* active_static_workers starts at the TOTAL thread count
         * (stride=15488) -- each thread decrements it exactly once
         * (not per task) the moment it exhausts its static supply. */
        int h_stride_init = (int)stride;
        CUDA_CHECK(cudaMemcpy(d_active_static_workers, &h_stride_init, sizeof(int), cudaMemcpyHostToDevice));
    }
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

    cudaEvent_t ev_h2d_start, ev_h2d_end,
                ev_kernel_start, ev_kernel_end, ev_d2h_end;
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

    /* Single unified kernel: 32x484 = 15488 threads, same shape as
     * 374. No separate seed pass -- task acquisition is via the free
     * grid-stride idx itself (see kernel body). */
    dim3 grid(MAX_BLOCKS);
    dim3 block(BLOCK);
    CUDA_CHECK(cudaEventRecord(ev_kernel_start));
    kernel_dfs_hybrid_maxd14<<<grid, block>>>(
        d_ld, d_rd, d_col, d_ctrl0, d_free, d_markctrl, d_wlo, d_meta_next,
        d_schedules, d_results, m, board_mask, n3, n4, stride,
        d_q, K_slots, d_active_static_workers, d_active_workers, d_push_overflow
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_kernel_end));
    CUDA_CHECK(cudaEventSynchronize(ev_kernel_end));

    int h_push_overflow = 0;
    CUDA_CHECK(cudaMemcpy(&h_push_overflow, d_push_overflow, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_push_overflow) {
        fprintf(stderr, "[gpu-hybrid-run] FAIL: queue overflow during hybrid kernel "
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

    float ms_h2d = 0, ms_kernel = 0, ms_d2h = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, ev_h2d_start, ev_h2d_end));
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
           "h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f total_ms=%.3f\n",
           (long long)N, (long long)m, K_threshold, total_sum,
           ms_h2d, ms_kernel, ms_d2h,
           (double)ms_h2d + ms_kernel + ms_d2h);

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
    cudaFree(d_active_static_workers); cudaFree(d_active_workers); cudaFree(d_push_overflow);
    free(h_ld); free(h_rd); free(h_col); free(h_ctrl0);
    free(h_free); free(h_markctrl); free(h_wlo); free(h_results);

    return (have_expected && total_sum != expected_total) ? 1 : 0;
}
#endif /* __CUDACC__ */
