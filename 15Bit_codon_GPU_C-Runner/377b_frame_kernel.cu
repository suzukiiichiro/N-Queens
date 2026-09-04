/*
 * 377b_frame_kernel.cu
 *
 * rev377b — First integration of the 377a-validated frame queue into
 * an actual DFS execution model. Per 376 spec + Suzuki-san's decision
 * this session: validated first on small-scale SYNTHETIC data (no
 * real N=21/N=18 files needed), by including an UNMODIFIED copy of
 * 374Py_kernel_maxd14.cu's process_one_task() in this same file as
 * `process_one_task_reference()` and comparing its output,
 * record-by-record, against the new frame-queue-based
 * `run_frame_queue_dfs()` on randomly generated inputs -- the same
 * "synthetic data first" method 363 used before real N=21 data was
 * available (363_README_append.md: "合成データ220,000件超...C/Python
 * バイト単位完全一致").
 *
 * DESIGN (376 spec, refined): the "push sibling, immediately descend
 * into child" pattern that 375 identified as the hotspot is replaced
 * with "push sibling AND push child as two independent Frames, then
 * return to the shared queue" -- full decoupling, so ANY worker can
 * pick up ANY pending frame from ANY task. Every arithmetic operation
 * below is copied verbatim from process_one_task_reference(); only
 * the control flow (stack push+continue -> queue push+return) and the
 * terminal-accumulation target (local `total` var -> atomicAdd into a
 * shared per-task result slot) changed.
 *
 * This file does NOT modify 374Py_kernel_maxd14.cu and is not yet
 * wired into the real GPU kernel signature -- that is 377c+, only
 * after this correctness proof and 377a's concurrency proof both hold.
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
#endif

/* ---------------------------------------------------------------------
 * Constants -- byte-identical to 374Py_kernel_maxd14.cu (362 spec
 * section 3/4). Copied, not shared by #include, per this project's
 * established per-revision-file convention.
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
 * PART 1: process_one_task_reference() -- VERBATIM copy of
 * 374Py_kernel_maxd14.cu's process_one_task(), stripped of its
 * #ifndef __CUDACC__ debug-only fprintf blocks (STACK-OVERFLOW /
 * ITER-CAP-HIT / SCHEDULE-PRECOMPUTE-RUNAWAY guards) for brevity --
 * everything that affects the RETURNED VALUE is unchanged. This is
 * the ground truth this revision's new code is checked against.
 * ===================================================================== */
HOSTDEV static uint64_t process_one_task_reference(
    uint32_t root_ld, uint32_t root_rd, uint32_t root_col,
    uint32_t root_a_in, uint32_t ctrl0, uint32_t markctrl, uint32_t w_lo,
    uint32_t bm, uint32_t n3, uint32_t n4
) {
    uint32_t jmark = markctrl & 31u;
    uint32_t endm  = (markctrl >> 5) & 31u;
    uint32_t mark1 = (markctrl >> 10) & 31u;
    uint32_t mark2 = (markctrl >> 15) & 31u;
    uint64_t total = 0;

    uint32_t root_a = root_a_in & bm;
    if (root_a == 0u) return 0;

    uint32_t schedule_raw = ctrl0;
    int      schedule_depth = 0;
    uint32_t schedule_lo = 0, schedule_hi = 0;
    uint32_t child_jmark_mask = 0;
    uint32_t future_check_mask = 0;
    int      terminal_parent_depth = 0;
    uint32_t terminal_is_base14 = 0;
    uint32_t root_action = 0;

    for (;;) {
        if (schedule_depth > 100) return UINT64_MAX; /* sentinel: caller must skip (not a valid maxd<=14-shaped task) */
        uint32_t schedule_fu = schedule_raw & 31u;
        uint32_t schedule_rowv = (schedule_raw >> 5) & 31u;

        if (((IS_P5_MASK >> schedule_fu) & 1u) != 0u) {
            if (schedule_rowv == mark1) schedule_fu = META_NEXT[schedule_fu];
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
                    schedule_nextfidu = META_NEXT[schedule_fu];
                }
            }
            uint32_t schedule_isju = (IS_JMARK_MASK >> schedule_fu) & 1u;
            if (schedule_isju != 0u && schedule_rowv == jmark) {
                frame_action = 1u;
                schedule_nextfidu = META_NEXT[schedule_fu];
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

    if (root_action == 2u) return (uint64_t)w_lo;
    if (root_action == 3u) {
        total += ((root_a & ~1u) != 0u) ? 1u : 0u;
        return total * (uint64_t)w_lo;
    }
    if (root_action == 1u) {
        root_a &= ~1u;
        if (root_a == 0u) return 0;
        root_ld |= 1u;
    }

    int      terminal_depth  = terminal_parent_depth;
    uint32_t terminal_base14 = terminal_is_base14;

    uint32_t save_sp  = 0;
    int      stack_ptr = 0;
    int      cur_depth = 0;
    uint32_t cur_ld = root_ld, cur_rd = root_rd, cur_col = root_col, cur_avail = root_a;
    uint64_t stack[MAXD14_ANCESTOR * 2];

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
            total += (terminal_base14 == 0u) ? 1u : (((pr_nf & ~1u) != 0u) ? 1u : 0u);
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
                if (stack_ptr + 1 >= MAXD14_ANCESTOR * 2) return UINT64_MAX; /* restored 374 guard: not a valid maxd<=14 task */
                stack[stack_ptr]   = (uint64_t)cur_ld | ((uint64_t)cur_rd << 32);
                stack[stack_ptr+1] = (uint64_t)cur_col | (((uint64_t)(cur_avail | ((uint32_t)cur_depth << 27))) << 32);
                stack_ptr += 2;
                save_sp   += 1u;
            }
            cur_ld = pr_nld; cur_rd = pr_nrd; cur_col = pr_ncol; cur_avail = pr_nf; cur_depth = 1;
        }
    }

    for (;;) {
        if (cur_avail == 0u) {
            if (save_sp == 0u) break;
            save_sp -= 1u; stack_ptr -= 2;
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
            total += (terminal_base14 == 0u) ? 1u : (((nf & ~1u) != 0u) ? 1u : 0u);
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
            if (stack_ptr + 1 >= MAXD14_ANCESTOR * 2) return UINT64_MAX; /* restored 374 guard: not a valid maxd<=14 task */
            stack[stack_ptr]   = (uint64_t)cur_ld | ((uint64_t)cur_rd << 32);
            stack[stack_ptr+1] = (uint64_t)cur_col | (((uint64_t)(cur_avail | ((uint32_t)cur_depth << 27))) << 32);
            stack_ptr += 2;
            save_sp   += 1u;
        }
        cur_ld = nld; cur_rd = nrd; cur_col = ncol; cur_avail = nf; cur_depth = next_depth;
    }
    return total * (uint64_t)w_lo;
}

/* =======================================================================
 * PART 2: frame-queue based re-implementation (376 spec + this
 * session's refinement: full decoupling -- push sibling AND child,
 * always return to the shared queue afterward).
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

/* =======================================================================
 * BUG FIX #3 (found via multi-thread stress testing at 8/16 threads,
 * after #1 and #2 above were already fixed): even with the signed-
 * arithmetic backlog check correct, the underlying design -- a bare
 * "push_idx/pop_idx modulo capacity" ring buffer with NO per-slot
 * synchronization between a producer finishing its write and a
 * consumer reading it -- has a genuine data race. If a producer
 * thread is descheduled by the OS between reserving slot X (via
 * atomicAdd on push_idx) and actually writing q->buf[X&mask], and
 * enough OTHER threads race far enough ahead that the ring wraps
 * back around to that same physical slot (i.e. some later index
 * Y with Y&mask == X&mask gets reserved and written), the delayed
 * producer's eventual write can silently clobber -- or be clobbered
 * by -- the newer occupant of that slot, with NO indication of error.
 * This is exactly what 8/16-thread runs exposed: occasional silent
 * loss of a terminal contribution (new_sum short of ref_sum by
 * exactly one w_lo), never at 1-4 threads (not enough scheduling
 * variance to trigger the race in this run's data volume).
 *
 * FIXED by switching to Dmitry Vyukov's bounded MPMC queue design:
 * every slot carries its own sequence number. A producer may only
 * write a slot once that slot's sequence number confirms the PREVIOUS
 * occupant has been fully consumed (seq == pos); a consumer may only
 * read a slot once its sequence number confirms the CURRENT producer
 * has finished writing (seq == pos+1). This closes the race: a slow
 * producer's write can never be silently overwritten, because no
 * later producer can claim that physical slot until the sequence
 * number proves this producer's write (and the following consumer's
 * read) has fully completed a full lap around the buffer.
 * ===================================================================== */
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

static void fq_init(FrameQueue* q, FQSlot* buf, uint32_t capacity) {
    q->buf = buf;
    q->capacity_mask = capacity - 1;
    q->enqueue_pos = 0;
    q->dequeue_pos = 0;
    for (uint32_t i = 0; i < capacity; i++) q->buf[i].seq = i;
}

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
            /* lost the CAS race for this pos -- reload and retry */
            pos = fq_atomic_load_u64(&q->enqueue_pos);
        } else if (dif < 0) {
            return 0; /* queue full: this slot's previous occupant not yet consumed */
        } else {
            pos = fq_atomic_load_u64(&q->enqueue_pos); /* someone else already advanced past us */
        }
    }
    slot->data = *f;
    fq_atomic_store_u64(&slot->seq, pos + 1); /* publish: now safe for a consumer to read */
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
            return 0; /* queue empty: this slot not yet published by a producer */
        } else {
            pos = fq_atomic_load_u64(&q->dequeue_pos);
        }
    }
    *out = slot->data;
    fq_atomic_store_u64(&slot->seq, pos + q->capacity_mask + 1); /* free for the NEXT lap */
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

/* One frame's worth of work: sibling-bit loop (arithmetic byte-
 * identical to process_one_task_reference's main loop), terminating
 * either by exhausting cur_avail (frame done, nothing pushed) or by
 * finding a live non-terminal child (pushes 1-2 new Frames, returns
 * -- does NOT continue processing the child itself; that is left for
 * whichever worker next pops it, possibly a different thread). */
HOSTDEV static void step_one_frame(const Frame* f, const TaskSchedule* ts,
                                    uint32_t bm, uint32_t n3, uint32_t n4,
                                    FrameQueue* q, uint64_t* results, int* push_overflow) {
    uint32_t cur_ld = f->cur_ld, cur_rd = f->cur_rd, cur_col = f->cur_col, cur_avail = f->cur_avail;
    int cur_depth = (int)f->cur_depth;
    uint32_t schedule_lo = ts->schedule_lo, schedule_hi = ts->schedule_hi;
    int terminal_depth = ts->terminal_depth;
    uint32_t terminal_base14 = ts->terminal_base14;
    uint32_t child_jmark_mask = ts->child_jmark_mask;
    uint32_t future_check_mask = ts->future_check_mask;

    for (;;) {
        if (cur_avail == 0u) return;
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
            atomic_add_result(results, f->task_id, contrib);
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
            Frame sib;
            sib.task_id = f->task_id; sib.cur_ld = cur_ld; sib.cur_rd = cur_rd;
            sib.cur_col = cur_col; sib.cur_avail = cur_avail; sib.cur_depth = (uint32_t)cur_depth;
            if (!fq_push(q, &sib)) { *push_overflow = 1; return; }
        }
        Frame child;
        child.task_id = f->task_id; child.cur_ld = nld; child.cur_rd = nrd;
        child.cur_col = ncol; child.cur_avail = nf; child.cur_depth = (uint32_t)next_depth;
        if (!fq_push(q, &child)) { *push_overflow = 1; return; }
        return; /* fully decoupled: this worker returns to the queue */
    }
}

/* Root setup: schedule precompute + root_action fast paths + root
 * 1-or-2-candidate fast path, verbatim arithmetic from
 * process_one_task_reference(). Either resolves the task fully
 * (atomicAdd into results directly, no frame needed) or seeds 1-2
 * initial Frames into the queue. */
HOSTDEV static void seed_task(uint32_t task_id,
                               uint32_t root_ld, uint32_t root_rd, uint32_t root_col,
                               uint32_t root_a_in, uint32_t ctrl0, uint32_t markctrl, uint32_t w_lo,
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
        if (schedule_depth > 100) return; /* not a valid maxd<=14-shaped task: skip (no frame seeded, no result added) */
        uint32_t schedule_fu = schedule_raw & 31u;
        uint32_t schedule_rowv = (schedule_raw >> 5) & 31u;
        if (((IS_P5_MASK >> schedule_fu) & 1u) != 0u) {
            if (schedule_rowv == mark1) schedule_fu = META_NEXT[schedule_fu];
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
                    schedule_nextfidu = META_NEXT[schedule_fu];
                }
            }
            uint32_t schedule_isju = (IS_JMARK_MASK >> schedule_fu) & 1u;
            if (schedule_isju != 0u && schedule_rowv == jmark) {
                frame_action = 1u;
                schedule_nextfidu = META_NEXT[schedule_fu];
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
            return;
        }
        /* BUG FIX (found during this session's synthetic-data test):
         * when pr_descend==0, the ORIGINAL code does NOT return here --
         * cur_ld/cur_rd/cur_col/cur_depth are left at their pre-fast-path
         * values (root_ld/root_rd/root_col/0) and cur_avail=root_rest,
         * and execution falls through into the general main DFS loop
         * with THAT state (i.e. the "second candidate" still needs full
         * general-purpose processing). An earlier draft of this function
         * had an unconditional `return;` here, silently dropping the
         * second candidate's contribution whenever the first candidate's
         * fast path did not descend -- caught by a ref_sum != new_sum
         * mismatch (off by exactly one w_lo) on synthetic data. */
        if (root_rest == 0u) return; /* no second candidate at all */
        Frame f0;
        f0.task_id = task_id; f0.cur_ld = root_ld; f0.cur_rd = root_rd;
        f0.cur_col = root_col; f0.cur_avail = root_rest; f0.cur_depth = 0;
        if (!fq_push(q, &f0)) { *push_overflow = 1; }
        return;
    }

    /* root_after_second != 0: more than 2 candidates, no fast path --
     * seed the general DFS state directly (cur_depth=0, cur_avail=root_a). */
    Frame f0;
    f0.task_id = task_id; f0.cur_ld = cur_ld; f0.cur_rd = cur_rd;
    f0.cur_col = cur_col; f0.cur_avail = cur_avail; f0.cur_depth = (uint32_t)cur_depth;
    if (!fq_push(q, &f0)) { *push_overflow = 1; }
}

#ifndef __CUDACC__
/* =======================================================================
 * CPU-only correctness test: random synthetic tasks, compared
 * record-by-record against process_one_task_reference().
 * ===================================================================== */

static uint32_t rand_bits_u32(unsigned int* seed, int nbits) {
    uint32_t v = (uint32_t)rand_r(seed) & ((nbits >= 32) ? 0xFFFFFFFFu : ((1u << nbits) - 1u));
    return v;
}

int main(int argc, char** argv) {
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
    if (argc >= 2) n_threads = atoi(argv[1]);
#endif
    long n_tasks = 20000;
    if (argc >= 3) n_tasks = atol(argv[2]);

    const int N = 21;
    uint32_t bm = (N >= 32) ? 0xFFFFFFFFu : ((1u << N) - 1u);
    uint32_t n3 = 1u << (N - 3);
    uint32_t n4 = 1u << (N - 4);

    /* --- generate synthetic tasks (same shape 363 used: random within
     * valid field widths, no attempt to guarantee "realistic" N-Queens
     * semantics -- process_one_task_reference() is well-defined for any
     * 32-bit inputs, so this is a valid black-box equivalence test). --- */
    uint32_t* ld_arr = malloc(sizeof(uint32_t) * n_tasks);
    uint32_t* rd_arr = malloc(sizeof(uint32_t) * n_tasks);
    uint32_t* col_arr = malloc(sizeof(uint32_t) * n_tasks);
    uint32_t* free_arr = malloc(sizeof(uint32_t) * n_tasks);
    uint32_t* ctrl0_arr = malloc(sizeof(uint32_t) * n_tasks);
    uint32_t* markctrl_arr = malloc(sizeof(uint32_t) * n_tasks);
    uint32_t* w_lo_arr = malloc(sizeof(uint32_t) * n_tasks);
    uint64_t* ref_results = malloc(sizeof(uint64_t) * n_tasks);
    uint64_t* new_results = calloc(n_tasks, sizeof(uint64_t));

    unsigned int seed = 42;
    for (long i = 0; i < n_tasks; i++) {
        ld_arr[i]  = rand_bits_u32(&seed, N);
        rd_arr[i]  = rand_bits_u32(&seed, N);
        col_arr[i] = rand_bits_u32(&seed, N);
        free_arr[i] = bm & ~(ld_arr[i] | rd_arr[i] | col_arr[i]);
        ctrl0_arr[i] = rand_bits_u32(&seed, 10); /* fu(5)+rowv(5)-ish range */
        uint32_t j = rand_bits_u32(&seed, 5), e = rand_bits_u32(&seed, 5);
        uint32_t m1 = rand_bits_u32(&seed, 5), m2 = rand_bits_u32(&seed, 5);
        markctrl_arr[i] = j | (e << 5) | (m1 << 10) | (m2 << 15);
        w_lo_arr[i] = 1u + rand_bits_u32(&seed, 4); /* small positive weight */
    }

    /* --- reference (serial, unchanged process_one_task) --- */
    unsigned char* skip = calloc(n_tasks, 1);
    long n_skipped = 0;
    for (long i = 0; i < n_tasks; i++) {
        ref_results[i] = process_one_task_reference(
            ld_arr[i], rd_arr[i], col_arr[i], free_arr[i], ctrl0_arr[i], markctrl_arr[i], w_lo_arr[i],
            bm, n3, n4);
        if (ref_results[i] == UINT64_MAX) {
            /* random ctrl0/markctrl did not shape a valid maxd<=14 task
             * (schedule precompute did not reach a terminal state within
             * the sanity cap) -- excluded from comparison, exactly as
             * 363_filter_maxd14_only.py excludes such records in
             * production rather than feeding them to this kernel. */
            skip[i] = 1;
            n_skipped++;
        }
    }

    /* --- new: seed all non-skipped tasks, then persistent workers drain the queue --- */
    /* NOTE: capacity is a fixed absolute size, NOT scaled by n_tasks --
     * unrealistic random synthetic tasks (this test) branch far more
     * bushily than real, constrained N-Queens data ever would, so
     * scaling capacity *with* n_tasks made the required allocation
     * balloon to multiple GB and segfault on malloc() (whose return
     * value this draft had also failed to check -- found alongside
     * this). Real N=21 data (377c+) is expected to need far less
     * capacity per concurrently-live task than this synthetic stress
     * test does; that will be measured directly once real data is
     * used, not extrapolated from this synthetic figure. */
    uint32_t capacity = 1u << 24; /* 16,777,216 slots * 32B(FQSlot) = 536MB, fixed */
    FQSlot* qbuf = malloc(sizeof(FQSlot) * capacity);
    if (!qbuf) {
        fprintf(stderr, "[377b] FAIL: malloc failed for queue buffer (capacity=%u, %.1f MB requested)\n",
                capacity, (double)(sizeof(FQSlot) * (size_t)capacity) / (1024.0*1024.0));
        return 1;
    }
    FrameQueue q;
    fq_init(&q, qbuf, capacity);
    TaskSchedule* schedules = malloc(sizeof(TaskSchedule) * n_tasks);
    int push_overflow = 0;

    for (long i = 0; i < n_tasks; i++) {
        if (skip[i]) continue;
        seed_task((uint32_t)i, ld_arr[i], rd_arr[i], col_arr[i], free_arr[i],
                   ctrl0_arr[i], markctrl_arr[i], w_lo_arr[i], bm, n3, n4,
                   &schedules[i], &q, new_results, &push_overflow);
    }
    if (push_overflow) {
        fprintf(stderr, "[377b] FAIL: queue overflow during seeding (capacity=%u too small)\n", capacity);
        return 1;
    }

    long active_workers = 0;
#ifdef _OPENMP
    #pragma omp parallel num_threads(n_threads)
#endif
    {
        for (;;) {
            Frame f;
            if (fq_pop(&q, &f)) {
                __sync_fetch_and_add(&active_workers, 1);
                int of = 0;
                step_one_frame(&f, &schedules[f.task_id], bm, n3, n4, &q, new_results, &of);
                if (of) { push_overflow = 1; }
                __sync_fetch_and_add(&active_workers, -1);
            } else {
                uint64_t cur_enq = fq_atomic_load_u64(&q.enqueue_pos);
                uint64_t cur_deq = fq_atomic_load_u64(&q.dequeue_pos);
                if (cur_deq >= cur_enq && __sync_fetch_and_add(&active_workers, 0) == 0) {
                    break; /* queue empty AND nobody is mid-step (so nobody can push more) */
                }
                /* else: spin -- someone may still push more work */
            }
        }
    }

    if (push_overflow) {
        fprintf(stderr, "[377b] DIAG: final enqueue_pos=%llu dequeue_pos=%llu capacity=%u\n",
                (unsigned long long)q.enqueue_pos, (unsigned long long)q.dequeue_pos, capacity);
        fprintf(stderr, "[377b] FAIL: queue overflow during frame processing (capacity=%u too small for n_tasks=%ld)\n",
                capacity, n_tasks);
        return 1;
    }

    long mismatches = 0;
    for (long i = 0; i < n_tasks; i++) {
        if (skip[i]) continue;
        if (ref_results[i] != new_results[i]) {
            mismatches++;
            if (mismatches <= 10) {
                fprintf(stderr, "[377b] MISMATCH task_id=%ld ref=%llu new=%llu "
                        "ld=%u rd=%u col=%u ctrl0=%u markctrl=%u w_lo=%u\n",
                        i, (unsigned long long)ref_results[i], (unsigned long long)new_results[i],
                        ld_arr[i], rd_arr[i], col_arr[i], ctrl0_arr[i], markctrl_arr[i], w_lo_arr[i]);
            }
        }
    }

    uint64_t ref_sum = 0, new_sum = 0;
    for (long i = 0; i < n_tasks; i++) {
        if (skip[i]) continue;
        ref_sum += ref_results[i]; new_sum += new_results[i];
    }

    printf("[377b] n_tasks=%ld n_skipped=%ld n_threads=%d ref_sum=%llu new_sum=%llu mismatches=%ld\n",
           n_tasks, n_skipped, n_threads, (unsigned long long)ref_sum, (unsigned long long)new_sum, mismatches);

    free(ld_arr); free(rd_arr); free(col_arr); free(free_arr);
    free(ctrl0_arr); free(markctrl_arr); free(w_lo_arr);
    free(ref_results); free(new_results); free(qbuf); free(schedules); free(skip);

    if (n_tasks - n_skipped < n_tasks / 10) {
        fprintf(stderr, "[377b] WARN: %ld/%ld tasks skipped (schedule precompute cap) -- "
                "too few valid tasks to be a meaningful test; consider a different RNG shape.\n",
                n_skipped, n_tasks);
    }
    if (mismatches != 0) {
        printf("===== 377b: FAIL (%ld/%ld task mismatches) =====\n", mismatches, n_tasks - n_skipped);
        return 1;
    }
    printf("===== 377b: PASS (all %ld valid tasks byte-identical to process_one_task_reference, %ld skipped) =====\n",
           n_tasks - n_skipped, n_skipped);
    return 0;
}
#endif /* !__CUDACC__ */
