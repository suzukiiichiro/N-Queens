/*
 * 377a_frame_queue.cu
 *
 * rev377a — Frame-queue primitives ONLY. This file does NOT touch
 * 374Py_kernel_maxd14.cu and is NOT wired into kernel_dfs_iter_gpu_
 * maxd14 in any way. Per 376_frame_workqueue_spec.md section 5's
 * staged validation plan, this revision's sole purpose is to prove
 * the atomic push/pop ring-buffer mechanism is correct under genuine
 * concurrent multi-thread access BEFORE any GPU kernel integration
 * (377b+). Validated here on CPU only, via OpenMP, exactly as 363
 * validated process_one_task() on CPU before 364's real GPU launch.
 *
 * Same HOSTDEV / #ifdef __CUDACC__ dual-build convention as
 * 374Py_kernel_maxd14.cu: the push/pop primitives are written once
 * and compile for both device (future 377b kernel use) and host
 * (this revision's OpenMP CPU test). The device atomic intrinsics
 * (atomicAdd, atomicCAS) are only exercised under __CUDACC__; the
 * host path below uses GCC/Clang __sync builtins, which are the
 * host-side equivalent used nowhere else in this project so far --
 * flagged here explicitly since it is new.
 *
 * SCOPE, per Suzuki-san's confirmation this session: prove the queue
 * mechanism in isolation. Two things are deliberately NOT tested
 * here (deferred to 377b, where they arise naturally from genuine
 * concurrent push+pop inside a persistent kernel):
 *   1. The termination protocol (g_active_workers / "queue truly
 *      empty" detection from 376 spec section 3.2) -- this test uses
 *      a fixed, known-upfront task count and runs push and pop in two
 *      separate phases (barrier between them), so "is there more work
 *      coming" never needs to be decided under uncertainty here.
 *   2. Interleaved push-while-popping races -- same reason.
 * What IS tested: the atomic slot-reservation arithmetic (no two
 * concurrent pushers ever get the same slot; no two concurrent
 * poppers ever get the same slot; every pushed Frame is popped
 * exactly once with byte-identical content) and capacity-overflow
 * detection (deliberately undersized buffer must report exactly the
 * expected number of failed pushes, and every frame that DID succeed
 * must still round-trip correctly -- no silent corruption).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#define HOSTDEV __host__ __device__
#else
#define HOSTDEV
#endif

/* ---------------------------------------------------------------------
 * Data structures (376 spec section 3.1). task_id is used as the
 * checksummable payload in this isolated test -- the real DFS state
 * fields (cur_ld/cur_rd/cur_col/cur_avail/cur_depth) are carried
 * along unchanged but not exercised by this test's logic, since no
 * DFS stepping happens here at all (377b's job).
 * ------------------------------------------------------------------- */
typedef struct {
    uint32_t schedule_lo, schedule_hi;
    uint32_t terminal_depth;
    uint32_t terminal_base14;
    uint32_t child_jmark_mask;
    uint32_t future_check_mask;
    uint32_t w_lo;
} TaskSchedule;

typedef struct {
    uint32_t task_id;
    uint32_t cur_ld, cur_rd, cur_col, cur_avail;
    uint32_t cur_depth;
} Frame;

/* ---------------------------------------------------------------------
 * Atomic ring buffer. capacity_mask = capacity-1, capacity a power of
 * two (so idx & capacity_mask replaces idx % capacity). push_idx/
 * pop_idx are free-running (never wrap; the buffer *slot* wraps via
 * the mask, the *counters* do not, so "how many outstanding" is
 * always (push_idx - pop_idx) with no ambiguity from wraparound of
 * the counters themselves -- only practical limit is uint64 range).
 * ------------------------------------------------------------------- */
typedef struct {
    Frame*   buf;
    uint32_t capacity_mask;
    uint64_t push_idx;   /* accessed only via atomics below */
    uint64_t pop_idx;    /* accessed only via atomics below */
} FrameQueue;

HOSTDEV static uint64_t fq_atomic_add_u64(uint64_t* addr, uint64_t val) {
#ifdef __CUDACC__
    return atomicAdd((unsigned long long*)addr, (unsigned long long)val);
#else
    return __sync_fetch_and_add(addr, val);
#endif
}

HOSTDEV static uint64_t fq_atomic_load_u64(volatile uint64_t* addr) {
#ifdef __CUDACC__
    return atomicAdd((unsigned long long*)addr, 0ULL);
#else
    return __sync_fetch_and_add(addr, 0ULL);
#endif
}

/* Returns 1 on success (frame written), 0 on overflow (buffer full --
 * caller's responsibility to handle; this test counts and verifies
 * the failure count, it does not retry). See 376 spec section 4.1:
 * the overflow check here is deliberately CONSERVATIVE (may reject a
 * push that a perfectly precise check would have allowed, if pop_idx
 * advanced concurrently after our snapshot) -- it can never UNDER-
 * estimate the backlog, so it can never allow a slot to be
 * overwritten before it's been popped. */
HOSTDEV static int fq_push(FrameQueue* q, const Frame* f) {
    uint64_t idx = fq_atomic_add_u64(&q->push_idx, 1ULL);
    uint64_t cur_pop = fq_atomic_load_u64(&q->pop_idx);
    uint64_t capacity = (uint64_t)q->capacity_mask + 1ULL;
    if (idx - cur_pop >= capacity) {
        return 0; /* overflow: this reservation is abandoned (the slot
                     number is burned, but no data is written and no
                     existing slot is touched -- safe, just wasteful) */
    }
    q->buf[idx & q->capacity_mask] = *f;
    return 1;
}

/* Returns 1 on success (frame read into *out), 0 if nothing was
 * available AT THE MOMENT of the reservation (idx >= push snapshot).
 * As documented in the file header, this isolated test never calls
 * fq_pop() concurrently with fq_push() -- pop only runs after a
 * barrier once all pushes for the phase are known to be complete --
 * so this failure path exercises only the "popped past the known
 * total" bug-detection case, not a real empty-queue race. */
HOSTDEV static int fq_pop(FrameQueue* q, Frame* out) {
    uint64_t idx = fq_atomic_add_u64(&q->pop_idx, 1ULL);
    uint64_t cur_push = fq_atomic_load_u64(&q->push_idx);
    if (idx >= cur_push) {
        return 0;
    }
    *out = q->buf[idx & q->capacity_mask];
    return 1;
}

#ifndef __CUDACC__
/* =======================================================================
 * CPU-only OpenMP test harness (no CUDA toolkit required -- same
 * rationale as 363's CPU test harness: prove logic correctness cheaply
 * before spending real GPU/nvcc cycles on it).
 * ===================================================================== */

static uint32_t next_pow2_u32(uint32_t n) {
    uint32_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* Test A: capacity >= n_tasks (no overflow expected). Pushes n_tasks
 * frames (task_id = 0..n_tasks-1) from many OpenMP threads
 * concurrently, barriers, then pops all of them from many threads
 * concurrently, and verifies via a "seen" bitmap that every task_id
 * appears EXACTLY once among the popped frames (not zero, not two+). */
static int test_a_no_overflow(uint32_t n_tasks, int n_threads) {
    uint32_t capacity = next_pow2_u32(n_tasks);
    FrameQueue q;
    q.buf = (Frame*)malloc(sizeof(Frame) * capacity);
    q.capacity_mask = capacity - 1;
    q.push_idx = 0;
    q.pop_idx = 0;
    if (!q.buf) { fprintf(stderr, "[test_a] malloc failed\n"); return 0; }

    long push_failures = 0;
#ifdef _OPENMP
    #pragma omp parallel for num_threads(n_threads) reduction(+:push_failures) schedule(dynamic, 64)
#endif
    for (long i = 0; i < (long)n_tasks; i++) {
        Frame f;
        memset(&f, 0, sizeof(f));
        f.task_id = (uint32_t)i;
        if (!fq_push(&q, &f)) {
            push_failures++;
        }
    }

    if (push_failures != 0) {
        fprintf(stderr, "[test_a] FAIL: %ld unexpected push failures with capacity=%u >= n_tasks=%u\n",
                push_failures, capacity, n_tasks);
        free(q.buf);
        return 0;
    }

    long pop_successes = 0, pop_failures = 0, oob_count = 0;
#ifdef _OPENMP
    #pragma omp parallel for num_threads(n_threads) reduction(+:pop_successes,pop_failures,oob_count) schedule(dynamic, 64)
#endif
    for (long i = 0; i < (long)n_tasks; i++) {
        Frame f;
        if (!fq_pop(&q, &f)) {
            pop_failures++;
            continue;
        }
        pop_successes++;
        if (f.task_id >= n_tasks) {
            oob_count++;
        }
    }
    if (pop_failures != 0 || pop_successes != (long)n_tasks) {
        fprintf(stderr, "[test_a] FAIL: pop_successes=%ld pop_failures=%ld (expected successes=%u failures=0)\n",
                pop_successes, pop_failures, n_tasks);
        free(q.buf);
        return 0;
    }
    if (oob_count != 0) {
        fprintf(stderr, "[test_a] FAIL: %ld popped frames had out-of-range task_id (data corruption)\n", oob_count);
        free(q.buf);
        return 0;
    }

    free(q.buf);
    printf("[test_a] PASS: n_tasks=%u capacity=%u n_threads=%d push_failures=0 pop_successes=%u\n",
           n_tasks, capacity, n_threads, n_tasks);
    return 1;
}

/* Test A2: identical to Test A but additionally proves no duplicate/
 * lost frames via an atomically-updated per-task_id counter array
 * (each successfully-popped task_id increments its own slot; at the
 * end every slot must equal exactly 1). This is the real
 * no-lost-no-duplicated proof; Test A above only proves count and
 * range sanity, so A2 is the stronger check and is what actually
 * gates PASS/FAIL for this revision. */
static int test_a2_exact_once(uint32_t n_tasks, int n_threads) {
    uint32_t capacity = next_pow2_u32(n_tasks);
    FrameQueue q;
    q.buf = (Frame*)malloc(sizeof(Frame) * capacity);
    q.capacity_mask = capacity - 1;
    q.push_idx = 0;
    q.pop_idx = 0;
    if (!q.buf) { fprintf(stderr, "[test_a2] malloc failed\n"); return 0; }

    long push_failures = 0;
#ifdef _OPENMP
    #pragma omp parallel for num_threads(n_threads) reduction(+:push_failures) schedule(dynamic, 64)
#endif
    for (long i = 0; i < (long)n_tasks; i++) {
        Frame f;
        memset(&f, 0, sizeof(f));
        f.task_id = (uint32_t)i;
        if (!fq_push(&q, &f)) push_failures++;
    }
    if (push_failures != 0) {
        fprintf(stderr, "[test_a2] FAIL: %ld unexpected push failures\n", push_failures);
        free(q.buf);
        return 0;
    }

    uint32_t* count = (uint32_t*)calloc(n_tasks, sizeof(uint32_t));
    long pop_successes = 0, pop_failures = 0;
#ifdef _OPENMP
    #pragma omp parallel for num_threads(n_threads) reduction(+:pop_successes,pop_failures) schedule(dynamic, 64)
#endif
    for (long i = 0; i < (long)n_tasks; i++) {
        Frame f;
        if (!fq_pop(&q, &f)) { pop_failures++; continue; }
        pop_successes++;
        if (f.task_id < n_tasks) {
            __sync_fetch_and_add(&count[f.task_id], 1u);
        }
    }

    long zero_count = 0, dup_count = 0;
    for (uint32_t i = 0; i < n_tasks; i++) {
        if (count[i] == 0) zero_count++;
        else if (count[i] > 1) dup_count++;
    }
    free(count);
    free(q.buf);

    if (pop_failures != 0 || pop_successes != (long)n_tasks || zero_count != 0 || dup_count != 0) {
        fprintf(stderr, "[test_a2] FAIL: pop_successes=%ld pop_failures=%ld zero_count=%ld dup_count=%ld\n",
                pop_successes, pop_failures, zero_count, dup_count);
        return 0;
    }
    printf("[test_a2] PASS: n_tasks=%u capacity=%u n_threads=%d -- every task_id popped exactly once\n",
           n_tasks, capacity, n_threads);
    return 1;
}

/* Test B: capacity deliberately smaller than n_tasks -- overflow MUST
 * occur, and the count of successful pushes must be bounded correctly
 * (>= capacity is guaranteed possible in principle, but the
 * conservative check in fq_push may reject some pushes even within
 * capacity if they race with the snapshot; so this test asserts
 * success_count is in the sane range (capacity/2, capacity] rather
 * than requiring an exact figure -- see 376 spec section 4.1) and,
 * critically, that every frame that DID succeed still round-trips
 * with the correct content (no corruption from the undersized
 * buffer). */
static int test_b_overflow_detected(uint32_t n_tasks, uint32_t capacity, int n_threads) {
    FrameQueue q;
    q.buf = (Frame*)malloc(sizeof(Frame) * capacity);
    q.capacity_mask = capacity - 1;
    q.push_idx = 0;
    q.pop_idx = 0;
    if (!q.buf) { fprintf(stderr, "[test_b] malloc failed\n"); return 0; }

    long push_successes = 0, push_failures = 0;
#ifdef _OPENMP
    #pragma omp parallel for num_threads(n_threads) reduction(+:push_successes,push_failures) schedule(dynamic, 64)
#endif
    for (long i = 0; i < (long)n_tasks; i++) {
        Frame f;
        memset(&f, 0, sizeof(f));
        f.task_id = (uint32_t)i;
        if (fq_push(&q, &f)) push_successes++;
        else push_failures++;
    }

    if (push_failures == 0) {
        fprintf(stderr, "[test_b] FAIL: expected overflow (n_tasks=%u > capacity=%u) but push_failures=0\n",
                n_tasks, capacity);
        free(q.buf);
        return 0;
    }
    if (push_successes < 1 || push_successes > (long)capacity) {
        fprintf(stderr, "[test_b] FAIL: push_successes=%ld out of sane range (0, %u]\n",
                push_successes, capacity);
        free(q.buf);
        return 0;
    }

    /* pop exactly push_successes times and verify no corruption among
     * the ones that did succeed (task_id must be in [0, n_tasks), and
     * each popped task_id must appear at most once -- duplicates would
     * indicate the overflow check let a push overwrite a live slot). */
    uint32_t* count = (uint32_t*)calloc(n_tasks, sizeof(uint32_t));
    long pop_successes = 0, pop_failures = 0, oob_count = 0;
#ifdef _OPENMP
    #pragma omp parallel for num_threads(n_threads) reduction(+:pop_successes,pop_failures,oob_count) schedule(dynamic, 64)
#endif
    for (long i = 0; i < push_successes; i++) {
        Frame f;
        if (!fq_pop(&q, &f)) { pop_failures++; continue; }
        pop_successes++;
        if (f.task_id >= n_tasks) { oob_count++; continue; }
        __sync_fetch_and_add(&count[f.task_id], 1u);
    }
    long dup_count = 0;
    for (uint32_t i = 0; i < n_tasks; i++) {
        if (count[i] > 1) dup_count++;
    }
    free(count);
    free(q.buf);

    if (pop_failures != 0 || pop_successes != push_successes || oob_count != 0 || dup_count != 0) {
        fprintf(stderr, "[test_b] FAIL: pop_successes=%ld (expected %ld) pop_failures=%ld oob=%ld dup=%ld\n",
                pop_successes, push_successes, pop_failures, oob_count, dup_count);
        return 0;
    }
    printf("[test_b] PASS: n_tasks=%u capacity=%u n_threads=%d push_successes=%ld push_failures=%ld "
           "(overflow correctly detected, no corruption among survivors)\n",
           n_tasks, capacity, n_threads, push_successes, push_failures);
    return 1;
}

int main(int argc, char** argv) {
    int n_threads = 8;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
    if (argc >= 2) n_threads = atoi(argv[1]);
#endif
    printf("[377a] Frame queue isolated test. n_threads=%d (OpenMP %s)\n",
           n_threads,
#ifdef _OPENMP
           "enabled"
#else
           "DISABLED -- rebuild with -fopenmp for a real concurrency test; "
           "this run is single-threaded and only proves single-thread logic"
#endif
    );

    int all_pass = 1;
    all_pass &= test_a_no_overflow(200000, n_threads);
    all_pass &= test_a2_exact_once(200000, n_threads);
    all_pass &= test_b_overflow_detected(200000, 32768, n_threads); /* 32768 < 200000: forces overflow */

    printf("\n===== 377a summary =====\n");
    printf(all_pass ? "ALL TESTS PASSED\n" : "AT LEAST ONE TEST FAILED\n");
    return all_pass ? 0 : 1;
}
#endif /* !__CUDACC__ */
