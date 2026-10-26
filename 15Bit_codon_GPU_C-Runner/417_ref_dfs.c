/*
 * 417_ref_dfs.c -- an independent reference for the packed-frame kernels: the same per-record
 * semantics (schedule precompute, mark arm, future check, j-mark, terminal count) written as a
 * plain RECURSIVE DFS with no explicit stack, no packed frames, no dead-bit assumptions and no
 * root fast path. It exists because 392_kernel_maxd16 (the 16-byte-frame reference used for
 * N <= 24) holds 15 frames and cannot run N = 25 (required_maxd 17).
 *
 * What is shared with the kernel and what is not:
 *   shared   the schedule precompute (host logic, copied from 363/415: META_NEXT, IS_* masks,
 *            block codes) and the arithmetic of one step (nld / nrd / ncol / nf, MARK_T64)
 *   NOT      everything the 12-byte layouts touch: push / pop, the frame fields, the dead-bit
 *            claims, depth bookkeeping, the root fast path, cur_x. Here every level is a C
 *            call frame holding ld / rd / col / avail in full 32-bit registers, nf is always
 *            bm & ~(nld | nrd | ncol), and avail is never reconstructed.
 * So an equality against this program checks the stack mechanics of a layout, which is what
 * 392 checked for 402..424; it does not re-derive the schedule semantics (those were matched
 * against the Codon GPU kernel in 391 and hold for every layout since).
 *
 * CLI and output are the CPU harness's: <N> <in_soa7_bin> <out_results_bin> [max_records]
 * -> 8 bytes (u64 LE) per record, "[kernel-cputest-done] N=.. records=.. out=.. total_sum=..".
 * Build: gcc -O2 -fopenmp -o 417_ref_dfs 417_ref_dfs.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static const uint32_t IS_BASE_MASK        = 69222408u;
static const uint32_t IS_JMARK_MASK       = 4u;
static const uint32_t IS_MARK_MASK        = 199209203u;
static const uint32_t IS_P5_MASK          = 3840u;
static const uint32_t SEL2_MASK           = 34742338u;
static const uint32_t BLOCK_CODE_B0_MASK  = 173707345u;
static const uint32_t BLOCK_CODE_B1_MASK  = 12689458u;
static const uint32_t BLOCK_CODE_B2_MASK  = 18088064u;
static const uint32_t OP_STEP3_MASK       = 24u;
static const uint64_t MARK_T64 = 0x0202062B17062202ULL;
static const uint8_t META_NEXT[28] = {1,2,3,3,2,6,2,2,0,4,5,7,13,14,14,14,17,14,14,20,21,21,21,25,21,21,26,26};

struct task {
    uint64_t schedule64;
    uint32_t child_jmark_mask, future_check_mask;
    int      terminal_depth;
    uint32_t terminal_base14;
    uint32_t bm, n4;
    uint64_t total;
};

/* one DFS level: place every candidate of `avail` in this row and recurse */
static void dfs(struct task *t, int depth, uint32_t ld, uint32_t rd, uint32_t col, uint32_t avail) {
    const uint32_t nibble = (uint32_t)(t->schedule64 >> (depth * 4)) & 15u;
    const uint32_t block_code = nibble & 7u;
    while (avail) {
        const uint32_t bit = avail & (0u - avail);
        avail ^= bit;
        uint32_t nld, nrd;
        if (block_code != 0u) {
            const uint32_t mt = (uint32_t)(MARK_T64 >> (block_code * 8u));
            const uint32_t step = mt & 3u;
            nld = ((ld | bit) << step) | ((mt >> 2) & 3u);
            nrd = ((rd | bit) >> step) | (((mt >> 4) & 3u) * t->n4);
        } else {
            nld = (ld | bit) << 1;
            nrd = (rd | bit) >> 1;
        }
        const uint32_t ncol = col | bit;
        uint32_t nf = t->bm & ~(nld | nrd | ncol);
        if (nf == 0u) continue;
        if ((t->future_check_mask >> depth) & 1u) {
            if ((t->bm & ~((nld << 1) | (nrd >> 1) | ncol)) == 0u) continue;
        }
        if (depth == t->terminal_depth) {
            t->total += t->terminal_base14 ? (((nf & ~1u) != 0u) ? 1u : 0u) : 1u;
            continue;
        }
        const uint32_t jm = (t->child_jmark_mask >> depth) & 1u;
        nf &= ~jm;
        if (nf == 0u) continue;
        nld |= jm;
        dfs(t, depth + 1, nld, nrd, ncol, nf);
    }
}

static uint64_t solve_record(uint32_t root_ld, uint32_t root_rd, uint32_t root_col, uint32_t root_a_in,
                             uint32_t ctrl0, uint32_t markctrl, uint32_t w_lo, uint32_t bm, uint32_t n4) {
    const uint32_t jmark = markctrl & 31u, endm = (markctrl >> 5) & 31u;
    const uint32_t mark1 = (markctrl >> 10) & 31u, mark2 = (markctrl >> 15) & 31u;
    uint32_t root_a = root_a_in & bm;
    if (root_a == 0u) return 0;

    /* schedule precompute -- as in the kernel (host logic) */
    uint32_t raw = ctrl0; int sdepth = 0;
    uint32_t lo = 0, hi = 0, child_jmark_mask = 0, future_check_mask = 0, root_action = 0, term_b14 = 0;
    int term_parent = 0;
    for (;;) {
        if (sdepth > 1000000) { fprintf(stderr, "[SCHEDULE-RUNAWAY] ctrl0=%u markctrl=%u\n", ctrl0, markctrl); abort(); }
        uint32_t fu = raw & 31u, rowv = (raw >> 5) & 31u;
        if (((IS_P5_MASK >> fu) & 1u) && rowv == mark1) fu = META_NEXT[fu];
        uint32_t action = 0, nibble = 0, fraw = 0, fcv = 0;
        if (((IS_BASE_MASK >> fu) & 1u) && rowv == endm) {
            action = (fu == 14u) ? 3u : 2u;
        } else {
            uint32_t ism = (IS_MARK_MASK >> fu) & 1u, code = 0, step = 1, usef = 1u - ism, nextf = fu;
            if (ism) {
                uint32_t mv = ((SEL2_MASK >> fu) & 1u) ? mark2 : mark1;
                if (rowv == mv) {
                    code = ((BLOCK_CODE_B0_MASK >> fu) & 1u) | (((BLOCK_CODE_B1_MASK >> fu) & 1u) << 1) | (((BLOCK_CODE_B2_MASK >> fu) & 1u) << 2);
                    step = 2u + ((OP_STEP3_MASK >> code) & 1u); usef = 0; nextf = META_NEXT[fu];
                }
            }
            if (((IS_JMARK_MASK >> fu) & 1u) && rowv == jmark) { action = 1u; nextf = META_NEXT[fu]; }
            uint32_t crow = rowv + step;
            if (usef && crow < endm) fcv = 1u;
            nibble = code | (fcv << 3); fraw = nextf | (crow << 5);
        }
        if (sdepth == 0) root_action = action;
        else {
            if (action == 1u) child_jmark_mask |= 1u << (sdepth - 1);
            else if (action >= 2u) { term_parent = sdepth - 1; term_b14 = (action == 3u); }
        }
        if (action >= 2u) break;
        if (fcv) future_check_mask |= 1u << sdepth;
        if (sdepth < 8) lo |= nibble << (sdepth * 4); else hi |= nibble << ((sdepth - 8) * 4);
        raw = fraw; sdepth++;
    }
    if (root_action == 2u) return w_lo;
    if (root_action == 3u) return (((root_a & ~1u) != 0u) ? 1u : 0u) * (uint64_t)w_lo;
    if (root_action == 1u) { root_a &= ~1u; if (root_a == 0u) return 0; root_ld |= 1u; }

    struct task t;
    t.schedule64 = ((uint64_t)hi << 32) | lo;
    t.child_jmark_mask = child_jmark_mask; t.future_check_mask = future_check_mask;
    t.terminal_depth = term_parent; t.terminal_base14 = term_b14;
    t.bm = bm; t.n4 = n4; t.total = 0;
    dfs(&t, 0, root_ld, root_rd, root_col, root_a);   /* col without ~bm: nf is bm & ~(...) explicitly */
    return t.total * (uint64_t)w_lo;
}

static uint32_t rd32(const unsigned char *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) { fprintf(stderr, "Usage: %s <N> <in_soa7_bin> <out_results_bin> [max_records]\n", argv[0]); return 1; }
    int64_t N = atoll(argv[1]);
    if (N < 5 || N > 31) { fprintf(stderr, "[ref-dfs] N=%lld out of range\n", (long long)N); return 3; }
    FILE *fin = fopen(argv[2], "rb"); if (!fin) { fprintf(stderr, "ERROR: cannot open %s\n", argv[2]); return 1; }
    fseek(fin, 0, SEEK_END); long fsize = ftell(fin); rewind(fin);
    if (fsize < 0 || fsize % 28) { fprintf(stderr, "ERROR: size %ld not a multiple of 28\n", fsize); return 1; }
    int64_t m = fsize / 28, max_records = (argc == 5) ? atoll(argv[4]) : -1;
    if (max_records >= 0 && max_records < m) m = max_records;
    uint32_t *f = (uint32_t*)malloc((size_t)m * 7 * sizeof(uint32_t)); uint64_t *out = (uint64_t*)malloc((size_t)m * sizeof(uint64_t));
    if (!f || !out) { fprintf(stderr, "ERROR: out of memory\n"); return 1; }
    unsigned char buf[28];
    for (int64_t i = 0; i < m; i++) { if (fread(buf, 1, 28, fin) != 28) { fprintf(stderr, "ERROR: short read\n"); return 1; } for (int k = 0; k < 7; k++) f[i * 7 + k] = rd32(buf + 4 * k); }
    fclose(fin);
    const uint32_t bm = (uint32_t)((1ULL << N) - 1), n4 = (uint32_t)(1ULL << (N - 4));
    uint64_t total = 0;
#ifdef _OPENMP
    fprintf(stderr, "[openmp] running with %d threads\n", omp_get_max_threads());
    #pragma omp parallel for schedule(dynamic, 64) reduction(+:total)
#endif
    for (int64_t i = 0; i < m; i++) {
        const uint32_t *r = f + i * 7;
        uint64_t c = solve_record(r[0], r[1], r[2], r[4], r[3], r[5], r[6], bm, n4);
        out[i] = c; total += c;
    }
    FILE *fo = fopen(argv[3], "wb"); if (!fo) { fprintf(stderr, "ERROR: cannot open %s\n", argv[3]); return 1; }
    for (int64_t i = 0; i < m; i++) { unsigned char o[8]; for (int b = 0; b < 8; b++) o[b] = (unsigned char)(out[i] >> (8 * b)); if (fwrite(o, 1, 8, fo) != 8) return 1; }
    fclose(fo);
    printf("[kernel-cputest-done] N=%lld records=%lld out=%s total_sum=%llu\n", (long long)N, (long long)m, argv[3], (unsigned long long)total);
    return 0;
}
