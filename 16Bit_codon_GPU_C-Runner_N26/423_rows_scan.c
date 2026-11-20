/*
 * 423_rows_scan.c -- host-only scan of a CRunner input (28 B/record) for the 423 search (NQ_LOGIC=mrv).
 * For every record it walks the schedule exactly as process_one_task_mrv does and builds the static row
 * masks (candidate columns of each row against the pre-placed queens only). It reports what the 423
 * kernel assumes about the WHOLE input, which the per-record equalities only sample:
 *
 *   rows        histogram of the number of row slots (must be <= MRV_MAXROWS = 18)
 *   pos         the largest row position (must be <= 31: 5 bits in the row word and in the frame)
 *   cover>rows  records whose static masks cover MORE columns than there are rows.   MUST BE 0:
 *               the 423 leaf adds popcount(last row's candidates); the DFS adds 1 if that set is not
 *               empty. They agree when a last row can never keep two candidates, i.e. when the rows can
 *               only ever use as many distinct columns as there are rows.
 *   info        records that the walk already decides (root action 2 / 3, an empty static row),
 *               records where popcount(free columns) != rows, records with column 0 free but in no row
 *
 * Usage: 423_rows_scan <N> <in_soa7_bin>
 * Build: gcc -O2 -o 423_rows_scan 423_rows_scan.c
 * Output: one "[rows-scan] ..." line; exit status 0 when the three conditions hold, 2 otherwise.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static const uint32_t IS_BASE_MASK = 69222408u, IS_JMARK_MASK = 4u, IS_MARK_MASK = 199209203u, IS_P5_MASK = 3840u;
static const uint32_t SEL2_MASK = 34742338u, BLOCK_CODE_B0_MASK = 173707345u, BLOCK_CODE_B1_MASK = 12689458u;
static const uint32_t BLOCK_CODE_B2_MASK = 18088064u, OP_STEP3_MASK = 24u;
static const uint64_t MARK_T64 = 0x0202062B17062202ULL;
static const uint8_t META_NEXT[28] = {1,2,3,3,2,6,2,2,0,4,5,7,13,14,14,14,17,14,14,20,21,21,21,25,21,21,26,26};

#define MAXROWS_SCAN 33
/* returns the number of row slots (0: decided by the root action or root_a == 0; -1: empty static row; -2: runaway) */
static int rows_of(uint32_t root_ld, uint32_t root_rd, uint32_t root_col, uint32_t root_a, uint32_t ctrl0, uint32_t markctrl,
                   uint32_t bm, uint32_t n4, uint32_t *cover, uint32_t *maxpos) {
    const uint32_t jmark = markctrl & 31u, endm = (markctrl >> 5) & 31u, mark1 = (markctrl >> 10) & 31u, mark2 = (markctrl >> 15) & 31u;
    uint32_t raw = ctrl0, L = root_ld, R = root_rd, pos = 0, cov = 0;
    int d = 0;
    root_a &= bm;
    if (root_a == 0u) return 0;
    for (;;) {
        if (d >= MAXROWS_SCAN - 1) return -2;
        uint32_t fu = raw & 31u, rowv = (raw >> 5) & 31u, action = 0, code = 0, step = 1, nextf;
        if (((IS_P5_MASK >> fu) & 1u) && rowv == mark1) fu = META_NEXT[fu];
        nextf = fu;
        if (((IS_BASE_MASK >> fu) & 1u) && rowv == endm) {
            action = (fu == 14u) ? 3u : 2u;
        } else {
            if ((IS_MARK_MASK >> fu) & 1u) {
                uint32_t mv = ((SEL2_MASK >> fu) & 1u) ? mark2 : mark1;
                if (rowv == mv) {
                    code = ((BLOCK_CODE_B0_MASK >> fu) & 1u) | (((BLOCK_CODE_B1_MASK >> fu) & 1u) << 1) | (((BLOCK_CODE_B2_MASK >> fu) & 1u) << 2);
                    step = 2u + ((OP_STEP3_MASK >> code) & 1u); nextf = META_NEXT[fu];
                }
            }
            if (((IS_JMARK_MASK >> fu) & 1u) && rowv == jmark) { action = 1u; nextf = META_NEXT[fu]; }
        }
        if (d == 0) {
            if (action >= 2u) return 0;
            if (action == 1u) { root_a &= ~1u; if (root_a == 0u) return 0; L |= 1u; }
            cov |= root_a;
        } else {
            uint32_t rc = bm & ~(L | R | root_col);
            if (action >= 2u) { if (action == 3u) rc &= ~1u; }
            else { const uint32_t jm = (action == 1u); rc &= ~jm; L |= jm; }
            if (rc == 0u) return -1;
            cov |= rc;
            if (pos > *maxpos) *maxpos = pos;
        }
        if (action >= 2u) { *cover = cov; return d + 1; }
        if (code) {
            const uint32_t mt = (uint32_t)(MARK_T64 >> (code * 8u)); const uint32_t s = mt & 3u;
            L = (L << s) | ((mt >> 2) & 3u); R = (R >> s) | (((mt >> 4) & 3u) * n4); pos += s;
        } else { L <<= 1; R >>= 1; pos += 1u; }
        raw = nextf | ((rowv + step) << 5); d++;
    }
}

static uint32_t rd32(const unsigned char *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "Usage: %s <N> <in_soa7_bin>\n", argv[0]); return 1; }
    int N = atoi(argv[1]);
    if (N < 5 || N > 27) { fprintf(stderr, "N=%d out of range (5..27)\n", N); return 1; }
    FILE *f = fopen(argv[2], "rb"); if (!f) { perror(argv[2]); return 1; }
    const uint32_t bm = (uint32_t)((1ULL << N) - 1), n4 = (uint32_t)(1ULL << (N - 4));
    long hist[MAXROWS_SCAN + 1]; memset(hist, 0, sizeof hist);
    long m = 0, decided = 0, empty_row = 0, runaway = 0, cover_gt = 0, cover_lt = 0, free_ne = 0, col0_phantom = 0;
    uint32_t maxpos = 0; int maxrows = 0;
    unsigned char buf[28];
    while (fread(buf, 1, 28, f) == 28) {
        uint32_t r[7]; for (int k = 0; k < 7; k++) r[k] = rd32(buf + 4 * k);
        uint32_t cover = 0;
        int n = rows_of(r[0], r[1], r[2], r[4], r[3], r[5], bm, n4, &cover, &maxpos);
        m++;
        if (n == 0) { decided++; continue; }
        if (n == -1) { empty_row++; continue; }
        if (n == -2) { runaway++; continue; }
        hist[n]++; if (n > maxrows) maxrows = n;
        const int pc = __builtin_popcount(cover), fc = __builtin_popcount(bm & ~r[2]);
        if (pc > n) cover_gt++;
        if (pc < n) cover_lt++;
        if (fc != n) free_ne++;
        if (((bm & ~r[2]) & 1u) && !(cover & 1u)) col0_phantom++;
    }
    fclose(f);
    printf("[rows-scan] N=%d records=%ld searched=%ld decided_at_root=%ld empty_static_row=%ld runaway=%ld max_rows=%d max_pos=%u "
           "cover_gt_rows=%ld cover_lt_rows=%ld free_ne_rows=%ld col0_free_but_unused=%ld\n",
           N, m, m - decided - empty_row - runaway, decided, empty_row, runaway, maxrows, maxpos, cover_gt, cover_lt, free_ne, col0_phantom);
    printf("[rows-scan-histogram]"); for (int n = 0; n <= MAXROWS_SCAN; n++) if (hist[n]) printf(" %d:%ld", n, hist[n]); printf("\n");
    return (runaway == 0 && cover_gt == 0 && maxrows <= 18 && maxpos <= 31u) ? 0 : 2;
}
