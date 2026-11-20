/* probe.c -- instrumented copy of the 419_ref_dfs semantics (plain arrays, recursive).
 * Measures, on real N=25 records:
 *   - steps per depth, immediate dead / fc dead / descents / terminal hits
 *   - steps spent inside maximal zero-solution subtrees (oracle-pruning bound)
 *   - effect of stronger pruning:  MODE 0 = kernel's pruning (1-row future check)
 *                                  MODE 1 = every remaining row keeps >= 1 candidate
 *                                  MODE 2 = MODE 1 + every free column is still reachable
 * Usage: probe <N> <in_soa7> <first> <count> <stride> <mode>
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

#define MD 32
static uint8_t  g_bc[MD];
static uint32_t g_jm, g_fc, g_bm, g_n4, g_b14;
static int      g_term, g_mode, g_maxla = 99;

static uint64_t st_steps[MD], st_imm[MD], st_fc[MD], st_la[MD], st_desc[MD], st_term[MD];
static uint64_t st_deadroot[MD], st_deadsteps[MD], st_nodes[MD], st_deadnodes_any[MD];
static uint64_t g_rowmismatch;

typedef struct { uint64_t sol, steps; } res_t;

static inline void advance(int e, uint32_t *L, uint32_t *R) {
    const uint32_t c = g_bc[e];
    if (c) { const uint32_t mt = (uint32_t)(MARK_T64 >> (c * 8u)); const uint32_t s = mt & 3u;
             *L = (*L << s) | ((mt >> 2) & 3u); *R = (*R >> s) | (((mt >> 4) & 3u) * g_n4); }
    else   { *L <<= 1; *R >>= 1; }
}

/* lookahead from the state just after placing at depth d (nld already carries jm of depth d) */
static inline int lookahead_dead(int d, uint32_t nld, uint32_t nrd, uint32_t ncol, uint32_t nf) {
    uint32_t L = nld, R = nrd, cover = nf;
    for (int e = d + 1; e <= g_term; e++) {
        advance(e, &L, &R);                     /* row e passes with no queen of its own */
        uint32_t cand = g_bm & ~(L | R | ncol);
        if (e == g_term) { if (g_b14) cand &= ~1u; }
        else { const uint32_t jm = (g_jm >> e) & 1u; cand &= ~jm; L |= jm; }
        if (cand == 0u) return 1;
        cover |= cand;
    }
    if (g_mode >= 2) { if (((g_bm & ~ncol) & ~cover & ~1u) != 0u) return 1; }
    return 0;
}

static inline int stepof(int e){ const uint32_t c=g_bc[e]; return c ? (int)((uint32_t)(MARK_T64 >> (c*8u)) & 3u) : 1; }
static int propagate_dead(int d, uint32_t nld, uint32_t nrd, uint32_t ncol, uint32_t nf) {
    uint32_t cand[MD]; int pos[MD]; uint8_t fixed[MD]; int n = 0;
    uint32_t L = nld, R = nrd; int p = 0;
    cand[n] = nf; pos[n] = 0; fixed[n] = 0; n++;
    for (int e = d + 1; e <= g_term; e++) {
        p += stepof(e); advance(e, &L, &R);
        uint32_t c = g_bm & ~(L | R | ncol);
        if (e == g_term) { if (g_b14) c &= ~1u; } else { const uint32_t jm = (g_jm >> e) & 1u; c &= ~jm; L |= jm; }
        if (c == 0u) return 1;
        cand[n] = c; pos[n] = p; fixed[n] = 0; n++;
    }
    for (;;) {
        int changed = 0;
        uint32_t once = 0, twice = 0;
        for (int i = 0; i < n; i++) { twice |= once & cand[i]; once |= cand[i]; }
        if (((g_bm & ~ncol) & ~once & ~1u) != 0u) return 1;
        const uint32_t uniq = once & ~twice;                 /* columns offered by exactly one row */
        for (int i = 0; i < n; i++) {
            if (fixed[i]) continue;
            uint32_t c = cand[i];
            const uint32_t u = c & uniq & ~1u;
            if (u) { if (u & (u - 1u)) return 1; c = u; }     /* two columns need this row */
            if ((c & (c - 1u)) == 0u) {                       /* forced */
                cand[i] = c; fixed[i] = 1; changed = 1;
                for (int j = 0; j < n; j++) if (j != i) {
                    const int k = pos[j] > pos[i] ? pos[j] - pos[i] : pos[i] - pos[j];
                    cand[j] &= ~(c | (c << k) | (c >> k));
                    if (cand[j] == 0u) return 1;
                }
                break;
            }
        }
        if (!changed) return 0;
    }
}
static res_t dfs(int depth, uint32_t ld, uint32_t rd, uint32_t col, uint32_t avail) {
    const uint32_t block_code = g_bc[depth];
    res_t me = {0, 0};
    uint64_t pend_steps = 0, pend_cnt = 0;
    st_nodes[depth]++;
    while (avail) {
        const uint32_t bit = avail & (0u - avail);
        avail ^= bit;
        me.steps++; st_steps[depth]++;
        uint32_t nld, nrd;
        if (block_code != 0u) {
            const uint32_t mt = (uint32_t)(MARK_T64 >> (block_code * 8u)); const uint32_t step = mt & 3u;
            nld = ((ld | bit) << step) | ((mt >> 2) & 3u);
            nrd = ((rd | bit) >> step) | (((mt >> 4) & 3u) * g_n4);
        } else { nld = (ld | bit) << 1; nrd = (rd | bit) >> 1; }
        const uint32_t ncol = col | bit;
        uint32_t nf = g_bm & ~(nld | nrd | ncol);
        if (nf == 0u) { st_imm[depth]++; continue; }
        if ((g_fc >> depth) & 1u) {
            if ((g_bm & ~((nld << 1) | (nrd >> 1) | ncol)) == 0u) { st_fc[depth]++; continue; }
        }
        if (depth == g_term) {
            uint64_t s = g_b14 ? (((nf & ~1u) != 0u) ? 1u : 0u) : 1u;
            me.sol += s; if (s) st_term[depth]++; else st_imm[depth]++;
            continue;
        }
        const uint32_t jm = (g_jm >> depth) & 1u;
        nf &= ~jm;
        if (nf == 0u) { st_imm[depth]++; continue; }
        nld |= jm;
        if (g_mode >= 1 && depth <= g_maxla && (g_mode >= 3 ? propagate_dead(depth, nld, nrd, ncol, nf) : lookahead_dead(depth, nld, nrd, ncol, nf))) { st_la[depth]++; continue; }
        st_desc[depth]++;
        res_t ch = dfs(depth + 1, nld, nrd, ncol, nf);
        me.sol += ch.sol; me.steps += ch.steps;
        if (ch.sol == 0) { pend_steps += ch.steps + 1; pend_cnt++; st_deadnodes_any[depth + 1]++; }
    }
    if (me.sol > 0) { st_deadsteps[depth + 1] += pend_steps; st_deadroot[depth + 1] += pend_cnt; }
    return me;
}

/* ---- generic solver (rules 10..14): rows kept as explicit candidate masks, any row order ---- */
static uint64_t g_gsteps; static int g_pol;
/* ---- hybrid (rule 20): rule-14 search (most constrained row first, column cover, forced moves) while more than
 *      SWITCH rows remain, then the kernel-style in-order DFS (incremental ld/rd, 1-row future check) on the rest.
 *      The rows' static masks cand[] already carry every queen placed by the top part. ---- */
static int g_switch = 99, g_toppol = 14;
static uint64_t bot(int j, int n, const uint32_t *cand, const int *pos, uint32_t ld, uint32_t rd, uint32_t col, uint32_t avail) {
    uint64_t sol = 0; const int k = pos[j + 1] - pos[j]; const uint32_t cn = cand[j + 1];
    if (j == n - 2) {
        while (avail) { const uint32_t bit = avail & (0u - avail); avail ^= bit; g_gsteps++;
            sol += (cn & ~(((ld | bit) << k) | ((rd | bit) >> k) | col | bit)) != 0u; }
        return sol;
    }
    const int k2 = pos[j + 2] - pos[j + 1]; const uint32_t cn2 = cand[j + 2];
    while (avail) {
        const uint32_t bit = avail & (0u - avail); avail ^= bit; g_gsteps++;
        const uint32_t nld = (ld | bit) << k, nrd = (rd | bit) >> k, ncol = col | bit;
        const uint32_t nf = cn & ~(nld | nrd | ncol);
        if (nf == 0u) continue;
        if ((cn2 & ~((nld << k2) | (nrd >> k2) | ncol)) == 0u) continue;
        sol += bot(j + 1, n, cand, pos, nld, nrd, ncol, nf);
    }
    return sol;
}
static uint64_t gsolve(int n, const uint32_t *cand, const int *pos, uint32_t freecols) {
    uint32_t cc[MD]; int best = 0;
    if (g_mode == 20 && n <= g_switch) {
        if (n == 1) { const int k = __builtin_popcount(cand[0]); g_gsteps += k; return (uint64_t)k; }
        return bot(0, n, cand, pos, 0u, 0u, 0u, cand[0]);
    }
    if (g_pol == 16 || g_pol == 17) { if (__builtin_popcount(cand[n - 1]) < __builtin_popcount(cand[0])) best = n - 1;
        if (g_pol == 17) { uint32_t once = 0; for (int i = 0; i < n; i++) once |= cand[i]; if ((freecols & ~once & ~1u) != 0u) return 0; }
    } else if (g_pol >= 13) {
        uint32_t once = 0, twice = 0;
        for (int i = 0; i < n; i++) { twice |= once & cand[i]; once |= cand[i]; }
        if ((freecols & ~once & ~1u) != 0u) return 0;
        if (g_pol >= 14) {
            const uint32_t uniq = once & ~twice & ~1u;
            int bp = 99;
            for (int i = 0; i < n; i++) { uint32_t c = cand[i]; const uint32_t u = c & uniq;
                if (u) { if (u & (u - 1u)) return 0; c = u; }
                cc[i] = c; const int pc = __builtin_popcount(c); if (pc < bp) { bp = pc; best = i; } }
            cand = cc;
        } else { int bp = 99; for (int i = 0; i < n; i++) { const int pc = __builtin_popcount(cand[i]); if (pc < bp) { bp = pc; best = i; } } }
    } else if (g_pol == 12) { int bp = 99; for (int i = 0; i < n; i++) { const int pc = __builtin_popcount(cand[i]); if (pc < bp) { bp = pc; best = i; } } }
    uint32_t av = cand[best]; const int pb = pos[best];
    if (n == 1) { const int k = __builtin_popcount(av); g_gsteps += k; return (uint64_t)k; }
    uint32_t nc[MD]; int np[MD]; uint64_t sol = 0;
    for (int j = 0, m = 0; j < n; j++) if (j != best) np[m++] = pos[j];
    while (av) {
        const uint32_t c = av & (0u - av); av ^= c; g_gsteps++;
        int m = 0, ok = 1;
        for (int j = 0; j < n; j++) { if (j == best) continue;
            const int k = pos[j] > pb ? pos[j] - pb : pb - pos[j];
            const uint32_t x = cand[j] & ~(c | (c << k) | (c >> k));
            if (x == 0u) { ok = 0; break; }
            nc[m++] = x; }
        if (ok) sol += gsolve(n - 1, nc, np, freecols & ~c);
    }
    return sol;
}
/* ---- rule 15: rule 14's search with fixed row slots (no array compaction), a distance table, and the kernel-style
 *      in-order DFS (bot) once SWITCH or fewer rows remain ---- */
static int g_pos0[MD]; static uint8_t g_dist[MD][MD];
static uint64_t osolve(uint32_t alive, int n, const uint32_t *cand, uint32_t freecols) {
    if (n <= g_switch) {
        uint32_t c2[MD]; int p2[MD]; int m = 0;
        for (uint32_t a = alive; a; a &= a - 1u) { const int j = __builtin_ctz(a); c2[m] = cand[j]; p2[m] = g_pos0[j]; m++; }
        if (m == 1) { const int k = __builtin_popcount(c2[0]); g_gsteps += k; return (uint64_t)k; }
        return bot(0, m, c2, p2, 0u, 0u, 0u, c2[0]);
    }
    uint32_t once = 0, twice = 0; int best = 0, bp = 99;
    for (uint32_t a = alive; a; a &= a - 1u) { const int j = __builtin_ctz(a); const uint32_t c = cand[j];
        twice |= once & c; once |= c; const int pc = __builtin_popcount(c); if (pc < bp) { bp = pc; best = j; } }
    if ((freecols & ~once & ~1u) != 0u) return 0;
    uint32_t av = cand[best];
    const uint32_t uniq = once & ~twice & ~1u;
    if (uniq && bp > 1) {
        for (uint32_t a = alive; a; a &= a - 1u) { const int j = __builtin_ctz(a); const uint32_t u = cand[j] & uniq;
            if (u) { if (u & (u - 1u)) return 0; best = j; av = u; break; } }
    }
    const uint32_t rest = alive & ~(1u << best); const uint8_t *dk = g_dist[best];
    uint32_t nc[MD]; uint64_t sol = 0;
    while (av) {
        const uint32_t c = av & (0u - av); av ^= c; g_gsteps++;
        int ok = 1;
        for (uint32_t a = rest; a; a &= a - 1u) { const int j = __builtin_ctz(a); const int k = dk[j];
            const uint32_t x = cand[j] & ~(c | (c << k) | (c >> k));
            if (x == 0u) { ok = 0; break; }
            nc[j] = x; }
        if (ok) sol += osolve(rest, n - 1, nc, freecols & ~c);
    }
    return sol;
}
static uint64_t gsolve_root(uint32_t root_ld, uint32_t root_rd, uint32_t root_col, uint32_t root_a) {
    uint32_t cand[MD]; int pos[MD]; int n = 0, p = 0; uint32_t L = root_ld, R = root_rd;
    cand[n] = root_a; pos[n] = 0; n++;
    for (int e = 0; e <= g_term; e++) {
        p += stepof(e); advance(e, &L, &R);
        uint32_t c = g_bm & ~(L | R | root_col);
        if (e == g_term) { if (g_b14) c &= ~1u; } else { const uint32_t jm = (g_jm >> e) & 1u; c &= ~jm; L |= jm; }
        if (c == 0u) return 0;
        cand[n] = c; pos[n] = p; n++;
    }
    if (g_pol == 11) {   /* static order: fewest root candidates first (insertion sort, stable) */
        for (int i = 1; i < n; i++) { uint32_t c = cand[i]; int q = pos[i], j = i - 1; const int pc = __builtin_popcount(c);
            while (j >= 0 && __builtin_popcount(cand[j]) > pc) { cand[j + 1] = cand[j]; pos[j + 1] = pos[j]; j--; }
            cand[j + 1] = c; pos[j + 1] = q; }
    }
    if (g_mode == 15) {
        for (int i = 0; i < n; i++) { g_pos0[i] = pos[i]; for (int j = 0; j < n; j++) g_dist[i][j] = (uint8_t)(pos[i] > pos[j] ? pos[i] - pos[j] : pos[j] - pos[i]); }
        return osolve((n >= 32) ? 0xFFFFFFFFu : ((1u << n) - 1u), n, cand, g_bm & ~root_col);
    }
    return gsolve(n, cand, pos, g_bm & ~root_col);
}
static uint64_t g_recsteps;
static uint64_t solve_record(uint32_t root_ld, uint32_t root_rd, uint32_t root_col, uint32_t root_a_in,
                             uint32_t ctrl0, uint32_t markctrl, uint32_t w_lo, uint32_t bm, uint32_t n4) {
    const uint32_t jmark = markctrl & 31u, endm = (markctrl >> 5) & 31u;
    const uint32_t mark1 = (markctrl >> 10) & 31u, mark2 = (markctrl >> 15) & 31u;
    uint32_t root_a = root_a_in & bm;
    g_recsteps = 0;
    if (root_a == 0u) return 0;
    uint32_t raw = ctrl0; int sdepth = 0;
    memset(g_bc, 0, sizeof g_bc);
    uint32_t child_jmark_mask = 0, future_check_mask = 0, root_action = 0, term_b14 = 0; int term_parent = 0;
    for (;;) {
        if (sdepth > 31) abort();
        uint32_t fu = raw & 31u, rowv = (raw >> 5) & 31u;
        if (((IS_P5_MASK >> fu) & 1u) && rowv == mark1) fu = META_NEXT[fu];
        uint32_t action = 0, nibble = 0, fraw = 0, fcv = 0;
        if (((IS_BASE_MASK >> fu) & 1u) && rowv == endm) { action = (fu == 14u) ? 3u : 2u; }
        else {
            uint32_t ism = (IS_MARK_MASK >> fu) & 1u, code = 0, step = 1, usef = 1u - ism, nextf = fu;
            if (ism) { uint32_t mv = ((SEL2_MASK >> fu) & 1u) ? mark2 : mark1;
                if (rowv == mv) { code = ((BLOCK_CODE_B0_MASK >> fu) & 1u) | (((BLOCK_CODE_B1_MASK >> fu) & 1u) << 1) | (((BLOCK_CODE_B2_MASK >> fu) & 1u) << 2);
                    step = 2u + ((OP_STEP3_MASK >> code) & 1u); usef = 0; nextf = META_NEXT[fu]; } }
            if (((IS_JMARK_MASK >> fu) & 1u) && rowv == jmark) { action = 1u; nextf = META_NEXT[fu]; }
            uint32_t crow = rowv + step;
            if (usef && crow < endm) fcv = 1u;
            nibble = code | (fcv << 3); fraw = nextf | (crow << 5);
        }
        if (sdepth == 0) root_action = action;
        else { if (action == 1u) child_jmark_mask |= 1u << (sdepth - 1);
               else if (action >= 2u) { term_parent = sdepth - 1; term_b14 = (action == 3u); } }
        if (action >= 2u) break;
        if (fcv) future_check_mask |= 1u << sdepth;
        g_bc[sdepth] = (uint8_t)(nibble & 7u);
        raw = fraw; sdepth++;
    }
    if (root_action == 2u) return w_lo;
    if (root_action == 3u) return (((root_a & ~1u) != 0u) ? 1u : 0u) * (uint64_t)w_lo;
    if (root_action == 1u) { root_a &= ~1u; if (root_a == 0u) return 0; root_ld |= 1u; }
    g_jm = child_jmark_mask; g_fc = future_check_mask; g_term = term_parent; g_b14 = term_b14; g_bm = bm; g_n4 = n4;
    /* rows left (term+2) vs free columns */
    { int freec = __builtin_popcount(bm & ~root_col); if (freec != term_parent + 2) g_rowmismatch++; }
    if (g_mode >= 10) { g_pol = (g_mode == 20) ? g_toppol : g_mode; g_gsteps = 0; uint64_t so = gsolve_root(root_ld, root_rd, root_col, root_a); g_recsteps = g_gsteps; return so * (uint64_t)w_lo; }
    res_t r = dfs(0, root_ld, root_rd, root_col, root_a);
    g_recsteps = r.steps;
    return r.sol * (uint64_t)w_lo;
}

static uint32_t rd32(const unsigned char *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }

int main(int argc, char **argv) {
    if (argc != 7) { fprintf(stderr, "usage\n"); return 1; }
    int N = atoi(argv[1]); long first = atol(argv[3]), count = atol(argv[4]), stride = atol(argv[5]); g_mode = atoi(argv[6]); if (getenv("MAXLA")) g_maxla = atoi(getenv("MAXLA")); if (getenv("SWITCH")) g_switch = atoi(getenv("SWITCH")); if (getenv("TOPPOL")) g_toppol = atoi(getenv("TOPPOL"));
    FILE *f = fopen(argv[2], "rb"); if (!f) return 1;
    const uint32_t bm = (uint32_t)((1ULL << N) - 1), n4 = (uint32_t)(1ULL << (N - 4));
    uint64_t total = 0, steps = 0; unsigned char buf[28];
    for (long k = 0; k < count; k++) {
        long i = first + k * stride;
        if (fseek(f, i * 28, SEEK_SET) || fread(buf, 1, 28, f) != 28) break;
        uint32_t r[7]; for (int j = 0; j < 7; j++) r[j] = rd32(buf + 4 * j);
        uint64_t c = solve_record(r[0], r[1], r[2], r[4], r[3], r[5], r[6], bm, n4);
        total += c; steps += g_recsteps;
        printf("REC %ld %llu %llu\n", i, (unsigned long long)c, (unsigned long long)g_recsteps);
    }
    printf("TOTAL mode=%d sum=%llu steps=%llu rowmismatch=%llu\n", g_mode, (unsigned long long)total, (unsigned long long)steps, (unsigned long long)g_rowmismatch);
    printf("D depth steps imm fc la desc term nodes deadroots deadsteps deadnodes\n");
    for (int d = 0; d < MD; d++) if (st_steps[d] || st_deadroot[d])
        printf("D %d %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n", d,
               (unsigned long long)st_steps[d], (unsigned long long)st_imm[d], (unsigned long long)st_fc[d], (unsigned long long)st_la[d],
               (unsigned long long)st_desc[d], (unsigned long long)st_term[d], (unsigned long long)st_nodes[d],
               (unsigned long long)st_deadroot[d], (unsigned long long)st_deadsteps[d], (unsigned long long)st_deadnodes_any[d]);
    return 0;
}
