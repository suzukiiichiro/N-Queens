/*
 * 419_sched_scan.c -- host-only scan of a CRunner input (28 B/record): runs the schedule precompute
 * (the kernel's state machine, host logic) for every record and reports what the per-depth schedule
 * word has to hold. Written for the 417 post-mortem: at N = 25 every record's schedule reaches depth 16,
 * one past the 16 nibbles of the maxd16 word.
 *
 *   histogram of the terminal depth (= required_maxd per record)
 *   per depth d >= 16: how many records carry a non-zero block code at d (the class 417 computed wrong)
 *   --extract <out.bin>   write the records with a non-zero block code at any depth >= 16 (the affected
 *                         class) in input order, optionally every k-th of them (--every k)
 *   --sample <out.bin> <n> write n equal-spaced records (any class)
 *
 * Usage: 419_sched_scan <N> <in_soa7_bin> [--extract out.bin [--every k]] [--sample out.bin n]
 * Build: gcc -O2 -o 419_sched_scan 419_sched_scan.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static const uint32_t IS_BASE_MASK = 69222408u, IS_JMARK_MASK = 4u, IS_MARK_MASK = 199209203u, IS_P5_MASK = 3840u;
static const uint32_t SEL2_MASK = 34742338u, BLOCK_CODE_B0_MASK = 173707345u, BLOCK_CODE_B1_MASK = 12689458u;
static const uint32_t BLOCK_CODE_B2_MASK = 18088064u, OP_STEP3_MASK = 24u;
static const uint8_t META_NEXT[28] = {1,2,3,3,2,6,2,2,0,4,5,7,13,14,14,14,17,14,14,20,21,21,21,25,21,21,26,26};

/* returns the terminal depth (-1 on runaway); fills code[0..depth-1] */
static int schedule_of(uint32_t ctrl0, uint32_t markctrl, uint8_t *code) {
    const uint32_t jmark = markctrl & 31u, endm = (markctrl >> 5) & 31u, mark1 = (markctrl >> 10) & 31u, mark2 = (markctrl >> 15) & 31u;
    uint32_t raw = ctrl0; int d = 0;
    for (;;) {
        if (d > 31) return -1;
        uint32_t fu = raw & 31u, rowv = (raw >> 5) & 31u;
        if (((IS_P5_MASK >> fu) & 1u) && rowv == mark1) fu = META_NEXT[fu];
        if (((IS_BASE_MASK >> fu) & 1u) && rowv == endm) return d;
        uint32_t bc = 0, step = 1, nextf = fu;
        if ((IS_MARK_MASK >> fu) & 1u) {
            uint32_t mv = ((SEL2_MASK >> fu) & 1u) ? mark2 : mark1;
            if (rowv == mv) {
                bc = ((BLOCK_CODE_B0_MASK >> fu) & 1u) | (((BLOCK_CODE_B1_MASK >> fu) & 1u) << 1) | (((BLOCK_CODE_B2_MASK >> fu) & 1u) << 2);
                step = 2u + ((OP_STEP3_MASK >> bc) & 1u); nextf = META_NEXT[fu];
            }
        }
        if (((IS_JMARK_MASK >> fu) & 1u) && rowv == jmark) nextf = META_NEXT[fu];
        code[d] = (uint8_t)bc;
        raw = nextf | ((rowv + step) << 5); d++;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <N> <in_soa7_bin> [--extract out.bin [--every k]] [--sample out.bin n]\n", argv[0]); return 1; }
    const char *extract = NULL, *sample = NULL; long every = 1, nsample = 0;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--extract") && i + 1 < argc) extract = argv[++i];
        else if (!strcmp(argv[i], "--every") && i + 1 < argc) every = atol(argv[++i]);
        else if (!strcmp(argv[i], "--sample") && i + 2 < argc) { sample = argv[++i]; nsample = atol(argv[++i]); }
        else { fprintf(stderr, "bad arg %s\n", argv[i]); return 1; }
    }
    FILE *f = fopen(argv[2], "rb"); if (!f) { perror(argv[2]); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    if (sz % 28) { fprintf(stderr, "size %ld not a multiple of 28\n", sz); return 1; }
    long m = sz / 28;
    FILE *fx = extract ? fopen(extract, "wb") : NULL, *fs = sample ? fopen(sample, "wb") : NULL;
    long hist[33] = {0}, runaway = 0, affected = 0, written = 0, sampled = 0;
    long bc_at[32] = {0};
    long sample_step = (nsample > 0 && m > nsample) ? m / nsample : 1;
    unsigned char buf[28];
    for (long i = 0; i < m; i++) {
        if (fread(buf, 1, 28, f) != 28) { fprintf(stderr, "short read\n"); return 1; }
        uint32_t ctrl0 = buf[12] | buf[13] << 8 | buf[14] << 16 | (uint32_t)buf[15] << 24;
        uint32_t markctrl = buf[20] | buf[21] << 8 | buf[22] << 16 | (uint32_t)buf[23] << 24;
        uint8_t code[32]; memset(code, 0, sizeof code);
        int d = schedule_of(ctrl0, markctrl, code);
        if (d < 0) { runaway++; continue; }
        hist[d]++;
        int aff = 0;
        for (int k = 16; k < d; k++) if (code[k]) { bc_at[k]++; aff = 1; }
        if (aff) { affected++; if (fx && (affected - 1) % every == 0) { fwrite(buf, 1, 28, fx); written++; } }
        if (fs && sampled < nsample && i % sample_step == 0) { fwrite(buf, 1, 28, fs); sampled++; }
    }
    fclose(f); if (fx) fclose(fx); if (fs) fclose(fs);
    printf("[sched-scan] N=%s records=%ld runaway=%ld\n", argv[1], m, runaway);
    printf("[sched-scan-histogram]"); for (int d = 0; d <= 32; d++) if (hist[d]) printf(" %d:%ld", d, hist[d]); printf("\n");
    printf("[sched-scan-beyond-word]"); for (int k = 16; k < 32; k++) if (bc_at[k]) printf(" depth%d_nonzero_code=%ld", k, bc_at[k]); printf("\n");
    printf("[sched-scan-affected] records_with_nonzero_code_at_depth>=16=%ld (%.4f%%)%s%s%ld\n", affected, m ? 100.0 * affected / m : 0.0,
           extract ? " extracted=" : "", extract ? "" : "", extract ? written : 0L);
    if (sample) printf("[sched-scan-sample] %s records=%ld step=%ld\n", sample, sampled, sample_step);
    return 0;
}
