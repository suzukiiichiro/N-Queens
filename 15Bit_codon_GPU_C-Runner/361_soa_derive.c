/*
 * 361_soa_derive.c
 *
 * rev361 -- CUDA C runner port, first implementation step (Open
 * Objectives item 6). Scope is deliberately limited to a single
 * variable per project discipline: this file ports ONLY
 * build_soa_for_range()+symmetry() (the bin -> SoA derivation function,
 * 28-branch funcid decision tree) from 361Py_soa_derive_c_port.py to C.
 * It does NOT touch kernel_dfs_iter_gpu_maxd14, K-batching, the stack
 * layout, or any GPU code -- there is no CUDA in this file at all, only
 * host-side C, because build_soa_for_range() itself runs on the CPU in
 * the Codon source too (it is host-side prep before the SoA arrays are
 * uploaded to the GPU).
 *
 * Spec basis: 338_maxd14_port_spec.md (28-leaf funcid decision tree,
 * build_soa_for_range/symmetry() C-language mapping) as updated by
 * 360_maxd14_port_spec_update.md (SoA arrays = 7, not 8: w_hi_arr is
 * gone since 351; save_sp/stack_ptr width choices are irrelevant to
 * THIS file, since save_sp/stack_ptr belong to the kernel, not to this
 * derivation function).
 *
 * Bin format (validated end-to-end in rev337): header-less, fixed
 * 16-byte little-endian records: ld(u32) rd(u32) col(u32) startijkl(u32).
 * This program reads the SAME raw stream bin 361Py resolves via
 * ensure_constellations_bin_stream() -- i.e. the pre-reorder cache file
 * -- so that record order on both sides is identical by construction
 * (both simply iterate the file in on-disk order; the
 * broadmarktail/chunkshape148/funcid_reorder pipeline is out of scope
 * for this revision).
 *
 * Output: for every record, in file order, 10 little-endian uint32_t
 * fields are appended to the output file, in this exact order:
 *   ld_arr[t]  rd_arr[t]  col_arr[t]  row_arr[t]  ctrl0_arr[t]
 *   free_arr[t]  markctrl_arr[t]  funcid_arr[t]  ijkl_arr[t]  w_lo[t]
 * This must byte-for-byte match the file 361Py's dump_soa_reference_c_port()
 * (bench_mode==32) produces for the same input bin.
 *
 * Build (host-only; no nvcc-specific flags needed, but nvcc's host
 * compiler works too since this is plain C):
 *   gcc -O2 -Wall -Wextra -o 361_soa_derive 361_soa_derive.c
 *
 * Usage:
 *   ./361_soa_derive <N> <in_bin_path> <out_bin_path>
 */

#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

typedef struct {
    int64_t ld;
    int64_t rd;
    int64_t col;
    int64_t startijkl;
} RawRecord;

typedef struct {
    uint32_t ld;
    uint32_t rd;
    uint32_t col;
    uint32_t row;
    uint32_t ctrl0;
    uint32_t free;
    uint32_t markctrl;
    uint32_t funcid;
    uint32_t ijkl;
    uint32_t w_lo;
} SoaOut;

/* ---------------------------------------------------------------------
 * geti/getj/getk/getl -- literal port of the Codon helpers (line 2812-
 * 2815 of 361Py): 5-bit fields packed into a 20-bit ijkl word.
 * ------------------------------------------------------------------- */
static inline int64_t geti(int64_t ijkl) { return (ijkl >> 15) & 0x1F; }
static inline int64_t getj(int64_t ijkl) { return (ijkl >> 10) & 0x1F; }
static inline int64_t getk(int64_t ijkl) { return (ijkl >> 5)  & 0x1F; }
static inline int64_t getl(int64_t ijkl) { return  ijkl        & 0x1F; }

/* symmetry90 -- literal port (line 2819 of 361Py). */
static inline int symmetry90(int64_t ijkl, int64_t N) {
    int64_t i = geti(ijkl), j = getj(ijkl), k = getk(ijkl), l = getl(ijkl);
    int64_t lhs = (i << 15) + (j << 10) + (k << 5) + l;
    int64_t rhs = (((N - 1 - k) << 15) + ((N - 1 - l) << 10) + (j << 5) + i);
    return lhs == rhs;
}

/* symmetry -- literal port (line 2818 of 361Py). Always returns 2, 4 or
 * 8, so a uint32_t (the "w_lo" field) is sufficient with no truncation
 * risk -- unlike ld/rd/col below, there is no wraparound behavior to
 * preserve here. */
static inline uint32_t symmetry(int64_t ijkl, int64_t N) {
    if (symmetry90(ijkl, N)) return 2u;
    int64_t i = geti(ijkl), j = getj(ijkl), k = getk(ijkl), l = getl(ijkl);
    if (i == N - 1 - j && k == N - 1 - l) return 4u;
    return 8u;
}

/*
 * build_soa_for_range_one -- literal port of the per-record body of
 * build_soa_for_range() (361Py lines ~2081-2258, inside the `for t in
 * range(m):` loop). Variable names (jmark, mark1, mark2, endmark,
 * target, j_lt_N3, j_eq_N3, j_eq_N2, k_lt_l, start_lt_k, start_lt_l,
 * l_eq_kp1, k_eq_lp1, j_gate) are kept identical to the Python source
 * so the 28-branch decision tree can be diffed side-by-side by eye.
 *
 * All intermediate arithmetic uses int64_t to mirror Codon's `int`
 * (64-bit signed) -- this matters because some of the shift
 * expressions below (e.g. the rd|= line using two chained shifts) can
 * carry bits above bit 31 before the final uint32_t truncation at the
 * bottom of this function; that truncation-after-64-bit-arithmetic
 * behavior must be preserved bit-for-bit, not approximated with 32-bit
 * intermediates.
 *
 * board_mask/small_mask are passed in (computed once per N by the
 * caller, not per record) to match build_soa_for_range()'s hoisting of
 * these two out of its per-record loop.
 */
static void build_soa_for_range_one(
    int64_t N, int64_t N1, int64_t N2,
    int64_t board_mask, int64_t small_mask,
    const RawRecord *rec,
    SoaOut *out
) {
    int64_t jmark = 0, mark1 = 0, mark2 = 0;
    int64_t endmark = 0, target = 0;

    int64_t start_ijkl = rec->startijkl;
    int64_t start = start_ijkl >> 20;
    int64_t ijkl  = start_ijkl & ((1LL << 20) - 1);

    int64_t j = getj(ijkl), k = getk(ijkl), l = getl(ijkl);

    int64_t ld  = rec->ld >> 1;
    int64_t rd  = rec->rd >> 1;
    int64_t col = (rec->col >> 1) | (~small_mask);
    col &= board_mask;

    int64_t LD = (1LL << (N1 - j)) | (1LL << (N1 - l));
    assert(N - start >= 0);
    ld |= LD >> (N - start);

    if (start > k) {
        int64_t shamt = N1 - (start - k + 1);
        assert(shamt >= 0);
        rd |= (1LL << shamt);
    }
    if (j >= 2 * N - 33 - start) {
        assert(N1 - j >= 0);
        assert(N2 - start >= 0);
        rd |= (1LL << (N1 - j)) << (N2 - start);
    }

    int64_t free = board_mask & ~(ld | rd | col);

    int j_lt_N3 = (j < N - 3);
    int j_eq_N3 = (j == N - 3);
    int j_eq_N2 = (j == N - 2);

    int k_lt_l     = (k < l);
    int start_lt_k = (start < k);
    int start_lt_l = (start < l);

    int l_eq_kp1 = (l == k + 1);
    int k_eq_lp1 = (k == l + 1);

    int j_gate = (j > 2 * N - 34 - start);

    if (j_lt_N3) {
        jmark = j + 1;
        endmark = N2;

        if (j_gate) {
            if (k_lt_l) {
                mark1 = k - 1; mark2 = l - 1;
                if (start_lt_l) {
                    if (start_lt_k) {
                        target = (!l_eq_kp1) ? 0 : 4;
                    } else {
                        target = 1;
                    }
                } else {
                    target = 2;
                }
            } else {
                mark1 = l - 1; mark2 = k - 1;
                if (start_lt_k) {
                    if (start_lt_l) {
                        target = (!k_eq_lp1) ? 5 : 7;
                    } else {
                        target = 6;
                    }
                } else {
                    target = 2;
                }
            }
        } else {
            if (k_lt_l) {
                mark1 = k - 1; mark2 = l - 1;
                target = (!l_eq_kp1) ? 8 : 9;
            } else {
                mark1 = l - 1; mark2 = k - 1;
                target = (!k_eq_lp1) ? 10 : 11;
            }
        }
    } else if (j_eq_N3) {
        endmark = N2;

        if (k_lt_l) {
            mark1 = k - 1; mark2 = l - 1;
            if (start_lt_l) {
                if (start_lt_k) {
                    target = (!l_eq_kp1) ? 12 : 15;
                } else {
                    mark2 = l - 1;
                    target = 13;
                }
            } else {
                target = 14;
            }
        } else {
            mark1 = l - 1; mark2 = k - 1;
            if (start_lt_k) {
                if (start_lt_l) {
                    target = (!k_eq_lp1) ? 16 : 18;
                } else {
                    mark2 = k - 1;
                    target = 17;
                }
            } else {
                target = 14;
            }
        }
    } else if (j_eq_N2) {
        if (k_lt_l) {
            endmark = N2;
            if (start_lt_l) {
                if (start_lt_k) {
                    mark1 = k - 1;
                    if (!l_eq_kp1) {
                        mark2 = l - 1;
                        target = 19;
                    } else {
                        target = 22;
                    }
                } else {
                    mark2 = l - 1;
                    target = 20;
                }
            } else {
                target = 21;
            }
        } else {
            if (start_lt_k) {
                if (start_lt_l) {
                    if (k < N2) {
                        mark1 = l - 1; endmark = N2;
                        if (!k_eq_lp1) {
                            mark2 = k - 1;
                            target = 23;
                        } else {
                            target = 24;
                        }
                    } else {
                        if (l != (N - 3)) {
                            mark2 = l - 1; endmark = N - 3;
                            target = 20;
                        } else {
                            endmark = N - 4;
                            target = 21;
                        }
                    }
                } else {
                    if (k != N2) {
                        mark2 = k - 1; endmark = N2;
                        target = 25;
                    } else {
                        endmark = N - 3;
                        target = 21;
                    }
                }
            } else {
                endmark = N2;
                target = 21;
            }
        }
    } else {
        endmark = N2;
        if (start > k) {
            target = 26;
        } else {
            mark1 = k - 1;
            target = 27;
        }
    }

    /* Final field assembly -- literal port of lines 2241-2258.
     * u32(...) casts in Codon truncate a 64-bit int to its low 32
     * bits; (uint32_t)(int64_t) does the same in C. */
    out->ld  = (uint32_t)ld;
    out->rd  = (uint32_t)rd;
    out->col = (uint32_t)col;
    out->row = (uint32_t)start;
    out->ctrl0 = (uint32_t)target | ((uint32_t)start << 5);
    out->free  = (uint32_t)free;
    out->markctrl =
          ((uint32_t)jmark & 31u)
        | (((uint32_t)endmark & 31u) << 5)
        | (((uint32_t)mark1  & 31u) << 10)
        | (((uint32_t)mark2  & 31u) << 15);
    out->funcid = (uint32_t)target;
    out->ijkl   = (uint32_t)ijkl;
    out->w_lo   = symmetry(ijkl, N);
}

static uint32_t read_u32_le(const unsigned char *p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static void write_u32_le(FILE *f, uint32_t v) {
    unsigned char b[4];
    b[0] = (unsigned char)(v & 0xFF);
    b[1] = (unsigned char)((v >> 8) & 0xFF);
    b[2] = (unsigned char)((v >> 16) & 0xFF);
    b[3] = (unsigned char)((v >> 24) & 0xFF);
    if (fwrite(b, 1, 4, f) != 4) {
        fprintf(stderr, "ERROR: short write\n");
        exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <N> <in_bin_path> <out_bin_path>\n", argv[0]);
        return 1;
    }
    int64_t N = atoll(argv[1]);
    const char *in_path = argv[2];
    const char *out_path = argv[3];

    FILE *fin = fopen(in_path, "rb");
    if (!fin) {
        fprintf(stderr, "ERROR: cannot open input bin '%s'\n", in_path);
        return 1;
    }
    if (fseeko(fin, 0, SEEK_END) != 0) { fprintf(stderr, "ERROR: fseek failed\n"); return 1; }
    off_t fsize = ftello(fin);
    if (fsize < 0 || fsize % 16 != 0) {
        fprintf(stderr, "ERROR: input bin size %lld is not a multiple of 16 (header-less 16-byte record format expected)\n", (long long)fsize);
        return 1;
    }
    rewind(fin);
    int64_t total_records = (int64_t)(fsize / 16);

    FILE *fout = fopen(out_path, "wb");
    if (!fout) {
        fprintf(stderr, "ERROR: cannot open output bin '%s' for writing\n", out_path);
        return 1;
    }

    int64_t N1 = N - 1;
    int64_t N2 = N - 2;
    int64_t board_mask = (1LL << N) - 1;
    int64_t small_mask = (1LL << (N - 2 > 0 ? N - 2 : 0)) - 1;

    unsigned char buf[16];
    uint64_t checksum = 0;
    int64_t t = 0;
    while (t < total_records) {
        size_t got = fread(buf, 1, 16, fin);
        if (got != 16) {
            fprintf(stderr, "ERROR: short read at record %lld (got %zu bytes)\n", (long long)t, got);
            return 1;
        }
        RawRecord rec;
        rec.ld        = (int64_t)read_u32_le(buf + 0);
        rec.rd        = (int64_t)read_u32_le(buf + 4);
        rec.col       = (int64_t)read_u32_le(buf + 8);
        rec.startijkl = (int64_t)read_u32_le(buf + 12);

        SoaOut o;
        build_soa_for_range_one(N, N1, N2, board_mask, small_mask, &rec, &o);

        uint32_t fields[10] = {
            o.ld, o.rd, o.col, o.row, o.ctrl0,
            o.free, o.markctrl, o.funcid, o.ijkl, o.w_lo
        };
        for (int fidx = 0; fidx < 10; fidx++) {
            checksum += (uint64_t)fields[fidx];
            write_u32_le(fout, fields[fidx]);
        }

        if (t < 5) {
            fprintf(stderr,
                "[soa-ref-dump][sample] t=%lld ld=%u rd=%u col=%u row=%u ctrl0=%u free=%u markctrl=%u funcid=%u ijkl=%u w=%u\n",
                (long long)t, o.ld, o.rd, o.col, o.row, o.ctrl0, o.free, o.markctrl, o.funcid, o.ijkl, o.w_lo);
        }
        t++;
    }

    fclose(fin);
    fclose(fout);

    printf("[soa-ref-dump-done] N=%lld src_bin=%s records=%lld out=%s checksum_u64=%llu\n",
           (long long)N, in_path, (long long)total_records, out_path, (unsigned long long)checksum);

    return 0;
}
