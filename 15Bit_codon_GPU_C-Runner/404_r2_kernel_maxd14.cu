/*
 * 363_kernel_maxd14.cu
 *
 * rev404-r2 -- host-side only: NQ_LAYOUT=auto now selects the 403 layout at
 *   every N (it was 402 for N <= 21, 403 for N = 22). One executable line in
 *   select_pack_layout(); process_one_task() and both kernel instantiations
 *   are untouched, so <402> and <403> keep 404's SASS. NQ_LAYOUT=402 still
 *   runs the 402 layout for N <= 21.
 *
 * WHY: 404 RESULT (2026-10-02 10:13) -- ADOPTED (A+B). All hard gates passed
 *   (S2 <402> sha dd619aaf / 463 / 130, S3 <403> sha 6a17e634 / 465 / 131,
 *   diagnostic shas, CPU equality x3, L1/L3/L5 results.bin == L0, Z4 == Z7,
 *   X4 == 403-r7's X7, 9 runs oracle + 1710 MHz, 160 B / 38 registers).
 *     L0 403_r7  N=21 l402  109,458.1
 *     L1 A                  108,911.0   L1/L0 0.9950  (-0.50%)
 *     L3 404 = A+B          108,960.4   L3/L1 1.0005  (+0.05%: B gains nothing)
 *     L5 A+B+C (diag)       107,959.3   L5/L3 0.9908  (-0.92%)
 *     Z7 403_r7  N=21 l403  110,298.5
 *     Z4 404     N=21 l403  108,123.2   Z4/Z7 0.9803  (-1.97%)
 *     X4 404     N=22       946,599.4   vs r7 X7 0.9770 (-2.31%)
 *     G21 108,961.8 (-0.43% vs 109,437)   G22 945,753.1 (-2.32% vs 968,206)
 *   PB was refuted to the letter (L3/L1 >= 1.000) while the adoption rule
 *   held; the registration had not foreseen B depending on the layout.
 *   Suzuki decided: adopt 404. Production = 404: N=21 108,962 ms, N=22
 *   945,753 ms (15:46).
 *   Reading: B (popc depth) is worth nothing under 402 and the full ~-1.5%
 *   under 403. After a pop, 402 delivers col at 2 ops (so depth at 4), 403
 *   delivers col at 1 op (depth at 3). Hypothesis, supported by Z4 < L3:
 *   with 404 the 403 layout is FASTER than 402 at N=21 (108,123 vs 108,960,
 *   -0.77%); through 403-r7 it was 0.8% slower.
 *   C (future check from the depth bit) gave -0.92% under 402 although the
 *   search showed one instruction; my call to leave it out was wrong. It is
 *   measured again here under both layouts (diagnostic) and is the next
 *   candidate.
 *
 * 404-r2 GATES / CELLS (harness 404_r2_validate.sh)
 *   S0 static: code region differs from 404 by that one line (1/1); gcc.
 *   S1 ptxas.  S2/S3 HARD: <402> sha dd619aaf, <403> sha 6a17e634 (== 404).
 *   SD diagnostic d5 = generator(C) on 404: <402> ab5608e9, <403> 7c67c3aa.
 *   Y2 HARD: CPU per-record equality vs 404 (N=22 auto, N=21 l403, N=21 auto
 *      -- auto is 402 in 404 and 403 here; the results must not move).  Y3.
 *   N=21 direct, same session (2 min each):
 *     A0 404     auto (402)        anchor 404 L3 108,960
 *     A1 404_r2  auto (403)        results.bin == A0  HARD
 *     A2 404_r2  NQ_LAYOUT=402     results.bin == A0  HARD
 *     C2 d5      NQ_LAYOUT=402     diagnostic (404 L5 107,959)
 *     C3 d5      NQ_LAYOUT=403     diagnostic (new)
 *   G21 ./404_r2Py -g 21 21;  G22 only with RUN_G22=1 (same SASS as 404).
 *   Q1 STATED: A1/A0 <= 0.994 (404: Z4/L3 = 0.9923). Refuted >= 0.9997.
 *   Q2 STATED: |A2 - A0| <= 0.05% (same SASS).
 *   Q3 HARD: G21 oracle, layout=403.  STATED: |G21 - A1| <= 0.15%.
 *   QC: C2/A0 and C3/A1, information only.  Q4: 1710 MHz.
 *   ADOPT (S0-S3, Y2, Y3, identities, A1/A0 <= 0.998, Q3): 404_r2 is
 *   production, N=21 = G21, N=22 unchanged (945,753; same SASS).
 *
 * rev404 -- two bookkeeping removals in the DFS loop (both layouts):
 *   A  save_sp is dropped. It was always equal to stack_ptr (both start at 0
 *      and move together at the root push, the main push and the pop); the
 *      duplicate dates from 402, which added stack_ptr for the two-array
 *      frame and kept save_sp. The pop's empty test is now stack_ptr == 0.
 *   B  the 4-bit depth LIFO (stack_depth) is dropped. Every step adds exactly
 *      one bit to col (bit comes out of avail = bm & ~(ld|rd|col)), so
 *      cur_depth == popc(cur_col) - popc(root_col) at all times; the pop
 *      recomputes it (POPC + subtract) instead of popping it.
 *   Code-region diff vs 403_r7 (comment-stripped): removed=11 added=4.
 *   Nothing else changes: packing, push guard, schedule decode, future check.
 *
 * WHY (static read of the <402> loop + 404 compile search, 2026-10-02, GPU 0)
 *   <402> hot loop 140: push 16 = pack 4, depth LIFO 3, save_sp 1, stack_ptr
 *   1, addresses 2, stores 2, MOV 2, BRA 1; pop 17 = unpack 5, depth LIFO 3,
 *   save_sp 1, stack_ptr 1, addresses 2, loads 2, branch 3. Less than half
 *   of the push/pop is the board itself.
 *   Search (real instructions / loop / push / pop / rest of loop, <402>):
 *     d0 as is      482 / 140 / 16 / 17 / 107   (<402> == 402_r5: control OK)
 *     d1 A          479 / 138 / 15 / 16 / 107   only the two IADD3 are gone
 *     d2 B          473 / 132 / 13 / 16 / 103
 *     d3 A+B        463 / 130 / 12 / 15 / 103   sha dd619aaf  <- this file
 *     d4 C          479 / 137 / 16 / 17 / 104   (C = future check from the
 *     d5 A+B+C      462 / 129 / 12 / 15 / 102    depth bit; rev242's form)
 *   <403>: d0 483 / 141, d3 465 / 131 (sha 6a17e634). Frame 160 B in all;
 *   registers 40 -> 38 in d3. CPU per-record equality vs d0 held for every
 *   variant (N=21 auto, N=21 layout 403, N=22 auto, 1,024 real records each).
 *   B also removed 4 instructions OUTSIDE push/pop: the two depth save MOVs
 *   at the top of a step and the re-materialised nibble extraction before
 *   the future check (SHL + SHF.R.U64, the 395a leftover). With that gone C
 *   adds one instruction only, against rev242's -1.3% precedent: C is NOT
 *   in this file; the harness measures it as a diagnostic cell (L5).
 *   COST TO WATCH: after a pop, cur_depth now sits 4 dependent ops behind
 *   the loads (col 2 -> POPC -> subtract) instead of 1. 403-r6/r7 put one
 *   chain level at ~0.3% and one push/pop instruction at ~0.4%.
 *   <402> no longer equals 402_r5's SASS. N=21 is guarded from here on by
 *   per-thread byte identity against 403_r7 and the oracle.
 *
 * 403-r7 RESULT (2026-10-01) -- ADOPTED. S2 <402> == 402_r5, S3 <403> sha
 *   db7e0ec2 / real 483 / loop 141, X7 results.bin == r6's X6, oracle and
 *   1710 MHz on all 4 runs. Z7/Z6 = 0.9922 (110,339.0 vs 111,203.2), X7 vs
 *   r6 X6 = 0.9917 (968,931.6 vs 977,058.5), G22 968,205.9. Production =
 *   403_r7: N=21 109,4xx unchanged, N=22 968,206 ms (16:08). Two push
 *   instructions for -0.78..-0.83% = ~0.4% each, the same as r5; r6's 0.71%
 *   reads as ~0.4% (count) + ~0.3% (one level off the ld chain).
 *
 * 404 GATES / CELLS (harness 404_validate.sh)
 *   S0 static: code region == generator(A+B) applied to 403_r7; save_sp and
 *      stack_depth gone; 402/403 pack statements identical to 403_r7; gcc.
 *   S1 ptxas 160 B / 0 spill / regs <= 44.
 *   S2 HARD: <402> sha dd619aaf, real 463, loop 130 (the searched d3).
 *   S3 HARD: <403> sha 6a17e634, real 465, loop 131.
 *   Y2 HARD: CPU per-record equality vs 403_r7 (N=22 auto, N=21 l403, N=21
 *      auto).  Y3 rc=3 cases.
 *   N=21 ladder, direct, layout 402, same session (2 min each):
 *     L0 403_r7 | L1 d1 (A) | L3 404 (A+B) | L5 d5 (A+B+C, diagnostic)
 *     L1/L3/L5 results.bin == L0 (HARD). d1 and d5 are built by the harness
 *     from 403_r7 with the search's generator and checked by SASS sha.
 *   Z7 / Z4  N=21 NQ_LAYOUT=403, 403_r7 vs 404; Z4 results.bin == Z7 (HARD)
 *   X4       N=22 direct, 404; results.bin == 403-r7's X7 (HARD)
 *   G21, G22 ./404Py -g (production path)
 *   PA STATED: L1/L0 <= 0.994 (2 push/pop instructions at ~0.4%).
 *   PB STATED: L3/L1 <= 0.980. <= 0.998: real, chain cost visible.
 *      >= 1.000: B loses -> 404-r2 ships d1 only.
 *   PX STATED: X4 vs r7 X7 <= 0.985.  PC: L5/L3, information only.
 *   P3 HARD: G21 oracle layout=402, G22 oracle layout=403.  P4: 1710 MHz.
 *   ADOPT (S0-S3, Y2, Y3, identities, L3/L0 <= 0.998, X4/X7 <= 0.998, P3):
 *   404 is production for N <= 22; N=21 = G21, N=22 = G22.
 *
 * rev403-r7 -- push side of the 403b layout: 7 packing instructions -> 5.
 *   Code-region diff vs 403_r6 (comment-stripped): removed=6 added=18, the
 *   three push lines of the 403 branch at both push sites (main and root):
 *     A: ((uint64_t)(cur_ld >> 1) << 44)  ->  ((uint64_t)((cur_ld << 11) &
 *        0xFFFFF000u) << 32)   (same value; SHF+SHL+OR -> SHL + masked-OR)
 *     B: (cur_rd & ~1u) | (cur_ld & 1u)   ->  inline PTX lop3.b32 0xD8
 *        (#ifdef __CUDACC__; #else the C spelling for the gcc CPU harness)
 *   Pop and the 402 branches are untouched.
 *
 * WHY (static read of the 403-r6 SASS, 2026-10-01, GPU time zero)
 *   CORRECTION: "496 instructions in both <402> and <403>" counted the tail
 *   alignment NOPs (the function is padded to 128 B). Real instructions /
 *   hot loop: <402> 482 / 140; original 403 489 / 145; 403_r5 488 / 144;
 *   403_r6 487 / 143. So r5 removed ONE loop instruction (not eight) for
 *   -0.40%, r6 one more for -0.71%, and 3 remain for +1.6% at N=21: one
 *   loop instruction on the push/pop path ~ 0.4-0.7%.
 *   Register-normalised, the loop bodies of <402> and <403> r6 agree section
 *   by section (53/53, 40/40, solution count 10/10, pop 17/17; only
 *   reorderings). The whole +3 is the push (16 vs 19): B takes 2 LOP3
 *   (402: 0) and (ld >> 1) << 44 takes SHF + SHL (402: the mask fuses).
 *   "The rest is the schedule" (403-r5 note below) was an artifact of the
 *   padded count. "Chain depth matters" (403-r6 reading) is back on hold:
 *   r6 shortened the chain AND removed an instruction. The push feeds no
 *   loop-carried value, so r7 changes the instruction count only.
 *
 * 403-r7 COMPILE SEARCH (2026-10-01 16:33, GPU time zero; six variants)
 *   w0 as is              487 / 143 / push 19  sha a8c1fffa (control OK)
 *   w1 B asm              485 / 142 / 18
 *   w2 A C 32-bit         485 / 142 / 18
 *   w3 w1 + w2            483 / 141 / 17       sha db7e0ec2  <- this file
 *   w4 w1 + A asm 0xF8    483 / 142 / 18       (ptxas split the WIDE)
 *   w5 w1 + A C 64-bit    483 / 141 / 17       sha db7e0ec2 (== w3)
 *   All: <402> identical to 402_r5, frame 160, 40 registers. w3 vs w0
 *   differs only in the push (and a reordering inside the mark block).
 *   <403> is now <402> + 1 (the one select for B, the layout's floor).
 *
 * 403-r6 RESULT (2026-09-30) -- ADOPTED. All hard gates passed (S2 <402>
 *   sha 76aabe60, S3 <403> sha a8c1fffa, CPU equality x3, X6 results.bin ==
 *   X5, 5 runs oracle + 1710 MHz). Z6/Z5 = 0.9929 (111,208.1 vs 112,001.0),
 *   X6/X5 = 0.9932 (977,058.5 vs 983,771.6), G22 976,391.2. Production =
 *   403_r6 (N=21 109,4xx unchanged, N=22 976,391 ms = 16:16).
 *
 * 403-r7 GATES / CELLS (harness 403_r7_validate.sh)
 *   S0 static: r7 statements present (2 sites), 402 statements == 403_r6,
 *      fingerprint removed=6 added=18, kernel sha, gcc builds.
 *   S1 ptxas 160 B / 0 spill / regs <= 44.  S2 HARD: <402> == 402_r5.
 *   S3 HARD: <403> SASS sha == db7e0ec2 (the searched w3), real 483, loop 141.
 *   Y2 HARD: CPU per-record equality vs 403_r6 (N=22 auto, N=21 l403, N=21
 *      auto). The CPU side runs the new C spelling of A; B's asm is checked
 *      by the GPU side (X7 byte identity + oracle).  Y3 rc=3 cases.
 *   Z6 / Z7  N=21 NQ_LAYOUT=403 direct, 403_r6 vs 403_r7 (same session)
 *   X6 / X7  N=22 direct, 403_r6 vs 403_r7; X7 results.bin == X6 (HARD)
 *   G22      ./403_r7Py -g 22 22 (production path)
 *   P1 STATED: X7/X6 <= 0.992 (2 loop instructions at 0.4-0.7% each).
 *      <= 0.998: real but smaller than the per-instruction rate (then part of
 *      r6's gain was chain depth). Refuted: >= 0.9997. Between: grey.
 *   P2 STATED: Z7/Z6 <= 0.992; same bands.
 *   P3 HARD: G22 oracle, layout=403.  P4: 1710 MHz.
 *   ADOPT (S0-S3, Y2, Y3, X7 identity, X7/X6 <= 0.998, P3): 403_r7 is
 *   production, N=22 = G22, N=21 unchanged. Grey: Suzuki decides.
 *
 * rev403-r6 -- pop-side ld of the 403b layout pinned to ONE LOP3 (inline PTX).
 *   Code-region diff vs 403_r5: the one pop line becomes a 9-line block
 *   (a43 temp; #ifdef __CUDACC__ asm lop3.b32 0xD8 / #else the C spelling
 *   for the gcc CPU harness). Nothing else changes; 402 branches untouched.
 *
 * WHY (403-r6 compile search, 2026-09-30, GPU time zero): 403_r5's <403>
 *   pop reconstructs ld as SHF.R.U32.HI (hi >> 11) -> LOP3 & 0x1ffffe ->
 *   LOP3 | (B & 1): 3 dependent ops, one more than cur_avail (2), so ld
 *   became the longest chain out of the pop (402: 1 op). Six C spellings of
 *   the same expression (& ~1u, ^-select, explicit 0xfffffffe, shift-back,
 *   64-bit select, mask variable) all compiled to byte-identical SASS
 *   (496, sha fd6f41b2): the front end canonicalizes them and ptxas never
 *   fuses the two LOP3s. The inline-PTX variant compiled to a single
 *   LOP3.LUT ... 0xb8 (496 total with one more alignment NOP; sha a8c1fffa),
 *   everything else in the kernel identical modulo branch addresses.
 *   Result: pop = 5 ops (col 1, avail 2, ld 2, rd 0) -- the 402 shape.
 *
 * 403-r5 RESULT (2026-09-30) -- ADOPTED via 403-r5b. All hard gates passed
 *   (<402> == 402_r5; <403> 504 -> 496; CPU equality x3; X1 results.bin ==
 *   X0). Z5/Z3 = 0.9959, X1/X0 = 0.9956, G22 983,142 / G22r 983,164
 *   (+0.0023%). Production N=22 = 983,153 ms (16:23), N=21 unchanged. The
 *   instruction count explains only ~15% of the 403 penalty at N=21
 *   (112,001 vs 109,4xx with equal counts); the rest is schedule/chains.
 *
 * 403-r6 GATES / CELLS (harness 403_r6_validate.sh)
 *   S0 static: pop block present, 402 statements == 403_r5, fingerprint
 *      removed=1 added=7 vs 403_r5 code region; gcc builds.
 *   S1 ptxas 160 B / 0 spill / regs <= 44.  S2 HARD: <402> == 402_r5.
 *   S3 HARD: <403> SASS sha == a8c1fffa (the searched v5 binary), 496.
 *   Y2 HARD: CPU per-record equality vs 403_r5 (N=22 auto, N=21 l403,
 *      N=21 auto).  Y3 rc=3 cases.
 *   Z5 / Z6  N=21 NQ_LAYOUT=403 direct, 403_r5 vs 403_r6 (same session)
 *   X5 / X6  N=22 direct, 403_r5 vs 403_r6; X6 results.bin == X5 (HARD)
 *   G22      ./403_r6Py -g 22 22 (production path)
 *   P1 STATED: X6/X5 <= 0.998 (one op off the longest pop chain).
 *      Refuted: >= 0.9997 (|diff| within 10x the N=22 floor). Between: grey.
 *   P2 STATED: Z6/Z5 <= 0.998; refuted >= 0.9997.
 *   P3 HARD: G22 oracle, layout=403.  P4: 1710 MHz.
 *   ADOPT (S0-S3, Y2, Y3, X6 identity, P1 held, P3): 403_r6 is production,
 *   N=22 = G22, N=21 unchanged. Grey: Suzuki decides (results identical,
 *   code strictly shorter, so a small real gain is still worth taking).
 *
 * rev403-r5 -- the 403 layout re-spelled ("403b"): split ld at the BOTTOM.
 *   Code-region diff vs 403_r3 (comment-stripped): removed=7 added=6, all in
 *   the three 403 branches plus the retired PACK403_LDLO define. The 402
 *   branches are byte-identical to 403_r3 / 402_r5, so the <402>
 *   instantiation must keep 402_r5's SASS (S2, hard).
 *
 * WHY (static SASS reading of 403_r4's binary, 2026-09-30, GPU time zero)
 *   <402> 496 vs <403> 504 instructions. The 8 extra are NOT a longer
 *   dependency chain: after the two LDLs both layouts reach the four loop-
 *   carried values within 2 dependent ops (402: ld 1, col 2, avail 2, rd 0;
 *   403: col 1, avail 2, ld 2, rd 1), and the critical path is cur_avail ->
 *   bit in both. The 8 are issue slots: pop +2 (IMAD.SHL, LOP3), main push
 *   +3 (SHF.R, LOP3 &1, LOP3 merge), root push +2, NOP +1. Five of the
 *   seven ALU ops come from the ld[0..19] | ld[20] split at the TOP of ld,
 *   two from rd & ~1.
 *
 * WHAT CHANGES (403 branches only; arithmetic on the live bits unchanged)
 *   push: A = col | avail<<22 | (ld>>1)<<44 ; B = (rd & ~1) | (ld & 1)
 *         (ld>>1)<<44 keeps ld[1..20]; ld's dead bit 21 (and any junk above)
 *         falls off the top of the 64-bit word; B is one LOP3 bit-select.
 *   pop:  col = A & 0x3fffff ; avail = (A>>22) & bm ;
 *         ld = ((A>>43) & ~1) | (B & 1) ; rd = B  (UNMASKED: bit 0 = ld[0],
 *         dead for rd -- the only uses of the saved rd are (rd|bit)>>step,
 *         step >= 1, verified over every use in process_one_task; the value
 *         printed as rd by the CPU harness's debug lines may differ in bit 0,
 *         accepted by Suzuki 2026-09-30).
 *   Expected <403> SASS: pop 5 ops (= 402's 5; rd back to depth 0),
 *   push 5 (402: 4), about 498 instructions in total. Round-trip test on
 *   5e7 random (col, avail, ld with junk bits 22..24, rd): col, avail, ld
 *   bits 0..20 and rd bits 1..31 all preserved, ld bit 21 read back as 0.
 *
 * 403-r5 GATES / CELLS (harness 403_r5_validate.sh)
 *   S0 static: 402 branch statements byte-identical to 403_r3; 403 branch
 *      statements as above; fingerprint removed=7 added=6; gcc; notes.
 *   S1 ptxas: both instantiations 160 B / 0 spill / regs <= 44.
 *   S2 HARD: <402> SASS identical to 402_r5 (N=21 production untouched).
 *   S3 STATED: <403> instruction count < 504 (info: exact count).
 *   Y2 HARD: CPU per-record equality on real sched records vs 403_r3:
 *      N=22 auto, N=21 NQ_LAYOUT=403, N=21 auto.  Y3: rc=3 cases.
 *   X0 403_r3 N=22 direct @960 helper 1+128 (anchor, r3 H960 987,831 +-0.2%)
 *   X1 403_r5 N=22 direct, same state; per-thread results.bin == X0 (HARD)
 *   Z3/Z5 N=21 NQ_LAYOUT=403 direct, 403_r3 vs 403_r5 (the +2.79% penalty)
 *   G22 ./403_r5Py -g 22 22 (production path, helper 1)
 *   P1 STATED: X1/X0 <= 0.985 (refuted if >= 0.997; between: grey).
 *   P2 STATED: Z5/Z3 <= 0.986 (recover >= half of the +2.79%; refuted >= 0.997).
 *   P3 HARD: G22 oracle, layout=403; info: G22 vs 987,079.
 *   P4: 1710 MHz all cells.
 *   ADOPT (S0-S2, Y2-Y3, X1 identity, P1 held, P3): 403_r5 is the production
 *   binary/.py; N=22 production value = G22; N=21 unchanged (same SASS).
 *
 * 403-r4 RESULT (2026-09-30): NOT adopted. G21 109,460.0 (same SASS, same
 *   env: +0.0002% vs r2d G21). Helper OFF at N=22 via the dispatcher: G22a
 *   987,601 / G22b 987,810 vs same-session control C22 (403_r2dPy, helper 1)
 *   987,049 = +0.056% / +0.077% (V4 REFUTED; C22 reproduced yesterday's
 *   G22 987,079 to -0.003%). Across all four N=22 states (dispatcher/direct
 *   x helper on/off) the spread is 0.13% with no monotone relation to the
 *   helper or to free_mb: 403-r3's -0.125% was the value of one direct-run
 *   state, not a helper effect. Production stays 403_r2d (helper for all N).
 *
 * rev403-r3 -- NO code change (code region byte-identical to 403_r2d,
 *   comment-stripped kernel sha 14cba3d9...). N=22 sweep: MAX_BLOCKS
 *   800 / 960 / 1040 x helper {1 + 128 MB, none}, all direct, on this one
 *   binary, plus a 960+helper replicate for the N=22 noise floor.
 *
 * 403-r2d RESULT (2026-09-29): ADOPTED as the production binary.
 *   S2 <402> SASS identical to 402_r5 (496), S3 <403> identical to 403
 *   (504). X0 402_r5 = 109,416.6; X1 403_r2d = 109,414.8 (-0.002%: identical
 *   SASS, identical time); X3 layout 403 = 112,472.7 (x1.0279, 403's E1
 *   112,464 to +0.008%); G21 = 109,459.8 (+0.021% vs 109,437); G22 =
 *   987,079.3 (+0.0015% vs 403's 987,065). All hard gates, all oracles,
 *   1710 MHz. Lesson of r2..r2d: performance reproducibility is guaranteed
 *   by SASS identity, not by source-text identity; a folded runtime branch
 *   is not the same front-end input as no branch (if constexpr is).
 *   Head: ONE binary for N <= 22; production N=21 109,4xx, N=22 987,0xx.
 *
 * 403-r3 (harness only)
 *   Cells (direct, N=22, BLOCK=32, sched order; helper state fixed within
 *   each ladder, as the standing rule requires):
 *     H960 H800 H1040 : helper 1 + 128 MB      (H960 = anchor, 987,079)
 *     N960 N800 N1040 : no helper (NQ_HELPER_CTX=0)
 *     H960r           : replicate of H960 at the end (N=22 noise floor)
 *   Q1 STATED: within the helper ladder 960 is the minimum and both 800 and
 *      1040 are >= +2% (the frame/L1 mechanism is per-SM, so N-independent;
 *      N=21 had +4% on both sides). Refuted if either neighbour is < +1%.
 *   Q2 STATED: helper effect at 960 is present: N960 - H960 in [+1%, +3%]
 *      (N=21: +2.26%). Refuted if |N960 - H960| < 0.5% (N=22 sits outside
 *      the N=21 notch: free_mb 20,665).
 *   Q3 info: N=22 replicate |H960r - H960| (expected <= 0.1%).
 *   Q4 HARD: every cell oracle MATCH; Q5 1710 MHz; GPU empty before each.
 *   Nothing is adopted here. If any cell beats H960 by > 1%, 403-r3b
 *   replicates it before any table change.
 *
 * rev403-r2d -- the two-layout source, spelled so the FRONT END sees 402_r5.
 *   Code-region diff vs 403_r2 (comment-stripped): removed=9 added=23;
 *   kernel region sha changes from 403_r2's 4b9ce994....
 *
 * 403-r2c RESULT (2026-09-29, compile-only): C0 held (nvcc deterministic:
 *   402_r5 rebuilt = 402_r5 binary, 496 instructions). R2 REFUTED: ALL eight
 *   spellings -- no __forceinline__, explicit instantiation in either order,
 *   two plain kernels with 402_r5's exact mangled name, and even the single-
 *   <402> builds with no second kernel in the unit -- give exactly the same
 *   63-instruction SASS difference. So neither the template, nor inlining,
 *   nor the second kernel is the cause. P1: the PTX already differs (69
 *   canonical lines): the loop-carried `mov`s for cur_ld/rd/col/avail/depth
 *   are emitted in a different order and the pop's ld.local is placed
 *   differently. The front end (NVVM) builds SSA from a function that HAS an
 *   `if (layout == 402) ... else ...` at the three stack sites; folding it
 *   afterwards leaves different value numbering / PHI order than 402_r5's
 *   branch-free function, and ptxas then ties differently. Conclusion: a
 *   folded runtime branch is not the same input as no branch.
 *
 * WHAT r2d CHANGES (spelling only; arithmetic unchanged)
 *   process_one_task becomes `template <int layout>` (C++/CUDA) and the
 *   three stack sites use LAYOUT_IF = `if constexpr`: the discarded layout
 *   is not instantiated, so the <402> body the front end sees is 402_r5's
 *   statements with no branch (plus a harmless compound-statement scope).
 *   The plain-C CPU harness (gcc -x c) has no templates or if constexpr:
 *   there POT_LAYOUT_PARAM is `, const int layout`, LAYOUT_IF is `if`, and
 *   POT_TARGS/POT_LAYOUT_ARG pass the runtime int -- exactly 403-r2's CPU
 *   path. The __global__ kernel template calls process_one_task<LAYOUT>.
 *   Local check: 4,000 synthetic N=21 and N=22 records (garbage in the dead
 *   bits): 402 branch, 403 branch and 403 agree byte for byte; g++ -std=c++17
 *   -fsyntax-only with stub CUDA headers accepts the template/if constexpr.
 *
 * GATES (403_r2d_validate.sh; in order; HARD unless marked)
 *   S0  static (402_r5 push/pop verbatim, LAYOUT_IF x3, plumbing, guards,
 *       gcc build, fingerprints).
 *   S1  ptxas: both instantiations 160 B / spill 0 / regs <= 44, 2 entries.
 *   S2  SASS of <402> IDENTICAL to 402_r5's kernel (branch targets included).
 *       Nothing runs on the GPU unless this holds.
 *   S3  STATED: SASS of <403> identical to 403's kernel.
 *   Y2  CPU per-record equality x3; Y3 refusals x2; Y4 X1 and X3 per-thread
 *       identical to X0 (402_r5), N=21 @960 helper 1+128 MB.
 * PRE-REGISTERED (403_r2d_README_append.md)
 *   Y5' STATED |X1 - X0| <= 0.05% (identical SASS => inside the noise floor).
 *   Y6' STATED X3/X0 in [1.025, 1.031] (403's +2.79% with 403's own SASS).
 *   Y7  HARD -g 21 21 MATCH, layout=402; STATED within 0.15% of 109,437.
 *   Y8  HARD -g 22 22 MATCH, layout=403; STATED within 0.5% of 987,065.
 *   Y9  1710 MHz in every cell.
 *   ADOPTION: S0-S2, Y2-Y5', Y7 hold -> 403_r2d is the production table row
 *   (one binary, N <= 22; N=21 109,4xx, N=22 987,0xx).
 *
 * rev403-r2 -- ONE BINARY FOR N <= 22: both 12-byte layouts, chosen per N.
 *   Kernel region sha256 (comment-stripped) changes from 403's c0982cbc...;
 *   code-region diff vs 403: removed=25 added=99 code lines. From this
 *   harness on, every .cu diff/sha gate strips C comments first, so notes
 *   added to this file by hand never move a fingerprint (standing rule that
 *   the 403 harness still violated on the .cu side).
 *
 * WHERE WE STAND (403, 2026-09-28)
 *   402_r5: production -g 21 21 = 109,437 ms (1:49.4), N <= 21 only.
 *   403:    N=22 -g 22 22 = 987,065 ms (16:27, oracle MATCH, free_mb 20,665,
 *           k=935); all four hard gates passed (ptxas 160 B / 40 regs, CPU
 *           equality on 8,192 real N=21 and 16,384 real N=22 records, N=23
 *           refused, N=21 full input per-thread identical to 402_r5).
 *           BUT W5 was REFUTED: at N=21 the 403 layout costs +2.79%
 *           (E0 402_r5 = 109,416 vs E1 403 = 112,464). The fields cross the
 *           word boundary (ld is split 20+1) and the pop's dependency chain
 *           gets longer -- 393-8 was right about exactly that class of
 *           instruction. So 402_r5 stays production for N=21 and 403 is the
 *           N=22 binary: TWO binaries, TWO table rows in the head.
 *
 * WHAT 403-r2 DOES (host + kernel plumbing, NO arithmetic change)
 *   process_one_task() gains a `const int layout` parameter. Each of the
 *   three stack sites (two pushes, one pop) is `if (layout == 402) {402
 *   text} else {403 text}`. The 402 text is the 402/402_r5 push/pop
 *   RECONSTRUCTED from the 402 README (word A = ld[0..20] | col << 21 |
 *   avail << 42, word B = rd); 403_r2_validate.sh extracts the real 402_r5
 *   statements and requires them to be present here character-for-
 *   character (whitespace-normalised) before anything is built. First run
 *   caught one: 402_r5's pop reads `cur_rd = stack_b[stack_ptr];` directly
 *   (no packed_b local); this file now does the same in the 402 branch and
 *   keeps `packed_b` inside the 403 branch only. 6/6 statements match.
 *   The __global__ kernel becomes `template <int LAYOUT>` instantiated for
 *   402 and 403; process_one_task is force-inlined (nvcc already inlined it
 *   in 402/403 -- one ptxas entry, one 160 B frame) so `layout` is a
 *   compile-time constant in each instantiation and no runtime branch is
 *   emitted. The host picks N <= 21 -> 402, N = 22 -> 403 (NQ_LAYOUT=402|
 *   403|auto overrides; 402 at N=22 is refused rc=3; N > 22 still rc=3),
 *   prints "[gpu-layout] N=.. requested=.. layout=.." and launches that
 *   instantiation. cudaFuncSetAttribute / GetAttributes address the chosen
 *   instantiation. The CPU harness does the same ("[cpu-layout] ...") and
 *   passes the choice into process_one_task at run time.
 *   Local check (this file vs 403 on 4,000 synthetic N=21 and 4,000 N=22
 *   records with garbage in ld above the board and in rd bit 0): the 402
 *   branch, the 403 branch and 403 itself agree byte for byte at N=21; the
 *   auto path agrees with 403 at N=22. NQ_LAYOUT=402 at N=22, NQ_LAYOUT=bad
 *   and N=23 all return 3.
 *
 * GATES BEFORE ANY TIMING IS READ (403_r2_validate.sh; in order; HARD)
 *   Y0  static: 402_r5's push/pop statements present verbatim; both
 *       layouts present; template instantiated twice; guards; gcc builds.
 *   Y1  ptxas: BOTH instantiations frame 160 B, spill 0/0, regs <= 44.
 *   Y2  CPU per-record equality on real sched records: N=21 auto == 402_r5;
 *       N=21 NQ_LAYOUT=403 == 403; N=22 auto == 403.
 *   Y3  N=23 -> rc=3; NQ_LAYOUT=402 at N=22 -> rc=3.
 *   Y4  N=21 full input, MB=960, helper 1+128 MB: X1 (403_r2 auto) and X3
 *       (403_r2 NQ_LAYOUT=403) both per-thread byte-identical to X0
 *       (402_r5); all MATCH.
 * PRE-REGISTERED (403_r2_README_append.md; fixed before execution)
 *   Y5  stated: |X1 - X0| <= 0.15% (the template is free: same SASS shape
 *       as 402_r5). Refuted if > 0.3% -> 403-r2 is NOT adopted and the
 *       table row goes back to 402_r5.
 *   Y6  stated: X3 / X0 in [1.025, 1.031] (reproduces 403's +2.79%; proves
 *       the 403 branch is what runs when the 403 layout is selected).
 *   Y7  HARD oracle for -g 21 21 (G21), and its CRunner log shows
 *       "[gpu-layout] N=21 requested=auto layout=402"; stated: G21 within
 *       0.15% of 109,437 (production restored on one binary).
 *   Y8  HARD oracle for -g 22 22 (G22, SKIP22=1 skips it), log shows
 *       layout=403; stated: G22 within 0.5% of 987,065 (first replicate of
 *       N=22 at this configuration; the N=22 noise floor is not yet known).
 *   Y9  mean SM clock within 2% of 1710 in every cell.
 *   ADOPTION: Y0-Y5 and Y7 all hold -> this binary IS the production table
 *   row (the .py already names it; nothing else changes) and the head goes
 *   back to ONE binary, ONE row, production 109,4xx ms for N=21 and
 *   987,0xx ms for N=22. Any of them failing -> revert the row to 402_r5.
 *
 * rev403 -- KERNEL CHANGE: the 12-byte frame re-laid out for N <= 22.
 *   Kernel region sha256 changes from 402's 122b3319...; code-region diff vs
 *   402_r5: removed=21 added=33. Host side: only the guard (N > 22 refused).
 *
 * WHERE WE STAND (402-r5, 2026-09-28): production -g 21 21 = 109,437 ms
 *   (1:49.4): 12-byte frame, MAX_BLOCKS 960, one forked helper context with
 *   128 MB. The 402 layout (3 x 21 bits) refuses N=22, and N=22 is the next
 *   goal (baseline 1,140,988 ms with the 208-byte frame at MB=800, 9 Sep).
 *
 * THE LAYOUT -- two dead bits buy the 3 missing bits
 *   ld bit N-1 is dead: the saved ld is only used as (ld|bit) << step,
 *   step >= 1, then masked with bm. So ld needs N-1 bits (21 at N=22).
 *   rd bit 0 is dead: the saved rd is only used as (rd|bit) >> step,
 *   step >= 1. So rd needs 31 bits.
 *   word A (u64) = col(22) | avail(22) << 22 | ld[0..19] << 44
 *   word B (u32) = (rd & ~1) | ld[20]
 *   22 + 22 + 21 + 31 = 96 bits. Push/pop cost: one extra AND/OR/shift each
 *   vs 402; local accesses still 2 per push and 2 per pop; frame still 160 B.
 *   Local checks: synthetic N=22 (250 records, 21,747,412 solutions) and
 *   N=21 (150 records) with GARBAGE deliberately written into ld above the
 *   board and into rd bit 0 / rd bits 24-31: 401_r4 (unpacked) and 403 agree
 *   byte for byte. N=23 refused (rc=3).
 *
 * GATES BEFORE ANY TIMING IS READ (in this order)
 *   W1  ptxas: frame 160 B, spill 0/0, registers <= 44.
 *   W2  CPU harness: 402_r5 vs 403 on the first CPU_CHECK_RECORDS real N=21
 *       sched records, AND 401_r4 vs 403 on the first CPU_CHECK_RECORDS_22
 *       real N=22 sched records -> per-record results byte-identical.  HARD.
 *   W3  N=23 refused (rc=3, [403-pack]).  HARD.
 *   W4  N=21 full input, MB=960, helper 1+128 MB: 402_r5 vs 403 per-thread
 *       results byte-identical, both MATCH.  HARD.
 * PRE-REGISTERED (403_README_append.md; fixed before execution)
 *   W5  stated: |E1 - E0| <= 0.3% at N=21 (the extra shuffling is free;
 *       E0 = 402_r5 @960 with helper, ~109,4xx).
 *   W6  HARD: -g 22 22 through the dispatcher (403 table: MB=960, helper
 *       1+128 MB) gives total_sum 2,691,008,701,644 MATCH.
 *   W7  stated: G22 kernel_ms <= 0.85 x 1,140,988 = 969,840 (the N=21 gains
 *       carry to N=22: frame/960 about -16%, helper about -2%).
 *       Refuted if G22 > 1,050,000: N=22 behaves differently (k=1,122 per
 *       thread, no tail effect; memory state is outside every notch measured).
 *   W8  informational: free_mb at N=22, and [gpu-config] k_per_thread_max.
 *   W9  mean SM clock within 2% of 1710 in every cell.
 *   FULL22=1 (optional, +20 min): 401_r4 direct N=22 @960 with helper, and
 *   per-thread byte-equality against G22's result file.
 *   Nothing is adopted; 403 is the enabling step for N=22 work (403-r2:
 *   MB sweep at N=22, helper on/off).
 *
 * rev402-r5 -- NO code change. Byte-identical to 402_r4_kernel_maxd14.cu
 * below the first #include. The production change is in the .py: the table
 * env_prefix gains NQ_HELPER_CTX=1 NQ_HELPER_MB=128.
 *
 * WHAT 402-r4 MEASURED (2026-09-28; -g path, N=21, MB=960, 2 rounds)
 *   G0  no helper          111,985.8            spread 0.025%
 *   G1  1 child            109,957.9  -1.81%
 *   G1m 1 child + 128 MB   109,459.0  -2.26%    spread 0.001%   <- winner
 *   G2  2 children         109,447.8  -2.27%    (|G1m-G2| = 0.010%: tie -> G1m)
 *   D1  direct, 1 child, no holder   111,995.9  = r3 H0 (+0.031%)
 *   D2  direct, 2 children           109,978.1  = r3 H2 (+0.008%)
 *   T1..T7 all held: a forked child IS an external holder, helpers never
 *   outlive the parent, clocks 1710 in every cell.
 *
 * WHAT r5 DOES -- adopt on the production path and confirm.
 *   .py: env_prefix "NQ_EXTRA_CTX=1 NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "
 *   (NQ_EXTRA_CTX=1 is inert since 9 Sep but is left in place: removing it
 *   would be a second change, and G1m was measured with it present.)
 *   Cells: G21a, G21b = bare -g 21 21 twice; G1921 = -g 19 21 (N=19, 20, 21
 *   through the CRunner path with the helper, oracle-gated each).
 *
 * PRE-REGISTERED (402_r5_README_append.md; fixed before execution)
 *   U1  HARD: dispatch.log shows env_prefix=NQ_MAX_BLOCKS=960 NQ_EXTRA_CTX=1
 *       NQ_HELPER_CTX=1 NQ_HELPER_MB=128 and the CRunner log shows
 *       [gpu-helper] helpers=1 helper_mb=128, MAX_BLOCKS=960.
 *   U2  G21a, G21b within +-0.15% of 109,459 and within 0.05% of each other.
 *   U3  -g 19 21: N=19, 20, 21 all MATCH their oracles; N=20 within +-1.0%
 *       of 16,48x (401-r3 Gd; the helper state at N=20 is unmeasured, so
 *       this is informational for 19/20, hard only on the oracles).
 *   U4  mean SM clock within 2% of 1710 in every cell.
 *   Adoption: U1 and U2 -> production = 109,46x ms (1:49.5).
 *
 * rev402-r4 -- HOST-SIDE ONLY: NQ_HELPER_CTX / NQ_HELPER_MB. Kernel region
 * unchanged from 402 (122b331980495f8b..., the 12-byte frame). Code-region
 * diff vs 402_r3: removed=0 added=87. The CPU test main() is untouched.
 *
 * WHAT 402-r3 MEASURED (2026-09-28; N=21, MB=960, external holders)
 *   H0 (1 holder, 2 ctx)        111,961   G0 (-g, 2 ctx)        111,965
 *   H2 (2 holders, 3 ctx)       109,969  -1.78%
 *   Gh (-g + 1 holder, 3 ctx)   109,974  -1.78%   (Gh = H2 to 0.004%)
 *   H3 (4 ctx)                  109,468  -2.23%
 *   H2b (0 + 128 MB, 3 ctx)     109,417  -2.27%
 *   H2c (0 + 384 MB, 3 ctx)     109,482  -2.21%
 *   Not a notch as at 800 (-0.29%, < 128 MiB wide) but a step: a third
 *   context is -1.78%, and any further occupancy is a flat -2.2%. The
 *   adoption rule of r3 was met (replicates 0.002% / 0.011%).
 *
 * WHAT r4 BUILDS -- the holder inside the binary.
 *   NQ_HELPER_CTX=<n>: fork n children before any CUDA call; each creates
 *   its own context (cudaFree(0)), optionally touches NQ_HELPER_MB MiB, then
 *   blocks on a pipe until the parent exits (PR_SET_PDEATHSIG as backstop).
 *   The parent waits for each child's readiness byte, so the state exists
 *   before the first cudaMalloc. Unset = nothing forked = 402-r3 exactly.
 *   Why a child process and not NQ_EXTRA_CTX: 401-r4 H1 (in-process context,
 *   -0.001%) vs H2 (other process, -0.288%) -- only another process's
 *   context counts, for a reason still unknown.
 *
 * CELLS (all N=21, MB=960 default; -g cells inherit NQ_HELPER_* from the
 * harness environment -- os.system passes the environment through, and the
 * table's env_prefix is unchanged)
 *   G0 -g, helpers 0            G1 -g, HELPER_CTX=1
 *   G1m -g, CTX=1 MB=128         G2 -g, HELPER_CTX=2
 *   D1 direct, HELPER_CTX=1, NO external holder  (must equal H0's 2-ctx value)
 *   D2 direct, HELPER_CTX=2, no holder           (must equal H2's 3-ctx value)
 *   round 2: G2 G1m G1 G0 (reversed)
 *
 * PRE-REGISTERED (402_r4_README_append.md; fixed before execution)
 *   T1  HARD: G0 within +-0.15% of 111,965, [gpu-helper] helpers=0.
 *   T2  HARD (mechanism): D1 within +-0.15% of 111,961 (r3 H0) -- one
 *       forked child = one external holder. D2 within +-0.15% of 109,969.
 *   T3  stated: G1 <= G0 - 1.5% (about 109,97x, as Gh).
 *   T4  stated: G1m and G2 both <= G0 - 2.0% and within 0.15% of each other.
 *   T5  every -g config's two runs within 0.05%.
 *   T6  no compute process remains after any cell (the gate before the next
 *       cell would catch a straggler).
 *   T7  mean SM clock within 2% of 1710 in every cell.
 *   ADOPTION: winner <= G0 - 1.5% with T5. Tie-break: if |G1m - G2| <= 0.15%
 *   take G1m (3 contexts, one child) over G2 (4 contexts, two children).
 *   402-r5 then puts the winner's NQ_HELPER_* into the table's env_prefix
 *   (executable-line change in the .py) and re-measures -g 21 21 once.
 *   Nothing is adopted from this run.
 *
 * rev402-r3 -- NO code change. Byte-identical to 402_r2_kernel_maxd14.cu
 * below the first #include (12-byte frame, 160 B/thread).
 *
 * WHAT 402-r2 SETTLED (2026-09-28; Q1-Q6 all held)
 *   A800 129,512.7  A880 117,122.4  A960 112,001.4  A1040 116,080.7
 *   (2 interleaved rounds, spread <= 0.028%)  G21 -g 21 21 = 112,000.4
 *   A10G_FINAL_DEFAULT_MAX_BLOCKS=960 adopted. Production 1:52.0.
 *
 * WHAT r3 ASKS -- does the third-context notch exist at MB=960?
 *   401-r4 H2 (2 holders, 3 real contexts) was -0.288% at MB=800 with the
 *   old frame; 395c-r5 showed the fast region is a notch < 128 MiB wide and
 *   moves with the binary's own device footprint. MB=960 changes that
 *   footprint (results buffer, stride), so the notch must be re-measured
 *   before anything is built to recover it. Zero code change; every state
 *   is made by external holders (permanent rule).
 *   N=21, MB=960 (default), 402-r3 binary:
 *     G0    -g 21 21                     production anchor (parent ctx: 2 ctx)
 *     H0    1 holder                     direct 2-ctx anchor
 *     H2    2 holders (0 MB each)        3 real ctx, +255 MiB
 *     Gh    -g 21 21 + 1 holder alive    production path + helper: 3 ctx
 *     H3    3 holders                    4 ctx (401-r4: 2-3 optimal, 5 bad)
 *     H2b   holder(0) + holder(128 MB)   3 ctx, +383 MiB  } is the valley
 *     H2c   holder(0) + holder(384 MB)   3 ctx, +639 MiB  } still at +255?
 *   round 2 repeats H0 H2 Gh (the cells any adoption would rest on).
 *
 * PRE-REGISTERED (402_r3_README_append.md; fixed before execution)
 *   S1  HARD: mean H0 within +-0.15% of 112,001 (402-r2 A960), free_mb 22018.
 *   S2  stated (weak): H2 <= H0 - 0.20%  (the notch survives the move to
 *       960, about -0.29% as at 800).  Alt: |H2 - H0| <= 0.10% -> the notch
 *       was specific to the 800 landscape; nothing to recover, axis closed.
 *   S3  |Gh - H2| <= 0.10% (a holder beside -g is the same 3-ctx state) and
 *       Gh <= G0 - 0.20%.
 *   S4  H3 not better than H2 by more than 0.10%.
 *   S5  neither H2b nor H2c better than H2 by more than 0.10% (exploratory:
 *       report, do not conclude).
 *   S6  mean SM clock within 2% of 1710 in every cell.
 *   ADOPTION RULE: only if S1-S3 hold and the round-2 replicates of Gh and
 *   G0-vs-Gh are within 0.05% does 402-r4 implement the helper in the
 *   binary (NQ_HELPER_CTX: fork a child that holds a context until exit),
 *   itself gated by an interleaved A/B on the -g path. Nothing is adopted
 *   from this run.
 *
 * rev402-r2 -- NO code change. Byte-identical to 402_kernel_maxd14.cu below
 * the first #include (kernel region 122b331980495f8b..., the 12-byte frame).
 *
 * WHAT 402 MEASURED (2026-09-28; 1 holder, 2 contexts, N=21, one run each)
 *   gates: ptxas 208 -> 160 B, spill 0/0, 40 regs; CPU equivalence on 8,192
 *   real records; N=22 refused; full-input per-thread results byte-identical
 *   E0  401_r4 @800   133,576.9   (base)
 *   E1  402    @800   129,523.5   -3.04%   (fewer sectors alone pay -- 393-8
 *                                           finally refuted, V6 in the good
 *                                           direction)
 *   P960       @960   111,997.0  -16.16%   12 warps/SM, 60 KB   <- best
 *   P1040      @1040  116,102.2  -13.08%   13 warps/SM, 65 KB
 *   P1120      @1120  120,777.8   -9.58%
 *   P1280      @1280  144,719.3   +8.34%
 *   111,997 ms = 1:52.0, under the 2:02.52 target -- from one run.
 *
 * WHAT r2 DOES -- promote on an interleaved A/B, as 402's adoption rule says.
 *   Direct, 1 holder, NQ_EXTRA_CTX=0, 402-r2 binary:
 *     round 1: A800 A880 A960 A1040     round 2: A1040 A960 A880 A800
 *   (880 = 11 warps/SM, 55 KB, never measured with the 160 B frame; the
 *   optimum is somewhere in 880..1040.)  Then G21 = -g 21 21 through the
 *   dispatcher with A10G_FINAL_DEFAULT_MAX_BLOCKS=960 (this .py), so the
 *   production path itself is measured with the new default.
 *
 * PRE-REGISTERED (402_r2_README_append.md; fixed before execution)
 *   Q1  HARD: mean A800 within +-0.15% of 129,524 (402 E1), free_mb 22018.
 *   Q2  every config's two runs within 0.05% of each other.
 *   Q3  stated: A960 is the minimum; A880 lands between A800 and A960;
 *       A1040 within +-0.3% of 116,102.  Alt: A880 < A960 -> 11 warps/SM.
 *   Q4  ADOPTION: the winner is <= A800 - 10% and satisfies Q2. If the winner
 *       is 960, this .py's default stands. If it is 880 or 1040, the number
 *       is adopted but the default is changed in 402-r3 (3-line .py, one G21).
 *   Q5  G21 within +-0.30% of mean A960 (or of the winner if the winner is
 *       960), [gpu-config] MAX_BLOCKS=960, dispatch.log shows NQ_MAX_BLOCKS=960.
 *   Q6  mean SM clock within 2% of 1710 in every cell.
 *   Every run oracle-gated; GPU empty except the holder before every cell.
 *   N=22 is still refused by this binary (rc=3): bare -g stops there.
 *
 * rev402 -- FIRST KERNEL CHANGE SINCE 395a r2. The DFS stack frame goes
 * from 16 to 12 bytes. Kernel region sha256 CHANGES (harness gates that it
 * changed, and that the code-region diff vs 401_r4 is exactly removed=21
 * added=46). Host side: only the N>21 refusal.
 *
 * WHY NOW (401-r3/r4, 2026-09-28, all in the 2-context holder state)
 *   MB=720 (9 warps/SM)  +7.86%   MB=800 (10)  base   MB=880 (11) +2.47%
 *   MB=1280 (16, 104 KB) +105.8%
 *   Inside L1 more warps pay steeply; 800 is the top of L1 at 208 B/thread
 *   (65 KB). The 208-byte frame is the gatekeeper (401). Shrinking it is the
 *   only way to add warps without leaving L1. 393-8 rejected frame packing
 *   because it adds instructions; 395a r2 measured that ALU count does not
 *   move the critical path in this stall-bound regime, so that objection is
 *   set aside -- and measured again here (V6).
 *
 * THE LAYOUT (and why 397 failed)
 *   old frame (2 x u64): ld|rd<<32 , col|(avail|depth<<27)<<32   = 16 B
 *   402 frame:  word A (u64) = (ld & M21) | col<<21 | avail<<42    (63 bits)
 *               word B (u32) = rd, all 32 bits
 *               depth: NOT in local memory -- 4 bits per frame in one
 *                      register (stack_depth), LIFO, 13 x 4 = 52 bits
 *   ld's bits above the board are dead: every use is a LEFT shift followed
 *   by a mask with bm (398's analysis). rd's high bits come DOWN and are
 *   constraints, so rd is kept whole -- 397 masked rd and produced
 *   427,786,151,034 solutions instead of 314,666,222,712. 401's idea of
 *   rebuilding the parent from the child state is not possible: at pop time
 *   the current node is an arbitrary descendant of the popped frame, not
 *   its child (frames of nodes with no remaining siblings are never pushed).
 *   Local-memory accesses stay 2 per push and 2 per pop.
 *   3 x 21 = 63 bits, so this packing is valid for N <= 21 ONLY. N=22 needs
 *   3 x 22 = 66 and a different split; both main()s refuse N > 21 right
 *   after parsing argv, before opening any file.
 *
 * BUDGET
 *   65 KB / 156 B = 426 threads/SM = 13 warps/SM (MAX_BLOCKS 1040 at
 *   BLOCK=32). 12 warps = 58.5 KB, 14 warps = 68.3 KB, 16 warps = 78 KB.
 *
 * GATES BEFORE ANY TIMING IS READ (in this order)
 *   V1  ptxas: frame 208 -> 152..168 B, spill 0/0, registers <= 44.
 *   V2  CPU harness (gcc, same process_one_task): 401_r4 vs 402 on the
 *       first CPU_CHECK_RECORDS real sched records -> per-record results
 *       byte-identical.  HARD.
 *   V3  the binary refuses N=22 (rc=3, [402-pack] line), no GPU work.  HARD.
 *   V4  full input, 2,025,282 records, MB=800, both binaries: per-thread
 *       result files byte-identical and both MATCH the oracle.  HARD -- the
 *       gate that stopped 397, now first.
 * PRE-REGISTERED PREDICTIONS (402_README_append.md; fixed before execution)
 *   V5  E0 (401_r4 binary, MB=800, 1 holder) within +-0.15% of 133,565.
 *   V6  stated: |E1 - E0| <= 0.5% at MB=800 -- the pack/unpack ALU is free
 *       and the smaller frame alone does not help at 10 warps.
 *       Alt: E1 >= E0 + 1% -> 393-8 was right after all; the sweep must
 *       first pay that back.
 *   V7  THE DECISION. stated: min(P960, P1040) <= E0 - 3% (point guess
 *       about -8% from r4's 9->10 slope of -7.3%, discounted for a
 *       flattening slope). Refuted if nothing is below E0 - 1%: then more
 *       warps do not pay beyond 10 even inside L1, and this axis closes.
 *   V8  P1120 and P1280 are both slower than the best of P960/P1040: the
 *       cliff is a capacity (~65 KB), not a warp count, and moves with the
 *       frame exactly as the budget predicts.
 *   V9  mean SM clock within 2% of 1710 in every cell.
 *   Adoption rule: nothing becomes a production default from this run. A
 *   winning MB is an A/B candidate for 402-r2 (interleaved, >= 2 reps).
 *   Every run oracle-gated; GPU empty except the holder before every cell.
 *
 * rev401-r4 -- NO code change. Byte-identical to 401_r3_kernel_maxd14.cu
 * below the first #include; kernel region still ebd7f523... (395a r2).
 *
 * WHAT 401-r3 FOUND (2026-09-28)
 *   R5 held: at N=20 direct NQ_EXTRA_CTX=2 = 17,379 (+5.44% vs -g 20 20 =
 *   16,482) while direct NQ_EXTRA_CTX=1 + external holder = 16,481 (-0.008%).
 *   R1 (hard gate) FAILED: direct N=21 MB=800 with NQ_EXTRA_CTX=2 = 139,181,
 *   which is 395c's D0 (139,188; ONE context, no other process) to 0.005%,
 *   although the log said extra_ctx=2 and free_mb=21762. G21 = 133,567,
 *   i.e. the 2-context value; the 395c-r6 +413 ms was not present (nor in
 *   401-r2's own -g 21 21 log of 9 Sep 22:33+, 133,556). Same driver.
 *   Reading: since the evening of 9 Sep an in-process cuCtxCreate context
 *   no longer counts for this ~4% effect; another process's context still
 *   does, reproducibly to 0.01%. Cause unknown (stated as a hypothesis).
 *   The N=21 ladder r3 ran (720 +5.15%, 880 +9.38%, 1280 +84.56%) was
 *   measured in that 1-context state and is NOT read under the
 *   pre-registration.
 *
 * WHAT r4 DOES -- the state is made by an external holder, never by
 * NQ_EXTRA_CTX (proposed permanent rule). All N=21, MB=800 unless noted,
 * one holder = 395c_ctx_holder with 0 MB.
 *   G21   -g 21 21                          production anchor
 *   H0    1 holder, NQ_EXTRA_CTX=0          2 ctx, free_mb 22018  <- anchor
 *   H1    1 holder, NQ_EXTRA_CTX=1          does an in-process ctx add?
 *   H2    2 holders, NQ_EXTRA_CTX=0         3 real ctx, free_mb 21762: is
 *                                           the r4/r5 notch still there?
 *   L720 L800 L880 L1280  1 holder, ctx=0   the ladder, in the right state
 *   SM clock / power / temperature sampled every 5 s in every cell (dropped
 *   in r3 by mistake).
 *
 * PRE-REGISTERED (401_r4_README_append.md; fixed before execution)
 *   P0  G21 within +-0.15% of 133,567 (r3), extra_ctx=1, free_mb 21762.
 *   P1  HARD: H0 within +-0.15% of 133,561 (395c-r4 D1 = 2 ctx), free_mb
 *       22018 +-2, extra_ctx=0. If it fails nothing below is read.
 *   P2  stated: |H1 - H0| <= 0.10% -- an in-process context adds nothing
 *       now. Alt: H1 <= H0 - 0.25% -> it counts again (time-varying).
 *   P3  stated, weak: H2 <= H0 - 0.25% (~133,18x-133,20x as 395c-r4 X2b):
 *       the notch exists but needs a REAL third context. If H2 ~ H0 within
 *       0.10%, the notch is gone; 133,56x is the production number.
 *   P4  L1280 >= H0 + 20%  (closes 401-r2 M3/M4).
 *   P5  L720 slower than L800 by +1.5..+6%.  Refuted if within +-0.3% or
 *       faster -> inside-L1 slope flat; 402 re-planned before writing.
 *   P6  L880 slower than L800.  If faster by >0.3%: A/B candidate only.
 *   P7  |L800 - H0| <= 0.05% (same state, replicate = noise floor).
 *   P8  mean SM clock within 2% of 1710 in every cell.
 *   Every run oracle-gated; GPU empty before every cell except our holders.
 *
 * rev401-r3 -- NO code change. Byte-identical to 401_r2_kernel_maxd14.cu
 * below the first #include; kernel region still ebd7f523... (395a r2).
 *
 * WHY r3 (2026-09-28): 401-r2 ran with N_LIST="19 20" only, so the decision
 * cell (M3, N=21) never ran. What did run refuted M2: the MB=1280 penalty
 * GROWS with N (N=19 +53.3%, N=20 +71.7%), so records-per-thread is not the
 * mechanism. r3 closes the axis on measurement rather than extrapolation,
 * and measures the two things 402 (104-byte packing) actually rests on.
 *
 * THREE QUESTIONS, ZERO CODE CHANGE
 *   (A) M3 at N=21: MB=800 vs MB=1280 (16 warps/SM, 104 KB, k=49.4).
 *       Stated prediction: penalty >= 20%, i.e. M4 "axis dead OUTSIDE L1".
 *   (B) The premise of 402, measured for the first time at N=21 with the
 *       current binary: is the slope INSIDE L1 still positive? 394g (484 ->
 *       704 -> 800 -> 968: -23.7%, -4.6%, +11.3%) says more warps pay
 *       strongly while the stack stays resident. r3 adds MB=720 (9 warps/SM,
 *       58.5 KB) and MB=880 (11 warps/SM, 71.5 KB, still >=99% hit per 401).
 *       If 720 is NOT slower than 800, more warps do not pay even inside L1
 *       and the packing must be re-planned before it is written.
 *   (C) The N=20 discrepancy found in 401-r2's logs: same binary, same input,
 *       same 3 contexts, same free_mb=21776, yet dispatcher -g 20 20 =
 *       16,480 ms and direct NQ_EXTRA_CTX=2 = 17,378 ms (+5.5%). V4 (397)
 *       was only ever confirmed at N=21. r3 interleaves three N=20 cells
 *       (dispatcher / direct 2 ctx / direct 1 ctx + external holder) to see
 *       whether it reproduces and which side the holder takes. This decides
 *       how 402's debug-scale cells must be run.
 *
 * PRE-REGISTERED (401_r3_README_append.md; fixed before execution)
 *   R0  dispatcher -g 21 21 within +-0.10% of 133,192 (production anchor).
 *   R1  direct N=21 MB=800 within +-0.15% of 133,197 (398 anchor), free_mb
 *       21762, extra_ctx=2. Hard: if this fails the N=21 cells are not
 *       comparable and R2-R4 are not read.
 *   R2  N=21 MB=1280 penalty >= 20% (point guess ~85% from the N=19/20
 *       trend). Closes M3/M4: occupancy beyond L1 is dead.
 *   R3  N=21 MB=720 slower than MB=800 by +1.5..+6% (394g's 8->10 step was
 *       -4.6%). Refuted if 720 is within +-0.3% or faster: then the inside-L1
 *       slope has flattened at 10 warps and 402's premise is weakened.
 *   R4  N=21 MB=880 slower than MB=800 (N=19 gave +6.8% at 71.5 KB).
 *       If 880 is FASTER by >0.3%, 800 was not the top of L1 and 402 gets a
 *       free rung; record it as an A/B candidate, do not adopt from one run.
 *   R5  the N=20 gap reproduces: direct-2ctx minus dispatcher >= +3%.
 *       Stated (weak): the external holder sides with the dispatcher.
 *       Falsified if all three N=20 cells agree within 0.1% (then r2's
 *       17,378 was a session artefact and V4 holds at N=20 too).
 *   Every run is oracle-gated. GPU must be empty before every cell (the
 *   holder, when present, is the only allowed compute process).
 *
 * rev401-r3 -- NO code change. Byte-identical to 401_r2_kernel_maxd14.cu
 * below the first #include; kernel region still ebd7f523... (395a r2).
 *
 * WHY r3 (2026-09-28): 401-r2 ran with N_LIST="19 20" only, so the decision
 * cell (M3, N=21) never ran. What did run refuted M2: the MB=1280 penalty
 * GROWS with N (N=19 +53.3%, N=20 +71.7%), so records-per-thread is not the
 * mechanism. r3 closes the axis on measurement rather than extrapolation,
 * and measures the two things 402 (104-byte packing) actually rests on.
 *
 * THREE QUESTIONS, ZERO CODE CHANGE
 *   (A) M3 at N=21: MB=800 vs MB=1280 (16 warps/SM, 104 KB, k=49.4).
 *       Stated prediction: penalty >= 20%, i.e. M4 "axis dead OUTSIDE L1".
 *   (B) The premise of 402, measured for the first time at N=21 with the
 *       current binary: is the slope INSIDE L1 still positive? 394g (484 ->
 *       704 -> 800 -> 968: -23.7%, -4.6%, +11.3%) says more warps pay
 *       strongly while the stack stays resident. r3 adds MB=720 (9 warps/SM,
 *       58.5 KB) and MB=880 (11 warps/SM, 71.5 KB, still >=99% hit per 401).
 *       If 720 is NOT slower than 800, more warps do not pay even inside L1
 *       and the packing must be re-planned before it is written.
 *   (C) The N=20 discrepancy found in 401-r2's logs: same binary, same input,
 *       same 3 contexts, same free_mb=21776, yet dispatcher -g 20 20 =
 *       16,480 ms and direct NQ_EXTRA_CTX=2 = 17,378 ms (+5.5%). V4 (397)
 *       was only ever confirmed at N=21. r3 interleaves three N=20 cells
 *       (dispatcher / direct 2 ctx / direct 1 ctx + external holder) to see
 *       whether it reproduces and which side the holder takes. This decides
 *       how 402's debug-scale cells must be run.
 *
 * PRE-REGISTERED (401_r3_README_append.md; fixed before execution)
 *   R0  dispatcher -g 21 21 within +-0.10% of 133,192 (production anchor).
 *   R1  direct N=21 MB=800 within +-0.15% of 133,197 (398 anchor), free_mb
 *       21762, extra_ctx=2. Hard: if this fails the N=21 cells are not
 *       comparable and R2-R4 are not read.
 *   R2  N=21 MB=1280 penalty >= 20% (point guess ~85% from the N=19/20
 *       trend). Closes M3/M4: occupancy beyond L1 is dead.
 *   R3  N=21 MB=720 slower than MB=800 by +1.5..+6% (394g's 8->10 step was
 *       -4.6%). Refuted if 720 is within +-0.3% or faster: then the inside-L1
 *       slope has flattened at 10 warps and 402's premise is weakened.
 *   R4  N=21 MB=880 slower than MB=800 (N=19 gave +6.8% at 71.5 KB).
 *       If 880 is FASTER by >0.3%, 800 was not the top of L1 and 402 gets a
 *       free rung; record it as an A/B candidate, do not adopt from one run.
 *   R5  the N=20 gap reproduces: direct-2ctx minus dispatcher >= +3%.
 *       Stated (weak): the external holder sides with the dispatcher.
 *       Falsified if all three N=20 cells agree within 0.1% (then r2's
 *       17,378 was a session artefact and V4 holds at N=20 too).
 *   Every run is oracle-gated. GPU must be empty before every cell (the
 *   holder, when present, is the only allowed compute process).
 *
 * rev401-r2 -- NO code change. Byte-identical to 401_kernel_maxd14.cu below
 * the first #include; kernel region still ebd7f523...
 *
 * WHY r2: 401's sweep was read through two harness errors of mine.
 *
 *   (1) The footprint model ignored the hardware limit of 16 blocks per SM.
 *       With BLOCK=32 a block is one warp, so MAX_BLOCKS above 1280 adds no
 *       resident warps at all -- the extra blocks queue. Every row above
 *       MB=1280 in 401's table reported a footprint it never had, and the
 *       "L2 FAILED: time is not monotone" verdict came entirely from that.
 *       Over the range where the footprint really does change, 800 to 1280,
 *       the curve is monotone: 2154.7, 2301.0, 2279.5, 2433.5, 2854.0,
 *       3032.6, 3304.5 ms. ncu agrees -- MB=2560 reported Achieved
 *       Occupancy 30.25% and 3.75 warps per scheduler, i.e. 15 warps/SM,
 *       not 32. It also explains why 400-r2 measured 10,729 ms at "stride
 *       81,920": that was 64:1280, sixteen blocks of two warps = 32
 *       warps/SM and 208 KB, a different point from 32:2560's 16 warps/SM
 *       and 104 KB.
 *
 *   (2) warps/SM and k_per_thread cannot be separated at one N. Both follow
 *       from stride. At N=19 going from 10 to 16 warps/SM also drops
 *       records per thread from 35.0 to 21.9, so the +53% penalty mixes an
 *       L1 effect with a tail effect and L5's refutation is confounded.
 *
 * WHAT DOES SURVIVE FROM 401
 *   The L1 curve, which is a per-SM capacity property and does not depend
 *   on N: 65.0 KB -> 99.51% hit, 78.0 KB -> 98.56%, 104.0 KB -> 97.68%.
 *   The production point is already at the edge; the effective L1 for local
 *   is around 70 KB. My L3 prediction (the 99% crossing between 80 and
 *   120 KB) was wrong -- it is between 71.5 and 78.0 KB.
 *
 * WHAT r2 DOES
 *   Runs the same MAX_BLOCKS ladder at N=19, N=20 and N=21, because raising
 *   N raises k at every rung:
 *       warps/SM   k at N=19   k at N=20   k at N=21
 *          10        35.0        53.5        79.1
 *          16        21.9        33.4        49.4
 *   At N=21 the 16-warp rung has MORE records per thread than N=19's
 *   baseline rung. If it is still slower there, more warps genuinely does
 *   not pay and the 104-byte packing should never be written. If it is
 *   faster, L5's refutation was a debug-scale artefact and the packing is
 *   back on the table.
 *   No ncu: 396 established that ncu cannot finish the 133 s N=21 kernel,
 *   and the L1 curve from 401 already covers the capacity question.
 *
 * rev401 -- NO code change. Byte-identical to 400_r3_kernel_maxd14.cu below
 * the first #include; kernel region still ebd7f523... NQ_BLOCK and
 * NQ_CARVEOUT are inherited and both stay inert unless set.
 *
 * WHAT 400-r3 SETTLED
 *   The carveout knob does nothing: the driver already gives L1 its maximum.
 *     default effective_pref=-1, NQ_CARVEOUT=0 -> effective_pref=0
 *     stride 81,920: 10,729.669 -> 10,729.866 ms (+0.002%)
 *     L1 hit rate  :     85.96% ->     85.92%
 *   But the diagnostic it carried found the wall exactly:
 *     stride  25,600   320 thr/SM   66.6 KB   L1 hit 99.48%   L2 read   478 M
 *     stride  81,920   987 thr/SM  205   KB   L1 hit 85.96%   L2 read 12,298 M
 *   Miss rate 0.52% -> 14.04% is 27x, and L2 traffic 25.7x. They agree.
 *   And the bottleneck INVERTS with it:
 *                       stride 25,600     stride 81,920
 *     long_scoreboard        12.9%            94.5%
 *     wait                   48.0%             3.1%
 *     branch_resolving       28.5%             1.8%
 *   Occupancy did rise as intended -- warps per scheduler 2.44 -> 7.71 --
 *   but the DFS stack fell out of L1 the moment it did.
 *
 *   So the 208-byte frame is the gatekeeper of the whole occupancy axis,
 *   and at the production point local memory is not a problem at all:
 *   99.48% hit, long_scoreboard 12.9%.
 *
 * WHAT 401 ASKS, WITHOUT WRITING ANY PACKING
 *   Before building a smaller frame it is worth knowing whether more warps
 *   help AT ALL while the working set still fits. 400-r2 and r3 only ever
 *   sampled strides far past the cliff. 401 sweeps footprint finely from
 *   66.6 KB upward in ~6.7 KB steps and asks one question:
 *
 *     is ANY configuration with more than 10 warps/SM faster than the
 *     320-thread baseline?
 *
 *   If yes, extending that range by halving the frame is worth the risk,
 *   and 402 builds it. If no configuration beats the baseline, occupancy
 *   never helps for this algorithm and the packing work would be wasted
 *   however well it were done. Either answer costs about ten minutes and
 *   no code.
 *
 * rev400-r3 -- ONE more host-side knob: NQ_CARVEOUT, plus a
 * cudaFuncGetAttributes report on every run. Kernel region untouched
 * (sha256 still ebd7f523...).
 *
 * WHAT 400-r2 FOUND, and what it overturned
 *   Raising the thread count to raise occupancy makes this kernel 6 to 7
 *   times SLOWER, and at every stride the BLOCK variants agree to within
 *   1%: the driver is stride, not block shape.
 *     stride  25,600  k/thr 35.0   2,120 ms
 *     stride  40,960  k/thr 21.9   3,849 ms   +82%
 *     stride  81,920  k/thr 11.0  10,699 ms  +405%
 *     stride 122,880  k/thr  7.3  16,242 ms  +666%
 *   Z2 held: at constant stride, BLOCK=64 is 1.07% FASTER and BLOCK=128 is
 *   6.54% slower (200 blocks over 80 SMs divides badly). Packaging is
 *   nearly neutral; thread count is not.
 *
 * A CORRECTION THAT MATTERS MORE THAN THE MEASUREMENT
 *   Revisions 396 through 399 repeatedly stated that 394f "balances
 *   cumulative per-thread work over the whole set". That was never checked.
 *   394f_permute_soa7.py's sched mode is a stable sort by (markctrl,
 *   ctrl0) -- schedule similarity. There is no stride in the tool and no
 *   balancing. What 394f actually buys is WARP COHERENCE: the 32 records
 *   sharing a warp are consecutive in the file and therefore have the same
 *   or adjacent control words, so the lanes follow the same path. That is
 *   exactly what 399 measured -- shuffling costs 35.2% and drops lanes from
 *   10.53 to 6.78 of 32.
 *   Warp coherence does not depend on stride, so it cannot be what breaks
 *   at larger strides. What changes with stride is resident threads per SM,
 *   and with them the local-memory working set. Hence r3.
 *
 * rev400-r2 -- identical to 400_kernel_maxd14.cu below the first #include
 * (whole code region sha256 gated). r2 exists only to rescale the harness:
 * 400 swept at N=20 and confirmed at N=21, which is a 45-minute run and no
 * way to debug a new harness. r2 shifts every N down one -- sweep at N=19
 * (2.1 s a run), confirm at N=20 (16.4 s), optional check at N=21 -- and
 * the whole thing lands in about 15 minutes. NPROF/NCONF/NBIG put it back
 * at production scale in one line.
 *
 * ONE CAVEAT THE SMALLER SCALE INTRODUCES: records per thread falls with N,
 * so the large-stride configurations lose more of 394f's balancing at N=19
 * (k=5.5 at stride 163,840) than they would at N=21 (k=12.4). The debug
 * scale is therefore BIASED AGAINST exactly the configurations the sweep is
 * looking for. A null result at N=19 does not refute the hypothesis; the
 * N=20 confirmation is what counts.
 *
 * rev400 -- ONE host-side knob: NQ_BLOCK. The kernel region is untouched
 * (sha256 still ebd7f523...) because BLOCK only ever reached the kernel as
 * blockDim.x and the launch configuration.
 *
 * WHY, from 399-r2's stall breakdown (N=20, excluding `selected`, which is
 * issue rather than stall):
 *     wait               45.05%   2.32 cycles per issue
 *     branch_resolving   26.80%   1.38
 *     long_scoreboard    13.98%   0.72
 *     no_instruction      7.57%   0.39
 *   N=19 gives the same ranking, so this is not an artefact of the proxy.
 *   `wait` is fixed-latency dependency stalling. Memory is 14%. That
 *   overturns the "L1-bound" reading 398-r2's Speed-of-Light numbers
 *   suggested: the L1 pipe is busy, but the warps are not waiting on it.
 *   With 2.43 of 12 warp slots per scheduler filled, there is nothing to
 *   hide even short latencies with.
 *
 * THE CEILING THAT WAS NEVER LIFTED
 *   On sm_86 an SM holds at most 16 blocks and 48 warps. BLOCK=32 makes a
 *   block exactly one warp, so 16 blocks/SM caps theoretical occupancy at
 *   33.3% however large the grid is -- and 800 blocks over 80 SMs reaches
 *   only 10 of those 16. MAX_BLOCKS has been swept before (394b, 394g);
 *   BLOCK has sat at 32 throughout, which is what actually holds the
 *   ceiling down. Registers do not: 37 per thread allows 52 warps.
 *
 * WHAT 400 DOES NOT ASSUME
 *   Raising warps also raises the resident local-memory footprint and cuts
 *   records per thread, which weakens 394f's balancing. Whether the net is
 *   positive is measured, not argued: every configuration is timed at N=20
 *   with three reps against the same baseline in the same session, the
 *   winner is confirmed at N=21, and ncu checks that occupancy actually
 *   rose and `wait` actually fell before any of it is believed.
 *
 * rev399-r2 -- NO code change. Byte-identical to 399_kernel_maxd14.cu below
 * the first #include; kernel region still the 395a r2 sha256 ebd7f523...
 *
 * WHAT 399 SETTLED, and what it lost
 *   Phase C, N=20, three reps each, every run oracle-MATCH:
 *     394f order (asis)  16,482.743 ms   spread 0.020%
 *     desc               16,548.483 ms   +0.399%
 *     asc                17,054.993 ms   +3.472%
 *     random shuffle     22,285.509 ms   +35.205%
 *   Record ORDER is worth 35.2%. 394f's schedule buys 26.0% over random.
 *   That is far past the 2% threshold X3 was registered at, so ordering is
 *   a live axis -- and X4 held: neither crude popcount sort beat 394f.
 *   Phase B says why: within a 32-record warp group, the variance of w_lo
 *   is 8.2% of the file-wide variance, so 394f already aligns warps on the
 *   schedule word. popcount(col) turned out to have zero variance at all --
 *   every record has the same number of placed queens -- so it is not a
 *   proxy for anything.
 *
 *   Phases A and D produced nothing: ERR_NVGPUCTRPERM. The 399 harness had
 *   the ncu permission probe and its sudo fallback abbreviated away to a
 *   bare command -v. That is a harness defect, not a machine problem; the
 *   same probe worked in 398-r2. So U5 -- the dominant stall reason -- is
 *   STILL unanswered, at the third attempt.
 *
 * WHY THE LOST MEASUREMENT IS NOW WORTH MORE
 *   The 35.2% swing is a calibration point. 398-r2 measured 10.53 of 32
 *   active threads per warp on the 394f order. If time were inversely
 *   proportional to lane utilisation, the shuffled order should show
 *     10.53 * 16,482.7 / 22,285.5 = 7.79 of 32.
 *   Measuring it puts a slope on the axis: how much time one more live lane
 *   is worth. Without that number, "lane utilisation is 32.9%" is a
 *   diagnosis with no price on it.
 *
 * AND A CHEAPER COST PROXY THAN GUESSWORK
 *   399's orderings sorted on popcount(free), which is a guess at the work
 *   per record. With stride >= records -- NQ_MAX_BLOCKS = 42784 at N=20 --
 *   each thread takes exactly one record and the output file becomes the
 *   per-RECORD solution count instead of a per-thread sum. That is a
 *   measured quantity, and sorting on it gives per-thread balance and
 *   intra-warp coherence at the same time: rank r lands on thread
 *   r mod stride, so every thread gets one record from each rank band and
 *   every warp gets 32 adjacent ranks.
 *
 * rev399 -- NO code change. Byte-identical to 398_r2_kernel_maxd14.cu below
 * the first #include; kernel region still the 395a r2 sha256 ebd7f523...
 *
 * WHAT 398-r2 MEASURED, on N=20 and N=19 with their own 394f schedules
 *   Avg. Active Threads Per Warp   10.53 / 32   (N=19: 10.25)
 *   sectors per local request       4.690 ld / 4.691 st  (N=19: 4.588)
 *   L1/TEX Cache Throughput        70.64%   <- the top pipe
 *   Compute (SM) Throughput        31.55%
 *   No Eligible                    60.53%   (N=21 measured 61.63%)
 *   Active Warps Per Scheduler      2.43    (N=21 measured 2.41)
 *   Warp Cycles Per Issued Instr    6.14
 *
 *   Two multiplicative losses on every local-memory access:
 *     lane utilisation      10.53/32 = 32.9%
 *     sector efficiency     84.2 B useful / 150.1 B moved = 56.1%
 *     combined                                              18.5%
 *   The kernel is L1-bound and the L1 traffic is the DFS stack in local
 *   memory: 33.7 TB of sector traffic in 16.4 s, about 2.05 TB/s, and
 *   164,000 local accesses per record.
 *
 *   The (a) volume vs (b) divergence fork posed by 397 was a false
 *   dichotomy. Both are the same wall seen from two sides: fewer bytes per
 *   frame would cut sectors roughly proportionally, and more live lanes
 *   would cut the waste per sector. 397's idea was sound; only its masking
 *   of rd was invalid.
 *
 * WHY LANE UTILISATION IS THE TARGET
 *   32.9% is the largest single inefficiency this project has measured. The
 *   likely mechanism is task-length divergence rather than branch-shape
 *   divergence: each thread runs process_one_task() to completion before
 *   taking its next record, so a warp's loop lasts as long as its longest
 *   lane and the rest idle.
 *   Which 32 records share a warp is decided entirely on the HOST. Thread
 *   tid takes records tid, tid+stride, tid+2*stride, ..., so at step k the
 *   32 lanes of warp w hold records [k*stride + w*32 .. +31] -- thirty-two
 *   CONSECUTIVE records in the scheduled file. Grouping is a property of
 *   the ordering, and the ordering is 394f's output.
 *   Record order is also safe to change: records are independent tasks and
 *   the result is their sum, so any permutation must produce the same
 *   total_sum. 399 exploits that -- every ordering it tries is still gated
 *   on the oracle.
 *
 * rev398-r2 -- NO code change. Byte-identical to 398_kernel_maxd14.cu below
 * the first #include; kernel region still the 395a r2 sha256 ebd7f523...
 *
 * WHY r2: 398's profiling plan needed the CRunner to run at N=20 and N=19,
 * and it does not. The dispatcher gates the mode-37 CRunner path on
 * "N >= 21", so -g 20 20 and -g 19 19 took the Codon GPU path instead --
 * correct answers (39,029,188,884 and 4,968,057,848, both ok) but no
 * CRunner log, no scheduled input, and nothing to profile. The first two
 * stages that build that input (constellation stream, soa_ref dump) live
 * inside the Codon binary, so no external script can produce it either.
 * r2 lowers that one bound. The .cu is untouched.
 *
 * rev398 -- NO code change. Byte-identical to 396_r2_kernel_maxd14.cu below
 * the first #include; the kernel region is back to the 395a r2 sha256
 * ebd7f523... 397 is DISCARDED: see below.
 *
 * WHY 397 FAILED, and what it taught
 *   397 repacked the DFS stack frame from 16 bytes to 12 by holding ld, rd
 *   and col as three 21-bit fields. ptxas confirmed the frame shrank
 *   208 -> 160 bytes with no new spills (37 -> 38 registers). But the
 *   equivalence gate -- both binaries over the full 2,025,282-record input,
 *   per-record output compared byte for byte -- rejected it before any
 *   timing was read: 397 produced total_sum 427,786,151,034 against the
 *   oracle 314,666,222,712.
 *
 *   The reason is structural and worth keeping. Every use of cur_ld is a
 *   LEFT shift and every use of cur_rd is a RIGHT shift:
 *     cur_ld : lines 611, 614, 733, 747   all <<
 *     cur_rd : lines 612, 615, 734, 748   all >>
 *   So bits of ld above the board mask move further up and are dead -- every
 *   use site covers them with bm. Bits of rd above the board mask move DOWN
 *   as the search descends and become live constraints several levels later.
 *   Masking rd drops constraints, which is exactly why 397 counted more
 *   solutions than exist.
 *
 *   With the true widths -- ld 20, rd 32, col 21, avail 21, depth 4 -- a
 *   frame needs 98 bits. Twelve bytes is 96. The packing route to testing
 *   the volume hypothesis is two bits short, and 398 does not pursue it.
 *
 * WHAT 398 DOES
 *   Measures instead of building. 397 was written before the diagnosis was
 *   complete, which is the mistake this project otherwise avoids. 398 goes
 *   back to the measurement that was deferred twice:
 *     - U5 is still unanswered. WarpStateStats has never once completed.
 *     - The (a) volume vs (b) divergence fork is still open, and it decides
 *       whether any packing work is worth doing at all.
 *   Both are answered by profiling a workload that ncu can actually finish.
 *   396 showed ncu succeeds on a 1.5 s kernel and fails on a 191 s one, so
 *   398 profiles N=20 (about 16 s) and N=19 (about 2 s), each with its own
 *   394f schedule and therefore balanced by construction -- unlike the
 *   record subsets that 396 and 396-r2 both had to reject.
 *
 * A CORRECTION CARRIED INTO THIS FILE
 *   Earlier notes in 397 said N=22/23 would need maxd16. That is wrong.
 *   select_static_maxd() returns 14 for required_maxd <= 14, and the rev372
 *   notes record that N=22 also has required_maxd=14, confirmed 3/3 in 366.
 *   N=22 runs on THIS kernel. N=23 has not been checked. So a packing that
 *   caps N at 21 would block the next target, not some distant one.
 *
 * rev396-r2 -- NO code change at all. Byte-identical to 396_kernel_maxd14.cu
 * below the first #include, gated by a sha256 of the whole code region.
 *
 * WHY r2 EXISTS: 396's proxy calibration (T1) FAILED, exactly as it was
 * designed to. Truncating the scheduled input to its first k rounds gave
 * 48.9-57.5 us/record against the full run's 65.765 -- 13% to 26% too
 * cheap, and non-monotonic in k. Differencing the prefix sums shows why:
 *   round 0        1471.2 ms      rounds 4-7     1577.7 ms/round
 *   round 1        1159.9 ms      rounds 8-15    1243.1 ms/round
 *   rounds 2-3     1188.1 ms      rounds 16-79   1773.5 ms/round
 *   global mean 1683.6 ms/round
 * The 394f schedule orders records, so the head of the file is the cheap
 * part and the tail carries the cost. A prefix is not a sample.
 *
 * r2 replaces truncation with STRATIFIED ROUND SAMPLING: take k rounds
 * spread evenly over all 79, not the first k. A round is 25,600 records --
 * exactly one per thread at stride 25,600 -- so sampling whole rounds
 * preserves the record-to-thread mapping and therefore the within-warp
 * divergence pattern exactly; only the trip count changes.
 *
 * r2 also adds a fallback: if stratified sampling still fails calibration,
 * the harness profiles the REAL full input one section at a time under a
 * time budget, starting with WarpStateStats. The deliverable of 396 is the
 * stall ranking, and it should not be blocked by a proxy question.
 *
 * rev396 -- NO code change at all. Byte-identical to
 * 395c_r6_kernel_maxd14.cu below the first #include; only this header note
 * differs, gated by a sha256 of the whole code region. Renamed so that the
 * four 396 artifacts carry one revision number.
 *
 * WHY 396 EXISTS
 *   395c closed with the production path at 133,192.071 ms (2:13.19), a
 *   400 ms / 0.299% gain adopted in r6 and verified by an alternating
 *   within-session A/B. More important than the gain: the noise floor is
 *   now 0.009-0.029% within a session and 0.005% ACROSS sessions. Every
 *   earlier optimisation decision on this kernel was made with a ruler
 *   somewhere between 0.3% and 1.6%.
 *
 *   So the bottleneck picture itself is due a re-measurement. 396 profiles
 *   the CURRENT kernel with ncu and produces a ranked stall breakdown. It
 *   changes nothing and is expected to speed nothing up; its output is the
 *   ranked list that decides what 397 attempts.
 *
 * TWO THINGS 396 DOES BEFORE IT PROFILES
 *   1. Calibrates the proxy. An N=21 full run is 133 s, far too long to
 *      replay under ncu, so the profile uses a truncated record set. But
 *      truncating changes k_per_thread (production: 2,025,282 / 25,600 =
 *      79.1 records per thread), and with it the loop and divergence
 *      behaviour. The harness sweeps k and picks the smallest one whose
 *      microseconds-per-record match the full run within 1%. A proxy that
 *      fails that test is not used.
 *   2. Measures the cost of measuring. Section count drives ncu replay
 *      count, which is not predictable from the section list. A one-section
 *      pilot runs first and the total is extrapolated from it; if the
 *      projection exceeds the budget the harness drops to a smaller proxy
 *      and says so, rather than hanging.
 *
 * A CAVEAT WORTH RECORDING
 *   ncu itself holds a CUDA context and device memory. 395c established
 *   that context count and device occupancy move this kernel by 0.3% to
 *   10%, so a profiled run is NOT in the production memory state and its
 *   wall time must not be compared with production timings. The counters
 *   and ratios are per-kernel and are what 396 reads; the wall clock under
 *   ncu is not evidence about anything.
 *
 * rev395c-r6 -- NO code change at all. Byte-identical to
 * 395c_r4_kernel_maxd14.cu below the first #include; only this header note
 * differs, and the harness gates that with a sha256 of the whole code
 * region (not just the kernel). Renamed so that the four r6 artifacts carry
 * one revision number, and because the r6 dispatch table now names this
 * binary with a non-empty env_prefix.
 *
 * WHY r6 EXISTS
 *   395c-r5 measured, on the real -g path and with all six pre-registered
 *   predictions holding:
 *     G0  -g 21 21, nothing extra          133,585.359  free_mb 22,018
 *     Gh  + one idle holder process        133,179.047  free_mb 21,762
 *     Gx  + NQ_EXTRA_CTX=1                 133,206.172  free_mb 21,762
 *     Gxb replicate of Gx                  133,172.016  free_mb 21,762
 *   |Gx - Gh| = 0.020%: a context in another process and a context in ours
 *   are indistinguishable on the production path too. The gain over G0 is
 *   0.309%, i.e. 413 ms, and the claim rule fixed before the run
 *   (>=0.20% and replicated within 0.05%) was met.
 *
 *   r6 turns that into the shipped default by putting NQ_EXTRA_CTX=1 in the
 *   dispatch table's env_prefix -- in the table, not in this binary's
 *   default, so that every run records which state it was in via
 *   [crunner-config] env_prefix= and dispatch.log. NQ_EXTRA_CTX stays 0 when
 *   unset, so a direct run is unchanged and stays usable as a control.
 *
 * KNOWN FRAGILITY, recorded deliberately
 *   r5 also swept the neighbourhood and found the fast window is a NARROW
 *   NOTCH, not a basin:
 *     +0 MB 133,561 | +128 MB 133,993 | +255 MB 133,181 | +384 MB 134,074
 *     | +512 MB 133,854 | +1024 MB 134,310
 *   Both neighbours of the optimum are slower than adding nothing at all,
 *   and the window is under 128 MiB wide. What lands us in it is that one
 *   CUDA context happens to cost 255 MiB on this driver. So this 413 ms is
 *   a property of THIS configuration, not a principle:
 *     - a driver update that changes per-context overhead moves the notch;
 *     - ANY change to this binary's own device footprint moves us out of it.
 *   maxd16 for N=22/23 will allocate substantially more, so the notch must
 *   be re-measured there before any of this is assumed to carry over.
 *
 * rev395c-r4 -- IDENTICAL to 395c_r3_kernel_maxd14.cu except for the
 * NQ_EXTRA_CTX treatment (search for EXTRA CONTEXTS below), which is inert
 * unless NQ_EXTRA_CTX is set to a positive value. Needs -lcuda at link time
 * (driver API). The __global__ kernel and process_one_task() remain
 * byte-identical to 395a r2 / 395c / r2 / r3.
 *
 * rev395c-r3 -- IDENTICAL to 395c_r2_kernel_maxd14.cu except for the
 * NQ_PAD_MB self-pad (search for SELF-PAD below), which is inert unless
 * NQ_PAD_MB is set to a positive value. The __global__ kernel and
 * process_one_task() remain byte-identical to 395a r2 / 395c / r2.
 *
 * rev395c-r2 -- IDENTICAL to 395c_kernel_maxd14.cu except for one added
 * host-side diagnostic block ("[gpu-mem]" / "[gpu-mem-base]", search for
 * PLACEMENT PROBE below). The __global__ kernel and process_one_task()
 * are byte-identical to 395c, which is byte-identical to 395a r2; the
 * harness gates this with a sha256 of the kernel region. The probe runs
 * after every cudaMalloc and before ev_h2d_start, allocates nothing, and
 * therefore changes neither placement nor any timed region. Its purpose
 * is to let the D0/D1/D2 cells be distinguished by device ADDRESS, not
 * only by elapsed time.
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
 *   /usr/local/cuda/bin/nvcc -O3 -arch=sm_86 -o 395c_kernel_maxd14 395c_kernel_maxd14.cu
 * Run (device):
 *   ./395c_kernel_maxd14 <N> <in_soa7_bin> <out_results_bin> [expected_total]
 *   e.g. ./395c_kernel_maxd14 21 constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin \
 *          /tmp/gpu_results.bin 314666222712
 *   395a: NQ_MAX_BLOCKS=800 ./395c_kernel_maxd14 21 <in> <out> 314666222712
 * Build (host-only CPU test, no CUDA toolkit needed, this sandbox):
 *   gcc -O2 -Wall -Wextra -x c -o 395c_kernel_maxd14_cputest 395c_kernel_maxd14.cu -lm
 * Build (host-only CPU test, OpenMP-parallelized outer loop):
 *   gcc -O2 -fopenmp -Wall -Wextra -x c -o 395c_kernel_maxd14_cputest_omp 395c_kernel_maxd14.cu -lm
 *   (394b: "-x c" moved BEFORE the source file in both gcc lines -- the
 *   389 header's order fails on gcc 13.3.0 with "file format not
 *   recognized"; noticed and recorded in 392, fixed in the header here.)
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
 * would exceed the 26-slot (13-ancestor) array; (2) a 1,000,000-
 * iteration hard cap on the schedule-precompute loop and a
 * 50,000,000-iteration cap on the main DFS loop, each aborting with
 * full diagnostic state if hit; (3) wall-clock timing (elapsed/rate)
 * on the progress heartbeat, plus a per-record slow-record warning.
 * None of this changes process_one_task()'s actual arithmetic or
 * control flow on the non-error path. It turned out the real N=21
 * hang was NOT a bug at all -- the schedule-precompute cap never
 * fired (confirmed via 363_filter_maxd14_only.py: every real N=21
 * record is exactly depth=14, none dropped), and the main-loop cap
 * never fired either: it was genuinely still running, just extremely
 * slowly, because a single CPU thread doing serially what the GPU
 * does across 15,488 parallel threads is fundamentally ~4-5 orders of
 * magnitude slower for this workload. r2 also adds an optional OpenMP
 * build (#ifdef _OPENMP) parallelizing the CPU-test harness's outer
 * per-record loop (process_one_task() itself is untouched) as a
 * partial mitigation, purely for faster CPU-side validation runs --
 * this has no bearing on the real kernel's behavior or performance.
 *
 * 364 ADDITION: a host-side GPU runner (main(), #ifdef __CUDACC__
 * only) that uploads a SoA7 input file (same 7-field format the CPU
 * test harness and 363_filter_maxd14_only.py use) plus META_NEXT to
 * device memory, launches kernel_dfs_iter_gpu_maxd14 with the
 * unchanged production 32x484 grid/block config, downloads results,
 * sums them, times each phase (H2D/kernel/D2H) via cudaEvent, and
 * optionally checks the total against an expected value. This is a
 * single-shot, non-chunked run: it establishes real-hardware
 * correctness first; wiring it into the exact 3-chunk measure2
 * protocol for a true +-3% comparison against the 356 anchor
 * (393.404s) is deferred to a later revision once correctness here is
 * confirmed. Nothing outside this new host-side main() differs from
 * 363 (r2) -- process_one_task() and the __global__ kernel itself are
 * byte-identical.
 *
 * 388 RENAME (pure rename, zero code change): per Suzuki's explicit
 * request, CRunner binaries referenced by crunner_dispatch_table()
 * (385Py onward) should carry the CURRENT revision's number rather
 * than a past revision's, so each revision stays self-contained
 * instead of silently depending on an old file by reference. This
 * file is renamed from 364_kernel_maxd14.cu to 388_kernel_maxd14.cu
 * with ONLY the header's illustrative build/run command examples
 * updated to match (5 occurrences, all in this comment block, lines
 * 42-50) -- everything from the first #include onward, including
 * process_one_task() and the __global__ kernel, is confirmed
 * byte-identical to 364_kernel_maxd14.cu by direct diff. Real-hardware
 * confirmed 2026-09-07: nvcc build under this filename succeeded,
 * N=21 total=314666222712 kernel_ms~=201237, matching 364's own
 * anchor.
 *
 * 389 RENAME (pure rename, zero code change, same reasoning as 388's
 * own rename note directly above -- this is now the standing policy
 * for every future revision, not a one-off): renamed from
 * 388_kernel_maxd14.cu to 389_kernel_maxd14.cu. Only the header's
 * illustrative build/run command examples were updated (the same 5
 * occurrences), plus this paragraph. Code region confirmed byte-
 * identical to 388_kernel_maxd14.cu (and therefore to 364's) by direct
 * diff. NOT YET real-hardware built under this filename.
 *
 * 394b CHANGE (host-side launch configuration only; kernel byte-
 * identical): derived from 389_kernel_maxd14.cu. Three edits, all in
 * main() under #ifdef __CUDACC__ -- process_one_task() and the
 * __global__ kernel are untouched and compile to the same 472 SASS
 * instructions 393 stage3 / 394a observed:
 *   (1) MAX_BLOCKS becomes an int initialised to 484 and overridable
 *       at run time via the environment variable NQ_MAX_BLOCKS
 *       (range-checked, [1,65535]; unset or empty = 484 = 389 behaviour
 *       exactly). stride = BLOCK*MAX_BLOCKS follows, as before. stride
 *       was ALREADY a kernel argument, so no kernel change is needed.
 *   (2) a "[gpu-config] BLOCK=.. MAX_BLOCKS=.. stride=.. threads=..
 *       k_per_thread_max=.." line is printed to stderr right after
 *       "[gpu-run]", so every log is self-describing.
 *   (3) "block= max_blocks= stride=" are APPENDED to the end of the
 *       "[gpu-run-done]" stdout line. The existing total_sum= /
 *       kernel_ms= fields are unchanged and stay in place, so the
 *       Codon side's key=value parser (crunner_parse_result) is
 *       unaffected. 394b's sweep harness drives this binary directly
 *       and does not go through the dispatcher at all.
 * Why: 393-9. ncu measured Active Warps Per Scheduler = 1.51 against a
 * hardware maximum of 12 and a 4-warp ceiling for this BLOCK=32 config.
 * 1.51 = 484 blocks / (80 SMs x 4 schedulers) exactly -- the grid is
 * simply too small to occupy the machine. 484x32 dates from before
 * K-batching (292) decoupled grid size from chunk size; only K has been
 * swept since. 394b sweeps MAX_BLOCKS in {484, 968, 1280, 1936} with
 * K=48 untouched. 1280x32 = 40,960 threads is the L1-footprint-limited
 * target (16 warps/SM x 32 x 208 B = 104 KB of 128 KB) and happens to
 * make 40,960 x 48 = 1,966,080 ~= all of N=21 in one launch.
 * Real-hardware confirmed 2026-09-08 (394b r1/r2): default config
 * reproduces 389 (201,231 / 201,237 ms, oracle MATCH); NQ_MAX_BLOCKS=968
 * gives 163,060..163,197 ms (-18.9%, three runs within 0.08%).
 *
 * 394c RENAME (header only, zero code change): 394b_kernel_maxd14.cu ->
 * 394c_kernel_maxd14.cu. The code region is byte-identical to 394b's
 * (and the kernel region to 389's). 394c's change is on the Codon side
 * (bench_mode=39 feeds this binary the chunkshape148-REORDERED input and
 * sets NQ_MAX_BLOCKS to the shaping stride); nothing here needed to
 * move. Real-hardware confirmed 2026-09-08 (394c factorial): A=201,236
 * (raw, 484), B=261,060 (reordered, 484, +29.7%), C=160,528 (raw, 968),
 * D=158,955 (reordered-without-iter_sort, 968). All four oracle MATCH.
 *
 * 394d RENAME (header only, zero code change): 394c_kernel_maxd14.cu ->
 * 394d_kernel_maxd14.cu. Code region byte-identical to 394c/394b, kernel
 * region to 389. 394d's change is on the Codon side (bench_mode=39 gains
 * bucket_run / iter_sort / input_stage knobs to decompose cell B).
 * Real-hardware confirmed 2026-09-08 (394d ladder, 484): raw 201,238;
 * base-only 296,902 (+47.5%); +scorestripe 295,850; +bucket_run 288,149;
 * +iter_sort=1 258,831; +isort9 261,063 (= 394c B to 0.001%). All MATCH.
 *
 * 394e RENAME (header only, zero code change): 394d_kernel_maxd14.cu ->
 * 394e_kernel_maxd14.cu. Code region byte-identical to 394d/394c/394b,
 * kernel region to 389. 394e drives this binary directly at
 * NQ_MAX_BLOCKS 483/484/485 on inputs shaped for 484, to separate an
 * intra-warp mechanism from an inter-thread (launch-tail) one.
 * Real-hardware confirmed 2026-09-08 (394e): raw 201,234 / 200,061 at
 * 484/485; base 296,899 / 282,374 / 290,851 at 484/485/483 -> the
 * penalty survives the stride shift: intra-warp (M1).
 *
 * 394f RENAME (header only, zero code change): 394e_kernel_maxd14.cu ->
 * 394f_kernel_maxd14.cu. Code region byte-identical to 394e..394b, kernel
 * region to 389. 394f drives this binary on permutations of the raw
 * input (random / state-sorted / free-sorted / schedule-sorted).
 * Real-hardware confirmed 2026-09-08 (394f, 484): raw 201,238; random
 * +44.2%; (col,ld,rd)-sorted +20.8%; free-count-sorted +21.2%;
 * schedule-sorted (stable) -1.71% -- the first order to beat raw.
 *
 * 394g RENAME (header only, zero code change): 394f_kernel_maxd14.cu ->
 * 394g_kernel_maxd14.cu. Code region byte-identical to 394f..394b,
 * kernel region to 389. 394g sweeps MAX_BLOCKS 704/800/864/928 (+484/968
 * anchors) on raw and on the L3 input.
 * Real-hardware confirmed 2026-09-08 (394g): raw 484 201,240 | 704
 * 153,619 | 800 146,585 | 864 158,390 | 928 151,444 | 968 163,133; L3 at
 * 800 = 139,404 (-30.7% vs the 389 anchor). All MATCH.
 *
 * 395a -- FIRST KERNEL CHANGE SINCE 364. process_one_task() main loop:
 * the nibble is split into `block_code = nibble_op & 7u` and
 * `fc_flag = nibble_op & 8u` immediately after extraction, and those two
 * scalars are what the block_code branch and the future check consume.
 * Nothing else in the kernel moves. Motivation and the evidence are in
 * the comment at the change site and in 394a_README_append.md (the
 * duplicated extraction carried 4.09% of samples). Semantics are
 * identical: both consumers read the same bits of the same nibble.
 * r1 RESULT (2026-09-08, real hardware): oracle MATCH, CPU-harness
 * per-record results identical to 394g on 2,048 real records, but
 * kernel_ms 139,527 / 139,542 vs the 394g control's 139,443 / 139,396
 * (+0.07%): the hoist alone did NOT change what nvcc emits.
 * r2: the schedule is materialised once per task as a 64-bit value
 * (schedule64, loop-invariant) and the main loop extracts its nibble
 * with a single funnel shift + AND, replacing the cur_depth<8 test, two
 * shifts and a select. This removes the predicated pair regardless of
 * the compiler's rematerialisation choice. Semantics unchanged
 * (schedule_depth <= 14 -> shift amount <= 52). The root fast path's
 * `schedule_lo & 15u` (depth 0) is untouched. Push guard untouched.
 * 395a_validate.sh checks the SASS with ncu when sudo is available and
 * always checks the N=21 oracle.
 * r2 RESULT (2026-09-08): oracle MATCH; SASS still 472; the cur_depth<8
 * predication is gone but nvcc still rematerialises the extraction
 * (2 x SHF.R.U64); -0.2% vs the 394g control. Safe, neutral, kept.
 *
 * 395b (REVERTED here) tried a register-resident top of the explicit
 * stack: +3.5% at 800x32 (144,032 vs 139,166 direct), +5.2% at 484. The
 * two extra `if (save_sp != 0u)` guards around push/pop are control-flow
 * shape changes in the hottest region -- the same class that failed in
 * 240, 266-269, 273, 326, 358 and 359. Seven for seven: kernel-local
 * control-flow edits are CLOSED as an axis for this kernel.
 *
 * 395c: kernel region is byte-identical to 395a r2 (the last confirmed-
 * safe kernel). Host side keeps the 395b policy default (NQ_MAX_BLOCKS
 * default 800). 395c's measurement is not about the kernel at all: it
 * isolates why the same binary on the same file runs ~4% faster when
 * launched by the Codon dispatcher than directly (395a: 133,580 vs
 * 139,179; 395b: 138,650 vs 144,032) -- see 395c_validate.sh.
 * NOT YET real-hardware built under this filename.
 * Also carries the 394b/394c/394f/394g production parameters, which live
 * on the Codon side (395aPy: MAX_BLOCKS 800 by default, passed via
 * NQ_MAX_BLOCKS; input in the L3 schedule-sorted order). This binary's
 * own NQ_MAX_BLOCKS default stays 484 for direct-invocation continuity.
 * NOT YET real-hardware built under this filename.
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
/* 403: the 12-byte frame for N <= 22, using two provably dead bits.
 *   ld's bit N-1 is dead: the saved ld is only ever used as (ld|bit) << step
 *   with step >= 1, then masked with bm, so bit N-1 leaves the board. ld
 *   therefore needs N-1 bits (<= 21).
 *   rd's bit 0 is dead: the saved rd is only ever used as (rd|bit) >> step
 *   with step >= 1, so bit 0 is shifted out. rd therefore needs 31 bits.
 *   403-r5 ("403b" spelling of the same 96-bit budget):
 *   word A (u64) = col(22) | avail(22) << 22 | ld[1..20] << 44   (64 bits)
 *   word B (u32) = (rd & ~1) | ld[0]                              (32 bits)
 *   (403 .. 403-r3 held ld[0..19] in A and ld[20] in B's bit 0; the split
 *   at the TOP of ld cost ld>>20, &1, and a <<20 on pop. Splitting at the
 *   BOTTOM lets every move be a shift the compiler already emits plus one
 *   LOP3 bit-select, and cur_rd is read back unmasked: its bit 0 carries
 *   ld[0], which is dead for rd.)
 *   22 + 22 + 21 + 31 = 96 bits = 12 bytes, valid for N <= 22. (402 used
 *   3 x 21 bits and was N <= 21 only.) Both main()s refuse N > 22 right after
 *   parsing N, before opening any file. Both dead-bit claims are gated by the
 *   harness: CPU per-record equality at N=21 and N=22 and the full-input
 *   per-thread equality at N=21, before any timing is read. */
#define PACK403_WIDTH 22
#define PACK403_MASK  ((1u << PACK403_WIDTH) - 1u)     /* col, avail: 22 bits */
/* 403-r5: PACK403_LDLO (ld bits 0..19 in word A) is gone -- word A now holds
 * ld >> 1 (bits 1..20) and word B holds ld bit 0 in rd's dead bit 0. */
/* 403-r2: BOTH 12-byte layouts live in this file and are selected per N by
 * the host (N <= 21 -> 402 layout, N = 22 -> 403 layout). process_one_task
 * takes the layout as a parameter; every caller passes a compile-time
 * constant (the __global__ kernel is a template instantiated twice, the
 * CPU harness passes the host's choice), and process_one_task is force-
 * inlined, so each instantiation contains ONE layout and no runtime branch.
 *   402 layout: word A = ld[0..20] | col << 21 | avail << 42 ; word B = rd.
 *   403 layout: word A = col | avail << 22 | ld[1..20] << 44 ; word B =
 *               (rd & ~1) | ld[0]   (403-r5 spelling; see the 403 note above).
 * NQ_LAYOUT=402|403|auto (default auto) overrides the host's choice; 402
 * with N = 22 is refused (rc=3). */
#define PACK_LAYOUT_402 402
#define PACK_LAYOUT_403 403
#define PACK402_WIDTH 21
#define PACK402_MASK  ((1u << PACK402_WIDTH) - 1u)     /* ld, col, avail: 21 bits */
#ifdef __CUDACC__
#define POT_FORCEINLINE __forceinline__
#else
#define POT_FORCEINLINE inline __attribute__((always_inline))
#endif
/* 403-r2d: the layout is a TEMPLATE parameter of process_one_task and the
 * three stack sites use `if constexpr`, so the discarded layout is not even
 * instantiated: the 402 instantiation's AST is 402_r5's (no dead branch for
 * the front end to fold, which is what re-ordered the PTX in 403-r2/r2c).
 * The plain-C CPU harness (gcc -x c) has no templates: there the layout is a
 * runtime parameter and a plain `if`, exactly as in 403-r2. */
#ifdef __cplusplus
#define POT_TEMPLATE      template <int layout>
#define POT_LAYOUT_PARAM
#define POT_TARGS(L)      <L>
#define POT_LAYOUT_ARG(L)
#define LAYOUT_IF(c)      if constexpr (c)
#else
#define POT_TEMPLATE
#define POT_LAYOUT_PARAM  , const int layout
#define POT_TARGS(L)
#define POT_LAYOUT_ARG(L) , L
#define LAYOUT_IF(c)      if (c)
#endif

/* ---------------------------------------------------------------------
 * process_one_task -- the per-constellation body (720-1028 of the
 * Codon source), factored out so it is callable from both the real
 * __global__ kernel (device) and a CPU test harness (host). Returns
 * the task's contribution to thread_total, i.e. total*w_lo, exactly
 * matching what the Codon kernel accumulates per idx before idx+=stride.
 * ------------------------------------------------------------------- */
POT_TEMPLATE
HOSTDEV POT_FORCEINLINE
static uint64_t process_one_task(
    uint32_t root_ld, uint32_t root_rd, uint32_t root_col,
    uint32_t root_a_in, uint32_t ctrl0, uint32_t markctrl, uint32_t w_lo,
    const uint8_t* __restrict__ meta_next,
    uint32_t bm, uint32_t n3, uint32_t n4
    POT_LAYOUT_PARAM   /* 403-r2d: template parameter (C++/CUDA) or runtime int (plain C) */
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

    /* 395a r2: the schedule as ONE loop-invariant 64-bit value. The main
     * loop below extracts its nibble with a single funnel shift instead of
     * a cur_depth<8 test, two shifts and a select (schedule_depth <= 14,
     * so cur_depth*4 <= 52 < 64 -- the shift is always defined). */
    const uint64_t schedule64 = ((uint64_t)schedule_hi << 32) | (uint64_t)schedule_lo;

    int      stack_ptr = 0;   /* 402: frame index (one frame = one entry in each array) */
    int      cur_depth = 0;
    uint32_t cur_ld = root_ld;
    uint32_t cur_rd = root_rd;
    uint32_t cur_col = root_col;
    uint32_t cur_avail = root_a;

    /* 402: 12-byte frame. Word A packs ld | col<<21 | avail<<42 (3 x 21 bits;
     * ld's bits above the board are dead -- every use shifts them further
     * up and masks with bm -- so they are dropped, exactly as 398 showed).
     * rd keeps its full 32 bits in word B: its high bits come DOWN and act
     * as constraints (the 397 mistake). depth leaves local memory entirely:
     * 4 bits per frame in ONE register, LIFO (13 x 4 = 52 bits). Frame
     * 16 B -> 12 B; local accesses stay 2 per push and 2 per pop. */
    uint64_t stack_a[MAXD14_ANCESTOR];
    uint32_t stack_b[MAXD14_ANCESTOR];
    const int depth_base = (int)__builtin_popcount(root_col);

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
                if (stack_ptr >= MAXD14_ANCESTOR) {
                    fprintf(stderr, "[STACK-OVERFLOW] debug_idx=%lld stack_ptr=%d cur_depth=%d "
                            "(root fast path) ld=%u rd=%u col=%u ctrl0=%u markctrl=%u\n",
                            (long long)debug_idx, stack_ptr, cur_depth, root_ld, root_rd, root_col, ctrl0, markctrl);
                    fflush(stderr);
                    abort();
                }
#endif
                LAYOUT_IF (layout == PACK_LAYOUT_402) {
                    stack_a[stack_ptr] = (uint64_t)(cur_ld & PACK402_MASK)
                                        | ((uint64_t)cur_col << 21) | ((uint64_t)cur_avail << 42);
                    stack_b[stack_ptr] = cur_rd;
                } else {
                    stack_a[stack_ptr] = (uint64_t)cur_col | ((uint64_t)cur_avail << 22)
                                        | ((uint64_t)((cur_ld << 11) & 0xFFFFF000u) << 32);
                    /* 403-r7: B in one LOP3 (0xD8 = c ? b : a, c = 1); A's ld part as shl + masked OR. */
                    { uint32_t pb403;
#ifdef __CUDACC__
                      asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(pb403) : "r"(cur_rd), "r"(cur_ld), "r"(1u));
#else
                      pb403 = (cur_rd & ~1u) | (cur_ld & 1u);
#endif
                      stack_b[stack_ptr] = pb403; }
                }
                stack_ptr += 1;
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
                    (long long)debug_idx, stack_ptr, (unsigned)stack_ptr, cur_depth, cur_avail, terminal_depth,
                    schedule_lo, schedule_hi, root_ld, root_rd, root_col, ctrl0, markctrl);
            fflush(stderr);
            abort();
        }
#endif
        if (cur_avail == 0u) {
            /* 404 (A): save_sp was always == stack_ptr; one counter is enough. */
            if (stack_ptr == 0) {
                break;
            }
            stack_ptr -= 1;
            uint64_t packed_a = stack_a[stack_ptr];
            LAYOUT_IF (layout == PACK_LAYOUT_402) {
                cur_ld  = (uint32_t)packed_a & PACK402_MASK;
                cur_col = (uint32_t)(packed_a >> 21) & PACK402_MASK;
                cur_avail = (uint32_t)(packed_a >> 42) & bm;
                cur_rd  = stack_b[stack_ptr];
            } else {
                uint32_t packed_b = stack_b[stack_ptr];
                cur_col = (uint32_t)packed_a & PACK403_MASK;
                cur_avail = (uint32_t)(packed_a >> 22) & bm;
                { const uint32_t a43 = (uint32_t)(packed_a >> 43);
#ifdef __CUDACC__
                  /* 403-r6: one 3-input LOP3 (immLut 0xD8 = c ? b : a with c = 1):
                   * ptxas does not fold the C spelling (a & ~1) | (b & 1) -- all six
                   * C spellings tried in 403-r6's compile search gave the same two
                   * LOP3s (mask 0x1ffffe, then merge). The asm pins it to one. */
                  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(cur_ld) : "r"(a43), "r"(packed_b), "r"(1u));
#else
                  cur_ld  = (a43 & ~1u) | (packed_b & 1u);
#endif
                }
                cur_rd  = packed_b;
            }
            /* 404 (B): depth is the number of queens placed since the root. */
            cur_depth = (int)__builtin_popcount(cur_col) - depth_base;
            continue;
        }

        /* 395a r1 (hoist only, kept): 394a's -lineinfo SourceCounters showed
         * nvcc rematerialising the nibble extraction (cur_depth<8 test,
         * shift, select: 5 SASS at 2899xx) for the future-check test,
         * because the first extraction's register was overwritten by
         * `nibble_op & 7u`. That duplicate carried 4.09% of all samples.
         * r1 split the nibble into block_code / fc_flag once, here -- and
         * measured +0.07%: the compiler still rematerialised.
         * 395a r2: make the extraction itself branchless and cheap -- one
         * funnel shift of the loop-invariant schedule64 plus an AND -- so
         * that whether or not the compiler chooses to recompute it, the
         * cur_depth<8 predication and the second shift are gone. */
        const uint32_t nibble_op  = (uint32_t)(schedule64 >> (cur_depth * 4)) & 15u;
        const uint32_t block_code = nibble_op & 7u;
        const uint32_t fc_flag    = nibble_op & 8u;
        uint32_t bit = cur_avail & (0u - cur_avail);
        cur_avail = cur_avail ^ bit;

        uint32_t nld = (cur_ld | bit) << 1;
        uint32_t nrd = (cur_rd | bit) >> 1;
        uint32_t ncol = cur_col | bit;
        if (block_code != 0u) {
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
            if (fc_flag != 0u) {   /* 395a: was (nibble_op & 8u), see above */
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
            if (stack_ptr >= MAXD14_ANCESTOR) {
                fprintf(stderr, "[STACK-OVERFLOW] debug_idx=%lld stack_ptr=%d cur_depth=%d "
                        "(main loop) ld=%u rd=%u col=%u ctrl0=%u markctrl=%u\n",
                        (long long)debug_idx, stack_ptr, cur_depth, root_ld, root_rd, root_col, ctrl0, markctrl);
                fflush(stderr);
                abort();
            }
#endif
            LAYOUT_IF (layout == PACK_LAYOUT_402) {
                stack_a[stack_ptr] = (uint64_t)(cur_ld & PACK402_MASK)
                                    | ((uint64_t)cur_col << 21) | ((uint64_t)cur_avail << 42);
                stack_b[stack_ptr] = cur_rd;
            } else {
                stack_a[stack_ptr] = (uint64_t)cur_col | ((uint64_t)cur_avail << 22)
                                    | ((uint64_t)((cur_ld << 11) & 0xFFFFF000u) << 32);
                /* 403-r7: B in one LOP3 (0xD8 = c ? b : a, c = 1); A's ld part as shl + masked OR. */
                { uint32_t pb403;
#ifdef __CUDACC__
                  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(pb403) : "r"(cur_rd), "r"(cur_ld), "r"(1u));
#else
                  pb403 = (cur_rd & ~1u) | (cur_ld & 1u);
#endif
                  stack_b[stack_ptr] = pb403; }
            }
            stack_ptr += 1;
        }
        cur_ld = nld;
        cur_rd = nrd;
        cur_col = ncol;
        cur_avail = nf;
        cur_depth = next_depth;
    }

    return total * (uint64_t)w_lo;
}

/* 403-r2: host-side layout selection, shared by the GPU runner and the CPU
 * harness. Returns PACK_LAYOUT_402 / PACK_LAYOUT_403, or -1 with a message
 * on stderr when the request cannot be honoured. Printed as
 * "[layout] N=.. requested=.. layout=.." so every log records which
 * layout ran. */
static int select_pack_layout(int64_t N, const char *tag) {
    const char *env_l = getenv("NQ_LAYOUT");
    const char *req = (env_l != NULL && env_l[0] != '\0') ? env_l : "auto";
    int layout;
    if (strcmp(req, "auto") == 0) {
        /* 404-r2: with the popc depth the 403 layout is the faster one at every N
         * (col comes out of the pop one op earlier); 402 stays reachable via NQ_LAYOUT=402. */
        layout = PACK_LAYOUT_403;
    } else if (strcmp(req, "402") == 0) {
        layout = PACK_LAYOUT_402;
    } else if (strcmp(req, "403") == 0) {
        layout = PACK_LAYOUT_403;
    } else {
        fprintf(stderr, "ERROR: NQ_LAYOUT='%s' must be auto, 402 or 403\n", req);
        return -1;
    }
    if (layout == PACK_LAYOUT_402 && N > PACK402_WIDTH) {
        fprintf(stderr, "[403-r2-layout] N=%lld unsupported with NQ_LAYOUT=402 (3 x 21 bits, N <= %d only)\n",
                (long long)N, PACK402_WIDTH);
        return -1;
    }
    fprintf(stderr, "[%s-layout] N=%lld requested=%s layout=%d\n", tag, (long long)N, req, layout);
    return layout;
}

#ifdef __CUDACC__
/* ---------------------------------------------------------------------
 * The real GPU kernel. Signature matches 362 spec section 1 exactly.
 * Grid-stride loop over K=ceil(m/stride) constellations per thread,
 * unchanged from the Codon source (292's design).
 * ------------------------------------------------------------------- */
template <int LAYOUT>
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
        thread_total += process_one_task POT_TARGS(LAYOUT) (
            ld_arr[idx], rd_arr[idx], col_arr[idx], root_a,
            ctrl0_arr[idx], markctrl_arr[idx], w_lo_arr[idx],
            meta_next, board_mask, n3, n4 POT_LAYOUT_ARG(LAYOUT)
        );
        idx += stride;
    }
    results[tid] = thread_total;
}

/* ---------------------------------------------------------------------
 * 364: host-side runner (real nvcc build only). Reads the same 7-field
 * SoA input format the CPU test harness uses (ld, rd, col, ctrl0, free,
 * markctrl, w_lo -- produced by 363_filter_maxd14_only.py from 361's
 * dump), uploads it plus META_NEXT to device memory, launches
 * kernel_dfs_iter_gpu_maxd14 with the unchanged production 32x484
 * grid/block config (stride = 484*32 = 15488, matching 292's K-batching
 * design), downloads the per-thread results, sums them on the host,
 * and reports both the total (for correctness) and elapsed time (for
 * later comparison against the 356 anchor once this is wired into the
 * same 3-chunk measure2 protocol -- 364 itself is a single-shot,
 * non-chunked run: correctness first, timing protocol parity later).
 * ------------------------------------------------------------------- */
/* 395c-r4: driver API, needed only for NQ_EXTRA_CTX (cuCtxCreate). This
 * include sits inside the #ifdef __CUDACC__ host-runner section, so the
 * plain-gcc CPU test build never sees it. Requires -lcuda at link time. */
#include <cuda.h>
/* 402-r4: helper-context child processes (fork/pipe/prctl). Host-runner only. */
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/prctl.h>
#define CU_CHECK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char *_s = NULL; cuGetErrorString(_r, &_s); \
        fprintf(stderr, "[CU-ERROR] %s:%d: %s failed: %s\n", \
                __FILE__, __LINE__, #call, _s ? _s : "?"); \
        exit(1); \
    } \
} while (0)

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
    if (N > PACK403_WIDTH) {
        fprintf(stderr, "[403-pack] N=%lld unsupported: the 12-byte frame holds col(22) avail(22) ld(21) rd(31) (N <= %d only)\n",
                (long long)N, PACK403_WIDTH);
        return 3;
    }
    const int layout = select_pack_layout(N, "gpu");   /* 403-r2 */
    if (layout < 0) return 3;
    const char *in_path = argv[2];
    const char *out_path = argv[3];
    int have_expected = (argc == 5);
    unsigned long long expected_total = have_expected ? strtoull(argv[4], NULL, 10) : 0ULL;

    /* 402-r4 HELPER CONTEXTS -- the ONE new treatment in this revision.
     * NQ_HELPER_CTX=<n> forks n child processes, each of which creates its
     * own CUDA context (cudaFree(0)), optionally touches NQ_HELPER_MB MiB of
     * device memory, and then blocks until this process exits. This is the
     * in-binary form of 395c_ctx_holder: 401-r4 showed that a context made
     * IN this process (NQ_EXTRA_CTX) no longer counts while another process's
     * context does (H1 -0.001% vs H2 -0.288%), and 402-r3 measured the value
     * at MB=960: a third context is -1.78%, and any further occupancy (a
     * fourth context, or +128/+384 MB) is a flat -2.2%.
     * The fork happens BEFORE any CUDA call in the parent, so each child
     * initialises the runtime fresh. Each child signals readiness through a
     * pipe (the parent waits for it, so the context exists before anything
     * below runs) and blocks on a second pipe whose write end only the parent
     * holds: when the parent exits, read() returns 0 and the child exits;
     * PR_SET_PDEATHSIG is the belt to that braces. Children _exit(), never
     * return into main(). With NQ_HELPER_CTX unset or 0 nothing is forked and
     * the code path is identical to 402-r3 -- gated statically and at runtime
     * ([gpu-helper] helpers=0). */
    int   helper_n = 0;
    long  helper_mb = 0;
    pid_t helper_pid[8];
    int   helper_keep_fd = -1;
    {
        long want = 0;
        const char *env_h = getenv("NQ_HELPER_CTX");
        if (env_h != NULL && env_h[0] != '\0') {
            long v = strtol(env_h, NULL, 10);
            if (v < 0 || v > 8) { fprintf(stderr, "ERROR: NQ_HELPER_CTX='%s' out of range [0,8]\n", env_h); return 1; }
            want = v;
        }
        const char *env_hm = getenv("NQ_HELPER_MB");
        if (env_hm != NULL && env_hm[0] != '\0') {
            long v = strtol(env_hm, NULL, 10);
            if (v < 0 || v > 20000) { fprintf(stderr, "ERROR: NQ_HELPER_MB='%s' out of range [0,20000]\n", env_hm); return 1; }
            helper_mb = v;
        }
        if (want > 0) {
            int keep[2];
            if (pipe(keep) != 0) { perror("pipe(keep)"); return 1; }
            for (long i = 0; i < want; i++) {
                int ready[2];
                if (pipe(ready) != 0) { perror("pipe(ready)"); return 1; }
                pid_t pid = fork();
                if (pid < 0) { perror("fork"); return 1; }
                if (pid == 0) {
                    /* child: own context, optional touched allocation, then wait for EOF */
                    close(keep[1]); close(ready[0]);
                    prctl(PR_SET_PDEATHSIG, SIGKILL);
                    char st = 1;
                    void *hp = NULL;
                    if (cudaFree(0) != cudaSuccess) st = 0;
                    if (st && helper_mb > 0) {
                        if (cudaMalloc(&hp, (size_t)helper_mb << 20) != cudaSuccess) st = 0;
                        else if (cudaMemset(hp, 0, (size_t)helper_mb << 20) != cudaSuccess) st = 0;
                        else if (cudaDeviceSynchronize() != cudaSuccess) st = 0;
                    }
                    if (write(ready[1], &st, 1) != 1) st = 0;
                    close(ready[1]);
                    if (st) { char c; ssize_t r; do { r = read(keep[0], &c, 1); } while (r < 0 && errno == EINTR); }
                    if (hp) cudaFree(hp);
                    _exit(st ? 0 : 1);
                }
                close(ready[1]);
                char st = 0; ssize_t r; do { r = read(ready[0], &st, 1); } while (r < 0 && errno == EINTR);
                close(ready[0]);
                if (r != 1 || st != 1) { fprintf(stderr, "ERROR: helper %ld (pid %d) failed to create its context\n", i, (int)pid); return 1; }
                helper_pid[helper_n++] = pid;
            }
            close(keep[0]);
            helper_keep_fd = keep[1];
        }
        fprintf(stderr, "[gpu-helper] helpers=%d helper_mb=%ld pids=", helper_n, helper_mb);
        for (int i = 0; i < helper_n; i++) fprintf(stderr, "%s%d", i ? "," : "", (int)helper_pid[i]);
        fprintf(stderr, "%s\n", helper_n ? "" : "-");
    }

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
    fprintf(stderr, "[gpu-run] N=%lld records=%lld src=%s\n", (long long)N, (long long)m, in_path);

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

    /* 394b: MAX_BLOCKS is now overridable at RUN time via the environment
     * variable NQ_MAX_BLOCKS (default 800 since 395b -- was 484 = the 389
     * value -- a plain invocation now runs the confirmed config). This is a
     * pure launch-configuration knob: stride is already a kernel ARGUMENT,
     * so the compiled kernel (process_one_task + __global__) is identical
     * for every value -- only grid.x, stride and the results buffer size
     * change on the host. See 394b_README_append.md for why this axis is
     * being swept (393-9: Active Warps/Scheduler=1.51 of a hardware 12). */
    /* 400: was `const int BLOCK = 32;`. Now an env knob, exactly like
     * NQ_MAX_BLOCKS above it and for the same reason -- it is a pure launch
     * configuration. stride is already a kernel ARGUMENT and the kernel
     * reads blockDim.x, so the compiled kernel is identical for every value:
     * the kernel-region sha256 stays ebd7f523..., which the harness gates.
     * WHY IT MATTERS: on sm_86 an SM holds at most 16 blocks and 48 warps.
     * With BLOCK=32 a block IS one warp, so 16 blocks/SM caps theoretical
     * occupancy at 16/48 = 33.3% no matter how large the grid gets -- and
     * the production grid of 800 blocks over 80 SMs only reaches 10 of
     * those 16. 399-r2 measured 2.43 active warps per scheduler out of 12
     * and found `wait` (fixed-latency dependency) to be 45% of stall
     * cycles, against 14% for memory. Too few warps to hide short
     * latencies is exactly what that pattern looks like.
     *   BLOCK=64  -> 16 blocks x 2 warps = 32/48 = 66.7%
     *   BLOCK=128 -> 12 blocks x 4 warps = 48/48 = 100%  (registers allow
     *                52 warps at 37 regs/thread, so they are not the cap)
     */
    int BLOCK = 32;
    {
        const char *env_bk = getenv("NQ_BLOCK");
        if (env_bk != NULL && env_bk[0] != '\0') {
            long v = strtol(env_bk, NULL, 10);
            if (v == 32 || v == 64 || v == 128 || v == 256 || v == 512 || v == 1024) {
                BLOCK = (int)v;
            } else {
                fprintf(stderr, "ERROR: NQ_BLOCK='%s' must be one of 32/64/128/256/512/1024\n", env_bk);
                return 1;
            }
        }
    }
    int MAX_BLOCKS = 800;   /* 395b: was 484. Standing policy: every entry point
                             * defaults to the CONFIRMED parameters (394g: 800).
                             * NQ_MAX_BLOCKS still overrides; the Codon dispatcher
                             * always passes it explicitly. */
    {
        const char *env_mb = getenv("NQ_MAX_BLOCKS");
        if (env_mb != NULL && env_mb[0] != '\0') {
            long v = strtol(env_mb, NULL, 10);
            if (v >= 1 && v <= 65535) {
                MAX_BLOCKS = (int)v;
            } else {
                fprintf(stderr, "ERROR: NQ_MAX_BLOCKS='%s' out of range [1,65535]\n", env_mb);
                return 1;
            }
        }
    }
    /* 400-r3: the L1 / shared-memory carveout.
     * On sm_86 one 128 KB unified cache is split between L1 and shared
     * memory, and cudaFuncAttributePreferredSharedMemoryCarveout asks for a
     * percentage to go to shared. This kernel uses NO shared memory and no
     * barriers (ptxas: "used 0 barriers"), so every byte given to shared is
     * a byte L1 cannot use -- and L1 is where the 208-byte-per-thread DFS
     * stack lives.
     * WHY THIS IS SUSPECTED: 400-r2 swept stride at N=19 and found
     *     10 warps/SM   320 thr x 208 B =  66.6 KB    2,120 ms
     *     16 warps/SM   512 thr x 208 B = 106.5 KB    3,849 ms  (+82%)
     *     32 warps/SM  1024 thr x 208 B = 213.0 KB   10,699 ms (+405%)
     *     48 warps/SM  1536 thr x 208 B = 319.5 KB   16,242 ms (+666%)
     *   and at each stride the BLOCK variants agreed to within 1%, so the
     *   driver is stride, not block shape. A 128 KB L1 would still hold the
     *   106.5 KB case comfortably, yet it is already 82% slower; a 64 KB L1
     *   would not, which fits the curve. Hence: find out what the carveout
     *   actually is, and try setting it to 0.
     * NQ_CARVEOUT unset means the attribute is never touched, so the run is
     * bit-identical to 400-r2. This is host-side only: the kernel-region
     * sha256 stays ebd7f523..., which the harness gates.
     * The cudaFuncGetAttributes call below reports the effective carveout,
     * the per-thread local size and the register count on every run, so the
     * default is visible in the log before any timing is interpreted. */
    {
        const char *env_co = getenv("NQ_CARVEOUT");
        int co_req = -1;
        if (env_co != NULL && env_co[0] != '\0') {
            long v = strtol(env_co, NULL, 10);
            if (v < 0 || v > 100) {
                fprintf(stderr, "ERROR: NQ_CARVEOUT='%s' out of range [0,100]\n", env_co);
                return 1;
            }
            co_req = (int)v;
            if (layout == PACK_LAYOUT_402) {
                CUDA_CHECK(cudaFuncSetAttribute(kernel_dfs_iter_gpu_maxd14<PACK_LAYOUT_402>,
                                                cudaFuncAttributePreferredSharedMemoryCarveout,
                                                co_req));
            } else {
                CUDA_CHECK(cudaFuncSetAttribute(kernel_dfs_iter_gpu_maxd14<PACK_LAYOUT_403>,
                                                cudaFuncAttributePreferredSharedMemoryCarveout,
                                                co_req));
            }
        }
        struct cudaFuncAttributes fa;
        if (layout == PACK_LAYOUT_402) {
            CUDA_CHECK(cudaFuncGetAttributes(&fa, kernel_dfs_iter_gpu_maxd14<PACK_LAYOUT_402>));
        } else {
            CUDA_CHECK(cudaFuncGetAttributes(&fa, kernel_dfs_iter_gpu_maxd14<PACK_LAYOUT_403>));
        }
        fprintf(stderr,
                "[gpu-carveout] requested=%d effective_pref=%d local_bytes_per_thread=%lld "
                "static_shared_bytes=%lld registers=%d\n",
                co_req, fa.preferredShmemCarveout,
                (long long)fa.localSizeBytes, (long long)fa.sharedSizeBytes, fa.numRegs);
    }

    const int64_t stride = (int64_t)BLOCK * MAX_BLOCKS; /* 15488 by default, matches 292's K-batching */
    fprintf(stderr, "[gpu-config] BLOCK=%d MAX_BLOCKS=%d stride=%lld threads=%lld k_per_thread_max=%lld\n",
            BLOCK, MAX_BLOCKS, (long long)stride, (long long)stride,
            (long long)((m + stride - 1) / stride));

    uint32_t board_mask = (uint32_t)((1ULL << N) - 1);
    uint32_t n3 = (uint32_t)(1ULL << (N - 3));
    uint32_t n4 = (uint32_t)(1ULL << (N - 4));

    /* 395c-r4 EXTRA CONTEXTS -- the ONE new treatment in this revision.
     * NQ_EXTRA_CTX=<n> makes THIS process create n additional CUDA contexts
     * on the device and hold them, idle, until the end. Each is popped off
     * the thread immediately, so every runtime API call below still runs on
     * the primary context exactly as before -- the extra contexts only exist.
     * They are created before the primary context (the first cudaMalloc) and
     * before any buffer, mirroring a holder process that was already running.
     * r3 showed that the CONTEXT COUNT is a real axis (1 -> 2 -> 3 contexts:
     * 139,185 -> 133,561 -> 133,172). This asks whether the extra context has
     * to live in ANOTHER PROCESS, or whether ours will do -- which decides
     * whether production needs a helper process at all.
     * With NQ_EXTRA_CTX unset or 0, nothing is created and the code path is
     * identical to 395c-r3. */
    CUcontext extra_ctx[16];
    int extra_n = 0;
    {
        long want = 0;
        const char *env_ctx = getenv("NQ_EXTRA_CTX");
        if (env_ctx != NULL && env_ctx[0] != '\0') {
            long v = strtol(env_ctx, NULL, 10);
            if (v < 0 || v > 16) {
                fprintf(stderr, "ERROR: NQ_EXTRA_CTX='%s' out of range [0,16]\n", env_ctx);
                return 1;
            }
            want = v;
        }
        if (want > 0) {
            CUdevice cu_dev;
            CU_CHECK(cuInit(0));
            CU_CHECK(cuDeviceGet(&cu_dev, 0));
            for (long i = 0; i < want; i++) {
                CUcontext c = NULL, popped = NULL;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
                /* CUDA 13 promoted cuCtxCreate to _v4: (pctx, params, flags, dev).
                 * NULL params = the default context, i.e. exactly what _v2 made. */
                CU_CHECK(cuCtxCreate(&c, NULL, 0, cu_dev));
#else
                CU_CHECK(cuCtxCreate(&c, 0, cu_dev));
#endif
                CU_CHECK(cuCtxPopCurrent(&popped));  /* leave the runtime on the primary context */
                extra_ctx[extra_n++] = c;
            }
        }
        fprintf(stderr, "[gpu-ctx] extra_ctx=%d\n", extra_n);
    }

    /* 395c-r3 SELF-PAD -- the ONE new treatment in this revision.
     * NQ_PAD_MB=<n> makes THIS process occupy n MiB of device memory, touched,
     * before any working buffer is allocated, and hold it until after the
     * kernel. It exists to separate the two factors 395c/r2 could not:
     *   "device memory is occupied"  vs  "a second CUDA context exists".
     * A foreign holder changes both at once; this changes only the first.
     * With NQ_PAD_MB unset or 0 NOTHING is allocated and the code path is
     * identical to 395c-r2 -- the harness gates that both statically (the
     * cudaMalloc is inside if (pad_mb > 0)) and at runtime ([gpu-pad]
     * pad_mb=0 in the default-config probe). */
    void *d_pad = NULL;
    long pad_mb = 0;
    {
        const char *env_pad = getenv("NQ_PAD_MB");
        if (env_pad != NULL && env_pad[0] != '\0') {
            long v = strtol(env_pad, NULL, 10);
            if (v < 0 || v > 20000) {
                fprintf(stderr, "ERROR: NQ_PAD_MB='%s' out of range [0,20000]\n", env_pad);
                return 1;
            }
            pad_mb = v;
        }
        if (pad_mb > 0) {
            CUDA_CHECK(cudaMalloc(&d_pad, (size_t)pad_mb << 20));
            CUDA_CHECK(cudaMemset(d_pad, 0, (size_t)pad_mb << 20));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        fprintf(stderr, "[gpu-pad] pad_mb=%ld ptr=%p\n", pad_mb, d_pad);
    }

    uint32_t *d_ld, *d_rd, *d_col, *d_ctrl0, *d_free, *d_markctrl, *d_wlo;
    uint8_t *d_meta_next;
    uint64_t *d_results;

    CUDA_CHECK(cudaMalloc(&d_ld,       (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_rd,       (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_col,      (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ctrl0,    (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_free,     (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_markctrl, (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_wlo,      (size_t)m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_meta_next, 28 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_results,  (size_t)stride * sizeof(uint64_t)));

    /* 395c-r2 PLACEMENT PROBE -- diagnostic only, provably non-perturbing.
     * It runs AFTER every cudaMalloc (so it cannot change where anything
     * lands) and BEFORE ev_h2d_start (so it is outside every timed region).
     * It allocates nothing: cudaMemGetInfo is a pure query. Its whole job is
     * to record, per cell, WHERE our buffers landed, so that D0/D1/D2 can be
     * told apart by address and not only by wall time. */
    {
        size_t mem_free = 0, mem_total = 0;
        CUDA_CHECK(cudaMemGetInfo(&mem_free, &mem_total));
        fprintf(stderr, "[gpu-mem] ld=%p rd=%p col=%p ctrl0=%p free=%p markctrl=%p wlo=%p meta=%p results=%p\n",
                (void*)d_ld, (void*)d_rd, (void*)d_col, (void*)d_ctrl0,
                (void*)d_free, (void*)d_markctrl, (void*)d_wlo,
                (void*)d_meta_next, (void*)d_results);
        fprintf(stderr, "[gpu-mem-base] ld=0x%llx ld_off2m=0x%llx span=%lld free_mb=%lld total_mb=%lld\n",
                (unsigned long long)(uintptr_t)d_ld,
                (unsigned long long)((uintptr_t)d_ld & 0x1FFFFFULL),
                (long long)((uintptr_t)d_results - (uintptr_t)d_ld),
                (long long)(mem_free >> 20), (long long)(mem_total >> 20));
    }

    cudaEvent_t ev_h2d_start, ev_h2d_end, ev_kernel_start, ev_kernel_end, ev_d2h_end;
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

    dim3 grid(MAX_BLOCKS);
    dim3 block(BLOCK);
    CUDA_CHECK(cudaEventRecord(ev_kernel_start));
    if (layout == PACK_LAYOUT_402) {
        kernel_dfs_iter_gpu_maxd14<PACK_LAYOUT_402><<<grid, block>>>(
            d_ld, d_rd, d_col, d_ctrl0, d_free, d_markctrl, d_wlo,
            d_meta_next, d_results,
            m, board_mask, n3, n4, stride
        );
    } else {
        kernel_dfs_iter_gpu_maxd14<PACK_LAYOUT_403><<<grid, block>>>(
            d_ld, d_rd, d_col, d_ctrl0, d_free, d_markctrl, d_wlo,
            d_meta_next, d_results,
            m, board_mask, n3, n4, stride
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_kernel_end));
    CUDA_CHECK(cudaEventSynchronize(ev_kernel_end));

    uint64_t *h_results = (uint64_t*)malloc((size_t)stride * sizeof(uint64_t));
    if (!h_results) {
        fprintf(stderr, "ERROR: host allocation failed for results (%lld)\n", (long long)stride);
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(h_results, d_results, (size_t)stride * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_d2h_end));
    CUDA_CHECK(cudaEventSynchronize(ev_d2h_end));

    float ms_h2d = 0, ms_kernel = 0, ms_d2h = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, ev_h2d_start, ev_h2d_end));
    CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, ev_kernel_start, ev_kernel_end));
    CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, ev_kernel_end, ev_d2h_end));

    unsigned long long total_sum = 0;
    for (int64_t t = 0; t < stride; t++) {
        total_sum += h_results[t];
    }

    FILE *fout = fopen(out_path, "wb");
    if (fout) {
        for (int64_t t = 0; t < stride; t++) {
            unsigned char outb[8];
            for (int b = 0; b < 8; b++) {
                outb[b] = (unsigned char)((h_results[t] >> (8*b)) & 0xFF);
            }
            fwrite(outb, 1, 8, fout);
        }
        fclose(fout);
    }

    printf("[gpu-run-done] N=%lld records=%lld total_sum=%llu "
           "h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f total_ms=%.3f "
           "block=%d max_blocks=%d stride=%lld\n",
           (long long)N, (long long)m, total_sum,
           ms_h2d, ms_kernel, ms_d2h, (double)ms_h2d + ms_kernel + ms_d2h,
           BLOCK, MAX_BLOCKS, (long long)stride);

    if (have_expected) {
        if (total_sum == expected_total) {
            printf("[gpu-run-correctness] MATCH expected=%llu\n", expected_total);
        } else {
            printf("[gpu-run-correctness] MISMATCH expected=%llu got=%llu\n", expected_total, total_sum);
        }
    }

    cudaFree(d_ld); cudaFree(d_rd); cudaFree(d_col); cudaFree(d_ctrl0);
    cudaFree(d_free); cudaFree(d_markctrl); cudaFree(d_wlo);
    cudaFree(d_meta_next); cudaFree(d_results);
    if (d_pad) cudaFree(d_pad);
    for (int _i = 0; _i < extra_n; _i++) cuCtxDestroy(extra_ctx[_i]);
    free(h_ld); free(h_rd); free(h_col); free(h_ctrl0);
    free(h_free); free(h_markctrl); free(h_wlo); free(h_results);

    /* 402-r4: release the helpers (EOF on the keep pipe) and reap them, so no
     * compute process outlives this one on the GPU. */
    if (helper_keep_fd >= 0) close(helper_keep_fd);
    for (int _i = 0; _i < helper_n; _i++) { int _st = 0; waitpid(helper_pid[_i], &_st, 0); }

    return (have_expected && total_sum != expected_total) ? 1 : 0;
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
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s <N> <in_soa7_bin> <out_results_bin> [max_records]\n", argv[0]);
        return 1;
    }
    int64_t N = atoll(argv[1]);
    if (N > PACK403_WIDTH) {
        fprintf(stderr, "[403-pack] N=%lld unsupported: the 12-byte frame holds col(22) avail(22) ld(21) rd(31) (N <= %d only)\n",
                (long long)N, PACK403_WIDTH);
        return 3;
    }
    const int layout = select_pack_layout(N, "cpu");   /* 403-r2 */
    if (layout < 0) return 3;
    const char *in_path = argv[2];
    const char *out_path = argv[3];
    int64_t max_records = (argc == 5) ? atoll(argv[4]) : -1;

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
    if (max_records >= 0 && max_records < m) {
        m = max_records;
        fprintf(stderr, "[limited-run] processing only the first %lld of %ld records\n",
                (long long)m, fsize / 28);
    }

    /* Load all needed records into memory upfront so the per-record work
     * below can be split across threads with OpenMP (when built with
     * -fopenmp; falls back to an ordinary serial loop otherwise). This
     * only restructures the test harness's I/O, not process_one_task()'s
     * logic, and each record is fully independent (no shared state),
     * so this is a safe, embarrassingly-parallel speedup for validation
     * purposes on real data, where a single CPU thread is far slower
     * than the 15,488-way GPU parallelism the kernel is designed for. */
    uint32_t *ld_a = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *rd_a = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *col_a = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *ctrl0_a = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *free_a = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *markctrl_a = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint32_t *wlo_a = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
    uint64_t *contrib_a = (uint64_t*)malloc((size_t)m * sizeof(uint64_t));
    if (!ld_a || !rd_a || !col_a || !ctrl0_a || !free_a || !markctrl_a || !wlo_a || !contrib_a) {
        fprintf(stderr, "ERROR: out of memory allocating %lld records\n", (long long)m);
        return 1;
    }

    unsigned char buf[28];
    for (int64_t idx = 0; idx < m; idx++) {
        if (fread(buf, 1, 28, fin) != 28) {
            fprintf(stderr, "ERROR: short read at record %lld\n", (long long)idx);
            return 1;
        }
        ld_a[idx]       = read_u32_le(buf + 0);
        rd_a[idx]       = read_u32_le(buf + 4);
        col_a[idx]      = read_u32_le(buf + 8);
        ctrl0_a[idx]    = read_u32_le(buf + 12);
        free_a[idx]     = read_u32_le(buf + 16);
        markctrl_a[idx] = read_u32_le(buf + 20);
        wlo_a[idx]      = read_u32_le(buf + 24);
    }
    fclose(fin);

    FILE *fout = fopen(out_path, "wb");
    if (!fout) {
        fprintf(stderr, "ERROR: cannot open output '%s'\n", out_path);
        return 1;
    }

    uint32_t board_mask = (uint32_t)((1ULL << N) - 1);
    uint32_t n3 = (uint32_t)(1ULL << (N - 3));
    uint32_t n4 = (uint32_t)(1ULL << (N - 4));

    uint64_t total_sum = 0;
    clock_t t_start = clock();

#ifdef _OPENMP
    fprintf(stderr, "[openmp] running with %d threads\n", omp_get_max_threads());
    int64_t completed = 0;
    #pragma omp parallel for schedule(dynamic, 64) reduction(+:total_sum)
    for (int64_t idx = 0; idx < m; idx++) {
        uint32_t root_a = free_a[idx] & board_mask;
        uint64_t contribution;
        if (root_a == 0u) {
            contribution = 0;
        } else {
            contribution = process_one_task(
                ld_a[idx], rd_a[idx], col_a[idx], root_a, ctrl0_a[idx], markctrl_a[idx], wlo_a[idx],
                META_NEXT, board_mask, n3, n4 POT_LAYOUT_ARG(layout),
                idx
            );
        }
        contrib_a[idx] = contribution;
        total_sum += contribution;
        #pragma omp atomic
        completed++;
        if (idx % 20000 == 0) {
            double elapsed = (double)(clock() - t_start) / CLOCKS_PER_SEC;
            fprintf(stderr, "[progress] completed~%lld/%lld elapsed=%.1fs (wall, /thread count not shown)\n",
                    (long long)completed, (long long)m, elapsed);
            fflush(stderr);
        }
    }
#else
    for (int64_t idx = 0; idx < m; idx++) {
        uint32_t root_a = free_a[idx] & board_mask;
        uint64_t contribution;
        clock_t t_rec_start = clock();
        if (root_a == 0u) {
            contribution = 0;
        } else {
            contribution = process_one_task(
                ld_a[idx], rd_a[idx], col_a[idx], root_a, ctrl0_a[idx], markctrl_a[idx], wlo_a[idx],
                META_NEXT, board_mask, n3, n4 POT_LAYOUT_ARG(layout),
                idx
            );
        }
        double rec_seconds = (double)(clock() - t_rec_start) / CLOCKS_PER_SEC;
        if (rec_seconds > 2.0) {
            fprintf(stderr, "[SLOW-RECORD] idx=%lld took %.2fs alone -- ld=%u rd=%u col=%u ctrl0=%u markctrl=%u w_lo=%u\n",
                    (long long)idx, rec_seconds, ld_a[idx], rd_a[idx], col_a[idx], ctrl0_a[idx], markctrl_a[idx], wlo_a[idx]);
            fflush(stderr);
        }
        contrib_a[idx] = contribution;
        total_sum += contribution;
        if (idx % 1000 == 0) {
            double elapsed = (double)(clock() - t_start) / CLOCKS_PER_SEC;
            double rate = (idx > 0) ? (double)idx / elapsed : 0.0;
            fprintf(stderr, "[progress] idx=%lld/%lld total_sum_so_far=%llu elapsed=%.1fs rate=%.0f rec/s\n",
                    (long long)idx, (long long)m, (unsigned long long)total_sum, elapsed, rate);
            fflush(stderr);
        }
    }
#endif

    for (int64_t idx = 0; idx < m; idx++) {
        uint64_t contribution = contrib_a[idx];
        unsigned char outb[8];
        for (int b = 0; b < 8; b++) {
            outb[b] = (unsigned char)((contribution >> (8*b)) & 0xFF);
        }
        if (fwrite(outb, 1, 8, fout) != 8) {
            fprintf(stderr, "ERROR: short write at record %lld\n", (long long)idx);
            return 1;
        }
    }

    fclose(fout);
    free(ld_a); free(rd_a); free(col_a); free(ctrl0_a);
    free(free_a); free(markctrl_a); free(wlo_a); free(contrib_a);

    printf("[kernel-cputest-done] N=%lld records=%lld out=%s total_sum=%llu\n",
           (long long)N, (long long)m, out_path, (unsigned long long)total_sum);
    return 0;
}
#endif /* !__CUDACC__ */
