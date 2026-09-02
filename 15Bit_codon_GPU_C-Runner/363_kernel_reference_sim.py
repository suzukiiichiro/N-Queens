#!/usr/bin/env python3
"""
363_kernel_reference_sim.py

A standalone, literal re-execution of kernel_dfs_iter_gpu_maxd14 (lines
713-1030 of 360Py/361Py/362Py/363Py) under plain CPython, used as an
INDEPENDENT reference to cross-validate 363_kernel_maxd14.cu -- this is
not derived from the C port; it is transcribed directly from the real
Codon kernel source, with only u32()/u64()/__array__[u64](n) given
trivial bitmask/list stand-ins so it runs outside Codon.

Every variable name is kept identical to the Codon source so the two
(this file and 363_kernel_maxd14.cu) can be diffed by eye. This mirrors
the same cross-validation method 361 already used successfully for
build_soa_for_range()+symmetry().

WHY THIS MATTERS FOR 363: kernel_dfs_iter_gpu_maxd14 is only ever
launched, in real production, on constellations whose required
schedule depth is <=14 (max_schedule_depth_of_tasks() filters records
into maxd14/16/18/20/21 groups before dispatch; only the maxd14 group
reaches this kernel). The real N=21 correctness oracle 314666222712
is the sum across ALL FIVE kernels, not this one alone -- so a
maxd14-only subset's total will NOT equal 314666222712, and must not
be compared against it. Use 363_filter_maxd14_only.py first to extract
only the maxd<=14 subset from 361's dump, then cross-check THIS
script's output against 363_kernel_maxd14.cu's CPU-test output on that
SAME subset (both should agree with each other, byte-for-byte and in
total, on whatever that subset's true total actually is).

Usage:
  python3 363_kernel_reference_sim.py <in_soa7.bin> <out_results.bin>

Input format: 7 x u32 LE per record (ld, rd, col, ctrl0, free,
markctrl, w_lo) -- the same layout 363_kernel_maxd14.cu's CPU test
harness and 363_filter_maxd14_only.py's output use.

Output format: 1 x u64 LE per record (that record's contribution,
total*w_lo), same order as input -- directly comparable via `cmp`
against 363_kernel_maxd14.cu's CPU-test output file.
"""
import sys
import struct

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def u32(x):
    return x & MASK32


def u64(x):
    return x & MASK64


MAXD14_ANCESTOR = 13

META_NEXT = [1, 2, 3, 3, 2, 6, 2, 2, 0, 4, 5, 7, 13, 14, 14, 14,
             17, 14, 14, 20, 21, 21, 21, 25, 21, 21, 26, 26]

IS_BASE_MASK = u32(69222408)
IS_JMARK_MASK = u32(4)
IS_MARK_MASK = u32(199209203)
IS_P5_MASK = u32(3840)
SEL2_MASK = u32(34742338)

BLOCK_CODE_B0_MASK = u32(173707345)
BLOCK_CODE_B1_MASK = u32(12689458)
BLOCK_CODE_B2_MASK = u32(18088064)

OP_STEP3_MASK = u32(24)   # codes 3,4
OP_ADD1_MASK = u32(32)    # code 5
OP_BL1_MASK = u32(12)     # codes 2,3
OP_BL2_MASK = u32(16)     # code 4
OP_KN3_MASK = u32(18)     # codes 1,4
OP_KN4_MASK = u32(8)      # code 3


def process_one_task(root_ld, root_rd, root_col, root_a_in, ctrl0, markctrl,
                      w_lo, bm, n3, n4):
    """Literal re-execution of the per-idx body inside kernel_dfs_iter_gpu_
    maxd14's grid-stride while loop (720-1028 of the Codon source),
    factored to match 363_kernel_maxd14.cu's process_one_task() exactly."""
    jmark = markctrl & u32(31)
    endm = (markctrl >> u32(5)) & u32(31)
    mark1 = (markctrl >> u32(10)) & u32(31)
    mark2 = (markctrl >> u32(15)) & u32(31)
    total = u64(0)

    root_a = root_a_in & bm
    if root_a == u32(0):
        return u64(0)

    schedule_raw = ctrl0
    schedule_depth = 0
    schedule_lo = u32(0)
    schedule_hi = u32(0)
    child_jmark_mask = u32(0)
    future_check_mask = u32(0)
    terminal_parent_depth = 0
    terminal_is_base14 = u32(0)
    root_action = u32(0)

    while True:
        schedule_fu = schedule_raw & u32(31)
        schedule_rowv = (schedule_raw >> u32(5)) & u32(31)

        if ((IS_P5_MASK >> schedule_fu) & u32(1)) != u32(0):
            if schedule_rowv == mark1:
                schedule_fu = u32(META_NEXT[int(schedule_fu)])

        frame_action = u32(0)
        frame_nibble = u32(0)
        frame_raw = u32(0)
        schedule_fcvu = u32(0)
        schedule_isbu = (IS_BASE_MASK >> schedule_fu) & u32(1)
        if schedule_isbu != u32(0) and schedule_rowv == endm:
            frame_action = u32(3) if schedule_fu == u32(14) else u32(2)
        else:
            schedule_ismu = (IS_MARK_MASK >> schedule_fu) & u32(1)
            schedule_block_code = u32(0)
            schedule_stepv = u32(1)
            schedule_use_futureu = u32(1) - schedule_ismu
            schedule_nextfidu = schedule_fu

            if schedule_ismu != u32(0):
                schedule_markv = mark2 if ((SEL2_MASK >> schedule_fu) & u32(1)) != u32(0) else mark1
                if schedule_rowv == schedule_markv:
                    schedule_block_code = (
                        ((BLOCK_CODE_B0_MASK >> schedule_fu) & u32(1))
                        | (((BLOCK_CODE_B1_MASK >> schedule_fu) & u32(1)) << u32(1))
                        | (((BLOCK_CODE_B2_MASK >> schedule_fu) & u32(1)) << u32(2))
                    )
                    schedule_stepv = u32(2) + ((OP_STEP3_MASK >> schedule_block_code) & u32(1))
                    schedule_use_futureu = u32(0)
                    schedule_nextfidu = u32(META_NEXT[int(schedule_fu)])

            schedule_isju = (IS_JMARK_MASK >> schedule_fu) & u32(1)
            if schedule_isju != u32(0):
                if schedule_rowv == jmark:
                    frame_action = u32(1)
                    schedule_nextfidu = u32(META_NEXT[int(schedule_fu)])

            schedule_child_rowu = schedule_rowv + schedule_stepv
            if schedule_use_futureu != u32(0) and schedule_child_rowu < endm:
                schedule_fcvu = u32(1)
            frame_nibble = schedule_block_code | (schedule_fcvu << u32(3))
            frame_raw = schedule_nextfidu | (schedule_child_rowu << u32(5))

        if schedule_depth == 0:
            root_action = frame_action
        else:
            parent_depth = schedule_depth - 1
            if frame_action == u32(1):
                child_jmark_mask |= u32(1) << u32(parent_depth)
            elif frame_action >= u32(2):
                terminal_parent_depth = parent_depth
                terminal_is_base14 = u32(1) if frame_action == u32(3) else u32(0)

        if frame_action >= u32(2):
            break

        if schedule_fcvu != u32(0):
            future_check_mask |= u32(1) << u32(schedule_depth)

        if schedule_depth < 8:
            schedule_lo |= frame_nibble << u32(schedule_depth * 4)
        else:
            schedule_hi |= frame_nibble << u32((schedule_depth - 8) * 4)
        schedule_raw = frame_raw
        schedule_depth += 1

    if root_action == u32(2):
        return u64(w_lo)
    if root_action == u32(3):
        total += u64(1) if ((root_a & ~u32(1)) != u32(0)) else u64(0)
        return total * u64(w_lo)
    if root_action == u32(1):
        root_a &= ~u32(1)
        if root_a == u32(0):
            return u64(0)
        root_ld |= u32(1)

    terminal_depth = terminal_parent_depth
    terminal_base14 = terminal_is_base14

    save_sp = u32(0)
    stack_ptr = 0
    cur_depth = 0
    cur_ld = root_ld
    cur_rd = root_rd
    cur_col = root_col
    cur_avail = root_a

    stack = [u64(0)] * (MAXD14_ANCESTOR * 2)

    root_rest = cur_avail & (cur_avail - u32(1))
    root_second = root_rest & (u32(0) - root_rest)
    root_after_second = root_rest ^ root_second

    if root_after_second == u32(0):
        root_first = cur_avail & (u32(0) - cur_avail)
        pr_nibble_op = schedule_lo & u32(15)
        pr_block_code = pr_nibble_op & u32(7)
        pr_bit = root_first

        if pr_block_code != u32(0):
            pr_stepu = u32(2) + ((OP_STEP3_MASK >> pr_block_code) & u32(1))
            pr_addvu = (OP_ADD1_MASK >> pr_block_code) & u32(1)
            pr_bLiu = (
                ((OP_BL1_MASK >> pr_block_code) & u32(1))
                | (((OP_BL2_MASK >> pr_block_code) & u32(1)) << u32(1))
            )
            pr_ktu = (
                ((OP_KN3_MASK >> pr_block_code) & u32(1))
                | (((OP_KN4_MASK >> pr_block_code) & u32(1)) << u32(1))
            )
            pr_bKu = (n3 & (u32(0) - (pr_ktu & u32(1)))) | (n4 & (u32(0) - (pr_ktu >> u32(1))))
            pr_nld = ((cur_ld | pr_bit) << pr_stepu) | pr_addvu | pr_bLiu
            pr_nrd = ((cur_rd | pr_bit) >> pr_stepu) | pr_bKu
        else:
            pr_nld = (cur_ld | pr_bit) << u32(1)
            pr_nrd = (cur_rd | pr_bit) >> u32(1)
        pr_ncol = cur_col | pr_bit
        pr_nf = bm & ~(pr_nld | pr_nrd | pr_ncol)
        pr_descend = u32(1)
        if pr_nf == u32(0):
            pr_descend = u32(0)
        if pr_descend != u32(0):
            if future_check_mask != u32(0):
                if (pr_nibble_op & u32(8)) != u32(0):
                    if (bm & ~((pr_nld << u32(1)) | (pr_nrd >> u32(1)) | pr_ncol)) == u32(0):
                        pr_descend = u32(0)

        if pr_descend != u32(0):
            if terminal_depth == 0:
                if terminal_base14 == u32(0):
                    total += u64(1)
                else:
                    total += u64(1) if ((pr_nf & ~u32(1)) != u32(0)) else u64(0)
                pr_descend = u32(0)

        if pr_descend != u32(0):
            pr_child_jmark = child_jmark_mask & u32(1)
            if pr_child_jmark != u32(0):
                pr_nf &= ~u32(1)
                if pr_nf == u32(0):
                    pr_descend = u32(0)
                else:
                    pr_nld |= u32(1)

        cur_avail = root_rest
        if pr_descend != u32(0):
            if cur_avail != u32(0):
                stack[stack_ptr] = u64(cur_ld) | (u64(cur_rd) << u64(32))
                stack[stack_ptr + 1] = u64(cur_col) | (u64(cur_avail | (u32(cur_depth) << u32(27))) << u64(32))
                stack_ptr += 2
                save_sp += u32(1)
            cur_ld = pr_nld
            cur_rd = pr_nrd
            cur_col = pr_ncol
            cur_avail = pr_nf
            cur_depth = 1

    while True:
        if cur_avail == u32(0):
            if save_sp == u32(0):
                break
            save_sp -= u32(1)
            stack_ptr -= 2
            packed_ldrd = stack[stack_ptr]
            packed_colav = stack[stack_ptr + 1]
            cur_ld = u32(packed_ldrd)
            cur_rd = u32(packed_ldrd >> u64(32))
            cur_col = u32(packed_colav)
            saved_avail = u32(packed_colav >> u64(32))
            cur_avail = saved_avail & bm
            cur_depth = int(saved_avail >> u32(27))
            continue

        if cur_depth < 8:
            nibble_op = (schedule_lo >> u32(cur_depth * 4)) & u32(15)
        else:
            nibble_op = (schedule_hi >> u32((cur_depth - 8) * 4)) & u32(15)
        bit = cur_avail & (u32(0) - cur_avail)
        cur_avail = cur_avail ^ bit

        nld = (cur_ld | bit) << u32(1)
        nrd = (cur_rd | bit) >> u32(1)
        ncol = cur_col | bit
        if (nibble_op & u32(7)) != u32(0):
            block_code = nibble_op & u32(7)
            stepu = u32(2) + ((OP_STEP3_MASK >> block_code) & u32(1))
            addvu = (OP_ADD1_MASK >> block_code) & u32(1)
            bLiu = (
                ((OP_BL1_MASK >> block_code) & u32(1))
                | (((OP_BL2_MASK >> block_code) & u32(1)) << u32(1))
            )
            ktu = (
                ((OP_KN3_MASK >> block_code) & u32(1))
                | (((OP_KN4_MASK >> block_code) & u32(1)) << u32(1))
            )
            bKu = (n3 & (u32(0) - (ktu & u32(1)))) | (n4 & (u32(0) - (ktu >> u32(1))))
            nld = ((cur_ld | bit) << stepu) | addvu | bLiu
            nrd = ((cur_rd | bit) >> stepu) | bKu
        nf = bm & ~(nld | nrd | ncol)
        if nf == u32(0):
            continue
        if future_check_mask != u32(0):
            if (nibble_op & u32(8)) != u32(0):
                if (bm & ~((nld << u32(1)) | (nrd >> u32(1)) | ncol)) == u32(0):
                    continue

        if cur_depth == terminal_depth:
            if terminal_base14 == u32(0):
                total += u64(1)
            else:
                total += u64(1) if ((nf & ~u32(1)) != u32(0)) else u64(0)
            continue

        child_jmark = (child_jmark_mask >> u32(cur_depth)) & u32(1)
        if child_jmark != u32(0):
            nf &= ~u32(1)
            if nf == u32(0):
                continue
            nld |= u32(1)

        next_depth = cur_depth + 1
        if cur_avail != u32(0):
            stack[stack_ptr] = u64(cur_ld) | (u64(cur_rd) << u64(32))
            stack[stack_ptr + 1] = u64(cur_col) | (u64(cur_avail | (u32(cur_depth) << u32(27))) << u64(32))
            stack_ptr += 2
            save_sp += u32(1)
        cur_ld = nld
        cur_rd = nrd
        cur_col = ncol
        cur_avail = nf
        cur_depth = next_depth

    return total * u64(w_lo)


def main():
    if len(sys.argv) not in (3, 4):
        print(f"Usage: {sys.argv[0]} <in_soa7.bin> <out_results.bin> [N=21]", file=sys.stderr)
        return 1
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, 'rb') as f:
        data = f.read()
    if len(data) % 28 != 0:
        print(f"ERROR: input size {len(data)} not a multiple of 28", file=sys.stderr)
        return 1
    n = len(data) // 28

    board_mask = None  # set from N below
    return_code = 0
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 21
    board_mask = u32((1 << N) - 1)
    n3 = u32(1 << (N - 3))
    n4 = u32(1 << (N - 4))

    total_sum = 0
    with open(out_path, 'wb') as fout:
        for i in range(n):
            ld, rd, col, ctrl0, free, markctrl, w_lo = struct.unpack_from('<7I', data, i * 28)
            contribution = int(process_one_task(ld, rd, col, free, ctrl0, markctrl, w_lo,
                                                 board_mask, n3, n4))
            total_sum += contribution
            fout.write(struct.pack('<Q', contribution))
            if i % 50000 == 0:
                print(f"[progress] idx={i}/{n} total_sum_so_far={total_sum}", file=sys.stderr)

    print(f"[kernel-refsim-done] N={N} records={n} out={out_path} total_sum={total_sum}")
    return return_code


if __name__ == "__main__":
    sys.exit(main())
