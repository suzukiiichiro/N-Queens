#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 08 pypy対応 int型ビットボード

row 0: . Q . . → 0100 → bit位置  1
row 1: . . . Q → 0001 → bit位置  3
row 2: Q . . . → 1000 → bit位置  0
row 3: . . Q . → 0010 → bit位置  2

[0,1,0,0,   0,0,0,1,   1,0,0,0,   0,0,1,0]


int型ビットボード：
0b0100000110000010（=16962）
クイーンの配置を、1つの整数値（int）で表現したもの。
各ビットが盤面上のセル（マス）に対応し、1=クイーンあり／0=なし。
Pythonのint型で、1つの整数として盤面を表現（ビットごとにセルを管理）

bitarray
bitarray('0100000110000010')
ビット列を配列として保持。スライスやindex操作が可能。固定長で高速。

bitarray.uint64
bitarray().frombytes(uint64_value.to_bytes(...))
o_bytes(...))	bitarrayを64ビットのバイナリ整数として扱う拡張機能（主にbitarray.util）

リスト（可視化用）：
[0, 1, 0, 0,   0, 0, 0, 1,   1, 0, 0, 0,   0, 0, 1, 0]
のように、整数から読み取ったビットを並べたもの。人が見やすく、盤面出力やデバッグに便利。これは盤面を行優先（row-major）に並べた可視的ビット列

"""

#pypyを使う場合はコメントを解除
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

from datetime import datetime
def rotate90(board, n):
    res = 0
    for i in range(n):
        for j in range(n):
            if board & (1 << (i * n + j)):
                res |= 1 << (j * n + (n - 1 - i))
    return res

def mirror_vertical(board, n):
    res = 0
    for i in range(n):
        for j in range(n):
            if board & (1 << (i * n + j)):
                res |= 1 << (i * n + (n - 1 - j))
    return res

def get_symmetries(board, n):
    results = []
    r = board
    for _ in range(4):
        results.append(r)
        results.append(mirror_vertical(r, n))
        r = rotate90(r, n)
    return results

def classify_symmetry(board, n, seen):
    syms = set(get_symmetries(board, n))
    sym_len = len(syms)
    if sym_len == 8:
        return 'COUNT8'
    elif sym_len == 4:
        return 'COUNT4'
    elif sym_len == 2:
        return 'COUNT2'
    else:
        return None

def backtrack(row, cols, hills, dales, board, n, seen, counts):
    if row == n:
        syms = get_symmetries(board, n)
        canonical = min(syms)
        if canonical in seen:
            return
        seen.append(canonical)
        cls = classify_symmetry(board, n, seen)
        if cls:
            counts[cls] += 1
        return
    free = ~(cols | hills | dales) & ((1 << n) - 1)
    while free:
        bit = free & -free
        free ^= bit
        col = bit.bit_length() - 1
        pos = row * n + col
        backtrack(
            row + 1,
            cols | bit,
            (hills | bit) << 1,
            (dales | bit) >> 1,
            board | (1 << pos),
            n, seen, counts
        )

def solve_n_queens_bitboard_int(n):
    seen = []  # set() は Codon で不安定なので list に
    counts = {'COUNT2': 0, 'COUNT4': 0, 'COUNT8': 0}
    backtrack(0, 0, 0, 0, 0, n, seen, counts)
    total = counts['COUNT2'] * 2 + counts['COUNT4'] * 4 + counts['COUNT8'] * 8
    unique = counts['COUNT2'] + counts['COUNT4'] + counts['COUNT8']
    return total, unique

if __name__ == '__main__':
    from datetime import datetime
    _min = 4
    max = 17
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(_min, max):
        start_time = datetime.now()
        total, unique = solve_n_queens_bitboard_int(size)
        time_elapsed = datetime.now() - start_time
        text = str(time_elapsed)[:-3]
        print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")

