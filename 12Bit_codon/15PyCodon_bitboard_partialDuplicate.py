#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 15 1行目以外でも部分対称除去（行列単位）
構築途中（例：2〜n-1行）でも、回転・ミラーで過去の構成と一致する盤面が出てくる場合は prune 可能

現在の solve_n_queens_bitboard_int() は、完成盤面（row == n）時点でのみ回転・ミラーを生成して seen または hash により重複判定しています。

途中構築時の部分対称性除去	❌ 未対応（明示的な部分盤面の照合・除去はしていない）
導入の判断	⏳ 実装可能だが、現時点ではコストの方が大きい
今後導入するなら？	n ≥ 14 以上かつ count分類が目的の高速モード としてオプション導入が妥当

✅1行目以外でも部分対称除去（行列単位）
✅「Zobrist Hash」 
✅マクロチェス（局所パターン）による構築制限
❌「ミラー＋90度回転」による構築時の重複排除
✅ 180度対称除去
✅ ビット演算による衝突枝刈り	同一列・対角線（↘ / ↙）との衝突を int のビット演算で高速除去	`free = ~(cols	hills
✅ 左右対称性除去	1行目のクイーンを左半分の列（0～n//2−1）に限定し、ミラー対称を除去	for col in range(n // 2):	済
✅ 中央列の特別処理（n奇数）	中央列は回転・ミラーで重複しないため個別に探索し、COUNT2分類に貢献	if n % 2 == 1: ブロック内で col = n // 2 を探索	済
✅ 角位置（col==0）とそれ以外で分岐	1行目の col == 0 を is_corner=True として分離し、COUNT2偏重を明示化	backtrack(..., is_corner=True) による分岐	済
✅ 対称性分類（COUNT2 / 4 / 8）	回転・反転の8通りから最小値を canonical にし、重複除去＆分類判定	len(set(symmetries)) による分類	済
"""

""" 15 1行目以外でも部分対称除去（行列単位）real    0m2.390s
solve_n_queens_bitboard_partialDuplicate(13)
特徴
is_partial_duplicateは同じような盤面なら打ち切るというものだが、完全解が必要な場合は使えない


最小値チェックを省力化。boardが巨大ビット列なので、ハッシュ値（32bit整数）にしてから最小値チェックしてる ただし、n28だと32bitだと確実に衝突するので64bitか128bitにする必要あり（回転チェックの時はハッシュ値ではない）

earlyPruning(if row >= 2 and free == 0:)が設定されているが、while free:にほぼ吸収されると思われる

1行目の角（列0）にクイーンを置いた場合を別処理している。ただ、別処理しているだけで枝刈りはしてない。フラグだけたててる（1行目の角（列0）にクイーンを置いた場合はdistinct = len(set(sym))せずCOUNT8してよい（最小値チェックは必要））。

長所
角スタート時だけ最終行 (n-1, n-1) にクイーンを置かない枝刈りが追加されてる

boardを１個のビット列にしている。symmetryOpsの回転、反転もバラさないままできている
左右対称の枝刈りをやっている（１行目は半分の列だけクイーンを置く）
COUNT2,4,8の判定distinct = len(set(sym))はcount = sum(1 for s in sym if s == canonical)よりは速い

短所
枝刈りが左右対称だけなので、対象解除法に比べると枝刈りが弱い

seen.add(canonical)で最小値の形を保存しているが、nが多くなってくると大変なことになる
symmetryOps内で90度回転を4回やっているが、90,180,270を1回ずつやったほうがよりよい
"""
#pypyを使う場合はコメントを解除
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

from datetime import datetime
import zlib

def solve_n_queens_bitboard_partialDuplicate(n: int):
    seen_hashes = set()
    partial_seen = set()
    counts = {'COUNT2': 0, 'COUNT4': 0, 'COUNT8': 0}
    corner_counts = {'COUNT2': 0, 'COUNT4': 0, 'COUNT8': 0}
    noncorner_counts = {'COUNT2': 0, 'COUNT4': 0, 'COUNT8': 0}

    def rotate90(board: int, rows: int, cols: int) -> int:
        res = 0
        for i in range(rows):
            row = (board >> (i * cols)) & ((1 << cols) - 1)
            for j in range(cols):
                if row & (1 << j):
                    res |= 1 << ((cols - 1 - j) * rows + i)
        return res

    def mirror_vertical(board: int, rows: int, cols: int) -> int:
        res = 0
        for i in range(rows):
            row = (board >> (i * cols)) & ((1 << cols) - 1)
            mirrored = 0
            for j in range(cols):
                if row & (1 << j):
                    mirrored |= 1 << (cols - 1 - j)
            res |= mirrored << (i * cols)
        return res

    def get_partial_symmetries(board: int, row: int) -> list[int]:
        results = []
        r = board
        for _ in range(4):
            results.append(r)
            results.append(mirror_vertical(r, row, n))
            r = rotate90(r, row, n)
        return results

    def hash_board(board: int, bits: int) -> int:
        return zlib.crc32(board.to_bytes((bits + 7) // 8, byteorder='big'))

    def classify_symmetry(board: int, n: int, seen_hashes: set[int]) -> str:
        sym = get_partial_symmetries(board, n)
        hashes = [hash_board(s, n * n) for s in sym]
        canonical = min(hashes)
        if canonical in seen_hashes:
            return ""
        seen_hashes.add(canonical)
        distinct = len(set(hashes))
        return 'COUNT8' if distinct == 8 else 'COUNT4' if distinct == 4 else 'COUNT2'

    def is_partial_duplicate(board: int, row: int) -> bool:
        # 部分盤面（row行まで）での対称性重複チェック
        partial_bits = row * n
        partial_board = board & ((1 << partial_bits) - 1)
        sym = get_partial_symmetries(partial_board, row)
        hashes = [hash_board(s, partial_bits) for s in sym]
        canonical = min(hashes)
        if canonical in partial_seen:
            return True
        partial_seen.add(canonical)
        return False

    def backtrack(row=0, cols=0, hills=0, dales=0, board=0, is_corner=False):
        if row == n:
            cls = classify_symmetry(board, n, seen_hashes)
            if cls:
                counts[cls] += 1
                (corner_counts if is_corner else noncorner_counts)[cls] += 1
            return

        # コメントアウト
        # if row == 2:
        #     if is_partial_duplicate(board, row):
        #         return

        free = ~(cols | hills | dales) & ((1 << n) - 1)

        if row >= 2 and free == 0:
            return

        if row == n - 1 and is_corner:
            free &= ~(1 << (n - 1))

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
                is_corner=is_corner
            )

    def start():
        col = 0
        bit = 1 << col
        pos = col
        backtrack(1, bit, bit << 1, bit >> 1, 1 << pos, is_corner=True)
        for col in range(1, n // 2):
            bit = 1 << col
            pos = col
            backtrack(1, bit, bit << 1, bit >> 1, 1 << pos, is_corner=False)
        if n % 2 == 1:
            col = n // 2
            bit = 1 << col
            pos = col
            backtrack(1, bit, bit << 1, bit >> 1, 1 << pos, is_corner=False)

    start()
    total = counts['COUNT2'] * 2 + counts['COUNT4'] * 4 + counts['COUNT8'] * 8
    return total,sum(counts.values())

if __name__ == '__main__':
    from datetime import datetime
    _min = 4
    max = 17
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(_min, max):
        start_time = datetime.now()
        total,unique=solve_n_queens_bitboard_partialDuplicate(size)
        time_elapsed = datetime.now() - start_time
        text = str(time_elapsed)[:-3]
        print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")

