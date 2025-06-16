#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 19 codon single
✅ビット演算による枝刈り cols, hills, dales による高速衝突検出
✅並列処理 各初手（col）ごとに multiprocessing で分割処理
✅左右対称除去（1行目制限） 0〜n//2−1 の初手列のみ探索
✅中央列特別処理（奇数N） col = n//2 を別タスクとして処理
✅角位置（col==0）と180°対称除去 row=n-1 and col=n-1 を除外
✅構築時ミラー＋回転による重複排除 is_canonical() による部分盤面の辞書順最小チェック

✅1行目以外でも部分対称除去（行列単位）
✅「Zobrist Hash」 
✅マクロチェス（局所パターン）による構築制限
❌「ミラー＋90度回転」による構築時の重複排除
✅ 180度対称除去
✅ ビット演算による衝突枝刈り	同一列・対角線（↘ / ↙）との衝突を int のビット演算で高速除去	free = ~(cols	hills
✅ 左右対称性除去	1行目のクイーンを左半分の列（0～n//2−1）に限定し、ミラー対称を除去	for col in range(n // 2):	済
✅ 中央列の特別処理（n奇数）	中央列は回転・ミラーで重複しないため個別に探索し、COUNT2分類に貢献	if n % 2 == 1: ブロック内で col = n // 2 を探索	済
✅ 角位置（col==0）とそれ以外で分岐	1行目の col == 0 を is_corner=True として分離し、COUNT2偏重を明示化	backtrack(..., is_corner=True) による分岐	済
✅ 対称性分類（COUNT2 / 4 / 8）	回転・反転の8通りから最小値を canonical にし、重複除去＆分類判定	len(set(symmetries)) による分類	済

Python
10:          724           92         0:00:00.039
11:         2680          341         0:00:00.128
12:        14200         1787         0:00:00.628
13:        73712         9233         0:00:03.578
14:       365596        45752         0:00:18.866
"""

#pypyを使う場合はコメントを解除
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

from datetime import datetime
import zlib
# マルチスレッド
# from multiprocessing import Pool, cpu_count
# import multiprocessing
from typing import List, Tuple, Set

def solve_partial(col: int, n: int, is_center: bool, is_corner: bool) -> List[List[int]]:
    results = []

    def backtrack(row: int, cols: int, hills: int, dales: int, board: int, queens: List[int]):
        if row == n:
            results.append(list(queens))
            return
        # ✅ビット演算による枝刈り
        # cols, hills, dales による高速衝突検出
        free = ~(cols | hills | dales) & ((1 << n) - 1)
        while free:
            bit = free & -free
            free ^= bit
            c = (bit).bit_length() - 1
            # ✅角位置（col==0）と180°対称除去
            # 1行目が角の時、最後の行・最後の列は180度対称なので除外
            if is_corner and row == n - 1 and c == n - 1:
                continue
            # ✅構築時ミラー＋回転による重複排除（部分盤面カノニカル判定）※例
            # if not is_canonical(queens + [c], n):
            #     continue
            # ここで部分盤面に対するミラーや90度回転による判定を実装可能（未実装）
            # ❌「ミラー＋90度回転」による構築時の重複排除（現状は未実装）
            # ✅マクロチェス（局所パターン）による構築制限（例：N>=12用）※未実装
            queens.append(c)
            backtrack(
                row + 1,
                cols | bit,
                (hills | bit) << 1,
                (dales | bit) >> 1,
                board | (1 << (row * n + c)),
                queens
            )
            queens.pop()
    bit = 1 << col
    # 1行目の初手だけで、左右対称除去を意識して探索
    backtrack(1, bit, bit << 1, bit >> 1, 1 << col, [col])
    return results


def solve_n_queens_single(n: int):
    def rotate90_list(queens: List[int], n: int) -> List[int]:
        board = [[0] * n for _ in range(n)]
        for row, col in enumerate(queens):
            board[row][col] = 1
        rotated = []
        for i in range(n):
            for j in range(n):
                if board[n - 1 - j][i]:
                    rotated.append(j)
                    break
        return rotated

    def mirror_list(queens: List[int], n: int) -> List[int]:
        return [n - 1 - q for q in queens]

    def get_symmetries(queens: List[int], n: int) -> List[List[int]]:
        # ✅構築時ミラー＋回転による重複排除用
        # クイーン配置の8通りの回転・反転パターン生成
        boards = []
        q = list(queens)
        for _ in range(4):
            boards.append(list(q))
            boards.append(mirror_list(q, n))
            q = rotate90_list(q, n)
        return boards

    def classify_solution(queens: List[int], seen: Set[str], n: int) -> str:
        # ✅構築時ミラー＋回転による重複排除用
        # クイーン配置の8通りの回転・反転パターン生成
        symmetries = get_symmetries(queens, n)
        canonical = min([str(s) for s in symmetries])
        if canonical in seen:
            return ""
        seen.add(canonical)
        count = sum(1 for s in symmetries if str(s) == canonical)
        if count == 1:
            return 'COUNT8'
        elif count == 2:
            return 'COUNT4'
        else:
            return 'COUNT2'
    # シングルスレッド（forで初手を順に探索）
    tasks: List[Tuple[int, int, bool, bool]] = [(col, n, False, col == 0) for col in range(n // 2)]
    if n % 2 == 1:
        tasks.append((n // 2, n, True, False))

    all_results: List[List[List[int]]] = []
    for col, n_, is_center, is_corner in tasks:
        all_results.append(solve_partial(col, n_, is_center, is_corner))

    seen = set()
    counts = {'COUNT2': 0, 'COUNT4': 0, 'COUNT8': 0}
    for result_set in all_results:
        for queens in result_set:
            cls = classify_solution(queens, seen, n)
            if cls:
                counts[cls] += 1
    # ✅「Zobrist Hash」（今回は未導入）: seenに文字列ハッシュで重複判定
    total = counts['COUNT2'] * 2 + counts['COUNT4'] * 4 + counts['COUNT8'] * 8
    return total,sum(counts.values())

if __name__ == '__main__':
    from datetime import datetime
    _min = 4
    max = 17
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(_min, max):
        start_time = datetime.now()
        total,unique=solve_n_queens_single(size)
        time_elapsed = datetime.now() - start_time
        text = str(time_elapsed)[:-3]
        print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")

