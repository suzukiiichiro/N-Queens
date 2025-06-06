#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# 05 対称性分類付き N-Queens Solver（COUNT2, COUNT4, COUNT8）
# COUNT2: 自身と180度回転だけが同型（計2通り）
# COUNT4: 自身＋鏡像 or 回転を含めて4通りまでが同型
# COUNT8: 8通りすべてが異なる → 最も情報量が多い配置
# 実行結果の 全解 は対称形も含めた「解の総数」に一致します（n=8なら92）

# pypyを使う場合はコメントを解除
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict

def rotate(board: List[int], n: int) -> List[int]:
    return [n - 1 - board.index(i) for i in range(n)]

def v_mirror(board: List[int], n: int) -> List[int]:
    return [n - 1 - i for i in board]

def reflect_all(board: List[int], n: int) -> List[List[int]]:
    result: List[List[int]] = []
    b = board[:]
    for _ in range(4):
        result.append(b[:])
        result.append(v_mirror(b, n))
        b = rotate(b, n)
    return result

def board_equals(a: List[int], b: List[int]) -> bool:
    return all(x == y for x, y in zip(a, b))

def get_classification(board: List[int], n: int) -> str:
    forms: List[List[int]] = reflect_all(board, n)
    canonical: List[int] = min(forms)
    count = 0
    for f in forms:
        if board_equals(f, canonical):
            count += 1
    if count == 1:
        return 'COUNT8'
    elif count == 2:
        return 'COUNT4'
    else:
        return 'COUNT2'

def is_safe(queens: List[int], row: int, col: int) -> bool:
    for r, c in enumerate(queens):
        if c == col or abs(c - col) == abs(r - row):
            return False
    return True

def solve_n_queens_symmetry_knuth(n: int) -> Tuple[int, int]:
    unique_set = set()
    # solutions: List[Tuple[str, List[int]]] = []
    # solutions:List[str]=[]
    counts: Dict[str, int] = {'COUNT2': 0, 'COUNT4': 0, 'COUNT8': 0}

    def backtrack(row: int, queens: List[int]) -> None:
        if row == n:
            forms: List[List[int]] = reflect_all(queens, n)
            canonical: List[int] = min(forms)
            key = tuple(canonical)
            if key not in unique_set:
                unique_set.add(key)
                cls: str = get_classification(queens, n)
                prev = counts[cls]
                counts[cls] = prev + 1
                # solutions.append((cls, queens[:]))
            return
        for col in range(n):
            if is_safe(queens, row, col):
                queens.append(col)
                backtrack(row + 1, queens)
                queens.pop()

    backtrack(0, [])
    total = counts['COUNT2'] * 2 + counts['COUNT4'] * 4 + counts['COUNT8'] * 8
    unique = counts['COUNT2'] + counts['COUNT4'] + counts['COUNT8']
    return total, unique

if __name__ == '__main__':
    _min: int = 4
    max: int = 18
    print(" N:        Total       Unique")
    for size in range(_min, max):
        result: Tuple[int, int] = solve_n_queens_symmetry_knuth(size)
        total, unique = result
        print(f"{size:2d}:{total:13d}{unique:13d}")

