#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# 05 対称性分類付き N-Queens Solver（COUNT2, COUNT4, COUNT8）
# COUNT2: 自身と180度回転だけが同型（計2通り）
# COUNT4: 自身＋鏡像 or 回転を含めて4通りまでが同型
# COUNT8: 8通りすべてが異なる → 最も情報量が多い配置
# 実行結果の 全解 は対称形も含めた「解の総数」に一致します（n=8なら92）

# pypyを使う場合はコメントを解除
#import pypyjit
#pypyjit.set_param('max_unroll_recursion=-1')

import time


# 05 対称性分類付き N-Queens Solver（COUNT2, COUNT4, COUNT8）
# COUNT2: 自身と180度回転だけが同型（計2通り）
# COUNT4: 自身＋鏡像 or 回転を含めて4通りまでが同型
# COUNT8: 8通りすべてが異なる → 最も情報量が多い配置
# 実行結果の 全解 は対称形も含めた「解の総数」に一致します（n=8なら92）

def solve_n_queens_symmetry_knuth(n:int)->tuple[int,int]:
  counts = {'COUNT2': 0, 'COUNT4': 0, 'COUNT8': 0}
  unique_set = set()
  solutions = []

  def rotate(board:list[int],n:int):
    return [n - 1 - board.index(i) for i in range(n)]

  def v_mirror(board:list[int],n:int)->int:
    return [n - 1 - i for i in board]

  def reflect_all(board:list[int],n:int)->int:
=======
def solve_n_queens_symmetry_knuth(n: int) -> tuple[int, int]:
  unique_set: set[str] = set()  # 🔥 タプルじゃなくて str を使う
  count: list[int] = [0, 0, 0]  # [COUNT2, COUNT4, COUNT8]

  def rotate(board: list[int], n: int) -> list[int]:
    return [n - 1 - board.index(i) for i in range(n)]

  def v_mirror(board: list[int], n: int) -> list[int]:
    return [n - 1 - i for i in board]

  def reflect_all(board: list[int], n: int) -> list[list[int]]:
    #回転とミラーで8通りを生成
  def board_equals(a:int,b:int)->bool:
    return all(x == y for x, y in zip(a, b))

  def get_classification(board:list[int],n:int)->int:
    #8つの対称形を比較して分類（2,4,8通り）
    forms = reflect_all(board, n)
    canonical = min(forms)
    count = sum(1 for f in forms if board_equals(f, canonical))
    if count == 1:
      return 2  # COUNT8
    elif count == 2:
      return 1  # COUNT4
    else:
      return 0  # COUNT2

    for r, c in enumerate(queens):
      if c == col or abs(c - col) == abs(r - row):
        return False
    return True

  def backtrack(row: int, queens: list[int]):
    if row == n:
      canonical = min(reflect_all(queens, n))
      key = str(canonical)  # 🔥 文字列化して保存
      if key not in unique_set:
        unique_set.add(key)
        cls = get_classification(queens, n)
        count[cls] += 1
  _min:int=4; # min()を使っているためリネーム
  max:int=18
  print(" N:        Total       Unique         hh:mm:ss.ms")
  for size in range(_min,max):
    start_time=datetime.now();
    #
    total,unique=solve_n_queens_symmetry_knuth(size)
    #
    time_elapsed=datetime.now()-start_time;
    text = str(time_elapsed)[:-3]
    print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")


  _min: int = 4
  _max: int = 18
  print(" N:        Total       Unique         seconds")
  for size in range(_min, _max):
    start_time: float = time.perf_counter()
    total, unique = solve_n_queens_symmetry_knuth(size)
    time_elapsed: float = time.perf_counter() - start_time
    print(f"{size:2d}:{total:13d}{unique:13d}{time_elapsed:15.3f}")
