#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# 05 対称性分類付き N-Queens Solver（COUNT2, COUNT4, COUNT8）
# COUNT2: 自身と180度回転だけが同型（計2通り）
# COUNT4: 自身＋鏡像 or 回転を含めて4通りまでが同型
# COUNT8: 8通りすべてが異なる → 最も情報量が多い配置
# 実行結果の 全解 は対称形も含めた「解の総数」に一致します（n=8なら92）

#・クイーンを置けるかどうかの判定をis_safe()でやっているがこれが遅い
#・is_safeはqueens配列にいままで置いたクイーンの位置を設定する、
#クイーンを置くたびに,queens配列をforで回し、攻撃範囲に抵触しないかチェックしてる
#・クイーンを全部置き終わった段階で、symmetryOpsして、
#回転させて最小値チェックしてるが、枝刈りしてないので意味ない
#・symmetryOpsのやり方も、unique_set配列に最小値の形を保存しているが、
#nが多くなってくると大変なことになる
"""
CentOS$ python 05PyCodon_symmetry_knuth.py
 N:        Total       Unique         hh:mm:ss.ms
 4:            2            1         0:00:00.000
 5:           10            2         0:00:00.000
 6:            4            1         0:00:00.000
 7:           40            6         0:00:00.003
 8:           92           12         0:00:00.014
 9:          352           46         0:00:00.063
10:          724           92         0:00:00.304
11:         2680          341         0:00:01.654
12:        14200         1787         0:00:09.971
13:        73712         9233         0:01:01.902
14:       365596        45752         0:06:50.602

05 14:       365596        45752         0:06:50.602
04 14:       365596        45752         0:07:06.993
03 14:       365596            0         0:00:09.962
02 14:       365596            0         0:00:19.668
01 14:       365596            0         0:07:01.731
"""

#pypyを使う場合はコメントを解除
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')

from datetime import datetime

def solve_n_queens_symmetry_knuth(n: int) -> tuple[int, int]:
  unique_set: set[str] = set()  # 🔥 タプルじゃなくて str を使う
  count: list[int] = [0, 0, 0]  # [COUNT2, COUNT4, COUNT8]

  def rotate(board: list[int], n: int) -> list[int]:
    return [n - 1 - board.index(i) for i in range(n)]

  def v_mirror(board: list[int], n: int) -> list[int]:
    return [n - 1 - i for i in board]

  def reflect_all(board: list[int], n: int) -> list[list[int]]:
    #回転とミラーで8通りを生成
    result = []
    b = board[:]
    for _ in range(4):
      result.append(b[:])
      result.append(v_mirror(b, n))
      b = rotate(b, n)
    return result

  def board_equals(a: list[int], b: list[int]) -> bool:
    return all(x == y for x, y in zip(a, b))

  def get_classification(board: list[int], n: int) -> int:
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

  def is_safe(queens: list[int], row: int, col: int) -> bool:
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
      return
    for col in range(n):
      if is_safe(queens, row, col):
        queens.append(col)
        backtrack(row + 1, queens)
        queens.pop()

  backtrack(0, [])
  total: int = count[0]*2 + count[1]*4 + count[2]*8
  unique: int = count[0] + count[1] + count[2]
  return total, unique

if __name__ == '__main__':
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

