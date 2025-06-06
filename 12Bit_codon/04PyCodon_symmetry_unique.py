#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# 04 ミラー・回転対称解の個別表示付き 
# rotate() と v_mirror() で盤面を回転・反転します。
# 各解の「最小形（辞書順最小の対称形）」のみを記録してユニーク性を
# 判定します。ユニークな配置が見つかると、対称形（8パターン）をす
# べて表示します。表示されるのは「Q」でクイーンを示した盤面です。

# pypyを使う場合はコメントを解除
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

from datetime import datetime

def solve_n_queens_symmetry_unique(n:int)->list[int,int]:
  def rotate(board:list[int])->list[int]:
    """正しい90度回転：board[row] = col → new_board[col] = N - 1 - row"""
    n = len(board)
    new_board:list[int] = [0] * n
    for r in range(n):
      new_board[board[r]] = n - 1 - r
    return new_board
  def v_mirror(board:list[int])->list[int]:
    """左右反転"""
    return [len(board) - 1 - x for x in board]
  def generate_symmetries(board:list[int])->list[int]:
    """8つの対称形を返す"""
    boards = []
    b:list[int] = board[:]
    for _ in range(4):
      boards.append(tuple(b))
      boards.append(tuple(v_mirror(b)))
      b = rotate(b)
    return set(boards)
  unique_solutions:list[int] = set()
  total_solutions:int = [0]  # リストでmutableに
  def is_safe(queens:list[int],row:int,col:int)->bool:
    for r, c in enumerate(queens):
      if c == col or abs(r - row) == abs(c - col):
        return False
    return True
  def count_equiv(board:list[int])->int:
    symmetries = generate_symmetries(board)
    return 8 // len(symmetries)
  def backtrack(row:int,queens:list[int])->None:
    if row == n:
      symmetries = generate_symmetries(queens)
      min_form = min(symmetries)
      if min_form not in unique_solutions:
        unique_solutions.add(min_form)
        equiv_count = 8 // len(symmetries)
        total_solutions[0] += len(symmetries)
      return
    for col in range(n):
      if is_safe(queens, row, col):
        queens.append(col)
        backtrack(row + 1, queens)
        queens.pop()
  backtrack(0, [])
  return total_solutions[0],len(unique_solutions)

if __name__ == '__main__':
  _min:int=4; # min()を使っているためリネーム
  max:int=18
  print(" N:        Total       Unique         hh:mm:ss.ms")
  for size in range(_min,max):
    start_time=datetime.now();
    #
    total,unique=solve_n_queens_symmetry_unique(size)
    #
    time_elapsed=datetime.now()-start_time;
    text = str(time_elapsed)[:-3]
    print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")


