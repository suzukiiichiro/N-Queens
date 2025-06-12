#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 09 【枝刈り】構築時対称性除去 
項目	実装の有無	説明

1. ビット演算による衝突検出	✅ 済み
cols, hills, dales をビット演算で管理し、配置可能な列を `free = ~(cols

2. 構築時対称性除去	❌ 未実装
1行目の配置制限がなく、すべての列を試している（row = 0 時に for col in 0..n-1）ため、ミラー対称の枝刈りが行われていない
"""

""" 09【枝刈り】構築時対称性除去 real    0m2.402s
# solve_n_queens_bitboard_int_pruned01(13)
長所
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

def solve_n_queens_bitboard_int(n: int):
  seen = set()
  counts = {'COUNT2': 0, 'COUNT4': 0, 'COUNT8': 0}
  def rotate90(board: int, n: int) -> int:
    res = 0
    for i in range(n):
      row = (board >> (i * n)) & ((1 << n) - 1)
      for j in range(n):
        if row & (1 << j):
          res |= 1 << ((n - 1 - j) * n + i)
    return res
  def mirror_vertical(board: int, n: int) -> int:
    res = 0
    for i in range(n):
      row = (board >> (i * n)) & ((1 << n) - 1)
      mirrored_row = 0
      for j in range(n):
        if row & (1 << j):
          mirrored_row |= 1 << (n - 1 - j)
      res |= mirrored_row << (i * n)
    return res
  def get_symmetries(board: int, n: int) -> list[int]:
    results = []
    r = board
    for _ in range(4):
      results.append(r)
      results.append(mirror_vertical(r, n))
      r = rotate90(r, n)
    return results
  def classify_symmetry(board: int, n: int, seen: set[int]) -> str:
    sym = get_symmetries(board, n)
    canonical = min(sym)
    if canonical in seen:
      return ""
    seen.add(canonical)
    distinct = len(set(sym))
    if distinct == 8:
      return 'COUNT8'
    elif distinct == 4:
      return 'COUNT4'
    else:
      return 'COUNT2'
  def backtrack(row=0, cols=0, hills=0, dales=0, board=0):
    if row == n:
      cls = classify_symmetry(board, n, seen)
      if cls:
        counts[cls] += 1
      return
    if row == 0:
      limit = n // 2
      for col in range(limit):
        bit = 1 << col
        pos = row * n + col
        backtrack(
            row + 1,
            cols | bit,
            (hills | bit) << 1,
            (dales | bit) >> 1,
            board | (1 << pos)
        )
      if n % 2 == 1:
        col = n // 2
        bit = 1 << col
        pos = row * n + col
        backtrack(
            row + 1,
            cols | bit,
            (hills | bit) << 1,
            (dales | bit) >> 1,
            board | (1 << pos)
        )
    else:
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
              board | (1 << pos)
          )
  backtrack()
  total = counts['COUNT2'] * 2 + counts['COUNT4'] * 4 + counts['COUNT8'] * 8
  return total,sum(counts.values())

if __name__ == '__main__':
    from datetime import datetime
    _min = 4
    max = 17
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(_min, max):
        start_time = datetime.now()
        total,unique=solve_n_queens_bitboard_int(size)
        time_elapsed = datetime.now() - start_time
        text = str(time_elapsed)[:-3]
        print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")

