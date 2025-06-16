#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 10【枝刈り】1行目の角（列0）にクイーンを置いた場合を別処理で分離する戦略的枝刈り
Knuth も推奨している有効な最適化です。これにより、探索空間をより戦略的に分割し、解の対称性分類（COUNT2 / 4 / 8）の分布を 構築前から制御できます。

✅ 各最適化と対応状況のまとめ
最適化・枝刈り手法	内容	実装箇所または対応方法	対応状況
✅ ビット演算による衝突枝刈り	同一列・対角線（↘ / ↙）との衝突を int のビット演算で高速除去	`free = ~(cols	hills
✅ 左右対称性除去	1行目のクイーンを左半分の列（0～n//2−1）に限定し、ミラー対称を除去	for col in range(n // 2):	済
✅ 中央列の特別処理（n奇数）	中央列は回転・ミラーで重複しないため個別に探索し、COUNT2分類に貢献	if n % 2 == 1: ブロック内で col = n // 2 を探索	済
✅ 角位置（col==0）とそれ以外で分岐	1行目の col == 0 を is_corner=True として分離し、COUNT2偏重を明示化	backtrack(..., is_corner=True) による分岐	済
✅ 対称性分類（COUNT2 / 4 / 8）	回転・反転の8通りから最小値を canonical にし、重複除去＆分類判定	len(set(symmetries)) による分類	済
"""

""" 10 1行目の角（列0）にクイーンを置いた場合を別処理で分離する戦略的枝刈りreal    0m2.295s
# solve_n_queens_bitboard_int_corner(13)
特徴
1行目の角（列0）にクイーンを置いた場合を別処理している。ただ、別処理しているだけで枝刈りはしてない。フラグだけたててる（1行目の角（列0）にクイーンを置いた場合はdistinct = len(set(sym))せずCOUNT8してよい（最小値チェックは必要））。

長所
boardを１個のビット列にしている。symmetryOpsの回転、反転もバラさないままできている
左右対称の枝刈りをやっている（１行目は半分の列だけクイーンを置く）
COUNT2,4,8の判定distinct = len(set(sym))はcount = sum(1 for s in sym if s == canonical)よりは速い

短所
枝刈りが左右対称だけなので、対象解除法に比べると枝刈りが弱い

seen.add(canonical)で最小値の形を保存しているが、nが多くなってくると大変なことになる
symmetryOps内で90度回転を4回やっているが、90,180,270を1回ずつやったほうがよりよい

CentOS$ python 10PyCodon_bitboard_int_corner.py
 N:        Total       Unique         hh:mm:ss.ms
 4:            2            1         0:00:00.000
 5:           10            2         0:00:00.000
 6:            4            1         0:00:00.000
 7:           40            6         0:00:00.002
 8:           92           12         0:00:00.005
 9:          352           46         0:00:00.027
10:          724           92         0:00:00.063
11:         2680          341         0:00:00.303
12:        14200         1787         0:00:01.618
13:        73712         9233         0:00:10.110
14:       365596        45752         0:00:51.604

10 14:       365596        45752         0:00:51.604
09 14:       365596        45752         0:00:51.615
08 14:       365596        45752         0:03:36.281
07  8:           92           12         0:00:00.027
06 14:       365596        45752         0:02:11.464
05 14:       365596        45752         0:06:50.602
04 14:       365596        45752         0:07:06.993
03 14:       365596            0         0:00:09.962
02 14:       365596            0         0:00:19.668
01 14:       365596            0         0:07:01.731
"""

#pypyを使う場合はコメントを解除
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

from datetime import datetime

def solve_n_queens_bitboard_int_corner(n: int):
  seen = set()
  counts = {'COUNT2': 0, 'COUNT4': 0, 'COUNT8': 0}
  corner_counts = {'COUNT2': 0, 'COUNT4': 0, 'COUNT8': 0}
  noncorner_counts = {'COUNT2': 0, 'COUNT4': 0, 'COUNT8': 0}

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

  def backtrack(row=0, cols=0, hills=0, dales=0, board=0, is_corner=False):
    if row == n:
      cls = classify_symmetry(board, n, seen)
      if cls:
        counts[cls] += 1
        if is_corner:
          corner_counts[cls] += 1
        else:
          noncorner_counts[cls] += 1
      return
    if row == 0:
      # 🔷 角（列0）の特別処理
      col = 0
      bit = 1 << col
      pos = row * n + col
      backtrack(
          row + 1,
          cols | bit,
          (hills | bit) << 1,
          (dales | bit) >> 1,
          board | (1 << pos),
          is_corner=True
      )
      # 🔷 左半分（1〜n//2-1）
      for col in range(1, n // 2):
        bit = 1 << col
        pos = row * n + col
        backtrack(
            row + 1,
            cols | bit,
            (hills | bit) << 1,
            (dales | bit) >> 1,
            board | (1 << pos),
            is_corner=False
        )
      # 🔷 中央列（奇数Nのみ）
      if n % 2 == 1:
        col = n // 2
        bit = 1 << col
        pos = row * n + col
        backtrack(
            row + 1,
            cols | bit,
            (hills | bit) << 1,
            (dales | bit) >> 1,
            board | (1 << pos),
            is_corner=False
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
            board | (1 << pos),
            is_corner=is_corner
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
        total,unique=solve_n_queens_bitboard_int_corner(size)
        time_elapsed = datetime.now() - start_time
        text = str(time_elapsed)[:-3]
        print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")

