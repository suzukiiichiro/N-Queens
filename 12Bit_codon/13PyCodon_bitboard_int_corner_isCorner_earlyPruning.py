#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 13  マクロチェス（局所パターン）による構築制限 
序盤の配置（例：1行目＋2行目）により、3行目以降のクイーン配置が詰まるパターン
特に cols | hills | dales が過半数を占めていると、有効配置がないことが多い
よって、2行目終了時点で pruning 条件を加えることで、無駄な探索を打ち切れる

現状の実装では？
cols, hills, dales は毎回正しくビット演算されており、
ただし row >= 2 以降に対して pruning 判定を入れていない
そのため、2行目で 致命的な配置があってもそのまま無駄に探索されている

✅マクロチェス（局所パターン）による構築制限
❌「ミラー＋90度回転」による構築時の重複排除
✅ 180度対称除去
✅ ビット演算による衝突枝刈り	同一列・対角線（↘ / ↙）との衝突を int のビット演算で高速除去	`free = ~(cols	hills
✅ 左右対称性除去	1行目のクイーンを左半分の列（0～n//2−1）に限定し、ミラー対称を除去	for col in range(n // 2):	済
✅ 中央列の特別処理（n奇数）	中央列は回転・ミラーで重複しないため個別に探索し、COUNT2分類に貢献	if n % 2 == 1: ブロック内で col = n // 2 を探索	済
✅ 角位置（col==0）とそれ以外で分岐	1行目の col == 0 を is_corner=True として分離し、COUNT2偏重を明示化	backtrack(..., is_corner=True) による分岐	済
✅ 対称性分類（COUNT2 / 4 / 8）	回転・反転の8通りから最小値を canonical にし、重複除去＆分類判定	len(set(symmetries)) による分類	済
"""

""" 13  マクロチェス（局所パターン）による構築制限 real    0m2.389s
#solve_n_queens_bitboard_int_corner_isCosrner_earlyPruning
特徴
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

def solve_n_queens_bitboard_int_corner_isCosrner_earlyPruning(n: int):
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
    free = ~(cols | hills | dales) & ((1 << n) - 1)
    # ✅ 安全な pruning（次の行にクイーンを置ける場所がない）
    if row >= 2 and free == 0:
      return

    # 🔧 回転180度対称の除去（角スタート時のみ）
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
    # col == 0（角）スタート
    col = 0
    bit = 1 << col
    pos = col
    backtrack(
        1,
        bit,
        bit << 1,
        bit >> 1,
        1 << pos,
        is_corner=True
    )
    # 左半分（1～n//2-1）
    for col in range(1, n // 2):
        bit = 1 << col
        pos = col
        backtrack(
            1,
            bit,
            bit << 1,
            bit >> 1,
            1 << pos,
            is_corner=False
        )
    # 中央列（n奇数のみ）
    if n % 2 == 1:
        col = n // 2
        bit = 1 << col
        pos = col
        backtrack(
            1,
            bit,
            bit << 1,
            bit >> 1,
            1 << pos,
            is_corner=False
        )
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
        total,unique=solve_n_queens_bitboard_int_corner_isCosrner_earlyPruning(size)
        time_elapsed = datetime.now() - start_time
        text = str(time_elapsed)[:-3]
        print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")

