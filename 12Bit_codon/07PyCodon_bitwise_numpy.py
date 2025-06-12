#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 2. NumPyのままやる場合は「64ビット越え演算を避ける」
# n=8までしか使えませんし、Python標準intのほうが速いことも多いです

# 07 numPy対応 ビットボード版 N-Queens 分類カウント 
# np.uint64 により最大64ビットの高速ビット操作が可能
# Python標準 int の代わりに NumPy の uint64 を利用
# 対称性をもとに COUNT2 / COUNT4 / COUNT8 を分類
# 
# 処理項目	内容
# np.uint64	安全な64ビット符号なし整数。Pythonのintよりビット演算が高速＆明示的
# 回転処理	(i,j) の Q を (j, n-1-i) に変換しながらビット再配置
# 盤面エンコード	n×n 盤面を1つの64ビット整数に収める（最大 n=8）

#pypyを使う場合はコメントを解除
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

from datetime import datetime
import numpy as np

def solve_n_queens_bitboard_np(n):
  seen=set()
  counts = {'COUNT2': 0, 'COUNT4': 0, 'COUNT8': 0}

  def rotate90(board, n):
    res = np.uint64(0)
    for i in range(n):
      row = (board >> np.uint64(i * n)) & np.uint64((1 << n) - 1)
      for j in range(n):
        if row & (1 << j):
          res |= np.uint64(1) << np.uint64((n - 1 - j) * n + i)
    return res

  def mirror_vertical(board, n):
    res = np.uint64(0)
    for i in range(n):
      row = (board >> np.uint64(i * n)) & np.uint64((1 << n) - 1)
      mirrored_row = np.uint64(0)
      for j in range(n):
        if row & (1 << j):
          mirrored_row |= np.uint64(1) << np.uint64(n - 1 - j)
      res |= mirrored_row << np.uint64(i * n)
    return res

  def get_symmetries(board, n):
    results = []
    r = board
    for _ in range(4):
      results.append(r)
      results.append(mirror_vertical(r, n))
      r = rotate90(r, n)
    return results

  def classify_symmetry(board, n):
    syms = set(get_symmetries(board, n))
    sym_len = len(syms)
    if sym_len == 8:
      return 'COUNT8'
    elif sym_len == 4:
      return 'COUNT4'
    elif sym_len == 2:
      return 'COUNT2'

  def backtrack(row=0, cols=0, hills=0, dales=0, board=np.uint64(0)):
    if row == n:
      syms = get_symmetries(board, n)   # 8通りの対称形を取得
      canonical = int(min(syms))        # 最小のものを代表とする
      if canonical in seen:             # すでに出現済みならスキップ
        return
      seen.add(canonical)               # 新しいユニーク解として登録
      cls = classify_symmetry(board, n) # このままでOK
      counts[cls] += 1
      return
    free = ~(cols | hills | dales) & ((1 << n) - 1)
    while free:
      bit = free & -free
      free ^= bit
      col = bit.bit_length() - 1
      pos = np.uint64(row * n + col)
      backtrack(
          row + 1,
          cols | bit,
          (hills | bit) << 1,
          (dales | bit) >> 1,
          board | (np.uint64(1) << pos)
      )

  backtrack()

  total = counts['COUNT2'] * 2 + counts['COUNT4'] * 4 + counts['COUNT8'] * 8
  return total,sum(counts.values())

if __name__ == '__main__':
  _min:int=4; # min()を使っているためリネーム
  max:int=9
  print(" N:        Total       Unique         hh:mm:ss.ms")
  for size in range(_min,max):
    start_time=datetime.now();
    #
    total,unique=solve_n_queens_bitboard_np(size)
    #
    time_elapsed=datetime.now()-start_time;
    text = str(time_elapsed)[:-3]
    print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")

