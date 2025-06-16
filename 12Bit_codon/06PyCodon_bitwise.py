#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 06 ビットボードによる対称性分類
# ビットボード（整数）で表現されたN-Queensの配置を、90度回転、180度回転、270度回転、左右反転（ミラー）のビット演算で処理し、同一性判定を高速に行って COUNT2, COUNT4, COUNT8 を分類する。
#
# 例） 4x4 の配置 [1, 3, 0, 2]
# 盤面：
# . Q . .
# . . . Q
# Q . . .
# . . Q .
#
# → 各行で Q のある位置にビット立てる
# → 0100（1<<2）, 0001（1<<0）, ... を結合して整数配列に
#
# ※ ただし行ではなく、列の配置を使えば1つの `n` ビット整数で列位置が表現できる
#
# board = [1, 3, 0, 2] などを sum(1 << (n * row + col)) にして1整数表現**全体が「1整数による圧縮ビットボード設計」**になっています。

""" 06 ビットボードによる対称性分類 real    0m5.642s
# solve_n_queens_bitwise_classification(13)
長所
boardを１個のビット列にしている。symmetryOpsの回転、反転もバラさないままできている
短所
クイーンを全部置き終わった段階で、symmetryOpsして、
回転させて最小値チェックしてるが、枝刈りしてないので意味ない
"""

#pypyを使う場合はコメントを解除
#import pypyjit
#pypyjit.set_param('max_unroll_recursion=-1')

from datetime import datetime

def solve_n_queens_bitwise_classification(n):
  seen = set()
  counts = {'COUNT2': 0, 'COUNT4': 0, 'COUNT8': 0}

  def rotate90(board, n):
    result = 0
    for i in range(n):
      for j in range(n):
        if board & (1 << (i * n + j)):
          result |= 1 << ((n - 1 - j) * n + i)
    return result

  def rotate180(board, n):
    return rotate90(rotate90(board, n), n)

  def rotate270(board, n):
    return rotate90(rotate180(board, n), n)

  def mirror_vertical(board, n):
    result = 0
    for i in range(n):
      row = (board >> (i * n)) & ((1 << n) - 1)
      mirrored = 0
      for j in range(n):
        if row & (1 << j):
          mirrored |= 1 << (n - 1 - j)
      result |= mirrored << (i * n)
    return result

  def get_symmetries(board, n):
    """lambda を使わずに 8通りの対称形を生成"""
    syms = set()
    b0 = board
    b1 = rotate90(b0, n)
    b2 = rotate180(b0, n)
    b3 = rotate270(b0, n)
    syms.add(b0)
    syms.add(mirror_vertical(b0, n))
    syms.add(b1)
    syms.add(mirror_vertical(b1, n))
    syms.add(b2)
    syms.add(mirror_vertical(b2, n))
    syms.add(b3)
    syms.add(mirror_vertical(b3, n))
    return syms

  def backtrack(row=0, cols=0, hills=0, dales=0, board=0):
    if row == n:
      symmetries = get_symmetries(board, n)
      canonical = min(symmetries)
      if canonical not in seen:
        seen.add(canonical)
        count = sum(1 for s in symmetries if s == canonical)
        if len(symmetries) == 8:
          counts['COUNT8'] += 1
        elif len(symmetries) == 4:
          counts['COUNT4'] += 1
        else:
          counts['COUNT2'] += 1
      return
    bits = ~(cols | hills | dales) & ((1 << n) - 1)
    while bits:
      bit = bits & -bits
      bits ^= bit
      pos = row * n + (bit.bit_length() - 1)
      """
      ここで pos = row * n + (bit.bit_length() - 1) なので、board は常に「1つの整数として、n×n盤面上のクイーン位置をビットで立てていく」方式です。つまり、board は以下の構造です：
      row0: 000...1...000  (← nビット)
      row1: 000...1...000
       ...
      rown: 000...1...000
      これらをまとめて、「row-major（行優先）で 1 つの整数に圧縮したビットボード」として保持しています。
      """
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
  _min:int=4; # min()を使っているためリネーム
  max:int=18
  print(" N:        Total       Unique         hh:mm:ss.ms")
  for size in range(_min,max):
    start_time=datetime.now();
    #
    total,unique=solve_n_queens_bitwise_classification(size)
    #
    time_elapsed=datetime.now()-start_time;
    text = str(time_elapsed)[:-3]
    print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")

