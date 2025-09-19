#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
bit 対象解除版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

fedora$ codon build -release 07Py_bit_symmetry_codon.py && ./07Py_bit_symmetry_codon 
 N:        Total       Unique        hh:mm:ss.ms
 4:            2            1         0:00:00.000
 5:           10            2         0:00:00.000
 6:            4            1         0:00:00.000
 7:           40            6         0:00:00.000
 8:           92           12         0:00:00.000
 9:          352           46         0:00:00.000
10:          724           92         0:00:00.000
11:         2680          341         0:00:00.000
12:        14200         1787         0:00:00.003
13:        73712         9233         0:00:00.018
14:       365596        45752         0:00:00.092
15:      2279184       285053         0:00:00.411
16:     14772512      1846955         0:00:02.702
^C
fedora$ 

07Py_bit_symmetry_codon.py（レビュー＆注釈つき）

ユーザー提供の「COUNT2/4/8 による対称性分類付きビット探索」を、
Codon 互換・可読性重視で最小修正＆詳細コメントを付けた版です。

主な修正点:
- Codon実行を想定し、`pypyjit` 関連をコメントアウト（Codonでは未サポート）
- クラス先頭で **全フィールドを型付き宣言**（Codonの静的化要件）
- `mask` の毎回算出を避け、局所でだけ使う場所に限定
- `bitmap` のビット除去を `^=` で表記（= `& ~bit`）
- 範囲は `range(nmin, nmax)` のまま（元コード準拠）。必要なら `+1` に変更してください

アルゴリズム要約:
- 角にQがある/ないケースを分けてバックトラック（`backTrack1/2`）。
- 対称評価 `symmetryops()` で、90/180/270 と垂直ミラーにより COUNT2/4/8 を分類。
- 最後に `unique = c2 + c4 + c8`、`total = c2*2 + c4*4 + c8*8` を合成。

検算値（Total）: N=4→2, N=5→10, N=6→4, N=7→40, N=8→92, N=9→352, N=10→724 ...
（Unique は 1,2,? ... N=8 は 92/12 など）
"""
from datetime import datetime
from typing import List

# pypy を使う場合はコメントを解除（Codon では無効）
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')


class NQueens07:
    # --- 結果/状態（Codon向けに事前宣言） ---
    total: int
    unique: int
    board: List[int]
    size: int
    bound1: int
    bound2: int
    topbit: int
    endbit: int
    sidemask: int
    lastmask: int
    count2: int
    count4: int
    count8: int

    def __init__(self) -> None:
        # 実体は init(size) で与える
        pass

    # ------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------
    def init(self, size: int) -> None:
        self.total = 0
        self.unique = 0
        self.board = [0 for _ in range(size)]
        self.size = size
        self.bound1 = 0
        self.bound2 = 0
        self.topbit = 0
        self.endbit = 0
        self.sidemask = 0
        self.lastmask = 0
        self.count2 = 0
        self.count4 = 0
        self.count8 = 0

    # ------------------------------------------------------------
    # 対称性評価（COUNT2/4/8 の分類）
    # board は row→bit のビットボード列
    # ------------------------------------------------------------
    def symmetryops(self, size: int) -> None:
        """対象解除: 90/180/270回転 + 垂直反転 で代表性を判定して count2/4/8 を増やす。"""
        # --- 90度回転 ---
        if self.board[self.bound2] == 1:
            own: int = 1
            ptn: int = 2
            while own <= size - 1:
                bit: int = 1
                you: int = size - 1
                # board[you] を ptn に合わせつつ、board[own] と bit を進める
                while self.board[you] != ptn and self.board[own] >= bit:
                    bit <<= 1
                    you -= 1
                if self.board[own] > bit:
                    return
                if self.board[own] < bit:
                    break
                own += 1
                ptn <<= 1
            # 90度回転が同型
            if own > size - 1:
                self.count2 += 1
                return
        # --- 180度回転 ---
        if self.board[size - 1] == self.endbit:
            own = 1
            you = size - 2
            while own <= size - 1:
                bit = 1
                ptn = self.topbit
                while self.board[you] != ptn and self.board[own] >= bit:
                    bit <<= 1
                    ptn >>= 1
                if self.board[own] > bit:
                    return
                if self.board[own] < bit:
                    break
                own += 1
                you -= 1
            # 180度回転が同型
            if own > size - 1:
                self.count4 += 1
                return
        # --- 270度回転 ---
        if self.board[self.bound1] == self.topbit:
            own = 1
            ptn = self.topbit >> 1
            while own <= size - 1:
                bit = 1
                you = 0
                while self.board[you] != ptn and self.board[own] >= bit:
                    bit <<= 1
                    you += 1
                if self.board[own] > bit:
                    return
                if self.board[own] < bit:
                    break
                own += 1
                ptn >>= 1
        self.count8 += 1

    # ------------------------------------------------------------
    # 角に Q が「ない」場合の探索
    # ------------------------------------------------------------
    def backTrack2(self, size: int, row: int, left: int, down: int, right: int) -> None:
        mask: int = (1 << size) - 1
        bitmap: int = mask & ~(left | down | right)
        if row == (size - 1):
            if bitmap:
                # (bitmap & lastmask) == 0 のときのみ代表性チェックへ
                if (bitmap & self.lastmask) == 0:
                    self.board[row] = bitmap
                    self.symmetryops(size)
            return
        # 上辺・下辺・両辺の制約
        if row < self.bound1:
            # bitmap &= ~sidemask  を  (bitmap|sidemask) ^ sidemask で実装（分岐なしテク）
            bitmap = (bitmap | self.sidemask) ^ self.sidemask
        else:
            if row == self.bound2:
                if (down & self.sidemask) == 0:
                    return
                if (down & self.sidemask) != self.sidemask:
                    bitmap &= self.sidemask
        # 候補を 1 ビットずつ試す（LSB 抽出）
        while bitmap:
            bit = -bitmap & bitmap
            bitmap ^= bit
            self.board[row] = bit
            self.backTrack2(size, row + 1, (left | bit) << 1, (down | bit), (right | bit) >> 1)

    # ------------------------------------------------------------
    # 角に Q が「ある」場合の探索
    # ------------------------------------------------------------
    def backTrack1(self, size: int, row: int, left: int, down: int, right: int) -> None:
        mask: int = (1 << size) - 1
        bitmap: int = mask & ~(left | down | right)
        if row == (size - 1):
            if bitmap:
                self.board[row] = bitmap
                self.count8 += 1
            return
        if row < self.bound1:
            # bitmap &= ~2 を (bitmap|2) ^ 2 で実装
            bitmap = (bitmap | 2) ^ 2
        while bitmap:
            bit = -bitmap & bitmap
            bitmap ^= bit
            self.board[row] = bit
            self.backTrack1(size, row + 1, (left | bit) << 1, (down | bit), (right | bit) >> 1)

    # ------------------------------------------------------------
    # メイン探索のオーケストレーション
    # ------------------------------------------------------------
    def NQueens(self, size: int) -> None:
        self.total = 0
        self.unique = 0
        self.count2 = self.count4 = self.count8 = 0

        self.topbit = 1 << (size - 1)
        self.endbit = 0
        self.lastmask = 0
        self.sidemask = 0

        # --- 角に Q があるケース ---
        self.bound1 = 2
        self.bound2 = 0
        self.board[0] = 1
        while self.bound1 > 1 and self.bound1 < size - 1:
            if self.bound1 < (size - 1):
                bit = 1 << self.bound1
                self.board[1] = bit
                self.backTrack1(size, 2, (2 | bit) << 1, (1 | bit), (2 | bit) >> 1)
            self.bound1 += 1

        # --- 角に Q がないケース ---
        self.topbit = 1 << (size - 1)
        self.endbit = self.topbit >> 1
        self.sidemask = self.topbit | 1
        self.lastmask = self.sidemask
        self.bound1 = 1
        self.bound2 = size - 2
        while self.bound1 > 0 and self.bound2 < size - 1 and self.bound1 < self.bound2:
            if self.bound1 < self.bound2:
                bit = 1 << self.bound1
                self.board[0] = bit
                self.backTrack2(size, 1, bit << 1, bit, bit >> 1)
            self.bound1 += 1
            self.bound2 -= 1
            self.endbit >>= 1
            self.lastmask = (self.lastmask << 1) | self.lastmask | (self.lastmask >> 1)

        self.unique = self.count2 + self.count4 + self.count8
        self.total = self.count2 * 2 + self.count4 * 4 + self.count8 * 8

    # ------------------------------------------------------------
    # CLI 入口
    # ------------------------------------------------------------
    def main(self) -> None:
        nmin: int = 4
        nmax: int = 19
        print(" N:        Total       Unique        hh:mm:ss.ms")
        for size in range(nmin, nmax):
            self.init(size)
            start_time = datetime.now()
            self.NQueens(size)
            dt = datetime.now() - start_time
            text = str(dt)[:-3]
            print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")


if __name__ == '__main__':
    NQueens07().main()
