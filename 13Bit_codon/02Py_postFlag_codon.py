#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
ポストフラグ版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens
Bash版ですが内容は同じです。

fedora$ codon build -release 02Py_postFlag_codon.py && ./02Py_postFlag_codon 5 raw
:
:
40312: 76542130
40313: 76542300
40314: 76542310
40315: 76543010
40316: 76543020
40317: 76543100
40318: 76543120
40319: 76543200
40320: 76543210

real	0m0.916s
user	0m0.437s
sys	0m0.444s
fedora$

02Py_postFlag_codon.py（レビュー＆注釈つき）

目的:
- ユーザー提供の NQueens02 をレビューし、不具合の指摘と最小修正（MinimalFix）を実装。
- 併せて、列フラグに加えて対角フラグも導入した「N-Queens 正式版」(WithDiagonals) を提示。

要点:
1) 元コードのバグ: 終了条件が `if row == self.size - 1:` になっているため、
   最下行 (row=size-1) に到達した **時点で印字** してしまい、
   その行 (row=size-1) の列選択が未確定の状態で出力される。
   → 正しくは `if row == self.size:` で、直前の再帰で最後の配置が済んだ段階を“解”とする。

2) 元コードは列フラグ（fa）だけのため、列の重複は避けるが **斜め衝突は未チェック**。
   → これは「全ての列が一意な順列（Permutation）列挙」に相当し、N-Queens の解ではない。

3) ベンチマークや大 N の場合は print が高コスト。必要に応じて抑制を推奨。

本ファイルには2クラスを用意:
- NQueens02_MinimalFix: バグ修正のみ（列ユニークの順列列挙）。
- NQueens02_WithDiagonals: 列＋2方向対角のフラグを追加した、正しい N-Queens。

"""
from typing import List
import sys
import time

# ------------------------------------------------------------
# 1) 最小修正版（列ユニークの順列列挙）
#    - バグ修正: 終了条件 row==size → その直前に最後の配置が完了している。
#    - 斜め判定なし（元仕様を保持）。
# ------------------------------------------------------------
class NQueens02_MinimalFix:
    size: int
    count: int
    aboard: List[int]    # row -> col
    used_col: List[int]  # 列使用フラグ（0/1）

    def __init__(self, size: int) -> None:
        self.size = size
        self.count = 0
        self.aboard = [0 for _ in range(self.size)]
        self.used_col = [0 for _ in range(self.size)]

    def printout(self) -> None:
        self.count += 1
        print(self.count, end=": ")
        for i in range(self.size):
            print(self.aboard[i], end="")
        print("")

    def nqueens(self, row: int) -> None:
        # 正しい終了条件: row==size（最後の行も既に配置済みの状態）
        if row == self.size:
            self.printout()
            return
        # 各列を試す（列ユニーク制約のみ）
        for col in range(self.size):
            if self.used_col[col] == 0:
                self.aboard[row] = col
                self.used_col[col] = 1
                self.nqueens(row + 1)
                self.used_col[col] = 0


# ------------------------------------------------------------
# 2) 正式版（列＋対角のフラグで N-Queens を解く）
#    - 2 方向の対角フラグ: ld (左下↙︎/右上↗︎), rd (右下↘︎/左上↖︎)
#    - ld のインデックス: (row - col) を 0..(2N-2) にオフセット
#    - rd のインデックス: (row + col) を 0..(2N-2) にそのまま利用
# ------------------------------------------------------------
class NQueens02_WithDiagonals:
    size: int
    count: int
    aboard: List[int]
    used_col: List[int]
    used_ld: List[int]
    used_rd: List[int]
    offset: int

    def __init__(self, size: int) -> None:
        self.size = size
        self.count = 0
        self.aboard = [0 for _ in range(self.size)]
        self.used_col = [0 for _ in range(self.size)]
        self.used_ld = [0 for _ in range(2 * self.size - 1)]
        self.used_rd = [0 for _ in range(2 * self.size - 1)]
        self.offset = self.size - 1  # (row-col) の負値を 0 始まりにずらす

    def printout(self) -> None:
        self.count += 1
        print(self.count, end=": ")
        for i in range(self.size):
            print(self.aboard[i], end="")
        print("")

    def nqueens(self, row: int) -> None:
        if row == self.size:
            self.printout()
            return
        for col in range(self.size):
            ld = row - col + self.offset  # 0..2N-2
            rd = row + col                # 0..2N-2
            if (self.used_col[col] | self.used_ld[ld] | self.used_rd[rd]) == 0:
                self.aboard[row] = col
                self.used_col[col] = 1
                self.used_ld[ld] = 1
                self.used_rd[rd] = 1
                self.nqueens(row + 1)
                self.used_col[col] = 0
                self.used_ld[ld] = 0
                self.used_rd[rd] = 0


# ------------------------------------------------------------
# 3) CLI 入口
# ------------------------------------------------------------

def main() -> None:
    # 使い方:
    #   python3 02Py_postFlag_codon_reviewed.py N [raw]
    #   raw を指定すると MinimalFix（列ユニークの順列）を実行。
    #   省略時は WithDiagonals（N-Queens 正式版）を実行。
    n = 8
    mode = "proper"
    if len(sys.argv) >= 2:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print("第1引数 N は整数で指定してください。例: 8")
            return
    if len(sys.argv) >= 3 and sys.argv[2].lower() == "raw":
        mode = "raw"

    t0 = time.perf_counter()
    if mode == "raw":
        solver = NQueens02_MinimalFix(n)
        solver.nqueens(0)
        total = solver.count
    else:
        solver = NQueens02_WithDiagonals(n)
        solver.nqueens(0)
        total = solver.count
    t1 = time.perf_counter()

    print(f"\nMode: {mode}")
    print(f"N: {n}")
    print(f"Total: {total}")
    print(f"Elapsed: {t1 - t0:.3f}s")


if __name__ == "__main__":
    main()
