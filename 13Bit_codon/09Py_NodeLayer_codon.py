#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
ノードレイヤー版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

fedora$ codon build -release 09Py_NodeLayer_codon.py && ./09Py_NodeLayer_codon
 N:        Total       Unique        hh:mm:ss.ms
 4:            2            0         0:00:00.000
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.000
 9:          352            0         0:00:00.000
10:          724            0         0:00:00.000
11:         2680            0         0:00:00.001
12:        14200            0         0:00:00.007
13:        73712            0         0:00:00.053
14:       365596            0         0:00:00.236
15:      2279184            0         0:00:01.423
16:     14772512            0         0:00:09.315
fedora$ 


09Py_NodeLayer_codon.py（レビュー＆注釈つき）

ノードレイヤー手法：
- 深さ k 行ぶんだけ探索して "部分状態（left, down, right）" をノード配列に蓄積。
- 以降は各ノードを独立に完全探索して合計（並列化しやすい）。

この修正版では以下を実施：
1) 行レベルの詳細コメント／Docstring を付加。
2) Codon を意識した型注釈の明確化。
3) `mask` の再計算を極力避けられるよう、引数で渡す版（内部ラッパを追加）。
4) 命名の整備（関数名や変数名の意味をコメントに付記）。

シフト規約（元コード踏襲）：
- 次行では `left` を **>> 1**、`right` を **<< 1** へ伝播（一般的な表記の逆だが、
  `bitmap = mask & ~(left | down | right)` とセットで一貫していれば正しい）。

k のデフォルトは 4（N>=4 を想定）。k を大きくするとノード数は増えるが、その分、
以降のサブツリーが浅くなる。マシンや N に応じてチューニングしてください。
"""
from datetime import datetime
from typing import List, Tuple


class NQueens_NodeLayer:
    """ノードレイヤー法で Total 解数を数える（Unique は未算出）。"""

    # ------------------------------------------------------------
    # 完全探索（部分状態 → 葉まで）。down==mask で解 1 件。
    # ------------------------------------------------------------
    def _solve_from_node(self, size: int, mask: int, left: int, down: int, right: int) -> int:
        if down == mask:
            return 1
        total = 0
        bitmap: int = mask & ~(left | down | right)
        while bitmap:
            bit: int = -bitmap & bitmap
            bitmap ^= bit
            total += self._solve_from_node(size, mask, (left | bit) >> 1, down | bit, (right | bit) << 1)
        return total

    # ------------------------------------------------------------
    # down の set bit 数を返す（Brian Kernighan 法）。
    # ------------------------------------------------------------
    @staticmethod
    def _popcount(n: int) -> int:
        cnt = 0
        while n:
            n &= n - 1
            cnt += 1
        return cnt

    # ------------------------------------------------------------
    # 深さ k まで探索してノード（left,down,right の 3-tuple）を蓄積。
    # nodes には [l0,d0,r0,l1,d1,r1,...] の順で push する（元コード互換）。
    # ------------------------------------------------------------
    def _collect_nodes(self, size: int, mask: int, k: int, nodes: List[int],
                        left: int, down: int, right: int) -> int:
        # すでに k 行ぶん置けているか？（down の set bit 数で判定）
        if self._popcount(down) == k:
            nodes.append(left)
            nodes.append(down)
            nodes.append(right)
            return 1
        total = 0
        bitmap: int = mask & ~(left | down | right)
        while bitmap:
            bit: int = -bitmap & bitmap
            bitmap ^= bit
            total += self._collect_nodes(size, mask, k, nodes,
                                         (left | bit) >> 1, down | bit, (right | bit) << 1)
        return total

    # ------------------------------------------------------------
    # ノードレイヤー探索の外側：k を固定して frontier を作り、各ノードから完全探索
    # ------------------------------------------------------------
    def solve_with_layer(self, size: int, k: int = 4) -> int:
        if size < 1:
            return 0
        mask: int = (1 << size) - 1
        nodes: List[int] = []
        # 深さ k の frontier を構築
        self._collect_nodes(size, mask, k, nodes, 0, 0, 0)
        # 3 要素で 1 ノード
        num_nodes: int = len(nodes) // 3
        total = 0
        # 各ノードを独立に探索（ここは将来的に並列化ポイント）
        for i in range(num_nodes):
            l = nodes[3 * i]
            d = nodes[3 * i + 1]
            r = nodes[3 * i + 2]
            total += self._solve_from_node(size, mask, l, d, r)
        return total


# ------------------------------------------------------------
# CLI（元コード互換）
# ------------------------------------------------------------
class NQueens_NodeLayer_CLI:
    def main(self) -> None:
        nmin: int = 4
        nmax: int = 18
        print(" N:        Total       Unique        hh:mm:ss.ms")
        for size in range(nmin, nmax):
            start = datetime.now()
            solver = NQueens_NodeLayer()
            total = solver.solve_with_layer(size, k=4)
            unique = 0
            dt = datetime.now() - start
            text = str(dt)[:-3]
            print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")


if __name__ == "__main__":
    NQueens_NodeLayer_CLI().main()
