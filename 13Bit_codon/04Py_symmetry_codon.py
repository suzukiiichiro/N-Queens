#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
バックトラッキング 対象解除版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

fedora$ codon build -release 04Py_symmetry_codon.py && ./04Py_symmetry_codon 
 N:        Total       Unique         hh:mm:ss.ms
 4:            2            1         0:00:00.000
 5:           10            2         0:00:00.000
 6:            4            1         0:00:00.000
 7:           40            6         0:00:00.000
 8:           92           12         0:00:00.000
 9:          352           46         0:00:00.000
10:          724           92         0:00:00.000
11:         2680          341         0:00:00.005
12:        14200         1787         0:00:00.055
13:        73712         9233         0:00:00.137
14:       365596        45752         0:00:00.769
15:      2279184       285053         0:00:04.810

04Py_symmetry_codon.py（レビュー＆注釈つき）

ユーザー提供の NQueens04 をベースに、以下を実施：
 1) 行レベルを含む詳細コメントの付与（関数の目的を明示）
 2) 不具合/紛らわしさの最小修正（Codon互換も考慮）
    - `min`/`max` の変数名は組み込みと衝突するため `minN`/`maxN` に変更
    - ループ範囲を `range(minN, maxN + 1)`（18 を含む）
    - `fa` は未使用なので削除（列ユニークは順列生成で保証されている）
    - `rotate()` のロジックを読みやすく（`incr` を明示の ±1 に）
    - 関数引数の型注釈を補完（Codon の静的化の助け）

本段は「順列生成 + 対角フラグ + 対称性（D4: 回転/反転）」で Unique/Total を計測する段階です。
`nequiv` は 1,2,4 のいずれかで、最終的に `nequiv*2` を返し、これが multiplicity（2/4/8）になります。

Codon/Python 共通で動作。
"""
from datetime import datetime
from typing import List

# pypy を使う場合はコメントを解除（Codon では使わないこと）
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')


class NQueens04:
    # 結果カウンタ
    total: int
    unique: int
    # 作業配列
    aboard: List[int]   # row -> col（順列ベース：0..size-1 の並べ替え）
    fb: List[int]       # ld（左下↙︎/右上↗︎）対角フラグ: 0..(2*size-2)
    fc: List[int]       # rd（右下↘︎/左上↖︎）対角フラグ: 0..(2*size-2)
    trial: List[int]    # 対称操作時の作業バッファ
    scratch: List[int]  # 回転の中間バッファ
    _off: int           # (row-col) の負値補正 = size-1
    _size: int          # 参照用に保持（任意）

    def __init__(self) -> None:
        # 実体は init(size) で都度作る
        pass

    # ------------------------------------------------------------
    # 初期化：指定サイズの作業領域を確保
    # ------------------------------------------------------------
    def init(self, size: int) -> None:
        self.total = 0
        self.unique = 0
        # 順列ベース：初期は [0,1,2,...,size-1]
        self.aboard = [i for i in range(size)]
        # 対角フラグ（列ユニークは順列で保証されるため不要）
        self.fb = [0 for _ in range(2 * size - 1)]
        self.fc = [0 for _ in range(2 * size - 1)]
        # 対称操作用バッファ
        self.trial = [0 for _ in range(size)]
        self.scratch = [0 for _ in range(size)]
        # (row - col) の負値補正用
        self._off = size - 1
        self._size = size

    # ------------------------------------------------------------
    # 盤の 90° 回転（row->col 形式を保ったまま写像）。
    # neg=1: 右回り、neg=0: 左回り（実装上、逆順コピーかどうかのフラグ）
    # ------------------------------------------------------------
    def rotate(self, chk: List[int], scr: List[int], n: int, neg: int) -> None:
        # 第1段：scr に chk を順方向/逆方向でコピー
        incr = 1 if neg else -1
        k = 0 if neg else n - 1
        for i in range(n):
            scr[i] = chk[k]
            k += incr
        # 第2段：scr の値（= 列）を添字として使い、chk に新しい列を埋め戻す
        k = n - 1 if neg else 0
        for i in range(n):
            chk[scr[i]] = k
            k -= incr

    # ------------------------------------------------------------
    # 垂直反転：列を左右反転（row->col の col 値を反転）
    # ------------------------------------------------------------
    def vmirror(self, chk: List[int], n: int) -> None:
        for i in range(n):
            chk[i] = (n - 1) - chk[i]

    # ------------------------------------------------------------
    # 辞書順比較：左 < 右 → 負値、左 == 右 → 0、左 > 右 → 正値
    # ------------------------------------------------------------
    def intncmp(self, lt: List[int], rt: List[int], n: int) -> int:
        for i in range(n):
            d = lt[i] - rt[i]
            if d != 0:
                return d
        return 0

    # ------------------------------------------------------------
    # 対称性評価：
    #  - 回転（90/180/270）と垂直反転（およびその回転）を用い、
    #    self.aboard が最小表現かを判定し、等価解の倍率を返す。
    # 戻り値：0（最小でない＝代表ではない）/ 2 / 4 / 8
    # ------------------------------------------------------------
    def symmetryops(self, size: int) -> int:
        nequiv = 0
        # trial に原盤をコピー
        for i in range(size):
            self.trial[i] = self.aboard[i]
        # 90°
        self.rotate(self.trial, self.scratch, size, 0)
        k = self.intncmp(self.aboard, self.trial, size)
        if k > 0:
            return 0
        if k == 0:
            nequiv = 1
        else:
            # 180°
            self.rotate(self.trial, self.scratch, size, 0)
            k = self.intncmp(self.aboard, self.trial, size)
            if k > 0:
                return 0
            if k == 0:
                nequiv = 2
            else:
                # 270°
                self.rotate(self.trial, self.scratch, size, 0)
                k = self.intncmp(self.aboard, self.trial, size)
                if k > 0:
                    return 0
                nequiv = 4
        # 垂直反転
        for i in range(size):
            self.trial[i] = self.aboard[i]
        self.vmirror(self.trial, size)
        k = self.intncmp(self.aboard, self.trial, size)
        if k > 0:
            return 0
        # 垂直反転後の回転
        if nequiv > 1:
            # 90
            self.rotate(self.trial, self.scratch, size, 1)
            k = self.intncmp(self.aboard, self.trial, size)
            if k > 0:
                return 0
            if nequiv > 2:
                # 180
                self.rotate(self.trial, self.scratch, size, 1)
                k = self.intncmp(self.aboard, self.trial, size)
                if k > 0:
                    return 0
                # 270
                self.rotate(self.trial, self.scratch, size, 1)
                k = self.intncmp(self.aboard, self.trial, size)
                if k > 0:
                    return 0
        return nequiv * 2  # 1→2倍, 2→4倍, 4→8倍

    # ------------------------------------------------------------
    # 順列生成 + 対角フラグでのバックトラック
    #   - row==0 のときは左右対称を避けるため、列の上限を (size+1)//2 に制限
    # ------------------------------------------------------------
    def nqueens(self, row: int, size: int) -> None:
        if row == size - 1:
            # 最終行の候補が現在の aboard[row]。対角衝突だけ確認（列は順列で一意）
            if self.fb[row - self.aboard[row] + self._off] or self.fc[row + self.aboard[row]]:
                return
            stotal = self.symmetryops(size)
            if stotal != 0:
                self.unique += 1
                self.total += stotal
            return

        # 1 行目は左右対称を避けるため半分だけ試す
        lim = size if row != 0 else (size + 1) // 2
        for i in range(row, lim):
            # row と i をスワップして次の行へ（順列生成）
            tmp = self.aboard[i]
            self.aboard[i] = self.aboard[row]
            self.aboard[row] = tmp
            # 対角フラグ（列は不要）
            ld = row - self.aboard[row] + self._off
            rd = row + self.aboard[row]
            if (self.fb[ld] | self.fc[rd]) == 0:
                self.fb[ld] = 1
                self.fc[rd] = 1
                self.nqueens(row + 1, size)
                self.fb[ld] = 0
                self.fc[rd] = 0
        # row を末尾に回して、次の外側ループへ（順列生成の定型テク）
        tmp = self.aboard[row]
        for i in range(row + 1, size):
            self.aboard[i - 1] = self.aboard[i]
        self.aboard[size - 1] = tmp

    # ------------------------------------------------------------
    # CLI 入口
    # ------------------------------------------------------------
    def main(self) -> None:
        minN: int = 4
        maxN: int = 18
        print(" N:        Total       Unique         hh:mm:ss.ms")
        for size in range(minN, maxN + 1):
            self.init(size)
            start_time = datetime.now()
            self.nqueens(0, size)
            time_elapsed = datetime.now() - start_time
            text = str(time_elapsed)[:-3]
            print(f"{size:2d}:{self.total:13d}{self.unique:13d}{text:>20s}")


if __name__ == '__main__':
    NQueens04().main()
