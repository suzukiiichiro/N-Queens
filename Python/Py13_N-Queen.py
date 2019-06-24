#!/usr/bin/env python

import logging
import threading
from threading import Thread
from datetime import datetime
# -*- coding: utf-8 -*-
#
# /**
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#
#  実行
#  $ python Py13_N-Queen.py
#
#
# １３．マルチスレッドとマルチプロセス
#
#  シングルスレッド
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.000
#  5:           10            2         0:00:00.000
#  6:            4            1         0:00:00.000
#  7:           40            6         0:00:00.000
#  8:           92           12         0:00:00.000
#  9:          352           46         0:00:00.002
# 10:          724           92         0:00:00.010
# 11:         2680          341         0:00:00.053
# 12:        14200         1787         0:00:00.265
# 13:        73712         9233         0:00:01.420
# 14:       365596        45752         0:00:08.174
# 15:      2279184       285053         0:00:51.030
#
# マルチスレッドにて
# start/joinを連ねて遅くとも間違いなく実行
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.000
#  5:           10            2         0:00:00.000
#  6:            4            1         0:00:00.000
#  7:           40            6         0:00:00.000
#  8:           92           12         0:00:00.001
#  9:          352           46         0:00:00.004
# 10:          724           92         0:00:00.016
# 11:         2680          341         0:00:00.069
# 12:        14200         1787         0:00:00.386
# 13:        73712         9233         0:00:01.863
# 14:       365596        45752         0:00:10.437
# 15:      2279184       285053         0:01:06.639
#
# マルチスレッドにて
# joinを処理末でまとめて実行。速いけど計測値が違う!!
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.000
#  5:           10            2         0:00:00.001
#  6:            4            1         0:00:00.000
#  7:           40            6         0:00:00.000
#  8:           92           12         0:00:00.001
#  9:          352           46         0:00:00.004
# 10:          724           92         0:00:00.016
# 11:         2680          341         0:00:00.073
# 12:        14200         1787         0:00:00.362
# 13:        20172         2525         0:00:00.510
# 14:        29040         3630         0:00:00.713
# 15:         8840         1105         0:00:01.142
#
#
# マルチスレッド・シングルスレッドの切り換えフラグ
bThread = True    # マルチスレッド
#bThread=False    # シングルスレッド
#
# マルチスレッド時にjoin()するか
# 遅いけどstart()する度にjoin()する。これにより、処理ロジックに
# 間違いのないことが確認できる
#
#enableJoin = True   # joinする 遅いけど正しい計算結果
enableJoin=False     # joinしない 速いけどおかしい計算結果
#
#
class Board:
    def __init__(self, lock):
        self.COUNT2 = 0
        self.COUNT4 = 0
        self.COUNT8 = 0
        self.lock = lock
    def setCount(self, COUNT8, COUNT4, COUNT2):
        with self.lock:
            self.COUNT8 += COUNT8
            self.COUNT4 += COUNT4
            self.COUNT2 += COUNT2
    def getUnique(self):
        return self.COUNT2 + self.COUNT4 + self.COUNT8
    def getTotal(self):
        return self.COUNT2 * 2 + self.COUNT4 * 4 + self.COUNT8 * 8
#
class WorkingEngine(Thread):
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(levelname)s] (%(threadName)-10s) %(message)s', )
    def __init__(self, size, nMore, info, B1, B2, bThread):
        super(WorkingEngine, self).__init__()
        self.size=size
        self.sizeE=size-1
        self.aBoard = [0 for i in range(size)]
        self.MASK = (1 << size) - 1
        self.info=info
        self.nMore=nMore
        self.child=None
        self.BOUND1 = B1
        self.BOUND2 = B2
        self.bThread=bThread
        self.TOPBIT=0
        self.ENDBIT=0
        self.SIDEMASK=0
        self.LASTMASK=0
        for i in range(size):
            self.aBoard[i] = i
        if nMore > 0:
            if bThread:
                self.child = WorkingEngine(size, nMore - 1, info, B1 - 1, B2 + 1, bThread);
                self.BOUND1 = B1
                self.BOUND2 = B2
                self.child.start()
                if enableJoin:
                    self.child.join()
            else:
                self.child=None
                self.run()
    def run(self):
        if self.child is None:
            if self.nMore > 0:
                self.aBoard[0] = 1
                self.sizeE = self.size - 1
                self.TOPBIT = 1 << (self.size - 1)
                self.BOUND1 = 2
                while (self.BOUND1>1) and (self.BOUND1 < self.sizeE):
                    self.rec_BOUND1(self.BOUND1)
                    self.BOUND1 += 1
                self.SIDEMASK = self.LASTMASK = (self.TOPBIT | 1)
                self.ENDBIT = (self.TOPBIT >> 1)
                self.BOUND1 = 1
                self.BOUND2 = self.size - 2
                while (self.BOUND1>0) and (self.BOUND2 < self.size - 1) and (self.BOUND1 < self.BOUND2):
                    self.rec_BOUND2(self.BOUND1, self.BOUND2)
                    self.BOUND1 += 1
                    self.BOUND2 -= 1
        else:
            self.aBoard[0] = 1
            self.sizeE = self.size - 1
            self.MASK = (1 << self.size) - 1
            self.TOPBIT = 1 << self.sizeE
            if (self.BOUND1 > 1) and (self.BOUND1 < self.sizeE):
                self.rec_BOUND1(self.BOUND1)
            self.ENDBIT = (self.TOPBIT >> self.BOUND1)
            self.SIDEMASK = self.LASTMASK = (self.TOPBIT | 1)
            if (self.BOUND1 > 0 ) and ( self.BOUND2 < self.size - 1) and (self.BOUND1 < self.BOUND2):
                for i in range(1, self.BOUND1):
                    self.LASTMASK = self.LASTMASK | self.LASTMASK >> 1 | self.LASTMASK << 1
                self.rec_BOUND2(self.BOUND1, self.BOUND2)
                self.ENDBIT >>=self.nMore
            if enableJoin:
                pass
            else:
                main_thread = threading.currentThread()
                for t in threading.enumerate():
                    if t is not main_thread:
                        t.join()
    def symmetryOps(self):
        # 90度回転
        if self.aBoard[self.BOUND2] == 1:
            own = 1
            ptn = 2
            while own <= self.size - 1:
                bit = 1
                you = self.size - 1
                while (self.aBoard[you] != ptn) and (self.aBoard[own] >= bit):
                    bit <<= 1
                    you -= 1
                if self.aBoard[own] > bit:
                    return
                if self.aBoard[own] < bit:
                    break
                own += 1
                ptn <<= 1
            # 90度回転して同型なら180度/270度回転も同型である */
            if own > self.size - 1:
                self.info.setCount(0, 0, 1)
                return
        # 180度回転
        if self.aBoard[self.size - 1] == self.ENDBIT:
            own = 1
            you = self.size - 1 - 1
            while own <= self.size - 1:
                bit = 1
                ptn = self.TOPBIT
                while (self.aBoard[you] != ptn) and (self.aBoard[own] >= bit):
                    bit <<= 1
                    ptn >>= 1
                if self.aBoard[own] > bit:
                    return
                if self.aBoard[own] < bit:
                    break
                own += 1
                you -= 1
            # 90度回転が同型でなくても180度回転が同型である事もある */
            if own > self.size - 1:
                self.info.setCount(0, 1, 0)
                return
        # 270度回転
        if self.aBoard[self.BOUND1] == self.TOPBIT:
            own = 1
            ptn = self.TOPBIT >> 1
            while own <= self.size - 1:
                bit = 1
                you = 0
                while (self.aBoard[you] != ptn) and (self.aBoard[own] >= bit):
                    bit <<= 1
                    you += 1
                if self.aBoard[own] > bit:
                    return
                if self.aBoard[own] < bit:
                    break
                own += 1
                ptn >>= 1
        self.info.setCount(1, 0, 0)
    def backTrack2(self, row, left, down, right):
        bitmap = self.MASK & ~(left | down | right)
        # 枝刈り
        if row == self.size - 1:
            if bitmap:
                # 枝刈り
                if (bitmap & self.LASTMASK) == 0:
                    self.aBoard[row] = bitmap
                    self.symmetryOps()
        else:
            if row < self.BOUND1:
                bitmap &= ~self.SIDEMASK
            elif row == self.BOUND2:
                if down & self.SIDEMASK == 0:
                    return
                if down & self.SIDEMASK != self.SIDEMASK:
                    bitmap &= self.SIDEMASK
            # 枝刈り
            if row != 0:
                lim = self.size
            else:
                lim = (self.size + 1) // 2  # 割り算の結果を整数にするには //
            # 枝刈り
            for i in range(row, lim):
                while bitmap:
                    bit = (-bitmap & bitmap)
                    self.aBoard[row] = bit
                    bitmap ^= self.aBoard[row]
                    self.backTrack2(row + 1, (left | bit) << 1, down | bit, (right | bit) >> 1)
    def backTrack1(self, row, left, down, right):
        bitmap = self.MASK & ~(left | down | right)
        if row == self.size - 1:
            if bitmap:
                self.aBoard[row] = bitmap
                self.info.setCount(1, 0, 0)
        else:
            if row < self.BOUND1:
                bitmap &= ~2  # bm|=2 bm^=2 (bm&=~2と同等)
            if row != 0:
                lim = self.size
            else:
                lim = (self.size + 1) // 2  # 割り算の結果を整数にするには //
            for i in range(row, lim):
                while bitmap:
                    bit = (-bitmap & bitmap)
                    self.aBoard[row] = bit
                    bitmap ^= self.aBoard[row]
                    self.backTrack1(row + 1, (left | bit) << 1, down | bit, (right | bit) >> 1)
    def rec_BOUND2(self, B1, B2):
        self.BOUND1 = B1
        self.BOUND2 = B2
        self.aBoard[0] = bit = (1 << B1)
        self.backTrack2(1, bit << 1, bit, bit >> 1)
        self.LASTMASK |= self.LASTMASK >> 1 | self.LASTMASK << 1
        self.ENDBIT >>= 1
    def rec_BOUND1(self, B1):
        self.BOUND1 = B1
        self.aBoard[1] = bit = (1 << B1)
        self.backTrack1(2, (2 | bit) << 1, (1 | bit), bit >> 1)
    def NQueen(self):
        if self.child is None:
            if self.nMore > 0:
                self.TOPBIT = 1 << (self.size - 1)
                self.aBoard[0] = 1
                self.sizeE = self.size - 1
                self.BOUND1 = 2
                while (self.BOUND1 > 1) and (self.BOUND1 < self.sizeE):
                    self.rec_BOUND1(self.BOUND1)
                    self.BOUND1 += 1
                self.SIDEMASK = self.LASTMASK = (self.TOPBIT | 1)
                self.ENDBIT = (self.TOPBIT >> 1)
                self.BOUND2 = self.size - 2
                self.BOUND1 = 1
                while (self.BOUND1 > 0 )and (self.BOUND2 < self.size - 1 ) and ( self.BOUND1 < self.BOUND2):
                    self.rec_BOUND2(self.BOUND1, self.BOUND2)
                    self.BOUND1 += 1
                    self.BOUND2 -= 1
def main():
    max = 16
    min = 4  # Nの最小値（スタートの値）を格納
    if bThread:
        print("マルチスレッドにて")
        if enableJoin:
            print("start/joinを連ねて遅くとも間違いなく実行")
        else:
            print("joinを処理末でまとめて実行。速いけど計測値が違う!!")
    else:
        print("シングルスレッドにて実行")
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for i in range(min, max):
        lock = threading.Lock()
        info = Board(lock)  # ボードクラス
        start_time = datetime.now()
        child = threading.Thread(target=WorkingEngine, args=(i, i, info, i - 1, 0, bThread), name='main().thread')
        child.start()
        child.join()
        time_elapsed = datetime.now() - start_time
        _text = '{}'.format(time_elapsed)
        text = _text[:-3]
        print("%2d:%13d%13d%20s" % (i, info.getTotal(), info.getUnique(), text))  # 出力
#
main()
#

