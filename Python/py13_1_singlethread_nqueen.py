#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" py13a_nqueen.py """
import logging
import threading
from threading import Thread
from datetime import datetime
#
#
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#
#  実行
#  $ python py13a_nqueen.py
#
#
# １３a．シングルスレッドの構築
#       マルチスレッドの構築の準備としてシングルスレッドを構築します
#
#
# シングルスレッドにて実行
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.002
#  5:           10            2         0:00:00.000
#  6:            4            1         0:00:00.000
#  7:           40            6         0:00:00.001
#  8:           92           12         0:00:00.002
#  9:          352           46         0:00:00.011
# 10:          724           92         0:00:00.044
# 11:         2680          341         0:00:00.192
# 12:        14200         1787         0:00:00.937
# 13:        73712         9233         0:00:05.205
# 14:       365596        45752         0:00:28.879
# 15:      2279184       285053         0:03:00.068
#
#
class Board:
    """ Board """
    #
    def __init__(self, lock):
        """ __init__ """
        self.count2 = 0
        self.count4 = 0
        self.count8 = 0
        self.lock = lock
    #
    def setcount(self, count8, count4, count2):
        """ setcount() """
        with self.lock:
            self.count8 += count8
            self.count4 += count4
            self.count2 += count2
    #
    def getunique(self):
        """ getunique() """
        return self.count2 + self.count4 + self.count8
    #
    def gettotal(self):
        """ gettotal() """
        return self.count2 * 2 + self.count4 * 4 + self.count8 * 8
#
class WorkingEngine(Thread): # pylint: disable=R0902
    """ WorkingEngine """
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(levelname)s] (%(threadName)-10s) %(message)s', )
    #
    def __init__(self, size, nmore, info, B1, B2): # pylint: disable=R0913
        """ ___init___ """
        super(WorkingEngine, self).__init__()
        self.size = size
        self.sizee = size-1
        self.aboard = [0 for i in range(size)]
        self.mask = (1 << size) - 1
        self.info = info
        self.nmore = nmore
        self.child = None
        self.bound1 = B1
        self.bound2 = B2
        self.topbit = 0
        self.endbit = 0
        self.sidemask = 0
        self.lastmask = 0
        for i in range(size):
            self.aboard[i] = i
    #
    def run(self):
        # シングルスレッド
        if self.child is None:
            if self.nmore > 0:
                self.aboard[0] = 1
                self.sizee = self.size - 1
                self.topbit = 1 << (self.size - 1)
                self.bound1 = 2
                while (self.bound1 > 1) and (self.bound1 < self.sizee):
                    self.rec_bound1(self.bound1)
                    self.bound1 += 1
                self.sidemask = self.lastmask = (self.topbit | 1)
                self.endbit = (self.topbit >> 1)
                self.bound1 = 1
                self.bound2 = self.size - 2
                while (self.bound1 > 0) and (self.bound2 < self.size - 1) and (self.bound1 < self.bound2):  # pylint: disable=C0301
                    self.rec_bound2(self.bound1, self.bound2)
                    self.bound1 += 1
                    self.bound2 -= 1
    #
    def symmetryops(self):  # pylint: disable=R0912,R0911,R0915
        """ symmetryops() """
        # 90度回転
        if self.aboard[self.bound2] == 1:
            own = 1
            ptn = 2
            while own <= self.size - 1:
                bit = 1
                you = self.size - 1
                while (self.aboard[you] != ptn) and (self.aboard[own] >= bit):
                    bit <<= 1
                    you -= 1
                if self.aboard[own] > bit:
                    return
                if self.aboard[own] < bit:
                    break
                own += 1
                ptn <<= 1
            # 90度回転して同型なら180度/270度回転も同型である */
            if own > self.size - 1:
                self.info.setcount(0, 0, 1)
                return
        # 180度回転
        if self.aboard[self.size - 1] == self.endbit:
            own = 1
            you = self.size - 1 - 1
            while own <= self.size - 1:
                bit = 1
                ptn = self.topbit
                while (self.aboard[you] != ptn) and (self.aboard[own] >= bit):
                    bit <<= 1
                    ptn >>= 1
                if self.aboard[own] > bit:
                    return
                if self.aboard[own] < bit:
                    break
                own += 1
                you -= 1
            # 90度回転が同型でなくても180度回転が同型である事もある */
            if own > self.size - 1:
                self.info.setcount(0, 1, 0)
                return
        # 270度回転
        if self.aboard[self.bound1] == self.topbit:
            own = 1
            ptn = self.topbit >> 1
            while own <= self.size - 1:
                bit = 1
                you = 0
                while (self.aboard[you] != ptn) and (self.aboard[own] >= bit):
                    bit <<= 1
                    you += 1
                if self.aboard[own] > bit:
                    return
                if self.aboard[own] < bit:
                    break
                own += 1
                ptn >>= 1
        self.info.setcount(1, 0, 0)
    #
    def backtrack2(self, row, left, down, right):
        """ backtrack2() """
        bitmap = self.mask & ~(left | down | right)
        # 枝刈り
        if row == self.size - 1:
            if bitmap:
                # 枝刈り
                if (bitmap & self.lastmask) == 0:
                    self.aboard[row] = bitmap
                    self.symmetryops()
        else:
            if row < self.bound1:
                bitmap &= ~self.sidemask
            elif row == self.bound2:
                if down & self.sidemask == 0:
                    return
                if down & self.sidemask != self.sidemask:
                    bitmap &= self.sidemask
            # 枝刈り
            if row != 0:
                lim = self.size
            else:
                lim = (self.size + 1) // 2  # 割り算の結果を整数にするには //
            # 枝刈り
            for i in range(row, lim): # pylint: disable=W0612
                while bitmap:
                    bit = (-bitmap & bitmap)
                    self.aboard[row] = bit
                    bitmap ^= self.aboard[row]
                    self.backtrack2(row + 1, (left | bit) << 1, down | bit, (right | bit) >> 1)
    #
    def backtrack1(self, row, left, down, right):
        """ backtrack1() """
        bitmap = self.mask & ~(left | down | right)
        if row == self.size - 1:
            if bitmap:
                self.aboard[row] = bitmap
                self.info.setcount(1, 0, 0)
        else:
            if row < self.bound1:
                bitmap &= ~2  # bm|=2 bm^=2 (bm&=~2と同等)
            if row != 0:
                lim = self.size
            else:
                lim = (self.size + 1) // 2  # 割り算の結果を整数にするには //
            for i in range(row, lim): # pylint: disable=W0612
                while bitmap:
                    bit = (-bitmap & bitmap)
                    self.aboard[row] = bit
                    bitmap ^= self.aboard[row]
                    self.backtrack1(row + 1, (left | bit) << 1, down | bit, (right | bit) >> 1)
    #
    def rec_bound2(self, bound1, bound2):
        """ rec_bound2() """
        self.bound1 = bound1
        self.bound2 = bound2
        self.aboard[0] = bit = (1 << bound1)
        self.backtrack2(1, bit << 1, bit, bit >> 1)
        self.lastmask |= self.lastmask >> 1 | self.lastmask << 1
        self.endbit >>= 1
    #
    def rec_bound1(self, bound1):
        """ rec_bound1() """
        self.bound1 = bound1
        self.aboard[1] = bit = (1 << bound1)
        self.backtrack1(2, (2 | bit) << 1, (1 | bit), bit >> 1)
    #
    def nqueen(self):
        """ nqueen() """
        if self.child is None:
            if self.nmore > 0:
                self.topbit = 1 << (self.size - 1)
                self.aboard[0] = 1
                self.sizee = self.size - 1
                self.bound1 = 2
                while (self.bound1 > 1) and (self.bound1 < self.sizee):
                    self.rec_bound1(self.bound1)
                    self.bound1 += 1
                self.sidemask = self.lastmask = (self.topbit | 1)
                self.endbit = (self.topbit >> 1)
                self.bound2 = self.size - 2
                self.bound1 = 1
                while (self.bound1 > 0) and (self.bound2 < self.size - 1) and (self.bound1 < self.bound2): # pylint: disable=C0301
                    self.rec_bound2(self.bound1, self.bound2)
                    self.bound1 += 1
                    self.bound2 -= 1
#
def main():
    """ main() """
    nmax = 16
    nmin = 4  # Nの最小値（スタートの値）を格納
    print("シングルスレッド")
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for i in range(nmin, nmax):
        lock = threading.Lock()
        info = Board(lock)
        start_time = datetime.now()
        child = WorkingEngine(i, i, info, i - 1, 0)
        child.start()
        child.join()
        time_elapsed = datetime.now() - start_time
        _text = '{}'.format(time_elapsed)
        text = _text[:-3]
        print("%2d:%13d%13d%20s" % (i, info.gettotal(), info.getunique(), text))  # 出力
main()

