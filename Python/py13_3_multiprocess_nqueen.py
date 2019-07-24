#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" py13d_nqueen.py """
import logging
import threading
from threading import Thread
from multiprocessing import Process
from datetime import datetime
#
#
#
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#
# 【Macでの注意】
# ローカル環境での実行時、OSの環境が『Mac OS High Sierra』以上の場合は以下のようなエラーが出ます。
# [__NSPlaceholderDate initialize] may have been in progress in another thread when fork() was called.
#
# これは『Mac OS High Sierra』以降、セキュリティ強化のためのマルチスレッド制限
# 用にfork()の振る舞いが変更されたことが原因です。
#
# そのため『Mac OS High Sierra』以降でPythonの並列化処理を利用するには、環境変
# 数『.bash_profile』に新しいセキュリティ規則の下でマルチスレッドアプリケーショ
# ンを許可する設定が必要になります。
# 
#
#  ~/.bash_profileへ以下を追加
#  export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
# 
#
#  実行
#  $ python py13d_nqueen.py
#
#
# １３d．マルチスレッドとマルチプロセス
#
#         マルチスレッドは、一つのコアの中に複数のスレッド起動して実行します。
#         ですので、Nが大きくなると処理速度は低下します。
#         この節では、より高速に実行できるマルチプロセスを構築します。
#         とはいえ、ものすごく単純なマルチプロセス対応となります。
#         
# １３a
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
# １３b
# マルチスレッドにて
# start/joinを連ねて遅くとも間違いなく実行
# やっていることはシングルスレッドと同等。
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.003
#  5:           10            2         0:00:00.002
#  6:            4            1         0:00:00.003
#  7:           40            6         0:00:00.004
#  8:           92           12         0:00:00.006
#  9:          352           46         0:00:00.016
# 10:          724           92         0:00:00.064
# 11:         2680          341         0:00:00.219
# 12:        14200         1787         0:00:00.977
# 13:        73712         9233         0:00:05.262
# 14:       365596        45752         0:00:29.435
# 15:      2279184       285053         0:03:05.539
#
# １３c
# マルチスレッドにて(コア一つを使い回しているので遅い）
# joinを処理末で実行。本来のマルチスレッド
# Nの数分スレッドが起動しそれぞれ並列処理
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.005
#  5:           10            2         0:00:00.002
#  6:            4            1         0:00:00.002
#  7:           40            6         0:00:00.003
#  8:           92           12         0:00:00.006
#  9:          352           46         0:00:00.015
# 10:          724           92         0:00:00.047
# 11:         2680          341         0:00:00.204
# 12:        14200         1787         0:00:01.022
# 13:        73712         9233         0:00:05.317
# 14:       365596        45752         0:00:31.083
# 15:      2279184       285053         0:03:13.819
#
# １３d 
# マルチプロセス
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.000
#  5:           10            2         0:00:00.000
#  6:            4            1         0:00:00.000
#  7:           40            6         0:00:00.000
#  8:           92           12         0:00:00.001
#  9:          352           46         0:00:00.004
# 10:          724           92         0:00:00.014
# 11:         2680          341         0:00:00.064
# 12:        14200         1787         0:00:00.316
# 13:        73712         9233         0:00:01.721
# 14:       365596        45752         0:00:10.285
# 15:      2279184       285053         0:01:05.723
#
#
#
# マルチプロセス      マルチスレッド
#    BPROCESS           BTHREAD         joinの抑制
#      True             True----------  ENABLEJSON
#     False             False        |--- True
#                                    |--- False
#
# マルチプロセスの実行
BPROCESS = True
#BPROCESS = False
#
# マルチスレッド・シングルスレッドの切り換えフラグ
BTHREAD = True          # マルチスレッド
#BTHREAD = False        # シングルスレッド

#ENABLEJOIN = True      # コンストラクタでjoinする 遅い
ENABLEJOIN = False      # run()の処理末尾でjoin() マルチスレッド完成型
#
                        #
#
#
# ボードクラス
class Board:
    """ Board """
    #
    def __init__(self, lock):
        """ __init__ """
        # print("Board:init")
        self.count2 = 0
        self.count4 = 0
        self.count8 = 0
        self.lock = lock
    #
    def setcount(self, count8, count4, count2):
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
# マルチプロセス
class mpWorkingEngine(Process): # pylint: disable=R0902
    """ WorkingEngine """
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(levelname)s] (%(threadName)-10s) %(message)s', )
    #
    def __init__(self, size, nmore, info, B1, B2, bthread): # pylint: disable=R0913
        """ ___init___ """
        super(mpWorkingEngine, self).__init__()
        global BTHREAD    # pylint: disable=W0603
        BTHREAD = bthread
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
        if nmore > 0:
            # マルチスレッド
            if bthread:
                self.child = mpWorkingEngine(size, nmore - 1, info, B1 - 1, B2 + 1, bthread)       # pylint: disable=C0301
                self.bound1 = B1
                self.bound2 = B2
                self.child.start()
#               # マルチプロセスなので以下の条件分岐が不要となる
#               # コンストラクタでjoin()する
#                if ENABLEJOIN:
#                    self.child.join()   # joinする
#            # シングルスレッド
#            else:
#                self.child = None
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
        # マルチスレッド
        else:
            self.aboard[0] = 1
            self.sizee = self.size - 1
            self.mask = (1 << self.size) - 1
            self.topbit = (1 << self.sizee)
            if (self.bound1 > 1) and (self.bound1 < self.sizee):
                self.rec_bound1(self.bound1)
            self.endbit = (self.topbit >> self.bound1)
            self.sidemask = self.lastmask = (self.topbit | 1)
            if (self.bound1 > 0) and (self.bound2 < self.size - 1) and (self.bound1 < self.bound2):
                for i in range(1, self.bound1): # pylint: disable=W0612
                    self.lastmask = self.lastmask | self.lastmask >> 1 | self.lastmask << 1
                self.rec_bound2(self.bound1, self.bound2)
                self.endbit >>= self.nmore
            # ここで処理を止めたい
            #print("finished")
#            # マルチプロセスなので以下の条件分岐が不要となる
#            # コンストラクタでjoin()する
#            if ENABLEJOIN:
#                pass
#            # run()の処理末尾でjoin()する
#            else:
#                self.child.join()
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
# マルチスレッド
class WorkingEngine(Thread): # pylint: disable=R0902
    """ WorkingEngine """
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(levelname)s] (%(threadName)-10s) %(message)s', )
    #
    def __init__(self, size, nmore, info, B1, B2, bthread): # pylint: disable=R0913
        """ ___init___ """
        super(WorkingEngine, self).__init__()
        global BTHREAD    # pylint: disable=W0603
        BTHREAD = bthread
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
        if nmore > 0:
            # マルチスレッド
            if bthread:
                self.child = WorkingEngine(size, nmore - 1, info, B1 - 1, B2 + 1, bthread)       # pylint: disable=C0301
                self.bound1 = B1
                self.bound2 = B2
                self.child.start()
                # コンストラクタでjoin()する
                if ENABLEJOIN:
                    self.child.join()   # joinする
            # シングルスレッド
            else:
                self.child = None
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
        # マルチスレッド
        else:
            self.aboard[0] = 1
            self.sizee = self.size - 1
            self.mask = (1 << self.size) - 1
            self.topbit = (1 << self.sizee)
            if (self.bound1 > 1) and (self.bound1 < self.sizee):
                self.rec_bound1(self.bound1)
            self.endbit = (self.topbit >> self.bound1)
            self.sidemask = self.lastmask = (self.topbit | 1)
            if (self.bound1 > 0) and (self.bound2 < self.size - 1) and (self.bound1 < self.bound2):
                for i in range(1, self.bound1): # pylint: disable=W0612
                    self.lastmask = self.lastmask | self.lastmask >> 1 | self.lastmask << 1
                self.rec_bound2(self.bound1, self.bound2)
                self.endbit >>= self.nmore
            # コンストラクタでjoin()する
            if ENABLEJOIN:
                pass
            # run()の処理末尾でjoin()する
            else:
                self.child.join()
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
# メイン
def main():
    """ main() """
    nmax = 16
    nmin = 4  # Nの最小値（スタートの値）を格納
    if BPROCESS:
        print("マルチプロセス")
    elif BTHREAD:
        if ENABLEJOIN:
            print("マルチスレッドにて")
            print("start/joinを連ねて遅くとも間違いなく実行")
            print("やっていることはシングルスレッドと同等。")
        else:
            print("マルチスレッドにて")
            print("コア一つを使い回しているにすぎない。遅い")
            print("joinを処理末で実行。本来のマルチスレッド")
            print("Nの数分スレッドが起動しそれぞれ並列処理")
    else:
        print("シングルスレッドにて実行")
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for i in range(nmin, nmax):
        lock = threading.Lock()
        info = Board(lock)
        start_time = datetime.now()
        if BPROCESS:        # マルチプロセス
            child = mpWorkingEngine(i, i, info, i - 1, 0, BTHREAD)
            child.start()
            child.join()
        else:               # スレッド or マルチスレッド
            child = WorkingEngine(i, i, info, i - 1, 0, BTHREAD)
            child.start()
            child.join()
        time_elapsed = datetime.now() - start_time
        _text = '{}'.format(time_elapsed)
        text = _text[:-3]
        print("%2d:%13d%13d%20s" % (i, info.gettotal(), info.getunique(), text))  # 出力

if __name__ == '__main__':
    main()

