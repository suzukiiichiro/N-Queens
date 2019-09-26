#!/usr/bin/env python

# -*- coding: utf-8 -*-
""" py13_4_multiprocess_nqueen.py """

from datetime import datetime
from multiprocessing import Pool as ThreadPool
#
# /**
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#
#
#  実行
#  $ python py13_4_multiprocess_nqueen.py
#
# １３−４．マルチプロセス
#  単一のCPUを複数のプロセスで使い回すマルチスレッドと異なり、
#  複数のCPUを複数のプロセスで使い回すマルチプロセスのほうが、
#  圧倒的に高速です
#  py13_1, 2, 3, では処理中に、「全ての」CPUの負荷が１００％になることはありませんでした。
#  処理中は全てのCPUの負荷が１００％になっていることを確認して下さい。
#
#
# py12_nqueen.py
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.000
#  5:           10            2         0:00:00.000
#  6:            4            1         0:00:00.000
#  7:           40            6         0:00:00.000
#  8:           92           12         0:00:00.000
#  9:          352           46         0:00:00.002
# 10:          724           92         0:00:00.012
# 11:         2680          341         0:00:00.057
# 12:        14200         1787         0:00:00.291
# 13:        73712         9233         0:00:01.512
# 14:       365596        45752         0:00:08.967
# 15:      2279184       285053         0:00:54.645
#
#
# シングルスレッドにて実行 py13_1_singlethread_nqueen.py
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.000
#  5:           10            2         0:00:00.000
#  6:            4            1         0:00:00.000
#  7:           40            6         0:00:00.000
#  8:           92           12         0:00:00.000
#  9:          352           46         0:00:00.003
# 10:          724           92         0:00:00.013
# 11:         2680          341         0:00:00.062
# 12:        14200         1787         0:00:00.309
# 13:        73712         9233         0:00:01.672
# 14:       365596        45752         0:00:09.599
# 15:      2279184       285053         0:01:00.039
#
#
# マルチスレッドにて実行 py13_2_multithread_nqueen.py
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.000
#  5:           10            2         0:00:00.000
#  6:            4            1         0:00:00.000
#  7:           40            6         0:00:00.000
#  8:           92           12         0:00:00.001
#  9:          352           46         0:00:00.004
# 10:          724           92         0:00:00.014
# 11:         2680          341         0:00:00.065
# 12:        14200         1787         0:00:00.315
# 13:        73712         9233         0:00:01.684
# 14:       365596        45752         0:00:09.602
# 15:      2279184       285053         0:00:59.849
#
#
# マルチスレッドにて py13_3_multithread_join_nqueen.py
# コア一つを使い回しているにすぎない。遅い
# joinを処理末で実行。本来のマルチスレッド
# Nの数分スレッドが起動しそれぞれ並列処理
#
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.001
#  5:           10            2         0:00:00.000
#  6:            4            1         0:00:00.000
#  7:           40            6         0:00:00.000
#  8:           92           12         0:00:00.001
#  9:          352           46         0:00:00.004
# 10:          724           92         0:00:00.015
# 11:         2680          341         0:00:00.069
# 12:        14200         1787         0:00:00.328
# 13:        73712         9233         0:00:01.919
# 14:       365596        45752         0:00:10.975
# 15:      2279184       285053         0:01:08.997
#
#
# マルチプロセス （にてるけど違う 現在作り中）
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.123
#  5:           10            2         0:00:00.115
#  6:            4            1         0:00:00.111
#  7:           40            6         0:00:00.117
#  8:          192           24         0:00:00.119
#  9:          600           76         0:00:00.115
# 10:         1324          166         0:00:00.119
# 11:         5560          697         0:00:00.121
# 12:        29896         3739         0:00:00.223
# 13:       154312        19293         0:00:00.746
# 14:       804416       100561         0:00:04.048
# 15:      5018016       627288         0:00:29.257
#
#
#
# グローバル変数
#
class Nqueen(): # pylint: disable=R0902
    """ nqueen() """
    # MAX = 16 # N = 15
    # self.aboard = [0 for i in range(MAX)]
    # self.topbit = 0
    # self.endbit = 0
    # self.sidemask = 0
    # self.lastmask = 0
    # self.bound1 = 0
    # self.bound2 = 0
    # MASK = 0
    #
    # 初期化
    def __init__(self, size):
        """ __init__"""
        self.size = size                    # N
        self._nthreads = self.size
        self.total = 0                      # スレッド毎の合計
        self.unique = 0
        self.gttotal = [0] * self.size      #総合計
        self.gtunique = [0] * self.size     #総合計

        self.aboard = [[i for i in range(2*size-1)] for j in range(self.size)]
        self.mask = (1<<size)-1
        self.count2 = 0
        self.count4 = 0
        self.count8 = 0
        self.bound1 = 0
        self.bound2 = 0
        self.sidemask = 0
        self.lastmask = 0
        self.topbit = 0
        self.endbit = 0
    #
    # 解法
    def solve(self):
        """ solve() """
        pool = ThreadPool(self.size)
        self.gttotal = list(pool.map(self.nqueen, range(self.size)))
        total = 0
        unique = 0
        #
        # ここをみて。あともう少しなんだけどな
        #print("gttotal:%s" % self.gttotal)
        #
        for _t, _u in self.gttotal:
            total += _t
            unique += _u
        pool.close()
        pool.join()
        #
        return total, unique
        #
        # ここをみて。あともう少しなんだけどな
        #return total/self.size, unique/self.size
    #
    # ユニーク値を出力
    def getunique(self):
        """ getunique() """
        return self.count2+self.count4+self.count8
    #
    # 合計を出力
    def gettotal(self):
        """ gettotal() """
        return self.count2*2+self.count4*4+self.count8*8
    #
    # 対称解除法
    def symmetryops(self):      # pylint: disable=R0912,R0911,R0915
        """ symmetryops() """
        # global count2  # pylint: disable=W0603
        # global count4  # pylint: disable=W0603
        # global count8  # pylint: disable=W0603
        # global self.aboard  # pylint: disable=W0603
        # global self.topbit  # pylint: disable=W0603
        # global self.endbit  # pylint: disable=W0603
        own = 0
        ptn = 0
        you = 0
        bit = 0
        # 90度回転
        if self.aboard[self.bound2] == 1:
            own = 1
            ptn = 2
            while own <= self.size-1:
                bit = 1
                you = self.size-1
                while (self.aboard[you] != ptn) and (self.aboard[own] >= bit):
                    bit <<= 1
                    you -= 1
                if self.aboard[own] > bit:
                    return
                if self.aboard[own] < bit:
                    break
                own += 1
                ptn <<= 1
            #90度回転して同型なら180度/270度回転も同型である */
            if own > self.size-1:
                self.count2 += 1
                return
        #180度回転
        if self.aboard[self.size-1] == self.endbit:
            own = 1
            you = self.size-1-1
            while own <= self.size-1:
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
            if own > self.size-1:
                self.count4 += 1
                return
        #270度回転
        if self.aboard[self.bound1] == self.topbit:
            own = 1
            ptn = self.topbit>>1
            while own <= self.size-1:
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
        self.count8 += 1
    #
    # BackTrack2
    def backtrack2(self, row, left, down, right): # pylint: disable=R0913
        """ backtrack2() """
        # global self.aboard       # pylint: disable=W0603
        # global self.lastmask     # pylint: disable=W0603
        # global self.bound1       # pylint: disable=W0603
        # global self.bound2       # pylint: disable=W0603
        bit = 0
        bitmap = self.mask&~(left|down|right)
        #枝刈り
        #if row == size:
        if row == self.size-1:
            if bitmap:
                #枝刈り
                if (bitmap&self.lastmask) == 0:
                    self.aboard[row] = bitmap
                    #symmetryOps_bitmap(size)
                    self.symmetryops()
            #else:
            #  self.aboard[row] = bitmap
            #  symmetryOps_bitmap(size)
        else:
            #枝刈り 上部サイド枝刈り
            if row < self.bound1:
                bitmap &= ~self.sidemask
            #枝刈り 下部サイド枝刈り
            elif row == self.bound2:
                if down&self.sidemask == 0:
                    return
                if down&self.sidemask != self.sidemask:
                    bitmap &= self.sidemask
            # 枝刈り
            if row != 0:
                lim = self.size
            else:
                lim = (self.size+1)//2 # 割り算の結果を整数にするには //
            # 枝刈り
            for i in range(row, lim):       # pylint: disable=W0612
                while bitmap:
                    bit = (-bitmap&bitmap)
                    self.aboard[row] = bit
                    bitmap ^= self.aboard[row]
                    self.backtrack2(row+1, (left|bit)<<1, down|bit, (right|bit)>>1)
    #
    # BackTrack1
    def backtrack1(self, row, left, down, right):  # pylint: disable=R0913
        """ backtrack1() """
        # global self.aboard       # pylint: disable=W0603,W0601
        # global count8       # pylint: disable=W0603,W0601
        # global self.bound1       # pylint: disable=W0603,W0601
        bit = 0
        bitmap = self.mask&~(left|down|right)
        #枝刈り
        #if row =  = size:
        if row == self.size-1:
            if bitmap:
                self.aboard[row] = bitmap
                #枝刈りにてcount8に加算
                self.count8 += 1
            #else:
            #  self.aboard[row] = bitmap
            #  symmetryOps_bitmap(size)
        else:
            #枝刈り 鏡像についても主対角線鏡像のみを判定すればよい
            # ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
            if row < self.bound1:
                bitmap &= ~2 # bm| = 2 bm^ = 2 (bm& = ~2と同等)
            # 枝刈り
            if row != 0:
                lim = self.size
            else:
                lim = (self.size+1)//2 # 割り算の結果を整数にするには //
            # 枝刈り
            for i in range(row, lim):   # pylint: disable=W0612
                while bitmap:
                    bit = (-bitmap&bitmap)
                    self.aboard[row] = bit
                    bitmap ^= self.aboard[row]
                    self.backtrack1(row+1, (left|bit)<<1, down|bit, (right|bit)>>1)
      #
    # メインメソッド
    #def nqueen(self, thr_index, row=0, depth=0):
    def nqueen(self, thr_index):
        """ nqueen() """
        # global self.aboard         # pylint: disable=W0603,W0601
        # self.aboard = self.aboard[thr_index]
        # size = self.size
        # start = 0 if (row > 0) else int(thr_index * (size / self._nthreads))
        # end = size - 1 if ((row > 0) or (thr_index == self._nthreads - 1)) else int((thr_index + 1) * (size / self._nthreads) - 1) # pylint: disable=C0301
        # global self.topbit         # pylint: disable=W0603,W0601
        # global self.endbit         # pylint: disable=W0603,W0601
        # global self.sidemask       # pylint: disable=W0603,W0601
        # global self.lastmask       # pylint: disable=W0603,W0601
        # global self.bound1         # pylint: disable=W0603,W0601
        # global self.bound2         # pylint: disable=W0603,W0601
        bit = 0
        self.aboard[0] = 1
        self.topbit = 1<<(self.size-1)
        self.bound1 = thr_index
        #
        # 364+ をコメントアウトして 363+を活かすと数は合う
        # ロジックはあっているはず。合計はもちろん違うけど
        #
        # for の時
        # gttotal:[(92, 12), (92, 12), (92, 12), (92, 12), (92, 12), (92, 12), (92, 12), (92, 12)]
        #  8:          736           96         0:00:00.119
        # 
        # if の時
        # gttotal:[(0, 0), (56, 7), (72, 9), (32, 4), (24, 3), (8, 1), (0, 0), (0, 0)]
        #  8:          192           24         0:00:00.118
        #
        # print(thr_index) # N=8の時は 0,1,2,3,4,5,6,7
        #
        #for self.bound1 in range(2, self.size-1):
        if self.bound1 > 1 and self.bound1 < self.size - 1:
            self.aboard[1] = bit = (1<<self.bound1)
            self.backtrack1(2, (2|bit)<<1, (1|bit), (bit>>1))
        self.sidemask = self.lastmask = (self.topbit|1)
        self.endbit = (self.topbit>>1)
        self.bound2 = self.size-2
        #
        # 374+ をコメントアウトして 373+を活かすと数は合う
        # ロジックはあっているはず
        #for self.bound1 in range(1, self.bound2):
        if self.bound1 > 0 and self.bound2 < self.size - 1 and self.bound1 < self.bound2:
            self.aboard[0] = bit = (1<<self.bound1)
            self.backtrack2(1, bit<<1, bit, bit>>1)
            self.lastmask |= self.lastmask>>1|self.lastmask<<1
            self.endbit >>= 1
            self.bound2 -= 1
        return self.gettotal(), self.getunique()
#
# メインメソッド
def main():
    """ main() """
    # global count2     # pylint: disable=W0603,W0601
    # global count4     # pylint: disable=W0603,W0601
    # global count8     # pylint: disable=W0603,W0601
    # global self.aboard     # pylint: disable=W0603,W0601
    nmin = 4                          # Nの最小値（スタートの値）を格納
    nmax = 16
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for i in range(nmin, nmax):
        # count2 = count4 = count8 = 0
        # mask = (1<<i)-1
        # for j in range(i):
        #     self.aboard[j] = j              # 盤を初期化
        start_time = datetime.now()
        nqueen_obj = Nqueen(i)
        total, unique = nqueen_obj.solve()
        # nqueen(i, mask)
        time_elapsed = datetime.now()-start_time
        _text = '{}'.format(time_elapsed)
        text = _text[:-3]
        # print("%2d:%13d%13d%20s" % (i, gettotal(), getunique(), text)) # 出力
        print("%2d:%13d%13d%20s" % (i, total, unique, text)) # 出力
#
main()
#
