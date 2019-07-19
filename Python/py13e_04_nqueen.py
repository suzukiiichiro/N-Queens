#!/usr/bin/env python

# -*- coding: utf-8 -*-
""" py13e_04_nqueen.py """

from datetime import datetime
from multiprocessing import Pool as ThreadPool

# /**
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#
#
#  実行
#  $ python py13e_04_nqueen.py
#
# １３e_04．マルチプロセス 04 （バックトラック＋対称解除法）(py04相当)
# マルチプロセスによるバックトラック＋対称解除法です。
#
# 実行結果
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.123
#  5:           10            2         0:00:00.115
#  6:            4            1         0:00:00.117
#  7:           40            6         0:00:00.118
#  8:           92           12         0:00:00.114
#  9:          352           46         0:00:00.115
# 10:          724           92         0:00:00.121
# 11:         2680          341         0:00:00.227
# 12:        14200         1787         0:00:00.746
# 13:        73712         9233         0:00:03.847
# 14:       365596        45752         0:00:26.896
# 15:      2279184       285053         0:02:50.486
#

class Nqueen():
    """ nqueen() """
    # 初期化
    def __init__(self, size):
        """ __init__"""
        self.size = size                    # N
        self._nthreads = self.size
        self.total = 0                      # スレッド毎の合計
        self.unique = 0
        self.gttotal = [0] * self.size      #総合計
        self.gtunique = [0] * self.size     #総合計
        self.aboard = [[0 for i in range(self.size)] for j in range(self.size)]
        self.count = 0
        self.FA = [[0 for i in range(2*size-1)] for j in range(self.size)]
        self.FB = [[0 for i in range(2*size-1)] for j in range(self.size)]
        self.FC = [[0 for i in range(2*size-1)] for j in range(self.size)]
        self.AT = [[0 for i in range(2*size-1)] for j in range(self.size)]
        self.AS = [[0 for i in range(2*size-1)] for j in range(self.size)]
    # 
    def solve(self):
        """ solve() """
        pool = ThreadPool(self.size)
        self.gttotal = list(pool.map(self.nqueen, range(self.size)))
        to=0
        uni=0
        for t,u in self.gttotal:
            to+=t
            uni+=u
        pool.close()
        pool.join()
        return to, uni
    # 回転
    def rotate(self, chk, scr, _n, neg):
        """ rotate() """
        incr = 0
        #int k = neg ? 0 : n-1
        k = 0 if neg else _n-1
        #int incr = (neg ? +1 : -1)
        incr = 1 if neg else -1
        for i in range(_n):
            scr[i] = chk[k]
            k += incr
        k = _n-1 if neg else 0
        for i in range(_n):
            chk[scr[i]] = k
            k -= incr
    #
    # 反転
    def vmirror(self, chk, neg):
        """ vMirror() """
        for i in range(neg):
            chk[i] = (neg-1)-chk[i]
    #
    #
    def intncmp(self, _lt, _rt, neg):
        """ intncmp() """
        rtn = 0
        for i in range(neg):
            rtn = _lt[i]-_rt[i]
            if rtn != 0:
                break
        return rtn
    #
    # 対称解除法
    def symmetryops(self, thr_index): # pylint: disable=R0911,R0912
        """ symmetryOps() """
        AT = self.AT[thr_index]
        AS = self.AS[thr_index]
        ABOARD = self.aboard[thr_index]
        size = self.size
        nquiev = 0
        # 回転・反転・対称チェックのためにboard配列をコピー
        for i in range(size):
            AT[i] = ABOARD[i]
        # 時計回りに90度回転
        self.rotate(AT, AS, size, 0)
        k = self.intncmp(ABOARD, AT, size)
        if k > 0:
            return 0            # pylint: disable=R0915
        if k == 0:
            nquiev = 1
        else:
            # 時計回りに180度回転
            self.rotate(AT, AS, size, 0)
            k = self.intncmp(ABOARD, AT, size)
            if k > 0:
                return 0        # pylint: disable=R0915
            if k == 0:
                nquiev = 2
            else:
                # 時計回りに270度回転
                self.rotate(AT, AS, size, 0)
                k = self.intncmp(ABOARD, AT, size)
                if k > 0:
                    return 0
                nquiev = 4
        # 回転・反転・対称チェックのためにboard配列をコピー
        for i in range(size):
            AT[i] = ABOARD[i]
        # 垂直反転
        self.vmirror(AT, size)
        k = self.intncmp(ABOARD, AT, size)
        if k > 0:
            return 0
        # -90度回転 対角鏡と同等
        if nquiev > 1:
            self.rotate(AT, AS, size, 1)
            k = self.intncmp(ABOARD, AT, size)
            if k > 0:
                return 0
            # -180度回転 水平鏡像と同等
            if nquiev > 2:
                self.rotate(AT, AS, size, 1)
                k = self.intncmp(ABOARD, AT, size)
                if k > 0:
                    return 0
                # -270度回転 反対角鏡と同等
                self.rotate(AT, AS, size, 1)
                k = self.intncmp(ABOARD, AT, size)
                if k > 0:
                    return 0
        return nquiev*2
#
    def nqueen(self, thr_index, row=0, depth=0):     # rowは横(行) colは縦(列)
        """nqueen()"""
        # self.count += 1
        # print(self.count)
        # print(thr_index)
        ABOARD = self.aboard[thr_index]
        FA = self.FA[thr_index]
        FB = self.FB[thr_index]
        FC = self.FC[thr_index]
        size = self.size
        start = 0 if (row > 0) else int(thr_index * (size / self._nthreads))
        end = size - 1 if ((row > 0) or (thr_index == self._nthreads - 1)) else int((thr_index + 1) * (size / self._nthreads) - 1)
        if row == size:
            stotal = self.symmetryops(thr_index)	      # 対称解除法の導入
            if stotal != 0:
                self.unique += 1
                self.total += stotal
        for i in range(start, end + 1):
            if FA[i] == 0 and FB[row-i+(size-1)] == 0 and FC[row+i] == 0:
                FA[i] = FB[row-i+(size-1)] = FC[row+i] = 1
                ABOARD[row] = i
                self.nqueen(thr_index, row + 1, depth + 1)
                FA[i] = FB[row-i+(size-1)] = FC[row+i] = 0
        if depth == 0:
            return self.total,self.unique
        return self.total, self.unique
def main():
    """main()"""
    print(" N:        Total       Unique        hh:mm:ss.ms")
    nmin = 4
    nmax = 16
    # nmin = 8
    # nmax = 9
    for size in range(nmin, nmax):
        start_time = datetime.now()
        nqueen_obj = Nqueen(size)
        total,unique = nqueen_obj.solve()
        # total = nqueen_obj.solve()
        # unique = 0
        time_elapsed = datetime.now()-start_time
        _text = '{}'.format(time_elapsed)
        text = _text[:-3]
        print("%2d:%13d%13d%20s" % (size, total, unique, text)) # 出力
if __name__ == "__main__":
    main()
