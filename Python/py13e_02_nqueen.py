#!/usr/bin/env python

# -*- coding: utf-8 -*-
""" py13e_02_nqueen.py """

from datetime import datetime
from multiprocessing import Pool as ThreadPool

# /**
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#
#
#  実行
#  $ python py13e_02_nqueen.py
#
# １３e_02．マルチプロセス 02(一般的なNクイーンのマルチプロセスの記述から） 
# マルチプロセスによる単純に縦と斜めの効きをチェックするバックトラックです
# Nの数だけプロセスを起動して、最後にそれぞれの集計値を合算して合計解を求めます
# まずは１３dまでの処理を考えずにマルチプロセスのロジックをpy03にたちもどって
# 一からから再構築してみます。
# 実行後、複数のCPU全てが100%使用されていることを確認して下さい。
#
#   実行結果
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            0         0:00:00.124
#  5:           10            0         0:00:00.115
#  6:            4            0         0:00:00.116
#  7:           40            0         0:00:00.117
#  8:           92            0         0:00:00.118
#  9:          352            0         0:00:00.117
# 10:          724            0         0:00:00.220
# 11:         2680            0         0:00:01.049
# 12:        14200            0         0:00:06.808
# 13:        73712            0         0:00:49.041
#
#
class Nqueen():
    """ nqueen() """
    def __init__(self, size):
        """ __init__"""
        self.size = size                    # N
        self._nthreads = self.size
        self.total = 0                      # スレッド毎の合計
        self.unique = 0
        self.gttotal = [0] * self.size      #総合計
        self.aboard = [[0 for i in range(self.size)] for j in range(self.size)]
        self.count=0
    #
    def solve(self):
        """ solve() """
        pool = ThreadPool(self.size)
        self.gttotal = pool.map(self.nqueen, range(self.size))
        pool.close()
        pool.join()
        return sum(self.gttotal)
    #
    def nqueen(self, thr_index, row=0, depth=0):     # rowは横(行) colは縦(列)
        """nqueen()"""
        # self.count += 1
        # print(self.count)
        # print(thr_index)
        size = self.size
        start = 0 if (row > 0) else int(thr_index * (size / self._nthreads))
        end = size - 1 if ((row > 0) or (thr_index == self._nthreads - 1)) else int((thr_index + 1) * (size / self._nthreads) - 1)
        if row == size:
            self.total += 1
        for i in range(start, end + 1):
            j = 0
            while(j < row and self.is_safe(i, j, row, thr_index)):
                j += 1
            if j < row:
                continue
            self.aboard[thr_index][row] = i
            self.nqueen(thr_index, row + 1, depth + 1)
        if depth == 0:
            return self.total
        return self.total
		#
    def is_safe(self, i, j, row, thr_index):
        """is_safe() """
        if self.aboard[thr_index][j] == i:      # 縦位置の検査
            return 0
        if abs(self.aboard[thr_index][j] - i) == row - j: # 斜めの検査 3x5 == 5x3
            return 0
        return 1
#
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
        total = nqueen_obj.solve()
        unique = 0
        time_elapsed = datetime.now()-start_time
        _text = '{}'.format(time_elapsed)
        text = _text[:-3]
        print("%2d:%13d%13d%20s" % (size, total, unique, text)) # 出力
if __name__ == "__main__":
    main()

