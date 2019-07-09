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
#
#   実行結果
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            0         0:00:00.127
#  5:           10            0         0:00:00.115
#  6:            4            0         0:00:00.117
#  7:           40            0         0:00:00.117
#  8:           92            0         0:00:00.113
#  9:          352            0         0:00:00.119
# 10:          724            0         0:00:00.227
# 11:         2680            0         0:00:01.152
# 12:        14200            0         0:00:07.610
# 13:        73712            0         0:00:50.093
#
#
class Nqueen():
    """ nqueen() """
    def __init__(self, size):
        """ __init__"""
        self.size = size                    # N
        self.total = 0                      # スレッド毎の合計
        self.unique = 0
        self.gttotal = [0] * self.size      #総合計
        self.aboard = [[0 for i in range(self.size)] for j in range(self.size)]
    def solve(self):
        """ solve() """
        pool = ThreadPool(self.size)
        self.gttotal = pool.map(self.nqueen, range(self.size))
        pool.close()
        pool.join()
        return sum(self.gttotal)
    def nqueen(self, thr_index, row=0):     # rowは横(行) colは縦(列)
        """nqueen()"""
        if row > 0:
            start = 0
            end = self.size -1
        else:
            start = thr_index
            end = thr_index
        if row == self.size:
            self.total += 1
        else:
            for i in range(start, end + 1):
                _v = 0
                while(_v < row and self.is_safe(i, _v, row, thr_index)):
                    _v += 1
                if _v < row:
                    continue
                self.aboard[thr_index][row] = i
                self.nqueen(thr_index, row + 1)
            if row == 0:
                return self.total
        return self.total
    def is_safe(self, i, _v, row, thr_index):
        """is_safe() """
        if self.aboard[thr_index][_v] == i:      # 縦位置の検査
            return 0
        if abs(self.aboard[thr_index][_v] - i) == row - _v: # 斜めの検査 3x5 == 5x3
            return 0
        return 1
def main():
    """main()"""
    print(" N:        Total       Unique        hh:mm:ss.ms")
    nmin = 4
    nmax = 16
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

