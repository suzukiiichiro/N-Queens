#!/usr/bin/env python

# -*- coding: utf-8 -*-
""" py13e_03_nqueen.py """

from datetime import datetime
from multiprocessing import Pool as ThreadPool

# /**
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#
#
#  実行
#  $ python py13e_03_nqueen.py
#
# １３e_03．マルチプロセス 03  (py03相当)
# マルチプロセスによる単純に縦と斜めの効きをチェックするバックトラックです
# Nの数だけプロセスを起動して、最後にそれぞれの集計値を合算して合計解を求めます
# 02 で実行したソースよりも多少高速な、py03_nqueen.pyのマルチプロセス版です。
#
#   実行結果
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            0         0:00:00.125
#  5:           10            0         0:00:00.114
#  6:            4            0         0:00:00.113
#  7:           40            0         0:00:00.118
#  8:           92            0         0:00:00.118
#  9:          352            0         0:00:00.117
# 10:          724            0         0:00:00.122
# 11:         2680            0         0:00:00.225
# 12:        14200            0         0:00:00.946
# 13:        73712            0         0:00:05.152
# 14:       365596            0         0:00:35.737
# 15:      2279184            0         0:04:05.152
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
				# py03で使用します
        self.FA = [[0 for i in range(2*size-1)] for j in range(self.size)]
        self.FB = [[0 for i in range(2*size-1)] for j in range(self.size)]
        self.FC = [[0 for i in range(2*size-1)] for j in range(self.size)]
    def solve(self):
        """ solve() """
        pool = ThreadPool(self.size)
        self.gttotal = pool.map(self.nqueen, range(self.size))
        pool.close()
        pool.join()
        return sum(self.gttotal)
    def nqueen(self, thr_index, row=0):     # rowは横(行) colは縦(列)
        """nqueen()"""
        #print(thr_index)
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
                # Py03相当
                if self.FA[thr_index][i] == 0 and self.FB[thr_index][row-i+(self.size-1)] == 0 and self.FC[thr_index][row+i] == 0:
                    self.FA[thr_index][i] = self.FB[thr_index][row-i+(self.size-1)] = self.FC[thr_index][row+i] = 1
                    self.aboard[thr_index][row] = i
                    self.nqueen(thr_index, row + 1)
                    #nqueen(row+1, self.size)   #再帰
                    self.FA[thr_index][i] = self.FB[thr_index][row-i+(self.size-1)] = self.FC[thr_index][row+i] = 0
                
                # 元のソース
                # _v = 0
                # while(_v < row and self.is_safe(i, _v, row, thr_index)):
                #     _v += 1
                # if _v < row:
                #     continue
                # self.aboard[thr_index][row] = i
                # self.nqueen(thr_index, row + 1)

            if row == 0:
                return self.total
        return self.total
    # def is_safe(self, i, _v, row, thr_index):
    #     """is_safe() """
    #     if self.aboard[thr_index][_v] == i:      # 縦位置の検査
    #         return 0
    #     if abs(self.aboard[thr_index][_v] - i) == row - _v: # 斜めの検査 3x5 == 5x3
    #         return 0
    #     return 1
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
