#!/usr/bin/env python

# -*- coding: utf-8 -*-
""" py05_nqueen.py """
from datetime import datetime

#  /**
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#
#   実行
#   $ python py05_nqueen.py
#
#   * ５．バックトラック＋対称解除法＋枝刈りと最適化
#   *
#   * 　単純ですのでソースのコメントを見比べて下さい。
#   *   単純ではありますが、枝刈りの効果は絶大です。
#
#   実行結果
# N:        Total       Unique        hh:mm:ss.ms
# 4:            2            1         0:00:00.000
# 5:           10            2         0:00:00.000
# 6:            4            1         0:00:00.000
# 7:           40            6         0:00:00.000
# 8:           92           12         0:00:00.002
# 9:          352           46         0:00:00.013
#10:          724           92         0:00:00.052
#11:         2680          341         0:00:00.269
#12:        14200         1787         0:00:01.283
#13:        73712         9233         0:00:07.756
#14:       365596        45752         0:00:42.300
#15:      2279184       285053         0:04:48.555
#
#
# グローバル変数
MAX = 16 # N = 15
TOTAL = 0
UNIQUE = 0
ABOARD = [0 for i in range(MAX)]
#FA = [0 for i in range(2*MAX-1)]	#縦列にクイーンを一つだけ配置
FB = [0 for i in range(2*MAX-1)]	#斜め列にクイーンを一つだけ配置
FC = [0 for i in range(2*MAX-1)]	#斜め列にクイーンを一つだけ配置
AT = [0 for i in range(MAX)]     #AT:ATrial[]
AS = [0 for i in range(MAX)]     #AS:AScrath[]
#
# 回転
def rotate(chk, scr, _n, neg):
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
def vmirror(chk, neg):
    """ vMirror() """
    for i in range(neg):
        chk[i] = (neg-1)-chk[i]
#
def intncmp(_lt, _rt, neg):
    """ intncmp() """
    rtn = 0
    for i in range(neg):
        rtn = _lt[i]-_rt[i]
        if rtn != 0:
            break
    return rtn
#
# 対称解除法
def symmetryops(size): # pylint: disable=R0911,R0912
    """ symmetryOps() """
    global AT               # pylint: disable=W0603
    global AS               # pylint: disable=W0603
    nquiev = 0
    # 回転・反転・対称チェックのためにboard配列をコピー
    for i in range(size):
        AT[i] = ABOARD[i]
    # 時計回りに90度回転
    rotate(AT, AS, size, 0)
    k = intncmp(ABOARD, AT, size)
    if k > 0:
        return 0            # pylint: disable=R0915
    if k == 0:
        nquiev = 1
    else:
        # 時計回りに180度回転
        rotate(AT, AS, size, 0)
        k = intncmp(ABOARD, AT, size)
        if k > 0:
            return 0        # pylint: disable=R0915
        if k == 0:
            nquiev = 2
        else:
            # 時計回りに270度回転
            rotate(AT, AS, size, 0)
            k = intncmp(ABOARD, AT, size)
            if k > 0:
                return 0
            nquiev = 4
    # 回転・反転・対称チェックのためにboard配列をコピー
    for i in range(size):
        AT[i] = ABOARD[i]
    # 垂直反転
    vmirror(AT, size)
    k = intncmp(ABOARD, AT, size)
    if k > 0:
        return 0
    # -90度回転 対角鏡と同等
    if nquiev > 1:
        rotate(AT, AS, size, 1)
        k = intncmp(ABOARD, AT, size)
        if k > 0:
            return 0
        # -180度回転 水平鏡像と同等
        if nquiev > 2:
            rotate(AT, AS, size, 1)
            k = intncmp(ABOARD, AT, size)
            if k > 0:
                return 0
            # -270度回転 反対角鏡と同等
            rotate(AT, AS, size, 1)
            k = intncmp(ABOARD, AT, size)
            if k > 0:
                return 0
    return nquiev*2
#
# ロジックメソッド
def nqueen(row, size):
    """ nqueen() """
    global ABOARD   # pylint: disable=W0603
    global TOTAL    # pylint: disable=W0603
    global UNIQUE   # pylint: disable=W0603
    #global FA      # pylint: disable=W0603
    global FB       # pylint: disable=W0603
    global FC       # pylint: disable=W0603
    # if row==size:
    if row == size-1: # 枝刈り
        # 追加
        if FB[row-ABOARD[row]+size-1] or FC[row+ABOARD[row]]:
            return
        #
        stotal = symmetryops(size)	  # 対称解除法の導入
        if stotal != 0:
            UNIQUE += 1            # ユニーク解を加算
            TOTAL += stotal             # 対称解除で得られた解数を加算
    else:
        # 枝刈り
        # pythonでは割り算の結果を整数で返すときは // にする
        #int lim=(row!=0) ? size : (size+1)/2
        lim = size if row != 0 else (size + 1) // 2
        #for i in range(size):  # 枝刈り
        for i in range(row, lim):
            # ABOARD[row]=i      # 不要
            # 交換
            tmp = ABOARD[i]
            ABOARD[i] = ABOARD[row]
            ABOARD[row] = tmp
            #
            # 枝刈り
            if FB[row-ABOARD[row]+size-1] or FC[row+ABOARD[row]]:
                pass
            else:
                FB[row-ABOARD[row]+size-1] = FC[row+ABOARD[row]] = 1
                nqueen(row+1, size) # 再帰
                FB[row-ABOARD[row]+size-1] = FC[row+ABOARD[row]] = 0
        # 交換
        tmp = ABOARD[row]
        for i in range(row+1, size):
            ABOARD[i-1] = ABOARD[i]
        ABOARD[size-1] = tmp
#
# メインメソッド
def main():
    """ main() """
    global TOTAL    # pylint: disable=W0603
    global UNIQUE   # pylint: disable=W0603
    global ABOARD   # pylint: disable=W0603
    global MAX      # pylint: disable=W0603
    nmin = 4                          # Nの最小値（スタートの値）を格納
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for i in range(nmin, MAX):
        TOTAL = 0
        UNIQUE = 0                   # 初期化
        for j in range(i):
            ABOARD[j] = j              # 盤を初期化
        start_time = datetime.now()
        nqueen(0, i)
        time_elapsed = datetime.now() - start_time
        _text = '{}'.format(time_elapsed)
        text = _text[:-3]
        print("%2d:%13d%13d%20s" % (i, TOTAL, UNIQUE, text)) # 出力
#
main()
#
