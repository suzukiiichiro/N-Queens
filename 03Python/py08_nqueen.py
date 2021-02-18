#!/usr/bin/env python

# -*- coding: utf-8 -*-
""" py08_nqueen.py """

from datetime import datetime
#
# /**
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#
#  実行
#  $ python py08_nqueen.py
#
#
# ８．ビットマップ＋枝刈り
# コメント部分の枝刈りを参照
#
#   実行結果
# N:        Total       Unique        hh:mm:ss.ms
# 4:            2            1         0:00:00.000
# 5:           10            2         0:00:00.000
# 6:            4            1         0:00:00.000
# 7:           40            6         0:00:00.002
# 8:           92           12         0:00:00.006
# 9:          352           46         0:00:00.030
#10:          724           92         0:00:00.092
#11:         2680          341         0:00:00.413
#12:        14200         1787         0:00:02.349
#13:        73712         9233         0:00:13.641
#14:       365596        45752         0:01:18.859
#15:      2279184       285053         0:08:49.312
#
#
# グローバル変数
MAX = 16 # N = 15
ABOARD = [0 for i in range(MAX)]
AT = [0 for i in range(MAX)]
AS = [0 for i in range(MAX)]
COUNT2 = 0
COUNT4 = 0
COUNT8 = 0
#
def rha(_ah, size):
    """ rha() """
    tmp = 0
    for i in range(size+1):
        if _ah&(1<<i):
            tmp |= (1<<size-i)
    return tmp
#
def vmirror_bitmap(_bf, _af, size):
    """ vmirrot_bitmap() """
    score = 0
    for i in range(size):
        score = _bf[i]
        _af[i] = rha(score, size-1)
#
def rotate_bitmap(_bf, _af, size):
    """ rotate_bitmap() """
    for i in range(size):
        tmp = 0
        for j in range(size):
            tmp |= ((_bf[j]>>i)&1)<<(size-j-1)  # x[j] の i ビット目を
        _af[i] = tmp                          # y[i] の j ビット目にする
#
def intncmp(_lt, _rt, neg):
    """ intncmp """
    rtn = 0
    for i in range(neg):
        rtn = _lt[i] - _rt[i]
        if rtn != 0:
            break
    return rtn
#
# ユニーク値を出力
def getunique():
    """ getunique() """
    return COUNT2+COUNT4+COUNT8
#
# 合計を出力
def gettotal():
    """ gettotal() """
    return COUNT2*2+COUNT4*4+COUNT8*8
#
# 対称解除法
def symmetryops_bitmap(size):      # pylint: disable=R0912,R0911
    """ symmetryops_bitmap() """
    nequiv = 0
    global COUNT2 # pylint: disable=W0603
    global COUNT4 # pylint: disable=W0603
    global COUNT8 # pylint: disable=W0603
    global AT     # pylint: disable=W0603
    global AS     # pylint: disable=W0603
    # 回転・反転・対称チェックのためにboard配列をコピー
    for i in range(size):
        AT[i] = ABOARD[i]
    rotate_bitmap(AT, AS, size)    #時計回りに90度回転
    k = intncmp(ABOARD, AS, size)
    if k > 0:
        return
    if k == 0:
        nequiv = 2
    else:
        rotate_bitmap(AS, AT, size)  #時計回りに180度回転
        k = intncmp(ABOARD, AT, size)
        if k > 0:
            return
        if k == 0:
            nequiv = 4
        else:
            rotate_bitmap(AT, AS, size)#時計回りに270度回転
            k = intncmp(ABOARD, AS, size)
            if k > 0:
                return
            nequiv = 8
    # 回転・反転・対称チェックのためにboard配列をコピー
    for i in range(size):
        AS[i] = ABOARD[i]
    vmirror_bitmap(AS, AT, size)   # 垂直反転
    k = intncmp(ABOARD, AT, size)
    if k > 0:
        return
    if nequiv > 2:                #-90度回転 対角鏡と同等
        rotate_bitmap(AT, AS, size)
        k = intncmp(ABOARD, AS, size)
        if k > 0:
            return
        if nequiv > 4:              #-180度回転 水平鏡像と同等
            rotate_bitmap(AS, AT, size)
            k = intncmp(ABOARD, AT, size)
            if k > 0:
                return        #-270度回転 反対角鏡と同等
            rotate_bitmap(AT, AS, size)
            k = intncmp(ABOARD, AS, size)
            if k > 0:
                return
    if nequiv == 2:
        COUNT2 += 1
    if nequiv == 4:
        COUNT4 += 1
    if nequiv == 8:
        COUNT8 += 1
#
# ロジックメソッド
def nqueen(size, mask, row, left, down, right): # pylint: disable=R0913
    """ nqueen() """
    global ABOARD     # pylint: disable=W0603
    bitmap = mask&~(left|down|right)
    if row == size:
        if bitmap:
            pass
        else:
            ABOARD[row] = bitmap
            symmetryops_bitmap(size)
    else:
      # 枝刈り
        if row != 0:
            lim = size
        else:
            lim = (size+1)//2 # 割り算の結果を整数にするには //
        # 枝刈り
        for i in range(row, lim): # pylint: disable=W0612
            while bitmap:
                bit = (-bitmap&bitmap)
                ABOARD[row] = bit
                bitmap ^= ABOARD[row]
                nqueen(size, mask, row+1, (left|bit)<<1, down|bit, (right|bit)>>1)
#
# メインメソッド
def main():
    """ main() """
    global COUNT2     # pylint: disable=W0603
    global COUNT4     # pylint: disable=W0603
    global COUNT8     # pylint: disable=W0603
    global ABOARD     # pylint: disable=W0603
    global MAX        # pylint: disable=W0603
    nmin = 4                          # Nの最小値（スタートの値）を格納
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for i in range(nmin, MAX):
        COUNT2 = COUNT4 = COUNT8 = 0
        for j in range(i):
            ABOARD[j] = j              # 盤を初期化
        mask = (1<<i)-1
        start_time = datetime.now()
        nqueen(i, mask, 0, 0, 0, 0)
        time_elapsed = datetime.now()-start_time
        _text = '{}'.format(time_elapsed)
        text = _text[:-3]
        print("%2d:%13d%13d%20s" % (i, gettotal(), getunique(), text)) # 出力
#
main()
#
