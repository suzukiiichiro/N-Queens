#!/usr/bin/env python

# -*- coding: utf-8 -*-
""" py10_nqueen.py """

from datetime import datetime
#
# /**
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#
#  実行
#  $ python py10_nqueen.py
#
#
# １０．クイーンの位置による分岐BOUND1,2
#   ここでは前章のBOUND1に引き続きBOUND2を追加します。
#   BOUND1とBOUND2の内容は同等です。
#
#   前章のコードは全ての解を求めた後に、ユニーク解以外の対称解を除去していた
#   ある意味、「生成検査法（generate ＆ test）」と同じである
#   問題の性質を分析し、バックトラッキング/前方検査法と同じように、無駄な探索を省略することを考える
#   ユニーク解に対する左右対称解を予め削除するには、1行目のループのところで、
#   右半分だけにクイーンを配置するようにすればよい
#   Nが奇数の場合、クイーンを1行目中央に配置する解は無い。
#   他の3辺のクィーンが中央に無い場合、その辺が上辺に来るよう回転し、場合により左右反転することで、
#   最小値解とすることが可能だから、中央に配置したものしかユニーク解には成り得ない
#   しかし、上辺とその他の辺の中央にクィーンは互いの効きになるので、配置することが出来ない
#
#
#   1. １行目角にクイーンがある場合、とそうでない場合で処理を分ける
#     １行目かどうかの条件判断はループ外に出してもよい
#     処理時間的に有意な差はないので、分かりやすいコードを示した
#   2.１行目角にクイーンがある場合、回転対称形チェックを省略することが出来る
#     １行目角にクイーンがある場合、他の角にクイーンを配置することは不可
#     鏡像についても、主対角線鏡像のみを判定すればよい
#     ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
#
#   １行目角にクイーンが無い場合、クイーン位置より右位置の８対称位置にクイーンを置くことはできない
#   置いた場合、回転・鏡像変換により得られる状態のユニーク判定値が明らかに大きくなる
#     ☓☓・・・Ｑ☓☓
#     ☓・・・／｜＼☓
#     ｃ・・／・｜・rt
#     ・・／・・｜・・
#     ・／・・・｜・・
#     lt・・・・｜・ａ
#     ☓・・・・｜・☓
#     ☓☓ｂ・・dn☓☓
#
#   １行目位置が確定した時点で、配置可能位置を計算しておく（☓の位置）
#   lt, dn, lt 位置は効きチェックで配置不可能となる
#   回転対称チェックが必要となるのは、クイーンがａ, ｂ, ｃにある場合だけなので、
#   90度、180度、270度回転した状態のユニーク判定値との比較を行うだけで済む
#
#
#   実行結果
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.000
#  5:           10            2         0:00:00.000
#  6:            4            1         0:00:00.000
#  7:           40            6         0:00:00.001
#  8:           92           12         0:00:00.005
#  9:          352           46         0:00:00.025
# 10:          724           92         0:00:00.080
# 11:         2680          341         0:00:00.368
# 12:        14200         1787         0:00:02.132
# 13:        73712         9233         0:00:12.465
# 14:       365596        45752         0:01:12.723
# 15:      2279184       285053         0:08:10.190
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
TOPBIT = 0
ENDBIT = 0
SIDEMASK = 0
LASTMASK = 0
BOUND1 = 0
BOUND2 = 0
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
    _af[i] = tmp                            # y[i] の j ビット目にする
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
  # 回転・反転・対称チェックのためにboard配列をコピー
  for i in range(size):
    AT[i] = ABOARD[i]
  rotate_bitmap(AT, AS, size)       # 時計回りに90度回転
  k = intncmp(ABOARD, AS, size)
  if k > 0:
    return
  if k == 0:
    nequiv = 2
  else:
    rotate_bitmap(AS, AT, size)   # 時計回りに180度回転
    k = intncmp(ABOARD, AT, size)
    if k > 0:
      return
    if k == 0:
      nequiv = 4
    else:
      rotate_bitmap(AT, AS, size) # 時計回りに270度回転
      k = intncmp(ABOARD, AS, size)
      if k > 0:
        return
      nequiv = 8
  # 回転・反転・対称チェックのためにboard配列をコピー
  for i in range(size):
    AS[i] = ABOARD[i]
  vmirror_bitmap(AS, AT, size)      # 垂直反転
  k = intncmp(ABOARD, AT, size)
  if k > 0:
    return
  if nequiv > 2:                    # -90度回転 対角鏡と同等
    rotate_bitmap(AT, AS, size)
    k = intncmp(ABOARD, AS, size)
    if k > 0:
      return
    if nequiv > 4:                # -180度回転 水平鏡像と同等
      rotate_bitmap(AS, AT, size)
      k = intncmp(ABOARD, AT, size)
      if k > 0:
        return                # -270度回転 反対角鏡と同等
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
# BackTrack2
def backtrack2(size, mask, row, left, down, right):  # pylint: disable=R0913
  """ backtrack2() """
  global ABOARD   # pylint: disable=W0603
  bit = 0
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
    for i in range(row, lim):    # pylint: disable=W0612
      while bitmap > 0:
        bit = (-bitmap&bitmap)
        ABOARD[row] = bit
        bitmap ^= ABOARD[row]
        backtrack2(size, mask, row+1, (left|bit)<<1, down|bit, (right|bit)>>1)
#
# BackTrack1
def backtrack1(size, mask, row, left, down, right):  # pylint: disable=R0913
  """ backtrack1() """
  global ABOARD       # pylint: disable=W0603
  bit = 0
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
    for i in range(row, lim):     # pylint: disable=W0612
      while bitmap:
        bit = (-bitmap&bitmap)
        ABOARD[row] = bit
        bitmap ^= ABOARD[row]
        backtrack1(size, mask, row+1, (left|bit)<<1, down|bit, (right|bit)>>1)
#
# メインメソッド
def nqueen(size, mask):
  """ nqueen() """
  global ABOARD       # pylint: disable=W0603
  global TOPBIT       # pylint: disable=W0603
  global ENDBIT       # pylint: disable=W0603
  global SIDEMASK     # pylint: disable=W0603
  global LASTMASK     # pylint: disable=W0603
  global BOUND1       # pylint: disable=W0603
  global BOUND2       # pylint: disable=W0603
  bit = 0
  TOPBIT = 1<<(size-1)
  ABOARD[0] = 1
  for BOUND1 in range(2, size-1):
    ABOARD[1] = bit = (1<<BOUND1)
    backtrack1(size, mask, 2, (2|bit)<<1, (1|bit), (bit>>1))
  SIDEMASK = LASTMASK = (TOPBIT|1)
  ENDBIT = (TOPBIT>>1)
  BOUND2 = size-2
  for BOUND1 in range(1, BOUND2):
    ABOARD[0] = bit = (1<<BOUND1)
    backtrack2(size, mask, 1, bit<<1, bit, bit>>1)
    LASTMASK |= LASTMASK>>1|LASTMASK<<1
    ENDBIT >>= 1
    BOUND2 -= 1
#
# メインメソッド
def main():
  """ main() """
  global COUNT2       # pylint: disable=W0603
  global COUNT4       # pylint: disable=W0603
  global COUNT8       # pylint: disable=W0603
  global ABOARD       # pylint: disable=W0603
  global MAX          # pylint: disable=W0603
  nmin = 4                          # Nの最小値（スタートの値）を格納
  print(" N:        Total       Unique        hh:mm:ss.ms")
  for i in range(nmin, MAX):
    COUNT2 = COUNT4 = COUNT8 = 0
    mask = (1<<i)-1
    for j in range(i):
      ABOARD[j] = j              # 盤を初期化
    start_time = datetime.now()
    nqueen(i, mask)
    time_elapsed = datetime.now()-start_time
    _text = '{}'.format(time_elapsed)
    text = _text[:-3]
    print("%2d:%13d%13d%20s" % (i, gettotal(), getunique(), text)) # 出力
#
main()
#
