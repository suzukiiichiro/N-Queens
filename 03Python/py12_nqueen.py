#!/usr/bin/env python

# -*- coding: utf-8 -*-
""" py12_nqueen.py """

from datetime import datetime
#
# /**
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#
#
#  実行
#  $ python py12_nqueen.py
#
# １２．対称解除法の最適化
#   対称解除法 symmetryOps_bitmap()を最適化し、不要な関数を除去します
#  コメントアウト部分を参照して下さい。
#
#   実行結果
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
# グローバル変数
MAX = 16 # N = 15
ABOARD = [0 for i in range(MAX)]
# aT = [0 for i in range(MAX)]
# aS = [0 for i in range(MAX)]
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
#def dtob(score,size):
#  bit = 1
#  c = [0 for i in range(size)]
#  for i in range(size):
#    if score&bit :
#      c[i] = '1'
#    else :
#      c[i] = '0'
#    bit<< = 1
#  #for (int i = size-1i> = 0i--){ putchar(c[i]) }
#  for i in range(size-1,-1,-1):
#    putchar(c[i])
#  print("\n")
##
#def rh(a,size):
#  tmp = 0
#  for i in range(size+1):
#    if a&(1<<i) :
#      tmp| = (1<<size-i)
#  return tmp
##
#def vMirror_bitmap(bf,af,size):
#  score = 0
#  for i in range(size):
#    score = bf[i]
#    af[i] = rh(score,size-1)
##
#def rotate_bitmap(bf,af,size):
#  for i in range(size):
#    t = 0
#    for j in range(size):
#      t| = ((bf[j]>>i)&1)<<(size-j-1)  # x[j] の i ビット目を
#    af[i] = t                          # y[i] の j ビット目にする
##
#def intncmp(lt,rt,n):
#  rtn = 0
#  for i in range(n):
#    rtn = lt[i]-rt[i]
#    if rtn! = 0:
#      break
#  return rtn
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
def symmetryops(size):      # pylint: disable=R0912,R0911,R0915
  """ symmetryops() """
  global COUNT2  # pylint: disable=W0603
  global COUNT4  # pylint: disable=W0603
  global COUNT8  # pylint: disable=W0603
  global ABOARD  # pylint: disable=W0603
  global TOPBIT  # pylint: disable=W0603
  global ENDBIT  # pylint: disable=W0603
  own = 0
  ptn = 0
  you = 0
  bit = 0
  # 90度回転
  if ABOARD[BOUND2] == 1:
    own = 1
    ptn = 2
    while own <= size-1:
      bit = 1
      you = size-1
      while (ABOARD[you] != ptn) and (ABOARD[own] >= bit):
        bit <<= 1
        you -= 1
      if ABOARD[own] > bit:
        return
      if ABOARD[own] < bit:
        break
      own += 1
      ptn <<= 1
    #90度回転して同型なら180度/270度回転も同型である */
    if own > size-1:
      COUNT2 += 1
      return
  #180度回転
  if ABOARD[size-1] == ENDBIT:
    own = 1
    you = size-1-1
    while own <= size-1:
      bit = 1
      ptn = TOPBIT
      while (ABOARD[you] != ptn) and (ABOARD[own] >= bit):
        bit <<= 1
        ptn >>= 1
      if ABOARD[own] > bit:
        return
      if ABOARD[own] < bit:
        break
      own += 1
      you -= 1
    # 90度回転が同型でなくても180度回転が同型である事もある */
    if own > size-1:
      COUNT4 += 1
      return
  #270度回転
  if ABOARD[BOUND1] == TOPBIT:
    own = 1
    ptn = TOPBIT>>1
    while own <= size-1:
      bit = 1
      you = 0
      while (ABOARD[you] != ptn) and (ABOARD[own] >= bit):
        bit <<= 1
        you += 1
      if ABOARD[own] > bit:
        return
      if ABOARD[own] < bit:
        break
      own += 1
      ptn >>= 1
  COUNT8 += 1
#
#def symmetryOps_bitmap(si):
#  nEquiv = 0
#  global COUNT2
#  global COUNT4
#  global COUNT8
#  global aT
#  global aS
#  # 回転・反転・対称チェックのためにboard配列をコピー
#  for i in range(si):
#    aT[i] = ABOARD[i]
#  rotate_bitmap(aT,aS,si)    #時計回りに90度回転
#  k = intncmp(ABOARD,aS,si)
#  if k>0: return
#  if k =  = 0: nEquiv = 2
#  else:
#    rotate_bitmap(aS,aT,si)  #時計回りに180度回転
#    k = intncmp(ABOARD,aT,si)
#    if k>0:return
#    if k =  = 0: nEquiv = 4
#    else:
#      rotate_bitmap(aT,aS,si)#時計回りに270度回転
#      k = intncmp(ABOARD,aS,si)
#      if k>0: return
#      nEquiv = 8
#  # 回転・反転・対称チェックのためにboard配列をコピー
#  for i in range(si):
#    aS[i] = ABOARD[i]
#  vMirror_bitmap(aS,aT,si)   # 垂直反転
#  k = intncmp(ABOARD,aT,si)
#  if k>0: return
#  if nEquiv>2:                #-90度回転 対角鏡と同等
#    rotate_bitmap(aT,aS,si)
#    k = intncmp(ABOARD,aS,si)
#    if k>0: return
#    if nEquiv>4:              #-180度回転 水平鏡像と同等
#      rotate_bitmap(aS,aT,si)
#      k = intncmp(ABOARD,aT,si)
#      if k>0:  return        #-270度回転 反対角鏡と同等
#      rotate_bitmap(aT,aS,si)
#      k = intncmp(ABOARD,aS,si)
#      if k>0: return
#  if nEquiv =  = 2: COUNT2+ = 1
#  if nEquiv =  = 4: COUNT4+ = 1
#  if nEquiv =  = 8: COUNT8+ = 1
#
# BackTrack2
def backtrack2(size, mask, row, left, down, right): # pylint: disable=R0913
  """ backtrack2() """
  global ABOARD       # pylint: disable=W0603
  global LASTMASK     # pylint: disable=W0603
  global BOUND1       # pylint: disable=W0603
  global BOUND2       # pylint: disable=W0603
  bit = 0
  bitmap = mask&~(left|down|right)
  #枝刈り
  #if row == size:
  if row == size-1:
    if bitmap:
      #枝刈り
      if (bitmap&LASTMASK) == 0:
        ABOARD[row] = bitmap
        #symmetryOps_bitmap(size)
        symmetryops(size)
    #else:
    #  ABOARD[row] = bitmap
    #  symmetryOps_bitmap(size)
  else:
    #枝刈り 上部サイド枝刈り
    if row < BOUND1:
      bitmap &= ~SIDEMASK
    #枝刈り 下部サイド枝刈り
    elif row == BOUND2:
      if down&SIDEMASK == 0:
        return
      if down&SIDEMASK != SIDEMASK:
        bitmap &= SIDEMASK
    # 枝刈り
    if row != 0:
      lim = size
    else:
      lim = (size+1)//2 # 割り算の結果を整数にするには //
    # 枝刈り
    for i in range(row, lim):       # pylint: disable=W0612
      while bitmap:
        bit = (-bitmap&bitmap)
        ABOARD[row] = bit
        bitmap ^= ABOARD[row]
        backtrack2(size, mask, row+1, (left|bit)<<1, down|bit, (right|bit)>>1)
#
# BackTrack1
def backtrack1(size, mask, row, left, down, right):  # pylint: disable=R0913
  """ backtrack1() """
  global ABOARD       # pylint: disable=W0603
  global COUNT8       # pylint: disable=W0603
  global BOUND1       # pylint: disable=W0603
  bit = 0
  bitmap = mask&~(left|down|right)
  #枝刈り
  #if row =  = size:
  if row == size-1:
    if bitmap:
      ABOARD[row] = bitmap
      #枝刈りにてCOUNT8に加算
      COUNT8 += 1
    #else:
    #  ABOARD[row] = bitmap
    #  symmetryOps_bitmap(size)
  else:
    #枝刈り 鏡像についても主対角線鏡像のみを判定すればよい
    # ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
    if row < BOUND1:
      bitmap &= ~2 # bm| = 2 bm^ = 2 (bm& = ~2と同等)
    # 枝刈り
    if row != 0:
      lim = size
    else:
      lim = (size+1)//2 # 割り算の結果を整数にするには //
    # 枝刈り
    for i in range(row, lim):   # pylint: disable=W0612
      while bitmap:
        bit = (-bitmap&bitmap)
        ABOARD[row] = bit
        bitmap ^= ABOARD[row]
        backtrack1(size, mask, row+1, (left|bit)<<1, down|bit, (right|bit)>>1)
  #
# メインメソッド
def nqueen(size, mask):
  """ nqueen() """
  # global ABOARD         # pylint: disable=W0603
  global TOPBIT         # pylint: disable=W0603
  global ENDBIT         # pylint: disable=W0603
  global SIDEMASK       # pylint: disable=W0603
  global LASTMASK       # pylint: disable=W0603
  global BOUND1         # pylint: disable=W0603
  global BOUND2         # pylint: disable=W0603
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
  global COUNT2     # pylint: disable=W0603
  global COUNT4     # pylint: disable=W0603
  global COUNT8     # pylint: disable=W0603
  global ABOARD     # pylint: disable=W0603
  global MAX        # pylint: disable=W0603
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
