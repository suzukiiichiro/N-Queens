#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# /**
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
# 
#  実行
#  $ python Py13_N-Queen.py
#
# １３．並列処理：pthreadと構造体
#
#   実行結果
#
#
# グローバル変数
MAX=16; # N=15
aBoard=[0 for i in range(MAX)];
aT=[0 for i in range(MAX)];
aS=[0 for i in range(MAX)];
bit=0;
COUNT2=0;
COUNT4=0;
COUNT8=0;
TOPBIT=0;
ENDBIT=0;
SIDEMASK=0;
LASTMASK=0;
#
# ユニーク値を出力
def getUnique():
  global COUNT2;
  global COUNT4;
  global COUNT8;
  return COUNT2+COUNT4+COUNT8;
#
# 合計を出力
def getTotal():
  global COUNT2;
  global COUNT4;
  global COUNT8;
  return COUNT2*2+COUNT4*4+COUNT8*8;
#
# 対称解除法
def symmetryOps(size):
  global COUNT2;
  global COUNT4;
  global COUNT8;
  global aBoard;
  global TOPBIT;
  global ENDBIT;
  own=0;
  ptn=0;
  you=0
  bit=0;
  # 90度回転
  if aBoard[BOUND2]==1:
    own=1; 
    ptn=2;
    while own<=size-1:
      bit=1; 
      you=size-1;
      while (aBoard[you]!=ptn) and (aBoard[own]>=bit):
        bit<<=1;
        you-=1;
      if aBoard[own]>bit:
        return;
      if aBoard[own]<bit:
        break;
      own+=1; 
      ptn<<=1;
    #90度回転して同型なら180度/270度回転も同型である */
    if own>size-1:
      COUNT2+=1; 
      return;
  #180度回転
  if aBoard[size-1]==ENDBIT:
    own=1;
    you=size-1-1;
    while own<=size-1:
      bit=1; 
      ptn=TOPBIT;
      while (aBoard[you]!=ptn) and (aBoard[own]>=bit):
        bit<<=1; 
        ptn>>=1;
      if aBoard[own]>bit:
        return; 
      if aBoard[own]<bit:
        break;
      own+=1; 
      you-=1;
    # 90度回転が同型でなくても180度回転が同型である事もある */
    if own>size-1:
      COUNT4+=1; 
      return;
  #270度回転
  if aBoard[BOUND1]==TOPBIT:
    own=1; 
    ptn=TOPBIT>>1;
    while own<=size-1:
      bit=1;
      you=0;
      while (aBoard[you]!=ptn) and (aBoard[own]>=bit):
        bit<<=1; 
        you+=1; 
      if aBoard[own]>bit:
        return;
      if aBoard[own]<bit:
        break;
      own+=1; 
      ptn>>=1;
  COUNT8+=1;
#
# BackTrack2
def backTrack2(size,mask,row,left,down,right):
  global aBoard;
  global LASTMASK;
  global BOUND1;
  global BOUND2;
  bit=0;
  bitmap=mask&~(left|down|right);
  #枝刈り
  #if row==size:
  if row==size-1:
    if bitmap:
      #枝刈り
      if (bitmap&LASTMASK)==0:
        aBoard[row]=bitmap;
        #symmetryOps_bitmap(size);
        symmetryOps(size);
    #else:
    #  aBoard[row]=bitmap;
    #  symmetryOps_bitmap(size);
  else:
    #枝刈り 上部サイド枝刈り
    if row<BOUND1:
      bitmap&=~SIDEMASK;
    #枝刈り 下部サイド枝刈り
    elif row==BOUND2:
      if down&SIDEMASK==0:
        return;
      if down&SIDEMASK!=SIDEMASK:
        bitmap&=SIDEMASK;
    # 枝刈り
    if row!=0:
      lim=size;
    else:
      lim=(size+1)//2; # 割り算の結果を整数にするには // 
    # 枝刈り
    for i in range(row,lim):
      while bitmap:
        bit=(-bitmap&bitmap);
        aBoard[row]=bit;
        bitmap^=aBoard[row];
        backTrack2(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
#
# BackTrack1
def backTrack1(size,mask,row,left,down,right):
  global aBoard;
  global COUNT8;
  global BOUND1;
  bit=0;
  bitmap=mask&~(left|down|right);
  #枝刈り
  #if row==size:
  if row==size-1:
    if bitmap:
      aBoard[row]=bitmap;
      #枝刈りにてCOUNT8に加算
      COUNT8+=1;
    #else:
    #  aBoard[row]=bitmap;
    #  symmetryOps_bitmap(size);
  else:
		#枝刈り 鏡像についても主対角線鏡像のみを判定すればよい
		# ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
    if row<BOUND1:
      bitmap&=~2; # bm|=2; bm^=2; (bm&=~2と同等)
    # 枝刈り
    if row!=0:
      lim=size;
    else:
      lim=(size+1)//2; # 割り算の結果を整数にするには // 
    # 枝刈り
    for i in range(row,lim):
      while bitmap:
        bit=(-bitmap&bitmap);
        aBoard[row]=bit;
        bitmap^=aBoard[row];
        backTrack1(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
#
# メインメソッド
from multiprocessing import Pool
def NQueen(size,mask):
  global aBoard;
  global TOPBIT;
  global ENDBIT;
  global SIDEMASK;
  global LASTMASK;
  global BOUND1;
  global BOUND2;
  bit=0;
  TOPBIT=1<<(size-1);
  aBoard[0]=1;
  with Pool(processes=1) as pool:
    for BOUND1 in range(2,size-1):
      aBoard[1]=bit=(1<<BOUND1);
      #backTrack1(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
      pool.map(backTrack1(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1)), range(BOUND1))
    SIDEMASK=LASTMASK=(TOPBIT|1);
    ENDBIT=(TOPBIT>>1);
    BOUND2=size-2;
    for BOUND1 in range(1,BOUND2):
      aBoard[0]=bit=(1<<BOUND1);
      #backTrack2(size,mask,1,bit<<1,bit,bit>>1);
      pool.map(backTrack2(size,mask,1,bit<<1,bit,bit>>1), range(BOUND2))
      LASTMASK|=LASTMASK>>1|LASTMASK<<1;
      ENDBIT>>=1;
      BOUND2-=1;
#
# メインメソッド
from datetime import datetime 
def main():
  global COUNT2;
  global COUNT4;
  global COUNT8;
  global aBoard;
  global MAX;
  min=4;                          # Nの最小値（スタートの値）を格納
  print(" N:        Total       Unique        hh:mm:ss.ms");
  for i in range(min,MAX):
      COUNT2=COUNT4=COUNT8=0;
      mask=(1<<i)-1;
      for j in range(i):
        aBoard[j]=j;              # 盤を初期化
      start_time = datetime.now() 
      NQueen(i,mask);
      time_elapsed=datetime.now()-start_time 
      _text='{}'.format(time_elapsed)
      text=_text[:-3]
      print("%2d:%13d%13d%20s" % (i,getTotal(),getUnique(),text)); # 出力
#
main();
#