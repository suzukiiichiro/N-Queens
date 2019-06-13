#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# /**
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
# 
#  実行
#  $ python Py08_N-Queen.py
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
MAX=16; # N=15
aBoard=[0 for i in range(MAX)];
aT=[0 for i in range(MAX)];
aS=[0 for i in range(MAX)];
bit=0;
COUNT2=0;
COUNT4=0;
COUNT8=0;
#
def dtob(score,size):
  bit=1; 
  c=[0 for i in range(size)];
  for i in range(size):
    if score&bit :
      c[i]='1';
    else : 
      c[i]='0';
    bit<<=1;
  #for (int i=size-1;i>=0;i--){ putchar(c[i]); }
  for i in range(size-1,-1,-1):
    putchar(c[i]);
  print("\n");
#
def rh(a,size):
  tmp=0;
  for i in range(size+1):
    if a&(1<<i) : 
      tmp|=(1<<size-i);
  return tmp;
#
def vMirror_bitmap(bf,af,size):
  score=0;
  for i in range(size):
    score=bf[i];
    af[i]=rh(score,size-1);
#
def rotate_bitmap(bf,af,size):
  for i in range(size):
    t=0;
    for j in range(size):
      t|=((bf[j]>>i)&1)<<(size-j-1);  # x[j] の i ビット目を
    af[i]=t;                          # y[i] の j ビット目にする
#
def intncmp(lt,rt,n):
  rtn=0;
  for i in range(n):
    rtn=lt[i]-rt[i];
    if rtn!=0:
      break;
  return rtn;
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
def symmetryOps_bitmap(si):
  nEquiv=0;
  global COUNT2;
  global COUNT4;
  global COUNT8;
  global aT;
  global aS;
  # 回転・反転・対称チェックのためにboard配列をコピー
  for i in range(si): aT[i]=aBoard[i];
  rotate_bitmap(aT,aS,si);    #時計回りに90度回転
  k=intncmp(aBoard,aS,si);
  if k>0: return;
  if k==0: 
    nEquiv=2;
  else:
    rotate_bitmap(aS,aT,si);  #時計回りに180度回転
    k=intncmp(aBoard,aT,si);
    if k>0:return;
    if k==0: nEquiv=4;
    else:
      rotate_bitmap(aT,aS,si);#時計回りに270度回転
      k=intncmp(aBoard,aS,si);
      if k>0: return;
      nEquiv=8;
  # 回転・反転・対称チェックのためにboard配列をコピー
  for i in range(si): aS[i]=aBoard[i];
  vMirror_bitmap(aS,aT,si);   # 垂直反転
  k=intncmp(aBoard,aT,si);
  if k>0: return;
  if nEquiv>2:                #-90度回転 対角鏡と同等
    rotate_bitmap(aT,aS,si);
    k=intncmp(aBoard,aS,si);
    if k>0: return;
    if nEquiv>4:              #-180度回転 水平鏡像と同等
      rotate_bitmap(aS,aT,si);
      k=intncmp(aBoard,aT,si);
      if k>0:  return;        #-270度回転 反対角鏡と同等
      rotate_bitmap(aT,aS,si);
      k=intncmp(aBoard,aS,si);
      if k>0: return;
  if nEquiv==2: COUNT2+=1;
  if nEquiv==4: COUNT4+=1;
  if nEquiv==8: COUNT8+=1;
#
# ロジックメソッド
def NQueen(size,mask,row,left,down,right):
  global aBoard;
  bitmap=mask&~(left|down|right);
  if row==size:
    if bitmap:
      pass;
    else:
      aBoard[row]=bitmap;
      symmetryOps_bitmap(size);
  else:
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
        NQueen(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
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
      for j in range(i):
        aBoard[j]=j;              # 盤を初期化
      mask=(1<<i)-1;
      start_time = datetime.now() 
      NQueen(i,mask,0,0,0,0);
      time_elapsed=datetime.now()-start_time 
      _text='{}'.format(time_elapsed)
      text=_text[:-3]
      print("%2d:%13d%13d%20s" % (i,getTotal(),getUnique(),text)); # 出力
#
main();
#
