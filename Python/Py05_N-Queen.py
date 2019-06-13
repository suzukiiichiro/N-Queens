#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  /**
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#  
#   実行
#   $ python Py05_N-Queen.py
#  
#   * ５．バックトラック＋対称解除法＋枝刈りと最適化
#   *
#   * 　単純ですのでソースのコメントを見比べて下さい。
#   *   単純ではありますが、枝刈りの効果は絶大です。
#  
#   実行結果
# N:        Total       Unique        hh:mm:ss.ms
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
MAX=16; # N=15
TOTAL=0;
UNIQUE=0;
aBoard=[0 for i in range(MAX)];
#fA=[0 for i in range(2*MAX-1)];        # 縦列にクイーンを一つだけ配置
fB=[0 for i in range(2*MAX-1)];         # 斜め列にクイーンを一つだけ配置
fC=[0 for i in range(2*MAX-1)];         # 斜め列にクイーンを一つだけ配置
aT=[0 for i in range(MAX)];             # aT:aTrial[]
aS=[0 for i in range(MAX)];             # aS:aScrath[]
#
# 回転
def rotate(chk,scr,n,neg):
  incr=0;
	#int k=neg ? 0 : n-1;
  k=0 if neg else n-1;
	#int incr=(neg ? +1 : -1);
  incr=1 if neg else -1;
  for i in range(n):
    scr[i]=chk[k];
    k+=incr;
  k=n-1 if neg else 0;
  for i in range(n):
    chk[scr[i]]=k;
    k-=incr;
#
# 反転
def vMirror(chk,n):
	for i in range(n):
		chk[i]=(n-1)-chk[i];
#
#
def intncmp(lt,rt,n):
	rtn=0;
	for i in range(n):
		rtn=lt[i]-rt[i];
		if rtn!=0: break;
	return rtn;
#
# 対称解除法
def symmetryOps(size):
  global aBoard;
  global aT;
  global aS;
  nEquiv=0;
	# 回転・反転・対称チェックのためにboard配列をコピー
  for i in range(size):
    aT[i]=aBoard[i];
  # 時計回りに90度回転
  rotate(aT,aS,size,0);
  k=intncmp(aBoard,aT,size);
  if k>0: return 0;
  if k==0: nEquiv=1;
  else:
    # 時計回りに180度回転
    rotate(aT,aS,size,0);
    k=intncmp(aBoard,aT,size);
    if k>0: return 0;
    if k==0: nEquiv=2;
    else:
      # 時計回りに270度回転
      rotate(aT,aS,size,0);
      k=intncmp(aBoard,aT,size);
      if k>0: return 0;
      nEquiv=4;
	# 回転・反転・対称チェックのためにboard配列をコピー
  for i in range(size):
    aT[i]=aBoard[i];
  # 垂直反転
  vMirror(aT,size);
  k=intncmp(aBoard,aT,size);
  if k>0: return 0;
  # -90度回転 対角鏡と同等
  if nEquiv>1:
    rotate(aT,aS,size,1);
    k=intncmp(aBoard,aT,size);
    if k>0: return 0;
    # -180度回転 水平鏡像と同等
    if nEquiv>2:
      rotate(aT,aS,size,1);
      k=intncmp(aBoard,aT,size);
      if k>0: return 0;
      # -270度回転 反対角鏡と同等
      rotate(aT,aS,size,1);
      k=intncmp(aBoard,aT,size);
      if k>0 : return 0;
  return nEquiv*2;
#
# ロジックメソッド
def NQueen(row,size):
  global aBoard;
  global TOTAL;
  global UNIQUE;
  #global fA;
  global fB;
  global fC;
  # if row==size:
  if row==size-1: # 枝刈り
    # 追加
    if fB[row-aBoard[row]+size-1] or fC[row+aBoard[row]] :
      return;
    #
    s=symmetryOps(size);	  # 対称解除法の導入
    if s!=0:
      UNIQUE+=1;            # ユニーク解を加算
      TOTAL+=s;             # 対称解除で得られた解数を加算
  else:
    # 枝刈り
    # pythonでは割り算の結果を整数で返すときは // にする
		#int lim=(row!=0) ? size : (size+1)/2;
    lim=size if row!=0 else (size+1)//2; 
    #for i in range(size):  # 枝刈り
    for i in range(row,lim):
      # aBoard[row]=i;      # 不要
      # 交換
      tmp=aBoard[i];
      aBoard[i]=aBoard[row];
      aBoard[row]=tmp;
      #
      # 枝刈り
      if fB[row-aBoard[row]+size-1] or fC[row+aBoard[row]]:
        pass
      else:
        fB[row-aBoard[row]+size-1]=fC[row+aBoard[row]]=1;
        NQueen(row+1,size); # 再帰
        fB[row-aBoard[row]+size-1]=fC[row+aBoard[row]]=0;
	  # 交換
    tmp=aBoard[row];
    for i in range(row+1,size):
      aBoard[i-1]=aBoard[i];
    aBoard[size-1]=tmp;
#
# メインメソッド
from datetime import datetime 
def main():
  global TOTAL;
  global UNIQUE;
  global aBoard;
  global MAX;
  min=4;                          # Nの最小値（スタートの値）を格納
  print(" N:        Total       Unique        hh:mm:ss.ms");
  for i in range(min,MAX):
      TOTAL=0;
      UNIQUE=0;                   # 初期化
      for j in range(i):
        aBoard[j]=j;              # 盤を初期化
      start_time = datetime.now() 
      NQueen(0,i)
      time_elapsed=datetime.now()-start_time 
      _text='{}'.format(time_elapsed)
      text=_text[:-3]
      print("%2d:%13d%13d%20s" % (i,TOTAL,UNIQUE,text)); # 出力
#
main();
#
