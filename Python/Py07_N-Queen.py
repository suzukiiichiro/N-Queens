#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  /**
#    Cで学ぶアルゴリズムとデータ構造
#    ステップバイステップでＮ−クイーン問題を最適化
#    一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
#  
#   コンパイル
#   $ gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm C07_N-Queen.c -o C07_N-Queen
#  
#   実行
#   $ ./C07_N-Queen
#  
#   ７．バックトラック＋ビットマップ＋対称解除法
#  
#   *     一つの解には、盤面を９０度、１８０度、２７０度回転、及びそれらの鏡像の合計
#   *     ８個の対称解が存在する。対照的な解を除去し、ユニーク解から解を求める手法。
#   * 
#   * ■ユニーク解の判定方法
#   *   全探索によって得られたある１つの解が、回転・反転などによる本質的に変わること
#   * のない変換によって他の解と同型となるものが存在する場合、それを別の解とはしない
#   * とする解の数え方で得られる解を「ユニーク解」といいます。つまり、ユニーク解とは、
#   * 全解の中から回転・反転などによる変換によって同型になるもの同士をグループ化する
#   * ことを意味しています。
#   * 
#   *   従って、ユニーク解はその「個数のみ」に着目され、この解はユニーク解であり、こ
#   * の解はユニーク解ではないという定まった判定方法はありません。ユニーク解であるか
#   * どうかの判断はユニーク解の個数を数える目的の為だけに各個人が自由に定義すること
#   * になります。もちろん、どのような定義をしたとしてもユニーク解の個数それ自体は変
#   * わりません。
#   * 
#   *   さて、Ｎクイーン問題は正方形のボードで形成されるので回転・反転による変換パター
#   * ンはぜんぶで８通りあります。だからといって「全解数＝ユニーク解数×８」と単純には
#   * いきません。ひとつのグループの要素数が必ず８個あるとは限らないのです。Ｎ＝５の
#   * 下の例では要素数が２個のものと８個のものがあります。
#   *
#   *
#   * Ｎ＝５の全解は１０、ユニーク解は２なのです。
#   * 
#   * グループ１: ユニーク解１つ目
#   * - - - Q -   - Q - - -
#   * Q - - - -   - - - - Q
#   * - - Q - -   - - Q - -
#   * - - - - Q   Q - - - -
#   * - Q - - -   - - - Q -
#   * 
#   * グループ２: ユニーク解２つ目
#   * - - - - Q   Q - - - -   - - Q - -   - - Q - -   - - - Q -   - Q - - -   Q - - - -   - - - - Q
#   * - - Q - -   - - Q - -   Q - - - -   - - - - Q   - Q - - -   - - - Q -   - - - Q -   - Q - - -
#   * Q - - - -   - - - - Q   - - - Q -   - Q - - -   - - - - Q   Q - - - -   - Q - - -   - - - Q -
#   * - - - Q -   - Q - - -   - Q - - -   - - - Q -   - - Q - -   - - Q - -   - - - - Q   Q - - - -
#   * - Q - - -   - - - Q -   - - - - Q   Q - - - -   Q - - - -   - - - - Q   - - Q - -   - - Q - -
#   *
#   * 
#   *   それでは、ユニーク解を判定するための定義付けを行いますが、次のように定義する
#   * ことにします。各行のクイーンが右から何番目にあるかを調べて、最上段の行から下
#   * の行へ順番に列挙します。そしてそれをＮ桁の数値として見た場合に最小値になるもの
#   * をユニーク解として数えることにします。尚、このＮ桁の数を以後は「ユニーク判定値」
#   * と呼ぶことにします。
#   * 
#   * - - - - Q   0
#   * - - Q - -   2
#   * Q - - - -   4   --->  0 2 4 1 3  (ユニーク判定値)
#   * - - - Q -   1
#   * - Q - - -   3
#   * 
#   * 
#   *   探索によって得られたある１つの解(オリジナル)がユニーク解であるかどうかを判定
#   * するには「８通りの変換を試み、その中でオリジナルのユニーク判定値が最小であるか
#   * を調べる」ことになります。しかし結論から先にいえば、ユニーク解とは成り得ないこ
#   * とが明確なパターンを探索中に切り捨てるある枝刈りを組み込むことにより、３通りの
#   * 変換を試みるだけでユニーク解の判定が可能になります。
#   *  
#  
#    実行結果
#    N:        Total       Unique        hh:mm:ss.ms
#    4:            2            1         0:00:00.000
#    5:           10            2         0:00:00.000
#    6:            4            1         0:00:00.000
#    7:           40            6         0:00:00.001
#    8:           92           12         0:00:00.005
#    9:          352           46         0:00:00.025
#   10:          724           92         0:00:00.077
#   11:         2680          341         0:00:00.341
#   12:        14200         1787         0:00:01.981
#   13:        73712         9233         0:00:11.558
#   14:       365596        45752         0:01:05.960
#   15:      2279184       285053         0:07:25.447
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
    while bitmap:
      bit=(-bitmap&bitmap);
      aBoard[row]=bit;
      bitmap^=aBoard[row];
      NQueen(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
#
# メインメソッド
from datetime import datetime 
def main():
  global aBoard;
  global MAX;
  global COUNT2;
  global COUNT4;
  global COUNT8;
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