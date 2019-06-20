#!/usr/bin/env python

# -*- coding: utf-8 -*-
#
import logging
import threading
import time
from threading import Thread

# /**
#   Pythonで学ぶアルゴリズムとデータ構造
#   ステップバイステップでＮ−クイーン問題を最適化
#   一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
# 
#  実行
#  $ python Py13_N-Queen.py
#
# １３．マルチスレッド
#
#        Step:1 一つのソースクラスに分解
#        Step:2 グローバル変数などを整理
#        Step:3 ロジックメソッドNQueen()を分解
#        Step:4 WorkingEngineクラスのコンストラクタを準備
#        Step:5 runメソッドの準備（シングルスレッド）
#        Step:6 runメソッド　マルチスレッドの実装
#        Step:7 マルチプロセス
#        Step:8 排他制御
#
#  シングルスレッド
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.000
#  5:           10            2         0:00:00.000
#  6:            4            1         0:00:00.000
#  7:           40            6         0:00:00.000
#  8:           92           12         0:00:00.000
#  9:          352           46         0:00:00.002
# 10:          724           92         0:00:00.010
# 11:         2680          341         0:00:00.053
# 12:        14200         1787         0:00:00.265
# 13:        73712         9233         0:00:01.420
# 14:       365596        45752         0:00:08.174
# 15:      2279184       285053         0:00:51.030
#
#  マルチスレッド 数が合わない！
#  N:        Total       Unique        hh:mm:ss.ms
#  4:            2            1         0:00:00.000
#  5:           10            2         0:00:00.001
#  6:            4            1         0:00:00.001
#  7:           40            6         0:00:00.000
#  8:           92           12         0:00:00.001
#  9:          352           46         0:00:00.003
# 10:          724           92         0:00:00.013
# 11:         2324          294         0:00:00.051
# 12:         5948          747         0:00:00.147
# 13:        16152         2019         0:00:00.301
# 14:        30808         3851         0:00:00.610
# 15:        70760         8845         0:00:01.495
#
#
# Step:2 除去
# グローバル変数
# MAX=16; # N=15
# aBoard=[0 for i in range(MAX)];
# bit=0;
# COUNT2=0;
# COUNT4=0;
# COUNT8=0;
# TOPBIT=0;
# ENDBIT=0;
# SIDEMASK=0;
# LASTMASK=0;

# ---------------------------------
# ボードの値を保存するボードクラス
# ---------------------------------
# 追加 Step:1 ボードクラスの作成
class Board:
  # 追加 Step:1 コンストラクタの追加
  # コンストラクタ
  def __init__(self,lock):
    global COUNT2; global COUNT4; global COUNT8; 
    COUNT2=0;
    COUNT4=0;
    COUNT8=0;
    self.lock=lock;
  # 追加 Step:1 加算するメソッドを追加
  def setCount(self,C8,C4,C2):
    with self.lock:
      global COUNT2; global COUNT4; global COUNT8; 
      COUNT8+=C8;
      COUNT4+=C4;
      COUNT2+=C2;
  # ユニーク値を出力
  # Step:1 パラメータにselfの追加
  #def getUnique():
  def getUnique(self):
    return COUNT2+COUNT4+COUNT8;
  #
  # 合計を出力
  # Step:1 パラメータにselfの追加
  # def getTotal():
  def getTotal(self):
    return COUNT2*2+COUNT4*4+COUNT8*8;
#
#
# ---------------------------------
# スレッドクラス
# ---------------------------------
#
# 追加 Step:1 スレッドクラスの作成
#class WorkingEngine:
#
# Step:5 スレッドの構築
class WorkingEngine(Thread):

  # デバッグ用出力フォーマット
  logging.basicConfig(level=logging.DEBUG,
    format='[%(levelname)s] (%(threadName)-10s) %(message)s',)
#
# Step:7 マルチプロセス
# from multiprocessing import Process
#class WorkingEngine(Process):
#
# Step:7 マルチプロセス
  #
  # 追加 Step:1 コンストラクタの追加
  #def __init__(self,size,info):
  #
  # Step:4 
  # コンストラクタのパラメータの追加
  def __init__(self,size,nMore,info,B1,B2,bThread):
    #
    # Step:5 スレッドの構築
    super(WorkingEngine, self).__init__();
    # メイン終了時にもスレッド終了する
    self.daemon = True  
    #
    # Step:2 グローバル変数の値を変更するので宣言が必要です
    global SIZE; SIZE=size;
    global aBoard; aBoard=[0 for i in range(size)];
    global MASK; MASK=(1<<size)-1;
    global INFO; INFO=info;
    #
    # Step:3 グローバル変数の追加
    global NMORE; NMORE=nMore;
    global child; child=None;
    # 
    # Step:5 B1,B2の処理を追加
    global BOUND1; BOUND1=B1;
    global BOUND2; BOUND2=B2;
    #
    # 追加 Step:1 グローバル変数の追加
    for i in range(size):
      aBoard[i]=i;              # 盤を初期化
    #
    # Step:4 スレッドの生成
    if nMore>0 :
      if bThread:
        child=WorkingEngine(size,nMore-1,info,B1-1,B2+1,bThread);
        # child=threading.Thread(target=self.WorkingEngine,args=(size,nMore-1,info,B1-1,B2+1,bThread),name=nMore);
        BOUND1=B1;
        BOUND2=B2;
	      # Step:6 
        child.start();
        # マルチスレッド
        # ここでjoin()はマルチスレッドの意味がない
        # child.join();  
        # 処理の確認は、 child.join()を活かし、さらにrun()の末尾
        # をコメントアウトして実行
        # 
        #main_thread = threading.currentThread()
        #for t in threading.enumerate():
        #  if t is not main_thread:
        #    t.join()
        #
        # ロギング用
        #logging.debug('start()');
      else:
        child=None;
        self.run();
       
        

  # Step:5 スレッドクラス実行で稼働するrunメソッドの実装
  def run(self):
    global aBoard; global SIZE; global SIZEE;
    global BOUND1; global BOUND2;
    global TOPBIT; global ENDBIT; global SIDEMASK; global LASTMASK;
    global CHILD; global NMORE;
    #
    # シングルスレッド
    if child==None:
      if NMORE>0 :
        aBoard[0]=1;
        SIZEE=SIZE-1;
        TOPBIT=1<<(SIZE-1);
        BOUND1=2;
        while BOUND1>1 and BOUND1<SIZEE :
          self.BOUND1(BOUND1);
          BOUND1+=1;
        SIDEMASK=LASTMASK=(TOPBIT|1);
        ENDBIT=(TOPBIT>>1);
        BOUND1=1;
        BOUND2=SIZE-2;
        while BOUND1>0 and BOUND2<SIZE-1 and BOUND1<BOUND2:
          self.BOUND2(BOUND1,BOUND2);
          BOUND1+=1;
          BOUND2-=1;
    else: # マルチスレッド
			#
      # Step:6 マルチスレッドの実装
      aBoard[0]=1;
      SIZEE=SIZE-1;
      MASK=(1<<SIZE)-1;
      TOPBIT=1<<SIZEE;
      if BOUND1>1 and BOUND1<SIZEE:
        self.BOUND1(BOUND1);
      ENDBIT=(TOPBIT>>BOUND1);
      SIDEMASK=LASTMASK=(TOPBIT|1);
      if BOUND1>0 and BOUND2<SIZE-1 and BOUND1<BOUND2:
        for i in range(1,BOUND1):
          LASTMASK=LASTMASK|LASTMASK>>1|LASTMASK<<1;
        self.BOUND2(BOUND1,BOUND2);
        ENDBIT>>=NMORE;
      #
      # Step:6 
      # マルチスレッドでのjoin()
      # CHILD.join(); # これではうごかない！
      main_thread = threading.currentThread()
      for t in threading.enumerate():
        if t is not main_thread:
          t.join()
  #
  # 対称解除法
  # Step:1 パラメータに selfを追加してsizeをグローバルへ変更
  #def symmetryOps(size):
  def symmetryOps(self):
    # Step:1 ボードクラスにカウント加算用メソッドを作成したので不要
    # global COUNT2;
    # global COUNT4;
    # global COUNT8;
    # Step:1 グローバル変数を追加

    # Step:2 グローバル変数の参照のみであれば宣言は不要です。
    # global INFO;
    # global aBoard;
    # global BOUND1;
    # global BOUND2;
    # global TOPBIT;
    # global ENDBIT;
    # global SIZE;
    own=0;
    ptn=0;
    you=0
    bit=0;
    # 90度回転
    if aBoard[BOUND2]==1:
      own=1; 
      ptn=2;
      while own<=SIZE-1:
        bit=1; 
        you=SIZE-1;
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
      if own>SIZE-1:
        # Step:1 カウンター処理を変更
        # COUNT2+=1; 
        INFO.setCount(0,0,1);
        return;
    #180度回転
    if aBoard[SIZE-1]==ENDBIT:
      own=1;
      you=SIZE-1-1;
      while own<=SIZE-1:
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
      if own>SIZE-1:
        # Step:1 カウンター処理を変更
        #COUNT4+=1; 
        INFO.setCount(0,1,0);
        return;
    #270度回転
    if aBoard[BOUND1]==TOPBIT:
      own=1; 
      ptn=TOPBIT>>1;
      while own<=SIZE-1:
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
    # Step:1 カウンター処理を変更
    #COUNT8+=1;
    INFO.setCount(1,0,0);
  #
  # BackTrack2
  # Step:1 パラメータに selfを追加
  #def backTrack2(size,mask,row,left,down,right):
  #
  # Step:1 size/maskをグローバルへ変更
  #def backTrack2(self,size,mask,row,left,down,right):
  def backTrack2(self,row,left,down,right):
    global aBoard;
    #
    # Step:2 グローバル変数の参照のみなら宣言は不要です。
    # global SIZE;
    # global MASK;
    # global LASTMASK;
    # global BOUND1;
    # global BOUND2;
    bit=0;
    bitmap=MASK&~(left|down|right);
    #枝刈り
    #if row==SIZE:
    if row==SIZE-1:
      if bitmap:
        #枝刈り
        if (bitmap&LASTMASK)==0:
          aBoard[row]=bitmap;
          #symmetryOps_bitmap(SIZE);
          #
          # Step:1 パラメータをなくして self呼び出しへ変更
          #symmetryOps(SIZE);
          self.symmetryOps();
      #else:
      #  aBoard[row]=bitmap;
      #  symmetryOps_bitmap(SIZE);
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
        lim=SIZE;
      else:
        lim=(SIZE+1)//2; # 割り算の結果を整数にするには // 
      # 枝刈り
      for i in range(row,lim):
        while bitmap:
          bit=(-bitmap&bitmap);
          aBoard[row]=bit;
          bitmap^=aBoard[row];
          #
          # Step:1 backTrack2の呼び出しを self.backTrack2へ変更
          #backTrack2(SIZE,MASK,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
          #
          # Step:1 SIZE,MASKをグローバルへ変更したことによりパラメータ除去
          # self.backTrack2(SIZE,MASK,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
          self.backTrack2(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
  #
  # BackTrack1
  # Step:1 パラメータに selfを追加
  #def backTrack1(size,mask,row,left,down,right):
  #
  # Step:1 size/maskをグローバルへ変更
  #def backTrack1(self,size,mask,row,left,down,right):
  def backTrack1(self,row,left,down,right):
    global aBoard;
    # Step:2 グローバル変数参照のみであれば宣言は不要です
    # global SIZE;
    # global MASK;
    # global BOUND1;
    #
    # Step:1 加算方法変更のため除去
    # global COUNT8;
    bit=0;
    bitmap=MASK&~(left|down|right);
    #枝刈り
    #if row==SIZE:
    if row==SIZE-1:
      if bitmap:
        aBoard[row]=bitmap;
        #枝刈りにてCOUNT8に加算
        #Step:1 加算方法を setCount()へ変更
        #COUNT8+=1;
        INFO.setCount(1,0,0);
      #else:
      #  aBoard[row]=bitmap;
      #  symmetryOps_bitmap(SIZE);
    else:
      #枝刈り 鏡像についても主対角線鏡像のみを判定すればよい
      # ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      if row<BOUND1:
        bitmap&=~2; # bm|=2; bm^=2; (bm&=~2と同等)
      # 枝刈り
      if row!=0:
        lim=SIZE;
      else:
        lim=(SIZE+1)//2; # 割り算の結果を整数にするには // 
      # 枝刈り
      for i in range(row,lim):
        while bitmap:
          bit=(-bitmap&bitmap);
          aBoard[row]=bit;
          bitmap^=aBoard[row];
          # Step:1 backTrack1 の呼び出しを selfに変更
          #backTrack1(SIZE,MASK,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
          # Step:1 SIZE,MASKをグローバルへ変更したことによりパラメータ除去
          #self.backTrack1(SIZE,MASK,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
          self.backTrack1(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
  #
  # Step:3 ロジックメソッドを分解しスレッド対応の準備をします。
  # 追加
  def BOUND2(self,B1,B2):
    global aBoard;
    global BOUND1;
    global BOUND2;
    global LASTMASK;
    global ENDBIT;
    bit=0;
    BOUND1=B1;
    BOUND2=B2;
    aBoard[0]=bit=(1<<B1);
    self.backTrack2(1,bit<<1,bit,bit>>1);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  #
  # Step:3 ロジックメソッドを分解しスレッド対応の準備をします。
  # 追加
  def BOUND1(self,B1):
    global aBoard;
    global BOUND1;
    bit=0;
    BOUND1=B1;
    aBoard[1]=bit=(1<<B1);
    self.backTrack1(2,(2|bit)<<1,(1|bit),bit>>1);
  #
  # メインメソッド
  # Step:1 パラメータに selfの追加
  #def NQueen(size,mask):
  def NQueen(self):
    #
    #Step:1 グローバル変数SIZE/MASK の追加
    #
    #Step:2 グローバル変数の参照のみであればSIZE/MASKの宣言は不要です。
    # global SIZE;
    # global MASK;
    global aBoard;
    global TOPBIT;
    global ENDBIT;
    global SIDEMASK;
    global LASTMASK;
    global BOUND1;
    global BOUND2;
    #
    # Step:1 sizeをグローバル変数SIZEへ変更
    #
    # Step:1 maskをグローバル変数MASKへ変更
    if CHILD==None:
      if NMORE>0 :
        bit=0;
        TOPBIT=1<<(SIZE-1);
        aBoard[0]=1;
        #
        # Step:3 以下の変数を追加
        SIZEE=SIZE-1;
        #
        # Step:3 ロジックメソッドの分解
        # for BOUND1 in range(2,SIZE-1):
        #   aBoard[1]=bit=(1<<BOUND1);
        #   # Step:1 backTrack1の呼び出しを self.backTrack1へ変更
        #   #backTrack1(SIZE,MASK,2,(2|bit)<<1,(1|bit),(bit>>1));
        #   # 
        #   # Step:1 SIZE,MASKをグローバルへ変更したことによりパラメータ除去
        #   #self.backTrack1(SIZE,MASK,2,(2|bit)<<1,(1|bit),(bit>>1));
        #   self.backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1));
        # 追加
        BOUND1=2;
        while BOUND1>1 and BOUND1<SIZEE :
          self.BOUND1(BOUND1);
          BOUND1+=1;


        SIDEMASK=LASTMASK=(TOPBIT|1);
        ENDBIT=(TOPBIT>>1);
        BOUND2=SIZE-2;
        #
        # Step:3 ロジックメソッドの分解
        # for BOUND1 in range(1,BOUND2):
        #   aBoard[0]=bit=(1<<BOUND1);
        #   # Step:1 backTrack2の呼び出しを self.backTrack2へ変更
        #   #backTrack2(SIZE,MASK,1,bit<<1,bit,bit>>1);
        #   #
        #   # Step:1 SIZE,MASKをグローバルへ変更したことによりパラメータ除去
        #   #self.backTrack2(SIZE,MASK,1,bit<<1,bit,bit>>1);
        #   self.backTrack2(1,bit<<1,bit,bit>>1);
        #   LASTMASK|=LASTMASK>>1|LASTMASK<<1;
        #   ENDBIT>>=1;
        #   BOUND2-=1;
        #
        # 追加
        BOUND1=1;
        while BOUND1>0 and BOUND2<SIZE-1 and BOUND1<BOUND2:
          self.BOUND2(BOUND1,BOUND2);
          BOUND1+=1;
          BOUND2-=1;
        # 追加
        #
#
# メインメソッド
from datetime import datetime 
def main():
  #
  # Step:1 カウンター処理方法変更のため以下は不要です。
  # global COUNT2;
  # global COUNT4;
  # global COUNT8;
  #
  # Step:1 スレッドクラスで初期化するのでaBoardの宣言は不要です。
  # global aBoard;
  #
  # Step:2 グローバル変数MAXをmaxに置き換え
  # global MAX;
  max=16;
  min=4;                          # Nの最小値（スタートの値）を格納
  print(" N:        Total       Unique        hh:mm:ss.ms");
  #
  # Step:4 スレッドの切り換え 現在は Falseで
  # Step:5 スレッドオフ / False
  # bThread=False;
  # 
  # Step:6 マルチスレッドの実行
  bThread=True;
  #
  #
  # Step:2 グローバル変数MAXをmaxに置き換え
  #for i in range(min,MAX):
  for i in range(min,max):
      #
      # Step:1 スレッドクラスで実行するので以下の処理は不要
      # COUNT2=COUNT4=COUNT8=0;
      # mask=(1<<i)-1;
      # for j in range(i):
      #   aBoard[j]=j;              # 盤を初期化
      #
      # Step:1 呼び出し方法の変更
      # NQueen(i,mask);
      #
      # Step:8 排他制御
      lock = threading.Lock();
      #
      # Step:8
      # ボードクラスにlockを渡します
      info=Board(lock);       # ボードクラス
      #
      # Step:4 
      start_time = datetime.now() 
      #
      # Step:4 呼び出しを変更
      #child=WorkingEngine(i,info);  # スレッドクラス
      #
      # Step:5 コンストラクタの構築
      # child=WorkingEngine(i,i,info,i-1,0,bThread);  # スレッドクラス
      child=threading.Thread(target=WorkingEngine,args=(i,i,info,i-1,0,bThread),name='main().thread')
      child.start();
      child.join();
      #
      # Step:4 WorkingEngineのコンストラクタで呼び出すので、
      # 以下の呼び出しを除去
      # child.NQueen();
      # 
      time_elapsed=datetime.now()-start_time 
      _text='{}'.format(time_elapsed)
      text=_text[:-3]
      #
      # Step:1 getTotal() getUnique()の呼び出しを info. へ変更
      #print("%2d:%13d%13d%20s" % (i,getTotal(),getUnique(),text)); # 出力
      print("%2d:%13d%13d%20s" % (i,info.getTotal(),info.getUnique(),text)); # 出力
#
main();
#
