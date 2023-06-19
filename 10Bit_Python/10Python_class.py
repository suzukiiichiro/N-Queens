#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import numpy as np
import copy
from datetime import datetime
"""
クラス対応版 Ｎクイーン

class Boardを作成
`pres_a` `pres_b`を`initchain()`のローカル変数へ変更し、`buildChain()`に引数で渡すように変更


詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

# 実行 
$ python <filename.py>

# 実行結果
bash-3.2$ python 09Python_numpy.py
キャリーチェーン
 N:        Total       Unique        hh:mm:ss.ms
 5:           10            2         0:00:00.001
 6:            4            1         0:00:00.003
 7:           40            6         0:00:00.016
 8:           92           12         0:00:00.071
 9:          352           46         0:00:00.300
10:          724           92         0:00:01.223
11:         2680          341         0:00:04.093
12:        14200         1788         0:00:13.462
"""
#
# グローバル変数
TOTAL=0
UNIQUE=0
# pres_a=np.array([0]*930)
# pres_b=np.array([0]*930)
# COUNTER=np.array([0]*3)  
# B=np.array([])          
#
#
class Board:
  def __init__(self,size):
    self.row=0
    self.left=0
    self.down=0
    self.right=0
    self.X=[-1 for i in range(size)]
    self.COUNTER=[0,0,0]
  def getUnique(self):
    return self.COUNTER[0]+self.COUNTER[1]+self.COUNTER[2]
  def getTotal(self):
    return self.COUNTER[0]*2 + self.COUNTER[1]*4 + self.COUNTER[2]*8
#
#
class nQueens(): # pylint:disable=RO902
  #
  # 初期化
  def __init__(self,size,B): # pylint:disable=R0913
    super(nQueens,self).__init__
    self.size=size
    self.B=B
    # self.B=Board(self.size)
  #
  # ボード外側２列を除く内側のクイーン配置処理
  def solve(self,row,left,down,right):
    total=0
    if not down+1:
      return 1
    while row&1:
      row>>=1
      left<<=1
      right>>=1
    row>>=1           # １行下に移動する
    bitmap=~(left|down|right)
    while bitmap!=0:
      bit=-bitmap&bitmap
      total+=self.solve(row,(left|bit)<<1,down|bit,(right|bit)>>1)
      bitmap^=bit
    return total
  #
  # キャリーチェーン　solve()を呼び出して再起を開始する
  def process(self,size,sym):
    # global B
    # global COUNTER
    # ROW=0
    # LEFT=1
    # DOWN=2
    # RIGHT=3
    # self.B.COUNTER[sym]+=self.solve(
    #       self.B.row>>2,
    #       self.B.left>>4,
    #       ((((self.B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
    #       (self.B.right>>4)<<(size-5)
    #       )
    return self.solve(
          self.B.row>>2,
          self.B.left>>4,
          ((((self.B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
          (self.B.right>>4)<<(size-5)
          )
  #
  # キャリーチェーン　対象解除
  def carryChainSymmetry(self,size,n,w,s,e):
    # global B
    # COUNT2=0
    # COUNT4=1
    # COUNT8=2
    # n,e,s=(N-2)*(N-1)-1-w の場合は最小値を確認する。
    ww=(size-2)*(size-1)-1-w
    w2=(size-2)*(size-1)-1
    # 対角線上の反転が小さいかどうか確認する
    if s==ww and n<(w2-e): return 
    # 垂直方向の中心に対する反転が小さいかを確認
    if e==ww and n>(w2-n): return
    # 斜め下方向への反転が小さいかをチェックする
    if n==ww and e>(w2-s): return
    # 【枝刈り】１行目が角の場合
    # １．回転対称チェックせずにCOUNT8にする
    if not self.B.X[0]:
      #self.process(size,2) # COUNT8
      self.B.COUNTER[2]+=self.process(size,2) # COUNT8
      return
    # n,e,s==w の場合は最小値を確認する。
    # : '右回転で同じ場合は、
    # w=n=e=sでなければ値が小さいのでskip
    # w=n=e=sであれば90度回転で同じ可能性 ';
    if s==w:
      if n!=w or e!=w: return
      #self.process(size,0) # COUNT2
      self.B.COUNTER[0]+=self.process(size,0) # COUNT2
      return
    # : 'e==wは180度回転して同じ
    # 180度回転して同じ時n>=sの時はsmaller?  ';
    if e==w and n>=s:
      if n>s: return
      #self.process(size,1) # COUNT4
      self.B.COUNTER[1]+=self.process(size,1) # COUNT4
      return
    #self.process(size,2)   # COUNT8
    self.B.COUNTER[2]+=self.process(size,2)   # COUNT8
    # print(self.B.COUNTER[2])
    return
  #
  # キャリーチェーン 効きのチェック dimxは行 dimyは列
  def placement(self,size,dimx,dimy):
    # global B
    # ROW=0
    # LEFT=1
    # DOWN=2
    # RIGHT=3
    if self.B.X[dimx]==dimy:
      return 1
    #
    #
    # 【枝刈り】Qが角にある場合の枝刈り
    #  ２．２列めにクイーンは置かない
    #  （１はcarryChainSymmetry()内にあります）
    #
    #  Qが角にある場合は、
    #  2行目のクイーンの位置 t_x[1]が BOUND1
    #  BOUND1行目までは2列目にクイーンを置けない
    # 
    #    +-+-+-+-+-+  
    #    | | | |X|Q| 
    #    +-+-+-+-+-+  
    #    | |Q| |X| | 
    #    +-+-+-+-+-+  
    #    | | | |X| |       
    #    +-+-+-+-+-+             
    #    | | | |Q| | 
    #    +-+-+-+-+-+ 
    #    | | | | | |      
    #    +-+-+-+-+-+  
    if self.B.X[0]:
      #
      # 【枝刈り】Qが角にない場合
      #
      #  +-+-+-+-+-+  
      #  |X|X|Q|X|X| 
      #  +-+-+-+-+-+  
      #  |X| | | |X| 
      #  +-+-+-+-+-+  
      #  | | | | | |
      #  +-+-+-+-+-+
      #  |X| | | |X|
      #  +-+-+-+-+-+
      #  |X|X| |X|X|
      #  +-+-+-+-+-+
      #
      #   １．上部サイド枝刈り
      #  if ((row<BOUND1));then        
      #    bitmap=$(( bitmap|SIDEMASK ));
      #    bitmap=$(( bitmap^=SIDEMASK ));
      #
      #  | | | | | |       
      #  +-+-+-+-+-+  
      #  BOUND1はt_x[0]
      #
      #  ２．下部サイド枝刈り
      #  if ((row==BOUND2));then     
      #    if (( !(down&SIDEMASK) ));then
      #      return ;
      #    fi
      #    if (( (down&SIDEMASK)!=SIDEMASK ));then
      #      bitmap=$(( bitmap&SIDEMASK ));
      #    fi
      #  fi
      #
      #  ２．最下段枝刈り
      #  LSATMASKの意味は最終行でBOUND1以下または
      #  BOUND2以上にクイーンは置けないということ
      #  BOUND2はsize-t_x[0]
      #  if(row==sizeE){
      #    //if(!bitmap){
      #    if(bitmap){
      #      if((bitmap&LASTMASK)==0){
      if self.B.X[0]!=-1:
        if(
          (dimx<self.B.X[0] or dimx>=size-self.B.X[0]) and 
          (dimy==0 or dimy==size-1) 
        ):
          return 0
        if(
          (dimx==size-1) and 
          (dimy<=self.B.X[0] or dimy>=size-self.B.X[0])
        ):
          return 0
    else:
      if self.B.X[1]!=-1:
        # bitmap=$(( bitmap|2 )); # 枝刈り
        # bitmap=$(( bitmap^2 )); # 枝刈り
        #((bitmap&=~2)); # 上２行を一行にまとめるとこうなります
        # ちなみに上と下は同じ趣旨
        # if (( (t_x[1]>=dimx)&&(dimy==1) ));then
        #   return 0;
        # fi
        if self.B.X[1]>=dimx and dimy==1:
          return 0
    if( (self.B.row & 1<<dimx) or 
        (self.B.left & 1<<(size-1-dimx+dimy)) or
        (self.B.down & 1<<dimy) or
        (self.B.right & 1<<(dimx+dimy)) 
    ): 
      return 0
    self.B.row|=1<<dimx
    self.B.left|=1<<(size-1-dimx+dimy)
    self.B.down|=1<<dimy
    self.B.right|=1<<(dimx+dimy)
    self.B.X[dimx]=dimy
    return 1
  #
  # チェーンのビルド
  def buildChain(self,size,pres_a,pres_b):
    # global B
    # global pres_a
    # global pres_b

    # B=np.array([0,0,0,0,[-1]*size]) # Bの初期化
    #wB=sB=eB=nB=np.array([[0 for i in range(size)] for j in range(size)])
    wB=sB=eB=nB=copy.deepcopy(self.B)
    # wB=np.array(copy.deepcopy(self.B))
    for w in range( (size//2)*(size-3) +1):
      #self.B=np.array(wB.copy())
      self.B=copy.deepcopy(wB)
      #B=np.array([0,0,0,0,[-1]*size]) # Bの初期化
      # B=[[0 for i in range(size)] for j in range(size)]
      # B[4]=[-1]*size       # X を -1 でsize分を初期化
      # １．０行目と１行目にクイーンを配置
      if self.placement(size,0,pres_a[w])==0:
        continue
      if self.placement(size,1,pres_b[w])==0:
        continue
      # ２．９０度回転
      nB=copy.deepcopy(self.B)
      mirror=(size-2)*(size-1)-w
      for n in range(w,mirror,1):
        self.B=copy.deepcopy(nB,memo=None, _nil=[])
        if self.placement(size,pres_a[n],size-1)==0:
          continue
        if self.placement(size,pres_b[n],size-2)==0:
          continue
        # ３．９０度回転
        eB=copy.deepcopy(self.B)
        for e in range(w,mirror,1):
          self.B=copy.deepcopy(eB,memo=None, _nil=[])
          if self.placement(size,size-1,size-1-pres_a[e])==0:
            continue
          if self.placement(size,size-2,size-1-pres_b[e])==0:
            continue
          # ４．９０度回転
          # sB=np.array(copy.deepcopy(self.B))
          for s in range(w,mirror,1):
            self.B=copy.deepcopy(sB,memo=None, _nil=[])
            if self.placement(size,size-1-pres_a[s],0)==0:
              continue
            if self.placement(size,size-1-pres_b[s],1)==0:
              continue
            # 対象解除法
            self.carryChainSymmetry(size,n,w,s,e)
  #
  # チェーンの初期化
  def initChain(self,size,pres_a,pres_b):
    # global pres_a
    # global pres_b
    idx=0
    for a in range(size):
      for b in range(size):
        if (a>=b and (a-b)<=1) or (b>a and (b-a<=1)):
          continue
        pres_a[idx]=a
        pres_b[idx]=b
        idx+=1
  #
  # キャリーチェーン
  def carryChain(self,size):
    # global B
    # global TOTAL
    # global UNIQUE
    # global COUNTER
    # TOTAL=UNIQUE=0
    # COUNTER[0]=COUNTER[1]=COUNTER[2]=np.array(0)
    # Bの初期化  [0, 0, 0, 0, [0, 0, 0, 0, 0]]
    # ↓
    """
    # [
    [0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0], 
    [-1,-1,-1,-1,-1]]
    """
    # B=np.array([[0 for i in range(size)] for j in range(5)])
    # B=[0]*5             # row/left/down/right/X
    #B[4]=array([-1]*size)       # X を -1 でsize分を初期化
    # B[4]=np.array([-1 for i in range(size)])       # X を -1 でsize分を初期化

    pres_a=[0]*930
    pres_b=[0]*930
    self.initChain(size,pres_a,pres_b)     # チェーンの初期化
    self.buildChain(size,pres_a,pres_b)    # チェーンのビルド
    # 集計
    # UNIQUE=self.B.COUNTER[0]+self.B.COUNTER[1]+self.B.COUNTER[2]
    # TOTAL=self.B.COUNTER[0]*2 + self.B.COUNTER[1]*4 + self.B.COUNTER[2]*8
#
# メイン
def main():
  # global TOTAL
  # global UNIQUE
  nmin = 5
  nmax = 24
  print("キャリーチェーン")
  print(" N:        Total       Unique        hh:mm:ss.ms")
  for size in range(nmin, nmax,1):
    start_time = datetime.now()
    # carryChain(i)
    B=Board(size)
    nq=nQueens(size,B)
    nq.carryChain(size)
    time_elapsed = datetime.now() - start_time
    _text = '{}'.format(time_elapsed)
    text = _text[:-3]
    #print("%2d:%13d%13d%20s" % (i, TOTAL, UNIQUE,text))  # 出力
    print("%2d:%13d%13d%20s" % (size, B.getTotal(),B.getUnique(),text))  # 出力
#
main()
#
#
# 実行
# size=5
# carryChain(size)    # ７．キャリーチェーン
# print("size:",size,"TOTAL:",TOTAL,"UNIQUE:",UNIQUE)
#
