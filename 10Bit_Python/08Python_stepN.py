#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import copy
from datetime import datetime
"""
ステップＮ版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

# 実行 
$ python <filename.py>

# 実行結果
bash-3.2$ python 08Python_stepN.py
キャリーチェーン
 N:        Total       Unique        hh:mm:ss.ms
 5:           10            2         0:00:00.001
 6:            4            1         0:00:00.004
 7:           40            6         0:00:00.023
 8:           92           12         0:00:00.112
 9:          352           46         0:00:00.514
10:          724           92         0:00:02.007
11:         2680          341         0:00:06.930
12:        14200         1788         0:00:21.436
"""
#
# グローバル変数
TOTAL=0
UNIQUE=0
pres_a=[0]*930
pres_b=[0]*930
COUNTER=[0]*3
B=[]
#
# ボード外側２列を除く内側のクイーン配置処理
def solve(row,left,down,right):
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
    total+=solve(row,(left|bit)<<1,down|bit,(right|bit)>>1)
    bitmap^=bit
  return total
#
# キャリーチェーン　solve()を呼び出して再起を開始する
def process(size,sym):
  global B
  global COUNTER
  # sym 0:COUNT2 1:COUNT4 2:COUNT8
  COUNTER[sym]+=solve(
        B[0]>>2,
        B[1]>>4,
        ((((B[2]>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
        (B[3]>>4)<<(size-5)
  )
#
# キャリーチェーン　対象解除
def carryChainSymmetry(size,n,w,s,e):
  global B
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
  if not B[4][0]:
    process(size,2) # COUNT8
    return
  # n,e,s==w の場合は最小値を確認する。
  # : '右回転で同じ場合は、
  # w=n=e=sでなければ値が小さいのでskip
  # w=n=e=sであれば90度回転で同じ可能性 ';
  if s==w:
    if n!=w or e!=w: return
    process(size,0) # COUNT2
    return
  # : 'e==wは180度回転して同じ
  # 180度回転して同じ時n>=sの時はsmaller?  ';
  if e==w and n>=s:
    if n>s: return
    process(size,1) # COUNT4
    return
  process(size,2)   # COUNT8
  return
#
# キャリーチェーン 効きのチェック dimxは行 dimyは列
def placement(size,dimx,dimy):
  global B
  if B[4][dimx]==dimy:
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
  if B[4][0]:
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
    if B[4][0]!=-1:
      if (dimx<B[4][0] or dimx>=size-B[4][0]) and (dimy==0 or dimy==size-1):
        return 0
      if (dimx==size-1) and (dimy<=B[4][0] or dimy>=size-B[4][0]):
        return 0
  else:
    if B[4][1]!=-1:
      # bitmap=$(( bitmap|2 )); # 枝刈り
      # bitmap=$(( bitmap^2 )); # 枝刈り
      #((bitmap&=~2)); # 上２行を一行にまとめるとこうなります
      # ちなみに上と下は同じ趣旨
      # if (( (t_x[1]>=dimx)&&(dimy==1) ));then
      #   return 0;
      # fi
      if B[4][1]>=dimx and dimy==1:
        return 0
  if (B[0] & 1<<dimx) or (B[1] & 1<<(size-1-dimx+dimy)) or (B[2] & 1<<dimy) or (B[3] & 1<<(dimx+dimy)): 
    return 0
  B[0]|=1<<dimx
  B[1]|=1<<(size-1-dimx+dimy)
  B[2]|=1<<dimy
  B[3]|=1<<(dimx+dimy)
  B[4][dimx]=dimy
  return 1
#
# チェーンのビルド
def buildChain(size):
  global B
  global pres_a
  global pres_b
  B=[0,0,0,0,[-1]*size] # Bの初期化
  wB=sB=eB=nB=[]
  wB=copy.deepcopy(B)
  for w in range( (size//2)*(size-3) +1):
    B=copy.deepcopy(wB)
    B=[0,0,0,0,[-1]*size] # Bの初期化
    # １．０行目と１行目にクイーンを配置
    if placement(size,0,pres_a[w])==0:
      continue
    if placement(size,1,pres_b[w])==0:
      continue
    # ２．９０度回転
    nB=copy.deepcopy(B)
    mirror=(size-2)*(size-1)-w
    for n in range(w,mirror,1):
      B=copy.deepcopy(nB)
      if placement(size,pres_a[n],size-1)==0:
        continue
      if placement(size,pres_b[n],size-2)==0:
        continue
      # ３．９０度回転
      eB=copy.deepcopy(B)
      for e in range(w,mirror,1):
        B=copy.deepcopy(eB)
        if placement(size,size-1,size-1-pres_a[e])==0:
          continue
        if placement(size,size-2,size-1-pres_b[e])==0:
          continue
        # ４．９０度回転
        sB=copy.deepcopy(B)
        for s in range(w,mirror,1):
          B=copy.deepcopy(sB)
          if placement(size,size-1-pres_a[s],0)==0:
            continue
          if placement(size,size-1-pres_b[s],1)==0:
            continue
          # 対象解除法
          carryChainSymmetry(size,n,w,s,e)
#
# チェーンの初期化
def initChain(size):
  global pres_a
  global pres_b
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
def carryChain(size):
  global B
  global TOTAL
  global UNIQUE
  global COUNTER
  TOTAL=UNIQUE=0
  COUNTER[0]=COUNTER[1]=COUNTER[2]=0
  # Bの初期化  [0, 0, 0, 0, [0, 0, 0, 0, 0]]
  B=[0]*5             # row/left/down/right/X
  B[4]=[0]*size       # X を0でsize分を初期化
  initChain(size)     # チェーンの初期化
  buildChain(size)    # チェーンのビルド
  # 集計
  UNIQUE=COUNTER[0]+COUNTER[1]+COUNTER[2]
  TOTAL=COUNTER[0]*2 + COUNTER[1]*4 + COUNTER[2]*8
#
# メイン
def main():
  global TOTAL
  global UNIQUE
  nmin = 5
  nmax = 24
  print("キャリーチェーン")
  print(" N:        Total       Unique        hh:mm:ss.ms")
  for i in range(nmin, nmax,1):
    start_time = datetime.now()
    carryChain(i)
    time_elapsed = datetime.now() - start_time
    _text = '{}'.format(time_elapsed)
    text = _text[:-3]
    print("%2d:%13d%13d%20s" % (i, TOTAL, UNIQUE,text))  # 出力
#
main()
#
#
# 実行
# size=5
# carryChain(size)    # ７．キャリーチェーン
# print("size:",size,"TOTAL:",TOTAL,"UNIQUE:",UNIQUE)
#
