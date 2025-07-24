#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
コンステレーション マルチプロセス版Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

fedora$ python 18Py_constellations_ProcessPool.py ^C
fedora$ pypy 18Py_constellations_ProcessPool_pypy.py
 N:        Total       Unique        hh:mm:ss.ms
 5:           18            0         0:00:00.034
 6:            4            0         0:00:00.070
 7:           40            0         0:00:00.122
 8:           92            0         0:00:00.169
 9:          352            0         0:00:00.156
10:          724            0         0:00:00.178
11:         2680            0         0:00:00.200
12:        14200            0         0:00:00.389
13:        73712            0         0:00:00.825
14:       365596            0         0:00:01.801
15:      2279184            0         0:00:03.841
16:     14772512            0         0:00:13.367
"""

# のこったプロセスをkillallするために必要
import subprocess

from operator import or_
from typing import List,Set,Dict
from datetime import datetime

# pypyを使うときは以下を活かしてcodon部分をコメントアウト
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')
#
# pypy では ThreadPool/ProcessPoolが動きます
from threading import Thread
from multiprocessing import Pool as ThreadPool
import concurrent
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
#
class NQueens18:
  def __init__(self):
    pass
  def SQd0B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    if row==endmark:
      tempcounter[0]+=1
      return
    while free:
      bit:int=free&-free  # 最下位ビットを取得
      # bit:int=-free&free  # 最下位ビットを取得
      free&=free-1  # 使用済みビットを削除
      # next_ld=(ld|bit)<<1
      # next_rd=(rd|bit)>>1
      # next_col=col|bit
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      # next_free:int=~(next_ld|next_rd|next_col)&((1<<N)-1)  # マスクを適用<<注意
      next_free:int=~(next_ld|next_rd|next_col) # オーバーフロー防止  # マスクを適用<<注意
      # if next_free:
      #   if row<endmark-1:
      #     if ~((next_ld<<1)|(next_rd>>1)|next_col)>0:
      #       self.SQd0B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
      #   else:
      #     self.SQd0B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
      if next_free and (row>=endmark-1 or ~((next_ld<<1)|(next_rd>>1)|next_col)>0):
        self.SQd0B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQd0BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    N3:int=N-3
    # if row==mark1:
    #   while free:
    #     bit:int=free&-free
    #     # bit:int=-free&free
    #     free&=free-1
    #     next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3)) #<<注意
    #     if next_free:
    #       self.SQd0B((ld|bit)<<2,((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return
    while row==mark1 and free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3)) #<<注意
      if next_free:
        self.SQd0B((ld|bit)<<2,((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQd0BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQd1BklB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    N4:int=N-4
    # if row==mark1:
    #   while free:
    #     bit:int=free&-free
    #     # bit:int=-free&free
    #     free&=free-1
    #     next_free:int=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|1|(1<<N4))
    #     if next_free:
    #       self.SQd1B(((ld|bit)<<3)|1,((rd|bit)>>3)|(1<<N4),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return
    while row==mark1 and free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|1|(1<<N4))
      if next_free:
        self.SQd1B(((ld|bit)<<3)|1,((rd|bit)>>3)|(1<<N4),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQd1BklB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQd1B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    if row==endmark:
      tempcounter[0]+=1
      return
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      # next_ld=(ld|bit)<<1
      # next_rd=(rd|bit)>>1
      # next_col=col|bit
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=~(next_ld|next_rd|next_col)&((1<<N)-1)
      # if next_free:
      #   if row+1<endmark:
      #     if ~((next_ld<<1)|(next_rd>>1)|next_col)>0:
      #       self.SQd1B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
      #   else:
      #     self.SQd1B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
      if next_free and (row+1>=endmark or ~((next_ld<<1)|(next_rd>>1)|next_col)>0):
        self.SQd1B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQd1BkBlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    N3:int=N-3
    # When row reaches mark1
    # if row==mark1:
    #   while free:
    #     bit:int=free&-free  # Extract the rightmost 1-bit
    #     # bit:int=-free&free  # Extract the rightmost 1-bit
    #     free&=free-1  # Remove the processed bit
    #     next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
    #     if next_free:
    #       # Recursive call with updated values
    #       self.SQd1BlB(((ld|bit)<<2),((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return
    while row==mark1 and free:
      bit:int=free&-free  # Extract the rightmost 1-bit
      # bit:int=-free&free  # Extract the rightmost 1-bit
      free&=free-1  # Remove the processed bit
      next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
      if next_free:
        # Recursive call with updated values
        self.SQd1BlB(((ld|bit)<<2),((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    # General case when row != mark1
    while free:
      bit:int=free&-free  # Extract the rightmost 1-bit
      # bit:int=-free&free  # Extract the rightmost 1-bit
      free&=free-1  # Remove the processed bit
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        # Recursive call with updated values
        self.SQd1BkBlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQd1BlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    # When row reaches mark2
    # if row==mark2:
    #   while free:
    #     # Extract the rightmost available position
    #     bit:int=free&-free
    #     # bit:int=-free&free
    #     free&=free-1
    #     # Update diagonal and column occupancies
    #     # next_ld=((ld|bit)<<2)|1
    #     # next_rd=(rd|bit)>>2
    #     # next_col=col|bit
    #     next_ld,next_rd,next_col=((ld|bit)<<2)|1,(rd|bit)>>2,col|bit
    #     next_free:int=~(next_ld|next_rd|next_col)&((1<<N)-1)
    #     # Recursive call if there are available positions
    #     # if next_free:
    #     #   if row+2<endmark:
    #     #     if ~((next_ld<<1)|(next_rd>>1)|next_col)>0:
    #     #       self.SQd1B(next_ld,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    #     #   else:
    #     #     self.SQd1B(next_ld,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    #     if next_free and (row+2>=endmark or ~((next_ld<<1)|(next_rd>>1)|next_col)>0):
    #       self.SQd1B(next_ld,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return
    while row==mark2 and free:
      # Extract the rightmost available position
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      # Update diagonal and column occupancies
      # next_ld=((ld|bit)<<2)|1
      # next_rd=(rd|bit)>>2
      # next_col=col|bit
      next_ld,next_rd,next_col=((ld|bit)<<2)|1,(rd|bit)>>2,col|bit
      next_free:int=~(next_ld|next_rd|next_col)&((1<<N)-1)
      # Recursive call if there are available positions
      # if next_free:
      #   if row+2<endmark:
      #     if ~((next_ld<<1)|(next_rd>>1)|next_col)>0:
      #       self.SQd1B(next_ld,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
      #   else:
      #     self.SQd1B(next_ld,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
      if next_free and (row+2>=endmark or ~((next_ld<<1)|(next_rd>>1)|next_col)>0):
        self.SQd1B(next_ld,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    # General case when row != mark2
    while free:
      # Extract the rightmost available position
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      # Update diagonal and column occupancies
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      # Recursive call if there are available positions
      if next_free:
        self.SQd1BlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQd1BlkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    N3:int=N-3  # Precomputed value for performance
    # Special case when row==mark1
    # if row==mark1:
    #   while free:
    #     bit:int=free&-free  # Extract the rightmost available position
    #     # bit:int=-free&free  # Extract the rightmost available position
    #     free&=free-1
    #     nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|2|(1<<N3))
    #     if nextfree:
    #       self.SQd1B(((ld|bit)<<3)|2,((rd|bit)>>3)|(1<<N3),col|bit,row+3,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return
    while row==mark1 and free:
      bit:int=free&-free  # Extract the rightmost available position
      # bit:int=-free&free  # Extract the rightmost available position
      free&=free-1
      nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|2|(1<<N3))
      if nextfree:
        self.SQd1B(((ld|bit)<<3)|2,((rd|bit)>>3)|(1<<N3),col|bit,row+3,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
    # General case
    while free:
      bit:int=free&-free  # Extract the rightmost available position
      # bit:int=-free&free  # Extract the rightmost available position
      free&=free-1
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if nextfree:
        self.SQd1BlkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQd1BlBkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    # Special case when row==mark1
    # if row==mark1:
    #   while free:
    #     bit:int=free&-free  # Extract the rightmost available position
    #     # bit:int=-free&free  # Extract the rightmost available position
    #     free&=free-1
    #     nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1)
    #     if nextfree:
    #       self.SQd1BkB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return
    while row==mark1 and free:
      bit:int=free&-free  # Extract the rightmost available position
      # bit:int=-free&free  # Extract the rightmost available position
      free&=free-1
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1)
      if nextfree:
        self.SQd1BkB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
    # General case
    while free:
      bit:int=free&-free  # Extract the rightmost available position
      # bit:int=-free&free  # Extract the rightmost available position
      free&=free-1
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if nextfree:
        self.SQd1BlBkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQd1BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    N3:int=N-3
    # Special case: when row equals mark2
    # if row==mark2:
    #   while free:
    #     bit:int=free&-free  # Extract the rightmost available position
    #     # bit:int=-free&free  # Extract the rightmost available position
    #     free&=free-1
    #     # Calculate the next free positions
    #     nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
    #     if nextfree:
    #       self.SQd1B((ld|bit)<<2,((rd|bit)>>2)|(1<<N3),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return
    while row==mark2 and free:
      bit:int=free&-free  # Extract the rightmost available position
      # bit:int=-free&free  # Extract the rightmost available position
      free&=free-1
      # Calculate the next free positions
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
      if nextfree:
        self.SQd1B((ld|bit)<<2,((rd|bit)>>2)|(1<<N3),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
    # General case
    while free:
      bit:int=free&-free  # Extract the rightmost available position
      # bit:int=-free&free  # Extract the rightmost available position
      free&=free-1
      # Calculate the next free positions
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if nextfree:
        self.SQd1BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQd2BlkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    N3:int=N-3
    # 行が mark1 に達した場合の特別処理
    # if row==mark1:
    #   while free:
    #     bit:int=free&-free  # 最下位ビットを取得
    #     # bit:int=-free&free  # 最下位ビットを取得
    #     free&=free-1  # 使用済みビットを削除
    #     # 次の free の計算
    #     nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|(1<<N3)|2)
    #     # 再帰的に SQd2B を呼び出す
    #     if nextfree:
    #       self.SQd2B((ld|bit)<<3|2,(rd|bit)>>3|(1<<N3),col|bit,row+3,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return
    while row==mark1 and free:
      bit:int=free&-free  # 最下位ビットを取得
      # bit:int=-free&free  # 最下位ビットを取得
      free&=free-1  # 使用済みビットを削除
      # 次の free の計算
      nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|(1<<N3)|2)
      # 再帰的に SQd2B を呼び出す
      if nextfree:
        self.SQd2B((ld|bit)<<3|2,(rd|bit)>>3|(1<<N3),col|bit,row+3,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
    # 一般的な再帰処理
    while free:
      bit:int=free&-free  # 最下位ビットを取得
      # bit:int=-free&free  # 最下位ビットを取得
      free&=free-1  # 使用済みビットを削除
      # 次の free の計算
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      # 再帰的に SQd2BlkB を呼び出す
      if nextfree:
        self.SQd2BlkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQd2BklB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    N4:int=N-4
    # row==mark1 の場合の処理
    # if row==mark1:
    #   while free:
    #     bit:int=free&-free  # 最下位のビットを取得
    #     # bit:int=-free&free  # 最下位のビットを取得
    #     free&=free-1  # 使用済みのビットを削除
    #     next_free:int=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|(1<<N4)|1)
    #     if next_free:
    #       self.SQd2B(((ld|bit)<<3)|1,((rd|bit)>>3)|(1<<N4),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return  # この分岐の処理が終わったらリターン
    while row==mark1 and free:
      bit:int=free&-free  # 最下位のビットを取得
      # bit:int=-free&free  # 最下位のビットを取得
      free&=free-1  # 使用済みのビットを削除
      next_free:int=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|(1<<N4)|1)
      if next_free:
        self.SQd2B(((ld|bit)<<3)|1,((rd|bit)>>3)|(1<<N4),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    # 通常の処理
    while free:
      bit:int=free&-free  # 最下位のビットを取得
      # bit:int=-free&free  # 最下位のビットを取得
      free&=free-1  # 使用済みのビットを削除
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQd2BklB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQd2BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    N3:int=N-3
    # `row==mark2` の場合の処理
    # if row==mark2:
    #   while free:
    #     bit:int=free&-free  # 最下位ビットを取得
    #     # bit:int=-free&free  # 最下位ビットを取得
    #     free&=free-1  # 使用済みビットを削除
    #     next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
    #     if next_free:
    #       self.SQd2B(((ld|bit)<<2),((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return  # `if row==mark2` の処理終了後に関数を終了
    while row==mark2 and free:
      bit:int=free&-free  # 最下位ビットを取得
      # bit:int=-free&free  # 最下位ビットを取得
      free&=free-1  # 使用済みビットを削除
      next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
      if next_free:
        self.SQd2B(((ld|bit)<<2),((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    # 通常の処理
    while free:
      bit:int=free&-free  # 最下位ビットを取得
      # bit:int=-free&free  # 最下位ビットを取得
      free&=free-1  # 使用済みビットを削除
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQd2BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQd2BlBkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    # if row==mark1:
    #   while free:
    #     bit:int=free&-free  # Get the lowest bit
    #     # bit:int=-free&free  # Get the lowest bit
    #     free&=free-1  # Remove the lowest bit
    #     next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1)
    #     if next_free:
    #       self.SQd2BkB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return
    while row==mark1 and free:
      bit:int=free&-free  # Get the lowest bit
      # bit:int=-free&free  # Get the lowest bit
      free&=free-1  # Remove the lowest bit
      next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1)
      if next_free:
        self.SQd2BkB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit:int=free&-free  # Get the lowest bit
      # bit:int=-free&free  # Get the lowest bit
      free&=free-1  # Remove the lowest bit
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQd2BlBkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQd2BlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    # if row==mark2:
    #   while free:
    #     bit:int=free&-free  # Get the lowest bit
    #     # bit:int=-free&free  # Get the lowest bit
    #     free&=free-1  # Remove the lowest bit
    #     next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1)
    #     if next_free:
    #       self.SQd2B(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return
    while row==mark2 and free:
      bit:int=free&-free  # Get the lowest bit
      # bit:int=-free&free  # Get the lowest bit
      free&=free-1  # Remove the lowest bit
      next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1)
      if next_free:
        self.SQd2B(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit:int=free&-free  # Get the lowest bit
      # bit:int=-free&free  # Get the lowest bit
      free&=free-1  # Remove the lowest bit
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQd2BlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQd2BkBlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    N3:int=N-3
    # row==mark1 の場合を先に処理
    # if row==mark1:
    #   while free:
    #     bit:int=free&-free
    #     # bit:int=-free&free
    #     free&=free-1
    #     nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
    #     if nextfree:
    #       self.SQd2BlB((ld|bit)<<2,((rd|bit)>>2)|(1<<N3),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return
    while row==mark1 and free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
      if nextfree:
        self.SQd2BlB((ld|bit)<<2,((rd|bit)>>2)|(1<<N3),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
    # 通常の処理
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if nextfree:
        self.SQd2BkBlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQd2B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    # rowがendmarkの場合の処理
    if row==endmark:
      if (free&(~1))>0:
        tempcounter[0]+=1
      return
    # 通常の処理
    while free:
      bit:int=free&-free  # 最も下位の1ビットを取得
      # bit:int=-free&free  # 最も下位の1ビットを取得
      free&=free-1  # 使用済みビットを削除
      # 次の左対角線、右対角線、列の状態を計算
      # next_ld=(ld|bit)<<1
      # next_rd=(rd|bit)>>1
      # next_col=col|bit
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      # 次の自由な位置を計算
      nextfree=~((next_ld)|(next_rd)|(next_col))
      # if nextfree:
      #   if row<endmark-1:
      #     if ~((next_ld<<1)|(next_rd>>1)|(next_col))>0:
      #       self.SQd2B(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
      #   else:
      #     self.SQd2B(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
      if nextfree and (row>=endmark-1 or ~((next_ld<<1)|(next_rd>>1)|(next_col))>0):
        self.SQd2B(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    # if row==mark2:
    #   while free:
    #     bit:int=free&-free
    #     # bit:int=-free&free
    #     free&=free-1
    #     nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1)
    #     if nextfree:
    #       self.SQBjrB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return
    while row==mark2 and free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1)
      if nextfree:
        self.SQBjrB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if nextfree:
        self.SQBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQBkBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    N3:int=N-3
    # if row==mark1:
    #   while free:
    #     bit:int=free&-free  # Isolate the rightmost 1 bit.
    #     # bit:int=-free&free  # Isolate the rightmost 1 bit.
    #     free&=free-1  # Remove the isolated bit from free.
    #     nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
    #     if nextfree:
    #       self.SQBlBjrB((ld|bit)<<2,((rd|bit)>>2)|(1<<N3),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return
    while row==mark1 and free:
      bit:int=free&-free  # Isolate the rightmost 1 bit.
      # bit:int=-free&free  # Isolate the rightmost 1 bit.
      free&=free-1  # Remove the isolated bit from free.
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
      if nextfree:
        self.SQBlBjrB((ld|bit)<<2,((rd|bit)>>2)|(1<<N3),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit:int=free&-free  # Isolate the rightmost 1 bit.
      # bit:int=-free&free  # Isolate the rightmost 1 bit.
      free&=free-1  # Remove the isolated bit from free.
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if nextfree:
        self.SQBkBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    if row==jmark:
      free&=~1  # Clear the least significant bit (mark position 0 unavailable).
      ld|=1  # Mark left diagonal as occupied for position 0.
      while free:
        bit:int=free&-free  # Get the lowest bit (first free position).
        # bit:int=-free&free  # Get the lowest bit (first free position).
        free&=free-1  # Remove this position from the free positions.
        # Calculate next free positions and diagonal/column states.
        # next_ld=(ld|bit)<<1
        # next_rd=(rd|bit)>>1
        # next_col=col|bit
        next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
        next_free:int=~((next_ld|next_rd|next_col))
        if next_free:
          self.SQB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
      return
    while free:
      bit:int=free&-free  # Get the lowest bit (first free position).
      # bit:int=-free&free  # Get the lowest bit (first free position).
      free&=free-1  # Remove this position from the free positions.
      # Calculate next free positions and diagonal/column states.
      # next_ld=(ld|bit)<<1
      # next_rd=(rd|bit)>>1
      # next_col=col|bit
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=~((next_ld|next_rd|next_col))
      if next_free:
        self.SQBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    if row==endmark:
      tempcounter[0]+=1
      return
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      # next_ld=(ld|bit)<<1
      # next_rd=(rd|bit)>>1
      # next_col=col|bit
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=~(next_ld|next_rd|next_col)&((1<<N)-1)
      # if next_free:
      #   if row<endmark-1:
      #     if ~((next_ld<<1)|(next_rd>>1)|next_col)>0:
      #         self.SQB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
      #   else:
      #     self.SQB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
      if next_free and (row>=endmark-1 or ~((next_ld<<1)|(next_rd>>1)|next_col)>0):
        self.SQB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQBlBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    # if row==mark1:
    #   while free:
    #     bit:int=free&-free
    #     # bit:int=-free&free
    #     free&=free-1
    #     next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1)
    #     if next_free:
    #       self.SQBkBjrB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return
    while row==mark1 and free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1)
      if next_free:
        self.SQBkBjrB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQBlBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    N3:int=N-3
    # if row==mark2:
    #   while free:
    #     bit:int=free&-free
    #     # bit:int=-free&free
    #     free&=free-1
    #     next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
    #     if next_free:
    #       self.SQBjrB(((ld|bit)<<2),((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return
    while row==mark2 and free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
      if next_free:
        self.SQBjrB(((ld|bit)<<2),((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQBklBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    N4:int=N-4
    # if row==mark1:
    #   while free:
    #     bit:int=free&-free
    #     # bit:int=-free&free
    #     free&=free-1
    #     next_free:int=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|(1<<N4)|1)
    #     if next_free:
    #       self.SQBjrB(((ld|bit)<<3)|1,((rd|bit)>>3)|(1<<N4),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return
    while row==mark1 and free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|(1<<N4)|1)
      if next_free:
        self.SQBjrB(((ld|bit)<<3)|1,((rd|bit)>>3)|(1<<N4),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQBklBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQBlkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    N3:int=N-3
    # if row==mark1:
    #   while free:
    #     bit:int=free&-free
    #     # bit:int=-free&free
    #     free&=free-1
    #     next_free:int=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|(1<<N3)|2)
    #     if next_free:
    #       self.SQBjrB(((ld|bit)<<3)|2,((rd|bit)>>3)|(1<<N3),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    #   return
    while row==mark1 and free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|(1<<N3)|2)
      if next_free:
        self.SQBjrB(((ld|bit)<<3)|2,((rd|bit)>>3)|(1<<N3),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQBlkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQBjlBkBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    N1:int=N-1
    if row==N1-jmark:
      rd|=1<<N1
      free&=~(1<<N1)
      # if next_free:
      self.SQBkBlBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2,tempcounter,N)
      return
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQBjlBkBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQBjlBlBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    N1:int=N-1
    if row==N1-jmark:
      rd|=1<<N1
      free&=~(1<<N1)
      # if next_free:
      self.SQBlBkBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2,tempcounter,N)
      return
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQBjlBlBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQBjlBklBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    N1:int=N-1
    if row==N1-jmark:
      rd|=1<<N1
      free&=~(1<<N1)
      # if next_free:
      self.SQBklBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2,tempcounter,N)
      return
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
          self.SQBjlBklBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  def SQBjlBlkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
    N1:int=N-1
    if row==N1-jmark:
      rd|=1<<N1
      free&=~(1<<N1)
      # if next_free:
      self.SQBlkBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2,tempcounter,N)
      return
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQBjlBlkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
  #
  """
  回転対称性をチェックする関数
  Args:
      ijkl_list (set): 回転対称性を保持する集合
      i,j,k,l (int): 配置のインデックス
      N (int): ボードのサイズ
  Returns:
      bool: 回転対称性が見つかった場合はTrue、見つからない場合はFalse
  """
  def check_rotations(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
    rot90=((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i
    rot180=((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k)
    rot270=(l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)
    # 回転対称性をチェック
    # if rot90 in ijkl_list:
    #   return True
    # if rot180 in ijkl_list:
    #   return True
    # if rot270 in ijkl_list:
    #   return True
    # return False
    return any(rot in ijkl_list for rot in (rot90, rot180, rot270))
  #
  def symmetry(self,ijkl:int,N:int)->int:
    """
    if self.geti(ijkl)==N-1-self.getj(ijkl) and self.getk(ijkl)==N-1-self.getl(ijkl):
      if self.symmetry90(ijkl,N):
        return 2
      else:
        return 4
    else:
      return 8
    """
    return 2 if self.symmetry90(ijkl,N) else 4 if self.geti(ijkl)==N-1-self.getj(ijkl) and self.getk(ijkl)==N-1-self.getl(ijkl) else 8
  #
  def symmetry90(self,ijkl:int,N:int)->bool:
    return ((self.geti(ijkl)<<15)+(self.getj(ijkl)<<10)+(self.getk(ijkl)<<5)+self.getl(ijkl))==(((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl))
  #
  """
  i,j,k,l のインデックスを1つの整数に変換する関数
  Args:
      i,j,k,l (int): 各インデックス
  Returns:
      int: i,j,k,l を基にした1つの整数
  """
  def to_ijkl(self,i:int,j:int,k:int,l:int)->int:
    return (i<<15)+(j<<10)+(k<<5)+l
  #
  """
  時計回りに90度回転する。
  Args:
      ijkl (int): 配置のエンコードされた整数
      N (int): ボードサイズ
  Returns:
      int: 90度回転後の配置をエンコードした整数
  """
  def rot90(self,ijkl:int,N:int)->int:
    return ((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl)
  #
  """
  垂直方向のミラーリングを行う。
  Args:
      ijkl (int): 配置のエンコードされた整数
      N (int): ボードサイズ
  Returns:
      int: ミラーリング後の配置をエンコードした整数
  """
  def mirvert(self,ijkl:int,N:int)->int:
    return self.to_ijkl(N-1-self.geti(ijkl),N-1-self.getj(ijkl),self.getl(ijkl),self.getk(ijkl))
  #
  """2つの値のうち最小値を返す"""
  def ffmin(self,a:int,b:int)->int:
    return min(a,b)
  #
  """iを抽出"""
  def geti(self,ijkl:int)->int:
    return (ijkl>>15)&0x1F
  #
  """jを抽出"""
  def getj(self,ijkl:int)->int:
    return (ijkl>>10)&0x1F
  #
  """kを抽出"""
  def getk(self,ijkl:int)->int:
    return (ijkl>>5)&0x1F
  #
  """lを抽出"""
  def getl(self,ijkl:int)->int:
    return ijkl&0x1F
  """
  クイーンの配置を回転・ミラーリングさせて最も左上に近い標準形に変換する
  Args:
      ijkl (int): 配置のエンコードされた整数
      N (int): ボードサイズ
  Returns:
      int: 標準形の配置をエンコードした整数
  """
  def jasmin(self,ijkl:int,N:int)->int:
    # 最初の最小値と引数を設定
    arg=0
    min_val=self.ffmin(self.getj(ijkl),N-1-self.getj(ijkl))
    # i: 最初の行（上端） 90度回転2回
    if self.ffmin(self.geti(ijkl),N-1-self.geti(ijkl))<min_val:
      arg=2
      min_val=self.ffmin(self.geti(ijkl),N-1-self.geti(ijkl))
    # k: 最初の列（左端） 90度回転3回
    if self.ffmin(self.getk(ijkl),N-1-self.getk(ijkl))<min_val:
      arg=3
      min_val=self.ffmin(self.getk(ijkl),N-1-self.getk(ijkl))
    # l: 最後の列（右端） 90度回転1回
    if self.ffmin(self.getl(ijkl),N-1-self.getl(ijkl))<min_val:
      arg=1
      min_val=self.ffmin(self.getl(ijkl),N-1-self.getl(ijkl))
    # 90度回転を arg 回繰り返す

    # codon で動きます
    for _ in range(arg):
      ijkl=self.rot90(ijkl,N)

    # codon で動かない
    # ijkl=reduce(lambda acc,_:self.rot90(acc,N),range(arg),ijkl) 

    # 必要に応じて垂直方向のミラーリングを実行
    if self.getj(ijkl)<N-1-self.getj(ijkl):
      ijkl=self.mirvert(ijkl,N)
    return ijkl
  #
  """
  ld: 左対角線の占領状態
  rd: 右対角線の占領状態
  col: 列の占領状態
  k: 特定の行
  l: 特定の行
  row: 現在の行
  queens: 配置済みのクイーンの数
  LD: 左端の特殊な占領状態
  RD: 右端の特殊な占領状態
  counter: コンステレーションのカウント
  constellations: コンステレーションリスト
  N: ボードサイズ
  preset_queens: 必要なクイーンの数
  """
  def set_pre_queens(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:list,constellations:List[Dict[str,int]],N:int,preset_queens:int)->None:
    mask=(1<<N)-1  # setPreQueensで使用
    # k行とl行はスキップ
    if row==k or row==l:
      self.set_pre_queens(ld<<1,rd>>1,col,k,l,row+1,queens,LD,RD,counter,constellations,N,preset_queens)
      return
    # クイーンの数がpreset_queensに達した場合、現在の状態を保存
    if queens==preset_queens:
      constellation= {"ld": ld,"rd": rd,"col": col,"startijkl": row<<20,"solutions":0}
      # 新しいコンステレーションをリストに追加
      constellations.append(constellation)
      counter[0]+=1
      return
    # 現在の行にクイーンを配置できる位置を計算
    free=~(ld|rd|col|(LD>>(N-1-row))|(RD<<(N-1-row)))&mask
    while free:
      bit:int=free&-free  # 最も下位の1ビットを取得
      # bit:int=-free&free  # 最も下位の1ビットを取得
      free&=free-1  # 使用済みビットを削除
      # クイーンを配置し、次の行に進む
      self.set_pre_queens((ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,counter,constellations,N,preset_queens)
  #
  def exec_solutions(self,value:list)->int:
    N,constellation=value  
  #def exec_solutions(self,constellations:List[Dict[str,int]],N:int)->None:
    # jmark=0  # ここで初期化
    # j=0
    # k=0
    # l=0
    # ijkl=0
    # ld=0
    # rd=0
    # col=0
    # start_ijkl=0
    # start=0
    # free=0
    # LD=0
    # endmark=0
    # mark1=0
    # mark2=0
    jmark=j=k=l=ijkl=ld=rd=col=start_ijkl=start=free=LD=endmark=mark1=mark2=0
    small_mask=(1<<(N-2))-1
    temp_counter=[0]
    #for constellation in constellations:
    # mark1=mark1
    # mark2=mark2
    mark1,mark2=mark1,mark2
    # mark2=mark2
    start_ijkl=constellation["startijkl"]
    start=start_ijkl>>20
    ijkl=start_ijkl&((1<<20)-1)
    # j=self.getj(ijkl)
    # k=self.getk(ijkl)
    # l=self.getl(ijkl)
    j,k,l=self.getj(ijkl),self.getk(ijkl),self.getl(ijkl)
    # 左右対角線と列の占有状況を設定
    # ld=constellation["ld"]>>1
    # rd=constellation["rd"]>>1
    # col=(constellation["col"]>>1)|(~small_mask)
    ld,rd,col=constellation["ld"]>>1,constellation["rd"]>>1,(constellation["col"]>>1)|(~small_mask)
    LD=(1<<(N-1-j))|(1<<(N-1-l))
    ld|=LD>>(N-start)
    # rd=constellation["rd"]>>1
    if start>k:
      rd|=(1<<(N-1-(start-k+1)))
    if j >= 2 * N-33-start:
      rd|=(1<<(N-1-j))<<(N-2-start)
    # col=(constellation["col"]>>1)|(~small_mask)
    free=~(ld|rd|col)
    # 各ケースに応じた処理
    if j<(N-3):
      jmark,endmark=j+1,N-2
      if j>2 * N-34-start:
        if k<l:
          mark1,mark2=k-1,l-1
          if start<l:
            if start<k:
              if l!=k+1:
                self.SQBkBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
              else: self.SQBklBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
            else: self.SQBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
          else: self.SQBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
        else:
          mark1,mark2=l-1,k-1
          if start<k:
            if start<l:
              if k!=l+1:
                self.SQBlBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
              else: self.SQBlkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
            else: self.SQBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
          else: self.SQBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
      else:
        if k<l:
          mark1,mark2=k-1,l-1
          if l!=k+1:
            self.SQBjlBkBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
          else: self.SQBjlBklBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
        else:
          mark1,mark2=l-1,k-1
          if k != l+1:
            self.SQBjlBlBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
          else: self.SQBjlBlkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
    elif j==(N-3):
      endmark=N-2
      if k<l:
        mark1,mark2=k-1,l-1
        if start<l:
          if start<k:
            if l != k+1: self.SQd2BkBlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
            else: self.SQd2BklB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
          else:
            mark2=l-1
            self.SQd2BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
        else: self.SQd2B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
      else:
        mark1,mark2=l-1,k-1
        endmark=N-2
        if start<k:
          if start<l:
            if k != l+1:
              self.SQd2BlBkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
            else: self.SQd2BlkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
          else:
            mark2=k-1
            self.SQd2BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
        else: self.SQd2B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
    elif j==N-2: # クイーンjがコーナーからちょうど1列離れている場合
      if k<l:  # kが最初になることはない、lはクイーンの配置の関係で最後尾にはなれない
        endmark=N-2
        if start<l:  # 少なくともlがまだ来ていない場合
          if start<k:  # もしkもまだ来ていないなら
            mark1=k-1
            if l != k+1:  # kとlが隣り合っている場合
              mark2=l-1
              self.SQd1BkBlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
            else: self.SQd1BklB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
          else:  # lがまだ来ていないなら
            mark2=l-1
            self.SQd1BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
          # すでにkとlが来ている場合
        else: self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
      else:  # l<k
        if start<k:  # 少なくともkがまだ来ていない場合
          if start<l:  # lがまだ来ていない場合
            if k<N-2:  # kが末尾にない場合
              mark1,endmark=l-1,N-2
              if k != l+1:  # lとkの間に空行がある場合
                mark2=k-1
                self.SQd1BlBkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
                # lとkの間に空行がない場合
              else: self.SQd1BlkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
            else:  # kが末尾の場合
              if l != (N-3):  # lがkの直前でない場合
                mark2,endmark=l-1,N-3
                self.SQd1BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
              else:  # lがkの直前にある場合
                endmark=N-4
                self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
          else:  # もしkがまだ来ていないなら
            if k != N-2:  # kが末尾にない場合
              mark2,endmark=k-1,N-2
              self.SQd1BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
            else:  # kが末尾の場合
              endmark=N-3
              self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
        else: # kとlはスタートの前
          endmark=N-2
          self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
    else:  # クイーンjがコーナーに置かれている場合
      endmark=N-2
      if start>k:
        self.SQd0B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
      else: # クイーンをコーナーに置いて星座を組み立てる方法と、ジャスミンを適用する方法
        mark1=k-1
        self.SQd0BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
      # 各コンステレーションのソリューション数を更新
    #constellation["solutions"]=temp_counter[0] * self.symmetry(ijkl,N)
    return temp_counter[0] * self.symmetry(ijkl,N)
    print(constellation)
    temp_counter[0]=0
  #
  def gen_constellations(self,ijkl_list:Set[int],constellations:List[Dict[str,int]],N:int,preset_queens:int)->None:
    halfN=(N+1)//2  # Nの半分を切り上げ
    # コーナーにクイーンがいない場合の開始コンステレーションを計算する
    """
    for k in range(1,halfN):
      for l in range(k+1,N-1):
        for i in range(k+1,N-1):
          if i==(N-1)-l:
            continue
          for j in range(N-k-2,0,-1):
            if j==i or l==j:
              continue
            if not self.check_rotations(ijkl_list,i,j,k,l,N):
              ijkl_list.add(self.to_ijkl(i,j,k,l))
    """
    ijkl_list.update(self.to_ijkl(i,j,k,l) for k in range(1,halfN) for l in range(k+1,N-1) for i in range(k+1,N-1) if i != (N-1)-l for j in range(N-k-2,0,-1) if j!=i and j!=l if not self.check_rotations(ijkl_list,i,j,k,l,N)
    )
    # コーナーにクイーンがある場合の開始コンステレーションを計算する
    # for j in range(1,N-2):
    #   for l in range(j+1,N-1):
    #     ijkl_list.add(self.to_ijkl(0,j,0,l))
    # [ijkl_list.add(self.to_ijkl(0,j,0,l)) for j in range(1,N-2) for l in range(j+1,N-1)]
    ijkl_list.update({self.to_ijkl(0,j,0,l) for j in range(1,N-2) for l in range(j+1,N-1)})


    # Jasmin変換
    ijkl_list_jasmin=set()
    # for start_constellation in ijkl_list:
    #   ijkl_list_jasmin.add(self.jasmin(start_constellation,N))
    # [ijkl_list_jasmin.add(self.jasmin(start_constellation,N)) for start_constellation in ijkl_list]
    ijkl_list_jasmin.update(self.jasmin(start_constellation, N) for start_constellation in ijkl_list)
    # ijkl_list_jasmin.update(map(lambda sc: self.jasmin(sc, N), ijkl_list))


    ijkl_list=ijkl_list_jasmin
    L=1<<(N-1)  # Lは左端に1を立てる
    for sc in ijkl_list:
      # i=self.geti(sc)
      # j=self.getj(sc)
      # k=self.getk(sc)
      # l=self.getl(sc)
      i,j,k,l=self.geti(sc),self.getj(sc),self.getk(sc),self.getl(sc)
      # ld=(L>>(i-1))|(1<<(N-k)) # 左対角線
      # rd=(L>>(i+1))|(1<<(l-1)) # 右対角線
      # col=1|L|(L>>i)|(L>>j) # 列
      ld,rd,col=(L>>(i-1))|(1<<(N-k)),(L>>(i+1))|(1<<(l-1)),1|L|(L>>i)|(L>>j) 
      # LD=(L>>j)|(L>>l) # 左端の対角線
      # RD=(L>>j)|(1<<k) # 右端の対角線
      LD,RD=(L>>j)|(L>>l),(L>>j)|(1<<k)
      counter=[0] # サブコンステレーションを生成
      self.set_pre_queens(ld,rd,col,k,l,1,3 if j==N-1 else 4,LD,RD,counter,constellations,N,preset_queens)
      current_size=len(constellations)
      # 生成されたサブコンステレーションにスタート情報を追加
      # for a in range(counter[0]):
      #   constellations[current_size-a-1]["startijkl"]|=self.to_ijkl(i,j,k,l)
      list(map(lambda target:target.__setitem__("startijkl",target["startijkl"]|self.to_ijkl(i,j,k,l)),(constellations[current_size-a-1] for a in range(counter[0]))))
class NQueens18_constellations_ProcessPool():
  def finalize(self)->None:
    cmd="killall pypy"  # python or pypy
    p = subprocess.Popen("exec " + cmd, shell=True)
    p.kill()
  def main(self)->None:
    nmin:int=5
    nmax:int=19
    preset_queens:int=4  # 必要に応じて変更
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      ijkl_list:Set[int]=set()
      constellations:List[Dict[str,int]]=[]
      NQ=NQueens18()
      NQ.gen_constellations(ijkl_list,constellations,size,preset_queens)
      pool=ThreadPool(size)
      params=[(size,constellations[i])for i in range(len(constellations))]
      pool.map(NQ.exec_solutions,params)
      total:int=sum(list(pool.map(NQ.exec_solutions,params)))
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
      self.finalize()
if __name__=="__main__":
  NQueens18_constellations_ProcessPool().main()

