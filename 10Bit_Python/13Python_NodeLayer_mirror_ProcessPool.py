#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
ノードレイヤー ミラー マルチプロセス版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens
"""

"""
CentOS-5.1$ pypy 13Python_NodeLayer_mirror_ProcessPool.py
 N:        Total       Unique        hh:mm:ss.ms
 4:            2            0         0:00:00.015
 5:           10            0         0:00:00.022
 6:            4            0         0:00:00.046
 7:           40            0         0:00:00.052
 8:           92            0         0:00:00.100
 9:          352            0         0:00:00.088
10:          724            0         0:00:00.105
11:         2680            0         0:00:00.151
12:        14200            0         0:00:00.141
13:        73712            0         0:00:00.225
14:       365596            0         0:00:00.633
15:      2279184            0         0:00:03.135
16:     14772512            0         0:00:20.478
17:     95815104            0         0:02:31.781
18:    666090624            0         0:18:07.638

CentOS-5.1$ pypy 10Python_bit_symmetry_ProcessPool.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184       285053         0:00:03.215
16:     14772512      1846955         0:00:16.017
17:     95815104     11977939         0:01:39.372
18:    666090624     83263591         0:11:29.141
"""


"""
CentOS-5.1$ pypy 13Python_NodeLayer_mirror_ProcessPool.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184            0         0:00:03.135

CentOS-5.1$ pypy 12Python_NodeLayer_mirror.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184            0         0:00:06.241

CentOS-5.1$ pypy 11Python_NodeLayer.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184            0         0:00:06.160

CentOS-5.1$ pypy 10Python_bit_symmetry_ProcessPool.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184       285053         0:00:01.998

CentOS-5.1$ pypy 09Python_bit_symmetry_ThreadPool.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184       285053         0:00:02.111

CentOS-5.1$ pypy 08Python_bit_symmetry.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184       285053         0:00:03.026

CentOS-5.1$ pypy 07Python_bit_mirror.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184            0         0:00:06.274

CentOS-5.1$ pypy 06Python_bit_backTrack.py
 N:        Total       Unique        hh:mm:ss.ms
15:      2279184            0         0:00:12.610

CentOS-5.1$ pypy 05Python_optimize.py
 N:        Total       Unique         hh:mm:ss.ms
15:      2279184       285053         0:00:14.413

CentOS-5.1$ pypy 04Python_symmetry.py
 N:        Total       Unique         hh:mm:ss.ms
15:      2279184       285053         0:00:46.629

CentOS-5.1$ pypy 03Python_backTracking.py
 N:        Total       Unique         hh:mm:ss.ms
15:      2279184            0         0:00:44.993
"""
import subprocess
from datetime import datetime

# pypyを使うときは以下を活かしてcodon部分をコメントアウト
# pypy では ThreadPool/ProcessPoolが動きます 
#
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')

from threading import Thread
from multiprocessing import Pool as ThreadPool
import concurrent
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

class NQueens21:
    def __init__(self):
      pass
    # ProcessPoolしない
    # def bitmap_solve_nodeLayer(self,size:int,left:int,down:int,right:int)->int:
    # ProcessPoolする
    def bitmap_solve_nodeLayer(self,value:list)->int:
      # ProcessPoolする
      size,left,down,right=value # 追加
      mask:int=(1<<size)-1
      counter:int=0
      if down==mask: # 解が見つかった場合
        return 1
      bitmap:int=mask&~(left|down|right)
      while bitmap:
        bit=-bitmap&bitmap
        # ProcessPoolしない
        # counter+=self.bitmap_solve_nodeLayer(size,(left|bit)<<1,down|bit,(right|bit)>>1)
        #
        # ProcessPoolする
        value=[size,(left|bit)<<1,down|bit,(right|bit)>>1]
        counter+=self.bitmap_solve_nodeLayer(value)
        bitmap&=~bit
      return counter

    def countBits_nodeLayer(self,n:int)->int:
      counter:int=0
      while n:
        n&=(n-1)  # 右端の1を削除
        counter+=1
      return counter

    # def kLayer_nodeLayer_recursive(self,size:int,nodes:list,k:int,left:int,down:int,right:int)->int:
    def kLayer_nodeLayer_recursive(self,size:int,nodes:list,k:int,left:int,down:int,right:int)->None:
      counter:int=0
      mask:int=(1<<size)-1
      # すべてのdownが埋まったら、解決策を見つけたことになる
      if self.countBits_nodeLayer(down)==k:
        nodes.append(left)
        nodes.append(down)
        nodes.append(right)
        return 
      bit:int=0
      bitmap:int=mask&~(left|down|right)
      while bitmap:
        bit=-bitmap&bitmap
        # 解を加えて対角線をずらす
        # counter+=self.kLayer_nodeLayer_recursive(size,nodes,k,(left|bit)<<1,down|bit,(right|bit)>>1)
        self.kLayer_nodeLayer_recursive(size,nodes,k,(left|bit)<<1,down|bit,(right|bit)>>1)
        bitmap^=bit
      # return counter

    def kLayer_nodeLayer(self,size:int,nodes:list,k:int)->None:
      # サイズが偶数の場合のループ処理
      limit:int=size//2-1 if size % 2 else size//2
      for i in range(size//2):
        bit:int=1<<i
        # 再帰的に呼び出される処理を実行
        self.kLayer_nodeLayer_recursive(size,nodes,k,bit<<1,bit,bit>>1)
      bit:int
      left:int
      down:int
      right:int
      # サイズが奇数の場合の追加処理
      if size % 2:
        bit=1<<((size-1)//2)
        left=bit<<1
        down=bit
        right=bit>>1
        for i in range(limit):
          bit=1<<i
          self.kLayer_nodeLayer_recursive(size,nodes,k,(left|bit)<<1,down|bit,(right|bit)>>1)

    def mirror_build_nodeLayer(self,size:int)->int:
      # ツリーの3番目のレイヤーにあるノードを生成
      nodes:list[int]=[]
      k:int=4  # 3番目のレイヤーを対象
      self.kLayer_nodeLayer(size,nodes,k)
      # 必要なのはノードの半分だけで、各ノードは3つの整数で符号化
      # ミラーでは/6 を /3に変更する 
      num_solutions=len(nodes)//3

      # ProcessPoolしない
      # 15:      2279184            0         0:00:06.571
      # total:int=0
      # for i in range(num_solutions):
      #   total+=self.bitmap_solve_nodeLayer(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2])
      #
      # ProcessPoolする
      # 15:      2279184            0         0:00:03.048
      pool=ThreadPool(size)
      params=[(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2]) for i in range(num_solutions)]
      results:list[int]=list(pool.map(self.bitmap_solve_nodeLayer,params))
      total:int=sum(results)

      return total*2

class NQueens21_NodeLayer:
  def finalize(self)->None:
    cmd="killall pypy"  # python or pypy
    p = subprocess.Popen("exec " + cmd, shell=True)
    p.kill()
  def main(self)->None:
    nmin:int=4
    nmax:int=19
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      NQ=NQueens21()
      total:int=NQ.mirror_build_nodeLayer(size)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
      self.finalize()

# メイン実行部分
if __name__=="__main__":
    NQueens21_NodeLayer().main()
