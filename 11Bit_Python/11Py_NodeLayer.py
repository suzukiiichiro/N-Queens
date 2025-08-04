#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
ノードレイヤー版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

fedora$ python 11Py_NodeLayer.py
 N:        Total       Unique        hh:mm:ss.ms
 4:            2            0         0:00:00.000
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.001
 9:          352            0         0:00:00.004
10:          724            0         0:00:00.017
11:         2680            0         0:00:00.082
12:        14200            0         0:00:00.401
13:        73712            0         0:00:02.108
14:       365596            0         0:00:12.317
15:      2279184            0         0:01:16.692
16:     14772512            0         0:08:29.672
"""
from datetime import datetime

# pypyを使う場合はコメントを解除
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

class NQueens11:
  def __init__(self)->None:
    pass
  def bitmap_solve_nodeLayer(self,size:int,left:int,down:int,right:int) ->int:
    mask:int=(1<<size)-1
    if down==mask:  # 解が見つかった場合
      return 1
    counter:int=0
    bit:int=0
    bitmap:int=mask&~(left|down|right)
    while bitmap:
      bit=-bitmap&bitmap
      counter+=self.bitmap_solve_nodeLayer(size,(left|bit)>>1,down|bit,(right|bit)<<1)
      bitmap^=bit
    return counter
  def countBits_nodeLayer(self, n:int) ->int:
    counter:int=0
    while n:
      n&=(n-1)  # 右端の1を削除
      counter+=1
    return counter
  # ノードレイヤーのバックトラック
  # 解を導き出すために、ノードにより、クイーンの移動の候補を列挙する処理を行
  # います。
  # ノードレイヤーは、解を導き出すための処理に加えて、クイーンの移動候補を列
  # 挙するための処理を行うため事実上、２回エイトクイーンを行っていることが、
  # 速度低下の最大のボトルネックと言えます。
  # また、この章ではＱが角にあるか・ないかを判定していません。
  # よって枝刈りなどは行っていません。Nodelayerの対象解除で角判定と、枝刈り
  # を行います。
  def kLayer_nodeLayer(self,size:int,nodes:list,k:int,left:int,down:int,right:int)->int:
    counter:int=0
    mask:int=(1<<size)-1
    # すべてのdownが埋まったら、解決策を見つけたことになる
    if self.countBits_nodeLayer(down)==k:
      nodes.append(left)
      nodes.append(down)
      nodes.append(right)
      return 1
    bit:int=0
    bitmap:int=mask&~(left|down|right)
    while bitmap:
      bit=-bitmap&bitmap
      # 解を加えて対角線をずらす
      counter+=self.kLayer_nodeLayer(size,nodes,k,(left|bit)>>1,down|bit,(right|bit)<<1)
      bitmap^=bit
    return counter
  def bitmap_build_nodeLayer(self, size:int) ->int:
    # ツリーの3番目のレイヤーにあるノード
    # （それぞれ連続する3つの数字でエンコードされる）のベクトル
    # レイヤー2以降はノードの数が均等なので対称性を利用できる。
    # レイヤ4には十分なノードがある（N16の場合、9844）。
    # ここではレイヤーを５に設定、Ｎに併せて増やしていく。
    # Ｎが増えればレイヤーは枯渇します。
    # Ｎが１６まではレイヤーは４で足りますが、以降、レイヤーは、５，６と
    # 増やす必要があり、レイヤーが増えることによって、速度は加速度的に遅
    # くなります。
    # ノードレイヤーの考え方はスマートではありますが、Ｎの最大化と高速化を
    # 求める場合は限界がまもなくおとずれるロジックです。
    nodes:list[int]=[]
    #
    # 第３引数の4は4行目までnqueenを実行し、それまでのleft,down,rightをnodes配列に格納するという意味になります。
    k:int=4  # 3番目のレイヤーを対象
    self.kLayer_nodeLayer(size,nodes,k,0,0,0)
    # 必要なのはノードの半分だけで、各ノードは3つの整数で符号化される
    # ３の整数というのは、一つのノードに`left` `down` `right` が格納されるからです。
    # nodes配列は3個で１セットで`left` `dwon` `right`の情報を同じ配列に格納します
    num_solutions:int=len(nodes)//3
    # 以下は、スレッドごとに格納された解をforで回して集計します。
    total:int=0
    for i in range(num_solutions):
      total+=self.bitmap_solve_nodeLayer(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2])
    return total
class NQueens11_NodeLayer:
  def main(self)->None:
    nmin:int=4
    nmax:int=18
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin, nmax):
      start_time=datetime.now()
      NQ=NQueens11()
      total=NQ.bitmap_build_nodeLayer(size)
      unique=0
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")
if __name__=="__main__":
  NQueens11_NodeLayer().main()
