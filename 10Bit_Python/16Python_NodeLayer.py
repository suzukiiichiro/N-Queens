#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ビットマップ版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

# 実行 
$ python <filename.py>

# 実行結果
1
 0 2 4 1 3
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+

2
 0 3 1 4 2
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+

3
 1 3 0 2 4
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+

4
 1 4 2 0 3
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+

5
 2 0 3 1 4
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+

6
 2 4 1 3 0
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+

7
 3 0 2 4 1
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+

8
 3 1 4 2 0
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+

9
 4 1 3 0 2
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+

10
 4 2 0 3 1
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+


bash-3.2$ python 07Python_carryChain.py
size: 5 TOTAL: 10 UNIQUE: 2
bash-3.2$

"""

#
# グローバル変数
MAX=21  # ボードサイズ最大値
TOTAL=0 # 解
#
# ビットマップ版ボードレイアウト出力
def printRecord_bitmap(size,flag):
  global TOTAL
  global board
  print(TOTAL)
  sEcho=""
  """
  ビットマップ版
     ビットマップ版からは、左から数えます
     上下反転左右対称なので、これまでの上から数える手法と
     rowを下にたどって左から数える方法と解の数に変わりはありません。
     0 2 4 1 3 
    +-+-+-+-+-+
    |O| | | | | 0
    +-+-+-+-+-+
    | | |O| | | 2
    +-+-+-+-+-+
    | | | | |O| 4
    +-+-+-+-+-+
    | |O| | | | 1
    +-+-+-+-+-+
    | | | |O| | 3
    +-+-+-+-+-+
  """
  if flag:
    for i in range(size):
      for j in range(size):
        if board[i]&1<<j:
          sEcho+=" " + str(j)
  else:
    """
    ビットマップ版以外
    (ブルートフォース、バックトラック、配置フラグ)
    上から数えます
       0 2 4 1 3 
      +-+-+-+-+-+
      |O| | | | |
      +-+-+-+-+-+
      | | | |O| |
      +-+-+-+-+-+
      | |O| | | |
      +-+-+-+-+-+
      | | | | |O|
      +-+-+-+-+-+
      | | |O| | |
      +-+-+-+-+-+
    """
    for i in range(size):
      sEcho+=" " + str(board[i])
  print(sEcho)

  print ("+",end="")
  for i in range(size):
    print("-",end="")
    if i<(size-1):
      print("+",end="")
  print("+")
  for i in range(size):
    print("|",end="")
    for j in range(size):
      if flag:
        if board[i]&1<<j:
          print("O",end="")
        else:
          print(" ",end="")
      else:
        if i==board[j]:
          print("O",end="")
        else:
          print(" ",end="")
      if j<(size-1):
        print("|",end="")
    print("|")
    if i in range(size-1):
      print("+",end="")
      for j in range(size):
        print("-",end="")
        if j<(size-1):
          print("+",end="")
      print("+")
  print("+",end="")
  for i in range(size):
    print("-",end="")
    if i<(size-1):
      print("+",end="")
  print("+")
  print("")
#
# ノードレイヤー

def bitmap_solve_nodeLayer(size, left, down, right):
    mask = (1 << size) - 1
    if down == mask:  # downがすべて専有され解が見つかる
        return 1
    counter = 0
    bit = 0
    bitmap = mask & ~(left | down | right)
    while bitmap:
        bit = -bitmap & bitmap
        counter += bitmap_solve_nodeLayer(size, (left | bit) >> 1, down | bit, (right | bit) << 1)
        bitmap ^= bit
    return counter


def countBits_nodeLayer(n):
    counter = 0
    while n:
        n &= (n - 1)  # 右端のゼロ以外の数字を削除
        counter += 1
    return counter


def kLayer_nodeLayer(size, nodes, k, left, down, right):
    counter = 0
    mask = (1 << size) - 1
    # すべてのdownが埋まったら、解決策を見つけたことになる
    if countBits_nodeLayer(down) == k:
        nodes.append(left)
        nodes.append(down)
        nodes.append(right)
        return 1
    bit = 0
    bitmap = mask & ~(left | down | right)
    while bitmap:
        bit = -bitmap & bitmap
        # 解を加えて対角線をずらす
        counter += kLayer_nodeLayer(size, nodes, k, (left | bit) >> 1, down | bit, (right | bit) << 1)
        bitmap ^= bit
    return counter


def bitmap_build_nodeLayer(size):
    # ツリーの3番目のレイヤーにあるノードを生成
    nodes = []
    k = 4  # 3番目のレイヤーを対象
    kLayer_nodeLayer(size, nodes, k, 0, 0, 0)

    # 必要なのはノードの半分だけで、各ノードは3つの整数で符号化される
    num_solutions = len(nodes) // 6
    total = 0
    print(nodes)
    for i in range(num_solutions):
        total += bitmap_solve_nodeLayer(size, nodes[3 * i], nodes[3 * i + 1], nodes[3 * i + 2])

    total *= 2
    return total

size=8
TOTAL=bitmap_build_nodeLayer(size)   # ４．ビットマップ
print("size:",size,"TOTAL:",TOTAL)