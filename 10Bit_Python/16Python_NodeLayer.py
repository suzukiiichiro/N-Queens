import os
from datetime import datetime
# pypyで再帰が高速化できる
# pypyを使う場合はコメントを解除
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
#
class NQueens21():
  def __init__(self):
    pass
  def bitmap_solve_nodeLayer(self,size, left, down, right):
    mask = (1 << size) - 1
    if down == mask:  # downがすべて専有され解が見つかる
      return 1
    counter = 0
    bit = 0
    bitmap = mask & ~(left | down | right)
    while bitmap:
      bit = -bitmap & bitmap
      counter += self.bitmap_solve_nodeLayer(size, (left | bit) >> 1, down | bit, (right | bit) << 1)
      bitmap ^= bit
    return counter
  def countBits_nodeLayer(self,n):
    counter = 0
    while n:
      n &= (n - 1)  # 右端のゼロ以外の数字を削除
      counter += 1
    return counter
  def kLayer_nodeLayer(self,size, nodes, k, left, down, right):
    counter = 0
    mask = (1 << size) - 1
    # すべてのdownが埋まったら、解決策を見つけたことになる
    if self.countBits_nodeLayer(down) == k:
      nodes.append(left)
      nodes.append(down)
      nodes.append(right)
      return 1
    bit = 0
    bitmap = mask & ~(left | down | right)
    while bitmap:
      bit = -bitmap & bitmap
      # 解を加えて対角線をずらす
      counter += self.kLayer_nodeLayer(size, nodes, k, (left | bit) >> 1, down | bit, (right | bit) << 1)
      bitmap ^= bit
    return counter
  def bitmap_build_nodeLayer(self,size):
    # ツリーの3番目のレイヤーにあるノードを生成
    nodes = []
    k = 4  # 3番目のレイヤーを対象
    self.kLayer_nodeLayer(size, nodes, k, 0, 0, 0)
    # 必要なのはノードの半分だけで、各ノードは3つの整数で符号化
    num_solutions = len(nodes) // 6
    total = 0
    for i in range(num_solutions):
      total += self.bitmap_solve_nodeLayer(size, nodes[3 * i], nodes[3 * i + 1], nodes[3 * i + 2])
    total *= 2
    return total
class NQueens21_NodeLayer:
  def main(self):
    nmin = 4
    nmax = 18
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for i in range(nmin, nmax):
      start_time=datetime.now()
      NQ=NQueens21()
      total=NQ.bitmap_build_nodeLayer(i)
      time_elapsed=datetime.now()-start_time
      _text='{}'.format(time_elapsed)
      text=_text[:-3]
      print("%2d:%13d%13d%20s"%(i,total,0, text))
#
# ノードレイヤー
# 15:      2279184            0         0:00:05.148
if __name__ == '__main__':
  NQueens21_NodeLayer().main()

