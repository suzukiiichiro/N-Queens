from datetime import datetime
# pypyで再帰が高速化できる

# pypyを使う場合はコメントを解除
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')


class NQueens11:
    def __init__(self):
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
    def kLayer_nodeLayer(self,size:int,nodes:list[int],k:int,left:int,down:int,right:int)->int:
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
        # ツリーの3番目のレイヤーにあるノードを生成
        nodes:list[int]=[]
        k:int=4  # 3番目のレイヤーを対象
        self.kLayer_nodeLayer(size,nodes,k,0,0,0)
        # 必要なのはノードの半分だけで、各ノードは3つの整数で符号化
        num_solutions:int=len(nodes)//6
        total:int=0
        for i in range(num_solutions):
          total+=self.bitmap_solve_nodeLayer(size,nodes[3*i],nodes[3*i+1],nodes[3*i+2])
        total *= 2
        return total
class NQueens11_NodeLayer:
  def main(self):
    nmin:int=4
    nmax:int=18
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin, nmax):
      start_time=datetime.now()
      NQ=NQueens11()
      total=NQ.bitmap_build_nodeLayer(size)
      time_elapsed=datetime.now()-start_time
      # `.format` の代わりに文字列として直接処理
      text=str(time_elapsed)[:-3]  
      print(f"{size:2d}:{total:13d}{0:13d}{text:20s}")

# $ python <filename>
# $ pypy <fileName>
# $ codon build -release <filename>
# ノードレイヤー
# 15:      2279184       285053         0:00:00.698
if __name__=="__main__":
    NQueens11_NodeLayer().main()
