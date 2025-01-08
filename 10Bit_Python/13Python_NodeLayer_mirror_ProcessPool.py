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
  def main(self)->None:
    nmin:int=5
    nmax:int=17
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      NQ=NQueens21()
      total:int=NQ.mirror_build_nodeLayer(size)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]  
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")

# メイン実行部分
if __name__=="__main__":
    NQueens21_NodeLayer().main()
