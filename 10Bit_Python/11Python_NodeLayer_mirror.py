from datetime import datetime
from typing import List

class NQueens21:
    def __init__(self):
        pass

    def bitmap_solve_nodeLayer(self, size: int, left: int, down: int, right: int) -> int:
        mask = (1 << size) - 1
        counter = 0
        if down == mask:  # 解が見つかった場合
            return 1
        
   
        bitmap = mask & ~(left | down | right)
        while bitmap:
            bit = -bitmap & bitmap
            counter += self.bitmap_solve_nodeLayer(size, (left | bit) << 1, down | bit, (right | bit) >> 1)
            bitmap &= ~bit
        return counter

    def countBits_nodeLayer(self, n: int) -> int:
        counter = 0
        while n:
            n &= (n - 1)  # 右端の1を削除
            counter += 1
        return counter

    def kLayer_nodeLayer_recursive(self, size: int, nodes: List[int], k: int, left: int, down: int, right: int) -> int:
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
            counter += self.kLayer_nodeLayer_recursive(size, nodes, k, (left | bit) << 1, down | bit, (right | bit) >> 1)
            bitmap ^= bit
        return counter
    def kLayer_nodeLayer(self,size: int, nodes: List[int], k: int):
        # サイズが偶数の場合のループ処理
        limit = size // 2 - 1 if size % 2 else size // 2
        for i in range(size // 2):
            bit = 1 << i
            # 再帰的に呼び出される処理を実行
            self.kLayer_nodeLayer_recursive(size, nodes, k, bit << 1, bit, bit >> 1)

        # サイズが奇数の場合の追加処理
        if size % 2:
            bit = 1 << ((size - 1) // 2)
            left = bit << 1
            down = bit
            right = bit >> 1
            for i in range(limit):
                bit = 1 << i
                self.kLayer_nodeLayer_recursive(size, nodes, k, (left | bit) << 1, down | bit, (right | bit) >> 1)

    def mirror_build_nodeLayer(self, size: int) -> int:
        # ツリーの3番目のレイヤーにあるノードを生成
        nodes: List[int] = []
        k = 4  # 3番目のレイヤーを対象
        self.kLayer_nodeLayer(size, nodes, k)
        # 必要なのはノードの半分だけで、各ノードは3つの整数で符号化
        # ミラーでは/6 を /3に変更する 
        num_solutions = len(nodes) // 3
        #print(nodes)
        #print(num_solutions)
        total = 0
        for i in range(num_solutions):
            total += self.bitmap_solve_nodeLayer(size, nodes[3 * i], nodes[3 * i + 1], nodes[3 * i + 2])
        total *= 2
        return total


class NQueens21_NodeLayer:
    def main(self) -> None:
        nmin = 5
        nmax = 17
        print(" N:        Total       Unique        hh:mm:ss.ms")
        for i in range(nmin, nmax):
            start_time = datetime.now()
            NQ = NQueens21()
            total = NQ.mirror_build_nodeLayer(i)
            time_elapsed = datetime.now() - start_time
            text = str(time_elapsed)[:-3]  # `.format` の代わりに文字列として直接処理
            print(f"{i:2d}:{total:13d}{0:13d}{text:20s}")


# メイン実行部分
if __name__ == "__main__":
    NQueens21_NodeLayer().main()
