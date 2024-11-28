from datetime import datetime
from typing import List

class NQueens21:
    def __init__(self):
        pass

    def bitmap_solve_nodeLayer(self, size: int, left: int, down: int, right: int) -> int:
        mask = (1 << size) - 1
        if down == mask:  # 解が見つかった場合
            return 1
        counter = 0
        bit = 0
        bitmap = mask & ~(left | down | right)
        while bitmap:
            bit = -bitmap & bitmap
            counter += self.bitmap_solve_nodeLayer(size, (left | bit) >> 1, down | bit, (right | bit) << 1)
            bitmap ^= bit
        return counter

    def countBits_nodeLayer(self, n: int) -> int:
        counter = 0
        while n:
            n &= (n - 1)  # 右端の1を削除
            counter += 1
        return counter

    def kLayer_nodeLayer(self, size: int, nodes: List[int], k: int, left: int, down: int, right: int) -> int:
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

    def bitmap_build_nodeLayer(self, size: int) -> int:
        # ツリーの3番目のレイヤーにあるノードを生成
        nodes: List[int] = []
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
    def main(self) -> None:
        nmin = 4
        nmax = 18
        print(" N:        Total       Unique        hh:mm:ss.ms")
        for i in range(nmin, nmax):
            start_time = datetime.now()
            NQ = NQueens21()
            total = NQ.bitmap_build_nodeLayer(i)
            time_elapsed = datetime.now() - start_time
            text = str(time_elapsed)[:-3]  # `.format` の代わりに文字列として直接処理
            print(f"{i:2d}:{total:13d}{0:13d}{text:20s}")


# メイン実行部分
if __name__ == "__main__":
    NQueens21_NodeLayer().main()
