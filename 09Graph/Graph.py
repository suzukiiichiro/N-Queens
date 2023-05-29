
# インストール
# pip install networkx
# pip install numpy
# pip install matplotlib

# 実行
# python Graph.ph
# 大きな画像はウインドウを最大化すると何となく見える
#

import networkx as nx

# 盤面の一辺のマス数
bord_size = 8

# グラフ生成
G = nx.Graph()

# ノードを作成する
for i in range(bord_size):
  for j in range(bord_size):
    G.add_node((i, j))

# 同じ行か同じ列に並ぶノードをエッジで結ぶ
for i in range(bord_size):
  for m in range(bord_size):
    for n in range(m+1, bord_size):
      G.add_edge((i, m), (i, n))
      G.add_edge((m, i), (n, i))

# 同じ斜め線状に並ぶノードをエッジで結ぶ
for n1 in G.nodes:
  for n2 in G.nodes:
    if n1 == n2:
      continue
    elif n1[0]+n1[1] == n2[0]+n2[1]:
      G.add_edge(n1, n2)
    elif n1[0]-n1[1] == n2[0]-n2[1]:
      G.add_edge(n1, n2)

# 補グラフを得る (お互いに取り合わない辺が結ばれている)
G_complement = nx.complement(G)

# サイズが変の数に等しいクリークの一覧を得る
answers = [
    clieque for clieque in nx.find_cliques(G_complement)
    if len(clieque) == bord_size
    ]

# 得られた解の個数
print(len(answers))
# 92


import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 30), facecolor="w")

for i, answer in enumerate(answers, start=1):
  bord = np.zeros([bord_size, bord_size])
  ax = fig.add_subplot(16, 6, i)
  for cell in answer:
    bord[cell] = 1
    ax.imshow(bord, cmap="gray_r")

plt.show()

