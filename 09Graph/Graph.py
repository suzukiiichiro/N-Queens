
# $B%$%s%9%H!<%k(B
# pip install networkx
# pip install numpy
# pip install matplotlib

# $B<B9T(B
# python Graph.ph
# $BBg$-$J2hA|$O%&%$%s%I%&$r:GBg2=$9$k$H2?$H$J$/8+$($k(B
#

import networkx as nx

# $BHWLL$N0lJU$N%^%9?t(B
bord_size = 8

# $B%0%i%U@8@.(B
G = nx.Graph()

# $B%N!<%I$r:n@.$9$k(B
for i in range(bord_size):
    for j in range(bord_size):
        G.add_node((i, j))

# $BF1$89T$+F1$8Ns$KJB$V%N!<%I$r%(%C%8$G7k$V(B
for i in range(bord_size):
    for m in range(bord_size):
        for n in range(m+1, bord_size):
            G.add_edge((i, m), (i, n))
            G.add_edge((m, i), (n, i))

# $BF1$8<P$a@~>u$KJB$V%N!<%I$r%(%C%8$G7k$V(B
for n1 in G.nodes:
    for n2 in G.nodes:
        if n1 == n2:
            continue
        elif n1[0]+n1[1] == n2[0]+n2[1]:
            G.add_edge(n1, n2)
        elif n1[0]-n1[1] == n2[0]-n2[1]:
            G.add_edge(n1, n2)


# $BJd%0%i%U$rF@$k(B ($B$*8_$$$K<h$j9g$o$J$$JU$,7k$P$l$F$$$k(B)
G_complement = nx.complement(G)

# $B%5%$%:$,JQ$N?t$KEy$7$$%/%j!<%/$N0lMw$rF@$k(B
answers = [
        clieque for clieque in nx.find_cliques(G_complement)
        if len(clieque) == bord_size
    ]

# $BF@$i$l$?2r$N8D?t(B
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
