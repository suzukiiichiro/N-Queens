#NetworkX
import matplotlib.pyplot as plt
import networkx as nx

#無向グラフ
G=nx.Graph()

G.add_nodes_from(["A","B","C","D","E","F"])
G.add_edges_from([("A","B"),("B","C"),("B","F"),("C","D"),("C","E"),("C","F"),("B","F")])

print("number of nodes: ", G.number_of_nodes())
print("nodes:",G.nodes())

print("number of edges:",G.number_of_edges())
print("edges:",G.edges())

print("dgrees:",G.degrees())

#nx.draw(G,with_labels = True)
#print(nx.info(G))
#
#plt.show()
