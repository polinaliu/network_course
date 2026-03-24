import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

G = nx.karate_club_graph()

# жадный алгоритм ньюмана
communities = list(greedy_modularity_communities(G))
print(f"Количество сообществ: {len(communities)}")

node_to_community = {}
for i, comm in enumerate(communities):
    for node in comm:
        node_to_community[node] = i

colors = [node_to_community[n] for n in G.nodes()]
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(8, 6))
nx.draw(G, pos, node_color=colors, cmap=plt.cm.Set1, with_labels=True, node_size=500, font_color='white')
plt.title("Клуб карате")
plt.show()