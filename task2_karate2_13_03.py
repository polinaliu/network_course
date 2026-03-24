import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.metrics import normalized_mutual_info_score


def create_test_network(big=30, small=4, num_small=3):

    G = nx.Graph()
    true_comm = {}
    comm_id = 0
    node_counter = 0

    # Большое сообщество
    big_nodes = list(range(node_counter, node_counter + big))
    node_counter += big

    # Внутри большого сообщества
    for i in big_nodes:
        for j in big_nodes:
            if i < j and np.random.random() < 0.25:
                G.add_edge(i, j)
    for node in big_nodes:
        true_comm[node] = comm_id
    comm_id += 1

    # Маленькие сообщества
    for _ in range(num_small):
        nodes = list(range(node_counter, node_counter + small))
        node_counter += small

        # Внутри маленького сообщества (полный граф)
        for u in nodes:
            for v in nodes:
                if u < v:
                    G.add_edge(u, v)

        for node in nodes:
            true_comm[node] = comm_id

        # Связи с большим сообществом
        for _ in range(small * 2):
            u = np.random.choice(big_nodes)
            v = np.random.choice(nodes)
            G.add_edge(u, v)

        comm_id += 1

    return G, true_comm

# Создаем одну тестовую сеть
G, true_comm = create_test_network(big=25, small=4, num_small=3)
true_count = len(set(true_comm.values()))
print(f"Истинное количество сообществ: {true_count}")

# Тестируем разные значения resolution
resolutions = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]
results = []

print(f"\n{'Resolution':13}  {'Найдено':7}  {'NMI':6}  {'Эффект':12}")

for res in resolutions:
    detected = list(greedy_modularity_communities(G, resolution=res))

    # NMI
    nodes_list = list(G.nodes())
    n_nodes = len(nodes_list)
    true_labels = np.zeros(n_nodes)
    pred_labels = np.zeros(n_nodes)

    for i, node in enumerate(nodes_list):
        true_labels[i] = true_comm[node]

    for i, comm in enumerate(detected):
        for node in comm:
            pred_labels[nodes_list.index(node)] = i

    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    results.append((res, len(detected), nmi))

    if len(detected) < true_count:
        effect = "СЛИЯНИЕ"
    elif len(detected) > true_count:
        effect = "ДРОБЛЕНИЕ"
    else:
        effect = "НОРМА"

    print(f"{res:10.2f}  {len(detected):10d}  {nmi:.4f}  {effect:12}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
res_vals = [r[0] for r in results]
detected_vals = [r[1] for r in results]
plt.plot(res_vals, detected_vals, 'ro-', linewidth=2, markersize=8)
plt.axhline(y=true_count, color='g', linestyle='--', label=f'Истинное ({true_count})')
plt.xlabel('Параметр resolution')
plt.ylabel('Найденное количество сообществ')
plt.title('Влияние resolution на количество сообществ')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
nmi_vals = [r[2] for r in results]
plt.plot(res_vals, nmi_vals, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Параметр resolution')
plt.ylabel('NMI')
plt.title('Влияние resolution на качество')
plt.grid(True, alpha=0.3)

# Визуализация сети с разными resolution
plt.subplot(1, 3, 3)
pos = nx.spring_layout(G, seed=42)

# Для resolution=0.7 (должно быть слияние)
detected_low = list(greedy_modularity_communities(G, resolution=0.7))
colors = np.zeros(G.number_of_nodes())
for i, comm in enumerate(detected_low):
    for node in comm:
        colors[node] = i

nx.draw(G, pos, node_color=colors, cmap=plt.cm.Set1,
        node_size=100, with_labels=False, alpha=0.8)
plt.title(f'Resolution=0.7: найдено {len(detected_low)} сообществ')

plt.tight_layout()
plt.show()
