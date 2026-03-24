import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from networkx.algorithms.community import greedy_modularity_communities, modularity

# 1. Генерация случайного графа по модели Эрдеша-Реньи
n = 1000
p = 0.01
G = nx.erdos_renyi_graph(n, p, seed=42)

print("\nГраф Эрдеша-Реньи")
print(f"Число вершин: {G.number_of_nodes()}")
print(f"Число ребер: {G.number_of_edges()}")

# 2.1 Распределение степеней вершин
deg = [d for n, d in G.degree()]
deg_count = Counter(deg)

plt.figure(figsize=(10, 6))
plt.bar(deg_count.keys(), deg_count.values(), alpha=0.7)
plt.xlabel('Степень вершины k')
plt.ylabel('Количество вершин')
plt.title('Распределение степеней вершин в ER')
plt.show()

print("\n1. Распределение степеней вершин (первые 10)")
for k in sorted(deg_count.keys())[:10]:
    print(f"Степень {k}: {deg_count[k]} вершин")

# 2.2 Средняя степень вершины
mid_deg = np.mean(deg)
print(f"\n2. Средняя степень вершины: {mid_deg:.3f}")

# 2.3 Коэффициент кластеризации
cluster = nx.average_clustering(G)
print(f"\n3. Коэффициент кластеризации: {cluster:.6f}")

# 2.4 Размер гигантской компоненты связности
large_comp = max(nx.connected_components(G), key=len)
size = len(large_comp)
print(f"\n4. Размер гигантской компоненты связности: {size}")

# 2.5 Средняя длина пути для связных компонент
print(f"\n5. Средняя длина пути для связных компонент:")
components = list(nx.connected_components(G))
for i, comp in enumerate(components):
    if len(comp) > 1:
        subgraph = G.subgraph(comp)
        mid_way = nx.average_shortest_path_length(subgraph)
        print(f"Компонента {i+1}: размер={len(comp)}, средняя длина пути={mid_way:.3f}")
    else:
        print(f"Компонента {i+1}: размер=1 (изолированная вершина)")

# вторая часть задания

# 3. Генерация стохастической блочной модели с двумя сообществами
block_sizes = [500, 500]
p_in = 0.1
p_out = 0.005

probs = [[p_in, p_out],
         [p_out, p_in]]

G_sbm = nx.stochastic_block_model(block_sizes, probs, seed=42)

print("\nСтохастическая блочная модель(SBM)")
print(f"Число вершин: {G_sbm.number_of_nodes()}")
print(f"Число ребер: {G_sbm.number_of_edges()}")
print(f"Вероятность внутри сообществ: {p_in}")
print(f"Вероятность между сообществами: {p_out}")

# 4. Сравнение характеристик СБМ с моделью Эрдеша-Реньи

# 4.1 Распределение степеней вершин
deg_sbm = [d for n, d in G_sbm.degree()]
deg_counts_sbm = Counter(deg_sbm)

plt.figure(figsize=(10, 6))
plt.bar(deg_counts_sbm.keys(), deg_counts_sbm.values(), alpha=0.7, color='orange')
plt.xlabel('Степень вершины k')
plt.ylabel('Количество вершин')
plt.title('Распределение степеней вершин (SBM)')
plt.tight_layout()
plt.show()

print("\n1. Распределение степеней вершин (SBM):")
for k in sorted(deg_counts_sbm.keys())[:10]:
    print(f"   Степень {k}: {deg_counts_sbm[k]} вершин")

# 4.2 Средняя степень вершины
mid_deg_sbm = np.mean(deg_sbm)
print(f"\n2. Средняя степень вершины: {mid_deg_sbm:.3f}")

# 4.3 Коэффициент кластеризации
cluster_sbm = nx.average_clustering(G_sbm)
print(f"\n3. Коэффициент кластеризации: {cluster_sbm:.6f}")

# 4.4 Размер гигантской компоненты связности
large_comp_sbm = max(nx.connected_components(G_sbm), key=len)
size_sbm = len(large_comp_sbm)
print(f"\n4. Размер гигантской компоненты: {size_sbm}")

# 4.5 Средняя длина пути (для гигантской компоненты)
G_giant_sbm = G_sbm.subgraph(large_comp_sbm)
mid_sbm = nx.average_shortest_path_length(G_giant_sbm)
print(f"\n5. Средняя длина пути в гиантской компоненте: {mid_sbm:.3f}")

# 4.6 Структура сообществ (используем модульность)
print("\n Структура сообществ: ")

# в SBM
comm_sbm = list(greedy_modularity_communities(G_sbm))
mod_sbm = modularity(G_sbm, comm_sbm)
print("\nСтохастическая блочная модель (SBM):")
print(f"Количество сообществ: {len(comm_sbm)}")
print(f"Размеры сообществ: {sorted([len(c) for c in comm_sbm], reverse=True)}")
print(f"Модульность: {mod_sbm:.4f}")

# в ER графе
comm_er = list(greedy_modularity_communities(G))
mod_er = modularity(G, comm_er)
print("\nМодель Эрдеша-Реньи (ER):")
print(f"Количество сообществ: {len(comm_er)}")
print(f"Размеры сообществ: {sorted([len(c) for c in comm_er], reverse=True)}")
print(f"Модульность: {mod_er:.4f}")


plt.subplot(2, 2, 3)
metrics = ['Кластеризация', 'Модульность']
er_compare = [cluster, mod_er]
sbm_compare = [cluster_sbm, mod_sbm]
x = np.arange(len(metrics))
width = 0.35
plt.bar(x - width/2, er_compare, width, label='ER', alpha=0.7)
plt.bar(x + width/2, sbm_compare, width, label='SBM', alpha=0.7)
plt.xlabel('Характеристика')
plt.ylabel('Значение')
plt.title('Сравнение кластеризации и модульности')
plt.xticks(x, metrics)
plt.legend()

# сравнение средней длины пути и степени
plt.subplot(2, 2, 4)
metrics2 = ['Ср. степень', 'Ср. длина пути']
er_compare2 = [mid_deg, mid_way]
sbm_compare2 = [mid_deg_sbm, mid_sbm]
plt.bar(x - width/2, er_compare2, width, label='ER', alpha=0.7)
plt.bar(x + width/2, sbm_compare2, width, label='SBM', alpha=0.7)
plt.xlabel('Характеристика')
plt.ylabel('Значение')
plt.title('Сравнение степени и длины пути')
plt.xticks(x, metrics2)
plt.legend()
plt.tight_layout()
plt.show()