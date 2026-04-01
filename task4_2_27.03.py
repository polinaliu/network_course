import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

# Создание графа SBM на 300 узлов
n = 300
sizes = [50, 50, 100, 100]
p_in = 0.2
p_out = 0.01

probs = np.full((4, 4), p_out)
np.fill_diagonal(probs, p_in)

G = nx.stochastic_block_model(sizes, probs, seed=42)
print(f"Граф: {G.number_of_nodes()} узлов, {G.number_of_edges()} ребер")


# Модель ICM
class ICM:
    def __init__(self, graph, prob=0.15):
        self.graph = graph
        self.prob = prob

    def sim(self, seeds):
        active = set(seeds)
        new = set(seeds)

        while new:
            next = set()
            for node in new:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in active:
                        if random.random() < self.prob:
                            next.add(neighbor)
            active.update(next)
            new = next

        return len(active)


# жадный алгоритм
def greedy_algorithm(G, k, icm, n=10):
    s = []

    for st in range(k):
        best_node = None
        best_spread = -1

        for node in G.nodes():
            if node not in s:
                candidates = s + [node]
                total_spread = 0
                for _ in range(n):
                    total_spread += icm.sim(candidates)
                mid_spread = total_spread / n

                if mid_spread > best_spread:
                    best_spread = mid_spread
                    best_node = node

        s.append(best_node)
        print(f"Шаг {st + 1}: узел {best_node}, распр = {best_spread:.2f}")

    return s


# выбор по степени
def degree_selection(G, k):
    degrees = dict(G.degree())
    return sorted(degrees, key=degrees.get, reverse=True)[:k]

# допустим k=5
k = 5
icm = ICM(G, prob=0.15)

print(f"\nЖадный алгоритм:")
greedy_seeds = greedy_algorithm(G, k, icm)

print(f"\nВыбор по степени:")
degree_seeds = degree_selection(G, k)
print(f"Узлы: {degree_seeds}")


# Оценка
def evaluate(seeds, icm, n_runs=50):
    return np.mean([icm.sim(seeds) for _ in range(n_runs)])

greedy_res = evaluate(greedy_seeds, icm)
deg_res = evaluate(degree_seeds, icm)

print(f"\nРезультаты:")
print(f"Жадный алгоритм: {greedy_res:.2f}")
print(f"По степени: {deg_res:.2f}")
print(f"Разница: {greedy_res - deg_res:.2f}")

print(f"Вывод:")
print(f"Жадный алгоритм показал результат на {(greedy_res/deg_res - 1)*100:.1f}% лучше")

# визуализация
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
pos = nx.spring_layout(G, seed=42)

# жадный алгоритм
colors_greedy = ['red' if node in greedy_seeds else 'blue' for node in G.nodes()]
nx.draw(G, pos, ax=axes[0], node_color=colors_greedy, node_size=50,
        edge_color='gray', alpha=0.6, with_labels=False)
axes[0].set_title(f'Жадный алгоритм\nРаспространение: {greedy_res:.1f}')

# по степени
colors_degree = ['red' if node in degree_seeds else 'blue' for node in G.nodes()]
nx.draw(G, pos, ax=axes[1], node_color=colors_degree, node_size=50,
        edge_color='gray', alpha=0.6, with_labels=False)
axes[1].set_title(f'По степени\nРаспространение: {deg_res:.1f}')

plt.tight_layout()
plt.show()