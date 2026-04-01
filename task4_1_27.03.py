import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

# Создание графа SBM на 300 узлов
n = 300
sizes = [50, 50, 100, 100]  # 2 сообщества по 50 и 2 по 100
p_in = 0.2
p_out = 0.01
probs = np.full((4, 4), p_out)
np.fill_diagonal(probs, p_in)

# 1. Создаем граф

G = nx.stochastic_block_model(sizes, probs, seed=42)
print(f"Граф: {G.number_of_nodes()} узлов, {G.number_of_edges()} ребер")

# 2. Модель ICM

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

        return len(active), active

# Выбираем начальное множество S0 (10 узлов с наибольшей степенью)
degrees = dict(G.degree())
S0 = sorted(degrees, key=degrees.get, reverse=True)[:10]
print(f"\nНачальное множество S0: {S0}")

# 3. Имитационное моделирование (несколько прогонов)
k = 10
sizes = []
icm = ICM(G, prob=0.15)

for i in range(k):
    size, _ = icm.sim(S0)
    sizes.append(size)
    print(f"  Прогон {i+1}: активировано {size} узлов")

mid_size = sum(sizes) / k
std_size = (sum((x - mid_size) ** 2 for x in sizes) / k) ** 0.5

print(f"\nРезультаты усреднения:")
print(f"  Средний размер активации: {mid_size:.2f}")
print(f"  Стандартное отклонение: {std_size:.2f}")
print(f"  Минимальный размер: {min(sizes)}")
print(f"  Максимальный размер: {max(sizes)}")

# 4. Визуализация (один прогон)
_, final_active = icm.sim(S0)

print(f"\nРезультаты прогона для визуализации:")
print(f"  Начальное множество S0 (красный цвет): {sorted(S0)}")
print(f"  Количество начальных узлов: {len(S0)}")
print(f"  Количество активированных узлов (оранжевый цвет): {len(final_active) - len(S0)}")
print(f"  Общее количество активированных узлов: {len(final_active)}")
print(f"  Количество неактивированных узлов (синий): {n - len(final_active)}")

color = []
for node in G.nodes():
    if node in S0:
        color.append('red')
    elif node in final_active:
        color.append('orange')
    else:
        color.append('blue')

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)
nx.draw(G, pos, node_color=color, node_size=50, edge_color='gray', alpha=0.6, with_labels=False, width=0.3)
plt.title(f'Распространение информации в ICM\n'
          f'Активировано: {len(final_active)} из {n} узлов ({len(final_active)/n*100:.1f}%)', fontsize=14)
plt.show()





