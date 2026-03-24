import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

N = 3
total_nodes = N + 1

A_star = [[0] * total_nodes for _ in range(total_nodes)]

for i in range(1, total_nodes):
    A_star[0][i] = 1
    A_star[i][0] = 1

print(f"\nНЕОРИЕНТИРОВАННАЯ ЗВЕЗДА С {total_nodes} УЗЛАМИ (N={N})")
print("\nМатрица смежности:")
for row in A_star:
    print(row)

G = nx.Graph()
G.add_nodes_from(range(total_nodes))

for i in range(total_nodes):
    for j in range(total_nodes):
        if A_star[i][j] == 1:
            G.add_edge(i, j)

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)
node_colors = ['red'] + ['lightblue'] * N
node_sizes = [800] + [500] * N
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
nx.draw_networkx_edges(G, pos, edge_color='gray', width=2)

plt.title(f"Неориентированная звезда (N={N}, всего узлов={total_nodes})")
plt.axis('off')
plt.tight_layout()
plt.show()

# 1. ЦЕНТРАЛЬНОСТЬ ПО СОБСТВЕННОМУ ВЕКТОРУ

print("\n1. ЦЕНТРАЛЬНОСТЬ ПО СОБСТВЕННОМУ ВЕКТОРУ")

A_array = np.array(A_star)
eigenvalues, eigenvectors = np.linalg.eig(A_array)

idx_max = np.argmax(eigenvalues.real)
lambda_max_num = eigenvalues[idx_max].real
print(f"\nλ_max = {lambda_max_num:.6f}")

eigenvector = eigenvectors[:, idx_max].real
eigenvector = np.abs(eigenvector)
eigenvector_norm = eigenvector / np.sum(eigenvector)

print(f"\nЦентральности:")
for i in range(total_nodes):
    if i == 0:
        print(f"Центр (узел {i}): {eigenvector_norm[i]:.6f}")
    else:
        print(f"Лист {i}: {eigenvector_norm[i]:.6f}")

# Проверка через NetworkX
print(f"\nПроверка через NetworkX:")
eig_nx = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
total_nx = sum(eig_nx.values())
for node in sorted(eig_nx.keys()):
    if node == 0:
        print(f"Центр (узел {node}): {eig_nx[node] / total_nx:.6f}")
    else:
        print(f"Лист {node}: {eig_nx[node] / total_nx:.6f}")
print()

# 2. PAGERANK

print("2. PAGERANK")

def pagerank_star(g, alpha, epsilon=1e-10, max_iter=1000):
    n = len(g)

    degree = [sum(row) for row in g]

    c = [1.0 / n] * n

    print(f"\nalpha = {alpha}")
    print(f"Начальные значения: c = {[round(x, 6) for x in c]}")
    print(f"Степени узлов: degree = {degree}")

    for t in range(max_iter):
        c_new = [0] * n
        for i in range(n):
            s = (1 - alpha) / n

            for j in range(n):
                if g[j][i] == 1:
                    s += alpha * (1.0 / degree[j]) * c[j]
            c_new[i] = s

        diff = max(abs(c_new[i] - c[i]) for i in range(n))
        if t < 5 or t % 20 == 0:
            print(f"Итерация {t + 1:3d}: c = {[round(x, 6) for x in c_new]}")

        if diff < epsilon:
            print(f"Сходимость достигнута на итерации {t + 1}")
            return c_new

        c = c_new

    return c


# Расчет для α = 0.25
pr25 = pagerank_star(A_star, 0.25)
print(f"\nРЕЗУЛЬТАТ для α=0.25:")
for i in range(total_nodes):
    if i == 0:
        print(f"Центр (узел {i}): {pr25[i]:.8f}")
    else:
        print(f"Лист {i}: {pr25[i]:.8f}")
print(f"Сумма = {sum(pr25):.8f}")

# Расчет для α = 0.50
pr50 = pagerank_star(A_star, 0.50)
print(f"\nРЕЗУЛЬТАТ для α=0.50:")
for i in range(total_nodes):
    if i == 0:
        print(f"Центр (узел {i}): {pr50[i]:.8f}")
    else:
        print(f"Лист {i}: {pr50[i]:.8f}")
print(f"Сумма = {sum(pr50):.8f}")

print("\nИЗМЕНЕНИЕ ПРИ УВЕЛИЧЕНИИ α:")
print(f"  Центр (узел 0): {pr25[0]:.6f} → {pr50[0]:.6f}  (изменение: {pr50[0] - pr25[0]:+.6f})")
for i in range(1, min(3, total_nodes)):
    print(f"Лист {i}: {pr25[i]:.6f} → {pr50[i]:.6f}  (изменение: {pr50[i] - pr25[i]:+.6f})")
print()

# 3. ЦЕНТРАЛЬНОСТЬ КАЦА-БОНАЧИЧА

print("3. ЦЕНТРАЛЬНОСТЬ КАЦА-БОНАЧИЧА")

def katz_bonacich_star(g, alpha, beta=1.0):

    n = len(g)
    g_array = np.array(g)

    I = np.eye(n)
    M = I - alpha * g_array
    det = np.linalg.det(M)

    if abs(det) < 1e-10:
        print("Центральность не определена однозначно.")
        return None

    ones = np.ones(n)

    c = beta * np.linalg.solve(M, ones)

    return c


print(f"Параметры: α = 0.5, β = 1.0")
katz = katz_bonacich_star(A_star, 0.5, 1.0)

if katz is not None:
    print(f"\nЦентральность Каца-Боначича:")
    for i in range(total_nodes):
        if i == 0:
            print(f"Центр (узел {i}): {katz[i]:.8f}")
        else:
            print(f"Лист {i}: {katz[i]:.8f}")

    katz_norm = katz / np.sum(katz)
    print(f"\nНормированная (Σ=1):")
    for i in range(total_nodes):
        if i == 0:
            print(f"Центр (узел {i}): {katz_norm[i]:.8f}")
        else:
            print(f"Лист {i}: {katz_norm[i]:.8f}")


# 4. УСЛОВИЕ ОДНОЗНАЧНОЙ ОПРЕДЕЛЕННОСТИ
print("\n4. УСЛОВИЕ ОДНОЗНАЧНОЙ ОПРЕДЕЛЕННОСТИ ДЛЯ ЗВЕЗДЫ")

lambda_max_star = np.sqrt(N)
print(f"Максимальное собственное значение звезды: λ_max = √{N} = {lambda_max_star:.6f}")
print(f"Параметр α = 0.5")
print(f"Условие существования: α < 1/λ_max")
print(f"1/λ_max = 1/√{N} = {1 / lambda_max_star:.6f}")

if 0.5 < 1 / lambda_max_star:
    print(f"\n✓ 0.5 < {1 / lambda_max_star:.6f} → центральность однозначно определена")
else:
    print(f"\n✗ 0.5 >= {1 / lambda_max_star:.6f} → центральность НЕ определена однозначно")

print(f"\nДля α=0.5 центральность однозначно определена при √N < 2")
print(f"√N < 2  →  N < 4")
print(f"При N=3: 3 < 4 → ДА")
print(f"При N=4: 4 = 4 → НЕТ (матрица вырождена)")
print(f"При N=5: 5 > 4 → НЕТ")