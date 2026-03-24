import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

A = [
    [0, 1, 1],
    [1, 0, 1],
    [1, 0, 0]
]

n = len(A)
print("СЕТЬ А")
print("Матрица смежности:")
for row in A:
    print(row)
print()

G = nx.DiGraph()
G.add_nodes_from(range(n))

for i in range(n):
    for j in range(n):
        if A[i][j] == 1:
            G.add_edge(i, j)

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)

nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
nx.draw_networkx_edges(G, pos, edge_color='gray',
                       arrows=True, arrowsize=20,
                       connectionstyle='arc3,rad=0.1',
                       min_source_margin=15, min_target_margin=15)

plt.title("Ориентированная сеть А")
plt.axis('off')
plt.tight_layout()
plt.show()

# 1. ЦЕНТРАЛЬНОСТЬ ПО СОБСТВЕННОМУ ВЕКТОРУ
print("\n1. ЦЕНТРАЛЬНОСТЬ ПО СОБСТВЕННОМУ ВЕКТОРУ")

AT = np.array(A).T

eigenvalues, eigenvectors = np.linalg.eig(AT)

lambda_max = np.max(eigenvalues.real)
print(f"\nλ_max = {lambda_max:.4f}")

idx = np.argmax(eigenvalues.real)
eigenvector = eigenvectors[:, idx].real
eigenvector_norm = eigenvector / np.sum(eigenvector)

print("\nЦентральность по собственному вектору (нормировано на Σ=1):")
print(f"c₁ = {eigenvector_norm[0]:.4f}")
print(f"c₂ = {eigenvector_norm[1]:.4f}")
print(f"c₃ = {eigenvector_norm[2]:.4f}")
print(f"Сумма = {np.sum(eigenvector_norm):.4f}")

# Проверка через NetworkX

print("\nПроверка через NetworkX:")
eig_nx = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
total_nx = sum(eig_nx.values())
for node in sorted(eig_nx.keys()):
    print(f"  c{node + 1} = {eig_nx[node] / total_nx:.4f}")
print()

# 2. PAGERANK

print("\n2. PAGERANK")

def pagerank(g, alpha, epsilon=1e-10, max_iter=1000):
    n = len(g)
    d_out = [sum(row) for row in g]
    dangling_nodes = [i for i in range(n) if d_out[i] == 0]

    c = [1.0 / n] * n

    for t in range(max_iter):
        c_new = [0] * n
        dangling_sum = sum(c[i] for i in dangling_nodes)

        for i in range(n):
            s = (1 - alpha) / n
            s += alpha * dangling_sum / n

            for j in range(n):
                if g[j][i] == 1:
                    s += alpha * (1.0 / d_out[j]) * c[j]

            c_new[i] = s

        # Нормализация
        total = sum(c_new)
        c_new = [x / total for x in c_new]

        diff = max(abs(c_new[i] - c[i]) for i in range(n))
        if diff < epsilon:
            return c_new
        c = c_new
    return c

# Расчет для α = 0.25
pr25 = pagerank(A, 0.25)
print(f"\nРЕЗУЛЬТАТ для α=0.25:")
print(f"c₁ = {pr25[0]:.6f}")
print(f"c₂ = {pr25[1]:.6f}")
print(f"c₃ = {pr25[2]:.6f}")
print(f"Сумма = {sum(pr25):.6f}")

# Расчет для α = 0.50
pr50 = pagerank(A, 0.50)
print(f"\nРЕЗУЛЬТАТ для α=0.50:")
print(f"c₁ = {pr50[0]:.6f}")
print(f"c₂ = {pr50[1]:.6f}")
print(f"c₃ = {pr50[2]:.6f}")
print(f"Сумма = {sum(pr50):.6f}")


# Анализ изменения
print("\nИЗМЕНЕНИЕ ПРИ УВЕЛИЧЕНИИ α:")
print(f"Узел 1: {pr25[0]:.4f} → {pr50[0]:.4f}  (изменение: {pr50[0] - pr25[0]:+.4f})")
print(f"Узел 2: {pr25[1]:.4f} → {pr50[1]:.4f}  (изменение: {pr50[1] - pr25[1]:+.4f})")
print(f"Узел 3: {pr25[2]:.4f} → {pr50[2]:.4f}  (изменение: {pr50[2] - pr25[2]:+.4f})")
print()

# 3. ЦЕНТРАЛЬНОСТЬ КАЦА-БОНАЧИЧА

print("\n3. ЦЕНТРАЛЬНОСТЬ КАЦА-БОНАЧИЧА")

def katz_bonacich(g, alpha, beta=1.0):
    n = len(g)

    gT = np.array(g).T
    I = np.eye(n)
    M = I - alpha * gT
    det = np.linalg.det(M)

    if abs(det) < 1e-10:
        print("Центральность не определена однозначно.")
        return None

    ones = np.ones(n)
    c = beta * np.linalg.solve(M, ones)

    return c

# Расчет для α = 0.5
print(f"Параметры: α = 0.5, β = 1.0")
katz = katz_bonacich(A, 0.5, 1.0)

if katz is not None:
    print(f"\nЦентральность Каца-Боначича:")
    print(f"c₁ = {katz[0]:.6f}")
    print(f"c₂ = {katz[1]:.6f}")
    print(f"c₃ = {katz[2]:.6f}")

    katz_norm = katz / np.sum(katz)
    print(f"\nНормированная (Σ=1):")
    print(f"c₁ = {katz_norm[0]:.6f}")
    print(f"c₂ = {katz_norm[1]:.6f}")
    print(f"c₃ = {katz_norm[2]:.6f}")