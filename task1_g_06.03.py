import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

A = [
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0]
]

n = len(A)
print("СЕТЬ 2")
print("Матрица смежности:")
for row in A:
    print(row)
print()

# Создание и визуализация графа
G = nx.DiGraph()
G.add_nodes_from(range(n))

for i in range(n):
    for j in range(n):
        if A[i][j] == 1:
            G.add_edge(i, j)

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=600)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
nx.draw_networkx_edges(G, pos, edge_color='gray',
                       arrows=True, arrowsize=20,
                       connectionstyle='arc3,rad=0.1',
                       min_source_margin=15, min_target_margin=15)

plt.title("Ориентированная сеть G")
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
for i in range(n):
    print(f"c{i + 1} = {eigenvector_norm[i]:.4f}")
print(f"Сумма = {np.sum(eigenvector_norm):.4f}")

# 2. PAGERANK
print("\n2. PAGERANK")


def pagerank(g, alpha, epsilon=1e-10, max_iter=1000):
    n = len(g)
    d_out = [sum(row) for row in g]

    # Обработка висячих узлов (с нулевой исходящей степенью)
    for i in range(n):
        if d_out[i] == 0:
            d_out[i] = n

    c = [1.0 / n] * n

    print(f"\nИсходящие степени: d_out = {d_out}")

    for t in range(max_iter):
        c_new = [0] * n
        for i in range(n):
            s = (1 - alpha) / n

            for j in range(n):
                if g[j][i] == 1:
                    s += alpha * (1.0 / d_out[j]) * c[j]
            c_new[i] = s

        diff = max(abs(c_new[i] - c[i]) for i in range(n))
        if t < 3 or t % 10 == 0:
            print(f"Итерация {t + 1:2d}: c = {[round(x, 6) for x in c_new]}")

        if diff < epsilon:
            print(f"Сходимость достигнута на итерации {t + 1}")
            return c_new

        c = c_new

    return c

# Расчет для α = 0.25
pr25 = pagerank(A, 0.25)
print(f"\nРЕЗУЛЬТАТ для α=0.25:")
for i in range(n):
    print(f"c{i + 1} = {pr25[i]:.6f}")
print(f"Сумма = {sum(pr25):.6f}")

# Расчет для α = 0.50
pr50 = pagerank(A, 0.50)
print(f"\nРЕЗУЛЬТАТ для α=0.50:")
for i in range(n):
    print(f"c{i + 1} = {pr50[i]:.6f}")
print(f"Сумма = {sum(pr50):.6f}")

# Расчет для α = 0.85
pr85 = pagerank(A, 0.85)
print(f"\nРЕЗУЛЬТАТ для α=0.85:")
for i in range(n):
    print(f"c{i + 1} = {pr85[i]:.6f}")
print(f"Сумма = {sum(pr85):.6f}")

# Анализ изменения
print("\nИЗМЕНЕНИЕ ПРИ УВЕЛИЧЕНИИ α:")
for i in range(n):
    print(f"Узел {i + 1}: {pr25[i]:.4f} → {pr50[i]:.4f}  (изменение: {pr50[i] - pr25[i]:+.4f})")
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
    for i in range(n):
        print(f"c{i + 1} = {katz[i]:.6f}")

    katz_norm = katz / np.sum(katz)
    print(f"\nНормированная:")
    for i in range(n):
        print(f"c{i + 1} = {katz_norm[i]:.6f}")

    print(f"\nСумма нормированных = {np.sum(katz_norm):.6f}")

# Расчет для α = 0.25
print(f"Параметры: α = 0.25, β = 1.0")
katz = katz_bonacich(A, 0.25, 1.0)

if katz is not None:
    print(f"\nЦентральность Каца-Боначича:")
    for i in range(n):
        print(f"c{i + 1} = {katz[i]:.6f}")

    katz_norm = katz / np.sum(katz)
    print(f"\nНормированная:")
    for i in range(n):
        print(f"c{i + 1} = {katz_norm[i]:.6f}")

    print(f"\nСумма нормированных = {np.sum(katz_norm):.6f}")

#Находим верхнюю границу альфа
print("\nДля однозначной определенности центральности Каца-Боначича необходимо, чтобы существовала обратная матрица g")
print("Для существования обратной матрицы необходимо, чтобы det(I - αgᵀ) ≠ 0")
print("Это условие выполняется, когда α ≠ 1/λ, где λ - собственные числа матрицы gᵀ.")
spectral_radius = np.max(np.abs(eigenvalues))
print(f"\nСпектральный радиус ρ(Gᵀ) = {spectral_radius:.6f}")

# Верхняя граница для α
alpha_max = 1.0 / spectral_radius
print(f"ВЕРХНЯЯ ГРАНИЦА α < 1/ρ(Gᵀ) = {alpha_max:.6f}")

print(f"ЦЕНТРАЛЬНОСТЬ КАЦА ОДНОЗНАЧНО ОПРЕДЕЛЕНА ПРИ: α < {alpha_max:.6f}")
