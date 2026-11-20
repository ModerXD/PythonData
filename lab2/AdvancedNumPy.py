import numpy as np
# 6
# 6.1
arr = np.arange(20)
print("Исходное:")
print(arr)
print("Каждый третий от первого:")
print(arr[1::3])
print("От 5 до 15 (не включая 15):")
print(arr[5:15])
print("Последние 5 наоборот:")
print(arr[:-6:-1])
print()

# 6.2
A = np.random.randint(0, 10, (6, 6))
print("Исходное:")
print(A)
print("Подматрица:")
print(A[1:4, 0:3])
print("Главная диагональ:")
print(np.diagonal(A))

print("Шахматы:")
rows, cols = np.indices(A.shape)
print(A[(rows + cols) % 2 == 0])
print()

# 6.3
a = np.array([10, 20, 30, 40, 50, 60])
m = np.arange(12).reshape(3, 4)

print("Исходное:")
print(a)
print("Исходное:")
print(m)
print("Элементы 0, 2, 4:")
print(a[[0, 2, 4]])
print("Элементы (0, 1), (1, 3), (2, 0):")
print(m[[0, 1, 2],  [1, 3, 0]])
print("Блок матрицы:")
print(m[0:2, 1:3])
print()

# 6.4
x = np.random.randn(1000)
print("Элементов больше 1:")
print(np.sum(x > 1))

# замена элементов меньше нуля на 0
x[x < 0] = 0

# замена элементов с соседями разных знаков на 0
mask = (
    ((x[:-2] > 0) & (x[2:] < 0)) |
    ((x[:-2] < 0) & (x[2:] > 0))
)
x[1:-1][mask] = 0

# 7
# 7.1
N, M = 50000, 200
fname = 'big.dat'
chunk = 5000

mm = np.memmap(fname, dtype=np.float32, mode='w+', shape=(N, M))
for i in range(0, N, chunk):
    mm[i:i + chunk] = np.random.rand(chunk, M)
mm.flush()
del mm

import os
size = os.path.getsize(fname)
print(f'Размер файла: {size} байт, ожидалось: {N * M * 4}')
print(f'Совпадает: {size == N * M * 4}')
print()

# 7.2
mm = np.memmap(fname, dtype=np.float32, mode='r', shape=(N, M))

g_min = np.inf
g_max = -np.inf
total = 0.0

for i in range(0, N, chunk):
    part = mm[i:i + chunk]
    g_min = min(g_min, part.min())
    g_max = max(g_max, part.max())
    total += part.sum(dtype=np.float64)

g_mean = total / (N * M)
print(f'min = {g_min}, max = {g_max}, mean = {g_mean}')
print()

# 7.3
def sum_by_rows(arr):
    s = 0.0
    n, m = arr.shape
    for i in range(n):
        for j in range(m):
            s += arr[i, j]
    return s

def sum_by_cols(arr):
    s = 0.0
    n, m = arr.shape
    for j in range(m):
        for i in range(n):
            s += arr[i, j]
    return s

C = np.random.rand(3000, 3000)
F = np.asfortranarray(C)

import time
for name, arr in (('C-order', C), ('F-order', F)):
    t0 = time.perf_counter()
    sum_by_rows(arr)
    t_rows = time.perf_counter() - t0

    t0 = time.perf_counter()
    sum_by_cols(arr)
    t_cols = time.perf_counter() - t0

    print(f'{name}: по строкам {t_rows:.2f} с, по столбцам {t_cols:.2f} с')
print()

# 7.4
n = 10**7
a = np.random.rand(n)
b = np.random.rand(n)

# способ 1: python-цикл с list
la, lb = a.tolist(), b.tolist()

def py_loop():
    res = []
    for x, y in zip(la, lb):
        res.append((x + y) / (x - y + 1))
    return res

# способ 2: numpy в одну строку (создаёт 3 временных массива)
def np_oneline():
    return (a + b) / (a - b + 1)

# способ 3: numpy с out= (буферы выделяем один раз)
buf1 = np.empty_like(a)
buf2 = np.empty_like(a)

def np_out():
    np.add(a, b, out=buf1) # a + b
    np.subtract(a, b, out=buf2) # a - b
    np.add(buf2, 1, out=buf2) # a - b + 1
    np.divide(buf1, buf2, out=buf1)
    return buf1

print(f'Результаты совпадают: {np.allclose(np_oneline(), np_out())}')

from timeit import timeit as ti
t_py = ti(py_loop, number=1)
t_np = ti(np_oneline, number=10) / 10
t_out = ti(np_out, number=10) / 10

print(f'Python-цикл:  {t_py:.3f} с')
print(f'NumPy 1 строка: {t_np:.4f} с')
print(f'NumPy out=:   {t_out:.4f} с')
print(f'Python / NumPy: {t_py / t_np:.0f}x, NumPy / out=: {t_np / t_out:.1f}x')

# 8
# 8.1
from scipy import sparse
size = 5000
k = 10000
row = np.random.randint(0, size, k)
col = np.random.randint(0, size, k)
data = np.random.rand(k) + 1

# специально делаем дубликаты индексов
row[1], col[1] = row[0], col[0]
row[3], col[3] = row[2], col[2]

coo = sparse.coo_matrix((data, (row, col)), shape=(size, size))
csr = coo.tocsr()

unique = np.unique(row * size + col).size
print(f'COO nnz (с дубликатами): {coo.nnz}')
print(f'CSR nnz: {csr.nnz}')
print(f'Уникальных пар индексов: {unique}')
print(f'nnz корректен: {csr.nnz == unique}')
print(f'Сумма элементов не изменилась: {np.isclose(coo.sum(), csr.sum())}')
print()

# 8.2
n = 4000
S = sparse.random(n, n, density=0.005, format='csr', dtype=np.float64)
D = S.toarray()
x = np.random.rand(n)

mem_dense = D.nbytes
mem_csr = S.data.nbytes + S.indices.nbytes + S.indptr.nbytes
print(f'Dense: {mem_dense / 1e6:.1f} МБ, CSR: {mem_csr / 1e6:.2f} МБ '
      f'(в {mem_dense / mem_csr:.0f} раз меньше)')

t_dense = ti(lambda: D @ x, number=50) / 50
t_csr = ti(lambda: S @ x, number=50) / 50
print(f'Dense: {t_dense:.2e} с, CSR: {t_csr:.2e} с (ускорение {t_dense / t_csr:.0f}x)')
print()

# 8.3
n = 2000
densities = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
x = np.random.rand(n)

t_sp, t_dn = [], []
print(f'{"density":>8} | {"sparse, с":>10} | {"dense, с":>10} | быстрее')
for d in densities:
    S = sparse.random(n, n, density=d, format='csr', dtype=np.float64)
    D = S.toarray()
    ts = ti(lambda: S @ x, number=30) / 30
    td = ti(lambda: D @ x, number=30) / 30
    t_sp.append(ts)
    t_dn.append(td)
    print(f'{d:>8} | {ts:>10.2e} | {td:>10.2e} | {"sparse" if ts < td else "dense"}')

# первая плотность, на которой sparse уже не выигрывает
crossover = next((d for d, ts, td in zip(densities, t_sp, t_dn) if ts >= td), None)
print(f'\nSparse перестаёт выигрывать при density = {crossover}')

import matplotlib.pyplot as plt
plt.plot(densities, t_sp, 'o-', label='CSR')
plt.plot(densities, t_dn, 's-', label='Dense')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Плотность')
plt.ylabel('Время A @ x, с')
plt.legend()
plt.grid(True)
plt.show()