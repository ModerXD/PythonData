import numpy as np

# Задача 1
# 1.1
arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
npsum = np.sum(arr)
arrsum = arr.sum()

print(f'Разница между суммами = {abs(arrsum-npsum)}') # равна 0

# 1.2
print(f'\nМашинный эпсилон float32 = {np.finfo(np.float32).eps}')
print(f'Машинный эпсилон float64 = {np.finfo(np.float64).eps}')

#1.3
one = np.float32(1.0)
less = np.float32(1e-8) # меньше, чем машинный эпсилон float32
result = one + less - one

print(f'\nfloat32: 1 + 1e-8 - 1 = {result}')
# результят 0, хотя должен быть равен less

# Задача 2
# 2.1
A = np.arange(-10, 10).reshape((4, 5))
print(f'Матрица:\n{A}')
print(f'Строка:\n{A[1]}')
print(f'Стобец:\n{A[:,3]}')
print(f'Подмассив:\n{A[1:3, 2:4]}')

# 2.2
print(f'Fancy-индексация:\n{A[np.ix_([0, 2, 3], [1, 4])]}')

# 2.3
A[A < 0] = 0
print(f'Матрица после обработки с логиечской индексацией:\n{A}')

# 2.4
B = np.arange(4).reshape((4, 1))
C = np.arange(5).reshape((1, 5))
print(f'Сложение без цикла:\n{B + C}')

# Задача 3
# 3.1
shape = (1000, 1000)
mm = np.memmap('data.dat', dtype=np.float64, mode='w+', shape=shape)
mm[:] = np.random.rand(*shape)
mm.flush()
del mm

# 3.2
mm = np.memmap('data.dat', dtype=np.float64, mode='r', shape=shape)
print(f'Строка номер 10: {mm[10][:5]}...')
print(f'Столбец номер 10: {mm[:, 10][:5]}...')

# 3.3
def row_major(arr):
    s = 0.0
    for i in range(arr.shape[0]):
        s += arr[i].sum()
    return s

def column_major(arr):
    s = 0.0
    for i in range(arr.shape[1]):
        s += arr[:, i].sum()
    return s

from timeit import timeit as ti
print(f'Обход по строкам: {ti(lambda: row_major(mm), number=20) / 20}')
print(f'Обход по столбцам: {ti(lambda: column_major(mm), number=20) / 20}')
# обход по столбцам занимает дольше, потому что
# в памяти соседние столбцы лежат не так близко, как строки

# Задача 4
# 4.1
from scipy import sparse

S = sparse.random(5, 5, density=0.2, format='csr', dtype=np.float64)
print(f'\n{S}')

# 4.2
S_coo = S.tocoo()
S_csc = S.tocsc()
print(f'{S_coo}\n{S_csc}')

# 4.3
v = np.random.rand(5)
result = S @ v
print(f'\nУмножение на плотный вектор:\n{result}')

# 4.4
BigS = sparse.random(1000, 1000, density=0.01, format='csr', dtype=np.float64)
BigArr = BigS.toarray()
BigV = np.random.rand(1000)

timeSparse = ti(lambda: BigS @ BigV, number=200) / 200
timeDense = ti(lambda: BigArr @ BigV, number=200) / 200

print(f'\nРазреженная: {timeSparse}')
print(f'Плотная: {timeDense}')
print(f'Разница во времени: {abs(timeSparse - timeDense)}')
print(f'Ускорение: {timeDense / timeSparse} раз')

# Задача 5
# 5.1
def squareSum(arr):
    s = 0.0
    for i in arr:
        s += i * i
    return s

from sumsq import sum_squares_cy

n = 10**7
testArr = np.random.rand(n)
lst = testArr.tolist()

pyTime = ti(lambda: squareSum(lst), number= 3) / 3
cyTime = ti(lambda: sum_squares_cy(testArr), number= 3) / 3

# 5.3
print(f'\nВремя выполнения на python: {pyTime} секунд')
print(f'Время выполнения на cython: {cyTime} секунд')
print(f'Ускорение: {pyTime / cyTime} раз')