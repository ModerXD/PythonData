import pandas as pd

# 1

# 1.1
print('Задание 1.1')
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(s)

# 1.2
print('\nЗадание 1.2')
slovar = {
    'num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'numButWritten': ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'],
    'addedRow': [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1],
}

df = pd.DataFrame(slovar)
print(df)

# 1.3
print('\nЗадание 1.3')
print(f'Первые строки:\n{df.head()}\n')
print(f'Последние строки:\n{df.tail()}\n')

print('Информация:')
df.info()

print(f'\nСтатистика:\n{df.describe()}')

# 2
import numpy as np
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry'],
    'department': ['IT', 'HR', 'IT', 'Finance', 'HR', 'IT', 'Finance', 'HR'],
    'salary': [80000, 60000, 90000, 75000, np.nan, 85000, 70000, 62000],
    'bonus': [5000, 3000, np.nan, 4000, 2000, 6000, 3500, np.nan],
    'hire_date': pd.to_datetime(['2020-01-15', '2019-03-22', '2021-07-01', '2018-11-05',
                                 '2022-02-10', '2020-09-30', '2019-12-12', '2021-05-20'])
}

dataFrame = pd.DataFrame(data)

# 2.1
print('\nЗадание 2.1')
print(dataFrame[dataFrame['department'] == 'IT'][['name', 'salary']])

# 2.2
print('\nЗадание 2.2')
print(f'Строки:\n{dataFrame.iloc[0:2]}')
print(f'Столбцы:\n{dataFrame.iloc[:, 0:1]}')

# 2.3
print('\nЗадание 2.3')
print(dataFrame[dataFrame['salary'] > 70000][['name']])

# 2.4
print('\nЗадание 2.4')
dataFrame['total_comp'] = dataFrame['salary'].fillna(0) + dataFrame['bonus'].fillna(0)
print(dataFrame[['name', 'total_comp']])

# 2.5
print('\nЗадание 2.5')
del dataFrame['hire_date']
print(dataFrame)


# 3

# 3.1
print('\nЗадание 3.1')
print('Количество пропусков в каждом столбце:')
print(dataFrame.isna().sum())

# 3.2
print('\nЗадание 3.2')
dataFrame['salary'] = dataFrame['salary'].fillna(
    dataFrame.groupby('department')['salary'].transform('mean')
)
print(dataFrame[['name', 'department', 'salary']])

# 3.3
print('\nЗадание 3.3')
dataFrame['bonus'] = dataFrame['bonus'].fillna(0)
print(dataFrame[['name', 'bonus']])

# 3.4
print('\nЗадание 3.4')
dataFrame = dataFrame.dropna(subset=['salary'])
print(dataFrame)


# 4

# 4.1
print('\nЗадание 4.1')
print('Средняя зарплата по отделам:')
print(dataFrame.groupby('department')['salary'].mean())

# 4.2
print('\nЗадание 4.2')
print(dataFrame.groupby('department').agg({
    'salary': 'mean',
    'bonus': 'max',
    'name': 'count'
}).rename(columns={
    'salary': 'Средняя зарплата',
    'bonus': 'Максимальный бонус',
    'name': 'Количество сотрудников'
}))

# 4.3
print('\nЗадание 4.3')
pivot = dataFrame.pivot_table(
    index='department',
    values='salary',
    aggfunc='mean'
)
print(pivot)

# 4.4
print('\nЗадание 4.4')
print(dataFrame.groupby('department')['salary'].mean().sort_values(ascending=False))


# 5

# 5.1
print('\nЗадание 5.1')
managers = pd.DataFrame({
    'department': ['IT', 'HR', 'Finance'],
    'manager': ['Alex', 'Maria', 'John']
})
print(managers)

# 5.2
print('\nЗадание 5.2')
dataFrame = dataFrame.merge(managers, on='department', how='left')
print(dataFrame)

# 5.3
print('\nЗадание 5.3')
new_employee = pd.DataFrame([{
    'name': 'Ivan',
    'department': 'IT',
    'salary': 95000,
    'bonus': 7000,
    'total_comp': 102000,
    'manager': 'Alex'
}])

dataFrame = pd.concat([dataFrame, new_employee], ignore_index=True)
print(dataFrame)

# 5.4
print('\nЗадание 5.4')
dataFrame = dataFrame.reset_index(drop=True)
print(dataFrame)


# 6

# 6.1
print('\nЗадание 6.1')
print('Топ-3 сотрудников по общей компенсации:')
print(dataFrame.nlargest(3, 'total_comp')[['name', 'department', 'total_comp']])

# 6.2
print('\nЗадание 6.2')
print(dataFrame.groupby('department')[['salary', 'bonus']].mean())

# 6.3
print('\nЗадание 6.3')
dataFrame['salary_above_75000'] = dataFrame['salary'] > 75000

result = dataFrame.groupby('department')['salary_above_75000'].mean() * 100

print('Доля сотрудников с зарплатой выше 75000 (%):')
print(result)
