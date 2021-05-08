import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform
import pandas

scale = 1/math.sqrt(2)
sizes = [10, 100, 1000]
mu = 10
coef = math.sqrt(3)
num = 1000

def mean(sample):
    return sum(sample)/len(sample)

def median(sample):
    size = len(sample)
    return (sample[int((size + 1)/2) - 1] + sample[-int((size + 1)/2)]) / 2

def halfsum(sample):
    return (sample[0] + sample[-1]) / 2

def quart_halfsum(sample):
    size = len(sample)
    if size % 4 != 0:
        return (sample[int(size / 4)] + sample[int(size * 3 / 4)]) / 2
    else:
        return (sample[int(size / 4) - 1] + sample[int(3 * size/4) - 1]) / 2

def trunc_mean(sample):
    size = len(sample)
    r = int(size/4)
    s = 0
    for i in range(r, size-r):
        s += sample[i]
    return s / (size - r * 2)

distributions = [['Нормальное распределение', 'Распределение Коши', 'Распределение Лапласа',
                  'Распределение Пуассона', 'Равномерное распределение'],
                 [lambda size: norm.rvs(size=size), lambda size: cauchy.rvs(size=size),
                  lambda size: laplace.rvs(size=size, scale=scale), lambda size: poisson.rvs(mu=mu, size=size),
                  lambda size: uniform.rvs(loc=-coef, size=size, scale=2 * coef)]]

def theoretical_prob(sample):
    min = np.quantile(sample, 0.25) - 1.5 * (np.quantile(sample, 0.75) - np.quantile(sample, 0.25))
    max = np.quantile(sample, 0.75) + 1.5 * (np.quantile(sample, 0.75) - np.quantile(sample, 0.25))
    return min, max

def ejection_num(sample, min, max):
    ej_num = 0
    for elem in sample:
        if elem < min or elem > max:
            ej_num += 1
    return ej_num

for dist_num in range(len(distributions[1])):
    sample_20 = distributions[1][dist_num](20)
    sample_100 = distributions[1][dist_num](100)
    plt.figure()
    plt.boxplot((sample_20, sample_100), patch_artist=True, labels=[20, 100])
    plt.xlabel("n")
    plt.ylabel("x")
    plt.title(distributions[0][dist_num], fontweight="bold")
plt.show()

exit()

table = []
column_name = ["Выборка", "Доля выбросов"]




for dist_num in range(len(distributions[1])):
    ejection_20 = 0
    ejection_100 = 0
    for i in range(1000):
        sample_20 = distributions[1][dist_num](20)
        sample_100 = distributions[1][dist_num](100)
        min_20, max_20 = theoretical_prob(sample_20)
        min_100, max_100 = theoretical_prob(sample_100)
        ejection_20 += ejection_num(sample_20, min_20, max_20)
        ejection_100 += ejection_num(sample_100, min_100, max_100)

    table.append([(distributions[0][dist_num] + "$, n=" + str(20) + "$"), np.around(ejection_20 / 1000 / 20, decimals=2)])
    table.append([(distributions[0][dist_num] + "$, n=" + str(100) + "$"), np.around(ejection_100 / 1000 / 100, decimals=2)])

file = open("D:\\Progs\\LaTeX\\МатСтат_лаб3\\tables.tex", "w", encoding='utf-8')
df = pandas.DataFrame(table, columns=column_name).to_latex(escape=False,index=False, caption="Доля выбросов", column_format="|l|c|",
                                                                             position="H")
file.write(df)
