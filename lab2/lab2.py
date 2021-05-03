from scipy.stats import norm, cauchy, laplace, poisson, uniform
import numpy as np
import math
import pandas
#from tabulate import tabulate

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
                  lambda size: uniform.rvs(loc=-coef, size=size, scale= 2 * coef)]]

characteristics = [["$\overline{x}$", "$med x$", "$z_R$",
                    "$z_Q$", "$z_{tr}$"],
                   [mean, median, halfsum, quart_halfsum, trunc_mean]]

rowNames = ["$E(z)$", "$D(z)$", "$E(z)+\sqrt{D(z)}$", "$E(z)-\sqrt{D(z)}$", "$\Hat{E}$"]

file = open("D:\\Progs\\LaTeX\\МатСтат_лаб2\\tables.tex", "w", encoding='utf-8')
file.write("\\section{Результаты}")

for k in range(len(distributions[1])):
    for size in sizes:
        table = [[0 for i in range(len(characteristics[1]))] for j in range(len(rowNames))]
        for i in range(num):
            sample = distributions[1][k](size)
            sample = np.sort(sample)
            for j in range(len(table[0])):
                table[0][j] += characteristics[1][j](sample)
            for j in range(len(table[1])):
                table[1][j] += characteristics[1][j](sample) ** 2
        for i in range(len(table)):
            for j in range(len(table[i])):
                table[i][j] /= num
        for j in range(len(table[1])):
            table[1][j] -= table[0][j] ** 2
        for j in range(len(table[1])):
            table[2][j] = table[0][j] + math.sqrt(table[1][j])
        for j in range(len(table[1])):
            table[3][j] = table[0][j] - math.sqrt(table[1][j])
        if k != 3:
            for j in range(len(table[1])):
                table[4][j] = ""
                num1 = table[0][j]
                num2 = table[2][j]
                num3 = table[3][j]
                if int(num1) != int(num2) or int(num1) != int(num3):
                    table[4][j] += "-"
                    continue
                table[4][j] += "0."
                while True:
                    num1 = (num1 - int(num1)) * 10
                    num2 = (num2 - int(num2)) * 10
                    num3 = (num1 - int(num3)) * 10
                    if int(num1) == int(num2) and int(num1) == int(num3):
                        table[4][j] += "0"
                    else:
                        break
        else:
            for j in range(len(table[1])):
                table[4][j] = ""
                num1 = table[0][j]
                num2 = table[2][j]
                num3 = table[3][j]
                d1 = int(num2) - int(num1)
                d2 = int(num1) - int(num3)
                if d1!=0:
                    d1 = "+" + str(d1)
                else:
                    d1 = "0"
                if d2 != 0:
                    d2 = "-" + str(d2)
                else:
                    d2 = "0"
                table[4][j] += "$" + str(int(num1)) +"^{" + d1 +"}_{" + d2 + "}$"
        for i in range(0, 4):
            for j in range(len(characteristics[0])):
                table[i][j] = np.around(table[i][j], decimals=6)

        df = pandas.DataFrame(table, rowNames, characteristics[0]).to_latex(caption=distributions[0][k] + ", $size=" +
                     str(size) + "$", escape=False, column_format="|l|" + "c|" *
                                                                            len(characteristics[0]),
                                                                            label=str(k) + "_" + str(size), position="h")
        file.write(df)

file.close()


