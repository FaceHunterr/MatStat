import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt

#метод наименьших квадратов
def LeastSquareMethod(x, y):
    beta_l = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    beta_0 = np.mean(y) - beta_l * np.mean(x)
    return beta_0, beta_l

def LeastModulusMethod(x, y, start):
    min_fun = lambda beta: np.sum(np.abs(y - beta[0] - beta[1] * x))
    result = opt.minimize(min_fun, start)
    beta_0 = result['x'][0]
    beta_1 = result['x'][1]
    return beta_0, beta_1

def getCoefs(x, y):
    beta_0_s, beta_l_s = LeastSquareMethod(x, y)
    beta_0_m, beta_1_m = LeastModulusMethod(x, y, np.array([beta_0_s, beta_l_s]))
    return beta_0_s, beta_l_s, beta_0_m, beta_1_m

def regrPlots(x, y, type, coefs):
    beta_0_s, beta_l_s, beta_0_m, beta_1_m = coefs
    plt.plot(x, x * (2 * np.ones(len(x))) + 2 * np.ones(len(x)), label='Модель', color='red')
    plt.plot(x, x * (beta_l_s * np.ones(len(x))) + beta_0_s * np.ones(len(x)), label='МНК', color='black')
    plt.plot(x, x * (beta_1_m * np.ones(len(x))) + beta_0_m * np.ones(len(x)), label='МНМ', color='blue')
    plt.scatter(x, y, label="Выборка", facecolors='none', edgecolors='black')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([-1.8, 2])
    plt.legend()
    plt.savefig(type + '.png', format='png')
    plt.close()

def criteriaComparison(x, coefs):
    beta_0_s, beta_l_s, beta_0_m, beta_1_m = coefs
    model = lambda x: 2 + 2 * x
    lsc = lambda x: beta_0_s + beta_l_s * x
    lmc = lambda x: beta_0_m + beta_1_m * x
    sum_ls, sum_lm = 0, 0
    for point in x:
        y_ls = lsc(point)
        y_lm = lmc(point)
        y_model = model(point)
        sum_ls += pow(y_model - y_ls, 2)
        sum_lm += pow(y_model - y_lm, 2)
    print('sum_ls =', sum_ls, " < ", 'sum_lm =', sum_lm) if sum_ls < sum_lm \
        else print('sum_lm =', sum_lm, " < ", 'sum_ls =', sum_ls)

x = np.linspace(-1.8, 2, 20)
y = 2 + 2 * x + stats.norm(0, 1).rvs(20)
for type in ['Without_perturbations', 'With_perturbations']:
    coefs = getCoefs(x, y)
    beta_0_s, beta_l_s, beta_0_m, beta_1_m = coefs
    print()
    print(type)
    print("МНК:")
    print('beta_0_s = ' + str(np.around(beta_0_s, decimals=2)))
    print('beta_l_s = ' + str(np.around(beta_l_s, decimals=2)))
    print()
    print("МНМ:")
    print('alpha_lm = ' + str(np.around(beta_0_m, decimals=2)))
    print('beta_lm = ' + str(np.around(beta_1_m, decimals=2)))
    print()
    regrPlots(x, y, type, coefs)
    criteriaComparison(x, coefs)
    y[0] += 10
    y[-1] -= 10