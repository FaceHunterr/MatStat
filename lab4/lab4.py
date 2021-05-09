import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform,gaussian_kde
import pandas
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF

scale = 1/math.sqrt(2)
sizes = [20, 60, 100]
mu = 10
coef = math.sqrt(3)

distributions = [['Нормальное распределение', 'Распределение Коши', 'Распределение Лапласа',
                  'Распределение Пуассона', 'Равномерное распределение'],
                 [lambda size: norm.rvs(size=size), lambda size: cauchy.rvs(size=size),
                  lambda size: laplace.rvs(size=size, scale=scale), lambda size: poisson.rvs(mu=mu, size=size),
                  lambda size: uniform.rvs(loc=-coef, size=size, scale=2 * coef)],
                 [lambda x: norm.cdf(x=x), lambda x: cauchy.cdf(x=x), lambda x: laplace.cdf(x=x, scale=scale),
                  lambda x:poisson.cdf(k=x, mu=mu), lambda x: uniform.cdf(x=x, loc=-coef, scale=(2 * coef))],
                 [lambda x:norm.pdf(x), lambda x:cauchy.pdf(x), lambda x:laplace.pdf(x, scale=scale),
                  lambda x:poisson.pmf(x, mu=mu), lambda x:uniform.pdf(x, loc=-coef, scale=(2 * coef))]
                 ]

factors = [0.5, 1, 2]

for i in range(len(distributions[0])):
    a, b, step = (6, 14, 1) if i == 3 else (-4, 4, 0.01)
    x_range = np.arange(a, b, step)
    samples = []
    for size in sizes:
        samples.append([elem for elem in distributions[1][i](size) if elem >= a or elem <= b])
    fig = plt.figure()
    for j in range(len(samples)):
        plt.subplot(1, 3, j+1)
        plt.title('n = ' + str(sizes[j]))
        plt.step(x_range, distributions[2][i](x_range), color='blue')
        array_ex = np.linspace(a, b)
        ecdf = ECDF(samples[j])
        y = ecdf(array_ex)
        plt.step(array_ex, y, color='black')
        plt.xlabel('x')
        plt.ylabel('F(x)')
        plt.subplots_adjust(wspace=0.5)
    plt.savefig("dis" + str(i) + '_CDF.png', format='png')

    for j in range(len(samples)):
        kern_names = ['$h = h_n/2$', '$h = h_n$', '$h = 2 * h_n$']
        fig, ax = plt.subplots(1, 3)
        plt.subplots_adjust(wspace=0.5)

        for k in range(len(factors)):
            kde = gaussian_kde(samples[j], bw_method='silverman')
            h_n = kde.factor
            fig.suptitle('n = ' + str(sizes[j]))
            ax[k].plot(x_range, distributions[3][i](x_range), color='black', alpha=0.5, label='pdf')
            ax[k].set_title(kern_names[k])
            sns.kdeplot(samples[j], ax=ax[k], bw=h_n * factors[k], label='kde', color='blue')
            ax[k].set_xlabel('x')
            ax[k].set_ylabel('f(x)')
            ax[k].set_ylim([0, 1])
            ax[k].set_xlim([a, b])
        plt.savefig("distr" + str(i) + '_KDE_n' + str(sizes[j]) + '.png', format='png')

plt.show()
