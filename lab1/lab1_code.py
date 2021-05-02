from scipy.stats import norm, cauchy, laplace, poisson, uniform
import numpy as np
import matplotlib.pyplot as plt
import math

sizes = [10, 50, 1000]
bins = [10, 16, 30]

def normalDistribution():
    for size in sizes:
        fig, ax = plt.subplots(1, 1)
        ax.hist(norm.rvs(size=size), bins=int(10 * math.log10(size)), range=(-3, 3), density=True, color="brown")
        x = np.linspace(norm.ppf(0.001), norm.ppf(0.999), 500)
        ax.plot(x, norm.pdf(x), "g")
        plt.grid()
        plt.ylabel("density")
        plt.title("Normal distribution, n = " + str(size))
    plt.show()

def cauchyDistribution():
    for size in sizes:
        fig, ax = plt.subplots(1, 1)
        ax.hist(cauchy.rvs(size=size), bins=int(10 * math.log10(size)), range=(-18, 18), density=True, color="brown")
        x = np.linspace(cauchy.ppf(0.02), cauchy.ppf(0.98), 500)
        ax.plot(x, cauchy.pdf(x), "g")
        plt.grid()
        plt.title("Cauchy distribution, n = " + str(size))
        plt.ylabel("density")
    plt.show()

def laplaceDistribution():
    scale = 1/math.sqrt(2)
    for size in sizes:
        fig, ax = plt.subplots(1, 1)
        ax.hist(laplace.rvs(size=size, scale=scale), bins=int(10 * math.log10(size)), range=(-5, 5), density=True, color="brown")
        x = np.linspace(laplace.ppf(0.001, scale=scale), laplace.ppf(0.999, scale=scale), 500)
        ax.plot(x, laplace.pdf(x, scale=scale), "g")
        plt.grid()
        plt.title("Laplace distribution, n = " + str(size))
        plt.ylabel("density")
    plt.show()

def poissonDistribution():
    mu = 10
    for size in sizes:
        fig, ax = plt.subplots(1, 1)
        ax.hist(poisson.rvs(mu=mu, size=size), color='brown', density=True)
        x = np.arange(poisson.ppf(0.002, mu=mu), poisson.ppf(0.998, mu=mu))
        ax.plot(x, poisson.pmf(x, mu=mu), '-')
        plt.grid()
        plt.title("Poisson distribution, n = " + str(size))
        plt.ylabel("density")
    plt.show()
    return

def uniformDistribution():
    coef = math.sqrt(3)
    for size in sizes:
        fig, ax = plt.subplots(1, 1)
        ax.hist(uniform.rvs(loc=-coef, size=size, scale= 2 * coef), bins=int(10 * math.log10(size)), density=True, color="brown")
        x = np.linspace(-coef, coef)
        ax.plot(x, uniform.pdf(x, loc=-coef, scale=(2 * coef)), "g")
        plt.grid()
        plt.title("Uniform distribution, n = " + str(size))
        plt.ylabel("density")
    plt.show()



normalDistribution()
cauchyDistribution()
laplaceDistribution()
poissonDistribution()
uniformDistribution()
