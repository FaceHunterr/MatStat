import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, pearsonr, spearmanr
from statistics import variance
import matplotlib.transforms as transforms
import pandas

SAMPLE = [20, 60, 100]
RHOS = [0, 0.5, 0.9]
MEAN = [0, 0]
table_rows = ['$E(z)$', '$E(z^2)$', '$D(z)$']
table_captures =  ['Двумерное нормальное распределение, $n=' + str(sample) + '$' for sample in SAMPLE]
table_captures_2 = ['Смесь нормальных распределений, $n=' + str(sample) + '$' for sample in SAMPLE]
table_columns = ['$r$', '$r_s$', '$r_Q$']

def get_rvs(num, cov, mixed):
    return multivariate_normal.rvs(MEAN, cov, num) if not mixed \
        else 0.9 * multivariate_normal.rvs(MEAN, [[1, 0.9], [0.9, 1]], num) \
             + 0.1 * multivariate_normal.rvs(MEAN, [[10, -0.9], [-0.9, 10]], num)

def quadrant_coef(x, y):
    med_x = np.median(x)
    med_y = np.median(y)
    return np.mean(np.sign(x - med_x) * np.sign(y - med_y))

def accumulate_coefs(num, cov, mixed):
    pearson_coefs = []
    spearman_coefs = []
    quadrant_coefs = []
    for _ in range(1000):
        sample = get_rvs(num, cov, mixed)
        x, y = sample[:, 0], sample[:, 1]
        pearson_coefs.append(pearsonr(x, y)[0])
        spearman_coefs.append(spearmanr(x, y)[0])
        quadrant_coefs.append(quadrant_coef(x, y))
    return pearson_coefs, spearman_coefs, quadrant_coefs

def build_table(num, cov, mixed):
    pearson, spearman, quadrant = accumulate_coefs(num, cov, mixed)
    p = np.median(pearson)
    s = np.median(spearman)
    q = np.median(quadrant)
    table = []

    table.append([np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])
    p = np.median([pow(pearson[k], 2) for k in range(1000)])
    s = np.median([pow(spearman[k], 2) for k in range(1000)])
    q = np.median([pow(quadrant[k], 2) for k in range(1000)])
    table.append([np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])
    p = variance(pearson)
    s = variance(spearman)
    q = variance(quadrant)
    table.append([np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    return table

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_ellipses(samples):
    num = len(samples[0])
    fig, ax = plt.subplots(1, len(samples))
    fig.suptitle("n = " + str(num))
    titles = ['$ \\rho = 0$', '$\\rho = 0.5 $', '$ \\rho = 0.9$']
    i = 0
    for sample in samples:
        x = sample[:, 0]
        y = sample[:, 1]
        ax[i].scatter(x, y, c='black', s=3)
        confidence_ellipse(x, y, ax[i], edgecolor='black')
        ax[i].scatter(np.mean(x), np.mean(y), c='black', s=3)
        ax[i].set_title(titles[i])
        i += 1
    plt.savefig(
        "Ellipse n = " + str(num) + ".png",
        format='png'
    )


# file = open("D:\\Progs\\LaTeX\\МатСтат_лаб5\\table1.tex", "w", encoding='utf-8')
#
# for sample_num in range(len(SAMPLE)):
#     for rho_ind in range(len(RHOS)):
#         cov = [[1.0, RHOS[rho_ind]], [RHOS[rho_ind], 1.0]]
#         table = build_table(SAMPLE[sample_num], cov, False)
#         df = pandas.DataFrame(table, index=table_rows, columns=table_columns).to_latex(escape=False,
#                                     caption=table_captures[sample_num] + ", $\\rho=" + str(RHOS[rho_ind]) +"$",
#                                                                 column_format="|c|c|c|c|",
#                                                                 position="H")
#         file.write(df)
#
#
# for sample_num in range(len(SAMPLE)):
#     table_mixed = build_table(SAMPLE[sample_num], None, True)
#     df = pandas.DataFrame(table_mixed, index=table_rows, columns=table_columns).to_latex(escape=False,
#                                                                                    caption=table_captures_2[
#                                                                                                sample_num],
#                                                                                    column_format="|c|c|c|c|",
#                                                                                    position="H")
#     file.write(df)

samples = []
for num in SAMPLE:
    for rho in RHOS:
        samples.append(get_rvs(num, [[1.0, rho], [rho, 1.0]], False))
    plot_ellipses(samples)
    samples = []

