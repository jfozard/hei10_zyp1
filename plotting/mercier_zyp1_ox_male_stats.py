
import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
from scipy.stats import chi2
from random import choices, seed

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rcParams.update({ 
    'figure.facecolor': 'none', 
    'figure.edgecolor': 'none',       
    'font.size': 20, 
    'figure.dpi': 72, 
    'figure.subplot.bottom' : .15, 
    'axes.labelsize':28,
    'savefig.edgecolor': 'none',
    'savefig.facecolor': 'none',
    
})

data = pd.read_csv('../mercier_zyp1/mercier zyp1 hei10 mlh1 data comb male hom.csv')
foci = data['MLH1 foci']
n= len(foci)
m = np.mean(foci)
v = np.var(foci, ddof=1)
CI_L = (n-1)*v/chi2.ppf(.95, n-1)
CI_U = (n-1)*v/chi2.ppf(.05, n-1)
print(n, m, v, CI_L, CI_U)

"""
# bootstrapped
seed(1234)
est = []
for i in range(10000):
    est.append(np.var(choices(foci, k=len(foci)), ddof=1))
print(np.percentile(est, 5), np.percentile(est, 95))
"""

n_c = len(foci)
n, c = np.unique(foci, return_counts=True)

nn = np.arange(81)
plt.bar(n,c/np.sum(c), color='r')
plt.xlabel('Number of MLH1 foci per cell')
plt.ylabel('Relative frequency')
plt.xlim(0, 80)
plt.ylim(0, 0.1)
mean_c = np.mean(foci) #np.sum(n*c)/np.sum(c)


plt.plot(nn, poisson(mean_c).pmf(nn), 'g-o')

plt.text(0.1, 0.9, f'$\mu={mean_c:.2f}$', transform=plt.gca().transAxes)


S2_c = np.var(foci, ddof=1)


plt.text(0.1, 0.7, f'$S^2 = {S2_c:.2f}$', transform=plt.gca().transAxes)
plt.text(0.1, 0.5, f'$N = {n_c}$', transform=plt.gca().transAxes)
plt.savefig('../output/data_output/mlh_data_ox.svg')

plt.show()

