
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

import matplotlib as mpl
from random import seed, choices

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


#

with open('../mercier_zyp1/MLH1 counts mercier paper fig S2.csv', 'r') as f:
    n = np.array(list(map(int, next(f).split(',')[1:])))
    c = np.array(list(map(int, next(f).split(',')[1:])))

from scipy.stats import poisson

plt.figure()

data_all = []
for nn, cc in zip(n,c):
    data_all += [nn]*cc

print(n, c)
nn = np.arange(np.max(n)+5)
plt.bar(n,c/np.sum(c), color='r')
plt.xlabel('Number of MLH1 foci per cell')
plt.ylabel('Relative frequency')
plt.xlim(0, 30)
mean_c_o = np.sum(n*c)/np.sum(c)
mean_c = np.mean(data_all) #np.sum(n*c)/np.sum(c)

print(mean_c_o, mean_c)

plt.plot(nn, poisson(mean_c).pmf(nn), 'g-o')

plt.text(0.7, 0.9, f'$\mu={mean_c:.2f}$', transform=plt.gca().transAxes)

s_c2 = np.sum(n*n*c)
e_c = np.sum(n*c)/np.sum(c)
n_c = np.sum(c)
S2_c_o = (s_c2 - e_c**2*n_c)/n_c

S2_c = np.var(data_all, ddof=1)

print(S2_c_o, S2_c)

plt.text(0.7, 0.7, f'$S^2 = {S2_c:.2f}$', transform=plt.gca().transAxes)
plt.text(0.7, 0.5, f'$N = {n_c}$', transform=plt.gca().transAxes)

## bootstrap test for variance

d = []
seed(12345)
for i in range(10000):
    s = poisson.rvs(mean_c, size=len(data_all))  #.choices(data_all, k=len(data_all))
    d.append(np.var(s, ddof=1)/np.mean(s))

print(S2_c/mean_c, np.percentile(d, 2.5), np.percentile(d, 97.5))

plt.savefig('../output/data_output/mlh_data.svg')

#plt.show()

