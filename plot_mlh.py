
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

import matplotlib as mpl

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

with open('MLH1 counts mercier paper fig S2.csv', 'r') as f:
    n = np.array(list(map(int, next(f).split(',')[1:])))
    c = np.array(list(map(int, next(f).split(',')[1:])))

from scipy.stats import poisson

plt.figure()

print(n, c)
nn = np.arange(np.max(n)+5)
plt.bar(n,c/np.sum(c), color='r')
plt.xlabel('Number of CO per cell')
plt.ylabel('Relative frequency')
plt.xlim(0, 30)
mean_c = np.sum(n*c)/np.sum(c)

plt.plot(nn, poisson(mean_c).pmf(nn), 'g-o')

plt.text(0.7, 0.9, f'$\mu={mean_c:.2f}$', transform=plt.gca().transAxes)

s_c2 = np.sum(n*n*c)
e_c = np.sum(n*c)/np.sum(c)
n_c = np.sum(c)
S2_c = (s_c2 - e_c**2*n_c)/n_c

plt.text(0.7, 0.7, f'$S^2 = {S2_c:.2f}$', transform=plt.gca().transAxes)
plt.text(0.7, 0.5, f'$N = {n_c}$', transform=plt.gca().transAxes)

plt.savefig('mlh_data.svg')

#plt.show()

