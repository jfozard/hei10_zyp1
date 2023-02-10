
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, levene
from random import choices, seed

from new_zyp1_plots import process_file
import rpy2

simulation_output_path = '../output/simulation_output/'

data_sim = process_file(simulation_output_path+'regtot-nuc-ox-poisson-0.0.dat')

cd_all = data_sim['cell_data']

n_co_cell = []
for cd in cd_all:
    co_cell = 0.0
    for q in range(5):
        p = cd.paths[q]
        co_cell += len(p.foci_pos)
    n_co_cell.append(co_cell)



#print(data_sim)

data = pd.read_csv('../mercier_zyp1/mercier zyp1 hei10 mlh1 data comb male hom.csv')
foci = data['MLH1 foci']
n= len(foci)
m = np.mean(foci)
v = np.var(foci, ddof=1)
CI_L = (n-1)*v/chi2.ppf(.95, n-1)
CI_U = (n-1)*v/chi2.ppf(.05, n-1)
print(' mercier stats', n, m, v, CI_L, CI_U)

print(np.var(n_co_cell, ddof=1), np.var(foci, ddof=1))
print(levene(n_co_cell, foci, center='mean'))

from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector, IntVector, StrVector
car = importr('car')

data = np.concatenate((n_co_cell, foci))
cond = np.concatenate((1*np.ones(len(n_co_cell)), 2*np.ones(len(foci))))

r = car.leveneTest(FloatVector(data.tolist()), IntVector(cond.tolist()), center=StrVector(['mean']))

print(r)

plt.figure()
plt.hist(foci)
plt.figure()
plt.hist(n_co_cell)
plt.show()
