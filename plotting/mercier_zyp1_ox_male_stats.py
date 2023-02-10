
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from random import choices, seed

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

plt.hist(foci)
plt.show()
