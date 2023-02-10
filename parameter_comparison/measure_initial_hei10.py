
import numpy as np

A = np.loadtxt('M_5_1_0.0.dat', delimiter=',')
print('mean, stddev', np.mean(A, axis=0), np.std(A, ddof=1, axis=0))
