import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def lin_fit(x, y, min_x=None, max_x=None, of=None, r2=False):
    X = np.array(x)
    Y = np.array(y)
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    if of is None:
        print(est2.summary())
    else:
        print(est2.summary(), file=of)
    lx = np.min(X) if min_x is None else min_x
    ux = np.max(X) if max_x is None else max_x
    X = np.linspace(lx, ux, 100)
    X2 = sm.add_constant(X)
    yy = est2.predict(X2)
    if not r2:
        return X, yy
    else:
        return X, yy, est2.rsquared

A = np.loadtxt('pch2_len_vs_n.txt', delimiter=',', skiprows=1)
print(A)
xx, yy = lin_fit(A[:,0], A[:,1])
plt.scatter(A[:,0], A[:,1])
plt.plot(xx,yy)
plt.show()
