
import numpy as np
import scipy.linalg as la
from skimage.feature import peak_local_max, blob_dog
from skimage.filters import gaussian

from scipy.spatial import cKDTree
#
from measure_all import measure_chrom2 as measure

from scipy.stats import poisson
from random import sample

from tifffile import imread

import matplotlib.pyplot as plt

import os

import matplotlib as mpl
from pathlib import Path

from collections import defaultdict
import numpy.random as npr

import pickle

import pandas as pd

mpl.rcParams.update({
    'figure.facecolor': 'none',
    'figure.edgecolor': 'none',
    'font.size': 20,
    'figure.dpi': 72,
    'figure.subplot.bottom' : .15,
    'axes.labelsize':28,
    'savefig.edgecolor': 'none',
    'savefig.facecolor': 'none',
    'svg.fonttype': 'none',

})

def rel_freq_hist(ax, m, bins=np.linspace(0,1,11), stacked=False, **ex_args):
    if np.min(m)<bins[0] or np.max(m)>bins[-1]:
        raise Exception
    if stacked==False:
        h, edges = np.histogram(m, bins)
        h = h/np.sum(h)
        ax.bar(edges[:-1], h, edges[1:]-edges[:-1], align='edge', **ex_args)
    else:
        b = np.zeros_like(bins)[:-1]
        for d in m:
            h, edges = np.histogram(d, bins)
            ax.bar(edges[:-1], h, edges[1:]-edges[:-1], align='edge', bottom = b)
            b += h

from scipy.stats import kstest
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


def make_sphere(R=5):
    x, y, z = np.ogrid[-R:R, -R:R, -R:R]
    sphere = x**2 + y**2 + (0.0909/0.0397 * z)**2 < R**2
    return sphere

def make_mask_s(p, melem, measure_stack):
    mask = melem
    
    R = melem.shape[0] // 2
    r, c, z = p

    m_data = np.zeros(melem.shape)
    s = measure_stack.shape
    o_1, o_2, o_3 = max(R-r, 0), max(R-c, 0), max(R-z,0)
    e_1, e_2, e_3 = min(R-r+s[0], 2*R), min(R-c+s[1], 2*R), min(R-z+s[2], 2*R)
    m_data[o_1:e_1,o_2:e_2,o_3:e_3] = measure_stack[max(r-R,0):min(r+R,s[0]),max(c-R,0):min(c+R,s[1]),max(z-R,0):min(z+R, s[2])]
    return mask, m_data

def measure_from_mask(mask, measure_stack):
    return np.sum(mask * measure_stack)

def max_from_mask(mask, measure_stack):
    return np.max(mask * measure_stack)


def measure_at_point(p, measure_stack, op='mean'):
    melem = make_sphere()
    if op=='mean':
        mask, m_data = make_mask_s(p, melem, measure_stack)
        melem_size = np.sum(melem)
        return float(measure_from_mask(mask, m_data) / melem_size)
    else:
        mask, m_data = make_mask_s(p, melem, measure_stack)
        return float(max_from_mask(mask, m_data))



def no_last_true(x):
    if len(x)==0:
        return 0
    idx = len(x) - 1 - np.argmax(x[::-1])
    if x[idx]:
        return idx+1
    else:
        return 0

def get_n_foci(A):
    N_foci = []
    for i in range(0, len(A.index)):
        n = no_last_true(~pd.isna(A.iloc[i,10:10+3]))
        N_foci.append(n)
    return N_foci

def first_upper(s):
    def i_upper(x):
        x = x[0:1].upper() + x[1:]
        return x

    return ' '.join(map(i_upper,s.split(' ')))
