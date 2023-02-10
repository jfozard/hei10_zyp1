
import numpy as np
import scipy.linalg as la
from skimage.feature import peak_local_max, blob_dog
from skimage.filters import gaussian

from scipy.spatial import cKDTree
#
from measure_all import measure_chrom2 as measure

from scipy.stats import poisson


from tifffile import imread

import matplotlib.pyplot as plt

import os

import matplotlib as mpl
from pathlib import Path

from collections import defaultdict
import numpy.random as npr

import pickle

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


from utils import *


import pandas as pd

base_path = '/media/foz/moredata2/ZYP1/Col-0 (For zyp1 comparison)/'
image_output_path = '../output/data_output2/'
data_output_path = '../output/data_output2/'

Path(image_output_path).mkdir(exist_ok=True, parents=True)

A = pd.read_csv(base_path+'Col-0 Focus_coordinates.csv')

B = pd.read_csv(base_path+'Col-0_SNT_Foci_Measurements_15.03.22.csv')

A_g = A.groupby(['Plant', 'Image'])
B_g = B.groupby(['Plant', 'Image'])

img_scale = np.array([0.0909, 0.0397, 0.0397])
dist_threshold = 0.5
unit_scale = 1


class CellData(object):
    def __init__(self):
        self.paths = {}

class PathData(object):
    def __init__(self):
        pass

def process():

    of_pos = open('new_pos.txt','w')
    of_int = open('new_int.txt', 'w')

    pd_all = []
    cd_all = []
    for k in B.iloc[::5].index:
        s = B.loc[k]
        path = base_path + first_upper(s.Plant) + ' image '+str(s.Image)+'/'
        A_plant = A.loc[ (A['Plant'].str.lower()==s.Plant) & (A['Image']==s.Image) ]
        N_foci = get_n_foci(B.iloc[k:k+5])

        pts_fp_raw = np.array(A_plant[['CoreZ', 'CoreY', 'CoreX']])
        pts_fp = pts_fp_raw[:,:3]*img_scale[np.newaxis,:]
        stack = imread(path + 'HEI10.tif')

        n_off_path = 0

        N_paths = 5
        paths = {i: imread(path + f'Path{i}.tif') for i in range(1,N_paths+1) }   

        traced_paths = {}

        def dist(x,y):
            return la.norm(np.array(x)-np.array(y))


        for i in range(1,N_paths+1):


            # Use manually supplied data to find correct end of paths.
            b = B.iloc[k+i-1]

            q, p, m = measure(paths[i], stack)
            
            p = p*img_scale[np.newaxis,:]

            idx = int(b['Focus 1 ID']) - 1
            first_man_focus = pts_fp[idx]
            if dist(p[0], first_man_focus) > dist(p[-1], first_man_focus):
                p = p[::-1]

            traced_paths[i] = p



        bg = np.median(stack)

        measured_intensities = []
        for i in range(len(pts_fp_raw)):
            measured_intensities.append(measure_at_point(pts_fp_raw[i].astype(int), stack)-bg)

        measured_intensities = np.array(measured_intensities)

###
        

        path_trees = { i: cKDTree(traced_paths[i]) for i in traced_paths }

        path_point_relpos = {i:[] for i in range(1,N_paths+1)}
        point_path_idx = {}
        path_point_idx = {i:[] for i in range(1,N_paths+1)}
        
        for j, p in enumerate(pts_fp):
            distances, pt_idx = {}, {}
            for i in range(1,N_paths+1):
                distances[i], pt_idx[i] = path_trees[i].query(p[:3])

            u = np.argmin(list(distances.values()))+1
            min_dist = distances[u]


            if min_dist < dist_threshold*unit_scale:
                path_point_relpos[u].append((pt_idx[u]/(len(traced_paths[u]-1))))
                point_path_idx[j] = u
                path_point_idx[u].append(j)
            else:
                print('Off path')
                n_off_path +=1


        all_point_idx_on_paths = np.array(list(point_path_idx))
        
        total_on_path_intensity = np.sum(measured_intensities[all_point_idx_on_paths])

        relative_intensities = measured_intensities / max(1e-10, total_on_path_intensity) * len(all_point_idx_on_paths)

        

        cd = CellData()
        cd.paths = []
        for i in range(1,N_paths+1): 


            pd = PathData()
            
            b = B.iloc[k+(i-1)]

            order = np.argsort(path_point_relpos[i])
            pp = np.array(path_point_idx[i])[order]

            pd.SC_length = b.iloc[4]
            pd.foci_pos = np.array(path_point_relpos[i])[order]
            pd.traced_path = traced_paths[i]
            pd.foci_intensities = [ relative_intensities[j] for j in pp]
            pd.abs_intensities = [ measured_intensities[j] for j in pp]

            of_pos.write(','.join(map(str, pd.foci_pos))+'\n')
            of_int.write(','.join(map(str, pd.foci_intensities))+'\n')
            
            
            cd.paths.append(pd)
            pd_all.append(pd)
        cd_all.append(cd)
    with open(data_output_path+'wt.pkl', 'wb') as f:
        pickle.dump({'cell_data': cd_all, 'path_data': pd_all}, f)

    of_pos.close()
    of_int.close()

process()
with open(data_output_path+'wt.pkl', 'rb') as f:
    d = pickle.load(f)
    cd_all = d['cell_data']
    pd_all = d['path_data']


pos_all_s = []
pos_all = []
for pd in pd_all:
    p = pd.foci_pos
    pos_all_s += list(p)
    pos_all += list(p) + list(1-p)
    
with open(image_output_path+'wt_pos_all.txt','w') as f:
    f.write(str(pos_all_s))
    

fig, ax = plt.subplots()
rel_freq_hist(ax, pos_all, bins=np.linspace(0,1,11), color='r')
plt.xlabel('Relative focus position along\nchromosome pair', fontsize=22)
plt.ylabel('Relative frequency')# density')
plt.text(0.65, 0.9, f'N={len(pos_all)//2}', transform=plt.gca().transAxes)
plt.xlim(0, 1)
plt.savefig(image_output_path+'wt_pos_all.svg')
plt.savefig(image_output_path+'wt_pos_all.png') #USED

spacing_abs= []
for pd in pd_all:
    p = pd.foci_pos
    L = pd.SC_length
    if len(p)>1:
        spacing_abs += list(L*np.diff(p))

        
fig, ax = plt.subplots()
rel_freq_hist(ax, spacing_abs, bins=np.linspace(0, 60, 11), color='r')
plt.xlim(0, 60)
plt.text(0.7, 0.9, f'N={len(spacing_abs)}', transform=plt.gca().transAxes)
plt.xlabel('Focus spacing along chromosome pair ($\\mu$m)', fontsize=22)
plt.ylabel('Relative frequency')


plt.savefig(image_output_path+'wt_spacing_abs.svg')
plt.savefig(image_output_path+'wt_spacing_abs.png') 


foci_intensities = defaultdict(list)

for pd in pd_all:
    n = len(pd.foci_pos)
    if n>0:
        foci_intensities[n] += list(pd.foci_intensities)

plt.figure()

plt.violinplot([v for v in foci_intensities.values() if len(v)>0], [i for i,v in foci_intensities.items() if len(v)>0], showmeans=True)
for i,v in foci_intensities.items():
    if len(v)>0:
        plt.scatter([i]*len(v), v)


data_n = [i  for i,v in foci_intensities.items() for j in v]
data_v = [u  for v in foci_intensities.values() for u in v]

with open('col0_intensity.dat', 'w') as f:
    f.write('num_co,intensity\n')
    for n,v in zip(data_n, data_v):
        f.write(f'{n},{v}\n')        

with open(image_output_path+'wt_intensity_lr.txt', 'w') as of:
    xx, yy = lin_fit(data_n, data_v, of=of)

plt.plot(xx, yy, 'r-')
plt.xticks([1,2,3])
plt.xlabel('Number of foci per chromosome pair', fontsize=24)
plt.ylabel('Focus relative intensity')

plt.savefig(image_output_path+'wt_intensity_violin.png') 
plt.savefig(image_output_path+'wt_intensity_violin.svg')


of_stats = open(image_output_path+'single_double_triple_tests.txt', 'w')

import rpy2

from rpy2.robjects.packages import importr

utils = importr('utils')


import rpy2.robjects as robjects

import rpy2.rinterface_lib.callbacks
    # Record output #
cap_stdout = []
cap_stderr = []
    # Dummy functions #
def add_to_stdout(line): cap_stdout.append(line)
def add_to_stderr(line): cap_stderr.append(line)
    # Keep the old functions #
stdout_orig = rpy2.rinterface_lib.callbacks.consolewrite_print
stderr_orig = rpy2.rinterface_lib.callbacks.consolewrite_warnerror
    # Set the call backs #
rpy2.rinterface_lib.callbacks.consolewrite_print     = add_to_stdout
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = add_to_stderr

data = rpy2.robjects.ListVector([('single', robjects.FloatVector(foci_intensities[1])),
                                 ('double', robjects.FloatVector(foci_intensities[2])),
                                 ('triple', robjects.FloatVector(foci_intensities[3])) ] )



dunn = importr('dunn.test')
stats = importr('stats')
base = importr('base')

out = dunn.dunn_test(data, method='bonferroni', list='TRUE', altp='TRUE', kw='TRUE')
print(out, file=of_stats)

out2 = stats.kruskal_test(data)
print(out2, file=of_stats)

of_stats.write('\n'.join(cap_stdout))
of_stats.write('\n'.join(cap_stderr))

of_stats.close()

no_co_cell = []
for cd in cd_all:
    count = 0
    for pd in cd.paths:
        count += len(pd.foci_pos)
    no_co_cell.append(count)

print(no_co_cell)
print('MAX CO CELL', np.max(no_co_cell))

plt.figure()
d = np.bincount(no_co_cell, minlength=31)[:31]
plt.bar(np.arange(31), d/np.sum(d))
plt.xticks([0,10,20,30])
plt.xlim(0, 30)

mean_n = np.mean(no_co_cell)
var_n = np.var(no_co_cell, ddof=1)
plt.text(0.7, 0.9, f'$\mu={mean_n:.2f}$', transform=plt.gca().transAxes)
plt.text(0.7, 0.7, f'$S^2={var_n:.2f}$', transform=plt.gca().transAxes)
plt.plot(np.arange(31), poisson(mean_n).pmf(np.arange(31)), 'g-o')
plt.xlabel('Number of CO per cell')
plt.ylabel('Relative frequency')
plt.savefig(image_output_path+f'wt_data-cell-number.svg')
plt.close()

