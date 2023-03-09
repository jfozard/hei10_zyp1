
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

PAIRS = True

import pandas as pd

base_path = '/media/foz/moredata2/ZYP1/ZYP1-1/'
image_output_path = '../output/data_output/'
data_output_path = '../output/data_output/'

Path(image_output_path).mkdir(exist_ok=True, parents=True)

A = pd.read_csv(base_path+'Focus_coordinates.csv')

B = pd.read_csv(base_path+'Copy of ZYP1_SNT_Foci_Measurements_18.2.22.csv')


img_scale = np.array([0.0909, 0.0397, 0.0397])
dist_threshold = 0.3
dist_threshold_group = 0.5
unit_scale = 1

def get_n_foci(A):
    N_foci = []
    for i in range(0, len(A.index), 2):
        n1 = no_last_true(~pd.isna(A.iloc[i,16:16+9]))
        n2 = no_last_true(~pd.isna(A.iloc[i+1,16:16+9]))
        n = max(n1, n2)
        N_foci += [n,n]
    return N_foci

class CellData(object):
    def __init__(self):
        self.paths = []

class PathData(object):
    def __init__(self):
        pass

def process():


    of_idx = open(image_output_path+'new_co_idx', 'w')
    of_pos = open(image_output_path+'new_zyp_pos.txt','w')
    of_int = open(image_output_path+'new_zyp_int.txt', 'w')

    pd_all = []
    cd_all = []
    for k in B.iloc[::10].index:
        s = B.loc[k]
        path = base_path + s.Plant + ' image '+s.Image[5:].strip()+'/'
        A_plant = A.loc[ (A['Plant']==s.Plant) & (A['Image']==int(s.Image[5:])) ]
        N_foci = get_n_foci(B.iloc[k:k+10])

        pts_fp_raw = np.array(A_plant[['CoreZ', 'CoreY', 'CoreX']])
        pts_fp = pts_fp_raw[:,:3]*img_scale[np.newaxis,:]

        stack = imread(path + 'HEI10.tif')

        n_off_path = 0

        paths = {i: imread(path + f'Path{i}.tif') for i in range(1,11 if PAIRS else 6) }   

        traced_paths = {}


        def dist(x,y):
            return la.norm(np.array(x)-np.array(y))


        for i in range(1,11,2):
        # Use manually supplied data to find correct end of paths.

            b = B.iloc[k+i-1]
            c = B.iloc[k+i]


            j = i+1
            q, p, m = measure(paths[i], stack)
 

            if len(q)!=len(p):
                print('LENGTHS', len(q), len(p))
           
            p = p*img_scale[np.newaxis,:]

            if not pd.isna(b['Focus 1 ID']):
                idx = int(b['Focus 1 ID']) - 1
                first_man_focus = pts_fp[idx]
                if dist(p[0], first_man_focus) > dist(p[-1], first_man_focus):
                     p = p[::-1]
            elif not pd.isna(c['Focus 1 ID']):
                idx = int(c['Focus 1 ID']) - 1
                first_man_focus = pts_fp_raw[idx]
                if dist(p[0], first_man_focus) > dist(p[-1], first_man_focus):
                    p = p[::-1]
            else:
                 print('No first focus?', b['Plant'], b['Image'], b['Path'])        

            traced_paths[i] = p
        
            path_start_i = p[0]

            _, p, m = measure(paths[j], stack)
            p = p*img_scale[np.newaxis,:]
            if dist(p[0], path_start_i) > dist(p[-1], path_start_i):
                p = p[::-1]
            traced_paths[j] = p

   
        bg = np.median(stack)
    
        measured_intensities = []
        for i in range(len(pts_fp_raw)):
            measured_intensities.append(measure_at_point(pts_fp_raw[i].astype(int), stack)-bg)

        measured_intensities = np.array(measured_intensities)

        # Place the foci onto the paths


        path_trees = { i: cKDTree(traced_paths[i]) for i in traced_paths }


        path_point_relpos = {i:[] for i in range(1,6)}
        point_path_idx = {}
        point_path_nearest = {}
        path_point_idx = {i:[] for i in range(1,6)}


        for j, p in enumerate(pts_fp):
            distances, pt_idx = {}, {}
            for i in range(1,11):
                distances[i], pt_idx[i] = path_trees[i].query(p[:3])
            group_distances = []
            for i in range(1,6):
                group_distances.append(0.5*sum((distances[2*i-1], distances[2*i])))


            group = np.argmin(group_distances)+1
            group_dist = group_distances[group-1]
            if group_dist < dist_threshold_group*unit_scale:
                if distances[2*group-1] <= distances[2*group]:
                    u = 2*group-1
                    o = 2*group
                else:
                    u = 2*group 
                    o = 2*group -1

                point_path_idx[j] = group
                point_path_nearest[j] = u
                path_point_relpos[group].append((pt_idx[u]/(len(traced_paths[u])-1)))
                path_point_idx[group].append(j)
                
            else:
                u = np.argmin(list(distances.values()))+1
                min_dist = distances[u]
                group = (u+1)//2


                if min_dist < dist_threshold*unit_scale:
                    point_path_idx[j] = group
                    point_path_nearest[j] = u
                    path_point_relpos[group].append((pt_idx[u]/(len(traced_paths[u])-1)))
                    path_point_idx[group].append(j)
                else:
                    
                    print('Off path', k, min_dist, u, pt_idx[u], len(traced_paths[u])-1)
                    n_off_path +=1    
      
        all_point_idx_on_paths = np.array(list(point_path_idx))

        print('On path', len(list(point_path_idx)))
        
        total_on_path_intensity = np.sum(measured_intensities[all_point_idx_on_paths])

        relative_intensities = measured_intensities / max(1e-10, total_on_path_intensity) * len(all_point_idx_on_paths)

        print(k, path_point_idx, file=of_idx)


        cd = CellData()
        cd.paths = []
        cd.points = pts_fp
        cd.point_path_idx = point_path_idx
        cd.path_point_idx = path_point_idx
        
        for group in range(1,6): 

            path_data = PathData()
            b = B.iloc[k+group*2-2]

            order = np.argsort(path_point_relpos[group])
            pp = np.array(path_point_idx[group])[order]

            L1 = B.iloc[k+group*2-2].iloc[4]
            L2 = B.iloc[k+group*2-1].iloc[4]
            L = 0.5*(L1+L2)
            path_data.SC_length = L

            path_data.foci_pos = np.array(path_point_relpos[group])[order]
            path_data.traced_path = (traced_paths[group*2-1], traced_paths[group*2])
            path_data.foci_intensities = [ relative_intensities[j] for j in pp]
            path_data.abs_intensities = [ measured_intensities[j] for j in pp]

            of_pos.write(','.join(map(str, path_data.foci_pos))+'\n')
            of_int.write(','.join(map(str, path_data.foci_intensities))+'\n')
            
            
            cd.paths.append(path_data)
            pd_all.append(path_data)
        cd_all.append(cd)

    with open(data_output_path+'zyp1.pkl', 'wb') as f:
        pickle.dump({'cell_data': cd_all, 'path_data': pd_all}, f)

    of_pos.close()
    of_int.close()

process()
with open(data_output_path+'zyp1.pkl', 'rb') as f:
    d = pickle.load(f)
    cd_all = d['cell_data']
    pd_all = d['path_data']


pos_all_s = []
pos_all = []
for pd in pd_all:
    p = pd.foci_pos
    pos_all_s += list(p)
    pos_all += list(p) + list(1-p)
    
with open(image_output_path+'zyp1_pos_all.txt','w') as f:
    f.write(str(pos_all_s))
    

fig, ax = plt.subplots()
rel_freq_hist(ax, pos_all, bins=np.linspace(0,1,11), color='r')#, density=True)
plt.xlabel('Relative focus position along\nchromosome pair', fontsize=22)
plt.ylabel('Relative frequency')
plt.text(0.65, 0.9, f'N={len(pos_all)//2}', transform=plt.gca().transAxes)
plt.xlim(0, 1)
plt.savefig(image_output_path+'pos_all.svg')
plt.savefig(image_output_path+'pos_all.png')

spacing_abs= []
for pd in pd_all:
    p = pd.foci_pos
    L = pd.SC_length
    if len(p)>1:
        spacing_abs += list(L*np.diff(p))

        
fig, ax = plt.subplots()
rel_freq_hist(ax, spacing_abs, bins=np.linspace(0, 78, 14), color='r')#, density=True)
plt.xlim(0, 78)
plt.text(0.7, 0.9, f'N={len(spacing_abs)}', transform=plt.gca().transAxes)
plt.xlabel('Focus spacing along chromosome pair ($\\mu$m)', fontsize=22)
plt.ylabel('Relative frequency')
plt.savefig(image_output_path+'spacing_abs.svg')
plt.savefig(image_output_path+'spacing_abs.png') # USED


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

with open('zyp1_intensity.dat', 'w') as f:
    f.write('num_co,intensity\n')
    for n,v in zip(data_n, data_v):
        f.write(f'{n},{v}\n')        

with open(image_output_path+'intensity_lr.txt', 'w') as of:
    xx, yy = lin_fit(data_n, data_v, of=of)

plt.plot(xx, yy, 'r-')
plt.xticks(np.arange(1, 10))
plt.xlabel('Number of foci per chromosome pair', fontsize=24)
plt.ylabel('Focus relative intensity')

plt.savefig(image_output_path+'intensity_violin.png')
plt.savefig(image_output_path+'intensity_violin.svg')


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
plt.savefig(image_output_path+f'data-cell-number.svg')
plt.close()

    
