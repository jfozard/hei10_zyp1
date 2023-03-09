
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

from utils import *

base_path = '/media/foz/moredata2/PCH2/'
data_output_path = '../output/data_output/'

image_output_path = '../output/data_output/'

Path(image_output_path).mkdir(exist_ok=True, parents=True)


A = pd.read_csv(base_path+'pch2 SC lengths.csv') 

B = pd.read_csv(base_path+'pch2_focus_coordinates.csv')

A_g = A.groupby(['Plant', 'Image'])
B_g = B.groupby(['Plant', 'Image'])

img_scale = np.array([0.0909, 0.0397, 0.0397])
dist_threshold = 0.5
unit_scale = 1


def no_last_true(x):
    if len(x)==0:
        return 0
    idx = len(x) - 1 - np.argmax(x[::-1])
    if x[idx]:
        return idx+1
    else:
        return 0

def get_paths(base_path, genotype, plant, image, path):
    rootdir = Path(base_path) / genotype
    subdir = None
    for f in rootdir.iterdir():
        if plant in f.name:
            subdir = f.name
    path = 'Path' + str(path)+'.tif'
    fn = rootdir / subdir / Path('image'+str(image)) / path
    fn_hei10 = rootdir / subdir / Path('image'+str(image)) / 'HEI10.tif'
    fn_zyp = rootdir / subdir / Path('image'+str(image)) / 'ZYP1.tif'
    return str(fn), str(fn_hei10), str(fn_zyp)


class CellData(object):
    def __init__(self):
        self.paths = {}

class PathData(object):
    def __init__(self):
        pass

def process():
    pd_all = []
    cd_all = []

    n_off_path = 0
    for cell_idx, group in enumerate(A_g.groups):

        paths_cell = A_g.get_group(group)
        foci_cell = B_g.get_group(group)

        s = paths_cell.iloc[0]
        _, path_hei10, path_zyp = get_paths(base_path, s.Genotype, s.Plant, s.Image, 1)

        N_paths = len(paths_cell)
        N_foci = len(foci_cell)


        pts_fp_raw = np.array(foci_cell[['CoreZ', 'CoreY', 'CoreX']])

        stack = imread(path_hei10)

        paths = {(i+1): imread(get_paths(base_path, paths_cell.iloc[i].Genotype, paths_cell.iloc[i].Plant, paths_cell.iloc[i].Image, i+1)[0]) for i in range(N_paths) }   

        traced_paths = {}

        for i in range(1,N_paths+1):

            q, p, m = measure(paths[i], stack)

            if len(q)!=len(p):
                print(cell_idx, i, 'LENGTHS', len(q), len(p))

                    
            p = p*img_scale[np.newaxis,:]
            traced_paths[i] = p

        pts_fp = pts_fp_raw[:,:3]*img_scale[np.newaxis,:]

        bg = np.median(stack)

        measured_intensities = []
        for i in range(len(pts_fp_raw)):
            measured_intensities.append(measure_at_point(pts_fp_raw[i].astype(int), stack)-bg)

        measured_intensities = np.array(measured_intensities)

        # Place the foci onto the paths

        nearest_path = []
        nearest_dist = []
        path_pos = []

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
                nearest_path.append(u)  
                nearest_dist.append(distances[u]/unit_scale)
                path_point_relpos[u].append((pt_idx[u]/(len(traced_paths[u]-1))))
                point_path_idx[j] = u
                path_point_idx[u].append(j)
            else:
                print('Off path')
                n_off_path += 1

        all_point_idx_on_paths = np.array(list(point_path_idx))
        
        total_on_path_intensity = np.sum(measured_intensities[all_point_idx_on_paths])

        relative_intensities = measured_intensities / max(1e-10, total_on_path_intensity) 

        cd = CellData()
        cd.paths = []
        cd.points = pts_fp
        cd.point_path_idx = point_path_idx
        cd.path_point_idx = path_point_idx
        for i in range(1, N_paths+1):
            pd = PathData()
            pd.SC_length = paths_cell['SC length'].iloc[i-1]
            pd.foci_pos = path_point_relpos[i]
            pd.traced_path = traced_paths[i]
            pd.foci_intensities = [ relative_intensities[j] for j in path_point_idx[i]]
            pd.abs_intensities = [ measured_intensities[j] for j in path_point_idx[i]]

            cd.paths.append(pd)
            pd_all.append(pd)
        cd_all.append(cd)

    with open(data_output_path+'pch2.pkl', 'wb') as f:
        pickle.dump({'cell_data': cd_all, 'path_data': pd_all}, f)

process()

with open(data_output_path+'pch2.pkl', 'rb') as f:
    d = pickle.load(f)
    cd_all = d['cell_data']
    pd_all = d['path_data']




spacing_all = [ x*p.SC_length for p in pd_all for x in np.diff(np.sort(p.foci_pos)) ]
print('MAX SPACING', np.max(spacing_all))
assert(np.max(spacing_all)<24)
plt.figure()
plt.hist(spacing_all, bins=np.linspace(0,24,13), color='r')
plt.xlim(0, 24)
plt.ylabel('Frequency')
plt.xlabel('Focus Spacing ($\\mu$m)')
plt.text(0.7, 0.9, f'$N={len(spacing_all)}$', transform=plt.gca().transAxes)
plt.savefig(image_output_path+'pch2_spacing.svg') 



cell_tot_co = []
for c in cd_all:
    n = 0
    for p in c.paths:
        n += len(p.foci_pos)
    cell_tot_co.append(n)

    
plt.figure()
n, c = np.unique(cell_tot_co, return_counts=True)

mean_n = np.mean(cell_tot_co)
var_n = np.var(cell_tot_co, ddof=1)

print('MAX CELL CO', np.max(cell_tot_co))


plt.bar(n, c, color='r')
plt.plot(np.arange(21), poisson(mean_n).pmf(np.arange(21))*len(cell_tot_co), 'g-o')
plt.text(0.05, 0.9, f'$\mu={mean_n:.2f}$', transform=plt.gca().transAxes)
plt.text(0.05, 0.7, f'$S^2={var_n:.2f}$', transform=plt.gca().transAxes)
plt.text(0.05, 0.5, f'$N={len(cell_tot_co)}$', transform=plt.gca().transAxes)

plt.xlim(0, 20)
plt.xlabel('Cell focus number')
plt.ylabel('Frequency')
plt.savefig(image_output_path+'pch2_cell_co_number.svg') 
plt.close()

n_co = [len(p.foci_pos) for p in pd_all]

plt.figure()
n, c = np.unique(n_co, return_counts=True)

print('MAX SEG CO', np.max(n_co))

mean_n = np.mean(n_co)
var_n = np.var(n_co, ddof=1)
plt.text(0.7, 0.9, f'$\mu={mean_n:.2f}$', transform=plt.gca().transAxes)
plt.text(0.7, 0.7, f'$S^2={var_n:.2f}$', transform=plt.gca().transAxes)
plt.text(0.7, 0.5, f'$N={len(n_co)}$', transform=plt.gca().transAxes)
plt.xlim(-0.5, 6.5)
plt.xlabel('Segment focus number')
plt.ylabel('Frequency')

plt.bar(n, c, color='r')
plt.savefig(image_output_path+'pch2_segment_co_number.svg') 

pos_all = [ u for p in pd_all for u in p.foci_pos ]

plt.figure()
plt.hist(pos_all, bins=np.linspace(0,1,11), color='r')
plt.xlim(0, 1)
plt.xlabel('Relative focus position along segment', fontsize=24)
plt.ylabel('Frequency')
plt.text(0.7, 0.9, f'$N={len(pos_all)}$', transform=plt.gca().transAxes)
plt.savefig(image_output_path+f'pch_hist.svg') 


    
length_by_no = defaultdict(list)
for pd in pd_all:
    foci = np.array(pd.foci_pos)
    length_by_no[len(foci)].append(pd.SC_length)


violin_max_n = 3

violin_data = defaultdict(list)
for v in length_by_no.keys():
    if v<violin_max_n:
        violin_data[v] = length_by_no[v]
    else:
        violin_data[violin_max_n] += length_by_no[v]

violin_pos = list(violin_data.keys())

violin_labels = [ str(i) if i < violin_max_n else f'{violin_max_n}+' for i in violin_pos]


plt.figure()
plt.violinplot(list(violin_data.values()), positions=list(violin_pos), showmeans=True)
plt.xticks(ticks=violin_pos, labels=violin_labels)
plt.xlabel('Segment focus number')
plt.ylabel('Segment length ($\\mu$m)')
plt.xlim(-0.5, violin_max_n+0.5)
plt.ylim(0, 40)
for i,v in violin_data.items():
    if len(v)>0:
        plt.scatter([i]*len(v), v, marker='x')


plt.savefig(image_output_path+f'pch_len_by_n.svg') 

data_n = []
data_l = []
for i,v in length_by_no.items():
    data_n += [i]*len(v)
    data_l += list(v)

print('len vs n regression')
xx, yy, r2 = lin_fit(data_n, data_l, r2=True)

with open(image_output_path+'pch2_len_vs_n.txt', 'w') as of:
    of.write('n,l\n')
    for n, l in zip(data_n, data_l):
        of.write(f'{n},{l}\n')  

single_L = []
single_rel_int = []
for pd in pd_all:
    if len(pd.foci_pos)==1:
        single_L.append(pd.SC_length)
        single_rel_int.append(pd.foci_intensities[0])

xx, yy, r2 = lin_fit(single_L, single_rel_int, r2=True, min_x=0, max_x=30)

plt.figure()
plt.scatter(single_L, single_rel_int, color='r')
plt.plot(xx, yy, 'k-')
plt.xlabel('Segment length ($\\mu$m)')
plt.ylabel('Relative intensity')
plt.xlim(0, 30)
plt.ylim(0, 0.3)
plt.text(0.4, 0.9, f'$N={len(single_L)}$  $R^2={r2:.2f}$', transform=plt.gca().transAxes)
plt.savefig(image_output_path+f'pch2_single_int_vs_L.svg') 
    



