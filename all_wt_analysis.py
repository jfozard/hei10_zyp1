
import numpy as np
import scipy.linalg as la

from scipy.spatial import cKDTree
from measure_all import measure_chrom2 as measure

from tifffile import imread

from scipy.stats import poisson

import matplotlib.pyplot as plt

import os

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
    'svg.fonttype': 'none',

})

# Generate a 5 voxel radius sphere.
def make_sphere(R=5):
    x, y, z = np.ogrid[-R:R, -R:R, -R:R]
    sphere = x**2 + y**2 + (0.0909/0.0397 * z)**2 < R**2
    return sphere

# Fix mask near edges of stack
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

# Find mean / max intensity of stack in a sphere around a point
def measure_at_point(p, measure_stack, op='mean'):
    melem = make_sphere()
    if op=='mean':
        mask, m_data = make_mask_s(p, melem, measure_stack)
        melem_size = np.sum(melem)
        return float(measure_from_mask(mask, m_data) / melem_size)
    else:
        mask, m_data = make_mask_s(p, melem, measure_stack)
        return float(max_from_mask(mask, m_data))



import pandas as pd

# Change these paths to point to downloaded data
data_base_path = '/media/foz/moredata2/ZYP1/Col-0 (For zyp1 comparison)/' 

A = pd.read_csv(data_base_path + 'Col-0 Focus_coordinates.csv')
B = pd.read_csv(data_base_path + 'Col-0_SNT_Foci_Measurements_15.03.22.csv')


img_scale = np.array([0.0909, 0.0397, 0.0397])
dist_threshold = 0.5
dist_threshold_group = 0.5
unit_scale = 1


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

pos_all = []
pos_n = { i:[] for i in range(10) }
intensities = []
intensities_n = { i:[] for i in range(10) }
foci_intensities = { i:[] for i in range(10) }
pos_all_s = []

spacing = []

def first_upper(s):
    def i_upper(x):
        x = x[0:1].upper() + x[1:]
        return x

    return ' '.join(map(i_upper,s.split(' ')))


for k in B.iloc[::5].index:
    s = B.loc[k]
    path = data_base_path + first_upper(s.Plant) + ' image '+str(s.Image)+'/'
    A_plant = A.loc[ (A['Plant'].str.lower()==s.Plant) & (A['Image']==s.Image) ]

    N_foci = get_n_foci(B.iloc[k:k+5])

    pts_fp_raw = np.array(A_plant[['CoreZ', 'CoreY', 'CoreX']])


    stack = imread(path + 'HEI10.tif')

    paths = {i: imread(path + f'Path{i}.tif') for i in range(1,6) }   

    traced_paths = {}
    path_starts = {}
    mean_hei10 = {}

    def dist(x,y):
        return la.norm(np.array(x)-np.array(y))


    for i in range(1,6):

        # Use Chris' data to find correct end of paths.

        b = B.iloc[k+i-1]

        q, p, m = measure(paths[i], stack)
        
        idx = int(b['Focus 1 ID']) - 1
        first_chris_focus = pts_fp_raw[idx]
        if dist(p[0]*img_scale, first_chris_focus*img_scale) > dist(p[-1]*img_scale, first_chris_focus*img_scale):
            p = p[::-1]
            m = m[::-1]


        p = p*img_scale[np.newaxis,:] # Convert from voxels to microns
        traced_paths[i] = p
        path_starts[i] = p[0]
        mean_hei10[i] = m


    pts_fp = pts_fp_raw[:,:3]*img_scale[np.newaxis,:] # Focipicker focus coordinates in microns

    bg = np.median(stack)
    
    measured_intensities = []
    for i in range(len(pts_fp_raw)):
        measured_intensities.append(measure_at_point(pts_fp_raw[i].astype(int), stack)-bg) # Measure mean intensity in 5 voxel sphere,
                                                                                           # subtract stack median as estimate of bg

    measured_intensities = np.array(measured_intensities)

    pts_fp_tree = cKDTree(pts_fp)

    # Place the foci onto the paths


    path_trees = { i: cKDTree(traced_paths[i]) for i in traced_paths }

    point_traces = {i:[] for i in range(1,6)}
    point_idx = {i:[] for i in range(1,6)}

    for j, p in enumerate(pts_fp):
        distances, pt_idx = {}, {}
        for i in range(1,6): # Find closest point on each trace
            distances[i], pt_idx[i] = path_trees[i].query(p[:3])
            
        group = np.argmin(list(distances.values()))+1
        group_dist = distances[group]
        
        if group_dist < dist_threshold_group: # Distance to trace sufficiently small
            u = group
            pp = (pt_idx[u]/(len(traced_paths[u]-1)))
            point_traces[u].append(pp)
            point_idx[u].append(j)


    cell_tot_int = 0.0

    # Normalize according to sum of intensities of all assigned foci
    n_co = 0
    for i in range(1, 6): 
        cell_tot_int += sum(measured_intensities[j] for j in point_idx[i] )
        n_co += len(point_idx[i])

    cell_tot_int /= n_co
    
        
    for i in range(1,6): 

        b = B.iloc[k+(i-1)]

        order = np.argsort(point_traces[i])
        pp = np.array(point_idx[i])[order]
        qq = np.array(point_traces[i])[order]
        
        
        if len(point_traces[i])>0:
            n = len(qq)
            pos_n[n] += list(qq)
            pos_n[n] += list(1-qq)
            pos_all += list(qq)
            pos_all += list(1-qq)

            pos_all_s += list(np.minimum(qq, 1-qq))

            foci_intensities[n] += [ measured_intensities[j]/cell_tot_int for j in pp ]
            if n>1:
                spacing += list(np.diff(qq))


from scipy.stats import kstest

import statsmodels.api as sm


# Linear regression using python libraries
def lin_fit(x, y, min_x=None, of=None, r2=False):
    X = np.array(x)
    Y = np.array(y)
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    if of is None:
        print(est2.summary())
    else:
        print(est2.summary(), file=of)
    if min_x is None:
        X = np.linspace(np.min(X), np.max(X), 100)
    else:
        X = np.linspace(min_x, np.max(X), 100)
    X2 = sm.add_constant(X)
    yy = est2.predict(X2)
    if not r2:
        return X, yy
    else:
        return X, yy, est2.rsquared


# Output file for testing
with open('wt_pos_all.txt','w') as f:
    f.write(str(pos_all_s))



plt.figure()
plt.hist(pos_all, bins=np.linspace(0,1,11), color='r', density=True)
plt.xlabel('Relative focus position along CP')
plt.ylabel('Frequency density')
plt.text(0.65, 0.9, f'N={len(pos_all)//2}', transform=plt.gca().transAxes)
plt.xlim(0, 1)
plt.savefig('wt_pos_all.svg')


plt.figure()
plt.hist(spacing, bins=np.linspace(0, 1, 11), color='r', density=True)
plt.xlim(0, 1)
plt.text(0.7, 0.9, f'N={len(spacing)}', transform=plt.gca().transAxes)

plt.xlabel('Relative focus spacing along CP')
plt.ylabel('Frequency density')

plt.savefig('wt_spacing.svg')


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
        

xx, yy = lin_fit(data_n, data_v)
plt.plot(xx, yy, 'r-')
plt.xticks([1,2,3])
plt.xlabel('Number of foci per CP')
plt.ylabel('Focus relative intensity')

import rpy2

from rpy2.robjects.packages import importr

utils = importr('utils')


#utils.chooseCRANmirror(ind=1)                                                                                                   
#utils.install_packages('dunn.test')                                                                                             
#utils.install_packages('kruskal')                                                                                               

import rpy2.robjects as robjects

data = rpy2.robjects.ListVector([('single', robjects.FloatVector(foci_intensities[1])),
                                 ('double', robjects.FloatVector(foci_intensities[2])),
                                 ('triple', robjects.FloatVector(foci_intensities[3])) ] )



dunn = importr('dunn.test')
stats = importr('stats')
base = importr('base')

out = dunn.dunn_test(data, method='bonferroni', list='TRUE', altp='TRUE', kw='TRUE')
print(out)

out2 = stats.kruskal_test(data)
print(out2)

"""

Kruskal-Wallis chi-squared = 17.5325, df = 2, p-value = 0


                           Comparison of x by group                            
                                 (Bonferroni)                                  
Col Mean-|
Row Mean |          1          2
---------+----------------------
       2 |   2.274758
         |     0.0688
         |
       3 |   4.153374   2.782924
         |    0.0001*    0.0162*


List of pairwise comparisons: Z statistic (adjusted p-value)
---------------------------
1 - 2 :  2.274758 (0.0688)
1 - 3 :  4.153374 (0.0001)*
2 - 3 :  2.782924 (0.0162)*

alpha = 0.05
Reject Ho if p <= alpha
$chi2
[1] 17.53248

$Z
[1] 2.274758 4.153375 2.782924

$altP
[1] 2.292042e-02 3.276076e-05 5.387136e-03

$altP.adjusted
[1] 6.876127e-02 9.828228e-05 1.616141e-02

$comparisons
[1] "1 - 2" "1 - 3" "2 - 3"



	Kruskal-Wallis rank sum test

data:  list(single = c(1.36440361020993, 1.7435728458505, 1.30901430003766, 0.956347391023763, 1.00571893356331, 1.1562373200348, 1.29653588517326, 0.960790786977514, 1.373319728453, 1.14283682856382, 0.839323262544075, 1.13663947395785, 0.986373254157045, 1.58350574626772, 1.21389722080761, 1.15943046678454, 1.00457419476663), double = c(0.921567420799711, 1.20761121088095, 0.647980051774539, 1.1948052742847, 1.18730802994954, 0.752096923243259, 0.80740265754209, 0.91682482131528, 0.91902298118222, 1.19881447093068, 1.22955027417947, 0.640000593716345, 0.764797506284857, 0.884190069235513, 1.39484116172868, 0.649615245736005, 0.840501813415501, 1.42785478957141, 0.745181696072145, 1.15514000802504, 1.11601986720642, 0.82615107406803, 0.935824745035352, 1.0597263559949, 1.08409789016101, 1.6830308014935, 0.672174605924444, 1.28179894756806, 0.583381766586341, 1.02009183932644, 1.23496929140793, 0.922972159870627, 0.608881542483492, 1.11354684330834, 1.44002456940352, 0.861198136322483, 0.621635579374904, 1.2558773961293, 0.665355942545814, 1.45664731863679, 0.917453466804958, 1.30486134440263, 0.909968572896671, 0.82484454608811, 1.57329174827591, 1.17563451226099, 0.93380407652834, 0.946666860366307, 0.8968553405421, 1.55360545557723, 0.539749641895977, 1.5714615412492), triple = c(0.351036452685949, 1.32850887773999, 0.940505928194468, 0.817915763264519, 0.714878871376991, 0.889030663845472, 0.377847180544851, 0.84806036484514, 0.795663537312218, 0.986454516001576, 0.791217047221856, 0.699274683156163, 1.01813431623961, 0.735586159656531, 1.04274594514347, 0.442431359323395, 0.862911058100592, 0.537678312074069, 0.815143098558238, 1.07954628077263, 0.616197555165186))
Kruskal-Wallis chi-squared = 17.532, df = 2, p-value = 0.0001559
"""

plt.savefig('wt_intensity_violin.png')
plt.savefig('wt_intensity_violin.svg')

