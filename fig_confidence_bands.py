# -*- coding: utf-8 -*-
"""
@author: Etienne Lasalle
"""

#set working directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.path.basename(__file__) + ' has started')

#imports
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
from multiprocessing import Pool

import graphs as gr
import processes as pr
import extended_persistence as pers

#plot parameters
plt.rcParams['figure.figsize'] = (6,4)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.autolayout'] = True
plt.rc('font', size=12)
plt.rc('axes', labelsize=14)

#save parameters
save_fig = True
transparent = True
prefix = 'fig\\'


#########################
# CONFIDENCE BAND FOR HKD
#########################

#%%Erdös-Renyi vs SBM graphs

#fix a seed for reproducibility
np.random.seed(1234)

#graphs parameters
n = 50
p = .5
def samplerER():
    connect = nx.to_numpy_array(nx.erdos_renyi_graph(n,p))
    egelems = pr.laplacian_egelems(connect, nullfirstegv=True)
    return(connect , egelems)

ns = [25,25]
ps = [[.75,.25],[.25,.75]]
def samplerSBM():
    connect = nx.to_numpy_array(nx.generators.community.stochastic_block_model(ns,ps))
    egelems = pr.laplacian_egelems(connect, nullfirstegv=True)
    return(connect , egelems)


#time grid
ts = np.linspace(0,.3,100)

#simulation parameters
n_samples = 100
alpha = .05
n_bootstrap = 1000

#computations
data0, data1 = [samplerER() for _ in range(n_samples)], [samplerER() for _ in range(n_samples)]
HKDs = np.array([[pr.HKD(g0[1], g1[1], t) for t in ts] for g0, g1  in zip(data0, data1)])
meanER, l_confER, h_confER = pr.confidence_band(HKDs, alpha, n_bootstrap)

data0, data1 = [samplerER() for _ in range(n_samples)], [samplerSBM() for _ in range(n_samples)]
HKDs = np.array([[pr.HKD(g0[1], g1[1], t) for t in ts] for g0, g1  in zip(data0, data1)])
meanSBM, l_confSBM, h_confSBM = pr.confidence_band(HKDs, alpha, n_bootstrap)

#plot
plt.figure(1), plt.clf()
plt.fill_between(ts, l_confER, h_confER, color='r', alpha = .3, )
plt.plot(ts, meanER, 'r', label='ER-ER')
plt.fill_between(ts, l_confSBM, h_confSBM, color='b', alpha = .3, )
plt.plot(ts, meanSBM, 'b', label='ER-SBM')
plt.xlabel('t'), plt.ylabel('HKD')
plt.legend()

#save figure
if save_fig:
    file = prefix + 'conf_band_HKD_ERvsSBM'
    plt.savefig(file, transparent=transparent)
    print('    ' + file + ' done')


#%%Erdös-Renyi vs SBM weighted graphs

#fix a seed for reproducibility
np.random.seed(1234)

#graphs parameters
n = 50
p = .5
def samplerER():
    connect = nx.to_numpy_array(nx.erdos_renyi_graph(n,p))
    connect = gr.add_weights(connect, 'uniform', distr_param=(0,2))     # add uniform weights in [0,2]
    egelems = pr.laplacian_egelems(connect, nullfirstegv=True)
    return(connect , egelems)

ns = [25,25]
ps = [[.75,.25],[.25,.75]]
def samplerSBM():
    connect = nx.to_numpy_array(nx.generators.community.stochastic_block_model(ns,ps))
    connect = gr.add_weights(connect, 'uniform', distr_param=(0,2))
    egelems = pr.laplacian_egelems(connect, nullfirstegv=True)
    return(connect , egelems)


#time grid
ts = np.linspace(0,.3,100)

#simulation parameters
n_samples = 100
alpha = .05
n_bootstrap = 1000

#computations
data0, data1 = [samplerER() for _ in range(n_samples)], [samplerER() for _ in range(n_samples)]
HKDs = np.array([[pr.HKD(g0[1], g1[1], t) for t in ts] for g0, g1  in zip(data0, data1)])
meanER, l_confER, h_confER = pr.confidence_band(HKDs, alpha, n_bootstrap)

data0, data1 = [samplerER() for _ in range(n_samples)], [samplerSBM() for _ in range(n_samples)]
HKDs = np.array([[pr.HKD(g0[1], g1[1], t) for t in ts] for g0, g1  in zip(data0, data1)])
meanSBM, l_confSBM, h_confSBM = pr.confidence_band(HKDs, alpha, n_bootstrap)

#plot
plt.figure(2), plt.clf()
plt.fill_between(ts, l_confER, h_confER, color='r', alpha = .3, )
plt.plot(ts, meanER, 'r', label='ER-ER')
plt.fill_between(ts, l_confSBM, h_confSBM, color='b', alpha = .3, )
plt.plot(ts, meanSBM, 'b', label='ER-SBM')
plt.xlabel('t'), plt.ylabel('HKD')
plt.legend()

#save figure
if save_fig:
    file = prefix + 'conf_band_HKD_ERvsSBM_weighted'
    plt.savefig(file, transparent=transparent)
    print('    ' + file + ' done')


print(os.path.basename(__file__) + ' is done')
