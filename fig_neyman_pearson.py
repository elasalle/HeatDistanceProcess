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


#%% testing SBM vs ER

#fix a seed for reproducibility
np.random.seed(1234)

#graphs parameters
n = 50
def samplerER(p):
    connect = nx.to_numpy_array(nx.erdos_renyi_graph(n,p))
    egelems = pr.laplacian_egelems(connect, nullfirstegv=True)
    return(connect , egelems)

p = .5
ns_samples = np.array([20,50,100,200,300])
c = 0.01 # multiplicative constant
p0s = p - c*np.log(ns_samples)/np.sqrt(ns_samples)
p1s = p + c*np.log(ns_samples)/np.sqrt(ns_samples)

#time grid
ts = np.linspace(0,.25,25)

#simulation parameters
alpha = .05
n_bootstrap = 1000
n_tests = 100


#computations for power
if __name__ == '__main__':
    pool = Pool()

    ti = time.time()

    powers, lerrs, uerrs = [], [], []
    for n_samples, p0, p1 in zip(ns_samples, p0s, p1s):
        tests = []
        for i in range(n_tests):
            data0, data1 = [samplerER(p0) for _ in range(n_samples)], [samplerER(p0) for _ in range(n_samples)]
            param = [ (d0[1],d1[1],ts) for d0, d1 in zip(data0, data1)]
            HKD0s = np.array(pool.map(pr.HKDs, param))
            data0, data1 = [samplerER(p1) for _ in range(n_samples)], [samplerER(p1) for _ in range(n_samples)]
            param = [ (d0[1],d1[1],ts) for d0, d1 in zip(data0, data1)]
            HKD1s = np.array(pool.map(pr.HKDs, param))
            tests.append(pr.test(HKD0s, HKD1s, alpha, n_bootstrap))
        print('n_samples = {} : done'.format(n_samples))
        power, lerr, uerr = pr.tests_conf_interval(tests)
        powers.append(power), lerrs.append(lerr), uerrs.append(uerr)
    errs = np.concatenate((lerrs,uerrs)).reshape((2,len(ns_samples)))
    tf = time.time()
    print('comp time : {:5.3f}'.format((tf-ti)/60))

    #plot
    plt.figure(1), plt.clf()
    plt.plot(ns_samples, powers, '--b')
    plt.errorbar(ns_samples, powers, errs, color='b', fmt='o', capsize=4.)
    plt.xlabel('sample size'), plt.ylabel('power')
    
    #save figure
    if save_fig:
        file = prefix + 'tests_HKD_ER_vs_SBM_Neyman_Pearson_regime'
        plt.savefig(file, transparent=transparent)
        print('    ' + file + ' done')


    print(os.path.basename(__file__) + ' is done')
