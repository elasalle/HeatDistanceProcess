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


###############################
# Testing HKD two-sample tests
###############################

##########################################################################
# ER vs ER (type I error estimation)

#fix a seed for reproducibility
np.random.seed(1234)

#graphs parameters
n = 50
p = .5
def samplerER():
    connect = nx.to_numpy_array(nx.erdos_renyi_graph(n,p))
    connect = gr.add_weights(connect, 'uniform', distr_param=(0,2))     # add uniform weights in [0,2] on the edges
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
ts = np.linspace(0,.3,25)

#simulation parameters
ns_samples = [25, 50, 100, 200]
alpha = .05
n_bootstrap = 1000
n_tests = 400

#computations of type I error
if __name__ == '__main__':
    pool = Pool()

    ti = time.time()

    typeIs, lerrs, uerrs = [], [], []
    for n_samples in ns_samples:
        tests = []
        for i in range(n_tests):
            data0, data1 = [samplerER() for _ in range(n_samples)], [samplerER() for _ in range(n_samples)]
            param = [ (d0[1],d1[1],ts) for d0, d1 in zip(data0, data1)]
            HKD0s = np.array(pool.map(pr.HKDs, param))
            data0, data1 = [samplerER() for _ in range(n_samples)], [samplerER() for _ in range(n_samples)]
            param = [ (d0[1],d1[1],ts) for d0, d1 in zip(data0, data1)]
            HKD1s = np.array(pool.map(pr.HKDs, param))
            tests.append(pr.test(HKD0s, HKD1s, alpha, n_bootstrap))
        print('    n_samples = {} : done'.format(n_samples))
        typeI, lerr, uerr = pr.tests_conf_interval(np.array(tests))
        typeIs.append(typeI), lerrs.append(lerr), uerrs.append(uerr)
    errs = np.concatenate((lerrs,uerrs)).reshape((2,len(ns_samples)))
    tf = time.time()
    print('    comp time : {:5.3f}'.format((tf-ti)/60))

    #plot
    plt.figure(1), plt.clf()
    plt.plot([ns_samples[0], ns_samples[-1]], [alpha,alpha], '--r')
    plt.plot(ns_samples, typeIs, '--b')
    plt.errorbar(ns_samples, typeIs, errs, color='b', fmt='o', capsize=4.)
    plt.xlabel('sample size'), plt.ylabel('level')
    plt.ylim([0,0.1])
    
    #save figure
    if save_fig:
        file = prefix + 'tests_HKD_ERw_typeI_error'
        plt.savefig(file, transparent=transparent)
        print('    '+file + ' done')




    ##########################################################################
    # ER vs SBM (power estimation)
    ti = time.time()

    powers, lerrs, uerrs = [], [], []
    for n_samples in ns_samples:
        tests = []
        for i in range(n_tests):
            data0, data1 = [samplerER() for _ in range(n_samples)], [samplerER() for _ in range(n_samples)]
            param = [ (d0[1],d1[1],ts) for d0, d1 in zip(data0, data1)]
            HKD0s = np.array(pool.map(pr.HKDs, param))
            data0, data1 = [samplerER() for _ in range(n_samples)], [samplerSBM() for _ in range(n_samples)]
            param = [ (d0[1],d1[1],ts) for d0, d1 in zip(data0, data1)]
            HKD1s = np.array(pool.map(pr.HKDs, param))
            tests.append(pr.test(HKD0s, HKD1s, alpha, n_bootstrap))
        print('    n_samples = {} : done'.format(n_samples))
        power, lerr, uerr = pr.tests_conf_interval(np.array(tests))
        powers.append(power), lerrs.append(lerr), uerrs.append(uerr)
    errs = np.concatenate((lerrs,uerrs)).reshape((2,len(ns_samples)))
    tf = time.time()
    print('    comp time : {:5.3f}'.format((tf-ti)/60))

    #plot
    plt.figure(2), plt.clf()
    plt.plot(ns_samples, powers, '--b')
    plt.errorbar(ns_samples, powers, errs, color='b', fmt='o', capsize=4.)
    plt.xlabel('sample size'), plt.ylabel('power')
    
    #save figure
    if save_fig:
        file = prefix + 'tests_HKD_ERvsSBMw_power'
        plt.savefig(file, transparent=transparent)
        print('    '+file + ' done')

    print(os.path.basename(__file__) + ' is done')
