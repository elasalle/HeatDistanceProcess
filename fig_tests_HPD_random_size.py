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
# Testing HPD two-sample tests
###############################

##########################################################################
# Disk vs Disk (type I error estimation)

#fix a seed for reproducibility
np.random.seed(1234)

#graphs parameters
n = 50
p = .5
def samplerDisk():
    connect = gr.geometric_graph_annulus(n,p,0, weighted=False, s=2, poisson_size=True)[0]
    egelems = pr.laplacian_egelems(connect, nullfirstegv=True)
    return(connect , egelems)
epsilon = 0.5
def samplerAnnulus():
    connect = gr.geometric_graph_annulus(n,p,epsilon, weighted=False, s=2, poisson_size=True)[0]
    egelems = pr.laplacian_egelems(connect, nullfirstegv=True)
    return(connect , egelems)

#time grid
ts = np.linspace(0,.3,10)

#simulation parameters
ns_samples = [25, 50, 100, 150]
alpha = .05
n_bootstrap = 1000
n_tests = 100
#400min


if __name__ == '__main__':
    pool = Pool()

    ti = time.time()

    # typeIs, lerrs, uerrs = [], [], []
    # for n_samples in ns_samples:
    #     tests = []
    #     for i in range(n_tests):
    #         data0, data1 = [samplerDisk() for _ in range(n_samples)], [samplerDisk() for _ in range(n_samples)]
    #         param = [ (d0,d1,ts) for d0, d1 in zip(data0, data1)]
    #         HPD0s = np.array(pool.map(pr.HPDs, param))
    #         data0, data1 = [samplerDisk() for _ in range(n_samples)], [samplerDisk() for _ in range(n_samples)]
    #         param = [ (d0,d1,ts) for d0, d1 in zip(data0, data1)]
    #         HPD1s = np.array(pool.map(pr.HPDs, param))
    #         tests.append(pr.test(HPD0s, HPD1s, alpha, n_bootstrap))
    #     print('n_samples = {} : done'.format(n_samples))
    #     typeI, lerr, uerr = pr.tests_conf_interval(np.array(tests))
    #     typeIs.append(typeI), lerrs.append(lerr), uerrs.append(uerr)
    # errs = np.concatenate((lerrs,uerrs)).reshape((2,len(ns_samples)))
    # tf = time.time()
    # print('comp time : {:5.3f}'.format((tf-ti)/60))
    #
    # #plot
    # plt.figure(1), plt.clf()
    # plt.plot([ns_samples[0], ns_samples[-1]], [alpha,alpha], '--r')
    # plt.plot(ns_samples, typeIs, '--b')
    # plt.errorbar(ns_samples, typeIs, errs, color='b', fmt='o', capsize=4.)
    # plt.xlabel('sample size'), plt.ylabel('level')
    #
    # #save figure
    # if save_fig:
    #     file = prefix + 'tests_HPD_Disk_rdm_size_typeI_error'
    #     plt.savefig(file, transparent=transparent)
    #     print(file + ' done')




    ##########################################################################
    # Disk vs Annulus (power estimation)
    ti = time.time()

    powers, lerrs, uerrs = [], [], []
    for n_samples in ns_samples:
        tests = []
        for i in range(n_tests):
            data0, data1 = [samplerDisk() for _ in range(n_samples)], [samplerDisk() for _ in range(n_samples)]
            param = [ (d0,d1,ts) for d0, d1 in zip(data0, data1)]
            HPD0s = np.array(pool.map(pr.HPDs, param))
            data0, data1 = [samplerDisk() for _ in range(n_samples)], [samplerAnnulus() for _ in range(n_samples)]
            param = [ (d0,d1,ts) for d0, d1 in zip(data0, data1)]
            HPD1s = np.array(pool.map(pr.HPDs, param))
            tests.append(pr.test(HPD0s, HPD1s, alpha, n_bootstrap))
        print('n_samples = {} : done'.format(n_samples))
        power, lerr, uerr = pr.tests_conf_interval(np.array(tests))
        powers.append(power), lerrs.append(lerr), uerrs.append(uerr)
    errs = np.concatenate((lerrs,uerrs)).reshape((2,len(ns_samples)))
    tf = time.time()
    print('comp time : {:5.3f}'.format((tf-ti)/60))

    #plot
    plt.figure(2), plt.clf()
    plt.plot(ns_samples, powers, '--b')
    plt.errorbar(ns_samples, powers, errs, color='b', fmt='o', capsize=4.)
    plt.xlabel('samples size'), plt.ylabel('power')

    #save figure
    if save_fig:
        file = prefix + 'tests_HPD_Disk_vs_Annulus_rdm_size_power'
        plt.savefig(file, transparent=transparent)
        print(file + ' done')

    print(os.path.basename(__file__) + ' is done')
