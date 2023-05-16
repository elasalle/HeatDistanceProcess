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
from grakel import Graph
from grakel import GraphletSampling, WeisfeilerLehman, ShortestPath, SubgraphMatching, RandomWalk, VertexHistogram, SvmTheta, NeighborhoodHash

import graphs as gr
import processes as pr
import extended_persistence as pers
from kernel_two_sample_test import kernel_two_sample_test

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

n = 50
p = .5
def samplerER():
    connect = nx.to_numpy_array(nx.erdos_renyi_graph(np.random.poisson(n),p))
    egelems = pr.laplacian_egelems(connect, nullfirstegv=True)
    return(connect , egelems)

n = 50
ps = [[.75,.25],[.25,.75]]
def samplerSBM():
    size = np.random.poisson(n)
    ns = [size//2, size//2]
    connect = nx.to_numpy_array(nx.generators.community.stochastic_block_model(ns,ps))
    egelems = pr.laplacian_egelems(connect, nullfirstegv=True)
    return(connect , egelems)

#time grid
ts = np.linspace(0,.3,10)

#simulation parameters
ns_samples = [25, 50, 100, 150]
alpha = .05
n_bootstrap = 1000
n_tests = 50
#400min

#whether we test with geometric graphs or ER-SBM graphs
#graph_distr = "geometric"
graph_distr = "ER-SBM"

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
    t_HPD, t_GS, t_WL, t_SP, t_VH, t_ST, t_NH, t_RW, t_SM = 0, 0, 0, 0, 0, 0, 0, 0, 0
    powers, lerrs, uerrs = [], [], []
    powers_GS, lerrs_GS, uerrs_GS = [], [], []
    powers_WL, lerrs_WL, uerrs_WL = [], [], []
    powers_SP, lerrs_SP, uerrs_SP = [], [], []
    powers_VH, lerrs_VH, uerrs_VH = [], [], []
    powers_ST, lerrs_ST, uerrs_ST = [], [], []
    powers_NH, lerrs_NH, uerrs_NH = [], [], []
    powers_RW, lerrs_RW, uerrs_RW = [], [], []
    #powers_SM, lerrs_SM, uerrs_SM = [], [], []
    for n_samples in ns_samples:
        tests, tests_GS, tests_WL, tests_SP, tests_VH, tests_ST, tests_NH, tests_RW, tests_SM = [], [], [], [], [], [], [], [], []
        for i in range(n_tests):
            if graph_distr=="geometric":
                data0, data1 = [samplerDisk() for _ in range(n_samples)], [samplerDisk() for _ in range(n_samples)]
            elif graph_distr=="ER-SBM":
                data0, data1 = [samplerER() for _ in range(n_samples)], [samplerER() for _ in range(n_samples)]
            param = [ (d0,d1,ts) for d0, d1 in zip(data0, data1)]
            graphsDisk, graphsDisk_w_labels = [ Graph(data[0]) for data in data0 ],  [ Graph(data[0], {i:str(np.sum(data[0][i,:])) for i in range(data[0].shape[0])}) for data in data0 ]
            graphsDisk_w_bothlabels = [ Graph(data[0], {i:str(np.sum(data[0][i,:])) for i in range(data[0].shape[0])}, {(i,j):"1" for i in range(data[0].shape[0]) for j in range(data[0].shape[0])}) for data in data0 ]
            ti_HPD = time.time()
            HPD0s = np.array(pool.map(pr.HPDs, param))
            t_HPD += (time.time()-ti_HPD)

            if graph_distr=="geometric":
                data0, data1 = [samplerDisk() for _ in range(n_samples)], [samplerAnnulus() for _ in range(n_samples)]
            elif graph_distr=="ER-SBM":
                data0, data1 = [samplerER() for _ in range(n_samples)], [samplerSBM() for _ in range(n_samples)]
            graphsAnnulus, graphsAnnulus_w_labels = [ Graph(data[0]) for data in data1 ], [ Graph(data[0], {i:str(np.sum(data[0][i,:])) for i in range(data[0].shape[0])}) for data in data1 ]
            graphsAnnulus_w_bothlabels = [ Graph(data[0], {i:str(np.sum(data[0][i,:])) for i in range(data[0].shape[0])}, {(i,j):"1" for i in range(data[0].shape[0]) for j in range(data[0].shape[0])}) for data in data1 ]
            param = [ (d0,d1,ts) for d0, d1 in zip(data0, data1)]
            ti_HPD = time.time()
            HPD1s = np.array(pool.map(pr.HPDs, param))
            tests.append(pr.test(HPD0s, HPD1s, alpha, n_bootstrap))
            t_HPD += (time.time()-ti_HPD)

            ti_GS = time.time()
            tests_GS.append(kernel_two_sample_test(graphsDisk, graphsAnnulus, GraphletSampling(k=4, sampling={'n_samples':50}), alpha, n_bootstrap))
            t_GS += (time.time()-ti_GS)
            ti_WL = time.time()
            tests_WL.append(kernel_two_sample_test(graphsDisk_w_labels, graphsAnnulus_w_labels, WeisfeilerLehman(), alpha, n_bootstrap))
            t_WL += (time.time()-ti_WL)
            ti_SP = time.time()
            tests_SP.append(kernel_two_sample_test(graphsDisk, graphsAnnulus, ShortestPath(with_labels=False), alpha, n_bootstrap))
            t_SP += (time.time()-ti_SP)
            ti_VH = time.time()
            tests_VH.append(kernel_two_sample_test(graphsDisk_w_labels, graphsAnnulus_w_labels, VertexHistogram(), alpha, n_bootstrap))
            t_VH += (time.time()-ti_VH)
            ti_ST = time.time()
            tests_ST.append(kernel_two_sample_test(graphsDisk, graphsAnnulus, SvmTheta(), alpha, n_bootstrap))
            t_ST += (time.time()-ti_ST)
            ti_NH = time.time()
            tests_NH.append(kernel_two_sample_test(graphsDisk_w_labels, graphsAnnulus_w_labels, NeighborhoodHash(), alpha, n_bootstrap))
            t_NH += (time.time()-ti_NH)
            ti_RW = time.time()
            tests_RW.append(kernel_two_sample_test(graphsDisk, graphsAnnulus, RandomWalk(), alpha, n_bootstrap))
            t_RW += (time.time()-ti_RW)
            #ti_SM = time.time()
            #tests_SM.append(kernel_two_sample_test(graphsDisk_w_bothlabels, graphsAnnulus_w_bothlabels, SubgraphMatching(k=3), alpha, n_bootstrap))
            #t_SM += (time.time()-ti_SM)

            print('tests {}/{}'.format(i+1,n_tests))

        print('    n_samples = {} : done'.format(n_samples))
        power, lerr, uerr = pr.tests_conf_interval(np.array(tests))
        powers.append(power), lerrs.append(lerr), uerrs.append(uerr)
        power_GS, lerr_GS, uerr_GS = pr.tests_conf_interval(np.array(tests_GS))
        powers_GS.append(power_GS), lerrs_GS.append(lerr_GS), uerrs_GS.append(uerr_GS)
        power_WL, lerr_WL, uerr_WL = pr.tests_conf_interval(np.array(tests_WL))
        powers_WL.append(power_WL), lerrs_WL.append(lerr_WL), uerrs_WL.append(uerr_WL)
        power_SP, lerr_SP, uerr_SP = pr.tests_conf_interval(np.array(tests_SP))
        powers_SP.append(power_SP), lerrs_SP.append(lerr_SP), uerrs_SP.append(uerr_SP)
        power_VH, lerr_VH, uerr_VH = pr.tests_conf_interval(np.array(tests_VH))
        powers_VH.append(power_VH), lerrs_VH.append(lerr_VH), uerrs_VH.append(uerr_VH)
        power_ST, lerr_ST, uerr_ST = pr.tests_conf_interval(np.array(tests_ST))
        powers_ST.append(power_ST), lerrs_ST.append(lerr_ST), uerrs_ST.append(uerr_ST)
        power_NH, lerr_NH, uerr_NH = pr.tests_conf_interval(np.array(tests_NH))
        powers_NH.append(power_NH), lerrs_NH.append(lerr_NH), uerrs_NH.append(uerr_NH)
        power_RW, lerr_RW, uerr_RW = pr.tests_conf_interval(np.array(tests_RW))
        powers_RW.append(power_RW), lerrs_RW.append(lerr_RW), uerrs_RW.append(uerr_RW)
        #power_SM, lerr_SM, uerr_SM = pr.tests_conf_interval(np.array(tests_SM))
        #powers_SM.append(power_SM), lerrs_SM.append(lerr_SM), uerrs_SM.append(uerr_SM)

    errs = np.concatenate((lerrs,uerrs)).reshape((2,len(ns_samples)))
    errs_GS = np.concatenate((lerrs_GS,uerrs_GS)).reshape((2,len(ns_samples)))
    errs_WL = np.concatenate((lerrs_WL,uerrs_WL)).reshape((2,len(ns_samples)))
    errs_SP = np.concatenate((lerrs_SP,uerrs_SP)).reshape((2,len(ns_samples)))
    errs_VH = np.concatenate((lerrs_VH,uerrs_VH)).reshape((2,len(ns_samples)))
    errs_ST = np.concatenate((lerrs_ST,uerrs_ST)).reshape((2,len(ns_samples)))
    errs_NH = np.concatenate((lerrs_NH,uerrs_NH)).reshape((2,len(ns_samples)))
    errs_RW = np.concatenate((lerrs_RW,uerrs_RW)).reshape((2,len(ns_samples)))
    #errs_SM = np.concatenate((lerrs_SM,uerrs_SM)).reshape((2,len(ns_samples)))
    tf = time.time()
    print('comp time : {:5.3f}min'.format((tf-ti)/60))
    print('compt time for HPD : {:5.3f}min'.format(t_HPD/60))
    print('compt time for GS : {:5.3f}min'.format(t_GS/60))
    print('compt time for WL : {:5.3f}min'.format(t_WL/60))
    print('compt time for SP : {:5.3f}min'.format(t_SP/60))
    print('compt time for VH : {:5.3f}min'.format(t_VH/60))
    print('compt time for ST : {:5.3f}min'.format(t_ST/60))
    print('compt time for NH : {:5.3f}min'.format(t_NH/60))
    print('compt time for RW : {:5.3f}min'.format(t_RW/60))
    #print('compt time for SM : {:5.3f}min'.format(t_SM/60))

    #plot
    plt.figure(2), plt.clf()
    plt.plot(ns_samples, powers, '--b')
    plt.errorbar(ns_samples, powers, errs, color='b', fmt='o', capsize=4., label='HPD')
    plt.plot(ns_samples, powers_GS, ':', color='darkorange')
    plt.errorbar(ns_samples, powers_GS, errs_GS, color='darkorange', fmt='^', capsize=4., label='GS')
    plt.plot(ns_samples, powers_WL, ':', color='g')
    plt.errorbar(ns_samples, powers_WL, errs_WL, color='g', fmt='X', capsize=4., label='WL')
    plt.plot(ns_samples, powers_SP, ':', color='c')
    plt.errorbar(ns_samples, powers_SP, errs_SP, color='c', fmt='s', capsize=4., label='SP')
    plt.plot(ns_samples, powers_VH, ':', color='tab:olive')
    plt.errorbar(ns_samples, powers_VH, errs_VH, color='tab:olive', fmt='D', capsize=4., label='VH')
    plt.plot(ns_samples, powers_ST, ':', color='tab:brown')
    plt.errorbar(ns_samples, powers_ST, errs_ST, color='tab:brown', fmt='*', capsize=4., label='ST')
    plt.plot(ns_samples, powers_NH, ':', color='m')
    plt.errorbar(ns_samples, powers_NH, errs_NH, color='m', fmt='P', capsize=4., label='NH')
    plt.plot(ns_samples, powers_RW, ':', color='tab:purple')
    plt.errorbar(ns_samples, powers_RW, errs_RW, color='tab:purple', fmt='p', capsize=4., label='RW')
    #plt.plot(ns_samples, powers_SM, ':', color='tab:gray')
    #plt.errorbar(ns_samples, powers_SM, errs_SM, color='tab:gray', fmt='h', capsize=4., label='SM')
    plt.legend()
    plt.xlabel('sample size'), plt.ylabel('power')

    #save figure
    if save_fig:
        if graph_distr=="geometric":
            file = prefix + 'tests_HPD_Disk_vs_Annulus_rdm_size_vs_kernels_power2'
        elif graph_distr=="ER-SBM":
            file = prefix + 'tests_HPD_ER_vs_SBM_rdm_size_vs_kernels_power2'
        plt.savefig(file, transparent=transparent)
        print(file + ' done')

    print(os.path.basename(__file__) + ' is done')
