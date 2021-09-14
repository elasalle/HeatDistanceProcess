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


#%%
#########################
# CONFIDENCE BAND FOR HPD
#########################


#simulation parameters
n_samples = 100
alpha = .05
n_bootstrap = 1000


if __name__ == '__main__':
    pool = Pool()

    ##########################################################################
    #Erdös-Renyi vs SBM graphs

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
    ts = np.linspace(0,.3,50)

    #computations
    ti = time.time()
    data0, data1 = [samplerER() for _ in range(n_samples)], [samplerER() for _ in range(n_samples)]
    param = [ (d0,d1,ts) for d0, d1 in zip(data0, data1)]
    HPDs = np.array(pool.map(pr.HPDs, param))
    meanER, l_confER, h_confER = pr.confidence_band(HPDs, alpha, n_bootstrap)

    data0, data1 = [samplerER() for _ in range(n_samples)], [samplerSBM() for _ in range(n_samples)]
    param = [ (d0,d1,ts) for d0, d1 in zip(data0, data1)]
    HPDs = np.array(pool.map(pr.HPDs, param))
    meanSBM, l_confSBM, h_confSBM = pr.confidence_band(HPDs, alpha, n_bootstrap)
    tf = time.time()
    print((tf-ti)/60)

    #plot
    plt.figure(3), plt.clf()
    plt.fill_between(ts, l_confER, h_confER, color='r', alpha = .3, )
    plt.plot(ts, meanER, 'r', label='ER-ER')
    plt.fill_between(ts, l_confSBM, h_confSBM, color='b', alpha = .3, )
    plt.plot(ts, meanSBM, 'b', label='ER-SBM')
    plt.xlabel('t'), plt.ylabel('HPD')
    plt.legend()

    #save figure
    if save_fig:
        file = prefix + 'conf_band_HPD_ERvsSBM'
        plt.savefig(file, transparent=transparent)
        print('    ' + file + ' done')


    ##########################################################################
    #Geometric graphs on disk vs annulus

    #fix a seed for reproducibility
    np.random.seed(1234)

    #graphs parameters
    n = 50
    p = .5
    def samplerDisk():
        connect = gr.geometric_graph_annulus(n,p,0)[0]
        egelems = pr.laplacian_egelems(connect, nullfirstegv=True)
        return(connect , egelems)
    epsilon = 0.5
    def samplerAnnulus():
        connect = gr.geometric_graph_annulus(n,p,epsilon)[0]
        egelems = pr.laplacian_egelems(connect, nullfirstegv=True)
        return(connect , egelems)

    #time grid
    ts = np.linspace(0,.3,50)

    #computations
    ti = time.time()
    data0, data1 = [samplerDisk() for _ in range(n_samples)], [samplerDisk() for _ in range(n_samples)]
    param = [ (d0,d1,ts) for d0, d1 in zip(data0, data1)]
    HPDs = np.array(pool.map(pr.HPDs, param))
    meanDisk, l_confDisk, h_confDisk = pr.confidence_band(HPDs, alpha, n_bootstrap)

    data0, data1 = [samplerDisk() for _ in range(n_samples)], [samplerAnnulus() for _ in range(n_samples)]
    param = [ (d0,d1,ts) for d0, d1 in zip(data0, data1)]
    HPDs = np.array(pool.map(pr.HPDs, param))
    meanAnnulus, l_confAnnulus, h_confAnnulus = pr.confidence_band(HPDs, alpha, n_bootstrap)
    tf = time.time()
    print((tf-ti)/60)

    #plot
    plt.figure(4), plt.clf()
    plt.fill_between(ts, l_confDisk, h_confDisk, color='r', alpha = .3, )
    plt.plot(ts, meanDisk, 'r', label='Disk-Disk')
    plt.fill_between(ts, l_confAnnulus, h_confAnnulus, color='b', alpha = .3, )
    plt.plot(ts, meanAnnulus, 'b', label='Disk-Annulus')
    plt.xlabel('t'), plt.ylabel('HPD')
    plt.legend()

    #save figure
    file = prefix + 'conf_band_HPD_DiskvsAnnulus'
    plt.savefig(file, transparent=transparent)
    print('    ' + file + ' done')


    ##########################################################################
    #Geometric weighted graphs on disk vs annulus

    #fix a seed for reproducibility
    np.random.seed(1234)

    #graphs parameters
    n = 50
    p = .5
    s = 2           #for the decay of the weights
    def samplerDisk():
        connect = gr.geometric_graph_annulus(n,p,0,True,s)[0]
        egelems = pr.laplacian_egelems(connect, nullfirstegv=True)
        return(connect , egelems)
    epsilon = 0.5
    def samplerAnnulus():
        connect = gr.geometric_graph_annulus(n,p,epsilon, True,s)[0]
        egelems = pr.laplacian_egelems(connect, nullfirstegv=True)
        return(connect , egelems)

    #time grid
    ts = np.linspace(0,1.,50)

    #computations
    ti = time.time()
    data0, data1 = [samplerDisk() for _ in range(n_samples)], [samplerDisk() for _ in range(n_samples)]
    param = [ (d0,d1,ts) for d0, d1 in zip(data0, data1)]
    HPDs = np.array(pool.map(pr.HPDs, param))
    meanDisk, l_confDisk, h_confDisk = pr.confidence_band(HPDs, alpha, n_bootstrap)

    data0, data1 = [samplerDisk() for _ in range(n_samples)], [samplerAnnulus() for _ in range(n_samples)]
    param = [ (d0,d1,ts) for d0, d1 in zip(data0, data1)]
    HPDs = np.array(pool.map(pr.HPDs, param))
    meanAnnulus, l_confAnnulus, h_confAnnulus = pr.confidence_band(HPDs, alpha, n_bootstrap)
    tf = time.time()
    print((tf-ti)/60)

    #plot
    plt.figure(5), plt.clf()
    plt.fill_between(ts, l_confDisk, h_confDisk, color='r', alpha = .3, )
    plt.plot(ts, meanDisk, 'r', label='Disk-Disk')
    plt.fill_between(ts, l_confAnnulus, h_confAnnulus, color='b', alpha = .3, )
    plt.plot(ts, meanAnnulus, 'b', label='Disk-Annulus')
    plt.xlabel('t'), plt.ylabel('HPD')
    plt.legend()

    #save figure
    if save_fig:
        file = prefix + 'conf_band_HPD_DiskvsAnnulus_weighted'
        plt.savefig(file, transparent=transparent)
        print('    ' + file + ' done')


    ##########################################################################
    #Geometric graphs on disk vs annulus with poissonian graph sizes

    #fix a seed for reproducibility
    np.random.seed(1234)

    #graphs parameters
    n = 50
    p = .5
    s = 2           #for the decay of the weights
    def samplerDisk():
        connect = gr.geometric_graph_annulus(n,p,0,False,s,True)[0]
        egelems = pr.laplacian_egelems(connect, nullfirstegv=True)
        return(connect , egelems)
    epsilon = 0.5
    def samplerAnnulus():
        connect = gr.geometric_graph_annulus(n,p,epsilon,False,s,True)[0]
        egelems = pr.laplacian_egelems(connect, nullfirstegv=True)
        return(connect , egelems)

    #time grid
    ts = np.linspace(0,.3,50)

    #computations
    ti = time.time()
    data0, data1 = [samplerDisk() for _ in range(n_samples)], [samplerDisk() for _ in range(n_samples)]
    param = [ (d0,d1,ts) for d0, d1 in zip(data0, data1)]
    HPDs = np.array(pool.map(pr.HPDs, param))
    meanDisk, l_confDisk, h_confDisk = pr.confidence_band(HPDs, alpha, n_bootstrap)

    data0, data1 = [samplerDisk() for _ in range(n_samples)], [samplerAnnulus() for _ in range(n_samples)]
    param = [ (d0,d1,ts) for d0, d1 in zip(data0, data1)]
    HPDs = np.array(pool.map(pr.HPDs, param))
    meanAnnulus, l_confAnnulus, h_confAnnulus = pr.confidence_band(HPDs, alpha, n_bootstrap)
    tf = time.time()
    print((tf-ti)/60)

    #plot
    plt.figure(6), plt.clf()
    plt.fill_between(ts, l_confDisk, h_confDisk, color='r', alpha = .3, )
    plt.plot(ts, meanDisk, 'r', label='Disk-Disk')
    plt.fill_between(ts, l_confAnnulus, h_confAnnulus, color='b', alpha = .3, )
    plt.plot(ts, meanAnnulus, 'b', label='Disk-Annulus')
    plt.xlabel('t'), plt.ylabel('HPD')
    plt.legend()

    #save figure
    if save_fig:
        file = prefix + 'conf_band_HPD_DiskvsAnnulus_rdmsize'
        plt.savefig(file, transparent=transparent)
        print('    ' + file + ' done')


    print(os.path.basename(__file__) + ' is done')
