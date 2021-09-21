# -*- coding: utf-8 -*-
"""
Created on Wed May 12 11:12:33 2021

@author: Etienne Lasalle
"""

#imports
import numpy as np
import gudhi as gd
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix
from scipy.stats import norm

import extended_persistence as pers



def laplacian_egelems(connect, nullfirstegv=True):
    """compute the eigen elements of the Laplacian matrix.

    Parameters
    ----------
    connect : (N,N) array_like
        adjacency matrix of the unweighted graph
    nullfirstegv : bool, default: True
        if True, the smallest eigenvalue is set to exactly zero.

    Returns
    -------
    egelems : list of length 2,
        The first element is the (N) array of eigenvalues.
        The second element is the (N,N) array of corresponding eigenvectors, put in columns.
    """
    egvals, egvects = np.linalg.eigh(laplacian(connect, normed=False))
    if nullfirstegv:
        egvals[0] = 0.
    egelems = [egvals, egvects]
    return egelems


#heat diffusion tools
def heat_kernel(egelems, time):
    """Compute the heat kernel at a given time using the eigen-elements of the Laplacian matrix."""
    egvals, egvects = egelems[0], egelems[1]
    e = np.diag(np.exp(-time*egvals))
    return egvects @ e @ egvects.transpose()

def hks(egelems, time):
    """compute the Heat Kernel Signature at a given time using the eigen-elements of the Laplacian matrix."""
    egvals, egvects = egelems[0], egelems[1]
    signature = np.square(egvects).dot(np.diag(np.exp(-time * egvals))).sum(axis=1)
    return signature

def HKD(egelems0, egelems1 , time):
    """compute the Heat Kernel Distance at a given time using the eigen-elements of the two Laplacian matrices."""
    hk0 = heat_kernel(egelems0, time)
    hk1 = heat_kernel(egelems1, time)
    #K0, K1 = len(egvals0)*hk0, len(egvals1)*hk1
    dist = np.linalg.norm(hk0-hk1, ord = 'fro')
    return dist

def HKDs(params):
    """compute the Heat Kernel Distances at a given times using the eigen-elements of the two Laplacian matrices.
    Used for parallelization.

    Parameters
    ----------
    params : tuple containing the paramters (egelems0, egelems1, ts)
        egelemes0 : eigen-elements of the first Laplacian matrix
        egelemes1 : eigen-elements of the second Laplacian matrix
        ts : array, containing the time at which the HKD must be computed.

    Returns
    -------
    dists : list
        The HKD values.
    """
    egelems0, egelems1 , ts = params
    dists = [ HKD(egelems0, egelems1 , t) for t in ts ]
    return dists

def HPD(data0, data1 , t, e=0):
    """compute the Heat Persistence Distance at a given time using the eigen-elements of the two Laplacian matrices."""
    hks0, hks1 = hks(data0[1],t) , hks(data1[1],t)
    base0, base1 = pers.get_base_simplex_from_adjacency(data0[0]), pers.get_base_simplex_from_adjacency(data1[0])
    dist =  pers.ext_bottleneck(pers.ext_pers(data0[0], hks0, base0), pers.ext_pers(data1[0], hks1, base1), e)
    return dist

def HPDs(params):
    """compute the Heat Persistence Distances at a given times using the eigen-elements of the two Laplacian matrices.
    Used for parallelization.

    Parameters
    ----------
    params : tuple containing the paramters (egelems0, egelems1, ts)
        egelemes0 : eigen-elements of the first Laplacian matrix
        egelemes1 : eigen-elements of the second Laplacian matrix
        ts : array, containing the time at which the HPD must be computed.

    Returns
    -------
    dists : list
        The HPD values.
    """
    if len(params)==3:
        data0, data1, ts = params
        e = 0
    else:
        data0, data1, ts, e = params
    base0, base1 = pers.get_base_simplex_from_adjacency(data0[0]), pers.get_base_simplex_from_adjacency(data1[0])
    dists = [ pers.ext_bottleneck(pers.ext_pers(data0[0],hks(data0[1],t),base0), pers.ext_pers(data1[0],hks(data1[1],t), base1), e)    for t in ts ]
    return dists





#statistical procedures
def confidence_band(dists, alpha, n_bootstrap):
    """Compute a confidence band around the empirical mean.

    Parameters
    ----------
    dist : (N,M) array_like
        the N distances functions computed at M times.
    alpha : float, between 0 and 1
        the desired level of the confidence band. For example 0.05.
    n_bootstrap : int
        number of bootstrap samples to be drawn.

    Returns
    -------
    mean_dists : (M,) numpy.array
        the empirical mean of the distances.
    l_conf : (M,) numpy.array
        the lower part of the confidence band.
    u_conf : (M,) numpy.array
        the upper part of the confidence band.
    """

    N = dists.shape[0]
    mean_dists = np.mean(dists, axis=0)
    dists_centered = dists - mean_dists

    ind = np.random.choice(np.arange(N), n_bootstrap*N, True)
    d, indptr, shape = np.ones(n_bootstrap*N), np.arange(n_bootstrap+1)*N, (n_bootstrap,N)
    resamp = csr_matrix((d, ind, indptr), shape)
    mean_boot = 1/np.sqrt(N)*csr_matrix.dot(resamp, dists_centered)
    maxs = np.max(np.abs(mean_boot), axis=1)
    Z = np.quantile(maxs, q = 1 - alpha , interpolation='higher')
    l_conf, h_conf = mean_dists-Z/np.sqrt(N), mean_dists+Z/np.sqrt(N)
    return mean_dists, l_conf, h_conf

def test(dists1, dists2, alpha, n_bootstrap):
    """Perform a two-sample test.

    Parameters
    ----------
    dist1 : (N1,M) array_like
        the first set of N1 distances functions computed at M times.
    dist2 : (N2,M) array_like
        the second set of N2 distances functions computed at M times.
    alpha : float, between 0 and 1
        the desired level of the test. For example 0.05.
    n_bootstrap : int
        number of bootstrap samples to be drawn.

    Returns
    -------
    null_rejected : bool,
        True, if the null hypothesis is rejected.
    """
    #distsi must be of shape (N_i,n_times)    ; n_times needs to be a common size
    N1, N2 = dists1.shape[0], dists2.shape[0]
    N = N1+N2
    coeff = np.sqrt(N1*N2/(N))

    #compute the test statistic
    mean1, mean2 = np.mean(dists1, axis=0), np.mean(dists2, axis=0)
    D = coeff*np.max(np.abs(mean1-mean2))

    #compute critical value
    dists = np.concatenate((dists1, dists2), axis=0)
    ind1 = np.random.choice(np.arange(N), n_bootstrap*N1, True)
    ind2 = np.random.choice(np.arange(N), n_bootstrap*N2, True)
    d1, indptr1, shape = np.ones(n_bootstrap*N1), np.arange(n_bootstrap+1)*N1, (n_bootstrap,N)
    d2, indptr2        = np.ones(n_bootstrap*N2), np.arange(n_bootstrap+1)*N2
    resamp1 = csr_matrix((d1, ind1, indptr1), shape)
    resamp2 = csr_matrix((d2, ind2, indptr2), shape)
    mean_boot1 = 1/N1*csr_matrix.dot(resamp1, dists)
    mean_boot2 = 1/N2*csr_matrix.dot(resamp2, dists)
    Dhats = coeff*np.max(np.abs(mean_boot1 - mean_boot2) ,axis=1)
    c = np.quantile(Dhats, q = 1 - alpha , interpolation='higher')

    #conclude if H0 is rejected
    null_rejected = (D > c)
    return null_rejected

def tests_conf_interval(tests, alpha = 0.05):
    """Compute a confidence interval on a sample of Bernoulli variable."""
    n = len(tests)
    p = np.sum(tests)/n
    if p==0:
        lerr = 0
        uerr = 3/n
    elif p==1:
        lerr = 3/n
        uerr = 0
    else:
        lerr = norm.ppf(1-alpha/2)*np.sqrt( p*(1-p) / n  )
        uerr = lerr
    return p, lerr, uerr
