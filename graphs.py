# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:19:36 2021

@author: Etienne Lasalle
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx
from scipy.spatial.distance import pdist, squareform


def add_weights(connect, distr='uniform', distr_param=None):
    """Add independent weights to an unweighted graph.

    Parameters
    ----------
    connect : (N,N) array_like
        adjacency matrix of the unweighted graph
    distr : string, default: 'uniform'
        the distribution of each weights. Can be either 'uniform' or 'gauss_abs' (i.e. absolute value of a Gaussian variable).
    distr_param : tuple of length 2 or None, default : None
        if distr='uniform', distr_param=(min,max) such that weights are drawn uniformly in [min, max]; if None min=0 and max=2.
        if distr='gauss_abs', distr_maram=(m,s) such that weights are drawn as the absolute value of a Gaussian variable of mean s and standard deviation s; if None m=0 and s=1.

    Returns
    -------
    connect : (N,N) array
        the weight matrix of the weighted graph.
        [description]
    """

    r,c = np.where(np.triu(connect)!=0) #collect the edges
    n_edges = len(r)
    if distr=='uniform':
        if distr_param is None:
            wmin, wmax = 0, 2
        else:
            wmin, wmax = distr_param
        weights = np.random.rand(n_edges)*(wmax-wmin)+wmin
    if distr=='gauss_abs':
        if distr_param is None:
            m,s = 0,1
        else:
            m,s = distr_param
        weights = np.abs(np.random.normal(m,s,n_edges))
    connect[r,c] = weights
    connect[c,r] = weights
    return connect





def geometric_graph_annulus(n,p,epsilon, weighted=False, s=2, poisson_size=False):
    """
    Return a geometric graph where points are taken uniformly on the annulus of inner radius epsilon and outer radius 1.
    The distance threshold is chosen so that a proportion p of all the possible edges will be actual edges.
    Weights depending on the distance between nodes can be added.

    Parameters
    ----------
    n : int
        The number of nodes in the graph
        Or parameter in the Poisson distribution, if poisson_size=True.
    p : float
        Proportion of edges to include in the graph.
        Should be between 0 and 1.
    epsilon : float
        Inner radius of the annulus.
        Should be between 0 and 1.
    weighted : bool, default: True
        edges are weighted with weights equal to exp(-s*dist) when dist is the distance between nodes
    s : float, default: 2.
        coefficient determining the decay of the weights with respect to the distance between nodes
    poisson_size : bool, default : False
        If True, the graph has size following a Poisson distribution of parameter n.

    Returns
    -------

    G : (N,N) numpy.array
        The adjacency (or weight) matrix of the generated graph.
    """

    #define the graph size
    if poisson_size:
        size = np.random.poisson(n)
    else:
        size = n
    #random polar coordinates
    theta = 2*np.pi*np.random.rand(size)
    r = np.sqrt((1-epsilon**2)*np.random.rand(size)+epsilon**2) #the square root ensure that point are taken uniformly on the annulus
    coord = np.array([r*np.cos(theta), r*np.sin(theta)]).transpose()

    #computation and ordering of the distances
    dists = pdist(coord)
    ordered_dists = np.sort(dists)[size:] #'n:' remove the zeros coming from the diagonal
    T = ordered_dists[int(p * size*(size-1)/2)] #select the threshold distance to have a proportion p of all possible edges

    #creation of the adjacency matrix
    G = 1.*(squareform(dists)<=T)
    np.fill_diagonal(G,0)
    if weighted:
        G = G*np.exp(-s*squareform(dists))
    return G, coord
