# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:15:57 2019

@author: Etienne Lasalle
"""

#imports
import numpy as np
import gudhi as gd
from itertools import combinations


################
    #PERSISTENCE
################

#Transform the adjacency matrix into a simplicial complex of dim 1.
def get_base_simplex_from_adjacency(A):
    """create the simplicial complex associted to the adjacency matrix A."""
    num_vertices = A.shape[0]
    st = gd.SimplexTree()
    [st.insert([i], filtration=-1e10) for i in range(num_vertices)]
    for i, j in combinations(range(num_vertices), r=2):
        if A[i, j] > 0:
            st.insert([i, j], filtration=-1e10)
    basesimplex =  list(st.get_filtration())
    return basesimplex

def ext_pers(A, filtration_val, basesimplex):
    """compute the extended persistence diagrams for a graph equipped with a function on its vertices.

    Parameters
    ----------
    A : (N,N) array_like
        the adjacency of the (weighted) graph
    filtration_val : (N,) array_like
        array containing the filtration values on the num_vertices
    basesimplex : list
        list containing the simplex encoding the graph. Can be computed with the function get_base_simplex_from_adjacency

    Returns
    -------
    dgmOrd0, dgmExt0, dgmRel1, dgmExt1 : tuple of diagrams
        The four extended persistence diagrams as arrays with two columns. Each line encodes the coordinates of a point in a diagram.
    """

    num_vertices = A.shape[0]
    (xs, ys) = np.where(np.triu(A))
    num_edges = len(xs)

    if len(filtration_val.shape) == 1:
        min_val, max_val = filtration_val.min(), filtration_val.max()
    else:
        min_val = min([filtration_val[xs[i], ys[i]] for i in range(num_edges)])
        max_val = max([filtration_val[xs[i], ys[i]] for i in range(num_edges)])

    st = gd.SimplexTree()
    st.set_dimension(2)

    for simplex, filt in basesimplex:
        st.insert(simplex=simplex + [-2], filtration=-3)

    if len(filtration_val.shape) == 1:
        if max_val == min_val:
            fa = -.5 * np.ones(filtration_val.shape)
            fd = .5 * np.ones(filtration_val.shape)
        else:
            fa = -2 + (filtration_val - min_val) / (max_val - min_val) #shift filt values in [-2, -1]
            fd = 2 - (filtration_val - min_val) / (max_val - min_val)  #shift and flip filt values in [1,2]
        for vid in range(num_vertices):
            st.assign_filtration(simplex=[vid], filtration=fa[vid])
            st.assign_filtration(simplex=[vid, -2], filtration=fd[vid])
    else:
        if max_val == min_val:
            fa = -.5 * np.ones(filtration_val.shape)
            fd = .5 * np.ones(filtration_val.shape)
        else:
            fa = -2 + (filtration_val - min_val) / (max_val - min_val)
            fd = 2 - (filtration_val - min_val) / (max_val - min_val)
        for eid in range(num_edges):
            vidx, vidy = xs[eid], ys[eid]
            st.assign_filtration(simplex=[vidx, vidy], filtration=fa[vidx, vidy])
            st.assign_filtration(simplex=[vidx, vidy, -2], filtration=fd[vidx, vidy])
        for vid in range(num_vertices):
            if len(np.where(A[vid, :] > 0)[0]) > 0:
                st.assign_filtration(simplex=[vid], filtration=min(fa[vid, np.where(A[vid, :] > 0)[0]]))
                st.assign_filtration(simplex=[vid, -2], filtration=min(fd[vid, np.where(A[vid, :] > 0)[0]]))

    st.make_filtration_non_decreasing()
    distorted_dgm = st.persistence()
    normal_dgm = dict()
    normal_dgm["Ord0"], normal_dgm["Rel1"], normal_dgm["Ext0"], normal_dgm["Ext1"] = [], [], [], []
    for point in range(len(distorted_dgm)):
        dim, b, d = distorted_dgm[point][0], distorted_dgm[point][1][0], distorted_dgm[point][1][1]
        pt_type = "unknown"
        if (-2 <= b <= -1 and -2 <= d <= -1) or (b == -.5 and d == -.5):
            pt_type = "Ord" + str(dim)
        if (1 <= b <= 2 and 1 <= d <= 2) or (b == .5 and d == .5):
            pt_type = "Rel" + str(dim)
        if (-2 <= b <= -1 and 1 <= d <= 2) or (b == -.5 and d == .5):
            pt_type = "Ext" + str(dim)
        if np.isinf(d):
            continue
        else:
            b, d = min_val + (2 - abs(b)) * (max_val - min_val), min_val + (2 - abs(d)) * (max_val - min_val)
            if b <= d:
                normal_dgm[pt_type].append(tuple([distorted_dgm[point][0], tuple([b, d])]))
            else:
                normal_dgm[pt_type].append(tuple([distorted_dgm[point][0], tuple([d, b])]))

    dgmOrd0 = np.array([normal_dgm["Ord0"][point][1] for point in range(len(normal_dgm["Ord0"]))])
    dgmExt0 = np.array([normal_dgm["Ext0"][point][1] for point in range(len(normal_dgm["Ext0"]))])
    dgmRel1 = np.array([normal_dgm["Rel1"][point][1] for point in range(len(normal_dgm["Rel1"]))])
    dgmExt1 = np.array([normal_dgm["Ext1"][point][1] for point in range(len(normal_dgm["Ext1"]))])
    if dgmOrd0.shape[0] == 0:
        dgmOrd0 = np.zeros([0, 2])
    if dgmExt1.shape[0] == 0:
        dgmExt1 = np.zeros([0, 2])
    if dgmExt0.shape[0] == 0:
        dgmExt0 = np.zeros([0, 2])
    if dgmRel1.shape[0] == 0:
        dgmRel1 = np.zeros([0, 2])
    return dgmOrd0, dgmExt0, dgmRel1, dgmExt1

def ext_bottleneck(dgms1, dgms2, e=0):
    """Compute the maximum of the four bottleneck distances, associated to each type of extended persistence diagrams."""
    dist = np.max([ gd.bottleneck_distance(dgms1[k] , dgms2[k], e) for k in range(4) ])
    return dist
