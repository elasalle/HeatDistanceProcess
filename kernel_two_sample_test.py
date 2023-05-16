import numpy as np
import grakel.kernels as kern
from grakel import Graph
from sys import stdout
from sklearn.metrics import pairwise_kernels


def compute_two_sample_kernels(data1, data2, kernel):
    #K11 = kernel.fit_transform(data1)
    #K12 = kernel.transform(data2)
    #K22 = kernel.fit_transform(data2)
    #K = np.concatenate( (np.concatenate((K11, K12), axis=1), np.concatenate((K12.transpose(), K22), axis=1)), axis=0)
    K = kernel.fit_transform(data1+data2)
    return K

#inspired by

def MMD2u(K, m, n):
    """The MMD^2_u unbiased statistic.
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
        1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
        2.0 / (m * n) * Kxy.sum()


def compute_null_distribution(K, m, n, iterations=10000, verbose=False,
                              random_state=None, marker_interval=1000):
    """Compute the bootstrap null-distribution of MMD2u.
    """
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        if verbose and (i % marker_interval) == 0:
            print(i),
            stdout.flush()
        idx = rng.permutation(m+n)
        K_i = K[idx, idx[:, None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)

    if verbose:
        print("")

    return mmd2u_null


def kernel_two_sample_test(X, Y, kernel, alpha=.05, iterations=int(1e4),
                           verbose=False, random_state=None, **kwargs):
    """Compute MMD^2_u, its null distribution and the p-value of the
    kernel two-sample test.

    Note that extra parameters captured by **kwargs will be passed to
    pairwise_kernels() as kernel parameters. E.g. if
    kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1),
    then this will result in getting the kernel through
    kernel_function(metric='rbf', gamma=0.1).
    """
    m = len(X)
    n = len(Y)
    #print('compute kernels')
    K = compute_two_sample_kernels(X, Y, kernel)
    mmd2u = MMD2u(K, m, n)
    if verbose:
        print("MMD^2_u = %s" % mmd2u)
        print("Computing the null distribution.")

    #print('compute null stat')
    mmd2u_null = compute_null_distribution(K, m, n, iterations,
                                           verbose=verbose,
                                           random_state=random_state)
    p_value = max(1.0/iterations, (mmd2u_null > mmd2u).sum() /
                  float(iterations))

    res = (p_value<alpha)
    if verbose:
        print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0/iterations))

    return res

def get_nonzero_values_of_edge_weights(graphs):
    l = graphs[0].shape[0]
    values = np.array([g[np.triu_indices(l,1)] for g in graphs]).flatten()
    values = values[values.nonzero()]
    return values

def compute_unweighted_grakel(adjs, quantile=.75):
    weight_thresh = np.quantile(get_nonzero_values_of_edge_weights(adjs), quantile)
    graphs = [ Graph(1*(adj>weight_thresh)) for adj in adjs]
    return graphs
