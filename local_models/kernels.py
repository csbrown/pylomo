import numpy as np
import sklearn.neighbors
import sklearn.pipeline
import sklearn.svm
import sklearn.decomposition
import sklearn.gaussian_process
import logging
import pickle
import joblib
import time
import heapq
import inspect
from . import loggin
from . import TLS_models
import functools
import collections
import scipy

try:
    sklearn.neighbors.ball_tree.VALID_METRICS.append("KernelDistance")
except:
    sklearn.neighbors._ball_tree.VALID_METRICS.append("KernelDistance")

def pairwise_sample(X, n=10000):
    pair_indices = np.random.choice(np.arange(X.shape[0]), size=(n*2,2))
    good_indices = pair_indices[:,0] != pair_indices[:,1]
    pair_indices = pair_indices[good_indices]
    return X[pair_indices[:,0]], X[pair_indices[:,1]]
    
def pairwise_dd(X, metric="euclidean", n=10000):
    x1, x2 = pairwise_sample(X,n)
    dd = np.sqrt((x1 - x2)**2) #this is unfortunate
    return np.mean(dd), np.std(dd)

def kernel_dist(kernel, x1, x2):
    x1, x2 = tuple(map(np.atleast_2d, [x1,x2]))
    return np.sqrt(self.kernel(x1)[:1].reshape(-1) - 2*self.kernel(x1,x2).reshape(-1) + self.kernel(x2)[:1].reshape(-1)).reshape((-1,1))

def KernelDistance(kernel):
    return sklearn.neighbors.dist_metrics.PyFuncDistance(functools.partial(kernel_dist, kernel))

class UnaryKernel(object):
    '''
        Base class for unary kernels (kernels that accept a pre-computed "distance" as input)
        inputs:
            bandwidth: float or callable that accepts a set of distances and returns a float
            k (optional): int for knn bandwidth iff bandwidth == "knn"
        children should implement:
            __call__: takes in a collection of distances and returns the kernel function evaluated at those values
            support_radius: takes in nothing and returns the support radius of this kernel
            d (optional): the derivative of the kernel function w.r.t. the input "distances"
    '''
    def __init__(self, bandwidth=None, k=None):
        if bandwidth == "knn":
            bandwidth = self.knn_bandwidth
        self.bandwidth = bandwidth
        self.k = k

    def knn_bandwidth(self, x, k=None):
        ''' Computes the bandwidth based on the kth-largest member of x '''
        if k is None:
            k = self.k
        if x.shape[0] > k:
            k_smallest_distances = heapq.nsmallest(k, x)
        else:
            k_smallest_distances = x
        return np.max(k_smallest_distances)

    def apply_bandwidth(self, x):
        ''' Divides x by the bandwidth '''
        if hasattr(self.bandwidth, "__call__"):
            bandwidth = self.bandwidth(x)
        else:
            bandwidth = self.bandwidth
        return x/bandwidth

    def __str__(self):
        string_contents = []
        string_contents.append(type(self).__name__)
        if hasattr(self.bandwidth, "__call__"):
            string_contents.append(self.bandwidth.__name__)
        else:
            string_contents.append("b{:020.010f}".format(self.bandwidth))
        if self.k is not None:
            string_contents.append("k{:010d}".format(self.k))
        return "_".join(string_contents)

    def __call__(self, x):
        raise NotImplementedError("UnaryKernel subclasses need to implement __call__")
    def support_radius(self):
        raise NotImplementedError("UnaryKernel subclasses need to implement support_radius")

class UniformKernel(UnaryKernel):
    ''' A "square" kernel that is 1 inside the support_radius and 0 else '''
    def __call__(self, x):
        x = self.apply_bandwidth(x)
        answer = np.zeros(x.shape)
        abs_x = np.abs(x)
        answer[abs_x <= 1] = 1
        return answer
    def support_radius(self):
        if hasattr(self.bandwidth, "__call__"):
            return np.inf
        return self.bandwidth

class TriCubeKernel(UnaryKernel):
    ''' A TriCube Kernel https://en.wikipedia.org/wiki/Kernel_%28statistics%29#Kernel_functions_in_common_use '''
    def __call__(self, x):
        x = self.apply_bandwidth(x)
        answer = np.zeros(x.shape)
        abs_x = np.abs(x)
        answer[abs_x < 1] = (70/81)*(1-abs_x[abs_x < 1]**3)**3
        return answer
    def support_radius(self):
        if hasattr(self.bandwidth, "__call__"):
            return np.inf
        return self.bandwidth
    def d(self, x):
        x = self.apply_bandwidth(x)
        answer = np.zeros(x.shape)
        abs_x = np.abs(x)
        answer[abs_x < 1] = (70/81)*(9)*(1-abs_x[abs_x < 1]**3)**2*(x[abs_x < 1]**2)/self.bandwidth*((x[abs_x < 1] < 0)*2 - 1)
        return answer

class GaussianKernel(UnaryKernel):
    ''' A Gaussian Kernel https://en.wikipedia.org/wiki/Kernel_%28statistics%29#Kernel_functions_in_common_use '''
    def __call__(self, x):
        x = self.apply_bandwidth(x)
        return scipy.stats.norm.pdf(x)
    def support_radius(self):
        return np.inf
    def d(self, x):
        x = self.apply_bandwidth(x)
        return -x*scipy.stats.norm.pdf(x)/self.bandwidth
