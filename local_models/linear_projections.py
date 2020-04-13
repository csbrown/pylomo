import numpy as np
import matplotlib.pyplot as plt
from ml_battery.utils import cmap
import os
from . import local_models
from .utils import *

import logging
import ml_battery.log
logger = logging.getLogger(__name__)

def min_max_range(data):
    min_nm, max_nm = np.min(data, axis=0), np.max(data, axis=0)
    return min_nm, max_nm, max_nm-min_nm

def transformate_data(data, kernel, linear_models, r=None, k=None):
    model_dimensions = linear_models.model.m
    data_dimensions = data.shape[1]
    data_n = data.shape[0]
    if r is not None:
        linear_params_cb = linear_models.transform(data,r=r,weighted=True,kernel=kernel)
    elif k is not None:
        linear_params_cb = linear_models.transform(data,k=k,weighted=True,kernel=kernel)
    linear_params_vecs = linear_params_cb[:,:data_dimensions*model_dimensions].reshape((data_n, model_dimensions, data_dimensions))
    linear_params_mean = linear_params_cb[:,data_dimensions*model_dimensions:]
    return linear_params_vecs, linear_params_mean
    
def make_local_odr_lines(linear_params_vecs, linear_params_means, spans):
    lines = sublinear_spaces(linear_params_vecs, spans) + linear_params_means
    return lines
    
def plt_local_odr_lines(lines):
    for i in range(lines.shape[1]):
        plt.plot(lines[:,i,0], lines[:,i,1])

def plt_local_odr_lines_3d(lines, ax):
    for i in range(lines.shape[1]):
        ax.plot(lines[:,i,0], lines[:,i,1], lines[:,i,2])

def plt_local_odr_surfaces_3d(lines, ax):
    for i in range(lines.shape[1]):
        xx = lines[:,i,0].reshape((2,2))
        yy = lines[:,i,1].reshape((2,2))
        zz = lines[:,i,2].reshape((2,2))
        ax.plot_surface(xx, yy, zz, alpha=0.3)
        
def plt_data(data, c):
    plt.scatter(data[:,0], data[:,1],c=cmap(c))

def plt_data_3d(data, ax, **kwargs):
    ax.scatter(data[:,0], data[:,1], data[:,2])
    
def plt_prettify(title, graph_ranges):
    plt.title(title)
    plt.xlim(graph_ranges[:,0])
    plt.ylim(graph_ranges[:,1])

def plt_bounds_3d(graph_ranges, ax):
    ax.set_xlim(graph_ranges[:,0])
    ax.set_ylim(graph_ranges[:,1])
    ax.set_zlim(graph_ranges[:,2])

def sane_graph_bounds(mins, maxes, ranges, range_pct=0.1):
    how_much_to_include_outside_range = range_pct*ranges
    return np.stack((mins - how_much_to_include_outside_range, maxes + how_much_to_include_outside_range))
    
def make_local_odr_lines_animation(linear_models, data, c, pth, bandwidths, range_pct = 0.1, kernel_cls=local_models.TriCubeKernel):
    data_mins, data_maxes, data_ranges = min_max_range(data)
    graph_bounds = sane_graph_bounds(data_mins, data_maxes, data_ranges, range_pct)
    spans = np.stack([(-0.2, 0.2, 2j) for i in range(linear_models.model.m)])
    
    os.makedirs(pth, exist_ok=1)
    timelog = local_models.loggin.TimeLogger(logger=logger, how_often=1, total=len(bandwidths), tag="odr_lines")
    for bandwidth in bandwidths:
        with timelog:
            figure = plt.figure()
            kernel_cb = kernel_cls(bandwidth=bandwidth)
            linear_params_vecs, linear_params_mean = transformate_data(data, kernel_cb, linear_models, r=kernel_cb.support_radius())
            lines = make_local_odr_lines(linear_params_vecs, linear_params_mean, spans)
            plt_local_odr_lines(lines)
            plt_data(data, c)
            title = "bandwidth_{:010.5f}".format(bandwidth)
            plt_prettify(title, graph_bounds)
            plt.savefig(os.path.join(pth, "{}.png".format(title)))
            
def make_local_odr_lines_animation_k(linear_models, data, c, pth, ks, range_pct = 0.1, kernel_cls=local_models.TriCubeKernel):
    data_mins, data_maxes, data_ranges = min_max_range(data)
    graph_bounds = sane_graph_bounds(data_mins, data_maxes, data_ranges, range_pct)
    spans = np.stack([(-0.2, 0.2, 2j) for i in range(linear_models.model.m)])
    kernel_vb = kernel_cls(bandwidth="knn")
    
    os.makedirs(pth, exist_ok=1)
    timelog = local_models.loggin.TimeLogger(logger=logger, how_often=1, total=len(bandwidths), tag="odr_lines")
    for k in ks:
        with timelog:
            figure = plt.figure()
            linear_params_vecs, linear_params_mean = transformate_data(data, kernel_vb, linear_models, k=k)
            lines = make_local_odr_lines(linear_params_vecs, linear_params_mean, spans)
            plt_local_odr_lines(lines)
            plt_data(data, c)
            title = "k_{:05d}".format(k)
            plt_prettify(title, graph_bounds)
            plt.savefig(os.path.join(pth, "{}.png".format(title)))
            
def make_local_odr_projections_animation_k(linear_models, data, c, pth, ks, range_pct=0.1, grid_steps=100, kernel=local_models.TriCubeKernel):
    timelog = local_models.loggin.TimeLogger(logger=logger, how_often=1, total=len(ks), tag="odr_projections_k", running_avg_length=10)
    min_nm, max_nm = np.min(data, axis=0), np.max(data, axis=0)
    nm_range = max_nm - min_nm
    how_much_to_include_outside_range =  range_pct*nm_range
    nm_ranges = np.stack((min_nm - how_much_to_include_outside_range, max_nm + how_much_to_include_outside_range))
    
    grid = get_global_grid(data, outside_range_pct=range_pct, n_steps=grid_steps)
    gridder = Grid2Vec().fit(grid)
    grid = gridder.transform(grid)
    projections = np.zeros(grid.shape)
    
    os.makedirs(pth, exist_ok=1)
    for k in ks:
        with timelog:
            kernel_kb = kernel(bandwidth="knn")        
            figure = plt.figure()
            linear_params_grid = linear_models.transform(grid, k=k, weighted=True, kernel=kernel_kb)
            linear_params_mean = linear_params_grid[:,2:]
            linear_params_vecs = linear_params_grid[:,:2]
            for i in range(grid.shape[0]):
                projections[i] = sublinear_project_vectorized(grid[i:i+1] - linear_params_mean[i:i+1], linear_params_vecs[i:i+1]) + linear_params_mean[i:i+1]- grid[i:i+1]
            plt.quiver(grid[:,0], grid[:,1], projections[:,0], projections[:,1], scale=10)
            plt.scatter(data[:,0], data[:,1],c=cmap(c))
            plt.title("k {:05d}".format(k))
            plt.xlim(nm_ranges[:,0])
            plt.ylim(nm_ranges[:,1])
            plt.savefig(os.path.join(pth, "k_{:05d}.png".format(k)))
                                                                       
def make_local_odr_projections_animation(linear_models, data, c, pth, bandwidths, range_pct=0.1, grid_steps=100, kernel=local_models.TriCubeKernel):
    timelog = local_models.loggin.TimeLogger(logger=logger, how_often=1, total=len(bandwidths), tag="odr_projections", running_avg_length=10)
    min_nm, max_nm = np.min(data, axis=0), np.max(data, axis=0)
    nm_range = max_nm - min_nm
    how_much_to_include_outside_range =  range_pct*nm_range
    nm_ranges = np.stack((min_nm - how_much_to_include_outside_range, max_nm + how_much_to_include_outside_range))
    
    grid = get_global_grid(data, outside_range_pct=range_pct, n_steps=grid_steps)
    gridder = Grid2Vec().fit(grid)
    grid = gridder.transform(grid)
    projections = np.zeros(grid.shape)
    
    os.makedirs(pth, exist_ok=1)
    for bandwidth in bandwidths:
        with timelog:
            kernel_cb = kernel(bandwidth=bandwidth)        
            figure = plt.figure()
            linear_params_grid = linear_models.transform(grid, r=kernel_cb.support_radius(), weighted=True, kernel=kernel_cb)
            linear_params_mean = linear_params_grid[:,2:]
            linear_params_vecs = linear_params_grid[:,:2]
            for i in range(grid.shape[0]):
                projections[i] = sublinear_project_vectorized(grid[i:i+1] - linear_params_mean[i:i+1], linear_params_vecs[i:i+1]) + linear_params_mean[i:i+1]- grid[i:i+1]
            plt.quiver(grid[:,0], grid[:,1], projections[:,0], projections[:,1], scale=10)
            plt.scatter(data[:,0], data[:,1],c=cmap(c))
            title = "bandwidth_{:010.5f}".format(bandwidth)
            plt.title(title)
            plt.xlim(nm_ranges[:,0])
            plt.ylim(nm_ranges[:,1])
            plt.savefig(os.path.join(pth, "{}.png".format(title)))
                                                                       
def make_odr_iterprojections_animation(linear_models, data, c, pth, bandwidth, range_pct=0.1, grid_steps=100, iterations=100, kernel=local_models.TriCubeKernel):
    timelog = local_models.loggin.TimeLogger(logger=logger, how_often=1, total=iterations, tag="odr_iterprojections", running_avg_length=10)
    min_nm, max_nm = np.min(data, axis=0), np.max(data, axis=0)
    nm_range = max_nm - min_nm
    how_much_to_include_outside_range =  range_pct*nm_range
    nm_ranges = np.stack((min_nm - how_much_to_include_outside_range, max_nm + how_much_to_include_outside_range))
    
    grid = get_global_grid(data, outside_range_pct=range_pct, n_steps=grid_steps)
    gridder = Grid2Vec()
    grid = gridder.fit_transform(grid)
    projections = grid.copy()
    kernel_cb = kernel(bandwidth=bandwidth)

    os.makedirs(pth, exist_ok=1)
    for i in range(iterations):
        with timelog:
            figure = plt.figure()
            plt.scatter(data[:,0], data[:,1],c=cmap(c))
            plt.scatter(projections[:,0], projections[:,1],c='r',s=0.1)
            plt.title("iteration_{:05d}".format(i))
            plt.xlim(nm_ranges[:,0])
            plt.ylim(nm_ranges[:,1])
            plt.savefig(os.path.join(pth, "iteration_{:05d}.png".format(i)))
            linear_params_grid = linear_models.transform(projections, r=kernel_cb.support_radius(), weighted=True, kernel=kernel_cb)
            err_pts = np.any(np.isnan(linear_params_grid), axis=1)
            logger.info("linear odr undefined at {} pts".format(err_pts.sum()))
            good_pts = np.logical_not(err_pts)
            projections, linear_params_grid = projections[good_pts], linear_params_grid[good_pts]
            linear_params_mean = linear_params_grid[:,2:]
            linear_params_vecs = linear_params_grid[:,:2]
            for i in range(projections.shape[0]):
                projections[i:i+1] += sublinear_project_vectorized(projections[i:i+1] - linear_params_mean[i:i+1], linear_params_vecs[i:i+1]) + linear_params_mean[i:i+1]- projections[i:i+1]


DEFAULT_VIEWS = [[30,45],[30,-135],[-30,-45],[-30,135]]
def make_local_odr_lines_animation_3d(linear_models, data, c, pth, bandwidths, range_pct = 0.1, kernel_cls=local_models.TriCubeKernel, views=DEFAULT_VIEWS):
    from mpl_toolkits.mplot3d import axes3d
    data_mins, data_maxes, data_ranges = min_max_range(data)
    graph_bounds = sane_graph_bounds(data_mins, data_maxes, data_ranges, range_pct)
    spans = np.stack([(-0.2, 0.2, 2j) for i in range(linear_models.model.m)])
    
    os.makedirs(pth, exist_ok=1)
    timelog = local_models.loggin.TimeLogger(logger=logger, how_often=1, total=len(bandwidths), tag="odr_lines_3d")
    for bandwidth in bandwidths:
        with timelog:
            kernel_cb = kernel_cls(bandwidth=bandwidth)
            linear_params_vecs, linear_params_mean = transformate_data(data, kernel_cb, linear_models, r=kernel_cb.support_radius())
            lines = make_local_odr_lines(linear_params_vecs, linear_params_mean, spans)
            fig, axes = plt.subplots(int(len(views)**0.5) + 1, int(len(views)**0.5), sharex='all', sharey='all')
            for i in range(len(views)):
                plt_local_odr_lines_3d(lines, axes[i])
                plt_data_3d(data, axes[i], c=cmap(c))
                plt_bounds_3d(graph_bounds, axes[i])
                axes[i].view_init(*views[i])
                
            title = "bandwidth_{:010.5f}"
            fig.suptitle(title)
            fig.savefig(os.path.join(pth, "{}.png".format(title)))


DEFAULT_VIEWS = [[30,45],[30,-135],[-30,-45],[-30,135]]
def make_local_odr_surfaces_animation_3d(linear_models, data, c, pth, bandwidths, range_pct = 0.1, kernel_cls=local_models.TriCubeKernel, views=DEFAULT_VIEWS):
    from mpl_toolkits.mplot3d import axes3d
    data_mins, data_maxes, data_ranges = min_max_range(data)
    graph_bounds = sane_graph_bounds(data_mins, data_maxes, data_ranges, range_pct)
    spans = np.stack([(-0.2, 0.2, 2j) for i in range(linear_models.model.m)])
    
    os.makedirs(pth, exist_ok=1)
    timelog = local_models.loggin.TimeLogger(logger=logger, how_often=1, total=len(bandwidths), tag="odr_lines_3d")
    for bandwidth in bandwidths:
        with timelog:
            kernel_cb = kernel_cls(bandwidth=bandwidth)
            linear_params_vecs, linear_params_mean = transformate_data(data, kernel_cb, linear_models, r=kernel_cb.support_radius())
            lines = make_local_odr_lines(linear_params_vecs, linear_params_mean, spans)
            fig, axes = plt.subplots(int(len(views)**0.5), int(len(views)**0.5-0.00001) + 1, subplot_kw=dict(projection='3d'))
            for i in range(len(views)):
                ax = axes[int(i/axes.shape[0]), i%axes.shape[0]]
                plt_local_odr_surfaces_3d(lines, ax)
                plt_data_3d(data, ax, c=cmap(c))
                plt_bounds_3d(graph_bounds, ax)
                ax.view_init(*views[i])
                
            title = "bandwidth_{:010.5f}".format(bandwidth)
            fig.suptitle(title)
            fig.savefig(os.path.join(pth, "{}.png".format(title)))


def make_odr_iterprojections_animation_3d(linear_models, data, c, pth, bandwidth, range_pct=0.1, grid_steps=100, iterations=100, kernel=local_models.TriCubeKernel, views=DEFAULT_VIEWS):
    timelog = local_models.loggin.TimeLogger(logger=logger, how_often=1, total=iterations, tag="odr_iterprojections", running_avg_length=10)
    data_mins, data_maxes, data_ranges = min_max_range(data)
    graph_bounds = sane_graph_bounds(data_mins, data_maxes, data_ranges, range_pct)
    
    grid = get_global_grid(data, outside_range_pct=range_pct, n_steps=grid_steps)
    gridder = Grid2Vec()
    grid = gridder.fit_transform(grid)
    projections = grid.copy()
    kernel_cb = kernel(bandwidth=bandwidth)

    os.makedirs(pth, exist_ok=1)
    for i in range(iterations):
        with timelog:
            fig, axes = plt.subplots(int(len(views)**0.5), int(len(views)**0.5-0.00001) + 1, subplot_kw=dict(projection='3d'))
            for i in range(len(views)):
                ax = axes[int(i/axes.shape[0]), i%axes.shape[0]]
                plt_data_3d(data, ax, c=cmap(c))
                plt_data_3d(projections, ax, c='r', s=0.1)
                plt_bounds_3d(graph_bounds, ax)
                ax.view_init(*views[i])
                
            title = "iteration_{:05d}".format(i)
            fig.suptitle(title)
            fig.savefig(os.path.join(pth, "{}.png".format(title)))

            linear_params_vecs, linear_params_mean = transformate_data(projections, kernel_cb, linear_models, r=kernel_cb.support_radius())
            err_pts = np.any(np.isnan(linear_params_vecs), axis=(1,2))
            logger.info("linear odr undefined at {} pts".format(err_pts.sum()))
            good_pts = np.logical_not(err_pts)
            projections, linear_params_vecs, linear_params_mean = projections[good_pts], linear_params_grid[good_pts], linear_params_mean[good_pts]
            for i in range(projections.shape[0]):
                projections[i:i+1] += sublinear_project_vectorized(projections[i:i+1] - linear_params_mean[i:i+1], linear_params_vecs[i:i+1]) + linear_params_mean[i:i+1]- projections[i:i+1]
