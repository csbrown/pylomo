import numpy as np
import matplotlib.pyplot as plt
import os
from ml_battery.utils import cmap
from . import utils
from . import local_models
from . import algorithms
from . import linear_projections


def make_mean_iterprojections_animation(mean_models, data, c, pth, bandwidth, range_pct=0.1, grid_steps=100, iterations=100, kernel=local_models.TriCubeKernel):
    shifter_maker = lambda projections: algorithms.mean_shift(mean_models, projections, bandwidth, iterations=iterations, kernel=kernel)
    iterprojections_animation(shifter_maker, data, c, pth, bandwidth, range_pct=range_pct, grid_steps=grid_steps)
    
def make_meanline_iterprojections_animation(mean_models, theta_generator, data, c, pth, bandwidth, range_pct=0.1, grid_steps=100, iterations=100, kernel=local_models.TriCubeKernel):
    line_projector = utils.ConstantAngleProjector(theta_generator)
    shifter_maker = lambda projections: algorithms.mean_line_projection_shift(mean_models, line_projector, projections, bandwidth, iterations=iterations, kernel=kernel)
    iterprojections_animation(shifter_maker, data, c, pth, bandwidth, range_pct=range_pct, grid_steps=grid_steps)
    
def make_tlsline_iterprojections_animation(linear_models, data, c, pth, range_pct=0.1, grid_steps=100, iterations=100, kernel=None):
    shifter_maker = lambda projections: algorithms.local_tls_shift(linear_models, projections, iterations=iterations, kernel=kernel)
    iterprojections_animation(shifter_maker, data, c, pth, range_pct=range_pct, grid_steps=grid_steps)
        
def iterprojections_animation(shifter_maker, data, c, pth, range_pct=0.1, grid_steps=100):
    data_mins, data_maxes, data_ranges = linear_projections.min_max_range(data)
    graph_bounds = linear_projections.sane_graph_bounds(data_mins, data_maxes, data_ranges, range_pct)

    grid = utils.get_global_grid(data, outside_range_pct=range_pct, n_steps=grid_steps)
    gridder = utils.Grid2Vec()
    grid = gridder.fit_transform(grid)
    projections = grid.copy()

    shifter = shifter_maker(projections)
    
    os.makedirs(pth, exist_ok=1)
    for i, projections in enumerate(shifter):
        figure = plt.figure()
        plt.scatter(data[:,0], data[:,1],c=cmap(c))
        plt.scatter(projections[:,0], projections[:,1],c='r',s=0.1)
        plt.title("iteration_{:05d}".format(i))
        plt.xlim(graph_bounds[:,0])
        plt.ylim(graph_bounds[:,1])
        plt.savefig(os.path.join(pth, "iteration_{:05d}.png".format(i)))
