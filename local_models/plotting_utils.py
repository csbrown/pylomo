import numpy as np
import os
import cv2
import mayavi
import mayavi.mlab
import matplotlib.pyplot as plt
import random

def imshow(pth, cv2color=cv2.IMREAD_COLOR, trim_border=True, **kwargs):
    img = cv2.imread(pth, cv2color)
    if trim_border:
        img = trim_whiteborder(img)
    plt.imshow(img, **kwargs)
    plt.axis("off")

def trim_whiteborder(img, ratio=1.1):
    img = cv2.bitwise_not(img)
    im_sq = img**2
    sum_sq_cols = np.sum(im_sq, axis=0)
    sum_sq_rows = np.sum(im_sq, axis=1)
    if len(img.shape) > 2:
        sum_sq_cols = np.sum(sum_sq_cols, axis=1)
        sum_sq_rows = np.sum(sum_sq_rows, axis=1)
    first_nonzero_col = (sum_sq_cols!=0).argmax()
    last_nonzero_col = sum_sq_cols.shape[0] - (sum_sq_cols[::-1]!=0).argmax()
    first_nonzero_row = (sum_sq_rows!=0).argmax()
    last_nonzero_row = sum_sq_rows.shape[0] - (sum_sq_rows[::-1]!=0).argmax()
    totally_trimmed = img[first_nonzero_row:last_nonzero_row, first_nonzero_col:last_nonzero_col]
    if len(img.shape) == 2:
        just_a_little_whitespace = np.zeros((
            int(totally_trimmed.shape[0]*ratio), 
            int(totally_trimmed.shape[1]*ratio)
        ), dtype=np.uint8)
    else:
        just_a_little_whitespace = np.zeros((
            int(totally_trimmed.shape[0]*ratio), 
            int(totally_trimmed.shape[1]*ratio),
            img.shape[2]
        ), dtype=np.uint8)
    jalw_middle_section_start = ((np.array(just_a_little_whitespace.shape) - np.array(totally_trimmed.shape))/2).astype(int)
    just_a_little_whitespace[
        jalw_middle_section_start[0]:jalw_middle_section_start[0] + totally_trimmed.shape[0],
        jalw_middle_section_start[1]:jalw_middle_section_start[1] + totally_trimmed.shape[1]
    ] = totally_trimmed
    #return just_a_little_whitespace
    return cv2.bitwise_not(just_a_little_whitespace)
       
def import_shit():
    import local_models.local_models
    import local_models.algorithms
    import local_models.utils
    import local_models.linear_projections
    import local_models.loggin
    import local_models.TLS_models
    import numpy as np
    import logging
    import string
    import ml_battery.log


    logger = logging.getLogger(__name__)

    #reload(local_models.local_models)
    #reload(lm)
    #reload(local_models.loggin)
    #reload(local_models.TLS_models)
    np.warnings.filterwarnings('ignore')
    return logger
    
def mean_center(data, weights=None):
    return data - np.average(data, axis=0,weights=weights)

def load_converged_data(pth):
    convergededs = []
    for dat in sorted(os.listdir(pth)):
        convergededs.append(np.loadtxt(os.path.join(pth, dat)))
    return np.concatenate(convergededs, axis=0)

def plt_grid(fig, grid, data_avg, data_std, colormap='gist_earth'):
    nodes = mayavi.mlab.points3d(grid[:,0], grid[:,1], grid[:,2], 
                                 scale_mode='scalar', scale_factor=1,
                                 colormap=colormap, figure=fig)
    nodes.glyph.scale_mode = 'scale_by_vector'
    nodes.mlab_source.dataset.point_data.vectors = np.ones((grid.shape[0],3))*(np.average(data_std)/60)
    nodes.mlab_source.dataset.point_data.scalars = (grid[:,1] - (data_avg[1]-3*data_std[1]))/(6*data_std[1])
    return nodes

def plt_data(fig, data, data_std):
    nodes = mayavi.mlab.points3d(data[:,0], data[:,1], data[:,2], 
                                 scale_mode='scalar', scale_factor=1,
                                 colormap='Greens', figure=fig)
    nodes.glyph.scale_mode = 'scale_by_vector'
    nodes.mlab_source.dataset.point_data.vectors = np.ones((data.shape[0],3))*(np.average(data_std)/60)
    nodes.mlab_source.dataset.point_data.scalars = np.ones((data.shape[0]))
    return nodes

def get_normals(kernel, linear_models, data):
    if hasattr(kernel.bandwidth, "__call__"):
        linear_params_vecs, linear_params_mean = local_models.linear_projections.transformate_data(data, kernel, linear_models, k=kernel.k)
    else:
        linear_params_vecs, linear_params_mean = local_models.linear_projections.transformate_data(data, kernel, linear_models, r=kernel.support_radius())
    return linear_params_vecs

def align_normals(data, normals, k=10, iterations=100):
    balltree = sklearn.neighbors.BallTree(data)
    pairwise_nearest_indices = balltree.query(data,k=k,sort_results=True,return_distance=False)
    for iteration in range(iterations):
        alignments = []
        for index in range(1,pairwise_nearest_indices.shape[1]):
            alignment = np.einsum("ij,ij->i",normals,normals[pairwise_nearest_indices[:,index]])
            alignments.append(alignment)
        alignment = np.average(alignments, axis=0)
        wrong_alignment = np.sign(alignment)
        normals = normals*wrong_alignment.reshape(-1,1)
    return normals

def align_edge_normals(data, normals, edge_range=0.1):
    data_mins, data_maxes, data_ranges = local_models.linear_projections.min_max_range(data)
    graph_bounds = local_models.linear_projections.sane_graph_bounds(data_mins, data_maxes, data_ranges, -edge_range)
    mins = data < graph_bounds[:1]
    maxes = data > graph_bounds[1:]
    mins_alignment = np.sign(np.einsum("ij,ij->i",mins,-1*normals))
    maxes_alignment = np.sign(np.einsum("ij,ij->i",maxes,normals))
    mins_alignment += np.logical_not(mins_alignment) # turn 0s into 1s (so they don't change)
    maxes_alignment += np.logical_not(maxes_alignment)    
    return normals*mins_alignment.reshape(-1,1)*maxes_alignment.reshape(-1,1)

def plt_normals(fig, normals, data, data_std):
    nodes = mayavi.mlab.quiver3d(data[:,0], data[:,1], data[:,2],
                                 normals[:,0], normals[:,1], normals[:,2],
                                 scale_mode='scalar', scale_factor=np.average(data_std)/5,
                                 colormap='Purples', figure=fig, line_width=1.0)
    return nodes

def normalize_view(fig, data_avg, data_std, azimuth=0, elevation=0, roll=None):
    mayavi.mlab.view(
        azimuth=azimuth, elevation=elevation, roll=roll, distance=15*np.average(data_std), 
        focalpoint=(data_avg[0], data_avg[1], data_avg[2]),
        figure=fig)
    
def plt_and_save(data, grid, normals, pth):
    data_avg = np.average(data, axis=0)
    data_std = np.std(data, axis=0)
    figure = mayavi.mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=(0,0,0), engine=None, size=(1000, 500))
    data_nodes = plt_data(figure, data, data_std)
    converged_nodes = plt_grid(figure, grid, data_avg, data_std)
    normal_vecs = plt_normals(figure, normals, grid, data_std)
    neg_normal_vecs = plt_normals(figure, -normals, grid, data_std)
    normalize_view(figure, data_avg, data_std)
    mayavi.mlab.savefig(pth, magnification=2)
    mayavi.mlab.close(figure)

def serialize_plt(pth):
    import zlib
    with open(pth, 'rb') as f:
        dat = f.read()
    return zlib.compress(dat)
   
def deserialize_plt(dat, pth):
    import zlib
    with open(pth, 'wb') as f:
        f.write(zlib.decompress(dat))
    return pth

def distributed_plt_and_save(data, grid, bandwidth):
    import numpy as np
    import mayavi
    import mayavi.mlab
    import string
    import os
    #on headless systems, tmux: "Xvfb :1 -screen 0 1280x1024x24 -auth localhost", then "export DISPLAY=:1" in the jupyter tmux
    mayavi.mlab.options.offscreen = True
    
    unique_id = "".join(np.random.choice(list(string.ascii_lowercase), replace=True, size=20))
    pth = "/ramfs/{}.png".format(unique_id)
    try:
        plt_and_save(data, grid, bandwidth, pth)
        result = serialize_plt(pth)
    except:
        os.remove(pth)


def mayavi_plt_pts(pts, pth= str(random.random()) + ".png", 
    display=True, focus=None, dist=None, trim=True, colormaps=None, offscreen=True):

    mayavi.mlab.options.offscreen = offscreen

    figure = mayavi.mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=(0,0,0), engine=None, size=(1000, 1000))

    if focus is None:
        focus = np.average(pts, axis=0)
    if dist is None:
        dist = np.std(pts, axis=0)
        
    if isinstance(pts, list):
        for i, pt in enumerate(pts):
            if colormaps is not None:
                nodes = plt_grid(figure, pt, focus, dist, colormaps[i])
            else:
                nodes = plt_grid(figure, pt, focus, dist)
    else:
        nodes = plt_grid(figure, pts, focus, dist)

    normalize_view(figure, focus, dist)
    mayavi.mlab.savefig(pth, figure=figure, magnification=2)
    mayavi.mlab.clf(figure)
    mayavi.mlab.close(figure)
    if display:
        return imshow(pth, trim_border=trim) 
