import numpy as np
import sklearn.base
import subprocess
import scipy.spatial
from . import local_models

def kernel_density(x, data, kernel, metric='euclidean'):
    cdist_matrix = scipy.spatial.distance.cdist(x, data, metric=metric)
    similarity_matrix = kernel(cdist_matrix)
    return np.sum(similarity_matrix, axis=1)

class ConstantAngleProjector(object):
    def __init__(self, theta_generator):
        self.theta_generator = theta_generator
    def __call__(self, p, mu):
        theta = next(self.theta_generator)
        pmu_vec = p - mu
        normal = np.stack((-pmu_vec[:,1], pmu_vec[:,0]), axis=1)
        unit_normal = normal/np.linalg.norm(normal, axis=1).reshape((-1,1))
        constant_angle_normal = unit_normal * (np.linalg.norm(pmu_vec, axis=1)/2 * np.arctan(theta)).reshape((-1,1))
        the_vector_from_mu_along_this_angle_to_p = (mu + pmu_vec/2) + constant_angle_normal
        unit_the_vector_from_mu_along_this_angle_to_p = the_vector_from_mu_along_this_angle_to_p/np.linalg.norm(the_vector_from_mu_along_this_angle_to_p, axis=1).reshape((-1,1))
        unit_the_vector_from_mu_along_this_angle_to_p = unit_the_vector_from_mu_along_this_angle_to_p.reshape((unit_the_vector_from_mu_along_this_angle_to_p.shape[0], 1, -1))
        return linear_project_pointwise_bases(p, unit_the_vector_from_mu_along_this_angle_to_p, mu)

def get_global_grid(x, outside_range_pct=0.1, n_steps=100):
    '''outside_range_pct and n_steps can be a float or a 1d array same len as x.shape[1]'''
    min_x, max_x = np.min(x, axis=0), np.max(x, axis=0)
    x_range = max_x - min_x
    how_much_to_include_outside_range =  outside_range_pct*x_range
    x_ranges = np.stack((min_x - how_much_to_include_outside_range, max_x + how_much_to_include_outside_range))
    x_step = ((x_ranges[1] - x_ranges[0])/n_steps)
    grid_limits = tuple(map(lambda i: slice(x_ranges[0,i], x_ranges[1,i], x_step[i]), range(x_ranges.shape[1])))
    g = np.mgrid[grid_limits]
    return g

class Grid2Vec(object):
    def fit(self, x):
        self.original_data_like_shape = [-1] + list(x.shape[1:])
        return self
    def transform(self, x):
        return np.vstack(map(np.ravel, x)).T
    def fit_transform(self, x):
        return self.fit(x).transform(x)
    def inverse_transform(self, x):
        return x.T.reshape(self.original_data_like_shape)

def local_model_projection_transformation(local_models_instance, x, local_model_params=None, **local_model_params_kwargs):
    '''pass in local_model_params if you've already computed them'''
    model_clone = sklearn.base.clone(local_models_instance.model)
    projected_x = np.zeros(x.shape)
    if local_model_params is None:
        local_model_params = local_models.transform(x, **local_model_params_kwargs)
    for i in range(x.shape[0]):
        local_models.set_learned_param_vector(model_clone, local_model_params[i])
        projected_x[i] = model_clone.project(x[i])
    return projected_x

def perp2d(v):
    a,b = v
    return np.array((-b,a))/np.linalg.norm([a,b])

def linear_space(center, normal, span):
    return np.einsum("i,j->ij", span, perp2d(normal)) + center

def sublinear_space(orthonormal_basis, tuple_of_slice_tuples):
    grid_limits = tuple(map(lambda x: slice(*x),tuple_of_slice_tuples))
    g = np.mgrid[grid_limits]
    g = Grid2Vec().fit_transform(g)
    return np.einsum('ij,ki->kj', orthonormal_basis, g)

def sublinear_spaces(orthonormal_bases, slice_dims_array):
    grid_limits = tuple(map(lambda x: slice(*x), slice_dims_array))
    g = np.mgrid[grid_limits]
    g = Grid2Vec().fit_transform(g)
    return np.einsum('ijk,lj->lik', orthonormal_bases, g)

def linear_project_vectorized(pts, normals):
    n_dot_n = np.einsum('ij,ij->i',normals,normals)
    x_dot_n = np.einsum('ij,ij->i',pts,normals)
    return ((1-x_dot_n)/n_dot_n).reshape((-1,1))*normals

def sublinear_project_vectorized(pts, orthonormal_basis, mean=0):
    #first, shift everything by the mean
    x = pts - mean
    #now the plane passes through the origin, and we can project onto the eigenspace directly
    projection = np.einsum('ij,kj->ki',orthonormal_basis,x).dot(orthonormal_basis)
    #now shift back
    return projection + mean

def linear_project_pointwise_bases(x, orthonormal_bases, mean=0):
    ''' projects a bunch of points x onto an equally long list of basis vectors (one set for each pt) given in orthonormal bases'''
    x = x-mean
    projections = np.einsum('ijk,ik,ijn->in', orthonormal_bases, x, orthonormal_bases)
    return projections + mean

def imgs2video(glob, outfile, framerate=50):
    import subprocess
    args = "ffmpeg -r {} -pattern_type glob -i \"{}\" -c:v libx264 -vf \"fps=25,format=yuv420p\" {} -y".format(
        framerate, glob, outfile)
    print(args)
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    print(subprocess.list2cmdline(args), output, err)

def video_html(src):
    return '''<video controls src="{}" type="video/mp4"/>'''.format(src)
