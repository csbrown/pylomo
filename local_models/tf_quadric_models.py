import tensorflow as tf
import tf_roots
import tf_roots.src.companion_eigv
from . import quadrics_utils
from .quadric_models import *
import numpy as np
from . import tensorflow_models
import functools
from . import TLS_models
from . import loggin
import logging

logger = logging.getLogger(__name__)

projection_funcs = {
    "k_mat": quadrics_utils.k_mat,
    "other_dims_as_x": lambda a,b,c,d,e,f,g,x: [quadrics_utils.y_as_x(a,b,c,d,e,f,g,x), quadrics_utils.z_as_x(a,b,c,d,e,f,g,x)],
}

def broadcastable_where(condition, x=None, y=None, *args, **kwargs):
    if x is None and y is None:
        return tf.where(condition, x, y, *args, **kwargs)
    else:
        _shape = tf.broadcast_dynamic_shape(tf.shape(condition), tf.shape(x))
        _broadcaster = tf.ones(_shape)
        x_broadcaster = tf.cast(_broadcaster, x.dtype)
        y_broadcaster = tf.cast(_broadcaster, y.dtype)
        return tf.where(
            condition & (_broadcaster > 0.0), 
            x * x_broadcaster,
            y * y_broadcaster,
            *args, **kwargs
        )


def SortEigenDecomposition(e, v):
    perm = np.argsort(e, -1)
    return np.take(e, perm, -1), np.take(v, perm, -1)

def tf_unrotate_quadric(U,L):
    return U@tf.diag(L)@tf.transpose(U)

def tf_rotate_and_translate_quadric(Q,X):
    # this translates and rotates Q to have a diagonal upper left block, and to "center" on the point in X batch-wise
    # since the upper left block doesn't depend on X, we keep the blocks separate for efficiency
    print("Q", Q)
    Qs = Q[:-1,:-1]
    L, U = tf.linalg.eigh(Qs)
    UL = L
    UTQs = tf.matmul(U,Qs,transpose_a=True)
    UTQsT = tf.transpose(UTQs,[1,0])
    print(UTQsT)
    XUTQsT = X@UTQsT
    print(XUTQsT)
    Q4U = Q[-1:,:-1]@U
    print(Q4U)
    UR = XUTQsT + Q4U
    BR = batch_dot(X@Qs,X) + 2*X@Q[:-1,-1:] + Q[-1,-1]
    print("L,U,UL,UR,BR",L,U,UL,UR,BR)
    return L,U,UL,UR,BR

def tf_circular_permutation(to_rotate, rotate_by):
    rowwise_indices = tf.reshape(tf.range(to_rotate.shape[-1], dtype=rotate_by.dtype), 
                              [1, to_rotate.shape[-1]])
    tiled_rowwise_indices = tf.tile(rowwise_indices, [tf.shape(to_rotate)[0],1])
    rotated_indices = tf.mod(tiled_rowwise_indices + tf.expand_dims(rotate_by, -1), to_rotate.shape[-1])
    return tf.gather(to_rotate, rotated_indices, batch_dims=1)    
    
def tf_batched_frontal_fill(to_zero, end, fill=0):
    rowwise_indices = tf.reshape(
        tf.range(tf.cast(tf.shape(to_zero)[-1], dtype=end.dtype), dtype=end.dtype), 
        [1, tf.shape(to_zero)[-1]])
    tiled_rowwise_indices = tf.tile(rowwise_indices, [tf.shape(to_zero)[0],1]) 
    mask = tf.less(tiled_rowwise_indices, tf.expand_dims(end, -1))
    fill_value = tf.cast(fill, to_zero.dtype)
    print("mask", mask)
    print("fill_value", fill_value)
    print("justbeforezeroing", to_zero)
    res = tf_where(mask, fill_value, to_zero)
    print("select_result", res)
    return res
    #return broadcastable_where(mask, tf.cast(fill, to_zero.dtype), to_zero)

def tf_where(condition, iftrue, iffalse):
    iftrue *= tf.cast(condition, iftrue.dtype)
    iffalse *= tf.cast(tf.logical_not(condition), iffalse.dtype)
    print("t",iftrue)
    print('f',iffalse)
    #return iffalse
    return iftrue + iffalse


BIG_NUMBER=9999999. #TODO: this is kinda janky
def tf_roots_possible_zeros(coeff_vector, zero_coeff_tol=1e-10):
    ''' 
        the coeff_vector should be the coefficients of x^n, x^{n-1} ... x^1, 1
        this method deals with possible zero coefficients, and pads stuff so that we always
            get back a vector of length coeff_vector.shape[0] - 1
        NB: If the leading coeffs are 0 (x^n, etc), 
            then this method returns that many nans.  
    '''
    print("coeff_vectors", coeff_vector)
    max_nonzero_coeff = tf.argmax(tf.cast(tf.abs(coeff_vector)>zero_coeff_tol,tf.int32), axis=-1)
    print("zero_indices", max_nonzero_coeff)
    farill_zeroed_coeffs = tf_batched_frontal_fill(coeff_vector, max_nonzero_coeff)
    print("zeroed", farill_zeroed_coeffs)
    print("cshape", farill_zeroed_coeffs.shape)
    rotated_coeffs = tf_circular_permutation(farill_zeroed_coeffs, max_nonzero_coeff)
    #kmat_rotated = squeezed_kmat
    print("rotated_coeffs", rotated_coeffs)
    # coeff_vectors is [x^n,x^n-1,....1]
    # tf_roots expects coeffs of [1, x^1,....x^n-1], and coeff of x^n is 1, so we have to rearrange
    backward_coeffs_to_nless1 = rotated_coeffs[...,:0:-1]
    print("backward_coeffs_to_nless1", backward_coeffs_to_nless1)
    unit_lead_coeffs_to_nless1 = backward_coeffs_to_nless1/rotated_coeffs[...,:1]
    print("unit_lead_coeffs_to_nless1", unit_lead_coeffs_to_nless1)
    roots = tf_roots.tf_roots.tf_roots(unit_lead_coeffs_to_nless1)
    print("roots", roots)
    root_sorter = tf.argsort(tf.abs(roots),axis=-1,direction='ASCENDING')
    sorted_roots = tf.gather(roots, root_sorter, batch_dims=len(coeff_vector.get_shape().as_list())-1)
    print("sorted_roots", sorted_roots)
    #TODO: THIS IS NOT GOOD FIX THIS
    #nanned_zero_roots = tf.where_v2(tf.abs(roots) < zero_coeff_tol, tf.constant(np.inf+0.j, dtype=roots.dtype), roots)
    nanned_zero_roots = tf_batched_frontal_fill(sorted_roots, max_nonzero_coeff, fill=BIG_NUMBER)
    return nanned_zero_roots
    return roots


def tf_ortho_project_x(UL,UR,BR):
    batch_shape = tf.shape(BR)[:-1]
    dims = UL.get_shape().as_list()
    ul_tiled = tf.tile(
        tf.reshape(UL, tf.concat([[1]*tf.rank(batch_shape), dims], axis=0)), 
        tf.concat([batch_shape, [1]*len(dims)], axis=0)
    )
    args = (
        [ul_tiled[...,i:i+1] for i in range(UL.shape[-1])] + 
        [BR[...,0:1]] + 
        [UR[...,i:i+1] for i in range(UR.shape[-1])]
    )

    # kmat is the coeffs of [x^n, x^n-1, ... x^1, 1]
    for i, arg in enumerate(args):
        print("args{}".format(i),arg)
        print(arg.shape)
    coeffses = quadrics_utils.k_mat(*args)
    for i, coef in enumerate(coeffses):
        print("coeffses{}".format(i),coef)
        print(coef.shape)
    kmat = tf.stack(coeffses, axis=-1)
    print("kmat", kmat)
    squeezed_kmat = kmat[:,0,:]

    # if leading coefficients are 0 this is bad.... however, we can make them trailing coefficients and just get
    # a bunch of extra roots at 0.  We discard those, and call it a day
    roots = tf_roots_possible_zeros(squeezed_kmat, zero_coeff_tol=1e-7)
    return args, roots
    
def tf_get_other_dims(UL,UR,BR):
    args, roots = tf_ortho_project_x(UL,UR,BR)

    other_dims = projection_funcs["other_dims_as_x"](*(list(args) + [roots]))
    pts = tf.stack([roots] + other_dims, axis=-1)
    pts_nonan = tf.where(
        tf.logical_or(tf.is_nan(tf.real(pts)), tf.is_nan(tf.imag(pts))), 
        tf.zeros_like(pts), pts)
    return pts_nonan
    
def tf_ortho_project_prerotated(UL,UR,BR,X,imag_0tol = 1e-10, quadric_constraint_tol=1e-10):
    pts = tf_get_other_dims(UL,UR,BR)
    print("pts", pts)
    
    #ortho_pts, ortho_dists = tf_min_dist_search_complexandquadric_constraint(UL,UR,BR,pts,imag_0tol=imag_0tol,quadric_constraint_tol=quadric_constraint_tol)
    ortho_pts, ortho_dists = tf_min_dist_search_complex_constraint(pts,imag_0tol=imag_0tol)

    return ortho_pts, ortho_dists

def tf_quadric_constraint(UL,UR,BR,pts):
    return batch_dot(tf.transpose(tf.diag(UL)@tf.transpose(pts)), pts) + 2*batch_dot(UR, pts) + BR
def tf_quadric_constraint(UL,UR,BR,pts):
    XU = broadcasted_matmul(tf.diag(UL), tf.expand_dims(pts,-1))[...,0]
    XUX = batch_dot(pts, XU, keep_dims=False)
    XUR = batch_dot(tf.expand_dims(UR,1),pts, keep_dims=False)
    return XUX + 2*XUR + BR
    
def tf_ortho_project(Q,X,imag_0tol=1e-10,quadric_constraint_tol=1e-10):
    if not Q.dtype.is_complex:
        Qc = tf.cast(Q, tf.complex128 if Q.dtype == tf.float64 else tf.complex64)
    else:
        Qc = Q
    if not X.dtype.is_complex:
        Xc = tf.cast(X, tf.complex128 if X.dtype == tf.float64 else tf.complex64)
    else:
        Xc = X
    L,U,UL,UR,BR = tf_rotate_and_translate_quadric(Qc,Xc)
    ortho_pts, ortho_dists = tf_ortho_project_prerotated(UL,UR,BR,Xc,imag_0tol=imag_0tol,quadric_constraint_tol=quadric_constraint_tol)
    return tf.real(tf_unrotate_and_translate(U,Xc,ortho_pts)), tf.real(ortho_dists)#, blah
    
def tf_unrotate_and_translate(U, orig_X, new_X):
    return new_X@tf.transpose(U) + orig_X

def tf_masked_dist(pts, mask=None):
    # returns the masked distances of points to the origin, subject to the given mask.  
    # pts shape (..., n) where n is the dimensionality of X, 
    # Other dims are treated as batch dims.
    if mask is not None:
        pts = tf.where_v2(tf.expand_dims(mask,-1), tf.fill(tf.shape(pts), tf.constant(np.inf, pts.dtype)), pts)
    dists = tf.reduce_sum(tf.abs(pts)**2, axis=-1)

    return dists

def tf_min_search(searchable, axis=-1):
    # finds the minimum indices in searchable along a particular axis
    return tf.expand_dims(tf.argmin(searchable, axis=axis), axis)

def tf_quadric_points_complex_mask(pts, imag_0tol=1e-6):
    # this generates a mask of candidate points
    # we don't want any imaginary stuff
    imaginary_part_metric = tf.reduce_sum(tf.abs(tf.imag(pts)), axis=-1)
    return tf.greater(imaginary_part_metric, imag_0tol)

def tf_nan_mask(pts):
    return tf.logical_or(tf.is_nan(tf.real(pts)), tf.is_nan(tf.imag(pts)))

def tf_quadric_points_quadric_constraint_mask(UL, UR, BR, pts, tol=1e-10):
    # this generates a mask of candidate points
    # we want points satisfying the quadric equation
    return tf.greater(tf.abs(tf_quadric_constraint(UL, UR, BR, pts)), tol)

def tf_min_dist_search_complex_constraint(searchable, imag_0tol=1e-6):
    # searches `searchable` for the closest real-valued point to the origin.  
    # searchable shape (..., r, n) where n is the dimensionality of X, 
    # r is the number of points searched over.  Other dims are treated as batch dims.

    searchable_shape = searchable.get_shape().as_list()
    batch_shape = searchable_shape[:len(searchable_shape)-2] 
    n_dims = searchable_shape[-1]
    n_searchable_pts = searchable_shape[-2]

    complex_mask = tf_quadric_points_complex_mask(searchable, imag_0tol=imag_0tol)
    full_mask = complex_mask
    print("the mask", full_mask)

    dists = tf.real(tf_masked_dist(searchable, full_mask))
    print("dists", dists)
    idx = tf_min_search(dists)

    return tf.gather_nd(searchable, idx, batch_dims=len(batch_shape)), tf.gather_nd(dists, idx, batch_dims=len(batch_shape))

def tf_min_dist_search_complexandquadric_constraint(UL,UR,BR,pts,imag_0tol=1e-3,quadric_constraint_tol=1e-3):
    # searches `searchable` for the closest real-valued point to the origin subject to some constraints  
    # searchable shape (..., r, n) where n is the dimensionality of X, 
    # r is the number of points searched over.  Other dims are treated as batch dims.    
    searchable_shape = pts.get_shape().as_list()
    batch_shape = searchable_shape[:len(searchable_shape)-2] 
    n_dims = searchable_shape[-1]
    n_searchable_pts = searchable_shape[-2]
    
    complex_mask = tf_quadric_points_complex_mask(pts, imag_0tol=imag_0tol)
    quadric_constraint_mask = tf_quadric_points_quadric_constraint_mask(UL, UR, BR, pts, tol=quadric_constraint_tol)
    full_mask = tf.logical_or(complex_mask, quadric_constraint_mask)
    full_mask = complex_mask

    dists = tf.real(tf_masked_dist(pts, full_mask))
    
    idx = tf_min_search(dists)

    return tf.gather_nd(pts, idx, batch_dims=len(batch_shape)), tf.gather_nd(dists, idx, batch_dims=len(batch_shape))



def tf_swap_axes(x, axes=(0,1)):
    #swap two axes
    perm = tf.concat((tf.range(0,axes[0]), axes[1:], tf.range(axes[0]+1,axes[1]), axes[:1], tf.range(axes[1]+1,tf.rank(x))), 0)
    return tf.transpose(x, perm)

def broadcasted_matmul(A, B):
    """ Matmul with fast broadcasting

    Args:
      A - [n, m]
      B - [..., m, k]

    Returns:
      AB - [..., n, k]
    """
    # [n, batch_size, k]
    AB = tf.tensordot(A, B, axes=[[-1], [-2]])
    # [batch_size, n, k]
    rankAB = tf.rank(AB)
    return tf.transpose(AB, perm=tf.concat((
        tf_circular_permutation(tf.expand_dims(tf.range(rankAB-1),0),tf.constant([1],dtype=tf.int32))[0],[rankAB-1]),0))
    
def batch_dot(a,b,keep_dims=True):
    return tf.reduce_sum(tf.multiply(a, b), -1, keep_dims=keep_dims)

def tf_boolean_mask_inverse(boolean_mask, true_bar, false_bar):
    stacked_bar = tf.concat((true_bar, false_bar), axis=0)
    index_mapping = tf.where(boolean_mask)
    true_index_mapping = tf.where_v2(boolean_mask)[:,0]
    false_index_mapping = tf.where_v2(tf.logical_not(boolean_mask))[:,0]
    stacked_index_mapping = tf.concat((true_index_mapping, false_index_mapping), axis=0)
    basic_indices = tf.range(tf.shape(stacked_index_mapping)[0])
    inverse_index_mapping = tf.gather(basic_indices, stacked_index_mapping)
    return tf.gather(stacked_bar, inverse_index_mapping)

def tf_rollaxis(t, shift):
    #roll the axes of t by shift steps
    rank_indices = tf.range(tf.rank(t))
    roller = tf.roll(rank_indices, shift, 0)
    return tf.transpose(t, roller)

def tf_insert_row(t, r, i, axis=0):
    #insert tensor r at index i of tensor t along axis 0
    #requires: r.shape[axis] = 1, r.shape[k] = t.shape[k] ∀k != axis
    rank_indices = tf.range(tf.rank(t))
    roller = tf.roll(rank_indices, -axis, 0)
    rolled_t = tf.transpose(t, roller)
    rolled_r = tf.transpose(r, roller)
    stackt = tf.concat((t[:i], r, t[i:]), axis=0)
    return tf.transpose(stackt, tf.roll(rank_indices, axis, 0))

def tf_munge(t, i, r, j, axis=0):
    #insert tensor t at indices i and tensor r at indices j on axis `axis`.
    i = tf.expand_dims(i, -1)
    j = tf.expand_dims(j, -1)
    rank_indices = tf.range(tf.rank(t))
    roller = tf.roll(rank_indices, -axis, 0)
    rolled_t = tf.transpose(t, roller)
    rolled_r = tf.transpose(r, roller)
    scatter_shape = tf.concat((tf.shape(i)[0:1] + tf.shape(j)[0:1], tf.shape(rolled_t)[1:]), axis=0)
    print("blah", rolled_t.shape, rolled_r.shape, i.shape, j.shape, scatter_shape)
    scattered = tf.scatter_nd(i, rolled_t, scatter_shape) + tf.scatter_nd(j, rolled_r, scatter_shape)
    return tf.transpose(scattered, tf.roll(rank_indices, axis, 0))

def munge_other_dims(n, i, *args, other_dims_func):
    dim_index = tf.constant([i])
    other_dims_indices = tf.concat((tf.range(i), tf.range(i+1, n)),axis=0)
    print(args[-1].shape)
    other_dims = tf.stack(other_dims_func(*args), axis=0)
    this_dim = tf.expand_dims(args[-1],0)
    print("munging_dims", other_dims.shape, this_dim.shape)
    munged = tf_insert_row(other_dims, this_dim, i, axis=0)
    print("munged", munged.shape)
    return munged
    return tf_munge(other_dims, other_dims_indices, this_dim, dim_index, axis=0)

def arrange_parabolic_projection_funcs(n):
    import importlib
    import subprocess
    import functools
    try:
        importlib.import_module("parabolic_utils_{}".format(n))
    except ImportError:
        subprocess.call(["python3","-m","local_models.quadric_models","--parabolic","{}".format(n)])
    projection_funcs = importlib.import_module("parabolic_utils_{}".format(n))
    parabolic_projection_funcs = {
        "k_mats": [getattr(projection_funcs, "k_mat{:d}_{}".format(i,n)) for i in range(n)],
        "other_dims_as_xs": [getattr(projection_funcs, "other_dims_as_x{:d}_{}".format(i,n)) for i in range(n)]
    }
    regular_parabolic_projection_funcs = {
        "k_mat": parabolic_projection_funcs["k_mats"][0], 
        "other_dims_as_x": functools.partial(
            munge_other_dims, n, 0, other_dims_func=parabolic_projection_funcs["other_dims_as_xs"][0]),
    }
    special_parabolic_projection_funcs = {
        "k_mat": parabolic_projection_funcs["k_mats"][1],
        "other_dims_as_x": functools.partial( 
            munge_other_dims, n, 1, other_dims_func=parabolic_projection_funcs["other_dims_as_xs"][1]),    
    }
    return regular_parabolic_projection_funcs, special_parabolic_projection_funcs


regular_parabolic_projection_funcs_3, special_parabolic_projection_funcs_3 = arrange_parabolic_projection_funcs(3)



def tf_sorted_rotate_and_translate_quadric(Q,X):
    # this translates and rotates Q to have a diagonal upper left block, and to "center" on the point in X batch-wise
    # since the upper left block doesn't depend on X, we keep the blocks separate for efficiency
    Qs = Q[:-1,:-1]
    L, U = tf.linalg.eigh(Qs)
    eig_sorter = tf.argsort(tf.abs(L),axis=-1,direction='ASCENDING')
    eig_unsorter = tf.gather(tf.range(eig_sorter.shape[0]), eig_sorter)
    sorted_eigs = tf.gather(L, eig_sorter)
    sorted_eigvecs = tf.gather(U, eig_sorter)
    L = sorted_eigs
    U = sorted_eigvecs
    UL = L
    UTQs = tf.matmul(U,Qs,transpose_a=True)
    UTQsT = tf.transpose(UTQs,[1,0])
    print(UTQsT)
    XUTQsT = X@UTQsT
    print(XUTQsT)
    Q4U = Q[-1:,:-1]@U
    print(Q4U)
    UR = XUTQsT + Q4U
    BR = batch_dot(X@Qs,X) + 2*X@Q[:-1,-1:] + Q[-1,-1]
    print("L,U,UL,UR,BR",L,U,UL,UR,BR)
    return L,U,UL,UR,BR

def tf_parabolic_project_x(UL,UR,BR, projection_funcs = regular_parabolic_projection_funcs_3):
    UL = UL[...,1:]
    batch_shape = tf.shape(BR)[:-1]
    dims = UL.get_shape().as_list()
    ul_tiled = tf.tile(
        tf.reshape(UL, tf.concat([[1]*tf.rank(batch_shape), dims], axis=0)), 
        tf.concat([batch_shape, [1]*len(dims)], axis=0)
    )
    
    #TODO: this is the only custom bit for parabolic stuff.... we should refactor this crap
    args = (
        [ul_tiled[...,i:i+1] for i in range(UL.shape[-1])] + 
        [BR[...,0:1]] + 
        [UR[...,i:i+1] for i in range(UR.shape[-1])]
    )

    # kmat is the coeffs of [x^n, x^n-1, ... x^1, 1]
    for i, arg in enumerate(args):
        print("args{}".format(i),arg)
        print(arg.shape)

    coeffses = projection_funcs["k_mat"](*args)
    for i, coef in enumerate(coeffses):
        print("coeffses{}".format(i),coef)
        print(coef.shape)
    kmat = tf.stack(coeffses, axis=-1)
    print("kmat", kmat)
    squeezed_kmat = kmat[:,0,:]

    # if leading coefficients are 0 this is bad.... however, we can make them trailing coefficients and just get
    # a bunch of extra roots at 0.  We discard those, and call it a day
    roots = tf_roots_possible_zeros(squeezed_kmat, zero_coeff_tol=0)
    return args, roots
    
def tf_parabolic_other_dims(UL,UR,BR, projection_funcs = regular_parabolic_projection_funcs_3):
    args, roots = tf_parabolic_project_x(UL,UR,BR, projection_funcs = projection_funcs)

    other_dims = projection_funcs["other_dims_as_x"](*(list(args) + [roots]))
    #pts = tf.stack([roots] + other_dims, axis=-1)
    pts = tf_rollaxis(other_dims, -1)
    pts_nonan = tf.where(
        tf.logical_or(tf.is_nan(tf.real(pts)), tf.is_nan(tf.imag(pts))), 
        tf.zeros_like(pts), pts)
    return pts_nonan

def tf_parabolic_project_prerotated(UL,UR,BR,X,imag_0tol = 1e-10, projection_funcs = regular_parabolic_projection_funcs_3):
    pts = tf_parabolic_other_dims(UL,UR,BR, projection_funcs=projection_funcs)
    print("pts", pts)
    
    #ortho_pts, ortho_dists = tf_min_dist_search_complexandquadric_constraint(UL,UR,BR,pts,imag_0tol=imag_0tol,quadric_constraint_tol=quadric_constraint_tol)
    ortho_pts, ortho_dists = tf_min_dist_search_complex_constraint(pts,imag_0tol=imag_0tol)

    return ortho_pts, ortho_dists

def tf_parabolic_project(Q,X,imag_0tol=1e-10, regular_projection_funcs = regular_parabolic_projection_funcs_3, 
    special_projection_funcs = special_parabolic_projection_funcs_3):
    if not Q.dtype.is_complex:
        Qc = tf.cast(Q, tf.complex128 if Q.dtype == tf.float64 else tf.complex64)
    else:
        Qc = Q
    if not X.dtype.is_complex:
        Xc = tf.cast(X, tf.complex128 if X.dtype == tf.float64 else tf.complex64)
    else:
        Xc = X
    L,U,UL,UR,BR = tf_sorted_rotate_and_translate_quadric(Qc,Xc)
    along_x_axis = tf.abs(UR[:,0]) < 1.e-5
    not_along_x_axis = tf.logical_not(along_x_axis)
    regular_pts_UR = tf.boolean_mask(UR, not_along_x_axis, axis=0)
    special_pts_UR = tf.boolean_mask(UR, along_x_axis, axis=0)
    regular_pts_BR = tf.boolean_mask(BR, not_along_x_axis, axis=0)
    special_pts_BR = tf.boolean_mask(BR, along_x_axis, axis=0)
    regular_ortho_pts, regular_ortho_dists = tf.cond(tf.cast(tf.shape(regular_pts_UR)[0], tf.bool),
        lambda: tf_parabolic_project_prerotated(
            UL,regular_pts_UR,regular_pts_BR,Xc,imag_0tol=imag_0tol,projection_funcs=regular_projection_funcs),
        lambda: (tf.zeros(tf.shape(regular_pts_UR), dtype=Qc.dtype), tf.zeros(tf.shape(regular_pts_UR)[:1], dtype=Qc.dtype.real_dtype)),
    )
    print("regular_pts", regular_ortho_pts)
    special_ortho_pts, special_ortho_dists = tf.cond(tf.cast(tf.shape(special_pts_UR)[0], tf.bool), 
        lambda: tf_parabolic_project_prerotated(
            UL,special_pts_UR,special_pts_BR,Xc,imag_0tol=imag_0tol,projection_funcs=special_projection_funcs),
        lambda: (tf.zeros(tf.shape(special_pts_UR), dtype=Qc.dtype), tf.zeros(tf.shape(special_pts_UR)[:1], dtype=Qc.dtype.real_dtype)),
    )
    print("special_pts", special_ortho_pts)
    ortho_pts = tf_boolean_mask_inverse(along_x_axis, special_ortho_pts, regular_ortho_pts)
    ortho_dists = tf_boolean_mask_inverse(along_x_axis, special_ortho_dists, regular_ortho_dists)
    #ortho_pts, ortho_dists = regular_ortho_pts, regular_ortho_dists
    return tf.real(tf_unrotate_and_translate(U,Xc,ortho_pts)), tf.real(ortho_dists)#, blah


def tf_complextype(tf_floattype):
    return tf.complex128 if tf_floattype == tf.float64 else tf.complex64

def nptriu(x):
    dim = tf.cast((tf.sqrt(1+8*tf.cast(x.shape[0],tf.float32))-1)/2, tf.int32)
    def nptriu_help(i):
        return tf.concat((tf.zeros(i,dtype=x.dtype), x[i*(dim+1) - tf.div(i*(i+1),2):(i+1)*(dim+1) - tf.div((i+1)*(i+2),2)]), axis=-1)
    return tf.map_fn(nptriu_help, tf.range(dim), dtype=x.dtype)

def nptriu_inv_indices(x):
    dim = x.get_shape().as_list()[0]
    n_indices = tf.div(dim*(dim+1),2)
    indices = tf.range(n_indices)
    pair_indices_1 = tf.cast(dim - tf.sqrt(tf.cast(4*dim*dim + 4*dim - 8*indices + 1, tf.float32))/2 + 1/2, tf.int32)
    pair_indices_2 = pair_indices_1 + indices - (n_indices - tf.div((dim-pair_indices_1)*(dim-pair_indices_1+1),2))
    return tf.stack((pair_indices_1, pair_indices_2), axis=-1)

def unit_norm_constraint(fq):
    return fq/tf.norm(fq)

def parabolic_constraint(fq, unit_norm=True, input_shape=3):
    with tf.name_scope("parabolic_constraint"):
        upper_q = nptriu(fq)
        lower_q = tf.transpose(upper_q)
        q = upper_q + lower_q - tf.diag(tf.diag_part(upper_q))
        l,u = tf.linalg.eigh(lower_q[:-1,:-1])
        e = tf.concat(
            (
                tf.concat((u, tf.zeros((input_shape,1), u.dtype)), axis=-1),
                tf.concat((tf.zeros((1,input_shape),u.dtype), tf.ones((1,1),u.dtype)), axis=-1)
            ),
            axis=0)
        e_T = tf.transpose(e)
        q_rot = e_T@q@e
        eig_sorter = tf.argsort(tf.abs(l),axis=-1,direction='DESCENDING')
        eig_unsorter = tf.gather(tf.range(eig_sorter.shape[0]), eig_sorter)
        sorted_eigs = tf.gather(l, eig_sorter)
        print(sorted_eigs)
        constrained_sorted_eigs = tf.concat((sorted_eigs[:-1],[0]), axis=0)
        print(constrained_sorted_eigs)
        constrained_eigs = tf.gather(constrained_sorted_eigs, eig_unsorter)
        constrained_q_rot = tf.concat(
            (tf.concat((tf.diag(constrained_eigs), q_rot[:-1,-1:]), axis=-1),
            q_rot[-1:,:]), axis=0)
        print(constrained_q_rot.shape)
        constrained_q = e@constrained_q_rot@e_T
        if unit_norm:
            constrained_q = constrained_q / tf.norm(constrained_q)
        constrained_fq = tf.gather_nd(constrained_q, nptriu_inv_indices(constrained_q))
        return constrained_fq 

def design_x(x):
    return np.concatenate((x,np.ones(list(x.shape[:-1])+[1])),axis=-1)

class fuckit(object):
    def __init__(self, message=None):
        self.message = message
    def __enter__(self): return self
    def __exit__(self, *args):
        if self.message is not None:
            print(self.message)
        return True


class TFParabolic(sklearn.base.BaseEstimator, tensorflow_models.PickleableTFModel):
    ''' 
        Parameters:
            n_epochs: number of training epochs
            learning_rate: make it bigger to learn faster, at the risk of killing your relu
            trainable: set to false if you want to not allow training.  For example, if you want to use this as part of another network
            batch_size: defaults to the entire dataset.
    '''
    def __init__(self, n_epochs=300, learning_rate=0.05, trainable=True, batch_size=None, loss_trim=0.9,
        log_epochs=False, model=None, weighted=True, input_shape=3, normalize_weights=True,
        projection_funcs = (regular_parabolic_projection_funcs_3, special_parabolic_projection_funcs_3)):
        
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.trainable = trainable
        self.batch_size = batch_size
        self.log_epochs = log_epochs
        self.model = model
        self.input_shape = input_shape
        self.loss_trim = loss_trim
        self.projection_funcs = projection_funcs
        self.weighted = weighted
        self.normalize_weights = normalize_weights

        if self.model is None:
            print("building!")
            self.build_model()
        self.get_global_sess()
        self.init_vars_(beta0=np.zeros(int((self.input_shape+2)*(self.input_shape+1)/2)))
        
        super().__init__()

    def fit(self, X, y=None, sample_weight=None, beta0=None,**fit_params):
        ''' Puts inputs into np format, initializes a session, builds the graph and then calls `.fit_` which can be overridden by individual models '''

        X = sklearn.utils.check_array(X)
        self.init_vars_(X,y,sample_weight,beta0)
    
        with self.model.graph_.as_default():
            #self.blah = self.sess.run(self.model.blah, feed_dict={self.model.x:X})
            fitted_model = self.fit_(X, sample_weight=sample_weight, **fit_params)
            self.coef_ = self.sess.run(self.model.fq)
            self.intercept_ = np.array([])
        
        return fitted_model

    def init_vars_(self, X=None, y=None, sample_weight=None, beta0=None):
    
        if beta0 is None:
            beta0 = self.fast_reasonable_q(X,sample_weight)
 
        #beta0 /= beta0[-1] #TODO If the point lies ON the quadric, the remaining variables go to inf!!!! This is probably a problem for the iterated method
        #self.init_q = beta0[:-1] 
        self.init_q = beta0#/np.linalg.norm(beta0)
    
        with self.model.graph_.as_default():
            self.sess.run(self.model.initializer)
            self.sess.run(self.model.fq_feed, feed_dict={self.model.fq_placeholder: self.init_q})

    def predict(self, X, return_dists=False):
        X = sklearn.utils.check_array(X)

        with self.model.graph_.as_default():
            self.sess.run(self.model.fq_feed, feed_dict={self.model.fq_placeholder: self.coef_})
            projections, dists = self.sess.run((self.model.orthogonal_projections, self.model.dists), feed_dict={self.model.x:X})
                
        if return_dists:
            return projections, dists
        return projections
    
    def get_global_sess(self):
        global TFQuadricSesh
        try:
            TFQuadricSesh
        except:
            TFQuadricSesh = tf.Session(graph = self.model.graph_)
        self.sess = TFQuadricSesh
    
    def get_global_graph(self):
        global TFQuadricGraph
        try:
            TFQuadricGraph
        except:
            TFQuadricGraph = tf.Graph()
        self.model.graph_ = TFQuadricGraph

    def build_model(self):
        ''' Initializes a new graph, and then calls the .build_model_ method, which must be implemented by a TFEstimator '''
        self.model = tensorflow_models.NameSpace()
        self.get_global_graph()
        with self.model.graph_.as_default():
            self.build_model_()
            
    def build_model_(self):
        ''' The actual network architecture '''

        fq_shape = int((self.input_shape+2)*(self.input_shape+1)/2)
        with tf.name_scope("quadric_regression"):

            with tf.name_scope("input"):
                input_dim = [None, self.input_shape]
                self.model.x = tf.placeholder(tf.complex128, shape=input_dim, name="input")
                #self.model.init_q = tf.placeholder(tf.float32, shape=int((self.input_shape + 2)*(self.input_shape + 1)/2 - 1), name="initial_quadric")
                if self.weighted:
                    self.model.sample_weight = tf.placeholder(tf.float64, shape=[None], name="sample_weight")
                    weights = self.model.sample_weight

            with tf.name_scope("quadric_projection"):
                self.model.fq = tf.Variable(np.zeros(fq_shape), name="flattened_quadric", trainable=self.trainable, dtype=tf.float64, 
                    constraint=functools.partial(parabolic_constraint, input_shape=self.input_shape, unit_norm=True))
                self.model.fq_placeholder = tf.placeholder(tf.float64, shape=self.model.fq.shape)
                self.model.fq_feed = self.model.fq.assign(self.model.fq_placeholder)
                q = nptriu(self.model.fq)
                self.model.q = q + tf.transpose(q) - tf.diag(tf.diag_part(q))
                if not self.model.q.dtype.is_complex:
                    self.model.qc = tf.cast(self.model.q, tf.complex128 if self.model.q.dtype == tf.float64 else tf.complex64)
                else:
                    self.model.qc = self.model.q
                self.model.orthogonal_projections, self.model.complex_dists = tf_parabolic_project(
                    self.model.qc, self.model.x, imag_0tol=1.e-4, 
                    regular_projection_funcs = self.projection_funcs[0], 
                    special_projection_funcs = self.projection_funcs[1])

                self.model.dists = tf.real(self.model.complex_dists)
                if self.loss_trim < 1:
                    trimmer = tf.cast(self.loss_trim*tf.cast(tf.shape(self.model.dists)[0], tf.float32), tf.int32)
                    dist_sorter = tf.argsort(self.model.dists)
                    self.model.dists = tf.gather(self.model.dists, dist_sorter)
                    self.model.dists = self.model.dists[:trimmer]
                    if self.weighted:
                        weights = tf.gather(weights, dist_sorter)
                        weights = weights[:trimmer]
                        if self.normalize_weights:
                            weights = weights/tf.reduce_sum(weights)*tf.cast(trimmer, weights.dtype)
      
            if self.trainable:
                with tf.name_scope("training"):
                    if not self.weighted:
                        self.model.loss = tf.reduce_sum(self.model.dists, name="loss")
                    else:
                        self.model.loss = tf.reduce_sum(self.model.dists*weights, name="loss")
                    #self.model.loss = tf.log(self.model.loss) #TODO: this is kinda jank.
                    
                    #self.model.optimizer = tf.train.AdamOptimizer(self.learning_rate, name="optimizer")
                    self.model.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate, name="optimizer")
                    self.model.grads = self.model.optimizer.compute_gradients(self.model.loss)
                    self.model.grad_application = self.model.optimizer.apply_gradients(self.model.grads)
                    with tf.control_dependencies([self.model.grad_application]):
                        self.model.train_step = tf.no_op(name="train_step")

            self.model.initializer = tf.global_variables_initializer()

        return self.model     

    def fast_reasonable_q(self, X, sample_weight=None):
        #this allows us to orient the data to set an initial set of parameters
        global_linear_model = TLS_models.LinearODR_mD(self.input_shape-1)
        global_linear_model.fit(X, sample_weight=sample_weight)
        global_linear_vecs = global_linear_model.cov_eigenvectors[global_linear_model.cov_eigenvalues_sorter]
        global_linear_variance = global_linear_model.cov_eigenvalues[global_linear_model.cov_eigenvalues_sorter]
        global_linear_mean = global_linear_model.intercept_

        transformed_X = (X-global_linear_mean)@global_linear_vecs.T

        transformed_q = np.zeros((self.input_shape+1, self.input_shape+1))

        # initialize to the best plane
        transformed_q[self.input_shape,:self.input_shape] = transformed_q[:self.input_shape,self.input_shape] = global_linear_vecs[-1]/2
        transformed_q[self.input_shape,self.input_shape] = -np.dot(global_linear_vecs[-1], global_linear_mean)

        #test stuff
        #TODO this needs something more permanent
        transformed_q[np.diag_indices(self.input_shape)] = 1/np.sum(global_linear_variance)*np.arange(self.input_shape)
        transformed_q[0,0] = 0

        #E = np.block([[global_linear_vecs, global_linear_mean.reshape(-1,1)],[np.zeros(self.input_shape),1]])
        #Einv = np.linalg.inv(E)
        #q = Einv.T@transformed_q@Einv
        #q = q / np.linalg.norm(q)
        q = transformed_q

        beta0 = q[np.triu_indices(self.input_shape+1)]

        return beta0

    def fit_(self, X, sample_weight=None, feed_dict_extras={}):
        ''' Trains for a number of epochs.  Model input must be in self.model.x, output in self.model.y, loss in self.model.loss, and training using self.model.train_step '''

        logger.info("fitting quadric model")


        self.fitted_qs = []
        log_str = "epoch: {:06d} ::: loss: {:.02e}"
        for epoch in range(self.n_epochs):
            batcher = tensorflow_models.np_batcher(X.shape[0], self.batch_size)
            epoch_loss = 0
            for batch in batcher:
            
                fitted_q = np.zeros((self.input_shape+1,self.input_shape+1))
                fitted_q[np.triu_indices(self.input_shape+1)] = self.sess.run(self.model.fq)
                fitted_q = fitted_q.T
                self.fitted_qs.append(fitted_q)
                #if (self.log_epochs): logger.info("fitted_q:: {}".format(str(fitted_q)))
                
            
                feed_dict={
                    self.model.x : X[batch]
                }
                if self.weighted:
                    feed_dict[self.model.sample_weight] = sample_weight[batch]
                feed_dict.update(feed_dict_extras)
                
                loss, _ = self.sess.run(
                    (self.model.loss, self.model.train_step),
                    feed_dict
                )
                
                epoch_loss += loss.sum()

            
            if self.log_epochs and not (isinstance(self.log_epochs, float) and epoch%int(1/self.log_epochs)):
                logger.info(log_str.format(epoch, epoch_loss.sum()))
                
        logger.info("finished fitting :::: loss: " + str(loss.sum()))
        self.final_loss = loss.sum()

        return self


class TFQuadric(sklearn.base.BaseEstimator, tensorflow_models.PickleableTFModel):
    ''' A zero hidden layer NN Classifier (i.e. Logit Regression).
        Parameters:
            n_epochs: number of training epochs
            learning_rate: make it bigger to learn faster, at the risk of killing your relu
            trainable: set to false if you want to not allow training.  For example, if you want to use this as part of another network
            batch_size: defaults to the entire dataset.
    '''
    def __init__(self, n_epochs=300, learning_rate=0.05, trainable=True, batch_size=None, 
        log_epochs=False, model=None, weighted=True, input_shape=3, parabolic_constraint=False):
        global TFQuadricGraph
        global TFQuadricSesh
        try:
            TFQuadricGraph
        except:
            TFQuadricGraph = tf.Graph()
        try:
            TFQuadricSesh
        except:
            TFQuadricSesh = tf.Session(graph = TFQuadricGraph)
        self.sess = TFQuadricSesh
        self.graph = TFQuadricGraph
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.trainable = trainable
        self.batch_size = batch_size
        self.log_epochs = log_epochs
        self.model = model
        self.input_shape = input_shape
        self.weighted = weighted
        #self.init_q = np.zeros((9,))
        self.init_q = np.zeros((10,))
        self.parabolic_constraint = parabolic_constraint
        if self.model is None:
            print("building!")
            self.build_model()
        
        super().__init__()

    def fit(self, X, y=None, sample_weight=None, beta0=None,**fit_params):
        ''' Puts inputs into np format, initializes a session, builds the graph and then calls `.fit_` which can be overridden by individual models '''

        X = sklearn.utils.check_array(X)
        self.init_vars_(X,y,sample_weight,beta0)
    
        with self.graph.as_default():
            #self.blah = self.sess.run(self.model.blah, feed_dict={self.model.x:X})
            fitted_model = self.fit_(X, sample_weight=sample_weight, **fit_params)
            self.coef_ = self.sess.run(self.model.fq)
            self.intercept_ = np.array([])
        
        return fitted_model

    def init_vars_(self, X, y=None, sample_weight=None, beta0=None):
        if beta0 is None:
            beta0 = self.fast_reasonable_q(X,sample_weight)
 
        #beta0 /= beta0[-1] #TODO If the point lies ON the quadric, the remaining variables go to inf!!!! This is probably a problem for the iterated method
        #self.init_q = beta0[:-1] 
        self.init_q = beta0#/np.linalg.norm(beta0)
        #print(self.init_q.shape)
    
        with self.graph.as_default():
            self.sess.run(self.model.initializer)
            self.sess.run(self.model.fq_feed, feed_dict={self.model.fq_placeholder: self.init_q})

    def predict(self, X, return_dists=False):
        X = sklearn.utils.check_array(X)

        with self.graph.as_default():
            #print(self.sess.run(self.model.fq))
            self.sess.run(self.model.fq_feed, feed_dict={self.model.fq_placeholder: self.coef_})
            #print(self.sess.run(self.model.fq))
            projections, dists = self.sess.run((self.model.orthogonal_projections, self.model.dists), feed_dict={self.model.x:X})
                
        if return_dists:
            return projections, dists
        return projections

    def build_model(self):
        ''' Initializes a new graph, and then calls the .build_model_ method, which must be implemented by a TFEstimator '''
        self.model = tensorflow_models.NameSpace()
        with self.graph.as_default():
            self.build_model_()
            
    def build_model_(self):
        ''' The actual network architecture '''
        with tf.name_scope("quadric_regression"):

            with tf.name_scope("input"):
                input_dim = [None, self.input_shape]
                self.model.x = tf.placeholder(tf.complex128, shape=input_dim, name="input")
                if self.weighted:
                    self.model.sample_weight = tf.placeholder(tf.float64, shape=[None], name="sample_weight")

            with tf.name_scope("quadric_projection"):
                if self.parabolic_constraint:
                    self.model.fq = tf.Variable(self.init_q, name="flattened_quadric", trainable=self.trainable, 
                        constraint=parabolic_constraint)
                else:
                    self.model.fq = tf.Variable(self.init_q, name="flattened_quadric", trainable=self.trainable,
                        constraint=unit_norm_constraint)
                print(self.model.fq)
                self.model.fq_placeholder = tf.placeholder(tf.float64, shape=self.init_q.shape)
                self.model.fq_feed = self.model.fq.assign(self.model.fq_placeholder)
                q = nptriu(self.model.fq)
                #TODO: I removed a /2 here and shit fell apart
                self.model.q = q + tf.transpose(q) - tf.diag(tf.diag_part(q))
                print(self.model.q)
                if not self.model.q.dtype.is_complex:
                    self.model.qc = tf.cast(self.model.q, tf.complex128 if self.model.q.dtype == tf.float64 else tf.complex64)
                else:
                    self.model.qc = self.model.q
                print(self.model.qc)
                self.model.orthogonal_projections, self.model.complex_dists = tf_ortho_project(
                    self.model.qc, self.model.x, imag_0tol=np.inf)
                print(self.model.orthogonal_projections)
                print(self.model.complex_dists)
                self.model.dists = tf.real(self.model.complex_dists)
      
            if self.trainable:
                with tf.name_scope("training"):
                    if not self.weighted:
                        self.model.loss = tf.reduce_sum(self.model.dists, name="loss")
                    else:
                        self.model.loss = tf.reduce_sum(self.model.dists*self.model.sample_weight, name="loss")
                    #self.model.loss = tf.log(self.model.loss) #TODO: this is kinda jank.
                    
                    self.model.optimizer = tf.train.AdamOptimizer(self.learning_rate, name="optimizer")
                    self.model.grads = self.model.optimizer.compute_gradients(self.model.loss)
                    self.model.grad_application = self.model.optimizer.apply_gradients(self.model.grads)
                    with tf.control_dependencies([self.model.grad_application]):
                        self.model.train_step = tf.no_op(name="train_step")

            self.model.initializer = tf.global_variables_initializer()

        return self.model     

    def fast_reasonable_q(self, X, sample_weight=None):
        #this allows us to orient the data to set an initial set of parameters
        global_linear_model = TLS_models.LinearODR_mD(2)
        global_linear_model.fit(X, sample_weight=sample_weight)
        global_linear_vecs = global_linear_model.cov_eigenvectors[global_linear_model.cov_eigenvalues_sorter]
        global_linear_std = global_linear_model.cov_eigenvalues[global_linear_model.cov_eigenvalues_sorter]
        global_linear_mean = global_linear_model.intercept_
        #print(global_linear_std)

        transformed_X = (X-global_linear_mean)@global_linear_vecs.T

        transformed_q = np.zeros((4,4))

        if False:
            # initialize to the best elliptic paraboloid
            transformed_q[np.diag_indices(3)] = 1/global_linear_model.cov_eigenvalues[global_linear_model.cov_eigenvalues_sorter]**2
            transformed_q[:3,-1] = transformed_q[-1,:3] = 1/global_linear_model.cov_eigenvalues[global_linear_model.cov_eigenvalues_sorter]
            transformed_q[3,3] = np.sum(global_linear_mean)
            transformed_q[2,2] = 0
        if self.parabolic_constraint:
            # initialize to the best plane
            transformed_q[3,:3] = transformed_q[:3,3] = global_linear_vecs[-1]
            transformed_q[3,3] = np.dot(global_linear_vecs[-1], -global_linear_mean)
            
            #test stuff
            transformed_q[np.diag_indices(3)] = 0.0001
            transformed_q[0,0] = 0

            #print(transformed_q)
            #transformed_q[:3,:3] = np.array([[0,1,1],[0,0,4],[0,0,3]])
            #transformed_q[:3,:3] = np.array([[0,1,1],[0,1,4],[0,0,3]])
        else:
            # an ellipse with axes in the various directions
            transformed_q[np.diag_indices(3)] = 1/global_linear_std
            # shift so that the ellipse hits the origin at the fitted plane.  TODO: do we want the ellipse to open up or down??
            #transformed_q[:3,-1] = transformed_q[-1,:3] = -2*np.sqrt(global_linear_std)
            transformed_q[2,-1] = transformed_q[-1,2] = -2/np.sqrt(global_linear_std[2])
            #transformed_q[3,3] = np.sum(
            #    transformed_q[np.diag_indices(3)],
            #    global_linear_mean**2
            #)
            transformed_q[3,3] = -1
        
        #print(transformed_q)
        E = np.block([[global_linear_vecs, global_linear_mean.reshape(-1,1)],[np.zeros(self.input_shape),1]])
        Einv = np.linalg.inv(E)
        q = Einv.T@transformed_q@Einv
        q = q / np.linalg.norm(q)
        #print(q)
        beta0 = q[np.triu_indices(self.input_shape+1)]
        #print(beta0.shape)
        return beta0

    def fit_(self, X, sample_weight=None, feed_dict_extras={}):
        ''' Trains for a number of epochs.  Model input must be in self.model.x, output in self.model.y, loss in self.model.loss, and training using self.model.train_step '''

        logger.info("fitting quadric model")


        self.fitted_qs = []
        for epoch in range(self.n_epochs):
            #print(epoch)
            batcher = tensorflow_models.np_batcher(X.shape[0], self.batch_size)

            for batch in batcher:
            
                fitted_q = np.zeros((self.input_shape+1,self.input_shape+1))
                fitted_q[np.triu_indices(self.input_shape+1)] = self.sess.run(self.model.fq)
                fitted_q = fitted_q.T
                self.fitted_qs.append(fitted_q)
            
                feed_dict={
                    self.model.x : X[batch]
                }
                if self.weighted:
                    feed_dict[self.model.sample_weight] = sample_weight[batch]
                feed_dict.update(feed_dict_extras)
                
                loss, grads, _ = self.sess.run(
                    (self.model.loss, self.model.grads, self.model.train_step),
                    feed_dict
                )
                
                log_str = "epoch: {:06d} ::: loss: {:08.02f} ::: grad {}"
                if self.log_epochs: logger.info(log_str.format(epoch, loss.sum(), 
                    ["{:06.02f}".format(grad) for grad in grads[0][0]]))
                
        logger.info("finished fitting :::: loss: " + str(loss.sum()))
        self.final_loss = loss.sum()

        return self








def test_parabolic_constraint():
    parabolic_q = np.array([[1,0,0,4],[0,0,0,5],[0,0,3,6],[4,5,6,7]])
    unparabolic_q = np.array([[1,0,0,4],[0,-1e-1,0,5],[0,0,3,6],[4,5,6,7]])

    u = np.arange(9).reshape(3,3)
    u += u.T
    _, u = np.linalg.eig(u)

    x = np.arange(12).reshape(4,3)*1.
    parabolic_q[:3,:3] = u@parabolic_q[:3,:3]@u.T
    parabolic_q = parabolic_q / np.linalg.norm(parabolic_q)
    unparabolic_q[:3,:3] = u@unparabolic_q[:3,:3]@u.T
    unparabolic_q = unparabolic_q / np.linalg.norm(unparabolic_q)
    
    with tf.Graph().as_default(), tf.Session() as sess:
        parabolic_constraint_test = sess.run(
            parabolic_constraint(
                tf.constant(
                    unparabolic_q[np.triu_indices(4)], 
                    dtype=tf.float32),
                unit_norm=True,
            ))
    assert np.allclose(parabolic_constraint_test, parabolic_q[np.triu_indices(4)])

def test_tf_munge():
    # test tf_munge
    munge1 = np.arange(12).reshape((4,3))
    munge2 = np.arange(12,20).reshape((4,2))
    munge1_indices = np.array([[1,2,4]])[0]
    munge2_indices = np.array([[0,3]])[0]
    with tf.Graph().as_default() as g, tf.Session() as sess:
        test_munge1 = tf.constant(munge1)
        test_munge2 = tf.constant(munge2)
        test_i1 = tf.constant(munge1_indices, dtype=tf.int32)
        test_i2 = tf.constant(munge2_indices, dtype=tf.int32)
        res = sess.run(tf_munge(test_munge1, test_i1, test_munge2, test_i2, axis=1))
    assert np.allclose(res,
        np.array([[12,  0,  1, 13,  2],
           [14,  3,  4, 15,  5],
           [16,  6,  7, 17,  8],
           [18,  9, 10, 19, 11]]))

def test_circular_permutation():
    # test circular permutation
    to_rot = np.arange(12).reshape((4,3))
    rot_by = np.array([0,2,-1,0])
    with tf.Graph().as_default() as g, tf.Session() as sess:
        test_to_rot = tf.constant(to_rot)
        test_rot_by = tf.constant(rot_by)
        res = sess.run(tf_circular_permutation(test_to_rot, test_rot_by))
    assert np.allclose(res, np.array([[ 0,  1,  2],
           [ 5,  3,  4],
           [ 8,  6,  7],
           [ 9, 10, 11]]))

def test_min_dist_search():
    # test min_dist_search
    with tf.Graph().as_default(), tf.Session() as sess:
        bar = np.array([
            [[2,3,4],[1,2,50.01]],
            [[4,5,8],[4,5,7]],
            [[100,100,100],[1+1j,1,1]]])[:,np.newaxis,:,:]
        test_roots = tf.placeholder(tf.complex128, bar.shape)
        sol_tf = tf_min_dist_search(test_roots,1e-1)
        res_pts, res_dists = sess.run(sol_tf, feed_dict={test_roots: bar})
        assert np.allclose(res_pts[:,0,:], 
            np.array([[2., 3., 4.],
                      [4., 5., 7.],
                      [100,100,100]]))

def test_rotate_and_translate_quadric():
    # test rotate_and_translate_quadric
    q = np.array([ # an elliptic paraboloid
        [0.5,0,0,0],
        [0,0.25,0,0],
        [0,0,0.1,-1],
        [0,0,-1,0.1],
    ])
    u = np.arange(9).reshape(3,3)
    u += u.T
    _, u = np.linalg.eig(u)

    x = np.arange(12).reshape(4,3)*1.

    E = np.block([[u, x[0].reshape(-1,1)],[np.zeros(3), 1]])
    Einv = np.linalg.inv(E)
    q_test = Einv.T@q@Einv
    with tf.Graph().as_default(), tf.Session() as sess:

        test_qs = tf.placeholder(q.dtype, q.shape)
        test_pts = tf.placeholder(x.dtype, x.shape)
        sol_tf = tf_rotate_and_translate_quadric(test_qs, test_pts)
        l,u,ul,ur,br = sess.run(sol_tf, feed_dict={test_pts: x, test_qs: q_test})
    res_q = np.block([[np.diag(ul), ur[0:1].T],[ur[0:1],br[0]]])
    ee = np.block([[u, x[0].reshape(-1,1)],[np.zeros(3),1]])
    eeinv = np.linalg.inv(ee)
    assert(np.allclose(eeinv.T@res_q@eeinv, q_test))

def test_quadric_ortho_projection():
    # test quadric_ortho_projection
    q = np.array([ # an elliptic paraboloid
        [0.5,0,0,0],
        [0,0.25,0,0],
        [0,0,0.1,-1],
        [0,0,-1,0.1],
    ])
    u = np.arange(9).reshape(3,3)
    u += u.T
    _, u = np.linalg.eig(u)

    x = np.arange(12).reshape(4,3)*1.

    E = np.block([[u, x[0].reshape(-1,1)],[np.zeros(3), 1]])
    Einv = np.linalg.inv(E)
    q_test = Einv.T@q@Einv
    with tf.Graph().as_default(), tf.Session() as sess:
        test_qs = tf.placeholder(tf.float32, q.shape)
        test_pts = tf.placeholder(tf.float32, x.shape)
        test_qsc = tf.cast(test_qs, tf.complex128 if test_qs.dtype == tf.float64 else tf.complex64)
        test_ptsc = tf.cast(test_pts, tf.complex128 if test_pts.dtype == tf.float64 else tf.complex64)
        sol_tf = tf_ortho_project(test_qsc, test_ptsc, imag_0tol=1e-3)
        res = sess.run(sol_tf, feed_dict={test_pts: x, test_qs: q_test})
    test_res = orthogonal_quadric_projection(x,q_test,projection_funcs)
    for i, r in enumerate(test_res):
        assert np.allclose(r, res[i], atol=1e-4, rtol=1e-4)

def test_rotate_and_translate_quadric_planar():
    planar_q = np.array([ # an elliptic paraboloid
        [0,0,0,1],
        [0,0,0,-2],
        [0,0,0,-1],
        [1,-2,-1,0.1],
    ])
    x = np.arange(12).reshape(4,3)*1.
    arbitrary_rot = np.arange(9).reshape(3,3)
    arbitrary_rot += arbitrary_rot.T
    _, eigen_rot = np.linalg.eig(arbitrary_rot)

    E = np.block([[eigen_rot, x[0].reshape(-1,1)],[np.zeros(3), 1]])
    Einv = np.linalg.inv(E)
    planar_q_rot = Einv.T@planar_q@Einv

    with tf.Session('') as sess:
        test_qs = tf.placeholder(tf.float32, planar_q.shape)
        test_pts = tf.placeholder(tf.float32, x.shape)
        test_qsc = tf.cast(test_qs, tf.complex128 if test_qs.dtype == tf.float64 else tf.complex64)
        test_ptsc = tf.cast(test_pts, tf.complex128 if test_pts.dtype == tf.float64 else tf.complex64)
        tatered = tf_rotate_and_translate_quadric(test_qsc, test_ptsc)
        l,u,ul,ur,br = sess.run(tatered, feed_dict={test_pts: x, test_qs: planar_q_rot})

    res_q = np.block([[np.diag(ul), ur[0:1].T],[ur[0:1],br[0]]])
    ee = np.block([[u, x[0].reshape(-1,1)],[np.zeros(3),1]])
    eeinv = np.linalg.inv(ee)
    assert(np.allclose(eeinv.T@res_q@eeinv, planar_q_rot))

def test_quadric_ortho_projection_planar():
    planar_q = np.array([ # an elliptic paraboloid
        [0,0,0,1],
        [0,0,0,-2],
        [0,0,0,-1],
        [1,-2,-1,0.1],
    ])
    x = np.arange(12).reshape(4,3)*1.
    with tf.Graph().as_default(), tf.Session() as sess:
        test_qs = tf.placeholder(tf.float32, planar_q.shape)
        test_pts = tf.placeholder(tf.float32, x.shape)
        test_qsc = tf.cast(test_qs, tf.complex128 if test_qs.dtype == tf.float64 else tf.complex64)
        test_ptsc = tf.cast(test_pts, tf.complex128 if test_pts.dtype == tf.float64 else tf.complex64)
        sol_tf = tf_ortho_project(test_qsc, test_ptsc, imag_0tol=1e-3)
        res = sess.run(sol_tf, feed_dict={test_pts: x, test_qs: planar_q})
    test_res = orthogonal_quadric_projection(x,planar_q,projection_funcs)
    for i, r in enumerate(test_res):
        assert np.allclose(r, res[i], atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_parabolic_constraint()
    test_min_dist_search()
    test_rotate_and_translate_quadric()
    test_rotate_and_translate_quadric_planar()
    test_quadric_ortho_projection()
    test_quadric_ortho_projection_planar()
