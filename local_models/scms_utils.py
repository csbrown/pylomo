from . import utils
import collections
import numpy as np

def linear_reject_pointwise_bases(x, bases, mean=0):
    x = x - mean #mean center everything
    projection = utils.linear_project_pointwise_bases(x, bases)
    rejection = x - projection
    rejection = rejection + mean #re-add the mean in
    return rejection

def scms(X, lm, kernel, iters=30, constraint_space=None, return_params=False, failure_delta=None):

    if failure_delta is None:
        failure_delta = np.average(lm.index.query(X, k=2)[0][:,1])*1e4
    for i in range(iters):
        X = np.copy(X)
        Xrange = np.arange(X.shape[0])
        params = lm.transform(X, r=kernel.support_radius(), weighted=True,
            kernel=kernel)
        normalized_params = params/np.sqrt(np.sum(params[:,:X.shape[1]]**2,axis=-1,keepdims=True))
        normals = normalized_params[:,:X.shape[1]]
        intercepts = normalized_params[:,X.shape[1]]
        biggest_normal_component = np.argmax(normals, axis=1)
        biggest_normal_component_indices = np.stack((Xrange, biggest_normal_component))
        biggest_normal_component_indices = tuple(map(tuple, biggest_normal_component_indices))

        plane_pt_component = -intercepts/normalized_params[biggest_normal_component_indices]
        plane_pts = np.zeros(normals.shape)
        plane_pts[biggest_normal_component_indices] = plane_pt_component

        normals = normals.reshape(X.shape[0], 1, X.shape[1])
        new_X = linear_reject_pointwise_bases(X, normals, plane_pts)
        failures = np.sqrt(np.sum((new_X-X)**2, axis=1)) > failure_delta
        successes = np.logical_not(failures)
        X[successes] = new_X[successes]
        if constraint_space is not None:
            X[successes] = utils.linear_project_pointwise_bases(X[successes], constraint_space[0][successes], constraint_space[1][successes])

        if return_params:
            yield X, successes, normals
        else:
            yield X, successes

def exhaust(gen):
    def exhauster(*args, **kwargs):
        for _ in gen(*args, **kwargs): pass
        return _
    return exhauster

import tempfile
import time
def scms_parallel_sharedmem(X, lm, kernel, iters=30, constraint_space=None, return_params=False, failure_delta=None, batch_size=100):
    batches = (np.array([0, batch_size]) + batch_size*i for i in range(int(np.ceil(X.shape[0]/batch_size))))
    with tempfile.NamedTemporaryFile(dir="/dev/shm") as shared_X_ramspace, tempfile.NamedTemporaryFile(dir="/dev/shm") as shared_constraint0_ramspace, tempfile.NamedTemporaryFile(dir="/dev/shm") as shared_constraint1_ramspace:
        shared_X = np.memmap(shared_X_ramspace, dtype=X.dtype,
                   shape=X.shape, mode='w+')
        shared_X[:] = X[:]
        if constraint_space is not None:
            shared_constraint_space0 = np.memmap(shared_constraint0_ramspace, dtype=constraint_space[0].dtype,
                   shape=constraint_space[0].shape, mode='w+')
            shared_constraint_space1 = np.memmap(shared_constraint1_ramspace, dtype=constraint_space[1].dtype,
                   shape=constraint_space[1].shape, mode='w+')
            shared_constraint_space0[:] = constraint_space[0][:]
            shared_constraint_space1[:] = constraint_space[1][:]
            constraint_space = (shared_constraint_space0, shared_constraint_space1)
        parallel_sols = joblib.Parallel(n_jobs=12)(joblib.delayed(exhaust(scms))(
            shared_X[slice(*batch)], lm, kernel, iters, 
            None if constraint_space is None else tuple(map(lambda c: c[slice(*batch)], constraint_space)), 
            return_params, failure_delta)
            for batch in batches)
    res = tuple(map(functools.partial(np.concatenate, axis=0), zip(*parallel_sols)))
    yield res
    
def scms_parallel(X, lm, kernel, iters=30, constraint_space=None, return_params=False, failure_delta=None, batch_size=100, n_jobs=12):
    batches = (np.array([0, batch_size]) + batch_size*i for i in range(int(np.ceil(X.shape[0]/batch_size))))
    parallel_sols = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(exhaust(scms))(
        X[slice(*batch)], lm, kernel, iters, 
        None if constraint_space is None else tuple(map(lambda c: c[slice(*batch)], constraint_space)), 
        return_params, failure_delta)
        for batch in batches)
    res = tuple(map(functools.partial(np.concatenate, axis=0), zip(*parallel_sols)))
    yield res



def orthogonal_project_scms(X, lm, kernel, scms_iters=30, newtons_iters=30, alpha=1e-2, return_everything=False, parallel=False, sharedmem=True, n_jobs=12, batch_size=100):
    #1. do scms to get *a* point on the surface, y
    #2. get the tangent plane at y
    scms_method = scms_parallel_sharedmem if (parallel and sharedmem) else (scms_parallel if parallel else scms)
    shifter = scms_method(X,lm,kernel,iters=scms_iters,return_params=True, n_jobs=n_jobs, batch_size=batch_size)
    for y, successes, normals in shifter: pass
    X = X[successes]
    y = y[successes]
    normals = normals[successes]
    yield X,y,normals
    Xy = y-X
    normalized_Xy = (Xy)/np.sqrt(np.sum(Xy**2,axis=1,keepdims=True))
    normalized_Xy = np.expand_dims(normalized_Xy, 1)
    surface_normal_aligned_Xy = normalized_Xy * np.sign(np.sum(normalized_Xy*normals, axis=-1, keepdims=True))
    constraint_vec = normalized_Xy
    #3. do scms while projecting along some convex combo of the line passing thru x and y, and 
    #   the line passing through x and along the normal vector to the tangent plane in 2 to get y'
    #4. y <- y'
    #5. GOTO 2
    for i in range(newtons_iters):
        constrained_shifter = scms_method(X,lm,kernel,iters=scms_iters,return_params=True,constraint_space=(constraint_vec, X), n_jobs=n_jobs, batch_size=batch_size)        
        for y, successes, normals in constrained_shifter: pass
        shifter = scms_method(y,lm,kernel,iters=scms_iters,return_params=True, n_jobs=n_jobs, batch_size=batch_size)
        for y, successes, normals in shifter: pass
        yield X,y,normals
        Xy = y-X
        normalized_Xy = (Xy)/np.sqrt(np.sum(Xy**2,axis=1,keepdims=True))
        normalized_Xy = np.expand_dims(normalized_Xy, 1)
        surface_normal_aligned_Xy = normalized_Xy * np.sign(np.sum(normalized_Xy*normals, axis=-1, keepdims=True))
        constraint_vec = surface_normal_aligned_Xy*(1-alpha) + normals*alpha
        constraint_vec = constraint_vec/np.sqrt(np.sum(constraint_vec**2, axis=-1, keepdims=True))
