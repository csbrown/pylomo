import numpy as np
import sklearn.neighbors
import sklearn.pipeline
import sklearn.svm
import sklearn.decomposition
import sklearn.gaussian_process
import logging
import inspect
from . import TLS_models
import functools
import collections
import scipy

def get_learned_param_vector_PCA(model):
    params = model.components_.reshape((1,-1))
    if params[0,0] < 0: # because PCA returns a vector or it's opposite arbitrarily
        params *= -1
    return params
def set_learned_param_vector_PCA(model, params):
    model.components_ = params.reshape((model.n_components_, -1))
    return model

def get_learned_param_vector_Pipeline(model):
    res = np.array([[]])
    for name, step in model.steps:
        res = np.concatenate((res, get_learned_param_vector(step)), axis=1)
    return res
def set_learned_param_vector_Pipeline(model, params):
    pass

def get_learned_param_vector_GaussianProcessRegressor(model):
    if hasattr(model, "kernel_"):
        return model.kernel_.theta
    return model.kernel.theta
def set_learned_param_vector_GaussianProcessRegressor(model, params):
    if hasattr(model, "kernel_"):
        model.kernel_.theta = params
    model.kernel.theta = params

def get_learned_param_vector_StandardScaler(model):
    if model.var_ is None:
        return model.mean_.reshape((1,-1))
    else:
        return np.concatenate((model.mean_.reshape((1,-1)), model.mean_.reshape((1,-1))), axis=1)
def set_learned_param_vector_StandardScaler(model, params):
    pass

def set_learned_param_vector_LinearODR(model,params):
    model.coef_ = params
    return model
def get_learned_param_vector_LinearODR(model):
    return np.concatenate((model.coef_.reshape((1,-1)), model.intercept_.reshape((1,-1))),axis=1)

def set_learned_param_vector_SVClinear(model,params):
    model.coef_ = params
    return model
def get_learned_param_vector_SVClinear(model):
    return np.concatenate((model.coef_.reshape((1,-1)), model.intercept_.reshape((1,-1))),axis=1)
    
def set_learned_param_vector(model, params):
    if isinstance(model, sklearn.decomposition.PCA):
        return set_learned_param_vector_PCA(model, params)
    if isinstance(model, TLS_models.LinearODR):
        return set_learned_param_vector_LinearODR(model, params) 
    if isinstance(model, sklearn.gaussian_process.GaussianProcessRegressor):
        return set_learned_param_vector_GaussianProcessRegressor(model, params)
    if isinstance(model, sklearn.svm.SVC) and model.kernel=="linear":
        return set_learned_param_vector_SVClinear(model, params)
    coef, model.intercept_ = params[:,:-1], params[:,-1]
    if len(coef.shape) > 1 and coef.shape[1] == 1:
        model.coef_ = coef[:,0]
    else:
        model.coef_ = coef  
    return model

def get_learned_param_vector(model):
    if isinstance(model, sklearn.decomposition.PCA):
        return get_learned_param_vector_PCA(model)
    if isinstance(model, sklearn.pipeline.Pipeline):
        return get_learned_param_vector_Pipeline(model)
    if isinstance(model, sklearn.preprocessing.StandardScaler):
        return get_learned_param_vector_StandardScaler(model)
    if isinstance(model, TLS_models.LinearODR):
        return get_learned_param_vector_LinearODR(model)
    if isinstance(model, sklearn.gaussian_process.GaussianProcessRegressor):
        return get_learned_param_vector_GaussianProcessRegressor(model)
    if isinstance(model, sklearn.svm.SVC) and model.kernel=="linear":
        return get_learned_param_vector_SVClinear(model)
    if len(model.coef_.shape) <= 1:
        coef = model.coef_.reshape((1,-1))
    else:
        coef = model.coef_
    return np.concatenate((coef, model.intercept_.reshape((1,-1))), axis=1)
