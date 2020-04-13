import numpy as np
import sklearn.neighbors
import sklearn
import logging
import joblib
import inspect
from . import loggin
import collections
from .kernels import *
from .default_model_features import *
logger = logging.getLogger(__name__)
logger.info("local models!")

class ConstantDistanceSortedIndex(object):
    ''' This implements the same interface as BallTree, but is specifically for 1d, constant sample rate data X
        X must be sorted with constant distance '''
    def __init__(self, X):
        self.X = X
        self.d = X[1] - X[0]
    def query_radius(self, y, r, return_distance=False):
        insert_positions = y.reshape(-1) - self.X[0]
        indices = np.array([np.arange(max(0, int((i - r)/self.d+0.999999)), min(self.X.shape[0], int((i+r)/self.d+1))) for i in insert_positions])
        if return_distance:
            distances = np.array([np.abs(self.X[indices[i]] - y[i]).astype(float) for i in range(len(indices))])
            return indices, distances
        return indices


class LocalModels(object):
    ''' Implements an sklearn model interface, complete with fit and transform methods.
        This class is a convenient pipeline for building localized models over some training dataset,
            and making predictions or other types of transformations from those individual localized models.
        e.g.
    
        ### EXAMPLE: Train a LOESS model on a noisy sin curve, and extract the local model parameters
        from local_models.local_models import *
        import numpy as np
        import matplotlib.pyplot as plt

        X_train = np.linspace(0,6,100).reshape(-1,1)
        y_train = np.sin(X_train) + np.random.normal(loc=0,scale=0.3,size=X_train.shape)
        X_test = np.linspace(-1,7,1000).reshape(-1,1)

        kernel = GaussianKernel(bandwidth = 1.)
        LOESS = LocalModels(sklearn.linear_model.LinearRegression(), kernel=kernel)
        LOESS.fit(X_train,y_train) # This just builds an index and stores x and y

        y_pred = LOESS.predict(X_test) # This makes local predictions at these various points
        model_features = LOESS.transform(X_test)

        plt.plot(X_test, y_pred)
        plt.plot(X_test, model_features)
        plt.scatter(X_train, y_train,c='r')
        plt.legend(["predictions", "slope", "intercept", "data"])
        plt.show()
    ''' 


    def __init__(self, model, kernel=None):
        self.model = model
        self.k = None
        self.r = None
        if kernel is not None:
            self.kernel = kernel
            if kernel.bandwidth == 'knn':
                self.k = kernel.k
            else:
                self.r = kernel.support_radius()

    def fit(self, model_features, model_targets = None, model_cluster_features = None, sample_weight=None, index=None, ball_tree_kwargs={}):
        #self.model_features = sklearn.utils.check_array(model_features)
        self.model_features = model_features
        if model_targets is not None: model_targets = model_targets
        if sample_weight is not None: sample_weight = sklearn.utils.check_array(sample_weight)
        if model_cluster_features is None:
            model_cluster_features = model_features
        else:
            model_cluster_features = model_cluster_features
        self.sample_weight = sample_weight
        self.model_targets = model_targets
        self.model_features = model_features

        logger.info("building ball tree")
        if index is not None:
            self.index = index
        else:
            self.index = sklearn.neighbors.BallTree(model_cluster_features, **ball_tree_kwargs)
        logger.info("finished building ball tree")
        return self

    def _query2data(self, indices, distances):
        if indices.shape[0] == 0: #if there is no data, send in nan data because whattya do?
            X = np.empty([1] + list(self.model_features.shape[1:]))
            X[:] = np.nan
            if self.model_targets is None:
                y = None
            else:
                y = np.empty([1] + list(self.model_targets.shape[1:]))
                y[:] = np.nan
            weights = np.ones([1])
        else:
            X = self.model_features[indices]
            y = None if self.model_targets is None else self.model_targets[indices]
            weights = distances
        return X,y,weights

    def _parallel_transform(self, model_cluster_features, k=None, r=None, batch_size=None, model_postprocessor=None,
                kernel=None, return_models=False, weighted=False, neighbor_beta0s=None, beta0=None):
        batch_size = min(batch_size, model_cluster_features.shape[0])
        n_batches = model_cluster_features.shape[0]/batch_size
        batch = np.array([0,batch_size])
        timelog = loggin.TimeLogger(logger=logger, how_often=max(int(n_batches/100),1), total=n_batches, tag=type(self.model).__module__ + "." + type(self.model).__name__)
        parallel_sols = joblib.Parallel()(joblib.delayed(self._loop_transform)(
            model_cluster_features[slice(*batch)], iz, dz, weighted=weighted, return_models=return_models, neighbor_beta0s=neighbor_beta0s, beta0=beta0, model_postprocessor=model_postprocessor)  
            for batch, iz, dz in self._batch_query(model_cluster_features, k=k, r=r, batch_size=batch_size, kernel=kernel))
        transforms, models = zip(*parallel_sols)
        parameter_features = []
        i = 0
        while batch[0] < model_cluster_features.shape[0]:
            parameter_features.append(transforms[i])
            batch += batch_size; i += 1
        return np.concatenate(parameter_features, axis=0), [model for model_list in models for model in model_list]


    def _loop_transform(self, model_cluster_features, iz, dz, weighted=False, neighbor_beta0s=False, return_models=False, beta0=None, model_postprocessor=None):
        if model_postprocessor is None: model_postprocessor=lambda m,q,x,y,w: get_learned_param_vector(m)
        parameter_features = []
        the_models = []
        if beta0 is None:
            beta0s = collections.defaultdict(lambda: None)
        else:
            beta0s = beta0
        for j in range(iz.shape[0]):
            X,y,weights = self._query2data(iz[j], dz[j])
            try:
                model_clone = self._train_local_model(X,y,weights,weighted,beta0s[j])
            except ValueError as e:
                logger.info("failed at query point: {}".format(model_cluster_features[j]))
                raise e
            if return_models:
                the_models.append(model_clone)
            parameter_features.append(model_postprocessor(model_clone, model_cluster_features[j:j+1,...], X,y,weights).reshape(1,-1))
            if j>0 and neighbor_beta0s:
                beta0 = parameter_features[-1][0]
        return np.concatenate(parameter_features, axis=0), the_models

    def _batch_transform(self, model_cluster_features, k=None, r=None, batch_size=None, model_postprocessor=None,
                kernel=None, return_models=False, weighted=False, neighbor_beta0s=None, beta0=None):
        batch_size = min(batch_size, model_cluster_features.shape[0])
        n_batches = model_cluster_features.shape[0]/batch_size
        timelog = loggin.TimeLogger(logger=logger, how_often=max(int(n_batches/100),1), total=n_batches, tag=type(self.model).__module__ + "." + type(self.model).__name__)
        parameter_features = []
        the_models = []
        for batch, iz, dz in self._batch_query(model_cluster_features, k=k, r=r, batch_size=batch_size, kernel=kernel):
            with timelog:
                batch_features, models = self._loop_transform(model_cluster_features[slice(*batch)], iz, dz, weighted, model_postprocessor=model_postprocessor,
                    neighbor_beta0s=neighbor_beta0s,return_models=return_models,beta0=None if beta0 is None else beta0[slice(*batch)])
                if return_models:
                    the_models += models
                parameter_features.append(batch_features)
        return np.concatenate(parameter_features, axis=0), the_models

    def _train_local_model(self, X, y, weights, weighted=False, beta0=None):
        model_clone = sklearn.base.clone(self.model)

        fit_available_args = inspect.getfullargspec(model_clone.fit)
        n_fit_args = 0 if fit_available_args.args is None else len(fit_available_args.args)
        n_fit_defaults = 0 if fit_available_args.defaults is None else len(fit_available_args.defaults)
        fit_available_kwargs = fit_available_args.args[n_fit_args - n_fit_defaults:] + fit_available_args.kwonlyargs
        fit_kwargs = {}
        if weighted and ("sample_weight" in fit_available_kwargs):
            fit_kwargs["sample_weight"] = weights
        elif weighted and not ("sample_weight" in fit_available_kwargs):
            raise Exception("fit cannot accept weights, choose a different model or turn off weighting")
        if beta0 is not None:
            fit_kwargs["beta0"] = beta0
        model_clone.fit(X, y, **fit_kwargs)
        return model_clone
    
    def _batch_query(self, model_cluster_features, k=None, r=None, batch_size=100, kernel=None, neighbor_beta0s=False):
        k = k or self.k
        r = r or self.r
        kernel = kernel or self.kernel
        model_cluster_features = sklearn.utils.check_array(model_cluster_features)
        batch = np.array([0,batch_size])
        while batch[0] < model_cluster_features.shape[0]:
            if k is None:
                iz, dz = self.index.query_radius(model_cluster_features[slice(*batch)], r, return_distance=True)
            else:
                dz, iz = self.index.query(model_cluster_features[slice(*batch)], k=k, return_distance=True)
            if kernel is not None:
                for i in range(dz.shape[0]):
                    dz[i] = kernel(dz[i])
            yield batch, iz, dz
            batch += batch_size

    def transform(self, model_cluster_features, k=None, r=None, batch_size=100, parallel=False, weighted=True, kernel=None, neighbor_beta0s=False, return_models=False, beta0=None, model_postprocessor=None):
        transformator = self._parallel_transform if parallel else self._batch_transform
        parameter_features, models = transformator(model_cluster_features, k=k, r=r, batch_size=batch_size, 
            kernel=kernel, return_models=return_models, weighted=weighted, beta0=beta0, neighbor_beta0s=neighbor_beta0s, model_postprocessor=model_postprocessor)
        if return_models:
            return parameter_features, models
        return parameter_features

    def fit_transform(self, model_features, model_targets, model_cluster_features=None, sample_weight=None, k=None, r=None, weighted=False, ball_tree_kwargs={}, batch_size=100, parallel=False, kernel=None, beta0=None):
        self.fit(model_features, model_targets, model_cluster_features=model_cluster_features, sample_weight=sample_weight, ball_tree_kwargs=ball_tree_kwargs)
        return self.transform(model_cluster_features, weighted=weighted, batch_size=batch_size, parallel=parallel, k=k, r=r, kernel=kernel, beta0=beta0)
    
    def predict(self, model_cluster_features, k=None, r=None, batch_size=100, parallel=False, weighted=True, kernel=None, neighbor_beta0s=False, beta0=None):
        return self.transform(model_cluster_features, k=k, r=r, batch_size=batch_size, parallel=parallel,
            weighted=weighted, kernel=kernel, neighbor_beta0s=neighbor_beta0s, beta0=beta0,
            model_postprocessor=lambda m,q,x,y,w: m.predict(q))
