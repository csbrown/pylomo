import scipy.optimize
import sklearn.utils
import sklearn.base
import numpy as np
import scipy.odr
from . import utils

def quad(x2,xy,y2,x,y, A, B, C, D, E, F):
    return A*x2 + B*xy + C*y2 + D*x + E*y + F
def quad_optimizer(theta, x2, xy, y2, x, y, sample_weight=None):
    A, B, C, D, E, F, lamb, mu = theta
    quadded = quad(x2,xy,y2,x,y,A,B,C,D,E,F)
    dldA = 2*np.sum(quadded*x2) - 4*lamb*C + 2*mu*A
    dldB = 2*np.sum(quadded*xy) + 2*lamb*B + 2*mu*B
    dldC = 2*np.sum(quadded*y2) - 4*lamb*A + 2*mu*C
    dldD = 2*np.sum(quadded*x) + 2*mu*D
    dldE = 2*np.sum(quadded*y) + 2*mu*E
    dldF = 2*np.sum(quadded) + 2*mu*F
    dldlamb = B**2 - 4*A*C
    dldmu = A**2+B**2+C**2+D**2+E**2+F**2 - 1
    return np.sum([dldA,dldB,dldC,dldD,dldE,dldF,dldlamb,dldmu])

class QuadraticRegression(sklearn.base.RegressorMixin):
    def __init__(self):
        self.intercept_ = np.array(())
    def fit(self, X, y=None, sample_weight=None):
        X = sklearn.utils.check_array(X)
        x2 = X[:,0]**2
        xy = X[:,0]*X[:,1]
        y2 = X[:,1]**2
        x = X[:,0]
        y = X[:,1]
        blah = np.array([1]*6)
        blah = blah/np.sum(blah**2)**0.5
        self.solution = scipy.optimize.minimize(quad_optimizer, list(blah) + [1,1], args=(x2, xy, y2, x, y, sample_weight))
        self.coef_ = self.solution.x[:-2]
        return self
    def predict(self, X):
        pass
    def get_params(self, *args, **kwargs):
        return dict()
    def set_params(self, params, *args, **kwargs):
        pass

def circle_optimizer(theta, x, y, sample_weight=None):
    x0, y0, r = theta
    loss = ((x-x0)**2 + (y-y0)**2 - r**2)**2
    if sample_weight is not None:
        loss *= sample_weight
    return np.sum(loss)
    
class CircleRegression(sklearn.base.RegressorMixin):
    def __init__(self):
        self.intercept_ = np.array(())
    def fit(self, X, y=None, sample_weight=None):
        X = sklearn.utils.check_array(X)
        x, y = X[:,0], X[:,1]
        theta0 = np.array([0]*2 + [1])
        self.solution = scipy.optimize.minimize(circle_optimizer, list(theta0), args=(x, y, sample_weight))
        self.coef_ = self.solution.x
        return self
    def predict(self, X):
        return np.sqrt(self.coef_[2]**2 - (X-self.coef_[0])**2) + self.coef_[1]
    def get_params(self, *args, **kwargs):
        return dict()
    def set_params(self, params, *args, **kwargs):
        pass
    
def spherical_optimizer(theta, x, sample_weight=None):
    loss = ((x - theta[:-1])**2 - theta[-1]**2)**2
    if sample_weight is not None:
        loss *= sample_weight
    return np.sum(loss)
class SphericalRegression(sklearn.base.RegressorMixin):
    def __init__(self):
        self.intercept_ = np.array(())
    def fit(self, X, y=None, sample_weight=None):
        X = sklearn.utils.check_array(X)
        theta0 = np.array([0]*X.shape[1] + [1])
        self.solution = scipy.optimize.minimize(spherical_optimizer, list(theta0), args=(x,sample_weight))
        self.coef_ = self.solution.x
        return self
    def predict(self, X):
        pass
    def get_params(self, *args, **kwargs):
        return dict()
    def set_params(self, params, *args, **kwargs):
        pass
    
def linear_optimizer(theta, x, sample_weight=None): 
    loss = (x.dot(theta) - 1)**2
    if sample_weight is not None:
        loss *= sample_weight
    return np.sum(loss)
class LinearRegression(sklearn.base.RegressorMixin):
    def __init__(self):
        self.intercept_ = np.array(())
    def fit(self, X, y=None, sample_weight=None, theta0 = None):
        X = sklearn.utils.check_array(X)
        if theta0 is None:
            theta0 = np.array([1]*X.shape[1])
        self.solution = scipy.optimize.minimize(linear_optimizer, list(theta0), args=(X, sample_weight))
        self.coef_ = self.solution.x
        return self
    def predict(self, X):
        pass
    def get_params(self, *args, **kwargs):
        return dict()
    def set_params(self, params, *args, **kwargs):
        pass
    def project(self, x):
        n_dot_n = self.coef_.dot(self.coef_)
        x_dot_n = self.coef_.dot(x)
        return ((1 - x_dot_n)/n_dot_n) * self.coef_

def lin(B,x):
    blah = np.einsum("i,ij->j", B, x)-1
    return blah.reshape((1,-1))
def lin_jacB(B,x):
    return x
def lin_jacx(B,x):
    return np.tile(B.reshape((-1,1)), (1,x.shape[1]))
class LinearODR(sklearn.base.RegressorMixin):
    def __init__(self):
        self.intercept_=np.array(())
    def fit(self, X, y=None, sample_weight=None, beta0=None):
        if X.shape[0] == 0:
            raise Exception("LinearODR breaks when fitting on empty dataset")
        if X.shape[0] < X.shape[1]:
            self.coef_ = np.empty((X.shape[1],))
            self.coef_[:] = np.nan
            return self
        if beta0 is None: beta0 = np.array((X.shape[1]) * [1])
        X = X.T
        data = scipy.odr.Data(X, 1, we=sample_weight)
        model = scipy.odr.Model(lin, fjacb=lin_jacB, fjacd=lin_jacx, implicit=True)
        #odr = scipy.odr.ODR(data, model, beta0=beta0, errfile="odrerr", rptfile="odrrpt", iprint=6666)
        odr = scipy.odr.ODR(data, model, beta0=beta0)
        odr.set_iprint(init=2, iter=2, final=2)
        out = odr.run()
        self.coef_ = out.beta
        self.cov_beta = out.cov_beta
        return self
    def predict(self, X):
        return (1-lin(self.coef_[:-1], X))/self.coef_[-1]
    def get_params(self, *args, **kwargs):
        return dict()
    def set_params(self, params, *args, **kwargs):
        pass
    def project(self, x):
        n_dot_n = self.coef_.dot(self.coef_)
        x_dot_n = self.coef_.dot(x)
        return ((1 - x_dot_n)/n_dot_n) * self.coef_

class Mean(sklearn.base.RegressorMixin):
    def __init__(self):
        self.coef_=np.array(())
    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is not None:
            self.intercept_ = np.average(X, weights=sample_weight, axis=0)
        else:
            self.intercept_ = np.average(X, axis=0)
        return self
    def predict(self, X):
        return np.repeat(self.intercept_.reshape((1,-1)), X.shape[0], axis=0)
    def get_params(self, *args, **kwargs):
        return dict()
    def set_params(self, params, *args, **kwargs):
        pass
    def project(self, x):
        return self.coef_ - x


class LinearODR_mD(sklearn.base.RegressorMixin):
    def __init__(self, m=None, invert=False):
        self.m = m
        self.invert = invert #take m smallest eigenvectors instead of largest
    def fit(self, X, y=None, sample_weight=None):
        if X.shape[0] == 0:
            raise Exception("LinearODR breaks when fitting on empty dataset")
        if (X.shape[0] < X.shape[1]):
            self.coef_ = np.empty((X.shape[1]*self.m,))
            self.coef_[:] = np.nan
            self.intercept_ = np.average(X, weights=sample_weight, axis=0)
            return self
        self.n = X.shape[1]
        #find a pca
        try: 
            self.intercept_ = np.average(X, weights=sample_weight, axis=0)
        except ZeroDivisionError:
            self.coef_ = np.empty((X.shape[1]*self.m,))
            self.coef_[:] = np.nan
            self.intercept_ = np.empty((X.shape[1],))
            self.intercept_[:] = np.nan
            return self

        if self.m == 0:
            self.coef_ = np.zeros([0])
            return self 
        covmat = np.cov(X, aweights=sample_weight, rowvar=False)
        if np.any(np.isnan(covmat)) or np.any(np.isinf(covmat)):
            self.coef_ = np.empty((X.shape[1]*self.m,))
            self.coef_[:] = np.nan
            self.intercept_ = np.empty((X.shape[1],))
            self.intercept_[:] = np.nan
            return self
            
        self.cov_eigenvalues, self.cov_eigenvectors = np.linalg.eig(covmat)
        self.cov_eigenvectors = self.cov_eigenvectors.T
        self.cov_eigenvalues_sorter = np.argsort(self.cov_eigenvalues)[::-1] #highest to lowest
        if not self.invert:
            self.coef_ = np.concatenate(self.cov_eigenvectors[self.cov_eigenvalues_sorter[:self.m]], axis=0)
        else:
            self.coef_ = np.concatenate(self.cov_eigenvectors[self.cov_eigenvalues_sorter[:-(self.m+1):-1]], axis=0)
        return self
    def get_params(self, *args, **kwargs):
        return {"m":self.m, "invert":self.invert}
    def set_params(self, params, *args, **kwargs):
        self.m = params["m"]
        self.invert = params["invert"]
    def project(self, X):
        return utils.sublinear_project_vectorized(X, self.coef_.reshape(X.shape[1],self.m).T, self.intercept_)
    def predict(self, X):
        return self.project(X)
    def project1d(self,x):
        ''' project x into each eigenvector individually.  This returns a x.shape[0] by x.shape[1] by x.shape[1] tensor '''
        #shift, project, unshift
        x = x-self.intercept_
        projections = np.einsum('ij,kj->kij',self.cov_eigenvectors[self.cov_eigenvalues_sorter],x)
        return projections + self.intercept_

        
