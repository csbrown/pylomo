import numpy as np
import patched_gpr
import matplotlib.pyplot as plt
from ml_battery.utils import cmap
import matplotlib as mpl
import cycler
import os
import shutil
import local_models.local_models
import sklearn.gaussian_process as gp
import sklearn
import logging
import ml_battery.log
from itertools import islice

logger = logging.getLogger(__name__)
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', CB_color_cycle)

def adjust_ax_labels(axs, n_ticks=None):
    try:
        axs_iterator = iter(axs)
    except TypeError as te:
        axs_iterator = iter([axs])
    for ax in axs_iterator:
        ax.axis("off")
    ax.axis("on")
    if n_ticks is not None:
        ax.locator_params(axis='x', nbins=n_ticks)
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

def clear(fig, axs):
    try:
        for ax in axs:
            ax.clear()
    except:
        axs.clear()
    fig.clear()
    plt.close(fig)
    plt.close("all")
    
def plt_gpr_params(X, ys, gpr_X = np.empty((0)), gpr_paramses = np.empty((0,0,0)), chg_ptses = [], kernel=None, filename = None, display=False, legend=True, ys_legend=["raw data"], axes_off=True):
    if len(gpr_paramses.shape) == 2:
        gpr_paramses = gpr_paramses[...,None]
    if len(ys.shape) == 1:
        ys = ys[:,None]
    fig, axs = plt.subplots(gpr_paramses.shape[2]+1, sharex=True, gridspec_kw={'hspace': 0})
    if gpr_paramses.shape[2]+1 == 1:
        axs = [axs]
    if axes_off:
        adjust_ax_labels(axs)
    artists = []
    extra_dims = []
    colors = (color for color in plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for var in range(gpr_paramses.shape[1]):
        c = next(colors)
        for param_set in range(gpr_paramses.shape[2]):
            gpr_param = gpr_paramses[:,var,param_set]
            ar = axs[param_set].plot(gpr_X, gpr_param,c=c)
        artists.append(ar[0])
    for var in range(ys.shape[1]):
        c = next(colors)
        ar = axs[-1].plot(X, ys[:,var], c=c)
        artists.append(ar[0])
    extra_dims.extend(ys_legend)
    for i, chg_pts in enumerate(chg_ptses):
        if i==0:
            extra_dims.append("true change points")
        elif i==1:
            extra_dims.append("pred change points")
        c = next(colors)
        for chg_pt in chg_pts:
            for ax in axs:
                ar = ax.axvline(chg_pt,c=c,lw=2,linestyle=(i*5,(5,5)))
        artists.append(ar)
    if legend:
        kernel_dims = []
        if kernel is not None:
            kernel_dims = list(map(lambda x: type(x).__name__, [k for k in patched_gpr.decompose_kernels(kernel) if k.n_dims]))
        axs[0].legend(artists, kernel_dims + extra_dims, loc="lower left")
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    if not display:
        clear(fig, axs)

class GPR(patched_gpr.GaussianProcessRegressor):
    def fit(self, X,y,sample_weight=None, **kwargs):
        if "beta0" in kwargs:
            self.kernel.theta = kwargs["beta0"]
            del kwargs["beta0"]
        sample_weight = sample_weight/np.sum(sample_weight)*sample_weight.shape[0] #sums to n
        the_model = super().fit(X,y,sample_weight, **kwargs)
        self.coef_ = the_model.kernel_.theta
        self.intercept_ = np.empty((0,))
        return the_model

class GPRNeighborFixedMixt(patched_gpr.GaussianProcessRegressor):
    def fit(self, X,y,sample_weight=None, **kwargs):
        beta0 = None
        if "beta0" in kwargs:
            beta0 = kwargs["beta0"]
            del kwargs["beta0"]
        sample_weight = sample_weight/np.sum(sample_weight)*sample_weight.shape[0] #sums to n
        the_model = super().fit(X,y,sample_weight, **kwargs)
        if beta0 is not None:
            beta0_model = sklearn.base.clone(self)
            beta0_model.kernel = self.kernel.clone_with_theta(beta0)
            beta0_model.fit(X,y,sample_weight,**kwargs)
            if beta0_model.log_marginal_likelihood_value_ < the_model.log_marginal_likelihood_value_:
                old_restart = the_model.n_restarts_optimizer
                the_model.n_restarts_optimizer = 400
                the_model = the_model.fit(X,y,sample_weight, **kwargs)
                the_model.n_restarts_optimizer = old_restart

        self.coef_ = the_model.kernel_.theta
        self.intercept_ = np.empty((0,))
        return the_model

def soft_bound(kernel,val,priors=None):
    if priors is None:
        priors = []
        for i,bound in enumerate(kernel.bounds):
            log_bounds = np.array(bound)
            priors.append(local_models.local_models.TriCubeKernel(np.abs(np.diff(log_bounds))))
    probs = np.empty(val.shape)
    d_probs = np.empty(val.shape)
    for i,bound in enumerate(kernel.bounds):
        log_bounds = np.array(bound)
        u = np.mean(log_bounds) - np.log(val)
        probs[i] = priors[i](u)
        d_probs[i] = priors[i].d(u)
    dprob_indices = np.arange(len(d_probs))
    for i in dprob_indices:
        d_probs[i] *= np.product(probs[dprob_indices != i])
    return np.product(probs), d_probs

class GPRSoftBounds(GPR):
    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        lml = super().log_marginal_likelihood(theta, eval_gradient)
        if theta is None:
            return lml
        if eval_gradient:
            lml, grad = lml
        prob, d_probs = soft_bound(self.kernel_, theta)
        print(prob)
        lml *= prob
        if eval_gradient:
            return lml, lml*d_probs + prob*grad
        else:
            return lml
