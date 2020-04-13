import numpy as np
import scipy.fftpack
import patched_gpr
import matplotlib.pyplot as plt
from ml_battery.utils import cmap
import matplotlib as mpl
import cycler
import os
import shutil
import local_models.local_models
import sklearn.gaussian_process as gp
import logging
import ml_battery.log
from itertools import islice
from gpr_utils import *

logger = logging.getLogger(__name__)

signal_subsample_rate = 5
SIGNAL_HZ = 250
HZ=int(SIGNAL_HZ/signal_subsample_rate)
EEG_CHANNELS=21
SECONDS_OF_SIGNAL=100
l = HZ*SECONDS_OF_SIGNAL
gpr_subsample_rate=10

mpl.rcParams['figure.figsize'] = [16.0*SECONDS_OF_SIGNAL/20, 8.0]
mpl.rcParams['font.size'] = int(mpl.rcParams['figure.figsize'][1]*4)
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', CB_color_cycle)

def adjust_ax_labels(axs, rng, n_ticks=None, hz=250):
    try:
        axs_iterator = iter(axs)
    except TypeError as te:
        axs_iterator = iter([axs])
    for ax in axs_iterator:
        ax.axis("off")
    ax.axis("on")
    if nticks is not None:
        ax.locator_params(axis='x', nbins=nticks)
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

def clear(fig, axs):
    for ax in axs:
        ax.clear()
    fig.clear()
    plt.close(fig)
    plt.close("all")

def plt_gpr_params(X, y, gpr_X, gpr_params, epipoint, kernel, filename, hz=HZ):
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    adjust_ax_labels(axs,(np.min(X), np.max(X)),hz=HZ)
    artists = []
    colors = (color for color in plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for var in range(gpr_params.shape[1]):
        c = next(colors)
        ar = axs[0].plot(gpr_X, gpr_params[:,var],c=c)
        artists.append(ar[0])
    if epipoint is not None:
        c = next(colors)
        for ax in axs:
            ar = ax.axvline(epipoint,c=c,lw=5)
        artists.append(ar)
    c = next(colors)
    ar = axs[1].plot(X, y, c=c)
    artists.append(ar[0])
    axs[0].legend(artists, list(map(lambda x: type(x).__name__, [k for k in patched_gpr.decompose_kernels(kernel) if k.n_dims])) + ["ictal point", "filtered eeg"],loc="lower left")
    plt.savefig(filename)
    clear(fig, axs)

 
def hamming2():
    '''\
    This version is based on a snippet from:
        https://web.archive.org/web/20081219014725/http://dobbscodetalk.com:80
                         /index.php?option=com_content&task=view&id=913&Itemid=85
        http://www.drdobbs.com/architecture-and-design/hamming-problem/228700538
        Hamming problem
        Written by Will Ness
        December 07, 2008
 
        When expressed in some imaginary pseudo-C with automatic
        unlimited storage allocation and BIGNUM arithmetics, it can be
        expressed as:
            hamming = h where
              array h;
              n=0; h[0]=1; i=0; j=0; k=0;
              x2=2*h[ i ]; x3=3*h[j]; x5=5*h[k];
              repeat:
                h[++n] = min(x2,x3,x5);
                if (x2==h[n]) { x2=2*h[++i]; }
                if (x3==h[n]) { x3=3*h[++j]; }
                if (x5==h[n]) { x5=5*h[++k]; } 
    '''
    h = 1
    _h=[h]    # memoized
    multipliers  = (2, 3, 5)
    multindeces  = [0 for i in multipliers] # index into _h for multipliers
    multvalues   = [x * _h[i] for x,i in zip(multipliers, multindeces)]
    yield h
    while True:
        h = min(multvalues)
        _h.append(h)
        for (n,(v,x,i)) in enumerate(zip(multvalues, multipliers, multindeces)):
            if v == h:
                i += 1
                multindeces[n] = i
                multvalues[n]  = x * _h[i]
        # cap the memoization
        mini = min(multindeces)
        if mini >= 1000:
            del _h[:mini]
            multindeces = [i - mini for i in multindeces]
        #
        yield h

def previous_best_fft_len(target):
    best_ham = None
    hams = hamming2()
    ham = 0
    while ham <= target:
        best_ham = ham
        ham = next(hams)
    return best_ham


#a computationally convenient approximation to the beta... sig on [0,1]
def kumaraswamy(sig,a=1.,b=1.):
    return a*b*sig**(a-1)*(1-sig**a)**(b-1)
def spectrum(sig, d):
    f = scipy.fftpack.rfftfreq(sig.size, d=d)
    y = scipy.fftpack.rfft(sig)
    return f, y
def kumaraswamy_filter(sig,d,a=1.,b=1.):
    f, y = spectrum(sig, d)
    max_f = np.max(f)
    kumaraswamy_filter = kumaraswamy(f/max_f,a,b)
    kumaraswamy_filter /= np.max(kumaraswamy_filter) #scale units
    y *= kumaraswamy_filter
    filtered_sig = scipy.fftpack.irfft(y)
    return filtered_sig

def get_filtered_data(data_file, data_dir):
    a = 1.2; b=10.
    filtered_data_dir = os.path.join(data_dir, "filtered_data")
    os.makedirs(filtered_data_dir, exist_ok=1)
    filtered_data_filename = os.path.join(filtered_data_dir, "kumaraswamy_filtered_data_eeg{}_a{:05.02f}_b{:05.02f}".format(data_file, a, b))

    if not os.path.isfile(filtered_data_filename):
        dat = np.loadtxt(os.path.join(data_dir, data_file))
        best_fft_len = previous_best_fft_len(dat.shape[0])
        data_offset = dat.shape[0] - best_fft_len
        dat = dat[data_offset:]
        filtered_dat = np.empty(dat[:,:EEG_CHANNELS].shape)
        for channel in range(EEG_CHANNELS):
            print(channel)
            filtered_dat[:,channel] = kumaraswamy_filter(dat[:,channel],1/SIGNAL_HZ,a,b)
        np.savetxt(filtered_data_filename, filtered_dat)
    else:
        dat = np.loadtxt(os.path.join(data_dir, data_file))
        filtered_dat = np.loadtxt(filtered_data_filename)
        data_offset = dat.shape[0] - filtered_dat.shape[0]

    return filtered_dat, data_offset

class GPR(patched_gpr.GaussianProcessRegressor):
    def fit(self, X,y,sample_weight=None, **kwargs):
        if "beta0" in kwargs:
            self.kernel.theta = kwargs["beta0"]
            del kwargs["beta0"]
        the_model = super().fit(X,y,sample_weight, **kwargs)
        self.coef_ = the_model.kernel_.theta
        self.intercept_ = np.empty((0,))
        return the_model

def get_base_waveform_theta(hz, bandwidth):
    n = 2*bandwidth-1
    X = np.arange(n)
    MEAN_DELTAWAVE_PERIOD = 2
    sample_deltawaves = 250*np.sin(2*np.pi*MEAN_DELTAWAVE_PERIOD/hz*X)

    kernel = np.sum((
        np.prod((#delta waves
            gp.kernels.ConstantKernel(constant_value=1e6, constant_value_bounds=[1e-10,1e10]),
            gp.kernels.RBF(length_scale=hz/10, length_scale_bounds=[1e-10,1e10]))),
        gp.kernels.WhiteKernel(noise_level=1e-9, noise_level_bounds=[1e-9,1e-9])
    ))

    regressor = GPR(kernel=kernel, normalize_y=True, n_restarts_optimizer=400, alpha=0)

    lm_kernel = local_models.local_models.TriCubeKernel(bandwidth=bandwidth)

    delta_wave_regressor = GPR(kernel=simple_kernel, normalize_y=True, n_restarts_optimizer=400,alpha=0)
    delta_wave_regressor.fit(X.reshape(-1,1), sample_deltawaves, sample_weight=lm_kernel(X-n/2))

    deltawave_c, deltawave_lengthscale = np.exp(simple_regressor.kernel_.theta[:2])
    return deltawave_c, deltawave_lengthscale

def get_exemplar_gpr_theta(exemplar_X, exemplar_y, hz, bandwidth, base_waveform_theta):

    lm_kernel = local_models.local_models.TriCubeKernel(bandwidth=bandwidth)
    kernel = np.sum((
        np.prod((#delta waves
            gp.kernels.ConstantKernel(constant_value=1, constant_value_bounds=[1e-10,1e10]),
            gp.kernels.RBF(length_scale=1, length_scale_bounds="fixed")
        )),
        gp.kernels.WhiteKernel(noise_level=1, noise_level_bounds=[1,1])
    ))
    kernel.theta[:2] = np.array(base_waveform_theta)

    regressor = GPR(kernel=kernel, normalize_y=True, n_restarts_optimizer=400, alpha=0)

    exemplar_gpr = regressor.fit(
        exemplar_X, exemplar_y,
        lm_kernel(np.abs(exemplar_X-np.mean(exemplar_X)))[:,0]) 

    return exemplar_gpr.kernel_.theta

def local_gpr_transform_all_channels(data_file, data_dir, transformed_data_dir, data_epipoint, subsample_rate, gpr_subsample_rate, bandwidth, base_waveform_theta):
    data, data_offset = get_filtered_data(data_file, data_dir)
    data_epipoint = data_epipoint - int(data_offset/subsample_rate)
    subsampled_dat = data[::subsample_rate]
    HZ = int(SIGNAL_HZ/subsample_rate)
    l = HZ*SECONDS_OF_SIGNAL
    n = 2*bandwidth-1

    ictal_rng = (max(0,data_epipoint-l), min(subsampled_dat.shape[0], data_epipoint+l))
    negative_ictal_rng = (max(0, int(data_epipoint/2)-l), min(subsampled_dat.shape[0], int(data_epipoint/2)+l))
    subsample_ictal_rng = (np.array(ictal_rng)/gpr_subsample_rate).astype(int)
    subsample_negative_ictal_rng = (np.array(negative_ictal_rng)/gpr_subsample_rate).astype(int)
    lm_kernel = local_models.local_models.TriCubeKernel(bandwidth=bandwidth)
    index_X = np.arange(subsampled_dat.shape[0]*1.).reshape(-1,1)
    index = local_models.local_models.ConstantDistanceSortedIndex(index_X.flatten())
    exemplar_rng = (HZ*4,HZ*4+n) 
    exemplar_X = index_X[slice(*exemplar_rng)]
    exemplar_y = subsampled_dat[slice(*exemplar_rng)]
    ictal_X = index_X[slice(*ictal_rng)]
    ictal_X_gpr_subsampled = index_X[ictal_rng[0] : ictal_rng[1] : gpr_subsample_rate]
    exemplar_X_gpr_subsampled = index_X[exemplar_rng[0] : exemplar_rng[1] : gpr_subsample_rate]
    negative_ictal_X = index_X[slice(*negative_ictal_rng)]
    negative_ictal_X_gpr_subsampled = index_X[negative_ictal_rng[0] : negative_ictal_rng[1] : gpr_subsample_rate]

    kernel = np.sum((
        np.prod((#delta waves
            gp.kernels.ConstantKernel(constant_value=1, constant_value_bounds=[1e-10,1e10]),
            gp.kernels.RBF(length_scale=1, length_scale_bounds="fixed")
        )),
        gp.kernels.WhiteKernel(noise_level=1, noise_level_bounds=[1,1])
    ))

    timelog = local_models.local_models.loggin.TimeLogger(
        logger=logger, 
        how_often=1, total=EEG_CHANNELS, 
        tag="transforming data for {}".format(data_file))
    
    for channel in range(EEG_CHANNELS):
        with timelog:
            exemplar_theta = get_exemplar_gpr_theta(exemplar_X, exemplar_y[:,channel], HZ, bandwidth, base_waveform_theta)
            kernel.theta = exemplar_theta

            local_regressor = GPR(kernel=kernel, normalize_y=True, n_restarts_optimizer=0, alpha=0)
            y = subsampled_dat[:,channel]
            gpr_models = local_models.local_models.LocalModels(local_regressor)
            gpr_models.fit(index_X,y,index=index)

            ictal_gpr_params = gpr_models.transform(
                ictal_X_gpr_subsampled, 
                r=lm_kernel.support_radius()-1, weighted=True, kernel=lm_kernel, neighbor_beta0s=False, 
                batch_size=int(negative_ictal_X_gpr_subsampled.shape[0]/10))

            negative_ictal_gpr_params = gpr_models.transform(
                negative_ictal_X_gpr_subsampled, 
                r=lm_kernel.support_radius()-1, weighted=True, kernel=lm_kernel, neighbor_beta0s=False, 
                batch_size=int(negative_ictal_X_gpr_subsampled.shape[0]/10))

            transformed_ictal_data_filename = os.path.join(transformed_data_dir, "ictal_transformed_data_k{}_rng{}_channel{:03d}.dat".format(str(lm_kernel), str(ictal_rng), channel))
            transformed_negative_ictal_data_filename = os.path.join(transformed_data_dir, "negative_ictal_transformed_data_k{}_rng{}_channel{:03d}.dat".format(str(lm_kernel), str(ictal_rng), channel))
            
            np.savetxt(transformed_ictal_data_filename, ictal_gpr_params)
            np.savetxt(transformed_negative_ictal_data_filename, negative_ictal_gpr_params)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("f", type=str)
    parser.add_argument("epipoint", type=int)
    args = parser.parse_args()
    data_file = args.f
    subsample_rate = 5
    data_epipoint = int(args.epipoint/subsample_rate)
    gpr_subsample_rate = 10
    HZ = int(SIGNAL_HZ/subsample_rate)
    bandwidth = 2*HZ
    base_waveform_theta = (4053.795201584327, 4.324318762299779)
    data_dir = "/home/brown/disk2/eeg/Phasespace/Phasespace/data/eeg-text"
    transformed_data_dir = os.path.join("/home/brown/disk2/eeg/transformed_data", data_file)
    os.makedirs(transformed_data_dir, exist_ok=1) 

    try:
        local_gpr_transform_all_channels(data_file, data_dir, transformed_data_dir, data_epipoint, subsample_rate, gpr_subsample_rate, bandwidth, base_waveform_theta)
    except Exception as e:
        shutil.rmtree(transformed_data_dir)
        raise e
