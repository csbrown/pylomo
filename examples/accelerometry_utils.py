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
from collections import defaultdict
import logging
import ml_battery.log
from itertools import islice
from gpr_utils import *

logger = logging.getLogger(__name__)

mpl.rcParams['figure.figsize'] = [16.0, 8.0]
mpl.rcParams['font.size'] = int(mpl.rcParams['figure.figsize'][1]*3)
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', CB_color_cycle)


SIGNAL_HZ = 52
CHANNEL_COLS=list(range(1,4))
CHANNELS = len(CHANNEL_COLS)
LABEL_COL = 4

data_file_format = "{}.csv"
data_alignments = defaultdict(int,{
    2:120,
    3:40,
    4:85,
    6:110,
    7:30,
    8:150,
    9:70,
    10:140,
    11:70,
    13:50,
    14:20
})

def local_gpr_transform_all_channels(data_file, transformed_data_dir, subsample_rate, gpr_subsample_rate, bandwidth, base_theta):
    data = np.loadtxt(data_file,delimiter=",")
    subsampled_dat = data[::subsample_rate]
    HZ = int(SIGNAL_HZ/subsample_rate)

    lm_kernel = local_models.local_models.TriCubeKernel(bandwidth=bandwidth)
    index_X = np.arange(subsampled_dat.shape[0]*1.).reshape(-1,1)
    index = local_models.local_models.ConstantDistanceSortedIndex(index_X.flatten())

    kernel = np.sum((
        np.prod((
            gp.kernels.ConstantKernel(constant_value=1, constant_value_bounds="fixed"),
            gp.kernels.RBF(length_scale=1, length_scale_bounds="fixed")
        )),
        gp.kernels.WhiteKernel(noise_level=1, noise_level_bounds=[1e-10,1e10])
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
