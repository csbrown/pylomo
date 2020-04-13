import numpy as np
import scipy.stats
import logging
import functools
import collections
import itertools
import sys
from . import utils
from . import local_models
from . import loggin
from . import linear_projections

logger = logging.getLogger(__name__)

def mean_shift(mean_models, data, iterations=100, kernel=None):
    timelog = loggin.TimeLogger(logger=logger, how_often=1, total=iterations, tag="mean_shift")

    yield data
    for i in range(iterations):
        with timelog:
            data = mean_models.transform(data, r=kernel.support_radius(), weighted=True, kernel=kernel)
            #err_pts = np.any(np.isnan(mean_grid), axis=1)
            #good_pts = np.logical_not(err_pts)
            #data = mean_grid[good_pts]
            yield data

            
def local_tls_shift(linear_models, data, iterations=100, kernel=None, safe=False):
    timelog = loggin.TimeLogger(logger=logger, how_often=1, total=iterations, tag="local_tls_shift")

    yield data
    for i in range(iterations):
        with timelog:
            if hasattr(kernel.bandwidth, "__call__"):
                linear_params_vecs, linear_params_mean = linear_projections.transformate_data(data, kernel, linear_models, k=kernel.k)
            else:
                linear_params_vecs, linear_params_mean = linear_projections.transformate_data(data, kernel, linear_models, r=kernel.support_radius())
            if safe:
                err_pts = np.any(np.isnan(linear_params_vecs), axis=(1,2))
                logger.info("linear odr undefined at {} pts".format(err_pts.sum()))
                good_pts = np.logical_not(err_pts)
                data[good_pts] = utils.linear_project_pointwise_bases(data[good_pts], linear_params_vecs[good_pts], linear_params_mean[good_pts])
            else:
                data = utils.linear_project_pointwise_bases(data, linear_params_vecs, linear_params_mean)
            yield data


def mean_line_projection_shift(mean_models, data, iterations=100, kernel=None, line_projector=None):
    timelog = loggin.TimeLogger(logger=logger, how_often=1, total=iterations, tag="mean_line_projection_shift")

    yield data
    for i in range(iterations):
        with timelog:
            mean_grid = mean_models.transform(data, r=kernel.support_radius(), weighted=True, kernel=kernel)
            data, mean_grid = data[good_pts], mean_grid[good_pts]
            if line_projector is not None:
                data = line_projector(data, mean_grid)
            else:
                data = mean_grid
            yield data

class ConvergenceFailureMonitor(object):
    BIG_NUMBER = sys.float_info.max/10
    def __init__(self, number_of_iterations_to_give_up_after_if_nothing_is_happening):
        self.fail_iterations = number_of_iterations_to_give_up_after_if_nothing_is_happening
        self.avgs = collections.deque([ConvergenceFailureMonitor.BIG_NUMBER/10]*int(self.fail_iterations/2) + [ConvergenceFailureMonitor.BIG_NUMBER]*int(self.fail_iterations/2), self.fail_iterations)
        self.update_new_avg()
        self.update_old_avg()
    def update_new_avg(self):
        self.new_avg = sum(map(lambda x: 2*x/self.fail_iterations, itertools.islice(self.avgs, 0, int(self.fail_iterations/2))))
    def update_old_avg(self):
        self.old_avg = sum(map(lambda x: 2*x/self.fail_iterations, itertools.islice(self.avgs, int(self.fail_iterations/2), self.fail_iterations)))
    def update(self, x):
        self.avgs.rotate()
        self.avgs[0] = x
        self.update_new_avg()
        self.update_old_avg()
    def __call__(self):
        return self.old_avg <= self.new_avg

def local_tls_shift_till_convergence(linear_models, data, tol=1e-8, 
                                     kernel=None, report=False, safe=False, number_of_iterations_to_give_up_after_if_nothing_is_happening=100):
    timelog = loggin.TimeLogger(logger=logger, how_often=1, tag="local_tls_shift_converger")
    
    number_of_iterations_to_give_up_after_if_nothing_is_happening = 100 #MAKE THIS EVEN
    avg_dist_fail_monitor = ConvergenceFailureMonitor(number_of_iterations_to_give_up_after_if_nothing_is_happening)
    unconverged_len_fail_monitor = ConvergenceFailureMonitor(number_of_iterations_to_give_up_after_if_nothing_is_happening)
    absolute_unconverged_len_fail_monitor = ConvergenceFailureMonitor(number_of_iterations_to_give_up_after_if_nothing_is_happening*1000)
    data = np.copy(data)
    unconverged = np.ones(data.shape[0], dtype=bool)
    i = 0
    while unconverged.any() and not (avg_dist_fail_monitor() and unconverged_len_fail_monitor()) and not absolute_unconverged_len_fail_monitor():
        with timelog:
            i += 1
            tls_iterations = local_tls_shift(linear_models, data[unconverged], iterations=1, kernel=kernel)
            old_data, data[unconverged] = next(tls_iterations), next(tls_iterations)
            distances_traveled = np.linalg.norm(data[unconverged] - old_data, axis=1)
            unconverged[unconverged] = distances_traveled > tol
            avg_dist_traveled = np.average(distances_traveled[~np.isnan(distances_traveled)]/tol)
            avg_dist_fail_monitor.update(avg_dist_traveled)
            total_unconverged = unconverged.sum()
            unconverged_len_fail_monitor.update(total_unconverged)
            absolute_unconverged_len_fail_monitor.update(total_unconverged)
            if not((i+1)%1000):
                logger.info(str(unconverged_len_fail_monitor.avgs) + str(avg_dist_fail_monitor.avgs) + str(unconverged_len_fail_monitor.old_avg))
            if not((i+1)%100):
                if np.isnan(avg_dist_traveled): 
                    logger.info("local_tls_shift_converger:: log10i:{:04.02f} unconverged:{:08d} avg_travel(tols):{}".format(np.log10(i), total_unconverged, "nan"))
                else:
                    logger.info("local_tls_shift_converger:: log10i:{:04.02f} unconverged:{:08d} avg_travel(tols):{:08.02f}".format(np.log10(i), total_unconverged, avg_dist_traveled))
            if report:
                yield data, scipy.stats.describe(distances_traveled)
            else:
                yield data
    if (avg_dist_fail_monitor() and unconverged_len_fail_monitor()):
        if np.isnan(avg_dist_traveled): 
            logger.info("convergence_failed!::  unconverged:{:08d} avg_travel(tols):{}".format(total_unconverged, "nan"))
        else:
            logger.info("convergence_failed!::  unconverged:{:08d} avg_travel(tols):{:08d}".format(total_unconverged, int(avg_dist_traveled)))
            logger.info("{}".format(str(avg_dist_fail_monitor.avgs)))
    if absolute_unconverged_len_fail_monitor():
        logger.info("convergence_failed!::  unconverged:{:08d} avg_travel(tols):{:08d}".format(total_unconverged, int(avg_dist_traveled)))
        logger.info("{}".format(str(avg_dist_fail_monitor.avgs)))

def local_mean_shift_till_convergence(mean_models, data, tol=1e-8,
                                     kernel=None,
                                     report=False):
    data = np.copy(data)
    unconverged = np.ones(data.shape[0], dtype=bool)
    while unconverged.any():
        tls_iterations = mean_shift(mean_models, data[unconverged], iterations=1, kernel=kernel)
        old_data, data[unconverged] = next(tls_iterations), next(tls_iterations)
        distances_traveled = np.linalg.norm(data[unconverged] - old_data, axis=1)
        unconverged[unconverged] = distances_traveled > tol
        if report:
            yield data, scipy.stats.describe(distances_traveled)
        else:
            yield data
