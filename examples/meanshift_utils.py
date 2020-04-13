def fit_meanshift(gpr_paramses, kernel):
    mean_regressor = local_models.TLS_models.LinearODR_mD(0)
    mean_models = local_models.local_models.LocalModels(mean_regressor)
    mean_models.fit(gpr_paramses)
    for i, (dat, report) in enumerate(local_models.algorithms.local_tls_shift_till_convergence(
                mean_models, gpr_paramses, kernel=kernel, report=True)):
        if not i%100:
            #print(report)
            pass
    return dat
    
def cleanup_meanshift(meanshifted_data, tol=1e-7): 
    import sklearn.neighbors
    bt = sklearn.neighbors.BallTree(meanshifted_data)
    bins = bt.query_radius(meanshifted_data, r=tol)
    cleanuped = np.empty(meanshifted_data.shape)
    for i, the_bin in enumerate(bins):
        cleanuped[i] = np.mean(meanshifted_data[the_bin], axis=0)
    return cleanuped
    
def pick_optimal_meanshift_bandwidth(initial_bandwidths, gpr_paramses, iterations=7):
    #go middle of the road to get two clusters.
    #i.e. find the minimum bandwidth giving 3, the max giving 1, then take their average
    print(iterations)
    checked_bandwidths = []
    given_n_clusters = []
    if iterations == 0:
        return checked_bandwidths, given_n_clusters
    for bandwidth in initial_bandwidths:
        ms_kernel = local_models.local_models.TriCubeKernel(bandwidth=bandwidth)
        fitted_meanshift = fit_meanshift(gpr_paramses, ms_kernel)
        fitted_meanshift = cleanup_meanshift(fitted_meanshift)
        observed_clusters = np.unique(fitted_meanshift, axis=0).shape[0]
        checked_bandwidths.append(bandwidth)
        given_n_clusters.append(observed_clusters)
    #get half of the ones, and half of the twos:
    the_ones = [checked_bandwidths[i] for i in range(len(checked_bandwidths)) if given_n_clusters[i]==1]
    the_twos = [checked_bandwidths[i] for i in range(len(checked_bandwidths)) if given_n_clusters[i]==2]
    the_more_than_twos = [checked_bandwidths[i] for i in range(len(checked_bandwidths)) if given_n_clusters[i]>2]
    median_the_ones = np.median(the_ones) if the_ones else max(initial_bandwidths)*2
    median_the_more_than_twos = np.median(the_more_than_twos) if the_more_than_twos else min(initial_bandwidths)/2
    median_the_twos = np.median(the_twos) if the_twos else np.mean([median_the_ones, median_the_more_than_twos])
    sub_checked, sub_n_clusters = pick_optimal_meanshift_bandwidth(
        np.concatenate((
            np.linspace(median_the_more_than_twos, median_the_twos, len(initial_bandwidths)),
            np.linspace(median_the_twos, median_the_ones, len(initial_bandwidths))
        )),
        gpr_paramses,
        iterations - 1
    )
    return checked_bandwidths + sub_checked, given_n_clusters + sub_n_clusters 

def mean_d_half_neighbors(dat):
    import sklearn.neighbors
    bt = sklearn.neighbors.BallTree(dat)
    dz, iz = bt.query(dat, k=int(dat.shape[0]/2), return_distance=True, sort_results=True)
    return np.mean([dz[i][-1] for i in range(len(dz))])

def gprify(X, y, gpr_kernel, bandwidth):
    lm_kernel = local_models.local_models.TriCubeKernel(bandwidth=bandwidth)
    exemplar_regressor = GPR(kernel=gpr_kernel, normalize_y=True, n_restarts_optimizer=7, alpha=0)
    exemplar_rng = (int(bandwidth), 3*int(bandwidth)-2)
    exemplar_X = X[slice(*exemplar_rng)]
    exemplar_y = y[slice(*exemplar_rng)]
    exemplar_regressor.fit(
        exemplar_X, 
        exemplar_y, 
        sample_weight = lm_kernel(np.abs(exemplar_X - np.mean(exemplar_X)))[:,0])
        
    regressor = GPR(kernel=exemplar_regressor.kernel_, normalize_y=True, n_restarts_optimizer=0, alpha=0)
    gpr_models = local_models.local_models.LocalModels(regressor)
    gpr_models.fit(X,y)
    gpr_params = gpr_models.transform(X, r=lm_kernel.support_radius()-1, weighted=True, kernel=lm_kernel, neighbor_beta0s=False, batch_size=X.shape[0])
    return gpr_params

def clusterify(X, iterations=5):
    halfspace_bandwidth = mean_d_half_neighbors(X)
    bs, ns = pick_optimal_meanshift_bandwidth(
        [halfspace_bandwidth/2, halfspace_bandwidth/4, halfspace_bandwidth/8], 
        X, iterations=iterations)
    if 2 in ns:
        optimal_meanshift_bandwidth = np.median([bs[i] for i in range(len(bs)) if ns[i] ==2])
    else:
        optimal_meanshift_bandwidth = np.median(bs)
    ms_kernel = local_models.local_models.TriCubeKernel(bandwidth=optimal_meanshift_bandwidth)
    meanshifted = fit_meanshift(X, ms_kernel)
    meanshifted = cleanup_meanshift(meanshifted)
    return optimal_meanshift_bandwidth, meanshifted

def get_clusterified_change_points(X, clusterified_data):
    return X[
        np.nonzero(
            np.diff(
                np.linalg.norm(clusterified_data, axis=1), 
                axis=0))]
    
def compare_change_points(y_pred, y_true):
    import sklearn.neighbors
    bt = sklearn.neighbors.BallTree(y_true)
    dz,iz = bt.query(y_pred, k=1, return_distance=True)
    return np.sum(dz)
    
def pick_optimal_gpr_bandwidth_for_changepoint_detection(
    X, y, gpr_kernel, bounding_bandwidths, observed_changepoints, 
    given_evaluations_at_bounding_bandwidths = [None,None], 
    given_optimal_meanshift_bandwidths = [None, None],
    given_optimal_meanshifts = [None, None],
    given_optimal_gpr_paramses = [None, None],
    iterations=7, meanshift_iterations=5):
    #get gpr params at each of the bounding bandwidths
    #pick_optimal_meanshift_bandwidth
    #compare each change point with nearest observed, this is the loss.
    #rinse and repeat at the midpoint and the lowest observed loss endpoint... (binary search)
    print(iterations)
    if iterations == 0:
        return [(
            bounding_bandwidths[i], 
            given_optimal_meanshift_bandwidths[i],
            given_optimal_meanshifts[i],
            given_optimal_gpr_paramses[i])
            for i in range(2) if given_evaluations_at_bounding_bandwidths[i] is not None
        ][0]
        
    evaluations_at_bounding_bandwidths = [None, None]
    optimal_meanshift_bandwidths = [None, None]
    optimal_meanshifts = [None, None]
    optimal_gpr_paramses = [None, None]
    for i, bandwidth in enumerate(bounding_bandwidths):
        if given_evaluations_at_bounding_bandwidths[i] is not None:
            evaluations_at_bounding_bandwidths[i] = given_evaluations_at_bounding_bandwidths[i]
            optimal_meanshift_bandwidths[i] = given_optimal_meanshift_bandwidths[i]
            optimal_meanshifts[i] = given_optimal_meanshifts[i]
            optimal_gpr_paramses[i] = given_optimal_gpr_paramses[i]
        else:
            gpr_params = gprify(X, y, gpr_kernel, bandwidth)
            optimal_meanshift_bandwidth, clusterified = clusterify(gpr_params, iterations=meanshift_iterations)
            changepts = get_clusterified_change_points(X, clusterified).reshape(-1,1)
            evaluations_at_bounding_bandwidths[i] = compare_change_points(changepts, observed_changepoints)
            optimal_meanshift_bandwidths[i] = optimal_meanshift_bandwidth
            optimal_meanshifts[i] = clusterified
            optimal_gpr_paramses[i] = gpr_params
    print("done iterating")
    best_bounding_bandwidth = np.argmin(evaluations_at_bounding_bandwidths)
    new_bounding_bandwidths = sorted([bounding_bandwidths[best_bounding_bandwidth], np.mean(bounding_bandwidths)])
    given_evaluations_at_bounding_bandwidths = [None, None]
    given_evaluations_at_bounding_bandwidths[best_bounding_bandwidth] = evaluations_at_bounding_bandwidths[best_bounding_bandwidth]
    given_optimal_meanshift_bandwidths = [None, None]
    given_optimal_meanshift_bandwidths[best_bounding_bandwidth] = optimal_meanshift_bandwidths[best_bounding_bandwidth]
    given_optimal_meanshifts = [None, None]
    given_optimal_meanshifts[best_bounding_bandwidth] = optimal_meanshifts[best_bounding_bandwidth]
    given_optimal_gpr_paramses = [None, None]
    given_optimal_gpr_paramses[best_bounding_bandwidth] = optimal_gpr_paramses[best_bounding_bandwidth]
    return pick_optimal_gpr_bandwidth_for_changepoint_detection(
        X, y, gpr_kernel, new_bounding_bandwidths, observed_changepoints,
        given_evaluations_at_bounding_bandwidths,
        given_optimal_meanshift_bandwidths,
        given_optimal_meanshifts,
        given_optimal_gpr_paramses,
        iterations=iterations-1, meanshift_iterations=meanshift_iterations)
