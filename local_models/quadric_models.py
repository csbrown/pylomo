from . import quadrics_utils
import numpy as np
import scipy.optimize
import sklearn.utils
import sklearn.base
import sympy
import itertools

def diagonalize_quadric_and_translate(Q,X):
    #Q is probably lower triangular, so we need to account for that
    Q = np.tril(Q,-1) + np.tril(Q).T
    Q_s = Q[:-1,:-1]
    L, U = np.linalg.eigh(Q_s)
    print(L,U)
    UTQs = U.T@Q_s
    UL = L
    #UR = UTQs@X.T + U.T@Q[:-1,-1:]
    UR = X@UTQs.T + Q[-1:,:-1]@U
    BR = np.einsum('kj,kj->k', X@Q_s, X).reshape(-1,1) + 2*X@Q[-1:,:-1].T + Q[-1,-1]
    return (U, UL, UR, BR)

def array_roots(coeffses):
    ret = np.zeros((coeffses.shape[0], coeffses.shape[1]-1), dtype=complex)
    for i, coeffs in enumerate(coeffses):
        try:
            nproots = np.roots(coeffs.flatten())
            ret[i] = np.pad(nproots, (0, coeffs.shape[0] - (nproots.shape[0] + 1)), constant_values=np.inf)
        except:
            print(nproots, np.pad(nproots, (0, coeffs.shape[0] - (nproots.shape[0] + 1)), constant_values=np.inf))
    return ret

PROJECTION_FUNCS3D = {
    "k_mat": quadrics_utils.k_mat,
    "other_dims_as_x": lambda a,b,c,d,e,f,g,x: [quadrics_utils.y_as_x(a,b,c,d,e,f,g,x), quadrics_utils.z_as_x(a,b,c,d,e,f,g,x)],
}
def orthogonal_quadric_project_x(X,Q,projection_funcs=PROJECTION_FUNCS3D):
    U, ul, ur, br = diagonalize_quadric_and_translate(Q,X)
    #put all of the dimensions right for math stuff
    ul = np.tile(ul.reshape(1,-1), (X.shape[0], 1))
    args = (
        list(ul.T[...,np.newaxis]) +
        [br] +
        list(ur.T[...,np.newaxis])
    )
    
    coeffses = projection_funcs["k_mat"](*args)
    coeffses = np.stack(coeffses, axis=1)

    return U,args,array_roots(coeffses)
 
def orthogonal_quadric_projection(X,Q,projection_funcs=PROJECTION_FUNCS3D):
    U, args, x = orthogonal_quadric_project_x(X,Q,projection_funcs)
    best_pts, best_dists = min_dist_search(U, args, x, projection_funcs=projection_funcs)
    return unrotate_and_translate(U, X, best_pts), best_dists

def min_dist_search(U, args, roots, projection_funcs=PROJECTION_FUNCS3D):
    x = np.ma.array(roots)
    candidates = x.imag == 0
    x.mask = ~candidates
    other_dims = projection_funcs["other_dims_as_x"](*(args + [x]))

    pts = np.ma.stack([x] + other_dims, axis=1)
    dists = np.sum(pts**2, axis=1)
    mindists = np.argmin(dists, axis=1)
    X_indices = np.arange(x.shape[0])
    best_pts = np.array(pts[X_indices,:,mindists], dtype=float)
    best_dists = np.array(dists[X_indices,mindists],dtype=float)
    return best_pts, best_dists

def unrotate_and_translate(U, orig_X, new_X):
    return new_X@U.T + orig_X
    #return np.einsum('ji,ki->kj',U,orig_X) + new_X

def collect_best(expr, measure=sympy.count_ops):
    best = expr
    best_score = measure(expr)
    perms = itertools.permutations(expr.free_symbols)
    permlen = np.math.factorial(len(expr.free_symbols))
    print(permlen)
    for i, perm in enumerate(perms):
        if (permlen > 1000) and not (i%int(permlen/100)):
            print(i)
        collected = sympy.collect(expr, perm)
        if measure(collected) < best_score:
            best_score = measure(collected)
            best = collected
        else:
            factored = sympy.factor(expr)
            if measure(factored) < best_score:
                best_score = measure(factored)
                best = factored
    return best
    
def product(args):
    arg = next(args)
    try:
        return arg*product(args)
    except:
        return arg
    
def rcollect_best(expr, measure=sympy.count_ops):
    best = collect_best(expr, measure)
    best_score = measure(best)
    if expr == best:
        return best
    if isinstance(best, sympy.Mul):
        return product(map(rcollect_best, best.args))
    if isinstance(best, sympy.Add):
        return sum(map(rcollect_best, best.args))

def derive_parabolic_orthogonal_projection_1D_polynomial(n):
    import sympy
    
    Q_sym = sympy.symarray("q", (n+1, n+1))
    Q = sympy.Matrix(np.zeros((n+1,n+1), dtype=int))
    for i, j in itertools.product(range(n+1), range(n+1)):
        if i == n or j == n or i == j:
            Q[i,j] = Q_sym[max(i,j),min(i,j)]
    Q[0,0] = 0
    print(Q)

    x_sym = sympy.symarray("x", n+1)
    X = sympy.Matrix(np.ones((n+1, 1), dtype=int))
    for i in range(n):
        X[i] = x_sym[i]
        
    P = sympy.Matrix(np.zeros((n-1, n+1), dtype=int))
    for i in range(n-1):
        P[i,0] = X[i+1]
        P[i,i+1] = -X[0]
    
    QXP = P*Q*X
    
    other_dims_as_x0 = [sympy.solve(QXP[i], X[i+1])[0] for i in range(n-1)] 
    
    XQX = sympy.expand((X.T*Q*X)[0])
    XQX_as_x0 = XQX.subs({X[i+1]:other_dims_as_x0[i] for i in range(n-1)})
    for sub in other_dims_as_x0:
        XQX_as_x0 *= sympy.fraction(sub)[1]**2
    XQX_as_x0 = sympy.cancel(XQX_as_x0)
    XQX_as_x0 = sympy.simplify(XQX_as_x0)
    XQX_as_x0 = sympy.poly(XQX_as_x0, X[0])
    
    return (X, Q, XQX_as_x0, other_dims_as_x0)

def derive_parabolic_orthogonal_projection_1D_polynomial2(n):
    import sympy
    
    Q_sym = sympy.symarray("q", (n+1, n+1))
    Q = sympy.Matrix(np.zeros((n+1,n+1), dtype=int))
    for i, j in itertools.product(range(n+1), range(n+1)):
        if i == n or j == n or i == j:
            Q[i,j] = Q_sym[max(i,j),min(i,j)]
    Q[0,0] = 0
    print(Q)

    x_sym = sympy.symarray("x", n+1)
    X = sympy.Matrix(np.ones((n+1, 1), dtype=int))
    for i in range(n):
        X[i] = x_sym[i]
        
    Ps = [sympy.Matrix(np.zeros((n-1, n+1), dtype=int)) for j in range(n)]
    for i in range(n-1):
        for j, P in enumerate(Ps):
            if i >= j:
                P[i,j] = X[i+1]
                P[i,i+1] = -X[j]
            else:
                P[i,j] = X[i]
                P[i,i] = -X[j]
    print(Ps)
 
    QXPs = [P*Q*X for P in Ps]
    
    other_dims_as_xjs = []
    for j, QXP in enumerate(QXPs):
        other_dims_as_xj = []
        for i, QXp in enumerate(QXP):
            if i >= j:
                other_dims_as_xj.append(sympy.solve(QXp, X[i+1])[0])
            else:
                other_dims_as_xj.append(sympy.solve(QXp, X[i])[0])
        other_dims_as_xjs.append(other_dims_as_xj)
    
    XQX = sympy.expand((X.T*Q*X)[0])
    XQX_as_xjs = []
    for j, other_dims_as_xj in enumerate(other_dims_as_xjs):
        subs = {}
        for i, other_dim_as_xj in enumerate(other_dims_as_xj):
            if i >= j:
                k = i+1
            else:
                k = i
            subs[X[k]] = other_dim_as_xj    
        XQX_as_xjs.append(XQX.subs(subs))

    for j, XQX_as_xj in enumerate(XQX_as_xjs):
        for sub in other_dims_as_xjs[j]:
            XQX_as_xjs[j] *= sympy.fraction(sub)[1]**2
        XQX_as_xjs[j] = sympy.cancel(XQX_as_xjs[j])
        XQX_as_xjs[j] = sympy.simplify(XQX_as_xjs[j])
        XQX_as_xjs[j] = sympy.poly(XQX_as_xjs[j], X[j])
    
    return (X, Q, XQX_as_xjs, other_dims_as_xjs)

def librarify_parabolic_equations2(n, collectify=True):
    X, Q, XQX_as_xjs, other_dims_as_xjs = derive_parabolic_orthogonal_projection_1D_polynomial2(n)
    if collectify:
        XQX_as_xjs_coeffs = [collectify_polynomial_coefficients(XQX_as_xj) for XQX_as_xj in XQX_as_xjs]
    else:
        XQX_as_xjs_coeffs = [XQX_as_xj.all_coeffs() for XQX_as_xj in XQX_as_xjs]

    k_mats = [sympy.Matrix(XQX_as_xj_coeffs) for XQX_as_xj_coeffs in XQX_as_xjs_coeffs]

    with open("parabolic_utils_{:d}.py".format(n), "w") as f:

        Q_args = list(map(str,[Q[i,i] for i in range(1,Q.shape[0])])) + list(map(str,[Q[-1,i] for i in range(Q.shape[0]-1)]))

        other_dims_as_xj_funcs = [funcify("other_dims_as_x{:d}_{:d}({})".format(j, n,
            ','.join(Q_args +
                [str(X[j])])
            ),
        str(other_dims_as_xjs[j])) for j in range(n)]

        k_mat_funcs = [funcify("k_mat{:d}_{:d}({})".format(j, n,
            ','.join(Q_args),
        ),
        str(k_mats[j].transpose().tolist()[0])) for j in range(n)]

        list(map(f.write, other_dims_as_xj_funcs + k_mat_funcs))





def librarify_parabolic_equations(n, collectify=True):
    X, Q, XQX_as_x0, other_dims_as_x0 = derive_parabolic_orthogonal_projection_1D_polynomial(n)
    if collectify:
        XQX_as_x0_coeffs = collectify_polynomial_coefficients(XQX_as_x0)
    else:
        XQX_as_x0_coeffs = XQX_as_x0.all_coeffs()

    k_mat = sympy.Matrix(XQX_as_x0_coeffs)

    with open("parabolic_utils_{:d}.py".format(n), "w") as f:

        Q_args = list(map(str,[Q[i,i] for i in range(1,Q.shape[0])])) + list(map(str,[Q[-1,i] for i in range(Q.shape[0]-1)]))

        other_dims_as_x_func = funcify("other_dims_as_x_{:d}({})".format(n,
            ','.join(Q_args +
                [str(X[0])])
            ),
        str(other_dims_as_x0))

        k_mat_func = funcify("k_mat_{:d}({})".format(n,
            ','.join(Q_args),
        ),
        str(k_mat.transpose().tolist()[0]))

        list(map(f.write, [other_dims_as_x_func, k_mat_func]))






def derive_quadratic_orthogonal_projection_1D_polynomial(n):
    import sympy
    
    Q_sym = sympy.symarray("q", (n+1, n+1))
    Q = sympy.Matrix(np.zeros((n+1,n+1), dtype=int))
    for i, j in itertools.product(range(n+1), range(n+1)):
        if i == n or j == n or i == j:
            Q[i,j] = Q_sym[max(i,j),min(i,j)]
    print(Q)

    x_sym = sympy.symarray("x", n+1)
    X = sympy.Matrix(np.ones((n+1, 1), dtype=int))
    for i in range(n):
        X[i] = x_sym[i]
        
    P = sympy.Matrix(np.zeros((n-1, n+1), dtype=int))
    for i in range(n-1):
        P[i,0] = X[i+1]
        P[i,i+1] = -X[0]
    
    QXP = P*Q*X
    
    other_dims_as_x0 = [sympy.solve(QXP[i], X[i+1])[0] for i in range(n-1)] 
    
    XQX = sympy.expand((X.T*Q*X)[0])
    XQX_as_x0 = XQX.subs({X[i+1]:other_dims_as_x0[i] for i in range(n-1)})
    for sub in other_dims_as_x0:
        XQX_as_x0 *= sympy.fraction(sub)[1]**2
    XQX_as_x0 = sympy.cancel(XQX_as_x0)
    XQX_as_x0 = sympy.simplify(XQX_as_x0)
    XQX_as_x0 = sympy.poly(XQX_as_x0, X[0])
    
    return (X, Q, XQX_as_x0, other_dims_as_x0)
   
def collectify_polynomial_coefficients(poly):
    return [rcollect_best(formula) for formula in poly.all_coeffs()]

def funcify(signature, returnable):
    return "def {}: return {}\n".format(signature, returnable)

def librarify_conic_equations(n, collectify=True):
    X, Q, XQX_as_x0, other_dims_as_x0 = derive_quadratic_orthogonal_projection_1D_polynomial(n)
    if collectify:
        XQX_as_x0_coeffs = collectify_polynomial_coefficients(XQX_as_x0)
    else:
        XQX_as_x0_coeffs = XQX_as_x0.all_coeffs()

    k_mat = sympy.Matrix(XQX_as_x0_coeffs)

    with open("quadrics_utils_{:d}.py".format(n), "w") as f:

        Q_args = list(map(str,[Q[i,i] for i in range(Q.shape[0])])) + list(map(str,[Q[-1,i] for i in range(Q.shape[0]-1)]))

        other_dims_as_x_func = funcify("other_dims_as_x_{:d}({})".format(n,
            ','.join(Q_args +
                [str(X[0])])
            ),
        str(other_dims_as_x0))

        k_mat_func = funcify("k_mat_{:d}({})".format(n,
            ','.join(Q_args),
        ),
        str(k_mat.transpose().tolist()[0]))

        list(map(f.write, [other_dims_as_x_func, k_mat_func]))

def librarify_quadric_equations():
    import sympy
    from sympy.abc import a,b,c,d,e,f,g,x,y,z

    Q = a*x**2 + b*y**2 + c*z**2 + 2*e*x + 2*f*y + 2*g*z + d
    y_as_x_num = f*x
    y_as_x_den = e-(b-a)*x
    y_as_x = y_as_x_num/y_as_x_den
    z_as_x_num = g*x
    z_as_x_den = e-(c-a)*x
    z_as_x = z_as_x_num/z_as_x_den
    Q_as_x = Q.subs({
        y: y_as_x,
        z: z_as_x,
    })

    bigQ = sympy.expand(sympy.simplify(Q_as_x*y_as_x_den**2*z_as_x_den**2))

    coeffs = list(map(sympy.factor, sympy.poly(bigQ,x).all_coeffs()))

    collected = []
    for coeff in coeffs:
        collected.append(rcollect_best(coeff))

    k_mat = sympy.Matrix(collected)
    k_jac = k_mat.jacobian([a,b,c,d,e,f,g])

    with open("quadrics_utils.py", "w") as f:
        Q_func = funcify("Q(a,b,c,d,e,f,g,x,y,z)",str(Q))
        y_as_x_func = funcify("y_as_x(a,b,c,d,e,f,g,x)", str(y_as_x))
        z_as_x_func = funcify("z_as_x(a,b,c,d,e,f,g,x)", str(z_as_x))
        Q_as_x_func = funcify("Q_as_x(a,b,c,d,e,f,g,x)", str(Q_as_x))
        k_mat_func = funcify("k_mat(a,b,c,d,e,f,g)", str(k_mat.transpose().tolist()[0]))
        k_jac_func = funcify("k_jac(a,b,c,d,e,f,g)", str(k_jac.tolist()))
        individual_k_funcs = [funcify("k{:01d}(a,b,c,d,e,f,g)".format(i), str(eq)) for i,eq in enumerate(collected[::-1])]
        list(map(f.write, [Q_func, y_as_x_func, z_as_x_func, Q_as_x_func, k_mat_func, k_jac_func] + individual_k_funcs)) 

def weighted_avg_and_std(values, weights=None, axis=None):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    if weights is None:
        return np.average(values, axis=axis), np.std(values, axis=axis)
    average = np.average(values, weights=weights, axis=axis)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return (average, np.sqrt(variance))

class QuadricModel(sklearn.base.RegressorMixin, sklearn.base.BaseEstimator):
    def __init__(self, n):
        self.n = n
    
    def _optimizer_logger(self, xk, *args):
        logging.info("quadric modeling: it{}, xk{}".format(self.i, xk))
        self.i += 1      
            
    def fit(self, X, y=None, sample_weight=None, beta0=None):
        logging.info("fitting quadric model")
        X = sklearn.utils.check_array(X)
        if beta0 is None:
            #this allows us to orient the data to set an initial set of parameters
            global_linear_model = local_models.TLS_models.LinearODR_mD(self.n-1)
            global_linear_model.fit(X, sample_weight=sample_weight)
            global_linear_vecs = global_linear_model.cov_eigenvectors[global_linear_model.cov_eigenvalues_sorter]
            global_linear_mean = global_linear_model.intercept_

            transformed_X = (X-global_linear_mean)@global_linear_vecs.T

            transformed_q = np.zeros((self.n+1,self.n+1))
            
            # an elliptic paraboloid pointing in the smallest eigenvector direction
            '''
            transformed_q[0,0] = 1/weighted_avg_and_std(transformed_X[0], weights=None)[1]**2 #flat-ish in x
            transformed_q[1,1] = 1/weighted_avg_and_std(transformed_X[1], weights=None)[1]**2 #flat-ish in y
            transformed_q[2,3] = transformed_q[3,2] = -1 # elliptic paraboloid pointing in z direction
            '''
            
            # an ellipse with axes in the various directions
            transformed_q[np.diag_indices(self.n)] = 1/global_linear_model.cov_eigenvalues[global_linear_model.cov_eigenvalues_sorter]**2
            transformed_q[self.n,self.n] = -1
            #print(np.diag(transformed_q), weighted_avg_and_std(transformed_X, weights = sample_weight, axis=0)[1])
            
            #print(transformed_q)
            #print(global_linear_mean)
            E = np.block([[global_linear_vecs, global_linear_mean.reshape(-1,1)],[np.zeros(self.n),1]])
            Einv = np.linalg.inv(E)
            q = Einv.T@transformed_q@Einv
            print(q)
            beta0 = q[np.triu_indices(self.n+1)]
        
        beta0 /= beta0[-1]
        
        self.i = 0
        self.solution = scipy.optimize.minimize(self._loss, beta0[:-1], args=(X, sample_weight), callback=self._optimizer_logger)#, method = 'Nelder-Mead')
        self.coef_ = self.solution.x
        self.intercept_ = np.array([])
        return self
            
    def _loss(self, beta, X, sample_weight=None):
        q = np.zeros((self.n+1,self.n+1))
        beta = np.concatenate((beta, [1]))
        q[np.triu_indices(self.n+1)] = beta
        q[self.n,self.n] = 1
        q += q.T
        q[np.diag_indices(self.n+1)] /= 2
        pts, dists = orthogonal_quadric_projection(X,q,projection_funcs)

        #pts, jac, dists = orthogonal_quadric_projection_with_jacobian(X,q,projection_funcs)
        
        if sample_weight is not None:
            dists *= sample_weight
        lost = np.sum(dists)
        #total_jac = np.einsum('ki,kij->ij', (X - pts), jac)
        #if lost > 20:
        #    print(q)
        #print(lost)
        return lost#, total_jac

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("--parabolic", action="store_true")
    args = parser.parse_args()
    if not args.parabolic:
        librarify_conic_equations(args.n)
    else:
        librarify_parabolic_equations2(args.n)
    #librarify_quadric_equations()
