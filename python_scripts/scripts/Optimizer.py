import numpy as np
import scipy as sc
import scipy.optimize as opt 


# class Optimizer(object):
class Optimizer(object):
        def __init__(self, X_sample, Y_sample, acquisition, bounds):
                self.dim = 2
                self.acquisition = acquisition
                self.bounds = bounds 
                self.x_sample = X_sample
                self.y_sample = Y_sample

        def propose_location(self, n_restarts):
                def min_obj(X):
                        # return 0 return Minimize the negative of acquisition function
                        return -self.acquisition(X.reshape(-1, self.dim), self.x_sample, self.y_sample)

                # Find the best optimum by starting from n_restart different random points.
                for x0 in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_restarts, self.dim)):
                        res = opt.minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')        
                        if res.fun < min_val:
                                min_val = res.fun[0]
                                min_x = res.x           
                        
                return min_x.reshape(-1, 1) 

# #Unconstrained Example
# x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
# res = opt.minimize(opt.rosen, x0, method='Nelder-Mead', tol=1e-6)
# res.x

# res = opt.minimize(opt.rosen, x0, method='BFGS', jac=opt.rosen_der,
#                options={'gtol': 1e-6, 'disp': True})
# print(res.x)
# print(res.hess_inv)

# #Constrained example
# fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
# cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
#         {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
#         {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
# bnds = ((0, None), (0, None))
# res = opt.minimize(fun, (2, 0), method='SLSQP', bounds=bnds,
#                constraints=cons)
# print(res.x)


