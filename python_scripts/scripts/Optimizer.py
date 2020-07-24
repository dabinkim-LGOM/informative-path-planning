import numpy as np
import scipy as sc
import scipy.optimize as opt 


# class Optimizer(object):

#Unconstrained Example
x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
res = opt.minimize(opt.rosen, x0, method='Nelder-Mead', tol=1e-6)
res.x

res = opt.minimize(opt.rosen, x0, method='BFGS', jac=opt.rosen_der,
               options={'gtol': 1e-6, 'disp': True})
print(res.x)
print(res.hess_inv)

#Constrained example
fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
bnds = ((0, None), (0, None))
res = opt.minimize(fun, (2, 0), method='SLSQP', bounds=bnds,
               constraints=cons)
print(res.x)
