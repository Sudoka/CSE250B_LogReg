"""
Demo of the L-BFGS algorithm, a quasi-Newton optimization method.
"""

import numpy as np
from scipy.optimize import minimize

x_true = np.arange(0,10,0.1)
m_true = 2.5
b_true = 1.0
y_true = m_true*x_true + b_true

def func(params, *args):
    x = args[0]
    y = args[1]
    m, b = params
    y_model = m*x+b
    error = y-y_model
    return sum(error**2)

initial_values = np.array([1.0, 0.0])
mybounds = [(None,2), (None,None)]

res = minimize(func, x0=initial_values, args=(x_true,y_true), method='L-BFGS-B')
print res
