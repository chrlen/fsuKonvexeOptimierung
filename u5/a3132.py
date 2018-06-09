
import descent as dc
import numpy as np
import pandas as pd
from math import exp
from functools import partial
# Global
MAXIT = 100
VERBOSE = True
SCALE = 0.000001


# Linear Regression
data = pd.read_csv("Advertising.csv")
eta = data['Sales']
xi = data['TV']
Q = np.vstack([[sum(xi**2), sum(xi)], [sum(xi), xi.shape[0]]])
q = np.array([-1 * sum(xi * eta), - 1 * sum(eta)])
c = sum(eta**2)
startAt = np.array([8, 8])

f_reg = partial(dc.evalQuadraticForm, Q=Q, q=q, c=c)
df_reg = partial(dc.evalFirstOrderGradientOfQuadraticForm, Q=Q, q=q)

tvSalesParams = dc.gradientDescentArmijoStepwidth(
    f_reg,
    df_reg,
    startAt,
    maxit=MAXIT,
    verbose=VERBOSE)

dc.plotFunction(f_reg, -20, 20, -20, 20, tvSalesParams,
                "Armijo gradient descent path")
dc.plotConvergence(f_reg, tvSalesParams,
                   'Convergece of Gradient Descent with Armijo-Stepwidth on TV-Sales Data')

# Exponential Function
def f_exp(x):
    res = exp(x[0] + 3 * x[1] - 0.1) + exp(x[0] - 3 * x[1] - 0.1) + exp(-1 * x[0] - 0.1)
    return(SCALE * res)


def df_exp(x):
    dx_0 = exp(x[0] + 3 * x[1] - 0.1) + exp(x[0] - 3 * x[1] - 0.1) - exp(-1 * x[0] - 0.1)
    dx_1 = exp(x[0] + 3 * x[1] - 0.1) * 3 + exp(x[0] - 3 * x[1] - 0.1) * 3
    return(np.array([ SCALE * dx_0, SCALE * dx_1]))


startAt = np.array([4, 4])

exp_optimum = dc.gradientDescentArmijoStepwidth(
    f_exp,
    df_exp,
    startAt,
    maxit=MAXIT,
    verbose=VERBOSE)

dc.plotFunction(f_exp, -5, 5, -5, 5, exp_optimum,
                "Armijo gradient descent path")

dc.plotConvergence(f_exp, exp_optimum,
                   'Convergece of Gradient Descent with Armijo-Stepwidth on Sum of Exponentials')
