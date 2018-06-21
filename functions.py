from functools import partial
import numpy as np
import pandas as pd

# Global
MAXIT = 10000
VERBOSE = True
DELTA = 0.02

# Helpers:


def evalQuadraticForm(x, Q, q, c):
    return 0.5 * x.dot(Q).dot(x) + q.dot(x) + c


def evalFirstOrderGradientOfQuadraticForm(x, Q, q):
    return Q.dot(x) + q

# Special functions


def rosenbrock(x):
    x_1, x_2 = x
    return(100 * (x_2 - x_1**2)**2 + (1 - x_1)**2)


def himmelblau(x):
    x_1, x_2 = x
    return((x_1**2 + x_2 - 11)**2 + (x_1 + x_2**2 - 7)**2)


def bazaraa_shetty(x):
    x_1, x_2 = x
    return((x_1 - 2))**4 + (x_1 - 2 * x_2)**2


def beale(x):
    x_1, x_2 = x
    return((1.5 - x_1 * (1 - x_2))**2 + (2.25 - x_1 * (1 - x_2**2))**2 + (2.625 - x_1 * (1 - x_2**3))**2)


def spellucci(x):
    x_1, x_2 = x
    return(2 * x_1**3 + x_2**2 + x_1**2 * x_2**2 + 4 * x_1 * x_2 + 3)


# Simple quadratic problem example:
Q = np.diag([4, 2])
q = np.array([-4, -2])
c = 3

simpleQuadraticFct = partial(evalQuadraticForm, Q=Q, q=q, c=c)
simpleQuadraticGradient = partial(
    evalFirstOrderGradientOfQuadraticForm, Q=Q, q=q)

# Linear Regression with TV-Sales data:
data = pd.read_csv("Advertising.csv")
eta = data['Sales']
xi = data['TV']

Q = np.vstack([[sum(xi**2), sum(xi)], [
    sum(xi), xi.shape[0]]])
q = np.array([-1 * sum(xi * eta), - 1 * sum(eta)])
c = sum(eta**2)

regressionFct = partial(evalQuadraticForm, Q=Q, q=q, c=c)
regressionGradient = partial(evalFirstOrderGradientOfQuadraticForm, Q=Q, q=q)
regressionHessian = lambda x: Q
# Exponential Function


def f_exp(x):
    return( exp(x[0] + 3 * x[1] - 0.1) + exp(x[0] - 3 * x[1] - 0.1) + exp(-1 * x[0] - 0.1))


def df_exp(x):
    dx_0 = exp(x[0] + 3 * x[1] - 0.1) + exp(x[0] -
                                            3 * x[1] - 0.1) - exp(-1 * x[0] - 0.1)
    dx_1 = exp(x[0] + 3 * x[1] - 0.1) * 3 + exp(x[0] - 3 * x[1] - 0.1) * 3
    return(np.array([dx_0, dx_1]))

# Robust regression with pseudo-huber-loss


def pseudoHuberLoss(x, delta, xi, eta):
    pairs = zip(xi, eta)
    res = delta**2 * sum(map(lambda pair: np.sqrt(1 + (
        (pair[0] * x[0] + x[1] - pair[1]) / delta)**2)  , pairs)) - delta**2 * xi.shape[0]
    return(res)


def pseudoHuberLossGradient(x, delta, xi, eta):
    pairs = zip(xi, eta)
    dx0 = sum([(pair[0]**2 * x[0] + pair[0] * x[1] - pair[0] * pair[1]) / delta**2 *
               np.sqrt(1 + ((pair[0] * x[0] + x[1] - pair[1]) / delta)**2) for pair in pairs])
    pairs = zip(xi, eta)
    dx1 = sum([((pair[0] * x[0] + x[1] - pair[1]) / delta**2) * np.sqrt(1 +
                                                                        ((pair[0] * x[0] + x[1] - pair[1]) / delta)**2) for pair in pairs])
    return(np.array([dx0, dx1]))

def calculatePseudoHuberHessian(x,delta,xi,eta):
    pairs = zip(xi, eta)
    dx0x0 = sum([ pair[0]**2 * (np.sqrt(1 + ((pair[0]*x[0]+x[1]-pair[1])/delta)**2) -    ((  (pair[0]**2 * x[0] + pair[0]*x[1] - pair[0]*pair[1]) * (pair[0]*x[0]+x[1]-pair[1]) *(pair[0]*delta + pair[0] * x[0] +x[1] - pair[1]))/(delta * np.sqrt(1+((pair[0]*x[0]+x[1]-pair[1])/delta)**2))))/(1+((pair[0]*x[0]+x[1]-pair[1])/(delta))**2) for pair in pairs])
    pairs = zip(xi, eta)
    dx0x1 = sum([ pair[0] * (np.sqrt(1 + ((pair[0]*x[0]+x[1]-pair[1])/delta)**2) -       ((  (pair[0]**2 * x[0] + pair[0]*x[1] - pair[0]*pair[1]) * (pair[0]*x[0]+x[1]-pair[1]) *(pair[0]*delta + pair[0] * x[0] +x[1] - pair[1]))/(delta * np.sqrt(1+((pair[0]*x[0]+x[1]-pair[1])/delta)**2))))/(1+((pair[0]*x[0]+x[1]-pair[1])/(delta))**2) for pair in pairs])
    pairs = zip(xi, eta)
    dx1x0 = sum([ pair[0] * (np.sqrt(1 + ((pair[0]*x[0]+x[1]-pair[1])/delta)**2) -       ((   (pair[0]*x[0]+x[1]-pair[1])**2 *(pair[0]*delta + pair[0] * x[0] +x[1] - pair[1]))/(delta * np.sqrt(1+((pair[0]*x[0]+x[1]-pair[1])/delta)**2))))/(1+((pair[0]*x[0]+x[1]-pair[1])/(delta))**2) for pair in pairs])
    pairs = zip(xi, eta)
    dx1x1 = sum([ (np.sqrt(1 + ((pair[0]*x[0]+x[1]-pair[1])/delta)**2) -                 ((   (pair[0]*x[0]+x[1]-pair[1])**2 *(delta + pair[0] * x[0] +x[1] - pair[1]))/(delta * np.sqrt(1+((pair[0]*x[0]+x[1]-pair[1])/delta)**2))))/(1+((pair[0]*x[0]+x[1]-pair[1])/(delta))**2) for pair in pairs])
    Q = np.vstack([[dx0x0,dx0x1], [dx1x0,dx1x1]])
    return Q

data = pd.read_csv("Advertising.csv")
eta = data['Sales']
xi = data['TV']

paritalPhl = partial(pseudoHuberLoss, delta=DELTA, eta=eta, xi=xi)
paritaldPhl = partial(pseudoHuberLossGradient, delta=DELTA, eta=eta, xi=xi)
pseudoHuberHessian = partial(calculatePseudoHuberHessian,delta=DELTA, eta=eta, xi=xi)


#
def squarerootExample(x):
    return(np.sqrt(1 + x**2))

def squarerootExampleGradient(x):
    return(x / squarerootExample(x))

def squarerootExampleHessian(x):
    return((1 + x**2)**(-3 / 2))
