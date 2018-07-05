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
simpleQuadraticQ = np.diag([4, 2])
simpleQuadraticq = np.array([-4, -2])
simpleQuadraticc = 3

simpleQuadraticFct = partial(evalQuadraticForm, Q=simpleQuadraticQ, q=simpleQuadraticq, c=simpleQuadraticc)
simpleQuadraticGradient = partial(evalFirstOrderGradientOfQuadraticForm, Q=simpleQuadraticQ, q=simpleQuadraticq)
simpleQuadraticHessian = lambda x: simpleQuadraticQ
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

def squarerootExample(x):
    return(np.sqrt(1 + x**2))

def squarerootExampleGradient(x):
    return(x / squarerootExample(x))

def squarerootExampleHessian(x):
    return((1 + x**2)**(-3 / 2))

#L1 SVM

def l1_svm(w,X,Y,c=10):
    t = 0.5 * w[:2].T.dot(w[:2])
    indexSet = range(X.shape[0])
    reg = sum([max(0,1 - Y[i]*(w[0] * X[i,0] + w[1] * X[i,1] + w[2])) for i in indexSet])
    return t + c * reg


    #t = 0.5 * w[:2].T.dot(w[:2])
    #pairs = zip(X,Y)
    #indexSet = range(X.shape[0])
    #dw1 = w[0]  - c *    sum([Y[i]* X[i,0]       if 1 - Y[i]*(w[0] * X[i,0] + w[1] * X[i,1] + w[2]) > 0 else 0   for i in indexSet])
    #dw1 = w[1]  - c *    sum([Y[i] *X[i,1]       if 1 - Y[i]*(w[0] * X[i,0] + w[1] * X[i,1] + w[2]) > 0 else 0   for i in indexSet])
    #db =        -c *    sum([Y[i]              if 1 - Y[i]*(w[0] * X[i,0] + w[1] * X[i,1] + w[2]) > 0 else 0   for i in indexSet])

    #return 0.5 * w[:2].T.dot(w[:2]) + c * sum([max(0,1-(pair[1]* (w[0] * pair[0][0] + w[1]* pair[0][1] + w[2]))) for pair in pairs])
    #return 0.5 * 


def dl1_svm(w,X,Y,c=10):
    pairs = zip(X,Y)
    dw1 = w[0] - c * sum([ pair[1] * pair[0][0] for pair in pairs])
    pairs = zip(X,Y)
    dw2 = w[1] - c * sum([ pair[1] * pair[0][1] for pair in pairs])
    pairs = zip(X,Y)
    db =  -c * sum([ pair[1] for pair in pairs])
    return(np.array([dw1,dw2,db]))

def hl1_svm(w):
    return(np.vstack([[1,0,0],[0,1,0],[0,0,1]]))


#L2 SVM
def l2_svm(w,X,Y,c=10):
    t = 0.5 * w[:2].T.dot(w[:2])
    indexSet = range(X.shape[0])
    reg = [max(0,1-Y[i]*(w[0] * X[i,0] + w[1] * X[i,1] + w[2])) for i in indexSet]
    maxReg = np.array(list(map(lambda x: max(0,x),reg)))**2
    return t + c * sum(maxReg)

def dl2_svm(w,X,Y,c=10):
    indexSet = range(X.shape[0])
    dw1 = w[0]  - 2 *   c * sum([Y[i] *X[i,0]  * max(0,(1 - Y[i]*(w[0] * X[i,0] + w[1] * X[i,1] + w[2]))) for i in indexSet])
    dw2 = w[1]  - 2 *   c * sum([Y[i] *X[i,1]  * max(0,(1 - Y[i]*(w[0] * X[i,0] + w[1] * X[i,1] + w[2]))) for i in indexSet])
    db =        - 2 *   c * sum([Y[i]          * max(0,(1 - Y[i]*(w[0] * X[i,0] + w[1] * X[i,1] + w[2]))) for i in indexSet])
    return(np.array([dw1,dw2,db]))

def hl2_svm(w,X,Y,c=10):
    indexSet = range(X.shape[0])
    dw1dw1 = 1 + c *    sum([2*Y[i]**2 * X[i,0]**2        if 1 - Y[i]*(w[0] * X[i,0] + w[1] * X[i,1] + w[2]) > 0 else 0   for i in indexSet])
    dw1dw2 = c *        sum([2*Y[i]**2 * X[i,0] * X[i,1]  if 1 - Y[i]*(w[0] * X[i,0] + w[1] * X[i,1] + w[2]) > 0 else 0   for i in indexSet])
    dw1db = c *         sum([2*Y[i]**2 * X[i,0]           if 1 - Y[i]*(w[0] * X[i,0] + w[1] * X[i,1] + w[2]) > 0 else 0   for i in indexSet])
    dw2dw1 = dw1dw2
    dw2dw2 = 1 + c *    sum([2*Y[i]**2 * X[i,1]**2        if 1 - Y[i]*(w[0] * X[i,0] + w[1] * X[i,1] + w[2]) > 0 else 0   for i in indexSet])
    dw2db = c *         sum([2*Y[i]**2 * X[i,1]           if 1 - Y[i]*(w[0] * X[i,0] + w[1] * X[i,1] + w[2]) > 0 else 0   for i in indexSet])
    dbdw1 = dw1db
    dbdw2 = c *         sum([2*Y[i]**2 * X[i,1]           if 1 - Y[i]*(w[0] * X[i,0] + w[1] * X[i,1] + w[2]) > 0 else 0   for i in indexSet])
    dbdb = c *          sum([2*Y[i]**2                    if 1 - Y[i]*(w[0] * X[i,0] + w[1] * X[i,1] + w[2]) > 0 else 0   for i in indexSet])

    return np.vstack([
        [dw1dw1,dw1dw2,dw1db],
        [dw2dw1,dw2dw2,dw2db],
        [dbdw1,dbdw2,dbdb]])
    

#Log SVM
def log_svm(w,X,Y,c=10):
    t = 0.5 * w[:2].T.dot(w[:2])
    indexSet = range(X.shape[0])
    reg = sum([np.log(1+np.exp(1 - Y[i]*(w[0] * X[i,0] + w[1] * X[i,1] + w[2]))) for i in indexSet ])
    return t + c * reg


