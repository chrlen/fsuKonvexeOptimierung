import numpy as np
import numpy.linalg as npl

def crit1(fx_thisTime, fx_nextTime, epsilon):
    return(fx_thisTime - fx_nextTime <= epsilon * max(1, abs(fx_thisTime)))

def crit2(x_thisTime, x_nextTime, epsilon2):
    return(npl.norm(x_nextTime - x_thisTime) <= epsilon2 * max(1, npl.norm(x_thisTime)))

def crit3(fx_thisTime, dfx_thisTime, epsilon3):
    return(npl.norm(dfx_thisTime) <= epsilon3 * max(1, np.abs(fx_thisTime)))

def orCriterias(f, df, x_thisTime, x_nextTime, epsilon, epsilon2, epsilon3):
    fx_thisTime = f(x_thisTime)
    fx_nextTime = f(x_nextTime)
    dfx_thisTime = df(x_thisTime)
    return crit1(fx_thisTime, fx_nextTime, epsilon) or crit2(x_thisTime, x_nextTime, epsilon2) or crit3(fx_thisTime, dfx_thisTime, epsilon3)



def armijoCrit(f, df, sigma, delta, x, d):
    return(f(x + sigma * d) <= f(x) + delta * sigma * df(x).T.dot(d))

def armijoStepwidth(x, f, df, d,
                    beta1=0.5,
                    beta2=0.5,
                    delta=0.01,
                    gamma=(1 / 10**4),
                    sigma_0=1,
                    maxit=20,
                    verbose=True):

    iterations = 0
    sigma = sigma_0

    while iterations < maxit:
        if(armijoCrit(f, df, sigma, delta, x, d)):
            if(verbose):
                print("Armijo took : " + str(iterations) +
                      " iterations, the choosen Stepwidth is: " + str(sigma))
            return(sigma)
        else:
            sigma = beta1 * sigma
            if verbose:
                print(str(sigma))
        iterations += 1
    if verbose:
        print("Armijo Stepwidth reached maximum number of iterations")
    return(sigma)




class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class GradientError(Error):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

class HessianError(Error):
    def __init__(self, next, message):
        self.next = next
        self.message = message

def approximateGradient(x, f, epsilon=0.01):
    n = x.shape[0]
    h = np.array([epsilon * max(1, abs(y)) for y in x])
    g = np.zeros(n)
    for k in range(n):
        d = np.zeros(n)
        d[k] = h[k]
        g[k] = (f(x + d) - f(x - d)) / 2 * h[k]
    return(g)

def checkGradient(x, f, df, epsilon=0.001):
    app = approximateGradient(x, f, epsilon)
    #print(app,df(x))
    #res = (npl.norm(df(x) - app) / (npl.norm(df(x)) + 1)) > epsilon
    #if res:
    #    raise GradientError("(npl.norm(df(x) - app) / (npl.norm(df(x)) + 1)) > epsilon",
    #                        "Gradient Check failed")


def approximateHessian(x, f, epsilon):
    n = x.shape[0]
    h = np.array([epsilon * max(1, abs(y)) for y in x])
    H = np.zeros([n, n])
    for k in range(n):
        d_one = np.zeros(n)
        d_one[k] = h[k]
        for l in range(n):
            d_two = np.zeros(n)
            d_two[l] = h[l]
            H[k, l] = (f(x + d_one + d_two) - f(x + d_one - d_two) +
                       f(x - d_one - d_two) -  f(x - d_one + d_two)) / 4 * h[k] * h[l]
    return(H)

def checkHessian(x, f, hf, epsilon=0.0001):
    res =  npl.norm(hf(x) - approximateHessian(x, f, epsilon)) * (npl.norm(hf(x)) + 1) > epsilon
    #if res:
    #    raise HessianError("npl.norm(hf(x) - approximateHessian(x, f, epsilon)) * (npl.norm(hf(x)) + 1) > epsilon","Hessian Check failed")