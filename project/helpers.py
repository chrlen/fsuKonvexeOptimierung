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
