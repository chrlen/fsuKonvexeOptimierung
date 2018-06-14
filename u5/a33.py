from functools import partial
import pandas as pd
import numpy as np
import descent as dc
from functools import partial

# Global
MAXIT = 10000
VERBOSE = True
DELTA = 0.02


def phl(x, delta, xi, eta):
    pairs = zip(xi, eta)
    res = delta**2 * sum(map(lambda pair: np.sqrt(1 + (
        (pair[0] * x[0] + x[1] - pair[1]) / delta)**2)	, pairs)) - delta**2 * xi.shape[0]
    return(res)


def dphl(x, delta, xi, eta):
    pairs = zip(xi, eta)
    dx0 = sum([(pair[0]**2 * x[0] + pair[0] * x[1] - pair[0] * pair[1]) / delta**2 *
               np.sqrt(1 + ((pair[0] * x[0] + x[1] - pair[1]) / delta)**2) for pair in pairs])
    pairs = zip(xi, eta)
    dx1 = sum([((pair[0] * x[0] + x[1] - pair[1]) / delta**2) * np.sqrt(1 +
                                                                        ((pair[0] * x[0] + x[1] - pair[1]) / delta)**2) for pair in pairs])
    return(np.array([dx0, dx1]))


data = pd.read_csv("Advertising.csv")
eta = data['Sales']
xi = data['TV']

paritalPhl = partial(phl, delta=DELTA, eta=eta, xi=xi)
paritaldPhl = partial(dphl, delta=DELTA, eta=eta, xi=xi)

startAt = np.array([10, 10])
delta = 0.02

tvSalesParams = dc.gradientDescentArmijoStepwidth(
    paritalPhl,
    paritaldPhl,
    startAt,
    maxit=MAXIT,
    verbose=VERBOSE)

#dc.plotFunction(paritalPhl,-20,20,-20,20,tvSalesParams,"Armijo gradient descent path")
#dc.plotConvergence(paritalPhl,tvSalesParams,'Convergece of Gradient Descent with Armijo-Stepwidth on TV-Sales Data');
