import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt


def evalQuadraticForm(x, Q, q, c):
    return 0.5 * x.dot(Q).dot(x) + q.dot(x) + c


def evalFirstOrderGradientOfQuadraticForm(x, Q, q):
    return Q.dot(x) + q


def evalField(f, X, Y):
    x_range = range(0, X.shape[0])
    y_range = range(0, X.shape[1])
    Z = np.zeros(X.shape)

    for x in x_range:
        for y in y_range:
            v = np.array([X[y, x], Y[y, x]])
            Z[y, x] = f(v)

    return Z


def plotFunction(f,
                 xMin, xMax,
                 yMin, yMax,
                 path,
                 title):
    x = np.arange(xMin, xMax, 0.1)
    y = np.arange(yMin, yMax, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = evalField(f, X, Y)

    plt.figure()
    plt.contourf(X, Y, Z)
    for i in range(1, len(path)):
        plt.plot([path[i][0], path[i - 1][0]],
                 [path[i][1], path[i - 1][1]],
                 color='white', marker='o')
    plt.colorbar()
    plt.title(title)
    plt.show()


def plotConvergence(f, path, title):
    opt = path[0]
    dist = list(map(lambda x: np.log(npl.norm(x - opt, ord=1) + 1), path))
    points = list(reversed(dist))
    plt.plot(points)
    plt.title(title)
    plt.show()


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


def armijoStepwidth(
        x,
        f,
        df,
        d,
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


def gradientDescentArmijoStepwidth(
        f,
        df,
        startAt,
        epsilon=(1 / (10**6)**2),
        epsilon2=(1 / float(10**6)),
        epsilon3=(1 / float(10**6)),
        beta1=0.5,
        beta2=0.5,
        delta=0.01,
        gamma=(1 / 10**4),
        sigma_0=1,
        maxit=1000,
        verbose=True):
    optimumNow = startAt
    if verbose:
        print("Starting at: " + str(startAt))
        print("epsilon: " + str(epsilon))
        print("epsilon2: " + str(epsilon2))
        print("epsilon3: " + str(epsilon3))
        print("f(x) | df(x) | sigma ")
    finished = False
    stepsTaken = list()
    stepsTaken.insert(0, optimumNow)
    iterations = 1

    while not finished:
        d = -1 * df(optimumNow)
        sigma_i = armijoStepwidth(
            x=optimumNow,
            f=f,
            df=df,
            d=d,
            beta1=beta1,
            beta2=beta2,
            delta=delta,
            gamma=gamma,
            sigma_0=sigma_0,
            verbose=False
        )
        optimumNext = optimumNow + sigma_i * d
        stepsTaken.insert(0, optimumNext)

        if verbose:
            print(str(optimumNow) + str(optimumNext) + str(f(optimumNow)
                                                           ) + " | " + str(df(optimumNow)) + " | " + str(sigma_i))
        if orCriterias(f, df, optimumNow, optimumNext, epsilon, epsilon2, epsilon3):
            if verbose:
                print("Gradient descent with Armijo Stepwidth took: " +
                      str(iterations) + " iterations.")
                print("Gradient descent with Armijo Stepwidth found: " +
                      str(optimumNext) + ".")
            return(stepsTaken)
        if iterations > maxit:
            if verbose:
                print("Gradient descent with Armijo Stepwidth took: " +
                      str(iterations) + " iterations.")
                print("Gradient descent with Armijo Stepwidth found: " +
                      str(optimumNext) + ".")
            return(stepsTaken)

        iterations += 1
        optimumNow = optimumNext


def approximateGradient(x, f, epsilon=0.0001):
    n = x.shape[0]
    h = np.array([epsilon * max(1, abs(y)) for y in x])
    g = np.zeros(n)
    for k in range(n):
        d = np.zeros(n)
        d[k] = h[k]
        g[k] = (f(x + d) - f(x - d)) / 2 * h[k]
    return(g)


def checkGradient(x, f, df, epsilon=0.001):
    return (npl.norm(df(x) - approximateGradient(x, f, epsilon)) / (npl.norm(df(x)) + 1)) > epsilon


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
    return npl.norm(hf(x) - approximateHessian(x, f, epsilon)) * (npl.norm(hf(x)) + 1) > epsilon


def dampedNewton(
        f,
        df,
        hf,
        startAt,
        checkGradientNTimes=3,
        epsilon=(1 / (10**6)**2),
        epsilon2=(1 / float(10**6)),
        epsilon3=(1 / float(10**6)),
        beta1=0.5,
        beta2=0.5,
        delta=0.01,
        gamma=(1 / 10**4),
        sigma_0=1,
        maxit=1000,
        verbose=True):

    optimumNow = startAt
    if verbose:
        print("Starting at: " + str(startAt))
        print("epsilon: " + str(epsilon))
        print("epsilon2: " + str(epsilon2))
        print("epsilon3: " + str(epsilon3))
        print(" x | f(x) | df(x) | hf(x) | simga_i | delta_i |")
    finished = False
    stepsTaken = list()
    stepsTaken.insert(0, optimumNow)
    iterations = 1

    while not finished:
            # Check gradient in first iterations
            if iterations < checkGradientNTimes:
                if not checkGradient(optimumNow, f, df):
                    if verbose:
                        print("Gradient-check failed in interation: " + str(iterations))
                    return([])
                if not checkHessian(optimumNow, f, hf):
                    if verbose:
                        print("Hessian-check failed in interation: " + str(iterations))
                    return([])

            hessian = hf(optimumNow)
            if len(hessian.shape) == 1:
                d = -1 * 1/(hessian) * optimumNow * df(optimumNow)
            else:
                id = np.eye(hessian.shape[0], hessian.shape[1])
                inv = npl.solve(hessian, id)
                d = -1 * inv.dot(optimumNow) * df(optimumNow)

            sigma_i = armijoStepwidth(
                x=optimumNow,
                f=f,
                df=df,
                d=d,
                beta1=beta1,
                beta2=beta2,
                delta=delta,
                gamma=gamma,
                sigma_0=sigma_0,
                verbose=True
            )
            print("Optimum now)",optimumNow)
            print("Optimum now)", type(optimumNow))
            print("Sigma*d: ",sigma_i*d)
            print("Sigma*d: ", type(sigma_i * d))
            print("Sum",optimumNow.__add__(sigma_i * d))


            optimumNext = optimumNow + sigma_i * d

            #optimumNext = optimumNow +  d

            stepsTaken.insert(0, optimumNext)

            if verbose:
                print(str(optimumNow) + " | " + str(f(optimumNow)) + " | " + str(df(optimumNow)) + " | " + str(hf(optimumNow)) + " | " + str(sigma_i) + " | " + str(d))

            # if orCriterias(f, df, optimumNow, optimumNext, epsilon, epsilon2, epsilon3):
            #     if verbose:
            #         print("Gradient descent with Armijo Stepwidth took: " +
            #               str(iterations) + " iterations.")
            #         print("Gradient descent with Armijo Stepwidth found: " +
            #               str(optimumNext) + ".")
            #     return(stepsTaken)
            if iterations > maxit:
                if verbose:
                    print("Gradient descent with Armijo Stepwidth took: " +
                          str(iterations) + " iterations.")
                    print("Gradient descent with Armijo Stepwidth found: " +
                          str(optimumNext) + ".")
                return(stepsTaken)

            iterations += 1
            optimumNow=optimumNext

