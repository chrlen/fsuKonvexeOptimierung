import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

def evalQuadraticForm(x,Q, q, c):
    return 0.5 *  x.dot(Q).dot(x) + q.dot(x) + c

def evalFirstOrderGradientOfQuadraticForm(x, Q, q):
    return Q.dot(x) + q

def evalField(Q,q,c,X,Y):
    x_range = range(0,X.shape[0])
    y_range = range(0,X.shape[1])
    Z = np.zeros(X.shape)
    
    for x in x_range:
        for y in y_range:
            v = np.array([X[y,x],Y[y,x]])
            Z[y,x] = evalQuadraticForm(v,Q,q,c)

    return Z
def plotFunction(Q,q,c,
                 xMin,xMax,
                 yMin,yMax,
                 path,
                 title):
    x = np.arange(xMin,xMax,0.1)
    y = np.arange(yMin,yMax,0.1)
    X, Y = np.meshgrid(x, y)
    Z = evalField(Q,q,c,X,Y)
    
    plt.figure()
    CS = plt.contourf(X, Y, Z)
    for i in range(1,len(path)):
        plt.plot([path[i][0],path[i-1][0]],
                  [path[i][1],path[i-1][1]],
                  color = 'white',marker ='o')
    plt.colorbar()
    plt.title(title)
    plt.show()



def crit1(fx_thisTime, fx_nextTime, epsilon):
    return(fx_thisTime - fx_nextTime <= epsilon * max(1, abs(fx_thisTime)))


def crit2(x_thisTime, x_nextTime, epsilon2):
    return(npl.norm(x_nextTime - x_thisTime) <= epsilon2 * max(1, npl.norm(x_thisTime)))


def crit3(fx_thisTime, dfx_thisTime, epsilon3):
    return(npl.norm(dfx_thisTime) <= epsilon3 * max(1, np.abs(fx_thisTime)))

#def orCriterias(Q, q, c, x_thisTime, x_nextTime, epsilon, epsilon2, epsilon3):
#    return crit1(fx_thisTime, fx_nextTime, epsilon) or crit2(x_thisTime, x_nextTime, epsilon2) or crit3(fx_thisTime, dfx_thisTime, epsilon3)

def orCriterias(Q, q, c, x_thisTime, x_nextTime, epsilon, epsilon2, epsilon3):
    if(crit2(x_thisTime, x_nextTime, epsilon2)):
        return True
    else:
        fx_thisTime = evalQuadraticForm(x_thisTime, Q, q, c)
        fx_nextTime = evalQuadraticForm(x_nextTime, Q, q, c)
        if crit1(fx_thisTime, fx_nextTime, epsilon):
            return True
        else:
            dfx_thisTime = evalFirstOrderGradientOfQuadraticForm(x_thisTime, Q, q)
            if crit3(fx_thisTime, dfx_thisTime, epsilon3):
                return True
            else:
                return False


def simpleGradientDescent(Q, q, c, startAt,
                            epsilon=(1/(10**6)**2 ),
                            epsilon2=1/float(10**6),
                            epsilon3=1/float(10**6),
                            maxit=1000,
                            verbose=True
                            ):
    optimumNow = startAt
    if verbose:
        print("Starting at: " + str(startAt))
        print("epsilon: " + str(epsilon))
        print("epsilon2: " + str(epsilon2))
        print("epsilon3: " + str(epsilon3))
    stepsTaken = list()


    finished = False
    iterations = 0

    while not finished:     
        d = -1 * evalFirstOrderGradientOfQuadraticForm(optimumNow,Q,q)

        z = d.dot(d) 
        n = d.dot(Q).dot(d)
        optimalStep = z/float(n)
        optimumNext = optimumNow + optimalStep * d
        
        stepsTaken.insert(0,optimumNow)

        if verbose:
            print("---- i =  " + str(iterations) + " ----")
            print("x_i: " + str(optimumNow))
            print("f(x_i): " + str(evalQuadraticForm(optimumNow,Q,q,c)))
            print("-df(x_i): " + str(-1 * evalFirstOrderGradientOfQuadraticForm(optimumNow,Q,q)))
            print("d: " + str(d))
            print("Optimal step: " + str(optimalStep))
            print("")

        if iterations == maxit:
            if verbose:
                print("Maximum number of iterations reached: " + str(maxit))
            return(stepsTaken)
        if orCriterias(Q, q, c, optimumNow, optimumNext, epsilon, epsilon2, epsilon3):
            return(stepsTaken)

        iterations += 1
        optimumNow = optimumNext


def conjugateGradientDescent(Q, q, c, startAt,
                            epsilon=(1/(10**6)**2 ),
                            epsilon2=(1/float(10**6)),
                            epsilon3=(1/float(10**6)),
                            maxit=3,
                            verbose=True
                            ):
    optimumNow = startAt
    print(epsilon)
    dNow = -1 * evalFirstOrderGradientOfQuadraticForm(optimumNow,Q,q)
    print("dNow: " + str(dNow))
    if verbose:
        print("Starting at: " + str(startAt))
        print("epsilon: " + str(epsilon))
        print("epsilon2: " + str(epsilon2))
        print("epsilon3: " + str(epsilon3))
    stepsTaken = list()


    finished = False
    iterations = 0

    while not finished:     
        #Berechne optimale Schrittweite
        sw_z = (-1 * dNow).dot(dNow)
        #print(sw_z) 
        sw_n = dNow.dot(Q).dot(dNow)
        #print(sw_n) 
        optimalStep = sw_z/float(sw_n)

        #Berechne naechstes Optimum
        optimumNext = optimumNow + optimalStep * dNow
        #print(optimumNext)
        #Berechne Suchrichtung
        sr_z = evalFirstOrderGradientOfQuadraticForm(optimumNext,Q,q)
        #print(sr_z)
        sr_n = evalFirstOrderGradientOfQuadraticForm(optimumNow,Q,q)
        #print(sr_n)
        beta = npl.norm(sr_z)**2 / npl.norm(sr_n)**2

        dNext = (-1 * sr_z) + beta * dNow

        stepsTaken.insert(0,optimumNow)

        if verbose:
            print("---- i =  " + str(iterations) + " ----")
            print("x_i: " + str(optimumNow))
            print("f(x_i): " + str(evalQuadraticForm(optimumNow,Q,q,c)))
            print("-df(x_i): " + str(-1 * evalFirstOrderGradientOfQuadraticForm(optimumNow,Q,q)))
            print("f(x_i+1): " + str(optimumNext))
            print("dNow: " + str(dNow))
            print("beta: " + str(beta))
            print("Optimal step: " + str(optimalStep))
            print("")

        if iterations == maxit:
            if verbose:
                print("Maximum number of iterations reached: " + str(maxit))
            return(stepsTaken)
        #if orCriterias(Q, q, c, optimumNow, optimumNext, epsilon, epsilon2, epsilon3):
        #    return(stepsTaken)

        iterations += 1
        optimumNow = optimumNext
        dNow = dNext



Q = np.diag([4, 2])
q = np.array([-4, -2])
c = 3


print("The Function is:")
print("Q = " + str(Q))
print("q = " + str(q))
print("c = " + str(c))
print(evalQuadraticForm(np.array([1,1]),Q,q,c))
print("")

#startAt = np.repeat(10,5)
startAt = np.array([5, -5])

#simpleGradientOptimum = simpleGradientDescent(Q, q, c, startAt, 0.00001)
#plotFunction(Q,q,c,-6,8,-6,8,simpleGradientOptimum,"Simple gradient descent path")

conjugateGradientOptimum = conjugateGradientDescent(Q, q, c, startAt, 0.00001)
#plotFunction(Q,q,c,-6,8,-6,8,conjugateGradientOptimum,"Simple gradient descent path")
