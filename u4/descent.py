import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import math
import matplotlib.pyplot as plt

def evalQuadraticForm(x,Q,q,c):
    return x.T.dot(Q).dot(x)+q.T.dot(x) + c  

def evalFirstOrderGradientOfQuadraticForm(x,Q,q):
    return Q.dot(x) + q

def crit1(fx_thisTime, fx_nextTime, epsilon):
    return(fx_thisTime - fx_nextTime <= epsilon * max(1, abs(fx_thisTime)))

def crit2(x_thisTime, x_nextTime, epsilon2):
    return(npl.norm(x_nextTime - x_thisTime) <= epsilon2 * max(1, npl.norm(x_thisTime)))

def crit3(fx_thisTime, dfx_thisTime,epsilon3):
    return(npl.norm(dfx_thisTime) <= epsilon3 * max(1,np.abs(fx_thisTime)))

def orCriterias(Q,q,c,x_thisTime, x_nextTime, epsilon, epsilon2, epsilon3):
    #Überprüfe erst das zweite Kriterium, da dieses am wenigsten Rechenaufwand benötigt
    if(crit2(x_thisTime,x_nextTime,epsilon2)):
        print("crit2")
        return True
    else:
        fx_thisTime = evalQuadraticForm(x_thisTime,Q,q,c)
        #print(fx_thisTime)
        fx_nextTime = evalQuadraticForm(x_nextTime,Q,q,c)
        #print(fx_nextTime)
        if crit1(fx_thisTime,fx_nextTime,epsilon):
            print("crit1")
            return True
        else:
            dfx_thisTime = evalFirstOrderGradientOfQuadraticForm(x_thisTime,Q,q)
            if crit3(fx_thisTime,dfx_thisTime,epsilon3):
                print("crit3")
                return True
            else:
                return False

def simpleGradientDescentQU(Q, q, c, startAt, epsilon=(1 / (10 ^ 5)),epsilon2 = math.sqrt(1 / (10 ^ 5)),epsilon3=math.sqrt(1 / (10 ^ 5)), maxit=1000, verbose=True):
    optimumNow = startAt
    stepsTaken = np.empty(1)
    stepsTaken = np.append(stepsTaken,optimumNow,axis = 0)

    finished = False
    iterations = 0

    while not finished:
        print(iterations)
        d = -1 * Q.dot(optimumNow) + q
        optimalStep = d.T.dot(d) / d.T.dot(Q).dot(d)
        optimumNext = optimumNow + optimalStep * d
        stepsTaken = np.append(stepsTaken,optimumNext,axis = 0)

        if iterations > maxit:
            return([optimumNow, stepsTaken])
        if orCriterias(Q,q,c,optimumNow,optimumNext,epsilon,epsilon2,epsilon3):
            return([optimumNow, stepsTaken])
        
        iterations = iterations + 1
        optimumNow = optimumNext


def conjugateGradientDescentQU(Q, q, c, startAt, epsilon, maxit=10000, plot=True, verbose=True):
    print("done!")

Q = np.diag([2, 2, 1, 1, 0.5])
q = np.array([-4, -4, -2, -2, -1])
c = 6.5
print("The Function is:")
print("Q = " + str(Q))
print("q = " + str(q))
print("c = " + str(c))

#startAt = np.repeat(10,5)
startAt = npr.random(5) * 1000
print("Starting at: " + str(startAt))

simpleGradientOptimum = simpleGradientDescentQU(Q, q, c, startAt, 0.00001)
print(simpleGradientOptimum)
#conjugateGradientOptimum = conjugateGradientDescentQU(
#    Q, q, c, startAt, 0.00001)
