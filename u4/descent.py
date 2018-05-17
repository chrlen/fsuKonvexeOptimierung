import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

def crit1(fx_this,fx_next,epsilon):
    return(fx_this -fx_next <= epsilon * np.max(1,np.abs(fx_this)))

def crit2(fx_this,fx_next,epsilon2):
    return(fx_next - fx_this <= epsilon2 * np.max(1,npl.norm(fx_this)))

#def crit3(fx_this,fx_next,epsilon2):
#    return(fx_next - fx_this <= epsilon2 * np.max(1,np.abs(fx_this)))

def simpleGradientDescentQU(Q,q,c,startAt,epsilon,maxit=10000,plot=True,verbose=True):
	optimumNow = startAt
	stepsTaken = list()
	stepsTaken.append(optimumNow)

	finished = False
	iterations = 0

	while not finished:
		print(iterations)
		iterations = iterations + 1



		if(iterations > maxit): 
			finished = True
			print("Maximum number of iterations reached")
	return([thisTime,stepsTaken])

def conjugateGradientDescentQU(Q,q,c,startAt,epsilon,maxit=10000,plot=True,verbose=True):
    print("done!")    
    
Q = np.diag([2,2,1,1,0.5])
q = np.array([-4,-4,-2,-2,-1])
c = np.array([6.5])
print("The Function is:")
print("Q = " + str(Q))
print("q = " + str(q))
print("c = " + str(c))

startAt = np.array([0,0])
print("Starting at: " +str(startAt))

simpleGradientOptimum = simpleGradientDescentQU(Q,q,c,startAt,0.00001)
conjugateGradientOptimum = conjugateGradientDescentQU(Q,q,c,startAt,0.00001)





