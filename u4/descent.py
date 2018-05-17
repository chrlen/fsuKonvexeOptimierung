import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

def crit1(fx_this,fx_next,epsilon):
    return(fx_this -fx_next <= epsilon * np.max(1,np.abs(fx_this)))

def crit2(fx_this,fx_next,epsilon2):
    return(fx_next - fx_this <= epsilon2 * np.max(1,npl.norm(fx_this)))

#def crit3(fx_this,fx_next,epsilon2):
#    return(fx_next - fx_this <= epsilon2 * np.max(1,np.abs(fx_this)))

def simpleGradientDescent(Q,q,c,epsilon,plot=TRUE,verbose=TRUE):
    print("done!")

def conjugateGradientDescent(Q,q,c,epsilon,plot=TRUE,verbose=TRUE):
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

simpleGradientOptimum = simpleGradientDescent(Q,q,c,0.00001)
conjugateGradientOptimum = conjugateGradientDescent(Q,q,c,0.00001)





