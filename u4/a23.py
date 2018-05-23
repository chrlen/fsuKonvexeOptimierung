import descent as dc
import numpy as np

Q = np.diag([4, 2])
q = np.array([-4, -2])
c = 3

print("The Function is:")
print("Q = " + str(Q))
print("q = " + str(q))
print("c = " + str(c))
print(dc.evalQuadraticForm(np.array([1,1]),Q,q,c))
print("")

startAt = np.array([5, -5])

simpleGradientOptimum = dc.simpleGradientDescentQU(Q, q, c, startAt, 0.00001)
dc.plotFunction(Q,q,c,-6,8,-6,8,simpleGradientOptimum,"Simple gradient descent path")
