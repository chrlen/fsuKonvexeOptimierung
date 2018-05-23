import descent as dc
import numpy as np

Q = np.diag([4, 2])
q = np.array([-4, -2])
c = 3

startAt = np.array([5, -5])

print("The Function is:")
print("Q = " + str(Q))
print("q = " + str(q))
print("c = " + str(c))
print("Starting at :" + str(startAt))
print("")

simpleGradientOptimum = dc.simpleGradientDescent(Q, q, c, startAt, 0.00001)
dc.plotFunction(Q,q,c,-6,8,-6,8,simpleGradientOptimum,"Simple gradient descent path")
print("Das mit dem einfachen Gradientenverfahren ermittelte Minumum liegt bei: " + str(simpleGradientOptimum[0]))
print("Das Verfahren hat " + str(len(simpleGradientOptimum)) + " Schritte durchlaufen.")