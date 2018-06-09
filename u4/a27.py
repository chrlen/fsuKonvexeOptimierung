import descent as dc
import pandas as pd
import numpy as np

data = pd.read_csv("Advertising.csv")

eta = data['Sales']
xi = data['TV']

Q = np.vstack([[sum(xi**2),sum(xi)],[
                  sum(xi),xi.shape[0]]])

q = np.array([-1 * sum(xi * eta),- 1 * sum(eta)])
c = sum(eta**2)

startAt = np.array([6,-6])

print("The Function is:")
print("Q = " + str(Q))
print("Eig(Q): " + str(np.linalg.eigvals(Q)))
print("q = " + str(q))
print("c = " + str(c))
print("Starting at :" + str(startAt))
print("")


simpleGradientOptimum = dc.simpleGradientDescent(Q, q, c, startAt, 0.00001)
dc.plotFunction(Q,q,c,-6,8,-6,8,simpleGradientOptimum,"Simple gradient descent path")
print("Das mit dem einfachen Gradientenverfahren ermittelte Minumum liegt bei: " + str(simpleGradientOptimum[0]))
print("Das Verfahren hat " + str(len(simpleGradientOptimum)) + " Schritte durchlaufen.")

conjugateGradientOptimum = dc.conjugateGradientDescent(Q, q, c, startAt, 0.00001)
dc.plotFunction(Q,q,c,-6,8,-6,8,conjugateGradientOptimum,"Conjugate gradient descent path")
print("Das mit dem Konjugierte-Gradientenverfahren ermittelte Minumum liegt bei: " + str(conjugateGradientOptimum[0]))
print("Das Verfahren hat " + str(len(conjugateGradientOptimum)) + " Schritte durchlaufen.")

