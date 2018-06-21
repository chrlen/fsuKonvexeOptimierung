import descent as dc
import functions as fct
import numpy as np

#print("-------- Sqrt-Function --------")
#sqrtMin = dc.dampedNewton(fct.squarerootExample,fct.squarerootExampleGradient,fct.squarerootExampleHessian,np.array([1]))

print("-------- Regression --------")
regmin = dc.dampedNewton(fct.regressionFct,fct.regressionGradient,fct.regressionHessian,np.array([6,-6]),delta = 0.0001,maxit=1000)
#dc.plotFunction(fct.regressionFct,50,-50,50,-50,regmin,"Regression")
dc.plotConvergence(fct.regressionFct,regmin,"bla")

#print("-------- Robust - Regression --------")
#phlmin = dc.dampedNewton(fct.paritalPhl,fct.paritaldPhl,fct.pseudoHuberHessian,np.array([1,1]))
