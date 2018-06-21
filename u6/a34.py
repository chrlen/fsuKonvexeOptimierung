import descent as dc
import functions as fct
import numpy as np

print("-------- Sqrt-Function --------")
sqrtMin = dc.dampedNewton(fct.squarerootExample,fct.squarerootExampleGradient,fct.squarerootExampleHessian,np.array([1]))

#print("-------- Regression --------")
#regmin = dc.dampedNewton(fct.regressionFct,fct.regressionGradient,fct.regressionHessian,np.array([1,1]))

#print("-------- Robust - Regression --------")
#phlmin = dc.dampedNewton(fct.paritalPhl,fct.paritaldPhl,fct.pseudoHuberHessian,np.array([1,1]))
