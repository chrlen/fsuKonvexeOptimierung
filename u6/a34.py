import descent as dc
import functions as fct
import numpy as np

regmin = dc.dampedNewton(fct.regressionFct,fct.regressionGradient,fct.regressionHessian,np.array([1,1]))