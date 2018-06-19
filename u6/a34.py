import descent as dc
import functions as fct
import numpy as np

cg = dc.checkGradient(np.array([100,100]),fct.regressionFct,fct.regressionGradient)
ch = dc.checkHessian(np.array([100,100]),fct.regressionFct,fct.regressionHessian)
