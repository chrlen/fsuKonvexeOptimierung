import descent2 as dc
import functions as fct

import numpy as np

regressionProblem = dc.NewtonOptimizationProblem(
	fct.regressionFct,
	fct.regressionGradient,
	fct.regressionHessian,
	np.array([6,-6]))

dc.optimize(regressionProblem)



#print(regressionProblem.isDone)