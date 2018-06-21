import descent2 as dc
import functions as fct

import numpy as np

regressionProblem = dc.SecondOrderOptimizationProblem(**{
	'f':fct.regressionFct,
	'df':fct.regressionGradient,
	'hf':fct.regressionHessian,
	'x':np.array([6,-6])})
	
print(regressionProblem.isDone)