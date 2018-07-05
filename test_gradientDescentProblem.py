from unittest import TestCase
from descent2 import GradientDescentProblem
from descent2 import optimize
import functions as fct
import numpy as np
class TestGradientDescentProblem(TestCase):

    def testOtimization(self):
        prob = GradientDescentProblem(
            fct.simpleQuadraticFct,
        fct.simpleQuadraticGradient,
        np.array([5,-5]))
        optimize(prob)
        print(prob.path)
        if not np.allclose(np.array([1,1]),prob.getMin()):
            self.fail(str(prob.getMin()))
    pass
