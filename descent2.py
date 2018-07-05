import abc
import numpy as np
import numpy.linalg as npl
import helpers as hlp


class OptimizationProblem(object):
    def __init__(self, function, gradient, startPoint, **kwargs):
        self.path = list()
        self.function = function
        self.gradient = gradient
        self.x_thisTime = startPoint
        self.path.insert(0, self.x_thisTime)
        self.sigma = kwargs.get('sigma_0', 1)

        # Gradient-Check
        self.gradientCheckEpsilon =         kwargs.get('gradientCheckEspilon', 0.01)
        self.hessianCheckEpsilon =          kwargs.get('hessianCheckEpsilon', 0.01)
        self.gradientApproximationEpsilon = kwargs.get('gradientApproximationEspilon', 0.00001)
        self.hessianApproximationEpsilon =  kwargs.get('hessianApproximationEpsilon', 0.00001)

        # Convergence-Check
        self.convergenceCheckEpsilon = kwargs.get(
            'convergenceCheckEpsilon', 0.0000001**2)
        self.convergenceCheckEpsilon2 = kwargs.get(
            'convergenceCheckEpsilon2', 0.0000001)
        self.convergenceCheckEpsilon3 = kwargs.get(
            'convergenceCheckEpsilon3', 0.0000001)

        # Armijo-Stepwidth
        self.armijoBeta1 = kwargs.get("armijoBeta1", 0.5)
        self.armijoBeta2 = kwargs.get("armijoBeta2", 0.5)
        self.armijoDelta = kwargs.get("armijoDelta", 0.01)
        self.armijoGamma = kwargs.get("armijoGamma", 0.00001)
        self.armijoSigma_0 = kwargs.get("armijoSigma_0", 1)
        self.armijoVerbose = kwargs.get("armijoVerbose", False)
        self.armijoMaxit = kwargs.get("armijoMaxit", 10)

        self.verbose = kwargs.get("verbose", True)
        self.maxit = kwargs.get("maxit", 100)
        self.direction = None
        self.x_nextTime = None

    @abc.abstractmethod
    def iterate(self):
        """Implement this to calculate a step of the optimization"""

    def prepareNext(self):
        self.path = [self.x_nextTime] + self.path
        self.x_thisTime = self.x_nextTime

    def printTableHead(self):
        print("--------------------------------------------------------")
        print("| Iteration | Step Length | FunValue    | GradientNorm |")

    @abc.abstractmethod
    def printState(self):
        print("--------------------------------------------------------")
        print('| {0}         |{1:.5E}  | {2:.5E} | {3:.5E}  |'.format(self.numOfIterations(),
                                                                      self.sigma,
                                                                      self.function(
                                                                          self.path[0]),
                                                                      npl.norm(self.path[0], ord=np.inf)))

    @abc.abstractmethod
    def printResult(self):
        print("--------------------------------------------------------")
        """Print Head of Table"""

    def checkGradient(self):
        if hasattr(self, "gradient"):
            hlp.checkGradient(
                self.x_thisTime,
                self.function,
                self.gradient,
                self.gradientCheckEpsilon,
                self.gradientApproximationEpsilon)
        if hasattr(self, "hessian"):
            hlp.checkHessian(self.x_thisTime,
                             self.function,
                             self.hessian,
                             self.hessianCheckEpsilon,
                             self.hessianApproximationEpsilon)

    def isDone(self):
        return hlp.orCriterias(
            self.function,
            self.gradient,
            self.x_thisTime,
            self.x_nextTime,
            self.convergenceCheckEpsilon,
            self.convergenceCheckEpsilon2,
            self.convergenceCheckEpsilon3)

    def numOfIterations(self):
        return len(self.path)

    def getMin(self):
        return self.path[0]


class NewtonOptimizationProblem(OptimizationProblem):
    def __init__(self, function, gradient, hessian, startpPoint, **kwargs):
        super().__init__(function, gradient, startpPoint, **kwargs)
        self.hessian = hessian

    def iterate(self):
        h = self.hessian(self.x_thisTime)
        id = np.eye(h.shape[0], h.shape[1])
        inv = npl.solve(h, id)
        self.direction = -1 * inv.dot(self.gradient(self.x_thisTime))

        self.sigma = hlp.armijoStepwidth(
            x=self.x_thisTime,
            f=self.function,
            df=self.gradient,
            d=self.direction,
            beta1=self.armijoBeta1,
            beta2=self.armijoBeta2,
            delta=self.armijoDelta,
            gamma=self.armijoGamma,
            sigma_0=self.armijoSigma_0,
            verbose=self.armijoVerbose,
            maxit =self.armijoMaxit)
        self.x_nextTime = self.x_thisTime + self.sigma * self.direction

    def printTableHead(self):
        print("Damped Newton descent with Startpoint: ", self.x_thisTime)
        super().printTableHead()

    def printResult(self):
        super().printResult()
        print("Damped Newton descent found the optimum: ", self.x_nextTime)


class GradientDescentProblem(OptimizationProblem):
    def __init__(self, function, gradient, startpPoint, **kwargs):
        super().__init__(function, gradient, startpPoint, **kwargs)

    def iterate(self):
        self.direction = -1 * self.gradient(self.x_thisTime)
        self.sigma = hlp.armijoStepwidth(
            x=self.x_thisTime,
            f=self.function,
            df=self.gradient,
            d=self.direction,
            beta1=self.armijoBeta1,
            beta2=self.armijoBeta2,
            delta=self.armijoDelta,
            gamma=self.armijoGamma,
            sigma_0=self.armijoSigma_0,
            verbose=self.armijoVerbose,
            maxit =self.armijoMaxit)
        self.x_nextTime = self.x_thisTime + self.sigma * self.direction

    def printTableHead(self):
        print("Gradient descent with Startpoint: ", self.x_thisTime)
        super().printTableHead()

    def printResult(self):
        super().printResult()
        print("Gradient descent found the optimum: ", self.x_nextTime)


def optimize(op, gradientCheck=1):
    iterations = 0
    if op.verbose:
        op.printTableHead()
    while op.numOfIterations() < op.maxit:
        if op.numOfIterations() < gradientCheck:
            op.checkGradient()
        op.iterate()
        if op.verbose:
            op.printState()
        if op.isDone():
            if op.verbose:
                # op.prepareNext()
                op.printResult()

            return
        op.prepareNext()
    if op.verbose:
        print("Maximum iterations reached")
        op.printResult()
