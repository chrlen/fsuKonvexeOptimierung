import abc
import helpers

class OptimizationProblem(object):
    @abc.abstractmethod
    def __init__(self, **kwargs):
        self.isDone = False
        self.x_nextTime = None
        self.path = list()

        """nothin"""
    @abc.abstractmethod
    def iterate(self):
        """nothin"""

    @abc.abstractmethod
    def printTableHead(self):
        print( "Iteration FunEval Step Length FunValue GradientNorm")

    @abc.abstractmethod
    def printState(self):
        """Print Head of Table"""
    @abc.abstractmethod
    def printResult(self):
        """Print Head of Table"""

class SecondOrderOptimizationProblem(OptimizationProblem):
    def __init__(self, **kwargs):
        super().__init__()
        self.function = kwargs["f"]
        self.gradient = kwargs["df"]
        self.hessian = kwargs["hf"]
        self.x_thisTime = kwargs["x"]
        self.sigma = None
        self.d = None

    def iterate():
    	d = -1 * self.df(optimumNow)
    	sigma_i = armijoStepwidth(
            x=optimumNow,
            f=f,
            df=df,
            d=d,
            beta1=beta1,
            beta2=beta2,
            delta=delta,
            gamma=gamma,
            sigma_0=sigma_0,
            verbose=False)

def optimize(optimizationProblem, maxit=1000, verbose=True):
    def printResult(optimizationProblem):
        print("The Minimum: ", optimizationProblem.path[0], " was found after ", len(
            optimizationProblem.path)," iterations!")
    iterations = 0
    if verbose:
        OptimizationProblem.printTable()
    while iterations < maxit:
        optimizationProblem.iterate()
        if verbose:
            optimizationProblem.printTable()
        if optimizationProblem.isDone:
            if verbose:
                printResult(optimizationProblem)
            return
    if verbose:
        print("Maximum iterations reached")
        optimizationProblem.printResult()
