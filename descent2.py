import abc
import numpy as np
import numpy.linalg as npl
import helpers as hlp

class OptimizationProblem(object):
    def __init__(self,function,gradient,startPoint, **kwargs):
        self.path = list()
        self.function = function
        self.gradient = gradient
        self.x_thisTime = startPoint
        self.path.insert(0,self.x_thisTime)
        self.sigma = kwargs.get('sigma_0',1)

        #Gradient-Check
        self.gradientCheckEpsilon = kwargs.get('gradientCheckEspilon',0.001)
        self.hessianCheckEpsilon = kwargs.get('hessianCheckEspilon',0.001)

        #Convergence-Check
        self.convergenceCheckEpsilon = kwargs.get('convergenceCheckEpsilon',0.0000001**2)        
        self.convergenceCheckEpsilon = kwargs.get('convergenceCheckEpsilon',0.0000001)
        self.convergenceCheckEpsilon = kwargs.get('convergenceCheckEpsilon',0.0000001)

        #Armijo-Stepwidth
        self.armijoBeta1 = kwargs.get("armijoBeta1",0.5)
        self.armijoBeta2 = kwargs.get("armijoBeta2",0.5)
        self.armijoDelta = kwargs.get("armijoDelta",0.01)
        self.armijoGamma = kwargs.get("armijoGamma",0.00001)
        self.armijoSigma_0 = kwargs.get("armijoSigma_0",1)
        self.armijoVerbose = kwargs.get("armijoVerbose",False)

        self.verbose = kwargs.get("verbose",True)
        self.direction = None
        self.x_nextTime = None

    @abc.abstractmethod
    def iterate(self):
         """Implement this to calculate a step of the optimization"""

    def prepareNext(self):
        self.path.insert(0,self.x_thisTime)
        self.x_nextTime = self.x_thisTime

    def printTableHead(self):
        print("--------------------------------------------------------")
        print("| Iteration | Step Length | FunValue    | GradientNorm |")

    @abc.abstractmethod
    def printState(self):
        print("--------------------------------------------------------")
        print('| {0}         |{1:.5E}  | {2:.5E} | {3:.5E}  |'.format(self.numOfIterations(),
                             self.sigma,
                             self.function(self.path[0]),
                             npl.norm(self.path[0],ord=np.inf)))

    @abc.abstractmethod
    def printResult(self):
        print("--------------------------------------------------------")
        """Print Head of Table"""

    def checkGradient(self):
        hlp.checkGradient(
            self.x_thisTime,
            self.function,
            self.gradient,
            self.gradientCheckEpsilon)
        hlp.checkHessian(self.x_thisTime,
            self.function,
            self.hessian,
            self.hessianCheckEpsilon)

    def isDone(self):
        return hlp.orCriterias(
            self.function, 
            self.gradient, 
            self.x_thisTime, 
            self.x_nextTime, 
            self.criteriaCheckEpsilon, 
            self.criteriaCheckEpsilon2, 
            self.criteriaCheckEpsilon3)

    def numOfIterations(self):
        return len(self.path)

class NewtonOptimizationProblem(OptimizationProblem):
    def __init__(self,function,gradient,hessian,startpPoint, **kwargs):
        super().__init__(function,gradient,startpPoint,**kwargs)
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
            verbose=self.armijoVerbose)
        self.x_nextTime = self.x_thisTime + self.sigma * self.direction

    def printTableHead(self):
        print("Damped Newton descent with Startpoint: " , self.x_thisTime)
        super().printTableHead()

    def printResult(self):
        super().printResult()
        print("Damped Newton descent found the optimum: ", self.x_nextTime)

def optimize(op, maxit=1000, gradientCheck=3, verbose=True):
    iterations = 0
    if verbose:
        op.printTableHead()
    while op.numOfIterations() < maxit:
        if op.numOfIterations() < gradientCheck:
            op.checkGradient()
        op.iterate()
        if verbose:
            op.printState()
        if op.isDone:
            if verbose:
                op.printResult()
            return
        op.prepareNext()
    if verbose:
        print("Maximum iterations reached")
        op.printResult()
