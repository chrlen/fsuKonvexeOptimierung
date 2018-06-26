import descent as dc
import functions as fct
import numpy as np

print("-------- Sqrt-Function --------")
sqrtMin = dc.dampedNewton(fct.squarerootExample,fct.squarerootExampleGradient,fct.squarerootExampleHessian,np.array([1]))
print(sqrtMin)


print("-------- Regression --------")
regmin = dc.dampedNewton(
    fct.regressionFct,
    fct.regressionGradient,
    fct.regressionHessian,
    np.array([6, -6]),
    checkGradientNTimes=3,
    epsilon=(0.000001**2),
    epsilon2=(0.000001),
    epsilon3=(0.000001),
    beta1=0.5,
    beta2=0.5,
    delta=0.01,
    gamma=(1 / 10 ** 4),
    sigma_0=1,
    maxit=400,
    maxitArmijo=4,
    verbose=True)
print(regmin)
dc.plotFunction(fct.regressionFct,-10,10,-10,10,regmin,"Linear Regression with Damped Newton")
dc.plotConvergence(fct.regressionFct, regmin, "Linear Regression, convergence")

print("-------- Robust - Regression --------")
phlmin = dc.dampedNewton(fct.paritalPhl,
                         fct.paritaldPhl,
                         fct.pseudoHuberHessian,
                         np.array([1, 1]),
                         checkGradientNTimes=3,
                         epsilon=(1 / (10 ** 6) ** 2),
                         epsilon2=(1 / float(10 ** 6)),
                         epsilon3=(1 / float(10 ** 6)),
                         beta1=0.5,
                         beta2=0.5,
                         delta=0.01,
                         gamma=(1 / 10 ** 4),
                         sigma_0=1,
                         maxit=1000,
                         maxitArmijo=30,
                         verbose=True)
dc.plotFunction(fct.paritalPhl,-10,10,-10,10,phlmin,"Robust Regression with Damped Newton")
dc.plotConvergence(fct.paritalPhl, phlmin, "Robust Regression, convergence")
