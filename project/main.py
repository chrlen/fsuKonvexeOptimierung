from sklearn import datasets
from sklearn.model_selection import train_test_split

import descent2 as dc
import functions as fct
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from itertools import combinations
import pandas as pd
import functions as fct
import helpers as hlp

iris = datasets.load_iris()
columnNames = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
# To seperate setosa from versicolor + virginica set both to 1
iris.target[iris.target == 1] = -1
iris.target[iris.target == 2] = -1
iris.target[iris.target == 0] = 1

subsets = combinations(range(len(columnNames)),2)

for subset in subsets:
    print("-------- " + columnNames[subset[0]] +
          " + " + columnNames[subset[1]] + " --------")

    X = iris.data[:, subset]
    Y = iris.target

    #-------- -------- Prepare Plots -------- --------
    plt.scatter(X[:,0],X[:,1],s=(Y==-1)*50,marker='^',c='g',alpha=0.5)
    plt.scatter(X[:,0],X[:,1],s=(Y==1 )*50,marker='*' ,c='r',alpha=0.5)
    plt.title(columnNames[subset[0]] + " + " + columnNames[subset[1]])
    plt.xlabel(columnNames[subset[0]])
    plt.ylabel(columnNames[subset[1]])
    
    #-------- -------- Train L1 SVM -------- --------
    partialL1_Svm = partial(fct.l1_svm, X=X, Y=Y)
    partialdL1_Svm = partial(fct.dl1_svm, X=X, Y=Y)

    partialAppDL1_Svm = partial(hlp.approximateGradient,f=partialL1_Svm,epsilon = 0.0001)

    print(partialAppDL1_Svm(np.array([1,2,0])))
    print(partialdL1_Svm(np.array([1,2,0])))

    def getID(x):
        print("x",x)
        return(np.eye(3))

    l1Problem = dc.NewtonOptimizationProblem(
        partialL1_Svm,
        partialAppDL1_Svm,
        getID,
        np.array([1, 2, 0]),
        **{"verbose": True,
            "maxit": 10000,
           "armijoMaxit": 10000,
           "armijoVerbose": False,
           'convergenceCheckEpsilon':  0.000000000003**2,
           'convergenceCheckEpsilon2': 0.000000000003,
           'convergenceCheckEpsilon3': 0.000000000003, 
           'gradientCheckEpsilon': 0.01,
           'hessianCheckEpsilon': 0.01,
           'gradientApproximationEpsilon': 0.00001,
           'hessianApproximationEpsilon': 0.00001}
    )

    dc.optimize(l1Problem)
    h1 = hlp.plot_line(X[:, 0], plt, l1Problem.getMin(), "yellow")

    # -------- -------- Train L2 SVM -------- --------
    partialL2_Svm = partial(fct.l2_svm, X=X, Y=Y)
    partialdL2_Svm = partial(fct.dl2_svm, X=X, Y=Y)
    partialhL2_Svm = partial(fct.hl2_svm, X=X, Y=Y)

    l2Problem = dc.NewtonOptimizationProblem(
        partialL2_Svm,
        partialdL2_Svm,
        partialhL2_Svm,

        np.array([1, 2, 0]),
        **{"verbose": True,
            "maxit": 10000,
           "armijoMaxit": 10000,
           "armijoVerbose": False,
           'convergenceCheckEpsilon':  0.000000000003**2,
           'convergenceCheckEpsilon2': 0.000000000003,
           'convergenceCheckEpsilon3': 0.000000000003, 
           'gradientCheckEpsilon': 0.01,
           'hessianCheckEpsilon': 0.01,
           'gradientApproximationEpsilon': 0.00001,
           'hessianApproximationEpsilon': 0.00001}
    )

    dc.optimize(l2Problem)

    #Plot descision border into scatterplot
    h2 = hlp.plot_line(X[:,0],plt,l2Problem.getMin(),"cyan")


    # -------- -------- Train L2 SVM -------- --------
    partialLog_Svm = partial(fct.log_svm, X=X, Y=Y)
    partialAppDLog_Svm = partial(hlp.approximateGradient,f=partialLog_Svm,epsilon = 0.0001)
    partialAppHLog_Svm = partial(hlp.approximateHessian, f=partialLog_Svm,epsilon = 0.0001)

    logProblem = dc.NewtonOptimizationProblem(
        partialLog_Svm,
        partialAppDLog_Svm,
        partialAppHLog_Svm,

        np.array([1, 2, 0]),
        **{"verbose": True,
            "maxit": 10000,
           "armijoMaxit": 10000,
           "armijoVerbose": False,
           'convergenceCheckEpsilon':  0.000000000003**2,
           'convergenceCheckEpsilon2': 0.000000000003,
           'convergenceCheckEpsilon3': 0.000000000003, 
           'gradientCheckEpsilon': 0.01,
           'hessianCheckEpsilon': 0.01,
           'gradientApproximationEpsilon': 0.00001,
           'hessianApproximationEpsilon': 0.00001}
    )
    dc.optimize(logProblem)

    # Plot descision border into scatterplot
    h3 = hlp.plot_line(X[:,0],plt,logProblem.getMin(),"magenta")

    plt.legend(['L1 SVM', 'L2 SVM','Logistic SVM'])
    plt.savefig("plot/"+columnNames[subset[0]] + "_" + columnNames[subset[1]]+'.png')
    plt.close()
