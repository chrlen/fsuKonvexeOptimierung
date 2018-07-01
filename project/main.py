from sklearn import datasets
from sklearn.model_selection import train_test_split

import descent2 as dc
import functions as fct
import numpy as np
C=10

iris = datasets.load_iris()
columnNames = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
#To seperate setosa from versicolor + virginica set both to 1
iris.target[iris.target==2] = 1
subsets = [[x,y] for x in range(0,iris.data.shape[1]) for y in range(x,iris.data.shape[1])]
for subset in subsets:
    print("-------- " + columnNames[subset[0]] + " + " + columnNames[subset[1]] + " --------")
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data[:,subset], iris.target, test_size=0.4, random_state=0)
    #Train L1 SVM
    #Train L2 SVM
    #Train Logistic SVM with Newton-Method
    logProblem = dc.NewtonOptimizationProblem(
        {'f':fct.lsvm,
         'df':fct.dlsvm,
         'hf:'fct.hlsvm,
         'x_0'np.array([0,0])
         }
    )
    #Calculate Error

#print(fct.lsvm(X_train,X_test,np.array([1,1,1,1]),1))

