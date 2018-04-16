import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def getGrid(x1_lower_limit,x1_upper_limit,x1_stepwidth,
				x2_lower_limit,x2_upper_limit,x2_stepwidth,
				func):
	X = np.arange(x1_lower_limit,x1_upper_limit,x1_stepwidth)
	Y = np.arange(x2_lower_limit,x2_upper_limit,x2_stepwidth)
	Z = np.zeros([len(X),len(Y)])

	X, Y = np.meshgrid(X, Y)

	Z = np.array([np.array([func(x_1,x_2) for x_1 in X]) for x_2 in Y])
	print(Z.shape)

	return([X,Y,Z])
	
def plotSurface(x1_lower_limit,x1_upper_limit,x1_stepwidth,
				x2_lower_limit,x2_upper_limit,x2_stepwidth,
				function,
				path,
				name):
	X,Y,Z = getGrid(x1_lower_limit,x1_upper_limit,x1_stepwidth,
				x2_lower_limit,x2_upper_limit,x2_stepwidth,
				function)
	print(Z)
	#fig = plt.figure()
	#ax = fig.gca(projection='3d')
	#surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
	plt.savefig(path + name + ".png")

def plotContour(x1_lower_limit,x1_upper_limit,x1_stepwidth,
				x2_lower_limit,x2_upper_limit,x2_stepwidth,
				function,
				path,
				name):
	print("plot it!")






#def plotSurface(dataframe,name,column,subplot):
#    subset = list(dataframe[['Dim','Sparsity',column]].groupby('Dim'))
#    dims = list(map(lambda x: int(x[0]), subset))
#    sparsities = list(map(lambda x: x[1]['Sparsity'] ,subset))[0]
#    colnames = list(map( lambda x :str(x),sparsities))
#    result = pd.DataFrame(columns= colnames)
#    for i in subset:
#        row = pd.DataFrame(np.matrix(i[1][column]), columns=colnames)
#        result = result.append(row, ignore_index=True)
#
#
#    fig = plt.figure(subplot)
#    plt.title('Bla')
#    ax = fig.gca(projection='3d')
#
#    Z = result.as_matrix()
#
#    Y = np.array(list(range(0,result.shape[0]))) # Y is Dimensions
#    ax.set_ylabel('Dimensions')
#    ax.set_yticklabels(dims)
#
#    X = np.array(list(range(0,result.shape[1]))) # X is Sparsity
#    ax.set_xlabel('Sparsity')
#    ax.set_xticklabels(sparsities)
#    #ax.set_xticks(sparsities)
#    X, Y = np.meshgrid(X, Y)
#
#    surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
#
#
#    # # Customize the z axis.
#    # #ax.set_zlim(-1.01, 1.01)
#    ax.zaxis.set_major_locator(LinearLocator(10))
#    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
#    ax.yaxis.set_major_locator(LinearLocator(10))
#
#    # # Add a color bar which maps values to colors.
#    fig.colorbar(surf, shrink=0.5, aspect=5)
#    plt.savefig('intel/'+ name +str(column)+'.png'   ) 