import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

colormap='viridis'

def getGrid(x1_lower_limit,x1_upper_limit,x1_stepwidth,
				x2_lower_limit,x2_upper_limit,x2_stepwidth,
				func):
	X = np.arange(x1_lower_limit,x1_upper_limit,x1_stepwidth)
	Y = np.arange(x2_lower_limit,x2_upper_limit,x2_stepwidth)
	X, Y = np.meshgrid(X, Y)
	Z = func([X,Y])
	return([X,Y,Z])
	
def plotSurface(x1_lower_limit,x1_upper_limit,x1_stepwidth,
				x2_lower_limit,x2_upper_limit,x2_stepwidth,
				function,
				path,
				name):
	X,Y,Z = getGrid(x1_lower_limit,x1_upper_limit,x1_stepwidth,
				x2_lower_limit,x2_upper_limit,x2_stepwidth,
				function)
	#print(Z)
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(X, Y, Z, cmap=colormap, linewidth=0, antialiased=False)
	ax.clabel(surf, fontsize=9, inline=1)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	plt.title(name + " surface")
	plt.savefig(path + name + "_surface.png")

def plotContour(x1_lower_limit,x1_upper_limit,x1_stepwidth,
				x2_lower_limit,x2_upper_limit,x2_stepwidth,
				function,
				path,
				name):
	X,Y,Z = getGrid(x1_lower_limit,x1_upper_limit,x1_stepwidth,
			x2_lower_limit,x2_upper_limit,x2_stepwidth,
			function)
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	cset = ax.contour(X, Y, Z, extend3d=False, cmap=colormap,stride=1)
	ax.clabel(cset, fontsize=9, inline=1)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	plt.title(name + " contour")
	plt.savefig(path + name + "_contour.png")
