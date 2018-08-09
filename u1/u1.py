from helpers import plotContour
from helpers import plotSurface
from functions import rosenbrock
plotSurface(-3.5,3.5,0.1,-3.5,3.5,0.1,rosenbrock,"plots/","rosenbrock")
plotContour(-3.5,3.5,0.1,-3.5,3.5,0.1,rosenbrock,"plots/","rosenbrock")

from functions import himmelblau
plotSurface(-10,10,0.1,-10,10,0.1,himmelblau,"plots/","himmelblau")
plotContour(-10,10,0.1,-10,10,0.1,himmelblau,"plots/","himmelblau")
#
from functions import bazaraa_shetty
plotSurface(-10,10,0.1,-10,10,0.1,bazaraa_shetty,"plots/","bazaraa_shetty")
plotContour(-10,10,0.1,-10,10,0.1,bazaraa_shetty,"plots/","bazaraa_shetty")
#
from functions import beale
plotSurface(-10,10,0.1,-10,10,0.1,beale,"plots/","beale")
plotContour(-10,10,0.1,-10,10,0.1,beale,"plots/","beale")
#
from functions import spellucci
plotSurface(-100,100,0.1,-100,100,0.1,spellucci,"plots/","spellucci")
plotContour(-100,100,0.1,-100,100,0.1,spellucci,"plots/","spellucci")





#import numpy as np
#
#x_1 = np.array([0,1,2,3])
#x_2 = np.array([0,1,2,3])
#
#print(rosenbrock(x_1,x_2))
#print(himmelblau(x_1,x_2))
#print(bazaraa_shetty(x_1,x_2))
#print(beale(x_1,x_2))
#print(spellucci(x_1,x_2))