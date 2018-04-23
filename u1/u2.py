from scipy.optimize import minimize
import numpy as np

from functions import rosenbrock
from functions import himmelblau
from functions import bazaraa_shetty
from functions import beale
from functions import spellucci

scipy_res = dict()

scipy_res['rosenbrock'] = minimize(rosenbrock,			np.array([0,0]),method='Nelder-Mead')
scipy_res['himmelblau'] = minimize(himmelblau,			np.array([0,0]),method='Nelder-Mead')
scipy_res['bazaraa_shetty'] = minimize(bazaraa_shetty,	np.array([0,0]),method='Nelder-Mead')
scipy_res['beale'] = minimize(beale,						np.array([0,0]),method='Nelder-Mead')
scipy_res['spellucci'] = minimize(spellucci,				np.array([0,0]),method='Nelder-Mead')

for i in scipy_res.keys():
	print(scipy_res[i]['x'])