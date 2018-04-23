def rosenbrock(x):
	x_1,x_2 = x
	return(100 * (x_2 - x_1**2)**2 + (1-x_1)**2)

def himmelblau(x):
	x_1,x_2 = x
	return((x_1**2 + x_2-11)**2+(x_1 + x_2**2-7)**2)

def bazaraa_shetty(x):
	x_1,x_2 = x
	return((x_1-2))**4+(x_1 - 2*x_2)**2

def beale(x):
	x_1,x_2 = x
	return( (1.5 - x_1*(1-x_2))**2 + (2.25 - x_1*(1-x_2**2))**2 + (2.625-x_1*(1-x_2**3))**2)

def spellucci(x):
	x_1,x_2 = x
	return(2*x_1**3 + x_2**2 +x_1**2 *x_2**2 + 4* x_1*x_2 + 3)
