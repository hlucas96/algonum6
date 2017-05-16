import numpy as np


def step_euler(y, t, h, f):
	return y + h * f(t,y)

def step_pt_milieu(y, t, h, f):
	return y + h * f(t + h/2, y + h * f(t, y)/2)

def step_heun(y, t, h, f):
	return y + h*(f(t, y) + f(t + h, y + h*f(t, y)))/2

def step_rk4(y, t, h, f):
	k1 = f(t, y)
	k2 = f(t + h/2, y + h*k1/2)
	k3 = f(t + h/2, y + h*k2/2)
	k4 = f(t + h, y + h*k3)
	return y + h*(k1 + 2*k2 + 2*k3 + k4)/6

def meth_n_step(y0, t0, N, h, f, meth):
	Y = np.empty(N, dtype=object)
	Y[0] = y0
	t = t0
	
	for i in range(1, N):
		Y[i] = meth(Y[i - 1], t, h, f)
		t += h
		
	return Y

def meth_epsilon(y0,t0,tf,eps,f,meth):
	tab = [y0]
	t = t0
	y = y0
	N = 2
	h = (tf-t0)/N
	yN = meth_n_step(y0,t0,N,h,f,meth)
	yN2 = meth_n_step(y0,t0,N/2,2*h,f,meth)
	while(np.norm(yN - yN2) > eps):
		yN2 = yN
		N *= 2
		h = (tf-t0)/N
		yN = meth_n_step(y0,t0,N,h,f,meth)
	return yN
