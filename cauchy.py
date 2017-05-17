import numpy as np
import matplotlib.pyplot as plt


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
	Y = [y0]
	t = t0
	for i in range(1, N):
		Y.append(meth(Y[-1], t, h, f))
		t += h
	return Y

def meth_epsilon(y0,t0,tf,eps,f,meth):
    t = t0
    N = 2
    h = (tf-t0)/N
    yN = meth_n_step(y0,t0,N,h,f,meth)
    yN2 = meth_n_step(y0,t0,N/2,2*h,f,meth)
    while(np.linalg.norm(yN[-1] - yN2[-1]) > eps):
        yN2 = list(yN)
        N *= 2
        h = (tf-t0)/N
        yN = list(meth_n_step(y0,t0,N,h,f,meth))
    return yN


def graph_N_step(y0,t0,N,h,f,sol):
	X = np.arange(t0, t0 + N*h, h)
	Y = meth_n_step(y0, t0, N, h, f, step_euler)
	plt.plot(X, Y, "b", label="Euler")

	Y = meth_n_step(y0, t0, N, h, f, step_pt_milieu)
	plt.plot(X, Y, "g", label="Point milieu")

	Y = meth_n_step(y0, t0, N, h, f, step_heun)
	plt.plot(X, Y, "c", label="Heun")

	Y = meth_n_step(y0, t0, N, h, f, step_rk4)
	plt.plot(X, Y, "r", label="Range-Kutta")
	if (sol != 0):
		Ry = sol(X)
		plt.plot(X, Ry, "k--", label="Solution exacte")

	plt.legend()
	plt.show()
