from cauchy import *
import numpy as np
import matplotlib.pyplot as plt

N = 50
h = 0.1
t0 = 0
tf = 5
y0 = 1
f = lambda t, y: y / (1 + t*t)
sol = lambda x: np.exp(np.arctan(x))
graph_N_step(y0,t0,N,h,f,sol)



def extract2(Y):
	n = len(Y)
	y1 = np.empty(n)
	y2 = np.empty(n)

	for i in range(n) :
		y1[i] = Y[i][0]
		y2[i] = Y[i][1]
	return y1, y2

def test_2d():
	N = 50
	h = 0.1
	t0 = 0
	Y0 = np.array([1, 0])
	f = lambda t, Y: np.array([-Y[1], Y[0]])
	X = np.arange(t0, t0 + N*h, h)


	y1, y2 = extract2(meth_n_step(Y0, t0, N, h, f, step_euler))
	plt.plot(X, y1, "b", label="euler")
	plt.plot(X, y2, "b")

	y1, y2 = extract2(meth_n_step(Y0, t0, N, h, f, step_pt_milieu))
	plt.plot(X, y1, "g", label="point milieu")
	plt.plot(X, y2, "g")

	y1, y2 = extract2(meth_n_step(Y0, t0, N, h, f, step_heun))
	plt.plot(X, y1, "c", label="heun")
	plt.plot(X, y2, "c")

	y1, y2 = extract2(meth_n_step(Y0, t0, N, h, f, step_rk4))
	plt.plot(X, y1, "r", label="range kutta")
	plt.plot(X, y2, "r")

	R1 = np.cos(X)
	R2 = np.sin(X)
	plt.plot(X, R1, "k--", label="resultat")
	plt.plot(X, R2, "k--")

	plt.legend()
	plt.show()
test_2d()

def graph_epsilon(y0,t0,tf,eps,f,sol):
    euler_eps = meth_epsilon(y0,t0,tf,eps,f,step_euler)
    pt_milieu_eps = meth_epsilon(y0,t0,tf,eps,f,step_pt_milieu)
    heun_eps = meth_epsilon(y0,t0,tf,eps,f,step_heun)
    rk4_eps = meth_epsilon(y0,t0,tf,eps,f,step_rk4)

    x_e = np.linspace(t0, tf, len(euler_eps))
    x_p = np.linspace(t0, tf, len(pt_milieu_eps))
    x_h = np.linspace(t0, tf, len(heun_eps))
    x_r = np.linspace(t0, tf, len(rk4_eps))
    real = [sol(t) for t in x_e]
    plt.plot(x_e, euler_eps, label="Euler")
    plt.plot(x_p, pt_milieu_eps, label="Point millieu")
    plt.plot(x_h, heun_eps, label="Heun")
    plt.plot(x_r, rk4_eps, label="Runge-Kutta")
    plt.plot(x_e, real, "k--", label="Solution exacte")
    plt.legend()
    plt.show()

y0 = np.array([1])
f = lambda t,y : y[0]/(1+t**2)
t0 = 0
tf = 5
eps = 0.0000001
sol = lambda t : np.exp(np.arctan(t))
graph_epsilon(y0,t0,tf,eps,f,sol)

y0 = np.array([1, 0])
f2 = lambda t,y : np.array([-y[1], y[0]])
t0 = 0
tf = 5
eps = 0.0000001
sol2 = lambda t : np.array([cos(t), sin(t)])
graph_epsilon(y0,t0,tf,eps,f,sol)
