from cauchy import *
import numpy as np
import matplotlib.pyplot as plt


def graph_N_step(y0, t0, N, h, f, sol):
	X = np.arange(t0, t0 + (N+1)*h, h)
	if (sol != 0):
		Ry = sol(X)
		plt.plot(X, Ry, "k", label="Solution exacte")

	Y = meth_n_step(y0, t0, N, h, f, step_euler)
	plt.plot(X, Y, "b--", label="Euler")

	Y = meth_n_step(y0, t0, N, h, f, step_pt_milieu)
	plt.plot(X, Y, "g--", label="Point milieu")

	Y = meth_n_step(y0, t0, N, h, f, step_heun)
	plt.plot(X, Y, "c--", label="Heun")

	Y = meth_n_step(y0, t0, N, h, f, step_rk4)
	plt.plot(X, Y, "r--", label="Range-Kutta")

	plt.legend()
	plt.show()

def test_N_step():
	N = 50
	h = 0.5
	t0 = 0
	tf = 5
	y0 = 1
	f = lambda t, y: y / (1 + t*t)
	sol = lambda x: np.exp(np.arctan(x))
	graph_N_step(y0, t0, N, h, f, sol)

def extract2(Y):
	n = len(Y)
	y1 = np.empty(n)
	y2 = np.empty(n)

	for i in range(n) :
		y1[i] = Y[i][0]
		y2[i] = Y[i][1]
	return y1, y2

def graph_2d(Y0, t0, N, h, f, sol):
	X = np.arange(t0, t0 + (N+1)*h, h)
	if (sol != 0):
		sol_tab = [sol(t) for t in X]
		R1, R2 = extract2(sol_tab)
		plt.plot(X, R1, "k", label="resultat")
		plt.plot(X, R2, "k")


	y1, y2 = extract2(meth_n_step(Y0, t0, N, h, f, step_euler))
	plt.plot(X, y1, "b--", label="euler")
	plt.plot(X, y2, "b--")

	y1, y2 = extract2(meth_n_step(Y0, t0, N, h, f, step_pt_milieu))
	plt.plot(X, y1, "g--", label="point milieu")
	plt.plot(X, y2, "g--")

	y1, y2 = extract2(meth_n_step(Y0, t0, N, h, f, step_heun))
	plt.plot(X, y1, "c--", label="heun")
	plt.plot(X, y2, "c--")

	y1, y2 = extract2(meth_n_step(Y0, t0, N, h, f, step_rk4))
	plt.plot(X, y1, "r--", label="range kutta")
	plt.plot(X, y2, "r--")

	plt.legend()
	plt.show()

def graph_2d_pred(Y0, t0, N, h, f):
	X = np.arange(t0, t0 + (N+1)*h, h)

	y1, y2 = extract2(meth_n_step(Y0, t0, N, h, f, step_rk4))
	plt.plot(X, y1, label="Proies")
	plt.plot(X, y2, label="Predateurs")

	plt.legend()
	plt.show()


def test_2d():
	N = 50
	h = 0.1
	t0 = 0
	Y0 = np.array([1, 0])
	f = lambda t, Y: np.array([-Y[1], Y[0]])
	sol = lambda t: np.array([np.cos(t), np.sin(t)])
	graph_2d(Y0,t0,N,h,f,sol)


def graph_epsilon(y0, t0, tf, eps, f, sol):
    euler_eps = meth_epsilon(y0, t0, tf, eps, f, step_euler)
    pt_milieu_eps = meth_epsilon(y0, t0, tf, eps, f, step_pt_milieu)
    heun_eps = meth_epsilon(y0, t0, tf, eps, f, step_heun)
    rk4_eps = meth_epsilon(y0, t0, tf, eps, f, step_rk4)

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

def graph_epsilon_2d(y0, t0, tf, eps, f, sol):
    euler_eps1, euler_eps2 = extract2(meth_epsilon(y0, t0, tf, eps, f, step_euler))
    pt_milieu_eps1, pt_milieu_eps2 = extract2(meth_epsilon(y0, t0, tf, eps, f, step_pt_milieu))
    heun_eps1, heun_eps2 = extract2(meth_epsilon(y0, t0, tf, eps, f, step_heun))
    rk4_eps1, rk4_eps2 = extract2(meth_epsilon(y0, t0, tf, eps, f, step_rk4))

    x_e = np.linspace(t0, tf, len(euler_eps1))
    x_p = np.linspace(t0, tf, len(pt_milieu_eps1))
    x_h = np.linspace(t0, tf, len(heun_eps1))
    x_r = np.linspace(t0, tf, len(rk4_eps1))
    real1, real2 = extract2([sol(t) for t in x_e])
    plt.plot(x_e, euler_eps1, label="Euler")
    plt.plot(x_e, euler_eps2)
    plt.plot(x_p, pt_milieu_eps1, label="Point millieu")
    plt.plot(x_p, pt_milieu_eps2)
    plt.plot(x_h, heun_eps1, label="Heun")
    plt.plot(x_h, heun_eps2)
    plt.plot(x_r, rk4_eps1, label="Runge-Kutta")
    plt.plot(x_r, rk4_eps2)
    plt.plot(x_e, real1, "k", label="Solution exacte")
    plt.plot(x_e, real2, "k")
    plt.legend()
    plt.show()

def test_epsilon():
	y0 = np.array([1.0])
	f = lambda t, y : y[0]/(1+t**2)
	t0 = 0.0
	tf = 5.0
	eps = 1e-3
	sol = lambda t : np.exp(np.arctan(t))
	graph_epsilon(y0, t0, tf, eps, f, sol)

def test_epsilon_2d():
	y0 = np.array([1, 0])
	f2 = lambda t, y : np.array([-y[1], y[0]])
	t0 = 0
	tf = 5
	eps = 1e-1
	sol2 = lambda t : np.array([np.cos(t), np.sin(t)])
	graph_epsilon_2d(y0, t0, tf, eps, f2, sol2)

if __name__ == "__main__":
	test_N_step()
	test_2d()
	test_epsilon()
	test_epsilon_2d()
