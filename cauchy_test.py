from cauchy import *
import numpy as np
import matplotlib.pyplot as plt



## test of meth_n_step
#
def test_1d():

	N = 50
	h = 0.1
	t0 = 0
	y0 = 1
	f = lambda t, y: y / (1 + t*t)
	X = np.arange(t0, t0 + N*h, h)


	Y = meth_n_step(y0, t0, N, h, f, step_euler)
	plt.plot(X, Y, "b", label="euler")

	Y = meth_n_step(y0, t0, N, h, f, step_pt_milieu)
	plt.plot(X, Y, "g", label="point milieu")

	Y = meth_n_step(y0, t0, N, h, f, step_heun)
	plt.plot(X, Y, "c", label="heun")

	Y = meth_n_step(y0, t0, N, h, f, step_rk4)
	plt.plot(X, Y, "r", label="range kutta")

	expect = lambda x: np.exp(np.arctan(x))
	Ry = expect(X)
	plt.plot(X, Ry, "k--", label="resultat")

	plt.legend()
	plt.show()
test_1d()



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
	
