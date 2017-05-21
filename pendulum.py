import numpy as np
from cauchy import meth_n_step, step_rk4


# return the abscissia of the intersection point between
# the line ((x1, y1), (x2, y2)) and abscissia axis
def zero(x1, y1, x2, y2):
	a = (y1 - y2) / (x1 - x2)
	b = y1 - (a * x1)
	return -b / a


# return the periode of the function function
# discribe by X and Y
def period(X, Y):
	n = len(X)
	k = 1
        
	while ((k < n) and (Y[k - 1] * Y[k] > 0)) :
		k += 1
	if not(k < n) :
		print("too short: can't find a starting point")
		return 1

	# sig(Y[k - 1]) != sig(Y[k])
	t0 = zero(X[k - 1], Y[k - 1], X[k], Y[k])
	nb = 0
	
	for i in range(k + 1, n) :
		if (Y[i - 1] * Y[i] <= 0) :
			nb += 1
			k = i	
	if (nb == 0) :
		print("too short: not a complet periode")
		return 1

        # sig(Y[k - 1]) != sig(Y[k])
	tn = zero(X[k - 1], Y[k - 1], X[k], Y[k])
	T = (tn - t0) / nb # average time of a semi-periode
	return 2 * T

         
# return the frequencies of a simple pendulum
# with initial theta given by the array
def frequencies(Theta, g, l):
        t0 = 0
        h = 0.3
        N = 50 # probably it's enough
        
        x = np.arange(t0, t0 + N*h, h)
        y = np.empty(N)
        # theta'' + (g/l)sin(theta) = 0
        F = lambda t, Y: np.array([Y[1], -(g/l) * np.sin(Y[0])])

        n = len(Theta)
        freq = np.empty(n)
        for i in range(n):

                Y0 = np.array([Theta[i], 0])
                Y = meth_n_step(Y0, t0, N, h, F, step_rk4)
                for j in range(N) :
                        y[j] = Y[j][0]

                freq[i] = 1 / period(x, y)

        return freq


# Y = [th1, th1', th2, th2']
def step_double_pendulum(Y, g, l):
        gl = g / l
        cos_dth = np.cos(Y[0] - Y[2])
        sin_dth = np.sin(Y[0] - Y[2])
        sin_th1 = np.sin(Y[0])
        sin_th2 = np.sin(Y[2])
        alpha = 1 / (1 - ((cos_dth * cos_dth) / 2))
        
        Y1 = alpha * (Y[1] * Y[1] * sin_dth * cos_dth / 2
                      + gl * sin_th2 * cos_dth
                      - Y[3] * Y[3] * sin_dth / 2
                      - gl * sin_th1)
        Y3 = alpha * (Y[3] * Y[3] * sin_dth * cos_dth / 2
                      + gl * sin_th1 * cos_dth
                      + Y[1] * Y[1] * sin_dth
                      - gl * sin_th2)

        return np.array([Y[1], Y1, Y[3], Y3]) # [th1', th1'', th2', th2'']


def pendulum_path(th1, th2, g, l, N, h):
        Y0 = np.array([th1, 0, th2, 0])
        F = lambda t, Y: step_double_pendulum(Y, g, l)
        sol = meth_n_step(Y0, 0, N, h, F, step_rk4)

        n = len(sol)
        x2 = np.empty(n)
        for i in range(n): x2[i] = np.sin(sol[i][0]) + np.sin(sol[i][2])
        y2 = np.empty(n)
        for i in range(n): y2[i] = -np.cos(sol[i][0]) - np.cos(sol[i][2])

        return x2, y2


def flip_over_ratio(th1, th2, g, l):
        N = 100
        h = 0.5
        Y0 = np.array([th1, 0, th2, 0])
        F = lambda t, Y: step_double_pendulum(Y, g, l)
        sol = meth_n_step(Y0, 0, N, h, F, step_rk4)

        pi = 3.14
        # theta2 = sol[:][2]
        for i in range(1, N):
                if (sol[i - 1][2] <= pi):
                        if (sol[i][2] >= pi):
                                return i / N
                else:
                        if (sol[i][2] <= pi):
                                return i / N
        return 1
